# Copyright (c) Fudan University.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms
# of the GNU General Public License version 3.

# This code is based on https://github.com/facebookresearch/llama and
# https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama/chatllama
import json
import math
import random
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed
from torch import nn
import torch.nn.functional as F
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


@dataclass
class ModelArgs:
    """This class is a modification of the ModelArgs class implemented in
    the LLaMA repo. The class has been modified for training, since the
    original one just supports inference.
    """

    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    # defined later by tokenizer
    vocab_size: int = -1
    # make SwiGLU hidden layer size multiple of large power of 2
    multiple_of: int = 256
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 1024

    # added attributes
    froze_embeddings: bool = True
    tensor_parallel: bool = True


class RMSNorm(torch.nn.Module):
    """This class is a modification of the RMSNorm class implemented in
    the LLaMA repo. The class has been modified for training, since the
    original one just supports inference.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [
        d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
    ]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """This class is a modification of the Attention class implemented in
    the LLaMA repo. The class has been modified for training, since the
    original one just supports inference.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        if args.tensor_parallel:
            self.n_local_heads = (
                args.n_heads // fs_init.get_model_parallel_world_size()
            )
        else:
            self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        if args.tensor_parallel:
            self.wq = ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            self.wk = ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            self.wv = ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            self.wo = RowParallelLinear(
                args.n_heads * self.head_dim,
                args.dim,
                bias=False,
                input_is_parallel=True,
                init_method=lambda x: x,
            )
        else:
            self.wq = nn.Linear(
                args.dim, args.n_heads * self.head_dim, bias=False
            )
            self.wk = nn.Linear(
                args.dim, args.n_heads * self.head_dim, bias=False
            )
            self.wv = nn.Linear(
                args.dim, args.n_heads * self.head_dim, bias=False
            )
            self.wo = nn.Linear(
                args.n_heads * self.head_dim, args.dim, bias=False
            )

        self.dim_cache = (
            args.max_batch_size,
            args.max_seq_len,
            self.n_local_heads,
            self.head_dim,
        )
        self.cache_k = torch.zeros(self.dim_cache).cuda()

        self.cache_v = torch.zeros(self.dim_cache).cuda()

    def forward(
        self,
        x: torch.Tensor,
        kv_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int
    ) -> torch.Tensor:

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.training:
            keys = xk
            values = xv
        else:
            self.cache_k.to(xk.device)
            self.cache_v.to(xv.device)
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv 
            keys = self.cache_k[:bsz, : start_pos + seqlen]  
            values = self.cache_v[:bsz, : start_pos + seqlen]  

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )
        if kv_mask is not None:
            scores = scores + kv_mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """This class is a modification of the FeedForward class implemented in
    the LLaMA repo. The class has been modified for training, since the
    original one just supports inference.
    """

    def __init__(
        self, dim: int, hidden_dim: int, multiple_of: int, tensor_parallel: bool
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of
        )

        if tensor_parallel:
            self.w1 = ColumnParallelLinear(
                dim,
                hidden_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            self.w2 = RowParallelLinear(
                hidden_dim,
                dim,
                bias=False,
                input_is_parallel=True,
                init_method=lambda x: x,
            )
            self.w3 = ColumnParallelLinear(
                dim,
                hidden_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
        else:
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """This class is a modification of the TransformerBlock class
    implemented in the LLaMA repo. The class has been modified for training,
    since the original one just supports inference.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            tensor_parallel=args.tensor_parallel,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.tensor_parallel = args.tensor_parallel

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int
    ) -> torch.Tensor:
        # modified from orignal code to enable external cache
        attention_mask = attention_mask[:, None, :, :]
        if self.tensor_parallel:
            attention_mask = attention_mask.expand(
                -1,
                self.n_heads // fs_init.get_model_parallel_world_size(),
                -1,
                -1,
            )
        else:
            attention_mask = attention_mask.expand(-1, self.n_heads, -1, -1)
        attn = self.attention.forward(
            self.attention_norm(x), attention_mask, freqs_cis, start_pos
        )
        h = x + attn
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """This class is a modification of the Transformer class implemented in
    the LLaMA repo. The class has been modified for training, since the
    original one just supports inference. The generate method was inspired by
    the generate function you can find in `llama.generation`.
    """

    def __init__(self, params: ModelArgs):
        super().__init__()

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        if params.tensor_parallel:
            self.n_local_heads = (
                params.n_heads // fs_init.get_model_parallel_world_size()
            )
        else:
            self.n_local_heads = params.n_heads

        self.head_dim = params.dim // params.n_heads

        if params.tensor_parallel:
            self.tok_embeddings = ParallelEmbedding(
                params.vocab_size, params.dim, init_method=lambda x: x
            )
        else:
            self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        if params.froze_embeddings:
            for param in self.tok_embeddings.parameters():
                param.requires_grad = False

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        if params.tensor_parallel:
            self.output = ColumnParallelLinear(
                params.dim,
                params.vocab_size,
                bias=False,
                init_method=lambda x: x,
            )
        else:
            self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.gradient_checkpoint = True

    def forward(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        attention_mask = attention_mask.detach()
        logits = self._forward(tokens, attention_mask, 0)
        return logits

    def _forward(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor, start_pos: int
    ) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        # mask has size (bsz, seqlen). It should be transformed in
        # (bsz, seqlen, seqlen)
        # if the mask is a boolean tensor, convert it to int
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.long()
        kv_mask = attention_mask[:, None, :].expand(_bsz, seqlen, seqlen)
        kv_mask = torch.tril(kv_mask, diagonal=0)
        kv_mask = 1 - kv_mask
        kv_mask = (
            torch.where(
                kv_mask == 1, kv_mask.new_tensor(-9223372036854775808), kv_mask
            )
            .detach()
            .long()
        )
        def create_custom_forward(module):
            def custom_forward(*inputs):
                # None for past_key_value
                return module(*inputs)

            return custom_forward

        for i, layer in enumerate(self.layers):
            if not self.training or not self.gradient_checkpoint:
                h = layer(
                    h, kv_mask, freqs_cis, start_pos
                )
            else:
                if self.gradient_checkpoint:
                    h = torch.utils.checkpoint.checkpoint(create_custom_forward(layer), h, kv_mask, freqs_cis, start_pos)

        h = self.norm(h)
        output = self.output(h)
        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        no_repeat_ngram_size=None,
    ):
        generated_tokens = []
        pre_pos = 0
        start_pos = input_ids.shape[1] # length of prompt

        for cur_pos in range(start_pos, start_pos+max_new_tokens):
            logits = self._forward(input_ids[:,pre_pos:cur_pos], attention_mask[:,pre_pos:cur_pos], pre_pos)[:,-1,:]
            if temperature > 0:
                probs = torch.softmax(logits.float() / temperature, dim=-1).type_as(logits)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token).unsqueeze(1)],
                dim=1,
            )
            generated_tokens.append(next_token)
            pre_pos = cur_pos

        sequences = torch.concat(
            (input_ids, torch.stack(generated_tokens, dim=1)), dim=1
        )
        return sequences


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    print("local_rank:", local_rank, "world_size:", world_size)

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    # torch.manual_seed(1) we set them outside this function
    return local_rank, world_size

def load_checkpoints(
    ckpt_dir: str, local_rank: int, world_size: int
) -> Tuple[dict, dict]:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(checkpoints), (
        f"Loading a checkpoint for MP={len(checkpoints)} but world "
        f"size is {world_size}"
    )
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    return checkpoint, params


def load_model(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    froze_embeddings: bool,
    zero: bool,
    tensor_parallel: bool,
    pipeline_parallel: bool,
    max_batch_size: int = 32,
    max_seq_len: int = 1024,
) -> Tuple[Transformer, HFLikeTokenizer]:
    assert zero + tensor_parallel + pipeline_parallel <= 1, \
        "ZeRO, Tensor Parallel and Pipeline Parallel are mutually exclusive now"
    checkpoint, params = load_checkpoints(ckpt_dir, local_rank, world_size)
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    model_args.froze_embeddings = froze_embeddings
    model_args.tensor_parallel = tensor_parallel
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    #torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    tokenizer = HFLikeTokenizer(tokenizer)

    return model, tokenizer
