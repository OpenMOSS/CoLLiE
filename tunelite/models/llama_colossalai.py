# Copyright (c) Fudan University.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms
# of the GNU General Public License version 3.

# This code based on https://github.com/facebookresearch/llama
import torch
from torch import nn
import torch.nn.functional as F

from sentencepiece import SentencePieceProcessor

import io
import os
import time
import math
import tqdm
import shutil
from io import BytesIO
from einops import rearrange
from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Callable, List, Union, Dict

try:
    import colossalai
    import colossalai.nn as col_nn
    from colossalai import kernel as K
    from colossalai.amp import AMP_TYPE
    from colossalai.core import global_context as gpc
    from colossalai.pipeline.utils import partition_uniform
    from colossalai.context.parallel_mode import ParallelMode
    from colossalai.utils.activation_checkpoint import checkpoint
    from colossalai.nn.layer.wrapper import PipelineSharedModuleWrapper
    from colossalai.utils.model.colo_init_context import ColoInitContext
    from colossalai.logging import get_dist_logger, disable_existing_loggers
    from colossalai.kernel.cuda_native.flash_attention import flash_attention_qkv
    
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Detected Colossal-AI is not installed. See https://github.com/hpcaitech/ColossalAI")

try:
    from apex.fused_dense import FusedDense as ApexFusedDense
    from apex.normalization.fused_layer_norm import FusedRMSNorm
except ModuleNotFoundError:
    ApexFusedDense = None
    FusedRMSNorm = None

try:
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.flash_attention import FlashAttention
    from flash_attn.ops.fused_dense import FusedDense as FlashAttnFusedDense
except ModuleNotFoundError:
    FlashAttention = None
    RotaryEmbedding = None
    FlashAttnFusedDense = None

try:
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import LowerTriangularMask
except ModuleNotFoundError:
    memory_efficient_attention = None
    LowerTriangularMask = None

class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        if self.eos_id in t:
            t = t[:t.index(self.eos_id)+1]
        return self.sp_model.decode(t)


class HFLikeTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

        # assign attributes from real tokenizer to masked one
        self.pad_id = self.tokenizer.pad_id
        self.eos_id = self.tokenizer.eos_id
        self.bos_id = self.tokenizer.bos_id

        # mask attribute to be similar to hugging face
        self.eos_token_id = self.tokenizer.eos_id
        self.pad_token_id = self.tokenizer.pad_id

        # to match hugging face attribute
        self.pad_token_id = self.pad_id

    def create_sequence_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = torch.where(
            tokens == self.tokenizer.pad_id,
            torch.zeros_like(tokens),
            torch.ones_like(tokens),
        )
        mask = torch.where(
            tokens == self.tokenizer.bos_id, torch.zeros_like(tokens), mask
        )
        mask = torch.where(
            tokens == self.tokenizer.eos_id, torch.zeros_like(tokens), mask
        )
        return mask

    def __call__(self, texts: Union[List[str], str], *args, **kwargs):
        if isinstance(texts, str):
            text = self.tokenizer.encode(texts, kwargs.get("bos", True), eos=kwargs.get("eos", True))
            tokens = torch.tensor(text).long()
            mask = torch.ones_like(tokens)
        else:
            texts = [
                self.tokenizer.encode(text, kwargs.get("bos", True), eos=kwargs.get("eos", True))
                for text in texts
            ]
            max_len = max(len(text) for text in texts)
            tokens = torch.full(
                (len(texts), max_len), self.tokenizer.pad_id
            ).long()
            for i, text in enumerate(texts):
                tokens[i, -len(text) :] = torch.tensor(  # noqa E203
                    text
                ).long()

            # TODO: decide how eos and bos should be handled - i need to mask
            # them? or not?
            mask = self.create_sequence_mask(tokens)
            for i in range(tokens.shape[0]):
                current_tokens = tokens[i, mask[i] == 1]
                tokens[
                    i, -len(current_tokens) - 1 : -1  # noqa E203
                ] = current_tokens
            mask = self.create_sequence_mask(tokens)

            # convert `pad_id` from -1 to 0, otherwise embedding will cause out
            # of bounds.
            tokens = torch.where(
                tokens == self.tokenizer.pad_id,
                torch.zeros_like(tokens),
                tokens,
            )
        output = {
            "input_ids": tokens,
            "attention_mask": mask,
        }
        return output

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

@dataclass
class ModelArgs:
    # model parameters
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    layer_norm_epsilon: float = 1e-5
    # implementation parameters
    dense: str = "raw"  # raw, fused, apex
    rms_norm: str = "raw"  # raw, apex
    attention: str = "raw"  # raw, flash, col_flash, mem_eff
    rotary_emb: str = "raw"  # raw, fused
    # parallel parameters
    pp_size: int = 8
    tp_size: int = 1
    tp_type: str = "1d" # 1d, 2d, 2.5d, 3d
    dp_size: int = 1
    micro_batch_size: int = 1
    # other parameters
    checkpoint: bool = False
    dropout: float = 0.1
    fp16: bool = True
    backend: str = "nccl"


class RMSNorm(nn.Module):
    def __init__(self, model_args: ModelArgs = ModelArgs()) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(model_args.hidden_size))
        self.variance_epsilon = model_args.layer_norm_epsilon

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, model_args: ModelArgs = ModelArgs()) -> None:
        super().__init__()
        self.model_args = model_args
        head_dim = self.model_args.hidden_size // self.model_args.num_attention_heads
        if self.model_args.rotary_emb == "raw":
            freqs = 1.0 / (10000.0 ** (
                torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
            t = torch.arange(1024
                             * 2, device=freqs.device)
            freqs = torch.outer(t, freqs).float()
            self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(
                torch.device(f"cuda:{gpc.get_local_rank(ParallelMode.PIPELINE)}"))
        elif self.model_args.rotary_emb == "fused":
            assert RotaryEmbedding is not None, \
                "Detected rotary_emb is not installed. See https://github.com/HazyResearch/flash-attention/tree/main/csrc/rotary"
            object.__setattr__(self, "rpoe", RotaryEmbedding(dim=head_dim))

    def forward(self,
                query: Optional[torch.Tensor] = None,
                key: Optional[torch.Tensor] = None,
                start_pos: int = 0,
                seq_len: int = 1024):
        if self.model_args.rotary_emb == "raw":
            t = query.dtype
            query = torch.view_as_complex(
                query.float().reshape(*query.shape[:-1], -1, 2))
            key = torch.view_as_complex(
                key.float().reshape(*key.shape[:-1], -1, 2))
            freqs_cis = self.freqs_cis[start_pos: start_pos + seq_len]
            shape = [d if i == 1 or i == query.ndim -
                     1 else 1 for i, d in enumerate(query.shape)]
            freqs_cis = freqs_cis.view(*shape)
            query = torch.view_as_real(query * freqs_cis).flatten(3)
            key = torch.view_as_real(key * freqs_cis).flatten(3)
            return query.type(t), key.type(t)
        elif self.model_args.rotary_emb == "fused":
            qkv = torch.stack([query, key, key], dim=2)
            output = object.__getattribute__(self, "rpoe")(qkv=qkv, seqlen_offset=start_pos)
            return output[:, :, 0, ...], output[:, :, 1, ...]


class TransformerBlock(nn.Module):
    def __init__(self, model_args: ModelArgs = ModelArgs()) -> None:
        super(TransformerBlock, self).__init__()
        self.model_args = model_args
        self.attention = nn.ModuleDict()
        self.mlp = nn.ModuleDict()
        self._construct()

    def _construct(self):
        if self.model_args.dense == "raw":
            self.attention["wqkv"] = col_nn.Linear(self.model_args.hidden_size,
                                                   self.model_args.hidden_size * 3,
                                                   bias=False)
            self.attention["wo"] = col_nn.Linear(self.model_args.hidden_size,
                                                 self.model_args.hidden_size,
                                                 bias=False)
            self.mlp["w1"] = col_nn.Linear(self.model_args.hidden_size,
                                           self.model_args.intermediate_size,
                                           bias=False)
            self.mlp["w2"] = col_nn.Linear(self.model_args.intermediate_size,
                                           self.model_args.hidden_size,
                                           bias=False)
            self.mlp["w3"] = col_nn.Linear(self.model_args.hidden_size,
                                           self.model_args.intermediate_size,
                                           bias=False)
        elif self.model_args.dense == "fused":
            assert FlashAttnFusedDense is not None, \
                "Detected fused_dense_lib is not installed. See https://github.com/HazyResearch/flash-attention/tree/main/csrc/fused_dense_lib"
            self.attention["wqkv"] = FlashAttnFusedDense(self.model_args.hidden_size,
                                                         self.model_args.hidden_size * 3,
                                                         bias=False)
            self.attention["wo"] = FlashAttnFusedDense(self.model_args.hidden_size,
                                                       self.model_args.hidden_size,
                                                       bias=False)
            self.mlp["w1"] = FlashAttnFusedDense(self.model_args.hidden_size,
                                                 self.model_args.intermediate_size,
                                                 bias=False)
            self.mlp["w2"] = FlashAttnFusedDense(self.model_args.intermediate_size,
                                                 self.model_args.hidden_size,
                                                 bias=False)
            self.mlp["w3"] = FlashAttnFusedDense(self.model_args.hidden_size,
                                                 self.model_args.intermediate_size,
                                                 bias=False)
        elif self.model_args.dense == "apex":
            assert ApexFusedDense is not None, \
                "Detected apex is not installed. See https://github.com/NVIDIA/apex"
            self.attention["wqkv"] = ApexFusedDense(self.model_args.hidden_size,
                                                    self.model_args.hidden_size * 3,
                                                    bias=False)
            self.attention["wo"] = ApexFusedDense(self.model_args.hidden_size,
                                                  self.model_args.hidden_size,
                                                  bias=False)
            self.mlp["w1"] = ApexFusedDense(self.model_args.hidden_size,
                                            self.model_args.intermediate_size,
                                            bias=False)
            self.mlp["w2"] = ApexFusedDense(self.model_args.intermediate_size,
                                            self.model_args.hidden_size,
                                            bias=False)
            self.mlp["w3"] = ApexFusedDense(self.model_args.hidden_size,
                                            self.model_args.intermediate_size,
                                            bias=False)
        if self.model_args.rms_norm == "raw":
            self.attention["norm"] = RMSNorm(self.model_args)
            self.mlp["norm"] = RMSNorm(self.model_args)
        elif self.model_args.rms_norm == "apex":
            self.attention["norm"] = FusedRMSNorm(
                normalized_shape=self.model_args.hidden_size,
                eps=self.model_args.layer_norm_epsilon)
            self.mlp["norm"] = FusedRMSNorm(
                normalized_shape=self.model_args.hidden_size,
                eps=self.model_args.layer_norm_epsilon)

        self.attention["dropout"] = col_nn.Dropout(
            self.model_args.dropout)
        self.mlp["dropout"] = col_nn.Dropout(
            self.model_args.dropout)

        if self.model_args.attention == "col_flash":
            assert flash_attention_qkv is not None, \
                "Detected triton is not installed. See https://github.com/openai/triton"
            def attention(**kwargs):
                kwargs["qkv"] = rearrange(kwargs["qkv"], "b n three h d -> (b n) three h d")
                output = flash_attention_qkv(**kwargs)
                output = rearrange(
                    output, "(b n) h d -> b n (h d)", n=kwargs.get("seq_len"))
                output = F.dropout(
                    output, p=self.model_args.dropout, training=self.training)
                return output
            object.__setattr__(self, "attention_fn", attention)
        elif self.model_args.attention == "flash":
            assert FlashAttention is not None, \
                "Detected flash_attn is not installed. See https://github.com/HazyResearch/flash-attention"
            def attention(**kwargs):
                output, _ = FlashAttention()(
                    kwargs["qkv"], causal=kwargs.get("causal", True))
                output = rearrange(
                    output, "b n h d -> b n (h d)", n=kwargs.get("seq_len"))
                output = F.dropout(
                    output, p=self.model_args.dropout, training=self.training)
                return output
            object.__setattr__(self, "attention_fn", attention)
        elif self.model_args.attention == "mem_eff":
            assert memory_efficient_attention is not None and LowerTriangularMask is not None, \
                "Detected xformers is not installed. See https://github.com/facebookresearch/xformers"
            def attention(**kwargs):
                query, key, value = torch.split(
                    kwargs["qkv"], split_size_or_sections=1, dim=2)
                query, key, value = query.squeeze(
                    2), key.squeeze(2), value.squeeze(2)
                batch_size, seq_len, head_num, head_dim = query.shape
                mask = None
                if kwargs.get("causal", True) and seq_len > 1:
                    # bigger_seq_len = (seq_len // 8 + 1) * 8
                    # mask = torch.full((batch_size, head_num, 
                    #                    bigger_seq_len, 
                    #                    bigger_seq_len), float("-inf"))
                    # mask = torch.triu(mask, diagonal=1).to(query.dtype).to(query.device)
                    # mask = mask[:, :, :seq_len, :seq_len]
                    mask = LowerTriangularMask()
                output = memory_efficient_attention(query=query,
                                                  key=key,
                                                  value=value,
                                                  attn_bias=mask,
                                                  p=self.model_args.dropout,
                                                  scale=1/math.sqrt(head_dim))
                output = rearrange(output, "b n h d -> b n (h d)")
                return output
            object.__setattr__(self, "attention_fn", attention)
        elif self.model_args.attention == "raw":
            def attention(**kwargs):
                query, key, value = torch.split(
                    kwargs["qkv"], split_size_or_sections=1, dim=2)
                query, key, value = query.squeeze(
                    2), key.squeeze(2), value.squeeze(2)
                batch_size, seq_len, head_num, head_dim = key.shape
                query, key, value = query.permute(0, 2, 1, 3), key.permute(
                    0, 2, 1, 3), value.permute(0, 2, 1, 3)
                attention_score = torch.matmul(query, key.transpose(
                    2, 3)) / math.sqrt(head_dim)
                if kwargs.get("causal", True) and seq_len > 1:
                    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
                    mask = torch.triu(mask, diagonal=1).to(
                        attention_score.device)
                    attention_score = attention_score + mask
                attention_score = F.softmax(
                    attention_score, dim=-1).type_as(value)
                output = torch.matmul(attention_score, value)
                output = output.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, head_dim * head_num)
                output = F.dropout(
                    output, p=self.model_args.dropout, training=self.training)
                return output
            object.__setattr__(self, "attention_fn", attention)

    def forward(self,
                hidden_states: Optional[torch.Tensor],
                past_key_value: Optional[torch.Tensor] = None,
                causal: bool = True,
                rpoe: Callable = None):
        
        assert hidden_states.ndim == 3, f"hidden_states.shape must be (B, N, H), but got {hidden_states.shape}"
        batch_size, seq_len, hidden_size = hidden_states.shape
        head_dim = self.model_args.hidden_size // self.model_args.num_attention_heads
        _hidden_states = self.attention["norm"](hidden_states)
        qkv = self.attention["wqkv"](_hidden_states)
        qkv = rearrange(qkv,
                        "b n (three h d) -> b n three h d",
                        h=self.model_args.num_attention_heads,
                        three=3)
        query, key, value = torch.split(
            qkv, split_size_or_sections=1, dim=2)
        query, key, value = query.squeeze(
            2), key.squeeze(2), value.squeeze(2)
        if past_key_value is None:
            start_pos = 0
        else:
            assert past_key_value.ndim == 5, "past_key_value.shape must be (B, N, 2, H, D)"
            start_pos = past_key_value.shape[1]
            past_key, past_value = torch.split(
                past_key_value, split_size_or_sections=1, dim=2)
            past_key, past_value = past_key.squeeze(
                2), past_value.squeeze(2)
            key, value = torch.cat([past_key, key], dim=-3), torch.cat(
                [past_value, value], dim=-3)
            query = torch.cat([torch.zeros_like(past_value), query], dim=-3)
            past_key_value = torch.stack([past_key, past_value], dim=2)
        query, key = rpoe(query=query, key=key,
                          start_pos=int(start_pos), seq_len=seq_len)
        qkv = torch.stack([query, key, value], dim=2)
        hidden_states = hidden_states + self.attention["wo"](
            self.attention_fn(qkv=qkv,
                              sm_scale=1 / math.sqrt(head_dim),
                              batch_size=batch_size,
                              seq_len=seq_len,
                              dropout_p=self.model_args.dropout,
                              causal=causal)
        )
        if past_key_value is not None:
            hidden_states = hidden_states[:, start_pos:, ...]
        _hidden_states = self.mlp["norm"](hidden_states)
        hidden_states = hidden_states + self.mlp["dropout"](
            self.mlp["w2"](F.silu(self.mlp["w1"](_hidden_states)) * self.mlp["w3"](_hidden_states)))
        if past_key_value is not None:
            return hidden_states, past_key_value
        else:
            return hidden_states


class Transformer(nn.Module):
    def __init__(self,
                 is_start: bool = False,
                 is_end: bool = False,
                 num_blocks: int = 0,
                 model_args: ModelArgs = ModelArgs()):
        super(Transformer, self).__init__()
        self.model_args = model_args
        self.is_start = is_start
        self.is_end = is_end
        if self.is_start:
            self.token_embedding = col_nn.Embedding(
                self.model_args.vocab_size,
                embedding_dim=self.model_args.hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.model_args) for _ in range(num_blocks)])
        if self.is_end:
            if self.model_args.rms_norm == "raw":
                self.norm = RMSNorm(model_args)
            if self.model_args.rms_norm == "apex":
                self.norm = FusedRMSNorm(
                    normalized_shape=self.model_args.hidden_size,
                    eps=self.model_args.layer_norm_epsilon)
            if self.model_args.rms_norm == "raw":
                self.language_model_head = col_nn.Linear(self.model_args.hidden_size,
                                                         self.model_args.vocab_size,
                                                         bias=False)
            elif self.model_args.rms_norm == "fused":
                assert FlashAttnFusedDense is not None, \
                    "Detected fused_dense_lib is not installed. See https://github.com/HazyResearch/flash-attention/tree/main/csrc/fused_dense_lib"
                self.language_model_head = FlashAttnFusedDense(self.model_args.hidden_size,
                                                               self.model_args.vocab_size,
                                                               bias=False)
            elif self.model_args.rms_norm == "apex":
                assert ApexFusedDense is not None, \
                    "Detected apex is not installed. See https://github.com/NVIDIA/apex"
                self.language_model_head = ApexFusedDense(self.model_args.hidden_size,
                                                          self.model_args.vocab_size,
                                                          bias=False)
        self.rope = RotaryPositionEmbedding(self.model_args)

    def forward(self,
                hidden_states: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[torch.Tensor] = None,
                **kwargs):
        if self.is_start:
            assert input_ids is not None, "`input_ids` is not allowed to be None in the first pipeline node. "
            hidden_states = self.token_embedding(input_ids)
        if past_key_value is not None:
            if 0 in past_key_value[0].shape:
                # past_key_value shape is (B, layers, N, 2, H, D)
                past_key_value.resize_(input_ids.shape[0], 
                                       len(self.blocks), 
                                       0,
                                       2, 
                                       self.model_args.num_attention_heads, 
                                       self.model_args.hidden_size // self.model_args.num_attention_heads)
        for i in range(len(self.blocks)):
            if self.model_args.checkpoint and self.training:
                hidden_states = checkpoint(self.blocks[i], True,
                                           hidden_states, 
                                           past_key_value[:, i, ...] if past_key_value is not None else past_key_value, 
                                           True, 
                                           self.rope)
            else:
                hidden_states = self.blocks[i](
                    hidden_states, 
                    past_key_value[:, i, ...] if past_key_value is not None else past_key_value, 
                    True, 
                    self.rope)
            if past_key_value is not None:
                past_key_value[:, i, ...] = hidden_states[1]
                hidden_states = hidden_states[0]
        if self.is_end:
            hidden_states = self.norm(hidden_states)
            hidden_states = self.language_model_head(hidden_states)
        
        return hidden_states
    
def prepare_distribution(model_args: ModelArgs = ModelArgs()) -> dict:
    CONFIG = dict(NUM_MICRO_BATCHES=model_args.micro_batch_size, 
                  parallel=dict(
                      pipeline=int(model_args.pp_size), 
                      tensor=dict(size=model_args.tp_size, mode=model_args.tp_type)
                      )
                  )
    if model_args.fp16:
        CONFIG["fp16"] = dict(mode=AMP_TYPE.NAIVE)
    colossalai.launch_from_torch(config=CONFIG, backend=model_args.backend)
    if "pipeline" in CONFIG["parallel"] and CONFIG["parallel"]["pipeline"] == 1:
        gpc.is_pipeline_first_stage = lambda: True
        gpc.is_pipeline_last_stage = lambda: True
        gpc._local_ranks[ParallelMode.PIPELINE] = 0
        gpc._world_sizes[ParallelMode.PIPELINE] = 1

def build_pipe(model_args: ModelArgs = ModelArgs()):
    prepare_distribution(model_args=model_args)
    disable_existing_loggers()
    logger = get_dist_logger()
    with ColoInitContext(device=torch.device(f"cuda:{gpc.get_local_rank(ParallelMode.PIPELINE)}")):
        if model_args.pp_size > 1:
            wrapper = PipelineSharedModuleWrapper(
                [0, model_args.pp_size - 1])
        parts = partition_uniform(
            model_args.num_hidden_layers, model_args.pp_size, num_chunks=1)[gpc.get_local_rank(ParallelMode.PIPELINE)]
        chunk_list = []
        for start, end in parts:
            logger.info(
                f'Rank{gpc.get_local_rank(ParallelMode.PIPELINE)} build layer {start}-{end}, {end - start}/{model_args.num_hidden_layers} layers')
            chunk = Transformer(is_start=gpc.is_pipeline_first_stage(), 
                                is_end=gpc.is_pipeline_last_stage(), 
                                num_blocks=end - start, 
                                model_args=model_args)
            if gpc.is_pipeline_first_stage() and gpc.get_world_size(ParallelMode.PIPELINE) > 1:
                wrapper.register_module(chunk.token_embedding)
            if gpc.is_pipeline_last_stage() and gpc.get_world_size(ParallelMode.PIPELINE) > 1:
                wrapper.register_module(chunk.language_model_head)
            chunk_list.append(chunk)
        if len(chunk_list) == 1:
            return chunk_list[0]
        else:
            return nn.ModuleList(chunk_list)


def load_state_dict(protocol: str="s3", 
                    source: str = "hf", 
                    file_folder: str = "/remote-home/share/llama/7B",
                    s3_folder: str = "hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf",
                    model_args: ModelArgs = ModelArgs()) -> Dict[str, torch.tensor]:
    assert source in ["hf", "raw", "tunelite"], "source must be hf or raw or tunelite"
    assert protocol in ["s3", "file"], "protocol must be one of s3, file"
    state_dict = OrderedDict()
    part_state_dict = OrderedDict()
    tempdir = [""]
    if gpc.get_global_rank() == 0:
        if protocol == "s3":
            from petrel_client.client import Client
            client = Client()
            if not s3_folder.endswith("/"):
                s3_folder = f"{s3_folder}/"
            if source == "raw" or source == "tunelite":
                weights = [weight for weight in client.list(s3_folder) if weight.endswith(".pth")]
            elif source == "hf":
                weights = [weight for weight in client.list(s3_folder) if weight.endswith(".bin")]
            with tqdm.tqdm(desc=f"Loading state dict", total=len(weights)) as pbar:
                for content in weights:
                    buffer = BytesIO()
                    buffer.write(client.get(f"{s3_folder}{content}"))
                    buffer.seek(0)
                    raw_state_dict = torch.load(buffer, map_location="cpu")
                    for key, value in raw_state_dict.items():
                        if source == "hf":
                            if key.endswith("q_proj.weight") or key.endswith("k_proj.weight"):
                                raw_state_dict[key] = rearrange(
                                    value, 
                                    "(h two t) d -> h two t d", 
                                    h=model_args.num_attention_heads, 
                                    two=2).transpose(1, 2).reshape(
                                        model_args.hidden_size, 
                                        model_args.hidden_size)
                            state_dict.update(raw_state_dict)
                        elif source == "raw":
                            if key in state_dict.keys():
                                if key.endswith("wo.weight") or key.endswith("w2.weight") or key.endswith("embeddings.weight"):
                                    state_dict[key] = torch.cat((state_dict[key], value), dim=1)
                                elif key.endswith("norm.weight"):
                                    pass
                                else:
                                    state_dict[key] = torch.cat((state_dict[key], value), dim=0)
                            else:
                                state_dict[key] = value
                            state_dict.update(raw_state_dict)
                        elif source == "tunelite":
                            state_dict.update(raw_state_dict)
                    buffer.close()
                    pbar.update(1)
        elif protocol == "file":
            if not file_folder.endswith("/"):
                file_folder = f"{file_folder}/"
            if source == "raw" or source == "tunelite":
                weights = [weight for weight in list(os.listdir(file_folder)) if weight.endswith(".pth")]
            elif source == "hf":
                weights = [weight for weight in list(os.listdir(file_folder)) if weight.endswith(".bin")]
            weights.sort(key=lambda s: int(s[-6:-4]))
            state_dict = OrderedDict()
            with tqdm.tqdm(desc=f"Loading state dict", total=len(weights)) as pbar:
                for weight in weights:
                    raw_state_dict = torch.load(os.path.join(file_folder, weight), map_location="cpu")
                    for key, value in raw_state_dict.items():
                        if source == "hf":
                            if key.endswith("q_proj.weight") or key.endswith("k_proj.weight"):
                                raw_state_dict[key] = rearrange(
                                    value, 
                                    "(h two t) d -> h two t d", 
                                    h=model_args.num_attention_heads, 
                                    two=2).transpose(1, 2).reshape(
                                        model_args.hidden_size, 
                                        model_args.hidden_size)
                                state_dict.update(raw_state_dict)
                        elif source == "raw":
                            if key in state_dict.keys():
                                if key.endswith("wo.weight") or key.endswith("w2.weight") or key.endswith("embeddings.weight"):
                                    state_dict[key] = torch.cat((state_dict[key], value), dim=1)
                                elif key.endswith("norm.weight"):
                                    pass
                                else:
                                    state_dict[key] = torch.cat((state_dict[key], value), dim=0)
                            else:
                                state_dict[key] = value
                            state_dict.update(raw_state_dict)
                        elif source == "tunelite":
                            state_dict.update(raw_state_dict)
                    pbar.update(1)
        parts = partition_uniform(model_args.num_hidden_layers, model_args.pp_size, num_chunks=1)
        tempdir[0] = f"/dev/shm/TuneLite-{round(time.time() * 1000)}/"
        os.makedirs(tempdir[0])
        for pp_rank, [(start, end)] in enumerate(parts):
            part_state_dict = OrderedDict()
            if source == "hf":
                if start == 0:
                    part_state_dict["token_embedding.weight"] = state_dict["model.embed_tokens.weight"]
                if end == model_args.num_hidden_layers:
                    part_state_dict["language_model_head.weight"] = state_dict["lm_head.weight"]
                    part_state_dict["norm.weight"] = state_dict["model.norm.weight"]
            elif source == "raw":
                if start == 0:
                    part_state_dict["token_embedding.weight"] = state_dict["tok_embeddings.weight"]
                if end == model_args.num_hidden_layers:
                    part_state_dict["language_model_head.weight"] = state_dict["output.weight"]
                    part_state_dict["norm.weight"] = state_dict["norm.weight"]
            for idx, key in enumerate(list(range(start, end))):
                if source == "hf":
                    part_state_dict[f"blocks.{idx}.attention.wo.weight"] = state_dict[f"model.layers.{key}.self_attn.o_proj.weight"]
                    part_state_dict[f"blocks.{idx}.attention.wqkv.weight"] = torch.cat(
                        (
                            state_dict[f"model.layers.{key}.self_attn.q_proj.weight"],
                            state_dict[f"model.layers.{key}.self_attn.k_proj.weight"],
                            state_dict[f"model.layers.{key}.self_attn.v_proj.weight"]
                        ), dim=0
                    )
                    part_state_dict[f"blocks.{idx}.attention.norm.weight"] = state_dict[f"model.layers.{key}.input_layernorm.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.w1.weight"] = state_dict[f"model.layers.{key}.mlp.gate_proj.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.w2.weight"] = state_dict[f"model.layers.{key}.mlp.down_proj.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.w3.weight"] = state_dict[f"model.layers.{key}.mlp.up_proj.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.norm.weight"] = state_dict[f"model.layers.{key}.post_attention_layernorm.weight"]
                elif source == "raw":
                    part_state_dict[f"blocks.{idx}.attention.wo.weight"] = state_dict[f"layers.{key}.attention.wo.weight"]
                    part_state_dict[f"blocks.{idx}.attention.wqkv.weight"] = torch.cat(
                        (
                            state_dict[f"layers.{key}.attention.wq.weight"],
                            state_dict[f"layers.{key}.attention.wk.weight"],
                            state_dict[f"layers.{key}.attention.wv.weight"]
                        ), dim=0
                    )
                    part_state_dict[f"blocks.{idx}.attention.norm.weight"] = state_dict[f"layers.{key}.attention_norm.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.w1.weight"] = state_dict[f"layers.{key}.feed_forward.w1.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.w2.weight"] = state_dict[f"layers.{key}.feed_forward.w2.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.w3.weight"] = state_dict[f"layers.{key}.feed_forward.w3.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.norm.weight"] = state_dict[f"layers.{key}.ffn_norm.weight"]
            # special cases
            for key in list(part_state_dict.keys()):
                if model_args.dense == "raw" and "blocks" in key and "norm" not in key:
                    part_state_dict[key.replace(
                        "weight", "module.module.weight")] = part_state_dict.pop(key)
                if "language_model_head" in key:
                    part_state_dict[key.replace(
                        "weight", "module.module.weight")] = part_state_dict.pop(key)
                if "token_embedding" in key:
                    part_state_dict[key.replace(
                        "weight", "module.weight")] = part_state_dict.pop(key)
            with open (os.path.join(tempdir[0], f"pipeline_{pp_rank}.pt"), "wb+") as f:
                torch.save(part_state_dict, f)
    del state_dict, part_state_dict
    torch.distributed.broadcast_object_list(tempdir, src=0)
    with open(os.path.join(tempdir[0], f"pipeline_{gpc.get_local_rank(ParallelMode.PIPELINE)}.pt"), "rb") as f:
        state_dict = torch.load(f)
    torch.distributed.barrier()
    if gpc.get_global_rank() == 0:
        shutil.rmtree(tempdir[0])
    return state_dict

def save_state_dict(model: nn.Module, 
                    protocol: str="s3", 
                    file_folder: str = "/mnt/lustre/zhangshuo/model",
                    s3_folder: str = "hdd:s3://opennlplab_hdd/models/llama-tunelite/llama-7b/"):
    assert protocol in ["s3", "file"], "protocol must be one of s3, file"
    tempdir = [""]
    if gpc.get_global_rank() == 0:
        tempdir[0] = f"/dev/shm/TuneLite-{round(time.time() * 1000)}/"
    torch.distributed.broadcast_object_list(tempdir, src=0)
    with open(os.path.join(tempdir[0], f"pipeline_{gpc.get_local_rank(ParallelMode.PIPELINE)}.pt"), "rb") as f:
        torch.save(model.state_dict(), f)
    torch.distributed.barrier()
    if gpc.get_global_rank() == 0:
        state_dict = OrderedDict()
        for i in range(gpc.get_pipeline_model_parallel_size()):
            with open(os.path.join(tempdir[0], f"pipeline_{i}.pt"), "rb") as f:
                state_dict.update(torch.load(f))
        if protocol == "s3":
            if not s3_folder.endswith("/"):
                s3_folder += "/"
            from petrel_client.client import Client
            client = Client()
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            buffer.seek(0)
            client.put(f"{s3_folder}model.pth", buffer)
            buffer.close()
        elif protocol == "file":
            with open(os.path.join(file_folder, "model.pth"), "wb+") as f:
                torch.save(state_dict, f)
        shutil.rmtree(tempdir[0])
            
    
def get_7B_llama(model_args: ModelArgs = ModelArgs()):
    for key, value in {
        "vocab_size": 32000,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32
    }.items():
        setattr(model_args, key, value)
    return build_pipe(model_args)

def get_13B_llama(model_args: ModelArgs = ModelArgs()):
    for key, value in {
        "vocab_size": 32000,
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "num_hidden_layers": 40,
        "num_attention_heads": 40
    }.items():
        setattr(model_args, key, value)
    return build_pipe(model_args)


def get_30B_llama(model_args: ModelArgs = ModelArgs()):
    for key, value in {
        "vocab_size": 32000,
        "hidden_size": 6656,
        "intermediate_size": 17920,
        "num_hidden_layers": 60,
        "num_attention_heads": 52
    }.items():
        setattr(model_args, key, value)
    return build_pipe(model_args)
