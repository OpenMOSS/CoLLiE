# Copyright (c) Fudan University.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms
# of the GNU General Public License version 3.

# This code based on https://github.com/facebookresearch/llama
import torch
from torch import nn
import torch.nn.functional as F

import re
import io
import os
import time
import math
import tqdm
import json
import shutil
import warnings
from io import BytesIO
from einops import rearrange
from dataclasses import dataclass
from collections import OrderedDict
from sentencepiece import SentencePieceProcessor
from typing import Optional, Callable, Dict, List, Union

try:
    import colossalai
    import colossalai.nn as col_nn
    from colossalai import kernel as K
    from colossalai.amp import AMP_TYPE
    from colossalai.core import global_context as gpc
    from colossalai.pipeline.utils import partition_uniform
    from colossalai.nn.layer.base_layer import ParallelLayer
    from colossalai.context.parallel_mode import ParallelMode
    from colossalai.utils.activation_checkpoint import checkpoint
    from colossalai.nn.layer.wrapper import PipelineSharedModuleWrapper
    from colossalai.utils.model.colo_init_context import ColoInitContext
    from colossalai.nn.layer.parallel_1d import Linear1D_Col, Linear1D_Row
    from colossalai.logging import get_dist_logger, disable_existing_loggers
    from colossalai.nn.layer.parallel_1d.layers import VocabParallelEmbedding1D, VocabParallelClassifier1D
except ModuleNotFoundError:
    warnings.warn(
        "Detected Colossal-AI is not installed. See https://github.com/hpcaitech/ColossalAI")
    # raise ModuleNotFoundError(
    #     "Detected Colossal-AI is not installed. See https://github.com/hpcaitech/ColossalAI")
    K = None
    gpc = None
    col_nn = None
    AMP_TYPE = None
    checkpoint = None
    colossalai = None
    ParallelMode = None
    Linear1D_Col = None
    Linear1D_Row = None
    ParallelMode = None
    ParallelLayer = None
    get_dist_logger = None
    ColoInitContext = None
    partition_uniform = None
    VocabParallelEmbedding1D = None
    disable_existing_loggers = None
    VocabParallelClassifier1D = None
    PipelineSharedModuleWrapper = None
    
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

        self.pad_token = self.tokenizer.pad_id
        self.eos_token = self.tokenizer.eos_id
        self.bos_token = self.tokenizer.bos_id

        # mask attribute to be similar to hugging face
        self.eos_token_id = self.tokenizer.eos_id
        self.pad_token_id = self.tokenizer.pad_id

        # to match hugging face attribute
        self.pad_token_id = self.pad_id

    def __call__(self, texts: Union[List[str], str], *args, **kwargs):
        if isinstance(texts, str):
            text = self.tokenizer.encode(texts, kwargs.get(
                "bos", True), eos=kwargs.get("eos", True))
            tokens = torch.tensor(text).long()
        else:
            texts = [
                self.tokenizer.encode(text, kwargs.get(
                    "bos", True), eos=kwargs.get("eos", True))
                for text in texts
            ]
            max_len = max(len(text) for text in texts)
            tokens = torch.full(
                (len(texts), max_len), self.tokenizer.pad_id
            ).long()
            for i, text in enumerate(texts):
                tokens[i, :len(text)] = torch.tensor(  # noqa E203
                    text
                ).long()
            tokens = torch.where(
                tokens == self.tokenizer.pad_id,
                torch.zeros_like(tokens),
                tokens,
            )
        output = {
            "input_ids": tokens
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
    micro_batch_num: int = 1
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
            t = torch.arange((2 ** 16)
                             * 2, device=freqs.device)
            freqs = torch.outer(t, freqs).float()
            self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(
                torch.device(f"cuda:{os.environ.get('LOCAL_RANK')}"))
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
            output = object.__getattribute__(self, "rpoe")(
                qkv=qkv, seqlen_offset=start_pos)
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
            self.attention["wq"] = Linear1D_Col(self.model_args.hidden_size,
                                                   self.model_args.hidden_size,
                                                   bias=False)
            self.attention["wk"] = Linear1D_Col(self.model_args.hidden_size,
                                                   self.model_args.hidden_size,
                                                   bias=False)
            self.attention["wv"] = Linear1D_Col(self.model_args.hidden_size,
                                                   self.model_args.hidden_size,
                                                   bias=False)
            self.attention["wo"] = Linear1D_Row(self.model_args.hidden_size,
                                                 self.model_args.hidden_size,
                                                 bias=False)
            self.mlp["w1"] = Linear1D_Col(self.model_args.hidden_size,
                                           self.model_args.intermediate_size,
                                           bias=False)
            self.mlp["w2"] = Linear1D_Row(self.model_args.intermediate_size,
                                           self.model_args.hidden_size,
                                           bias=False)
            self.mlp["w3"] = Linear1D_Col(self.model_args.hidden_size,
                                           self.model_args.intermediate_size,
                                           bias=False)
        elif self.model_args.dense == "fused":
            assert FlashAttnFusedDense is not None, \
                "Detected fused_dense_lib is not installed. See https://github.com/HazyResearch/flash-attention/tree/main/csrc/fused_dense_lib"
            self.attention["wq"] = FlashAttnFusedDense(self.model_args.hidden_size,
                                                         self.model_args.hidden_size,
                                                         bias=False)
            self.attention["wk"] = FlashAttnFusedDense(self.model_args.hidden_size,
                                                         self.model_args.hidden_size,
                                                         bias=False)
            self.attention["wv"] = FlashAttnFusedDense(self.model_args.hidden_size,
                                                         self.model_args.hidden_size,
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
            self.attention["wq"] = ApexFusedDense(self.model_args.hidden_size,
                                                    self.model_args.hidden_size * 3,
                                                    bias=False)
            self.attention["wk"] = ApexFusedDense(self.model_args.hidden_size,
                                                    self.model_args.hidden_size * 3,
                                                    bias=False)
            self.attention["wv"] = ApexFusedDense(self.model_args.hidden_size,
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
            from colossalai.kernel.cuda_native.flash_attention import flash_attention_qkv

            def attention(**kwargs):
                kwargs["qkv"] = rearrange(
                    kwargs["qkv"], "b n three h d -> (b n) three h d")
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
        self.key_cache = [None for _ in range(self.model_args.micro_batch_num)]
        self.value_cache = [None for _ in range(self.model_args.micro_batch_num)]
        self.micro_batch_counter = 0
        
    def clean_cache(self):
        self.key_cache = [None for _ in range(self.model_args.micro_batch_num)]
        self.value_cache = [None for _ in range(self.model_args.micro_batch_num)]
        self.micro_batch_counter = 0 # 现在自己处于第几个 micro batch
        torch.cuda.empty_cache()

    def forward(self,
                hidden_states: Optional[torch.Tensor],
                causal: bool = True,
                use_cache: bool = False,
                rpoe: Callable = None):

        assert hidden_states.ndim == 3, f"hidden_states.shape must be (B, N, H), but got {hidden_states.shape}"
        batch_size, seq_len, hidden_size = hidden_states.shape
        head_dim = self.model_args.hidden_size // self.model_args.num_attention_heads
        _hidden_states = self.attention["norm"](hidden_states)
        query, key, value = self.attention["wq"](_hidden_states), self.attention["wk"](_hidden_states), self.attention["wv"](_hidden_states)
        qkv = rearrange(torch.stack([query, key, value], dim=2), "b n three (h d) -> b n three h d", h=self.model_args.num_attention_heads // gpc.get_world_size(ParallelMode.TENSOR))
        query, key, value = torch.split(
            qkv, split_size_or_sections=1, dim=2)
        query, key, value = query.squeeze(
            2), key.squeeze(2), value.squeeze(2)
        if use_cache:
            start_pos = self.key_cache[self.micro_batch_counter].shape[1] if self.key_cache[self.micro_batch_counter] is not None else 0
        else:
            start_pos = 0
        query, key = rpoe(query=query, key=key,
                          start_pos=start_pos, seq_len=seq_len)
        
        if use_cache:
            if self.key_cache[self.micro_batch_counter] is None or self.value_cache[self.micro_batch_counter] is None:
                self.key_cache[self.micro_batch_counter] = key
                self.value_cache[self.micro_batch_counter] = value
            else:
                query = torch.concat(
                    [torch.zeros_like(self.key_cache[self.micro_batch_counter]).type_as(self.key_cache[self.micro_batch_counter]), query], dim=1)
                key = self.key_cache[self.micro_batch_counter] = torch.concat(
                    [self.key_cache[self.micro_batch_counter], key], dim=1)
                value = self.value_cache[self.micro_batch_counter] = torch.concat(
                    [self.value_cache[self.micro_batch_counter], value], dim=1)
        self.micro_batch_counter = self.micro_batch_counter + 1
        if self.micro_batch_counter >= len(self.key_cache):
            self.micro_batch_counter = 0
        qkv = torch.stack([query, key, value], dim=2)
        attention_output = self.attention_fn(qkv=qkv,
                                             sm_scale=1 / math.sqrt(head_dim),
                                             batch_size=batch_size,
                                             seq_len=seq_len + start_pos,
                                             dropout_p=self.model_args.dropout,
                                             causal=causal)
        if use_cache:
            attention_output = attention_output[:, -seq_len:]
        hidden_states = hidden_states + self.attention["wo"](
            attention_output
        )
        _hidden_states = self.mlp["norm"](hidden_states)
        hidden_states = hidden_states + self.mlp["dropout"](
            self.mlp["w2"](F.silu(self.mlp["w1"](_hidden_states)) * self.mlp["w3"](_hidden_states)))
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
            # self.token_embedding = nn.Embedding(
            #     self.model_args.vocab_size,
            #     embedding_dim=self.model_args.hidden_size)
            self.token_embedding = VocabParallelEmbedding1D(
                self.model_args.vocab_size,
                embedding_dim=self.model_args.hidden_size
            )
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
                self.language_model_head = VocabParallelClassifier1D(self.model_args.hidden_size,
                                                        self.model_args.vocab_size,
                                                        bias=False,
                                                        gather_output=True)
                # self.language_model_head = nn.Linear(self.model_args.hidden_size,
                #                                         self.model_args.vocab_size,
                #                                         bias=False)
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
                use_cache: torch.Tensor = torch.zeros(1, dtype=torch.bool),
                **kwargs):
        if self.is_start:
            assert input_ids is not None, "`input_ids` is not allowed to be None in the first pipeline node. "
            hidden_states = self.token_embedding(input_ids)
        for i in range(len(self.blocks)):
            if self.model_args.checkpoint and self.training:
                hidden_states = checkpoint(self.blocks[i], True,
                                           hidden_states,
                                           True,
                                           use_cache[0],
                                           self.rope)
            else:
                hidden_states = self.blocks[i](
                    hidden_states,
                    True,
                    use_cache[0],
                    self.rope)
        if self.is_end:
            hidden_states = self.norm(hidden_states)
            hidden_states = self.language_model_head(hidden_states)
        return hidden_states


def prepare_distribution(model_args: ModelArgs = ModelArgs()) -> dict:
    CONFIG = dict(NUM_MICRO_BATCHES=model_args.micro_batch_num,
                  parallel=dict(
                      pipeline=int(model_args.pp_size),
                      tensor=dict(size=model_args.tp_size, mode="1d")
                  )
                  )
    if model_args.fp16:
        CONFIG["fp16"] = dict(mode=AMP_TYPE.NAIVE)
    if model_args.tp_size > 1:
        if model_args.dense != "raw":
            warnings.warn("Fused dense is not supported in tensor parallelism. ")
            model_args.dense = "raw"
    colossalai.launch_from_torch(config=CONFIG, backend=model_args.backend)
    if "pipeline" in CONFIG["parallel"] and CONFIG["parallel"]["pipeline"] == 1:
        gpc.is_pipeline_first_stage = lambda: True
        gpc.is_pipeline_last_stage = lambda: True
        gpc._local_ranks[ParallelMode.PIPELINE] = 0
        gpc._world_sizes[ParallelMode.PIPELINE] = 1
    if "tensor" in CONFIG["parallel"] and CONFIG["parallel"]["tensor"]["size"] == 1:
        gpc._local_ranks[ParallelMode.TENSOR] = 0
        gpc._world_sizes[ParallelMode.TENSOR] = 1


def build_pipe(model_args: ModelArgs = ModelArgs()):
    prepare_distribution(model_args=model_args)
    disable_existing_loggers()
    logger = get_dist_logger()
    with ColoInitContext(device=torch.device(f"cuda:{os.environ.get('LOCAL_RANK')}"), dtype=torch.float16 if model_args.fp16 else torch.float32):
        if model_args.pp_size > 1:
            wrapper = PipelineSharedModuleWrapper(
                [0, model_args.pp_size - 1])
        parts = partition_uniform(
            model_args.num_hidden_layers, model_args.pp_size, num_chunks=1)[gpc.get_local_rank(ParallelMode.PIPELINE)]
        chunk_list = []
        for start, end in parts:
            logger.info(
                f'Rank{os.environ.get("LOCAL_RANK")} build layer {start}-{end}, {end - start}/{model_args.num_hidden_layers} layers')
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


def load_state_dict(protocol: str = "s3",
                    format: str = "hf",
                    file_folder: str = "/remote-home/share/llama/7B",
                    s3_folder: str = "hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf",
                    model_args: ModelArgs = ModelArgs()) -> Dict[str, torch.tensor]:
    assert format in ["hf", "raw"], "format must be hf or raw"
    assert protocol in ["s3", "file"], "protocol must be one of s3, file"
    state_dict = OrderedDict()
    part_state_dict = OrderedDict()
    tempdir = [""]
    if not torch.distributed.is_initialized() or gpc.get_local_rank(ParallelMode.GLOBAL) == 0:
        if protocol == "s3":
            from petrel_client.client import Client
            client = Client()
            if not s3_folder.endswith("/"):
                s3_folder = f"{s3_folder}/"
            if format == "raw":
                if client.contains(os.path.join(s3_folder, "params.json")):
                    params = json.loads(client.get(os.path.join(s3_folder, "params.json")).decode())
                    for key, value in {
                        "hidden_size": params["dim"],
                        "intermediate_size": params["multiple_of"] * ((int(2 * 4 * model_args.hidden_size / 3) + params["multiple_of"] - 1) // params["multiple_of"]),
                        "num_hidden_layers": params["n_layers"],
                        "num_attention_heads": params["n_heads"],
                        "layer_norm_epsilon": params["norm_eps"]
                    }.items():
                        setattr(model_args, key, value)
                weights = [weight for weight in client.list(
                    s3_folder) if weight.endswith(".pth") or weight.endswith(".pt")]
            elif format == "hf":
                if client.contains(os.path.join(s3_folder, "config.json")):
                    config = json.loads(client.get(os.path.join(s3_folder, "config.json")).decode())
                    for key, value in {
                        "vocab_size": config["vocab_size"],
                        "hidden_size": config["hidden_size"],
                        "intermediate_size": config["intermediate_size"],
                        "num_hidden_layers": config["num_hidden_layers"],
                        "num_attention_heads": config["num_attention_heads"],
                        "fp16": config["torch_dtype"] == "float16",
                        "layer_norm_epsilon": config["rms_norm_eps"]
                    }.items():
                        setattr(model_args, key, value)
                weights = [weight for weight in client.list(
                    s3_folder) if weight.endswith(".bin")]
            with tqdm.tqdm(desc=f"Loading state dict", total=len(weights)) as pbar:
                for content in weights:
                    buffer = BytesIO()
                    buffer.write(client.get(f"{s3_folder}{content}"))
                    buffer.seek(0)
                    raw_state_dict = torch.load(buffer, map_location="cpu")
                    for key, value in raw_state_dict.items():
                        if format == "hf":
                            if key.endswith("q_proj.weight") or key.endswith("k_proj.weight"):
                                raw_state_dict[key] = rearrange(
                                    value,
                                    "(h two t) d -> h two t d",
                                    h=model_args.num_attention_heads,
                                    two=2).transpose(1, 2).reshape(
                                        model_args.hidden_size,
                                        model_args.hidden_size)
                            state_dict.update(raw_state_dict)
                        elif format == "raw":
                            if key in state_dict.keys():
                                if key.endswith("wo.weight") or key.endswith("w2.weight") or key.endswith("embeddings.weight"):
                                    state_dict[key] = torch.cat(
                                        (state_dict[key], value), dim=1)
                                elif key.endswith("norm.weight"):
                                    pass
                                else:
                                    state_dict[key] = torch.cat(
                                        (state_dict[key], value), dim=0)
                            else:
                                state_dict[key] = value
                    buffer.close()
                    pbar.update(1)
        elif protocol == "file":
            if format == "raw":
                if os.path.exists(os.path.join(file_folder, "params.json")):
                    params = json.loads(open(os.path.join(file_folder, "params.json"), mode="r").read())
                    for key, value in {
                        "hidden_size": params["dim"],
                        "intermediate_size": params["multiple_of"] * ((int(2 * 4 * model_args.hidden_size / 3) + params["multiple_of"] - 1) // params["multiple_of"]),
                        "num_hidden_layers": params["n_layers"],
                        "num_attention_heads": params["n_heads"],
                        "layer_norm_epsilon": params["norm_eps"]
                    }.items():
                        setattr(model_args, key, value)
                weights = [weight for weight in list(
                    os.listdir(file_folder)) if weight.endswith(".pth") or weight.endswith(".pt")]
            elif format == "hf":
                if os.path.exists(os.path.join(file_folder, "config.json")):
                    config = json.loads(open(os.path.join(file_folder, "config.json"), mode="r").read())
                    for key, value in {
                        "vocab_size": config["vocab_size"],
                        "hidden_size": config["hidden_size"],
                        "intermediate_size": config["intermediate_size"],
                        "num_hidden_layers": config["num_hidden_layers"],
                        "num_attention_heads": config["num_attention_heads"],
                        "fp16": config["torch_dtype"] == "float16",
                        "layer_norm_epsilon": config["rms_norm_eps"]
                    }.items():
                        setattr(model_args, key, value)
                weights = [weight for weight in list(
                    os.listdir(file_folder)) if weight.endswith(".bin")]
            state_dict = OrderedDict()
            with tqdm.tqdm(desc=f"Loading state dict", total=len(weights)) as pbar:
                for weight in weights:
                    raw_state_dict = torch.load(os.path.join(
                        file_folder, weight), map_location="cpu")
                    for key, value in raw_state_dict.items():
                        if format == "hf":
                            if key.endswith("q_proj.weight") or key.endswith("k_proj.weight"):
                                raw_state_dict[key] = rearrange(
                                    value,
                                    "(h two t) d -> h two t d",
                                    h=model_args.num_attention_heads,
                                    two=2).transpose(1, 2).reshape(
                                        model_args.hidden_size,
                                        model_args.hidden_size)
                                state_dict.update(raw_state_dict)
                            else:
                                state_dict.update(raw_state_dict)
                        elif format == "raw":
                            if key in state_dict.keys():
                                if key.endswith("wo.weight") or key.endswith("w2.weight") or key.endswith("embeddings.weight"):
                                    state_dict[key] = torch.cat(
                                        (state_dict[key], value), dim=1)
                                elif key.endswith("norm.weight"):
                                    pass
                                else:
                                    state_dict[key] = torch.cat(
                                        (state_dict[key], value), dim=0)
                            else:
                                state_dict[key] = value
                    pbar.update(1)
        parts = partition_uniform(
            model_args.num_hidden_layers, model_args.pp_size if torch.distributed.is_initialized() else 1, num_chunks=1)
        tempdir[0] = f"/dev/shm/Collie-{round(time.time() * 1000)}/"
        os.makedirs(tempdir[0], exist_ok=True)
        for pp_rank, [(start, end)] in enumerate(parts):
            part_state_dict = OrderedDict()
            if format == "hf":
                if start == 0:
                    part_state_dict["token_embedding.weight"] = state_dict["model.embed_tokens.weight"]
                if end == model_args.num_hidden_layers:
                    part_state_dict["language_model_head.weight"] = state_dict["lm_head.weight"]
                    part_state_dict["norm.weight"] = state_dict["model.norm.weight"]
            elif format == "raw":
                if start == 0:
                    part_state_dict["token_embedding.weight"] = state_dict["tok_embeddings.weight"]
                if end == model_args.num_hidden_layers:
                    part_state_dict["language_model_head.weight"] = state_dict["output.weight"]
                    part_state_dict["norm.weight"] = state_dict["norm.weight"]
            for idx, key in enumerate(list(range(start, end))):
                if format == "hf":
                    part_state_dict[f"blocks.{idx}.attention.wo.weight"] = state_dict[f"model.layers.{key}.self_attn.o_proj.weight"]
                    part_state_dict[f"blocks.{idx}.attention.wq.weight"] = state_dict[f"model.layers.{key}.self_attn.q_proj.weight"]
                    part_state_dict[f"blocks.{idx}.attention.wk.weight"] = state_dict[f"model.layers.{key}.self_attn.k_proj.weight"]
                    part_state_dict[f"blocks.{idx}.attention.wv.weight"] = state_dict[f"model.layers.{key}.self_attn.v_proj.weight"]
                    part_state_dict[f"blocks.{idx}.attention.norm.weight"] = state_dict[
                        f"model.layers.{key}.input_layernorm.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.w1.weight"] = state_dict[f"model.layers.{key}.mlp.gate_proj.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.w2.weight"] = state_dict[f"model.layers.{key}.mlp.down_proj.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.w3.weight"] = state_dict[f"model.layers.{key}.mlp.up_proj.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.norm.weight"] = state_dict[
                        f"model.layers.{key}.post_attention_layernorm.weight"]
                elif format == "raw":
                    part_state_dict[f"blocks.{idx}.attention.wo.weight"] = state_dict[f"layers.{key}.attention.wo.weight"]
                    part_state_dict[f"blocks.{idx}.attention.wq.weight"] = state_dict[f"layers.{key}.attention.wq.weight"]
                    part_state_dict[f"blocks.{idx}.attention.wk.weight"] = state_dict[f"layers.{key}.attention.wk.weight"]
                    part_state_dict[f"blocks.{idx}.attention.wv.weight"] = state_dict[f"layers.{key}.attention.wv.weight"]
                    part_state_dict[f"blocks.{idx}.attention.norm.weight"] = state_dict[f"layers.{key}.attention_norm.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.w1.weight"] = state_dict[f"layers.{key}.feed_forward.w1.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.w2.weight"] = state_dict[f"layers.{key}.feed_forward.w2.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.w3.weight"] = state_dict[f"layers.{key}.feed_forward.w3.weight"]
                    part_state_dict[f"blocks.{idx}.mlp.norm.weight"] = state_dict[f"layers.{key}.ffn_norm.weight"]
            with open(os.path.join(tempdir[0], f"pipeline_{pp_rank}.pt"), "wb+") as f:
                torch.save(part_state_dict, f)
    del state_dict, part_state_dict
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(tempdir, src=0)
        with open(os.path.join(tempdir[0], f"pipeline_{gpc.get_local_rank(ParallelMode.PIPELINE)}.pt"), "rb") as f:
            state_dict = torch.load(f)
        torch.distributed.barrier()
    else:
        with open(os.path.join(tempdir[0], f"pipeline_0.pt"), "rb") as f:
            state_dict = torch.load(f)
    if not torch.distributed.is_initialized() or gpc.get_local_rank(ParallelMode.GLOBAL) == 0:
        shutil.rmtree(tempdir[0])
    return state_dict


def save_parallel_model(model: nn.Module,
                        protocol: str = "s3",
                        format: str = "hf",
                        file_folder: str = "/mnt/lustre/zhangshuo/model",
                        s3_folder: str = "hdd:s3://opennlplab_hdd/models/llama-collie/llama-7b/",
                        raw_tp_size: int = 1,
                        raw_tp_device_map: Optional[Dict] = None,
                        model_args: ModelArgs = ModelArgs()):
    assert protocol in ["s3", "file"], "protocol must be one of s3, file"
    assert format in ["hf", "raw"], "format must be hf or raw"
    tempdir = [""]
    if gpc.get_local_rank(ParallelMode.GLOBAL) == 0:
        tempdir[0] = f"/dev/shm/Collie-{round(time.time() * 1000)}/"
        os.makedirs(tempdir[0], exist_ok=True)
    torch.distributed.broadcast_object_list(tempdir, src=0)
    with ParallelLayer.use_local_state_dict():
        part_state_dict = model.state_dict()
    for key in list(part_state_dict.keys()):
        module = model
        for subkey in key.split(".")[:-1]:
            if subkey.isdigit():
                module = module[int(subkey)]
            else:
                module = getattr(module, subkey)
        if "col" in module.__class__.__name__.lower()  or "vocab" in module.__class__.__name__.lower():
            part_state_dict[f"{key}-col"] = part_state_dict[key].cpu()
        elif "row" in module.__class__.__name__.lower():
            part_state_dict[f"{key}-row"] = part_state_dict[key].cpu()
        else:
            part_state_dict[key] = part_state_dict[key].cpu()
    with open(os.path.join(tempdir[0], f"pipeline_{gpc.get_local_rank(ParallelMode.PIPELINE)}_tensor_{gpc.get_local_rank(ParallelMode.TENSOR)}.pt"), "wb") as f:
        torch.save(part_state_dict, f)
    torch.distributed.barrier()
    if gpc.get_local_rank(ParallelMode.GLOBAL) == 0:
        state_dict = OrderedDict()
        parts = partition_uniform(model_args.num_hidden_layers, model_args.pp_size, num_chunks=1)
        files = [file for file in list(os.listdir(tempdir[0])) if file.endswith(".pt")]
        pp_tp_map = OrderedDict({file: list(map(int, re.match(r'pipeline_(\d+)_tensor_(\d+)\.pt', file).groups())) for file in files})
        pp_range = range(min([value[0] for value in pp_tp_map.values()]), max([value[0] for value in pp_tp_map.values()]) + 1)
        tp_range = range(max([value[1] for value in pp_tp_map.values()]) + 1)
        for pp in pp_range:
            # max_layer = max([-1] + [int(match.groups()[0]) for match in [re.match(r'blocks\.(\d+)\..*', key) for key in state_dict.keys()] if match is not None])
            lower_bound, upper_bound = parts[pp]
            for tp in tp_range:
                with open(os.path.join(tempdir[0], f'pipeline_{pp}_tensor_{tp}.pt'), "rb") as f:
                    part_state_dict = torch.load(f, map_location="cpu")
                    for key in list(part_state_dict.keys()):
                        if key.startswith("blocks."):
                            idx = int(re.match(r'blocks\.(\d+)\..*', key).groups()[0])
                            part_state_dict[f"blocks.{idx + lower_bound}.{'.'.join(key.split('.')[2:])}"] = part_state_dict.pop(key)
                    for key in list(part_state_dict.keys()):
                        if key.endswith("-col") or key.endswith("-row"):
                            if key not in state_dict.keys():
                                state_dict[key] = [None] * len(tp_range)
                                state_dict[key][tp] = part_state_dict[key]
                            else:
                                state_dict[key][tp] = part_state_dict[key]
                        else:
                            state_dict[key] = part_state_dict[key]
        for key in list(state_dict.keys()):
            if key.endswith("-col") and isinstance(state_dict[key], List):
                if len(state_dict[key]) == 1:
                    state_dict[key.replace("-col", "")] = state_dict.pop(key)[0]
                else:
                    state_dict[key.replace("-col", "")] = torch.cat(state_dict.pop(key), dim=0)
            elif key.endswith("-row") and isinstance(state_dict[key], List):
                if len(state_dict[key]) == 1:
                    state_dict[key.replace("-row", "")] = state_dict.pop(key)[0]
                else:
                    state_dict[key.replace("-row", "")] = torch.cat(state_dict.pop(key), dim=1)
        save_state_dict(
            state_dict, 
            protocol=protocol, 
            format=format, 
            file_folder=file_folder, 
            s3_folder=s3_folder, 
            raw_tp_size=raw_tp_size, 
            raw_tp_device_map=raw_tp_device_map, 
            model_args=model_args)
        shutil.rmtree(tempdir[0])

def save_state_dict(state_dict: Dict,
                    protocol: str = "s3",
                    format: str = "hf",
                    file_folder: str = "/mnt/lustre/zhangshuo/model",
                    s3_folder: str = "hdd:s3://opennlplab_hdd/models/llama-collie/llama-7b/",
                    raw_tp_size: int = 1,
                    raw_tp_device_map: Optional[Dict] = None,
                    raw_multiple_of: int = 256,
                    save_to_buffer: bool = False,
                    model_args: ModelArgs = ModelArgs()):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        warnings.warn("Only rank 0 should save the state_dict.")
        return
    if save_to_buffer:
        buffers = {}
    folder = {
        "s3": s3_folder,
        "file": file_folder
    }
    raw_state_dict = OrderedDict()
    hf_state_dict = OrderedDict()
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, (model_args.hidden_size // model_args.num_attention_heads),
                          2).float() / (model_args.hidden_size // model_args.num_attention_heads)))
    if raw_tp_device_map is None:
        raw_tp_device_map = {device: 'cpu' for device in range(raw_tp_size)}
    def set_tensor_parallel(w: torch.Tensor, dim: int):
        if raw_tp_size == 1:
            return w
        else:
            return [tensor.to(torch.device(raw_tp_device_map[device] if isinstance(raw_tp_device_map[device], str) else f"cuda:{raw_tp_device_map[device]}"))
                    for tensor, device in zip(torch.chunk(w, raw_tp_size, dim=dim), list(range(raw_tp_size)))]
        
    def reshape_wq_wk(w: torch.Tensor):
        return w.view(model_args.num_attention_heads, model_args.hidden_size // model_args.num_attention_heads //
                      2, 2, model_args.hidden_size).transpose(1, 2).reshape(model_args.hidden_size, model_args.hidden_size)
        
    def save_obj(obj, path, protocol: str="file"):
        if save_to_buffer:
            if isinstance(obj, str):
                buffers[path] = obj.encode()
            else:
                buffer = BytesIO()
                torch.save(obj, buffer)
                buffer.seek(0)
                buffers[path] = buffer.read()
        else:
            if protocol == "file":
                os.makedirs(os.path.dirname(path), exist_ok=True)
                if isinstance(obj, str):
                    with open(path, "w") as f:
                        f.write(obj)
                else:
                    with open(path, "wb") as f:
                        torch.save(obj, f)
            elif protocol == "s3":
                from petrel_client.client import Client
                client = Client()
                buffer = BytesIO()
                if isinstance(obj, str):
                    buffer.write(obj.encode())
                else:
                    torch.save(obj, buffer)
                buffer.seek(0)
                client.put(path, buffer)
                buffer.close()

    with tqdm.tqdm(state_dict.items(), desc=f"Converting state dict", total=len(state_dict.items())) as pbar:
        for step, (key, value) in enumerate(pbar):
            if format == "hf":
                if key.endswith("wq.weight"):
                    hf_state_dict[key.replace("blocks", "model.layers").replace(
                        "attention.wq.weight", "self_attn.q_proj.weight")] = reshape_wq_wk(value)
                if key.endswith("wk.weight"):
                    hf_state_dict[key.replace("blocks", "model.layers").replace(
                        "attention.wk.weight", "self_attn.k_proj.weight")] = reshape_wq_wk(value)
                if key.endswith("wv.weight"):
                    hf_state_dict[key.replace("blocks", "model.layers").replace(
                        "attention.wv.weight", "self_attn.v_proj.weight")] = value
                if key.endswith("wo.weight"):
                    hf_state_dict[key.replace("blocks", "model.layers").replace(
                        "attention.wo.weight", "self_attn.o_proj.weight")] = value
                if key.endswith("mlp.w1.weight"):
                    hf_state_dict[key.replace("blocks", "model.layers").replace(
                        "w1.weight", "gate_proj.weight")] = value
                if key.endswith("mlp.w2.weight"):
                    hf_state_dict[key.replace("blocks", "model.layers").replace(
                        "w2.weight", "down_proj.weight")] = value
                if key.endswith("mlp.w3.weight"):
                    hf_state_dict[key.replace("blocks", "model.layers").replace(
                        "w3.weight", "up_proj.weight")] = value
                if key.endswith("attention.norm.weight"):
                    hf_state_dict[key.replace("blocks", "model.layers").replace(
                        "attention.norm.weight", "input_layernorm.weight")] = value
                if key.endswith("mlp.norm.weight"):
                    hf_state_dict[key.replace("blocks", "model.layers").replace(
                        "mlp.norm.weight", "post_attention_layernorm.weight")] = value
                if key.endswith("token_embedding.weight"):
                    hf_state_dict["model.embed_tokens.weight"] = value
                if key.endswith("language_model_head.weight"):
                    hf_state_dict["lm_head.weight"] = value
                if key == "norm.weight":
                    hf_state_dict["model.norm.weight"] = value
            if format == "raw":
                if key.endswith("wq.weight") or key.endswith("wk.weight") or \
                    key.endswith("wv.weight"):
                    raw_state_dict[key.replace("blocks", "layers")] = set_tensor_parallel(value, dim=0)
                if key.endswith("wo.weight"):
                    raw_state_dict[key.replace("blocks", "layers")] = set_tensor_parallel(value, dim=1)
                if key.endswith("mlp.w1.weight") or key.endswith("mlp.w3.weight"):
                    raw_state_dict[key.replace("blocks", "layers").replace(
                        "mlp", "feed_forward")] = set_tensor_parallel(value, dim=0)
                if key.endswith("mlp.w2.weight"):
                    raw_state_dict[key.replace("blocks", "layers").replace(
                        "mlp", "feed_forward")] = set_tensor_parallel(value, dim=1)
                if key.endswith("attention.norm.weight"):
                    raw_state_dict[key.replace("blocks", "layers").replace(
                        "attention.norm.weight", "attention_norm.weight")] = value
                if key.endswith("mlp.norm.weight"):
                    raw_state_dict[key.replace("blocks", "layers").replace(
                        "mlp.norm.weight", "ffn_norm.weight")] = value
                if key.endswith("token_embedding.weight"):
                    raw_state_dict["tok_embeddings.weight"] = set_tensor_parallel(value, dim=1)
                if key.endswith("language_model_head.weight"):
                    raw_state_dict["output.weight"] = set_tensor_parallel(value, dim=0)
                if key == "norm.weight":
                    raw_state_dict["norm.weight"] = value
            pbar.update(1)
    if format == "raw":
        for layer in range(model_args.num_hidden_layers):
            raw_state_dict[f'layers.{layer}.attention.inner_attention.rope.freqs'] = inv_freq
        for i in range(raw_tp_size):
            save_obj({key: value[i] if isinstance(value, list) else value
                    for key, value in raw_state_dict.items()}, os.path.join(folder[protocol], "consolidated.{:0>2}.pth".format(i)), protocol)
        params = {"dim": model_args.hidden_size, 
                  "multiple_of": raw_multiple_of, 
                  "n_heads": model_args.num_attention_heads, 
                  "n_layers": model_args.num_hidden_layers, 
                  "norm_eps": model_args.layer_norm_epsilon, 
                  "vocab_size": -1}
        save_obj(json.dumps(params, indent=4), os.path.join(folder[protocol], "params.json"), protocol)
    if format == "hf":
        model_index = OrderedDict({
            "weight_map": {},
            "metadata": {"total_size": 0}
        })
        with tqdm.tqdm(range(model_args.num_hidden_layers), desc=f"Saving state dict", total=model_args.num_hidden_layers) as pbar:
            for layer in pbar:
                filename = f"pytorch_model-{layer + 1}-of-{model_args.num_hidden_layers}.bin"
                layer_state_dict = {key: value for key, value in hf_state_dict.items(
                ) if key.startswith(f"model.layers.{layer}.")}
                if layer == 0:
                    layer_state_dict["model.embed_tokens.weight"] = hf_state_dict["model.embed_tokens.weight"]
                if layer == model_args.num_hidden_layers - 1:
                    layer_state_dict["lm_head.weight"] = hf_state_dict["lm_head.weight"]
                    layer_state_dict["model.norm.weight"] = hf_state_dict["model.norm.weight"]
                
                layer_state_dict[f"model.layers.{layer}.self_attn.rotary_emb.inv_freq"] = inv_freq
                model_index["weight_map"].update({
                    key: filename for key in layer_state_dict.keys()
                })
                save_obj(layer_state_dict, os.path.join(folder[protocol], filename), protocol)
                pbar.update(1)
        save_obj(json.dumps(model_index, indent=4), os.path.join(folder[protocol], "pytorch_model.bin.index.json"), protocol)
        config = {"architectures": ["LLaMAForCausalLM"], 
                "bos_token_id": 0, 
                "eos_token_id": 1, 
                "hidden_act": "silu", 
                "hidden_size": model_args.hidden_size, 
                "intermediate_size": model_args.intermediate_size, 
                "initializer_range": 0.02, 
                "max_sequence_length": 2048, 
                "model_type": "llama", 
                "num_attention_heads": model_args.num_attention_heads, 
                "num_hidden_layers": model_args.num_hidden_layers, 
                "pad_token_id": -1, 
                "rms_norm_eps": model_args.layer_norm_epsilon, 
                "torch_dtype": "float16" if model_args.fp16 else "float32", 
                "transformers_version": "4.27.0.dev0", 
                "use_cache": True, 
                "vocab_size": model_args.vocab_size}
        save_obj(json.dumps(config, indent=4), os.path.join(folder[protocol], "config.json"), protocol)
    if save_to_buffer:
        return buffers



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

def get_65B_llama(model_args: ModelArgs = ModelArgs()):
    for key, value in {
        "vocab_size": 32000,
        "hidden_size": 8192,
        "intermediate_size": 22016,
        "num_hidden_layers": 80,
        "num_attention_heads": 64
    }.items():
        setattr(model_args, key, value)
    return build_pipe(model_args)

def get_7B_llama_config(model_args: ModelArgs = ModelArgs()):
    for key, value in {
        "vocab_size": 32000,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32
    }.items():
        setattr(model_args, key, value)
    return model_args


def get_13B_llama_config(model_args: ModelArgs = ModelArgs()):
    for key, value in {
        "vocab_size": 32000,
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "num_hidden_layers": 40,
        "num_attention_heads": 40
    }.items():
        setattr(model_args, key, value)
    return model_args


def get_30B_llama_config(model_args: ModelArgs = ModelArgs()):
    for key, value in {
        "vocab_size": 32000,
        "hidden_size": 6656,
        "intermediate_size": 17920,
        "num_hidden_layers": 60,
        "num_attention_heads": 52
    }.items():
        setattr(model_args, key, value)
    return model_args

def get_65B_llama_config(model_args: ModelArgs = ModelArgs()):
    for key, value in {
        "vocab_size": 32000,
        "hidden_size": 8192,
        "intermediate_size": 22016,
        "num_hidden_layers": 80,
        "num_attention_heads": 64
    }.items():
        setattr(model_args, key, value)
    return model_args