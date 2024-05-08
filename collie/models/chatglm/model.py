import gc
import json
import math
import os
import numbers
from collections import OrderedDict
from typing import Any, Optional, Tuple, Union, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.checkpoint
from deepspeed.pipe import LayerSpec, TiedLayerSpec
from einops import rearrange
from megatron.core import parallel_state, tensor_parallel
from torch import Tensor, nn
from torch.nn.modules.module import Module
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PretrainedConfig, dtype_byte_size

from collie.config import CollieConfig
from collie.driver.io import IODriver
from collie.log.logger import logger
from collie.models.base import CollieModelForCausalLM
from collie.module import (
    ColumnParallelLinearWithoutBias,
    ColumnParallelLMHead,
    RowParallelLinearWithoutBias,
)
from collie.utils import concat_tensor, dict_as_params, env, progress 
from collie.models.utils import (kv_cache_to_inputs_for_model, inputs_to_kv_cache_for_model,\
                                kv_cache_to_inputs_for_layer, inputs_to_kv_cache_for_layer)
# try:
#     from flash_attn.flash_attention import FlashAttention
# except (ModuleNotFoundError, ImportError):
#     FlashAttention = None


# class RotaryPositionEmbedding(nn.Module):
#     def __init__(self, head_dim: int) -> None:
#         super().__init__()
#         inv_freq = 1.0 / (10000.0 ** (
#             torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
#         self.register_buffer('inv_freq', inv_freq)

#     def forward(self,
#                 query: torch.Tensor,
#                 key: torch.Tensor,
#                 seq_len: int,
#                 start_pos: int = 0):
#         t = query.dtype
#         query = torch.view_as_complex(
#             query.float().reshape(*query.shape[:-1], -1, 2))
#         key = torch.view_as_complex(
#             key.float().reshape(*key.shape[:-1], -1, 2))
#         freqs = torch.outer(torch.arange(
#             (2 ** 16) * 2, device=self.inv_freq.device), self.inv_freq).float()
#         freqs_cis = torch.polar(torch.ones_like(freqs), freqs)[
#             start_pos: start_pos + seq_len]
#         print(freqs_cis.shape)
#         freqs_cis = torch.cat([freqs_cis, freqs_cis], dim=-1)
#         shape = [d if i == 1 or i == query.ndim -
#                  1 else 1 for i, d in enumerate(query.shape)]
#         freqs_cis = freqs_cis.view(*shape)
#         query = torch.view_as_real(query * freqs_cis).flatten(3)
#         key = torch.view_as_real(key * freqs_cis).flatten(3)
#         return query.type(t), key.type(t)


class LayerNorm(Module):
    r""" Copy from torch.nn.LayerNorm
    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the module of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> # NLP Example
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = nn.LayerNorm(embedding_dim)
        >>> # Activate module
        >>> layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = torch.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = nn.LayerNorm([C, H, W])
        >>> output = layer_norm(input)

    .. image:: ../_static/img/nn/layer_norm.jpg
        :scale: 50 %

    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = torch.nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        # input dtype convert to weight.dtype before calculate
        input_dtype = input.dtype
        input = input.to(self.weight.dtype)
        output = F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)
        return output.to(input_dtype)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq.half()
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer("inv_freq", inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        pass

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def _apply(self, fn):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), F.embedding(
        position_id, sin.squeeze(1)
    ).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k


class ChatGLMLayer(nn.Module):
    def __init__(self, config: CollieConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.attention = nn.ModuleDict(
            {
                "query_key_value": ColumnParallelLinearWithoutBias(
                    config.hidden_size,
                    config.hidden_size * 3,
                    gather_output=False,
                    init_method=lambda x: x,
                ),
                "dense": RowParallelLinearWithoutBias(
                    config.hidden_size,
                    config.hidden_size,
                    input_is_parallel=True,
                    init_method=lambda x: x,
                ),
                "rotary_emb": RotaryEmbedding(
                    self.config.hidden_size // (self.config.num_attention_heads * 2)
                ),
            }
        )
        # self.input_layernorm = nn.LayerNorm(
        #     config.hidden_size, eps=config.layernorm_epsilon
        # )
        self.input_layernorm = LayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )
        self.mlp = nn.ModuleDict(
            {
                "dense_h_to_4h": ColumnParallelLinearWithoutBias(
                    config.hidden_size,
                    config.inner_hidden_size,
                    gather_output=False,
                    init_method=lambda x: x,
                ),
                "dense_4h_to_h": RowParallelLinearWithoutBias(
                    config.inner_hidden_size,
                    config.hidden_size,
                    input_is_parallel=True,
                    init_method=lambda x: x,
                ),
            }
        )
        # self.post_attention_layernorm = nn.LayerNorm(
        #     config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = LayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon)
        self.alpha = (2 * self.config.num_layers) ** 0.5
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size // config.tp_size
        # 务必保持变量名一致
        self.use_cache = False
        self.hidden_states = None

    def get_masks(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
        attention_mask.tril_()
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()

        return attention_mask

    def _forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[torch.Tensor] = None,  # 3
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # if attention_mask is None:
        #     # attention_mask = torch.ones_like(input_ids)
        #     attention_mask = self.get_masks(input_ids, hidden_states.device)
        if not self.training:
            self.hidden_states = hidden_states
        else:
            self.hidden_states = None
        assert (
            hidden_states.ndim == 3
        ), f"hidden_states.shape must be (B, N, H), but got {hidden_states.shape}"
        batch_size, seq_len, _ = hidden_states.shape
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        hidden_states = hidden_states.permute(1, 0, 2).contiguous()  # [N, B, H]
        hidden_states = self.input_layernorm(hidden_states)
        query_key_value = self.attention["query_key_value"](hidden_states)
        query_key_value = rearrange(    
            query_key_value, "b n (h d) -> b n h d", d=head_dim * 3
        )
        # [1, 1, 32, 384]
        query, key, value = torch.chunk(query_key_value, 3, dim=-1)

        query1, query2 = query.chunk(2, dim=-1)
        key1, key2 = key.chunk(2, dim=-1)
        cos, sin = self.attention["rotary_emb"](query1, seq_len=position_ids.max() + 1)
        _position_ids, _block_position_ids = (
            position_ids[:, 0, :].transpose(0, 1).contiguous(),
            position_ids[:, 1, :].transpose(0, 1).contiguous(),
        )
        query1, key1 = apply_rotary_pos_emb_index(query1, key1, cos, sin, _position_ids)
        query2, key2 = apply_rotary_pos_emb_index(
            query2, key2, cos, sin, _block_position_ids
        )
        query = torch.concat([query1, query2], dim=(query1.ndim - 1))
        key = torch.concat([key1, key2], dim=(key1.ndim - 1))
        # query.shape=key.shape [1, 1, 32, 128]
        
        new_layer_past = None
        if not self.training and self.use_cache:
            if past_key_values is not None:
                cache_k, cache_v = past_key_values
                # 这里是恢复 sq 作为第一个维度，转为 [sq, b, sk, hd]
                if env.pp_size > 1:
                    cache_k = rearrange(cache_k, "b sq sk hd -> sq b sk hd")
                    cache_v = rearrange(cache_v, "b sq sk hd -> sq b sk hd")
                key = torch.cat([cache_k, key], dim=0)
                value = torch.cat([cache_v, value], dim=0)
            # 这里转置的原因是 pipeline 生成时在 pipeline_engine时候会对batch划分，指定dim=2，故需要转置
            if env.pp_size > 1:
                new_layer_past = (
                    rearrange(key, "sq b sk hd -> b sq sk hd"),
                    rearrange(value, "sq b sk hd -> b sq sk hd"),
                )
            else:
                new_layer_past = (key, value)

        # seqlen, batch, num_attention_heads, hidden_size_per_attention_head
        seq_len, b, nh, hidden_size = key.shape
        query_key_layer_scaling_coeff = float(self.layer_id + 1)
        query_layer = query / (math.sqrt(hidden_size) * query_key_layer_scaling_coeff)
        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn] [1, 32, 128]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn] [24, 32, 128]
        key_layer = key.view(output_size[3], output_size[0] * output_size[1], -1)

        matmul_result = torch.zeros(
            1, 1, 1,
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=1.0,
        )
        # [32, 1, 24]
        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)
        if not (attention_mask == 0).all():
            # if auto-regressive, skip
            attention_scores.masked_fill_(attention_mask, -10000.0)
        dtype = attention_scores.dtype
        attention_scores = attention_scores.float()
        attention_scores = attention_scores * query_key_layer_scaling_coeff
        attention_probs = F.softmax(attention_scores, dim=-1)

        attention_probs = attention_probs.type(dtype)
        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value.size(1), value.size(2), query_layer.size(0), value.size(3))

        # change view [sk, b * np, hn]
        value_layer = value.view(value.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        # matmul: [b * np, sq, hn] [24,32, 128]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.attention["dense"](context_layer)
        # Residual connection.
        hidden_states = hidden_states * self.alpha + attention_output
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = hidden_states * self.alpha + \
            self.mlp["dense_4h_to_h"](F.gelu(self.mlp["dense_h_to_4h"](hidden_states)))
        hidden_states = hidden_states.permute(1, 0, -1)
        
        return hidden_states, new_layer_past

    def forward(self, inputs: dict):
        if "past_key_values" in inputs:
            all_pasts = inputs["past_key_values"]
            # all_pasts = all_pasts.permute(0, 1, 3, 2, 4, 5)
            for i in range(self.config.num_layers):
                layer_past = all_pasts[i]
                inputs[f"past_key_values_layer{i}_key"] = layer_past[0]
                inputs[f"past_key_values_layer{i}_value"] = layer_past[1]
            del inputs["past_key_values"]      
        
        past_key_values = inputs_to_kv_cache_for_layer(idx=self.layer_id,
                                                        inputs=inputs)
            
        attention_mask = inputs.get("attention_mask", None)
        if past_key_values is not None:
            attention_mask = torch.zeros(1, 1, device=inputs["hidden_states"].device).bool()
        else:
            attention_mask = self.get_masks(inputs["input_ids"], inputs["input_ids"].device)
        
        attention_mask = attention_mask.contiguous()   
        
        if self.config.checkpointing and self.training:
            inputs["hidden_states"], new_layer_past = torch.utils.checkpoint.checkpoint(
                self._forward,
                inputs["hidden_states"],
                inputs["input_ids"],
                inputs["position_ids"],
                past_key_values,
                attention_mask,
            )
        else:
            inputs["hidden_states"], new_layer_past = self._forward(
                hidden_states=inputs["hidden_states"],
                input_ids=inputs["input_ids"],
                position_ids=inputs["position_ids"],
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )
        inputs["position_ids"] = inputs["position_ids"].contiguous()
        inputs.update(kv_cache_to_inputs_for_layer(idx=self.layer_id,
                                                   new_layer_past=new_layer_past))
        
        return inputs

class ChatGLMModel(nn.Module):
    def __init__(self, config: CollieConfig) -> None:
        super().__init__()
        self.config = config
        self.word_embeddings = self._get_word_embedding_with_position_ids_cls(config)(
            self.config.vocab_size, self.config.hidden_size
        )
        self.layers = nn.Sequential(
            *[ChatGLMLayer(self.config, i) for i in range(self.config.num_layers)]
        )
        self.final_layernorm = LayerNorm(
            self.config.hidden_size, eps=self.config.layernorm_epsilon
        )

    def get_masks(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
        attention_mask.tril_()
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()
        return attention_mask
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        **kwargs,
    ):
        inputs = {"input_ids": input_ids}
        
        if input_ids == None:
            inputs["hidden_states"] = kwargs["inputs_embeds"]
        else:
            inputs.update(dict(zip(["hidden_states", "input_ids", "position_ids"], 
                                   self.word_embeddings(input_ids))))
        
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
                 
        inputs.update(kv_cache_to_inputs_for_model(past_key_values))
        all_hidden_states = ()
        for layer in self.layers:
            all_hidden_states += (inputs["hidden_states"],)
            inputs.update(layer(inputs))
        all_hidden_states += (inputs["hidden_states"],)
        inputs["hidden_states"] = self.final_layernorm(inputs["hidden_states"])

        past_key_values = inputs_to_kv_cache_for_model(self.config.num_layers, inputs)

        return BaseModelOutputWithPast(
            last_hidden_state=inputs["hidden_states"],
            hidden_states=all_hidden_states,
            past_key_values=past_key_values,
        )

    @classmethod
    def pipeline_layers(cls, config: CollieConfig):
        """
        Get layers of pipeline.

        :return: list
        """
        if isinstance(config, str):
            config = CollieConfig.from_pretrained(config)
        return [
            ("word_embeddings", TiedLayerSpec(
                "word_embeddings",
                dict_as_params(
                    input_keys="input_ids",
                    output_keys=["hidden_states", "input_ids", "position_ids"],
                ),
                cls._get_word_embedding_with_position_ids_cls(config),
                config.vocab_size,
                config.hidden_size,
            )),
            ("layers", [
                LayerSpec(ChatGLMLayer, config, i) 
                for i in range(config.num_layers)
            ]),
            ("final_layernorm", LayerSpec(
                dict_as_params(input_keys="hidden_states", output_keys="hidden_states"),
                # nn.LayerNorm,
                LayerNorm,
                config.hidden_size,
                eps=config.layernorm_epsilon,
            )),
        ]

    @staticmethod
    def _get_position_ids(
        config, input_ids: torch.Tensor, past_position_id: Optional[torch.Tensor]
    ):
        if past_position_id is not None:
            return torch.cat(
                (
                    past_position_id,
                    torch.stack(
                        (
                            past_position_id[:, 0, -1].unsqueeze(-1),
                            past_position_id[:, 1, -1].unsqueeze(-1) + 1,
                        ),
                        dim=1,
                    ),
                ),
                dim=2,
            )
        MASK, gMASK = config.mask_token_id, config.gmask_token_id
        seqs = input_ids.tolist()
        device = input_ids.device
        mask_positions, use_gmasks = [], []
        for seq in seqs:
            mask_token = gMASK if gMASK in seq else MASK
            use_gmask = mask_token == gMASK
            mask_positions.append(seq.index(mask_token))
            use_gmasks.append(use_gmask)
        batch_size, seq_length = input_ids.shape
        if use_gmasks is None:
            use_gmasks = [False] * batch_size
        context_lengths = [seq.tolist().index(config.bos_token_id) for seq in input_ids]
        if config.position_encoding_2d:
            position_ids = (
                torch.arange(seq_length, dtype=torch.long, device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )
            for i, context_length in enumerate(context_lengths):
                position_ids[i, context_length:] = mask_positions[i]
            block_position_ids = [
                torch.cat(
                    (
                        torch.zeros(context_length, dtype=torch.long, device=device),
                        torch.arange(
                            seq_length - context_length, dtype=torch.long, device=device
                        )
                        + 1,
                    )
                )
                for context_length in context_lengths
            ]
            block_position_ids = torch.stack(block_position_ids, dim=0)
            position_ids = torch.stack((position_ids, block_position_ids), dim=1)
        else:
            position_ids = (
                torch.arange(seq_length, dtype=torch.long, device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )
            for i, context_length in enumerate(context_lengths):
                if not use_gmasks[i]:
                    position_ids[i, context_length:] = mask_positions[i]
        return position_ids

    @classmethod
    def _get_word_embedding_with_position_ids_cls(cls, config):
        class WordEmbeddingWithPositionIdsAndInputIds(
            tensor_parallel.VocabParallelEmbedding
        ):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # 这个实际上是 past_position_ids
                self.past_key_values = None
                self.use_cache = True

            def forward(self, input_):
                position_ids = cls._get_position_ids(
                    config,
                    input_,
                    None if self.past_key_values is None else self.past_key_values[0],
                )
                if not self.training and self.use_cache:
                    # self.past_key_values = (self.past_key_values, self.past_key_values)
                    if self.past_key_values is not None:
                        position_ids = position_ids[:, :, self.past_key_values[0].shape[-1]:]
                    self.past_key_values = (position_ids, position_ids)
                return super().forward(input_), input_, position_ids

        return WordEmbeddingWithPositionIdsAndInputIds


class ChatGLMForCausalLM(CollieModelForCausalLM):
    def __init__(self, config: CollieConfig) -> None:
        super().__init__(config)
        self.model = ChatGLMModel(config)
        self.lm_head = ColumnParallelLinearWithoutBias(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        # GenerationMixin 需要的额外参数
        self.config.is_decoder = True
        self.main_input_name = "input_ids"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        logits = self.lm_head(output.last_hidden_state)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=None,
        )

    def get_masks(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
        attention_mask.tril_()
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()

        return attention_mask
    
    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ):
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]

            if attention_mask is not None and attention_mask.dtype == torch.bool:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((*attention_mask.shape[:3], 1))], dim=3)
                new_attention_mask = attention_mask[:, :, -1:].clone()
                new_attention_mask[..., -1] = False
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, new_attention_mask], dim=2
                )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id[:, 1, :] += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        return model_kwargs
    
    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            use_cache: bool = None,
            **kwargs
    ) -> dict:
        self.set_cache(use_cache)
        # only last token for input_ids if past is not None
        if past_key_values is not None:
            last_token = input_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None and attention_mask.dtype == torch.bool:
                attention_mask = attention_mask[:, :, -1:]
            else:
                attention_mask = None

            return {
                "input_ids": last_token,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask
            }
        else:
            if attention_mask is not None and attention_mask.dtype != torch.bool:
                logger.warning_once(f"The dtype of attention mask ({attention_mask.dtype}) is not bool")
                attention_mask = None
            if attention_mask is None:
                attention_mask = self.get_masks(
                    input_ids,
                    device=input_ids.device
                )

            return {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask
            }

    def clean_cache(self):
        self._clean_hidden_states([*self.model.layers, self.lm_head])
        # 别忘了清理 word_embeddings 里的 past_position_ids
        # self._clean_past_key_values(self.model.layers, self.word_embeddings)
        self._set_use_cache(self.model.layers, False)

    def set_cache(self, use_cache):
        self._set_use_cache(self.model.layers, use_cache)

    @classmethod
    def pipeline_layers(cls, config: CollieConfig):
        """
        Get layers of pipeline.

        :return: list
        """
        if isinstance(config, str):
            config = CollieConfig.from_pretrained(config)
        return [
            ("model", ChatGLMModel.pipeline_layers(config)),
            ("lm_head", TiedLayerSpec(
                "lm_head",
                dict_as_params(input_keys="hidden_states", output_keys="logits"),
                ColumnParallelLMHead,
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )),
        ]

    @staticmethod
    def load_parallel_state_dict(
        path: str,
        config: Union[CollieConfig, str],
        process_exclusion: bool = False,
        **kwargs,
    ):
        ...

    @staticmethod
    def load_parallel_state_dict(
        path: str,
        config: Union[CollieConfig, str],
        process_exclusion: bool = False,
        protocol: str = "file",
        format: str = "hf",
        **kwargs,
    ):
        """
        Load state_dict from ``path``.

        The format of pretrained model should be the same as that of
        `huggingface`.

        :return: state_dict. Note that the state_dict should be processed
            properly to match the current rank.
        """
        assert format in ["hf", "meta"], "Only support hf and meta format"
        if isinstance(config, str):
            config = CollieConfig.from_pretrained(config)
        io_driver = IODriver.from_protocol(protocol)
        if not io_driver.exists(path):
            raise FileNotFoundError(f"folder {path} not found.")
        state_dict = OrderedDict()
        weights = []
        parts = None
        # 如果开启了进程互斥，那么每个进程都会显示进度条，否则只显示 RANK0 的
        hide_progress = not process_exclusion and int(os.environ.get("RANK", "0")) != 0
        if dist.is_initialized() and process_exclusion:
            # 如果启动了进程互斥，则要进行 dist.get_world_size() 次循环
            rank_order = range(dist.get_world_size())
        else:
            # 不开启只进行一次循环
            rank_order = range(1)
        for rank in rank_order:
            # 如果开启了进程互斥，那么只有对应 RANK 的能进入循环；不开启进程互斥的话就都可以进
            if int(os.environ.get("RANK", "0")) == rank or not process_exclusion:
                # PP 分层的方法保存在了 os.environ["COLLIE_PP_PARTS"], 格式类似于 [0, 17, 35], 左闭右开
                if env.is_pipeline:
                    # 保存的是 json 格式
                    parts = env.pipeline_parts
                # 如果存在 .index.json 文件的话，此时不同的 pp 进程可以按需加载自己需要的权重
                # 优先加载 model.safetensors.index.json 文件中的权重
                if io_driver.exists(os.path.join(path, "model.safetensors.index.json")):
                    index_json_file_path = os.path.join(path, "model.safetensors.index.json")
                elif io_driver.exists(os.path.join(path, "pytorch_model.bin.index.json")):
                    index_json_file_path = os.path.join(path, "pytorch_model.bin.index.json")
                else:
                    index_json_file_path = None
                if (
                    index_json_file_path is not None
                    and "COLLIE_PP_PARTS" in os.environ.keys()
                ):
                    weight_map = json.loads(
                        io_driver.load(
                            index_json_file_path, mode="r"
                        )
                    )["weight_map"]
                    # layers 表示自己需要的层
                    layers = list(
                        range(
                            parts[int(os.environ["COLLIE_PP_RANK"])],
                            parts[int(os.environ["COLLIE_PP_RANK"]) + 1],
                        )
                    )
                    # 筛选出形似 model.layers.0 这样的层。包含两个条件：1. 有数字的层；2. 数字加一要在 layers 里面（因为最开始还有个 embedding 占一层）
                    weights.extend(
                        [
                            value
                            for key, value in weight_map.items()
                            if len(key.split(".")) > 2
                            and key.split(".")[2].isdigit()
                            and (int(key.split(".")[2]) + 1) in layers
                        ]
                    )
                    # 去重
                    weights = list(set(weights))
                    # 继续筛选，如果有 0 层，那么就要加载 embedding；如果有最后一层，那么就要加载 lm_head；如果有倒数第二层，那么就要加载 norm
                    if 0 in layers:
                        weights.append(weight_map["transformer.word_embeddings.weight"])
                    if max(parts) - 1 in layers:
                        weights.append(weight_map["lm_head.weight"])
                    if max(parts) - 2 in layers:
                        weights.append(weight_map["transformer.final_layernorm.weight"])
                        weights.append(weight_map["transformer.final_layernorm.bias"])
                else:
                    # 如果没有 .index.json 文件的话，那么就加载所有的权重
                    # 优先加载 safetensors 存储的权重
                    weights = [
                        weight
                        for weight in io_driver.list(path)
                        if weight.endswith(".safetensors")
                    ]
                    if len(weights) == 0:
                        weights = [
                            weight
                            for weight in io_driver.list(path)
                            if weight.endswith(".bin")
                        ]
                with progress(
                    weights,
                    desc="Loading state dict",
                    total=len(weights),
                    disable=hide_progress,
                ) as pbar:
                    for weight in pbar:
                        part_state_dict = io_driver.load(
                            os.path.join(path, weight), mode="rb"
                        )
                        for key in list(part_state_dict.keys()):
                            part_state_dict[
                                key.replace("transformer.", "model.")
                            ] = part_state_dict.pop(key)
                        state_dict.update(part_state_dict)
                        del part_state_dict
                if parts is not None:
                    # 这一步是 pp 的复筛
                    layers = list(
                        range(
                            parts[int(os.environ["COLLIE_PP_RANK"])],
                            parts[int(os.environ["COLLIE_PP_RANK"]) + 1],
                        )
                    )
                    for key in list(state_dict.keys()):
                        if key.startswith("layers"):
                            layer = int(key.split(".")[1])
                            if layer + 1 not in layers:
                                # 形似 model.layers.0 这样的层，筛选掉数字加一不在 layers 里面得
                                state_dict.pop(key)
                        if key.endswith("word_embeddings.weight"):
                            if 0 not in layers:
                                state_dict.pop(key)
                        if key == "final_layernorm.weight":
                            if max(parts) - 2 not in layers:
                                state_dict.pop(key)
                        if key == "final_layernorm.bias":
                            if max(parts) - 2 not in layers:
                                state_dict.pop(key)
                        if key.endswith("lm_head.weight"):
                            if max(parts) - 1 not in layers:
                                state_dict.pop(key)
                # 根据用户配置的新的 tp size 进行分割
                for key in list(state_dict.keys()):
                    filte_list = [
                        "query_key_value.weight",
                        "query_key_value.bias",
                        "dense_h_to_4h.weight",
                        "dense_h_to_4h.bias",
                        "word_embeddings.weight",
                        "lm_head.weight",
                    ]
                    need_split = any([key.endswith(filte) for filte in filte_list])
                    # if env.pp_size > 1:
                    #     # embedding 层和 lm_head 都需要切
                    #     need_split = (
                    #         need_split or int(key.split(".")[0]) == max(parts) - 1
                    #     )
                    #     need_split = need_split or int(key.split(".")[0]) == min(parts)
                    if need_split:
                        tensor = (
                            list(torch.chunk(state_dict[key], config.tp_size, dim=0))[
                                int(os.environ.get("COLLIE_TP_RANK", "0"))
                            ]
                            .detach()
                            .clone()
                        )
                        del state_dict[key]
                        if process_exclusion:
                            # CPU 内存回收（速度很慢）
                            gc.collect()
                        state_dict[key] = tensor
                    elif key.endswith("dense.weight") or key.endswith(
                        "dense_4h_to_h.weight"
                    ):
                        tensor = (
                            list(torch.chunk(state_dict[key], config.tp_size, dim=1))[
                                int(os.environ.get("COLLIE_TP_RANK", "0"))
                            ]
                            .detach()
                            .clone()
                        )
                        del state_dict[key]
                        if process_exclusion:
                            # CPU 内存回收（速度很慢）
                            gc.collect()
                        state_dict[key] = tensor
            if dist.is_initialized() and process_exclusion:
                # 如果选择了进程互斥，那么本次循环中不需要加载权重的进程需等待
                dist.barrier()
        return state_dict

    @staticmethod
    def save_parallel_state_dict(
        state_dict: dict,
        path: str,
        config: CollieConfig,
        process_exclusion: bool = False,
        **kwargs,
    ):
        ...

    @staticmethod
    def save_parallel_state_dict(
        state_dict: dict,
        path: str,
        config: CollieConfig,
        process_exclusion: bool = False,
        protocol: str = "file",
    ):
        """
        Save state_dict to ``path``.

        The format of saved state dict should be the same as that of
        `huggingface`.
        """
        io_driver = IODriver.from_protocol(protocol)

        def reshape_wq_wk(w: torch.Tensor):
            return (
                w.view(
                    config.num_attention_heads,
                    config.hidden_size // config.num_attention_heads // 2,
                    2,
                    config.hidden_size,
                )
                .transpose(1, 2)
                .reshape(config.hidden_size, config.hidden_size)
            )

        # gather to tp rank 0
        if env.is_pipeline:
            layers = env.pipeline_layers_idx
            parts = env.pipeline_parts
            for key in list(state_dict.keys()):
                if key == "tied_modules.word_embeddings.word_embeddings.weight":
                    if 0 in layers:
                        state_dict[
                            "transformer.word_embeddings.weight"
                        ] = state_dict.pop(key)
                    elif max(layers) - 1 in layers:
                        state_dict["lm_head.weight"] = state_dict.pop(key)
                else:
                    layer = int(key.split(".")[0])
                    if layer == max(parts) - 2:
                        state_dict[
                            key.replace(f"{layer}.", "transformer.final_layernorm.")
                        ] = state_dict.pop(key)
                    else:
                        state_dict[
                            key.replace(f"{layer}.", f"transformer.layers.{layer - 1}.")
                        ] = state_dict.pop(key)
        if dist.is_initialized() and process_exclusion:
            # 如果启动了进程互斥，则要进行 pp_size 次循环
            rank_order = range(config.pp_size)
        else:
            # 不开启只进行一次循环
            rank_order = range(1)
        dst = parallel_state.get_tensor_model_parallel_src_rank()
        with progress(
            rank_order,
            desc="Saving model",
            disable=int(os.environ.get("RANK", "0")) != 0,
        ) as pbar:
            for rank in pbar:
                if env.dp_rank == 0 and (env.pp_rank == rank or not process_exclusion):
                    if config.tp_size > 1:
                        for key in sorted(list(state_dict.keys())):
                            tensor_list = None
                            if env.tp_rank == 0:
                                tensor_list = [
                                    torch.zeros_like(state_dict[key])
                                    .to(state_dict[key].dtype)
                                    .cuda()
                                    for _ in range(config.tp_size)
                                ]
                            dist.gather(
                                state_dict[key].cuda(),
                                dst=dst,
                                gather_list=tensor_list,
                                group=env.tp_group,
                            )
                            if env.tp_rank == 0:
                                filte_list = [
                                    "query_key_value.weight",
                                    "query_key_value.bias",
                                    "dense_h_to_4h.weight",
                                    "dense_h_to_4h.bias",
                                    "word_embeddings.weight",
                                    "lm_head.weight",
                                ]
                                need_split = any(
                                    [key.endswith(filte) for filte in filte_list]
                                )
                                if env.pp_size > 1:
                                    # embedding 层和 lm_head 都需要切
                                    need_split = (
                                        need_split
                                        or int(key.split(".")[0]) == max(parts) - 1
                                    )
                                    need_split = need_split or int(
                                        key.split(".")[0]
                                    ) == min(parts)
                                if need_split:
                                    state_dict[key] = concat_tensor(tensor_list, dim=0)
                                    if process_exclusion:
                                        # CPU 内存回收（速度很慢）
                                        gc.collect()
                                elif key.endswith("dense.weight") or key.endswith(
                                    "dense_4h_to_4.weight.weight"
                                ):
                                    state_dict[key] = concat_tensor(tensor_list, dim=1)
                                    if process_exclusion:
                                        # CPU 内存回收（速度很慢）
                                        gc.collect()
                    if env.tp_rank == 0:
                        # Save gathered weights
                        if env.is_pipeline:
                            ckpt_name = f"pytorch_model-{env.pp_rank+1:05d}-of-{config.pp_size:05d}.bin"
                            total_size = 0
                            weight_map = {}
                            for name, weight in state_dict.items():
                                weight_size = weight.numel() * dtype_byte_size(
                                    weight.dtype
                                )
                                weight_map[name] = ckpt_name
                                total_size += weight_size
                            index_dict = dict(
                                total_size=total_size, weight_map=weight_map
                            )
                            index_dicts = [None for _ in range(env.pp_size)]
                            dist.gather_object(
                                index_dict, index_dicts if env.pp_rank == 0 else None, group=env.pp_group
                            )
                            if env.pp_rank == 0:
                                total_size = 0
                                weight_map = {}
                                for _index_dict in index_dicts:
                                    total_size += _index_dict["total_size"]
                                    weight_map.update(_index_dict["weight_map"])
                                merged_dict = {
                                    "metadata": {"total_size": total_size},
                                    "weight_map": weight_map,
                                }
                                io_driver.save(
                                    json.dumps(merged_dict, indent=2, sort_keys=True)
                                    + "\n",
                                    os.path.join(path, "pytorch_model.bin.index.json"),
                                )
                        else:
                            ckpt_name = f"pytorch_model.bin"
                        ckpt_path = os.path.join(path, ckpt_name)
                        io_driver.save(state_dict, ckpt_path)
                if dist.is_initialized() and process_exclusion:
                    dist.barrier()
        if env.rank == 0:
            config.save_pretrained(path, protocol=protocol)
        dist.barrier()
