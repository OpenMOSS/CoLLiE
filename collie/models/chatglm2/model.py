from typing import List, Dict, Any
import math
import os
import gc
import json

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.checkpoint
from torch.nn import LayerNorm
from einops import rearrange
from megatron.core import tensor_parallel
from megatron.core import parallel_state
from deepspeed.pipe import LayerSpec
try:
    from flash_attn.flash_attention import FlashAttention
except ModuleNotFoundError:
    FlashAttention = None

from collie.log.logger import logger
from collie.config import CollieConfig
from collie.models.base import CollieModelForCausalLM
from collie.driver.io import IODriver
from collie.module import ColumnParallelLinearWithoutBias, RowParallelLinearWithoutBias, ColumnParallelLMHead
from collie.utils import progress, env, dict_as_params

from typing import Any, Union, Optional
from collections import OrderedDict
from transformers.modeling_utils import dtype_byte_size
from transformers.modeling_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
 

def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.
        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )

@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)


class CoreAttention(torch.nn.Module):
    def __init__(self, config: CollieConfig, layer_number):
        super(CoreAttention, self).__init__()

        self.config = config
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        pytorch_major_version = int(torch.__version__.split('.')[0])
        if pytorch_major_version >= 2:
            query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
            if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 is_causal=True)
            else:
                if attention_mask is not None:
                    attention_mask = ~attention_mask
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 attention_mask)
            context_layer = context_layer.permute(2, 0, 1, 3)
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.reshape(*new_context_layer_shape)
        else:
            # Raw attention scores

            # [b, np, sq, sk]
            output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

            # [sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

            # preallocting input tensor: [b * np, sq, sk]
            matmul_input_buffer = torch.empty(
                output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype,
                device=query_layer.device
            )
            # Raw attention scores. [b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )
            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.view(*output_size)

            # ===========================
            # Attention probs and dropout
            # ===========================
            # attention scores and attention mask [b, np, sq, sk]
            if self.attention_softmax_in_fp32:
                attention_scores = attention_scores.float()
            if self.coeff is not None:
                attention_scores = attention_scores * self.coeff
            if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
                attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                            device=attention_scores.device, dtype=torch.bool)
                attention_mask.tril_()
                attention_mask = ~attention_mask

            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = attention_probs.type_as(value_layer)
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.attention_dropout(attention_probs)
            # =========================
            # Context layer. [sq, b, hp]
            # =========================

            # value_layer -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))
            # change view [sk, b * np, hn]
            value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
            # change view [b * np, sq, sk]
            attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
            # matmul: [b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
            # change view [b, np, sq, hn]
            context_layer = context_layer.view(*output_size)
            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition // self.config.tp_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class ChatGLM2Layer(nn.Module):
    def __init__(self, config: CollieConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = max(1, layer_id)
        
        self.qkv_hidden_size = 3 * config.hidden_size
        
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection
        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = config.hidden_size // config.num_attention_heads
        
        #  Multi-Query Attention 技术 更高效的推理速度和更低的显存占用
        if config.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    config.hidden_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.core_attention = CoreAttention(config, self.layer_id)
        self.attention = nn.ModuleDict({
            "query_layer": ColumnParallelLinearWithoutBias(
                config.hidden_size,
                config.hidden_size,
                gather_output=False,
                init_method=lambda x: x,
                bias=config.add_bias_linear or config.add_qkv_bias,
            ),
            "key_layer": ColumnParallelLinearWithoutBias(
                config.hidden_size,
                self.hidden_size_per_attention_head * config.multi_query_group_num,
                gather_output=False,
                init_method=lambda x: x,
                bias=config.add_bias_linear or config.add_qkv_bias
            ),
            "value_layer": ColumnParallelLinearWithoutBias(
                config.hidden_size,
                self.hidden_size_per_attention_head * config.multi_query_group_num,
                gather_output=False,
                init_method=lambda x: x,
                bias=config.add_bias_linear or config.add_qkv_bias
            ),
            "dense": RowParallelLinearWithoutBias(
                config.hidden_size,
                config.hidden_size,
                input_is_parallel=True,
                init_method=lambda x: x,
                bias=config.add_bias_linear
            )
        })
        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.mlp = nn.ModuleDict({
            "dense_h_to_4h_up_proj": ColumnParallelLinearWithoutBias(
                config.hidden_size,
                config.ffn_hidden_size,
                gather_output=False,
                init_method=lambda x: x,
                bias=config.add_bias_linear
            ),
            "dense_h_to_4h_down_proj": ColumnParallelLinearWithoutBias(
                config.hidden_size,
                config.ffn_hidden_size,
                gather_output=False,
                init_method=lambda x: x,
                bias=config.add_bias_linear
            ),
            "dense_4h_to_h": RowParallelLinearWithoutBias(
                config.ffn_hidden_size,
                config.hidden_size,
                input_is_parallel=True,
                init_method=lambda x: x,
                bias=config.add_bias_linear
            )
        })
        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu
        self.multi_query_attention = config.multi_query_attention
        self.num_attention_heads_per_partition = config.num_attention_heads
        self.hidden_dropout = config.hidden_dropout
        
        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon,
                                             dtype=config.torch_dtype)
        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon,
                                                      dtype=config.torch_dtype)
        # 务必保持变量名一致
        self.use_cache = False
        self.past_key_values = None
        self.hidden_states = None
    
    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length, _ = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        if past_length:
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                        device=input_ids.device), full_attention_mask), dim=-1)
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask
    
    def _forward(self, hidden_states: torch.Tensor, attention_mask, rotary_pos_emb):
        # hidden_states: [s, b, h]
        assert hidden_states.ndim == 3, f"hidden_states.shape must be (S, B, H), but got {hidden_states.shape}"

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        if self.multi_query_attention:
            query_layer = self.attention["query_layer"](layernorm_output)
            key_layer = self.attention["key_layer"](layernorm_output)
            value_layer = self.attention["value_layer"](layernorm_output)
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            query_layer, key_layer, value_layer = rearrange(query_layer, "b n (h d) -> b n h d", d=head_dim), \
            rearrange(key_layer, "b n (h d) -> b n h d", d=head_dim), \
            rearrange(value_layer, "b n (h d) -> b n h d", d=head_dim)
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
        
           
        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            rotary_pos_emb = rotary_pos_emb.to(query_layer.device)
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)
        
        # adjust key and value for inference
        if not self.training and self.use_cache:
            if self.past_key_values is not None:
                cache_k, cache_v = self.past_key_values
                # query_layer = torch.cat([cache_k, query_layer], dim=0)
                key_layer = torch.cat((cache_k, key_layer), dim=0)
                value_layer = torch.cat((cache_v, value_layer), dim=0)
            self.past_key_values = (key_layer, value_layer)
        else:
            self.past_key_values = None

        # Multi query attention
        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:2] + (self.num_attention_heads_per_partition// self.config.tp_size, self.hidden_size_per_attention_head )
            )
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2] + (self.num_attention_heads_per_partition// self.config.tp_size, self.hidden_size_per_attention_head)
            )
        # ==================================
        # core attention computation
        # ==================================
        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
        # =================
        # Output. [sq, b, h]
        # =================
        # context_layer = context_layer[:, start_pos:, :]
        attention_output = self.attention["dense"](context_layer)
        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        up_proj = self.mlp["dense_h_to_4h_up_proj"](layernorm_output)
        down_proj = self.mlp["dense_h_to_4h_down_proj"](layernorm_output)
        activation_proj = F.silu(up_proj) * down_proj
        
        # intermediate_parallel = self.activation_func(intermediate_parallel)
        mlp_output = self.mlp["dense_4h_to_h"](activation_proj)
        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        return output
    
    def forward(self, inputs: dict):
        inputs["rotary_pos_emb"] = inputs["rotary_pos_emb"].to(inputs["hidden_states"].device)
        # 在第一层输入改变 attention_mask
        # if self.layer_id == 1:
        full_attention_mask = inputs.get("full_attention_mask", None)
        # if self.layer_id == 1:
        if full_attention_mask is None:
            if (inputs['attention_mask'] is not None and not inputs['attention_mask'].all()) or (self.past_key_values and inputs["hidden_states"].shape[1] != 1):
                full_attention_mask = self.get_masks(inputs["hidden_states"], self.past_key_values, padding_mask=inputs['attention_mask'])
                    # inputs['full_attention_mask'] = full_attention_mask
                    # print(f"[Debug] ChatGLM2Lyaer: layer_id: {self.layer_id} full_attention_mask: {full_attention_mask.shape}")
                # else:
                #     inputs.pop("attention_mask")
            # else:
            #     attention_mask = inputs["full_attention_mask"]
                # inputs.pop("full_attention_mask")
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        inputs["hidden_states"] = inputs["hidden_states"].transpose(0, 1).contiguous()
        if self.config.checkpointing and self.training:
            inputs["hidden_states"] = torch.utils.checkpoint.checkpoint(
                self._forward,
                inputs["hidden_states"],
                full_attention_mask,
                inputs["rotary_pos_emb"], 
            )
        else:
            inputs["hidden_states"] = self._forward(
                inputs["hidden_states"],
                full_attention_mask,
                inputs["rotary_pos_emb"], 
            )
        # 将输入维度转为 [b, s, h]
        inputs["hidden_states"] = inputs["hidden_states"].transpose(0, 1).contiguous()
        return inputs


class ChatGLM2ForCausalLM(CollieModelForCausalLM):
    def __init__(self, config: CollieConfig) -> None:
        super().__init__(config)
        self.word_embeddings = self._get_word_embedding_with_position_ids_cls(config)(
            self.config.padded_vocab_size,
            self.config.hidden_size
        )
        
        # Rotary positional embeddings
        self.seq_length = config.seq_length
        # self.config.checkpointing = True
        self.layers = nn.Sequential(
            *[ChatGLM2Layer(self.collie_config, i+1) for i in range(self.config.num_layers)])
        self.final_layernorm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.layernorm_epsilon
        )
        self.lm_head = ColumnParallelLinearWithoutBias(
            self.config.hidden_size,
            self.config.padded_vocab_size,
            bias=False
        )
        # GenerationMixin 需要的额外参数
        self.config.is_decoder=True
        self.main_input_name = "input_ids"

    def forward(self, input_ids: torch.Tensor, **kwargs):
        _, seq_length = input_ids.shape
        attention_mask = kwargs.get("attention_mask", None)
        full_attention_mask = kwargs.get("full_attention_mask", None)
        
        past_key_values=self._get_past_key_values(self.layers)
        if past_key_values is not None and not self.training:
            input_ids = input_ids[:, -1:]
        assert input_ids.ndim == 2, f"input_ids.shape must be (B, N), but got {input_ids.shape}"
        
        inputs = dict(zip(["hidden_states", "rotary_pos_emb"], self.word_embeddings(input_ids)))
        
        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        
        if full_attention_mask is not None:
            inputs["attention_mask"] = full_attention_mask

        all_hidden_states = ()
        for layer in self.layers:
            all_hidden_states += (inputs["hidden_states"],)
            inputs = layer(inputs)
        inputs["hidden_states"] = self.final_layernorm(inputs["hidden_states"])
        logits = self.lm_head(inputs["hidden_states"])

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=self._get_past_key_values(self.layers),
            hidden_states=all_hidden_states,
            attentions=None
        )
    
    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        if past_length:
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                        device=input_ids.device), full_attention_mask), dim=-1)
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask
    
    @staticmethod
    def _get_rotary_embedding(config, position_ids):
        seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        # rotary_dim = rotary_dim // config.tp_size
        rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, original_impl=config.original_rope,
                                              dtype=config.torch_dtype)
        rotary_pos_emb = rotary_pos_emb(seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        
        return rotary_pos_emb
    
    @staticmethod
    def _get_position_ids(input_ids: torch.Tensor, past_position_id):  
        if past_position_id is not None:
            pre_seq_len = past_position_id[0][-1] + 1
        else:
            pre_seq_len = 0
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(pre_seq_len, seq_length+pre_seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        return position_ids

    def _update_model_kwargs_for_generation(
            self,
            outputs,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        model_kwargs["is_first_forward"] = False
        return model_kwargs
    
    def prepare_inputs_for_generation(self,
                                      input_ids: torch.Tensor,
                                      past_key_values: Optional[list] = None,
                                      attention_mask: Optional[torch.Tensor] = None,
                                      **kwargs):
        self._set_use_cache(self.layers, kwargs.get("use_cache", self.generation_config.use_cache))
        # only last token for input_ids if past is not None
        if past_key_values is None:
            self._clean_past_key_values(self.layers)
        else:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            self._set_past_key_values(self.layers, past_key_values)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def clean(self):
        self._clean_hidden_states([*self.layers, self.lm_head])
        # 别忘了清理 word_embeddings 里的 past_position_ids
        self._clean_past_key_values(self.layers)
        self._set_use_cache(self.layers, False)
        
    @classmethod
    def _get_word_embedding_with_position_ids_cls(cls, config):
        class WordEmbeddingWithPositionIdsAndInputIds(tensor_parallel.VocabParallelEmbedding):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # 这个实际上是 past_position_ids
                self.past_key_values = None
                self.use_cache = True
                
            def forward(self, input_):
                position_ids = cls._get_position_ids(input_, None if self.past_key_values is None else self.past_key_values[0])
                rotary_pos_emb = cls._get_rotary_embedding(config, position_ids)
                if not self.training and self.use_cache:
                    # self.past_key_values = (self.past_key_values, self.past_key_values)
                    self.past_key_values = (position_ids, position_ids)
                return super().forward(input_), rotary_pos_emb
        return WordEmbeddingWithPositionIdsAndInputIds
    
    @classmethod
    def pipeline_layers(cls, config: CollieConfig):
        """
        Get layers of pipeline.

        :return: list
        """
        if isinstance(config, str):
            config = CollieConfig.from_pretrained(config)
        layers = [
            LayerSpec(
            dict_as_params(input_keys="input_ids", output_keys=["hidden_states", "rotary_pos_emb"]),
            cls._get_word_embedding_with_position_ids_cls(config),
            config.padded_vocab_size,
            config.hidden_size),
            *[LayerSpec(ChatGLM2Layer, config, i+1)
              for i in range(config.num_layers)],
            LayerSpec(dict_as_params(input_keys="hidden_states", output_keys="hidden_states"),
                RMSNorm,
                config.hidden_size,
                eps=config.layernorm_epsilon),
            LayerSpec(
            dict_as_params(input_keys="hidden_states", output_keys="logits"),
            ColumnParallelLMHead,
            config.hidden_size,
            config.padded_vocab_size,
            bias=False)
        ]
        return layers

    @staticmethod
    def load_parallel_state_dict(path: str, config: Union[CollieConfig, str],
                                 process_exclusion: bool = False, **kwargs):...
    @staticmethod
    def load_parallel_state_dict(path: str,
                                 config: Union[CollieConfig, str],
                                 process_exclusion: bool = False,
                                 protocol: str = 'file',
                                 format: str = 'hf', **kwargs):
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
                # 如果存在 pytorch_model.bin.index.json 文件的话，此时不同的 pp 进程可以按需加载自己需要的权重
                if io_driver.exists(os.path.join(path, "pytorch_model.bin.index.json")) and "COLLIE_PP_PARTS" in os.environ.keys():
                    weight_map = json.loads(io_driver.load(os.path.join(path, "pytorch_model.bin.index.json"), mode="r"))["weight_map"]
                    # layers 表示自己需要的层
                    layers = list(range(parts[int(os.environ["COLLIE_PP_RANK"])], parts[int(os.environ["COLLIE_PP_RANK"]) + 1]))
                    # 筛选出形似 model.layers.0 这样的层。包含两个条件：1. 有数字的层；2. 数字加一要在 layers 里面（因为最开始还有个 embedding 占一层）
                    weights.extend([value for key, value in weight_map.items() \
                        if len(key.split(".")) > 3 \
                            and key.split(".")[3].isdigit() \
                                and (int(key.split(".")[3]) + 1) in layers])
                    # 去重
                    weights = list(set(weights))
                    # 继续筛选，如果有 0 层，那么就要加载 embedding；如果有最后一层，那么就要加载 lm_head；如果有倒数第二层，那么就要加载 norm
                    if 0 in layers:
                        weights.append(weight_map["transformer.embedding.word_embeddings.weight"])
                    if max(parts) - 1 in layers:
                        weights.append(weight_map["transformer.output_layer.weight"])
                    if max(parts) - 2 in layers:
                        weights.append(weight_map["transformer.encoder.final_layernorm.weight"])
                        # weights.append(weight_map["transformer.final_layernorm.bias"])
                else:
                    # 如果没有 pytorch_model.bin.index.json 文件的话，那么就加载所有的权重
                    weights = [weight for weight in io_driver.list(path) if weight.endswith(".bin")]
                with progress(weights, desc="Loading state dict", total=len(weights), disable=hide_progress) as pbar:
                    for weight in pbar:
                        part_state_dict = io_driver.load(os.path.join(path, weight), mode="rb")
                        for key in list(part_state_dict.keys()):
                            if "encoder" in key:
                                key_weights = part_state_dict.pop(key)
                                # 这里需要手动将 qkv 的 weight 矩阵拆解，以便适应于 tp 的情况
                                if 'self_attention.query_key_value.weight' in key:
                                    # 其参数为 [4608, 4096]  ，故对第一维度进行拆解
                                    (query_layer, key_layer, value_layer) = key_weights.split(
                                        [
                                            config.hidden_size,
                                            config.kv_channels * config.multi_query_group_num,
                                            config.kv_channels * config.multi_query_group_num
                                        ],
                                        dim=0,
                                    )
                                    key = key.replace("transformer.encoder.", "").replace("self_attention", "attention")
                                    query_layer_name = key.replace("query_key_value", "query_layer")
                                    key_layer_name = key.replace("query_key_value", "key_layer")
                                    value_layer_name = key.replace("query_key_value", "value_layer")
                                    part_state_dict[query_layer_name] = query_layer
                                    part_state_dict[key_layer_name] = key_layer
                                    part_state_dict[value_layer_name] = value_layer
                                elif 'self_attention.query_key_value.bias' in key:
                                    # 其参数为[4608]  ，故对第一维度进行拆解 
                                    (query_layer_bias, key_layer_bias, value_layer_bias) = key_weights.split(
                                        [
                                            config.hidden_size,
                                            config.kv_channels * config.multi_query_group_num,
                                            config.kv_channels * config.multi_query_group_num
                                        ],
                                        dim=0,
                                    )
                                    key = key.replace("transformer.encoder.", "").replace("self_attention", "attention")
                                    query_layer_name = key.replace("query_key_value", "query_layer")
                                    key_layer_name = key.replace("query_key_value", "key_layer")
                                    value_layer_name = key.replace("query_key_value", "value_layer")
                                    part_state_dict[query_layer_name] = query_layer_bias
                                    part_state_dict[key_layer_name] = key_layer_bias
                                    part_state_dict[value_layer_name] = value_layer_bias
                                # 这里手动将 mlp 的 dense_h_to_4h.weight 解开
                                elif 'mlp.dense_h_to_4h.weight' in key:
                                    (dense_h_to_4h_up_proj, dense_h_to_4h_down_proj) = key_weights.split([
                                        config.ffn_hidden_size,
                                        config.ffn_hidden_size
                                    ], dim=0)
                                    key = key.replace("transformer.encoder.", "")
                                    dense_h_to_4h_up_proj_name = key.replace("dense_h_to_4h", "dense_h_to_4h_up_proj")
                                    dense_h_to_4h_down_proj_name = key.replace("dense_h_to_4h", "dense_h_to_4h_down_proj")
                                    part_state_dict[dense_h_to_4h_down_proj_name] = dense_h_to_4h_down_proj
                                    part_state_dict[dense_h_to_4h_up_proj_name] = dense_h_to_4h_up_proj
                                else:
                                    part_state_dict[key.replace("transformer.encoder.", "").replace("self_attention", "attention")] = key_weights
                            else:
                                part_state_dict[key.replace("transformer.", "").replace("embedding.", "")] = part_state_dict.pop(key)
                            
                            
                        state_dict.update(part_state_dict)
                        del part_state_dict
                if parts is not None:
                    # 这一步是 pp 的复筛
                    layers = list(range(parts[int(os.environ["COLLIE_PP_RANK"])], parts[int(os.environ["COLLIE_PP_RANK"]) + 1]))
                    for key in list(state_dict.keys()):
                        if key.startswith("layers"):
                            layer = int(key.split(".")[1])
                            if layer + 1 in layers:
                                state_dict[key.replace(f"layers.{layer}", f"{layer + 1}").replace("self_attention", "attention")] = state_dict.pop(key)
                            else:
                                # 形似 model.layers.0 这样的层，筛选掉数字加一不在 layers 里面得
                                state_dict.pop(key)
                        if key.endswith("word_embeddings.weight"):
                            if 0 in layers:
                                state_dict["0.weight"] = state_dict.pop(key)
                            else:
                                state_dict.pop(key)
                        if key == "final_layernorm.weight":
                            if max(parts) - 2 in layers:
                                state_dict[f"{max(parts) - 2}.weight"] = state_dict.pop(key)
                            else:
                                state_dict.pop(key)
                        if key.endswith("output_layer.weight"):
                            if max(parts) - 1 in layers:
                                state_dict[f"{max(parts) - 1}.weight"] = state_dict.pop(key)
                            else:
                                state_dict.pop(key)
                        if key.endswith("rotary_pos_emb.inv_freq"):
                            state_dict.pop(key)
                else:
                    # 只有tp 时候， 也需要替换某些层
                    for key in list(state_dict.keys()):
                        if key.endswith("output_layer.weight"):
                            state_dict[f"lm_head.weight"] = state_dict.pop(key)
                        if key.endswith("rotary_pos_emb.inv_freq"):
                            state_dict.pop(key)
                
                need_column_list = ["query_layer.weight", "query_layer.bias", "key_layer.weight", "key_layer.bias", "value_layer.weight", "value_layer.bias", "word_embeddings.weight",
                                    "lm_head.weight", "dense_h_to_4h_up_proj.weight", "dense_h_to_4h_down_proj.weight"]
                    
                need_row_list = ["dense.weight", "dense_4h_to_h.weight"]
                # 根据用户配置的新的 tp size 进行分割
                for key in list(state_dict.keys()):
                    need_column_split = any([key.endswith(need_key) for need_key in need_column_list])
                    need_row_split = any([key.endswith(need_key) for need_key in need_row_list])
                    if env.pp_size > 1:
                         # embedding 层和 lm_head 都需要切
                         need_column_split = need_column_split or int(key.split(".")[0]) == max(parts) - 1 or int(key.split(".")[0]) == min(parts)
                    
                    if need_column_split:
                        tensor = list(torch.chunk(state_dict[key], config.tp_size, dim=0))[int(os.environ.get("COLLIE_TP_RANK", "0"))].detach().clone()
                        del state_dict[key]
                        if process_exclusion:
                            # CPU 内存回收（速度很慢）
                            gc.collect()
                        state_dict[key] = tensor
                    elif need_row_split:
                        tensor = list(torch.chunk(state_dict[key], config.tp_size, dim=1))[int(os.environ.get("COLLIE_TP_RANK", "0"))].detach().clone()
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
    def save_parallel_state_dict(state_dict: dict, path: str,
                                 config: CollieConfig,
                                 process_exclusion: bool = False, **kwargs):...
    @staticmethod
    def save_parallel_state_dict(state_dict: dict,
                                 path: str,
                                 config: CollieConfig,
                                 process_exclusion: bool = False,
                                 protocol: str = 'file'):
        """
        Save state_dict to ``path``.

        The format of saved state dict should be the same as that of
        `huggingface`.
        """
        io_driver = IODriver.from_protocol(protocol)
        def reshape_wq_wk(w: torch.Tensor):
            return w.view(config.num_attention_heads,
                          config.hidden_size // config.num_attention_heads // 2,
                          2,
                          config.hidden_size).transpose(1, 2).reshape(config.hidden_size,
                                                                    config.hidden_size)
        # gather to tp rank 0
        if env.is_pipeline:
            layers = env.pipeline_layers_idx
            parts = env.pipeline_parts
            for key in list(state_dict.keys()):
                if 0 in layers:
                    state_dict["transformer.word_embeddings.weight"] = state_dict.pop(key)
                elif max(layers) - 1 in layers:
                    state_dict["lm_head.weight"] = state_dict.pop(key)
                else:
                    layer = int(key.split(".")[0])
                    if layer == max(parts) - 2:
                        state_dict[key.replace(f"{layer}.", "transformer.final_layernorm.")] = state_dict.pop(key)
                    else:
                        state_dict[key.replace(f"{layer}.", f"transformer.layers.{layer - 1}.")] = state_dict.pop(key)
        if dist.is_initialized() and process_exclusion:
            # 如果启动了进程互斥，则要进行 pp_size 次循环
            rank_order = range(config.pp_size)
        else:
            # 不开启只进行一次循环
            rank_order = range(1)
        dst = parallel_state.get_tensor_model_parallel_src_rank()
        with progress(rank_order, desc="Saving model", disable=int(os.environ.get("RANK", "0")) != 0) as pbar:
            for rank in pbar:
                if env.dp_rank == 0 \
                    and (env.pp_rank == rank
                         or not process_exclusion):
                    for key in sorted(list(state_dict.keys())):
                        device = state_dict[key].device
                        tensor_list = None
                        if env.tp_rank == 0:
                            tensor_list = [torch.zeros_like(state_dict[key]).to(state_dict[key].dtype).cuda() for _ in range(config.tp_size)]
                        dist.gather(state_dict[key].cuda(), dst=dst, gather_list=tensor_list, group=env.tp_group)
                        if env.tp_rank == 0:
                            need_column_list = ["query_layer.weight", "query_layer.bias", "key_layer.weight", "key_layer.bias", "value_layer.weight", "value_layer.bias", "word_embeddings.weight",
                                    "lm_head.weight", "dense_h_to_4h_up_proj.weight", "dense_h_to_4h_down_proj.weight"]
                            need_row_list = ["dense.weight", "dense_4h_to_h.weight"]
                            need_column_split = any([key.endswith(need_key) for need_key in need_column_list])
                            need_row_split = any([key.endswith(need_key) for need_key in need_row_list])
                            if env.pp_size > 1:
                                # embedding 层和 lm_head 都需要切
                                try:
                                    need_column_split = need_column_split or int(key.split(".")[0]) == max(parts) - 1 or int(key.split(".")[0]) == min(parts)
                                except:
                                    pass
                            if need_column_split:
                                state_dict[key] = torch.cat(tensor_list, dim=0).detach().clone().to(device)
                                del tensor_list
                                if process_exclusion:
                                    # CPU 内存回收（速度很慢）
                                    gc.collect()
                            elif need_row_split:
                                state_dict[key] = torch.cat(tensor_list, dim=1).detach().clone().to(device)
                                del tensor_list
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
                                weight_size = weight.numel() * dtype_byte_size(weight.dtype)
                                weight_map[name] = ckpt_name
                                total_size += weight_size
                            index_dict = dict(total_size=total_size, weight_map=weight_map)
                            tmp_index_file = os.path.join(path, "_tmp_index_{}.json")
                            io_driver.save(
                                json.dumps(index_dict), tmp_index_file.format(env.pp_rank)
                            )
                        else:
                            ckpt_name = f"pytorch_model.bin"
                        ckpt_path = os.path.join(path, ckpt_name)
                        io_driver.save(state_dict, ckpt_path)
                if dist.is_initialized() and process_exclusion:
                    dist.barrier()
        if env.rank == 0:
            config.save_pretrained(path)
        if env.rank == 0 and env.is_pipeline:
            # merge
            tmp_index_files = [tmp_index_file.format(i) for i in range(config.pp_size)]
            total_size = 0
            weight_map = {}
            for _file in tmp_index_files:
                _index_dict = json.loads(io_driver.load(_file, mode="r"))
                total_size += _index_dict["total_size"]
                weight_map.update(_index_dict["weight_map"])
                os.remove(_file)
            merged_dict = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map
            }
            io_driver.save(
                json.dumps(merged_dict, indent=2, sort_keys=True) + "\n",
                os.path.join(path, "pytorch_model.bin.index.json")
            )
        dist.barrier()