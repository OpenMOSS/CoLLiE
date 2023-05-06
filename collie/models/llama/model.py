import sys
sys.path.append("/mnt/lustre/zhangshuo/projects/collie-main/collie/Megatron-LM/")
sys.path.append("/mnt/lustre/zhangshuo/projects/collie/")

import torch
from torch import nn
import torch.nn.functional as F

from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
from deepspeed.pipe import LayerSpec, TiedLayerSpec

from megatron.core import tensor_parallel

from apex.normalization.fused_layer_norm import FusedRMSNorm

import math
from einops import rearrange

try:
    from flash_attn.flash_attention import FlashAttention
except ModuleNotFoundError:
    FlashAttention = None

from arguments import LlamaArguments

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (
            torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                seq_len: int,
                start_pos: int = 0):
        t = query.dtype
        query = torch.view_as_complex(
            query.float().reshape(*query.shape[:-1], -1, 2))
        key = torch.view_as_complex(
            key.float().reshape(*key.shape[:-1], -1, 2))
        freqs = torch.outer(torch.arange((2 ** 16) * 2, device=self.inv_freq.device), self.inv_freq).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)[start_pos: start_pos + seq_len]
        shape = [d if i == 1 or i == query.ndim -
                    1 else 1 for i, d in enumerate(query.shape)]
        freqs_cis = freqs_cis.view(*shape)
        query = torch.view_as_real(query * freqs_cis).flatten(3)
        key = torch.view_as_real(key * freqs_cis).flatten(3)
        return query.type(t), key.type(t)

class LlamaLayer(nn.Module):
    def __init__(self, args: LlamaArguments) -> None:
        super().__init__()
        self.args = args
        self.q_proj = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            args.hidden_size,
            bias=False,
            gather_output=False
        )
        self.k_proj = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            args.hidden_size,
            bias=False,
            gather_output=False
        )
        self.v_proj = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            args.hidden_size,
            bias=False,
            gather_output=False
        )
        self.o_proj = tensor_parallel.RowParallelLinear(
            args.hidden_size,
            args.hidden_size,
            bias=False,
            input_is_parallel=True
        )
        self.input_layernorm = FusedRMSNorm(
            normalized_shape=args.hidden_size,
            eps=args.layer_norm_epsilon
        )
        self.gate_proj = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            args.intermediate_size,
            bias=False,
            gather_output=False
        )
        self.up_proj = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            args.intermediate_size,
            bias=False,
            gather_output=False
        )
        self.down_proj = tensor_parallel.RowParallelLinear(
            args.intermediate_size,
            args.hidden_size,
            bias=False,
            input_is_parallel=True
        )
        self.post_attention_layernorm = FusedRMSNorm(
            normalized_shape=args.hidden_size,
            eps=args.layer_norm_epsilon
        )
        self.rotary_emb = RotaryPositionEmbedding(self.args.hidden_size // self.args.num_attention_heads)
    
    def _forward(self, hidden_states: torch.Tensor):
        assert hidden_states.ndim == 3, f"hidden_states.shape must be (B, N, H), but got {hidden_states.shape}"
        batch_size, seq_len, _ = hidden_states.shape
        head_dim = self.args.hidden_size // self.args.num_attention_heads
        _hidden_states = self.input_layernorm(hidden_states)
        query, key, value = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        query, key, value = rearrange(query, "b n (h d) -> b n h d", d=head_dim), \
            rearrange(key, "b n (h d) -> b n h d", d=head_dim), \
                rearrange(value, "b n (h d) -> b n h d", d=head_dim)
        query, key = self.rotary_emb(query, key, seq_len)
        if self.args.use_flash:
            assert FlashAttention is not None, \
                "Detected flash_attn is not installed. See https://github.com/HazyResearch/flash-attention"
            qkv = torch.stack([query, key, value], dim=2)
            output, _ = FlashAttention()(qkv, causal=True)
            output = rearrange(output, "b n h d -> b n (h d)", n=seq_len)
            output = F.dropout(output, p=self.args.dropout, training=self.training)
        else:
            query, key, value = query.permute(0, 2, 1, 3), key.permute(
                0, 2, 1, 3), value.permute(0, 2, 1, 3)
            attention_score = torch.matmul(query, key.transpose(
                2, 3)) / math.sqrt(head_dim)
            if seq_len > 1:
                mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
                mask = torch.triu(mask, diagonal=1).to(
                    attention_score.device)
                attention_score = attention_score + mask
            attention_score = F.softmax(
                attention_score, dim=-1).type_as(value)
            output = torch.matmul(attention_score, value)
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            output = F.dropout(output, p=self.args.dropout, training=self.training)
        hidden_states = hidden_states + self.o_proj(output)
        _hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.mlp["dropout"](self.mlp["w2"](F.silu(self.mlp["w1"](_hidden_states)) * self.mlp["w3"](_hidden_states)))
        hidden_states = hidden_states + F.dropout(self.down_proj(F.silu(self.gate_proj(_hidden_states)) * self.up_proj(_hidden_states)), p=self.args.dropout, training=self.training)
        return hidden_states
    
    def forward(self, hidden_states: torch.Tensor):
        if self.args.checkpointing:
            return checkpoint(self.forward, hidden_states)
        else:
            return self._forward(hidden_states)
        
class LlamaModel(nn.Module):
    def __init__(self, args: LlamaArguments) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            args.vocab_size,
            args.hidden_size
        )
        self.layers = nn.Sequential(*[LlamaLayer(args) for _ in range(args.num_hidden_layers)])
        self.norm = FusedRMSNorm(
            normalized_shape=args.hidden_size,
            eps=args.layer_norm_epsilon
        )
        self.lm_head = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            args.vocab_size,
            bias=False
        )
        
    def __new__(cls, args: LlamaArguments) -> object:
        if args.pp_size == 1:
            return super().__new__(LlamaModel)
        else:
            return [
                TiedLayerSpec(
                    "embed_tokens",
                    tensor_parallel.VocabParallelEmbedding,
                    args.vocab_size,
                    args.hidden_size),
                *[LayerSpec(LlamaLayer, args) for _ in range(args.num_hidden_layers)],
                TiedLayerSpec(
                    "embed_tokens",
                    tensor_parallel.ColumnParallelLinear,
                    args.hidden_size,
                    args.vocab_size,
                    bias=False)
            ]
    
    def forward(self, input_ids: torch.Tensor):
        assert input_ids.ndim == 2, f"input_ids.shape must be (B, N), but got {input_ids.shape}"
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
    
if __name__ == "__main__":
    import deepspeed
    from megatron.core import parallel_state
    from deepspeed.pipe import PipelineModule
    from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
    args = LlamaArguments()
    args.pp_size = 1
    args.tp_size = 2
    deepspeed.init_distributed(dist_backend='nccl', init_method="env://")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=args.tp_size, pipeline_model_parallel_size=args.pp_size)
    tensor_parallel.model_parallel_cuda_manual_seed(args.seed)
    if args.pp_size > 1:
        model = PipelineModule(
            layers=LlamaModel(args),
            num_stages=args.pp_size,
            topology=PipeModelDataParallelTopology(num_pp=args.pp_size, num_dp=args.dp_size, num_mp=args.tp_size)
        )
    else:
        model = LlamaModel(args)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config={
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                "lr": 0.001,
                "betas": [
                    0.8,
                    0.999
                ],
                "eps": 1e-8,
                "weight_decay": 3e-7
                }
            }
        }
    )
    input_ids = torch.randint(0, args.vocab_size, (8, 1024), dtype=torch.long).cuda(0)
    print(engine(input_ids))