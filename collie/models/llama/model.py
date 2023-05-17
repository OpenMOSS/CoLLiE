import os
import gc
import json

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
from deepspeed.pipe import LayerSpec, TiedLayerSpec

from megatron.core import tensor_parallel
from megatron.core import parallel_state

from apex.normalization.fused_layer_norm import FusedRMSNorm

import math
from einops import rearrange

try:
    from flash_attn.flash_attention import FlashAttention
except ModuleNotFoundError:
    FlashAttention = None

from collie.log.logger import logger
from collie.models.base import BaseModel
from collie.driver.io.file import FileIODriver
from collie.trainer.arguments import load_config
from collie.driver.io.petrel import PetrelIODriver
from collie.models.llama.arguments import LlamaArguments
from collie.module import ColumnParallelLinearWithoutBias, RowParallelLinearWithoutBias, ColumnParallelLMHead
from collie.utils import progress, env

from typing import Union, Optional
from collections import OrderedDict
from transformers.modeling_utils import dtype_byte_size
from transformers.modeling_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

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
        freqs = torch.outer(torch.arange(
            (2 ** 16) * 2, device=self.inv_freq.device), self.inv_freq).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)[
            start_pos: start_pos + seq_len]
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
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": ColumnParallelLinearWithoutBias(
                    args.hidden_size,
                    args.hidden_size,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x
                ),
                "k_proj": ColumnParallelLinearWithoutBias(
                    args.hidden_size,
                    args.hidden_size,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x
                ),
                "v_proj": ColumnParallelLinearWithoutBias(
                    args.hidden_size,
                    args.hidden_size,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x
                ),
                "o_proj": RowParallelLinearWithoutBias(
                    args.hidden_size,
                    args.hidden_size,
                    bias=False,
                    input_is_parallel=True,
                    init_method=lambda x: x
                ),
                "rotary_emb": RotaryPositionEmbedding(
                    self.args.hidden_size // self.args.num_attention_heads)
            }
        )
        self.input_layernorm = FusedRMSNorm(
            normalized_shape=args.hidden_size,
            eps=args.layer_norm_epsilon
        )
        self.mlp = nn.ModuleDict({
            "gate_proj": ColumnParallelLinearWithoutBias(
                args.hidden_size,
                args.intermediate_size,
                bias=False,
                gather_output=False,
                init_method=lambda x: x
            ),
            "up_proj": ColumnParallelLinearWithoutBias(
                args.hidden_size,
                args.intermediate_size,
                bias=False,
                gather_output=False,
                init_method=lambda x: x
            ),
            "down_proj": RowParallelLinearWithoutBias(
                args.intermediate_size,
                args.hidden_size,
                bias=False,
                input_is_parallel=True,
                init_method=lambda x: x
            )
        })
        self.post_attention_layernorm = FusedRMSNorm(
            normalized_shape=args.hidden_size,
            eps=args.layer_norm_epsilon
        )
        # 务必保持变量名一致
        self.use_cache = True
        self.past_key_values = None
        self.hidden_states = None

    def _forward(self, hidden_states: torch.Tensor):
        if not self.training:
            self.hidden_states = hidden_states
        assert hidden_states.ndim == 3, f"hidden_states.shape must be (B, N, H), but got {hidden_states.shape}"
        batch_size, seq_len, _ = hidden_states.shape
        head_dim = self.args.hidden_size // self.args.num_attention_heads
        _hidden_states = self.input_layernorm(hidden_states)
        query, key, value = self.self_attn["q_proj"](_hidden_states), self.self_attn["k_proj"](
            _hidden_states), self.self_attn["v_proj"](_hidden_states)
        query, key, value = rearrange(query, "b n (h d) -> b n h d", d=head_dim), \
            rearrange(key, "b n (h d) -> b n h d", d=head_dim), \
            rearrange(value, "b n (h d) -> b n h d", d=head_dim)
        if not self.training and self.past_key_values is not None:
            start_pos = self.past_key_values[0].shape[1]
        else:
            start_pos = 0
        query, key = self.self_attn["rotary_emb"](query, key, seq_len, start_pos)
        if not self.training and self.use_cache:
            if self.past_key_values is not None:
                query = torch.cat([self.past_key_values[0], query], dim=1)
                key = torch.cat([self.past_key_values[0], key], dim=1)
                value = torch.cat([self.past_key_values[1], value], dim=1)
            self.past_key_values = [key, value]

        if self.args.use_flash:
            assert FlashAttention is not None, \
                "Detected flash_attn is not installed. See https://github.com/HazyResearch/flash-attention"
            qkv = torch.stack([query, key, value], dim=2)
            output, _ = FlashAttention()(qkv, causal=True)
            output = rearrange(output, "b n h d -> b n (h d)")
            output = F.dropout(output, p=self.args.dropout,
                               training=self.training)
        else:
            query, key, value = query.permute(0, 2, 1, 3), key.permute(
                0, 2, 1, 3), value.permute(0, 2, 1, 3)
            attention_score = torch.matmul(query, key.transpose(
                2, 3)) / math.sqrt(head_dim)
            if seq_len + start_pos > 1:
                mask = torch.full((1, 1, seq_len + start_pos, seq_len + start_pos), float("-inf"))
                mask = torch.triu(mask, diagonal=1).to(
                    attention_score.device)
                attention_score = attention_score + mask
            attention_score = F.softmax(
                attention_score, dim=-1).type_as(value)
            output = torch.matmul(attention_score, value)
            output = output.transpose(1, 2).contiguous().view(
                batch_size, seq_len + start_pos, -1)
            output = F.dropout(output, p=self.args.dropout,
                               training=self.training)
        output = output[:, start_pos:, :]
        hidden_states = hidden_states + self.self_attn["o_proj"](output)
        _hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + F.dropout(self.mlp["down_proj"](F.silu(self.mlp["gate_proj"](
            _hidden_states)) * self.mlp["up_proj"](_hidden_states)), p=self.args.dropout, training=self.training)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor):
        if self.args.checkpointing:
            return checkpoint(self._forward, hidden_states)
        else:
            return self._forward(hidden_states)


class LlamaForCasualLM(BaseModel):
    def __init__(self, args: Union[LlamaArguments, str]) -> None:
        super().__init__()
        if isinstance(args, str):
            args = load_config(args)
        self.args = args
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            self.args.vocab_size,
            self.args.hidden_size
        )
        self.layers = nn.Sequential(
            *[LlamaLayer(self.args) for _ in range(self.args.num_hidden_layers)])
        self.norm = FusedRMSNorm(
            normalized_shape=self.args.hidden_size,
            eps=self.args.layer_norm_epsilon
        )
        self.lm_head = ColumnParallelLMHead(
            self.args.hidden_size,
            self.args.vocab_size,
            bias=False
        )
        # GenerationMixin 需要的额外参数
        self.config = PretrainedConfig(is_decoder=True)
        self.main_input_name = "input_ids"
        

    def forward(self, input_ids: torch.Tensor):
        past_key_values=self._get_past_key_values(self.layers)
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        assert input_ids.ndim == 2, f"input_ids.shape must be (B, N), but got {input_ids.shape}"
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=self._get_past_key_values(self.layers),
            hidden_states=self._get_hidden_states([*self.layers, self.lm_head]),
            attentions=None
        )
        
    def prepare_inputs_for_generation(self, 
                                      input_ids: torch.Tensor,
                                      past_key_values: Optional[list] = None,
                                      attention_mask: Optional[torch.Tensor] = None):
        self._set_use_cache(self.layers, self.generation_config.use_cache)
        if past_key_values is None:
            self._clean_past_key_values()
        else:
            self._set_past_key_values(self.layers, past_key_values)
        return {"input_ids": input_ids}

    @classmethod
    def pipeline_layers(cls, args: Union[LlamaArguments, str]):
        """
        Get layers of pipeline.

        :return: list
        """
        if isinstance(args, str) and os.path.exists(args):
            args = load_config(args)
        return [TiedLayerSpec(
            "embed_tokens",
            tensor_parallel.VocabParallelEmbedding,
            args.vocab_size,
            args.hidden_size),
            *[LayerSpec(LlamaLayer, args)
              for _ in range(args.num_hidden_layers)],
            LayerSpec(FusedRMSNorm,
                      normalized_shape=args.hidden_size,
                      eps=args.layer_norm_epsilon),
            TiedLayerSpec(
            "embed_tokens",
            ColumnParallelLMHead,
            args.hidden_size,
            args.vocab_size,
            bias=False)
        ]

    @staticmethod
    def load_parallel_state_dict(path: str, args: Union[LlamaArguments, str],
                                 process_exclusion: bool = False):...
    @staticmethod
    def load_parallel_state_dict(path: str, 
                                 args: Union[LlamaArguments, str],
                                 process_exclusion: bool = False,
                                 protocol: str = 'file', 
                                 format: str = 'hf'):
        """
        Load state_dict from ``path``.

        The format of pretrained model should be the same as that of
        `huggingface`.

        :return: state_dict. Note that the state_dict should be processed
            properly to match the current rank.
        """
        assert format in ["hf", "meta"], "Only support hf and meta format"
        assert protocol in ["file", "petrel"], "Only support file and petrel protocol"
        if isinstance(args, str) and os.path.exists(args):
            args = load_config(args)
        IODriver = FileIODriver if protocol == 'file' else PetrelIODriver
        if not IODriver.exists(path):
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
                    parts = env.pipline_parts
                if format == "hf":
                    # 根据 huggingface 中的 config.json 更新一下用户配置
                    if IODriver.exists(os.path.join(path, "config.json")):
                        config = json.loads(IODriver.load(os.path.join(path, "config.json"), mode="r"))
                        for key, value in {
                            "vocab_size": config["vocab_size"],
                            "hidden_size": config["hidden_size"],
                            "intermediate_size": config["intermediate_size"],
                            "num_hidden_layers": config["num_hidden_layers"],
                            "num_attention_heads": config["num_attention_heads"],
                            "layer_norm_epsilon": config["rms_norm_eps"]
                        }.items():
                            setattr(args, key, value)
                    # 如果存在 pytorch_model.bin.index.json 文件的话，此时不同的 pp 进程可以按需加载自己需要的权重
                    if IODriver.exists(os.path.join(path, "pytorch_model.bin.index.json")) and "COLLIE_PP_PARTS" in os.environ.keys():
                        weight_map = json.loads(IODriver.load(os.path.join(path, "pytorch_model.bin.index.json"), mode="r"))["weight_map"]
                        # layers 表示自己需要的层
                        layers = list(range(parts[int(os.environ["COLLIE_PP_RANK"])], parts[int(os.environ["COLLIE_PP_RANK"]) + 1]))
                        # 筛选出形似 model.layers.0 这样的层。包含两个条件：1. 有数字的层；2. 数字加一要在 layers 里面（因为最开始还有个 embedding 占一层）
                        weights.extend([value for key, value in weight_map.items() \
                            if len(key.split(".")) > 2 \
                                and key.split(".")[2].isdigit() \
                                    and (int(key.split(".")[2]) + 1) in layers])
                        # 去重
                        weights = list(set(weights))
                        # 继续筛选，如果有 0 层，那么就要加载 embedding；如果有最后一层，那么就要加载 lm_head；如果有倒数第二层，那么就要加载 norm
                        if 0 in layers:
                            weights.append(weight_map["model.embed_tokens.weight"])
                        if max(parts) - 1 in layers:
                            weights.append(weight_map["lm_head.weight"])
                        if max(parts) - 2 in layers:
                            weights.append(weight_map["model.norm.weight"])
                    else:
                        # 如果没有 pytorch_model.bin.index.json 文件的话，那么就加载所有的权重
                        weights = [weight for weight in IODriver.list(path) if weight.endswith(".bin")]
                    with progress(weights, desc="Loading state dict", total=len(weights), disable=hide_progress) as pbar:
                        for weight in pbar:
                            part_state_dict = IODriver.load(os.path.join(path, weight), mode="rb")
                            for key in list(part_state_dict.keys()):
                                # 对 q_proj.weight 和 k_proj.weight 进行 reshape
                                if key.endswith("q_proj.weight") or key.endswith("k_proj.weight"):
                                    part_state_dict[key] = rearrange(
                                        part_state_dict[key],
                                        "(h two t) d -> h two t d",
                                        h=args.num_attention_heads,
                                        two=2).transpose(1, 2).reshape(
                                            args.hidden_size,
                                            args.hidden_size)
                                part_state_dict[key.replace("model.", "")] = part_state_dict.pop(key)
                            state_dict.update(part_state_dict)
                            del part_state_dict
                elif format == "meta":
                    # meta 权重的格式，需要补充 inv_freq 的权重
                    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, (args.hidden_size // args.num_attention_heads),
                                2).float() / (args.hidden_size // args.num_attention_heads)))
                    # 根据 meta 中的 params.json 更新一下用户配置
                    if IODriver.exists(os.path.join(path, "params.json")):
                        params = json.loads(IODriver.load(os.path.join(path, "params.json"), mode="r"))
                        for key, value in {
                            "hidden_size": params["dim"],
                            "intermediate_size": params["multiple_of"] * ((int(2 * 4 * args.hidden_size / 3) + params["multiple_of"] - 1) // params["multiple_of"]),
                            "num_hidden_layers": params["n_layers"],
                            "num_attention_heads": params["n_heads"],
                            "layer_norm_epsilon": params["norm_eps"]
                        }.items():
                            setattr(args, key, value)
                    # 权重全部加载
                    weights = [weight for weight in IODriver.list(path) if (weight.endswith(".pt") or weight.endswith(".pth"))]
                    # 因为 meta 的权重默认 按照张量并行分割，cat 的时候存在顺序问题，所以先排序一下
                    weights = sorted(weights, key=lambda x: int(x.split(".")[1]))
                    with progress(weights, desc="Loading state dict", total=len(weights), disable=hide_progress) as pbar:
                        for weight in pbar:
                            part_state_dict = IODriver.load(os.path.join(path, weight), mode="rb")
                            for key in list(part_state_dict.keys()):
                                # if key.startswith("layers"):
                                #     layer = int(key.split(".")[1])
                                #     # meta 权重的格式，需要补充 inv_freq 的权重
                                #     part_state_dict[f"layers.{layer}.self_attn.rotary_emb.inv_freq"] = inv_freq
                                raw_key = key
                                key = key.replace("attention", "self_attn")
                                key = key.replace("inner_self_attn.rope.freqs", "rotary_emb.inv_freq")
                                key = key.replace("wo", "o_proj")
                                key = key.replace("wq", "q_proj")
                                key = key.replace("wk", "k_proj")
                                key = key.replace("wv", "v_proj")
                                key = key.replace("feed_forward", "mlp")
                                key = key.replace("w1", "gate_proj")
                                key = key.replace("w2", "down_proj")
                                key = key.replace("w3", "up_proj")
                                key = key.replace("self_attn_norm", "input_layernorm")
                                key = key.replace("ffn_norm", "post_attention_layernorm")
                                key = key.replace("tok_embeddings", "embed_tokens")
                                key = key.replace("output", "lm_head")
                                # 按照 hf 的格式更新字典
                                part_state_dict[key] = part_state_dict.pop(raw_key)
                            for key in list(part_state_dict.keys()):
                                if not key in state_dict.keys():
                                    state_dict[key] = part_state_dict[key]
                                else:
                                    # 组装一下
                                    if key.endswith("q_proj.weight") \
                                        or key.endswith("k_proj.weight") \
                                            or key.endswith("v_proj.weight") \
                                                or key.endswith("gate_proj.weight") \
                                                    or key.endswith("up_proj.weight") \
                                                        or key.endswith("lm_head.weight"):
                                                            state_dict[key] = torch.cat((state_dict[key], part_state_dict[key]), dim=0)
                                    if key.endswith("o_proj.weight") \
                                        or key.endswith("down_proj.weight") \
                                            or key.endswith("embed_tokens.weight"):
                                                state_dict[key] = torch.cat((state_dict[key], part_state_dict[key]), dim=1)
                            del part_state_dict
                if parts is not None:
                    # 这一步是 pp 的复筛
                    layers = list(range(parts[int(os.environ["COLLIE_PP_RANK"])], parts[int(os.environ["COLLIE_PP_RANK"]) + 1]))
                    for key in list(state_dict.keys()):
                        if key.startswith("layers"):
                            layer = int(key.split(".")[1])
                            if layer + 1 in layers:
                                state_dict[key.replace(f"layers.{layer}", f"{layer + 1}")] = state_dict.pop(key)
                            else:
                                # 形似 model.layers.0 这样的层，筛选掉数字加一不在 layers 里面得
                                state_dict.pop(key)
                        if key.endswith("embed_tokens.weight"):
                            if 0 in layers:
                                state_dict["tied_modules.embed_tokens.weight"] = state_dict.pop(key)
                            else:
                                state_dict.pop(key)
                        if key == "norm.weight":
                            if max(parts) - 2 in layers:
                                state_dict[f"{max(parts) - 2}.weight"] = state_dict.pop(key)
                            else:
                                state_dict.pop(key)
                        if key.endswith("lm_head.weight"):
                            if max(parts) - 1 in layers:
                                state_dict["tied_modules.embed_tokens.weight"] = state_dict.pop(key)
                            else:
                                state_dict.pop(key)
                # 根据用户配置的新的 tp size 进行分割
                for key in list(state_dict.keys()):
                    if key.endswith("q_proj.weight") \
                        or key.endswith("k_proj.weight") \
                            or key.endswith("v_proj.weight") \
                                or key.endswith("gate_proj.weight") \
                                    or key.endswith("up_proj.weight") \
                                        or key.endswith("embed_tokens.weight") \
                                            or key.endswith("lm_head.weight"):
                                                tensor = list(torch.chunk(state_dict[key], args.tp_size, dim=0))[int(os.environ.get("COLLIE_TP_RANK", "0"))].detach().clone()
                                                del state_dict[key]
                                                if process_exclusion:
                                                    # CPU 内存回收（速度很慢）
                                                    gc.collect()
                                                state_dict[key] = tensor
                    elif key.endswith("o_proj.weight") \
                        or key.endswith("down_proj.weight"):
                            tensor = list(torch.chunk(state_dict[key], args.tp_size, dim=1))[int(os.environ.get("COLLIE_TP_RANK", "0"))].detach().clone()
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
                                 args: LlamaArguments,
                                 process_exclusion: bool = False):...
    @staticmethod
    def save_parallel_state_dict(state_dict: dict,
                                 path: str, 
                                 args: LlamaArguments,
                                 process_exclusion: bool = False,
                                 protocol: str = 'file'):
        """
        Save state_dict to ``path``.

        The format of saved state dict should be the same as that of
        `huggingface`.
        """
        assert protocol in ["file", "petrel"], f"Only support file and petrel protocol, not `{protocol}`."
        IODriver = FileIODriver if protocol == 'file' else PetrelIODriver
        def reshape_wq_wk(w: torch.Tensor):
            return w.view(args.num_attention_heads, 
                          args.hidden_size // args.num_attention_heads // 2, 
                          2, 
                          args.hidden_size).transpose(1, 2).reshape(args.hidden_size, 
                                                                    args.hidden_size)
        # gather to tp rank 0
        if env.is_pipeline:
            layers = env.pipline_layers_idx
            parts = env.pipline_parts
            for key in list(state_dict.keys()):
                if key == "tied_modules.embed_tokens.weight":
                    if 0 in layers:
                        state_dict["model.embed_tokens.weight"] = state_dict.pop(key)
                    elif max(layers) - 1 in layers:
                        state_dict["lm_head.weight"] = state_dict.pop(key)
                else:
                    layer = int(key.split(".")[0])
                    if layer == max(parts) - 2:
                        state_dict[key.replace(f"{layer}.", "model.norm.")] = state_dict.pop(key)
                    else:
                        state_dict[key.replace(f"{layer}.", f"model.layers.{layer - 1}.")] = state_dict.pop(key)
        if dist.is_initialized() and process_exclusion:
            # 如果启动了进程互斥，则要进行 pp_size 次循环
            rank_order = range(args.pp_size)
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
                            tensor_list = [torch.zeros_like(state_dict[key]).to(state_dict[key].dtype).cuda() for _ in range(args.tp_size)]
                        dist.gather(state_dict[key].cuda(), dst=dst, gather_list=tensor_list, group=env.tp_group)
                        if env.tp_rank == 0:
                            if key.endswith("q_proj.weight") \
                                or key.endswith("k_proj.weight") \
                                    or key.endswith("v_proj.weight") \
                                        or key.endswith("gate_proj.weight") \
                                            or key.endswith("up_proj.weight") \
                                                or key.endswith("embed_tokens.weight") \
                                                    or key.endswith("lm_head.weight"):
                                                        state_dict[key] = torch.cat(tensor_list, dim=0).detach().clone().to(device)
                                                        if key.endswith("q_proj.weight")  or key.endswith("k_proj.weight"):
                                                            state_dict[key] = reshape_wq_wk(state_dict[key])
                                                        del tensor_list
                                                        if process_exclusion:
                                                            # CPU 内存回收（速度很慢）
                                                            gc.collect()
                            elif key.endswith("o_proj.weight") \
                                or key.endswith("down_proj.weight"):
                                    state_dict[key] = torch.cat(tensor_list, dim=1).detach().clone().to(device)
                                    del tensor_list
                                    if process_exclusion:
                                        # CPU 内存回收（速度很慢）
                                        gc.collect()
                    if env.tp_rank == 0:
                        # Save gathered weights
                        if env.is_pipeline:
                            ckpt_name = f"pytorch_model-{env.pp_rank+1:05d}-of-{args.pp_size:05d}.bin"
                            total_size = 0
                            weight_map = {}
                            for name, weight in state_dict.items():
                                weight_size = weight.numel() * dtype_byte_size(weight.dtype)
                                weight_map[name] = ckpt_name
                                total_size += weight_size
                            index_dict = dict(total_size=total_size, weight_map=weight_map)
                            tmp_index_file = os.path.join(path, "_tmp_index_{}.json")
                            IODriver.save(
                                json.dumps(index_dict), tmp_index_file.format(env.pp_rank)
                            )
                        else:
                            ckpt_name = f"pytorch_model.bin"
                        ckpt_path = os.path.join(path, ckpt_name)
                        IODriver.save(state_dict, ckpt_path)
                if dist.is_initialized() and process_exclusion:
                    dist.barrier()
        if dist.get_rank() == 0:
            config = {"architectures": ["LlamaForCausalLM"], 
                      "bos_token_id": 0, 
                      "eos_token_id": 1, 
                      "hidden_act": "silu", 
                      "hidden_size": args.hidden_size, 
                      "intermediate_size": args.intermediate_size, 
                      "initializer_range": 0.02, 
                      "max_sequence_length": 2048, 
                      "model_type": "llama", 
                      "num_attention_heads": args.num_attention_heads, 
                      "num_hidden_layers": args.num_hidden_layers, 
                      "pad_token_id": -1, 
                      "rms_norm_eps": args.layer_norm_epsilon, 
                      "torch_dtype": "float16" if args.fp16 else "float32", 
                      "transformers_version": "4.27.0.dev0", 
                      "use_cache": True, 
                      "vocab_size": args.vocab_size}
            IODriver.save(json.dumps(config, indent=4, sort_keys=True), os.path.join(path, "config.json"))
        if env.rank == 0 and env.is_pipeline:
            # merge
            tmp_index_files = [tmp_index_file.format(i) for i in range(args.pp_size)]
            total_size = 0
            weight_map = {}
            for _file in tmp_index_files:
                _index_dict = json.loads(IODriver.load(_file, mode="r"))
                total_size += _index_dict["total_size"]
                weight_map.update(_index_dict["weight_map"])
                os.remove(_file)
            merged_dict = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map
            }
            IODriver.save(
                json.dumps(merged_dict, indent=2, sort_keys=True) + "\n",
                os.path.join(path, "pytorch_model.bin.index.json")
            )
        dist.barrier()