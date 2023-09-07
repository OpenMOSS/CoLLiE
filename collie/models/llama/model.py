import gc
import json
import math
import os
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from deepspeed.accelerator import get_accelerator
from deepspeed.pipe import LayerSpec, TiedLayerSpec
from einops import rearrange
from megatron.core import parallel_state, tensor_parallel
from torch import nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import dtype_byte_size
from accelerate.utils.modeling import set_module_tensor_to_device
from collie.config import CollieConfig
from collie.driver.io import IODriver
from collie.log.logger import logger
from collie.models.base import CollieModelForCausalLM
from collie.models.utils import flash_attention
from collie.module import (
    ColumnParallelLinearWithoutBias,
    ColumnParallelLMHead,
    RowParallelLinearWithoutBias,
)
from collie.utils import concat_tensor, dict_as_params, env, progress


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, seq_len: int, start_pos: int = 0
    ):
        t = query.dtype
        query = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
        key = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
        freqs = torch.outer(
            torch.arange((2**16) * 2, device=self.inv_freq.device), self.inv_freq
        ).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)[
            start_pos : start_pos + seq_len
        ]
        shape = [
            d if i == 1 or i == query.ndim - 1 else 1 for i, d in enumerate(query.shape)
        ]
        freqs_cis = freqs_cis.view(*shape)
        query = torch.view_as_real(query * freqs_cis).flatten(3)
        key = torch.view_as_real(key * freqs_cis).flatten(3)
        return query.type(t), key.type(t)


class RMSNormalize(nn.Module):
    def __init__(self, dim=None, dtype=torch.float, eps=1e-5, weight=None):
        super(RMSNormalize, self).__init__()
        self.dtype = dtype
        if weight is not None:
            self.weight = weight
        else:
            self.weight = nn.Parameter(
                torch.ones(
                    dim, dtype=dtype, device=get_accelerator().current_device_name()
                )
            )
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight

    def post_init(self):
        if self.weight.device==torch.device("meta"):
            device = 'cpu'
        else:
            device = self.weight.device
        return torch.ones(
                    self.weight.shape, dtype=self.dtype, device=device
                )

class LlamaLayer(nn.Module):
    def __init__(self, config: CollieConfig, layer_idx) -> None:
        super().__init__()
        self.idx = layer_idx
        self.config = config
        if hasattr(config, "num_key_value_heads"):
            # llama2 (transformers >= 4.31.0)
            self.num_key_value_heads = config.num_key_value_heads
        else:
            self.num_key_value_heads = config.num_attention_heads
        self.num_key_value_groups = (
            config.num_attention_heads // self.num_key_value_heads
        )
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": ColumnParallelLinearWithoutBias(
                    config.hidden_size,
                    self.num_heads * self.head_dim,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x,
                ),
                "k_proj": ColumnParallelLinearWithoutBias(
                    config.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x,
                ),
                "v_proj": ColumnParallelLinearWithoutBias(
                    config.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x,
                ),
                "o_proj": RowParallelLinearWithoutBias(
                    self.num_heads * self.head_dim,
                    config.hidden_size,
                    bias=False,
                    input_is_parallel=True,
                    init_method=lambda x: x,
                ),
                "rotary_emb": RotaryPositionEmbedding(
                    self.config.hidden_size // self.config.num_attention_heads
                ),
            }
        )
        self.input_layernorm = RMSNormalize(
            dim=config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": ColumnParallelLinearWithoutBias(
                    config.hidden_size,
                    config.intermediate_size,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x,
                ),
                "up_proj": ColumnParallelLinearWithoutBias(
                    config.hidden_size,
                    config.intermediate_size,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x,
                ),
                "down_proj": RowParallelLinearWithoutBias(
                    config.intermediate_size,
                    config.hidden_size,
                    bias=False,
                    input_is_parallel=True,
                    init_method=lambda x: x,
                ),
            }
        )
        self.post_attention_layernorm = RMSNormalize(
            dim=config.hidden_size, eps=config.rms_norm_eps
        )
        # 务必保持变量名一致
        self.use_cache = self.config.model_config.use_cache
        self.hidden_states = None

    def _forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if not self.training:
            self.hidden_states = hidden_states
        else:
            self.hidden_states = None
        layer_past = None
        if past_key_values is not None:
            layer_past = past_key_values[self.idx]
        assert (
            hidden_states.ndim == 3
        ), f"hidden_states.shape must be (B, N, H), but got {hidden_states.shape}"
        batch_size, seq_len, _ = hidden_states.shape
        _hidden_states = self.input_layernorm(hidden_states)
        query, key, value = (
            self.self_attn["q_proj"](_hidden_states),
            self.self_attn["k_proj"](_hidden_states),
            self.self_attn["v_proj"](_hidden_states),
        )
        query, key, value = (
            rearrange(query, "b n (h d) -> b n h d", d=self.head_dim),
            rearrange(key, "b n (h d) -> b n h d", d=self.head_dim),
            rearrange(value, "b n (h d) -> b n h d", d=self.head_dim),
        )
        if layer_past is not None:
            start_pos = layer_past[0].shape[2]
        else:
            start_pos = 0
        query, key = self.self_attn["rotary_emb"](query, key, seq_len, start_pos)
        if self.num_key_value_groups > 1:
            key = torch.repeat_interleave(key, dim=2, repeats=self.num_key_value_groups)
            value = torch.repeat_interleave(
                value, dim=2, repeats=self.num_key_value_groups
            )
        if layer_past is not None:
            # past_key: batch_size, num_heads, seq_len, head_dim
            past_key = (
                layer_past[0]
                .reshape(*layer_past[0].shape[:-1], 2, -1)
                .permute(0, 2, 1, 4, 3)
                .reshape(batch_size, start_pos, self.num_heads, -1)
            )
            query = torch.cat([past_key, query], dim=1)
            key = torch.cat([past_key, key], dim=1)
            value = torch.cat([layer_past[1].permute([0, 2, 1, 3]), value], dim=1)
        new_layer_past = None
        if self.use_cache and not self.training:
            # 调整成和 hf 兼容的格式，方便 prefix tuning
            present_key = (
                key.reshape(*key.shape[:-1], -1, 2)
                .permute(0, 2, 1, 4, 3)
                .reshape(
                    batch_size,
                    self.num_heads // self.config.tp_size,
                    seq_len + start_pos,
                    -1,
                )
            )
            new_layer_past = torch.stack(
                (present_key, value.permute([0, 2, 1, 3])), dim=0
            )
        attention_mask = (
            attention_mask
            if attention_mask is not None
            else torch.ones((query.shape[0], query.shape[1])).to(hidden_states.device)
        )
        if self.config.use_flash:
            output = flash_attention(query, key, value, attention_mask)
        else:
            query, key, value = (
                query.permute(0, 2, 1, 3),
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
            )
            attention_score = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )
            if seq_len + start_pos > 1:
                mask = torch.full(
                    (1, 1, seq_len + start_pos, seq_len + start_pos), float("-inf")
                )
                mask = torch.triu(mask, diagonal=1).to(attention_score.device)
                attention_score = attention_score + mask
            key_padding_mask = (
                1.0 - attention_mask.unsqueeze(1).unsqueeze(2)
            ) * torch.finfo(attention_score.dtype).min
            attention_score = F.softmax(
                attention_score + key_padding_mask, dim=-1
            ).type_as(value)
            output = torch.matmul(attention_score, value)
            output = (
                output.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len + start_pos, -1)
            )
        output = F.dropout(output, p=self.config.dropout, training=self.training)
        output = output[:, start_pos:, :]
        hidden_states = hidden_states + self.self_attn["o_proj"](output)
        _hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + F.dropout(
            self.mlp["down_proj"](
                F.silu(self.mlp["gate_proj"](_hidden_states))
                * self.mlp["up_proj"](_hidden_states)
            ),
            p=self.config.dropout,
            training=self.training,
        )
        return hidden_states, new_layer_past

    def forward(self, inputs: dict):
        if self.config.checkpointing and self.training:
            hidden_states, new_layer_past = torch.utils.checkpoint.checkpoint(
                self._forward,
                inputs["hidden_states"],
                inputs.get("attention_mask", None),
                inputs.get("past_key_values", None),
            )
        else:
            hidden_states, new_layer_past = self._forward(**inputs)
        inputs["hidden_states"] = hidden_states
        new_past_key_values = inputs.get("new_past_key_values", None)
        if new_layer_past is not None:
            new_layer_past.unsqueeze_(0)
            if new_past_key_values is None:
                new_past_key_values = new_layer_past
            else:
                new_past_key_values = concat_tensor(
                    [new_past_key_values, new_layer_past]
                ).to(new_layer_past.device)
            inputs["new_past_key_values"] = new_past_key_values

        return inputs


class LlamaModel(nn.Module):
    def __init__(self, config: CollieConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [LlamaLayer(self.config, i) for i in range(self.config.num_hidden_layers)]
        )
        self.norm = RMSNormalize(
            dim=self.config.hidden_size, eps=self.config.rms_norm_eps
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        if input_ids == None:
            inputs["hidden_states"] = kwargs["inputs_embeds"]
        else:
            inputs["hidden_states"] = self.embed_tokens(inputs["input_ids"])
        if past_key_values is not None:
            inputs["past_key_values"] = past_key_values
        all_hidden_states = ()
        for layer in self.layers:
            all_hidden_states += (inputs["hidden_states"],)
            inputs.update(layer(inputs))
        inputs["hidden_states"] = self.norm(inputs["hidden_states"])
        all_hidden_states += (inputs["hidden_states"],)

        past_key_values = None
        if "new_past_key_values" in inputs:
            past_key_values = inputs["new_past_key_values"]

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

        if config.tie_word_embeddings:
            embed_tokens = TiedLayerSpec(
                "embed_tokens",
                dict_as_params(input_keys="input_ids", output_keys="hidden_states"),
                tensor_parallel.VocabParallelEmbedding,
                config.vocab_size,
                config.hidden_size,
            )
        else:
            embed_tokens = LayerSpec(
                dict_as_params(input_keys="input_ids", output_keys="hidden_states"),
                tensor_parallel.VocabParallelEmbedding,
                config.vocab_size,
                config.hidden_size,
            )

        layers = [
            LayerSpec(LlamaLayer, config, i) for i in range(config.num_hidden_layers)
        ]
        norm = LayerSpec(
            dict_as_params(input_keys="hidden_states", output_keys="hidden_states"),
            RMSNormalize,
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        return [
            ("embed_tokens", embed_tokens),
            ("layers", layers),
            ("norm", norm),
        ]


class LlamaForCausalLM(CollieModelForCausalLM):
    base_model_prefix = "model"

    def __init__(self, config: CollieConfig) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = ColumnParallelLinearWithoutBias(
            self.collie_config.hidden_size, self.collie_config.vocab_size, bias=False
        )
        # GenerationMixin 需要的额外参数
        self.config.is_decoder = True
        if config.model_config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.main_input_name = "input_ids"

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
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

    def clean_cache(self):
        self._clean_hidden_states([*self.model.layers, self.lm_head])
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

        if config.tie_word_embeddings:
            lm_head = TiedLayerSpec(
                "embed_tokens",
                dict_as_params(input_keys="hidden_states", output_keys="logits"),
                ColumnParallelLMHead,
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
        else:
            lm_head = LayerSpec(
                dict_as_params(input_keys="hidden_states", output_keys="logits"),
                ColumnParallelLMHead,
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )

        return [("model", LlamaModel.pipeline_layers(config)), ("lm_head", lm_head)]

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
        **kwargs,
    ):
        """
        Load state_dict from ``path``.

        The format of pretrained model should be the same as that of
        `huggingface`.

        :return: state_dict. Note that the state_dict should be processed
            properly to match the current rank.
        """
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
                if hasattr(config, "num_key_value_heads"):
                    # llama2 (transformers >= 4.31.0)
                    num_key_value_heads = config.num_key_value_heads
                else:
                    num_key_value_heads = config.num_attention_heads
                head_dim = config.hidden_size // config.num_attention_heads
                # 如果存在 pytorch_model.bin.index.json 文件的话，此时不同的 pp 进程可以按需加载自己需要的权重
                if (
                    io_driver.exists(os.path.join(path, "pytorch_model.bin.index.json"))
                    and "COLLIE_PP_PARTS" in os.environ.keys()
                ):
                    weight_map = json.loads(
                        io_driver.load(
                            os.path.join(path, "pytorch_model.bin.index.json"), mode="r"
                        )
                    )["weight_map"]
                    # layers 表示自己需要的层
                    layers = env.pipeline_layers_idx
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
                        weights.append(weight_map["model.embed_tokens.weight"])
                    if max(parts) - 1 in layers:
                        weights.append(weight_map["lm_head.weight"])
                    if max(parts) - 2 in layers:
                        weights.append(weight_map["model.norm.weight"])
                else:
                    # 如果没有 pytorch_model.bin.index.json 文件的话，那么就加载所有的权重
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
                            # 对 q_proj.weight 和 k_proj.weight 进行 reshape
                            if key.endswith("q_proj.weight"):
                                part_state_dict[key] = (
                                    rearrange(
                                        part_state_dict[key],
                                        "(h two t) d -> h two t d",
                                        h=config.num_attention_heads,
                                        two=2,
                                    )
                                    .transpose(1, 2)
                                    .reshape(config.hidden_size, config.hidden_size)
                                )
                            elif key.endswith("k_proj.weight"):
                                part_state_dict[key] = (
                                    rearrange(
                                        part_state_dict[key],
                                        "(h two t) d -> h two t d",
                                        h=num_key_value_heads,
                                        two=2,
                                    )
                                    .transpose(1, 2)
                                    .reshape(
                                        num_key_value_heads * head_dim,
                                        config.hidden_size,
                                    )
                                )
                        state_dict.update(part_state_dict)
                        del part_state_dict
                if parts is not None:
                    # 这一步是 pp 的复筛
                    layers = env.pipeline_layers_idx
                    for key in list(state_dict.keys()):
                        if key.startswith("layers"):
                            layer = int(key.split(".")[1])
                            if layer + 1 not in layers:
                                state_dict.pop(key)
                        if key.endswith("embed_tokens.weight"):
                            if 0 not in layers:
                                state_dict.pop(key)
                        if key == "norm.weight":
                            if max(parts) - 2 not in layers:
                                state_dict.pop(key)
                        if key.endswith("lm_head.weight"):
                            if max(parts) - 1 not in layers:
                                state_dict.pop(key)
                # 根据用户配置的新的 tp size 进行分割
                for key in list(state_dict.keys()):
                    col_filter = [
                        "q_proj.weight",
                        "k_proj.weight",
                        "v_proj.weight",
                        "gate_proj.weight",
                        "up_proj.weight",
                        "embed_tokens.weight",
                        "lm_head.weight",
                    ]
                    col_split = any([key.endswith(filter) for filter in col_filter])

                    if col_split:
                        tensor = (
                            list(torch.chunk(state_dict[key], config.tp_size, dim=0))[
                                env.tp_rank
                            ]
                            .detach()
                            .clone()
                        )
                        del state_dict[key]
                        if process_exclusion:
                            # CPU 内存回收（速度很慢）
                            gc.collect()
                        state_dict[key] = tensor
                    elif key.endswith("o_proj.weight") or key.endswith(
                        "down_proj.weight"
                    ):
                        tensor = (
                            list(torch.chunk(state_dict[key], config.tp_size, dim=1))[
                                env.tp_rank
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

        def reshape_wq_wk(w: torch.Tensor, kv=False):
            if hasattr(config, "num_key_value_heads"):
                # llama2 (transformers >= 4.31.0)
                num_key_value_heads = config.num_key_value_heads
            else:
                num_key_value_heads = config.num_attention_heads
            head_dim = config.hidden_size // config.num_attention_heads
            if kv:
                num_heads = num_key_value_heads
            else:
                num_heads = config.num_attention_heads
            return (
                w.view(num_heads, head_dim // 2, 2, config.hidden_size)
                .transpose(1, 2)
                .reshape(num_heads * head_dim, config.hidden_size)
            )

        # gather to tp rank 0
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
                            col_filter = [
                                "q_proj.weight",
                                "k_proj.weight",
                                "v_proj.weight",
                                "gate_proj.weight",
                                "up_proj.weight",
                                "embed_tokens.weight",
                                "lm_head.weight",
                            ]
                            col_split = any(
                                [key.endswith(filter) for filter in col_filter]
                            )

                            if col_split:
                                state_dict[key] = concat_tensor(tensor_list, dim=0)

                                if process_exclusion:
                                    # CPU 内存回收（速度很慢）
                                    gc.collect()

                            elif key.endswith("o_proj.weight") or key.endswith(
                                "down_proj.weight"
                            ):
                                state_dict[key] = concat_tensor(tensor_list, dim=1)

                                if process_exclusion:
                                    # CPU 内存回收（速度很慢）
                                    gc.collect()
                            if key.endswith("q_proj.weight"):
                                state_dict[key] = reshape_wq_wk(state_dict[key])
                            if key.endswith("k_proj.weight"):
                                state_dict[key] = reshape_wq_wk(
                                    state_dict[key], kv=True
                                )
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
                                index_dict, index_dicts if env.pp_rank == 0 else None
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
