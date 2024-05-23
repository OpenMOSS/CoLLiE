import json
import os
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
from deepspeed.pipe import LayerSpec
from torch import distributed as dist
from torch import nn
from transformers.activations import NewGELUActivation
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from collie.config import CollieConfig
from collie.driver.io import IODriver
from collie.log import logger
from collie.models.base import CollieModelForCausalLM
from collie.module import (
    ColumnParallelLinearWithoutBias,
    ColumnParallelLMHead,
    RowParallelLinearWithoutBias,
    VocabParallelEmbedding,
)
from collie.utils import env, progress
from collie.utils.utils import concat_tensor, dict_as_params, stack_tensor

from .utils import (
    _state_dict_to_load,
    _state_dict_to_save,
    _weight_name_in_current_rank,
    apply_rotary_pos_emb,
    create_sinusoidal_positions,
    set_index_dict,
)

__all__ = ["Moss003MoonForCausalLM"]


class MossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        max_positions = config.n_positions
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = config.n_embd
        self.num_attention_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        ).to(torch.get_default_dtype())
        self.qkv_proj = ColumnParallelLinearWithoutBias(
            self.embed_dim,
            self.embed_dim * 3,
            bias=False,
            gather_output=True,
        )

        self.out_proj = RowParallelLinearWithoutBias(
            self.embed_dim, self.embed_dim, bias=False, input_is_parallel=False
        )
        self.rotary_dim = config.rotary_dim
        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)

    def _split_heads(self, x, n_head, dim_head, mp_num):
        reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
        reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into n_ctx
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}"
            )
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        # compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.causal_mask[
            :, :, key_length - query_length : key_length, :key_length
        ]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights = attn_weights / self.scale_attn
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device
        )
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        qkv = self.qkv_proj(hidden_states)
        # TODO(enijkamp): factor out number of logical TPU-v4 cores or make forward pass agnostic
        mp_num = 4
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)
        query = self._split_heads(
            query, self.num_attention_heads, self.head_dim, mp_num=mp_num
        )
        key = self._split_heads(
            key, self.num_attention_heads, self.head_dim, mp_num=mp_num
        )

        value = self._split_heads(
            value, self.num_attention_heads, self.head_dim, mp_num=mp_num
        )

        embed_positions = self.embed_positions
        if embed_positions.device != position_ids.device:
            embed_positions = embed_positions.to(position_ids.device)
            self.embed_positions = embed_positions

        sincos = embed_positions[position_ids]
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
            q_rot = apply_rotary_pos_emb(q_rot, sin, cos)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            key = apply_rotary_pos_emb(key, sin, cos)
            query = apply_rotary_pos_emb(query, sin, cos)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1).to(value.dtype)

        if use_cache is True:
            present = stack_tensor([key, value]).to(key.device)
        else:
            present = None

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )

        attn_output = self._merge_heads(
            attn_output, self.num_attention_heads, self.head_dim
        )
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)

        return outputs  # a, present


class MossMLP(nn.Module):
    def __init__(
        self, intermediate_size, config
    ):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()
        embed_dim = config.n_embd

        self.fc_in = ColumnParallelLinearWithoutBias(
            embed_dim,
            intermediate_size,
            gather_output=False,
        )
        self.fc_out = RowParallelLinearWithoutBias(
            intermediate_size,
            embed_dim,
            input_is_parallel=True,
        )

        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states) -> torch.FloatTensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class MossBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = MossAttention(config)
        self.mlp = MossMLP(inner_dim, config)
        self.config = config
        self.idx = layer_idx

        self.use_cache = False
        self.hidden_states = None

    def set_norm_precision_to_float32(self):
        self.ln_1.weight.data = self.ln_1.weight.data.to(torch.float32) 
        self.ln_1.bias.data   = self.ln_1.bias.data.to(torch.float32) 

    def _forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)

    def forward(self, inputs):
        hidden_states = inputs["hidden_states"]
        attention_mask = inputs.get("attention_mask", None)
        past_key_values = inputs.get("past_key_values", None)
        new_past_key_values = inputs.get("new_past_key_values", None)
        if past_key_values is not None:
            layer_past = past_key_values[self.idx]
        else:
            layer_past = None
        if not self.training:
            self.hidden_states = hidden_states

        end_pos = hidden_states.shape[1]

        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(1)
        position_ids = inputs.get("position_ids", None)
        if position_ids is None:
            position_ids = torch.arange(
                past_length, end_pos + past_length, dtype=torch.long
            ).cuda()
            position_ids = position_ids.unsqueeze(0).view(-1, end_pos)

        if self.config.gradient_checkpointing and self.training:
            self.use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self._forward),
                hidden_states,
                None,
                attention_mask,
                position_ids,
            )
        else:
            outputs = self._forward(
                hidden_states,
                position_ids=position_ids,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=None,
                use_cache=self.use_cache,
            )

        if self.use_cache:
            present = outputs[1].unsqueeze(0)
            # 1, 2, bsz, seqlen, ...
            if new_past_key_values is None:
                # 第一层
                new_past_key_values = present
            else:
                # 后续几层
                new_past_key_values = concat_tensor([new_past_key_values, present]).to(
                    present.device
                )

        # hidden_states
        output = {"hidden_states": outputs[0]}
        if attention_mask is not None:
            output["attention_mask"] = attention_mask
        if position_ids is not None:
            output["positions_ids"] = position_ids
        if past_key_values is not None:
            output["past_key_values"] = past_key_values
        if new_past_key_values is not None:
            output["new_past_key_values"] = new_past_key_values

        return output


class Moss003MoonModel(nn.Module):
    def __init__(self, config):
        super(Moss003MoonModel, self).__init__()
        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = VocabParallelEmbedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([MossBlock(config, i) for i in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def set_norm_precision_to_float32(self):
        self.ln_f.weight.data = self.ln_f.weight.data.to(torch.float32) 
        self.ln_f.bias.data   = self.ln_f.bias.data.to(torch.float32) 

    def forward(
        self,
        input_ids,
        attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            dtype = self.wte.weight.dtype
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        hidden_states = self.drop(inputs_embeds)

        all_hidden_states = ()
        input_dict = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }
        for i, l in enumerate(self.h):
            all_hidden_states += (input_dict["hidden_states"],)
            input_dict.update(l(input_dict))

        hidden_states = self.ln_f(input_dict["hidden_states"])
        all_hidden_states += (hidden_states,)

        past_key_values = None
        if "new_past_key_values" in input_dict:
            past_key_values = input_dict["new_past_key_values"]

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            past_key_values=past_key_values,
        )

    @classmethod
    def pipeline_layers(cls, config):
        if isinstance(config, str):
            config = CollieConfig.from_pretrained(config)

        def pre_forward(input_dict):
            input_ids = input_dict["input_ids"]
            attention_mask = input_dict.get("attention_mask", None)
            batch_size = input_ids.shape[0]
            if attention_mask is not None:
                if batch_size <= 0:
                    raise ValueError("batch_size has to be defined and > 0")
                attention_mask = attention_mask.view(batch_size, -1)
                attention_mask = attention_mask[:, None, None, :]
                dtype = torch.float32
                if "fp16" in config.ds_config:
                    if config.ds_config["fp16"].get("enabled", False):
                        dtype = torch.float16
                if "bf16" in config.ds_config:
                    if config.ds_config["bf_16"].get("enabled", False):
                        dtype = torch.bfloat16
                attention_mask = attention_mask.to(dtype=dtype)
                attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
                input_dict["attention_mask"] = attention_mask
            return input_dict

        wte = dict_as_params("input_ids", "hidden_states")(
            VocabParallelEmbedding, config.vocab_size, config.n_embd
        )
        drop = dict_as_params("hidden_states", "hidden_states")(
            nn.Dropout, config.embd_pdrop
        )
        h = [LayerSpec(MossBlock, config, i) for i in range(config.n_layer)]
        ln_f = dict_as_params("hidden_states", "hidden_states")(
            nn.LayerNorm, config.n_embd, eps=config.layer_norm_epsilon
        )

        layers = [
            ("pre_forward", pre_forward),
            ("wte", wte),
            ("drop", drop),
            ("h", h),
            ("ln_f", ln_f),
        ]

        return layers


class Moss003MoonForCausalLM(CollieModelForCausalLM):
    """
    支持 3D 并行的 Moss-moon 模型。

    :param config: :class:`.CollieConfig`
    """

    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self.transformer = Moss003MoonModel(config)
        self.lm_head = ColumnParallelLinearWithoutBias(config.n_embd, config.vocab_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        inputs_embeds=None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = output.last_hidden_state
        all_hidden_states = output.hidden_states
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    def set_cache(self, use_cache: bool = False):
        self._set_use_cache(self.transformer.h, use_cache)

    def prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = None,
        past_key_values: Optional[list] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
        }

    def clean_cache(self):
        self._clean_hidden_states([*self.transformer.h, self.lm_head])
        self._set_use_cache(self.transformer.h, False)

    @classmethod
    def pipeline_layers(cls, config):
        if isinstance(config, str):
            config = CollieConfig.from_pretrained(config)

        transformer = Moss003MoonModel.pipeline_layers(config)
        lm_head = dict_as_params("hidden_states", "logits")(
            ColumnParallelLMHead,
            config.n_embd,
            config.vocab_size,
        )

        layers = [("transformer", transformer), ("lm_head", lm_head)]

        return layers

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
        从 ``path`` 中加载模型权重。``path`` 中的模型权重应当是 huggingface 格式。

        :param path:
        :param config:
        :param process_exclusion: 是否每个 rank 各自独立、互斥地加载模型权重。在模
            型规模较大时，该参数可以帮助节省内存。
        :return: 一个字典，每个字典都包含当前 rank 上模型需要的权重。
        """
        # Actually Moss only supports `hf` format
        if isinstance(config, str):
            config = CollieConfig.from_pretrained(config)
        io_driver = IODriver.from_protocol(protocol)
        if not io_driver.exists(path) and protocol == "file":
            raise FileNotFoundError(f"folder {path} not found.")

        # 如果开启了进程互斥，那么每个进程都会显示进度条，否则只显示 RANK0 的
        hide_progress = not process_exclusion and env.rank != 0
        for cur_rank in range(dist.get_world_size()):
            if process_exclusion:
                dist.barrier()
            if cur_rank != env.rank:
                continue
            # 如果存在 .index.json 文件的话，此时不同的 pp 进程可以按需加载自己需要的权重
            # 优先加载 model.safetensors.index.json 文件中的权重
            if io_driver.exists(os.path.join(path, "model.safetensors.index.json")):
                index_json_file_path = os.path.join(path, "model.safetensors.index.json")
            elif io_driver.exists(os.path.join(path, "pytorch_model.bin.index.json")):
                index_json_file_path = os.path.join(path, "pytorch_model.bin.index.json")
            else:
                index_json_file_path = None
            # start load
            state_dict = OrderedDict()
            if index_json_file_path is not None and env.is_pipeline:
                # 有 index 且是流水线
                weight_map = json.loads(io_driver.load(index_json_file_path, mode="r"))[
                    "weight_map"
                ]
                # layers 表示当前 rank 自己需要的层
                cur_names = _weight_name_in_current_rank(weight_map.keys())
                weights = set(weight_map[name] for name in cur_names)
            else:
                # 如果没有 .index.json 文件的话，那么就加载所有的权重
                # 优先加载 safetensors 存储的权重
                weights = [
                    weight
                    for weight in io_driver.list(path)
                    if weight.endswith(".safetensors")
                ]
                if len(weights) == 0:
                    # 如果没有 safetensors 文件，那么就加载 bin 文件
                    weights = [
                        weight
                        for weight in io_driver.list(path)
                        if weight.endswith(".bin")
                    ]

            desc = "Loading state dict"
            if process_exclusion:
                desc += f" on pp={env.pp_rank} tp={env.tp_rank} dp={env.dp_rank}"
            for weight in progress(weights, desc, disable=hide_progress):
                part_state_dict = io_driver.load(os.path.join(path, weight), mode="rb")
                state_dict.update(
                    _state_dict_to_load(
                        part_state_dict, env.tp_rank, config.tp_size, process_exclusion
                    )
                )

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
        将模型权重保存到 ``path`` 路径。保存的格式同 ``huggingface`` 格式。

        在保存时会在 dp rank 0 上将所有张量并行的权重合并至 tp_rank 0，然后按照流水
        线的各个阶段分别保存为 sharded checkpoint 的形式。

        :param state_dict: 模型权重
        :param path:
        :param config:
        :param process_exclusion: 是否每个 rank 各自独立、互斥地保存模型权重。在模
            型规模较大时，该参数可以帮助节省内存。
        """
        io_driver = IODriver.from_protocol(protocol)
        if env.rank == 0:
            config.save_pretrained(path, protocol=protocol)

        # gather to tp rank 0
        desc = "Saving state dict"
        # 没有 process_exclusion 的时候就不显示了
        hide_progress = not process_exclusion or env.rank != 0
        for cur_pp_rank in progress(range(env.pp_size), desc, disable=hide_progress):
            if process_exclusion:
                dist.barrier()
            if env.dp_rank != 0:
                continue
            # continue execution when dp_rank == 0
            if cur_pp_rank != env.pp_rank:
                continue
            # continue when pp_rank is available

            state_dict = _state_dict_to_save(
                state_dict, env.tp_rank, config.tp_size, env.tp_group, process_exclusion
            )
            if env.tp_rank != 0:
                continue
            # save at tp_rank 0
            # Save gathered weights
            if env.is_pipeline:
                ckpt_name = (
                    f"pytorch_model-{env.pp_rank+1:05d}-of-{config.pp_size:05d}.bin"
                )
                index_dict = set_index_dict(state_dict, ckpt_name)
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
        dist.barrier()
