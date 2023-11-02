""" **CoLLie** 中的可复用模块
"""
__all__ = [
    'ColumnParallelLinearWithoutBias',
    'ColumnParallelLMHead',
    'RowParallelLinearWithoutBias',
    'GPTLMLoss',
    'PipelineGenerationMixin',
]

import os
import copy
import types
import json
import torch
import warnings
import inspect
from collections import OrderedDict
from types import MethodType
from typing import Mapping, Optional, List, Sequence, Dict, Any, Tuple

from torch import nn
from torch import distributed as dist
from transformers.generation.configuration_utils import GenerationConfig
from megatron.core.tensor_parallel import (ColumnParallelLinear,
                                           RowParallelLinear,
                                           VocabParallelEmbedding)
from megatron.core import parallel_state
from deepspeed.runtime.pipe.module import PipelineModule, LayerSpec, TiedLayerSpec
from deepspeed.runtime.pipe.topology import (PipeModelDataParallelTopology,
                                             PipelineParallelGrid)
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.activation_checkpointing import checkpointing
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.pipe.topology import ProcessTopology
from deepspeed.runtime import utils as ds_utils
from transformers.generation.utils import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from collie.log import logger
from collie.utils import env, broadcast_tensor, setup_ds_engine, stack_tensor, concat_tensor

class ColumnParallelLinearWithoutBias(ColumnParallelLinear):
    """重写 ``megatron`` 提供的列并行全连接层以去掉结果中的 ``bias``。
    
    在 ``tp_size`` 为 1 时可以返回普通的全连接层（支持 `peft` 中的 `lora` 方法替换全连接层）
    """
    def forward(self, input_):
        return super().forward(input_)[0]
    
    def __new__(cls, *args, **kwargs):
        if env.tp_size == 1:
            naive_kwargs = {}
            if "output_size" in kwargs:
                naive_kwargs["output_size"] = kwargs["output_size"]
            if "input_size" in kwargs:
                naive_kwargs["input_size"] = kwargs["input_size"]
            if "bias" in kwargs:
                naive_kwargs["bias"] = kwargs["bias"]
            return nn.Linear(*args, **naive_kwargs)
        return super().__new__(cls)
    
class LinearWithHiddenStates(nn.Linear):
    """重写 ``torch.nn.Linear`` 以支持在 ``eval`` 时保存隐藏状态（用于流水线并行中）
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.hidden_states = None
        
    def forward(self, input_):
        if not self.training:
            self.hidden_states = input_
        else:
            self.hidden_states = None
        return super().forward(input_)
    
class ColumnParallelLMHead(ColumnParallelLinearWithoutBias):
    """
    重写 ``megatron`` 提供的列并行全连接层以支持在 ``eval`` 时保存隐藏状态（用于流水
    线并行），在 ``tp_size`` 为 1 时返回普通的全连接层（支持 ``peft`` 中的 ``lora``
    方法替换全连接层）。
    """
    def __init__(self, *args, **kwargs):
        super(ColumnParallelLMHead, self).__init__(*args, **kwargs)
        self.hidden_states = None

    def forward(self, input_):
        if not self.training:
            self.hidden_states = input_
        else:
            self.hidden_states = None
        return super().forward(input_)
    
    def __new__(cls, *args, **kwargs):
        if env.tp_size == 1:
            naive_kwargs = {}
            if "output_size" in kwargs:
                naive_kwargs["output_size"] = kwargs["output_size"]
            if "input_size" in kwargs:
                naive_kwargs["input_size"] = kwargs["input_size"]
            if "bias" in kwargs:
                naive_kwargs["bias"] = kwargs["bias"]
            return LinearWithHiddenStates(*args, **naive_kwargs)
        return super().__new__(cls)

class RowParallelLinearWithoutBias(RowParallelLinear):
    """
    重写 ``megatron`` 提供的行并行全连接层以去掉结果中的 ``bias``。在 ``tp_size``
    为 1 时返回普通的全连接层（支持 ``peft`` 中的 ``lora`` 方法替换全连接层）
    """
    def forward(self, input_):
        return super().forward(input_)[0]
    
    def __new__(cls, *args, **kwargs):
        if env.tp_size == 1:
            naive_kwargs = {}
            if "output_size" in kwargs:
                naive_kwargs["output_size"] = kwargs["output_size"]
            if "input_size" in kwargs:
                naive_kwargs["input_size"] = kwargs["input_size"]
            if "bias" in kwargs:
                naive_kwargs["bias"] = kwargs["bias"]
            return nn.Linear(*args, **naive_kwargs)
        return super().__new__(cls)

class GPTLMLoss(torch.nn.Module):
    """最基本的 GPT 语言模型的损失函数。

    :param ignore_index: 忽略的标签的 ``index``，默认为 **-100**
    """
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)  # ignore <pad> when compute loss
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """ 计算损失
        :param logits: 语言模型的输出
        :param labels: 真实标签
        """
        shift_logits = logits[..., :-1, :].float().contiguous()
        shift_labels = labels[..., 1:].contiguous().to(logits.device)
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class PipelineGenerationMixin(GenerationMixin):
    """
    重写 ``transformers`` 提供的 ``GenerationMixin`` 以支持 **CoLLie** 中的流水线
    模型。

    :param engine: `DeepSpeedEngine` 实例，可由 :meth:`~collie.utils.\
        setup_ds_engine` 函数生成
    """
    def __init__(self) -> None:
        self.config = self.collie_config.model_config
        self.config.is_decoder=True
        self.generation_config = GenerationConfig()
        self.main_input_name = "input_ids"
        self.device = torch.device("cuda")
        self.engine_container = []
        self.layers = None
        self._find_layers()
        self.is_contrastive_search = False
        
    def set_engine(self, engine: DeepSpeedEngine):
        """设置DeepSpeed Engine
        """
        self.engine_container.append(engine)

    def generate(self, *args, **kwargs):
        """开始迭代的生成过程
        """
        if len(self.engine_container) == 0:
            self.engine_container.append(setup_ds_engine(config=self.collie_config, model=self)[0])
        self.engine_container[-1].eval()
        self.forward_type = "generate"
        res = super().generate(*args, **kwargs)
        self._clean_hidden_states()
        # contrastive learning
        if self.is_contrastive_search:
            src = self.engine_container[-1].grid.stage_to_global(self.engine_container[-1].num_stages - 1)
            res = broadcast_tensor(res, dtype=res.dtype, src=src,
                                   ndim=len(res.shape), group=env.pp_group)
            self.is_contrastive_search = False
        return res
    
    def contrastive_search(self, *args, **kwargs):
        self.is_contrastive_search = True
        return super().contrastive_search(*args, **kwargs)
        
    def generate_forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                use_cache: bool = True, 
                past_key_values: Optional[Tuple[torch.Tensor]] = None,
                **kwargs) -> torch.Tensor:
        """ 进行迭代的流水线模型的前向传播（生成）
        """
        if use_cache:
            logger.rank_zero_warning(
                "In Pipeline Parallelism, `use_cache=True` will result in "
                "slowing down the generate process.", once=True
            )
        inputs = {}
        if input_ids is not None:
            inputs["input_ids"] = input_ids
            inputs["labels"] = inputs["input_ids"]
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        if position_ids is not None:
            inputs["position_ids"] = position_ids
        if inputs_embeds is not None:
            inputs["inputs_embeds"] = inputs_embeds
        if past_key_values is not None:
            # TODO 这里先按照输入的 past key values 是没有 split 版本的处理
            if not isinstance(past_key_values, torch.Tensor):
                # stack 起来
                stack_past_key_values = [None for _ in range(len(past_key_values))]
                for i, layer_past in enumerate(past_key_values):
                    if not isinstance(layer_past, torch.Tensor):
                        stack_past_key_values[i] = stack_tensor(layer_past)
                    else:
                        stack_past_key_values[i] = layer_past
                del past_key_values
                past_key_values = stack_tensor(stack_past_key_values)
            inputs["past_key_values"] = past_key_values

        outputs = self.engine_container[-1].generate_batch(inputs)
        hidden_states = self._get_hidden_states()
        if self.is_contrastive_search:
            # contrastive search 时每个 stage 拿到的 last_hidden_states
            # 不一样，所以广播出去
            src = self.engine_container[-1].grid.stage_to_global(self.engine_container[-1].num_stages - 1)
            if hidden_states is not None:
                last_hidden_states = hidden_states[-1]
            else:
                # 防止流水线段数过多时某些 stage 没有分到 block
                hidden_states = []
                last_hidden_states = None
            last_hidden_states = broadcast_tensor(
                last_hidden_states, src=src, group=env.pp_group
            )
            hidden_states.append(last_hidden_states)
        
        # 还原 past key values
        if "new_past_key_values" in outputs:
            past_key_values = outputs["new_past_key_values"]
        else:
            past_key_values = None

        return CausalLMOutputWithPast(
            loss=None,
            logits=outputs["logits"],
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=None
        )
        
    def train_forward(self,
                labels: torch.Tensor,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None, 
                past_key_values: Optional[Tuple[torch.Tensor]] = None,
                **kwargs) -> torch.Tensor:
        """ 进行一次流水线模型的正反向传播
        """
        inputs = {}
        if input_ids is not None:
            inputs["input_ids"] = input_ids
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        if position_ids is not None:
            inputs["position_ids"] = position_ids
        if inputs_embeds is not None:
            inputs["inputs_embeds"] = inputs_embeds
        if past_key_values is not None:
            # prefix tuning
            # TODO 这里先按照输入的 past key values 是没有 split 版本的处理
            if not isinstance(past_key_values, torch.Tensor):
                # stack 起来
                past_key_values = torch.stack(past_key_values)
            inputs["past_key_values"] = past_key_values
        inputs["labels"] = labels
        loss = self.engine_container[-1].train_batch(inputs)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )
        
    def eval_forward(self,
                labels: torch.Tensor,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None, 
                past_key_values: Optional[Tuple[torch.Tensor]] = None,
                **kwargs) -> torch.Tensor:
        """ 进行一次流水线模型的正反向传播
        """
        inputs = {}
        if input_ids is not None:
            inputs["input_ids"] = input_ids
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        if position_ids is not None:
            inputs["position_ids"] = position_ids
        if inputs_embeds is not None:
            inputs["inputs_embeds"] = inputs_embeds
        if past_key_values is not None:
            # TODO 这里先按照输入的 past key values 是没有 split 版本的处理
            if not isinstance(past_key_values, torch.Tensor):
                # stack 起来
                stack_past_key_values = [None for _ in range(len(past_key_values))]
                for i, layer_past in enumerate(past_key_values):
                    if not isinstance(layer_past, torch.Tensor):
                        stack_past_key_values[i] = stack_tensor(layer_past)
                    else:
                        stack_past_key_values[i] = layer_past
                del past_key_values
                past_key_values = stack_tensor(stack_past_key_values)
            inputs["past_key_values"] = past_key_values

        inputs["labels"] = labels
        outputs = self.engine_container[-1].eval_batch(inputs)
        hidden_states = self._get_hidden_states()
        # 还原 past key values
        if "new_past_key_values" in outputs:
            past_key_values = outputs["new_past_key_values"]
        else:
            past_key_values = None
        return CausalLMOutputWithPast(
            loss=None,
            logits=outputs["logits"],
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=None
        )
    
    def prepare_inputs_for_generation(self, 
                                      input_ids: Optional[torch.Tensor] = None,
                                      inputs_embeds: Optional[torch.Tensor] = None,
                                      past_key_values: Optional[list] = None,
                                      attention_mask: Optional[torch.Tensor] = None,
                                      use_cache: bool = False,
                                      **kwargs):
        self._set_use_cache(use_cache)
        if past_key_values is not None:
            if not isinstance(past_key_values, torch.Tensor) and None in past_key_values:
                past_key_values = None
        return self.engine_container[-1].module.prepare_inputs(
            input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values,
            attention_mask=attention_mask, use_cache=use_cache, **kwargs
        )

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            warnings.warn(f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)")
            # raise ValueError(
            #     f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
            #     " generate arguments will also show up in this list)"
            # )
    
    def can_generate(self) -> bool:
        """ 判断当前流水线模型是否可以进行生成
        """
        return True
    
    def _find_layers(self):
        """ 从流水线 `engine` 中找到所有的层
        """
        self.layers = self.forward_funcs
            
    def _get_hidden_states(self, attr_name: str="hidden_states"):
        """ 从所有层中获取 `hidden_states`
        """
        all_hidden_states = []
        for layer in self.layers:
            if hasattr(layer, attr_name):
                all_hidden_states.append(getattr(layer, attr_name))
        return tuple(all_hidden_states) if None not in all_hidden_states else None
    
    def _clean_hidden_states(self, attr_name: str="hidden_states"):
        """ 清除所有层中的 `hidden_states`
        """
        for layer in self.layers:
            if hasattr(layer, attr_name):
                object.__setattr__(layer, attr_name, None)
                
    def _set_hidden_states(self, hidden_states: List[torch.Tensor], attr_name: str="hidden_states"):
        """ 设置所有层中的 `hidden_states`
        """
        hidden_states = iter(hidden_states)
        for layer in self.layers:
            if hasattr(layer, attr_name):
                object.__setattr__(layer, attr_name, next(hidden_states))   
                
    def _set_use_cache(self, use_cache: bool=True, attr_name: str="use_cache"):
        """ 设置所有层中的 `use_cache`
        """ 
        for layer in self.layers:
            if hasattr(layer, attr_name):
                object.__setattr__(layer, attr_name, use_cache)

class PipelineModel(PipelineModule, PipelineGenerationMixin):
    """
    重写 ``megatron`` 提供的 ``PipelineModule`` 以支持 **CoLLie** 中的
    :class:`.Trainer`。

    :param layers: 分层化的模型，为 `callable` 组成的 `list`
    :param topology: 模型的拓扑结构
    :param loss_fn: 损失函数
    :param seed_layers: 是否对每一层使用不同的随机种子
    :param seed_fn: 随机种子生成函数
    :param base_seed: 随机种子的基数
    :param partition_method: 模型分割方法
    :param activation_checkpoint_interval: 激活检查点间隔
    :param activation_checkpoint_func: 激活检查点函数
    :param checkpointable_layers: 可检查点的层
    """
    def __init__(self,
                 config,
                 layers: Sequence[callable],
                 topology: ProcessTopology,
                 loss_fn: callable=None,
                 seed_layers: bool=False,
                 seed_fn: callable=None,
                 base_seed: int=1234,
                 partition_method: str='parameters',
                 activation_checkpoint_interval: int=0,
                 activation_checkpoint_func: callable=checkpointing.checkpoint,
                 checkpointable_layers=None):
        """
        Rewrite PipelineModule to use megaton's process group
        """
        nn.Module.__init__(self)
        self.collie_config = config
        if topology is None:
            raise RuntimeError('must provide topology')

        self.micro_offset = 0

        self.loss_fn = loss_fn

        self.checkpointable_layers = checkpointable_layers
        if checkpointable_layers is not None:
            assert isinstance(checkpointable_layers, list), "param `checkpointable_layers` must be type of list."

        self.seed_layers = seed_layers
        self.seed_fn = seed_fn
        self.base_seed = base_seed
        if dist.get_rank() == 0:
            try:
                seed_str = self.seed_fn.__name__
            except AttributeError:
                seed_str = None
            print(f'SEED_LAYERS={self.seed_layers} BASE_SEED={self.base_seed} SEED_FN={seed_str}')

        # Setup world info
        self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
        self.global_rank = dist.get_rank(group=self.world_group)
        self.world_size = dist.get_world_size(group=self.world_group)
        self.local_rank = int(os.environ.get("LOCAL_RANK", None))
        assert self.local_rank != None

        pp_size, dp_size, tp_size = topology.dims
        if int(os.environ.get('WORLD_SIZE')) != pp_size * dp_size * tp_size:
            logger.rank_zero_warning("The world size is not equal to the product of the parallel sizes set."
                                f"{int(os.environ.get('WORLD_SIZE'))} != {pp_size} * {dp_size} * {tp_size}.")
            dp_size = int(os.environ.get('WORLD_SIZE')) // (tp_size * pp_size)
            logger.rank_zero_warning(f"Set dp_size to {dp_size}.")
        topology = PipeModelDataParallelTopology(
            num_pp=pp_size, 
            num_dp=dp_size, 
            num_mp=tp_size)
        self._topo = topology
        self.num_stages = self._topo.get_dim('pipe')

        # Construct communicators for pipeline topology
        # Replace with our grid
        self._grid = MultiParallelGrid(self._topo)

        self.stage_id = self._topo.get_coord(self.global_rank).pipe

        # Initialize partition information
        self._layer_prefix, self._layer_specs = self._flatten_layers(layers)
        assert len(self._layer_prefix) == len(self._layer_specs)
        assert len(self._layer_prefix) == len(set(self._layer_prefix))
        self._num_layers = len(self._layer_specs)
        self._local_start = 0
        self._local_stop = None
        self._partition_layers(method=partition_method)

        self.forward_funcs = []
        self.fwd_map = {}
        self.tied_modules = nn.ModuleDict()
        self.tied_weight_attrs = {}

        self._build()
        self.to(get_accelerator().device_name(self.local_rank))

        self.tied_comms = self._index_tied_modules()
        self._synchronize_tied_weights()

        self.activation_checkpoint_interval = activation_checkpoint_interval
        self.activation_checkpoint_func = activation_checkpoint_func

        os.environ["COLLIE_PP_PARTS"] = json.dumps(self.parts)
        os.environ["COLLIE_PP_RANK"] = str(self.stage_id)
        os.environ["COLLIE_DP_RANK"] = str(self._grid.data_parallel_id)
        
        PipelineGenerationMixin.__init__(self)
        
        self.inner_forward = False
        self.forward_type = "train" # train, eval, generate
        self.skip_input_embedding()

    def _flatten_layers(self, layers):
        # layers: list of tuple/layer
        _layers = []
        _names = []
        for i, layer in enumerate(layers):
            if isinstance(layer, tuple):
                assert len(layer) == 2, len(layer)
                # name, layer or name, list[layer]
                if isinstance(layer[1], list):
                    _n, _l = self._flatten_layers(layer[1])
                    _layers.extend(_l)
                    _names.extend([f"{layer[0]}.{n}" for n in _n])
                else:
                    _names.append(str(layer[0]))
                    _layers.append(layer[1])
            else:
                assert not isinstance(layer, list)
                # func, Module, LayerSpec, TiedLayerSpec
                _names.append(str(len(_names)))
                _layers.append(layer)

        return _names, _layers
    
    def name_to_pipeline(self, name):
        for idx, prefix in enumerate(self._layer_prefix):
            if not name.startswith(prefix + "."):
                continue
            _layer = self._layer_specs[idx]
            if isinstance(_layer, TiedLayerSpec):
                # {prefix}.weight -> tied_modules.{key}.weight
                return name.replace(prefix, f"tied_modules.{_layer.key}", 1)
            else:
                return name.replace(prefix, str(idx), 1)

    def name_from_pipeline(self, name, ):
        name_split = name.split(".")
        if name_split[0] == "tied_modules":
            # 当前 rank 一个 TiedLayerSpec 对应的层可能不唯一，返回一个 list 或者 string
            name_pp = []
            tied_key = name_split[1]
            name_pp_suffix = ".".join(name_split[2:])
            for i in range(self._local_start, self._local_stop):
                if not isinstance(self._layer_specs[i], TiedLayerSpec):
                    continue
                if self._layer_specs[i].key == tied_key:
                    name_pp.append(f"{self._layer_prefix[i]}.{name_pp_suffix}")
            return name_pp if len(name_pp) > 1 else name_pp[0]

        idx = int(name_split[0])
        name_split[0] = f"{self._layer_prefix[idx]}"
        if isinstance(self._layer_specs[idx], TiedLayerSpec):
            # tied_modules.{key}.weight -> {prefix}.weight
            name_split.pop(1)
        return ".".join(name_split)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        super().state_dict(*args, destination=destination, prefix="", keep_vars=keep_vars)
        for key in list(destination.keys()):
            key_pp = self.name_from_pipeline(key)
            if isinstance(key_pp, list):
                for _key_pp in key_pp:
                    destination[prefix + _key_pp] = destination[key].detach().clone()
                destination.pop(key)
            else:
                key_pp = prefix + key_pp
                destination[key_pp] = destination.pop(key)
        return destination

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True):
        for key in list(state_dict.keys()):
            key_pp = self.name_to_pipeline(key)
            state_dict[key_pp] = state_dict.pop(key)
        super().load_state_dict(state_dict, strict)

    def forward(self, *args, **kwargs):
        if not self.inner_forward:
            if self.forward_type == "generate":
                return self.generate_forward(*args, **kwargs)
            elif self.forward_type == "train":
                return self.train_forward(*args, **kwargs)
            elif self.forward_type == "eval":
                return self.eval_forward(*args, **kwargs)
            else:
                raise RuntimeError("Wrong forward type!")
        else:
            # hack: super(PipelineModel, self).forward 只能接收一个参数，这个参数的类型是 dict
            if "input_ids" in kwargs.keys() and isinstance(kwargs["input_ids"], dict):
                return super(PipelineModel, self).forward(kwargs["input_ids"])
            else:
                return super(PipelineModel, self).forward(*args, **kwargs)
        
    def get_input_embedding(self):
        if env.pp_rank != 0:
            return None, None
        for name, layer in enumerate(self.forward_funcs):
            if isinstance(layer, (nn.Embedding, VocabParallelEmbedding)):
                return name, layer
        return None, None
    
    def get_lm_head(self):
        if env.pp_rank != env.pp_size - 1:
            return None, None
        for name, layer in enumerate(reversed(self.forward_funcs)):
            if isinstance(layer, (ColumnParallelLinear, nn.Linear)):
                return len(self.forward_funcs) - name - 1, layer
        return None, None
    
    def set_input_embedding(self, name, embedding):
        if self.get_input_embedding()[1] is not None:
            if self.get_input_embedding()[1] in list(self.tied_modules.values()):
                key = list(self.tied_modules.keys())[list(self.tied_modules.values()).index(self.get_input_embedding()[1])]
                self.tied_modules[key] = embedding
            elif self.get_input_embedding()[1] in list(self._modules.values()):
                self.add_module(str(name), embedding)
        self.forward_funcs[name] = embedding

    def set_lm_head(self, name, lm_head):
        if self.get_lm_head()[1] is not None:
            if self.get_lm_head()[1] in list(self.tied_modules.values()):
                key = list(self.tied_modules.keys())[list(self.tied_modules.values()).index(self.get_lm_head()[1])]
                self.tied_modules[key] = lm_head
            elif self.get_lm_head()[1] in list(self._modules.values()):
                self.add_module(str(name), lm_head)
        self.forward_funcs[name] = lm_head
        
    def tie_weights(self):
        pass
    
    def skip_input_embedding(self):
        input_embedding = self.get_input_embedding()[1]
        if input_embedding is not None and isinstance(input_embedding, nn.Module):
            raw_foward = input_embedding.forward
            def _forward(self, inputs):
                if isinstance(inputs, dict):
                    if "inputs_embeds" in inputs.keys():
                        inputs["hidden_states"] = inputs.pop("inputs_embeds")
                        return inputs
                    else:
                        return raw_foward(inputs)
                else:
                    if hasattr(self, "raw_forward"):
                        return self.raw_forward(inputs)
                    return raw_foward(inputs)
            object.__setattr__(input_embedding, "forward", MethodType(_forward, input_embedding))
        
        
class MultiParallelGrid(PipelineParallelGrid):
    """
    重写以支持 ``megatron`` 中的张量并行进程组
    """ 
    def __init__(self, topology):
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._topo = topology

        self.data_parallel_size = max(self._topo.get_dim('data'), 1)
        self.pipe_parallel_size = max(self._topo.get_dim('pipe'), 1)
        self.model_parallel_size = max(self._topo.get_dim('model'), 1)
        self.slice_parallel_size = self.model_parallel_size
        assert self._is_grid_valid(), "Invalid Grid"

        self.stage_id = self.get_stage_id()
        self.data_parallel_id = self.get_data_parallel_id()

        # Create new ProcessGroups for all model parallelism. DeepSpeedLight uses these
        # to detect overflow, etc.
        self.ds_model_proc_group = parallel_state.get_model_parallel_group()
        self.ds_model_world_size = self.ds_model_proc_group.size()
        self.ds_model_rank = self.ds_model_proc_group.rank()
        assert self.ds_model_rank > -1
        assert self.ds_model_proc_group is not None

        # Create new ProcessGroup for gradient all-reduces - these are the data parallel groups
        self.dp_group = list(parallel_state._DATA_PARALLEL_GLOBAL_RANKS)
        self.dp_proc_group = parallel_state.get_data_parallel_group()

        self.is_first_stage = (self.stage_id == 0)
        self.is_last_stage = (self.stage_id == (self.pipe_parallel_size - 1))

        self.p2p_groups = self._build_p2p_groups()

        # Create new ProcessGroup for pipeline collectives - these are pipe parallel groups
        self.pp_group = list(parallel_state._PIPELINE_GLOBAL_RANKS)
        self.pp_proc_group = parallel_state.get_pipeline_model_parallel_group()

        # Create new ProcessGroup for model (tensor-slicing) collectives
        self.slice_proc_group = parallel_state.get_tensor_model_parallel_group()
        self.slice_group = list(dist.distributed_c10d._pg_group_ranks[self.slice_proc_group].keys())