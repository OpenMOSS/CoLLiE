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
from types import MethodType
from typing import Optional, List, Sequence, Dict

from torch import nn
from torch import distributed as dist
from transformers.generation.configuration_utils import GenerationConfig
from megatron.core.tensor_parallel import (ColumnParallelLinear,
                                           RowParallelLinear,
                                           VocabParallelEmbedding)
from megatron.core import parallel_state
from deepspeed.runtime.pipe.module import PipelineModule
from deepspeed.runtime.pipe.topology import (PipeModelDataParallelTopology,
                                             PipelineParallelGrid)
from deepspeed.runtime.utils import set_random_seed
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.activation_checkpointing import checkpointing
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.pipe.topology import ProcessTopology
from transformers.generation.utils import GenerationConfig, GenerationMixin
from transformers.modeling_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from collie.log import logger
from collie.utils import env, broadcast_tensor, setup_ds_engine

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

    # def forward(self, outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], *args):
    #     """ 计算损失
    #     :param logits: 模型的输出
    #     :param labels: 真实标签
    #     """
    #     labels_mask = None
    #     if isinstance(labels, dict):
    #         if "labels_mask" in labels.keys():
    #             labels_mask = labels["labels_mask"]
    #         labels = labels["labels"]
    #     # TODO key
    #     if isinstance(outputs, dict):
    #         logits = outputs["logits"]
    #     if labels_mask is not None:
    #         labels = labels.masked_fill(labels_mask==1, value=self.ignore_index)
    #     shift_logits = logits[..., :-1, :].contiguous()
    #     shift_labels = labels[..., 1:].contiguous().to(logits.device)
    #     # Flatten the tokens
    #     return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """ 计算损失
        :param logits: 语言模型的输出
        :param labels: 真实标签
        """
        shift_logits = logits[..., :-1, :].contiguous()
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
        self.config = PretrainedConfig(is_decoder=True)
        self.generation_config = GenerationConfig()
        self.main_input_name = "input_ids"
        self.device = torch.device("cuda")
        self.engine_container = [None]
        self.layers = None
        self._find_layers()
        self.is_contrastive_search = False
        self.inner_forward = True
        
    def set_engine(self, engine: DeepSpeedEngine):
        """设置DeepSpeed Engine
        """
        self.engine_container[0] = engine

    def generate(self, *args, **kwargs):
        """开始迭代的生成过程
        """
        if len(self.engine_container) == 0:
            self.engine_container[0] = setup_ds_engine(config=self.collie_config, model=self)[0]
        self.engine_container[0].eval()
        self.inner_forward = False
        res = super().generate(*args, **kwargs)
        self.inner_forward = True
        self._clean_past_key_values()
        self._clean_hidden_states()
        # contrastive learning
        if self.is_contrastive_search:
            src = self.engine_container[0].grid.stage_to_global(self.engine_container[0].num_stages - 1)
            res = broadcast_tensor(res, dtype=res.dtype, src=src,
                                   ndim=len(res.shape), group=env.pp_group)
            self.is_contrastive_search = False
        return res
    
    def contrastive_search(self, *args, **kwargs):
        self.is_contrastive_search = True
        return super().contrastive_search(*args, **kwargs)
        
    def generation_forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                use_cache: bool = True, **kwargs) -> torch.Tensor:
        """ 进行一次流水线模型的前向传播
        """
        if use_cache:
            logger.rank_zero_warning(
                "In Pipeline Parallelism, `use_cache=True` will result in "
                "slowing down the generate process.", once=True
            )
        inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        if position_ids is not None:
            inputs["position_ids"] = position_ids
        inputs["labels"] = inputs["input_ids"]
        self.inner_forward = True
        outputs = self.engine_container[0].generate_batch(inputs, use_cache)
        self.inner_forward = False
        hidden_states = self._get_hidden_states()
        if self.is_contrastive_search:
            # contrastive search 时每个 stage 拿到的 last_hidden_states
            # 不一样，所以广播出去
            src = self.engine_container[0].grid.stage_to_global(self.engine_container[0].num_stages - 1)
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
        return CausalLMOutputWithPast(
            loss=None,
            logits=outputs["logits"],
            past_key_values=self._get_past_key_values(),
            hidden_states=hidden_states,
            attentions=None
        )
    
    def prepare_inputs_for_generation(self, 
                                      input_ids: torch.Tensor,
                                      past_key_values: Optional[list] = None,
                                      attention_mask: Optional[torch.Tensor] = None,
                                      use_cache: bool = False,
                                      **kwargs):
        self._set_use_cache(use_cache)
        if past_key_values is None:
            self._clean_past_key_values()
        else:
            self._set_past_key_values(past_key_values)
        return self.engine_container[0].module.prepare_inputs(
            input_ids=input_ids, past_key_values=past_key_values,
            attention_mask=attention_mask, use_cache=use_cache, **kwargs
        )
    
    def can_generate(self) -> bool:
        """ 判断当前流水线模型是否可以进行生成
        """
        return True
    
    def _find_layers(self):
        """ 从流水线 `engine` 中找到所有的层
        """
        self.layers = self.forward_funcs
    
    def _get_past_key_values(self, attr_name: str="past_key_values"):
        """ 从所有层中获取 `past_key_values`
        """
        past_key_values = []
        for layer in self.layers:
            if hasattr(layer, attr_name) and getattr(layer, attr_name) is not None:
                past_key_values.append(getattr(layer, attr_name))
        return past_key_values if len(past_key_values) > 1 else None
    
    def _clean_past_key_values(self, attr_name: str="past_key_values"):
        """ 清除所有层中的 `past_key_values`
        """
        for layer in self.layers:
            if hasattr(layer, attr_name):
                object.__setattr__(layer, attr_name, None)
                
    def _set_past_key_values(self, past_key_values: List[List[torch.Tensor]], attr_name: str="past_key_values"):
        """ 设置所有层中的 `past_key_values`
        """
        past_key_values = iter(past_key_values)
        for layer in self.layers:
            if hasattr(layer, attr_name):
                object.__setattr__(layer, attr_name, next(past_key_values))
            
    def _get_hidden_states(self, attr_name: str="hidden_states"):
        """ 从所有层中获取 `hidden_states`
        """
        all_hidden_states = []
        for layer in self.layers:
            if hasattr(layer, attr_name) and getattr(layer, attr_name) is not None:
                all_hidden_states.append(getattr(layer, attr_name))
        return all_hidden_states if len(all_hidden_states) > 1 else None
    
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
            logger.rank_zero_warning("Set dp_size to {dp_size}.")
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
        self._layer_specs = list(layers)
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
        
    def forward(self, *args, **kwargs):
        if not self.inner_forward:
            return self.generation_forward(*args, **kwargs)
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
        if self.get_input_embedding()[1] is not None and self.get_input_embedding()[1] in list(self.tied_modules.values()):
            key = list(self.tied_modules.keys())[list(self.tied_modules.values()).index(self.get_input_embedding()[1])]
            self.tied_modules[key] = embedding
        self.forward_funcs[name] = embedding
        
    def set_lm_head(self, name, lm_head):
        if self.get_lm_head()[1] is not None and self.get_lm_head()[1] in list(self.tied_modules.values()):
            key = list(self.tied_modules.keys())[list(self.tied_modules.values()).index(self.get_lm_head()[1])]
            self.tied_modules[key] = lm_head
        self.forward_funcs[name] = lm_head
        
        
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