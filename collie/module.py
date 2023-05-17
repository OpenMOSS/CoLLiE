import os
import json
import torch
from inspect import signature
from typing import Optional, Sequence

from torch import nn
from torch import distributed as dist
from megatron.core.tensor_parallel import (ColumnParallelLinear,
                                           RowParallelLinear,
                                           VocabParallelEmbedding)
from megatron.core import parallel_state
from deepspeed.runtime.pipe.module import PipelineModule
from deepspeed.runtime.pipe.topology import (PipeModelDataParallelTopology,
                                             PipelineParallelGrid)
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.runtime.activation_checkpointing import checkpointing
from deepspeed.accelerator import get_accelerator
from transformers.generation.utils import GenerationConfig, GenerationMixin
from transformers.modeling_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from collie.log import logger
from collie.trainer.arguments import Arguments

class ColumnParallelLinearWithoutBias(ColumnParallelLinear):
    def forward(self, input_):
        return super().forward(input_)[0]
    
class ColumnParallelLMHead(ColumnParallelLinearWithoutBias):
    def __init__(self, *args, **kwargs):
        super(ColumnParallelLMHead, self).__init__(*args, **kwargs)
        self.hidden_states = None

    def forward(self, input_):
        if not self.training:
            self.hidden_states = input_
        return super().forward(input_)

class RowParallelLinearWithoutBias(RowParallelLinear):
    def forward(self, input_):
        return super().forward(input_)[0]

class GPTLMLoss(torch.nn.Module):
    def __init__(self, ignore_index=0):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)  # ignore <pad> when compute loss

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(logits.device)
        print(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), shift_logits.shape, shift_labels.shape, self.loss.ignore_index)
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class PipelineModel(PipelineModule):
    def __init__(self,
                 layers,
                 topology,
                 loss_fn=None,
                 seed_layers=False,
                 seed_fn=None,
                 base_seed=1234,
                 partition_method='parameters',
                 activation_checkpoint_interval=0,
                 activation_checkpoint_func=checkpointing.checkpoint,
                 checkpointable_layers=None):
        """
        Rewrite PipelineModule to use megaton's process group
        """
        nn.Module.__init__(self)

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


class CollieCausalLM(nn.Module, GenerationMixin):
    def __init__(self, engine: DeepSpeedEngine, config: GenerationConfig = GenerationConfig()) -> None:
        super().__init__()
        self.config = PretrainedConfig(is_decoder=True)
        self.main_input_name = "input_ids"
        self.device = torch.device("cuda")
        self.engine = engine
        self.args: Arguments = self.engine.module.args
        self.layers = None
        self.communicate_buffer_shape = None
        if isinstance(config, dict):
            config = GenerationConfig.from_dict(config)
        self.generation_config = config
        self._find_layers()
        self._clean_past_key_values()
        
    def forward(self, input_ids: torch.Tensor, past_key_values: Optional[list] = None, *args, **kwargs) -> torch.Tensor:
        if past_key_values is not None:
            self._set_past_key_values(past_key_values)
            start_pos = past_key_values[0][0].shape[1]
        else:
            start_pos = 0
        if self.generation_config.use_cache:
            input_ids = input_ids[:, start_pos:]
        if isinstance(self.engine, PipelineEngine):
            batch = (input_ids, input_ids)
            if self.communicate_buffer_shape is None:
                self.communicate_buffer_shape = batch[0].shape
            else:
                if self.communicate_buffer_shape != batch[0].shape:
                    self.engine.reset_activation_shape()
                    self.engine.total_loss = None
                    self.communicate_buffer_shape = batch[0].shape
            _, logits = self.engine.eval_batch(
                data_iter=iter([batch]),
                return_logits=True,
                compute_loss=False,
                reduce_output=None
            )
            src_rank = self.engine.grid.stage_to_global(self.engine.num_stages - 1)
            if logits is not None:
                logits = logits.detach().clone()
                ndim = torch.tensor([logits.ndim]).int().cuda()
            else:
                ndim = torch.tensor([3]).int().cuda()
            dist.broadcast(tensor=ndim, src=src_rank, group=self.engine.mpu.get_pipe_parallel_group())
            if logits is not None:
                shape = torch.tensor(list(logits.shape)).int().cuda()
            else:
                shape = torch.tensor([0] * int(ndim.data)).int().cuda()
            dist.broadcast(tensor=shape, src=src_rank, group=self.engine.mpu.get_pipe_parallel_group())
            dtype = torch.float32
            try:
                if self.args.ds_config["fp16"]["enabled"]:
                    dtype = torch.float16
            except KeyError:
                pass
            try:
                if self.args.ds_config["bf16"]["enabled"]:
                    dtype = torch.bfloat16
            except KeyError:
                pass
            if logits is None:
                logits = torch.zeros(tuple(shape.cpu().numpy().tolist())).to(dtype).cuda()
            dist.broadcast(tensor=logits, src=src_rank, group=self.engine.mpu.get_pipe_parallel_group())
        else:
            logits = self.engine(input_ids)
            logits = logits.detach().clone()
        if self.generation_config.use_cache:
            past_key_values = self._get_past_key_values()
        else:
            past_key_values = None
            self._clean_past_key_values()
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values
        )
    
    def prepare_inputs_for_generation(self, 
                                      input_ids, 
                                      past_key_values: Optional[list] = None, 
                                      attention_mask: Optional[torch.Tensor] = None, *args, **kwargs):
        return {"input_ids": input_ids, "past_key_values": past_key_values}
    
    def can_generate(self) -> bool:
        return True
    
    def _find_layers(self):
        if isinstance(self.engine.module, PipelineModel):
            self.layers = self.engine.module.forward_funcs
        else:
            for value in self.engine.module.__dict__["_modules"].values():
                if isinstance(value, nn.Sequential) \
                    or isinstance(value, nn.ModuleList) \
                        or isinstance(value, Sequence):
                            for layer in value:
                                if hasattr(layer, "eval"):
                                    layer.eval()
                            if self.layers is None:
                                self.layers = [layer for layer in value]
                            else:
                                self.layers.extend([layer for layer in value])
    
    def _get_past_key_values(self):
        if self.layers is None:
            raise ValueError("The layers of the model is not found.")
        past_key_values = []
        for layer in self.layers:
            if hasattr(layer, "past_key_values") and layer.past_key_values is not None:
                past_key_values.append(layer.past_key_values)
        return past_key_values if len(past_key_values) > 1 else None
    
    def _clean_past_key_values(self):
        if self.layers is None:
            raise ValueError("The layers of the model is not found.")
        for layer in self.layers:
            if hasattr(layer, "past_key_values"):
                object.__setattr__(layer, "past_key_values", None)
        get_accelerator().empty_cache()
                
    def _set_past_key_values(self, past_key_values: list):
        if self.layers is None:
            raise ValueError("The layers of the model is not found.")
        past_key_values = iter(past_key_values)
        for layer in self.layers:
            if hasattr(layer, "past_key_values"):
                object.__setattr__(layer, "past_key_values", next(past_key_values))


class MultiParallelGrid(PipelineParallelGrid):
    """
    Rewrite to use process group from megatron.
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
        self.slice_group = dist.get_process_group_ranks(self.slice_proc_group)
