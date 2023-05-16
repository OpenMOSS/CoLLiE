from megatron.core.tensor_parallel import (ColumnParallelLinear,
                                           RowParallelLinear,
                                           VocabParallelEmbedding)
from deepspeed.runtime.pipe.module import PipelineModule
from deepspeed.runtime.pipe.topology import ProcessTopology, PipeModelDataParallelTopology
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.pipe.engine import PipelineEngine
from transformers.generation.utils import GenerationConfig, GenerateOutput, GenerationMixin
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

import os
import json
import torch
import random
from torch import nn
import torch.distributed as dist
from typing import Callable, Union, Tuple, Optional, Sequence

from collie.log import logger
from collie.trainer.arguments import Arguments, load_config

class ColumnParallelLinearWithoutBias(ColumnParallelLinear):
    def forward(self, input_):
        return super().forward(input_)[0]
    
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
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
class PipelineModel(PipelineModule):
    def __init__(self, *args, **kwargs):
        for idx, param in enumerate(args):
            if isinstance(param, ProcessTopology):
                pp_size, dp_size, tp_size = param.dims
                if int(os.environ.get('WORLD_SIZE')) != pp_size * dp_size * tp_size:
                    logger.rank_zero_warning("The world size is not equal to the product of the parallel sizes set."
                                     f"{int(os.environ.get('WORLD_SIZE'))} != {pp_size} * {dp_size} * {tp_size}.")
                    dp_size = int(os.environ.get('WORLD_SIZE')) // (tp_size * pp_size)
                    logger.rank_zero_warning("Set dp_size to {dp_size}.")
                args[idx] = PipeModelDataParallelTopology(
                    num_pp=pp_size, 
                    num_dp=dp_size, 
                    num_mp=tp_size)
                break
        for key in kwargs.keys():
            if isinstance(kwargs[key], ProcessTopology):
                pp_size, dp_size, tp_size = kwargs[key].dims
                if int(os.environ.get('WORLD_SIZE')) != pp_size * dp_size * tp_size:
                    logger.rank_zero_warning("The world size is not equal to the product of the parallel sizes set."
                                     f"{int(os.environ.get('WORLD_SIZE'))} != {pp_size} * {dp_size} * {tp_size}.")
                    dp_size = int(os.environ.get('WORLD_SIZE')) // (tp_size * pp_size)
                    logger.rank_zero_warning("Set dp_size to {dp_size}.")
                kwargs[key] = PipeModelDataParallelTopology(
                    num_pp=pp_size, 
                    num_dp=dp_size, 
                    num_mp=tp_size)
                break
        super().__init__(*args, **kwargs)
        os.environ["COLLIE_PP_PARTS"] = json.dumps(self.parts)
        os.environ["COLLIE_PP_RANK"] = str(self.stage_id)
        os.environ["COLLIE_DP_RANK"] = str(self._grid.data_parallel_id)


class CollieCasualLM(nn.Module, GenerationMixin):
    def __init__(self, engine: DeepSpeedEngine, config: GenerationConfig = GenerationConfig()) -> None:
        super().__init__()
        self.config = PretrainedConfig(is_decoder=True)
        self.main_input_name = "input_ids"
        self.device = torch.device("cuda")
        self.engine = engine
        self.args: Arguments = self.engine.module.args
        self.layers = None
        self.communicate_buffer_shape = None
        self.generation_config = config
        self._find_layers()
        self._clean_past_key_values()
        
    def forward(self, input_ids: torch.Tensor, past_key_values: Optional[list] = None, *args, **kwargs) -> torch.Tensor:
        if past_key_values is not None:
            self._set_past_key_values(past_key_values)
            start_pos = past_key_values[0][0].shape[1]
        else:
            start_pos = 0
        if isinstance(self.engine, PipelineEngine):
            batch = (input_ids[:, start_pos:], input_ids[:, start_pos:])
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
            logits = self.engine(input_ids[:, start_pos:])
            logits = logits.detach().clone()
        past_key_values = self._get_past_key_values()
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
                
    def _set_past_key_values(self, past_key_values: list):
        if self.layers is None:
            raise ValueError("The layers of the model is not found.")
        past_key_values = iter(past_key_values)
        for layer in self.layers:
            if hasattr(layer, "past_key_values"):
                object.__setattr__(layer, "past_key_values", next(past_key_values))