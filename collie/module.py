from megatron.core.tensor_parallel import (ColumnParallelLinear,
                                            RowParallelLinear,
                                            VocabParallelEmbedding)
from deepspeed.runtime.pipe.module import PipelineModule
from deepspeed.runtime.pipe.topology import ProcessTopology, PipeModelDataParallelTopology

import torch
import torch.distributed as dist

from collie.log import logger

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
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
class PipelineModel(PipelineModule):
    def __init__(self, *args, **kwargs):
        for idx, param in enumerate(args):
            if isinstance(param, ProcessTopology):
                pp_size, dp_size, tp_size = param.dims
                if dist.get_world_size() != pp_size * dp_size * tp_size:
                    logger.rank_zero_warning("The world size is not equal to the product of the parallel sizes set."
                                     f"{dist.get_world_size()} != {pp_size} * {dp_size} * {tp_size}.")
                    dp_size = dist.get_world_size() // (tp_size * pp_size)
                    logger.rank_zero_warning("Set dp_size to {dp_size}.")
                args[idx] = PipeModelDataParallelTopology(
                    num_pp=pp_size, 
                    num_dp=dp_size, 
                    num_mp=tp_size)
                break
        for key in kwargs.keys():
            if isinstance(kwargs[key], ProcessTopology):
                pp_size, dp_size, tp_size = kwargs[key].dims
                if dist.get_world_size() != pp_size * dp_size * tp_size:
                    logger.rank_zero_warning("The world size is not equal to the product of the parallel sizes set."
                                     f"{dist.get_world_size()} != {pp_size} * {dp_size} * {tp_size}.")
                    dp_size = dist.get_world_size() // (tp_size * pp_size)
                    logger.rank_zero_warning("Set dp_size to {dp_size}.")
                kwargs[key] = PipeModelDataParallelTopology(
                    num_pp=pp_size, 
                    num_dp=dp_size, 
                    num_mp=tp_size)
                break
        super().__init__(*args, **kwargs)
        
            
    