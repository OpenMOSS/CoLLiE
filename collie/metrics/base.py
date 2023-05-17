from abc import ABC, abstractmethod
import torch.distributed as dist
from typing import Any

from collie.trainer.trainer import Trainer

class BaseMetric(ABC):
    def __init__(self, 
                 gather_result: bool=False) -> None:
        self.gather_result = gather_result
        
    def construct(self, trainer: Trainer):
        self.trainer = trainer
        
    @abstractmethod
    def update(self, result: Any):
        raise NotImplementedError
    
    def gather(self, result: Any):
        if self.trainer.args.dp_size == 1:
            return [result]
        group = self.trainer.engine.mpu.get_data_parallel_group()
        result_list = [None for _ in range(self.trainer.args.dp_size)]
        dist.all_gather_object(result_list, result, group=group)
        return result_list