from abc import ABC, abstractmethod
import torch.distributed as dist
from typing import Any, Dict, List, Optional

import torch

from collie.utils import apply_to_collection

class BaseMetric(ABC):
    """
    **Metric** 的基类。

    :param gather_result: 在计算 metric 的时候是否自动将各个进程上的输入进行聚合后再输入到 update 之中。
    """
    def __init__(self, 
                 gather_result: bool=False) -> None:
        self.gather_result = gather_result
        
    def construct(self, trainer):
        """
        将 trainer 传入到 metric 中以便于 gather 时候使用
        """
        self.trainer = trainer
    
    def reset(self):
        r"""
        用来重置 init 中定义的值。在调用 get_metric 方法后会自动调用一次该方法
        """
        pass

    @abstractmethod
    def get_metric(self) -> Optional[Dict]:
        raise NotImplementedError()

    @abstractmethod
    def update(self, result: Dict):
        r"""
        :param result: 经过 gather 后的输入。一般为如下格式的字典::
        
                {
                    'logits': [logit1, logit2, ..., logit_dp_size],
                    'labels': [label1, label2, ..., label_dp_size]
                }
            
            其中 ``dp_size`` 为 并行的卡数量
        """
        raise NotImplementedError
    
    def gather(self, result: Dict[str, torch.Tensor]) -> Dict[str, List]:
        r"""
        将不同进程上的 result 数据聚合在一起，使用了 DDP 情况。

        :param result: :class `Trainer` 中 eval_fn 返回的结果。类型为 Dict[str, torch.Tensor]。
            例如::
            
                result = {'logits': logit, 'labels': label}

        :return: 经过 gather 后的结果。类型为 Dict[str, torch.Tensor]。

            当 ``dp_size`` 不为 1 时 (即开启了数据并行的情况下), 会把不同 dp 进程的 ``result`` 按照第一个维度进行拼接。

        """
        if self.trainer.config.dp_size > 1:
            group = self.trainer.engine.mpu.get_data_parallel_group()
            for key in result.keys():
                gather_list = [torch.zeros_like(result[key]).to(result[key].dtype).to(result[key].device) for _ in range(self.trainer.config.dp_size)]
                dist.all_gather(gather_list, result[key], group=group)
                result[key] = torch.cat(gather_list, dim=0)
        return result