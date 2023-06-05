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
    
    def gather(self, result: Dict) -> Dict[str, List]:
        r"""
        将不同进程上的 result 数据聚合在一起，使用了 DDP 情况。

        :param result: :class `Trainer` 中 eval_fn 返回的结果。类型为 Dict。
            例如::
            
                result = {'logits': logit, 'labels': label}

        :return: 经过 gather 后的结果。假设 ``result`` 为::
            
                result = {'logits': logit, 'labels': label}

            其根据 ``dp_size`` 的值有如下两种情况：

            * dp_size 为 ``1``, 其返回值为::
            
                {'logits': [logit], 'labels': [label]}

            * dp_size > 1, 其返回值为::
            
                {
                    'logits': [logit1, logit2, ..., logit_dp_size],
                    'labels': [label1, label2, ..., label_dp_size]
                }

        """
        def _to_device(tensor, device):
            return tensor.contiguous().to(device)

        gather_result = {key_name: [] for key_name in result.keys()}
        if self.trainer.config.dp_size == 1:
            result_list = [result]
        else:
            group = self.trainer.engine.mpu.get_data_parallel_group()
            result = apply_to_collection(result, torch.Tensor, _to_device,
                                         device=torch.device("cpu"))
            result_list = [None for _ in range(self.trainer.config.dp_size)]
            dist.all_gather_object(result_list, result, group=group)
        for result_dp in result_list:
            for name, value in result_dp.items():
                gather_result[name].append(value)

        return gather_result