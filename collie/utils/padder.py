""" **CoLLie** 中的通用 ``collate_fn`` 构造器
"""
from typing import Sequence, Any, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np

__all__ = [
    "ColliePadder"
]

class ColliePadder:
    """ **CoLLie** 中的通用 ``collate_fn`` 构造器
    :param padding_token: 用于填充的 token
    :param padding_left: 是否在左侧填充
    """
    def __init__(self, 
                 padding_token: int=-100,
                 padding_left: bool = False) -> None:
        self.padding_token = padding_token
        self.padding_left = padding_left
        
    def collate_fn(self, batch: Sequence[Any]) -> torch.Tensor:
        """ 用于填充的 ``collate_fn``
        :param batch: 一个 batch 的数据
        :return: 填充后的 batch
        """
        batch = list(batch)
        if isinstance(batch[0], torch.Tensor):
            pass
        elif isinstance(batch[0], np.ndarray):
            batch = [torch.from_numpy(x) for x in batch]
        elif isinstance(batch[0], list):
            batch = [torch.tensor(x) for x in batch]
        else:
            raise TypeError(f"Unsupported type: {type(batch[0])}")
        for i in range(len(batch)):
            sample = batch[i]
            shape = []
            for s in sample.shape:
                if s > 1:
                    shape.append(s)
            if not shape:
                shape.append(1)
            sample = sample.view(*shape)
            batch[i] = sample
        max_shape = max([x.shape for x in batch])
        for i in range(len(batch)):
            shape = (torch.tensor(max_shape) - torch.tensor(batch[i].shape)).cpu().tolist()
            if self.padding_left:
                batch[i] = F.pad(batch[i], [shape.pop() if d % 2 == 0 else 0 for d in range(len(shape) * 2)], value=self.padding_token)
            else:
                batch[i] = F.pad(batch[i], [shape.pop() if (d + 1) % 2 == 0 else 0 for d in range(len(shape) * 2)], value=self.padding_token)
        return torch.stack(batch, dim=0)
    
    def __call__(self, batch: List[Tuple]) -> Any:
        padded_batch = []
        for i in range(len(batch[0])):
            if isinstance(batch[0][i], (torch.Tensor, np.ndarray, list)):
                padded_batch.append(self.collate_fn([x[i] for x in batch]))
            elif isinstance(batch[0][i], Sequence):
                padded_batch.extend([self.collate_fn([x[i][j] for j in range(len(batch[0][i])) for x in batch])])
            else:
                raise TypeError(f"Unsupported type: {type(batch[0][i])}")
        return tuple(padded_batch)