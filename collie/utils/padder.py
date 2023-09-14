""" **CoLLie** 中的通用 ``collate_fn`` 构造器
"""
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["ColliePadder"]


class ColliePadder:
    """**CoLLie** 中的通用 ``collate_fn`` 构造器

    :param padding_token: 用于填充模型输入数据 (input_ids) 的 token，为一个 ``Dict`` 决定不同的字段使用不同 id
    :param labels_padding_token: 用于填充模型标签数据 (labels) 的 token
    :param padding_left: 是否在左侧填充
    """

    def __init__(
        self,
        padding_token_id: dict = {"attention_mask": 0, "labels": -100},
        padding_left: bool = False,
    ) -> None:
        self.padding_token_id = padding_token_id
        self.padding_left = padding_left
        self.key = "input_ids"

    def collate_fn(self, batch: Sequence[Any]) -> torch.Tensor:
        """用于填充的 ``collate_fn``

        :param batch: 一个 batch 的数据
        :return: 填充后的 batch
        """
        padding_token_id = self.padding_token_id.get(self.key, 0)
        batch = list(batch)
        if isinstance(batch[0], torch.Tensor):
            pass
        elif isinstance(batch[0], (int, float)):
            batch = [torch.tensor(x).cuda() for x in batch]
        elif isinstance(batch[0], np.ndarray):
            batch = [torch.from_numpy(x).cuda() for x in batch]
        elif isinstance(batch[0], list):
            batch = [torch.tensor(x).cuda() for x in batch]
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
            shape = (
                (torch.tensor(max_shape) - torch.tensor(batch[i].shape)).cpu().tolist()
            )
            if self.padding_left:
                batch[i] = F.pad(
                    batch[i],
                    [shape.pop() if d % 2 == 0 else 0 for d in range(len(shape) * 2)],
                    value=padding_token_id,
                )
            else:
                batch[i] = F.pad(
                    batch[i],
                    [
                        shape.pop() if (d + 1) % 2 == 0 else 0
                        for d in range(len(shape) * 2)
                    ],
                    value=padding_token_id,
                )
        return torch.stack(batch, dim=0).cuda()

    def __call__(self, batch: List[Any]) -> Any:
        padded_batch = None
        if isinstance(batch[0], (torch.Tensor, np.ndarray, list, int, float)):
            padded_batch = self.collate_fn([x for x in batch])
        elif isinstance(batch[0], tuple):
            padded_batch = tuple(
                [self.collate_fn([x[j] for x in batch]) for j in range(len(batch[0]))]
            )
        elif isinstance(batch[0], Dict):
            padded_dict = {}
            for key in batch[0].keys():
                self.key = key
                if isinstance(
                    batch[0][key], (torch.Tensor, np.ndarray, list, int, float)
                ):
                    padded_dict[key] = self.collate_fn([x[key] for x in batch])
                elif isinstance(batch[0][key], tuple) and isinstance(
                    batch[0][key][0], (torch.Tensor, np.ndarray, list)
                ):
                    padded_dict[key] = [
                        self.collate_fn([x[key][j] for x in batch])
                        for j in range(len(batch[0][key]))
                    ]
                elif isinstance(batch[0][key], tuple) and isinstance(
                    batch[0][key][0], str
                ):
                    padded_dict[key] = [x[key] for x in batch]
                else:
                    raise TypeError(f"Unsupported type: {type(batch[0][key])}")
            padded_batch = padded_dict
        else:
            raise TypeError(f"Unsupported type: {type(batch[0])}")
        return padded_batch
