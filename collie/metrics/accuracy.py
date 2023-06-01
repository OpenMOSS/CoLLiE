from typing import Dict, Optional, List

import torch
import numpy as np

from collie.metrics.base import BaseMetric
from collie.log import logger

def seq_len_to_mask(seq_len, max_len: Optional[int]=None):
    r"""
    将一个表示 ``sequence length`` 的一维数组转换为二维的 ``mask`` ，不包含的位置为 **0**。

    .. code-block::
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])

    :param seq_len: 大小为 ``(B,)`` 的长度序列；
    :param int max_len: 将长度补齐或截断到 ``max_len``。默认情况（为 ``None``）使用的是 ``seq_len`` 中最长的长度；
        但在 :class:`torch.nn.DataParallel` 等分布式的场景下可能不同卡的 ``seq_len`` 会有区别，所以需要传入
        ``max_len`` 使得 ``mask`` 的补齐或截断到该长度。
    :return: 大小为 ``(B, max_len)`` 的 ``mask``， 元素类型为 ``bool`` 或 ``uint8``
    """
    max_len = int(max_len) if max_len is not None else int(seq_len.max())

    assert seq_len.ndim == 1, f"seq_len can only have one dimension, got {seq_len.ndim == 1}."
    batch_size = seq_len.shape[0]
    broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
    mask = broad_cast_seq_len < seq_len.unsqueeze(1)
    return mask

    
class AccuracyMetric(BaseMetric):
    """
    计算准确率的 metric

    :param gather_result: 在计算 metric 的时候是否自动将各个进程上的输入进行聚合后再输入到 update 之中。
    """

    def __init__(self, gather_result: bool=False):
        super().__init__(gather_result=gather_result)
        self.correct = 0
        self.total = 0
    
    def reset(self):
        """
        重置参数
        """
        self.correct = 0
        self.total = 0

    def get_metric(self)->Dict:
        r"""
        :meth:`get_metric` 函数将根据 :meth:`update` 函数累计的评价指标统计量来计算最终的评价结果。

        :return: 字典形式的评测结果，例如::

                {"acc": float, 'total': float, 'correct': float}
        """
        evaluate_result = {'acc': round(self.correct / (self.total + 1e-12), 6),
                           'total': self.total, 'correct': self.correct}
        return evaluate_result
    
    def update(self, result:Dict):
        r"""
        :meth:`update` 函数将针对一个批次的预测结果做评价指标的累计。

        :param result: 类型为 Dict 且 keys 至少包含["pred", "target"]

            * pred - 预测的 tensor, tensor 的形状可以是 ``torch.Size([B,])`` 、``torch.Size([B, n_classes])`` 、
              ``torch.Size([B, max_len])`` 或 ``torch.Size([B, max_len, n_classes])``
            * target - 真实值的 tensor, tensor 的形状可以是 ``torch.Size([B,])`` 、``torch.Size([B, max_len])``
              或 ``torch.Size([B, max_len])``
            * seq_len - 序列长度标记, 标记的形状可以是 ``None``,  或者 ``torch.Size([B])`` 。
              如果 mask 也被传进来的话 ``seq_len`` 会被忽略
        """
        if "pred" not in result.keys():
            raise ValueError(f"pred not in result!")
        if "target" not in result.keys():
            raise ValueError(f"target not in result!")
        pred = result.get("pred")
        target = result.get("target")
        
        # ddp 时候需要手动 gahter 所有数据。 默认输入的类型都是tensor
        if isinstance(pred, List):
            pred = torch.stack(pred, dim=0)
        
        if isinstance(target, List):
            target = torch.stack(target, dim=0)

        seq_len = None
        if "seq_len" in result.keys():
            seq_len = result.get("seq_len")

        if seq_len is not None and target.dim() > 1:
            max_len = target.size(1)
            masks = seq_len_to_mask(seq_len=seq_len, max_len=max_len)
        else:
            masks = None
        if pred.dim() == target.dim():
            if torch.numel(pred) !=torch.numel(target):
                raise RuntimeError(f"when pred have same dimensions with target, they should have same element numbers."
                                   f" while target have shape:{target.shape}, "
                                   f"pred have shape: {pred.shape}")

            pass
        elif pred.dim() == target.dim() + 1:
            pred = pred.argmax(dim=-1)
            if seq_len is None and target.dim() > 1:
                logger.warning("You are not passing `seq_len` to exclude pad when calculate accuracy.")
        else:
            raise RuntimeError(f"when pred have size:{pred.shape}, target should have size: {pred.shape} or "
                               f"{pred.shape[:-1]}, got {target.shape}.")

        if masks is not None:
            self.correct += torch.sum(torch.eq(pred, target).masked_fill(masks.eq(False), 0)).item()
            self.total += torch.sum(masks).item()
        else:
            self.correct += torch.sum(torch.eq(pred, target)).item()
            self.total += np.prod(list(pred.size()))