from typing import List, Dict
from collections import Counter

import torch

from collie.log import logger
from collie.metrics.base import BaseMetric
from collie.utils.seq_len_to_mask import seq_len_to_mask

__all__ = ['ClassifyFPreRecMetric']


def _compute_f_pre_rec(beta_square, tp, fn, fp):
    r"""

    :param tp: int, true positive
    :param fn: int, false negative
    :param fp: int, false positive
    :return: (f, pre, rec)
    """
    pre = tp / (fp + tp + 1e-13)
    rec = tp / (fn + tp + 1e-13)
    f = (1 + beta_square) * pre * rec / (beta_square * pre + rec + 1e-13)

    return f, pre, rec


class ClassifyFPreRecMetric(BaseMetric):
    """计算分类结果 **F值** 的 ``Metric``。
    
    :param gather_result: 在计算 metric 的时候是否自动将各个进程上的输入进行聚合后再输入到 update 之中。
    :param tag_vocab: 标签的 vocabulary(Dict类型)。默认值为``None``。
        若为 ``None`` 则使用数字来作为标签内容，否则使用 vocab 来作为标签内容。
    :param only_gross: 是否只计算总的 ``f1``, ``precision``, ``recall`` 的值；
        如果为 ``False``，不仅返回总的 ``f1``, ``pre``, ``rec``, 还会返回每个
        label 的 ``f1``, ``pre``, ``rec``。
    :param f_type: `micro` 或 `macro`。

        * `micro` : 通过先计算总体的 TP，FN 和 FP 的数量，再计算 f, precision,
          recall；
        * `macro` : 分布计算每个类别的 f, precision, recall，然后做平均（各类别 f
          的权重相同）。

    :param beta: **f_beta** 分数中的 ``beta`` 值。常用为 ``beta=0.5, 1, 2`` 若
        为 0.5 则 **精确率** 的权重高于 **召回率**；若为1，则两者平等；若为2，则
        **召回率** 权重高于 **精确率**。**f_beta** 分数的计算公式为：

        .. math::

            f_{beta} = \\frac{(1 + {beta}^{2})*(pre*rec)}{({beta}^{2}*pre + rec)}
    """
    
    def __init__(self, gather_result: bool = False, tag_vocab=None, only_gross: bool = True, f_type='micro', beta=1) -> None:
        super().__init__(gather_result)
        self.f_type = f_type
        self.beta = beta
        self.beta_square = self.beta**2
        self.only_gross = only_gross
        self.tag_vocab = tag_vocab
        if self.tag_vocab is not None:
            self.tag_vocab_revert = {v: k for k,v in self.tag_vocab}
        
        self._tp: Counter = Counter()
        self._fp: Counter = Counter()
        self._fn: Counter = Counter()
    
    def reset(self):
        """重置 ``tp``, ``fp``, ``fn`` 的值。"""
        self._tp.clear()
        self._fp.clear()
        self._fn.clear()
    
    def get_metric(self) -> Dict:
        r"""
        :meth:`get_metric` 函数将根据 :meth:`update` 函数累计的评价指标统计量来计
        算最终的评价结果。

        :return: 包含以下内容的字典：``{"f1": float, "pre": float, "rec": float}``
        """
        evaluate_result = {}
        if not self.only_gross or self.f_type == 'macro':
            tags = set(self._fn.keys())
            tags.update(set(self._fp.keys()))
            tags.update(set(self._tp.keys()))
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                if self.tag_vocab is not None:
                    tag_name = self.tag_vocab_revert[tag]
                else:
                    tag_name = int(tag)
                tp = self._tp[tag]
                fn = self._fn[tag]
                fp = self._fp[tag]
                f, pre, rec = _compute_f_pre_rec(self.beta_square, tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if not self.only_gross and tag != '':  # tag!=''防止无tag的情况
                    f_key = 'f-{}'.format(tag_name)
                    pre_key = 'pre-{}'.format(tag_name)
                    rec_key = 'rec-{}'.format(tag_name)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec
                    
            if self.f_type == 'macro':
                evaluate_result['f'] = f_sum / len(tags)
                evaluate_result['pre'] = pre_sum / len(tags)
                evaluate_result['rec'] = rec_sum / len(tags)
                
        if self.f_type == 'micro':
            f, pre, rec = _compute_f_pre_rec(self.beta_square,
                                             sum(self._tp.values()),
                                             sum(self._fn.values()),
                                             sum(self._fp.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec
            
        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)
            
        return evaluate_result
    
    def update(self, result: Dict):
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
        assert "pred" in result and "target" in result, "pred and target  must in result, but they not."
        pred = result['pred']
        target = result['target']
        
        # ddp 时候需要手动 gahter 所有数据。 默认输入的类型都是tensor
        if isinstance(pred, List):
            pred = torch.stack(pred, dim=0)
        
        if isinstance(target, List):
            target = torch.stack(target, dim=0)
        
        seq_len = None
        if "seq_len" in result:
            seq_len = result['seq_len']
        if seq_len is not None and target.ndim > 1:
            max_len = target.shape[-1]
            masks = seq_len_to_mask(seq_len=seq_len, max_len=max_len)
        else:
            masks = torch.ones_like(target)
            
        masks = masks.eq(1)
        
        if pred.dim() == target.dim():
            if len(pred.flatten()) != len(target.flatten()):
                raise RuntimeError(
                    'when pred have same dimensions with target, they should '
                    'have same element numbers. while target have element '
                    f'numbers:{len(pred.flatten())}, pred have element '
                    f'numbers: {len(target.flatten())}')
        elif pred.dim() == target.dim()+1:
            pred = pred.argmax(axis=-1)
            if seq_len is None and target.dim() > 1:
                logger.warning_once(
                    'You are not passing `seq_len` to exclude pad when '
                    'calculate accuracy.')
        else:
            raise RuntimeError(
                f'when pred have '
                f'size:{pred.shape}, target should have size: {pred.shape} or '
                f'{pred.shape[:-1]}, got {target.shape}.')
        
        target = target.masked_select(masks)
        pred = pred.masked_select(masks)
        
        target_idxes = set(target.reshape(-1).tolist())
        for target_idx in target_idxes:
            self._tp[target_idx] += torch.sum((pred == target_idx).long().masked_fill(target != target_idx, 0)).item()
            self._fp[target_idx] += torch.sum((pred == target_idx).long().masked_fill(target == target_idx, 0)).item()
            self._fn[target_idx] += torch.sum((pred != target_idx).long().masked_fill(target != target_idx, 0)).item()    
