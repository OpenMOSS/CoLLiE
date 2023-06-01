from typing import Optional, Sequence
from operator import length_hint

import torch

from .rich_progress import f_rich_progress

__all__ = ["find_tensors", "progress", "dictToObj"]

def find_tensors():
    """
    打印出垃圾回收区的所有张量。

    Adopted from https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/3
    """
    import torch
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.dtype, obj.device)
        except:
            pass

        
class progress:
    """
    包装了 ``rich`` 进度条的类。

    .. code-block::

        for batch in progress(dataloader):
            # do something

    .. code-block::

        with progress(dataloader) as bar:
            for batch in bar:
                # do something
                bar.set_postfix(Loss=1.0)

    .. code-block::

        bar = progress(dataloader)
        for batch in bar:
            # do something
            bar.set_postfix(Loss=1.0)

    :param sequence: 需要遍历的序列，需要是一个可以迭代的对象。
    :param desc: 进度条最左侧的描述语句。
    :param total: 遍历对象的总数。如果为 ``None`` 则会自动进行计算。
    :param completed: 标识进度条的总进度。
    :param upgrade_period: 进度条更新的时间间隔。
    :param disable: 调整进度条是否可见。
    :param post_desc: 进度条最右侧的补充描述语句。
    """
    def __init__(self, sequence, desc="Workin on...", total=None, completed=0,
                 upgrade_period=0.1, disable=False, post_desc: str = ""):
        self.bar = f_rich_progress
        self.bar.set_disable(disable)
        self.sequence = sequence

        self.total = float(length_hint(sequence)) if total is None else total
        self.completed = completed
        self.task_id = self.bar.add_task(
            desc, upgrade_period=upgrade_period, completed=completed,
            post_desc=post_desc, visible=not disable, total=total
        )

    def __iter__(self):
        yield from self.bar.track(
            self.sequence, task_id=self.task_id, total=self.total)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ...

    def __del__(self):
        self.bar.destroy_task(self.task_id)

    def set_post_desc(self, post_desc: str):
        """
        设置进度条最右侧的补充描述语句。

        .. code-block::

            bar = progress(dataloader)
            dataloader.set_post_desc("Loss=1.0")
        """
        self.bar.update(self.task_id, post_desc=post_desc, advance=0)

    def set_postfix(self, **kwargs):
        """
        设置进度条最右侧的补充描述语句。

        对于传入的每一对 key 和 value 将以 ``key1: value1, key2: value2, ..``
        的格式进行显示。

        .. code-block::

            bar = progress(dataloader)
            dataloader.set_postfix(Loss=1.0, Batch=1)
        """
        post_desc = ", ".join([f"{k}: {v}" for k, v in kwargs.items()])
        self.set_post_desc(post_desc)

    def set_description(self, desc):
        """
        设置进度条最左侧的描述语句。
        """
        self.update(desc=desc)

    def update(
        self, desc: Optional[str] = None, total: Optional[float] = None,
        completed: Optional[float] = None, advance: Optional[float] = None,
        visible: Optional[bool] = None, refresh: bool = False,
        post_desc: Optional[str] = None,
    ) -> None:
        """
        对进度条的内容进行更新，可以更加详细地改变进度条的内容。

        :param desc: 进度条最左侧的描述语句。
        :param total: 遍历对象的总数。如果为 ``None`` 则不会发生改变。
        :param completed: 标识进度条的总进度。
        :param advance: 该次进度条更新的进度。
        :param visible: 调整进度条是否可见。
        :param refresh: 是否强制刷新进度条。
        :param post_desc: 进度条最右侧的补充描述语句。
        """
        if post_desc is None:
            self.bar.update(self.task_id, total=total, completed=completed,
                        advance=advance, description=desc, visible=visible,
                        refresh=refresh)
        else:
            self.bar.update(self.task_id, total=total, completed=completed,
                        advance=advance, description=desc, visible=visible,
                        refresh=refresh, post_desc=post_desc)

def _split_batch(batch, micro_batch_size, micro_batch_num):
    """
    将 ``batch`` 划分为 ``micro_batch_num`` 个 ``micro_batch_size`` 大小。

    仅在流水线情况的训练和验证中用到。

    :param batch: tuple from dataloader
    :param micro_batch_size:
    :param micro_batch_num:
    :return: tuple
    """
    # Assume batch first.
    assert len(batch) == 2, len(batch)
    inputs = batch[0]
    labels = batch[1]
    if isinstance(labels, Sequence):
        labels_split = (torch.split(label, micro_batch_size) for label in labels)
        print(labels[0].shape)
        labels_split = list(zip(*labels_split))
    else:
        labels_split = torch.split(labels, micro_batch_size)
    if isinstance(inputs, torch.Tensor):
        inputs_split = torch.split(inputs, micro_batch_size)
        assert len(inputs_split) == micro_batch_num, len(inputs_split)
    else:
        inputs_split = (torch.split(input_, micro_batch_size) for input_ in inputs)
        inputs_split = list(zip(*inputs_split))
    
    batch_split = ()
    for input_split, label_split in zip(inputs_split, labels_split):
        print(label_split)
        batch_split += ((input_split, label_split), )

    return batch_split

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d