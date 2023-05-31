from typing import Optional, Sequence
from operator import length_hint

import torch

from .rich_progress import f_rich_progress

class classproperty:
    """
    Reference to https://github.com/hottwaj/classproperties/tree/main

    Decorator for a Class-level property.
    Credit to Denis Rhyzhkov on Stackoverflow: https://stackoverflow.com/a/13624858/1280629"""
    def __init__(self, fget, cached=False):
        self.fget = fget
        self.cached = cached

    def __get__(self, owner_self, owner_cls):
        val = self.fget(owner_cls)
        if self.cached:
            setattr(owner_cls, self.fget.__name__, val)
        return val


def find_tensors():
    """
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
        self.bar.update(self.task_id, post_desc=post_desc, advance=0)

    def set_postfix(self, **kwargs):
        post_desc = ", ".join([f"{k}: {v}" for k, v in kwargs.items()])
        self.set_post_desc(post_desc)

    def set_description(self, desc):
        self.update(desc=desc)

    def update(
        self, desc: Optional[str] = None, total: Optional[float] = None,
        completed: Optional[float] = None, advance: Optional[float] = None,
        visible: Optional[bool] = None, refresh: bool = False,
        post_desc: Optional[str] = None,
    ) -> None:
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
    Split batch to ``micro_batch_num`` micro batches of batch_size
    ``micro_batch_size``

    Only used in Pipeline to hack train_batch

    :param batch: tuple from dataloader
    :param micro_batch_size:
    :param micro_batch_num:
    """
    # Assume batch first.
    assert len(batch) >= 2, len(batch)
    inputs = batch[0]
    labels = batch[1]
    if isinstance(labels, Sequence):
        labels_split = [torch.split(label, micro_batch_size) for label in labels]
    else:
        labels_split = torch.split(labels, micro_batch_size)
    if isinstance(inputs, torch.Tensor):
        inputs_split = torch.split(inputs, micro_batch_size)
        assert len(inputs_split) == micro_batch_num, len(inputs_split)
    else:
        # tuple of tensor
        assert isinstance(inputs, (tuple, list))
        inputs_split = [() for _ in range(micro_batch_num)]
        for tensor in inputs:
            assert isinstance(tensor, torch.Tensor), type(tensor)
            tensor_split = torch.split(inputs, micro_batch_size)
            assert len(tensor_split) == micro_batch_num, len(tensor_split)
            for i in range(micro_batch_num):
                inputs_split[i] += (tensor_split[i], )
    
    batch_split = ()
    for input_split, labels_split in zip(inputs_split, labels_split):
        batch_split += ((input_split, labels_split), )

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