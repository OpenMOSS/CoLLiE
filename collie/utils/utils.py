from typing import Optional
from operator import length_hint

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

    def __init__(self, sequence, desc="Workin on...", total=None,
                 upgrade_period=0.1, disable=False, post_desc: str = ""):
        self.bar = f_rich_progress
        self.bar.set_disable(disable)
        self.total = float(length_hint(sequence)) if total is None else total
        self.task_id = self.bar.add_task(
            desc, upgrade_period=upgrade_period, post_desc=post_desc,
            visible=not disable, total=self.total
        )
        self.sequence = sequence

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

    def update(
        self, desc: Optional[str] = None, total: Optional[float] = None,
        completed: Optional[float] = None, advance: Optional[float] = None,
        visible: Optional[bool] = None, refresh: bool = False,
        post_desc: Optional[str] = None,
    ) -> None:
        if post_desc is not None:
            self.bar.update(self.task_id, total=total, completed=completed,
                        advance=advance, description=desc, visible=visible,
                        refresh=refresh)
        else:
            self.bar.update(self.task_id, total=total, completed=completed,
                        advance=advance, description=desc, visible=visible,
                        refresh=refresh, post_desc=post_desc)