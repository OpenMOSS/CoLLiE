from typing import Optional

from rich.progress import Progress, TimeRemainingColumn, BarColumn, TimeElapsedColumn, TextColumn, ProgressColumn, Text
from .rich_progress import f_rich_progress

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

class SpeedColumn(ProgressColumn):
    """
    显示 task 的速度。

    """
    def render(self, task):
        speed = task.speed
        if speed is None:
            return Text('-- it./s', style='progress.data.speed')
        if speed > 0.1:
            return Text(str(round(speed, 2))+' it./s', style='progress.data.speed')
        else:
            return Text(str(round(1/speed, 2))+' s/it.', style='progress.data.speed')
        
class progress:

    def __init__(self, sequence, desc="Workin on...", total=None,
                 upgrade_period=0.1, disable=False, post_desc: str = ""):
        self.bar = f_rich_progress
        self.bar.set_disable(disable)
        self.task_id = self.bar.add_task(
            desc, upgrade_period=upgrade_period, post_desc=post_desc
        )
        self.sequence = sequence
        self.total = total

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