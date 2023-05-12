from typing import Optional

from rich.progress import Progress, TimeRemainingColumn, BarColumn, TimeElapsedColumn, TextColumn, ProgressColumn, Text

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
        columns = [TextColumn("[progress.description]{task.description}")] if desc else []
        columns.extend(["[progress.percentage]{task.percentage:>3.0f}%"])
        columns.extend([
            BarColumn(),
            SpeedColumn(),
            TimeElapsedColumn(),
            "/",
            TimeRemainingColumn(),
            TextColumn("{task.fields[post_desc]}",justify="right")
        ])
        self.bar = Progress(*columns, disable=disable)
        self.task_id = self.bar.add_task(
            desc, total=total, upgrade_period=upgrade_period,
            post_desc=post_desc
        )
        self.sequence = sequence

    def __iter__(self):
        with self.bar:
            yield from self.bar.track(self.sequence, task_id=self.task_id)

    def __enter__(self):
        self.bar.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.bar.stop()

    def set_post_desc(self, post_desc: str):
        self.bar.update(self.task_id, post_desc=post_desc, advance=0)

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