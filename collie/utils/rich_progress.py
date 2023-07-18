import os
import sys
import json
from typing import Any, Union, Optional

from rich.progress import Progress, Console, GetTimeCallable, get_console, TaskID, Live, Text, ProgressSample
from rich.progress import ProgressColumn, TimeRemainingColumn, BarColumn, TimeElapsedColumn, TextColumn

__all__ = [
    'f_rich_progress'
]

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# 如果不打印的时候，使得整个 progress 没有任何意义
class DummyFRichProgress:
    def __getattr__(self, item):
        return DummyFRichProgress()
    
    def track(self, sequence, *args, **kwargs):
        for value in sequence:
            yield value

    def __call__(self, *args, **kwargs):
        # 防止用户通过 DummyFRichProgress.console.print() 这种调用
        return None

    @property
    def dummy(self)->bool:
        """
        当前对象是否是 dummy 的 rich 对象。

        :return:
        """
        return True


class FRichProgress(Progress, metaclass=Singleton):
    def new_progess(self, *columns: Union[str, ProgressColumn],
                    # 这里将 auto_refresh 关掉是想要避免单独开启线程，同时也是为了避免pdb的时候会持续刷新
                    auto_refresh: bool = False,
                    refresh_per_second: float = 10,
                    speed_estimate_period: float = 30.0,
                    transient: bool = True,
                    redirect_stdout: bool = True,
                    redirect_stderr: bool = True,
                    get_time: Optional[GetTimeCallable] = None,
                    disable: bool = False,
                    expand: bool = False):
        for task_id in self.task_ids:  # 首先移除已有的
            self.remove_task(task_id)

        assert (
                refresh_per_second is None or refresh_per_second > 0
        ), "refresh_per_second must be > 0"

        # stop previous columns
        self.stop()

        # do not change these variables
        # self._lock = RLock()
        # self._tasks: Dict[TaskID, Task] = {}
        # self._task_index: TaskID = TaskID(0)

        if len(columns) != 0:
            self.columns = columns

        self.speed_estimate_period = speed_estimate_period

        self.disable = disable
        self.expand = expand

        self.live = Live(
            console=get_console(),
            auto_refresh=auto_refresh,
            refresh_per_second=refresh_per_second,
            transient=transient,
            redirect_stdout=redirect_stdout,
            redirect_stderr=redirect_stderr,
            get_renderable=self.get_renderable,
        )
        self.get_time = get_time or self.console.get_time
        self.print = self.console.print
        self.print_json = lambda x, **kwargs: self.console.print_json(json.dumps(x), **kwargs)
        self.log = self.console.log
        self.auto_refresh = auto_refresh
        self.transient = transient
        self.redirect_stdout = redirect_stdout
        self.redirect_stderr = redirect_stderr
        self.refresh_per_second = refresh_per_second
        self._need_renew_live = False

        return self

    def set_transient(self, transient: bool = True):
        """
        设置是否在bar运行结束之后不关闭

        :param transient:
        :return:
        """
        self.new_progess(transient=transient)

    def set_disable(self, flag: bool = True):
        """
        设置当前 progress bar 的状态，如果为 True ，则不会显示进度条了。

        :param flag:
        :return:
        """
        self.disable = flag

    def add_task(
            self,
            description: str = 'Progress',
            start: bool = True,
            total: float = 100.0,
            completed: int = 0,
            visible: bool = True,
            **fields: Any,
    ) -> TaskID:
        # 如果需要替换，应该是由于destroy的时候给换掉了
        if self._need_renew_live:
            self.live = Live(
                console=get_console(),
                auto_refresh=self.auto_refresh,
                refresh_per_second=self.refresh_per_second,
                transient=self.transient,
                redirect_stdout=self.redirect_stdout,
                redirect_stderr=self.redirect_stderr,
                get_renderable=self.get_renderable,
            )
            self._need_renew_live = False
        if not self.live.is_started:
            self.start()
        post_desc = fields.pop('post_desc', '')
        return super().add_task(description=description,
                                start=start,
                                total=total,
                                completed=completed,
                                visible=visible,
                                post_desc=post_desc,
                                **fields)

    def stop_task(self, task_id: TaskID) -> None:
        if task_id in self._tasks:
            super().stop_task(task_id)

    def remove_task(self, task_id: TaskID) -> None:
        if task_id in self._tasks:
            super().remove_task(task_id)

    def destroy_task(self, task_id: TaskID):
        if task_id in self._tasks:
            super().stop_task(task_id)
            super().remove_task(task_id)
            self.refresh()  # 使得bar不残留
        if len(self._tasks) == 0:
            # 这里将这个line函数给hack一下防止stop的时候打印出空行
            old_line = getattr(self.live.console, 'line')
            setattr(self.live.console, 'line', lambda *args,**kwargs:...)
            self.live.stop()
            setattr(self.live.console, 'line', old_line)
            # 在 jupyter 的情况下需要替换一下，不然会出不打印的问题。
            self._need_renew_live = False

    def start(self) -> None:
        super().start()
        self.console.show_cursor(show=True)

    def update(
            self,
            task_id: TaskID,
            *,
            total: Optional[float] = None,
            completed: Optional[float] = None,
            advance: Optional[float] = None,
            description: Optional[str] = None,
            visible: Optional[bool] = None,
            refresh: bool = True,
            **fields: Any,
    ) -> None:
        """Update information associated with a task.

        Args:
            task_id (TaskID): Task id (returned by add_task).
            total (float, optional): Updates task.total if not None.
            completed (float, optional): Updates task.completed if not None.
            advance (float, optional): Add a value to task.completed if not None.
            description (str, optional): Change task description if not None.
            visible (bool, optional): Set visible flag if not None.
            refresh (bool): Force a refresh of progress information. Default is False.
            **fields (Any): Additional data fields required for rendering.
        """
        with self._lock:
            task = self._tasks[task_id]
            completed_start = task.completed

            if total is not None and total != task.total:
                task.total = total
                task._reset()
            if advance is not None:
                task.completed += advance
            if completed is not None:
                task.completed = completed
            if description is not None:
                task.description = description
            if visible is not None:
                task.visible = visible
            task.fields.update(fields)
            update_completed = task.completed - completed_start

            current_time = self.get_time()
            old_sample_time = current_time - self.speed_estimate_period
            _progress = task._progress

            popleft = _progress.popleft
            # 这里修改为至少保留一个，防止超长时间的迭代影响判断
            while len(_progress)>1 and _progress[0].timestamp < old_sample_time:
                popleft()
            if update_completed > 0:
                _progress.append(ProgressSample(current_time, update_completed))
            if task.completed >= task.total and task.finished_time is None:
                task.finished_time = task.elapsed

        if refresh:
            self.refresh()

    @property
    def dummy(self) -> bool:
        """
        当前对象是否是 dummy 的 rich 对象。

        :return:
        """
        return False

    def not_empty(self):
        return len(self._tasks) != 0


class SpeedColumn(ProgressColumn):
    """
    显示 task 的速度。

    """
    def render(self, task: "Task"):
        speed = task.speed
        if speed is None:
            return Text('-- it./s', style='progress.data.speed')
        if speed > 0.1:
            return Text(str(round(speed, 2))+' it./s', style='progress.data.speed')
        else:
            return Text(str(round(1/speed, 2))+' s/it.', style='progress.data.speed')


if (sys.stdin and sys.stdin.isatty()):
    # TODO 是不是应该可以手动关掉，防止一些 debug 问题
    f_rich_progress = FRichProgress().new_progess(
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        BarColumn(),
        SpeedColumn(),
        TimeElapsedColumn(),
        "/",
        TimeRemainingColumn(),
        TextColumn("{task.fields[post_desc]}", justify="right"),
        transient=True,
        disable=False,
        speed_estimate_period=600,
        auto_refresh=False,
    )
else:
    f_rich_progress = DummyFRichProgress()
