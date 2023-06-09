import os
from typing import Callable, Union

from collie.log.logger import logger
from collie.driver.io import IODriver
from collie.utils import env
from .has_monitor_callback import HasMonitorCallback

__all__ = ['LoadBestModelCallback']


class LoadBestModelCallback(HasMonitorCallback):
    r"""保存 monitor 值最佳的模型，并在训练结束的时候重新加载模型的 ``Callbcak``。

    默认会在加载之后删除权重文件。仅在训练正常结束的时候才能加载最好的模型。

    :param folder: 保存的文件夹。
    :param process_exclusion: -- 是否互斥地执行保存操作；在模型规模较大时该参数可以
        节省一定的内存。
    :param monitor: 监控的 metric 值。

        * 为 ``str`` 时，
          CoLLiE 将尝试直接使用该名称从 ``evaluation`` 的结果中寻找，如果最终在
          ``evaluation`` 结果中没有找到完全一致的名称，则将使用最长公共字符串算法
          从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor``。
        * 为 :class:`Callable` 时，
          则接受参数为 ``evaluation`` 的结果（字典类型），返回一个 ``float`` 值作
          为 ``monitor`` 的结果，如果当前结果中没有相关的 ``monitor`` 值则返回
          ``None``。
    :param larger_better: 该 metric 值是否是越大越好；
    :param delete_after_train: 在训练结束后是否删掉模型；
    :param kwargs: 传给 :meth:`.Trainer.save_model` 和 :meth:`.Trainer.\
        load_model` 的额外参数。
    """

    def __init__(self,
                 folder: str,
                 process_exclusion: bool = False,
                 monitor: Union[str, Callable, None] = None,
                 larger_better: bool = True,
                 delete_after_train: bool = True,
                 **kwargs
                 ):
        super().__init__(
            monitor=monitor,
            larger_better=larger_better,
            must_have_monitor=True)
        self.save_folder = folder
        self.delete_after_train = delete_after_train
        self.meta = {'epoch': -1, 'batch': -1}
        self.process_exclusion = process_exclusion
        self.kwargs = kwargs
        self.real_save_folder = os.path.join(folder, "best")

    def on_evaluate_end(self, trainer, results):
        if self.is_better_results(results, keep_if_better=True):
            self.meta['epoch'] = trainer.epoch_idx
            self.meta['batch'] = trainer.batch_idx
            trainer.save_model(
                self.real_save_folder, self.process_exclusion,
                **self.kwargs
            )

    def on_train_end(self, trainer):
        if abs(self.monitor_value) != float('inf'):
            # 如果是 inf 说明从来没有运行过。
            logger.info(f'Loading best model from {self.real_save_folder} '
                        f"with '{self._real_monitor}: {self.monitor_value} "
                        f"(achieved in Epoch: {self.meta['epoch']}, Batch in "
                        f"epoch: {self.meta['batch']}) ...")
            trainer.load_model(self.real_save_folder, self.process_exclusion,
                               **self.kwargs)
            if self.delete_after_train:
                self._delete_folder()

    def _delete_folder(self):
        if env.rank == 0:
            protocol = self.kwargs.get("protocol", "file")
            driver = IODriver.from_protocol(protocol)
            driver.delete(self.save_folder)
