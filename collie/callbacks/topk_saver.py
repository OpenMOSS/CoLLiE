import os
from copy import deepcopy
from typing import Callable, Dict, Optional, Tuple, Union

from collie.log.logger import logger
from collie.driver.io import IODriver
from collie.utils import env
from .has_monitor_callback import ResultsMonitor

__all__ = ['TopkSaver']


class Saver:
    r"""
    执行保存操作的类，包含模型或断点的保存函数。

    保存的文件组织结构为::

        - folder  # 当前初始化的参数
            - folder_name  # 由 save() 调用时传入。

    :param folder: 保存在哪个文件夹下，默认为当前 folder 下。
    :param process_exclusion: -- 是否互斥地执行保存操作；在模型规模较大时该参数可以
        节省一定的内存。
    :param model_only: 是否仅保存模型的权重；如果为 ``True`` 则仅会保存模型权重，
        否则还会额外保存 optimizer、训练步数等断点信息以用于断点重训，可以通过
        :meth:`.Trainer.load_checkpoint` 加载重新进行训练。该保存路径还可以通过
        :meth:`.CollieForCausalLM.from_pretrained` 函数或者 :meth:`.Trainer.\
        load_model` 加载到模型中；同时也可以直接加载到对应的 huggingface 模型中。
    :param kwargs: 传给 :meth:`.Trainer.save_checkpoint` 或者 :meth:`.Trainer.\
        save_model` 的额外参数。
    """

    def __init__(self, folder: Optional[str] = None, model_only: bool = True,
                 process_exclusion: bool = False, **kwargs):
        if folder is None:
            folder = os.path.abspath(os.getcwd())

        self.save_folder = folder
        self.model_only = model_only
        self.process_exclusion = process_exclusion
        self.kwargs = kwargs

        if model_only:
            self.save_fn_name = "save_model"
        else:
            self.save_fn_name = "save_checkpoint"
        logger.info('The checkpoint will be saved in this folder '
                    f'for this time: {self.save_folder}.')

    def save(self, trainer, folder_name):
        """
        执行保存的函数，将数据保存在::

            - folder/
                - folder_name  # 当前函数参数

        :param trainer: Trainer 对象
        :param folder_name: 保存的 folder 名称，将被创建。
        :return: 实际发生保存的 folder 绝对路径。如果为 None 则没有创建。
        """
        folder = os.path.join(self.save_folder, folder_name)
        save_fn = getattr(trainer, self.save_fn_name)
        save_fn(folder, self.process_exclusion, **self.kwargs)

        return folder
    
    def rm(self, folder_name):
        r"""移除 folder/folder_name 。其中 folder 为用户在初始化指定，
        timestamp 为当前脚本的启动时间。

        :param folder_name: 需要移除的路径。
        :return:
        """
        folder = os.path.join(self.save_folder, folder_name)
        io_driver = IODriver.from_protocol(self.kwargs.get("protocol", "file"))
        if env.rank == 0:
            io_driver.delete(folder)

    def state_dict(self):
        states = {'save_folder': str(self.save_folder)}
        return states

    def load_state_dict(self, states):
        save_folder = states['save_folder']
        # 用户手动传入的 folder 应有最高的优先级
        if self.folder is not None:
            logger.info(
                'Detected: The checkpoint was previously saved in '
                f'{save_folder}, different from the folder {self.save_folder} '
                'you provided, what you provide has higher priority.')
        elif not os.path.exists(save_folder):
            logger.info(
                f'The resuming checkpoint folder {save_folder} is not exists, '
                f'checkpoint will save to {os.path.abspath(self.save_folder)}.')
        else:
            logger.info(f'Resume to save checkpoint in path: {save_folder}.')
            self.save_folder = save_folder


class TopkQueue:
    """用于维护处于 topk 的 key, value 对。

    :param int topk: 整数，-1 表示所有数据都是 topk 的; 如果是 0, 表示没有任何数据
        是满足 topk 的。
    """

    def __init__(self, topk):
        assert isinstance(topk, int)
        self.topk = topk
        self.topk_dict = {}  # 其中 key 为保存的内容，value 是对应的性能。

    def push(self, key, value) -> Tuple[Union[str, None], Union[float, None]]:
        r"""将 key/value 推入 topk 的 queue 中，以 value 为标准，如果满足 topk 则
        保留此次推入的信息，同时如果新推入的数据将之前的数据挤出了 topk ，则会返回被
        挤出的 (key, value)；如果返回为 (None, None)，说明满足 topk 且没有数据被挤
        出。如果不满足 topk ，则返回推入的 (key, value) 本身。这里排序只根据 value
        是否更大了判断，因此如果有的情况是越小越好，请在输入前取负号。

        :param str key:
        :param float value: 如果为 None，则不做任何操作。
        :return: （1）返回输入的 (key, value) ，说明不满足 topk;
            (2) 返回(None, None)，说明满足 topk 且没有被挤出过去的记录;
            (3)返回非输入的 (key, value) , 说明输入满足 topk，且挤出了之前的记录。
        """
        if value is None:
            return key, value
        if self.topk < 0:
            return None, None
        if self.topk == 0:
            return key, value
        if len(self.topk_dict) < self.topk:
            self.topk_dict[key] = value
            return None, None
        min_key = min(self.topk_dict, key=lambda x: self.topk_dict[x])
        if self.topk_dict[min_key] > value:
            return key, value
        else:
            min_value = self.topk_dict.pop(min_key)
            self.topk_dict[key] = value
            return min_key, min_value

    def state_dict(self):
        return deepcopy(self.topk_dict)

    def load_state_dict(self, states):
        self.topk_dict.update(states)

    def __str__(self):
        return f'topk-{self.topk}'

    def __bool__(self):
        # 当 topk 为 0 时，表明该 topk_queue 无意义。
        return self.topk != 0


class TopkSaver(ResultsMonitor, Saver):
    r"""用于识别 topk 模型并保存，也可以仅当一个保存 Saver 使用。

    保存路径为::

        - folder/
            - epoch_{epoch_idx}-batch_{batch_idx}-{topk_monitor}_{monitor_value}/  # 满足topk条件存储文件名

    :param topk: 保存表现最好的 ``topk`` 个模型，-1 为保存所有模型；0 为都不保
        存；大于 0 的数为保存 ``topk`` 个；
    :param monitor: 监控的 metric 值：

        * 为 ``None`` 时，不设置监控值。
        * 为 ``str`` 时，
          collie 将尝试直接使用该名称从 ``evaluation`` 的结果中寻找，如果最终在
          ``evaluation`` 结果中没有找到完全一致的名称，则将使用最长公共字符串算法
          从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor``。
        * 为 :class:`Callable` 时，
          则接受参数为 ``evaluation`` 的结果（字典类型），返回一个 ``float`` 值作
          为 ``monitor`` 的结果，如果当前结果中没有相关的 ``monitor`` 值则返回
          ``None``。
    :param larger_better: 该 monitor 是否越大越好。
    :param folder: 保存在哪个文件夹下，默认为当前 folder 下。
    :param process_exclusion: -- 是否互斥地执行保存操作；在模型规模较大时该参数可以
        节省一定的内存。
    :param model_only: 是否仅保存模型的权重；如果为 ``True`` 则仅会保存模型权重，
        否则还会额外保存 optimizer、训练步数等断点信息以用于断点重训，可以通过
        :meth:`.Trainer.load_checkpoint` 加载重新进行训练。该保存路径还可以通过
        :meth:`.CollieForCausalLM.from_pretrained` 函数或者 :meth:`.Trainer.\
        load_model` 加载到模型中；同时也可以直接加载到对应的 huggingface 模型中。
    :param kwargs: 传给 :meth:`.Trainer.save_checkpoint` 或者 :meth:`.Trainer.\
        save_model` 的额外参数。
    """

    def __init__(self,
                 topk: int = 0,
                 monitor: Optional[Union[str, Callable]] = None,
                 larger_better: bool = True,
                 folder: Optional[str] = None,
                 process_exclusion: bool = False,
                 model_only: bool = True,
                 **kwargs):
        if topk is None:
            topk = 0
        ResultsMonitor.__init__(self, monitor, larger_better)
        Saver.__init__(self, folder, model_only, process_exclusion)

        if monitor is not None and topk == 0:
            raise RuntimeError('`monitor` is set, but `topk` is 0.')
        if topk != 0 and monitor is None:
            raise RuntimeError('`topk` is set, but `monitor` is None.')

        self.topk_queue = TopkQueue(topk)

    def save_topk(self, trainer, results: Dict) -> Optional[str]:
        r"""根据 ``results`` 是否满足 topk 的相关设定决定是否保存，如果发生了保存，
        将返回保存的文件夹。如果返回为 ``None``，则说明此次没有满足 topk 要求，没
        有发生保存。

        :param trainer:
        :param results: evaluate 的结果。
        :return: 如果满足 topk 的相关设定，则返回保存的文件夹；否则返回 ``None``。
        """
        if self.monitor is not None and self.topk_queue:
            monitor_value = self.get_monitor_value(results)
            if monitor_value is None:
                return None
            key = f'epoch_{trainer.epoch_idx}-' \
                  f'batch_{trainer.batch_idx}' \
                  f'-{self.monitor_name}_{monitor_value}'
            pop_key, pop_value = self.topk_queue.push(
                key, monitor_value if self.larger_better else -monitor_value)
            if pop_key == key:  # 说明不足以构成 topk，被退回了
                return None
            folder = self.save(trainer, key)

            if pop_key and pop_key != key:  # 说明需要移除之前的 topk
                self.rm(pop_key)
            return folder
        else:
            return None

    def state_dict(self):
        states = {
            'topk_queue': self.topk_queue.state_dict(),
            'save_folder': str(self.save_folder),
        }
        if isinstance(self._real_monitor, str):
            states['_real_monitor'] = self._real_monitor

        return states

    def load_state_dict(self, states):
        topk_queue_states = states['topk_queue']
        self.topk_queue.load_state_dict(topk_queue_states)

        save_folder = states['save_folder']
        # 用户手动传入的 folder 应有最高的优先级
        if self.folder is not None:
            logger.info(
                'Detected: The checkpoint was previously saved in '
                f'{save_folder}, different from the folder {self.save_folder} '
                'you provided, what you provide has higher priority.')
        elif not os.path.exists(save_folder):
            logger.info(
                f'The resuming checkpoint folder {save_folder} is not exists, '
                f'checkpoint will save to {os.path.abspath(self.save_folder)}.')
        else:
            logger.info(f'Resume to save checkpoint in path: {save_folder}.')
            self.save_folder = save_folder

        if '_real_monitor' in states:
            self._real_monitor = states['_real_monitor']
