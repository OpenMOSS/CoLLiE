import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from collie.log.logger import logger
from .callback import Callback
from .topk_saver import TopkSaver

__all__ = ['CheckpointCallback']


class CheckpointCallback(Callback):
    r"""用于保存断点 ``checkpoint`` 的 ``Callback``。

    其保存的文件目录以及文件名命名规则如下::

        - folder/
            - epoch_{epoch_idx}/  # 满足 every_n_epochs 条件保存的模型
            - epoch_{epoch_idx}-batch_{batch_idx}/  # 满足 every_n_batches 保存的模型
            - last/  # 最后一个 epoch 的保存
            - epoch_{epoch_idx}-batch_{batch_idx}-{monitor}_{monitor_value}/  # 满足topk条件存储文件名

    默认情况下，本 checkpoint 只保存了 model 的状态；如还需保存 Trainer 的状态
    以断点重训的话，请使用 ``model_only=False``。

    :param folder: 保存的文件夹，如果为 ``None`` ，默认使用当前文件夹。
    :param every_n_epochs: 多少个 epoch 保存一次。
    :param every_n_batches: 多少个 batch 保存一次。
    :param process_exclusion: -- 是否互斥地执行保存操作；在模型规模较大时该参数可以
        节省一定的内存。
    :param model_only: 是否仅保存模型的权重；如果为 ``True`` 则仅会保存模型权重，
        否则还会额外保存 optimizer、训练步数等断点信息以用于断点重训，可以通过
        :meth:`.Trainer.load_checkpoint` 加载重新进行训练。该保存路径还可以通过
        :meth:`.CollieForCausalLM.from_pretrained` 函数或者 :meth:`.Trainer.\
        load_model` 加载到模型中；同时也可以直接加载到对应的 huggingface 模型中。
    :param peft_only: 是否只保存 adapter；当未使用 ``peft`` 时该项无效
    :param monitor: 监控的 metric 值。

        * 为 ``str`` 时，
          collie 将尝试直接使用该名称从 ``evaluation`` 的结果中寻找，如果最终在
          ``evaluation`` 结果中没有找到完全一致的名称，则将使用最长公共字符串算法
          从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor``。
        * 为 :class:`Callable` 时，
          则接受参数为 ``evaluation`` 的结果（字典类型），返回一个 ``float`` 值作
          为 ``monitor`` 的结果，如果当前结果中没有相关的 ``monitor`` 值则返回
          ``None``。
    :param larger_better: monitor 的值是否时越大越好。
    :param topk: 保存 monitor 结果中的 ``topk`` 个。
    :param last: 如果为 ``True``，将在每次 epoch 运行结束都保存一次，会覆盖之前的
        保存。如果为 ``False`` 则不会保存 ``last`` 文件。
    :param max: 最多保留多少个通过 ``every_n_batches`` 和 ``every_n_epochs`` 保存
        的权重（如果设置了的话）；如果为 ``None`` 或 0，则会保留所有的权重文件。
    :param kwargs: 传给 :meth:`.Trainer.save_checkpoint` 或者 :meth:`.Trainer.\
        save_model` 、 :meth:`.Trainer.save_peft` 的额外参数。
    """

    def __init__(
            self,
            folder: Optional[Union[str, Path]] = None,
            every_n_epochs: Optional[int] = None,
            every_n_batches: Optional[int] = None,
            process_exclusion: bool = False,
            model_only: bool = True,
            peft_only: bool = True,
            monitor: Optional[Union[str, Callable]] = None,
            larger_better: bool = True,
            topk: int = 0,
            last: bool = False,
            max: Optional[int] = None,
            **kwargs):
        super().__init__()
        if every_n_epochs is not None:
            if not isinstance(every_n_epochs, int) or every_n_epochs < 0:
                raise ValueError(
                    'Parameter `every_n_epochs` should be an int and greater '
                    'than or equal to 0.')
        if every_n_epochs is None or every_n_epochs == 0:
            every_n_epochs = sys.maxsize # 使得没有数字可以整除

        if every_n_batches is not None:
            if not isinstance(every_n_batches, int) or every_n_batches < 0:
                raise ValueError(
                    'Parameter `every_n_batches` should be an int and greater '
                    'than or equal to 0.')
        if every_n_batches is None or every_n_batches == 0:
            every_n_batches = sys.maxsize

        if max is not None:
            if not isinstance(max, int) and max < 0:
                raise ValueError(
                    'Parameter `max` should be an int and greater than or '
                    'equal to 0.')

        self.topk_saver = TopkSaver(
            topk=topk,
            monitor=monitor,
            larger_better=larger_better,
            folder=folder,
            process_exclusion=process_exclusion,
            model_only=model_only,
            peft_only=peft_only,
            **kwargs)
        self.topk_saver.log_name = self.__class__.__name__

        self.topk = topk

        self.every_n_epochs = every_n_epochs
        self.every_n_batches = every_n_batches
        self.last = last
        self.max = max if max is not None else 0
        self.ckpt_queue = []

    def on_after_trainer_initialized(self, trainer):
        if self.topk_saver.topk_queue and trainer.evaluators is None:
            logger.warning(
                f'You set `topk={self.topk}`, but `eval_dataset` is '
                'not set in Trainer.')

    def on_evaluate_end(self, trainer, results):
        self.topk_saver.save_topk(trainer, results)

    def on_train_epoch_end(self, trainer):
        if (trainer.epoch_idx + 1) % self.every_n_epochs == 0:
            folder_name = f'epoch_{trainer.epoch_idx + 1}'
            self.topk_saver.save(trainer, folder_name=folder_name)
            self.ckpt_queue.append(folder_name)
            if self.max > 0 and len(self.ckpt_queue) > self.max:
                self.topk_saver.rm(self.ckpt_queue.pop(0))
        if self.last:
            folder_name = f'last'
            self.topk_saver.save(trainer, folder_name=folder_name)

    def on_train_batch_end(self, trainer, loss):
        if (trainer.batch_idx + 1) % self.every_n_batches == 0:
            folder_name = f'epoch_{trainer.epoch_idx}' \
                          f'-batch_{trainer.batch_idx + 1}'
            self.topk_saver.save(trainer, folder_name=folder_name)
            self.ckpt_queue.append(folder_name)
            if self.max > 0 and len(self.ckpt_queue) > self.max:
                self.topk_saver.rm(self.ckpt_queue.pop(0))

    def on_save_checkpoint(self, trainer) -> Dict:
        states = {}
        states['topk_saver'] = self.topk_saver.state_dict()
        return states

    def on_load_checkpoint(self, trainer, states):
        topk_saver_states = states['topk_saver']
        self.topk_saver.load_state_dict(topk_saver_states)