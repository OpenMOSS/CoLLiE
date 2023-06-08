from typing import Sequence, Dict

from collie.utils.utils import _get_fun_msg
from collie.log import logger
from .callback import Callback

def prepare_callback(callbacks):
    """
    遍历 callbacks，并且加入 :class:`.ProgressCallback`。
    """
    _callbacks = []
    if callbacks is not None:
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        if not isinstance(callbacks, Sequence):
            raise ValueError("Parameter `callbacks` should be type 'List' or 'Tuple'.")
        callbacks = list(callbacks)
        for _callback in callbacks:
            if not isinstance(_callback, Callback):
                raise TypeError(f"callbacks must be of Callback type, instead of `{type(_callback)}`")
        _callbacks += callbacks

    return _callbacks

def _exec_callback(func):

    def wrapper(manager, *args, **kwargs):
        if manager.callbacks != None:
            for callback in manager.callbacks:
                callback_fn = getattr(callback, func.__name__)
                try:
                    callback_fn(*args, **kwargs)
                except (KeyboardInterrupt) as e:
                    raise e
                except BaseException as e:
                    logger.error(f"The following callback_fn raise exception:{_get_fun_msg(callback_fn)}.")
                    raise e
            
    return wrapper

class CallbackManager:
    def __init__(self, callbacks):
        self.callbacks = callbacks

    @_exec_callback
    def on_after_trainer_initialized(self, trainer):
        pass

    @_exec_callback
    def on_train_begin(self, trainer):

        pass

    @_exec_callback
    def on_train_end(self, trainer):
        pass

    @_exec_callback
    def on_train_epoch_begin(self, trainer):
        pass

    @_exec_callback
    def on_train_epoch_end(self, trainer):
        pass

    @_exec_callback
    def on_train_batch_begin(self, trainer, batch):
        pass

    @_exec_callback
    def on_train_batch_end(self, trainer, loss):
        pass

    @_exec_callback
    def on_save_model(self, trainer):
        pass

    @_exec_callback
    def on_load_model(self, trainer):
        pass

    def on_save_checkpoint(self, trainer):
        r"""
        用于断点重训的 callback 的保存函数；
        该函数主要涉及callback 的状态的保存；我们会调用每一个 callback 的
        :func:`on_save_checkpoint` 方法，该方法应当返回一个字典，其中包含着
        断点重训应当保存的状态；

        :param trainer: :class:`~.Trainer` 实例；
        :return: 一个包含上述内容的字典，格式如下:
        .. code-block::

            {
                "callback_name_1": {
                        ...
                    }
                }
            }
        """

        states: Dict[str, dict] = {}
        # 1. 每一个 callback 的状态；
        # 如果有两个 callback 的 name 相同，那么我们只会保存第一个；
        _duplicated_callbacks = []
        for callback in self.callbacks:
            callback_name = callback.callback_name
            if callback_name in states:
                _duplicated_callbacks.append(callback_name)
                # 对于 callback_name 有重复的 callback，我们仍旧会调用其
                # `on_save_checkpoint` 函数，就如同调用其它 callback 函数
                #  一样，但是其结果并不会存储在 states 中返回；
                callback.on_save_checkpoint(trainer)
            else:
                states[callback_name] = callback.on_save_checkpoint(trainer)

        if len(_duplicated_callbacks) > 0:
            logger.warning(
                f'Notice these callback_name: {_duplicated_callbacks} '
                'are duplicated, collie will only save the first callback\'s '
                'state.')

        return states

    def on_load_checkpoint(self, trainer, states):
        r"""
        用于断点重训的加载函数，对应于断点重训的保存函数；

        :param trainer: :class:`.Trainer` 实例；
        :param states: 同 :func:`on_save_checkpoint` 函数的返回值；
        """
        # 恢复每一个 callback 的单独的状态；
        # 每一个我们自己提供的类 callback，都需要重写其特定的 `callback_name`
        # 方法，保证如果两个 callback 的 callback_name 一样，
        #  那么它们就应该是同一个对象；
        _loaded_callbacks = set()
        _duplicated_callbacks = set()
        for each_callback in self.all_callbacks:
            callback_name = each_callback.callback_name
            if callback_name in states and \
                    callback_name not in _loaded_callbacks:
                _loaded_callbacks.add(callback_name)
                # 这里要注意，我们已经确保每一个 callback 的
                # `on_load_checkpoint` 函数拿到的就是其自己的状态；
                each_callback.on_load_checkpoint(
                    trainer, states[callback_name])
            else:
                _duplicated_callbacks.add(callback_name)

        if len(_duplicated_callbacks) > 0:
            logger.warning(
                f'Notice these callback_name: {_duplicated_callbacks} '
                'are duplicated, CoLLiE will only save the first callback\'s '
                'state.')

    @_exec_callback
    def on_evaluate_begin(self, trainer):
        pass

    @_exec_callback
    def on_evaluate_end(self, trainer, results):
        pass