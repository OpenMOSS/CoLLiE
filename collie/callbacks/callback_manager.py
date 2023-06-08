from typing import Sequence

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

    @_exec_callback
    def on_save_checkpoint(self, trainer):
        pass

    @_exec_callback
    def on_load_checkpoint(self, trainer):
        pass

    @_exec_callback
    def on_evaluate_begin(self, trainer):
        pass

    @_exec_callback
    def on_evaluate_end(self, trainer, results):
        pass