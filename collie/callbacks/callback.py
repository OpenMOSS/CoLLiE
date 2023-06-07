
__all__ = [
    'Callback',
]

class Callback:
    """
    回调机制的基类，包含了数个回调时机。所有的 ``Callback`` 都应该继承该类。
    """
    def on_after_trainer_initialized(self, trainer):
        r"""
        在 ``Trainer`` 初始化后会被触发；

        :param trainer: :class:`.Trainer` 实例。
        """
        pass

    def on_train_begin(self, trainer):
        r"""
        在训练开始前会被触发；

        :param trainer: :class:`.Trainer` 实例。
        """
        pass

    def on_train_end(self, trainer):
        r"""
        在训练完成后会被触发；

        :param trainer: :class:`.Trainer` 实例。
        """
        pass

    def on_train_epoch_begin(self, trainer):
        r"""
        在训练过程中的每一个 epoch 开始前会被触发；

        :param trainer: :class:`.Trainer` 实例。
        """
        pass

    def on_train_epoch_end(self, trainer):
        r"""
        在训练过程中的每一个 epoch 完成后会被触发。

        :param trainer: :class:`.Trainer` 实例。
        """
        pass

    def on_train_batch_begin(self, trainer, batch):
        r"""
        在训练一个 batch 之前触发。

        :param trainer: :class:`.Trainer` 实例；
        :param batch: 当次的 batch 数据。
        """
        pass

    def on_train_batch_end(self, trainer, loss):
        r"""
        完成一个 batch 的训练（forward）、梯度回传（backward）、梯度更新（step）、
        梯度置零后会触发。

        :param trainer: :class:`.Trainer` 实例；
        """
        pass

    def on_save_model(self, trainer):
        r"""
        当调用 :meth:`Trainer.save_model() <collie.trainer.Trainer.save_model>` 时调用，此刻模型还未保存。

        :param trainer: :class:`.Trainer` 实例；
        """
        pass

    def on_load_model(self, trainer):
        r"""
        当调用 :meth:`Trainer.load_model() <collie.trainer.Trainer.load_model>` 加载模型时调用，此刻模型还未加载。

        :param trainer: :class:`.Trainer` 实例；
        """
        pass

    def on_save_checkpoint(self, trainer):
        r"""
        当 Trainer 将要保存 checkpoint 的时候触发 (即调用 :meth:`Trainer.save_checkpoint() <collie.trainer.Trainer.save_checkpoint>`
        函数时)，该函数用于保存当前 callback 在恢复时需要的相关数据。

        :param trainer: :class:`.Trainer` 实例；
        """
        pass

    def on_load_checkpoint(self, trainer):
        r"""
        当 Trainer 要恢复 checkpoint 的时候触发（即调用 :meth:`Trainer.load_checkpoint() <collie.trainer.Trainer.load_checkpoint>`
        函数时）。

        :param trainer: :class:`.Trainer` 实例；
        """
        pass

    def on_evaluate_begin(self, trainer):
        r"""
        在将要进行 ``evaluate`` 时调用。如果 :class:`.CollieConfig` 的
        ``eval_per_n_steps`` 不为 0，则会在 :meth:`on_train_batch_end` 后触发；
        如果 ``eval_per_n_epochs`` 不为 0，则会在 :meth:`on_train_epoch_end` 后
        触发。

        :param trainer: :class:`.Trainer` 实例；
        """
        pass

    def on_evaluate_end(self, trainer, results):
        r"""
        结束 evaluate 时调用，并把 evaluate 的结果传入。

        :param trainer: :class:`.Trainer` 实例；
        :param results: 评测的结果，通常是个 ``dict``；
        """
        pass

    @property
    def callback_name(self):
        r"""
        ``callback`` 的名称，我们会使用该名称从 ``checkpoint`` 中读取的相应的 ``state`` 并传递给 :meth:`on_load_checkpoint` 函数。

        :return: 用于区分该 ``callback`` 实例的名称；
        """
        return self.__class__.__name__
