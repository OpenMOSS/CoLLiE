import os

from collie.callbacks.callback_manager import CallbackManager
from collie.utils.dist_utils import env

class TrainerEventTrigger:
    callback_manager: CallbackManager

    def on_setup_parallel_model(self):
        self.callback_manager.on_setup_parallel_model(self)

    def on_after_trainer_initialized(self):
        self.callback_manager.on_after_trainer_initialized(self)

    def on_train_begin(self):
        self.callback_manager.on_train_begin(self)

    def on_train_end(self):
        self.callback_manager.on_train_end(self)

    def on_train_epoch_begin(self):
        self.callback_manager.on_train_epoch_begin(self)

    def on_train_epoch_end(self):
        self.callback_manager.on_train_epoch_end(self)

    def on_train_batch_begin(self, batch):
        self.callback_manager.on_train_batch_begin(self, batch)

    def on_train_batch_end(self, loss):
        self.callback_manager.on_train_batch_end(self, loss)

    def on_save_model(self):
        self.callback_manager.on_save_model(self)

    def on_load_model(self):
        self.callback_manager.on_load_model(self)

    def on_save_checkpoint(self):
        self.callback_manager.on_save_checkpoint(self)

    def on_load_checkpoint(self, states):
        self.callback_manager.on_load_checkpoint(self, states)

    def on_evaluate_begin(self):
        self.callback_manager.on_evaluate_begin(self)

    def on_evaluate_end(self, results):
        self.callback_manager.on_evaluate_end(self, results)

def _merge_peft(path, prefix, io_driver):
    """
    在 pp 情况下将分开保存的 peft 合并到同一个文件
    """
    if env.pp_size == 1:
        return
    full_dict = {}
    for pp in range(env.pp_size):
        cur_name = os.path.join(path, f"{prefix}_{pp}.bin")
        full_dict.update(io_driver.load(cur_name, "b"))
        io_driver.delete(cur_name)
    # TODO merge pp to hf
    io_driver.save(full_dict, os.path.join(path, f"{prefix}.bin"))

def _is_name_in_current_rank(name):
    # TODO convert hf to pp
    name_split = name.split(".")
    for name_part in name_split:
        try:
            layer_idx = int(name_part)
        except ValueError:
            continue
        if layer_idx in env.pipeline_layers_idx:
            return True
        else:
            return False
    # 不可能走到这里
    raise ValueError("Not a pipeline peft checkpoint.")

def _split_peft(state: dict):
    if env.pp_size == 1:
        return
    for name in list(state.keys()):
        if not _is_name_in_current_rank(name):
            state.pop(name)
    return state