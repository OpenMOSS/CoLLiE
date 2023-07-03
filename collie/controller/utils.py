from collie.callbacks.callback_manager import CallbackManager

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