from deepspeed.monitor.monitor import MonitorMaster, Monitor

from collie.config import CollieConfig
from collie.log.print import print
from collie.utils import dictToObj

import time
from typing import Sequence

class DummyDeepSpeedMonitor(Monitor):
    def __init__(self, monitor_config=None) -> None:
        pass
    
    def write_events(self, event_list):
        pass

def get_monitor(config: CollieConfig):
    if "monitor_config" in config.ds_config.keys():
        config.ds_config["monitor_config"]["enabled"] = True
        if "tensorboard" not in config.ds_config["monitor_config"].keys():
            config.ds_config["monitor_config"]["tensorboard"] = {"enabled": False}
        if "wandb" not in config.ds_config["monitor_config"].keys():
            config.ds_config["monitor_config"]["wandb"] = {"enabled": False}
        if "csv_monitor" not in config.ds_config["monitor_config"].keys():
            config.ds_config["monitor_config"]["csv_monitor"] = {"enabled": False}
        return MonitorMaster(dictToObj(config.ds_config["monitor_config"]))
    else:
        return DummyDeepSpeedMonitor(config.ds_config["monitor_config"])
    
class BaseMonitor:
    def __init__(self, config) -> None:
        self.config = config
        self.monitor = get_monitor(config)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def set_trainer(self, trainer):
        self.trainer = trainer
    
    
class StepTimeMonitor(BaseMonitor):
    def __init__(self, config) -> None:
        super().__init__(config)
        
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.write_events([(f"Training Time Epoch {self.trainer.epoch_idx}", time.time() - self.start, self.trainer.batch_idx)])
        
class MultiMonitors:
    def __init__(self, trainer, monitors: Sequence[BaseMonitor]) -> None:
        self.monitors = monitors
        for monitor in self.monitors:
            monitor.set_trainer(trainer)
        self.trainer = trainer
        pass
    
    def __enter__(self):
        for monitor in self.monitors:
            monitor.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for monitor in self.monitors:
            monitor.__exit__(exc_type, exc_val, exc_tb)
        pass