""" **CoLLie** 训练过程中的监控器模块，用于记录模型训练过程中的统计信息。
"""

__all__ = [
    "get_monitor",
    "BaseMonitor",
    "StepTimeMonitor",
    "TGSMonitor",
    "MemoryMonitor",
    "LossMonitor",
    "EvalMonitor"
]
from deepspeed.monitor.monitor import MonitorMaster, Monitor

from collie.config import CollieConfig
from collie.log.print import print
from collie.utils import dictToObj
from collie.utils.dist_utils import env

import time
import datetime
from typing import Sequence
from functools import reduce

class DummyDeepSpeedMonitor(Monitor):
    def __init__(self, monitor_config=None) -> None:
        pass
    
    def write_events(self, event_list):
        pass

def get_monitor(config: CollieConfig):
    """
    用于获取DeepSpeed监控器实例的函数。通过这个函数，可以启用一个或多个监控后端（如PyTorch的Tensorboard、WandB和简单的CSV文件）实时记录指标。
    
    :param config:一个CollieConfig对象，用于配置监控器的行为
    :return: (MonitorMaster | DummyDeepSpeedMonitor)如果传入的CollieConfig已经包含有Monitor的相关配置，则返回MonitorMaster实例；否则返回DummyDeepSpeedMonitor实例。
    """
    if "monitor_config" in config.ds_config.keys():
        config.ds_config["monitor_config"]["enabled"] = True
        if "tag" in config.ds_config["monitor_config"].keys():
            tag = config.ds_config["monitor_config"]["tag"]
        else:
            tag = ""
        if "tensorboard" not in config.ds_config["monitor_config"].keys():
            config.ds_config["monitor_config"]["tensorboard"] = {"enabled": False}
        else:
            config.ds_config["monitor_config"]["tensorboard"]["job_name"] = tag + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if "wandb" not in config.ds_config["monitor_config"].keys():
            config.ds_config["monitor_config"]["wandb"] = {"enabled": False}
        else:
            config.ds_config["monitor_config"]["wandb"]["job_name"] = tag + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if "csv_monitor" not in config.ds_config["monitor_config"].keys():
            config.ds_config["monitor_config"]["csv_monitor"] = {"enabled": False}
        else:
            config.ds_config["monitor_config"]["csv_monitor"]["job_name"] = tag + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        return MonitorMaster(dictToObj(config.ds_config["monitor_config"]))
    else:
        return DummyDeepSpeedMonitor(config.ds_config)
    
class BaseMonitor:
    """
    BaseMonitor是一个基础的监控器类，用于记录模型训练过程中的统计信息。
    其中，`trainer` 会将需要统计的数据存放到 `item` 中，目前 item 的内容为:
        .. code-block::
        item = {
            "batch": (input_ids, labels),
            "epoch_idx": 0,
            "batch_idx": 1,
            "global_batch_idx": 1,
            "loss": 0.1,
            "eval_result": {
                "acc": 0.1,
                "ppl": 0.1,
                ...
            },
            "memory_allocated": 7000000000,
            "mode": "train"
        }
        
    :param config: 用户传入的config，类型为CollieConfig
    """
    def __init__(self, config) -> None:
        self.config = config
        self.monitor = get_monitor(config)
        self.item = {}
        
    def __enter__(self):
        return self.item
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    
class StepTimeMonitor(BaseMonitor):
    """ 用来记录每个step的时间
    """
    def __enter__(self):
        self.start = time.time()
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.item["mode"] == "train":
            self.monitor.write_events([(f"Step Time", time.time() - self.start, self.item['global_batch_idx'])])
            
class TGSMonitor(BaseMonitor):
    """ 用来记录每秒每张 GPU 可训练的 token 数 (token / s / GPU)
    """
    def __enter__(self):
        self.start = time.time()
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.item["mode"] == "train" and "batch" in self.item.keys():
            self.monitor.write_events([(f"TGS", reduce(lambda x, y: x * y, self.item["batch"][0].shape) / (env.pp_size * env.tp_size * (time.time() - self.start)), self.item['global_batch_idx'])])
        
class MemoryMonitor(BaseMonitor):
    """ 用来记录每个step的内存占用
    """
    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'memory_allocated' in self.item.keys() and self.item["mode"] == "train":
            self.monitor.write_events([(f"Memory Allocated", self.item['memory_allocated'], self.item['global_batch_idx'])])
            
class LossMonitor(BaseMonitor):
    """ 用来记录每个step的loss
    """
    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'loss' in self.item.keys() and self.item["mode"] == "train":
            self.monitor.write_events([(f"Training Loss", self.item['loss'], self.item['global_batch_idx'])])
            
class EvalMonitor(BaseMonitor):
    """ 用来记录每个step的eval结果，仅支持 **int** 和 **float** 类型的结果
    """
    def __enter__(self):
        self.step = 0
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'eval_result' in self.item.keys() and self.item["mode"] == "eval":
            for key, value in self.item['eval_result'].items():
                    self.monitor.write_events([(f"Metric {key}", value, self.step)])
            self.step += 1
        
class _MultiMonitors:
    def __init__(self, monitors: Sequence[BaseMonitor]) -> None:
        self.monitors = monitors
        self.item = {}
    
    def __enter__(self):
        for monitor in self.monitors:
            monitor.__enter__()
        return self.item
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if env.pp_rank == 0 and env.tp_rank == 0 and env.dp_rank == 0:
            for monitor in self.monitors:
                monitor.item = self.item
                monitor.__exit__(exc_type, exc_val, exc_tb)