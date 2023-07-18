""" **CoLLie** 训练过程中的监控器模块，用于记录模型训练过程中的统计信息。
"""

__all__ = [
    "get_monitor",
    "BaseMonitor",
    "StepTimeMonitor",
    "TGSMonitor",
    "MemoryMonitor",
    "LossMonitor",
    "EvalMonitor",
    "NetworkIOMonitor",
    "DiskIOMonitor",
    "CPUMemoryMonitor"
]
from deepspeed.monitor.monitor import MonitorMaster, Monitor

from collie.config import CollieConfig
from collie.log.print import print
from collie.utils import dictToObj
from collie.utils.dist_utils import env

import time
import psutil
import datetime
from typing import Sequence, Optional, Dict
from functools import reduce

class DummyDeepSpeedMonitor(Monitor):
    def __init__(self, monitor_config=None) -> None:
        pass
    
    def write_events(self, event_list):
        pass

def get_monitor(config: CollieConfig):
    """
    用于获取DeepSpeed监控器实例的函数。
    通过这个函数，可以启用一个或多个监控后端（如PyTorch的Tensorboard、WandB和简单的CSV文件）实时记录指标。
    
    :param config:一个CollieConfig对象，用于配置监控器的行为
    :return: (MonitorMaster | DummyDeepSpeedMonitor) 如果传入的CollieConfig已经包含有Monitor的相关配置，则返回MonitorMaster实例；否则返回DummyDeepSpeedMonitor实例。
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
            config.ds_config["monitor_config"]["tensorboard"]["job_name"] = tag
        if "wandb" not in config.ds_config["monitor_config"].keys():
            config.ds_config["monitor_config"]["wandb"] = {"enabled": False}
        else:
            config.ds_config["monitor_config"]["wandb"]["job_name"] = tag
            config.ds_config["monitor_config"]["wandb"]["config"] = config
        if "csv_monitor" not in config.ds_config["monitor_config"].keys():
            config.ds_config["monitor_config"]["csv_monitor"] = {"enabled": False}
        else:
            config.ds_config["monitor_config"]["csv_monitor"]["job_name"] = tag + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        monitor = MonitorMaster(dictToObj(config.ds_config["monitor_config"]))
        config.ds_config["monitor_config"]["wandb"].pop("config", {})
        return monitor
    else:
        return DummyDeepSpeedMonitor(config.ds_config)
    
class BaseMonitor:
    """BaseMonitor是一个基础的监控器类，用于记录模型训练过程中的统计信息
        其中，`trainer` 会将需要统计的数据存放到 `item` 中，目前 item 的内容为：

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
            
class NetworkIOMonitor(BaseMonitor):
    """ 用来记录每个step的网络带宽情况
    """
    def __init__(self, config, interface: Optional[str] = None) -> None:
        super().__init__(config)
        self.interface = interface
        start_time = time.time()
        start_info = psutil.net_io_counters(pernic=self.interface is not None, nowrap=True)
        time.sleep(5)
        end_time = time.time()
        end_info = psutil.net_io_counters(pernic=self.interface is not None, nowrap=True)
        if self.interface is not None:
            assert isinstance(self.end_info, (dict, Dict)) and self.interface in self.end_info.keys(), f"Wrong interface! Interface can only be selected in {list(self.end_info.keys())}."
            self.start_info = self.start_info[self.interface]
            self.end_info = self.end_info[self.interface]
        self.norm_sent = (end_info.bytes_sent - start_info.bytes_sent) / (end_time - start_time)
        self.norm_recv = (end_info.bytes_recv - start_info.bytes_recv) / (end_time - start_time)
        
    def __enter__(self):
        self.start_time = time.time()
        self.start_info = psutil.net_io_counters(pernic=self.interface is not None, nowrap=True)
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.item["mode"] == "train":
            self.end_time = time.time()
            self.end_info = psutil.net_io_counters(pernic=self.interface is not None, nowrap=True)
            if self.interface is not None:
                assert isinstance(self.end_info, (dict, Dict)) and self.interface in self.end_info.keys(), f"Wrong interface! Interface can only be selected in {list(self.end_info.keys())}."
                self.start_info = self.start_info[self.interface]
                self.end_info = self.end_info[self.interface]
            self.monitor.write_events([(f"Network IO (Sent) bytes/second", ((self.end_info.bytes_sent - self.start_info.bytes_sent) / (self.end_time - self.start_time)) - self.norm_sent, self.item['global_batch_idx'])])
            self.monitor.write_events([(f"Network IO (Recv) bytes/second", ((self.end_info.bytes_recv - self.start_info.bytes_recv) / (self.end_time - self.start_time)) - self.norm_recv, self.item['global_batch_idx'])])
            
class DiskIOMonitor(BaseMonitor):
    """ 用来记录每个step的硬盘读写情况
    """
    def __init__(self, config, disk: Optional[str] = None) -> None:
        super().__init__(config)
        self.disk = disk
        
    def __enter__(self):
        self.start_time = time.time()
        self.start_info = psutil.disk_io_counters(perdisk=self.disk is not None, nowrap=True)
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.item["mode"] == "train":
            self.end_time = time.time()
            self.end_info = psutil.disk_io_counters(perdisk=self.disk is not None, nowrap=True)
            if self.disk is not None:
                assert isinstance(self.end_info, (dict, Dict)) and self.disk in self.end_info.keys(), f"Wrong disk! Disk can only be selected in {list(self.end_info.keys())}."
                self.start_info = self.start_info[self.disk]
                self.end_info = self.end_info[self.disk]
            self.monitor.write_events([(f"Disk IO (Read) bytes/second", (self.end_info.read_bytes - self.start_info.read_bytes) / (self.end_time - self.start_time), self.item['global_batch_idx'])])
            self.monitor.write_events([(f"Disk IO (Write) bytes/second", (self.end_info.write_bytes - self.start_info.write_bytes) / (self.end_time - self.start_time), self.item['global_batch_idx'])])
            
class TGSMonitor(BaseMonitor):
    """ 用来记录每秒每张 GPU 可训练的 token 数 (token / s / GPU)
    """
    def __enter__(self):
        self.start = time.time()
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.item["mode"] == "train" and "batch" in self.item.keys():
            self.monitor.write_events([(f"TGS", reduce(lambda x, y: x * y, self.item["batch"]["input_ids"].shape) / (env.pp_size * env.tp_size * (time.time() - self.start)), self.item['global_batch_idx'])])
            
class CPUMemoryMonitor(BaseMonitor):
    """ 用来记录每个step的CPU内存占用
    """
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.item["mode"] == "train":
            self.monitor.write_events([(f"CPU Memory Used", psutil.virtual_memory().used, self.item['global_batch_idx'])])
        
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
    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'eval_result' in self.item.keys() and 'global_batch_idx' in self.item.keys() and self.item["mode"] == "eval":
            for key, value in self.item['eval_result'].items():
                self.monitor.write_events([(f"Metric {key}", value, self.item["global_batch_idx"])])
            
class LRMonitor(BaseMonitor):
    """用来记录每个step的learning rate
    """
    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'loss' in self.item.keys() and self.item["mode"] == "train":
            self.monitor.write_events([(f"Learning Rate", self.item['lr'], self.item['global_batch_idx'])])  
        
class _MultiMonitors:
    def __init__(self, monitors: Sequence[BaseMonitor]) -> None:
        self.monitors = monitors
        self.item = {}
    
    def __enter__(self):
        for monitor in self.monitors:
            monitor.__enter__()
        return self.item
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for monitor in self.monitors:
            monitor.item = self.item
            monitor.__exit__(exc_type, exc_val, exc_tb)