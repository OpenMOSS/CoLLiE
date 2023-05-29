from deepspeed.monitor.monitor import MonitorMaster, Monitor

from collie.config import CollieConfig
from collie.log.print import print

import time

class DummyMonitor(Monitor):
    def __init__(self, monitor_config=None) -> None:
        pass
    
    def write_events(self, event_list):
        pass

def get_monitor(config: CollieConfig):
    if "monitor_config" in config.ds_config.keys():
        return MonitorMaster(config.ds_config["monitor_config"])
    else:
        return DummyMonitor(config.ds_config["monitor_config"])
    
def monitor_step_time(monitor: Monitor, name: str = "time per step", step_idx: int = 0):
    start = time.time()
    yield
    monitor.write_events([(name, time.time() - start, step_idx)])