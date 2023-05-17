from deepspeed.monitor.monitor import MonitorMaster, Monitor

from collie.trainer.arguments import Arguments
from collie.log.print import print

import time

class DummyMonitor(Monitor):
    def __init__(self, monitor_config=None) -> None:
        pass
    
    def write_events(self, event_list):
        pass

def get_monitor(args: Arguments):
    if "monitor_config" in args.ds_config.keys():
        return MonitorMaster(args.ds_config["monitor_config"])
    else:
        return DummyMonitor(args.ds_config["monitor_config"])
    
def monitor_step_time(monitor: Monitor, name: str = "time per step", step_idx: int = 0):
    start = time.time()
    yield
    monitor.write_events([(name, time.time() - start, step_idx)])