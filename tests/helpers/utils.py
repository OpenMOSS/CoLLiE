import importlib
import subprocess
import socket

def import_class(model_type):
    from collie import models
    return getattr(models, model_type)

def create_ds_config(fp16=True, zero=0, offload=True, optimizer=None, lr=None):
    init_ds_config = {
        "fp16": {
            "enabled": False
        },
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,
        "steps_per_print": 2147483647,
    }
    if fp16:
        init_ds_config["fp16"]["enabled"] = True
    if zero > 0:
        assert zero <= 3
        zero_optimization = {
            "stage": zero,
        }
        if offload:
            zero_optimization["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": False
            }
        init_ds_config["zero_optimization"] = zero_optimization
    if optimizer is not None:
        init_ds_config["optimizer"] = {
            "type": optimizer,
            "params": {
                "lr": lr,
            }
        }
    return init_ds_config

def launch_test(cmd, err_log=None):
    """

    :param err_log: error log 文件
    """
    if err_log is not None:
        with open(err_log, "w") as fp:
            ret = subprocess.run(cmd, shell=True, check=True, stderr=fp, stdout=fp)
    else:
        ret = subprocess.run(cmd, shell=True, check=True)

def find_free_port() -> str:
    """
    Find a free port from localhost.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port