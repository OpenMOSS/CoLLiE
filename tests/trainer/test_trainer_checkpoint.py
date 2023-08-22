import os
import random
import shutil

import pytest

from tests.helpers import launch_test


@pytest.mark.parametrize(
    ["model_type", "model_path", "dp_size", "tp_size", "pp_size", "zero"],
    [
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 1, 2, 2, 1],
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 2, 2, 1, 0],
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 2, 2, 1, 3],
    ]
)
def test_checkpoint_collie(model_type, model_path, dp_size, tp_size, pp_size, zero):
    kwargs_list = [
        # save
        {"format": "collie", "load": False, "zero": zero},
        # load to collie
        {"format": "collie", "load": True, "zero": zero},
        # load to hf with collie
        {"format": "hf", "load": True, "zero": 3},
        {"format": "hf", "load": True, "zero": 0}
    ]
    for kwargs in kwargs_list:
        master_port = find_free_port()
        err_log = f"checkpoint_collie::model_{model_path.split('/')[-1]}-dp_{dp_size}-tp_{tp_size}-pp_{pp_size}-zero_{kwargs['zero']}-format_{kwargs['format']}-load_{kwargs['load']}"
        ws = dp_size * tp_size * pp_size
        cmd = f"torchrun --master_port {master_port} --nproc_per_node {ws} " \
            f"_test_checkpoint_zero.py --model_type {model_type} " \
            f"--model_path {model_path} --dp_size {dp_size} " \
            f"--tp_size {tp_size} --pp_size {pp_size} --folder _ckpt "
        load = kwargs.pop("load")
        extra = " ".join([f"--{key} {value}" for key, value in kwargs.items()])
        if load:
            extra += " --load"
        cmd += extra
        print(cmd)
        launch_test(cmd, err_log)
    shutil.rmtree("_ckpt")
    os.system("rm logits zero_save_*")

@pytest.mark.parametrize(
    ["model_type", "model_path", "dp_size", "tp_size", "pp_size", "zero"],
    [
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 4, 1, 1, 3],
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 2, 2, 1, 0],
    ]
)
def test_checkpoint_hf(model_type, model_path, dp_size, tp_size, pp_size, zero):
    kwargs_list = [
        # save
        {"format": "hf", "load": False, "zero": zero},
        # load to collie
        {"format": "collie", "load": True, "zero": 3},
        {"format": "collie", "load": True, "zero": 0},
        # load to hf with collie
        {"format": "collie", "load": True, "zero": zero},
    ]
    for kwargs in kwargs_list:
        master_port = find_free_port()
        err_log = f"checkpoint_hf::model_{model_path.split('/')[-1]}-dp_{dp_size}-tp_{tp_size}-pp_{pp_size}-zero_{kwargs['zero']}-format_{kwargs['format']}-load_{kwargs['load']}"
        ws = dp_size * tp_size * pp_size
        cmd = f"torchrun --master_port {master_port} --nproc_per_node {ws} " \
            f"_test_checkpoint_zero.py --model_type {model_type} " \
            f"--model_path {model_path} --dp_size {dp_size} " \
            f"--tp_size {tp_size} --pp_size {pp_size} --folder _ckpt "
        load = kwargs.pop("load")
        extra = " ".join([f"--{key} {value}" for key, value in kwargs.items()])
        if load:
            extra += " --load"
        cmd += extra
        launch_test(cmd, err_log)
    shutil.rmtree("_ckpt")
    os.system("rm logits zero_save_*")

@pytest.mark.parametrize(
    ["model_type", "model_path", "dp_size", "tp_size", "pp_size", "zero"],
    [
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 2, 2, 1, 3],
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 2, 2, 1, 0],
    ]
)
def test_checkpoint_lomo(model_type, model_path, dp_size, tp_size, pp_size, zero):
    kwargs_list = [
        # save
        {"format": "collie", "load": False, "zero": zero},
        # load to collie
        {"format": "collie", "load": True, "zero": zero},
        # load to hf with collie
        {"format": "hf", "load": True, "zero": 3},
        {"format": "hf", "load": True, "zero": 0}
    ]
    for kwargs in kwargs_list:
        master_port = find_free_port()
        err_log = f"checkpoint_lomo::model_{model_path.split('/')[-1]}-dp_{dp_size}-tp_{tp_size}-pp_{pp_size}-zero_{kwargs['zero']}-format_{kwargs['format']}-load_{kwargs['load']}"
        ws = dp_size * tp_size * pp_size
        cmd = f"torchrun --master_port {master_port} --nproc_per_node {ws} " \
            f"_test_checkpoint_zero.py --model_type {model_type} " \
            f"--model_path {model_path} --dp_size {dp_size} " \
            f"--tp_size {tp_size} --pp_size {pp_size} --folder _ckpt --lomo "
        load = kwargs.pop("load")
        extra = " ".join([f"--{key} {value}" for key, value in kwargs.items()])
        if load:
            extra += " --load"
        cmd += extra
        launch_test(cmd, err_log)
    shutil.rmtree("_ckpt")
    os.system("rm logits zero_save_*")