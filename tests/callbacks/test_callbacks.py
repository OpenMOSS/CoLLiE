import pytest

from tests.helpers import launch_test, find_free_port

@pytest.mark.parametrize(
    ["model_type", "model_path", "dp_size", "tp_size", "pp_size", "zero"],
    [
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 1, 2, 2, 1],
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 2, 2, 1, 1],
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 2, 2, 1, 3],
    ]
)
def test_load_best_model_callback(model_type, model_path, dp_size, tp_size,
                                  pp_size, zero):
    master_port = find_free_port()
    err_log = f"load_best::model_{model_path.split('/')[-1]}-dp_{dp_size}-tp_{tp_size}-pp_{pp_size}-zero_{zero}"
    ws = dp_size * tp_size * pp_size
    cmd = f"torchrun --master_port {master_port} --nproc_per_node {ws} " \
          f"_test_load_best_callback.py --model_type {model_type} " \
          f"--model_path {model_path} --zero {zero} --dp_size {dp_size} " \
          f"--tp_size {tp_size} --pp_size {pp_size} --folder _load_best"
    launch_test(cmd, err_log)

@pytest.mark.parametrize(
    ["model_type", "model_path", "dp_size", "tp_size", "pp_size", "zero", "model_only"],
    [
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 1, 2, 2, 1, True],
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 2, 2, 1, 1, True],
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 2, 2, 1, 3, True],
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 1, 2, 2, 1, False],
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 2, 2, 1, 1, False],
        ["Moss003MoonForCausalLM", "Salesforce/codegen-350M-mono", 2, 2, 1, 3, False],
    ]
)
def test_checkpoint_callback(model_type, model_path, dp_size, tp_size, pp_size,
                             zero, model_only):
    master_port = find_free_port()
    err_log = f"checkpoint::model_{model_path.split('/')[-1]}-dp_{dp_size}-tp_{tp_size}-pp_{pp_size}-zero_{zero}_modelonly_{model_only}"
    ws = dp_size * tp_size * pp_size
    cmd = f"torchrun --master_port {master_port} --nproc_per_node {ws} " \
          f"_test_checkpoint_callback.py --model_type {model_type} " \
          f"--model_path {model_path} --zero {zero} --dp_size {dp_size} " \
          f"--tp_size {tp_size} --pp_size {pp_size} --folder _ckpt"
    if model_only:
        cmd += " --model_only"
    launch_test(cmd, err_log)
