import os
import gc
import torch
import datetime

from py3nvml import py3nvml

PRINT_TENSOR_SIZES = True
# clears GPU cache frequently, showing only actual memory usage
EMPTY_CACHE = True
gpu_profile_fn = (f"{datetime.datetime.now():%d-%b-%y-%H:%M:%S}"
                  f"-gpu_mem_prof.txt")
if 'GPU_DEBUG' in os.environ:
    print('profiling gpu usage to ', gpu_profile_fn)

_last_tensor_sizes = set()


def _trace_lines(frame, event, arg):
    if event != 'line':
        return
    if EMPTY_CACHE:
        torch.cuda.empty_cache()
    co = frame.f_code
    func_name = co.co_name
    line_no = frame.f_lineno
    filename = co.co_filename
    py3nvml.nvmlInit()
    mem_used = _get_gpu_mem_used()
    where_str = f"{func_name} in {filename}:{line_no}"
    with open(gpu_profile_fn, 'a+') as f:
        f.write(f"{where_str} --> {mem_used:<7.1f}Mb\n")
        if PRINT_TENSOR_SIZES:
            _print_tensors(f, where_str)

    py3nvml.nvmlShutdown()


def trace_calls(frame, event, arg):
    if event != 'call':
        return
    co = frame.f_code
    func_name = co.co_name

    try:
        trace_into = str(os.environ['TRACE_INTO'])
    except:
        print(os.environ)
        exit()
    if func_name in trace_into.split(' '):
        return _trace_lines
    return


def _get_gpu_mem_used():
    handle = py3nvml.nvmlDeviceGetHandleByIndex(
        int(os.environ['GPU_DEBUG']))
    meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.used/1024**2


def _print_tensors(f, where_str):
    global _last_tensor_sizes
    for tensor in _get_tensors():
        if not hasattr(tensor, 'dbg_alloc_where'):
            tensor.dbg_alloc_where = where_str
    new_tensor_sizes = {(x.type(), tuple(x.shape), x.dbg_alloc_where)
                        for x in _get_tensors()}
    for t, s, loc in new_tensor_sizes - _last_tensor_sizes:
        f.write(f'+ {loc:<50} {str(s):<20} {str(t):<10}\n')
    for t, s, loc in _last_tensor_sizes - new_tensor_sizes:
        f.write(f'- {loc:<50} {str(s):<20} {str(t):<10}\n')
    _last_tensor_sizes = new_tensor_sizes


def _get_tensors(gpu_only=True):
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        except Exception as e:
            pass