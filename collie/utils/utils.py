import os
import functools
import inspect
import dataclasses
from types import MethodType
from typing import (Callable, Any, Dict, Union, Mapping, Sequence, Tuple,
                    Optional, List)
from collections import defaultdict, OrderedDict
from operator import length_hint
from copy import deepcopy

import torch

from collie.log.logger import logger
from .rich_progress import f_rich_progress

__all__ = ["find_tensors", "progress", "dictToObj", "apply_to_collection", 
           "dict_as_params"]

def find_tensors():
    """
    打印出垃圾回收区的所有张量。

    Adopted from https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/3
    """
    import torch
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.dtype, obj.device)
        except:
            pass

        
class progress:
    """
    包装了 ``rich`` 进度条的类。

    .. code-block::

        for batch in progress(dataloader):
            # do something

    .. code-block::

        with progress(dataloader) as bar:
            for batch in bar:
                # do something
                bar.set_postfix(Loss=1.0)

    .. code-block::

        bar = progress(dataloader)
        for batch in bar:
            # do something
            bar.set_postfix(Loss=1.0)

    :param sequence: 需要遍历的序列，需要是一个可以迭代的对象。
    :param desc: 进度条最左侧的描述语句。
    :param total: 遍历对象的总数。如果为 ``None`` 则会自动进行计算。
    :param completed: 标识进度条的总进度。
    :param upgrade_period: 进度条更新的时间间隔。
    :param disable: 调整进度条是否可见。
    :param post_desc: 进度条最右侧的补充描述语句。
    """
    def __init__(self, sequence, desc="Workin on...", total=None, completed=0,
                 upgrade_period=0.1, disable=False, post_desc: str = ""):
        self.bar = f_rich_progress
        self.bar.set_disable(disable)
        self.sequence = sequence

        self.total = float(length_hint(sequence)) if total is None else total
        self.completed = completed
        self.task_id = self.bar.add_task(
            desc, upgrade_period=upgrade_period, completed=completed,
            post_desc=post_desc, visible=not disable, total=total
        )

    def __iter__(self):
        yield from self.bar.track(
            self.sequence, task_id=self.task_id, total=self.total)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ...

    def __del__(self):
        self.bar.destroy_task(self.task_id)

    def set_post_desc(self, post_desc: str):
        """
        设置进度条最右侧的补充描述语句。

        .. code-block::

            bar = progress(dataloader)
            dataloader.set_post_desc("Loss=1.0")
        """
        self.bar.update(self.task_id, post_desc=post_desc, advance=0)

    def set_postfix(self, **kwargs):
        """
        设置进度条最右侧的补充描述语句。

        对于传入的每一对 key 和 value 将以 ``key1: value1, key2: value2, ..``
        的格式进行显示。

        .. code-block::

            bar = progress(dataloader)
            dataloader.set_postfix(Loss=1.0, Batch=1)
        """
        post_desc = ", ".join([f"{k}: {v}" for k, v in kwargs.items()])
        self.set_post_desc(post_desc)

    def set_description(self, desc):
        """
        设置进度条最左侧的描述语句。
        """
        self.update(desc=desc)

    def update(
        self, desc: Optional[str] = None, total: Optional[float] = None,
        completed: Optional[float] = None, advance: Optional[float] = None,
        visible: Optional[bool] = None, refresh: bool = False,
        post_desc: Optional[str] = None,
    ) -> None:
        """
        对进度条的内容进行更新，可以更加详细地改变进度条的内容。

        :param desc: 进度条最左侧的描述语句。
        :param total: 遍历对象的总数。如果为 ``None`` 则不会发生改变。
        :param completed: 标识进度条的总进度。
        :param advance: 该次进度条更新的进度。
        :param visible: 调整进度条是否可见。
        :param refresh: 是否强制刷新进度条。
        :param post_desc: 进度条最右侧的补充描述语句。
        """
        if post_desc is None:
            self.bar.update(self.task_id, total=total, completed=completed,
                        advance=advance, description=desc, visible=visible,
                        refresh=refresh)
        else:
            self.bar.update(self.task_id, total=total, completed=completed,
                        advance=advance, description=desc, visible=visible,
                        refresh=refresh, post_desc=post_desc)

def _split_batch(batch, micro_batch_size, micro_batch_num):
    """
    将 ``batch`` 划分为 ``micro_batch_num`` 个 ``micro_batch_size`` 大小。

    仅在流水线情况的训练和验证中用到。

    :param batch: tuple from dataloader
    :param micro_batch_size:
    :param micro_batch_num:
    :return: tuple
    """
    # Assume batch first.
    assert len(batch) == 2, len(batch)
    inputs = batch[0]
    labels = batch[1]
    # micro_batch_num = inputs.shape[0] // micro_batch_size
    if isinstance(labels, Sequence):
        labels_split = [torch.split(label, micro_batch_size) for label in labels]
        labels_split = list(zip(*labels_split))
    elif isinstance(labels, torch.Tensor):
        labels_split = torch.split(labels, micro_batch_size)
    elif isinstance(labels, dict):
        labels_split = {}
        for key in list(labels.keys()):
            if isinstance(labels[key], torch.Tensor):
                labels_split[key] = torch.split(labels[key], micro_batch_size)
            elif isinstance(labels[key], Sequence):
                labels_split[key] = [torch.split(label, micro_batch_size) for label in labels[key]]
                labels_split[key] = list(zip(*labels_split[key]))
        labels_split = [{key: value[i] for key, value in labels_split.items()} for i in range(micro_batch_num)]
    if isinstance(inputs, torch.Tensor):
        inputs_split = torch.split(inputs, micro_batch_size)
        assert len(inputs_split) == micro_batch_num, len(inputs_split)
    elif isinstance(inputs, dict):
        inputs_split = {}
        for key in list(inputs.keys()):
            if isinstance(inputs[key], torch.Tensor):
                inputs_split[key] = torch.split(inputs[key], micro_batch_size)
            elif isinstance(inputs[key], Sequence):
                inputs_split[key] = [torch.split(input_, micro_batch_size) for input_ in inputs[key]]
                inputs_split[key] = list(zip(*inputs_split[key]))
        inputs_split = [{key: value[i] for key, value in inputs_split.items()} for i in range(micro_batch_num)]
    else:
        inputs_split = (torch.split(input_, micro_batch_size) for input_ in inputs)
        inputs_split = list(zip(*inputs_split))
    
    batch_split = ()
    for input_split, label_split in zip(inputs_split, labels_split):
        batch_split += ((input_split, label_split), )

    return batch_split

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d

def _is_namedtuple(obj: object) -> bool:
    # https://github.com/pytorch/pytorch/blob/v1.8.1/torch/nn/parallel/scatter_gather.py#L4-L8
    return isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")


def _is_dataclass_instance(obj: object) -> bool:
    # https://docs.python.org/3/library/dataclasses.html#module-level-decorators-classes-and-functions
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def apply_to_collection(
        data: Any,
        dtype: Union[type, Any, Tuple[Union[type, Any]]],
        function: Callable,
        *args: Any,
        wrong_dtype: Optional[Union[type, Tuple[type]]] = None,
        include_none: bool = True,
        **kwargs: Any,
) -> Any:
    """
    递归地对 ``data`` 中的元素执行函数 ``function``，且仅在满足元素为 ``dtype`` 时执行。

    该函数参考了 `pytorch-lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_ 的实现

    :param data: 需要进行处理的数据集合或数据；
    :param dtype: 数据的类型，函数 ``function`` 只会被应用于 ``data`` 中类型为 ``dtype`` 的数据；
    :param function: 对数据进行处理的函数；
    :param args: ``function`` 所需要的其它参数；
    :param wrong_dtype: ``function`` 一定不会生效的数据类型。
        如果数据既是 ``wrong_dtype`` 类型又是 ``dtype`` 类型那么也不会生效；
    :param include_none: 是否包含执行结果为 ``None`` 的数据，默认为 ``True``；
    :param kwargs: ``function`` 所需要的其它参数；
    :return: 经过 ``function`` 处理后的数据集合；
    """
    # Breaking condition
    if isinstance(data, dtype) and (wrong_dtype is None or not isinstance(data, wrong_dtype)):
        return function(data, *args, **kwargs)

    elem_type = type(data)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        out = []
        for k, v in data.items():
            v = apply_to_collection(
                v, dtype, function, *args, wrong_dtype=wrong_dtype, include_none=include_none, **kwargs
            )
            if include_none or v is not None:
                out.append((k, v))
        if isinstance(data, defaultdict):
            return elem_type(data.default_factory, OrderedDict(out))
        return elem_type(OrderedDict(out))

    is_namedtuple = _is_namedtuple(data)
    is_sequence = isinstance(data, Sequence) and not isinstance(data, str)
    if is_namedtuple or is_sequence:
        out = []
        for d in data:
            v = apply_to_collection(
                d, dtype, function, *args, wrong_dtype=wrong_dtype, include_none=include_none, **kwargs
            )
            if include_none or v is not None:
                out.append(v)
        return elem_type(*out) if is_namedtuple else elem_type(out)

    if _is_dataclass_instance(data):
        # make a deepcopy of the data,
        # but do not deepcopy mapped fields since the computation would
        # be wasted on values that likely get immediately overwritten
        fields = {}
        memo = {}
        for field in dataclasses.fields(data):
            field_value = getattr(data, field.name)
            fields[field.name] = (field_value, field.init)
            memo[id(field_value)] = field_value
        result = deepcopy(data, memo=memo)
        # apply function to each field
        for field_name, (field_value, field_init) in fields.items():
            if field_init:
                v = apply_to_collection(
                    field_value,
                    dtype,
                    function,
                    *args,
                    wrong_dtype=wrong_dtype,
                    include_none=include_none,
                    **kwargs,
                )
            if not field_init or (not include_none and v is None):  # retain old value
                v = getattr(data, field_name)
            setattr(result, field_name, v)
        return result

    # data is neither of dtype, nor a collection
    return data

def _get_fun_msg(fn, with_fp=True)->str:
    """
    获取函数的基本信息，帮助报错::

        >>>> print(_get_fun_msg(_get_fun_msg))
        `_get_fun_msg(fn) -> str`(In file:/Users/hnyan/Desktop/projects/fastNLP/fastNLP/fastNLP/core/utils/utils.py)

    :param callable fn:
    :param with_fp: 是否包含函数所在的文件信息；
    :return:
    """
    if isinstance(fn, functools.partial):
        return _get_fun_msg(fn.func)
    try:
        fn_name = fn.__qualname__ + str(inspect.signature(fn))
    except:
        fn_name = str(fn)
    if with_fp:
        try:
            fp = '(In file:' + os.path.abspath(inspect.getfile(fn)) + ')'
        except:
            fp = ''
    else:
        fp = ''
    msg = f'`{fn_name}`' + fp
    return msg

def _check_valid_parameters_number(fn,
                                   expected_params: List[str],
                                   fn_name=None):
    r"""检查一个函数是否需要 expected_params 参数(检测数量是否匹配)。除掉 self （如
    果是method），给定默认值的参数等。如果匹配不上，就会进行报错。

    :param fn: 需要检测的函数，可以是 method 或者 function 。
    :param expected_params: 期待应该支持的参数。
    :param fn_name: fn 的名字，当传入的 fn 不是 callable 的时候方便报错。
    :return:
    """
    if fn_name is not None:
        assert callable(
            fn), f'`{fn_name}` should be callable, instead of `{type(fn)}`.'

    try:
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        name = ''
        if isinstance(fn, functools.partial) and not hasattr(fn, '__name__'):
            name = 'partial:'
            f = fn.func
            while isinstance(f, functools.partial):
                name += 'partial:'
                f = f.func
            fn.__name__ = name + f.__name__  # type: ignore
        inspect.getcallargs(fn, *args, *expected_params, **kwargs)
        if name:  # 如果一开始没有name的，需要给人家删除掉
            delattr(fn, '__name__')

    except TypeError as e:
        logger.error(
            f'The function:{_get_fun_msg(fn)} will be provided with '
            f'parameters:{expected_params}. The following exception will '
            'happen.')
        raise e

def dict_as_params(input_keys: Union[str, Sequence[str]], output_keys: Union[str, Sequence[str]]):
    """
    从输入的字典中顺次取出 ``input_keys`` 作为模型的输入，并且将模型的输出以
    ``output_keys`` 为 key 放入字典中作为输出。在这一过程中多余的 key 并不会被丢
    弃。

    可以用于 ``nn.LayerNorm`` 这些在流水线并行中一般不需要改变 forward 过程但需要
    改变输入输出结构的模型，使用该函数可以避免频繁地重写这些模型，并且可以适应流水线
    ``LayerSpec`` 的初始化。

    .. code-block::
        dict_as_params(input_keys="input_ids", output_keys="hidden_states")(nn.Embedding, vocab_size, hidden_size)

        LayerSpec(
            dict_as_params(input_keys="input_ids", output_keys="hidden_states"),
            nn.Embbedding, vocab_size, hidden_size
        )

    :param input_keys: 该模型输入需要的 key。``dict_as_params`` 会从输入的字典中
        依次取出 ``input_keys`` 的内容传入模型。
    :param output_keys: 该模型输出对应的 key。``dict_as_params`` 会依次将模型的
        输出和 ``output_keys`` 进行对应，并放入字典中作为最终的输出。

    .. info::

        在使用该函数时，请您注意输入输出顺序和 ``input_keys`` ``output_keys`` 顺序
        的对应关系，避免将错误的 key 赋给了对应的张量。
    """
    def _inner(cls: type, *args, **kwargs):
        obj = cls(*args, **kwargs)
        raw_foward = obj.forward
        def _forward(self, dict_inputs: dict):
            if isinstance(input_keys, str):
                inputs = [dict_inputs[input_keys]]
            elif isinstance(input_keys, Sequence):
                inputs = [dict_inputs[k] for k in input_keys]
            else:
                raise ValueError(f"input_keys should be str or Sequence[str], but got {type(input_keys)}")
            outputs = raw_foward(*inputs)
            if isinstance(output_keys, str):
                dict_inputs[output_keys] = outputs
            elif isinstance(output_keys, Sequence):
                assert isinstance(outputs, Sequence) and len(outputs) == len(output_keys), \
                    "outputs should be Sequence and have the same length as output_keys"
                for k, v in zip(output_keys, outputs):
                    dict_inputs[k] = v
            else:
                raise ValueError(f"output_keys should be str or Sequence[str], but got {type(output_keys)}")
            return dict_inputs
        obj.forward = MethodType(_forward, obj)
        return obj
    return _inner
