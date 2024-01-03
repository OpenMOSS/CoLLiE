import os
import functools
import inspect
from inspect import Parameter
from accelerate.utils.modeling import find_tied_parameters
import dataclasses
from types import MethodType
from typing import (Callable, Any, Dict, Union, Mapping, Sequence, Tuple,
                    Optional, List, AnyStr)
from collections import defaultdict, OrderedDict
from operator import length_hint
from copy import deepcopy

import torch

from collie.log.logger import logger
from .rich_progress import f_rich_progress

__all__ = ["find_tensors", "progress", "dictToObj", "apply_to_collection", 
           "dict_as_params", "initization_mapping", "is_static_method", 
           "auto_param_call", "get_keys_to_not_convert", "concat_tensor"]

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


def concat_tensor(tensor_list, dim=0):
    """
    拼接 ``tensor_list`` 中的张量，并且在拼接时将张量转移到 cpu 上来避免显存的增加。

    :return: 拼接后位于 cpu 上的张量
    """
    tensor_list_cpu = [t.detach().cpu().clone() for t in tensor_list]
    tensor_list.clear()
    # del tensor_list
    ret = torch.cat(tensor_list_cpu, dim=dim)
    return ret

def stack_tensor(tensor_list, dim=0):
    """
    叠加 ``tensor_list`` 中的张量，并且在叠加时将张量转移到 cpu 上来避免显存的增加。

    :return: 叠加后位于 cpu 上的张量
    """
    tensor_list_cpu = [t.detach().cpu().clone() for t in tensor_list]
    tensor_list.clear()
    # del tensor_list
    ret = torch.stack(tensor_list_cpu, dim=dim)
    return ret

        
class progress:
    """包装了 ``rich`` 进度条的类。

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
            post_desc=post_desc, visible=not disable, total=self.total
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

    def reset(
        self, desc: Optional[str] = None, total: Optional[float] = None, completed: int = 0, visible: Optional[bool] = None, 
        post_desc: Optional[str] = None,
    ):
        """
        重置进度条，可以重置进度条的显示时间。

        :param desc: 进度条最左侧的描述语句。
        :param total: 遍历对象的总数。如果为 ``None`` 则不会发生改变。
        :param completed: 标识进度条的总进度。
        :param visible: 调整进度条是否可见。
        :param post_desc: 进度条最右侧的补充描述语句。
        """
        if post_desc is None:
            self.bar.reset(self.task_id, description=desc, total=total,
                           completed=completed, visible=visible)
        else:
            self.bar.reset(self.task_id, description=desc, total=total,
                           completed=completed, visible=visible,
                           post_desc=post_desc)

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
            
def _split_past_key_values(past_key_values, micro_batch_size, micro_batch_num):
    if micro_batch_num == 1:
        return (past_key_values,)
    past_kv_split = [() for _ in range(micro_batch_num)]
    for layer_past in past_key_values:
        if isinstance(layer_past, (tuple, list)):
            assert len(layer_past) == 2
        else:
            # prefix tuning 的 past key values 是个 tensor
            assert isinstance(layer_past, torch.Tensor)
            assert layer_past.shape[0] == 2
        key_split = torch.split(layer_past[0], micro_batch_size)
        value_split = torch.split(layer_past[1], micro_batch_size)
        assert len(key_split) == micro_batch_num, len(key_split)
        assert len(value_split) == micro_batch_num, len(value_split)
        for i in range(micro_batch_num):
            past_kv_split[i] += ((key_split[i], value_split[i]),)
    return tuple(past_kv_split)


def _split_dict(inputs, micro_batch_size, micro_batch_num):
    inputs_split = {}
    for key in list(inputs.keys()):
        dim = 0
        if key == "past_key_values":
            dim = 2
        if isinstance(inputs[key], torch.Tensor):
            inputs_split[key] = torch.split(inputs[key], micro_batch_size, dim)
        elif isinstance(inputs[key], Sequence):
            inputs_split[key] = [torch.split(input_, micro_batch_size, dim) for input_ in inputs[key]]
            inputs_split[key] = list(zip(*inputs_split[key]))
    inputs_split = [{key: value[i] for key, value in inputs_split.items()} for i in range(micro_batch_num)]
    return inputs_split

def _split_batch(batch, micro_batch_size, micro_batch_num):
    """
    将 ``batch`` 划分为 ``micro_batch_num`` 个 ``micro_batch_size`` 大小。

    仅在流水线情况的训练和验证中用到。

    :param batch: tuple from dataloader
    :param micro_batch_size:
    :param micro_batch_num:
    :return: tuple
    """
    if isinstance(batch, torch.Tensor):
        batch_split = torch.split(batch, micro_batch_size)
    elif isinstance(batch, dict):
        batch_split = _split_dict(batch, micro_batch_size, micro_batch_num)
    else:
        raise NotImplementedError(f"Invalid type of batch: {type(batch)}"
                                  "Must be Tensor or dict.")
    assert len(batch_split) == micro_batch_num, len(batch_split)

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

        >>> print(_get_fun_msg(_get_fun_msg))
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
    """ 使用字典作为参数输入的辅助函数
    
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

    .. note::

        在使用该函数时，请您注意输入输出顺序和 ``input_keys`` ``output_keys`` 顺序
        的对应关系，避免将错误的 key 赋给了对应的张量。
    """
    def _inner(cls: type, *args, **kwargs):
        obj = cls(*args, **kwargs)
        object.__setattr__(obj, "dict_as_params_input_keys", input_keys)
        object.__setattr__(obj, "dict_as_params_output_keys", output_keys)
        raw_forward = obj.forward
        def _forward(self, dict_inputs: dict):
            if isinstance(input_keys, str):
                inputs = [dict_inputs[input_keys]]
            elif isinstance(input_keys, Sequence):
                inputs = [dict_inputs[k] for k in input_keys]
            else:
                raise ValueError(f"input_keys should be str or Sequence[str], but got {type(input_keys)}")
            outputs = raw_forward(*inputs)
            if isinstance(output_keys, str):
                dict_inputs[output_keys] = outputs
                for k, v in dict_inputs.items():
                    if k != output_keys and k != "past_key_values":
                        dict_inputs[k] = v.detach()
            elif isinstance(output_keys, Sequence):
                assert isinstance(outputs, Sequence) and len(outputs) == len(output_keys), \
                    "outputs should be Sequence and have the same length as output_keys"
                for k, v in zip(output_keys, outputs):
                    dict_inputs[k] = v
                for k, v in dict_inputs.items():
                    if k not in output_keys and k != "past_key_values":
                        dict_inputs[k] = v.detach()
            else:
                raise ValueError(f"output_keys should be str or Sequence[str], but got {type(output_keys)}")
            return dict_inputs
        object.__setattr__(obj, "raw_forward", raw_forward)
        obj.forward = MethodType(_forward, obj)
        return obj
    return _inner

def is_static_method(func):
    """ 判断一个函数是否是静态方法。
    """
    if inspect.isfunction(func):
        if inspect.ismethod(func):
            # 对于绑定方法，检查其是否由staticmethod装饰器修饰
            return isinstance(func.__func__, staticmethod)
        else:
            # 对于普通函数，检查其是否由staticmethod装饰器修饰
            return isinstance(func, staticmethod)
    return False

def auto_param_call(fn: Callable, *args, signature_fn: Optional[Callable] = None,
                    mapping: Optional[Dict] = None) -> Any:
    r"""
    该函数会根据输入函数的形参名从 ``*args`` （均为 **dict** 类型）中找到匹配的值进行调用，如果传入的数据与 ``fn`` 的形参不匹配，可以通过
    ``mapping`` 参数进行转换。``mapping`` 参数中的一对 ``(key, value)`` 表示在 ``*args`` 中找到 ``key`` 对应的值，并将这个值传递给形参中名为
    ``value`` 的参数。

    1. 该函数用来提供给用户根据字符串匹配从而实现自动调用；
    2. 注意 ``mapping`` 默认为 ``None``，如果您希望指定输入和运行函数的参数的对应方式，那么您应当让 ``mapping`` 为一个字典传入进来；
       如果 ``mapping`` 不为 ``None``，那么我们一定会先使用 ``mapping`` 将输入的字典的 ``keys`` 修改过来，因此请务必亲自检查 ``mapping`` 的正确性；
    3. 如果输入的函数的参数有默认值，那么如果之后的输入中没有该参数对应的值，我们就会使用该参数对应的默认值，否则也会使用之后的输入的值；
    4. 如果输入的函数是一个 ``partial`` 函数，情况同第三点，即和默认参数的情况相同；

    Examples::

        >>> # 1
        >>> loss_fn = CrossEntropyLoss()  # 如果其需要的参数为 def CrossEntropyLoss(y, pred)；
        >>> batch = {"x": 20, "y": 1}
        >>> output = {"pred": 0}
        >>> acc = auto_param_call(loss_fn, batch, output)

        >>> # 2
        >>> def test_fn(x, y, a, b=10):
        >>>     return x + y + a + b
        >>> print(auto_param_call(test_fn, {"x": 10}, {"y": 20, "a": 30}))  # res: 70
        >>> print(auto_param_call(partial(test_fn, a=100), {"x": 10}, {"y": 20}))  # res: 140
        >>> print(auto_param_call(partial(test_fn, a=100), {"x": 10}, {"y": 20, "a": 200}))  # res: 240

    :param fn: 用来进行实际计算的函数，其参数可以包含有默认值；
    :param args: 一系列的位置参数，应当为一系列的字典，我们需要从这些输入中提取 ``fn`` 计算所需要的实际参数；
    :param signature_fn: 函数，用来替换 ``fn`` 的函数签名，如果该参数不为 ``None``，那么我们首先会从该函数中提取函数签名，
        然后通过该函数签名提取参数值后，再传给 ``fn`` 进行实际的运算；
    :param mapping: 一个字典，用来更改其前面的字典的键值；

    :return:  ``fn`` 运行的结果；
    """

    if signature_fn is not None:
        if not callable(signature_fn):
            raise ValueError(f"Parameter `signature_fn` should be `Callable`.")
        _need_params = OrderedDict(inspect.signature(signature_fn).parameters)
    else:
        _need_params = OrderedDict(inspect.signature(fn).parameters)
    _kwargs = None
    for _name, _param in _need_params.items():
        if _param.kind == Parameter.VAR_POSITIONAL:
            fn_msg = _get_fun_msg(fn if signature_fn is None else signature_fn)
            raise ValueError(f"It is not allowed to have parameter `*args` in your function:{fn_msg}.")
        if _param.kind == Parameter.VAR_KEYWORD:
            _kwargs = (_name, _param)

    if _kwargs is not None:
        _need_params.pop(_kwargs[0])

    _default_params = {}
    for _name, _param in _need_params.items():
        if _param.default != Parameter.empty:
            _default_params[_name] = _param.default

    if mapping is not None:
        fn_msg = _get_fun_msg(fn if signature_fn is None else signature_fn)
        assert isinstance(mapping, Dict), f"Exception happens when calling {fn_msg}. " \
                                          f"Parameter `mapping` should be of 'Dict' type, instead of {type(mapping)}."

    _has_params = {}
    duplicate_names = []
    for arg in args:
        if not isinstance(arg, (Dict, dict)):
            fn_msg = _get_fun_msg(fn if signature_fn is None else signature_fn)
            raise TypeError(f"Exception happens when calling {fn_msg}. "
                            f"The input part of function `auto_param_call` must be `Dict` type, instead of {type(arg)}.")
        for _name, _value in arg.items():
            if mapping is not None and _name in mapping:
                _name = mapping[_name]

            if _name not in _has_params:
                if _kwargs is not None or _name in _need_params:
                    _has_params[_name] = _value
            # 同一参数对象在两个输入的资源中都出现，造成混淆；
            elif _name in _need_params and not (_has_params[_name] is _value):
                duplicate_names.append(_name)
    if duplicate_names:
        fn_msg = _get_fun_msg(fn if signature_fn is None else signature_fn)
        raise ValueError(f"The following key present in several inputs:{duplicate_names} when calling {fn_msg}.")

    # 将具有默认值但是没有被输入修改过的参数值传进去；
    for _name, _value in _default_params.items():
        if _name not in _has_params:
            _has_params[_name] = _value

    if len(_has_params) < len(_need_params):
        miss_params = list(set(_need_params.keys()) - set(_has_params.keys()))
        fn_msg = _get_fun_msg(fn if signature_fn is None else signature_fn)
        _provided_keys = _get_keys(args)
        raise ValueError(f"The parameters:`{miss_params}` needed by function:{fn_msg} "
                         f"are not found in the input keys({_provided_keys}).")

    return fn(**_has_params)

def _get_keys(args:List[Dict]) -> List[List[str]]:
    """
    返回每个 dict 的 keys

    :param args:
    :return:
    """
    _provided_keys = []
    for arg in args:
        _provided_keys.append(list(arg.keys()))
    return _provided_keys

def get_keys_to_not_convert(model):
    r"""
    An utility function to get the key of the module to keep in full precision if any For example for CausalLM modules
    we may want to keep the lm_head in full precision for numerical stability reasons. For other architectures, we want
    to keep the tied weights of the model. The function will return a list of the keys of the modules to not convert in
    int8.

    :param model: Input model
    """

    tied_params = find_tied_parameters(model)
    # For compatibility with Accelerate < 0.18
    if isinstance(tied_params, dict):
        tied_keys = list(tied_params.values())
    else:
        tied_keys = sum([x[1:] for x in tied_params], [])
    has_tied_params = len(tied_keys) > 0

    # otherwise they have an attached head
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]

    # add last module together with tied weights
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = tied_keys + list(intersection)

    # remove ".weight" from the keys
    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    return filtered_module_names