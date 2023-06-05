import os
import torch
import inspect
import importlib
from abc import abstractmethod
from typing import Union, Optional, Sequence, List
from huggingface_hub import snapshot_download

import deepspeed
from torch import nn
from torch import distributed as dist
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from megatron.core import parallel_state
from transformers.generation.utils import GenerationMixin
from transformers.generation.utils import GenerationConfig
from collie.module import PipelineModel, GPTLMLoss
from collie.config import CollieConfig, load_config
from collie.log import logger
from collie.utils import setup_distribution, is_zero3_enabled, env

class CollieModelForCausalLM(nn.Module, GenerationMixin):
    """
    **CoLLiE** 的基础模型。如果要实现新的模型，必须继承该基类。

    ``CollieModelForCausalLM`` 统一了非流水线模型和流水线模型的接口，并且可以执行
    生成任务。

    为了适应流水线线的生成过程，每个新模型除了实现基类中的抽象方法外，还需要满足：

    1. 每一个 layer 包含一个 ``use_cache`` 属性来决定前向传播中是否使用
       ``past_key_values``。
    2. 将 Attention 过程中产生的 key 和 value 保存在每一个 layer 的
       ``past_key_value`` 属性中。
    3. 将每层的 hidden_states 保存在每一个 layer 的 ``hidden_states`` 属性中。
    4. 将 lm_head 的输入 hidden_states 保存在 ``hidden_states`` 属性中。也可以使
       用 :class:`~collie.module.ColumnParallelLMHead` 来自动地保存。

    """
    main_input_name = "input_ids"
    def __init__(self, config: CollieConfig) -> None:
        super().__init__()
        self.device = torch.device("cuda")
        self.generation_config = GenerationConfig()
        self.config = config
            
    def _get_past_key_values(self, layers: Sequence[nn.Module], attr_name: str="past_key_values"):
        past_key_values = []
        for layer in layers:
            assert hasattr(layer, attr_name), f"{layer} does not have {attr_name}"
            if getattr(layer, attr_name) is not None:
                past_key_values.append(getattr(layer, attr_name))
        return past_key_values if len(past_key_values) > 1 else None
    
    def _clean_past_key_values(self, layers: Sequence[nn.Module], attr_name: str="past_key_values"):
        for layer in layers:
            if hasattr(layer, attr_name):
                object.__setattr__(layer, attr_name, None)
                
    def _set_past_key_values(self, layers: Sequence[nn.Module], past_key_values: List[List[torch.Tensor]], attr_name: str="past_key_values"):
        past_key_values = iter(past_key_values)
        for layer in layers:
            if hasattr(layer, attr_name):
                object.__setattr__(layer, attr_name, next(past_key_values))
            
    def _get_hidden_states(self, layers: Sequence[nn.Module], attr_name: str="hidden_states"):
        past_key_values = []
        for layer in layers:
            assert hasattr(layer, attr_name), f"{layer} does not have {attr_name}"
            if getattr(layer, attr_name) is not None:
                past_key_values.append(getattr(layer, attr_name))
        return past_key_values if len(past_key_values) > 1 else None
    
    def _clean_hidden_states(self, layers: Sequence[nn.Module], attr_name: str="hidden_states"):
        for layer in layers:
            if hasattr(layer, attr_name):
                object.__setattr__(layer, attr_name, None)
                
    def _set_hidden_states(self, layers: Sequence[nn.Module], hidden_states: List[torch.Tensor], attr_name: str="hidden_states"):
        hidden_states = iter(hidden_states)
        for layer in layers:
            if hasattr(layer, attr_name):
                object.__setattr__(layer, attr_name, next(hidden_states))    
                
    def _set_use_cache(self, layers: Sequence[nn.Module], use_cache: bool=True, attr_name: str="use_cache"):
        for layer in layers:
            if hasattr(layer, attr_name):
                object.__setattr__(layer, attr_name, use_cache)    
    
    def can_generate(self) -> bool:
        return True

    def generate(self, *args, **kwargs):
        """
        生成函数。用法同 ``huggingface``。
        """
        res = super().generate(*args, **kwargs)
        self.clean()
        return res

    @classmethod
    def from_config(cls, config: Union[CollieConfig, str], **kwargs):
        """
        从 ``config`` 中加载一个模型。

        :param config: 接受一个字符串或 :class:`.CollieConfig`。为字符串时，会先从
            该 ``str`` 代表的路径或远程链接中加载 config，再进行初始化
        """
        if isinstance(config, str):
            config = CollieConfig.from_pretrained(config, **kwargs)
        setup_distribution(config)
        model_cls = cls._get_model_cls(config)
        if config.pp_size == 1:
            with deepspeed.zero.Init(data_parallel_group=parallel_state.get_data_parallel_group(), enabled=is_zero3_enabled(config)):
                model = super().__new__(model_cls)
                model.__init__(config)
                dist.barrier()
                return model
        else:
            pipeline_model =  PipelineModel(
                layers=model_cls.pipeline_layers(config),
                base_seed=config.seed,
                partition_method=config.pp_partition_method,
                topology=PipeModelDataParallelTopology(
                    num_pp=config.pp_size,
                    num_dp=config.dp_size,
                    num_mp=config.tp_size
                ), loss_fn=GPTLMLoss()
            )
            setattr(pipeline_model, "config", config)
            setattr(pipeline_model, "save_parallel_state_dict", cls.save_parallel_state_dict)
            setattr(pipeline_model, "load_parallel_state_dict", cls.load_parallel_state_dict)
            return pipeline_model
            
    def __new__(cls, config: CollieConfig, **kwargs):
        return cls.from_config(config, **kwargs)
    
    @abstractmethod
    def clean(self):
        """
        清理 ``past_key_value`` 和 ``hidden_states`` 状态的函数。
        """
        raise NotImplementedError(
            "`clean` should be implemented to clear caches for generation."
        )

    @classmethod
    def from_pretrained(cls, model_path_or_name: str, config: Optional[Union[CollieConfig, str]] = None, **kwargs):
        """
        从 ``model_path_or_name`` 中加载预训练好的模型。

        :param model_path_or_name: ``huggingface`` 格式预训练模型的本地路径或名
            称。
        :param config: 可以是字符串或者 :class:`~CollieConfig`。如果为字符串，则会
            使用该字符串代表的模型设置；如果为 ``None``，从 ``model_path_or_name``
            中加载模型设置。

        :param kwargs:
            * process_exclusion - 是否每个 rank 各自独立、互斥地加载模型权重。在模
              型规模较大时，该参数可以帮助节省内存。

            其余 ``kwargs`` 的内容会用于设置 :class:`.CollieConfig` 的内容。
        """
        process_exclusion = kwargs.pop("process_exclusion", False)
        if dist.is_initialized() and process_exclusion:
            logger.warning(
                "Distributed group is not initialized and `process_exclusion` "
                "will not take effect."
            )
        if not os.path.exists(model_path_or_name):
            model_path_or_name = snapshot_download(model_path_or_name)
        if config is None:
            config = model_path_or_name
        if isinstance(config, str):
            # prevent duplicate `from_pretrained`` in load_parallel
            config = CollieConfig.from_pretrained(config, **kwargs)
        model = cls.from_config(config)
        state_dict = {}
        if not is_zero3_enabled(config) or env.dp_rank == 0:
            state_dict = cls.load_parallel_state_dict(
                path=model_path_or_name, config=config,
                process_exclusion=process_exclusion, **kwargs
            )
        if is_zero3_enabled(config):
            for name, param in model.named_parameters():
                with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                    if env.dp_rank == 0:
                        param.data.copy_(state_dict[name].data)
        else:
            model.load_state_dict(state_dict)
        return model
    
    # def save_pretrained(self, **kwargs):
    #     path = kwargs.get("path", kwargs.get("save_directory", None))
    #     assert path is not None, "Please specify `path` or `save_directory`."
    #     self.save_parallel_state_dict(
    #         self.state_dict(),
    #         path=path,
    #         config=self.config,
    #         protocol=kwargs.get("protocol", "file"),
    #         process_exclusion=kwargs.get("process_exclusion", False)
    #     )

    @classmethod
    def pipeline_layers(cls, config: Union[CollieConfig, str]):
        """
        获取流水线模型。

        :return: 一个列表，包含一系列层；这些模型会用于初始化流水线并行模型并且
            进行划分。
        """
        raise NotImplementedError(
            "To use pipeline parallelism, you need to implement "
            "`pipeline_layers` for your model."
        )

    @staticmethod
    @abstractmethod
    def load_parallel_state_dict(path: str, config: Union[CollieConfig, str],
                                 process_exclusion: bool = False, **kwargs):
        """
        从 ``path`` 中加载模型权重。``path`` 中的模型权重应当是 huggingface 格式。

        :param path:
        :param config:
        :param process_exclusion: 是否每个 rank 各自独立、互斥地加载模型权重。在模
            型规模较大时，该参数可以帮助节省内存。
        :return: 一个字典，每个字典都包含当前 rank 上模型需要的权重。
        """
        raise NotImplementedError(
            "Every model should implement `load_parallel_state_dict` "
            "to properly load a state dict for the cuurent rank."
        )
    
    @staticmethod
    @abstractmethod
    def save_parallel_state_dict(state_dict: dict, path: str,
                                 config: CollieConfig,
                                 process_exclusion: bool = False, **kwargs):
        """
        将模型权重保存到 ``path`` 路径。保存的格式同 ``huggingface`` 格式。

        :param state_dict: 模型权重
        :param path:
        :param config:
        :param process_exclusion: 是否每个 rank 各自独立、互斥地保存模型权重。在模
            型规模较大时，该参数可以帮助节省内存。
        """
        raise NotImplementedError(
            "Every model should implement `save_parallel_state_dict` "
            "to properly save a state dict for the cuurent rank."
        )
    
    @classmethod
    def _get_model_cls(cls, config: Union[CollieConfig, str]):
        model_cls = cls
        if isinstance(config, str):
            config = load_config(config)
        if cls.__name__ == "CollieModelForCausalLM":
            mod = importlib.import_module(
                ".model", f"collie.models.{config.model_type}")
            classes = inspect.getmembers(mod, inspect.isclass)
            for name, _cls in classes:
                if not issubclass(_cls, CollieModelForCausalLM):
                    continue
                if name.lower().startswith(config.model_type):
                    model_cls = _cls
                    break
            if model_cls.__name__ == cls.__name__:
                raise ValueError(
                    f"Unexpected model type `{config.model_type}`"
                )
        else:
            if not cls.__name__.lower().startswith(config.model_type):
                logger.rank_zero_warning(
                    f"The pretrained model's type {config.model_type} does "
                    f"not match the current model {cls.__name__}."
                )
        return model_cls