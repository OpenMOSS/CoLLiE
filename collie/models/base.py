import os
import types
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
from megatron.core import tensor_parallel
from transformers.generation.utils import GenerationMixin
from transformers.generation.utils import GenerationConfig
from collie.module import PipelineModel, GPTLMLoss
from collie.config import CollieConfig, load_config
from collie.log import logger
from collie.utils import setup_distribution, is_zero3_enabled, env, initization_mapping, dict_as_params

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
        # transformers 的 GenerateMixin 要求 config 必须为 PretrainedConfig，备份一下 collie 的配置
        self.collie_config = config
            
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
            setattr(pipeline_model, "collie_config", config)
            for method in cls.overwrite_pipeline_methods() + [cls.resize_token_embeddings, 
                                                              cls.save_parallel_state_dict, 
                                                              cls.load_parallel_state_dict]:
                object.__setattr__(pipeline_model, method.__name__, types.MethodType(method, pipeline_model))
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
    def overwrite_pipeline_methods() -> Sequence[callable]:
        return []

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
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> None:
        embedding_name, embedding = self.get_input_embedding()
        lm_head_name, lm_head = self.get_lm_head()
        if embedding is None and lm_head is None:
            return
        new_num_tokens_embedding = new_num_tokens
        new_num_tokens_lm_head = new_num_tokens
        if embedding is not None:
            if is_zero3_enabled(self.collie_config):
                with deepspeed.zero.GatheredParameters(embedding.weight, modifier_rank=None):
                    old_embedding_tokens, embedding_dim = embedding.weight.size()
                    old_lm_head_tokens = old_embedding_tokens
            else:
                old_embedding_tokens, embedding_dim = embedding.weight.size()
                old_lm_head_tokens = old_embedding_tokens
        if lm_head is not None:
            if is_zero3_enabled(self.collie_config):
                with deepspeed.zero.GatheredParameters(lm_head.weight, modifier_rank=None):
                    old_lm_head_tokens, lm_head_dim = lm_head.weight.size()
                    old_embedding_tokens = old_lm_head_tokens
            else:
                old_lm_head_tokens, lm_head_dim = lm_head.weight.size()
                old_embedding_tokens = old_lm_head_tokens
        if isinstance(embedding, tensor_parallel.VocabParallelEmbedding):
            assert new_num_tokens % env.tp_size == 0, "The new number of tokens must be divisible by the tensor parallel size."
            old_embedding_tokens = old_embedding_tokens * env.tp_size
        if isinstance(lm_head, tensor_parallel.ColumnParallelLinear):
            assert new_num_tokens % env.tp_size == 0, "The new number of tokens must be divisible by the tensor parallel size."
            old_lm_head_tokens = old_lm_head_tokens * env.tp_size
        if new_num_tokens is None or new_num_tokens == old_embedding_tokens:
            return
        if embedding is not None:
            with deepspeed.zero.Init(data_parallel_group=parallel_state.get_data_parallel_group(), enabled=is_zero3_enabled(self.collie_config)):
                if hasattr(embedding, "dict_as_params_input_keys") and \
                    hasattr(embedding, "dict_as_params_output_keys"):
                        new_embedding = dict_as_params(
                            input_keys=embedding.dict_as_params_input_keys,
                            output_keys=embedding.dict_as_params_output_keys,
                        )(embedding.__class__, new_num_tokens_embedding, embedding_dim).to(embedding.weight.device).to(embedding.weight.dtype)
                else:
                    new_embedding = embedding.__class__(
                        new_num_tokens_embedding, 
                        embedding_dim).to(embedding.weight.device).to(embedding.weight.dtype)
            tp_devide_rank = old_embedding_tokens // (new_num_tokens // env.tp_size)
            if env.tp_rank < tp_devide_rank:
                start_pos_old = (new_num_tokens // env.tp_size) * env.tp_rank
                end_pos_old = (new_num_tokens // env.tp_size) * (env.tp_rank + 1)
                start_pos_new = 0
                end_pos_new = new_num_tokens // env.tp_size
            elif env.tp_rank == tp_devide_rank:
                start_pos_old = (new_num_tokens // env.tp_size) * env.tp_rank
                end_pos_old = old_embedding_tokens
                start_pos_new = 0
                end_pos_new = old_embedding_tokens - (new_num_tokens // env.tp_size) * tp_devide_rank
            elif env.tp_rank > tp_devide_rank:
                start_pos_old = 0
                end_pos_old = 0
                start_pos_new = 0
                end_pos_new = 0
            if is_zero3_enabled(self.collie_config):
                with deepspeed.zero.GatheredParameters([new_embedding.weight, embedding.weight], modifier_rank=0):
                    if env.tp_size > 1 and isinstance(new_embedding, tensor_parallel.VocabParallelEmbedding):
                        weights_list = [embedding.weight.clone() for _ in range(env.tp_size)]
                        dist.all_gather(weights_list, embedding.weight, group=parallel_state.get_tensor_model_parallel_group())
                        embedding.weight = nn.Parameter(torch.concat(weights_list, dim=0))
                    if env.dp_rank == 0:
                        new_embedding.weight.data[start_pos_new:end_pos_new, :] \
                            = embedding.weight.data[start_pos_old:end_pos_old, :]
                        if end_pos_new < (new_num_tokens // env.tp_size):
                            initization_method = initization_mapping.get(self.collie_config.initization_method, torch.nn.init.normal_)
                            if self.collie_config.initization_method_params is not None:
                                initization_method = initization_method(new_embedding.weight[end_pos_new:new_num_tokens // env.tp_size, :], 
                                                                        **self.collie_config.initization_method_params)
                            else:
                                initization_method(new_embedding.weight[end_pos_new:new_num_tokens // env.tp_size, :])
            else:
                if env.tp_size > 1 and isinstance(new_embedding, tensor_parallel.VocabParallelEmbedding):
                    weights_list = [embedding.weight.clone() for _ in range(env.tp_size)]
                    dist.all_gather(weights_list, embedding.weight, group=parallel_state.get_tensor_model_parallel_group())
                    embedding.weight = nn.Parameter(torch.concat(weights_list, dim=0))
                new_embedding.weight.data[start_pos_new:end_pos_new, :] \
                    = embedding.weight.data[start_pos_old:end_pos_old, :]
                if end_pos_new < (new_num_tokens // env.tp_size):
                    initization_method = initization_mapping.get(self.collie_config.initization_method, torch.nn.init.normal_)
                    if self.collie_config.initization_method_params is not None:
                        initization_method = initization_method(new_embedding.weight[end_pos_new:new_num_tokens // env.tp_size, :], 
                                                                **self.collie_config.initization_method_params)
                    else:
                        initization_method(new_embedding.weight[end_pos_new:new_num_tokens // env.tp_size, :])
            self.set_input_embedding(embedding_name, new_embedding)
        if lm_head is not None:
            if embedding is not None and id(lm_head.weight) == id(embedding.weight):
                lm_head.weight = new_embedding.weight
                return
            with deepspeed.zero.Init(data_parallel_group=parallel_state.get_data_parallel_group(), enabled=is_zero3_enabled(self.collie_config)):
                if hasattr(lm_head, "dict_as_params_input_keys") and \
                    hasattr(lm_head, "dict_as_params_output_keys"):
                        new_lm_head = dict_as_params(
                            input_keys=lm_head.dict_as_params_input_keys,
                            output_keys=lm_head.dict_as_params_output_keys,
                        )(lm_head.__class__,
                          lm_head_dim, 
                          new_num_tokens_lm_head, 
                          bias=lm_head.bias is not None).to(lm_head.weight.device).to(lm_head.weight.dtype)
                else:
                    new_lm_head = lm_head.__class__(
                        embedding_dim, 
                        new_num_tokens_lm_head,
                        bias=lm_head.bias is not None).to(lm_head.weight.device).to(lm_head.weight.dtype)
            tp_devide_rank = old_lm_head_tokens // (new_num_tokens // env.tp_size)
            if env.tp_rank < tp_devide_rank:
                start_pos_old = (new_num_tokens // env.tp_size) * env.tp_rank
                end_pos_old = (new_num_tokens // env.tp_size) * (env.tp_rank + 1)
                start_pos_new = 0
                end_pos_new = new_num_tokens // env.tp_size
            elif env.tp_rank == tp_devide_rank:
                start_pos_old = (new_num_tokens // env.tp_size) * env.tp_rank
                end_pos_old = old_lm_head_tokens
                start_pos_new = 0
                end_pos_new = old_lm_head_tokens - (new_num_tokens // env.tp_size) * tp_devide_rank
            elif env.tp_rank > tp_devide_rank:
                start_pos_old = 0
                end_pos_old = 0
                start_pos_new = 0
                end_pos_new = 0
            if is_zero3_enabled(self.collie_config):
                with deepspeed.zero.GatheredParameters([new_lm_head.weight, lm_head.weight] + \
                    [new_lm_head.bias, lm_head.bias] if lm_head.bias is not None else [], modifier_rank=0):
                    if env.tp_size > 1 and isinstance(new_lm_head, tensor_parallel.ColumnParallelLinear):
                        weights_list = [lm_head.weight.clone() for _ in range(env.tp_size)]
                        dist.all_gather(weights_list, lm_head.weight, group=parallel_state.get_tensor_model_parallel_group())
                        lm_head.weight = nn.Parameter(torch.concat(weights_list, dim=0))
                        if lm_head.bias is not None:
                            bias_list = [lm_head.bias.clone() for _ in range(env.tp_size)]
                            dist.all_gather(bias_list, lm_head.bias, group=parallel_state.get_tensor_model_parallel_group())
                            lm_head.bias = nn.Parameter(torch.concat(bias_list, dim=0))
                    if env.dp_rank == 0:
                        new_lm_head.weight.data[start_pos_new:end_pos_new, :] \
                            = lm_head.weight.data[start_pos_old:end_pos_old, :]
                        if lm_head.bias is not None:
                            new_lm_head.bias.data[start_pos_new:end_pos_new] \
                                = lm_head.bias.data[start_pos_old:end_pos_old]
                        if end_pos_new < (new_num_tokens // env.tp_size):
                            initization_method = initization_mapping.get(self.collie_config.initization_method, torch.nn.init.normal_)
                            if self.collie_config.initization_method_params is not None:
                                initization_method = initization_method(new_lm_head.weight[end_pos_new:new_num_tokens // env.tp_size, :], 
                                                                        **self.collie_config.initization_method_params)
                                if lm_head.bias is not None:
                                    initization_method(new_lm_head.bias[end_pos_new:new_num_tokens // env.tp_size], 
                                                        **self.collie_config.initization_method_params)
                            else:
                                initization_method(new_lm_head.weight[end_pos_new:new_num_tokens // env.tp_size, :])
                                if lm_head.bias is not None:
                                    initization_method(new_lm_head.bias[end_pos_new:new_num_tokens // env.tp_size])
            else:
                if env.tp_size > 1 and isinstance(new_lm_head, tensor_parallel.ColumnParallelLinear):
                    weights_list = [lm_head.weight.clone() for _ in range(env.tp_size)]
                    dist.all_gather(weights_list, lm_head.weight, group=parallel_state.get_tensor_model_parallel_group())
                    lm_head.weight = nn.Parameter(torch.concat(weights_list, dim=0))
                    if lm_head.bias is not None:
                        bias_list = [lm_head.bias.clone() for _ in range(env.tp_size)]
                        dist.all_gather(bias_list, lm_head.bias, group=parallel_state.get_tensor_model_parallel_group())
                        lm_head.bias = nn.Parameter(torch.concat(bias_list, dim=0))
                new_lm_head.weight.data[start_pos_new:end_pos_new, :] \
                    = lm_head.weight.data[start_pos_old:end_pos_old, :]
                if lm_head.bias is not None:
                    new_lm_head.bias.data[start_pos_new:end_pos_new] \
                        = lm_head.bias.data[start_pos_old:end_pos_old]
                if end_pos_new < (new_num_tokens // env.tp_size):
                    initization_method = initization_mapping.get(self.collie_config.initization_method, torch.nn.init.normal_)
                    if self.collie_config.initization_method_params is not None:
                        initization_method = initization_method(new_lm_head.weight[end_pos_new:new_num_tokens // env.tp_size, :], 
                                                                **self.collie_config.initization_method_params)
                        if lm_head.bias is not None:
                            initization_method(new_lm_head.bias[end_pos_new:new_num_tokens // env.tp_size], 
                                                **self.collie_config.initization_method_params)
                    else:
                        initization_method(new_lm_head.weight[end_pos_new:new_num_tokens // env.tp_size, :])
                        if lm_head.bias is not None:
                            initization_method(new_lm_head.bias[end_pos_new:new_num_tokens // env.tp_size])
            self.set_lm_head(lm_head_name, new_lm_head)
            
                        
    def get_input_embedding(self):
        for name, module in self.named_children():
            if isinstance(module, (nn.Embedding, tensor_parallel.VocabParallelEmbedding)):
                return name, module
        return None, None
    
    def get_lm_head(self):
        lm_head = None
        lm_head_name = None
        for name, module in self.named_children():
            if isinstance(module, (tensor_parallel.ColumnParallelLinear, nn.Linear)):
                lm_head = module
                lm_head_name = name
        return lm_head_name, lm_head
    
    def set_input_embedding(self, name, embedding):
        self.add_module(name, embedding)
        
    def set_lm_head(self, name, lm_head):
        self.add_module(name, lm_head)