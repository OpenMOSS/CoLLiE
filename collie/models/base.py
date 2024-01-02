import os
import types
import torch
import inspect
import importlib
from abc import abstractmethod
from types import MethodType
from typing import Union, Optional, Sequence, List, Tuple
from huggingface_hub import snapshot_download

import deepspeed
from torch import nn
from torch import distributed as dist
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from peft import get_peft_model
from megatron.core import parallel_state
from megatron.core import tensor_parallel
from accelerate.big_modeling import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.generation.utils import GenerationMixin
from transformers.generation.utils import GenerationConfig
from transformers.utils import ContextManagers
from collie.module import PipelineModel, GPTLMLoss
from collie.config import CollieConfig, load_config
from collie.log import logger
from collie.utils import setup_distribution, is_zero3_enabled, env, \
    dict_as_params, get_keys_to_not_convert, concat_tensor

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
    base_model_prefix = ""
    _can_generate = False

    def __init__(self, config: CollieConfig) -> None:
        super().__init__()
        self.device = torch.device("cuda")
        self.dtype = config.model_config.torch_dtype
        self.generation_config = GenerationConfig()
        self.config = config.model_config
        # transformers 的 GenerateMixin 要求 config 必须为 PretrainedConfig，备份一下 collie 的配置
        self.collie_config = config

    def _get_hidden_states(self, layers: Sequence[nn.Module], attr_name: str="hidden_states"):
        all_hidden_states = []
        for layer in layers:
            assert hasattr(layer, attr_name), f"{layer} does not have {attr_name}"
            all_hidden_states.append(getattr(layer, attr_name))
        return tuple(all_hidden_states) if None not in all_hidden_states else None

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
        self.clean_cache()
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
        if 'train_micro_batch_size_per_gpu' in config.ds_config:
            assert config.ds_config['train_micro_batch_size_per_gpu'] == config.train_micro_batch_size, \
                "train_micro_batch_size_per_gpu in ds_config should be the same as train_micro_batch_size"
        config.ds_config['train_micro_batch_size_per_gpu'] = config.train_micro_batch_size
        setup_distribution(config)
        model_cls = cls._get_model_cls(config)
        model = None
        contexts = []
        if (config.low_cpu_mem_usage or \
            getattr(config.quantization_config, "load_in_4bit", False) or \
                config.quantization_config.load_in_8bit) and \
                    not is_zero3_enabled(config):
            contexts.append(init_empty_weights())
        if config.pp_size == 1:
            if is_zero3_enabled(config):
                contexts.append(deepspeed.zero.Init(
                    data_parallel_group=parallel_state.get_data_parallel_group(),
                    config_dict_or_path=config.ds_config  # config is necessary for bf16
                ))
            with ContextManagers(contexts):
                model = super().__new__(model_cls)
                model.__init__(config)
        else:
            logger.info("Pipeline initialization starts, the provided loss_fn is not currently being used; it will be utilized in trainer.")
            model = PipelineModel(
                config=config,
                layers=model_cls.pipeline_layers(config),
                base_seed=config.seed,
                partition_method=config.pp_partition_method,
                topology=PipeModelDataParallelTopology(
                    num_pp=config.pp_size,
                    num_dp=config.dp_size,
                    num_mp=config.tp_size
                ), loss_fn=GPTLMLoss()
            )
            setattr(model, "config", config)
            setattr(model, "collie_config", config)
            setattr(model, "save_parallel_state_dict", cls.save_parallel_state_dict)
            setattr(model, "load_parallel_state_dict", cls.load_parallel_state_dict)
            for method in cls.overwrite_pipeline_methods() + [cls.resize_token_embeddings, cls.prepare_inputs, cls.enable_input_require_grads]:
                object.__setattr__(model, method.__name__, types.MethodType(method, model))
        if kwargs.get("init_params", True):
            post_init_funcs = {}
            # 记录 post_init 函数
            for name, layer in model.named_modules():
                if hasattr(layer, 'post_init'):
                    post_init_funcs[name+".weight"] = layer.post_init
            context_zero3_init = []
            if is_zero3_enabled(config):
                context_zero3_init.append(deepspeed.zero.Init(
                    data_parallel_group=parallel_state.get_data_parallel_group(),
                    config_dict_or_path=config.ds_config
                ))
            with ContextManagers(context_zero3_init):
                for name, param in model.named_parameters():
                    contexts = []
                    if is_zero3_enabled(config):
                        contexts.append(deepspeed.zero.GatheredParameters(param, modifier_rank=0))
                    with ContextManagers(contexts):
                        if param.device == torch.device("meta"):
                            if name not in post_init_funcs:
                                value = config.init_method['init_func'](torch.empty((*param.data.size(),), dtype=config.model_config.torch_dtype,device="cpu"), **config.init_method['init_kwargs'])
                            else:
                                value = post_init_funcs[name]()
                            set_module_tensor_to_device(
                                module=model, tensor_name=name, device="cpu",
                                value=value,
                                dtype=config.model_config.torch_dtype,
                            )
                        else:
                            if name not in post_init_funcs:
                                param.data = config.init_method['init_func'](torch.zeros_like(param.data), **config.init_method['init_kwargs']).to(config.model_config.torch_dtype).to(param.device)
                            else:
                                param.data = post_init_funcs[name]()

        if kwargs.get("get_peft", True) and config.peft_config.peft_type is not None:
            model = get_peft_model(model, config.peft_config)
            model.print_trainable_parameters()
        # set model dtype to deepspeed dtype under zero3, because the model is initialized with deepspeed.zero.Init()
        # if is_zero3_enabled(config):
        #     if 'fp16' in config.ds_config and config.ds_config['fp16']['enabled']:
        #         ds_dtype = torch.float16
        #     elif 'bf16' in config.ds_config and config.ds_config['bf16']['enabled']:
        #         ds_dtype = torch.bfloat16
        #     else:
        #         ds_dtype = torch.float32
        #     model = model.to(config.model_config.torch_dtype)
        #     if config.model_config.torch_dtype != ds_dtype:
        #         logger.warning(f"model dtype {config.model_config.torch_dtype} is not equal to deepspeed dtype {ds_dtype},"
        #                        f" set model dtype to {ds_dtype}")
        #         config.model_config.torch_dtype = ds_dtype
        #         model.dtype = ds_dtype
        #         model.config = config.model_config
        #         model = model.to(ds_dtype)
        return model

    def __new__(cls, config: CollieConfig, **kwargs):
        return cls.from_config(config, **kwargs)

    @abstractmethod
    def clean_cache(self):
        """
        清理 ``use_cache`` 和 ``hidden_states`` 状态的函数。
        """
        raise NotImplementedError(
            "`clean_cache` should be implemented to clear caches for generation."
        )

    @abstractmethod
    def set_cache(self, use_cache):
        """
        设置 ``use_cache`` 的函数。

        :param use_cache: 是否在生成时使用缓存的 ``past_key_values``。如果为
            ``True`` 则会保存前向传播过程中 Attention 的 key 和 value 用于下一次
            生成。可以参考 :meth:`_set_use_cache` 的代码来设置。
        """
        raise NotImplementedError(
            "`set_cache` should be implemented to set caches for generation."
        )

    @classmethod
    def from_pretrained(cls, model_path_or_name: str, config: Union[CollieConfig, str], **kwargs):
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
        if not os.path.exists(model_path_or_name) and kwargs.get("protocol", "file") == "file":
            model_path_or_name = snapshot_download(model_path_or_name)
        if config is None:
            config = model_path_or_name
        if isinstance(config, str):
            # prevent duplicate ``from_pretrained`` in load_parallel
            config = CollieConfig.from_pretrained(config, **kwargs)
        if config.model_config.torch_dtype is None and \
            (getattr(config.quantization_config, "load_in_4bit", False) or \
                config.quantization_config.load_in_8bit):
                config.model_config.torch_dtype = torch.float16
        # Actually build the model and do not init the params
        model = cls.from_config(config, init_params=False, get_peft=False)
        model = model.to(config.model_config.torch_dtype)
        # quantization
        if getattr(config.quantization_config, "load_in_4bit", False) or \
            config.quantization_config.load_in_8bit:
            setattr(model, "is_loaded_in_4bit", getattr(config.quantization_config, "load_in_4bit", False))
            setattr(model, "is_loaded_in_8bit", config.quantization_config.load_in_8bit)
            from transformers.utils.bitsandbytes import replace_with_bnb_linear, \
                set_module_quantized_tensor_to_device
            llm_int8_skip_modules = config.quantization_config.llm_int8_skip_modules
            # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
            modules_to_not_convert = []
            if llm_int8_skip_modules is None:
                modules_to_not_convert = get_keys_to_not_convert(model)
            else:
                modules_to_not_convert = llm_int8_skip_modules

            if not isinstance(modules_to_not_convert, list):
                modules_to_not_convert = [modules_to_not_convert]

            modules_to_not_convert.extend(getattr(model, "_keep_in_fp32_modules", []))
            model = replace_with_bnb_linear(
                model, modules_to_not_convert=modules_to_not_convert, quantization_config=config.quantization_config
            )
        # load state dict
        state_dict = {}
        if not is_zero3_enabled(config) or env.dp_rank == 0:
            state_dict = cls.load_parallel_state_dict(
                path=model_path_or_name, config=config,
                process_exclusion=process_exclusion, **kwargs
            )
        if isinstance(model, PipelineModel):
            for key in list(state_dict.keys()):
                key_pp = model.name_to_pipeline(key)
                state_dict[key_pp] = state_dict.pop(key)

        context_zero3_init = []
        if is_zero3_enabled(config):
            context_zero3_init.append(deepspeed.zero.Init(
                data_parallel_group=parallel_state.get_data_parallel_group(),
                config_dict_or_path=config.ds_config
            ))

        post_init_funcs = {}
        # 记录 post_init 函数
        for name, layer in model.named_modules():
            if hasattr(layer, 'post_init') and name+".weight" not in state_dict.keys():
                post_init_funcs[name+".weight"] = layer.post_init
        # load checkpoint and dispatch
        with ContextManagers(context_zero3_init):
            for name, param in model.named_parameters():
                if name not in state_dict.keys() and (not is_zero3_enabled(config) or env.dp_rank == 0):
                    logger.warning(f"Missing key: {name}!")
                contexts = []
                if is_zero3_enabled(config):
                    contexts.append(deepspeed.zero.GatheredParameters(param, modifier_rank=0))
                with ContextManagers(contexts):
                    value = state_dict.get(name, config.init_method['init_func'](torch.zeros_like(param.data), **config.init_method['init_kwargs']).to(config.model_config.torch_dtype),)
                    
                    if name in post_init_funcs and name not in state_dict:
                        value = post_init_funcs[name]()
                        
                    if getattr(config.quantization_config, "load_in_4bit", False) or \
                        config.quantization_config.load_in_8bit:
                            set_module_quantized_tensor_to_device(
                                module=model,
                                tensor_name=name,
                                device="cpu",
                                value=value,
                            )
                    else:
                        if param.device == torch.device("meta"):
                            set_module_tensor_to_device(
                                module=model, tensor_name=name, device="cpu",
                                value=value,
                                dtype=config.model_config.torch_dtype
                            )
                        else:
                            if name in state_dict:
                                assert param.data.shape == state_dict[name].data.shape, f"The shape of the parameter corresponding to the `{name}` does not match: {param.data.shape} vs {state_dict[name].data.shape}"
                            param.data = value.to(param.device)
            
        
        if config.peft_config.peft_type is not None:
            model = get_peft_model(model, config.peft_config)
            model.print_trainable_parameters()
        return model


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

    def tie_weights(self):
        pass

    @abstractmethod
    def prepare_inputs(self,
                       input_ids: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None,
                       use_cache: bool = None,
                       past_key_values: Optional[Tuple[torch.Tensor]] = None,
                       **kwargs):
        """
        在生成过程中更新 ``input_ids``、``attention_mask`` 等输入参数的函数。
        """
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "use_cache": use_cache, "past_key_values": past_key_values
        }

    def prepare_inputs_for_generation(self,
                                      input_ids: torch.Tensor,
                                      attention_mask: Optional[torch.Tensor] = None,
                                      use_cache: bool = None,
                                      past_key_values: Optional[Tuple[torch.Tensor]] = None,
                                      **kwargs):
        """
        生成过程中更新输入和 cache 状态的函数，包含设置 use_cache 和 past_key_values
        以及更新输入两个过程。
        """
        self.set_cache(use_cache)
        if past_key_values is not None:
            if not isinstance(past_key_values, torch.Tensor) and None in past_key_values:
                past_key_values = None
        return self.prepare_inputs(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=use_cache, past_key_values=past_key_values
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
                        weights_list = [embedding.weight.clone().cuda() for _ in range(env.tp_size)]
                        dist.all_gather(weights_list, embedding.weight.cuda(), group=parallel_state.get_tensor_model_parallel_group())
                        embedding.weight = nn.Parameter(concat_tensor(weights_list, dim=0))
                    if env.dp_rank == 0:
                        new_embedding.weight.data[start_pos_new:end_pos_new, :] \
                            = embedding.weight.data[start_pos_old:end_pos_old, :]
                        if end_pos_new < (new_num_tokens // env.tp_size):
                            init_method = self.collie_config.init_method['init_func']
                            init_method(new_embedding.weight[end_pos_new:new_num_tokens // env.tp_size, :], **self.collie_config.init_method['init_kwargs'])
            else:
                if env.tp_size > 1 and isinstance(new_embedding, tensor_parallel.VocabParallelEmbedding):
                    weights_list = [embedding.weight.clone().cuda() for _ in range(env.tp_size)]
                    dist.all_gather(weights_list, embedding.weight.cuda(), group=parallel_state.get_tensor_model_parallel_group())
                    embedding.weight = nn.Parameter(concat_tensor(weights_list, dim=0))
                new_embedding.weight.data[start_pos_new:end_pos_new, :] \
                    = embedding.weight.data[start_pos_old:end_pos_old, :]
                if end_pos_new < (new_num_tokens // env.tp_size):
                    init_method = self.collie_config.init_method['init_func']
                    init_method(new_embedding.weight[end_pos_new:new_num_tokens // env.tp_size, :], **self.collie_config.init_method['init_kwargs'])
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
                with deepspeed.zero.GatheredParameters(
                        [new_lm_head.weight, lm_head.weight] + [new_lm_head.bias, lm_head.bias]
                        if lm_head.bias is not None else [new_lm_head.weight, lm_head.weight],
                        modifier_rank=0
                ):
                    if env.tp_size > 1 and isinstance(new_lm_head, tensor_parallel.ColumnParallelLinear):
                        weights_list = [lm_head.weight.clone().cuda() for _ in range(env.tp_size)]
                        dist.all_gather(weights_list, lm_head.weight.cuda(), group=parallel_state.get_tensor_model_parallel_group())
                        lm_head.weight = nn.Parameter(concat_tensor(weights_list, dim=0))
                        if lm_head.bias is not None:
                            bias_list = [lm_head.bias.clone().cuda() for _ in range(env.tp_size)]
                            dist.all_gather(bias_list, lm_head.bias.cuda(), group=parallel_state.get_tensor_model_parallel_group())
                            lm_head.bias = nn.Parameter(concat_tensor(bias_list, dim=0))
                    if env.dp_rank == 0:
                        new_lm_head.weight.data[start_pos_new:end_pos_new, :] \
                            = lm_head.weight.data[start_pos_old:end_pos_old, :]
                        if lm_head.bias is not None:
                            new_lm_head.bias.data[start_pos_new:end_pos_new] \
                                = lm_head.bias.data[start_pos_old:end_pos_old]
                        if end_pos_new < (new_num_tokens // env.tp_size):
                            init_method = self.collie_config.init_method['init_func']
                            init_method(new_lm_head.weight[end_pos_new:new_num_tokens // env.tp_size, :], **self.collie_config.init_method['init_kwargs'])
                            if lm_head.bias is not None:
                                init_method(new_lm_head.bias[end_pos_new:new_num_tokens // env.tp_size])
            else:
                if env.tp_size > 1 and isinstance(new_lm_head, tensor_parallel.ColumnParallelLinear):
                    weights_list = [lm_head.weight.clone().cuda() for _ in range(env.tp_size)]
                    dist.all_gather(weights_list, lm_head.weight.cuda(), group=parallel_state.get_tensor_model_parallel_group())
                    lm_head.weight = nn.Parameter(concat_tensor(weights_list, dim=0))
                    if lm_head.bias is not None:
                        bias_list = [lm_head.bias.clone() for _ in range(env.tp_size)]
                        dist.all_gather(bias_list, lm_head.bias.cuda(), group=parallel_state.get_tensor_model_parallel_group())
                        lm_head.bias = nn.Parameter(concat_tensor(bias_list, dim=0))
                new_lm_head.weight.data[start_pos_new:end_pos_new, :] \
                    = lm_head.weight.data[start_pos_old:end_pos_old, :]
                if lm_head.bias is not None:
                    new_lm_head.bias.data[start_pos_new:end_pos_new] \
                        = lm_head.bias.data[start_pos_old:end_pos_old]
                if end_pos_new < (new_num_tokens // env.tp_size):
                    init_method = self.collie_config.init_method['init_func']
                    init_method(new_lm_head.weight[end_pos_new:new_num_tokens // env.tp_size, :], **self.collie_config.init_method['init_kwargs'])
                    if lm_head.bias is not None:
                            init_method(new_lm_head.bias[end_pos_new:new_num_tokens // env.tp_size])
            self.set_lm_head(lm_head_name, new_lm_head)


    def get_input_embedding(self):
        for name, module in self.named_children():
            if isinstance(module, (nn.Embedding, tensor_parallel.VocabParallelEmbedding)):
                return name, module
        base_model = getattr(self, self.base_model_prefix, None)
        if base_model is None:
            return None, None
        for name, module in base_model.named_children():
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
        base_model = getattr(self, self.base_model_prefix, self)
        base_model.add_module(name, embedding)

    def set_lm_head(self, name, lm_head):
        self.add_module(name, lm_head)

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)
            elif isinstance(output, dict):
                output["hidden_states"].requires_grad_(True)
        input_embedding = self.get_input_embedding()[1]
        if input_embedding is not None:
            self._require_grads_hook = input_embedding.register_forward_hook(make_inputs_require_grads)