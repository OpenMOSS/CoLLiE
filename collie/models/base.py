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
from transformers.generation.utils import GenerationMixin
from transformers.generation.utils import GenerationConfig
from collie.module import PipelineModel, GPTLMLoss
from collie.config import CollieConfig, load_config
from collie.log import logger
from collie.utils import setup_distribution, Zero3_Init, zero3_load_state_dict, is_zero3_enabled

class BaseModel(nn.Module, GenerationMixin):
    """
    Base model of CoLLiE.

    Every new model should inherit this class.
    """
    main_input_name = "input_ids"
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda")
        self.generation_config = GenerationConfig()
            
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
        res = super().generate(*args, **kwargs)
        self.clean()
        return res

    @classmethod
    def from_config(cls, config: Union[CollieConfig, str], **kwargs):
        """
        Load arguments from config.
        """
        if isinstance(config, str):
            config = CollieConfig.from_pretrained(config, **kwargs)
        setup_distribution(config)
        model_cls = cls._get_model_cls(config)
        if config.pp_size == 1:
            with Zero3_Init(config):
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
        raise NotImplementedError(
            "`clean` should be implemented to clear caches for generation."
        )

    @classmethod
    def from_pretrained(cls, model_path_or_name: str, config: Optional[Union[CollieConfig, str]] = None, **kwargs):
        """
        :param model_path_or_name: str
        :param config: str, CollieConfig or None. If None, we will load
            arguments from `model_path_or_name`.
        :param kwargs:
            - process_exclusion: Whether to load checkpoints one by one to 
              save memory.
            parameters to be set at CollieConfig.
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
        state_dict = cls.load_parallel_state_dict(
            path=model_path_or_name, config=config,
            process_exclusion=process_exclusion,
        )
        if is_zero3_enabled(config):
            zero3_load_state_dict(model, state_dict)
        else:
            model.load_state_dict(state_dict)
        return model

    @classmethod
    def pipline_layers(cls, config: Union[CollieConfig, str]):
        """
        Get layers of pipeline.

        :return: list
        """
        raise NotImplementedError(
            "To use pipeline parallelism, you need to implement "
            "`pipeline_layers` for your model."
        )

    @staticmethod
    @abstractmethod
    def load_parallel_state_dict(path: str, config: Union[CollieConfig, str],
                                 process_exclusion: bool = False):
        """
        Load state_dict from ``path``.

        The format of pretrained model should be the same as that of
        `huggingface`.

        :param path:
        :param config:
        :param process_exclusion: Whether to load checkpoints one by one to 
            save memory.
        :return: state_dict. Note that the state_dict should be processed
            properly to match the current rank.
        """
        raise NotImplementedError(
            "Every model should implement `load_parallel_state_dict` "
            "to properly load a state dict for the cuurent rank."
        )
    
    @staticmethod
    @abstractmethod
    def save_parallel_state_dict(state_dict: dict, path: str,
                                 config: CollieConfig,
                                 process_exclusion: bool = False):
        """
        Save ``state_dict`` to ``path``.

        The format of saved state dict should be the same as that of
        `huggingface`.
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
        if cls.__name__ == "BaseModel":
            mod = importlib.import_module(
                ".model", f"collie.models.{config.model_type}")
            classes = inspect.getmembers(mod, inspect.isclass)
            for name, _cls in classes:
                if not issubclass(_cls, BaseModel):
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