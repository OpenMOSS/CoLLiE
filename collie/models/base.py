import os
import inspect
import importlib
from abc import abstractmethod
from typing import Union, Optional
from huggingface_hub import snapshot_download

from torch import nn
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from collie.module import PipelineModel, GPTLMLoss
from collie.trainer.arguments import Arguments, load_config
from collie.log import logger
from collie.utils import setup_distributation

class BaseModel(nn.Module):
    """
    Base model of CoLLiE.

    Every new model should inherit this class.
    """
    @classmethod
    def from_config(cls, args: Union[Arguments, str], **kwargs):
        """
        Load arguments from config.
        """
        if isinstance(args, str) and os.path.exists(args):
            args = load_config(args)
        args.update(**kwargs)
        setup_distributation(args)
        model_cls = cls._get_model_cls(args)
        if args.pp_size == 1:
            return model_cls(args)
        else:
            return PipelineModel(
                layers=model_cls.pipeline_layers(args), base_seed=args.seed,
                partition_method=args.pp_partition_method,
                topology=PipeModelDataParallelTopology(
                    num_pp=args.pp_size,
                    num_dp=args.dp_size,
                    num_mp=args.tp_size
                ), loss_fn=GPTLMLoss()
            )
            
    def __new__(cls, args: Arguments, **kwargs):
        return cls.from_config(args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_path_or_name: str, args:Optional[Union[Arguments, str]] = None, **kwargs):
        """
        :param path: str
        :param args: str, Arguments or None.
        :param kwargs: parameters to be set at Arguments.
        """
        if not os.path.exists(model_path_or_name):
            model_path_or_name = snapshot_download(model_path_or_name)
        if isinstance(args, str) and os.path.exists(args):
            args = load_config(args)
        if args is None or isinstance(args, str):
            args = Arguments.from_pretrained(model_path_or_name, **kwargs)
        model = cls.from_config(args)
        model.load_state_dict(cls.load_parallel_state_dict(args, model_path_or_name))
        return model

    @classmethod
    def pipline_layers(cls, args: Union[Arguments, str]):
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
    def load_parallel_state_dict(path: str, args: Union[Arguments, str]):
        """
        Load state_dict from ``path``.

        The format of pretrained model should be the same as that of
        `huggingface`.

        :return: state_dict. Note that the state_dict should be processed
            properly to match the current rank.
        """
        raise NotImplementedError(
            "Every model should implement `load_parallel_state_dict` "
            "to properly load a state dict for the cuurent rank."
        )
    
    @staticmethod
    @abstractmethod
    def save_parallel_state_dict(path: str, args: Union[Arguments, str]):
        """
        Save state_dict to ``path``.

        The format of saved state dict should be the same as that of
        `huggingface`.
        """
        raise NotImplementedError(
            "Every model should implement `save_parallel_state_dict` "
            "to properly save a state dict for the cuurent rank."
        )
    
    @classmethod
    def _get_model_cls(cls, args: Union[Arguments, str]):
        model_cls = cls
        if isinstance(args, str) and os.path.exists(args):
            args = load_config(args)
        if cls.__name__ == "BaseModel":
            mod = importlib.import_module(
                ".model", f"collie.models.{args.model_type}")
            classes = inspect.getmembers(mod, inspect.isclass)
            for name, _cls in classes:
                if not issubclass(_cls, BaseModel):
                    continue
                if name.lower().startswith(args.model_type):
                    model_cls = _cls
                    break
            if model_cls.__name__ == cls.__name__:
                raise ValueError(
                    f"Unexpected model type `{args.model_type}`"
                )
        else:
            if not cls.__name__.lower().startswith(args.model_type):
                logger.warning(
                    f"The pretrained model's type {args.model_type} does not "
                    f"match the current model {cls.__name__}."
                )
        return model_cls