import inspect
import importlib
from abc import abstractmethod

from torch import nn
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from collie.module import PipelineModel, GPTLMLoss
from collie.trainer.arguments import Arguments
from collie.log import logger
from ..utils import setup_distributation

class BaseModel(nn.Module):
    """
    Base model of CoLLiE.

    Every new model should inherit this class.
    """
    @classmethod
    def from_config(cls, args: Arguments, **kwargs):
        """
        Load arguments from config.
        """
        args.update(**kwargs)
        model_cls = cls._get_model_cls(args)
        if args.pp_size == 1:
            return model_cls(args)
        else:
            return PipelineModel(
                layers=model_cls.pipeline_layers(), base_seed=args.seed,
                partition_method=args.pp_partition_method,
                topology=PipeModelDataParallelTopology(
                    num_pp=args.pp_size,
                    num_dp=args.dp_size,
                    num_mp=args.tp_size
                ), loss_fn=GPTLMLoss()
            )

    @classmethod
    def from_pretrained(cls, path, args=None, **kwargs):
        """
        :param path: str
        :param args: str, Arguments or None.
        :param kwargs: parameters to be set at Arguments.
        """
        if args is None or isinstance(args, str):
            args = Arguments.from_pretrained(path, **kwargs)

        setup_distributation(args)
        model = cls.from_config(args)
        # model.load_state_dict(cls.load_parallel_state_dict(args, path))
        return model

    @classmethod
    def pipline_layers(cls, args):
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
    def load_parallel_state_dict(args, path):
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
    def save_parallel_state_dict(args, path):
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
    def _get_model_cls(cls, args):
        model_cls = cls
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