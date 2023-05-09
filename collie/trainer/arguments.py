import os
import importlib
import inspect
from dataclasses import dataclass, field
from typing import Union

from collie.log import logger

@dataclass
class Arguments:
    """Arguments for Trainer.
    """
    use_flash: bool = field(
        default=True,
        metadata={
            "help": "Whether to use FlashAttention."
        }
    )
    checkpointing: bool = field(
        default=True,
        metadata={
            "help": "Whether to use activation checkpointing."
        }
    )
    seed: int = field(
        default=42,
        metadata={
            "help": "Random seed that will be set at the beginning of training."
        }
    )
    pp_size: int = field(
        default=1,
        metadata={
            "help": "Pipeline parallelism degree."
        }
    )
    tp_size: int = field(
        default=1,
        metadata={
            "help": "Tensor parallelism degree."
        }
    )
    dp_size: int = field(
        default=1,
        metadata={
            "help": "Data parallelism degree."
        }
    )
    pp_partition_method: str = field(
        default='parameters',
        metadata={
            "help": "Partition method for pipeline parallelism. Default is 'parameters'."
        }
    )
    train_epochs: int = field(
        default=100,
        metadata={
            "help": "Number of training epochs."
        }
    )
    ds_config: Union[str, dict] = field(
        default="",
        metadata={
            "help": "DeepSpeed configuration file."
        }
    )
    model_type: str = field(
        default="",
        metadata={
            "help": "Type of model. Such as 'moss', 'llama'."
        }
    )

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """
        Load pretrained model arguments.

        :param path:
        :param kwargs:
            - suffix: The suffix of config file. Used only when ``path`` is a
              directory. Choices: ['json', 'yaml']. Default: 'json'
            The remained kwargs is used to adjust arguments.
        """
        suffix = kwargs.pop("suffix", "json")
        if os.path.isdir(path):
            path = os.path.join(path, f"config.{suffix}")
        json_config = load_config(path)
        arg_cls = cls._get_cls(json_config)
        argument = arg_cls()
        json_config.update(kwargs)
        argument.update(**json_config)

        return argument
    
    def update(self, **kwargs):
        unexpected = set()
        for key, value in kwargs.items():
            if key in dir(self):
                setattr(self, key, value)
            else:
                unexpected.add(key)
        if len(unexpected) != 0:
            logger.warning(
                f"The following arguments from `from_pretrained` are not "
                f"defined in {self.__class__.__name__} and will be ignored:\n"
                f"{list(unexpected)}"
            )

    @classmethod
    def _get_cls(cls, json_config):
        model_type = json_config.get("model_type", None)
        # for hf
        if model_type is None:
            raise ValueError(
                "'model_type' must be set in your config file to figure out "
                "the type of pretrained model."
            )
        if cls.model_type != "" and cls.model_type != model_type:
            logger.warning(
                f"The model type of pretrained config `{model_type}` does not "
                f"match the current model's type `{cls.model_type}`, which "
                f"may cause some unexpected behaviours."
            )
        if cls.model_type != "":
            return cls

        mod = importlib.import_module(
            ".arguments", package=f"collie.models.{model_type}"
        )
        classes = inspect.getmembers(mod, inspect.isclass)
        for name, arg_cls in classes:
            if arg_cls.model_type == model_type:
                return arg_cls

        raise NotImplementedError(f"Unexpected Argument type `{model_type}`")

    def __post_init__(self):
        if isinstance(self.ds_config, str):
            self.ds_config = load_config(self.ds_config)
        assert isinstance(self.ds_config, dict), self.ds_config

    def __str__(self) -> str:        

        width = os.get_terminal_size().columns // 2 * 2
        title = self.__class__.__name__
        single_side = (width - len(title) - 2) // 2
        r = f"\n{'-' * single_side} {title} {'-' * single_side}\n"
        r += _repr_dict(self.__dict__, 0)
        r += f"\n{'-' * width}\n"

        return r

    
def load_config(path: str):
    content = {}
    if path.lower().endswith("yaml"):
        import yaml
        content = yaml.load(open(path, "r"), Loader=yaml.SafeLoader)
    elif path.lower().endswith("json"):
        import json
        content = json.load(open(path, "r"))
    return content

def _repr_dict(d, depth):
    if not isinstance(d, dict):
        return f" {d}"
    space = "    "
    r = ""
    for k, v in d.items():
        r += f"\n{space * depth}{k}:" + _repr_dict(v, depth+1)
    return r
