import os
import importlib
import inspect
from dataclasses import dataclass, field
from typing import Union

from collie.log import logger
from transformers import PretrainedConfig, AutoConfig

@dataclass
class CollieConfig:
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
    eval_per_n_steps: int = field(
        default=0,
        metadata={
            "help": "Evaluate every n steps."
        }
    )
    eval_per_n_epochs: int = field(
        default=0,
        metadata={
            "help": "Evaluate every n epochs."
        }
    )
    train_micro_batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size (one step) for training."
        }
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of gradient accumulation steps."
        }
    )
    eval_batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size for evaluation."
        }
    )
    ds_config: Union[str, dict] = field(
        default="",
        metadata={
            "help": "DeepSpeed configuration file."
        }
    )
    model_config: PretrainedConfig = field(
        default=None,
        metadata={
            "help": "Model configuration."
        }
    )

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs):
        """
        Load pretrained model arguments.

        :param path:
        :param kwargs:
            The remained kwargs is used to adjust arguments.
        """
        cfg = cls()
        for key in list(kwargs.keys()):
            if key in cls.__annotations__.keys():
                setattr(cfg, key, kwargs.pop(key))

        cfg.model_config = AutoConfig.from_pretrained(name_or_path, **kwargs)

        return cfg
    
    def save_pretrained(self, path):
        self.model_config.save_pretrained(path)

    def __getattr__(self, name):
        return getattr(self.model_config, name)

    def __post_init__(self):
        if isinstance(self.ds_config, str):
            self.ds_config = load_config(self.ds_config)
        assert isinstance(self.ds_config, dict), self.ds_config

    def __str__(self) -> str:        
        title = self.__class__.__name__
        r = f"{title}:\n"
        r += _repr_dict(self.__dict__, 0)
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
    if isinstance(d, PretrainedConfig):
        d = d.to_diff_dict()
    if not isinstance(d, dict):
        return f" {d}"
    space = "    "
    r = ""
    for k, v in d.items():
        r += f"\n{space * depth}{k}:" + _repr_dict(v, depth+1)
    return r
