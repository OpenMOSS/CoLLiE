from dataclasses import dataclass, field
from typing import Union

from deepspeed.runtime.config import DeepSpeedConfig
import sys
sys.path.append("../../")
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

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        json_config = load_config(path)
        unexpected = set()
        init_dict = {}
        for key, value in json_config.items():
            if key in cls.__dict__:
                init_dict[key] = value
            else:
                unexpected.add(key)
        for key, value in kwargs.items():
            if key in cls.__dict__:
                init_dict[key] = value
            else:
                unexpected.add(key)

        if len(unexpected) != 0:
            logger.rank_zero_warning(
                f"The following arguments from `from_pretrained` are not "
                f"defined in {cls.__class__.__name__} and will be ignored:\n"
                f"{list(unexpected)}"
            )

        return cls(**init_dict)
    
    def __post_init__(self):
        if isinstance(self.ds_config, str):
            self.ds_config = load_config(self.ds_config)
        assert isinstance(self.ds_config, dict), self.ds_config

    
def load_config(path: str):
    content = {}
    if path.lower().endswith("yaml"):
        import yaml
        content = yaml.load(open(path, "r"), Loader=yaml.SafeLoader)
    elif path.lower().endswith("json"):
        import json
        content = json.load(open(path, "r"))
    return content
