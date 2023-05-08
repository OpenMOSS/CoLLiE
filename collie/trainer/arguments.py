from dataclasses import dataclass, field

from typing import Union

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
    
def load_config(path: str):
    content = {}
    if path.lower().endswith("yaml"):
        import yaml
        content = yaml.load(open(path, "r"), Loader=yaml.SafeLoader)
    elif path.lower().endswith("json"):
        import json
        content = json.load(open(path, "r"))
    return content
        