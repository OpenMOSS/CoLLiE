from dataclasses import dataclass, field
from typing import Optional
from tunelite.arguments import TuneLiteArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="llama-7B")
    cache_dir: Optional[str] = field(default='../llama/checkpoint')
    # llama_dir: Optional[str] = field(default='/remote-home/klv/exps/MossOn3090/llama')


@dataclass
class DataArguments:
    data_dir: str = field(default='data')
    dataset_name: str = field(default='openbookqa')
    refresh: bool = field(default=False, metadata={"help": "Whether to refresh the data."})

    data_tag: str = field(default='src')
    prompt_type: str = field(default='natural', metadata={"help": "The type of prompt, including [natural, brown]."})
    train_on_inputs: bool = field(default=False, metadata={"help": "Whether to train on input."})
    max_length: int = field(default=1024)
    few_shot_size: int = field(default=-1)


@dataclass
class MyTuneLiteArguments(TuneLiteArguments):
    pass
