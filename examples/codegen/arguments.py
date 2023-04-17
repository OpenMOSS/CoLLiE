from dataclasses import dataclass, field
from typing import Optional
from collie.arguments import CollieArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="llama-7B")

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
    in_context_learning: bool = field(default=False, metadata={"help": "Whether to use in-context learning."})


@dataclass
class MyCollieArguments(CollieArguments):
    length_normalization: bool = field(default=True, metadata={"help": "Whether to normalize the loss by the length of the input."})
    unconditional_normalization: bool = field(default=False, metadata={"help": "Whether to normalize the loss by the length of the input."})
