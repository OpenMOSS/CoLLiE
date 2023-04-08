from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments, Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-1.3b")
    cache_dir: Optional[str] = field(default='/remote-home/share/llms')
    llama_dir: Optional[str] = field(default='/remote-home/share/llama')


@dataclass
class DataArguments:
    data_dir: str = field(default='data')
    dataset_name: str = field(default='hellaswag')
    refresh: bool = field(default=False, metadata={"help": "Whether to refresh the data."})

    data_tag: str = field(default='src')
    train_on_inputs: bool = field(default=False, metadata={"help": "Whether to train on input."})
    max_length: int = field(default=1024)


@dataclass
class TuneLiteArguments(Seq2SeqTrainingArguments):
    tag: str = field(default="debug", metadata={"help": "Tag for the experiment."})

    max_new_tokens: int = field(default=100, metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."})
    temperature: float = field(default=1.0, metadata={"help": "The value used to modulate the next token probabilities."})
    top_p: float = field(default=1.0, metadata={"help": "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation."})
