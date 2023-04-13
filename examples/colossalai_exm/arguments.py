from dataclasses import dataclass, field
from collie.trainer.colossalai_trainer import TrainerArgs
from collie.models.llama_colossalai import ModelArgs

@dataclass
class DataArguments:
    data_dir: str = field(default='data')
    dataset_name: str = field(default='openbookqa')
    refresh: bool = field(
        default=False, metadata={"help": "Whether to refresh the data."}
    )

    data_tag: str = field(default='src')
    prompt_type: str = field(
        default='natural',
        metadata={"help": "The type of prompt, including [natural, brown]."}
    )
    train_on_inputs: bool = field(
        default=False, metadata={"help": "Whether to train on input."}
    )
    max_length: int = field(default=1024)
    few_shot_size: int = field(default=-1)

@dataclass
class ModelArguments(ModelArgs):
    model_path: str = field(default='/remote-home/share/llama/7B')
    tokenizer_path: str = field(
        default="/remote-home/share/llama/tokenizer.model"
    )
    source: str = field(
        default="raw",
        metadata={"help": "If the checkpoint from huggingface or metaAI. including ['raw', 'hf']."}
        )
    protocol: str = field(
        default="file",
        metadata={"help": "protocol to load state dict to pipieline model. including ['file', 's3']."})