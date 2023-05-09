from dataclasses import dataclass, field

from collie.trainer.arguments import Arguments

@dataclass
class LlamaArguments(Arguments):
    vocab_size: int = field(
        default=32000,
        metadata={
            "help": "vocab size of the model"
        }
    )
    hidden_size: int = field(
        default=4096,
        metadata={
            "help": "hidden size of the model"
        }
    )
    num_hidden_layers: int = field(
        default=32,
        metadata={
            "help": "number of hidden layers in the model"
        }
    )
    num_attention_heads: int = field(
        default=32,
        metadata={
            "help": "number of attention heads in the model"
        }
    )
    intermediate_size: int = field(
        default=11008,
        metadata={
            "help": "intermediate size of the model"
        }
    )
    layer_norm_epsilon: float = field(
        default=1e-5,
        metadata={
            "help": "layer norm epsilon"
        }
    )
    dropout: float = field(
        default=0.1,
        metadata={
            "help": "Dropout probability."
        }
    )
    model_type: str = "llama"