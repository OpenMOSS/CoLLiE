from dataclasses import dataclass, field

@dataclass
class TrainerArgs:
    seed: int = field(
        default=42,
        metadata={
            "help": "Random seed that will be set at the beginning of training."
        }
    )
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
    