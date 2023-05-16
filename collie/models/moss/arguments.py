from dataclasses import dataclass, field

from collie.trainer.arguments import Arguments

@dataclass
class MossArguments(Arguments):
    vocab_size: int = 51200
    n_embd: int = 1024
    n_ctx: int = 2048
    n_head: int = 16
    n_layer: int = 20
    n_positions: int = 2048
    layer_norm_epsilon: float = 1e-5
    rotary_dim: int = 32
    n_inner: int = None

    attn_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    resid_pdrop: float = 0.0

    model_type: str = "moss"

    checkpointing: bool = True
