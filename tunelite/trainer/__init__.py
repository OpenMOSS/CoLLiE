__all__ = [
    'InplaceZeroTrainer',
    'InplaceTensorTrainer',
]

try:
    from .inplace_zero_trainer import InplaceZeroTrainer
    from .inplace_tensor_trainer import InplaceTensorTrainer
except:
    InplaceZeroTrainer = None
    InplaceTensorTrainer = None