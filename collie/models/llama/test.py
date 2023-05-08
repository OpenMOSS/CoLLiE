import torch
import torch.nn as nn
from torch.utils.data import Dataset

import deepspeed
from deepspeed.pipe import LayerSpec
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology

class Layer1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1000000000)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
class Layer2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1000000000, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class DummyDataset(Dataset):
    def __init__(self):
        pass
        
    def __len__(self):
        return 100
    
    def __getitem__(self, idx):
        return torch.tensor([idx]).float(), torch.tensor([idx]).float()
    
deepspeed.init_distributed(dist_backend='nccl', init_method="env://")
model = PipelineModule(
    layers=[LayerSpec(Layer1), LayerSpec(Layer2)],
    num_stages=2,
    topology=PipeModelDataParallelTopology(num_pp=2, num_dp=1, num_mp=1),
    loss_fn=lambda x, y: (print(x), x.sum())[1]
)
dataset = DummyDataset()
engine, optimizer, training_dataloader, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=dataset,
        config={
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                "lr": 0.001,
                "betas": [
                    0.8,
                    0.999
                ],
                "eps": 1e-8,
                "weight_decay": 3e-7
                }
            },
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu"
                },
                "contiguous_gradients": True,
                "overlap_comm": True,
                "sub_group_size": 1e12,
                "reduce_bucket_size": "auto"
            }
        }
    )
print(engine.train_batch())