# The simplist example of using collie, enjoy 3D parallism!
# Command CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=8 simple.py
import sys
sys.path.append("/mnt/lustre/zhangshuo/projects/collie-main/collie/Megatron-LM/")
sys.path.append("/mnt/lustre/zhangshuo/projects/collie/")
from collie.trainer.trainer import Trainer
from collie.models.llama.model import LlamaModel
from collie.models.llama.arguments import LlamaArguments
from collie.models.llama.utils import load_parallel_state_dict
from torch.utils.data import Dataset
import torch
args = LlamaArguments(
    use_flash=False,checkpointing=True,seed=42,pp_size=2,tp_size=2,dp_size=1,dropout=0,
    ds_config={
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 1,"offload_optimizer": {"device": "cpu", "pin_memory": False}},
        "optimizer": {"type": "Adam","params": {"lr": 2e-6,"betas": [0.8,0.999],"eps": 1e-8,"weight_decay": 3e-7}
        },
    }
)
class DummyDataset(Dataset):
    def __init__(self):
        pass
        
    def __len__(self):
        return 100
    
    def __getitem__(self, idx):
        # batch 格式: 数据和 label 的 tuple
        return torch.tensor([1619, 1024, 338, 29871]), \
    torch.tensor([1619, 1024, 338, 29871])
dataset = DummyDataset()
model = LlamaModel(args)
state_dict = LlamaModel.load_parallel_state_dict(
    # path="hdd:s3://opennlplab_hdd/models/llama/llama-7b-raw/",
    path="/mnt/lustre/zhangshuo/model/epoch-1-step-2000-raw",
    protocol="file",
    format="meta",
    process_exclusion=True,
    args=args)
model.load_state_dict(state_dict)

# trainer = Trainer(model, train_dataset=dataset, args=args)
# trainer.train()