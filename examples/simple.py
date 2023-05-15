# The simplist example of using collie, enjoy 3D parallism!
# Command CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=8 simple.py
import sys
sys.path.append("/mnt/lustre/zhangshuo/projects/collie-main/collie/Megatron-LM/")
sys.path.append("/mnt/lustre/zhangshuo/projects/collie/")
from transformers import AutoTokenizer
from collie.trainer.trainer import Trainer
from collie.models.llama.model import LlamaModel
from collie.models.llama.arguments import LlamaArguments
from collie.module import CollieCasualLM
from torch.utils.data import Dataset
import torch
args = LlamaArguments(
    use_flash=True,checkpointing=False,seed=42,pp_size=2,tp_size=2,dp_size=2,dropout=0,
    ds_config={
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": True},
        # "zero_optimization": {"stage": 1, "offload_optimizer": {"device": "cpu", "pin_memory": False}},
        # "optimizer": {"type": "Adam","params": {"lr": 2e-6,"betas": [0.8,0.999],"eps": 1e-8,"weight_decay": 3e-7
        #                                         }
        # },
    }
)
class DummyDataset(Dataset):
    def __init__(self):
        pass
        
    def __len__(self):
        return 100
    
    def __getitem__(self, idx):
        return "It's well known that Collie is a framework for "
    
torch.set_default_tensor_type(torch.HalfTensor)
dataset = DummyDataset()
model = LlamaModel(args)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
state_dict = LlamaModel.load_parallel_state_dict(
    path="hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf/",
    # path="/mnt/lustre/zhangshuo/model/epoch-1-step-2000-raw",
    protocol="petrel",
    format="hf",
    process_exclusion=False,
    args=args)
LlamaModel.save_parallel_state_dict(
    state_dict,
    path="/mnt/lustre/zhangshuo/model/test",
    args=args
)
# model.load_state_dict(state_dict)
# trainer = Trainer(model, train_dataset=dataset, optimizer=optimizer, args=args)
# generation_model = CollieCasualLM(trainer.engine)
# trainer.engine.module.eval()
# generation_model.generate(input_ids=torch.Tensor([[1, 6324, 29892, 278, 17251, 338, 2675, 304]]).long().cuda())
# trainer.train()