import sys
sys.path.append("/mnt/lustre/zhangshuo/projects/collie-main/collie/Megatron-LM/")
sys.path.append("/mnt/lustre/zhangshuo/projects/collie/")
from collie.models.llama.model import LlamaModel, LlamaArguments
from collie.trainer.trainer import Trainer
from collie.metrics.decode import DecodeMetric

from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig
from torch.utils.data import Dataset
import torch

tokenizer = LlamaTokenizer.from_pretrained("/mnt/lustre/zhangshuo/model/llama-7b-hf")

args = LlamaArguments.from_pretrained("/mnt/lustre/zhangshuo/model/llama-7b-hf")
args.pp_size = 2
args.ds_config = {
    "fp16": {"enabled": True}
}

class GenerationDataset(Dataset):
    def __init__(self, sentence: str = "Hi") -> None:
        super().__init__()
        self.sentence = sentence
        
    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        return self.sentence
    
def collate_fn(batch):
    input_ids = tokenizer.encode(batch, return_tensors="pt")
    return input_ids, input_ids

model = LlamaModel.from_pretrained("/mnt/lustre/zhangshuo/model/llama-7b-hf", args=args)
trainer = Trainer(model=model,
                  eval_dataset=GenerationDataset("It's a beautiful day to"),
                  eval_config=GenerationConfig(),
                  metrics=[DecodeMetric(tokenizer=tokenizer)],
                  eval_dataset_collate_fn=collate_fn,
                  args=args)
trainer.eval()