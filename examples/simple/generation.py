import sys
sys.path.append("/mnt/lustre/zhangshuo/projects/collie/")
from collie.models.llama.model import LlamaForCasualLM
from collie.trainer.trainer import Trainer
from collie.config import CollieConfig
from collie.metrics.decode import DecodeMetric

from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig
from torch.utils.data import Dataset
import torch

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", 
                                           padding_side="left")
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2

config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf")
config.dp_size = 2
config.pp_size = 2
config.tp_size = 2
config.eval_batch_size = 4
config.ds_config = {
    "fp16": {"enabled": True}
}

class GenerationDataset(Dataset):
    def __init__(self, sentences = []) -> None:
        super().__init__()
        self.sentences = sentences
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        return self.sentences[index]
    
def collate_fn(batch):
    input_ids = tokenizer(batch, 
                          return_tensors="pt", 
                          padding=True)["input_ids"]
    return input_ids, input_ids
dataset = GenerationDataset([
    "It's a beautiful day to",
    "This is my favorite",
    "As we discussed yesterday",
    "I'm going to",
    "As well known to all",
    "A",
    "So this is the reason why",
    "We have to"
])
model = LlamaForCasualLM.from_pretrained("/mnt/lustre/zhangshuo/model/test/", config=config)

trainer = Trainer(model=model,
                  config=config,
                  train_dataset=dataset,
                  eval_dataset=dataset,
                  eval_config=GenerationConfig(max_new_tokens=100),
                  metrics=[DecodeMetric(tokenizer=tokenizer)],
                  eval_dataset_collate_fn=collate_fn,
                  train_dataset_collate_fn=collate_fn)
trainer.eval()