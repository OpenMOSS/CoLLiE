# The simplist example of using collie, enjoy 3D parallism!
# Command CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=8 alpaca.py
import sys
sys.path.append("/mnt/petrelfs/gutianle/Megatron-LM/")
sys.path.append("/mnt/petrelfs/gutianle/collie/")
from transformers import LlamaTokenizer
from collie.trainer.trainer import Trainer
from collie.models.llama.model import LlamaModel
from collie.models.llama.arguments import LlamaArguments
from collie.metrics.decode import DecodeMetric
from transformers.generation.utils import GenerationConfig
from torch.utils.data import Dataset
import torch
import json

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", 
                                           padding_side="left")
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2

args = LlamaArguments.from_pretrained("decapoda-research/llama-7b-hf")
args.pp_size = 2
args.tp_size = 2
args.train_epochs = 3
args.train_micro_batch_size = 64
args.eval_batch_size = 32
args.eval_per_n_steps = 4
args.ds_config = {
    "fp16": {"enabled": True},
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 2e-5
        }
    }
}


# Alpaca数据
class AlpacaDataset(Dataset):
    def __init__(self, data_path):
        self.alpaca = self.load_data(data_path)
        
    def load_data(self, data_path):
        with open(data_path, encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self):
        return len(self.alpaca)
    
    def __getitem__(self, idx):
        return self.alpaca[idx]
    
def train_collate_fn(batch, max_length=512):
    input_ids_list = []
    label_ids_list = []
    for sample in batch:
        sample_prompt = sample['prompt']
        sample_label = sample['label']
        sample_prompt_ids = [1] + tokenizer.encode(sample_prompt, add_special_tokens=False)
        sample_label_ids = tokenizer.encode(sample_label, add_special_tokens=False) + [2]
        # heuristic truncation
        if len(sample_prompt_ids) + len(sample_label_ids) >= max_length:
            if len(sample_label_ids) >= max_length:
                # truncate front 64 tokens
                sample_label_ids = sample_label_ids[:63] + [2]
            temp_length = (max_length - len(sample_label_ids)) // 2
            sample_prompt_ids = sample_prompt_ids[:temp_length] + sample_prompt_ids[-temp_length:]
        input_ids_list.append(sample_prompt_ids + sample_label_ids)
        label_ids_list.append([0] * len(sample_prompt_ids) + sample_label_ids)
    # pad to longest
    batch_size = len(input_ids_list)
    longest = max(len(input_ids) for input_ids in input_ids_list)
    input_ids_tensor = torch.full((batch_size, longest), 0).long()
    label_ids_tensor = torch.full((batch_size, longest), 0).long()
    for i in range(batch_size):
        input_ids = input_ids_list[i]
        label_ids = label_ids_list[i]
        assert len(input_ids) == len(label_ids)
        input_ids_tensor[i, -len(input_ids):] = torch.LongTensor(input_ids)
        label_ids_tensor[i, -len(label_ids):] = torch.LongTensor(label_ids)
    return input_ids_tensor, label_ids_tensor

def eval_collate_fn(batch, max_length=512):
    input_ids_list = []
    label_ids_list = []
    for sample in batch:
        sample_prompt = sample['prompt']
        sample_label = sample['label']
        sample_prompt_ids = [1] + tokenizer.encode(sample_prompt, add_special_tokens=False)
        sample_label_ids = tokenizer.encode(sample_label, add_special_tokens=False) + [2]
        # heuristic truncation
        if len(sample_prompt_ids) + len(sample_label_ids) >= max_length:
            if len(sample_label_ids) >= max_length:
                # truncate front 64 tokens
                sample_label_ids = sample_label_ids[:63] + [2]
            temp_length = (max_length - len(sample_label_ids)) // 2
            sample_prompt_ids = sample_prompt_ids[:temp_length] + sample_prompt_ids[-temp_length:]
        input_ids_list.append(sample_prompt_ids + sample_label_ids)
        label_ids_list.append([0] * len(sample_prompt_ids) + sample_label_ids)
    # pad to longest
    batch_size = len(input_ids_list)
    longest = max(len(input_ids) for input_ids in input_ids_list)
    input_ids_tensor = torch.full((batch_size, longest), 0).long()
    label_ids_tensor = torch.full((batch_size, longest), 0).long()
    for i in range(batch_size):
        input_ids = input_ids_list[i]
        label_ids = label_ids_list[i]
        assert len(input_ids) == len(label_ids)
        input_ids_tensor[i, -len(input_ids):] = torch.LongTensor(input_ids)
        label_ids_tensor[i, -len(label_ids):] = torch.LongTensor(label_ids)
    return input_ids_tensor, label_ids_tensor


dataset = AlpacaDataset('alpaca_data.json')
train_dataset = dataset[:-32]
eval_dataset = dataset[-32:]

model = LlamaModel(args)
state_dict = LlamaModel.load_parallel_state_dict(
    path="hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf/",
    protocol="petrel",
    format="hf",
    process_exclusion=False,
    args=args)
model.load_state_dict(state_dict)

trainer = Trainer(
    model = model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    train_dataset_collate_fn=train_collate_fn,
    eval_dataset_collate_fn=eval_collate_fn,
    eval_config=GenerationConfig(max_new_tokens=128, eos_token_id=2, pad_token_id=0, bos_token_id=1),
    metrics=[DecodeMetric(tokenizer=tokenizer)],
    args=args
)

torch.cuda.empty_cache()
trainer.train()