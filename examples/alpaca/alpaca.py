# The simplist example of using collie, enjoy 3D parallism!
# Command CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=8 simple.py
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

args = LlamaArguments(
    use_flash=True,checkpointing=False,seed=42,pp_size=2,tp_size=2,dp_size=2,dropout=0,
    ds_config={
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": True},
    }
)

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", 
                                           padding_side="left", 
                                           add_eos_token=True)
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
args = LlamaArguments.from_pretrained("decapoda-research/llama-7b-hf")
args.pp_size = 2
args.tp_size = 2
args.train_epochs = 10
args.train_micro_batch_size = 1
args.eval_batch_size = 1
args.eval_per_n_steps = 2
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
    
def collate_fn(batch, tokenizer):
    batch_inputs = []
    batch_labels = []
    for sample in batch:
        sample_prompt = sample['prompt']
        sample_label = sample['prompt'] + sample['label']
        batch_inputs.append(sample_prompt)
        batch_labels.append(sample_label)
    input_ids = tokenizer(batch_inputs, return_tensors='pt', padding=True)
    label_ids = tokenizer(batch_labels, return_tensors='pt', padding=True)
    return input_ids, label_ids

    
torch.set_default_tensor_type(torch.HalfTensor)
dataset = AlpacaDataset('alpaca_data.json')
train_dataset = dataset[:-32]
eval_dataset = dataset[-32:]

model = LlamaModel(args)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
state_dict = LlamaModel.load_parallel_state_dict(
    path="hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf/",
    protocol="petrel",
    format="hf",
    process_exclusion=False,
    args=args)
model.load_state_dict(state_dict)
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    eval_config=GenerationConfig(max_new_tokens=100),
    metrics=[DecodeMetric(tokenizer=tokenizer, gather_result=True, only_rank0_update=True)],
    eval_dataset_collate_fn=collate_fn,
    train_dataset_collate_fn=collate_fn,
    args=args
)
torch.cuda.empty_cache()
trainer.eval()