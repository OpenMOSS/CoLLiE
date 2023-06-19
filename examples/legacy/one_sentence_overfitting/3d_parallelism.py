import sys
sys.path.append("../..")
from collie.models.llama.model import LlamaForCausalLM
from collie.controller.trainer import Trainer
from collie.metrics.decode import DecodeMetric
from collie.config import CollieConfig
from collie.utils import GradioProvider, setup_distribution
from collie import TGSMonitor, MemoryMonitor, LossMonitor, EvalMonitor
from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig
from torch.utils.data import Dataset
import torch
import os

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", 
                                           padding_side="left",
                                           add_eos_token=False)
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf")
config.tp_size = 1
config.dp_size = 2
config.pp_size = 1
config.train_epochs = 1000
config.train_micro_batch_size = 2
config.gradient_accumulation_steps = 1
# config.eval_batch_size = 1
# config.eval_per_n_steps = 20
config.ds_config = {
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 3}
}

model = LlamaForCausalLM.from_pretrained("/mnt/petrelfs/zhangshuo/model/llama-7b-hf", config=config)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# train_sample = tokenizer("Collie is a python package for finetuning large language models.</s>", return_tensors="pt").input_ids.squeeze(0)
# eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)
# train_dataset = [(train_sample, train_sample) for _ in range(128000)]
# eval_dataset = [(eval_sample, eval_sample)]
sample1 = torch.ones((100,), dtype=torch.long)
sample2 = torch.ones((200,), dtype=torch.long)
dataset = [(sample1, sample1), (sample2, sample2)]
train_dataset = dataset * 100
eval_dataset = dataset
trainer = Trainer(
    model = model,
    optimizer=optimizer,
    config=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    generation_config=GenerationConfig(max_new_tokens=128, 
                                 eos_token_id=2, 
                                 pad_token_id=0, 
                                 bos_token_id=1,
                                 use_cache=False),
    monitors=[
        TGSMonitor(config),
        MemoryMonitor(config),
        LossMonitor(config),
        EvalMonitor(config)
    ],
    metrics={
        "decode": DecodeMetric(tokenizer=tokenizer)},
)
trainer.train()