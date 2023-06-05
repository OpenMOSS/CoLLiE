import sys
sys.path.append("../..")
from collie.models.llama.model import LlamaForCausalLM
from collie.trainer.trainer import Trainer
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
                                           add_eos_token=True)
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf")
config.tp_size = 8
config.dp_size = 1
config.pp_size = 1
config.train_epochs = 1000
config.train_micro_batch_size = 1
config.gradient_accumulation_steps = 64
config.eval_batch_size = 1
config.eval_per_n_steps = 20
config.ds_config = {
    "fp16": {"enabled": True}
}

model = LlamaForCausalLM.from_config(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# state_dict = LlamaForCausalLM.load_parallel_state_dict(
#     path="hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf",
#     config=config,
#     protocol="petrel",
#     format="hf"
# )
# model.load_state_dict(state_dict)
# train_sample = tokenizer("Collie is a python package for finetuning large language models.", return_tensors="pt").input_ids.squeeze(0)
eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)[:-1,]
train_sample = torch.randint(0, 100, (1024,))
train_dataset = [(train_sample, train_sample) for _ in range(128000)]
eval_dataset = [(eval_sample, eval_sample)]
trainer = Trainer(
    model = model,
    optimizer=optimizer,
    config=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    eval_config=GenerationConfig(max_new_tokens=128, 
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
        "decode": DecodeMetric(tokenizer=tokenizer, save_path="resutl.txt", save_to_file=True)},
)
trainer.train()