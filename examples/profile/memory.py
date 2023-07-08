import sys

sys.path.append("../../")
from collie.models.llama.model import LlamaForCausalLM
from collie.controller import Trainer, EvaluatorForGeneration
from collie.metrics.decode import DecodeMetric
from collie.config import CollieConfig
from collie import TGSMonitor, MemoryMonitor, LossMonitor, EvalMonitor, NetworkIOMonitor
from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig
import torch
import os
# os.environ['WANDB_MODE'] = 'disabled'

# tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b",
#                                            padding_side="left",
#                                            add_eos_token=False)
# tokenizer.bos_token_id = 1
# tokenizer.eos_token_id = 2
config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf")
config.tp_size = 1
config.dp_size = 1
config.pp_size = 8
config.train_epochs = 1000
config.train_micro_batch_size = 8
config.gradient_accumulation_steps = 1
config.use_flash = True
config.ds_config = {
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 1},
    "monitor_config": {
        "enabled": True,
        "wandb": {
            "enabled": True,
            "team": "collie_exp",
            "project": "memory_profile",
            "group": "llama-7b",
            "job_name": "pp_8*256_flashattn"
        }
    },
    # "data_types": {
    #     "grad_accum_dtype": "fp32"
    # }
}
torch.cuda.memory_reserved()
# model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", config=config)
model = LlamaForCausalLM.from_config(config=config)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# train_sample = tokenizer("Collie is a python package for finetuning large language models.", return_tensors="pt").input_ids.squeeze(0)
# eval_sample = tokenizer("Collie is", return_tensors="pt")
train_sample = torch.randint(3, 20000, [256])
train_dataset = [{"input_ids": train_sample, "labels": train_sample} for _ in range(50)]
monitors = [
    MemoryMonitor(config),
    NetworkIOMonitor(config)
]
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    config=config,
    train_dataset=train_dataset,
    monitors=monitors,
)
trainer.train()
