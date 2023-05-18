import sys
import torch
sys.path.append("/mnt/lustre/zhangshuo/projects/collie/")
from collie.models.llama.model import LlamaForCasualLM
from collie.trainer.trainer import Trainer
from collie.metrics.decode import DecodeMetric
from collie.config import CollieConfig
from collie.utils import zero3_load_state_dict

from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", 
                                           padding_side="left", 
                                           add_eos_token=True)
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
tokenizer.pad_token_id = 0
config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf")
config.dp_size = 8
config.train_epochs = 10
config.train_micro_batch_size = 1
config.eval_batch_size = 1
config.eval_per_n_steps = 20
config.ds_config = {
    "fp16": {"enabled": True},
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 2e-5
        }
    },
    "zero_optimization": {
        "stage": 3,
    }
}
model = LlamaForCasualLM.from_config(config)
state_dict = LlamaForCasualLM.load_parallel_state_dict(
    path="hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf",
    config=config,
    protocol="petrel",
    format="hf"
)
zero3_load_state_dict(model, state_dict)
train_sample = tokenizer("Collie is a python package for finetuning large language models.", return_tensors="pt").input_ids.squeeze(0)
eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)[:-1,]
train_dataset = [(train_sample, train_sample) for _ in range(1000)]
eval_dataset = [(eval_sample, eval_sample)]
trainer = Trainer(
    model = model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    eval_config=GenerationConfig(max_new_tokens=32, 
                                 eos_token_id=2, 
                                 pad_token_id=0, 
                                 bos_token_id=1),
    metrics=[DecodeMetric(tokenizer=tokenizer)],
    config=config
)
trainer.train()