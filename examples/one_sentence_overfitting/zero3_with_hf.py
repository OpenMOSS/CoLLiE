import sys
import torch
sys.path.append("../..")
from collie.trainer.trainer import Trainer
from collie.metrics.decode import DecodeMetric
from collie.config import CollieConfig
from collie.utils import setup_distribution
from collie.utils.data_provider import GradioProvider

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("gpt2", 
                                           padding_side="left", 
                                           add_eos_token=False,
                                           add_bos_token=True)
config = CollieConfig.from_pretrained("gpt2")
config.dp_size = 2
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
setup_distribution(config)
model = AutoModelForCausalLM.from_pretrained("gpt2")
train_sample = tokenizer("Collie is a python package for finetuning large language models." + tokenizer.eos_token, return_tensors="pt").input_ids.squeeze(0)
eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)
train_dataset = [(train_sample, train_sample) for _ in range(1000)]
eval_dataset = [(eval_sample, eval_sample)]
trainer = Trainer(
    model = model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    generation_config=GenerationConfig(max_new_tokens=32, 
                                       eos_token_id=tokenizer.eos_token_id),
    metrics={"decode": DecodeMetric(tokenizer=tokenizer)},
    data_provider=GradioProvider(tokenizer, stream=True),
    config=config
)
trainer.train()