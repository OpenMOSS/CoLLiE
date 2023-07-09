import sys
sys.path.append("../")
from typing import Dict

import argparse
import copy
import json
import pandas as pd
import torch
from torch import distributed as dst
from transformers import AutoTokenizer, GenerationConfig
from rich.progress import track

from collie.metrics import RougeMetric
from collie.controller import Trainer
from collie.utils import env
from collie import CollieConfig, ChatGLM2ForCausalLM, DashProvider, LossMonitor, TGSMonitor, MemoryMonitor, LRMonitor, CollieDatasetForTraining, CollieDatasetForGeneration, EvaluatorForGeneration


def load_data(path_dict):
    data_bundle = {}
    for name, path in path_dict.items():
        with open(path, encoding="utf8") as f:
            data_df = pd.DataFrame([json.loads(l) for l in f.readlines()], columns=['text', 'summary'])
        dataset = []
        if 'train' in name:
            for _, row in track(data_df.iterrows(), description=name, total=len(data_df), disable=env.rank != 0):
                dataset.append({ "input": f"Text: {' '.join(row['text'].split()[:500])} \nSummary: {row['summary']}","output": f"{row['summary']}"})
        else:
            for _, row in track(data_df.iterrows(), description=name, total=len(data_df), disable=env.rank != 0):
                dataset.append({ "text": f"Text: {' '.join(row['text'].split()[:500])} \nSummary: ","target": f"{row['summary']}"})
        data_bundle[name] = dataset
    return data_bundle

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", default="/mnt/petrelfs/hongjiawei/datasets/billsum/us_train_data_final_OFFICIAL.jsonl")
parser.add_argument("--dev_path", default="/mnt/petrelfs/hongjiawei/datasets/billsum/ca_test_data_final_OFFICIAL.jsonl")
parser.add_argument("--test_path", default="/mnt/petrelfs/hongjiawei/datasets/billsum/us_test_data_final_OFFICIAL.jsonl")
parser.add_argument("--max_tokens", default=1500)
parser.add_argument("--model_path", default="THUDM/chatglm2-6b")
args = parser.parse_args()

config = CollieConfig.from_pretrained(args.model_path, trust_remote_code=True)
config.pp_size = 4
config.tp_size = 2
config.train_micro_batch_size = 1
config.eval_batch_size = 1
config.gradient_accumulation_steps = 128
config.eval_per_n_epochs = 1
config.train_epochs = 10
config.ds_config = {
    "fp16": {
        "enabled": True
    },
    "monitor_config": {
        "enabled": True,
        "wandb": {
            "enabled": True,
            "team": "00index",
            "project": "collie",
            "group": "summary"
        }
    }
}
config.seed = 1024

path_dict = {"train": args.train_path,
             "dev": args.dev_path,
             "test": args.test_path}

tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_eos_token=False, trust_remote_code=True)

data_bundle = load_data(path_dict)

traine_dataset = CollieDatasetForTraining(data_bundle['train'][:10],
                                          tokenizer=tokenizer)
eval_dataset = CollieDatasetForGeneration(data_bundle['dev'],
                                               tokenizer=tokenizer)

# Prepare model
model = ChatGLM2ForCausalLM.from_pretrained(args.model_path, config=config)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer, step_size=1, gamma=0.9
)
evaluator_rouge = EvaluatorForGeneration(
    model=model,
    config=config,
    dataset=eval_dataset,
    metrics={'rouge': RougeMetric()},
    generation_config=GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=100,
    )
)
# Prepare Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    lr_scheduler=lr_scheduler,
    config=config,
    optimizer=optimizer,
    train_dataset=traine_dataset,
    monitors=[
        LossMonitor(config),
        TGSMonitor(config),
        MemoryMonitor(config),
        LRMonitor(config)
    ],
    data_provider=DashProvider(tokenizer, port=12888, stream=True,
                                 generation_config=GenerationConfig(
                                     eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.pad_token_id,
                                     max_new_tokens=100,
                                 )),
    evaluators=[evaluator_rouge]
)
trainer.train()