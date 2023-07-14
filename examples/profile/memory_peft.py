import sys

sys.path.append("../../")
import os
import time
import argparse

import torch
from transformers import LlamaTokenizer, AutoTokenizer
from transformers.generation.utils import GenerationConfig

from collie.models.llama.model import LlamaForCausalLM
from collie import Callback
from collie.utils import env
from collie.controller import Trainer, EvaluatorForGeneration
from collie.metrics.decode import DecodeMetric
from collie.config import CollieConfig
from collie.optim import Adan, Lomo, Lion, SophiaG
from collie.utils import env
from collie import TGSMonitor, MemoryMonitor, LossMonitor, EvalMonitor, NetworkIOMonitor
import wandb
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

os.environ['WANDB_MODE'] = 'disabled'
nvmlInit()

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model_name", type=str, default="huggyllama/llama-7b")
arg_parser.add_argument("--optim", type=str, default="adam")
arg_parser.add_argument("--tp_size", type=int, default=8)
arg_parser.add_argument("--pp_size", type=int, default=1)
args = arg_parser.parse_args()

model_name = args.model_name.split("/")[-1]
record_file_name = f"{model_name}_{args.optim}_tp{args.tp_size}_pp{args.pp_size}_1*2048_gas2_flashattn.txt"


class NVMemoryCallback(Callback):
    def on_train_batch_end(self, trainer, loss):
        handles = []
        for i in range(8):
            handles.append(nvmlDeviceGetHandleByIndex(i))
        meminfo = []
        for handle in handles:
            meminfo.append(nvmlDeviceGetMemoryInfo(handle))
        if env.local_rank == 0:
            with open(record_file_name, "a") as f:
                f.write(f"Step {trainer.global_batch_idx} "
                        f"Total: {sum([i.used for i in meminfo]) / 1024 ** 2} MB "
                        f"Every GPU: {[i.used / 1024 ** 2 for i in meminfo]} MB\n")
            print(f"Total: {sum([i.used for i in meminfo]) / 1024 ** 2} MB "
                  f"Every GPU: {[i.used / 1024 ** 2 for i in meminfo]} MB")

wandb_config = {
    "entity": "collie_exp",
    "project": "memory_profile",
    "group": args.model_name.split("/")[-1],
    "name": f"tp{args.tp_size}_pp{args.pp_size}_{args.optim}_1*2048_gas2_flashattn",
    # "config": vars(args),
}
if 'LOCAL_RANK' in os.environ:
    if int(os.environ['LOCAL_RANK']) == 0:
        wandb.init(
            **wandb_config
        )
elif int(os.environ["SLURM_LOCALID"]) == 0:
    wandb.init(
        **wandb_config
    )

config = CollieConfig.from_pretrained(args.model_name)
config.tp_size = args.tp_size
config.dp_size = 1
config.pp_size = args.pp_size
config.train_epochs = 3
config.train_micro_batch_size = 1
config.gradient_accumulation_steps = 2
config.use_flash = True
if args.optim == "lomo":
    config.gradient_accumulation_steps = 1
    config.ds_config = {
        "fp16": {"enabled": True},
        "monitor_config": {
            "enabled": True,
            "wandb": {
                "enabled": True,
                "team": "collie_exp",
                "project": "memory_profile",
                "group": args.model_name.split("/")[-1],
                "job_name": f"tp{args.tp_size}_{args.optim}_1*2048_gas1_flashattn"
            }
        },
        # "data_types": {
        #     "grad_accum_dtype": "fp32"
        # }
    }
else:
    config.ds_config = {
        "bf16": {"enabled": True},
        # "zero_optimization": {"stage": 1},
        "monitor_config": {
            "enabled": True,
            "wandb": {
                "enabled": True,
                "team": "collie_exp",
                "project": "memory_profile",
                "group": args.model_name.split("/")[-1],
                "job_name": f"tp{args.tp_size}_pp{args.pp_size}_{args.optim}_1*2048_gas{config.gradient_accumulation_steps}_flashattn"
            }
        },
        # "steps_per_print": 1,
        "zero_allow_untested_optimizer": True,
        # "data_types": {
        #     "grad_accum_dtype": "fp32"
        # }
    }

# model = LlamaForCausalLM.from_pretrained(args.model_name, config=config)
model = LlamaForCausalLM.from_config(config=config)
if args.optim == "adam":
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
elif args.optim == "adan":
    optimizer = Adan(model.parameters(), lr=2e-5)
elif args.optim == "lion":
    optimizer = Lion(model.parameters(), lr=2e-5)
elif args.optim == "lomo":
    optimizer = Lomo(model, lr=2e-5)
elif args.optim == "sophiag":
    optimizer = SophiaG(model.parameters(), lr=2e-5)
else:
    raise ValueError(f"Invalid optimizer {args.optim}")
# tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# long_text = "Collie is a python package for finetuning large language models. " * 100
# train_sample = tokenizer(long_text, max_length=2048, truncation=True, return_tensors="pt").input_ids.squeeze(0)
# print(f"train sample length: {train_sample.shape[0]}")
train_sample = torch.randint(100, 20000, (2048,))
train_dataset = [{"input_ids": train_sample, "labels": train_sample} for _ in range(50)]
monitors = [
    MemoryMonitor(config),
    LossMonitor(config),
    TGSMonitor(config),
]
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    config=config,
    train_dataset=train_dataset,
    monitors=monitors,
    callbacks=[NVMemoryCallback()],
)
trainer.train()
