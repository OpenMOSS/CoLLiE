import sys
sys.path.append("../../")
import os
import time
import argparse

import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from collie.models.llama.model import LlamaForCausalLM
from collie import Callback
from collie.controller import Trainer
from collie.config import CollieConfig
from collie.optim import Adan, Lomo, Lion, SophiaG
from collie.utils import env

nvmlInit()

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model_name", type=str, default="huggyllama/llama-65b")
arg_parser.add_argument("--optim", type=str, default="lomo")
arg_parser.add_argument("--tp_size", type=int, default=8)
arg_parser.add_argument("--pp_size", type=int, default=1)
arg_parser.add_argument("--use_flash", type=int, default=1)
args = arg_parser.parse_args()

model_name = args.model_name.split("/")[-1]
record_file_name = f"{model_name}_{args.optim}_tp{args.tp_size}_pp{args.pp_size}_1*2048_gas2_flashattn.txt"
if not os.path.exists("memory"):
    os.mkdir("memory")
record_file_name = os.path.join("memory", record_file_name)
with open(record_file_name, "w") as f:
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    f.write(f"{current_time}\n")


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


config = CollieConfig.from_pretrained(args.model_name)
config.tp_size = args.tp_size
config.dp_size = 1
config.pp_size = args.pp_size
config.train_epochs = 1
config.train_micro_batch_size = 1
config.gradient_accumulation_steps = 2
config.use_flash = args.use_flash
if args.optim == "lomo":
    config.gradient_accumulation_steps = 1
    config.ds_config = {
        "bf16": {"enabled": True},
    }
else:
    config.ds_config = {
        "bf16": {"enabled": True},
        # "zero_optimization": {"stage": 1},
        "zero_allow_untested_optimizer": True,
    }

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
train_sample = torch.randint(100, 20000, (2048,))
train_dataset = [{"input_ids": train_sample, "labels": train_sample} for _ in range(50)]

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    config=config,
    train_dataset=train_dataset,
    callbacks=[NVMemoryCallback()],
)
trainer.train()
