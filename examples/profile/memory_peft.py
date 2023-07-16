import sys

sys.path.append("../../")
import os
import time
import argparse

import torch
from transformers import LlamaTokenizer, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from peft import PromptTuningInit, PromptTuningConfig, TaskType, LoraConfig, PrefixTuningConfig, PromptEncoderConfig

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
arg_parser.add_argument("--peft", type=str, default="lora")
arg_parser.add_argument("--pp_size", type=int, default=1)
arg_parser.add_argument("--use_flash", type=int, default=1)
args = arg_parser.parse_args()

model_name = args.model_name.split("/")[-1]
record_file_name = f"{model_name}_{args.peft}_pp{args.pp_size}_1*2048_gas2_flashattn.txt"
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
config.tp_size = 1
config.dp_size = 1
config.pp_size = args.pp_size
config.train_epochs = 1
config.train_micro_batch_size = 1
config.gradient_accumulation_steps = 2
config.use_flash = args.use_flash
config.ds_config = {
    "bf16": {"enabled": True},
}
if args.peft == "lora":
    config.peft_config = LoraConfig(
        r=4,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
elif args.peft == "prefix-tuning":
    config.peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        num_virtual_tokens=20
    )
elif args.peft == "p-tuning":
    config.peft_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        encoder_hidden_size=128
    )
elif args.peft == "prompt-tuning":
    config.peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=8,
        prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
        tokenizer_name_or_path=args.model_name,
    )
else:
    raise NotImplementedError


model = LlamaForCausalLM.from_config(config=config)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

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
