"""
这是一个测试不同的模型不同的并行配置下的TGS的程序。
"""
import sys
sys.path.append('../../')
sys.path.append("/mnt/petrelfs/gutianle/Megatron-LM/")
import argparse
import torch

from collie.config import CollieConfig
from collie.data import CollieDatasetForTraining
from collie.models.llama.model import LlamaForCausalLM
from collie.models.moss_moon import Moss003MoonForCausalLM
from collie.utils.monitor import TGSMonitor, LossMonitor
from collie.controller.trainer import Trainer
from collie.module import GPTLMLoss

from transformers import LlamaTokenizer, AutoTokenizer
from datasets import load_dataset

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--model_name', type=str, default="decapoda-research/llama-7b-hf")
arg_parser.add_argument('--tp_size', type=int, default=1)
arg_parser.add_argument('--pp_size', type=int, default=1)
arg_parser.add_argument('--batch_size', type=int, default=1)
arg_parser.add_argument('--zero3', type=int, default=0)
arg_parser.add_argument('--zeroplus', type=int, default=0)
arg_parser.add_argument("--use_flash", type=int, default=1)
args = arg_parser.parse_args()
tag = f"tp{args.tp_size}pp{args.pp_size}bs{args.batch_size}_"
if args.use_flash:
    tag += "flash"
if args.zero3:
    tag += 'zero3'
elif args.zeroplus:
    tag += 'zeroplus'

config = CollieConfig.from_pretrained(args.model_name)
config.tp_size = args.tp_size
config.dp_size = 1
config.pp_size = args.pp_size
config.train_epochs = 1
config.train_micro_batch_size = args.batch_size
config.use_flash = args.use_flash
config.ds_config = {
    "fp16": {
        "enabled": True
    },
    "zero_allow_untested_optimizer": True,
    "zero_force_ds_cpu_optimizer": False,
    "monitor_config": {
        "enabled": True,
        "tag": tag,
        "csv_monitor": {
            "enabled": True,
            "output_path": "./results/"
        }
    }
}

if args.zero3:
    config.update(
        {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": False
                }
            }
        }
    )
elif args.zeroplus:
    config.update({
        "zero_optimization": {
            "stage": 3,
            "zero_quantized_weights": True,
            "zero_hpz_partition_size": 16,
            "zero_quantized_gradients": True
        }
    })

if "decapoda-research/llama" in args.model_name:
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

dataset = [
    {
        "text": f"{sample['text']}"
    } for sample in load_dataset(path="NeelNanda/pile-10k", split="train")
][:1000]
dataset = CollieDatasetForTraining(dataset=dataset, tokenizer=tokenizer, shuffle=True, max_length=2048, seed=42)

if "llama" in args.model_name:
    model = LlamaForCausalLM.from_pretrained(args.model_name,config)
elif "moss" in args.model_name:
    model = Moss003MoonForCausalLM.from_pretrained(args.model_name, config=config)
    
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

monitors = [TGSMonitor(config), LossMonitor(config)]

trainer = Trainer(
    model = model,
    optimizer = optimizer,
    config = config,
    train_dataset = dataset,
    monitors = monitors,
    loss_fn = GPTLMLoss(-100)
)

trainer.train()

# torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=8 tgs_strategies.py 