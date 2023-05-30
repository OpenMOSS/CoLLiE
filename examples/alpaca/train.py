# Command CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=8 train.py
import sys
sys.path.append('../../')
sys.path.append("/mnt/petrelfs/gutianle/Megatron-LM/")
import os

import torch
import torch.distributed as dist
from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig

from collie.models.llama.model import LlamaForCausalLM
from collie.metrics.decode import DecodeMetric
from collie.module import GPTLMLoss, PipelineGenerationMixin
from collie.config import CollieConfig
from collie.trainer.trainer import Trainer
from collie.log import logger
from collie.utils import env, setup_distribution

from alpaca_metric import AlpacaDecodeMetric
from alpaca import AlpacaDataset, train_collate_fn, eval_collate_fn


pretrained_model = 'decapoda-research/llama-7b-hf'


DS_CONFIG = {
    "fp16": {
        "enabled": False
    },
    "zero_allow_untested_optimizer": True,
    "zero_force_ds_cpu_optimizer": False,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 9e-6,
            "weight_decay": 0.1
        }
    },

    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": False
        }
    },
    "steps_per_print": 2000,
}


generate_config = GenerationConfig(
    max_new_tokens=128,
    eos_token_id=2,
    pad_token_id=0,
    bos_token_ids=1
)

config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf")
config.tp_size = 2
config.dp_size = 2
config.pp_size = 2
config.train_epochs = 3
config.train_micro_batch_size = 16
config.eval_batch_size = 32
config.eval_per_n_steps = 100
config.ds_config = {
    "fp16": {"enabled": True},
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 2e-5
        }
    }
}
           
def eval_fn(trainer, batch, train_meta):
    input_ids, labels = batch
    if env.pp_size > 1:
        generation_model = PipelineGenerationMixin(
            engine=trainer.engine
        )
    else:
        generation_model = trainer.model
    gen_res = generation_model.generate(
        input_ids=input_ids.cuda(),
        attention_mask=torch.ones_like(input_ids).cuda(),
        generation_config=trainer.eval_config
    )
    return {
        "generate": gen_res,
        "train_meta": train_meta
    }
    
    
setup_distribution(config)

# tokenizer
tokenizer = LlamaTokenizer.from_pretrained(pretrained_model, 
                                           padding_side="left")
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2




dataset = AlpacaDataset('alpaca_data.json', tokenizer)
train_dataset = dataset[:-32]
eval_dataset = dataset[-32:]


# load model from s3
model = LlamaForCausalLM(config)
state_dict = LlamaForCausalLM.load_parallel_state_dict(
    path="hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf",
    config=config,
    protocol="petrel",
    format="hf"
)
model.load_state_dict(state_dict)

# metric
metrics = {'decode': AlpacaDecodeMetric(tokenizer=tokenizer)}

# trainer
trainer = Trainer(
    model = model,
    config = config,
    loss_fn = GPTLMLoss(-100),
    eval_fn = eval_fn,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    train_dataset_collate_fn=lambda x:train_collate_fn(x, tokenizer=tokenizer),
    eval_dataset_collate_fn=lambda x:eval_collate_fn(x, tokenizer=tokenizer),
    eval_config = generate_config,
    metrics = metrics
)

# train
trainer.train()