import sys
sys.path.append("../../")
import os

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationConfig

from collie.models.moss import MossForCausalLM
from collie.module import GPTLMLoss, PipelineGenerationMixin
from collie.config import CollieConfig
from collie.log import logger
from collie.utils import env, setup_distribution
# from collie.trainer import Trainer
from mytrainer import Trainer

from moss_002_sft import get_dataset, collate_fn
from metric import SFTDecodeMetric, SFTAccMetric

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

# Generation config
# Used in SFTDecodeMetric if necessary
generate_config = GenerationConfig(
    max_new_tokens=200, do_sample=True, top_k=40, top_p=0.8,
    temperature=0.7, repetition_penalty=1.02, eos_token_id=106068
)
pretrained_model = "fnlp/moss-moon-003-sft"
data_dir = "moss_002_sft_data"
generate_dir = "generate_res"
data_num = -1
test_size = 12

# Collie Configuration
config = CollieConfig.from_pretrained(
    pretrained_model, tp_size=2, dp_size=2, pp_size=2, train_epochs=10,
    eval_per_n_steps=0, eval_per_n_epochs=1, train_micro_batch_size=2,
    gradient_accumulation_steps=2, eval_batch_size=1, ds_config=DS_CONFIG,
    trust_remote_code=True
)

def eval_fn(trainer, batch, train_meta):
    # batch: tuple
    # train_meta: dict contains epoch_idx, batch_idx, last_loss
    input_ids, labels = batch
    # forward
    if env.pp_size > 1:
        logits = trainer.engine.eval_batch(batch)
    else:
        logits = trainer.engine(input_ids=input_ids.cuda()).logits
    shift_preds = logits[..., :-1, :].argmax(dim=-1)
    shift_labels = labels[..., 1:].to(logits.device)
    right = (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum()
    total = (shift_labels != -100).sum()
    # generate
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
        "total": total,
        "right": right,
        "generate": gen_res,
        "train_meta": train_meta
    }

# Usually it will be set up when initializing a model.
# In order to process and save data, we call this method manually.
setup_distribution(config)

# tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
generate_config.pad_token_id = tokenizer.pad_token_id
train_dataset, val_dataset = get_dataset(tokenizer, data_dir, num=data_num, test_size=test_size)

# load from petrel s3
s3_folder = os.path.join("hdd:s3://opennlplab_hdd/models", pretrained_model)
model = MossForCausalLM(config)
# state_dict = MossForCausalLM.load_parallel_state_dict(s3_folder, config, process_exclusion=False, protocol="petrel")
# model.load_state_dict(state_dict)
# load from local:
# model = MossForCausalLM.from_pretrained(pretrained_model, config=config)

# metrics = [SFTAccMetric(), SFTDecodeMetric(tokenizer, generate_dir)]
metrics = [SFTAccMetric()]
trainer = Trainer(
    model, config, loss_fn=GPTLMLoss(-100), eval_fn=eval_fn,
    train_dataset=train_dataset, eval_dataset=val_dataset,
    train_dataset_collate_fn=lambda x: collate_fn(x, tokenizer),
    eval_dataset_collate_fn=lambda x: collate_fn(x, tokenizer),
    metrics=metrics, eval_config=generate_config
)
trainer.train()