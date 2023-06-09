import sys
sys.path.append("../../")
import os
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationConfig

from collie.models.moss import MossForCausalLM
from collie.module import GPTLMLoss
from collie.config import CollieConfig
from collie.log import logger
from collie.utils import env, setup_distribution
from collie.controller import Trainer

from moss_002_sft import get_dataset, collate_fn
from metric import SFTDecodeMetric, SFTAccMetric

DS_CONFIG = {
    "fp16": {
        "enabled": True
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

pretrained_model = "fnlp/moss-moon-003-sft"
data_dir = "moss_002_sft_data"
data_num = -1
test_size = 256

# Collie Configuration
config = CollieConfig.from_pretrained(
    pretrained_model, tp_size=2, dp_size=1, pp_size=4, train_epochs=10,
    eval_per_n_steps=0, eval_per_n_epochs=1, train_micro_batch_size=2,
    gradient_accumulation_steps=64, eval_batch_size=1, ds_config=DS_CONFIG,
    trust_remote_code=True
)

def eval_fn(trainer, batch):
    # batch: tuple
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
    return {
        "total": total,
        "right": right,
    }

# Usually it will be set up when initializing a model.
# In order to process and save data, we call this method manually.
setup_distribution(config)

# tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
train_dataset, val_dataset = get_dataset(tokenizer, data_dir, num=data_num, test_size=test_size)

# load from petrel s3
# s3_folder = os.path.join("hdd:s3://opennlplab_hdd/models", pretrained_model)
# model = MossForCausalLM(config)
# state_dict = MossForCausalLM.load_parallel_state_dict(s3_folder, config, process_exclusion=False, protocol="petrel")
# model.load_state_dict(state_dict)
# load from local:
model = MossForCausalLM.from_pretrained(pretrained_model, config=config)

# metrics = {"sftacc": SFTAccMetric(), "sftdecode": SFTDecodeMetric(tokenizer)}
metrics = {"sftacc": SFTAccMetric()}
trainer = Trainer(
    model, config, loss_fn=GPTLMLoss(-100), eval_fn=eval_fn,
    train_dataset=train_dataset, eval_dataset=val_dataset,
    train_dataset_collate_fn=lambda x: collate_fn(x, tokenizer),
    eval_dataset_collate_fn=lambda x: collate_fn(x, tokenizer),
    metrics=metrics
)
trainer.train()