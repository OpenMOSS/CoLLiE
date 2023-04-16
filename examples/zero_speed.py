import sys

sys.path.append('..')
import os
from dataclasses import dataclass

import torch
import wandb
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig

from mcqa.arguments import ModelArguments, DataArguments, MyCollieArguments
from collie.trainer import InplaceZeroTrainer
from collie.log import print


@dataclass
class DataCollatorForSpeed:
    tokenizer: None
    max_length: None

    def __call__(self, batch):
        text = [e['text'] for e in batch]
        tokenized_text = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        if tokenized_text['input_ids'].shape[1] != self.max_length:
            print(tokenized_text['input_ids'].shape)
        return {
            'input_ids': tokenized_text['input_ids'],
            'attention_mask': tokenized_text['attention_mask'],
            'labels': tokenized_text['input_ids'],
        }


def speed_test():
    torch.set_default_dtype(torch.bfloat16)
    parser = HfArgumentParser((ModelArguments, DataArguments, MyCollieArguments))
    if sys.argv[-1].endswith(".yaml"):
        model_args, data_args, collie_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, collie_args = parser.parse_args_into_dataclasses()
    if collie_args.local_rank in [-1, 0]:
        wandb.init(
            project="collie",
            entity='collie_exp',
            config=collie_args
        )

    """
    init model
    """
    ds_config = collie_args.deepspeed
    dschf = HfDeepSpeedConfig(ds_config)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.gradient_checkpointing = collie_args.gradient_checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=True,
        config=config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token_id = 0

    """
    prepare dataset
    """
    dataset = load_from_disk("/remote-home/klv/exps/MossOn3090/data/pile-10k")['train'].select(range(1000))

    """
    init trainer
    """
    trainer = InplaceZeroTrainer(
        model=model,
        collie_args=collie_args,
        data_collator=DataCollatorForSpeed(tokenizer=tokenizer, max_length=data_args.max_length),
        train_dataset=dataset,
        tokenizer=tokenizer,
        eval_dataset=None,
        compute_metrics=None,
    )

    # test train speed
    trainer.train()
    # test generate speed
    # eval_dataloader = trainer.get_eval_dataloader(dataset)


if __name__ == '__main__':
    speed_test()
