import sys
sys.path.append('..')
import tunelite as tl
import wandb
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, HfArgumentParser, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig
from tunelite.arguments import ModelArguments, DataArguments, TuneLiteArguments
import torch
from datasets import load_from_disk
import os
import datetime


def collate_fn(batch, tokenizer, max_len=512):
    text = [e['text'] for e in batch]
    tknz_batch = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return {
        'input_ids': tknz_batch['input_ids'],
        'attention_mask': tknz_batch['attention_mask'],
        'labels': tknz_batch['input_ids']
    }


def compute_metrics(all_pred, eval_dataset):
    print(len(all_pred), len(eval_dataset))
    return {'my_metric': len(all_pred[0]) - len(eval_dataset[0])}  # dummy metric


def train():
    torch.set_default_dtype(torch.bfloat16)
    parser = HfArgumentParser((ModelArguments, DataArguments, TuneLiteArguments))
    if sys.argv[-1].endswith(".yaml"):
        model_args, data_args, tl_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, tl_args = parser.parse_args_into_dataclasses()

    if tl_args.local_rank in [-1, 0]:
        wandb.init(
            project="tunelite",
            name=tl_args.run_name,
            tags=[data_args.data_tag, tl_args.tag],
            config={'model_args': model_args, 'data_args': data_args, 'tl_args': tl_args},
        )

    if tl_args.local_rank == 0:
        print(f"[{datetime.datetime.today()}] Loading model.")
    # load model
    ds_config = tl_args.deepspeed
    dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.gradient_checkpointing = True
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        local_files_only=True,
        config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if tl_args.local_rank == 0:
        print(f"[{datetime.datetime.today()}] Loading dataset.")
    dataset = load_from_disk("/remote-home/klv/exps/MossOn3090/data/pile-10k")['train'].select(range(1000))
    eval_dataloader = DataLoader(
        dataset.select(range(4)),
        batch_size=tl_args.per_device_train_batch_size,
        collate_fn=lambda x: collate_fn(x, tokenizer, max_len=data_args.max_length)
    )
    trainer = tl.trainer.InplaceZeroTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=collate_fn,
        eval_dataloader=eval_dataloader,
        eval_dataset=dataset.select(range(4)),
        tl_args=tl_args,
        data_args=data_args,
        compute_metrics=compute_metrics,
    )
    trainer.train()


# run with $deepspeed --include localhost:3,4 train_inplace_zero.py config/zero_args.yaml
if __name__ == "__main__":
    train()
