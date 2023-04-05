import sys
sys.path.append('..')
import tunelite as tl
from tunelite.models import llama
import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
from transformers import HfArgumentParser
from tunelite.arguments import ModelArguments, DataArguments, TuneLiteArguments
import os
from transformers import set_seed
import wandb


class MyDataset(Dataset):
    def __init__(self, data, tknz, max_len=512):
        super().__init__()
        self.data = data
        self.tknz = tknz
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tknz_text = self.tknz(
            self.data[idx]['text'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
        )
        return {
            'input_ids': tknz_text['input_ids'],
            'attention_mask': tknz_text['attention_mask'],
            'labels': tknz_text['input_ids']
        }


def collate_fn(batch, tknz, max_len=512):
    text = [e['text'] for e in batch]
    tknz_batch = tknz(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    eos_tensors = torch.tensor([tknz.eos_token_id] * tknz_batch['input_ids'].shape[0]).unsqueeze(1)
    tknz_batch['input_ids'] = torch.cat((tknz_batch['input_ids'][:, :max_len-1], eos_tensors), dim=1)
    tknz_batch['attention_mask'] = tknz_batch['attention_mask'][:, :max_len]
    return {
        'input_ids': tknz_batch['input_ids'],
        'attention_mask': tknz_batch['attention_mask'],
        'labels': tknz_batch['input_ids']
    }


def compute_metrics(all_pred, eval_dataset):
    print(len(all_pred), len(eval_dataset))
    return {'my_metric': len(all_pred[0]) - len(eval_dataset[0])}  # dummy metric


def train():
    model, tokenizer = llama.load_model(
        ckpt_dir='/remote-home/share/llama/13B/',  # 7B, 13B, 30B, 65B
        tokenizer_path='/remote-home/share/llama/tokenizer.model',
        froze_embeddings=False,
        zero=False,
        tensor_parallel=True,
        pipeline_parallel=False,
        max_batch_size=8,
        max_seq_len=1024,
    )
    tokenizer.pad_token_id = 0

    parser = HfArgumentParser((ModelArguments, DataArguments, TuneLiteArguments))
    if sys.argv[-1].endswith(".yaml"):
        model_args, data_args, tl_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, tl_args = parser.parse_args_into_dataclasses()
    set_seed(tl_args.seed)
    if tl_args.local_rank in [-1, 0]:
        wandb.init(
            project="tunelite",
            name=tl_args.run_name,
            tags=[data_args.data_tag, tl_args.tag],
            config={'model_args': model_args, 'data_args': data_args, 'tl_args': tl_args},
        )

    dataset = load_from_disk("/remote-home/klv/exps/MossOn3090/data/pile-10k")['train'].select(range(1000))
    train_dataloader = DataLoader(
        dataset,
        batch_size=tl_args.per_device_train_batch_size,
        collate_fn=lambda x: collate_fn(x, tokenizer, max_len=data_args.max_length)
    )
    eval_dataloader = DataLoader(
        dataset.select(range(4)),
        batch_size=tl_args.per_device_train_batch_size,
        collate_fn=lambda x: collate_fn(x, tokenizer, max_len=data_args.max_length)
    )

    trainer = tl.trainer.InplaceTensorTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        eval_dataset=dataset.select(range(4)),
        tl_args=tl_args,
        compute_metrics=compute_metrics,
    )
    trainer.train()


# run with $torchrun --nproc_per_node 2 train_inplace_tensor.py config/tensor_args.yaml
if __name__ == "__main__":
    train()
