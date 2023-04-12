import os
import sys
GRANDPA = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(GRANDPA)
from datasets import load_dataset,Dataset

from tunelite.models.llama_colossalai import HFLikeTokenizer, Tokenizer, ModelArgs, get_7B_llama, load_state_dict, get_7B_llama
from tunelite.trainer.colossalai_trainer import ColossalaiTrainer, TrainerArgs

import torch
from torch.utils.data import DataLoader

from mydataloader import *
from typing import Sequence
import transformers

import pdb

def _tokenize_fn(strings: Sequence[str], tokenizer: HFLikeTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized =  tokenizer(
                        strings,
                        max_length=1024,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt',
                        bos=True,
                        eos=True
                    )
    input_ids = labels = tokenized['input_ids'][:,:1023]
    input_ids_lens = labels_lens = tokenized['input_ids'].ne(tokenizer.pad_token_id).sum().item()
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
    

def collate_fn(batch, tokenizer, max_length=1024, bos=True, eos=True):
    text = [e['source'] + e['target'] for e in batch]
    tknz_batch = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        bos=bos,
        eos=eos
    )
    tknz_batch['input_ids'] = tknz_batch['input_ids'][:, :max_length-1]
    return {
        'input_ids': tknz_batch['input_ids'].long()
    }, tknz_batch['input_ids'].long()

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path)
    return dict(train_dataset=train_dataset, eval_dataset=None)


def main():
    tokenizer = HFLikeTokenizer(
        tokenizer=Tokenizer(model_path='/mnt/petrelfs/zhangshuo/projects/OptiLLM/colossalai/llama/tokenizer.model'))
    def compute_metrics(batch, generated_batch, epoch, step):
        print("\n")
        print("\n".join([tokenizer.decode(token.tolist()) for token in generated_batch[0]["input_ids"]][:1]))
        print("\n")
    model_args = ModelArgs()
    model_args.pp_size = 7
    model_args.micro_batch_size = 2
    model_args.fp16 = True
    model_args.checkpoint = True
    model_args.dense = "fused"
    model_args.attention = "flash"
    model_args.rotary_emb = "fused"
    
    trainer_args = TrainerArgs()
    trainer_args.eval_max_length = 128
    trainer_args.eval_per_steps = 10
    trainer_args.eval_per_epoches = 1
    trainer_args.learning_rate = 2e-5
    
    model = get_7B_llama(model_args)
    # state_dict = load_state_dict(model_args=model_args, s3_folder="hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf")
    # model.load_state_dict(state_dict)
    # (DatasetDict | Dataset | IterableDatasetDict | IterableDataset)
    data_path = '/mnt/lustre/zhangshuo/projects/stanford_alpaca/alpaca_data.json'
    data_module = make_supervised_data_module(tokenizer, data_path)
    dataset = data_module['train_dataset']
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )
    print('成功导入train_dataloader')
    print('加载数据',len(dataset))
    eval_dataloader = DataLoader(
        [{"instruction": "Create a classification task by clustering the given list of items.", 
          "input": "Apples, oranges, bananas, strawberries, pineapples",
          "output": ""
          } for _ in range(8)],
        batch_size=8,
        collate_fn=lambda x: collate_fn(x, tokenizer),
    )
    print('成功导入eval_dataloader')
    trainer = ColossalaiTrainer(model=model,
                                train_dataloader=train_dataloader,
                                eval_dataloader=eval_dataloader,
                                tokenizer=tokenizer,
                                compute_metrics=compute_metrics,
                                trainer_args=trainer_args)
    print('成功初始化traniner!')
    trainer.train()
    
    
# Command: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=8 train_colossalai.py
if __name__ == "__main__":
    try:
        main()
    except:
        # if os.environ.get("RANK") == "0" or os.environ.get("RANK") == f"{int(os.environ.get('WORLD_SIZE'))-1}":
        import rich
        console = rich.console.Console()
        console.print_exception(show_locals=False)
        print(f"\nExceptions at Rank: {os.environ.get('RANK')}\n")