import os
import sys
sys.path.append("/mnt/lustre/zhangshuo/projects/TuneLite")

from datasets import load_dataset
from transformers import HfArgumentParser

from tunelite.models.llama_colossalai import HFLikeTokenizer, Tokenizer, ModelArgs, get_7B_llama, load_state_dict
from tunelite.trainer.colossalai_trainer import ColossalaiTrainer, TrainerArgs

import torch
from torch.utils.data import DataLoader

def collate_fn(batch, tokenizer, max_length=1024, bos=True, eos=True):
    text = [e['text'] for e in batch]
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
    

def main():
    tokenizer = HFLikeTokenizer(
        tokenizer=Tokenizer(model_path='/mnt/petrelfs/zhangshuo/projects/OptiLLM/colossalai/llama/tokenizer.model'))
    def compute_metrics(batch, generated_batch, epoch, step):
        print("\n")
        print([tokenizer.decode(token.tolist()) for token in generated_batch[0]["input_ids"]])
        print("\n")
    model_args = ModelArgs()
    model_args.pp_size = 8
    model_args.micro_batch_size = 4
    model_args.fp16 = True
    model_args.checkpoint = True
    model_args.dense = "fused"
    trainer_args = TrainerArgs()
    trainer_args.eval_max_length = 128
    trainer_args.eval_per_steps = 10
    trainer_args.eval_per_epoches = 1
    trainer_args.learning_rate = 2e-4
    model = get_7B_llama(model_args)
    state_dict = load_state_dict(model_args=model_args)
    model.load_state_dict(state_dict)
    dataset = load_dataset("NeelNanda/pile-10k")["train"]
    train_dataloader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=lambda x: collate_fn(x, tokenizer, 1024),
    )
    eval_dataloader = DataLoader(
        [{"text": "When I was young, I used to "} for _ in range(14)],
        batch_size=4,
        collate_fn=lambda x: collate_fn(x, tokenizer, 1024, eos=False),
    )
    trainer = ColossalaiTrainer(model=model,
                                train_dataloader=train_dataloader,
                                eval_dataloader=eval_dataloader,
                                tokenizer=tokenizer,
                                compute_metrics=compute_metrics,
                                trainer_args=trainer_args)
    trainer.train()
    
        
if __name__ == "__main__":
    try:
        main()
    except:
        import rich
        console = rich.console.Console()
        console.print_exception(show_locals=False)