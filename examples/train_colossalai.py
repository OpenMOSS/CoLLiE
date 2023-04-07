import sys
sys.path.append("..")

from datasets import load_dataset
from transformers import HfArgumentParser

from tunelite.models.llama_colossalai import HFLikeTokenizer, Tokenizer, ModelArgs, get_7B_llama, load_state_dict
from tunelite.trainer.colossalai_trainer import ColossalaiTrainer, GenerativeDataloader, TrainerArgs

import torch
from torch.utils.data import DataLoader

def collate_fn(batch, tokenizer, max_length=1024):
    text = [e['text'] for e in batch]
    tknz_batch = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    eos_tensors = torch.tensor([tokenizer.eos_token_id] * tknz_batch['input_ids'].shape[0]).unsqueeze(1)
    tknz_batch['input_ids'] = torch.cat((tknz_batch['input_ids'][:, :max_length-1], eos_tensors), dim=1)
    tknz_batch['attention_mask'] = tknz_batch['attention_mask'][:, :max_length]
    return {
        'input_ids': tknz_batch['input_ids']
    }, tknz_batch['input_ids']

def main():
    tokenizer = HFLikeTokenizer(
        tokenizer=Tokenizer(model_path='/mnt/petrelfs/zhangshuo/projects/OptiLLM/colossalai/llama/tokenizer.model'))
    model_args = ModelArgs()
    trainer_args = TrainerArgs()
    model = get_7B_llama(model_args)
    
    state_dict = load_state_dict()
    print(list(state_dict.keys()))
    # dataset = load_dataset("NeelNanda/pile-10k")["train"]
    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     collate_fn=lambda x: collate_fn(x, tokenizer, 1024),
    # )
    eval_dataloader = DataLoader(
        [{"text": "I want to"}],
        collate_fn=lambda x: collate_fn(x, tokenizer, 1024),
    )
    train_dataloader = DataLoader(
        [{"text": "I want to"}],
        collate_fn=lambda x: collate_fn(x, tokenizer, 1024),
    )
    trainer = ColossalaiTrainer(model=model,
                                train_dataloader=train_dataloader,
                                eval_dataloader=eval_dataloader,
                                tokenizer=tokenizer,
                                compute_metrics=None,
                                trainer_args=trainer_args)
    trainer.train()
    
        
if __name__ == "__main__":
    try:
        main()
    except:
        import rich
        console = rich.console.Console()
        console.print_exception(show_locals=True)