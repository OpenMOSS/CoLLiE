import os
import sys
import torch
import json
from torch.utils.data import DataLoader
from colossalai.utils import print_rank_0

sys.path.append("../../")
from datasets import load_dataset
from collie.models.llama_colossalai import HFLikeTokenizer, Tokenizer, ModelArgs
from collie.models.llama_colossalai import load_state_dict, get_7B_llama, get_13B_llama
from collie.trainer.colossalai_trainer import ColossalaiTrainer, TrainerArgs

tokenizer_path = '/mnt/petrelfs/zhangshuo/projects/OptiLLM/colossalai/llama/tokenizer.model'
alpaca_data_path = './alpaca_data.json'
llama_7b_path = "hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf"
llama_13b_path = "hdd:s3://opennlplab_hdd/models/llama/llama-13b-hf"


def collate_fn(mode, batch, tokenizer, max_length=512):
    input_ids_list, label_ids_list = [], []
    assert mode in ['train', 'eval']
    if mode == 'train':
        for sample in batch:
            sample_prompt = sample['prompt']
            sample_label = sample['label']
            sample_prompt_ids = tokenizer(sample_prompt, bos=True, eos=False)['input_ids'].tolist()
            sample_label_ids = tokenizer(sample_label, bos=False, eos=True)['input_ids'].tolist()
            # heuristic truncation:
            if len(sample_prompt_ids) + len(sample_label_ids) >= max_length:
                if len(sample_label_ids) >= max_length:  # too too long, should be rare case
                    sample_label_ids = sample_label_ids[:64]
                temp_length = (max_length - len(sample_label_ids)) // 2
                sample_prompt_ids = sample_prompt_ids[:temp_length] + sample_prompt_ids[-temp_length:]
                # print_rank_0(f"[DEBUG] do heuristic truncation due to the too long train sample.")
            input_ids_list.append(sample_prompt_ids + sample_label_ids)
            label_ids_list.append([0] * len(sample_prompt_ids) + sample_label_ids)
        # pad to longest
        batch_size = len(input_ids_list)
        longest = max(len(input_ids) for input_ids in input_ids_list)
        assert longest <= max_length, f'detect train input length = {longest} exceed max_length = {max_length}'
        input_ids_tensor = torch.full((batch_size, longest), 0).long()
        label_ids_tensor = torch.full((batch_size, longest), 0).long()
        for i in range(batch_size):
            input_ids = input_ids_list[i]
            label_ids = label_ids_list[i]
            assert len(input_ids) == len(label_ids)
            input_ids_tensor[i, :len(input_ids)] = torch.LongTensor(input_ids)
            label_ids_tensor[i, :len(label_ids)] = torch.LongTensor(label_ids)
        return {'input_ids': input_ids_tensor}, label_ids_tensor
    else:
        for sample in batch:
            sample_prompt = sample['prompt']
            sample_prompt_ids = tokenizer(sample_prompt, bos=True, eos=False)['input_ids'].tolist()
            # heuristic truncation:
            if len(sample_prompt_ids) >= max_length - 64:  # reserve 64 for generate position
                temp_length = (max_length - 64) // 2
                sample_prompt_ids = sample_prompt_ids[:temp_length] + sample_prompt_ids[-temp_length:]
                # print_rank_0(f"[DEBUG] do heuristic truncation due to the too long eval sample.")
            input_ids_list.append(sample_prompt_ids)
        # pad to max_length
        batch_size = len(input_ids_list)
        longest = max(len(input_ids) for input_ids in input_ids_list)
        assert longest <= max_length, f'detect eval input length = {longest} exceed max_length-64 = {max_length - 64}'
        input_ids_tensor = torch.full((batch_size, max_length), 0).long()
        for i in range(batch_size):
            input_ids = input_ids_list[i]
            input_ids_tensor[i, :len(input_ids)] = torch.LongTensor(input_ids)
        labels = [sample['label'] for sample in batch]
        return {'input_ids': input_ids_tensor}, labels


def main():
    tokenizer = HFLikeTokenizer(
        tokenizer=Tokenizer(tokenizer_path)
    )

    

    model_args = ModelArgs()
    model_args.pp_size = 8
    model_args.micro_batch_num = 4
    model_args.fp16 = True
    model_args.checkpoint = True
    model_args.dense: str = "fused"  # raw, fused, apex
    model_args.rms_norm: str = "raw"  # raw, apex
    model_args.attention: str = "flash"  # raw, flash, col_flash, mem_eff
    model_args.rotary_emb: str = "fused"  # raw, fused

    trainer_args = TrainerArgs()
    trainer_args.epochs = 100
    trainer_args.eval_max_length = 128
    trainer_args.eval_per_steps = 500
    trainer_args.eval_per_epoches = 1
    trainer_args.eval_stop_tokens = [2]
    trainer_args.eval_use_cache = False

    with open(alpaca_data_path, encoding='utf-8') as f:
        alpaca_data = json.load(f)
    # part of data for debug
    # train_data = alpaca_data[:2560]
    # eval_data = alpaca_data[-32:] # reserve last 32 sample for eval
    # full data for real train
    train_data = alpaca_data[:49920]
    eval_data = alpaca_data[-32:]  # reserve last 32 sample for eval
    train_dataloader = DataLoader(
        train_data,
        batch_size=4,
        collate_fn=lambda x: collate_fn(mode='train', batch=x, tokenizer=tokenizer, max_length=1024),
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_data,
        batch_size=4,
        collate_fn=lambda x: collate_fn(mode='eval', batch=x, tokenizer=tokenizer, max_length=1024),
        drop_last=True,
    )
    model = get_13B_llama(model_args)
    state_dict = load_state_dict(model_args=model_args, s3_folder=llama_13b_path)
    # model = get_7B_llama(model_args)
    # state_dict = load_state_dict(model_args=model_args, s3_folder=llama_7b_path)
    model.load_state_dict(state_dict)
    
    def compute_metrics(batch, epoch, step, max_show_samples=4):
        print("\n")
        for tokens in batch[0]["input_ids"][:max_show_samples]:
            print("-" * 20)
            tokens = tokens.tolist()
            # print(tokens)
            sentence = tokenizer.decode(tokens)
            print(sentence)
            print("-" * 20)
        print("\n")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-5)
    trainer = ColossalaiTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        trainer_args=trainer_args
    )
    trainer.train()
    # trainer.eval()


# Command: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=8 train_alpaca.py
if __name__ == "__main__":
    try:
        main()
    except:
        if os.environ.get("RANK") == "0" or os.environ.get("RANK") == f"{int(os.environ.get('WORLD_SIZE')) - 1}":
            import rich

            console = rich.console.Console()
            console.print_exception(show_locals=True)
        print(f"\nExceptions at Rank: {os.environ.get('RANK')}\n")
