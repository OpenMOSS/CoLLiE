import os
import json
import copy
import random

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset
from tqdm import tqdm

from collie.log import logger
from collie.utils import env

class SFTDataset(Dataset):
    # https://github.com/OpenLMLab/MOSS/blob/main/finetune_moss.py
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = copy.deepcopy(self.dataset["input_ids"][index])
        no_loss_spans = copy.deepcopy(self.dataset["no_loss_spans"][index])
        
        data = torch.tensor(data, dtype=torch.long)
        label = copy.deepcopy(data)

        for no_loss_span in no_loss_spans:
            label[no_loss_span[0] : no_loss_span[1]] = -100

        return data, label
    
def collate_fn(batch, tokenizer):
    batch_input_ids, batch_labels = [], []
    for input_ids, label in batch:
        batch_input_ids.append(input_ids)
        batch_labels.append(label)

    batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.eos_token_id)
    batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)

    return batch_input_ids, batch_labels

def process(sample, tokenizer, max_len):
    chat = sample["plain_text"].split("<eoa>")[:-1]
    num_turns = sample["num_turns"]
    meta_instruction = sample["prefix"]

    # encode instruction
    instruction_ids = tokenizer.encode(meta_instruction)
    assert isinstance(instruction_ids, list), instruction_ids
    assert len(instruction_ids) > 0, len(instruction_ids)
    input_ids = copy.deepcopy(instruction_ids)
    # We do not calculate loss for instruction.
    no_loss_spans = [(0, len(instruction_ids))]

    for i in range(num_turns):
        # Collect dialogues
        cur_turn_ids = []
        cur_no_loss_spans = []
        # Add to cur_turn_ids
        cur_turn_ids.extend(tokenizer.encode(chat[i] + "<eoa>"))
        # if key == 'Tool Responses':
        #     # The format tokens (<|Results|>:...<eor>\n) should have losses. 
        #     cur_no_loss_spans.append((len(input_ids + cur_turn_ids) + 5, len(input_ids + cur_turn_ids + cur_ids) - 2))
        if len(input_ids + cur_turn_ids) > max_len:
            # Too long, break
            break
        # Extend input_ids
        input_ids.extend(cur_turn_ids)
        no_loss_spans.extend(cur_no_loss_spans)

    if len(input_ids) == len(instruction_ids):
        # No dialogue, return
        return {"input_ids": [], "no_loss_span": []}
    else:
        return {"input_ids": input_ids, "no_loss_spans": no_loss_spans}


def load_data(save_dir, tokenizer, max_len, num=-1) -> HFDataset:
    info_file = os.path.join(save_dir, "info.json")
    if os.path.exists(save_dir):
        logger.info(f"Loading moss-002-sft from {save_dir}")
        with open(info_file) as fp:
            info = json.load(fp)
        if info["max_len"] != max_len:
            logger.rank_zero_warning(
                f"The loaded data's `max_len` is {info['max_len']}, which is "
                f"not equal to your setting: {max_len}"
            )
    else:
        logger.info(f"Loading moss-002-sft from datasets")
        if env.rank == 0:
            moss_sft = load_dataset("fnlp/moss-002-sft-data", split="train[:500]")
            moss_sft = moss_sft.map(lambda x:process(x, tokenizer, max_len), num_proc=10)
            moss_sft = moss_sft.filter(lambda x:len(x["input_ids"]) != 0)
            info = {
                "name": "moss-002-sft", "max_len": max_len,
                "total": len(moss_sft)
            }
            moss_sft.save_to_disk(save_dir)
            with open(info_file, "w") as fp:
                json.dump(info, fp, indent=4)
    env.barrier()

    moss_sft = HFDataset.load_from_disk(save_dir)
    if num != -1:
        moss_sft = moss_sft.select(range(num))
    logger.info(
        f"Load successfully, total {len(moss_sft)} samples.")
    
    return moss_sft

def get_dataset(tokenizer, save_dir, max_len=2048, num=-1, test_size=0.1):
    moss_sft_data = load_data(save_dir, tokenizer, max_len, num)
    moss_sft_split = moss_sft_data.train_test_split(test_size=test_size)
    train_dataset = SFTDataset(moss_sft_split["train"])
    val_dataset = SFTDataset(moss_sft_split["test"])
    env.barrier()

    return train_dataset, val_dataset

