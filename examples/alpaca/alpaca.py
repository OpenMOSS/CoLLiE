import os
import json
import copy
import random

import torch
from torch.utils.data import Dataset

# dataset
class AlpacaDataset(Dataset):
    def __init__(self, data_path='dummy', tokenizer=None):
        if data_path != 'dummy':
            self.alpaca = self.load_data(data_path)
        else:
            self.alpaca = ['This is my life.' for _ in range(4)]
        
    def load_data(self, data_path):
        with open(data_path, encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self):
        return len(self.alpaca)
    
    def __getitem__(self, idx):
        return self.alpaca[idx]
    
def train_collate_fn(batch, max_length=512, tokenizer=None):
    input_ids_list = []
    label_ids_list = []
    for sample in batch:
        sample_prompt = sample['prompt']
        sample_label = sample['label']
        sample_prompt_ids = [1] + tokenizer.encode(sample_prompt, add_special_tokens=False)
        sample_label_ids = tokenizer.encode(sample_label, add_special_tokens=False)
        # label长度四分之一 prompt长度四分之三
        label_length = 128
        prompt_length = max_length - label_length
        # label处理
        if len(sample_label_ids) < label_length - 1:
            sample_label_ids = sample_label_ids + ( (label_length -1 - len(sample_label_ids)) * [-100] ) + [2]
        else:
        # 如果label的长度大于128，则裁剪到128
            sample_label_ids = sample_label_ids[:label_length - 1] + [2]
        # prompt处理
        if len(sample_prompt_ids) <= prompt_length:
            sample_prompt_ids = [0] * (prompt_length - len(sample_prompt_ids)) + sample_prompt_ids
        else:
            temp_length = prompt_length // 2
            sample_prompt_ids = sample_prompt_ids[:temp_length] + sample_prompt_ids[-temp_length:]
        input_ids_list.append(sample_prompt_ids)
        label_ids_list.append(sample_prompt_ids + sample_label_ids)
    # pad to longest
    batch_size = len(input_ids_list)
    input_ids_tensor = torch.full((batch_size, prompt_length), 0).long()
    label_ids_tensor = torch.full((batch_size, prompt_length + label_length), 0).long()
    for i in range(batch_size):
        input_ids = input_ids_list[i]
        label_ids = label_ids_list[i]
        input_ids_tensor[i, -len(input_ids):] = torch.LongTensor(input_ids)
        label_ids_tensor[i, -len(label_ids):] = torch.LongTensor(label_ids)
    return (label_ids_tensor, label_ids_tensor)

def eval_collate_fn(batch, max_length=512, tokenizer=None):
    input_ids_list = []
    label_ids_list = []
    for sample in batch:
        sample_prompt = sample['prompt']
        sample_label = sample['label']
        sample_prompt_ids = [1] + tokenizer.encode(sample_prompt, add_special_tokens=False)
        sample_label_ids = tokenizer.encode(sample_label, add_special_tokens=False)
        # label长度四分之一 prompt长度四分之三
        label_length = 128
        prompt_length = max_length - label_length
        # label处理
        if len(sample_label_ids) < label_length - 1:
            sample_label_ids = sample_label_ids + ( (label_length -1 - len(sample_label_ids)) * [-100] ) + [2]
        else:
        # 如果label的长度大于128，则裁剪到128
            sample_label_ids = sample_label_ids[:label_length - 1] + [2]
        # prompt处理
        if len(sample_prompt_ids) <= prompt_length:
            sample_prompt_ids = [0] * (prompt_length - len(sample_prompt_ids)) + sample_prompt_ids
        else:
            temp_length = prompt_length // 2
            sample_prompt_ids = sample_prompt_ids[:temp_length] + sample_prompt_ids[-temp_length:]
        input_ids_list.append(sample_prompt_ids)
        label_ids_list.append(sample_prompt_ids + sample_label_ids)
    # pad to longest
    batch_size = len(input_ids_list)
    input_ids_tensor = torch.full((batch_size, prompt_length), 0).long()
    label_ids_tensor = torch.full((batch_size, prompt_length + label_length), 0).long()
    for i in range(batch_size):
        input_ids = input_ids_list[i]
        label_ids = label_ids_list[i]
        input_ids_tensor[i, -len(input_ids):] = torch.LongTensor(input_ids)
        label_ids_tensor[i, -len(label_ids):] = torch.LongTensor(label_ids)
    return (input_ids_tensor, label_ids_tensor)
