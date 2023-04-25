import json

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

def get_prompt(sample):
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
    prompt += f"### Instruction:\n{sample['instruction']}\n"
    if sample["input"] != "":
        prompt += f"### Input:\n{sample['input']}\n"
    prompt += f"### Response:\n"

    return prompt, sample["output"]

class AlpacaDataset(Dataset):
    def __init__(self, alpaca, tokenizer, max_len, train=True):
        self.alpaca = alpaca
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train = train
        self.process()

    def process(self):
        self.data = []
        self.label = []
        for sample in self.alpaca:
            prompt, label = get_prompt(sample)
            prompt_ids = self.tokenizer(prompt)["input_ids"]
            label_ids = self.tokenizer(label)["input_ids"]
            # heuristic truncation:
            if self.train:
                if len(prompt_ids) + len(label_ids) >= self.max_len:
                    if len(label_ids) >= self.max_len:
                        # too too long, should be rare case
                        label_ids = label_ids[:self.max_len]
                    temp_length = (self.max_len - len(label_ids)) // 2
                    prompt_ids = prompt_ids[:temp_length] + prompt_ids[-temp_length:]

                input_len = len(prompt_ids)
                prompt_ids = prompt_ids + label_ids
                label_ids = [0] * input_len + label_ids
                assert len(prompt_ids) == len(label_ids), f"{len(prompt_ids)}, {len(label_ids)}"
            else:
                if len(prompt_ids) >= self.max_len - 64:
                    # reserve 64 for generate position
                    temp_length = (self.max_len - 64) // 2
                    prompt_ids = prompt_ids[:temp_length] + prompt_ids[-temp_length:]
                # print_rank_0(f"[DEBUG] do heuristic truncation due to the too long eval sample.")
            
            self.data.append(prompt_ids)
            self.label.append(label_ids)

    def __len__(self):
        return len(self.alpaca)

    def __getitem__(self, index):
        return {
            "input_ids": self.data[index], 
            "label": self.label[index]
        }
    
class Collator:
    def __init__(self, tokenzer, train=True):
        self.tokenizer = tokenzer
        self.train = train
    
    def __call__(self, features):
        batch_size = len(features)
        longest = max([len(feature["input_ids"]) for feature in features])
        input_ids_tensor = torch.full((batch_size, longest), 0).long()
        if self.train:
            ret_label_ids = torch.full((batch_size, longest), 0).long()
        else:
            ret_label_ids = [feature['label'] for feature in features]
        for i in range(batch_size):
            input_ids = features[i]["input_ids"]
            label_ids = features[i]["label"]
            input_ids_tensor[i, :len(input_ids)] = torch.LongTensor(input_ids)
            if self.train:
                ret_label_ids[i, :len(label_ids)] = torch.LongTensor(label_ids)
        return {"input_ids": input_ids_tensor, "label": ret_label_ids}


def load_alpaca(tokenizer, max_len, num=-1, test_size=0.2):
    # alpaca = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
    split_ = "train"
    if num != -1:
        split_ += f"[:{num}]"
    alpaca = load_dataset("tatsu-lab/alpaca", split=split_)

    # set seed so that every rank got the same data
    alpaca = alpaca.train_test_split(test_size=test_size, seed=42)
    train_alpaca = alpaca["train"]
    test_alpaca = alpaca["test"]

    train_dataset = AlpacaDataset(train_alpaca, tokenizer, max_len, True)
    eval_dataset = AlpacaDataset(test_alpaca, tokenizer, max_len, False)
    return train_dataset, eval_dataset
