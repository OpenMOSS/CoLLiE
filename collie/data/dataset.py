from torch.utils.data import Dataset
import torch
from torch.utils.data.dataset import Dataset
from collie.driver.io import FileIODriver
from transformers import PreTrainedTokenizer

import os
import io
import json
import mmap
import random
import threading
import numpy as np
from functools import reduce
from typing import Optional, Dict, List, Sequence, Tuple
    
class _ShardContainer(list):
    def __init__(self, path, shuffle: bool=False, seed: int = 1024) -> None:
        list.__init__([])
        self.file = None
        self.file_name = None
        self.shuffle = shuffle
        self.seed = seed
        self.threadlocal = threading.local()
        meta_files = []
        for file in FileIODriver.list(path):
            if file.endswith(".meta"):
                meta_files.append(os.path.join(path, file))
        self.meta = [[{"file": file, "offset": meta[0], "length":  meta[1]} for meta in FileIODriver.load(file, mode="rb")] for file in meta_files]
        self.meta = np.hstack(self.meta).tolist()
        self.indices = list(range(len(self.meta)))
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.indices)
            
    def _get_mmap(self, path):
        if not hasattr(self.threadlocal, "handles"):
            with open(path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.threadlocal.handles = [f, mm]
                if path.endswith(".gz") or path.endswith(".bz") or path.endswith(".bz2"):
                    raise NotImplementedError(
                        "Compressed files are not supported because .seek() would require "
                        "rereading the entire file, making performance too slow."
                    )
        return self.threadlocal.handles[-1]
            
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
        meta = self.meta[self.indices[index]]
        file_name: str = meta["file"]
        if self.file is None or self.file_name != file_name.replace(".meta", "") or(isinstance(self.file, mmap.mmap) and self.file.closed):
            self.file_name = file_name.replace(".meta", "")
            if self.file is not None:
                self.file.close()
            self.file = self._get_mmap(self.file_name)
        self.file.seek(meta["offset"])
        return json.loads(self.file.readline().decode())
    
class CollieDatasetForTraining(Dataset):
    """ **CoLLie** 中的基本数据格式，可用于预训练、微调、生成任务。需提供的数据格式形似：

        .. code-block::
    
            [
                {
                    "text": "这是prompt部分的文本",
                },
                ...
            ]
        
        或者:

        .. code-block::

            [
                {
                    "input": "这是prompt部分的文本",
                    "output": "这是output部分的文本"
                },
                ...
            ]
        
        当使用第二种数据格式时，只有 `output` 部分的 token 会参与 loss计算。
    """
    def __init__(self,
                 dataset: Sequence[Dict],
                 tokenizer: Optional[PreTrainedTokenizer]=None,
                 add_special_tokens: bool=True,
                 shuffle: bool=False, 
                 seed: int = 1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.indices = list(range(len(self.dataset)))
        if shuffle:
            random.seed(seed)
            random.shuffle(self.indices)
        
    def __len__(self):
        return len(self.dataset)
    
    def _inspect_special_tokens_length(self):
        ids_with_special_tokens = self.tokenizer('a', add_special_tokens=True).input_ids
        ids_without_special_tokens = self.tokenizer('a', add_special_tokens=False).input_ids
        for bos_length in range(len(ids_with_special_tokens)):
            if ids_with_special_tokens[bos_length] == ids_without_special_tokens[0]:
                break
        bos_length += 1
        eos_length = len(ids_with_special_tokens) - len(ids_without_special_tokens) - bos_length
        return bos_length, eos_length
    
    def __getitem__(self, index) -> Tuple:
        if index > len(self):
            raise IndexError("Index out of range.")
        labels_mask = None
        index = self.indices[index]
        if self.tokenizer is None:
            input_ids = torch.tensor(self.dataset[index]["tokens"])
            if "labels_mask" in self.dataset[index].keys():
                labels_mask = torch.tensor(self.dataset[index]["labels_mask"])
            else:
                labels_mask = torch.tensor(self.dataset[index]["label_mask"]) if "label_mask" in self.dataset[index].keys() else None
        else:
            if "text" in self.dataset[0].keys():
                input_ids = self.tokenizer(self.dataset[index]["text"], add_special_tokens=self.add_special_tokens).input_ids
            elif "input" in self.dataset[0].keys() and "output" in self.dataset[0].keys():
                input_ids = self.tokenizer(self.dataset[index]["input"] + self.dataset[index]["output"], add_special_tokens=self.add_special_tokens).input_ids
                labels_mask = torch.ones_like(torch.tensor(input_ids))
                context_length = len(self.tokenizer(self.dataset[index]["input"], add_special_tokens=self.add_special_tokens).input_ids)
                _, eos_length = self._inspect_special_tokens_length()
                context_length -= eos_length
                labels_mask[context_length - 1:] = 0
                labels_mask = labels_mask.cpu().tolist()
            else:
                raise ValueError("Dataset must have one or two fields.")
        if labels_mask is None:
            return input_ids, {
                "labels": input_ids
            }
        else:
            return input_ids, {
                "labels": input_ids,
                "labels_mask": labels_mask
            }
    
    @classmethod
    def from_json(cls, 
                  path: str, 
                  tokenizer: Optional[PreTrainedTokenizer]=None,
                  shuffle: bool=False, 
                  seed: int = 1024):
        dataset = cls(dataset=json.loads(FileIODriver.load(path, mode="r")), shuffle=shuffle, seed=seed, tokenizer=tokenizer)
        return dataset
        
    @classmethod
    def from_processed(cls, 
                       path: str, 
                       shuffle: bool=False, 
                       seed: int = 1024):
        dataset = cls(dataset=_ShardContainer(path), shuffle=shuffle, seed=seed)
        return dataset
    
    def save_propressed(self, path: str, shard_size: int=4):
        shard = io.BytesIO()
        shard_idx = 0
        meta = np.empty((0, 2), int)
        for i in self.indices:
            labels = self[i][1]
            data = {
                "tokens": labels["labels"]
            }
            data.update({key: value for key, value in labels.items() if key != "labels"})
            bytes_data = json.dumps(data).encode() + "\n".encode()
            offset = shard.tell()
            length = len(data["tokens"])
            shard.write(bytes_data)
            meta = np.append(meta, np.array([[offset, length]]), axis=0)
            if shard.tell() > shard_size * 1024 * 1024 or i == len(self) - 1:
                file = f"collie-dataset-shard-{shard_idx}.bin"
                shard.seek(0)
                FileIODriver.save(meta, os.path.join(path, file + ".meta"))
                FileIODriver.save(shard.read().decode(), os.path.join(path, file))
                shard.close()
                shard = io.BytesIO()
                meta = np.empty((0, 2), int)
                shard_idx += 1
                
class CollieDatasetForClassification(CollieDatasetForTraining):
    """ **CoLLie** 中的分类任务数据集，须搭配 :class:`~collie.controller.evaluator.ClassficationEvaluator` 
        使用。需提供的数据格式形似:
    
        .. code-block::

            [
                {
                    "input": "这是prompt部分的文本",
                    "output": ["类别1", "类别2", "类别3"],
                    "target": 0
                },
                ...
            ]
    """
    def __getitem__(self, index) -> Tuple:
        if index > len(self):
            raise IndexError("Index out of range.")
        index = self.indices[index]
        if self.tokenizer is None:
            input_ids = tuple(self.dataset[index]["tokens"])
            target = self.dataset[index]["target"]
        else:
            if "input" in self.dataset[0].keys() and "output" in self.dataset[0].keys() and "target" in self.dataset[0].keys():
                input_ids = tuple([self.tokenizer(self.dataset[index]["input"] + output, add_special_tokens=self.add_special_tokens).input_ids for output in self.dataset[index]["output"]])
                target = self.dataset[index]["target"]
            else:
                raise ValueError("CollieDatasetForClassification must have three fields (`input`, `output` and `target`).")
        return input_ids, {
            "labels": input_ids,
            "target": target
        }