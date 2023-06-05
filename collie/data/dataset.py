from torch.utils.data import Dataset
import torch
from transformers import PreTrainedTokenizer
from huggingface_hub import snapshot_download
from torch.utils.data.dataset import ConcatDataset, Dataset
from collie.driver.io import FileIODriver, PetrelIODriver

import os
import io
import json
import mmap
import math
import random
import threading
import numpy as np
from functools import reduce
from typing import Optional, Dict, List, Sequence, Union

class _ShardContainer(list):
    def __init__(self, path, shuffle: bool=False, seed: int = 1024, protocol: str="file") -> None:
        list.__init__([])
        self.file = None
        self.file_name = None
        self.shuffle = shuffle
        self.seed = seed
        self.threadlocal = threading.local()
        self.protocol = protocol
        
        assert protocol in ["file", "petrel"], f"Only support file and petrel protocol, not `{protocol}`."
        self.IODriver = FileIODriver if protocol == 'file' else PetrelIODriver
        meta_files = []
        for file in self.IODriver.list(path):
            if file.endswith(".meta"):
                meta_files.append(os.path.join(path, file))
        self.meta = [[{"file": file, "offset": meta[0], "length":  meta[1]} for meta in self.IODriver.load(file, mode="rb")] for file in meta_files]
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
            if self.protocol == "petrel":
                self.file = self.IODriver.load_buffer(self.file_name)
            elif self.protocol == "file":
                self.file = self._get_mmap(self.file_name)
        self.file.seek(meta["offset"])
        return self.tensorify(json.loads(self.file.readline().decode()))
    
    @property
    def lengths(self):
        return [meta["length"] for meta in self.meta]
    
    @staticmethod
    def tensorify(sample: dict):
        result = {}
        for key, value in sample.items():
            try:
                result[key] = torch.tensor(value)
            except TypeError:
                pass
        return result
        
class CollieDataset(Dataset):
    def __init__(self, 
                 path: str, 
                 packed_batch_size: int = 1,
                 packed_drop_last: bool = False,
                 shuffle: bool = False, 
                 seed: int = 1024, 
                 suffix: Optional[str] = None,
                 protocol: str="file") -> None:
        super().__init__()
        self.path = path
        self.packed_batch_size = packed_batch_size
        self.packed_drop_last = packed_drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.suffix = suffix
        self.data = []
        assert protocol in ["file", "petrel"], f"Only support file and petrel protocol, not `{self.protocol}`."
        self.IODriver = FileIODriver if protocol == 'file' else PetrelIODriver
        self._load_all()
        
    @property
    def length(self):
        if isinstance(self.data, list):
            return [len(sample["token"]) for sample in self.data]        
        elif isinstance(self.data, _ShardContainer):
            return self.data.lengths

    def load(self, file) -> List[Dict]:
        return [self.tensorify(sample) for sample in json.loads(self.IODriver.load(file, mode="r"))] 
    
    @staticmethod
    def tensorify(sample: dict):
        result = {}
        for key, value in sample.items():
            try:
                result[key] = torch.tensor(value)
            except TypeError:
                pass
        return result
        
    def _load_all(self):
        for file in self.IODriver.walk(self.path, self.suffix):
            data_in_file = self.load(os.path.join(self.path, file))
            assert all("tokens" in sample.keys() for sample in data_in_file), "All samples must have `tokens` field."
            assert all(isinstance(value, torch.Tensor) for sample in data_in_file for value in sample.values()), "All values must be torch.Tensor."
            self.data.extend(data_in_file )
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.data)

    def __len__(self):
        if self.packed_drop_last:
            return math.floor(float(len(self.data)) / float(self.packed_batch_size))
        else:
            return math.ceil(float(len(self.data)) / float(self.packed_batch_size))
    
    def __getitem__(self, index):
        if index > len(self):
            raise IndexError("Index out of range.")
        if self.packed_batch_size == 1:
            return self.data[index]["tokens"], (self.data[index]["tokens"], *[value for key, value in self.data[index].items() if key != "tokens"])
        else:
            index = index * self.packed_batch_size
            if index + self.packed_batch_size > len(self.data):
                packed_batch_size = len(self.data) - index
            else:
                packed_batch_size = self.packed_batch_size
            tokens = reduce(lambda x, y: torch.concat((x, y), dim=-1), [self.data[i]["tokens"] for i in range(index, index + packed_batch_size)])
            cu_seq_len = torch.cumsum(torch.tensor([len(self.data[i]["tokens"]) for i in range(index, index + packed_batch_size)]), dim=0)
            extra_labels = []
            for key in self.data[index].keys():
                if key != "tokens":
                    extra_labels.append(reduce(lambda x, y: torch.concat((x, y), dim=-1), [self.data[i][key] for i in range(index, index + packed_batch_size)]))
            return (tokens, cu_seq_len), (tokens, *extra_labels)
    
    def filter(self, func):
        self.data = [sample for sample in self.data if func(sample)]
        return self
    
    def apply(self, func):
        self.data = [func(sample) for sample in self.data]
        return self
    
    def split(self, *args):
        """ 以给定的比例分割数据集
        """
        return [self.data[int(sum(args[:i]) / sum(args) * len(self.data)):int(sum(args[:i+1]) / sum(args) * len(self.data))] for i in range(len(args))]
    
    def save_propressed(self, path: str, shard_size: int=4, field: list = ["tokens"], protocol: str="file"):
        assert protocol in ["file", "petrel"], f"Only support file and petrel protocol, not `{protocol}`."
        IODriver = FileIODriver if protocol == 'file' else PetrelIODriver
        shard = io.BytesIO()
        shard_idx = 0
        meta = np.empty((0, 2), int)
        for i in range(len(self.data)):
            data = self.data[i]
            assert all(isinstance(value, torch.Tensor) for value in data.values()), "All values must be torch.Tensor."
            data = {key: value.cpu().tolist() for key, value in data.items() if key in field}
            bytes_data = json.dumps(data).encode() + "\n".encode()
            offset = shard.tell()
            length = len(data["tokens"])
            shard.write(bytes_data)
            meta = np.append(meta, np.array([[offset, length]]), axis=0)
            if shard.tell() > shard_size * 1024 * 1024 or i == len(self.data) - 1:
                file = f"collie-dataset-shard-{shard_idx}.bin"
                shard.seek(0)
                IODriver.save(meta, os.path.join(path, file + ".meta"))
                IODriver.save(shard.read().decode(), os.path.join(path, file))
                shard.close()
                shard = io.BytesIO()
                meta = np.empty((0, 2), int)
                shard_idx += 1
            
            
    
    @classmethod
    def from_processed(cls, 
                       path: str, 
                       packed_batch_size: int = 1,
                       packed_drop_last: bool = False,
                       shuffle: bool=False, 
                       seed: int = 1024, 
                       protocol: str="file"):
        assert protocol in ["file", "petrel"], f"Only support file and petrel protocol, not `{protocol}`."
        dataset = cls(path, shuffle=shuffle, seed=seed, protocol=protocol, packed_drop_last=packed_drop_last, packed_batch_size=packed_batch_size, suffix=".collie_donot_load_file")
        dataset.data = _ShardContainer(path, shuffle=shuffle, seed=seed, protocol=protocol)
        return dataset