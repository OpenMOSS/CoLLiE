from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from torch.utils.data.dataset import ConcatDataset, Dataset

from collie.driver.io import FileIODriver, PetrelIODriver

import os
import io
import json
import random
import numpy as np
from typing import Optional, Callable, Dict
from abc import ABC, abstractmethod

class _ShardContainer(list):
    def __init__(self, path, shuffle: bool=False, seed: int = 1024, protocol: str="file") -> None:
        list.__init__([])
        self.file = None
        self.file_name = None
        self.shuffle = shuffle
        self.seed = seed
        
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
            
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
        meta = self.meta[self.indices[index]]
        if "obj" in meta.keys():
            return meta["obj"]
        else:
            file: str = meta["file"]
            if self.file is None or self.file_name != file:
                self.file_name = file.replace(".meta", "")
                if self.file is not None:
                    self.file.close()
                self.file = self.IODriver.load_buffer(self.file_name)
            self.file.seek(meta["offset"])
            return json.loads(self.file.read(meta["length"]).decode())
    
    def __add__(self, other):
        if isinstance(other, _ShardContainer):
            self.meta += other.meta
        else:
            self.meta += [{"obj": obj} for obj in other]
        self.indices = list(range(len(self.meta)))
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.indices)
        return self
        
class CollieDataset(Dataset, ABC):
    def __init__(self, 
                 path: str, 
                 tokenizer: PreTrainedTokenizer,
                 template: str = "{text}",
                 shuffle: bool = False, 
                 seed: int = 1024, 
                 suffix: Optional[str] = None,
                 protocol: str="file") -> None:
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.template = template
        self.shuffle = shuffle
        self.seed = seed
        self.protocol = protocol
        self.suffix = suffix
        self.data = []

    def load(self, file, protocol: str="file"):
        assert protocol in ["file", "petrel"], f"Only support file and petrel protocol, not `{self.protocol}`."
        IODriver = FileIODriver if protocol == 'file' else PetrelIODriver
        
    def _load_all(self):
        assert self.protocol in ["file", "petrel"], f"Only support file and petrel protocol, not `{self.protocol}`."
        self.IODriver = FileIODriver if self.protocol == 'file' else PetrelIODriver
        for file in self.IODriver.walk(self.path, self.suffix):
            self.data.extend(self.load(os.path.join(self.path, file), protocol=self.protocol))
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.data)
                
    def __add__(self, other):
        assert isinstance(other, CollieDataset), f"Only support adding two CollieDataset, not `{type(other)}`."
        if len(self.data) == 0:
            self._load_all()
        if len(other.data) == 0:
            other._load_all()
        self.data += other.data
        return self

    def __len__(self):
        if len(self.data) == 0:
            self._load_all()
        return len(self.data)
    
    def __getitem__(self, index):
        if len(self.data) == 0:
            self._load_all()
        return self.data[index]
    
    def save_propressed(self, path: str, shard_size: int="4", protocol: str="file"):
        assert protocol in ["file", "petrel"], f"Only support file and petrel protocol, not `{protocol}`."
        IODriver = FileIODriver if protocol == 'file' else PetrelIODriver
        shard = io.BytesIO()
        shard_idx = 0
        meta = np.array([])
        while len(self.data) != 0:
            data = self.data.pop(0)
            bytes_data = json.dumps(data).encode() + "\n".encode()
            offset = shard.tell()
            length = len(bytes_data)
            if shard.tell() + length > shard_size * 1024 * 1024:
                file = f"collie-dataset-shard-{shard_idx}.bin"
                shard.seek(0)
                IODriver.save(meta, os.path.join(path, file + ".meta"))
                IODriver.save_buffer(shard.read().decode(), os.path.join(path, file))
                shard = io.BytesIO()
                meta = np.array([])
            meta = np.append(meta, [offset, length])
    
    @classmethod
    def from_processed(cls, path: str, shuffle: bool=False, seed: int = 1024, protocol: str="file"):
        assert protocol in ["file", "petrel"], f"Only support file and petrel protocol, not `{protocol}`."
        dataset = cls(path, shuffle=shuffle, seed=seed, protocol=protocol)
        dataset.data = _ShardContainer(path, shuffle=shuffle, seed=seed, protocol=protocol)
        return dataset