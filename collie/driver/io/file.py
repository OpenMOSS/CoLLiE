from collie.driver.io.base import IODriver

import os
import torch
import shutil

class FileIODriver(IODriver):
    @staticmethod
    def load(path: str, mode: str):
        assert os.path.exists(path), f"File {path} does not exist."
        if 'b' in mode.lower():
            return torch.load(path, map_location=torch.device('cpu'))
        else:
            with open(path, 'r') as f:
                return f.read()
            
    @staticmethod
    def save(obj, path: str, append: bool = False):
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        if isinstance(obj, str):
            if append:
                with open(path, 'a+') as f:
                    f.write(obj)
            else:
                with open(path, 'w+') as f:
                    f.write(obj)
        else:
            torch.save(obj, path)
            
    @staticmethod
    def exists(path: str) -> bool:
        return os.path.exists(path)
    
    @staticmethod
    def list(path: str):
        return os.listdir(path)
    
    @staticmethod
    def delete(path: str):
        shutil.rmtree(path)

    @staticmethod
    def makedirs(path: str, exist_ok: bool = False):
        os.makedirs(path, exist_ok=exist_ok)