from collie.driver.io.base import IODriver

import os
import io
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
    def load_buffer(path: str):
        assert os.path.exists(path), f"File {path} does not exist."
        buffer = io.BytesIO()
        with open(path, 'rb') as f:
            buffer.write(f.read())
            buffer.seek(0)
            return buffer
            
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
    def walk(path: str, suffix: str = None):
        if suffix is None:
            suffix = ""
        file_list = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(suffix):
                    file_list.append(os.path.join(root, f))

        return file_list
    
    @staticmethod
    def delete(path: str):
        shutil.rmtree(path)

    @staticmethod
    def makedirs(path: str, exist_ok: bool = False):
        os.makedirs(path, exist_ok=exist_ok)