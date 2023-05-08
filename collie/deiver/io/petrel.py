from collie.driver.io.base import IODriver

import os
import torch
from io import BytesIO

class PetrelIODriver(IODriver):
    @staticmethod
    def load(path: str, mode: str):
        from petrel_client.client import Client
        client = Client()
        obj = client.get(path)
        if 'b' in mode.lower():
            buffer = BytesIO()
            buffer.write(obj)
            buffer.seek(0)
            obj = torch.load(buffer, map_location=torch.device('cpu'))
            buffer.close()
            return obj
        else:
            return obj.decode()
            
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
        from petrel_client.client import Client
        client = Client()
        return client.contains(path)
    
    @staticmethod
    def list_dir(path: str):
        from petrel_client.client import Client
        client = Client()
        return list(client.list(path))
    
    @staticmethod
    def delete(path: str):
        from petrel_client.client import Client
        client = Client()
        client.delete(path)