from collie.driver.io.base import IODriver

import os
import torch
from io import BytesIO
from typing import Optional
from contextlib import contextmanager

@contextmanager
def no_proxy():
    backup = {}
    if "http_proxy" in os.environ:
        backup["http_proxy"] = os.environ["http_proxy"]
        del os.environ["http_proxy"]
    if "https_proxy" in os.environ:
        backup["https_proxy"] = os.environ["https_proxy"]
        del os.environ["https_proxy"]
    if "HTTP_PROXY" in os.environ:
        backup["HTTP_PROXY"] = os.environ["HTTP_PROXY"]
        del os.environ["HTTP_PROXY"]
    if "HTTPS_PROXY" in os.environ:
        backup["HTTPS_PROXY"] = os.environ["HTTPS_PROXY"]
        del os.environ["HTTPS_PROXY"]
    yield
    for key, value in backup.items():
        os.environ[key] = value

class PetrelIODriver(IODriver):
    @staticmethod
    def load(path: str, mode: str):
        with no_proxy():
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
    def load_buffer(path: str):
        with no_proxy():
            from petrel_client.client import Client
            client = Client()
            obj = client.get(path)
            buffer = BytesIO()
            buffer.write(obj)
            buffer.seek(0)
            return buffer

    @staticmethod
    def save(obj, path: str, append: bool = False):
        with no_proxy():
            from petrel_client.client import Client
            client = Client()
            buffer = BytesIO()
            if isinstance(obj, str):
                if append:
                    pre_obj = PetrelIODriver.load(path, 'r')
                    obj = pre_obj + obj
                buffer.write(obj.encode())
            else:
                torch.save(obj, buffer)
            buffer.seek(0)
            client.put(path, buffer)
            buffer.close()

    @staticmethod
    def exists(path: str) -> bool:
        with no_proxy():
            from petrel_client.client import Client
            client = Client()
            return client.contains(path) or client.isdir(path)

    @staticmethod
    def list(path: str):
        with no_proxy():
            from petrel_client.client import Client
            client = Client()
            return list(client.list(path))

    @staticmethod
    def walk(path: str, suffix: Optional[str]=None):
        with no_proxy():
            if not path.endswith("/"):
                path += "/"
            file_list = []
            dir_list = PetrelIODriver.list(path)
            for sub_path in dir_list:
                if sub_path.endswith("/"):
                    file_list += list(map(lambda x: sub_path + x, PetrelIODriver.walk(path + sub_path, suffix)))
                else:
                    if suffix is None or sub_path.endswith(suffix):
                        file_list.append(sub_path)
            return file_list

    @staticmethod
    def delete(path: str):
        with no_proxy():
            from petrel_client.client import Client
            client = Client()
            client.delete(path)

    @staticmethod
    def makedirs(path: str, exist_ok: bool = False):
        pass