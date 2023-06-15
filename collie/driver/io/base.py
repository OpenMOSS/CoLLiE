from abc import ABC, abstractmethod
from typing import Optional
import io

class IODriver(ABC):
    @staticmethod
    @abstractmethod
    def load(path: str, mode: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_buffer(path: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def save(obj, path: str, append: bool = False):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def save_buffer(obj: io.BytesIO, path: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def exists(path: str) -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def list(path: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def walk(path: str, suffix: Optional[str]):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def delete(path: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def makedirs(path: str, exist_ok: bool = False):
        raise NotImplementedError

    @staticmethod
    def from_protocol(protocol):
        if protocol == "file":
            from .file import FileIODriver
            return FileIODriver
        elif protocol == "petrel":
            from .petrel import PetrelIODriver
            return PetrelIODriver
        else:
            raise ValueError(f"Only support file and petrel protocol, not `{protocol}`.")