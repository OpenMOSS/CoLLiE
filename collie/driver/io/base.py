from abc import ABC, abstractmethod
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
    def walk(path: str):
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def delete(path: str):
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def makedirs(path: str, exist_ok: bool = False):
        raise NotImplementedError