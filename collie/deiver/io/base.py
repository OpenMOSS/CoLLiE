from abc import ABC, abstractmethod

class IODriver(ABC):
    @staticmethod
    @abstractmethod
    def load(path: str, mode: str):
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def save(obj, path: str):
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
    def delete(path: str):
        raise NotImplementedError