from collie.driver.io.file import FileIODriver
from collie.driver.io.petrel import PetrelIODriver
from collie.models.llama.arguments import LlamaArguments

def load_parallel_state_dict(folder: str,
                             protocol: str = 'file',
                             format: str = 'hf',
                             args: LlamaArguments = LlamaArguments()
                             ):
    IODriver = FileIODriver if protocol == 'file' else PetrelIODriver

def save_parallel_state_dict(state_doct: dict,
                             folder: str,
                             protocol: str = 'file',
                             args: LlamaArguments = LlamaArguments()):
    IODriver = FileIODriver if protocol == 'file' else PetrelIODriver
    
def load_state_dict(folder: str,
                    protocol: str = 'file',
                    format: str = 'hf',
                    args: LlamaArguments = LlamaArguments()):
    IODriver = FileIODriver if protocol == 'file' else PetrelIODriver
    
def save_state_dict(state_doct: dict,
                    folder: str,
                    protocol: str = 'file',
                    args: LlamaArguments = LlamaArguments()):
    IODriver = FileIODriver if protocol == 'file' else PetrelIODriver