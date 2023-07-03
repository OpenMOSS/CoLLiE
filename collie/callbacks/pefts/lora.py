from ..callback import Callback
from peft import get_peft_model

class LoRACallback(Callback):
    '''用于使用LoRA进行微调
    '''
    def __init__(self) -> None:
        super().__init__()
        
    def on_setup_parallel_model(self, trainer):
        peft_config = trainer.config.peft_config
        trainer.model = get_peft_model(trainer.model, peft_config)