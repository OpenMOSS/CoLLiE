from collie import CollieModelForCausalLM, BaseProvider, env, \
    broadcast_tensor, _GenerationStreamer, PipelineGenerationMixin, CollieConfig, setup_ds_engine
from deepspeed.runtime.pipe.engine import PipelineEngine
from transformers import PreTrainedModel
import torch.distributed as dist
import torch

from typing import Union

class Server:
    def __init__(self,
                 model: Union[CollieModelForCausalLM, PreTrainedModel],
                 data_provider: BaseProvider,
                 config: CollieConfig) -> None:
        self.model = model
        self.data_provider = data_provider
        self.collie_config = config
        
    def start(self):
        if env.rank == 0:
            self.data_provider.start_provider()
        
    def data_provider_handler(self):
        if self.data_provider is None:
            return None
        has_data = torch.tensor(False).cuda()
        input_ids = None
        if dist.get_rank() == 0:
            input_ids = self.data_provider.get_data()
            if input_ids is not None:
                has_data = ~has_data
                input_ids = input_ids.cuda()
        dist.broadcast(has_data, 0)
        if not has_data:
            return
        input_ids = broadcast_tensor(input_ids, src=0)
        generation_model = self.model
        if not generation_model.can_generate():
            return
        use_stream = self.data_provider.stream
        streamer = _GenerationStreamer(server=self.data_provider)
        generated_ids = generation_model.generate(
            input_ids=input_ids.cuda(), 
            attention_mask=torch.ones_like(input_ids).cuda(), 
            generation_config=self.data_provider.generation_config,
            streamer=streamer if use_stream else None
        )
        if not use_stream:
            self.data_provider.put_feedback(generated_ids[0].cpu())
    
    def run(self):
        self.start()
        while True:
            self.data_provider_handler()
    