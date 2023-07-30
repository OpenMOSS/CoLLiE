import sys
import torch
sys.path.append("..")

from collie import Server, LlamaForCausalLM, CollieConfig, GradioProvider
from transformers import LlamaTokenizer, GenerationConfig

config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf", trust_remote_code=True)
config.pp_size = 1
config.tp_size = 1
model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf", config=config).cuda()
tokenizer = LlamaTokenizer.from_pretrained(
    "decapoda-research/llama-7b-hf", add_eos_token=False)
data_provider = GradioProvider(tokenizer=tokenizer, stream=True)
data_provider.generation_config = GenerationConfig(max_new_tokens=250)
server = Server(model, data_provider)
server.run()