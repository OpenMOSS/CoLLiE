import sys
import torch
sys.path.append("..")

from collie import Server, LlamaForCausalLM, DashProvider, CollieConfig
from transformers import LlamaTokenizer, GenerationConfig

config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf", trust_remote_code=True)
config.pp_size = 1
config.tp_size = 1
model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_13b", config=config).cuda()
tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_13b", add_eos_token=False)
data_provider = DashProvider(tokenizer=tokenizer)
data_provider.generation_config = GenerationConfig(max_new_tokens=250)
server = Server(model, data_provider)
server.run()