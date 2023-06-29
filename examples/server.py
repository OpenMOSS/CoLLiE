import sys
sys.path.append("../..")
from collie import Server, LlamaForCausalLM, DashProvider, CollieConfig
from transformers import LlamaTokenizer, GenerationConfig

config = CollieConfig.from_pretrained("/mnt/petrelfs/zhangshuo/model/llama-7b-hf")
config.pp_size = 2
config.tp_size = 1
model = LlamaForCausalLM.from_pretrained(
    "/mnt/petrelfs/zhangshuo/model/llama-7b-hf", config=config).half().cuda()
tokenizer = LlamaTokenizer.from_pretrained(
    "/mnt/petrelfs/zhangshuo/model/llama-7b-hf", add_eos_token=False)
data_provider = DashProvider(tokenizer=tokenizer)
data_provider.generation_config = GenerationConfig(max_new_tokens=250)
server = Server(model, data_provider, config=config)
server.run()