import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig

from collie.models.mistral2 import MistralForCausalLM, MistralConfig
from collie import CollieConfig, env

model_name_or_path = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

config = CollieConfig.from_pretrained(model_name_or_path)

config.dp_size = 1
config.tp_size = 2
config.pp_size = 2
# config.architectures = ["MistralForCausalLM"]
print("------------------------------")
model = MistralForCausalLM.from_pretrained(model_name_or_path, config=config).cuda()
model.eval()
print("------------------------------")
prompt = "Llama is a"
# prompt = "Q:What do we eat for tonight?A:"
inputs = tokenizer(prompt, return_tensors="pt")
print("inputs:")
print(inputs)


gen_config = GenerationConfig(max_new_tokens=256, early_stopping=False, eos_token_id=2)

outs = model.generate(inputs["input_ids"].cuda(), generation_config=gen_config)
if env.local_rank == 0:
    print("outs:")
    print(outs)
    print("last:")
    print(tokenizer.decode(outs[0], skip_special_tokens=True))