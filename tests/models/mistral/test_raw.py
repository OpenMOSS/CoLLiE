import sys

import torch

sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from collie.models.mistral.modeling_mistral import MistralForCausalLM

model_name_or_path = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = MistralForCausalLM.from_pretrained(model_name_or_path).cuda()
model.eval()
prompt = "Llama is a"
# prompt = "Q:What do we eat for tonight?A:"
inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)
gen_config = GenerationConfig(max_new_tokens=256, early_stopping=False, eos_token_id=2)
outs = model.generate(inputs["input_ids"].cuda(), generation_config=gen_config)

print(outs)
print(tokenizer.decode(outs[0], skip_special_tokens=True))