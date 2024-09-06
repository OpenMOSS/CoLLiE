import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig

from collie.models import Qwen2ForCausalLM
from collie import CollieConfig, env

model_name_or_path = "Qwen/Qwen1.5-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

config = CollieConfig.from_pretrained(model_name_or_path)

config.dp_size = 1
config.tp_size = 2
config.pp_size = 1

model = Qwen2ForCausalLM.from_pretrained(model_name_or_path, config=config).cuda()
model.eval()

# prompt = "Write a response that appropriately completes the request: tell me a joke: "
prompt = "Llama is a"
inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)


gen_config = GenerationConfig(max_new_tokens=256, early_stopping=False, eos_token_id=0)

outs = model.generate(inputs["input_ids"].cuda(), generation_config=gen_config)
if env.local_rank == 0:
    print(outs)
    print(tokenizer.decode(outs[0], skip_special_tokens=True))