import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig

from collie.models import LlamaForCausalLM
from collie import CollieConfig, env

model_name_or_path = "huggyllama/llama-7b"

tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
config = CollieConfig.from_pretrained(model_name_or_path,
        trust_remote_code=True)
config.dp_size = 1
config.tp_size = 2
config.pp_size = 1
model = LlamaForCausalLM.from_pretrained(model_name_or_path, config=config).cuda()
model.eval()
prompt = "[Round 0]\n\n\n\n问：你是谁？\n\n答："
inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)
gen_config = GenerationConfig(max_new_tokens=256, early_stopping=True, eos_token_id=0)

outs = model.generate(inputs["input_ids"].cuda(), generation_config=gen_config)
if env.local_rank == 0:
        print(outs)
        print(tokenizer.decode(outs[0], skip_special_tokens=True))