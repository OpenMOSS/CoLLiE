import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig

from collie.models import LlamaForCausalLM
from collie import  CollieConfig, env

tokenizer = AutoTokenizer.from_pretrained(
        "huggyllama/llama-7b",
        trust_remote_code=True,
    )
config = CollieConfig.from_pretrained("huggyllama/llama-7b",
        trust_remote_code=True)
config.dp_size = 1
config.pp_size = 4
model = LlamaForCausalLM.from_pretrained("huggyllama/llama-7b", config=config).cuda()
prompt = "Llama is a"
inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)
model.eval()
gen_config = GenerationConfig(max_new_tokens=48, early_stopping=True, eos_token_id=2)

outs = model.generate(inputs["input_ids"].cuda(), generation_config=gen_config)
if env.local_rank == 0:
    print(outs)
    print(tokenizer.decode(outs[0], skip_special_tokens=True))