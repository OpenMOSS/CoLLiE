import sys

import torch

sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig

from collie.models import InternLM2ForCausalLM
from collie import CollieConfig, env

model_name_or_path = "internlm/internlm2-7b"

tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
config = CollieConfig.from_pretrained(model_name_or_path,
        trust_remote_code=True)
config.dp_size = 1
config.tp_size = 1
config.pp_size = 2
# config.use_flash = False
model = InternLM2ForCausalLM.from_pretrained(model_name_or_path, config=config).cuda()
# model = model.to(torch.float32)
# model.set_cache(False)
prompt = "Llama is a"
inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)
model.eval()
gen_config = GenerationConfig(max_new_tokens=128, early_stopping=True, eos_token_id=2)

outs = model.generate(inputs["input_ids"].cuda(), generation_config=gen_config)
if env.local_rank == 0:
    print(outs)
    print(tokenizer.decode(outs[0], skip_special_tokens=True))