import sys

import torch

sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from collie.models.internlm2.model_hf import InternLM2ForCausalLM
# from collie import CollieConfig, env

model_name_or_path = "internlm/internlm2-7b"

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
)
model = InternLM2ForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, ).cuda()
model.eval()
prompt = "Llama is a"
inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)
gen_config = GenerationConfig(max_new_tokens=2, early_stopping=True, eos_token_id=2)

outs = model.generate(inputs["input_ids"].cuda(), generation_config=gen_config)
# if env.local_rank == 0:
print(outs)
print(tokenizer.decode(outs[0], skip_special_tokens=True))

'''
Llama is a 3D game engine that is designed to be used by game developers to create games. It is a free and open source project that is developed by the Llama team. The engine is written in C++ and uses OpenGL for rendering. Llama is a cross-platform engine that can be used on Windows, Linux, and Mac OS X.
Llama is a 3D game engine that is designed to be used by game developers to create games. It is a free and open source project that is developed by the Llama team. The engine is written in C++ and uses OpenGL for rendering. Llama is a cross-platform engine that can be used on Windows, Linux, and Mac OS X.
Llama is a 3D game engine that is designed to be used by game developers to create games. It is a free and open source project that is developed by the Llama team. The engine is written in C++ and uses OpenGL for rendering. Llama is a cross-platform engine that can be used on Windows, Linux, and Mac OS X.
Llama is a 3D game engine that is designed to be used by game developers to create games. It is a free and open source project that is
'''