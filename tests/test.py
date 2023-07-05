from transformers import AutoTokenizer, AutoModel, GenerationConfig
import torch

# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()
# inputs = tokenizer("[Round {}]\n问：你是谁？\n答：", return_tensors="pt")
# config = GenerationConfig(max_new_tokens=100, eos_token_id=2)
# print(inputs)
# gen_inp = model.generate(inputs=inputs.input_ids.cuda(),
#                                       attention_mask=inputs.attention_mask.cuda(),
#                                       generation_config=config)
# print(gen_inp)
# print(tokenizer.decode(gen_inp[0]))



import sys
sys.path.append("../")

from collie.models.chatglm2.model import ChatGLM2ForCausalLM
from collie import CollieConfig, setup_ds_engine, PipelineGenerationMixin, Server, DashProvider, env
from transformers import AutoTokenizer, AutoModel, GenerationConfig

config = CollieConfig.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
config.pp_size = 2
config.tp_size = 1
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

model = ChatGLM2ForCausalLM.from_pretrained("THUDM/chatglm2-6b", config=config).half().cuda()
inputs = tokenizer("[Round {}]\n问：你是谁？\n答：", return_tensors="pt")
print(inputs)
# engine, _, _, _ = setup_ds_engine(config, model)
# model = PipelineGenerationMixin(engine)
config = GenerationConfig(max_new_tokens=2, eos_token_id=2)
model.eval()
gen_inp = model.generate(inputs=inputs.input_ids.cuda(),
                                      attention_mask=inputs.attention_mask.cuda(),
                                      generation_config=config)
print(gen_inp)
if env.rank == 0:
    print(tokenizer.decode(gen_inp[0]))


# data_provider = DashProvider(tokenizer=tokenizer)
# data_provider.generation_config = GenerationConfig(max_new_tokens=250)
# server = Server(model, data_provider, config=config)
# server.run()
