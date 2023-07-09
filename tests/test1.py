from transformers import AutoTokenizer, AutoModel, GenerationConfig
import torch

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()
inputs = tokenizer("[Round {}]\n问：你是谁？怎么改善睡眠？\n答：", return_tensors="pt")
config = GenerationConfig(max_new_tokens=200, eos_token_id=2)
print(inputs)
model.eval()
gen_inp = model.generate(inputs=inputs.input_ids.cuda(),
                                      attention_mask=inputs.attention_mask.cuda(),
                                      generation_config=config)
print(gen_inp)
print(tokenizer.decode(gen_inp[0]))