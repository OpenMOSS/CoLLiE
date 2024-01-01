import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig, AutoModel
from collie.models import ChatGLM2ForCausalLM, LlamaForCausalLM
from collie import CollieConfig, env, Trainer

pretrained_path = "huggyllama/llama-7b"

# tokenizer = AutoTokenizer.from_pretrained(
#         pretrained_path,
#         trust_remote_code=True,
#     )
config = CollieConfig.from_pretrained(pretrained_path,
        trust_remote_code=True)
config.tp_size = 2
config.pp_size = 4
model = LlamaForCausalLM.from_pretrained(pretrained_path, config=config).cuda()

trainer = Trainer(model, config=config)
trainer.save_model("./dev")

# model.save_parallel_state_dict(model.state_dict(), "./dev", config=config)

print("start loading")
model = AutoModel.from_pretrained("./dev", trust_remote_code=True)
