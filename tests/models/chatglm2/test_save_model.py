import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig, AutoModel
from collie.models import ChatGLM2ForCausalLM, LlamaForCausalLM
from collie import CollieConfig, env, Trainer

pretrained_path = "THUDM/chatglm2-6b"

# tokenizer = AutoTokenizer.from_pretrained(
#         pretrained_path,
#         trust_remote_code=True,
#     )
config = CollieConfig.from_pretrained(pretrained_path,
        trust_remote_code=True)
config.tp_size = 1
config.pp_size = 2
model = ChatGLM2ForCausalLM.from_pretrained(pretrained_path, config=config)
trainer = Trainer(model, config=config)
trainer.save_model("./dev")
print("start loading")
model = AutoModel.from_pretrained("./dev", trust_remote_code=True)
