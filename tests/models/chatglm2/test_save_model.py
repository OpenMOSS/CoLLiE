import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig, AutoModel
from collie.models import ChatGLM2ForCausalLM, LlamaForCausalLM
from collie import  CollieConfig, env

pretrained_path = "THUDM/chatglm2-6b"

# tokenizer = AutoTokenizer.from_pretrained(
#         pretrained_path,
#         trust_remote_code=True,
#     )
config = CollieConfig.from_pretrained(pretrained_path,
        trust_remote_code=True)
config.tp_size = 1
config.pp_size = 8
model = ChatGLM2ForCausalLM.from_pretrained(pretrained_path, config=config).cuda()
model.save_parallel_state_dict(model.state_dict(), "./dev", config=config)


# model = AutoModel.from_pretrained("./dev", trust_remote_code=True)
