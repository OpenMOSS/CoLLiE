# 使用 collie 模型进行张量并行模型推理
import sys
sys.path.append("/mnt/lustre/zhangshuo/projects/collie/")
from collie.models.llama.model import LlamaForCausalLM
from collie.config import CollieConfig

from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", 
                                           padding_side="left")
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2

args = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf")
args.dp_size = 1
args.pp_size = 1
args.tp_size = 4
model = LlamaForCausalLM.from_pretrained("/mnt/lustre/zhangshuo/model/test/", args=args).half().cuda()
input_ids = tokenizer("It's a beautiful day to", return_tensors="pt").input_ids.cuda()
input_ids = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=100,
                                                                                   eos_token_id=2))
print(tokenizer.decode(input_ids[0]))
