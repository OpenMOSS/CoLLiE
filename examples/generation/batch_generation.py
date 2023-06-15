# 使用 collie 模型进行批量并行模型推理
import sys
sys.path.append("../..")
from collie.models.llama.model import LlamaForCausalLM
from collie.controller.evaluator import Evaluator
from collie.config import CollieConfig
from collie.metrics.decode import DecodeMetric
from collie.data.dataset import CollieDatasetForTraining

from transformers import LlamaTokenizer
from torch.utils.data import Dataset
from transformers.generation.utils import GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained("/mnt/petrelfs/zhangshuo/model/llama-7b-hf")

config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf")
config.dp_size = 1
config.pp_size = 1
config.tp_size = 8
config.eval_batch_size = 4
config.ds_config = {
    "fp16": {"enabled": True}
}


dataset = CollieDatasetForTraining([
    {"text": "It's a beautiful day to"},
    {"text": "This is my favorite"},
    {"text": "As we discussed yesterday"},
    {"text": "I'm going to"},
    {"text": "As well known to all"},
    {"text": "A"},
    {"text": "So this is the reason why"},
    {"text": "We have to"}
], tokenizer=tokenizer)
model = LlamaForCausalLM.from_pretrained("/mnt/petrelfs/zhangshuo/model/llama-7b-hf", config=config)

evaluator = Evaluator(model=model,
                  config=config,
                  tokenizer=tokenizer,
                  dataset=dataset,
                  generation_config=GenerationConfig(max_new_tokens=100),
                  metrics={"pred": DecodeMetric()})
evaluator.eval()
