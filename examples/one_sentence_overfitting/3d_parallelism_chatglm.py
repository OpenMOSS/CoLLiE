import sys
sys.path.append("../..")
from collie.models.chatglm.model import ChatGLMForCausalLM
from collie.trainer.trainer import Trainer
from collie.metrics.decode import DecodeMetric
from collie.config import CollieConfig
from collie.utils import GradioProvider, setup_distribution
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationConfig
from torch.utils.data import Dataset
import torch
import os

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", 
                                           padding_side="left", 
                                           trust_remote_code=True)
config = CollieConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
config.tp_size = 2
config.dp_size = 1
config.pp_size = 2
config.train_epochs = 1000
config.train_micro_batch_size = 32
config.eval_batch_size = 1
config.eval_per_n_steps = 20
config.ds_config = {
    "fp16": {"enabled": True},
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 2e-5
        }
    }
}

model = ChatGLMForCausalLM.from_pretrained("/mnt/petrelfs/zhangshuo/model/chatglm-6b", config=config)
train_sample = torch.cat((tokenizer("[Round {0}]\n问：大语言模型中moss和chatglm哪个比较厉害\n答：", return_tensors="pt").input_ids, tokenizer("当然是moss了<eop>", return_tensors="pt", add_special_tokens=False).input_ids), dim=-1).squeeze(0)
eval_sample = tokenizer("[Round {0}]\n问：大语言模型中moss和chatglm哪个比较厉害\n答：", return_tensors="pt").input_ids.squeeze(0)
train_dataset = [(train_sample, (train_sample, train_sample)) for _ in range(128000)]
eval_dataset = [(eval_sample, eval_sample)]
trainer = Trainer(
    model = model,
    config=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    generation_config=GenerationConfig(max_new_tokens=128, 
                                 eos_token_id=tokenizer.eos_token_id, 
                                 use_cache=False),
    metrics={
        "decode": DecodeMetric(tokenizer=tokenizer)},
)
trainer.train()