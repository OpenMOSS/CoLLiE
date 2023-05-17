import sys
sys.path.append("/mnt/lustre/zhangshuo/projects/collie/")
from collie.models.llama.model import LlamaModel, LlamaArguments
from collie.trainer.trainer import Trainer
from collie.metrics.decode import DecodeMetric
from collie.optim.inplace_sgd import InplaceSGD

from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig
from torch.utils.data import Dataset
import torch

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", 
                                           padding_side="left", 
                                           add_eos_token=True)
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
args = LlamaArguments.from_pretrained("decapoda-research/llama-7b-hf")
args.tp_size = 8
args.train_epochs = 10
args.train_micro_batch_size = 1
args.eval_batch_size = 1
# args.eval_per_n_steps = 2
args.ds_config = {
    "fp16": {"enabled": True},
    # "optimizer": {
    #     "type": "Adam",
    #     "params": {
    #         "lr": 2e-5
    #     }
    # }
}
model = LlamaModel(args)
# state_dict = LlamaModel.load_parallel_state_dict(
#     path="hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf",
#     args=args,
#     protocol="petrel",
#     format="hf"
# )
# model.load_state_dict(state_dict)
optimizer = InplaceSGD(model, lr=2e-5)
train_sample = tokenizer("Collie is a python package for finetuning large language models.", return_tensors="pt").input_ids.squeeze(0)
eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)[:-1,]
train_dataset = [(train_sample, train_sample) for _ in range(100)]
eval_dataset = [(eval_sample, eval_sample)]
trainer = Trainer(
    model = model,
    optimizer=optimizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    eval_config=GenerationConfig(max_new_tokens=128, 
                                 eos_token_id=2, 
                                 pad_token_id=0, 
                                 bos_token_id=1,
                                 use_cache=False),
    metrics=[DecodeMetric(tokenizer=tokenizer)],
    args=args
)
trainer.train()