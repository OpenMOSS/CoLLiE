import sys
sys.path.append("../..")
from collie.models.llama.model import LlamaForCausalLM
from collie.trainer.trainer import Trainer
from collie.metrics.decode import DecodeMetric
from collie.config import CollieConfig
from collie.utils import GradioProvider
from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig
from torch.utils.data import Dataset
import torch

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", 
                                           padding_side="left")
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf")
config.tp_size = 4
config.dp_size = 1
config.pp_size = 2
config.train_epochs = 1000
config.train_micro_batch_size = 8
config.eval_batch_size = 1
# config.eval_per_n_steps = 20
config.ds_config = {
    "fp16": {"enabled": True},
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 2e-5
        }
    }
}
model = LlamaForCausalLM(config)
state_dict = LlamaForCausalLM.load_parallel_state_dict(
    path="hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf",
    config=config,
    protocol="petrel",
    format="hf"
)
model.load_state_dict(state_dict)
train_sample = torch.concat((tokenizer("Collie is a python package for finetuning large language models.", return_tensors="pt").input_ids.squeeze(0), torch.tensor([2])), dim=0)
eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)
train_dataset = [(train_sample, train_sample) for _ in range(1000)]
eval_dataset = [(eval_sample, eval_sample)]
server = GradioProvider(tokenizer=tokenizer, stream=True, port=8080)
trainer = Trainer(
    model = model,
    config=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_provider=server,
    generation_config=GenerationConfig(max_new_tokens=128, 
                                 eos_token_id=2, 
                                 pad_token_id=0, 
                                 bos_token_id=1,
                                 use_cache=False),
    metrics=[DecodeMetric(tokenizer=tokenizer)],
)
trainer.train()