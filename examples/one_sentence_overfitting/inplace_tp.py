import sys

sys.path.append("../..")
print(sys.path)

from transformers import LlamaTokenizer, AutoModelForCausalLM
from collie.models.llama.model import LlamaForCausalLM
from transformers.generation.utils import GenerationConfig

from collie.utils import setup_distribution
from collie.config import CollieConfig
from collie.trainer.trainer import Trainer
from collie.metrics.decode import DecodeMetric
from collie.optim import InplaceSGD

###############################
#         init config         #
###############################
model_path = "/home/ubuntu/projects/collie/cache/llama-7b"
config = CollieConfig.from_pretrained(model_path)
print("config loaded")
config.pp_size = 1
config.tp_size = 2
config.dp_size = 1
config.train_epochs = 1000
config.train_micro_batch_size = 1
config.eval_batch_size = 1
config.eval_per_n_steps = 10
config.use_flash = False
config.ds_config = {
    "fp16": {"enabled": True},
    "train_micro_batch_size_per_gpu": 1,
    "zero_allow_untested_optimizer": True,
    "zero_force_ds_cpu_optimizer": False,
}

###############################
#     setup distribution      #
###############################
setup_distribution(config)

###############################
#         init model          #
###############################
model = LlamaForCausalLM.from_pretrained(model_path, config)
print("Model loaded")

###############################
#        init optim           #
###############################
optimizer = InplaceSGD(
    model,
    lr=0.001,
    zero_enabled=False,
    clip_grad_value=1.0,
    # clip_grad_norm=5.0,
)

###############################
#        init dataset         #
###############################
tokenizer = LlamaTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    padding_side="left",
    add_eos_token=True
)
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
print("Tokenizer loaded")
train_sample = tokenizer("Collie is a python package for finetuning large language models.",
                         return_tensors="pt").input_ids.squeeze(0)
eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)[:-1, ]
train_dataset = [(train_sample, train_sample) for _ in range(100)]
eval_dataset = [(eval_sample, eval_sample)]

###############################
#        init trainer         #
###############################
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    generation_config=GenerationConfig(max_new_tokens=128,
                                 eos_token_id=2,
                                 pad_token_id=0,
                                 bos_token_id=1,
                                 use_cache=False),
    metrics=[DecodeMetric(tokenizer=tokenizer)],
    config=config
)
trainer.train()
