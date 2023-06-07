import sys


sys.path.append("../..")
print(sys.path)

from transformers import LlamaTokenizer, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from peft import get_peft_model, LoraConfig, TaskType

from collie.utils import setup_distribution
from collie.config import CollieConfig
from collie.trainer.trainer import Trainer
from collie.metrics.decode import DecodeMetric
from collie.models.moss.model import MossForCausalLM

config = CollieConfig.from_pretrained("fnlp/moss-moon-003-sft", trust_remote_code=True)
print("config loaded")
config.pp_size = 1
config.tp_size = 1
config.dp_size = 4
config.train_epochs = 1000
config.checkpointing = False
config.train_micro_batch_size = 1
config.eval_batch_size = 1
config.eval_per_n_steps = 25
config.use_flash = False
config.ds_config = {
    "fp16": {"enabled": True},
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 2e-5
        }
    },
    "zero_optimization": {
        "stage": 3,
    },
    "train_micro_batch_size_per_gpu": 1,
    "zero_allow_untested_optimizer": True,
    "zero_force_ds_cpu_optimizer": False,
}
setup_distribution(config)
model = MossForCausalLM.from_pretrained("/mnt/lustre/zhangshuo/model/moss-moon-003-sft", config)
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["qkv_proj", "out_proj", "fc_in", "fc_out"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
print("Model loaded")
tokenizer = AutoTokenizer.from_pretrained(
    "fnlp/moss-moon-003-sft",
    local_files_only=True,
    padding_side="left",
    add_eos_token=True,
    # cache_dir="/home/ubuntu/projects/collie/cache/",
    trust_remote_code=True,
)
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
print("Tokenizer loaded")
train_sample = tokenizer("Collie is a python package for finetuning large language models.",
                         return_tensors="pt").input_ids.squeeze(0)
eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)[:-1, ]
train_dataset = [(train_sample, train_sample) for _ in range(100)]
eval_dataset = [(eval_sample, eval_sample)]
trainer = Trainer(
    model=model,
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
