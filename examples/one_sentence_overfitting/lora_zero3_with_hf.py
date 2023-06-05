import sys

sys.path.append("../..")
print(sys.path)

from transformers import LlamaTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.generation.utils import GenerationConfig
from peft import get_peft_model, LoraConfig, TaskType

from collie.utils import setup_distribution
from collie.config import CollieConfig
from collie.trainer.trainer import Trainer
from collie.metrics.decode import DecodeMetric

model_path = "/mnt/petrelfs/zhangshuo/model/llama-7b-hf/"
config = CollieConfig.from_pretrained(model_path)
print("config loaded")
config.pp_size = 1
config.tp_size = 1
config.dp_size = 8
config.train_epochs = 1000
config.train_micro_batch_size = 1
config.eval_batch_size = 1
config.eval_per_n_steps = 10
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
dschf = HfDeepSpeedConfig(config.ds_config)  # keep this object alive
model = AutoModelForCausalLM.from_pretrained(model_path)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
print("Model loaded")
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
