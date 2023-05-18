import sys

sys.path.append("../..")
print(sys.path)

from transformers import LlamaTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.generation.utils import GenerationConfig
from peft import get_peft_model, LoraConfig, TaskType

from collie.utils import setup_distributation
from collie.config import CollieConfig
from collie.trainer.trainer import Trainer
from collie.metrics.decode import DecodeMetric

model_path = "/home/ubuntu/projects/collie/cache/llama-7b"
args = CollieConfig.from_pretrained(model_path)
print("Args loaded")
args.pp_size = 1
args.tp_size = 1
# args.dp_size = 8
args.train_epochs = 1000
args.train_micro_batch_size = 1
args.eval_batch_size = 1
args.eval_per_n_steps = 0
args.ds_config = {
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
setup_distributation(args)
dschf = HfDeepSpeedConfig(args.ds_config)  # keep this object alive

# config = AutoConfig.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_config(config)
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
    eval_config=GenerationConfig(max_new_tokens=128,
                                 eos_token_id=2,
                                 pad_token_id=0,
                                 bos_token_id=1,
                                 use_cache=False),
    metrics=[DecodeMetric(tokenizer=tokenizer)],
    config=args
)
trainer.train()
