import sys
import argparse
sys.path.append("../..")
print(sys.path)

from transformers import LlamaTokenizer, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.generation.utils import GenerationConfig

from collie.utils import setup_distribution
from collie.config import CollieConfig
from collie.trainer.trainer import Trainer
from collie.metrics.decode import DecodeMetric
from peft import get_peft_model, LoraConfig, TaskType, PrefixTuningConfig, PromptTuningConfig, PromptTuningInit, \
    PromptEncoderConfig

parser = argparse.ArgumentParser()
parser.add_argument("--peft_type", type=str, default="lora")
args = parser.parse_args()

model_path = "facebook/opt-125m"
config = CollieConfig.from_pretrained(model_path)
print("config loaded")
config.pp_size = 1
config.tp_size = 1
config.dp_size = 2
config.train_epochs = 1000
config.train_micro_batch_size = 1
config.eval_batch_size = 1
config.eval_per_n_steps = 20000
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

if args.peft_type == "lora":
    config.peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
elif args.peft_type == "prefix-tuning":
    # Prefix-tuning doesn't support gradient checkpointing.
    config.peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
    )
elif args.peft_type == "prompt-tuning":
    config.peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=16,
        tokenizer_name_or_path=model_path,
        prompt_tuning_init_text="Introduce collie:",
    )
elif args.peft_type == "p-tuning":
    config.peft_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        encoder_hidden_size=128
    )
else:
    raise ValueError(f"Unknown PEFT type: {args.peft_type}")

setup_distribution(config)
# dschf = HfDeepSpeedConfig(config.ds_config)  # keep this object alive
model = AutoModelForCausalLM.from_pretrained(model_path)
# model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, config.peft_config)
model.print_trainable_parameters()
print("Model loaded")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False
)
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
    config=config
)
trainer.train()
