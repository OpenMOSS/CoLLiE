import sys

sys.path.append("../..")
from collie.models.llama.model import LlamaForCausalLM
from collie.controller import Trainer, EvaluatorForGeneration
from collie.metrics.decode import DecodeMetric
from collie.config import CollieConfig
from collie import TGSMonitor, MemoryMonitor, LossMonitor, EvalMonitor, GradioProvider
from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig
import torch

tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b", padding_side="left", add_eos_token=False)
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
config = CollieConfig.from_pretrained("huggyllama/llama-7b")
config.tp_size = 4
config.dp_size = 1
config.pp_size = 2
config.train_epochs = 1000
config.train_micro_batch_size = 2
config.gradient_accumulation_steps = 1
config.use_flash = False
config.eval_batch_size = 1
config.eval_per_n_steps = 20
config.ds_config = {
    "fp16": {"enabled": True},
    # "data_types": {
    #     "grad_accum_dtype": "fp32"
    # }
}

model = LlamaForCausalLM.from_pretrained("huggyllama/llama-7b", config=config)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
train_sample = tokenizer("Collie is a python package for finetuning large language models.</s>",
                         return_tensors="pt").input_ids.squeeze(0)
eval_sample = tokenizer("Collie is", return_tensors="pt")
train_dataset = [{"input_ids": train_sample, "labels": train_sample} for _ in range(128000)]
eval_dataset = [{"input_ids": eval_sample.input_ids, "attention_mask": eval_sample.attention_mask}]

evaluator = EvaluatorForGeneration(
    model=model, dataset=eval_dataset, tokenizer=tokenizer, config=config,
    generation_config=GenerationConfig(
        max_new_tokens=128,
        eos_token_id=2,
        pad_token_id=0,
        bos_token_id=1,
        use_cache=False
    ), metrics={'decode': DecodeMetric()},
    monitors=[
        TGSMonitor(config),
        MemoryMonitor(config),
        LossMonitor(config),
        EvalMonitor(config)
    ],

)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    config=config,
    train_dataset=train_dataset,
    # evaluators=[evaluator],
)

trainer.train()
