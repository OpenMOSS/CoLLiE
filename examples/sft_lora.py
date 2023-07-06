import sys
import os
sys.path.append("../")
import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, GenerationConfig
from collie import Trainer, EvaluatorForPerplexity, CollieConfig, PPLMetric, \
    DecodeMetric, CollieDatasetForTraining, CollieDatasetForGeneration, \
    LossMonitor, TGSMonitor, MemoryMonitor, EvalMonitor, GradioProvider, \
        EvaluatorForGeneration, LRMonitor, BleuMetric, LlamaForCausalLM, \
            CheckpointCallback
from peft import LoraConfig, TaskType
# Prepare training config
config = CollieConfig.from_pretrained("/mnt/petrelfs/share_data/zhangshuo/model/MOSS_7B_Base")
config.train_micro_batch_size = 1
config.gradient_accumulation_steps = 1
config.eval_batch_size = 1
config.eval_per_n_steps = 10
config.train_epochs = 100
# pipeline 配置 8 卡， 划分方法采用 uniform
config.pp_size = 8
config.pp_partition_method = "uniform"

config.ds_config = {
    "monitor_config": {
        "enabled": True,
        "csv_monitor": {
            "enabled": True,
            "output_path": "./ds_logs/",
            "job_name": "full_finetuning_moss_7b"
        }
    },
    "fp16": {
        "enabled": True
    },
    # "zero_optimization": {
    #     "stage": 3,
    # }
}
config.seed = 1024
config.peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
# Prepare training dataset
train_dataset = [
    {
        "input": f"""<s>下面是描述任务的指令，并与提供进一步符合上下文的输入。请编写适当完成请求的响应。

### 指令：
{sample["instruction"]}

### 输入：
{sample["input"]}

### 响应：
""" if len(sample["input"].strip()) != 0 else f"""<s>下面是描述任务的指令。请编写适当完成请求的响应。

### 指令：
{sample["instruction"]}

### 响应：
""",
        "output": f"{sample['output']}</s>"
    } for sample in load_dataset("Chinese-Vicuna/guanaco_belle_merge_v1.0", split="train[:-100]")
]
# Prepare perplexity evaluation dataset
ratio = 0.01
eval_dataset_ppl, train_dataset = train_dataset[:int(
    len(train_dataset) * ratio)], train_dataset[int(len(train_dataset) * ratio):]
# Prepare generation based evaluation dataset
eval_dataset_bleu = [
    {
        "text": f"""<s>下面是描述任务的指令，并与提供进一步符合上下文的输入。请编写适当完成请求的响应。

### 指令：
{sample["instruction"]}

### 输入：
{sample["input"]}

### 响应：
""" if len(sample["input"].strip()) != 0 else f"""<s>下面是描述任务的指令。请编写适当完成请求的响应。

### 指令：
{sample["instruction"]}

### 响应：
""",
        "target": " ".join(f"{sample['output']}</s>".split())
    } for sample in load_dataset("Chinese-Vicuna/guanaco_belle_merge_v1.0", split="train[-100:]")
]
# Prepare model
model = LlamaForCausalLM.from_pretrained(
    "/mnt/petrelfs/share_data/zhangshuo/model/MOSS_7B_Base", config=config)
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# 这里 filter 掉不需要的划分的参数，防止开始 fp16 时候显存暴增的问题
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
tokenizer = LlamaTokenizer.from_pretrained(
    "/mnt/petrelfs/share_data/zhangshuo/model/MOSS_7B_Base", add_bos_token=False)
# Convert to CoLLie Dataset
train_dataset = CollieDatasetForTraining(train_dataset,
                                          tokenizer=tokenizer)
eval_dataset_ppl = CollieDatasetForTraining(eval_dataset_ppl,
                                            tokenizer=tokenizer)
eval_dataset_bleu = CollieDatasetForGeneration(eval_dataset_bleu,
                                               tokenizer=tokenizer)
model.enable_input_require_grads()
# Prepare Evaluator
evaluator_ppl = EvaluatorForPerplexity(
    model=model,
    config=config,
    dataset=eval_dataset_ppl,
    monitors=[
        EvalMonitor(config)
    ],
    metrics={
        "ppl": PPLMetric(gather_result=True)
    },
)
evaluator_bleu = EvaluatorForGeneration(
    model=model,
    config=config,
    dataset=eval_dataset_bleu,
    monitors=[
        EvalMonitor(config)
    ],
    metrics={
        # "bleu": BleuMetric(gather_result=True, ngram=1),
        "decode": DecodeMetric()
    },
    generation_config=GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=100,
        use_cache=False
    )
)
# Prepare Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    optimizer=optimizer,
    train_dataset=train_dataset,
    monitors=[
        LossMonitor(config),
        TGSMonitor(config),
        MemoryMonitor(config),
        LRMonitor(config)
    ],
    data_provider=GradioProvider(tokenizer, port=13000, stream=True,
                                 generation_config=GenerationConfig(
                                     eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.pad_token_id,
                                     max_new_tokens=250,
                                 )),
    evaluators=[evaluator_bleu],
    callbacks=[
        CheckpointCallback(
            folder="/mnt/petrelfs/hongjiawei/collie/examples/tmp", 
            every_n_epochs=1, last=True, every_n_batches=20)
    ]
)
# Command: torchrun --standalone --nproc_per_node=8 sft.py
trainer.train()
