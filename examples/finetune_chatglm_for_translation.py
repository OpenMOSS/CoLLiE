import sys
sys.path.append("..")
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig
from functools import partial
from collie import Trainer, EvaluatorForPerplexity, ChatGLMForCausalLM, CollieConfig, PPLMetric, AccuracyMetric, DecodeMetric, CollieDatasetForTraining, CollieDatasetForGeneration, \
    LossMonitor, TGSMonitor, MemoryMonitor, EvalMonitor, GradioProvider, EvaluatorForGeneration, LRMonitor, BleuMetric, DashProvider
config = CollieConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
config.pp_size = 8
config.train_micro_batch_size = 1
config.eval_batch_size = 1
config.gradient_accumulation_steps = 128
config.eval_per_n_epochs = 1
config.train_epochs = 10
config.ds_config = {
    "fp16": {
        "enabled": True
    },
    "monitor_config": {
        "enabled": True,
        "wandb": {
            "enabled": True,
            "team": "00index",
            "project": "collie",
            "group": "translation"
        }
    }
}
config.seed = 1024
# Prepare training dataset
train_dataset = [
    {
        "input": f"[Round {0}]\n问：把下面这句话从法语翻译成英语. {sample['translation']['fr']}\n答：[gMASK]<sop>",
        "output": f"{sample['translation']['en']}<eop>"
    } for sample in load_dataset("iwslt2017", name="iwslt2017-fr-en", split="train[:8000]")
]
# Prepare perplexity evaluation dataset
ratio = 0.01
eval_dataset_ppl, train_dataset = train_dataset[:int(
    len(train_dataset) * ratio)], train_dataset[int(len(train_dataset) * ratio):]
# Prepare classification evaluation dataset
eval_dataset_bleu = [
    {
        "text": f"[Round {0}]\n问：把下面这句话从法语翻译成英语. {sample['translation']['fr']}\n答：[gMASK]<sop>",
        "target": " ".join(f"{sample['translation']['en']}<eop>".split())
    } for sample in load_dataset("iwslt2017", name="iwslt2017-fr-en", split="train[100:150]")
]
# Prepare model
model = ChatGLMForCausalLM.from_pretrained(
    "/mnt/petrelfs/zhangshuo/model/chatglm-6b", config=config)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer=optimizer, T_max=config.train_epochs * len(train_dataset) / (config.train_micro_batch_size * config.gradient_accumulation_steps))
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer, step_size=1, gamma=0.9
)
tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm-6b", trust_remote_code=True)
# 默认的tokenizer不会把[gMASK]当作一个token，所以需要手动添加
tokenizer.unique_no_split_tokens.append("[gMASK]")
# Convert to CoLLie Dataset
traine_dataset = CollieDatasetForTraining(train_dataset,
                                          tokenizer=partial(tokenizer, add_special_tokens=False))
eval_dataset_ppl = CollieDatasetForTraining(eval_dataset_ppl,
                                            tokenizer=partial(tokenizer, add_special_tokens=False))
eval_dataset_bleu = CollieDatasetForGeneration(eval_dataset_bleu,
                                               tokenizer=partial(tokenizer, add_special_tokens=False))
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
        "bleu": BleuMetric(gather_result=True, ngram=1),
        "decode": DecodeMetric()
    },
    generation_config=GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=100,
    )
)
# Prepare Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    lr_scheduler=lr_scheduler,
    config=config,
    optimizer=optimizer,
    train_dataset=traine_dataset,
    monitors=[
        LossMonitor(config),
        TGSMonitor(config),
        MemoryMonitor(config),
        LRMonitor(config)
    ],
    evaluators=[evaluator_ppl, evaluator_bleu]
)
trainer.train()
# trainer.save_checkpoint(path="/mnt/petrelfs/zhangshuo/model/test_save_checkpoint", mode="model")