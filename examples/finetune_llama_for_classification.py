import sys
sys.path.append("..")
from collie import Trainer, PerplexityEvaluator, LlamaForCausalLM, CollieConfig, PPLMetric, AccuracyMetric, DecodeMetric, CollieDatasetForTraining, CollieDatasetForClassification, \
    LossMonitor, TGSMonitor, MemoryMonitor, EvalMonitor, GradioProvider, ClassficationEvaluator, LRMonitor
from transformers import LlamaTokenizer
from datasets import load_dataset
import torch

config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf")
config.dp_size = 4
config.train_micro_batch_size = 16
config.eval_batch_size = 2
config.gradient_accumulation_steps = 1
config.eval_per_n_steps = 300
# config.checkpointing = False
config.ds_config = {
    "fp16": {
        "enabled": True
    },
    # "monitor_config": {
    #     "enabled": True,
    #     "wandb": {
    #         "enabled": True,
    #         "team": "00index",
    #         "project": "collie",
    #         "group": "test_evaluator"
    #     }
    # },
    "zero_optimization": {
        "stage": 3,
    }
}
config.seed = 1024
model = LlamaForCausalLM.from_pretrained("/mnt/petrelfs/zhangshuo/model/llama-7b-hf", config=config)
# model = LlamaForCausalLM.from_config(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
### Prepare training dataset
train_dataset = [
    {
        "input": f"Comment: {sample['text']}. The sentiment of this comment is: ",
        "output": "positive." if sample["label"] else "negative."
    } for sample in load_dataset("imdb", split="train")
]
# train_dataset = [
#     {
#         "text": f"Comment: {sample['text']}. The sentiment of this comment is: {'positive.' if sample['label'] else 'negative.'}",
#     } for sample in load_dataset("imdb", split="train")
# ]
### Prepare perplexity evaluation dataset
ratio = 0.01
eval_dataset_ppl, train_dataset = train_dataset[:int(len(train_dataset) * ratio)], train_dataset[int(len(train_dataset) * ratio):]
### Prepare classification evaluation dataset
eval_dataset_cls = [
    {
        "input": f"Comment: {sample['text']}. The sentiment of this comment is: ",
        "output": ["negative.", "positive."],
        "target": sample["label"] 
    } for sample in load_dataset("imdb", split="test")
][:1000]
### Convert to CoLLie Dataset
traine_dataset = CollieDatasetForTraining(train_dataset, 
                                          tokenizer=LlamaTokenizer.from_pretrained("/mnt/petrelfs/zhangshuo/model/llama-7b-hf", add_eos_token=True))
eval_dataset_ppl = CollieDatasetForTraining(eval_dataset_ppl, 
                                            tokenizer=LlamaTokenizer.from_pretrained("/mnt/petrelfs/zhangshuo/model/llama-7b-hf", add_eos_token=True))
eval_dataset_cls = CollieDatasetForClassification(eval_dataset_cls, 
                                              tokenizer=LlamaTokenizer.from_pretrained("/mnt/petrelfs/zhangshuo/model/llama-7b-hf", add_eos_token=True))
### Prepare Evaluator
evaluator_ppl = PerplexityEvaluator(
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
evaluator_cls = ClassficationEvaluator(
    model=model,
    config=config,
    dataset=eval_dataset_cls,
    monitors=[
        EvalMonitor(config)
    ],
    metrics={
        "acc": AccuracyMetric(gather_result=True)
    },
)
### Prepare Trainer
trainer = Trainer(
    model=model,
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
    data_provider=GradioProvider(LlamaTokenizer.from_pretrained("/mnt/petrelfs/zhangshuo/model/llama-7b-hf"), port=12300, stream=True),
    evaluators=[evaluator_ppl, evaluator_cls]
)
trainer.train()
# trainer.save_checkpoint(path="/mnt/petrelfs/zhangshuo/model/test_save_checkpoint", mode="model")

