import sys

sys.path.append("../../../")
from collie import Trainer, EvaluatorForPerplexity, InternLM2ForCausalLM, CollieConfig, PPLMetric, AccuracyMetric, \
    DecodeMetric, CollieDatasetForTraining, CollieDatasetForClassification, \
    LossMonitor, TGSMonitor, MemoryMonitor, EvalMonitor, GradioProvider, EvaluatorForClassfication, LRMonitor
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch

config = CollieConfig.from_pretrained("internlm/internlm2-7b", trust_remote_code=True)
config.dp_size = 4
config.tp_size = 1
config.pp_size = 1
config.train_micro_batch_size = 4
config.eval_batch_size = 4
config.gradient_accumulation_steps = 1
config.eval_per_n_steps = 300
config.train_epochs = 1
# config.checkpointing = False
config.ds_config = {
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
    }
}
config.seed = 1024
# setup_distribution(config)
model = InternLM2ForCausalLM.from_pretrained("internlm/internlm2-7b", config=config)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
### Prepare training dataset
train_dataset = [
    {
        "input": f"Comment: {sample['text']}. The sentiment of this comment is: ",
        "output": "positive." if sample["label"] else "negative."
    } for sample in load_dataset("imdb", split="train")
]
### Prepare perplexity evaluation dataset
ratio = 0.01
eval_dataset_ppl, train_dataset = train_dataset[:int(len(train_dataset) * ratio)], train_dataset[
                                                                                   int(len(train_dataset) * ratio):]
### Prepare classification evaluation dataset
eval_dataset_cls = [
                       {
                           "input": f"Comment: {sample['text']}. The sentiment of this comment is: ",
                           "output": ["negative.", "positive."],
                           "target": sample["label"]
                       } for sample in load_dataset("imdb", split="test")
                   ][:1000]
### Convert to CoLLie Dataset
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-7b", trust_remote_code=True, add_eos_token=True)
traine_dataset = CollieDatasetForTraining(train_dataset,
                                          tokenizer=tokenizer)
eval_dataset_ppl = CollieDatasetForTraining(eval_dataset_ppl,
                                            tokenizer=tokenizer)
eval_dataset_cls = CollieDatasetForClassification(eval_dataset_cls,
                                                  tokenizer=tokenizer)
### Prepare Evaluator
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
evaluator_cls = EvaluatorForClassfication(
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
    evaluators=[evaluator_cls],
    # evaluators=[evaluator_ppl, evaluator_cls]
)
trainer.train()
# trainer.save_checkpoint(path="/mnt/petrelfs/zhangshuo/model/test_save_checkpoint", mode="model")
