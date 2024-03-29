import json
import sys
sys.path.append('../../')
from collie import (
    CheckpointCallback, CollieConfig, CollieDatasetForTraining, EvalMonitor, EvaluatorForPerplexity,
    Qwen2ForCausalLM, LRMonitor, LlamaForCausalLM, LossMonitor, MemoryMonitor, PPLMetric, TGSMonitor, Trainer,
)
import torch
from transformers import AutoTokenizer

BASE_MODEL_PATH = "huggyllama/llama-7b"
BASE_MODEL_PATH = "Qwen/Qwen1.5-7B"
MODEL_PATH = BASE_MODEL_PATH

config = CollieConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
config.dp_size = 1
config.tp_size = 4
config.pp_size = 1
config.train_micro_batch_size = 8
config.eval_batch_size = 8
config.gradient_accumulation_steps = 1
config.eval_per_n_steps = 500
config.train_epochs = 5
# config.checkpointing = False
config.ds_config = {
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
    },
    "monitor_config": {
        "enabled": True,
        "tag": f"will_test",
        "tensorboard": {
            "enabled": True,
            "output_path": "./ds_tb_logs/",
        },
        "csv_monitor": {
            "enabled": True,
            "output_path": "./ds_csv_logs",
        }
    },
}
config.seed = 1024

assert "Qwen" in MODEL_PATH, "Please specify a qwen2 model path, e.g. 'Qwen/Qwen1.5-7B'"
model = Qwen2ForCausalLM.from_pretrained(MODEL_PATH, config=config, trust_remote_code=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-15)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

with open("./zero_shot.json") as fin:
    data_list = json.load(fin)
dataset = []
for data in data_list[:1]:
    dataset.append(
        {
            "input": data["conversations"][0]["value"],
            "output": data["conversations"][1]["value"],
        }
    )

ratio = 0.01
eval_dataset_ppl, train_dataset = dataset[:int(len(dataset) * ratio)], dataset[int(len(dataset) * ratio):]

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, add_eos_token=True)
traine_dataset = CollieDatasetForTraining(
    train_dataset,
    tokenizer=tokenizer,
    max_length=4096
)
eval_dataset_ppl = CollieDatasetForTraining(
    eval_dataset_ppl,
    tokenizer=tokenizer,
    max_length=4096
)

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

### Prepare Trainer
callbacks = [
    CheckpointCallback(
        "./models_2",
        every_n_epochs=2,  # 每2个 epoch 保存一次
        model_only=True,  # 仅保存模型权重，不保存optimzer、训练步数等断点重训信息
    )
]

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
    evaluators=[evaluator_ppl],
    callbacks=callbacks,
    # evaluators=[evaluator_ppl, evaluator_cls]
)
trainer.train()
trainer.save_checkpoint(path="./ckpt", mode="model")
