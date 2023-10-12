"""
## Introduction

This example is to finetune InternLM using parameter effecient finetuning method
on imdb dataset. This example will show you how to:

1. Finetune a model on a dataset using LoRA.
2. Save, load and merge LoRA weights.

## Run
```bash
torchrun --nproc_per_node=8 finetune_internlm_for_classification.py --bs=4 --dp=8
```
"""
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from collie import (
    AccuracyMetric,
    CheckpointCallback,
    CollieConfig,
    CollieDatasetForClassification,
    CollieDatasetForTraining,
    EvalMonitor,
    EvaluatorForClassfication,
    InternLMForCausalLM,
    LossMonitor,
    LRMonitor,
    MemoryMonitor,
    TGSMonitor,
    Trainer,
)
from peft import LoraConfig, TaskType


def main(args):
    # CoLLiE Config

    config = CollieConfig.from_pretrained(args.model, trust_remote_code=True)
    config.use_flash = True
    config.dp_size = args.dp
    config.train_micro_batch_size = args.bs
    config.gradient_accumulation_steps = args.accum
    config.eval_batch_size = 1
    config.eval_per_n_steps = 100
    config.train_epochs = args.epoch
    config.ds_config = {
        "monitor_config": {
            "enabled": True,
            "csv_monitor": {
                "enabled": True,
                "output_path": "ds_logs/csv",
                "job_name": "lora_finetune_internlm_for_classification",
            },
        },
        "fp16": {"enabled": True},
    }
    config.seed = 42

    ## In order to use peft, we need to set peft_config
    config.peft_config = LoraConfig(
        base_model_name_or_path=args.model,
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ]
        if args.ogdu
        else [
            "q_proj",
            "v_proj",
            "k_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    # Dataset
    train_dataset = [
        {
            "input": f"Comment: {sample['text']}. The sentiment of this comment is: ",
            "output": "positive." if sample["label"] else "negative.",
        }
        for sample in load_dataset("imdb", split="train")
    ]

    ### Prepare classification evaluation dataset
    eval_dataset_cls = [
        {
            "input": f"Comment: {sample['text']}. The sentiment of this comment is: ",
            "output": ["negative.", "positive."],
            "target": sample["label"],
        }
        for sample in load_dataset("imdb", split="test")
    ]

    train_dataset = CollieDatasetForTraining(train_dataset, tokenizer)
    eval_dataset_cls = CollieDatasetForClassification(eval_dataset_cls, tokenizer)

    # Model
    model = InternLMForCausalLM.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
    )
    model.set_cache(False)
    model.enable_input_require_grads()

    # Optimizer

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    # Evaluator
    evaluator_cls = EvaluatorForClassfication(
        model=model,
        config=config,
        dataset=eval_dataset_cls,
        monitors=[EvalMonitor(config)],
        metrics={"acc": AccuracyMetric(gather_result=True)},
    )

    # Trainer
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
            LRMonitor(config),
        ],
        evaluators=[evaluator_cls],
        callbacks=[
            CheckpointCallback(
                folder=f"./lora/{args.jobname}",
                every_n_epochs=1,
                last=True,
                adapter_name=args.jobname,
            )
        ],
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobname", type=str, default="default")
    parser.add_argument("--model", type=str, default="internlm/internlm-7b")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--ogdu", type=bool, default=False)
    args = parser.parse_args()
    main(args)
