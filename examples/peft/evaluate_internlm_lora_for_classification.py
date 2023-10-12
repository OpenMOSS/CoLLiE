"""
## Introduction

This example is to evaluate InternLM with a LoRA adapter
on imdb dataset. This example will show you how to:

1. Load a CoLLiE model with a LoRA adapter.
2. Perform evaluation for a peft model.

## Run
```bash
torchrun --nproc_per_node=8 evaluate_internlm_lora_for_classification.py
```
"""
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from collie import (
    AccuracyMetric,
    CollieConfig,
    CollieDatasetForClassification,
    EvalMonitor,
    EvaluatorForClassfication,
    InternLMForCausalLM,
    Trainer,
)
from peft import LoraConfig


def main(args):
    # CoLLiE Config

    config = CollieConfig.from_pretrained(args.base, trust_remote_code=True)
    config.use_flash = True
    config.eval_batch_size = 1
    config.peft_config = LoraConfig.from_pretrained(args.lora, trust_remote_code=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base,
        trust_remote_code=True,
    )

    # Dataset
    eval_dataset_cls = [
        {
            "input": f"Comment: {sample['text']}. The sentiment of this comment is: ",
            "output": ["negative.", "positive."],
            "target": sample["label"],
        }
        for sample in load_dataset("imdb", split="test")
    ][:32]

    eval_dataset_cls = CollieDatasetForClassification(eval_dataset_cls, tokenizer)

    # Model
    model = InternLMForCausalLM.from_pretrained(
        args.base,
        config=config,
        trust_remote_code=True,
    )
    model.set_cache(False)
    model.eval()

    # Load LoRA weights
    trainer = Trainer(model=model, config=config)
    trainer.load_peft(args.lora)

    # Evaluator
    evaluator_cls = EvaluatorForClassfication(
        model=model,
        config=config,
        dataset=eval_dataset_cls,
        monitors=[EvalMonitor(config)],
        metrics={"acc": AccuracyMetric(gather_result=True)},
    )

    # Train
    res = evaluator_cls.eval()
    print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="internlm/internlm-7b")
    parser.add_argument("--lora", type=str)
    args = parser.parse_args()
    main(args)
