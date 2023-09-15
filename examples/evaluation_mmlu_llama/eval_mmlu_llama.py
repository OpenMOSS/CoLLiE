"""
## Introduction

This example is to evaluate Llama on Massive Multitask Language Understanding
(MMLU) dataset.

MMLU is a new benchmark designed to measure knowledge acquired during
pretraining by evaluating models exclusively in zero-shot and few-shot
settings.

You can get a further introduction at https://paperswithcode.com/dataset/mmlu

Some of the following codes are from https://github.com/hendrycks/test

## Preparation

If the dataset already exists, pass the -d argument to specify the path.
Otherwise, download the dataset by the following command.

``` bash
wget -c https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar xvf data.tar
```

Now the MMLU dataset is in the `./data` directory.

## Customization

You can customize your prompt template in `utils_mmlu.py`.

## Run
```bash
torchrun --nproc_per_node=8 eval_mmlu_llama.py --dp 8
```

Refer to line 167 ~ 176 for more arguments.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from utils_mmlu import *

from collie import (
    AccuracyMetric,
    CollieConfig,
    CollieDatasetForClassification,
    EvaluatorForClassfication,
    LlamaForCausalLM,
    env,
)


def generate_eval_data(k, subject, dev_df, test_df):
    eval_data = []
    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        eval_data.append(
            {
                "input": prompt,
                "output": choices,
                "target": choice_to_id[test_df.iloc[i, test_df.shape[1] - 1]],
            }
        )
    return eval_data


def main(args):
    model_name = args.model.split("/")[-1]
    config = CollieConfig.from_pretrained(args.model, trust_remote_code=True)
    config.tp_size = args.tp
    config.dp_size = args.dp
    config.pp_size = args.pp
    config.eval_batch_size = args.bs

    model = LlamaForCausalLM.from_pretrained(args.model, config=config)
    model.set_cache(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    eval_dataset = CollieDatasetForClassification(
        {},
        tokenizer,
        style=args.style,
    )
    acc_evaluator = EvaluatorForClassfication(
        model=model,
        dataset=eval_dataset,
        tokenizer=tokenizer,
        config=config,
        metrics={"acc": AccuracyMetric(gather_result=True)},
        max_new_tokens=args.max_new_tokens,
    )

    if env.rank == 0:
        all_cors = []
        subcat_cors = {
            subcat: []
            for subcat_lists in subcategories.values()
            for subcat in subcat_lists
        }
        cat_cors = {cat: [] for cat in categories}
        results = {}

    for i, subject in enumerate(subjects):
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        acc_evaluator.dataset = CollieDatasetForClassification(
            generate_eval_data(args.ntrain, subject, dev_df, test_df),
            tokenizer,
            style=args.style,
        )
        acc_evaluator.eval_dataloader = None

        result = acc_evaluator.eval()
        result = result["acc#acc"]
        if env.rank == 0:
            print(f"[{i+1}/{len(subjects)}] Average accuracy {result:.3f} - {subject}")
            result = [result for _ in range(len(test_df))]  # 便于加权平均
            subcats = subcategories[subject]
            for subcat in subcats:
                subcat_cors[subcat].append(result)
                for key in categories.keys():
                    if subcat in categories[key]:
                        cat_cors[key].append(result)
            all_cors.append(result)
            results[subject] = result

    if env.rank == 0:
        for subcat in subcat_cors:
            if len(subcat_cors[subcat]) == 0:
                continue
            subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
            print(f"Average accuracy {subcat_acc:.3f} - {subcat}")

        for cat in cat_cors:
            if len(cat_cors[cat]) == 0:
                continue
            cat_acc = np.mean(np.concatenate(cat_cors[cat]))
            print(f"Average accuracy {cat_acc:.3f} - {cat}")

        weighted_acc = np.mean(np.concatenate(all_cors))
        print(f"Average accuracy: {weighted_acc:.3f}")
        with open(
            os.path.join(args.save_dir, f"{model_name}_mmlu.json"), "w", encoding="utf8"
        ) as f:
            json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="test_results")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--style", "-l", type=str, default="harness")
    parser.add_argument("--max_new_tokens", "-t", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--bs", type=int, default=1)

    args = parser.parse_args()
    main(args)
