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

Refer to line 170 ~ 180 for more arguments.
"""

import argparse
import json
import os

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


# Transform data to the format of CollieDatasetForClassification.
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

    # `dataset` is required to init evaluator. Pass an empty dict here.
    # The real dataset will be passed in the following loop.
    acc_evaluator = EvaluatorForClassfication(
        model=model,
        dataset={},
        tokenizer=tokenizer,
        config=config,
        metrics={"acc": AccuracyMetric(gather_result=True)},
        max_new_tokens=args.max_new_tokens,
    )

    if env.rank == 0:
        all_cors = {"total": 0, "correct": 0}
        subcat_cors = {
            subcat: {"total": 0, "correct": 0}
            for subcat_lists in subcategories.values()
            for subcat in subcat_lists
        }
        cat_cors = {cat: {"total": 0, "correct": 0} for cat in categories}
        results = {}

    for i, subject in enumerate(subjects):
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        # Pass the real dataset.
        acc_evaluator.dataset = CollieDatasetForClassification(
            generate_eval_data(args.ntrain, subject, dev_df, test_df),
            tokenizer,
            style=args.style,
        )
        # Evaluator will cache the dataloader.
        # Set it to None to use the new dataset.
        acc_evaluator.eval_dataloader = None

        evaluate_result = acc_evaluator.eval()
        if env.rank == 0:
            print(
                f"[{i+1}/{len(subjects)}] Average accuracy {evaluate_result['acc#acc']:.3f} - {subject}"
            )
            subcats = subcategories[subject]
            # Collie returns int64, which is not JSON serializable.
            # Convert them to int.
            evaluate_result["acc#total"] = int(evaluate_result["acc#total"])
            evaluate_result["acc#correct"] = int(evaluate_result["acc#correct"])
            for subcat in subcats:
                subcat_cors[subcat]["total"] += evaluate_result["acc#total"]
                subcat_cors[subcat]["correct"] += evaluate_result["acc#correct"]
                for key in categories.keys():
                    if subcat in categories[key]:
                        cat_cors[key]["total"] += evaluate_result["acc#total"]
                        cat_cors[key]["correct"] += evaluate_result["acc#correct"]
            all_cors["total"] += evaluate_result["acc#total"]
            all_cors["correct"] += evaluate_result["acc#correct"]
            results[subject] = evaluate_result

    if env.rank == 0:
        for subcat in subcat_cors:
            if subcat_cors[subcat]["total"] == 0:
                continue
            subcat_acc = subcat_cors[subcat]["correct"] / subcat_cors[subcat]["total"]
            print(f"Average accuracy {subcat_acc:.3f} - {subcat}")

        for cat in cat_cors:
            if cat_cors[cat]["total"] == 0:
                continue
            cat_acc = cat_cors[cat]["correct"] / cat_cors[cat]["total"]
            print(f"Average accuracy {cat_acc:.3f} - {cat}")

        weighted_acc = all_cors["correct"] / all_cors["total"]
        print(f"Average accuracy: {weighted_acc:.3f}")

        model_name = args.model.split("/")[-1]
        os.makedirs(args.save_dir, exist_ok=True)
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
