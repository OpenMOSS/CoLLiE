import os
import sys
sys.path.append("../../")

import torch
from torch.utils.data import Subset
from transformers import HfArgumentParser
from transformers import set_seed

from collie.models.codegen import CodeGenForCausalLM, CodeGenConfig
from collie.models.codegen_tokenizer import CodeGenTokenizer
from collie.log import print
from arguments import ModelArguments, DataArguments, MyCollieArguments
from mydatasets import MyDataset, get_dataset_info
from mytrainer import MyInplaceZeroTrainer
from utils import DataCollatorForCauselLM, EvalDataCollatorForCauselLM


def compute_metrics(all_pred, eval_dataset):
    preds = all_pred
    golds = [ins['answer'] for ins in eval_dataset.data]
    assert len(preds) == len(golds), f"# of predictions {len(preds)} doesn't match # of references {len(golds)}."

    acc = round(sum([int(pred == gold) for pred, gold in zip(preds, golds)]) / len(golds), 6)
    result = {'acc': acc}
    return result


def train():
    # ========== 1. logs and args ==========
    torch.set_default_dtype(torch.bfloat16)

    parser = HfArgumentParser((ModelArguments, DataArguments, MyCollieArguments))
    if sys.argv[-1].endswith(".yaml"):
        model_args, data_args, collie_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, collie_args = parser.parse_args_into_dataclasses()
    set_seed(collie_args.seed)

    # ========== 2. Load pretrained model and tokenizer. ==========
    config = CodeGenConfig.from_pretrained(model_args.model_name_or_path)
    config.gradient_checkpointing = collie_args.gradient_checkpointing
    model = CodeGenForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)

    tokenizer = CodeGenTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token_id = 0

    # ========== 3. Preprocessing the datasets. ==========
    dataset_info = get_dataset_info(data_args.dataset_name)
    train_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.exemplar_split)
    if data_args.few_shot_size != -1:
        # few_shot_indices = sample(range(len(train_dataset)), data_args.few_shot_size)
        train_dataset = Subset(train_dataset, range(data_args.few_shot_size))

    eval_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.eval_split)
    if dataset_info.test_split:
        test_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.test_split)
        eval_dataset = {
            # 'validation': eval_dataset,
            'test': test_dataset
        }

    # ========== 4. Initialize our Trainer. ==========
    trainer = MyInplaceZeroTrainer(
        model=model,
        collie_args=collie_args,
        data_collator={'train': DataCollatorForCauselLM(tokenizer, max_length=data_args.max_length, padding_side='left'),
                       'eval': EvalDataCollatorForCauselLM(tokenizer, max_length=data_args.max_length, padding_side='left')},
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

if __name__ == "__main__":
    train()
