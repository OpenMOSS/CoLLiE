import sys

sys.path.append('..')
import os
from random import sample

import torch
from torch.utils.data import Subset
from transformers import HfArgumentParser
from transformers import set_seed
from tunelite.models import llama
import wandb

from V2.arguments import ModelArguments, DataArguments, TuneLiteArguments
from V2.collator_v2 import DataCollatorForCauselLM, EvalDataCollatorForCauselLM
from V2.datasets_v2 import MyDataset, get_dataset_info, get_train_dataloader, get_eval_dataloader
from V2.tensor_trainer_v2 import MyInplaceTensorTrainer


def compute_metrics(all_pred, eval_dataset):
    preds = all_pred
    golds = [ins['answer'] for ins in eval_dataset.data]
    assert len(preds) == len(golds), f"# of predictions {len(preds)} doesn't match # of references {len(golds)}."

    acc = round(sum([int(pred == gold) for pred, gold in zip(preds, golds)]) / len(golds), 6)
    result = {'acc': acc}
    return result


def train():
    # ========== 1. logs and args ==========
    local_rank, world_size = llama.setup_model_parallel()
    parser = HfArgumentParser((ModelArguments, DataArguments, TuneLiteArguments))
    if sys.argv[-1].endswith(".yaml"):
        model_args, data_args, tl_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, tl_args = parser.parse_args_into_dataclasses()
    set_seed(tl_args.seed)

    if tl_args.local_rank in [-1, 0]:
        wandb.init(
            project="tunelite",
            name=tl_args.run_name,
            tags=[data_args.data_tag, tl_args.tag],
            config={'model_args': model_args, 'data_args': data_args, 'tl_args': tl_args},
        )

    # ========== 2. Load pretrained model and tokenizer. ==========
    model, tokenizer = llama.load_model(
        ckpt_dir=model_args.model_name_or_path,  # 7B, 13B, 30B, 65B
        tokenizer_path=os.path.join(model_args.llama_dir, 'tokenizer.model'),
        local_rank=local_rank,
        world_size=world_size,
        froze_embeddings=False,
        zero=False,
        tensor_parallel=True,
        pipeline_parallel=False,
        max_batch_size=4 * tl_args.per_device_eval_batch_size,
        max_seq_len=data_args.max_length,
    )
    torch.cuda.empty_cache()
    tokenizer.pad_token_id = 0

    # ========== 3. Preprocessing the datasets. ==========
    dataset_info = get_dataset_info(data_args.dataset_name)
    train_dataset = MyDataset(data_args, tokenizer, dataset_info, dataset_info.exemplar_split)
    if data_args.few_shot_size != -1:
        # few_shot_indices = sample(range(len(train_dataset)), data_args.few_shot_size)
        train_dataset = Subset(train_dataset, range(data_args.few_shot_size))
    train_dataloader = get_train_dataloader(
        train_dataset,
        DataCollatorForCauselLM(tokenizer, max_length=data_args.max_length, padding_side='left'),
        tl_args
    )
    test_dataset = MyDataset(data_args, tokenizer, dataset_info, dataset_info.test_split)
    test_dataloader = get_eval_dataloader(
        test_dataset,
        EvalDataCollatorForCauselLM(tokenizer, max_length=data_args.max_length, padding_side='left'),
        tl_args
    )

    # ========== 4. Initialize our Trainer. ==========
    print(f"***** Running Training *****")
    print(f"  Num examples: {len(train_dataset)}")
    print(f"  Num Epochs: {tl_args.num_train_epochs}")
    print(f"  Batch Size: {tl_args.per_device_train_batch_size}")

    trainer = MyInplaceTensorTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        eval_dataset=test_dataset,
        tl_args=tl_args,
        compute_metrics=compute_metrics,
    )
    trainer.train()


# run with $torchrun --nproc_per_node 2 train_inplace_tensor.py config/tensor_args.yaml
if __name__ == "__main__":
    train()
