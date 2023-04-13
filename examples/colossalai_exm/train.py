import os
import sys

sys.path.append("../../")

import torch
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from transformers import set_seed

from collie.log import print
from arguments import ModelArguments, DataArguments, TrainerArgs
from mydatasets import MyDataset, get_dataset_info
from utils import DataCollatorForCauselLM, EvalDataCollatorForCauselLM
from utils import get_llama
from mytrainer import MyColossalaiTrainer


def compute_metrics(batch, generated_batch, epoch, step):
    """
    
    :param batch: 一个元组，第一个位置是字典，包含 input_ids, attention_mask 等；
        第二个位置是 labels。
    :param generated_batch: 生成的结果，是一个 tensor。
    :param epoch:
    :param step:
    """
    print(batch[0])
    golds = batch[0]["answer"].tolist()
    preds = batch[0]["input_ids"].tolist()
    assert len(preds) == len(golds), f"# of predictions {len(preds)} doesn't match # of references {len(golds)}."

    acc = round(sum([int(pred == gold) for pred, gold in zip(preds, golds)]) / len(golds), 6)
    result = {'acc': acc}
    return result


def train():
    """
    Parse args
    """
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainerArgs))
    if sys.argv[-1].endswith(".yaml"):
        model_args, data_args, collie_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, collie_args = parser.parse_args_into_dataclasses()
    # set_seed(collie_args.seed)

    # ========== 2. Load pretrained model and tokenizer. ==========
    llama_pipeline, tokenizer = get_llama(model_args)
    torch.cuda.empty_cache()
    tokenizer.pad_token_id = 0

    # ========== 3. Preprocessing the datasets. ==========
    dataset_info = get_dataset_info(data_args.dataset_name)
    train_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.exemplar_split)

    eval_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.eval_split)
    if dataset_info.test_split:
        test_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.test_split)
        # eval_dataset = {'validation': eval_dataset, 'test': test_dataset}

    train_dataloader = DataLoader(
        train_dataset, batch_size=1,
        collate_fn=DataCollatorForCauselLM(
            tokenizer, max_length=data_args.max_length, padding_side='left'
        ),
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=1,
        collate_fn=EvalDataCollatorForCauselLM(
            tokenizer, max_length=data_args.max_length, padding_side='left'
        ),
    )
    optimizer = torch.optim.SGD(llama_pipeline.parameters(), lr=0.0001)
    # ========== 4. Initialize our Trainer. ==========
    trainer = MyColossalaiTrainer(
        model=llama_pipeline, tokenizer=tokenizer,
        optimizer=optimizer,
        train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
        compute_metrics=compute_metrics, trainer_args=collie_args
    )
    trainer.train()


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py hf_args.yaml
if __name__ == "__main__":
    train()
