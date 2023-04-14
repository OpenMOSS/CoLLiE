import copy
import os
import sys

from random import sample

import torch
from torch.utils.data import Subset
from transformers import HfArgumentParser
from transformers import set_seed
from dataclasses import asdict
import wandb
# os.environ['WANDB_MODE'] = 'debug'

python_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print("PYTHON_PATH", python_path)
sys.path.append(python_path)

from collie.models import llama
from collie.log import print
from arguments import ModelArguments, DataArguments, MyCollieArguments
from mydatasets import MyDataset, get_dataset_info
from mytrainer import MyInplaceTensorTrainer
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
    local_rank, world_size = llama.setup_model_parallel()
    parser = HfArgumentParser((ModelArguments, DataArguments, MyCollieArguments))
    if sys.argv[-1].endswith(".yaml"):
        model_args, data_args, collie_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, collie_args = parser.parse_args_into_dataclasses()
    set_seed(collie_args.seed)
    assert local_rank == collie_args.local_rank
    assert world_size == collie_args.world_size

    model_name = model_args.model_name_or_path.split('/')[-1]
    tag_name = '_'.join([data_args.dataset_name, model_name, collie_args.tag] if collie_args.tag else [data_args.dataset_name, model_name])
    hparam_name = 'output'
    if collie_args.optim != 'sgd':
        hparam_name += '_' + collie_args.optim
    if collie_args.learning_rate != 5e-4:
        hparam_name += '_lr' + str(collie_args.learning_rate)
    if collie_args.per_device_train_batch_size != 8:
        hparam_name += '_bs' + str(collie_args.per_device_train_batch_size)
    if collie_args.lr_scheduler_type != 'linear':
        hparam_name += '_' + collie_args.lr_scheduler_type
    if collie_args.warmup != 0:
        hparam_name += '_warmup' + str(collie_args.warmup)
    if collie_args.clip_grad_value:
        hparam_name += '_clipgrad' + str(collie_args.clip_grad_value)
    if collie_args.clip_loss_value:
        hparam_name += '_cliploss' + str(collie_args.clip_loss_value)
    assert collie_args.clip_grad_value is None or collie_args.clip_loss_value is None
    collie_args.output_dir = os.path.join('outputs', tag_name, hparam_name)

    if collie_args.tag == 'debug':
        os.environ['WANDB_MODE'] = 'offline'
    if collie_args.local_rank in [-1, 0]:
        wandb_config = copy.deepcopy(asdict(collie_args))
        wandb_config.update(asdict(model_args))
        wandb_config.update(asdict(data_args))
        wandb.init(
            project="collie",
            name=tag_name if hparam_name == 'output' else '_'.join([tag_name, hparam_name.replace('output_', '')]),
            config=wandb_config
        )

    # ========== 2. Load pretrained model and tokenizer. ==========
    model, tokenizer = llama.load_model(
        ckpt_dir=os.path.join(model_args.cache_dir, model_args.model_name_or_path.split('-')[-1]),  # 7B, 13B, 30B, 65B
        tokenizer_path=os.path.join(model_args.cache_dir, 'tokenizer.model'),
        local_rank=collie_args.local_rank,
        world_size=collie_args.world_size,
        froze_embeddings=False,
        zero=False,
        tensor_parallel=True,
        pipeline_parallel=False,
        max_batch_size=4 * collie_args.per_device_eval_batch_size,
        max_seq_len=data_args.max_length,
    )
    torch.cuda.empty_cache()
    tokenizer.pad_token_id = 0

    # ========== 3. Preprocessing the datasets. ==========
    dataset_info = get_dataset_info(data_args.dataset_name)
    train_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.exemplar_split)
    # if data_args.few_shot_size != -1:
    #     # few_shot_indices = sample(range(len(train_dataset)), data_args.few_shot_size)
    #     train_dataset = Subset(train_dataset, range(data_args.few_shot_size))
    eval_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.eval_split)
    if dataset_info.test_split:
        test_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.test_split)
        eval_dataset = {'validation': eval_dataset, 'test': test_dataset}

    # ========== 4. Initialize our Trainer. ==========
    trainer = MyInplaceTensorTrainer(
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


# run with $torchrun --nproc_per_node 2 train_inplace_tensor.py config/tensor_args.yaml
if __name__ == "__main__":
    train()
