import copy
import os
import sys

from random import sample

import torch
from torch.utils.data import Subset
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import set_seed
from dataclasses import asdict
from transformers.deepspeed import HfDeepSpeedConfig
import wandb
# os.environ['WANDB_MODE'] = 'debug'

python_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print("PYTHON_PATH", python_path)
sys.path.append(python_path)
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
    # torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    parser = HfArgumentParser((ModelArguments, DataArguments, MyCollieArguments))
    if sys.argv[-1].endswith(".yaml"):
        model_args, data_args, collie_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, collie_args = parser.parse_args_into_dataclasses()
    set_seed(collie_args.seed)

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
            entity='collie_exp',
            name=tag_name if hparam_name == 'output' else '_'.join([tag_name, hparam_name.replace('output_', '')]),
            config=wandb_config
        )

    # ========== 2. Load pretrained model and tokenizer. ==========
    ds_config = collie_args.deepspeed
    dschf = HfDeepSpeedConfig(ds_config)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.gradient_checkpointing = collie_args.gradient_checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=True,
        config=config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
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


# run with $torchrun --nproc_per_node 2 train_inplace_tensor.py config/tensor_args.yaml
if __name__ == "__main__":
    train()
