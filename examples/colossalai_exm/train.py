import os
import sys

sys.path.append("../../")

import torch
from torch.utils.data import DataLoader, Subset
from torch.nn import CrossEntropyLoss
from transformers import HfArgumentParser
from transformers import set_seed
from transformers.trainer_pt_utils import nested_numpify
from colossalai.nn.lr_scheduler import LinearWarmupLR

from collie.log import print
from arguments import ModelArguments, DataArguments, TrainerArguments
from mydatasets import MyDataset, get_dataset_info
from utils import DataCollatorForCauselLM, EvalDataCollatorForCauselLM
from utils import get_llama
from mytrainer import MyColossalaiTrainer


def compute_metrics(batch, epoch, step):
    """
    
    :param batch: 一个元组，第一个位置是字典，包含 输出的 logits, attention_mask
        等；第二个位置是 labels。
    :param generated_batch: 生成的结果，是一个 tensor。
    :param epoch:
    :param step:
    """
    data_dict, labels = batch
    logits = data_dict["input_ids"]
    split_size = data_dict["split_size"]

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
                    shift_labels.view(-1)).view_as(shift_labels)
    loss = loss.mean(dim=1)
    # 将 loss 按照 Split_size 分组，得到每一组问题的 loss
    # 进而选出 answer
    group_loss = loss.split(split_size)
    preds = torch.stack([torch.argmin(l) for l in group_loss], dim=0)

    preds = nested_numpify(preds).tolist()
    golds = data_dict["answer"].tolist()
    
    assert len(preds) == len(golds), f"# of predictions {len(preds)} doesn't match # of references {len(golds)}."

    acc = round(sum([int(pred == gold) for pred, gold in zip(preds, golds)]) / len(golds), 6)
    result = {'acc': acc}

    return result


def train():
    """
    Parse args
    """
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainerArguments))
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

    if data_args.few_shot_size != -1:
        train_dataset = Subset(train_dataset, range(data_args.few_shot_size))

    train_dataloader = DataLoader(
        train_dataset, batch_size=model_args.micro_batch_num,
        collate_fn=DataCollatorForCauselLM(
            tokenizer, max_length=data_args.max_length, padding_side='left'
        ), drop_last=True, shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=model_args.micro_batch_num,
        collate_fn=EvalDataCollatorForCauselLM(
            tokenizer, max_length=data_args.max_length, padding_side='left'
        ), drop_last=True
    )
    optimizer = torch.optim.SGD(llama_pipeline.parameters(), lr=collie_args.learning_rate)
    lr_scheduler = LinearWarmupLR(
        optimizer, total_steps=len(train_dataloader) * collie_args.epochs,
        warmup_steps=collie_args.warmup
    )
    # ========== 4. Initialize our Trainer. ==========
    trainer = MyColossalaiTrainer(
        model=llama_pipeline, tokenizer=tokenizer,
        optimizer=optimizer, lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
        compute_metrics=compute_metrics, trainer_args=collie_args
    )
    trainer.train()


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py hf_args.yaml
if __name__ == "__main__":
    train()
