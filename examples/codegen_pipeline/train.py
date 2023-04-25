import sys
sys.path.append("../../")
import os
import datetime

import torch
from torch.nn import CrossEntropyLoss

from collie.models.codegen_pipeline import get_codegen_pipeline, get_codgen_config
from collie.trainer.deepspeed_pipeline_trainer import PipelineTrainer
from collie.arguments import CollieArguments

from process import load_alpaca, Collator
from utils import get_wandb_name

def loss_fn(logits, label, clip_loss_value):
    # logits is a tuple when use_cache=True
    if not isinstance(logits, torch.Tensor):
        logits = logits[0]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = label[:, 1:].contiguous()
    # Flatten the tokens
    if clip_loss_value is not None:
        loss_fct = CrossEntropyLoss(reduction='none', ignore_index=0)
        loss = loss_fct(shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
                        shift_labels.view(-1).cuda())
        loss.data.clamp_(min=-clip_loss_value, max=clip_loss_value)
        loss = loss.mean()
    else:
        loss_fct = CrossEntropyLoss(ignore_index=0)
        loss = loss_fct(shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
                        shift_labels.view(-1).cuda())

    return loss

def compute_metrics(all_preds, dataset, output_file):
    with open(output_file, "a") as fp:
        for i in range(len(all_preds)):
            fp.write("------------------------pred------------------------\n")
            fp.write(all_preds[i] + "\n")
            fp.write("-----------------------output-----------------------\n")
            fp.write(dataset.alpaca[i]["output"] + "\n")
            fp.write("====================================================\n")

    return {"acc": 1}

if __name__ == "__main__":
    # CUDA_LAUNCH_BLOCKING=1 bash run.sh
    collie_args = CollieArguments(
        output_dir="output",
        deepspeed="ds_config.json",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        evaluation_strategy="epoch",
        eval_steps=1,
        metric_for_best_model="acc",
        learning_rate=0.01,
        report_to="wandb"
    )
    collie_args.max_new_tokens = 256
    collie_args.num_stages = int(os.environ["WORLD_SIZE"])

    collie_args.save_path = "hdd:s3://opennlplab_hdd/xsh/350M-test"
    collie_args.save_protocol = "s3"
    
    collie_args.load_path = "/mnt/petrelfs/xingshuhao.dispatch/.cache/huggingface/hub/models--Salesforce--codegen-350M-mono/snapshots/40b7a3b6e99e73bdb497a14b740e7167b3413c74"
    collie_args.load_protocol = "file"
    collie_args.tokenizer_path = "Salesforce/codegen-16B-mono"

    ### dataset
    collie_args.data_num = 100
    collie_args.test_size = 0.06
    max_len = 256

    today = datetime.datetime.today()
    # y-m-d
    cur_date = str(today.date())
    # h:m:s.ms
    cur_time = str(today.time())

    config = get_codgen_config(
        collie_args.load_path, protocol=collie_args.load_protocol
    )
    config.gradient_checkpointing = True

    codegen_pipeline, tokenizer = get_codegen_pipeline(
        collie_args.load_path, config=config, one_by_one=True,
        num_stages=collie_args.num_stages, protocol=collie_args.load_protocol,
        tokenizer_path=collie_args.tokenizer_path,
        loss_fn=lambda logits, label: loss_fn(
            logits, label, collie_args.clip_loss_value
        ),
    )
    train_dataset, eval_dataset = load_alpaca(tokenizer, max_len,
                                              num=collie_args.data_num,
                                              test_size=collie_args.test_size)

    optimizer = torch.optim.SGD(codegen_pipeline.parameters(), lr=collie_args.learning_rate)
    lr_scheduler = None
    trainer = PipelineTrainer(
        model=codegen_pipeline,
        collie_args=collie_args,
        data_collator={
            "train": Collator(tokenizer, True),
            "eval": Collator(tokenizer, False)
        },
        compute_metrics=lambda p, d: compute_metrics(p, d, f"output-{cur_date}-{cur_time}"),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        # params for initialize pipeline
    )

    # init wandb
    name = get_wandb_name(trainer, optimizer)
    if trainer.allow_print and collie_args.report_to == "wandb":
        trainer.wandb.init(
            project=f"collie-{cur_date}-moss_pipeline",
            name=f"{cur_time}-{name}",
        )
    trainer.train()

    trainer.save(collie_args.save_path, collie_args.save_protocol)
    trainer.eval(
        trainer.global_step, collie_args.num_train_epochs,
        trainer.eval_dataset, trainer.eval_dataloader, 'eval'
    )
