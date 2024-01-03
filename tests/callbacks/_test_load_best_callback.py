import sys
sys.path.append("../../")
import os
import shutil
import traceback

from transformers import AutoTokenizer
from collie import models
from collie import Trainer, CollieConfig
from collie.module import GPTLMLoss
from collie.callbacks import LoadBestModelCallback
from collie.metrics import BaseMetric
from collie.utils import env
from collie.log import logger

from tests.helpers import create_ds_config, import_class

class SFTAccMetric(BaseMetric):
    def __init__(self):
        super(SFTAccMetric, self).__init__(gather_result=True)
        self.right = 0
        self.total = 0

    def reset(self):
        self.right = 0
        self.total = 0

    def update(self, result):
        """

        :param result: dict. Gathered result of eval_fn. Contains `right`,
            `total` in this case.
        """
        self.right += result["right"].sum().item()
        self.total += result["total"].sum().item()

    def get_metric(self):
        acc = self.right / self.total
        return {"acc": acc}

def test_load_best_callback(model_type, model_path, folder, dp_size,
                            tp_size, pp_size, zero):
    if pp_size > 1:
        assert zero <= 1
    ds_config = create_ds_config(zero=zero, offload=True, optimizer="AdamW", lr=2e-5)
    try:
        def eval_fn(trainer, batch):
            # batch: dict
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            # forward
            if env.pp_size > 1:
                trainer.engine.module.forward_type = "eval"
                logits = trainer.engine.module(**batch)["logits"]
                trainer.engine.module.forward_type = "train"
            else:
                logits = trainer.engine(input_ids=input_ids.cuda()).logits
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:].to(logits.device)
            right = (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum()
            total = (shift_labels != -100).sum()
            return {
                "total": total.view(1,),
                "right": right.view(1,),
            }
        config = CollieConfig.from_pretrained(
            model_path, tp_size=tp_size, dp_size=dp_size, pp_size=pp_size, 
            train_epochs=1, eval_per_n_steps=2, eval_per_n_epochs=0,
            train_micro_batch_size=2, gradient_accumulation_steps=2,
            eval_batch_size=1, ds_config=ds_config, trust_remote_code=True
        )
        # tokenizer and dataset
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        train_sample = tokenizer("Collie is a python package for finetuning large language models.", return_tensors="pt").input_ids.squeeze(0)
        eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)[:-1,]
        train_dataset = [{"input_ids": train_sample, "labels": train_sample} for _ in range(400)]
        eval_dataset = [{"input_ids": eval_sample, "labels": eval_sample}]

        model_cls = import_class(model_type)
        model = model_cls.from_pretrained(model_path, config=config)

        callbacks = [LoadBestModelCallback(folder, monitor="acc_metric#acc")]
        metrics = {"acc_metric": SFTAccMetric()}
        trainer = Trainer(
            model, config, loss_fn=GPTLMLoss(-100),
            train_dataset=train_dataset, callbacks=callbacks,
            eval_dataset=eval_dataset, eval_fn=eval_fn,
            metrics=metrics
        )
        trainer.train()
    finally:
        if os.path.exists(folder):
            logger.info(f"folders in checkpoint {folder}/:\n{os.listdir(folder)}")
            if env.rank == 0:
                shutil.rmtree(folder)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="_ckpt", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--dp_size", type=int)
    parser.add_argument("--tp_size", type=int)
    parser.add_argument("--pp_size", type=int)
    parser.add_argument("--zero", default=1, type=int)
    args = parser.parse_args()
    test_load_best_callback(
        model_type=args.model_type, model_path=args.model_path,
        folder=args.folder, dp_size=args.dp_size, tp_size=args.tp_size,
        pp_size=args.pp_size, zero=args.zero
    )
