import sys
sys.path.append("../../")
import os
import shutil
import traceback

from transformers import AutoTokenizer
from collie import Trainer, CollieConfig
from collie.module import GPTLMLoss
from collie.callbacks import LoadBestModelCallback
from collie.metrics import BaseMetric
from collie.models import MossForCausalLM
from collie.utils import env
from collie.log import logger

DS_CONFIG = {
    "fp16": {
        "enabled": True
    },
    "zero_allow_untested_optimizer": True,
    "zero_force_ds_cpu_optimizer": False,

    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 2e-5,
            "weight_decay": 0.1
        }
    },

    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": False
        }
    },
    "steps_per_print": 2000,
}

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
        self.right += sum(result["right"]).cpu().item()
        self.total += sum(result["total"]).cpu().item()

    def get_metric(self):
        acc = self.right / self.total
        return acc

def test_load_best_callback(pretrained_model, folder, dp_size, tp_size,
                             pp_size):
    try:
        def eval_fn(trainer, batch):
            # batch: tuple
            input_ids, labels = batch
            # forward
            if env.pp_size > 1:
                logits = trainer.engine.eval_batch(batch)
            else:
                logits = trainer.engine(input_ids=input_ids.cuda()).logits
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:].to(logits.device)
            right = (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum()
            total = (shift_labels != -100).sum()
            return {
                "total": total,
                "right": right,
            }
        config = CollieConfig.from_pretrained(
            pretrained_model, tp_size=tp_size, dp_size=dp_size, pp_size=pp_size, 
            train_epochs=1, eval_per_n_steps=2, eval_per_n_epochs=0,
            train_micro_batch_size=2, gradient_accumulation_steps=2,
            eval_batch_size=1, ds_config=DS_CONFIG, trust_remote_code=True
        )
        # tokenizer and dataset
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
        train_sample = tokenizer("Collie is a python package for finetuning large language models.", return_tensors="pt").input_ids.squeeze(0)
        eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)[:-1,]
        train_dataset = [(train_sample, train_sample) for _ in range(500)]
        eval_dataset = [(eval_sample, eval_sample)]

        model = MossForCausalLM.from_pretrained(pretrained_model, config=config)

        callbacks = [LoadBestModelCallback(folder, monitor="acc")]
        metrics = {"acc": SFTAccMetric()}
        trainer = Trainer(
            model, config, loss_fn=GPTLMLoss(-100),
            train_dataset=train_dataset, callbacks=callbacks,
            eval_dataset=eval_dataset, eval_fn=eval_fn,
            metrics=metrics
        )
        trainer.train()
    except Exception as e:
        logger.error(traceback.format_exc())
    finally:
        if os.path.exists(folder):
            logger.info(f"folders in checkpoint {folder}/:\n{os.listdir(folder)}")
            if env.rank == 0:
                shutil.rmtree(folder)


if __name__ == "__main__":
    pretrained_model = "/mnt/petrelfs/xingshuhao.dispatch/.cache/huggingface/hub/models--Salesforce--codegen-350M-mono/snapshots/40b7a3b6e99e73bdb497a14b740e7167b3413c74"
    test_load_best_callback(pretrained_model, folder="_ckpt", dp_size=2,
                             tp_size=1, pp_size=2)
