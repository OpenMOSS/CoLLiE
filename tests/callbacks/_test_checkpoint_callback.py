import sys
sys.path.append("../../")
import os
import shutil
import traceback

from transformers import AutoTokenizer
from collie import Trainer, CollieConfig
from collie.module import GPTLMLoss
from collie.callbacks import CheckpointCallback
from collie.models import Moss003MoonForCausalLM
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

def check_and_load(trainer, folder, subfolder, model_only):
    path = os.path.join(folder, subfolder)
    assert os.path.exists(path), path
    if model_only:
        trainer.load_model(path)
    else:
        trainer.load_checkpoint(path)

def test_checkpoint_callback(pretrained_model, model_only, folder,
                             dp_size, tp_size, pp_size):
    try:
        config = CollieConfig.from_pretrained(
            pretrained_model, tp_size=tp_size, dp_size=dp_size, pp_size=pp_size, 
            train_epochs=5, eval_per_n_steps=0, eval_per_n_epochs=0,
            train_micro_batch_size=2, gradient_accumulation_steps=2,
            eval_batch_size=1, ds_config=DS_CONFIG, trust_remote_code=True
        )
        # tokenizer and dataset
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
        train_sample = tokenizer("Collie is a python package for finetuning large language models.", return_tensors="pt").input_ids.squeeze(0)
        train_dataset = [{"input_ids": train_sample, "labels": train_sample} for _ in range(100)]

        model = Moss003MoonForCausalLM.from_pretrained(pretrained_model, config=config)

        every_n_epochs = 2
        every_n_batches = 10
        last = True
        max = 3
        callbacks = [CheckpointCallback(folder, every_n_epochs=every_n_epochs,
                                        every_n_batches=every_n_batches,
                                        last=last, model_only=model_only,
                                        max=max)]
        trainer = Trainer(
            model, config, loss_fn=GPTLMLoss(-100),
            train_dataset=train_dataset, callbacks=callbacks
        )
        trainer.train()
        assert os.path.exists(folder)
        ckpts = []
        for epoch in range(config.train_epochs):
            if (epoch + 1) % every_n_epochs == 0:
                ckpts.append(f"epoch_{epoch + 1}")
            for n in range(trainer.steps_per_epoch // every_n_batches):
                ckpts.append(f"epoch_{epoch}-batch_{(n + 1)*every_n_batches}")
        if last:
            check_and_load(trainer, folder, "last", model_only)
        if max is not None and max > 0:
            for folder_name in ckpts[:max]:
                assert not os.path.exists(os.path.join(folder, folder_name))
            ckpts = ckpts[-max:]
            for folder_name in ckpts:
                assert os.path.exists(os.path.join(folder, folder_name))
        for folder_name in ckpts:
            check_and_load(trainer, folder, folder_name, model_only)
    except Exception as e:
        logger.error(traceback.format_exc())
    finally:
        if os.path.exists(folder):
            logger.info(f"folders in checkpoint {folder}/:\n{os.listdir(folder)}")
            if env.rank == 0:
                shutil.rmtree(folder)


if __name__ == "__main__":
    pretrained_model = "/mnt/petrelfs/xingshuhao.dispatch/.cache/huggingface/hub/models--Salesforce--codegen-350M-mono/snapshots/40b7a3b6e99e73bdb497a14b740e7167b3413c74"
    test_checkpoint_callback(pretrained_model, model_only=True,
                                folder="_ckpt", dp_size=2, tp_size=1,
                                pp_size=2)
