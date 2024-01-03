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
from tests.helpers import create_ds_config, import_class

def check_and_load(trainer, folder, subfolder, model_only):
    path = os.path.join(folder, subfolder)
    assert os.path.exists(path), path
    if model_only:
        trainer.load_model(path)
    else:
        trainer.load_checkpoint(path)

def test_checkpoint_callback(model_type, model_path, folder, model_only,
                             dp_size, tp_size, pp_size, zero):
    try:
        ds_config = create_ds_config(fp16=True, zero=zero, offload=True, optimizer="Adam", lr=2e-5)
        config = CollieConfig.from_pretrained(
            model_path, tp_size=tp_size, dp_size=dp_size, pp_size=pp_size, 
            train_epochs=5, eval_per_n_steps=0, eval_per_n_epochs=0,
            train_micro_batch_size=2, gradient_accumulation_steps=2,
            eval_batch_size=1, ds_config=ds_config, trust_remote_code=True
        )
        # tokenizer and dataset
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        train_sample = tokenizer("Collie is a python package for finetuning large language models.", return_tensors="pt").input_ids.squeeze(0)
        train_dataset = [{"input_ids": train_sample, "labels": train_sample} for _ in range(100)]

        model_cls = import_class(model_type)
        model = model_cls.from_pretrained(model_path, config=config)

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
            for n in range(trainer.steps_per_epoch // every_n_batches):
                ckpts.append(f"epoch_{epoch}-batch_{(n + 1)*every_n_batches}")
            if (epoch + 1) % every_n_epochs == 0:
                ckpts.append(f"epoch_{epoch + 1}")
        if last:
            check_and_load(trainer, folder, "last", model_only)
        print(ckpts)
        if max is not None and max > 0:
            for folder_name in ckpts[:max]:
                assert not os.path.exists(os.path.join(folder, folder_name)), folder_name
            ckpts = ckpts[-max:]
            for folder_name in ckpts:
                assert os.path.exists(os.path.join(folder, folder_name)), folder_name
        for folder_name in ckpts:
            check_and_load(trainer, folder, folder_name, model_only)
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
    parser.add_argument("--model_only", action="store_true")
    parser.add_argument("--dp_size", type=int)
    parser.add_argument("--tp_size", type=int)
    parser.add_argument("--pp_size", type=int)
    parser.add_argument("--zero", default=1, type=int)
    args = parser.parse_args()
    test_checkpoint_callback(args.model_type, args.model_path, folder="_ckpt",
                             model_only=args.model_only, dp_size=args.dp_size,
                             tp_size=args.tp_size, pp_size=args.pp_size,
                             zero=args.zero)
