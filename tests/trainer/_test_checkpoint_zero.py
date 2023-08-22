"""
测试使用 CollieForCausalLM 多维并行且使用 zero 1情况下 checkpoint 的保存和加载是否正确。
"""
import sys
sys.path.append("../../")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from collie.models.moss_moon import Moss003MoonForCausalLM
from collie.module import GPTLMLoss
from collie.config import CollieConfig
from collie.data import CollieDataLoader
from collie.log import logger
from collie.utils import env, setup_distribution, ColliePadder
from collie.controller import Trainer
from collie.optim import Lomo
from tests.helpers import create_ds_config, import_class

def compare(d1, d2, key=""):
    assert type(d1) == type(d2), f"Key: {key}, {type(d1)} vs {type(d2)}"
    if isinstance(d1, dict):
        for key in d1.keys():
            assert key in d2, key
            compare(d1[key], d2[key], key)
    elif isinstance(d1, (list, tuple)):
        assert len(d1) == len(d2), f"Key: {key}, {len(d1)} vs {len(d2)}"
        for idx in range(len(d1)):
            compare(d1[idx], d2[idx])
    elif isinstance(d1, (str, int, float)):
        assert d1 == d2, \
            f"Key: {key}, d1: {d1}, d2:{d2}"
    elif isinstance(d1, torch.Tensor):
        assert torch.allclose(d1, d2), \
            f"Key: {key}, d1: {d1}, d2:{d2}"

def save_checkpoint(trainer, eval_sample, config, ckpt_dir, format):
    """
    保存 checkpoint，同时将 optimizer state 和 eval_sample 的 logits 
    分别保存在 zero_save_dp{}_tp{}_pp{} 和 logits 文件中。
    """
    logger.info("Save checkpoint")
    trainer.save_checkpoint(ckpt_dir)
    # optimizer 和 logits
    if trainer.optimizer is not None:
        state_dict_before = trainer.optimizer.state_dict()
    else:
        state_dict_before = trainer.engine.optimizer.state_dict()
    client_state = {}
    eval_dataset = [{
        "input_ids": eval_sample,
        "labels": eval_sample
    }]
    dataloader = CollieDataLoader(
        eval_dataset, config.eval_batch_size,
        config.gradient_accumulation_steps, shuffle=False,
        collate_fn=ColliePadder(), num_workers=0
    )
    trainer.engine.eval()
    if config.pp_size == 1:
        client_state["logits"] = trainer.engine(eval_sample.unsqueeze(0).cuda()).logits.detach().cpu()[0]
    else:
        # accu * evalbatch_size, seqlen(2), vocabsize
        # 所以取第一个
        trainer.engine.module.forward_type = "eval"
        client_state["logits"] = trainer.engine.module(**next(iter(dataloader)))["logits"].detach().cpu()[0]
        trainer.engine.module.forward_type = "train"
    client_state["format"] = format
    torch.save(state_dict_before, f"zero_save_dp{env.dp_rank}_tp{env.tp_rank}_pp{env.pp_rank}")
    if env.rank == 0:
        torch.save(client_state, f"logits")

    env.barrier()

def check_weights(trainer, config, ckpt_dir, eval_sample, hf_model):
    """
    只加载 ckpt_dir 的权重，检查 eval_sample 的 logits是否一致。
    """
    logger.info("Check model weights")
    env.barrier()
    client_state = torch.load("logits")
    logits_before = client_state["logits"]
    # huggingface 模型
    hf_model.eval()
    hf_logits = hf_model(eval_sample.cuda()).logits.detach().cpu().to(logits_before)
    if not torch.allclose(hf_logits, logits_before, 0, 1e-2):
        logger.info(
            f"The logits from {ckpt_dir} is not equal to logits from "
            f"huggingface model: {logits_before} \n vs \n {hf_logits}"
        )
    # collie 包裹的模型
    trainer.load_model(ckpt_dir)
    eval_dataset = [{
        "input_ids": eval_sample,
        "labels": eval_sample
    }]
    dataloader = CollieDataLoader(
        eval_dataset, config.eval_batch_size,
        config.gradient_accumulation_steps, shuffle=False,
        collate_fn=ColliePadder(), num_workers=0
    )
    trainer.engine.eval()
    if config.pp_size == 1:
        logits = trainer.engine(eval_sample.unsqueeze(0).cuda()).logits.detach().cpu()[0]
    else:
        # accu * evalbatch_size, seqlen(2), vocabsize
        # 所以取第一个
        trainer.engine.module.forward_type = "eval"
        logits = trainer.engine.module(**next(iter(dataloader)))["logits"].detach().cpu()[0]
        trainer.engine.module.forward_type = "train"
    logits = logits.to(logits_before)
    if not torch.allclose(logits, logits_before, 0, 1e-2):
        logger.info(
            f"The logits from {ckpt_dir} is not equal to logits from "
            f"loaded collie model: {logits_before} \n vs \n {logits}"
        )

def check_checkpoint(trainer, config, ckpt_dir):
    """
    加载 checkpoint，并检查 optimizer。

    注意前后并行设置必须一样。
    """
    logger.info("Check optimizer state")
    trainer.load_checkpoint(ckpt_dir)
    env.barrier()
    # TODO 不同tp的optimizer会不一致
    if trainer.optimizer is not None:
        state_dict = trainer.optimizer.state_dict()
    else:
        state_dict = trainer.engine.optimizer.state_dict()
    state_dict_before = torch.load(f"zero_save_dp{env.dp_rank}_tp{env.tp_rank}_pp{env.pp_rank}")
    # 测试 optimizer state_dict
    compare(state_dict_before, state_dict)

def test_checkpoint(model_type, pretrained_model, ckpt_dir, zero_stage,
                    format, load, dp_size, tp_size, pp_size, lomo):
    """
    模型保存后再加载进行验证

    :param pretrained_model:
    :param ckpt_dir:
    :param format: collie & hf。
    :param load: 是否进行加载。如果加载出来的 format 和当前的 format 不同，则只会
        比较权重的正确性；否则会额外比较 optimizer state dict
    """
    # Collie Configuration
    if format == "hf":
        dp_size *= tp_size * pp_size
        tp_size = 1
        pp_size = 1
    if pp_size != 1:
        zero_stage = 1

    # 初始化
    if load:
        # 加载 huggingface
        hf_model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, trust_remote_code=True, torch_dtype=torch.float16
        )
    if lomo:
        ds_config = create_ds_config(fp16=True, zero=zero_stage, offload=True, optimizer=None)
    else:
        ds_config = create_ds_config(fp16=True, zero=zero_stage, offload=True, optimizer="Adam", lr=2e-5)
    config = CollieConfig.from_pretrained(
        pretrained_model, tp_size=tp_size, dp_size=dp_size, pp_size=pp_size, 
        train_epochs=1, eval_per_n_steps=0, eval_per_n_epochs=0,
        train_micro_batch_size=2, gradient_accumulation_steps=4,
        eval_batch_size=1, ds_config=ds_config, trust_remote_code=True,
        use_flash=False
    )
    # tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    train_sample = tokenizer("Collie is a python package for finetuning large language models.", return_tensors="pt").input_ids.squeeze(0)
    eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)[:-1,]
    train_dataset = [{"input_ids": train_sample, "labels": train_sample} for _ in range(200)]

    if format == "collie":
        model = import_class(model_type).from_pretrained(pretrained_model, config=config)
    elif format == "hf":
        setup_distribution(config)
        model = AutoModelForCausalLM.from_pretrained(pretrained_model)
    else:
        raise NotImplementedError

    if lomo:
        optimizer = Lomo(model, lr=2e-5, clip_grad_norm=1.0)
    else:
        optimizer = None

    trainer = Trainer(
        model, config, loss_fn=GPTLMLoss(-100),
        train_dataset=train_dataset, optimizer=optimizer
    )
    
    if not load:
        trainer.train()
        save_checkpoint(trainer, eval_sample, config, ckpt_dir, format)
    else:
        hf_model = hf_model.cuda()
        # 加载
        check_weights(trainer, config, ckpt_dir, eval_sample, hf_model)
        del hf_model
        env.barrier()
        client_state = torch.load("logits")
        format_before = client_state["format"]
        if format == format_before:
            check_checkpoint(trainer, config, ckpt_dir)

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
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--format", default=1, type=str)
    parser.add_argument("--lomo", action="store_true")
    args = parser.parse_args()

    test_checkpoint(args.model_type, args.model_path, args.folder, args.zero,
                    dp_size=args.dp_size, tp_size=args.tp_size,
                    pp_size=args.pp_size, format=args.format, load=args.load,
                    lomo=args.lomo)