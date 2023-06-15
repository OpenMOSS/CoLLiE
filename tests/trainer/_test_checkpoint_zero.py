"""
测试使用 CollieForCausalLM 多维并行且使用 zero 1情况下 checkpoint 的保存和加载是否正确。
"""
import sys
sys.path.append("../../")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from collie.models.moss import MossForCausalLM
from collie.module import GPTLMLoss
from collie.config import CollieConfig
from collie.data import CollieDataLoader
from collie.log import logger
from collie.utils import env, setup_distribution
from collie.controller import Trainer

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
        # "offload_optimizer": {
        #     "device": "cpu",
        #     "pin_memory": False
        # }
    },
    "steps_per_print": 2000,
}

def init(pretrained_model, format, dp_size, tp_size, pp_size):
    config = CollieConfig.from_pretrained(
        pretrained_model, tp_size=tp_size, dp_size=dp_size, pp_size=pp_size, 
        train_epochs=1, eval_per_n_steps=0, eval_per_n_epochs=0,
        train_micro_batch_size=2, gradient_accumulation_steps=4,
        eval_batch_size=1, ds_config=DS_CONFIG, trust_remote_code=True
    )
    # tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    train_sample = tokenizer("Collie is a python package for finetuning large language models.", return_tensors="pt").input_ids.squeeze(0)
    eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)[:-1,]
    train_dataset = [(train_sample, train_sample) for _ in range(400)]

    if format == "collie":
        model = MossForCausalLM.from_pretrained(pretrained_model, config=config)
    elif format == "hf":
        setup_distribution(config)
        model = AutoModelForCausalLM.from_pretrained(pretrained_model)
    else:
        raise NotImplementedError

    trainer = Trainer(
        model, config, loss_fn=GPTLMLoss(-100),
        train_dataset=train_dataset
    )
    return config, trainer, eval_sample

def save_checkpoint(trainer, eval_sample, config, ckpt_dir, format):
    """
    保存 checkpoint，同时将 optimizer state 和 eval_sample 的 logits 
    分别保存在 zero_save_dp{}_tp{}_pp{} 和 logits 文件中。
    """
    logger.info("Save checkpoint")
    trainer.save_checkpoint(ckpt_dir)
    # optimizer 和 logits
    state_dict_before = trainer.engine.optimizer.state_dict()
    client_state = {}
    dataloader = CollieDataLoader(
        [(eval_sample, eval_sample)], config.eval_batch_size,
        config.gradient_accumulation_steps, shuffle=False,
    )
    trainer.engine.eval()
    if config.pp_size == 1:
        client_state["logits"] = trainer.engine(eval_sample.unsqueeze(0).cuda()).logits.detach().cpu()[0]
    else:
        # accu * evalbatch_size, seqlen(2), vocabsize
        # 所以取第一个
        client_state["logits"] = trainer.engine.eval_batch(next(iter(dataloader))).detach().cpu()[0]
    client_state["format"] = format
    torch.save(state_dict_before, f"zero_save_dp{env.dp_rank}_tp{env.tp_rank}_pp{env.pp_rank}")
    if env.rank == 0:
        torch.save(client_state, f"logits")

    env.barrier()

def check_weights(trainer, config, ckpt_dir, eval_sample):
    """
    只加载 ckpt_dir 的权重，检查 eval_sample 的 logits是否一致。
    """
    logger.info("Check model weights")
    client_state = torch.load("logits")
    logits_before = client_state["logits"]
    # huggingface 模型
    hf_model = AutoModelForCausalLM.from_pretrained(ckpt_dir).cuda()
    hf_logits = hf_model(eval_sample.cuda()).logits.detach().cpu()
    if not torch.allclose(hf_logits.to(logits_before), logits_before, 0, 1e-4):
        logger.info(
            f"The logits from {ckpt_dir} is not equal to logits from "
            f"huggingface model: {logits_before} \n vs \n {hf_logits}"
        )
    del hf_model
    # collie 包裹的模型
    trainer.load_model(ckpt_dir)
    dataloader = CollieDataLoader(
        [(eval_sample, eval_sample)], config.eval_batch_size,
        config.gradient_accumulation_steps, shuffle=False,
    )
    trainer.engine.eval()
    if config.pp_size == 1:
        logits = trainer.engine(eval_sample.unsqueeze(0).cuda()).logits.detach().cpu()[0]
    else:
        # accu * evalbatch_size, seqlen(2), vocabsize
        # 所以取第一个
        logits = trainer.engine.eval_batch(next(iter(dataloader))).detach().cpu()[0]
    if not torch.allclose(logits.to(logits_before), logits_before, 0, 1e-4):
        logger.info(
            f"The logits from {ckpt_dir} is not equal to logits from "
            f"loaded collie model: {logits_before} \n vs \n {hf_logits}"
        )

def check_checkpoint(trainer, config, ckpt_dir):
    """
    加载 checkpoint，并检查 optimizer。

    注意前后并行设置必须一样。
    """
    logger.info("Check optimizer state")
    trainer.load_checkpoint(ckpt_dir)
    env.barrier()
    state_dict = trainer.engine.optimizer.state_dict()
    state_dict_before = torch.load(f"zero_save_dp{env.dp_rank}_tp{env.tp_rank}_pp{env.pp_rank}")
    # 测试 optimizer state_dict
    compare(state_dict_before, state_dict)

def test_checkpoint(pretrained_model, ckpt_dir, zero_stage, format, load, dp_size, tp_size, pp_size):
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
    DS_CONFIG["zero_optimization"]["stage"] = zero_stage
    config, trainer, eval_sample = init(pretrained_model, format, dp_size,
                                        tp_size, pp_size)
    
    if not load:
        # trainer.train()
        save_checkpoint(trainer, eval_sample, config, ckpt_dir, format)
    else:
        # 加载
        check_weights(trainer, config, ckpt_dir, eval_sample)
        env.barrier()
        client_state = torch.load("logits")
        format_before = client_state["format"]
        if format == format_before:
            check_checkpoint(trainer, config, ckpt_dir)

if __name__ == "__main__":
    # pretrained_model = "Salesforce/codegen-350M-mono"
    pretrained_model = "/mnt/petrelfs/xingshuhao.dispatch/.cache/huggingface/hub/models--Salesforce--codegen-350M-mono/snapshots/40b7a3b6e99e73bdb497a14b740e7167b3413c74"
    ckpt_dir = "3dckpt"
    dp_size = 2
    tp_size = 2
    pp_size = 1
    zero_stage = 3
    ## 用 collie 的模型保存
    # kwargs = dict(format="collie", load=False)
    ## 用 hf 的模型传入 collie 保存
    kwargs = dict(format="hf", load=False)
    ## 用 collie 的模型加载
    kwargs = dict(format="collie", load=True)
    ## 用 hf 模型传入 collie 加载
    # kwargs = dict(format="hf", load=True)

    test_checkpoint(pretrained_model, ckpt_dir, zero_stage, dp_size=dp_size,
                    tp_size=tp_size, pp_size=pp_size, **kwargs)