"""
一个使用CoLLie对LLaMA基座进行Prompt tuning的实例（只支持张量并行+数据并行）。
"""
import os
import sys
sys.path.append('../../')
sys.path.append("/mnt/petrelfs/gutianle/Megatron-LM/")
import json
import torch

from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig
from peft import get_peft_model

from collie.config import CollieConfig

from collie.data import CollieDatasetForTraining
from collie.data import CollieDataLoader

from collie.controller.trainer import Trainer
from collie.controller.evaluator import EvaluatorForPerplexity, EvaluatorForGeneration

from collie.models.llama.model import LlamaForCausalLM
from collie.utils.dist_utils import setup_distribution

from collie.utils.monitor import StepTimeMonitor, TGSMonitor, MemoryMonitor, LossMonitor, EvalMonitor
from collie.metrics import DecodeMetric, PPLMetric, BleuMetric
from collie.module import GPTLMLoss

from datasets import load_dataset

from peft import (
    get_peft_config,
    get_peft_model,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    PromptEncoderConfig,
    PeftType
)

# 1. 设置路径
# 1.1 预训练模型路径
pretrained_model = "decapoda-research/llama-7b-hf"
# 1.2 Eval的decode结果保存路径
save_path = './result/'

# 2. 设置配置
# 2.1 加载配置
config = CollieConfig.from_pretrained(pretrained_model)

# 2.2 添加配置
# config.tp_size = 2
# config.dp_size = 2
config.pp_size = 4
config.train_epochs = 1
config.train_micro_batch_size = 1
config.eval_batch_size = 32
config.eval_per_n_steps = 100
config.checkpointing = False
# 2.3 添加Prompt Tuning配置
config.peft_config = PromptTuningConfig(
    task_type = TaskType.CAUSAL_LM,
    prompt_tuning_init = PromptTuningInit.TEXT,
    num_virtual_tokens = 8,
    token_dim = 4096,
    num_attention_heads = 32,
    num_layers = 32,
    num_transformer_submodules = None,
    prompt_tuning_init_text="Classify if the comment is positive or negative.",
    tokenizer_name_or_path=pretrained_model
)
config.ds_config = {
    "fp16": {"enabled": True},
    # "monitor_config": {
    #     "enabled": True,
    #     "tag": "sophia_alpaca",
    #     "csv_monitor": {
    #         "enabled": True,
    #         "output_path": "./ds_logs/"
    #     }
    # }
}

# 3. 设置tokenizer
tokenizer = LlamaTokenizer.from_pretrained(pretrained_model, padding_side="left")

# 4. 加载数据集
train_dataset = [
    {
        "input": f"{sample['text']}.",
        "output": "positive" if sample["label"] else "negative."
    } for sample in load_dataset("imdb", split="train")
]
train_dataset = CollieDatasetForTraining(train_dataset, tokenizer)[:-32]
eval_dataset = train_dataset[:32]

# 5. 加载预训练模型
model = LlamaForCausalLM.from_pretrained(pretrained_model, config, use_cache=True)

# 6. 设置优化器
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

# 7. 添加监视器
monitors = [
    StepTimeMonitor(config),
    TGSMonitor(config),
    MemoryMonitor(config),
    LossMonitor(config),
    EvalMonitor(config)
]

# 8. 添加Evaluator
evaluator_ppl = EvaluatorForPerplexity(
    model = model,
    config = config,
    dataset = eval_dataset,
    monitors = [
        EvalMonitor(config)
    ],
    metrics = {
        'ppl': PPLMetric()
    }
)
evaluator_decode = EvaluatorForGeneration(
    model = model,
    config = config,
    tokenizer = tokenizer,
    dataset = eval_dataset,
    monitors = [
        EvalMonitor(config)
    ],
    metrics = {
        'decode': DecodeMetric(save_to_file = True, save_path = save_path)
    }

)

# 9. 实例化trainer
trainer = Trainer(
    model = model,
    config = config,
    loss_fn = GPTLMLoss(-100),
    optimizer = optimizer,
    train_dataset = train_dataset,
    monitors = monitors,
    evaluators = [evaluator_ppl, evaluator_decode]
)

# 10. 训练/验证
trainer.train()

#  Command CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=4 finetune_llama_prompt_tuning.py