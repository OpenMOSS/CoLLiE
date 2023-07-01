"""
一个使用CoLLie对LLaMA基座进行全参量Instruct Tuning，从而得到Alpaca的实例。
"""
import sys
sys.path.append('../../')
sys.path.append("/mnt/petrelfs/gutianle/Megatron-LM/")
import json
import torch

from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig

from collie.config import CollieConfig

from collie.data import CollieDatasetForTraining
from collie.data import CollieDataLoader

from collie.controller.trainer import Trainer
from collie.controller.evaluator import EvaluatorForPerplexity, EvaluatorForGeneration

from collie.models.llama.model import LlamaForCausalLM
from collie.utils.monitor import StepTimeMonitor, TGSMonitor, MemoryMonitor, LossMonitor, EvalMonitor
from collie.metrics import DecodeMetric, PPLMetric, BleuMetric
from collie.module import GPTLMLoss

# 1. 设置路径
# 1.1 预训练模型路径
pretrained_model = 'decapoda-research/llama-7b-hf'
# 1.2 数据集路径
data_path = 'alpaca.json'
# 1.3 Eval的decode结果保存路径
save_path = './result'

# 2. 设置配置
# 2.1 加载配置
config = CollieConfig.from_pretrained(pretrained_model)
# 2.2 添加配置
config.tp_size = 2
config.dp_size = 2
config.pp_size = 1
config.train_epochs = 1
config.train_micro_batch_size = 8
config.eval_batch_size = 32
config.eval_per_n_steps = 100
config.ds_config = {
    "fp16": {"enabled": True},
    "monitor_config": {
        "enabled": True,
        "tag": "sophia_alpaca",
        "csv_monitor": {
            "enabled": True,
            "output_path": "./ds_logs/"
        }
    }
}

# 3. 设置tokenizer
tokenizer = LlamaTokenizer.from_pretrained(pretrained_model, padding_side="left")

# 4. 加载数据集
dataset = CollieDatasetForTraining.from_json(data_path, tokenizer=tokenizer)
train_dataset = dataset[:-32]
eval_dataset = dataset[-32:]

# 5. 加载预训练模型
model = LlamaForCausalLM(config)
state_dict = LlamaForCausalLM.load_parallel_state_dict(
    path="hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf",
    config=config,
    protocol="petrel",
    format="hf"
)
model.load_state_dict(state_dict)

# 6. 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

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

#  Command CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=4 train.py