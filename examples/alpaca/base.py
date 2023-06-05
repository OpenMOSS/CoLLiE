# Command CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=8 base_alpaca.py
import sys
sys.path.append('../../')
sys.path.append("/mnt/petrelfs/gutianle/Megatron-LM/")
import os

import torch
import torch.distributed as dist
import torch.optim as optim
from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig

from collie.models.llama.model import LlamaForCausalLM
from collie.metrics.decode import DecodeMetric
from collie.module import GPTLMLoss, PipelineGenerationMixin
from collie.trainer.trainer import Trainer
from collie.log import logger
from collie.utils import env, setup_distribution
from collie.utils.monitor import StepTimeMonitor, TGSMonitor, MemoryMonitor, LossMonitor, EvalMonitor
from collie.optim.lion import Lion

from alpaca_metric import AlpacaDecodeMetric
from alpaca import AlpacaDataset, train_collate_fn, eval_collate_fn


class BaseAlpaca:
    """
    这是一个基础的BaseAlpaca类，调用即可基于LLaMa训练一个Alpaca。
    
    :param config: 一个CollieConfig实例，可以设置并行相关配置
    """
    def __init__(self, config) -> None:
        # CollieConfig
        self.config = config
        # generationConfig
        self.generationConfig = self._setGenerationConfig()
    
    # 设置生成Config
    def _setGenerationConfig(self):
        generate_config = GenerationConfig(
            max_new_tokens=128,
            eos_token_id=2,
            pad_token_id=0,
            bos_token_ids=1
        )
        return generate_config
    
    def eval_fn(self, trainer, batch):
        input_ids, labels = batch
        if env.pp_size > 1:
            generation_model = PipelineGenerationMixin(
                engine=trainer.engine
            )
        else:
            generation_model = trainer.model
        gen_res = generation_model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=torch.ones_like(input_ids).cuda(),
            generation_config=trainer.generation_config
        )
        return {
            "generate": gen_res,
        }
    
    # 训练
    def train(self):
        setup_distribution(self.config)
        pretrained_model = 'decapoda-research/llama-7b-hf'
        # tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(pretrained_model, 
                                                padding_side="left")
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2


        dataset = AlpacaDataset('alpaca_data.json', tokenizer)
        train_dataset = dataset[:-32]
        eval_dataset = dataset[-32:]
        
        # load model from s3
        model = LlamaForCausalLM(self.config)
        state_dict = LlamaForCausalLM.load_parallel_state_dict(
            path="hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf",
            config=self.config,
            protocol="petrel",
            format="hf"
        )
        model.load_state_dict(state_dict)
        
        # optimizer = optim.Adam(model.parameters(), lr=2e-5)
        optimizer = Lion(model.parameters(), lr=1e-6)

        # monitor
        monitors = [
            StepTimeMonitor(self.config),
            TGSMonitor(self.config),
            MemoryMonitor(self.config),
            LossMonitor(self.config),
            EvalMonitor(self.config)    
        ]

        # metric
        metrics = {'decode': AlpacaDecodeMetric(tokenizer=tokenizer)}

        # trainer
        trainer = Trainer(
            model = model,
            config = self.config,
            loss_fn = GPTLMLoss(-100),
            eval_fn = self.eval_fn,
            optimizer = optimizer,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            train_dataset_collate_fn=lambda x:train_collate_fn(x, tokenizer=tokenizer),
            eval_dataset_collate_fn=lambda x:eval_collate_fn(x, tokenizer=tokenizer),
            generation_config = self.generationConfig,
            monitors = monitors,
            metrics = metrics
        )
        # train
        trainer.train()