import sys

sys.path.append('../../')
sys.path.append("/mnt/petrelfs/gutianle/Megatron-LM/")


import torch
import torch.distributed as dist

from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

from collie.config import CollieConfig
from collie.controller.trainer import Trainer
from collie.metrics import BaseMetric
from collie.models.llama.model import LlamaForCausalLM
from collie.metrics.decode import DecodeMetric
from collie.utils import env, setup_distribution

from collie.optim.adan import Adan
from collie.optim.inplace_sgd import InplaceSGD
from collie.optim.lion import Lion
from collie.optim.sophiag import SophiaG

from collie.utils.monitor import StepTimeMonitor, TGSMonitor, MemoryMonitor, LossMonitor, EvalMonitor

class LLaMaWithOptim:
    """
    LLaMAWithOptim是一个测试各种Optimizer效果的类。
    Optimizer的选择有Adan, InplaceSGD, Lion和SophiaG（具体可见collie.optim)
    本类提供四种测试选择：
        * test_adan：测试Adan优化器
        * test_inplaceSGD：测试InplaceSGD优化器
        * test_lion：测试Lion优化器
        * test_sophiag：测试SophiaG优化器
    :param name: 指定的monitor的文件夹名称，用于区分不同的Optimizer的效果
    """
    def __init__(self, name) -> None:
        self.optimizer = None
        self.name = name
        self.config = self._setConfig() 
        self.model = self._getModel()
        
                
    def _setConfig(self):
        config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf")
        config.tp_size = 4
        config.dp_size = 2
        config.pp_size = 1
        config.train_epochs = 3
        config.train_micro_batch_size = 1
        config.ds_config = {
            "fp16": {"enabled": True},
            "monitor_config": {
                "enabled": True,
                "tensorboard": {
                    "enabled": True,
                    "output_path": "./ds_logs/",
                    "job_name": f"alpaca_optimizer_{self.name}"
                }
            }
        }
        return config

    def _train(self):
        setup_distribution(self.config)
        monitors = [
            StepTimeMonitor(self.config),
            TGSMonitor(self.config),
            MemoryMonitor(self.config),
            LossMonitor(self.config),
            EvalMonitor(self.config)
        ]
        tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", padding_side="left", add_eos_token=True)
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        model = self.model
        train_sample = tokenizer("Collie is a python package for finetuning large language models.", return_tensors="pt").input_ids.squeeze(0)
        train_dataset = [(train_sample, train_sample) for _ in range(1000)]
        trainer = Trainer(
            model = model,
            train_dataset = train_dataset,
            metrics = {'decode': DecodeMetric(tokenizer=tokenizer)},
            optimizer = self.optimizer,
            monitors = monitors,
            config = self.config
        )
        trainer.train()
    
    def _getModel(self):
        model = LlamaForCausalLM.from_config(self.config)
        state_dict = LlamaForCausalLM.load_parallel_state_dict(
            path="hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf",
            config= self.config,
            protocol="petrel",
            format="hf"
        )
        model.load_state_dict(state_dict)
        return model
        
    """
    测试Adan优化器
    """
    def test_adan(self):
        self.optimizer = Adan(self.model.parameters(), lr=2e-5)
        self._train()
        
    """
    测试Sophia优化器
    """
    def test_sophiag(self):
        self.optimizer = SophiaG(self.model.parameters(), lr=2e-5)
        self._train()
    
    """
    测试Lion优化器
    """
    def test_lion(self):
        self.optimizer = Lion(self.model.parameters(), lr=2e-5)
        self._train()
    

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:28002 --nnodes=1 --nproc_per_node=8 test_optim.py
optim_test = LLaMaWithOptim('inplace_sgd')
optim_test.test_inplaceSGD()