from collie.trainer.arguments import Arguments, load_config
from collie.hack import hack_deepspeed, hack_megatron
from collie.module import GPTLMLoss
from collie.log.print import print
from collie.log import logger

import re
import os
import tqdm
import torch
import deepspeed
import subprocess
from dataclasses import asdict
import torch.distributed as dist
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.utils import set_random_seed
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from megatron.core import parallel_state, tensor_parallel

from typing import Optional, Callable, Union

class Trainer:
    def __init__(self, 
                 model: torch.nn.Module,
                 loss_fn: Callable = GPTLMLoss(),
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 train_dataset: Optional[torch.utils.data.Dataset] = None,
                 args: Union[Arguments, str] = Arguments()) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.loss_fn = loss_fn
        self.args = args
        self.set_ds_config()
        self.setup_distributation()
        self.setup_parallel_model()
    
    def set_seed(self):
        """Set random seed for reproducibility.
        """
        tensor_parallel.model_parallel_cuda_manual_seed(self.args.seed)
        set_random_seed(self.args.seed)
    
    def setup_distributation(self) -> None:
        """Setup the distributed training environment.
        Support two kinds of distributed training:
        1. launch from torchrun
            eg: torchrun --standalone --nproc_per_node=8 train.py
        2. launch from slurm
            eg. srun --partition=xxx --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --job-name=xxx --kill-on-bad-exit=1 train.py
        """
        hack_deepspeed();hack_megatron()
        if "WORLD_SIZE" in os.environ.keys():
            # launch from pytorch
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", "27001")
        elif "SLURM_JOB_NODELIST" in os.environ.keys():
            # launch from slurm
            node_list_str = os.environ["SLURM_JOB_NODELIST"]
            node_list = []
            result = re.search(r"\[(.*?)\]", node_list_str)
            if result is None:
                node_list.append(node_list_str)
            else:
                node_list.extend([item for item in result.groups(1)[0].split(",")])
                for i in node_list:
                    if "-" in i:
                        node_list.extend(list(map(lambda x: f"{x}", range(int(i.split("-")[0]), int(i.split("-")[1]) + 1))))
                        node_list.remove(i)
                node_list = list(map(lambda x: re.sub(r"\[(.*?)\]", x, node_list_str), node_list))
                node_list = sorted(node_list)
                master_addr = node_list[0]
                result = subprocess.run(["scontrol", "show", "node", master_addr], capture_output=True)
                result = re.search(r"NodeAddr=(.*?)\s", result.stdout.decode())
                if result:
                    master_addr = result.groups(1)[0]
                if "MASTER_PORT" in os.environ.keys():
                    master_port = os.environ["MASTER_PORT"]
                else:
                    master_port = 27002
                os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
                os.environ["RANK"] = os.environ["SLURM_PROCID"]
                os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        deepspeed.init_distributed(dist_backend='nccl', 
                                   init_method="tcp://{}:{}".format(
                                       master_addr, 
                                       master_port),
                                   world_size=int(os.environ["WORLD_SIZE"]),
                                   rank=int(os.environ["RANK"]))
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=self.args.tp_size)
        # random seed has to be set after deepspeed.init_distributed
        self.set_seed()
        torch.cuda.set_device(torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"])))
        
    def set_ds_config(self):
        if isinstance(self.args, str):
            self.args = load_config(self.args)
        if isinstance(self.args.ds_config, str):
            self.args.ds_config = load_config(self.args.ds_config)
        print("Collie config", asdict(self.args))
        
    def setup_parallel_model(self):
        """Setup parallel model.
        """
        if dist.get_world_size() != self.args.tp_size * self.args.dp_size * self.args.pp_size:
            logger.rank_zero_warning("The world size is not equal to the product of the parallel sizes set."
                                     f"{dist.get_world_size()} != {self.args.tp_size} * {self.args.dp_size} * {self.args.dp_size}.")
            self.args.dp_size = dist.get_world_size() // (self.args.tp_size * self.args.pp_size)
            logger.rank_zero_warning(f"Set dp_size to {self.args.dp_size}.")
        if self.args.pp_size > 1:
            self.model.loss_fn = self.loss_fn
        self.engine, self.optimizer, self.training_dataloader, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=[p for p in self.model.parameters() if p.requires_grad],
            optimizer=self.optimizer,
            training_data=self.train_dataset,
            mpu=parallel_state if self.args.pp_size == 1 else None,
            config=self.args.ds_config
        )
        
    def train(self):
        with tqdm.tqdm(range(self.args.train_epochs), disable=dist.get_rank() != 0) as tqbar_epoch:
            for epoch_idx in tqbar_epoch:
                with tqdm.tqdm(self.training_dataloader, disable=dist.get_rank() != 0) as tqbar_batch:
                    for batch_idx, batch in enumerate(tqbar_batch):
                        if self.args.pp_size > 1:
                            loss = self.engine.train_batch(data_iter=iter([batch]))
                        else:
                            input_ids, label = batch
                            logits = self.engine(input_ids)
                            loss = self.loss_fn(logits, label)
                            self.engine.backward(loss)
                            self.engine.step()
                        tqbar_batch.update(1)
                        tqbar_batch.set_postfix(
                            loss=loss.item(), 
                            batch=f"{batch_idx + 1}/"
                            f"{len(self.train_dataset) / self.args.ds_config['train_micro_batch_size_per_gpu'] / self.args.ds_config['gradient_accumulation_steps']}")
                tqbar_epoch.set_postfix(epoch=f"{epoch_idx + 1}/{self.args.train_epochs}")
                tqbar_epoch.update(1)