from collie.trainer.arguments import TrainerArgs
from collie.log.print import print

import re
import os
import torch
import deepspeed
import subprocess
from deepspeed.runtime.utils import set_random_seed
from megatron.core import parallel_state, tensor_parallel

from typing import Optional

class Trainer:
    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: Optional[torch.optim.Optimizer],
                 train_dataset: Optional[torch.utils.data.Dataset],
                 args: TrainerArgs) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.args = args
        self.setup_distributation()
    
    def set_random_seed(self):
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
                                   world_size=1,
                                   rank=0)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=self.args.tp_size)
        # random seed has to be set after deepspeed.init_distributed
        self.set_random_seed()
        torch.cuda.set_device(torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"])))
        
    def setup_parallel_model(self):
        ...