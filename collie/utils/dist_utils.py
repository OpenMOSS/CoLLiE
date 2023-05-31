import os
import copy
import json
import re
import subprocess

import torch
from torch import distributed as dist
import deepspeed
from deepspeed.runtime.utils import set_random_seed
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.engine import DeepSpeedOptimizerCallable, DeepSpeedSchedulerCallable
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.accelerator import get_accelerator
from megatron.core import parallel_state, tensor_parallel

from typing import Union, Optional

from .utils import classproperty, _split_batch
from collie.config import load_config, CollieConfig

DTYPE_ENUM = [
    torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.uint8,
    torch.int8, torch.int16, torch.int32, torch.int64, torch.bool
]

def zero3_load_state_dict(model: torch.nn.Module, state_dict: dict):
    for name, param in model.named_parameters():
        with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
            param.data = state_dict[name].data.to(param.device).to(param.dtype)

def is_zero3_enabled(config: CollieConfig):
    if isinstance(config.ds_config, str) and os.path.exists(config.ds_config):
        config.ds_config = load_config(config.ds_config)
    if isinstance(config.ds_config, dict) \
            and "zero_optimization" in config.ds_config.keys() \
            and "stage" in config.ds_config["zero_optimization"].keys() \
            and config.ds_config["zero_optimization"]["stage"] == 3:
        return True
    else:
        return False

def setup_ds_engine(
        config: CollieConfig,
        model: torch.nn.Module,
        optimizer: Optional[Union[torch.optim.Optimizer, DeepSpeedOptimizerCallable]] = None,
        lr_scheduler: Optional[Union[torch.optim.lr_scheduler._LRScheduler, DeepSpeedSchedulerCallable]] = None
):
    if config.pp_size != 1 or config.tp_size != 1:
        from collie.models import CollieModelForCausalLM
        from collie.module import PipelineModel
        assert isinstance(model, CollieModelForCausalLM) or isinstance(model, PipelineModel), "Currently pipeline or tensor parallelism only supports Collie models."
    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        mpu=parallel_state if config.pp_size == 1 else None,
        config=config.ds_config
    )
    return engine, optimizer, _, lr_scheduler


def setup_distribution(config) -> None:
    """Set up the distributed training environment.
    Support two kinds of distributed training:
    1. launch from torchrun
        eg: torchrun --standalone --nproc_per_node=8 train.py
    2. launch from slurm
        eg. srun --partition=xxx --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --job-name=xxx --kill-on-bad-exit=1 train.py
    """
    if torch.distributed.is_initialized():
        return
    if isinstance(config, str):
        config = load_config(config)
    if isinstance(config.ds_config, str):
        config.ds_config = load_config(config.ds_config)
    if "train_micro_batch_size_per_gpu" not in config.ds_config.keys():
        config.ds_config["train_micro_batch_size_per_gpu"] = config.train_micro_batch_size
    if "gradient_accumulation_steps" not in config.ds_config.keys():
        config.ds_config["gradient_accumulation_steps"] = config.gradient_accumulation_steps
    patch_deepspeed(config)
    patch_megatron()
    if "WORLD_SIZE" in os.environ.keys():
        # launch from pytorch
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "27001")
    elif "SLURM_PROCID" in os.environ.keys():
        # launch from slurm
        if "SLURM_JOB_NODELIST" in os.environ.keys():
            node_list_str = os.environ["SLURM_JOB_NODELIST"]
            node_list = []
            result = re.search(r"\[(.*?)\]", node_list_str)
            if result is None:
                node_list.append(node_list_str)
            else:
                node_list.extend([item for item in result.groups(1)[0].split(",")])
                for i in node_list:
                    if "-" in i:
                        node_list.extend(
                            list(map(lambda x: f"{x}", range(int(i.split("-")[0]), int(i.split("-")[1]) + 1))))
                        node_list.remove(i)
                node_list = list(map(lambda x: re.sub(r"\[(.*?)\]", x, node_list_str), node_list))
            node_list = sorted(node_list)
            master_addr = node_list[0]
            os.environ["MASTER_ADDR"] = f"{master_addr}"
            result = subprocess.run(["scontrol", "show", "node", master_addr], capture_output=True)
            result = re.search(r"NodeAddr=(.*?)\s", result.stdout.decode())
            if result:
                master_addr = result.groups(1)[0]
                os.environ["MASTER_ADDR"] = f"{master_addr}"
        else:
            master_addr = "localhost"
            os.environ["MASTER_ADDR"] = f"{master_addr}"
        if "MASTER_PORT" in os.environ.keys():
            master_port = os.environ["MASTER_PORT"]
        else:
            master_port = 27002
            os.environ["MASTER_PORT"] = f"{master_port}"
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    deepspeed.init_distributed(dist_backend='nccl',
                               auto_mpi_discovery=False,
                               init_method="tcp://{}:{}".format(
                                   master_addr,
                                   master_port),
                               world_size=int(os.environ["WORLD_SIZE"]),
                               rank=int(os.environ["RANK"]))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=config.tp_size,
        pipeline_model_parallel_size=config.pp_size
    )
    # random seed has to be set after deepspeed.init_distributed
    set_seed(config)
    torch.cuda.set_device(torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"])))
    os.environ["COLLIE_PP_RANK"] = "0"
    os.environ["COLLIE_TP_RANK"] = str(parallel_state.get_tensor_model_parallel_rank())
    os.environ["COLLIE_DP_RANK"] = str(parallel_state.get_data_parallel_rank())


def set_seed(config):
    """Set random seed for reproducibility.
    """
    tensor_parallel.model_parallel_cuda_manual_seed(config.seed)
    set_random_seed(config.seed)

def patch_pipeline_engine(config):
    """
    Replace train_batch and eval_batch to fit our Trainer and GenerationMixin.
    """
    raw_train_batch = copy.deepcopy(PipelineEngine.train_batch)
    raw_eval_batch = copy.deepcopy(PipelineEngine.eval_batch)
    def train_batch(self, batch):
        # batch tuple, batch_size is micro_batch * accumulate_steps
        batch = _split_batch(batch, self.train_micro_batch_size_per_gpu(),
                             self.gradient_accumulation_steps())
        data_iter = iter(batch)
        return raw_train_batch(self, data_iter)

    def eval_batch(self, batch):
        batch = _split_batch(batch, config.eval_batch_size,
                             self.gradient_accumulation_steps())
        data_iter = iter(batch)
        logits = raw_eval_batch(self, data_iter, return_logits=False,
                                    compute_loss=False, reduce_output=None)
        # logits: list
        # len(logits) = micro_batch_nums
        # Assume batch first
        if logits is not None:
            assert isinstance(logits, list), type(logits)
            logits = torch.cat(logits, dim=0)
        src_rank = self.grid.stage_to_global(self.num_stages - 1)
        logits = broadcast_tensor(logits, src=src_rank,
                                  group=env.pp_group)
        return logits
    
    PipelineEngine.train_batch = train_batch
    PipelineEngine.eval_batch = eval_batch

def patch_deepspeed(config):
    if hasattr(config, "ds_config") \
            and "zero_optimization" in config.ds_config.keys() \
            and "offload_optimizer" in config.ds_config["zero_optimization"].keys() \
            and "pin_memory" in config.ds_config["zero_optimization"]["offload_optimizer"].keys() \
            and not config.ds_config["zero_optimization"]["offload_optimizer"]["pin_memory"]:
        get_accelerator().pin_memory = lambda x: x
    if hasattr(config, "ds_config") \
            and "zero_optimization" in config.ds_config.keys() \
            and "offload_param" in config.ds_config["zero_optimization"].keys() \
            and "pin_memory" in config.ds_config["zero_optimization"]["offload_param"].keys() \
            and not config.ds_config["zero_optimization"]["offload_param"]["pin_memory"]:
        get_accelerator().pin_memory = lambda x: x
    raw_init = copy.deepcopy(DeepSpeedZeroOptimizer.__init__)

    def safe_init(self, *args, **kwargs):
        while True:
            try:
                raw_init(self, *args, **kwargs)
                break
            except RuntimeError as e:
                continue

    DeepSpeedZeroOptimizer.__init__ = safe_init
    raw_initialize_optimizer_states = copy.deepcopy(DeepSpeedZeroOptimizer.initialize_optimizer_states)

    def safe_initialize_optimizer_states(self, *args, **kwargs):
        while True:
            try:
                raw_initialize_optimizer_states(self, *args, **kwargs)
                break
            except RuntimeError as e:
                continue

    DeepSpeedZeroOptimizer.initialize_optimizer_states = safe_initialize_optimizer_states
    patch_pipeline_engine(config)

def patch_megatron():
    parallel_state.get_model_parallel_world_size = lambda: parallel_state.get_tensor_model_parallel_world_size()
    parallel_state.get_model_parallel_rank = lambda: parallel_state.get_tensor_model_parallel_rank()
    parallel_state.get_pipe_parallel_rank = lambda: parallel_state.get_pipeline_model_parallel_rank()

def broadcast_tensor(tensor, dtype=None, src=0, shape=None,
                     ndim=None, group=None):
    """
    Broadcast ``tensor`` from ``src``.

    if ``ndim`` and ``shape`` is None, we will broadcast ``tensor``'s
    shape first. It is required that every rank's parameter is the same.
    :param tensor: Tensor to broadcast. In source rank ``tensor`` must
        be a ``torch.Tensor``.
    """
    ndim = ndim if shape is None else len(shape)
    if ndim is None:
        if src == env.rank:
            ndim_tensor = torch.tensor(len(tensor.shape), dtype=torch.int).cuda()
        else:
            ndim_tensor = torch.tensor(0, dtype=torch.int).cuda()
        dist.broadcast(ndim_tensor, src, group)
        ndim = ndim_tensor.item()
    if shape is None:
        if src == env.rank:
            shape_tensor = torch.tensor(tensor.shape, dtype=torch.int).cuda()
        else:
            shape_tensor = torch.zeros(ndim, dtype=torch.int).cuda()
        dist.broadcast(shape_tensor, src, group)
        shape = shape_tensor.tolist()
    if dtype is None:
        if src == env.rank:
            dtype_idx = DTYPE_ENUM.index(tensor.dtype)
            dtype_idx_tensor = torch.tensor(dtype_idx, dtype=torch.int).cuda()
        else:
            dtype_idx_tensor = torch.tensor(0, dtype=torch.int).cuda()
        dist.broadcast(dtype_idx_tensor, src, group)
        dtype = DTYPE_ENUM[dtype_idx_tensor.item()]
    if src != env.rank:
        tensor = torch.zeros(shape, dtype=dtype).cuda().to(dtype)
    dist.broadcast(tensor, src, group)
    return tensor

class env:
    @classproperty
    def rank(self):
        return int(os.getenv("RANK", "0"))

    @classproperty
    def local_rank(self):
        return int(os.getenv("LOCAL_RANK", "0"))

    @staticmethod
    def barrier(group=None):
        if dist.is_initialized():
            torch.distributed.barrier(group)

    @classproperty
    def world_size(self):
        return int(os.getenv("WORLD_SIZE", "1"))

    @classproperty
    def pp_rank(self):
        if not dist.is_initialized():
            return 0
        return parallel_state.get_pipeline_model_parallel_rank()

    @classproperty
    def dp_rank(self):
        if not dist.is_initialized():
            return 0
        return parallel_state.get_data_parallel_rank()

    @classproperty
    def tp_rank(self):
        if not dist.is_initialized():
            return 0
        return parallel_state.get_tensor_model_parallel_rank()

    @classproperty
    def mp_rank(self):
        if not dist.is_initialized():
            return 0
        return parallel_state.get_model_parallel_group().rank()

    @classproperty
    def is_pipeline(self):
        return "COLLIE_PP_PARTS" in os.environ.keys()

    @classproperty
    def pipline_parts(self):
        if "COLLIE_PP_PARTS" in os.environ.keys():
            parts = json.loads(os.environ["COLLIE_PP_PARTS"])
        else:
            parts = None

        return parts

    @classproperty
    def pipline_layers_idx(self):
        """
        :return: list or None
        """
        parts = self.pipline_parts
        if parts is None:
            return None
        else:
            stage = self.pp_rank
            return list(range(parts[stage], parts[stage + 1]))

    @classproperty
    def tp_group(self):
        return parallel_state.get_tensor_model_parallel_group()

    @classproperty
    def pp_group(self):
        return parallel_state.get_pipeline_model_parallel_group()

    @classproperty
    def dp_group(self):
        return parallel_state.get_data_parallel_group()

    @classproperty
    def dp_size(self):
        if not dist.is_initialized():
            return 1
        return parallel_state.get_data_parallel_world_size()

    @classproperty
    def tp_size(self):
        if not dist.is_initialized():
            return 1
        return parallel_state.get_tensor_model_parallel_world_size()

    @classproperty
    def pp_size(self):
        if not dist.is_initialized():
            return 1
        return parallel_state.get_pipeline_model_parallel_world_size()

    @classproperty
    def is_last_stage(self):
        if not dist.is_initialized():
            return True
        return parallel_state.is_pipeline_last_stage()

    @classproperty
    def is_first_stage(self):
        if not dist.is_initialized():
            return True
        return parallel_state.is_pipeline_first_stage()
