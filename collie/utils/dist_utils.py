""" **CoLLie** 中的分布式工具。
"""
import copy
import json
import os
import re
import subprocess
import socket

import deepspeed
import torch
from deepspeed.accelerator import get_accelerator
from deepspeed.monitor.wandb import WandbMonitor
from deepspeed.runtime import zero
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.engine import (
    DeepSpeedEngine,
    DeepSpeedOptimizerCallable,
    DeepSpeedSchedulerCallable,
)
from deepspeed.runtime.hybrid_engine import DeepSpeedHybridEngine
from deepspeed.runtime.pipe import LayerSpec, PipelineModule
from deepspeed.runtime.utils import set_random_seed
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from torch import distributed as dist
from torch.multiprocessing import Process, set_start_method

try:
    from transformers.integrations.deepspeed import _hf_deepspeed_config_weak_ref, HfDeepSpeedConfig
except ImportError:
    from transformers.deepspeed import _hf_deepspeed_config_weak_ref, HfDeepSpeedConfig

from typing import Optional, Union
from weakref import ref

from megatron.core import parallel_state, tensor_parallel

from collie.config import CollieConfig, load_config
from peft import PeftModel

from .peft_utils import patch_peft

__all__ = [
    "env",
    "setup_distribution",
    "set_seed",
    "setup_ds_engine",
    "zero3_load_state_dict",
    "is_zero3_enabled",
    "broadcast_tensor",
    "patch_deepspeed",
    "patch_megatron",
    "patch_pipeline_engine",
    "launch",
]

DTYPE_ENUM = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]


def zero3_load_state_dict(model: torch.nn.Module, state_dict: dict):
    """用于加载 ZeRO stage 3 的模型参数。"""
    for name, param in model.named_parameters():
        with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
            param.data = state_dict[name].data.to(param.device).to(param.dtype)


def is_zero3_enabled(config: CollieConfig):
    """判断是否启用了 ZeRO stage 3。"""
    if isinstance(config.ds_config, str) and os.path.exists(config.ds_config):
        config.ds_config = load_config(config.ds_config)
    if (
        isinstance(config.ds_config, dict)
        and "zero_optimization" in config.ds_config.keys()
        and "stage" in config.ds_config["zero_optimization"].keys()
        and config.ds_config["zero_optimization"]["stage"] == 3
    ):
        return True
    else:
        return False


def setup_ds_engine(
    config: CollieConfig,
    model: torch.nn.Module,
    optimizer: Optional[
        Union[torch.optim.Optimizer, DeepSpeedOptimizerCallable]
    ] = None,
    lr_scheduler: Optional[
        Union[torch.optim.lr_scheduler._LRScheduler, DeepSpeedSchedulerCallable]
    ] = None,
):
    """启动 DeepSpeed 引擎。

    :param config: **CoLLie** 的配置
    :param model: 模型
    :param optimizer: 优化器
    :param lr_scheduler: 学习率调度器
    :return: DeepSpeed 引擎、优化器、dataloader (为 None)、学习率调度器
    """
    if config.pp_size != 1 or config.tp_size != 1:
        from collie.models import CollieModelForCausalLM
        from collie.module import PipelineModel
        from peft import PeftModel

        if isinstance(model, PeftModel):
            assert isinstance(
                model.get_base_model(), (CollieModelForCausalLM, PipelineModel)
            ), "Currently pipeline or tensor parallelism only supports Collie models."
        else:
            assert isinstance(
                model, (CollieModelForCausalLM, PipelineModel)
            ), "Currently pipeline or tensor parallelism only supports Collie models."
    if "train_micro_batch_size_per_gpu" not in config.ds_config.keys():
        config.ds_config[
            "train_micro_batch_size_per_gpu"
        ] = config.train_micro_batch_size
    if "gradient_accumulation_steps" not in config.ds_config.keys():
        config.ds_config[
            "gradient_accumulation_steps"
        ] = config.gradient_accumulation_steps
    print(config.ds_config)
    engine, optimizer, _, lr_scheduler = initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        mpu=parallel_state if config.pp_size == 1 else None,
        config=config.ds_config,
    )
    config.train_micro_batch_size = engine.train_micro_batch_size_per_gpu()
    config.gradient_accumulation_steps = engine.gradient_accumulation_steps()
    return engine, optimizer, _, lr_scheduler


def _decompose_slurm_nodes(s):
    # 使用正则表达式找到所有符合模式的子串
    sub_strings = re.findall(r"[\w-]+\-\[[^\]]*\]|[\w-]+\-\d+", s)

    results = []

    for sub_s in sub_strings:
        # 搜索括号内的元素
        bracket_content = re.search("\[([^\]]+)\]", sub_s)
        if bracket_content:
            # 获取前缀部分
            prefix = sub_s.split("[")[0]
            # 获取括号内的所有元素
            elements = bracket_content.group(1).split(",")
            for element in elements:
                if "-" in element:  # 如果元素是一个范围
                    start, end = [int(i) for i in element.split("-")]
                    results.extend(f"{prefix}{i}" for i in range(start, end + 1))
                else:  # 如果元素是一个单独的数字
                    results.append(prefix + element)
        else:  # 如果没有括号，直接添加到结果中
            results.append(sub_s)
    return results


def port_used(host: str, port: int) -> bool:
    "检查端口是否被占用"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))  # 尝试绑定到本地地址和指定端口
            return True  # 如果绑定成功，返回True，表示端口是空闲的
        except socket.error as e:
            return False  # 如果绑定失败，返回False，表示端口已被占用


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]



def setup_distribution(config) -> None:
    """设置分布式环境。

    可以支持多机情况下的分布式训练：

    1. launch from torchrun
       eg: ``torchrun --standalone --nproc_per_node=8 train.py``
    2. launch from slurm
       eg. ``srun --partition=xxx --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --job-name=xxx --kill-on-bad-exit=1 train.py``

    :param config: :class:`.CollieConfig` 有关分布式并行的策略配置
    """
    if torch.distributed.is_initialized():
        return
    if isinstance(config, str):
        config = load_config(config)
    if isinstance(config.ds_config, str):
        config.ds_config = load_config(config.ds_config)
    config.valid_config()
    patch_bitesandbytes(config)
    patch_transformers(config)
    patch_deepspeed(config)
    patch_megatron()
    patch_peft(config)
    if "WORLD_SIZE" in os.environ.keys():
        # launch from pytorch
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "27001")
    elif "SLURM_PROCID" in os.environ.keys():
        # launch from slurm
        if "SLURM_JOB_NODELIST" in os.environ.keys():
            node_list_str = os.environ["SLURM_JOB_NODELIST"]
            node_list = _decompose_slurm_nodes(node_list_str)
            # result = re.search(r"\[(.*?)\]", node_list_str)
            # if result is None:
            #     node_list.extend(node_list_str.split(","))
            # else:
            #     node_list.extend([item for item in result.groups(1)[0].split(",")])
            #     for i in node_list:
            #         if "-" in i:
            #             node_list.extend(
            #                 list(map(lambda x: f"{x}", range(int(i.split("-")[0]), int(i.split("-")[1]) + 1))))
            #             node_list.remove(i)
            #     node_list = list(map(lambda x: re.sub(r"\[(.*?)\]", x, node_list_str), node_list))
            node_list = sorted(node_list)
            master_addr = node_list[0]
            os.environ["MASTER_ADDR"] = f"{master_addr}"
            result = subprocess.run(
                ["scontrol", "show", "node", master_addr], capture_output=True
            )
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
            master_port = "27002"
            os.environ["MASTER_PORT"] = master_port
        if port_used(master_addr, int(master_port)):
            free_port = find_free_port(master_addr)
            raise RuntimeError(f"Port {master_port} is already in use, Please switch to port {free_port} by executing `export MASTER_PORT={free_port}` in terminal.")
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    deepspeed.init_distributed(
        dist_backend="nccl",
        auto_mpi_discovery=False,
        init_method="tcp://{}:{}".format(master_addr, master_port),
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=config.tp_size,
        pipeline_model_parallel_size=config.pp_size,
    )
    # random seed has to be set after deepspeed.init_distributed
    set_seed(config)
    torch.cuda.set_device(torch.device("cuda:{}".format(os.environ["LOCAL_RANK"])))
    os.environ["COLLIE_PP_RANK"] = "0"
    os.environ["COLLIE_TP_RANK"] = str(parallel_state.get_tensor_model_parallel_rank())
    os.environ["COLLIE_DP_RANK"] = str(parallel_state.get_data_parallel_rank())


def set_seed(config):
    """设置随机数种子。"""
    tensor_parallel.model_parallel_cuda_manual_seed(config.seed)
    set_random_seed(config.seed)


def patch_deepspeed(config):
    if (
        hasattr(config, "ds_config")
        and "zero_optimization" in config.ds_config.keys()
        and "offload_optimizer" in config.ds_config["zero_optimization"].keys()
        and "pin_memory"
        in config.ds_config["zero_optimization"]["offload_optimizer"].keys()
        and not config.ds_config["zero_optimization"]["offload_optimizer"]["pin_memory"]
    ):
        get_accelerator().pin_memory = lambda x: x
    if (
        hasattr(config, "ds_config")
        and "zero_optimization" in config.ds_config.keys()
        and "offload_param" in config.ds_config["zero_optimization"].keys()
        and "pin_memory"
        in config.ds_config["zero_optimization"]["offload_param"].keys()
        and not config.ds_config["zero_optimization"]["offload_param"]["pin_memory"]
    ):
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
    raw_initialize_optimizer_states = copy.deepcopy(
        DeepSpeedZeroOptimizer.initialize_optimizer_states
    )

    def safe_initialize_optimizer_states(self, *args, **kwargs):
        while True:
            try:
                raw_initialize_optimizer_states(self, *args, **kwargs)
                break
            except RuntimeError as e:
                continue

    DeepSpeedZeroOptimizer.initialize_optimizer_states = (
        safe_initialize_optimizer_states
    )

    raw_wandb_init = copy.deepcopy(WandbMonitor.__init__)

    def collie_wandb_init(self, wandb_config):
        raw_wandb_init(self, wandb_config)
        import wandb

        wandb.run.name = wandb_config.job_name
        # this causes bugs
        # wandb.config.update(wandb_config.config, allow_val_change=True)

    WandbMonitor.__init__ = collie_wandb_init

    # LayerSpec
    def layer_spec_init(self, typename, *module_args, **module_kwargs):
        self.typename = typename
        self.module_args = module_args
        self.module_kwargs = module_kwargs

        if dist.is_initialized():
            self.global_rank = dist.get_rank()
        else:
            self.global_rank = -1

    LayerSpec.__init__ = layer_spec_init


def patch_transformers(config):
    global _hf_deepspeed_config_weak_ref
    ds_config = HfDeepSpeedConfig(config.ds_config)
    # weak ref -> strong ref
    _hf_deepspeed_config_weak_ref = lambda: ds_config


def patch_megatron():
    parallel_state.get_model_parallel_world_size = (
        lambda: parallel_state.get_tensor_model_parallel_world_size()
    )
    parallel_state.get_model_parallel_rank = (
        lambda: parallel_state.get_tensor_model_parallel_rank()
    )
    parallel_state.get_pipe_parallel_rank = (
        lambda: parallel_state.get_pipeline_model_parallel_rank()
    )


def patch_bitesandbytes(config: CollieConfig):
    # 较低版本的 transformers 没有 load_in_4bit
    if (
        getattr(config.quantization_config, "load_in_4bit", False)
        or config.quantization_config.load_in_8bit
    ):
        from bitsandbytes.nn import Int8Params, Params4bit

        raw_cuda_8bit = Int8Params.cuda

        def cuda_8bit(self, device):
            if self.data.is_cuda:
                return self
            else:
                return raw_cuda_8bit(self, device)

        Int8Params.cuda = cuda_8bit
        raw_cuda_4bit = Params4bit.cuda

        def cuda_4bit(self, device):
            if self.data.is_cuda:
                return self
            else:
                return raw_cuda_4bit(self, device)

        Params4bit.cuda = cuda_4bit


def broadcast_tensor(tensor, dtype=None, src=0, shape=None, ndim=None, group=None):
    """从 ``src`` 广播 ``tensor``。

    该函数支持广播 ``tensor`` 的维度和类型。如果 ``ndim`` 和 ``shape`` 为 ``None``
    则会首先广播 ``tensor`` 的维度。

    .. code-block::

        if rank == 0:
            logits = torch.ones(4,2)
        else:
            logits = None

        # 其它 rank 上并不知道 logits 的维度，因此 ndim 和 shape 均为 None
        logits = broadcast_tensor(logits, src=0)

    .. warning::

        请确保 ``ndim``、``shape`` 和 ``dtype`` 参数在所有 rank 上保持一致，
        否则会导致分布式进程卡死。

    :param tensor: 要广播的张量。注意在 ``src`` rank 上必须为一个张量。
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


class Env:
    """**CoLLiE** 的环境变量，可以从中获取各种并行的 world_size 和 rank。

    调用时直接导入已经实例化好的对象 ``env`` 即可。

    .. code-block::

        from collie.utils import env
        print(env.dp_rank)
    """

    @property
    def seed(self):
        """
        随机数种子
        """
        return int(os.getenv("COLLIE_SEED"))

    @property
    def rank(self):
        """
        Global rank。
        """
        return int(os.getenv("RANK", "0"))

    @property
    def local_rank(self):
        """
        Local rank。
        """
        return int(os.getenv("LOCAL_RANK", "0"))

    @staticmethod
    def barrier(group=None):
        """
        在 ``group`` 上进行同步。
        """
        if dist.is_initialized():
            torch.distributed.barrier(group)

    @property
    def world_size(self):
        """
        分布式训练的 world size。
        """
        return int(os.getenv("WORLD_SIZE", "1"))

    @property
    def pp_rank(self):
        """
        流水线并行的 rank。
        """
        if not dist.is_initialized():
            return 0
        return parallel_state.get_pipeline_model_parallel_rank()

    @property
    def dp_rank(self):
        """
        数据并行的 rank。
        """
        if not dist.is_initialized():
            return 0
        return parallel_state.get_data_parallel_rank()

    @property
    def tp_rank(self):
        """
        张量并行的 rank。
        """
        if not dist.is_initialized():
            return 0
        return parallel_state.get_tensor_model_parallel_rank()

    @property
    def mp_rank(self):
        """
        模型并行的 rank。模型并行与数据并行相对，同时包含了张量并行和模型并行。
        """
        if not dist.is_initialized():
            return 0
        return parallel_state.get_model_parallel_group().rank()

    @property
    def is_pipeline(self):
        """
        判断是否是流水线并行。
        """
        return "COLLIE_PP_PARTS" in os.environ.keys()

    @property
    def pipeline_parts(self):
        """
        返回流水线并行中模型切分的分界点，长度为 ``pp_size + 1``。

        如果不存在流水线并行则返回 ``None``。

        e.g. pp_size 为 2时，返回形如 [0, 13, 25] 的结果。
        """
        if "COLLIE_PP_PARTS" in os.environ.keys():
            parts = json.loads(os.environ["COLLIE_PP_PARTS"])
        else:
            parts = None

        return parts

    @property
    def pipeline_layers_idx(self):
        """
        返回流水线并行中当前 rank 切分的模型索引。

        如果不存在流水线并行返回 ``None``。
        """
        parts = self.pipeline_parts
        if parts is None:
            return None
        else:
            stage = self.pp_rank
            return list(range(parts[stage], parts[stage + 1]))

    @property
    def tp_group(self):
        """
        张量并行的通信组。
        """
        return parallel_state.get_tensor_model_parallel_group()

    @property
    def pp_group(self):
        """
        流水线并行的通信组。
        """
        return parallel_state.get_pipeline_model_parallel_group()

    @property
    def dp_group(self):
        """
        数据并行的通信组。
        """
        return parallel_state.get_data_parallel_group()

    @property
    def dp_size(self):
        """
        数据并行的 world size。
        """
        if not dist.is_initialized():
            return 1
        return parallel_state.get_data_parallel_world_size()

    @property
    def tp_size(self):
        """
        张量并行的 world size。
        """
        if not dist.is_initialized():
            return 1
        return parallel_state.get_tensor_model_parallel_world_size()

    @property
    def pp_size(self):
        """
        流水线并行的 world size。
        """
        if not dist.is_initialized():
            return 1
        return parallel_state.get_pipeline_model_parallel_world_size()

    @property
    def is_last_stage(self):
        """
        是否是流水线的最后一个阶段。
        """
        if not dist.is_initialized():
            return True
        return parallel_state.is_pipeline_last_stage()

    @property
    def is_first_stage(self):
        """
        是否是流水线的第一个阶段。
        """
        if not dist.is_initialized():
            return True
        return parallel_state.is_pipeline_first_stage()


env = Env()


def launch(target: callable, devices: str, port: int = 12701):
    """在一台节点上以 torchrun 风格启动多进程
    :param target: 启动的函数
    :param devices: 启动的设备，以逗号分隔
    :param port: 启动的端口
    """

    def _wrapper(environ):
        torch.set_default_device(torch.device(int(environ.get("LOCAL_RANK"))))
        os.environ.update(environ)
        target()

    set_start_method("fork")
    processes = []
    for rank in range(len(devices.split(","))):
        environ = copy.deepcopy(os.environ)
        environ["MASTER_ADDR"] = "localhost"
        environ["MASTER_PORT"] = str(port)
        environ["WORLD_SIZE"] = str(len(devices.split(",")))
        environ["RANK"] = str(rank)
        environ["LOCAL_RANK"] = str(rank)
        environ["CUDA_VISIBLE_DEVICES"] = devices
        processes.append(Process(target=_wrapper, args=(environ,)))
    [p.start() for p in processes]
    [p.join() for p in processes]


def initialize(
    args=None,
    model: torch.nn.Module = None,
    optimizer=None,
    model_parameters: Optional[torch.nn.Module] = None,
    training_data: Optional[torch.utils.data.Dataset] = None,
    lr_scheduler=None,
    mpu=None,
    dist_init_required: Optional[bool] = None,
    collate_fn=None,
    config=None,
    config_params=None,
):
    # Disable zero.Init context if it's currently enabled
    zero.partition_parameters.shutdown_init_context()

    assert model is not None, "deepspeed.initialize requires a model"

    global dist
    from deepspeed import comm as dist

    dist_backend = get_accelerator().communication_backend_name()
    dist.init_distributed(
        dist_backend=dist_backend, dist_init_required=dist_init_required
    )

    # Set config using config_params for backwards compat
    if config is None and config_params is not None:
        config = config_params

    # Check for deepscale_config for backwards compat
    if hasattr(args, "deepscale_config") and args.deepscale_config is not None:
        if hasattr(args, "deepspeed_config"):
            assert (
                args.deepspeed_config is None
            ), "Not sure how to proceed, we were given both a deepscale_config and deepspeed_config"
        args.deepspeed_config = args.deepscale_config
        args.deepscale_config = None

    # Check that we have only one config passed
    if hasattr(args, "deepspeed_config") and args.deepspeed_config is not None:
        assert (
            config is None
        ), "Not sure how to proceed, we were given deepspeed configs in the deepspeed arguments and deepspeed.initialize() function call"
        config = args.deepspeed_config
    assert (
        config != None
    ), "DeepSpeed requires --deepspeed_config to specify configuration file"
    if not isinstance(model, PipelineModule) and not (
        isinstance(model, PeftModel)
        and isinstance(model.get_base_model(), PipelineModule)
    ):
        config_class = DeepSpeedConfig(config, mpu)
        if config_class.hybrid_engine.enabled:
            engine = DeepSpeedHybridEngine(
                args=args,
                model=model,
                optimizer=optimizer,
                model_parameters=model_parameters,
                training_data=training_data,
                lr_scheduler=lr_scheduler,
                mpu=mpu,
                dist_init_required=dist_init_required,
                collate_fn=collate_fn,
                config=config,
                config_class=config_class,
            )
        else:
            engine = DeepSpeedEngine(
                args=args,
                model=model,
                optimizer=optimizer,
                model_parameters=model_parameters,
                training_data=training_data,
                lr_scheduler=lr_scheduler,
                mpu=mpu,
                dist_init_required=dist_init_required,
                collate_fn=collate_fn,
                config=config,
                config_class=config_class,
            )
    else:
        from .pipeline_engine import ColliePipelineEngine

        assert mpu is None, "mpu must be None with pipeline parallelism"
        mpu = model.mpu()
        config_class = DeepSpeedConfig(config, mpu)
        engine = ColliePipelineEngine(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            training_data=training_data,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=dist_init_required,
            collate_fn=collate_fn,
            config=config,
            config_class=config_class,
        )

    return_items = [
        engine,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler,
    ]
    return tuple(return_items)
