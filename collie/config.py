import os
from dataclasses import dataclass, field, is_dataclass, asdict
from typing import Any, Union, Callable

from transformers import PretrainedConfig, AutoConfig, BitsAndBytesConfig
from peft.utils import PeftConfig, PeftType
from peft.mapping import get_peft_config
import torch

__all__ = ["CollieConfig"]

@dataclass
class CollieConfig:
    """
    **CoLLiE** 的配置类。

    :param seed: 随机数种子。
    :param pp_size: 流水线并行的大小。
    :param tp_size: 张量并行大小。
    :param dp_size: 数据并行大小。
    :param pp_partition_method: 流水线的切分策略，包括以下几种取值：

        * ``'parameters'`` - 默认情况下的取值。根据可训练的参数数量进行切分，保证所
          有 rank 上的计算时间是接近的。
        * ``'uniform'`` - 根据模型的层数进行切分，保证每个 rank 上的模型层数是接近
          的。
        * ``'type:[regex]'`` - 根据指定的 layer 进行切分，保证与 ``[regex]`` 名称
          正则匹配的 layer 在每个 rank 上的数目是接近的。比如 ``type:transformer``
          会使得每个 rank 上 Transformer 层的数目接近。该正则匹配不分大小写。 
    :param train_epochs: 训练时的迭代次数。
    :param eval_per_n_steps: 训练的一个 epoch 中，每隔多少 step 进行一次验证。
    :param eval_per_n_epochs: 训练过程中每隔多少次迭代进行一次验证。
    :param train_micro_batch_size: 每个 gpu 上的 batch_size，与 deepspeed 设置中
        的 ``train_micro_batch_size_per_gpu`` 作用相同。如果 ``ds_config`` 中没
        有指定 ``train_micro_batch_size_per_gpu``，则会将 :class:`CollieConfig`
        的 ``train_micro_batch_size`` 作为 ``train_micro_batch_size_per_gpu``
        的值。在流水线并行中，该项代表一个 micro batch 的大小。
    :param gradient_accumulation_steps: 梯度累积的 step 数目。与 deepspeed 设置中
        的 ``gradient_accumulation_steps`` 作用相同。如果 ``ds_config`` 中没有指
        定 ``gradient_accumulation_steps``，则会将 :class:`CollieConfig`的
        ``gradient_accumulation_steps`` 作为 ``gradient_accumulation_steps`` 的
        值。在流水线并行中，该项代表流水线的 micro batch 的数目。
    :param eval_batch_size: 验证时的 batch 大小。在流水线中代表验证时一个 micro
        batch 的大小。
    :param checkpointing: 是否使用梯度检查点，该设置可以节省显存。
    :param use_flash: 是否使用 `FlashAttention <https://github.com/HazyResearch/flash-attention>`_ 。
        仅对部分模型有效。
    :param dropout: :class:`Dropout` 的概率。仅对部分模型有效。
    :param init_method: 初始化方法。必须是一个接收一个 ``torch.Tensor`` 
        并返回一个 ``torch.Tensor`` 的可调用对象。
    :param low_cpu_mem_usage: 是否在初始化模型时尝试减少 CPU 占用
    :param ds_config: **DeepSpeed** 的配置文件。可以是一个路径或字典。
    :param model_config: 模型设置。一般情况下无需手动设置，而是通过
        :meth:`from_pretrained` 获取，
    :param peft_config: Peft 的配置。
    :param quantization_config: 模型的量化配置
    """
    seed: int = field(
        default=42,
        metadata={
            "help": "Random seed that will be set at the beginning of training."
        }
    )
    pp_size: int = field(
        default=1,
        metadata={
            "help": "Pipeline parallelism degree."
        }
    )
    tp_size: int = field(
        default=1,
        metadata={
            "help": "Tensor parallelism degree."
        }
    )
    dp_size: int = field(
        default=1,
        metadata={
            "help": "Data parallelism degree."
        }
    )
    pp_partition_method: str = field(
        default='parameters',
        metadata={
            "help": "Partition method for pipeline parallelism. Default is 'parameters'."
        }
    )
    train_epochs: int = field(
        default=100,
        metadata={
            "help": "Number of training epochs."
        }
    )
    eval_per_n_steps: int = field(
        default=0,
        metadata={
            "help": "Evaluate every n steps."
        }
    )
    eval_per_n_epochs: int = field(
        default=0,
        metadata={
            "help": "Evaluate every n epochs."
        }
    )
    train_micro_batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size (one step) for training."
        }
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of gradient accumulation steps."
        }
    )
    eval_batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size for evaluation."
        }
    )
    checkpointing: bool = field(
        default=True,
        metadata={
            "help": "Whether to use activation checkpointing."
        }
    )
    use_flash: bool = field(
        default=True,
        metadata={
            "help": "Whether to use flash attention."
        }
    )
    dropout: float = field(
        default=0.0,
        metadata={
            "help": "Dropout probability."
        }
    )
    init_method: Callable = field(
        default_factory=lambda: torch.nn.init.uniform_,
        metadata={
            "help": "Initialization method. Possible values are 'none', 'normal', 'xavier_normal', "
            "'xavier_uniform', 'kaiming_normal', 'kaiming_uniform', 'orthogonal', 'sparse', "
            "'eye', 'dirac'. Default is 'none'."
        }
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={
            "help": "Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model."
        }
    )
    ds_config: Union[str, dict] = field(
        default="",
        metadata={
            "help": "DeepSpeed configuration file."
        }
    )
    model_config: PretrainedConfig = field(
        default=PretrainedConfig(),
        metadata={
            "help": "Model configuration."
        }
    )
    peft_config: PeftConfig = field(
        default=PeftConfig(),
        metadata={
            "help": "PEFT configuration."
        }
    )
    quantization_config: BitsAndBytesConfig = field(
        default=BitsAndBytesConfig(),
        metadata={
            "help": "Configuration parameters for the `bitsandbytes` library"
        }
    )

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs):
        """
        加载预训练模型的设置。

        :param path: 预训练模型设置的路径，支持本地路径或 ``huggingface`` 上的仓库
            名称。
        :param kwargs: 其它的设置。可以通过该参数设置 ``pp_size``、``dp_size`` 等
            训练参数和 ``vocab_size`` 等关于模型的参数。
        """
        cfg = cls()
        for key in list(kwargs.keys()):
            if key in cls.__annotations__.keys():
                setattr(cfg, key, kwargs.pop(key))
        cfg.model_config = AutoConfig.from_pretrained(name_or_path, **kwargs)

        return cfg
    
    def save_pretrained(self, path):
        """
        保存预训练模型的设置。
        """
        self.model_config.save_pretrained(path)

    def __getattr__(self, name):
        return getattr(self.model_config, name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__annotations__.keys():
            super().__setattr__(name, value)
        else:
            setattr(self.model_config, name, value)

    def __post_init__(self):
        if isinstance(self.ds_config, str):
            self.ds_config = load_config(self.ds_config)
        if isinstance(self.model_config, str):
            self.model_config = AutoConfig.from_pretrained(self.model_config, trust_remote_code=True)
        if isinstance(self.peft_config, str):
            self.peft_config = load_config(self.peft_config)
        if isinstance(self.peft_config, dict):
            self.peft_config = get_peft_config(self.peft_config)
        if isinstance(self.quantization_config, str):
            self.quantization_config = BitsAndBytesConfig.from_dict(load_config(self.quantization_config))
        assert isinstance(self.ds_config, dict), self.ds_config
        os.environ["COLLIE_SEED"] = str(self.seed)

    def __str__(self) -> str:        
        title = self.__class__.__name__
        r = f"{title}:\n"
        r += _repr_dict(self.__dict__, 0)
        return r
    
    def valid_config(self):
        if "zero_optimization" in self.ds_config.keys() \
            and "stage" in self.ds_config["zero_optimization"].keys() \
            and self.ds_config["zero_optimization"]["stage"] == 3:
                assert self.pp_size == 1, "Pipeline is not compatible with Zero3."
        if self.tp_size > 1:
            assert self.peft_config.peft_type != PeftType.LORA, "Tensor parallelism is not compatible with LoRa"
            assert not self.quantization_config.load_in_4bit and not self.quantization_config.load_in_8bit, \
                "Tensor parallelism is not compatible with int8 quantization and int4 quantization"

def load_config(path: str):
    content = {}
    if path.lower().endswith("yaml"):
        import yaml
        content = yaml.load(open(path, "r"), Loader=yaml.SafeLoader)
    elif path.lower().endswith("json"):
        import json
        content = json.load(open(path, "r"))
    return content

def _repr_dict(d, depth):
    if isinstance(d, CollieConfig):
        return ""
    if isinstance(d, PretrainedConfig):
        d = d.to_diff_dict()
    if not isinstance(d, dict):
        return f" {d}"
    space = "    "
    r = ""
    for k, v in d.items():
        r += f"\n{space * depth}{k}:" + _repr_dict(v, depth+1)
    return r

