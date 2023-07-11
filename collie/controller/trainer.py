"""
``Trainer`` 是 **CoLLie** 中的训练器，它负责整个训练过程的控制。包含训练功能、验证功能、保存断点功能等。
"""
__all__ = [
    "Trainer"
]

import os
import json
import logging
import glob
from typing import Optional, Callable, Union, Tuple, Iterable, Any, Dict, Sequence, List
from collections import OrderedDict
from functools import reduce

import torch
import deepspeed
import numpy as np
from torch import nn
import torch.distributed as dist
from torch.optim.lr_scheduler import _LRScheduler
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.engine import DeepSpeedSchedulerCallable
from transformers.modeling_utils import PreTrainedModel, load_state_dict
from peft import PeftModel, PeftConfig, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import PreTrainedTokenizerBase
from transformers.utils import ContextManagers

from collie.config import CollieConfig
from collie.module import PipelineGenerationMixin, GPTLMLoss, PipelineModel
from collie.driver.io import IODriver
from collie.log import logger
from collie.utils import progress, env, setup_ds_engine, BaseProvider, is_zero3_enabled, \
    BaseMonitor, _MultiMonitors, broadcast_tensor, ColliePadder, auto_param_call
from collie.optim import Lomo
from collie.models.base import CollieModelForCausalLM
from .evaluator import Evaluator
from .server import Server
from collie.data import CollieDataLoader
from collie.callbacks.callback import Callback
from collie.callbacks.callback_manager import CallbackManager, prepare_callback
from .utils import TrainerEventTrigger

class Trainer(TrainerEventTrigger):
    r"""
    **CoLLie** 训练器，支持快速分布式训练和验证。

    :param model: 用于训练和验证的模型，可以使用 **CoLLie** 实现的模型或 transformers 提供的模型：

        * **CoLLie** 实现的模型 :class:`.CollieModelForCausalLM` 可支持的并行方式包括：张量并行、流水线并行、`ZeRO`
        * transformers 提供的模型 ``transformers.PreTrainedModel`` 只支持 `ZeRO`
        
    :param config: 用于训练和验证的配置
    :param tokenizer: 用于训练和验证的分词器，该分词器将用于:
        * 保存模型时 `trainer.save_model` 时自动同时保存 `tokenizer`
        * 使用 :class:`~collie.controller.evaluator.EvaluatorForGeneration` 进行基于生成的验证时，使用 `tokenizer` 对生成的结果进行解码
        若无上述需求，可不传入 `tokenizer`
    :param loss_fn: 用于计算 loss 的函数，默认使用 :meth:`~collie.module.GPTLMLoss`
    :param train_fn: 用于训练的函数，默认使用 :meth:`~collie.controller.Trainer.train_fn`
    :param eval_fn: 用于验证的函数

        .. note::

            **CoLLie** 未提供默认的验证策略，若未传入 ``eval_fn``，但传入了 ``eval_dataset``，则会抛出异常。若不需要自定义验证循环，
            可以考虑使用 **CoLLie** 定义的多种验证器，例如 :class:`~collie.controller.evaluator.EvaluatorForPerplexity`、
            :class:`~collie.controller.evaluator.EvaluatorForClassfication`、:class:`~collie.controller.evaluator.EvaluatorForGeneration` 等。
        
    :param optimizer: 训练过程中的优化器，当为 `None` 的时候会尝试使用 ``config.ds_config`` 定义的优化器
    :param lr_scheduler: 训练过程中的学习率调度器；
    :param train_dataset: 用于训练的数据集。
    :param eval_dataset: 用于验证的数据集。
        **CoLLie** 可接收的 ``train_dataset`` 和 ``eval_dataset`` 为可迭代对象，例如 ``torch.utils.data.Dataset`` 或 ``List``。
        可以使用 :class:`~collie.data.CollieDatasetForTraining` 快速将数据集转换为 **CoLLie** 可接收的数据集。
        
        .. note::

            当未提供 ``train_dataset_collate_fn`` 或 ``eval_dataset_collate_fn`` 时，``train_dataset`` 和 ``eval_dataset`` 
            的取值应当为 `Dict` 类型
                
            注意: 上述数据格式为训练所需的格式, 同时 **CoLLie** 提供了多种验证器, 所要求的格式各有不同, 详见 :class:`~collie.controller.evaluator.Evaluator`
    
    :param callbacks: 训练中触发的 :class:`.Callback` 类，可以是列表。
    :param train_dataset_collate_fn: 用于训练数据集的 `collate_fn`。
    :param eval_dataset_collate_fn: 用于验证数据集的 `collate_fn`。
        ``train_dataset_collate_fn`` 与 ``eval_dataset_collate_fn`` 只可接受一个参数，为 ``train_dataset`` 或 ``eval_dataset`` 迭代值组成的 ``List``。
        
        .. note::

            ``train_dataset_collate_fn`` 和 ``eval_dataset_collate_fn`` 的返回值必须是 `Dict` 类型
                
            注意: 上述数据格式为训练所需的格式, 同时 **CoLLie** 提供了多种验证器, 所要求的格式各有不同, 详见 :class:`~collie.controller.evaluator.Evaluator`
                
        例如:

        .. code-block:: python
    
            from transformers import AutoTokenizer
            def collate_fn(batch):
                # batch = ["样本1", "样本2", ...]
                tokenizer = AutoTokenizer.from_pretrained("fnlp/moss-moon-003-sft", padding_side="left", trust_remote_code=True)
                input_ids = tokenizer(batch, return_tensors="pt", padding=True)["input_ids"]
                return {"input_ids": input_ids, "labels": input_ids}
            
    :param data_provider: 额外的数据提供器，可在 ``eval_dataset`` 之外额外注入验证数据，例如通过前端网页或 http 请求等， 详见 :class:`~collie.utils.data_provider.BaseProvider`
    :param monitors: 用于监控训练过程的监控器，详见 :class:`~collie.utils.monitor.BaseMonitor`
    :param metrics: 用于传给 ``Trainer`` 内部训练过程中的对 eval_dataset 进行验证。
        其应当为一个字典，其中 key 表示 monitor，value 表示一个
        metric，例如 ``{"acc1": Accuracy(), "acc2": Accuracy()}``

        目前我们支持的 ``metric`` 的种类有以下几种：

        * Collie 自己的 ``metric``：详见 :class:`.BaseMetric`
        * 继承 Collie 基类的自定义 Metric
    :param evaluators: 验证器。当传入多个 :class:`.Evaluator` 时会依次执行
        evaluator 的验证方法。
    """
    def __init__(self, 
                 model: torch.nn.Module,
                 config: CollieConfig,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 loss_fn: Callable = GPTLMLoss(),
                 train_fn: Optional[Callable] = None,
                 eval_fn: Optional[Callable] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 lr_scheduler: Optional[Union[_LRScheduler, DeepSpeedSchedulerCallable]] = None,
                 train_dataset: Optional[torch.utils.data.Dataset] = None,
                 eval_dataset: Optional[torch.utils.data.Dataset] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 train_dataset_collate_fn: Optional[Callable] = ColliePadder(),
                 eval_dataset_collate_fn: Optional[Callable] = ColliePadder(padding_left=True),
                 data_provider: Optional[BaseProvider] = None,
                 monitors: Sequence[BaseMonitor] = [],
                 metrics: Optional[Dict] = None,
                 evaluators: Optional[List] = None) -> None:
        self.config = config
        if isinstance(optimizer, Lomo):
            if config.pp_size > 1:
                raise ValueError("Lomo is incompatible with pipeline parallelism.")
            if self.config.gradient_accumulation_steps > 1:
                logger.rank_zero_warning(
                    f"Lomo is incompatible with gradient accumulation, "
                    f"set gradient_accumulation_steps from {self.config.gradient_accumulation_steps} to 1."
                )
                self.config.ds_config["gradient_accumulation_steps"] = 1

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.loss_fn = loss_fn
        if train_fn is not None:
            self.train_fn = train_fn
        if eval_fn is not None:
            self.eval_fn = eval_fn
        self.train_dataset_collate_fn = train_dataset_collate_fn
        self.eval_dataset_collate_fn = eval_dataset_collate_fn
        if self.tokenizer is not None and self.tokenizer.pad_token_id is not None \
            and isinstance(self.train_dataset_collate_fn, ColliePadder) \
                and "input_ids" not in self.train_dataset_collate_fn.padding_token_id.keys():
            self.train_dataset_collate_fn.padding_token_id["input_ids"] = self.tokenizer.pad_token_id
        if self.tokenizer is not None and self.tokenizer.pad_token_id is not None \
            and isinstance(self.eval_dataset_collate_fn, ColliePadder) \
                and "input_ids" not in self.eval_dataset_collate_fn.padding_token_id.keys():
            self.train_dataset_collate_fn.padding_token_id["input_ids"] = self.tokenizer.pad_token_id
        
        callbacks = prepare_callback(callbacks)
        self.callback_manager = CallbackManager(callbacks)
        self.setup_parallel_model()
        if isinstance(self.engine.module, PipelineGenerationMixin):
            self.engine.module.set_engine(self.engine)
        if isinstance(self.engine.module, PeftModel) and isinstance(self.engine.module.get_base_model(), PipelineGenerationMixin):
            self.engine.module.get_base_model().set_engine(self.engine)
        self.data_provider = data_provider
        self.monitor = _MultiMonitors(monitors)
        self.server = None
        if self.data_provider is not None:
            self.server = Server(model=self.model, data_provider=self.data_provider)
            self.server.start()
        if evaluators is not None and eval_dataset is not None:
            logger.rank_zero_warning(
                "Note that you have set both `evaluators` and `eval_dataset` "
                "and the later will not take effect."
            )
        if evaluators is None:
            evaluators = []
        if not isinstance(evaluators, Sequence):
            evaluators = [evaluators]
        if self.eval_dataset is not None:
            assert eval_fn is not None, "eval_fn should not be None when eval_dataset is not None."
            evaluator = Evaluator(model=self.model, dataset=eval_dataset, metrics=metrics, eval_fn=eval_fn,
                config=config, collate_fn=eval_dataset_collate_fn, data_provider=None)
            evaluator.monitor = self.monitor
            evaluators.append(evaluator)
        for evaluator in evaluators:
            if self.tokenizer is not None:
                evaluator.tokenizer = self.tokenizer
            evaluator.engine = self.engine
            evaluator.server = self.server
            evaluator.model = self.model
        self.evaluators = evaluators

        self.checkpoint_file = "collie_dp{}_pp{}_tp{}.pt".format(
            env.dp_rank, env.pp_rank, env.tp_rank
        )
        self.zero_checkpoint_file = "collie_zero_dp{}_pp{}_tp{}.pt".format(
            env.dp_rank, env.pp_rank, env.tp_rank
        )

        self.init_state_dict()
        self.on_after_trainer_initialized()
        torch.cuda.empty_cache()

    def init_state_dict(self):
        """初始化优化器的自身状态字典
        """
        self.epoch_idx = 0
        self.batch_idx = 0
            
    def state_dict(self):
        """获取优化器的自身状态字典
        """
        return {
            "epoch_idx": self.epoch_idx,
            "batch_idx": self.batch_idx
        }
        
    def load_state_dict(self, state_dict: dict):
        """加载优化器的自身状态
        """
        self.epoch_idx = state_dict["epoch_idx"]
        self.batch_idx = state_dict["batch_idx"]
        
    @property
    def global_batch_idx(self):
        """获取当前全局步数
        """
        return self.epoch_idx * self.steps_per_epoch + self.batch_idx

    def setup_parallel_model(self):
        """
        初始化分布式模型。
        """
        if dist.get_world_size() != self.config.tp_size * self.config.dp_size * self.config.pp_size:
            logger.rank_zero_warning("The world size is not equal to the product of the parallel sizes set."
                                     f"{dist.get_world_size()} != {self.config.tp_size} * {self.config.dp_size} * {self.config.pp_size}.")
            self.config.dp_size = dist.get_world_size() // (self.config.tp_size * self.config.pp_size)
            logger.rank_zero_warning(f"Set dp_size to {self.config.dp_size}.")
        self.on_setup_parallel_model()
        if self.config.pp_size > 1:
            # GPTLMLoss 是 Module，会被 nn.Module 加入 _Modules
            # 如果 loss_fn 是一个函数就会在此时报错
            if not isinstance(self.loss_fn, torch.nn.Module):
                del self.model.loss_fn
            self.model.loss_fn = self.loss_fn
        if isinstance(self.optimizer, Lomo):
            self.engine, _, _, _ = setup_ds_engine(
                model=self.model,
                config=self.config,
            )
        else:
            self.engine, self.optimizer, _, self.lr_scheduler = setup_ds_engine(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                config=self.config
            )
        self.config.train_micro_batch_size = self.engine.train_micro_batch_size_per_gpu()
        self.config.gradient_accumulation_steps = self.engine.gradient_accumulation_steps()
        # train_dataloader
        if self.train_dataset is None:
            self.train_dataloader = None
            self.steps_per_epoch = 0
        else:
            self.train_dataloader = CollieDataLoader(
                self.train_dataset, self.config.train_micro_batch_size,
                self.config.gradient_accumulation_steps, shuffle=True,
                collate_fn=self.train_dataset_collate_fn, drop_last=False
            )
            self.steps_per_epoch = len(self.train_dataloader)

        # set logger level
        deepspeed_logging_level = logging.ERROR if 'logging_level' not in self.config.ds_config \
            else self.config.ds_config['logging_level']
        deepspeed.utils.logging.logger.setLevel(deepspeed_logging_level)

    def train(self, dataloader: Optional[Iterable] = None):
        """训练循环
        
        :param dataloader: 用于训练的数据集，为 ``Iterable`` 对象 ，当为 ``None`` 时，使用由 ``train_dataset`` 生成的 ``train_dataloader``
        """
        train_dataloader = self.train_dataloader
        loss = None
        if dataloader is not None:
            train_dataloader = dataloader
        self.on_train_begin()
        tqbar_epoch = progress(
            range(self.epoch_idx, self.config.train_epochs),
            desc="Training Epoch", disable=env.rank != 0,
            completed=self.epoch_idx, total=self.config.train_epochs
        )
        tqbar_batch = progress(
            train_dataloader, desc="Training Batch: ",
            disable=env.rank != 0, total=self.steps_per_epoch
        )
        for self.epoch_idx in tqbar_epoch:
            self.on_train_epoch_begin()
            tqbar_epoch.set_description(f"Training Epoch: {self.epoch_idx} / {self.config.train_epochs}")
            tqbar_batch.reset(
                f"Training Batch: {self.batch_idx} / {self.steps_per_epoch}",
                completed=self.batch_idx,
            )
            for self.batch_idx, batch in enumerate(tqbar_batch, start=self.batch_idx):     
                tqbar_batch.set_description(f"Training Batch: {self.batch_idx} / {self.steps_per_epoch}")
                if self.server is not None:
                    self.server.data_provider_handler()
                self.engine.train()
                self.on_train_batch_begin(batch)
                if isinstance(self.engine.module, PipelineModel):
                    self.engine.module.forward_type = "train"
                if isinstance(self.engine.module, PeftModel) and isinstance(self.engine.module.get_base_model(), PipelineModel):
                    self.engine.module.get_base_model().forward_type = "train"
                with self.monitor as item:
                    loss = self.train_fn(self, batch, self.global_batch_idx)
                    item.update({"loss": round(loss, 4),
                                    "lr": self.lr,
                                    "batch": batch,
                                    "batch_idx": self.batch_idx,
                                    "epoch_idx": self.epoch_idx,
                                    "global_batch_idx": self.global_batch_idx,
                                    "memory_allocated": torch.cuda.max_memory_allocated(),
                                    "mode": "train"}
                        )
                tqbar_batch.set_postfix(Loss=round(loss, 4))
                self.on_train_batch_end(loss)
                if self.config.eval_per_n_steps > 0 and (self.batch_idx + 1) % self.config.eval_per_n_steps == 0:
                    self.eval()
            if self.config.eval_per_n_epochs > 0 and (self.epoch_idx + 1) % self.config.eval_per_n_epochs == 0:
                self.eval()
            self.on_train_epoch_end()
            self.batch_idx = 0
        self.on_train_end()
        self.epoch_idx = 0
                
    def eval(self, dataloader: Optional[Iterable] = None):
        """验证循环

        :param dataloader: 用于验证的数据集，为 ``Iterable`` 对象 ，当为 ``None`` 时，使用 ``eval_dataset`` 生成的 ``eval_dataloader``
        """
        if len(self.evaluators) == 0:
            return
        self.on_evaluate_begin()
        eval_results = {}
        for evaluator in self.evaluators:
            evaluator.global_batch_idx = self.global_batch_idx
            results = evaluator.eval(dataloader)
            eval_results.update(results)
        # TODO deal with results
        self.on_evaluate_end(results)

    @staticmethod
    def train_fn(trainer, batch: Dict, global_step: int) -> float:
        """一次训练的基本单元

        :param trainer: 训练器
        :param batch: 一个 batch 的数据，类型为 ``Dict``

            .. note::
                
                根据提供的 ``train_dataset`` 和 ``train_dataset_collate_fn`` 的不同，`labels` 的类型也会有所不同，详见 :class:`.Trainer`
    
        :param global_step: 当前的全局步数
        
        :return: 当前 batch 的 loss
        """
        if trainer.config.pp_size > 1:
            if isinstance(trainer.engine.module, PipelineModel):
                trainer.engine.module.forward_type = "train"
            if isinstance(trainer.engine.module, PeftModel) and isinstance(trainer.engine.module.get_base_model(), PipelineModel):
                trainer.engine.modulee.get_base_model().forward_type = "train"
            loss = trainer.engine.module(**batch)["loss"]
        else:
            # concat prompt labels for p-tuning
            if trainer.config.peft_config and trainer.config.peft_config.peft_type in ["PROMPT_TUNING", "P_TUNING"]:
                batch_size = batch["input_ids"].shape[0]
                prefix_labels = torch.full((batch_size, trainer.config.peft_config.num_virtual_tokens), -100).to(batch['labels'].device)
                batch['labels'] = torch.cat((prefix_labels, batch['labels']), dim=1)
            outputs = trainer.engine(**batch)
            loss = auto_param_call(trainer.loss_fn, {**batch, **outputs}, 
                                   signature_fn=trainer.loss_fn.forward if isinstance(trainer.loss_fn, nn.Module) else trainer.loss_fn)
            if not isinstance(trainer.optimizer, Lomo):
                trainer.engine.backward(loss)
                trainer.engine.step()
            else:
                # for lomo only
                if trainer.optimizer.clip_grad_norm is not None:
                    trainer.optimizer.grad_norm(loss)
                    if trainer.optimizer.loss_scaler and trainer.optimizer.loss_scaler.has_overflow_serial:
                        print(f"Gradient overflow, skipping step {global_step}")
                        if trainer.optimizer.zero3_enabled:
                            trainer.engine.optimizer.get_param_coordinator(training=True).reset_step()
                        return loss.detach().cpu().item()
                    if trainer.optimizer.zero3_enabled:
                        trainer.engine.optimizer.get_param_coordinator(training=True).reset_step()
                        # zero-3 doesn't support backward twice, so need an additional forward here
                        outputs = trainer.engine(**batch)
                        loss = auto_param_call(trainer.loss_fn, {**batch, **outputs}, 
                                               signature_fn=trainer.loss_fn.forward if isinstance(trainer.loss_fn, nn.Module) else trainer.loss_fn)
                if trainer.lr_scheduler:
                    lr = trainer.lr_scheduler.step(global_step)
                else:
                    lr = trainer.optimizer.lr
                trainer.optimizer.fused_backward(loss, lr)
                if trainer.optimizer.zero3_enabled:  # TODO: should tp do this too?
                    trainer.engine.optimizer.get_param_coordinator(training=True).reset_step()
        return loss.detach().cpu().item()
    
    @property
    def lr(self):
        if self.lr_scheduler:
            lr = self.lr_scheduler.get_last_lr()[0]
        else:
            lr = self.optimizer.param_groups[0]['lr']
        return lr
    
    def save_peft(self, path: str, process_exclusion: bool = False, adapter_name="default", **kwargs):...
    def save_peft(self, path: str, process_exclusion: bool = False, adapter_name="default",
                   protocol: str = "file",  **kwargs):
        """
        保存 adapter 部分权重，当未使用 ``peft`` 时，该方法等同于 ``save_model``
        
        :param path: 模型保存路径
        """
        
        if not isinstance(self.model, PeftModel):
            return self.save_model(path=path, protocol=protocol, process_exclusion=process_exclusion, **kwargs)
        io_driver = IODriver.from_protocol(protocol)
        io_driver.makedirs(path, exist_ok=True)
        # TODO 支持原 peft 里那样多个 adapter 的保存和加载
        contexts = []
        state_dict = get_peft_model_state_dict(
            self.model, adapter_name=adapter_name
        )
        named_parameters = {name: param for name, param in self.engine.module.named_parameters() if any([name.replace(f"{adapter_name}.", "") == k for k in state_dict.keys()])}
        if adapter_name == "default":
            name_prefix = "adapter_model"
        else:
            name_prefix = f"adapter_model_{adapter_name}"
        if env.pp_size == 1:
            name = f"{name_prefix}.bin"
        else:
            name = f"{name_prefix}_{env.pp_rank}.bin"
        if is_zero3_enabled(self.config):
            contexts.append(deepspeed.zero.GatheredParameters(list(named_parameters.values())))
        with ContextManagers(contexts):
            if env.dp_rank == 0 or not is_zero3_enabled(self.config):
                for key in named_parameters.keys():
                    state_dict[key.replace(f"{adapter_name}.", "")] = named_parameters[key].data
                if env.dp_rank == 0 and env.tp_rank == 0:
                    io_driver.save(state_dict, os.path.join(path, name))
        if env.rank == 0:
            io_driver.save(json.dumps(self.config.peft_config.__dict__), os.path.join(path, "adapter_config.json"))
        
    def load_peft(self, path: str, process_exclusion: bool = False, adapter_name="default", **kwargs):...
    def load_peft(self, path: str, process_exclusion: bool = False, adapter_name="default",
                   protocol: str = "file", **kwargs):
        """
        加载 adapter 部分权重，当未使用 ``peft`` 时，该方法等同于 ``load_model``
        
        :param path: 模型保存路径
        """
        io_driver = IODriver.from_protocol(protocol)
        if not isinstance(self.model, PeftModel):
            return self.load_model(path=path, protocol=protocol, process_exclusion=process_exclusion, **kwargs)
        peft_config_dict = json.loads(io_driver.load(os.path.join(path, "adapter_config.json"), mode="j"))
        loaded_peft_config = PeftConfig()
        for key, value in peft_config_dict.items():
            if hasattr(loaded_peft_config, key):
                setattr(loaded_peft_config, key, value)
        if loaded_peft_config.task_type != self.config.peft_config.task_type:
            raise ValueError(
                f"The task type `{loaded_peft_config.task_type}` from "
                f"checkpoint `{path}` is not the current task type"
                f"{self.config.peft_config.task_type}"
            )
        if adapter_name == "default":
            name_prefix = "adapter_model"
        else:
            name_prefix = f"adapter_model_{adapter_name}"
        if env.pp_size == 1:
            name = f"{name_prefix}.bin"
        else:
            name = f"{name_prefix}_{env.pp_rank}.bin"
        assert io_driver.exists(os.path.join(path, name)), f"{name} does not exist."
        loaded_state_dict = io_driver.load(os.path.join(path, name), mode="rb")
        named_parameters = {name: param for name, param in self.engine.module.named_parameters() if any([name.replace(f"{adapter_name}.", "") == k for k in loaded_state_dict.keys()])}
        contexts = []
        if is_zero3_enabled(self.config):
            contexts.append(deepspeed.zero.GatheredParameters(list(named_parameters.values()), modifier_rank=0))
        with ContextManagers(contexts):
            if env.dp_rank == 0 or not is_zero3_enabled(self.config):
                set_peft_model_state_dict(self.model, loaded_state_dict,
                                          adapter_name="default")
                    
    def save_model(self, path: str, process_exclusion: bool = False, **kwargs):...
    def save_model(self, path: str, process_exclusion: bool = False,
                   protocol: str = "file", **kwargs):
        """
        保存模型。

        :param path: 模型保存路径
        :param process_exclusion: 是否开启进程互斥，当开启流水线并行时开启此项可以节省内存（仅限 **CoLLie** 内实现的模型，对 `transformers` 提供的模型本项无效）
        """
        dist.barrier()
        io_driver = IODriver.from_protocol(protocol)
        io_driver.makedirs(path, exist_ok=True)
        self.on_save_model()
        if isinstance(self.engine.module, CollieModelForCausalLM) or isinstance(self.engine.module, PipelineModel):
            if is_zero3_enabled(self.config):
                self._checkpoint_prologue()
                with deepspeed.zero.GatheredParameters(list(self.engine.module.parameters(recurse=True))):
                    self.engine.module.save_parallel_state_dict(
                        state_dict=self.engine.module.state_dict(),
                        path=path,
                        config=self.config,
                        process_exclusion=process_exclusion,
                        protocol=protocol
                    )
                self._checkpoint_epilogue()
            else:
                self.engine.module.save_parallel_state_dict(
                    state_dict=self.engine.module.state_dict(),
                    path=path,
                    config=self.config,
                    process_exclusion=process_exclusion,
                    protocol=protocol
                )
        elif isinstance(self.engine.module, PreTrainedModel):
            if is_zero3_enabled(self.config):
                self._checkpoint_prologue()
                with deepspeed.zero.GatheredParameters(list(self.engine.module.parameters(recurse=True))):
                    self.engine.module.save_pretrained(
                        save_directory=path,
                        **kwargs
                    )
                self._checkpoint_epilogue()
            else:
                self.engine.module.save_pretrained(
                    save_directory=path,
                    **kwargs
                )

    def load_model(self, path: str, process_exclusion: bool = False, **kwargs):...
    def load_model(self, path: str, process_exclusion: bool = False,
                   protocol: str = 'file', **kwargs):
        io_driver = IODriver.from_protocol(protocol)
        assert io_driver.exists(path), f"`{path}` does not exist."
        self.on_load_model()
        if isinstance(self.engine.module, CollieModelForCausalLM) or isinstance(self.engine.module, PipelineModel):
            if is_zero3_enabled(self.config):
                self._checkpoint_prologue()
                with deepspeed.zero.GatheredParameters(list(self.engine.module.parameters(recurse=True)), modifier_rank=0):
                    if env.rank == 0:
                        self.engine.module.load_state_dict(self.engine.module.load_parallel_state_dict(
                            path=path, config=self.config, process_exclusion=process_exclusion, protocol=protocol
                        ))
                self._checkpoint_epilogue()
            else:
                self.engine.module.load_state_dict(
                    self.engine.module.load_parallel_state_dict(
                        path=path, config=self.config, process_exclusion=process_exclusion, protocol=protocol
                    )
                )
        elif isinstance(self.engine.module, PreTrainedModel):
            if is_zero3_enabled(self.config):
                index = None
                if io_driver.exists(os.path.join(path, "pytorch_model.bin.index.json")):
                    weight_map = json.loads(io_driver.load(os.path.join(path, "pytorch_model.bin.index.json"), mode="r"))["weight_map"]
                    index = OrderedDict()
                    for key, value in weight_map.items():
                        if value not in index.keys():
                            index[value] = [key]
                        else:
                            index[value].append(key)
                self._checkpoint_prologue()
                if index is not None:
                    for key, value in index.items():
                        with deepspeed.zero.GatheredParameters([self.engine.module.state_dict()[attr] for attr in value], modifier_rank=0):
                            if env.dp_rank == 0:
                                state_dict = io_driver.load(os.path.join(path, key), mode="br")
                                for attr in value:
                                    self.engine.module.state_dict()[attr].copy_(state_dict[attr])
                else:
                    with deepspeed.zero.GatheredParameters(list(self.engine.module.parameters(recurse=True)), modifier_rank=0):
                        if env.dp_rank == 0:
                            state_dict = reduce(lambda x, y: {**x, **y}, [io_driver.load(file, mode="rb") for file in glob.glob(os.path.join(path, "*.bin"))])
                            self.engine.module.load_state_dict(state_dict)
                self._checkpoint_epilogue()
            else:
                index = None
                if io_driver.exists(os.path.join(path, "pytorch_model.bin.index.json")):
                    weight_map = json.loads(io_driver.load(os.path.join(path, "pytorch_model.bin.index.json"), mode="r"))["weight_map"]
                    index = OrderedDict()
                    for key, value in weight_map.items():
                        if value not in index.keys():
                            index[value] = [key]
                        else:
                            index[value].append(key)
                if index is not None:
                    for key, value in index.items():
                        state_dict = io_driver.load(os.path.join(path, key), mode="br")
                        for attr in value:
                            self.engine.module.state_dict()[attr].copy_(state_dict[attr])
                else:
                    state_dict = reduce(lambda x, y: {**x, **y}, [io_driver.load(file) for file in glob.glob(os.path.join(path, "*.bin"))])
                    self.engine.module.load_state_dict(state_dict)

    def save_checkpoint(self, path: str, process_exclusion: bool = False, **kwargs):...
    def save_checkpoint(self, path: str, process_exclusion: bool = False, 
                        protocol: str="file", **kwargs):
        """保存训练器断点功能

        :param path: 断点保存路径
        :param process_exclusion: 是否开启进程互斥，当开启流水线并行时开启此项可以节省内存（仅限 **CoLLie** 内实现的模型，对 `transformers` 提供的模型本项无效）
        """
        dist.barrier()
        io_driver = IODriver.from_protocol(protocol)
        io_driver.makedirs(path, exist_ok=True)
        callback_states = self.on_save_checkpoint()
        # save parallel_settings
        if env.dp_rank == 0:
            dist_config = {
                "dp_size": env.dp_size, "tp_size": env.tp_size,
                "pp_size": env.pp_size
            }
            io_driver.save(json.dumps(dist_config), os.path.join(path, "collie.json"))
        engine = self.engine
        # DeepSpeedEngine.save_checkpoint

        self.save_model(path, protocol=protocol)

        # Prepare for checkpoint save by ensuring all parameters are partitioned
        self._checkpoint_prologue()
        
        ## DeepSpeedEngine._save_checkpoint
        zero_optimizer_state = engine.zero_optimization() or engine.bfloat16_enabled()
        state = dict(optimizer=engine.optimizer.state_dict() if engine.optimizer and not zero_optimizer_state else None,
                    lr_scheduler=engine.lr_scheduler.state_dict() if engine.lr_scheduler is not None else None,
                    data_sampler=engine.training_dataloader.data_sampler.state_dict() if
                    (engine.training_dataloader is not None and engine.curriculum_learning_enabled()) else None,
                    sparse_tensor_module_names=engine.sparse_tensor_module_names,
                    skipped_steps=engine.skipped_steps,
                    global_steps=engine.global_steps,
                    global_samples=engine.global_samples,
                    callback_states=callback_states)

        if env.rank == 0 or engine.zero_optimization_partition_weights():
            io_driver.save(state, os.path.join(path, self.checkpoint_file))

        if engine.save_zero_checkpoint:
            self._save_zero_checkpoint(path, io_driver)

        self._checkpoint_epilogue()

        dist.barrier()

    def load_checkpoint(self, path: str, process_exclusion: bool = False, **kwargs):...
    def load_checkpoint(self, path: str, process_exclusion: bool = False,
                        protocol: str = 'file', **kwargs):
        """训练器断点加载

        :param path: 断点保存路径
        :param process_exclusion: 是否开启进程互斥，当开启流水线并行时开启此项可以节省内存（仅限 **CoLLie** 内实现的模型，对 ``transformers`` 提供的模型本项无效）
        """
        io_driver = IODriver.from_protocol(protocol)
        assert io_driver.exists(path), f"`{path}` does not exist."
        engine = self.engine
        # check
        loaded_args = json.loads(io_driver.load(os.path.join(path, "collie.json"), "r"))
        assert loaded_args["dp_size"] == env.dp_size and \
            loaded_args["tp_size"] == env.tp_size and \
            loaded_args["pp_size"] == env.pp_size, \
            "Loaded checkpoint's world_size is not equal to the current " \
            f"settings: dp * tp * pp {loaded_args['dp_size']} * " \
            f"{loaded_args['tp_size']} * {loaded_args['pp_size']}" \
            f"!= {env.dp_size} * {env.tp_size} * {env.pp_size}."
        
        self.load_model(path, protocol=protocol)

        # DeepSpeed.load_checkpoint
        if engine.zero_optimization_partition_weights():
            ckpt_file = self.checkpoint_file
        else:
            ckpt_file = "collie_dp0_pp0_tp0.pt"
        checkpoint = io_driver.load(os.path.join(path, ckpt_file), "b")

        # Prepare for checkpoint load by ensuring all parameters are partitioned
        self._checkpoint_prologue()

        has_zero_optimizer_state = engine.zero_optimization() or engine.bfloat16_enabled()

        if engine.optimizer is not None and not has_zero_optimizer_state:
            engine.optimizer.load_state_dict(checkpoint['optimizer'])

        if engine.lr_scheduler is not None:
            engine.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        if engine.training_dataloader is not None and engine.curriculum_learning_enabled(
        ) and 'data_sampler' in checkpoint:
            engine.training_dataloader.data_sampler.load_state_dict(checkpoint['data_sampler'])

        if 'sparse_tensor_module_names' in checkpoint:
            sparse_tensor_module_names = checkpoint['sparse_tensor_module_names']
        elif 'csr_tensor_module_names' in checkpoint:
            sparse_tensor_module_names = checkpoint['csr_tensor_module_names']
        else:
            sparse_tensor_module_names = None
        if sparse_tensor_module_names is not None:
            engine.sparse_tensor_module_names = sparse_tensor_module_names

        engine.global_steps = checkpoint['global_steps']
        engine.global_samples = checkpoint.get('global_samples', engine.global_steps * engine.train_batch_size())
        engine.skipped_steps = checkpoint['skipped_steps']

        # load_zero_checkpoint = engine.zero_optimization() or engine.bfloat16_enabled()
        load_zero_checkpoint = engine.save_zero_checkpoint
        if load_zero_checkpoint:
            success = self._load_zero_checkpoint(path, io_driver)
            if not success:
                engine.optimizer._restore_from_bit16_weights()

        self._checkpoint_epilogue()

        self.on_load_checkpoint(checkpoint["callback_states"])

    def _save_zero_checkpoint(self, path, driver):
        """保存 `ZeRO` 的状态
        """
        zero_path = os.path.join(path, self.zero_checkpoint_file)
        zero_sd = self.engine.optimizer.state_dict()
        driver.save(zero_sd, zero_path)

    def _load_zero_checkpoint(self, path, driver):
        """加载 `ZeRO` 的状态
        """
        engine = self.engine
        
        zero_sd_list = []
        for dp_rank in range(engine.dp_world_size):
            zero_ckpt = os.path.join(path, self.zero_checkpoint_file)
            zero_ckpt = zero_ckpt.replace(F"dp{env.dp_rank}", f"dp{dp_rank}")
            zero_sd_list.append(driver.load(zero_ckpt, "b"))

        engine.optimizer.load_state_dict(
            state_dict_list=zero_sd_list,
            load_from_fp32_weights=engine.zero_load_from_fp32_weights(),
        )

        return True

    def _checkpoint_prologue(self):
        # Lomo + zero3 时因为没有将 optimizer 传给 engine
        # engine.optimizer 是 DeepSpeedZeRoOffload 而非 stage 3
        # 故进行判断
        if not self.engine.zero_optimization_partition_weights():
            return
        if isinstance(self.optimizer, Lomo):
            param_offload = self.engine.optimizer
        else:
            param_offload = self.engine.optimizer.parameter_offload
        param_offload.partition_all_parameters()
        
    def _checkpoint_epilogue(self):
        if not self.engine.zero_optimization_partition_weights():
            return
        if isinstance(self.optimizer, Lomo):
            persistent_params = self.engine.optimizer.persistent_parameters
        else:
            persistent_params = self.engine.optimizer.parameter_offload.persistent_parameters
        persistent_params[0].all_gather(persistent_params)
        
        if len(persistent_params) > 0:
            persistent_params[0].all_gather(persistent_params)
