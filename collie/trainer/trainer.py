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
import math
from typing import Optional, Callable, Union, Tuple, Iterable, Any, Dict, Sequence, List
from collections import OrderedDict
from functools import reduce

import torch
import deepspeed
import torch.distributed as dist
from torch.optim.lr_scheduler import _LRScheduler
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.runtime.engine import DeepSpeedSchedulerCallable
from transformers.generation.utils import GenerationConfig
from transformers.modeling_utils import PreTrainedModel

from collie.config import CollieConfig
from collie.module import PipelineGenerationMixin, GPTLMLoss, PipelineModel
from collie.driver.io.file import FileIODriver
from collie.driver.io.petrel import PetrelIODriver
from collie.log import logger
from collie.utils import progress, env, setup_ds_engine, BaseProvider, _GenerationStreamer, is_zero3_enabled, BaseMonitor, _MultiMonitors, broadcast_tensor, ColliePadder
from collie.optim import InplaceSGD
from collie.models.base import CollieModelForCausalLM
from .evaluator import Evaluator
from collie.data import CollieDataLoader
from collie.callbacks.callback import Callback
from collie.callbacks.callback_manager import CallbackManager, prepare_callback
from .utils import TrainerEventTrigger

class Trainer(TrainerEventTrigger):
    r"""
    **CoLLie** 训练器，支持快速分布式训练和验证。

    :param model: 用于训练和验证的模型，可以使用 **CoLLie** 实现的模型或 transformers 提供的模型：

        * **CoLLie** 实现的模型 :class:`~collie.CollieModelForCausalLM` 可支持的并行方式包括：张量并行、流水线并行、`ZeRO`
        * transformers 提供的模型 ``transformers.PreTrainedModel`` 只支持 `ZeRO`
        
    :param config: 用于训练和验证的配置
    :param loss_fn: 用于计算 loss 的函数，默认使用 :meth:`~collie.module.GPTLMLoss`
    :param train_fn: 用于训练的函数，默认使用 :meth:`~collie.trainer.Trainer.train_fn`
    :param eval_fn: 用于验证的函数，默认使用 :meth:`~collie.trainer.Trainer.eval_fn`
    :param optimizer: 训练过程中的优化器，当为 `None` 的时候会尝试使用 ``config.ds_config`` 定义的优化器
    :param lr_scheduler: 训练过程中的学习率调度器；
    :param train_dataset: 用于训练的数据集。
    :param eval_dataset: 用于验证的数据集。
        **CoLLie** 可接收的 ``train_dataset`` 和 ``eval_dataset`` 为可迭代对象，例如 ``torch.utils.data.Dataset`` 或 ``List``
        
        .. note::

            当未提供 ``train_dataset_collate_fn`` 或 ``eval_dataset_collate_fn`` 时，``train_dataset`` 和 ``eval_dataset`` 的取值应当为长度为 **2** 的 `Tuple` 类型，例如 `(a, b)`，其中:
                
            * `a` 即 ``input_ids``， 为 ``torch.Tensor`` 类型，表示模型的的输入
            * `b` 可以为 ``torch.Tensor``，也可以是由 ``torch.Tensor`` 组成的任意长度的 `Tuple`，此项会作为 `loss_fn` 的第二个参数传入
    
    :param callbacks: 训练中触发的 :class:`.Callback` 类，可以是列表。
    :param train_dataset_collate_fn: 用于训练数据集的 `collate_fn`。
    :param eval_dataset_collate_fn: 用于验证数据集的 `collate_fn`。
        ``train_dataset_collate_fn`` 与 ``eval_dataset_collate_fn`` 只可接受一个参数，为 ``train_dataset`` 或 ``eval_dataset`` 迭代值组成的 ``List``。
        
        .. note::

            ``train_dataset_collate_fn`` 和 ``eval_dataset_collate_fn`` 的返回值必须是为长度为 **2** 的 `Tuple` 类型，，例如 `(a, b)`，其中:
            
            * `a` 即 ``input_ids``，为 ``torch.Tensor`` 类型，表示模型的的输入
            * `b` 可以为 ``torch.Tensor``，也可以是由 ``torch.Tensor`` 组成的任意长度的 ``Tuple``，此项会作为 ``loss_fn`` 的第二个参数传入
                
        例如:

        .. code-block:: python
    
            from transformers import AutoTokenizer
            def collate_fn(batch):
                # batch = ["样本1", "样本2", ...]
                tokenizer = AutoTokenizer.from_pretrained("fnlp/moss-moon-003-sft", padding_side="left", trust_remote_code=True)
                input_ids = tokenizer(batch, return_tensors="pt", padding=True)["input_ids"]
                # 第二个 input_ids 会被用于 loss_fn 的 label
                return input_ids, input_ids
            
    :param generation_config: 用于验证的配置
        **CoLLie** 默认的 ``eval_fn`` 为进行一次生成过程，因此本项配置主要控制生成过程的参数。当自定义 ``eval_fn`` 时，本项配置将不会生效
    :param data_provider: 额外的数据提供器，可在 ``eval_dataset`` 之外额外注入验证数据，例如通过前端网页或 http 请求等， 详见 :class:`~collie.utils.data_provider.BaseProvider`
    :param monitors: 用于监控训练过程的监控器，详见 :class:`~collie.utils.monitor.BaseMonitor`
    :param metrics: 用于传给 ``Trainer`` 内部训练过程中的对 eval_dataset 进行验证。
        其应当为一个字典，其中 key 表示 monitor，value 表示一个
        metric，例如 ``{"acc1": Accuracy(), "acc2": Accuracy()}``

        目前我们支持的 ``metric`` 的种类有以下几种：

        * Collie 自己的 ``metric``：详见 :class:`.BaseMetric`
        * 继承 Collie 基类的自定义 Metric
    """
    def __init__(self, 
                 model: torch.nn.Module,
                 config: CollieConfig,
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
                 generation_config: GenerationConfig = GenerationConfig(),
                 data_provider: Optional[BaseProvider] = None,
                 monitors: Sequence[BaseMonitor] = [],
                 metrics: Optional[Dict] = None,
                 evaluators: Optional[List] = None) -> None:
        self.config = config
        if isinstance(optimizer, InplaceSGD):
            if config.pp_size > 1:
                raise ValueError("InplaceSGD is incompatible with pipeline parallelism.")
            if self.config.gradient_accumulation_steps > 1:
                logger.rank_zero_warning(
                    f"InplaceSGD is incompatible with gradient accumulation, "
                    f"set gradient_accumulation_steps from {self.config.gradient_accumulation_steps} to 1."
                )
                self.config.ds_config["gradient_accumulation_steps"] = 1

        self.model = model
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
        self.generation_config = generation_config
        self.communicate_buffer_shape = None
        self.setup_parallel_model()
        get_accelerator().empty_cache()
        self.data_provider = data_provider
        self.monitor = _MultiMonitors(monitors)
        if self.data_provider is not None and dist.get_rank() == 0:
            self.data_provider.start_provider()

        if evaluators is not None and eval_dataset is not None:
            logger.rank_zero_warning(
                "Note that you have set both `evaluators` and `eval_dataset` "
                "and the later will not take effect."
            )

        if evaluators is None or (hasattr(evaluators, "__len__") and len(evaluators) == 0):
            if eval_dataset is not None:
                evaluators = Evaluator(model=model, dataset=eval_dataset, metrics=metrics, eval_fn=eval_fn,
                    config=config, collate_fn=eval_dataset_collate_fn, data_provider=None,
                    generation_config=generation_config)
                evaluators.engine = self.engine
                evaluators.monitor = self.monitor
                evaluators.data_provider = self.data_provider
            else:
                evaluators = []
        if not isinstance(evaluators, Sequence):
            evaluators = [evaluators]
        for evaluator in evaluators:
            evaluator.engine = self.engine

        self.evaluators = evaluators

        callbacks = prepare_callback(callbacks)
        self.callback_manager = CallbackManager(callbacks)

        self.checkpoint_file = "collie_dp{}_pp{}_tp{}.pt".format(
            env.dp_rank, env.pp_rank, env.tp_rank
        )
        self.zero_checkpoint_file = "collie_zero_dp{}_pp{}_tp{}.pt".format(
            env.dp_rank, env.pp_rank, env.tp_rank
        )

        self.init_state_dict()
        self.on_after_trainer_initialized()

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

    def data_provider_handler(self):
        """当初始化 :class:`collie.Trainer` 的过程中提供了 ``data_provider`` 时会使用此方法。
            ``data_provider`` 中维持一个异步队列 ``queue.Queue``，该方法会不断从中取出数据，放入模型中进行生成
        """
        if self.data_provider is None:
            return None
        has_data = torch.tensor(False).cuda()
        input_ids = None
        if dist.get_rank() == 0:
            input_ids = self.data_provider.get_data()
            if input_ids is not None:
                has_data = ~has_data
                input_ids = input_ids.cuda()
        dist.broadcast(has_data, 0)
        if not has_data:
            return
        input_ids = broadcast_tensor(input_ids, src=0)
        if isinstance(self.engine, PipelineEngine):
            generation_model = PipelineGenerationMixin(
                engine=self.engine
            )
        else:
            generation_model = self.engine.module
        if not generation_model.can_generate():
            return
        use_stream = self.data_provider.stream
        streamer = _GenerationStreamer(server=self.data_provider)
        generated_ids = generation_model.generate(
            input_ids=input_ids.cuda(), 
            attention_mask=torch.ones_like(input_ids).cuda(), 
            generation_config=self.generation_config,
            streamer=streamer if use_stream else None
        )
        if not use_stream:
            self.data_provider.put_feedback(generated_ids[0].cpu())
        get_accelerator().empty_cache()

    def setup_parallel_model(self):
        """
        初始化分布式模型。
        """
        if dist.get_world_size() != self.config.tp_size * self.config.dp_size * self.config.pp_size:
            logger.rank_zero_warning("The world size is not equal to the product of the parallel sizes set."
                                     f"{dist.get_world_size()} != {self.config.tp_size} * {self.config.dp_size} * {self.config.dp_size}.")
            self.config.dp_size = dist.get_world_size() // (self.config.tp_size * self.config.pp_size)
            logger.rank_zero_warning(f"Set dp_size to {self.config.dp_size}.")
        if self.config.pp_size > 1:
            # GPTLMLoss 是 Module，会被 nn.Module 加入 _Modules
            # 如果 loss_fn 是一个函数就会在此时报错
            if not isinstance(self.loss_fn, torch.nn.Module):
                del self.model.loss_fn
            self.model.loss_fn = self.loss_fn
        if isinstance(self.optimizer, InplaceSGD):
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
        loss = 0.0
        if dataloader is not None:
            train_dataloader = dataloader

        self.on_train_begin()
        with progress(range(self.epoch_idx, self.config.train_epochs), desc="Training Epoch", disable=env.rank != 0, completed=self.epoch_idx, total=self.config.train_epochs) as tqbar_epoch:
            for self.epoch_idx in tqbar_epoch:
                self.on_train_epoch_begin()
                tqbar_epoch.set_description(f"Training Epoch: {self.epoch_idx} / {self.config.train_epochs}")

                with progress(train_dataloader, desc="Training Batch: ", disable=env.rank != 0, completed=self.batch_idx, total=self.steps_per_epoch) as tqbar_batch:
                    for self.batch_idx, batch in enumerate(tqbar_batch, start=self.batch_idx):
                        tqbar_batch.set_description(f"Training Batch: {self.batch_idx} / {self.steps_per_epoch}")
                        self.data_provider_handler()
                        self.engine.train()
                        get_accelerator().empty_cache()
                        self.on_train_batch_begin(batch)
                        with self.monitor as item:
                            loss = self.train_fn(self, batch, self.epoch_idx * self.steps_per_epoch + self.batch_idx)
                            item.update({"loss": loss,
                                         "batch": batch,
                                         "batch_idx": self.batch_idx,
                                         "epoch_idx": self.epoch_idx,
                                         "global_batch_idx": self.global_batch_idx,
                                         "memory_allocated": torch.cuda.max_memory_allocated(),
                                         "mode": "train"})
                        tqbar_batch.set_postfix(Loss=round(loss, 4))
                        self.on_train_batch_end(loss)
                        if self.config.eval_per_n_steps > 0 and (self.batch_idx + 1) % self.config.eval_per_n_steps == 0:
                            self.eval()
                if self.config.eval_per_n_epochs > 0 and (self.epoch_idx + 1) % self.config.eval_per_n_epochs == 0:
                        self.eval()
                self.on_train_epoch_end()
                self.batch_idx = 0
                if self.config.eval_per_n_epochs > 0 and (self.epoch_idx + 1) % self.config.eval_per_n_epochs == 0:
                    self.eval()
                self.on_train_epoch_end()
        self.epoch_idx = 0
                
    def eval(self, dataloader: Optional[Iterable] = None):
        """验证循环

        :param dataloader: 用于验证的数据集，为 ``Iterable`` 对象 ，当为 ``None`` 时，使用 ``eval_dataset`` 生成的 ``eval_dataloader``
        """
        if len(self.evaluators) == 0:
            return
        self.on_evaluate_begin()
        eval_results = []
        for evaluator in self.evaluators:
            results = evaluator.eval(dataloader)
            eval_results.append(results)
        # TODO deal with results
        self.on_evaluate_end(results)

    @staticmethod
    def train_fn(trainer, batch: Tuple, global_step: int) -> float:
        """一次训练的基本单元

        :param trainer: 训练器
        :param batch: 一个 batch 的数据，类型为长度为 2 的 ``Tuple``，其中第一个元素为 ``input_ids``，第二个元素为 ``labels``

            .. note::
                
                根据提供的 ``train_dataset`` 和 ``train_dataset_collate_fn`` 的不同，`labels` 的类型也会有所不同，详见 :class:`~collie.trainer.Trainer`
    
        :param global_step: 当前的全局步数
        
        :return: 当前 batch 的 loss
        """
        if trainer.config.pp_size > 1:
            loss = trainer.engine.train_batch(batch)
        else:
            input_ids, labels = batch
            # concat prompt labels for p-tuning
            if trainer.config.peft_config and trainer.config.peft_config.peft_type in ["PROMPT_TUNING", "P_TUNING"]:
                batch_size = input_ids.shape[0]
                prefix_labels = torch.full((batch_size, trainer.config.peft_config.num_virtual_tokens), -100).to(labels.device)
                labels = torch.cat((prefix_labels, labels), dim=1)

            logits = trainer.engine(input_ids=input_ids.cuda()).logits
            loss = trainer.loss_fn(logits, labels)
            if not isinstance(trainer.optimizer, InplaceSGD):
                trainer.engine.backward(loss)
                trainer.engine.step()
            else:
                # for inplace_sgd only
                if trainer.optimizer.clip_grad_norm is not None:
                    trainer.optimizer.grad_norm(loss)
                    if trainer.optimizer.zero_enabled:
                        trainer.engine.optimizer.get_param_coordinator(training=True).reset_step()
                        # zero-3 doesn't support backward twice, so need an additional forward here
                        logits = trainer.engine(input_ids=input_ids.cuda()).logits
                        loss = trainer.loss_fn(logits, labels)
                if trainer.lr_scheduler:
                    lr = trainer.lr_scheduler.step(global_step)
                else:
                    lr = trainer.optimizer.lr
                trainer.optimizer.backward_step(loss, lr)
                if trainer.optimizer.zero_enabled:  # TODO: should tp do this too?
                    trainer.engine.optimizer.get_param_coordinator(training=True).reset_step()
        return loss.detach().cpu().item()

    def save_checkpoint(self, path: str, process_exclusion: bool = False, mode: str = "trainer", **kwargs):...
    def save_checkpoint(self, path: str, process_exclusion: bool = False, mode: str = "trainer", 
                        protocol: str="file", **kwargs):
        """保存训练器断点功能

        :param path: 断点保存路径
        :param process_exclusion: 是否开启进程互斥，当开启流水线并行时开启此项可以节省内存（仅限 **CoLLie** 内实现的模型，对 `transformers` 提供的模型本项无效）
        :param mode: 断点保存模式，支持 ``'trainer'`` 和 ``'model'`` 两种模式

            * ``'trainer'`` 模式下保存的断点包含了训练器的状态，包括模型参数、`epoch_idx` 和 `batch_idx` 等数据集状态、以及优化器的状态等
            * ``'model'`` 模式下保存的断点仅包含模型的状态，不包含训练器的状态
        """
        dist.barrier()
        assert protocol in ["file", "petrel"], f"Only support file and petrel protocol, not `{protocol}`."
        assert mode in ["trainer", "model"], f"Only support `trainer` and `model` mode, not `{mode}`."
        IODriver = FileIODriver if protocol == 'file' else PetrelIODriver
        IODriver.makedirs(path, exist_ok=True)
        if mode == "model":
            if isinstance(self.engine.module, CollieModelForCausalLM) or isinstance(self.engine.module, PipelineModel):
                if is_zero3_enabled(self.config):
                    with deepspeed.zero.GatheredParameters(list(self.engine.module.parameters(recurse=True))):
                        self.engine.module.save_parallel_state_dict(
                            state_dict=self.engine.module.state_dict(),
                            path=path,
                            config=self.config,
                            process_exclusion=process_exclusion,
                            protocol=protocol
                        )
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
                    with deepspeed.zero.GatheredParameters(list(self.engine.module.parameters(recurse=True))):
                        self.engine.module.save_pretrained(
                            save_directory=path,
                            **kwargs
                        )
                else:
                    self.engine.module.save_pretrained(
                        save_directory=path,
                        **kwargs
                    )
        if mode == "trainer":
            # save parallel_settings
            if env.dp_rank == 0:
                dist_config = {
                    "dp_size": env.dp_size, "tp_size": env.tp_size,
                    "pp_size": env.pp_size
                }
                IODriver.save(json.dumps(dist_config), os.path.join(path, "collie.json"))
            engine = self.engine
            # DeepSpeedEngine.save_checkpoint
            
            if engine.zero_optimization_partition_weights():
                # Prepare for checkpoint save by ensuring all parameters are partitioned
                engine.optimizer.checkpoint_event_prologue()

            ## DeepSpeedEngine._save_checkpoint
            zero_optimizer_state = engine.zero_optimization() or engine.bfloat16_enabled()
            state = dict(module=engine.module.state_dict(), 
                         optimizer=engine.optimizer.state_dict() if engine.optimizer and not zero_optimizer_state else None,
                        lr_scheduler=engine.lr_scheduler.state_dict() if engine.lr_scheduler is not None else None,
                        data_sampler=engine.training_dataloader.data_sampler.state_dict() if
                        (engine.training_dataloader is not None and engine.curriculum_learning_enabled()) else None,
                        sparse_tensor_module_names=engine.sparse_tensor_module_names,
                        skipped_steps=engine.skipped_steps,
                        global_steps=engine.global_steps,
                        global_samples=engine.global_samples)

            if env.rank == 0 or engine.zero_optimization_partition_weights():
                IODriver.save(state, os.path.join(path, self.checkpoint_file))

            if engine.save_zero_checkpoint:
                self._save_zero_checkpoint(path, IODriver)

            if engine.zero_optimization_partition_weights():
                engine.optimizer.checkpoint_event_epilogue()

        dist.barrier()

    def load_checkpoint(self, path: str, process_exclusion: bool = False, mode: str = "trainer", **kwargs):...
    def load_checkpoint(self, path: str, process_exclusion: bool = False, mode: str = "trainer",
                        protocol: str = 'file', **kwargs):
        """训练器断点加载

        :param path: 断点保存路径
        :param process_exclusion: 是否开启进程互斥，当开启流水线并行时开启此项可以节省内存（仅限 **CoLLie** 内实现的模型，对 ``transformers`` 提供的模型本项无效）
        :param mode: 断点加载模式，支持 ``'trainer'`` 和 ``'model'`` 两种模式

            * ``'trainer'`` 模式下加载的断点包含了训练器的状态，包括模型参数、``epoch_idx`` 和 ``batch_idx`` 等数据集状态、以及优化器的状态等
            * ``'model'`` 模式下加载的断点仅包含模型的状态，不包含训练器的状态
        """
        assert protocol in ["file", "petrel"], f"Only support file and petrel protocol, not `{protocol}`."
        assert mode in ["trainer", "model"], f"Only support `trainer` and `model` mode, not `{mode}`."
        IODriver = FileIODriver if protocol == 'file' else PetrelIODriver
        assert IODriver.exists(path), f"`{path}` does not exist."
        engine = self.engine
        if mode == "model":
            if isinstance(self.engine.module, CollieModelForCausalLM) or isinstance(self.engine.module, PipelineModel):
                if is_zero3_enabled(self.config):
                    if env.dp_rank == 0:
                        state_dict = self.engine.module.load_parallel_state_dict(
                            path=path, config=self.config, process_exclusion=process_exclusion, protocol=protocol
                            )
                    for attr in state_dict.keys():
                        with deepspeed.zero.GatheredParameters(self.engine.module.state_dict()[attr], modifier_rank=0):
                            if env.dp_rank == 0:
                                self.engine.module.state_dict()[attr].copy_(state_dict[attr])
                else:
                    self.engine.module.load_state_dict(
                        self.engine.module.load_parallel_state_dict(
                            path=path, config=self.config, process_exclusion=process_exclusion, protocol=protocol
                        )
                    )
            elif isinstance(self.engine.module, PreTrainedModel):
                if is_zero3_enabled(self.config):
                    index = None
                    if IODriver.exists(os.path.join(path, "pytorch_model.bin.index.json")):
                        weight_map = json.loads(IODriver.load(os.path.join(path, "pytorch_model.bin.index.json"), mode="r"))["weight_map"]
                        index = OrderedDict()
                        for key, value in weight_map.items():
                            if value not in index.keys():
                                index[value] = [key]
                            else:
                                index[value].append(key)
                    if index is not None:
                        for key, value in index.items():
                            with deepspeed.zero.GatheredParameters([self.engine.module.state_dict()[attr] for attr in value], modifier_rank=0):
                                if env.dp_rank == 0:
                                    state_dict = IODriver.load(os.path.join(path, key), mode="br")
                                    for attr in value:
                                        self.engine.module.state_dict()[attr].copy_(state_dict[attr])
                    else:
                        with deepspeed.zero.GatheredParameters(list(self.engine.module.parameters(recurse=True)), modifier_rank=0):
                            if env.dp_rank == 0:
                                state_dict = reduce(lambda x, y: {**x, **y}, [IODriver.load(file, mode="rb") for file in glob.glob(os.path.join(path, "*.bin"))])
                                self.engine.module.load_state_dict(state_dict)
                else:
                    index = None
                    if IODriver.exists(os.path.join(path, "pytorch_model.bin.index.json")):
                        weight_map = json.loads(IODriver.load(os.path.join(path, "pytorch_model.bin.index.json"), mode="r"))["weight_map"]
                        index = OrderedDict()
                        for key, value in weight_map.items():
                            if value not in index.keys():
                                index[value] = [key]
                            else:
                                index[value].append(key)
                    if index is not None:
                        for key, value in index.items():
                            state_dict = IODriver.load(os.path.join(path, key), mode="br")
                            for attr in value:
                                self.engine.module.state_dict()[attr].copy_(state_dict[attr])
                    else:
                        state_dict = reduce(lambda x, y: {**x, **y}, [IODriver.load(file, "b") for file in glob.glob(os.path.join(path, "*.bin"))])
                        self.engine.module.load_state_dict(state_dict)
        if mode == "trainer":
            # check
            loaded_args = json.loads(IODriver.load(os.path.join(path, "collie.json"), "r"))
            assert loaded_args["dp_size"] == env.dp_size and \
                loaded_args["tp_size"] == env.tp_size and \
                loaded_args["pp_size"] == env.pp_size, \
                "Loaded checkpoint's world_size is not equal to the current " \
                f"settings: dp * tp * pp {loaded_args['dp_size']} * " \
                f"{loaded_args['tp_size']} * {loaded_args['pp_size']}" \
                f"!= {env.dp_size} * {env.tp_size} * {env.pp_size}."

            # DeepSpeed.load_checkpoint
            if engine.zero_optimization_partition_weights():
                ckpt_file = self.checkpoint_file
            else:
                ckpt_file = "collie_dp0_pp0_tp0.pt"
            checkpoint = IODriver.load(os.path.join(path, ckpt_file), "b")

            module = checkpoint["module"]
            engine.module.load_state_dict(module)

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

            load_zero_checkpoint = engine.zero_optimization() or engine.bfloat16_enabled()
            if load_zero_checkpoint:
                success = self._load_zero_checkpoint(path, IODriver)
                if not success:
                    engine.optimizer._restore_from_bit16_weights()

            if engine.zero_optimization_partition_weights():
                engine.optimizer.checkpoint_event_epilogue()

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
