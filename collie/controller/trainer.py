"""
``Trainer`` 是 **CoLLie** 中的训练器，它负责整个训练过程的控制。包含训练功能、验证功能、保存断点功能等。
"""
__all__ = ["Trainer"]

import glob
import json
import logging
import os
from collections import OrderedDict
from functools import reduce
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from deepspeed.runtime.engine import DeepSpeedSchedulerCallable
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ContextManagers

from collie.callbacks.callback import Callback
from collie.callbacks.callback_manager import CallbackManager, prepare_callback
from collie.config import CollieConfig
from collie.data import CollieDataLoader
from collie.driver.io import IODriver
from collie.log import logger
from collie.models.base import CollieModelForCausalLM
from collie.module import GPTLMLoss, PipelineGenerationMixin, PipelineModel
from collie.utils import (
    BaseMonitor,
    ColliePadder,
    _MultiMonitors,
    auto_param_call,
    env,
    is_zero3_enabled,
    progress,
    setup_ds_engine,
)
from collie.utils.peft_utils import pp_merge_peft, load_peft
from peft import PeftModel, PeftType, PromptLearningConfig, get_peft_model_state_dict

from .evaluator import Evaluator
from .server import Server
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

    :param server: 用于打开一个交互界面，随时进行生成测试，详见 :class:`~collie.controller.server.Server`
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

    def __init__(
        self,
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
        server: Optional[Server] = None,
        monitors: Sequence[BaseMonitor] = [],
        metrics: Optional[Dict] = None,
        evaluators: Optional[List] = None,
    ) -> None:
        self.config = config
        if "Lomo" in optimizer.__class__.__name__:
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
        self.resume_from_checkpoint = False
        if train_fn is not None:
            self.train_fn = train_fn
        if eval_fn is not None:
            self.eval_fn = eval_fn
        self.train_dataset_collate_fn = train_dataset_collate_fn
        self.eval_dataset_collate_fn = eval_dataset_collate_fn
        if (
            self.tokenizer is not None
            and self.tokenizer.pad_token_id is not None
            and isinstance(self.train_dataset_collate_fn, ColliePadder)
            and "input_ids" not in self.train_dataset_collate_fn.padding_token_id.keys()
        ):
            self.train_dataset_collate_fn.padding_token_id[
                "input_ids"
            ] = self.tokenizer.pad_token_id
        if (
            self.tokenizer is not None
            and self.tokenizer.pad_token_id is not None
            and isinstance(self.eval_dataset_collate_fn, ColliePadder)
            and "input_ids" not in self.eval_dataset_collate_fn.padding_token_id.keys()
        ):
            self.train_dataset_collate_fn.padding_token_id[
                "input_ids"
            ] = self.tokenizer.pad_token_id

        callbacks = prepare_callback(callbacks)
        self.callback_manager = CallbackManager(callbacks)
        self.setup_parallel_model()
        if isinstance(self.engine.module, PipelineGenerationMixin):
            self.engine.module.set_engine(self.engine)
        if isinstance(self.engine.module, PeftModel) and isinstance(
            self.engine.module.get_base_model(), PipelineGenerationMixin
        ):
            self.engine.module.get_base_model().set_engine(self.engine)
        self.monitor = _MultiMonitors(monitors)
        self.server = server
        if self.server is not None:
            self.server.start()
        if evaluators is None:
            evaluators = []
        if not isinstance(evaluators, Sequence):
            evaluators = [evaluators]
        if self.eval_dataset is not None:
            assert (
                eval_fn is not None
            ), "eval_fn should not be None when eval_dataset is not None."
            evaluator = Evaluator(
                model=self.model,
                dataset=eval_dataset,
                metrics=metrics,
                eval_fn=eval_fn,
                config=config,
                collate_fn=eval_dataset_collate_fn,
            )
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
        """初始化优化器的自身状态字典"""
        self.epoch_idx = 0
        self.batch_idx = 0

    def state_dict(self):
        """获取优化器的自身状态字典"""
        return {"epoch_idx": self.epoch_idx, "batch_idx": self.batch_idx}

    def load_state_dict(self, state_dict: dict):
        """加载优化器的自身状态"""
        self.epoch_idx = state_dict["epoch_idx"]
        self.trained_batch_idx = state_dict["batch_idx"]
        self.resume_from_checkpoint = True

    @property
    def global_batch_idx(self):
        """获取当前全局步数"""
        return self.epoch_idx * self.steps_per_epoch + self.batch_idx

    def setup_parallel_model(self):
        """
        初始化分布式模型。
        """
        if (
            dist.get_world_size()
            != self.config.tp_size * self.config.dp_size * self.config.pp_size
        ):
            logger.rank_zero_warning(
                "The world size is not equal to the product of the parallel sizes set."
                f"{dist.get_world_size()} != {self.config.tp_size} * {self.config.dp_size} * {self.config.pp_size}."
            )
            self.config.dp_size = dist.get_world_size() // (
                self.config.tp_size * self.config.pp_size
            )
            logger.rank_zero_warning(f"Set dp_size to {self.config.dp_size}.")
        self.on_setup_parallel_model()
        if self.config.pp_size > 1:
            # GPTLMLoss 是 Module，会被 nn.Module 加入 _Modules
            # 如果 loss_fn 是一个函数就会在此时报错
            if not isinstance(self.loss_fn, torch.nn.Module):
                del self.model.loss_fn
            self.model.loss_fn = self.loss_fn
        if "Lomo" in self.optimizer.__class__.__name__:
            self.engine, _, _, _ = setup_ds_engine(
                model=self.model,
                config=self.config,
            )
        else:
            self.engine, self.optimizer, _, self.lr_scheduler = setup_ds_engine(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                config=self.config,
            )
        # train_dataloader
        if self.train_dataset is None:
            self.train_dataloader = None
            self.steps_per_epoch = 0
        else:
            self.train_dataloader = CollieDataLoader(
                self.train_dataset,
                self.config.train_micro_batch_size,
                self.config.gradient_accumulation_steps,
                shuffle=True,
                collate_fn=self.train_dataset_collate_fn,
                drop_last=False,
                num_workers=self.config.dataloader_num_workers,
            )
            self.steps_per_epoch = len(self.train_dataloader)

        # set logger level
        deepspeed_logging_level = (
            logging.ERROR
            if "logging_level" not in self.config.ds_config
            else self.config.ds_config["logging_level"]
        )
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
            desc="Training Epoch",
            disable=env.rank != 0,
            completed=self.epoch_idx,
            total=self.config.train_epochs,
        )
        tqbar_batch = progress(
            train_dataloader,
            desc="Training Batch: ",
            disable=env.rank != 0,
            total=self.steps_per_epoch,
        )
        for self.epoch_idx in tqbar_epoch:
            if not train_dataloader.curriculum_learning_enabled:
                tqbar_batch.sequence.sampler.set_epoch(self.epoch_idx)

            self.on_train_epoch_begin()
            tqbar_epoch.set_description(
                f"Training Epoch: {self.epoch_idx} / {self.config.train_epochs}"
            )
            tqbar_batch.reset(
                f"Training Batch: {self.batch_idx} / {self.steps_per_epoch}",
                completed=self.batch_idx,
            )
            for self.batch_idx, batch in enumerate(tqbar_batch, start=self.batch_idx):
                tqbar_batch.set_description(
                    f"Training Batch: {self.batch_idx} / {self.steps_per_epoch}"
                )
                # skip trained batch
                if self.resume_from_checkpoint:
                    if self.batch_idx <= self.trained_batch_idx:
                        continue
                    else:
                        self.resume_from_checkpoint = False

                if self.server is not None:
                    self.server.data_provider_handler()
                self.engine.train()
                self.on_train_batch_begin(batch)
                if isinstance(self.engine.module, PipelineModel):
                    self.engine.module.forward_type = "train"
                if isinstance(self.engine.module, PeftModel) and isinstance(
                    self.engine.module.get_base_model(), PipelineModel
                ):
                    self.engine.module.get_base_model().forward_type = "train"
                with self.monitor as item:
                    loss = self.train_fn(self, batch, self.global_batch_idx)
                    item.update(
                        {
                            "loss": round(loss, 4),
                            "lr": self.lr,
                            "batch": batch,
                            "batch_idx": self.batch_idx,
                            "epoch_idx": self.epoch_idx,
                            "global_batch_idx": self.global_batch_idx,
                            "memory_allocated": torch.cuda.max_memory_allocated(),
                            "mode": "train",
                        }
                    )
                tqbar_batch.set_postfix(Loss=round(loss, 4))
                self.on_train_batch_end(loss)
                if (
                    self.config.eval_per_n_steps > 0
                    and (self.batch_idx + 1) % (self.config.eval_per_n_steps * self.config.gradient_accumulation_steps) == 0
                ):
                    self.eval()
            if self.resume_from_checkpoint is False:
                self.on_train_epoch_end()
                if (
                    self.config.eval_per_n_epochs > 0
                    and (self.epoch_idx + 1) % self.config.eval_per_n_epochs == 0
                ):
                    self.eval()
            self.resume_from_checkpoint = False
            self.batch_idx = 0
        self.on_train_end()

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
            if isinstance(trainer.engine.module, PeftModel) and isinstance(
                trainer.engine.module.get_base_model(), PipelineModel
            ):
                trainer.engine.module.get_base_model().forward_type = "train"
            loss = trainer.engine.module(**batch)["loss"]
        else:
            outputs = trainer.engine(**batch)
            # concat prompt labels for p-tuning
            if trainer.config.peft_config and trainer.config.peft_config.peft_type in [
                "PROMPT_TUNING",
                "P_TUNING",
            ]:
                batch_size = batch["input_ids"].shape[0]
                prefix_labels = torch.full(
                    (batch_size, trainer.config.peft_config.num_virtual_tokens), -100
                ).to(batch["labels"].device)
                batch["labels"] = torch.cat((prefix_labels, batch["labels"]), dim=1)
            loss = auto_param_call(
                trainer.loss_fn,
                {**batch, **outputs},
                signature_fn=trainer.loss_fn.forward
                if isinstance(trainer.loss_fn, nn.Module)
                else trainer.loss_fn,
            )
            if not ("Lomo" in trainer.optimizer.__class__.__name__):
                trainer.engine.backward(loss)
                trainer.engine.step()
            else:
                # for lomo or adalomo
                if trainer.optimizer.clip_grad_norm is not None or (
                    hasattr(trainer.optimizer, "loss_scaler") and trainer.optimizer.loss_scaler is not None
                ):
                    trainer.optimizer.grad_norm(loss)
                    if (
                        trainer.optimizer.loss_scaler
                        and trainer.optimizer.loss_scaler.has_overflow_serial
                    ):
                        print(f"Gradient overflow, skipping step {global_step}")
                        if trainer.optimizer.zero3_enabled:
                            trainer.engine.optimizer.get_param_coordinator(
                                training=True
                            ).reset_step()
                        return loss.detach().cpu().item()
                    if trainer.optimizer.zero3_enabled:
                        trainer.engine.optimizer.get_param_coordinator(
                            training=True
                        ).reset_step()
                        # zero-3 doesn't support backward twice, so need an additional forward here
                        outputs = trainer.engine(**batch)
                        loss = auto_param_call(
                            trainer.loss_fn,
                            {**batch, **outputs},
                            signature_fn=trainer.loss_fn.forward
                            if isinstance(trainer.loss_fn, nn.Module)
                            else trainer.loss_fn,
                        )
                if trainer.lr_scheduler is not None:
                    lr = trainer.lr_scheduler.get_last_lr()[0]
                else:
                    lr = trainer.optimizer.lr
                trainer.optimizer.fused_backward(loss, lr)
                if trainer.lr_scheduler is not None:
                    trainer.lr_scheduler.step()
                if trainer.optimizer.zero3_enabled:
                    trainer.engine.optimizer.get_param_coordinator(
                        training=True
                    ).reset_step()
        dist.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=env.dp_group, async_op=False)
        return loss.detach().cpu().item()

    @property
    def lr(self):
        if self.lr_scheduler:
            lr = self.lr_scheduler.get_last_lr()[0]
        else:
            lr = self.optimizer.param_groups[0]["lr"]
        return lr

    def save_peft(
        self,
        path: str,
        selected_adapters: Optional[List[str]] = None,
        process_exclusion: bool = False,
        **kwargs,
    ):
        ...

    def save_peft(
        self,
        path: str,
        selected_adapters: Optional[List[str]] = None,
        process_exclusion: bool = False,
        protocol: str = "file",
        **kwargs,
    ):
        """
        保存 adapter 部分权重，当未使用 ``peft`` 时，该方法等同于 ``save_model``

        :param path: 模型保存路径
        :param selected_adapters: 保存时保存哪些 adapter；为 ``None`` 时则会保存
            所有的 adapter
        :param process_exclusion:
        """
        if not isinstance(self.model, PeftModel):
            return self.save_model(
                path=path,
                protocol=protocol,
                process_exclusion=process_exclusion,
                **kwargs,
            )
        if selected_adapters is None:
            selected_adapters = list(self.model.peft_config.keys())
        else:
            if any(
                selected_adapter_name not in list(self.model.peft_config.keys())
                for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.model.peft_config.keys())} - got {selected_adapters}."
                )
        io_driver = IODriver.from_protocol(protocol)
        for adapter_name in selected_adapters:
            peft_config = self.model.peft_config[adapter_name]
            contexts = []
            state_dict = get_peft_model_state_dict(
                self.model, adapter_name=adapter_name
            )
            output_dir = (
                os.path.join(path, adapter_name) if adapter_name != "default" else path
            )
            io_driver.makedirs(output_dir, exist_ok=True)
            if isinstance(peft_config, PromptLearningConfig):
                parameters = [self.model.prompt_encoder[adapter_name].embedding.weight]
            else:
                parameters = [
                    param
                    for name, param in self.engine.module.named_parameters()
                    if any(
                        [
                            name.replace(f"{adapter_name}.", "") == k
                            for k in state_dict.keys()
                        ]
                    )
                ]
            pp_save = env.pp_size > 1 and peft_config.peft_type in (
                PeftType.LORA,
                PeftType.ADALORA,
            )
            name_prefix = "adapter_model"
            if not pp_save:
                name = f"{name_prefix}.bin"
            else:
                name = f"{name_prefix}_{env.pp_rank}.bin"
            if is_zero3_enabled(self.config):
                contexts.append(deepspeed.zero.GatheredParameters(parameters))
                # 加上以避免报错 assert not param.ds_active_sub_modules
                # TODO
                self._checkpoint_prologue()
            with ContextManagers(contexts):
                if (
                    env.dp_rank == 0
                    or not is_zero3_enabled(self.config)
                    and (pp_save or env.pp_rank == 0)
                ):
                    state_dict = get_peft_model_state_dict(
                        self.model, adapter_name=adapter_name
                    )
                    if env.dp_rank == 0 and env.tp_rank == 0:
                        io_driver.save(state_dict, os.path.join(path, name))
            if is_zero3_enabled(self.config):
                self._checkpoint_epilogue()
            env.barrier()
            if env.rank == 0:
                peft_config.save_pretrained(output_dir)
                if pp_save:
                    pp_merge_peft(path, name_prefix, io_driver)

    def load_peft(
        self,
        path: str,
        adapter_name="default",
        is_trainable: bool = False,
        process_exclusion: bool = False,
        **kwargs,
    ):
        ...

    def load_peft(
        self,
        path: str,
        adapter_name="default",
        is_trainable: bool = False,
        process_exclusion: bool = False,
        protocol: str = "file",
        **kwargs,
    ):
        """
        加载 adapter 部分权重，当未使用 ``peft`` 时，该方法等同于 ``load_model``

        :param path: 模型保存路径
        :param adapter_name: 当前加载的 adapter 名称
        :param is_trainable: 是否允许加载的 adapter 进行训练
        :param process_exclusion:
        """
        io_driver = IODriver.from_protocol(protocol)
        if not isinstance(self.model, PeftModel):
            return self.load_model(
                path=path,
                protocol=protocol,
                process_exclusion=process_exclusion,
                **kwargs,
            )

        load_peft(
            self.engine.module,
            self.config,
            path,
            adapter_name,
            is_trainable,
            protocol,
        )

    def save_model(self, path: str, process_exclusion: bool = False, **kwargs):
        ...

    def save_model(
        self,
        path: str,
        process_exclusion: bool = False,
        protocol: str = "file",
        **kwargs,
    ):
        """
        保存模型。

        :param path: 模型保存路径
        :param process_exclusion: 是否开启进程互斥，当开启流水线并行时开启此项可以节省内存（仅限 **CoLLie** 内实现的模型，对 `transformers` 提供的模型本项无效）
        """
        dist.barrier()
        io_driver = IODriver.from_protocol(protocol)
        io_driver.makedirs(path, exist_ok=True)
        self.on_save_model()

        # 保存 config 和 tokenizer
        if env.rank == 0:
            try:
                model_id = self.config.model_config.name_or_path
                AutoConfig.from_pretrained(model_id, trust_remote_code=True).save_pretrained(path)
                AutoTokenizer.from_pretrained(model_id, trust_remote_code=True).save_pretrained(path)
            except Exception as e:
                logger.rank_zero_warning("Save config and tokenizer failed")
                logger.rank_zero_warning(str(e))
        
        if isinstance(self.model, PeftModel):
            self.save_peft(
                path=path,
                protocol=protocol,
                process_exclusion=process_exclusion,
                **kwargs,
            )
            model_to_save = self.engine.module.get_base_model()
        else:
            model_to_save = self.engine.module
        
        if isinstance(model_to_save, CollieModelForCausalLM) or isinstance(
            model_to_save, PipelineModel
        ):
            if is_zero3_enabled(self.config):
                state_dict = {}
                self._checkpoint_prologue()
                for name, param in model_to_save.named_parameters():
                    with deepspeed.zero.GatheredParameters(param):
                        if env.dp_rank == 0:
                            state_dict[name] = param.detach().cpu()
                self._checkpoint_epilogue()
            else:
                if env.dp_rank == 0:
                    state_dict = model_to_save.state_dict()
                else:
                    state_dict = {}
            model_to_save.save_parallel_state_dict(
                state_dict=state_dict,
                path=path,
                config=self.config,
                process_exclusion=process_exclusion,
                protocol=protocol,
            )
        elif isinstance(model_to_save, PreTrainedModel):
            if is_zero3_enabled(self.config):
                self._checkpoint_prologue()
                with deepspeed.zero.GatheredParameters(
                    list(model_to_save.parameters(recurse=True))
                ):
                    model_to_save.save_pretrained(save_directory=path, **kwargs)
                self._checkpoint_epilogue()
            else:
                model_to_save.save_pretrained(save_directory=path, **kwargs)

    def load_model(self, path: str, process_exclusion: bool = False, **kwargs):
        ...

    def load_model(
        self,
        path: str,
        process_exclusion: bool = False,
        protocol: str = "file",
        **kwargs,
    ):
        io_driver = IODriver.from_protocol(protocol)
        assert io_driver.exists(path), f"`{path}` does not exist."
        self.on_load_model()
        if isinstance(self.engine.module, CollieModelForCausalLM) or isinstance(
            self.engine.module, PipelineModel
        ):
            if is_zero3_enabled(self.config):
                self._checkpoint_prologue()
                with deepspeed.zero.GatheredParameters(
                    list(self.engine.module.parameters(recurse=True)), modifier_rank=0
                ):
                    if env.rank == 0:
                        self.engine.module.load_state_dict(
                            self.engine.module.load_parallel_state_dict(
                                path=path,
                                config=self.config,
                                process_exclusion=process_exclusion,
                                protocol=protocol,
                            )
                        )
                self._checkpoint_epilogue()
            else:
                self.engine.module.load_state_dict(
                    self.engine.module.load_parallel_state_dict(
                        path=path,
                        config=self.config,
                        process_exclusion=process_exclusion,
                        protocol=protocol,
                    )
                )
        elif isinstance(self.engine.module, PreTrainedModel):
            index = None
            if io_driver.exists(os.path.join(path, "pytorch_model.bin.index.json")):
                weight_map = json.loads(
                    io_driver.load(
                        os.path.join(path, "pytorch_model.bin.index.json"), mode="r"
                    )
                )["weight_map"]
                index = OrderedDict()
                for key, value in weight_map.items():
                    if value not in index.keys():
                        index[value] = [key]
                    else:
                        index[value].append(key)
            if is_zero3_enabled(self.config):
                self._checkpoint_prologue()
                if index is not None:
                    for key, value in index.items():
                        # 用 state dict 会没办法 gather
                        param_list = [
                            p
                            for n, p in self.engine.module.named_parameters()
                            if n in value
                        ]
                        with deepspeed.zero.GatheredParameters(
                            param_list, modifier_rank=0
                        ):
                            if env.dp_rank == 0:
                                state_dict = io_driver.load(
                                    os.path.join(path, key), mode="br"
                                )
                                for attr in value:
                                    self.engine.module.state_dict()[attr].copy_(
                                        state_dict[attr]
                                    )
                else:
                    with deepspeed.zero.GatheredParameters(
                        list(self.engine.module.parameters(recurse=True)),
                        modifier_rank=0,
                    ):
                        if env.dp_rank == 0:
                            state_dict = reduce(
                                lambda x, y: {**x, **y},
                                [
                                    io_driver.load(file, mode="rb")
                                    for file in glob.glob(os.path.join(path, "*.bin"))
                                ],
                            )
                            self.engine.module.load_state_dict(state_dict)
                self._checkpoint_epilogue()
            else:
                if index is not None:
                    for key, value in index.items():
                        state_dict = io_driver.load(os.path.join(path, key), mode="br")
                        for attr in value:
                            self.engine.module.state_dict()[attr].copy_(
                                state_dict[attr]
                            )
                else:
                    state_dict = reduce(
                        lambda x, y: {**x, **y},
                        [
                            io_driver.load(file, mode="rb")
                            for file in glob.glob(os.path.join(path, "*.bin"))
                        ],
                    )
                    self.engine.module.load_state_dict(state_dict)

    def save_checkpoint(self, path: str, process_exclusion: bool = False, **kwargs):
        ...

    def save_checkpoint(
        self,
        path: str,
        process_exclusion: bool = False,
        protocol: str = "file",
        **kwargs,
    ):
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
            trainer_state_dict = {
                "dp_size": env.dp_size,
                "tp_size": env.tp_size,
                "pp_size": env.pp_size,
            }
            trainer_state_dict.update(self.state_dict())
            io_driver.save(json.dumps(trainer_state_dict), os.path.join(path, "trainer_state_dict.json"))

        engine = self.engine
        # DeepSpeedEngine.save_checkpoint

        self.save_model(path, protocol=protocol)

        # Prepare for checkpoint save by ensuring all parameters are partitioned
        self._checkpoint_prologue()

        ## DeepSpeedEngine._save_checkpoint
        zero_optimizer_state = engine.zero_optimization() or engine.bfloat16_enabled()
        state = dict(
            optimizer=engine.optimizer.state_dict()
            if engine.optimizer and not zero_optimizer_state
            else None,
            lr_scheduler=engine.lr_scheduler.state_dict()
            if engine.lr_scheduler is not None
            else None,
            data_sampler=engine.training_dataloader.data_sampler.state_dict()
            if (
                engine.training_dataloader is not None
                and engine.curriculum_learning_enabled()
            )
            else None,
            sparse_tensor_module_names=engine.sparse_tensor_module_names,
            skipped_steps=engine.skipped_steps,
            global_steps=engine.global_steps,
            global_samples=engine.global_samples,
            callback_states=callback_states,
        )

        if env.dp_rank == 0 or engine.zero_optimization_partition_weights():
            io_driver.save(state, os.path.join(path, self.checkpoint_file))

        if engine.save_zero_checkpoint:
            self._save_zero_checkpoint(path, io_driver)

        self._checkpoint_epilogue()

        dist.barrier()

    def load_checkpoint(self, path: str, process_exclusion: bool = False, **kwargs):
        ...

    def load_checkpoint(
        self,
        path: str,
        process_exclusion: bool = False,
        protocol: str = "file",
        **kwargs,
    ):
        """训练器断点加载

        :param path: 断点保存路径
        :param process_exclusion: 是否开启进程互斥，当开启流水线并行时开启此项可以节省内存（仅限 **CoLLie** 内实现的模型，对 ``transformers`` 提供的模型本项无效）
        """
        io_driver = IODriver.from_protocol(protocol)
        assert io_driver.exists(path), f"`{path}` does not exist."
        engine = self.engine
        # check并行方法一致
        loaded_args = json.loads(io_driver.load(os.path.join(path, "trainer_state_dict.json"), "r"))
        assert (
            loaded_args["dp_size"] == env.dp_size
            and loaded_args["tp_size"] == env.tp_size
            and loaded_args["pp_size"] == env.pp_size
        ), (
            "Loaded checkpoint's world_size is not equal to the current "
            f"settings: dp * tp * pp {loaded_args['dp_size']} * "
            f"{loaded_args['tp_size']} * {loaded_args['pp_size']}"
            f"!= {env.dp_size} * {env.tp_size} * {env.pp_size}."
        )

        self.load_model(path, protocol=protocol)

        # DeepSpeed.load_checkpoint
        if engine.zero_optimization_partition_weights():
            ckpt_file = self.checkpoint_file
        else:
            ckpt_file = f"collie_dp0_pp{env.pp_rank}_tp{env.tp_rank}.pt"
        checkpoint = io_driver.load(os.path.join(path, ckpt_file), "b")

        # Prepare for checkpoint load by ensuring all parameters are partitioned
        self._checkpoint_prologue()

        has_zero_optimizer_state = (
            engine.zero_optimization() or engine.bfloat16_enabled()
        )

        if engine.optimizer is not None and not has_zero_optimizer_state:
            engine.optimizer.load_state_dict(checkpoint["optimizer"])

        if engine.lr_scheduler is not None:
            engine.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        if (
            engine.training_dataloader is not None
            and engine.curriculum_learning_enabled()
            and "data_sampler" in checkpoint
        ):
            engine.training_dataloader.data_sampler.load_state_dict(
                checkpoint["data_sampler"]
            )

        if "sparse_tensor_module_names" in checkpoint:
            sparse_tensor_module_names = checkpoint["sparse_tensor_module_names"]
        elif "csr_tensor_module_names" in checkpoint:
            sparse_tensor_module_names = checkpoint["csr_tensor_module_names"]
        else:
            sparse_tensor_module_names = None
        if sparse_tensor_module_names is not None:
            engine.sparse_tensor_module_names = sparse_tensor_module_names

        engine.global_steps = checkpoint["global_steps"]
        engine.global_samples = checkpoint.get(
            "global_samples", engine.global_steps * engine.train_batch_size()
        )
        engine.skipped_steps = checkpoint["skipped_steps"]
        # self.load_state_dict(checkpoint['trainer_state_dict'])
        self.load_state_dict(loaded_args)

        # load_zero_checkpoint = engine.zero_optimization() or engine.bfloat16_enabled()
        load_zero_checkpoint = engine.save_zero_checkpoint
        if load_zero_checkpoint:
            success = self._load_zero_checkpoint(path, io_driver)
            if not success:
                engine.optimizer._restore_from_bit16_weights()

        self._checkpoint_epilogue()
        self.on_load_checkpoint(checkpoint["callback_states"])

    def _save_zero_checkpoint(self, path, driver):
        """保存 `ZeRO` 的状态"""
        zero_path = os.path.join(path, self.zero_checkpoint_file)
        zero_sd = self.engine.optimizer.state_dict()
        driver.save(zero_sd, zero_path)

    def _load_zero_checkpoint(self, path, driver):
        """加载 `ZeRO` 的状态"""
        engine = self.engine

        zero_sd_list = []
        for dp_rank in range(engine.dp_world_size):
            zero_ckpt = os.path.join(path, self.zero_checkpoint_file)
            zero_ckpt = zero_ckpt.replace(f"dp{env.dp_rank}", f"dp{dp_rank}")
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
        if isinstance(self.engine.optimizer, DeepSpeedZeRoOffload):
            param_offload = self.engine.optimizer
        else:
            param_offload = self.engine.optimizer.parameter_offload
        param_offload.partition_all_parameters()

    def _checkpoint_epilogue(self):
        if not self.engine.zero_optimization_partition_weights():
            return
        if isinstance(self.engine.optimizer, DeepSpeedZeRoOffload):
            persistent_params = self.engine.optimizer.persistent_parameters
        else:
            persistent_params = (
                self.engine.optimizer.parameter_offload.persistent_parameters
            )
        persistent_params[0].all_gather(persistent_params)

        if len(persistent_params) > 0:
            persistent_params[0].all_gather(persistent_params)
