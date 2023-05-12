from collie.trainer.arguments import Arguments, load_config
from collie.module import CollieCasualLM, GPTLMLoss
from collie.log.print import print
from collie.log import logger
from collie.utils import progress

import torch
import deepspeed
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from megatron.core import parallel_state
from deepspeed.runtime.constants import ROUTE_EVAL
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.pipe.engine import PipelineEngine
from transformers.generation.utils import GenerationConfig

from typing import Optional, Callable, Union, Tuple, Iterable, Any, Dict, Sequence

class Trainer:
    def __init__(self, 
                 model: torch.nn.Module,
                 loss_fn: Callable = GPTLMLoss(),
                 train_fn: Optional[Callable] = None,
                 eval_fn: Optional[Callable] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 train_dataset: Optional[torch.utils.data.Dataset] = None,
                 eval_dataset: Optional[torch.utils.data.Dataset] = None,
                 train_dataset_collate_fn: Optional[Callable] = None,
                 eval_dataset_collate_fn: Optional[Callable] = None,
                 eval_config: GenerationConfig = GenerationConfig(),
                 metrics: Sequence = [],
                 args: Union[Arguments, str] = Arguments()) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.loss_fn = loss_fn
        if train_fn is not None:
            self.train_fn = train_fn
        if eval_fn is not None:
            self.eval_fn = eval_fn
        self.train_dataset_collate_fn = train_dataset_collate_fn
        self.eval_dataset_collate_fn = eval_dataset_collate_fn
        self.eval_config = eval_config
        self.metrics = metrics
        self.args = args
        self.communicate_buffer_shape = None
        self.set_ds_config()
        self.setup_parallel_model()
        self.init_metrics()
        
    def set_ds_config(self):
        if isinstance(self.args, str):
            self.args = load_config(self.args)
        if isinstance(self.args.ds_config, str):
            self.args.ds_config = load_config(self.args.ds_config)
        if "train_micro_batch_size_per_gpu" not in self.args.ds_config.keys():
            self.args.ds_config["train_micro_batch_size_per_gpu"] = self.args.train_micro_batch_size
        if "gradient_accumulation_steps" not in self.args.ds_config.keys():
            self.args.ds_config["gradient_accumulation_steps"] = self.args.gradient_accumulation_steps
        print(self.args)
        
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
        self.engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=[p for p in self.model.parameters() if p.requires_grad],
            optimizer=self.optimizer,
            mpu=parallel_state if self.args.pp_size == 1 else None,
            config=self.args.ds_config
        )
        self.args.train_micro_batch_size = self.engine.train_micro_batch_size_per_gpu()
        self.args.gradient_accumulation_steps = self.engine.gradient_accumulation_steps()

        # train_dataloader
        if self.train_dataset is None:
            self.train_dataloader = None
        if self.args.pp_size == 1:
            self.train_dataloader = self.engine.deepspeed_io(
                self.train_dataset, collate_fn=self.train_dataset_collate_fn
            )
        else:
            # PipelineModule._build_data_iter
            sampler = DistributedSampler(
                self.train_dataset, num_replicas=self.engine.dp_world_size,
                rank=self.engine.mpu.get_data_parallel_rank(), shuffle=False
            )
            self.train_dataloader = self.engine.deepspeed_io(
                self.train_dataset, data_sampler=sampler,
                collate_fn=self.train_dataset_collate_fn
            )
        if self.eval_dataset is not None:
            self.eval_dataloader = self.engine.deepspeed_io(
                self.eval_dataset,
                batch_size=self.args.eval_batch_size,
                route=ROUTE_EVAL,
                pin_memory=True,
                data_sampler=None,
                collate_fn=self.eval_dataset_collate_fn,
                num_local_io_workers=None
            )
        else:
            self.eval_dataloader = None
        
    def init_metrics(self):
        for metric in self.metrics:
            metric.construct(self)
        
    def train(self, dataloader: Optional[Iterable] = None):
        self.engine.train()
        train_dataloader = self.train_dataloader
        if dataloader is not None:
            train_dataloader = dataloader
        with progress(range(self.args.train_epochs), desc="Training Epoch: ", disable=dist.get_rank() != 0) as tqbar_epoch:
            for epoch_idx in tqbar_epoch:
                with progress(train_dataloader, desc="Training Batch: ", disable=dist.get_rank() != 0) as tqbar_batch:
                    for batch_idx, batch in enumerate(tqbar_batch):
                        if isinstance(self.engine, PipelineEngine):
                            if self.communicate_buffer_shape is None:
                                self.communicate_buffer_shape = batch[0].shape
                            else:
                                if self.communicate_buffer_shape != batch[0].shape:
                                    self.engine.reset_activation_shape()
                                    self.communicate_buffer_shape = batch[0].shape
                        loss = self.train_fn(self, batch)
                        tqbar_batch.set_postfix(
                            loss=round(loss, 2), 
                            batch=f"{batch_idx + 1}/{len(self.train_dataloader)}")
                tqbar_epoch.set_postfix(epoch=f"{epoch_idx + 1}/{self.args.train_epochs}")
                
    def eval(self, 
             dataloader: Optional[Iterable] = None, 
             train_meta: Dict = {"epoch_idx": 0, "batch_idx": 0, "last_loss": 0.0}):
        self.engine.eval()
        # reset buffer size after training.
        eval_dataloader = self.eval_dataloader
        if dataloader is not None:
            eval_dataloader = dataloader
        num_eval_batches = len(self.eval_dataloader)
        with progress(eval_dataloader, desc="Evaluating Batch: ", disable=dist.get_rank() != 0, total=num_eval_batches) as tqbar_batch:
            for batch_idx, batch in enumerate(tqbar_batch):
                if isinstance(self.engine, PipelineEngine):
                    self.engine.reset_activation_shape()
                    self.engine.total_loss = None
                result = self.eval_fn(self, batch, train_meta)
                for metric in self.metrics:
                    if metric.gather_result:
                        result = metric.gather(result)
                    if not metric.only_rank0_update or dist.get_rank() == 0:
                        metric.update(result)
                tqbar_batch.set_postfix(
                    batch=f"{batch_idx + 1}/{num_eval_batches}")
                
    @staticmethod
    def train_fn(trainer, batch: Tuple) -> float:
        if trainer.args.pp_size > 1:
            loss = trainer.engine.train_batch(data_iter=iter([batch]))
        else:
            input_ids, labels = batch
            logits = trainer.engine(input_ids)
            loss = trainer.loss_fn(logits, labels)
            trainer.engine.backward(loss)
            trainer.engine.step()
        return loss.item()
        
    @staticmethod
    def eval_fn(trainer, 
                batch: Tuple, 
                train_meta: Dict = {"epoch_idx": 0, "batch_idx": 0, "last_loss": 0.0}) -> Any:
        input_ids, labels = batch
        generation_model = CollieCasualLM(
            engine=trainer.engine,
            config=trainer.eval_config
        )
        input_ids = generation_model.generate(input_ids=input_ids.cuda())
        return {
            "input_ids": input_ids,
            "labels": labels,
            "train_meta": train_meta
        }