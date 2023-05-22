from typing import Optional, Iterable, Dict

import torch.distributed as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.pipe.engine import PipelineEngine

from collie.utils import progress, env
from collie.trainer import Trainer as OriginTrainer

class Trainer(OriginTrainer):
    def eval(self, 
             dataloader: Optional[Iterable] = None, 
             train_meta: Dict = {"epoch_idx": 0, "batch_idx": 0, "last_loss": 0.0}):
        eval_dataloader = self.eval_dataloader
        if dataloader is not None:
            eval_dataloader = dataloader
        num_eval_batches = len(self.eval_dataloader)
        with progress(eval_dataloader, desc="Evaluating Batch: ", disable=dist.get_rank() != 0, total=num_eval_batches) as tqbar_batch:
            for batch_idx, batch in enumerate(tqbar_batch):
                if isinstance(self.engine, PipelineEngine):
                    self.engine.reset_activation_shape()
                    if self.engine.total_loss is not None:
                        total_loss = self.engine.total_loss.detach().clone()
                    else:
                        total_loss = None
                    self.engine.total_loss = None
                self.generation_server_handler()
                self.engine.eval()
                result = self.eval_fn(self, batch, train_meta)
                get_accelerator().empty_cache()
                if isinstance(self.engine, PipelineEngine):
                    self.engine.total_loss = total_loss
                if (self.config.pp_size == 1 or env.pp_rank == self.config.pp_size - 1) \
                    and (self.config.tp_size == 1 or env.tp_rank == self.config.tp_size - 1):
                    for metric in self.metrics:
                        if metric.gather_result:
                            result = metric.gather(result)
                        metric.update(result)
                tqbar_batch.set_postfix(
                    batch=f"{batch_idx + 1}/{num_eval_batches}")
            # get_metric 的功能现在还在完善，暂时继承一个 Trainer 来收集结果
            if (self.config.pp_size == 1 or env.pp_rank == self.config.pp_size - 1) \
                and (self.config.tp_size == 1 or env.tp_rank == self.config.tp_size - 1):
                for metric in self.metrics:
                    metric.get_metric(train_meta)
        if isinstance(self.engine, PipelineEngine):
            self.engine.reset_activation_shape()
            self.communicate_buffer_shape = None