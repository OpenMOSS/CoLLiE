from collie.trainer.arguments import Arguments, load_config
from collie.module import GPTLMLoss
from collie.log.print import print
from collie.log import logger

import tqdm
import torch
import deepspeed
from dataclasses import asdict
import torch.distributed as dist
from megatron.core import parallel_state

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
        self.setup_parallel_model()
        
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
                            logits = self.engine(input_ids.cuda())
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