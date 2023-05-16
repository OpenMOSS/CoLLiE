import os
import json
from typing import Optional, Callable, Union, Tuple, Iterable, Any, Dict, Sequence

from collie.trainer.arguments import Arguments, load_config
from collie.module import CollieCausalLM, GPTLMLoss, PipelineModel
from collie.driver.io.file import FileIODriver
from collie.driver.io.petrel import PetrelIODriver
from collie.log.print import print
from collie.log import logger
from collie.utils import progress, env

import os
import torch
import deepspeed
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from megatron.core import parallel_state
from deepspeed.runtime.constants import ROUTE_EVAL
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.runtime.zero.utils import ZeRORuntimeException
from transformers.generation.utils import GenerationConfig

class Trainer:
    def __init__(self, 
                 model: torch.nn.Module,
                 args: Union[Arguments, str],
                 loss_fn: Callable = GPTLMLoss(),
                 train_fn: Optional[Callable] = None,
                 eval_fn: Optional[Callable] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 train_dataset: Optional[torch.utils.data.Dataset] = None,
                 eval_dataset: Optional[torch.utils.data.Dataset] = None,
                 train_dataset_collate_fn: Optional[Callable] = None,
                 eval_dataset_collate_fn: Optional[Callable] = None,
                 eval_config: GenerationConfig = GenerationConfig(),
                 metrics: Sequence = []) -> None:
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
        get_accelerator().empty_cache()

        self.checkpoint_file = "collie_dp{}_pp{}_tp{}.pt".format(
            env.dp_rank, env.pp_rank, env.tp_rank
        )
        self.zero_checkpoint_file = "collie_zero_dp{}_pp{}_tp{}.pt".format(
            env.dp_rank, env.pp_rank, env.tp_rank
        )
        
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
        loss = 0.0
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
                        if self.args.eval_per_n_steps > 0 and (batch_idx + 1) % self.args.eval_per_n_steps == 0:
                            self.eval(train_meta={"epoch_idx": epoch_idx, "batch_idx": batch_idx, "last_loss": loss})
                tqbar_epoch.set_postfix(epoch=f"{epoch_idx + 1}/{self.args.train_epochs}")
                if self.args.eval_per_n_epochs > 0 and (epoch_idx + 1) % self.args.eval_per_n_epochs == 0:
                            self.eval(train_meta={"epoch_idx": epoch_idx, "batch_idx": 0, "last_loss": loss})
                
    def eval(self, 
             dataloader: Optional[Iterable] = None, 
             train_meta: Dict = {"epoch_idx": 0, "batch_idx": 0, "last_loss": 0.0}):
        self.engine.eval()
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
                result = self.eval_fn(self, batch, train_meta)
                if isinstance(self.engine, PipelineEngine):
                    self.engine.total_loss = total_loss
                for metric in self.metrics:
                    if metric.gather_result:
                        result = metric.gather(result)
                    if not metric.only_rank0_update or dist.get_rank() == 0:
                        metric.update(result)
                tqbar_batch.set_postfix(
                    batch=f"{batch_idx + 1}/{num_eval_batches}")
        if isinstance(self.engine, PipelineEngine):
            self.engine.reset_activation_shape()
                
    @staticmethod
    def train_fn(trainer, batch: Tuple) -> float:
        if trainer.args.pp_size > 1:
            loss = trainer.engine.train_batch(data_iter=iter([batch]))
        else:
            input_ids, labels = batch
            logits = trainer.engine(input_ids.cuda())
            loss = trainer.loss_fn(logits, labels)
            trainer.engine.backward(loss)
            trainer.engine.step()
        return loss.item()
        
    @staticmethod
    def eval_fn(trainer, 
                batch: Tuple, 
                train_meta: Dict = {"epoch_idx": 0, "batch_idx": 0, "last_loss": 0.0}) -> Any:
        input_ids, labels = batch
        generation_model = CollieCausalLM(
            engine=trainer.engine,
            config=trainer.eval_config
        )
        input_ids = generation_model.generate(input_ids=input_ids.cuda(), attention_mask=torch.ones_like(input_ids).cuda())
        return {
            "input_ids": input_ids,
            "labels": labels,
            "train_meta": train_meta
        }

    def save_checkpoint(self, path: str, process_exclusion: bool = False):...
    def save_checkpoint(self, path: str, process_exclusion: bool = False,
                        protocol: str="file"):
        assert protocol in ["file", "petrel"], f"Only support file and petrel protocol, not `{protocol}`."
        IODriver = FileIODriver if protocol == 'file' else PetrelIODriver
        IODriver.makedirs(path, exist_ok=True)
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
        state = dict(optimizer=engine.optimizer.state_dict() if engine.optimizer and not zero_optimizer_state else None,
                     lr_scheduler=engine.lr_scheduler.state_dict() if engine.lr_scheduler is not None else None,
                     data_sampler=engine.training_dataloader.data_sampler.state_dict() if
                     (engine.training_dataloader is not None and engine.curriculum_learning_enabled()) else None,
                     sparse_tensor_module_names=engine.sparse_tensor_module_names,
                     skipped_steps=engine.skipped_steps,
                     global_steps=engine.global_steps,
                     global_samples=engine.global_samples)

        IODriver.save(state, os.path.join(path, self.checkpoint_file))

        if engine.save_zero_checkpoint:
            self._save_zero_checkpoint(path, IODriver)

        if engine.zero_optimization_partition_weights():
            engine.optimizer.checkpoint_event_epilogue()

        # state dict
        state_dict = self.model.state_dict()
        self.model.save_parallel_state_dict(
            state_dict, path, self.args, process_exclusion,
            protocol=protocol
        )

        dist.barrier()

    def load_checkpoint(self, path: str, process_exclusion: bool = False):...
    def load_checkpoint(self, path: str, process_exclusion: bool = False,
                        protocol: str = 'file'):
        assert protocol in ["file", "petrel"], f"Only support file and petrel protocol, not `{protocol}`."
        IODriver = FileIODriver if protocol == 'file' else PetrelIODriver
        assert IODriver.exists(path), f"`{path}` does not exist."
        engine = self.engine
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
            # Prepare for checkpoint load by ensuring all parameters are partitioned
            engine.optimizer.checkpoint_event_prologue()

        ## DeepSpeed._load_checkpoint
        checkpoint = IODriver.load(os.path.join(path, self.checkpoint_file), "b")

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

        # state_dict
        state_dict = self.model.load_parallel_state_dict(
            path=path, args=self.args, process_exclusion=process_exclusion,
        )
        self.model.load_state_dict(state_dict)

    def _save_zero_checkpoint(self, path, driver):
        zero_path = os.path.join(path, self.zero_checkpoint_file)
        zero_sd = self.engine.optimizer.state_dict()
        driver.save(zero_sd, zero_path)

    def _load_zero_checkpoint(self, path, driver):
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
