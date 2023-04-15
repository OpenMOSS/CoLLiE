import os.path
import sys
import operator
from typing import Iterable
from itertools import chain

import tqdm
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DistributedSampler, DataLoader
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler, SequentialDistributedSampler
from transformers.trainer_utils import has_length, seed_worker

try:
    import deepspeed
    from deepspeed import comm as dist
    from deepspeed.accelerator import get_accelerator
except:
    pass

from .utils import LearningRateScheduler, WandbLogger
from collie.log import print


class InplaceZeroTrainer:
    def __init__(
            self,
            model,
            collie_args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
    ):
        self.collie_args = collie_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.wandb = WandbLogger(collie_args)
        self.allow_print = self.collie_args.local_rank in [0, -1]
        self.metrics = {}

        if 'deepspeed' not in sys.modules:
            raise ModuleNotFoundError(
                "Detected DeepSpeed is not installed. See https://github.com/microsoft/DeepSpeed")

        # Initialize deepspeed engine
        self.model, _, _, _ = deepspeed.initialize(
            config=collie_args.deepspeed,
            model=model,
        )

        # get train_dataloader and eval_dataloader
        if isinstance(data_collator, dict):
            assert 'train' in data_collator and 'eval' in data_collator, "data_collator should be a dict with keys 'train' and 'eval'."
            self.train_data_collator = data_collator['train']
            self.eval_data_collator = data_collator['eval']
        else:
            self.train_data_collator = self.eval_data_collator = data_collator
        self.train_dataloader = self.get_train_dataloader()
        if isinstance(self.eval_dataset, dict):
            self.eval_dataloader = {}
            for prefix in self.eval_dataset.keys():
                self.eval_dataloader[prefix] = self.get_eval_dataloader(self.eval_dataset[prefix])
        else:
            self.eval_dataloader = self.get_eval_dataloader()

        # setup learning rate
        self.num_steps_per_epoch = len(self.train_dataloader)
        self.global_step = 1
        self.n_steps = self.num_steps_per_epoch * self.collie_args.num_train_epochs
        self.lr_scheduler = LearningRateScheduler(learning_rate=self.collie_args.learning_rate,
                                                  warmup=self.collie_args.warmup,
                                                  schedule=self.collie_args.lr_scheduler_type,
                                                  n_steps=self.n_steps)
        self.lr = 0

        # register inplace grad hook
        self.grad_func = self.inplace_grad()
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.register_hook(self.grad_func)

        get_accelerator().empty_cache()

    def inplace_grad(self):
        # An approximation of in-place grad update under zero3 of deepspeed
        def func(x):
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        one_dim_grad = p.grad.view(-1)
                        partition_size = p.ds_tensor.numel()
                        start = partition_size * self.collie_args.local_rank
                        end = start + partition_size

                        if end > p.grad.numel():
                            partitioned_grad = one_dim_grad.narrow(0, start, p.grad.numel() - start)
                            # partitioned_grad = torch.cat([partitioned_grad, torch.zeros(end - p.grad.numel()).cuda()])
                            partitioned_p = p.ds_tensor.narrow(0, 0, p.grad.numel() - start)
                            if self.collie_args.clip_grad_value is not None:
                                # Gradients are modified in-place.
                                partitioned_grad.clamp_(min=-self.collie_args.clip_grad_value,
                                                        max=self.collie_args.clip_grad_value)
                            partitioned_p -= (self.lr * partitioned_grad)
                        else:
                            partitioned_grad = one_dim_grad.narrow(0, start, partition_size)
                            if self.collie_args.clip_grad_value is not None:
                                # Gradients are modified in-place.
                                partitioned_grad.clamp_(min=-self.collie_args.clip_grad_value,
                                                        max=self.collie_args.clip_grad_value)
                            p.ds_tensor -= (self.lr * partitioned_grad)
                        p.grad = None
            return x

        return func

    def train(self):
        for epoch in range(self.collie_args.num_train_epochs):
            print(f"***** Running Training *****")
            print(f"  Num examples: {len(self.train_dataset)}")
            print(f"  Num Epochs: {self.collie_args.num_train_epochs}")
            print(f"  Batch Size: {self.collie_args.per_device_train_batch_size}")
            if self.allow_print:
                self.wandb.log({'train/epoch': epoch}, step=self.global_step)

            with tqdm.tqdm(self.train_dataloader, disable=not self.allow_print) as tqb:
                for step, batch in enumerate(tqb, start=1):
                    self.model.train()
                    outs = self.model(
                        input_ids=batch['input_ids'].cuda(),
                        attention_mask=batch['attention_mask'].cuda(),
                    )
                    # Shift so that tokens < n predict n
                    shift_logits = outs.logits[..., :-1, :].contiguous()
                    shift_labels = batch['labels'][:, 1:].contiguous()
                    # Flatten the tokens
                    if self.collie_args.clip_loss_value is not None:
                        loss_fct = CrossEntropyLoss(reduction='none')
                        loss = loss_fct(shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
                                        shift_labels.view(-1).cuda())
                        loss.data.clamp_(min=-self.collie_args.clip_loss_value, max=self.collie_args.clip_loss_value)
                        loss = loss.mean()
                    else:
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
                                        shift_labels.view(-1).cuda())

                    # update the learning rate
                    self.global_step = self.num_steps_per_epoch * epoch + step
                    self.lr = self.lr_scheduler.step(self.global_step)
                    loss.backward()

                    # update the last one since the hook function will not be called for the last parameter
                    self.grad_func(0)
                    self.model.optimizer.get_param_coordinator(training=True).reset_step()

                    tqb.set_postfix({'loss': loss.item()})
                    if self.allow_print:
                        self.wandb.log(
                            {
                                'train/loss': loss.item(),
                                'train/learning_rate': self.lr,
                                'train/global_step': self.global_step,
                            },
                            step=self.global_step
                        )

                    if self.collie_args.do_eval and self.collie_args.evaluation_strategy == 'steps' and \
                            self.global_step % self.collie_args.eval_steps == 0:
                        if isinstance(self.eval_dataset, dict):
                            for prefix in self.eval_dataset.keys():
                                assert prefix in self.eval_dataloader.keys(), "eval_dataset and eval_dataloader should have the same keys."
                                self.eval(self.global_step, epoch, self.eval_dataset[prefix],
                                          self.eval_dataloader[prefix], prefix)
                        else:
                            self.eval(self.global_step, epoch, self.eval_dataset, self.eval_dataloader, 'eval')

            if self.collie_args.do_eval and self.collie_args.evaluation_strategy == 'epoch':
                if isinstance(self.eval_dataset, dict):
                    for prefix in self.eval_dataset.keys():
                        assert prefix in self.eval_dataloader.keys(), "eval_dataset and eval_dataloader should have the same keys."
                        self.eval(self.global_step, epoch, self.eval_dataset[prefix], self.eval_dataloader[prefix],
                                  prefix)
                else:
                    self.eval(self.global_step, epoch, self.eval_dataset, self.eval_dataloader, 'eval')

    def eval(
            self,
            step: int,
            epoch: int,
            dataset: torch.utils.data.Dataset,
            dataloader: DataLoader,
            eval_prefix: str
    ):
        r"""
                Shared by both eval(validation) and predict(test).
                This method will be called by the trainer to evaluate the model.
                """
        print(f"***** Running {eval_prefix} *****")
        print(f"  Num examples: {len(dataset)}")
        print(f"  Batch size: {self.collie_args.per_device_eval_batch_size}")

        with tqdm.tqdm(dataloader, disable=not self.allow_print) as tqb:
            all_preds = None
            self.model.eval()
            for batch in tqb:
                with torch.no_grad():
                    pred = self.eval_step(batch)
                    all_preds = pred if all_preds is None else all_preds + pred

            all_preds_gather = [None for _ in range(self.collie_args.world_size)]
            torch.distributed.all_gather_object(all_preds_gather, all_preds)
            all_pred_merged = list(chain(*all_preds_gather))

            result = self.compute_metrics(all_pred_merged, dataset)
            result = {f"{eval_prefix}/{k}": v for k, v in result.items()}
            prefix_metric_for_best_model = f'{eval_prefix}/{self.collie_args.metric_for_best_model}'
            result_value = result[prefix_metric_for_best_model]

            if self.allow_print:
                print(f'epoch: {epoch}, step: {step}, {self.collie_args.metric_for_best_model}: {result_value}')
                self.wandb.log(result, step=step)

                if self.is_better(result, prefix_metric_for_best_model):
                    self.wandb.set_summary(f'{eval_prefix}/best_{self.collie_args.metric_for_best_model}', result_value)
                    self.wandb.set_summary(f'{eval_prefix}/best_epoch', epoch)
                    self.wandb.set_summary(f'{eval_prefix}/best_step', step)
                    self.metrics[prefix_metric_for_best_model] = result_value

    def eval_step(self, batch):
        self.model.eval()
        logits = self.model.generate(
            batch['input_ids'].cuda(),
            batch['attention_mask'].cuda(),
            max_new_tokens=self.collie_args.max_new_tokens,
            temperature=self.collie_args.temperature,
            top_p=self.collie_args.top_p
        )
        logits = logits.tolist()
        pred_texts = self.tokenizer.batch_decode(logits)
        return pred_texts

    def is_better(self, result_dict, key):
        """
        判断 ``result`` 是否更好。

        :param result:
        """
        op = operator.gt if self.collie_args.greater_is_better else operator.lt
        return (
                key not in self.metrics or op(result_dict[key], self.metrics[key])
        )

    def get_train_sampler(self):
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
        # `self.collie_args.seed`) if data_seed isn't provided.
        # Further on in this method, we default to `self.collie_args.seed` instead.
        seed = self.collie_args.data_seed if self.collie_args.data_seed is not None else self.collie_args.seed

        if self.collie_args.group_by_length:
            return DistributedLengthGroupedSampler(
                self.collie_args.per_device_train_batch_size * self.collie_args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                num_replicas=self.collie_args.world_size,
                rank=self.collie_args.local_rank,
                lengths=None,
                model_input_name="input_ids",
                seed=seed,
            )
        else:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.collie_args.world_size,
                rank=self.collie_args.local_rank,
                seed=seed
            )

    def get_train_dataloader(self):
        """
            Returns the training [`~torch.utils.data.DataLoader`].
            Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
            training if necessary) otherwise.
            Subclass and override this method if you want to inject some custom behavior.
            """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        data_collator = self.train_data_collator
        train_sampler = self.get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.collie_args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.collie_args.dataloader_drop_last,
            num_workers=self.collie_args.dataloader_num_workers,
            pin_memory=self.collie_args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def get_eval_sampler(self, eval_dataset):
        return SequentialDistributedSampler(
            eval_dataset,
            num_replicas=self.collie_args.world_size,
            rank=self.collie_args.local_rank,
            # batch_size=self.collie_args.per_device_eval_batch_size
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """
            Returns the evaluation [`~torch.utils.data.DataLoader`].

            Subclass and override this method if you want to inject some custom behavior.

            Args:
                eval_dataset (`torch.utils.data.Dataset`, *optional*):
                    If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                    by the `model.forward()` method are automatically removed. It must implement `__len__`.
            """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.eval_data_collator

        eval_sampler = self.get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.collie_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.collie_args.dataloader_drop_last,
            num_workers=self.collie_args.dataloader_num_workers,
            pin_memory=self.collie_args.dataloader_pin_memory,
        )

    def save_model(self):
        with GatheredParameters(self.model.parameters(), modifier_rank=0) as gathered_params:
            if self.collie_args.local_rank == 0:
                state_dict = self.model.module.state_dict()
                torch.save(state_dict, os.path.join(self.collie_args.output_dir, 'pytorch_model.bin'))
            torch.distributed.barrier()


class GatheredParameters:
    def __init__(self, params, modifier_rank=None, fwd_module=None, enabled=True):
        """A context that collects parameters that were partitioned via a
        :class:`deepspeed.zero.Init` context. The parameters are partitioned
        again upon exit.

        Args:
            params (``torch.nn.Parameter``): A single parameter, or an iterable of parameters (list, tuple, generator) of parameters to collect.
                It's assumed that all parameters are zero params.
            modifier_rank (int, optional): If specified, this rank's parameter will be
                broadcasted on exit from the context. This argument is required if ``params`` are
                modified, so that all processes have a consistent view of the data. Defaults
                to ``None``.
            fwd_module (``torch.nn.Module``, optional): If specified, ``params`` will be
                registered as external parameters of ``fwd_module``. See :meth:`deepspeed.zero.register_external_parameter`.
            enabled (bool, optional): If ``False``, this context is a no-op. Defaults to ``True``.

        Important: Make sure to use ``modifier_rank`` that is not ``None`` (e.g., ``modifier_rank=0``)
        if you need the GPU memory allocated by gather to be released upon exit from the context manager.

        Important: if ``params`` isn't an iterable of parameters or a single parameter it'll be silently ignored!

        Examples
        ========

        #. Allocate a partitioned module, initialize its weight on rank 0, and update all
           processes.

            .. code-block:: python

                with deepspeed.zero.Init():
                    linear = torch.nn.Linear(1000,1000)

                with deepspeed.zero.GatheredParameters(linear.weight,
                                                       modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
                        linear.weight.zero_()

                with deepspeed.zero.GatheredParameters(linear.weight,
                                                       modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
                        linear.weight.zero_()

        #. Collect a partitioned weight to pass to another module during
           training. The parameter will be registered as an external parameter
           and made available during the backward pass.

            .. code-block:: python
                :emphasize-lines: 6

                def forward(self, input):
                    x = self.layer1(input)

                    # self.layer1.weight is required by self.layer2.forward
                    with deepspeed.zero.GatheredParameters(self.layer1.weight,
                                                           fwd_module=self):
                        y = self.layer2(x, self.layer1.weight)
                    return y


        #. Pretrained model loading

            .. code-block:: python

                with deepspeed.zero.Init():
                    model = MyModel()

                state_dict = torch.load(model_path, map_location="cpu")

                def load(module: nn.Module, prefix=""):
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                        if deepspeed.comm.get_rank() == 0:
                            module._load_from_state_dict(state_dict, prefix)

                    for name, child in module._modules.items():
                        if child is not None:
                            load(child, prefix + name + ".")

                load(model, prefix="")

        If this approach is not used, then the full model will first be copied to each GPU. For models
        bigger than the memory of a single GPU, this method is required.
        """

        self.enabled = enabled
        if not enabled:
            return

        if isinstance(params, Iterable) and not isinstance(params, torch.Tensor):
            # deal with generators like model.parameters()
            # must convert to list to be able to iterate more than once if we get a generator
            params = list(params)
        else:
            # single param
            params = [params]

        self.params = [p for p in params if hasattr(p, "ds_id")]
        self.src_rank = None
        if modifier_rank is not None:
            if self.params[0].ds_process_group == dist.get_world_group():
                self.src_rank = modifier_rank
            else:
                # A group was specified; convert DP rank to global rank
                self.src_rank = dist.get_global_rank(self.params[0].ds_process_group,
                                                     modifier_rank)

    def __enter__(self):
        if not self.enabled:
            return
        self.params[0].all_gather(param_list=self.params)

    def __exit__(self, *exc):
        if not self.enabled:
            return
        if self.src_rank is None:
            self.params[0].partition(param_list=self.params, has_been_updated=False)
            return

        handles = [
            dist.broadcast(p,
                           self.src_rank,
                           group=p.ds_process_group,
                           async_op=True) for p in self.params
        ]
        for h in handles:
            h.wait()
        for p in self.params[0]:
            p.active_sub_modules = None
        self.params[0].partition(param_list=self.params, has_been_updated=True)
