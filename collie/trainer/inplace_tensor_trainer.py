import operator
import os
import json
from shutil import copyfile
from dataclasses import asdict

import tqdm
from itertools import chain
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import has_length, seed_worker

from collie.log import print
from .utils import LearningRateScheduler, WandbLogger


class InplaceTensorTrainer:
    def __init__(
            self,
            model,
            collie_args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
            cache_dir=None,
    ):
        self.model = model
        self.collie_args = collie_args
        if isinstance(data_collator, dict):
            assert 'train' in data_collator and 'eval' in data_collator, "data_collator should be a dict with keys 'train' and 'eval'."
            self.train_data_collator = data_collator['train']
            self.eval_data_collator = data_collator['eval']
        else:
            self.train_data_collator = self.eval_data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.cache_dir = cache_dir
        self.wandb = WandbLogger(collie_args)
        self.allow_print = self.collie_args.local_rank in [0, -1]
        self.metrics = {}
        
        self.train_dataloader = self.get_train_dataloader()
        if isinstance(self.eval_dataset, dict):
            self.eval_dataloader = {}
            for prefix in self.eval_dataset.keys():
                self.eval_dataloader[prefix] = self.get_eval_dataloader(self.eval_dataset[prefix])
        else:
            self.eval_dataloader = self.get_eval_dataloader()

        self.num_steps_per_epoch = len(self.train_dataloader)
        self.global_step = 1
        self.n_steps = self.num_steps_per_epoch * self.collie_args.num_train_epochs
        self.lr_scheduler = LearningRateScheduler(learning_rate=self.collie_args.learning_rate,
                                                  warmup=self.collie_args.warmup,
                                                  schedule=self.collie_args.lr_scheduler_type,
                                                  n_steps=self.n_steps)
        self.lr = 0
        self.gather_norm = False
        self.grad_norms = []
        self.clip_coef = None

        # register inplace grad hook
        self.grad_func = self.inplace_grad()
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.register_hook(self.grad_func)

    def inplace_grad(self):
        # An approximation of in-place grad update
        def func(x):
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        if self.gather_norm:
                            self.grad_norms.append(torch.norm(p.grad, 2.0))
                            p.grad = None
                        else:
                            if self.collie_args.clip_grad_value is not None and self.collie_args.clip_grad_value > 0:
                                # Gradients are modified in-place.
                                p.grad.data.clamp_(min=-self.collie_args.clip_grad_value, max=self.collie_args.clip_grad_value)
                            if self.collie_args.clip_grad_norm is not None and self.collie_args.clip_grad_norm > 0 and self.clip_coef is not None:
                                p.grad.data.mul_(self.clip_coef)
                            p.data -= (self.lr * p.grad.data)
                            p.grad = None
            return x

        return func

    def train(self):
        for epoch in range(self.collie_args.num_train_epochs):
            print(f"***** Running Training *****")
            print(f"  Num examples: {len(self.train_dataset)}")
            print(f"  Current Epochs: {epoch}")
            print(f"  Batch Size: {self.collie_args.per_device_train_batch_size}")
            if self.allow_print:
                self.wandb.log({'train/epoch': epoch}, step=self.global_step)

            with tqdm.tqdm(self.train_dataloader, disable=not self.allow_print) as tqb:
                for step, batch in enumerate(tqb, start=1):
                    self.model.train()
                    outs = self.model(batch['input_ids'], batch['attention_mask'])

                    # Shift so that tokens < n predict n
                    shift_logits = outs[..., :-1, :].contiguous()
                    shift_labels = batch['labels'][:, 1:].contiguous()
                    # Flatten the tokens
                    if self.collie_args.clip_loss_value is not None:
                        loss_fct = CrossEntropyLoss(reduction='none')
                        loss = loss_fct(shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
                                        shift_labels.view(-1))
                        loss.data.clamp_(min=-self.collie_args.clip_loss_value, max=self.collie_args.clip_loss_value)
                        loss = loss.mean()
                    else:
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
                                        shift_labels.view(-1))

                    # update the learning rate
                    self.global_step = self.num_steps_per_epoch * epoch + step
                    self.lr = self.lr_scheduler.step(self.global_step)
                    if self.collie_args.clip_grad_norm is not None and self.collie_args.clip_grad_norm > 0:
                        self.gather_norm = True
                        self.grad_norms = []

                        loss.backward(retain_graph=True)
                        # update the last one since the hook function will not be called for the last parameter
                        self.grad_func(0)

                        with torch.no_grad():
                            # The norm is computed over all gradients together, as if they were
                            # concatenated into a single vector. Gradients are modified in-place.
                            self.grad_norms = torch.stack(self.grad_norms)
                            device = torch.device(f"cuda:{self.collie_args.local_rank}")
                            all_grad_norms = torch.zeros(self.collie_args.world_size * self.grad_norms.shape[0], dtype=self.grad_norms.dtype, device=device)
                            torch.distributed.all_gather_into_tensor(all_grad_norms, self.grad_norms)

                            total_norm = torch.norm(all_grad_norms, 2.0)
                            self.clip_coef = float(self.collie_args.clip_grad_norm) / (total_norm + 1e-6)
                            self.clip_coef = torch.clamp(self.clip_coef, max=1.0)
                        self.gather_norm = False

                    loss.backward()
                    # update the last one since the hook function will not be called for the last parameter
                    self.grad_func(0)

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

                    if self.collie_args.save_strategy == 'steps' and self.global_step % self.collie_args.save_steps == 0:
                        self.save_model(self.global_step)

                    if self.collie_args.do_eval and self.collie_args.evaluation_strategy == 'steps' and \
                            self.global_step % self.collie_args.eval_steps == 0:
                        if isinstance(self.eval_dataset, dict):
                            for prefix in self.eval_dataset.keys():
                                assert prefix in self.eval_dataloader.keys(), "eval_dataset and eval_dataloader should have the same keys."
                                self.eval(self.global_step, epoch, self.eval_dataset[prefix], self.eval_dataloader[prefix], prefix)
                        else:
                            self.eval(self.global_step, epoch, self.eval_dataset, self.eval_dataloader, 'eval')

            if self.collie_args.save_strategy == 'epoch':
                self.save_model(epoch)

            if self.collie_args.do_eval and self.collie_args.evaluation_strategy == 'epoch':
                if isinstance(self.eval_dataset, dict):
                    for prefix in self.eval_dataset.keys():
                        assert prefix in self.eval_dataloader.keys(), "eval_dataset and eval_dataloader should have the same keys."
                        self.eval(self.global_step, epoch, self.eval_dataset[prefix], self.eval_dataloader[prefix], prefix)
                else:
                    self.eval(self.global_step, epoch, self.eval_dataset, self.eval_dataloader, 'eval')

    def save_model(self, index):
        assert self.cache_dir is not None
        save_dir = os.path.join(self.collie_args.output_dir, f"checkpoint-{index}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"consolidated.{self.collie_args.local_rank:02d}.pth")
        print('Saving model to', save_path)

        states = {name: param.cpu().detach().clone() for name, param in self.model.state_dict().items()}
        torch.save(states, save_path)

        if self.collie_args.local_rank == 0:
            copyfile(os.path.join(self.cache_dir, 'params.json'), os.path.join(save_dir, 'params.json'))
        torch.distributed.barrier()

    def eval(self,
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
        print(f"  Current Epochs: {epoch}")
        print(f"  Batch size: {self.collie_args.per_device_eval_batch_size}")

        with tqdm.tqdm(dataloader, disable=not self.allow_print) as tqb:
            all_preds = None
            self.model.eval()
            for batch in tqb:
                with torch.no_grad():
                    pred = self.eval_step(batch)
                    all_preds = pred if all_preds is None else all_preds + pred

            result = self.compute_metrics(all_preds, dataset)
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
            batch['input_ids'], batch['attention_mask'],
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
            key not in self.metrics or \
            op(result_dict[key], self.metrics[key])
        )

    def get_train_sampler(self):
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = torch.Generator(device='cuda')
        # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
        # `self.collie_args.seed`) if data_seed isn't provided.
        # Further on in this method, we default to `self.collie_args.seed` instead.
        seed = self.collie_args.data_seed if self.collie_args.data_seed is not None else self.collie_args.seed
        generator.manual_seed(seed)

        if self.collie_args.group_by_length:
            return LengthGroupedSampler(
                self.collie_args.per_device_train_batch_size * self.collie_args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=None,
                model_input_name="input_ids",
                generator=generator,
            )
        else:
            return RandomSampler(self.train_dataset, generator=generator)

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
        return SequentialSampler(eval_dataset)

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