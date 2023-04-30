import os.path
import sys
import operator
from collections import OrderedDict
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
        self.wandb = WandbLogger(collie_args)
        self.allow_print = self.collie_args.local_rank in [0, -1]
        if self.collie_args.do_eval:
            self.metrics = {}
            self.compute_metrics = compute_metrics

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
            if self.collie_args.do_eval:
                self.eval_data_collator = data_collator['eval']
        else:
            self.train_data_collator = self.eval_data_collator = data_collator
        self.train_dataloader = self.get_train_dataloader()
        if self.collie_args.do_eval:
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
        # for grad norm
        self.gather_norm = False
        self.grad_norms = []
        self.clip_coef = None
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
                        if self.gather_norm:
                            self.grad_norms.append(torch.norm(p.grad, 2.0))
                            p.grad = None
                        else:
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
                                if self.collie_args.clip_grad_norm is not None and self.collie_args.clip_grad_norm > 0 and self.clip_coef is not None:
                                    partitioned_grad.mul_(self.clip_coef)
                                partitioned_p -= (self.lr * partitioned_grad)
                            else:
                                partitioned_grad = one_dim_grad.narrow(0, start, partition_size)
                                if self.collie_args.clip_grad_value is not None:
                                    # Gradients are modified in-place.
                                    partitioned_grad.clamp_(min=-self.collie_args.clip_grad_value,
                                                            max=self.collie_args.clip_grad_value)
                                if self.collie_args.clip_grad_norm is not None and self.collie_args.clip_grad_norm > 0 and self.clip_coef is not None:
                                    partitioned_grad.mul_(self.clip_coef)
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
        state_dict = OrderedDict()
        output_dir = self.collie_args.output_dir
        for n, p in self.model.module.named_parameters():
            state_dict[n] = (p.ds_tensor.detach().cpu(), p.ds_numel, p.ds_shape)
        os.makedirs(output_dir, exist_ok=True)
        # save model shards
        with open(os.path.join(output_dir, f'pytorch_model-{self.collie_args.local_rank}.bin'), 'wb') as f:
            torch.save(state_dict, f)
        torch.distributed.barrier()
        # merge model shards
        if self.collie_args.local_rank == 0:
            # save config
            self.model.module.config.save_pretrained(output_dir)
            for rank in range(1, self.collie_args.world_size):
                with open(os.path.join(output_dir, f'pytorch_model-{rank}.bin'), 'rb') as f:
                    state_dict_rank = torch.load(f)
                    for n in state_dict_rank:
                        print(n, state_dict[n][0].shape)
                        state_dict[n] = (
                            torch.cat([state_dict[n][0], state_dict_rank[n][0]], dim=0),
                            state_dict[n][1],
                            state_dict[n][2]
                        )
                        print(n, state_dict[n][0].shape)
                # remove shard files
                os.remove(os.path.join(output_dir, f'pytorch_model-{rank}.bin'))
            # reshape to original shape
            for n in state_dict:
                numel = state_dict[n][1]
                shape = state_dict[n][2]
                state_dict[n] = state_dict[n][0][:numel].view(shape)

            # save inv_freq for llama
            if self.model.module.config.model_type == "llama":
                num_layers = self.model.module.config.num_hidden_layers
                head_dim = self.model.module.config.hidden_size // self.model.module.config.num_attention_heads
                base = 10000.0
                inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
                for layer in num_layers:
                    state_dict[f'model.layers.{layer}.self_attn.rotary_emb.inv_freq'] = inv_freq


            with open(os.path.join(output_dir, f'pytorch_model.bin'), 'wb') as f:
                torch.save(state_dict, f)
        torch.distributed.barrier()
