import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from transformers.trainer_pt_utils import nested_numpify, nested_concat, LengthGroupedSampler
from transformers.trainer_utils import has_length, seed_worker
import tqdm
import operator
import wandb

from .utils import LearningRateScheduler


class InplaceTensorTrainer:
    def __init__(
            self,
            model,
            tl_args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
    ):
        self.model = model
        self.tl_args = tl_args
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
        self.n_steps = self.num_steps_per_epoch * self.tl_args.num_train_epochs
        self.lr_scheduler = LearningRateScheduler(learning_rate=self.tl_args.learning_rate,
                                                  warmup=self.tl_args.warmup,
                                                  schedule=self.tl_args.lr_scheduler_type,
                                                  n_steps=self.n_steps)
        self.lr = 0

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
                    if p.requires_grad and p.grad is not None and p.shape != torch.Size([0]):
                        if self.tl_args.clip_grad_value is not None:
                            # Graidiens are modified in-palce.
                            p.grad.data.clamp_(min=-self.tl_args.clip_grad_value, max=self.tl_args.clip_grad_value)
                        p.data -= (self.lr * p.grad.data)
                        p.grad = None
            return x

        return func

    def train(self):
        for epoch in range(self.tl_args.num_train_epochs):
            print(f"***** Running Training *****")
            print(f"  Num examples: {len(self.train_dataset)}")
            print(f"  Num Epochs: {self.tl_args.num_train_epochs}")
            print(f"  Batch Size: {self.tl_args.per_device_train_batch_size}")
            wandb.log({'train/epoch': epoch}, step=self.global_step)

            with tqdm.tqdm(self.train_dataloader, disable=self.tl_args.local_rank not in [0, -1]) as tqb:
                for step, batch in enumerate(tqb, start=1):
                    self.model.train()
                    outs = self.model(batch['input_ids'], batch['attention_mask'])

                    # Shift so that tokens < n predict n
                    shift_logits = outs[..., :-1, :].contiguous()
                    shift_labels = batch['labels'][:, 1:].contiguous()
                    # Flatten the tokens
                    if self.tl_args.clip_loss_value is not None:
                        loss_fct = CrossEntropyLoss(reduction='none')
                        loss = loss_fct(shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
                                        shift_labels.view(-1))
                        loss.data.clamp_(min=-self.tl_args.clip_loss_value, max=self.tl_args.clip_loss_value)
                        loss = loss.mean()
                    else:
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
                                        shift_labels.view(-1))

                    # update the learning rate
                    self.global_step = self.num_steps_per_epoch * epoch + step
                    self.lr = self.lr_scheduler.step(self.global_step)
                    loss.backward()

                    # update the last one since the hook function will not be called for the last parameter
                    self.grad_func(0)
                    tqb.set_postfix({'loss': loss.item()})
                    if self.tl_args.local_rank in [0, -1]:
                        logs = {
                            'train/loss': loss.item(),
                            'train/learning_rate': self.lr,
                            'train/global_step': self.global_step,
                        }
                        wandb.log(logs, step=self.global_step)

                    if self.tl_args.do_eval and self.tl_args.evaluation_strategy == 'steps' and \
                            self.global_step % self.tl_args.eval_steps == 0:
                        if isinstance(self.eval_dataset, dict):
                            for prefix in self.eval_dataset.keys():
                                assert prefix in self.eval_dataloader.keys(), "eval_dataset and eval_dataloader should have the same keys."
                                self.eval(self.global_step, epoch, self.eval_dataset[prefix], self.eval_dataloader[prefix], prefix)
                        else:
                            self.eval(self.global_step, epoch, self.eval_dataset, self.eval_dataloader, 'eval')

            if self.tl_args.do_eval and self.tl_args.evaluation_strategy == 'epoch':
                if isinstance(self.eval_dataset, dict):
                    for prefix in self.eval_dataset.keys():
                        assert prefix in self.eval_dataloader.keys(), "eval_dataset and eval_dataloader should have the same keys."
                        self.eval(self.global_step, epoch, self.eval_dataset[prefix], self.eval_dataloader[prefix], prefix)
                else:
                    self.eval(self.global_step, epoch, self.eval_dataset, self.eval_dataloader, 'eval')

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
        print(f"  Batch size: {self.tl_args.per_device_eval_batch_size}")

        with tqdm.tqdm(dataloader, disable=self.tl_args.local_rank not in [0, -1]) as tqb:
            all_preds = None
            self.model.eval()
            for batch in tqb:
                with torch.no_grad():
                    pred = self.eval_step(batch)
                    all_preds = pred if all_preds is None else all_preds + pred

            result = self.compute_metrics(all_preds, dataset)
            result = {f"{eval_prefix}/{k}": v for k, v in result.items()}
            prefix_metric_for_best_model = f'{eval_prefix}/{self.tl_args.metric_for_best_model}'

            if self.tl_args.local_rank in [-1, 0]:
                print(f'epoch: {epoch}, step: {step}, {self.tl_args.metric_for_best_model}: '
                      f'{result[prefix_metric_for_best_model]}')
                wandb.log(result, step=step)

            op = operator.gt if self.tl_args.greater_is_better else operator.lt
            if prefix_metric_for_best_model not in self.metrics or op(result[prefix_metric_for_best_model], self.metrics[prefix_metric_for_best_model]):
                self.metrics[prefix_metric_for_best_model] = result[prefix_metric_for_best_model]
                if self.tl_args.local_rank in [-1, 0]:
                    wandb.run.summary[f'{eval_prefix}/best_{self.tl_args.metric_for_best_model}'] = \
                        result[prefix_metric_for_best_model]
                    wandb.run.summary[f'{eval_prefix}/best_epoch'] = epoch
                    wandb.run.summary[f'{eval_prefix}/best_step'] = step

    def eval_step(self, batch):
        self.model.eval()
        logits = self.model.generate(
            batch['input_ids'], batch['attention_mask'],
            max_new_tokens=self.tl_args.max_new_tokens,
            temperature=self.tl_args.temperature,
            top_p=self.tl_args.top_p
        )
        logits = logits.tolist()
        pred_texts = self.tokenizer.batch_decode(logits)
        return pred_texts
    
    def get_train_sampler(self):
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = torch.Generator(device='cuda')
        # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
        # `self.tl_args.seed`) if data_seed isn't provided.
        # Further on in this method, we default to `self.tl_args.seed` instead.
        seed = self.tl_args.data_seed if self.tl_args.data_seed is not None else self.tl_args.seed
        generator.manual_seed(seed)

        if self.tl_args.group_by_length:
            return LengthGroupedSampler(
                self.tl_args.per_device_train_batch_size * self.tl_args.gradient_accumulation_steps,
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
            batch_size=self.tl_args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.tl_args.dataloader_drop_last,
            num_workers=self.tl_args.dataloader_num_workers,
            pin_memory=self.tl_args.dataloader_pin_memory,
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
            batch_size=self.tl_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.tl_args.dataloader_drop_last,
            num_workers=self.tl_args.dataloader_num_workers,
            pin_memory=self.tl_args.dataloader_pin_memory,
        )
