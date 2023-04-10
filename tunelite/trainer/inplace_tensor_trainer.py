from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from transformers.trainer_pt_utils import nested_numpify, nested_concat
from transformers.trainer_utils import seed_worker
import tqdm
import torch
import wandb

from .utils import LearningRateScheduler


class InplaceTensorTrainer:
    def __init__(
            self,
            model,
            tokenizer,
            train_dataloader,
            eval_dataloader,
            eval_dataset,
            tl_args,
            compute_metrics,
    ):
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.eval_dataset = eval_dataset
        self.model = model
        self.tokenizer = tokenizer
        self.tl_args = tl_args
        self.metric = None
        self.compute_metrics = compute_metrics

        self.num_steps_per_epoch = len(self.train_dataloader)
        self.global_step = 0
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
                        wandb.log(logs)

                    if self.tl_args.evaluation_strategy == 'steps' and \
                            self.global_step % self.tl_args.eval_steps == 0:

                        if self.tl_args.do_eval:
                            self.eval(self.global_step, epoch, self.eval_dataset, self.eval_dataloader, 'eval')
                        # if self.tl_args.do_predict:
                        #     self.eval(step, epoch, self.eval_dataset, self.eval_dataloader, 'predict')
            if self.tl_args.evaluation_strategy == 'epoch':
                if self.tl_args.do_eval:
                    self.eval(self.global_step, epoch, self.eval_dataset, self.eval_dataloader, 'eval')
                # if self.tl_args.do_predict:
                #     self.eval(step, epoch, self.eval_dataset, self.eval_dataloader, 'predict')

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

            result = self.compute_metrics(all_preds, self.eval_dataset)
            result = {f"{eval_prefix}/{k}": v for k, v in result.items()}
            result['epoch'] = epoch
            result['step'] = step

            prefix_metric_for_best_model = f'{eval_prefix}/{self.tl_args.metric_for_best_model}'

            if self.tl_args.local_rank in [-1, 0]:
                print(f'epoch: {epoch}, step: {step}, {self.tl_args.metric_for_best_model}: '
                      f'{result[prefix_metric_for_best_model]}')
                wandb.log(result)

            if self.tl_args.greater_is_better:
                if self.metric is None or result[prefix_metric_for_best_model] > self.metric:
                    if self.tl_args.local_rank in [-1, 0]:
                        wandb.run.summary[f'{eval_prefix}/best_{self.tl_args.metric_for_best_model}'] = \
                            result[prefix_metric_for_best_model]
                        wandb.run.summary[f'{eval_prefix}/best_epoch'] = epoch
                        wandb.run.summary[f'{eval_prefix}/best_step'] = step
            else:
                if self.metric is None or result[prefix_metric_for_best_model] < self.metric:
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
