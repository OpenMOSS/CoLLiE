from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from transformers.trainer_pt_utils import nested_numpify, nested_concat
from transformers.trainer_utils import seed_worker
import tqdm
import torch
import wandb


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
                        p.data -= (self.tl_args.learning_rate * p.grad.data)
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

                    loss.backward()
                    # update the last one since the hook function will not be called for the last parameter
                    self.grad_func(0)
                    tqb.set_postfix({'loss': loss.item()})
                    if self.tl_args.local_rank in [0, -1]:
                        wandb.log({'loss': loss.item()})

                    if step % self.tl_args.eval_steps == 0 or step == len(self.train_dataloader):
                        self.eval(step, epoch)

    def eval(self, step, epoch):
        with tqdm.tqdm(self.eval_dataloader, disable=self.tl_args.local_rank not in [0, -1]) as tqb:
            all_preds = None
            self.model.eval()
            for batch in tqb:
                with torch.no_grad():
                    pred = self.eval_step(batch)
                    all_preds = pred if all_preds is None else all_preds + pred

            result = self.compute_metrics(all_preds, self.eval_dataset)
            result['epoch'] = epoch
            result['step'] = step

            if self.tl_args.local_rank in [-1, 0]:
                print('epoch: ', epoch, 'step: ', step, self.tl_args.metric_for_best_model, ': ',
                      result[self.tl_args.metric_for_best_model])
                wandb.log(result)

            if self.tl_args.greater_is_better:
                if self.metric is None or result[self.tl_args.metric_for_best_model] > self.metric:
                    if self.tl_args.local_rank in [-1, 0]:
                        wandb.run.summary['best_'+self.tl_args.metric_for_best_model] = result[self.tl_args.metric_for_best_model]
                        wandb.run.summary['best_epoch'] = epoch
                        wandb.run.summary['best_step'] = step
            else:
                if self.metric is None or result[self.tl_args.metric_for_best_model] < self.metric:
                    if self.tl_args.local_rank in [-1, 0]:
                        wandb.run.summary['best_'+self.tl_args.metric_for_best_model] = result[self.tl_args.metric_for_best_model]
                        wandb.run.summary['best_epoch'] = epoch
                        wandb.run.summary['best_step'] = step

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
