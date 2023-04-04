from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from transformers.trainer_pt_utils import nested_numpify, nested_concat
from transformers.trainer_utils import seed_worker
import tqdm
from .utils import inplace_grad
import deepspeed
import torch
import wandb
from deepspeed.accelerator import get_accelerator


class InplaceZeroTrainer:
    def __init__(
            self,
            model,
            tokenizer,
            train_dataset,
            data_collator,
            eval_dataloader,
            eval_dataset,
            tl_args,
            data_args,
            compute_metrics,
    ):
        # Initialize deepspeed engine
        model, _, self.train_dataloader, _ = deepspeed.initialize(
            config=tl_args.deepspeed,
            model=model,
            training_data=train_dataset,
            collate_fn=lambda x: data_collator(x, tokenizer, max_len=data_args.max_length),
        )
        get_accelerator().empty_cache()

        # register inplace grad hook
        self.grad_func = inplace_grad(model, lr=tl_args.learning_rate)
        for n, p in model.named_parameters():
            p.register_hook(self.grad_func)

        # get eval dataloader
        self.eval_dataset = eval_dataset
        self.eval_dataloader = eval_dataloader

        self.model = model
        self.tokenizer = tokenizer
        self.tl_args = tl_args
        self.metric = None
        self.compute_metrics = compute_metrics

    def train(self):
        for epoch in range(self.tl_args.num_train_epochs):
            with tqdm.tqdm(self.train_dataloader, disable=self.tl_args.local_rank not in [0, -1]) as tqb:
                for step, batch in enumerate(tqb, start=1):
                    self.model.train()
                    outs = self.model(
                        input_ids=batch['input_ids'].cuda(self.tl_args.local_rank),
                        attention_mask=batch['attention_mask'].cuda(self.tl_args.local_rank),
                        labels=batch['labels'].cuda(self.tl_args.local_rank),
                    )
                    outs['loss'].backward()

                    # update the last one since the hook function will not be called for the last parameter
                    self.grad_func(0)
                    self.model.optimizer.get_param_coordinator(training=True).reset_step()

                    tqb.set_postfix({'loss': outs['loss'].item()})
                    if self.tl_args.local_rank in [0, -1]:
                        wandb.log({'loss': outs['loss'].item()})

                    if step % self.tl_args.eval_steps == 0 or step == len(self.train_dataloader):
                        self.eval(step, epoch)

    def eval(self, step, epoch):
        with tqdm.tqdm(self.eval_dataloader, disable=self.tl_args.local_rank not in [0, -1]) as tqb:
            all_preds = None
            self.model.eval()
            for batch in tqb:
                with torch.no_grad():
                    logits = self.model.module.generate(
                        batch['input_ids'].cuda(self.tl_args.local_rank),
                        max_new_tokens=self.tl_args.max_new_tokens,
                        temperature=self.tl_args.temperature,
                        top_p=self.tl_args.top_p
                    )
                    logits = logits.tolist()
                    pred_texts = self.tokenizer.batch_decode(logits)
                    print(pred_texts)
                    all_preds = pred_texts if all_preds is None else all_preds + pred_texts

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
                        wandb.run.summary['best_' + self.tl_args.metric_for_best_model] = result[
                            self.tl_args.metric_for_best_model]
                        wandb.run.summary['best_epoch'] = epoch
                        wandb.run.summary['best_step'] = step
            else:
                if self.metric is None or result[self.tl_args.metric_for_best_model] < self.metric:
                    if self.tl_args.local_rank in [-1, 0]:
                        wandb.run.summary['best_' + self.tl_args.metric_for_best_model] = result[
                            self.tl_args.metric_for_best_model]
                        wandb.run.summary['best_epoch'] = epoch
                        wandb.run.summary['best_step'] = step
