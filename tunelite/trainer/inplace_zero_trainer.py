import sys
import operator

import tqdm
import torch

try:
    import deepspeed
    from deepspeed.accelerator import get_accelerator
except:
    pass

from .utils import inplace_grad, WandbLogger

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
        if 'deepspeed' not in sys.modules:
            raise ModuleNotFoundError(
                "Detected DeepSpeed is not installed. See https://github.com/microsoft/DeepSpeed")
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
        self.wandb = WandbLogger(tl_args)
        self.allow_print = self.tl_args.local_rank in [0, -1]

    def train(self):
        for epoch in range(self.tl_args.num_train_epochs):
            with tqdm.tqdm(self.train_dataloader, disable=not self.allow_print) as tqb:
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
                    if self.allow_print:
                        self.wandb.log({'loss': outs['loss'].item()})

                    if step % self.tl_args.eval_steps == 0 or step == len(self.train_dataloader):
                        self.eval(step, epoch)

    def eval(self, step, epoch):
        with tqdm.tqdm(self.eval_dataloader, disable=not self.allow_print) as tqb:
            all_preds = None
            self.model.eval()
            for batch in tqb:
                with torch.no_grad():
                    pred = self.eval_step(batch)
                    all_preds = pred if all_preds is None else all_preds + pred

            result = self.compute_metrics(all_preds, self.eval_dataset)
            result['epoch'] = epoch
            result['step'] = step
            result_value = result[self.tl_args.metric_for_best_model]

            if self.allow_print:
                print('epoch: ', epoch, 'step: ', step, self.tl_args.metric_for_best_model, ': ', result_value)
                self.wandb.log(result)

                if self.is_better(result_value):
                    self.wandb.set_summary('best_' + self.tl_args.metric_for_best_model, result_value)
                    self.wandb.set_summary(f'best_epoch', epoch)
                    self.wandb.set_summary(f'best_step', step)
                    self.metric = result_value

    def eval_step(self, batch):
        logits = self.model.generate(
            batch['input_ids'], batch['attention_mask'],
            max_new_tokens=self.tl_args.max_new_tokens,
            temperature=self.tl_args.temperature,
            top_p=self.tl_args.top_p
        )
        logits = logits.tolist()
        pred_texts = self.tokenizer.batch_decode(logits)
        return pred_texts
    
    def is_better(self, result_dict, key):
        """
        判断 ``result`` 是否更好。

        :param result:
        """
        op = operator.gt if self.tl_args.greater_is_better else operator.lt
        return (
            key not in self.metrics or \
            op(result_dict[key], self.metrics[key])
        )
