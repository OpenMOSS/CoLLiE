import sys

import torch
import numpy as np

try:
    import wandb
except:
    pass

# An approximation of in-place grad update
def inplace_grad(model, lr=5e-4):
    def func(x):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None and p.shape != torch.Size([0]):
                    p.data -= (lr * p.grad.data)
                    p.grad = None
        return x
    return func

class GPTLMLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad> when compute loss

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class LearningRateScheduler:
    r"""
    Learning rate scheduler with warmup.

        :param warmup: if ``warmup`` is an integer, ``warmup`` stands for warmup steps, if ``warmup`` is a float,
            such as 0.1, then it stands for warmup_ratio.
        :param schedule: the learning rate will be adjusted according to ``schedule`` strategy,
            which can be: linear or constant.
    """

    def __init__(self,
                 warmup: float,
                 schedule: str,
                 learning_rate: float,
                 n_steps: int = 0):

        self.warmup = max(warmup, 0.)
        self.schedule = schedule
        self.initial_lr = learning_rate

        if self.warmup > 1:
            self.warmup = self.warmup / n_steps
        self.t_steps = max(2, n_steps)

        if self.schedule == 'constant':
            self.get_lr = self._get_constant_lr
        elif self.schedule == 'linear':
            self.get_lr = self._get_linear_lr
        else:
            raise NotImplementedError("Only support 'linear', 'constant'.")

    def _get_constant_lr(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return 1

    def _get_linear_lr(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)

    def step(self, global_step):
        progress = global_step / self.t_steps
        return self.initial_lr * self.get_lr(progress)
    
class WandbLogger:
    """
    使用 wandb 记录信息的类。

    :param collie_args: Collie 的参数
    """
    def __init__(self, collie_args):
        self.collie_args = collie_args
        # report_to is a list
        self.able = "wandb" in getattr(collie_args, "report_to", [])
        if self.able and 'wandb' not in sys.modules:
            raise ModuleNotFoundError(
                "Detected Wandb not installed while you have set "
                "`report_to=['wandb']` in your collie config. Please "
                "either set `report_to` to another value or install wandb.")

    def log(self, *args, **kwargs):
        if self.able:
            wandb.log(*args, **kwargs)

    def set_summary(self, key, value):
        if self.able:
            wandb.run.summary[key] = value
