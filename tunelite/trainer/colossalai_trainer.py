import sys

sys.path.append("..")

from .utils import GPTLMLoss, sample_top_p

import os
import tqdm
import torch
from itertools import cycle
from functools import partial

from typing import List, Tuple, Dict

from dataclasses import dataclass, field

try:
    import colossalai
    from colossalai.core import global_context as gpc
    from colossalai.context.parallel_mode import ParallelMode
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Detected Colossal-AI is not installed. See https://github.com/hpcaitech/ColossalAI")


@dataclass
class TrainerArgs:
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for SGD."})
    epochs: int = field(
        default=10,
        metadata={"help": "Total number of training epochs to perform. "
                          "If it is set to -1, the training will run forever."
                          "If it is set to 0, the training will not run."})
    eval_per_steps: int = field(
        default=10,
        metadata={"help": "The number of steps to perform evaluation. "
                          "If it is set to -1, the evaluation will run after every training step."
                          "If it is set to 0, the evaluation will not run after training step."})
    eval_per_epoches: int = field(
        default=1,
        metadata={"help": "The number of epochs to perform evaluation. "
                          "If it is set to -1, the evaluation will run after every training epoch."
                          "If it is set to 0, the evaluation will not run after training epoch."})
    eval_max_length: int = field(
        default=128,
        metadata={"help": "The maximum length of generated text when evaluating."
                          "If it is set to -1, the evaluation will run until the stop tokens are generated."})
    eval_stop_tokens: List[int] = field(
        default_factory=partial(list, [2]),
        metadata={"help": "The stop tokens when evaluating or generating."})
    eval_top_p: float = field(
        default=0.9,
        metadata={"help": "The top_p when evaluating or generating."})
    eval_temperature: float = field(
        default=1.0,
        metadata={"help": "The temperature when evaluating or generating."})
    eval_use_cache: bool = field(
        default=True,
        metadata={"help": "Whether to use key/value cache when evaluating or generating."})
    inplace: bool = field(
        default=False,
        metadata={"help": "Whether to use inplace gradient update."})


def inplace_grad(model, lr=5e-4, micro_batch_num: int = 1):
    def func(x):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None and p.shape != torch.Size([0]):
                    p.data -= (lr * p.grad.data) / micro_batch_num
                    p.grad = None
        return x

    return func


class ColossalaiTrainer:
    def __init__(self,
                 model,
                 tokenizer,
                 train_dataloader,
                 eval_dataloader,
                 compute_metrics=None,
                 trainer_args: TrainerArgs = TrainerArgs()) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.trainer_args = trainer_args
        if self.trainer_args.inplace:
            self.grad_func = inplace_grad(model, lr=trainer_args.learning_rate,
                                          micro_batch_num=model.model_args.micro_batch_num)
            for n, p in self.model.named_parameters():
                p.register_hook(self.grad_func)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=trainer_args.learning_rate)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=trainer_args.learning_rate)
        self.engine, self.train_dataloader, self.eval_dataloader, _ = colossalai.initialize(
            model=self.model,
            train_dataloader=train_dataloader,
            test_dataloader=eval_dataloader,
            optimizer=self.optimizer,
            criterion=GPTLMLoss()
        )

    def train(self):
        self.engine.train()

        def train_loop(epoch: int = 0):
            with tqdm.tqdm(self.train_dataloader, disable=not gpc.is_pipeline_last_stage()) as tqb:
                for step, batch in enumerate(tqb, start=1):
                    self.engine.zero_grad()
                    _, _, loss = self.engine.execute_schedule(
                        cycle([batch]),
                        forward_only=False,
                        return_loss=True,
                        return_output_label=False,
                    )
                    if not self.trainer_args.inplace:
                        self.engine.step()
                    torch.cuda.empty_cache()
                    if gpc.is_pipeline_last_stage():
                        tqb.set_postfix({'epoch': epoch, 'step': step, 'loss': loss.item()})

                    if self.trainer_args.eval_per_steps == 0:
                        continue
                    elif self.trainer_args.eval_per_steps == -1:
                        self.eval(epoch, step)
                    elif self.trainer_args.eval_per_steps > 0 and step % self.trainer_args.eval_per_steps == 0:
                        self.eval(epoch, step)

        if self.trainer_args.epochs == 0:
            return
        elif self.trainer_args.epochs == -1:
            epoch = 0
            while True:
                train_loop(epoch)
                epoch = epoch + 1
                if self.trainer_args.eval_per_epoches == 0:
                    continue
                elif self.trainer_args.eval_per_epoches == -1:
                    self.eval(epoch, 0)
                elif self.trainer_args.eval_per_epoches > 0 and epoch % self.trainer_args.eval_per_epoches == 0:
                    self.eval(epoch, 0)
        elif self.trainer_args.epochs > 0:
            for epoch in range(self.trainer_args.epochs):
                train_loop(epoch)
                if self.trainer_args.eval_per_epoches == 0:
                    continue
                elif self.trainer_args.eval_per_epoches == -1:
                    self.eval(epoch, 0)
                elif self.trainer_args.eval_per_epoches > 0 and (epoch + 1) % self.trainer_args.eval_per_epoches == 0:
                    self.eval(epoch, 0)

    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor,
                 max_length: int = 1024,
                 use_cache: bool = True,
                 stop_tokens: List[int] = [2]):
        assert input_ids.ndim == 2, "input_ids must be 2D tensor (B, N)"
        min_len = min([len(torch.nonzero(sample)) for sample in input_ids])
        if max_length > input_ids.shape[1]:
            input_ids = torch.nn.functional.pad(input_ids, (0, max_length - input_ids.shape[1]), value=0)
        stop_flag = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        cached_len = 0
        with tqdm.tqdm(range(min_len, max_length), disable=not gpc.is_pipeline_last_stage()) as tqb:
            for current_pos in tqb:
                try:
                    working_batch_idx = torch.flatten(torch.argwhere(~stop_flag))
                    if len(working_batch_idx) == 0:
                        raise StopIteration
                    self.engine.eval()
                    hidden_states, label, _ = self.engine.execute_schedule(
                        cycle([({
                                    "input_ids": input_ids[working_batch_idx,
                                                 cached_len:current_pos] if use_cache else input_ids[working_batch_idx,
                                                                                           :current_pos],
                                    "use_cache": torch.ones(input_ids.shape[0],
                                                            dtype=torch.bool) if use_cache else torch.zeros(
                                        input_ids.shape[0], dtype=torch.bool)
                                }, input_ids)]),
                        forward_only=True,
                        return_loss=False,
                        return_output_label=True,
                    )
                    cached_len = current_pos
                    next_tokens = torch.zeros(input_ids.shape[0], 1, dtype=torch.long).to(
                        torch.device(f"cuda:{os.environ['LOCAL_RANK']}"))
                    if gpc.is_pipeline_last_stage():
                        next_tokens = torch.argmax(hidden_states[:, -1, :], dim=-1)
                        next_tokens = torch.unsqueeze(next_tokens, dim=-1)
                    torch.distributed.broadcast(next_tokens, src=gpc.get_world_size(ParallelMode.PIPELINE) - 1)
                    next_tokens = next_tokens.to(input_ids.device)
                    input_ids[working_batch_idx, current_pos] = torch.where(
                        input_ids[working_batch_idx, current_pos] == 0, next_tokens[:, 0],
                        input_ids[working_batch_idx, current_pos])
                    tqb.set_postfix({'generating': f"{current_pos}/{max_length}"})
                    for i in range(len(torch.flatten(input_ids[:, current_pos]).tolist())):
                        if torch.flatten(input_ids[:, current_pos]).tolist()[i] in stop_tokens:
                            stop_flag[i] = True
                    torch.cuda.empty_cache()
                except StopIteration:
                    break
        return input_ids

    def eval(self, epoch=0, step=0):
        with tqdm.tqdm(self.eval_dataloader, disable=not gpc.is_pipeline_last_stage()) as tqb:
            for eval_step, batch in enumerate(tqb, start=1):
                input_dict = batch[0]
                label = batch[1]
                with torch.no_grad():
                    input_dict['input_ids'] = self.generate(input_dict['input_ids'],
                                                            max_length=self.trainer_args.eval_max_length,
                                                            stop_tokens=self.trainer_args.eval_stop_tokens,
                                                            use_cache=self.trainer_args.eval_use_cache)
                    if gpc.is_pipeline_last_stage() and self.compute_metrics is not None:
                        self.compute_metrics((input_dict, label), epoch, step)
                tqb.set_postfix({'evaluating': f"{eval_step}/{len(self.eval_dataloader)}"})
            torch.cuda.empty_cache()