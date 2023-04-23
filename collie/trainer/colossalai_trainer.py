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
    from colossalai.zero.sharded_optim.low_level_optim import LowLevelZeroOptimizer
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Detected Colossal-AI is not installed. See https://github.com/hpcaitech/ColossalAI")

@dataclass
class TrainerArgs:
    epochs: int = field(
        default=10,
        metadata={"help": "Total number of training epochs to perform. "
                  "If it is set to -1, the training will run forever."
                  "If it is set to 0, the training will not run."})
    learning_rate: int = field(
        default=1e-3,
        metadata={"help": "Learning rate of training."})
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
    eval_use_cache : bool = field(
        default=True,
        metadata={"help": "Whether to use key/value cache when evaluating or generating."})

class ColossalaiTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 tokenizer,
                 train_dataloader,
                 eval_dataloader,
                 lr_scheduler = None,
                 compute_metrics = None,
                 trainer_args: TrainerArgs = TrainerArgs()) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.trainer_args = trainer_args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
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
            with tqdm.tqdm(self.train_dataloader, disable=not gpc.is_pipeline_last_stage() or gpc.get_local_rank(ParallelMode.TENSOR) != gpc.get_world_size(ParallelMode.TENSOR) - 1) as tqb:
                for step, batch in enumerate(tqb, start=1):
                    _, _, loss = self.engine.execute_schedule(
                        cycle([batch]),
                        forward_only=False,
                        return_loss=True,
                        return_output_label=False,
                    )
                    self.engine.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
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
        # flag vector to control early stop batch generate
        stop_generate_flag_vector = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        min_len = min([len(torch.nonzero(sample)) for sample in input_ids])
        if max_length > input_ids.shape[1]:
            input_ids = torch.nn.functional.pad(input_ids, (0, max_length - input_ids.shape[1]), value=0)
        cached_len = 0
        # generate loop:
        with tqdm.tqdm(range(min_len, max_length), disable=not gpc.is_pipeline_last_stage() or gpc.get_local_rank(ParallelMode.TENSOR) != gpc.get_world_size(ParallelMode.TENSOR) - 1) as tqb:
            for current_pos in tqb:
                active_batch_idx = torch.flatten(torch.argwhere(~stop_generate_flag_vector))
                try:
                    self.engine.eval()
                    hidden_states, _, _ = self.engine.execute_schedule(
                        cycle([({
                            "input_ids": input_ids[:, cached_len:current_pos] if use_cache else input_ids[:, :current_pos],
                            "use_cache": torch.ones(input_ids.shape[0], dtype=torch.bool) if use_cache else torch.zeros(input_ids.shape[0], dtype=torch.bool)
                        }, input_ids)]),
                        forward_only=True,
                        return_loss=False,
                        return_output_label=True,
                    )
                    torch.cuda.empty_cache()
                    cached_len = current_pos
                    next_tokens_list = [torch.full((input_ids.shape[0], 1), -1, dtype=torch.long, device=torch.device(f"cuda:{os.environ['LOCAL_RANK']}")) for _ in range(int(os.environ.get("WORLD_SIZE")))]
                    if gpc.is_pipeline_last_stage():
                        # top-p分布
                        scores = hidden_states[:,-1,:]
                        sorted_logits, sorted_indicies = torch.sort(scores, descending=True)
                        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                        sorted_indices_to_remove = cumulative_probs > self.trainer_args.eval_top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indicies, sorted_indices_to_remove)
                        scores = scores.masked_fill(indices_to_remove, -float('inf'))
                        # 温度采样
                        scores = torch.nn.functional.softmax(scores / self.trainer_args.eval_temperature, dim=-1)
                        # next_tokens_list[int(os.environ.get("RANK"))] = torch.argmax(hidden_states[:, -1, :], dim=-1)
                        
                        next_tokens_list[int(os.environ.get("RANK"))] = torch.multinomial(scores, num_samples=1).squeeze(1)
                        next_tokens_list[int(os.environ.get("RANK"))] = torch.unsqueeze(next_tokens_list[int(os.environ.get("RANK"))], dim=-1)
                    torch.distributed.all_gather(next_tokens_list, next_tokens_list[int(os.environ.get("RANK"))])
                    next_tokens = next_tokens_list[0]
                    for i in range(len(next_tokens_list)):
                        if next_tokens_list[i].sum() >= 0:
                            next_tokens = next_tokens_list[i]
                            break
                    next_tokens = next_tokens.to(input_ids.device)
                    input_ids[active_batch_idx, current_pos] = torch.where(
                        input_ids[active_batch_idx, current_pos] == 0,
                        next_tokens[active_batch_idx, 0],
                        input_ids[active_batch_idx, current_pos]
                    )
                    tqb.set_postfix({'generating': f"{current_pos}/{max_length}"})
                    # check if can stop generate using stop_generate_flag_vector
                    for stop_token in stop_tokens:
                        for i in range(len(stop_generate_flag_vector)):
                            if not stop_generate_flag_vector[i] and \
                                    stop_token in input_ids[i][:current_pos]:
                                stop_generate_flag_vector[i] = True
                    if all(stop_generate_flag_vector):
                        raise StopIteration
                    torch.cuda.empty_cache()
                except StopIteration:
                    break
        # clean caches for key & value
        try:
            _ = [block.clean_cache() for block in self.engine.model.blocks]
        except AttributeError:
            try:
                _ = [block.clean_cache() for block in self.engine.model.model.blocks]
            except AttributeError:
                pass
        return input_ids

    def eval(self, epoch=0, step=0):
        with tqdm.tqdm(self.eval_dataloader, disable=not gpc.is_pipeline_last_stage()) as tqb:
            for eval_step, batch in enumerate(tqb, start=1):
                input_dict = batch[0]
                label = batch[1]
                with torch.no_grad():
                    input_dict['input_ids'] = self.generate(
                        input_dict['input_ids'],
                        max_length=self.trainer_args.eval_max_length,
                        stop_tokens=self.trainer_args.eval_stop_tokens,
                        use_cache=self.trainer_args.eval_use_cache
                    )
                    torch.cuda.empty_cache()
                    if gpc.is_pipeline_last_stage() and gpc.get_local_rank(ParallelMode.TENSOR) == gpc.get_world_size(ParallelMode.TENSOR) - 1 and self.compute_metrics is not None:
                        self.compute_metrics((input_dict, label), epoch, step)
                tqb.set_postfix({'evaluating': f"{eval_step}/{len(self.eval_dataloader)}"})
            torch.cuda.empty_cache()
