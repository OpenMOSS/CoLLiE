import sys
sys.path.append("..")

from .utils import inplace_grad, GPTLMLoss, sample_top_p

import tqdm
import torch
from itertools import cycle
from functools import partial

from typing import List, Tuple, Dict, Union, Iterable, Optional, Any

from dataclasses import dataclass, field

try:
    import colossalai
    from colossalai.core import global_context as gpc
    from colossalai.context.parallel_mode import ParallelMode
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Detected Colossal-AI is not installed. See https://github.com/hpcaitech/ColossalAI")
    
class GenerativeDataloader:
    def __init__(self,
                 sentences: Optional[List[str]] = None,
                 input_ids: Optional[torch.Tensor] = None,
                 max_length: Optional[int] = None,
                 stop_tokens: Optional[List[int]] = None,
                 tokenizer: Optional[Any] = None,
                 device: torch.DeviceObjType = torch.device("cpu")
                 ) -> None:
        self.input_ids = input_ids.to(device)
        self.max_length = max_length
        if sentences is not None:
            assert tokenizer is not None, "tokenizer should not be None when sentences is not None"
            tokens = [tokenizer(sentence)["input_ids"] for sentence in sentences]
            tokens_max_length = max([token.shape[0] for token in tokens])
            self.input_ids = torch.full((len(tokens), tokens_max_length), 0, dtype=torch.long).to(device)
            for i, token in enumerate(tokens):
                if len(token) > self.max_length:
                    token = token[:self.max_length]
                    self.input_ids[i, :token.shape[0]] = token
        orginal_shape = self.input_ids.shape
        self.input_ids = torch.flatten(self.input_ids)
        for i in range(self.input_ids.shape[0]):
            if self.input_ids[i] in stop_tokens:
                self.input_ids[i] = 0
        self.input_ids = self.input_ids.view(orginal_shape)
        self.stop_tokens = stop_tokens
        self.tokenizer = tokenizer
        self.current_pos = 1
        self.stop = False
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_pos >= self.max_length and self.max_length > 0 or self.stop:
            raise StopIteration
        data = {
            "input_ids": self.input_ids[:self.current_pos]
        }, self.input_ids
        self.current_pos += 1
        return data
        
    def __len__(self):
        if self.max_length is None:
            return sys.maxsize
        else:
            return self.max_length
        
    def __getitem__(self, index):
        return next(self)
    
    def append(self, token: torch.Tensor):
        for i in torch.flatten(token).tolist():
            if i in self.stop_tokens:
                self.stop = True
        assert token.shape == (self.input_ids.shape[0], 1), "token should be a 2D tensor with shape (batch_size, 1), but got {}".format(token.shape)
        if self.current_pos >= self.input_ids.shape[1]:
            self.input_ids = torch.cat([self.input_ids, torch.zeros_like(token)], dim=1)
        self.input_ids[:, self.current_pos] = torch.where(self.input_ids[:, self.current_pos] == 0, token[:, 0], self.input_ids[:, self.current_pos])
        
    def get_input_ids(self):
        return self.input_ids
    
    def get_sentences(self):
        return [self.tokenizer.decode(token.tolist()) for token in self.input_ids]
        
    
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
        default=1024,
        metadata={"help": "The maximum length of generated text when evaluating."
                  "If it is set to -1, the evaluation will run until the stop tokens are generated."})
    eval_stop_tokens: List[int] = field(
        default_factory=partial(list, [2]),
        metadata={"help": "The stop tokens when evaluating."})
    eval_top_p: float = field(
        default=0.9,
        metadata={"help": "The top_p when evaluating."})
    eval_temperature: float = field(
        default=1.0,
        metadata={"help": "The temperature when evaluating."})
    

class ColossalaiTrainer:
    def __init__(self,
                 model,
                 tokenizer,
                 train_dataloader,
                 eval_dataloader,
                 compute_metrics = None,
                 trainer_args: TrainerArgs = TrainerArgs()) -> None:
        self.grad_func = inplace_grad(model, lr=trainer_args.learning_rate)
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.register_hook(self.grad_func)
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics
        self.trainer_args = trainer_args
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=trainer_args.learning_rate)
        self.engine, self.train_dataloader, self.eval_dataloader, _ = colossalai.initialize(
            model=self.model,
            train_dataloader=self.train_dataloader,
            test_dataloader=self.eval_dataloader,
            optimizer=self.optimizer,
            criterion=GPTLMLoss()
        )
        
    def train(self):
        def train_loop(epoch: int = 0):
            dataloader = iter(self.train_dataloader)
            with tqdm.tqdm(self.train_dataloader, disable=not gpc.is_pipeline_last_stage()) as tqb:
                for step, batch in enumerate(tqb, start=1):
                    self.engine.train()
                    self.engine.zero_grad()
                    hidden_state, label, loss = self.engine.execute_schedule(
                        cycle([batch]),
                        forward_only=False,
                        return_loss=True,
                        return_output_label=False,
                    )
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
                 batch: Union[Dict[str, torch.Tensor], GenerativeDataloader],
                 max_length: int = 1024,
                 stop_tokens: List[int] = [2],):
        if isinstance(batch, dict):
            dataloader = GenerativeDataloader(
                input_ids=batch["input_ids"],
                max_length=max_length,
                stop_tokens=stop_tokens,
                device=torch.device(f"cuda:{gpc.get_local_rank(ParallelMode.PIPELINE)}"),
                )
        else:
            dataloader = batch
        with tqdm.tqdm(range(max_length), disable=not gpc.is_pipeline_last_stage()) as tqb:
            for token in tqb:
                try:
                    hidden_state, label, _ = self.engine.execute_schedule(
                        dataloader,
                        forward_only=True,
                        return_loss=False,
                        return_output_label=True,
                    )
                    next_tokens = torch.zeros(dataloader.input_ids.shape[0], 1, dtype=torch.long).to(dataloader.input_ids.device)
                    if gpc.is_pipeline_last_stage():
                        next_tokens = torch.argmax(hidden_state[:, -1, :], dim=-1)
                        next_tokens = torch.unsqueeze(next_tokens, dim=-1)
                    torch.distributed.broadcast(next_tokens, src=gpc.get_world_size(ParallelMode.PIPELINE) - 1)
                    dataloader.append(next_tokens)
                    tqb.set_postfix({'generating': f"{token}/{max_length}"})
                except StopIteration:
                    break
        return dataloader.get_input_ids()
    
    def eval(self, epoch, step):
        with tqdm.tqdm(self.eval_dataloader, disable=not gpc.is_pipeline_last_stage()) as tqb:
            for step, batch in enumerate(tqb, start=1):
                batch, label = batch
                with torch.no_grad():
                    generated_batch = self.generate(batch,
                                                    max_length=self.trainer_args.eval_max_length,
                                                    stop_tokens=self.trainer_args.eval_stop_tokens)
                    if gpc.is_pipeline_last_stage() and self.compute_metrics is not None:
                        self.compute_metrics(batch, generated_batch, epoch, step)
                tqb.set_postfix({'evaluating': f"{step}/{len(self.eval_dataloader)}"})
                    