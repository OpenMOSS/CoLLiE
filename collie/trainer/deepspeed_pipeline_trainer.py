import os
import sys
import json
from itertools import cycle

import tqdm
import torch
from torch.utils.data import DistributedSampler

try:
    import deepspeed
    from deepspeed import comm as dist
    from deepspeed.accelerator import get_accelerator
except:
    pass

from collie.log import logger
from collie.models.checkpoint_engine import get_checkpoint_engine
from .inplace_zero_trainer import InplaceZeroTrainer
from .utils import WandbLogger, sample_top_p

class PipelineTrainer(InplaceZeroTrainer):
    def __init__(
        self,
        model,
        collie_args,
        data_collator,
        train_dataset,
        eval_dataset,
        tokenizer,
        compute_metrics,
        optimizer=None,
        lr_scheduler=None,
    ):
        """
        Trainer for training DeepSpeed Pipeline

        :param model: deepspeed.pipe.PipelineModule
        """
        self.collie_args = collie_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.wandb = WandbLogger(collie_args)
        self.metrics = {}
        
        if 'deepspeed' not in sys.modules:
            raise ModuleNotFoundError(
                "Detected DeepSpeed is not installed. See https://github.com/microsoft/DeepSpeed")
        
        self.engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            config=collie_args.deepspeed,
            model=model,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
        )

        # get train_dataloader and eval_dataloader
        if isinstance(data_collator, dict):
            assert 'train' in data_collator and 'eval' in data_collator, "data_collator should be a dict with keys 'train' and 'eval'."
            self.train_data_collator = data_collator['train']
            self.eval_data_collator = data_collator['eval']
        else:
            self.train_data_collator = self.eval_data_collator = data_collator
        self.train_dataloader = self.get_train_dataloader()

        if isinstance(self.eval_dataset, dict):
            self.eval_dataloader = {}
            for prefix in self.eval_dataset.keys():
                self.eval_dataloader[prefix] = self.get_eval_dataloader(self.eval_dataset[prefix])
        else:
            self.eval_dataloader = self.get_eval_dataloader()

        self.allow_print = self.is_first_stage()

        if self.collie_args.per_device_train_batch_size != self.engine.train_micro_batch_size_per_gpu():
            if self.allow_print:
                logger.warning(
                    "`per_device_train_batch_size` {} is not equal to "
                    "`train_micro_batch_size_per_gpu` {}, we will set "
                    "later as batch size.".format(
                        self.collie_args.per_device_train_batch_size,
                        self.engine.train_micro_batch_size_per_gpu()
                    )
                )
            self.collie_args.per_device_train_batch_size = self.engine.train_micro_batch_size_per_gpu()

        self.num_steps_per_epoch = len(self.train_dataloader)
        self.global_step = 1
        self.n_steps = self.num_steps_per_epoch * self.collie_args.num_train_epochs

        get_accelerator().empty_cache()
        
    def train(self):
        for epoch in range(self.collie_args.num_train_epochs):
            logger.info(f"***** Running Training *****")
            logger.info(f"  Num examples: {len(self.train_dataset)}")
            logger.info(f"  Num Epochs: {self.collie_args.num_train_epochs}")
            logger.info(f"  Batch Size: {self.collie_args.per_device_train_batch_size}")
            if self.allow_print:
                self.wandb.log({'train/epoch': epoch}, step=self.global_step)

            with tqdm.tqdm(self.train_dataloader,
                           disable=not self.allow_print) as tqb:
                for step, batch in enumerate(tqb, start=1):
                    self.engine.reset_activation_shape()
                    loss = self.engine.train_batch(
                        cycle([(
                            (batch["input_ids"], torch.tensor(0)), batch["label"]
                        )])
                    )

                    self.global_step = self.num_steps_per_epoch * epoch + step

                    tqb.set_postfix({'loss': loss.item()})
                    if self.lr_scheduler:
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.collie_args.learning_rate
                    if self.allow_print:
                        self.wandb.log(
                            {
                                'train/loss': loss.item(),
                                'train/learning_rate': lr,
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
            dataloader: torch.utils.data.DataLoader,
            eval_prefix: str
    ):
        r"""
        Shared by both eval(validation) and predict(test).
        This method will be called by the trainer to evaluate the model.
        """
        logger.info(f"***** Running {eval_prefix} *****")
        logger.info(f"  Num examples: {len(dataset)}")
        logger.info(f"  Batch size: {self.collie_args.per_device_eval_batch_size}")
        self.engine.total_loss = None

        with tqdm.tqdm(dataloader, disable=not self.allow_print) as tqb:
            all_preds = None
            for batch in tqb:
                pred = self.eval_step(batch)
                all_preds = pred if all_preds is None else all_preds + pred

            result = self.compute_metrics(all_preds, dataset)
            result = {f"{eval_prefix}/{k}": v for k, v in result.items()}
            prefix_metric_for_best_model = f'{eval_prefix}/{self.collie_args.metric_for_best_model}'
            result_value = result[prefix_metric_for_best_model]

            if self.allow_print:
                logger.info(f'epoch: {epoch}, step: {step}, {self.collie_args.metric_for_best_model}: {result_value}')
                self.wandb.log(result, step=step)

                if self.is_better(result, prefix_metric_for_best_model):
                    self.wandb.set_summary(f'{eval_prefix}/best_{self.collie_args.metric_for_best_model}', result_value)
                    self.wandb.set_summary(f'{eval_prefix}/best_epoch', epoch)
                    self.wandb.set_summary(f'{eval_prefix}/best_step', step)
                    self.metrics[prefix_metric_for_best_model] = result_value

        self.engine.total_loss = None

    def eval_step(self, batch):
        # sample
        # TODO: process of generate can be replaced on gpu?
        eos_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            pad_token_id = self.tokenizer.eos_token_id
        eos_token_id_tensor = torch.LongTensor(eos_token_id).cuda()

        sentence_ids = batch["input_ids"].cuda()
        batch_size = sentence_ids.shape[0]
        unfinished_sequences = sentence_ids.new(batch_size).fill_(1)
    
        past_key, past_value = None, None
        use_cache = torch.tensor(1)

        self.comm_kv_shape = False
        while True:
            # set input
            if past_key is not None:
                assert past_value is not None
                input_ids = sentence_ids[:, -1].unsqueeze(-1)
            else:
                input_ids = sentence_ids

            # since shape of past_key and past_value changes every lopp
            # we need to reset some attributes of pipeline
            self.engine.reset_activation_shape()
            self.engine.total_loss = None
            
            # The second label is unused.
            if past_key is not None:
                assert past_value is not None
                output = self.engine.eval_batch(
                    cycle([
                        ((input_ids, use_cache, past_key, past_value), batch["label"])
                    ]), compute_loss=False, reduce_output=None,
                )
            else:
                output = self.engine.eval_batch(
                    cycle([((input_ids, use_cache), batch["label"])]), compute_loss=False, reduce_output=None,
                )
            # output:
            # last_stage: [[logits, past_key, past_value]]
            # other stages: None
            dist.barrier()

            if self.is_last_stage():
                # logits: bsz, seq_len, vocab_size
                # past_key: layers, num_heads, seqlen, n_embed/head
                # past_value: layers, num_heads, seqlen, n_embed/head
                logits, past_key, past_value = output[0]
                next_token_logits = logits[:, -1, :].float()

                probs = torch.softmax(next_token_logits.float() / self.collie_args.temperature, dim=-1)
                next_tokens = sample_top_p(probs, self.collie_args.top_p)
                # [batch_size, 1] -> [batch_size]
                next_tokens = next_tokens.reshape([-1])

                # concat sentence
                next_tokens = next_tokens * unfinished_sequences + \
                    pad_token_id * (1 - unfinished_sequences)

                # update generated ids, model inputs, and length for next step
                sentence_ids = torch.cat([sentence_ids, next_tokens[:, None]], dim=-1)

                # if eos_token was found in one sentence, set sentence to finished
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.repeat(eos_token_id_tensor.shape[0], 1) \
                            .ne(eos_token_id_tensor.unsqueeze(1)) \
                            .prod(dim=0)
                )
            else:
                sentence_ids = torch.cat(
                    [sentence_ids, torch.zeros(batch_size, 1).to(sentence_ids)],
                    dim=-1)
                
            past_key, past_value = self._broadcast_past_kv(past_key, past_value)

            dist.broadcast(
                unfinished_sequences, src=self.num_stages - 1
            )
            dist.broadcast(
                sentence_ids, src=self.num_stages - 1
            )

            dist.barrier()

            if unfinished_sequences.max() == 0 or sentence_ids.shape[-1] >= self.collie_args.max_new_tokens:
                break

        pred_texts = self.tokenizer.batch_decode(sentence_ids.tolist())
        get_accelerator().empty_cache()
        return pred_texts
    
    def save(self, save_dir, protocol="file"):
        """

        :param save_dir:
        :protocol: ['file', 's3']
        """
        logger.info("Saving to", save_dir)
        ckpt_engine = get_checkpoint_engine(protocol)
        module = self.engine.module
        # PipelineModule.save_state_dict
        ckpt_engine.makedirs(save_dir, exist_ok=True)
        # pipeline information
        pipepine_info = {
            "num_stages": self.num_stages,
            "n_layers": module._num_layers,
        }
        if self.is_first_stage():
            ckpt_engine.save_json(
                pipepine_info, os.path.join(save_dir, "pipeline.json")
            )
            # model config
            ckpt_engine.save_json(
                module._collie_config.to_diff_dict(),
                os.path.join(save_dir, "config.json")
            )

        module.save_state_dict(save_dir, ckpt_engine)
    
    def get_train_dataloader(self):
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        # PipelineMoudle._build_data_iter
        sampler = DistributedSampler(
            self.train_dataset, num_replicas=self.engine.dp_world_size,
            rank=self.engine.mpu.get_data_parallel_rank(), shuffle=False
        )
        # Build a loader and make it repeating.
        pipe_dataloader = self.engine.deepspeed_io(
            self.train_dataset, data_sampler=sampler,
            collate_fn=self.train_data_collator
        )
        return pipe_dataloader

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

        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.collie_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.collie_args.dataloader_drop_last,
            num_workers=self.collie_args.dataloader_num_workers,
            pin_memory=self.collie_args.dataloader_pin_memory,
        )

    def is_first_stage(self):
        return self.engine.is_first_stage()
    
    def is_last_stage(self):
        return self.engine.is_last_stage()
    
    @property
    def num_stages(self):
        return self.engine.num_stages
    
    @property
    def stage_id(self):
        return self.engine.stage_id
    
    def _broadcast_past_kv(self, past_key, past_value):
        if not self.comm_kv_shape:
            # broadcast shape
            if self.is_last_stage():
                kv_shape = torch.tensor(past_key.shape).cuda()
            else:
                kv_shape = torch.zeros(5).long().cuda()
            dist.broadcast(kv_shape, src=self.num_stages - 1)
            self.comm_kv_shape = True
            if not self.is_last_stage():
                # TODO float32 or float16?
                model_dtype, grad_dtype = self.engine.get_data_types()
                # past_key always float32
                past_key = torch.zeros(kv_shape.tolist()).cuda()
                # past_value can be float16
                past_value = torch.zeros(kv_shape.tolist(), dtype=model_dtype).cuda()
        else:
            if not self.is_last_stage():
                l, b, h, s, e = past_key.shape
                past_key = torch.concat(
                    [past_key, torch.zeros(l, b, h, 1, e).to(past_key)], dim=-2
                )
                past_value = torch.concat(
                    [past_value, torch.zeros(l, b, h, 1, e).to(past_value)], dim=-2
                )

        dist.broadcast(past_key, src=self.num_stages - 1)
        dist.broadcast(past_value, src=self.num_stages - 1)

        return past_key, past_value