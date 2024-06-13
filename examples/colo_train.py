from typing import Callable, List, Union, Optional
from functools import partial
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.optim import Adam
import torch.nn as nn
from torch.optim import Optimizer
# from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers import AutoTokenizer
import datasets
import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam

import datasets
from transformers import AutoTokenizer, PreTrainedTokenizer

from colossalai.booster.plugin.dp_plugin_base import DPPluginBase

class GLUEDataBuilder:
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        plugin: DPPluginBase,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.plugin = plugin

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        if not getattr(self.tokenizer, "pad_token", None):
            self.tokenizer.pad_token = self.tokenizer._eos_token
        self.setup()

    def setup(self):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return self.plugin.prepare_dataloader(
            self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return self.plugin.prepare_dataloader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [
                self.plugin.prepare_dataloader(self.dataset[x], batch_size=self.eval_batch_size)
                for x in self.eval_splits
            ]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return self.plugin.prepare_dataloader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [
                self.plugin.prepare_dataloader(self.dataset[x], batch_size=self.eval_batch_size)
                for x in self.eval_splits
            ]

    def convert_to_features(self, example_batch):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding="max_length", truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features

colossalai.launch_from_torch(seed=42)

coordinator = DistCoordinator()

plugin = HybridParallelPlugin(
    tp_size=1,
    pp_size=1,
    num_microbatches=None,
    microbatch_size=1,
    enable_all_optimization=True,
    zero_stage=1,
    precision="fp16",
    initial_scale=1,
)

# Launch ColossalAI
# import time
# time.sleep(10)
# print("dist inited")

NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1

# def tokenize_batch(batch, tokenizer: Optional[AutoTokenizer] = None, max_length: int = 2048):
#     texts = [sample["sentence1"] + sample["sentence2"] for sample in batch]
#     data = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
#     data = {k: v.cuda() for k, v in data.items()}
#     data["labels"] = torch.tensor([0])
#     return data

tokenizer = AutoTokenizer.from_pretrained("/fs-computility/llm/shared/hongjiawei/math_project/url_recall/Atom-7B")
# dataset = datasets.load_dataset("glue", "mrpc")
# train_dataloader = plugin.prepare_dataloader(
#     dataset["train"],
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
#     collate_fn=partial(tokenize_batch, tokenizer=tokenizer, max_length=512),
# )
data_builder = GLUEDataBuilder(
        "/fs-computility/llm/shared/hongjiawei/math_project/url_recall/Atom-7B", plugin, "mrpc", train_batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE
    )
train_dataloader = data_builder.train_dataloader()
test_dataloader = data_builder.test_dataloader()
cfg = AutoConfig.from_pretrained("/fs-computility/llm/shared/hongjiawei/math_project/url_recall/Atom-7B", num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained("/fs-computility/llm/shared/hongjiawei/math_project/url_recall/Atom-7B", config=cfg).cuda()

lr = LEARNING_RATE * coordinator.world_size
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr, eps=1e-8)

# lr scheduler
total_steps = len(train_dataloader) * NUM_EPOCHS
num_warmup_steps = int(WARMUP_FRACTION * total_steps)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_steps,
)
def move_to_cuda(batch):
    return {k: v.cuda() for k, v in batch.items()}

def _criterion(outputs, inputs):
    return outputs.loss

booster = Booster(plugin=plugin)

model, optimizer, _criterion, _, lr_scheduler = booster.boost(
    model, optimizer, criterion=_criterion, lr_scheduler=lr_scheduler
)

def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    _criterion: Callable,
    lr_scheduler,
    train_dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    print_flag = (not use_pipeline and coordinator.is_master()) or (use_pipeline and is_pp_last_stage)
    total_step = len(train_dataloader)

    model.train()
    optimizer.zero_grad()
    train_dataloader_iter = iter(train_dataloader)
    with tqdm(
        range(total_step),
        desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]",
        disable=not print_flag,
    ) as pbar:
        # Forward pass
        for _ in pbar:
            if use_pipeline:
                outputs = booster.execute_pipeline(
                    train_dataloader_iter, model, _criterion, optimizer, return_loss=True
                )
                # Backward and optimize
                if is_pp_last_stage:
                    loss = outputs["loss"]
                    pbar.set_postfix({"loss": loss.item()})
            else:
                data = next(train_dataloader_iter)
                data = move_to_cuda(data)
                outputs = model(**data)
                loss = _criterion(outputs, None)
                # Backward
                booster.backward(loss, optimizer)
                pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()


for epoch in range(NUM_EPOCHS):
    train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, train_dataloader, booster, coordinator)