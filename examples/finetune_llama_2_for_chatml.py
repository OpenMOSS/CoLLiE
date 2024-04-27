# 这段脚本将展示如何使用 CoLLiE 将 Llama-2-7b 模型进行微调为支持 ChatML 聊天模板的对话模型。
# 这段脚本使用 OpenChat 提供的 UltraChat 数据集。该数据集是由 THUNLP 构建的多轮对话数据集。
# OpenChat 将原始 UltraChat 数据集整理为 ShareGPT 格式，以便于使用。
import datasets
import torch
from collie import (
    CollieConfig,
    CollieDatasetForTemplatedMultiTurnChat,
    EvalMonitor,
    EvaluatorForPerplexity,
    LlamaForCausalLM,
    LossMonitor,
    LRMonitor,
    MemoryMonitor,
    PPLMetric,
    TGSMonitor,
    Trainer,
)
from collie.data.template_utils import prepare_chatml_messages
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

model_path = "meta-llama/Llama-2-7b-hf"

config = CollieConfig.from_pretrained(model_path)
config.pp_size = 8
config.pp_partition_method = "uniform"
config.train_micro_batch_size = 1
config.eval_batch_size = 1
config.gradient_accumulation_steps = 4
config.eval_per_n_epochs = 1
config.train_epochs = 2
config.use_flash = False
config.ds_config = {
    "bf16": {"enabled": True},
}
config.seed = 196705814

# Prepare model
model = LlamaForCausalLM.from_pretrained(model_path, config=config)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare dataset
def map_fn(e):
    """
    本函数将 ShareGPT 格式的数据集转化为 CoLLiE 支持的格式。

    e["conversations"] 的格式为：
    [
        {"from": "gpt", "value": "Hello, how are you?"},
        {"from": "human", "value": "I am fine, thank you!"},
        ...
    ]

    转化后的格式为：
    [
        {"role": "assistant", "content": "Hello, how are you?"},
        {"role": "user", "content": "I am fine, thank you!"},
        ...
    ]

    由于 UltraChat 数据集不包含 System 信息，这里没有处理 "role": "system" 的情况。
    """
    return {
        "conversations": [
            {
                "role": "assistant" if x["from"] == "gpt" else "user",
                "content": x["value"],
            }
            for x in e["conversations"]
        ],
    }


dataset = (
    datasets.load_dataset("openchat/ultrachat-sharegpt")["train"]
    .shuffle(seed=config.seed)
    .select(list(range(1152)))
    .map(map_fn)
    .train_test_split(test_size=128)
)


train_dataset = list(dataset["train"])  # 1024
eval_dataset = list(dataset["test"])  # 128

# Convert to CoLLie Dataset
train_dataset = CollieDatasetForTemplatedMultiTurnChat(  # 使用这个类来处理多轮对话数据集。它会为原始数据应用聊天模板，并将非模型输入的部分的 label 设置为 -100
    train_dataset,
    tokenizer=tokenizer,
    template_fn=prepare_chatml_messages,  # 我们使用 ChatML 模板来处理对话数据。CoLLiE 还支持其他很多种模板，比如 MOSS。你可以 从 collie.data.template_utils 中导入其他模板函数。
    shuffle=True,
    text_field="conversations",
    seed=config.seed,
)
eval_dataset = CollieDatasetForTemplatedMultiTurnChat(
    eval_dataset,
    tokenizer=tokenizer,
    template_fn=prepare_chatml_messages,
    text_field="conversations",
)

total_step = len(train_dataset) * config.train_epochs // config.train_micro_batch_size
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_step * 0.03),
    num_training_steps=total_step
)

# Prepare Evaluator
evaluator_ppl = EvaluatorForPerplexity(
    model=model,
    config=config,
    dataset=eval_dataset,
    monitors=[EvalMonitor(config)],
    metrics={"ppl": PPLMetric(gather_result=True)},
)

# Prepare Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    lr_scheduler=lr_scheduler,
    config=config,
    optimizer=optimizer,
    train_dataset=train_dataset,
    monitors=[
        LossMonitor(config),
        TGSMonitor(config),
        MemoryMonitor(config),
        LRMonitor(config),
    ],
    evaluators=[evaluator_ppl],
)

trainer.train()
