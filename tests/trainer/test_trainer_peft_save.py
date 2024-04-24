import sys

from peft import LoraConfig, TaskType

sys.path.append("../../")
from collie import Trainer, EvaluatorForPerplexity, InternLM2ForCausalLM, CollieConfig, PPLMetric, AccuracyMetric, \
    CollieDatasetForTraining, CollieDatasetForClassification, \
    LossMonitor, TGSMonitor, MemoryMonitor, LRMonitor
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import torch

model_id = "/cpfs01/shared/public/lvkai/.cache/huggingface/hub/models--internlm--internlm2-7b/snapshots/3a7c99d08679adc9c93eeb283f415b0bcdd19fba/"
model_id = "/fs-computility/llm/shared/lvkai/.cache/huggingface/hub/models--internlm--internlm2-7b/snapshots/ea4312ff4a5175723a17fbbbd74fa27514dd87b5"
config = CollieConfig.from_pretrained(model_id, trust_remote_code=True)
config.dp_size = 1
config.tp_size = 1
config.pp_size = 4
config.pp_partition_method = "uniform"
config.train_micro_batch_size = 4
config.eval_batch_size = 4
config.gradient_accumulation_steps = 2
config.train_epochs = 1
# config.checkpointing = False
config.ds_config = {
    "bf16": {
        "enabled": True
    },
    # "monitor_config": {
    #     "enabled": True,
    #     "tag": f"test_grad_accumulation_gas{config.gradient_accumulation_steps}_dp{config.dp_size}_bs{config.train_micro_batch_size}",
    #     "csv_monitor": {
    #         "enabled": True,
    #         "output_path": "ds_logs",
    #         "team": "collie_exp",
    #         "project": "llama_alpaca",
    #         "group": f"collie_test",
    #     },
    #     # "wandb": {
    #     #     "enabled": True,
    #     #     "team": "collie_exp",
    #     #     "project": "collie_test",
    #     #     "group": f"trainer_grad_accumulation",
    #     # }
    # },
    "zero_optimization": {
        "stage": 0,
    }
}
# config.peft_config = LoraConfig(
#         r=8,
#         lora_alpha=32,
#         target_modules=[
#             'wqkv', 'wo', 'w1', 'w2', "w3"
#         ],
#         lora_dropout=0.1,
#         task_type=TaskType.CAUSAL_LM,
#     )
config.seed = 1024
# setup_distribution(config)
model = InternLM2ForCausalLM.from_pretrained(model_id, config=config)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Prepare training dataset
train_dataset = [
                    {
                        "input": f"Comment: {sample['text']}. The sentiment of this comment is: ",
                        "output": "positive." if sample["label"] else "negative."
                    } for sample in load_dataset("/fs-computility/llm/shared/lvkai/.cache/huggingface/datasets/imdb/plain_text/0.0.0/e6281661ce1c48d982bc483cf8a173c1bbeb5d31", split="train")
                ][:400]

# Convert to CoLLie Dataset
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-7b", trust_remote_code=True, add_eos_token=True)
trainer_dataset = CollieDatasetForTraining(train_dataset,
                                           tokenizer=tokenizer)

total_step = len(trainer_dataset) * config.train_epochs // (config.train_micro_batch_size * config.dp_size * config.gradient_accumulation_steps)

print(f"Total step: {total_step}, total epoch: {config.train_epochs}, dataset size: {len(trainer_dataset)}")

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=10,
    num_training_steps=total_step
)
# Prepare Trainer
trainer = Trainer(
    model=model,
    lr_scheduler=lr_scheduler,
    config=config,
    optimizer=optimizer,
    train_dataset=trainer_dataset,
)
trainer.train()
# trainer.save_peft("./test_peft_save")
# trainer.save_checkpoint(path="/mnt/petrelfs/zhangshuo/model/test_save_checkpoint", mode="model")
