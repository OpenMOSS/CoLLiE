import sys
sys.path.append("../..")
import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, GenerationConfig
from collie import Trainer, EvaluatorForPerplexity, LlamaForCausalLM, CollieConfig, PPLMetric, AccuracyMetric, DecodeMetric, CollieDatasetForTraining, CollieDatasetForGeneration, \
    LossMonitor, TGSMonitor, MemoryMonitor, EvalMonitor, GradioProvider, EvaluatorForGeneration, LRMonitor, BleuMetric, DashProvider
from collie.metrics.rouge import RougeMetric
config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf")
config.pp_size = 8
config.train_micro_batch_size = 1
config.eval_batch_size = 1
config.gradient_accumulation_steps = 128
config.eval_per_n_epochs = 1
config.train_epochs = 1000
config.ds_config = {
    "fp16": {
        "enabled": True
    },
    "monitor_config": {
        "enabled": True,
        "wandb": {
            "enabled": True,
            "team": "00index",
            "project": "collie",
            "group": "translation"
        }
    }
}
config.seed = 1024
model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf", config=config)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
tokenizer = LlamaTokenizer.from_pretrained(
    "decapoda-research/llama-7b-hf", add_eos_token=False)
# Prepare training dataset
train_dataset = [
    {
        "input": f"Translate from French to English. French: {sample['translation']['fr']} English: ",
        "output": f"{sample['translation']['en']}</s>"
    } for sample in load_dataset("iwslt2017", name="iwslt2017-fr-en", split="train[:8000]")
]
# Prepare perplexity evaluation dataset
ratio = 0.01
eval_dataset_ppl, train_dataset = train_dataset[:int(
    len(train_dataset) * ratio)], train_dataset[int(len(train_dataset) * ratio):]
# Prepare classification evaluation dataset
eval_dataset_bleu = [
    {
        "text": f"Translate from French to English. French: {sample['translation']['fr']} English: ",
        "target": " ".join(f"{sample['translation']['en']}</s>".split())
    } for sample in load_dataset("iwslt2017", name="iwslt2017-fr-en", split="test[:50]")
]
# Convert to CoLLie Dataset
traine_dataset = CollieDatasetForTraining(train_dataset,
                                          tokenizer=tokenizer)
eval_dataset_ppl = CollieDatasetForTraining(eval_dataset_ppl,
                                            tokenizer=tokenizer)
eval_dataset_bleu = CollieDatasetForGeneration(eval_dataset_bleu,
                                               tokenizer=tokenizer)
# Prepare Evaluator
evaluator_ppl = EvaluatorForPerplexity(
    model=model,
    config=config,
    dataset=eval_dataset_ppl,
    monitors=[
        EvalMonitor(config)
    ],
    metrics={
        "ppl": PPLMetric(gather_result=True)
    },
)
evaluator_bleu = EvaluatorForGeneration(
    model=model,
    config=config,
    dataset=eval_dataset_bleu,
    monitors=[
        EvalMonitor(config)
    ],
    metrics={
        # "bleu": BleuMetric(gather_result=True, ngram=1),
        "decode": DecodeMetric(),
        "rouge": RougeMetric()
    },
    generation_config=GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=100,
    ),
    skip_special_tokens=True
)
# Prepare Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    optimizer=optimizer,
    train_dataset=traine_dataset,
    monitors=[
        LossMonitor(config),
        TGSMonitor(config),
        MemoryMonitor(config),
        LRMonitor(config)
    ],
    data_provider=DashProvider(tokenizer, port=12888, stream=True,
                                 generation_config=GenerationConfig(
                                     eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.pad_token_id,
                                     max_new_tokens=100,
                                 )),
    evaluators=[evaluator_ppl, evaluator_bleu]
)
trainer.train()
# trainer.save_checkpoint(path="/mnt/petrelfs/zhangshuo/model/test_save_checkpoint", mode="model")
