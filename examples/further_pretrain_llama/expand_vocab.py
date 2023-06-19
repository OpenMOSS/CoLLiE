import sys
sys.path.append("../..")
from collie import Trainer, EvaluatorForPerplexity, LlamaForCausalLM, CollieConfig, PPLMetric, CollieDatasetForTraining, \
    LossMonitor, TGSMonitor, MemoryMonitor, EvalMonitor, GradioProvider, LRMonitor, CheckpointCallback
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
from transformers import LlamaTokenizer, GenerationConfig
from datasets import load_dataset
import torch
# 准备配置，这里是 collie 训练的核心
config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf")
config.pp_size = 2
config.tp_size = 4
config.dp_size = 1
config.train_micro_batch_size = 8
config.eval_batch_size = 8
config.gradient_accumulation_steps = 16
config.eval_per_n_epochs = 1
config.train_epochs = 5000
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
            "group": "further_pretrain_llama"
        }
    },
    # "zero_optimization": {
    #     "stage": 1,
    # }
}
# 合并词表
llama_tokenizer = LlamaTokenizer.from_pretrained(
    "/mnt/petrelfs/zhangshuo/model/llama-7b-hf")
chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model.Load("./chinese_sp.model")
llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
chinese_spm = sp_pb2_model.ModelProto()
chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())
llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
for p in chinese_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
llama_tokenizer.sp_model.LoadFromSerializedProto(llama_spm.SerializeToString())
# 准备中英文混合预训练数据集
dataset = CollieDatasetForTraining(
    dataset=[
        {
            "text": sample["sentence"] + "</s>"
        } for sample in load_dataset("facebook/flores", name="zho_Hans", split="devtest")
    ] + [
        {
            "text": sample["sentence"] + "</s>"
        } for sample in load_dataset("facebook/flores", name="zho_Hans", split="dev")
    ] + [
        {
            "text": sample["sentence"] + "</s>"
        } for sample in load_dataset("facebook/flores", name="eng_Latn", split="devtest")
    ], tokenizer=llama_tokenizer, shuffle=True, seed=42
)
ratio = 0.01
eval_dataset, train_dataset = dataset[:int(
    len(dataset) * ratio)], dataset[int(len(dataset) * ratio):]
# 准备模型并调整 embedding 层大小，设置只训练 embedding 和 lm_head 层，加速收敛
model = LlamaForCausalLM.from_pretrained(
    "/mnt/petrelfs/zhangshuo/model/llama-7b-hf", config=config)
model.resize_token_embeddings(len(llama_tokenizer) + 7)  # 取个整
for p in model.parameters():
    p.requires_grad = False
# 因为 embedding 和 lm_head 在 pipeline 的情况下被分割到了不同的进程，所以要判断一下自己是否有 embedding 层
if model.get_input_embedding()[1] is not None:
    model.get_input_embedding()[1].weight.requires_grad = True
if model.get_lm_head()[1] is not None:
    model.get_lm_head()[1].weight.requires_grad = True
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.train_epochs * len(train_dataset), eta_min=0)
# 准备验证器，指标为PPL
evaluator = EvaluatorForPerplexity(
    model=model,
    config=config,
    dataset=eval_dataset,
    metrics={
        "ppl": PPLMetric(gather_result=True)
    }
)
# 准备训练器
trainer = Trainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    tokenizer=llama_tokenizer,
    monitors=[LossMonitor(config), TGSMonitor(config), MemoryMonitor(
        config), EvalMonitor(config), LRMonitor(config)],
    # 打开一个交互界面，方便随时 human eval
    data_provider=GradioProvider(generation_config=GenerationConfig(
        eos_token_id=llama_tokenizer.eos_token_id, max_length=128),
        tokenizer=llama_tokenizer, port=12311),
    evaluators=[evaluator]
)

trainer.train()
