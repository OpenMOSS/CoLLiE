import sys
sys.path.append("../..")
import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, GenerationConfig
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from collie import Trainer, PerplexityEvaluator, LlamaForCausalLM, CollieConfig, PPLMetric, AccuracyMetric, DecodeMetric, CollieDatasetForTraining, CollieDatasetForGeneration, \
    LossMonitor, TGSMonitor, MemoryMonitor, EvalMonitor, GradioProvider, Evaluator, LRMonitor, BleuMetric, DashProvider, env
config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf")
config.pp_size = 2
config.tp_size = 4
config.dp_size = 1
config.train_micro_batch_size = 1
config.eval_batch_size = 1
config.gradient_accumulation_steps = 1
config.eval_per_n_epochs = 1
config.train_epochs = 10
config.checkpointing = False
config.ds_config = {
    "fp16": {
        "enabled": True
    },
    # "monitor_config": {
    #     "enabled": True,
    #     "wandb": {
    #         "enabled": True,
    #         "team": "00index",
    #         "project": "collie",
    #         "group": "further_pretrain_llama"
    #     }
    # },
    # "zero_optimization": {
    #     "stage": 1,
    # }
}
# 合并词表
llama_tokenizer = LlamaTokenizer.from_pretrained("/mnt/petrelfs/zhangshuo/model/llama-7b-hf")
# chinese_sp_model = spm.SentencePieceProcessor()
# chinese_sp_model.Load("./chinese_sp.model")
# llama_spm = sp_pb2_model.ModelProto()
# llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
# chinese_spm = sp_pb2_model.ModelProto()
# chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())
# llama_spm_tokens_set=set(p.piece for p in llama_spm.pieces)
# for p in chinese_spm.pieces:
#     piece = p.piece
#     if piece not in llama_spm_tokens_set:
#         new_p = sp_pb2_model.ModelProto().SentencePiece()
#         new_p.piece = piece
#         new_p.score = 0
#         llama_spm.pieces.append(new_p)
# llama_tokenizer.sp_model.LoadFromSerializedProto(llama_spm.SerializeToString())
# 准备模型并调整 embedding 层大小，设置只训练 embedding 层
model = LlamaForCausalLM.from_pretrained("/mnt/petrelfs/zhangshuo/model/llama-7b-hf", config=config)
# model.resize_token_embeddings(len(llama_tokenizer) + 7)
# for p in model.parameters():
#     p.requires_grad = False
# model.get_input_embedding().weight.requires_grad = True
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
# 由于 llama 中 embedding 与 lm_head 共享权重，因此跳过这一步
# model.get_lm_head().weight.requires_grad = True
test_dataset = CollieDatasetForTraining(
    dataset=[
        {
            "text": "测试"
        }
    ], tokenizer=llama_tokenizer
)
trainer = Trainer(
    model=model,
    config=config,
    train_dataset=test_dataset,
    optimizer=optimizer
)
trainer.train()
# print(trainer.engine.module.embed_tokens.weight.shape)