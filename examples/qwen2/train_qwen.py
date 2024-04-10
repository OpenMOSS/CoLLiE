import json
import sys

# srun -p a800 --cpus-per-task=4 --mem-per-cpu=4G --gres=gpu:6 --job-name=qwen_test python train_qwen.py

sys.path.append('../../')
from collie import (
    CheckpointCallback, CollieConfig, CollieDatasetForTraining, EvalMonitor, EvaluatorForPerplexity,
    Qwen2ForCausalLM, QWenLMHeadModel, LRMonitor, LossMonitor, MemoryMonitor, PPLMetric, TGSMonitor, Trainer,
)
import torch
from transformers import AutoTokenizer, AddedToken
import math

BASE_MODEL_PATH = "/remote-home/share/models/Qwen1.5-7B"
MODEL_PATH = BASE_MODEL_PATH

config = CollieConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
config.dp_size = 4
config.tp_size = 2
config.pp_size = 1
config.train_micro_batch_size = 4
config.eval_batch_size = 8
config.gradient_accumulation_steps = 1
config.eval_per_n_steps = 500
config.train_epochs = 30
# config.checkpointing = False
config.ds_config = {
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
    },
    "monitor_config": {
        "enabled": True,
        "tag": f"test",
        # "tensorboard": {
        #     "enabled": True,
        #     "output_path": "./ds_tb_logs/",
        # },
        "csv_monitor": {
            "enabled": True,
            "output_path": "./ds_csv_logs",
        }
    },
}
config.seed = 1024

# print(dist.get_world_size())

assert "Qwen" in MODEL_PATH, "Please specify a qwen2 model path, e.g. 'Qwen/Qwen-7B'"
model = Qwen2ForCausalLM.from_pretrained(MODEL_PATH, config=config, trust_remote_code=True)

lr = 1e-5    # lr不能太小
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

warmup_T = 50	# 周期
lr_max = lr	# 最大值
lr_min = 1e-6	# 最小值
cos_T = 100

gamma = 0.99
# lambda是个系数，还要乘上base_lr
lambda_warmup = lambda step: (step * (lr_max - lr_min) / warmup_T + lr_min) / lr_max if  step < warmup_T else \
    gamma ** (step-warmup_T)
#     (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos((step-warmup_T)/(cos_T-warmup_T)*math.pi)))/0.1
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda_warmup)
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # <=====

# special_tokens_list = [
#     "<[user_intent]>", "<[user_config]>", "<[apis]>", "<[query]>", "<[plan]>", "<[think]>", "<[answer]>",
#     "[Plan]", "[Call]", "[Response]", "[NoCall]", "[Ask]", "[Answer]", "[ToFill]"
# ]
data_path = "/remote-home/share/storage/moss/honor_demo/simple_data.json"
with open(data_path) as fin:
    data_list = json.load(fin)
dataset = []
for data in data_list:
    # if int(data["id"]) > 400:
    #     continue
    conversation = data["conversation"]
    # for _ in range(100):
    for i in range(len(conversation) // 2):
        dataset.append(
            {
                "input": conversation[2 * i]["value"],
                "output": conversation[2 * i + 1]["value"],
            }
        )
        # dataset.append(
        #     {
        #         "input": conversation[0]["value"],
        #         "output": conversation[1]["value"] + "<|endoftext|>",
        #     }
        # )

# data_path = "/remote-home/qtu/RY_tool/old_data.json"
# with open(data_path) as fin:
#     dataset = json.load(fin)

# dataset = [
#     {
#         'input': '我们的荣耀小组有：',
#         'output': '郑.逸宁、魏.海洋、涂.倩。<|endoftext|>'
#     } for _ in range(100)
# ]

# dataset = [
#     {
#         'input': "<[user_config]>:{}\n<[query]>:能帮我安排一下5月20日从洛杉矶到纽约的旅行吗？需要预订机票和酒店，酒店我打算5月20日入住到5月23日退房。\n<[apis]>:{'name': 'Plane.real-time_status', 'description': '获取特定航班的状态和实时信息', 'parameters': {'flight_id': {'description': '航班号', 'type': 'str'}, 'date': {'description': '出发日期', 'type': 'str'}}, 'response': {'flight_status': {'description': '航班状态', 'type': 'str'}, 'delay': {'description': '飞机是否延误', 'type': 'bool'}, 'reason': {'description': '如果延误则给出理由', 'type': 'str', 'default': None}}},{'name': 'Plane.check_in', 'description': '提供特定航班的值机信息', 'parameters': {'flight_id': {'description': '航班号', 'type': 'str'}}, 'response': {'check_in_info': {'description': '值机信息，值机柜台或值机岛', 'type': 'str'}, 'check_in_time': {'description': '值机时间', 'type': 'int'}, 'check_in_status': {'description': '值机状态', 'type': 'str'}}},{'name': 'Hotel.search', 'description': '根据地点、入住日期和退房日期等信息搜索酒店信息', 'parameters': {'location': {'description': '目的地地点', 'type': 'str'}, 'check_in_date': {'description': '入住日期', 'type': 'str'}, 'check_out_date': {'description': '退房日期', 'type': 'str'}, 'radius': {'description': '酒店和目的地的最远距离', 'type': 'str', 'default': 'inf'}, 'min_price': {'description': '最低价格', 'type': 'str', 'default': 0}, 'max_price': {'description': '最高价格', 'type': 'str', 'default': 'inf'}}, 'response': {'hotel_name': {'description': '酒店名称', 'type': 'str'}, 'hotel_address': {'description': '酒店的详细地址', 'type': 'str'}}},{'name': 'Plane.search', 'description': '根据出发城市/机场、到达城市/机场和时间搜索符合条件的航班信息', 'parameters': {'from': {'description': '出发城市/机场', 'type': 'str'}, 'to': {'description': '到达城市/机场', 'type': 'str'}, 'date': {'description': '出发日期', 'type': 'int'}}, 'response': {'flight_id': {'description': '航班号', 'type': 'str'}, 'departure_date': {'description': '起飞日期（当地时间）', 'type': 'int'}, 'departure_city': {'description': '起飞城市', 'type': 'str'}, 'departure_airport': {'description': '起飞机场', 'type': 'str'}, 'arrival_date': {'description': '到达日期（当地时间）', 'type': 'int'}, 'arrival_city': {'description': '到达城市', 'type': 'str'}, 'arrival_airport': {'description': '到达机场', 'type': 'str'}, 'airline': {'description': '航空公司', 'type': 'str'}, 'plane_type': {'description': '飞机型号', 'type': 'str'}, 'plan_departure_time': {'description': '计划起飞时间（当地时间）', 'type': 'str'}, 'plan_arrival_time': {'description': '计划到达时间（当地时间）', 'type': 'str'}, 'fly_duration': {'description': '飞行时长', 'type': 'int'}}},{'name': 'Weather.wind', 'description': '获取特定城市当前的风速和风向信息', 'parameters': {'city': {'description': '查询的城市', 'type': 'str'}}, 'response': {'wind_speed': {'description': '风速', 'type': 'int'}, 'wind_direction': {'description': '风向', 'type': 'str'}}}\n<[think]>:",
#         'output': "Plane.search, Hotel.search[Plan]用户需要预订5月20日从洛杉矶到纽约的机票和酒店。首先，我需要使用Plane.search API来搜索5月20日从洛杉矶到纽约的航班信息。API需要from、to和date参数，我将使用洛杉矶作为起点，纽约作为目的地，日期设定为20230520。调用API后会得到航班的具体信息。调用Plane.search API的相关信息：from='洛杉矶', to='纽约', date=20230520[Call]"
#     } for _ in range(100)
# ]

with open("/remote-home/qtu/RY_tool/sft/test_data.json", "w+") as f:
    json.dump(dataset, f, indent = 1, ensure_ascii=False)

ratio = 0.05
k = int(len(dataset) * 0.05)
train_dataset = dataset[:-k]
eval_dataset_ppl = dataset[-k:]
print("total_data num = ", len(dataset))
print("train_data num = ", len(train_dataset))
print("test_data num = ", len(eval_dataset_ppl))
# print(eval_dataset_ppl)
# eval_dataset_ppl, train_dataset = dataset[:int(len(dataset) * ratio)], dataset[int(len(dataset) * ratio):]
# tokenizer_path = "/remote-home/qtu/RY_tool/qwen_tokenizer/Qwen-tokenizer"
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, add_eos_token=True)
# for sp_token in special_tokens_list:
#     tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken(sp_token)]})
tokenizer_path = "/remote-home/share/storage/moss/honor_demo/honor_qwen2_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,trust_remote_code=True, add_eos_token=True)
tokenizer.padding_side = "left"

# test1 = "hello"
# test2 = "hello, I am a student."
# prompt = [test1, test2]
# input = tokenizer.encode(prompt)
# print(input)
# tokenizer.pad_token = tokenizer.unk_token
traine_dataset = CollieDatasetForTraining(
    train_dataset,
    tokenizer=tokenizer,
    max_length=4096
)
eval_dataset_ppl = CollieDatasetForTraining(
    eval_dataset_ppl,
    tokenizer=tokenizer,
    max_length=4096
)

### Prepare Evaluator
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

### Prepare Trainer
callbacks = [
    CheckpointCallback(
        "/remote-home/share/storage/moss/honor_demo/0407_1.5_7b_new_simple_2",
        every_n_epochs=5,  # 每 n 个 epoch 保存一次
        model_only=True,  # 仅保存模型权重，不保存optimzer、训练步数等断点重训信息
    )
]

trainer = Trainer(
    model=model,
    lr_scheduler=lr_scheduler,
    config=config,
    optimizer=optimizer,
    train_dataset=traine_dataset,
    monitors=[
        LossMonitor(config),
        TGSMonitor(config),
        MemoryMonitor(config),
        LRMonitor(config)
    ],
    evaluators=[evaluator_ppl],
    callbacks=callbacks,
)
trainer.train()