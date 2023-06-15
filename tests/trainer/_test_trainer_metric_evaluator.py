import sys
import copy

import torch
from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig

sys.path.append("../../")

from collie.config import CollieConfig
from collie.controller.trainer import Trainer
from collie.metrics import BaseMetric
from collie.models.llama.model import LlamaForCausalLM
from collie.controller.trainer import Trainer
from collie.metrics.decode import DecodeMetric
from collie.config import CollieConfig
from collie.controller.evaluator import Evaluator
from collie.utils import zero3_load_state_dict, setup_distribution, GradioProvider
from collie.utils.monitor import StepTimeMonitor, TGSMonitor, MemoryMonitor, LossMonitor, EvalMonitor
from collie.utils.data_provider import GradioProvider, DashProvider



def _test_gpt2():
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2", 
                                            padding_side="left", 
                                            add_eos_token=True)
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    config = CollieConfig.from_pretrained("gpt2")
    config.dp_size = 2
    config.train_epochs = 10
    config.train_micro_batch_size = 1
    config.eval_batch_size = 1
    config.eval_per_n_steps = 20
    config.ds_config = {
        "fp16": {"enabled": True},
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 2e-5
            }
        },
        "zero_optimization": {
            "stage": 3,
        },
        "monitor_config": {
            "enabled": True,
            "csv_monitor": {
                "enabled": True,
                "output_path": "./ds_logs/",
                "job_name": "test"
            }
        }
    }
    setup_distribution(config)
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    train_sample = tokenizer("Collie is a python package for finetuning large language models.", return_tensors="pt").input_ids.squeeze(0)
    eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)[:-1,]
    train_dataset = [(train_sample, train_sample) for _ in range(1000)]
    eval_dataset = [(eval_sample, eval_sample)]
    trainer = Trainer(
        model = model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        generation_config=GenerationConfig(max_new_tokens=32, 
                                    eos_token_id=2, 
                                    pad_token_id=0, 
                                    bos_token_id=1),
        metrics={"decodeMetric": DecodeMetric(tokenizer=tokenizer, gather_result=True),
                 "decodeMetric1": DecodeMetric(tokenizer=tokenizer, gather_result=True)},
        config=config,
        monitors=[LossMonitor(config), MemoryMonitor(config)]
    )
    trainer.train()

def _test_init_engine():

    tokenizer = AutoTokenizer.from_pretrained("gpt2", 
                                            padding_side="left", 
                                            add_eos_token=True)
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    config = CollieConfig.from_pretrained("gpt2")
    config.dp_size = 2
    config.train_epochs = 10
    config.train_micro_batch_size = 1
    config.eval_batch_size = 1
    config.eval_per_n_steps = 20
    config.ds_config = {
        "fp16": {"enabled": True},
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 2e-5
            }
        },
        "zero_optimization": {
            "stage": 3,
        },
        "monitor_config": {
            "enabled": True,
            "csv_monitor": {
                "enabled": True,
                "output_path": "./ds_logs/",
                "job_name": "test"
            }
        }
    }
    setup_distribution(config)
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    train_sample = tokenizer("Collie is a python package for finetuning large language models.", return_tensors="pt").input_ids.squeeze(0)
    eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)[:-1,]
    train_dataset = [(train_sample, train_sample) for _ in range(1000)]
    eval_dataset = [(eval_sample, eval_sample)]

    evaluator_config = copy.deepcopy(config)
    evaluator_config.ds_config.pop("optimizer")
    evaluator = Evaluator(model, dataset=eval_dataset, metrics={"decodeMetric": DecodeMetric(tokenizer=tokenizer, gather_result=True),
                 "decodeMetric1": DecodeMetric(tokenizer=tokenizer, gather_result=True)}, config=config,
                  generation_config=GenerationConfig(max_new_tokens=32, 
                                    eos_token_id=2, 
                                    pad_token_id=0, 
                                    bos_token_id=1))
    trainer = Trainer(
        model = model,
        train_dataset=train_dataset,
        config=config,
        monitors=[LossMonitor(config), MemoryMonitor(config)],
        evaluators=evaluator
    )
    trainer.train()

def _test_only_evaluator():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", 
                                            padding_side="left", 
                                            add_eos_token=True)
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    config = CollieConfig.from_pretrained("gpt2")
    config.dp_size = 2
    config.train_epochs = 10
    config.train_micro_batch_size = 1
    config.eval_batch_size = 1
    config.eval_per_n_steps = 20
    config.ds_config = {
        "fp16": {"enabled": True},
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 2e-5
            }
        },
        "zero_optimization": {
            "stage": 3,
        },
        "monitor_config": {
            "enabled": True,
            "csv_monitor": {
                "enabled": True,
                "output_path": "./ds_logs/",
                "job_name": "test"
            }
        }
    }
    setup_distribution(config)
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    train_sample = tokenizer("Collie is a python package for finetuning large language models.", return_tensors="pt").input_ids.squeeze(0)
    eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)[:-1,]
    train_dataset = [(train_sample, train_sample) for _ in range(1000)]
    eval_dataset = [(eval_sample, eval_sample)]

    evaluator_config = copy.deepcopy(config)
    evaluator_config.ds_config.pop("optimizer")
    evaluator = Evaluator(model, dataset=eval_dataset, metrics={"decodeMetric": DecodeMetric(tokenizer=tokenizer, gather_result=True),
                 "decodeMetric1": DecodeMetric(tokenizer=tokenizer, gather_result=True)}, config=config,
                  generation_config=GenerationConfig(max_new_tokens=32, 
                                    eos_token_id=2, 
                                    pad_token_id=0, 
                                    bos_token_id=1))
    evaluator.eval()


def _test_gpt2_data_provider():
    # import sys
    # import torch
    # sys.path.append("../..")
    # from collie.controller.trainer import Trainer
    # from collie.metrics.decode import DecodeMetric
    # from collie.config import CollieConfig
    # from collie.utils import setup_distribution
    # from collie.utils.data_provider import GradioProvider, DashProvider

    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # from transformers.generation.utils import GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained("gpt2", 
                                            padding_side="left", 
                                            add_eos_token=False,
                                            add_bos_token=True)
    config = CollieConfig.from_pretrained("gpt2")
    config.dp_size = 2
    config.train_epochs = 10
    config.train_micro_batch_size = 1
    config.eval_batch_size = 1000
    config.eval_per_n_steps = 20
    config.ds_config = {
        "fp16": {"enabled": True},
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 2e-5
            }
        },
        "zero_optimization": {
            "stage": 3,
        }
    }
    setup_distribution(config)
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    train_sample = tokenizer("Collie is a python package for finetuning large language models." + tokenizer.eos_token, return_tensors="pt").input_ids.squeeze(0)
    eval_sample = tokenizer("Collie is", return_tensors="pt").input_ids.squeeze(0)
    train_dataset = [(train_sample, train_sample) for _ in range(1000)]
    eval_dataset = [(eval_sample, eval_sample)]
    trainer = Trainer(
        model = model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        generation_config=GenerationConfig(max_new_tokens=32, 
                                        eos_token_id=tokenizer.eos_token_id),
        metrics={"decode": DecodeMetric(tokenizer=tokenizer)},
        data_provider=DashProvider(tokenizer, stream=True, port=7889),
        # data_provider=GradioProvider(tokenizer=tokenizer, port=7879),
        config=config
    )
    trainer.train()

if __name__ == "__main__":
    # _test_llama()
    # _test_gpt2()
    # _test_init_engine()
    # _test_only_evaluator()
    _test_gpt2_data_provider()

# def test():
#     from dash import Dash, html, Input, Output, dcc

#     app = Dash(__name__)
#     app.layout = html.Div([
#         html.H6("更改文本框中的值以查看回调操作！"),
#         html.Div(["输入：",
#                 dcc.Input(id='my-input', value='初始值', type='text')]),
#         html.Br(),
#         html.Div(id='my-output'),
#     ])

#     app.run_server(port=7878, host="0.0.0.0", debug=True)
# import torch.distributed as dist
# import multiprocessing as mp
# config = CollieConfig.from_pretrained("gpt2")
# config.dp_size = 2
# config.train_epochs = 10
# config.train_micro_batch_size = 1
# config.eval_batch_size = 1000
# config.eval_per_n_steps = 20
# config.ds_config = {
#     "fp16": {"enabled": True},
#     "optimizer": {
#         "type": "Adam",
#         "params": {
#             "lr": 2e-5
#         }
#     },
#     "zero_optimization": {
#         "stage": 3,
#     }
# }
# setup_distribution(config)
# if dist.get_rank() == 0:
#     ctx = mp.get_context("fork")
#     process = ctx.Process(target=test)
#     process.start()
