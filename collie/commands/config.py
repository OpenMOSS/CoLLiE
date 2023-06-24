import argparse

import yaml
from bullet import Bullet, Input, Numbers, VerticalPrompt, colors, styles


def config_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("config")
    else:
        parser = argparse.ArgumentParser("Accelerate config command")

    parser.add_argument(
        "--config_file",
        default="./collie_default.yml",
        help="If --config_file is specified, the configuration file will be generated at the given path. Otherwise, the configuration file will be generated at ./collie_default.yml.",
        type=str,
    )

    if subparsers is not None:
        parser.set_defaults(entrypoint=config_command_entry)
    return parser


_prompt_argname_map = {
    "随机数种子": "seed",
    "流水线并行的大小": "pp_size",
    "张量并行大小": "tp_size",
    "数据并行大小": "dp_size",
    "流水线的切分策略": "pp_partition_method",
    "训练时的迭代次数": "train_epochs",
    "训练的一个 epoch 中，每隔多少 step 进行一次验证": "eval_per_n_steps",
    "训练过程中每隔多少次迭代进行一次验证": "eval_per_n_epochs",
    "每个 gpu 上的 batch_size": "train_micro_batch_size",
    "梯度累积的 step 数目": "gradient_accumulation_steps",
    "验证时的 batch 大小": "eval_batch_size",
    "你希望使用梯度检查点吗？": "checkpointing",
    "你希望使用 FlashAttention 吗？": "use_flash",
    "Dropout 的概率": "dropout",
    "初始化方法": "initization_method",
    "DeepSpeed 配置": "ds_config",
}


def _parse(v):
    if v.isdigit():
        return int(v)
    elif v.replace(".", "", 1).isdigit():
        return float(v)
    elif v in ["Yes", "No"]:
        return v == "Yes"
    else:
        return v


def config_command_entry(args):
    word_color = colors.foreground["cyan"]

    config_command_cli = VerticalPrompt(
        [
            Input("随机数种子", default="42", word_color=word_color),
            Input("训练时的迭代次数", default="100", word_color=word_color),
            Input(
                "训练的一个 epoch 中，每隔多少 step 进行一次验证",
                default="0",
                word_color=word_color,
            ),
            Input("训练过程中每隔多少次迭代进行一次验证", default="0", word_color=word_color),
            Input("每个 gpu 上的 batch_size", default="1", word_color=word_color),
            Input("梯度累积的 step 数目", default="1", word_color=word_color),
            Input("验证时的 batch 大小", default="1", word_color=word_color),
            Bullet(
                "你希望使用梯度检查点吗？",
                choices=[
                    "Yes",
                    "No",
                ],
                bullet=" >",
                word_color=word_color,
            ),
            Bullet(
                "你希望使用 FlashAttention 吗？",
                choices=[
                    "Yes",
                    "No",
                ],
                bullet=" >",
                word_color=word_color,
            ),
            Input("Dropout 的概率", default="0.0", word_color=word_color),
            Bullet(
                "初始化方法",
                choices=[
                    "normal",
                    "xavier_normal",
                    "xavier_uniform",
                    "kaiming_normal",
                    "kaiming_uniform",
                    "orthogonal",
                    "sparse",
                    "eye",
                    "dirac",
                ],
                bullet=" >",
                word_color=word_color,
            ),
            Input("DeepSpeed 配置", default="ds_config.yml", word_color=word_color),
            Input("流水线并行的大小", default="1", word_color=word_color),
            Input("张量并行大小", default="1", word_color=word_color),
            Input("数据并行大小", default="1", word_color=word_color),
            Bullet(
                "流水线的切分策略",
                choices=["parameters", "uniform", "type:[regex]"],
                bullet=" >",
                margin=2,
                word_color=word_color,
            ),
        ]
    )

    regx_cli = VerticalPrompt(
        [
            Input(
                "你选择的流水线切分策略为 type:[regex]，请在此输入正则表达式：",
                default="",
                word_color=word_color,
            )
        ],
    )

    result = config_command_cli.launch()
    config = {_prompt_argname_map[k]: _parse(v) for k, v in result}

    if config["pp_partition_method"] == "type:[regex]":
        regx_result = regx_cli.launch()
        config["pp_partition_method"] = regx_result[0][1]

    with open(args.config_file, "w") as f:
        yaml.dump(config, f, Dumper=yaml.SafeDumper)
        print(f"🎉 配置文件已保存至 {args.config_file}")
