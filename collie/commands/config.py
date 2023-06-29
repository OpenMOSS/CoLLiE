import argparse

import yaml
from .bullet import Bullet, Input, VerticalPrompt, colors


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
    "Seed: ": "seed",
    "Size for pipeline parallelism: ": "pp_size",
    "Size for tensor parallelism: ": "tp_size",
    "Size for data paralellism: ": "dp_size",
    "Choose the partition method for pipeline parallelism.": "pp_partition_method",
    "Training epochs: ": "train_epochs",
    "After how many training steps in an epoch is the evaluation performed? ": "eval_per_n_steps",
    "After how many training epochs is the evaluation performed? ": "eval_per_n_epochs",
    "Micro batch size for each GPU: ": "train_micro_batch_size",
    "Steps for gradient accumulation: ": "gradient_accumulation_steps",
    "Batch size for evaluation: ": "eval_batch_size",
    "Do you want to use the gradient checkpointing?": "checkpointing",
    "Do you want to use FlashAttention?": "use_flash",
    "The possibility for dropout: ": "dropout",
    "Choose the method for parameter initialization": "initization_method",  # TODO: a spelling mistake?
    "Path to the configuration file for DeepSpeed: ": "ds_config",  # TODO: choose between using an existing config or to generate a new one.
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
            Input("Seed: ", default="42", word_color=word_color),
            Input("Training epochs: ", default="100", word_color=word_color),
            Input(
                "After how many training steps in an epoch is the evaluation performed? ",
                default="0",
                word_color=word_color,
            ),
            Input(
                "After how many training epochs is the evaluation performed? ",
                default="0",
                word_color=word_color,
            ),
            Input(
                "Micro batch size for each GPU: ", default="1", word_color=word_color
            ),
            Input(
                "Steps for gradient accumulation: ", default="1", word_color=word_color
            ),
            Input("Batch size for evaluation: ", default="1", word_color=word_color),
            Bullet(
                "Do you want to use the gradient checkpointing?",
                choices=[
                    "Yes",
                    "No",
                ],
                bullet=" >",
                word_color=word_color,
            ),
            Bullet(
                "Do you want to use FlashAttention?",
                choices=[
                    "Yes",
                    "No",
                ],
                bullet=" >",
                word_color=word_color,
            ),
            Input(
                "The possibility for dropout: ", default="0.0", word_color=word_color
            ),
            Bullet(
                "Choose the method for parameter initialization",
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
            Input(
                "Path to the configuration file for DeepSpeed: ",
                default="ds_config.yml",
                word_color=word_color,
            ),
            Input(
                "Size for pipeline parallelism: ", default="1", word_color=word_color
            ),
            Input("Size for tensor parallelism: ", default="1", word_color=word_color),
            Input("Size for data paralellism: ", default="1", word_color=word_color),
            Bullet(
                "Choose the partition method for pipeline parallelism.",
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
                "You've selected the type:[regex] method, please enter the regex: ",
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
        print(f"🎉 Configuration saved to {args.config_file}")
