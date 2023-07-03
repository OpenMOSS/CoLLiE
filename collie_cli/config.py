import argparse

import yaml

from .bullet import Bullet, Input, VerticalPrompt, colors

description = "Launches an interactive instruction to create and save a configuration file for CoLLiE. The configuration file will be saved at the given path, default to ./collie_default.yml."


def config_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("config", description=description)
    else:
        parser = argparse.ArgumentParser(
            "CoLLiE config command", description=description
        )

    parser.usage = "collie config [<args>]"

    parser.add_argument(
        "--config_file",
        "-c",
        default="./collie_default.yml",
        help="If --config_file is specified, the configuration file will be generated at the given path. Otherwise, the configuration file will be generated at ./collie_default.yml.",
        type=str,
    )

    if subparsers is not None:
        parser.set_defaults(entrypoint=config_command_entry)
    return parser


_prompt_argname_map = {
    "Seed": "seed",
    "Size for pipeline parallelism": "pp_size",
    "Size for tensor parallelism": "tp_size",
    "Size for data paralellism": "dp_size",
    "Choose the partition method for pipeline parallelism.": "pp_partition_method",
    "Training epochs": "train_epochs",
    "After how many training steps in an epoch is the evaluation performed? ": "eval_per_n_steps",
    "After how many training epochs is the evaluation performed? ": "eval_per_n_epochs",
    "Micro batch size for each GPU": "train_micro_batch_size",
    "Steps for gradient accumulation": "gradient_accumulation_steps",
    "Batch size for evaluation": "eval_batch_size",
    "Do you want to use the gradient checkpointing?": "checkpointing",
    "Do you want to use FlashAttention?": "use_flash",
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


def generate_deepspeed_config(
    word_color,
    train_micro_batch_size_per_gpu,
    gradient_accumulation_steps,
):
    print("\n")
    use_existing_config = (
        Bullet(
            "Do you have an existing DeepSpeed config?",
            choices=[
                "Yes",
                "No",
            ],
            bullet="> ",
            word_color=word_color,
        ).launch()
        == "Yes"
    )

    config = {}
    if use_existing_config:
        print("\n")
        config = Input("Path to the DeepSpeed config", default="ds_config.yml").launch()
    else:
        print("\nPlease follow the instructions to generate a new DeepSpeed config.")

        # Precision
        precision = Bullet(
            "\nSelect the training precision",
            choices=["defualt", "fp16 (required for ZeRO)", "bf16"],
            bullet="> ",
            word_color=word_color,
        ).launch()
        if precision == "bf16":
            config["bf16"] = {"enabled": True}
        elif precision == "fp16 (required for ZeRO)":
            config["fp16"] = {
                "enabled": True,
            }
            config["fp16"]["auto_cast"] = (
                Bullet(
                    "\nDo you want to enable auto-cast?",
                    choices=["Yes", "No"],
                    bullet="> ",
                    word_color=word_color,
                ).launch()
                == "Yes"
            )
            # ZeRO
            if (
                Bullet(
                    "\nDo you want to use ZeRO memory optimizations?",
                    choices=["Yes", "No"],
                    bullet="> ",
                    word_color=word_color,
                ).launch()
                == "Yes"
            ):
                zero_config = {}

                zero_config["stage"] = Bullet(
                    "\nSelect the ZeRO stage",
                    choices=["0", "1", "2", "3"],
                    bullet="> ",
                    word_color=word_color,
                ).launch()

                # Offload parameters
                if (
                    zero_config["stage"] == "3"
                    and Bullet(
                        "\nDo you want to offload parameters?",
                        choices=["Yes", "No"],
                        bullet="> ",
                        word_color=word_color,
                    ).launch()
                    == "Yes"
                ):
                    offload_param = {}
                    offload_param["device"] = Bullet(
                        "\nDevice: ",
                        choices=["cpu", "nvme"],
                        bullet="> ",
                        word_color=word_color,
                    ).launch()
                    if offload_param["device"] == "nvme":
                        offload_param["nvme_path"] = Input(
                            "\nOffload path", word_color=word_color
                        ).launch()
                    zero_config["offload_param"] = offload_param

                # Offload optimizer states
                if (
                    zero_config["stage"] != "0"
                    and Bullet(
                        "\nDo you want to offload optimizer states?",
                        choices=["Yes", "No"],
                        bullet="> ",
                        word_color=word_color,
                    ).launch()
                    == "Yes"
                ):
                    offload_optimizer = {}
                    offload_optimizer["device"] = (
                        Bullet(
                            "\nDevice: ",
                            choices=["cpu", "nvme"],
                            bullet="> ",
                            word_color=word_color,
                        ).launch()
                        if zero_config["stage"] == "3"
                        else "cpu"
                    )
                    if offload_optimizer["device"] == "nvme":
                        offload_optimizer["nvme_path"] = Input(
                            "\nOffload path", word_color=word_color
                        ).launch()
                    zero_config["offload_optimizer"] = offload_optimizer

                config["zero_optimization"] = zero_config

    return config


def config_command_entry(args):
    word_color = colors.foreground["cyan"]

    config_command_cli = VerticalPrompt(
        [
            Input("Seed", default="42", word_color=word_color),
            Input("Training epochs", default="100", word_color=word_color),
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
            Input("Micro batch size for each GPU", default="1", word_color=word_color),
            Input(
                "Steps for gradient accumulation", default="1", word_color=word_color
            ),
            Input("Batch size for evaluation", default="1", word_color=word_color),
            Bullet(
                "Do you want to use the gradient checkpointing?",
                choices=[
                    "Yes",
                    "No",
                ],
                bullet="> ",
                word_color=word_color,
            ),
            Bullet(
                "Do you want to use FlashAttention?",
                choices=[
                    "Yes",
                    "No",
                ],
                bullet="> ",
                word_color=word_color,
            ),
            # Input("The possibility for dropout", default="0.0", word_color=word_color),
            # Bullet(
            #     "Choose the method for parameter initialization",
            #     choices=[
            #         "normal",
            #         "xavier_normal",
            #         "xavier_uniform",
            #         "kaiming_normal",
            #         "kaiming_uniform",
            #         "orthogonal",
            #         "sparse",
            #         "eye",
            #         "dirac",
            #     ],
            #     bullet="> ",
            #     word_color=word_color,
            # ),
            # Input(
            #     "Path to the configuration file for DeepSpeed",
            #     default="ds_config.yml",
            #     word_color=word_color,
            # ),
            Input("Size for pipeline parallelism", default="1", word_color=word_color),
            Input("Size for tensor parallelism", default="1", word_color=word_color),
            Input("Size for data paralellism", default="1", word_color=word_color),
            Bullet(
                "Choose the partition method for pipeline parallelism.",
                choices=["parameters", "uniform", "type:[regex]"],
                bullet="> ",
                margin=2,
                word_color=word_color,
            ),
        ]
    )

    result = config_command_cli.launch()
    config = {_prompt_argname_map[k]: _parse(v) for k, v in result}

    if config["pp_partition_method"] == "type:[regex]":
        config["pp_partition_method"] = Input(
            "You've selected the type:[regex] method, please enter the regex",
            default="",
            word_color=word_color,
        ).launch()

    config["ds_config"] = generate_deepspeed_config(
        word_color,
        config["train_micro_batch_size"],
        config["gradient_accumulation_steps"],
    )

    with open(args.config_file, "w") as f:
        yaml.dump(config, f, Dumper=yaml.SafeDumper)
        print(f"🎉 Configuration saved to {args.config_file}")
