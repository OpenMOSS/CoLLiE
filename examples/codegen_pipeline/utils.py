def get_wandb_name(trainer, optimizer):

    hp_dict = {
        "pp": trainer.collie_args.num_stages,
        "lr": trainer.collie_args.learning_rate,
        "fp16": trainer.engine.fp16_enabled(),
        "offload": trainer.engine.zero_cpu_offload(),
        "mbnum": trainer.engine.gradient_accumulation_steps(),
        "bsz": trainer.engine.train_micro_batch_size_per_gpu(),
        "optim": optimizer.__class__.__name__,
        "schd": trainer.lr_scheduler.__class__.__name__ \
                if trainer.lr_scheduler is not None else "none",
        # "warmup": trainer.colli_args.warmup,
        # "clipgradnorm": trainer.colli_args.clip_grad_norm,
        # "clipgrad": trainer.colli_args.clip_grad_value,
        "cliploss": trainer.collie_args.clip_loss_value,
    }
    names = []
    for key, value in hp_dict.items():
        names.append(f"{key}_{value}")
    name = "-".join(names)

    return name

