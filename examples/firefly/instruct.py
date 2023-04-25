from collie.models.llama_colossalai import ModelArgs, load_state_dict, save_parallel_model, Tokenizer, HFLikeTokenizer, build_pipe
from collie.trainer.colossalai_trainer import ColossalaiTrainer, TrainerArgs

import torch

def main():
    tokenizer = HFLikeTokenizer(
        tokenizer=Tokenizer(
            model_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model'))
    model_args = ModelArgs()
    model_args.attention = "flash"
    model_args.pp_size = 8
    model_args.tp_size = 1
    model_args.micro_batch_num = 512
    
    trainer_args = TrainerArgs()
    trainer_args.epochs = 50
    trainer_args.eval_per_steps = 100
    trainer_args.eval_per_epoches = 1
    trainer_args.eval_stop_tokens = [tokenizer.eos_token]
    
    state_dict = load_state_dict(
        s3_folder="hdd:s3://opennlplab_hdd/models/llama-415-hf/further-train-7B",
        model_args=model_args
    )
    model = build_pipe(model_args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    trainer = ColossalaiTrainer(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer
    )
    
if __name__ == "__main__":
    main()