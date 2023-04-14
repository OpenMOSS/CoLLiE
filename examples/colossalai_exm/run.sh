port=$(shuf -i25000-30000 -n1)

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,6 torchrun --standalone --nnodes=1 --nproc_per_node=6 --master_port="$port"  train.py hf_args.yaml
CUDA_VISIBLE_DEVICES=0,1,2,4,6,7 srun -p moss -w SH-IDC1-10-140-24-20 --job-name=llama_obqa_7b --ntasks-per-node=1 torchrun --standalone --nnodes=1 --nproc_per_node=6 --master_port="$port"  train.py hf_args.yaml