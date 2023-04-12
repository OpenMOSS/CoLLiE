port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,6 torchrun --standalone --nnodes=1 --nproc_per_node=6 --master_port="$port"  train.py hf_args.yaml