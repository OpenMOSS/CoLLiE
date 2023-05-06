set -x
port=$(shuf -i25000-30000 -n1)

# for tensor trainer with inplace sgd
#CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port="$port" examples/mcqa/train_tensor.py examples/mcqa/hf_args_tensor.yaml

# for zero trainer with inplace sgd
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:4,5 train_zero.py hf_args_zero.yaml
