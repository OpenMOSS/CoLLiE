port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port="$port" examples/mcqa/train.py examples/mcqa/hf_args.yaml
