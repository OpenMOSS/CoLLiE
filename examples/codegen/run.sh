set -x
port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=2,3,4,5 srun -p moss -w SH-IDC1-10-140-24-20 --job-name=codegen-collie deepspeed --master_port "$port" train_zero.py hf_args_zero.yaml
