set -x
port=$(shuf -i25000-30000 -n1)

srun -p moss -w SH-IDC1-10-140-24-13 --job-name=collie_codegen_pipeline --ntasks-per-node=1 deepspeed --master_port "$port" train.py