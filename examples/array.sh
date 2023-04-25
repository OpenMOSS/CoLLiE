#!/bin/bash
#SBATCH -p moss
#SBATCH --gres=gpu:8
#SBATCH --job-name test_collie
#SBATCH -w SH-IDC1-10-140-24-14,SH-IDC1-10-140-24-15
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=2
#SBATCH --output=slurm-%j.out

echo "1111111"
echo ${SLURM_NODEID}
echo ${SLURM_SUBMIT_HOST}
srun torchrun --nproc_per_node=8 --nnodes=2 --master_addr="10.140.24.14" --node_rank=${SLURM_NODEID} --master_port=27008  /mnt/lustre/zhangshuo/projects/collie/examples/test.py