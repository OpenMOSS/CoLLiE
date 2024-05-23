#!/bin/bash

BASEDIR=/cpfs01/shared/pj-test/llm-env

#export HOME=/cpfs01/user/$USER
export PATH=$PATH:$BASEDIR/dep/miniconda3/bin

export TORCH_CUDA_ARCH_LIST="8.0"
export CUDA_HOME=${BASEDIR}/dep/cuda-11.7
export GCC_HOME=${BASEDIR}/dep/gcc-10.2.0
export MPFR_HOME=${BASEDIR}/dep/mpfr-4.1.0
export LD_LIBRARY_PATH=${GCC_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${MPFR_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${GCC_HOME}/bin:${CUDA_HOME}/bin:$PATH

export CC=${GCC_HOME}/bin/gcc
export CXX=${GCC_HOME}/bin/c++

export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=eth0
#export NCCL_DEBUG=INFO

#source /cpfs01/shared/pj-test/llm-env/llm-test/bin/activate $BASEDIR/llm-test

torchrun --nproc_per_node 4 test_generation.py
#torchrun --nproc_per_node 4 test_train.py
#torchrun --nproc_per_node 4 test_save.py