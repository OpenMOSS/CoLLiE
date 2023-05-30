FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# python3
RUN apt-get update -y && apt-get install -y git python3 python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# requirements
RUN pip install torch==2.0.0+cu117 --index-url https://download.pytorch.org/whl/cu117 --no-cache-dir
RUN pip install rich tqdm einops websockets fastapi
RUN pip install deepspeed protobuf==3.20.1 sentencepiece transformers
RUN pip install git+https://github.com/NVIDIA/Megatron-LM.git@main#egg=megatron.core
