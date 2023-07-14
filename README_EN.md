# CoLLiE
<div align="center">
 <img src="docs/assets/images/collie_icon.svg" width="400px">

**CoLLiE**: **Co**llaborative Tuning of **L**arge **L**anguage Models **i**n an **E**fficient Way

</div>

## Table of Contents
- [Features](#features)
- [Benchmark](#benchmark)
- [Usage](#usage)
- [Installation](#installation)
- [Supported Models](#supported-models)

## Benchmark
### Throughput
WIP

## Features
<div align="center">
    <img src="docs/assets/images/features.svg" width="800px">
</div>

CoLLiE provides collaborative and efficient tuning methods for large language models based on *DeepSpeed* and *PyTorch*. 
It primarily includes the following four features:
- Parallelism Strategies
  - Data Parallelism
  - [Pipeline Parallelism](https://arxiv.org/pdf/1811.06965.pdf)
  - [Tensor Parallelism](https://github.com/NVIDIA/Megatron-LM)
  - [Zero Redundancy Optimizer (ZeRO)](https://arxiv.org/pdf/1910.02054.pdf)
- Models
  - [Flash Attention](https://github.com/HazyResearch/flash-attention)
- Memory-efficient Fine-tuning Methods
  - Lomo
  - [LoRA](https://arxiv.org/pdf/2106.09685.pdf)
- Friendly Usage

CoLLiE has rewritten models using *Megatron-LM* and *Flash Attention*, allowing you to enjoy 3D parallelism simply 
by modifying ```config.dp_size```, ```config.pp_size```, and ```config.tp_size``` (note that the product of these three parallelism sizes should equal # of GPUs). 
Moreover, you can choose whether to use Flash Attention by changing ``config.use_flash``. 
To facilitate user convenience, CoLLiE's models also support methods similar to Huggingface's, where you can load weights from HF using ```model.from_pretrained()```.

If you don't want to write a training loop yourself, CoLLiE provides a [trainer](collie/trainer/trainer.py).
All you need to do is provide the config and dataset to conduct your custom training process.

## Usage

### Documentation and Examples
CoLLiE provides [online documentation](https://openlmlab-collie.readthedocs.io/zh_CN/latest/). More examples are available at [examples](examples).

### Launch Scripts
CoLLiE offers integration with [torchrun](https://pytorch.org/docs/stable/elastic/run.html) and [slurm](https://github.com/SchedMD/slurm) to enable easy launching of jobs on a single or multiple nodes.

## Installation
```bash
git clone https://github.com/OpenLMLab/collie.git
cd collie
python setup.py install
```

## Supported Models

- [MOSS-MOON](https://github.com/OpenLMLab/MOSS)
    - [moss-moon-003-base](https://huggingface.co/fnlp/moss-moon-003-base)
    - [moss-moon-003-sft](https://huggingface.co/fnlp/moss-moon-003-sft)
    - [moss-moon-003-sft-plugin](https://huggingface.co/fnlp/moss-moon-003-sft-plugin)
- [InternLM](https://github.com/InternLM/InternLM)
    - [internlm-7b](https://huggingface.co/internlm/internlm-7b)
    - [internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b)
    - [internlm-chat-7b-8k](https://huggingface.co/internlm/internlm-chat-7b-8k)
- [LLaMA](https://github.com/facebookresearch/llama)
    - [llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)
    - [llama-13b-hf](https://huggingface.co/decapoda-research/llama-13b-hf)
    - [llama-30b-hf](https://huggingface.co/decapoda-research/llama-30b-hf)
    - [llama-65b-hf](https://huggingface.co/decapoda-research/llama-65b-hf)
- [OpenLLaMA]
    - [open_llama_3b](https://huggingface.co/openlm-research/open_llama_3b)
    - [open_llama_7b](https://huggingface.co/openlm-research/open_llama_7b)
    - [open_llama_13b](https://huggingface.co/openlm-research/open_llama_13b)
    - [open_llama_7b_v2](https://huggingface.co/openlm-research/open_llama_7b_v2)
- [ChatGLM](https://github.com/THUDM/ChatGLM-6B)
    - [chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
- [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)
    - [chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)

