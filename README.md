# CoLLiE
<div align="center">
 <img src="docs/assets/images/collie_icon.svg" width="400px">

**CoLLiE**: **Co**llaborative Tuning of **L**arge **L**anguage models **i**n an **E**fficient way

</div>

## Table of Contents
- [Features](#features)
- [Benchmark](#benchmark)
- [Usage](#usage)
- [Installation](#installation)


## Features
CoLLiE provides 

model with flash-attn
implementations with megatron and suitable for pp
支持hf格式的权重

上面的features可以通过config设置直接使用

memory-efficient fine-tuning
in-place sgd, peft
与model，trainer和config解耦

提供了trainer，通过改变config可以直接使用

- Parallelism Strategies
  - Data Parallelism
  - [Pipeline Parallelism]()
  - [Tensor Parallelism](https://github.com/NVIDIA/Megatron-LM)
  - [Zero Redundancy Optimizer (ZeRO)](https://arxiv.org/pdf/1910.02054.pdf)
- Models
  - [FlashAttention](https://github.com/HazyResearch/flash-attention)
- Memory-efficient Fine-tuning Methods
  - Inplace SGD
  - [LoRA](https://arxiv.org/pdf/2106.09685.pdf)
- Friendly Usage

## Benchmark
### Throughput
WIP

## Usage
collie try to be user-friendly
### Config

3d parallel
DP、TP、PP
ZeRO
可以通过config设置直接切换并行方法without any other change

enjoy your 3d parallel just with config modification

并行模式通过config控制，只需要改变config
### Examples
More examples are available at [examples](examples).
### Launch Scripts
CoLLiE offers integration with [torchrun](https://pytorch.org/docs/stable/elastic/run.html) and [slurm](https://github.com/SchedMD/slurm) to enable easy launching of jobs on a single or multiple nodes.

## Installation
```bash
git clone https://github.com/OpenLMLab/collie.git
cd collie
python setup.py install
```
