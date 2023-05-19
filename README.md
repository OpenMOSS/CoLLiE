# CoLLiE
<div align="center">
 <img src="docs/assets/images/collie_icon.svg" width="400px">

**CoLLiE**: **Co**llaborative Tuning of **L**arge **L**anguage models **i**n an **E**fficient way

</div>

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmark](#benchmark)

## Features

3d parallel
DP、TP、PP
ZeRO
可以通过config设置直接切换并行方法without any other change

model: flash-attn
model implementations with megatron and suitable for pp
with flash-attn
可以通过config设置直接使用

memory-efficient fine-tuning
in-place sgd, peft
与model，trainer和config解耦

## Installation
```bash
git clone https://github.com/OpenLMLab/collie.git
cd collie
python setup.py install
```

## Usage
### Config
并行模式通过config控制，只需要改变config
### Examples
one-sentence overfitting examlpes and generation examples
### Launch Script
支持torchrun & slurm

## Benchmark
### Throughput
WIP