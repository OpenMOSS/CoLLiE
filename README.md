# CoLLiE
<div align="center">
 <img src="docs/assets/images/collie_icon.svg" width="400px">

**CoLLiE**: **Co**llaborative Tuning of **L**arge **L**anguage Models **i**n an **E**fficient Way

</div>

## 目录
- [特点](#特点)
- [评测](#评测)
- [安装](#安装)
- [使用](#使用)

## 评测
### 吞吐量
待完成

## 特点
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
  - Inplace SGD
  - [LoRA](https://arxiv.org/pdf/2106.09685.pdf)
- Friendly Usage

CoLLiE has rewritten models using *Megatron-LM* and *Flash Attention*, allowing you to enjoy 3D parallelism simply 
by modifying ```config.dp_size```, ```config.pp_size```, and ```config.tp_size``` (note that the product of these three parallelism sizes should equal # of GPUs). 
Moreover, you can choose whether to use Flash Attention by changing ``config.use_flash``. 
To facilitate user convenience, CoLLiE's models also support methods similar to Huggingface's, where you can load weights from HF using ```model.from_pretrained()```.

If you don't want to write a training loop yourself, CoLLiE provides a [trainer](collie/trainer/trainer.py).
All you need to do is provide the config and dataset to conduct your custom training process.

## Usage

### Examples
More examples are available at [examples](examples).

### Launch Scripts
CoLLiE offers integration with [torchrun](https://pytorch.org/docs/stable/elastic/run.html) and [slurm](https://github.com/SchedMD/slurm) to enable easy launching of jobs on a single or multiple nodes.

## Installation
### 1. Apex
```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
### 2. Megatron-LM
```bash
pip install git+https://github.com/NVIDIA/Megatron-LM.git@main#egg=megatron.core
```
### 3. CoLLie
```bash
git clone https://github.com/OpenLMLab/collie.git
cd collie
pip install -r requirements.txt
python setup.py install
```
