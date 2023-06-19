[**中文**](./README.md) | [**English**](./README_EN.md)

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
|            | 7B   | 13B  | 30B  | 65B  |
| ---------- | ---- | ---- | ---- | ---- |
| Finetune   | 2    | 3    | 6    | 16   |
| LoRA       | 1    | 1    | 1    | 2    |
| LOMO       | 1    | 1    | 1    | 2    |

注：在使用Adam优化器的情况下，各个模型需要的最少的GPU（A100）数量

## 特点
<div align="center">
    <img src="docs/assets/images/features.svg" width="800px">
</div>

CoLLiE 基于 *DeepSpeed* 和 *PyTorch*，为大型语言模型提供协作式和高效的调优方法。
它主要包括以下四个特点：
- 并行策略
  - 数据并行 (DP)
  - [流水线并行 (PP)](https://arxiv.org/pdf/1811.06965.pdf)
  - [张量并行 (TP)](https://arxiv.org/pdf/2104.04473.pdf)
  - [零冗余优化器 (ZeRO)](https://arxiv.org/pdf/1910.02054.pdf)
- 模型架构
  - [Flash Attention](https://arxiv.org/pdf/2205.14135.pdf)
- 内存高效的微调方法
  - Lomo
  - [LoRA](https://arxiv.org/pdf/2106.09685.pdf)
- 用户友好的使用方式

CoLLiE已使用 *Megatron-LM* 和 *Flash Attention* 重写模型，只需修改 ```config.dp_size```，```config.pp_size```，和```config.tp_size```，就能简单地享受 3D 并行（注意，这三个并行性尺寸的乘积应等于GPU的数量）。
此外，您可以通过更改 ```config.use_flash``` 来选择是否使用 Flash Attention。
为了方便用户，CoLLiE 的模型还支持类似于 🤗Huggingface 的方法，您可以使用 ```model.from_pretrained()``` 从HF加载权重。
如果你不想自己编写训练循环，CoLLiE提供了一个 [训练器](collie/trainer/trainer.py)。你需要做的只是提供配置和数据集来进行你的自定义训练过程。

## 使用

### 示例
更多示例可在 [示例](examples) 中查看。

### 启动脚本
CoLLiE提供了与 [torchrun](https://pytorch.org/docs/stable/elastic/run.html) 和 [slurm](https://github.com/SchedMD/slurm) 的集成，以便在单个或多个节点上轻松启动任务。

## 安装
```bash
pip install git+https://github.com/OpenLMLab/collie.git
```