# Finetuning LLaMA

## Dependencies
```=python
pytorch
fairscale
```
## Download files from offical LLaMA
Please download the model files and the tokenizer model from offical [LLaMA repo](https://github.com/facebookresearch/llama).
After downloading, please make the model file 65B/ and tokenizer.model in this folder. If you want to change the path, please check the load\_model function.

## Command
We provide a minimum example of tuning LLaMA model, which asks the model to fit only one sentence using LM objective. Note that the number of nproc relies on the model size, 1 for 7B, 2 for 13B, 4 for 30B, and 8 for 65B. 
```=bash
$torchrun --nproc_per_node 8 mini_llama.py
```
