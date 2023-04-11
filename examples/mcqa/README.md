# TuneLite example on OpenBookQA dataset

Support on full dataset finetuning and few-shot finetuning on 3090.

## How to run the example

> sh example/mcqa/run.sh

May need single device of GPU to finetune 7B, 2 devices for 13B, 4 devices for 30B and 8 devices for 65B.

If you want to finetune LLaMA on the full dataset, please set `few_shot_size = -1`. 
Otherwise, if you want to do few-shot finetuning, set `few_shot_size = K (shot)`.

