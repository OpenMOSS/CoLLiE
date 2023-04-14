# CoLLiE example with colossalai

Support on full dataset finetuning and few-shot finetuning on 3090 using colossalai pipeline.

## How to run the example

> sh example/colossalai_exm/run.sh

If you want to finetune LLaMA on the full dataset, please set `few_shot_size = -1`. 
Otherwise, if you want to do few-shot finetuning, set `few_shot_size = K (shot)`.

## Results

| Dataset | Settings | Model | Performance |
|---------|----------|-------|-------------|
| OpenBookQA | hf_args.yaml | llama 7B | accuracy: 0.722 |
