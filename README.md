# TuneLite

A Light Toolkit to Finetune Large Language Models.

## Beta release (0.0.1)

Requirements:

* CUDA 11.8 
* PyTorch 1.13.1
* Python 3.8

To install:

```bash
python setup.py install
```

## Features
1. Utilizes Colossal-AI's pipeline parallelism mode (requires installing [Colossal-AI](https://github.com/hpcaitech/ColossalAI)).
2. Replaces the original attention structure with flash-attention (requires installing [flash-attn](https://github.com/HazyResearch/flash-attention)).
3. Replaces the original dense layer with fused-dense (requires installing [fused_dense_lib](https://github.com/HazyResearch/flash-attention/tree/main/csrc/fused_dense_lib)).
4. Replaces the original rotary position encoding with rotary_emb (requires installing [rotary_emb](https://github.com/HazyResearch/flash-attention/tree/main/csrc/rotary)).
5. Employs fairscale's tensor parallelism mode (requires installing [fairscale](https://github.com/facebookresearch/fairscale)).
6. Utilizes deepspeed's ZeRO (requires installing [deepspeed](https://github.com/microsoft/DeepSpeed)).
7. Support Inplace SGD.

## Roadmap

+ [x] Utilizes Colossal-AI's pipeline parallelism.
+ [x] Utilizes FairScale's tensor parallelism.
+ [x] Utilizes Deepspeed's ZeRO.
+ [x] Implement Inplace SGD.
+ [x] Reimplement LlaMA with Colossal-AI APIs.
+ [ ] Support Colossal-AI's tensor parallelism and ZeRO CPU offload.
+ [ ] Speed Benchmark.
+ [ ] Add more examples.

## How to use TuneLite

Here's a simple example to run pipeline parallel:

```python
from tunelite.models.llama_colossalai import get_13b_llama, load_state_dict
from tunelite.trainer.colossalai_trainer import ColossalaiTrainer

model = get_13b_llama()
dataloader = mydataloader(*data_args)
trainer = ColossalaiTrainer(*tl_args)
trainer.train()
```

To run tensor parallel with tensor parallel and inplace:

```python
import tunelite as tl
from tunelite.models import llama
model = llama.load_model(*model_args)
dataloader = mydataloader(*data_args)
trainer = tl.Trainer.InplaceTensorTrainer(*tl_args)
trainer.train()
```

For more examples, check [TuneLite Examples](https://github.com/OpenLMLab/TuneLite/tree/main/examples)).
