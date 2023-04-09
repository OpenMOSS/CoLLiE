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
from tunelite.models.llama_colossalai import HFLikeTokenizer, Tokenizer, ModelArgs, get_7B_llama, load_state_dict
from tunelite.trainer.colossalai_trainer import ColossalaiTrainer, TrainerArgs
from torch.utils.data import DataLoader

tokenizer = HFLikeTokenizer(tokenizer=Tokenizer(model_path='./tokenizer.model'))
model_args = ModelArgs()
model_args.pp_size = 8 # your pipeline parallel degree
model = get_7B_llama(model_args=model_args)
state_dict = load_state_dict(
    protocol="file", 
    source="raw", 
    file_folder="./llama/7B",  # where consolidated.00.pth in it
    model_args=model_args)
model.load_state_dict(state_dict)
train_sample = {
    "input_ids": tokenizer("TuneLite is a python package for training large language models", return_tensors="pt")["input_ids"].long()
}, tokenizer("TuneLite is a python package for training large language models", return_tensors="pt")["input_ids"].long()
eval_sample = {
    "input_ids": tokenizer("TuneLite is a python package for", return_tensors="pt")["input_ids"].long()
}, tokenizer("TuneLite is a python package for", return_tensors="pt")["input_ids"].long()

train_dataloader = DataLoader(
    [train_sample for _ in range(1000)],
    batch_size=1
)
eval_dataloader = DataLoader(
    [eval_sample for _ in range(1000)],
    batch_size=1
)
def compute_metrics(batch, generated_batch, epoch, step):
    print("\n")
    print(tokenizer.decode(generated_batch[0]["input_ids"].tolist()))
    print("\n")
    
trainer = ColossalaiTrainer(model=model,
                            train_dataloader=train_dataloader,
                            eval_dataloader=eval_dataloader,
                            tokenizer=tokenizer,
                            compute_metrics=None)
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
