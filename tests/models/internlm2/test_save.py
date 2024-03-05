import sys
import time
sys.path.append("../../../")
import torch
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from collie.models import ChatGLM2ForCausalLM, LlamaForCausalLM, InternLM2ForCausalLM
from collie import CollieConfig, env, Trainer

pretrained_path = "internlm/internlm2-7b"

config = CollieConfig.from_pretrained(pretrained_path,
        trust_remote_code=True)
config.ds_config = {
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 0,
    }
}
config.dp_size = 1
config.tp_size = 4
config.pp_size = 1
# model = LlamaForCausalLM.from_pretrained(pretrained_path, config=config).cuda()
model = InternLM2ForCausalLM.from_pretrained(pretrained_path, config=config)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
trainer = Trainer(model, optimizer=optimizer, config=config)
start_time = time.time()
print(f"start saving")
trainer.save_model("./dev")
print(f"end saving {time.time() - start_time}")
model.save_parallel_state_dict(model.state_dict(), "./dev", config=config)

if env.rank == 0:
    print("start loading")
    model = AutoModelForCausalLM.from_pretrained("./dev", trust_remote_code=True).cuda()
    print("end loading")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_path,
        trust_remote_code=True,
    )
    input_ids = tokenizer("Llama is a", return_tensors="pt")["input_ids"].cuda()
    outs = model.generate(input_ids, max_new_tokens=256, eos_token_id=2)
    print(tokenizer.decode(outs[0], skip_special_tokens=True))
