import sys
import time
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from collie.models import ChatGLM2ForCausalLM, LlamaForCausalLM
from collie import CollieConfig, env, Trainer

pretrained_path = "huggyllama/llama-7b"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_path,
    trust_remote_code=True,
)
config = CollieConfig.from_pretrained(pretrained_path,
        trust_remote_code=True)
config.ds_config = {
    "fp16": {
        "enabled": True
    },
    # "zero_optimization": {
    #     "stage": 3,
    # }
}
config.dp_size = 1
config.tp_size = 2
config.pp_size = 2
# model = LlamaForCausalLM.from_pretrained(pretrained_path, config=config).cuda()
model = LlamaForCausalLM.from_pretrained(pretrained_path, config=config)
trainer = Trainer(model, config=config)
start_time = time.time()
print(f"start saving")
trainer.save_model("./dev")
print(f"end saving {time.time() - start_time}")
# model.save_parallel_state_dict(model.state_dict(), "./dev", config=config)

if env.rank == 0:
    print("start loading")
    model = AutoModelForCausalLM.from_pretrained("./dev", trust_remote_code=True).cuda()
    print("end loading")
    input_ids = tokenizer("Llama is a", return_tensors="pt")["input_ids"].cuda()
    outs = model.generate(input_ids, max_new_tokens=256, eos_token_id=2)
    print(tokenizer.decode(outs[0], skip_special_tokens=True))
