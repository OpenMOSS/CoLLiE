import torch
import torch.nn.functional as F
import random
import tqdm
from llama_model import setup_model_parallel, load_model, inplace_grad

def train():
    torch.manual_seed(42)
    random.seed(42)
    local_rank, world_size = setup_model_parallel()

    model, tokenizer = load_model(
        ckpt_dir='65B/',  #7B, 13B, 30B, 65B
        tokenizer_path='tokenizer.model',
        local_rank=local_rank,
        world_size=world_size,
        froze_embeddings=False,
        use_fairscale=True,
        max_batch_size=1,
        max_seq_len=1024,
    )

    grad_func = inplace_grad(model, lr=5e-4)
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.register_hook(grad_func)

    model.train()
    labels = tokenizer(['Even with all the recent advancements in large language models, full research access to them remains limited because of the resources that are required to train and run such large models. This restricted access has limited researchers’ ability to understand how and why these large language models work, hindering progress on efforts to improve their robustness and mitigate known issues, such as bias, toxicity, and the potential for generating misinformation.'])
    prompt = tokenizer(['Even with all the recent advancements in large language models, '])

    for epoch in range(1):
        with tqdm.trange(500) as tq:
            for i in tq:
                out = model(labels['input_ids'], labels['attention_mask'])
                _out = out[:,:-1].contiguous().view(-1, 32000) #vocab size
                _label = labels['input_ids'][:,1:].to(out.device).contiguous().view(-1)
                loss = F.cross_entropy(_out, _label, ignore_index=0)
                loss.backward()
                grad_func(0) #update the last one since the hook function will not be called for the last parameter
                tq.set_postfix({'loss': loss.item()})
                if i%30==0:
                    with torch.no_grad():
                        pred = model.generate(prompt['input_ids'][:,:-1], prompt['attention_mask'][:,:-1], 100, 0.1, 0.5, 5)
                    if local_rank==0:
                        print('GPU Memory: ', torch.cuda.max_memory_reserved()/1024/1024/1024)
                        print('Generated: ', tokenizer.decode(pred[0].tolist()))
                        print('Loss: ', loss.item())
        # if you want to save the model, please uncomment the following line. Each process will save its own part of parameters, so
        # you have multiple files for one model. For loading, please see the load_checkpoints, you need to put all saved files in a folder.
        # And copy the params.json from the original model, it is just a config file. 
        # torch.save(model.state_dict(), 'tmp_save.'+'{0:02}'.format(local_rank)+'.pth') 
if __name__=='__main__':
    train()
