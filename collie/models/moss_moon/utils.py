import gc

import torch
from torch import distributed as dist
from transformers.modeling_utils import dtype_byte_size
from collie.utils import env, concat_tensor

def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)

def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    return (tensor * cos) + (rotate_every_two(tensor) * sin)

def _weight_name_in_current_rank(names):
    if not env.is_pipeline:
        return names
    layers = env.pipeline_layers_idx
    parts = env.pipeline_parts
    cur_names = []
    # Moss003MoonForCausalLM 的模型顺序为：
    # vocab: transformer.wte.weight
    # dropout: not in dict
    # MossBlock: transformer.h.{idx}.xxx
    # layernorm transformer.ln_f.xxx
    # linear: lm_head.xxx
    for name in names:
        # 找到 MossBlock。idx 对应到 layers_idx 需要 +2
        if len(name.split(".")) > 2 and name.split(".")[2].isdigit() \
            and (int(name.split(".")[2]) + 3) in layers:
                cur_names.append(name)
        if 1 in layers and name.startswith("transformer.wte."):
            # 1 层，embedding
            cur_names.append(name)
        if max(parts) - 1 in layers and name.startswith("lm_head."):
            # 最后一个，lm_head
            cur_names.append(name)
        if max(parts) - 2 in layers and name.startswith("transformer.ln_f."):
            # 倒数第二个 layer norm
            cur_names.append(name)

    return cur_names

def _tp_split_dim(name):
    """
    Return split/merge dim of tensor parallelism.

    :param name: huggingface format
    :return: int or None
    """
    chunk_dim = None
    if name.endswith("wte.weight"):
        # ParallelVocab
        chunk_dim = 0
    elif name.endswith("mlp.fc_out.weight") or \
        name.endswith("attn.out_proj.weight"):
            # RowParallelLinear
            chunk_dim = 1
    elif name.endswith("attn.qkv_proj.weight") or \
        "mlp.fc_in" in name or "lm_head" in name:
            # ColumnParallelLinear
            chunk_dim = 0

    return chunk_dim

def _split_weights(state_dict, tp_rank, tp_size, process_exclusion):
    if tp_size == 1:
        return state_dict
    for name in list(state_dict.keys()):
        param = state_dict[name]
        chunk_dim = _tp_split_dim(name)
        if chunk_dim is None:
            continue

        tensor = list(torch.chunk(param, tp_size, dim=chunk_dim))[tp_rank].detach().clone()

        del state_dict[name]
        if process_exclusion:
            # CPU 内存回收（速度很慢）
            gc.collect()
        state_dict[name] = tensor
    
    return state_dict

def _state_dict_to_load(state_dict, tp_rank, tp_size, process_exclusion):
    """
    Split weights in state_dict and rename if using pipeline.
    """
    state_dict = _split_weights(state_dict, tp_rank, tp_size,
                                    process_exclusion)
    # 流水线情况下，弹出不需要的并且更名
    if env.is_pipeline:
        cur_names = _weight_name_in_current_rank(state_dict.keys())
        for name in list(state_dict.keys()):
            if name not in cur_names:
                state_dict.pop(name)

    return state_dict

def _state_dict_to_save(state_dict, tp_rank, tp_size, tp_group,
                        process_exclusion):
    """
    Gather weights tp tp_rank 0 in tp_group.
    """
    if tp_size == 1:
        return state_dict
    for name in list(state_dict.keys()):
        param = state_dict[name]
        chunk_dim = _tp_split_dim(name)
        if chunk_dim is None:
            continue
        # gather to tp_rank 0
        gather_list = [torch.empty_like(param) for _ in range(tp_size)]
        dist.all_gather(gather_list, param, group=tp_group)
        if tp_rank == 0:
            tensor = concat_tensor(gather_list, chunk_dim)
            del state_dict[name]
            del gather_list
            state_dict[name] = tensor
        else:
            del gather_list
            del state_dict[name]
        if process_exclusion:
            # CPU 内存回收（速度很慢）
            gc.collect()

    return state_dict
    
def set_index_dict(state_dict, weight_name):
    total_size = 0
    weight_map = {}
    for name, weight in state_dict.items():
        weight_size = weight.numel() * dtype_byte_size(weight.dtype)
        weight_map[name] = weight_name
        total_size += weight_size

    return dict(total_size=total_size, weight_map=weight_map)