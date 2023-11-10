import json
import os

import torch
from einops import rearrange

from collie.log import logger


def flash_attention(query, key, value, attention_mask):
    """
    应用 Flash Attention

    :param query: batzh_size, seq_len, heads, head_dim
    :param key: batzh_size, seq_len, heads, head_dim
    :param value: batzh_size, seq_len, heads, head_dim
    :param attetion_mask: batch_size, seq_len
    """
    import flash_attn

    version = flash_attn.__version__.split(".")[0]
    batch_size, seq_len, _, _ = query.shape
    if int(version) < 2:
        from flash_attn.flash_attention import FlashAttention

        qkv = torch.stack([query, key, value], dim=2)
        output, _ = FlashAttention()(qkv, causal=True)
        output = rearrange(output, "b n h d -> b n (h d)")
    else:
        from flash_attn.bert_padding import pad_input, unpad_input
        from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func

        kv = torch.stack([key, value], dim=2)
        q_unpad, indices, cu_seqlens_q, max_seqlen_q = unpad_input(
            query, attention_mask
        )
        kv_unpad, indices, cu_seqlens_kv, max_seqlen_kv = unpad_input(
            kv, attention_mask
        )
        output_unpad = flash_attn_varlen_kvpacked_func(
            q_unpad,
            kv_unpad,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            0.0,
            softmax_scale=None,
            causal=True,
        )
        output = pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            indices,
            batch_size,
            seq_len,
        )
    return output


def kv_cache_to_inputs_for_model(past_key_values):
    """
    在模型的输入阶段，将嵌套元组形式的past_key_values转化为inputs字典中的每个字段
    """
    inputs = {}
    if past_key_values is not None:
        for i, past_key_value in enumerate(past_key_values):
            inputs[f"past_key_values_layer{i}_key"] = past_key_value[0]
            inputs[f"past_key_values_layer{i}_value"] = past_key_value[1]
    return inputs


def inputs_to_kv_cache_for_model(num_hidden_layers, inputs):
    """
    在模型的输出阶段，将inputs字典中的kv_cache转化为嵌套元组形式的past_key_values
    """
    past_key_values = ()
    try:
        for i in range(0, num_hidden_layers):
            past_key_values += ((inputs[f"past_key_values_layer{i}_key"], 
                                inputs[f"past_key_values_layer{i}_value"]), )
    except:
        pass
    return past_key_values if past_key_values != () else None


def kv_cache_to_inputs_for_layer(idx, new_layer_past):
    """
    在第idx层的输出阶段，将元组形式的new_layer_past转化为inputs字典中的每个字段
    """
    inputs = {}
    if new_layer_past is not None:
        inputs[f"past_key_values_layer{idx}_key"] = new_layer_past[0]
        inputs[f"past_key_values_layer{idx}_value"] = new_layer_past[1]
    return inputs


def inputs_to_kv_cache_for_layer(idx, inputs):
    """
    在第idx层的输入阶段，将inputs字典中的kv_cahce转化为元组形式的layer_past
    """
    if f"past_key_values_layer{idx}_key" in inputs and f"past_key_values_layer{idx}_value" in inputs:
        return (inputs[f"past_key_values_layer{idx}_key"], 
                inputs[f"past_key_values_layer{idx}_value"]) 
    else:
        return None



# Index dict merging is now handled by pp rank 0 without tmp file.
# This function is deprecated.
# def merge_index_dict(path, file_list, driver):
#     """
#     合并分散的 index json
#     """
#     total_size = 0
#     weight_map = {}
#     for _file in file_list:
#         _file = os.path.join(path, _file)
#         if not driver.exists(_file):
#             logger.rank_zero_warning(f"Detect missing index file {_file}, skip merging.")
#             return
#         _index_dict = json.loads(driver.load(_file, mode="r"))
#         total_size += _index_dict["total_size"]
#         weight_map.update(_index_dict["weight_map"])
#         driver.delete(_file)
#     merged_dict = {
#         "metadata": {"total_size": total_size},
#         "weight_map": weight_map
#     }
#     driver.save(
#         json.dumps(merged_dict, indent=2, sort_keys=True) + "\n",
#         os.path.join(path, "pytorch_model.bin.index.json")
#     )
