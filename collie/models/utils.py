import os
import json

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
    version = flash_attn.__version__.split('.')[0]
    batch_size, seq_len, _, _ = query.shape
    if int(version) < 2:
        from flash_attn.flash_attention import FlashAttention
        qkv = torch.stack([query, key, value], dim=2)
        output, _ = FlashAttention()(qkv, causal=True)
        output = rearrange(output, "b n h d -> b n (h d)")
    else:
        from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
        from flash_attn.bert_padding import unpad_input, pad_input
        kv = torch.stack([key, value], dim=2)
        q_unpad, indices, cu_seqlens_q, max_seqlen_q = unpad_input(query, attention_mask)
        kv_unpad, indices, cu_seqlens_kv, max_seqlen_kv = unpad_input(kv, attention_mask)
        output_unpad = flash_attn_varlen_kvpacked_func(
            q_unpad, kv_unpad, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, 0.0, softmax_scale=None, causal=True
        )
        output = pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices,  batch_size, seq_len
        )
    return output

def merge_index_dict(path, file_list, driver):
    """
    合并分散的 index json
    """
    total_size = 0
    weight_map = {}
    for _file in file_list:
        _file = os.path.join(path, _file)
        if not driver.exists(_file):
            logger.rank_zero_warning(f"Detect missing index file {_file}, skip merging.")
            return
        _index_dict = json.loads(driver.load(_file, mode="r"))
        total_size += _index_dict["total_size"]
        weight_map.update(_index_dict["weight_map"])
        driver.delete(_file)
    merged_dict = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }
    driver.save(
        json.dumps(merged_dict, indent=2, sort_keys=True) + "\n",
        os.path.join(path, "pytorch_model.bin.index.json")
    )