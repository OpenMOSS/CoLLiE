import torch
from einops import rearrange


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
