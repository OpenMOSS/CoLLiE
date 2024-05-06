import math

import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_one_row_block(
    start_m,
    Q: tl.const,
    K: tl.const,
    V: tl.const,
    Out,
    Lse,
    softmax_scale: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_om: tl.constexpr,
    actual_seqlen_q,
    actual_seqlen_k,
    window_size_global,
    SEQOFFSET,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM) 
    q_ptrs = (Q + (offs_m[:, None] * stride_qm + offs_d[None, :]))
    l_ptrs = Lse + offs_m
    out_ptrs = (Out + (offs_m[:, None] * stride_om + offs_d[None, :]))
    if EVEN_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < actual_seqlen_q, other=0.0, cache_modifier=".cg")
        
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = softmax_scale * log2e
    # load k, v 
    offs_n_base = tl.arange(0, BLOCK_N)
    k_ptrs = (K + (offs_d[:, None] + offs_n_base[None, :] * stride_kn)) # (BLOCK_HEADDIM, BLOCK_N)
    v_ptrs = (V + (offs_n_base[:, None] * stride_vn + offs_d[None, :]))
    global_end_n = tl.cdiv(window_size_global, BLOCK_N) * BLOCK_N
    global_end_n = tl.multiple_of(global_end_n, BLOCK_N)
    # loop of global part(could be 0)
    for start_n in range(0, global_end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        
        # load k, v 
        if EVEN_N:
            k = tl.load(k_ptrs + start_n * stride_kn, cache_modifier=".cg")
            v = tl.load(v_ptrs + start_n * stride_vn, cache_modifier=".cg")
        else:
            mask_n = offs_n < actual_seqlen_k  
            k = tl.load(k_ptrs + start_n * stride_kn, mask=mask_n[None, :], other=0.0, cache_modifier=".cg") # (BLOCK_HEADDIM, BLOCK_N)
            v = tl.load(v_ptrs + start_n * stride_vn, mask=mask_n[:, None], other=0.0, cache_modifier=".cg")

        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, qk, input_precision="tf32")
        
        # no need to mask for EVEN_N in `global` case
        # if IS_GLOBAL:
        qk += tl.where( # True will not mask
            (offs_m[:, None] >= offs_n[None, :]) &
            (((SEQOFFSET + offs_m)[:, None] <= offs_n[None, :]) | ((offs_n < window_size_global)[None, :]))
            , 0, float("-inf"))
        
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_i_new[:, None] * qk_scale)
        m_i = m_i_new

        # -- scale and update acc: acc *= alpha[:, None]--
        acc *= alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc)

        # -- update m_i and l_i --
        l_i = tl.fma(l_i, alpha, tl.sum(p, 1))

    local_start_n = tl.maximum(((start_m * BLOCK_M + SEQOFFSET) // BLOCK_N) * BLOCK_N, global_end_n)
    local_start_n = tl.multiple_of(local_start_n, BLOCK_N)
    end_n = tl.minimum((start_m + 1) * BLOCK_M, actual_seqlen_k)
    for start_n in range(local_start_n, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        
        # load k, v 
        if EVEN_N:
            k = tl.load(k_ptrs + start_n * stride_kn, cache_modifier=".cg")
            v = tl.load(v_ptrs + start_n * stride_vn, cache_modifier=".cg")
        else:
            mask_n = offs_n < actual_seqlen_k  
            k = tl.load(k_ptrs + start_n * stride_kn, mask=mask_n[None, :], other=0.0, cache_modifier=".cg") # (BLOCK_HEADDIM, BLOCK_N)
            v = tl.load(v_ptrs + start_n * stride_vn, mask=mask_n[:, None], other=0.0, cache_modifier=".cg")

        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, qk, input_precision="tf32")
        
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where(offs_n[None, :] < actual_seqlen_k, 0, float("-inf"))
        # if IS_GLOBAL:
        qk += tl.where( # True will not mask
            (offs_m[:, None] >= offs_n[None, :]) & ### $
            (((SEQOFFSET + offs_m)[:, None] <= offs_n[None, :])) # `local` part so we need not to (start_n + offs_n < window_size_global)
            , 0, float("-inf"))
        
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_i_new[:, None] * qk_scale)
        # -- scale and update acc: acc *= alpha[:, None]--
        acc *= alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc, input_precision="tf32")

        # -- update m_i and l_i --
        l_i = tl.fma(l_i, alpha, tl.sum(p, 1))
        m_i = m_i_new
            
    acc = acc * (1.0 / l_i[:, None]) # reduce the number of division
    l = tl.fma(m_i, softmax_scale, tl.log(l_i)) # log(normalizer)
    # initialize pointers to output
    if EVEN_M:
        tl.store(l_ptrs, l, cache_modifier=".cs") # .cs is for data accessed once
        tl.store(out_ptrs, acc, cache_modifier=".cs")
    else:
        mask_m = offs_m < actual_seqlen_q
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cs")
        tl.store(out_ptrs, acc, mask=mask_m[:, None], cache_modifier=".cs")

@triton.heuristics(
    {
        "BLOCK_M": lambda args: 128,
        "BLOCK_N": lambda args: 128, # 64 or 128
        "EVEN_M": lambda args: args["actual_seqlen_q"] & 127 == 0, #  % 128 == 0 
        "EVEN_N": lambda args: args["actual_seqlen_k"] & 127 == 0, #  % 128 == 0
        "num_warps": lambda args: 8,
        "num_stages": lambda args: 3, # for faster forward pass
    }
)
@triton.jit
def _fwd_kernel(
    Q: tl.const,
    K: tl.const,
    V: tl.const,
    Out,
    Lse,
    softmax_scale: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    actual_seqlen_q: tl.constexpr,
    actual_seqlen_k: tl.constexpr,
    max_seqlen_q_rounded,
    nheads: tl.constexpr,
    nheads_k: tl.constexpr,
    window_size_global: tl.constexpr,
    SEQOFFSET: tl.constexpr,
    d_rounded: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_b = tl.program_id(2)
    # invalid grid block for this seq
    if actual_seqlen_q <= start_m * BLOCK_M:
        return 
    
    off_h = tl.program_id(1)
    Q += ( off_b * stride_qb + off_h * stride_qh )
    Out += ( off_b * stride_ob + off_h * stride_oh )
    Lse += (off_b * nheads + off_h) * max_seqlen_q_rounded
    
    off_h_kv = off_h * nheads_k // nheads
    K += ( off_b * stride_kb + off_h_kv * stride_kh )
    V += ( off_b * stride_vb + off_h_kv * stride_vh )

    _fwd_kernel_one_row_block(
        start_m,
        Q,
        K,
        V,
        Out,
        Lse,
        softmax_scale,
        stride_qm,
        stride_kn,
        stride_vn,
        stride_om,
        actual_seqlen_q,
        actual_seqlen_k,
        window_size_global,
        SEQOFFSET,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_HEADDIM=d_rounded,
    )

@triton.heuristics(
    {
        "BLOCK_M": lambda args: 128,
        "EVEN_M": lambda args: args["actual_seqlen_q"] & 127 == 0, #  % 128 == 0
        # "num_warps": lambda args: 8,
        # "num_stages": lambda args: 1,
    }
)
@triton.jit
def _bwd_preprocess_do_o_dot(
    Out: tl.const,
    DO: tl.const,
    Delta,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dom: tl.constexpr,
    actual_seqlen_q: tl.constexpr,
    max_seqlen_q_rounded,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    if actual_seqlen_q <= start_m * BLOCK_M:
        return 
    
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o_ptrs = (Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :])
    do_ptrs = (
        DO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :]
    )
    EVEN_M = actual_seqlen_q % BLOCK_M == 0

    if EVEN_M:
        o = tl.load(o_ptrs, cache_modifier=".cg").to(tl.float32)
        do = tl.load(do_ptrs, cache_modifier=".cg").to(tl.float32)
    else:
        mask_m = (offs_m < actual_seqlen_q)[:, None]
        o = tl.load(o_ptrs, mask=mask_m, other=0.0, cache_modifier=".cg").to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m, other=0.0, cache_modifier=".cg").to(tl.float32)
    
    delta = tl.sum(o * do, axis=1)
    # off_b * tl.num_programs(1) + off_h == off_b * nheads + off_h
    delta_ptrs = Delta + (off_b * tl.num_programs(1) + off_h) * max_seqlen_q_rounded + offs_m
    # write-back
    if EVEN_M:
        tl.store(delta_ptrs, delta, cache_modifier=".cs")
    else:
        tl.store(delta_ptrs, delta, mask=(offs_m < actual_seqlen_q), cache_modifier=".cs")

@triton.heuristics(
    {
        "BLOCK_M": lambda args: 64,
        "BLOCK_N": lambda args: 64, # Reducing block sizes
        "EVEN_M": lambda args: args["actual_seqlen_q"] & 63 == 0, #  % 64 == 0
        "EVEN_N": lambda args: args["actual_seqlen_k"] & 63 == 0, #  % 64 == 0
        "num_warps": lambda args: 4, 
        "num_stages": lambda args: 2,
    }
)
@triton.jit
def _bwd_dk_dv_kernel(
    Q: tl.const,
    K: tl.const,
    V: tl.const,
    DO: tl.const,
    DK,
    DV,
    LSE: tl.const,
    D: tl.const,
    softmax_scale: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    nheads: tl.constexpr,
    nheads_k: tl.constexpr,
    actual_seqlen_q: tl.const,
    actual_seqlen_k: tl.const,
    max_seqlen_q_rounded,
    window_size_global: tl.constexpr,
    SEQOFFSET: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    off_h = tl.program_id(1)
    start_n = tl.program_id(0)
    if actual_seqlen_k <= start_n * BLOCK_N:
        return 
    
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = softmax_scale * log2e

    off_b = tl.program_id(2)
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    off_h_kv = off_h * nheads_k // nheads
    K += off_b * stride_kb + off_h_kv * stride_kh
    V += off_b * stride_vb + off_h_kv * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    # pointer to row-wise quantities in value-like data
    off_hb = off_b * nheads + off_h
    D += off_hb * max_seqlen_q_rounded
    LSE += off_hb * max_seqlen_q_rounded
    
    begin_m = ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n_slice = offs_n[None, :]
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # k & v transposed here
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n_slice * stride_kn + offs_d[:, None]) # transposed here
    v_ptrs = V + (offs_n_slice * stride_vn + offs_d[:, None]) # transposed here
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    # load k , v
    if EVEN_N:
        k = tl.load(k_ptrs, cache_modifier=".cg")
        v = tl.load(v_ptrs, cache_modifier=".cg")
    else:
        mask_n = offs_n_slice < actual_seqlen_k
        k = tl.load(k_ptrs, mask=mask_n, other=0.0, cache_modifier=".cg")
        v = tl.load(v_ptrs, mask=mask_n, other=0.0, cache_modifier=".cg")
    
    # loop over rows
    num_block_m = tl.cdiv(actual_seqlen_q, BLOCK_M)
    end_m = num_block_m * BLOCK_M # $
    global_end_n = tl.cdiv(window_size_global, BLOCK_N) - 1
    local_start_n = tl.cdiv(actual_seqlen_q + SEQOFFSET, BLOCK_N) - 1 # actual_seqlen_k-window_size_left
    if (start_n > global_end_n) & (start_n < local_start_n):
        end_m = tl.cdiv((start_n + 1) * BLOCK_N - SEQOFFSET, BLOCK_M) * BLOCK_M # $
    for start_m in range(begin_m, end_m, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        # load q, l, do on-chip
        if EVEN_M:
            q = tl.load(q_ptrs, cache_modifier=".cg")
            do = tl.load(do_ptrs, cache_modifier=".cg")
            l = tl.load(LSE + offs_m, cache_modifier=".cg")
            Di = tl.load(D + offs_m, cache_modifier=".cg")
        else:
            mask_m = offs_m < actual_seqlen_q
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0, cache_modifier=".cg")
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0, cache_modifier=".cg")
            l = tl.load(LSE + offs_m, mask=mask_m, cache_modifier=".cg")
            Di = tl.load(D + offs_m, mask=mask_m, cache_modifier=".cg")
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
            
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, qk, input_precision="tf32")
        
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where(offs_n_slice < actual_seqlen_k, 0, float("-inf"))
        qk += tl.where( # True will not mask
            (offs_m[:, None] >= offs_n_slice) &
            (((SEQOFFSET + offs_m)[:, None] <= offs_n_slice) | (offs_n_slice < window_size_global))
            , 0, float("-inf"))

        p = tl.math.exp2(qk * qk_scale - l[:, None] * log2e)
        dv = tl.dot(tl.trans(p.to(do.dtype)), do, dv, input_precision="tf32")
        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp = tl.dot(do, v, dp, input_precision="tf32")

        ds = (p * (dp - Di[:, None])).to(q.dtype)
        # compute dk = dot(ds.T, q)
        dk = tl.dot(tl.trans(ds), q, dk, input_precision="tf32")
        
    dk *= softmax_scale
    # write-back
    if EVEN_N:
        tl.store(dv_ptrs, dv.to(k.dtype), cache_modifier=".cs")
        tl.store(dk_ptrs, dk.to(k.dtype), cache_modifier=".cs")
    else:
        mask_n = offs_n < actual_seqlen_k
        tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None], cache_modifier=".cs")
        tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None], cache_modifier=".cs")


@triton.heuristics(
    {
        "BLOCK_M": lambda args: 64,
        "BLOCK_N": lambda args: 64, # Reducing block sizes
        "EVEN_M": lambda args: args["actual_seqlen_q"] & 63 == 0, #  % 64 == 0
        "EVEN_N": lambda args: args["actual_seqlen_k"] & 63 == 0, #  % 64 == 0
        "num_warps": lambda args: 4, 
        "num_stages": lambda args: 2,
    }
)
@triton.jit
def _bwd_dq_kernel(
    Q: tl.const,
    K: tl.const,
    V: tl.const,
    DO: tl.const,
    DQ,
    LSE: tl.const,
    D: tl.const,
    softmax_scale: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dqb: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqm: tl.constexpr,
    nheads: tl.constexpr,
    nheads_k: tl.constexpr,
    actual_seqlen_q: tl.const,
    actual_seqlen_k: tl.const,
    max_seqlen_q_rounded,
    window_size_global: tl.constexpr,
    SEQOFFSET: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    start_m = tl.program_id(0)
    # invalid grid block for this seq
    if actual_seqlen_q <= start_m * BLOCK_N:
        return 
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = softmax_scale * log2e

    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    off_h_kv = off_h * nheads_k // nheads
    K += off_b * stride_kb + off_h_kv * stride_kh
    V += off_b * stride_vb + off_h_kv * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    # pointer to row-wise quantities in value-like data
    off_hb = off_b * nheads + off_h
    D += off_hb * max_seqlen_q_rounded
    LSE += off_hb * max_seqlen_q_rounded
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM) 
    offs_m_slice = offs_m[:, None]
    q_ptrs = (Q + (offs_m_slice * stride_qm + offs_d[None, :]))
    do_ptrs = (DO + (offs_m_slice * stride_dom + offs_d[None, :]))
    dq_ptrs = DQ + (offs_m_slice * stride_dqm + offs_d[None, :])

    # load q & do: it will stay in SRAM throughout
    # load q & do & load l & delta: it will stay in SRAM throughout
    if EVEN_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
        do = tl.load(do_ptrs, cache_modifier=".cg")
        l = tl.load(LSE + offs_m, cache_modifier=".cg")
        Di = tl.load(D + offs_m, cache_modifier=".cg")
    else:
        mask_m = offs_m < actual_seqlen_q
        l = tl.load(LSE + offs_m, mask=mask_m, cache_modifier=".cg")
        Di = tl.load(D + offs_m, mask=mask_m, cache_modifier=".cg")
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")
        do = tl.load(do_ptrs, mask=mask_m[:, None], cache_modifier=".cg")
    l_slice_scale = l[:, None] * log2e

    # dq init
    dq = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
        
    # load k, v. k, v transposed here
    offs_n_base_slice = (tl.arange(0, BLOCK_N))[None, :]
    k_ptrs = (K + (offs_d[:, None] + offs_n_base_slice * stride_kn)) # (BLOCK_HEADDIM, BLOCK_N)
    v_ptrs = (V + (offs_d[:, None] + offs_n_base_slice * stride_vn))
    global_end_n = tl.cdiv(window_size_global, BLOCK_N) * BLOCK_N
    global_end_n = tl.multiple_of(global_end_n, BLOCK_N)
    # loop of global part(could be 0)
    for start_n in range(0, global_end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_slice = start_n + offs_n_base_slice
        
        # load k, v 
        if EVEN_N:
            k = tl.load(k_ptrs + start_n * stride_kn, cache_modifier=".cg")
            v = tl.load(v_ptrs + start_n * stride_vn, cache_modifier=".cg")
        else:
            mask_n = offs_n_slice < actual_seqlen_k  
            k = tl.load(k_ptrs + start_n * stride_kn, mask=mask_n, other=0.0, cache_modifier=".cg") # (BLOCK_HEADDIM, BLOCK_N)
            v = tl.load(v_ptrs + start_n * stride_vn, mask=mask_n, other=0.0, cache_modifier=".cg")

        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, qk, input_precision="tf32")
        
        # no need to mask for EVEN_N in `global` case
        # if IS_GLOBAL:
        qk += tl.where( # True will not mask
            (offs_m_slice >= offs_n_slice) &
            ((SEQOFFSET + offs_m_slice <= offs_n_slice) | (offs_n_slice < window_size_global))
            , 0, float("-inf"))
        
        # -- compute p ---
        p = tl.math.exp2(qk * qk_scale - l_slice_scale)
        # compute dq = dot(p, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp = tl.dot(do.to(q.dtype), v, dp, input_precision="tf32")

        ds = (p * (dp - Di[:, None])).to(q.dtype)
        dq = tl.dot(ds, tl.trans(k), dq)

    local_start_n = tl.maximum(((start_m * BLOCK_M + SEQOFFSET) // BLOCK_N) * BLOCK_N, global_end_n)
    local_start_n = tl.multiple_of(local_start_n, BLOCK_N)
    end_n = tl.minimum((start_m + 1) * BLOCK_M, actual_seqlen_k)
    for start_n in range(local_start_n, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_slice = start_n + offs_n_base_slice
        
        # load k, v 
        if EVEN_N:
            k = tl.load(k_ptrs + start_n * stride_kn, cache_modifier=".cg")
            v = tl.load(v_ptrs + start_n * stride_vn, cache_modifier=".cg")
        else:
            mask_n = offs_n_slice < actual_seqlen_k  
            k = tl.load(k_ptrs + start_n * stride_kn, mask=mask_n, other=0.0, cache_modifier=".cg") # (BLOCK_HEADDIM, BLOCK_N)
            v = tl.load(v_ptrs + start_n * stride_vn, mask=mask_n, other=0.0, cache_modifier=".cg")

        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, qk, input_precision="tf32")
        
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where(offs_n_slice < actual_seqlen_k, 0, float("-inf"))
        # if IS_GLOBAL:
        qk += tl.where( # True will not mask
            (offs_m_slice >= offs_n_slice) & ### $
            ((SEQOFFSET + offs_m_slice <= offs_n_slice)) # `local` part so we need not to (start_n + offs_n < window_size_global)
            , 0, float("-inf"))
        
        # -- compute p ---
        p = tl.math.exp2(qk * qk_scale - l_slice_scale)
        # compute dq = dot(p, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp = tl.dot(do.to(q.dtype), v, dp, input_precision="tf32")

        ds = (p * (dp - Di[:, None])).to(q.dtype)
        dq = tl.dot(ds, tl.trans(k), dq)
            
    dq *= softmax_scale
    if EVEN_M:
        tl.store(dq_ptrs, dq.to(q.dtype), cache_modifier=".cs")
    else:
        tl.store(dq_ptrs, dq.to(q.dtype), mask=offs_m_slice < actual_seqlen_q, cache_modifier=".cs")

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias=None, causal=False, softmax_scale=None, window_size=(-1,-1,-1),
                ):
        """
        q: ((b s), nheads, headdim)
        k, v: ((b s) nheads, headdim)
        bias: deleted.
        """
        # Make sure that the last dimension is contiguous
        batch, seqlen_q, num_heads, d = q.shape
        _, seqlen_k, num_heads_k, dk = k.shape
        assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
        assert num_heads % num_heads_k == 0, "num_heads must be divisible by num_heads_k"
        assert d == dk and dk == v.shape[-1] and num_heads_k == v.shape[-2], "num_heads and head dimensions must match"
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
        assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
        assert q.is_cuda and k.is_cuda and v.is_cuda, "All tensors must be sent to GPU"
        # I choose not to support the case where `d` is not a power of 2, 
        # which is sufficient for current LLM and simplifies the mask for load/store.
        assert d in {16, 32, 64, 128}, "Only support d in {16, 32, 64, 128}"

        # In training, `window_size_left + window_size_global` is never larger than `max_seqlen_k`.
        # in training, causal is always true.
        # causal=True
        softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
        ctx.softmax_scale = softmax_scale

        window_size_left = window_size[1] if window_size[1] >= 0 and window_size[1] <= seqlen_k else seqlen_k
        window_size_global = window_size[0] if window_size[0] > 0 and window_size[0] < seqlen_k else 0
        SEQOFFSET = seqlen_k - seqlen_q - window_size_left

        ctx.window_size_global, ctx.window_size_left, ctx.SEQOFFSET = window_size_global, window_size_left, SEQOFFSET

        seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
        lse = torch.empty((batch, num_heads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
        o = torch.empty_like(q)
        # using 3d grid to avoid div & rem
        grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), num_heads, batch)
        _fwd_kernel[grid](
            q,
            k,
            v,
            o,
            lse,
            softmax_scale,
            q.stride(0),
            q.stride(2),
            q.stride(1),
            k.stride(0),
            k.stride(2),
            k.stride(1),
            v.stride(0),
            v.stride(2),
            v.stride(1),
            o.stride(0),
            o.stride(2),
            o.stride(1),
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            num_heads,
            num_heads_k,
            window_size_global, # window_size_global
            SEQOFFSET, # window_size_left
            # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
            d, # d_rounded (BLOCK_HEADDIM) actually
            # BLOCK_M=128,
            # BLOCK_N=64,
            # num_warps=num_warps,
            # num_stages=1,
        )

        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            if do.stride(-1) != 1: do = do.contiguous()
            batch, seqlen_q, num_heads, d = q.shape
            _, seqlen_k, num_heads_k, _ = k.shape
            kv_group_size = num_heads // num_heads_k
            delta = torch.empty_like(lse)
            max_seqlen_q_rounded = lse.shape[-1]

            grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), num_heads, batch)
            _bwd_preprocess_do_o_dot[grid](
                o,
                do,
                delta,
                o.stride(0),
                o.stride(2),
                o.stride(1),
                do.stride(0),
                do.stride(2),
                do.stride(1),
                seqlen_q,
                max_seqlen_q_rounded,
                d,
            )
            dk_expanded = torch.empty((batch, seqlen_k, num_heads, d), dtype=do.dtype, device=do.device)
            dv_expanded = torch.empty((batch, seqlen_k, num_heads, d), dtype=do.dtype, device=do.device)
            grid = lambda META: (triton.cdiv(seqlen_k, META["BLOCK_N"]), num_heads, batch)
            _bwd_dk_dv_kernel[grid](
                q,
                k,
                v,
                do,
                dk_expanded,
                dv_expanded,
                lse,
                delta,
                ctx.softmax_scale,
                q.stride(0),
                q.stride(2),
                q.stride(1),
                k.stride(0),
                k.stride(2),
                k.stride(1),
                v.stride(0),
                v.stride(2),
                v.stride(1),
                do.stride(0),
                do.stride(2),
                do.stride(1),
                dk_expanded.stride(0),
                dk_expanded.stride(2),
                dk_expanded.stride(1),
                dv_expanded.stride(0),
                dv_expanded.stride(2),
                dv_expanded.stride(1),
                num_heads,
                num_heads_k,
                seqlen_q,
                seqlen_k,
                max_seqlen_q_rounded,
                ctx.window_size_global,
                ctx.SEQOFFSET,
                d, #BLOCK_HEADDIM=d,
                # BLOCK_M=128, BLOCK_N=128,
                # num_warps=8,
                # num_stages=1,
            )
            dk = dk_expanded.reshape(batch, seqlen_k, num_heads_k, kv_group_size, d).sum(dim=3, keepdim=False)
            dv = dv_expanded.reshape(batch, seqlen_k, num_heads_k, kv_group_size, d).sum(dim=3, keepdim=False)
            dq = torch.zeros_like(q)
            grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), num_heads, batch)
            _bwd_dq_kernel[grid](
                q,
                k,
                v,
                do,
                dq,
                lse,
                delta,
                ctx.softmax_scale,
                q.stride(0),
                q.stride(2),
                q.stride(1),
                k.stride(0),
                k.stride(2),
                k.stride(1),
                v.stride(0),
                v.stride(2),
                v.stride(1),
                do.stride(0),
                do.stride(2),
                do.stride(1),
                dq.stride(0),
                dq.stride(2),
                dq.stride(1),
                num_heads,
                num_heads_k,
                seqlen_q,
                seqlen_k,
                max_seqlen_q_rounded,
                ctx.window_size_global,
                ctx.SEQOFFSET,
                d, #BLOCK_HEADDIM=d,
                # BLOCK_M=128, BLOCK_N=128,
                # num_warps=8,
                # num_stages=1,
            )

        # This is how many gradients you have to return as many arguments forward
        return dq, dk, dv, None, None, None, None, None, None, None, None

flash_attn_func = FlashAttnFunc.apply
