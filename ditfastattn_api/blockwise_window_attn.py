import torch
import torch.nn.functional as F

# import pytest
import os

# os.environ["TRITON_INTERPRET"] = "1"
import math
import torch
from itertools import chain
import triton
import triton.language as tl
from time import time
import pdb


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [128]
    for BN in [128]
    # for s in ([1] if is_hip() else [3, 4, 7])
    for s in [3]
    for w in [8]
    # for w in [4]
]

# add N_CTXV as the sequence length of vision tokens
@triton.jit
def _headwise_attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    # dbg_block_ptr,
    left_window_size,
    right_window_size,
    cur_block_row_id,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    N_CTXV: tl.constexpr,
    fp8_v: tl.constexpr,
):
    # arrow_mask = tl.load(arrow_mask_ptr + tl.arange(0, N_CTXV // BLOCK_N))
    # tl.device_print("arrow_mask", arrow_mask[0])
    # n_rows = N_CTXV // BLOCK_M

    start_block_ind = tl.maximum((-left_window_size)//BLOCK_M +cur_block_row_id,0)
    end_block_ind=tl.minimum(right_window_size//BLOCK_M+cur_block_row_id, N_CTX//BLOCK_M)
    
    kv_movement_offset=BLOCK_N*start_block_ind
    K_block_ptr = tl.advance(K_block_ptr, (0, kv_movement_offset))
    V_block_ptr = tl.advance(V_block_ptr, (kv_movement_offset, 0))
    for col_block_ind in tl.range(start_block_ind, end_block_ind):
        
        # # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        # # ######################### debug #################################
        # # tl.store(dbg_block_ptr, qk)
        # # #################################################################

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        # -- update output accumulator -- 
        acc *= alpha[:, None]
        
        # update acc
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(V_block_ptr.dtype.element_ty)
        acc = tl.dot(p, v, acc)
        # -- update m_i and l_i
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    return acc, l_i, m_i
    
    


# @triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _blockwise_window_mm_attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    left_window_size_ptr,
    right_window_size_ptr,
    M,
    Out,  #
    # debug_ptr, # for debug output
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,  #
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,  #
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_vn: tl.constexpr,  #
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,  #
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    N_CTXV: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
):
    # tl.static_assert(BLOCK_N <= HEAD_DIM)
    # start_m = tl.program_id(0)
    # cur_row_id = tl.num_programs(0)
    cur_block_row_id = tl.program_id(0)
    off_hz = tl.program_id(1)
    # tl.device_print("pid", start_m)
    off_z = off_hz // H
    cur_head_id = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + cur_head_id.to(tl.int64) * stride_qh
    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(cur_block_row_id * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(cur_block_row_id * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # row_mask_len = N_CTXV // BLOCK_N
    left_window_size = tl.load(left_window_size_ptr + cur_head_id, eviction_policy="evict_last")
    right_window_size = tl.load(right_window_size_ptr + cur_head_id, eviction_policy="evict_last")

    # initialize offsets
    offs_m = cur_block_row_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_block_ptr)
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(K.dtype.element_ty)

    # tested, works fine
    acc, l_i, m_i = _headwise_attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,  #
        # dbg_block_ptr,
        left_window_size,
        right_window_size,
        cur_block_row_id,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,  #
        offs_m,
        offs_n,
        N_CTX,
        N_CTXV,
        V.dtype.element_ty == tl.float8e5,  #
        # V.dtype.element_ty == tl.float16,  #
    )
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sm_scale:float,
        vis_token_len: int,
        txt_token_len: int,
        left_window_size: torch.LongTensor,
        right_window_size: torch.LongTensor,
        block_size: int = 64,
    ):
        """
        arrow_mask: int32 Tensor [NHQ, N_CTXV//BLOCK_N + N_CTXV//BLOCK_N - 1]
        bs: batch_size
        nhq: n head of query
        hdq: head dim of query
        """
        bs, nhq, query_len, hdq = q.shape
        _, nhkv, token_len, hdk = k.shape
        _, nhkv, token_len, hdv = v.shape

        # shape constraints
        assert (
            query_len == token_len and query_len == txt_token_len + vis_token_len
        ), f"query_len={query_len} ctx_len={token_len} ctx_text={txt_token_len} ctx_vision={vis_token_len}"
        assert hdq == hdk and hdk == hdv
        assert hdk in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)

        # Lse
        L = torch.empty((bs, nhq, query_len), device=q.device, dtype=torch.float32)

        BLOCK_M = block_size
        BLOCK_N = block_size
        extra_kern_args = {}

        extra_kern_args.update(
            {
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
            }
        )

        grid = (triton.cdiv(query_len, block_size), bs * nhq, 1)

        # print(block_size,grid)
        _blockwise_window_mm_attn_fwd[grid](
            q,
            k,
            v,
            sm_scale,
            left_window_size,
            right_window_size,
            L,
            o,  #
            # debug,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),  #
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),  #
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),  #
            q.shape[0],
            q.shape[1],  #
            N_CTX=token_len,
            N_CTXV=vis_token_len,
            HEAD_DIM=hdk,  #
            **extra_kern_args,
        )
        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Backward pass is not implemented yet.")


def blockwise_window_mm_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    vis_token_len: int,
    txt_token_len: int,
    left_window_size: torch.LongTensor,
    right_window_size: torch.LongTensor,
    block_size: int = 64,
    sm_scale=None,
):
    """
    Compute the attention using the provided Triton kernel.
    Parameters:
    - q: Query tensor of shape (BATCH, N_HEAD, N_CTX, HEAD_DIM)
    - k: Key tensor of shape (BATCH, N_HEAD, N_CTX, HEAD_DIM)
    - v: Value tensor of shape (BATCH, N_HEAD, N_CTX, HEAD_DIM)
    - window_size: [N_HEAD]
    Returns:
    - o: Output tensor of shape (BATCH, N_HEAD, N_CTX, HEAD_DIM)

    TODO:
    accuracy debug
    support text tokens
    support IS_DIVISIBL
    support GQA
    support backward
    """
    assert txt_token_len == 0
    assert vis_token_len % block_size == 0
    # Ensure the input tensors are on the same device
    assert q.device == k.device == v.device, "All input tensors must be on the same device"
    # Ensure the input tensors have the correct shape
    assert q.shape == k.shape == v.shape, "Query, key, and value tensors must have the same shape"
    # Call the Triton kernel's forward method
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(k.shape[-1])

    o = _attention.apply(q, k, v, sm_scale, vis_token_len, txt_token_len, left_window_size, right_window_size, block_size)
    return o
    