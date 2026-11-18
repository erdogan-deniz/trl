# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mamba2's chunked scan (SSD) over a tree-packed row, run by the tree kernels of the gated delta rule.

SSD is gated linear attention with a scalar decay per head (https://huggingface.co/papers/2405.21060): the state `S_t =
exp(dt_t A) S_{t-1} + B_t (dt_t x_t)^T` is read out as `y_t = C_t^T S_t + D x_t`. With queries `C`, keys `B`, values
`dt x` and log-gates `dt A`, that is the gated delta rule without its delta (WY) term, so the same chunk and tree-state
kernels run it: each segment starts from its parent's final state, and in backward sums its children's state gradients.
SSD's own kernels compute the elementwise `dt x` and `D x` terms, and the query and key gradients, summed over the
heads that share a group of `B` and `C`.
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .tree_gated_delta_rule import (
    IS_NVIDIA_HOPPER,
    TRITON_ABOVE_3_4_0,
    TRITON_ABOVE_3_7_1,
    backward_states,
    chunk_bwd_dv_local,
    chunk_fwd_o,
    chunk_local_cumsum_scalar,
    exp2,
    forward_states,
    tree_plan,
)


def tree_mamba2_chunk_scan(
    scan, hidden_states, dt, A, B, C, chunk_size, D=None, dt_bias=None, dt_limit=(0.0, math.inf), **tree
):
    """
    Tree version of transformers' `mamba2_chunk_scan` (`scan`, unused): `(1, tokens, heads, head_dim)` inputs
    `hidden_states`, `(1, tokens, heads)` raw time steps `dt`, `(1, tokens, groups, state)` `B` and `C`.
    """
    dt = F.softplus(dt + dt_bias.to(dt.dtype)).clamp(*dt_limit).float()
    return TreeSSD.apply(C, B, hidden_states, dt, A * dt, D, tree_plan(tree))


def tree_mamba2_split_conv1d_scan_combined(fused, *args, **tree):
    """
    Tree version of transformers' fused `mamba2_split_conv1d_scan_combined`: `None`, what transformers' own fallback
    returns without `mamba_ssm`, so the layer runs unfused, through the conv and scan above.
    """
    return None


class TreeSSD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, x, dt, g, D, tree):
        q, k, x, dt, g = [t.contiguous() for t in (q, k, x, dt, g)]
        layout = dict(cu_seqlens=tree["tree_cu_seqlens_q"], chunk_indices=tree["tree_chunk_indices"])
        g = chunk_local_cumsum_scalar(g, scale=1.4426950408889634, **layout)
        v = scale_values(x, dt)
        h, _ = forward_states(k, None, v, g, tree)
        out = chunk_fwd_o(q, k, v, h, g=g, scale=1.0, **layout)
        add_skip(out, x, D)
        ctx.save_for_backward(q, k, x, dt, g, D)
        ctx.tree = tree
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, x, dt, g, D = ctx.saved_tensors
        tree = ctx.tree
        do = do.contiguous()
        layout = dict(cu_seqlens=tree["tree_cu_seqlens_q"], chunk_indices=tree["tree_chunk_indices"])
        v = scale_values(x, dt)
        h, _ = forward_states(k, None, v, g, tree)
        dv = chunk_bwd_dv_local(q, k, do, g=g, scale=1.0, **layout)
        dh, dv = backward_states(q, k, None, g, do, dv, tree, 1.0)
        dq, dk, dg = grouped_bwd_dqkg(q, k, v, g, h, do, dh, **layout)
        dg = chunk_local_cumsum_scalar(dg, reverse=True, **layout)
        dx, ddt, dD = skip_bwd(do, dv, x, dt, D)
        return dq, dk, dx, ddt, dg, dD, None


# --- Elementwise terms: values `dt x` and skip `D x` ---


@triton.jit(do_not_specialize=["T"])
def scale_values_kernel(x, dt, v, T, HV: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    m = (o_t < T)[:, None] & (o_v < V)[None, :]
    p = o_t[:, None].to(tl.int64) * (HV * V) + i_h * V + o_v[None, :]
    b_dt = tl.load(dt + o_t * HV + i_h, mask=o_t < T, other=0.0)
    b_x = tl.load(x + p, mask=m, other=0.0).to(tl.float32)
    tl.store(v + p, (b_x * b_dt[:, None]).to(v.dtype.element_ty), mask=m)


@triton.jit(do_not_specialize=["T"])
def add_skip_kernel(o, x, D, T, HV: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    m = (o_t < T)[:, None] & (o_v < V)[None, :]
    p = o_t[:, None].to(tl.int64) * (HV * V) + i_h * V + o_v[None, :]
    b_x = tl.load(x + p, mask=m, other=0.0).to(tl.float32)
    b_o = tl.load(o + p, mask=m, other=0.0).to(tl.float32)
    tl.store(o + p, (b_o + tl.load(D + i_h).to(tl.float32) * b_x).to(o.dtype.element_ty), mask=m)


@triton.jit(do_not_specialize=["T"])
def skip_bwd_kernel(
    do, dv, x, dt, D, dx, ddt, dD, T, HV: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    m_t = o_t < T
    m = m_t[:, None] & (o_v < V)[None, :]
    p = o_t[:, None].to(tl.int64) * (HV * V) + i_h * V + o_v[None, :]
    b_x = tl.load(x + p, mask=m, other=0.0).to(tl.float32)
    b_do = tl.load(do + p, mask=m, other=0.0).to(tl.float32)
    b_dv = tl.load(dv + p, mask=m, other=0.0).to(tl.float32)
    b_dt = tl.load(dt + o_t * HV + i_h, mask=m_t, other=0.0)
    b_D = tl.load(D + i_h).to(tl.float32)
    tl.store(dx + p, (b_dv * b_dt[:, None] + b_D * b_do).to(dx.dtype.element_ty), mask=m)
    tl.store(ddt + o_t * HV + i_h, tl.sum(b_dv * b_x, axis=1), mask=m_t)
    tl.store(dD + i_t * HV + i_h, tl.sum(b_do * b_x))


def elementwise_grid(x):
    _, tokens, value_heads, value_dim = x.shape
    # Disable FP fusion, as Triton 3.10 does by default (see #8136)
    meta = dict(
        T=tokens, HV=value_heads, V=value_dim, BT=64, BV=triton.next_power_of_2(value_dim), enable_fp_fusion=False
    )
    return (triton.cdiv(tokens, 64), value_heads), meta


def scale_values(x, dt):
    v = torch.empty_like(x)
    grid, meta = elementwise_grid(x)
    scale_values_kernel[grid](x, dt, v, **meta)
    return v


def add_skip(o, x, D):
    grid, meta = elementwise_grid(x)
    add_skip_kernel[grid](o, x, D, **meta)


def skip_bwd(do, dv, x, dt, D):
    grid, meta = elementwise_grid(x)
    dx = torch.empty_like(x)
    ddt = torch.empty_like(dt)
    dD = dt.new_empty(grid)
    skip_bwd_kernel[grid](do, dv, x, dt, D, dx, ddt, dD, **meta)
    return dx, ddt, dD.sum(0).to(D.dtype)


# --- Query and key gradients, summed over the heads of a group ---


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps in [4, 8] for num_stages in [2, 3]
    ],
    key=["H", "HV", "K", "V", "BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def grouped_bwd_dqkg_kernel(
    q,
    k,
    v,
    g,
    h,
    do,
    dh,
    dq,
    dk,
    dg,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HP: tl.constexpr,
):
    i_k, i_tg, i_hs = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    S: tl.constexpr = HV // H // HP
    i_hq, i_s = i_hs // S, i_hs % S
    dg += i_k.to(tl.int64) * T * HV
    dq += i_s.to(tl.int64) * T * H * K
    dk += i_s.to(tl.int64) * T * H * K
    i_n, i_t = tl.load(chunk_indices + i_tg * 2).to(tl.int32), tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int32)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = (eos - bos).to(tl.int32)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m_qk = m_t[:, None] & m_k[None, :]
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    last = min(i_t * BT + BT, T) - 1

    p_qk = (bos + o_t[:, None]) * (H * K) + i_hq * K + o_k[None, :]
    b_q = tl.load(q + p_qk, mask=m_qk, other=0.0)
    b_k = tl.load(k + p_qk, mask=m_qk, other=0.0)
    b_dq_sum = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk_sum = tl.zeros([BT, BK], dtype=tl.float32)

    for i_g in range(HP):
        i_h = (i_hq * S + i_s) * HP + i_g
        b_dg_last = tl.zeros([1], dtype=tl.float32)
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_ds = tl.zeros([BT, BT], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_v = o_v < V
            m_tv = m_t[:, None] & m_v[None, :]
            p_tv = (bos + o_t[:, None]) * (HV * V) + i_h * V + o_v[None, :]
            p_kv = (i_tg * HV + i_h) * K * V + o_v[:, None] + o_k[None, :] * V
            m_kv = m_v[:, None] & m_k[None, :]
            b_v = tl.load(v + p_tv, mask=m_tv, other=0.0)
            b_do = tl.load(do + p_tv, mask=m_tv, other=0.0)
            b_h = tl.load(h + p_kv, mask=m_kv, other=0.0)
            b_dh = tl.load(dh + p_kv, mask=m_kv, other=0.0)
            b_dg_last += tl.sum(b_h * b_dh)
            b_ds += tl.dot(b_do, tl.trans(b_v))
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

        b_g = tl.load(g + (bos + o_t) * HV + i_h, mask=m_t, other=0.0)
        b_g_last = tl.load(g + (bos + last) * HV + i_h)
        b_dg_last *= exp2(b_g_last)
        b_dq = b_dq * exp2(b_g)[:, None]
        b_dk = b_dk * tl.where(m_t, exp2(-b_g + b_g_last), 0)[:, None]
        b_dg_last += tl.sum(b_dk * b_k)
        b_ds = tl.where(m_A, b_ds * exp2(b_g[:, None] - b_g[None, :]), 0).to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        b_dg = tl.sum(b_dq * b_q, axis=1) - tl.sum(b_dk * b_k, axis=1)
        b_dg = tl.where(o_t < last, b_dg, b_dg + b_dg_last)
        tl.store(dg + (bos + o_t) * HV + i_h, b_dg, mask=m_t)
        b_dq_sum += b_dq
        b_dk_sum += b_dk

    tl.store(dq + p_qk, b_dq_sum.to(dq.dtype.element_ty), mask=m_qk)
    tl.store(dk + p_qk, b_dk_sum.to(dk.dtype.element_ty), mask=m_qk)


def grouped_bwd_dqkg(q, k, v, g, h, do, dh, cu_seqlens, chunk_indices):
    if IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0 and not TRITON_ABOVE_3_7_1:
        raise RuntimeError(
            "Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect results for gated chunk_bwd_dqkwg (see "
            "#640). Please upgrade Triton to >= 3.7.1."
        )
    _, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    BK = min(max(triton.next_power_of_2(key_dim), 16), 64)
    BV = min(max(triton.next_power_of_2(value_dim), 16), 64)
    NK = triton.cdiv(key_dim, BK)
    HP = math.gcd(value_heads // heads, 8)
    NS = value_heads // heads // HP
    dq = q.new_empty(NS, *q.shape, dtype=q.dtype if NS == 1 else torch.float32)
    dk = torch.empty_like(dq)
    dg = g.new_empty(NK, *g.shape)
    # Disable FP fusion, as Triton 3.10 does by default (see #8136)
    grouped_bwd_dqkg_kernel[(NK, len(chunk_indices), value_heads // HP)](
        q, k, v, g, h, do, dh, dq, dk, dg, cu_seqlens, chunk_indices,
        T=tokens, H=heads, HV=value_heads, K=key_dim, V=value_dim, BT=64, BK=BK, BV=BV, HP=HP,
        enable_fp_fusion=False,
    )  # fmt: skip
    return dq.sum(0).to(q.dtype), dk.sum(0).to(k.dtype), dg.sum(0)
