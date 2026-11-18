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
"""

import torch
import torch.nn.functional as F

from .tree_gated_delta_rule import (
    backward_states,
    chunk_bwd_dqkwg,
    chunk_bwd_dv_local,
    chunk_fwd_o,
    chunk_local_cumsum_scalar,
    forward_states,
    tree_plan,
)


def tree_mamba2_chunk_scan(x, dt, A, B, C, D, dt_bias, dt_limit, tree):
    """
    Mamba2's SSD over a packed tree, from the inputs of transformers' `mamba2_chunk_scan`: `(1, tokens, heads,
    head_dim)` inputs `x`, `(1, tokens, heads)` raw time steps `dt`, `(1, tokens, groups, state)` `B` and `C`.
    """
    dt = F.softplus(dt + dt_bias.to(dt.dtype)).clamp(*dt_limit).float()
    v = (x.float() * dt[..., None]).to(x.dtype)
    y = TreeSSD.apply(C, B, v, A * dt, tree_plan(tree))
    return y + D[:, None] * x.float()


class TreeSSD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, tree):
        q, k, v, g = [x.contiguous() for x in (q, k, v, g)]
        layout = dict(cu_seqlens=tree["tree_cu_seqlens_q"], chunk_indices=tree["tree_chunk_indices"])
        g = chunk_local_cumsum_scalar(g, scale=1.4426950408889634, **layout)
        h, _ = forward_states(k, None, v, g, tree)
        out = chunk_fwd_o(q, k, v, h, g=g, scale=1.0, **layout)
        ctx.save_for_backward(q, k, v, g)
        ctx.tree = tree
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, g = ctx.saved_tensors
        tree = ctx.tree
        do = do.contiguous()
        layout = dict(cu_seqlens=tree["tree_cu_seqlens_q"], chunk_indices=tree["tree_chunk_indices"])
        h, _ = forward_states(k, None, v, g, tree)
        dv = chunk_bwd_dv_local(q, k, do, g=g, scale=1.0, **layout)
        dh, dv = backward_states(q, k, None, g, do, dv, tree, 1.0)
        dq, dk, _, dg = chunk_bwd_dqkwg(q, k, v, do, h, dh, w=None, g=g, dv=None, scale=1.0, **layout)
        dg = chunk_local_cumsum_scalar(dg, reverse=True, **layout)
        return dq.to(q), dk.to(k), dv.to(v), dg, None
