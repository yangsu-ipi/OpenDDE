# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Anthropic, PBC
# Follows the arithmetic of OpenDDE's DiffusionTransformer (Aureka AI
# Research, Apache-2.0, derived from Protenix, ByteDance, Apache-2.0); see
# NOTICE.
"""The sampler's token transformer as one fused fp32 stack (dit_fused).

Per block, stock runs an adaptive LayerNorm, four attention projections, the
pair-biased attention, a sigmoid gate, the output projection and gate, a
residual add, a second adaptive LayerNorm, two transition projections, SiLU
times the second, the last projection, its gate and a residual add, each as
its own GEMM or elementwise pass. This stack keeps the fp32 residual stream
and the stock attention call (``F.scaled_dot_product_attention`` on fp32
operands with the pair bias, scale 1, q divided by sqrt(c_hidden)) and the
step-invariant pair biases of ``dit_hoist``, and changes only:

- k|v|g in one GEMM;
- Triton row kernels for AdaLN, the attention gate, SiLU-multiply, and the
  output gate with the residual add fused with the next AdaLN.

The q, a1 and a2 projections and the conditioning stay stock calls: on the
RTX PRO 6000, merging them (as the kit's stack does) gets cuBLAS kernels that
round less accurately than stock's from about 2,000 tokens. With the merges
kept here the error against fp64 is within 1.001x of stock's at 86-2,300
tokens (kernel benchmark, 2026-10-21). The LayerNorm statistics and k|v|g
GEMM tiling differ, so rounding differs from stock (fast only). Forward only,
fp32, no autocast; any other call runs the stock forward.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from opendde.model import step_invariant
from opendde.model.optimized import dispatch

SITE = "dit_fused"

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001 -- Triton is optional: the stock op serves
    triton = None

C_A, C_S, HEADS, C_HIDDEN = 768, 384, 16, 48

if triton is not None:

    @triton.jit
    def _sigmoid(x):
        return 1.0 / (1.0 + tl.exp(-x))

    @triton.jit
    def _adaln_kernel(A, X1, X2, OUT, Ns, C, s_a, s_x, s_o, eps, BLOCK: tl.constexpr):
        r = tl.program_id(0).to(tl.int64)
        rs = r % Ns
        j = tl.arange(0, BLOCK)
        m = j < C
        a = tl.load(A + r * s_a + j, mask=m, other=0.0)
        mean = tl.sum(a, axis=0) / C
        d = tl.where(m, a - mean, 0.0)
        var = tl.sum(d * d, axis=0) / C
        an = d / tl.sqrt(var + eps)
        x1 = tl.load(X1 + rs * s_x + j, mask=m, other=0.0)
        x2 = tl.load(X2 + rs * s_x + j, mask=m, other=0.0)
        tl.store(OUT + r * s_o + j, _sigmoid(x1) * an + x2, mask=m)

    @triton.jit
    def _gate_kernel(
        O, G, OUT, N, Cd, HC, s_ob, s_oh, s_on, s_g, s_out, BLOCK: tl.constexpr
    ):
        r = tl.program_id(0).to(tl.int64)
        b = r // N
        n = r % N
        j = tl.arange(0, BLOCK)
        m = j < HC
        h = (j // Cd).to(tl.int64)
        c = j % Cd
        o = tl.load(O + b * s_ob + h * s_oh + n * s_on + c, mask=m, other=0.0)
        g = tl.load(G + r * s_g + j, mask=m, other=0.0)
        tl.store(OUT + r * s_out + j, o * _sigmoid(g), mask=m)

    @triton.jit
    def _swiglu_kernel(X1, X2, OUT, F_, s_x, s_o, BLOCK: tl.constexpr):
        r = tl.program_id(0).to(tl.int64)
        j = tl.arange(0, BLOCK)
        m = j < F_
        x1 = tl.load(X1 + r * s_x + j, mask=m, other=0.0)
        x2 = tl.load(X2 + r * s_x + j, mask=m, other=0.0)
        tl.store(OUT + r * s_o + j, x1 * _sigmoid(x1) * x2, mask=m)

    @triton.jit
    def _resgate_adaln_kernel(
        GL, X, RES, X1, X2, AN, Ns, C, s_gl, s_x, s_res, s_xx, s_an, eps,
        HAS_NEXT: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        # RES[r] += sigmoid(GL[r % Ns]) * X[r]; then AN[r] = AdaLN(RES[r])
        r = tl.program_id(0).to(tl.int64)
        rs = r % Ns
        j = tl.arange(0, BLOCK)
        m = j < C
        gl = tl.load(GL + rs * s_gl + j, mask=m, other=0.0)
        x = tl.load(X + r * s_x + j, mask=m, other=0.0)
        res = tl.load(RES + r * s_res + j, mask=m, other=0.0)
        a = tl.where(m, _sigmoid(gl) * x + res, 0.0)
        tl.store(RES + r * s_res + j, a, mask=m)
        if HAS_NEXT:
            mean = tl.sum(a, axis=0) / C
            d = tl.where(m, a - mean, 0.0)
            var = tl.sum(d * d, axis=0) / C
            an = d / tl.sqrt(var + eps)
            x1 = tl.load(X1 + rs * s_xx + j, mask=m, other=0.0)
            x2 = tl.load(X2 + rs * s_xx + j, mask=m, other=0.0)
            tl.store(AN + r * s_an + j, _sigmoid(x1) * an + x2, mask=m)


def _adaln(a, x1, x2, eps):
    M, C = a.shape
    out = torch.empty_like(a)
    _adaln_kernel[(M,)](
        a, x1, x2, out, x1.shape[0], C, a.stride(0), x1.stride(0), out.stride(0),
        eps, BLOCK=triton.next_power_of_2(C), num_warps=4,
    )  # fmt: skip
    return out


def _gate(o, g, N):
    B, H, _, Cd = o.shape
    M, HC = g.shape
    out = torch.empty((M, HC), dtype=o.dtype, device=o.device)
    _gate_kernel[(M,)](
        o, g, out, N, Cd, HC, o.stride(0), o.stride(1), o.stride(2), g.stride(0),
        out.stride(0), BLOCK=triton.next_power_of_2(HC), num_warps=4,
    )  # fmt: skip
    return out


def _swiglu(x1, x2):
    M, F_ = x1.shape
    out = torch.empty_like(x1)
    _swiglu_kernel[(M,)](
        x1, x2, out, F_, x1.stride(0), out.stride(0),
        BLOCK=triton.next_power_of_2(F_), num_warps=4,
    )  # fmt: skip
    return out


def _resgate_adaln(gl, x, res, x1, x2, eps):
    """In place res += sigmoid(gl) * x; returns AdaLN(res) by x1, x2, or None
    when x1 is None (the last block)."""
    M, C = x.shape
    has_next = x1 is not None
    an = torch.empty_like(res) if has_next else res
    x1 = x1 if has_next else gl
    x2 = x2 if has_next else gl
    _resgate_adaln_kernel[(M,)](
        gl, x, res, x1, x2, an, gl.shape[0], C, gl.stride(0), x.stride(0),
        res.stride(0), x1.stride(0), an.stride(0), eps, HAS_NEXT=has_next,
        BLOCK=triton.next_power_of_2(C), num_warps=4,
    )  # fmt: skip
    return an if has_next else None


class FusedTokenStack:
    """The packed weights of one DiffusionTransformer and its fused forward."""

    def __init__(self, dt) -> None:
        self.blocks = list(dt.blocks)
        W = lambda lin: lin.weight.detach().float()  # noqa: E731
        with torch.no_grad():
            self.w_q, self.b_q, self.w_kvg = [], [], []
            self.w_o, self.w_b = [], []
            for blk in self.blocks:
                apb, ctb = blk.attention_pair_bias, blk.conditioned_transition_block
                att = apb.attention
                self.w_q.append(W(att.linear_q))
                self.b_q.append(att.linear_q.bias.detach().float())
                self.w_kvg.append(
                    torch.cat(
                        [W(att.linear_k), W(att.linear_v), W(att.linear_g)]
                    ).contiguous()
                )
                self.w_o.append(W(att.linear_o).contiguous())
                self.w_b.append(W(ctb.linear_nobias_b).contiguous())
        first = self.blocks[0]
        self.eps_a = float(first.attention_pair_bias.layernorm_a.layernorm_a.eps)

    def _bias(self, apb, a, z, extra_attn_bias, enable_efficient_fusion):
        # The pair bias exactly as stock's standard_multihead_attention gets it.
        memo = step_invariant.active("dit_hoist")
        key = (z, extra_attn_bias, enable_efficient_fusion, a.dim())
        produce = lambda: apb._standard_pair_bias(  # noqa: E731
            a, z, extra_attn_bias, enable_efficient_fusion
        )
        return produce() if memo is None else memo.get(apb, "pair_bias", key, produce)

    @torch.autocast("cuda", enabled=False)
    def forward(self, a, s, z, enable_efficient_fusion, extra_attn_bias):
        S, N = a.shape[-3], a.shape[-2]
        M, Ns = S * N, s.shape[-3] * N
        A = a.reshape(M, C_A).float().clone()
        s2 = s.reshape(Ns, C_S).float()
        nb = len(self.blocks)
        # The conditioning as stock computes it (its GEMMs get split-K kernels
        # that a packed conditioning GEMM does not get at large sizes).
        cond = []
        for blk in self.blocks:
            apb, ctb = blk.attention_pair_bias, blk.conditioned_transition_block
            terms = []
            for ada in (apb.layernorm_a, ctb.adaln):
                s_ln = ada.layernorm_s(s2)
                terms += [ada.linear_s(s_ln), ada.linear_nobias_s(s_ln)]
            cond.append(terms + [apb.linear_a_last(s2), ctb.linear_s(s2)])
        scale = math.sqrt(C_HIDDEN)
        an = _adaln(A, cond[0][0], cond[0][1], self.eps_a)
        for i, blk in enumerate(self.blocks):
            x1, x2, y1, y2, g_att, g_ff = cond[i]
            apb = blk.attention_pair_bias
            bias = self._bias(apb, a, z, extra_attn_bias, enable_efficient_fusion)
            # q as stock computes it (cuBLAS picks a split-K kernel for this
            # biased projection, which rounds more accurately than the kernel
            # a merged projection gets); k|v|g in one GEMM.
            q = F.linear(an, self.w_q[i], self.b_q[i])
            kvg = torch.mm(an, self.w_kvg[i].t())
            q, k, v = (
                t.view(S, N, HEADS, C_HIDDEN).transpose(-2, -3)
                for t in (q, kvg[:, :C_A], kvg[:, C_A : 2 * C_A])
            )
            q = q / scale
            if bias.dim() != q.dim():
                bias = bias.unsqueeze(dim=-3)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)
            og = _gate(o, kvg[:, 2 * C_A :], N)
            x = torch.mm(og, self.w_o[i].t())
            an2 = _resgate_adaln(g_att, x, A, y1, y2, self.eps_a)
            # a1 and a2 as stock computes them: a merged a1|a2 GEMM rounds
            # less accurately from about 2,000 tokens.
            ctb = blk.conditioned_transition_block
            hb = _swiglu(ctb.linear_nobias_a1(an2), ctb.linear_nobias_a2(an2))
            xb = torch.mm(hb, self.w_b[i].t())
            nxt = cond[i + 1] if i + 1 < nb else (None, None)
            an = _resgate_adaln(g_ff, xb, A, nxt[0], nxt[1], self.eps_a)
        return A.view(a.shape)


def eligible(dt, a, s, n_queries, n_keys) -> bool:
    if triton is None:
        return dispatch.fallback(SITE, "triton unavailable")
    if torch.is_grad_enabled():
        return dispatch.fallback(SITE, "autograd enabled")
    if n_queries is not None or n_keys is not None:
        return False  # local attention: the atom transformers' call form
    if a.dim() != 3 or s.dim() != 3 or a.shape[-1] != C_A or s.shape[-1] != C_S:
        return dispatch.fallback(SITE, f"shapes {tuple(a.shape)} {tuple(s.shape)}")
    if s.shape[0] not in (1, a.shape[0]) or s.shape[1] != a.shape[1]:
        return dispatch.fallback(SITE, f"shapes {tuple(a.shape)} {tuple(s.shape)}")
    if not a.is_cuda or a.dtype != torch.float32 or s.dtype != torch.float32:
        return dispatch.fallback(SITE, f"input {a.dtype} on {a.device.type}")
    apb = dt.blocks[0].attention_pair_bias
    if apb._foldcp_diffusion_bias_row_chunk_size() > 0:
        return dispatch.fallback(SITE, "Fold-CP bias streaming")
    return dispatch.available(SITE, a.device)


def stack(dt) -> FusedTokenStack:
    """The fused stack of ``dt``, built once (inference weights are fixed)."""
    st = dt.__dict__.get("_opendde_fused_stack")
    if st is None:
        st = FusedTokenStack(dt)
        dt.__dict__["_opendde_fused_stack"] = st
    return st
