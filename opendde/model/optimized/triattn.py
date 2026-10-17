# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Anthropic, PBC
# Adapted from the kit's flash triangle-attention kernel (k2b), whose
# arithmetic follows the cuEquivariance triangle_attention call it replaces;
# see NOTICE.
"""Flash triangle attention in Triton for optimization_mode fast (triattn_k2b).

Replaces the cuEquivariance ``triangle_attention`` call of ``Attention`` under
bf16 autocast at head dimension 32:

    out[b, i, h, q] = softmax_j(scale * q . k_j + bias[b, h, q, j]
                                (-1e9 where mask[b, i, j] is False)) @ v

bf16 tensor-core products with fp32 accumulation, an fp32 online softmax in
the base-2 domain, P cast to bf16 for the PV product. The result is not
bitwise the stock one: on sm_120 its error against an fp32 reference is the
same as stock's (kernel sweep, 2026-10-17). One fixed launch config, no
atomics, so a call gives the same bits on every run. Forward only.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from opendde.model.optimized import dispatch

SITE = "triattn_k2b"

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001 -- Triton is optional: the stock op serves
    triton = None

# The measured row (sm_120, bf16, D = 32, H = 12): the one config that was
# at least 1.33x faster than stock at every size from 128 to 2,047 tokens.
CONFIG = dict(BLOCK_M=64, BLOCK_N=64, ROWS=2, num_warps=4, num_stages=1)
HEAD_DIM = 32
HEADS = 12
MIN_TOKENS = 128
MAX_TOKENS = 2047
BIAS16_MIN_KEYS = 256  # below this a call is launch-bound: no 16-bit bias copy

if triton is not None:

    @triton.jit
    def _bias_prep(
        Bias,
        Out32,
        Out16,
        Flags,
        sbb,
        sbh,
        sbq,
        sbk,
        H,
        SQ,
        SK,
        SKp,
        NQB,
        PB: tl.constexpr,
        PK: tl.constexpr,
        MAKE16: tl.constexpr,
    ):
        # Out32 = the fp32 bias with unit key stride and a 16-element row
        # pitch (padding zeroed); with MAKE16 also its 16-bit copy and a flag
        # per program saying whether that cast was lossless.
        pid_q = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = (pid_bh // H).to(tl.int64)
        h = (pid_bh % H).to(tl.int64)
        rows = pid_q * PB + tl.arange(0, PB)
        rmask = rows < SQ
        src = Bias + b * sbb + h * sbh
        dst = pid_bh.to(tl.int64) * SQ * SKp
        nbad = tl.zeros([PB], dtype=tl.int32)
        for k0 in range(0, SK, PK):
            cols = k0 + tl.arange(0, PK)
            m2 = rmask[:, None] & (cols < SK)[None, :]
            w2 = rmask[:, None] & (cols < SKp)[None, :]
            x = tl.load(
                src + rows[:, None] * sbq + cols[None, :] * sbk, mask=m2, other=0.0
            ).to(tl.float32)
            o = dst + rows[:, None] * SKp + cols[None, :]
            tl.store(Out32 + o, x, mask=w2)
            if MAKE16:
                x16 = x.to(Out16.dtype.element_ty)
                tl.store(Out16 + o, x16, mask=w2)
                nbad += tl.sum((x16.to(tl.float32) != x).to(tl.int32), 1)
        if MAKE16:
            ok = (tl.sum(nbad, 0) == 0).to(tl.int32)
            tl.store(Flags + pid_bh * NQB + pid_q, ok)

    @triton.jit
    def _row_step(
        acc,
        m_i,
        l_i,
        q,
        k_base,
        v_base,
        m_base,
        k_lo,
        v_lo,
        m_lo,
        k_off,
        v_off,
        m_off,
        s_bias,
        kvm,
        qk_scale,
        MASK_N: tl.constexpr,
        APPLY_MASK: tl.constexpr,
    ):
        # One key tile for one row: logits in the base-2 domain (log2(e) is
        # folded into qk_scale and s_bias), online softmax update.
        if MASK_N:
            k = tl.load(k_base + k_lo + k_off, mask=kvm[:, None], other=0.0)
            v = tl.load(v_base + v_lo + v_off, mask=kvm[:, None], other=0.0)
        else:
            k = tl.load(k_base + k_lo + k_off)
            v = tl.load(v_base + v_lo + v_off)
        s = tl.dot(q, tl.trans(k))
        s = s * qk_scale + s_bias
        if APPLY_MASK:
            if MASK_N:
                mk = tl.load(m_base + m_lo + m_off, mask=kvm, other=1)
            else:
                mk = tl.load(m_base + m_lo + m_off)
            # stock adds -1e9 to masked logits; a select keeps a fully
            # masked row finite (it averages v, as stock does in fp32)
            s = tl.where(mk[None, :] != 0, s, -1.4426950408889634e9)
        if MASK_N:
            s = tl.where(kvm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # (s - m) is formed before any scaling, so an unchanged running max
        # gives alpha == 1 exactly and a fully masked row gives p == 1
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        return acc, m_new, l_new

    @triton.jit
    def _attend(
        q0,
        q1,
        kb0,
        kb1,
        vb0,
        vb1,
        mb0,
        mb1,
        bias_bh,
        b_off,
        k_off,
        v_off,
        m_off,
        offs_n,
        skk,
        svk,
        smk,
        SEQ_K,
        qk_scale,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ROWS: tl.constexpr,
        APPLY_MASK: tl.constexpr,
    ):
        acc0 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        m0 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l0 = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc1 = acc0
        m1 = m0
        l1 = l0
        kv_all = offs_n < BLOCK_N
        hi = (SEQ_K // BLOCK_N) * BLOCK_N
        for start_n in range(0, hi, BLOCK_N):
            s_bias = (
                tl.load(bias_bh + start_n + b_off).to(tl.float32) * 1.4426950408889634
            )
            k_lo = start_n * skk
            v_lo = start_n * svk
            m_lo = start_n * smk
            acc0, m0, l0 = _row_step(
                acc0, m0, l0, q0, kb0, vb0, mb0, k_lo, v_lo, m_lo, k_off, v_off,
                m_off, s_bias, kv_all, qk_scale, MASK_N=False, APPLY_MASK=APPLY_MASK,
            )  # fmt: skip
            if ROWS >= 2:
                acc1, m1, l1 = _row_step(
                    acc1, m1, l1, q1, kb1, vb1, mb1, k_lo, v_lo, m_lo, k_off, v_off,
                    m_off, s_bias, kv_all, qk_scale, MASK_N=False,
                    APPLY_MASK=APPLY_MASK,
                )  # fmt: skip
        if hi < SEQ_K:
            kvm = (hi + offs_n) < SEQ_K
            s_bias = (
                tl.load(bias_bh + hi + b_off, mask=kvm[None, :], other=0.0).to(
                    tl.float32
                )
                * 1.4426950408889634
            )
            k_lo = hi * skk
            v_lo = hi * svk
            m_lo = hi * smk
            acc0, m0, l0 = _row_step(
                acc0, m0, l0, q0, kb0, vb0, mb0, k_lo, v_lo, m_lo, k_off, v_off,
                m_off, s_bias, kvm, qk_scale, MASK_N=True, APPLY_MASK=APPLY_MASK,
            )  # fmt: skip
            if ROWS >= 2:
                acc1, m1, l1 = _row_step(
                    acc1, m1, l1, q1, kb1, vb1, mb1, k_lo, v_lo, m_lo, k_off, v_off,
                    m_off, s_bias, kvm, qk_scale, MASK_N=True, APPLY_MASK=APPLY_MASK,
                )  # fmt: skip
        return acc0, m0, l0, acc1, m1, l1

    @triton.jit
    def _flash_triattn_fwd(
        Q,
        K,
        V,
        Bias32,
        Bias16,
        Flags,
        Mask,
        Out,
        sqb,
        sqi,
        sqh,
        sqq,
        skb,
        ski,
        skh,
        skk,
        svb,
        svi,
        svh,
        svk,
        sbb,
        sbh,
        sbq,
        smb,
        smi,
        smk,
        sob,
        soi,
        soh,
        soq,
        N_ROWS,
        SEQ_Q,
        SEQ_K,
        H,
        n_flags,
        qk_scale,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ROWS: tl.constexpr,
        HAS_MASK: tl.constexpr,
        BIAS16: tl.constexpr,
        MSCAN: tl.constexpr,
        NFLAG: tl.constexpr,
    ):
        # grid: (query tiles, row groups of ROWS rows, B * H); each program
        # loads the bias tile once for its ROWS rows.
        pid_q = tl.program_id(0)
        pid_r = tl.program_id(1)
        pid_bh = tl.program_id(2)
        b = (pid_bh // H).to(tl.int64)
        h = (pid_bh % H).to(tl.int64)
        offs_m = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_mc = tl.minimum(offs_m, SEQ_Q - 1)  # loads clamped, stores masked
        offs_d = tl.arange(0, HEAD_DIM)
        offs_n = tl.arange(0, BLOCK_N)
        qk_scale2 = qk_scale * 1.4426950408889634
        k_off = offs_n[:, None] * skk + offs_d[None, :]
        v_off = offs_n[:, None] * svk + offs_d[None, :]
        m_off = offs_n * smk
        b_off = offs_mc[:, None] * sbq + offs_n[None, :]
        q_off = offs_mc[:, None] * sqq + offs_d[None, :]
        bias32_bh = Bias32 + b * sbb + h * sbh
        bias16_bh = Bias16 + b * sbb + h * sbh

        r0 = pid_r * ROWS
        i0 = tl.minimum(r0 + 0, N_ROWS - 1).to(tl.int64)
        q0 = tl.load(Q + b * sqb + i0 * sqi + h * sqh + q_off)
        kb0 = K + b * skb + i0 * ski + h * skh
        vb0 = V + b * svb + i0 * svi + h * svh
        mb0 = Mask + b * smb + i0 * smi
        q1 = q0
        kb1 = kb0
        vb1 = vb0
        mb1 = mb0
        if ROWS >= 2:
            i1 = tl.minimum(r0 + 1, N_ROWS - 1).to(tl.int64)
            q1 = tl.load(Q + b * sqb + i1 * sqi + h * sqh + q_off)
            kb1 = K + b * skb + i1 * ski + h * skh
            vb1 = V + b * svb + i1 * svi + h * svh
            mb1 = Mask + b * smb + i1 * smi

        # 16-bit bias tiles only if every prep program found the cast
        # lossless: the fp32 values rebuilt in registers are then identical
        use16 = False
        if BIAS16:
            foffs = tl.arange(0, NFLAG)
            fmin = tl.min(tl.load(Flags + foffs, mask=foffs < n_flags, other=1), 0)
            use16 = fmin != 0
        # rows whose keys are all unmasked take the mask-free loop (the same
        # result for such rows, fewer instructions per logit)
        all_unmasked = True
        if HAS_MASK:
            soff = tl.arange(0, MSCAN)
            mmin = tl.min(
                tl.load(mb0 + soff * smk, mask=soff < SEQ_K, other=1).to(tl.int32), 0
            )
            for ms in range(MSCAN, SEQ_K, MSCAN):
                mmin = tl.minimum(
                    mmin,
                    tl.min(
                        tl.load(
                            mb0 + (ms + soff) * smk, mask=(ms + soff) < SEQ_K, other=1
                        ).to(tl.int32),
                        0,
                    ),
                )
            if ROWS >= 2:
                for ms in range(0, SEQ_K, MSCAN):
                    mmin = tl.minimum(
                        mmin,
                        tl.min(
                            tl.load(
                                mb1 + (ms + soff) * smk,
                                mask=(ms + soff) < SEQ_K,
                                other=1,
                            ).to(tl.int32),
                            0,
                        ),
                    )
            all_unmasked = mmin != 0

        # the bias pointer's element type is part of the compiled code, so
        # each combination is its own branch
        if all_unmasked:
            if use16:
                acc0, m0, l0, acc1, m1, l1 = _attend(
                    q0, q1, kb0, kb1, vb0, vb1, mb0, mb1, bias16_bh, b_off, k_off,
                    v_off, m_off, offs_n, skk, svk, smk, SEQ_K, qk_scale2,
                    HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, ROWS=ROWS,
                    APPLY_MASK=False,
                )  # fmt: skip
            else:
                acc0, m0, l0, acc1, m1, l1 = _attend(
                    q0, q1, kb0, kb1, vb0, vb1, mb0, mb1, bias32_bh, b_off, k_off,
                    v_off, m_off, offs_n, skk, svk, smk, SEQ_K, qk_scale2,
                    HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, ROWS=ROWS,
                    APPLY_MASK=False,
                )  # fmt: skip
        else:
            if use16:
                acc0, m0, l0, acc1, m1, l1 = _attend(
                    q0, q1, kb0, kb1, vb0, vb1, mb0, mb1, bias16_bh, b_off, k_off,
                    v_off, m_off, offs_n, skk, svk, smk, SEQ_K, qk_scale2,
                    HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, ROWS=ROWS,
                    APPLY_MASK=True,
                )  # fmt: skip
            else:
                acc0, m0, l0, acc1, m1, l1 = _attend(
                    q0, q1, kb0, kb1, vb0, vb1, mb0, mb1, bias32_bh, b_off, k_off,
                    v_off, m_off, offs_n, skk, svk, smk, SEQ_K, qk_scale2,
                    HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, ROWS=ROWS,
                    APPLY_MASK=True,
                )  # fmt: skip

        row_ok = (offs_m < SEQ_Q)[:, None]
        o_off = offs_mc[:, None] * soq + offs_d[None, :]
        o_bh = Out + b * sob + h * soh
        tl.store(
            o_bh + i0 * soi + o_off,
            (acc0 / l0[:, None]).to(Out.dtype.element_ty),
            mask=row_ok,
        )
        if ROWS >= 2:
            tl.store(
                o_bh + i1 * soi + o_off,
                (acc1 / l1[:, None]).to(Out.dtype.element_ty),
                mask=row_ok & ((r0 + 1) < N_ROWS),
            )


def _one_row_bias(bias: torch.Tensor) -> Optional[torch.Tensor]:
    """The [1, H, SQ, SK] triangle bias, which chunk_layer may have expanded
    over the chunk rows with stride 0; None for a genuinely per-row bias."""
    while bias.dim() > 4 and bias.shape[0] == 1:
        bias = bias[0]
    if bias.dim() != 4:
        return None
    if bias.shape[0] > 1:
        if bias.stride(0) != 0:
            return None
        bias = bias[:1]
    return bias


def serve(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: list[torch.Tensor],
    c_hidden: int,
) -> Optional[torch.Tensor]:
    """The stock call ``cuequivariance_triangular_attn(...)[0]`` for an
    eligible call, else None (the caller then runs the stock op)."""
    if triton is None:
        return _skip("triton unavailable")
    if torch.is_grad_enabled():
        return _skip("autograd enabled")
    if not (
        torch.is_autocast_enabled()
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    ):
        return _skip("not under bf16 autocast")
    if q.dim() != 4 or len(biases) != 2:
        return _skip(f"rank {q.dim()}")
    rows, heads, n, d = q.shape
    if d != HEAD_DIM or heads != HEADS or c_hidden != HEAD_DIM:
        return _skip(f"heads {heads} x {d}")
    if not MIN_TOKENS <= n <= MAX_TOKENS:
        return _skip(f"{n} tokens outside {MIN_TOKENS}-{MAX_TOKENS}")
    bias = _one_row_bias(biases[1])
    if bias is None or tuple(bias.shape) != (1, heads, n, n):
        return _skip("per-row or unexpected triangle bias")
    mask = (biases[0] == 0).bool()
    if tuple(mask.shape) != (rows, 1, 1, n):
        return _skip(f"mask shape {tuple(mask.shape)}")
    if not dispatch.available(SITE, q.device):
        return None
    out = dispatch.guarded(
        SITE, _forward, q, k, v, bias.float(), mask, 1.0 / math.sqrt(c_hidden)
    )
    return None if out is None else out[0]


def _skip(reason: str) -> None:
    dispatch.fallback(SITE, reason)
    return None


def _forward(q, k, v, bias, mask, scale) -> torch.Tensor:
    """The 5-D result of cuEquivariance's triangle_attention for 4-D inputs."""
    adt = torch.get_autocast_dtype("cuda")
    q, k, v = (t.to(adt)[None] for t in (q, k, v))  # [1, N, H, S, D], as cuEq
    for t in (q, k, v):
        assert t.stride(-1) == 1
    bias = bias[None]  # [1, 1, H, SQ, SK]
    mask_u8 = mask[None].view(torch.uint8)  # [1, N, 1, 1, SK], same storage
    B, N, H, SQ, D = q.shape
    SK = k.shape[3]
    out = torch.empty((B, N, H, SQ, D), dtype=q.dtype, device=q.device)
    want16 = SK >= BIAS16_MIN_KEYS
    SKp = -(-SK // 16) * 16
    PB, PK = 32, 128
    nqb = triton.cdiv(SQ, PB)
    n_flags = B * H * nqb
    bias32 = torch.empty((B, 1, H, SQ, SKp), dtype=torch.float32, device=q.device)
    if want16:
        bias16 = torch.empty((B, 1, H, SQ, SKp), dtype=q.dtype, device=q.device)
        flags = torch.empty((n_flags,), dtype=torch.int32, device=q.device)
    else:
        bias16 = bias32  # never read (BIAS16 = 0)
        flags = torch.ones((1,), dtype=torch.int32, device=q.device)
        n_flags = 1
    _bias_prep[(nqb, B * H)](
        bias, bias32, bias16, flags,
        bias.stride(0), bias.stride(2), bias.stride(3), bias.stride(4),
        H, SQ, SK, SKp, nqb,
        PB=PB, PK=PK, MAKE16=want16, num_warps=4, num_stages=2,
    )  # fmt: skip
    c = CONFIG
    grid = (triton.cdiv(SQ, c["BLOCK_M"]), triton.cdiv(N, c["ROWS"]), B * H)
    _flash_triattn_fwd[grid](
        q, k, v, bias32, bias16, flags, mask_u8, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        bias32.stride(0), bias32.stride(2), bias32.stride(3),
        mask_u8.stride(0), mask_u8.stride(1), mask_u8.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N, SQ, SK, H, n_flags,
        float(scale),
        HEAD_DIM=D, BLOCK_M=c["BLOCK_M"], BLOCK_N=c["BLOCK_N"], ROWS=c["ROWS"],
        HAS_MASK=True, BIAS16=1 if want16 else 0, MSCAN=1024,
        NFLAG=max(1024, triton.next_power_of_2(n_flags)),
        num_warps=c["num_warps"], num_stages=c["num_stages"],
    )  # fmt: skip
    return out
