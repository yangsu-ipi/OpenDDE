# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Anthropic, PBC
# Portions restate the arithmetic of Protenix's pair-stack modules
# (Copyright 2024 ByteDance and/or its affiliates, Apache-2.0); see NOTICE.
"""Triangle multiplication with Triton gated projections (trimul_triton).

Serves the cuEquivariance path of ``TriangleMultiplicativeUpdate`` under bf16
autocast with ``c_z == c_hidden == 384``. The pipeline is the stock fused op's:

    x_in = LN_in(z)                          cuEquivariance's own LayerNorm
    ab   = sigmoid(x_in @ g_in^T) * (x_in @ p_in^T) * mask        Triton
    x    = outgoing: a @ b^T | incoming: a^T @ b per channel     cuBLAS
    out  = sigmoid(x_in @ g_out^T) * (LN_out(x) @ p_out^T) (+ z)  Triton

Both LayerNorms are cuEquivariance's ``layer_norm_transpose`` and the
contraction is the same unpadded cuBLAS call, so only the two gated GEMMs
are new. They accumulate each output in fp32 in the same K order and round
once, which gives the stock op's bits at every measured size (101-2,047
tokens, bf16 and fp32 pairs, with and without padding masks, both
directions). No autotuning, no
atomics: the result is the same on every run. Forward only.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from opendde.model.optimized import dispatch

SITE = "trimul_triton"

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001 -- Triton is optional: the stock op serves
    triton = None

# Launch settings of the measured row (sm_120, bf16, C = 384). The output
# tile is the largest whose pipelined loop fits in 99 KB of shared memory.
PROJECTION_TILE = dict(BM=128, BN=128, BK=32, num_warps=4, num_stages=4)
OUTPUT_TILE = dict(BM=64, BN=64, BK=32, num_warps=4, num_stages=2)
CHANNELS = 384
MIN_TOKENS = 101  # cuEquivariance runs its PyTorch fallback at 100 or fewer
MAX_TOKENS = 2047  # the output kernel's grid axis 1 is ceil(N * N / 64) <= 65535
_Z_DTYPES = (torch.bfloat16, torch.float32)  # the trunk's pair is bf16

if triton is not None:

    @triton.jit(do_not_specialize=["XSA"])
    def _gated_projection(
        xin_ptr,
        wpg_ptr,
        mask_ptr,
        ab_ptr,
        N,
        XSA,
        NKT,
        C: tl.constexpr,
        HAS_MASK: tl.constexpr,
        ALIGNED: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        # One dot per K step over the row-interleaved [4C, C] weight (row 2n
        # = p_in[n], row 2n + 1 = g_in[n]): each channel's (p, g) pair lands
        # in adjacent accumulator columns. The output is stored channel-major
        # as the planes ab[n, i, k]. grid: (4C / BN, N * NKT)
        pid_n = tl.program_id(0)
        pid = tl.program_id(1)
        i = pid // NKT
        kt = pid - i * NKT
        if ALIGNED:
            XS = XSA * 8
        else:
            XS = XSA
        PLANE = XS * XS
        offs_k = kt * BM + tl.arange(0, BM)
        k_ok = offs_k < N
        rows = (i * N + offs_k).to(tl.int64)
        offs_n = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, C, BK):
            offs_c = k0 + tl.arange(0, BK)
            xt = tl.load(
                xin_ptr + rows[:, None] * C + offs_c[None, :],
                mask=k_ok[:, None],
                other=0.0,
            )
            wt = tl.load(wpg_ptr + offs_n[None, :] * C + offs_c[:, None])
            acc = tl.dot(xt, wt, acc)
        pg = tl.reshape(acc, (BM, BN // 2, 2))
        p, gg = tl.split(pg)
        g = 1.0 / (1.0 + tl.exp(-gg))
        o = g * p
        if HAS_MASK:
            mk = tl.load(mask_ptr + rows, mask=k_ok, other=0.0).to(tl.float32)
            o = o * mk[:, None]
        offs_no = pid_n * (BN // 2) + tl.arange(0, BN // 2)
        if ALIGNED:
            oT = tl.trans(o.to(ab_ptr.dtype.element_ty))
            dst = (
                ab_ptr
                + offs_no[:, None].to(tl.int64) * PLANE
                + i.to(tl.int64) * XS
                + offs_k[None, :]
            )
            if (kt + 1) * BM <= N:
                tl.store(dst, oT)
            else:
                tl.store(dst, oT, mask=k_ok[None, :])
        else:
            dst = (
                ab_ptr
                + offs_no[None, :].to(tl.int64) * PLANE
                + i.to(tl.int64) * XS
                + offs_k[:, None].to(tl.int64)
            )
            tl.store(dst, o.to(ab_ptr.dtype.element_ty), mask=k_ok[:, None])

    @triton.jit
    def _gated_output(
        xo_ptr,
        xin_ptr,
        z_ptr,
        wpo_ptr,
        wgo_ptr,
        out_ptr,
        M,
        C: tl.constexpr,
        RESIDUAL: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        # out = sigmoid(x_in @ g_out^T) * (x_o @ p_out^T), rounded once to the
        # compute dtype; with RESIDUAL, + z as an explicit round-to-nearest
        # add that is never contracted into an FMA with the product, as stock
        # adds z to the rounded update. grid: (C / BN, ceil(M / BM))
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_m = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
        m_ok = offs_m < M
        offs_n = pid_n * BN + tl.arange(0, BN)
        acc_p = tl.zeros((BM, BN), dtype=tl.float32)
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, C, BK):
            offs_c = k0 + tl.arange(0, BK)
            xo = tl.load(
                xo_ptr + offs_m[:, None] * C + offs_c[None, :],
                mask=m_ok[:, None],
                other=0.0,
            )
            xi = tl.load(
                xin_ptr + offs_m[:, None] * C + offs_c[None, :],
                mask=m_ok[:, None],
                other=0.0,
            )
            wpo = tl.load(wpo_ptr + offs_n[None, :] * C + offs_c[:, None])
            wgo = tl.load(wgo_ptr + offs_n[None, :] * C + offs_c[:, None])
            acc_p = tl.dot(xo, wpo, acc_p)
            acc_g = tl.dot(xi, wgo, acc_g)
        g = 1.0 / (1.0 + tl.exp(-acc_g))
        o = (g * acc_p).to(xo_ptr.dtype.element_ty)
        if RESIDUAL:
            zr = tl.load(
                z_ptr + offs_m[:, None] * C + offs_n[None, :],
                mask=m_ok[:, None],
                other=0.0,
            )
            o = tl.inline_asm_elementwise(
                "add.rn.f32 $0, $1, $2;",
                "=r,r,r",
                [o.to(tl.float32), zr.to(tl.float32)],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            ).to(out_ptr.dtype.element_ty)
        tl.store(
            out_ptr + offs_m[:, None] * C + offs_n[None, :],
            o.to(out_ptr.dtype.element_ty),
            mask=m_ok[:, None],
        )


def eligible(module, z: torch.Tensor, mask: Optional[torch.Tensor]) -> bool:
    """Whether this call matches the measured row; the caller has already
    chosen the cuEquivariance path (CUDA, c_z == c_hidden)."""
    if triton is None:
        return dispatch.fallback(SITE, "triton unavailable")
    if torch.is_grad_enabled():
        return dispatch.fallback(SITE, "autograd enabled")
    if not (
        torch.is_autocast_enabled()
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    ):
        return dispatch.fallback(SITE, "not under bf16 autocast")
    if module.c_z != CHANNELS or z.dim() != 3 or z.dtype not in _Z_DTYPES:
        return dispatch.fallback(SITE, f"shape {tuple(z.shape)} {z.dtype}")
    if not z.is_contiguous():
        return dispatch.fallback(SITE, "non-contiguous input")
    n = z.shape[0]
    if not MIN_TOKENS <= n <= MAX_TOKENS:
        return dispatch.fallback(SITE, f"{n} tokens outside {MIN_TOKENS}-{MAX_TOKENS}")
    if mask is not None and tuple(mask.shape) != tuple(z.shape[:-1]):
        return dispatch.fallback(SITE, f"mask shape {tuple(mask.shape)}")
    if os.environ.get("CUEQ_TRITON_TUNING"):
        # tuned cuEquivariance kernels are not the ones this row matched
        return dispatch.fallback(SITE, "CUEQ_TRITON_TUNING set")
    return dispatch.available(SITE, z.device)


def _packed_weights(module, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """The GEMM weights cast as cuEquivariance casts them under autocast;
    LayerNorm parameters stay in their own dtype. Cached per weight version."""
    params = (
        module.layer_norm_in.weight,
        module.layer_norm_in.bias,
        module.linear_a_p.weight,
        module.linear_b_p.weight,
        module.linear_a_g.weight,
        module.linear_b_g.weight,
        module.layer_norm_out.weight,
        module.layer_norm_out.bias,
        module.linear_z.weight,
        module.linear_g.weight,
    )
    key = (dtype,) + tuple((p.data_ptr(), p._version) for p in params)
    cache = getattr(module, "_trimul_triton_weights", None)
    if cache is not None and cache[0] == key:
        return cache[1]
    with torch.no_grad():
        p_in = torch.cat([module.linear_a_p.weight, module.linear_b_p.weight], 0)
        g_in = torch.cat([module.linear_a_g.weight, module.linear_b_g.weight], 0)
        p_in, g_in = p_in.to(dtype), g_in.to(dtype)
        weights = dict(
            ln_in_w=module.layer_norm_in.weight.detach().contiguous(),
            ln_in_b=module.layer_norm_in.bias.detach().contiguous(),
            # row 2n = p_in[n], row 2n + 1 = g_in[n]
            pg_in=torch.stack([p_in, g_in], dim=1)
            .reshape(-1, p_in.shape[1])
            .contiguous(),
            ln_out_w=module.layer_norm_out.weight.detach().contiguous(),
            ln_out_b=module.layer_norm_out.bias.detach().contiguous(),
            p_out=module.linear_z.weight.detach().to(dtype).contiguous(),
            g_out=module.linear_g.weight.detach().to(dtype).contiguous(),
        )
    module._trimul_triton_weights = (key, weights)
    return weights


def forward(
    module, z: torch.Tensor, mask: Optional[torch.Tensor], residual: bool
) -> torch.Tensor:
    """The stock cuEquivariance branch's result: the update in the compute
    dtype, or ``z + update`` in z's dtype when ``residual``."""
    from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose

    n, _, c = z.shape
    m = n * n
    cdt = torch.get_autocast_dtype("cuda")
    w = _packed_weights(module, cdt)
    eps = 1e-5
    xin = (
        layer_norm_transpose(
            z[None], w["ln_in_w"], w["ln_in_b"], eps=eps, layout="bijd->bijd"
        )[0]
        .reshape(m, c)
        .to(cdt)
    )
    ab = torch.empty(2 * c, n, n, dtype=cdt, device=z.device)
    mask_flat = None
    if mask is not None:
        mask_flat = mask.reshape(-1)
        if not mask_flat.is_floating_point():
            mask_flat = mask_flat.to(torch.float32)
        mask_flat = mask_flat.contiguous()
    t = PROJECTION_TILE
    nkt = triton.cdiv(n, t["BM"])
    aligned = n % 8 == 0
    _gated_projection[((4 * c) // t["BN"], n * nkt)](
        xin,
        w["pg_in"],
        mask_flat if mask_flat is not None else z,
        ab,
        n,
        n // 8 if aligned else n,
        nkt,
        C=c,
        HAS_MASK=mask_flat is not None,
        ALIGNED=aligned,
        BM=t["BM"],
        BN=t["BN"],
        BK=t["BK"],
        num_warps=t["num_warps"],
        num_stages=t["num_stages"],
    )
    a, b = ab[:c], ab[c:]
    x = torch.empty(c, n, n, dtype=cdt, device=z.device)
    if module._outgoing:
        torch.matmul(a, b.transpose(-1, -2), out=x)
    else:
        torch.matmul(a.transpose(-1, -2), b, out=x)
    del ab, a, b
    xo = (
        layer_norm_transpose(
            x[:, None], w["ln_out_w"], w["ln_out_b"], eps=eps, layout="dbij->bijd"
        )[0]
        .reshape(m, c)
        .to(cdt)
    )
    del x
    out = torch.empty(z.shape, dtype=z.dtype if residual else cdt, device=z.device)
    t = OUTPUT_TILE
    _gated_output[(c // t["BN"], triton.cdiv(m, t["BM"]))](
        xo,
        xin,
        z,
        w["p_out"],
        w["g_out"],
        out,
        m,
        C=c,
        RESIDUAL=residual,
        BM=t["BM"],
        BN=t["BN"],
        BK=t["BK"],
        num_warps=t["num_warps"],
        num_stages=t["num_stages"],
    )
    return out
