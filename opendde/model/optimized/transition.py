# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Anthropic, PBC
# Follows the arithmetic of OpenDDE's Transition (Aureka AI Research,
# Apache-2.0, derived from Protenix, ByteDance, Apache-2.0); see NOTICE.
"""Pair transition with one fused SiLU-multiply kernel (transition_triton).

Under bf16 autocast, stock ``Transition.forward`` runs per row chunk:

    y = LN(x)                 fp32
    a = linear_a(y)           y cast to bf16, bf16 GEMM
    a = silu(a)               fp32 opmath, rounded to bf16
    b = linear_b(y)           y cast to bf16 again, the same GEMM shape
    b *= a                    fp32 multiply, rounded to bf16
    b = linear_o(b)
    outputs[rows] = b

This path casts y to bf16 once, calls the same cuBLAS GEMMs, and replaces
the SiLU and the multiply (three passes over the hidden tensor) with one
Triton kernel that rounds at the same two points. The chunking and the output
tensor are the stock ones, so the result is bitwise the stock result.
Forward only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from opendde.model.optimized import dispatch

SITE = "transition_triton"

try:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice
    except ImportError:  # pragma: no cover - older Triton layouts
        from triton.language.extra.cuda import libdevice
except Exception:  # noqa: BLE001 -- Triton is optional: the stock op serves
    triton = None

CELL = (384, 1536)  # (c_in, hidden) of the measured row
MIN_ROWS = 256 * 256  # measured from 256 tokens; smaller calls are not faster
BLOCK = 2048

if triton is not None:

    @triton.jit
    def _silu_mul(a_ptr, b_ptr, h_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        a = tl.load(a_ptr + off, mask=m, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + off, mask=m, other=0.0)
        # F.silu on bf16: fp32 opmath with CUDA expf, one rounding to bf16
        s = (a / (1.0 + libdevice.exp(-a))).to(tl.bfloat16)
        # b *= a on bf16: fp32 product, one rounding to bf16
        h = (b.to(tl.float32) * s.to(tl.float32)).to(tl.bfloat16)
        tl.store(h_ptr + off, h, mask=m)


def eligible(module, x: torch.Tensor) -> bool:
    if triton is None:
        return dispatch.fallback(SITE, "triton unavailable")
    if torch.is_grad_enabled():
        return dispatch.fallback(SITE, "autograd enabled")
    if not (
        torch.is_autocast_enabled()
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    ):
        return dispatch.fallback(SITE, "not under bf16 autocast")
    if (module.c_in, module.n * module.c_in) != CELL:
        return False  # another transition width: not this kernel's call site
    if x.dtype not in (torch.bfloat16, torch.float32) or not x.is_cuda:
        return dispatch.fallback(SITE, f"input {x.dtype} on {x.device.type}")
    if x.numel() // module.c_in < MIN_ROWS:
        return False  # single-representation and small pair calls
    for linear in (
        module.linear_no_bias_a,
        module.linear_no_bias_b,
        module.linear_no_bias,
    ):
        if linear.bias is not None or getattr(linear, "precision", None) is not None:
            return dispatch.fallback(SITE, "linear variant")
    return dispatch.available(SITE, x.device)


def _bf16_weights(module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The weights as autocast casts them, cached per weight version."""
    params = (
        module.linear_no_bias_a.weight,
        module.linear_no_bias_b.weight,
        module.linear_no_bias.weight,
    )
    key = tuple((p.data_ptr(), p._version) for p in params)
    cache = getattr(module, "_transition_triton_weights", None)
    if cache is None or cache[0] != key:
        with torch.no_grad():
            cache = (
                key,
                tuple(p.detach().to(torch.bfloat16).contiguous() for p in params),
            )
        module._transition_triton_weights = cache
    return cache[1]


def forward(module, x: torch.Tensor, chunk_rows: int) -> torch.Tensor:
    """``module(x)`` for an eligible call; ``chunk_rows`` is the stock split."""
    wa, wb, wo = _bf16_weights(module)
    other_dims = x.shape[:-1]
    size = x.shape[-2]
    x = x.reshape(-1, module.c_in)
    if x.shape[0] > chunk_rows:
        chunks = x.split(chunk_rows, dim=-2)
    elif size >= 3200:
        chunks = torch.chunk(x, 8, dim=-2)
    else:
        chunks = (x,)
    outputs = torch.empty((x.shape[0], module.c_in), dtype=x.dtype, device=x.device)
    start = 0
    for chunk in chunks:
        # the LayerNorm runs as in stock, under the caller's autocast; the
        # GEMM operands are then bf16 already, so autocast leaves them as is
        y = module.layernorm1(chunk).to(torch.bfloat16)
        a = F.linear(y, wa)
        b = F.linear(y, wb)
        del y
        n = b.numel()
        _silu_mul[(triton.cdiv(n, BLOCK),)](a, b, b, n, BLOCK=BLOCK, num_warps=4)
        del a
        b = F.linear(b, wo)
        outputs[start : start + b.shape[0]] = b
        start += b.shape[0]
        del b
    return outputs.reshape(*other_dims, module.c_in)
