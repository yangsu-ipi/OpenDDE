# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Anthropic, PBC
# Adapted from the kit's triattn_native v11 package, sm_80 member (cuda_80:
# triattn_sm80.py and csrc/), trimmed to bf16 at head dimension 32; see NOTICE.
"""Triangle attention for A100 in optimization_mode fast (triattn_native).

Serves the cuEquivariance ``triangle_attention`` call of ``Attention`` under
bf16 autocast at head dimension 32 with the kit's sm_80 CUDA kernel
(``csrc/triattn_sm80``: ``mma.sync`` bf16 products with fp32 accumulation,
``ldmatrix``, a ``cp.async`` multistage ring):

    out[b, i, h, q] = softmax_j(scale * q . k_j + bias[b, h, q, j]
                                (masked keys excluded) ) @ v

The fp32 pair bias, pre-divided by ``scale``, is the initial accumulator of
the logits; the softmax is the kit family's max-free streaming form in base 2
with exact power-of-two renormalisation, and tiles it cannot finish are
recomputed by the running-max form of the same routine. A pair row with no
attendable key gives the mean of ``v``, as cuEquivariance does. The result is
not bitwise the stock one: on A100 its error against an fp32 reference is
within 1.06x of stock's (kernel benchmark, 2026-10-18). No atomics in the
output, so a call gives the same bits on every run. Forward only.

The extension is compiled at first use with ``torch.utils.cpp_extension``
(nvcc, sm_80 code only, about a minute; cached under ``TORCH_EXTENSIONS_DIR``)
and requires the CUDA 12.8 ``nvcc`` of the measured build.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import torch

from opendde.model.optimized import dispatch
from opendde.model.optimized import triattn_exact

SITE = "triattn_native"
HEAD_DIM = 32
# Measured row (A100 80GB PCIe, 2026-10-18): 2.1-2.4x triattn_exact from 300
# tokens, 1.17x at 192; slower than triattn_exact at 128 tokens.
MIN_TOKENS = 192
MAX_TOKENS = 2047
SOURCES = Path(__file__).with_name("csrc") / "triattn_sm80"
NVCC_FLAGS = (
    "-O3",
    "-std=c++17",
    "-gencode",
    "arch=compute_80,code=sm_80",
    "--expt-relaxed-constexpr",
    "-lineinfo",
    "-DNDEBUG",
    "-DTS_HAVE_D32",
)
_FIX_MIN = 1 << 18  # int32 entries of the first fix buffer (S up to about 2,300)

_EXT = None
_FIX: dict[torch.device, torch.Tensor] = {}
_CENSUS: dict[torch.device, torch.Tensor] = {}


# ------------------------------------------------------------------ serving


def serve(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: list[torch.Tensor],
    c_hidden: int,
) -> Optional[torch.Tensor]:
    """The stock call ``cuequivariance_triangular_attn(...)[0]`` (rounding
    aside) for an eligible call, else None (the caller then runs the next
    kernel or the stock op)."""
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
    if d != HEAD_DIM or c_hidden != HEAD_DIM:
        return _skip(f"head dimension {d}")
    if not MIN_TOKENS <= n <= MAX_TOKENS:
        return _skip(f"{n} tokens outside {MIN_TOKENS}-{MAX_TOKENS}")
    if tuple(biases[1].shape) != (1, heads, n, n):
        return _skip(f"bias shape {tuple(biases[1].shape)}")
    mask = (biases[0] == 0).bool()
    if tuple(mask.shape) != (rows, 1, 1, n):
        return _skip(f"mask shape {tuple(mask.shape)}")
    q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))
    if not all(_aligned(t) for t in (q, k, v)):
        return _skip("q/k/v strides not multiples of 8 elements")
    reason = triattn_exact._toolchain_reason()
    if reason is not None:
        return _skip(reason)
    if not dispatch.available(SITE, q.device):
        return None
    return dispatch.guarded(
        SITE,
        _forward,
        q,
        k,
        v,
        biases[1].float(),
        mask,
        1.0 / math.sqrt(c_hidden),
    )


def _skip(reason: str) -> None:
    dispatch.fallback(SITE, reason)
    return None


def _aligned(t: torch.Tensor) -> bool:
    if t.stride(-1) != 1 or t.data_ptr() % 16:
        return False
    return all(s >= 0 and s % 8 == 0 for s in t.stride()[:-1])


# ------------------------------------------------------------------ launch


def _forward(q, k, v, bias, mask, scale) -> torch.Tensor:
    """[N, H, S, 32] bf16 q, k, v; bias [1, H, S, S] fp32 shared by the N
    pair rows; mask [N, 1, 1, S] bool, True = attend. Returns [N, H, S, 32]."""
    ext = _build()
    N, H, S, D = q.shape
    q5, k5, v5 = q[None], k[None], v[None]
    geo = ext.geometry(D, False)
    dense = all(t.stride(3) <= 512 for t in (q5, k5, v5))
    if S <= int(geo[8]) or (dense and S <= int(geo[9])):
        geo = ext.geometry(D, True)  # the small-S geometry the kernel will use
    fix = _fix_buffer(q.device, int(ext.fix_elems(1, N, H, S, D)))
    mask8 = mask.reshape(1, N, S).contiguous().view(torch.uint8)
    keyany, rows, maskw = ext.stage_mask(mask8, _census(q.device), int(geo[0]))
    bias_staged = ext.stage_bias(bias, float(scale), keyany)  # [B=1, H, S, S]
    out = torch.empty(1, N, H, S, D, dtype=torch.bfloat16, device=q.device)
    ext.fwd(q5, k5, v5, bias_staged, rows, maskw, float(scale), out, fix, None, 0)
    return out[0]


def _fix_buffer(device: torch.device, n: int) -> torch.Tensor:
    """The device's fix list (tiles the max-free pass hands to the running-max
    pass), grown geometrically and kept for the life of the process."""
    buf = _FIX.get(device)
    if buf is None or buf.numel() < n:
        size = max(n, _FIX_MIN, 2 * buf.numel() if buf is not None else 0)
        _FIX[device] = buf = torch.zeros(size, dtype=torch.int32, device=device)
    return buf


def _census(device: torch.device) -> torch.Tensor:
    """The kernel's mask census counters (ragged, fully masked rows)."""
    c = _CENSUS.get(device)
    if c is None:
        _CENSUS[device] = c = torch.zeros(4, dtype=torch.int32, device=device)
    return c


def _build():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"  # sm_80 code only, no PTX
        try:
            _EXT = load(
                name="opendde_triattn_sm80",
                sources=[
                    str(SOURCES / "sm80_binding.cu"),
                    str(SOURCES / "inst_d32.cu"),
                ],
                extra_include_paths=[str(SOURCES)],
                extra_cuda_cflags=list(NVCC_FLAGS),
                extra_cflags=["-O3", "-std=c++17", "-DTS_HAVE_D32"],
            )
        finally:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous
    return _EXT
