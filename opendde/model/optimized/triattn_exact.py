# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Anthropic, PBC
# Adapted from the kit's triattn_exact package (cuda_mma route, v3 kernel),
# written against the observable numerics of the cuEquivariance
# triangle_attention call it replaces; see NOTICE.
"""Triangle attention bit-identical to cuEquivariance's (triattn_exact).

Serves the cuEquivariance ``triangle_attention`` call of ``Attention`` under
bf16 autocast at head dimension 32 with a CUDA kernel (``mma.sync`` code path
for sm_80, ``csrc/triattn_v3.cu``) whose results are bitwise the library's:

    out[b, i, h, q] = softmax_j(scale * q . k_j + bias[b, h, q, j]
                                (-1e9 where mask[b, i, j] is False)) @ v

The kernel is compiled with nvcc at first use (``--fmad=false``: every FMA is
explicit in the source) into a private cache under ``$TORCH_EXTENSIONS_DIR``
and loaded with ctypes; it launches on torch's current stream. Before its
first call on a device it is checked bitwise against the live library on
three small shapes, and a mismatch disables it for the process. Forward
only.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import math
import os
import re
import shutil
import stat
import subprocess
import threading
from pathlib import Path
from typing import Optional

import torch

from opendde.model.optimized import dispatch
from opendde.utils.logger import get_logger

logger = get_logger(__name__)

SITE = "triattn_exact"
HEAD_DIM = 32
# Measured row (A100 80GB PCIe, 2026-10-17): bitwise at 101-2,047 tokens,
# 1.04-1.71x the stock kernel from 192 tokens; slower below about 176.
MIN_TOKENS = 192
MAX_TOKENS = 2047
NVCC_RELEASE = "12.8"  # the compiler of the measured build
SOURCE = Path(__file__).with_name("csrc") / "triattn_v3.cu"
NVCC_FLAGS = (
    "-O3",
    "-std=c++17",
    "--shared",
    "-Xcompiler",
    "-fPIC",
    "-lineinfo",
    "--fmad=false",
    "-gencode",
    "arch=compute_80,code=sm_80",
)

# Launch configurations of the kernel: cfg id -> (G pair rows per CTA, BM
# query rows per CTA). 9 and 10 serve the S % BM tail rows.
_CFG = {0: (4, 64), 2: (2, 64), 3: (1, 64), 4: (2, 128), 9: (4, 16), 10: (4, 32)}
_CFG_ORDER = (4, 0, 2, 3)
_CFG_RELCOST = {4: 1.00, 0: 1.09, 2: 1.20, 3: 1.35}
_FILL_CTAS = 2 * 132  # the kit's configuration choice, as measured
_VSCAN_MIN_QBLOCKS = 4
_GRID_LIMIT = 65535
_SELFCHECK_SHAPES = ((1, 8, 4, 128, True), (1, 3, 2, 191, False), (2, 4, 4, 136, True))


class _LaunchArgs(ctypes.Structure):
    _fields_ = [
        ("q", ctypes.c_void_p),
        ("k", ctypes.c_void_p),
        ("v", ctypes.c_void_p),
        ("bias", ctypes.c_void_p),
        ("bias_bf16", ctypes.c_int),
        ("mask", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("B", ctypes.c_int),
        ("N", ctypes.c_int),
        ("H", ctypes.c_int),
        ("S", ctypes.c_int),
        *[(f"s{t}{d}", ctypes.c_longlong) for t in "qkv" for d in ("B", "N", "H", "S")],
        ("sbB", ctypes.c_longlong),
        ("sbH", ctypes.c_longlong),
        ("sbQ", ctypes.c_longlong),
        ("sbK", ctypes.c_longlong),
        ("smB", ctypes.c_longlong),
        ("smN", ctypes.c_longlong),
        ("scale", ctypes.c_float),
        ("G", ctypes.c_int),
        ("nwarps", ctypes.c_int),
        ("bias_fast", ctypes.c_int),
        ("stream", ctypes.c_void_p),
        ("cfg", ctypes.c_int),
        ("pad_", ctypes.c_int),
        ("vbad", ctypes.c_void_p),
    ]


_lock = threading.Lock()
_lib = None
_checked: set[int] = set()


# ------------------------------------------------------------------ serving


def serve(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: list[torch.Tensor],
    c_hidden: int,
) -> Optional[torch.Tensor]:
    """The stock call ``cuequivariance_triangular_attn(...)[0]`` for an
    eligible call, else None (the caller then runs the stock op)."""
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
    if rows > _GRID_LIMIT:
        return _skip(f"{rows} rows")
    if tuple(biases[1].shape) != (1, heads, n, n):
        return _skip(f"bias shape {tuple(biases[1].shape)}")
    bias = _dezero(biases[1].float())
    mask = (biases[0] == 0).bool()
    if tuple(mask.shape) != (rows, 1, 1, n):
        return _skip(f"mask shape {tuple(mask.shape)}")
    q, k, v = (_dezero(t.to(torch.bfloat16)) for t in (q, k, v))
    if not all(_aligned(t) for t in (q, k, v)):
        return _skip("q/k/v rows not 16-byte aligned")
    reason = _toolchain_reason()
    if reason is not None:
        return _skip(reason)
    if not dispatch.available(SITE, q.device):
        return None
    if q.device.index not in _checked and not _self_check(q.device):
        return None
    out = dispatch.guarded(
        SITE,
        _attention,
        q[None],
        k[None],
        v[None],
        bias[None],
        _dezero(mask)[None].contiguous(),
        1.0 / math.sqrt(c_hidden),
    )
    return None if out is None else out[0]


def _skip(reason: str) -> None:
    dispatch.fallback(SITE, reason)
    return None


def _dezero(t: torch.Tensor) -> torch.Tensor:
    """Materialize expanded (stride-0) dimensions; values are unchanged."""
    if any(st == 0 and sz > 1 for st, sz in zip(t.stride(), t.shape)):
        return t.contiguous()
    return t


def _aligned(t: torch.Tensor) -> bool:
    if t.stride(-1) != 1 or t.data_ptr() % 16:
        return False
    return all(st % 8 == 0 for st, sz in zip(t.stride()[:-1], t.shape) if sz > 1)


# ------------------------------------------------------------------ launch


def _pick_cfg(B: int, N: int, H: int, S: int) -> int:
    """Cheapest configuration (padded rows x pair rows x relative cost) among
    those that fill the GPU; the smallest CTA shape for small grids."""
    best, best_cost = 3, None
    for cfg in _CFG_ORDER:
        G, BM = _CFG[cfg]
        n_groups = (N + G - 1) // G
        if ((S + BM - 1) // BM) * H * B * n_groups < _FILL_CTAS and cfg != 3:
            continue
        main_rows, tail_rows, tail_cfg = _plan_rows(S, BM)
        if tail_rows:
            padded = main_rows + _CFG[tail_cfg][1] + 24  # +24: the extra launch
        else:
            padded = (S + BM - 1) // BM * BM
        cost = padded * n_groups * G * _CFG_RELCOST[cfg]
        if best_cost is None or cost < best_cost:
            best, best_cost = cfg, cost
    return best


def _plan_rows(S: int, BM: int) -> tuple[int, int, Optional[int]]:
    """Split off the S % BM leftover query rows for a small-BM launch, unless
    they fill at least 3/4 of a block or S < 257."""
    rem = S % BM
    if rem == 0 or S - rem < 256 or rem > (3 * BM) // 4:
        return S, 0, None
    tail_cfg = 9 if rem <= 16 else (10 if rem <= 32 else (0 if rem <= 64 else None))
    if tail_cfg is None:
        return S, 0, None
    return S - rem, rem, tail_cfg


def _stage_bias(b5: torch.Tensor) -> torch.Tensor:
    """The bias itself when it streams with 16-byte copies, else a copy with
    16-byte row pitch (values, so the arithmetic, unchanged)."""
    if _bias_fast(b5):
        return b5
    B, _, H, S, _ = b5.shape
    per16 = 16 // b5.element_size()
    padded = torch.empty(
        (B, 1, H, S, (S + per16 - 1) // per16 * per16),
        dtype=b5.dtype,
        device=b5.device,
    )
    staged = padded[..., :S]
    staged.copy_(b5)
    return staged


def _bias_fast(b5: torch.Tensor) -> int:
    es = b5.element_size()
    if b5.stride(-1) != 1 and b5.shape[-1] > 1:
        return 0
    for dim in (0, 2, 3):
        if b5.shape[dim] > 1 and (b5.stride(dim) * es) % 16:
            return 0
    return 1 if b5.data_ptr() % 16 == 0 else 0


def _attention(q5, k5, v5, b5, m5, scale) -> torch.Tensor:
    """cuEquivariance's 5-D triangle_attention result: q, k, v bf16
    [B, N, H, S, 32]; bias [B, 1, H, S, S]; mask None or bool [B, N, 1, 1, S]."""
    lib = _load()
    B, N, H, S, _ = q5.shape
    cfg = _pick_cfg(B, N, H, S)
    G, BM = _CFG[cfg]
    if B * ((N + G - 1) // G) > _GRID_LIMIT:
        raise ValueError(f"grid of {B * ((N + G - 1) // G)} pair-row groups")
    b5 = _stage_bias(b5)
    out = torch.empty((B, N, H, S, HEAD_DIM), dtype=torch.bfloat16, device=q5.device)
    a = _LaunchArgs()
    a.q, a.k, a.v, a.bias = (t.data_ptr() for t in (q5, k5, v5, b5))
    a.bias_bf16 = int(b5.dtype == torch.bfloat16)
    a.mask = m5.data_ptr() if m5 is not None else None
    a.out = out.data_ptr()
    a.B, a.N, a.H, a.S = B, N, H, S
    for name, t in (("q", q5), ("k", k5), ("v", v5)):
        for i, d in enumerate("BNHS"):
            setattr(a, f"s{name}{d}", t.stride(i))
    a.sbB = b5.stride(0) if b5.shape[0] == B else 0
    a.sbH, a.sbQ, a.sbK = b5.stride(2), b5.stride(3), b5.stride(4)
    if m5 is not None:
        a.smB = m5.stride(0) if m5.shape[0] == B else 0
        a.smN = m5.stride(1)
    else:
        a.smB = a.smN = 0
    a.scale = float(scale)
    a.bias_fast = _bias_fast(b5)
    a.stream = torch.cuda.current_stream(q5.device).cuda_stream
    a.G, a.nwarps, a.cfg = G, 4, cfg
    _, tail_rows, tail_cfg = _plan_rows(S, BM)
    a.pad_ = tail_rows  # > 0: this launch serves rows [0, S - tail_rows)
    vbad = None
    n_qblocks = (S + BM - 1) // BM + (1 if tail_rows else 0)
    if m5 is not None and n_qblocks >= _VSCAN_MIN_QBLOCKS:
        # Marks the fully masked key tiles whose V holds NaN/Inf, which the
        # kernel then does not skip (0 * NaN must propagate as in the library).
        vbad = torch.empty(
            B * N * H * ((S + 63) // 64), dtype=torch.uint8, device=q5.device
        )
    with torch.cuda.device(q5.device):
        rc = 0
        if vbad is not None:
            rc = lib.triattn_v3_vscan(ctypes.byref(a), ctypes.c_void_p(vbad.data_ptr()))
            a.vbad = vbad.data_ptr()
        if rc == 0:
            rc = lib.triattn_v3_launch(ctypes.byref(a))
        if rc == 0 and tail_rows:
            a.cfg, a.G = tail_cfg, _CFG[tail_cfg][0]
            a.pad_ = -tail_rows  # < 0: the tail rows [S - tail_rows, S)
            rc = lib.triattn_v3_launch(ctypes.byref(a))
    if rc != 0:
        raise RuntimeError(f"triattn_v3 launch failed with CUDA error {rc} (cfg {cfg})")
    return out


# ------------------------------------------------------------- build, check


def _nvcc() -> Optional[str]:
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home and os.path.exists(os.path.join(cuda_home, "bin", "nvcc")):
        return os.path.join(cuda_home, "bin", "nvcc")
    return shutil.which("nvcc")


@functools.lru_cache(maxsize=None)
def _nvcc_version() -> Optional[str]:
    nvcc = _nvcc()
    if nvcc is None:
        return None
    try:
        return subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _toolchain_reason() -> Optional[str]:
    version = _nvcc_version()
    if version is None:
        return "nvcc not found"
    release = re.search(r"release (\d+\.\d+)", version)
    if release is None or release.group(1) != NVCC_RELEASE:
        return f"nvcc {release.group(1) if release else 'unknown'}"
    return None


def _cache_dir() -> Path:
    """A build directory only this account can write."""
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions"
    )
    path = Path(root) / "opendde_triattn_exact"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise PermissionError(f"{path} is writable by another account")
    return path


def _load():
    global _lib
    with _lock:
        if _lib is not None:
            return _lib
        source = SOURCE.read_bytes()
        key = hashlib.sha256(
            source + _nvcc_version().encode() + " ".join(NVCC_FLAGS).encode()
        ).hexdigest()[:16]
        so = _cache_dir() / f"triattn_v3_{key}.so"
        if not so.exists():
            tmp = so.with_name(f"{so.name}.{os.getpid()}.tmp")
            logger.info("building the triattn_exact kernel with %s", _nvcc())
            subprocess.run(
                [_nvcc(), *NVCC_FLAGS, "-o", str(tmp), str(SOURCE)],
                capture_output=True,
                text=True,
                check=True,
            )
            os.replace(tmp, so)
        lib = ctypes.CDLL(str(so))
        for fn in ("triattn_v3_launch", "triattn_v3_vscan"):
            getattr(lib, fn).restype = ctypes.c_int
        lib.triattn_v3_launch.argtypes = [ctypes.POINTER(_LaunchArgs)]
        lib.triattn_v3_vscan.argtypes = [ctypes.POINTER(_LaunchArgs), ctypes.c_void_p]
        if lib.triattn_v3_sizeof_launchargs() != ctypes.sizeof(_LaunchArgs):
            raise RuntimeError("triattn_v3 launch-argument layout mismatch")
        _lib = lib
        return lib


def _self_check(device: torch.device) -> bool:
    """Compare with the live library bitwise on three small shapes (with and
    without a mask, a fully masked row, 4-byte bias); disable the kernel for
    the process on any difference."""
    from cuequivariance_torch.primitives.triangle import triangle_attention

    g = torch.Generator(device=device).manual_seed(20260915)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        for B, N, H, S, use_mask in _SELFCHECK_SHAPES:
            proj = torch.randn(
                B, N, S, 3 * H * HEAD_DIM, device=device, generator=g
            ).to(torch.bfloat16)
            q, k, v = (
                proj[..., i * H * HEAD_DIM : (i + 1) * H * HEAD_DIM]
                .unflatten(-1, (H, HEAD_DIM))
                .permute(0, 1, 3, 2, 4)
                for i in range(3)
            )
            bias = (
                torch.randn(B, S, S, H, device=device, generator=g)
                .permute(0, 3, 1, 2)
                .unsqueeze(1)
            )
            mask = None
            if use_mask:
                lengths = torch.randint(1, S + 1, (B, N), device=device, generator=g)
                lengths[0, 0] = 0
                mask = (torch.arange(S, device=device) < lengths[..., None]).view(
                    B, N, 1, 1, S
                )
            ref = triangle_attention(q, k, v, bias, mask=mask)
            ref = ref[0] if isinstance(ref, tuple) else ref
            ours = dispatch.guarded(
                SITE, _attention, q, k, v, bias, mask, 1.0 / math.sqrt(HEAD_DIM)
            )
            if ours is None:
                return False
            if ref.shape != ours.shape or not torch.equal(
                ref.view(torch.int16), ours.view(torch.int16)
            ):
                dispatch.disable(
                    SITE,
                    RuntimeError(
                        f"self-check mismatch with cuEquivariance at B{B} N{N} H{H} "
                        f"S{S} mask={use_mask}"
                    ),
                )
                return False
    _checked.add(device.index)
    logger.info("triattn_exact self-check passed on %s", device)
    return True
