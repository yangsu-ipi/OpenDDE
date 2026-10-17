# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Anthropic, PBC
"""Where each optimized kernel may run: the measured GPUs and software stacks.

A row names the compute capability and the torch, Triton and cuEquivariance
versions a kernel was validated on. A call on any other device or stack runs
the stock code, and the first fallback of each reason is logged once.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from importlib import metadata
from typing import Optional

import torch

from opendde.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Row:
    capability: tuple[int, int]
    torch_prefix: str
    triton: str
    cuequivariance_ops: Optional[str] = None  # required when the kernel calls it


# Bit-identical to the stock op at OpenDDE's shapes (kernel benchmark on
# RTX PRO 6000 Blackwell, 2026-10-17; D1-D15 byte-identical end to end).
ROWS: dict[str, tuple[Row, ...]] = {
    "trimul_triton": (Row((12, 0), "2.7.", "3.3.1", "0.10.0"),),
    "transition_triton": (Row((12, 0), "2.7.", "3.3.1"),),
    # Not bitwise: error against fp32 as stock's (fast only; tile sweep on
    # RTX PRO 6000 Blackwell, 2026-10-17).
    "triattn_k2b": (Row((12, 0), "2.7.", "3.3.1"),),
}

_DISABLED: dict[str, str] = {}
_LOGGED: set[tuple[str, str]] = set()


def _version(dist: str) -> Optional[str]:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


@functools.lru_cache(maxsize=None)
def _stack() -> dict[str, Optional[str]]:
    try:
        import triton

        triton_version = triton.__version__
    except Exception:  # noqa: BLE001 -- Triton is optional: no row matches
        triton_version = None
    return {
        "torch": torch.__version__,
        "triton": triton_version,
        "cuequivariance_ops": _version("cuequivariance-ops-torch-cu12"),
    }


@functools.lru_cache(maxsize=None)
def _row_reason(site: str, device_index: int) -> Optional[str]:
    capability = torch.cuda.get_device_capability(device_index)
    stack = _stack()
    for row in ROWS[site]:
        if row.capability != capability:
            continue
        if not stack["torch"].startswith(row.torch_prefix):
            return f"torch {stack['torch']}"
        if stack["triton"] != row.triton:
            return f"triton {stack['triton']}"
        if (
            row.cuequivariance_ops
            and stack["cuequivariance_ops"] != row.cuequivariance_ops
        ):
            return f"cuequivariance-ops {stack['cuequivariance_ops']}"
        return None
    return f"compute capability {capability[0]}.{capability[1]}"


def available(site: str, device: torch.device) -> bool:
    """Whether ``site``'s kernel has a row for this device and stack."""
    if site in _DISABLED:
        return False
    if device.type != "cuda":
        return fallback(site, f"device {device.type}")
    reason = _row_reason(site, device.index or 0)
    if reason is not None:
        return fallback(site, reason)
    if (site, "in use") not in _LOGGED:
        _LOGGED.add((site, "in use"))
        logger.info("optimized kernel %s in use on %s", site, device)
    return True


def fallback(site: str, reason: str) -> bool:
    """Record that ``site`` runs stock code for ``reason`` (logged once)."""
    if (site, reason) not in _LOGGED:
        _LOGGED.add((site, reason))
        logger.info(
            "optimized kernel %s not used: %s; running the stock op", site, reason
        )
    return False


def guarded(site: str, kernel, *args, **kwargs):
    """``kernel(*args, **kwargs)``, or None after disabling ``site`` if it
    raised anything but out-of-memory. The kernels write only new tensors,
    so the caller can then run the stock op on the same inputs."""
    try:
        return kernel(*args, **kwargs)
    except torch.OutOfMemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 -- e.g. a Triton build or launch error
        disable(site, exc)
        return None


def disable(site: str, exc: BaseException) -> None:
    """Stop using ``site``'s kernel in this process after it failed."""
    _DISABLED[site] = repr(exc)
    logger.warning(
        "optimized kernel %s failed and is disabled for this process (%r); "
        "running the stock op",
        site,
        exc,
    )
