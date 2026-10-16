# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Identity for the opt-in trunk disk cache (``OPENDDE_TRUNK_CACHE``).

A cached trunk result may be served only to the same input features, recycle
count, loaded weights, numerical path and platform that produced it. The key
therefore covers, besides the input features and ``N_cycle``:

- the content digest of the checkpoint bytes that were loaded;
- the trunk-relevant configuration, precision, TF32, deterministic setting,
  effective triangle kernels and LayerNorm type;
- the effective optimization decisions;
- the GPU compute capability and device name;
- the torch, CUDA runtime, Triton and cuEquivariance versions and the OpenDDE
  version.

The checkpoint digest is computed only when the cache is enabled, from the same
bytes passed to ``torch.load``, and bound to the runner-managed model after a
strict load together with a snapshot of its parameters and buffers. The cache is
bypassed when the model has no such identity (a non-strict load or a directly
constructed model) or when its tensors changed after the snapshot.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

import torch

FORMAT_VERSION = 2


def cache_enabled() -> bool:
    return bool(os.environ.get("OPENDDE_TRUNK_CACHE"))


def read_checkpoint(path: str, map_location: Any) -> tuple[Any, str]:
    """Load a checkpoint from bytes read once, returning it with their SHA-256.

    Hashing and then reopening the file would leave a window in which the file
    could change between the two reads."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    checkpoint = torch.load(
        io.BytesIO(data), map_location=map_location, weights_only=False
    )
    del data
    return checkpoint, digest


def _named_tensors(model: torch.nn.Module):
    yield from (("param:" + n, t) for n, t in model.named_parameters())
    yield from (("buffer:" + n, t) for n, t in model.named_buffers())


def _state(tensor: torch.Tensor) -> tuple:
    return (
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
        tensor._version,
        tensor.data_ptr(),
    )


@dataclass(frozen=True)
class ModelIdentity:
    """The loaded checkpoint's digest and a snapshot of the model's tensors.

    The snapshot holds the tensor objects themselves (compared with ``is``),
    not ``id()`` values, which Python reuses once an object is freed."""

    checkpoint_sha256: str
    tensors: tuple[tuple[str, torch.Tensor, tuple], ...]

    @classmethod
    def capture(cls, model: torch.nn.Module, checkpoint_sha256: str) -> "ModelIdentity":
        return cls(
            checkpoint_sha256,
            tuple((name, t, _state(t)) for name, t in _named_tensors(model)),
        )

    def mismatch(self, model: torch.nn.Module) -> Optional[str]:
        """Why the model no longer matches the snapshot, or None if it does."""
        current = dict(_named_tensors(model))
        names = [name for name, _, _ in self.tensors]
        if sorted(current) != sorted(names):
            return "the model's parameter or buffer set changed after loading"
        for name, tensor, state in self.tensors:
            now = current[name]
            if now is not tensor:
                return f"{name} was replaced after loading"
            if _state(now) != state:
                return f"{name} changed after loading"
        return None


def _version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _dump(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


def cache_context(
    configs: Any,
    device: torch.device,
    checkpoint_sha256: str,
    decisions: Optional[dict[str, bool]] = None,
) -> dict[str, Any]:
    """Everything besides the input features that the trunk result depends on."""
    from opendde.version import __version__

    platform: dict[str, Any] = {"device_type": device.type}
    if device.type == "cuda":
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        platform["capability"] = list(torch.cuda.get_device_capability(index))
        platform["device_name"] = torch.cuda.get_device_name(index)
    return {
        "format": FORMAT_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "settings": {
            "dtype": configs.dtype,
            "enable_tf32": configs.enable_tf32,
            "deterministic": configs.deterministic,
            "triangle_multiplicative": configs.triangle_multiplicative,
            "triangle_attention": configs.triangle_attention,
            "layernorm_type": os.environ.get("LAYERNORM_TYPE", "torch"),
            "model": _dump(configs.model),
            "skip_amp": _dump(configs.skip_amp),
            "infer_setting": _dump(configs.infer_setting),
            "dims": {
                k: getattr(configs, k)
                for k in ("c_s", "c_z", "c_s_inputs", "c_token", "blocks_per_ckpt")
            },
        },
        "optimizations": dict(sorted((decisions or {}).items())),
        "platform": platform,
        "versions": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "triton": _version("triton"),
            "cuequivariance_torch": _version("cuequivariance-torch"),
            "cuequivariance_ops_torch": _version("cuequivariance-ops-torch-cu12"),
            "opendde": __version__,
        },
    }


def context_digest(context: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(context, sort_keys=True, default=str).encode()
    ).hexdigest()


def cache_key(
    input_feature_dict: dict[str, Any],
    n_cycle: int,
    context: dict[str, Any],
    skip_features: frozenset[str] = frozenset(),
) -> str:
    """Hash of the context, ``N_cycle`` and every tensor feature the trunk reads
    (guidance features ``user_*`` and ``skip_features`` excluded)."""
    h = hashlib.sha256()
    h.update(
        ("format=%d;context=%s;" % (FORMAT_VERSION, context_digest(context))).encode()
    )
    h.update(("cycle=%d;" % n_cycle).encode())
    for k in sorted(input_feature_dict.keys()):
        v = input_feature_dict[k]
        if k.startswith("user_") or k in skip_features or not torch.is_tensor(v):
            continue
        h.update(k.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(str(v.dtype).encode())
        h.update(
            v.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
            if v.numel()
            else b""
        )
    return h.hexdigest()[:32]
