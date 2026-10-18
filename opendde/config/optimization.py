# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Optimization mode, explicit-option record and per-optimization plan.

``--optimization_mode`` selects one numerical class:

- ``off``: unmodified OpenDDE behavior (the default).
- ``exact``: byte-identical prediction artifacts to ``off`` under the
  deterministic recipe; only changes proven identical may run.
- ``fast``: everything in ``exact`` plus kernels whose results differ from
  ``off`` in rounding only, each measured against an fp32 reference on the
  GPUs where it may run.
- ``big``: reserved; not available in this release.

Options a user supplies explicitly take precedence over any optimization that
would otherwise choose them (for example an explicit ``--triatt_kernel auto``
keeps the stock ``auto`` selection), so every entry point records which of
those options were supplied before defaults are applied.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional

OPTIMIZATION_MODES = ("off", "exact", "fast", "big")
AVAILABLE_OPTIMIZATION_MODES = ("off", "exact", "fast")

# Runtime options an optimization could otherwise choose, by the name used in
# ``get_default_runner``/``inference_jsons`` and on the ``pred`` command line,
# with the default applied when the caller omits them.
OVERLAPPING_OPTION_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "dtype": "bf16",
        "enable_tf32": False,
        "trimul_kernel": "auto",
        "triatt_kernel": "auto",
        "enable_cache": True,
        "enable_fusion": True,
    }
)
# The same options as dotted config keys (``runner.inference.run()``).
OVERLAPPING_CONFIG_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "dtype": "dtype",
        "enable_tf32": "enable_tf32",
        "triangle_multiplicative": "trimul_kernel",
        "triangle_attention": "triatt_kernel",
        "enable_diffusion_shared_vars_cache": "enable_cache",
        "enable_efficient_fusion": "enable_fusion",
    }
)
# Environment variables that select an implementation; setting one counts as
# an explicit choice.
OVERLAPPING_ENVIRONMENT = ("LAYERNORM_TYPE", "OPENDDE_TRUNK_CACHE")

# Optimizations available under each mode, registered as they are ported:
# name -> the modes that may run it.
EXACT_AND_ABOVE = ("exact", "fast", "big")
FAST_AND_ABOVE = ("fast", "big")
OPTIMIZATIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        # Drop the [N_atom, N_atom] int64 ``bond_mask`` feature, which no
        # inference consumer reads, before the feature dict leaves the host.
        "drop_bond_mask": EXACT_AND_ABOVE,
        # Evaluate each class of bitwise-identical (padding) template slots
        # once; the stock accumulation over all slots is kept.
        "template_dedup": EXACT_AND_ABOVE,
        # Encode each confidence JSON once (json.dumps) and write it in one
        # call instead of streaming json.dump; the bytes are identical.
        "json_oneshot": EXACT_AND_ABOVE,
        # Structural-token role-pair projections: one host read per call
        # instead of a torch.any read per role pair.
        "structural_token_sync": EXACT_AND_ABOVE,
        # Diffusion sampler: decide each step's gamma from one host copy of the
        # noise schedule instead of reading a device element per step.
        "host_noise_schedule": EXACT_AND_ABOVE,
        # Diffusion denoiser: compute its step-invariant conditioning tensors
        # (pair biases, adaptive-LayerNorm terms, gates, atom conditioning)
        # once per sampler call (opendde.model.step_invariant).
        "dit_hoist": EXACT_AND_ABOVE,
        # Diffusion denoiser: prepare the token transformer's pair input
        # (normalize, permute, contiguous copy) once per sampler call; it is
        # reused only while its source is the same tensor, as under dit_hoist.
        "zprep_hoist": EXACT_AND_ABOVE,
        # Triangle multiplication: Triton gated projections around the stock
        # cuEquivariance LayerNorms and cuBLAS contraction, bit-identical to
        # it on the rows of opendde.model.optimized.dispatch.
        "trimul_triton": EXACT_AND_ABOVE,
        # Pair transition: one fused SiLU-multiply kernel between the stock
        # GEMMs, bit-identical on the rows of opendde.model.optimized.dispatch.
        "transition_triton": EXACT_AND_ABOVE,
        # Triangle attention: a CUDA kernel bit-identical to cuEquivariance's
        # on the rows of opendde.model.optimized.dispatch.
        "triattn_exact": EXACT_AND_ABOVE,
        # Triangle attention: the flash triangle-attention Triton kernel in
        # place of cuEquivariance's (rounding differs; error against fp32 as
        # stock's) on the rows of opendde.model.optimized.dispatch.
        "triattn_k2b": FAST_AND_ABOVE,
        # Triangle attention on A100: the kit's sm_80 CUDA kernel in place of
        # cuEquivariance's (rounding differs; error against fp32 as stock's) on
        # the rows of opendde.model.optimized.dispatch.
        "triattn_native": FAST_AND_ABOVE,
    }
)
# The overlapping options whose kernel choice an optimization replaces: an
# explicit choice of any of them keeps the stock kernels.
OPTIMIZATION_OVERLAPS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "trimul_triton": ("trimul_kernel",),
        "triattn_exact": ("triatt_kernel",),
        "triattn_k2b": ("triatt_kernel",),
        "triattn_native": ("triatt_kernel",),
    }
)


class _Unset:
    """Marks an omitted keyword argument in the Python entry points."""

    _instance: Optional["_Unset"] = None

    def __new__(cls) -> "_Unset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<default>"

    def __bool__(self) -> bool:
        return False


UNSET: Any = _Unset()


def resolve_overlapping_options(
    supplied: Mapping[str, Any],
    explicit_options: Optional[Iterable[str]] = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return the effective option values and the sorted explicit option names.

    ``supplied`` maps each overlapping option to the caller's value, ``UNSET``
    where the caller omitted it. When ``explicit_options`` is given (the
    command line, which always passes concrete values), it names the explicit
    options instead of the omitted-argument markers.
    """
    unknown = set(supplied) - set(OVERLAPPING_OPTION_DEFAULTS)
    if unknown:
        raise ValueError(f"not overlapping options: {sorted(unknown)}")
    if explicit_options is None:
        explicit = {name for name, value in supplied.items() if value is not UNSET}
    else:
        explicit = set(explicit_options)
        unknown = explicit - set(OVERLAPPING_OPTION_DEFAULTS)
        if unknown:
            raise ValueError(f"not overlapping options: {sorted(unknown)}")
    values = {
        name: OVERLAPPING_OPTION_DEFAULTS[name] if value is UNSET else value
        for name, value in supplied.items()
    }
    return values, sorted(explicit)


def explicit_config_keys(keys: Iterable[str]) -> list[str]:
    """Overlapping options named by dotted config overrides."""
    return sorted(
        {OVERLAPPING_CONFIG_KEYS[k] for k in keys if k in OVERLAPPING_CONFIG_KEYS}
    )


def explicit_from_arg_str(arg_str: str) -> list[str]:
    """Overlapping options named in a ``runner.inference.run()`` argument
    string (``--key value`` or ``--key=value``)."""
    import shlex

    keys = [
        arg[2:].split("=", 1)[0] for arg in shlex.split(arg_str) if arg.startswith("--")
    ]
    return explicit_config_keys(keys)


def explicit_environment(
    environ: Optional[Mapping[str, str]] = None,
) -> list[str]:
    """Implementation-selecting environment variables that are set."""
    environ = os.environ if environ is None else environ
    return [f"env:{name}" for name in OVERLAPPING_ENVIRONMENT if environ.get(name)]


def validate_optimization_mode(mode: str, *, foldcp_enabled: bool) -> str:
    """Reject unknown and unavailable modes, and any mode other than ``off``
    under Fold-CP. Call before model construction."""
    if mode not in OPTIMIZATION_MODES:
        raise ValueError(
            f"optimization_mode must be one of {OPTIMIZATION_MODES}; got {mode!r}."
        )
    if mode not in AVAILABLE_OPTIMIZATION_MODES:
        raise ValueError(
            f"optimization_mode {mode!r} is not available in this release; "
            f"use one of {AVAILABLE_OPTIMIZATION_MODES}."
        )
    if foldcp_enabled and mode != "off":
        raise ValueError(
            f"optimization_mode {mode!r} is not supported with Fold-CP: no "
            "optimization is qualified with Fold-CP's sharded pair state and "
            "rank-synchronized sampler. Use --optimization_mode off."
        )
    return mode


@dataclass(frozen=True)
class Decision:
    enabled: bool
    reason: str


@dataclass(frozen=True)
class OptimizationPlan:
    """The resolved optimization choices for one runner."""

    mode: str
    capability: Optional[tuple[int, int]]
    device_name: Optional[str]
    explicit: tuple[str, ...]
    decisions: Mapping[str, Decision] = field(default_factory=dict)

    def enabled(self, name: str) -> bool:
        decision = self.decisions.get(name)
        return bool(decision and decision.enabled)

    def summary(self) -> str:
        if not self.decisions:
            body = "no optimizations apply; the stock path runs"
        else:
            body = ", ".join(
                f"{name}={'on' if d.enabled else 'off'} ({d.reason})"
                for name, d in sorted(self.decisions.items())
            )
        explicit = ", ".join(self.explicit) or "none"
        return (
            f"optimization_mode={self.mode} on {self.device_name or 'cpu'}: "
            f"{body}; explicit options: {explicit}"
        )


def optimization_enabled(configs: Any, name: str) -> bool:
    """Whether the runner enabled ``name`` for this run.

    The runner publishes its plan's decisions as ``configs.optimization_decisions``;
    code that sees only the configuration (data pipeline, model) reads them
    here. Without a published plan every optimization is off."""
    if name not in OPTIMIZATIONS:
        raise KeyError(f"unknown optimization {name!r}")
    decisions = getattr(configs, "optimization_decisions", None) or {}
    return bool(decisions.get(name, False))


def build_optimization_plan(configs: Any, device: Any) -> OptimizationPlan:
    """Resolve each registered optimization for a validated configuration.

    ``off`` runs nothing. Under another mode an optimization runs unless the
    mode does not permit it or the user switched it off through
    ``optimization.<name>``.
    """
    mode = getattr(configs, "optimization_mode", "off")
    capability = name = None
    if getattr(device, "type", None) == "cuda":
        import torch

        try:
            index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            capability = tuple(torch.cuda.get_device_capability(index))
            name = torch.cuda.get_device_name(index)
        except (RuntimeError, AssertionError) as exc:
            # Unqueryable device: record no capability, so no optimization
            # that depends on one can be selected for it.
            name = f"cuda (capability unavailable: {exc})"
    explicit = tuple(getattr(configs, "explicit_options", None) or ())
    switches = getattr(configs, "optimization", None) or {}
    if not isinstance(switches, Mapping):
        switches = dict(switches)
    decisions = {}
    for opt, modes in OPTIMIZATIONS.items():
        if mode not in modes:
            decisions[opt] = Decision(False, f"not part of mode {mode}")
        elif switches.get(opt) is False:
            decisions[opt] = Decision(False, f"optimization.{opt}=false")
        elif overlap := [
            o for o in OPTIMIZATION_OVERLAPS.get(opt, ()) if o in explicit
        ]:
            decisions[opt] = Decision(False, f"explicit {', '.join(overlap)}")
        else:
            decisions[opt] = Decision(True, f"mode {mode}")
    return OptimizationPlan(
        mode=mode,
        capability=capability,
        device_name=name,
        explicit=explicit,
        decisions=MappingProxyType(decisions),
    )
