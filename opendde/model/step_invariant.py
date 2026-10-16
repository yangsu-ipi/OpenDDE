# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Step-invariant denoiser tensors, computed once per sample_diffusion call.

The diffusion sampler calls the denoiser once per step (more often under
guidance) with the same conditioning: the trunk single and pair outputs, the
cached pair conditioning and atom features. Several tensors the denoiser builds
from them, such as the pair biases of the token transformer, the atom
transformer's adaptive-LayerNorm terms and the token transformer's pair input,
are therefore the same at every step, and stock OpenDDE recomputes them each
time.

A memo site wraps the stock expression for one such tensor. The first
evaluation runs the stock producer and records its result together with a
fingerprint of the producer's inputs: for each tensor its storage address,
shape, strides, dtype, device and version counter, and the tensor itself, so
its storage stays alive and the address cannot name other memory. A later
evaluation at the same site returns the recorded result only if the inputs have
the same fingerprint and nothing has written into the recorded result since
(its version counters are unchanged). Any other evaluation runs the producer and
records again. Each site keeps one entry, dropped before its replacement is
computed, and the memo is discarded when the sampler call ends.

With deterministic kernels the producer gives the same bits for the same
inputs, so a reused result is bitwise the tensor stock would recompute. Calls
with autograd enabled, as in denoiser-path guidance, run stock code and leave
the memo untouched. The model weights are not part of the fingerprint: they do
not change during a sampler call.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Callable, Iterable, Iterator, Optional

import torch

# The memo optimizations: dit_hoist covers the conditioning tensors of the
# denoiser, zprep_hoist the token transformer's prepared pair input.
MEMO_OPTIMIZATIONS = ("dit_hoist", "zprep_hoist")

_MEMO: contextvars.ContextVar[Optional["StepInvariantMemo"]] = contextvars.ContextVar(
    "opendde_step_invariant_memo", default=None
)


def _fingerprint(values: tuple[Any, ...]) -> Optional[tuple[Any, ...]]:
    """The memo key of a producer's inputs, or None if they cannot be keyed."""
    key = []
    for value in values:
        if isinstance(value, torch.Tensor):
            if value.is_inference():  # inference tensors have no version counter
                return None
            key.append(
                (
                    "tensor",
                    value.data_ptr(),
                    tuple(value.shape),
                    value.stride(),
                    value.dtype,
                    value.device,
                    value._version,
                )
            )
        elif value is None or isinstance(value, (bool, int, float, str)):
            key.append(("value", type(value), value))
        else:
            raise TypeError(f"step-invariant memo input of type {type(value)}")
    return tuple(key)


def _result_versions(result: Any) -> Optional[tuple[int, ...]]:
    tensors = result if isinstance(result, tuple) else (result,)
    if not all(isinstance(t, torch.Tensor) and not t.is_inference() for t in tensors):
        return None
    return tuple(t._version for t in tensors)


class _Entry:
    __slots__ = ("owner", "inputs", "key", "result", "versions")

    def __init__(self, owner, inputs, key, result, versions):
        self.owner = owner
        self.inputs = inputs  # held: keeps the fingerprinted storage alive
        self.key = key
        self.result = result
        self.versions = versions


class StepInvariantMemo:
    """The memo of one sample_diffusion call; see the module docstring."""

    def __init__(self, names: frozenset[str]) -> None:
        self.names = names
        self._entries: dict[tuple[int, str], _Entry] = {}
        self.hits = 0
        self.records = 0

    def get(
        self,
        owner: torch.nn.Module,
        site: str,
        inputs: tuple[Any, ...],
        producer: Callable[[], Any],
    ) -> Any:
        """The result of ``producer()`` for ``owner``'s ``site``.

        ``inputs`` must list every tensor and setting the producer reads apart
        from ``owner``'s weights."""
        slot = (id(owner), site)
        key = _fingerprint(inputs)
        entry = self._entries.get(slot)
        if (
            entry is not None
            and key is not None
            and entry.owner is owner
            and entry.key == key
            and _result_versions(entry.result) == entry.versions
        ):
            self.hits += 1
            return entry.result
        self._entries.pop(slot, None)  # never hold two results for one site
        del entry
        result = producer()
        versions = _result_versions(result)
        if key is not None and versions is not None:
            self._entries[slot] = _Entry(owner, inputs, key, result, versions)
            self.records += 1
        return result

    def clear(self) -> None:
        self._entries.clear()


@contextlib.contextmanager
def step_invariant_memo(names: Iterable[str]) -> Iterator[None]:
    """Memoize the sites of the given optimizations for one sampler call."""
    enabled = frozenset(names)
    unknown = enabled - set(MEMO_OPTIMIZATIONS)
    if unknown:
        raise ValueError(f"not a step-invariant memo optimization: {sorted(unknown)}")
    if not enabled:
        yield
        return
    memo = StepInvariantMemo(enabled)
    token = _MEMO.set(memo)
    try:
        yield
    finally:
        _MEMO.reset(token)
        memo.clear()


def active(name: str) -> Optional[StepInvariantMemo]:
    """The memo serving optimization ``name`` here, or None to run stock code."""
    memo = _MEMO.get()
    if memo is None or name not in memo.names or torch.is_grad_enabled():
        return None
    return memo
