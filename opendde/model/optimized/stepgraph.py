# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Anthropic, PBC
"""The sampler's denoiser step as one CUDA graph per sampler call (stepgraph).

An ``exact`` item: the outputs are byte-identical to the eager sampler
(D1-D15 on RTX PRO 6000 Blackwell, 2026-10-23). Within one
``sample_diffusion`` call every denoiser call has the same signature and the
step-invariant tensors of ``dit_hoist`` are recorded by the first call. The
first call runs eagerly; the second runs eagerly, is then captured on static
copies of ``x_noisy`` and ``t_hat`` and replayed once, and the replay must be
bitwise the eager result; the third is replayed and also run eagerly, and
must again be bitwise equal; the rest are replayed (copy in, one graph
launch, copy out). Any mismatch or capture error falls back to the eager
denoiser for the rest of the call. The atom-to-token mean of the atom
encoder, whose deterministic ``scatter_add`` synchronizes with the host and
cannot be captured, is served inside this context by a segment sum with the
same order of fp32 additions (``opendde.model.utils``). Guided sampling, which
runs autograd inside the step, and Fold-CP calls run eagerly.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Iterator, Optional

import torch

from opendde.model.optimized import dispatch
from opendde.utils.logger import get_logger

logger = get_logger(__name__)
SITE = "stepgraph"

# The segment-sum tables of the current sampler call (atom_to_token_idx data
# pointer -> (gather index, valid mask, counts)); None outside a graphed call.
SEGMENTS: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "opendde_stepgraph_segments", default=None
)


def segment_table(atom_to_token_idx: torch.Tensor, n_token: int):
    """Per token, its atoms in ascending position (the order of the stable
    sort the deterministic scatter_add accumulates in), padded."""
    idx = atom_to_token_idx.reshape(-1)
    order = torch.argsort(idx, stable=True)
    counts = torch.bincount(idx, minlength=n_token)
    max_k = int(counts.max().item()) if idx.numel() else 0
    starts = torch.cumsum(counts, 0) - counts
    j = torch.arange(max_k, device=idx.device)
    pos = starts[:, None] + j[None, :]
    valid = j[None, :] < counts[:, None]
    gather = order[pos.clamp(max=max(idx.numel() - 1, 0))]
    gather = torch.where(valid, gather, torch.zeros_like(gather))
    return gather, valid, counts


def segment_mean(x_atom: torch.Tensor, table, n_token: int) -> torch.Tensor:
    """scatter_mean over dim -2, bitwise the deterministic scatter_add path:
    per token, 0 + a_0 + a_1 + ... in fp32 in ascending atom position, then
    divided by max(count, 1)."""
    gather, valid, counts = table
    acc = torch.zeros(
        x_atom.shape[:-2] + (n_token, x_atom.shape[-1]),
        dtype=torch.float32,
        device=x_atom.device,
    )
    for j in range(gather.shape[1]):
        term = x_atom.index_select(-2, gather[:, j]).float()
        acc = torch.where(valid[:, j, None], acc + term, acc)
    out = acc.to(x_atom.dtype)
    count = counts.clamp(min=1).to(x_atom.dtype)[:, None]
    return out.true_divide_(count)


@contextlib.contextmanager
def segments() -> Iterator[dict]:
    tables: dict = {}
    token = SEGMENTS.set(tables)
    try:
        yield tables
    finally:
        SEGMENTS.reset(token)


class StepGraph:
    """One sampler call's denoiser graph."""

    def __init__(self) -> None:
        self.calls = 0
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.disabled = False
        self.x = self.t_base = self.t = self.out = None
        self.replays = 0

    def _eager(self, denoise_net, x_noisy, t_hat, kwargs):
        return denoise_net(x_noisy=x_noisy, t_hat_noise_level=t_hat, **kwargs)

    def _disable(self, reason: str) -> None:
        self.disabled = True
        self.graph = None
        dispatch.fallback(SITE, reason)

    def _static_t(self, t_hat: torch.Tensor):
        # Keep the sampler's layout: one level expanded over the samples.
        if t_hat.dim() >= 1 and t_hat.stride(-1) == 0:
            base = t_hat[..., :1].clone()
            return base, base.expand(t_hat.shape)
        base = t_hat.clone()
        return base, base

    def _copy_t(self, t_hat: torch.Tensor) -> None:
        src = t_hat[..., :1] if self.t is not self.t_base else t_hat
        self.t_base.copy_(src)

    def __call__(self, denoise_net, x_noisy, t_hat_noise_level, **kwargs):
        t_hat = t_hat_noise_level
        self.calls += 1
        if self.disabled or self.calls == 1:
            return self._eager(denoise_net, x_noisy, t_hat, kwargs)
        if self.graph is None:
            eager = self._eager(denoise_net, x_noisy, t_hat, kwargs)
            try:
                self.x = x_noisy.clone()
                self.t_base, self.t = self._static_t(t_hat)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    self._eager(denoise_net, self.x, self.t, kwargs)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.out = self._eager(denoise_net, self.x, self.t, kwargs)
                graph.replay()
                if not torch.equal(self.out, eager):
                    self._disable("capture replay differs from the eager step")
                    return eager
                self.graph = graph
                logger.info("stepgraph: denoiser step captured")
            except Exception as exc:  # noqa: BLE001 -- e.g. an op that cannot be captured
                self._disable(f"capture failed: {exc!r}"[:300])
            return eager
        self.x.copy_(x_noisy)
        self._copy_t(t_hat)
        self.graph.replay()
        self.replays += 1
        out = self.out.clone()
        if self.calls == 3:
            eager = self._eager(denoise_net, x_noisy, t_hat, kwargs)
            if not torch.equal(out, eager):
                self._disable("replay on new inputs differs from the eager step")
                return eager
        return out


def eligible(device: torch.device, guided: bool, inplace_safe: bool) -> bool:
    if device.type != "cuda":
        return False
    if torch.is_grad_enabled():
        return dispatch.fallback(SITE, "autograd enabled")
    if guided:
        return dispatch.fallback(SITE, "guidance enabled")
    if not inplace_safe:
        return dispatch.fallback(SITE, "inplace_safe off")
    return dispatch.available(SITE, device)
