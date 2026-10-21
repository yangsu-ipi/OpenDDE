# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Anthropic, PBC
"""dit_fused: the fused token-transformer stack against the stock forward.

The Triton row kernels need a GPU; here they are replaced by torch versions
of the same formulas, so the packing (merged GEMMs, folded LayerNorm weights,
conditioning order, residual and gate wiring) is checked on the CPU.
"""

import pytest
import torch
import torch.nn.functional as F

from opendde.model.modules.transformer import DiffusionTransformer
from opendde.model.optimized import dit_fused
from opendde.model.step_invariant import step_invariant_memo


def _ln(a, eps):
    return F.layer_norm(a, a.shape[-1:], None, None, eps)


def _rows(t, m):
    return t.repeat(m // t.shape[0], 1)


@pytest.fixture
def torch_kernels(monkeypatch):
    def adaln(a, x1, x2, eps):
        n = a.shape[0]
        return torch.sigmoid(_rows(x1, n)) * _ln(a, eps) + _rows(x2, n)

    def gate(o, g, N):
        B, H, _, Cd = o.shape
        return o.transpose(1, 2).reshape(B * N, H * Cd) * torch.sigmoid(g)

    def swiglu(x1, x2):
        return F.silu(x1) * x2

    def resgate_adaln(gl, x, res, x1, x2, eps):
        res += torch.sigmoid(_rows(gl, res.shape[0])) * x
        return None if x1 is None else adaln(res, x1, x2, eps)

    for name, fn in (
        ("_adaln", adaln),
        ("_gate", gate),
        ("_swiglu", swiglu),
        ("_resgate_adaln", resgate_adaln),
    ):
        monkeypatch.setattr(dit_fused, name, fn)


def _module(n_blocks=2):
    torch.manual_seed(0)
    dt = DiffusionTransformer(c_a=768, c_s=384, c_z=8, n_blocks=n_blocks, n_heads=16)
    with torch.no_grad():
        for name, p in dt.named_parameters():
            if p.dim() >= 2:
                p.normal_(0, p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.normal_(1, 0.1)
            else:
                p.normal_(0, 0.1)
    return dt.eval()


@pytest.mark.parametrize("s_rows", [3, 1])
@pytest.mark.parametrize("fusion", [True, False])
@pytest.mark.parametrize("extra", [False, True])
def test_the_fused_stack_matches_stock(torch_kernels, s_rows, fusion, extra):
    dt = _module()
    N = 6
    a = torch.randn(3, N, 768)
    s = torch.randn(s_rows, N, 384)
    z = torch.randn(8, N, N) if fusion else torch.randn(N, N, 8)
    extra_bias = torch.randn(N, N) if extra else None
    with torch.no_grad(), step_invariant_memo(["dit_hoist"]):
        stock = dt(a, s, z, enable_efficient_fusion=fusion, extra_attn_bias=extra_bias)
        fused = dit_fused.FusedTokenStack(dt).forward(a, s, z, fusion, extra_bias)
    torch.testing.assert_close(fused, stock, rtol=1e-4, atol=1e-4)


def test_cpu_and_grad_calls_run_stock():
    dt = _module(n_blocks=1)
    a, s = torch.randn(2, 4, 768), torch.randn(2, 4, 384)
    assert not dit_fused.eligible(dt, a, s, None, None)  # CPU
    with torch.enable_grad():
        assert not dit_fused.eligible(dt, a, s, None, None)
    assert not dit_fused.eligible(dt, a, s, 32, 128)  # local attention
