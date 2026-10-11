"""Tests for the tier 1 comparison metrics (scripts/qualification/tier1_compare.py)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts" / "qualification")
)
import tier1_compare as t1  # noqa: E402


def test_tensor_error_identical_and_scaled():
    ref = torch.randn(64, 32, generator=torch.Generator().manual_seed(0))
    same = t1.tensor_error(ref, ref.clone())
    assert same["rel_fro"] == 0 and same["max_over_rms"] == 0 and same["finite"]
    bumped = ref.clone()
    bumped[0, 0] += 10.0
    err = t1.tensor_error(ref, bumped)
    rms = ref.square().mean().sqrt().item()
    assert err["max_over_rms"] == pytest.approx(10.0 / rms, rel=1e-5)
    # One outlier in 2,048 elements stays out of the 99.9th percentile.
    assert err["tail_over_rms"] == 0


def test_tensor_error_zero_reference_uses_absolute_values():
    ref = torch.zeros(10)
    other = torch.full((10,), 0.5)
    err = t1.tensor_error(ref, other)
    assert err["ref_rms"] == 0
    assert err["max_over_rms"] == pytest.approx(0.5)
    assert err["rel_fro"] == pytest.approx(other.norm().item())


def test_distogram_near_tie_is_separated_from_material_change():
    ref = torch.tensor([[0.501, 0.499, 0.0], [0.9, 0.05, 0.05], [0.2, 0.7, 0.1]])
    other = torch.tensor([[0.499, 0.501, 0.0], [0.9, 0.05, 0.05], [0.7, 0.2, 0.1]])
    err = t1.distogram_error(ref, other)
    assert err["argmax_changed"] == pytest.approx(2 / 3)
    # The first pair is a near tie (margin 0.002); the third has margin 0.5.
    assert err["argmax_changed_margin_0.01"] == pytest.approx(1 / 2)
    assert err["max_abs_prob"] == pytest.approx(0.5)


def test_head_error_identical_is_zero():
    gen = torch.Generator().manual_seed(1)
    head = {
        "plddt_logits": torch.randn(2, 5, 50, generator=gen),
        "resolved_logits": torch.randn(2, 5, 2, generator=gen),
        "pae_logits": torch.randn(2, 3, 3, 64, generator=gen).half(),
        "pde_logits": torch.randn(2, 3, 3, 64, generator=gen).half(),
        "ptm": torch.tensor([0.5, 0.6]),
        "iptm": torch.tensor([0.4, 0.3]),
    }
    err = t1.head_error(head, {k: v.clone() for k, v in head.items()})
    assert all(v == 0 for k, v in err.items() if not k.endswith("finite"))
    shifted = dict(head, iptm=head["iptm"] + 0.1)
    assert t1.head_error(head, shifted)["ranking_core_max_abs"] == pytest.approx(0.08)
