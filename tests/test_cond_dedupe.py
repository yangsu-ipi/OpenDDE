# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""cond_dedupe: the sampler's single conditioning computed for one sample."""

import pytest
import torch

from opendde.model.generator import sample_diffusion
from opendde.model.modules import diffusion
from opendde.model.step_invariant import step_invariant_memo
from tests.test_step_invariant_memo import _guidance, _inputs, _module


@pytest.fixture
def sm120_row(monkeypatch):
    monkeypatch.setattr(diffusion.dispatch, "available", lambda site, device: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))


@pytest.mark.parametrize(
    "n_token, applies",
    [(1, True), (64, True), (65, False), (76, False), (77, True), (1120, True)]
    + [(1121, False), (2300, False)],
)
def test_runs_only_inside_the_measured_token_windows(sm120_row, n_token, applies):
    device = torch.device("cuda", 0)
    assert diffusion._cond_dedupe_applies(5, n_token, device) is applies


def test_runs_only_for_the_measured_sample_count(sm120_row):
    device = torch.device("cuda", 0)
    assert not diffusion._cond_dedupe_applies(4, 200, device)


def test_unmeasured_capability_and_cpu_run_stock(monkeypatch):
    assert not diffusion._cond_dedupe_applies(5, 200, torch.device("cpu"))
    monkeypatch.setattr(diffusion.dispatch, "available", lambda site, device: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    assert not diffusion._cond_dedupe_applies(5, 200, torch.device("cuda", 0))


def _sample(dedupe, monkeypatch, **kwargs):
    """Samples with cond_dedupe on or off, and the sample counts the
    conditioning was computed for."""
    monkeypatch.setattr(diffusion, "_cond_dedupe_applies", lambda *args: True)
    module = _module()
    inputs = _inputs(module, True)
    counts = []
    forward = module.diffusion_conditioning.forward

    def counting(t_hat_noise_level, *args, **kw):
        counts.append(t_hat_noise_level.shape[-1])
        return forward(t_hat_noise_level, *args, **kw)

    monkeypatch.setattr(module.diffusion_conditioning, "forward", counting)
    torch.manual_seed(3)
    with torch.no_grad(), diffusion.cond_dedupe(dedupe):  # guidance enables grad
        with step_invariant_memo(kwargs.pop("memo", [])):
            out = sample_diffusion(N_sample=5, **inputs, **kwargs)
    return out, counts


@pytest.mark.parametrize("fusion", [True, False])
@pytest.mark.parametrize("inplace_safe", [True, False])
@pytest.mark.parametrize("memo", [[], ["dit_hoist", "zprep_hoist"]])
def test_samples_match_stock_up_to_rounding(monkeypatch, fusion, inplace_safe, memo):
    kwargs = dict(enable_efficient_fusion=fusion, inplace_safe=inplace_safe)
    stock, stock_counts = _sample(False, monkeypatch, memo=memo, **kwargs)
    deduped, counts = _sample(True, monkeypatch, memo=memo, **kwargs)
    assert set(stock_counts) == {5} and set(counts) == {1}
    assert deduped.shape == stock.shape
    torch.testing.assert_close(deduped, stock, rtol=1e-4, atol=1e-4)


def test_guided_samples_match_stock_up_to_rounding(monkeypatch):
    kwargs = dict(enable_efficient_fusion=True, inplace_safe=True)
    kwargs["guidance_configs"] = _guidance()
    stock, _ = _sample(False, monkeypatch, **kwargs)
    deduped, counts = _sample(True, monkeypatch, **kwargs)
    assert 1 in counts
    torch.testing.assert_close(deduped, stock, rtol=1e-4, atol=1e-4)


def test_a_per_sample_noise_level_runs_stock(monkeypatch):
    # Only a noise level expanded over the samples (stride 0) is deduplicated.
    monkeypatch.setattr(diffusion, "_cond_dedupe_applies", lambda *args: True)
    module = _module()
    inputs = _inputs(module, True)
    counts = []
    forward = module.diffusion_conditioning.forward

    def counting(t_hat_noise_level, *args, **kw):
        counts.append(t_hat_noise_level.shape[-1])
        return forward(t_hat_noise_level, *args, **kw)

    monkeypatch.setattr(module.diffusion_conditioning, "forward", counting)
    n_atom = inputs["input_feature_dict"]["atom_to_token_idx"].shape[0]
    with torch.no_grad(), diffusion.cond_dedupe(True):
        module(
            x_noisy=torch.randn(5, n_atom, 3),
            t_hat_noise_level=torch.linspace(1.0, 5.0, 5),
            input_feature_dict=inputs["input_feature_dict"],
            s_inputs=inputs["s_inputs"],
            s_trunk=inputs["s_trunk"],
            z_trunk=None,
            pair_z=inputs["pair_z"],
            p_lm=inputs["p_lm"],
            c_l=inputs["c_l"],
        )
    assert counts == [5]
