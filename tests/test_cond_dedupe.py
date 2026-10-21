# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""cond_dedupe: the sampler's single conditioning computed for one sample."""

import pytest
import torch

from opendde.model.generator import sample_diffusion
from opendde.model.modules import diffusion
from opendde.model.step_invariant import step_invariant_memo
from tests.test_step_invariant_memo import _guidance, _inputs, _module


def test_runs_where_the_dispatch_row_allows(monkeypatch):
    assert not diffusion._cond_dedupe_applies(torch.device("cpu"))
    assert [row.capability for row in diffusion.dispatch.ROWS["cond_dedupe"]] == [
        (12, 0)
    ]
    monkeypatch.setattr(diffusion.dispatch, "available", lambda site, device: True)
    assert diffusion._cond_dedupe_applies(torch.device("cuda", 0))


def _conditioning(module, inputs, x_noisy, t_hat, dedupe):
    """The single conditioning the denoiser computes for these samples."""
    seen = []
    forward = module.diffusion_conditioning.forward

    def recording(*args, **kw):
        out = forward(*args, **kw)
        seen.append(out[0])
        return out

    module.diffusion_conditioning.forward = recording
    try:
        with torch.no_grad(), diffusion.cond_dedupe(dedupe):
            module(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=inputs["input_feature_dict"],
                s_inputs=inputs["s_inputs"],
                s_trunk=inputs["s_trunk"],
                z_trunk=None,
                pair_z=inputs["pair_z"],
                p_lm=inputs["p_lm"],
                c_l=inputs["c_l"],
            )
    finally:
        del module.diffusion_conditioning.forward
    return seen[0]


def test_the_conditioning_is_stocks_one_sample_conditioning(monkeypatch):
    # The basis of standard (b) for this item (plan revision 118): every value
    # it changes is one stock computes when run with one sample.
    monkeypatch.setattr(diffusion, "_cond_dedupe_applies", lambda *args: True)
    module = _module()
    inputs = _inputs(module, True)
    n_atom = inputs["input_feature_dict"]["atom_to_token_idx"].shape[0]
    x_noisy = torch.randn(5, n_atom, 3)
    t_hat = torch.tensor([3.0])
    deduped = _conditioning(module, inputs, x_noisy, t_hat.expand(5), True)
    one_sample = _conditioning(module, inputs, x_noisy[:1], t_hat, False)
    assert deduped.shape[-3] == 1
    assert torch.equal(deduped, one_sample)


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
