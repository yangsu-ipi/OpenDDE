# SPDX-License-Identifier: Apache-2.0
"""Tests for the private, opt-in instrumentation hooks."""

from collections import Counter

import pytest
import torch

from opendde.model.generator import InferenceNoiseScheduler, sample_diffusion
from opendde.tfg.potentials import CLASS_REGISTRY, Potential, register
from opendde.utils import instrumentation

if "_MockGeoPotential" not in CLASS_REGISTRY:

    @register
    class _MockGeoPotential(Potential):
        """Coordinate-only test potential: energy = scale * ||coords||^2."""

        def _eval(self, coords, feats, params, need_grad):
            scale = float(params.get("scale", 1.0))
            energy = scale * (coords**2).flatten(start_dim=-2).sum(dim=-1)
            if need_grad:
                return energy, 2.0 * scale * coords
            return energy


def _mock_denoiser(*, x_noisy, t_hat_noise_level, **kwargs):
    return x_noisy * 0.9


def _inputs(n_token=2, c=8, n_atom=6, n_step=4):
    g = torch.Generator().manual_seed(123)
    s_inputs = torch.randn(n_token, c, generator=g)
    z_trunk = torch.randn(n_token, n_token, c, generator=g)
    return dict(
        denoise_net=_mock_denoiser,
        input_feature_dict={"atom_to_token_idx": torch.zeros(n_atom, dtype=torch.long)},
        s_inputs=s_inputs,
        s_trunk=s_inputs,
        z_trunk=z_trunk,
        pair_z=None,
        p_lm=None,
        c_l=None,
        noise_schedule=InferenceNoiseScheduler()(N_step=n_step),
    )


def _tfg_cfg(eps_std):
    return {
        "enable": True,
        "rho": 0.0,
        "mu": 0.1,
        "mc": {"std": eps_std, "batch": 2},
        "steps": {
            "tfg_outer": 2,
            "tfg_inner": 1,
            "projection_outer": 0,
            "projection_inner": 0,
        },
        "terms": {"_MockGeoPotential": {"interval": 1, "weight": 0.01}},
    }


@pytest.fixture
def records():
    seen = []
    instrumentation.install_draw_recorder(
        lambda record, value: seen.append((record, value.clone()))
    )
    yield seen
    instrumentation.install_draw_recorder(None)


def _run(guidance_configs=None):
    return sample_diffusion(
        N_sample=3,
        diffusion_chunk_size=2,
        rollout_seed=7,
        guidance_configs=guidance_configs,
        **_inputs(),
    )


def test_hooks_are_no_ops_without_recorders():
    value = torch.arange(3)
    assert instrumentation.record_draw("site", "gaussian", value, sample_dim=0) is value
    instrumentation.set_draw_scope(step=1)
    with instrumentation.phase("stage"):
        pass


@pytest.mark.parametrize("eps_std", [None, 0.0, 0.5])
def test_draw_recorder_leaves_samples_unchanged(eps_std):
    cfg = None if eps_std is None else _tfg_cfg(eps_std)
    stock = _run(cfg)
    seen = []
    instrumentation.install_draw_recorder(lambda record, value: seen.append(record))
    try:
        traced = _run(cfg)
    finally:
        instrumentation.install_draw_recorder(None)
    assert seen
    assert torch.equal(stock, traced)


@pytest.mark.parametrize("eps_std", [None, 0.0, 0.5])
def test_draw_records_cover_every_site_with_scope(records, eps_std):
    cfg = None if eps_std is None else _tfg_cfg(eps_std)
    _run(cfg)
    n_chunk, n_step, n_outer = 2, 4, 2
    expected = {
        "sampler.initial_noise": n_chunk,
        "sampler.step_noise": n_chunk * n_step,
        "augmentation.rotation": n_chunk * n_step,
        "augmentation.translation": n_chunk * n_step,
    }
    if cfg is not None:
        expected["tfg.outer_noise"] = n_chunk * n_step * n_outer
        if eps_std:
            expected["tfg.eps"] = n_chunk * n_step
    assert Counter(r["site"] for r, _ in records) == expected

    for record, value in records:
        n_in_chunk = 2 if record["chunk"] == 0 else 1
        assert record["sample_offset"] == 2 * record["chunk"]
        assert value.shape[record["sample_dim"]] == n_in_chunk
        if record["site"] == "sampler.initial_noise":
            assert record["step"] is None
        else:
            assert record["step"] in range(n_step)
        if record["site"] == "tfg.outer_noise":
            assert record["outer"] in range(n_outer)
    rotations = [v for r, v in records if r["site"] == "augmentation.rotation"]
    assert all(v.shape[-2:] == (3, 3) for v in rotations)
