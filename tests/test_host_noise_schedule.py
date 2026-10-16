# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""host_noise_schedule: the per-step gamma decision from a host copy, same samples."""

import numpy as np
import pytest
import torch

from opendde.model import generator
from opendde.model.generator import (
    InferenceNoiseScheduler,
    host_noise_schedule,
    sample_diffusion,
)
from opendde.tfg.potentials import CLASS_REGISTRY, Potential, register

if "_HostScheduleGeoPotential" not in CLASS_REGISTRY:

    @register
    class _HostScheduleGeoPotential(Potential):
        def _eval(self, coords, feats, params, need_grad):
            energy = (coords**2).flatten(start_dim=-2).sum(dim=-1)
            return (energy, 2.0 * coords) if need_grad else energy


def _inputs(n_step=8):
    g = torch.Generator().manual_seed(123)
    s = torch.randn(2, 8, generator=g)
    return dict(
        denoise_net=lambda *, x_noisy, t_hat_noise_level, **_: x_noisy * 0.9,
        input_feature_dict={"atom_to_token_idx": torch.zeros(6, dtype=torch.long)},
        s_inputs=s,
        s_trunk=s,
        z_trunk=torch.randn(2, 2, 8, generator=g),
        pair_z=None,
        p_lm=None,
        c_l=None,
        noise_schedule=InferenceNoiseScheduler()(N_step=n_step),
    )


def _guidance():
    return {
        "enable": True,
        "rho": 0.0,
        "mu": 0.1,
        "mc": {"std": 0.0, "batch": 1},
        "steps": {
            "tfg_outer": 1,
            "tfg_inner": 2,
            "projection_outer": 0,
            "projection_inner": 0,
        },
        "terms": {"_HostScheduleGeoPotential": {"interval": 1, "weight": 0.01}},
    }


def _sample(enabled, **kwargs):
    torch.manual_seed(0)
    np.random.seed(0)
    with host_noise_schedule(enabled):
        return sample_diffusion(N_sample=3, **kwargs, **_inputs())


def test_schedule_crosses_gamma_min():
    schedule = InferenceNoiseScheduler()(N_step=8)
    assert (schedule[1:] > 1.0).any() and (schedule[1:] <= 1.0).any()


@pytest.mark.parametrize("guided", [False, True])
def test_samples_are_identical_with_the_host_schedule(guided):
    kwargs = {"guidance_configs": _guidance()} if guided else {}
    assert torch.equal(_sample(False, **kwargs), _sample(True, **kwargs))


def test_switch_is_scoped_and_off_by_default():
    assert generator._HOST_NOISE_SCHEDULE.get() is False
    with host_noise_schedule(True):
        assert generator._HOST_NOISE_SCHEDULE.get() is True
        with host_noise_schedule(False):
            assert generator._HOST_NOISE_SCHEDULE.get() is False
        assert generator._HOST_NOISE_SCHEDULE.get() is True
    assert generator._HOST_NOISE_SCHEDULE.get() is False


def test_host_copy_is_taken_once_per_call(monkeypatch):
    copies = []
    original = torch.Tensor.to

    def counted(self, *args, **kwargs):
        if args and args[0] == "cpu":
            copies.append(tuple(self.shape))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", counted)
    _sample(True)
    assert copies.count((9,)) == 1  # the 9-element schedule of 8 steps
