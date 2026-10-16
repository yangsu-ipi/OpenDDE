# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import pytest
import torch

from opendde.config.inference import build_inference_config
from opendde.model import opendde, trunk_cache

CONFIGS = build_inference_config(fill_required_with_null=True)


class _Model(torch.nn.Module):
    """A stand-in model with a verified checkpoint identity, as the runner
    binds after a strict load of hashed checkpoint bytes."""

    def __init__(self, calls):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.register_buffer("stat", torch.zeros(1))
        self.calls = calls
        self.configs = CONFIGS
        self.trunk_cache_identity = trunk_cache.ModelIdentity.capture(self, "c" * 64)

    def get_pairformer_output(self, input_feature_dict, **_kwargs):
        self.calls.append(1)
        input_feature_dict["d_lm"] = torch.full((2,), float(len(self.calls)))
        return torch.ones(1), torch.ones(2), torch.ones(3)

    def _maybe_foldcp_mesh(self):
        return None


def _module(calls):
    return _Model(calls)


def _features(**extra):
    features = {"residue_index": torch.arange(4), "restype": torch.ones(4, 2)}
    features.update(extra)
    return features


def _run(module, features):
    return opendde.OpenDDE._trunk_with_cache(
        module, features, N_cycle=1, inplace_safe=False, chunk_size=None
    )


def test_trunk_cache_is_off_without_a_directory(monkeypatch):
    monkeypatch.delenv("OPENDDE_TRUNK_CACHE", raising=False)
    calls = []
    module = _module(calls)
    _run(module, _features())
    _run(module, _features())
    assert len(calls) == 2


def test_trunk_cache_ignores_guidance_features(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE", str(tmp_path))
    calls = []
    module = _module(calls)
    first = _features(user_distance_restraint_index=torch.zeros(2, 0))
    second = _features(user_distance_restraint_index=torch.ones(2, 3))
    _run(module, first)
    out = _run(module, second)
    assert len(calls) == 1
    assert torch.equal(out[0], torch.ones(1))
    assert torch.equal(second["d_lm"], first["d_lm"])


def test_trunk_cache_key_follows_other_features_and_cycles(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE", str(tmp_path))
    calls = []
    module = _module(calls)
    _run(module, _features())
    _run(module, _features(restype=torch.zeros(4, 2)))
    opendde.OpenDDE._trunk_with_cache(
        module, _features(), N_cycle=2, inplace_safe=False, chunk_size=None
    )
    assert len(calls) == 3


def test_trunk_cache_read_only_mode_does_not_write(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE", str(tmp_path))
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE_MODE", "r")
    calls = []
    module = _module(calls)
    _run(module, _features())
    _run(module, _features())
    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_trunk_cache_is_bypassed_under_foldcp(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE", str(tmp_path))
    calls = []
    module = _module(calls)
    module._maybe_foldcp_mesh = lambda: object()
    _run(module, _features())
    _run(module, _features())
    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mode", ["r", "w", "rw"])
def test_trunk_cache_modes_are_accepted(monkeypatch, tmp_path, mode):
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE", str(tmp_path))
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE_MODE", mode)
    _run(_module([]), _features())


def test_trunk_cache_is_bypassed_with_several_model_seeds(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE", str(tmp_path))
    calls = []
    module = _module(calls)
    module.N_model_seed = 2
    _run(module, _features())
    _run(module, _features())
    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_trunk_cache_rejects_an_unknown_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE", str(tmp_path))
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE_MODE", "x")
    with pytest.raises(ValueError, match="OPENDDE_TRUNK_CACHE_MODE"):
        _run(_module([]), _features())


def test_trunk_cache_write_only_mode_never_reads(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE", str(tmp_path))
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE_MODE", "w")
    calls = []
    module = _module(calls)
    _run(module, _features())
    _run(module, _features())
    assert len(calls) == 2
    assert len(list(tmp_path.iterdir())) == 1


def test_trunk_cache_recomputes_when_the_file_is_damaged(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE", str(tmp_path))
    calls = []
    module = _module(calls)
    _run(module, _features())
    (cache_file,) = list(tmp_path.iterdir())
    cache_file.write_bytes(b"not a torch file")
    out = _run(module, _features())
    assert len(calls) == 2
    assert torch.equal(out[0], torch.ones(1))


def test_trunk_cache_write_failure_keeps_the_result(monkeypatch, tmp_path):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x")
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE", str(blocker))
    out = _run(_module([]), _features())
    assert torch.equal(out[0], torch.ones(1))
