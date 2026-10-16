# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Trunk cache identity: checkpoint digest, model-state guard and key context."""

import hashlib
from types import SimpleNamespace

import pytest
import torch

import runner.inference as inference
from opendde.config.inference import build_inference_config
from opendde.model import opendde, trunk_cache

CONFIGS = build_inference_config(fill_required_with_null=True)
CPU = torch.device("cpu")


class _Model(torch.nn.Module):
    def __init__(self, calls=None, digest="a" * 64):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.register_buffer("stat", torch.zeros(2))
        self.calls = [] if calls is None else calls
        self.configs = CONFIGS.model_copy(deep=True)
        self.trunk_cache_identity = trunk_cache.ModelIdentity.capture(self, digest)

    def get_pairformer_output(self, input_feature_dict, **_kwargs):
        self.calls.append(1)
        return torch.ones(1), torch.ones(2), torch.ones(3)

    def _maybe_foldcp_mesh(self):
        return None


def _add_in_place(tensor, value):
    with torch.no_grad():
        tensor.add_(value)


def _features():
    return {"residue_index": torch.arange(4), "restype": torch.ones(4, 2)}


def _run(model, features=None):
    return opendde.OpenDDE._trunk_with_cache(
        model, features or _features(), N_cycle=1, inplace_safe=False, chunk_size=None
    )


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE", str(tmp_path))
    monkeypatch.delenv("OPENDDE_TRUNK_CACHE_MODE", raising=False)
    monkeypatch.delenv("LAYERNORM_TYPE", raising=False)
    return tmp_path


# ------------------------------------------------------------- checkpoint


def test_read_checkpoint_hashes_the_loaded_bytes(tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save({"model": {"w": torch.arange(3.0)}}, path)
    checkpoint, digest = trunk_cache.read_checkpoint(str(path), "cpu")
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert torch.equal(checkpoint["model"]["w"], torch.arange(3.0))


def _runner(tmp_path, model, strict=True):
    path = tmp_path / "ckpt.pt"
    torch.save({"model": {k: v.clone() for k, v in model.state_dict().items()}}, path)
    configs = SimpleNamespace(load_strict=strict)
    return path, SimpleNamespace(
        configs=configs, device=CPU, model=model, print=lambda *_a, **_k: None
    )


def test_runner_hashes_the_checkpoint_only_with_the_cache_on(monkeypatch, tmp_path):
    model = _Model()
    path, fake = _runner(tmp_path, model)
    monkeypatch.setattr(inference, "resolve_checkpoint_path", lambda _c: str(path))
    monkeypatch.delenv("OPENDDE_TRUNK_CACHE", raising=False)

    def no_hash(*_a, **_k):
        raise AssertionError("the default path must not hash the checkpoint")

    monkeypatch.setattr(trunk_cache, "read_checkpoint", no_hash)
    inference.InferenceRunner.load_checkpoint(fake)
    assert model.trunk_cache_identity is None


def test_runner_binds_an_identity_after_a_strict_hashed_load(monkeypatch, tmp_path):
    model = _Model()
    path, fake = _runner(tmp_path, model)
    monkeypatch.setattr(inference, "resolve_checkpoint_path", lambda _c: str(path))
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE", str(tmp_path / "cache"))
    inference.InferenceRunner.load_checkpoint(fake)
    identity = model.trunk_cache_identity
    assert identity.checkpoint_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity.mismatch(model) is None


def test_runner_clears_the_identity_after_a_non_strict_load(monkeypatch, tmp_path):
    model = _Model()
    path, fake = _runner(tmp_path, model, strict=False)
    monkeypatch.setattr(inference, "resolve_checkpoint_path", lambda _c: str(path))
    monkeypatch.setenv("OPENDDE_TRUNK_CACHE", str(tmp_path / "cache"))
    inference.InferenceRunner.load_checkpoint(fake)
    assert model.trunk_cache_identity is None


# ------------------------------------------------------------- model guard


def test_unchanged_model_matches():
    model = _Model()
    assert model.trunk_cache_identity.mismatch(model) is None


@pytest.mark.parametrize(
    "change, text",
    [
        (lambda m: _add_in_place(m.lin.weight, 1), "changed"),
        (lambda m: m.stat.add_(1), "changed"),
        (
            lambda m: setattr(
                m.lin, "weight", torch.nn.Parameter(m.lin.weight.detach().clone())
            ),
            "replaced",
        ),
        (lambda m: setattr(m, "stat", m.stat.clone()), "replaced"),
        (lambda m: setattr(m.lin.weight, "data", torch.zeros(2, 2)), "changed"),
        (lambda m: m.register_buffer("extra", torch.zeros(1)), "set changed"),
    ],
    ids=[
        "in-place-param",
        "in-place-buffer",
        "param-replaced",
        "buffer-replaced",
        "data-reassigned",
        "tensor-added",
    ],
)
def test_model_changes_after_loading_are_detected(change, text):
    model = _Model()
    change(model)
    reason = model.trunk_cache_identity.mismatch(model)
    assert reason is not None and text in reason


def test_raw_storage_writes_through_data_are_not_certified():
    """Documented limit: a write through ``.data`` bumps no version counter, so
    the guard cannot see it; cache use is limited to the runner lifecycle."""
    model = _Model()
    model.lin.weight.data.add_(1)
    assert model.trunk_cache_identity.mismatch(model) is None


def test_a_replaced_parameter_with_the_same_version_is_detected():
    model = _Model()
    old = model.lin.bias
    new = torch.nn.Parameter(old.detach().clone())
    while new._version < old._version:  # match the counter: identity must decide
        _add_in_place(new, 0)
    assert new._version == old._version and torch.equal(new, old)
    model.lin.bias = new
    assert "replaced" in model.trunk_cache_identity.mismatch(model)


# ------------------------------------------------------------- key


def test_key_depends_on_each_context_part():
    base = trunk_cache.cache_context(CONFIGS, CPU, "a" * 64, {"x": True})
    key = trunk_cache.cache_key(_features(), 1, base)
    assert key == trunk_cache.cache_key(_features(), 1, base)
    variants = [
        trunk_cache.cache_context(CONFIGS, CPU, "b" * 64, {"x": True}),
        trunk_cache.cache_context(CONFIGS, CPU, "a" * 64, {"x": False}),
    ]
    for field, value in (
        ("dtype", "fp32"),
        ("enable_tf32", True),
        ("triangle_attention", "torch"),
        ("deterministic", True),
    ):
        configs = CONFIGS.model_copy(deep=True)
        setattr(configs, field, value)
        variants.append(trunk_cache.cache_context(configs, CPU, "a" * 64, {"x": True}))
    keys = {trunk_cache.cache_key(_features(), 1, v) for v in variants}
    assert key not in keys and len(keys) == len(variants)
    assert trunk_cache.cache_key(_features(), 2, base) != key


def test_key_context_names_platform_versions_and_layernorm(monkeypatch):
    context = trunk_cache.cache_context(CONFIGS, CPU, "a" * 64)
    assert context["versions"]["torch"] == torch.__version__
    assert context["versions"]["opendde"]
    assert context["platform"] == {"device_type": "cpu"}
    assert context["settings"]["layernorm_type"] == "torch"
    monkeypatch.setenv("LAYERNORM_TYPE", "fast_layernorm")
    other = trunk_cache.cache_context(CONFIGS, CPU, "a" * 64)
    assert trunk_cache.context_digest(other) != trunk_cache.context_digest(context)


def test_key_ignores_guidance_and_skipped_features():
    context = trunk_cache.cache_context(CONFIGS, CPU, "a" * 64)
    plain = trunk_cache.cache_key(_features(), 1, context, frozenset({"tfg"}))
    extra = dict(_features(), user_x=torch.ones(3), tfg=torch.ones(1))
    assert trunk_cache.cache_key(extra, 1, context, frozenset({"tfg"})) == plain


# ------------------------------------------------------------- cache use


def test_identity_less_model_bypasses_the_cache(cache):
    model = _Model()
    model.trunk_cache_identity = None
    _run(model)
    _run(model)
    assert len(model.calls) == 2 and list(cache.iterdir()) == []


def test_hit_after_miss_and_bypass_after_mutation(cache):
    model = _Model()
    _run(model)
    _run(model)
    assert len(model.calls) == 1
    _add_in_place(model.lin.weight, 1)
    _run(model)
    assert len(model.calls) == 2


def test_cache_misses_across_checkpoints_layernorm_and_decisions(cache, monkeypatch):
    first = _Model(digest="a" * 64)
    _run(first)
    other = _Model(calls=first.calls, digest="b" * 64)  # another checkpoint
    _run(other)
    assert len(first.calls) == 2
    monkeypatch.setenv("LAYERNORM_TYPE", "fast_layernorm")
    _run(first)
    assert len(first.calls) == 3
    monkeypatch.delenv("LAYERNORM_TYPE")
    first.configs.optimization_decisions = {"drop_bond_mask": True}
    _run(first)
    assert len(first.calls) == 4
    assert len(list(cache.iterdir())) == 4


def test_entries_with_another_format_or_context_are_recomputed(cache):
    model = _Model()
    _run(model)
    (path,) = list(cache.iterdir())
    blob = torch.load(path, weights_only=False)
    blob["context_digest"] = "0" * 64
    torch.save(blob, path)
    _run(model)
    assert len(model.calls) == 2
    blob = torch.load(path, weights_only=False)  # rewritten by the miss
    del blob["format"]
    torch.save(blob, path)
    _run(model)
    assert len(model.calls) == 3
