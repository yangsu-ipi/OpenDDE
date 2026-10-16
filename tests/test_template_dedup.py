# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""template_dedup: identical template slots evaluated once, output bit-identical."""

from unittest import mock

import pytest
import torch

from opendde.model.modules.pairformer import TemplateEmbedder

N, C_Z, T = 6, 16, 4


@pytest.fixture
def embedder():
    torch.manual_seed(0)
    module = TemplateEmbedder(n_blocks=1, c=8, c_z=C_Z).eval()
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0, 0.2)
    return module


def _slot(seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "template_aatype": torch.randint(0, 31, (N,), generator=g),
        "template_distogram": torch.rand(N, N, 39, generator=g),
        "template_pseudo_beta_mask": (torch.rand(N, N, generator=g) > 0.3).float(),
        "template_unit_vector": torch.randn(N, N, 3, generator=g),
        "template_backbone_frame_mask": (torch.rand(N, N, generator=g) > 0.3).float(),
    }


def _features(pattern):
    """pattern[t]: the seed of slot t; equal seeds give bitwise-equal slots."""
    slots = [_slot(s) for s in pattern]
    feats = {k: torch.stack([s[k] for s in slots]) for k in slots[0]}
    feats["asym_id"] = torch.tensor([0, 0, 0, 1, 1, 1])
    return feats


def _z():
    return torch.randn(N, N, C_Z, generator=torch.Generator().manual_seed(9))


def _forward(module, feats, dedup):
    module.dedup_identical_slots = dedup
    calls = []
    original = module.single_template_forward

    def counted(*args, **kwargs):
        calls.append(kwargs["template_id"])
        return original(*args, **kwargs)

    with mock.patch.object(module, "single_template_forward", counted):
        with torch.no_grad():
            out = module(feats, _z())
    return out, calls


@pytest.mark.parametrize(
    "pattern, distinct",
    [
        ([1, 2, 3, 4], [0, 1, 2, 3]),  # all different: nothing saved
        ([1, 2, 2, 2], [0, 1]),  # one hit and three padding slots
        ([5, 5, 5, 5], [0]),  # no templates: four identical dummies
        ([1, 2, 1, 2], [0, 1]),  # interleaved classes
    ],
)
def test_dedup_is_bit_identical_and_evaluates_each_class_once(
    embedder, pattern, distinct
):
    stock, stock_calls = _forward(embedder, _features(pattern), dedup=False)
    dedup, dedup_calls = _forward(embedder, _features(pattern), dedup=True)
    assert stock_calls == [0, 1, 2, 3]
    assert dedup_calls == distinct
    assert torch.equal(stock, dedup)


def test_default_is_off(embedder):
    assert TemplateEmbedder(n_blocks=1, c=8, c_z=C_Z).dedup_identical_slots is False


def test_grad_mode_runs_the_stock_loop(embedder):
    embedder.dedup_identical_slots = True
    feats = _features([5, 5, 5, 5])
    calls = []
    original = embedder.single_template_forward

    def counted(*args, **kwargs):
        calls.append(kwargs["template_id"])
        return original(*args, **kwargs)

    with mock.patch.object(embedder, "single_template_forward", counted):
        embedder(feats, _z())  # grad enabled
    assert calls == [0, 1, 2, 3]


def test_unexpected_slot_shape_runs_the_stock_loop(embedder):
    feats = _features([5, 5, 5, 5])
    extra = torch.randn(1, N, N, 3, generator=torch.Generator().manual_seed(4))
    # a leading dimension other than the slot count: the stock loop still reads
    # slots 0-3, but the class decision refuses it
    feats["template_unit_vector"] = torch.cat([feats["template_unit_vector"], extra])
    assert embedder._template_slot_classes(feats, T) is None
    stock, _ = _forward(embedder, feats, dedup=False)
    dedup, calls = _forward(embedder, feats, dedup=True)
    assert calls == [0, 1, 2, 3] and torch.equal(stock, dedup)


def test_slot_classes_are_memoised_per_item_and_invalidated(embedder):
    feats = _features([1, 2, 2, 2])
    reads = []
    original_tolist = torch.Tensor.tolist

    def counted(self):
        reads.append(1)
        return original_tolist(self)

    with mock.patch.object(torch.Tensor, "tolist", counted):
        assert embedder._template_slot_classes(feats, T) == [0, 1, 1, 1]
        assert embedder._template_slot_classes(feats, T) == [0, 1, 1, 1]
        assert len(reads) == 1  # one host read per item, reused across recycles
        with torch.no_grad():
            feats["template_distogram"][3].add_(1.0)  # in-place change
        assert embedder._template_slot_classes(feats, T) == [0, 1, 1, 3]
        assert len(reads) == 2
        feats["template_aatype"] = feats["template_aatype"].clone()  # new tensor
        embedder._template_slot_classes(feats, T)
        assert len(reads) == 3
