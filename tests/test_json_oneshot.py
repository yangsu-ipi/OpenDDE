# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""json_oneshot: one json.dumps + one write gives the bytes of streaming json.dump."""

import numpy as np
import pytest
import torch

from opendde.utils.file_io import save_json, save_json_oneshot
from runner.dumper import DataDumper


def _documents():
    g = torch.Generator().manual_seed(0)
    yield {  # summary-confidence shaped
        "plddt": torch.tensor(71.25),
        "ptm": torch.tensor(0.5123456789),
        "chain_pair_iptm": torch.rand(3, 3, generator=g),
        "has_clash": torch.tensor(False),
        "num_recycles": 10,
        "name": "Zn²⁺ complex é",
        "missing": None,
    }
    yield {  # full-data shaped: token x token maps and per-atom arrays
        "token_pair_pae": torch.rand(17, 17, generator=g) * 31.75,
        "contact_probs": torch.rand(17, 17, generator=g).to(torch.float16),
        "atom_plddt": np.linspace(0, 100, 41, dtype=np.float32),
        "token_chain_ids": ["A"] * 9 + ["B"] * 8,
    }
    yield {  # non-finite and extreme floats
        "values": [float("nan"), float("inf"), -float("inf"), 1e-320, 1.7e308, -0.0],
        "array": np.array([np.nan, 1.5, -2.25]),
        "nested": {"a": [[1, 2], [3.0, 4.5]], "b": {"c": True}},
    }


@pytest.mark.parametrize("indent", [4, None])
def test_oneshot_bytes_match_the_streaming_writer(tmp_path, indent):
    for i, doc in enumerate(_documents()):
        a, b = tmp_path / f"stock{i}.json", tmp_path / f"oneshot{i}.json"
        save_json(doc, a, indent=indent)
        save_json_oneshot(doc, b, indent=indent)
        assert a.read_bytes() == b.read_bytes(), (i, indent)


def test_oneshot_leaves_the_input_unchanged(tmp_path):
    doc = {"pae": torch.ones(2, 2)}
    save_json_oneshot(doc, tmp_path / "x.json")
    assert torch.is_tensor(doc["pae"])


def test_dumper_selects_the_writer(monkeypatch):
    assert DataDumper("out").json_oneshot is False
    assert DataDumper("out", json_oneshot=True).json_oneshot is True
