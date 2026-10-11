"""Tests for the tier 2 gate evaluation (scripts/qualification/tier2_gates.py)."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "qualification"))
import tier2_gates as t2  # noqa: E402


def _pair(anchor, lddt=0.99, p95=0.2, fnat=0.95, rs=0.001):
    return {
        "anchor_rmsd": anchor,
        "lddt": {"polymer": {"score": lddt}, "protein": {"score": lddt}},
        "atom_error": {"p95": p95},
        "interface": {"fnat": fnat},
        "ligands": [],
        "confidence_diff": {"ranking_score": rs, "plddt": 0.1},
    }


def _entry(**overrides):
    entry = {
        "id": "D4",
        "job": "7wux",
        "paired_status": ["ok", "ok"],
        "pairs": [_pair(0.1) for _ in range(10)],
        "spread_off": 3.5,
        "spread_cand": 3.5,
        "min_pair_off": 1.3,
        "min_pair_cand": 1.3,
        "contact_freq_l1": 0.01,
        "plddt_median_off": 92.8,
        "plddt_median_cand": 92.8,
        "ref_off": {"lddt": [0.92] * 10, "fnat": [0.9] * 10},
        "ref_cand": {"lddt": [0.92] * 10, "fnat": [0.9] * 10},
    }
    entry.update(overrides)
    return entry


@pytest.fixture(scope="module")
def limits():
    return json.loads(
        (ROOT / "scripts/qualification/inputs/tier2_limits.json").read_text()
    )


def test_stock_like_entry_passes(limits):
    gates = t2.evaluate_entry(_entry(), limits)
    assert all(g["pass"] for g in gates.values()), gates


def test_duplicate_samples_fail(limits):
    gates = t2.evaluate_entry(_entry(min_pair_cand=0.0), limits)
    assert not gates["no_duplicate_samples"]["pass"]


def test_switched_pairs_fail_stable_count(limits):
    pairs = [_pair(0.1)] * 8 + [_pair(12.0)] * 2
    gates = t2.evaluate_entry(_entry(pairs=pairs), limits)
    assert not gates["outcome_count"]["pass"]


def test_reference_drop_is_one_sided(limits):
    better = _entry(ref_cand={"lddt": [0.95] * 10, "fnat": [0.95] * 10})
    assert all(g["pass"] for g in t2.evaluate_entry(better, limits).values())
    worse = _entry(ref_cand={"lddt": [0.90] * 10, "fnat": [0.9] * 10})
    assert not t2.evaluate_entry(worse, limits)["ensemble_ref_lddt_median"]["pass"]


def test_per_row_override_and_p95_gate(limits):
    gates = t2.evaluate_entry(_entry(pairs=[_pair(0.1, p95=1.4)] * 10), limits)
    assert gates["p95_error"]["limit"] == 1.0 and not gates["p95_error"]["pass"]
    d2 = t2.evaluate_entry(
        _entry(id="D2", job="7r6r", pairs=[_pair(0.1, p95=1.4)] * 10), limits
    )
    assert "p95_error" not in d2
