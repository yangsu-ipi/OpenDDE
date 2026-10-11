"""Tests for the tier 2 limit provenance audit (scripts/qualification/limit_audit.py)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "qualification"))
import limit_audit as la  # noqa: E402

LIMITS = {"rows": {"D1": {"class": "stable"}, "D2": {"class": "flexible"}}}


def _result(row_id, gates):
    return {"id": row_id, "job": row_id.lower(), "gates": gates}


def _gate(value, limit, status="pass"):
    return {"status": status, "value": value, "limit": limit}


@pytest.mark.parametrize(
    "x, expected",
    [(0.0142, 0.015), (0.556, 0.6), (0.79, 0.8), (0.0061, 0.008), (0.3, 0.3)],
)
def test_round_up(x, expected):
    assert la.round_up(x) == pytest.approx(expected)


def test_badness_scales():
    assert la.badness("lddt_polymer", _gate(0.97, 0.95)) == pytest.approx((0.03, 0.05))
    assert la.badness("outcome_count", _gate(8, 2)) == (2, 8)
    assert la.badness("ensemble_ref_fnat_median", _gate(-0.01, 0.05)) == (0.0, 0.05)
    lo, hi = la.badness("ensemble_spread_ratio", _gate(2.0, [0.1, 10]))
    assert hi == pytest.approx(2.302585) and lo == pytest.approx(0.693147)
    assert la.badness("lddt_polymer", {"status": "n/a"}) is None


def test_per_type_lddt_pooled_with_polymer():
    assert (
        la.family("ensemble_ref_lddt_dna_lowest") == "ensemble_ref_lddt_polymer_lowest"
    )
    assert la.family("lddt_rna") == "lddt_polymer"
    rows = la.audit(
        [_result("D1", {"lddt_dna": _gate(0.99, 0.95)})],
        [_result("D1", {"lddt_polymer": _gate(0.98, 0.95)})],
        LIMITS,
    )
    assert len(rows) == 1 and rows[0]["members"] == ["lddt_dna", "lddt_polymer"]


def test_precision_binding_needs_no_action():
    rows = la.audit(
        [_result("D1", {"pair_interface_fnonnat": _gate(0.07, 0.15)})],
        [_result("D1", {"pair_interface_fnonnat": _gate(0.05, 0.15)})],
        LIMITS,
    )
    assert rows[0]["binding"] == "precision" and rows[0]["action"] == "none"


def test_kernel_binding_within_doubled_precision_needs_no_action():
    rows = la.audit(
        [_result("D2", {"ensemble_contact_frequency": _gate(0.6, 1.0)})],
        [_result("D2", {"ensemble_contact_frequency": _gate(0.7, 1.0)})],
        LIMITS,
    )
    assert rows[0]["binding"] == "kernel" and rows[0]["action"] == "none"


def test_kernel_set_limit_is_rederived_and_retested():
    passes = la.audit(
        [_result("D2", {"pair_interface_fnat": _gate(0.85, 0.6)})],
        [_result("D2", {"pair_interface_fnat": _gate(0.72, 0.6)})],
        LIMITS,
    )[0]
    assert passes["action"] == "rederive"
    assert passes["precision_only_limit"] == pytest.approx(0.3)
    assert passes["kernel_passes_rederived"]
    fails = la.audit(
        [_result("D2", {"ensemble_ref_fnonnat_median": _gate(0.007, 0.06)})],
        [_result("D2", {"ensemble_ref_fnonnat_median": _gate(0.028, 0.06)})],
        LIMITS,
    )[0]
    assert fails["action"] == "rederive" and not fails["kernel_passes_rederived"]


def test_rules_and_degenerate_groups():
    rows = la.audit(
        [_result("D1", {"outcome_count": _gate(10, 9), "lddt_rna": _gate(1.0, 0.95)})],
        [_result("D1", {"outcome_count": _gate(9, 9), "lddt_rna": _gate(0.99, 0.95)})],
        LIMITS,
    )
    by_gate = {r["gate"]: r for r in rows}
    assert by_gate["outcome_count"]["action"] == "none (rule)"
    assert by_gate["lddt_polymer"]["action"].startswith("degenerate")
