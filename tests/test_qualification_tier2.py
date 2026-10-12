"""Tests for the tier 2 gate evaluation (scripts/qualification/tier2_gates.py)."""

import copy
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "qualification"))
import tier2_gates as t2  # noqa: E402

MANIFEST = json.loads((ROOT / "scripts/qualification/inputs/manifest.json").read_text())
ROWS = {r["id"]: r for r in MANIFEST["rows"]}


def _jobs(row_id):
    return json.loads((ROOT / ROWS[row_id]["input"]).read_text())


def _facts(row_id, job_name=None):
    jobs = _jobs(row_id)
    job = (
        jobs[0] if job_name is None else next(j for j in jobs if j["name"] == job_name)
    )
    return t2.job_facts(ROWS[row_id], job)


@pytest.fixture(scope="module")
def limits():
    return json.loads(
        (ROOT / "scripts/qualification/inputs/tier2_limits.json").read_text()
    )


def _guidance(rule):
    kind = (rule or {}).get("kind")
    if kind == "contacts":
        return {"a": {"contacts_ok": True}, "b": {"contacts_ok": True}}
    if kind == "epitope":
        return {"a": {"epitope_ok": True}, "b": {"epitope_ok": True}}
    return {"a": {"contact_mean": 6.0}, "b": {"contact_mean": 6.0}}


def _entry(row_id, job, facts, limits, seeds=(101, 102), per_seed=5):
    """A stock-like measurement that every declared gate should pass."""
    types = ["polymer"] + facts["types"]
    pair = {
        "anchor_rmsd": 0.1,
        "lddt": {k: {"score": 0.999} for k in types},
        "atom_error": {"p95": 0.1},
        "ligands": [
            {"gated": True, "rmsd": 0.3, "pocket_jaccard": 0.95, "truncated": False}
            for _ in range(facts["gated_ligands"])
        ],
        "confidence_diff": {"ranking_score": 0.0, "plddt": 0.0, "iptm": 0.0},
    }
    if facts["multichain"]:
        pair["interface"] = {"fnat": 0.99, "fnonnat": 0.01}
    if facts["guided"]:
        pair["guidance"] = _guidance(t2._limit(limits, row_id, job, "guidance"))
    n = len(seeds) * per_seed
    ref = {
        "status": "ok",
        "lddt": {k: 0.9 for k in types},
        "fnat": 0.8,
        "fnonnat": 0.2,
        "fnat_min_pair": 0.7,
    }
    entry = {
        "id": row_id,
        "job": job,
        "paired_status": ["ok"] * len(seeds),
        "pairs": [copy.deepcopy(pair) for _ in range(n)],
        "samples_off": {str(s): list(range(per_seed)) for s in seeds},
        "samples_cand": {str(s): list(range(per_seed)) for s in seeds},
        "spread_off": 3.5,
        "spread_cand": 3.5,
        "spread_same_seed_off": 3.0,
        "spread_same_seed_cand": 3.0,
        "min_pair_off": 1.3,
        "min_pair_cand": 1.3,
        "same_seed_pairs_off": [1.3 + 0.1 * i for i in range(10 * len(seeds))],
        "same_seed_pairs_cand": [1.3 + 0.1 * i for i in range(10 * len(seeds))],
        "plddt_median_off": 92.8,
        "plddt_median_cand": 92.8,
    }
    if facts["multichain"]:
        entry["contact_freq_l1"] = 0.01
    if facts["reference"]:
        entry["ref_off"] = [copy.deepcopy(ref) for _ in range(n)]
        entry["ref_cand"] = [copy.deepcopy(ref) for _ in range(n)]
    if facts["zn"]:
        entry["zn"] = _zn(*[(s, 15, 15) for s in seeds])
    return entry


def _zn(*per_seed):
    """A Zn summary from (seed, eligible ions, same-site ions) per seed pair."""
    by_seed = [
        {"seeds": [s, s], "status": "ok", "same_site_of": n, "same_site": k}
        for s, n, k in per_seed
    ]
    return {
        "same_site_of": sum(b["same_site_of"] for b in by_seed),
        "same_site": sum(b["same_site"] for b in by_seed),
        "by_seed": by_seed,
    }


def _failed(gates):
    return {k: v for k, v in gates.items() if v["status"] == "fail"}


def test_every_manifest_job_has_complete_limits(limits):
    """A stock-like entry passes on every development and held-out job: each
    has a class and every limit its facts require."""
    for row_id, row in ROWS.items():
        for job in _jobs(row_id):
            facts = t2.job_facts(row, job)
            entry = _entry(row_id, job["name"], facts, limits)
            gates = t2.evaluate_entry(entry, limits, facts)
            assert not _failed(gates), (row_id, job["name"], _failed(gates))


def test_held_out_facts_drive_required_gates(limits):
    h2 = t2.evaluate_entry(
        _entry("H2", "7pzb", _facts("H2"), limits), limits, _facts("H2")
    )
    for name in ("lddt_dna", "ligand_rmsd", "ensemble_ref_lddt_dna_median"):
        assert h2[name]["status"] == "pass"
    h3 = t2.evaluate_entry(
        _entry("H3", "9sat_unconstrained", _facts("H3"), limits), limits, _facts("H3")
    )
    assert h3["zn_same_site"]["status"] == "pass"
    assert h3["zn_claim"]["value"] == "qualified"
    h8 = t2.evaluate_entry(
        _entry("H8", "h8_dna_duplex", _facts("H8"), limits), limits, _facts("H8")
    )
    assert h8["pair_interface_fnonnat"]["status"] == "pass"
    assert "ensemble_ref_lddt_polymer_median" not in h8


def test_missing_class_mapping_fails(limits):
    facts = _facts("D4")
    entry = _entry("D99", "7wux", facts, limits)
    gates = t2.evaluate_entry(entry, limits, facts)
    assert gates["limits_declared"]["status"] == "fail"
    assert list(_failed(gates)) == ["limits_declared"]


def test_missing_sample_fails(limits):
    facts = _facts("D4")
    entry = _entry("D4", "7wux", facts, limits)
    entry["samples_cand"]["102"] = [0, 1, 2, 3]
    entry["pairs"] = entry["pairs"][:9]
    entry["paired_status"] = ["ok", "fail: sample sets differ"]
    failed = _failed(t2.evaluate_entry(entry, limits, facts))
    assert {"sample_count_cand", "paired_conclusive"} <= set(failed)


def test_extra_seed_fails(limits):
    facts = _facts("D2")
    entry = _entry("D2", "7r6r", facts, limits, seeds=(101, 102, 103))
    failed = _failed(t2.evaluate_entry(entry, limits, facts))
    assert {"sample_count_off", "sample_count_cand", "paired_conclusive"} <= set(failed)


def test_missing_reference_score_fails(limits):
    facts = _facts("D4")
    entry = _entry("D4", "7wux", facts, limits)
    entry["ref_cand"][3] = {"status": "fail: no reference chain match"}
    gates = t2.evaluate_entry(entry, limits, facts)
    assert gates["reference_complete_cand"]["status"] == "fail"
    entry = _entry("D4", "7wux", facts, limits)
    entry["ref_cand"] = entry["ref_cand"][:9]
    gates = t2.evaluate_entry(entry, limits, facts)
    assert gates["reference_complete_cand"]["status"] == "fail"
    entry = _entry("D2", "7r6r", _facts("D2"), limits)
    del entry["ref_off"][0]["lddt"]["dna"]
    gates = t2.evaluate_entry(entry, limits, _facts("D2"))
    assert gates["reference_complete_off"]["status"] == "fail"


def test_missing_reference_fnat_fails_but_absent_interface_is_declared(limits):
    facts = _facts("D2")
    entry = _entry("D2", "7r6r", facts, limits)
    entry["ref_cand"][0]["fnat"] = None
    gates = t2.evaluate_entry(entry, limits, facts)
    assert gates["ensemble_ref_fnat_median"]["status"] == "fail"
    entry = _entry("D2", "7r6r", facts, limits)
    for s in entry["ref_off"] + entry["ref_cand"]:
        s["fnat_min_pair"] = None
    gates = t2.evaluate_entry(entry, limits, facts)
    assert gates["ensemble_ref_fnat_min_pair_median"]["status"] == "n/a"


@pytest.mark.parametrize(
    "row_id, job, mutate, gate",
    [
        (
            "D4",
            "7wux",
            lambda e: e.pop("contact_freq_l1"),
            "ensemble_contact_frequency",
        ),
        ("D4", "7wux", lambda e: e.update(spread_off=0.0), "ensemble_spread_ratio"),
        ("D4", "7wux", lambda e: e.update(min_pair_cand=None), "no_duplicate_samples"),
        (
            "D4",
            "7wux",
            lambda e: e.update(plddt_median_cand=None),
            "ensemble_plddt_median",
        ),
        (
            "D4",
            "7wux",
            lambda e: e["pairs"][2]["confidence_diff"].pop("ranking_score"),
            "ranking_score",
        ),
        (
            "D4",
            "7wux",
            lambda e: e["pairs"][2]["confidence_diff"].pop("plddt"),
            "mean_plddt",
        ),
        ("D4", "7wux", lambda e: e["pairs"][4]["ligands"].pop(), "ligand_rmsd"),
        ("D4", "7wux", lambda e: e["pairs"][4].pop("interface"), "pair_interface_fnat"),
        ("D2", "7r6r", lambda e: e["pairs"][1]["lddt"].pop("dna"), "lddt_dna"),
        ("D9", "1a14_contact", lambda e: e["pairs"][7].pop("guidance"), "guidance"),
        ("D14", "6e4y", lambda e: e.pop("zn"), "zn_same_site"),
        (
            "D4",
            "7wux",
            lambda e: e["pairs"][0].update(anchor_rmsd=float("nan")),
            "anchor_rmsd",
        ),
    ],
)
def test_absent_metrics_fail(limits, row_id, job, mutate, gate):
    facts = _facts(row_id, job)
    entry = _entry(row_id, job, facts, limits)
    mutate(entry)
    gates = t2.evaluate_entry(entry, limits, facts)
    assert gates[gate]["status"] == "fail", gates[gate]


def test_undeclared_limit_fails(limits):
    facts = _facts("D4")
    trimmed = copy.deepcopy(limits)
    del trimmed["classes"]["stable"]["pair_fnonnat_max"]
    gates = t2.evaluate_entry(_entry("D4", "7wux", facts, limits), trimmed, facts)
    assert gates["pair_interface_fnonnat"]["status"] == "fail"
    assert "not declared" in gates["pair_interface_fnonnat"]["reason"]


def test_guidance_rule_on_unguided_row_fails(limits):
    facts = _facts("D2")
    extra = copy.deepcopy(limits)
    extra["rows"]["D2"]["guidance"] = {"kind": "contacts", "min": 9}
    gates = t2.evaluate_entry(_entry("D2", "7r6r", facts, limits), extra, facts)
    assert gates["guidance"]["status"] == "fail"


def test_zn_gate(limits):
    facts = _facts("D14")
    entry = _entry("D14", "6e4y", facts, limits)
    entry["zn"] = {"same_site": 5, "same_site_of": 8}
    gates = t2.evaluate_entry(entry, limits, facts)
    assert gates["zn_same_site"]["status"] == "fail"
    assert gates["zn_claim"]["value"] == "fail"
    entry["zn"] = {"same_site": 29, "same_site_of": 39}
    gates = t2.evaluate_entry(entry, limits, facts)
    assert gates["zn_same_site"]["status"] == "fail"
    entry["zn"] = _zn((101, 20, 19), (102, 19, 19), (201, 5, 5))
    gates = t2.evaluate_entry(entry, limits, facts)
    assert "extra Zn seeds" in gates["zn_same_site"]["reason"]


def test_zn_row_without_rule_fails(limits):
    facts = _facts("D14")
    extra = copy.deepcopy(limits)
    del extra["rows"]["D14"]["zn_rule"]
    gates = t2.evaluate_entry(_entry("D14", "6e4y", facts, limits), extra, facts)
    assert gates["zn_same_site"]["status"] == "fail"


def _h3(limits, zn):
    facts = _facts("H3")
    entry = _entry("H3", "9sat_unconstrained", facts, limits)
    entry["zn"] = zn
    return t2.evaluate_entry(entry, limits, facts)


def test_h3_floor_without_extra_seeds_is_incomplete(limits):
    """Two same-outcome pairs with three Zn give at most six eligible ions."""
    gates = _h3(limits, _zn((101, 6, 6), (102, 0, 0)))
    assert gates["zn_same_site"]["status"] == "fail"
    assert "run the next extra seed" in gates["zn_same_site"]["reason"]
    assert gates["zn_claim"]["value"] == "unresolved"


def test_h3_extra_seeds_reach_the_minimum(limits):
    gates = _h3(limits, _zn((101, 6, 6), (102, 0, 0), (201, 3, 3), (202, 3, 3)))
    assert gates["zn_same_site"]["status"] == "pass"
    assert gates["zn_claim"]["value"] == "qualified"
    gates = _h3(limits, _zn((101, 6, 6), (102, 0, 0), (201, 6, 3)))
    assert gates["zn_same_site"]["status"] == "fail"
    assert gates["zn_claim"]["value"] == "fail"


def test_h3_unresolved_at_the_cap_does_not_qualify_zn(limits):
    zn = _zn((101, 3, 3), (102, 0, 0), *[(201 + i, 1, 1) for i in range(4)])
    gates = _h3(limits, zn)
    assert gates["zn_same_site"]["status"] == "n/a"
    assert gates["zn_claim"]["value"] == "unresolved"
    assert not _failed(gates)
    result = t2._result("H3", "9sat_unconstrained", gates)
    assert result["pass"] and t2.zn_claim([result]) == "unresolved"


@pytest.mark.parametrize(
    "zn, reason",
    [
        (_zn((101, 6, 6), (102, 6, 6), (201, 3, 3)), "run after 12"),
        (
            _zn((101, 3, 3), (102, 0, 0), *[(201 + i, 1, 1) for i in range(5)]),
            "more than 4",
        ),
        (_zn((101, 6, 6), (102, 0, 0), (101, 6, 6)), "new same-seed"),
    ],
)
def test_h3_extra_seed_rule_violations_fail(limits, zn, reason):
    gates = _h3(limits, zn)
    assert reason in gates["zn_same_site"]["reason"]
    assert gates["zn_claim"]["value"] == "fail"


def test_h5_zn_is_report_only(limits):
    facts = _facts("H5")
    entry = _entry("H5", "9sat_pocket", facts, limits)
    for p in entry["pairs"]:
        p["anchor_rmsd"] = 12.0
    entry["zn"] = _zn((101, 0, 0), (102, 0, 0))
    gates = t2.evaluate_entry(entry, limits, facts)
    assert gates["zn_same_site"]["status"] == "report"
    assert gates["zn_claim"]["value"] == "report-only"
    assert not _failed(gates)
    result = t2._result("H5", "9sat_pocket", gates)
    assert t2.zn_claim([result]) == "none"


def test_zn_claim_aggregate():
    def r(claim, failed=()):
        return {
            "failed": list(failed),
            "gates": {"zn_claim": {"status": "report", "value": claim}},
        }

    assert t2.zn_claim([r("qualified"), r("report-only")]) == "qualified"
    assert t2.zn_claim([r("qualified"), r("unresolved")]) == "unresolved"
    assert t2.zn_claim([r("unresolved"), r("fail")]) == "fail"
    assert t2.zn_claim([r("qualified", ["ensemble_near_duplicates"])]) == "fail"
    assert t2.zn_claim([r("unresolved", ["zn_same_site"])]) == "unresolved"


def test_no_same_outcome_pairs_is_declared_not_applicable(limits):
    facts = _facts("D10")
    entry = _entry("D10", "1a14_pocket", facts, limits)
    for p in entry["pairs"]:
        p["anchor_rmsd"] = 12.0
    gates = t2.evaluate_entry(entry, limits, facts)
    assert gates["outcome_count"]["status"] == "pass"
    assert gates["lddt_polymer"]["status"] == "n/a"
    assert gates["guidance_epitope"]["status"] == "pass"
    assert gates["ensemble_ref_lddt_polymer_median"]["status"] == "pass"


def test_quality_gates_are_one_sided(limits):
    facts = _facts("D4")
    entry = _entry("D4", "7wux", facts, limits)
    for s in entry["ref_cand"]:
        s["lddt"] = {k: 0.95 for k in s["lddt"]}
        s["fnonnat"] = 0.1
    assert not _failed(t2.evaluate_entry(entry, limits, facts))
    for s in entry["ref_cand"]:
        s["lddt"] = {k: 0.85 for k in s["lddt"]}
    gates = t2.evaluate_entry(entry, limits, facts)
    assert gates["ensemble_ref_lddt_polymer_median"]["status"] == "fail"
    assert gates["ensemble_ref_lddt_protein_median"]["status"] == "fail"


def test_duplicate_and_outcome_and_p95(limits):
    facts = _facts("D4")
    entry = _entry("D4", "7wux", facts, limits, per_seed=5)
    entry["min_pair_cand"] = 0.0
    entry["pairs"][0]["anchor_rmsd"] = 12.0
    entry["pairs"][1]["anchor_rmsd"] = 12.0
    entry["pairs"][2]["atom_error"]["p95"] = 1.4
    failed = _failed(t2.evaluate_entry(entry, limits, facts))
    assert {"no_duplicate_samples", "outcome_count", "p95_error"} <= set(failed)
    d2 = t2.evaluate_entry(
        _entry("D2", "7r6r", _facts("D2"), limits), limits, _facts("D2")
    )
    assert d2["p95_error"]["status"] == "report"


def test_unmeasured_job_fails(limits):
    facts = _facts("D13", "7r6r")
    data = {"entries": [_entry("D13", "7r6r", facts, limits)]}
    results = t2.evaluate(data, limits, MANIFEST)
    by_job = {r["job"]: r for r in results}
    assert by_job["7r6r"]["pass"]
    assert not by_job["1a14_unconstrained"]["pass"]
    assert by_job["1a14_unconstrained"]["failed"] == ["measured"]
    results = t2.evaluate(data, limits, MANIFEST, rows=["D13", "D2"])
    assert not [r for r in results if r["id"] == "D2"][0]["pass"]


def test_partial_collapse_fails_near_duplicate_gate(limits):
    facts = _facts("D3")
    entry = _entry("D3", "1a14_unconstrained", facts, limits)
    # three near-copies of sample 0 in each seed: six close pairs per seed
    entry["same_seed_pairs_cand"] = [0.1] * 12 + [3.0] * 8
    entry["min_pair_cand"] = 0.1
    gates = t2.evaluate_entry(entry, limits, facts)
    assert gates["ensemble_near_duplicates"]["status"] == "fail"
    assert gates["ensemble_near_duplicates"]["value"]["pairs_below"] == 12
    entry["same_seed_pairs_cand"] = [0.1] + [3.0] * 19
    gates = t2.evaluate_entry(entry, limits, facts)
    assert gates["ensemble_near_duplicates"]["status"] == "pass"
    entry.pop("same_seed_pairs_off")
    gates = t2.evaluate_entry(entry, limits, facts)
    assert gates["ensemble_near_duplicates"]["status"] == "fail"
