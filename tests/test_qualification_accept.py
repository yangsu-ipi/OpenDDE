"""Tests for full candidate acceptance (scripts/qualification/accept.py).

Each test builds a complete synthetic evidence set for every sm_120 row, then
breaks one thing that must stop a full acceptance."""

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "qualification"))
import accept  # noqa: E402
import tier2_gates as t2  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "tier2_fixtures", ROOT / "tests" / "test_qualification_tier2.py"
)
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)

INPUTS = ROOT / "scripts" / "qualification" / "inputs"
COMMIT = "c" * 40
VERSIONS = {"torch": "2.7.1+cu128"}
H3_JOB = "9sat_unconstrained"


def _trace(run: Path, row: str, seeds, salt: str = "") -> None:
    meta = {
        "cli_args": ["pred", "-o", str(run / row), "--seeds", ",".join(map(str, seeds))]
        + ["--deterministic", "true"],
        "exit_code": 0,
        "fingerprint": {
            "commit": COMMIT,
            "dirty_paths": [],
            "gpu": {"capability": [12, 0]},
            "versions": VERSIONS,
        },
        "rollouts": [{"seed": s} for s in seeds],
    }
    records = [
        {
            "sample_name": row,
            "seed": s,
            "rollout": 0,
            "chunk": 0,
            "step": None,
            "site": "sampler.initial_noise",
            "kind": "gaussian",
            "outer": None,
            "sample": 0,
            "ordinal": 0,
            "dtype": "float32",
            "shape": [3],
            "raw_sha256": f"{row}-{s}{salt}",
            "cmp_sha256": f"{row}-{s}",
        }
        for s in seeds
    ]
    run.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(meta)] + [json.dumps(r) for r in records]
    (run / f"{row}.trace.jsonl").write_text("\n".join(lines) + "\n")


def _artifacts(run: Path, row: str, jobs, seeds, content: str = "x") -> None:
    for job in jobs:
        for s in seeds:
            pred = run / row / job / f"seed_{s}" / "predictions"
            pred.mkdir(parents=True, exist_ok=True)
            (pred / f"{job}_sample_0.cif").write_text(content)


def _tier1(row, jobs, seeds):
    tensor = {"rel_fro": 0.001, "tail_over_rms": 0.1, "finite": True}
    head = {f"{h}_max_abs_prob": 0.0 for h in ("plddt", "pae", "pde", "resolved")}
    head |= {f"{h}_finite": True for h in ("plddt", "pae", "pde", "resolved")}
    head |= {"ptm_max_abs": 0.0, "iptm_max_abs": 0.0, "ranking_core_max_abs": 0.0}
    return [
        {
            "id": row,
            "job": job,
            "seed": s,
            "status": "ok",
            "s_inputs": dict(tensor),
            "s_trunk": dict(tensor),
            "z_trunk": dict(tensor),
            "distogram": {"argmax_changed_margin_0.1": 0.0, "finite": True},
            "head_source": "head_fixed",
            "head": dict(head),
        }
        for job in jobs
        for s in seeds
    ]


class World:
    """A passing evidence set and the inputs directory accept.py reads."""

    def __init__(self, tmp: Path, h3_extra=(), h3_zn=None):
        self.tmp = tmp
        self.inputs = tmp / "inputs"
        shutil.copytree(INPUTS, self.inputs, ignore=shutil.ignore_patterns("msa"))
        self.manifest = json.loads((INPUTS / "manifest.json").read_text())
        self.table = {r["id"]: r for r in self.manifest["rows"]}
        self.limits = json.loads((INPUTS / "tier2_limits.json").read_text())
        held_out = [f"H{i}" for i in range(1, 11)]
        self.seeds = {f"D{i}": [101, 102] for i in range(1, 16)}
        for n, row in enumerate(held_out):
            self.seeds[row] = [201 + 2 * n, 202 + 2 * n]
        self.h3_extra = list(h3_extra)
        self.log = {
            "draws": [
                {
                    "id": "sm_120-1",
                    "gpu": "sm_120",
                    "candidate": "fast-test",
                    "seeds": {r: self.seeds[r] for r in held_out},
                    "zn_extra": {"H3": self.h3_extra} if self.h3_extra else {},
                }
            ]
        }
        self.sets = []
        for name, rows in (
            ("development", [f"D{i}" for i in range(1, 16)]),
            ("held-out", held_out),
        ):
            self.sets.append(self._build_set(name, rows, h3_zn))
        if self.h3_extra:
            self._build_zn_set()
        self.write()

    def jobs(self, row):
        return [
            j["name"] for j in json.loads((ROOT / self.table[row]["input"]).read_text())
        ]

    def _build_set(self, name, rows, h3_zn):
        base = self.tmp / name
        runs = {k: base / k for k in ("off", "cand", "repeat", "baseline")}
        tier1, entries = [], []
        for row in rows:
            seeds, jobs = self.seeds[row], self.jobs(row)
            for key in ("off", "cand"):
                _trace(runs[key], row, seeds)
            for key in runs:
                _artifacts(runs[key], row, jobs, seeds)
            tier1 += _tier1(row, jobs, seeds)
            for job in json.loads((ROOT / self.table[row]["input"]).read_text()):
                facts = t2.job_facts(self.table[row], job)
                e = fx._entry(row, job["name"], facts, self.limits, seeds=tuple(seeds))
                if row == "H3" and h3_zn is not None:
                    e["zn"] = h3_zn
                entries.append(e)
        (base / "tier1.json").write_text(json.dumps({"entries": tier1}))
        measure = {
            "off": str(runs["off"]),
            "cand": str(runs["cand"]),
            "entries": entries,
        }
        (base / "tier2.json").write_text(json.dumps(measure))
        s = {
            "name": name,
            "rows": rows,
            "off": str(runs["off"]),
            "cand": str(runs["cand"]),
            "cand_repeat": str(runs["repeat"]),
            "tier1": str(base / "tier1.json"),
            "tier2": str(base / "tier2.json"),
        }
        if name == "development":
            s["off_baseline"] = str(runs["baseline"])
        return s

    def _build_zn_set(self):
        base = self.tmp / "zn-extra"
        held = self.tmp / "held-out"
        for key in ("off", "cand"):
            _trace(base / key, "H3", self.h3_extra)
            _artifacts(base / key, "H3", [H3_JOB], self.h3_extra)
            for s in self.h3_extra:
                link = held / key / "H3" / H3_JOB / f"seed_{s}"
                link.symlink_to(base / key / "H3" / H3_JOB / f"seed_{s}")
        self.sets.append(
            {
                "name": "zn-extra",
                "zn_only": True,
                "rows": ["H3"],
                "off": str(base / "off"),
                "cand": str(base / "cand"),
            }
        )

    def write(self):
        (self.inputs / "held_out_log.json").write_text(json.dumps(self.log))

    def evidence(self):
        return {
            "gpu": "sm_120",
            "route": "port-chosen",
            "candidate": "fast-test",
            "commit": COMMIT,
            "held_out_draw": "sm_120-1",
            "sets": copy.deepcopy(self.sets),
        }

    def freeze(self):
        path = self.inputs / "acceptance.json"
        acc = json.loads(path.read_text())
        acc["gpus"]["sm_120"]["frozen"] = {
            "tier1_limits_sha256": accept.sha256(self.inputs / "tier1_limits.json"),
            "tier2_limits_sha256": accept.sha256(self.inputs / "tier2_limits.json"),
        }
        path.write_text(json.dumps(acc))

    def edit_tier2(self, set_index, fn):
        path = Path(self.sets[set_index]["tier2"])
        data = json.loads(path.read_text())
        fn(data)
        path.write_text(json.dumps(data))

    def run(self, evidence=None):
        return accept.accept(evidence or self.evidence(), self.inputs)


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)
    stored = {
        "digests": json.loads((INPUTS / "digests.json").read_text()),
        "external": json.loads((INPUTS / "external_digests.json").read_text()),
    }
    monkeypatch.setattr(accept.freeze_inputs, "digests", lambda m: stored["digests"])
    monkeypatch.setattr(
        accept.freeze_inputs, "external_digests", lambda m: stored["external"]
    )


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _failed(verdict, fragment):
    return [c for c in verdict["failed"] if fragment in c]


def test_complete_pass_is_provisional_until_frozen_then_full(world):
    verdict = world.run()
    assert verdict["failed"] == []
    assert verdict["exit_code"] == accept.PROVISIONAL
    evaluated = {
        c["check"].split()[0]
        for c in verdict["checks"]
        if c["check"].endswith(" tier 2")
    }
    required = json.loads((INPUTS / "acceptance.json").read_text())
    assert evaluated == set(required["gpus"]["sm_120"]["routes"]["port-chosen"]["rows"])
    assert {
        c["check"].split()[0]
        for c in verdict["checks"]
        if "random inputs" in c["check"]
    } == evaluated
    world.freeze()
    verdict = world.run()
    assert verdict["exit_code"] == accept.FULL and verdict["zn_claim"] == "qualified"


def test_changed_limits_after_freeze_fail(world):
    world.freeze()
    path = world.inputs / "tier2_limits.json"
    limits = json.loads(path.read_text())
    limits["classes"]["flexible"]["pair_fnat_min"] = 0.1
    path.write_text(json.dumps(limits))
    verdict = world.run()
    assert verdict["exit_code"] == accept.FAIL
    assert _failed(verdict, "tier2 limits match the frozen hash")


def test_missing_held_out_row_fails(world):
    evidence = world.evidence()
    evidence["sets"][1]["rows"].remove("H4")
    verdict = world.run(evidence)
    assert verdict["exit_code"] == accept.FAIL
    assert _failed(verdict, "required rows present")


def test_missing_job_fails(world):
    world.edit_tier2(
        0,
        lambda d: d.update(
            entries=[
                e
                for e in d["entries"]
                if (e["id"], e["job"]) != ("D13", "1a14_unconstrained")
            ]
        ),
    )
    verdict = world.run()
    assert _failed(verdict, "D13 1a14_unconstrained tier 2")
    assert verdict["exit_code"] == accept.FAIL


def test_development_seeds_on_a_held_out_row_fail(world):
    """Run with seeds 101/102, H10 would be a copy of D4."""
    held = Path(world.sets[1]["off"]).parent
    for key in ("off", "cand"):
        _trace(held / key, "H10", [101, 102])
    verdict = world.run()
    assert _failed(verdict, "H10 off seeds") and _failed(verdict, "H10 cand seeds")
    assert verdict["exit_code"] == accept.FAIL


def test_reused_held_out_seed_fails(world):
    world.log["draws"].append(
        {
            "id": "sm_120-0",
            "gpu": "sm_120",
            "candidate": "old",
            "seeds": {"H1": [201, 299]},
        }
    )
    world.write()
    verdict = world.run()
    assert _failed(verdict, "held-out seeds used once")


def test_unmatched_random_input_fails(world):
    _trace(Path(world.sets[0]["cand"]), "D5", [101, 102], salt="-changed")
    verdict = world.run()
    assert _failed(verdict, "D5 random inputs match off")


def test_uncommitted_candidate_fails(world):
    path = Path(world.sets[0]["cand"]) / "D2.trace.jsonl"
    lines = path.read_text().splitlines()
    meta = json.loads(lines[0])
    meta["fingerprint"]["dirty_paths"] = ["opendde/model/x.py"]
    path.write_text("\n".join([json.dumps(meta)] + lines[1:]) + "\n")
    assert _failed(world.run(), "D2 cand provenance")


def test_missing_tier1_evidence_fails(world):
    path = Path(world.sets[1]["tier1"])
    data = json.loads(path.read_text())
    data["entries"] = [e for e in data["entries"] if e["id"] != "H8"]
    path.write_text(json.dumps(data))
    verdict = world.run()
    assert _failed(verdict, "H8 h8_dna_duplex seed")


def test_tier1_limit_and_head_source(world):
    path = Path(world.sets[0]["tier1"])
    data = json.loads(path.read_text())
    data["entries"][0]["z_trunk"]["rel_fro"] = 0.2
    data["entries"][1]["head_source"] = "head"
    path.write_text(json.dumps(data))
    verdict = world.run()
    assert len(_failed(verdict, "tier 1")) == 2


def test_nondeterministic_candidate_fails(world):
    _artifacts(Path(world.sets[0]["cand_repeat"]), "D1", world.jobs("D1"), [101], "y")
    assert _failed(world.run(), "D1 candidate deterministic")


def test_off_drift_fails(world):
    _artifacts(Path(world.sets[0]["off_baseline"]), "D4", world.jobs("D4"), [102], "y")
    assert _failed(world.run(), "D4 off matches step 0 baseline")


def _h3_zn(primary, extra):
    return fx._zn(*primary, *extra)


def test_h3_extra_seeds_reach_a_conclusive_pass(tmp_path):
    zn = _h3_zn([(205, 6, 6), (206, 0, 0)], [(221, 3, 3), (222, 3, 3)])
    world = World(tmp_path, h3_extra=[221, 222], h3_zn=zn)
    world.freeze()
    verdict = world.run()
    assert verdict["failed"] == [] and verdict["exit_code"] == accept.FULL


def test_h3_unresolved_at_the_cap_is_not_an_acceptance(tmp_path):
    extra = [221, 222, 223, 224]
    zn = _h3_zn([(205, 3, 3), (206, 0, 0)], [(s, 1, 1) for s in extra])
    world = World(tmp_path, h3_extra=extra, h3_zn=zn)
    world.freeze()
    verdict = world.run()
    assert verdict["failed"] == []
    assert verdict["zn_claim"] == "unresolved"
    assert verdict["exit_code"] == accept.PARTIAL


def test_h3_extra_seed_measured_elsewhere_fails(tmp_path):
    zn = _h3_zn([(205, 6, 6), (206, 0, 0)], [(221, 6, 6)])
    world = World(tmp_path, h3_extra=[221], h3_zn=zn)
    link = Path(world.sets[1]["off"]) / "H3" / H3_JOB / "seed_221"
    link.unlink()
    other = tmp_path / "elsewhere"
    other.mkdir()
    link.symlink_to(other)
    assert _failed(world.run(), "extra seed 221 measured from its runs")


def test_h3_extra_seeds_out_of_log_order_fail(tmp_path):
    zn = _h3_zn([(205, 6, 6), (206, 0, 0)], [(222, 3, 3), (221, 3, 3)])
    world = World(tmp_path, h3_extra=[221, 222], h3_zn=zn)
    assert _failed(world.run(), "Zn seed order")


def test_h3_incomplete_extra_seeds_fail(tmp_path):
    """Below ten eligible ions with seeds left to run is a failure, not partial."""
    zn = _h3_zn([(205, 6, 6), (206, 0, 0)], [(221, 1, 1)])
    world = World(tmp_path, h3_extra=[221], h3_zn=zn)
    verdict = world.run()
    assert _failed(verdict, "H3 9sat_unconstrained tier 2")
    assert verdict["exit_code"] == accept.FAIL
