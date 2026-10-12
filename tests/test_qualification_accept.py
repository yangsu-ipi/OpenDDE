"""Tests for full candidate acceptance (scripts/qualification/accept.py).

Each test builds a complete synthetic evidence set for every sm_120 row, in a
throwaway Git checkout with its own held-out log history, then breaks one
thing that must stop a full acceptance.

Traces carry one recorded draw per rollout instead of the ~3,000 a real
rollout has, so ``trace_tools.check_manifest`` is replaced by a stand-in that
expects one draw per recorded rollout; the real manifest check has its own
self-test (``trace_tools.py selftest``)."""

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "qualification"))
import accept  # noqa: E402
import run_rows  # noqa: E402
import tier2_gates as t2  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "tier2_fixtures", ROOT / "tests" / "test_qualification_tier2.py"
)
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)

INPUTS = ROOT / "scripts" / "qualification" / "inputs"
VERSIONS = {"torch": "2.7.1+cu128"}
HOST_ENV = {"OPENDDE_QUAL_DATA": "/qual", "OPENDDE_QUAL_KALIGN": "/bin/kalign"}
CAND_ARGS = ["--optimization_mode", "fast"]
H3_JOB = "9sat_unconstrained"
N_SAMPLE = 5
DEV = [f"D{i}" for i in range(1, 16)]
HELD = [f"H{i}" for i in range(1, 11)]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=t@t"]
        + ["-c", "commit.gpgsign=false", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _content(rel: str, salt: str = "") -> str:
    return rel + salt


def _fake_manifest(meta, records):
    def key(r):
        return (r["sample_name"], r["seed"], r["rollout"])

    expected = Counter(key(r) for r in meta["rollouts"])
    observed = Counter(key(r) for r in records)
    return {
        "ok": expected == observed,
        "expected": sum(expected.values()),
        "observed": sum(observed.values()),
        "missing_by_site": {},
        "extra_by_site": {},
    }


def write_trace(run: Path, row: str, rec, rollouts, salt="", drop=0) -> None:
    records = [
        {
            "sample_name": r["sample_name"],
            "seed": r["seed"],
            "rollout": r["rollout"],
            "chunk": 0,
            "step": None,
            "site": "sampler.initial_noise",
            "kind": "gaussian",
            "outer": None,
            "sample": 0,
            "ordinal": 0,
            "dtype": "float32",
            "shape": [3],
            "raw_sha256": f"{row}-{r['sample_name']}-{r['seed']}{salt}",
            "cmp_sha256": "x",
        }
        for r in rollouts
    ][drop:]
    lines = [json.dumps({**rec, "rollouts": rollouts})]
    lines += [json.dumps(r) for r in records]
    (run / f"{row}.trace.jsonl").write_text("\n".join(lines) + "\n")


def write_artifacts(run: Path, row: str, jobs, seeds, salt: str = "") -> None:
    for rel in accept.expected_artifacts(jobs, seeds, N_SAMPLE):
        path = run / row / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_content(rel, salt))


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
    """A passing evidence set, its run records and the Git checkout accept.py reads."""

    def __init__(
        self, tmp: Path, h3_extra=(), h3_zn=None, frozen=False, cand_args=CAND_ARGS
    ):
        self.tmp = tmp
        self.repo = tmp / "repo"
        self.inputs = self.repo / "inputs"
        shutil.copytree(INPUTS, self.inputs, ignore=shutil.ignore_patterns("msa"))
        self.manifest = json.loads((INPUTS / "manifest.json").read_text())
        self.table = {r["id"]: r for r in self.manifest["rows"]}
        self.limits = json.loads((INPUTS / "tier2_limits.json").read_text())
        acceptance = json.loads((INPUTS / "acceptance.json").read_text())
        self.recipe = acceptance["recipe"]
        self.jobs_run = 0
        self.seeds = {row: [101, 102] for row in DEV}
        for n, row in enumerate(HELD):
            self.seeds[row] = [201 + 2 * n, 202 + 2 * n]
        baseline = {
            "rows": {
                row: {
                    rel: hashlib.sha256(_content(rel).encode()).hexdigest()
                    for rel in accept.expected_artifacts(
                        self.jobs(row), self.seeds[row], N_SAMPLE
                    )
                }
                for row in DEV
            }
        }
        (self.inputs / "baseline_off.json").write_text(json.dumps(baseline))
        self.log = {"description": "test", "draws": []}
        self.write_log()
        if frozen:
            acceptance["gpus"]["sm_120"]["frozen"] = {
                "tier1_limits_sha256": accept.sha256(self.inputs / "tier1_limits.json"),
                "tier2_limits_sha256": accept.sha256(self.inputs / "tier2_limits.json"),
            }
            (self.inputs / "acceptance.json").write_text(json.dumps(acceptance))
        _git(self.repo, "init", "-q")
        self.code = self.commit("candidate code")
        self.cand_args = list(cand_args)
        self.draw = {
            "id": "sm_120-1",
            "gpu": "sm_120",
            "candidate": "fast-test",
            "code_commit": self.code,
            "candidate_args": list(cand_args),
            "seeds": {r: self.seeds[r] for r in HELD},
            "zn_extra": {},
        }
        self.log["draws"].append(self.draw)
        self.draw_commit = self.commit_log()
        self.h3_extra = list(h3_extra)
        self.extra_commits = {}
        for s in self.h3_extra:
            self.draw["zn_extra"].setdefault("H3", []).append(s)
            self.extra_commits[s] = self.commit_log()
        self.sets = [
            self._build_set("development", DEV, self.code, h3_zn),
            self._build_set("held-out", HELD, self.draw_commit, h3_zn),
        ]
        for s in self.h3_extra:
            self._build_zn_set(s)

    # Git and log

    def commit(self, message="x") -> str:
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "--no-verify", "--allow-empty", "-m", message)
        return _git(self.repo, "rev-parse", "HEAD")

    def write_log(self):
        (self.inputs / "held_out_log.json").write_text(json.dumps(self.log, indent=1))

    def commit_log(self, message="log") -> str:
        self.write_log()
        return self.commit(message)

    # Runs

    def jobs(self, row):
        return [
            j["name"] for j in json.loads((ROOT / self.table[row]["input"]).read_text())
        ]

    def rollouts(self, row, seeds, n_sample=N_SAMPLE):
        guided = "--use_tfg_guidance" in self.table[row]["flags"]
        pairs = [(s, j) for s in seeds for j in self.jobs(row)]
        return [
            {
                "sample_name": j,
                "seed": s,
                "rollout": i,
                "N_sample": n_sample,
                "N_step": 200,
                "diffusion_chunk_size": 5,
                "rollout_seed": s,
                "guidance_configs": {"enable": guided},
            }
            for i, (s, j) in enumerate(pairs)
        ]

    def write_run(self, run: Path, row, seeds, commit, kind, capture=None, replay=None):
        """Write what ``run_rows.py`` leaves for one row: run record and trace."""
        tier1 = kind.startswith("tier1")
        extra = list(self.recipe["tier1_extra" if tier1 else "extra"])
        if kind in accept.CANDIDATE_KINDS:
            extra += self.cand_args
        env = {**HOST_ENV, **self.table[row].get("env", {})}
        overrides, cli, _ = run_rows.build_command(
            self.table[row],
            self.manifest,
            run / row,
            ",".join(map(str, seeds)),
            extra,
            environ=env,
        )
        self.jobs_run += 1
        traced = kind in ("off", "cand")
        rec = {
            "cli_args": cli,
            "config_overrides": overrides,
            "wrapper": {
                "draw_trace": str(run / f"{row}.trace.jsonl") if traced else None,
                "phase_timing": None,
                "tier1_capture": capture,
                "tier1_replay": replay,
            },
            "env": env,
            "exit_code": 0,
            "fingerprint": {
                "commit": commit,
                "dirty_paths": [],
                "host": "gnode2",
                "slurm_job_id": str(self.jobs_run),
                "utc": "2026-10-12T00:00:00Z",
                "gpu": {"capability": [12, 0]},
                "versions": VERSIONS,
            },
        }
        run.mkdir(parents=True, exist_ok=True)
        (run / f"{row}.run.json").write_text(json.dumps(rec))
        if traced:
            write_trace(run, row, rec, self.rollouts(row, seeds))
        return rec

    def _build_set(self, name, rows, commit, h3_zn):
        base = self.tmp / name
        runs = {k: base / k for k in ("off", "cand", "repeat", "t1_off", "t1_cand")}
        caps = {k: base / f"cap_{k}" for k in ("off", "cand")}
        tier1, entries = [], []
        for row in rows:
            seeds, jobs = self.seeds[row], self.jobs(row)
            for kind in ("off", "cand", "repeat"):
                self.write_run(runs[kind], row, seeds, commit, kind)
                write_artifacts(runs[kind], row, jobs, seeds)
            cap_off, cap_cand = str(caps["off"] / row), str(caps["cand"] / row)
            self.write_run(runs["t1_off"], row, seeds, commit, "tier1 off", cap_off)
            self.write_run(
                runs["t1_cand"], row, seeds, commit, "tier1 cand", cap_cand, cap_off
            )
            tier1 += _tier1(row, jobs, seeds)
            for job in json.loads((ROOT / self.table[row]["input"]).read_text()):
                facts = t2.job_facts(self.table[row], job)
                e = fx._entry(row, job["name"], facts, self.limits, seeds=tuple(seeds))
                if row == "H3" and h3_zn is not None:
                    e["zn"] = h3_zn
                entries.append(e)
        t1 = {"ref": str(caps["off"]), "other": str(caps["cand"]), "entries": tier1}
        (base / "tier1.json").write_text(json.dumps(t1))
        measure = {
            "off": str(runs["off"]),
            "cand": str(runs["cand"]),
            "entries": entries,
        }
        (base / "tier2.json").write_text(json.dumps(measure))
        return {
            "name": name,
            "rows": list(rows),
            "off": str(runs["off"]),
            "cand": str(runs["cand"]),
            "cand_repeat": str(runs["repeat"]),
            "tier1_off": str(runs["t1_off"]),
            "tier1_cand": str(runs["t1_cand"]),
            "tier1": str(base / "tier1.json"),
            "tier2": str(base / "tier2.json"),
        }

    def _build_zn_set(self, seed):
        base = self.tmp / f"zn-{seed}"
        held = self.tmp / "held-out"
        for kind in ("off", "cand"):
            self.write_run(base / kind, "H3", [seed], self.extra_commits[seed], kind)
            write_artifacts(base / kind, "H3", [H3_JOB], [seed])
            link = held / kind / "H3" / H3_JOB / f"seed_{seed}"
            link.symlink_to(base / kind / "H3" / H3_JOB / f"seed_{seed}")
        self.sets.append(
            {
                "name": f"zn-{seed}",
                "zn_only": True,
                "rows": ["H3"],
                "seeds": [seed],
                "off": str(base / "off"),
                "cand": str(base / "cand"),
            }
        )

    def run_dir(self, row, kind):
        s = self.sets[0] if row.startswith("D") else self.sets[1]
        return Path(s["cand_repeat" if kind == "repeat" else kind])

    def edit_record(self, row, kind, fn):
        """Change a run record, and its trace header to match."""
        run = self.run_dir(row, kind)
        path = run / f"{row}.run.json"
        rec = json.loads(path.read_text())
        fn(rec)
        path.write_text(json.dumps(rec))
        trace = run / f"{row}.trace.jsonl"
        if trace.is_file():
            lines = trace.read_text().splitlines()
            meta = {**json.loads(lines[0]), **rec}
            trace.write_text("\n".join([json.dumps(meta)] + lines[1:]) + "\n")

    def rewrite_traces(self, row, rollouts, **kw):
        for kind in ("off", "cand"):
            run = self.run_dir(row, kind)
            rec = json.loads((run / f"{row}.run.json").read_text())
            write_trace(run, row, rec, rollouts, **kw)

    # Evidence

    def evidence(self):
        return {
            "gpu": "sm_120",
            "route": "port-chosen",
            "candidate": "fast-test",
            "held_out_draw": "sm_120-1",
            "sets": copy.deepcopy(self.sets),
        }

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
    monkeypatch.setattr(accept.trace_tools, "check_manifest", _fake_manifest)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _failed(verdict, fragment):
    return [c for c in verdict["failed"] if fragment in c]


def _fails(world, fragment, evidence=None):
    verdict = world.run(evidence)
    assert _failed(verdict, fragment), verdict["failed"]
    assert verdict["exit_code"] == accept.FAIL
    return verdict


# Complete evidence


def test_complete_pass_is_provisional_until_frozen_then_full(tmp_path):
    world = World(tmp_path / "a")
    verdict = world.run()
    assert verdict["failed"] == []
    assert verdict["exit_code"] == accept.PROVISIONAL
    assert verdict["code_commit"] == world.code
    evaluated = {
        c["check"].split()[0]
        for c in verdict["checks"]
        if c["check"].endswith(" tier 2")
    }
    required = json.loads((INPUTS / "acceptance.json").read_text())
    assert evaluated == set(required["gpus"]["sm_120"]["routes"]["port-chosen"]["rows"])
    for fragment in ("random inputs", "off rollouts", "repeat is a separate run"):
        rows = {
            c["check"].split()[0] for c in verdict["checks"] if fragment in c["check"]
        }
        assert rows == evaluated, fragment
    world = World(tmp_path / "b", frozen=True)
    verdict = world.run()
    assert verdict["exit_code"] == accept.FULL and verdict["zn_claim"] == "qualified"


def test_changed_limits_after_freeze_fail(tmp_path):
    world = World(tmp_path, frozen=True)
    path = world.inputs / "tier2_limits.json"
    limits = json.loads(path.read_text())
    limits["classes"]["flexible"]["pair_fnat_min"] = 0.1
    path.write_text(json.dumps(limits))
    verdict = _fails(world, "tier2 limits match the frozen hash")
    assert _failed(verdict, "acceptance checkout clean")


def test_missing_held_out_row_fails(world):
    evidence = world.evidence()
    evidence["sets"][1]["rows"].remove("H4")
    _fails(world, "required rows present", evidence)


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
    _fails(world, "D13 1a14_unconstrained tier 2")


# Seeds, commits and the held-out log


def test_development_seeds_on_a_held_out_row_fail(world):
    """Run with seeds 101/102, H10 would be a copy of D4."""
    for kind in ("off", "cand"):
        run = world.run_dir("H10", kind)
        world.write_run(run, "H10", [101, 102], world.draw_commit, kind)
    verdict = _fails(world, "H10 off command")
    assert _failed(verdict, "H10 cand command") and _failed(verdict, "H10 off rollouts")


def test_reused_held_out_seed_fails(world):
    world.log["draws"].append(
        {
            "id": "sm_120-0",
            "gpu": "sm_120",
            "candidate": "old",
            "seeds": {"H1": [201, 299]},
        }
    )
    world.commit_log()
    verdict = _fails(world, "held-out seeds used once")
    assert _failed(verdict, "held-out seeds used once across the log history")


def test_draw_without_candidate_arguments_fails(tmp_path):
    world = World(tmp_path, cand_args=[])
    _fails(world, "records the code commit and candidate arguments")


def test_seed_absent_from_its_run_commit_fails(world):
    """A held-out run made before its draw was committed."""
    world.edit_record("H1", "off", lambda r: r["fingerprint"].update(commit=world.code))
    verdict = _fails(world, "H1 off seeds logged at its commit")
    assert not _failed(verdict, "H1 off provenance")


def test_code_or_limit_change_between_runs_fails(world):
    path = world.inputs / "tier2_limits.json"
    original = path.read_text()
    path.write_text(original + "\n")
    changed = world.commit("change the limits")
    path.write_text(original)
    world.commit("restore them")
    for kind in ("off", "cand"):
        world.edit_record("H5", kind, lambda r: r["fingerprint"].update(commit=changed))
    verdict = _fails(world, "H5 off provenance")
    assert not _failed(verdict, "differs from the code commit only")
    assert not _failed(verdict, "H5 off seeds logged")


def test_run_commit_outside_the_checkout_history_fails(world):
    _git(world.repo, "checkout", "-q", "-b", "side", world.code)
    side = world.commit("side")
    _git(world.repo, "checkout", "-q", "-")
    world.edit_record("D3", "cand", lambda r: r["fingerprint"].update(commit=side))
    _fails(world, "D3 cand provenance")


def test_edited_log_entry_fails(world):
    original = copy.deepcopy(world.log)
    world.draw["seeds"]["H1"] = [298, 299]
    world.commit_log("edit a draw")
    world.log = original
    world.commit_log("put it back")
    verdict = _fails(world, "held-out log history only appends")
    assert not _failed(verdict, "across the log history")


def test_seed_from_a_removed_entry_fails(world):
    original = copy.deepcopy(world.log)
    world.log["draws"].append(
        {
            "id": "scratch",
            "gpu": "sm_120",
            "candidate": "x",
            "seeds": {"H1": [201, 280]},
        }
    )
    world.commit_log("draw again")
    world.log = original
    world.commit_log("remove it")
    verdict = _fails(world, "held-out seeds used once across the log history")
    assert "held-out seeds used once" not in verdict["failed"]


def test_dirty_checkout_fails(world):
    (world.repo / "stray.py").write_text("x = 1\n")
    _fails(world, "acceptance checkout clean")


# Command line and environment


def test_wrong_input_path_fails(world):
    def edit(rec):
        rec["cli_args"][rec["cli_args"].index("-i") + 1] = "examples/other.json"

    world.edit_record("D2", "off", edit)
    _fails(world, "D2 off command")


def test_undeclared_override_fails(world):
    world.edit_record("D9", "cand", lambda r: r["config_overrides"].append("x.y=1"))
    _fails(world, "D9 cand command")


def test_candidate_run_with_the_off_command_fails(world):
    """A candidate whose mode flag was dropped would match off everywhere else."""
    world.write_run(world.run_dir("D2", "cand"), "D2", [101, 102], world.code, "off")
    verdict = _fails(world, "D2 cand command")
    assert not _failed(verdict, "D2 random inputs")


def test_wrong_layernorm_environment_fails(world):
    world.edit_record(
        "D4", "cand", lambda r: r["env"].update(LAYERNORM_TYPE="fast_layernorm")
    )
    _fails(world, "D4 cand environment")


def test_row_environment_is_required(world):
    world.edit_record("D11", "off", lambda r: r["env"].pop("OPENDDE_RIGID_CONTACT"))
    _fails(world, "D11 off environment")


def test_uncommitted_candidate_fails(world):
    world.edit_record(
        "D2", "cand", lambda r: r["fingerprint"].update(dirty_paths=["opendde/x.py"])
    )
    _fails(world, "D2 cand provenance")


def test_trace_from_another_run_fails(world):
    path = world.run_dir("D6", "off") / "D6.run.json"
    rec = json.loads(path.read_text())
    rec["fingerprint"]["slurm_job_id"] = "other"
    path.write_text(json.dumps(rec))
    _fails(world, "D6 traces usable")


# Traces and artifacts


def test_unmatched_random_input_fails(world):
    run = world.run_dir("D5", "cand")
    rec = json.loads((run / "D5.run.json").read_text())
    write_trace(run, "D5", rec, world.rollouts("D5", [101, 102]), salt="-x")
    _fails(world, "D5 random inputs match off")


def test_same_draw_missing_from_both_traces_fails(world):
    world.rewrite_traces("D2", world.rollouts("D2", [101, 102]), drop=1)
    verdict = _fails(world, "D2 off draws complete")
    assert not _failed(verdict, "D2 random inputs")


def test_same_rollout_missing_from_both_traces_fails(world):
    world.rewrite_traces("D2", world.rollouts("D2", [101]))
    verdict = _fails(world, "D2 off rollouts")
    assert not _failed(verdict, "D2 random inputs")
    assert not _failed(verdict, "draws complete")


def test_wrong_sample_count_fails(world):
    world.rewrite_traces("H2", world.rollouts("H2", world.seeds["H2"], n_sample=4))
    _fails(world, "H2 cand rollouts")


def test_same_job_missing_from_candidate_and_repeat_fails(world):
    for kind in ("cand", "repeat"):
        shutil.rmtree(world.run_dir("D13", kind) / "D13" / "1a14_unconstrained")
    verdict = _fails(world, "D13 cand artifacts complete")
    assert not _failed(verdict, "D13 candidate deterministic")


# Repeat and baseline


def test_repeat_aliased_to_candidate_fails(world):
    evidence = world.evidence()
    evidence["sets"][0]["cand_repeat"] = evidence["sets"][0]["cand"]
    verdict = _fails(world, "D1 repeat is a separate run", evidence)
    assert not _failed(verdict, "D1 candidate deterministic")


def test_repeat_copied_from_candidate_fails(world):
    evidence = world.evidence()
    copied = world.tmp / "copied"
    shutil.copytree(evidence["sets"][1]["cand"], copied, symlinks=True)
    evidence["sets"][1]["cand_repeat"] = str(copied)
    verdict = _fails(world, "H6 repeat is a separate run", evidence)
    assert _failed(verdict, "H6 repeat provenance")


def test_repeat_without_run_record_fails(world):
    (world.run_dir("D1", "repeat") / "D1.run.json").unlink()
    _fails(world, "D1 repeat run record")


def test_nondeterministic_candidate_fails(world):
    write_artifacts(world.run_dir("D1", "repeat"), "D1", world.jobs("D1"), [101], "y")
    _fails(world, "D1 candidate deterministic")


def test_off_drift_fails(world):
    for kind in ("off", "cand", "repeat"):
        write_artifacts(world.run_dir("D4", kind), "D4", world.jobs("D4"), [102], "y")
    verdict = _fails(world, "D4 off matches step 0 baseline")
    assert not _failed(verdict, "D4 candidate deterministic")


def test_missing_baseline_fails(world):
    (world.inputs / "baseline_off.json").unlink()
    world.commit("drop the baseline")
    verdict = _fails(world, "step 0 baseline digests present")
    assert _failed(verdict, "D1 off matches step 0 baseline")


# Tier 1


def test_missing_tier1_evidence_fails(world):
    path = Path(world.sets[1]["tier1"])
    data = json.loads(path.read_text())
    data["entries"] = [e for e in data["entries"] if e["id"] != "H8"]
    path.write_text(json.dumps(data))
    _fails(world, "H8 h8_dna_duplex seed")


def test_tier1_limit_and_head_source(world):
    path = Path(world.sets[0]["tier1"])
    data = json.loads(path.read_text())
    data["entries"][0]["z_trunk"]["rel_fro"] = 0.2
    data["entries"][1]["head_source"] = "head"
    path.write_text(json.dumps(data))
    verdict = world.run()
    assert len([c for c in verdict["failed"] if c.endswith(" tier 1")]) == 2


def test_substituted_tier1_comparison_fails(world):
    path = Path(world.sets[0]["tier1"])
    data = json.loads(path.read_text())
    data["ref"] = str(world.tmp / "other_capture")
    path.write_text(json.dumps(data))
    _fails(world, "D7 tier 1 measured its declared captures")


def test_tier1_candidate_replaying_another_off_fails(world):
    path = Path(world.sets[0]["tier1_cand"]) / "D3.run.json"
    rec = json.loads(path.read_text())
    rec["wrapper"]["tier1_replay"] = str(world.tmp / "elsewhere" / "D3")
    path.write_text(json.dumps(rec))
    _fails(world, "D3 tier 1 measured its declared captures")


def test_tier1_run_with_the_full_recipe_fails(world):
    run = Path(world.sets[1]["tier1_off"])
    world.write_run(run, "H9", world.seeds["H9"], world.draw_commit, "off")
    verdict = _fails(world, "H9 tier 1 off command")
    assert _failed(verdict, "H9 tier 1 off wrapper options")


# H3 extra Zn seeds


def _h3_zn(primary, extra):
    return fx._zn(*primary, *extra)


def test_h3_extra_seeds_in_log_only_commits_reach_a_conclusive_pass(tmp_path):
    zn = _h3_zn([(205, 6, 6), (206, 0, 0)], [(221, 3, 3), (222, 3, 3)])
    world = World(tmp_path, h3_extra=[221, 222], h3_zn=zn, frozen=True)
    assert len({world.draw_commit, *world.extra_commits.values()}) == 3
    verdict = world.run()
    assert verdict["failed"] == [] and verdict["exit_code"] == accept.FULL


def test_h3_unresolved_at_the_cap_is_not_an_acceptance(tmp_path):
    extra = [221, 222, 223, 224]
    zn = _h3_zn([(205, 3, 3), (206, 0, 0)], [(s, 1, 1) for s in extra])
    world = World(tmp_path, h3_extra=extra, h3_zn=zn, frozen=True)
    verdict = world.run()
    assert verdict["failed"] == []
    assert verdict["zn_claim"] == "unresolved"
    assert verdict["exit_code"] == accept.PARTIAL


def test_h3_extra_seed_run_before_its_log_commit_fails(tmp_path):
    zn = _h3_zn([(205, 6, 6), (206, 0, 0)], [(221, 3, 3), (222, 3, 3)])
    world = World(tmp_path, h3_extra=[221, 222], h3_zn=zn)
    path = Path(world.sets[-1]["off"]) / "H3.run.json"
    rec = json.loads(path.read_text())
    rec["fingerprint"]["commit"] = world.extra_commits[221]
    path.write_text(json.dumps(rec))
    _fails(world, "H3 extra seeds 222 off seeds logged at its commit")


def test_h3_extra_seed_without_its_run_fails(tmp_path):
    zn = _h3_zn([(205, 6, 6), (206, 0, 0)], [(221, 3, 3), (222, 3, 3)])
    world = World(tmp_path, h3_extra=[221, 222], h3_zn=zn)
    evidence = world.evidence()
    evidence["sets"].pop()
    verdict = _fails(world, "H3 extra Zn runs cover the logged seeds", evidence)
    assert _failed(verdict, "extra seed 222 measured from its runs")


def test_h3_extra_seed_measured_elsewhere_fails(tmp_path):
    zn = _h3_zn([(205, 6, 6), (206, 0, 0)], [(221, 6, 6)])
    world = World(tmp_path, h3_extra=[221], h3_zn=zn)
    link = Path(world.sets[1]["off"]) / "H3" / H3_JOB / "seed_221"
    link.unlink()
    other = tmp_path / "elsewhere"
    other.mkdir()
    link.symlink_to(other)
    _fails(world, "extra seed 221 measured from its runs")


def test_h3_extra_seeds_out_of_log_order_fail(tmp_path):
    zn = _h3_zn([(205, 6, 6), (206, 0, 0)], [(222, 3, 3), (221, 3, 3)])
    world = World(tmp_path, h3_extra=[221, 222], h3_zn=zn)
    _fails(world, "Zn seed order")


def test_h3_incomplete_extra_seeds_fail(tmp_path):
    """Below ten eligible ions with seeds left to run is a failure, not partial."""
    zn = _h3_zn([(205, 6, 6), (206, 0, 0)], [(221, 1, 1)])
    world = World(tmp_path, h3_extra=[221], h3_zn=zn)
    _fails(world, "H3 9sat_unconstrained tier 2")


# run_rows


def test_run_rows_runs_the_command_accept_rebuilds(tmp_path, monkeypatch):
    calls = []

    def call(cmd, **kw):
        calls.append((cmd, kw["env"]))
        return 0

    monkeypatch.setattr(run_rows.subprocess, "call", call)
    monkeypatch.setenv("OPENDDE_QUAL_DATA", "/qual")
    monkeypatch.setenv("OPENDDE_QUAL_KALIGN", "/bin/kalign")
    out = tmp_path / "runs"
    argv = ["run_rows.py", "--rows", "D5,D11", "--out", str(out), "--draw-trace"]
    monkeypatch.setattr(sys, "argv", argv + ["--", "--deterministic", "true"])
    assert run_rows.main() == 0
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    rows = {r["id"]: r for r in manifest["rows"]}
    assert len(calls) == 2
    for (cmd, env), row in zip(calls, ("D5", "D11")):
        split = cmd.index("--")
        wrapper, cli = cmd[2:split], cmd[split + 1 :]
        overrides, expected, row_env = run_rows.build_command(
            rows[row], manifest, out / row, "101,102", ["--deterministic", "true"]
        )
        assert cli == expected
        sets = [wrapper[i + 1] for i, a in enumerate(wrapper) if a == "--set"]
        assert sets == overrides
        record = wrapper[wrapper.index("--run-record") + 1]
        assert record == str(out / f"{row}.run.json")
        assert all(env[k] == v for k, v in row_env.items())
