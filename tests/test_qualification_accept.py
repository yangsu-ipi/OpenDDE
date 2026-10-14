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
import assets  # noqa: E402
import provenance  # noqa: E402
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
CONTRACT_TAG = "qual-test-contract-1"
H3_JOB = "9sat_unconstrained"
N_SAMPLE = 5
DEV = [f"D{i}" for i in range(1, 16)]
HELD = [f"H{i}" for i in range(1, 11)]
DIGESTS = json.loads((INPUTS / "digests.json").read_text())
EXTERNAL = json.loads((INPUTS / "external_digests.json").read_text())
EFFECTIVE = json.loads((INPUTS / "effective_inputs.json").read_text())
ASSET_ROOT = "/qual-assets"
ASSET_REF = {
    name: {
        "path": f"{ASSET_ROOT}/{rel}",
        "sha256": hashlib.sha256(name.encode()).hexdigest(),
    }
    for name, rel in assets.LAYOUT.items()
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=t@t"]
        + ["-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _content(rel: str, salt: str = "") -> str:
    return rel + salt


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _digests(jobs, seeds) -> dict[str, str]:
    """Digests of the artifacts ``write_artifacts`` writes (content = path)."""
    return {
        rel: _sha(_content(rel))
        for rel in accept.expected_artifacts(jobs, seeds, N_SAMPLE)
    }


def _captures(capture: str, jobs, seeds) -> dict[str, str]:
    return {
        f"{job}/seed_{s}/tier1.pt": _sha(f"{capture}/{job}/{s}")
        for job in jobs
        for s in seeds
    }


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
        self,
        tmp: Path,
        h3_extra=(),
        h3_zn=None,
        frozen=False,
        cand_args=CAND_ARGS,
        candidate_change=None,
        receipt="before",
        receipt_changes=None,
    ):
        """The checkout has a contract commit (tagged ``CONTRACT_TAG`` when
        ``frozen``), then the candidate's code commit, which changes model code
        outside the qualification directory and applies ``candidate_change``.
        The development set is run at the code commit; its rehearsal receipt is
        committed ``before`` the draw, ``with`` it, ``after`` it or not at all
        (None)."""
        self.tmp = tmp
        self.repo = tmp / "repo"
        self.qual = self.repo / "scripts" / "qualification"
        self.inputs = self.qual / "inputs"
        shutil.copytree(INPUTS, self.inputs, ignore=shutil.ignore_patterns("msa"))
        (self.qual / "zn_sites.py").write_text("SAME_OUTCOME = 2.0\n")
        (self.repo / "opendde").mkdir()
        (self.repo / "opendde" / "model.py").write_text("stock = True\n")
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
            "rows": {row: _digests(self.jobs(row), self.seeds[row]) for row in DEV}
        }
        (self.inputs / "baseline_off.json").write_text(json.dumps(baseline))
        self.log = {"description": "test", "receipts": [], "draws": []}
        self.write_log()
        acceptance["gpus"]["sm_120"]["frozen"] = (
            {
                "tier1_limits_sha256": accept.sha256(self.inputs / "tier1_limits.json"),
                "tier2_limits_sha256": accept.sha256(self.inputs / "tier2_limits.json"),
                "contract_tag": CONTRACT_TAG,
            }
            if frozen
            else {}
        )
        (self.inputs / "acceptance.json").write_text(json.dumps(acceptance))
        reference = {"gpus": {"sm_120": {"root": ASSET_ROOT, "assets": ASSET_REF}}}
        (self.inputs / "runtime_assets.json").write_text(json.dumps(reference))
        _git(self.repo, "init", "-q")
        self.contract = self.commit("contract")
        if frozen:
            _git(self.repo, "tag", "-a", CONTRACT_TAG, "-m", "contract")
        (self.repo / "opendde" / "model.py").write_text("stock = False\n")
        if candidate_change:
            candidate_change(self)
        self.code = self.commit("candidate code")
        self.cand_args = list(cand_args)
        self.head = self.code
        dev = self._build_set("development", DEV, self.code, h3_zn)
        self.receipt = {
            "id": "rehearsal-1",
            "gpu": "sm_120",
            "candidate": "fast-test",
            "route": "dev-rehearsal",
            "contract_tag": CONTRACT_TAG if frozen else None,
            "code_commit": self.code,
            "candidate_args": list(cand_args),
            "evidence_sha256": accept.evidence_digest([dev]),
            "verdict_sha256": _sha("verdict"),
            "status": accept.REHEARSAL_STATUS,
            "exit_code": accept.REHEARSAL,
            "gate_sha256": _sha("gate"),
            "gate_commit": self.contract if frozen else None,
            "gate_approved_tag": CONTRACT_TAG if frozen else None,
            **(receipt_changes or {}),
        }
        if receipt == "before":
            self.log["receipts"].append(self.receipt)
            self.receipt_commit = self.commit_log("receipt")
        self.draw = {
            "id": "sm_120-1",
            "gpu": "sm_120",
            "candidate": "fast-test",
            "code_commit": self.code,
            "candidate_args": list(cand_args),
            "receipt": "rehearsal-1",
            "seeds": {r: self.seeds[r] for r in HELD},
            "zn_extra": {},
        }
        self.log["draws"].append(self.draw)
        if receipt == "with":
            self.log["receipts"].append(self.receipt)
        self.draw_commit = self.commit_log()
        if receipt == "after":
            self.log["receipts"].append(self.receipt)
            self.commit_log("late receipt")
        self.h3_extra = list(h3_extra)
        self.extra_commits = {}
        for s in self.h3_extra:
            self.draw["zn_extra"].setdefault("H3", []).append(s)
            self.extra_commits[s] = self.commit_log()
        self.head = _git(self.repo, "rev-parse", "HEAD")
        self.sets = [dev, self._build_set("held-out", HELD, self.draw_commit, h3_zn)]
        for s in self.h3_extra:
            self._build_zn_set(s)

    def imports(self):
        """Model packages loaded from this checkout."""
        return {pkg: str(self.repo / pkg) for pkg in ("opendde", "runner")}

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
        jobs = self.jobs(row)
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
            "preexisting": {"output": False, "tier1_capture": False if tier1 else None},
            "exception": None,
            "outputs": {
                "prediction_sha256": {} if tier1 else _digests(jobs, seeds),
                "tier1_sha256": _captures(capture, jobs, seeds) if tier1 else {},
            },
            "exit_code": 0,
            "provenance": self.provenance(row, seeds, cli, run / row),
            "fingerprint": {
                "commit": commit,
                "dirty_paths": [],
                "host": "gnode2",
                "slurm_job_id": str(self.jobs_run),
                "utc": "2026-10-12T00:00:00Z",
                "gpu": {"capability": [12, 0]},
                "versions": VERSIONS,
                "imports": self.imports(),
            },
        }
        run.mkdir(parents=True, exist_ok=True)
        (run / f"{row}.run.json").write_text(json.dumps(rec))
        if traced:
            write_trace(run, row, rec, self.rollouts(row, seeds))
        return rec

    def provenance(self, row, seeds, cli, row_out):
        """The contract-6 input provenance ``run_opendde.py`` records for a
        run that read its frozen inputs and this GPU's frozen assets."""
        spec = self.table[row]
        flags = {
            "use_msa": accept._cli_flag(cli, "--use_msa", True),
            "use_template": accept._cli_flag(cli, "--use_template", False),
            "use_rna_msa": accept._cli_flag(cli, "--use_rna_msa", False),
        }
        key = Path(spec["input"]).as_posix()
        effective = EFFECTIVE.get(key) if flags["use_msa"] else None
        read = effective["effective_sha256"] if effective else DIGESTS[key]
        _, jobs = provenance.effective_input(
            (ROOT / spec["input"]).read_bytes(), flags["use_msa"]
        )
        referenced = {
            p: {
                "resolved": str(ROOT / p),
                "before": DIGESTS[Path(p).as_posix()],
                "after": DIGESTS[Path(p).as_posix()],
            }
            for p in provenance.referenced_files(jobs)
        }
        names = accept.BASE_ASSETS + (
            accept.TEMPLATE_ASSETS if flags["use_template"] else ()
        )
        prov = {
            "refused": None,
            "source": {
                "path": spec["input"],
                "resolved": str(ROOT / key),
                "sha256": DIGESTS[key],
            },
            "flags": flags,
            "expected_effective_sha256": read,
            "parser_reads": [
                {
                    "path": "effective.json",
                    "resolved": "effective.json",
                    "sha256": read,
                    "matches_parse": True,
                }
            ],
            "referenced": referenced,
            "template_mmcif": None,
            "template_cache": None,
            "kalign": None,
            "assets": {
                n: {
                    "path": ASSET_REF[n]["path"],
                    "resolved": ASSET_REF[n]["path"],
                    "sha256": ASSET_REF[n]["sha256"],
                }
                for n in names
            },
            "seeds": {job: sorted(seeds) for job in self.jobs(row)},
        }
        if flags["use_template"]:
            prefix = accept._override(spec, "data.template.prot_template_mmcif_dir")
            tree = {
                k[len(prefix) + 1 :]: v
                for k, v in EXTERNAL.items()
                if k.startswith(prefix + "/")
            }
            prov["template_mmcif"] = {
                "dir": "/qual/template_mmcif",
                "before": tree,
                "after": dict(tree),
            }
            prov["template_cache"] = {
                "path": str(row_out / "template_cache"),
                "inside_output": True,
                "absent_before": True,
                "absent_at_parse": True,
            }
            prov["kalign"] = {
                "requested": HOST_ENV["OPENDDE_QUAL_KALIGN"],
                "requested_resolved": ASSET_REF["kalign_binary_path"]["path"],
                "executable": True,
            }
        return prov

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
            for e in _tier1(row, jobs, seeds):
                key = f"{e['job']}/seed_{e['seed']}/tier1.pt"
                e["ref_sha256"] = _captures(cap_off, jobs, seeds)[key]
                e["other_sha256"] = _captures(cap_cand, jobs, seeds)[key]
                tier1.append(e)
            extra = self.h3_extra if row == "H3" else []
            for job in json.loads((ROOT / self.table[row]["input"]).read_text()):
                facts = t2.job_facts(self.table[row], job)
                e = fx._entry(row, job["name"], facts, self.limits, seeds=tuple(seeds))
                if row == "H3" and h3_zn is not None:
                    e["zn"] = h3_zn
                e["seed_pairs"] = [[s, s] for s in seeds]
                e["zn_extra_seeds"] = list(extra)
                inputs = _digests([job["name"]], list(seeds) + list(extra))
                e["inputs"] = {"off": inputs, "cand": dict(inputs)}
                entries.append(e)
        source = {"commit": self.head, "dirty_paths": [], "imports": self.imports()}
        t1 = {
            "ref": str(caps["off"]),
            "other": str(caps["cand"]),
            "source": source,
            "entries": tier1,
        }
        (base / "tier1.json").write_text(json.dumps(t1))
        measure = {
            "off": str(runs["off"]),
            "cand": str(runs["cand"]),
            "draw": "sm_120-1",
            "source": dict(source),
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
    effective = json.loads((INPUTS / "effective_inputs.json").read_text())
    monkeypatch.setattr(accept.freeze_inputs, "effective_digests", lambda m: effective)
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
    contract = [
        c["check"]
        for c in verdict["checks"]
        if "contract tag" in c["check"] and not c["check"].startswith("rehearsal")
    ]
    assert contract == [
        "contract tag present",
        "contract tag is the newest",
        "code commit descends from the contract tag",
        "qualification files unchanged since the contract tag",
    ]
    names = {c["check"] for c in verdict["checks"]}
    assert {
        "draw names a logged rehearsal receipt",
        "rehearsal receipt passed",
        "rehearsal receipt matches the draw",
        "rehearsal receipt names the contract tag",
        "rehearsal receipt records a gate at the contract tag",
        "development evidence matches the rehearsal receipt",
        "rehearsal receipt committed before the draw",
        "each candidate drawn at most once",
    } <= names


def _bump_constant(world):
    (world.qual / "zn_sites.py").write_text("SAME_OUTCOME = 2.5\n")


def _loosen_limits_and_rehash(world):
    path = world.inputs / "tier2_limits.json"
    limits = json.loads(path.read_text())
    limits["classes"]["flexible"]["pair_fnat_min"] = 0.1
    path.write_text(json.dumps(limits))
    acceptance = json.loads((world.inputs / "acceptance.json").read_text())
    acceptance["gpus"]["sm_120"]["frozen"]["tier2_limits_sha256"] = accept.sha256(path)
    (world.inputs / "acceptance.json").write_text(json.dumps(acceptance))


@pytest.mark.parametrize("change", [_bump_constant, _loosen_limits_and_rehash])
def test_qualification_change_in_the_candidate_commit_fails(tmp_path, change):
    """A gate constant, or limits changed together with their frozen hash."""
    world = World(tmp_path, frozen=True, candidate_change=change)
    verdict = _fails(world, "qualification files unchanged since the contract tag")
    assert verdict["failed"] == ["qualification files unchanged since the contract tag"]


def test_code_commit_not_descending_from_the_tag_fails(tmp_path):
    """The same qualification tree, but tagged on unrelated history."""
    world = World(tmp_path, frozen=True)
    tree = _git(world.repo, "rev-parse", f"{CONTRACT_TAG}^{{tree}}")
    orphan = _git(world.repo, "commit-tree", tree, "-m", "elsewhere")
    _git(world.repo, "tag", "-f", "-a", CONTRACT_TAG, "-m", "moved", orphan)
    verdict = _fails(world, "code commit descends from the contract tag")
    assert verdict["failed"] == [
        "rehearsal receipt records a gate at the contract tag",
        "code commit descends from the contract tag",
    ]


def test_missing_contract_tag_fails(tmp_path):
    world = World(tmp_path, frozen=True)
    _git(world.repo, "tag", "-d", CONTRACT_TAG)
    _fails(world, "contract tag present")


def test_freeze_without_a_contract_tag_fails(tmp_path):
    """Limit hashes alone do not declare a complete freeze."""

    def drop_tag_name(world):
        acceptance = json.loads((world.inputs / "acceptance.json").read_text())
        acceptance["gpus"]["sm_120"]["frozen"]["contract_tag"] = None
        (world.inputs / "acceptance.json").write_text(json.dumps(acceptance))

    world = World(tmp_path, frozen=True, candidate_change=drop_tag_name)
    verdict = _fails(world, "contract tag present")
    assert "qualification files unchanged since the contract tag" not in [
        c["check"] for c in verdict["checks"]
    ]


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
            "seeds": {"H1": [201, 999]},
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
    world.draw["seeds"]["H1"] = [998, 999]
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


# Seed pairs and the run-to-measurement digest chain


def _edit_json(path: Path, fn) -> None:
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def _entry_of(data, row, job=None):
    return next(
        e
        for e in data["entries"]
        if e["id"] == row and (job is None or e["job"] == job)
    )


def test_swapped_seed_pairs_fail(world):
    """Crossed pairs keep both seed sets and sample counts but change the pairs."""
    seeds = world.seeds["H2"]
    world.edit_tier2(
        1,
        lambda d: _entry_of(d, "H2").update(
            seed_pairs=[[seeds[0], seeds[1]], [seeds[1], seeds[0]]]
        ),
    )
    verdict = _fails(world, "H2 7pzb tier 2 seed pairs")
    assert not _failed(verdict, "H2 7pzb tier 2 cand seeds")


def test_missing_tier1_capture_fails(world):
    key = "1a14_pocket/seed_102/tier1.pt"
    path = Path(world.sets[0]["tier1_cand"]) / "D10.run.json"
    _edit_json(path, lambda r: r["outputs"]["tier1_sha256"].pop(key))
    verdict = _fails(world, "D10 tier 1 cand captures complete")
    assert _failed(verdict, "D10 1a14_pocket seed 102 tier 1 inputs match its runs")


def test_tier1_comparison_of_other_captures_fails(world):
    path = Path(world.sets[1]["tier1"])
    _edit_json(path, lambda d: _entry_of(d, "H7").update(other_sha256=_sha("other")))
    _fails(world, "H7 9lh2_contact seed 213 tier 1 inputs match its runs")


def test_artifact_changed_after_its_run_fails(world):
    run = world.run_dir("D8", "cand")
    write_artifacts(run, "D8", world.jobs("D8"), [101], "-edited")
    write_artifacts(
        world.run_dir("D8", "repeat"), "D8", world.jobs("D8"), [101], "-edited"
    )
    verdict = _fails(world, "D8 cand artifacts match its run record")
    assert not _failed(verdict, "D8 candidate deterministic")


def test_measurement_of_an_earlier_run_fails(world):
    """A rerun at the same paths left a measurement of the earlier outputs."""
    job = world.jobs("D12")[0]
    rel = f"{job}/seed_101/predictions/{job}_sample_0.cif"
    for kind in ("cand", "repeat"):
        run = world.run_dir("D12", kind)
        (run / "D12" / rel).write_text("rerun")
        world.edit_record(
            "D12",
            kind,
            lambda r: r["outputs"]["prediction_sha256"].update({rel: _sha("rerun")}),
        )
    verdict = _fails(world, f"D12 {job} tier 2 cand inputs match its runs")
    assert not _failed(verdict, "D12 cand artifacts match its run record")


def test_measurement_at_another_code_revision_fails(world):
    path = world.inputs / "tier1_limits.json"
    original = path.read_text()
    path.write_text(original + "\n")
    changed = world.commit("change tier 1 limits")
    path.write_text(original)
    world.commit("restore them")
    world.edit_tier2(1, lambda d: d["source"].update(commit=changed))
    _fails(world, "held-out tier 2 measurement provenance")


def test_measurement_from_a_dirty_checkout_fails(world):
    _edit_json(
        Path(world.sets[0]["tier1"]),
        lambda d: d["source"].update(dirty_paths=[" M scripts/qualification/x.py"]),
    )
    _fails(world, "development tier 1 measurement provenance")


def test_run_record_with_an_exception_fails(world):
    world.edit_record("D7", "off", lambda r: r.update(exception="RuntimeError: x"))
    _fails(world, "D7 off run record")


def test_run_into_an_existing_directory_fails(world):
    world.edit_record("H4", "cand", lambda r: r["preexisting"].update(output=True))
    _fails(world, "H4 cand fresh output")


def test_tier1_capture_into_an_existing_directory_fails(world):
    path = Path(world.sets[0]["tier1_off"]) / "D2.run.json"
    _edit_json(path, lambda r: r["preexisting"].update(tier1_capture=True))
    _fails(world, "D2 tier 1 off fresh output")


def test_h3_extra_seed_inputs_must_match_their_runs(tmp_path):
    zn = _h3_zn([(205, 6, 6), (206, 0, 0)], [(221, 3, 3), (222, 3, 3)])
    world = World(tmp_path, h3_extra=[221, 222], h3_zn=zn)
    path = Path(world.sets[-1]["cand"]) / "H3.run.json"
    rel = f"{H3_JOB}/seed_222/predictions/{H3_JOB}_sample_3.cif"
    _edit_json(
        path, lambda r: r["outputs"]["prediction_sha256"].update({rel: "0" * 64})
    )
    verdict = _fails(world, f"H3 {H3_JOB} tier 2 cand inputs match its runs")
    assert _failed(verdict, "H3 extra seeds 222 cand artifacts match its run record")


# Measurement and run tools


def test_draw_gives_each_row_its_own_seed_pairs():
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    log = {
        "draws": [
            {
                "id": "d",
                "seeds": {"H1": [201, 202], "H3": [205, 206]},
                "zn_extra": {"H3": [221]},
            }
        ]
    }
    pairs, extra = t2.pairs_from_draw(manifest, log, "d", ["D1", "H1", "H3"])
    assert pairs == {
        "D1": [(101, 101), (102, 102)],
        "H1": [(201, 201), (202, 202)],
        "H3": [(205, 205), (206, 206)],
    }
    assert extra == {"H3": [221]}
    with pytest.raises(SystemExit):
        t2.pairs_from_draw(manifest, log, "d", ["H2"])
    with pytest.raises(SystemExit):
        t2.pairs_from_draw(manifest, log, "missing", ["D1"])


def test_measure_with_a_draw_records_pairs_inputs_and_source(tmp_path, monkeypatch):
    log = {"draws": [{"id": "d", "seeds": {"H1": [201, 202], "H8": [215, 216]}}]}
    (tmp_path / "log.json").write_text(json.dumps(log))
    runs = {tag: tmp_path / tag for tag in ("off", "cand")}
    jobs = {"H1": "7r6r", "H8": "h8_dna_duplex"}
    for tag, run in runs.items():
        for row, seeds in log["draws"][0]["seeds"].items():
            write_artifacts(run, row, [jobs[row]], seeds)
    calls = []

    def measure_row(row, off, cand, pairs, extra):
        calls.append((row["id"], pairs, extra))
        return [{"id": row["id"], "job": jobs[row["id"]], "pairs": []}]

    monkeypatch.setattr(t2, "measure_row", measure_row)
    out = tmp_path / "measure.json"
    argv = ["tier2_gates.py", "measure", str(runs["off"]), str(runs["cand"])]
    argv += ["--rows", "H1,H8", "--draw", "d", "--log", str(tmp_path / "log.json")]
    monkeypatch.setattr(sys, "argv", argv + ["--out", str(out)])
    assert t2.main() == 0
    assert calls == [
        ("H1", [(201, 201), (202, 202)], []),
        ("H8", [(215, 215), (216, 216)], []),
    ]
    data = json.loads(out.read_text())
    assert data["draw"] == "d" and set(data["source"]) == {
        "commit",
        "dirty_paths",
        "imports",
    }
    h8 = _entry_of(data, "H8")
    assert h8["seed_pairs"] == [[215, 215], [216, 216]]
    assert h8["inputs"]["off"] == _digests(["h8_dna_duplex"], [215, 216])


def test_run_rows_refuses_existing_paths(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_rows.subprocess, "call", lambda cmd, **kw: calls.append(cmd) or 0
    )
    out = tmp_path / "runs"
    (out / "D1").mkdir(parents=True)
    (out / "D7.run.json").write_text("{}")
    monkeypatch.setattr(
        sys, "argv", ["run_rows.py", "--rows", "D1,D7,D8", "--out", str(out)]
    )
    assert run_rows.main() == 1
    assert len(calls) == 1 and str(out / "D8") in calls[0]


D1_INPUT = "examples/cofold/protein_200.json"


def _wrapper(monkeypatch):
    pytest.importorskip("torch")
    import run_opendde
    from runner import cli

    monkeypatch.setattr(run_opendde, "_install", lambda *a: None)
    monkeypatch.setattr(run_opendde, "fingerprint", lambda: {"commit": "c"})
    return run_opendde, cli.opendde_cli


def test_run_record_reports_an_uncaught_exception(tmp_path, monkeypatch):
    run_opendde, opendde_cli = _wrapper(monkeypatch)

    def fail(**kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(opendde_cli, "main", fail)
    record = tmp_path / "D1.run.json"
    argv = ["run_opendde.py", "--run-record", str(record), "--", "pred", "-i", D1_INPUT]
    argv += ["-o", str(tmp_path / "D1"), "--use_msa", "false"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError):
        run_opendde.main()
    rec = json.loads(record.read_text())
    assert rec["exit_code"] == 1 and rec["exception"] == "RuntimeError: boom"
    assert rec["preexisting"] == {"output": False, "tier1_capture": None}


def test_run_record_lists_outputs_and_existing_directories(tmp_path, monkeypatch):
    run_opendde, opendde_cli = _wrapper(monkeypatch)
    out = tmp_path / "D1"
    capture = tmp_path / "cap" / "D1"
    capture.mkdir(parents=True)

    def run(**kw):
        write_artifacts(tmp_path, "D1", ["job"], [101])
        (capture / "job" / "seed_101").mkdir(parents=True)
        (capture / "job" / "seed_101" / "tier1.pt").write_bytes(b"t1")

    monkeypatch.setattr(opendde_cli, "main", run)
    record = tmp_path / "D1.run.json"
    argv = ["run_opendde.py", "--run-record", str(record)]
    argv += ["--tier1-capture", str(capture), "--", "pred", "-i", D1_INPUT]
    monkeypatch.setattr(sys, "argv", argv + ["-o", str(out), "--use_msa", "false"])
    assert run_opendde.main() == 0
    rec = json.loads(record.read_text())
    assert rec["exit_code"] == 0 and rec["exception"] is None
    assert rec["preexisting"] == {"output": False, "tier1_capture": True}
    assert rec["outputs"]["prediction_sha256"] == _digests(["job"], [101])
    assert rec["outputs"]["tier1_sha256"] == {
        "job/seed_101/tier1.pt": hashlib.sha256(b"t1").hexdigest()
    }


def test_tier1_compare_records_capture_digests(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import tier1_compare

    monkeypatch.setattr(tier1_compare, "compare_capture", lambda a, b: {})
    for root, data in (("ref", b"a"), ("other", b"b")):
        path = tmp_path / root / "D1" / "job" / "seed_101" / "tier1.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
    out = tmp_path / "t1.json"
    argv = ["tier1_compare.py", str(tmp_path / "ref"), str(tmp_path / "other")]
    monkeypatch.setattr(sys, "argv", argv + ["--out", str(out)])
    assert tier1_compare.main() == 0
    data = json.loads(out.read_text())
    entry = data["entries"][0]
    assert entry["ref_sha256"] == hashlib.sha256(b"a").hexdigest()
    assert entry["other_sha256"] == hashlib.sha256(b"b").hexdigest()
    assert set(data["source"]) == {"commit", "dirty_paths", "imports"}


# Duplicate and unexpected measurement entries


def _check(verdict, name):
    return next(c for c in verdict["checks"] if c["check"] == name)


@pytest.mark.parametrize("failing_first", [True, False])
def test_duplicate_tier2_entry_fails_in_either_order(world, failing_first):
    """A passing duplicate must not replace a failing measurement, nor the reverse."""

    def add(d):
        entry = _entry_of(d, "D4", "7wux")
        failing = copy.deepcopy(entry)
        failing["min_pair_cand"] = 0.0
        d["entries"].remove(entry)
        pair = [failing, entry] if failing_first else [entry, failing]
        d["entries"] += pair

    world.edit_tier2(0, add)
    verdict = _fails(world, "development tier 2 entries unique")
    gates = _check(verdict, "D4 7wux tier 2")
    assert not gates["ok"] and gates["detail"] == ["unique"]


@pytest.mark.parametrize("failing_first", [True, False])
def test_duplicate_tier1_entry_fails_in_either_order(world, failing_first):
    path = Path(world.sets[0]["tier1"])

    def add(d):
        entry = _entry_of(d, "D7")
        failing = copy.deepcopy(entry)
        failing["z_trunk"]["rel_fro"] = 0.2
        d["entries"].remove(entry)
        d["entries"] += [failing, entry] if failing_first else [entry, failing]

    _edit_json(path, add)
    _fails(world, "development tier 1 entries unique")


def test_tier2_entry_outside_the_set_fails(world):
    def add(d):
        held = json.loads(Path(world.sets[1]["tier2"]).read_text())
        d["entries"].append(_entry_of(held, "H1"))

    world.edit_tier2(0, add)
    verdict = _fails(world, "development tier 2 entries expected")
    assert "H1" in str(_check(verdict, "development tier 2 entries expected"))


def test_tier2_entry_for_a_job_the_row_lacks_fails(world):
    def add(d):
        entry = copy.deepcopy(_entry_of(d, "D4", "7wux"))
        entry["job"] = "not_in_input"
        d["entries"].append(entry)

    world.edit_tier2(0, add)
    verdict = _fails(world, "development tier 2 entries expected")
    assert not _check(verdict, "D4 not_in_input tier 2")["ok"]
    assert _failed(verdict, "D4 not_in_input tier 2 off inputs match its runs")


def test_tier1_entry_for_an_unexpected_seed_fails(world):
    path = Path(world.sets[0]["tier1"])

    def add(d):
        entry = copy.deepcopy(_entry_of(d, "D7"))
        entry["seed"] = 103
        d["entries"].append(entry)

    _edit_json(path, add)
    verdict = _fails(world, "development tier 1 entries expected")
    assert verdict["failed"] == [
        "development tier 1 entries expected",
        "development evidence matches the rehearsal receipt",
    ]


def test_measure_refuses_a_row_given_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(t2, "measure_row", lambda *a: pytest.fail("measured"))
    argv = ["tier2_gates.py", "measure", str(tmp_path / "off"), str(tmp_path / "c")]
    argv += ["--rows", "D1,D7,D1", "--out", str(tmp_path / "m.json")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        t2.main()
    assert exc.value.code == 2 and not (tmp_path / "m.json").exists()


def test_tier1_compare_rows_excludes_other_rows(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import tier1_compare

    monkeypatch.setattr(tier1_compare, "compare_capture", lambda a, b: {})
    for root in ("ref", "other"):
        for row in ("D1", "D7", "H1"):
            path = tmp_path / root / row / "job" / "seed_101" / "tier1.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(row.encode())
    out = tmp_path / "t1.json"
    argv = ["tier1_compare.py", str(tmp_path / "ref"), str(tmp_path / "other")]
    monkeypatch.setattr(sys, "argv", argv + ["--out", str(out), "--rows", "D1,D7"])
    assert tier1_compare.main() == 0
    rows = [e["id"] for e in json.loads(out.read_text())["entries"]]
    assert rows == ["D1", "D7"]


# Development-only rehearsal


def _rehearsal(world, **changes):
    evidence = {
        "gpu": "sm_120",
        "route": "dev-rehearsal",
        "candidate": "fast-test",
        "code_commit": world.code,
        "candidate_args": list(world.cand_args),
        "sets": copy.deepcopy(world.sets[:1]),
    }
    evidence.update(changes)
    return evidence


def test_rehearsal_passes_with_exit_4_never_0(tmp_path):
    world = World(tmp_path, frozen=True)
    verdict = world.run(_rehearsal(world))
    assert verdict["failed"] == []
    assert verdict["exit_code"] == accept.REHEARSAL != accept.FULL
    assert verdict["status"] == "rehearsal passed (not an acceptance)"
    assert verdict["rehearsal"] and verdict["held_out_draw"] is None
    assert verdict["code_commit"] == world.code and verdict["zn_claim"] == "qualified"
    names = [c["check"] for c in verdict["checks"]]
    for name in (
        "contract tag present",
        "qualification files unchanged since the contract tag",
        "D2 cand command",
        "D5 random inputs match off (raw)",
        "D7 repeat is a separate run",
        "D14 6e4y tier 2",
    ):
        assert name in names, name
    assert not [n for n in names if n.startswith("H") or "seeds logged" in n]
    assert "held-out draw logged" not in names


def test_rehearsal_without_a_freeze_fails(world):
    verdict = _fails(world, "rehearsal under a declared freeze", _rehearsal(world))
    assert verdict["failed"] == ["rehearsal under a declared freeze"]


def _drop_record(world):
    (world.run_dir("D2", "cand") / "D2.run.json").unlink()


def _off_command_for_cand(world):
    world.write_run(world.run_dir("D2", "cand"), "D2", [101, 102], world.code, "off")


def _other_random_inputs(world):
    run = world.run_dir("D5", "cand")
    rec = json.loads((run / "D5.run.json").read_text())
    write_trace(run, "D5", rec, world.rollouts("D5", [101, 102]), salt="-x")


@pytest.mark.parametrize(
    "break_it, fragment",
    [
        (_drop_record, "D2 cand run record"),
        (_off_command_for_cand, "D2 cand command"),
        (_other_random_inputs, "D5 random inputs match off"),
    ],
)
def test_rehearsal_failures(tmp_path, break_it, fragment):
    world = World(tmp_path, frozen=True)
    break_it(world)
    _fails(world, fragment, _rehearsal(world))


def test_rehearsal_rejects_held_out_evidence(tmp_path):
    world = World(tmp_path, frozen=True)
    _fails(
        world,
        "rehearsal names no held-out draw",
        _rehearsal(world, held_out_draw="sm_120-1"),
    )
    _fails(world, "no undeclared rows", _rehearsal(world, sets=world.sets))


def test_rehearsal_requires_declared_candidate_arguments(tmp_path):
    world = World(tmp_path, frozen=True)
    name = "rehearsal declares the code commit and candidate arguments"
    _fails(world, name, _rehearsal(world, candidate_args=[]))
    _fails(world, name, _rehearsal(world, code_commit=world.code[:12]))


# Rehearsal receipt and the held-out draw


def test_draw_without_a_receipt_fails(tmp_path):
    world = World(tmp_path, receipt=None)
    verdict = _fails(world, "draw names a logged rehearsal receipt")
    assert verdict["failed"] == ["draw names a logged rehearsal receipt"]


@pytest.mark.parametrize("when", ["with", "after"])
def test_receipt_not_committed_before_the_draw_fails(tmp_path, when):
    world = World(tmp_path, receipt=when)
    verdict = _fails(world, "rehearsal receipt committed before the draw")
    assert verdict["failed"] == ["rehearsal receipt committed before the draw"]


@pytest.mark.parametrize(
    "changes, fragment",
    [
        ({"status": "fail", "exit_code": 1}, "rehearsal receipt passed"),
        ({"code_commit": "f" * 40}, "rehearsal receipt matches the draw"),
        ({"candidate_args": ["--optimization_mode", "big"]}, "matches the draw"),
        ({"candidate_args": ["fast", "--optimization_mode"]}, "matches the draw"),
        ({"evidence_sha256": _sha("other")}, "development evidence matches"),
        ({"contract_tag": "qual-test-contract-9"}, "names the contract tag"),
        ({"gate_commit": "f" * 40}, "records a gate at the contract tag"),
        ({"gate_approved_tag": "qual-test-contract-9"}, "records a gate"),
        ({"gate_sha256": None}, "records a gate at the contract tag"),
    ],
)
def test_mismatched_receipt_fails(tmp_path, changes, fragment):
    world = World(tmp_path, receipt_changes=changes)
    verdict = _fails(world, fragment)
    assert len(verdict["failed"]) == 1, verdict["failed"]


def _edit_cand_record(world):
    world.edit_record("D3", "cand", lambda r: r.update(note="rewritten"))


def _edit_trace(world):
    path = world.run_dir("D3", "off") / "D3.trace.jsonl"
    path.write_text(path.read_text() + "\n")


def _reformat_measurement(world):
    path = Path(world.sets[0]["tier2"])
    path.write_text(json.dumps(json.loads(path.read_text()), indent=2))


def _edit_repeat_record(world):
    world.edit_record("D3", "repeat", lambda r: r.update(note="rerun"))


@pytest.mark.parametrize(
    "change",
    [_edit_cand_record, _edit_trace, _reformat_measurement, _edit_repeat_record],
)
def test_changed_development_evidence_fails(world, change):
    change(world)
    _fails(world, "development evidence matches the rehearsal receipt")


def test_changed_repeat_artifact_fails(world):
    run = world.run_dir("D3", "repeat") / "D3"
    path = sorted(run.rglob("*.cif"))[0]
    path.write_text(path.read_text() + "x")
    _fails(world, "D3 repeat artifacts match its run record")


def test_edited_or_removed_receipt_fails(world):
    world.log["receipts"][0]["verdict_sha256"] = _sha("edited")
    world.commit_log("edit the receipt")
    _fails(world, "held-out log history only appends")
    world.log["receipts"] = []
    world.commit_log("remove the receipt")
    verdict = _fails(world, "held-out log history only appends")
    assert _failed(verdict, "draw names a logged rehearsal receipt")


def _second_draw(world, receipt_id, code):
    draw = {
        "id": "sm_120-2",
        "gpu": "sm_120",
        "candidate": "fast-test",
        "code_commit": code,
        "candidate_args": list(world.cand_args),
        "receipt": receipt_id,
        "seeds": {"H1": [280, 281]},
        "zn_extra": {},
    }
    world.log["draws"].append(draw)
    world.commit_log("second draw")


def test_second_draw_naming_a_used_receipt_fails(world):
    _second_draw(world, "rehearsal-1", "e" * 40)
    verdict = _fails(world, "each rehearsal receipt named by at most one draw")
    assert not _failed(verdict, "each candidate drawn at most once")


def test_same_candidate_drawn_again_fails(world):
    world.log["receipts"].append({**world.receipt, "id": "rehearsal-2"})
    world.commit_log("second receipt")
    _second_draw(world, "rehearsal-2", world.code)
    verdict = _fails(world, "each candidate drawn at most once")
    assert not _failed(verdict, "named by at most one draw")


def test_superseded_contract_tag_fails(tmp_path):
    world = World(tmp_path, frozen=True)
    newer = CONTRACT_TAG.replace("-1", "-2")
    _git(world.repo, "tag", "-a", newer, "-m", "amended", world.code)
    verdict = _fails(world, "contract tag is the newest")
    assert verdict["failed"] == ["contract tag is the newest"]


def test_rehearsal_digest_and_receipt_match_the_log_entry(tmp_path):
    """The receipt a passing rehearsal writes is the one acceptance checks."""
    world = World(tmp_path, frozen=True)
    verdict = world.run(_rehearsal(world))
    assert verdict["exit_code"] == accept.REHEARSAL
    assert verdict["evidence_sha256"] == world.receipt["evidence_sha256"]
    gate = {"ok": True, "gate_commit": world.contract, "approved_tag": CONTRACT_TAG}
    gate_text = json.dumps(gate)
    receipt = accept.make_receipt(verdict, "rehearsal-1", "verdict text", gate_text)
    assert receipt["verdict_sha256"] == _sha("verdict text")
    assert receipt["gate_sha256"] == _sha(gate_text)
    hashes = ("verdict_sha256", "gate_sha256")
    keep = {k: v for k, v in world.receipt.items() if k not in hashes}
    assert {k: v for k, v in receipt.items() if k not in hashes} == keep


def test_receipt_written_only_for_a_passing_rehearsal(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDDE_QUAL_DATA", str(tmp_path))
    (tmp_path / "e.json").write_text("{}")
    passing, failing = tmp_path / "gate-ok.json", tmp_path / "gate-fail.json"
    passing.write_text(
        json.dumps({"ok": True, "gate_commit": "c", "approved_tag": "t"})
    )
    failing.write_text(json.dumps({"ok": False}))
    cases = [
        (accept.REHEARSAL, passing, True),
        (accept.REHEARSAL, failing, False),
        (accept.REHEARSAL, None, False),
        (accept.FAIL, passing, False),
    ]
    for n, (code, gate, written) in enumerate(cases):
        verdict = {"status": "s", "exit_code": code, "failed": [], "zn_claim": "x"}
        monkeypatch.setattr(accept, "accept", lambda e, v=verdict: dict(v))
        out, receipt = tmp_path / f"v{n}.json", tmp_path / f"r{n}.json"
        argv = ["accept.py", str(tmp_path / "e.json"), "--out", str(out)]
        argv += ["--receipt", "r", str(receipt)]
        argv += ["--gate-report", str(gate)] if gate else []
        monkeypatch.setattr(sys, "argv", argv)
        assert accept.main() == code
        assert receipt.exists() == written
        if written:
            entry = json.loads(receipt.read_text())
            assert (
                entry["verdict_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
            )
            assert entry["gate_sha256"] == hashlib.sha256(gate.read_bytes()).hexdigest()
            assert (entry["gate_commit"], entry["gate_approved_tag"]) == ("c", "t")


# Model packages loaded from the checkout, and unreadable measurements


def test_run_loading_another_checkouts_code_fails(world):
    """An editable install resolving opendde to another clone."""
    other = {"opendde": "/home/u/OpenDDE/opendde", "runner": "/home/u/OpenDDE/runner"}
    world.edit_record("D2", "cand", lambda r: r["fingerprint"].update(imports=other))
    verdict = _fails(world, "D2 cand imports from the checkout")
    assert not _failed(verdict, "D2 cand provenance")


def test_run_without_import_paths_fails(world):
    world.edit_record("D4", "off", lambda r: r["fingerprint"].pop("imports"))
    _fails(world, "D4 off imports from the checkout")


def test_measurement_loading_another_checkouts_code_fails(world):
    world.edit_tier2(
        0, lambda d: d["source"]["imports"].update(opendde="/elsewhere/opendde")
    )
    verdict = _fails(world, "development tier 2 measurement imports from the checkout")
    assert not _failed(verdict, "development tier 2 measurement provenance")


def test_import_paths_follow_sys_path_before_an_installed_package(
    tmp_path, monkeypatch
):
    """PYTHONPATH (here, sys.path) wins over an editable-install finder."""
    import provenance

    for pkg in provenance.PACKAGES:
        (tmp_path / pkg).mkdir()
        (tmp_path / pkg / "__init__.py").write_text("")
        monkeypatch.delitem(sys.modules, pkg, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert provenance.import_paths() == {
        pkg: str((tmp_path / pkg).resolve()) for pkg in provenance.PACKAGES
    }
    assert provenance.git_state()["imports"] == provenance.import_paths()


def test_run_fingerprint_records_import_paths(monkeypatch):
    pytest.importorskip("torch")
    import provenance
    import run_opendde

    assert run_opendde.fingerprint()["imports"] == provenance.import_paths()


@pytest.mark.parametrize("tier", ["tier1", "tier2"])
def test_missing_measurement_file_fails_without_raising(world, tier):
    evidence = world.evidence()
    evidence["sets"][0][tier] = str(world.tmp / "absent.json")
    tag = "tier 1" if tier == "tier1" else "tier 2"
    _fails(world, f"development {tag} measurement readable", evidence)


def test_unreadable_measurement_file_fails_without_raising(world):
    Path(world.sets[1]["tier2"]).write_text("{not json")
    _fails(world, "held-out tier 2 measurement readable")


# Held-out seed range (contract-6: 201-999)


@pytest.mark.parametrize("seed", [200, 1000])
def test_held_out_seed_outside_the_range_fails(world, seed):
    world.log["draws"].append(
        {"id": "sm_120-0", "gpu": "sm_120", "candidate": "old", "seeds": {"H1": [seed]}}
    )
    world.commit_log()
    _fails(world, "held-out seeds in 201-999")


def test_held_out_seed_at_the_new_upper_bound_is_in_range(world):
    world.log["draws"].append(
        {"id": "sm_120-0", "gpu": "sm_120", "candidate": "old", "seeds": {"H1": [999]}}
    )
    world.commit_log()
    verdict = world.run()
    assert not _failed(verdict, "held-out seeds in")


# Input provenance (contract-6)


def _edit_prov(world, row, kind, fn):
    world.edit_record(row, kind, lambda r: fn(r["provenance"]))


def test_complete_pass_checks_provenance_of_every_run(world):
    verdict = world.run()
    assert verdict["failed"] == []
    names = {c["check"] for c in verdict["checks"]}
    for check in (
        "D4 off input read as frozen",
        "H10 cand referenced files stable and frozen",
        "D5 off template mmCIF files stable and frozen",
        "D5 off template cache new at the parser read",
        "D5 off requested Kalign is the frozen binary",
        "H2 tier 1 off runtime assets match the frozen reference",
        "H9 repeat output holds the declared seeds",
        "every run used one path and digest per runtime asset",
        "runtime-asset reference frozen for this GPU",
    ):
        assert check in names, check


def test_missing_or_refused_provenance_fails(world):
    world.edit_record("D1", "off", lambda r: r.pop("provenance"))
    _fails(world, "D1 off input provenance recorded")
    world.edit_record(
        "D2", "cand", lambda r: r.update(provenance={"refused": "search"})
    )
    _fails(world, "D2 cand input provenance recorded")


def test_json_changed_before_the_parser_read_fails(world):
    """An unchanged input must be read with its source digest."""
    _edit_prov(
        world, "H1", "cand", lambda p: p["parser_reads"][0].update(sha256="0" * 64)
    )
    _fails(world, "H1 cand input read as frozen")


def test_legacy_msa_input_read_without_its_conversion_fails(world):
    """Run from another working directory, the conversion drops the MSA
    paths; the parser then reads neither the source nor the frozen
    effective bytes."""
    _edit_prov(
        world, "H2", "off", lambda p: p["parser_reads"][0].update(sha256=_sha("no msa"))
    )
    _fails(world, "H2 off input read as frozen")
    _edit_prov(
        world,
        "D4",
        "cand",
        lambda p: p["parser_reads"][0].update(
            sha256=DIGESTS["scripts/qualification/inputs/d4_7wux.json"]
        ),
    )
    _fails(world, "D4 cand input read as frozen")


def test_source_digest_not_frozen_fails(world):
    _edit_prov(world, "D7", "repeat", lambda p: p["source"].update(sha256="0" * 64))
    _fails(world, "D7 repeat input read as frozen")


def test_referenced_file_changed_during_the_run_fails(world):
    def edit(p):
        key = next(iter(p["referenced"]))
        p["referenced"][key]["after"] = "0" * 64

    _edit_prov(world, "D3", "off", edit)
    _fails(world, "D3 off referenced files stable and frozen")


def test_referenced_file_missing_from_the_record_fails(world):
    _edit_prov(world, "D13", "cand", lambda p: p["referenced"].popitem())
    _fails(world, "D13 cand referenced files stable and frozen")


def test_asset_override_without_a_frozen_reference_fails(world):
    def edit(p):
        p["assets"]["checkpoint"].update(
            resolved="/data/shared/models/opendde/checkpoint/opendde.pt"
        )

    _edit_prov(world, "D6", "off", edit)
    verdict = _fails(world, "D6 off runtime assets match the frozen reference")
    assert _failed(verdict, "every run used one path and digest per runtime asset")


def test_runtime_asset_changed_between_runs_fails(world):
    """Same path, size and modification time, different bytes: only the
    content digest shows it."""
    _edit_prov(
        world,
        "H5",
        "repeat",
        lambda p: p["assets"]["ccd_components_file"].update(sha256="0" * 64),
    )
    verdict = _fails(world, "H5 repeat runtime assets match the frozen reference")
    assert _failed(verdict, "every run used one path and digest per runtime asset")


def test_missing_runtime_asset_reference_fails(world):
    (world.inputs / "runtime_assets.json").write_text(json.dumps({"gpus": {}}))
    verdict = _fails(world, "runtime-asset reference frozen for this GPU")
    assert _failed(verdict, "D1 off runtime assets match the frozen reference")


def test_actual_seed_mismatch_fails(world):
    job = world.jobs("H6")[0]
    _edit_prov(
        world, "H6", "cand", lambda p: p["seeds"].update({job: [p["seeds"][job][0]]})
    )
    _fails(world, "H6 cand output holds the declared seeds")


def test_prepopulated_template_cache_fails(world):
    _edit_prov(
        world, "D5", "cand", lambda p: p["template_cache"].update(absent_at_parse=False)
    )
    _fails(world, "D5 cand template cache new at the parser read")


def test_template_mmcif_changed_during_the_run_fails(world):
    def edit(p):
        key = next(iter(p["template_mmcif"]["after"]))
        p["template_mmcif"]["after"][key] = "0" * 64

    _edit_prov(world, "D5", "off", edit)
    _fails(world, "D5 off template mmCIF files stable and frozen")


def test_kalign_other_than_the_frozen_binary_fails(world):
    _edit_prov(
        world,
        "D5",
        "repeat",
        lambda p: p["kalign"].update(requested_resolved="/usr/bin/kalign"),
    )
    _fails(world, "D5 repeat requested Kalign is the frozen binary")


def test_changed_effective_digest_reference_fails(world):
    path = world.inputs / "effective_inputs.json"
    _edit_json(
        path,
        lambda d: d["scripts/qualification/inputs/d2_7r6r.json"].update(
            effective_sha256="0" * 64
        ),
    )
    verdict = _fails(world, "effective input digests")
    assert _failed(verdict, "D2 off input read as frozen")
