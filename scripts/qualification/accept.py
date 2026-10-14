#!/usr/bin/env python
"""Decide full acceptance of a candidate on one GPU (qualification_protocol.md, E5).

Usage:
    python scripts/qualification/accept.py EVIDENCE.json [--out VERDICT.json]
        [--receipt ID RECEIPT.json]

Run from the root of the acceptance checkout on the GPU host, with
``OPENDDE_QUAL_DATA`` set. This command, not ``tier2_gates.py evaluate`` or
any other diagnostic, decides acceptance. The rows a GPU and route must cover
and the run recipe come from ``inputs/acceptance.json``, never from the
command line or the evidence file.

EVIDENCE.json names the GPU, route, candidate and held-out draw, and lists run
sets. Every run directory is a ``run_rows.py --out`` directory, whose
``<row>.run.json`` records the run. A full set has ``rows`` and the runs
``off`` and ``cand`` (``--draw-trace``), ``cand_repeat`` (a second candidate
run, traced or not), ``tier1_off`` and ``tier1_cand`` (tier 1 capture runs),
``tier1`` (a ``tier1_compare.py`` output of those captures) and ``tier2`` (a
``tier2_gates.py measure`` output of ``off`` and ``cand``). A set with
``"zn_only": true`` holds H3 extra Zn seeds: ``rows``, ``seeds``, ``off`` and
``cand``, traces only.

A route with ``"rehearsal": true`` (``dev-rehearsal``) rehearses a candidate
on the development rows before any held-out draw. Its evidence names
``code_commit`` and ``candidate_args`` instead of ``held_out_draw``, which it
must not name. Only the checks that need a draw are skipped (the current
log's draw entry, seeds logged at each run's commit, H3's extra seeds); every
other check below runs unchanged, and a declared freeze is required. With
``--receipt ID PATH --gate-report GATE.json`` a passing rehearsal writes its
held-out log receipt (candidate, contract tag, code commit, ordered
arguments, the evidence digest, the verdict's sha256, status and exit code,
and the passing gate report's sha256, commit and approved tag), to be appended to ``receipts`` in
``held_out_log.json`` and committed before the draw that names it. The
development sets of the later acceptance must be declared exactly as in the
rehearsal, since their digest is compared with the receipt's.

Checks:
- every required row appears in exactly one full set, and no other row does;
- the frozen inputs match their digests, repository and external, and the
  expected effective digests of inputs ``pred`` converts from the legacy MSA
  format (``effective_inputs.json``);
- the held-out log is well formed: seeds from 201-999, each used once in the
  whole log, two per held-out row, extra Zn seeds only where ``zn_rule``
  allows them; the draw records the candidate's code commit and its declared
  arguments; receipt ids are unique, a receipt is named by at most one draw,
  and a candidate (code commit and ordered arguments) is drawn at most once;
- the draw names a passing rehearsal receipt of the same GPU, candidate, code
  commit and ordered arguments, recording this contract tag and a passing
  gate run at the tag's commit, present in the log before the commit that
  first adds the draw, whose evidence digest equals that of the development
  sets being accepted (``evidence_digest``: set declarations, every run
  record and trace, both measurement files);
- Git: the checkout is clean and differs from the draw's code commit only in
  the held-out log, and the log's history only ever appends (receipts and
  draws);
- per run, from its run record: exit status 0 and no exception; output and
  capture directories that did not exist before the run; a commit that differs from the
  code commit only in the held-out log, is an ancestor of the checkout and,
  for a held-out row, already logs the run's seeds; no uncommitted changes;
  this GPU; the command line ``run_rows.build_command`` gives for the row,
  seeds and recipe (``off`` exactly the stock command, candidate runs the
  stock command plus the declared arguments); the recipe environment; the
  wrapper options of its kind; and one set of software versions across runs;
- per run, from its input provenance (contract-6, ``run_opendde.py``): the
  source JSON has its frozen digest; the inference parser read it once, with
  the source digest or, for a legacy-MSA input, the frozen effective digest;
  every MSA, RNA-MSA and template file the effective JSON names had its
  frozen digest before and after the run; a template run's mmCIF files had
  their frozen digests before and after, its template cache was new at the
  parser read, and it requested the frozen Kalign binary; the selected
  runtime assets (checkpoint, both CCD files and, for template runs, the
  release-date and obsolete-PDB files and the resolved Kalign) have the
  paths and digests of this GPU's frozen reference (``runtime_assets.json``),
  with one path and digest per asset across all runs; and each job's output
  layout holds exactly the declared seeds;
- per traced pair: traces belong to their run records; rollouts are exactly
  the input's jobs times the seeds, with the recipe's sample count, step
  count, chunk size and guidance setting; every expected draw is recorded on
  both sides; ``off`` and candidate rollout settings and raw draw hashes match;
- per row: ``off``, candidate and repeat runs hold exactly the expected
  prediction artifacts, byte for byte the ones their run records list; the repeat is a separate run and byte-identical to the
  candidate; development ``off`` artifacts match the committed step 0
  baseline digests;
- per row: the tier 1 comparison measured the declared tier 1 captures (each
  entry's capture digests equal those the capturing runs recorded, for every
  job and seed), the candidate capture replayed the declared ``off`` capture,
  and every job and seed passes the approved tier 1 limits;
- measurements: each tier 1 and tier 2 measurement was made at a commit that
  passes the run-commit rule, with no uncommitted changes; each set's tier 2
  file has exactly one entry per row and job, and its tier 1 file exactly one
  per tier 1 row, job and seed, with no other entries;
- tier 2: every entry measured exactly the same-seed pairs of the row's
  logged seeds and its logged extra Zn seeds, and the artifact digests it
  read equal those its runs recorded; every job of every required row passes
  ``tier2_gates.evaluate`` with the expected seeds, H3's extra seeds are covered by their own runs and
  measured from them in log order, and the aggregate Zn claim is
  ``qualified``;
- once ``acceptance.json`` declares a freeze: the limits files match its
  hashes, and its ``contract_tag`` exists, is the newest tag of its numbered
  family in the checkout, is an ancestor of the draw's code
  commit, and the qualification directory (code, constants, inputs, limits,
  recipe, baseline, wrapper) is unchanged from the tag apart from the
  held-out log. This runs at the candidate's commit, so it catches accidental
  change, not deliberate tampering.

Run this command through ``qual_gate.py --accept`` from a checkout of the
approved contract tag: the gate checks that this checkout's qualification
files are the approved tag's, which this command cannot establish about
itself, and a full qualification claim is the gate's combined record, not
this command's exit status alone.

Exit status: 0 full acceptance; 1 failure; 2 every check passes except the
Zn claim, which is reported as a partial result without Zn and is not an
acceptance (an H3 unresolved at its seed cap is a held-out failure); 3 every
check passes but no freeze is declared yet (provisional); 4 a rehearsal in
which every check passes (not an acceptance). A rehearsal never exits 0.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import compare_outputs
import freeze_inputs
import provenance
import run_rows
import tier2_gates
import trace_tools

INPUTS = Path("scripts/qualification/inputs")
HELD_OUT_RANGE = (201, 999)
HELD_OUT_SEEDS_PER_ROW = 2
FULL, FAIL, PARTIAL, PROVISIONAL, REHEARSAL = 0, 1, 2, 3, 4
CANDIDATE_KINDS = ("cand", "repeat", "tier1 cand")
REHEARSAL_STATUS = "rehearsal passed (not an acceptance)"
EVIDENCE_RUNS = ("off", "cand", "cand_repeat", "tier1_off", "tier1_cand")
HEX = set("0123456789abcdef")
BASE_ASSETS = ("checkpoint", "ccd_components_file", "ccd_components_rdkit_mol_file")
TEMPLATE_ASSETS = ("release_dates_path", "obsolete_pdbs_path", "kalign_binary_path")


class Checks:
    def __init__(self):
        self.items: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: Any = None) -> bool:
        self.items.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    def failed(self) -> list[dict[str, Any]]:
        return [c for c in self.items if not c["ok"]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flag(args: list[str], name: str):
    return args[args.index(name) + 1] if name in args[:-1] else None


def _last(args: list[str], name: str):
    """The value of the last occurrence of ``name`` (later flags override)."""
    found = [i for i, a in enumerate(args[:-1]) if a == name]
    return args[found[-1] + 1] if found else None


def _get(entry: dict[str, Any], dotted: str):
    """``section.key``; the key itself may contain dots (``margin_0.1``)."""
    value: Any = entry
    for part in dotted.split(".", 1):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _same_path(a, b) -> bool:
    return a is not None and b is not None and Path(a).resolve() == Path(b).resolve()


class Repo:
    """Read-only Git queries on the checkout that holds the inputs."""

    def __init__(self, path: Path):
        self.path = path

    @staticmethod
    def _run(cwd: Path, *args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(cwd), *args],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    @classmethod
    def find(cls, root: Path):
        top = cls._run(root, "rev-parse", "--show-toplevel")
        return None if top is None else cls(Path(top.strip()))

    def git(self, *args: str) -> str | None:
        return self._run(self.path, *args)

    def commit(self, rev) -> str | None:
        if not isinstance(rev, str) or not rev:
            return None
        out = self.git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
        return out.strip() if out else None

    def changed(self, a: str, b: str) -> list[str] | None:
        out = self.git("diff", "--name-only", a, b)
        return None if out is None else sorted(out.splitlines())

    def is_ancestor(self, a: str, b: str) -> bool:
        return self.git("merge-base", "--is-ancestor", a, b) is not None

    def show(self, rev: str, path: str) -> str | None:
        return self.git("show", f"{rev}:{path}")

    def history(self, rev: str, path: str) -> list[str]:
        out = self.git("rev-list", "--reverse", "--topo-order", rev, "--", path)
        return out.split() if out else []


class Git:
    """The acceptance checkout, the draw's code commit and per-run commit checks."""

    def __init__(self, repo: Repo, code: str, head: str, log_path: str):
        self.repo, self.code, self.head, self.log_path = repo, code, head, log_path
        self.versions: list[tuple[str, Any]] = []
        self._runs: dict[str, tuple[bool, Any]] = {}
        self._logs: dict[str, Any] = {}

    def run_commit(self, rev) -> tuple[bool, Any]:
        """A run's commit must hold the code commit's tree except for the
        held-out log, and lie in the checkout's history."""
        key = json.dumps(rev)
        if key not in self._runs:
            full = self.repo.commit(rev)
            changed = self.repo.changed(self.code, full) if full else None
            ok = (
                changed is not None
                and set(changed) <= {self.log_path}
                and self.repo.is_ancestor(full, self.head)
            )
            self._runs[key] = (ok, {"commit": rev, "changed": changed})
        return self._runs[key]

    def draw_at(self, rev, draw_id):
        if not isinstance(rev, str):
            return None
        if rev not in self._logs:
            try:
                self._logs[rev] = json.loads(self.repo.show(rev, self.log_path) or "")
            except json.JSONDecodeError:
                self._logs[rev] = None
        draws = [
            d
            for d in (self._logs[rev] or {}).get("draws", [])
            if d.get("id") == draw_id
        ]
        return draws[0] if len(draws) == 1 else None


def check_digests(checks: Checks, manifest: Path) -> None:
    for name, current, stored in (
        ("digests", freeze_inputs.digests, "digests.json"),
        ("external digests", freeze_inputs.external_digests, "external_digests.json"),
        (
            "effective input digests",
            freeze_inputs.effective_digests,
            "effective_inputs.json",
        ),
    ):
        try:
            now = current(manifest)
        except SystemExit as error:
            checks.add(name, False, str(error))
            continue
        frozen = json.loads(manifest.with_name(stored).read_text())
        diff = sorted(
            k for k in frozen.keys() | now.keys() if frozen.get(k) != now.get(k)
        )
        checks.add(name, not diff, {"files": len(frozen), "differing": diff})


def _seed_places(log):
    for draw in log.get("draws", []):
        for kind in ("seeds", "zn_extra"):
            for row, seeds in draw.get(kind, {}).items():
                for s in seeds:
                    yield s, f"{draw.get('id')}/{kind}/{row}"


def _declared(code, args) -> bool:
    """A full commit hash and a nonempty list of candidate arguments."""
    return (
        isinstance(code, str)
        and len(code) == 40
        and set(code) <= HEX
        and isinstance(args, list)
        and bool(args)
        and all(isinstance(a, str) for a in args)
    )


def load_provenance_references(checks: Checks, root: Path, gpu) -> dict[str, Any]:
    """The frozen references input provenance is checked against: input and
    external digests, effective digests of converted legacy-MSA inputs, and
    this GPU's runtime-asset paths and digests (``runtime_assets.json``)."""
    refs: dict[str, Any] = {}
    for key, name in (
        ("digests", "digests.json"),
        ("external", "external_digests.json"),
        ("effective", "effective_inputs.json"),
    ):
        try:
            refs[key] = json.loads((root / name).read_text())
        except (OSError, json.JSONDecodeError):
            refs[key] = None
    try:
        assets = json.loads((root / "runtime_assets.json").read_text())
        refs["assets"] = ((assets.get("gpus") or {}).get(gpu) or {}).get("assets")
    except (OSError, json.JSONDecodeError):
        refs["assets"] = None
    checks.add(
        "runtime-asset reference frozen for this GPU",
        isinstance(refs["assets"], dict)
        and all(
            isinstance(refs["assets"].get(n), dict)
            for n in BASE_ASSETS + TEMPLATE_ASSETS
        ),
        gpu,
    )
    return refs


def check_rehearsal(checks: Checks, evidence) -> dict[str, Any]:
    """A rehearsal declares its code commit and candidate arguments and names
    no held-out draw; return them as a draw without seeds."""
    checks.add(
        "rehearsal names no held-out draw",
        evidence.get("held_out_draw") is None,
        evidence.get("held_out_draw"),
    )
    code, args = evidence.get("code_commit"), evidence.get("candidate_args")
    checks.add(
        "rehearsal declares the code commit and candidate arguments",
        _declared(code, args),
        {"code_commit": code, "candidate_args": args},
    )
    return {"code_commit": code, "candidate_args": args, "seeds": {}, "zn_extra": {}}


def check_log(checks: Checks, log, draw_id, gpu, candidate, held_out_rows, limits):
    """Return the evidence's draw from the current log ({} if it is missing)."""
    seen: dict[int, str] = {}
    repeated = []
    for s, where in _seed_places(log):
        if s in seen:
            repeated.append([s, seen[s], where])
        seen[s] = where
    checks.add("held-out seeds used once", not repeated, repeated)
    lo, hi = HELD_OUT_RANGE
    out_of_range = sorted(s for s in seen if not lo <= s <= hi)
    checks.add(f"held-out seeds in {lo}-{hi}", not out_of_range, out_of_range)
    check_log_receipts(checks, log)
    draws = [d for d in log.get("draws", []) if d.get("id") == draw_id]
    if not checks.add("held-out draw logged", len(draws) == 1, draw_id):
        return {}
    draw = draws[0]
    checks.add(
        "held-out draw matches evidence",
        draw.get("gpu") == gpu and draw.get("candidate") == candidate,
        {"gpu": draw.get("gpu"), "candidate": draw.get("candidate")},
    )
    code, args = draw.get("code_commit"), draw.get("candidate_args")
    checks.add(
        "held-out draw records the code commit and candidate arguments",
        _declared(code, args),
        {"code_commit": code, "candidate_args": args},
    )
    seeds = draw.get("seeds", {})
    bad = {
        r: seeds.get(r)
        for r in held_out_rows
        if len(set(seeds.get(r, []))) != HELD_OUT_SEEDS_PER_ROW
    }
    checks.add("two logged seeds per held-out row", not bad, bad)
    for row, s in draw.get("zn_extra", {}).items():
        rule = tier2_gates._limit(limits, row, "", "zn_rule")
        cap = rule.get("extra_seeds_max") if isinstance(rule, dict) else None
        checks.add(
            f"{row} extra Zn seeds allowed",
            cap is not None and len(s) <= cap,
            {"rule": rule, "seeds": s},
        )
    return draw


def _repeated(values) -> list:
    counts = Counter(values)
    return sorted((v for v, n in counts.items() if n > 1), key=str)


def check_log_receipts(checks: Checks, log) -> None:
    """Receipt ids are unique, a receipt is named by at most one draw, and a
    candidate (code commit and ordered arguments) is drawn at most once."""
    receipts = [r.get("id") for r in log.get("receipts", [])]
    checks.add(
        "rehearsal receipt ids unique", not _repeated(receipts), _repeated(receipts)
    )
    draws = log.get("draws", [])
    named = [d["receipt"] for d in draws if d.get("receipt") is not None]
    checks.add(
        "each rehearsal receipt named by at most one draw",
        not _repeated(named),
        _repeated(named),
    )
    candidates = [
        json.dumps([d["code_commit"], d.get("candidate_args")])
        for d in draws
        if d.get("code_commit") is not None
    ]
    checks.add(
        "each candidate drawn at most once",
        not _repeated(candidates),
        _repeated(candidates),
    )


def evidence_digest(sets) -> str:
    """Canonical sha256 of development evidence: each set's declaration, the
    run record and trace of every run role and row, and both measurement
    files, as sorted ``[role, row, file, sha256]`` entries (``None`` for a
    missing file). Run records list their artifact digests, so this binds
    every output."""

    def digest(path):
        return provenance.sha256_file(path) if path and path.is_file() else None

    entries = []
    for s in sets:
        name = s.get("name")
        text = json.dumps(s, sort_keys=True).encode()
        entries.append(["set", name, "declaration", hashlib.sha256(text).hexdigest()])
        for role in EVIDENCE_RUNS:
            run = Path(s[role]) if s.get(role) else None
            for row in s.get("rows", []):
                for file in (f"{row}.run.json", f"{row}.trace.jsonl"):
                    entries.append([role, row, file, digest(run and run / file)])
        for role in ("tier1", "tier2"):
            path = Path(s[role]) if s.get(role) else None
            entries.append([role, name, "measurement", digest(path)])
    canonical = json.dumps(sorted(entries, key=json.dumps)).encode()
    return hashlib.sha256(canonical).hexdigest()


def check_receipt(checks: Checks, log, draw, git, digest, contract_tag) -> None:
    """The draw names a passing rehearsal receipt of the same candidate and
    contract tag, committed before the draw, whose evidence is the
    development evidence being accepted."""
    rid = draw.get("receipt")
    found = [r for r in log.get("receipts", []) if r.get("id") == rid]
    if not checks.add(
        "draw names a logged rehearsal receipt",
        rid is not None and len(found) == 1,
        rid,
    ):
        return
    receipt = found[0]
    checks.add(
        "rehearsal receipt passed",
        receipt.get("status") == REHEARSAL_STATUS
        and receipt.get("exit_code") == REHEARSAL,
        {"status": receipt.get("status"), "exit_code": receipt.get("exit_code")},
    )
    keys = ("gpu", "candidate", "code_commit", "candidate_args")
    differing = {
        k: [receipt.get(k), draw.get(k)] for k in keys if receipt.get(k) != draw.get(k)
    }
    checks.add("rehearsal receipt matches the draw", not differing, differing)
    checks.add(
        "rehearsal receipt names the contract tag",
        receipt.get("contract_tag") == contract_tag,
        {"receipt": receipt.get("contract_tag"), "contract": contract_tag},
    )
    tagged = (
        git.repo.commit(f"refs/tags/{contract_tag}") if git and contract_tag else None
    )
    gate_sha = receipt.get("gate_sha256")
    checks.add(
        "rehearsal receipt records a gate at the contract tag",
        isinstance(gate_sha, str)
        and len(gate_sha) == 64
        and set(gate_sha) <= HEX
        and receipt.get("gate_approved_tag") == contract_tag
        and receipt.get("gate_commit") == tagged,
        {
            "gate_sha256": gate_sha,
            "gate_approved_tag": receipt.get("gate_approved_tag"),
            "gate_commit": receipt.get("gate_commit"),
            "tag_commit": tagged,
        },
    )
    checks.add(
        "development evidence matches the rehearsal receipt",
        receipt.get("evidence_sha256") == digest,
        {"receipt": receipt.get("evidence_sha256"), "evidence": digest},
    )
    versions = git.versions if git else []
    first = next(
        (
            i
            for i, (_, v) in enumerate(versions)
            if any(d.get("id") == draw.get("id") for d in (v or {}).get("draws", []))
        ),
        None,
    )
    before = versions[first - 1][1] if first else None
    checks.add(
        "rehearsal receipt committed before the draw",
        any(r.get("id") == rid for r in (before or {}).get("receipts", [])),
        None if first is None else versions[first][0][:12],
    )


def make_receipt(
    verdict, receipt_id, verdict_text: str, gate_text: str
) -> dict[str, Any]:
    """The held-out log entry for a passing rehearsal (``--receipt``), with
    the passing ``qual_gate.py`` report that preceded it."""
    gate = json.loads(gate_text)
    return {
        "id": receipt_id,
        "gpu": verdict.get("gpu"),
        "candidate": verdict.get("candidate"),
        "route": verdict.get("route"),
        "contract_tag": verdict.get("contract_tag"),
        "code_commit": verdict.get("code_commit"),
        "candidate_args": verdict.get("candidate_args"),
        "evidence_sha256": verdict.get("evidence_sha256"),
        "verdict_sha256": hashlib.sha256(verdict_text.encode()).hexdigest(),
        "status": verdict.get("status"),
        "exit_code": verdict.get("exit_code"),
        "gate_sha256": hashlib.sha256(gate_text.encode()).hexdigest(),
        "gate_commit": gate.get("gate_commit"),
        "gate_approved_tag": gate.get("approved_tag"),
    }


def _without_zn_extra(draw):
    return {k: v for k, v in draw.items() if k != "zn_extra"}


def _extends(old, new) -> bool:
    """``new`` keeps every receipt and draw of ``old`` unchanged, except that
    extra Zn seeds may be appended, and may add receipts and draws after them."""
    ra, rb = old.get("receipts", []), new.get("receipts", [])
    if rb[: len(ra)] != ra:
        return False
    a, b = old.get("draws", []), new.get("draws", [])
    if len(b) < len(a):
        return False
    for d, e in zip(a, b):
        if _without_zn_extra(d) != _without_zn_extra(e):
            return False
        za, zb = d.get("zn_extra", {}), e.get("zn_extra", {})
        if any(zb.get(row, [])[: len(s)] != s for row, s in za.items()):
            return False
    return True


def check_contract(checks: Checks, git: Git | None, root: Path, tag) -> None:
    """The draw's code commit descends from the frozen contract tag and holds
    the tag's qualification directory (code, constants, inputs, limits,
    recipe, baseline and wrapper) unchanged apart from the held-out log."""
    repo = git.repo if git else None
    tagged = repo.commit(f"refs/tags/{tag}") if repo and isinstance(tag, str) else None
    if not checks.add("contract tag present", tagged is not None, tag):
        return
    match = re.fullmatch(r"(.+-)(\d+)", tag)
    family = []
    if match:
        listed = (repo.git("tag", "-l", f"{match[1]}*") or "").split()
        family = [
            int(m[1])
            for name in listed
            if (m := re.fullmatch(re.escape(match[1]) + r"(\d+)", name))
        ]
    checks.add(
        "contract tag is the newest",
        bool(family) and int(match[2]) == max(family),
        {"tag": tag, "numbers": sorted(family)},
    )
    checks.add(
        "code commit descends from the contract tag",
        repo.is_ancestor(tagged, git.code),
        {"tag": tag, "code_commit": git.code},
    )
    qual = root.resolve().parent.relative_to(repo.path.resolve()).as_posix()
    out = repo.git("diff", "--name-only", tagged, git.code, "--", qual)
    changed = None if out is None else sorted(set(out.splitlines()) - {git.log_path})
    checks.add(
        "qualification files unchanged since the contract tag",
        changed == [],
        changed,
    )


def check_git(checks: Checks, root: Path, draw) -> Git | None:
    repo = Repo.find(root)
    if not checks.add("inputs are in a Git checkout", repo is not None):
        return None
    log_path = (
        (root / "held_out_log.json")
        .resolve()
        .relative_to(repo.path.resolve())
        .as_posix()
    )
    code = repo.commit(draw.get("code_commit"))
    if not checks.add("code commit present", code is not None, draw.get("code_commit")):
        return None
    head = repo.commit("HEAD")
    status = repo.git("status", "--porcelain")
    checks.add(
        "acceptance checkout clean",
        status == "",
        None if status is None else status.splitlines(),
    )
    changed = repo.changed(code, head)
    checks.add(
        "acceptance checkout differs from the code commit only in the held-out log",
        changed is not None and set(changed) <= {log_path},
        changed,
    )
    versions = []
    for rev in repo.history(head, log_path):
        try:
            versions.append((rev, json.loads(repo.show(rev, log_path) or "")))
        except json.JSONDecodeError:
            versions.append((rev, None))
    unreadable = [rev[:12] for rev, v in versions if v is None]
    rewritten = [
        new[:12]
        for (_, a), (new, b) in zip(versions, versions[1:])
        if a is not None and b is not None and not _extends(a, b)
    ]
    checks.add(
        "held-out log history only appends",
        bool(versions) and not unreadable and not rewritten,
        {"versions": len(versions), "unreadable": unreadable, "rewritten": rewritten},
    )
    places: dict[int, str] = {}
    reused = set()
    for _, log in versions:
        for s, where in _seed_places(log or {}):
            if places.setdefault(s, where) != where:
                reused.add((s, places[s], where))
    checks.add(
        "held-out seeds used once across the log history",
        not reused,
        sorted(reused),
    )
    git = Git(repo, code, head, log_path)
    git.versions = versions
    return git


def _record(run_dir: Path, row: str):
    path = run_dir / f"{row}.run.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _trace(path: Path):
    if not path.is_file():
        return None, None
    return trace_tools.load(path)


def expected_artifacts(jobs, seeds, n_sample: int) -> set[str]:
    return {
        f"{job}/seed_{s}/predictions/{name}"
        for job in jobs
        for s in seeds
        for i in range(n_sample)
        for name in (
            f"{job}_sample_{i}.cif",
            f"{job}_full_data_sample_{i}.json",
            f"{job}_summary_confidence_sample_{i}.json",
        )
    }


class Context:
    """What every run check needs: inputs, recipe, draw, Git state and the
    frozen provenance references (input digests, effective input digests and
    this GPU's runtime assets)."""

    def __init__(self, checks, manifest, recipe, draw, git, capability, frozen=None):
        self.checks, self.manifest, self.recipe = checks, manifest, recipe
        self.table = {r["id"]: r for r in manifest["rows"]}
        self.draw, self.git, self.capability = draw, git, capability
        self.versions: set[str] = set()
        self.frozen = frozen or {}
        self.assets: dict[str, set] = {}
        self._references: dict[tuple, list[str]] = {}

    def jobs(self, row):
        return [
            j["name"] for j in json.loads(Path(self.table[row]["input"]).read_text())
        ]

    def references(self, row, use_msa: bool) -> list[str]:
        """The files the row's effective input names, as frozen-digest keys."""
        key = (self.table[row]["input"], use_msa)
        if key not in self._references:
            source = Path(key[0]).read_bytes()
            _, jobs = provenance.effective_input(source, use_msa)
            self._references[key] = sorted(
                {Path(p).as_posix() for p in provenance.referenced_files(jobs)}
            )
        return self._references[key]

    def seeds(self, row):
        if self.table[row]["role"] == "development":
            return list(self.manifest["development_seeds"])
        return list(self.draw.get("seeds", {}).get(row, []))


def _n_sample(rec) -> int:
    value = _last((rec or {}).get("cli_args") or [], "--sample")
    return int(value) if value and value.isdigit() else 0


def check_run(ctx: Context, label, run_dir: Path, row, seeds, kind, zn=False):
    """Check one run's record; return it (None if missing or failed)."""
    name = f"{row} {label}"
    rec = _record(run_dir, row)
    if not ctx.checks.add(
        f"{name} run record",
        rec is not None and rec.get("exit_code") == 0 and rec.get("exception") is None,
        None
        if rec is None
        else {"exit_code": rec.get("exit_code"), "exception": rec.get("exception")},
    ):
        return None
    before = rec.get("preexisting")
    tier1 = kind.startswith("tier1")
    ctx.checks.add(
        f"{name} fresh output",
        isinstance(before, dict)
        and before.get("output") is False
        and before.get("tier1_capture") is (False if tier1 else None),
        before,
    )
    fp = rec.get("fingerprint") or {}
    args = rec.get("cli_args") or []
    out = _flag(args, "-o")
    commit_ok, commit_detail = (
        ctx.git.run_commit(fp.get("commit"))
        if ctx.git
        else (False, {"commit": fp.get("commit")})
    )
    capability = list((fp.get("gpu") or {}).get("capability") or [])
    ctx.checks.add(
        f"{name} provenance",
        commit_ok
        and not fp.get("dirty_paths")
        and capability == ctx.capability
        and _same_path(out, run_dir / row),
        {
            **commit_detail,
            "dirty_paths": fp.get("dirty_paths"),
            "capability": capability,
            "out": out,
        },
    )
    ctx.versions.add(json.dumps(fp.get("versions"), sort_keys=True))
    check_imports(ctx, name, fp.get("imports"))
    check_input_provenance(ctx, name, rec, row, seeds)
    if ctx.table[row]["role"] == "held-out":
        draw = (
            ctx.git.draw_at(fp.get("commit"), ctx.draw.get("id")) if ctx.git else None
        )
        if zn:
            logged = (draw or {}).get("zn_extra", {}).get(row, [])
            present = draw is not None and set(seeds) <= set(logged)
        else:
            present = draw is not None and draw.get("seeds", {}).get(row) == list(seeds)
        ctx.checks.add(f"{name} seeds logged at its commit", present, seeds)
    extra = list(ctx.recipe["tier1_extra" if kind.startswith("tier1") else "extra"])
    if kind in CANDIDATE_KINDS:
        extra += list(ctx.draw.get("candidate_args") or [])
    env = rec.get("env")
    try:
        overrides, expected, row_env = run_rows.build_command(
            ctx.table[row],
            ctx.manifest,
            Path(out or "-"),
            ",".join(str(s) for s in seeds),
            extra,
            environ=env or {},
        )
    except SystemExit as error:
        ctx.checks.add(f"{name} command", False, str(error))
        return rec
    same = args == expected and rec.get("config_overrides") == overrides
    ctx.checks.add(
        f"{name} command",
        same,
        None
        if same
        else {
            "expected": [overrides, expected],
            "recorded": [rec.get("config_overrides"), args],
        },
    )
    host = set(ctx.recipe["host_env"])
    found_env = None if env is None else {k: v for k, v in env.items() if k not in host}
    ctx.checks.add(
        f"{name} environment",
        found_env == {**ctx.recipe["env"], **row_env},
        found_env,
    )
    wrapper = rec.get("wrapper")
    if not isinstance(wrapper, dict):
        ctx.checks.add(f"{name} wrapper options", False, wrapper)
        return rec
    trace = wrapper.get("draw_trace")
    traced = _same_path(trace, run_dir / f"{row}.trace.jsonl")
    if kind in ("off", "cand"):
        trace_ok = traced
    elif kind == "repeat":
        trace_ok = trace is None or traced
    else:
        trace_ok = trace is None
    ctx.checks.add(
        f"{name} wrapper options",
        trace_ok
        and (wrapper.get("tier1_capture") is not None) == tier1
        and (wrapper.get("tier1_replay") is not None) == (kind == "tier1 cand")
        and wrapper.get("phase_timing") is None,
        wrapper,
    )
    return rec


def _cli_flag(cli, name: str, default: bool) -> bool:
    value = _last(cli, name)
    return (
        default
        if value is None
        else value.lower() in ("true", "1", "yes", "y", "t", "on")
    )


def _override(row: dict, key: str):
    for text in row.get("overrides", []):
        name, sep, value = text.partition("=")
        if sep and name == key:
            return value
    return None


def check_input_provenance(ctx: Context, name, rec, row, seeds) -> None:
    """A run read its frozen input and referenced files, used this GPU's
    frozen runtime assets and produced exactly its declared seeds
    (contract-6; ``run_opendde.py`` records the evidence)."""
    prov = rec.get("provenance")
    if not ctx.checks.add(
        f"{name} input provenance recorded",
        isinstance(prov, dict)
        and prov.get("refused") is None
        and not prov.get("finish_error"),
        None
        if not isinstance(prov, dict)
        else {k: prov.get(k) for k in ("refused", "finish_error")},
    ):
        return
    spec, cli = ctx.table[row], rec.get("cli_args") or []
    flags = {
        "use_msa": _cli_flag(cli, "--use_msa", True),
        "use_template": _cli_flag(cli, "--use_template", False),
        "use_rna_msa": _cli_flag(cli, "--use_rna_msa", False),
    }
    digests = ctx.frozen.get("digests") or {}
    key = Path(spec["input"]).as_posix()
    source = digests.get(key)
    effective = (
        (ctx.frozen.get("effective") or {}).get(key) if flags["use_msa"] else None
    )
    expected = effective["effective_sha256"] if effective else source
    src = prov.get("source") or {}
    reads = prov.get("parser_reads") or []
    read = reads[0] if len(reads) == 1 else {}
    ctx.checks.add(
        f"{name} input read as frozen",
        source is not None
        and src.get("path") == spec["input"]
        and src.get("sha256") == source
        and prov.get("flags") == flags
        and read.get("matches_parse") is True
        and read.get("sha256") == expected
        and prov.get("expected_effective_sha256") == expected,
        {
            "source": [src.get("sha256"), source],
            "parser_reads": [r.get("sha256") for r in reads],
            "expected_read": expected,
            "flags": [prov.get("flags"), flags],
        },
    )
    want = set(ctx.references(row, flags["use_msa"]))
    found = {
        Path(p).as_posix(): state for p, state in (prov.get("referenced") or {}).items()
    }
    unstable = sorted(
        k
        for k in want | set(found)
        if k not in want
        or k not in found
        or digests.get(k) is None
        or found[k].get("before") != digests[k]
        or found[k].get("after") != digests[k]
    )
    ctx.checks.add(
        f"{name} referenced files stable and frozen",
        not unstable,
        {"files": len(want), "differing": unstable[:5]},
    )
    ref = ctx.frozen.get("assets")
    if flags["use_template"]:
        check_template_provenance(ctx, name, prov, spec, ref)
    required = BASE_ASSETS + (TEMPLATE_ASSETS if flags["use_template"] else ())
    got = prov.get("assets") or {}
    wrong = {
        n: [(got.get(n) or {}).get(k) for k in ("resolved", "sha256")]
        for n in required
        if not isinstance(ref, dict)
        or n not in ref
        or (got.get(n) or {}).get("resolved") != ref[n].get("path")
        or (got.get(n) or {}).get("sha256") != ref[n].get("sha256")
    }
    ctx.checks.add(
        f"{name} runtime assets match the frozen reference",
        isinstance(ref, dict) and not wrong,
        {"reference": isinstance(ref, dict), "differing": wrong},
    )
    for n in required:
        if n in got:
            ctx.assets.setdefault(n, set()).add(
                (got[n].get("resolved"), got[n].get("sha256"))
            )
    jobs = ctx.jobs(row)
    expected_seeds = {job: sorted(seeds) for job in jobs}
    ctx.checks.add(
        f"{name} output holds the declared seeds",
        prov.get("seeds") == expected_seeds,
        {"recorded": prov.get("seeds"), "declared": expected_seeds},
    )


def check_template_provenance(ctx: Context, name, prov, spec, ref) -> None:
    prefix = _override(spec, "data.template.prot_template_mmcif_dir")
    external = ctx.frozen.get("external") or {}
    tree = (
        {
            k[len(prefix) + 1 :]: v
            for k, v in external.items()
            if k.startswith(prefix + "/")
        }
        if prefix
        else {}
    )
    mmcif = prov.get("template_mmcif") or {}
    ctx.checks.add(
        f"{name} template mmCIF files stable and frozen",
        bool(tree) and mmcif.get("before") == tree and mmcif.get("after") == tree,
        {"frozen": len(tree), "before_matches": mmcif.get("before") == tree},
    )
    cache = prov.get("template_cache") or {}
    ctx.checks.add(
        f"{name} template cache new at the parser read",
        cache.get("inside_output") is True
        and cache.get("absent_before") is True
        and cache.get("absent_at_parse") is True,
        cache,
    )
    kalign = prov.get("kalign") or {}
    frozen = ((ref or {}).get("kalign_binary_path") or {}).get("path")
    ctx.checks.add(
        f"{name} requested Kalign is the frozen binary",
        frozen is not None
        and kalign.get("executable") is True
        and kalign.get("requested_resolved") == frozen,
        {"requested": kalign.get("requested_resolved"), "frozen": frozen},
    )


def check_rollouts(ctx: Context, name, meta, jobs, seeds, cli) -> None:
    n_sample, n_step = _last(cli, "--sample"), _last(cli, "--step")
    guided = _last(cli, "--use_tfg_guidance") == "true"
    rollouts = meta.get("rollouts") or []
    found = Counter((r.get("sample_name"), r.get("seed")) for r in rollouts)
    expected = Counter((j, s) for j in jobs for s in seeds)
    wrong = [
        {k: r.get(k) for k in ("sample_name", "seed", "N_sample", "N_step")}
        for r in rollouts
        if str(r.get("N_sample")) != n_sample
        or str(r.get("N_step")) != n_step
        or r.get("diffusion_chunk_size") != ctx.recipe["diffusion_chunk_size"]
        or bool((r.get("guidance_configs") or {}).get("enable")) != guided
    ]
    ctx.checks.add(
        f"{name} rollouts",
        found == expected and not wrong,
        {
            "missing": [list(k) for k in expected - found],
            "unexpected": [list(k) for k in found - expected],
            "wrong_settings": wrong,
        },
    )


def _rollout_key(r):
    return (str(r.get("sample_name")), str(r.get("seed")), str(r.get("rollout")))


def check_traces(ctx: Context, name, off_dir, cand_dir, recs, jobs, seeds, row):
    loaded = {
        tag: _trace(run / f"{row}.trace.jsonl")
        for tag, run in (("off", off_dir), ("cand", cand_dir))
    }
    if not ctx.checks.add(
        f"{name} traces present", all(m is not None for m, _ in loaded.values())
    ):
        return
    problems = []
    for tag, (meta, records) in loaded.items():
        problems += [f"{tag}: {r}" for r in trace_tools.unusable(meta, records)]
        rec = recs.get(tag)
        if rec is None:
            problems.append(f"{tag}: no usable run record")
        elif meta.get("cli_args") != rec.get("cli_args") or meta.get(
            "fingerprint"
        ) != rec.get("fingerprint"):
            problems.append(f"{tag}: trace header differs from its run record")
    if not ctx.checks.add(f"{name} traces usable", not problems, problems):
        return
    for tag, (meta, records) in loaded.items():
        check_rollouts(ctx, f"{name} {tag}", meta, jobs, seeds, recs[tag]["cli_args"])
        draws = trace_tools.check_manifest(meta, records)
        ctx.checks.add(
            f"{name} {tag} draws complete",
            draws["ok"],
            {
                k: draws.get(k)
                for k in ("expected", "observed", "missing_by_site", "extra_by_site")
            },
        )
    a, b = (
        sorted(m.get("rollouts") or [], key=_rollout_key) for m, _ in loaded.values()
    )
    ctx.checks.add(f"{name} off and candidate rollout settings match", a == b)
    cmp = trace_tools.compare(loaded["off"][1], loaded["cand"][1], "raw_sha256")
    ctx.checks.add(
        f"{name} random inputs match off (raw)",
        cmp["ok"],
        {
            k: cmp[k]
            for k in ("compared", "only_in_a", "only_in_b", "mismatched_by_site")
        },
    )


def _outputs(rec, kind: str) -> dict[str, str]:
    return ((rec or {}).get("outputs") or {}).get(kind) or {}


def check_artifacts(ctx: Context, name, row_dir: Path, expected: set[str], rec):
    """The run directory holds exactly the expected artifacts, byte for byte
    the ones its run record lists."""
    found = provenance.prediction_digests(row_dir)
    recorded = _outputs(rec, "prediction_sha256")
    differing = sorted(
        k for k in set(found) | set(recorded) if found.get(k) != recorded.get(k)
    )
    ctx.checks.add(
        f"{name} artifacts complete",
        bool(expected) and set(found) == expected,
        {
            "missing": len(expected - set(found)),
            "unexpected": sorted(set(found) - expected)[:5],
        },
    )
    ctx.checks.add(
        f"{name} artifacts match its run record",
        rec is not None and bool(recorded) and not differing,
        {"differing": len(differing), "first": differing[:3]},
    )


def _identity(rec):
    fp = (rec or {}).get("fingerprint") or {}
    return [fp.get("host"), fp.get("slurm_job_id"), fp.get("utc")]


def check_repeat(ctx: Context, row, cand: Path, repeat: Path, rec_cand, rec_rep):
    ctx.checks.add(
        f"{row} repeat is a separate run",
        rec_cand is not None
        and rec_rep is not None
        and not _same_path(cand, repeat)
        and _identity(rec_cand) != _identity(rec_rep),
        {"cand": _identity(rec_cand), "repeat": _identity(rec_rep)},
    )
    det = compare_outputs.compare(cand / row, repeat / row)
    ctx.checks.add(f"{row} candidate deterministic", det["ok"], det)


def check_drift(ctx: Context, row, off: Path, baseline) -> None:
    base = (baseline or {}).get("rows", {}).get(row)
    now = compare_outputs.digest(off / row)["artifacts"]
    differing = sorted(
        k for k in set(base or {}) | set(now) if (base or {}).get(k) != now.get(k)
    )
    ctx.checks.add(
        f"{row} off matches step 0 baseline",
        base is not None and bool(now) and not differing,
        {"baseline": base is not None, "differing": len(differing)},
    )


def _wrapper(rec) -> dict:
    return (rec or {}).get("wrapper") or {}


def check_tier1_runs(ctx: Context, s, row, seeds, data) -> None:
    runs = s.get("tier1_off"), s.get("tier1_cand")
    if not ctx.checks.add(f"{row} tier 1 runs declared", all(runs)):
        return
    rec_off = check_run(ctx, "tier 1 off", Path(runs[0]), row, seeds, "tier1 off")
    rec_cand = check_run(ctx, "tier 1 cand", Path(runs[1]), row, seeds, "tier1 cand")
    cap_off = _wrapper(rec_off).get("tier1_capture")
    cap_cand = _wrapper(rec_cand).get("tier1_capture")
    replay = _wrapper(rec_cand).get("tier1_replay")
    ref, other = data.get("ref"), data.get("other")
    expected = {f"{job}/seed_{s}/tier1.pt" for job in ctx.jobs(row) for s in seeds}
    for tag, rec in (("off", rec_off), ("cand", rec_cand)):
        found = set(_outputs(rec, "tier1_sha256"))
        ctx.checks.add(
            f"{row} tier 1 {tag} captures complete",
            rec is not None and found == expected,
            {
                "missing": sorted(expected - found)[:5],
                "unexpected": sorted(found - expected)[:5],
            },
        )
    entries = {(e["id"], e["job"], e["seed"]): e for e in data.get("entries", [])}
    for job in ctx.jobs(row):
        for s in seeds:
            e = entries.get((row, job, s)) or {}
            key = f"{job}/seed_{s}/tier1.pt"
            want = (
                _outputs(rec_off, "tier1_sha256").get(key),
                _outputs(rec_cand, "tier1_sha256").get(key),
            )
            got = (e.get("ref_sha256"), e.get("other_sha256"))
            ctx.checks.add(
                f"{row} {job} seed {s} tier 1 inputs match its runs",
                None not in want and got == want,
                {"measured": got, "recorded": want},
            )
    ctx.checks.add(
        f"{row} tier 1 measured its declared captures",
        ref is not None
        and other is not None
        and _same_path(Path(ref) / row, cap_off)
        and _same_path(Path(other) / row, cap_cand)
        and _same_path(replay, cap_off),
        {
            "ref": ref,
            "other": other,
            "capture_off": cap_off,
            "capture_cand": cap_cand,
            "replay": replay,
        },
    )


def check_tier1(checks: Checks, row, jobs, seeds, data, limits):
    entries = {(e["id"], e["job"], e["seed"]): e for e in data.get("entries", [])}
    for job in jobs:
        for seed in seeds:
            e = entries.get((row, job, seed))
            name = f"{row} {job} seed {seed} tier 1"
            if e is None or e.get("status") != "ok":
                checks.add(name, False, "missing" if e is None else e.get("status"))
                continue
            over = {
                k: _get(e, k)
                for k, lim in limits["max"].items()
                if _get(e, k) is None or not _get(e, k) <= lim
            }
            not_finite = [k for k in limits["finite"] if _get(e, k) is not True]
            source = e.get("head_source") == limits["head_source"]
            checks.add(
                name,
                not over and not not_finite and source,
                {
                    "over": over,
                    "not_finite": not_finite,
                    "head_source": e.get("head_source"),
                },
            )


def _measure_dir(data, key, row, job, seed) -> Path:
    return Path(data[key]) / row / job / f"seed_{seed}"


def check_zn_sets(ctx: Context, zn_sets) -> dict[str, dict[int, tuple[Path, Path]]]:
    """Check each extra-seed run pair; return {row: {seed: (off, cand)}}."""
    runs: dict[str, dict[int, list[tuple[Path, Path]]]] = {}
    for s in zn_sets:
        seeds = list(s.get("seeds") or [])
        off, cand = Path(s["off"]), Path(s["cand"])
        label = "extra seeds " + ",".join(str(x) for x in seeds)
        for row in s.get("rows", []):
            if row not in ctx.table:
                ctx.checks.add(f"{row} in manifest", False)
                continue
            recs = {
                tag: check_run(ctx, f"{label} {tag}", run, row, seeds, tag, zn=True)
                for tag, run in (("off", off), ("cand", cand))
            }
            jobs = ctx.jobs(row)
            check_traces(ctx, f"{row} {label}", off, cand, recs, jobs, seeds, row)
            expected = expected_artifacts(jobs, seeds, _n_sample(recs["off"]))
            for tag, run in (("off", off), ("cand", cand)):
                check_artifacts(
                    ctx, f"{row} {label} {tag}", run / row, expected, recs[tag]
                )
            for seed in seeds:
                runs.setdefault(row, {}).setdefault(seed, []).append((off, cand))
    for row, by_seed in runs.items():
        logged = list(ctx.draw.get("zn_extra", {}).get(row, []))
        once = all(len(v) == 1 for v in by_seed.values())
        ctx.checks.add(
            f"{row} extra Zn runs cover the logged seeds",
            once and sorted(by_seed) == sorted(logged),
            {"logged": logged, "run": sorted(by_seed)},
        )
    return {
        row: {seed: pairs[0] for seed, pairs in by_seed.items()}
        for row, by_seed in runs.items()
    }


def check_measurement_source(ctx: Context, name, data) -> None:
    """A measurement must come from the candidate's code, like a run."""
    source = data.get("source") or {}
    ok, detail = (
        ctx.git.run_commit(source.get("commit"))
        if ctx.git
        else (False, {"commit": source.get("commit")})
    )
    ctx.checks.add(
        f"{name} measurement provenance",
        ok and source.get("dirty_paths") == [],
        {**detail, "dirty_paths": source.get("dirty_paths")},
    )
    check_imports(ctx, f"{name} measurement", source.get("imports"))


def check_imports(ctx: Context, name, imports) -> None:
    """``opendde`` and ``runner`` were loaded from the acceptance checkout,
    not from another checkout through an editable install."""
    root = ctx.git.repo.path.resolve() if ctx.git else None
    imports = imports if isinstance(imports, dict) else {}
    outside = {
        pkg: imports.get(pkg)
        for pkg in provenance.PACKAGES
        if root is None
        or not isinstance(imports.get(pkg), str)
        or not Path(imports[pkg]).resolve().is_relative_to(root)
    }
    ctx.checks.add(
        f"{name} imports from the checkout",
        not outside,
        {"outside": outside, "checkout": str(root)},
    )


def _seed_outputs(rec, job, seed) -> dict[str, str]:
    prefix = f"{job}/seed_{seed}/"
    return {
        k: v
        for k, v in _outputs(rec, "prediction_sha256").items()
        if k.startswith(prefix)
    }


def check_tier2_inputs(ctx: Context, e, seeds, extra, recs, extra_runs) -> None:
    """A tier 2 entry measured the row's logged same-seed pairs and extra Zn
    seeds, and read exactly the artifacts its runs recorded."""
    row, job = e["id"], e["job"]
    name = f"{row} {job} tier 2"
    pairs = [[s, s] for s in seeds]
    ctx.checks.add(
        f"{name} seed pairs",
        e.get("seed_pairs") == pairs and e.get("zn_extra_seeds") == list(extra),
        {
            "measured": [e.get("seed_pairs"), e.get("zn_extra_seeds")],
            "logged": [pairs, list(extra)],
        },
    )
    for i, tag in enumerate(("off", "cand")):
        want: dict[str, str] = {}
        for s in seeds:
            want.update(_seed_outputs(recs.get(tag), job, s))
        for s in extra:
            run = extra_runs.get(s)
            want.update(_seed_outputs(_record(run[i], row) if run else None, job, s))
        got = (e.get("inputs") or {}).get(tag)
        differing = sorted(
            k for k in set(got or {}) | set(want) if (got or {}).get(k) != want.get(k)
        )
        ctx.checks.add(
            f"{name} {tag} inputs match its runs",
            bool(want) and not differing,
            {"differing": len(differing), "first": differing[:3]},
        )


def _check_zn_extra(checks, row, seeds, extra, t2, extra_runs):
    for e in t2.get("entries", []):
        if e["id"] != row or not e.get("zn"):
            continue
        by_seed = [b["seeds"][0] for b in e["zn"].get("by_seed") or []]
        checks.add(
            f"{row} {e['job']} Zn seed order",
            by_seed == list(seeds) + list(extra),
            by_seed,
        )
        for s in extra:
            pair = extra_runs.get(s)
            same = pair is not None and all(
                _same_path(
                    _measure_dir(t2, key, row, e["job"], s),
                    run / row / e["job"] / f"seed_{s}",
                )
                for key, run in zip(("off", "cand"), pair)
            )
            checks.add(f"{row} {e['job']} extra seed {s} measured from its runs", same)


def check_entry_keys(ctx: Context, s, route, t1, t2) -> None:
    """Each expected measurement once and nothing else: tier 2 per row and
    job, tier 1 per tier 1 row, job and seed. Evaluators index entries by
    these keys, so a duplicate could otherwise replace a failed measurement."""
    rows = [r for r in s.get("rows", []) if r in ctx.table]
    tier1_rows = [r for r in rows if r in route.get("tier1_rows", [])]
    expected = {
        "tier 1": {
            (r, job, seed)
            for r in tier1_rows
            for job in ctx.jobs(r)
            for seed in ctx.seeds(r)
        },
        "tier 2": {(r, job) for r in rows for job in ctx.jobs(r)},
    }
    for tag, data in (("tier 1", t1), ("tier 2", t2)):
        if data is None:
            continue
        fields = ("id", "job", "seed") if tag == "tier 1" else ("id", "job")
        found = Counter(
            tuple(e.get(f) for f in fields) for e in data.get("entries", [])
        )
        twice = sorted((k for k, n in found.items() if n > 1), key=str)
        unexpected = sorted(set(found) - expected[tag], key=str)
        name = f"{s.get('name')} {tag}"
        ctx.checks.add(f"{name} entries unique", not twice, twice[:5])
        ctx.checks.add(f"{name} entries expected", not unexpected, unexpected[:5])


def _load_measurement(checks: Checks, s, key, tag):
    """A declared measurement file, or None with a failed check if it cannot
    be read (missing or not JSON); the verdict then fails instead of the
    command raising."""
    path = s.get(key)
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        checks.add(
            f"{s.get('name')} {tag} measurement readable",
            False,
            f"{type(error).__name__}: {error}",
        )
        return None


def check_row(ctx: Context, s, row, route, baseline, t1, t1_limits):
    """Every check on one row of a full set except tier 2."""
    off, cand = Path(s["off"]), Path(s["cand"])
    seeds, jobs = ctx.seeds(row), ctx.jobs(row)
    recs = {
        tag: check_run(ctx, tag, run, row, seeds, tag)
        for tag, run in (("off", off), ("cand", cand))
    }
    check_traces(ctx, row, off, cand, recs, jobs, seeds, row)
    expected = expected_artifacts(jobs, seeds, _n_sample(recs["off"]))
    for tag, run in (("off", off), ("cand", cand)):
        check_artifacts(ctx, f"{row} {tag}", run / row, expected, recs[tag])
    if ctx.checks.add(f"{row} repeat run declared", bool(s.get("cand_repeat"))):
        repeat = Path(s["cand_repeat"])
        rec_rep = check_run(ctx, "repeat", repeat, row, seeds, "repeat")
        check_artifacts(ctx, f"{row} repeat", repeat / row, expected, rec_rep)
        check_repeat(ctx, row, cand, repeat, recs["cand"], rec_rep)
    if row in route.get("drift_rows", []):
        check_drift(ctx, row, off, baseline)
    if row in route.get("tier1_rows", []):
        if ctx.checks.add(f"{row} tier 1 evidence declared", t1 is not None):
            check_tier1_runs(ctx, s, row, seeds, t1)
            check_tier1(ctx.checks, row, jobs, seeds, t1, t1_limits)
    return recs


def accept(evidence: dict[str, Any], root: Path = INPUTS) -> dict[str, Any]:
    checks = Checks()
    required_file = json.loads((root / "acceptance.json").read_text())
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    table = {r["id"]: r for r in manifest["rows"]}
    t2_path, t1_path = root / "tier2_limits.json", root / "tier1_limits.json"
    t2_limits = json.loads(t2_path.read_text())
    t1_limits = json.loads(t1_path.read_text())
    gpu = required_file["gpus"].get(evidence.get("gpu"))
    route = (gpu or {}).get("routes", {}).get(evidence.get("route"))
    if not checks.add("GPU and route declared", gpu is not None and route is not None):
        return _verdict(checks, evidence, frozen=False)
    required = list(route["rows"])
    rehearsal = bool(route.get("rehearsal"))

    full_sets = [s for s in evidence.get("sets", []) if not s.get("zn_only")]
    zn_sets = [s for s in evidence.get("sets", []) if s.get("zn_only")]
    listed = [r for s in full_sets for r in s.get("rows", [])]
    missing = [r for r in required if r not in listed]
    extra = sorted({r for r in listed if r not in required})
    twice = sorted({r for r in listed if listed.count(r) > 1})
    checks.add("required rows present", not missing, missing)
    checks.add("no undeclared rows", not extra, extra)
    checks.add("each row in one set", not twice, twice)

    check_digests(checks, manifest_path)
    if rehearsal:
        draw = check_rehearsal(checks, evidence)
    else:
        held_out = [r for r in required if table.get(r, {}).get("role") == "held-out"]
        log = json.loads((root / "held_out_log.json").read_text())
        draw = check_log(
            checks,
            log,
            evidence.get("held_out_draw"),
            evidence.get("gpu"),
            evidence.get("candidate"),
            held_out,
            t2_limits,
        )
    git = check_git(checks, root, draw)
    frozen_refs = load_provenance_references(checks, root, evidence.get("gpu"))
    ctx = Context(
        checks,
        manifest,
        required_file["recipe"],
        draw,
        git,
        list(gpu["capability"]),
        frozen_refs,
    )
    baseline_path = root / required_file["baseline"]
    baseline = (
        json.loads(baseline_path.read_text()) if baseline_path.is_file() else None
    )
    checks.add("step 0 baseline digests present", baseline is not None)

    extra_runs = check_zn_sets(ctx, zn_sets)
    entries = []
    results = []
    for s in full_sets:
        t1 = _load_measurement(checks, s, "tier1", "tier 1")
        t2 = _load_measurement(checks, s, "tier2", "tier 2")
        for tag, data in (("tier 1", t1), ("tier 2", t2)):
            if data is not None:
                check_measurement_source(ctx, f"{s.get('name')} {tag}", data)
        check_entry_keys(ctx, s, route, t1, t2)
        for row in s.get("rows", []):
            if row not in table:
                checks.add(f"{row} in manifest", False)
                continue
            recs = check_row(ctx, s, row, route, baseline, t1, t1_limits)
            if not checks.add(f"{row} tier 2 evidence declared", t2 is not None):
                continue
            seeds = ctx.seeds(row)
            checks.add(
                f"{row} tier 2 measured these runs",
                _same_path(t2.get("off"), s["off"])
                and _same_path(t2.get("cand"), s["cand"]),
            )
            for e in t2.get("entries", []):
                if e["id"] != row:
                    continue
                for tag in ("off", "cand"):
                    found = sorted(int(k) for k in (e.get(f"samples_{tag}") or {}))
                    checks.add(
                        f"{row} {e['job']} tier 2 {tag} seeds",
                        found == sorted(seeds),
                        found,
                    )
                check_tier2_inputs(
                    ctx,
                    e,
                    seeds,
                    draw.get("zn_extra", {}).get(row, []),
                    recs,
                    extra_runs.get(row, {}),
                )
                entries.append(e)
            logged_extra = draw.get("zn_extra", {}).get(row)
            if logged_extra:
                _check_zn_extra(
                    checks, row, seeds, logged_extra, t2, extra_runs.get(row, {})
                )
        t2_rows = [r for r in s.get("rows", []) if r in table]
        if t2 is not None and t2_rows:
            results += tier2_gates.evaluate(
                {"entries": entries}, t2_limits, manifest, t2_rows
            )
            entries = []
    for r in results:
        checks.add(f"{r['id']} {r['job']} tier 2", r["pass"], r["failed"])
    logged_rows = {r for r, v in draw.get("zn_extra", {}).items() if v}
    unlogged = sorted({r for z in zn_sets for r in z.get("rows", [])} - logged_rows)
    checks.add("extra Zn runs have logged seeds", not unlogged, unlogged)
    checks.add(
        "every run shares one set of software versions",
        len(ctx.versions) == 1,
        sorted(ctx.versions),
    )
    checks.add(
        "every run used one path and digest per runtime asset",
        bool(ctx.assets) and all(len(v) == 1 for v in ctx.assets.values()),
        {n: sorted(v, key=str) for n, v in ctx.assets.items() if len(v) != 1},
    )
    zn = tier2_gates.zn_claim(results)
    development = [
        s
        for s in full_sets
        if all(table.get(r, {}).get("role") == "development" for r in s.get("rows", []))
    ]
    digest = evidence_digest(development)
    if not rehearsal:
        tag = gpu.get("frozen", {}).get("contract_tag")
        check_receipt(checks, log, draw, git, digest, tag)
    frozen = gpu.get("frozen", {})
    hashes = {"tier1": sha256(t1_path), "tier2": sha256(t2_path)}
    fields = [f"{k}_limits_sha256" for k in hashes] + ["contract_tag"]
    is_frozen = any(frozen.get(f) for f in fields)
    if is_frozen:
        for k, v in hashes.items():
            checks.add(
                f"{k} limits match the frozen hash",
                frozen.get(f"{k}_limits_sha256") == v,
            )
        check_contract(checks, git, root, frozen.get("contract_tag"))
    if rehearsal:
        checks.add("rehearsal under a declared freeze", is_frozen)
        checks.add("rehearsal Zn claim qualified", zn == "qualified", zn)
    verdict = _verdict(
        checks, evidence, is_frozen, zn, hashes, draw.get("code_commit"), rehearsal
    )
    verdict["candidate_args"] = draw.get("candidate_args")
    verdict["evidence_sha256"] = digest
    verdict["contract_tag"] = frozen.get("contract_tag")
    return verdict


def _verdict(
    checks,
    evidence,
    frozen,
    zn="none",
    hashes=None,
    code_commit=None,
    rehearsal=False,
):
    """Any failed check fails. A rehearsal with every check passing exits 4,
    never 0. Otherwise, with every check passing, a Zn claim other than
    ``qualified`` (H3 unresolved at its seed cap) is a partial result without
    Zn, which is not an acceptance; an undeclared freeze makes it provisional."""
    failed = checks.failed()
    if failed:
        status, code = "fail", FAIL
    elif rehearsal:
        status, code = REHEARSAL_STATUS, REHEARSAL
    elif zn != "qualified":
        status, code = "partial without Zn (not an acceptance)", PARTIAL
    elif not frozen:
        status, code = "provisional: contract not frozen", PROVISIONAL
    else:
        status, code = "full acceptance", FULL
    return {
        "status": status,
        "exit_code": code,
        "zn_claim": zn,
        "gpu": evidence.get("gpu"),
        "route": evidence.get("route"),
        "candidate": evidence.get("candidate"),
        "code_commit": code_commit,
        "held_out_draw": evidence.get("held_out_draw"),
        "rehearsal": rehearsal,
        "limits_sha256": hashes,
        "failed": [c["check"] for c in failed],
        "checks": checks.items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--receipt",
        nargs=2,
        metavar=("ID", "PATH"),
        help="after a passing rehearsal, write its held-out log receipt to PATH",
    )
    parser.add_argument(
        "--gate-report",
        type=Path,
        help="the passing qual_gate.py report recorded in the receipt",
    )
    args = parser.parse_args()
    if "OPENDDE_QUAL_DATA" not in os.environ:
        print("error: OPENDDE_QUAL_DATA is not set")
        return FAIL
    verdict = accept(json.loads(args.evidence.read_text()))
    text = json.dumps(verdict, indent=1, default=str) + "\n"
    if args.out:
        args.out.write_text(text)
    if args.receipt:
        gate_text = args.gate_report.read_text() if args.gate_report else None
        gate_ok = gate_text is not None and json.loads(gate_text).get("ok") is True
        if verdict["exit_code"] != REHEARSAL:
            print("error: --receipt needs a passing rehearsal; no receipt written")
        elif not gate_ok:
            print("error: --receipt needs a passing --gate-report; no receipt written")
        else:
            receipt = make_receipt(verdict, args.receipt[0], text, gate_text)
            Path(args.receipt[1]).write_text(json.dumps(receipt, indent=1) + "\n")
    print(f"{verdict['status']} (Zn claim: {verdict['zn_claim']})")
    for name in verdict["failed"]:
        print(f"  failed: {name}")
    return verdict["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
