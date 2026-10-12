#!/usr/bin/env python
"""Decide full acceptance of a candidate on one GPU (qualification_protocol.md, E5).

Usage:
    python scripts/qualification/accept.py EVIDENCE.json [--out VERDICT.json]

Run from the repository root on the GPU host, with ``OPENDDE_QUAL_DATA`` set.
This command, not ``tier2_gates.py evaluate`` or any other diagnostic, decides
acceptance. The rows a GPU and route must cover come from
``inputs/acceptance.json``, never from the command line or the evidence file.

EVIDENCE.json names the GPU, route, candidate, commit and held-out draw, and
lists run sets. Each set has ``rows`` and the run directories ``off`` and
``cand`` written by ``run_rows.py --draw-trace``. A full set also names
``cand_repeat`` (a second candidate run, for determinism), ``tier1`` (a
``tier1_compare.py`` output) and ``tier2`` (a ``tier2_gates.py measure``
output), and the development set names ``off_baseline`` for the drift check. A
set with ``"zn_only": true`` holds H3's extra Zn seeds: traces only.

Checks:
- every required row appears in exactly one full set, and no other row does;
- the frozen inputs match their digests, repository and external;
- the held-out log is well formed: seeds from 201-299, each used once in the
  whole log, two per held-out row, extra Zn seeds only where ``zn_rule``
  allows them;
- per row: ``off`` and candidate traces are usable, match under the raw hash,
  come from the evidence commit with no uncommitted changes, on this GPU, with
  the same software versions, deterministic, writing to the declared
  directory, and use the expected seeds (development rows 101 and 102; a
  held-out row its logged seeds; H3's extra runs its logged extra seeds);
- per row: two candidate runs give byte-identical prediction artifacts, and
  development rows' ``off`` matches the step 0 baseline byte for byte;
- per row: every job and seed passes the approved tier 1 limits;
- tier 2: every job of every required row passes ``tier2_gates.evaluate``
  with the expected seeds, H3's Zn seed pairs are its primary seeds followed
  by its logged extra seeds (read from the extra runs), and the aggregate Zn
  claim is ``qualified``;
- the limits files match the hashes frozen in ``acceptance.json``.

Exit status: 0 full acceptance; 1 failure; 2 every check passes except the
Zn claim, which is reported as a partial result without Zn and is not an
acceptance (an H3 unresolved at its seed cap is a held-out failure); 3 every
check passes but the limits are not frozen yet (provisional).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import compare_outputs
import freeze_inputs
import tier2_gates
import trace_tools

INPUTS = Path("scripts/qualification/inputs")
DEV_SEEDS = (101, 102)
HELD_OUT_RANGE = (201, 299)
HELD_OUT_SEEDS_PER_ROW = 2
FULL, FAIL, PARTIAL, PROVISIONAL = 0, 1, 2, 3


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


def _get(entry: dict[str, Any], dotted: str):
    """``section.key``; the key itself may contain dots (``margin_0.1``)."""
    value: Any = entry
    for part in dotted.split(".", 1):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def check_digests(checks: Checks, manifest: Path) -> None:
    for name, current, stored in (
        ("digests", freeze_inputs.digests, "digests.json"),
        ("external digests", freeze_inputs.external_digests, "external_digests.json"),
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


def check_log(checks: Checks, log, draw_id, gpu, candidate, held_out_rows, limits):
    """Return {row: seeds} and {row: extra Zn seeds} for the evidence's draw."""
    seen: dict[int, str] = {}
    repeated = []
    for draw in log.get("draws", []):
        for kind in ("seeds", "zn_extra"):
            for row, seeds in draw.get(kind, {}).items():
                for s in seeds:
                    where = f"{draw.get('id')}/{kind}/{row}"
                    if s in seen:
                        repeated.append([s, seen[s], where])
                    seen[s] = where
    checks.add("held-out seeds used once", not repeated, repeated)
    lo, hi = HELD_OUT_RANGE
    out_of_range = sorted(s for s in seen if not lo <= s <= hi)
    checks.add("held-out seeds in 201-299", not out_of_range, out_of_range)
    draws = [d for d in log.get("draws", []) if d.get("id") == draw_id]
    if not held_out_rows:
        return {}, {}
    if not checks.add("held-out draw logged", len(draws) == 1, draw_id):
        return {}, {}
    draw = draws[0]
    checks.add(
        "held-out draw matches evidence",
        draw.get("gpu") == gpu and draw.get("candidate") == candidate,
        {"gpu": draw.get("gpu"), "candidate": draw.get("candidate")},
    )
    seeds = {r: list(v) for r, v in draw.get("seeds", {}).items()}
    bad = {
        r: seeds.get(r)
        for r in held_out_rows
        if len(set(seeds.get(r, []))) != HELD_OUT_SEEDS_PER_ROW
    }
    checks.add("two logged seeds per held-out row", not bad, bad)
    extra = {r: list(v) for r, v in draw.get("zn_extra", {}).items()}
    for row, s in extra.items():
        rule = tier2_gates._limit(limits, row, "", "zn_rule")
        cap = rule.get("extra_seeds_max") if isinstance(rule, dict) else None
        checks.add(
            f"{row} extra Zn seeds allowed",
            cap is not None and len(s) <= cap,
            {"rule": rule, "seeds": s},
        )
    return seeds, extra


def _trace(path: Path):
    if not path.is_file():
        return None, None
    return trace_tools.load(path)


def check_traces(checks: Checks, row, off_dir, cand_dir, seeds, ev, capability):
    name = f"{row} traces"
    meta_a, rec_a = _trace(off_dir / f"{row}.trace.jsonl")
    meta_b, rec_b = _trace(cand_dir / f"{row}.trace.jsonl")
    if not checks.add(f"{name} present", meta_a is not None and meta_b is not None):
        return
    problems = [f"off: {r}" for r in trace_tools.unusable(meta_a, rec_a)]
    problems += [f"cand: {r}" for r in trace_tools.unusable(meta_b, rec_b)]
    checks.add(f"{name} usable", not problems, problems)
    cmp = trace_tools.compare(rec_a, rec_b, "raw_sha256")
    checks.add(
        f"{row} random inputs match off (raw)",
        cmp["ok"],
        {
            k: cmp[k]
            for k in ("compared", "only_in_a", "only_in_b", "mismatched_by_site")
        },
    )
    fp_a, fp_b = meta_a.get("fingerprint", {}), meta_b.get("fingerprint", {})
    for tag, meta, fp, run_dir in (
        ("off", meta_a, fp_a, off_dir),
        ("cand", meta_b, fp_b, cand_dir),
    ):
        args = meta.get("cli_args", [])
        out = _flag(args, "-o")
        checks.add(
            f"{row} {tag} provenance",
            fp.get("commit") == ev["commit"]
            and not fp.get("dirty_paths")
            and list((fp.get("gpu") or {}).get("capability") or []) == capability
            and _flag(args, "--deterministic") == "true"
            and out is not None
            and Path(out).resolve() == (run_dir / row).resolve(),
            {
                "commit": fp.get("commit"),
                "dirty_paths": fp.get("dirty_paths"),
                "capability": (fp.get("gpu") or {}).get("capability"),
                "out": out,
            },
        )
        found = sorted({r["seed"] for r in meta.get("rollouts", [])})
        checks.add(f"{row} {tag} seeds", found == sorted(seeds), found)
    checks.add(
        f"{row} off and candidate share software versions",
        fp_a.get("versions") == fp_b.get("versions"),
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
    capability = list(gpu["capability"])
    required = list(route["rows"])

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
    held_out = [r for r in required if table.get(r, {}).get("role") == "held-out"]
    log = json.loads((root / "held_out_log.json").read_text())
    logged, zn_extra = check_log(
        checks,
        log,
        evidence.get("held_out_draw"),
        evidence.get("gpu"),
        evidence.get("candidate"),
        held_out,
        t2_limits,
    )

    def seeds_of(row):
        if table[row]["role"] == "development":
            return list(DEV_SEEDS)
        return logged.get(row, [])

    entries = []
    results = []
    for s in full_sets:
        off, cand = Path(s["off"]), Path(s["cand"])
        repeat = Path(s["cand_repeat"]) if s.get("cand_repeat") else None
        baseline = Path(s["off_baseline"]) if s.get("off_baseline") else None
        t1 = json.loads(Path(s["tier1"]).read_text()) if s.get("tier1") else None
        t2 = json.loads(Path(s["tier2"]).read_text()) if s.get("tier2") else None
        for row in s.get("rows", []):
            if row not in table:
                checks.add(f"{row} in manifest", False)
                continue
            seeds = seeds_of(row)
            jobs = [
                j["name"] for j in json.loads(Path(table[row]["input"]).read_text())
            ]
            check_traces(checks, row, off, cand, seeds, evidence, capability)
            if checks.add(f"{row} repeat run declared", repeat is not None):
                det = compare_outputs.compare(cand / row, repeat / row)
                checks.add(f"{row} candidate deterministic", det["ok"], det)
            if row in route.get("drift_rows", []):
                if checks.add(f"{row} off baseline declared", baseline is not None):
                    drift = compare_outputs.compare(off / row, baseline / row)
                    checks.add(f"{row} off matches step 0 baseline", drift["ok"], drift)
            if row in route.get("tier1_rows", []):
                if checks.add(f"{row} tier 1 evidence declared", t1 is not None):
                    check_tier1(checks, row, jobs, seeds, t1, t1_limits)
            if not checks.add(f"{row} tier 2 evidence declared", t2 is not None):
                continue
            ok_paths = (
                Path(t2.get("off", "")).resolve() == off.resolve()
                and Path(t2.get("cand", "")).resolve() == cand.resolve()
            )
            checks.add(f"{row} tier 2 measured these runs", ok_paths)
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
                entries.append(e)
            if row in zn_extra:
                _check_zn_extra(
                    checks, row, seeds, zn_extra[row], t2, zn_sets, evidence, capability
                )
        t2_rows = [r for r in s.get("rows", []) if r in table]
        if t2 is not None and t2_rows:
            results += tier2_gates.evaluate(
                {"entries": entries}, t2_limits, manifest, t2_rows
            )
            entries = []
    for r in results:
        checks.add(f"{r['id']} {r['job']} tier 2", r["pass"], r["failed"])
    undeclared = sorted({r for z in zn_sets for r in z.get("rows", [])} - set(zn_extra))
    checks.add("extra Zn runs have logged seeds", not undeclared, undeclared)
    zn = tier2_gates.zn_claim(results)
    frozen = gpu.get("frozen", {})
    hashes = {"tier1": sha256(t1_path), "tier2": sha256(t2_path)}
    is_frozen = all(frozen.get(f"{k}_limits_sha256") for k in hashes)
    if is_frozen:
        for k, v in hashes.items():
            checks.add(
                f"{k} limits match the frozen hash",
                frozen[f"{k}_limits_sha256"] == v,
            )
    return _verdict(checks, evidence, is_frozen, zn, hashes)


def _check_zn_extra(checks, row, seeds, extra, t2, zn_sets, ev, capability):
    sets = [s for s in zn_sets if row in s.get("rows", [])]
    if not checks.add(f"{row} extra Zn runs declared", len(sets) == 1):
        return
    off, cand = Path(sets[0]["off"]), Path(sets[0]["cand"])
    check_traces(checks, row, off, cand, extra, ev, capability)
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
            same = all(
                _measure_dir(t2, key, row, e["job"], s).resolve()
                == (run / row / e["job"] / f"seed_{s}").resolve()
                for key, run in (("off", off), ("cand", cand))
            )
            checks.add(f"{row} {e['job']} extra seed {s} measured from its runs", same)


def _verdict(checks, evidence, frozen, zn="none", hashes=None):
    """Any failed check fails. With every check passing, a Zn claim other than
    ``qualified`` (H3 unresolved at its seed cap) is a partial result without
    Zn, which is not an acceptance; unfrozen limits make the result provisional."""
    failed = checks.failed()
    if failed:
        status, code = "fail", FAIL
    elif zn != "qualified":
        status, code = "partial without Zn (not an acceptance)", PARTIAL
    elif not frozen:
        status, code = "provisional: limits not frozen", PROVISIONAL
    else:
        status, code = "full acceptance", FULL
    return {
        "status": status,
        "exit_code": code,
        "zn_claim": zn,
        "gpu": evidence.get("gpu"),
        "route": evidence.get("route"),
        "candidate": evidence.get("candidate"),
        "commit": evidence.get("commit"),
        "held_out_draw": evidence.get("held_out_draw"),
        "limits_sha256": hashes,
        "failed": [c["check"] for c in failed],
        "checks": checks.items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if "OPENDDE_QUAL_DATA" not in os.environ:
        print("error: OPENDDE_QUAL_DATA is not set")
        return FAIL
    verdict = accept(json.loads(args.evidence.read_text()))
    if args.out:
        args.out.write_text(json.dumps(verdict, indent=1, default=str) + "\n")
    print(f"{verdict['status']} (Zn claim: {verdict['zn_claim']})")
    for name in verdict["failed"]:
        print(f"  failed: {name}")
    return verdict["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
