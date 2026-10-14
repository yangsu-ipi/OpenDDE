#!/usr/bin/env python
"""Audit registered run stores against the held-out do-not-run list (contract-6, protocol E5).

Usage:
    python scripts/qualification/run_audit.py --registry REGISTRY.json [--out REPORT.json]

Run from the repository root. REGISTRY.json lists the run stores to search
(``{"stores": [{"path": ..., "registered": ..., "note": ...}]}``; it lives
outside ``scripts/qualification`` so registering a store is not a contract
amendment). Every ``*.run.json`` record and every ``<job>/seed_<s>``
directory under a store is examined.

For a recorded run, the input bytes come from the record's contract-6
provenance (source digest, looked up in ``signature_inventory.json``), or
else from ``git show <recorded commit>:<input>``; an input whose bytes cannot
be recovered that way is classified from the current file and reported as
unverified. The seeds actually used come from the output layout (or the
record's provenance), not from the command line. An output directory with no
record is reported with its job name and seeds and classified only by job
name.

Findings, using the inventory's sequence families:

- ``stop``: an input containing a held-out-only sequence, or a
  restraint-protected sequence together with a restraint, at any seed; or an
  input containing a shared (7wux or 9lh2) sequence at a reserved seed;
  unless the run's row and seeds belong to a logged draw in
  ``held_out_log.json``.
- ``deviation``: a reserved seed on any other input (recorded, not a stop).
- ``ok`` otherwise.

The command exits 1 if any stop is found. The audit covers only runs whose
inputs and seeds can be reconstructed; registration cannot show that no
unrecorded run occurred, so report every unverified run and missing store as
a gap in the qualification claim.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import provenance

INPUTS = Path("scripts/qualification/inputs")


def _git_show(commit, path) -> bytes | None:
    if not isinstance(commit, str) or not commit or commit == "unknown":
        return None
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{Path(path).as_posix()}"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _last(args, name):
    found = [i for i, a in enumerate(args[:-1]) if a == name]
    return args[found[-1] + 1] if found else None


class Auditor:
    def __init__(self, inventory: dict, log: dict):
        self.inventory = inventory
        self.families = {k: set(v) for k, v in inventory["families"].items()}
        lo, hi = inventory["held_out_seed_range"]
        self.reserved = range(lo, hi + 1)
        self.by_sha = {f["sha256"]: (p, f) for p, f in inventory["files"].items()}
        self.by_job = {}
        for path, f in inventory["files"].items():
            for job in f["jobs"]:
                self.by_job.setdefault(job["name"], []).append((path, job))
        self.logged: dict[str, set] = {}
        for draw in log.get("draws", []):
            for kind in ("seeds", "zn_extra"):
                for row, seeds in (draw.get(kind) or {}).items():
                    self.logged.setdefault(row, set()).update(seeds)

    def classify(self, signatures: list[dict], seeds, rows) -> tuple[str, str]:
        reserved = sorted(s for s in seeds if s in self.reserved)
        logged = bool(rows) and any(
            set(seeds) <= self.logged.get(row, set()) for row in rows
        )
        protected = shared = False
        for sig in signatures:
            seqs = {p[1] for p in sig["polymers"]}
            if seqs & self.families["held_out_only"] or (
                sig["restraint"] and seqs & self.families["restraint_protected"]
            ):
                protected = True
            if seqs & self.families["shared"]:
                shared = True
        if logged:
            return "ok", "run of a logged draw"
        if protected:
            return "stop", "protected input outside a logged draw"
        if shared and reserved:
            return "stop", f"7wux or 9lh2 input at reserved seeds {reserved}"
        if reserved:
            return "deviation", f"reserved seeds {reserved} on a development input"
        return "ok", ""

    def record(self, path: Path) -> dict[str, Any]:
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {
                "record": str(path),
                "finding": "gap",
                "detail": "unreadable record",
            }
        cli = rec.get("cli_args") or []
        source = _last(cli, "-i") or _last(cli, "--input")
        prov = rec.get("provenance") or {}
        data, status = None, "unverified"
        digest = (prov.get("source") or {}).get("sha256")
        if digest in self.by_sha:
            status = "verified (recorded digest)"
            signatures = [j["signature"] for j in self.by_sha[digest][1]["jobs"]]
            rows = self.by_sha[digest][1]["rows"]
        else:
            data = _git_show((rec.get("fingerprint") or {}).get("commit"), source or "")
            if data is not None:
                status = "verified (recorded commit)"
            elif source and Path(source).is_file():
                data = Path(source).read_bytes()
            entry = self.by_sha.get(provenance.sha256_bytes(data)) if data else None
            rows = entry[1]["rows"] if entry else []
            try:
                jobs = json.loads(data) if data else []
            except json.JSONDecodeError:
                jobs = []
            signatures = [
                provenance.entity_signature(j) for j in jobs if isinstance(j, dict)
            ]
        row_dir = path.with_name(path.name.removesuffix(".run.json"))
        out = Path(_last(cli, "-o") or row_dir)
        if not out.is_dir():
            out = row_dir
        names = [d.name for d in out.iterdir() if d.is_dir()] if out.is_dir() else []
        seeds = sorted(
            {
                s
                for v in (
                    prov.get("seeds") or provenance.job_seeds(out, names)
                ).values()
                for s in v
            }
        )
        finding, detail = (
            self.classify(signatures, seeds, rows)
            if signatures
            else ("gap", "input bytes not recoverable")
        )
        return {
            "record": str(path),
            "input": source,
            "input_status": status,
            "rows": rows,
            "seeds": seeds,
            "finding": finding,
            "detail": detail,
        }

    def bare(self, seed_dir: Path) -> dict[str, Any]:
        job = seed_dir.parent.name
        seed = int(provenance.SEED_DIR.fullmatch(seed_dir.name)[1])
        matches = self.by_job.get(job, [])
        signatures = [j["signature"] for _, j in matches]
        rows = sorted(
            {r for p, _ in matches for r in self.inventory["files"][p]["rows"]}
        )
        finding, detail = (
            self.classify(signatures, [seed], rows)
            if signatures
            else (
                ("deviation", f"reserved seed {seed}, unknown input")
                if seed in self.reserved
                else ("ok", "unknown input")
            )
        )
        return {
            "output": str(seed_dir),
            "input": None,
            "input_status": "unverified (job name only)",
            "rows": rows,
            "seeds": [seed],
            "finding": finding,
            "detail": detail,
        }


def audit(registry: dict, inventory: dict, log: dict) -> dict[str, Any]:
    auditor = Auditor(inventory, log)
    runs, gaps = [], []
    for store in registry.get("stores", []):
        root = Path(store["path"])
        if not root.is_dir():
            gaps.append(f"store missing: {root}")
            continue
        covered = set()
        for path in sorted(root.rglob("*.run.json")):
            result = auditor.record(path)
            runs.append(result)
            covered.add(path.with_name(path.name.removesuffix(".run.json")))
        for seed_dir in sorted(root.rglob("seed_*")):
            if not seed_dir.is_dir() or not provenance.SEED_DIR.fullmatch(
                seed_dir.name
            ):
                continue
            if any(seed_dir.is_relative_to(c) for c in covered):
                continue
            runs.append(auditor.bare(seed_dir))
    count = {
        k: sum(r["finding"] == k for r in runs) for k in ("stop", "deviation", "gap")
    }
    unverified = sum(r["input_status"].startswith("unverified") for r in runs)
    return {
        "ok": count["stop"] == 0,
        "stores": [s["path"] for s in registry.get("stores", [])],
        "counts": {**count, "unverified": unverified, "runs": len(runs)},
        "gaps": gaps,
        "findings": [r for r in runs if r["finding"] != "ok"],
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(
        json.loads(args.registry.read_text()),
        json.loads((INPUTS / "signature_inventory.json").read_text()),
        json.loads((INPUTS / "held_out_log.json").read_text()),
    )
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("ok", "counts", "gaps")}, indent=1))
    for r in report["findings"]:
        print(f"  {r['finding']}: {r.get('record') or r.get('output')} {r['detail']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
