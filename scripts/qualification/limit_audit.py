#!/usr/bin/env python
"""Audit which harmless stock contrast sets each tier 2 limit (protocol C8).

Usage:
    python scripts/qualification/limit_audit.py PRECISION_EVAL.json
        KERNEL_EVAL.json --limits LIMITS.json [--out AUDIT.json]

Both inputs are ``tier2_gates.py evaluate`` outputs: bf16 against fp32
(precision) and ``off`` against the ``torch`` triangle-kernel swap (kernel).
Every gated value is mapped onto a "badness" scale on which the limit is an
upper bound: deficits from 1 for lDDT, Fnat, pocket Jaccard and Zn
same-site fraction, missing same-outcome pairs for the outcome count,
|log ratio| for the spread ratio, |candidate - off| for the constrained
distance, and the value itself for at-most gates. Limits are grouped by
gate and limit value: most C9 limits are shared across classes and were set
from the worst value over every row that uses them, and a row override with
its own value forms its own group. Per-polymer-type lDDT gates share the
all-polymer limit (C9), so they are pooled with it.

Per group the audit records the worst precision value, the worst kernel-swap
value, the binding contrast and the precision-only limit: twice the worst
precision value, rounded up to 1, 1.5, 2, 2.5, 3, 4, 5, 6 or 8 times a power
of ten (the C9 doubling rule). A group needs action only when the kernel swap
is binding and the current limit is looser than the precision-only limit;
the audit then reports whether the kernel swap passes the precision-only
limit. A worst precision value of exactly 0 gives no scale for doubling; such
a group is reported as degenerate instead of rederived. Count rules (outcome floors, guidance counts, the exact-duplicate
floor and the near-duplicate count) are recorded but not rederived.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

UNIT = ("lddt_", "pair_interface_fnat", "ligand_pocket", "zn_same_site")
AT_MOST = (
    "pair_interface_fnonnat",
    "ligand_rmsd",
    "ranking_score",
    "mean_plddt",
    "p95_error",
    "ensemble_ref_",
    "ensemble_contact_frequency",
    "ensemble_plddt_median",
)
RULES = {
    "outcome_count": "count floor (C1)",
    "guidance_contacts": "guidance count (C7)",
    "guidance_epitope": "guidance count (C7)",
    "no_duplicate_samples": "exact-duplicate floor",
    "ensemble_near_duplicates": "declared from precision only (C8)",
    "guidance_contact_mean": "guidance tolerance (C7)",
}
TYPES = ("protein", "dna", "rna")
GRID = (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10)
SAMPLES = 10


def round_up(x: float) -> float:
    if x <= 0:
        return 0.0
    p = 10 ** math.floor(math.log10(x))
    return next(round(g * p, 12) for g in GRID if g * p >= x * (1 - 1e-12))


def badness(name: str, gate: dict[str, Any]):
    """Return (value, limit) on a scale where the limit is an upper bound."""
    v, lim = gate.get("value"), gate.get("limit")
    if gate.get("status") not in ("pass", "fail") or v is None:
        return None
    if name.startswith(UNIT):
        return 1.0 - v, 1.0 - lim
    if name == "ensemble_spread_ratio":
        return abs(math.log(v)), max(abs(math.log(lim[0])), abs(math.log(lim[1])))
    if name == "guidance_contact_mean":
        return abs(v[1] - v[0]), lim[0]
    if name == "outcome_count":
        return SAMPLES - v, SAMPLES - lim
    if name == "ensemble_near_duplicates":
        return v["pairs_below"], lim
    if name in RULES:
        return v, lim
    if name.startswith(AT_MOST):
        return max(v, 0.0), lim
    return None


def family(name: str) -> str:
    """Per-type lDDT gates take the all-polymer limit (C9)."""
    for t in TYPES:
        name = name.replace(f"lddt_{t}", "lddt_polymer")
    return name


def _class_of(limits: dict[str, Any], row_id: str, job: str):
    rows = limits.get("rows", {})
    for name in (f"{row_id}/{job}", row_id):
        if name in rows and "class" in rows[name]:
            return rows[name]["class"]
    return None


def audit(precision, kernel, limits) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, Any]] = {}
    for contrast, results in (("precision", precision), ("kernel", kernel)):
        for r in results:
            for name, gate in r["gates"].items():
                b = badness(name, gate)
                if b is None:
                    continue
                b = (b[0], b[1], name)
                key = (family(name), json.dumps(gate["limit"]))
                grp = groups.setdefault(
                    key,
                    {
                        "gate": key[0],
                        "classes": set(),
                        "members": set(),
                        "limit": gate["limit"],
                        "limit_badness": b[1],
                        "jobs": set(),
                        "precision": None,
                        "kernel": None,
                    },
                )
                grp["jobs"].add(f"{r['id']} {r['job']}")
                grp["members"].add(name)
                grp["classes"].add(str(_class_of(limits, r["id"], r["job"])))
                worst = grp[contrast]
                if worst is None or b[0] > worst[0]:
                    grp[contrast] = (b[0], f"{r['id']} {r['job']} {name}")
    out = []
    for grp in groups.values():
        p, k = grp["precision"], grp["kernel"]
        row = {
            "gate": grp["gate"],
            "classes": sorted(grp["classes"]),
            "members": sorted(grp["members"]),
            "limit": grp["limit"],
            "limit_badness": grp["limit_badness"],
            "jobs": sorted(grp["jobs"]),
            "precision_worst": p[0] if p else None,
            "precision_job": p[1] if p else None,
            "kernel_worst": k[0] if k else None,
            "kernel_job": k[1] if k else None,
        }
        if grp["gate"] in RULES:
            row["source"] = RULES[grp["gate"]]
            row["action"] = "none (rule)"
        elif p is None:
            row["source"] = "kernel only"
            row["action"] = "rederive: no precision value"
        elif p[0] == 0 and k and k[0] > 0:
            row["binding"] = "kernel"
            row["action"] = "degenerate: precision shows no change"
        else:
            binding = "kernel" if k and k[0] > p[0] else "precision"
            rederived = round_up(2 * p[0])
            row["binding"] = binding
            row["precision_only_limit"] = rederived
            looser = grp["limit_badness"] > rederived + 1e-12
            if binding == "kernel" and looser:
                row["action"] = "rederive"
                row["kernel_passes_rederived"] = k[0] <= rederived
            else:
                row["action"] = "none"
        out.append(row)
    out.sort(key=lambda r: (r["gate"], str(r["limit"])))
    return out


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("precision", type=Path)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--limits", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = audit(
        json.loads(args.precision.read_text())["results"],
        json.loads(args.kernel.read_text())["results"],
        json.loads(args.limits.read_text()),
    )
    for r in rows:
        print(
            f"{r['gate']:34s} {','.join(r['classes']):24s} lim {_fmt(r['limit_badness']):>7s}"
            f"  P {_fmt(r['precision_worst']):>8s} ({r['precision_job'] or '-'})"
            f"  K {_fmt(r['kernel_worst']):>8s} ({r['kernel_job'] or '-'})"
            f"  2P {_fmt(r.get('precision_only_limit')):>6s}"
            f"  {r.get('binding', '-'):9s} {r['action']}"
            + (
                f" kernel {'passes' if r['kernel_passes_rederived'] else 'FAILS'}"
                if "kernel_passes_rederived" in r
                else ""
            )
        )
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
