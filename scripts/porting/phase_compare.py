#!/usr/bin/env python
"""Controlled-timing comparison of two configurations (kit-item disposition, plan revision 117).

Usage:
    python scripts/porting/phase_compare.py --base DIR [DIR ...] --cand DIR [DIR ...]
        [--rows D1,...,D15] [--out REPORT.json]

Each DIR is a run_rows.py --phase-timing output directory (one <row>.phases.json per row). A row's model forward
time in a DIR is the sum of its "predict" phase records over all seeds (cold and warm alike); passes of one
configuration are averaged. Prints, per row, the base and candidate forward time and the candidate's saving, and the
saving of the summed forward time, against the disposition thresholds (under 3% summed and under 5% on every row
means not worth porting). The spread between base passes is printed beside it as the measurement's noise.
"""

import argparse
import json
from pathlib import Path


def forward_seconds(directory: Path, row: str) -> float:
    data = json.loads((directory / f"{row}.phases.json").read_text())
    return sum(r["seconds"] for r in data["phases"] if r["phase"] == "predict")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", nargs="+", type=Path, required=True)
    ap.add_argument("--cand", nargs="+", type=Path, required=True)
    ap.add_argument("--rows", default=",".join(f"D{i}" for i in range(1, 16)))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = args.rows.split(",")
    report = {
        "base": [str(d) for d in args.base],
        "cand": [str(d) for d in args.cand],
        "rows": {},
    }
    tot_b = tot_c = 0.0
    print(
        f"{'row':5s} {'base s':>9s} {'cand s':>9s} {'saving':>8s} {'base spread':>12s}"
    )
    for row in rows:
        b = [forward_seconds(d, row) for d in args.base]
        c = [forward_seconds(d, row) for d in args.cand]
        mb, mc = sum(b) / len(b), sum(c) / len(c)
        spread = (max(b) - min(b)) / mb if len(b) > 1 else 0.0
        report["rows"][row] = {
            "base": b,
            "cand": c,
            "saving": 1 - mc / mb,
            "base_spread": spread,
        }
        tot_b += mb
        tot_c += mc
        print(
            f"{row:5s} {mb:9.1f} {mc:9.1f} {100 * (1 - mc / mb):7.2f}% {100 * spread:11.2f}%"
        )
    saving = 1 - tot_c / tot_b
    worst = max(report["rows"].values(), key=lambda r: r["saving"])["saving"]
    report.update(
        summed_base=tot_b, summed_cand=tot_c, summed_saving=saving, max_row_saving=worst
    )
    report["under_threshold"] = saving < 0.03 and worst < 0.05
    print(
        f"summed: base {tot_b:.1f} s, cand {tot_c:.1f} s, saving {100 * saving:.2f}%; largest row saving {100 * worst:.2f}%"
    )
    print(
        "under the (a) threshold (3% summed, 5% every row):", report["under_threshold"]
    )
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
