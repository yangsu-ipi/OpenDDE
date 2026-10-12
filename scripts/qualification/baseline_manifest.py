#!/usr/bin/env python
"""Write the step 0 ``off`` baseline digest manifest (qualification_protocol.md, E5).

Usage:
    python scripts/qualification/baseline_manifest.py --out OUT.json
        --recipe TEXT --runs RUN_DIR ROWS SOURCE [--runs ...]

Each ``--runs`` names a ``run_rows.py`` output directory, the comma-separated
development rows taken from it and a description of the source run (job IDs
and commit; no host paths). The manifest records, per row, the sha256 of every
prediction artifact under ``RUN_DIR/<row>`` (``compare_outputs.artifacts``),
keyed by its path relative to the row directory. ``accept.py`` compares a new
``off`` run's artifacts with these digests for the drift check, so the
baseline is never a path supplied by the evidence set.
"""

import argparse
import json
from pathlib import Path

import compare_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument(
        "--runs", nargs=3, action="append", required=True, metavar="ARG"
    )
    args = parser.parse_args()
    sources, rows = [], {}
    for run_dir, names, source in args.runs:
        names = names.split(",")
        for row in names:
            if row in rows:
                raise SystemExit(f"error: {row} given twice")
            found = compare_outputs.digest(Path(run_dir) / row)
            if not found["ok"]:
                raise SystemExit(f"error: no prediction artifacts for {row}")
            rows[row] = found["artifacts"]
        sources.append({"rows": names, "source": source})
    manifest = {
        "description": (
            "Step 0 stock off prediction-artifact digests for the drift check "
            "(qualification_protocol.md, E5). Paths are relative to the row's "
            "output directory."
        ),
        "recipe": args.recipe,
        "sources": sources,
        "rows": rows,
    }
    args.out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"{len(rows)} rows, {sum(len(v) for v in rows.values())} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
