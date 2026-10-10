#!/usr/bin/env python
"""Run frozen qualification rows through the stock CLI wrapper.

Usage:
    python scripts/qualification/run_rows.py --rows D1,D2,... --out DIR
        [--seeds 101,102] [--draw-trace] [--phase-timing] [--manifest PATH]
        [-- EXTRA_PRED_FLAGS ...]

Run from the repository root. Each row runs as
``run_opendde.py [--draw-trace ...] --set sorted_by_ranking_score=false
[--set <row overrides>] -- pred -i <input> -o DIR/<row> <common flags>
<row flags> --seeds <seeds> <extra flags>`` with the row's environment added.
Extra flags come last, so ``-- --dtype fp32 --deterministic true`` overrides
the common flags. Held-out rows must be given explicit held-out seeds.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WRAPPER = Path(__file__).with_name("run_opendde.py")


def main() -> int:
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds")
    parser.add_argument("--draw-trace", action="store_true")
    parser.add_argument("--phase-timing", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("scripts/qualification/inputs/manifest.json"),
    )
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    rows = {r["id"]: r for r in manifest["rows"]}
    failures = 0
    for row_id in args.rows.split(","):
        row = rows[row_id]
        if row["role"] != "development" and not args.seeds:
            raise SystemExit(f"error: {row_id} is held-out; pass held-out --seeds")
        seeds = args.seeds or ",".join(str(s) for s in manifest["development_seeds"])
        wrapper = [sys.executable, str(WRAPPER)]
        if args.draw_trace:
            wrapper += ["--draw-trace", str(args.out / f"{row_id}.trace.jsonl")]
        if args.phase_timing:
            wrapper += ["--phase-timing", str(args.out / f"{row_id}.phases.json")]
        wrapper += ["--set", "sorted_by_ranking_score=false"]
        for override in row.get("overrides", []):
            wrapper += ["--set", override]
        cmd = wrapper + [
            "--",
            "pred",
            "-i",
            row["input"],
            "-o",
            str(args.out / row_id),
            *manifest["common_flags"],
            *row["flags"],
            "--seeds",
            seeds,
            *extra,
        ]
        env = {**os.environ, **row.get("env", {})}
        args.out.mkdir(parents=True, exist_ok=True)
        print(f"=== {row_id}: {' '.join(cmd[2:])}", flush=True)
        if row.get("env"):
            print(f"    env {row['env']}", flush=True)
        start = time.time()
        with open(args.out / f"{row_id}.log", "w") as log:
            code = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"    exit {code} in {time.time() - start:.0f}s", flush=True)
        failures += code != 0
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
