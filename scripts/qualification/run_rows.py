#!/usr/bin/env python
"""Run frozen qualification rows through the stock CLI wrapper.

Usage:
    python scripts/qualification/run_rows.py --rows D1,D2,... --out DIR
        [--seeds 101,102] [--draw-trace] [--phase-timing] [--manifest PATH]
        [--tier1-capture DIR [--tier1-replay DIR]]
        [-- EXTRA_PRED_FLAGS ...]

Run from the repository root. Each row runs as
``run_opendde.py [--draw-trace ...] --run-record DIR/<row>.run.json
--set sorted_by_ranking_score=false
[--set <row overrides>] -- pred -i <input> -o DIR/<row> <common flags>
<row flags> --seeds <seeds> <extra flags>`` with the row's environment added.
Extra flags come last, so ``-- --dtype fp32 --deterministic true`` overrides
the common flags. Held-out rows must be given explicit held-out seeds.

Row flags and overrides may name host-specific locations through environment
variables (``${OPENDDE_QUAL_DATA}``, ``${OPENDDE_QUAL_KALIGN}``), and ``{out}``
stands for the row's output directory. An unset variable is an error.

``--tier1-capture DIR`` passes ``DIR/<row>`` to ``run_opendde.py
--tier1-capture``, and ``--tier1-replay DIR`` passes ``DIR/<row>`` as the
recorded reference for the fixed-input confidence-head pass.

``build_command`` is the one construction of a row's command; the acceptance
command (``accept.py``) rebuilds each recorded run's command with it.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WRAPPER = Path(__file__).with_name("run_opendde.py")


VARIABLE = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _expand(value: str, out: Path, environ=os.environ) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1) or match.group(2)
        return environ.get(name, match.group(0))

    expanded = VARIABLE.sub(replace, value).replace("{out}", str(out))
    if "$" in expanded:
        raise SystemExit(f"error: unset environment variable in {value!r}")
    return expanded


def build_command(
    row: dict, manifest: dict, row_out: Path, seeds: str, extra=(), environ=os.environ
) -> tuple[list[str], list[str], dict[str, str]]:
    """Return a row's config overrides, ``pred`` arguments and added environment."""
    overrides = ["sorted_by_ranking_score=false"]
    overrides += [_expand(o, row_out, environ) for o in row.get("overrides", [])]
    cli_args = [
        "pred",
        "-i",
        row["input"],
        "-o",
        str(row_out),
        *manifest["common_flags"],
        *(_expand(flag, row_out, environ) for flag in row["flags"]),
        "--seeds",
        seeds,
        *extra,
    ]
    return overrides, cli_args, dict(row.get("env", {}))


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
    parser.add_argument("--tier1-capture", type=Path)
    parser.add_argument("--tier1-replay", type=Path)
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
        if args.tier1_capture:
            wrapper += ["--tier1-capture", str(args.tier1_capture / row_id)]
        if args.tier1_replay:
            wrapper += ["--tier1-replay", str(args.tier1_replay / row_id)]
        wrapper += ["--run-record", str(args.out / f"{row_id}.run.json")]
        row_out = args.out / row_id
        overrides, cli_args, row_env = build_command(
            row, manifest, row_out, seeds, extra
        )
        for override in overrides:
            wrapper += ["--set", override]
        cmd = wrapper + ["--", *cli_args]
        env = {**os.environ, **row_env}
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
