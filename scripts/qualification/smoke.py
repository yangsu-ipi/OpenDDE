#!/usr/bin/env python
"""Per-GPU D1 stock smoke test at a contract tag (qualification_protocol.md, E5 and E6).

Usage:
    python scripts/qualification/smoke.py --gpu GPU --out DIR

Run on the GPU from the root of a clean checkout of the contract tag that the
GPU's frozen block in ``inputs/acceptance.json`` names, with
``OPENDDE_ROOT_DIR``, ``OPENDDE_QUAL_DATA`` and ``OPENDDE_QUAL_KALIGN`` set for
the GPU's frozen asset copy. A GPU may not rehearse or draw under a tag until
this test has passed there: ``accept.py --receipt`` (through ``qual_gate.py
--write-receipt --smoke``) writes a rehearsal receipt only with a passing
record of the same GPU at the approved tag's commit, the receipt records the
record's sha256, GPU, tag and tag commit, and ``accept.py`` checks them at the
draw.

Preconditions, checked before any run: the checkout is clean and at the tag's
commit, ``DIR`` does not exist, the GPU's runtime-asset preflight passes and
the frozen input digests match. Then, into the new ``DIR``, stock ``off`` of
D1 at the development seeds with the recipe's ``extra`` arguments, traced
(``off``) and untraced (``repeat``); a tier 1 ``off`` capture (``t1_off``,
``cap_off``) and a capture replaying it (``t1_rep``, ``cap_rep``) with
``tier1_extra``; a second launch into ``off``, which must be refused;
``tier1_compare.py`` of the two captures (``t1.json``); and ``tier2_gates.py
measure --draw`` of ``off`` against ``repeat`` with a scratch log
(``measure.json``). ``accept.py``'s per-run and per-row checks are applied to
them, the repeat and tier 1 replay as candidate runs without candidate
arguments: run records, provenance and commands, traces, artifacts, the
repeat's byte identity, drift against the GPU's step 0 baseline, tier 1 and
tier 2 against the GPU's limits, measurement provenance, frozen limit hashes
and the contract checks. No held-out input is run and no limit is changed.

Writes ``DIR/smoke.json``: ``gpu``, ``contract_tag``, ``tag_commit``,
``checkout_commit``, the commands and their exit codes, every check, ``failed``
and ``ok``. Exit status 0 when every check passes, 1 otherwise.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import accept
import assets
import tier2_gates
import trace_tools

ROW = "D1"
SCRIPTS = "scripts/qualification"


def _commands(out: Path, recipe: dict[str, Any]) -> list[tuple[str, list[str], bool]]:
    """(name, arguments, whether the command must succeed), run in order."""
    py = [sys.executable]
    rows = py + [f"{SCRIPTS}/run_rows.py", "--rows", ROW]
    extra, t1 = list(recipe["extra"]), list(recipe["tier1_extra"])
    return [
        ("off", rows + ["--out", str(out / "off"), "--draw-trace", "--", *extra], True),
        ("repeat", rows + ["--out", str(out / "repeat"), "--", *extra], True),
        (
            "tier1 off",
            rows
            + ["--out", str(out / "t1_off"), "--tier1-capture", str(out / "cap_off")]
            + ["--", *t1],
            True,
        ),
        (
            "tier1 replay",
            rows
            + ["--out", str(out / "t1_rep"), "--tier1-capture", str(out / "cap_rep")]
            + ["--tier1-replay", str(out / "cap_off"), "--", *t1],
            True,
        ),
        ("rerun into off", rows + ["--out", str(out / "off"), "--", *extra], False),
        (
            "tier1 compare",
            py
            + [
                f"{SCRIPTS}/tier1_compare.py",
                str(out / "cap_off"),
                str(out / "cap_rep"),
            ]
            + ["--rows", ROW, "--out", str(out / "t1.json")],
            True,
        ),
        (
            "tier2 measure",
            py
            + [f"{SCRIPTS}/tier2_gates.py", "measure", str(out / "off")]
            + [str(out / "repeat"), "--rows", ROW, "--draw", "smoke"]
            + ["--log", str(out / "log.json"), "--out", str(out / "measure.json")],
            True,
        ),
    ]


def _run(cmd: list[str], cwd: Path, log: Path) -> int:
    with open(log, "w") as f:
        return subprocess.run(
            cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT
        ).returncode


def smoke(gpu: str, out: Path, root: Path = accept.INPUTS, runner=_run) -> dict:
    checks = accept.Checks()
    acc = json.loads((root / "acceptance.json").read_text())
    block = acc["gpus"].get(gpu)
    record: dict[str, Any] = {"gpu": gpu, "dir": str(out), "commands": []}

    def finish():
        record["checks"] = checks.items
        record["failed"] = [c["check"] for c in checks.failed()]
        record["ok"] = not record["failed"]
        return record

    if not checks.add("GPU declared", block is not None, gpu):
        return finish()
    frozen = block.get("frozen") or {}
    tag = frozen.get("contract_tag")
    repo = accept.Repo.find(root)
    head = repo.commit("HEAD") if repo else None
    tagged = repo.commit(f"refs/tags/{tag}") if repo and tag else None
    record.update(contract_tag=tag, tag_commit=tagged, checkout_commit=head)
    status = repo.git("status", "--porcelain") if repo else None
    checks.add("checkout clean", status == "", status)
    checks.add("checkout is the contract tag commit", tagged and head == tagged, tag)
    checks.add("output directory is new", not out.exists(), str(out))
    pre = assets.preflight(gpu, root / "runtime_assets.json")
    checks.add("runtime-asset preflight", pre["ok"], pre.get("problems"))
    accept.check_digests(checks, root / "manifest.json")
    if checks.failed():
        return finish()

    out.mkdir(parents=True)
    record["created"] = True
    (out / "log.json").write_text(json.dumps({"draws": [{"id": "smoke", "seeds": {}}]}))
    for name, cmd, must_pass in _commands(out, acc["recipe"]):
        code = runner(cmd, repo.path, out / f"{name.replace(' ', '_')}.log")
        record["commands"].append({"name": name, "cmd": cmd, "exit_code": code})
        if must_pass:
            checks.add(f"{name} exit 0", code == 0, code)
        else:
            checks.add(f"{name} refused", code != 0, code)
    if checks.failed():
        return finish()

    manifest = json.loads((root / "manifest.json").read_text())
    log_path = (
        (root / "held_out_log.json")
        .resolve()
        .relative_to(repo.path.resolve())
        .as_posix()
    )
    git = accept.Git(repo, head, head, log_path)
    draw = {"id": "smoke", "code_commit": head, "candidate_args": []}
    refs = accept.load_provenance_references(checks, root, gpu)
    ctx = accept.Context(
        checks, manifest, acc["recipe"], draw, git, block["capability"], refs
    )
    seeds = ctx.seeds(ROW)
    jobs = ctx.jobs(ROW)
    off, rep = out / "off", out / "repeat"
    rec_off = accept.check_run(ctx, "off", off, ROW, seeds, "off")
    rec_rep = accept.check_run(ctx, "repeat", rep, ROW, seeds, "repeat")
    if rec_off is None or rec_rep is None:  # check_run recorded the failure
        return finish()
    meta, records = trace_tools.load(off / f"{ROW}.trace.jsonl")
    checks.add(
        f"{ROW} off trace header matches its run record",
        meta["cli_args"] == rec_off["cli_args"]
        and meta["fingerprint"] == rec_off["fingerprint"],
    )
    accept.check_rollouts(ctx, f"{ROW} off", meta, jobs, seeds, rec_off["cli_args"])
    m = trace_tools.check_manifest(meta, records)
    checks.add(f"{ROW} off draws complete", m["ok"], m["observed"])
    expected = accept.expected_artifacts(jobs, seeds, accept._n_sample(rec_off))
    accept.check_artifacts(ctx, f"{ROW} off", off / ROW, expected, rec_off)
    accept.check_artifacts(ctx, f"{ROW} repeat", rep / ROW, expected, rec_rep)
    accept.check_repeat(ctx, ROW, off, rep, rec_off, rec_rep)
    baseline = json.loads((root / block["baseline"]).read_text())
    accept.check_drift(ctx, ROW, off, baseline)
    t1 = json.loads((out / "t1.json").read_text())
    accept.check_measurement_source(ctx, "smoke tier 1", t1)
    accept.check_tier1_runs(
        ctx,
        {"tier1_off": str(out / "t1_off"), "tier1_cand": str(out / "t1_rep")},
        ROW,
        seeds,
        t1,
    )
    t1_limits = json.loads((root / block["tier1_limits"]).read_text())
    accept.check_tier1(checks, ROW, jobs, seeds, t1, t1_limits)
    t2 = json.loads((out / "measure.json").read_text())
    accept.check_measurement_source(ctx, "smoke tier 2", t2)
    for e in t2["entries"]:
        accept.check_tier2_inputs(
            ctx, e, seeds, [], {"off": rec_off, "cand": rec_rep}, {}
        )
    t2_limits = json.loads((root / block["tier2_limits"]).read_text())
    for r in tier2_gates.evaluate(t2, t2_limits, manifest, [ROW]):
        checks.add(
            f"{r['id']} {r['job']} tier 2 (off against its repeat)",
            r["pass"],
            r["failed"],
        )
    route = block["routes"]["port-chosen"]
    accept.check_entry_keys(ctx, {"name": "smoke", "rows": [ROW]}, route, t1, t2)
    for tier in ("tier1", "tier2"):
        checks.add(
            f"{tier} limits match the frozen hash",
            frozen.get(f"{tier}_limits_sha256")
            == accept.sha256(root / block[f"{tier}_limits"]),
        )
    accept.check_contract(checks, git, root, tag)
    checks.add(
        "every run shares one set of software versions",
        len(ctx.versions) == 1,
        sorted(ctx.versions),
    )
    checks.add(
        "every run used one path and digest per runtime asset",
        bool(ctx.assets) and all(len(v) == 1 for v in ctx.assets.values()),
        {n: len(v) for n, v in ctx.assets.items()},
    )
    return finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = smoke(args.gpu, args.out)
    text = json.dumps(record, indent=1, default=str) + "\n"
    if record.get("created"):
        (args.out / "smoke.json").write_text(text)
    for c in record["checks"]:
        detail = "" if c["ok"] else "  " + json.dumps(c["detail"], default=str)[:300]
        print(("ok   " if c["ok"] else "FAIL ") + c["check"] + detail)
    print(
        f"smoke {'passed' if record['ok'] else 'FAILED'}: {record['gpu']} at "
        f"{record.get('contract_tag')} ({len(record['checks'])} checks, "
        f"{len(record['failed'])} failed)"
    )
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
