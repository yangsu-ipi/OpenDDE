#!/usr/bin/env python
"""Exact transfer comparison of tier 1 captures and prediction artifacts (protocol E6).

Usage:
    python scripts/porting/tier1_payload_compare.py declare --rows D1,...,D15
        --contrasts k,fp32 [--exclude D4:fp32] [--full-runs] [--full-only ln]
        [--manifest PATH] --out INVENTORY.json
    python scripts/porting/tier1_payload_compare.py compare --inventory INVENTORY.json
        --a ROOT_A --b ROOT_B [--relocate-a OLD=NEW ...] [--relocate-b OLD=NEW ...]
        --out REPORT.json

Run from the repository root. This helper is deliberately outside
``scripts/qualification``: it checks whether archived calibration evidence
transfers to a replacement host before an sm_80 freeze, and is not part of the
frozen acceptance contract. It reuses the frozen ``accept.expected_artifacts``
and ``compare_outputs.artifacts`` without changing them.

``declare`` writes the expected inventory from the frozen input manifest
alone, never from files on disk: development rows only, their input job
lists, the development seeds and the recipes named. Each ROOT is laid out as
``run_rows.py`` writes calibration runs:

- ``cap_off/<row>/<job>/seed_<s>/tier1.pt``: reference ``off`` captures, once;
- ``cap_<c>/...``: each contrast's captures, replayed from ``cap_off``;
- ``t1_off/``, ``t1_<c>/``: the tier 1 runs (one sample), with run records;
- with ``--full-runs``, ``off/`` and ``<c>/``: full calibration runs (the
  manifest's sample count); ``--full-only`` adds full runs without captures.

``--exclude ROW:CONTRAST`` removes only that contrast's captures and artifacts
for that row (for example D4's fp32 gap); the row's reference ``off`` stays
required.

``compare`` requires each side's inventory to equal the declaration exactly
(missing or unexpected files on either side fail, including the same file
missing from both sides). For every declared capture it requires the same
recursive keys, tensor shapes and dtypes and bit-identical tensor values in
logical element order (strides and storage layout may differ), exact equality
of every non-tensor value (floats bit for bit), finite values in every
floating tensor, and the role's structure (``off``: ``head``, no
``replay_source``; contrast: ``head_fixed`` and ``replay_source``). The only
field allowed to differ is ``replay_source``: on each side it must name, after
that side's ``--relocate`` prefixes, the same row/job/seed's reference capture
in that side's ``cap_off``, and the tier 1 run records must name the same
capture and replay directories. A contrast capture passes only if its
reference capture passes. Prediction artifacts must be byte-identical. Every
compared file's SHA-256 is reported; a file-hash difference with matching
payloads is a serialization-only difference, not numerical drift.

Exit status 0 means every check passed; 1 means at least one failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qualification"))

import accept  # noqa: E402  (frozen; only expected_artifacts is used)
import compare_outputs  # noqa: E402

MANIFEST = Path("scripts/qualification/inputs/manifest.json")
TIER1_SAMPLES = 1
SEED_DIR = re.compile(r"seed_(\d+)")
ROLE_KEYS = {
    "off": {"required": ("head",), "forbidden": ("head_fixed", "replay_source")},
    "contrast": {"required": ("head_fixed", "replay_source"), "forbidden": ("head",)},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sample_count(manifest: dict) -> int:
    flags = manifest["common_flags"]
    return int(flags[flags.index("--sample") + 1])


# ---------------------------------------------------------------- declare


def declare(
    manifest_path: Path,
    rows: list[str],
    contrasts: list[str],
    exclude: list[tuple[str, str]],
    full_runs: bool,
    full_only: list[str],
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    table = {r["id"]: r for r in manifest["rows"]}
    seeds = [int(s) for s in manifest["development_seeds"]]
    for row in rows:
        if row not in table:
            raise SystemExit(f"error: {row} is not in the manifest")
        if table[row].get("role") != "development":
            raise SystemExit(f"error: {row} is not a development row")
    for name in contrasts + full_only:
        if name == "off" or not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise SystemExit(f"error: bad contrast name {name!r}")
    if len(set(contrasts + full_only)) != len(contrasts + full_only):
        raise SystemExit("error: a contrast is named twice")
    for row, name in exclude:
        if row not in rows or name not in contrasts + full_only:
            raise SystemExit(f"error: exclusion {row}:{name} names no declared recipe")
    excluded = set(exclude)

    jobs = {
        row: [j["name"] for j in json.loads(Path(table[row]["input"]).read_text())]
        for row in rows
    }
    captures = []
    for role in ["off", *contrasts]:
        for row in rows:
            if (row, role) in excluded:
                continue
            for job in jobs[row]:
                for seed in seeds:
                    captures.append(
                        {"role": role, "row": row, "job": job, "seed": seed}
                    )

    n_full = _sample_count(manifest)
    runs = [(f"t1_{name}", name, TIER1_SAMPLES) for name in ["off", *contrasts]]
    if full_runs:
        runs += [(name, name, n_full) for name in ["off", *contrasts, *full_only]]
    artifacts = {}
    for run, name, n_sample in runs:
        artifacts[run] = sorted(
            f"{row}/{path}"
            for row in rows
            if (row, name) not in excluded
            for path in accept.expected_artifacts(jobs[row], seeds, n_sample)
        )
    if not captures or not any(artifacts.values()):
        raise SystemExit("error: the declaration is empty")
    return {
        "description": "Expected tier 1 capture and prediction-artifact inventory "
        "(protocol E6), declared from the frozen manifest, never from files on disk.",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "rows": rows,
        "seeds": seeds,
        "contrasts": contrasts,
        "full_runs": full_runs,
        "full_only": full_only,
        "exclusions": [{"row": r, "contrast": c} for r, c in exclude],
        "samples": {"tier1": TIER1_SAMPLES, "full": n_full},
        "counts": {
            "captures": {
                role: sum(1 for c in captures if c["role"] == role)
                for role in ["off", *contrasts]
            },
            "artifacts": {run: len(paths) for run, paths in artifacts.items()},
        },
        "captures": captures,
        "artifacts": artifacts,
    }


# ---------------------------------------------------------------- payloads


def _bits(t: torch.Tensor) -> torch.Tensor:
    flat = t.detach().contiguous().reshape(-1)
    if flat.dtype == torch.bool:
        return flat.to(torch.uint8)
    return flat.view(torch.uint8)


def payload_diff(a: Any, b: Any, path: str = "") -> list[str]:
    """Differences between two capture payloads; empty means identical."""
    where = path or "<root>"
    if type(a) is not type(b):
        return [f"{where}: type {type(a).__name__} vs {type(b).__name__}"]
    if isinstance(a, dict):
        out = []
        ka, kb = set(a), set(b)
        out += [f"{path}/{k}: missing on b" for k in sorted(ka - kb, key=str)]
        out += [f"{path}/{k}: missing on a" for k in sorted(kb - ka, key=str)]
        for k in sorted(ka & kb, key=str):
            if path == "" and k == "replay_source":
                continue  # compared through provenance, not as a value
            out += payload_diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{where}: length {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += payload_diff(x, y, f"{path}[{i}]")
        return out
    if isinstance(a, torch.Tensor):
        if a.layout != torch.strided or b.layout != torch.strided:
            return [f"{where}: non-strided tensor"]
        if a.dtype != b.dtype:
            return [f"{where}: dtype {a.dtype} vs {b.dtype}"]
        if tuple(a.shape) != tuple(b.shape):
            return [f"{where}: shape {tuple(a.shape)} vs {tuple(b.shape)}"]
        if not torch.equal(_bits(a), _bits(b)):
            return [f"{where}: tensor bits differ"]
        return []
    if isinstance(a, float):
        if struct.pack("<d", a) != struct.pack("<d", b):
            return [f"{where}: float bits differ ({a!r} vs {b!r})"]
        return []
    if a is None or isinstance(a, (bool, int, str, bytes)):
        return [] if a == b else [f"{where}: {a!r} vs {b!r}"]
    return [f"{where}: unsupported value type {type(a).__name__}"]


def nonfinite(obj: Any, path: str = "") -> list[str]:
    """Floating or complex tensors holding NaN or infinity."""
    if isinstance(obj, dict):
        return [
            m for k in sorted(obj, key=str) for m in nonfinite(obj[k], f"{path}/{k}")
        ]
    if isinstance(obj, (list, tuple)):
        return [m for i, x in enumerate(obj) for m in nonfinite(x, f"{path}[{i}]")]
    if isinstance(obj, torch.Tensor) and (obj.is_floating_point() or obj.is_complex()):
        bad = int((~torch.isfinite(obj)).sum())
        if bad:
            return [f"{path or '<root>'}: {bad} non-finite value(s)"]
    return []


def structure_problems(record: Any, role: str) -> list[str]:
    if not isinstance(record, dict):
        return ["capture is not a dict"]
    keys = ROLE_KEYS["off" if role == "off" else "contrast"]
    out = [f"missing {k}" for k in keys["required"] if k not in record]
    out += [f"unexpected {k} for role {role}" for k in keys["forbidden"] if k in record]
    if role != "off" and not isinstance(record.get("replay_source"), (str, type(None))):
        out.append("replay_source is not a string")
    return out


# ---------------------------------------------------------------- provenance


def parse_relocations(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for item in items:
        old, sep, new = item.partition("=")
        if not sep or not old or not new:
            raise SystemExit(f"error: --relocate needs OLD=NEW, got {item!r}")
        out.append((old.rstrip("/"), new.rstrip("/")))
    return sorted(out, key=lambda p: len(p[0]), reverse=True)


def relocate(path: str, maps: list[tuple[str, str]]) -> str:
    for old, new in maps:
        if path == old or path.startswith(old + "/"):
            return new + path[len(old) :]
    return path


def _same(path: str | None, expected: Path, maps) -> bool:
    if path is None:
        return False
    return Path(relocate(str(path), maps)).resolve() == expected.resolve()


def provenance_problems(
    root: Path, maps, role: str, row: str, job: str, seed: int, record: dict
) -> list[str]:
    out = []
    reference = root / "cap_off" / row / job / f"seed_{seed}" / "tier1.pt"
    if role != "off" and not _same(record.get("replay_source"), reference, maps):
        out.append(
            f"replay_source {record.get('replay_source')!r} does not name "
            f"{reference} after relocation"
        )
    run_record = root / f"t1_{role}" / f"{row}.run.json"
    try:
        wrapper = json.loads(run_record.read_text())
    except (OSError, ValueError) as exc:
        return out + [f"run record {run_record} unreadable: {exc}"]
    if wrapper.get("exit_code") != 0:
        out.append(f"run record {run_record} exit code {wrapper.get('exit_code')!r}")
    wrapper = wrapper.get("wrapper") or {}
    if not _same(wrapper.get("tier1_capture"), root / f"cap_{role}" / row, maps):
        out.append(f"run record names capture dir {wrapper.get('tier1_capture')!r}")
    if role == "off":
        if wrapper.get("tier1_replay") is not None:
            out.append(
                f"reference run record names a replay dir {wrapper['tier1_replay']!r}"
            )
    elif not _same(wrapper.get("tier1_replay"), root / "cap_off" / row, maps):
        out.append(f"run record names replay dir {wrapper.get('tier1_replay')!r}")
    return out


# ---------------------------------------------------------------- inventory


def found_captures(root: Path, roles: list[str]) -> tuple[set, list[str]]:
    keys, odd = set(), []
    for role in roles:
        base = root / f"cap_{role}"
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_dir():
                continue
            parts = path.relative_to(base).parts
            m = SEED_DIR.fullmatch(parts[2]) if len(parts) == 4 else None
            if m is None or parts[3] != "tier1.pt":
                odd.append(str(path.relative_to(root)))
                continue
            keys.add((role, parts[0], parts[1], int(m.group(1))))
    return keys, odd


def found_artifacts(root: Path, run: str) -> set[str]:
    base = root / run
    return set(compare_outputs.artifacts(base)) if base.is_dir() else set()


# ---------------------------------------------------------------- compare


def helper_identity() -> dict[str, Any]:
    me = Path(__file__).resolve()

    def git(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(me.parent), *args],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "path": str(me),
        "sha256": sha256_file(me),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", str(me))),
    }


def compare(
    inventory: dict[str, Any], root_a: Path, root_b: Path, maps_a, maps_b
) -> dict[str, Any]:
    failures: list[str] = []
    roles = ["off", *inventory["contrasts"]]
    expected = {
        (c["role"], c["row"], c["job"], int(c["seed"])) for c in inventory["captures"]
    }
    if not expected or not any(inventory["artifacts"].values()):
        failures.append("declaration is empty")

    sides = {"a": (root_a, maps_a), "b": (root_b, maps_b)}
    for side, (root, _) in sides.items():
        keys, odd = found_captures(root, roles)
        for key in sorted(expected - keys):
            failures.append(f"{side}: missing capture {'/'.join(map(str, key))}")
        for key in sorted(keys - expected):
            failures.append(f"{side}: unexpected capture {'/'.join(map(str, key))}")
        for path in odd:
            failures.append(f"{side}: unexpected file {path}")

    entries, passed = [], {}
    order = sorted(expected, key=lambda k: (k[0] != "off", k))  # references first
    for role, row, job, seed in order:
        rel = Path(f"cap_{role}") / row / job / f"seed_{seed}" / "tier1.pt"
        pa, pb = root_a / rel, root_b / rel
        entry = {"role": role, "row": row, "job": job, "seed": seed, "problems": []}
        if not (pa.is_file() and pb.is_file()):
            entry["problems"].append("capture missing")
        else:
            entry["a_sha256"], entry["b_sha256"] = sha256_file(pa), sha256_file(pb)
            entry["file_hash_equal"] = entry["a_sha256"] == entry["b_sha256"]
            ra = torch.load(pa, map_location="cpu", weights_only=True)
            rb = torch.load(pb, map_location="cpu", weights_only=True)
            for side, rec, (root, maps) in (
                ("a", ra, sides["a"]),
                ("b", rb, sides["b"]),
            ):
                entry["problems"] += [
                    f"{side}: {m}" for m in structure_problems(rec, role)
                ]
                entry["problems"] += [f"{side}: {m}" for m in nonfinite(rec)]
                if isinstance(rec, dict):
                    entry["problems"] += [
                        f"{side}: {m}"
                        for m in provenance_problems(
                            root, maps, role, row, job, seed, rec
                        )
                    ]
            diff = payload_diff(ra, rb)
            entry["payload_equal"] = not diff
            entry["problems"] += diff
            del ra, rb
        if role != "off" and not passed.get(("off", row, job, seed), False):
            entry["problems"].append("reference capture did not pass")
        passed[(role, row, job, seed)] = not entry["problems"]
        failures += [f"{role}/{row}/{job}/seed_{seed}: {m}" for m in entry["problems"]]
        entries.append(entry)

    artifacts = {}
    for run, paths in inventory["artifacts"].items():
        want = set(paths)
        result = {"expected": len(want), "differing": []}
        for side, (root, _) in sides.items():
            have = found_artifacts(root, run)
            for p in sorted(want - have):
                failures.append(f"{side}: missing artifact {run}/{p}")
            for p in sorted(have - want):
                failures.append(f"{side}: unexpected artifact {run}/{p}")
        for p in sorted(want):
            fa, fb = root_a / run / p, root_b / run / p
            if fa.is_file() and fb.is_file() and sha256_file(fa) != sha256_file(fb):
                result["differing"].append(p)
                failures.append(f"artifact {run}/{p}: bytes differ")
        artifacts[run] = result

    return {
        "ok": not failures,
        "failures": failures,
        "summary": {
            "captures_expected": len(expected),
            "captures_passed": sum(passed.values()),
            "serialization_only_differences": sum(
                1
                for e in entries
                if e.get("payload_equal") and e.get("file_hash_equal") is False
            ),
            "artifacts_expected": sum(r["expected"] for r in artifacts.values()),
            "artifacts_differing": sum(len(r["differing"]) for r in artifacts.values()),
        },
        "captures": entries,
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("declare")
    d.add_argument("--rows", required=True)
    d.add_argument("--contrasts", required=True)
    d.add_argument("--exclude", action="append", default=[])
    d.add_argument("--full-runs", action="store_true")
    d.add_argument("--full-only", default="")
    d.add_argument("--manifest", type=Path, default=MANIFEST)
    d.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--inventory", type=Path, required=True)
    c.add_argument("--a", type=Path, required=True)
    c.add_argument("--b", type=Path, required=True)
    c.add_argument("--relocate-a", action="append", default=[])
    c.add_argument("--relocate-b", action="append", default=[])
    c.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "declare":
        exclude = []
        for item in args.exclude:
            row, sep, name = item.partition(":")
            if not sep:
                raise SystemExit(f"error: --exclude needs ROW:CONTRAST, got {item!r}")
            exclude.append((row, name))
        split = lambda s: [x for x in s.split(",") if x]  # noqa: E731
        inventory = declare(
            args.manifest,
            split(args.rows),
            split(args.contrasts),
            exclude,
            args.full_runs,
            split(args.full_only),
        )
        if args.out.exists():
            raise SystemExit(
                f"error: {args.out} exists; declarations are not rewritten"
            )
        args.out.write_text(json.dumps(inventory, indent=1) + "\n")
        print(json.dumps(inventory["counts"], indent=1))
        return 0

    inventory = json.loads(args.inventory.read_text())
    report = compare(
        inventory,
        args.a,
        args.b,
        parse_relocations(args.relocate_a),
        parse_relocations(args.relocate_b),
    )
    report["inventory"] = {
        "path": str(args.inventory),
        "sha256": sha256_file(args.inventory),
    }
    report["roots"] = {"a": str(args.a), "b": str(args.b)}
    report["relocations"] = {"a": args.relocate_a, "b": args.relocate_b}
    report["helper"] = helper_identity()
    report["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"ok": report["ok"], **report["summary"]}, indent=1))
    for line in report["failures"][:20]:
        print("FAIL", line)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
