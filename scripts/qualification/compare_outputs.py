#!/usr/bin/env python
"""List, hash and compare OpenDDE prediction artifacts.

Commands:
    digest OUT_DIR              hash every prediction artifact under OUT_DIR
    compare A_DIR B_DIR         compare two output trees byte for byte
    scan OUT_DIR [--term T...]  look for run-specific values in artifacts

Prediction artifacts are the files the dumper writes under
``<job>/seed_<seed>/predictions/``: the structure CIFs and the confidence
JSONs. Everything else in an output tree (logs, error reports) is listed
separately and never compared.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ARTIFACT_PATTERNS = ("*.cif", "*.json")
TIMESTAMP = re.compile(
    rb"\b(19|20)\d\d-[01]\d-[0-3]\d([T ][0-2]\d:[0-5]\d)?|\b[0-2]\d:[0-5]\d:[0-5]\d\b"
)


def artifacts(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for predictions in sorted(root.rglob("predictions")):
        if not predictions.is_dir():
            continue
        for pattern in ARTIFACT_PATTERNS:
            for path in sorted(predictions.glob(pattern)):
                found[str(path.relative_to(root))] = path
    return found


def other_files(root: Path) -> list[str]:
    listed = set(artifacts(root))
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and str(p.relative_to(root)) not in listed
    )


def digest(root: Path) -> dict[str, Any]:
    files = artifacts(root)
    return {
        "ok": bool(files),
        "artifacts": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in files.items()
        },
        "other_files": other_files(root),
    }


def compare(a: Path, b: Path) -> dict[str, Any]:
    da, db = digest(a)["artifacts"], digest(b)["artifacts"]
    differing = sorted(k for k in set(da) & set(db) if da[k] != db[k])
    return {
        "ok": bool(da) and set(da) == set(db) and not differing,
        "compared": len(set(da) & set(db)),
        "only_in_a": sorted(set(da) - set(db)),
        "only_in_b": sorted(set(db) - set(da)),
        "differing": differing,
    }


def scan(root: Path, terms: list[str]) -> dict[str, Any]:
    hits: dict[str, list[str]] = {}
    needles = [t.encode() for t in terms if t]
    for name, path in artifacts(root).items():
        data = path.read_bytes()
        found = [t.decode() for t in needles if t in data]
        stamps = sorted({m.group(0).decode() for m in TIMESTAMP.finditer(data)})
        if found or stamps:
            hits[name] = found + [f"timestamp-like: {s}" for s in stamps[:5]]
    return {"ok": not hits, "artifacts_scanned": len(artifacts(root)), "hits": hits}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("digest").add_argument("root", type=Path)
    cmp = sub.add_parser("compare")
    cmp.add_argument("a", type=Path)
    cmp.add_argument("b", type=Path)
    sc = sub.add_parser("scan")
    sc.add_argument("root", type=Path)
    sc.add_argument("--term", action="append", default=[])
    args = parser.parse_args()
    if args.command == "digest":
        result = digest(args.root)
    elif args.command == "compare":
        result = compare(args.a, args.b)
    else:
        result = scan(args.root, args.term)
    print(json.dumps(result, indent=1))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
