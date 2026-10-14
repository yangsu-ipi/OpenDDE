#!/usr/bin/env python
"""Write or check the signature inventory of qualification inputs (contract-6, protocol E5).

Usage:
    python scripts/qualification/inventory.py write --token-counts COUNTS.json [MANIFEST]
    python scripts/qualification/inventory.py check [MANIFEST]

Run from the repository root. The inventory
(``inputs/signature_inventory.json``) is the frozen reference data of the
held-out do-not-run audit (``run_audit.py``). It lists every manifest input
and every repository example JSON (``examples/**/*.json``, which includes
the protected siblings such as ``9lh2_pocket.json`` that are not in
``digests.json``) with its SHA-256, the manifest rows that use it, and each
job's entity signature (``provenance.entity_signature``). From the rows it
derives the sequence families the do-not-run list protects:

- ``held_out_only``: polymer sequences of held-out rows that no development
  row contains (9sat, 9xqn, 7pzb, H1, H8 and H9). An input containing one is
  protected at any seed outside a logged draw.
- ``shared``: polymer sequences held-out rows share with development rows
  (7wux through H10 and D4, 9lh2 through H7 and D8). An input containing one
  must never run at a reserved seed outside a logged draw.
- ``restraint_protected``: shared sequences of a held-out row that carries a
  guidance restraint (9lh2). An input containing one together with a
  restraint is protected at any seed.

It also records the held-out token counts used for disk bounds, as given in
COUNTS.json (``{row: {"tokens": n, "basis": text}}``); ``check`` carries them
over and recomputes everything else.
"""

import argparse
import json
import sys
from pathlib import Path

import provenance

MANIFEST = Path("scripts/qualification/inputs/manifest.json")
GENERATED = ("-update-msa.json", "-final-updated.json")


def candidate_inputs(manifest: dict) -> list[str]:
    paths = {Path(r["input"]).as_posix() for r in manifest["rows"]}
    for path in Path("examples").rglob("*.json"):
        if path.name.endswith(GENERATED) or "msa" in path.parts:
            continue
        paths.add(path.as_posix())
    return sorted(paths)


def _jobs(path: str) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return [j for j in data if isinstance(j, dict)] if isinstance(data, list) else []


def build(manifest: dict, token_counts: dict) -> dict:
    rows_by_input: dict[str, list[str]] = {}
    for row in manifest["rows"]:
        rows_by_input.setdefault(Path(row["input"]).as_posix(), []).append(row["id"])
    files = {}
    for path in candidate_inputs(manifest):
        jobs = _jobs(path)
        if not jobs:
            continue
        files[path] = {
            "sha256": provenance.sha256_file(Path(path)),
            "rows": rows_by_input.get(path, []),
            "jobs": [
                {"name": j.get("name"), "signature": provenance.entity_signature(j)}
                for j in jobs
            ],
        }

    def sequences(role):
        found, restrained = set(), set()
        for row in manifest["rows"]:
            if row["role"] != role:
                continue
            for job in files[Path(row["input"]).as_posix()]["jobs"]:
                if row.get("job") not in (None, job["name"]):
                    continue
                seqs = {p[1] for p in job["signature"]["polymers"]}
                found |= seqs
                if job["signature"]["restraint"]:
                    restrained |= seqs
        return found, restrained

    development, _ = sequences("development")
    held_out, restrained = sequences("held-out")
    shared = held_out & development
    return {
        "description": (
            "Signature inventory of qualification inputs and repository examples "
            "(inventory.py, contract-6). Reference data of the do-not-run audit "
            "(run_audit.py); sequence families as in inventory.py's docstring."
        ),
        "held_out_seed_range": manifest["held_out_seed_range"],
        "families": {
            "held_out_only": sorted(held_out - development),
            "shared": sorted(shared),
            "restraint_protected": sorted(shared & restrained),
        },
        "token_counts": token_counts,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("write", "check"))
    parser.add_argument("manifest", nargs="?", type=Path, default=MANIFEST)
    parser.add_argument("--token-counts", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    path = args.manifest.with_name("signature_inventory.json")
    if args.command == "write":
        if args.token_counts is None:
            parser.error("write needs --token-counts")
        counts = json.loads(args.token_counts.read_text())
        held = sorted(r["id"] for r in manifest["rows"] if r["role"] == "held-out")
        if sorted(counts) != held:
            parser.error(f"token counts must cover exactly {held}")
        path.write_text(json.dumps(build(manifest, counts), indent=1) + "\n")
        print(f"wrote {path}")
        return 0
    frozen = json.loads(path.read_text())
    current = build(manifest, frozen.get("token_counts", {}))
    differing = sorted(
        k for k in frozen.keys() | current.keys() if frozen.get(k) != current.get(k)
    )
    files = sorted(
        f
        for f in frozen.get("files", {}).keys() | current["files"].keys()
        if frozen.get("files", {}).get(f) != current["files"].get(f)
    )
    print(
        json.dumps(
            {"ok": not differing, "differing": differing, "files": files}, indent=1
        )
    )
    return 0 if not differing else 1


if __name__ == "__main__":
    sys.exit(main())
