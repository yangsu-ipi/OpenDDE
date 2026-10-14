#!/usr/bin/env python
"""Record or check digests of the frozen qualification inputs.

Usage:
    python scripts/qualification/freeze_inputs.py write [--external | --effective] [MANIFEST]
    python scripts/qualification/freeze_inputs.py check [--external | --effective] [MANIFEST]

Run from the repository root. ``write`` hashes every input JSON named in the
manifest and every MSA or template file those JSONs reference, and stores the
result in ``digests.json`` next to the manifest. ``check`` recomputes the
digests and exits non-zero if any file changed, appeared or disappeared.

With ``--external`` the same is done for the host-side data that rows list in
``external_data`` (for example D5's template mmCIF files), stored in
``external_digests.json``. Entries name their location through environment
variables such as ``${OPENDDE_QUAL_DATA}``, which must be set; digest keys
keep the unexpanded form so they are the same on every host.

With ``--effective`` (contract-6) it records, in ``effective_inputs.json``,
the source and expected effective SHA-256 of every input JSON that ``pred``
converts from the legacy ``proteinChain.msa`` format for a row run with
``--use_msa true`` (``provenance.effective_input``, a mirror of the runner's
format-only conversion that never runs OpenDDE on a held-out input). The
inference parser of such a run must read exactly the effective bytes.
"""

import os

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import provenance

PATH_KEYS = ("pairedMsaPath", "unpairedMsaPath", "templatesPath")
DIR_KEYS = ("precomputed_msa_dir",)


def _referenced(value: Any) -> Iterator[Path]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in PATH_KEYS and isinstance(item, str):
                yield Path(item)
            elif key in DIR_KEYS and isinstance(item, str):
                yield from sorted(p for p in Path(item).rglob("*") if p.is_file())
            else:
                yield from _referenced(item)
    elif isinstance(value, list):
        for item in value:
            yield from _referenced(item)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digests(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text())
    files: set[Path] = set()
    for row in manifest["rows"]:
        input_path = Path(row["input"])
        files.add(input_path)
        files.update(_referenced(json.loads(input_path.read_text())))
    out = {}
    for path in sorted(files, key=lambda p: p.as_posix()):
        key = path.as_posix().removeprefix("./")
        if not path.is_file():
            raise SystemExit(f"error: missing input file {key}")
        out[key] = _sha256(path)
    return out


def external_digests(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text())
    out = {}
    for row in manifest["rows"]:
        for entry in row.get("external_data", []):
            root = os.path.expandvars(entry)
            if "$" in root:
                raise SystemExit(f"error: unset environment variable in {entry!r}")
            root_path = Path(root)
            files = (
                sorted(p for p in root_path.rglob("*") if p.is_file())
                if root_path.is_dir()
                else [root_path]
            )
            if not files or not all(p.is_file() for p in files):
                raise SystemExit(f"error: missing external data {entry}")
            for path in files:
                key = (
                    entry
                    if path == root_path
                    else f"{entry}/{path.relative_to(root_path).as_posix()}"
                )
                out[key] = _sha256(path)
    return out


def _uses_msa(row: dict, manifest: dict) -> bool:
    flags = [*manifest["common_flags"], *row["flags"]]
    found = [i for i, a in enumerate(flags[:-1]) if a == "--use_msa"]
    return True if not found else flags[found[-1] + 1].lower() == "true"


def effective_digests(manifest_path: Path) -> dict[str, dict[str, str]]:
    manifest = json.loads(manifest_path.read_text())
    out = {}
    for row in manifest["rows"]:
        if not _uses_msa(row, manifest):
            continue
        source = Path(row["input"]).read_bytes()
        effective, _ = provenance.effective_input(source, use_msa=True)
        if effective != source:
            out[Path(row["input"]).as_posix()] = {
                "source_sha256": provenance.sha256_bytes(source),
                "effective_sha256": provenance.sha256_bytes(effective),
            }
    return dict(sorted(out.items()))


def main() -> int:
    argv = [a for a in sys.argv[1:] if a not in ("--external", "--effective")]
    external = "--external" in sys.argv[1:]
    effective = "--effective" in sys.argv[1:]
    if not argv or argv[0] not in ("write", "check") or (external and effective):
        print(__doc__, file=sys.stderr)
        return 2
    manifest_path = Path(
        argv[1] if len(argv) > 1 else "scripts/qualification/inputs/manifest.json"
    )
    if external:
        digest_path = manifest_path.with_name("external_digests.json")
        current = external_digests(manifest_path)
    elif effective:
        digest_path = manifest_path.with_name("effective_inputs.json")
        current = effective_digests(manifest_path)
    else:
        digest_path = manifest_path.with_name("digests.json")
        current = digests(manifest_path)
    if argv[0] == "write":
        digest_path.write_text(json.dumps(current, indent=1) + "\n")
        print(f"wrote {len(current)} digests to {digest_path}")
        return 0
    frozen = json.loads(digest_path.read_text())
    changed = sorted(
        k for k in frozen.keys() & current.keys() if frozen[k] != current[k]
    )
    missing = sorted(frozen.keys() - current.keys())
    added = sorted(current.keys() - frozen.keys())
    print(
        json.dumps(
            {
                "ok": not (changed or missing or added),
                "files": len(frozen),
                "changed": changed,
                "missing": missing,
                "added": added,
            },
            indent=1,
        )
    )
    return 0 if not (changed or missing or added) else 1


if __name__ == "__main__":
    sys.exit(main())
