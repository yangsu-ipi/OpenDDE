#!/usr/bin/env python
"""Record or check digests of the frozen qualification inputs.

Usage:
    python scripts/qualification/freeze_inputs.py write [MANIFEST]
    python scripts/qualification/freeze_inputs.py check [MANIFEST]

Run from the repository root. ``write`` hashes every input JSON named in the
manifest and every MSA or template file those JSONs reference, and stores the
result in ``digests.json`` next to the manifest. ``check`` recomputes the
digests and exits non-zero if any file changed, appeared or disappeared.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

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


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("write", "check"):
        print(__doc__, file=sys.stderr)
        return 2
    manifest_path = Path(
        sys.argv[2]
        if len(sys.argv) > 2
        else "scripts/qualification/inputs/manifest.json"
    )
    digest_path = manifest_path.with_name("digests.json")
    current = digests(manifest_path)
    if sys.argv[1] == "write":
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
