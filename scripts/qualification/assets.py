#!/usr/bin/env python
"""Verify, freeze and preflight the private runtime-asset copy (contract-6, protocol E5).

Usage:
    python scripts/qualification/assets.py verify --copy ROOT --source SRC --kalign-source KALIGN
    python scripts/qualification/assets.py freeze --copy ROOT --gpu sm_120 [--reference PATH]
    python scripts/qualification/assets.py preflight --gpu sm_120 [--reference PATH]

The copy holds, under ROOT, what qualification runs select through
``OPENDDE_ROOT_DIR`` and ``OPENDDE_QUAL_KALIGN``: ``checkpoint/opendde.pt``,
``common/components.cif``, ``common/components.cif.rdkit_mol.pkl``,
``common/release_date_cache.json``, ``common/obsolete_to_successor.json`` and
``bin/kalign``. Make it with plain ``cp`` or ``rsync`` without link options.

``verify`` checks that every copied asset is an independent regular file:
not a symlink, resolving inside ROOT, with one hard link, on a device and
inode different from its source in SRC (the shared store) or KALIGN, and
with the same bytes. ``freeze`` records each asset's resolved path and
SHA-256 for one GPU in ``inputs/runtime_assets.json``, the freeze-time
reference ``accept.py`` compares every qualification run with. ``preflight``
checks, before a held-out batch or smoke test, that ``OPENDDE_ROOT_DIR`` and
``OPENDDE_QUAL_KALIGN`` name the frozen copy and every frozen file still
exists with its digest (and Kalign is executable), so a missing or changed
asset stops the batch before it spends seeds. All three exit non-zero on any
failure. A digest proves the bytes at the moment it is taken, under the
protocol's accident-only trust boundary.
"""

import argparse
import json
import os
import stat
import sys
from pathlib import Path

import provenance

REFERENCE = Path(__file__).with_name("inputs") / "runtime_assets.json"
# Logical name (the effective-configuration key) and path under the copy root.
LAYOUT = {
    "checkpoint": "checkpoint/opendde.pt",
    "ccd_components_file": "common/components.cif",
    "ccd_components_rdkit_mol_file": "common/components.cif.rdkit_mol.pkl",
    "release_dates_path": "common/release_date_cache.json",
    "obsolete_pdbs_path": "common/obsolete_to_successor.json",
    "kalign_binary_path": "bin/kalign",
}


def verify(copy: Path, source: Path, kalign_source: Path) -> dict:
    root = Path(os.path.realpath(copy))
    problems = {}
    for name, rel in LAYOUT.items():
        path = copy / rel
        origin = kalign_source if name == "kalign_binary_path" else source / rel
        found = []
        try:
            info = os.lstat(path)
        except OSError as error:
            problems[name] = [f"missing: {error}"]
            continue
        if stat.S_ISLNK(info.st_mode):
            found.append("symlink")
        elif not stat.S_ISREG(info.st_mode):
            found.append("not a regular file")
        if not Path(os.path.realpath(path)).is_relative_to(root):
            found.append("resolves outside the copy")
        if info.st_nlink != 1:
            found.append(f"{info.st_nlink} hard links")
        try:
            other = os.stat(origin)
            if (other.st_dev, other.st_ino) == (info.st_dev, info.st_ino):
                found.append("same device and inode as its source")
            elif not found and provenance.sha256_file(path) != provenance.sha256_file(
                origin
            ):
                found.append("bytes differ from its source")
        except OSError as error:
            found.append(f"source unreadable: {error}")
        if name == "kalign_binary_path" and not os.access(path, os.X_OK):
            found.append("not executable")
        if found:
            problems[name] = found
    return {"ok": not problems, "copy": str(root), "problems": problems}


def freeze(copy: Path, gpu: str, reference: Path) -> dict:
    data = json.loads(reference.read_text()) if reference.is_file() else {}
    root = os.path.realpath(copy)
    assets = {
        name: {
            "path": os.path.realpath(copy / rel),
            "sha256": provenance.sha256_file(copy / rel),
        }
        for name, rel in LAYOUT.items()
    }
    data.setdefault("gpus", {})[gpu] = {"root": root, "assets": assets}
    reference.write_text(json.dumps(data, indent=1) + "\n")
    return data["gpus"][gpu]


def preflight(gpu: str, reference: Path, environ=os.environ) -> dict:
    entry = (json.loads(reference.read_text()).get("gpus") or {}).get(gpu)
    if not entry:
        return {"ok": False, "problems": {"reference": f"no frozen copy for {gpu}"}}
    problems = {}
    root = environ.get("OPENDDE_ROOT_DIR")
    if root is None or os.path.realpath(root) != entry["root"]:
        problems["OPENDDE_ROOT_DIR"] = [root, entry["root"]]
    kalign = environ.get("OPENDDE_QUAL_KALIGN")
    frozen_kalign = entry["assets"]["kalign_binary_path"]["path"]
    if kalign is None or os.path.realpath(kalign) != frozen_kalign:
        problems["OPENDDE_QUAL_KALIGN"] = [kalign, frozen_kalign]
    for name, asset in entry["assets"].items():
        state = provenance.file_state(asset["path"])
        if state["sha256"] != asset["sha256"]:
            problems[name] = "missing or changed"
        elif name == "kalign_binary_path" and not os.access(asset["path"], os.X_OK):
            problems[name] = "not executable"
    return {"ok": not problems, "gpu": gpu, "problems": problems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("verify", "freeze", "preflight"))
    parser.add_argument("--copy", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--kalign-source", type=Path)
    parser.add_argument("--gpu")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    if args.command == "verify":
        if not (args.copy and args.source and args.kalign_source):
            parser.error("verify needs --copy, --source and --kalign-source")
        result = verify(args.copy, args.source, args.kalign_source)
    elif args.command == "freeze":
        if not (args.copy and args.gpu):
            parser.error("freeze needs --copy and --gpu")
        result = {"ok": True, **freeze(args.copy, args.gpu, args.reference)}
    else:
        if not args.gpu:
            parser.error("preflight needs --gpu")
        result = preflight(args.gpu, args.reference)
    print(json.dumps(result, indent=1))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
