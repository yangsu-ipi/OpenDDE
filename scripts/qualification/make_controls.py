#!/usr/bin/env python
"""Build negative-control candidate runs from a stock ``off`` run.

Usage:
    python scripts/qualification/make_controls.py OFF_RUN OUT_DIR --kind KIND
        --rows D1,D2,... [--source-row ROW] [--manifest PATH]

Run from the repository root with ``OPENDDE_ROOT_DIR`` set. Each control is a
run directory with the normal layout (``<row>/<job>/seed_<s>/predictions``),
so ``compare_structures.py``, ``reference_metrics.py`` and ``tier2_gates.py``
score it like any candidate. Confidence JSON files are copied unchanged:
these controls test the structure gates, and the tier 1 head test covers
confidence arithmetic. Kinds:

- ``displaced``: in every sample, the smallest polymer chain moves 8 A away
  from the complex centre and turns 20 degrees about its own centre. Rows with
  one polymer chain are skipped.
- ``hinge``: in every sample, the last 30% of the residues of the largest
  polymer chain turn 20 degrees about the hinge residue's anchor atom.
- ``jitter``: every heavy atom moves by Gaussian noise, 0.5 A per coordinate.
- ``collapsed``: every sample of a seed is replaced by that seed's sample 0.
- ``ligand_out``: each gated ligand (6OI, ATR, NAG, PCG) moves 6 A away from
  the complex centre. Rows without one are skipped.
- ``substitute``: the row's samples are taken from ``--source-row`` of the
  same run, for example stock D11 (``rho=0``) as a D12 candidate.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from biotite.structure.io import pdbx

from opendde.data.inference.json_to_feature import SampleDictToFeatures

GATED_LIGANDS = ("6OI", "ATR", "NAG", "PCG")
AXIS = np.array([0.3, 0.5, 0.8])


def rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    t = np.radians(degrees)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k)


def _write(src: Path, dst: Path, coord: np.ndarray) -> None:
    cif = pdbx.CIFFile.read(str(src))
    site = cif.block["atom_site"]
    n = len(site["Cartn_x"].as_array())
    if len(coord) != n:
        raise ValueError(f"{src}: {n} atoms in CIF, {len(coord)} coordinates")
    for k, name in enumerate(("Cartn_x", "Cartn_y", "Cartn_z")):
        site[name] = np.array([f"{v:.4f}" for v in coord[:, k]])
    dst.parent.mkdir(parents=True, exist_ok=True)
    cif.write(str(dst))


def _coords(path: Path) -> np.ndarray:
    block = pdbx.CIFFile.read(str(path)).block
    return pdbx.get_structure(block, model=1, use_author_fields=False).coord.astype(
        np.float64
    )


def _polymer_chains(atoms):
    heavy = atoms.element != "H"
    polymer = heavy & np.isin(atoms.mol_type, ("protein", "dna", "rna"))
    chains = list(dict.fromkeys(atoms.chain_id[polymer].tolist()))
    return polymer, chains


def transform(kind: str, atoms, xyz: np.ndarray, rng) -> np.ndarray | None:
    out = xyz.copy()
    polymer, chains = _polymer_chains(atoms)
    if kind == "displaced":
        if len(chains) < 2:
            return None
        target = min(chains, key=lambda c: int((polymer & (atoms.chain_id == c)).sum()))
        sel = atoms.chain_id == target
        centre, whole = out[sel].mean(0), out[polymer].mean(0)
        direction = centre - whole
        direction /= np.linalg.norm(direction) + 1e-9
        out[sel] = (out[sel] - centre) @ rotation(AXIS, 20).T + centre + 8.0 * direction
        return out
    if kind == "hinge":
        target = max(chains, key=lambda c: int((polymer & (atoms.chain_id == c)).sum()))
        chain_mask = atoms.chain_id == target
        residues = sorted(set(atoms.res_id[chain_mask].tolist()))
        hinge = residues[int(0.7 * len(residues))]
        anchor_name = (
            "CA" if atoms.mol_type[np.where(chain_mask)[0][0]] == "protein" else "C4'"
        )
        pivot_idx = np.where(
            chain_mask & (atoms.res_id == hinge) & (atoms.atom_name == anchor_name)
        )[0]
        pivot = (
            out[pivot_idx[0]]
            if len(pivot_idx)
            else out[chain_mask & (atoms.res_id == hinge)].mean(0)
        )
        sel = chain_mask & (atoms.res_id > hinge)
        out[sel] = (out[sel] - pivot) @ rotation(AXIS, 20).T + pivot
        return out
    if kind == "jitter":
        heavy = atoms.element != "H"
        out[heavy] += rng.normal(scale=0.5, size=(int(heavy.sum()), 3))
        return out
    if kind == "ligand_out":
        whole = out[polymer].mean(0)
        moved = False
        for chain in dict.fromkeys(
            atoms.chain_id[atoms.is_ligand.astype(bool)].tolist()
        ):
            sel = atoms.chain_id == chain
            if not np.isin(atoms.res_name[sel], GATED_LIGANDS).any():
                continue
            direction = out[sel].mean(0) - whole
            direction /= np.linalg.norm(direction) + 1e-9
            out[sel] += 6.0 * direction
            moved = True
        return out if moved else None
    raise ValueError(kind)


def _copy_json(
    src_pred: Path, dst_pred: Path, src_job: str, dst_job: str, k_src: int, k_dst: int
):
    for kind in ("summary_confidence", "full_data"):
        src = src_pred / f"{src_job}_{kind}_sample_{k_src}.json"
        if src.exists():
            shutil.copyfile(src, dst_pred / f"{dst_job}_{kind}_sample_{k_dst}.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("off_run", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--source-row")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("scripts/qualification/inputs/manifest.json"),
    )
    args = parser.parse_args()
    rows = {r["id"]: r for r in json.loads(args.manifest.read_text())["rows"]}
    made = {}
    for row_id in args.rows.split(","):
        row = rows[row_id]
        jobs = json.loads(Path(row["input"]).read_text())
        if args.kind == "substitute":
            source = rows[args.source_row]
            src_jobs = json.loads(Path(source["input"]).read_text())
            if len(src_jobs) != 1 or len(jobs) != 1:
                raise SystemExit("substitute needs single-job rows")
            src_job, dst_job = src_jobs[0]["name"], jobs[0]["name"]
            count = 0
            for seed_dir in sorted(
                (args.off_run / source["id"] / src_job).glob("seed_*")
            ):
                src_pred = seed_dir / "predictions"
                dst_pred = args.out / row_id / dst_job / seed_dir.name / "predictions"
                dst_pred.mkdir(parents=True, exist_ok=True)
                for cif in sorted(src_pred.glob(f"{src_job}_sample_*.cif")):
                    k = int(cif.stem.rsplit("_", 1)[1])
                    shutil.copyfile(cif, dst_pred / f"{dst_job}_sample_{k}.cif")
                    _copy_json(src_pred, dst_pred, src_job, dst_job, k, k)
                    count += 1
            made[row_id] = count
            continue
        flags = row["flags"]
        guided = any(
            a == "--use_tfg_guidance" and b == "true" for a, b in zip(flags, flags[1:])
        )
        count = 0
        for job in jobs:
            atoms = SampleDictToFeatures(job, guided).get_atom_array()
            for seed_dir in sorted(
                (args.off_run / row_id / job["name"]).glob("seed_*")
            ):
                src_pred = seed_dir / "predictions"
                dst_pred = (
                    args.out / row_id / job["name"] / seed_dir.name / "predictions"
                )
                cifs = sorted(src_pred.glob(f"{job['name']}_sample_*.cif"))
                for cif in cifs:
                    k = int(cif.stem.rsplit("_", 1)[1])
                    if args.kind == "collapsed":
                        dst_pred.mkdir(parents=True, exist_ok=True)
                        source_cif = src_pred / f"{job['name']}_sample_0.cif"
                        shutil.copyfile(source_cif, dst_pred / cif.name)
                        _copy_json(src_pred, dst_pred, job["name"], job["name"], 0, k)
                        count += 1
                        continue
                    seed = int(seed_dir.name.split("_")[1])
                    rng = np.random.default_rng(seed * 100 + k)
                    new = transform(args.kind, atoms, _coords(cif), rng)
                    if new is None:
                        break
                    _write(cif, dst_pred / cif.name, new)
                    _copy_json(src_pred, dst_pred, job["name"], job["name"], k, k)
                    count += 1
        made[row_id] = count
    print(json.dumps({"kind": args.kind, "samples": made}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
