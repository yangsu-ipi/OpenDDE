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
- ``zn_moved``: in every sample, one Zn ion that has a coordinating polymer
  atom moves 6 A out of its pocket, away from its coordinating atoms (further,
  in 2 A steps, if that position clashes).
- ``zn_wrong_site``: in every sample, one such Zn ion moves to a plausible
  site that the sample does not use: 2.1 A from a His, Asp, Glu or Cys donor
  atom, along the donor's bond direction, at least 8 A from every Zn and at
  least 2.5 A from every other heavy atom.

The Zn kinds choose the ion and the site with the sample's own random
generator. A sample with no bound Zn, or no free site, is copied unchanged
and counted.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from biotite.structure.io import pdbx

from zn_sites import coordination

from opendde.data.inference.json_to_feature import SampleDictToFeatures

GATED_LIGANDS = ("6OI", "ATR", "NAG", "PCG")
AXIS = np.array([0.3, 0.5, 0.8])
ZN_DONORS = {
    ("HIS", "ND1"): ("CG", "CE1"),
    ("HIS", "NE2"): ("CD2", "CE1"),
    ("ASP", "OD1"): ("CG",),
    ("ASP", "OD2"): ("CG",),
    ("GLU", "OE1"): ("CD",),
    ("GLU", "OE2"): ("CD",),
    ("CYS", "SG"): ("CB",),
}
ZN_BOND = 2.1


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


def _bound_zn(atoms, xyz):
    ions = np.where(atoms.res_name == "ZN")[0]
    polymer = np.isin(atoms.mol_type, ("protein", "dna", "rna"))
    donors = np.where(
        polymer & np.isin(np.char.upper(atoms.element.astype(str)), ("N", "O", "S"))
    )[0]
    coord = coordination(xyz, ions, donors)
    return [(int(i), c) for i, c in zip(ions, coord) if len(c)], ions


def zn_moved(atoms, xyz, rng) -> np.ndarray | None:
    bound, _ = _bound_zn(atoms, xyz)
    if not bound:
        return None
    ion, ligands = bound[int(rng.integers(len(bound)))]
    direction = xyz[ion] - xyz[ligands].mean(0)
    direction /= np.linalg.norm(direction) + 1e-9
    heavy = np.where(
        np.isin(atoms.mol_type, ("protein", "dna", "rna")) & (atoms.element != "H")
    )[0]
    for step in (6.0, 8.0, 10.0, 12.0):
        new = xyz[ion] + step * direction
        if np.linalg.norm(xyz[heavy] - new, axis=1).min() >= 3.0:
            out = xyz.copy()
            out[ion] = new
            return out
    return None


def zn_wrong_site(atoms, xyz, rng) -> np.ndarray | None:
    bound, ions = _bound_zn(atoms, xyz)
    if not bound:
        return None
    ion = bound[int(rng.integers(len(bound)))][0]
    heavy = np.where(atoms.element != "H")[0]
    lookup = {
        (str(atoms.chain_id[i]), int(atoms.res_id[i]), str(atoms.atom_name[i])): i
        for i in heavy
    }
    sites = []
    for i in heavy:
        parents = ZN_DONORS.get((str(atoms.res_name[i]), str(atoms.atom_name[i])))
        if parents is None or atoms.mol_type[i] != "protein":
            continue
        if np.linalg.norm(xyz[ions] - xyz[i], axis=1).min() < 8.0:
            continue
        key = (str(atoms.chain_id[i]), int(atoms.res_id[i]))
        idx = [lookup.get((*key, name)) for name in parents]
        if None in idx:
            continue
        direction = xyz[i] - xyz[idx].mean(0)
        direction /= np.linalg.norm(direction) + 1e-9
        new = xyz[i] + ZN_BOND * direction
        others = heavy[(heavy != i) & ~np.isin(heavy, ions)]
        if np.linalg.norm(xyz[others] - new, axis=1).min() < 2.5:
            continue
        sites.append(new)
    if not sites:
        return None
    out = xyz.copy()
    out[ion] = sites[int(rng.integers(len(sites)))]
    return out


def transform(kind: str, atoms, xyz: np.ndarray, rng) -> np.ndarray | None:
    if kind == "zn_moved":
        return zn_moved(atoms, xyz, rng)
    if kind == "zn_wrong_site":
        return zn_wrong_site(atoms, xyz, rng)
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
    made, unchanged = {}, {}
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
                    if new is None and args.kind.startswith("zn_"):
                        dst_pred.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(cif, dst_pred / cif.name)
                        _copy_json(src_pred, dst_pred, job["name"], job["name"], k, k)
                        unchanged[row_id] = unchanged.get(row_id, 0) + 1
                        count += 1
                        continue
                    if new is None:
                        break
                    _write(cif, dst_pred / cif.name, new)
                    _copy_json(src_pred, dst_pred, job["name"], job["name"], k, k)
                    count += 1
        made[row_id] = count
    print(json.dumps({"kind": args.kind, "samples": made, "unchanged": unchanged}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
