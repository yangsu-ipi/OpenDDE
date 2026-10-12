#!/usr/bin/env python
"""Measure Zn placement in two runs, against deposited sites and pairwise.

Usage:
    python scripts/qualification/zn_sites.py OFF_RUN CAND_RUN --rows D14,D15
        [--seed-pairs 101:101,102:102] --out OUT.json

Run from the repository root with ``OPENDDE_ROOT_DIR`` and (for reference
rows) ``OPENDDE_QUAL_DATA`` set. Coordinating atoms are polymer N, O and S
atoms within ``COORD`` of a Zn ion. For each row and job it records:

- **Deposited sites** (rows with a reference): each deposited Zn with at
  least one polymer coordinating atom defines a site, mapped onto the
  prediction through the reference chain assignment. In every sample the Zn
  ions are assigned to sites one-to-one (Hungarian, cost = the largest
  distance from the ion to the site's coordinating atoms), so swapped
  equivalent ions are not scored as displacements. Per site: that largest
  distance, the number of site atoms within ``COORD``, and the pocket-relative
  displacement from the deposited Zn after superposing the mapped polymer
  atoms within ``POCKET`` of it.
- **Paired ions** (``off`` sample k against candidate sample k): each ``off``
  ion with a coordinating atom is matched one-to-one to a candidate ion by
  pocket-relative distance (the candidate's polymer atoms within ``POCKET`` of
  the ``off`` ion superposed onto ``off``'s). Per matched ion: that distance and
  the Jaccard overlap of the two ions' coordinating residues. The summary
  counts, in same-outcome pairs (anchor RMSD at most ``SAME_OUTCOME``), the
  matched ions within ``SAME_SITE`` of each other: the proposed Zn gate.
- **Coordination** (unpaired): every ion's coordinating-atom count in every
  sample of both runs. The summary gives, per run, the fraction of ions over
  all samples with at least ``MIN_DONORS`` coordinating atoms, whatever the
  pair's outcome: the proposed fallback for rows whose paired check is
  inconclusive (protocol C5).

Ions without any coordinating atom are counted but not matched. The script
reports quantities only; limits belong in the qualification protocol.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from compare_structures import _samples, load_coords
from qual_structures import anchor_distance, build_reference, kabsch
from reference_metrics import reference_map, row_reference, score_sample
from scipy.optimize import linear_sum_assignment

from opendde.data.inference.json_to_feature import SampleDictToFeatures

COORD = 2.8
SAME_SITE = 2.0
SAME_OUTCOME = 2.0
POCKET = 8.0
DONORS = ("N", "O", "S")
MIN_DONORS = 2


def _guided(row) -> bool:
    flags = row["flags"]
    return any(
        a == "--use_tfg_guidance" and b == "true" for a, b in zip(flags, flags[1:])
    )


def _donor_mask(atoms) -> np.ndarray:
    polymer = np.isin(atoms.mol_type, ("protein", "dna", "rna"))
    return polymer & np.isin(np.char.upper(atoms.element.astype(str)), DONORS)


def _residue_key(atoms, i: int) -> str:
    return f"{atoms.chain_id[i]}:{atoms.res_name[i]}{atoms.res_id[i]}"


def coordination(xyz, ions, donors, cutoff: float = COORD) -> list[np.ndarray]:
    """Donor atom indices within ``cutoff`` of each ion."""
    d = np.linalg.norm(xyz[donors][None, :, :] - xyz[ions][:, None, :], axis=-1)
    return [donors[row <= cutoff] for row in d]


def pocket_superpose(fixed_xyz, mobile_xyz, centre, polymer, radius: float = POCKET):
    """Superpose ``mobile`` onto ``fixed`` on the polymer atoms within
    ``radius`` of ``centre`` in ``fixed``; returns (r, t, n_atoms)."""
    near = polymer[np.linalg.norm(fixed_xyz[polymer] - centre, axis=1) <= radius]
    if len(near) < 3:
        return None, None, len(near)
    r, t = kabsch(fixed_xyz[near], mobile_xyz[near])
    return r, t, len(near)


def assign(cost: np.ndarray) -> list[tuple[int, int]]:
    """One-to-one assignment of rows to columns minimizing total cost."""
    if cost.size == 0:
        return []
    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), cols.tolist()))


def deposited_sites(rmap) -> list[dict[str, Any]]:
    dep = rmap.dep
    zn = np.where(dep.res_name == "ZN")[0]
    polymer = np.where(
        np.isin(dep.chain_id, list({d for m in rmap.matches.values() for d in m}))
        & np.isin(np.char.upper(dep.element.astype(str)), DONORS)
    )[0]
    sites = []
    for i in zn:
        d = np.linalg.norm(dep.coord[polymer] - dep.coord[i], axis=1)
        atoms = polymer[d <= COORD]
        sites.append(
            {
                "dep_index": int(i),
                "label": f"Zn {dep.chain_id[i]}",
                "dep_atoms": atoms.tolist(),
                "ligands": [
                    f"{dep.chain_id[j]}:{dep.res_name[j]}{dep.res_id[j]}.{dep.atom_name[j]}"
                    for j in atoms
                ],
            }
        )
    return sites


def score_sites(rmap, sites, xyz, ions) -> list[dict[str, Any]]:
    scored = score_sample(rmap, xyz)
    if scored["status"] != "ok":
        return [{"label": s["label"], "status": scored["status"]} for s in sites]
    p, d = rmap.correspondence(scored["mapping"])
    dep_to_pred = dict(zip(d.tolist(), p.tolist()))
    dep = rmap.dep
    usable = []
    out = []
    for s in sites:
        pred_atoms = [dep_to_pred.get(j) for j in s["dep_atoms"]]
        if not s["dep_atoms"]:
            out.append({"label": s["label"], "status": "no polymer ligand"})
        elif None in pred_atoms:
            out.append({"label": s["label"], "status": "ligand atom unmapped"})
        else:
            usable.append((s, np.array(pred_atoms)))
            out.append(None)
    cost = np.array(
        [
            [np.linalg.norm(xyz[a] - xyz[ion], axis=1).max() for ion in ions]
            for _, a in usable
        ]
    ).reshape(len(usable), len(ions))
    pairs = dict(assign(cost))
    k = 0
    for n, entry in enumerate(out):
        if entry is not None:
            continue
        s, atoms = usable[k]
        if k not in pairs:
            out[n] = {"label": s["label"], "status": "no ion assigned"}
            k += 1
            continue
        ion = ions[pairs[k]]
        dist = np.linalg.norm(xyz[atoms] - xyz[ion], axis=1)
        centre = dep.coord[s["dep_index"]].astype(np.float64)
        near = np.linalg.norm(dep.coord[d] - centre, axis=1) <= POCKET
        disp = None
        if near.sum() >= 3:
            r, t = kabsch(dep.coord[d][near].astype(np.float64), xyz[p][near])
            disp = float(np.linalg.norm(xyz[ion] @ r.T + t - centre))
        out[n] = {
            "label": s["label"],
            "status": "ok",
            "ion": int(ion),
            "max_ligand_distance": float(dist.max()),
            "ligands_within": int((dist <= COORD).sum()),
            "n_ligands": int(len(dist)),
            "pocket_displacement": disp,
        }
        k += 1
    return out


def pair_ions(atoms, xyz_a, xyz_b, ions, donors, polymer) -> dict[str, Any]:
    coord_a = coordination(xyz_a, ions, donors)
    coord_b = coordination(xyz_b, ions, donors)
    bound = [n for n, c in enumerate(coord_a) if len(c)]
    frames = []
    cost = np.full((len(bound), len(ions)), 1e6)
    for row, n in enumerate(bound):
        r, t, _ = pocket_superpose(xyz_a, xyz_b, xyz_a[ions[n]], polymer)
        frames.append((r, t))
        if r is None:
            continue
        moved = xyz_b[ions] @ r.T + t
        cost[row] = np.linalg.norm(moved - xyz_a[ions[n]], axis=1)
    out = []
    for row, col in assign(cost):
        n = bound[row]
        res_a = {_residue_key(atoms, i) for i in coord_a[n]}
        res_b = {_residue_key(atoms, i) for i in coord_b[col]}
        union = res_a | res_b
        out.append(
            {
                "off_ion": int(ions[n]),
                "cand_ion": int(ions[col]),
                "pocket_distance": None
                if frames[row][0] is None
                else float(cost[row, col]),
                "off_ligands": sorted(res_a),
                "cand_ligands": sorted(res_b),
                "jaccard": len(res_a & res_b) / len(union) if union else None,
            }
        )
    return {
        "bound_off": len(bound),
        "bound_cand": sum(1 for c in coord_b if len(c)),
        "donors_off": [len(c) for c in coord_a],
        "donors_cand": [len(c) for c in coord_b],
        "ions": out,
    }


def measure_row(row, off_run: Path, cand_run: Path, seed_pairs):
    out = []
    for job in json.loads(Path(row["input"]).read_text()):
        atoms = SampleDictToFeatures(job, _guided(row)).get_atom_array()
        ions = np.where(atoms.res_name == "ZN")[0]
        if not len(ions):
            continue
        donors = np.where(_donor_mask(atoms))[0]
        polymer = np.where(
            np.isin(atoms.mol_type, ("protein", "dna", "rna")) & (atoms.element != "H")
        )[0]
        anchor_ref = build_reference(atoms, job)
        reference = row_reference(row, job["name"])
        rmap = reference_map(atoms, reference) if reference else None
        sites = deposited_sites(rmap) if rmap else []
        entry: dict[str, Any] = {
            "id": row["id"],
            "job": job["name"],
            "n_ions": int(len(ions)),
            "sites": [{k: s[k] for k in ("label", "ligands")} for s in sites],
            "samples": [],
        }
        for seed_a, seed_b in seed_pairs:
            pred_a = (
                off_run / row["id"] / job["name"] / f"seed_{seed_a}" / "predictions"
            )
            pred_b = (
                cand_run / row["id"] / job["name"] / f"seed_{seed_b}" / "predictions"
            )
            files_a, files_b = (
                _samples(pred_a, job["name"]),
                _samples(pred_b, job["name"]),
            )
            if sorted(files_a) != sorted(files_b) or not files_a:
                entry["samples"].append(
                    {"seeds": [seed_a, seed_b], "status": "sample sets differ"}
                )
                continue
            for k in sorted(files_a):
                xyz_a = load_coords(files_a[k], atoms)
                xyz_b = load_coords(files_b[k], atoms)
                sample = {
                    "seeds": [seed_a, seed_b],
                    "k": k,
                    "status": "ok",
                    "anchor_rmsd": anchor_distance(anchor_ref, xyz_a, xyz_b),
                    "paired": pair_ions(atoms, xyz_a, xyz_b, ions, donors, polymer),
                }
                if rmap is not None:
                    sample["sites_off"] = score_sites(rmap, sites, xyz_a, ions)
                    sample["sites_cand"] = score_sites(rmap, sites, xyz_b, ions)
                entry["samples"].append(sample)
        out.append(entry)
    return out


def summarize(entry: dict[str, Any]) -> dict[str, Any]:
    ok = [s for s in entry["samples"] if s["status"] == "ok"]
    ions = [i for s in ok for i in s["paired"]["ions"]]
    dist = [i["pocket_distance"] for i in ions if i["pocket_distance"] is not None]
    jac = [i["jaccard"] for i in ions if i["jaccard"] is not None]
    summary: dict[str, Any] = {
        "id": entry["id"],
        "job": entry["job"],
        "samples": len(ok),
        "bound_off": sum(s["paired"]["bound_off"] for s in ok),
        "bound_cand": sum(s["paired"]["bound_cand"] for s in ok),
        "paired_pocket_distance_median": float(np.median(dist)) if dist else None,
        "paired_pocket_distance_max": float(max(dist)) if dist else None,
        "paired_same_ligands": sum(1 for j in jac if j == 1.0),
        "paired_ions": len(jac),
    }
    same = [
        i for s in ok if s["anchor_rmsd"] <= SAME_OUTCOME for i in s["paired"]["ions"]
    ]
    within = sum(
        1
        for i in same
        if i["pocket_distance"] is not None and i["pocket_distance"] <= SAME_SITE
    )
    summary["same_outcome_samples"] = sum(
        1 for s in ok if s["anchor_rmsd"] <= SAME_OUTCOME
    )
    summary["same_site"] = within
    summary["same_site_of"] = len(same)
    summary["same_site_fraction"] = within / len(same) if same else None
    for tag in ("off", "cand"):
        counts = [n for s in ok for n in s["paired"][f"donors_{tag}"]]
        summary[f"coordinated_{tag}"] = sum(1 for n in counts if n >= MIN_DONORS)
        summary[f"ions_{tag}"] = len(counts)
        summary[f"coordinated_fraction_{tag}"] = (
            summary[f"coordinated_{tag}"] / len(counts) if counts else None
        )
    if summary["ions_off"] and summary["ions_cand"]:
        summary["coordinated_drop"] = (
            summary["coordinated_fraction_off"] - summary["coordinated_fraction_cand"]
        )
    else:
        summary["coordinated_drop"] = None
    sites = {}
    for s in ok:
        for tag in ("off", "cand"):
            for site in s.get(f"sites_{tag}", []):
                rec = sites.setdefault(site["label"], {"off": [], "cand": []})
                if site["status"] == "ok":
                    rec[tag].append(site)
    summary["sites"] = {
        label: {
            tag: {
                "occupied": sum(
                    1 for x in rec[tag] if x["ligands_within"] == x["n_ligands"]
                ),
                "scored": len(rec[tag]),
                "displacement_median": float(
                    np.median([x["pocket_displacement"] for x in rec[tag]])
                )
                if rec[tag]
                else None,
            }
            for tag in ("off", "cand")
        }
        for label, rec in sites.items()
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("off_run", type=Path)
    parser.add_argument("cand_run", type=Path)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--seed-pairs", default="101:101,102:102")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("scripts/qualification/inputs/manifest.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = {r["id"]: r for r in json.loads(args.manifest.read_text())["rows"]}
    seed_pairs = [
        tuple(int(x) for x in p.split(":")) for p in args.seed_pairs.split(",")
    ]
    entries = []
    for row_id in args.rows.split(","):
        entries.extend(
            measure_row(rows[row_id], args.off_run, args.cand_run, seed_pairs)
        )
    summaries = [summarize(e) for e in entries]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "off": str(args.off_run),
                "cand": str(args.cand_run),
                "seed_pairs": seed_pairs,
                "summaries": summaries,
                "entries": entries,
            },
            indent=1,
        )
    )
    print(json.dumps(summaries, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
