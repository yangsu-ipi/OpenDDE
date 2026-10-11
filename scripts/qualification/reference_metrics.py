#!/usr/bin/env python
"""Quality of each predicted sample against a deposited reference structure.

Usage:
    python scripts/qualification/reference_metrics.py RUN --rows D2,D3,...
        [--manifest PATH] --out OUT.json

Run from the repository root with ``OPENDDE_ROOT_DIR`` and
``OPENDDE_QUAL_DATA`` set. Each manifest row with a ``reference`` entry names
a frozen deposited mmCIF (``${OPENDDE_QUAL_DATA}/references/<id>.cif``,
checked by ``freeze_inputs.py check --external``). For every sample of every
seed in RUN (``<row>/<job>/seed_<s>/predictions``) this reports:

- the predicted-to-deposited chain mapping: each predicted polymer chain is
  aligned by sequence to deposited chains of the same polymer type. The
  deposited structure is the asymmetric unit, or assembly 1 (symmetry copies
  labelled ``<chain>.<sym_id>``) when that matches more predicted chains, as
  for a crystallographic dimer. Among one-to-one assignments that map the most
  predicted chains, the one with the lowest anchor RMSD (protein CA, else
  nucleic-acid C4') after superposition is used;
- lDDT of the predicted polymer heavy atoms against the deposited ones
  (conventional, reference-directional: ``symmetric=False``; otherwise the
  protocol's settings), for all polymer atoms and per polymer type;
- interface contacts in the manner of DockQ's Fnat: the fraction of deposited
  interchain residue contacts (heavy atoms within 5 A) reproduced, overall and
  for the worst chain pair with at least 5 deposited contacts, and the
  fraction of predicted contacts absent from the deposited structure;
- the sample's confidence scalars, for intrinsic ensemble statistics.

Only atoms present in both structures with the same residue and atom names
are compared; residues missing from the deposited structure are excluded.
Ligands and ions are not compared here.
"""

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any

import biotite.sequence as bseq
import biotite.sequence.align as balign
import biotite.structure as struc
import numpy as np
from biotite.structure.io import pdbx
from compare_structures import _confidence, _samples, load_coords
from qual_structures import LDDT_SETTINGS, build_reference, kabsch, rmsd
from scipy.spatial import cKDTree

from opendde.data.inference.json_to_feature import SampleDictToFeatures

INTERFACE_DISTANCE = 5.0
MIN_PAIR_CONTACTS = 5
MIN_IDENTITY = 0.9
MIN_COVERAGE = 0.5
MODIFIED_PARENT = {"MSE": "MET"}
AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}  # fmt: skip
NA_ONE = {
    "A": "A", "C": "C", "G": "G", "U": "U", "T": "T",
    "DA": "A", "DC": "C", "DG": "G", "DT": "T", "DU": "U",
}  # fmt: skip


def load_deposited(path: Path, assembly: bool = False):
    block = pdbx.CIFFile.read(str(path)).block
    if assembly:
        atoms = pdbx.get_assembly(
            block, model=1, use_author_fields=False, altloc="first"
        )
        atoms.chain_id = np.array(
            [f"{c}.{s}" for c, s in zip(atoms.chain_id, atoms.sym_id)], dtype="U12"
        )
    else:
        atoms = pdbx.get_structure(
            block, model=1, use_author_fields=False, altloc="first"
        )
    atoms = atoms[
        (atoms.element != "H") & (atoms.element != "D") & (atoms.res_name != "HOH")
    ]
    return atoms


def _polymer_type(res_names) -> str | None:
    names = set(res_names)
    if names & AMINO_ACIDS or names & set(MODIFIED_PARENT):
        return "protein"
    if names & {"DA", "DC", "DG", "DT"}:
        return "dna"
    if names & {"A", "C", "G", "U"}:
        return "rna"
    return None


def _one_letter(res_name: str, kind: str) -> str:
    res_name = MODIFIED_PARENT.get(res_name, res_name)
    if kind == "protein":
        try:
            return bseq.ProteinSequence.convert_letter_3to1(res_name)
        except KeyError:
            return "X"
    letter = NA_ONE.get(res_name, "N")
    return "T" if letter == "U" else letter


def _residues(atoms, chain_mask):
    idx = np.where(chain_mask)[0]
    starts = struc.get_residue_starts(atoms[idx])
    return [(int(atoms.res_id[idx[s]]), str(atoms.res_name[idx[s]])) for s in starts]


def _align(pred_res, ref_res, kind):
    if kind == "protein":
        a = bseq.ProteinSequence("".join(_one_letter(n, kind) for _, n in pred_res))
        b = bseq.ProteinSequence("".join(_one_letter(n, kind) for _, n in ref_res))
        matrix = balign.SubstitutionMatrix.std_protein_matrix()
    else:
        letters = lambda res: "".join(_one_letter(n, kind) for _, n in res)  # noqa: E731
        a = bseq.NucleotideSequence(letters(pred_res).replace("N", "A"))
        b = bseq.NucleotideSequence(letters(ref_res).replace("N", "A"))
        matrix = balign.SubstitutionMatrix.std_nucleotide_matrix()
    ali = balign.align_optimal(
        a, b, matrix, gap_penalty=(-10, -1), terminal_penalty=False
    )[0]
    pairs = [(int(i), int(j)) for i, j in ali.trace if i >= 0 and j >= 0]
    same = sum(
        MODIFIED_PARENT.get(pred_res[i][1], pred_res[i][1])
        == MODIFIED_PARENT.get(ref_res[j][1], ref_res[j][1])
        for i, j in pairs
    )
    identity = same / len(pairs) if pairs else 0.0
    coverage = len(pairs) / max(1, min(len(pred_res), len(ref_res)))
    return pairs, identity, coverage


class ReferenceMap:
    """Candidate chain matches between a featurized prediction and a deposited
    structure, with atom correspondences per matched chain pair."""

    def __init__(self, atom_array, deposited, exclude_mask=None):
        self.pred = atom_array
        self.dep = deposited
        heavy = atom_array.element != "H"
        self.pred_polymer = heavy & np.isin(
            atom_array.mol_type, ("protein", "dna", "rna")
        )
        if exclude_mask is not None:
            self.pred_polymer &= ~exclude_mask
        pred_chains = list(
            dict.fromkeys(atom_array.chain_id[self.pred_polymer].tolist())
        )
        dep_chains = list(dict.fromkeys(deposited.chain_id.tolist()))
        dep_kind = {}
        for c in dep_chains:
            dep_kind[c] = _polymer_type(
                deposited.res_name[deposited.chain_id == c].tolist()
            )
        self.matches: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        self.kind = {}
        for pc in pred_chains:
            mask = self.pred_polymer & (atom_array.chain_id == pc)
            kind = str(atom_array.mol_type[np.where(mask)[0][0]])
            self.kind[pc] = kind
            pred_res = _residues(atom_array, atom_array.chain_id == pc)
            for dc in dep_chains:
                if dep_kind[dc] != kind:
                    continue
                ref_res = _residues(deposited, deposited.chain_id == dc)
                pairs, identity, coverage = _align(pred_res, ref_res, kind)
                if identity < MIN_IDENTITY or coverage < MIN_COVERAGE:
                    continue
                self.matches.setdefault(pc, {})[dc] = self._atom_pairs(
                    pc, dc, pred_res, ref_res, pairs
                )

    def _atom_pairs(self, pc, dc, pred_res, ref_res, pairs):
        p_idx, d_idx = [], []
        pred_by_res: dict[int, list[tuple[str, int]]] = {}
        for i in np.where(self.pred_polymer & (self.pred.chain_id == pc))[0]:
            pred_by_res.setdefault(int(self.pred.res_id[i]), []).append(
                (str(self.pred.atom_name[i]), int(i))
            )
        dep_lookup = {}
        for j in np.where(self.dep.chain_id == dc)[0]:
            dep_lookup.setdefault(
                (int(self.dep.res_id[j]), str(self.dep.atom_name[j])), j
            )
        for i, j in pairs:
            (pr, pn), (dr, dn) = pred_res[i], ref_res[j]
            if MODIFIED_PARENT.get(pn, pn) != MODIFIED_PARENT.get(dn, dn):
                continue
            for name, pi in pred_by_res.get(pr, []):
                dj = dep_lookup.get((dr, name))
                if dj is not None:
                    p_idx.append(pi)
                    d_idx.append(int(dj))
        return np.array(p_idx, dtype=np.int64), np.array(d_idx, dtype=np.int64)

    def _injective(self, limit: int = 100_000):
        chains = sorted(self.matches)
        options = [sorted(self.matches[c]) + [None] for c in chains]
        out = []
        for choice in itertools.product(*options):
            mapped = [d for d in choice if d is not None]
            if mapped and len(set(mapped)) == len(mapped):
                out.append({c: d for c, d in zip(chains, choice) if d is not None})
                if len(out) > limit:
                    raise ValueError("reference assignment search exceeds limit")
        return out

    def max_mapped(self) -> int:
        return max((len(m) for m in self._injective()), default=0)

    def assignments(self):
        """One-to-one assignments that map the most predicted chains."""
        maps = self._injective()
        best = max((len(m) for m in maps), default=0)
        return [m for m in maps if len(m) == best]

    def correspondence(self, mapping):
        p = np.concatenate([self.matches[pc][dc][0] for pc, dc in mapping.items()])
        d = np.concatenate([self.matches[pc][dc][1] for pc, dc in mapping.items()])
        return p, d


def reference_map(atom_array, path: Path, exclude_mask=None) -> "ReferenceMap":
    """Map against the asymmetric unit, or assembly 1 if it maps more chains."""
    unit = ReferenceMap(atom_array, load_deposited(path), exclude_mask)
    if unit.max_mapped() == len(unit.kind):
        return unit
    assembly = ReferenceMap(
        atom_array, load_deposited(path, assembly=True), exclude_mask
    )
    return assembly if assembly.max_mapped() > unit.max_mapped() else unit


def _anchor(names, kinds):
    if "protein" in kinds:
        return names == "CA"
    return names == "C4'"


def _interface_contacts(xyz, chain, residue, atom_mask):
    idx = np.where(atom_mask)[0]
    pairs = cKDTree(xyz[idx]).query_pairs(INTERFACE_DISTANCE, output_type="ndarray")
    i, j = idx[pairs[:, 0]], idx[pairs[:, 1]]
    keep = chain[i] != chain[j]
    out = set()
    for a, b in zip(i[keep], j[keep]):
        ka, kb = (chain[a], residue[a]), (chain[b], residue[b])
        out.add((ka, kb) if ka < kb else (kb, ka))
    return out


def score_sample(rmap: ReferenceMap, xyz: np.ndarray) -> dict[str, Any]:
    if not np.isfinite(xyz).all():
        return {"status": "fail: non-finite coordinates"}
    best = None
    for mapping in rmap.assignments():
        p, d = rmap.correspondence(mapping)
        anchor = _anchor(rmap.dep.atom_name[d], set(rmap.kind.values()))
        if anchor.sum() < 3:
            continue
        r, t = kabsch(rmap.dep.coord[d][anchor].astype(np.float64), xyz[p][anchor])
        value = rmsd(rmap.dep.coord[d][anchor], xyz[p][anchor] @ r.T + t)
        if best is None or value < best[0]:
            best = (value, mapping, p, d)
    if best is None:
        return {"status": "fail: no reference chain match"}
    anchor_rmsd, mapping, p, d = best
    ref = rmap.dep[d]
    sub = ref.copy()
    sub.coord = xyz[p].astype(np.float32)
    # Score in predicted chain labels so per-type masks and chain pairs follow the prediction.
    ref.chain_id = rmap.pred.chain_id[p]
    sub.chain_id = rmap.pred.chain_id[p]
    settings = {**LDDT_SETTINGS, "symmetric": False}
    kinds = rmap.pred.mol_type[p]
    out: dict[str, Any] = {
        "status": "ok",
        "mapping": mapping,
        "unmapped_chains": sorted(set(rmap.kind) - set(mapping)),
        "matched_atoms": int(len(p)),
        "anchor_rmsd": anchor_rmsd,
        "lddt": {"polymer": float(struc.lddt(ref, sub, **settings))},
    }
    for kind in ("protein", "dna", "rna"):
        mask = kinds == kind
        if mask.any():
            out["lddt"][kind] = float(
                struc.lddt(ref, sub, atom_mask=mask, partner_mask=mask, **settings)
            )
    chain = np.asarray(ref.chain_id)
    residue = np.asarray(rmap.pred.res_id[p])
    all_atoms = np.ones(len(p), dtype=bool)
    dep_contacts = _interface_contacts(ref.coord, chain, residue, all_atoms)
    pred_contacts = _interface_contacts(sub.coord, chain, residue, all_atoms)
    if dep_contacts:
        out["fnat"] = len(dep_contacts & pred_contacts) / len(dep_contacts)
        out["fnonnat"] = (
            len(pred_contacts - dep_contacts) / len(pred_contacts)
            if pred_contacts
            else 0.0
        )
        per_pair = {}
        for (ca, _), (cb, _) in dep_contacts:
            per_pair.setdefault((ca, cb), [0, 0])[0] += 1
        for key in dep_contacts & pred_contacts:
            per_pair[(key[0][0], key[1][0])][1] += 1
        pairs = {
            f"{a}-{b}": v[1] / v[0]
            for (a, b), v in per_pair.items()
            if v[0] >= MIN_PAIR_CONTACTS
        }
        out["fnat_pairs"] = pairs
        out["fnat_min_pair"] = min(pairs.values()) if pairs else None
    return out


def row_reference(row: dict[str, Any], job: str) -> Path | None:
    """A row's ``reference`` is a path, or a mapping from job name to path."""
    entry = row.get("reference")
    if isinstance(entry, dict):
        entry = entry.get(job)
    if not entry:
        return None
    path = os.path.expandvars(entry)
    if "$" in path:
        raise SystemExit(f"error: unset environment variable in {entry!r}")
    return Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--rows", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("scripts/qualification/inputs/manifest.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = {r["id"]: r for r in manifest["rows"]}
    entries = []
    for row_id in args.rows.split(","):
        row = rows[row_id]
        flags = row["flags"]
        guided = any(
            a == "--use_tfg_guidance" and b == "true" for a, b in zip(flags, flags[1:])
        )
        window = (row.get("comparison") or {}).get("exclude_constraint_window")
        for job in json.loads(Path(row["input"]).read_text()):
            reference = row_reference(row, job["name"])
            atom_array = SampleDictToFeatures(job, guided).get_atom_array()
            exclude = None
            if window is not None:
                exclude = build_reference(
                    atom_array, job, exclude_window=window
                ).exclude_mask
            rmap = reference_map(atom_array, reference, exclude) if reference else None
            for seed_dir in sorted((args.run / row_id / job["name"]).glob("seed_*")):
                pred = seed_dir / "predictions"
                for k, cif in sorted(_samples(pred, job["name"]).items()):
                    entry = {
                        "id": row_id,
                        "job": job["name"],
                        "seed": int(seed_dir.name.split("_")[1]),
                        "sample": k,
                        "confidence": _confidence(pred, job["name"], k),
                    }
                    if rmap is not None:
                        entry["reference"] = score_sample(
                            rmap, load_coords(cif, atom_array)
                        )
                    entries.append(entry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"run": str(args.run), "entries": entries}, indent=1, default=str)
    )
    bad = [e for e in entries if e.get("reference", {}).get("status", "ok") != "ok"]
    print(f"{len(entries)} samples, {len(bad)} not ok")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
