#!/usr/bin/env python
"""Symmetry-completeness screen for the frozen qualification inputs.

Usage:
    python scripts/qualification/symmetry_screen.py [MANIFEST] > screen.json

Run from the repository root with ``OPENDDE_ROOT_DIR`` set (the CCD data is
needed to build ligand graphs). The screen depends only on the inputs, never
on a prediction. For every manifest row it reports:

- Equivalent-chain assignments: chains are grouped by identical entity
  content. ``chain_assignments`` is the number of whole-complex assignments
  that permute chains within groups and preserve every inter-chain covalent
  bond and every chain named by a guidance constraint. ``chain_permutations``
  is the unconstrained product of group factorials.
- Ligand atom matchings: for each ligand chain, the number of graph
  automorphisms of its heavy atoms with covalent attachment atoms held fixed:
  with bond orders (aromatic bonds as one type), with conjugated terminal
  groups made equivalent as the comparison harness does
  (``matches_symmetrized``), and with elements only (an upper bound).

A row is exhaustive when ``chain_assignments`` is at most ``CHAIN_LIMIT`` and
every ligand's element-only count is at most ``LIGAND_CAP`` (RDKit
``CalcRMS``'s default ``maxMatches``).
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from qual_structures import (
    CHAIN_LIMIT,
    LIGAND_CAP,
    chain_layout,
    chain_permutations,
    ligand_mol,
    symmetry_maps,
    valid_assignments,
)

from opendde.data.inference.json_to_feature import SampleDictToFeatures


def _count(mol, symmetrize: bool) -> int:
    maps, _ = symmetry_maps(mol, LIGAND_CAP, symmetrize=symmetrize)
    return len(maps)


def _ligand_matches(atom_array, layout) -> list[dict[str, Any]]:
    out = []
    heavy = atom_array.element != "H"
    bonds = atom_array.bonds.as_array()
    for chain in layout.chain_ids:
        indices = layout.chain_atoms[chain]
        if not atom_array.hetero[indices].all():
            continue
        indices = indices[heavy[indices]]
        members = set(indices.tolist())
        fixed = {
            int(i) if int(i) in members else int(j)
            for i, j, _ in bonds
            if (int(i) in members) != (int(j) in members)
        }
        with_orders = ligand_mol(atom_array, indices, fixed, with_orders=True)
        elements = ligand_mol(atom_array, indices, fixed, with_orders=False)
        out.append(
            {
                "chain": str(chain),
                "residues": sorted(set(atom_array.res_name[indices].tolist())),
                "heavy_atoms": len(indices),
                "attachment_atoms": len(fixed),
                "matches_bond_orders": _count(with_orders, False),
                "matches_symmetrized": _count(with_orders, True),
                "matches_elements_only": _count(elements, False),
            }
        )
    return out


def screen_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = json.loads(Path(row["input"]).read_text())
    if row.get("job"):
        jobs = [j for j in jobs if j["name"] == row["job"]]
    flags = row["flags"]
    guided = any(
        a == "--use_tfg_guidance" and b == "true" for a, b in zip(flags, flags[1:])
    )
    results = []
    for job in jobs:
        atom_array = SampleDictToFeatures(job, guided).get_atom_array()
        layout = chain_layout(atom_array, job)
        assignments = valid_assignments(layout)
        ligands = _ligand_matches(atom_array, layout)
        exhaustive = (
            assignments is not None
            and len(assignments) <= CHAIN_LIMIT
            and all(lig["matches_elements_only"] <= LIGAND_CAP for lig in ligands)
        )
        results.append(
            {
                "id": row["id"],
                "job": job["name"],
                "chains": len(layout.chain_ids),
                "groups": {", ".join(g): len(g) for g in layout.groups if len(g) > 1},
                "constraint_named_chains": sorted(layout.named),
                "inter_chain_bonds": len(layout.inter_bonds),
                "chain_permutations": chain_permutations(layout),
                "chain_assignments": None if assignments is None else len(assignments),
                "ligands": ligands,
                "exhaustive": bool(exhaustive),
            }
        )
    return results


def main() -> int:
    manifest_path = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "scripts/qualification/inputs/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    seen: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    results = []
    for row in manifest["rows"]:
        key = (row["input"], row.get("job"))
        if key not in seen:
            seen[key] = screen_row(row)
        results.extend({**r, "id": row["id"]} for r in seen[key])
    print(
        json.dumps(
            {"chain_limit": CHAIN_LIMIT, "ligand_cap": LIGAND_CAP, "rows": results},
            indent=1,
            default=lambda v: v.item() if isinstance(v, np.generic) else str(v),
        )
    )
    return 0 if all(r["exhaustive"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
