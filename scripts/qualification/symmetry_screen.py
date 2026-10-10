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
  automorphisms of its heavy atoms with covalent attachment atoms held fixed,
  counted with bond orders (aromatic bonds as one type) and with elements only.
  The element-only count is an upper bound on what any equivalence of
  conjugated terminal groups can add.

A row is exhaustive when ``chain_assignments`` is at most ``CHAIN_LIMIT`` and
every ligand's element-only count is at most ``LIGAND_CAP`` (RDKit
``CalcRMS``'s default ``maxMatches``).
"""

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from opendde.data.inference.json_to_feature import SampleDictToFeatures

CHAIN_LIMIT = 100_000
LIGAND_CAP = 1_000_000
ENUMERATION_LIMIT = 1_000_000
POLYMER_TYPES = ("proteinChain", "dnaSequence", "rnaSequence")
CONTENT_SKIP = {"count", "id"}


def _entity_signature(entity_type: str, entity: dict[str, Any]) -> str:
    content = {
        k: v
        for k, v in entity.items()
        if k not in CONTENT_SKIP and "msa" not in k.lower() and "templ" not in k.lower()
    }
    return json.dumps([entity_type, content], sort_keys=True)


def _constrained_chains(job: dict[str, Any], entity_chains: dict[str, list[str]]):
    """Chains a guidance constraint names, by entity copy or by chain id."""
    constraint = job.get("constraint") or {}
    named: set[str] = set(constraint.get("movable_chains") or [])

    def add(entity: Any, copy: Any) -> None:
        chains = entity_chains.get(str(entity), [])
        index = int(copy) - 1
        if 0 <= index < len(chains):
            named.add(chains[index])

    for contact in constraint.get("contact") or []:
        add(contact["entity1"], contact.get("copy1", 1))
        add(contact["entity2"], contact.get("copy2", 1))
    epitope = constraint.get("epitope") or {}
    for residue in epitope.get("residues") or []:
        add(residue["entity"], residue.get("copy", 1))
    return named


def _chain_assignments(atom_array, job: dict[str, Any]) -> dict[str, Any]:
    chain_ids = list(dict.fromkeys(atom_array.chain_id.tolist()))
    entity_of = {}
    for chain in chain_ids:
        entity_of[chain] = str(
            atom_array.label_entity_id[atom_array.chain_id == chain][0]
        )
    entity_chains: dict[str, list[str]] = {}
    for chain in chain_ids:
        entity_chains.setdefault(entity_of[chain], []).append(chain)

    signatures = {}
    for index, type2entity in enumerate(job["sequences"]):
        ((entity_type, entity),) = type2entity.items()
        signatures[str(index + 1)] = _entity_signature(entity_type, entity)

    named = _constrained_chains(job, entity_chains)
    groups: dict[str, list[str]] = {}
    for chain in chain_ids:
        label = chain if chain in named else signatures[entity_of[chain]]
        groups.setdefault(label, []).append(chain)
    unconstrained: dict[str, int] = {}
    for chain in chain_ids:
        sig = signatures[entity_of[chain]]
        unconstrained[sig] = unconstrained.get(sig, 0) + 1
    permutations = math.prod(math.factorial(n) for n in unconstrained.values())
    constrained_total = math.prod(math.factorial(len(g)) for g in groups.values())

    bonds = atom_array.bonds.as_array()
    inter = []
    for i, j, _ in bonds:
        ci, cj = atom_array.chain_id[i], atom_array.chain_id[j]
        if ci != cj:
            ki = (int(atom_array.res_id[i]), str(atom_array.atom_name[i]))
            kj = (int(atom_array.res_id[j]), str(atom_array.atom_name[j]))
            inter.append(((str(ci), ki), (str(cj), kj)))
    bond_set = {frozenset(b) for b in inter}

    if constrained_total > ENUMERATION_LIMIT:
        valid = None
    else:
        movable = [g for g in groups.values() if len(g) > 1]
        valid = 0
        for perms in itertools.product(*(itertools.permutations(g) for g in movable)):
            mapping = {c: c for c in chain_ids}
            for group, perm in zip(movable, perms):
                mapping.update(dict(zip(group, perm)))
            if all(
                frozenset(((mapping[a[0]], a[1]), (mapping[b[0]], b[1]))) in bond_set
                for a, b in inter
            ):
                valid += 1
    return {
        "chains": len(chain_ids),
        "groups": {", ".join(g): len(g) for g in groups.values() if len(g) > 1},
        "constraint_named_chains": sorted(named),
        "inter_chain_bonds": len(inter),
        "chain_permutations": permutations,
        "chain_assignments": valid,
    }


def _bond_type(value: int):
    # biotite BondType: 1 single, 2 double, 3 triple, 5-7 aromatic single/double/triple, 9 aromatic
    if value in (5, 6, 7, 9):
        return Chem.BondType.AROMATIC
    return {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }.get(int(value), Chem.BondType.UNSPECIFIED)


def _count_matches(
    atom_array, indices: np.ndarray, fixed: set[int], with_orders: bool
) -> int:
    position = {int(i): k for k, i in enumerate(indices)}
    mol = Chem.RWMol()
    for k, i in enumerate(indices):
        element = str(atom_array.element[i]).capitalize()
        atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(element)
        if int(i) in fixed:
            # A unique pseudo-element keeps each attachment atom in place.
            atomic_number = 110 + sorted(fixed).index(int(i))
        atom = Chem.Atom(atomic_number)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    for i, j, bond_type in atom_array.bonds.as_array():
        if int(i) in position and int(j) in position:
            mol.AddBond(
                position[int(i)],
                position[int(j)],
                _bond_type(bond_type) if with_orders else Chem.BondType.SINGLE,
            )
    mol = mol.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    matches = mol.GetSubstructMatches(
        mol, uniquify=False, useChirality=False, maxMatches=LIGAND_CAP + 1
    )
    return len(matches)


def _ligand_matches(atom_array) -> list[dict[str, Any]]:
    out = []
    heavy = atom_array.element != "H"
    bonds = atom_array.bonds.as_array()
    for chain in dict.fromkeys(atom_array.chain_id[atom_array.hetero].tolist()):
        indices = np.where((atom_array.chain_id == chain) & heavy)[0]
        members = set(indices.tolist())
        fixed = {
            int(i) if int(i) in members else int(j)
            for i, j, _ in bonds
            if (int(i) in members) != (int(j) in members)
        }
        out.append(
            {
                "chain": str(chain),
                "residues": sorted(set(atom_array.res_name[indices].tolist())),
                "heavy_atoms": len(indices),
                "attachment_atoms": len(fixed),
                "matches_bond_orders": _count_matches(atom_array, indices, fixed, True),
                "matches_elements_only": _count_matches(
                    atom_array, indices, fixed, False
                ),
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
        chains = _chain_assignments(atom_array, job)
        ligands = _ligand_matches(atom_array)
        exhaustive = (
            chains["chain_assignments"] is not None
            and chains["chain_assignments"] <= CHAIN_LIMIT
            and all(lig["matches_elements_only"] <= LIGAND_CAP for lig in ligands)
        )
        results.append(
            {
                "id": row["id"],
                "job": job["name"],
                **chains,
                "ligands": ligands,
                "exhaustive": exhaustive,
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
        )
    )
    return 0 if all(r["exhaustive"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
