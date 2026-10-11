"""Structure-comparison helpers for qualification.

Implements the comparison rules in ``porting_docs/qualification_protocol.md``:
one whole-complex equivalent-chain assignment chosen by a declared anchor
criterion, one anchor superposition applied to the whole complex, biotite
lDDT with explicit settings and a contact-pair count, and symmetry-aware
ligand RMSD through RDKit ``CalcRMS`` with an explicitly enumerated atom map.

Every function works on an OpenDDE featurized ``AtomArray`` (from
``SampleDictToFeatures(job).get_atom_array()``) plus coordinate arrays in the
same atom order, which is the order OpenDDE writes prediction CIFs in.
"""

import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Any

import biotite.structure as struc
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Geometry import Point3D
from scipy.spatial import cKDTree

LIGAND_CAP = 1_000_000
CHAIN_LIMIT = 100_000
POLYMER_TYPES = ("protein", "dna", "rna")
LDDT_SETTINGS = {
    "aggregation": "all",
    "inclusion_radius": 15.0,
    "distance_bins": (0.5, 1.0, 2.0, 4.0),
    "exclude_same_residue": True,
    "exclude_same_chain": False,
    "symmetric": True,
}
POCKET_DISTANCE = 4.0
# Interchain residue contact: any heavy-atom pair within this distance (DockQ's Fnat).
INTERFACE_DISTANCE = 5.0
# OpenDDE's epitope rule (opendde/tfg/epitope_guidance.py): a residue is reached
# when one of its heavy atoms is within GATE of a paratope heavy atom.
EPITOPE_GATE = 5.0
ATOM_ERROR_PERCENTILES = (50, 95, 99)
# RDKit's conjugated terminal group pattern (MolAlign symmetrizeTerminalAtoms).
TERMINAL_GROUP = Chem.MolFromSmarts(
    "[O,N;D1;$([O,N;D1]-[*]=[O,N;D1]),$([O,N;D1]=[*]-[O,N;D1])]~[*]"
)


# ----------------------------------------------------------------------------
# Equivalent chains
# ----------------------------------------------------------------------------


def entity_signature(entity_type: str, entity: dict[str, Any]) -> str:
    """Entity content that decides chain equivalence (no counts, ids or MSAs)."""
    content = {
        k: v
        for k, v in entity.items()
        if k not in ("count", "id")
        and "msa" not in k.lower()
        and "templ" not in k.lower()
    }
    return json.dumps([entity_type, content], sort_keys=True)


@dataclass
class ChainLayout:
    chain_ids: list[str]
    entity_chains: dict[str, list[str]]
    chain_atoms: dict[str, np.ndarray]
    signature_groups: list[list[str]]
    groups: list[list[str]]
    named: set[str]
    inter_bonds: list[tuple[tuple[str, tuple], tuple[str, tuple]]]


def constraint_atom(layout: ChainLayout, atom_array, entity, copy, position, name):
    chain = layout.entity_chains[str(entity)][int(copy) - 1]
    idx = layout.chain_atoms[chain]
    hit = idx[
        (atom_array.res_id[idx] == int(position)) & (atom_array.atom_name[idx] == name)
    ]
    if len(hit) != 1:
        raise ValueError(
            f"constraint atom {chain}:{position}:{name} matched {len(hit)}"
        )
    return int(hit[0])


def _constrained_chains(job: dict[str, Any], entity_chains: dict[str, list[str]]):
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
    for residue in (constraint.get("epitope") or {}).get("residues") or []:
        add(residue["entity"], residue.get("copy", 1))
    return named


def chain_layout(atom_array, job: dict[str, Any]) -> ChainLayout:
    chain_ids = list(dict.fromkeys(atom_array.chain_id.tolist()))
    chain_atoms = {c: np.where(atom_array.chain_id == c)[0] for c in chain_ids}
    entity_of = {
        c: str(atom_array.label_entity_id[chain_atoms[c][0]]) for c in chain_ids
    }
    entity_chains: dict[str, list[str]] = {}
    for chain in chain_ids:
        entity_chains.setdefault(entity_of[chain], []).append(chain)

    signatures = {}
    for index, type2entity in enumerate(job["sequences"]):
        ((entity_type, entity),) = type2entity.items()
        signatures[str(index + 1)] = entity_signature(entity_type, entity)

    named = _constrained_chains(job, entity_chains)
    by_signature: dict[str, list[str]] = {}
    by_label: dict[str, list[str]] = {}
    for chain in chain_ids:
        sig = signatures[entity_of[chain]]
        by_signature.setdefault(sig, []).append(chain)
        by_label.setdefault(chain if chain in named else sig, []).append(chain)

    for group in by_signature.values():
        keys = [
            list(
                zip(
                    atom_array.res_id[chain_atoms[c]],
                    atom_array.atom_name[chain_atoms[c]],
                )
            )
            for c in group
        ]
        if any(k != keys[0] for k in keys[1:]):
            raise ValueError(f"equivalent chains {group} differ in atom layout")

    inter = []
    for i, j, _ in atom_array.bonds.as_array():
        ci, cj = str(atom_array.chain_id[i]), str(atom_array.chain_id[j])
        if ci != cj:
            ki = (int(atom_array.res_id[i]), str(atom_array.atom_name[i]))
            kj = (int(atom_array.res_id[j]), str(atom_array.atom_name[j]))
            inter.append(((ci, ki), (cj, kj)))
    return ChainLayout(
        chain_ids=chain_ids,
        entity_chains=entity_chains,
        chain_atoms=chain_atoms,
        signature_groups=list(by_signature.values()),
        groups=list(by_label.values()),
        named=named,
        inter_bonds=inter,
    )


def chain_permutations(layout: ChainLayout) -> int:
    return math.prod(math.factorial(len(g)) for g in layout.signature_groups)


def valid_assignments(layout: ChainLayout, limit: int = CHAIN_LIMIT):
    """Chain mappings (reference chain -> candidate chain) that keep every
    inter-chain bond and every constraint-named chain. Identity comes first,
    then itertools order. Returns None if the search space exceeds ``limit``."""
    movable = [g for g in layout.groups if len(g) > 1]
    if math.prod(math.factorial(len(g)) for g in movable) > limit:
        return None
    bond_set = {frozenset(b) for b in layout.inter_bonds}
    out = []
    for perms in itertools.product(*(itertools.permutations(g) for g in movable)):
        mapping = {c: c for c in layout.chain_ids}
        for group, perm in zip(movable, perms):
            mapping.update(dict(zip(group, perm)))
        if all(
            frozenset(((mapping[a[0]], a[1]), (mapping[b[0]], b[1]))) in bond_set
            for a, b in layout.inter_bonds
        ):
            out.append(mapping)
    return out


def permutation_index(layout: ChainLayout, mapping: dict[str, str], n_atom: int):
    """Index array so that ``candidate[index]`` is in reference chain labels."""
    index = np.empty(n_atom, dtype=np.int64)
    for ref_chain, cand_chain in mapping.items():
        index[layout.chain_atoms[ref_chain]] = layout.chain_atoms[cand_chain]
    return index


# ----------------------------------------------------------------------------
# Ligand graphs and symmetry-aware RMSD
# ----------------------------------------------------------------------------


def _rdkit_bond(value: int, with_orders: bool):
    if not with_orders:
        return Chem.BondType.SINGLE
    # biotite BondType: 1 single, 2 double, 3 triple, 5-7 aromatic single/double/triple, 9 aromatic
    if int(value) in (5, 6, 7, 9):
        return Chem.BondType.AROMATIC
    return {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }.get(int(value), Chem.BondType.UNSPECIFIED)


def ligand_mol(atom_array, indices: np.ndarray, fixed: set[int], with_orders=True):
    """Heavy-atom graph of one ligand chain. Each attachment atom in ``fixed``
    gets a unique pseudo-element so no automorphism can move it."""
    position = {int(i): k for k, i in enumerate(indices)}
    table = Chem.GetPeriodicTable()
    fixed_order = sorted(fixed)
    mol = Chem.RWMol()
    for i in indices:
        if int(i) in fixed:
            atomic_number = 110 + fixed_order.index(int(i))
        else:
            atomic_number = table.GetAtomicNumber(
                str(atom_array.element[i]).capitalize()
            )
        atom = Chem.Atom(atomic_number)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    for i, j, bond_type in atom_array.bonds.as_array():
        if int(i) in position and int(j) in position:
            mol.AddBond(
                position[int(i)], position[int(j)], _rdkit_bond(bond_type, with_orders)
            )
    mol = mol.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def symmetrize_terminal_groups(mol):
    """Copy of ``mol`` with conjugated terminal O/N treated as equivalent, as
    RDKit ``CalcRMS(symmetrizeConjugatedTerminalGroups=True)`` does."""
    out = Chem.RWMol(mol)
    for terminal, neighbour in mol.GetSubstructMatches(TERMINAL_GROUP, uniquify=True):
        out.GetAtomWithIdx(terminal).SetFormalCharge(0)
        out.GetBondBetweenAtoms(terminal, neighbour).SetBondType(Chem.BondType.SINGLE)
    out = out.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(out)
    return out


def symmetry_maps(mol, cap: int = LIGAND_CAP, symmetrize: bool = True):
    """All automorphisms of ``mol`` (up to ``cap``) and whether the search was
    truncated. A match ``m`` maps atom ``q`` to atom ``m[q]``."""
    query = symmetrize_terminal_groups(mol) if symmetrize else mol
    matches = query.GetSubstructMatches(
        query, uniquify=False, useChirality=False, maxMatches=cap + 1
    )
    return list(matches[:cap]), len(matches) > cap


def _with_conformer(mol, xyz: np.ndarray):
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for k, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(k, Point3D(float(x), float(y), float(z)))
    out.AddConformer(conf, assignId=True)
    return out


def symmetric_rmsd(mol, ref_xyz: np.ndarray, prb_xyz: np.ndarray, maps) -> float:
    """In-place (no realignment) RMSD minimized over ``maps``, via CalcRMS."""
    if mol.GetNumAtoms() == 1:
        return float(np.linalg.norm(prb_xyz[0] - ref_xyz[0]))
    atom_map = [[(int(m[q]), q) for q in range(len(m))] for m in maps]
    return float(
        rdMolAlign.CalcRMS(
            _with_conformer(mol, prb_xyz), _with_conformer(mol, ref_xyz), map=atom_map
        )
    )


# ----------------------------------------------------------------------------
# Superposition and lDDT
# ----------------------------------------------------------------------------


def kabsch(fixed: np.ndarray, mobile: np.ndarray):
    """Rotation ``r`` and translation ``t`` with ``mobile @ r.T + t ~ fixed``."""
    cf, cm = fixed.mean(axis=0), mobile.mean(axis=0)
    h = (mobile - cm).T @ (fixed - cf)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cf - cm @ r.T


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean())) if len(a) else float("nan")


def contact_pair_count(xyz_a, xyz_b, atom_mask, partner_mask, residue_key, radius):
    """Distinct atom pairs (one in ``atom_mask``, the other in ``partner_mask``,
    different residues) within ``radius`` in either structure."""
    idx = np.where(atom_mask | partner_mask)[0]
    if len(idx) < 2:
        return 0
    n = len(atom_mask)
    found = []
    for xyz in (xyz_a, xyz_b):
        pairs = cKDTree(xyz[idx]).query_pairs(radius, output_type="ndarray")
        i, j = idx[pairs[:, 0]], idx[pairs[:, 1]]
        keep = ((atom_mask[i] & partner_mask[j]) | (atom_mask[j] & partner_mask[i])) & (
            residue_key[i] != residue_key[j]
        )
        lo, hi = np.minimum(i[keep], j[keep]), np.maximum(i[keep], j[keep])
        found.append(lo.astype(np.int64) * n + hi)
    return int(len(np.union1d(found[0], found[1])))


def interchain_contacts(xyz, mask, chain_id, residue_key) -> set[tuple[str, str]]:
    """Residue pairs on different chains with a heavy-atom pair within
    ``INTERFACE_DISTANCE``, as sorted residue-key pairs."""
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return set()
    pairs = cKDTree(xyz[idx]).query_pairs(INTERFACE_DISTANCE, output_type="ndarray")
    i, j = idx[pairs[:, 0]], idx[pairs[:, 1]]
    keep = chain_id[i] != chain_id[j]
    out = set()
    for a, b in zip(residue_key[i[keep]], residue_key[j[keep]]):
        out.add((a, b) if a < b else (b, a))
    return out


def lddt_score(atom_array, xyz_a, xyz_b, atom_mask, partner_mask, residue_key):
    """(score, pair count). Score is None when no contacts exist (inconclusive)."""
    pairs = contact_pair_count(
        xyz_a,
        xyz_b,
        atom_mask,
        partner_mask,
        residue_key,
        LDDT_SETTINGS["inclusion_radius"],
    )
    if pairs == 0:
        return None, 0
    ref = atom_array.copy()
    sub = atom_array.copy()
    ref.coord = xyz_a.astype(np.float32)
    sub.coord = xyz_b.astype(np.float32)
    score = struc.lddt(
        ref, sub, atom_mask=atom_mask, partner_mask=partner_mask, **LDDT_SETTINGS
    )
    return float(score), pairs


# ----------------------------------------------------------------------------
# Paired comparison
# ----------------------------------------------------------------------------


@dataclass
class LigandChain:
    chain: str
    indices: np.ndarray
    mol: Any
    maps: list
    truncated: bool
    res_name: str = ""


@dataclass
class Reference:
    atom_array: Any
    layout: ChainLayout
    assignments: list
    anchor_mask: np.ndarray
    polymer_mask: np.ndarray
    type_masks: dict[str, np.ndarray]
    exclude_mask: np.ndarray
    residue_key: np.ndarray
    ligands: list[LigandChain]
    contacts: list[dict[str, Any]] = field(default_factory=list)
    anchor_type: str = ""
    epitope: dict[str, Any] | None = None


def _ligand_chains(atom_array, layout: ChainLayout, cap: int) -> list[LigandChain]:
    out = []
    heavy = atom_array.element != "H"
    bonds = atom_array.bonds.as_array()
    for chain in layout.chain_ids:
        idx = layout.chain_atoms[chain]
        if not atom_array.is_ligand[idx].all():
            continue
        idx = idx[heavy[idx]]
        members = set(idx.tolist())
        fixed = {
            int(i) if int(i) in members else int(j)
            for i, j, _ in bonds
            if (int(i) in members) != (int(j) in members)
        }
        mol = ligand_mol(atom_array, idx, fixed)
        maps, truncated = symmetry_maps(mol, cap)
        res_name = "+".join(dict.fromkeys(atom_array.res_name[idx].tolist()))
        out.append(LigandChain(str(chain), idx, mol, maps, truncated, res_name))
    return out


def build_reference(
    atom_array,
    job: dict[str, Any],
    exclude_window: int | None = None,
    cap: int = LIGAND_CAP,
) -> Reference:
    layout = chain_layout(atom_array, job)
    assignments = valid_assignments(layout)
    if assignments is None:
        raise ValueError("equivalent-chain search space exceeds CHAIN_LIMIT")
    heavy = atom_array.element != "H"
    mol_type = atom_array.mol_type
    type_masks = {
        t: heavy & (mol_type == t) for t in POLYMER_TYPES if (mol_type == t).any()
    }
    polymer_mask = heavy & np.isin(mol_type, POLYMER_TYPES)
    residue_key = np.array(
        [f"{c}:{r}" for c, r in zip(atom_array.chain_id, atom_array.res_id)]
    )

    contacts = []
    exclude = np.zeros(len(atom_array), dtype=bool)
    for c in (job.get("constraint") or {}).get("contact") or []:
        i = constraint_atom(
            layout,
            atom_array,
            c["entity1"],
            c.get("copy1", 1),
            c["position1"],
            c["atom1"],
        )
        j = constraint_atom(
            layout,
            atom_array,
            c["entity2"],
            c.get("copy2", 1),
            c["position2"],
            c["atom2"],
        )
        contacts.append(
            {"atoms": (i, j), "min": c["min_distance"], "max": c["max_distance"]}
        )
        if exclude_window is not None:
            for k in (i, j):
                exclude |= (atom_array.chain_id == atom_array.chain_id[k]) & (
                    np.abs(atom_array.res_id - atom_array.res_id[k]) <= exclude_window
                )

    if "protein" in type_masks:
        anchor_type, anchor_name = "protein", "CA"
        anchor = type_masks["protein"] & (atom_array.atom_name == anchor_name)
    else:
        anchor_type, anchor_name = "nucleic acid", "C4'"
        anchor = polymer_mask & (atom_array.atom_name == anchor_name)
    anchor &= ~exclude
    if anchor.sum() < 3:
        raise ValueError("fewer than 3 anchor atoms")
    return Reference(
        atom_array=atom_array,
        layout=layout,
        assignments=assignments,
        anchor_mask=anchor,
        polymer_mask=polymer_mask,
        type_masks=type_masks,
        exclude_mask=exclude,
        residue_key=residue_key,
        ligands=_ligand_chains(atom_array, layout, cap),
        contacts=contacts,
        epitope=_epitope(atom_array, layout, job),
        anchor_type=f"{anchor_type} {anchor_name}",
    )


def _epitope(atom_array, layout: ChainLayout, job: dict[str, Any]):
    epitope = (job.get("constraint") or {}).get("epitope")
    if not epitope:
        return None
    if epitope.get("paratope", "all") != "all":
        raise ValueError("only paratope 'all' is supported by the harness")
    heavy = atom_array.element != "H"
    residues = []
    seen = set()
    for res in epitope["residues"]:
        chain = layout.entity_chains[str(res["entity"])][int(res.get("copy", 1)) - 1]
        key = (chain, int(res["position"]))
        if key in seen:
            continue
        seen.add(key)
        idx = np.where(
            (atom_array.chain_id == chain) & (atom_array.res_id == key[1]) & heavy
        )[0]
        if not len(idx):
            raise ValueError(f"epitope residue {key} has no heavy atoms")
        residues.append(idx)
    movable = (job.get("constraint") or {}).get("movable_chains") or []
    paratope = np.where(np.isin(atom_array.chain_id, movable) & heavy)[0]
    if not len(paratope):
        raise ValueError("epitope paratope is empty")
    k = max(1, math.ceil(float(epitope.get("min_fraction", 1.0)) * len(residues)))
    return {"residues": residues, "paratope": paratope, "k": k}


def guidance_status(ref: Reference, xyz: np.ndarray) -> dict[str, Any] | None:
    """Constraint satisfaction of one structure: every contact within its
    bounds, and OpenDDE's epitope rule (at least K residues reached)."""
    out: dict[str, Any] = {}
    if ref.contacts:
        dists = [
            float(np.linalg.norm(xyz[c["atoms"][0]] - xyz[c["atoms"][1]]))
            for c in ref.contacts
        ]
        out["contacts_ok"] = all(
            c["min"] <= d <= c["max"] for c, d in zip(ref.contacts, dists)
        )
        out["contact_mean"] = float(np.mean(dists))
    if ref.epitope:
        tree = cKDTree(xyz[ref.epitope["paratope"]])
        reached = sum(
            bool((tree.query(xyz[idx])[0] <= EPITOPE_GATE).any())
            for idx in ref.epitope["residues"]
        )
        out["epitope_reached"] = reached
        out["epitope_ok"] = reached >= ref.epitope["k"]
    return out or None


def _secondary(ref: Reference, a: np.ndarray, b_fit: np.ndarray) -> float:
    """Mean squared deviation of non-anchor heavy atoms under the anchor fit,
    with each ligand at its best symmetry map."""
    ligand_atoms = np.zeros(len(a), dtype=bool)
    total, count = 0.0, 0
    for lig in ref.ligands:
        ligand_atoms[lig.indices] = True
        ra, rb = a[lig.indices], b_fit[lig.indices]
        best = min(((rb[list(m)] - ra) ** 2).sum() for m in lig.maps)
        total += best
        count += len(lig.indices)
    rest = ref.polymer_mask & ~ref.anchor_mask & ~ref.exclude_mask & ~ligand_atoms
    total += float(((b_fit[rest] - a[rest]) ** 2).sum())
    count += int(rest.sum())
    return total / count if count else 0.0


def choose_assignment(ref: Reference, a: np.ndarray, b: np.ndarray):
    """Declared rule: minimize anchor RMSD after anchor superposition; break
    exact ties by the non-anchor deviation; then by enumeration order."""
    best = None
    for order, mapping in enumerate(ref.assignments):
        idx = permutation_index(ref.layout, mapping, len(a))
        bp = b[idx]
        r, t = kabsch(a[ref.anchor_mask], bp[ref.anchor_mask])
        b_fit = bp @ r.T + t
        key = (
            rmsd(a[ref.anchor_mask], b_fit[ref.anchor_mask]),
            _secondary(ref, a, b_fit),
            order,
        )
        if best is None or key < best[0]:
            best = (key, mapping, b_fit)
    return best


def compare_pair(ref: Reference, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
    }
    if not out["finite"]:
        out["status"] = "fail: non-finite coordinates"
        return out
    (anchor_rmsd, _, order), mapping, b_fit = choose_assignment(ref, a, b)
    out["anchor"] = ref.anchor_type
    out["assignments"] = len(ref.assignments)
    out["assignment"] = {k: v for k, v in mapping.items() if k != v}
    out["anchor_rmsd"] = anchor_rmsd
    keep = ref.polymer_mask & ~ref.exclude_mask
    out["polymer_rmsd"] = rmsd(a[keep], b_fit[keep])
    errors = np.linalg.norm(a[keep] - b_fit[keep], axis=1)
    out["atom_error"] = {
        f"p{q}": float(np.percentile(errors, q)) for q in ATOM_ERROR_PERCENTILES
    }
    out["atom_error"]["max"] = float(errors.max())

    lddt = {}
    status = "ok"
    for name, mask in [("polymer", ref.polymer_mask)] + list(ref.type_masks.items()):
        mask = mask & ~ref.exclude_mask
        score, pairs = lddt_score(ref.atom_array, a, b_fit, mask, mask, ref.residue_key)
        lddt[name] = {"score": score, "pairs": pairs}
        if score is None:
            status = "inconclusive: no lDDT contacts"
    out["lddt"] = lddt

    interface_mask = ref.polymer_mask & ~ref.exclude_mask
    if len(set(ref.atom_array.chain_id[interface_mask].tolist())) > 1:
        chain_id = np.asarray(ref.atom_array.chain_id)
        contacts_a = interchain_contacts(a, interface_mask, chain_id, ref.residue_key)
        contacts_b = interchain_contacts(
            b_fit, interface_mask, chain_id, ref.residue_key
        )
        out["interface"] = {
            "contacts_a": len(contacts_a),
            "contacts_b": len(contacts_b),
            "fnat": len(contacts_a & contacts_b) / len(contacts_a)
            if contacts_a
            else None,
            "fnonnat": len(contacts_b - contacts_a) / len(contacts_b)
            if contacts_b
            else None,
        }

    polymer_xyz_a, polymer_xyz_b = a[ref.polymer_mask], b_fit[ref.polymer_mask]
    polymer_res = ref.residue_key[ref.polymer_mask]
    ligands = []
    for lig in ref.ligands:
        ra, rb = a[lig.indices], b_fit[lig.indices]
        pocket = []
        for xyz_p, xyz_l in ((polymer_xyz_a, ra), (polymer_xyz_b, rb)):
            near = cKDTree(xyz_l).query_ball_point(
                xyz_p, POCKET_DISTANCE, return_length=True
            )
            pocket.append(set(polymer_res[near > 0].tolist()))
        union = pocket[0] | pocket[1]
        ligands.append(
            {
                "chain": lig.chain,
                "res_name": lig.res_name,
                "heavy_atoms": len(lig.indices),
                "maps": len(lig.maps),
                "truncated": lig.truncated,
                "rmsd": symmetric_rmsd(lig.mol, ra, rb, lig.maps),
                "centroid_shift": float(
                    np.linalg.norm(ra.mean(axis=0) - rb.mean(axis=0))
                ),
                "pocket_jaccard": len(pocket[0] & pocket[1]) / len(union)
                if union
                else 1.0,
                "pocket_sizes": [len(pocket[0]), len(pocket[1])],
            }
        )
        if lig.truncated and status == "ok":
            status = "inconclusive: ligand map search truncated"
    out["ligands"] = ligands

    if ref.contacts:
        rows = []
        for c in ref.contacts:
            i, j = c["atoms"]
            da = float(np.linalg.norm(a[i] - a[j]))
            db = float(np.linalg.norm(b_fit[i] - b_fit[j]))
            rows.append(
                {
                    "a": da,
                    "b": db,
                    "diff": abs(db - da),
                    "a_ok": c["min"] <= da <= c["max"],
                    "b_ok": c["min"] <= db <= c["max"],
                }
            )
        out["contacts"] = rows
    guidance_a, guidance_b = guidance_status(ref, a), guidance_status(ref, b_fit)
    if guidance_a is not None:
        out["guidance"] = {"a": guidance_a, "b": guidance_b}
    out["status"] = status
    return out


def anchor_distance(ref: Reference, a: np.ndarray, b: np.ndarray) -> float:
    """Anchor RMSD under the declared assignment rule, for set-level matching."""
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    return choose_assignment(ref, a, b)[0][0]
