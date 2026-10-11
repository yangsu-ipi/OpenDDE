"""Tests for the qualification structure-comparison helpers.

These cover the step 0 checks the porting plan requires before the ligand
symmetry rule is frozen: explicit-map parity with RDKit's automatic CalcRMS on
carboxylate, nitro and phosphate fixtures, a fixed covalent attachment atom,
the match-cap boundary and repeatable enumeration. They also cover the
equivalent-chain assignment and the lDDT contact count.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit")
import biotite.structure as struc  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem, rdMolAlign  # noqa: E402

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts" / "qualification")
)
import qual_structures as qs  # noqa: E402


def _embedded(smiles: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    mol = Chem.RemoveHs(mol)
    return mol, mol.GetConformer().GetPositions()


@pytest.mark.parametrize(
    "smiles",
    [
        "CC(=O)[O-]",
        "OC(=O)c1ccccc1",
        "[O-][N+](=O)c1ccccc1",
        "CP(=O)(O)O",
    ],
)
def test_explicit_map_matches_rdkit_automatic(smiles):
    mol, ref = _embedded(smiles)
    terminal = sorted({m[0] for m in mol.GetSubstructMatches(qs.TERMINAL_GROUP)})
    assert len(terminal) >= 2
    prb = ref.copy()
    prb[[terminal[0], terminal[1]]] = prb[[terminal[1], terminal[0]]]
    prb += np.random.default_rng(0).normal(scale=0.1, size=prb.shape)

    maps, truncated = qs.symmetry_maps(mol)
    ours = qs.symmetric_rmsd(mol, ref, prb, maps)
    automatic = rdMolAlign.CalcRMS(
        qs._with_conformer(mol, prb), qs._with_conformer(mol, ref)
    )
    assert not truncated
    assert ours == pytest.approx(automatic, abs=1e-6)
    numpy_best = min(
        np.sqrt(((prb[list(m)] - ref) ** 2).sum(axis=1).mean()) for m in maps
    )
    assert ours == pytest.approx(numpy_best, abs=1e-6)

    plain, _ = qs.symmetry_maps(mol, symmetrize=False)
    if len(plain) < len(maps):
        # Without the terminal-group equivalence the swap looks like a pose error.
        assert qs.symmetric_rmsd(mol, ref, prb, plain) > ours + 0.3


def _carbon_graph(n_atom: int, bonds: list[tuple[int, int]]):
    atoms = struc.AtomArray(n_atom)
    atoms.element[:] = "C"
    atoms.bonds = struc.BondList(
        n_atom, np.array([(i, j, struc.BondType.SINGLE) for i, j in bonds])
    )
    return atoms


def test_attachment_atom_stays_fixed():
    atoms = _carbon_graph(4, [(0, 1), (0, 2), (0, 3)])
    idx = np.arange(4)
    free = qs.ligand_mol(atoms, idx, set())
    fixed = qs.ligand_mol(atoms, idx, {1})
    free_maps, _ = qs.symmetry_maps(free)
    fixed_maps, _ = qs.symmetry_maps(fixed)
    assert len(free_maps) == 6
    assert len(fixed_maps) == 2
    assert all(m[1] == 1 for m in fixed_maps)

    ref = np.array([[0, 0, 0], [1.5, 0, 0], [-0.5, 1.4, 0], [-0.5, -0.7, 1.2]])
    prb = ref.copy()
    prb[[1, 2]] = prb[[2, 1]]
    assert qs.symmetric_rmsd(free, ref, prb, free_maps) == pytest.approx(0, abs=1e-6)
    assert qs.symmetric_rmsd(fixed, ref, prb, fixed_maps) > 0.5


def test_match_cap_boundary_and_repeatability():
    atoms = _carbon_graph(5, [(0, 1), (0, 2), (0, 3), (0, 4)])
    mol = qs.ligand_mol(atoms, np.arange(5), set())
    maps, truncated = qs.symmetry_maps(mol, cap=24)
    assert len(maps) == 24 and not truncated
    capped, truncated = qs.symmetry_maps(mol, cap=23)
    assert len(capped) == 23 and truncated
    again, _ = qs.symmetry_maps(mol, cap=24)
    assert again == maps


def test_ligand_mol_aromatic_and_orders():
    mol, _ = _embedded("OC(=O)c1ccccc1")
    n = mol.GetNumAtoms()
    atoms = struc.AtomArray(n)
    atoms.element[:] = [a.GetSymbol().upper() for a in mol.GetAtoms()]
    kinds = {
        Chem.BondType.SINGLE: struc.BondType.SINGLE,
        Chem.BondType.DOUBLE: struc.BondType.DOUBLE,
        Chem.BondType.AROMATIC: struc.BondType.AROMATIC,
    }
    atoms.bonds = struc.BondList(
        n,
        np.array(
            [
                (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), kinds[b.GetBondType()])
                for b in mol.GetBonds()
            ]
        ),
    )
    built = qs.ligand_mol(atoms, np.arange(n), set())
    assert len(qs.symmetry_maps(built)[0]) == len(qs.symmetry_maps(mol)[0]) == 4


def _homodimer(n_res: int = 6):
    atoms = struc.AtomArray(2 * n_res)
    atoms.chain_id[:] = ["A"] * n_res + ["B"] * n_res
    atoms.res_id[:] = list(range(1, n_res + 1)) * 2
    atoms.res_name[:] = "ALA"
    atoms.atom_name[:] = "CA"
    atoms.element[:] = "C"
    atoms.hetero[:] = False
    atoms.add_annotation("mol_type", dtype="U8")
    atoms.mol_type[:] = "protein"
    atoms.add_annotation("is_ligand", dtype=bool)
    atoms.add_annotation("label_entity_id", dtype="U4")
    atoms.label_entity_id[:] = "1"
    atoms.bonds = struc.BondList(2 * n_res)
    x = np.arange(n_res) * 3.8
    atoms.coord[:n_res] = np.stack([x, np.zeros(n_res), np.zeros(n_res)], axis=1)
    atoms.coord[n_res:] = np.stack([x, np.full(n_res, 6.0), np.sin(x) * 2.0], axis=1)
    job = {"sequences": [{"proteinChain": {"sequence": "A" * n_res, "count": 2}}]}
    return atoms, job


def test_chain_assignment_recovers_swapped_copies():
    atoms, job = _homodimer()
    ref = qs.build_reference(atoms, job)
    assert len(ref.assignments) == 2
    a = atoms.coord.astype(np.float64)
    n = len(atoms) // 2
    b = np.concatenate([a[n:], a[:n]])
    theta = 0.7
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    b = b @ rot.T + np.array([5.0, -2.0, 1.0])
    result = qs.compare_pair(ref, a, b)
    assert result["status"] == "ok"
    assert result["assignment"] == {"A": "B", "B": "A"}
    assert result["anchor_rmsd"] == pytest.approx(0, abs=1e-5)
    assert result["lddt"]["polymer"]["score"] == pytest.approx(1.0)
    assert result["lddt"]["polymer"]["pairs"] > 0


def test_constraint_named_chain_fixes_assignment():
    atoms, job = _homodimer()
    job["constraint"] = {"movable_chains": ["B"]}
    assert len(qs.build_reference(atoms, job).assignments) == 1


def test_lddt_without_contacts_is_inconclusive():
    atoms = struc.AtomArray(2)
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = [1, 2]
    atoms.atom_name[:] = "CA"
    atoms.element[:] = "C"
    far = np.array([[0.0, 0, 0], [50.0, 0, 0]])
    mask = np.ones(2, dtype=bool)
    key = np.array(["A:1", "A:2"])
    assert qs.lddt_score(atoms, far, far, mask, mask, key) == (None, 0)
    near = np.array([[0.0, 0, 0], [5.0, 0, 0]])
    score, pairs = qs.lddt_score(atoms, near, near, mask, mask, key)
    assert pairs == 1 and score == pytest.approx(1.0)


def test_epitope_rule_and_atom_error_percentiles():
    atoms, job = _homodimer()
    job["constraint"] = {
        "movable_chains": ["B"],
        "epitope": {
            "residues": [{"entity": "1", "copy": 1, "position": "1", "residue": "A"}],
            "paratope": "all",
            "min_fraction": 1.0,
        },
    }
    ref = qs.build_reference(atoms, job)
    a = atoms.coord.astype(np.float64)
    assert qs.guidance_status(ref, a) == {"epitope_reached": 0, "epitope_ok": False}
    close = a.copy()
    close[6] = close[0] + np.array([0.0, 4.0, 0.0])
    assert qs.guidance_status(ref, close)["epitope_ok"]
    result = qs.compare_pair(ref, a, a)
    assert result["atom_error"]["p95"] == pytest.approx(0, abs=1e-6)
    assert result["guidance"]["a"] == result["guidance"]["b"]
