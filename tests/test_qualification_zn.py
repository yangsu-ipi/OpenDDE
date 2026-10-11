"""Tests for the Zn site measurement (scripts/qualification/zn_sites.py)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rdkit")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "qualification"))
import zn_sites as zn  # noqa: E402
from qual_structures import kabsch  # noqa: E402


def _rotation(degrees: float) -> np.ndarray:
    t = np.radians(degrees)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def _toy():
    """Two pockets of four donor atoms each, 20 A apart, with a Zn in each."""
    rng = np.random.default_rng(0)
    pocket = rng.normal(scale=2.0, size=(4, 3))
    pocket = pocket / np.linalg.norm(pocket, axis=1, keepdims=True) * 2.2
    xyz = np.concatenate([pocket, pocket + [20, 0, 0], [[0, 0, 0], [20, 0, 0]]])
    atoms = SimpleNamespace(
        chain_id=np.array(["A"] * 8 + ["Z1", "Z2"]),
        res_name=np.array(["HIS"] * 8 + ["ZN", "ZN"]),
        res_id=np.array([1, 2, 3, 4, 11, 12, 13, 14, 1, 1]),
    )
    donors = np.arange(8)
    ions = np.array([8, 9])
    return atoms, xyz, donors, ions


def test_coordination_uses_cutoff():
    _, xyz, donors, ions = _toy()
    coord = zn.coordination(xyz, ions, donors)
    assert [c.tolist() for c in coord] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert [len(c) for c in zn.coordination(xyz, ions, donors, cutoff=2.0)] == [0, 0]


def test_assign_is_one_to_one():
    cost = np.array([[5.0, 1.0], [1.0, 5.0]])
    assert sorted(zn.assign(cost)) == [(0, 1), (1, 0)]
    assert zn.assign(np.zeros((0, 2))) == []


def test_swapped_ions_are_not_displacements():
    atoms, xyz, donors, ions = _toy()
    cand = xyz.copy()
    cand[[8, 9]] = cand[[9, 8]]
    out = zn.pair_ions(atoms, xyz, cand, ions, donors, donors)
    assert out["bound_off"] == out["bound_cand"] == 2
    for ion in out["ions"]:
        assert ion["pocket_distance"] == pytest.approx(0.0, abs=1e-9)
        assert ion["jaccard"] == 1.0
        assert ion["off_ion"] != ion["cand_ion"]


def test_pocket_frame_ignores_rigid_motion_and_detects_a_moved_ion():
    atoms, xyz, donors, ions = _toy()
    cand = xyz @ _rotation(30).T + [3.0, -1.0, 2.0]
    out = zn.pair_ions(atoms, xyz, cand, ions, donors, donors)
    assert max(i["pocket_distance"] for i in out["ions"]) < 1e-9
    cand[9] += [0.0, 0.0, 4.0]
    out = zn.pair_ions(atoms, xyz, cand, ions, donors, donors)
    moved = [i for i in out["ions"] if i["off_ion"] == 9][0]
    assert moved["pocket_distance"] == pytest.approx(4.0, abs=1e-6)
    assert moved["jaccard"] < 1.0


def test_pocket_superpose_needs_three_atoms():
    _, xyz, _, _ = _toy()
    r, t, n = zn.pocket_superpose(xyz, xyz, xyz[8], np.array([0, 1]))
    assert r is None and n == 2
    r, t, n = zn.pocket_superpose(xyz, xyz, xyz[8], np.arange(8))
    assert n == 4
    ref_r, _ = kabsch(xyz[:4], xyz[:4])
    assert np.allclose(r, ref_r)
