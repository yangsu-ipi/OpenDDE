"""Tests for the exact tier 1 transfer comparison (scripts/porting/tier1_payload_compare.py).

Synthetic roots follow the calibration layout: ``cap_off`` reference captures,
``cap_<c>`` contrast captures replayed from them, ``t1_*`` tier 1 runs with run
records and one-sample prediction artifacts, and optional full runs.
"""

import json
import shutil
import zlib
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "porting"))
sys.path.insert(0, str(ROOT / "scripts" / "qualification"))

import accept  # noqa: E402
import tier1_payload_compare as tpc  # noqa: E402

ORIG = "/orig/porting/runs/cal"
SEEDS = [101, 102]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A manifest with two development rows (R2 has two jobs) and a held-out row."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "r1.json").write_text(json.dumps([{"name": "a"}]))
    (tmp_path / "in" / "r2.json").write_text(json.dumps([{"name": "b"}, {"name": "c"}]))
    manifest = {
        "common_flags": ["--sample", "2", "--step", "200"],
        "development_seeds": SEEDS,
        "rows": [
            {"id": "R1", "role": "development", "input": "in/r1.json"},
            {"id": "R2", "role": "development", "input": "in/r2.json"},
            {"id": "H1", "role": "held-out", "input": "in/r1.json"},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return tmp_path, path


JOBS = {"R1": ["a"], "R2": ["b", "c"]}


def _tensors(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "s_inputs": torch.randn(4, 3, generator=g),
        "z_trunk": torch.randn(4, 4, 2, generator=g),
        "distogram_probs": torch.rand(4, 4, 5, generator=g),
    }


def _head(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed + 7)
    return {
        "plddt_logits": torch.randn(3, 2, generator=g),
        "pae_logits": torch.randn(3, 3, generator=g).to(torch.float16),
        "ptm": torch.tensor([0.5]),
        "iptm": torch.tensor([0.25]),
    }


def make_root(root: Path, contrasts=("k",), exclude=(), full=False, prefix=ORIG):
    """Write a complete calibration root. Replay paths and run records use
    ``prefix`` in place of ``root`` (as the VM paths are stored in the archive)."""
    excluded = set(exclude)
    for role in ["off", *contrasts]:
        for row, jobs in JOBS.items():
            if (row, role) in excluded:
                continue
            for job in jobs:
                for seed in SEEDS:
                    base = zlib.crc32(f"{row}/{job}/{seed}".encode()) % 1000
                    rec = _tensors(base)
                    if role == "off":
                        rec["x_pred_coords"] = torch.ones(1, 6, 3)
                        rec["head"] = _head(base)
                    else:
                        rec["head_fixed"] = _head(base + 1)
                        rec["replay_source"] = (
                            f"{prefix}/cap_off/{row}/{job}/seed_{seed}/tier1.pt"
                        )
                    out = root / f"cap_{role}" / row / job / f"seed_{seed}" / "tier1.pt"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(rec, out)
            run = root / f"t1_{role}"
            run.mkdir(parents=True, exist_ok=True)
            (run / f"{row}.run.json").write_text(
                json.dumps(
                    {
                        "exit_code": 0,
                        "wrapper": {
                            "tier1_capture": f"{prefix}/cap_{role}/{row}",
                            "tier1_replay": None
                            if role == "off"
                            else f"{prefix}/cap_off/{row}",
                        },
                    }
                )
            )
            for path in accept.expected_artifacts(jobs, SEEDS, 1):
                f = run / row / path
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(f"{role} {row} {path}")
            if full:
                for path in accept.expected_artifacts(jobs, SEEDS, 2):
                    f = root / role / row / path
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(f"full {role} {row} {path}")
    return root


def _inv(manifest, contrasts=("k",), exclude=(), full=False):
    return tpc.declare(manifest, ["R1", "R2"], list(contrasts), list(exclude), full, [])


def _cmp(inv, a, b, prefix_a=ORIG, prefix_b=ORIG):
    return tpc.compare(
        inv,
        a,
        b,
        tpc.parse_relocations([f"{prefix_a}={a}"]),
        tpc.parse_relocations([f"{prefix_b}={b}"]),
    )


def _pair(world, **kw):
    tmp, manifest = world
    a = make_root(tmp / "a", **kw)
    b = make_root(tmp / "b", prefix="/other/host/cal", **kw)
    return manifest, a, b


def _edit(path: Path, fn):
    rec = torch.load(path, weights_only=True)
    fn(rec)
    torch.save(rec, path)


def _flip_bit(rec):
    bits = rec["z_trunk"].view(-1).view(torch.int32)
    bits[0] = bits[0] ^ 1


def _fails(report, text):
    assert not report["ok"]
    assert any(text in f for f in report["failures"]), report["failures"][:10]


# ------------------------------------------------------------- declare


def test_declaration_records_manifest_and_frozen_modules(world):
    _, manifest = world
    inv = _inv(manifest)
    assert inv["manifest"]["absolute"] == str(manifest.resolve())
    assert inv["manifest"]["sha256"] == tpc.sha256_file(manifest)
    assert set(inv["frozen_modules"]) == {"accept.py", "compare_outputs.py"}
    assert inv["frozen_modules"]["accept.py"]["sha256"] == tpc.sha256_file(
        Path(accept.__file__)
    )


def test_declare_counts_and_independence_from_disk(world):
    _, manifest = world
    inv = _inv(manifest, contrasts=("k", "fp32"), exclude=[("R2", "fp32")], full=True)
    assert inv["counts"]["captures"] == {"off": 6, "k": 6, "fp32": 2}
    assert inv["counts"]["artifacts"]["t1_off"] == 6 * 3
    assert inv["counts"]["artifacts"]["t1_fp32"] == 2 * 3
    assert inv["counts"]["artifacts"]["fp32"] == 2 * 2 * 3
    assert inv["samples"] == {"tier1": 1, "full": 2}
    # the excluded row keeps its reference captures
    assert {c["row"] for c in inv["captures"] if c["role"] == "off"} == {"R1", "R2"}


def test_declare_refuses_held_out_rows_and_bad_exclusions(world):
    _, manifest = world
    with pytest.raises(SystemExit, match="not a development row"):
        tpc.declare(manifest, ["R1", "H1"], ["k"], [], False, [])
    with pytest.raises(SystemExit, match="names no declared recipe"):
        tpc.declare(manifest, ["R1"], ["k"], [("R1", "fp32")], False, [])
    with pytest.raises(SystemExit, match="bad contrast"):
        tpc.declare(manifest, ["R1"], ["off"], [], False, [])


def test_expected_artifacts_follow_the_frozen_verifier(world):
    _, manifest = world
    inv = _inv(manifest)
    want = sorted(
        f"R2/{p}" for p in accept.expected_artifacts(["b", "c"], SEEDS, 1)
    ) + sorted(f"R1/{p}" for p in accept.expected_artifacts(["a"], SEEDS, 1))
    assert inv["artifacts"]["t1_k"] == sorted(want)


# ------------------------------------------------------------- identity


def test_identical_roots_with_relocated_paths_pass(world):
    manifest, a, b = _pair(world, contrasts=("k",), full=True)
    report = _cmp(_inv(manifest, full=True), a, b, prefix_b="/other/host/cal")
    assert report["ok"], report["failures"][:10]
    assert report["summary"]["captures_passed"] == 12
    # replay_source differs between the sides, so contrast files differ in bytes
    k = [e for e in report["captures"] if e["role"] == "k"]
    assert all(e["payload_equal"] and not e["file_hash_equal"] for e in k)


def test_archive_self_comparison_passes(world):
    tmp, manifest = world
    a = make_root(tmp / "a", full=True)
    assert _cmp(_inv(manifest, full=True), a, a)["ok"]


def test_different_strides_with_equal_values_pass(world):
    manifest, a, b = _pair(world)
    path = b / "cap_off" / "R1" / "a" / "seed_101" / "tier1.pt"

    def restride(rec):
        z = rec["z_trunk"]
        rec["z_trunk"] = z.transpose(0, 1).contiguous().transpose(0, 1)
        assert rec["z_trunk"].stride() != z.stride()

    _edit(path, restride)
    report = _cmp(_inv(manifest), a, b, prefix_b="/other/host/cal")
    assert report["ok"], report["failures"][:10]
    entry = next(
        e
        for e in report["captures"]
        if e["role"] == "off" and e["row"] == "R1" and e["seed"] == 101
    )
    assert entry["payload_equal"] and not entry["file_hash_equal"]
    assert report["summary"]["serialization_only_differences"] >= 1


@pytest.mark.parametrize(
    "edit, text",
    [
        (_flip_bit, "tensor bits differ"),
        (lambda r: r.pop("distogram_probs"), "missing on b"),
        (lambda r: r.__setitem__("extra", torch.zeros(1)), "missing on a"),
        (lambda r: r["head"].pop("ptm"), "missing on b"),
        (lambda r: r.__setitem__("s_inputs", r["s_inputs"].reshape(3, 4)), "shape"),
        (lambda r: r.__setitem__("s_inputs", r["s_inputs"].double()), "dtype"),
    ],
    ids=["one-bit", "missing-key", "added-key", "nested-key", "shape", "dtype"],
)
def test_payload_faults_fail(world, edit, text):
    manifest, a, b = _pair(world)
    _edit(b / "cap_off" / "R2" / "c" / "seed_102" / "tier1.pt", edit)
    _fails(_cmp(_inv(manifest), a, b, prefix_b="/other/host/cal"), text)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_tensors_fail_even_when_identical(world, value):
    tmp, manifest = world
    a = make_root(tmp / "a")
    _edit(
        a / "cap_k" / "R1" / "a" / "seed_101" / "tier1.pt",
        lambda r: r["head_fixed"]["plddt_logits"].view(-1).__setitem__(0, value),
    )
    _fails(_cmp(_inv(manifest), a, a), "non-finite")


def test_float_and_string_values_compare_exactly(world):
    manifest, a, b = _pair(world)
    pa = a / "cap_off" / "R1" / "a" / "seed_102" / "tier1.pt"
    pb = b / "cap_off" / "R1" / "a" / "seed_102" / "tier1.pt"
    _edit(pa, lambda r: r.update(scale=0.1, note="x"))
    _edit(pb, lambda r: r.update(scale=0.1 + 2**-56, note="x"))
    _fails(_cmp(_inv(manifest), a, b, prefix_b="/other/host/cal"), "float bits differ")
    _edit(pb, lambda r: r.update(scale=0.1, note="y"))
    _fails(_cmp(_inv(manifest), a, b, prefix_b="/other/host/cal"), "'x' vs 'y'")


# ------------------------------------------------------------- provenance


def test_replay_path_naming_another_reference_fails(world):
    manifest, a, b = _pair(world)
    # make seed 102's reference payload identical to seed 101's, then point
    # seed 101's contrast at seed 102: the payloads match but the role does not
    off = b / "cap_off" / "R1" / "a"
    shutil.copy(off / "seed_101" / "tier1.pt", off / "seed_102" / "tier1.pt")
    shutil.copy(
        a / "cap_off/R1/a/seed_101/tier1.pt", a / "cap_off/R1/a/seed_102/tier1.pt"
    )
    _edit(
        b / "cap_k" / "R1" / "a" / "seed_101" / "tier1.pt",
        lambda r: r.update(
            replay_source="/other/host/cal/cap_off/R1/a/seed_102/tier1.pt"
        ),
    )
    _fails(_cmp(_inv(manifest), a, b, prefix_b="/other/host/cal"), "does not name")


def test_unrelocated_replay_path_fails(world):
    manifest, a, b = _pair(world)
    report = tpc.compare(_inv(manifest), a, b, [], [])
    _fails(report, "does not name")


def test_run_record_naming_another_replay_dir_fails(world):
    manifest, a, b = _pair(world)
    rec_path = b / "t1_k" / "R2.run.json"
    rec = json.loads(rec_path.read_text())
    rec["wrapper"]["tier1_replay"] = "/other/host/cal/cap_off/R1"
    rec_path.write_text(json.dumps(rec))
    _fails(_cmp(_inv(manifest), a, b, prefix_b="/other/host/cal"), "replay dir")


def test_failed_reference_fails_its_contrast(world):
    manifest, a, b = _pair(world)
    _edit(
        b / "cap_off" / "R1" / "a" / "seed_101" / "tier1.pt",
        lambda r: r["s_inputs"].mul_(2),
    )
    report = _cmp(_inv(manifest), a, b, prefix_b="/other/host/cal")
    _fails(report, "reference capture did not pass")


def test_role_structure_is_enforced(world):
    manifest, a, b = _pair(world)
    for root in (a, b):
        _edit(
            root / "cap_off" / "R1" / "a" / "seed_101" / "tier1.pt",
            lambda r: r.update(replay_source="x"),
        )
    _fails(
        _cmp(_inv(manifest), a, b, prefix_b="/other/host/cal"),
        "unexpected replay_source for role off",
    )


# ------------------------------------------------------------- inventory


def test_capture_missing_on_one_side_fails(world):
    manifest, a, b = _pair(world)
    (b / "cap_k" / "R2" / "b" / "seed_101" / "tier1.pt").unlink()
    _fails(
        _cmp(_inv(manifest), a, b, prefix_b="/other/host/cal"),
        "b: missing capture k/R2/b/101",
    )


def test_same_capture_missing_on_both_sides_fails(world):
    manifest, a, b = _pair(world)
    for root in (a, b):
        (root / "cap_k" / "R2" / "b" / "seed_101" / "tier1.pt").unlink()
    report = _cmp(_inv(manifest), a, b, prefix_b="/other/host/cal")
    _fails(report, "a: missing capture k/R2/b/101")
    _fails(report, "b: missing capture k/R2/b/101")


def test_two_empty_roots_fail(world):
    tmp, manifest = world
    (tmp / "a").mkdir()
    (tmp / "b").mkdir()
    _fails(_cmp(_inv(manifest), tmp / "a", tmp / "b"), "missing capture")


def test_same_unexpected_capture_on_both_sides_fails(world):
    manifest, a, b = _pair(world)
    for root in (a, b):
        src = root / "cap_k" / "R1" / "a" / "seed_101" / "tier1.pt"
        dst = root / "cap_k" / "R1" / "a" / "seed_103" / "tier1.pt"
        dst.parent.mkdir()
        shutil.copy(src, dst)
    report = _cmp(_inv(manifest), a, b, prefix_b="/other/host/cal")
    _fails(report, "a: unexpected capture k/R1/a/103")
    _fails(report, "b: unexpected capture k/R1/a/103")


def test_same_unexpected_artifact_on_both_sides_fails(world):
    manifest, a, b = _pair(world)
    for root in (a, b):
        (
            root / "t1_off" / "R1" / "a" / "seed_101" / "predictions" / "a_sample_9.cif"
        ).write_text("x")
    _fails(
        _cmp(_inv(manifest), a, b, prefix_b="/other/host/cal"), "unexpected artifact"
    )


def test_artifact_byte_difference_fails(world):
    manifest, a, b = _pair(world, full=True)
    (b / "off" / "R2" / "c" / "seed_102" / "predictions" / "c_sample_1.cif").write_text(
        "changed"
    )
    _fails(
        _cmp(_inv(manifest, full=True), a, b, prefix_b="/other/host/cal"),
        "bytes differ",
    )


def test_d4_style_gap_allows_only_the_excluded_contrast(world):
    tmp, manifest = world
    gap = dict(contrasts=("k", "fp32"), exclude=[("R2", "fp32")])
    a = make_root(tmp / "a", **gap)
    b = make_root(tmp / "b", prefix="/other/host/cal", **gap)
    inv = _inv(manifest, **gap)
    assert _cmp(inv, a, b, prefix_b="/other/host/cal")["ok"]
    # the excluded captures appearing on one side are unexpected
    extra = b / "cap_fp32" / "R2" / "b" / "seed_101" / "tier1.pt"
    extra.parent.mkdir(parents=True)
    shutil.copy(b / "cap_k" / "R2" / "b" / "seed_101" / "tier1.pt", extra)
    _fails(
        _cmp(inv, a, b, prefix_b="/other/host/cal"), "unexpected capture fp32/R2/b/101"
    )


def test_symmetric_omission_of_the_gap_rows_reference_fails(world):
    tmp, manifest = world
    gap = dict(contrasts=("k", "fp32"), exclude=[("R2", "fp32")])
    a = make_root(tmp / "a", **gap)
    b = make_root(tmp / "b", prefix="/other/host/cal", **gap)
    for root in (a, b):
        shutil.rmtree(root / "cap_off" / "R2")
    report = _cmp(_inv(manifest, **gap), a, b, prefix_b="/other/host/cal")
    _fails(report, "a: missing capture off/R2/b/101")
    _fails(report, "b: missing capture off/R2/b/101")


# ------------------------------------------------------------- command line


def test_command_line_exit_status(world):
    manifest, a, b = _pair(world)
    tmp = manifest.parent
    script = ROOT / "scripts" / "porting" / "tier1_payload_compare.py"
    inv = tmp / "inv.json"
    run = lambda *args: subprocess.run(
        [sys.executable, str(script), *args],  # noqa: E731
        capture_output=True,
        text=True,
        cwd=tmp,
    )
    assert (
        run(
            "declare",
            "--manifest",
            str(manifest),
            "--rows",
            "R1,R2",
            "--contrasts",
            "k",
            "--out",
            str(inv),
        ).returncode
        == 0
    )
    assert (
        run(
            "declare",
            "--manifest",
            str(manifest),
            "--rows",
            "R1",
            "--contrasts",
            "k",
            "--out",
            str(inv),
        ).returncode
        != 0
    )  # never rewritten
    ok = run(
        "compare",
        "--inventory",
        str(inv),
        "--a",
        str(a),
        "--b",
        str(b),
        "--relocate-a",
        f"{ORIG}={a}",
        "--relocate-b",
        f"/other/host/cal={b}",
        "--out",
        str(tmp / "r.json"),
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
    report = json.loads((tmp / "r.json").read_text())
    assert report["helper"]["sha256"] == tpc.sha256_file(script)
    assert report["inventory"]["sha256"] == tpc.sha256_file(inv)
    qual = ROOT / "scripts" / "qualification"
    for name in ("accept.py", "compare_outputs.py"):
        entry = report["frozen_modules"][name]
        assert Path(entry["path"]) == (qual / name).resolve()
        assert entry["sha256"] == tpc.sha256_file(qual / name)
    declared = json.loads(inv.read_text())
    assert declared["frozen_modules"] == report["frozen_modules"]
    assert report["manifest"] == declared["manifest"]
    assert Path(declared["manifest"]["absolute"]) == manifest.resolve()
    assert declared["manifest"]["sha256"] == tpc.sha256_file(manifest)
    (b / "cap_k" / "R1" / "a" / "seed_101" / "tier1.pt").unlink()
    bad = run(
        "compare",
        "--inventory",
        str(inv),
        "--a",
        str(a),
        "--b",
        str(b),
        "--relocate-a",
        f"{ORIG}={a}",
        "--relocate-b",
        f"/other/host/cal={b}",
        "--out",
        str(tmp / "r2.json"),
    )
    assert bad.returncode == 1
