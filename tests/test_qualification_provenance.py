"""Tests for contract-6 run provenance (scripts/qualification/run_opendde.py,
provenance.py, assets.py, inventory.py and run_audit.py).

The wrapper tests run ``run_opendde.main`` with a stand-in for ``pred`` that
performs the real input preprocessing (``preprocess_input``, including the
legacy MSA conversion) and calls the real parser-read target, so the
recorded input provenance is that of the stock code path without a model."""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "qualification"))
import assets  # noqa: E402
import inventory  # noqa: E402
import provenance  # noqa: E402
import run_audit  # noqa: E402

INPUTS = ROOT / "scripts" / "qualification" / "inputs"
D1 = "examples/cofold/protein_200.json"
D4 = "scripts/qualification/inputs/d4_7wux.json"
D5 = "examples/cofold/template/example_9fm7.json"
EFFECTIVE = json.loads((INPUTS / "effective_inputs.json").read_text())


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _wrapper(monkeypatch, pred=None, configs=None):
    """``run_opendde`` with only the parser-read hook installed and ``pred``
    replaced by ``pred(cli_args)`` (default: preprocess, parse, lay out
    every job's seeds)."""
    import run_opendde
    from runner import batch_inference, cli
    from runner import inference

    def install(timer, trace, tier1, prov=None):
        if prov is not None:
            run_opendde._install_provenance(prov)
            if configs is not None:
                prov.capture_config(configs)

    def default_pred(args):
        src, out = args[args.index("-i") + 1], Path(args[args.index("-o") + 1])
        use_msa = (
            args[args.index("--use_msa") + 1] == "true" if "--use_msa" in args else True
        )
        path = batch_inference.preprocess_input(src, out_dir=str(out), use_msa=use_msa)
        jobs = inference._load_inference_jobs_synchronized(path)
        seeds = args[args.index("--seeds") + 1].split(",")
        for job in jobs:
            for s in seeds:
                (out / job["name"] / f"seed_{s}").mkdir(parents=True)

    # Restored after each test, since the hook wraps the module attribute.
    monkeypatch.setattr(
        inference,
        "_load_inference_jobs_synchronized",
        inference._load_inference_jobs_synchronized,
    )
    monkeypatch.setattr(run_opendde, "_install", install)
    monkeypatch.setattr(run_opendde, "fingerprint", lambda: {"commit": "c"})
    monkeypatch.setattr(
        cli.opendde_cli, "main", lambda args, **kw: (pred or default_pred)(args)
    )
    return run_opendde


def _run(monkeypatch, tmp_path, cli, overrides=(), **kw):
    run_opendde = _wrapper(monkeypatch, **kw)
    record = tmp_path / "row.run.json"
    argv = ["run_opendde.py", "--run-record", str(record)]
    for o in overrides:
        argv += ["--set", o]
    monkeypatch.setattr(sys, "argv", argv + ["--", "pred", *cli])
    code = run_opendde.main()
    return code, json.loads(record.read_text())


def _cli(src, out, *flags, seeds="101,102"):
    return ["-i", src, "-o", str(out), *flags, "--seeds", seeds]


# Inputs the wrapper accepts


def test_unchanged_single_file_input(tmp_path, monkeypatch):
    code, rec = _run(
        monkeypatch, tmp_path, _cli(D1, tmp_path / "D1", "--use_msa", "false")
    )
    prov = rec["provenance"]
    source = _sha((ROOT / D1).read_bytes())
    assert code == 0 and prov["refused"] is None
    assert prov["source"]["sha256"] == source
    assert prov["expected_effective_sha256"] == source
    assert [r["sha256"] for r in prov["parser_reads"]] == [source]
    assert prov["parser_reads"][0]["matches_parse"] is True
    assert prov["referenced"] == {}
    assert prov["seeds"] == {"synthetic_protein_200": [101, 102]}


def test_d4_legacy_conversion_reads_the_frozen_effective_bytes(tmp_path, monkeypatch):
    code, rec = _run(
        monkeypatch, tmp_path, _cli(D4, tmp_path / "D4", "--use_msa", "true")
    )
    prov = rec["provenance"]
    frozen = EFFECTIVE[D4]
    assert code == 0
    assert prov["source"]["sha256"] == frozen["source_sha256"]
    assert prov["expected_effective_sha256"] == frozen["effective_sha256"]
    assert [r["sha256"] for r in prov["parser_reads"]] == [frozen["effective_sha256"]]
    assert prov["parser_reads"][0]["path"].endswith("d4_7wux-update-msa.json")
    digests = json.loads((INPUTS / "digests.json").read_text())
    assert prov["referenced"] and all(
        s["before"] == s["after"] == digests[Path(p).as_posix()]
        for p, s in prov["referenced"].items()
    )


@pytest.mark.parametrize("path", sorted(EFFECTIVE))
def test_mirror_matches_the_runner_conversion(path, tmp_path):
    """The frozen effective digests come from the mirror; the runner must
    write the same bytes (H2 is never converted by the runner here: its
    format is covered by the synthetic case below)."""
    from runner.msa_search import update_infer_json

    source = (ROOT / path).read_bytes()
    mirrored, _ = provenance.effective_input(source, use_msa=True)
    assert _sha(mirrored) == EFFECTIVE[path]["effective_sha256"]
    if "h2_" in path:
        return
    written, _ = update_infer_json(path, str(tmp_path), json_output_dir=str(tmp_path))
    assert Path(written).read_bytes() == mirrored


def test_synthetic_h2_format_conversion_matches(tmp_path, monkeypatch):
    """Protein dimer with a legacy MSA directory, two DNA duplexes and a
    ligand, like H2, without running the runner on H2 itself."""
    from runner.msa_search import update_infer_json

    msa = tmp_path / "msa" / "1"
    msa.mkdir(parents=True)
    (msa / "pairing.a3m").write_text(">q\nMK\n")
    (msa / "non_pairing.a3m").write_text(">q\nMK\n")
    job = {
        "name": "synthetic_h2",
        "modelSeeds": [],
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "MK",
                    "count": 2,
                    "id": ["A", "B"],
                    "msa": {
                        "precomputed_msa_dir": "./msa/1",
                        "pairing_db": "uniref100",
                    },
                }
            },
            {"dnaSequence": {"sequence": "ACGT", "count": 2, "id": ["C", "D"]}},
            {"dnaSequence": {"sequence": "TTGA", "count": 2, "id": ["E", "F"]}},
            {"ligand": {"ligand": "CCD_PCG", "count": 2, "id": ["G", "H"]}},
        ],
    }
    source = tmp_path / "h2_like.json"
    source.write_text(json.dumps([job], indent=1))
    monkeypatch.chdir(tmp_path)
    written, _ = update_infer_json(
        str(source), str(tmp_path / "out"), json_output_dir=str(tmp_path / "gen")
    )
    mirrored, jobs = provenance.effective_input(source.read_bytes(), use_msa=True)
    assert Path(written).read_bytes() == mirrored
    assert provenance.referenced_files(jobs) == [
        "./msa/1/non_pairing.a3m",
        "./msa/1/pairing.a3m",
    ]


def _template_setup(tmp_path):
    kalign = tmp_path / "kalign"
    kalign.write_text("#!/bin/sh\n")
    kalign.chmod(0o755)
    mmcif = tmp_path / "mmcif"
    mmcif.mkdir()
    (mmcif / "1abc.cif").write_text("data_1abc\n")
    out = tmp_path / "D5"
    overrides = [
        f"data.template.prot_template_mmcif_dir={mmcif}",
        "data.template.fetch_remote=false",
        f"data.template.prot_template_cache_dir={out}/template_cache",
    ]
    flags = [
        "--use_msa",
        "true",
        "--use_template",
        "true",
        "--kalign_binary_path",
        str(kalign),
    ]
    return kalign, mmcif, out, overrides, flags


def test_d5_template_inputs_are_recorded(tmp_path, monkeypatch):
    kalign, mmcif, out, overrides, flags = _template_setup(tmp_path)
    code, rec = _run(monkeypatch, tmp_path, _cli(D5, out, *flags), overrides)
    prov = rec["provenance"]
    assert code == 0 and prov["refused"] is None
    assert "./examples/cofold/template/9fm7/hmmsearch.a3m" in prov["referenced"]
    tree = {"1abc.cif": _sha(b"data_1abc\n")}
    assert prov["template_mmcif"]["before"] == prov["template_mmcif"]["after"] == tree
    assert prov["template_cache"] == {
        "path": f"{out}/template_cache",
        "inside_output": True,
        "absent_before": True,
        "absent_at_parse": True,
    }
    assert prov["kalign"]["executable"] is True
    assert prov["kalign"]["requested_resolved"] == os.path.realpath(kalign)


# Refusals: the run is not started


def _refused(monkeypatch, tmp_path, cli, overrides=(), fragment=""):
    started = []
    code, rec = _run(
        monkeypatch, tmp_path, cli, overrides, pred=lambda a: started.append(a)
    )
    assert code == 2 and not started
    assert fragment in rec["provenance"]["refused"]
    assert rec["exception"].startswith("refused:")
    return rec


def test_directory_input_is_refused(tmp_path, monkeypatch):
    _refused(
        monkeypatch,
        tmp_path,
        _cli("examples/cofold", tmp_path / "o"),
        fragment="single input JSON",
    )


@pytest.mark.parametrize(
    "entity,flags,fragment",
    [
        (
            {"proteinChain": {"sequence": "MK", "count": 1}},
            ["--use_msa", "true"],
            "protein MSA search",
        ),
        (
            {
                "proteinChain": {
                    "sequence": "MK",
                    "count": 1,
                    "pairedMsaPath": "PAIRED",
                    "unpairedMsaPath": "PAIRED",
                }
            },
            ["--use_msa", "true", "--use_template", "true"],
            "template search",
        ),
        (
            {"rnaSequence": {"sequence": "GGCA", "count": 1}},
            ["--use_msa", "true", "--use_rna_msa", "true"],
            "RNA MSA search",
        ),
    ],
)
def test_searches_are_refused(tmp_path, monkeypatch, entity, flags, fragment):
    paired = tmp_path / "pairing.a3m"
    paired.write_text(">q\nMK\n")
    text = json.dumps([{"name": "j", "sequences": [entity]}]).replace(
        "PAIRED", str(paired)
    )
    source = tmp_path / "in.json"
    source.write_text(text)
    _refused(
        monkeypatch,
        tmp_path,
        _cli(str(source), tmp_path / "o", *flags),
        fragment=fragment,
    )


def test_legacy_msa_row_from_another_directory_is_refused(tmp_path, monkeypatch):
    """From elsewhere the converted JSON would silently lose its MSA paths;
    with no MSA left, pred would search, so the run never starts."""
    monkeypatch.chdir(tmp_path)
    _refused(
        monkeypatch,
        tmp_path,
        _cli(str(ROOT / D4), tmp_path / "o", "--use_msa", "true"),
        fragment="protein MSA search",
    )


@pytest.mark.parametrize("state", ["missing", "not executable"])
def test_kalign_that_would_fall_back_to_path_is_refused(tmp_path, monkeypatch, state):
    kalign, _, out, overrides, flags = _template_setup(tmp_path)
    if state == "missing":
        kalign.unlink()
    else:
        kalign.chmod(0o644)
    _refused(
        monkeypatch,
        tmp_path,
        _cli(D5, out, *flags),
        overrides,
        fragment="kalign_binary_path",
    )


def test_prepopulated_template_cache_is_refused_before_launch(tmp_path, monkeypatch):
    _, _, out, overrides, flags = _template_setup(tmp_path)
    (out / "template_cache").mkdir(parents=True)
    _refused(
        monkeypatch,
        tmp_path,
        _cli(D5, out, *flags),
        overrides,
        fragment="template cache",
    )


def test_template_cache_filled_before_the_parser_read_fails_the_run(
    tmp_path, monkeypatch
):
    _, _, out, overrides, flags = _template_setup(tmp_path)
    from runner import inference

    def pred(args):
        (out / "template_cache").mkdir(parents=True)
        inference._load_inference_jobs_synchronized(D5)

    with pytest.raises(RuntimeError, match="prepopulated"):
        _run(monkeypatch, tmp_path, _cli(D5, out, *flags), overrides, pred=pred)
    rec = json.loads((tmp_path / "row.run.json").read_text())
    assert rec["provenance"]["template_cache"]["absent_at_parse"] is False


# What the record shows for accept.py to refuse


def test_json_changed_before_the_parser_read_is_recorded(tmp_path, monkeypatch):
    source = tmp_path / "in.json"
    source.write_text((ROOT / D1).read_text())
    from runner import inference

    def pred(args):
        data = json.loads(source.read_text())
        data[0]["modelSeeds"] = [7]
        source.write_text(json.dumps(data))
        inference._load_inference_jobs_synchronized(str(source))

    code, rec = _run(
        monkeypatch,
        tmp_path,
        _cli(str(source), tmp_path / "o", "--use_msa", "false"),
        pred=pred,
    )
    prov = rec["provenance"]
    assert code == 0
    assert prov["parser_reads"][0]["sha256"] != prov["source"]["sha256"]


def test_referenced_file_changed_during_the_run_is_recorded(tmp_path, monkeypatch):
    msa = tmp_path / "pairing.a3m"
    msa.write_text(">q\nMK\n")
    source = tmp_path / "in.json"
    source.write_text(
        json.dumps(
            [
                {
                    "name": "j",
                    "sequences": [
                        {
                            "proteinChain": {
                                "sequence": "MK",
                                "count": 1,
                                "pairedMsaPath": str(msa),
                            }
                        }
                    ],
                }
            ]
        )
    )

    def pred(args):
        msa.write_text(">q\nMR\n")

    code, rec = _run(
        monkeypatch,
        tmp_path,
        _cli(str(source), tmp_path / "o", "--use_msa", "true"),
        pred=pred,
    )
    state = rec["provenance"]["referenced"][str(msa)]
    assert state["before"] == _sha(b">q\nMK\n") and state["after"] == _sha(b">q\nMR\n")


def test_actual_seeds_come_from_the_output_layout(tmp_path, monkeypatch):
    out = tmp_path / "o"

    def pred(args):
        (out / "synthetic_protein_200" / "seed_5123").mkdir(parents=True)

    _, rec = _run(monkeypatch, tmp_path, _cli(D1, out, "--use_msa", "false"), pred=pred)
    assert rec["provenance"]["seeds"] == {"synthetic_protein_200": [5123]}


def _configs(tmp_path, content=b"weights"):
    from opendde.config.inference import build_inference_config
    from opendde.config.model_registry import DEFAULT_MODEL_NAME

    configs = build_inference_config(
        model_name=DEFAULT_MODEL_NAME, fill_required_with_null=True
    )
    files = {}
    for name in ("ckpt.pt", "components.cif", "components.pkl"):
        files[name] = tmp_path / name
        files[name].write_bytes(content if name == "ckpt.pt" else name.encode())
    configs.load_checkpoint_path = str(files["ckpt.pt"])
    configs.data.ccd_components_file = str(files["components.cif"])
    configs.data.ccd_components_rdkit_mol_file = str(files["components.pkl"])
    configs.use_template = False
    return configs, files


def test_runtime_assets_are_hashed_from_the_effective_configuration(
    tmp_path, monkeypatch
):
    configs, files = _configs(tmp_path)
    _, rec = _run(
        monkeypatch,
        tmp_path,
        _cli(D1, tmp_path / "o", "--use_msa", "false"),
        configs=configs,
    )
    got = rec["provenance"]["assets"]
    assert set(got) == {
        "checkpoint",
        "ccd_components_file",
        "ccd_components_rdkit_mol_file",
    }
    assert got["checkpoint"]["resolved"] == os.path.realpath(files["ckpt.pt"])
    assert got["checkpoint"]["sha256"] == _sha(b"weights")


def test_checkpoint_changed_with_the_same_path_size_and_time_is_detected(
    tmp_path, monkeypatch
):
    configs, files = _configs(tmp_path)
    ckpt = files["ckpt.pt"]
    (tmp_path / "a").mkdir()
    _, first = _run(
        monkeypatch,
        tmp_path / "a",
        _cli(D1, tmp_path / "o1", "--use_msa", "false"),
        configs=configs,
    )
    stat = ckpt.stat()
    ckpt.write_bytes(b"weighTs")
    os.utime(ckpt, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert (
        ckpt.stat().st_size == stat.st_size
        and ckpt.stat().st_mtime_ns == stat.st_mtime_ns
    )
    (tmp_path / "b").mkdir()
    _, second = _run(
        monkeypatch,
        tmp_path / "b",
        _cli(D1, tmp_path / "o2", "--use_msa", "false"),
        configs=configs,
    )
    a = first["provenance"]["assets"]["checkpoint"]
    b = second["provenance"]["assets"]["checkpoint"]
    assert a["resolved"] == b["resolved"] and a["sha256"] != b["sha256"]


# The private asset copy


def _store(root: Path) -> None:
    for name, rel in assets.LAYOUT.items():
        if name == "kalign_binary_path":
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)


def _copy_setup(tmp_path):
    source, copy = tmp_path / "shared", tmp_path / "copy"
    _store(source)
    kalign = tmp_path / "kalign-src"
    kalign.write_text("#!/bin/sh\n")
    kalign.chmod(0o755)
    _store(copy)
    (copy / "bin").mkdir()
    (copy / "bin" / "kalign").write_text("#!/bin/sh\n")
    (copy / "bin" / "kalign").chmod(0o755)
    return source, copy, kalign


def test_independent_copy_verifies_and_freezes(tmp_path):
    source, copy, kalign = _copy_setup(tmp_path)
    assert assets.verify(copy, source, kalign)["ok"]
    reference = tmp_path / "runtime_assets.json"
    entry = assets.freeze(copy, "sm_120", reference)
    assert entry["assets"]["checkpoint"]["path"] == os.path.realpath(
        copy / "checkpoint/opendde.pt"
    )
    env = {
        "OPENDDE_ROOT_DIR": str(copy),
        "OPENDDE_QUAL_KALIGN": str(copy / "bin" / "kalign"),
    }
    assert assets.preflight("sm_120", reference, env)["ok"]
    (copy / "common" / "components.cif").write_text("changed")
    assert assets.preflight("sm_120", reference, env)["problems"] == {
        "ccd_components_file": "missing or changed"
    }
    assert not assets.preflight(
        "sm_120", reference, {**env, "OPENDDE_ROOT_DIR": str(source)}
    )["ok"]


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_copy_linked_to_the_shared_store_fails(tmp_path, link):
    source, copy, kalign = _copy_setup(tmp_path)
    target = copy / "checkpoint" / "opendde.pt"
    target.unlink()
    (target.symlink_to if link == "symlink" else target.hardlink_to)(
        source / "checkpoint" / "opendde.pt"
    )
    result = assets.verify(copy, source, kalign)
    assert not result["ok"] and list(result["problems"]) == ["checkpoint"]


# Signature inventory and the do-not-run audit


def test_inventory_families_and_frozen_file():
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    frozen = json.loads((INPUTS / "signature_inventory.json").read_text())
    assert inventory.build(manifest, frozen["token_counts"]) == frozen
    families = frozen["families"]
    h1 = json.loads((ROOT / "examples/cofold/input.json").read_text())[0]
    assert h1["sequences"][0]["proteinChain"]["sequence"] in families["held_out_only"]
    d4 = json.loads((ROOT / D4).read_text())[0]
    assert d4["sequences"][0]["proteinChain"]["sequence"] in families["shared"]
    assert len(families["restraint_protected"]) == 2
    assert sorted(frozen["token_counts"]) == sorted(f"H{i}" for i in range(1, 11))
    assert "examples/tfg/9lh2/9lh2_pocket.json" in frozen["files"]


def _record(path: Path, source: str, seeds, commit="none"):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = path.name.removesuffix(".run.json")
    path.write_text(
        json.dumps(
            {
                "cli_args": ["pred", "-i", source, "-o", str(path.parent / row)],
                "fingerprint": {"commit": commit},
            }
        )
    )
    for s in seeds:
        (path.parent / row / "job" / f"seed_{s}").mkdir(parents=True)


def test_audit_findings(tmp_path):
    store = tmp_path / "store"
    _record(store / "a" / "D1.run.json", D1, [101, 102])
    _record(store / "b" / "D1.run.json", D1, [250])
    _record(store / "c" / "H1.run.json", "examples/cofold/input.json", [101])
    _record(store / "d" / "D4.run.json", D4, [250])
    _record(store / "e" / "D4.run.json", D4, [101])
    _record(store / "f" / "H1.run.json", "examples/cofold/input.json", [301])
    (store / "bare" / "user_test" / "seed_7").mkdir(parents=True)
    frozen = json.loads((INPUTS / "signature_inventory.json").read_text())
    log = {"draws": [{"id": "x", "seeds": {"H1": [301, 302]}}]}
    report = run_audit.audit(
        {"stores": [{"path": str(store)}, {"path": str(tmp_path / "gone")}]},
        frozen,
        log,
    )
    found = {
        Path(r.get("record") or r["output"]).parent.name: r["finding"]
        for r in report["runs"]
    }
    assert found == {
        "a": "ok",
        "b": "deviation",
        "c": "stop",
        "d": "stop",
        "e": "ok",
        "f": "ok",
        "user_test": "stop",
    }
    assert not report["ok"] and report["gaps"] == [
        f"store missing: {tmp_path / 'gone'}"
    ]
    assert all(r["input_status"].startswith("unverified") for r in report["runs"])
