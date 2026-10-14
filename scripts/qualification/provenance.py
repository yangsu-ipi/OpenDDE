"""Source state and file digests shared by qualification records (protocol E5).

A producing run records the digests of what it wrote and a measurement
records the digests of what it read, both keyed by the path relative to the
row directory (``<job>/seed_<s>/predictions/<file>`` or
``<job>/seed_<s>/tier1.pt``), so ``accept.py`` can compare the two. Both
also record where the ``opendde`` and ``runner`` packages resolve, since an
editable install can load another checkout's code than the one whose commit
is recorded.

From contract-6, a recorded run also states its input provenance
(``run_opendde.py``): the helpers below compute, without importing the model
packages, the input JSON ``pred`` parses after its format-only legacy MSA
conversion, the MSA and template files that JSON references, whether
``pred`` would search for missing MSAs or templates, the seeds a run's
output layout holds, and the entity signature of an input job.
"""

import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import compare_outputs

REPO = Path(__file__).resolve().parents[2]

# Keys under an entity that name MSA or template files (protein chains use all
# three; RNA chains use ``unpairedMsaPath``).
REFERENCE_KEYS = ("pairedMsaPath", "unpairedMsaPath", "templatesPath")
POLYMERS = ("proteinChain", "dnaSequence", "rnaSequence")
SEED_DIR = re.compile(r"seed_(-?\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_state(path) -> dict[str, Any]:
    """Resolved path and SHA-256 of a file; ``sha256`` is None if it cannot
    be read as a regular file."""
    out: dict[str, Any] = {"resolved": os.path.realpath(path), "sha256": None}
    try:
        if Path(path).is_file():
            out["sha256"] = sha256_file(Path(path))
    except OSError as error:
        out["error"] = f"{type(error).__name__}: {error}"
    return out


def tree_digests(root) -> dict[str, str] | None:
    """SHA-256 of every regular file under ``root`` by relative path; None if
    ``root`` is not a directory."""
    root = Path(root)
    if not root.is_dir():
        return None
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def convert_legacy_msa(jobs: Any) -> tuple[Any, bool]:
    """A copy of ``jobs`` after ``pred``'s format-only legacy MSA conversion,
    and whether anything was converted.

    Mirrors ``runner.msa_search.convert_msa_to_new_format``: each protein
    chain's ``msa`` entry is removed and ``<precomputed_msa_dir>/pairing.a3m``
    and ``non_pairing.a3m`` become ``pairedMsaPath`` and ``unpairedMsaPath``
    when those files exist, resolved against the working directory. A run
    from another directory therefore silently loses its MSA paths, which the
    expected effective digest catches. ``tests/test_qualification_provenance.py``
    checks the mirror against the real conversion."""
    converted = False
    jobs = copy.deepcopy(jobs)
    for job in jobs if isinstance(jobs, list) else []:
        if not isinstance(job, dict) or not isinstance(job.get("sequences"), list):
            continue
        for sequence in job["sequences"]:
            chain = sequence.get("proteinChain") if isinstance(sequence, dict) else None
            if not isinstance(chain, dict) or "msa" not in chain:
                continue
            info = chain.pop("msa")
            converted = True
            directory = info.get("precomputed_msa_dir")
            if directory:
                paired = f"{directory}/pairing.a3m"
                unpaired = f"{directory}/non_pairing.a3m"
                if os.path.exists(paired):
                    chain["pairedMsaPath"] = paired
                if os.path.exists(unpaired):
                    chain["unpairedMsaPath"] = unpaired
    return jobs, converted


def effective_input(source: bytes, use_msa: bool) -> tuple[bytes, Any]:
    """The bytes and jobs ``pred`` parses for a single input file when it
    performs no search: with ``use_msa`` a legacy ``msa`` entry is converted
    and the result written with ``json.dump(..., indent=4)``
    (``runner.msa_search.update_infer_json``); otherwise, or if nothing needs
    converting, the source itself."""
    jobs = json.loads(source)
    if not use_msa:
        return source, jobs
    converted, changed = convert_legacy_msa(jobs)
    if not changed:
        return source, jobs
    return json.dumps(converted, indent=4).encode(), converted


def _entities(jobs):
    for job in jobs if isinstance(jobs, list) else []:
        for sequence in (job or {}).get("sequences", []) or []:
            if isinstance(sequence, dict):
                for kind, entity in sequence.items():
                    if isinstance(entity, dict):
                        yield job, kind, entity


def referenced_files(jobs) -> list[str]:
    """The MSA, RNA-MSA and template paths a job list names, as written."""
    found = {
        entity[key]
        for _, _, entity in _entities(jobs)
        for key in REFERENCE_KEYS
        if isinstance(entity.get(key), str)
    }
    return sorted(found)


def search_reasons(jobs, use_msa: bool, use_template: bool, use_rna_msa: bool):
    """Why ``pred`` would search for an MSA, template or RNA MSA on these
    (effective) jobs; empty if it would not. Mirrors ``need_msa_search``,
    ``update_template_info`` and ``update_rna_msa_info``."""
    reasons = []
    for job, kind, entity in _entities(jobs):
        name = job.get("name")
        if kind == "proteinChain" and use_msa:
            paths = [entity.get("pairedMsaPath"), entity.get("unpairedMsaPath")]
            if all(p is None for p in paths) or any(
                p is not None and not os.path.exists(p) for p in paths
            ):
                reasons.append(f"{name}: protein MSA search")
        if kind == "proteinChain" and use_template:
            path = entity.get("templatesPath")
            if path is None or not os.path.exists(path):
                reasons.append(f"{name}: template search")
        if kind == "rnaSequence" and use_rna_msa and entity.get("sequence", ""):
            path = entity.get("unpairedMsaPath")
            if path is None or not os.path.exists(path):
                reasons.append(f"{name}: RNA MSA search")
    return reasons


def job_seeds(out_dir: Path, jobs) -> dict[str, list[int]]:
    """The seeds each job's output layout holds (``<job>/seed_<s>``)."""
    out = {}
    for name in jobs:
        job_dir = Path(out_dir) / name
        seeds = []
        if job_dir.is_dir():
            for child in job_dir.iterdir():
                match = SEED_DIR.fullmatch(child.name)
                if match and child.is_dir():
                    seeds.append(int(match[1]))
        out[name] = sorted(seeds)
    return out


def entity_signature(job: dict) -> dict[str, Any]:
    """What identifies an input job's complex for the do-not-run audit:
    polymer type, sequence and count, ligands and ions with counts, and
    whether it carries a guidance restraint (``constraint``)."""
    polymers, others = [], []
    for _, kind, entity in _entities([job]):
        count = entity.get("count", 1)
        if kind in POLYMERS:
            polymers.append([kind, entity.get("sequence", ""), count])
        else:
            label = entity.get("ligand") or entity.get("ion") or kind
            others.append([kind, str(label), count])
    return {
        "polymers": sorted(polymers),
        "other": sorted(others),
        "restraint": bool(job.get("constraint")),
    }


PACKAGES = ("opendde", "runner")


def import_paths() -> dict[str, str | None]:
    """The directory each of ``PACKAGES`` resolves to in this interpreter
    (``sys.path`` first, then any editable-install finder); ``None`` if not
    found. Set ``PYTHONPATH`` to the checkout to load its own code."""
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        origin = spec.origin if spec else None
        out[name] = str(Path(origin).resolve().parent) if origin else None
    return out


def git_state() -> dict:
    """Commit and uncommitted paths of the checkout these scripts run from,
    and where the model packages resolve."""

    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(REPO), *args],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty_paths": _git("status", "--porcelain").splitlines(),
        "imports": import_paths(),
    }


def prediction_digests(row_dir: Path) -> dict[str, str]:
    """Every prediction artifact under a run's row directory."""
    return {
        rel: sha256_file(path)
        for rel, path in compare_outputs.artifacts(row_dir).items()
    }


def seed_prediction_digests(row_dir: Path, job: str, seeds) -> dict[str, str]:
    """Prediction artifacts of one job's seeds, following linked seed
    directories (H3's extra Zn seeds are linked into the held-out runs)."""
    out = {}
    for seed in seeds:
        pred = row_dir / job / f"seed_{seed}" / "predictions"
        for pattern in compare_outputs.ARTIFACT_PATTERNS:
            for path in sorted(pred.glob(pattern)):
                out[f"{job}/seed_{seed}/predictions/{path.name}"] = sha256_file(path)
    return out


def capture_digests(capture_dir: Path) -> dict[str, str]:
    """Every tier 1 capture (``<job>/seed_<s>/tier1.pt``) under a row's capture
    directory."""
    return {
        path.relative_to(capture_dir).as_posix(): sha256_file(path)
        for path in sorted(capture_dir.glob("*/seed_*/tier1.pt"))
    }
