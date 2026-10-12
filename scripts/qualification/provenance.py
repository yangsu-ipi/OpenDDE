"""Source state and file digests shared by qualification records (protocol E5).

A producing run records the digests of what it wrote and a measurement
records the digests of what it read, both keyed by the path relative to the
row directory (``<job>/seed_<s>/predictions/<file>`` or
``<job>/seed_<s>/tier1.pt``), so ``accept.py`` can compare the two.
"""

import hashlib
import subprocess
from pathlib import Path

import compare_outputs

REPO = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict:
    """Commit and uncommitted paths of the checkout these scripts run from."""

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
