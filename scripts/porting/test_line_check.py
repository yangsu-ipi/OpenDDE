"""Tests for the acceptance-line checker (scripts/porting/line_check.py).

Each test builds a throwaway bare remote and a repository holding a copy of
the real ``scripts/qualification`` code (whose frozen ``_extends`` the checker
loads), a held-out log and a line ledger, then the acceptance lines of one
scenario: a normal merge, a merge resolved toward the porting branch,
concurrent lines, an unlisted ref, a missing ref, a moved closed ref and an
open entry."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import line_check  # noqa: E402

pytest.importorskip("rdkit")  # the copied accept.py imports the qualification stack


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
        + ["-c", "commit.gpgsign=false", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class Lines:
    """A remote, a porting branch and acceptance lines with their ledger."""

    def __init__(self, tmp: Path):
        self.remote = tmp / "remote.git"
        _git(tmp, "init", "-q", "--bare", str(self.remote))
        self.repo = tmp / "repo"
        qual = self.repo / "scripts" / "qualification"
        (qual / "inputs").mkdir(parents=True)
        for f in (ROOT / "scripts" / "qualification").glob("*.py"):
            shutil.copy(f, qual / f.name)
        (self.repo / "porting_docs").mkdir()
        _git(tmp, "init", "-q", "-b", "porting", str(self.repo))
        _git(self.repo, "remote", "add", "origin", str(self.remote))
        self.ledger: dict[str, str] = {}
        self.write_log({"receipts": [], "draws": []})
        self.write_ledger()
        self.commit("base")

    def write_log(self, log):
        path = self.repo / line_check.LOG
        path.write_text(json.dumps(log, indent=1) + "\n")

    def read_log(self):
        return json.loads((self.repo / line_check.LOG).read_text())

    def write_ledger(self):
        text = "# Qualification line ledger\n"
        for branch, row in self.ledger.items():
            text += f"\n## {branch}\n\n| Field | Value |\n| --- | --- |\n{row}"
        (self.repo / line_check.LEDGER).write_text(text)

    def commit(self, message) -> str:
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "--allow-empty", "-m", message)
        return _git(self.repo, "rev-parse", "HEAD")

    def line(self, name, start="porting") -> str:
        """An acceptance line from ``start``: a receipt commit and a draw
        commit, pushed; returns its tip. Leaves ``porting`` checked out."""
        branch = f"accept/{name}"
        _git(self.repo, "checkout", "-q", "-b", branch, start)
        log = self.read_log()
        log["receipts"].append({"id": f"r-{name}"})
        self.write_log(log)
        self.commit(f"receipt {name}")
        log["draws"].append({"id": f"d-{name}", "receipt": f"r-{name}"})
        self.write_log(log)
        tip = self.commit(f"draw {name}")
        _git(self.repo, "push", "-q", "origin", branch)
        _git(self.repo, "checkout", "-q", "porting")
        return tip

    def close(self, name, tip, status="rejected"):
        self.ledger[f"accept/{name}"] = (
            f"| Status | **{status}** |\n| Final branch tip | `{tip}` |\n"
        )
        self.write_ledger()
        return self.commit(f"close {name}")

    def merge(self, name, *strategy):
        _git(self.repo, "merge", "-q", "--no-edit", *strategy, f"accept/{name}")

    def check(self):
        report = line_check.check(self.repo, str(self.remote))
        return report, [c["check"] for c in report["checks"] if not c["ok"]]


@pytest.fixture
def lines(tmp_path):
    return Lines(tmp_path)


def test_closed_line_merged_normally_passes(lines):
    tip = lines.line("sm120-a")
    lines.merge("sm120-a")
    lines.close("sm120-a", tip)
    report, failed = lines.check()
    assert failed == [] and report["ok"]
    assert report["ledger"] == {"accept/sm120-a": {"status": "rejected", "tip": tip}}


def test_merge_resolved_toward_the_porting_branch_fails(lines):
    """`-s ours` keeps the tip as an ancestor but discards the line's log."""
    tip = lines.line("sm120-a")
    lines.merge("sm120-a", "-s", "ours")
    lines.close("sm120-a", tip)
    _, failed = lines.check()
    assert failed == ["candidate log extends the log at accept/sm120-a's tip"]


def test_concurrent_lines_fail(lines):
    """Two lines from the same base: the merged log holds both, but cannot
    extend each line's log (their receipts interleave)."""
    a = lines.line("sm120-a")
    b = lines.line("sm80-b")
    lines.merge("sm120-a")
    with pytest.raises(subprocess.CalledProcessError):
        lines.merge("sm80-b")  # both edited the held-out log
    log = json.loads(_git(lines.repo, "show", f"{a}:{line_check.LOG}"))
    other = json.loads(_git(lines.repo, "show", f"{b}:{line_check.LOG}"))
    log["receipts"] += other["receipts"]
    log["draws"] += other["draws"]
    lines.write_log(log)
    lines.commit("resolve")
    lines.close("sm120-a", a)
    lines.close("sm80-b", b)
    _, failed = lines.check()
    assert failed == ["candidate log extends the log at accept/sm80-b's tip"]


def test_unlisted_ref_fails(lines):
    tip = lines.line("sm120-a")
    lines.merge("sm120-a")
    lines.close("sm120-a", tip)
    lines.line("sm80-unlisted", start="accept/sm120-a")
    _, failed = lines.check()
    assert failed == ["every acceptance ref has a ledger entry"]


def test_missing_ref_fails(lines):
    tip = lines.line("sm120-a")
    lines.merge("sm120-a")
    lines.close("sm120-a", tip)
    _git(lines.repo, "push", "-q", "origin", "--delete", "accept/sm120-a")
    _, failed = lines.check()
    assert "every ledger entry's ref is on the remote" in failed


def test_moved_closed_ref_fails(lines):
    tip = lines.line("sm120-a")
    lines.merge("sm120-a")
    lines.close("sm120-a", tip)
    _git(lines.repo, "checkout", "-q", "accept/sm120-a")
    lines.commit("after closure")
    _git(lines.repo, "push", "-q", "origin", "accept/sm120-a")
    _git(lines.repo, "checkout", "-q", "porting")
    _, failed = lines.check()
    assert failed == ["accept/sm120-a is at its recorded tip"]


def test_open_entry_fails(lines):
    tip = lines.line("sm120-a")
    lines.merge("sm120-a")
    lines.ledger["accept/sm120-a"] = (
        f"| Status | open |\n| Final branch tip | `{tip}` |\n"
    )
    lines.write_ledger()
    lines.commit("open entry")
    _, failed = lines.check()
    assert failed == ["no ledger entry open"]


def test_the_real_ledger_parses():
    entries = line_check.parse_ledger(
        (ROOT / "porting_docs" / "qualification_line_ledger.md").read_text()
    )
    assert entries["accept/sm120-fast-25267ec"] == {
        "status": "rejected",
        "tip": "9def88b9c6d0ed8d1abecee200d3c312d4b9f401",
    }
