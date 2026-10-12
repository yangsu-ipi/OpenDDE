"""Tests for the contract-version gate (scripts/qualification/qual_gate.py).

Each test builds a bare "remote" with numbered contract tags, a gate checkout
of the approved tag and candidate clones, so the gate is exercised on real Git
history rather than on a stale-tag check inside ``accept.py``.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "qualification"))
import qual_gate  # noqa: E402

FAMILY = "qual-test-contract-"
QUAL = Path("scripts/qualification")
# A stand-in verifier: writes a verdict with the evidence's exit status, a
# receipt naming the gate report when asked, and dirties its checkout when the
# evidence says "touch".
STUB = """import json
import sys
from pathlib import Path

VERSION = {n}
args = sys.argv[1:]
evidence = json.loads(Path(args[0]).read_text())
Path(args[args.index("--out") + 1]).write_text(json.dumps({{"exit": evidence["exit"]}}))
if "--receipt" in args:
    i = args.index("--receipt")
    gate = args[args.index("--gate-report") + 1]
    Path(args[i + 2]).write_text(json.dumps({{"id": args[i + 1], "gate": gate}}))
if evidence.get("touch"):
    Path("stray.txt").write_text("x")
sys.exit(evidence["exit"])
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=t@t"]
        + ["-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_contract(repo: Path, n: int, receipts=()) -> None:
    inputs = repo / QUAL / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    (repo / QUAL / "accept.py").write_text(STUB.format(n=n))
    acceptance = {"gpus": {"sm_120": {"frozen": {"contract_tag": f"{FAMILY}{n}"}}}}
    (inputs / "acceptance.json").write_text(json.dumps(acceptance))
    log = {"receipts": list(receipts), "draws": []}
    (inputs / "held_out_log.json").write_text(json.dumps(log))


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--no-verify", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


class Remote:
    """A bare remote holding contract tags 1 and 2, and clones of it."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.bare = tmp / "remote.git"
        _git(tmp, "init", "-q", "--bare", str(self.bare))
        work = tmp / "work"
        _git(tmp, "clone", "-q", str(self.bare), str(work))
        (work / "model.py").write_text("stock = True\n")
        for n in (1, 2):
            _write_contract(work, n)
            _commit(work, f"contract {n}")
            _git(work, "tag", "-a", f"{FAMILY}{n}", "-m", f"contract {n}")
        _git(work, "push", "-q", "origin", "HEAD:refs/heads/main", "--tags")
        self.work = work

    def clone(self, name: str, rev: str) -> Path:
        path = self.tmp / name
        _git(self.tmp, "clone", "-q", str(self.bare), str(path))
        _git(path, "checkout", "-q", "-B", "candidate", rev)
        return path

    def candidate(self, name: str, base: int, change=None) -> Path:
        """A candidate branched from contract ``base`` with a model change."""
        path = self.clone(name, f"{FAMILY}{base}")
        (path / "model.py").write_text("stock = False\n")
        if change:
            change(path)
        _commit(path, "candidate")
        return path

    def tag(self, n: int) -> None:
        """Push contract tag ``n`` from the working clone."""
        _write_contract(self.work, n)
        _commit(self.work, f"contract {n}")
        _git(self.work, "tag", "-a", f"{FAMILY}{n}", "-m", f"contract {n}")
        _git(self.work, "push", "-q", "origin", "HEAD:refs/heads/main", "--tags")


@pytest.fixture
def remote(tmp_path):
    return Remote(tmp_path)


def _run(remote, candidate, gate_tag=2, receipt=None):
    gate_repo = remote.tmp / f"gate-{gate_tag}"
    if not gate_repo.exists():
        remote.clone(gate_repo.name, f"{FAMILY}{gate_tag}")
    return qual_gate.gate(candidate, gate_repo, "sm_120", "origin", receipt)


def _failed(report):
    return [c["check"] for c in report["checks"] if not c["ok"]]


def test_current_candidate_passes(remote):
    report = _run(remote, remote.candidate("cand", 2))
    assert report["ok"], _failed(report)
    assert report["approved_tag"] == f"{FAMILY}2"
    assert report["tag_commit"] == _git(
        remote.bare, "rev-parse", f"{FAMILY}2^{{commit}}"
    )
    assert report["gate_commit"] == report["tag_commit"]


def test_candidate_from_an_older_contract_is_refused(remote):
    """Its own verifier lacks the newer checks; the gate does not run it."""
    report = _run(remote, remote.candidate("cand", 1))
    assert set(_failed(report)) == {
        "candidate names the approved tag",
        "approved tag is an ancestor of the candidate",
        "candidate qualification files match the approved tag",
    }


def test_changed_qualification_file_is_refused(remote):
    def edit(path):
        (path / QUAL / "accept.py").write_text(STUB.format(n=2) + "# edited\n")

    report = _run(remote, remote.candidate("cand", 2, edit))
    assert _failed(report) == ["candidate qualification files match the approved tag"]


def test_tag_not_an_ancestor_is_refused(remote):
    """The same files and tag name on unrelated history."""
    cand = remote.clone("cand", f"{FAMILY}2")
    tree = _git(cand, "rev-parse", "HEAD^{tree}")
    orphan = _git(cand, "commit-tree", tree, "-m", "elsewhere")
    _git(cand, "checkout", "-q", "-B", "candidate", orphan)
    report = _run(remote, cand)
    assert _failed(report) == ["approved tag is an ancestor of the candidate"]


def test_approved_tag_comes_from_the_remote(remote):
    """A newer tag that exists only locally neither approves nor blocks."""
    cand = remote.candidate("cand", 2)
    _git(cand, "tag", "-a", f"{FAMILY}3", "-m", "local only")
    report = _run(remote, cand)
    assert report["ok"] and report["approved_tag"] == f"{FAMILY}2"


def test_locally_moved_tag_is_refused(remote):
    cand = remote.candidate("cand", 2)
    _git(cand, "tag", "-f", "-a", f"{FAMILY}2", "-m", "moved", f"{FAMILY}1")
    report = _run(remote, cand)
    assert "approved tag in the candidate checkout matches the remote" in _failed(
        report
    )


def test_dirty_candidate_or_gate_is_refused(remote):
    cand = remote.candidate("cand", 2)
    (cand / QUAL / "stray.py").write_text("x = 1\n")
    assert _failed(_run(remote, cand)) == ["candidate checkout clean"]
    report = _run(remote, remote.candidate("cand2", 2), gate_tag=1)
    assert _failed(report) == ["gate checkout clean and at the approved tag"]


def _with_receipt(tag):
    def change(path):
        receipt = {"id": "rehearsal-1"}
        if tag is not None:
            receipt["contract_tag"] = tag
        log = path / QUAL / "inputs" / "held_out_log.json"
        log.write_text(json.dumps({"receipts": [receipt], "draws": []}))

    return change


def test_receipt_records_the_approved_tag(remote):
    cand = remote.candidate("cand", 2, _with_receipt(f"{FAMILY}2"))
    assert _run(remote, cand, receipt="rehearsal-1")["ok"]


@pytest.mark.parametrize("tag", [None, f"{FAMILY}1"])
def test_receipt_without_or_with_another_tag_is_refused(remote, tag):
    cand = remote.candidate("cand", 2, _with_receipt(tag))
    report = _run(remote, cand, receipt="rehearsal-1")
    assert not report["ok"]
    if tag is None:
        assert _failed(report) == ["receipt logged with a contract tag"]
    else:
        assert "no contract tag created while the draw is open" in _failed(report)


def test_tag_created_while_a_draw_is_open_is_refused(remote):
    cand = remote.candidate("cand", 2, _with_receipt(f"{FAMILY}2"))
    remote.tag(3)
    report = _run(remote, cand, receipt="rehearsal-1")
    assert _failed(report) == ["no contract tag created while the draw is open"]
    report = _run(remote, cand)
    assert report["approved_tag"] == f"{FAMILY}3" and not report["ok"]


def test_main_exit_status(remote, monkeypatch, capsys):
    cand = remote.candidate("cand", 2)
    gate_repo = remote.clone("gate", f"{FAMILY}2")
    monkeypatch.setattr(qual_gate, "GATE_REPO", gate_repo)
    out = remote.tmp / "gate.json"
    monkeypatch.setattr(sys, "argv", ["qual_gate.py", str(cand), "--out", str(out)])
    assert qual_gate.main() == 0
    assert json.loads(out.read_text())["ok"]
    monkeypatch.setattr(sys, "argv", ["qual_gate.py", str(remote.candidate("old", 1))])
    assert qual_gate.main() == 1


# The approved tag comes from the gate checkout's remote


def test_stale_candidate_remote_cannot_hide_a_newer_tag(remote):
    """The candidate's origin lacks tag 3; the gate's remote has it."""
    stale = remote.tmp / "stale.git"
    _git(remote.tmp, "clone", "-q", "--bare", str(remote.bare), str(stale))
    cand = remote.candidate("cand", 2)
    _git(cand, "remote", "set-url", "origin", str(stale))
    remote.tag(3)
    report = _run(remote, cand, gate_tag=2)
    assert report["approved_tag"] == f"{FAMILY}3"
    assert report["remote"] == str(remote.bare)
    assert {
        "gate checkout clean and at the approved tag",
        "candidate names the approved tag",
    } <= set(_failed(report))


def test_candidate_remotes_are_not_consulted(remote):
    cand = remote.candidate("cand", 2)
    _git(cand, "remote", "set-url", "origin", str(remote.tmp / "missing.git"))
    report = _run(remote, cand)
    assert report["ok"], _failed(report)


# The entrypoint: gate, verifier, post-verdict check, combined record


def _evidence(tmp, **fields):
    path = tmp / f"evidence-{len(list(tmp.glob('evidence-*')))}.json"
    path.write_text(json.dumps(fields))
    return path


def _entry(remote, cand, evidence, name, **kw):
    gate_repo = remote.tmp / "gate-2"
    if not gate_repo.exists():
        remote.clone(gate_repo.name, f"{FAMILY}2")
    record_dir = remote.tmp / name
    code, record = qual_gate.run_entrypoint(
        cand, evidence, record_dir, gate_repo, "sm_120", "origin", **kw
    )
    return code, record, record_dir


def test_entrypoint_does_not_run_the_verifier_after_a_failed_gate(remote):
    cand = remote.candidate("old", 1)
    code, record, out = _entry(remote, cand, _evidence(remote.tmp, exit=0), "rec")
    assert code == qual_gate.GATE_FAILED and not record["ok"]
    assert record["verdict"] is None and not (out / "verdict.json").exists()
    assert json.loads((out / "gate.json").read_text())["ok"] is False


def test_entrypoint_full_pass_binds_gate_and_verdict(remote):
    cand = remote.candidate("cand", 2)
    code, record, out = _entry(remote, cand, _evidence(remote.tmp, exit=0), "rec")
    assert code == 0 and record["ok"] and record["candidate_unchanged"]
    gate_text = (out / "gate.json").read_text()
    assert record["gate_sha256"] == qual_gate._sha256(gate_text)
    verdict_text = (out / "verdict.json").read_text()
    assert record["verdict"] == {
        "exit_code": 0,
        "sha256": qual_gate._sha256(verdict_text),
    }
    assert record["candidate_commit"] == _git(cand, "rev-parse", "HEAD")
    assert json.loads((out / "record.json").read_text()) == record


def test_entrypoint_rehearsal_writes_a_receipt_with_the_gate_report(remote):
    cand = remote.candidate("cand", 2)
    code, record, out = _entry(
        remote,
        cand,
        _evidence(remote.tmp, exit=4),
        "rec",
        write_receipt="rehearsal-1",
    )
    assert code == 4 and not record["ok"]
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt == {"id": "rehearsal-1", "gate": str((out / "gate.json").resolve())}


def test_entrypoint_refuses_a_candidate_changed_during_the_verdict(remote):
    cand = remote.candidate("cand", 2)
    evidence = _evidence(remote.tmp, exit=0, touch=True)
    code, record, _ = _entry(remote, cand, evidence, "rec")
    assert code == 1 and not record["ok"] and not record["candidate_unchanged"]
    assert record["verdict"]["exit_code"] == 0


def test_entrypoint_command_line(remote, monkeypatch):
    cand = remote.candidate("cand", 2)
    gate_repo = remote.clone("gate", f"{FAMILY}2")
    monkeypatch.setattr(qual_gate, "GATE_REPO", gate_repo)
    evidence = _evidence(remote.tmp, exit=0)
    out = remote.tmp / "rec"
    argv = ["qual_gate.py", str(cand), "--accept", str(evidence)]
    monkeypatch.setattr(sys, "argv", argv + ["--record-dir", str(out)])
    assert qual_gate.main() == 0 and (out / "record.json").exists()
    with pytest.raises(SystemExit):
        qual_gate.main()  # the record directory now exists
    monkeypatch.setattr(
        sys, "argv", ["qual_gate.py", str(cand), "--write-receipt", "r"]
    )
    with pytest.raises(SystemExit):
        qual_gate.main()  # --write-receipt needs --accept
