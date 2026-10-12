#!/usr/bin/env python
"""Independent contract-version gate and qualification entrypoint
(qualification_protocol.md, E5).

Usage:
    python scripts/qualification/qual_gate.py CANDIDATE_CHECKOUT
        [--gpu sm_120] [--remote origin] [--receipt ID] [--out GATE.json]
    python scripts/qualification/qual_gate.py CANDIDATE_CHECKOUT
        [--gpu sm_120] [--remote origin] [--receipt ID]
        --accept EVIDENCE.json --record-dir DIR [--write-receipt ID]

Run from a clean checkout of the approved contract tag. The gate compares
files rather than running a second verifier: when the candidate's
``scripts/qualification`` is identical to the approved tag's apart from the
held-out log, the candidate's ``accept.py`` is the approved verifier. A
candidate branched from an older contract fails here even though its own
``accept.py`` lacks the newer checks.

The approved tag is the newest ``<family>N`` tag on this gate checkout's
remote (``--remote``; the family is that of the tag this checkout names). The
candidate checkout's remotes are never consulted: the tag is fetched into it
from the gate's remote URL. With ``--receipt ID`` the approved tag must also be
the tag that receipt records: no contract tag may be created while a draw is
open. Checks:
- the gate's own checkout is clean and at the approved tag's commit;
- the approved tag, fetched into the candidate checkout, resolves to the same
  commit as on the remote;
- the candidate checkout is clean, its ``acceptance.json`` names the approved
  tag for the GPU, the tag is an ancestor of its HEAD, and
  ``scripts/qualification`` differs from the tag only in the held-out log;
- with ``--receipt``: the receipt exists and records the approved tag.

Without ``--accept`` the gate runs alone, as the recorded preflight before
each batch of held-out runs; exit 0 passes, 1 fails. With ``--accept`` it is
the qualification entrypoint for a rehearsal or a final acceptance: it writes
``gate.json`` to a new ``--record-dir``, runs the candidate's ``accept.py`` on
the evidence only if the gate passed (``verdict.json``; with
``--write-receipt ID`` a passing rehearsal also writes ``receipt.json``,
recording this gate report), then checks that the candidate checkout is still
clean at the commit the gate approved, and writes ``record.json``. A full
qualification claim is that record with ``ok`` true and exit status 0, not
``accept.py``'s exit status alone. Exit status: ``accept.py``'s own when the
gate and the post-verdict check pass; 5 when the gate fails (the verifier is
not run); 1 when the candidate changed during the verdict.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

GATE_REPO = Path(__file__).resolve().parents[2]
QUAL = "scripts/qualification"
ACCEPTANCE = f"{QUAL}/inputs/acceptance.json"
LOG = f"{QUAL}/inputs/held_out_log.json"
GATE_FAILED = 5


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _commit(repo: Path, rev: str) -> str | None:
    out = _git(repo, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
    return out.strip() if out else None


def _show_json(repo: Path, rev: str, path: str) -> Any:
    try:
        return json.loads(_git(repo, "show", f"{rev}:{path}") or "")
    except json.JSONDecodeError:
        return None


def _contract_tag(acceptance, gpu):
    frozen = ((acceptance or {}).get("gpus", {}).get(gpu) or {}).get("frozen") or {}
    return frozen.get("contract_tag")


def remote_tags(repo: Path, remote: str, family: str) -> dict[int, tuple[str, str]]:
    """``{N: (tag, peeled commit)}`` for the family's tags on ``repo``'s
    remote."""
    out = _git(repo, "ls-remote", "--tags", remote) or ""
    peeled: dict[str, str] = {}
    for line in out.splitlines():
        sha, ref = line.split("\t")
        name = ref.removeprefix("refs/tags/")
        if name.endswith("^{}"):
            peeled[name[:-3]] = sha
        else:
            peeled.setdefault(name, sha)
    found = {}
    for name, sha in peeled.items():
        m = re.fullmatch(re.escape(family) + r"(\d+)", name)
        if m:
            found[int(m[1])] = (name, sha)
    return found


def _state(candidate: Path) -> dict[str, Any]:
    status = _git(candidate, "status", "--porcelain")
    return {
        "commit": _commit(candidate, "HEAD"),
        "dirty": None if status is None else status.splitlines(),
    }


def gate(
    candidate: Path,
    gate_repo: Path = GATE_REPO,
    gpu: str = "sm_120",
    remote: str = "origin",
    receipt_id: str | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name, ok, detail=None) -> bool:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    url = (_git(gate_repo, "remote", "get-url", remote) or "").strip() or None
    report: dict[str, Any] = {
        "candidate": str(candidate),
        "receipt": receipt_id,
        "remote": url,
    }

    def done():
        report["checks"] = checks
        report["ok"] = all(c["ok"] for c in checks)
        return report

    own = _contract_tag(_show_json(gate_repo, "HEAD", ACCEPTANCE), gpu)
    m = re.fullmatch(r"(.+-)(\d+)", own) if isinstance(own, str) else None
    if not add("gate checkout names a numbered contract tag", m is not None, own):
        return done()
    tags = remote_tags(gate_repo, remote, m[1]) if url else {}
    if not add("contract tags on the gate's remote", bool(tags), url):
        return done()
    newest = max(tags)
    log = _show_json(candidate, "HEAD", LOG) or {}
    if receipt_id is not None:
        found = [r for r in log.get("receipts", []) if r.get("id") == receipt_id]
        recorded = found[0].get("contract_tag") if len(found) == 1 else None
        if not add("receipt logged with a contract tag", recorded, receipt_id):
            return done()
        approved = recorded
        add(
            "no contract tag created while the draw is open",
            tags[newest][0] == approved,
            {"receipt": approved, "newest on remote": tags[newest][0]},
        )
    else:
        approved = tags[newest][0]
    remote_commit = dict(tags.values()).get(approved)
    report.update(approved_tag=approved, tag_commit=remote_commit)
    if not add("approved tag on the remote", remote_commit is not None, approved):
        return done()

    status = _git(gate_repo, "status", "--porcelain")
    report["gate_commit"] = _commit(gate_repo, "HEAD")
    add(
        "gate checkout clean and at the approved tag",
        status == "" and report["gate_commit"] == remote_commit,
        {"gate_commit": report["gate_commit"], "dirty": (status or "").splitlines()},
    )

    ref = f"refs/tags/{approved}"
    _git(candidate, "fetch", "--quiet", url, f"{ref}:{ref}")
    local = _commit(candidate, ref)
    add(
        "approved tag in the candidate checkout matches the remote",
        local == remote_commit,
        {"local": local, "remote": remote_commit},
    )
    state = _state(candidate)
    report["candidate_commit"] = state["commit"]
    add("candidate checkout clean", state["dirty"] == [], state["dirty"])
    named = _contract_tag(_show_json(candidate, "HEAD", ACCEPTANCE), gpu)
    add("candidate names the approved tag", named == approved, named)
    ancestor = (
        local is not None
        and state["commit"] is not None
        and _git(candidate, "merge-base", "--is-ancestor", local, state["commit"])
        is not None
    )
    add("approved tag is an ancestor of the candidate", ancestor)
    out = _git(candidate, "diff", "--name-only", remote_commit, "HEAD", "--", QUAL)
    changed = None if out is None else sorted(set(out.splitlines()) - {LOG})
    add(
        "candidate qualification files match the approved tag",
        changed == [],
        changed,
    )
    return done()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def run_entrypoint(
    candidate: Path,
    evidence: Path,
    record_dir: Path,
    gate_repo: Path = GATE_REPO,
    gpu: str = "sm_120",
    remote: str = "origin",
    receipt_id: str | None = None,
    write_receipt: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Gate, then the candidate's verifier, then the post-verdict check;
    return the exit status and the combined record."""
    record_dir.mkdir(parents=True)
    report = gate(candidate, gate_repo, gpu, remote, receipt_id)
    gate_text = json.dumps(report, indent=1) + "\n"
    gate_path = record_dir / "gate.json"
    gate_path.write_text(gate_text)
    record: dict[str, Any] = {
        "gate_ok": report["ok"],
        "gate_sha256": _sha256(gate_text),
        "approved_tag": report.get("approved_tag"),
        "gate_commit": report.get("gate_commit"),
        "candidate_commit": report.get("candidate_commit"),
        "evidence": str(evidence),
        "evidence_sha256": _sha256(evidence.read_text()),
    }

    def finish(code: int):
        record["exit_code"] = code
        record["ok"] = code == 0
        (record_dir / "record.json").write_text(json.dumps(record, indent=1) + "\n")
        return code, record

    if not report["ok"]:
        record["verdict"] = None
        return finish(GATE_FAILED)
    verdict_path = record_dir / "verdict.json"
    cmd = [
        sys.executable,
        str(candidate / QUAL / "accept.py"),
        str(evidence.resolve()),
        "--out",
        str(verdict_path.resolve()),
    ]
    if write_receipt:
        receipt_path = (record_dir / "receipt.json").resolve()
        cmd += ["--receipt", write_receipt, str(receipt_path)]
        cmd += ["--gate-report", str(gate_path.resolve())]
    result = subprocess.run(cmd, cwd=candidate, env=dict(os.environ))
    verdict_text = verdict_path.read_text() if verdict_path.is_file() else None
    record["verdict"] = {
        "exit_code": result.returncode,
        "sha256": None if verdict_text is None else _sha256(verdict_text),
    }
    after = _state(candidate)
    record["candidate_after"] = after
    unchanged = after["commit"] == report["candidate_commit"] and after["dirty"] == []
    record["candidate_unchanged"] = unchanged
    if not unchanged or verdict_text is None:
        return finish(1)
    return finish(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--gpu", default="sm_120")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--receipt", help="the open draw's receipt id")
    parser.add_argument("--out", type=Path, help="gate report (gate only)")
    parser.add_argument("--accept", type=Path, help="evidence for accept.py")
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--write-receipt", help="receipt id for a passing rehearsal")
    args = parser.parse_args()
    if args.accept is None:
        if args.record_dir or args.write_receipt:
            parser.error("--record-dir and --write-receipt need --accept")
        report = gate(args.candidate, GATE_REPO, args.gpu, args.remote, args.receipt)
        if args.out:
            args.out.write_text(json.dumps(report, indent=1) + "\n")
        for c in report["checks"]:
            print(("ok   " if c["ok"] else "FAIL ") + c["check"])
        print(
            f"gate {'passed' if report['ok'] else 'FAILED'}: approved "
            f"{report.get('approved_tag')}, candidate {report.get('candidate_commit')}"
        )
        return 0 if report["ok"] else 1
    if args.record_dir is None or args.out is not None:
        parser.error("--accept needs --record-dir (and no --out)")
    if args.record_dir.exists():
        parser.error(f"{args.record_dir} exists; use a new record directory")
    code, record = run_entrypoint(
        args.candidate,
        args.accept,
        args.record_dir,
        GATE_REPO,
        args.gpu,
        args.remote,
        args.receipt,
        args.write_receipt,
    )
    print(
        f"gate {'passed' if record['gate_ok'] else 'FAILED'}; verdict "
        f"{record['verdict']}; candidate unchanged "
        f"{record.get('candidate_unchanged')}; exit {code}"
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
