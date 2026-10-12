#!/usr/bin/env python
"""Independent contract-version gate (qualification_protocol.md, E5).

Usage:
    python scripts/qualification/qual_gate.py CANDIDATE_CHECKOUT
        [--gpu sm_120] [--remote origin] [--receipt ID] [--out GATE.json]

Run from a clean checkout of the approved contract tag, before ``accept.py``
runs in the candidate checkout: at rehearsal, before each batch of held-out
runs and at final acceptance. The gate compares files rather than running a
second verifier: when the candidate's ``scripts/qualification`` is identical
to the approved tag's apart from the held-out log, the candidate's
``accept.py`` is the approved verifier. A candidate branched from an older
contract fails here even though its own ``accept.py`` lacks the newer checks.

The approved tag is the newest ``<family>N`` tag on the remote (the family is
that of the tag this gate's own checkout names), never one chosen by the
candidate. With ``--receipt ID`` it must also be the tag the candidate's
held-out log records in that receipt: no contract tag may be created while a
draw is open. Checks:
- the gate's own checkout is clean and at the approved tag's commit;
- the approved tag, fetched into the candidate checkout, resolves to the same
  commit as on the remote;
- the candidate checkout is clean, its ``acceptance.json`` names the approved
  tag for the GPU, the tag is an ancestor of its HEAD, and
  ``scripts/qualification`` differs from the tag only in the held-out log;
- with ``--receipt``: the receipt exists and records the approved tag.

Exit status: 0 the gate passes; 1 it fails. The report records the approved
tag, its commit, the gate's commit and the candidate's commit.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

GATE_REPO = Path(__file__).resolve().parents[2]
QUAL = "scripts/qualification"
ACCEPTANCE = f"{QUAL}/inputs/acceptance.json"
LOG = f"{QUAL}/inputs/held_out_log.json"


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
    """``{N: (tag, peeled commit)}`` for the family's tags on the remote."""
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

    report: dict[str, Any] = {"candidate": str(candidate), "receipt": receipt_id}

    def done():
        report["checks"] = checks
        report["ok"] = all(c["ok"] for c in checks)
        return report

    own = _contract_tag(_show_json(gate_repo, "HEAD", ACCEPTANCE), gpu)
    m = re.fullmatch(r"(.+-)(\d+)", own) if isinstance(own, str) else None
    if not add("gate checkout names a numbered contract tag", m is not None, own):
        return done()
    family = m[1]
    tags = remote_tags(candidate, remote, family)
    if not add("contract tags on the remote", bool(tags), remote):
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
    _git(candidate, "fetch", "--quiet", remote, f"{ref}:{ref}")
    local = _commit(candidate, ref)
    add(
        "approved tag in the candidate checkout matches the remote",
        local == remote_commit,
        {"local": local, "remote": remote_commit},
    )
    head = _commit(candidate, "HEAD")
    report["candidate_commit"] = head
    status = _git(candidate, "status", "--porcelain")
    add("candidate checkout clean", status == "", (status or "").splitlines())
    named = _contract_tag(_show_json(candidate, "HEAD", ACCEPTANCE), gpu)
    add("candidate names the approved tag", named == approved, named)
    ancestor = (
        local is not None
        and head is not None
        and _git(candidate, "merge-base", "--is-ancestor", local, head) is not None
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--gpu", default="sm_120")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--receipt")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
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


if __name__ == "__main__":
    sys.exit(main())
