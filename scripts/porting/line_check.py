#!/usr/bin/env python
"""Read-only acceptance-line checker, run before a later candidate's rehearsal
(qualification_protocol.md, E5: held-out log continuity across candidates).

Usage:
    python scripts/porting/line_check.py CANDIDATE_CHECKOUT --remote URL
        [--out REPORT.json]

CANDIDATE_CHECKOUT is a clean checkout of the candidate's code commit; URL is
the approved remote. The checker reads the line ledger
(``porting_docs/qualification_line_ledger.md``) and the held-out log at the
checkout's HEAD, and the visible ``accept/sm120-*`` and ``accept/sm80-*``
branches on the remote, and requires:

- one ledger entry per visible ref and a visible ref per entry (an unlisted
  ref or a missing ref fails);
- every entry closed (``accepted``, ``rejected`` or ``abandoned``), with its
  ref at the entry's recorded final tip (a moved ref fails);
- every recorded tip an ancestor of the candidate;
- the candidate's held-out log extending the log at every tip under
  ``_extends`` from the candidate's own (frozen) ``accept.py``: a line merged
  with its log discarded (for example ``git merge -s ours``) or two
  concurrent lines fail here although their tips are ancestors.

It fetches each tip's objects into the checkout (``git fetch URL <tip>``) and
changes nothing on the remote or in the working tree. Exit status 0 when every
check passes, 1 otherwise.
"""

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

LEDGER = "porting_docs/qualification_line_ledger.md"
LOG = "scripts/qualification/inputs/held_out_log.json"
QUAL = "scripts/qualification"
REF = re.compile(r"accept/sm(120|80)-[A-Za-z0-9._-]+")
CLOSED = ("accepted", "rejected", "abandoned")


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_ledger(text: str) -> dict[str, dict[str, Any]]:
    """``{branch: {"status": word, "tip": sha or None}}`` from the ledger's
    ``## accept/...`` sections and their Status and Final branch tip rows."""
    entries = {}
    parts = re.split(r"^## (accept/\S+)[ \t]*$", text, flags=re.M)
    for branch, body in zip(parts[1::2], parts[2::2]):
        status = re.search(r"^\| Status \| (.*?) \|\s*$", body, re.M)
        cell = status[1].lower() if status else ""
        word = next((w for w in CLOSED if w in cell), "open")
        tip = re.search(r"^\| Final branch tip \| (.*?) \|\s*$", body, re.M)
        sha = re.search(r"\b[0-9a-f]{40}\b", tip[1]) if tip else None
        entries[branch] = {"status": word, "tip": sha[0] if sha else None}
    return entries


def remote_refs(repo: Path, remote: str) -> dict[str, str] | None:
    out = _git(repo, "ls-remote", "--heads", remote)
    if out is None:
        return None
    refs = {}
    for line in out.splitlines():
        sha, name = line.split("\t")
        name = name.removeprefix("refs/heads/")
        if REF.fullmatch(name):
            refs[name] = sha
    return refs


def _frozen_extends(candidate: Path):
    """``_extends`` from the candidate's own scripts/qualification/accept.py."""
    qual = str(candidate / QUAL)
    sys.path.insert(0, qual)
    try:
        spec = importlib.util.spec_from_file_location(
            "candidate_accept", candidate / QUAL / "accept.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(qual)
    return module._extends


def _log_at(repo: Path, rev: str) -> Any:
    try:
        return json.loads(_git(repo, "show", f"{rev}:{LOG}") or "")
    except json.JSONDecodeError:
        return None


def check(candidate: Path, remote: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name, ok, detail=None) -> bool:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    head = (_git(candidate, "rev-parse", "HEAD") or "").strip() or None
    status = _git(candidate, "status", "--porcelain")
    report: dict[str, Any] = {
        "candidate": str(candidate),
        "head": head,
        "remote": remote,
    }

    def done():
        report["checks"] = checks
        report["ok"] = all(c["ok"] for c in checks)
        return report

    add("candidate checkout clean", status == "", (status or "").splitlines())
    ledger_text = _git(candidate, "show", f"HEAD:{LEDGER}")
    if not add("ledger present at the candidate", ledger_text is not None, LEDGER):
        return done()
    ledger = parse_ledger(ledger_text)
    refs = remote_refs(candidate, remote)
    if not add("acceptance refs listed on the remote", refs is not None, remote):
        return done()
    report.update(ledger=ledger, refs=refs)
    unlisted = sorted(set(refs) - set(ledger))
    missing = sorted(set(ledger) - set(refs))
    add("every acceptance ref has a ledger entry", not unlisted, unlisted)
    add("every ledger entry's ref is on the remote", not missing, missing)
    still_open = sorted(b for b, e in ledger.items() if e["status"] not in CLOSED)
    add("no ledger entry open", not still_open, still_open)
    extends = _frozen_extends(candidate)
    log = _log_at(candidate, "HEAD")
    add("candidate held-out log readable", log is not None)
    for branch, entry in sorted(ledger.items()):
        tip = entry["tip"]
        if not add(f"{branch} records its final tip", tip is not None, entry):
            continue
        if branch in refs:
            add(
                f"{branch} is at its recorded tip",
                refs[branch] == tip,
                {"remote": refs[branch], "ledger": tip},
            )
        _git(candidate, "fetch", "--quiet", remote, tip)
        known = _git(candidate, "cat-file", "-e", f"{tip}^{{commit}}") is not None
        if not add(f"{branch} tip resolved", known, tip):
            continue
        ancestor = _git(candidate, "merge-base", "--is-ancestor", tip, "HEAD")
        add(f"{branch} tip is an ancestor of the candidate", ancestor is not None, tip)
        old = _log_at(candidate, tip)
        add(
            f"candidate log extends the log at {branch}'s tip",
            old is not None and log is not None and extends(old, log),
            tip,
        )
    return done()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = check(args.candidate.resolve(), args.remote)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    for c in report["checks"]:
        detail = "" if c["ok"] else "  " + json.dumps(c["detail"])[:300]
        print(("ok   " if c["ok"] else "FAIL ") + c["check"] + detail)
    print(f"line check {'passed' if report['ok'] else 'FAILED'}: {report.get('head')}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
