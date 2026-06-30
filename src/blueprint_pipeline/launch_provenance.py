"""Shared source-provenance gates for paid provider launches."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "launch_provenance.v1"

GIT_EVIDENCE_UNAVAILABLE_NOTE = (
    "Paid launch blocked because the committed source boundary could not be "
    "verified. Re-run only after git evidence is available, or pass the explicit "
    "dirty-tree override and accept the provenance risk."
)
DIRTY_WORKTREE_PAID_LAUNCH_NOTE = (
    "Paid launch blocked from a dirty tree so cloud frames cannot be confused "
    "with committed source. Commit/stash first, or pass the explicit dirty-tree "
    "override and preserve this git_evidence in the launch manifest."
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_worktree_evidence(
    *,
    repo_root: Path | None = None,
    max_dirty_entries: int = 200,
) -> dict[str, Any]:
    """Return the committed SHA + dirty state used as a paid-launch provenance boundary."""
    root = (repo_root or _repo_root()).expanduser().resolve()

    def _git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    head = _git("rev-parse", "HEAD")
    if head.returncode != 0:
        return {
            "status": "unavailable",
            "repo_root": str(root),
            "dirty": None,
            "error": (head.stderr or head.stdout).strip()[:500],
        }
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status.returncode != 0:
        return {
            "status": "unavailable",
            "repo_root": str(root),
            "git_sha": head.stdout.strip(),
            "dirty": None,
            "error": (status.stderr or status.stdout).strip()[:500],
        }
    dirty_entries = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "status": "available",
        "repo_root": str(root),
        "git_sha": head.stdout.strip(),
        "dirty": bool(dirty_entries),
        "dirty_entries_count": len(dirty_entries),
        "dirty_entries": dirty_entries[:max_dirty_entries],
        "dirty_entries_truncated": len(dirty_entries) > max_dirty_entries,
    }


def evaluate_dirty_tree_paid_launch_gate(
    *,
    git_evidence: Mapping[str, Any],
    allow_paid: bool,
    allow_dirty_paid_launch: bool,
) -> dict[str, Any]:
    if not allow_paid or allow_dirty_paid_launch:
        return {"launch_allowed": True, "blockers": [], "note": None}
    if git_evidence.get("status") != "available":
        return {
            "launch_allowed": False,
            "blockers": ["git_worktree_evidence_unavailable"],
            "note": GIT_EVIDENCE_UNAVAILABLE_NOTE,
        }
    if git_evidence.get("dirty"):
        return {
            "launch_allowed": False,
            "blockers": ["dirty_worktree_paid_launch_blocked"],
            "note": DIRTY_WORKTREE_PAID_LAUNCH_NOTE,
        }
    return {"launch_allowed": True, "blockers": [], "note": None}
