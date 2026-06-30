from __future__ import annotations

import subprocess
from pathlib import Path

from blueprint_pipeline.launch_provenance import (
    DIRTY_WORKTREE_PAID_LAUNCH_NOTE,
    GIT_EVIDENCE_UNAVAILABLE_NOTE,
    evaluate_dirty_tree_paid_launch_gate,
    git_worktree_evidence,
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_dirty_tree_paid_launch_gate_allows_clean_paid_launch() -> None:
    gate = evaluate_dirty_tree_paid_launch_gate(
        git_evidence={"status": "available", "dirty": False},
        allow_paid=True,
        allow_dirty_paid_launch=False,
    )

    assert gate == {"launch_allowed": True, "blockers": [], "note": None}


def test_dirty_tree_paid_launch_gate_blocks_dirty_paid_launch() -> None:
    gate = evaluate_dirty_tree_paid_launch_gate(
        git_evidence={"status": "available", "dirty": True},
        allow_paid=True,
        allow_dirty_paid_launch=False,
    )

    assert gate["launch_allowed"] is False
    assert gate["blockers"] == ["dirty_worktree_paid_launch_blocked"]
    assert gate["note"] == DIRTY_WORKTREE_PAID_LAUNCH_NOTE


def test_dirty_tree_paid_launch_gate_blocks_unavailable_git_evidence() -> None:
    gate = evaluate_dirty_tree_paid_launch_gate(
        git_evidence={"status": "unavailable", "dirty": None},
        allow_paid=True,
        allow_dirty_paid_launch=False,
    )

    assert gate["launch_allowed"] is False
    assert gate["blockers"] == ["git_worktree_evidence_unavailable"]
    assert gate["note"] == GIT_EVIDENCE_UNAVAILABLE_NOTE


def test_dirty_tree_paid_launch_gate_bypasses_without_paid_launch_or_with_override() -> None:
    dirty_evidence = {"status": "available", "dirty": True}

    assert evaluate_dirty_tree_paid_launch_gate(
        git_evidence=dirty_evidence,
        allow_paid=False,
        allow_dirty_paid_launch=False,
    )["launch_allowed"] is True
    assert evaluate_dirty_tree_paid_launch_gate(
        git_evidence=dirty_evidence,
        allow_paid=True,
        allow_dirty_paid_launch=True,
    )["launch_allowed"] is True


def test_git_worktree_evidence_reports_clean_and_dirty_temp_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "Tests")
    tracked = repo / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")

    clean = git_worktree_evidence(repo_root=repo)
    assert clean["status"] == "available"
    assert clean["git_sha"]
    assert clean["dirty"] is False
    assert clean["dirty_entries_count"] == 0
    assert clean["dirty_entries"] == []

    tracked.write_text("dirty\n", encoding="utf-8")
    dirty = git_worktree_evidence(repo_root=repo)
    assert dirty["status"] == "available"
    assert dirty["git_sha"] == clean["git_sha"]
    assert dirty["dirty"] is True
    assert dirty["dirty_entries_count"] == 1
    assert dirty["dirty_entries"][0].endswith("tracked.txt")
