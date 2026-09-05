"""The running release identity is the exact detached commit that owns a module."""

from __future__ import annotations

from pathlib import Path

from blueprint_pipeline import task_evaluation_launch_preparation_worker as preparation_worker
from blueprint_pipeline.task_evaluation_release_identity import running_release_commit

COMMIT = "9ae62694166fc4c7e54d318d5e2922108ec389d2"


def _worktree(tmp_path: Path, head: str) -> Path:
    """A release checkout as ``git worktree add --detach`` leaves it: a ``.git`` pointer file."""

    git_dir = tmp_path / "source" / ".git" / "worktrees" / COMMIT
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text(head + "\n", encoding="utf-8")
    release = tmp_path / "releases" / COMMIT
    (release / "src" / "blueprint_pipeline").mkdir(parents=True)
    (release / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    module = release / "src" / "blueprint_pipeline" / "worker.py"
    module.write_text("", encoding="utf-8")
    return module


def test_detached_release_worktree_reports_its_exact_commit(tmp_path: Path) -> None:
    module = _worktree(tmp_path, COMMIT.upper())
    assert running_release_commit(module) == COMMIT


def test_branch_checkout_has_no_release_identity(tmp_path: Path) -> None:
    module = _worktree(tmp_path, "ref: refs/heads/main")
    assert running_release_commit(module) == ""


def test_plain_repository_reads_head_directly(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text(COMMIT + "\n", encoding="utf-8")
    module = repo / "src" / "worker.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    assert running_release_commit(module) == COMMIT


def test_missing_repository_has_no_release_identity(tmp_path: Path) -> None:
    module = tmp_path / "loose" / "worker.py"
    module.parent.mkdir()
    module.write_text("", encoding="utf-8")
    assert running_release_commit(module) == ""


def test_preparation_worker_keeps_its_exported_name_for_the_same_helper() -> None:
    assert preparation_worker.running_worker_source_commit is running_release_commit
