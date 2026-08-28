from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_ephemeral_checkout_retention import (
    APPLY_ACKNOWLEDGEMENT,
    apply_ephemeral_checkout_retention_plan,
    build_ephemeral_checkout_retention_plan,
)
from blueprint_pipeline.task_evaluation_release_retention import (
    ReleaseRetentionError,
    _write_exclusive,
)


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "retention@example.invalid")
    _git(repo, "config", "user.name", "Retention Test")
    (repo / "payload.txt").write_text("first\n", encoding="utf-8")
    _git(repo, "add", "payload.txt")
    _git(repo, "commit", "-m", "first")
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "payload.txt").write_text("second\n", encoding="utf-8")
    _git(repo, "commit", "-am", "second")
    second = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", second)
    return repo, first, second


def _worktree(repo: Path, root: Path, commit: str) -> Path:
    root.mkdir(exist_ok=True)
    path = root / commit
    _git(repo, "worktree", "add", "--detach", str(path), commit)
    return path


def test_reaps_only_clean_remote_restageable_unpinned_checkouts(
    tmp_path: Path,
) -> None:
    repo, first, second = _repository(tmp_path)
    root_a = tmp_path / "source-only"
    root_b = tmp_path / "diagnostic-releases"
    stale = _worktree(repo, root_a, first)
    kept = _worktree(repo, root_b, second)

    plan = build_ephemeral_checkout_retention_plan(
        checkout_roots=(root_a, root_b), keep_commits=(second,)
    )
    assert [row["source_commit"] for row in plan["eligible_checkouts"]] == [
        first
    ]
    assert plan["retained_checkouts"][0]["source_commit"] == second
    assert plan["predicted_removed_bytes"] > 0
    plan_path = tmp_path / "plan.json"
    _write_exclusive(plan_path, plan)

    result = apply_ephemeral_checkout_retention_plan(
        dry_run_plan_path=plan_path,
        acknowledgement=APPLY_ACKNOWLEDGEMENT,
        receipt_out=tmp_path / "receipt.json",
    )

    assert result["status"] == "applied"
    assert result["removed_bytes"] == plan["predicted_removed_bytes"]
    assert not stale.exists()
    assert kept.is_dir()
    assert result["evidence_artifacts_removed"] is False


def test_dirty_or_not_remotely_restageable_checkout_blocks(
    tmp_path: Path,
) -> None:
    repo, first, second = _repository(tmp_path)
    root = tmp_path / "source-only"
    checkout = _worktree(repo, root, first)
    _git(repo, "update-ref", "-d", "refs/remotes/origin/main")
    with pytest.raises(
        ReleaseRetentionError,
        match="ephemeral_checkout_retention_remote_ref_missing",
    ):
        build_ephemeral_checkout_retention_plan(checkout_roots=(root,))

    _git(repo, "update-ref", "refs/remotes/origin/main", second)
    (checkout / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(
        ReleaseRetentionError,
        match="ephemeral_checkout_retention_checkout_dirty",
    ):
        build_ephemeral_checkout_retention_plan(checkout_roots=(root,))
