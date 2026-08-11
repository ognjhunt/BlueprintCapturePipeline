from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import stage_task_evaluation_control_plane_release as releases


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(repo: Path, name: str, payload: str) -> str:
    (repo / name).write_text(payload, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-m", name)
    return _git(repo, "rev-parse", "HEAD")


def _source_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "source"
    subprocess.run(["git", "init", "--initial-branch=main", str(repo)], check=True)
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Blueprint tests")
    first = _commit(repo, "first.txt", "first")
    _git(repo, "update-ref", "refs/remotes/origin/main", first)
    second = _commit(repo, "second.txt", "second")
    _git(repo, "update-ref", "refs/remotes/origin/main", second)
    return repo, first, second


def test_git_argv_scopes_safe_directory_to_the_requested_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "source"
    checkout.mkdir()

    assert releases._git_argv(checkout, "status", "--porcelain") == [
        "git",
        "-c",
        f"safe.directory={checkout.resolve()}",
        "-C",
        str(checkout.resolve()),
        "status",
        "--porcelain",
    ]


def test_run_git_accepts_explicitly_trusted_different_owner_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _first, second = _source_repo(tmp_path)
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")

    assert releases._run_git(source, "rev-parse", "HEAD") == second


def test_stages_clean_protected_main_ancestor_then_activates_atomically(tmp_path: Path) -> None:
    source, first, second = _source_repo(tmp_path)
    release_root = tmp_path / "releases"
    state_root = tmp_path / "state"
    active_link = tmp_path / "active-control-plane"

    staged = releases.stage_task_evaluation_control_plane_release(
        source_repo=source,
        source_commit=first,
        release_root=release_root,
        state_root=state_root,
        active_link=active_link,
    )

    release = release_root / first
    assert staged["status"] == "staged"
    assert staged["source_commit"] == first
    assert staged["origin_main_at_stage"] == second
    assert staged["created_release_checkout"] is True
    assert staged["activated"] is False
    assert _git(release, "rev-parse", "HEAD") == first
    assert _git(release, "status", "--porcelain", "--untracked-files=no") == ""
    assert active_link.exists() is False
    stage_receipt = json.loads((state_root / first / "stage.json").read_text())
    assert stage_receipt["receipt_digest"].startswith("sha256:")

    third = _commit(source, "third.txt", "third")
    _git(source, "update-ref", "refs/remotes/origin/main", third)

    activated = releases.stage_task_evaluation_control_plane_release(
        source_repo=source,
        source_commit=first,
        release_root=release_root,
        state_root=state_root,
        active_link=active_link,
        activate=True,
    )

    assert activated["created_release_checkout"] is False
    assert activated["activated"] is True
    assert active_link.is_symlink()
    assert active_link.resolve() == release.resolve()
    activation_receipt = json.loads((state_root / first / "activation.json").read_text())
    assert activation_receipt["status"] == "activated"
    assert activation_receipt["active_link_target"] == str(release.resolve())
    assert json.loads((state_root / first / "stage.json").read_text()) == stage_receipt


def test_rejects_unmerged_source_commit_before_creating_a_release(tmp_path: Path) -> None:
    source, _first, _second = _source_repo(tmp_path)
    _git(source, "checkout", "-b", "unmerged")
    unmerged = _commit(source, "unmerged.txt", "unmerged")
    _git(source, "checkout", "main")
    release_root = tmp_path / "releases"

    with pytest.raises(
        releases.ControlPlaneReleaseError,
        match="source_not_protected_main",
    ):
        releases.stage_task_evaluation_control_plane_release(
            source_repo=source,
            source_commit=unmerged,
            release_root=release_root,
            state_root=tmp_path / "state",
            active_link=tmp_path / "active",
        )

    assert not release_root.exists()


def test_rejects_an_existing_release_path_with_the_wrong_identity(tmp_path: Path) -> None:
    source, first, second = _source_repo(tmp_path)
    release_root = tmp_path / "releases"
    conflicting = release_root / first
    conflicting.mkdir(parents=True)
    (conflicting / "not-a-git-release").write_text("wrong", encoding="utf-8")

    with pytest.raises(
        releases.ControlPlaneReleaseError,
        match="checkout_invalid|git_command_failed",
    ):
        releases.stage_task_evaluation_control_plane_release(
            source_repo=source,
            source_commit=first,
            release_root=release_root,
            state_root=tmp_path / "state",
            active_link=tmp_path / "active",
        )

    assert _git(source, "rev-parse", "HEAD") == second
