from __future__ import annotations

import json
import stat
import subprocess
import time
from pathlib import Path

import pytest

from blueprint_pipeline import (
    task_evaluation_scene_configuration_diagnostic_release as release,
)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(repo: Path, name: str, contents: str) -> str:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-m", name)
    return _git(repo, "rev-parse", "HEAD")


def _pushed_codex_branch(tmp_path: Path) -> tuple[Path, str, str]:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)
    repo = tmp_path / "source"
    subprocess.run(["git", "init", "--initial-branch=main", str(repo)], check=True)
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Blueprint tests")
    _git(repo, "remote", "add", "origin", str(remote))
    _commit(repo, "base.txt", "base\n")
    branch = "codex/scene-fast-fix"
    _git(repo, "checkout", "-b", branch)
    commit = _commit(repo, "src/blueprint_pipeline/fix.py", "FIX = True\n")
    _git(repo, "push", "-u", "origin", branch)
    return repo, branch, commit


def test_stages_exact_pushed_branch_source_only_and_reuses_it_in_seconds(
    tmp_path: Path,
) -> None:
    source, branch, commit = _pushed_codex_branch(tmp_path)
    release_root = tmp_path / "diagnostic-releases"
    state_root = tmp_path / "diagnostic-state"
    active_production_link = tmp_path / "production-active"
    production_runtime = tmp_path / "system-runtimes" / "splat-render"
    production_runtime.mkdir(parents=True)
    sentinel = production_runtime / "must-not-copy-or-change"
    sentinel.write_bytes(b"runtime-sentinel")

    first = release.stage_scene_configuration_diagnostic_release(
        source_repo=source.resolve(),
        source_commit=commit,
        remote_branch=branch,
        release_root=release_root.resolve(),
        state_root=state_root.resolve(),
    )
    started = time.monotonic()
    second = release.stage_scene_configuration_diagnostic_release(
        source_repo=source.resolve(),
        source_commit=commit,
        remote_branch=branch,
        release_root=release_root.resolve(),
        state_root=state_root.resolve(),
    )
    wall_seconds = time.monotonic() - started

    checkout = release_root / commit
    assert first["created_release_checkout"] is True
    assert first["active_release_link_updated"] is False
    assert first["runtime_assets_copied"] is False
    assert first["systemd_units_reinstalled"] is False
    assert first["systemd_services_restarted"] is False
    assert first["diagnostic_only"] is True
    assert first["development_only"] is True
    assert first["qualification_eligible"] is False
    assert first["offering_publication_permitted"] is False
    assert second["created_release_checkout"] is False
    assert second["reused_existing_checkout"] is True
    assert second["source_materialization_elapsed_ms"] < 5_000
    assert wall_seconds < 5.0
    assert _git(checkout, "rev-parse", "HEAD") == commit
    assert _git(checkout, "branch", "--show-current") == ""
    assert _git(checkout, "status", "--porcelain", "--untracked-files=all") == ""
    assert active_production_link.exists() is False
    assert sentinel.read_bytes() == b"runtime-sentinel"
    receipt_path = Path(first["receipt_path"])
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o440
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == json.loads(
        Path(second["receipt_path"]).read_text(encoding="utf-8")
    )


def test_validation_rechecks_live_remote_tip_and_refuses_when_it_moves(
    tmp_path: Path,
) -> None:
    source, branch, commit = _pushed_codex_branch(tmp_path)
    staged = release.stage_scene_configuration_diagnostic_release(
        source_repo=source.resolve(),
        source_commit=commit,
        remote_branch=branch,
        release_root=(tmp_path / "releases").resolve(),
        state_root=(tmp_path / "state").resolve(),
    )
    newer = _commit(source, "newer.py", "NEWER = True\n")
    assert newer != commit
    _git(source, "push", "origin", branch)

    with pytest.raises(
        release.SceneConfigurationDiagnosticReleaseError,
        match="remote_ref_moved",
    ):
        release.validate_scene_configuration_diagnostic_release_receipt(
            staged["receipt_path"],
            expected_source_commit=commit,
            expected_release_path=staged["release_path"],
        )


def test_stage_fetches_exact_branch_when_production_clone_lacks_local_object(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)
    publisher = tmp_path / "publisher"
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(publisher)], check=True
    )
    _git(publisher, "config", "user.email", "tests@example.invalid")
    _git(publisher, "config", "user.name", "Blueprint tests")
    _git(publisher, "remote", "add", "origin", str(remote))
    _commit(publisher, "base.txt", "base\n")
    _git(publisher, "push", "-u", "origin", "main")
    branch = "codex/remote-only-scene-fix"
    _git(publisher, "checkout", "-b", branch)
    branch_commit = _commit(publisher, "scene_fix.py", "FIX = True\n")
    _git(publisher, "push", "-u", "origin", branch)

    source = tmp_path / "production-clone"
    subprocess.run(
        [
            "git",
            "clone",
            "--no-local",
            "--single-branch",
            "--branch",
            "main",
            str(remote),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    absent = subprocess.run(
        ["git", "-C", str(source), "cat-file", "-e", f"{branch_commit}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert absent.returncode != 0

    staged = release.stage_scene_configuration_diagnostic_release(
        source_repo=source.resolve(),
        source_commit=branch_commit,
        remote_branch=branch,
        release_root=(tmp_path / "releases").resolve(),
        state_root=(tmp_path / "state").resolve(),
    )

    assert staged["source_commit"] == branch_commit
    assert staged["local_tracking_ref"] == f"refs/remotes/origin/{branch}"
    assert _git(source, "branch", "--show-current") == "main"
    assert _git(source, "rev-parse", f"refs/remotes/origin/{branch}") == branch_commit
    assert _git(Path(staged["release_path"]), "rev-parse", "HEAD") == branch_commit


def test_validation_refuses_dirty_or_wrong_allocator_checkout(tmp_path: Path) -> None:
    source, branch, commit = _pushed_codex_branch(tmp_path)
    staged = release.stage_scene_configuration_diagnostic_release(
        source_repo=source.resolve(),
        source_commit=commit,
        remote_branch=branch,
        release_root=(tmp_path / "releases").resolve(),
        state_root=(tmp_path / "state").resolve(),
    )
    with pytest.raises(
        release.SceneConfigurationDiagnosticReleaseError,
        match="allocator_checkout_mismatch",
    ):
        release.validate_scene_configuration_diagnostic_release_receipt(
            staged["receipt_path"],
            expected_source_commit=commit,
            expected_release_path=(tmp_path / "other-release").resolve(),
        )

    (Path(staged["release_path"]) / "untracked.py").write_text(
        "DIRTY = True\n", encoding="utf-8"
    )
    with pytest.raises(
        release.SceneConfigurationDiagnosticReleaseError,
        match="checkout_not_clean",
    ):
        release.validate_scene_configuration_diagnostic_release_receipt(
            staged["receipt_path"],
            expected_source_commit=commit,
            expected_release_path=staged["release_path"],
        )


def test_rejects_non_codex_branch_before_creating_any_release(tmp_path: Path) -> None:
    source, _branch, commit = _pushed_codex_branch(tmp_path)
    release_root = tmp_path / "releases"
    with pytest.raises(
        release.SceneConfigurationDiagnosticReleaseError,
        match="remote_branch_invalid",
    ):
        release.stage_scene_configuration_diagnostic_release(
            source_repo=source.resolve(),
            source_commit=commit,
            remote_branch="feature/scene-fix",
            release_root=release_root.resolve(),
            state_root=(tmp_path / "state").resolve(),
        )
    assert not release_root.exists()


def test_receipt_digest_and_mode_fail_closed(tmp_path: Path) -> None:
    source, branch, commit = _pushed_codex_branch(tmp_path)
    staged = release.stage_scene_configuration_diagnostic_release(
        source_repo=source.resolve(),
        source_commit=commit,
        remote_branch=branch,
        release_root=(tmp_path / "releases").resolve(),
        state_root=(tmp_path / "state").resolve(),
    )
    receipt = Path(staged["receipt_path"])
    receipt.chmod(0o640)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["offering_publication_permitted"] = True
    receipt.write_text(json.dumps(value), encoding="utf-8")
    receipt.chmod(0o440)
    with pytest.raises(
        release.SceneConfigurationDiagnosticReleaseError,
        match="digest_mismatch",
    ):
        release.validate_scene_configuration_diagnostic_release_receipt(
            receipt,
            expected_source_commit=commit,
            expected_release_path=staged["release_path"],
        )

    receipt.chmod(0o660)
    with pytest.raises(
        release.SceneConfigurationDiagnosticReleaseError,
        match="receipt_invalid",
    ):
        release.validate_scene_configuration_diagnostic_release_receipt(
            receipt,
            expected_source_commit=commit,
            expected_release_path=staged["release_path"],
        )
    receipt.chmod(0o640)


def test_absolute_roots_are_required(tmp_path: Path) -> None:
    source, branch, commit = _pushed_codex_branch(tmp_path)
    with pytest.raises(
        release.SceneConfigurationDiagnosticReleaseError,
        match="release_root_must_be_absolute",
    ):
        release.stage_scene_configuration_diagnostic_release(
            source_repo=source.resolve(),
            source_commit=commit,
            remote_branch=branch,
            release_root=Path("relative-releases"),
            state_root=(tmp_path / "state").resolve(),
        )


def test_release_and_state_roots_cannot_be_symlinks(tmp_path: Path) -> None:
    source, branch, commit = _pushed_codex_branch(tmp_path)
    real_releases = tmp_path / "real-releases"
    real_releases.mkdir()
    linked_releases = tmp_path / "linked-releases"
    linked_releases.symlink_to(real_releases, target_is_directory=True)
    with pytest.raises(
        release.SceneConfigurationDiagnosticReleaseError,
        match="root_symlink",
    ):
        release.stage_scene_configuration_diagnostic_release(
            source_repo=source.resolve(),
            source_commit=commit,
            remote_branch=branch,
            release_root=linked_releases,
            state_root=(tmp_path / "state").resolve(),
        )
