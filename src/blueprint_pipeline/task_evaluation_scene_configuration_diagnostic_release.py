"""Immutable pushed-branch source identity for scene diagnostic retries.

This is deliberately not a production release promoter.  It materializes one
detached checkout of the exact tip of a pushed ``codex/*`` branch in a
diagnostic-only release root, writes a digest-bound receipt, and never knows
the production active-link path or any systemd command.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess  # nosec B404 - fixed Git executable and argv only
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "task_evaluation_scene_configuration_diagnostic_release.v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_BRANCH = re.compile(r"codex/[a-z0-9][a-z0-9._/-]{0,180}")


class SceneConfigurationDiagnosticReleaseError(ValueError):
    """The source-only diagnostic release identity could not be proven."""


def _canonical_digest(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("receipt_digest", None)
    payload = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _absolute(value: str | Path, *, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SceneConfigurationDiagnosticReleaseError(
            f"scene_configuration_diagnostic_release_{field}_must_be_absolute"
        )
    return path


def _valid_branch(value: str) -> bool:
    if _BRANCH.fullmatch(value) is None:
        return False
    segments = value.split("/")
    return bool(
        all(segment not in {"", ".", ".."} for segment in segments)
        and not value.endswith("/")
        and not any(segment.endswith(".lock") for segment in segments)
    )


def _git_argv(repo: Path, *arguments: str) -> list[str]:
    checkout = repo.resolve()
    return [
        "git",
        "-c",
        f"safe.directory={checkout}",
        "-C",
        str(checkout),
        *arguments,
    ]


def _git(
    repo: Path,
    *arguments: str,
    allow_failure: bool = False,
    timeout_seconds: int = 30,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(  # nosec B603 - fixed Git executable and argv
            _git_argv(repo, *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_git_unavailable"
        ) from exc
    if result.returncode != 0 and not allow_failure:
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_git_command_failed"
        )
    return result


def _remote_tip(repo: Path, branch: str) -> str:
    reference = f"refs/heads/{branch}"
    result = _git(
        repo,
        "ls-remote",
        "--exit-code",
        "origin",
        reference,
        timeout_seconds=20,
    )
    fields = result.stdout.strip().split()
    commit = fields[0].lower() if len(fields) == 2 else ""
    if (
        fields[1:] != [reference]
        or _COMMIT.fullmatch(commit) is None
    ):
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_remote_ref_invalid"
        )
    return commit


def _fetch_exact_remote_branch(repo: Path, branch: str, expected_commit: str) -> str:
    """Materialize the remote-only commit without moving the source checkout."""

    remote_ref = f"refs/heads/{branch}"
    tracking_ref = f"refs/remotes/origin/{branch}"
    _git(
        repo,
        "fetch",
        "--no-tags",
        "--force",
        "origin",
        f"{remote_ref}:{tracking_ref}",
        timeout_seconds=120,
    )
    fetched = _git(
        repo, "rev-parse", "--verify", f"{tracking_ref}^{{commit}}"
    ).stdout.strip().lower()
    if fetched != expected_commit:
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_fetched_ref_mismatch"
        )
    return tracking_ref


def _assert_release_checkout(path: Path, source_commit: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_checkout_invalid"
        )
    head = _git(path, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip().lower()
    if head != source_commit:
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_checkout_commit_mismatch"
        )
    symbolic = _git(path, "symbolic-ref", "-q", "HEAD", allow_failure=True)
    if symbolic.returncode == 0 or symbolic.stdout.strip():
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_checkout_not_detached"
        )
    if _git(
        path, "status", "--porcelain", "--untracked-files=all"
    ).stdout.strip():
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_checkout_not_clean"
        )


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o022:
            raise OSError("unsafe receipt")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_receipt_invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_receipt_invalid"
        )
    receipt = dict(value)
    if receipt.get("receipt_digest") != _canonical_digest(receipt):
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_receipt_digest_mismatch"
        )
    return receipt


def _write_exact(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0),
            0o440,
        )
    except FileExistsError:
        if path.read_bytes() != payload:
            raise SceneConfigurationDiagnosticReleaseError(
                "scene_configuration_diagnostic_release_receipt_conflict"
            )
        return
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short diagnostic release receipt write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o440)
    finally:
        os.close(descriptor)


def validate_scene_configuration_diagnostic_release_receipt(
    receipt_path: str | Path,
    *,
    expected_source_commit: str,
    expected_release_path: str | Path | None = None,
) -> dict[str, Any]:
    """Re-prove the detached checkout and the live pushed branch tip."""

    expected = expected_source_commit.strip().lower()
    if _COMMIT.fullmatch(expected) is None:
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_source_commit_invalid"
        )
    path = _absolute(receipt_path, field="receipt_path")
    receipt = _read_receipt(path)
    branch = str(receipt.get("remote_branch") or "")
    source_repo = Path(str(receipt.get("source_repo") or ""))
    release_path = Path(str(receipt.get("release_path") or ""))
    release_root = Path(str(receipt.get("release_root") or ""))
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "staged"
        or receipt.get("source_commit") != expected
        or not _valid_branch(branch)
        or receipt.get("remote_name") != "origin"
        or receipt.get("remote_ref") != f"refs/heads/{branch}"
        or receipt.get("remote_ref_tip_commit") != expected
        or receipt.get("local_tracking_ref")
        != f"refs/remotes/origin/{branch}"
        or not source_repo.is_absolute()
        or source_repo.is_symlink()
        or not source_repo.is_dir()
        or not release_root.is_absolute()
        or release_root.is_symlink()
        or release_path != release_root / expected
        or receipt.get("release_checkout_detached") is not True
        or receipt.get("release_checkout_clean") is not True
        or receipt.get("source_only_release") is not True
        or receipt.get("diagnostic_only") is not True
        or receipt.get("development_only") is not True
        or receipt.get("qualification_eligible") is not False
        or receipt.get("configured_revision_publication_permitted") is not False
        or receipt.get("offering_publication_permitted") is not False
        or receipt.get("terminal_e2e_completion_permitted") is not False
        or receipt.get("active_release_link_updated") is not False
        or receipt.get("runtime_assets_copied") is not False
        or receipt.get("systemd_units_reinstalled") is not False
        or receipt.get("systemd_services_restarted") is not False
        or receipt.get("provider_mutation_performed") is not False
        or receipt.get("raw_secret_values_recorded") is not False
    ):
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_receipt_invalid"
        )
    if expected_release_path is not None:
        required_release = _absolute(
            expected_release_path, field="expected_release_path"
        ).resolve()
        if release_path.resolve() != required_release:
            raise SceneConfigurationDiagnosticReleaseError(
                "scene_configuration_diagnostic_release_allocator_checkout_mismatch"
            )
    _assert_release_checkout(release_path, expected)
    if _remote_tip(source_repo, branch) != expected:
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_remote_ref_moved"
        )
    tracked = _git(
        source_repo,
        "rev-parse",
        "--verify",
        f"refs/remotes/origin/{branch}^{{commit}}",
    ).stdout.strip().lower()
    if tracked != expected:
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_tracking_ref_mismatch"
        )
    return receipt


def stage_scene_configuration_diagnostic_release(
    *,
    source_repo: str | Path,
    source_commit: str,
    remote_branch: str,
    release_root: str | Path,
    state_root: str | Path,
) -> dict[str, Any]:
    """Materialize an idempotent source-only release without activating it."""

    started = time.monotonic()
    source = _absolute(source_repo, field="source_repo")
    releases = _absolute(release_root, field="release_root")
    state = _absolute(state_root, field="state_root")
    commit = source_commit.strip().lower()
    if _COMMIT.fullmatch(commit) is None:
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_source_commit_invalid"
        )
    if not _valid_branch(remote_branch):
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_remote_branch_invalid"
        )
    if source.is_symlink() or not source.is_dir():
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_source_repo_invalid"
        )
    if releases.is_symlink() or state.is_symlink():
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_root_symlink"
        )
    if (
        releases == source
        or releases in source.parents
        or source in releases.parents
        or state == source
        or state in source.parents
        or state == releases
        or state in releases.parents
    ):
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_roots_overlap"
        )
    if _remote_tip(source, remote_branch) != commit:
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_not_remote_ref_tip"
        )
    tracking_ref = _fetch_exact_remote_branch(source, remote_branch, commit)
    if _remote_tip(source, remote_branch) != commit:
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_remote_ref_moved_during_fetch"
        )
    resolved = _git(
        source, "rev-parse", "--verify", f"{commit}^{{commit}}"
    ).stdout.strip().lower()
    if resolved != commit:
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_source_commit_unavailable"
        )
    release_path = releases / commit
    created = False
    if release_path.exists() or os.path.lexists(release_path):
        _assert_release_checkout(release_path, commit)
    else:
        releases.mkdir(parents=True, exist_ok=True, mode=0o755)
        result = _git(
            source,
            "worktree",
            "add",
            "--detach",
            str(release_path),
            commit,
            allow_failure=True,
            timeout_seconds=120,
        )
        if result.returncode != 0:
            if release_path.exists() and not release_path.is_symlink():
                _assert_release_checkout(release_path, commit)
            else:
                raise SceneConfigurationDiagnosticReleaseError(
                    "scene_configuration_diagnostic_release_checkout_create_failed"
                )
        else:
            created = True
            _assert_release_checkout(release_path, commit)
    release_path = release_path.resolve()
    receipt_path = state / commit / "diagnostic-release.json"
    if receipt_path.parent.is_symlink():
        raise SceneConfigurationDiagnosticReleaseError(
            "scene_configuration_diagnostic_release_state_path_symlink"
        )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "staged",
        "source_commit": commit,
        "source_repo": str(source),
        "remote_name": "origin",
        "remote_branch": remote_branch,
        "remote_ref": f"refs/heads/{remote_branch}",
        "remote_ref_tip_commit": commit,
        "local_tracking_ref": tracking_ref,
        "release_root": str(releases),
        "release_path": str(release_path),
        "release_checkout_detached": True,
        "release_checkout_clean": True,
        "source_only_release": True,
        "diagnostic_only": True,
        "development_only": True,
        "qualification_eligible": False,
        "configured_revision_publication_permitted": False,
        "offering_publication_permitted": False,
        "terminal_e2e_completion_permitted": False,
        "active_release_link_updated": False,
        "runtime_assets_copied": False,
        "systemd_units_reinstalled": False,
        "systemd_services_restarted": False,
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = _canonical_digest(receipt)
    _write_exact(receipt_path, receipt)
    validated = validate_scene_configuration_diagnostic_release_receipt(
        receipt_path,
        expected_source_commit=commit,
        expected_release_path=release_path,
    )
    return {
        **validated,
        "receipt_path": str(receipt_path),
        "created_release_checkout": created,
        "reused_existing_checkout": not created,
        "source_materialization_elapsed_ms": int(
            (time.monotonic() - started) * 1000
        ),
        "remote_ref_tip_revalidated": True,
    }


__all__ = [
    "SCHEMA_VERSION",
    "SceneConfigurationDiagnosticReleaseError",
    "stage_scene_configuration_diagnostic_release",
    "validate_scene_configuration_diagnostic_release_receipt",
]
