#!/usr/bin/env python3
"""Stage one immutable source checkout for Task Evaluation control-plane work.

The normal production repository checkout may advance with protected ``main``.
That checkout is suitable for building a release, but it is not a valid runtime
identity for a profile that names a particular source commit.  This utility
creates one detached, clean Git worktree for that commit and, only when
explicitly requested, atomically switches the non-secret active-release link.

It never runs an allocator, contacts a provider, reads a credential, or edits
the mutable source checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess  # nosec B404 - fixed git argv over operator-supplied paths
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "task_evaluation_control_plane_release.v1"


class ControlPlaneReleaseError(ValueError):
    """Raised when the immutable release boundary cannot be proven."""


def _canonical_digest(value: Mapping[str, Any], *, digest_field: str) -> str:
    body = dict(value)
    body.pop(digest_field, None)
    payload = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_exact(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ControlPlaneReleaseError(
                f"task_evaluation_control_plane_release_receipt_conflict:{path.name}"
            )


def _read_exact_receipt(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlPlaneReleaseError(
            f"task_evaluation_control_plane_release_receipt_invalid:{path.name}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ControlPlaneReleaseError(
            f"task_evaluation_control_plane_release_receipt_invalid:{path.name}"
        )
    receipt = dict(value)
    if receipt.get("receipt_digest") != _canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        raise ControlPlaneReleaseError(
            f"task_evaluation_control_plane_release_receipt_digest_mismatch:{path.name}"
        )
    return receipt


def _git_argv(repo: Path, *args: str) -> list[str]:
    """Build a Git command that trusts only the intended immutable checkout.

    The production promoter may run as a service account while the source
    checkout is owned by a deployment account.  Git otherwise rejects that
    normal split as a dubious-ownership repository before the identity checks
    can run.  Scope the exception to this one resolved checkout instead of
    changing a user's global Git configuration.
    """

    checkout = repo.resolve()
    return [
        "git",
        "-c",
        f"safe.directory={checkout}",
        "-C",
        str(checkout),
        *args,
    ]


def _run_git(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(  # nosec B603 - fixed Git executable and argv
            _git_argv(repo, *args),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError as exc:
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_git_unavailable"
        ) from exc
    if completed.returncode != 0:
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_git_command_failed"
        )
    return completed.stdout.strip()


def _valid_commit(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _absolute_path(value: str | Path, *, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ControlPlaneReleaseError(
            f"task_evaluation_control_plane_release_{field}_must_be_absolute"
        )
    return path


def _is_ancestor(repo: Path, commit: str, ref: str) -> bool:
    try:
        completed = subprocess.run(  # nosec B603 - fixed Git executable and argv
            _git_argv(repo, "merge-base", "--is-ancestor", commit, ref),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError as exc:
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_git_unavailable"
        ) from exc
    return completed.returncode == 0


def _assert_source(repo: Path, source_commit: str) -> tuple[str, str]:
    commit = source_commit.strip().lower()
    if not _valid_commit(commit):
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_source_commit_invalid"
        )
    if not repo.is_dir() or repo.is_symlink():
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_source_repo_invalid"
        )
    if _run_git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_source_checkout_not_clean"
        )
    resolved = _run_git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").lower()
    if resolved != commit:
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_source_commit_unavailable"
        )
    origin_main = _run_git(repo, "rev-parse", "--verify", "origin/main^{commit}").lower()
    if not _valid_commit(origin_main) or not _is_ancestor(repo, commit, "origin/main"):
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_source_not_protected_main"
        )
    return commit, origin_main


def _assert_release_checkout(path: Path, source_commit: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_checkout_invalid"
        )
    if _run_git(path, "rev-parse", "--verify", "HEAD^{commit}").lower() != source_commit:
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_checkout_commit_mismatch"
        )
    if _run_git(path, "status", "--porcelain", "--untracked-files=no"):
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_checkout_not_clean"
        )


def _create_release_checkout(*, source_repo: Path, release_path: Path, source_commit: str) -> bool:
    if release_path.exists() or os.path.lexists(release_path):
        _assert_release_checkout(release_path, source_commit)
        return False
    release_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(  # nosec B603 - fixed Git executable and argv
            _git_argv(
                source_repo,
                "worktree",
                "add",
                "--detach",
                str(release_path),
                source_commit,
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except OSError as exc:
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_worktree_unavailable"
        ) from exc
    if completed.returncode != 0:
        # A concurrent promoter may have made the exact immutable checkout.
        if release_path.exists() and not release_path.is_symlink():
            _assert_release_checkout(release_path, source_commit)
            return False
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_worktree_create_failed"
        )
    _assert_release_checkout(release_path, source_commit)
    return True


def _install_release_git_index_readability(
    *, source_repo: Path, release_path: Path
) -> dict[str, Any]:
    """Make the detached worktree index readable by the runtime account.

    Production deploys run as root while the allocator runs as ``blueprint``.
    A restrictive deploy umask made Git create a root:root 0640 worktree index:
    ``rev-parse`` still worked, but the required clean-checkout ``git status``
    failed before allocation. Git's normal index mode is 0644 and it contains
    only tree metadata, so install that exact read-only posture here.
    """

    index_value = _run_git(release_path, "rev-parse", "--git-path", "index")
    candidate = Path(index_value)
    index_path = (
        candidate.resolve()
        if candidate.is_absolute()
        else (release_path / candidate).resolve()
    )
    admin_root = (source_repo.resolve() / ".git" / "worktrees").resolve()
    if (
        index_path.name != "index"
        or admin_root not in index_path.parents
        or index_path.is_symlink()
        or not index_path.is_file()
    ):
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_git_index_invalid"
        )
    try:
        index_path.chmod(0o644)
    except OSError as exc:
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_git_index_unreadable"
        ) from exc
    if stat.S_IMODE(index_path.stat().st_mode) != 0o644:
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_git_index_unreadable"
        )
    return {
        "git_index_path": str(index_path),
        "git_index_mode": "0644",
        "runtime_readable": True,
    }


def _activate_release(*, active_link: Path, release_path: Path) -> None:
    if active_link.exists() and not active_link.is_symlink():
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_active_link_conflict"
        )
    active_link.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{active_link.name}.", suffix=".tmp", dir=active_link.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        os.symlink(str(release_path), temporary)
        os.replace(temporary, active_link)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def stage_task_evaluation_control_plane_release(
    *,
    source_repo: str | Path,
    source_commit: str,
    release_root: str | Path,
    state_root: str | Path,
    active_link: str | Path,
    activate: bool = False,
) -> dict[str, Any]:
    """Create (or prove) one detached source tree and optionally activate it."""

    source = _absolute_path(source_repo, field="source_repo")
    releases = _absolute_path(release_root, field="release_root")
    state = _absolute_path(state_root, field="state_root")
    active = _absolute_path(active_link, field="active_link")
    if releases == source or releases in source.parents or source in releases.parents:
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_root_overlaps_source_repo"
        )
    if active == source or active in source.parents:
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_active_link_overlaps_source_repo"
        )
    if state == source or state in source.parents or state == releases or state in releases.parents:
        raise ControlPlaneReleaseError(
            "task_evaluation_control_plane_release_state_root_overlaps_checkout"
        )

    commit, origin_main = _assert_source(source, source_commit)
    release_path = releases / commit
    created = _create_release_checkout(
        source_repo=source, release_path=release_path, source_commit=commit
    )
    release_path = release_path.resolve()
    git_index = _install_release_git_index_readability(
        source_repo=source, release_path=release_path
    )
    stage_path = state / commit / "stage.json"
    if stage_path.exists():
        stage_receipt = _read_exact_receipt(stage_path)
        if (
            stage_receipt.get("schema_version") != SCHEMA_VERSION
            or stage_receipt.get("status") != "staged"
            or stage_receipt.get("source_commit") != commit
            or stage_receipt.get("source_repo") != str(source)
            or stage_receipt.get("release_path") != str(release_path)
        ):
            raise ControlPlaneReleaseError(
                "task_evaluation_control_plane_release_stage_receipt_conflict"
            )
    else:
        stage_receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "staged",
            "source_commit": commit,
            "source_repo": str(source),
            "origin_main_at_stage": origin_main,
            "source_commit_is_ancestor_of_origin_main": True,
            "release_path": str(release_path),
            "release_checkout_clean": True,
            "provider_mutation_performed": False,
            "raw_secret_values_recorded": False,
        }
        stage_receipt["receipt_digest"] = _canonical_digest(
            stage_receipt, digest_field="receipt_digest"
        )
        _write_exact(stage_path, stage_receipt)

    if activate:
        _activate_release(active_link=active, release_path=release_path)
        activation_receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "activated",
            "source_commit": commit,
            "release_path": str(release_path),
            "active_link": str(active),
            "active_link_target": str(release_path),
            "provider_mutation_performed": False,
            "raw_secret_values_recorded": False,
        }
        activation_receipt["receipt_digest"] = _canonical_digest(
            activation_receipt, digest_field="receipt_digest"
        )
        _write_exact(state / commit / "activation.json", activation_receipt)
    result = dict(stage_receipt)
    result["created_release_checkout"] = created
    result["release_git_index"] = git_index
    result["activated"] = activate
    result["active_link"] = str(active)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--active-link", required=True)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = stage_task_evaluation_control_plane_release(
            source_repo=args.source_repo,
            source_commit=args.source_commit,
            release_root=args.release_root,
            state_root=args.state_root,
            active_link=args.active_link,
            activate=args.activate,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                    "raw_secret_values_recorded": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
