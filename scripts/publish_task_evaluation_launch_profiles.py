#!/usr/bin/env python3
"""Validate and immutably publish Pipeline launch profiles plus WebApp catalog.

Profiles contain no secret values. They bind canonical secret profile IDs,
allocator arguments, source/spec digests, spend/TTL ceilings, terminal evidence,
reconciliation providers, and WebApp sync policy. The generated WebApp catalog
contains only the public descriptor needed to select that exact profile.
"""

from __future__ import annotations

import argparse
import grp
import hashlib
import json
import os
import pwd
import stat
import subprocess  # nosec B404 - fixed runuser/sha256sum argv over validated paths
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.host_resident_launch_inputs import PRODUCTION_LAUNCH_INPUT_ROOTS
from blueprint_pipeline.task_evaluation_launch_catalog import build_catalog_payload
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    TaskEvaluationLaunchError,
    validate_launch_profile,
    verify_profile_immutable_inputs,
)

DEFAULT_SERVICE_ACCOUNT = "blueprint"
DEFAULT_SERVICE_GROUP = "blueprint"
RUNUSER_PATH = "/usr/sbin/runuser"
SHA256SUM_PATH = "/usr/bin/sha256sum"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TaskEvaluationLaunchError(f"profile_json_object_required:{path}")
    return dict(value)


def _write_exact(path: Path, payload: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
        return True
    except FileExistsError:
        if path.read_bytes() != payload:
            raise TaskEvaluationLaunchError(f"immutable_profile_conflict:{path.name}")
        return False


def _under_production_root(path: Path) -> bool:
    value = str(path)
    return any(
        value == root or value.startswith(root.rstrip("/") + "/")
        for root in PRODUCTION_LAUNCH_INPUT_ROOTS
    )


def _service_identity(
    target_root: Path,
    service_account: str | None,
    service_group: str | None,
) -> tuple[str, str, int, int]:
    """Resolve the consumer account without weakening production defaults.

    Library callers publish into temporary directories in tests and local
    rehearsals, where no ``blueprint`` account exists. Production roots always
    resolve the real service account and fail closed when the host was not
    provisioned; an omitted account outside those roots means the caller
    itself is the consumer.
    """

    account = service_account
    if account is None:
        account = (
            DEFAULT_SERVICE_ACCOUNT
            if _under_production_root(target_root)
            else pwd.getpwuid(os.geteuid()).pw_name
        )
    try:
        entry = pwd.getpwnam(account)
    except KeyError as exc:
        raise TaskEvaluationLaunchError(
            f"launch_profile_service_account_missing:{account}"
        ) from exc
    group = service_group
    if group is None:
        group = (
            DEFAULT_SERVICE_GROUP
            if _under_production_root(target_root)
            else grp.getgrgid(entry.pw_gid).gr_name
        )
    try:
        group_entry = grp.getgrnam(group)
    except KeyError as exc:
        raise TaskEvaluationLaunchError(
            f"launch_profile_service_group_missing:{group}"
        ) from exc
    if entry.pw_gid != group_entry.gr_gid and account not in group_entry.gr_mem:
        raise TaskEvaluationLaunchError(
            f"launch_profile_service_account_group_mismatch:{account}:{group}"
        )
    return account, group, entry.pw_uid, group_entry.gr_gid


def _production_root_for(path: Path) -> Path | None:
    resolved = path.resolve()
    for value in PRODUCTION_LAUNCH_INPUT_ROOTS:
        root = Path(value).resolve()
        if resolved == root or root in resolved.parents:
            return root
    return None


def _install_parent_traversal(path: Path, *, boundary: Path, gid: int, name: str) -> None:
    """Grant group traversal on exact ancestors without touching sibling bytes."""

    directories: list[Path] = []
    current = path.parent
    while current != boundary:
        if boundary not in current.parents:
            raise TaskEvaluationLaunchError(
                f"launch_profile_immutable_input_outside_control_plane:{name}"
            )
        directories.append(current)
        current = current.parent
    for directory in reversed(directories):
        try:
            if directory.is_symlink() or not directory.is_dir():
                raise OSError(f"unsafe input parent: {directory}")
            mode = stat.S_IMODE(directory.stat().st_mode)
            os.chown(directory, -1, gid)
            directory.chmod(mode | stat.S_IXGRP)
        except OSError as exc:
            raise TaskEvaluationLaunchError(
                f"launch_profile_immutable_input_parent_permission_install_failed:{name}"
            ) from exc


def _install_service_directory(path: Path, *, gid: int) -> None:
    """Seal one named directory; never recursively re-own existing contents."""

    try:
        if path.is_symlink() or not path.is_dir():
            raise OSError(f"unsafe service directory: {path}")
        os.chown(path, -1, gid)
        path.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
    except OSError as exc:
        raise TaskEvaluationLaunchError(
            "launch_profile_directory_permission_install_failed"
        ) from exc


def _seal_published_profile(
    path: Path, *, expected_digest: str, account: str, uid: int, gid: int
) -> None:
    """Make the final profile service-readable and prove its exact bytes reopen."""

    try:
        os.chown(path, -1, gid)
        path.chmod(stat.S_IRUSR | stat.S_IRGRP)
    except OSError as exc:
        raise TaskEvaluationLaunchError(
            f"launch_profile_permission_install_failed:{path.name}"
        ) from exc
    observed = _digest_as_account(path, account=account, uid=uid)
    metadata = path.stat()
    if (
        observed != expected_digest
        or metadata.st_gid != gid
        or stat.S_IMODE(metadata.st_mode) != stat.S_IRUSR | stat.S_IRGRP
    ):
        raise TaskEvaluationLaunchError(
            f"launch_profile_consumer_unreadable:{path.name}"
        )


def _digest_as_account(path: Path, *, account: str, uid: int) -> str:
    if os.geteuid() == uid:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if os.geteuid() != 0:
        return ""
    try:
        completed = subprocess.run(  # nosec B603 - fixed executable and argv
            [RUNUSER_PATH, "-u", account, "--", SHA256SUM_PATH, str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0 or not completed.stdout.strip():
        return ""
    return "sha256:" + completed.stdout.split()[0]


def _seal_immutable_input_permissions(
    profile: Mapping[str, Any],
    *,
    target_root: Path,
    account: str,
    uid: int,
    gid: int,
) -> None:
    """Make exact profile inputs read-only and prove the service can read them."""

    inputs: dict[Path, tuple[str, str]] = {}
    for item in profile.get("immutable_inputs") or []:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "invalid")
        path = Path(str(item.get("path") or "")).expanduser()
        if path.is_symlink() or not path.is_file():
            raise TaskEvaluationLaunchError(f"launch_profile_immutable_input_missing:{name}")
        resolved = path.resolve()
        if _under_production_root(target_root) and not _under_production_root(resolved):
            raise TaskEvaluationLaunchError(
                f"launch_profile_immutable_input_outside_control_plane:{name}"
            )
        inputs[resolved] = (name, str(item.get("digest") or ""))

    for path, (name, expected_digest) in inputs.items():
        boundary = _production_root_for(path)
        if _under_production_root(target_root):
            if boundary is None:
                raise TaskEvaluationLaunchError(
                    f"launch_profile_immutable_input_outside_control_plane:{name}"
                )
            _install_parent_traversal(path, boundary=boundary, gid=gid, name=name)
        try:
            os.chown(path, -1, gid)
            path.chmod(stat.S_IRUSR | stat.S_IRGRP)
        except OSError as exc:
            raise TaskEvaluationLaunchError(
                f"launch_profile_immutable_input_permission_install_failed:{name}"
            ) from exc
        observed = _digest_as_account(path, account=account, uid=uid)
        mode = stat.S_IMODE(path.stat().st_mode)
        if (
            observed != expected_digest
            or path.stat().st_gid != gid
            or mode != stat.S_IRUSR | stat.S_IRGRP
        ):
            raise TaskEvaluationLaunchError(
                f"launch_profile_immutable_input_consumer_unreadable:{name}"
            )


def publish_profiles(
    *,
    profile_paths: Sequence[str | Path],
    profile_dir: str | Path,
    webapp_catalog_out: str | Path,
    service_account: str | None = None,
    service_group: str | None = None,
) -> dict[str, Any]:
    published: list[dict[str, Any]] = []
    target_root = Path(profile_dir).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    account, _group, uid, gid = _service_identity(
        target_root, service_account, service_group
    )
    _install_service_directory(target_root, gid=gid)
    for source_value in profile_paths:
        source_input = Path(source_value).expanduser()
        if source_input.is_symlink():
            raise TaskEvaluationLaunchError(f"launch_profile_source_invalid:{source_input}")
        source = source_input.resolve()
        if not source.is_file():
            raise TaskEvaluationLaunchError(f"launch_profile_source_invalid:{source}")
        profile = _read(source)
        blockers = validate_launch_profile(profile)
        blockers.extend(verify_profile_immutable_inputs(profile))
        if blockers:
            raise TaskEvaluationLaunchError(",".join(sorted(set(blockers))))
        _seal_immutable_input_permissions(
            profile,
            target_root=target_root,
            account=account,
            uid=uid,
            gid=gid,
        )
        # Permission installation is metadata-only. Reopen every input after
        # it so a partial or racy handoff cannot publish changed bytes.
        blockers = verify_profile_immutable_inputs(profile)
        if blockers:
            raise TaskEvaluationLaunchError(",".join(sorted(set(blockers))))
        payload = (json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n").encode()
        target = target_root / f"{profile['profile_id']}.json"
        created = _write_exact(target, payload)
        _seal_published_profile(
            target,
            expected_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
            account=account,
            uid=uid,
            gid=gid,
        )
        published.append(
            {
                "profile_id": profile["profile_id"],
                "profile_digest": profile["profile_digest"],
                "path": str(target),
                "created": created,
            }
        )

    # The catalog is what the WebApp reads to resolve a profile_id, so it must
    # describe every published profile -- not just the ones named in this
    # invocation. Building it from the arguments meant a profile could be
    # published and still be invisible: a launch against it was rejected at
    # lookup, and the profile directory looked correct while the catalog listed
    # a single stale entry. Enumerating the directory makes the catalog a
    # function of published state instead of of a command line.
    #
    # The projection is shared with the start-up reconciler so the artifact
    # written here and the one repaired there cannot disagree about what the
    # catalog *is* -- fixing this writer did nothing for the catalogs already on
    # disk, which is how a stale one stayed in service after the fix deployed.
    catalog_payload = build_catalog_payload(target_root)
    catalog_path = Path(webapp_catalog_out).expanduser().resolve()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_bytes(catalog_payload)
    return {
        "schema_version": "task_evaluation_launch_profile_publication.v1",
        "status": "published",
        "profiles": published,
        "webapp_catalog_path": str(catalog_path),
        "webapp_catalog_contains_allocator_arguments": False,
        "webapp_catalog_contains_secret_values": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--webapp-catalog-out", required=True)
    parser.add_argument(
        "--service-account",
        help=(
            "Account that must read immutable profile inputs. Defaults to blueprint "
            "for production roots and the invoking account elsewhere."
        ),
    )
    parser.add_argument(
        "--service-group",
        help=(
            "Group that receives exact read/traverse access. Defaults to blueprint "
            "for production roots and the account's primary group elsewhere."
        ),
    )
    args = parser.parse_args(argv)
    try:
        result = publish_profiles(
            profile_paths=args.profile,
            profile_dir=args.profile_dir,
            webapp_catalog_out=args.webapp_catalog_out,
            service_account=args.service_account,
            service_group=args.service_group,
        )
    except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "task_evaluation_launch_profile_publication.v1",
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "blockers": [str(exc)],
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
