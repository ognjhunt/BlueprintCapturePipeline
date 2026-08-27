"""Ownership, mode, and service-account readback for imported CAD evidence."""

from __future__ import annotations

import os
import stat
from collections.abc import Sequence
from pathlib import Path


HOST_FILE_MODE = 0o640
HOST_DIRECTORY_MODE = 0o750


class SimReadyCadAgentHostImportError(ValueError):
    """One source binding or imported receipt failed closed."""


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _ownership_policy(owner_uid: int, owner_gid: int) -> tuple[int, int]:
    if (
        not isinstance(owner_uid, int)
        or isinstance(owner_uid, bool)
        or owner_uid < 0
        or not isinstance(owner_gid, int)
        or isinstance(owner_gid, bool)
        or owner_gid < 0
    ):
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_ownership_policy_invalid"
        )
    return owner_uid, owner_gid


def _directories_for_files(files: Sequence[Path], roots: Sequence[Path]) -> set[Path]:
    directories = {root for root in roots}
    for path in files:
        for root in roots:
            if _is_inside(path, root):
                cursor = path.parent
                while _is_inside(cursor, root):
                    directories.add(cursor)
                    if cursor == root:
                        break
                    cursor = cursor.parent
                break
    return directories


def _seal_ownership_and_readback(
    *,
    files: Sequence[Path],
    roots: Sequence[Path],
    owner_uid: int,
    owner_gid: int,
) -> tuple[int, int]:
    owner_uid, owner_gid = _ownership_policy(owner_uid, owner_gid)
    unique_files = sorted(set(files), key=str)
    directories = sorted(_directories_for_files(unique_files, roots), key=str)
    for directory in directories:
        if directory.is_symlink() or not directory.is_dir():
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_ownership_directory_invalid"
            )
        metadata = directory.stat()
        if metadata.st_uid != owner_uid or metadata.st_gid != owner_gid:
            try:
                os.chown(directory, owner_uid, owner_gid)
            except PermissionError as exc:
                raise SimReadyCadAgentHostImportError(
                    "cad_host_import_ownership_chown_failed"
                ) from exc
        if stat.S_IMODE(metadata.st_mode) != HOST_DIRECTORY_MODE:
            try:
                os.chmod(directory, HOST_DIRECTORY_MODE)
            except PermissionError as exc:
                raise SimReadyCadAgentHostImportError(
                    "cad_host_import_ownership_chmod_failed"
                ) from exc
        metadata = directory.stat()
        if (
            metadata.st_uid != owner_uid
            or metadata.st_gid != owner_gid
            or stat.S_IMODE(metadata.st_mode) != HOST_DIRECTORY_MODE
        ):
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_ownership_readback_failed"
            )
    for path in unique_files:
        if path.is_symlink() or not path.is_file():
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_ownership_file_invalid"
            )
        metadata = path.stat()
        if metadata.st_uid != owner_uid or metadata.st_gid != owner_gid:
            try:
                os.chown(path, owner_uid, owner_gid)
            except PermissionError as exc:
                raise SimReadyCadAgentHostImportError(
                    "cad_host_import_ownership_chown_failed"
                ) from exc
        if stat.S_IMODE(metadata.st_mode) != HOST_FILE_MODE:
            try:
                os.chmod(path, HOST_FILE_MODE)
            except PermissionError as exc:
                raise SimReadyCadAgentHostImportError(
                    "cad_host_import_ownership_chmod_failed"
                ) from exc
        metadata = path.stat()
        if (
            metadata.st_uid != owner_uid
            or metadata.st_gid != owner_gid
            or stat.S_IMODE(metadata.st_mode) != HOST_FILE_MODE
        ):
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_ownership_readback_failed"
            )
    if os.geteuid() == owner_uid:
        try:
            for path in unique_files:
                with path.open("rb") as stream:
                    stream.read(1)
        except OSError as exc:
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_service_account_readback_failed"
            ) from exc
    elif os.geteuid() == 0 and hasattr(os, "fork"):
        child = os.fork()
        if child == 0:  # pragma: no cover - exercised only by root host install
            try:
                os.setgroups([owner_gid])
                os.setgid(owner_gid)
                os.setuid(owner_uid)
                for path in unique_files:
                    with path.open("rb") as stream:
                        stream.read(1)
            except Exception:
                os._exit(1)
            os._exit(0)
        _pid, status = os.waitpid(child, 0)
        if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_service_account_readback_failed"
            )
    else:
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_service_account_readback_unavailable"
        )
    return len(unique_files), len(directories)


__all__ = [
    "HOST_DIRECTORY_MODE",
    "HOST_FILE_MODE",
    "SimReadyCadAgentHostImportError",
    "_directories_for_files",
    "_is_inside",
    "_ownership_policy",
    "_seal_ownership_and_readback",
]
