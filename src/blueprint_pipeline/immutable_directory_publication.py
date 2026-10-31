"""Publish a staged immutable directory without replacing an existing target."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


class ImmutableDirectoryPublicationError(ValueError):
    """A staged directory could not be installed under an exclusive name."""


def _remove_owned_tree(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o700)
    root.chmod(0o700)
    shutil.rmtree(root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_staged_immutable_directory(
    *,
    staging: str | Path,
    destination: str | Path,
    manifest_name: str,
    output_exists_code: str,
) -> None:
    """Install complete staged bytes under an exclusive directory, all at once.

    ``Path.mkdir(exist_ok=False)`` is the no-overwrite reservation, and the rename
    replaces only that empty directory, so a reader sees nothing or everything. Both
    directories must be siblings so the rename stays on one filesystem. No file's link
    count or change time moves, where linking each one and unlinking its staged alias
    took 77,000 link calls to publish one 38,505 file package (2026-10-31).
    """

    staged = Path(staging).resolve()
    requested_target = Path(destination).expanduser().absolute()
    target = requested_target.parent.resolve() / requested_target.name
    manifest = staged / manifest_name
    if (
        not staged.is_dir()
        or staged.is_symlink()
        or staged.parent != target.parent
        or not manifest.is_file()
        or manifest.is_symlink()
        or any(path.is_symlink() for path in staged.rglob("*"))
        or not output_exists_code
    ):
        raise ImmutableDirectoryPublicationError(
            "immutable_directory_publication_input_invalid"
        )
    try:
        target.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise ImmutableDirectoryPublicationError(output_exists_code) from exc
    try:
        staged.chmod(0o700)  # some systems refuse to rename a directory the owner cannot write
        os.rename(staged, target)
        target.chmod(0o555)
        _fsync_directory(target)
        _fsync_directory(target.parent)
    except BaseException:  # a stop (SystemExit) or interrupt leaves nothing half-built either
        _remove_owned_tree(target)
        raise


__all__ = [
    "ImmutableDirectoryPublicationError",
    "publish_staged_immutable_directory",
]
