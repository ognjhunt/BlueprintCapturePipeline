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


def _link_immutable_file(source: str, destination: str) -> str:
    os.link(source, destination, follow_symlinks=False)
    return destination


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
    """Install complete staged bytes manifest-last under an exclusive directory.

    ``Path.mkdir(exist_ok=False)`` is the no-overwrite reservation.  Installing the
    manifest last keeps concurrent readers fail-closed until every inventoried
    byte is present.  Both directories must be siblings so staging cleanup and
    publication remain governed by the same parent.
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
        children = sorted(
            staged.iterdir(),
            key=lambda path: (path.name == manifest_name, path.name),
        )
        for source in children:
            installed = target / source.name
            if source.is_dir():
                shutil.copytree(
                    source,
                    installed,
                    symlinks=False,
                    copy_function=_link_immutable_file,
                )
            else:
                _link_immutable_file(str(source), str(installed))
        _remove_owned_tree(staged)
        target.chmod(0o555)
        _fsync_directory(target)
        _fsync_directory(target.parent)
    except Exception:
        _remove_owned_tree(target)
        raise


__all__ = [
    "ImmutableDirectoryPublicationError",
    "publish_staged_immutable_directory",
]
