"""Walk a runtime tree from its directory listings, with no path operation per file.

A deploy walks the 38,505 files of the scene-configuration toolchain about ten times.
``Path.rglob`` with ``is_file``, ``stat`` and ``relative_to`` for each cost several times
what the filesystem did, on trees whose bytes nothing had touched.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path


def walk_tree(root: str | Path) -> Iterator[tuple[str, os.DirEntry[str]]]:
    """Every entry under ``root`` as (POSIX path relative to it, entry), a directory before its contents.

    A symlink is reported and never followed. As with ``Path.rglob``, a directory that
    cannot be read lists nothing, so a caller comparing against an inventory still refuses.
    """
    pending = [("", os.fspath(root))]
    while pending:
        prefix, directory = pending.pop()
        try:
            with os.scandir(directory) as listing:
                entries = list(listing)
        except OSError:
            continue
        for entry in entries:
            relative = prefix + entry.name
            yield relative, entry
            if entry.is_dir(follow_symlinks=False):
                pending.append((relative + "/", entry.path))


def path_order(relative: str) -> list[str]:
    """The order ``sorted`` gives ``Path`` objects, which manifests hold their files in."""
    return relative.split("/")
