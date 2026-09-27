"""Measure disk usage the way the disk sees it: once per inode, in allocated blocks.

Hardlinks are how the control plane shares bytes between content stores, compiled
episodes and launch sets, so a byte count per name overstates usage (on 2026-09-26 two
cache roots "held" about 470 GB on a 165 GB disk). Every measurement here counts each
``(st_dev, st_ino)`` once and reports allocated bytes (``st_blocks * 512``).
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

MAX_TREE_SCAN_ENTRIES = 100_000
MAX_TRACKED_SHARED_INODES = 50_000


@dataclass(frozen=True)
class TreeUsage:
    allocated_bytes: int = 0
    apparent_bytes: int = 0
    files: int = 0
    directories: int = 0
    unique_inodes: int = 0
    shared_inodes: int = 0
    unreadable: int = 0


def allocated_bytes(metadata: os.stat_result) -> int:
    """Allocated bytes, or the apparent size when the filesystem reports no blocks.

    APFS reports zero blocks for directories and for data stored inline, so zero
    would under-count; the apparent size is the conservative stand-in.
    """

    blocks = getattr(metadata, "st_blocks", None)
    if isinstance(blocks, int) and blocks > 0:
        return blocks * 512
    return int(metadata.st_size)


def tree_usage(path: str | Path) -> TreeUsage:
    """Unique-inode usage of ``path`` (a file or a directory tree); never follows symlinks."""

    root = Path(path)
    try:
        top = os.lstat(root)
    except FileNotFoundError:
        return TreeUsage()
    except OSError:
        return TreeUsage(unreadable=1)
    # Only multiply-linked inodes can be reached twice, so only they are remembered.
    # Directories are never deduplicated: their link count is 2 + subdirectories and
    # says nothing about sharing.
    seen: set[tuple[int, int]] = set()
    totals = {
        "allocated": 0,
        "apparent": 0,
        "files": 0,
        "directories": 0,
        "unique": 0,
        "shared": 0,
        "unreadable": 0,
    }

    def account(metadata: os.stat_result) -> bool:
        if not stat.S_ISDIR(metadata.st_mode) and metadata.st_nlink > 1:
            key = (metadata.st_dev, metadata.st_ino)
            if key in seen:
                return True
            if len(seen) >= MAX_TRACKED_SHARED_INODES:
                return False
            seen.add(key)
            totals["shared"] += 1
        totals["unique"] += 1
        totals["allocated"] += allocated_bytes(metadata)
        totals["apparent"] += int(metadata.st_size)
        return True

    if not account(top):
        return TreeUsage(unreadable=1)
    if not stat.S_ISDIR(top.st_mode):
        totals["files"] += 1
    else:
        totals["directories"] += 1
        pending = [root]
        scanned_entries = 0
        truncated = False
        while pending and not truncated:
            directory = pending.pop()
            try:
                with os.scandir(directory) as iterator:
                    for entry in iterator:
                        scanned_entries += 1
                        if scanned_entries > MAX_TREE_SCAN_ENTRIES:
                            totals["unreadable"] += 1
                            truncated = True
                            break
                        try:
                            metadata = entry.stat(follow_symlinks=False)
                        except OSError:
                            totals["unreadable"] += 1
                            continue
                        if not account(metadata):
                            totals["unreadable"] += 1
                            truncated = True
                            break
                        if stat.S_ISDIR(metadata.st_mode):
                            totals["directories"] += 1
                            pending.append(Path(entry.path))
                        else:
                            totals["files"] += 1
            except OSError:
                totals["unreadable"] += 1
    return TreeUsage(
        allocated_bytes=totals["allocated"],
        apparent_bytes=totals["apparent"],
        files=totals["files"],
        directories=totals["directories"],
        unique_inodes=totals["unique"],
        shared_inodes=totals["shared"],
        unreadable=totals["unreadable"],
    )


__all__ = ["TreeUsage", "allocated_bytes", "tree_usage"]
