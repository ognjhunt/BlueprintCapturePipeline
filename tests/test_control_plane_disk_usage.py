# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_disk_usage.py
from __future__ import annotations

import os

from blueprint_pipeline.control_plane_disk_usage import tree_usage


def _allocated(path):
    metadata = os.lstat(path)
    return getattr(metadata, "st_blocks", 0) * 512 or metadata.st_size


def test_hardlinked_file_counts_once(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    (root / "a.bin").write_bytes(b"x" * 100_000)
    os.link(root / "a.bin", root / "b.bin")
    usage = tree_usage(root)
    assert usage.unique_inodes == 2  # the directory and one file inode
    assert usage.files == 2  # two names
    assert usage.apparent_bytes == 100_000 + os.lstat(root).st_size
    assert usage.allocated_bytes == _allocated(root / "a.bin") + _allocated(root)
    assert usage.shared_inodes == 1


def test_missing_path_is_zero_and_symlinks_are_not_followed(tmp_path):
    assert tree_usage(tmp_path / "absent").allocated_bytes == 0
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"y" * 50_000)
    root = tmp_path / "work"
    root.mkdir()
    (root / "link").symlink_to(outside)
    usage = tree_usage(root)
    assert usage.apparent_bytes < 50_000


def test_unreadable_entries_are_counted_not_raised(tmp_path, monkeypatch):
    root = tmp_path / "work"
    (root / "sub").mkdir(parents=True)
    real_scandir = os.scandir

    def failing_scandir(path):
        if str(path).endswith("sub"):
            raise PermissionError("denied")
        return real_scandir(path)

    monkeypatch.setattr("blueprint_pipeline.control_plane_disk_usage.os.scandir", failing_scandir)
    assert tree_usage(root).unreadable == 1
