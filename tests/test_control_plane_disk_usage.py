# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_disk_usage.py
from __future__ import annotations

import os

from blueprint_pipeline import control_plane_disk_usage as usage_module
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


def test_tree_walk_stops_at_entry_budget_without_materializing_directory(
    tmp_path, monkeypatch,
):
    root = tmp_path / "work"
    root.mkdir()
    for index in range(30):
        (root / f"{index}.bin").write_bytes(b"x")
    monkeypatch.setattr(usage_module, "MAX_TREE_SCAN_ENTRIES", 3, raising=False)
    real_scandir = os.scandir
    seen = []

    class CountingScandir:
        def __init__(self, path):
            self.iterator = real_scandir(path)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.iterator.close()

        def __iter__(self):
            return self

        def __next__(self):
            entry = next(self.iterator)
            seen.append(entry.name)
            return entry

    monkeypatch.setattr(usage_module.os, "scandir", CountingScandir)
    usage = tree_usage(root)
    assert usage.unreadable == 1
    assert len(seen) <= 4


def test_many_distinct_hardlinks_mark_measurement_incomplete(tmp_path, monkeypatch):
    root = tmp_path / "work"
    root.mkdir()
    other_links = tmp_path / "other-links"
    other_links.mkdir()
    for index in range(4):
        path = root / f"{index}.bin"
        path.write_bytes(b"x")
        os.link(path, other_links / path.name)
    monkeypatch.setattr(usage_module, "MAX_TRACKED_SHARED_INODES", 2, raising=False)
    usage = tree_usage(root)
    assert usage.unreadable == 1
    assert usage.shared_inodes == 2
