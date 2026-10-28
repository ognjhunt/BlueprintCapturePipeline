"""Usage observation must not follow paths replaced after partition selection."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import control_plane_disk_usage as usage


def _survey(root: Path, previous: dict | None = None) -> dict:
    return usage.survey_usage([str(root)], aliases={}, prefixes=(str(root),), classify=lambda _path: None,
        statvfs=lambda _path: SimpleNamespace(f_blocks=10000, f_bfree=5000, f_frsize=4096),
        max_entries=40, max_seconds=10, mountinfo=str(root.parent / "absent-mountinfo"),
        continuation=previous, partitions_per_mount=1)


def _tree(base: Path) -> tuple[Path, Path, Path]:
    root, outside = base / "work-volume", base / "outside"
    selected = root / "a"
    selected.mkdir(parents=True)
    outside.mkdir()
    (selected / "data").write_bytes(b"x" * 4096)
    (outside / "outside-payload").write_bytes(b"z" * 1048576)
    return root, selected, outside


def _unknown_without_outside_bytes(result: dict) -> None:
    assert result["mounts"][0]["surveyed_bytes"] < 1048576
    assert result["inventory_complete"] is False
    assert result["unreadable_bytes"] is None
    assert result["status"] == "truncated" and result["retirement_authority"] is False
    assert "partition_identity_refused" in result["continuation_status"]


@pytest.mark.parametrize("phase", ("before_open", "before_scandir", "after_scandir"))
def test_selected_directory_swap_is_refused_at_each_use_boundary(tmp_path, monkeypatch, phase):
    root, selected, outside = _tree(tmp_path.resolve())
    first = _survey(root)
    selected_info, outside_info = os.lstat(selected), os.lstat(outside)
    real_open, real_scandir = os.open, os.scandir
    swapped = []

    def swap():
        if not swapped:
            selected.rename(root.parent / "moved-a")
            selected.symlink_to(outside, target_is_directory=True)
            swapped.append(True)

    def opening(path, flags, *args, **kwargs):
        if phase == "before_open" and path == "a" and flags & os.O_DIRECTORY:
            swap()
        return real_open(path, flags, *args, **kwargs)

    @contextmanager
    def scanning(fd):
        info = os.fstat(fd)
        assert (info.st_dev, info.st_ino) != (outside_info.st_dev, outside_info.st_ino)
        target = (info.st_dev, info.st_ino) == (selected_info.st_dev, selected_info.st_ino)
        if target and phase == "before_scandir":
            swap()
        with real_scandir(fd) as iterator:
            yield iterator
        if target and phase == "after_scandir":
            swap()

    monkeypatch.setattr(os, "open", opening)
    monkeypatch.setattr(os, "scandir", scanning)
    result = _survey(root, first["continuation"])
    assert swapped
    _unknown_without_outside_bytes(result)
    assert result["path_identity_refusals"] == 1


def test_nested_directory_swap_is_refused_without_following_cached_stat(tmp_path, monkeypatch):
    root, selected, outside = _tree(tmp_path.resolve())
    nested = selected / "nested"
    nested.mkdir()
    (nested / "data").write_bytes(b"x" * 4096)
    first = _survey(root)
    real_open = os.open
    swapped = []

    def opening(path, flags, *args, **kwargs):
        if path == "nested" and flags & os.O_DIRECTORY and not swapped:
            nested.rename(root.parent / "moved-nested")
            nested.symlink_to(outside, target_is_directory=True)
            swapped.append(True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", opening)
    result = _survey(root, first["continuation"])
    assert swapped
    _unknown_without_outside_bytes(result)


def test_root_swap_during_partition_inventory_discards_unverified_candidates(tmp_path, monkeypatch):
    root, _selected, outside = _tree(tmp_path.resolve())
    first = _survey(root)
    root_info, outside_info = os.lstat(root), os.lstat(outside)
    real_scandir = os.scandir
    swapped = []

    def scanning(fd):
        opened = os.fstat(fd)
        assert (opened.st_dev, opened.st_ino) != (outside_info.st_dev, outside_info.st_ino)
        if (opened.st_dev, opened.st_ino) == (root_info.st_dev, root_info.st_ino) and not swapped:
            root.rename(root.parent / "moved-root")
            root.symlink_to(outside, target_is_directory=True)
            swapped.append(True)
        return real_scandir(fd)

    monkeypatch.setattr(os, "scandir", scanning)
    result = _survey(root, first["continuation"])
    assert swapped and result["selected_partitions"] == 0
    _unknown_without_outside_bytes(result)
    assert result["mounts"][0]["surveyed_bytes"] == 0


def test_real_directory_replacement_with_same_device_refuses_selected_identity(tmp_path, monkeypatch):
    root, selected, outside = _tree(tmp_path.resolve())
    first = _survey(root)
    real_walk = usage._UsageWalk.walk
    swapped = []

    def walking(walker, path, metadata):
        if Path(path) == selected and not swapped:
            selected.rename(root.parent / "moved-a")
            outside.rename(selected)
            swapped.append(True)
        return real_walk(walker, path, metadata)

    monkeypatch.setattr(usage._UsageWalk, "walk", walking)
    result = _survey(root, first["continuation"])
    assert swapped
    _unknown_without_outside_bytes(result)
    assert result["mounts"][0]["surveyed_bytes"] == 0
