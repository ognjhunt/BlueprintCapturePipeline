"""ADP-009D/day28: observed allocations are neither exclusive nor reclaimable."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_owner_target_measurement.py

import os
import stat

import pytest

from blueprint_pipeline.control_plane_disk_usage import allocated_bytes
from tests.test_owner_target_version_descriptors import enrolled, files


def measured(path, expected, monkeypatch=None):
    from blueprint_pipeline.control_plane_lane_owner_target_measurement import _measure_target_allocated
    owner = files(expected)
    fd = owner.open(path, os.O_RDONLY | os.O_DIRECTORY, target=True)
    try:
        return _measure_target_allocated(owner, fd, expected["folder_identity"], owner.budget)
    finally:
        owner.finish()


def test_sparse_hardlinked_and_directory_metadata_count_once(enrolled, monkeypatch):
    path, expected = enrolled
    payload = path / "payload"
    with payload.open("wb") as stream:
        stream.truncate(1024 * 1024)
    os.link(payload, path / "alias")
    child = path / "child"
    child.mkdir()
    (child / "tiny").write_bytes(b"small")
    unique = {}
    for item in (path, payload, path / "alias", path / ".lane-scratch.v1.json", child, child / "tiny"):
        info = item.stat()
        unique[(info.st_dev, info.st_ino)] = allocated_bytes(info)
    monkeypatch.setattr(os, "read", lambda *a, **kw: pytest.fail("payload content read"))
    result = measured(path, expected)
    assert result["measurement_complete"] is True
    assert result["measured_allocated_bytes"] == sum(unique.values())
    assert result["allocation_scope"] == "names_within_one_target_not_exclusive_physical_ownership"
    assert result["candidate_bytes"] is result["eta_seconds"] is None
    assert result["mutations"] == 0


@pytest.mark.parametrize("unsupported", ["symlink", "fifo", "cross_device"])
def test_unsupported_payload_cannot_be_false_complete_zero(enrolled, monkeypatch, unsupported):
    path, expected = enrolled
    item = path / "item"
    if unsupported == "symlink":
        item.symlink_to("elsewhere")
    elif unsupported == "fifo":
        os.mkfifo(item)
    else:
        item.write_bytes(b"x")
        original = os.stat
        def statted(name, *args, **kw):
            info = original(name, *args, **kw)
            if name == "item":
                values = list(info)
                values[stat.ST_DEV] += 1
                return os.stat_result(values)
            return info
        monkeypatch.setattr(os, "stat", statted)
    result = measured(path, expected)
    assert result["measurement_complete"] is False
    assert result["measured_allocated_bytes"] is None
    assert result["kept_reasons"] == ["target_measurement_incomplete"]


def test_same_inode_change_between_directory_passes_refuses(enrolled, monkeypatch):
    path, expected = enrolled
    item = path / "item"
    item.write_bytes(b"x")
    original = os.stat
    count = 0
    def statted(name, *args, **kw):
        nonlocal count
        if name == "item":
            count += 1
            if count == 2:
                item.write_bytes(b"changed")
        return original(name, *args, **kw)
    monkeypatch.setattr(os, "stat", statted)
    result = measured(path, expected)
    assert result["measurement_complete"] is False and result["measured_allocated_bytes"] is None


def test_entry_bound_is_checked_before_retention(enrolled, monkeypatch):
    from blueprint_pipeline import control_plane_lane_owner_target_measurement as m
    path, expected = enrolled
    (path / "item").write_bytes(b"x")
    monkeypatch.setattr(m, "MAX_MEASUREMENT_ENTRIES", 1)
    result = measured(path, expected)
    assert result["measurement_complete"] is False and result["measured_allocated_bytes"] is None


def test_other_target_bytes_are_not_aggregated(enrolled):
    path, expected = enrolled
    before = measured(path, expected)
    unrelated = path.parent / "other"
    unrelated.mkdir()
    (unrelated / "payload").write_bytes(b"other")
    assert measured(path, expected) == before
