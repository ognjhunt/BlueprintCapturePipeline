# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_legacy_owner.py
"""Historical owner attribution must bind an actual unchanged directory tree."""

import os

import pytest


def test_generation_snapshot_detects_same_size_payload_rewrite(tmp_path):
    from blueprint_pipeline.control_plane_lane_legacy_owner import snapshot_generation

    root = tmp_path / "work"
    target = root / "lanes" / "diagnostics" / "old-1"
    target.mkdir(parents=True)
    payload = target / "one.log"
    payload.write_bytes(b"alpha")
    first = snapshot_generation(target, allowed_roots=(root,))
    assert first["path"] == str(target)
    assert first["target"]["type"] == "directory"
    assert first["lane_root"]["path"] == str(root / "lanes")
    assert first["lane"]["path"] == str(root / "lanes" / "diagnostics")
    assert first["lane"]["identity"]["ino"] == os.stat(root / "lanes" / "diagnostics").st_ino
    assert first["tree"]["entries"] == 2
    payload.write_bytes(b"bravo")
    second = snapshot_generation(target, allowed_roots=(root,))
    assert second["tree"]["digest"] != first["tree"]["digest"]


def test_generation_snapshot_refuses_linked_child_and_bounded_scan(tmp_path):
    from blueprint_pipeline.control_plane_lane_legacy_owner import (
        LegacyOwnerError, snapshot_generation,
    )

    root = tmp_path / "work"
    target = root / "old-1"
    target.mkdir(parents=True)
    (target / "one.log").write_bytes(b"one")
    with pytest.raises(LegacyOwnerError, match="legacy_target_measurement_incomplete"):
        snapshot_generation(target, allowed_roots=(root,), max_entries=1)
    (target / "linked").symlink_to("one.log")
    with pytest.raises(LegacyOwnerError, match="legacy_target_measurement_incomplete"):
        snapshot_generation(target, allowed_roots=(root,))


def test_generation_snapshot_bounds_aggregate_names_and_encoded_paths(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_lane_legacy_owner as owner

    root = tmp_path / "work"
    target = root / "old-1"
    target.mkdir(parents=True)
    nested = target
    for index in range(4):
        nested = nested / ("n" * 55 + str(index))
        nested.mkdir()
    (nested / "one.log").write_bytes(b"one")
    monkeypatch.setattr(owner, "MAX_GENERATION_BYTES", 128)
    with pytest.raises(owner.LegacyOwnerError, match="legacy_target_measurement_incomplete"):
        owner.snapshot_generation(target, allowed_roots=(root,))


def test_generation_snapshot_refuses_root_and_target_alias(tmp_path):
    from blueprint_pipeline.control_plane_lane_legacy_owner import (
        LegacyOwnerError, snapshot_generation,
    )

    root = tmp_path / "work"
    target = root / "old-1"
    target.mkdir(parents=True)
    linked = tmp_path / "alias"
    linked.symlink_to(root, target_is_directory=True)
    with pytest.raises(LegacyOwnerError, match="legacy_target_unsafe"):
        snapshot_generation(linked / "old-1", allowed_roots=(linked,))
    with pytest.raises(LegacyOwnerError, match="legacy_target_unsafe"):
        snapshot_generation(root, allowed_roots=(root,))


def test_generation_snapshot_refuses_replaced_name_during_open(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_lane_legacy_owner as owner

    root = tmp_path / "work"
    target = root / "old-1"
    target.mkdir(parents=True)
    replacement = root / "other"
    replacement.mkdir()
    original_open = os.open
    moved = False

    def substitute(path, flags, *args, **kwargs):
        nonlocal moved
        if path == "old-1" and not moved:
            moved = True
            os.rename(target, root / "saved")
            os.rename(replacement, target)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(owner.os, "open", substitute)
    with pytest.raises(owner.LegacyOwnerError, match="legacy_target_changed"):
        owner.snapshot_generation(target, allowed_roots=(root,))
