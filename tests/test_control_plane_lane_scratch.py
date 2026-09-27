"""Lane scratch leases name their owner, expiry and cleanup without moving data."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_scratch.py

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _create(root: Path, **overrides):
    from blueprint_pipeline.control_plane_lane_scratch import create_lane_scratch

    options = dict(root=root, lane="agent-a", name="job-1", owner="owner-a",
                   run_ref="run-1", reason="diagnostic", class_intent="scratch",
                   cleanup="owner_review", ttl_seconds=3600, now=lambda: 1000.0)
    options.update(overrides)
    return create_lane_scratch(**options)


def test_lane_scratch_is_published_with_a_sealed_lease(tmp_path: Path) -> None:
    root = tmp_path / "lanes"
    root.mkdir()

    folder = _create(root)

    assert folder == root / "agent-a" / "job-1"
    lease = json.loads((folder / ".lane-scratch.v1.json").read_text())
    assert lease["schema_version"] == "control_plane_lane_scratch.v1"
    assert {key: lease[key] for key in ("lane", "owner", "run_ref", "reason", "class_intent", "cleanup")} == {
        "lane": "agent-a", "owner": "owner-a", "run_ref": "run-1", "reason": "diagnostic",
        "class_intent": "scratch", "cleanup": "owner_review",
    }
    assert (lease["created_at_epoch"], lease["expires_at_epoch"]) == (1000.0, 4600.0)
    assert lease["lease_digest"] == canonical_digest(lease, digest_field="lease_digest")


def test_cache_lease_has_a_budget_and_scene_reference(tmp_path: Path) -> None:
    root = tmp_path / "lanes"
    root.mkdir()

    folder = _create(root, run_ref=None, scene_ref="scene-1", class_intent="cache",
                     size_budget_bytes=8192)

    lease = json.loads((folder / ".lane-scratch.v1.json").read_text())
    assert lease["scene_ref"] == "scene-1" and "run_ref" not in lease
    assert lease["size_budget_bytes"] == 8192


@pytest.mark.parametrize("overrides", [
    {"owner": ""}, {"run_ref": None}, {"lane": "../escape"}, {"name": ".."},
    {"cleanup": "erase_anything"}, {"ttl_seconds": 0}, {"ttl_seconds": 14 * 86400 + 1},
    {"class_intent": "cache"}, {"reason": "/secret/path"},
])
def test_invalid_lane_scratch_request_leaves_no_folder(tmp_path: Path, overrides) -> None:
    from blueprint_pipeline.control_plane_lane_scratch import LaneScratchError

    root = tmp_path / "lanes"
    root.mkdir()
    with pytest.raises(LaneScratchError):
        _create(root, **overrides)
    assert not (root / "agent-a" / "job-1").exists()


def test_duplicate_and_symlink_lane_parent_are_refused(tmp_path: Path) -> None:
    from blueprint_pipeline.control_plane_lane_scratch import LaneScratchError

    root = tmp_path / "lanes"
    root.mkdir()
    existing = _create(root)
    with pytest.raises(LaneScratchError):
        _create(root)
    assert existing.is_dir()
    (root / "agent-b").symlink_to(root / "agent-a", target_is_directory=True)
    with pytest.raises(LaneScratchError):
        _create(root, lane="agent-b", name="job-2")
    assert not (root / "agent-a" / "job-2").exists()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(LaneScratchError):
        _create(linked_root, lane="agent-c", name="job-3")


def test_torn_publication_leaves_no_final_folder(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import control_plane_lane_scratch as scratch

    root = tmp_path / "lanes"
    root.mkdir()
    monkeypatch.setattr(scratch.os, "rename", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("rename failed")))
    with pytest.raises(scratch.LaneScratchError):
        _create(root)
    assert not (root / "agent-a" / "job-1").exists()


def test_renew_and_release_reject_a_stale_digest_without_deleting_data(tmp_path: Path) -> None:
    from blueprint_pipeline.control_plane_lane_scratch import (
        LaneScratchError, release_lane_scratch, renew_lane_scratch,
    )

    root = tmp_path / "lanes"
    root.mkdir()
    folder = _create(root)
    (folder / "payload.bin").write_bytes(b"retain")
    first = json.loads((folder / ".lane-scratch.v1.json").read_text())

    renewed = renew_lane_scratch(root=root, lane="agent-a", name="job-1", owner="owner-a",
                                 expected_digest=first["lease_digest"], ttl_seconds=7200,
                                 now=lambda: 1200.0)
    assert renewed["expires_at_epoch"] == 8400.0
    with pytest.raises(LaneScratchError):
        release_lane_scratch(root=root, lane="agent-a", name="job-1", owner="owner-a",
                             expected_digest=first["lease_digest"], now=lambda: 1300.0)
    released = release_lane_scratch(root=root, lane="agent-a", name="job-1", owner="owner-a",
                                    expected_digest=renewed["lease_digest"], now=lambda: 1300.0)
    assert released["released_at_epoch"] == 1300.0
    assert (folder / "payload.bin").read_bytes() == b"retain"


def test_tampered_or_symlinked_lease_cannot_be_renewed(tmp_path: Path) -> None:
    from blueprint_pipeline.control_plane_lane_scratch import (
        LaneScratchError, renew_lane_scratch,
    )

    root = tmp_path / "lanes"
    root.mkdir()
    folder = _create(root)
    lease_path = folder / ".lane-scratch.v1.json"
    lease = json.loads(lease_path.read_text())
    lease["cleanup"] = "delete"
    lease_path.write_text(json.dumps(lease))
    with pytest.raises(LaneScratchError):
        renew_lane_scratch(root=root, lane="agent-a", name="job-1", owner="owner-a",
                           expected_digest=lease["lease_digest"], ttl_seconds=3600)
    lease_path.unlink()
    target = tmp_path / "outside.json"
    target.write_text(json.dumps(lease))
    lease_path.symlink_to(target)
    with pytest.raises(LaneScratchError):
        renew_lane_scratch(root=root, lane="agent-a", name="job-1", owner="owner-a",
                           expected_digest=lease["lease_digest"], ttl_seconds=3600)
    assert target.read_text() == json.dumps(lease)


def test_list_lane_scratch_is_paginated_and_skips_unsafe_folders(tmp_path: Path) -> None:
    from blueprint_pipeline.control_plane_lane_scratch import list_lane_scratch

    root = tmp_path / "lanes"
    root.mkdir()
    _create(root, name="a")
    _create(root, name="b")
    _create(root, name="c")
    (root / "agent-a" / "bad").symlink_to(tmp_path)

    first = list_lane_scratch(root=root, lane="agent-a", limit=2, offset=0)
    assert [row["name"] for row in first["leases"]] == ["a", "b"]
    assert first["next_offset"] == 2
    second = list_lane_scratch(root=root, lane="agent-a", limit=2, offset=2)
    assert [row["name"] for row in second["leases"]] == ["c"]
    assert second["next_offset"] is None


def test_door_command_writes_bounded_receipts_and_release_keeps_payload(tmp_path: Path) -> None:
    from blueprint_pipeline.control_plane_lane_scratch_door import main

    root = tmp_path / "lanes"
    root.mkdir()
    folder = _create(root)
    (folder / "payload.bin").write_bytes(b"keep")
    lease = json.loads((folder / ".lane-scratch.v1.json").read_text())
    result = tmp_path / "result.json"
    base = ["--root", str(root), "--lane", "agent-a", "--result-out", str(result)]
    assert main(["ls", *base, "--limit", "1", "--offset", "0"]) == 0
    assert json.loads(result.read_text())["leases"][0]["lease_digest"] == lease["lease_digest"]
    assert main(["release", *base, "--name", "job-1", "--owner", "owner-a",
                 "--expected-digest", lease["lease_digest"]]) == 0
    assert json.loads(result.read_text())["status"] == "released"
    assert (folder / "payload.bin").read_bytes() == b"keep"
