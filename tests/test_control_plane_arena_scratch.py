# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_arena_scratch.py
"""New Arena attempt folders have a lease; historical attempts remain readable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.control_plane_arena_scratch import (
    ArenaScratchError, prepare_arena_attempt, resolve_arena_attempt,
)


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    inputs = tmp_path / "task-evaluation-inputs"
    lanes = inputs / "lanes"
    lanes.mkdir(parents=True)
    return inputs, lanes


def test_new_attempt_has_sealed_evidence_lease_before_any_payload(tmp_path: Path) -> None:
    inputs, lanes = _roots(tmp_path)
    folder = prepare_arena_attempt(
        "r33", owner="operator-a", run_ref="run-33", ttl_seconds=86400,
        inputs_root=inputs, lane_root=lanes, now=lambda: 1000,
    )
    assert folder == lanes / "arena" / "arena-launch-r33"
    assert sorted(path.name for path in folder.iterdir()) == [".lane-scratch.v1.json"]
    lease = json.loads((folder / ".lane-scratch.v1.json").read_text())
    assert (lease["owner"], lease["run_ref"], lease["class_intent"], lease["cleanup"]) == (
        "operator-a", "run-33", "evidence", "owner_review",
    )
    assert not (inputs / "arena-launch-r33").exists()
    digest = lease["lease_digest"]
    assert prepare_arena_attempt("r33", owner="operator-a", run_ref="run-33",
                                 inputs_root=inputs, lane_root=lanes, now=lambda: 1001) == folder
    assert json.loads((folder / ".lane-scratch.v1.json").read_text())["lease_digest"] == digest
    with pytest.raises(ArenaScratchError, match="arena_scratch_owner_mismatch"):
        prepare_arena_attempt("r33", owner="operator-b", run_ref="run-33",
                              inputs_root=inputs, lane_root=lanes, now=lambda: 1001)
    with pytest.raises(ArenaScratchError, match="arena_scratch_owner_mismatch"):
        prepare_arena_attempt("r33", owner="operator-a", scene_ref="run-33",
                              inputs_root=inputs, lane_root=lanes, now=lambda: 1001)
    with pytest.raises(ArenaScratchError, match="arena_scratch_inactive"):
        resolve_arena_attempt("r33", writable=True, inputs_root=inputs, lane_root=lanes,
                              now=lambda: 87401)


def test_new_attempt_requires_operator_metadata_and_valid_tag(tmp_path: Path) -> None:
    inputs, lanes = _roots(tmp_path)
    with pytest.raises(ArenaScratchError, match="arena_scratch_metadata_required"):
        prepare_arena_attempt("r34", inputs_root=inputs, lane_root=lanes)
    with pytest.raises(ArenaScratchError, match="arena_scratch_tag_invalid"):
        prepare_arena_attempt("../r34", owner="operator-a", run_ref="run-34", ttl_seconds=86400,
                              inputs_root=inputs, lane_root=lanes)
    assert not (lanes / "arena" / "arena-launch-r34").exists()


def test_legacy_attempt_is_readable_but_needs_review_before_writing(tmp_path: Path) -> None:
    inputs, lanes = _roots(tmp_path)
    legacy = inputs / "arena-launch-r20"
    legacy.mkdir()
    with pytest.raises(ArenaScratchError, match="arena_scratch_legacy_unproven"):
        prepare_arena_attempt("r20", inputs_root=inputs, lane_root=lanes)
    (legacy / "arena_packet").write_text("not a packet directory")
    with pytest.raises(ArenaScratchError, match="arena_scratch_legacy_unproven"):
        resolve_arena_attempt("r20", inputs_root=inputs, lane_root=lanes)
    (legacy / "arena_packet").unlink()
    (legacy / "arena_packet").mkdir()
    assert resolve_arena_attempt("r20", inputs_root=inputs, lane_root=lanes) == legacy
    with pytest.raises(ArenaScratchError, match="arena_scratch_legacy_write_requires_review"):
        resolve_arena_attempt("r20", writable=True, inputs_root=inputs, lane_root=lanes)
    with pytest.raises(ArenaScratchError, match="arena_scratch_legacy_write_requires_review"):
        prepare_arena_attempt("r20", inputs_root=inputs, lane_root=lanes)
    assert not (legacy / ".lane-scratch.v1.json").exists()
    assert not (lanes / "arena" / "arena-launch-r20").exists()


def test_new_tag_cannot_claim_legacy_path_by_adding_a_marker(tmp_path: Path) -> None:
    inputs, lanes = _roots(tmp_path)
    fake_legacy = inputs / "arena-launch-r33"
    (fake_legacy / "arena_packet").mkdir(parents=True)
    with pytest.raises(ArenaScratchError, match="arena_scratch_legacy_tag_unrecognized"):
        resolve_arena_attempt("r33", inputs_root=inputs, lane_root=lanes)
    with pytest.raises(ArenaScratchError, match="arena_scratch_legacy_tag_unrecognized"):
        prepare_arena_attempt("r33", inputs_root=inputs, lane_root=lanes)


def test_ambiguous_or_symlink_attempt_refuses(tmp_path: Path) -> None:
    inputs, lanes = _roots(tmp_path)
    created = prepare_arena_attempt("r33", owner="operator-a", scene_ref="scene-33",
                                    ttl_seconds=86400, inputs_root=inputs, lane_root=lanes)
    (inputs / "arena-launch-r33").mkdir()
    with pytest.raises(ArenaScratchError, match="arena_scratch_ambiguous"):
        resolve_arena_attempt("r33", inputs_root=inputs, lane_root=lanes)
    (inputs / "arena-launch-r33").rmdir()
    (inputs / "arena-launch-r33").symlink_to(created)
    with pytest.raises(ArenaScratchError, match="arena_scratch_path_unsafe"):
        resolve_arena_attempt("r33", inputs_root=inputs, lane_root=lanes)


def test_lane_parent_symlink_refuses_before_following_attempt(tmp_path: Path) -> None:
    inputs, lanes = _roots(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (lanes / "arena").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ArenaScratchError, match="arena_scratch_root_unsafe"):
        resolve_arena_attempt("r33", inputs_root=inputs, lane_root=lanes)
