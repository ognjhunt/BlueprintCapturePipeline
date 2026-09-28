from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_scene_reach_probe as probe
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    packet = tmp_path / "packet"
    packet.mkdir()
    request = {
        "scene_id": "captured-scene",
        "task_id": "book-to-mark",
        "task_spec": {
            "task_kind": "rigid_pick_place",
            "subject_asset_id": "book",
            "start_pose_world": [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            "target_position_world_m": [0.2, 0.0, 0.3],
            "subject_collision_bounds_scoring_frame_m": {
                "minimum": [-0.1, -0.1, -0.01],
                "maximum": [0.1, 0.1, 0.01],
            },
        },
    }
    (packet / "native_task_arena_packet_request.v1.json").write_text(json.dumps(request))
    receipt = {
        "receipt_digest": "sha256:" + "a" * 64,
        "arena_scene_plan_digest": "sha256:" + "b" * 64,
        "source_bindings": [
            {
                "semantic_role": "scene_collision",
                "staged_relative_path": "assets/collision.usda",
            }
        ],
    }
    monkeypatch.setattr(
        probe,
        "verify_native_task_arena_packet",
        lambda _: (packet, receipt, []),
    )

    class EmptyIndex:
        def __init__(self, *, usd_path: str) -> None:
            assert usd_path.endswith("assets/collision.usda")

        def obstacle_boxes(self) -> list:
            return []

    monkeypatch.setattr(probe, "UsdSceneSpatialIndex", EmptyIndex)
    return packet


def test_static_probe_reports_body_motion_when_vertical_gap_exceeds_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _packet(tmp_path, monkeypatch)
    report = probe.probe_g1_scene_static_reach(
        source_packet_dir=packet,
        floor_z_m=0.0,
        radius_m=0.7,
        grid_step_m=0.1,
    )
    assert report["sampled_nominal_stance_count"] > 0
    assert report["interpretation"] == "body_motion_or_new_stance_required"
    assert report["minimum_shoulder_lowering_if_horizontally_aligned_m"] == 0.33
    assert report["pick"]["within_nominal_arm_span"] is False
    assert report["place"]["within_nominal_arm_span"] is False
    assert report["source_packet_receipt_digest"] == "sha256:" + "a" * 64
    assert report["report_digest"] == canonical_digest(report, digest_field="report_digest")


def test_probe_rejects_rotated_task_bounds_instead_of_measuring_wrong_box(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _packet(tmp_path, monkeypatch)
    path = packet / "native_task_arena_packet_request.v1.json"
    request = json.loads(path.read_text())
    request["task_spec"]["start_pose_world"][3:] = [0.0, 0.0, 0.70710678, 0.70710678]
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="g1_reach_probe_task_unsupported"):
        probe.probe_g1_scene_static_reach(source_packet_dir=packet, floor_z_m=0.0)
