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


def _facing(x: float, y: float, z: float, toward: tuple[float, float]) -> dict:
    import math

    half = math.atan2(toward[1] - y, toward[0] - x) / 2.0
    return {"position_world_m": [x, y, z], "orientation_xyzw": [0.0, 0.0, math.sin(half), math.cos(half)]}


def test_the_closest_stance_whose_nominal_arm_span_covers_pick_and_place_is_chosen() -> None:
    # Pelvis at 0.79 m, shoulders 0.29 m above it: 1.08 m. A pick on a 0.9 m table, a place beside it.
    start, target = [0.0, 0.0, 0.9], [0.0, 0.25, 0.9]
    far = _facing(1.2, 0.1, 0.79, (0.0, 0.0))
    near = _facing(0.3, 0.1, 0.79, (0.0, 0.0))
    index, record = probe.g1_stance_static_reach([far, near, far], start=start, target=target)
    assert index == 1 and record["chosen_rank"] == 1 and record["within_nominal_arm_span"] is True
    assert record["rule"] == probe.STANCE_RULE and record["clear_stances_measured"] == 3
    assert max(record["pick_nominal_shoulder_distance_m"], record["place_nominal_shoulder_distance_m"]) <= (
        record["nominal_arm_span_m"])
    assert record["interpretation"] == "nominal_static_reach_candidate"
    assert record["claim_ceiling"] == "development_only_static_geometry"
    # None in span: the closest stays, flagged, never refused.
    index, record = probe.g1_stance_static_reach([far, far], start=start, target=target)
    assert (index, record["chosen_rank"], record["within_nominal_arm_span"]) == (0, 0, False)
    assert record["interpretation"] == "body_motion_or_new_stance_required"
    assert record["pick_nominal_shoulder_distance_m"] > record["nominal_arm_span_m"]
    for stances, bad_target in (([], target), ([near], [0.0, 0.25]), ([near], [0.0, float("nan"), 0.9])):
        with pytest.raises(ValueError, match="g1_stance_reach_inputs_invalid"):
            probe.g1_stance_static_reach(stances, start=start, target=bad_target)


def test_the_stance_is_chosen_for_the_task_hand_not_whichever_shoulder_is_nearer() -> None:
    # The review's case: the place is on the robot's left of a stance facing the pick, so only the left arm
    # reaches it from there; the right hand carries, so a stance beside both points is chosen instead.
    start, target = [0.0, 0.0, 0.9], [0.0, -0.4, 0.9]
    left_only = _facing(0.3, 0.0, 0.79, (0.0, 0.0))
    beside = _facing(0.0, -0.35, 0.79, (0.0, 0.0))
    index, nearer = probe.g1_stance_static_reach([left_only, beside], start=start, target=target)
    assert (index, nearer["shoulder"]) == (0, "nearer")
    index, right = probe.g1_stance_static_reach([left_only, beside], start=start, target=target, task_hand="right")
    assert (index, right["shoulder"], right["within_nominal_arm_span"]) == (1, "right", True)
    assert max(right["pick_nominal_shoulder_distance_m"], right["place_nominal_shoulder_distance_m"]) <= 0.45
    with pytest.raises(ValueError, match="g1_stance_reach_inputs_invalid"):
        probe.g1_stance_static_reach([beside], start=start, target=target, task_hand="both")
