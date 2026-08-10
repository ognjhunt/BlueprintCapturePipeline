from __future__ import annotations

import math
import json
from pathlib import Path

import pytest

from blueprint_pipeline.franka_base_orientation import (
    FrankaBaseOrientationError,
    resolve_franka_base_orientation,
)


def _target(phase_id: str, x: float, y: float, z: float = 1.0) -> dict:
    return {
        "phase_id": phase_id,
        "target_position_world_m": [x, y, z],
    }


def test_original_rigid_fixture_resolves_its_geometry_not_a_scene_constant() -> None:
    receipt = resolve_franka_base_orientation(
        base_position_world_m=[3.4681748, -2.8100837, 0.2766791],
        phase_targets=[_target("can_start", 3.4681748, -3.3100837)],
        source_receipt_digest="sha256:" + "a" * 64,
    )

    assert receipt["status"] == "resolved_candidate"
    assert receipt["resolved_yaw_world_rad"] == pytest.approx(-math.pi / 2.0)
    assert receipt["maximum_observed_deviation_rad"] == pytest.approx(0.0)
    assert receipt["claim_boundary"]["native_ik_qualified"] is False


def test_articulated_fixture_centers_the_full_door_sweep() -> None:
    base = [1.75, 1.99, 0.0]
    closed = [2.0937, 1.8068, 1.0225]
    middle = [2.0515, 2.0293, 1.0225]
    open_state = [1.908894, 2.206646, 1.0225]

    receipt = resolve_franka_base_orientation(
        base_position_world_m=base,
        phase_targets=[
            _target("door_0deg", *closed),
            _target("door_27deg", *middle),
            _target("door_55deg", *open_state),
        ],
        source_receipt_digest="sha256:" + "b" * 64,
    )

    endpoint_bearings = [
        math.atan2(closed[1] - base[1], closed[0] - base[0]),
        math.atan2(open_state[1] - base[1], open_state[0] - base[0]),
    ]
    assert receipt["status"] == "resolved_candidate"
    assert receipt["resolved_yaw_world_rad"] == pytest.approx(
        sum(endpoint_bearings) / 2.0
    )
    assert receipt["maximum_observed_deviation_rad"] < math.pi / 4.0


def test_phase_span_outside_allowed_front_arc_is_typed_blocker() -> None:
    receipt = resolve_franka_base_orientation(
        base_position_world_m=[0.0, 0.0, 0.0],
        phase_targets=[
            _target("front", 1.0, 0.0),
            _target("back", -1.0, 0.0),
        ],
        source_receipt_digest="sha256:" + "c" * 64,
        maximum_allowed_deviation_rad=1.0,
    )

    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == [
        "franka_base_orientation_phase_span_exceeds_limit"
    ]


def test_duplicate_or_coincident_targets_fail_before_gpu() -> None:
    with pytest.raises(FrankaBaseOrientationError) as excinfo:
        resolve_franka_base_orientation(
            base_position_world_m=[0.0, 0.0, 0.0],
            phase_targets=[
                _target("grasp", 0.0, 0.0),
                _target("grasp", 1.0, 0.0),
            ],
            source_receipt_digest="not-a-digest",
        )

    assert excinfo.value.errors == (
        "franka_base_orientation_phase_id_invalid:1",
        "franka_base_orientation_phase_target_at_base:grasp",
        "franka_base_orientation_source_receipt_digest_invalid",
    )


def test_checked_second_scene_orientation_is_regenerated_from_camera_receipt() -> None:
    manifest_root = (
        Path(__file__).parents[1] / "docs/arm_decision_proof_v1/manifests"
    )
    camera = json.loads(
        (manifest_root / "second_scene_840796_task_camera_resolution.v1.json").read_text()
    )
    expected = json.loads(
        (
            manifest_root
            / "second_scene_840796_franka_base_orientation.v1.json"
        ).read_text()
    )
    targets = [
        {
            "phase_id": f"door_{int(row['angle_degrees']):02d}deg",
            "target_position_world_m": row["handle_world_m"],
        }
        for row in camera["external_camera"]["per_state_visibility"]
    ]

    actual = resolve_franka_base_orientation(
        base_position_world_m=[*camera["franka_base_xy_world_m"], 0.0],
        phase_targets=targets,
        source_receipt_digest=camera["receipt_digest"],
    )

    assert actual == expected
