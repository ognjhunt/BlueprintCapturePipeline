from __future__ import annotations

import math

import pytest

from blueprint_pipeline.articulated_task_cameras import (
    ArticulatedTaskCameraError,
    CAMERA_RESOLUTION_SCHEMA_VERSION,
    resolve_articulated_task_cameras,
)


HINGE = [1.617248144, 1.829218141, 1.2859256235]
HANDLE_MID = [2.0937, 1.8068, 1.0225]
STATES = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]
BASE = [1.75, 1.99]
DOOR_AABB = {
    "aabb_min": [1.617248144, 1.7, 0.939981249],
    "aabb_max": [2.331180256, 1.86, 1.631869998],
}


def _resolve(**overrides):
    arguments = {
        "hinge_origin_world_m": list(HINGE),
        "handle_closed_midpoint_world_m": list(HANDLE_MID),
        "door_state_angles_degrees": list(STATES),
        "franka_base_xy_world_m": list(BASE),
        "task_door_closed_aabb_m": dict(DOOR_AABB),
        "obstacles": [],
        "external_camera_candidates": [
            {"camera_id": "external_front_left", "position_world_m": [0.95, 2.55, 1.45]},
            {"camera_id": "external_front", "position_world_m": [1.55, 2.95, 1.45]},
        ],
        "overview_camera_candidates": [
            {"camera_id": "overview_wide", "position_world_m": [3.05, 3.3, 1.9]},
        ],
        "image_width": 320,
        "image_height": 180,
        "vertical_fov_degrees": 55.0,
        "frame_margin_fraction": 0.05,
        "minimum_handle_pixels": 24,
    }
    arguments.update(overrides)
    return resolve_articulated_task_cameras(**arguments)


def test_resolution_selects_policy_and_review_cameras() -> None:
    receipt = _resolve()

    assert receipt["schema_version"] == CAMERA_RESOLUTION_SCHEMA_VERSION
    assert receipt["status"] == "cameras_locally_resolved"
    assert receipt["external_camera"]["camera_id"] in {
        "external_front_left",
        "external_front",
    }
    assert receipt["external_camera"]["role"] == "policy_input"
    assert receipt["wrist_camera"]["role"] == "policy_input"
    assert receipt["overview_camera"]["role"] == "review_only"
    assert receipt["overview_camera"]["policy_input"] is False
    assert receipt["overview_camera"]["scores_episode"] is False
    for row in receipt["external_camera"]["per_state_visibility"]:
        assert row["handle_pixels"] >= 24
        assert row["handle_inside_frame_margin"] is True
        assert row["moving_door_visible"] is True
    assert len(receipt["external_camera"]["per_state_visibility"]) == len(STATES)
    assert receipt["claim_boundary"]["analytic_projection_not_rendered_frames"] is True
    assert receipt["claim_boundary"]["native_render_verification_required"] is True
    assert receipt["receipt_digest"].startswith("sha256:")


def test_camera_behind_the_door_plane_is_rejected() -> None:
    receipt = _resolve(
        external_camera_candidates=[
            {"camera_id": "external_front_left", "position_world_m": [0.95, 2.55, 1.45]},
            {"camera_id": "external_behind", "position_world_m": [2.0, 0.4, 1.45]},
        ]
    )

    rejected = {row["camera_id"]: row for row in receipt["rejected_candidates"]}
    assert "external_behind" in rejected
    assert any(
        "handle_behind_door_face" in reason
        for reason in rejected["external_behind"]["reasons"]
    )


def test_occluded_camera_is_rejected() -> None:
    receipt = _resolve(
        external_camera_candidates=[
            {"camera_id": "external_front_left", "position_world_m": [0.95, 2.55, 1.45]},
        ],
        obstacles=[
            {
                "obstacle_id": "column",
                "world_aabb_min_m": [1.30, 2.05, 0.0],
                "world_aabb_max_m": [1.60, 2.35, 2.2],
            }
        ],
    )

    assert receipt["status"] == "articulated_task_cameras_unresolved"
    rejected = {row["camera_id"]: row for row in receipt["rejected_candidates"]}
    assert any(
        "handle_occluded" in reason for reason in rejected["external_front_left"]["reasons"]
    )


def test_no_admissible_external_camera_is_typed_abstention() -> None:
    receipt = _resolve(
        external_camera_candidates=[
            {"camera_id": "external_too_far", "position_world_m": [40.0, 40.0, 1.45]},
        ]
    )

    assert receipt["status"] == "articulated_task_cameras_unresolved"
    assert receipt["external_camera"] is None
    assert receipt["blockers"] == ["articulated_task_external_camera_unresolved"]


def test_wrist_camera_tracks_the_moving_handle() -> None:
    receipt = _resolve()

    rows = receipt["wrist_camera"]["per_state_visibility"]
    assert len(rows) == len(STATES)
    assert all(row["handle_pixels"] >= 24 for row in rows)
    targets = [row["handle_world_m"] for row in rows]
    assert targets[0] != targets[-1]
    radii = [
        math.hypot(target[0] - HINGE[0], target[1] - HINGE[1]) for target in targets
    ]
    assert max(radii) - min(radii) < 1e-6


def test_invalid_margin_fails_closed() -> None:
    with pytest.raises(ArticulatedTaskCameraError) as excinfo:
        _resolve(frame_margin_fraction=0.9)

    assert any("frame_margin_invalid" in error for error in excinfo.value.errors)
