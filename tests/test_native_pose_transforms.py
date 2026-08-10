from __future__ import annotations

import math

import pytest

from blueprint_pipeline.native_pose_transforms import (
    NativePoseTransformError,
    body_local_point_midpoint_geometry,
    pose_world_to_base,
    world_to_base_rotation_row_major_xyzw,
)


def test_world_pose_is_expressed_in_a_rotated_robot_root() -> None:
    half = math.sqrt(0.5)

    position, quaternion = pose_world_to_base(
        position_world=[1.0, 2.0, 0.0],
        quaternion_world_xyzw=[0.0, 0.0, half, half],
        base_position_world=[1.0, 1.0, 0.0],
        base_quaternion_world_xyzw=[0.0, 0.0, half, half],
    )

    assert position == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-12)
    assert quaternion == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1.0e-12)


def test_world_jacobian_rows_rotate_into_root_axes() -> None:
    half = math.sqrt(0.5)

    rotation = world_to_base_rotation_row_major_xyzw(
        [0.0, 0.0, half, half]
    )

    assert rotation == pytest.approx(
        [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0], abs=1.0e-12
    )


def test_authored_tool_points_not_link_origins_measure_gripper_aperture() -> None:
    half = math.sqrt(0.5)
    grasp_frame = {
        "kind": "body_local_point_midpoint",
        "body_names": ["left_inner_finger", "right_inner_finger"],
        "body_local_points_m": {
            "left_inner_finger": [0.0, 0.0, 0.046],
            "right_inner_finger": [0.0, 0.0, 0.046],
        },
    }
    # Open: link origins are close, but the authored pad points face outward.
    opened = body_local_point_midpoint_geometry(
        grasp_frame=grasp_frame,
        body_poses_world={
            "left_inner_finger": [-0.01, 0.0, 0.0, 0.0, -half, 0.0, half],
            "right_inner_finger": [0.01, 0.0, 0.0, 0.0, half, 0.0, half],
        },
    )
    # Closed: link origins move apart while the authored pad points meet.
    closed = body_local_point_midpoint_geometry(
        grasp_frame=grasp_frame,
        body_poses_world={
            "left_inner_finger": [-0.046, 0.0, 0.0, 0.0, half, 0.0, half],
            "right_inner_finger": [0.046, 0.0, 0.0, 0.0, -half, 0.0, half],
        },
    )

    assert opened["separation_m"] == pytest.approx(0.112, abs=1.0e-12)
    assert closed["separation_m"] == pytest.approx(0.0, abs=1.0e-12)
    assert opened["midpoint_world_m"] == pytest.approx([0.0, 0.0, 0.0])


def test_raw_body_midpoint_contract_is_rejected_for_articulated_grippers() -> None:
    with pytest.raises(NativePoseTransformError, match="native_grasp_frame_kind_invalid"):
        body_local_point_midpoint_geometry(
            grasp_frame={
                "kind": "body_midpoint",
                "body_names": ["left", "right"],
            },
            body_poses_world={},
        )
