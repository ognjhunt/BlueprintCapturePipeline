from __future__ import annotations

import math

import pytest

from blueprint_pipeline.native_pose_transforms import (
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
