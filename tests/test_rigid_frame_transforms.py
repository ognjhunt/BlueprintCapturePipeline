from __future__ import annotations

import math

import pytest

from blueprint_pipeline.rigid_frame_transforms import (
    RigidFrameTransformError,
    position_base_to_world,
    position_world_to_base,
    vector_world_to_base,
)


def _yaw_quaternion(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def test_world_base_round_trip_preserves_a_registered_point() -> None:
    base = [1.75, 1.99, 0.0]
    quaternion = _yaw_quaternion(-math.pi / 2.0)
    world = [2.0937, 1.8068, 1.0225]

    local = position_world_to_base(
        position_world_m=world,
        base_position_world_m=base,
        base_quaternion_world_xyzw=quaternion,
    )
    restored = position_base_to_world(
        position_base_m=local,
        base_position_world_m=base,
        base_quaternion_world_xyzw=quaternion,
    )

    assert restored == pytest.approx(world)
    assert local == pytest.approx([0.1832, 0.3437, 1.0225], abs=1e-6)


def test_vector_transform_never_applies_base_translation() -> None:
    quaternion = _yaw_quaternion(-math.pi / 2.0)

    assert vector_world_to_base(
        vector_world=[0.0, -1.0, 0.0],
        base_quaternion_world_xyzw=quaternion,
    ) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_a_degenerate_base_quaternion_fails_closed() -> None:
    with pytest.raises(RigidFrameTransformError, match="frame_quaternion_invalid"):
        position_world_to_base(
            position_world_m=[0.0, 0.0, 0.0],
            base_position_world_m=[0.0, 0.0, 0.0],
            base_quaternion_world_xyzw=[0.0, 0.0, 0.0, 0.0],
        )
