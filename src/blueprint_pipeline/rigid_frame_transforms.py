"""Small, simulator-independent transforms between world and robot frames.

Robot solvers consume targets in the articulation-root frame.  Site and task
contracts, on the other hand, describe geometry in the registered world frame.
Using a world-named vector directly as a base-frame target happens to work only
when the robot is at the origin with identity rotation.  This module makes the
join explicit and testable before a simulator is started.

Quaternions use the Isaac Lab ``(x, y, z, w)`` convention throughout.
"""

from __future__ import annotations

import math
from typing import Sequence


class RigidFrameTransformError(ValueError):
    """A stable frame-contract error."""


def _vector(value: Sequence[float], *, length: int, label: str) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise RigidFrameTransformError(f"{label}_invalid") from exc
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise RigidFrameTransformError(f"{label}_invalid")
    return result


def normalize_quaternion_xyzw(value: Sequence[float]) -> list[float]:
    quaternion = _vector(value, length=4, label="frame_quaternion")
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm <= 1.0e-12:
        raise RigidFrameTransformError("frame_quaternion_invalid")
    return [item / norm for item in quaternion]


def quaternion_conjugate_xyzw(value: Sequence[float]) -> list[float]:
    x, y, z, w = normalize_quaternion_xyzw(value)
    return [-x, -y, -z, w]


def quaternion_multiply_xyzw(
    left: Sequence[float], right: Sequence[float]
) -> list[float]:
    lx, ly, lz, lw = normalize_quaternion_xyzw(left)
    rx, ry, rz, rw = normalize_quaternion_xyzw(right)
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def rotate_vector_xyzw(
    quaternion_xyzw: Sequence[float], vector: Sequence[float]
) -> list[float]:
    x, y, z, w = normalize_quaternion_xyzw(quaternion_xyzw)
    vx, vy, vz = _vector(vector, length=3, label="frame_vector")
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def position_world_to_base(
    *,
    position_world_m: Sequence[float],
    base_position_world_m: Sequence[float],
    base_quaternion_world_xyzw: Sequence[float],
) -> list[float]:
    """Express a registered-world point in the robot articulation-root frame."""

    position = _vector(position_world_m, length=3, label="world_position")
    base = _vector(base_position_world_m, length=3, label="base_position")
    delta = [position[index] - base[index] for index in range(3)]
    return rotate_vector_xyzw(
        quaternion_conjugate_xyzw(base_quaternion_world_xyzw), delta
    )


def position_base_to_world(
    *,
    position_base_m: Sequence[float],
    base_position_world_m: Sequence[float],
    base_quaternion_world_xyzw: Sequence[float],
) -> list[float]:
    """Express a robot-root point in the registered world frame."""

    base = _vector(base_position_world_m, length=3, label="base_position")
    rotated = rotate_vector_xyzw(base_quaternion_world_xyzw, position_base_m)
    return [base[index] + rotated[index] for index in range(3)]


def vector_world_to_base(
    *,
    vector_world: Sequence[float],
    base_quaternion_world_xyzw: Sequence[float],
) -> list[float]:
    """Rotate a direction into the robot root without applying translation."""

    return rotate_vector_xyzw(
        quaternion_conjugate_xyzw(base_quaternion_world_xyzw), vector_world
    )


def rigid_offset_in_body_frame(
    *,
    body_position_world: Sequence[float],
    body_quaternion_world_xyzw: Sequence[float],
    child_position_world: Sequence[float],
    child_quaternion_world_xyzw: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Express a child's world pose as a constant offset in its body frame."""

    body = _vector(body_position_world, length=3, label="body_position")
    child = _vector(child_position_world, length=3, label="child_position")
    body_inverse = quaternion_conjugate_xyzw(body_quaternion_world_xyzw)
    delta = [child[index] - body[index] for index in range(3)]
    return (
        rotate_vector_xyzw(body_inverse, delta),
        quaternion_multiply_xyzw(
            body_inverse, child_quaternion_world_xyzw
        ),
    )


def apply_rigid_offset(
    *,
    body_position_world: Sequence[float],
    body_quaternion_world_xyzw: Sequence[float],
    offset_position_body: Sequence[float],
    offset_quaternion_body_xyzw: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Rebuild a child's world pose from a live body and rigid offset."""

    body = _vector(body_position_world, length=3, label="body_position")
    rotated = rotate_vector_xyzw(
        body_quaternion_world_xyzw, offset_position_body
    )
    return (
        [body[index] + rotated[index] for index in range(3)],
        quaternion_multiply_xyzw(
            body_quaternion_world_xyzw, offset_quaternion_body_xyzw
        ),
    )


__all__ = [
    "RigidFrameTransformError",
    "apply_rigid_offset",
    "normalize_quaternion_xyzw",
    "position_base_to_world",
    "position_world_to_base",
    "quaternion_conjugate_xyzw",
    "quaternion_multiply_xyzw",
    "rigid_offset_in_body_frame",
    "rotate_vector_xyzw",
    "vector_world_to_base",
]
