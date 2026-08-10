"""Small simulator-independent pose transforms for native robot controllers."""

from __future__ import annotations

import math
from collections.abc import Sequence


class NativePoseTransformError(ValueError):
    """Stable malformed-pose errors."""


def _quaternion(value: Sequence[float]) -> tuple[float, float, float, float]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise NativePoseTransformError("native_pose_quaternion_invalid") from exc
    if len(result) != 4 or not all(math.isfinite(item) for item in result):
        raise NativePoseTransformError("native_pose_quaternion_invalid")
    norm = math.sqrt(sum(item * item for item in result))
    if norm <= 1.0e-12:
        raise NativePoseTransformError("native_pose_quaternion_invalid")
    return tuple(item / norm for item in result)


def _multiply(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = left
    bx, by, bz, bw = right
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _rotate(quaternion: Sequence[float], vector: Sequence[float]) -> list[float]:
    x, y, z, w = quaternion
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def pose_world_to_base(
    *,
    position_world: Sequence[float],
    quaternion_world_xyzw: Sequence[float],
    base_position_world: Sequence[float],
    base_quaternion_world_xyzw: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Express a world pose in the robot root frame expected by native IK."""

    try:
        position = [float(value) for value in position_world]
        base_position = [float(value) for value in base_position_world]
    except (TypeError, ValueError) as exc:
        raise NativePoseTransformError("native_pose_position_invalid") from exc
    if (
        len(position) != 3
        or len(base_position) != 3
        or not all(math.isfinite(value) for value in (*position, *base_position))
    ):
        raise NativePoseTransformError("native_pose_position_invalid")
    quaternion = _quaternion(quaternion_world_xyzw)
    base = _quaternion(base_quaternion_world_xyzw)
    inverse = (-base[0], -base[1], -base[2], base[3])
    delta = [position[index] - base_position[index] for index in range(3)]
    return _rotate(inverse, delta), list(_multiply(inverse, quaternion))


def world_to_base_rotation_row_major_xyzw(
    base_quaternion_world_xyzw: Sequence[float],
) -> list[float]:
    """Return the matrix that expresses world Jacobian rows in robot-root axes."""

    base = _quaternion(base_quaternion_world_xyzw)
    inverse = (-base[0], -base[1], -base[2], base[3])
    columns = [
        _rotate(inverse, basis)
        for basis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ]
    return [columns[column][row] for row in range(3) for column in range(3)]


__all__ = [
    "NativePoseTransformError",
    "pose_world_to_base",
    "world_to_base_rotation_row_major_xyzw",
]
