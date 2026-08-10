"""Small simulator-independent pose transforms for native robot controllers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


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


def body_local_point_midpoint_geometry(
    *,
    grasp_frame: Mapping[str, Any],
    body_poses_world: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Measure an authored two-point grasp frame from native body poses.

    Articulated grippers can move link origins away from each other while their
    contact pads move together.  Consequently, raw body-origin distance is not
    a gripper aperture and the raw origin midpoint is not necessarily the
    semantic tool midpoint.  This contract transforms one authored local point
    per body into world space before measuring either quantity.
    """

    if grasp_frame.get("kind") != "body_local_point_midpoint":
        raise NativePoseTransformError("native_grasp_frame_kind_invalid")
    body_names = grasp_frame.get("body_names")
    local_points = grasp_frame.get("body_local_points_m")
    if (
        not isinstance(body_names, Sequence)
        or isinstance(body_names, (str, bytes))
        or len(body_names) != 2
        or len(set(str(name) for name in body_names)) != 2
        or not isinstance(local_points, Mapping)
    ):
        raise NativePoseTransformError("native_grasp_frame_contract_invalid")

    names = [str(name) for name in body_names]
    world_points: list[list[float]] = []
    normalized_local_points: dict[str, list[float]] = {}
    for name in names:
        try:
            pose = [float(value) for value in body_poses_world[name]]
            local = [float(value) for value in local_points[name]]
        except (KeyError, TypeError, ValueError) as exc:
            raise NativePoseTransformError(
                f"native_grasp_frame_body_or_point_invalid:{name}"
            ) from exc
        if (
            len(pose) < 7
            or len(local) != 3
            or not all(math.isfinite(value) for value in (*pose[:7], *local))
        ):
            raise NativePoseTransformError(
                f"native_grasp_frame_body_or_point_invalid:{name}"
            )
        quaternion = _quaternion(pose[3:7])
        rotated = _rotate(quaternion, local)
        world_points.append(
            [pose[axis] + rotated[axis] for axis in range(3)]
        )
        normalized_local_points[name] = local

    midpoint = [
        (world_points[0][axis] + world_points[1][axis]) / 2.0
        for axis in range(3)
    ]
    separation = math.dist(world_points[0], world_points[1])
    return {
        "kind": "body_local_point_midpoint",
        "body_names": names,
        "body_local_points_m": normalized_local_points,
        "world_points_m": world_points,
        "midpoint_world_m": midpoint,
        "separation_m": separation,
        "measurement_authority": "native_body_poses_plus_authored_local_tool_points",
    }


__all__ = [
    "NativePoseTransformError",
    "body_local_point_midpoint_geometry",
    "pose_world_to_base",
    "world_to_base_rotation_row_major_xyzw",
]
