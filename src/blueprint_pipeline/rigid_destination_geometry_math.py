"""Runtime-safe geometry math shared by destination qualification and compilation."""
from __future__ import annotations

import math
from typing import Any


class DestinationGeometryMathError(RuntimeError):
    """Qualified destination geometry cannot be interpreted."""


def _numeric_bounds(value: Any) -> tuple[list[float], list[float]]:
    try:
        lower = [float(item) for item in value["minimum"]]
        upper = [float(item) for item in value["maximum"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise DestinationGeometryMathError(
            "episode_compiler_destination_geometry_invalid"
        ) from exc
    if (
        len(lower) != 3
        or len(upper) != 3
        or any(low >= high for low, high in zip(lower, upper, strict=True))
        or not all(math.isfinite(item) for item in (*lower, *upper))
    ):
        raise DestinationGeometryMathError(
            "episode_compiler_destination_geometry_invalid"
        )
    return lower, upper


def _rotate_xyzw(vector: list[float], quaternion: list[float]) -> list[float]:
    x, y, z, w = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _multiply_xyzw(left: list[float], right: list[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    result = [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]
    norm = math.sqrt(sum(value * value for value in result))
    return [value / norm for value in result]


def _subject_bounds_in_scoring_frame(
    *, bounds: Any, transform: Any
) -> tuple[list[float], list[float]]:
    lower, upper = _numeric_bounds(bounds)
    try:
        offset = [float(item) for item in transform["position_m"]]
        orientation = [float(item) for item in transform["orientation_xyzw"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise DestinationGeometryMathError(
            "episode_compiler_destination_subject_geometry_invalid"
        ) from exc
    if (
        len(offset) != 3
        or len(orientation) != 4
        or not math.isclose(
            sum(item * item for item in orientation),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise DestinationGeometryMathError(
            "episode_compiler_destination_subject_geometry_invalid"
        )
    inverse = [-orientation[0], -orientation[1], -orientation[2], orientation[3]]
    corners = [
        _rotate_xyzw(
            [x - offset[0], y - offset[1], z - offset[2]], inverse
        )
        for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    ]
    return (
        [min(point[axis] for point in corners) for axis in range(3)],
        [max(point[axis] for point in corners) for axis in range(3)],
    )


