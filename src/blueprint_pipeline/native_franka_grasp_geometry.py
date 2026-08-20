"""Live Robotiq finger-pad geometry and controlled-body to TCP transform."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .native_franka_action_math import grasp_orientation_contact_xyzw
from .rigid_frame_transforms import (
    quaternion_conjugate_xyzw,
    quaternion_multiply_xyzw,
    rotate_vector_xyzw,
)


SCHEMA_VERSION = "native_franka_grasp_geometry.v1"


class NativeFrankaGraspGeometryError(ValueError):
    """The live gripper geometry could not define one physical TCP frame."""


def measure_live_robotiq_grasp_geometry(
    *,
    stage: Any,
    controlled_body_position_world_m: Sequence[float],
    controlled_body_quaternion_world_xyzw: Sequence[float],
) -> dict[str, Any]:
    """Measure pad centers and the rigid body-to-grasp transform from live USD."""

    from pxr import Usd, UsdGeom

    body_position = [float(value) for value in controlled_body_position_world_m]
    body_quaternion = [
        float(value) for value in controlled_body_quaternion_world_xyzw
    ]
    if (
        len(body_position) != 3
        or len(body_quaternion) != 4
        or not all(math.isfinite(value) for value in (*body_position, *body_quaternion))
    ):
        raise NativeFrankaGraspGeometryError(
            "native_franka_grasp_geometry_body_pose_invalid"
        )
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    matches: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    ranges: dict[str, list[list[float]] | None] = {"left": None, "right": None}
    for prim in Usd.PrimRange(
        stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()
    ):
        path = str(prim.GetPath())
        lowered = path.lower()
        if "/robot/" not in lowered:
            continue
        side = next(
            (
                label
                for label in ("left", "right")
                if f"{label}_inner_finger" in lowered
                or f"{label}finger" in lowered
            ),
            None,
        )
        if side is None or not prim.IsA(UsdGeom.Boundable):
            continue
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = [float(value) for value in aligned.GetMin()]
        maximum = [float(value) for value in aligned.GetMax()]
        if not all(math.isfinite(value) for value in (*minimum, *maximum)):
            continue
        matches[side].append(
            {"prim_path": path, "minimum_world_m": minimum, "maximum_world_m": maximum}
        )
        current = ranges[side]
        if current is None:
            ranges[side] = [minimum, maximum]
        else:
            ranges[side] = [
                [min(current[0][axis], minimum[axis]) for axis in range(3)],
                [max(current[1][axis], maximum[axis]) for axis in range(3)],
            ]
    if any(ranges[side] is None for side in ("left", "right")):
        raise NativeFrankaGraspGeometryError(
            "native_franka_grasp_geometry_pad_bounds_missing"
        )
    centers = {
        side: [
            (ranges[side][0][axis] + ranges[side][1][axis]) / 2.0
            for axis in range(3)
        ]
        for side in ("left", "right")
    }
    midpoint = [
        (centers["left"][axis] + centers["right"][axis]) / 2.0
        for axis in range(3)
    ]
    jaw_world = [
        centers["left"][axis] - centers["right"][axis] for axis in range(3)
    ]
    approach_world = [
        midpoint[axis] - body_position[axis] for axis in range(3)
    ]
    separation = math.sqrt(sum(value * value for value in jaw_world))
    tool_offset = math.sqrt(sum(value * value for value in approach_world))
    if not 0.01 <= separation <= 0.12 or not 0.05 <= tool_offset <= 0.30:
        raise NativeFrankaGraspGeometryError(
            "native_franka_grasp_geometry_physical_bounds_invalid"
        )
    grasp_quaternion_world = grasp_orientation_contact_xyzw(
        approach_axis=approach_world, jaw_axis=jaw_world
    )
    body_inverse = quaternion_conjugate_xyzw(body_quaternion)
    body_to_grasp_position = rotate_vector_xyzw(body_inverse, approach_world)
    body_to_grasp_quaternion = quaternion_multiply_xyzw(
        body_inverse, grasp_quaternion_world
    )
    pad_centers_body = {
        side: rotate_vector_xyzw(
            body_inverse,
            [centers[side][axis] - body_position[axis] for axis in range(3)],
        )
        for side in ("left", "right")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_authority": "live_usd_world_bounds_of_robotiq_inner_finger_geometry",
        "matched_boundables": matches,
        "pad_bounds_world_m": ranges,
        "pad_centers_world_m": centers,
        "pad_centers_controlled_body_m": pad_centers_body,
        "grasp_frame_world": {
            "position_world_m": midpoint,
            "orientation_xyzw": grasp_quaternion_world,
        },
        "controlled_body_to_grasp_frame": {
            "position_controlled_body_m": body_to_grasp_position,
            "orientation_xyzw": body_to_grasp_quaternion,
        },
        "pad_separation_m": separation,
        "body_to_grasp_frame_distance_m": tool_offset,
        "passed": True,
        "blockers": [],
    }


def validate_measured_grasp_geometry(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reopen an injected/runtime measurement before a servo trusts it."""

    try:
        transform = value["controlled_body_to_grasp_frame"]
        position = [float(item) for item in transform["position_controlled_body_m"]]
        quaternion = [float(item) for item in transform["orientation_xyzw"]]
        pads = value["pad_centers_controlled_body_m"]
        left = [float(item) for item in pads["left"]]
        right = [float(item) for item in pads["right"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeFrankaGraspGeometryError(
            "native_franka_grasp_geometry_measurement_invalid"
        ) from exc
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("passed") is True
        and all(len(row) == size for row, size in ((position, 3), (quaternion, 4), (left, 3), (right, 3)))
        and all(math.isfinite(item) for row in (position, quaternion, left, right) for item in row)
    ):
        raise NativeFrankaGraspGeometryError(
            "native_franka_grasp_geometry_measurement_invalid"
        )
    return dict(value)


__all__ = [
    "NativeFrankaGraspGeometryError",
    "SCHEMA_VERSION",
    "measure_live_robotiq_grasp_geometry",
    "validate_measured_grasp_geometry",
]
