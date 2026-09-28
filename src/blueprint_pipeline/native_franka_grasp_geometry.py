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


def _controlled_body_prim(
    *,
    stage: Any,
    body_position_world_m: Sequence[float],
    time_code: Any,
    UsdGeom: Any,
) -> Any | None:
    """Resolve the spawned robot body whose live pose the caller supplied."""

    xforms = UsdGeom.XformCache(time_code)
    for body_name in ("panda_hand", "base_link"):
        candidates = [
            prim
            for prim in stage.Traverse()
            if prim.GetName() == body_name
            and "/robot/" in str(prim.GetPath()).lower()
            and prim.IsA(UsdGeom.Xformable)
        ]
        if candidates:
            return min(
                candidates,
                key=lambda prim: math.dist(
                    [
                        float(value)
                        for value in xforms.GetLocalToWorldTransform(
                            prim
                        ).ExtractTranslation()
                    ],
                    body_position_world_m,
                ),
            )
    return None


def _project_body_bounds_to_live_world(
    *,
    minimum_body_m: Sequence[float],
    maximum_body_m: Sequence[float],
    body_position_world_m: Sequence[float],
    body_quaternion_world_xyzw: Sequence[float],
) -> list[list[float]]:
    corners_world = []
    for x in (minimum_body_m[0], maximum_body_m[0]):
        for y in (minimum_body_m[1], maximum_body_m[1]):
            for z in (minimum_body_m[2], maximum_body_m[2]):
                rotated = rotate_vector_xyzw(
                    body_quaternion_world_xyzw, [x, y, z]
                )
                corners_world.append(
                    [
                        float(body_position_world_m[axis]) + rotated[axis]
                        for axis in range(3)
                    ]
                )
    return [
        [min(point[axis] for point in corners_world) for axis in range(3)],
        [max(point[axis] for point in corners_world) for axis in range(3)],
    ]


def measure_live_robotiq_grasp_geometry(
    *,
    stage: Any,
    controlled_body_position_world_m: Sequence[float],
    controlled_body_quaternion_world_xyzw: Sequence[float],
) -> dict[str, Any]:
    """Measure pad centers and the rigid body-to-grasp transform from live USD."""

    from pxr import Usd, UsdGeom, UsdPhysics

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
    controlled_body = _controlled_body_prim(
        stage=stage,
        body_position_world_m=body_position,
        time_code=Usd.TimeCode.Default(),
        UsdGeom=UsdGeom,
    )
    robot_path_prefix = None
    if controlled_body is not None:
        controlled_path = str(controlled_body.GetPath()).lower()
        robot_path_prefix = controlled_path.split("/robot/", 1)[0] + "/robot/"
    matches: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for prim in Usd.PrimRange(
        stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()
    ):
        path = str(prim.GetPath())
        lowered = path.lower()
        if (
            "/robot/" not in lowered
            or (
                robot_path_prefix is not None
                and not lowered.startswith(robot_path_prefix)
            )
        ):
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
        collision = UsdPhysics.CollisionAPI(prim)
        if (
            side is None
            or not prim.IsA(UsdGeom.Imageable)
            or not prim.HasAPI(UsdPhysics.CollisionAPI)
            or collision.GetCollisionEnabledAttr().Get() is False
        ):
            continue
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = [float(value) for value in aligned.GetMin()]
        maximum = [float(value) for value in aligned.GetMax()]
        if not all(math.isfinite(value) for value in (*minimum, *maximum)):
            continue
        center = [
            (minimum[axis] + maximum[axis]) / 2.0 for axis in range(3)
        ]
        row = {
            "prim_path": path,
            "stage_minimum_world_m": minimum,
            "stage_maximum_world_m": maximum,
            "stage_center_world_m": center,
        }
        if controlled_body is None:
            center_body = rotate_vector_xyzw(
                quaternion_conjugate_xyzw(body_quaternion),
                [center[axis] - body_position[axis] for axis in range(3)],
            )
            bounds_body = None
        else:
            relative = cache.ComputeRelativeBound(
                prim, controlled_body
            ).ComputeAlignedRange()
            minimum_body = [float(value) for value in relative.GetMin()]
            maximum_body = [float(value) for value in relative.GetMax()]
            if not all(
                math.isfinite(value)
                for value in (*minimum_body, *maximum_body)
            ):
                continue
            center_body = [
                (minimum_body[axis] + maximum_body[axis]) / 2.0
                for axis in range(3)
            ]
            bounds_body = [minimum_body, maximum_body]
            row["minimum_controlled_body_m"] = minimum_body
            row["maximum_controlled_body_m"] = maximum_body
            finger_prim = prim
            finger_name = f"{side}_inner_finger"
            while finger_prim and finger_prim.GetName() != finger_name:
                finger_prim = finger_prim.GetParent()
            if finger_prim:
                finger_relative = cache.ComputeRelativeBound(
                    prim, finger_prim
                ).ComputeAlignedRange()
                finger_minimum = [
                    float(value) for value in finger_relative.GetMin()
                ]
                finger_maximum = [
                    float(value) for value in finger_relative.GetMax()
                ]
                if all(
                    math.isfinite(value)
                    for value in (*finger_minimum, *finger_maximum)
                ):
                    row["center_inner_finger_body_m"] = [
                        (finger_minimum[axis] + finger_maximum[axis]) / 2.0
                        for axis in range(3)
                    ]
        center_world = rotate_vector_xyzw(body_quaternion, center_body)
        center_world = [
            body_position[axis] + center_world[axis] for axis in range(3)
        ]
        projected_bounds = (
            None
            if bounds_body is None
            else _project_body_bounds_to_live_world(
                minimum_body_m=bounds_body[0],
                maximum_body_m=bounds_body[1],
                body_position_world_m=body_position,
                body_quaternion_world_xyzw=body_quaternion,
            )
        )
        row.update(
            {
                "minimum_world_m": (
                    minimum if projected_bounds is None else projected_bounds[0]
                ),
                "maximum_world_m": (
                    maximum if projected_bounds is None else projected_bounds[1]
                ),
                "center_world_m": center_world,
                "center_controlled_body_m": center_body,
                "distance_from_controlled_body_m": math.sqrt(
                    sum(value * value for value in center_body)
                ),
            }
        )
        matches[side].append(row)
    if any(not matches[side] for side in ("left", "right")):
        raise NativeFrankaGraspGeometryError(
            "native_franka_grasp_geometry_pad_bounds_missing"
        )
    selected = {
        side: max(
            matches[side],
            key=lambda row: float(row["distance_from_controlled_body_m"]),
        )
        for side in ("left", "right")
    }
    ranges = {
        side: [
            list(selected[side]["minimum_world_m"]),
            list(selected[side]["maximum_world_m"]),
        ]
        for side in ("left", "right")
    }
    centers = {
        side: list(selected[side]["center_world_m"])
        for side in ("left", "right")
    }
    centers_body = {
        side: list(selected[side]["center_controlled_body_m"])
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
    body_to_grasp_position = [
        (centers_body["left"][axis] + centers_body["right"][axis]) / 2.0
        for axis in range(3)
    ]
    body_to_grasp_quaternion = quaternion_multiply_xyzw(
        body_inverse, grasp_quaternion_world
    )
    pad_centers_body = centers_body
    controlled_body_stage_position = None
    controlled_body_pose_disagreement = None
    if controlled_body is not None:
        stage_position = UsdGeom.XformCache(
            Usd.TimeCode.Default()
        ).GetLocalToWorldTransform(controlled_body).ExtractTranslation()
        controlled_body_stage_position = [
            float(stage_position[axis]) for axis in range(3)
        ]
        controlled_body_pose_disagreement = math.dist(
            controlled_body_stage_position, body_position
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_authority": (
            "live_usd_controlled_body_relative_bounds_of_distal_robotiq_"
            "inner_finger_collision_prim_projected_through_live_body_pose"
            if controlled_body is not None
            else "live_usd_world_bounds_of_distal_robotiq_inner_finger_"
            "collision_prim"
        ),
        "controlled_body_usd_prim_path": (
            None if controlled_body is None else str(controlled_body.GetPath())
        ),
        "controlled_body_stage_position_world_m": controlled_body_stage_position,
        "controlled_body_stage_to_live_translation_disagreement_m": (
            controlled_body_pose_disagreement
        ),
        "matched_collision_candidates": matches,
        "selected_pad_colliders": {
            side: dict(selected[side]) for side in ("left", "right")
        },
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
