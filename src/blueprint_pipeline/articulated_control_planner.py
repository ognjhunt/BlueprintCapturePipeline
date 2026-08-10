"""Plan the handle path a scripted control must follow to open a hinged door.

A rigid pick/place positive can move in straight lines: the object goes where
the gripper puts it. A hinged door cannot. Its handle is pinned to a circle
whose radius is the distance from the hinge axis, and any waypoint off that
circle commands the arm to pull the handle off the door - which either fights
the joint constraint or, worse, succeeds and scores a broken asset.

So the geometry is derived here rather than authored: the arc, the plane it
lies in, the direction to approach from, and the direction to retreat along
once the door has moved. Nothing is assumed about which way the door faces or
whether its hinge is world-vertical, because neither is true in general and
both would silently produce a plausible-looking wrong path.

The planner also reports the load the arm has to supply. A scripted positive
that passes proves the *program* works; it says nothing about whether a real
arm could do the same thing until the hinge torque is divided by the lever arm
and compared against the arm's actual capability. That conversion is cheap and
easy to omit, so it is part of the plan rather than left to the reader.

Kinematics stay out. This module resolves *where the handle must be*; turning
those poses into joint commands is the injected IK preflight's job, and keeping
the split means the arc is testable without a solver or a GPU.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence


ARTICULATED_CONTROL_PLAN_SCHEMA_VERSION = "articulated_control_plan.v1"
DEFAULT_WAYPOINT_COUNT = 6
DEFAULT_APPROACH_STANDOFF_M = 0.12
MINIMUM_LEVER_ARM_M = 1e-3
MINIMUM_HANDLE_PROTRUSION_M = 8e-3


class ArticulatedControlPlannerError(ValueError):
    """Stable, sorted articulated control-planning failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _vector(value: Any, error: str) -> list[float]:
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ArticulatedControlPlannerError([error]) from exc
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ArticulatedControlPlannerError([error])
    return values


def _normalize(vector: Sequence[float], error: str) -> list[float]:
    length = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(length) or length <= 0.0:
        raise ArticulatedControlPlannerError([error])
    return [value / length for value in vector]


def _cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _rotate_about_axis(
    vector: Sequence[float], axis_unit: Sequence[float], radians: float
) -> list[float]:
    """Rodrigues rotation, so an off-axis hinge is no more special than a plumb one."""

    cos = math.cos(radians)
    sin = math.sin(radians)
    crossed = _cross(axis_unit, vector)
    scaled = _dot(axis_unit, vector) * (1.0 - cos)
    return [
        vector[index] * cos + crossed[index] * sin + axis_unit[index] * scaled
        for index in range(3)
    ]


def plan_articulated_handle_trajectory(
    *,
    hinge_point_world_m: Sequence[float],
    hinge_axis_world: Sequence[float],
    handle_grasp_point_closed_world_m: Sequence[float],
    open_angle_degrees: float,
    authored_limit_degrees: float,
    waypoint_count: int = DEFAULT_WAYPOINT_COUNT,
    approach_standoff_m: float = DEFAULT_APPROACH_STANDOFF_M,
    joint_damping_n_m_s_per_rad: float | None = None,
    sweep_duration_s: float | None = None,
) -> dict[str, Any]:
    """Resolve the arc a door handle sweeps, and the load holding it back."""

    hinge_point = _vector(
        hinge_point_world_m, "articulated_control_plan_hinge_point_invalid"
    )
    axis = _normalize(
        _vector(hinge_axis_world, "articulated_control_plan_hinge_axis_invalid"),
        "articulated_control_plan_hinge_axis_degenerate",
    )
    handle = _vector(
        handle_grasp_point_closed_world_m,
        "articulated_control_plan_handle_point_invalid",
    )

    errors: list[str] = []
    for name, value in (
        ("open_angle_degrees", open_angle_degrees),
        ("authored_limit_degrees", authored_limit_degrees),
        ("approach_standoff_m", approach_standoff_m),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"articulated_control_plan_{name}_invalid")
        elif not math.isfinite(float(value)):
            errors.append(f"articulated_control_plan_{name}_invalid")
    if isinstance(waypoint_count, bool) or not isinstance(waypoint_count, int):
        errors.append("articulated_control_plan_waypoint_count_invalid")
    elif waypoint_count < 2:
        errors.append("articulated_control_plan_waypoint_count_invalid")
    if errors:
        raise ArticulatedControlPlannerError(errors)

    open_angle = float(open_angle_degrees)
    limit = float(authored_limit_degrees)
    standoff = float(approach_standoff_m)
    if open_angle <= 0.0:
        errors.append("articulated_control_plan_open_angle_not_positive")
    # The limit is the asset's own contract. Commanding past it makes the
    # solver's constraint handling the thing under test, not the door.
    if abs(open_angle) > abs(limit):
        errors.append(
            "articulated_control_plan_open_angle_beyond_authored_limit:"
            f"{open_angle}>{limit}"
        )
    if standoff <= 0.0:
        errors.append("articulated_control_plan_approach_standoff_not_positive")

    offset = [handle[index] - hinge_point[index] for index in range(3)]
    axial_length = _dot(offset, axis)
    axial = [axis[index] * axial_length for index in range(3)]
    radial = [offset[index] - axial[index] for index in range(3)]
    lever_arm = math.sqrt(sum(value * value for value in radial))
    if lever_arm < MINIMUM_LEVER_ARM_M:
        # Pulling on the axis produces no moment however hard the arm pulls.
        errors.append("articulated_control_plan_handle_on_hinge_axis")
    if errors:
        raise ArticulatedControlPlannerError(errors)

    radial_unit = [value / lever_arm for value in radial]
    base = [hinge_point[index] + axial[index] for index in range(3)]

    def _at(angle_degrees: float) -> list[float]:
        rotated = _rotate_about_axis(radial, axis, math.radians(angle_degrees))
        return [base[index] + rotated[index] for index in range(3)]

    # The handle's initial velocity is the direction the door swings, which is
    # by definition its outward side. Deriving the normal this way means a door
    # that opens the other way, or hangs off a slanted hinge, needs no flag.
    def _outward_at(angle_degrees: float) -> list[float]:
        rotated_radial = _rotate_about_axis(
            radial_unit, axis, math.radians(angle_degrees)
        )
        return _normalize(
            _cross(axis, rotated_radial),
            "articulated_control_plan_outward_normal_degenerate",
        )

    step = open_angle / float(waypoint_count - 1)
    waypoints: list[dict[str, Any]] = []
    for index in range(waypoint_count):
        angle = step * index
        waypoints.append(
            {
                "waypoint_index": index,
                "door_angle_degrees": angle,
                "position_world_m": _at(angle),
                "radius_m": lever_arm,
                "outward_normal_world": _outward_at(angle),
            }
        )

    closed_normal = _outward_at(0.0)
    open_normal = _outward_at(open_angle)
    approach = {
        "position_world_m": [
            waypoints[0]["position_world_m"][index] + closed_normal[index] * standoff
            for index in range(3)
        ],
        "outward_normal_world": closed_normal,
        "standoff_m": standoff,
    }
    retreat = {
        "position_world_m": [
            waypoints[-1]["position_world_m"][index] + open_normal[index] * standoff
            for index in range(3)
        ],
        "outward_normal_world": open_normal,
        "standoff_m": standoff,
    }

    load: dict[str, Any] = {
        "lever_arm_m": lever_arm,
        "swept_angle_degrees": open_angle,
        "hinge_torque_n_m": None,
        "handle_force_n": None,
        "mean_angular_velocity_rad_s": None,
        "damping_source": "authored_joint_drive"
        if joint_damping_n_m_s_per_rad is not None
        else "unreported",
    }
    if joint_damping_n_m_s_per_rad is not None and sweep_duration_s:
        damping = float(joint_damping_n_m_s_per_rad)
        duration = float(sweep_duration_s)
        if damping < 0.0 or duration <= 0.0 or not math.isfinite(damping):
            raise ArticulatedControlPlannerError(
                ["articulated_control_plan_load_parameters_invalid"]
            )
        velocity = math.radians(open_angle) / duration
        torque = damping * velocity
        load["mean_angular_velocity_rad_s"] = velocity
        load["hinge_torque_n_m"] = torque
        load["handle_force_n"] = torque / lever_arm

    phases = [
        {"phase_id": "approach", "position_world_m": approach["position_world_m"]},
        {"phase_id": "grasp", "position_world_m": waypoints[0]["position_world_m"]},
        *[
            {
                "phase_id": f"sweep_{row['waypoint_index']:02d}",
                "position_world_m": row["position_world_m"],
            }
            for row in waypoints[1:]
        ],
        {"phase_id": "release", "position_world_m": waypoints[-1]["position_world_m"]},
        {"phase_id": "retreat", "position_world_m": retreat["position_world_m"]},
    ]

    return {
        "schema_version": ARTICULATED_CONTROL_PLAN_SCHEMA_VERSION,
        "hinge_point_world_m": hinge_point,
        "hinge_axis_world_unit": axis,
        "waypoints": waypoints,
        "approach_pose": approach,
        "retreat_pose": retreat,
        "phases": phases,
        "required_load": load,
        "claim_boundary": {
            "geometry_only_no_kinematics_resolved": True,
            "load_is_authored_drive_not_measured_hardware": True,
            "gasket_seal_force_not_modelled": True,
        },
    }


def detect_articulated_handle_grasp_point(
    *,
    usd_path: str | Path,
    member_prim_path: str,
    hinge_point_world_m: Sequence[float],
    hinge_axis_world: Sequence[float],
    minimum_protrusion_m: float = MINIMUM_HANDLE_PROTRUSION_M,
) -> dict[str, Any]:
    """Find where to grasp a hinged member, by shape rather than by name.

    Generated twins do not come with a prim called ``handle``, and the part
    numbering carries no meaning, so the grasp point has to be recovered from
    geometry. A handle is whatever stands proud of the panel on the side the
    door is pulled from - which also distinguishes it from shelves and liners,
    since those protrude just as far in the opposite direction.
    """

    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ArticulatedControlPlannerError(
            ["articulated_handle_openusd_runtime_missing"]
        ) from exc

    source = Path(usd_path).expanduser().resolve()
    if not source.is_file():
        raise ArticulatedControlPlannerError(["articulated_handle_source_missing"])
    hinge_point = _vector(
        hinge_point_world_m, "articulated_control_plan_hinge_point_invalid"
    )
    axis = _normalize(
        _vector(hinge_axis_world, "articulated_control_plan_hinge_axis_invalid"),
        "articulated_control_plan_hinge_axis_degenerate",
    )

    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ArticulatedControlPlannerError(["articulated_handle_source_unreadable"])
    member = stage.GetPrimAtPath(str(member_prim_path))
    if not member.IsValid():
        raise ArticulatedControlPlannerError(["articulated_handle_member_missing"])

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    parts: list[tuple[str, list[float], list[float], float]] = []
    for prim in Usd.PrimRange(member):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if bounds.IsEmpty():
            continue
        low = [float(value) for value in bounds.GetMin()]
        high = [float(value) for value in bounds.GetMax()]
        volume = 1.0
        for index in range(3):
            volume *= max(high[index] - low[index], 0.0)
        parts.append((str(prim.GetPath()), low, high, volume))
    if not parts:
        raise ArticulatedControlPlannerError(["articulated_handle_member_has_no_mesh"])

    panel = max(parts, key=lambda row: row[3])
    panel_corners = [
        [
            panel[1][0] if a else panel[2][0],
            panel[1][1] if b else panel[2][1],
            panel[1][2] if c else panel[2][2],
        ]
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    ]

    # A hinge anchors on the door's front face, not through its middle, so the
    # direction from the hinge to the panel's centroid is not the direction the
    # door extends - it leans toward the face by however thick the door is, and
    # tilts "outward" off the true normal by the same angle. The door's own
    # plane is what settles it: the widest span it has perpendicular to the
    # hinge axis is its width, and the normal is square to that and the axis.
    reference = _normalize(
        [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0],
        "articulated_handle_outward_normal_degenerate",
    )
    basis_u = _normalize(
        [
            reference[index] - axis[index] * _dot(reference, axis)
            for index in range(3)
        ],
        "articulated_handle_outward_normal_degenerate",
    )
    basis_v = _cross(axis, basis_u)
    # Search for the *thinnest* direction, not the widest. A box's projected
    # span is w|cos| + d|sin|, so maximizing it always drifts to a diagonal and
    # never recovers an edge; minimizing it lands exactly on the slab normal,
    # which is what a door panel's thickness direction is.
    thinnest_span, normal = None, None
    for step_index in range(180):
        radians = math.pi * step_index / 180.0
        direction = [
            basis_u[index] * math.cos(radians) + basis_v[index] * math.sin(radians)
            for index in range(3)
        ]
        projections = [_dot(corner, direction) for corner in panel_corners]
        span = max(projections) - min(projections)
        if thinnest_span is None or span < thinnest_span:
            thinnest_span, normal = span, direction
    if normal is None:
        raise ArticulatedControlPlannerError(
            ["articulated_handle_member_on_hinge_axis"]
        )
    radial_unit = _normalize(
        _cross(normal, axis), "articulated_handle_member_on_hinge_axis"
    )

    centre = [
        (panel[1][index] + panel[2][index]) / 2.0 for index in range(3)
    ]
    offset = [centre[index] - hinge_point[index] for index in range(3)]
    if _dot(offset, radial_unit) < 0.0:
        radial_unit = [-value for value in radial_unit]
    # The side the door swings toward is the side it is pulled from.
    outward = _normalize(
        _cross(axis, radial_unit), "articulated_handle_outward_normal_degenerate"
    )

    def _extent(low: Sequence[float], high: Sequence[float]) -> float:
        return max(
            _dot([low[0] if a else high[0], low[1] if b else high[1],
                  low[2] if c else high[2]], outward)
            for a in (0, 1)
            for b in (0, 1)
            for c in (0, 1)
        )

    panel_face = _extent(panel[1], panel[2])
    protruding = [
        row
        for row in parts
        if row[0] != panel[0]
        and _extent(row[1], row[2]) - panel_face >= float(minimum_protrusion_m)
    ]
    if not protruding:
        raise ArticulatedControlPlannerError(
            ["articulated_handle_no_protruding_handle_on_member"]
        )

    low = [min(row[1][index] for row in protruding) for index in range(3)]
    high = [max(row[2][index] for row in protruding) for index in range(3)]
    grasp = [(low[index] + high[index]) / 2.0 for index in range(3)]
    grasp_offset = [grasp[index] - hinge_point[index] for index in range(3)]
    grasp_radial = [
        grasp_offset[index] - axis[index] * _dot(grasp_offset, axis)
        for index in range(3)
    ]
    lever_arm = math.sqrt(sum(value * value for value in grasp_radial))

    return {
        "handle_prim_path": min(row[0] for row in protruding),
        "handle_prim_paths": sorted(row[0] for row in protruding),
        "panel_prim_path": panel[0],
        "grasp_point_world_m": grasp,
        "handle_aabb_min_m": low,
        "handle_aabb_max_m": high,
        "outward_normal_world": outward,
        "protrusion_m": max(_extent(row[1], row[2]) for row in protruding)
        - panel_face,
        "lever_arm_m": lever_arm,
        "claim_boundary": {
            "handle_identified_by_geometry_not_by_name": True,
            "grasp_point_is_a_bbox_centroid_not_a_grasp_synthesis": True,
        },
    }


__all__ = [
    "ARTICULATED_CONTROL_PLAN_SCHEMA_VERSION",
    "detect_articulated_handle_grasp_point",
    "ArticulatedControlPlannerError",
    "plan_articulated_handle_trajectory",
]
