"""Local Franka base placement search for the articulated door task.

This is a deterministic geometric screen over the observed scene: it ranks
candidate base positions that keep the footprint out of obstacles, the target
keepout, and the full door-swing corridor while keeping every frozen
door-state handle waypoint inside the frozen reach annulus with an
unobstructed approach line. It deliberately claims nothing native: IK, contact,
joint limits, and floor support are qualified only by the native gates, and a
locally admissible candidate may still fail there. An empty result is a typed
task-construction abstention with the per-constraint rejection histogram, so
the smallest blocking constraint is visible without a simulator.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


PLACEMENT_SEARCH_SCHEMA_VERSION = "articulated_franka_base_placement_search.v1"


class ArticulatedFrankaPlacementError(ValueError):
    """Stable, sorted placement-search input failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _finite_vector(value: Any, length: int, error: str) -> list[float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != length
    ):
        raise ArticulatedFrankaPlacementError([error])
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ArticulatedFrankaPlacementError([error]) from exc
    if any(not math.isfinite(item) for item in result):
        raise ArticulatedFrankaPlacementError([error])
    return result


def _positive(value: Any, error: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArticulatedFrankaPlacementError([error])
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ArticulatedFrankaPlacementError([error])
    return number


def _aabb(value: Any, error: str) -> tuple[list[float], list[float]]:
    if not isinstance(value, Mapping):
        raise ArticulatedFrankaPlacementError([error])
    minimum = _finite_vector(value.get("aabb_min"), 3, error)
    maximum = _finite_vector(value.get("aabb_max"), 3, error)
    if any(minimum[axis] > maximum[axis] for axis in range(3)):
        raise ArticulatedFrankaPlacementError([error])
    return minimum, maximum


def _circle_intersects_aabb_2d(
    x: float, y: float, radius: float, minimum: Sequence[float], maximum: Sequence[float]
) -> bool:
    nearest_x = min(max(x, minimum[0]), maximum[0])
    nearest_y = min(max(y, minimum[1]), maximum[1])
    return math.hypot(x - nearest_x, y - nearest_y) <= radius


def _segment_intersects_aabb_2d(
    start: Sequence[float],
    end: Sequence[float],
    minimum: Sequence[float],
    maximum: Sequence[float],
    inflation: float,
) -> bool:
    low = 0.0
    high = 1.0
    for axis in range(2):
        lower = minimum[axis] - inflation
        upper = maximum[axis] + inflation
        delta = end[axis] - start[axis]
        if abs(delta) <= 1e-15:
            if start[axis] < lower or start[axis] > upper:
                return False
            continue
        axis_low = (lower - start[axis]) / delta
        axis_high = (upper - start[axis]) / delta
        if axis_low > axis_high:
            axis_low, axis_high = axis_high, axis_low
        low = max(low, axis_low)
        high = min(high, axis_high)
        if low > high:
            return False
    return True




def _circle_intersects_triangle_2d(
    x: float, y: float, radius: float, triangle: Sequence[Sequence[float]]
) -> bool:
    vertices = [(float(p[0]), float(p[1])) for p in triangle]

    def _sign(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1])

    # A vertical wall triangle projects to a segment or a single point. Its
    # 2D area is zero, so the point-in-triangle sign test degenerates to
    # "contains everything"; only the edge-distance test is meaningful there.
    doubled_area = abs(_sign(vertices[0], vertices[1], vertices[2]))
    if doubled_area > 1e-12:
        d1 = _sign(vertices[0], vertices[1], (x, y))
        d2 = _sign(vertices[1], vertices[2], (x, y))
        d3 = _sign(vertices[2], vertices[0], (x, y))
        has_negative = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_positive = (d1 > 0) or (d2 > 0) or (d3 > 0)
        if not (has_negative and has_positive):
            return True
    for start, end in ((0, 1), (1, 2), (2, 0)):
        ax, ay = vertices[start]
        bx, by = vertices[end]
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-18:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_sq))
        if math.hypot(x - (ax + t * dx), y - (ay + t * dy)) <= radius:
            return True
    return False


def search_franka_base_placement(
    *,
    hinge_origin_world_m: Sequence[float],
    joint_axis_world: Sequence[float],
    handle_closed_midpoint_world_m: Sequence[float],
    member_vertical_interval_m: Sequence[float],
    door_radius_m: float,
    door_state_angles_degrees: Sequence[float],
    obstacles: Sequence[Mapping[str, Any]],
    triangle_shell_obstacles: Sequence[Mapping[str, Any]] = (),
    keepout_aabbs: Sequence[Mapping[str, Any]],
    search_region_aabb_m: Mapping[str, Any],
    base_footprint_radius_m: float = 0.16,
    base_obstacle_z_interval_m: Sequence[float] = (0.0, 0.75),
    base_reach_center_z_m: float = 0.333,
    reach_maximum_m: float = 0.855,
    reach_minimum_m: float = 0.30,
    reach_margin_m: float = 0.05,
    arm_clearance_radius_m: float = 0.07,
    door_corridor_margin_m: float = 0.10,
    grid_resolution_m: float = 0.05,
    maximum_candidates: int = 16,
) -> dict[str, Any]:
    """Rank locally admissible base positions or emit a typed abstention."""

    hinge = _finite_vector(hinge_origin_world_m, 3, "placement_hinge_invalid")
    axis = _finite_vector(joint_axis_world, 3, "placement_axis_invalid")
    axis_norm = math.sqrt(sum(value * value for value in axis))
    if axis_norm <= 1e-12 or abs(axis[2] / axis_norm) < 0.99:
        raise ArticulatedFrankaPlacementError(["placement_joint_axis_not_vertical"])
    handle = _finite_vector(
        handle_closed_midpoint_world_m, 3, "placement_handle_midpoint_invalid"
    )
    member_interval = _finite_vector(
        member_vertical_interval_m, 2, "placement_member_interval_invalid"
    )
    if member_interval[0] >= member_interval[1]:
        raise ArticulatedFrankaPlacementError(["placement_member_interval_invalid"])
    door_radius = _positive(door_radius_m, "placement_door_radius_invalid")
    states: list[float] = []
    for value in door_state_angles_degrees:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ArticulatedFrankaPlacementError(["placement_door_states_invalid"])
        states.append(float(value))
    if len(states) < 2:
        raise ArticulatedFrankaPlacementError(["placement_door_states_invalid"])

    obstacle_rows: list[dict[str, Any]] = []
    for index, row in enumerate(obstacles):
        if not isinstance(row, Mapping):
            raise ArticulatedFrankaPlacementError([f"placement_obstacle_invalid:{index}"])
        minimum = _finite_vector(
            row.get("world_aabb_min_m"), 3, f"placement_obstacle_invalid:{index}"
        )
        maximum = _finite_vector(
            row.get("world_aabb_max_m"), 3, f"placement_obstacle_invalid:{index}"
        )
        obstacle_rows.append(
            {
                "obstacle_id": str(row.get("obstacle_id") or f"obstacle_{index}"),
                "minimum": minimum,
                "maximum": maximum,
            }
        )
    keepouts = [
        {"label": str(row.get("label") or f"keepout_{index}"), "bounds": _aabb(row, f"placement_keepout_invalid:{index}")}
        for index, row in enumerate(keepout_aabbs)
    ]
    shell_rows: list[dict[str, Any]] = []
    for index, row in enumerate(triangle_shell_obstacles):
        if not isinstance(row, Mapping) or not isinstance(row.get("triangles"), list):
            raise ArticulatedFrankaPlacementError(
                [f"placement_shell_obstacle_invalid:{index}"]
            )
        triangles = []
        for triangle in row["triangles"]:
            if (
                not isinstance(triangle, Sequence)
                or len(triangle) != 3
                or any(len(point) != 3 for point in triangle)
            ):
                raise ArticulatedFrankaPlacementError(
                    [f"placement_shell_obstacle_invalid:{index}"]
                )
            z_low = min(float(point[2]) for point in triangle)
            z_high = max(float(point[2]) for point in triangle)
            triangles.append((triangle, z_low, z_high))
        shell_rows.append(
            {
                "obstacle_id": str(row.get("obstacle_id") or f"shell_{index}"),
                "triangles": triangles,
            }
        )
    region_min, region_max = _aabb(search_region_aabb_m, "placement_search_region_invalid")

    footprint = _positive(base_footprint_radius_m, "placement_footprint_invalid")
    base_band = _finite_vector(
        base_obstacle_z_interval_m, 2, "placement_base_band_invalid"
    )
    reach_maximum = _positive(reach_maximum_m, "placement_reach_invalid")
    reach_minimum = _positive(reach_minimum_m, "placement_reach_invalid")
    reach_margin = float(reach_margin_m)
    if not math.isfinite(reach_margin) or reach_margin < 0.0 or reach_minimum >= reach_maximum:
        raise ArticulatedFrankaPlacementError(["placement_reach_invalid"])
    reach_center_z = float(base_reach_center_z_m)
    if not math.isfinite(reach_center_z) or reach_center_z < 0.0:
        raise ArticulatedFrankaPlacementError(["placement_reach_center_invalid"])
    arm_clearance = _positive(arm_clearance_radius_m, "placement_arm_clearance_invalid")
    corridor_margin = float(door_corridor_margin_m)
    if not math.isfinite(corridor_margin) or corridor_margin < 0.0:
        raise ArticulatedFrankaPlacementError(["placement_corridor_margin_invalid"])
    resolution = _positive(grid_resolution_m, "placement_grid_resolution_invalid")
    if not isinstance(maximum_candidates, int) or maximum_candidates < 1:
        raise ArticulatedFrankaPlacementError(["placement_candidate_cap_invalid"])

    handle_radial = [handle[0] - hinge[0], handle[1] - hinge[1]]
    handle_radius = math.hypot(*handle_radial)
    if handle_radius <= 1e-9:
        raise ArticulatedFrankaPlacementError(["placement_handle_midpoint_invalid"])
    handle_angle = math.atan2(handle_radial[1], handle_radial[0])
    handle_waypoints = [
        (
            state,
            [
                hinge[0] + handle_radius * math.cos(handle_angle + math.radians(state)),
                hinge[1] + handle_radius * math.sin(handle_angle + math.radians(state)),
                handle[2],
            ],
        )
        for state in states
    ]
    corridor_radius = max(handle_radius, door_radius)
    corridor_low = min(0.0, states[0])
    corridor_high = max(states)
    # The swinging member occupies its vertical interval only; a base column
    # whose obstacle band never reaches that interval cannot be struck by the
    # door, so the corridor keep-out applies only when the bands overlap. The
    # arm itself is qualified against the moving door by the native gates.
    corridor_applies = min(base_band[1], member_interval[1]) > max(
        base_band[0], member_interval[0]
    )

    histogram: dict[str, int] = {}

    def _reject(reason: str) -> None:
        histogram[reason] = histogram.get(reason, 0) + 1

    candidates: list[dict[str, Any]] = []
    x_steps = max(1, int(math.floor((region_max[0] - region_min[0]) / resolution)) + 1)
    y_steps = max(1, int(math.floor((region_max[1] - region_min[1]) / resolution)) + 1)
    for x_index in range(x_steps):
        for y_index in range(y_steps):
            x = region_min[0] + x_index * resolution
            y = region_min[1] + y_index * resolution
            if x > region_max[0] or y > region_max[1]:
                continue

            keepout_hit = False
            for keepout in keepouts:
                minimum, maximum = keepout["bounds"]
                if _circle_intersects_aabb_2d(x, y, footprint, minimum, maximum):
                    keepout_hit = True
                    break
            if keepout_hit:
                _reject("base_inside_target_keepout")
                continue

            corridor_distance = math.hypot(x - hinge[0], y - hinge[1])
            if corridor_applies and corridor_distance <= corridor_radius + corridor_margin + footprint:
                relative = math.degrees(
                    math.atan2(y - hinge[1], x - hinge[0]) - handle_angle
                )
                while relative > 180.0:
                    relative -= 360.0
                while relative < -180.0:
                    relative += 360.0
                angular_pad = math.degrees(
                    math.atan2(footprint + corridor_margin, max(corridor_distance, 1e-6))
                )
                if corridor_low - angular_pad <= relative <= corridor_high + angular_pad:
                    _reject("base_in_door_swing_corridor")
                    continue

            obstacle_hit = None
            for row in obstacle_rows:
                if (
                    min(base_band[1], row["maximum"][2]) > max(base_band[0], row["minimum"][2])
                    and _circle_intersects_aabb_2d(
                        x, y, footprint, row["minimum"], row["maximum"]
                    )
                ):
                    obstacle_hit = row["obstacle_id"]
                    break
            if obstacle_hit is not None:
                _reject("base_footprint_obstacle")
                continue

            shell_hit = None
            for row in shell_rows:
                for triangle, z_low, z_high in row["triangles"]:
                    if min(base_band[1], z_high) <= max(base_band[0], z_low):
                        continue
                    if _circle_intersects_triangle_2d(x, y, footprint, triangle):
                        shell_hit = row["obstacle_id"]
                        break
                if shell_hit:
                    break
            if shell_hit is not None:
                _reject("base_footprint_shell_triangle")
                continue

            reach_rows = []
            reach_ok = True
            worst_margin = math.inf
            for state, waypoint in handle_waypoints:
                distance = math.sqrt(
                    (waypoint[0] - x) ** 2
                    + (waypoint[1] - y) ** 2
                    + (waypoint[2] - reach_center_z) ** 2
                )
                reach_rows.append({"angle_degrees": state, "reach_m": distance})
                if distance > reach_maximum - reach_margin or distance < reach_minimum:
                    reach_ok = False
                    break
                worst_margin = min(worst_margin, reach_maximum - reach_margin - distance)
            if not reach_ok:
                _reject("handle_arc_unreachable")
                continue

            approach_blocked = None
            for state, waypoint in handle_waypoints:
                for row in obstacle_rows:
                    z_low = min(waypoint[2] - 0.15, base_band[0])
                    z_high = waypoint[2] + 0.15
                    if min(z_high, row["maximum"][2]) <= max(z_low, row["minimum"][2]):
                        continue
                    if _segment_intersects_aabb_2d(
                        (x, y), waypoint, row["minimum"], row["maximum"], arm_clearance
                    ):
                        approach_blocked = row["obstacle_id"]
                        break
                if approach_blocked:
                    break
            if approach_blocked is not None:
                _reject("approach_line_blocked")
                continue

            clearance = math.inf
            for row in obstacle_rows:
                nearest_x = min(max(x, row["minimum"][0]), row["maximum"][0])
                nearest_y = min(max(y, row["minimum"][1]), row["maximum"][1])
                clearance = min(
                    clearance, math.hypot(x - nearest_x, y - nearest_y) - footprint
                )
            score = worst_margin + min(clearance, 0.5)
            candidates.append(
                {
                    "base_xy_world_m": [round(x, 6), round(y, 6)],
                    "score": round(score, 6),
                    "worst_reach_margin_m": round(worst_margin, 6),
                    "minimum_obstacle_clearance_m": (
                        round(clearance, 6) if math.isfinite(clearance) else None
                    ),
                    "per_state_reach_m": [
                        {
                            "angle_degrees": row["angle_degrees"],
                            "reach_m": round(row["reach_m"], 6),
                        }
                        for row in reach_rows
                    ],
                    "franka_base_class_box": {
                        "aabb_min": [
                            round(x - footprint, 6),
                            round(y - footprint, 6),
                            0.0,
                        ],
                        "aabb_max": [
                            round(x + footprint, 6),
                            round(y + footprint, 6),
                            round(base_band[1], 6),
                        ],
                    },
                }
            )

    candidates.sort(key=lambda row: (-row["score"], row["base_xy_world_m"][0], row["base_xy_world_m"][1]))
    selected = candidates[:maximum_candidates]

    receipt: dict[str, Any] = {
        "schema_version": PLACEMENT_SEARCH_SCHEMA_VERSION,
        "status": (
            "base_candidates_locally_admissible"
            if selected
            else "franka_base_placement_infeasible"
        ),
        "hinge_origin_world_m": hinge,
        "handle_closed_midpoint_world_m": handle,
        "handle_arc_radius_m": round(handle_radius, 6),
        "member_vertical_interval_m": member_interval,
        "door_radius_m": door_radius,
        "door_corridor_applies_to_base_band": corridor_applies,
        "door_state_angles_degrees": states,
        "search_region_aabb_m": {"aabb_min": region_min, "aabb_max": region_max},
        "parameters": {
            "base_footprint_radius_m": footprint,
            "base_obstacle_z_interval_m": base_band,
            "base_reach_center_z_m": reach_center_z,
            "reach_maximum_m": reach_maximum,
            "reach_minimum_m": reach_minimum,
            "reach_margin_m": reach_margin,
            "arm_clearance_radius_m": arm_clearance,
            "door_corridor_margin_m": corridor_margin,
            "grid_resolution_m": resolution,
            "maximum_candidates": maximum_candidates,
        },
        "obstacle_count": len(obstacle_rows),
        "triangle_shell_obstacle_count": len(shell_rows),
        "keepout_labels": sorted(keepout["label"] for keepout in keepouts),
        "evaluated_cell_count": x_steps * y_steps,
        "rejection_histogram": dict(sorted(histogram.items())),
        "candidates": selected,
        "claim_boundary": {
            "local_geometric_screen_only": True,
            "native_ik_and_contact_required": True,
            "floor_support_native_readback_required": True,
            "approach_line_check_is_heuristic": True,
            "physical_equivalence_proven": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = [
    "ArticulatedFrankaPlacementError",
    "PLACEMENT_SEARCH_SCHEMA_VERSION",
    "search_franka_base_placement",
]
