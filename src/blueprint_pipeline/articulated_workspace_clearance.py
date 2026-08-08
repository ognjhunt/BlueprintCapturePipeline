"""Deterministic articulated-member sweep clearance against bound obstacles.

This is an early, simulator-free rejection gate.  It rotates an observed
candidate member centerline through a preregistered angle and tests it against
exact source-obstacle AABBs.  A collision of a zero-thickness centerline is a
strong rejection: adding the real member thickness cannot restore clearance.
Passing remains candidate evidence and never replaces native collider, IK, or
contact qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "articulated_workspace_clearance.v1"


class ArticulatedWorkspaceClearanceError(ValueError):
    """Stable, sorted sweep-clearance failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _finite_vector(value: Any, length: int, error: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ArticulatedWorkspaceClearanceError([error])
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ArticulatedWorkspaceClearanceError([error]) from exc
    if any(not math.isfinite(item) for item in result):
        raise ArticulatedWorkspaceClearanceError([error])
    return result


def _segment_aabb_intersection(
    start: Sequence[float], end: Sequence[float], minimum: Sequence[float], maximum: Sequence[float]
) -> tuple[bool, list[float] | None]:
    """Liang-Barsky segment/AABB intersection in the horizontal plane."""

    low = 0.0
    high = 1.0
    for axis in range(2):
        delta = float(end[axis]) - float(start[axis])
        if abs(delta) <= 1e-15:
            if float(start[axis]) < float(minimum[axis]) or float(start[axis]) > float(maximum[axis]):
                return False, None
            continue
        axis_low = (float(minimum[axis]) - float(start[axis])) / delta
        axis_high = (float(maximum[axis]) - float(start[axis])) / delta
        if axis_low > axis_high:
            axis_low, axis_high = axis_high, axis_low
        low = max(low, axis_low)
        high = min(high, axis_high)
        if low > high:
            return False, None
    return True, [
        float(start[axis]) + low * (float(end[axis]) - float(start[axis]))
        for axis in range(2)
    ]


def evaluate_revolute_member_sweep(
    *,
    hinge_origin_world_m: Sequence[float],
    closed_endpoint_world_m: Sequence[float],
    member_vertical_interval_m: Sequence[float],
    start_angle_degrees: float,
    end_angle_degrees: float,
    obstacles: Sequence[Mapping[str, Any]],
    angular_resolution_degrees: float = 0.25,
    member_half_thickness_m: float = 0.0,
) -> dict[str, Any]:
    """Return the first deterministic centerline collision, if one exists."""

    hinge = _finite_vector(hinge_origin_world_m, 3, "sweep_hinge_invalid")
    endpoint = _finite_vector(closed_endpoint_world_m, 3, "sweep_endpoint_invalid")
    vertical = _finite_vector(member_vertical_interval_m, 2, "sweep_vertical_interval_invalid")
    if vertical[0] >= vertical[1]:
        raise ArticulatedWorkspaceClearanceError(["sweep_vertical_interval_invalid"])
    try:
        start_angle = float(start_angle_degrees)
        end_angle = float(end_angle_degrees)
        resolution = float(angular_resolution_degrees)
        half_thickness = float(member_half_thickness_m)
    except (TypeError, ValueError) as exc:
        raise ArticulatedWorkspaceClearanceError(["sweep_parameter_invalid"]) from exc
    if (
        not all(math.isfinite(value) for value in (start_angle, end_angle, resolution, half_thickness))
        or end_angle == start_angle
        or resolution <= 0.0
        or half_thickness < 0.0
    ):
        raise ArticulatedWorkspaceClearanceError(["sweep_parameter_invalid"])
    if abs(endpoint[2] - hinge[2]) > 1e-9:
        raise ArticulatedWorkspaceClearanceError(["sweep_centerline_not_horizontal"])
    radius = math.hypot(endpoint[0] - hinge[0], endpoint[1] - hinge[1])
    if radius <= 0.0:
        raise ArticulatedWorkspaceClearanceError(["sweep_member_radius_invalid"])
    source_angle = math.atan2(endpoint[1] - hinge[1], endpoint[0] - hinge[0])

    normalized_obstacles: list[dict[str, Any]] = []
    for index, obstacle in enumerate(obstacles):
        if not isinstance(obstacle, Mapping):
            raise ArticulatedWorkspaceClearanceError([f"sweep_obstacle_{index}_invalid"])
        minimum = _finite_vector(
            obstacle.get("world_aabb_min_m"), 3, f"sweep_obstacle_{index}_aabb_invalid"
        )
        maximum = _finite_vector(
            obstacle.get("world_aabb_max_m"), 3, f"sweep_obstacle_{index}_aabb_invalid"
        )
        if any(minimum[axis] >= maximum[axis] for axis in range(3)):
            raise ArticulatedWorkspaceClearanceError([f"sweep_obstacle_{index}_aabb_invalid"])
        normalized_obstacles.append(
            {
                "obstacle_id": str(obstacle.get("obstacle_id") or f"obstacle_{index}"),
                "world_aabb_min_m": minimum,
                "world_aabb_max_m": maximum,
                "source_receipt_digest": obstacle.get("source_receipt_digest"),
            }
        )

    direction = 1.0 if end_angle > start_angle else -1.0
    step_count = int(math.ceil(abs(end_angle - start_angle) / resolution))
    angles = [
        start_angle + index * direction * resolution for index in range(step_count)
    ] + [end_angle]
    first_collision: dict[str, Any] | None = None
    collision_obstacle_ids: set[str] = set()
    for angle_degrees in angles:
        angle = source_angle + math.radians(angle_degrees)
        rotated_endpoint = [
            hinge[0] + radius * math.cos(angle),
            hinge[1] + radius * math.sin(angle),
        ]
        for obstacle in normalized_obstacles:
            minimum = obstacle["world_aabb_min_m"]
            maximum = obstacle["world_aabb_max_m"]
            vertical_overlap = min(vertical[1], maximum[2]) > max(vertical[0], minimum[2])
            if not vertical_overlap:
                continue
            inflated_minimum = [minimum[0] - half_thickness, minimum[1] - half_thickness]
            inflated_maximum = [maximum[0] + half_thickness, maximum[1] + half_thickness]
            intersects, point = _segment_aabb_intersection(
                hinge, rotated_endpoint, inflated_minimum, inflated_maximum
            )
            if not intersects:
                continue
            collision_obstacle_ids.add(obstacle["obstacle_id"])
            if first_collision is None:
                first_collision = {
                    "angle_degrees": round(angle_degrees, 9),
                    "obstacle_id": obstacle["obstacle_id"],
                    "intersection_xy_world_m": [round(value, 9) for value in point or []],
                    "rotated_endpoint_xy_world_m": [
                        round(value, 9) for value in rotated_endpoint
                    ],
                }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_by_observed_obstacle" if first_collision else "clearance_candidate_only",
        "sweep": {
            "hinge_origin_world_m": hinge,
            "closed_endpoint_world_m": endpoint,
            "member_radius_m": radius,
            "member_vertical_interval_m": vertical,
            "start_angle_degrees": start_angle,
            "end_angle_degrees": end_angle,
            "angular_resolution_degrees": resolution,
            "member_half_thickness_m": half_thickness,
            "sample_count": len(angles),
        },
        "obstacles": normalized_obstacles,
        "first_collision": first_collision,
        "collision_obstacle_ids": sorted(collision_obstacle_ids),
        "claim_boundary": {
            "zero_thickness_centerline_collision_is_strong_rejection": half_thickness == 0.0,
            "clear_result_is_not_native_dynamic_qualification": True,
            "franka_base_pose_resolved": False,
            "ik_or_contact_qualified": False,
        },
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    return result


def load_bound_collision_obstacle(path: str | Path) -> dict[str, Any]:
    """Load one exact SAGE identity receipt as a sweep obstacle."""

    source = Path(path).expanduser().resolve()
    try:
        receipt = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArticulatedWorkspaceClearanceError(["sweep_obstacle_receipt_invalid"]) from exc
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != "interiorgs_sage_collision_identity.v1":
        raise ArticulatedWorkspaceClearanceError(["sweep_obstacle_receipt_invalid"])
    if receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest"):
        raise ArticulatedWorkspaceClearanceError(["sweep_obstacle_receipt_digest_invalid"])
    matches = receipt.get("whole_object_matches")
    if not isinstance(matches, list) or len(matches) != 1:
        raise ArticulatedWorkspaceClearanceError(["sweep_obstacle_unique_collision_match_missing"])
    match = matches[0]
    target = receipt.get("target") or {}
    return {
        "obstacle_id": f"{target.get('semantic_label')}:{target.get('interiorgs_instance_id')}",
        "world_aabb_min_m": match["world_aabb_min_m"],
        "world_aabb_max_m": match["world_aabb_max_m"],
        "source_receipt_digest": receipt["receipt_digest"],
        "source_file_sha256": _sha256(source),
    }


def validate_articulated_workspace_clearance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a retained clearance receipt without rerunning geometry."""

    payload = json.loads(json.dumps(value))
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("sweep_clearance_schema_invalid")
    if payload.get("status") not in {
        "blocked_by_observed_obstacle",
        "clearance_candidate_only",
    }:
        errors.append("sweep_clearance_status_invalid")
    collision = payload.get("first_collision")
    if payload.get("status") == "blocked_by_observed_obstacle" and not isinstance(
        collision, Mapping
    ):
        errors.append("sweep_clearance_collision_missing")
    if payload.get("status") == "clearance_candidate_only" and collision is not None:
        errors.append("sweep_clearance_unexpected_collision")
    if payload.get("receipt_digest") != canonical_digest(
        payload, digest_field="receipt_digest"
    ):
        errors.append("sweep_clearance_digest_invalid")
    if errors:
        raise ArticulatedWorkspaceClearanceError(errors)
    return payload


__all__ = [
    "ArticulatedWorkspaceClearanceError",
    "SCHEMA_VERSION",
    "evaluate_revolute_member_sweep",
    "load_bound_collision_obstacle",
    "validate_articulated_workspace_clearance",
]
