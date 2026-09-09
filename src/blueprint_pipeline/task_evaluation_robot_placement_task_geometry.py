"""Source-bound task-object volumes for provisional robot placement screening."""
from __future__ import annotations

from collections.abc import Mapping
from itertools import product
import math
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA = "task_evaluation_robot_placement_task_occupancy.v1"


def _vector(value: Any, count: int) -> list[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != count
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in value)):
        raise ValueError("robot_placement_task_geometry_invalid")
    return [float(v) for v in value]


def _world_bounds(bounds: Mapping[str, Any], pose: list[float]) -> dict[str, Any]:
    lower, upper = _vector(bounds.get("minimum"), 3), _vector(bounds.get("maximum"), 3)
    if any(a >= b for a, b in zip(lower, upper, strict=True)):
        raise ValueError("robot_placement_task_geometry_invalid")
    pose = _vector(pose, 7)
    x, y, z, w = pose[3:]
    if not math.isclose(x*x+y*y+z*z+w*w, 1., abs_tol=1e-6):
        raise ValueError("robot_placement_task_geometry_invalid")
    corners = []
    for a, b, c in product(*zip(lower, upper, strict=True)):
        tx, ty, tz = 2*(y*c-z*b), 2*(z*a-x*c), 2*(x*b-y*a)
        corners.append([pose[0]+a+w*tx+y*tz-z*ty,
                        pose[1]+b+w*ty+z*tx-x*tz,
                        pose[2]+c+w*tz+x*ty-y*tx])
    return {"minimum": [min(p[i] for p in corners) for i in range(3)],
            "maximum": [max(p[i] for p in corners) for i in range(3)]}


def task_occupancy_from_native_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    bounds = plan.get("subject_collision_bounds_scoring_frame_m")
    required = plan.get("task_occupancy_required") is True
    if bounds is None and required:
        raise ValueError("robot_placement_required_task_collision_bounds_missing")
    regions = []
    if bounds is not None:
        if not isinstance(bounds, Mapping):
            raise ValueError("robot_placement_task_geometry_invalid")
        start = _vector(plan.get("start_scoring_pose_world"), 7)
        destination = [*_vector(plan.get("destination_position_world_m"), 3),
                       *_vector(plan.get("destination_orientation_xyzw"), 4)]
        for name, pose in (("subject_start", start), ("subject_destination", destination)):
            regions.append({"region_id": name, "bounds_world_m": _world_bounds(bounds, pose),
                            "source_scoring_pose_world": pose})
    result = {"schema_version": SCHEMA, "status": "available" if regions else "unavailable",
        "required": required, "source_plan_digest": plan["plan_digest"],
        "subject_asset_id": plan.get("subject_asset_id"), "regions": regions,
        "measurement_source": "source_collision_bounds_transformed_by_authored_task_poses",
        "native_qualification_claimed": False}
    result["occupancy_digest"] = canonical_digest(result, digest_field="occupancy_digest")
    return validate_task_occupancy(result)


def validate_task_occupancy(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if (result.get("schema_version") != SCHEMA or result.get("native_qualification_claimed") is not False
            or result.get("occupancy_digest") != canonical_digest(result, digest_field="occupancy_digest")
            or result.get("status") not in {"available", "unavailable"}):
        raise ValueError("robot_placement_task_occupancy_invalid")
    regions = result.get("regions")
    if (not isinstance(regions, list) or (result["status"] == "available") != bool(regions)
            or any(not isinstance(region, Mapping) for region in regions)
            or (result.get("required") is True and not regions)):
        raise ValueError("robot_placement_task_occupancy_invalid")
    if regions and [r.get("region_id") for r in regions] != ["subject_start", "subject_destination"]:
        raise ValueError("robot_placement_task_occupancy_invalid")
    for region in regions:
        bounds = region.get("bounds_world_m")
        if not isinstance(bounds, Mapping):
            raise ValueError("robot_placement_task_occupancy_invalid")
        lower, upper = _vector(bounds.get("minimum"), 3), _vector(bounds.get("maximum"), 3)
        if any(a >= b for a, b in zip(lower, upper, strict=True)):
            raise ValueError("robot_placement_task_occupancy_invalid")
    return result
