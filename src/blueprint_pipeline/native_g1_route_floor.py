"""A G1 navigation packet's scanned floor along its route, as sampled heights only; never mesh bytes.

2026-11-09, the library job: after a re-posed splat its levelling fitted an observed floor (tilt 1.82°), and the
navigation start moved 1.93 m down the aisle. There the scanned floor stood 6.5-9.4 cm above the levelled floor and
sloped across the stance, so the spawn check read overlap and planning funded no G1 row ($0). Choosing a scene fix
(flatten the walk corridor, or fit the floor locally) needs the floor's profile between the start and the goal.

The packet's static collision (``scene_collision``, as the spawn check loads it) is sampled on CPU at fixed points:
along the straight line from the robot's start to the goal disc's centre every ``STEP_M``, at ``LATERAL_M`` either
side of each, and at the disc's centre and ``RIM_POINTS`` points on its rim. At each point the report gives the
highest scanned floor-like surface (upward normal within ``FLOOR_NORMAL_MIN_Z``) inside ``WINDOW_M`` of the levelled
floor, and whether the walked-floor estimate covers it. Heights are relative to the goal disc's plane, which the
packet puts on the levelled floor. The walked floor (``atlas_world_levelling._fill_walked_floor``) is told apart as
exactly horizontal faces at that height. Written once beside the packet, measured or not; never a gate or a flag.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest

#: v2 (2026-11-09): a face's normal is read either way up, as the spawn check reads it (``|n_z|``). v1 counted
#: only faces wound upward, and the library job's rebuilt mesh has floor faces wound downward: v1 found no scanned
#: floor at the start where the spawn check found it 6.5-8.5 cm above the levelled floor.
SCHEMA_VERSION = "native_g1_route_floor_profile.v2"
FILENAME = SCHEMA_VERSION + ".json"
STEP_M = 0.25
LATERAL_M = 0.3
RIM_POINTS = 8
#: A surface at most about 25° from level reads as floor-like, whichever way its face is wound
#: (``native_g1_spawn_clearance.FLOOR_NORMAL_MIN_Z``, read as ``|n_z|`` there too).
FLOOR_NORMAL_MIN_Z = 0.9
#: Heights searched, relative to the levelled floor: what a foot could stand on or sink under.
WINDOW_M = (-0.3, 0.5)
#: A walked-floor face is exactly horizontal, at the levelled floor's height.
WALKED_NORMAL_MIN_Z = 0.99999
WALKED_HEIGHT_TOLERANCE_M = 0.001
CLAIM_BOUNDARY = ("sampled_heights_only: the packet's static collision sampled on CPU at these points; heights are "
                  "relative to the goal disc's plane (the levelled floor); no vertex, face or name leaves the host; "
                  "no simulator ran")


def sample_points(start_xy: Sequence[float], goal_xy: Sequence[float], goal_radius_m: float) -> list[dict[str, Any]]:
    """The route's centre line every ``STEP_M`` (both ends included), ``LATERAL_M`` either side, then the goal disc."""
    start, goal = np.asarray(start_xy[:2], dtype=float), np.asarray(goal_xy[:2], dtype=float)
    length = float(np.linalg.norm(goal - start))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("g1_route_floor_route_invalid")
    forward = (goal - start) / length
    left = np.array([-forward[1], forward[0]])
    steps = int(math.ceil(length / STEP_M))
    rows = []
    for index in range(steps + 1):
        along = min(index * STEP_M, length)
        for lateral in (-LATERAL_M, 0.0, LATERAL_M):
            xy = start + forward * along + left * lateral
            rows.append({"id": f"route_{index:03d}", "kind": "route", "along_m": round(along, 3),
                         "lateral_m": lateral, "xy": xy})
    rows.append({"id": "goal_centre", "kind": "goal", "along_m": round(length, 3), "lateral_m": 0.0, "xy": goal})
    for index in range(RIM_POINTS):
        angle = 2.0 * math.pi * index / RIM_POINTS
        rows.append({"id": f"goal_rim_{index}", "kind": "goal", "along_m": round(length, 3), "lateral_m": None,
                     "xy": goal + goal_radius_m * np.array([math.cos(angle), math.sin(angle)])})
    return rows


def surface_heights(triangles: np.ndarray, points_xy: np.ndarray, floor_z: float) -> list[dict[str, Any]]:
    """At each xy point: the highest scanned floor-like surface in the window, its face count, and the walked floor."""
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    points_xy = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if len(triangles):
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        norms = np.linalg.norm(normals, axis=1)
        unit_z = np.abs(np.divide(normals[:, 2], norms, out=np.zeros(len(norms)), where=norms > 0))
        heights = triangles[:, :, 2] - floor_z
        keep = ((unit_z >= FLOOR_NORMAL_MIN_Z) & (heights.max(axis=1) >= WINDOW_M[0])
                & (heights.min(axis=1) <= WINDOW_M[1]))
        low, high = points_xy.min(axis=0), points_xy.max(axis=0)
        keep &= np.all(triangles[:, :, :2].max(axis=1) >= low, axis=1)
        keep &= np.all(triangles[:, :, :2].min(axis=1) <= high, axis=1)
        triangles, unit_z, heights = triangles[keep], unit_z[keep], heights[keep]
        walked = (unit_z >= WALKED_NORMAL_MIN_Z) & np.all(np.abs(heights) <= WALKED_HEIGHT_TOLERANCE_M, axis=1)
    else:
        walked = np.zeros(0, dtype=bool)
    a = triangles[:, 0, :2] if len(triangles) else np.zeros((0, 2))
    v0 = triangles[:, 2, :2] - a if len(triangles) else np.zeros((0, 2))
    v1 = triangles[:, 1, :2] - a if len(triangles) else np.zeros((0, 2))
    d00, d01, d11 = (v0 * v0).sum(axis=1), (v0 * v1).sum(axis=1), (v1 * v1).sum(axis=1)
    denominator = d00 * d11 - d01 * d01
    rows = []
    for point in points_xy:
        v2 = point - a
        d20, d21 = (v2 * v0).sum(axis=1), (v2 * v1).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (d11 * d20 - d01 * d21) / denominator
            v = (d00 * d21 - d01 * d20) / denominator
        covers = (np.abs(denominator) > 1e-12) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
        z = (triangles[:, 0, 2] + u * (triangles[:, 2, 2] - triangles[:, 0, 2])
             + v * (triangles[:, 1, 2] - triangles[:, 0, 2]) - floor_z) if len(triangles) else np.zeros(0)
        scanned = covers & ~walked & (z >= WINDOW_M[0]) & (z <= WINDOW_M[1])
        rows.append({"scanned_top_m": round(float(z[scanned].max()), 4) if scanned.any() else None,
                     "scanned_faces": int(scanned.sum()), "walked_floor": bool((covers & walked).any())})
    return rows


def _summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def span(rows):
        tops = [row["scanned_top_m"] for row in rows if row["scanned_top_m"] is not None]
        return {"min": min(tops), "max": max(tops), "points": len(tops)} if tops else None

    centre = [row for row in samples if row["kind"] == "route" and row["lateral_m"] == 0.0]
    goal = [row for row in samples if row["kind"] == "goal"]
    return {"route_centre_scanned_top_m": span(centre), "goal_scanned_top_m": span(goal),
            "points": len(samples),
            "points_scanned_above_2cm": sum(1 for row in samples if (row["scanned_top_m"] or 0.0) > 0.02),
            "points_with_no_scanned_floor": sum(1 for row in samples if row["scanned_top_m"] is None),
            "points_on_walked_floor": sum(1 for row in samples if row["walked_floor"])}


def route_floor_profile(packet_dir: str | Path) -> dict[str, Any] | None:
    """The packet's route floor profile; None when its plan has no G1 navigation goal. Raises on unreadable input."""
    from .native_g1_spawn_clearance import _bound_file, _pose_matrix
    from .scene_physics_preflight import load_background_triangles

    root = Path(packet_dir)
    plan = json.loads((root / "native_task_arena_scene_plan.v1.json").read_text(encoding="utf-8"))
    goal = (plan.get("task_spec") or {}).get("g1_navigation_goal")
    if not isinstance(goal, Mapping):
        return None
    start = [float(v) for v in plan["robot"]["base_pose_world"]["position_world_m"]]
    centre = [float(v) for v in goal["center_world_m"]]
    radius = float((goal.get("visible_target_marker") or {}).get("radius_m") or goal["acceptance_radius_m"])
    if not all(math.isfinite(v) for v in [*start, *centre, radius]):
        raise ValueError("g1_route_floor_route_invalid")
    points = sample_points(start, centre, radius)
    xy = np.asarray([row["xy"] for row in points])
    middle = xy.mean(axis=0)
    reach = float(np.linalg.norm(xy - middle, axis=1).max()) + 0.5
    collisions = [row for row in plan["objects"] if row.get("semantic_role") == "scene_collision"]
    background = _bound_file(root, collisions[0]["usd_path"], collisions[0]["sha256"])
    pose = collisions[0].get("pose_world")
    triangles = load_background_triangles(background, centre=[*middle, centre[2]], radius=reach,
                                          world_from_stage=None if pose is None else _pose_matrix(pose))
    heights = surface_heights(triangles, xy, centre[2])
    samples = [{key: row[key] for key in ("id", "kind", "along_m", "lateral_m")} | height
               for row, height in zip(points, heights)]
    value = {"schema_version": SCHEMA_VERSION, "status": "measured", "reason": None,
             "scene_plan_digest": plan.get("plan_digest"), "floor_z_world_m": round(centre[2], 6),
             "start_xy_m": [round(v, 4) for v in start[:2]], "goal_xy_m": [round(v, 4) for v in centre[:2]],
             "goal_radius_m": radius, "route_length_m": round(float(np.linalg.norm(xy[-1 - RIM_POINTS] - start[:2])), 3),
             "step_m": STEP_M, "lateral_m": LATERAL_M, "floor_normal_min_z": FLOOR_NORMAL_MIN_Z,
             "window_m": list(WINDOW_M), "background_triangles_reached": int(len(triangles)),
             "samples": samples, "summary": _summary(samples), "claim_boundary": CLAIM_BOUNDARY,
             "simulator_executed": False, "provider_mutation_performed": False}
    value["report_digest"] = canonical_digest(value, digest_field="report_digest")
    return value


def ensure_packet_route_floor_profile(packet_dir: str | Path) -> dict[str, Any] | None:
    """Write the packet's route floor profile once beside it, measured or unmeasured; never raises.

    An unreadable input is written as ``unmeasured`` with its reason, so a pass never loads the room again for it.
    None for a packet with no G1 navigation goal, or when nothing could be written.
    """
    from .deployment_embodiment_preparation_contract import write_once
    from .native_g1_spawn_clearance import _code

    path = Path(packet_dir) / FILENAME
    try:
        if path.is_file() and not path.is_symlink():
            return json.loads(path.read_text(encoding="utf-8"))
        try:
            value = route_floor_profile(packet_dir)
        except Exception as exc:  # noqa: BLE001 - an unreadable input is recorded by name
            value = {"schema_version": SCHEMA_VERSION, "status": "unmeasured",
                     "reason": _code(exc, "g1_route_floor_unreadable"), "claim_boundary": CLAIM_BOUNDARY,
                     "simulator_executed": False, "provider_mutation_performed": False}
            value["report_digest"] = canonical_digest(value, digest_field="report_digest")
        if value is not None:
            write_once(path, value)
        return value
    except Exception:  # noqa: BLE001 - evidence for a scene decision never blocks anything
        return None


__all__ = ["FILENAME", "SCHEMA_VERSION", "ensure_packet_route_floor_profile", "route_floor_profile",
           "sample_points", "surface_heights"]
