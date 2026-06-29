#!/usr/bin/env python3
"""Isaac Sim GPU runner: MuJoCo-parity G1 walk-to-target eval in the sim-ready Lightwheel kitchen.

Runs inside Isaac's python (``/isaac-sim/python.sh``) on the GPU worker. Self-contained: the
policy module ``isaac_g1_policy.py`` is shipped alongside this script in the bundle. Per
navigation scenario it drives the SAME deterministic walk-to-target controller as the MuJoCo
lane — proposing collision-checked candidate root poses, probing each via a PhysX overlap
query, kinematically placing the G1, and RTX-rendering overview + robot-POV frames into MP4 —
and records the MuJoCo-schema trace. Emits ``isaac_g1_kitchen_parity_result.json`` (same
task-outcome contract) + traces + MP4s and uploads the out dir via the provider signed-PUT.

Honesty boundary: Stage A is a *kinematic* navigation preview (parity with MuJoCo's preview
controller), RTX-rendered on Isaac. It is not dynamic locomotion and not a learned policy; the
GR00T N1.7 SONIC stage swaps the policy (``--policy groot_sonic``) without changing this harness.

The Isaac-API calls (boot, stage, PhysX overlap, Replicator render) are GPU-only and verified
on the worker, not locally; the non-Isaac helpers are unit-tested in the repo.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence


def _log(msg: str) -> None:
    """Flushed, timestamped progress line so the heartbeat-uploaded console shows exactly how
    far the runner got (Isaac ops between scene-load and render give no output otherwise)."""
    print(f"[parity {time.strftime('%H:%M:%S')}] {msg}", flush=True)

# --- policy import: bundle dir on the worker, package in the repo (tests) ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import isaac_g1_policy as policy_mod  # bundle (worker)
except Exception:  # noqa: BLE001
    from blueprint_pipeline import isaac_g1_policy as policy_mod  # repo (tests)

RESULT_SCHEMA_VERSION = "isaac_g1_kitchen_parity_result.v1"
# robot footprint half-extent (m) for the PhysX overlap probe (approx G1 standing bbox)
ROBOT_FOOTPRINT_HALF_EXTENT = (0.28, 0.28, 0.62)
ROBOT_PELVIS_HEIGHT_M = 0.79
PLACEMENT_GROUND_TRUTH_MAX_FOOTPRINT_CENTER_DELTA_M = 0.10
TASK_STANCE_SCHEMA_VERSION = "task_stance_plan.v1"
TASK_STANCE_TARGET_KEYS = (
    "task_target_position_xyz",
    "manipulation_target_position_xyz",
    "target_object_position_xyz",
    "look_at_position_xyz",
)
TASK_STANCE_TARGET_BBOX_KEY_PAIRS = (
    ("task_target_bbox_min_xyz", "task_target_bbox_max_xyz"),
    ("target_object_bbox_min_xyz", "target_object_bbox_max_xyz"),
)
TASK_STANCE_FALLBACK_TARGET_KEYS = ("raw_target_position_xyz", "target")
TASK_STANCE_TARGET_OBJECT_KEYS = ("target_object_id", "task_target_object_id", "object_id")
TASK_STANCE_APPROACH_KEYS = (
    "approach_position_xyz",
    "robot_start_position_xyz",
    "raw_spawn_position_xyz",
    "start",
)
TASK_STANCE_ANGLE_OFFSETS_DEG = (0, -15, 15, -30, 30, -45, 45, -60, 60, -90, 90, -120, 120, 180)
TASK_STANCE_DEFAULT_SURFACE_STANDOFF_M = 0.85
TASK_STANCE_DEFAULT_VALIDATION_STANDOFF_RANGE_M = (0.4, 1.2)
TASK_STANCE_CLOSE_REACH_TARGET_TOKENS = (
    "door",
    "drawer",
    "handle",
    "hatch",
    "lid",
    "panel",
    "cabinet",
    "cupboard",
    "refrigerator",
    "fridge",
    "freezer",
    "dishwasher",
    "oven",
    "microwave",
    "washer",
    "dryer",
)
TASK_STANCE_CLOSE_REACH_ACTION_TOKENS = (
    "open",
    "close",
    "pull",
    "push",
    "slide",
    "grasp",
    "grab",
    "reach",
    "turn",
    "unlatch",
    "latch",
)
TASK_STANCE_CLOSE_REACH_GAP_RANGE_M = (0.08, 0.72)
MANIPULATION_READY_ARM_SELECTIONS = ("right", "left", "both")
MANIPULATION_READY_ARM_JOINT_DELTAS = {
    "left": {
        "left_shoulder_pitch_joint": -0.85,
        "left_shoulder_roll_joint": 0.15,
        "left_shoulder_yaw_joint": 0.10,
        "left_elbow_joint": -0.23,
        "left_wrist_roll_joint": -0.10,
        "left_wrist_pitch_joint": -0.15,
    },
    "right": {
        "right_shoulder_pitch_joint": -0.85,
        "right_shoulder_roll_joint": -0.15,
        "right_shoulder_yaw_joint": -0.10,
        "right_elbow_joint": -0.23,
        "right_wrist_roll_joint": 0.10,
        "right_wrist_pitch_joint": -0.15,
    },
}


# ============================ testable helpers (no isaacsim) ============================

def load_request(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_scenarios(request: Mapping[str, Any]) -> list[dict]:
    """Normalize scenarios to {scenario_id, route_points:[[x,y,z],...], start, target, instruction}.
    Accepts explicit route_points, spawn_position_xyz + target_position_xyz, or a task-only
    manipulation scenario whose target is resolved later from the loaded scene."""
    out: list[dict] = []
    for raw in request.get("scenarios", []) or []:
        sid = str(raw.get("scenario_id") or raw.get("id") or f"scenario_{len(out)+1}")
        instruction = str(raw.get("instruction") or raw.get("description") or "").strip()
        route = raw.get("route_points") or raw.get("waypoints")
        start = raw.get("spawn_position_xyz") or raw.get("start") or (route[0] if route else None)
        target = raw.get("target_position_xyz") or raw.get("target") or (route[-1] if route else None)
        if start is None or target is None:
            if not instruction:
                continue
            out.append({
                "scenario_id": sid,
                "route_points": [],
                "instruction": instruction,
                "scenario_eval_run_id": raw.get("scenario_eval_run_id"),
                "floor_z_hint": float(raw.get("floor_z_hint", 0.05)),
                "task_target_deferred": True,
                "deferred_task_resolution": "scene_placement_task_label",
            })
            for key in (
                "task_target_position_xyz",
                "manipulation_target_position_xyz",
                "target_object_position_xyz",
                "look_at_position_xyz",
                "approach_position_xyz",
                "robot_start_position_xyz",
                "stance_distance_candidates_m",
                "preferred_stance_distance_m",
                "min_stance_distance_m",
                "max_stance_distance_m",
                "target_object_id",
                "task_target_object_id",
                "object_id",
                "target_object_label",
                "task_target_object_label",
                "task",
                "task_description",
                "description",
                "task_instruction",
            ):
                if key in raw:
                    out[-1][key] = raw[key]
            continue
        raw_start = [float(c) for c in start]
        raw_target = [float(c) for c in target]
        start = [float(c) for c in start]
        target = [float(c) for c in target]
        if not route:
            route = [start, target]
        route = [[float(c) for c in p] for p in route]
        # lift the navigation route to pelvis height so the root trace is realistic
        route = [[p[0], p[1], ROBOT_PELVIS_HEIGHT_M] for p in route]
        out.append({
            "scenario_id": sid,
            "route_points": route,
            "start": [start[0], start[1], ROBOT_PELVIS_HEIGHT_M],
            "target": [target[0], target[1], ROBOT_PELVIS_HEIGHT_M],
            "instruction": instruction,
            "scenario_eval_run_id": raw.get("scenario_eval_run_id"),
            "raw_spawn_position_xyz": raw_start,
            "raw_target_position_xyz": raw_target,
            "floor_z_hint": float(raw.get("floor_z_hint", raw_start[2] if len(raw_start) > 2 else 0.0)),
        })
        for key in (
            "task_target_position_xyz",
            "manipulation_target_position_xyz",
            "target_object_position_xyz",
            "look_at_position_xyz",
            "approach_position_xyz",
            "robot_start_position_xyz",
            "stance_distance_candidates_m",
            "preferred_stance_distance_m",
            "min_stance_distance_m",
            "max_stance_distance_m",
            "target_object_id",
            "task_target_object_id",
            "object_id",
        ):
            if key in raw:
                out[-1][key] = raw[key]
    return out


def _optional_xyz(value) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        vals = [float(v) for v in value]
    except Exception:  # noqa: BLE001
        return None
    if len(vals) < 2:
        return None
    if len(vals) == 2:
        vals.append(ROBOT_PELVIS_HEIGHT_M)
    return (vals[0], vals[1], vals[2])


def _target_bounds_for_scenario(
    scenario: Mapping[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    for min_key, max_key in TASK_STANCE_TARGET_BBOX_KEY_PAIRS:
        bbox_min = _optional_xyz(scenario.get(min_key))
        bbox_max = _optional_xyz(scenario.get(max_key))
        if bbox_min is None or bbox_max is None:
            continue
        if any(bbox_max[i] < bbox_min[i] for i in range(3)):
            continue
        return bbox_min, bbox_max
    return None


def _target_bounds_for_stance_plan(
    stance_plan: Mapping[str, Any] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not stance_plan:
        return None
    bounds = stance_plan.get("task_target_bounds")
    if isinstance(bounds, Mapping):
        bbox_min = _optional_xyz(bounds.get("bbox_min_xyz"))
        bbox_max = _optional_xyz(bounds.get("bbox_max_xyz"))
        if bbox_min is not None and bbox_max is not None:
            return bbox_min, bbox_max
    return _target_bounds_for_scenario(stance_plan)


def _surface_affordance_point_for_stance(
    stance_plan: Mapping[str, Any] | None,
    root_pose: Sequence[float] | None,
) -> tuple[float, float, float] | None:
    """Target face nearest the stance, for camera/reach focus.

    Object resolution should return the whole object centroid, but manipulation needs the reachable
    surface, not a point inside an appliance/cabinet volume. This projects the resolved target point
    onto the bbox face nearest the robot. It is scene-agnostic: no kitchen or object coordinates.
    """
    if not stance_plan or root_pose is None:
        return None
    target = _optional_xyz(stance_plan.get("task_target_xyz"))
    if target is None:
        return None
    bounds = _target_bounds_for_stance_plan(stance_plan)
    if bounds is None:
        return target
    bbox_min, bbox_max = bounds
    cx, cy, cz = target
    dx = float(root_pose[0]) - cx
    dy = float(root_pose[1]) - cy
    x = min(max(cx, bbox_min[0]), bbox_max[0])
    y = min(max(cy, bbox_min[1]), bbox_max[1])
    z = min(max(cz, bbox_min[2]), bbox_max[2])
    if abs(dx) >= abs(dy):
        x = bbox_max[0] if dx >= 0.0 else bbox_min[0]
    else:
        y = bbox_max[1] if dy >= 0.0 else bbox_min[1]
    return (float(x), float(y), float(z))


def _half_extent_along_bounds(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    direction_xy: tuple[float, float],
) -> float:
    if bounds is None:
        return 0.0
    bbox_min, bbox_max = bounds
    hx = max(0.0, 0.5 * (bbox_max[0] - bbox_min[0]))
    hy = max(0.0, 0.5 * (bbox_max[1] - bbox_min[1]))
    ux, uy = direction_xy
    return abs(float(ux)) * hx + abs(float(uy)) * hy


def _xy_rect_overlap_and_gap(
    a_min: Sequence[float],
    a_max: Sequence[float],
    b_min: Sequence[float],
    b_max: Sequence[float],
) -> dict[str, float | bool]:
    """2D rectangle relation in the scene floor plane.

    This is deliberately pure so placement validation can be tested without Isaac. ``overlap_area``
    catches visual/geometry interpenetration; ``gap_m`` catches near-zero clearance when the boxes do
    not overlap.
    """
    overlap_x = min(float(a_max[0]), float(b_max[0])) - max(float(a_min[0]), float(b_min[0]))
    overlap_y = min(float(a_max[1]), float(b_max[1])) - max(float(a_min[1]), float(b_min[1]))
    overlaps = overlap_x > 0.0 and overlap_y > 0.0
    if overlaps:
        gap = 0.0
        area = overlap_x * overlap_y
    else:
        sep_x = max(float(b_min[0]) - float(a_max[0]), float(a_min[0]) - float(b_max[0]), 0.0)
        sep_y = max(float(b_min[1]) - float(a_max[1]), float(a_min[1]) - float(b_max[1]), 0.0)
        gap = math.hypot(sep_x, sep_y)
        area = 0.0
    return {
        "overlaps_xy": bool(overlaps),
        "overlap_area_xy_m2": round(float(area), 6),
        "gap_m": round(float(gap), 6),
    }


def _placement_validation_passed(validation: Mapping[str, Any] | None) -> bool:
    if validation is None:
        return True
    blockers = validation.get("blockers")
    if isinstance(blockers, Sequence) and not isinstance(blockers, (str, bytes)) and blockers:
        return False
    return str(validation.get("status") or "accepted").lower() in {
        "accepted",
        "passed",
        "valid",
        "clear",
        "ok",
    }


def _task_stance_angle_priority(offset_deg: float) -> int:
    """Bucket placement rays by approach fidelity before falling back to backside poses."""
    offset = abs(float(offset_deg))
    if offset <= 45.0:
        return 0
    if offset <= 90.0:
        return 1
    if offset <= 120.0:
        return 2
    return 3


def _task_stance_selection_key(record: Mapping[str, Any]) -> tuple[float, float, float]:
    """Sort key for accepted task-stance candidates.

    Candidate generation tries near distances before far distances, but an accepted pose on the
    opposite side of the target can appear before a farther pose on the intended approach ray.
    This key preserves the useful near-first behavior inside an approach bucket while preventing
    a backside/through-target pose from beating a farther room-side stance.
    """
    try:
        offset = abs(float(record.get("angle_offset_deg", 180.0)))
    except Exception:  # noqa: BLE001
        offset = 180.0
    try:
        distance = float(record.get("standoff_from_target_surface_m", float("inf")))
    except Exception:  # noqa: BLE001
        distance = float("inf")
    if not bool(record.get("approach_bias_enabled")):
        return (
            0.0,
            distance,
            offset,
        )
    return (
        float(_task_stance_angle_priority(offset)),
        distance,
        offset,
    )


def _rounded_xyz(value: Sequence[float], *, digits: int = 6) -> list[float]:
    return [round(float(value[i]), digits) for i in range(3)]


def _xy_focus_overlap(
    obj: Any,
    focus_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    *,
    margin_m: float,
) -> bool:
    if focus_bounds is None:
        return True
    try:
        bbox_min = getattr(obj, "bbox_min")
        bbox_max = getattr(obj, "bbox_max")
        focus_min, focus_max = focus_bounds
        return not (
            float(bbox_max[0]) < float(focus_min[0]) - float(margin_m)
            or float(bbox_min[0]) > float(focus_max[0]) + float(margin_m)
            or float(bbox_max[1]) < float(focus_min[1]) - float(margin_m)
            or float(bbox_min[1]) > float(focus_max[1]) + float(margin_m)
        )
    except Exception:  # noqa: BLE001
        return True


def _aligned_box_min_max_center_size(box) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return min/max/center/size for USD aligned boxes across pxr API variants.

    ``Gf.Range3d`` exposes ``GetMin``/``GetMax``/``GetSize`` but not ``GetCenter`` in the Isaac
    worker build. Some local fakes and older code paths expose ``GetCenter``/``GetSize`` only. Keep
    both forms supported so placement validation cannot fail before it evaluates the actual geometry.
    """
    get_min = getattr(box, "GetMin", None)
    get_max = getattr(box, "GetMax", None)
    if callable(get_min) and callable(get_max):
        bmin_raw = get_min()
        bmax_raw = get_max()
        bmin = [float(bmin_raw[i]) for i in range(3)]
        bmax = [float(bmax_raw[i]) for i in range(3)]
        center = [0.5 * (bmin[i] + bmax[i]) for i in range(3)]
        size = [bmax[i] - bmin[i] for i in range(3)]
        return bmin, bmax, center, size
    center_raw = box.GetCenter()
    size_raw = box.GetSize()
    center = [float(center_raw[i]) for i in range(3)]
    size = [float(size_raw[i]) for i in range(3)]
    bmin = [center[i] - 0.5 * size[i] for i in range(3)]
    bmax = [center[i] + 0.5 * size[i] for i in range(3)]
    return bmin, bmax, center, size


def _with_xyz(scenario: Mapping[str, Any], key: str, xyz: Sequence[float]) -> dict[str, Any]:
    out = dict(scenario)
    out[key] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    return out


def _unit_xy_from(a: Sequence[float], b: Sequence[float]) -> tuple[float, float] | None:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return None
    return (dx / mag, dy / mag)


def _target_text_for_semantic_stance(scenario: Mapping[str, Any] | None) -> str:
    if not scenario:
        return ""
    values: list[str] = []
    for key in (
        "target_object_id",
        "task_target_object_id",
        "target_object_label",
        "task_target_object_label",
        "object_id",
        "prim_path",
    ):
        value = scenario.get(key)
        if value is not None:
            values.append(str(value))
    selected = scenario.get("selected")
    if isinstance(selected, Mapping):
        for key in ("target_object_id", "target_object_label", "prim_path"):
            value = selected.get(key)
            if value is not None:
                values.append(str(value))
    return " ".join(values).replace("_", " ").replace("-", " ").lower()


def _task_text_for_semantic_stance(scenario: Mapping[str, Any] | None) -> str:
    if not scenario:
        return ""
    values: list[str] = []
    for key in (
        "instruction",
        "task",
        "task_description",
        "description",
        "task_instruction",
    ):
        value = scenario.get(key)
        if value is not None:
            values.append(str(value))
    return " ".join(values).replace("_", " ").replace("-", " ").lower()


def _is_close_reach_task_target(scenario: Mapping[str, Any] | None) -> bool:
    """True when task intent plus target identity imply a close hand-on-surface stance.

    This is semantic and scene-agnostic: the target still comes from scene resolution, and the final
    pose still comes from bounds/clearance validation. The classifier only switches the reach envelope
    from counter-working distance to closer articulated-surface distance.
    """
    task_text = _task_text_for_semantic_stance(scenario)
    target_text = _target_text_for_semantic_stance(scenario)
    if not task_text or not target_text:
        return False
    has_action = any(token in task_text for token in TASK_STANCE_CLOSE_REACH_ACTION_TOKENS)
    has_target = any(token in target_text for token in TASK_STANCE_CLOSE_REACH_TARGET_TOKENS)
    return bool(has_action and has_target)


def _close_reach_surface_standoff_candidates() -> list[float]:
    half_xy = max(float(ROBOT_FOOTPRINT_HALF_EXTENT[0]), float(ROBOT_FOOTPRINT_HALF_EXTENT[1]))
    clearances = (
        max(0.12, half_xy * 0.40),
        max(0.18, half_xy * 0.65),
        max(0.27, half_xy * 0.95),
        max(0.40, half_xy * 1.40),
        max(0.55, half_xy * 2.00),
    )
    defaults = (TASK_STANCE_DEFAULT_SURFACE_STANDOFF_M, TASK_STANCE_DEFAULT_SURFACE_STANDOFF_M * 1.25)
    return sorted({round(half_xy + float(gap), 4) for gap in clearances + defaults})


def _validation_standoff_range_for_scenario(
    scenario: Mapping[str, Any] | None,
) -> tuple[float, float]:
    if _is_close_reach_task_target(scenario):
        return TASK_STANCE_CLOSE_REACH_GAP_RANGE_M
    return TASK_STANCE_DEFAULT_VALIDATION_STANDOFF_RANGE_M


def task_stance_distance_candidates(scenario: Mapping[str, Any] | None = None) -> list[float]:
    """Generic robot-profile stance distances around a task target.

    The defaults are derived from the robot footprint, not from a kitchen/sink coordinate. Scenario
    metadata can override them when a task compiler knows a better reach/standoff envelope.
    """
    scenario = scenario or {}
    raw = scenario.get("stance_distance_candidates_m")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        distances = sorted({round(float(v), 4) for v in raw if float(v) > 0.0})
    elif _is_close_reach_task_target(scenario):
        distances = _close_reach_surface_standoff_candidates()
    else:
        half_xy = max(float(ROBOT_FOOTPRINT_HALF_EXTENT[0]), float(ROBOT_FOOTPRINT_HALF_EXTENT[1]))
        base = max(TASK_STANCE_DEFAULT_SURFACE_STANDOFF_M, half_xy * 3.0)
        # Keep the default samples dense around the useful counter-working band. In the Lightwheel
        # kitchen, the first non-clipping centered sink stance can be only a few centimeters farther
        # out than the closest sampled point, while the next coarse jump is already too far to reach.
        distances = [round(base * scale, 4) for scale in (1.0, 1.25, 1.55, 1.65, 1.9, 2.3)]
    min_d = scenario.get("min_stance_distance_m")
    max_d = scenario.get("max_stance_distance_m")
    if min_d is not None:
        distances = [d for d in distances if d >= float(min_d)]
    if max_d is not None:
        distances = [d for d in distances if d <= float(max_d)]
    preferred = scenario.get("preferred_stance_distance_m")
    if preferred is not None:
        pref = float(preferred)
        distances.sort(key=lambda d: (abs(d - pref), d))
    return distances


def task_stance_target_for_scenario(
    scenario: Mapping[str, Any],
    manipulation_look_at=None,
    *,
    allow_navigation_target_fallback: bool = True,
) -> tuple[float, float, float] | None:
    if manipulation_look_at is not None:
        return _optional_xyz(manipulation_look_at)
    for key in TASK_STANCE_TARGET_KEYS:
        target = _optional_xyz(scenario.get(key))
        if target is not None:
            return target
    if allow_navigation_target_fallback:
        for key in TASK_STANCE_FALLBACK_TARGET_KEYS:
            target = _optional_xyz(scenario.get(key))
            if target is not None:
                return target
    return None


def plan_task_stance(
    *,
    scenario: Mapping[str, Any],
    manipulation_look_at=None,
    probe_collision=None,
    floor_z_hint: float | None = None,
    robot_footprint_half_extent: Sequence[float] = ROBOT_FOOTPRINT_HALF_EXTENT,
    placement_validator=None,
) -> dict[str, Any]:
    """Select a task start pose around a target object without scene-specific coordinates.

    The target is the object/workspace to face, not the root pose. Candidate root poses are sampled
    around the target, biased toward the scenario's approach/start side, and accepted only when the
    provided scene collision probe reports no hits. When object bounds are available, stance
    distances are measured from the target footprint surface, not the center point; this prevents a
    robot from being accepted visually inside a sink/counter just because the pelvis center is a
    short radius from a small faucet or basin centroid.
    """
    target = task_stance_target_for_scenario(scenario, manipulation_look_at)
    if target is None:
        return {
            "schema_version": TASK_STANCE_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["missing_task_stance_target"],
            "candidates": [],
        }
    approach = None
    for key in TASK_STANCE_APPROACH_KEYS:
        approach = _optional_xyz(scenario.get(key))
        if approach is not None:
            break
    primary = _unit_xy_from(approach, target) if approach is not None else None
    if primary is None:
        primary = (-1.0, 0.0)
    primary_angle = math.atan2(primary[1], primary[0])
    target_bounds = _target_bounds_for_scenario(scenario)
    floor_z = float(
        floor_z_hint
        if floor_z_hint is not None
        else scenario.get("floor_z_hint", 0.0)
    )
    root_z = floor_z + ROBOT_PELVIS_HEIGHT_M
    distances = task_stance_distance_candidates(scenario)
    if not distances:
        return {
            "schema_version": TASK_STANCE_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["empty_task_stance_distance_candidates"],
            "task_target_xyz": [round(float(v), 6) for v in target],
            "candidates": [],
        }
    probe = probe_collision or (lambda pose, yaw: 0)
    candidates: list[dict[str, Any]] = []
    accepted_candidate_indices: list[int] = []
    rejected_by_placement_validation = 0
    for distance_m in distances:
        for offset_deg in TASK_STANCE_ANGLE_OFFSETS_DEG:
            angle = primary_angle + math.radians(float(offset_deg))
            ux, uy = math.cos(angle), math.sin(angle)
            target_surface_offset = _half_extent_along_bounds(target_bounds, (ux, uy))
            center_distance_m = target_surface_offset + float(distance_m)
            pose = (
                float(target[0]) + ux * center_distance_m,
                float(target[1]) + uy * center_distance_m,
                root_z,
            )
            yaw = math.atan2(float(target[1]) - pose[1], float(target[0]) - pose[0])
            collision_count = int(probe(pose, yaw))
            record = {
                "candidate_kind": "task_stance",
                "pose": [round(float(v), 6) for v in pose],
                "yaw": round(float(yaw), 6),
                "distance_to_target_m": round(float(center_distance_m), 6),
                "standoff_from_target_surface_m": round(float(distance_m), 6),
                "target_surface_offset_m": round(float(target_surface_offset), 6),
                "angle_offset_deg": int(offset_deg),
                "approach_bias_enabled": bool(approach is not None),
                "scene_collision_contact_count": collision_count,
            }
            candidates.append(record)
            if collision_count == 0:
                if placement_validator is not None:
                    try:
                        validation = placement_validator(pose, yaw, record)
                    except Exception as exc:  # noqa: BLE001 - fail closed on validator failures
                        validation = {
                            "status": "blocked",
                            "blockers": ["placement_validator_error"],
                            "error": repr(exc),
                        }
                    record["placement_validation"] = validation
                    if not _placement_validation_passed(validation):
                        rejected_by_placement_validation += 1
                        continue
                accepted_candidate_indices.append(len(candidates) - 1)
    if accepted_candidate_indices:
        selected_candidate_index = min(
            accepted_candidate_indices,
            key=lambda idx: (_task_stance_selection_key(candidates[idx]), idx),
        )
        record = candidates[selected_candidate_index]
        accepted = {
            "schema_version": TASK_STANCE_SCHEMA_VERSION,
            "status": "accepted",
            "task_target_xyz": [round(float(v), 6) for v in target],
            "approach_point_xyz": (
                [round(float(v), 6) for v in approach] if approach is not None else None
            ),
            "robot_footprint_half_extent": [round(float(v), 6) for v in robot_footprint_half_extent],
            "floor_z_hint": round(floor_z, 6),
            "accepted_pose": record["pose"],
            "accepted_yaw": record["yaw"],
            "selected_candidate_index": selected_candidate_index,
            "accepted_candidate_count": len(accepted_candidate_indices),
            "stance_selection_key": [
                round(float(v), 6) for v in _task_stance_selection_key(record)
            ],
            "stance_selection_strategy": (
                "validated candidates are sorted by standoff distance when no real approach "
                "hint exists; otherwise by approach-angle bucket, standoff distance, then "
                "absolute angle offset"
            ),
            "candidates": candidates,
            "claim_boundary": (
                "Task stance is selected from scene collision probes around the task target. "
                "It is placement evidence, not full dynamic locomotion or manipulation success."
            ),
        }
        if target_bounds is not None:
            accepted["task_target_bounds"] = {
                "bbox_min_xyz": [round(float(v), 6) for v in target_bounds[0]],
                "bbox_max_xyz": [round(float(v), 6) for v in target_bounds[1]],
            }
        if "placement_validation" in record:
            accepted["placement_validation"] = record["placement_validation"]
        return accepted
    blocked = {
        "schema_version": TASK_STANCE_SCHEMA_VERSION,
        "status": "blocked",
        "blockers": [
            "no_validated_task_stance_candidate"
            if rejected_by_placement_validation
            else "no_collision_free_task_stance_candidate"
        ],
        "task_target_xyz": [round(float(v), 6) for v in target],
        "approach_point_xyz": [round(float(v), 6) for v in approach] if approach is not None else None,
        "robot_footprint_half_extent": [round(float(v), 6) for v in robot_footprint_half_extent],
        "floor_z_hint": round(floor_z, 6),
        "candidates": candidates,
        "placement_validation_rejected_candidate_count": rejected_by_placement_validation,
    }
    if target_bounds is not None:
        blocked["task_target_bounds"] = {
            "bbox_min_xyz": [round(float(v), 6) for v in target_bounds[0]],
            "bbox_max_xyz": [round(float(v), 6) for v in target_bounds[1]],
        }
    return blocked


def assemble_collision_summary(*, actions: Sequence[Mapping[str, Any]],
                               rejected_probe_total: int, response_event_total: int) -> dict:
    """Build the collision_summary that compute_task_outcome consumes from the per-step trace."""
    committed = sum(int(a.get("scene_collision_contact_count") or 0) for a in actions)
    return {
        "robot_scene_contact_event_count": committed,
        "rejected_scene_collision_probe_count": int(rejected_probe_total),
        "near_miss_event_count": int(rejected_probe_total),
        "collision_response_event_count": int(response_event_total),
        "clearance_threshold_m": policy_mod.TASK_CLEARANCE_THRESHOLD_M,
    }


def mp4_command(frames_glob: str, fps: int, out_path: str) -> list[str]:
    """ffmpeg command to assemble numbered PNG frames into an MP4 (yuv420p, web-playable)."""
    return ["ffmpeg", "-y", "-framerate", str(fps), "-pattern_type", "glob", "-i", frames_glob,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]


def build_result(*, scenarios: Sequence[Mapping[str, Any]], outcomes: Sequence[Mapping[str, Any]],
                 policy_id: str, kitchen_usd: str, g1_usd: str | None,
                 blockers: Sequence[str],
                 physics_articulation_contact_reports: Sequence[Mapping[str, Any]] | None = None) -> dict:
    passed = sum(1 for o in outcomes if o.get("task_success"))
    status = "completed" if outcomes and not blockers else "blocked"
    contact_summary = summarize_physics_articulation_contact_reports(
        physics_articulation_contact_reports or []
    )
    proof_boundary = (
        "Isaac RTX-rendered kinematic walk-to-target preview (parity with the MuJoCo preview "
        "controller). Not dynamic locomotion, not a learned policy, not deployment readiness."
    )
    if contact_summary["all_have_support_contact_evidence"]:
        proof_boundary = (
            "Isaac RTX-rendered kinematic walk-to-target preview plus opt-in PhysX articulation "
            "standing/contact settle samples. This upgrades the standing placement evidence to "
            "physics-stepped support/contact evidence, but it is still not full dynamic locomotion, "
            "not a learned balance controller, and not deployment readiness."
        )
    elif contact_summary["scenario_count"] > 0:
        proof_boundary = (
            "Isaac RTX-rendered kinematic walk-to-target preview plus opt-in PhysX articulation "
            "standing/contact settle samples. The physics settle completed, but support-contact "
            "events were not observed, so this does not prove support contact, full dynamic "
            "locomotion, learned balance control, or deployment readiness."
        )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "policy_id": policy_id,
        "kitchen_usd": kitchen_usd,
        "g1_usd": g1_usd,
        "scenario_count": len(scenarios),
        "scenarios_executed": len(outcomes),
        "scenarios_passed": passed,
        "rendered_by_isaac_rtx": True,
        "blockers": list(blockers),
        "scenarios": [
            {"scenario_id": s.get("scenario_id"), **o}
            for s, o in zip(scenarios, outcomes)
        ],
        "proof_boundary": proof_boundary,
    }
    if contact_summary["scenario_count"] > 0:
        result["physics_articulation_standing_contact_summary"] = contact_summary
        result["physics_articulation_standing_contact_reports"] = [
            dict(report) for report in physics_articulation_contact_reports or []
        ]
    return result


def summarize_physics_articulation_contact_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scenario_count = len(reports)
    completed = [r for r in reports if r.get("status") == "completed"]
    contact_records = sum(int(r.get("contact_event_count") or 0) for r in reports)
    support_records = sum(int(r.get("support_contact_event_count") or 0) for r in reports)
    return {
        "scenario_count": scenario_count,
        "completed_scenario_count": len(completed),
        "contact_event_count": contact_records,
        "support_contact_event_count": support_records,
        "all_completed": bool(scenario_count and len(completed) == scenario_count),
        "all_have_support_contact_evidence": bool(
            scenario_count and all(int(r.get("support_contact_event_count") or 0) > 0 for r in reports)
        ),
        "root_pose_teleport_during_physics_settle": any(
            bool(r.get("root_pose_teleport_during_physics_settle")) for r in reports
        ),
        "claim_boundary": (
            "PhysX articulation standing/contact settle evidence only. This does not prove full "
            "dynamic locomotion, learned balance, task manipulation success, or deployment readiness."
        ),
    }


def upload_zip(out_dir: Path, put_url: str | None) -> int | None:
    if not put_url:
        return None
    import urllib.request
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out_dir).as_posix())
    req = urllib.request.Request(put_url, data=buf.getvalue(), method="PUT",
                                 headers={"Content-Type": "application/zip"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return int(getattr(r, "status", 200))


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) for a rotation about +Z."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _norm(v):
    m = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) or 1.0
    return (v[0] / m, v[1] / m, v[2] / m)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """USD-camera look-at orientation as (w, x, y, z). The camera views along its local -Z with
    +Y up; we build the basis [x, y, z] with z = -forward and convert to a quaternion."""
    forward = _norm((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
    zc = (-forward[0], -forward[1], -forward[2])            # camera local +Z (out of screen)
    xc = _norm(_cross(up, zc))
    if xc == (0.0, 0.0, 0.0):                               # up parallel to view dir
        xc = _norm(_cross((0.0, 1.0, 0.0), zc))
    yc = _cross(zc, xc)
    m00, m01, m02 = xc[0], yc[0], zc[0]
    m10, m11, m12 = xc[1], yc[1], zc[1]
    m20, m21, m22 = xc[2], yc[2], zc[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return (w, x, y, z)


def project_point_to_pixel(world_pt, eye, target, up, vfov_deg: float, width: int, height: int):
    """Pinhole-project a world point into the camera image. Returns (u, v, depth) in pixels if the
    point is in front of the camera and within frame, else None. Used to build the G1 skeleton
    landmarks (joint world positions -> 2D image landmarks) for OSCAR conditioning."""
    fwd = _norm((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
    right = _norm(_cross(fwd, up))
    if right == (0.0, 0.0, 0.0):
        right = _norm(_cross(fwd, (0.0, 1.0, 0.0)))
    tup = _cross(right, fwd)
    rel = (world_pt[0] - eye[0], world_pt[1] - eye[1], world_pt[2] - eye[2])
    z = rel[0] * fwd[0] + rel[1] * fwd[1] + rel[2] * fwd[2]
    if z <= 1e-6:
        return None
    x = rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2]
    y = rel[0] * tup[0] + rel[1] * tup[1] + rel[2] * tup[2]
    f = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    u = width / 2.0 + f * (x / z)
    v = height / 2.0 - f * (y / z)
    if 0.0 <= u < width and 0.0 <= v < height:
        return (u, v, z)
    return None


def scene_framing(scenarios: Sequence[Mapping[str, Any]]) -> tuple[tuple[float, float, float], float]:
    """Center + radius of all scenario route points, for the static overview camera."""
    pts = [p for sc in scenarios for p in sc.get("route_points", [])]
    if not pts:
        return (0.0, 0.0, ROBOT_PELVIS_HEIGHT_M), 4.0
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    radius = max((math.hypot(p[0] - cx, p[1] - cy) for p in pts), default=2.0)
    return (cx, cy, ROBOT_PELVIS_HEIGHT_M), max(2.5, radius)


def _materialize_deferred_task_route(
    scenario: dict[str, Any],
    *,
    stance_plan: Mapping[str, Any],
    root_pose: Sequence[float],
    look_at: Sequence[float] | None,
) -> None:
    """Fill legacy navigation fields from the dynamically resolved manipulation stance.

    Task-only scenarios intentionally enter without scene coordinates. Once USD/scene-placement has
    resolved the target and accepted a clear stance, the legacy policy/outcome code can use this
    dynamically-derived route without any placeholder coordinates becoming evidence.
    """
    target = _optional_xyz(look_at) if look_at is not None else None
    if target is None:
        target = _optional_xyz(stance_plan.get("task_target_xyz"))
    if target is None:
        return
    root = [float(root_pose[0]), float(root_pose[1]), float(root_pose[2])]
    nav_target = [float(target[0]), float(target[1]), ROBOT_PELVIS_HEIGHT_M]
    floor_z = float(root[2]) - ROBOT_PELVIS_HEIGHT_M
    scenario["start"] = root
    scenario["target"] = nav_target
    scenario["route_points"] = [root, nav_target]
    scenario["raw_spawn_position_xyz"] = [root[0], root[1], floor_z]
    scenario["raw_target_position_xyz"] = [float(target[0]), float(target[1]), float(target[2])]
    scenario["task_target_deferred"] = False
    scenario["deferred_task_resolution"] = "materialized_from_task_stance_plan"


def follow_cam_pose(root_pose, yaw, *, back: float = 2.2, up: float = 1.6):
    """Eye + target for a robot-POV follow camera: behind and above the root, looking ahead."""
    fx, fy = math.cos(yaw), math.sin(yaw)
    eye = (root_pose[0] - fx * back, root_pose[1] - fy * back, root_pose[2] + up)
    target = (root_pose[0] + fx * 1.5, root_pose[1] + fy * 1.5, root_pose[2] + 0.2)
    return eye, target


def verify_cam_pose(root_pose, yaw, *, back: float = 2.4, up: float = 1.5, side: float = 1.2):
    """3rd-person VERIFICATION camera: pulled back behind + above + to the side so the WHOLE robot AND
    the workspace it faces are both in frame — proves where the robot is actually standing (vs the
    egocentric POV, which shows only what the robot looks at)."""
    fx, fy = math.cos(yaw), math.sin(yaw)
    px, py = -fy, fx  # perpendicular (left of facing) for a 3/4 angle that reveals body-vs-counter gap
    eye = (root_pose[0] - fx * back + px * side, root_pose[1] - fy * back + py * side, root_pose[2] + up)
    target = (root_pose[0] + fx * 0.45, root_pose[1] + fy * 0.45, root_pose[2] + 0.25)  # robot torso/front
    return eye, target


def manipulation_cam_pose(
    root_pose,
    yaw,
    *,
    eye_forward: float = 0.12,
    eye_height: float = 1.35,
    target_forward: float = 0.6,
    target_height: float = 0.9,
    look_at=None,
    shoulder_side: float = 0.38,
    reach_arm: str = "right",
):
    """Eye + target for an EGOCENTRIC manipulation POV: from the robot's head, looking down-forward
    at the workspace directly in front (the sink/faucet and the robot's hands).

    Unlike ``follow_cam_pose`` (a chase shot behind+above, framing the whole robot walking across the
    room) this frames the local task region. Heights are absolute so the view sits at head level and
    looks at counter level — the in-distribution, coherent view a manipulation WAM can actually
    predict, instead of a room-scale navigation scene it collapses to blur on.

    ``look_at`` (a fixed world x,y,z — e.g. the affordance/handle surface) pins the target so the
    workspace stays centered regardless of the policy's noisy final yaw; without it the target is
    derived yaw-relative (forward of the robot)."""
    fx, fy = math.cos(yaw), math.sin(yaw)
    if look_at is not None:
        affordance = (float(look_at[0]), float(look_at[1]), float(look_at[2]))
        rx, ry = fy, -fx
        arm_norm = str(reach_arm or "right").strip().lower()
        if arm_norm == "left":
            active_side = -1.0
        elif arm_norm == "both":
            active_side = 0.0
        else:
            active_side = 1.0
        # Keep the camera at a head-mounted seed pose. The target still blends slightly toward the
        # active shoulder so the hand/forearm remain in the policy seed instead of a handle-only crop.
        shoulder_hint = (
            root_pose[0] + rx * float(shoulder_side) * active_side * 0.65,
            root_pose[1] + ry * float(shoulder_side) * active_side * 0.65,
            min(max(float(eye_height) - 0.10, affordance[2] + 0.04), affordance[2] + 0.22),
        )
        target_blend = 0.30 if active_side else 0.14
        target = (
            affordance[0] * (1.0 - target_blend) + shoulder_hint[0] * target_blend,
            affordance[1] * (1.0 - target_blend) + shoulder_hint[1] * target_blend,
            affordance[2] * (1.0 - target_blend) + shoulder_hint[2] * target_blend,
        )
        eye = (
            root_pose[0] + fx * float(eye_forward),
            root_pose[1] + fy * float(eye_forward),
            max(float(eye_height), affordance[2] + 0.32),
        )
    else:
        eye = (root_pose[0] + fx * eye_forward, root_pose[1] + fy * eye_forward, eye_height)
        target = (root_pose[0] + fx * target_forward, root_pose[1] + fy * target_forward, target_height)
    return eye, target


def _weighted_xyz(points: Sequence[tuple[Sequence[float], float]]) -> tuple[float, float, float] | None:
    total = 0.0
    acc = [0.0, 0.0, 0.0]
    for xyz, weight in points:
        try:
            w = float(weight)
            vals = [float(xyz[i]) for i in range(3)]
        except Exception:  # noqa: BLE001
            continue
        if w <= 0.0:
            continue
        total += w
        for i in range(3):
            acc[i] += vals[i] * w
    if total <= 0.0:
        return None
    return (acc[0] / total, acc[1] / total, acc[2] / total)


def _manipulation_camera_target_with_arm_context(
    affordance,
    arm_points: Mapping[str, Sequence[float]] | None,
) -> tuple[float, float, float]:
    """Aim between the affordance and the active arm, not only at the handle.

    A pure handle-centered ray can crop the forearm out of an egocentric frame. This target remains
    task anchored while pulling the camera aim toward the wrist/elbow chain when those robot links are
    available from the USD asset.
    """
    aff = (float(affordance[0]), float(affordance[1]), float(affordance[2]))
    pts: list[tuple[Sequence[float], float]] = [(aff, 0.38)]
    if arm_points:
        if arm_points.get("hand") is not None:
            pts.append((arm_points["hand"], 0.26))
        if arm_points.get("wrist") is not None:
            pts.append((arm_points["wrist"], 0.24))
        if arm_points.get("elbow") is not None:
            pts.append((arm_points["elbow"], 0.12))
    return _weighted_xyz(pts) or aff


def _projection_dict(px) -> dict[str, Any] | None:
    if px is None:
        return None
    return {
        "available": True,
        "u_px": round(float(px[0]), 2),
        "v_px": round(float(px[1]), 2),
        "depth_m": round(float(px[2]), 4),
    }


def _normalize_reach_arm_selection(arm: str) -> str:
    selection = str(arm or "right").strip().lower()
    return selection if selection in {"left", "right", "both"} else "right"


def _required_manipulation_arms(arm: str) -> tuple[str, ...]:
    selection = _normalize_reach_arm_selection(arm)
    return ("left", "right") if selection == "both" else (selection,)


def _manipulation_pov_geometry_single(
    *,
    arm_points: Mapping[str, Sequence[float]] | None,
    affordance,
    eye,
    target,
    up=(0.0, 0.0, 1.0),
    vfov_deg: float,
    width: int,
    height: int,
    arm: str = "right",
) -> dict[str, Any]:
    """Machine-check that the actual USD arm links are visible in the head POV.

    The articulated skeleton trace is optional and may be disabled for crash-safe kinematic renders,
    so manipulation media needs an independent USD-link projection check. This is an initial seed
    frame gate: the target affordance and task-side forearm/gripper must be visible, and the arm must
    be held forward toward the workspace. It deliberately does NOT require the gripper to be at the
    affordance; the policy/WAM rollout is responsible for completing the manipulation.
    """
    blockers: list[str] = []
    aff = (float(affordance[0]), float(affordance[1]), float(affordance[2]))
    target_px = project_point_to_pixel(aff, eye, target, up, vfov_deg, width, height)
    projected: list[dict[str, Any]] = []
    available_roles = sorted(str(k) for k in (arm_points or {}).keys())
    for role in ("shoulder", "elbow", "wrist", "hand"):
        pt = (arm_points or {}).get(role)
        if pt is None:
            continue
        px = project_point_to_pixel(
            (float(pt[0]), float(pt[1]), float(pt[2])),
            eye,
            target,
            up,
            vfov_deg,
            width,
            height,
        )
        proj = _projection_dict(px)
        if proj is None:
            continue
        projected.append({
            "landmark_id": f"{arm}_{role}_link",
            "link_role": role,
            "image_projection": proj,
        })

    roles_in_frame = {str(item["link_role"]) for item in projected}
    effector_roles = roles_in_frame.intersection({"hand", "wrist"})
    forearm_roles = roles_in_frame.intersection({"elbow", "wrist", "hand"})
    if not available_roles:
        blockers.append("manipulation_pov_arm_links_unavailable")
    if target_px is None:
        blockers.append("manipulation_pov_target_not_in_frame")
    if not effector_roles:
        blockers.append("manipulation_pov_arm_not_in_frame")
    if len(forearm_roles) < 2:
        blockers.append("manipulation_pov_forearm_not_in_frame")

    effector_distances: dict[str, float] = {}
    for role in ("wrist", "hand"):
        pt = (arm_points or {}).get(role)
        if pt is None:
            continue
        effector_distances[role] = round(
            math.sqrt(sum((float(pt[i]) - aff[i]) ** 2 for i in range(3))),
            4,
        )
    arm_extension: dict[str, Any] = {
        "status": "unverified",
        "blockers": ["manipulation_pov_arm_extension_unverified"],
    }
    if arm_points and (arm_points.get("shoulder") is not None):
        shoulder_pt = arm_points["shoulder"]
        spans = []
        for role in ("wrist", "hand"):
            pt = arm_points.get(role)
            if pt is None:
                continue
            spans.append(math.sqrt(sum((float(pt[i]) - float(shoulder_pt[i])) ** 2 for i in range(3))))
        effector_pt = (arm_points.get("hand") or arm_points.get("wrist"))
        if effector_pt is not None:
            shoulder = tuple(float(shoulder_pt[i]) for i in range(3))
            effector = tuple(float(effector_pt[i]) for i in range(3))
            task_dir = _norm((aff[0] - shoulder[0], aff[1] - shoulder[1], aff[2] - shoulder[2]))
            arm_dir = _norm((
                effector[0] - shoulder[0],
                effector[1] - shoulder[1],
                effector[2] - shoulder[2],
            ))
            arm_len = math.sqrt(sum((effector[i] - shoulder[i]) ** 2 for i in range(3)))
            alignment = sum(arm_dir[i] * task_dir[i] for i in range(3))
            vertical_drop_ratio = (
                abs(effector[2] - shoulder[2]) / arm_len if arm_len > 1e-6 else 1.0
            )
            extension_blockers: list[str] = []
            # Low alignment catches the bad seed class where the arm is visible but hanging down or
            # tucked sideways instead of held out toward the task workspace. Distance to the affordance
            # is only metadata; the initial seed must not pre-solve the task.
            if alignment < 0.35:
                extension_blockers.append("manipulation_pov_arm_not_extended_forward")
            if arm_len < 0.12:
                extension_blockers.append("manipulation_pov_arm_extension_too_short")
            arm_extension = {
                "status": "PASS" if not extension_blockers else "FAIL",
                "blockers": extension_blockers,
                "shoulder_to_effector_m": round(float(arm_len), 4),
                "alignment_to_affordance_direction": round(float(alignment), 4),
                "vertical_drop_ratio": round(float(vertical_drop_ratio), 4),
                "claim_boundary": (
                    "Forward extension checks initial manipulation readiness only. It does not require "
                    "contact with the affordance or prove task completion."
                ),
            }
    if arm_extension.get("status") != "PASS":
        blockers.extend(str(b) for b in (arm_extension.get("blockers") or []))

    return {
        "schema_version": "manipulation_pov_geometry.v1",
        "status": "PASS" if not blockers else "FAIL",
        "blockers": sorted(set(blockers)),
        "camera": "robot_pov",
        "reach_arm": arm,
        "target_affordance_xyz": [round(float(v), 6) for v in aff],
        "target_in_frame": target_px is not None,
        "target_projection": _projection_dict(target_px),
        "available_arm_link_roles": available_roles,
        "arm_roles_in_frame": sorted(roles_in_frame),
        "arm_landmarks_in_frame": len(projected),
        "effector_distance_to_affordance_m": effector_distances,
        "effector_distance_is_metadata_only": True,
        "arm_extension": arm_extension,
        "projected_landmarks": projected,
        "claim_boundary": (
            "This checks initial camera framing of USD robot links against the resolved task affordance. "
            "It is not manipulation success, contact proof, physical reach validation, or deployment "
            "readiness."
        ),
    }


def _manipulation_pov_geometry(
    *,
    arm_points: Mapping[str, Sequence[float]] | None,
    affordance,
    eye,
    target,
    up=(0.0, 0.0, 1.0),
    vfov_deg: float,
    width: int,
    height: int,
    arm: str = "right",
    arm_points_by_arm: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> dict[str, Any]:
    """Validate a manipulation seed POV for one or more requested arms."""
    selection = _normalize_reach_arm_selection(arm)
    required_arms = _required_manipulation_arms(selection)
    if len(required_arms) == 1:
        side = required_arms[0]
        side_points = (arm_points_by_arm or {}).get(side) if arm_points_by_arm else None
        report = _manipulation_pov_geometry_single(
            arm_points=side_points or arm_points,
            affordance=affordance,
            eye=eye,
            target=target,
            up=up,
            vfov_deg=vfov_deg,
            width=width,
            height=height,
            arm=side,
        )
        report["reach_arm"] = selection
        report["required_arms"] = [side]
        return report

    per_arm: dict[str, dict[str, Any]] = {}
    for side in required_arms:
        per_arm[side] = _manipulation_pov_geometry_single(
            arm_points=(arm_points_by_arm or {}).get(side) or {},
            affordance=affordance,
            eye=eye,
            target=target,
            up=up,
            vfov_deg=vfov_deg,
            width=width,
            height=height,
            arm=side,
        )
    primary = per_arm.get("right") or next(iter(per_arm.values()))
    side_failures = [side for side, report in per_arm.items() if report.get("status") != "PASS"]
    blockers = sorted({
        str(blocker)
        for report in per_arm.values()
        for blocker in (report.get("blockers") or [])
    } | {
        f"manipulation_pov_{side}_arm_seed_failed"
        for side in side_failures
    })
    target_px = primary.get("target_projection")
    projected = [
        landmark
        for report in per_arm.values()
        for landmark in (report.get("projected_landmarks") or [])
    ]
    roles_in_frame = sorted({
        str(role)
        for report in per_arm.values()
        for role in (report.get("arm_roles_in_frame") or [])
    })
    available_roles = sorted({
        str(role)
        for report in per_arm.values()
        for role in (report.get("available_arm_link_roles") or [])
    })
    extension_by_arm = {
        side: report.get("arm_extension")
        for side, report in per_arm.items()
    }
    extension_blockers = sorted({
        str(blocker)
        for extension in extension_by_arm.values()
        if isinstance(extension, Mapping)
        for blocker in (extension.get("blockers") or [])
    })
    return {
        "schema_version": "manipulation_pov_geometry.v1",
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "camera": "robot_pov",
        "reach_arm": selection,
        "required_arms": list(required_arms),
        "target_affordance_xyz": primary.get("target_affordance_xyz"),
        "target_in_frame": bool(primary.get("target_in_frame")),
        "target_projection": target_px,
        "available_arm_link_roles": available_roles,
        "available_arm_link_roles_by_arm": {
            side: report.get("available_arm_link_roles") or []
            for side, report in per_arm.items()
        },
        "arm_roles_in_frame": roles_in_frame,
        "arm_roles_in_frame_by_arm": {
            side: report.get("arm_roles_in_frame") or []
            for side, report in per_arm.items()
        },
        "arm_landmarks_in_frame": len(projected),
        "effector_distance_to_affordance_m": (
            primary.get("effector_distance_to_affordance_m") or {}
        ),
        "effector_distance_to_affordance_m_by_arm": {
            side: report.get("effector_distance_to_affordance_m") or {}
            for side, report in per_arm.items()
        },
        "effector_distance_is_metadata_only": True,
        "arm_extension": {
            "status": "PASS" if not extension_blockers else "FAIL",
            "blockers": extension_blockers,
            "by_arm": extension_by_arm,
            "claim_boundary": (
                "Forward extension checks initial manipulation readiness only. It does not require "
                "contact with the affordance or prove task completion."
            ),
        },
        "projected_landmarks": projected,
        "per_arm_geometry": per_arm,
        "claim_boundary": (
            "This checks initial camera framing of USD robot links against the resolved task affordance. "
            "It is not manipulation success, contact proof, physical reach validation, or deployment "
            "readiness."
        ),
    }


def _fraction_from_histogram(hist: Sequence[int], indexes: range) -> float:
    total = float(sum(hist))
    if total <= 0.0:
        return 0.0
    return float(sum(hist[i] for i in indexes)) / total


def _image_luma_extreme_fractions(gray_img, box) -> dict[str, float]:
    crop = gray_img.crop(tuple(int(v) for v in box))
    hist = crop.histogram()
    return {
        "dark_fraction": round(_fraction_from_histogram(hist, range(0, 14)), 6),
        "bright_fraction": round(_fraction_from_histogram(hist, range(242, 256)), 6),
    }


def _pov_seed_frame_quality(frame_path: Path | str) -> dict[str, Any]:
    """Detect obvious camera self-occlusion/clipping in a saved robot POV seed frame.

    This deliberately uses broad image statistics, not object-specific semantics: a robot-head seed
    with a large near-black edge wedge is usually a camera inside/behind robot geometry or a clipped
    near-field body part. Dark task objects in the center are allowed; edge occlusion is not.
    """
    try:
        from PIL import Image  # type: ignore
        img = Image.open(frame_path).convert("L")
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": "manipulation_pov_seed_frame_quality.v1",
            "status": "FAIL",
            "blockers": ["manipulation_pov_frame_unreadable"],
            "error": repr(exc),
        }
    w, h = img.size
    edge_w = max(1, int(round(w * 0.16)))
    lower_y = max(0, int(round(h * 0.45)))
    regions = {
        "left_edge": (0, 0, edge_w, h),
        "right_edge": (max(0, w - edge_w), 0, w, h),
        "left_lower_edge": (0, lower_y, edge_w, h),
        "right_lower_edge": (max(0, w - edge_w), lower_y, w, h),
    }
    metrics = {
        name: _image_luma_extreme_fractions(img, box)
        for name, box in regions.items()
    }
    edge_dark = max(metrics["left_edge"]["dark_fraction"], metrics["right_edge"]["dark_fraction"])
    lower_edge_dark = max(
        metrics["left_lower_edge"]["dark_fraction"],
        metrics["right_lower_edge"]["dark_fraction"],
    )
    blockers: list[str] = []
    if edge_dark > 0.38 or lower_edge_dark > 0.46:
        blockers.append("manipulation_pov_edge_self_occlusion")
    return {
        "schema_version": "manipulation_pov_seed_frame_quality.v1",
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "frame_path": str(frame_path),
        "image_size_px": [int(w), int(h)],
        "edge_band_fraction": 0.16,
        "regions": metrics,
        "max_edge_dark_fraction": round(float(edge_dark), 6),
        "max_lower_edge_dark_fraction": round(float(lower_edge_dark), 6),
        "claim_boundary": (
            "Image statistics only catch gross edge self-occlusion/clipping in the seed POV. "
            "They do not validate task success or object contact."
        ),
    }


def _select_manipulation_camera_target_for_visible_arm(
    affordance,
    arm_points: Mapping[str, Sequence[float]] | None,
    eye,
    initial_target,
    *,
    vfov_deg: float,
    width: int,
    height: int,
    arm: str = "right",
    arm_points_by_arm: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Pick a task-anchored look-at that keeps the active forearm in the head POV when possible."""
    aff = (float(affordance[0]), float(affordance[1]), float(affordance[2]))
    candidates: list[tuple[str, tuple[float, float, float]]] = [
        ("affordance_arm_context", tuple(float(v) for v in initial_target)),
        ("affordance", aff),
    ]
    if arm_points:
        hand = arm_points.get("hand")
        wrist = arm_points.get("wrist")
        elbow = arm_points.get("elbow")
        if hand is not None or wrist is not None:
            pts: list[tuple[Sequence[float], float]] = [(aff, 0.22)]
            if hand is not None:
                pts.append((hand, 0.34))
            if wrist is not None:
                pts.append((wrist, 0.30))
            if elbow is not None:
                pts.append((elbow, 0.14))
            weighted = _weighted_xyz(pts)
            if weighted is not None:
                candidates.append(("forearm_weighted", weighted))
        if wrist is not None and hand is not None:
            weighted = _weighted_xyz([(aff, 0.35), (wrist, 0.30), (hand, 0.35)])
            if weighted is not None:
                candidates.append(("effector_weighted", weighted))

    best_name = candidates[0][0]
    best_target = candidates[0][1]
    best_score = -1.0
    scored: list[dict[str, Any]] = []
    for name, candidate in candidates:
        geom = _manipulation_pov_geometry(
            arm_points=arm_points,
            arm_points_by_arm=arm_points_by_arm,
            affordance=aff,
            eye=eye,
            target=candidate,
            vfov_deg=vfov_deg,
            width=width,
            height=height,
            arm=arm,
        )
        roles = set(geom.get("arm_roles_in_frame") or [])
        score = 0.0
        if geom.get("target_in_frame"):
            score += 6.0
        if roles.intersection({"hand", "wrist"}):
            score += 8.0
        score += 2.0 * len(roles.intersection({"elbow", "wrist", "hand"}))
        if len(roles.intersection({"elbow", "wrist", "hand"})) >= 2:
            score += 5.0
        if geom.get("status") == "PASS":
            score += 10.0
        else:
            score -= 1.5 * len(geom.get("blockers") or [])
        if score > best_score:
            best_name = name
            best_target = candidate
            best_score = score
        scored.append({
            "candidate": name,
            "score": round(score, 3),
            "status": geom.get("status"),
            "target_in_frame": geom.get("target_in_frame"),
            "arm_roles_in_frame": geom.get("arm_roles_in_frame"),
        })
    return best_target, {
        "selected_camera_target": best_name,
        "camera_target_score": round(best_score, 3),
        "camera_target_candidates": scored,
    }


# ============================ Isaac-only (GPU worker) ============================

def _boot_sim(headless: bool = True):
    from isaacsim import SimulationApp  # type: ignore
    return SimulationApp({"headless": headless, "renderer": "RayTracedLighting"})


def _extension_toggle():
    try:
        from isaacsim.core.utils.extensions import enable_extension, disable_extension  # type: ignore
    except Exception:  # noqa: BLE001
        from omni.isaac.core.utils.extensions import enable_extension, disable_extension  # type: ignore
    return enable_extension, disable_extension


def _enable_and_import_replicator():
    """Enable the Replicator extension (needed for render products) and import it. Must be
    called AFTER SimulationApp boots — omni.* modules are not importable before Kit starts."""
    enable_extension, _ = _extension_toggle()
    enable_extension("omni.replicator.core")
    import omni.replicator.core as rep  # type: ignore
    return rep


def _disable_physics_cooking() -> None:
    """Disable ONLY the PhysX collision-*cooking* extension (not the physx core, which the RTX
    renderer depends on), so the 47-object kitchen's SDF/convex cooking can't block the render.
    Also push every collision approximation to the cheapest box via carb settings, so any residual
    cooking is trivial. The kinematic preview needs no kitchen physics anyway."""
    _, disable_extension = _extension_toggle()
    try:
        disable_extension("omni.physx.cooking")
    except Exception:  # noqa: BLE001
        pass
    try:
        import carb  # type: ignore
        s = carb.settings.get_settings()
        s.set_bool("/physics/cooking/ujitsoCollisionCooking", False)
        s.set_bool("/persistent/physics/visualizationDisplayColliders", False)
        s.set_bool("/physics/collisionConeCustomGeometry", False)
    except Exception:  # noqa: BLE001
        pass


def _open_stage(usd_path: str):
    import omni.usd  # type: ignore
    ctx = omni.usd.get_context()
    ctx.open_stage(usd_path)
    return ctx.get_stage()


def _task_description_for_scenario(scenario: Mapping[str, Any]) -> str:
    """Best-effort natural-language task string for the scenario (for task->object resolution)."""
    for key in ("instruction", "task", "task_description", "description", "task_instruction"):
        val = scenario.get(key)
        if val:
            return str(val).strip()
    return ""


def _resolve_task_target_via_scene_placement(stage, scenario: Mapping[str, Any]) -> dict[str, Any] | None:
    """Resolve the task's target OBJECT from the scene when no explicit object id/coords are given.

    Unlike :func:`_resolve_task_target_from_stage` (which needs the caller to name the prim id), this
    enumerates EVERY object in the scene via the swappable ``scene_placement`` spatial index and maps
    the task description ("turn on the faucet") onto one of them. No scene-specific coordinates and no
    foreknowledge of the prim id — it works for any task in any USD site. The package is imported
    lazily + optionally so a worker without it (or without pxr) degrades to the id-driven path.
    """
    task = _task_description_for_scenario(scenario)
    if not task:
        return None
    # The bundle dir (where scene_placement is shipped) is added to sys.path at module load, but
    # Isaac's SimulationApp boot rewrites sys.path and drops it — and this resolver runs AFTER boot.
    # Re-add the runner's own dir so the bundle-first import below can still find the package.
    bundle_dir = str(Path(__file__).resolve().parent)
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)
    try:
        from scene_placement import (  # type: ignore # worker: flat package in the bundle dir
            UsdSceneSpatialIndex,
            resolve_target_by_label,
        )
    except Exception as inner_exc:  # noqa: BLE001
        try:
            from blueprint_pipeline.scene_placement import (  # type: ignore # repo / tests
                UsdSceneSpatialIndex,
                resolve_target_by_label,
            )
        except Exception:  # noqa: BLE001
            # Surface the INNER (bundle) import error, not the repo-fallback's — that's the one
            # that matters on the worker. Include sys.path head for diagnosis.
            return {"status": "blocked", "blockers": ["scene_placement_unavailable"],
                    "error": repr(inner_exc), "bundle_dir": bundle_dir, "sys_path_head": sys.path[:6]}
    try:
        index = UsdSceneSpatialIndex(stage=stage)
        objects = list(index.objects())
    except Exception as exc:  # noqa: BLE001
        return {"status": "blocked", "blockers": ["scene_placement_index_failed"], "error": repr(exc)}
    if not objects:
        return {"status": "blocked", "blockers": ["scene_placement_no_objects"], "task": task}
    target = resolve_target_by_label(task, objects)
    if target is None:
        return {
            "status": "blocked",
            "blockers": ["scene_placement_no_task_match"],
            "task": task,
            "object_labels": sorted({o.label for o in objects})[:40],
        }
    size = target.size()
    prim_path = ""
    extra = getattr(target, "extra", None)
    if isinstance(extra, Mapping):
        prim_path = str(extra.get("prim_path", "") or "")
    return {
        "status": "resolved",
        "source": "scene_placement_task_label",
        "selected": {
            "target_object_id": target.id,
            "target_object_label": target.label,
            "prim_path": prim_path,
            "match_kind": "task_label",
            "center_xyz": [round(float(c), 6) for c in target.centroid],
            "size_xyz": [round(float(s), 6) for s in size],
            "bbox_min_xyz": [round(float(c), 6) for c in target.bbox_min],
            "bbox_max_xyz": [round(float(c), 6) for c in target.bbox_max],
            "footprint_center_xy": [
                round(float(c), 6) for c in target.footprint_center()
            ],
        },
        "task": task,
        "objects_considered": len(objects),
    }


def _resolve_task_target_from_stage(stage, scenario: Mapping[str, Any]) -> dict[str, Any] | None:
    """Resolve a task target from USD prim bounds when a scene/task compiler provides an object id.

    This is the generic fallback for sites that do not ship a separate object-location JSON. It is
    intentionally object-id driven; it does not know about kitchens, sinks, dishwashers, or counters.
    When no object id is supplied (or the id isn't found), it defers to
    :func:`_resolve_task_target_via_scene_placement`, which maps the task description onto a scene
    object — so a scenario with only a natural-language task still resolves a target dynamically.
    """
    target_ids = [
        str(scenario.get(key)).strip()
        for key in TASK_STANCE_TARGET_OBJECT_KEYS
        if scenario.get(key)
    ]
    if not target_ids:
        # No explicit object id given — derive the target object from the task description via the
        # scene_placement spatial index (enumerate scene objects, map task -> object). Dynamic path.
        return _resolve_task_target_via_scene_placement(stage, scenario)
    try:
        from pxr import Usd, UsdGeom  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"status": "blocked", "blockers": ["usd_target_bounds_unavailable"], "error": repr(exc)}
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    matches: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_name = str(prim.GetName())
        path_segments = [segment.lower() for segment in prim_path.split("/") if segment]
        matched_id = next(
            (
                tid
                for tid in target_ids
                if tid.lower() == prim_name.lower() or tid.lower() in path_segments
            ),
            None,
        )
        match_kind = "exact_prim_name_or_path_segment"
        if not matched_id:
            text = f"{prim_path} {prim_name}".lower()
            matched_id = next((tid for tid in target_ids if tid.lower() in text), None)
            match_kind = "ancestor_or_text_match"
        if not matched_id:
            continue
        try:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if box.IsEmpty():
                continue
            bbox_min, bbox_max, center, size = _aligned_box_min_max_center_size(box)
        except Exception:  # noqa: BLE001
            continue
        center_xyz = [round(float(center[i]), 6) for i in range(3)]
        size_xyz = [round(float(size[i]), 6) for i in range(3)]
        bbox_min_xyz = [round(float(bbox_min[i]), 6) for i in range(3)]
        bbox_max_xyz = [round(float(bbox_max[i]), 6) for i in range(3)]
        matches.append({
            "target_object_id": matched_id,
            "prim_path": prim_path,
            "match_kind": match_kind,
            "path_depth": len(path_segments),
            "center_xyz": center_xyz,
            "size_xyz": size_xyz,
            "bbox_min_xyz": bbox_min_xyz,
            "bbox_max_xyz": bbox_max_xyz,
            "footprint_center_xy": [round(0.5 * (bbox_min_xyz[i] + bbox_max_xyz[i]), 6)
                                    for i in range(2)],
            "volume_proxy": round(float(size[0] * size[1] * size[2]), 9),
        })
    if not matches:
        # The supplied object id(s) weren't found as prims — fall back to task->object resolution
        # over the full scene catalog before giving up.
        sp = _resolve_task_target_via_scene_placement(stage, scenario)
        if sp is not None and sp.get("status") == "resolved":
            return sp
        return {
            "status": "blocked",
            "blockers": ["target_object_id_not_found_in_usd_stage"],
            "target_object_ids": target_ids,
            "scene_placement_fallback": sp,
        }
    matches.sort(key=lambda item: (
        0 if item["match_kind"] == "exact_prim_name_or_path_segment" else 1,
        item["path_depth"],
        -float(item["volume_proxy"]),
        len(item["prim_path"]),
    ))
    return {
        "status": "resolved",
        "source": "usd_prim_bounds",
        "selected": matches[0],
        "matches_considered": matches[:10],
    }


def _scene_placement_stand_plan(
    target_resolution,
    probe,
    *,
    floor_z: float = 0.05,
    scenario: Mapping[str, Any] | None = None,
    placement_validator=None,
):
    """Place the robot for a scene_placement-resolved target.

    When a scenario is supplied, the runner uses the approach-biased task-stance planner with the
    resolved target bounds, so the robot stands on the aisle/start side with distance measured from
    the target surface. Without scenario context, it falls back to the generic open-side
    ``compute_stand_pose`` solver. Returns ``None`` when the resolution lacks geometry, so the
    caller can fall back to plain ``plan_task_stance``.
    """
    sel = (target_resolution or {}).get("selected") or {}
    center = sel.get("center_xyz")
    size = sel.get("size_xyz")
    if not center or not size or len(center) != 3 or len(size) != 3:
        return None
    cx, cy, cz = (float(c) for c in center)
    sx, sy, sz = (abs(float(s)) for s in size)
    bbox_min = sel.get("bbox_min_xyz") or [cx - sx / 2.0, cy - sy / 2.0, cz - sz / 2.0]
    bbox_max = sel.get("bbox_max_xyz") or [cx + sx / 2.0, cy + sy / 2.0, cz + sz / 2.0]
    if scenario is not None:
        stance_scenario = dict(scenario)
        stance_scenario["target_object_position_xyz"] = [cx, cy, cz]
        stance_scenario["target_object_bbox_min_xyz"] = bbox_min
        stance_scenario["target_object_bbox_max_xyz"] = bbox_max
        if sel.get("target_object_id") is not None:
            stance_scenario.setdefault("target_object_id", sel.get("target_object_id"))
        if sel.get("target_object_label") is not None:
            stance_scenario.setdefault("target_object_label", sel.get("target_object_label"))
        plan = plan_task_stance(
            scenario=stance_scenario,
            probe_collision=probe,
            floor_z_hint=floor_z,
            placement_validator=placement_validator,
        )
        plan["source"] = "scene_placement_surface_offset_task_stance"
        if plan.get("status") == "accepted":
            idx = int(plan.get("selected_candidate_index") or 0)
            candidates = plan.get("candidates") or []
            chosen = candidates[idx] if 0 <= idx < len(candidates) else {}
            plan["stand_clear"] = True
            plan["standoff_m"] = chosen.get("standoff_from_target_surface_m")
            plan["notes"] = (
                "approach-biased scene target stance; distance measured from target "
                "footprint surface"
            )
        return plan
    bundle_dir = str(Path(__file__).resolve().parent)
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)
    try:
        try:
            from scene_placement import SceneObject, compute_stand_pose  # worker bundle
        except Exception:  # noqa: BLE001
            from blueprint_pipeline.scene_placement import SceneObject, compute_stand_pose  # repo/tests
    except Exception as exc:  # noqa: BLE001
        return {"status": "blocked", "blockers": ["scene_placement_unavailable"], "error": repr(exc)}
    target = SceneObject(
        id=str(sel.get("target_object_id") or "target"),
        label=str(sel.get("target_object_label") or ""),
        bbox_min=(float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
        bbox_max=(float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
        centroid=(cx, cy, cz),
        source="usd",
    )
    pose = compute_stand_pose(
        target, probe=probe, pelvis_height=ROBOT_PELVIS_HEIGHT_M, floor_z=floor_z,
        standing_distance=TASK_STANCE_DEFAULT_SURFACE_STANDOFF_M, include_diagonals=True,
    )
    return {
        "schema_version": TASK_STANCE_SCHEMA_VERSION,
        # compute_stand_pose always yields a non-clipping best-effort pose (closest clear spot, or
        # the farthest probed spot when boxed in), so this is always usable placement.
        "status": "accepted",
        "source": "scene_placement_compute_stand_pose",
        "accepted_pose": [round(float(v), 6) for v in pose.position],
        "accepted_yaw": round(float(pose.yaw), 6),
        "task_target_xyz": [round(cx, 6), round(cy, 6), round(cz, 6)],
        "stand_clear": bool(pose.clear),
        "standoff_m": round(float(pose.standoff_m), 6),
        "task_target_bounds": {
            "bbox_min_xyz": [round(float(v), 6) for v in bbox_min],
            "bbox_max_xyz": [round(float(v), 6) for v in bbox_max],
        },
        "candidates": [],
        "notes": pose.notes,
    }


def _plan_task_stance_for_stage(
    *,
    stage,
    scenario: Mapping[str, Any],
    manipulation_look_at,
    probe,
    no_collision_probe: bool,
    robot_prim_path: str | None = None,
) -> dict[str, Any]:
    stance_scenario = dict(scenario)
    target_resolution = None
    explicit_target = task_stance_target_for_scenario(
        stance_scenario,
        manipulation_look_at,
        allow_navigation_target_fallback=False,
    )
    if explicit_target is None:
        target_resolution = _resolve_task_target_from_stage(stage, stance_scenario)
        if target_resolution and target_resolution.get("status") == "resolved":
            selected = (
                target_resolution.get("selected")
                if isinstance(target_resolution.get("selected"), Mapping)
                else {}
            )
            stance_scenario = _with_xyz(
                stance_scenario,
                "target_object_position_xyz",
                selected["center_xyz"],
            )
            if selected.get("bbox_min_xyz") is not None and selected.get("bbox_max_xyz") is not None:
                stance_scenario["target_object_bbox_min_xyz"] = selected["bbox_min_xyz"]
                stance_scenario["target_object_bbox_max_xyz"] = selected["bbox_max_xyz"]
            if selected.get("target_object_id") is not None:
                stance_scenario.setdefault("target_object_id", selected.get("target_object_id"))
            if selected.get("target_object_label") is not None:
                stance_scenario.setdefault("target_object_label", selected.get("target_object_label"))
    if no_collision_probe:
        return {
            "schema_version": TASK_STANCE_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["task_stance_collision_probe_disabled"],
            "target_resolution": target_resolution,
            "claim_boundary": (
                "Task stance placement requires a scene collision/clearance probe; "
                "without it the runner must not claim the robot is standing clear."
            ),
        }
    target_bounds = _target_bounds_for_scenario(stance_scenario)
    validation_floor_z = float(stance_scenario.get("floor_z_hint", 0.05) or 0.05)
    validation_scene_objects = (
        _placement_obstacles_for_stage(stage, focus_bounds=target_bounds)
        if target_bounds is not None
        else []
    )
    validation_target_object = (
        _target_object_from_stance_plan(
            {
                "task_target_xyz": stance_scenario.get("target_object_position_xyz")
                or stance_scenario.get("task_target_position_xyz"),
                "task_target_bounds": (
                    {
                        "bbox_min_xyz": list(target_bounds[0]),
                        "bbox_max_xyz": list(target_bounds[1]),
                    }
                    if target_bounds is not None
                    else None
                ),
                "target_resolution": target_resolution,
            }
        )
        if target_bounds is not None
        else None
    )
    validation_standoff_range = _validation_standoff_range_for_scenario(stance_scenario)
    placement_validator = (
        _placement_validator_for_stage(
            stage,
            robot_prim_path,
            target_bounds,
            target_object=validation_target_object,
            scene_objects=validation_scene_objects,
            floor_z=validation_floor_z,
            standoff_range=validation_standoff_range,
        )
        if robot_prim_path and target_bounds is not None
        else None
    )
    # For a scene_placement-resolved target, keep the task object bounds while honoring the
    # scenario approach/start side. That places sink/faucet tasks in the counter-to-island aisle
    # rather than accepting a too-near point around the fixture centroid.
    if target_resolution and str(target_resolution.get("source") or "").startswith("scene_placement"):
        sp_plan = _scene_placement_stand_plan(
            target_resolution, probe,
            floor_z=float(stance_scenario.get("floor_z_hint", 0.05) or 0.05),
            scenario=stance_scenario,
            placement_validator=placement_validator,
        )
        if sp_plan is not None:
            sp_plan["target_resolution"] = target_resolution
            return sp_plan
    stance_plan = plan_task_stance(
        scenario=stance_scenario,
        manipulation_look_at=manipulation_look_at,
        probe_collision=probe,
        floor_z_hint=stance_scenario.get("floor_z_hint"),
        placement_validator=placement_validator,
    )
    if target_resolution is not None:
        stance_plan["target_resolution"] = target_resolution
    return stance_plan


def _resolve_asset_uri(value: str) -> str:
    """Resolve a relative Isaac asset path (e.g. 'Isaac/Robots/Unitree/G1/g1.usd') against the
    Isaac assets root on the worker. Absolute paths / URIs pass through unchanged."""
    if "://" in value or value.startswith("/") or value.startswith("omniverse:"):
        return value
    try:
        from isaacsim.storage.native import get_assets_root_path  # type: ignore
        root = get_assets_root_path()
        if root:
            return root.rstrip("/") + "/" + value.lstrip("/")
    except Exception:  # noqa: BLE001
        pass
    return value


def _bind_g1(stage, g1_usd: str, prim_path: str = "/World/G1"):
    """Reference the official Isaac G1 USD and verify it is a controllable, collidable articulation."""
    from pxr import UsdPhysics  # type: ignore
    g1_prim = stage.DefinePrim(prim_path, "Xform")
    g1_prim.GetReferences().AddReference(g1_usd)
    art_count = collision_count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_count += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_count += 1
    return {
        "prim_path": prim_path,
        "controllable_articulation_detected": art_count > 0,
        "collision_enabled_verified": collision_count > 0,
        "articulation_root_api_prim_count": art_count,
        "collision_api_prim_count": collision_count,
    }


def _setup_g1_articulation(prim_path: str):
    """Create + initialize an Isaac Articulation on the bound G1 so we can drive its joints
    (the procedural walk gait) and read its link world poses (the skeleton). Returns
    (articulation, dof_index_by_name, default_joint_positions, link_names). GPU-only."""
    from isaacsim.core.prims import SingleArticulation  # type: ignore
    art = SingleArticulation(prim_path=prim_path, name="g1")
    art.initialize()
    dof_names = list(art.dof_names or [])
    dof_index = {n: i for i, n in enumerate(dof_names)}
    import numpy as np  # type: ignore
    default = np.asarray(art.get_joint_positions()).astype("float32")
    link_names = list(getattr(art, "body_names", []) or [])
    return art, dof_index, default, link_names


def manipulation_ready_arm_joint_deltas(arm: str = "both") -> dict[str, float]:
    """Joint deltas that raise G1 forearms into a first-person manipulation-ready pose.

    The values are relative to the standing keyframe, so the pose is portable across Isaac and
    MuJoCo G1 assets without hard-coding absolute default qpos values.
    """
    selection = str(arm or "both").strip().lower()
    if selection not in MANIPULATION_READY_ARM_SELECTIONS:
        raise ValueError(f"unknown manipulation arm selection: {arm!r}")
    sides = ("left", "right") if selection == "both" else (selection,)
    out: dict[str, float] = {}
    for side in sides:
        out.update(MANIPULATION_READY_ARM_JOINT_DELTAS[side])
    return out


def _apply_joint_deltas(targets, default, dof_index, deltas: Mapping[str, float]) -> list[str]:
    applied: list[str] = []
    for name, delta in deltas.items():
        idx = dof_index.get(name)
        if idx is not None and idx < len(targets) and idx < len(default):
            targets[idx] = default[idx] + float(delta)
            applied.append(name)
    return applied


def _joint_targets_for_pose(
    default,
    dof_index,
    *,
    phase,
    moving,
    manipulation_ready: bool = False,
    manipulation_reach_arm: str = "both",
):
    import numpy as np  # type: ignore
    targets = np.array(default, dtype="float32", copy=True)
    _apply_joint_deltas(targets, default, dof_index, policy_mod.gait_joint_deltas(phase, moving))
    if manipulation_ready:
        _apply_joint_deltas(
            targets,
            default,
            dof_index,
            manipulation_ready_arm_joint_deltas(manipulation_reach_arm),
        )
    return targets


def _apply_articulation_joint_targets(art, targets):
    """Prefer Isaac's articulation action path, falling back to direct joint state writes on older
    worker images. The return value is persisted in the contact report for auditability."""
    try:
        from isaacsim.core.utils.types import ArticulationAction  # type: ignore
        art.apply_action(ArticulationAction(joint_positions=targets))
        return "articulation_action_position_targets"
    except Exception:  # noqa: BLE001
        art.set_joint_positions(targets)
        return "direct_joint_state_position_set"


def _drive_g1_walk(
    art,
    dof_index,
    default,
    *,
    root_pose,
    yaw,
    phase,
    moving,
    manipulation_ready: bool = False,
    manipulation_reach_arm: str = "both",
):
    """Set the G1 root world pose + joint positions = standing + gait deltas (kinematic pose)."""
    import numpy as np  # type: ignore
    w, x, y, z = yaw_to_quat(float(yaw))
    art.set_world_pose(position=np.asarray(root_pose, dtype="float32"),
                       orientation=np.asarray([w, x, y, z], dtype="float32"))
    targets = _joint_targets_for_pose(
        default,
        dof_index,
        phase=phase,
        moving=moving,
        manipulation_ready=manipulation_ready,
        manipulation_reach_arm=manipulation_reach_arm,
    )
    art.set_joint_positions(targets)
    return targets


def _g1_skeleton_world_positions(art, link_names):
    """World-space positions of the G1 links (the skeleton landmarks before projection)."""
    import numpy as np  # type: ignore
    positions, _ = art.get_link_world_poses()
    positions = np.asarray(positions)
    return [(link_names[i] if i < len(link_names) else f"link_{i}",
             (float(positions[i][0]), float(positions[i][1]), float(positions[i][2])))
            for i in range(len(positions))]


def _project_skeleton(skeleton_world, *, eye, target, up, vfov_deg, width, height):
    """Project G1 link world positions into the camera -> OSCAR-schema landmark list. Each landmark
    is {landmark_id, image_projection:{available,u_px,v_px,depth_m}} (the exact shape the OSCAR WAM
    input-package materialization reads)."""
    landmarks = []
    for name, wp in skeleton_world:
        px = project_point_to_pixel(wp, eye, target, up, vfov_deg, width, height)
        if px is not None:
            landmarks.append({"landmark_id": name, "image_projection": {
                "available": True, "u_px": round(px[0], 2), "v_px": round(px[1], 2),
                "depth_m": round(px[2], 4)}})
    return landmarks


def _g1_link_rest_offsets(stage, prim_path: str):
    """Pure-USD G1 skeleton: rest-pose offset (in the root frame) of each link prim under the G1.
    No physics/tensor-view (which gets invalidated on this G1 USD) — just the link transforms.
    Returns [(name, (dx,dy,dz)), ...]. Per-step world = root_pose + Rz(yaw) @ offset."""
    from pxr import Usd, UsdGeom  # type: ignore
    xc = UsdGeom.XformCache()
    root_prim = stage.GetPrimAtPath(prim_path)
    rt = xc.GetLocalToWorldTransform(root_prim).ExtractTranslation()
    root = (float(rt[0]), float(rt[1]), float(rt[2]))
    offs = []
    for prim in Usd.PrimRange(root_prim):
        name = prim.GetName()
        if "link" not in name.lower() or not prim.IsA(UsdGeom.Xformable):
            continue
        t = xc.GetLocalToWorldTransform(prim).ExtractTranslation()
        offs.append((name, (float(t[0]) - root[0], float(t[1]) - root[1], float(t[2]) - root[2])))
    return offs


def _rest_skeleton_world(offsets, root_pose, yaw):
    """Place the rest-pose link offsets at the robot's per-step root pose (translate + Z-rotate)."""
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    out = []
    for name, (ox, oy, oz) in offsets:
        out.append((name, (root_pose[0] + cy * ox - sy * oy,
                           root_pose[1] + sy * ox + cy * oy,
                           root_pose[2] + oz)))
    return out


def skeleton_world_for_frame(*, art_ctx, rest_offsets, root_pose, yaw):
    """Return the best available G1 skeleton for a rendered frame.

    Some Isaac worker images expose a controllable G1 articulation with valid joints but no body
    names. Reading link poses in that state can invalidate the PhysX tensor view, so fall back to
    the USD rest-offset skeleton unless the articulation has usable link names.
    """
    if art_ctx is not None and art_ctx.get("link_names"):
        try:
            return _g1_skeleton_world_positions(art_ctx["art"], art_ctx["link_names"])
        except Exception as exc:  # noqa: BLE001
            _log(f"G1 articulation skeleton read failed ({exc!r}); using USD skeleton fallback")
    if rest_offsets is not None:
        return _rest_skeleton_world(rest_offsets, root_pose, yaw)
    return []


def compute_arm_reach_skeleton(skeleton, target, reach_frac, *, arm: str = "right"):
    """Re-pose one arm of a world-space skeleton so its hand reaches toward ``target`` (the faucet).

    The walk policy never moves the arms, so the skeleton (OSCAR's action conditioning) just shows a
    rigid robot. This rotates the arm chain about the shoulder so the hand travels from its rest spot
    to the target as ``reach_frac`` goes 0->1 — turning the skeleton-video into an actual reach. Each
    arm link keeps its rest fractional distance from the shoulder (rigid straight-arm reach), and the
    reach is clamped to the arm's length so it never overstretches. Pure geometry, GPU-independent.

    ``skeleton`` is ``[(name, (x,y,z)), ...]``; returns the same shape with the arm links re-placed.
    """
    if target is None or reach_frac <= 0.0:
        return skeleton
    if str(arm).lower() == "both":
        out = skeleton
        for side in ("left", "right"):
            out = compute_arm_reach_skeleton(out, target, reach_frac, arm=side)
        return out
    arm_keys = ("shoulder", "elbow", "wrist", "hand")
    prefix = f"{arm}_"
    arm_pts = [(n, p) for n, p in skeleton if n.startswith(prefix) and any(k in n for k in arm_keys)]
    sh = [p for n, p in arm_pts if "shoulder" in n]
    hand = [p for n, p in arm_pts if "hand" in n]
    if not sh or not hand:
        return skeleton

    def centroid(ps):
        return tuple(sum(c) / len(ps) for c in zip(*ps))

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def add(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def scale(a, s):
        return (a[0] * s, a[1] * s, a[2] * s)

    def length(a):
        return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

    shoulder = centroid(sh)
    hand_rest = centroid(hand)
    arm_len = length(sub(hand_rest, shoulder)) or 1e-6
    to_target = sub(target, shoulder)
    tlen = length(to_target) or 1e-6
    reach_dist = min(arm_len, tlen)
    hand_reach = add(shoulder, scale(to_target, reach_dist / tlen))  # clamped along shoulder->target
    frac = max(0.0, min(1.0, float(reach_frac)))
    hand_now = add(scale(hand_rest, 1.0 - frac), scale(hand_reach, frac))
    out = []
    for n, p in skeleton:
        if n.startswith(prefix) and any(k in n for k in arm_keys):
            f = length(sub(p, shoulder)) / arm_len  # rest fractional distance along the arm
            out.append((n, add(shoulder, scale(sub(hand_now, shoulder), f))))
        else:
            out.append((n, p))
    return out


def arm_reach_rotation(shoulder, rest_elbow, target, reach_frac):
    """Axis (unit xyz) + angle (radians) of the kinematic SHOULDER rotation that swings the rest
    upper-arm bone (shoulder->rest_elbow) toward the target (shoulder->target), scaled by reach_frac.

    Axis-agnostic: it derives the rotation from the rest bone and the desired bone direction, so there
    is NO hardcoded joint axis to inspect on the G1 USD. Applied about the shoulder pivot it points the
    upper arm at the object; the elbow/wrist/hand follow rigidly. Pure geometry, GPU-independent."""
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def length(a):
        return math.sqrt(dot(a, a))

    def norm(a):
        L = length(a) or 1e-9
        return (a[0] / L, a[1] / L, a[2] / L)

    rest = norm(sub(rest_elbow, shoulder))
    want = norm(sub(target, shoulder))
    d = max(-1.0, min(1.0, dot(rest, want)))
    angle = math.acos(d) * max(0.0, min(1.0, float(reach_frac)))
    axis = cross(rest, want)
    if length(axis) < 1e-6:
        axis = (0.0, 0.0, 1.0)  # rest ~parallel/antiparallel to want -> arbitrary axis (angle ~0/pi)
    return norm(axis), angle


def _find_arm_link(links: dict, *keys: str):
    """First Xformable link prim whose name contains ALL keys (case-insensitive)."""
    for name, prim in links.items():
        low = name.lower()
        if all(k.lower() in low for k in keys):
            return prim
    return None


def _pose_arm_kinematic_usd(stage, prim_path: str, target, *, arm: str = "right",
                            reach_frac: float = 1.0) -> int:
    """Kinematically pose the G1 arm(s) into a manipulation-ready forward seed.

    Pure USD: rotate the shoulder link about its pivot so the shoulder->effector direction points
    toward the task workspace; children follow at the arm's natural length. This makes the forearm and
    gripper visible in the robot POV without claiming contact or task completion. No physics tensor
    view, so it cannot trigger the articulation-drive crash. With ``arm="both"`` both arms move
    forward for the egocentric seed. Returns the number of arms posed. GPU/USD only.
    """
    from pxr import Usd, UsdGeom, Gf  # type: ignore
    sides = ("left", "right") if arm == "both" else (arm,)
    root = stage.GetPrimAtPath(prim_path)
    links = {p.GetName(): p for p in Usd.PrimRange(root)
             if p.IsA(UsdGeom.Xformable) and "link" in p.GetName().lower()}
    posed = 0
    for side in sides:
        shoulder = (_find_arm_link(links, side, "shoulder", "pitch")
                    or _find_arm_link(links, side, "shoulder"))
        # Align shoulder->effector with shoulder->workspace so the arm is extended forward in the
        # initial seed. The effector is not translated to the affordance; distance remains metadata for
        # the downstream policy/WAM rollout.
        effector = (_find_arm_link(links, side, "hand") or _find_arm_link(links, side, "palm")
                    or _find_arm_link(links, side, "wrist") or _find_arm_link(links, side, "elbow"))
        if shoulder is None or effector is None:
            continue
        xc = UsdGeom.XformCache()  # fresh cache per arm (previous arm's mutation invalidated it)
        sh_w = xc.GetLocalToWorldTransform(shoulder)
        el_w = xc.GetLocalToWorldTransform(effector)
        sp = sh_w.ExtractTranslation()
        ep = el_w.ExtractTranslation()
        axis, angle = arm_reach_rotation((sp[0], sp[1], sp[2]), (ep[0], ep[1], ep[2]),
                                         (float(target[0]), float(target[1]), float(target[2])),
                                         reach_frac)
        if angle < 1e-4:
            continue
        rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*axis), math.degrees(angle)))
        pivot = Gf.Vec3d(sp[0], sp[1], sp[2])
        # rotate the shoulder's world transform about the shoulder pivot (USD row-vector convention)
        m_pivot = Gf.Matrix4d().SetTranslate(-pivot) * rot * Gf.Matrix4d().SetTranslate(pivot)
        new_world = sh_w * m_pivot
        parent_world = xc.GetLocalToWorldTransform(shoulder.GetParent())
        new_local = new_world * parent_world.GetInverse()
        xf = UsdGeom.Xformable(shoulder)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(new_local)
        posed += 1
    return posed


def _capture_robot_neutral_descendant_xforms(stage, prim_path: str) -> dict[str, Any]:
    """Capture robot descendant local transforms before any per-task arm posing mutates the stage.

    Warm workers reuse one USD stage for many jobs. Any pure-USD reach pose must therefore be applied
    from the same neutral robot seed every time, otherwise arm/link transforms compound across jobs.
    Root placement is intentionally excluded; it is re-authored from the dynamic stance each frame.
    """
    from pxr import Usd, UsdGeom  # type: ignore
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return {}
    xc = UsdGeom.XformCache()
    neutral: dict[str, Any] = {}
    for prim in Usd.PrimRange(root):
        if prim.GetPath() == root.GetPath():
            continue
        try:
            if not prim.IsA(UsdGeom.Xformable):
                continue
            parent_world = xc.GetLocalToWorldTransform(prim.GetParent())
            world = xc.GetLocalToWorldTransform(prim)
            neutral[str(prim.GetPath())] = world * parent_world.GetInverse()
        except Exception:  # noqa: BLE001
            continue
    return neutral


def _restore_robot_neutral_descendant_xforms(stage, neutral_xforms: Mapping[str, Any]) -> int:
    """Restore descendant local transforms captured by
    :func:`_capture_robot_neutral_descendant_xforms`.
    """
    if not neutral_xforms:
        return 0
    from pxr import UsdGeom  # type: ignore
    restored = 0
    for path, local_matrix in neutral_xforms.items():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        try:
            if not prim.IsA(UsdGeom.Xformable):
                continue
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(local_matrix)
            restored += 1
        except Exception:  # noqa: BLE001
            continue
    return restored


def _setup_articulated_g1(prim_path: str, *, gravity_z: float = 0.0):
    """Create a physics SimulationContext (gravity OFF, so the kinematic walk pose holds without the
    G1 collapsing), play it, and initialize the G1 articulation for joint driving + link readback.
    Returns a context dict. GPU-only."""
    from isaacsim.core.api import SimulationContext  # type: ignore
    ctx = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
    ctx.initialize_physics()
    try:
        ctx.get_physics_context().set_gravity(float(gravity_z))
    except Exception:  # noqa: BLE001
        pass
    art, dof_index, default, link_names = _setup_g1_articulation(prim_path)
    ctx.play()
    return {"ctx": ctx, "art": art, "dof_index": dof_index, "default": default,
            "link_names": link_names, "dof_count": len(dof_index), "gravity_z": float(gravity_z)}


def _setup_physics_context_only(*, gravity_z: float = -9.81):
    """Create/play a PhysX SimulationContext without creating a SingleArticulation tensor view.

    The official G1 USD currently invalidates Isaac's tensor view when this runner drives or reads
    the articulation through SingleArticulation. Dynamic-standing contact proof only needs the
    authored USD articulation, gravity, collisions, and contact reporting, so this mode avoids the
    tensor API entirely.
    """
    from isaacsim.core.api import SimulationContext  # type: ignore
    ctx = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
    ctx.initialize_physics()
    try:
        ctx.get_physics_context().set_gravity(float(gravity_z))
    except Exception:  # noqa: BLE001
        pass
    ctx.play()
    return {
        "ctx": ctx,
        "art": None,
        "dof_index": {},
        "default": [],
        "link_names": [],
        "dof_count": 0,
        "gravity_z": float(gravity_z),
        "tensor_view_used": False,
    }


def _sim_step(ctx, *, render: bool = False) -> None:
    try:
        ctx.step(render=render)
    except TypeError:
        ctx.step()


def _safe_articulation_world_pose(art) -> dict[str, Any]:
    try:
        pos, quat = art.get_world_pose()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}
    try:
        return {
            "available": True,
            "position_xyz": [round(float(v), 6) for v in pos],
            "orientation_wxyz": [round(float(v), 6) for v in quat],
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}


def _safe_usd_root_world_pose(stage, prim_path: str) -> dict[str, Any]:
    try:
        from pxr import UsdGeom  # type: ignore
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return {"available": False, "error": "prim_not_found"}
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        pos = transform.ExtractTranslation()
        return {
            "available": True,
            "position_xyz": [round(float(pos[i]), 6) for i in range(3)],
            "source": "usd_xform_cache",
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}


def _path_from_encoded_sdf(value) -> str:
    try:
        from pxr import PhysicsSchemaTools  # type: ignore
        return str(PhysicsSchemaTools.intToSdfPath(int(value)))
    except Exception:  # noqa: BLE001
        return str(value)


def _vec3_to_list(value) -> list[float] | None:
    try:
        return [round(float(value[i]), 6) for i in range(3)]
    except Exception:  # noqa: BLE001
        try:
            return [round(float(getattr(value, axis)), 6) for axis in ("x", "y", "z")]
        except Exception:  # noqa: BLE001
            return None


def _enable_contact_reports(stage, robot_prim_path: str, *, threshold: float = 0.0) -> dict[str, Any]:
    """Apply PhysX contact-report API to the articulation/root and likely foot links.

    This is best-effort because Isaac worker images and G1 USD variants differ. A failure should
    block only the contact report, not the render path.
    """
    try:
        from pxr import PhysxSchema, Usd, UsdPhysics  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "error": repr(exc), "enabled_paths": []}
    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        return {"status": "unavailable", "error": "robot_prim_not_found", "enabled_paths": []}
    enabled: list[str] = []
    candidates = []
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        if (
            prim.GetPath() == root.GetPath()
            or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            or prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or ("foot" in name and prim.HasAPI(UsdPhysics.CollisionAPI))
        ):
            candidates.append(prim)
    for prim in candidates:
        try:
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api.CreateThresholdAttr().Set(float(threshold))
            enabled.append(str(prim.GetPath()))
        except Exception:  # noqa: BLE001
            continue
    return {"status": "enabled" if enabled else "unavailable", "enabled_paths": enabled}


def _contact_report_records(robot_prim_path: str, *, max_records: int = 40) -> list[dict[str, Any]]:
    try:
        from omni.physx import get_physx_simulation_interface  # type: ignore
    except Exception:  # noqa: BLE001
        return []
    try:
        report = get_physx_simulation_interface().get_contact_report()
    except Exception:  # noqa: BLE001
        return []
    if not report:
        return []
    try:
        headers, data = report[0], report[1] if len(report) > 1 else []
    except Exception:  # noqa: BLE001
        return []
    records: list[dict[str, Any]] = []
    for header in list(headers)[:max_records]:
        actor0 = _path_from_encoded_sdf(getattr(header, "actor0", ""))
        actor1 = _path_from_encoded_sdf(getattr(header, "actor1", ""))
        collider0 = _path_from_encoded_sdf(getattr(header, "collider0", actor0))
        collider1 = _path_from_encoded_sdf(getattr(header, "collider1", actor1))
        joined = " ".join((actor0, actor1, collider0, collider1)).lower()
        if robot_prim_path.lower() not in joined and "/world/g1" not in joined:
            continue
        offset = int(getattr(header, "contact_data_offset", 0) or 0)
        count = int(getattr(header, "num_contact_data", 0) or 0)
        samples = []
        for sample in list(data)[offset: offset + min(count, 3)]:
            samples.append({
                "position_xyz": _vec3_to_list(getattr(sample, "position", None)),
                "normal_xyz": _vec3_to_list(getattr(sample, "normal", None)),
                "impulse": (
                    round(float(getattr(sample, "impulse", 0.0) or 0.0), 6)
                    if hasattr(sample, "impulse") else None
                ),
            })
        records.append({
            "actor0": actor0,
            "actor1": actor1,
            "collider0": collider0,
            "collider1": collider1,
            "contact_data_count": count,
            "samples": samples,
        })
    return records


def _is_support_contact(record: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(record.get(key) or "").lower()
        for key in ("actor0", "actor1", "collider0", "collider1")
    )
    return ("foot" in text or "ankle" in text or "toe" in text) and (
        "floor" in text or "ground" in text or "room" in text or "kitchen" in text
    )


def _settle_dynamic_standing_contacts(
    *,
    stage,
    art_ctx,
    robot_prim_path: str,
    root_pose,
    yaw,
    phase,
    moving,
    settle_steps: int,
    scenario_id: str,
    manipulation_ready: bool = False,
    manipulation_reach_arm: str = "both",
    root_pose_seeded_before_tensor_view: bool = True,
) -> dict[str, Any]:
    """Run a bounded PhysX standing/contact settle without mutating the G1 USD xform after the
    articulation tensor view exists.

    The policy route remains kinematic. This mode upgrades each sampled placement by stepping the
    real articulation against the scene with gravity and contact reporting; it is not a full dynamic
    walking controller.
    """
    art = art_ctx.get("art")
    ctx = art_ctx["ctx"]
    tensor_view_used = art is not None
    targets = None
    command_mode = "usd_physx_articulation_default_drives_no_tensor_view"
    if tensor_view_used:
        targets = _joint_targets_for_pose(
            art_ctx["default"],
            art_ctx["dof_index"],
            phase=phase,
            moving=moving,
            manipulation_ready=manipulation_ready,
            manipulation_reach_arm=manipulation_reach_arm,
        )
        # Do not call art.set_world_pose() here. On the official G1 USD that invalidates the
        # PhysX tensor view with the same failure as mutating the USD root xform after initialization.
        # Dynamic-standing mode pre-seeds the root USD transform before the physics context is played.
        command_mode = _apply_articulation_joint_targets(art, targets)
    before = (
        _safe_articulation_world_pose(art)
        if tensor_view_used else _safe_usd_root_world_pose(stage, robot_prim_path)
    )
    contact_setup = _enable_contact_reports(stage, robot_prim_path)
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    executed = 0
    for _ in range(max(0, int(settle_steps))):
        try:
            if tensor_view_used:
                _apply_articulation_joint_targets(art, targets)
            _sim_step(ctx, render=False)
            executed += 1
            if len(records) < 80:
                records.extend(_contact_report_records(robot_prim_path, max_records=20))
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
            break
    after = (
        _safe_articulation_world_pose(art)
        if tensor_view_used else _safe_usd_root_world_pose(stage, robot_prim_path)
    )
    support_records = [r for r in records if _is_support_contact(r)]
    return {
        "schema_version": "isaac_g1_physics_articulation_standing_contact_report.v1",
        "status": "completed" if executed == max(0, int(settle_steps)) and not errors else "blocked",
        "scenario_id": scenario_id,
        "gravity_z": art_ctx.get("gravity_z"),
        "tensor_view_used": tensor_view_used,
        "requested_settle_steps": int(settle_steps),
        "executed_settle_steps": executed,
        "seed_root_pose_xyz": [round(float(v), 6) for v in root_pose],
        "seed_root_yaw_rad": round(float(yaw), 6),
        "root_pose_seeded_before_tensor_view": bool(root_pose_seeded_before_tensor_view),
        "root_pose_seeded_once_before_settle": False,
        "root_pose_teleport_during_physics_settle": False,
        "usd_root_xform_mutated_after_tensor_view": False,
        "joint_command_mode": command_mode,
        "contact_report_setup": contact_setup,
        "contact_event_count": len(records),
        "support_contact_event_count": len(support_records),
        "sample_contact_records": records[:20],
        "root_pose_before_settle": before,
        "root_pose_after_settle": after,
        "errors": errors,
        "claim_boundary": (
            "Physics articulation standing/contact settle for this sampled placement only; not "
            "full dynamic walking, learned balance control, task success, safety validation, or "
            "deployment readiness."
        ),
    }


def _overlap_probe(robot_prim_path: str, ground_prim_path: str = "/World/GroundPlane"):
    """Return probe(pose, yaw) -> scene-collision hit count using a PhysX box overlap of the
    robot footprint at the candidate pose, excluding the robot's own prims and the ground."""
    from omni.physx import get_physx_scene_query_interface  # type: ignore
    import carb  # type: ignore

    sqi = get_physx_scene_query_interface()
    hx, hy, hz = ROBOT_FOOTPRINT_HALF_EXTENT

    def probe(pose, yaw) -> int:
        hits = {"n": 0}

        def report(hit):  # noqa: ANN001
            path = str(getattr(hit, "collision", "") or getattr(hit, "rigid_body", ""))
            if not path.startswith(robot_prim_path) and not path.startswith(ground_prim_path):
                hits["n"] += 1
            return True  # keep scanning

        w, x, y, z = yaw_to_quat(float(yaw))
        sqi.overlap_box(
            carb.Float3(hx, hy, hz),
            carb.Float3(float(pose[0]), float(pose[1]), float(pose[2])),
            carb.Float4(x, y, z, w),  # PhysX quat order is (x,y,z,w)
            report, False,
        )
        return hits["n"]

    return probe


def _set_root_xform(stage, prim_path: str, pose, yaw) -> None:
    from pxr import UsdGeom, Gf  # type: ignore

    xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))
    xform.AddRotateZOp().Set(math.degrees(float(yaw)))


def _matrix4_to_rows(matrix) -> list[list[float]]:
    rows: list[list[float]] = []
    for i in range(4):
        rows.append([round(float(matrix[i][j]), 6) for j in range(4)])
    return rows


def _xform_op_record(op) -> dict[str, Any]:
    try:
        value = op.Get()
    except Exception as exc:  # noqa: BLE001
        value = repr(exc)
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        try:
            value_out: Any = [round(float(value[i]), 6) for i in range(len(value))]
        except Exception:  # noqa: BLE001
            value_out = str(value)
    else:
        try:
            value_out = round(float(value), 6)
        except Exception:  # noqa: BLE001
            value_out = str(value)
    return {
        "op_name": str(op.GetOpName()),
        "op_type": str(op.GetOpType()),
        "is_inverse": bool(op.IsInverseOp()),
        "value": value_out,
    }


def _prim_transform_snapshot(stage, prim_path: str) -> dict[str, Any]:
    from pxr import UsdGeom  # type: ignore

    prim = stage.GetPrimAtPath(prim_path)
    valid = bool(prim and prim.IsValid())
    snap: dict[str, Any] = {"prim_path": prim_path, "valid": valid}
    if not valid:
        return snap
    try:
        xform = UsdGeom.Xformable(prim)
        snap["xform_ops"] = [_xform_op_record(op) for op in xform.GetOrderedXformOps()]
    except Exception as exc:  # noqa: BLE001
        snap["xform_ops_error"] = repr(exc)
    try:
        cache = UsdGeom.XformCache()
        snap["local_to_world_matrix"] = _matrix4_to_rows(cache.GetLocalToWorldTransform(prim))
    except Exception as exc:  # noqa: BLE001
        snap["local_to_world_error"] = repr(exc)
    return snap


def _root_transform_diagnostics(stage, prim_path: str) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(prim_path)
    parent_path = ""
    if prim and prim.IsValid():
        parent = prim.GetParent()
        if parent and parent.IsValid():
            parent_path = str(parent.GetPath())
    return {
        "root": _prim_transform_snapshot(stage, prim_path),
        "parent": _prim_transform_snapshot(stage, parent_path) if parent_path else None,
        "pseudo_root": _prim_transform_snapshot(stage, str(stage.GetPseudoRoot().GetPath())),
    }


def _robot_upright_report(stage, prim_path: str, *, max_tilt_deg: float = 12.0) -> dict[str, Any]:
    """Verify the placed robot root is upright enough for a policy seed frame."""
    if not hasattr(stage, "GetPrimAtPath"):
        return {
            "schema_version": "robot_upright_report.v1",
            "status": "unverified",
            "blockers": [],
            "prim_path": prim_path,
            "reason": "stage_object_does_not_expose_usd_prim_lookup",
        }
    try:
        from pxr import UsdGeom, Gf  # type: ignore
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return {
                "schema_version": "robot_upright_report.v1",
                "status": "blocked",
                "blockers": ["robot_upright_prim_unavailable"],
                "prim_path": prim_path,
            }
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        try:
            up = matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
            up_vec = (float(up[0]), float(up[1]), float(up[2]))
        except Exception:  # noqa: BLE001
            up_vec = (float(matrix[0][2]), float(matrix[1][2]), float(matrix[2][2]))
        norm = math.sqrt(sum(v * v for v in up_vec)) or 1e-9
        up_unit = tuple(v / norm for v in up_vec)
        cos_tilt = max(-1.0, min(1.0, up_unit[2]))
        tilt_deg = math.degrees(math.acos(cos_tilt))
        blockers = [] if tilt_deg <= float(max_tilt_deg) else ["robot_root_not_upright"]
        return {
            "schema_version": "robot_upright_report.v1",
            "status": "passed" if not blockers else "blocked",
            "blockers": blockers,
            "prim_path": prim_path,
            "root_up_vector_world": [round(float(v), 6) for v in up_unit],
            "tilt_deg": round(float(tilt_deg), 4),
            "max_tilt_deg": round(float(max_tilt_deg), 4),
            "claim_boundary": (
                "Upright validation checks root orientation for an initial policy seed only; it is not "
                "dynamic balance, locomotion, safety, or deployment validation."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": "robot_upright_report.v1",
            "status": "blocked",
            "blockers": ["robot_upright_report_failed"],
            "prim_path": prim_path,
            "error": repr(exc),
        }


def _world_bbox_for_prim(stage, prim_path: str) -> dict[str, list[float]] | None:
    """Compute an aligned world bbox for a prim after its current stage transform is applied."""
    from pxr import Usd, UsdGeom  # type: ignore

    prim = stage.GetPrimAtPath(prim_path)
    is_valid = getattr(prim, "IsValid", None)
    if prim is None or (callable(is_valid) and not is_valid()):
        return None
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if box.IsEmpty():
        return None
    bbox_min, bbox_max, center, size = _aligned_box_min_max_center_size(box)
    return {
        "bbox_min_xyz": _rounded_xyz(bbox_min),
        "bbox_max_xyz": _rounded_xyz(bbox_max),
        "center_xyz": _rounded_xyz(center),
        "size_xyz": _rounded_xyz(size),
    }


def _footprint_center_xy_from_bbox(bbox: Mapping[str, Sequence[float]]) -> list[float]:
    bmin = bbox["bbox_min_xyz"]
    bmax = bbox["bbox_max_xyz"]
    return [
        round(0.5 * (float(bmin[0]) + float(bmax[0])), 6),
        round(0.5 * (float(bmin[1]) + float(bmax[1])), 6),
    ]


def _xy_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _place_root(
    stage,
    prim_path: str,
    pose,
    yaw,
    *,
    align_footprint_center: bool = True,
    max_xy_error_m: float = PLACEMENT_GROUND_TRUTH_MAX_FOOTPRINT_CENTER_DELTA_M,
) -> dict[str, Any]:
    """Place the G1 root and align its actual USD footprint center to ``pose``.

    The G1 asset root is not assumed to be the pelvis/footprint center. We first apply the requested
    transform, query the actual world AABB with ``UsdGeom.BBoxCache``, and, when the footprint center
    misses the requested pose by more than ``max_xy_error_m``, translate the root by that measured
    world-frame offset. Callers that ignore the return value still get corrected placement; the
    diagnostics are written into ``placement_validation.json`` by the manipulation runner.
    """
    requested_pose = (float(pose[0]), float(pose[1]), float(pose[2]))
    requested_xy = [requested_pose[0], requested_pose[1]]
    _set_root_xform(stage, prim_path, requested_pose, yaw)
    diagnostics: dict[str, Any] = {
        "schema_version": "place_root_diagnostics.v1",
        "status": "placed",
        "prim_path": prim_path,
        "requested_pose_xyz": _rounded_xyz(requested_pose),
        "requested_yaw_rad": round(float(yaw), 6),
        "max_xy_error_m": round(float(max_xy_error_m), 6),
        "footprint_center_alignment_enabled": bool(align_footprint_center),
        "correction_applied": False,
    }
    if not align_footprint_center:
        return diagnostics
    initial_bbox = _world_bbox_for_prim(stage, prim_path)
    if initial_bbox is None:
        diagnostics["status"] = "bbox_unavailable"
        diagnostics["blockers"] = ["placed_robot_bbox_unavailable"]
        diagnostics["xform_diagnostics"] = _root_transform_diagnostics(stage, prim_path)
        return diagnostics
    initial_center = _footprint_center_xy_from_bbox(initial_bbox)
    initial_error = _xy_distance(initial_center, requested_xy)
    diagnostics["initial_world_aabb"] = initial_bbox
    diagnostics["initial_footprint_center_xy"] = initial_center
    diagnostics["initial_xy_error_m"] = round(float(initial_error), 6)
    if initial_error <= max_xy_error_m:
        diagnostics["final_world_aabb"] = initial_bbox
        diagnostics["final_footprint_center_xy"] = initial_center
        diagnostics["final_xy_error_m"] = diagnostics["initial_xy_error_m"]
        return diagnostics

    dx = float(initial_center[0]) - requested_pose[0]
    dy = float(initial_center[1]) - requested_pose[1]
    corrected_pose = (requested_pose[0] - dx, requested_pose[1] - dy, requested_pose[2])
    diagnostics["status"] = "corrected"
    diagnostics["correction_applied"] = True
    diagnostics["measured_offset_xy_m"] = [round(dx, 6), round(dy, 6)]
    diagnostics["corrected_root_translation_xyz"] = _rounded_xyz(corrected_pose)
    diagnostics["xform_diagnostics_before_correction"] = _root_transform_diagnostics(stage, prim_path)
    _set_root_xform(stage, prim_path, corrected_pose, yaw)
    final_bbox = _world_bbox_for_prim(stage, prim_path)
    if final_bbox is None:
        diagnostics["status"] = "blocked"
        diagnostics["blockers"] = ["placed_robot_bbox_unavailable_after_correction"]
        diagnostics["xform_diagnostics_after_correction"] = _root_transform_diagnostics(stage, prim_path)
        return diagnostics
    final_center = _footprint_center_xy_from_bbox(final_bbox)
    final_error = _xy_distance(final_center, requested_xy)
    diagnostics["final_world_aabb"] = final_bbox
    diagnostics["final_footprint_center_xy"] = final_center
    diagnostics["final_xy_error_m"] = round(float(final_error), 6)
    diagnostics["xform_diagnostics_after_correction"] = _root_transform_diagnostics(stage, prim_path)
    if final_error > max_xy_error_m:
        diagnostics["status"] = "blocked"
        diagnostics["blockers"] = ["placed_robot_footprint_center_mismatch_after_correction"]
    return diagnostics


def _placement_validator_for_stage(
    stage,
    robot_prim_path: str,
    target_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    *,
    target_object=None,
    scene_objects: Sequence[Any] | None = None,
    floor_z: float | None = None,
    max_root_to_bbox_center_xy_m: float = PLACEMENT_GROUND_TRUTH_MAX_FOOTPRINT_CENTER_DELTA_M,
    min_target_gap_m: float = 0.05,
    standoff_tolerance_m: float = 0.2,
    standoff_range: tuple[float, float] | None = None,
):
    """Validate the placed G1 geometry, not just the planned root pose.

    This catches the kitchen failure mode where the stance planner reports an aisle root but the
    referenced G1 asset lands with its visible/collidable bbox in the sink or cabinet footprint.
    """
    target_min, target_max = target_bounds
    target_center = (
        0.5 * (float(target_min[0]) + float(target_max[0])),
        0.5 * (float(target_min[1]) + float(target_max[1])),
        0.5 * (float(target_min[2]) + float(target_max[2])),
    )
    validation_standoff_range = (
        tuple(float(v) for v in standoff_range)
        if standoff_range is not None
        else TASK_STANCE_DEFAULT_VALIDATION_STANDOFF_RANGE_M
    )

    def validate(pose, yaw, record: Mapping[str, Any] | None = None) -> dict[str, Any]:
        root_diagnostics = _place_root(stage, robot_prim_path, pose, yaw)
        bbox = _world_bbox_for_prim(stage, robot_prim_path)
        result: dict[str, Any] = {
            "status": "accepted",
            "blockers": [],
            "robot_prim_path": robot_prim_path,
            "planned_pose": _rounded_xyz(pose),
            "planned_yaw": round(float(yaw), 6),
            "max_root_to_bbox_center_xy_m": round(float(max_root_to_bbox_center_xy_m), 6),
            "min_target_gap_m": round(float(min_target_gap_m), 6),
            "validation_standoff_range_m": [
                round(float(validation_standoff_range[0]), 6),
                round(float(validation_standoff_range[1]), 6),
            ],
            "place_root_diagnostics": root_diagnostics,
        }
        if bbox is None:
            result["status"] = "blocked"
            result["blockers"] = ["placed_robot_bbox_unavailable"]
            return result
        result["placed_robot_bbox"] = bbox
        bbox_center = bbox["center_xyz"]
        root_to_center_xy = math.hypot(
            float(bbox_center[0]) - float(pose[0]),
            float(bbox_center[1]) - float(pose[1]),
        )
        relation = _xy_rect_overlap_and_gap(
            bbox["bbox_min_xyz"],
            bbox["bbox_max_xyz"],
            target_min,
            target_max,
        )
        result["root_to_robot_bbox_center_xy_m"] = round(float(root_to_center_xy), 6)
        result["target_bbox_relation"] = relation
        blockers: list[str] = []
        if target_object is not None and scene_objects is not None:
            try:
                from blueprint_pipeline.scene_placement import validate_stand_pose  # type: ignore
            except Exception:
                from scene_placement import validate_stand_pose  # type: ignore
            deterministic_floor_z = (
                float(floor_z)
                if floor_z is not None
                else float(pose[2]) - ROBOT_PELVIS_HEIGHT_M
            )
            obstacles = list(scene_objects)
            if not any(
                str(getattr(o, "id", "")) == str(getattr(target_object, "id", ""))
                for o in obstacles
            ):
                obstacles.append(target_object)
            verdict = validate_stand_pose(
                tuple(float(v) for v in pose),
                float(yaw),
                target_object,
                obstacles,
                floor_z=deterministic_floor_z,
                footprint_half_extent=ROBOT_FOOTPRINT_HALF_EXTENT,
                pelvis_height=ROBOT_PELVIS_HEIGHT_M,
                max_facing_error_deg=30.0,
                standoff_range=validation_standoff_range,
                standoff_obstacles=_find_standoff_fixtures(obstacles, target_object),
            )
            adjusted_verdict, suppressed_clips = _adjust_verdict_for_broad_aabb_false_positives(
                verdict=verdict,
                obstacles=obstacles,
                target_obj=target_object,
                record=record,
            )
            if suppressed_clips:
                result["deterministic_geometry_raw"] = _placement_verdict_to_dict(verdict)
                result["deterministic_geometry_adjustments"] = {
                    "suppressed_broad_aabb_clips": suppressed_clips,
                    "claim_boundary": (
                        "Suppression is allowed only after PhysX reports zero contacts; it corrects "
                        "coarse USD AABB occupancy, not physical collision or task success."
                    ),
                }
                verdict = adjusted_verdict
            result["deterministic_geometry"] = _placement_verdict_to_dict(verdict)
            if not verdict.ok:
                blockers.append("placement_geometry_invalid")
        if root_to_center_xy > max_root_to_bbox_center_xy_m:
            blockers.append("placed_robot_bbox_center_far_from_root_pose")
        if relation["overlaps_xy"]:
            blockers.append("placed_robot_bbox_overlaps_target_bbox")
        else:
            required_gap = float(min_target_gap_m)
            if record is not None:
                try:
                    planned_standoff = float(record.get("standoff_from_target_surface_m"))
                except Exception:  # noqa: BLE001
                    planned_standoff = 0.0
                if planned_standoff > 0.0:
                    dx = target_center[0] - float(pose[0])
                    dy = target_center[1] - float(pose[1])
                    mag = math.hypot(dx, dy)
                    if mag > 1e-6:
                        robot_half_extent = _half_extent_along_bounds(
                            (
                                tuple(float(v) for v in bbox["bbox_min_xyz"]),
                                tuple(float(v) for v in bbox["bbox_max_xyz"]),
                            ),
                            (dx / mag, dy / mag),
                        )
                        required_gap = max(
                            required_gap,
                            planned_standoff - robot_half_extent - float(standoff_tolerance_m),
                        )
                        result["robot_half_extent_toward_target_m"] = round(
                            float(robot_half_extent), 6
                        )
            result["required_target_gap_m"] = round(float(required_gap), 6)
            if float(relation["gap_m"]) < required_gap:
                blockers.append("placed_robot_target_gap_below_threshold")
        if blockers:
            result["status"] = "blocked"
            result["blockers"] = blockers
        return result

    return validate


def _scene_objects_for_stage(stage) -> list[Any]:
    """Grouped scene-placement object catalog from the current USD stage.

    This stays grouped because it is used for target resolution and human-readable artifact samples:
    a sink assembly should be one target object. Use :func:`_placement_obstacles_for_stage` for clip
    validation, where grouped counter/cabinet AABBs are too coarse.
    """
    bundle_dir = str(Path(__file__).resolve().parent)
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)
    try:
        try:
            from scene_placement import UsdSceneSpatialIndex  # type: ignore
        except Exception:  # noqa: BLE001
            from blueprint_pipeline.scene_placement import UsdSceneSpatialIndex  # type: ignore
        objects = list(UsdSceneSpatialIndex(stage=stage).objects())
    except Exception:
        return []
    filtered = []
    for obj in objects:
        text = f"{getattr(obj, 'id', '')} {getattr(obj, 'label', '')}".lower()
        if any(token in text for token in ("g1", "unitree", "robot", "placementdebug")):
            continue
        filtered.append(obj)
    return filtered


def _placement_obstacles_for_stage(
    stage,
    *,
    focus_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    focus_margin_m: float = 2.5,
) -> list[Any]:
    """Validation-only obstacle catalog from USD.

    Target resolution wants one object per named assembly, but placement clipping wants the opposite:
    a broad cabinet/counter assembly AABB can cover open aisle floor. Prefer fine leaf-Gprim boxes
    when the index exposes them, and fall back to grouped objects on older bundles. When target bounds
    are known, keep only the target neighborhood so validation cannot spend the startup window walking
    unrelated decorative geometry.
    """
    bundle_dir = str(Path(__file__).resolve().parent)
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)
    try:
        try:
            from scene_placement import UsdSceneSpatialIndex  # type: ignore
        except Exception:  # noqa: BLE001
            from blueprint_pipeline.scene_placement import UsdSceneSpatialIndex  # type: ignore
        index = UsdSceneSpatialIndex(stage=stage)
        obstacle_boxes = getattr(index, "obstacle_boxes", None)
        objects = list(obstacle_boxes() if callable(obstacle_boxes) else index.objects())
    except Exception:
        objects = []
    filtered = []
    for obj in objects:
        text = f"{getattr(obj, 'id', '')} {getattr(obj, 'label', '')}".lower()
        if any(token in text for token in ("g1", "unitree", "robot", "placementdebug")):
            continue
        if not _xy_focus_overlap(obj, focus_bounds, margin_m=focus_margin_m):
            continue
        filtered.append(obj)
    shell = [
        obj
        for obj in _placement_shell_obstacles_for_stage(stage)
        if _xy_focus_overlap(obj, focus_bounds, margin_m=focus_margin_m)
    ]
    return filtered + shell


def _placement_shell_obstacles_for_stage(stage) -> list[Any]:
    """Validation-only wall/window/door obstacles.

    ``UsdSceneSpatialIndex`` intentionally excludes the structural shell for object targeting. Robot
    placement needs the opposite for walls: a pose outside or through the room shell is invalid even
    when it does not overlap a movable object. Keep floor/ground/ceiling/lights excluded so standing
    on the floor remains valid.
    """
    traverse = getattr(stage, "Traverse", None)
    if not callable(traverse):
        return []
    try:
        from pxr import Usd, UsdGeom  # type: ignore
    except Exception:
        return []
    try:
        from blueprint_pipeline.scene_placement import SceneObject  # type: ignore
    except Exception:
        try:
            from scene_placement import SceneObject  # type: ignore
        except Exception:
            return []
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    try:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    except TypeError:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    out = []
    seen: set[str] = set()
    include_words = ("wall", "window")
    exclude_words = (
        "floor",
        "ground",
        "ceiling",
        "light",
        "g1",
        "unitree",
        "robot",
        "placementdebug",
        "cabinet",
        "dishwasher",
        "oven",
        "fridge",
        "refrigerator",
    )
    traverse = getattr(stage, "Traverse", None)
    if not callable(traverse):
        return []
    for prim in traverse():
        prim_path = str(prim.GetPath())
        text = f"{prim_path} {prim.GetName()}".lower()
        if not any(word in text for word in include_words):
            continue
        if any(word in text for word in exclude_words):
            continue
        try:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if box.IsEmpty():
                continue
            bbox_min, bbox_max, center, _size = _aligned_box_min_max_center_size(box)
        except Exception:  # noqa: BLE001
            continue
        obj_id = _safe_shell_obstacle_id(prim_path.strip("/") or str(prim.GetName()))
        size_x = abs(float(bbox_max[0]) - float(bbox_min[0]))
        size_y = abs(float(bbox_max[1]) - float(bbox_min[1]))
        if min(size_x, size_y) > 0.35:
            is_wall_mesh = False
            mesh_type = getattr(UsdGeom, "Mesh", None)
            if mesh_type is not None:
                try:
                    is_wall_mesh = bool(prim.IsA(mesh_type))
                except Exception:  # noqa: BLE001
                    is_wall_mesh = False
            if is_wall_mesh and ("wall" in text or "wallcollider" in text):
                for edge_id, edge_min, edge_max in _synthesized_room_edge_shell_boxes(
                    obj_id,
                    bbox_min,
                    bbox_max,
                ):
                    if edge_id in seen:
                        continue
                    seen.add(edge_id)
                    out.append(SceneObject(
                        id=edge_id,
                        label=f"{str(prim.GetName()) or 'wall'}_{edge_id.rsplit('_', 1)[-1]}",
                        bbox_min=edge_min,
                        bbox_max=edge_max,
                        centroid=(
                            0.5 * (edge_min[0] + edge_max[0]),
                            0.5 * (edge_min[1] + edge_max[1]),
                            0.5 * (edge_min[2] + edge_max[2]),
                        ),
                        source="usd_shell",
                    ))
            continue
        if obj_id in seen:
            continue
        seen.add(obj_id)
        out.append(SceneObject(
            id=obj_id,
            label=str(prim.GetName()) or "shell_obstacle",
            bbox_min=(float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
            bbox_max=(float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
            centroid=(float(center[0]), float(center[1]), float(center[2])),
            source="usd_shell",
        ))
    return out


def _synthesized_room_edge_shell_boxes(
    obj_id: str,
    bbox_min: Sequence[float],
    bbox_max: Sequence[float],
    *,
    thickness_m: float = 0.08,
) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """Thin wall-edge proxies for broad room wall meshes.

    Some kitchen assets author multiple wall planes as one connected mesh, so its single AABB spans
    most of the room and cannot be used as a clip obstacle. Dropping it loses the side/back wall
    clearance check entirely. These proxies represent only the AABB shell edges, which is enough to
    reject poses pressed into the side wall or behind the counter without treating the whole room as
    occupied.
    """
    min_x, min_y, min_z = (float(v) for v in bbox_min)
    max_x, max_y, max_z = (float(v) for v in bbox_max)
    t = abs(float(thickness_m))
    return [
        (
            f"{obj_id}_xmin",
            (min_x - 0.5 * t, min_y, min_z),
            (min_x + 0.5 * t, max_y, max_z),
        ),
        (
            f"{obj_id}_xmax",
            (max_x - 0.5 * t, min_y, min_z),
            (max_x + 0.5 * t, max_y, max_z),
        ),
        (
            f"{obj_id}_ymin",
            (min_x, min_y - 0.5 * t, min_z),
            (max_x, min_y + 0.5 * t, max_z),
        ),
        (
            f"{obj_id}_ymax",
            (min_x, max_y - 0.5 * t, min_z),
            (max_x, max_y + 0.5 * t, max_z),
        ),
    ]


def _safe_shell_obstacle_id(value: str) -> str:
    text = (value or "shell_obstacle").strip().lower().replace("/", "_")
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
    return cleaned.strip("_") or "shell_obstacle"


def _target_object_from_stance_plan(stance_plan: Mapping[str, Any]):
    try:
        from blueprint_pipeline.scene_placement import SceneObject  # type: ignore
    except Exception:
        try:
            from scene_placement import SceneObject  # type: ignore
        except Exception:
            return None
    selected = {}
    target_resolution = stance_plan.get("target_resolution")
    if isinstance(target_resolution, Mapping):
        raw_selected = target_resolution.get("selected")
        if isinstance(raw_selected, Mapping):
            selected = dict(raw_selected)
    bounds = stance_plan.get("task_target_bounds")
    bbox_min = bbox_max = None
    if isinstance(bounds, Mapping):
        bbox_min = bounds.get("bbox_min_xyz")
        bbox_max = bounds.get("bbox_max_xyz")
    if bbox_min is None:
        bbox_min = selected.get("bbox_min_xyz")
    if bbox_max is None:
        bbox_max = selected.get("bbox_max_xyz")
    center = (
        selected.get("center_xyz")
        or stance_plan.get("task_target_xyz")
        or selected.get("centroid")
    )
    if bbox_min is None or bbox_max is None or center is None:
        return None
    label = str(selected.get("target_object_label") or selected.get("label") or "target")
    obj_id = str(selected.get("target_object_id") or selected.get("id") or label or "target")
    return SceneObject(
        id=obj_id,
        label=label,
        bbox_min=(float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
        bbox_max=(float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
        centroid=(float(center[0]), float(center[1]), float(center[2])),
        source="stance_plan",
    )


def _scene_object_to_dict(obj) -> dict[str, Any]:
    return {
        "id": str(getattr(obj, "id", "")),
        "label": str(getattr(obj, "label", "")),
        "bbox_min_xyz": _rounded_xyz(getattr(obj, "bbox_min")),
        "bbox_max_xyz": _rounded_xyz(getattr(obj, "bbox_max")),
        "centroid_xyz": _rounded_xyz(getattr(obj, "centroid")),
        "source": str(getattr(obj, "source", "") or ""),
    }


def _scene_object_xy_size_area(obj) -> tuple[float, float, float]:
    try:
        bbox_min = getattr(obj, "bbox_min")
        bbox_max = getattr(obj, "bbox_max")
        sx = max(0.0, float(bbox_max[0]) - float(bbox_min[0]))
        sy = max(0.0, float(bbox_max[1]) - float(bbox_min[1]))
        return sx, sy, sx * sy
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, 0.0


def _is_structural_or_target_obstacle(obj, target_obj) -> bool:
    obj_id = str(getattr(obj, "id", "") or "")
    target_id = str(getattr(target_obj, "id", "") or "")
    if obj_id and target_id and obj_id == target_id:
        return True
    text = f"{obj_id} {getattr(obj, 'label', '')} {getattr(obj, 'source', '')}".lower()
    return any(token in text for token in ("wall", "window", "floor", "ground", "ceiling", "usd_shell"))


def _broad_aabb_false_positive_clip_ids(
    *,
    verdict,
    obstacles: Sequence[Any],
    target_obj,
    record: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    try:
        contact_count = int((record or {}).get("scene_collision_contact_count", 1))
    except Exception:  # noqa: BLE001
        contact_count = 1
    if contact_count != 0:
        return []
    clipped = list(getattr(verdict, "clipping", []) or [])
    if not clipped:
        return []
    by_id = {str(getattr(obj, "id", "") or ""): obj for obj in obstacles}
    footprint_area = (2.0 * float(ROBOT_FOOTPRINT_HALF_EXTENT[0])) * (
        2.0 * float(ROBOT_FOOTPRINT_HALF_EXTENT[1])
    )
    broad_area_threshold = max(2.0, footprint_area * 12.0)
    broad_span_threshold = max(
        1.0,
        max(float(ROBOT_FOOTPRINT_HALF_EXTENT[0]), float(ROBOT_FOOTPRINT_HALF_EXTENT[1])) * 4.0,
    )
    suppressed: list[dict[str, Any]] = []
    for obj_id, overlap_area in clipped:
        obj = by_id.get(str(obj_id))
        if obj is None or _is_structural_or_target_obstacle(obj, target_obj):
            continue
        sx, sy, area = _scene_object_xy_size_area(obj)
        if area < broad_area_threshold or min(sx, sy) < broad_span_threshold:
            continue
        suppressed.append(
            {
                "object_id": str(obj_id),
                "overlap_area_xy_m2": round(float(overlap_area), 6),
                "obstacle_xy_size_m": [round(float(sx), 6), round(float(sy), 6)],
                "obstacle_xy_area_m2": round(float(area), 6),
                "reason": (
                    "zero PhysX contacts and broad non-structural AABB; treated as a coarse "
                    "USD occupancy false positive"
                ),
            }
        )
    return suppressed


def _adjust_verdict_for_broad_aabb_false_positives(
    *,
    verdict,
    obstacles: Sequence[Any],
    target_obj,
    record: Mapping[str, Any] | None,
):
    suppressed = _broad_aabb_false_positive_clip_ids(
        verdict=verdict,
        obstacles=obstacles,
        target_obj=target_obj,
        record=record,
    )
    if not suppressed:
        return verdict, []
    suppressed_ids = {item["object_id"] for item in suppressed}
    remaining_clips = [
        (obj_id, area)
        for obj_id, area in (getattr(verdict, "clipping", []) or [])
        if str(obj_id) not in suppressed_ids
    ]
    failures = [
        failure
        for failure in (getattr(verdict, "failures", []) or [])
        if not str(failure).startswith("clips:")
    ]
    if remaining_clips:
        failures.append("clips:" + ",".join(str(obj_id) for obj_id, _ in remaining_clips))
    ok = not failures
    notes = (
        "placement valid; suppressed broad AABB clip false positives"
        if ok
        else "INVALID: " + "; ".join(str(failure) for failure in failures)
    )
    return replace(
        verdict,
        ok=ok,
        failures=failures,
        clipping=remaining_clips,
        notes=notes,
    ), suppressed


def _placement_verdict_to_dict(verdict) -> dict[str, Any]:
    return {
        "ok": bool(getattr(verdict, "ok", False)),
        "failures": list(getattr(verdict, "failures", []) or []),
        "clipping": [
            {"object_id": str(obj_id), "overlap_area_xy_m2": float(area)}
            for obj_id, area in (getattr(verdict, "clipping", []) or [])
        ],
        "near_clearance": [
            {"object_id": str(obj_id), "gap_m": float(gap)}
            for obj_id, gap in (getattr(verdict, "near_clearance", []) or [])
        ],
        "outside_boundary": list(getattr(verdict, "outside_boundary", []) or []),
        "min_obstacle_clearance_m": getattr(verdict, "min_obstacle_clearance_m", None),
        "facing_error_deg": getattr(verdict, "facing_error_deg", None),
        "standoff_m": getattr(verdict, "standoff_m", None),
        "on_floor": bool(getattr(verdict, "on_floor", False)),
        "notes": str(getattr(verdict, "notes", "") or ""),
    }


def _footprint_box_for_pose(pose, half_extent=ROBOT_FOOTPRINT_HALF_EXTENT) -> dict[str, list[float]]:
    hx, hy, hz = (abs(float(v)) for v in half_extent)
    return {
        "bbox_min_xyz": _rounded_xyz((float(pose[0]) - hx, float(pose[1]) - hy, float(pose[2]) - hz)),
        "bbox_max_xyz": _rounded_xyz((float(pose[0]) + hx, float(pose[1]) + hy, float(pose[2]) + hz)),
    }


def _find_standoff_fixtures(scene_objects: Sequence[Any], target_obj) -> list[Any]:
    target_text = (
        f"{getattr(target_obj, 'label', '')} {getattr(target_obj, 'id', '')}"
    ).lower()
    # If the resolver selected the actual fixture (sink/counter/cabinet/etc.), standoff must be
    # measured against that target's own AABB. Using a broad nearby cabinet run can otherwise make a
    # robot far from the sink look reachable because it is merely in front of generic drawers.
    if any(
        word in target_text
        for word in ("sink", "counter", "cabinet", "dishwasher", "island", "stove", "oven")
    ):
        return []
    target_xy = getattr(target_obj, "footprint_center")()
    out = []
    fixture_words = ("counter", "cabinet", "sink", "basin", "dishwasher", "island")
    for obj in scene_objects:
        label = str(getattr(obj, "label", "") or "").lower()
        obj_id = str(getattr(obj, "id", "") or "").lower()
        if not any(word in f"{label} {obj_id}" for word in fixture_words):
            continue
        try:
            cx, cy = getattr(obj, "footprint_center")()
        except Exception:  # noqa: BLE001
            continue
        if math.hypot(float(cx) - float(target_xy[0]), float(cy) - float(target_xy[1])) <= 1.75:
            out.append(obj)
    return out


def _build_placement_validation_manifest(
    *,
    stage,
    robot_prim_path: str,
    stance_plan: Mapping[str, Any] | None,
    accepted_pose,
    accepted_yaw: float,
    root_diagnostics: Mapping[str, Any] | None,
    scene_objects: Sequence[Any],
    scenario_id: str,
    visual_qc: Mapping[str, Any] | None = None,
    topdown_frame: str | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    accepted_pose_xyz = _rounded_xyz(accepted_pose)
    floor_z = (
        float((stance_plan or {}).get("floor_z_hint"))
        if stance_plan and (stance_plan or {}).get("floor_z_hint") is not None
        else float(accepted_pose[2]) - ROBOT_PELVIS_HEIGHT_M
    )
    target_obj = _target_object_from_stance_plan(stance_plan or {})
    if target_obj is None:
        blockers.append("placement_validation_missing_target_bounds")
    intended_geometry: dict[str, Any] | None = None
    if target_obj is not None:
        try:
            from blueprint_pipeline.scene_placement import validate_stand_pose  # type: ignore
        except Exception:
            from scene_placement import validate_stand_pose  # type: ignore
        selected_record = None
        try:
            candidates = list((stance_plan or {}).get("candidates") or [])
            idx = int((stance_plan or {}).get("selected_candidate_index") or 0)
            if 0 <= idx < len(candidates):
                selected_record = candidates[idx]
        except Exception:  # noqa: BLE001
            selected_record = None
        validation_standoff_range = TASK_STANCE_DEFAULT_VALIDATION_STANDOFF_RANGE_M
        try:
            recorded_range = (
                ((stance_plan or {}).get("placement_validation") or {}).get(
                    "validation_standoff_range_m"
                )
            )
            if (
                isinstance(recorded_range, Sequence)
                and not isinstance(recorded_range, (str, bytes))
                and len(recorded_range) == 2
            ):
                validation_standoff_range = (float(recorded_range[0]), float(recorded_range[1]))
        except Exception:  # noqa: BLE001
            validation_standoff_range = TASK_STANCE_DEFAULT_VALIDATION_STANDOFF_RANGE_M
        obstacles = list(scene_objects)
        # If the target was resolved from the stance plan but not present in the current catalog,
        # include it so clipping into the target itself is still detected.
        if not any(str(getattr(o, "id", "")) == str(getattr(target_obj, "id", "")) for o in obstacles):
            obstacles.append(target_obj)
        verdict = validate_stand_pose(
            tuple(float(v) for v in accepted_pose),
            float(accepted_yaw),
            target_obj,
            obstacles,
            floor_z=floor_z,
            footprint_half_extent=ROBOT_FOOTPRINT_HALF_EXTENT,
            pelvis_height=ROBOT_PELVIS_HEIGHT_M,
            max_facing_error_deg=30.0,
            standoff_range=validation_standoff_range,
            standoff_obstacles=_find_standoff_fixtures(obstacles, target_obj),
        )
        adjusted_verdict, suppressed_clips = _adjust_verdict_for_broad_aabb_false_positives(
            verdict=verdict,
            obstacles=obstacles,
            target_obj=target_obj,
            record=selected_record,
        )
        if suppressed_clips:
            intended_geometry = {
                **_placement_verdict_to_dict(adjusted_verdict),
                "raw_geometry": _placement_verdict_to_dict(verdict),
                "adjustments": {
                    "suppressed_broad_aabb_clips": suppressed_clips,
                    "claim_boundary": (
                        "Suppression is allowed only after PhysX reports zero contacts; it corrects "
                        "coarse USD AABB occupancy, not physical collision or task success."
                    ),
                },
            }
            verdict = adjusted_verdict
        else:
            intended_geometry = _placement_verdict_to_dict(verdict)
        if not verdict.ok:
            blockers.append("placement_geometry_invalid")
    actual_bbox = _world_bbox_for_prim(stage, robot_prim_path)
    upright_report = _robot_upright_report(stage, robot_prim_path)
    if upright_report.get("status") == "blocked":
        blockers.append("placement_robot_not_upright")
    ground_truth: dict[str, Any] = {
        "robot_prim_path": robot_prim_path,
        "accepted_pose_xyz": _rounded_xyz(accepted_pose),
        "accepted_pose_xy": [round(float(accepted_pose[0]), 6), round(float(accepted_pose[1]), 6)],
        "max_xy_error_m": PLACEMENT_GROUND_TRUTH_MAX_FOOTPRINT_CENTER_DELTA_M,
        "place_root_diagnostics": dict(root_diagnostics or {}),
        "upright_report": upright_report,
    }
    if actual_bbox is None:
        ground_truth["status"] = "blocked"
        ground_truth["blockers"] = ["placed_robot_bbox_unavailable"]
        ground_truth["xform_diagnostics"] = _root_transform_diagnostics(stage, robot_prim_path)
        blockers.append("placement_ground_truth_bbox_unavailable")
    else:
        actual_center = _footprint_center_xy_from_bbox(actual_bbox)
        actual_center_xyz = actual_bbox["center_xyz"]
        xy_error = _xy_distance(actual_center, ground_truth["accepted_pose_xy"])
        computed_offset_xyz = [
            round(float(actual_center_xyz[0]) - float(accepted_pose[0]), 6),
            round(float(actual_center_xyz[1]) - float(accepted_pose[1]), 6),
            round(float(actual_center_xyz[2]) - float(accepted_pose[2]), 6),
        ]
        ground_truth.update({
            "status": (
                "passed"
                if xy_error <= PLACEMENT_GROUND_TRUTH_MAX_FOOTPRINT_CENTER_DELTA_M
                else "blocked"
            ),
            "actual_world_aabb": actual_bbox,
            "actual_footprint_center_xyz": actual_center_xyz,
            "actual_footprint_center_xy": actual_center,
            "xy_error_m": round(float(xy_error), 6),
            "computed_xyz_offset_m": computed_offset_xyz,
        })
        if xy_error > PLACEMENT_GROUND_TRUTH_MAX_FOOTPRINT_CENTER_DELTA_M:
            ground_truth["blockers"] = ["placed_robot_footprint_center_mismatch"]
            ground_truth["xform_diagnostics"] = _root_transform_diagnostics(stage, robot_prim_path)
            blockers.append("placement_ground_truth_center_mismatch")
    if visual_qc is not None and visual_qc.get("status") != "passed":
        blockers.append("placement_visual_qc_failed")
    manifest = {
        "schema_version": "placement_validation.v1",
        "status": "PASS" if not blockers else "FAIL",
        "scenario_id": scenario_id,
        "blockers": sorted(set(blockers)),
        "accepted_pose": accepted_pose_xyz,
        "accepted_yaw": round(float(accepted_yaw), 6),
        "floor_z": round(float(floor_z), 6),
        "robot_footprint_half_extent": _rounded_xyz(ROBOT_FOOTPRINT_HALF_EXTENT),
        "robot_footprint_box_at_accepted_pose": _footprint_box_for_pose(accepted_pose),
        "target": _scene_object_to_dict(target_obj) if target_obj is not None else None,
        "scene_object_count": len(scene_objects),
        "scene_objects_sample": [_scene_object_to_dict(o) for o in list(scene_objects)[:40]],
        "intended_geometry": intended_geometry,
        "ground_truth_placement": ground_truth,
        "visual_qc": dict(visual_qc) if visual_qc is not None else None,
        "topdown_debug_frame": topdown_frame,
        "claim_boundary": (
            "Placement validation checks USD/world-frame geometry and rendered visual placement only. "
            "It is not manipulation success, learned policy success, physical robot readiness, safety "
            "validation, or deployment approval."
        ),
    }
    return manifest


def _placement_validation_passed_manifest(manifest: Mapping[str, Any] | None) -> bool:
    return bool(manifest and manifest.get("status") == "PASS" and not manifest.get("blockers"))


def _placement_visual_qc_target_label(stance_plan: Mapping[str, Any] | None) -> str:
    target = _target_object_from_stance_plan(stance_plan or {})
    if target is None:
        return "target"
    label = str(getattr(target, "label", "") or "").strip()
    return label or str(getattr(target, "id", "") or "target")


def _run_placement_visual_qc(
    frame_paths: Sequence[Path],
    *,
    target_label: str,
    task_description: str,
) -> dict[str, Any]:
    try:
        try:
            from render_visual_qc import qc_robot_placement_frames  # type: ignore
        except Exception:  # noqa: BLE001
            from blueprint_pipeline.render_visual_qc import qc_robot_placement_frames
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": "robot_placement_visual_qc.v1",
            "status": "blocked",
            "target": target_label,
            "task_description": task_description,
            "frames_reviewed": 0,
            "blockers": ["placement_visual_qc_import_failed"],
            "error": repr(exc),
            "per_frame": [],
        }
    return qc_robot_placement_frames(
        list(frame_paths),
        target_label,
        task_description=task_description,
        sample_n=min(4, len(frame_paths)),
    )


def _run_task_visual_qc(
    placement_frame_paths: Sequence[Path],
    pov_frame_paths: Sequence[Path],
    *,
    target_label: str,
    task_description: str,
) -> dict[str, Any]:
    try:
        try:
            from render_visual_qc import (  # type: ignore
                qc_manipulation_pov_frames,
                qc_robot_placement_frames,
            )
        except Exception:  # noqa: BLE001
            from blueprint_pipeline.render_visual_qc import (
                qc_manipulation_pov_frames,
                qc_robot_placement_frames,
            )
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": "robot_task_visual_qc.v1",
            "status": "blocked",
            "target": target_label,
            "task_description": task_description,
            "frames_reviewed": 0,
            "blockers": ["visual_qc_import_failed"],
            "error": repr(exc),
            "placement": None,
            "manipulation_pov": None,
        }
    placement_report = qc_robot_placement_frames(
        list(placement_frame_paths),
        target_label,
        task_description=task_description,
        sample_n=min(4, len(placement_frame_paths)),
    )
    pov_report = (
        qc_manipulation_pov_frames(
            list(pov_frame_paths),
            target_label,
            task_description=task_description,
            sample_n=min(4, len(pov_frame_paths)),
        )
        if pov_frame_paths
        else None
    )
    blockers: list[str] = []
    if placement_report.get("status") != "passed":
        blockers.extend(placement_report.get("blockers") or ["placement_visual_qc_failed"])
    if pov_report is not None and pov_report.get("status") != "passed":
        blockers.extend(pov_report.get("blockers") or ["manipulation_pov_visual_qc_failed"])
    frames_reviewed = int(placement_report.get("frames_reviewed") or 0)
    if pov_report is not None:
        frames_reviewed += int(pov_report.get("frames_reviewed") or 0)
    return {
        "schema_version": "robot_task_visual_qc.v1",
        "status": "passed" if not blockers else "blocked",
        "target": target_label,
        "task_description": task_description,
        "frames_reviewed": frames_reviewed,
        "blockers": sorted(set(str(b) for b in blockers)),
        "placement": placement_report,
        "manipulation_pov": pov_report,
    }


def _place_camera(stage, cam_path: str, eye, target) -> None:
    from pxr import UsdGeom, Gf  # type: ignore
    w, x, y, z = look_at_quat(eye, target)
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])))
    xform.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))


def _prim_world_translation(stage, prim) -> tuple[float, float, float] | None:
    try:
        from pxr import UsdGeom  # type: ignore
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        pos = matrix.ExtractTranslation()
        return (float(pos[0]), float(pos[1]), float(pos[2]))
    except Exception:  # noqa: BLE001
        return None


def _robot_authored_camera_mount(stage, robot_prim_path: str) -> dict[str, Any] | None:
    try:
        from pxr import Usd, UsdGeom  # type: ignore
        root = stage.GetPrimAtPath(robot_prim_path)
        if not root or not root.IsValid():
            return None
        for prim in Usd.PrimRange(root):
            try:
                is_camera = prim.IsA(UsdGeom.Camera)
            except Exception:  # noqa: BLE001
                is_camera = False
            if not is_camera:
                continue
            pos = _prim_world_translation(stage, prim)
            if pos is not None:
                return {
                    "source": "authored_robot_camera",
                    "prim_path": str(prim.GetPath()),
                    "eye_xyz": pos,
                }
    except Exception:  # noqa: BLE001
        return None
    return None


def _robot_link_mount(stage, robot_prim_path: str) -> dict[str, Any] | None:
    try:
        from pxr import Usd, UsdGeom  # type: ignore
        root = stage.GetPrimAtPath(robot_prim_path)
        if not root or not root.IsValid():
            return None
        candidates: list[tuple[int, Any]] = []
        preferences = (
            (0, ("camera",)),
            (1, ("head", "link")),
            (2, ("neck", "link")),
        )
        for prim in Usd.PrimRange(root):
            try:
                if not prim.IsA(UsdGeom.Xformable):
                    continue
            except Exception:  # noqa: BLE001
                continue
            name = prim.GetName().lower()
            for rank, tokens in preferences:
                if all(token in name for token in tokens):
                    candidates.append((rank, prim))
                    break
        for rank, prim in sorted(candidates, key=lambda item: item[0]):
            pos = _prim_world_translation(stage, prim)
            if pos is not None:
                return {
                    "source": "robot_link_mount",
                    "prim_path": str(prim.GetPath()),
                    "link_rank": rank,
                    "mount_role": (
                        "camera_link" if rank == 0 else "head_link" if rank == 1 else "neck_link"
                    ),
                    "eye_xyz": pos,
                }
    except Exception:  # noqa: BLE001
        return None
    return None


def _robot_arm_link_points(stage, robot_prim_path: str, *, arm: str = "right") -> dict[str, tuple[float, float, float]]:
    try:
        from pxr import Usd, UsdGeom  # type: ignore
        root = stage.GetPrimAtPath(robot_prim_path)
        if not root or not root.IsValid():
            return {}
        side = str(arm or "right").strip().lower()
        if side not in {"left", "right"}:
            side = "right"
        wanted = {
            "shoulder": ("shoulder",),
            "elbow": ("elbow",),
            "wrist": ("wrist",),
            "hand": ("hand", "palm"),
        }
        out: dict[str, tuple[float, float, float]] = {}
        for prim in Usd.PrimRange(root):
            try:
                if not prim.IsA(UsdGeom.Xformable):
                    continue
            except Exception:  # noqa: BLE001
                continue
            name = prim.GetName().lower()
            if side not in name or "link" not in name:
                continue
            for key, tokens in wanted.items():
                if key in out:
                    continue
                if any(token in name for token in tokens):
                    pos = _prim_world_translation(stage, prim)
                    if pos is not None:
                        out[key] = pos
                    break
        return out
    except Exception:  # noqa: BLE001
        return {}


def _robot_arm_link_points_by_arm(
    stage,
    robot_prim_path: str,
    *,
    arm: str = "right",
) -> dict[str, dict[str, tuple[float, float, float]]]:
    return {
        side: _robot_arm_link_points(stage, robot_prim_path, arm=side)
        for side in _required_manipulation_arms(arm)
    }


def _average_arm_link_points(
    arm_points_by_arm: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, tuple[float, float, float]]:
    averaged: dict[str, tuple[float, float, float]] = {}
    for role in ("shoulder", "elbow", "wrist", "hand"):
        pts = [
            tuple(float(v) for v in points[role])
            for points in arm_points_by_arm.values()
            if points.get(role) is not None
        ]
        if not pts:
            continue
        averaged[role] = (
            sum(pt[0] for pt in pts) / len(pts),
            sum(pt[1] for pt in pts) / len(pts),
            sum(pt[2] for pt in pts) / len(pts),
        )
    return averaged


def _robot_head_lens_eye_from_mount(
    mount_eye: Sequence[float],
    yaw: float,
    *,
    authored_camera: bool = False,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Return the render eye for the robot-mounted POV.

    Authored USD cameras are used exactly. For a bare head/neck link, use a small robot-relative
    forward/up lens offset so the camera sits at the face of the head instead of inside the link mesh.
    """
    raw_eye = (float(mount_eye[0]), float(mount_eye[1]), float(mount_eye[2]))
    if authored_camera:
        return raw_eye, {"lens_offset_xyz_robot_frame": [0.0, 0.0, 0.0]}
    fx, fy = math.cos(float(yaw)), math.sin(float(yaw))
    forward_m = max(0.05, float(ROBOT_FOOTPRINT_HALF_EXTENT[0]) * 0.28)
    up_m = max(0.015, float(ROBOT_FOOTPRINT_HALF_EXTENT[2]) * 0.03)
    eye = (
        raw_eye[0] + fx * forward_m,
        raw_eye[1] + fy * forward_m,
        raw_eye[2] + up_m,
    )
    return eye, {
        "lens_offset_xyz_robot_frame": [round(forward_m, 6), 0.0, round(up_m, 6)],
        "raw_mount_eye_xyz": [round(v, 6) for v in raw_eye],
    }


def _robot_mounted_manipulation_cam_pose(
    stage,
    robot_prim_path: str,
    root_pose,
    yaw,
    *,
    look_at=None,
    reach_arm: str = "right",
    vfov_deg: float | None = None,
    width: int | None = None,
    height: int | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float], dict[str, Any]]:
    fallback_eye, fallback_target = manipulation_cam_pose(
        root_pose,
        yaw,
        look_at=look_at,
        reach_arm=reach_arm,
    )
    if look_at is None:
        return fallback_eye, fallback_target, {"source": "root_yaw_fallback_no_look_at"}
    mount = _robot_authored_camera_mount(stage, robot_prim_path) or _robot_link_mount(stage, robot_prim_path)
    if mount is None:
        return fallback_eye, fallback_target, {"source": "root_yaw_fallback_no_robot_mount"}
    reach_selection = _normalize_reach_arm_selection(reach_arm)
    arm_points_by_arm = _robot_arm_link_points_by_arm(
        stage,
        robot_prim_path,
        arm=reach_selection,
    )
    if reach_selection == "both":
        arm_points = _average_arm_link_points(arm_points_by_arm)
    else:
        arm_points = dict(arm_points_by_arm.get(reach_selection) or {})
    target = _manipulation_camera_target_with_arm_context(look_at, arm_points)
    eye, lens_meta = _robot_head_lens_eye_from_mount(
        mount["eye_xyz"],
        yaw,
        authored_camera=mount.get("source") == "authored_robot_camera",
    )
    target_meta: dict[str, Any] = {}
    if vfov_deg is not None and width is not None and height is not None:
        target, target_meta = _select_manipulation_camera_target_for_visible_arm(
            look_at,
            arm_points,
            eye,
            target,
            vfov_deg=float(vfov_deg),
            width=int(width),
            height=int(height),
            arm=reach_selection,
            arm_points_by_arm=arm_points_by_arm,
        )
    arm_points_by_arm_json = {
        side: {
            key: [round(float(v), 6) for v in value]
            for key, value in sorted(points.items())
        }
        for side, points in sorted(arm_points_by_arm.items())
    }
    return eye, target, {
        "source": mount.get("source"),
        "mount_prim_path": mount.get("prim_path"),
        "mount_role": mount.get("mount_role"),
        "required_arms": list(_required_manipulation_arms(reach_selection)),
        "arm_link_points_used": sorted(arm_points),
        "arm_link_points_xyz": {
            key: [round(float(v), 6) for v in value]
            for key, value in sorted(arm_points.items())
        },
        "arm_link_points_by_arm_xyz": arm_points_by_arm_json,
        **target_meta,
        **lens_meta,
        "claim_boundary": (
            "POV camera is mounted from robot USD geometry when an authored camera/head/neck link "
            "is available; orientation is aimed at the task affordance plus visible arm context."
        ),
    }


def _safe_prim_segment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value or "debug"))
    cleaned = cleaned.strip("_") or "debug"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _set_flat_debug_box(stage, path: str, bbox_min, bbox_max, *, color, z_lift: float) -> None:
    from pxr import UsdGeom, Gf  # type: ignore

    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    center = (
        0.5 * (float(bbox_min[0]) + float(bbox_max[0])),
        0.5 * (float(bbox_min[1]) + float(bbox_max[1])),
        float(z_lift),
    )
    size = (
        max(0.03, float(bbox_max[0]) - float(bbox_min[0])),
        max(0.03, float(bbox_max[1]) - float(bbox_min[1])),
        0.08,
    )
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3d(size[0], size[1], size[2]))
    gprim = UsdGeom.Gprim(cube.GetPrim())
    gprim.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    try:
        gprim.CreateDisplayOpacityAttr([0.9])
    except Exception:  # noqa: BLE001
        pass


def _set_yaw_debug_arrow(stage, path: str, pose, yaw: float, *, color, z_lift: float) -> None:
    from pxr import UsdGeom, Gf  # type: ignore

    fx, fy = math.cos(float(yaw)), math.sin(float(yaw))
    length = 0.75
    width = 0.055
    center = (
        float(pose[0]) + fx * length * 0.5,
        float(pose[1]) + fy * length * 0.5,
        float(z_lift) + 0.02,
    )
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddRotateZOp().Set(math.degrees(float(yaw)))
    xform.AddScaleOp().Set(Gf.Vec3d(length, width, 0.1))
    gprim = UsdGeom.Gprim(cube.GetPrim())
    gprim.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])


def _debug_overlay_objects(scene_objects: Sequence[Any], target_obj, pose) -> list[Any]:
    if target_obj is None:
        return []
    target_xy = getattr(target_obj, "footprint_center")()
    wanted_words = ("target", "sink", "counter", "cabinet", "dishwasher", "basin", "faucet", "island")
    out = []
    for obj in scene_objects:
        label = str(getattr(obj, "label", "") or "").lower()
        obj_id = str(getattr(obj, "id", "") or "").lower()
        try:
            cx, cy = getattr(obj, "footprint_center")()
        except Exception:  # noqa: BLE001
            continue
        near_target = math.hypot(float(cx) - float(target_xy[0]), float(cy) - float(target_xy[1])) <= 2.2
        near_robot = math.hypot(float(cx) - float(pose[0]), float(cy) - float(pose[1])) <= 2.2
        named = any(word in f"{label} {obj_id}" for word in wanted_words)
        if named and (near_target or near_robot):
            out.append(obj)
    return out[:24]


def _update_topdown_debug_scene(
    stage,
    *,
    root_path: str,
    robot_pose,
    robot_yaw: float,
    stance_plan: Mapping[str, Any] | None,
    scene_objects: Sequence[Any],
    floor_z: float,
) -> dict[str, Any]:
    from pxr import UsdGeom  # type: ignore

    try:
        stage.RemovePrim(root_path)
    except Exception:  # noqa: BLE001
        pass
    UsdGeom.Scope.Define(stage, root_path)
    target_obj = _target_object_from_stance_plan(stance_plan or {})
    # Keep debug footprints above the noisy reconstruction and cabinet tops so the orthographic
    # render shows unambiguous xy occupancy instead of hiding the overlays under floor/wall geometry.
    z_lift = float(floor_z) + 2.75
    robot_box = _footprint_box_for_pose(robot_pose)
    _set_flat_debug_box(
        stage,
        f"{root_path}/robot_footprint",
        robot_box["bbox_min_xyz"],
        robot_box["bbox_max_xyz"],
        color=(0.05, 0.9, 0.15),
        z_lift=z_lift,
    )
    _set_yaw_debug_arrow(
        stage,
        f"{root_path}/robot_yaw_arrow",
        robot_pose,
        robot_yaw,
        color=(0.0, 1.0, 1.0),
        z_lift=z_lift,
    )
    overlay_objects = _debug_overlay_objects(scene_objects, target_obj, robot_pose)
    if target_obj is not None:
        _set_flat_debug_box(
            stage,
            f"{root_path}/target_footprint",
            getattr(target_obj, "bbox_min"),
            getattr(target_obj, "bbox_max"),
            color=(1.0, 0.05, 0.05),
            z_lift=z_lift + 0.015,
        )
    for idx, obj in enumerate(overlay_objects):
        _set_flat_debug_box(
            stage,
            f"{root_path}/object_{idx:02d}_{_safe_prim_segment(getattr(obj, 'id', 'object'))}",
            getattr(obj, "bbox_min"),
            getattr(obj, "bbox_max"),
            color=(1.0, 0.65, 0.05),
            z_lift=z_lift + 0.03,
        )
    xs = [float(robot_pose[0])]
    ys = [float(robot_pose[1])]
    if target_obj is not None:
        bmin = getattr(target_obj, "bbox_min")
        bmax = getattr(target_obj, "bbox_max")
        xs.extend([float(bmin[0]), float(bmax[0])])
        ys.extend([float(bmin[1]), float(bmax[1])])
    for obj in overlay_objects:
        bmin = getattr(obj, "bbox_min")
        bmax = getattr(obj, "bbox_max")
        xs.extend([float(bmin[0]), float(bmax[0])])
        ys.extend([float(bmin[1]), float(bmax[1])])
    center_xy = (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)))
    radius = max(max(xs) - min(xs), max(ys) - min(ys), 2.0) * 0.65 + 0.6
    return {
        "root_path": root_path,
        "robot_footprint_box": robot_box,
        "overlay_object_count": len(overlay_objects),
        "center_xy": [round(center_xy[0], 6), round(center_xy[1], 6)],
        "radius_m": round(float(radius), 6),
        "overlay_z_lift": round(float(z_lift), 6),
    }


def _place_topdown_debug_camera(stage, cam_path: str, *, center_xy, radius_m: float,
                                width: int, height: int, floor_z: float) -> None:
    from pxr import UsdGeom, Gf  # type: ignore

    cam = UsdGeom.Camera(stage.GetPrimAtPath(cam_path))
    try:
        cam.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)
        view_width = max(1.0, float(radius_m) * 2.0)
        cam.GetHorizontalApertureAttr().Set(view_width)
        cam.GetVerticalApertureAttr().Set(view_width * (float(height) / float(width)))
        cam.GetClippingRangeAttr().Set((0.01, 2000.0))
    except Exception:  # noqa: BLE001
        pass
    eye_z = float(floor_z) + max(4.0, float(radius_m) * 2.5)
    # A USD camera with identity orientation already looks down local -Z. For an overhead camera,
    # translating it above the scene is more stable than using the generic look-at path's singular
    # straight-down orientation case.
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), eye_z))


def _write_topdown_debug_layout_image(
    path: Path,
    *,
    robot_pose,
    robot_yaw: float,
    stance_plan: Mapping[str, Any] | None,
    scene_objects: Sequence[Any],
    width: int,
    height: int,
) -> dict[str, Any] | None:
    """Write a deterministic xy layout beside the RTX top-down frame.

    The RTX overhead render can be visually noisy in the reconstructed kitchen. This pure 2D artifact
    uses the exact same world-frame AABBs as placement validation, so the robot footprint, target, and
    counter/cabinet footprints are readable without camera foreshortening.
    """
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    target_obj = _target_object_from_stance_plan(stance_plan or {})
    robot_box = _footprint_box_for_pose(robot_pose)
    overlay_objects = _debug_overlay_objects(scene_objects, target_obj, robot_pose)
    boxes: list[tuple[str, Any, tuple[int, int, int]]] = [
        ("robot", robot_box, (0, 170, 60)),
    ]
    if target_obj is not None:
        boxes.append(("target", {
            "bbox_min_xyz": getattr(target_obj, "bbox_min"),
            "bbox_max_xyz": getattr(target_obj, "bbox_max"),
        }, (220, 30, 30)))
    for obj in overlay_objects:
        boxes.append(("object", {
            "bbox_min_xyz": getattr(obj, "bbox_min"),
            "bbox_max_xyz": getattr(obj, "bbox_max"),
        }, (230, 150, 20)))
    xs: list[float] = []
    ys: list[float] = []
    for _kind, box, _color in boxes:
        bmin = box["bbox_min_xyz"]
        bmax = box["bbox_max_xyz"]
        xs.extend([float(bmin[0]), float(bmax[0])])
        ys.extend([float(bmin[1]), float(bmax[1])])
    if not xs or not ys:
        return None
    pad = 0.45
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad
    span_x = max(0.1, max_x - min_x)
    span_y = max(0.1, max_y - min_y)
    scale = min((width - 80) / span_x, (height - 80) / span_y)
    offset_x = 40 + 0.5 * ((width - 80) - span_x * scale)
    offset_y = 40 + 0.5 * ((height - 80) - span_y * scale)

    def px(x: float, y: float) -> tuple[float, float]:
        return (
            offset_x + (float(x) - min_x) * scale,
            height - (offset_y + (float(y) - min_y) * scale),
        )

    img = Image.new("RGB", (int(width), int(height)), (248, 248, 244))
    draw = ImageDraw.Draw(img, "RGBA")
    for kind, box, color in boxes:
        bmin = box["bbox_min_xyz"]
        bmax = box["bbox_max_xyz"]
        p0 = px(float(bmin[0]), float(bmin[1]))
        p1 = px(float(bmax[0]), float(bmax[1]))
        rect = [min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])]
        alpha = 160 if kind == "robot" else 110
        draw.rectangle(rect, fill=(*color, alpha), outline=(*color, 255), width=3 if kind == "robot" else 2)
    cx, cy = px(float(robot_pose[0]), float(robot_pose[1]))
    fx, fy = math.cos(float(robot_yaw)), math.sin(float(robot_yaw))
    tip = px(float(robot_pose[0]) + fx * 0.7, float(robot_pose[1]) + fy * 0.7)
    draw.line([(cx, cy), tip], fill=(0, 140, 190, 255), width=5)
    draw.ellipse([tip[0] - 7, tip[1] - 7, tip[0] + 7, tip[1] + 7], fill=(0, 140, 190, 255))
    draw.text((18, 14), "top-down placement layout: green=robot footprint, cyan=facing, red=target, orange=fixtures",
              fill=(30, 30, 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return {
        "path": str(path),
        "bounds_xy": [round(min_x, 6), round(min_y, 6), round(max_x, 6), round(max_y, 6)],
        "overlay_object_count": len(overlay_objects),
    }


def _force_cheap_collision(stage, approximation: str = "boundingCube") -> int:
    """Override every mesh collision approximation. The 47-object kitchen's default SDF cooking on
    non-watertight meshes takes >4 min and blocks the RTX render; ``boundingCube`` cooks ~instantly
    but is coarse (collision volumes far bigger than the visual shape, which shoves the robot off a
    head-on approach). ``convexHull`` is shape-accurate enough for the robot to stand centered + close
    and still cooks far faster than SDF (a watertight convex per mesh). Visual geometry is untouched.
    Returns the number of meshes overridden."""
    from pxr import UsdPhysics  # type: ignore
    tokens = {
        "boundingCube": UsdPhysics.Tokens.boundingCube,
        "convexHull": UsdPhysics.Tokens.convexHull,
        "convexDecomposition": UsdPhysics.Tokens.convexDecomposition,
    }
    approx = tokens.get(approximation, UsdPhysics.Tokens.boundingCube)
    n = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr().Set(approx)
            n += 1
    return n


def _prune_to_focus(stage, route_points, focus_radius: float, keep_substrings) -> dict:
    """Task-aware scene subset: deactivate kitchen object prims whose placement is farther than
    ``focus_radius`` (m, xy) from the robot's route, keeping the task region + structural shell
    (walls/floor/lights). Deactivated prims are excluded from BOTH PhysX cooking and the render,
    so this is the lever for 'the scene is too large'. Returns {kept, pruned, kept_names}."""
    from pxr import UsdGeom  # type: ignore
    xc = UsdGeom.XformCache()
    keep_subs = [s.strip().lower() for s in keep_substrings if s and s.strip()]
    pts = [(float(p[0]), float(p[1])) for p in route_points] or [(0.0, 0.0)]
    root = stage.GetPrimAtPath("/root")
    if not (root and root.IsValid()):
        root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    kept, pruned, kept_names = 0, 0, []
    for child in root.GetChildren():
        name = child.GetName()
        low = name.lower()
        if any(s in low for s in keep_subs):
            kept += 1
            kept_names.append(name)
            continue
        try:
            t = xc.GetLocalToWorldTransform(child).ExtractTranslation()
            pos = (float(t[0]), float(t[1]))
            dmin = min(math.hypot(pos[0] - x, pos[1] - y) for x, y in pts)
        except Exception:  # noqa: BLE001
            kept += 1
            kept_names.append(name)
            continue
        if dmin <= focus_radius:
            kept += 1
            kept_names.append(name)
        else:
            child.SetActive(False)
            pruned += 1
    return {"kept": kept, "pruned": pruned, "kept_names": kept_names[:40]}


def _make_render_product(camera_path: str, width: int, height: int):
    import omni.replicator.core as rep  # type: ignore
    rp = rep.create.render_product(camera_path, (width, height))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([rp])
    return annot


def _software_denoise_image(img):
    """Best-effort CPU denoise for review PNGs when RTX/NGX denoising is unavailable on a pod."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
        arr = np.asarray(img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        den = cv2.fastNlMeansDenoisingColored(bgr, None, 6, 6, 7, 21)
        return Image.fromarray(cv2.cvtColor(den, cv2.COLOR_BGR2RGB))
    except Exception:  # noqa: BLE001
        pass
    try:
        from PIL import ImageFilter  # type: ignore
        return img.filter(ImageFilter.MedianFilter(size=3))
    except Exception:  # noqa: BLE001
        return img


def _save_rgb(annot, out_path: Path, *, software_denoise: bool = False) -> bool:
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore
    data = annot.get_data()
    if data is None or getattr(data, "size", 0) == 0:
        return False
    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    img = Image.fromarray(arr.astype("uint8"))
    if software_denoise:
        img = _software_denoise_image(img)
    img.save(out_path)
    return True


def camera_aperture_for_fov(vfov_deg: float, width: int, height: int, focal_mm: float = 20.0):
    """Focal length + (horizontal, vertical) aperture that give a camera a vertical FOV of
    ``vfov_deg`` at the render aspect ratio. USD's default 50mm/20.955mm camera is a ~24deg
    telephoto — far too zoomed for the manipulation POV (it fills the frame with the dark sink
    basin) and it does NOT match the FOV the skeleton projection assumes, so the projected
    landmarks misalign with the render. Pure trig (no USD) so it is unit-testable."""
    vap = 2.0 * float(focal_mm) * math.tan(math.radians(float(vfov_deg)) / 2.0)
    hap = vap * (float(width) / float(height))
    return float(focal_mm), hap, vap


def _set_camera_fov(stage, cam_path: str, vfov_deg: float, width: int, height: int) -> None:
    """Set a USD camera's focal length + apertures so its vertical FOV == ``vfov_deg`` (matching the
    skeleton-projection FOV) instead of the narrow ~17deg default. GPU/USD only."""
    from pxr import UsdGeom  # type: ignore
    focal, hap, vap = camera_aperture_for_fov(vfov_deg, width, height)
    cam = UsdGeom.Camera(stage.GetPrimAtPath(cam_path))
    cam.GetFocalLengthAttr().Set(focal)
    cam.GetHorizontalApertureAttr().Set(hap)
    cam.GetVerticalApertureAttr().Set(vap)
    try:
        # A small near plane keeps the head-mounted POV from embedding in its own head mesh, but a
        # 0.01-0.02 near with a 1000-2000 far wrecks the depth-buffer precision the RTX denoiser relies
        # on -> heavy salt-and-pepper grain. A 0.05/50 range (1000:1) avoids embedding and remains
        # broad enough for room-scale manipulation scenes without wasting depth precision.
        cam.GetClippingRangeAttr().Set((0.05, 50.0))
    except Exception:  # noqa: BLE001
        pass


def _add_workspace_fill_light(stage, target, *, intensity: float, height: float = 2.0,
                              path: str = "/World/WorkspaceFill") -> None:
    """Add a local sphere fill light above the manipulation workspace (the faucet) so the dark sink
    basin + the reaching arm are lit. Intensity is configurable (blind-tunable via re-render). The
    default scene has a single distant key light that leaves the basin interior in shadow. GPU/USD."""
    from pxr import UsdLux, UsdGeom, Gf  # type: ignore
    light = UsdLux.SphereLight.Define(stage, path)
    light.CreateIntensityAttr(float(intensity))
    light.CreateRadiusAttr(0.4)
    # Idempotent on a reused (warm --serve) stage: reuse the existing translate op instead of adding a
    # second one (AddTranslateOp raises "xformOp:translate already exists" on the 2nd+ job otherwise).
    xf = UsdGeom.Xformable(light.GetPrim())
    translate_op = next(
        (op for op in xf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )
    if translate_op is None:
        translate_op = xf.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(target[0]), float(target[1]) - 0.25, float(height)))


def _add_pov_headlamp(stage, eye, look_at, *, intensity: float = 20000.0,
                      path: str = "/World/PovHeadlamp") -> None:
    """Camera-side fill light for the manipulation POV so the REACHING ARM + gripper are front-lit.

    The workspace fill light sits at the door/affordance, BEYOND the arm from the head-mounted camera,
    so the camera otherwise sees only the arm's shadow side (it renders black). This places a SOFT fill
    in front of the camera eye, aimed toward the workspace, lighting the arms+grippers the camera sees.

    It MUST be soft: a small, very bright sphere this close to the arm is a path-tracing firefly source
    (salt-and-pepper grain that the denoiser can't recover). So use a LARGE radius + a CAPPED intensity
    + a little more standoff -> even fill, no fireflies. Idempotent on a reused (warm --serve) stage."""
    from pxr import UsdLux, UsdGeom, Gf  # type: ignore
    fx, fy, fz = (float(look_at[0]) - float(eye[0]),
                  float(look_at[1]) - float(eye[1]),
                  float(look_at[2]) - float(eye[2]))
    length = math.sqrt(fx * fx + fy * fy + fz * fz) or 1e-6
    pos = (float(eye[0]) + fx / length * 0.30,
           float(eye[1]) + fy / length * 0.30,
           float(eye[2]) + fz / length * 0.30 + 0.10)  # 30cm toward the workspace, a touch above
    light = UsdLux.SphereLight.Define(stage, path)
    # Cap intensity (callers pass the bright 30000 workspace value) and use a large soft radius so the
    # close camera-side fill does not create fireflies on the nearby arm.
    light.CreateIntensityAttr().Set(min(float(intensity), 6000.0))
    light.CreateRadiusAttr().Set(0.5)
    xf = UsdGeom.Xformable(light.GetPrim())
    translate_op = next(
        (op for op in xf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )
    if translate_op is None:
        translate_op = xf.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*pos))


def _neutralize_environment(stage, *, intensity: float = 1500.0) -> int:
    """Replace any outdoor-HDRI DomeLight in the loaded scene with a NEUTRAL uniform environment.

    The Lightwheel kitchen ships a DomeLight (e.g. ``DomeLight_01`` with ``texture/chive 7000x3500.hdr``)
    that projects an outdoor cityscape, visible through the kitchen windows — an incongruous background
    for an enclosed manipulation scene. Clearing the HDRI texture + setting a neutral bright color turns
    it into even ambient: windows read neutral (no city) AND the dark cabinet/basin surfaces get lifted
    by global fill. Returns the number of dome lights neutralized. GPU/USD only."""
    from pxr import UsdLux, Sdf, Gf  # type: ignore
    n = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "DomeLight":
            continue
        dome = UsdLux.DomeLight(prim)
        for attr_name in ("inputs:texture:file", "texture:file"):
            attr = prim.GetAttribute(attr_name)
            if attr and attr.IsValid():
                try:
                    attr.Set(Sdf.AssetPath(""))  # drop the HDRI -> uniform dome (no cityscape)
                except Exception:  # noqa: BLE001
                    pass
        try:
            dome.CreateColorAttr().Set(Gf.Vec3f(0.92, 0.92, 0.95))  # neutral cool-white
            dome.CreateIntensityAttr(float(intensity))
        except Exception:  # noqa: BLE001
            pass
        n += 1
    return n


def run_scenarios(*, kitchen_usd: str, g1_usd: str, scenarios: Sequence[dict], out_dir: Path,
                  policy_id: str, steps: int, width: int, height: int, fps: int,
                  warmup_frames: int, capture_every: int, no_collision_probe: bool = False,
                  per_scenario_seconds: int = 480, focus_radius: float = 0.0,
                  keep_substrings: Sequence[str] = ("room", "floor", "wall", "ground", "ceiling", "light"),
                  disable_physx: bool = False, settle_seconds: int = 0,
                  cheap_collision: bool = False, articulated: bool = False,
                  camera_vfov_deg: float = 50.0, manipulation_cam: bool = False,
                  manipulation_look_at=None, render_subframes: int = 1,
                  manipulation_reach: bool = False, manipulation_reach_arm: str = "both",
                  fill_light_intensity: float = 0.0,
                  physics_articulation_drive: bool = False,
                  dynamic_standing_contact_steps: int = 0,
                  neutral_environment: bool = False,
                  kinematic_arm_pose: bool = False,
                  collision_approximation: str = "boundingCube",
                  verify_cam: bool = False,
                  manipulation_stand: bool = False,
                  software_denoise: bool = True,
                  serve: bool = False, serve_dir: "Path | None" = None,
                  serve_idle_timeout_s: float = 600.0,
                  serve_max_jobs: "int | None" = None) -> dict:
    """GPU orchestration: boot Isaac, load scene + G1, run the controller per scenario with RTX
    render + (optional) PhysX collision probe, emit traces + MP4s + outcomes. Instrumented with
    flushed progress + a per-scenario wall-clock cap so it cannot hang silently.

    Warm mode (``serve=True``): after the one-time setup (Isaac boot, scene + G1 load, cameras,
    settle), keep the process alive and render a STREAM of task scenarios pulled from ``serve_dir``
    via :func:`blueprint_pipeline.warm_render_server.serve_render_loop`, so each rerun skips image
    pull + Isaac boot + stage load + most settle. Single-shot behavior (``serve=False``) is unchanged."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _log("booting Isaac (headless RTX) ...")
    sim = _boot_sim(headless=True)
    _log("Isaac booted; enabling Replicator")
    rep = _enable_and_import_replicator()  # after boot: omni.* now importable + extension enabled
    _log("Replicator ready")
    if disable_physx:
        # NOTE: confirmed on GPU to break the RTX renderer (hangs at render-product creation) —
        # kept only for experiments. Keep PhysX on and use settle_seconds instead.
        _disable_physics_cooking()
        _log("PhysX cooking disabled (WARNING: breaks the renderer on this image)")
    blockers: list[str] = []
    outcomes: list[dict] = []
    physics_contact_reports: list[dict[str, Any]] = []
    result = None
    if dynamic_standing_contact_steps > 0:
        articulated = True
        physics_articulation_drive = True
        if int(steps) != 1:
            blockers.append("physics_articulation_dynamic_standing_contact_requires_single_step")
        if len(scenarios) != 1:
            blockers.append("physics_articulation_dynamic_standing_contact_requires_single_scenario")
    try:
        _log(f"opening kitchen USD: {kitchen_usd}")
        stage = _open_stage(_resolve_asset_uri(kitchen_usd))
        if cheap_collision:
            nc = _force_cheap_collision(stage, approximation=collision_approximation)
            _log(f"forced {collision_approximation} collision on {nc} mesh-collision prims")
        _log("kitchen stage open; binding G1 articulation")
        binding = _bind_g1(stage, _resolve_asset_uri(g1_usd))
        _log(f"G1 binding: articulation={binding['controllable_articulation_detected']} "
             f"collision={binding['collision_enabled_verified']}")
        (out_dir / "g1_binding.json").write_text(json.dumps(binding, indent=2))
        if not binding["controllable_articulation_detected"]:
            blockers.append("official_isaac_unitree_g1_articulation_api_unverified")
        robot_neutral_xforms: dict[str, Any] = {}
        if kinematic_arm_pose or manipulation_reach:
            robot_neutral_xforms = _capture_robot_neutral_descendant_xforms(
                stage,
                binding["prim_path"],
            )
            _log(f"G1 neutral descendant xforms captured: {len(robot_neutral_xforms)} prim(s)")
        has_deferred_task_targets = any(sc.get("task_target_deferred") for sc in scenarios)
        if focus_radius > 0 and has_deferred_task_targets:
            _log(
                "focus prune skipped: one or more task-only scenarios need scene-placement "
                "target resolution before route points exist"
            )
            (out_dir / "focus_prune.json").write_text(json.dumps({
                "status": "skipped",
                "reason": "deferred_task_target_requires_scene_resolution",
            }, indent=2))
        elif focus_radius > 0:
            route_pts = [p for sc in scenarios for p in sc.get("route_points", [])]
            pr = _prune_to_focus(stage, route_pts, focus_radius, keep_substrings)
            _log(f"focus prune (r={focus_radius}m): kept {pr['kept']} objects, deactivated {pr['pruned']}")
            (out_dir / "focus_prune.json").write_text(json.dumps(pr, indent=2))
        placement_scene_objects = _scene_objects_for_stage(stage)
        (out_dir / "placement_scene_objects.json").write_text(
            json.dumps([_scene_object_to_dict(obj) for obj in placement_scene_objects], indent=2)
        )
        _log(f"placement scene object catalog: {len(placement_scene_objects)} object(s)")
        rest_offsets = None
        art_ctx = None
        dynamic_seed_decisions: dict[str, Any] = {}
        preplanned_task_stance_plans: dict[str, dict[str, Any]] = {}
        if articulated:
            rest_offsets = _g1_link_rest_offsets(stage, binding["prim_path"])
            if dynamic_standing_contact_steps > 0 and scenarios:
                if no_collision_probe:
                    def seed_probe(pose, yaw):  # noqa: ANN001
                        return 0
                else:
                    seed_probe = _overlap_probe(binding["prim_path"])
                if manipulation_stand:
                    seed_sid = scenarios[0]["scenario_id"]
                    seed_stance_plan = _plan_task_stance_for_stage(
                        stage=stage,
                        scenario=scenarios[0],
                        manipulation_look_at=manipulation_look_at,
                        probe=seed_probe,
                        no_collision_probe=no_collision_probe,
                        robot_prim_path=binding["prim_path"],
                    )
                    preplanned_task_stance_plans[seed_sid] = seed_stance_plan
                    (out_dir / "dynamic_task_stance_seed_plan.json").write_text(
                        json.dumps(seed_stance_plan, indent=2)
                    )
                    if seed_stance_plan.get("status") == "accepted":
                        seed_root = tuple(float(v) for v in seed_stance_plan["accepted_pose"])
                        seed_yaw = float(seed_stance_plan["accepted_yaw"])
                        dynamic_seed_decisions[seed_sid] = {
                            "source": "task_stance_plan",
                            "root_pose": seed_root,
                            "yaw": seed_yaw,
                        }
                        _place_root(stage, binding["prim_path"], seed_root, seed_yaw)
                        _log(
                            "dynamic standing/contact root seeded from task stance plan before "
                            f"articulation tensor view: pose={seed_root}, yaw={seed_yaw:.4f}"
                        )
                    else:
                        blockers.append("task_stance_plan_failed")
                        _log(
                            "dynamic standing/contact task stance seed blocked "
                            f"{seed_stance_plan.get('blockers') or seed_stance_plan.get('status')}"
                        )
                else:
                    seed_policy = policy_mod.make_policy(policy_id)
                    seed_policy.reset(scenarios[0])
                    seed_decision = seed_policy.step(
                        policy_mod.StepContext(
                            step=0,
                            num_steps=steps,
                            probe_collision=seed_probe,
                        )
                    )
                    dynamic_seed_decisions[scenarios[0]["scenario_id"]] = seed_decision
                    _place_root(stage, binding["prim_path"], seed_decision.root_pose, seed_decision.yaw)
                    _log(
                        "dynamic standing/contact root seeded before articulation tensor view: "
                        f"pose={seed_decision.root_pose}, yaw={seed_decision.yaw:.4f}"
                    )
            # The physics articulation drive is OPT-IN and default-OFF. Dynamic standing/contact
            # proof intentionally uses a plain SimulationContext with no SingleArticulation tensor
            # view because the official G1 USD invalidates that tensor view during joint drive/read.
            if physics_articulation_drive:
                try:
                    gravity_z = -9.81 if dynamic_standing_contact_steps > 0 else 0.0
                    if dynamic_standing_contact_steps > 0:
                        art_ctx = _setup_physics_context_only(gravity_z=gravity_z)
                        _log(
                            "G1 USD PhysX articulation settle ready: "
                            f"tensor_view_used={art_ctx['tensor_view_used']}, gravity_z={gravity_z}"
                        )
                    else:
                        art_ctx = _setup_articulated_g1(binding["prim_path"], gravity_z=gravity_z)
                        _log(
                            "G1 articulation drive ready: "
                            f"{art_ctx['dof_count']} joints, {len(art_ctx['link_names'])} links, "
                            f"gravity_z={gravity_z}"
                        )
                except Exception as exc:  # noqa: BLE001
                    blockers.append("official_isaac_unitree_g1_joint_drive_unavailable")
                    _log(f"G1 articulation drive unavailable ({exc!r}); using USD skeleton fallback")
                if art_ctx is not None and not art_ctx.get("link_names"):
                    _log("G1 articulation body/link names unavailable; using USD skeleton fallback for landmarks")
            _log(f"G1 skeleton (USD rest offsets): {len(rest_offsets)} link landmarks")
        from pxr import UsdGeom, UsdLux  # type: ignore
        UsdGeom.Scope.Define(stage, "/World")
        over_cam = "/World/Cameras/overview"
        pov_cam = "/World/Cameras/robot_pov"
        verify_cam_path = "/World/Cameras/verify"
        topdown_cam_path = "/World/Cameras/placement_topdown"
        UsdGeom.Camera.Define(stage, over_cam)
        UsdGeom.Camera.Define(stage, pov_cam)
        if verify_cam:
            UsdGeom.Camera.Define(stage, verify_cam_path)
        if manipulation_stand:
            UsdGeom.Camera.Define(stage, topdown_cam_path)
        key = UsdLux.DistantLight.Define(stage, "/World/Key")
        try:
            key.CreateIntensityAttr(3000.0)  # lift the global key so the workspace is not crushed dark
        except Exception:  # noqa: BLE001
            pass
        # POV camera: widen from USD's ~17deg telephoto default to the projection FOV so the frame
        # shows the lit workspace (not a zoomed crop of the dark basin) AND the rendered view matches
        # the skeleton projection. Overview gets a wide FOV so it frames the whole scene.
        pov_vfov_deg = max(float(camera_vfov_deg), 68.0) if manipulation_cam else float(camera_vfov_deg)
        _set_camera_fov(stage, pov_cam, pov_vfov_deg, width, height)
        _set_camera_fov(stage, over_cam, 60.0, width, height)
        if verify_cam:
            _set_camera_fov(stage, verify_cam_path, 55.0, width, height)
        if manipulation_cam and fill_light_intensity > 0 and manipulation_look_at is not None:
            _add_workspace_fill_light(stage, manipulation_look_at, intensity=fill_light_intensity)
            _log(f"workspace fill light @ {tuple(round(float(c),2) for c in manipulation_look_at)} "
                 f"intensity={fill_light_intensity}")
        if neutral_environment:
            try:
                n_dome = _neutralize_environment(stage)
                _log(f"neutralized {n_dome} outdoor-HDRI dome light(s) -> enclosed neutral environment")
            except Exception as exc:  # noqa: BLE001
                _log(f"environment neutralize skipped ({exc!r})")
        _log(f"creating render products ({width}x{height})")
        over_annot = _make_render_product(over_cam, width, height)
        pov_annot = _make_render_product(pov_cam, width, height)
        verify_annot = _make_render_product(verify_cam_path, width, height) if verify_cam else None
        topdown_annot = (
            _make_render_product(topdown_cam_path, width, height) if manipulation_stand else None
        )
        if software_denoise:
            _log("software PNG denoise enabled for saved render frames")
        center, radius = scene_framing(scenarios)
        _place_camera(stage, over_cam,
                      (center[0] + radius * 1.4, center[1] - radius * 1.4, center[2] + radius * 1.1),
                      center)
        _log("render products + overview camera ready")
        if settle_seconds > 0:
            # Let PhysX finish async collision-cooking BEFORE we render — rendering *during*
            # cooking is what hangs frame 2+. A pure wait lets the background cook threads drain.
            _log(f"settling {settle_seconds}s for PhysX cooking to drain before rendering")
            t_settle = time.time()
            while time.time() - t_settle < settle_seconds:
                time.sleep(15)
                _log(f"  settle {int(time.time() - t_settle)}/{settle_seconds}s")
            _log("settle complete; starting render")
        if no_collision_probe:
            _log("collision probe DISABLED (policy goes direct every step)")
            def probe(pose, yaw):  # noqa: ANN001
                return 0
        else:
            probe = _overlap_probe(binding["prim_path"])
        def _render_scenario(sc):
            sid = sc["scenario_id"]
            sdir = out_dir / sid
            (sdir / "frames").mkdir(parents=True, exist_ok=True)
            stand_root = stand_yaw = None
            stance_plan = None
            scene_objects_for_validation: list[Any] = []
            placement_validation_manifest: dict[str, Any] | None = None
            placement_validation_path = sdir / "placement_validation.json"
            placement_validation_blocker_recorded = False
            placement_topdown_frame_path: Path | None = None
            placement_topdown_layout_frame_path: Path | None = None
            pov_reach_arm = _normalize_reach_arm_selection(manipulation_reach_arm)
            rendered_reach_arm = pov_reach_arm
            pov_geometry_path = sdir / "manipulation_pov_geometry.json"
            pov_geometry_records: list[dict[str, Any]] = []
            pov_geometry_report: dict[str, Any] | None = None
            pov_geometry_blocker_recorded = False
            last_root_diagnostics: dict[str, Any] | None = None
            stance_plan_path = sdir / "task_stance_plan.json"
            def _write_pov_geometry_report() -> dict[str, Any] | None:
                if not pov_geometry_records:
                    return None
                geom_blockers = sorted({
                    str(blocker)
                    for record in pov_geometry_records
                    for blocker in (record.get("blockers") or [])
                })
                report = {
                    "schema_version": "manipulation_pov_geometry_index.v1",
                    "status": "PASS" if not geom_blockers else "FAIL",
                    "blockers": geom_blockers,
                    "scenario_id": sid,
                    "frames_checked": len(pov_geometry_records),
                    "frames": pov_geometry_records,
                    "claim_boundary": (
                        "Frame geometry proves task-affordance and USD arm-link visibility in the "
                        "render camera. It is not manipulation success or physical robot validation."
                    ),
                }
                pov_geometry_path.write_text(json.dumps(report, indent=2))
                return report
            if manipulation_stand:
                stance_plan = preplanned_task_stance_plans.get(sid)
                if stance_plan is None:
                    stance_plan = _plan_task_stance_for_stage(
                        stage=stage,
                        scenario=sc,
                        manipulation_look_at=manipulation_look_at,
                        probe=probe,
                        no_collision_probe=no_collision_probe,
                        robot_prim_path=binding["prim_path"],
                    )
                stance_plan_path.write_text(json.dumps(stance_plan, indent=2))
                if stance_plan.get("status") == "accepted":
                    stand_root = tuple(float(v) for v in stance_plan["accepted_pose"])
                    stand_yaw = float(stance_plan["accepted_yaw"])
                    validation_target_obj = _target_object_from_stance_plan(stance_plan)
                    if validation_target_obj is not None:
                        t_obs = time.time()
                        scene_objects_for_validation = _placement_obstacles_for_stage(
                            stage,
                            focus_bounds=(validation_target_obj.bbox_min, validation_target_obj.bbox_max),
                        )
                        _log(
                            f"scenario {sid}: placement validation obstacles "
                            f"{len(scene_objects_for_validation)} object(s) in "
                            f"{time.time() - t_obs:.1f}s"
                        )
                    _log(
                        f"scenario {sid}: task stance accepted -> "
                        f"{tuple(round(float(c), 2) for c in stand_root)} yaw={stand_yaw:.2f} "
                        f"after {len(stance_plan.get('candidates', []))} candidate probe(s)"
                    )
                else:
                    blockers.append("task_stance_plan_failed")
                    _log(
                        f"scenario {sid}: task stance blocked "
                        f"{stance_plan.get('blockers') or stance_plan.get('status')}"
                    )
            # When no explicit look-at is given, the manipulation camera + arm reach face the SAME
            # target the stance was planned around (resolved from the scene+task via scene_placement),
            # so a fully dynamic render needs NO hardcoded coordinates at all. Falls back to the
            # explicit manipulation_look_at when one is provided (identical to prior behavior).
            effective_look_at = manipulation_look_at
            if (effective_look_at is None and stance_plan is not None
                    and stance_plan.get("status") == "accepted"):
                effective_look_at = (
                    _surface_affordance_point_for_stance(stance_plan, stand_root)
                    or stance_plan.get("task_target_xyz")
                )
                if effective_look_at is not None:
                    _log(f"scenario {sid}: camera/reach look-at resolved from scene+task surface -> "
                         f"{tuple(round(float(c), 2) for c in effective_look_at)}")
                    # The pre-loop fill light only fires for an explicit look-at; light the
                    # dynamically-resolved workspace too so the dynamic (no-coords) path isn't dark.
                    if manipulation_cam and fill_light_intensity > 0:
                        try:
                            _add_workspace_fill_light(stage, effective_look_at,
                                                      intensity=fill_light_intensity)
                            _log(f"scenario {sid}: workspace fill light @ "
                                 f"{tuple(round(float(c),2) for c in effective_look_at)} "
                                 f"intensity={fill_light_intensity} (resolved)")
                        except Exception as exc:  # noqa: BLE001 - lighting is best-effort
                            _log(f"dynamic workspace fill light skipped ({exc!r})")
            if sc.get("task_target_deferred"):
                if (
                    manipulation_stand
                    and stance_plan is not None
                    and stance_plan.get("status") == "accepted"
                    and stand_root is not None
                ):
                    _materialize_deferred_task_route(
                        sc,
                        stance_plan=stance_plan,
                        root_pose=stand_root,
                        look_at=effective_look_at,
                    )
                    _log(
                        f"scenario {sid}: task-only route materialized from accepted stance "
                        f"start={tuple(round(float(c), 2) for c in sc['start'])} "
                        f"target={tuple(round(float(c), 2) for c in sc['target'])}"
                    )
                else:
                    blockers.append("task_only_target_resolution_failed")
                    _log(f"scenario {sid}: task-only target could not be resolved into an accepted stance")
                    return None
            if not sc.get("route_points") or sc.get("start") is None or sc.get("target") is None:
                blockers.append("scenario_route_missing_after_task_resolution")
                _log(f"scenario {sid}: missing start/target/route after task resolution")
                return None
            pol = policy_mod.make_policy(policy_id)
            pol.reset(sc)
            t_sc = time.time()
            _log(f"scenario {sid}: warmup {warmup_frames} render frames (capped {per_scenario_seconds}s)")
            for wi in range(warmup_frames):
                if time.time() - t_sc > per_scenario_seconds:
                    _log(f"warmup hit time cap at frame {wi}")
                    break
                ts = time.time()
                rep.orchestrator.step()
                _log(f"warmup frame {wi} render took {time.time() - ts:.1f}s")
            actions: list[dict] = []
            skel_rows: list[dict] = []
            trace = (sdir / "trace.jsonl").open("w")
            rejected_total = response_total = 0
            cap = 0
            truncated = False
            _log(f"scenario {sid}: stepping {steps}")
            for step in range(steps):
                if manipulation_stand and stand_root is None:
                    truncated = True
                    break
                if time.time() - t_sc > per_scenario_seconds:
                    _log(f"scenario {sid}: per-scenario cap {per_scenario_seconds}s hit at step {step}; truncating")
                    truncated = True
                    break
                ctx = policy_mod.StepContext(step=step, num_steps=steps, probe_collision=probe)
                decision = pol.step(ctx)
                if manipulation_stand and stand_root is not None:
                    # Manipulation task: place the robot at the probed clear-floor task stance,
                    # facing the target. The target is what the robot faces, not the pelvis position.
                    decision.root_pose = stand_root
                    decision.yaw = stand_yaw
                    decision.desired_root_position = stand_root
                    decision.policy_action = "accepted_task_stance_collision_checked_placement"
                    if stance_plan is not None:
                        candidates = stance_plan.get("candidates", [])
                        decision.collision_probe_candidate_count = len(candidates)
                        decision.rejected_collision_probe_count = int(
                            stance_plan.get("selected_candidate_index") or 0
                        )
                        decision.rejected_probes = [
                            c for c in candidates
                            if int(c.get("scene_collision_contact_count") or 0) > 0
                        ]
                rejected_total += decision.rejected_collision_probe_count
                if decision.policy_action != "accepted_direct_collision_checked_motion":
                    response_total += 1
                route_distance_m = policy_mod.route_distance(sc["route_points"])
                alpha = 0.0 if steps <= 1 else step / float(steps - 1)
                phase = policy_mod.gait_phase(alpha, route_distance_m)
                moving = route_distance_m > 0.05 and step < max(1, steps - 1)
                if art_ctx is not None:
                    if dynamic_standing_contact_steps > 0:
                        report = _settle_dynamic_standing_contacts(
                            stage=stage,
                            art_ctx=art_ctx,
                            robot_prim_path=binding["prim_path"],
                            root_pose=decision.root_pose,
                            yaw=decision.yaw,
                            phase=phase,
                            moving=moving,
                            settle_steps=dynamic_standing_contact_steps,
                            scenario_id=sid,
                            manipulation_ready=bool(manipulation_reach),
                            manipulation_reach_arm=manipulation_reach_arm,
                            root_pose_seeded_before_tensor_view=sid in dynamic_seed_decisions,
                        )
                        report["step"] = step
                        physics_contact_reports.append(report)
                        if report["status"] != "completed":
                            blockers.append("physics_articulation_standing_contact_settle_failed")
                            _log(f"dynamic standing/contact settle failed at step {step}: {report['errors']}")
                    else:
                        _drive_g1_walk(
                            art_ctx["art"],
                            art_ctx["dof_index"],
                            art_ctx["default"],
                            root_pose=decision.root_pose,
                            yaw=decision.yaw,
                            phase=phase,
                            moving=moving,
                            manipulation_ready=bool(manipulation_reach),
                            manipulation_reach_arm=manipulation_reach_arm,
                        )
                else:
                    if robot_neutral_xforms:
                        restored = _restore_robot_neutral_descendant_xforms(stage, robot_neutral_xforms)
                        if step == 0:
                            _log(
                                f"scenario {sid}: neutral G1 descendant xforms restored "
                                f"({restored} prims) before root placement/reach pose"
                            )
                    last_root_diagnostics = _place_root(
                        stage,
                        binding["prim_path"],
                        decision.root_pose,
                        decision.yaw,
                    )
                    if (
                        manipulation_stand
                        and stance_plan is not None
                        and stance_plan.get("status") == "accepted"
                        and placement_validation_manifest is None
                    ):
                        placement_validation_manifest = _build_placement_validation_manifest(
                            stage=stage,
                            robot_prim_path=binding["prim_path"],
                            stance_plan=stance_plan,
                            accepted_pose=decision.root_pose,
                            accepted_yaw=decision.yaw,
                            root_diagnostics=last_root_diagnostics,
                            scene_objects=scene_objects_for_validation,
                            scenario_id=sid,
                        )
                        placement_validation_path.write_text(
                            json.dumps(placement_validation_manifest, indent=2)
                        )
                        if not _placement_validation_passed_manifest(placement_validation_manifest):
                            if "placement_validation_failed" not in blockers:
                                blockers.append("placement_validation_failed")
                            placement_validation_blocker_recorded = True
                            _log(
                                f"scenario {sid}: placement validation FAILED "
                                f"{placement_validation_manifest.get('blockers')}"
                            )
                        else:
                            _log(
                                f"scenario {sid}: placement validation PASS "
                                f"xy_error={placement_validation_manifest['ground_truth_placement'].get('xy_error_m')}"
                            )
                    # Show manipulation-ready arms in the RENDERED frame (pure USD, no physics tensor
                    # -> no crash). The same requested arm set is used for render, camera metadata, and
                    # geometry validation so a both-arm seed cannot be validated as a one-arm frame.
                    if (kinematic_arm_pose and manipulation_reach
                            and effective_look_at is not None):
                        arm_frac = 1.0 if manipulation_cam else alpha
                        try:
                            posed_count = _pose_arm_kinematic_usd(
                                stage,
                                binding["prim_path"],
                                effective_look_at,
                                arm=rendered_reach_arm,
                                reach_frac=arm_frac,
                            )
                            if step == 0:
                                _log(
                                    f"scenario {sid}: kinematic arm pose requested "
                                    f"arm={rendered_reach_arm} "
                                    f"requested_arm={manipulation_reach_arm} "
                                    f"posed_count={posed_count} "
                                    f"target={tuple(round(float(c), 3) for c in effective_look_at)}"
                                )
                        except Exception as exc:  # noqa: BLE001 - pose is best-effort, never blocks frames
                            if step == 0:
                                _log(f"kinematic arm pose skipped ({exc!r})")
                rec = policy_mod.action_record(
                    decision=decision, step=step, sim_time_s=step / float(fps), target=sc["target"],
                    scenario_eval_run_id=sc.get("scenario_eval_run_id"))
                actions.append(rec)
                trace.write(json.dumps(rec) + "\n")
                if step % max(1, capture_every) == 0:
                    cam_meta: dict[str, Any] | None = None
                    if manipulation_cam:
                        eye, tgt, cam_meta = _robot_mounted_manipulation_cam_pose(
                            stage,
                            binding["prim_path"],
                            decision.root_pose,
                            decision.yaw,
                            look_at=effective_look_at,
                            reach_arm=pov_reach_arm,
                            vfov_deg=pov_vfov_deg,
                            width=width,
                            height=height,
                        )
                        if cap == 0:
                            _log(
                                "scenario "
                                f"{sid}: robot_pov camera source={cam_meta.get('source')} "
                                f"mount={cam_meta.get('mount_prim_path')} "
                                f"arm_links={cam_meta.get('arm_link_points_used')} "
                                f"target_select={cam_meta.get('selected_camera_target')}"
                            )
                    else:
                        eye, tgt = follow_cam_pose(decision.root_pose, decision.yaw)
                    if (
                        manipulation_cam
                        and manipulation_reach
                        and effective_look_at is not None
                        and cam_meta is not None
                    ):
                        pov_geom = _manipulation_pov_geometry(
                            arm_points=cam_meta.get("arm_link_points_xyz") or {},
                            arm_points_by_arm=cam_meta.get("arm_link_points_by_arm_xyz") or {},
                            affordance=effective_look_at,
                            eye=eye,
                            target=tgt,
                            vfov_deg=pov_vfov_deg,
                            width=width,
                            height=height,
                            arm=pov_reach_arm,
                        )
                        pov_geom = {
                            **pov_geom,
                            "step": step,
                            "frame_index": cap,
                            "camera_meta": cam_meta,
                        }
                        pov_geometry_records.append(pov_geom)
                        pov_geometry_report = _write_pov_geometry_report()
                        if cap == 0:
                            _log(
                                f"scenario {sid}: manipulation POV geometry "
                                f"{pov_geom.get('status')} roles={pov_geom.get('arm_roles_in_frame')} "
                                f"target_in_frame={pov_geom.get('target_in_frame')}"
                            )
                        if pov_geom.get("status") != "PASS" and not pov_geometry_blocker_recorded:
                            blockers.append("manipulation_pov_geometry_failed")
                            pov_geometry_blocker_recorded = True
                    _place_camera(stage, pov_cam, eye, tgt)  # POV camera (manipulation egocentric or follow)
                    if manipulation_cam and manipulation_reach and effective_look_at is not None:
                        # Front-light the reaching arm+gripper from the camera side (the workspace fill
                        # light is BEYOND the arm, leaving its camera-facing side in shadow/black).
                        _add_pov_headlamp(stage, eye, effective_look_at,
                                          intensity=(fill_light_intensity if fill_light_intensity > 0 else 30000.0))
                    if verify_annot is not None:
                        v_eye, v_tgt = verify_cam_pose(decision.root_pose, decision.yaw)
                        _place_camera(stage, verify_cam_path, v_eye, v_tgt)  # 3rd-person: SHOW the robot
                    debug_root_path = (
                        f"/World/PlacementDebug/{_safe_prim_segment(sid)}"
                        if topdown_annot is not None and stance_plan is not None
                        else None
                    )
                    if debug_root_path is not None:
                        try:
                            stage.RemovePrim(debug_root_path)
                        except Exception:  # noqa: BLE001 - absence is fine
                            pass
                    if articulated and (art_ctx is not None or rest_offsets is not None):
                        skel = skeleton_world_for_frame(
                            art_ctx=art_ctx,
                            rest_offsets=rest_offsets,
                            root_pose=decision.root_pose,
                            yaw=decision.yaw,
                        )
                        if manipulation_reach and effective_look_at is not None:
                            # For manipulation POVs the first frame is already "task started": arms
                            # visible in the workspace. Navigation/follow shots can still ramp.
                            reach_frac = 1.0 if manipulation_cam else alpha
                            skel = compute_arm_reach_skeleton(skel, effective_look_at, reach_frac,
                                                              arm=rendered_reach_arm)
                        lms = _project_skeleton(skel, eye=eye, target=tgt, up=(0.0, 0.0, 1.0),
                                                vfov_deg=pov_vfov_deg, width=width, height=height)
                        if cap == 0:
                            _log(f"step {step}: skeleton {len(skel)} links -> {len(lms)} landmarks in POV frame")
                        skel_rows.append({
                            "episode_id": sid,
                            "scenario_eval_run_id": sc.get("scenario_eval_run_id") or sid,
                            "step": step, "sim_time_s": round(step / float(fps), 6),
                            "camera": "robot_pov", "landmarks": lms,  # OSCAR reads row["landmarks"]
                            "projected_landmark_count": len(lms)})
                    ts = time.time()
                    # Accumulate N RTX subframes on the static (robot placed) frame to drain the
                    # RayTracedLighting denoiser's grain — a single step leaves heavy noise that an
                    # OSCAR start frame should not inherit.
                    for _ in range(max(1, render_subframes)):
                        rep.orchestrator.step()
                    rdt = time.time() - ts
                    over_ok = _save_rgb(
                        over_annot,
                        sdir / "frames" / f"overview_{cap:04d}.png",
                        software_denoise=software_denoise,
                    )
                    pov_frame_path = sdir / "frames" / f"robot_pov_{cap:04d}.png"
                    pov_ok = _save_rgb(pov_annot, sdir / "frames" / f"robot_pov_{cap:04d}.png",
                                       software_denoise=software_denoise)
                    if manipulation_cam and manipulation_reach and pov_geometry_records:
                        frame_quality = (
                            _pov_seed_frame_quality(pov_frame_path)
                            if pov_ok
                            else {
                                "schema_version": "manipulation_pov_seed_frame_quality.v1",
                                "status": "FAIL",
                                "blockers": ["manipulation_pov_frame_not_saved"],
                                "frame_path": str(pov_frame_path),
                            }
                        )
                        last_pov_geom = dict(pov_geometry_records[-1])
                        frame_blockers = [str(b) for b in (frame_quality.get("blockers") or [])]
                        merged_blockers = sorted(set((last_pov_geom.get("blockers") or []) + frame_blockers))
                        last_pov_geom["seed_frame_quality"] = frame_quality
                        last_pov_geom["blockers"] = merged_blockers
                        last_pov_geom["status"] = "PASS" if not merged_blockers else "FAIL"
                        pov_geometry_records[-1] = last_pov_geom
                        pov_geometry_report = _write_pov_geometry_report()
                        if frame_quality.get("status") != "PASS" and not pov_geometry_blocker_recorded:
                            blockers.append("manipulation_pov_geometry_failed")
                            pov_geometry_blocker_recorded = True
                    if verify_annot is not None:
                        _save_rgb(verify_annot, sdir / "frames" / f"verify_{cap:04d}.png",
                                  software_denoise=software_denoise)
                    if topdown_annot is not None and stance_plan is not None and debug_root_path is not None:
                        floor_z = float(decision.root_pose[2]) - ROBOT_PELVIS_HEIGHT_M
                        if stance_plan.get("floor_z_hint") is not None:
                            floor_z = float(stance_plan.get("floor_z_hint"))
                        topdown_debug = _update_topdown_debug_scene(
                            stage,
                            root_path=debug_root_path,
                            robot_pose=decision.root_pose,
                            robot_yaw=decision.yaw,
                            stance_plan=stance_plan,
                            scene_objects=scene_objects_for_validation,
                            floor_z=floor_z,
                        )
                        _place_topdown_debug_camera(
                            stage,
                            topdown_cam_path,
                            center_xy=topdown_debug["center_xy"],
                            radius_m=float(topdown_debug["radius_m"]),
                            width=width,
                            height=height,
                            floor_z=floor_z,
                        )
                        for _ in range(max(1, render_subframes)):
                            rep.orchestrator.step()
                        placement_topdown_frame_path = sdir / "frames" / f"placement_topdown_{cap:04d}.png"
                        _save_rgb(
                            topdown_annot,
                            placement_topdown_frame_path,
                            software_denoise=software_denoise,
                        )
                        placement_topdown_layout_frame_path = (
                            sdir / "frames" / f"placement_topdown_layout_{cap:04d}.png"
                        )
                        _write_topdown_debug_layout_image(
                            placement_topdown_layout_frame_path,
                            robot_pose=decision.root_pose,
                            robot_yaw=decision.yaw,
                            stance_plan=stance_plan,
                            scene_objects=scene_objects_for_validation,
                            width=width,
                            height=height,
                        )
                        try:
                            stage.RemovePrim(debug_root_path)
                        except Exception:  # noqa: BLE001
                            pass
                    if cap == 0 or rdt > 5:
                        _log(f"scenario {sid}: frame {cap} captured (render {rdt:.1f}s, overview_ok={over_ok})")
                    cap += 1
            trace.close()
            if articulated and skel_rows:
                with (sdir / "g1_projected_skeleton_trace.jsonl").open("w") as sf:
                    for r in skel_rows:
                        sf.write(json.dumps(r) + "\n")
                total_lm = sum(r["projected_landmark_count"] for r in skel_rows)
                _log(f"scenario {sid}: skeleton trace {len(skel_rows)} frames, {total_lm} total landmarks")
            scenario_contact_reports = [
                r for r in physics_contact_reports if r.get("scenario_id") == sid
            ]
            if scenario_contact_reports:
                (sdir / "physics_articulation_standing_contact_reports.json").write_text(
                    json.dumps(scenario_contact_reports, indent=2)
                )
            if manipulation_stand:
                if placement_validation_manifest is None and stance_plan is not None and stand_root is not None:
                    placement_validation_manifest = _build_placement_validation_manifest(
                        stage=stage,
                        robot_prim_path=binding["prim_path"],
                        stance_plan=stance_plan,
                        accepted_pose=stand_root,
                        accepted_yaw=float(stand_yaw),
                        root_diagnostics=last_root_diagnostics,
                        scene_objects=scene_objects_for_validation,
                        scenario_id=sid,
                        topdown_frame=(
                            str(placement_topdown_layout_frame_path or placement_topdown_frame_path)
                            if (placement_topdown_layout_frame_path or placement_topdown_frame_path)
                            else None
                        ),
                    )
                if placement_validation_manifest is not None:
                    placement_visual_frames = sorted((sdir / "frames").glob("verify_*.png"))
                    pov_visual_frames = sorted((sdir / "frames").glob("robot_pov_*.png"))
                    visual_qc = _run_task_visual_qc(
                        placement_visual_frames,
                        pov_visual_frames if manipulation_cam else [],
                        target_label=_placement_visual_qc_target_label(stance_plan),
                        task_description=_task_description_for_scenario(sc),
                    )
                    placement_validation_manifest["visual_qc"] = visual_qc
                    placement_validation_manifest["topdown_debug_frame"] = (
                        str(placement_topdown_layout_frame_path or placement_topdown_frame_path)
                        if (placement_topdown_layout_frame_path or placement_topdown_frame_path)
                        else None
                    )
                    placement_validation_manifest["orthographic_topdown_frame"] = (
                        str(placement_topdown_frame_path) if placement_topdown_frame_path else None
                    )
                    placement_validation_manifest["topdown_layout_frame"] = (
                        str(placement_topdown_layout_frame_path)
                        if placement_topdown_layout_frame_path
                        else None
                    )
                    blockers_now = set(placement_validation_manifest.get("blockers") or [])
                    if visual_qc.get("status") != "passed":
                        blockers_now.add("placement_visual_qc_failed")
                    if pov_geometry_report is not None:
                        placement_validation_manifest["manipulation_pov_geometry"] = {
                            "status": pov_geometry_report.get("status"),
                            "path": str(pov_geometry_path),
                            "blockers": pov_geometry_report.get("blockers", []),
                            "frames_checked": pov_geometry_report.get("frames_checked"),
                        }
                        if pov_geometry_report.get("status") != "PASS":
                            blockers_now.add("manipulation_pov_geometry_failed")
                            blockers_now.update(pov_geometry_report.get("blockers") or [])
                    placement_validation_manifest["blockers"] = sorted(blockers_now)
                    placement_validation_manifest["status"] = (
                        "PASS" if not placement_validation_manifest["blockers"] else "FAIL"
                    )
                    placement_validation_path.write_text(
                        json.dumps(placement_validation_manifest, indent=2)
                    )
                    if not _placement_validation_passed_manifest(placement_validation_manifest):
                        if "placement_validation_failed" not in blockers:
                            blockers.append("placement_validation_failed")
                        if not placement_validation_blocker_recorded:
                            _log(
                                f"scenario {sid}: placement validation FAILED "
                                f"{placement_validation_manifest.get('blockers')}"
                            )
                    else:
                        _log(f"scenario {sid}: placement validation final PASS")
            _log(f"scenario {sid}: {cap} frames captured, truncated={truncated}; assembling MP4 + outcome")
            summary = assemble_collision_summary(actions=actions, rejected_probe_total=rejected_total,
                                                 response_event_total=response_total)
            outcome = policy_mod.compute_task_outcome(
                actions=actions, start=sc["start"], target=sc["target"],
                route_distance_m=policy_mod.route_distance(sc["route_points"]),
                collision_summary=summary, bounded_steps=len(actions), model_timestep_s=1.0 / float(fps))
            outcome["frames_captured"] = cap
            outcome["truncated"] = truncated
            if stance_plan is not None:
                outcome["task_stance_plan"] = {
                    "status": stance_plan.get("status"),
                    "path": str(stance_plan_path),
                    "accepted_pose": stance_plan.get("accepted_pose"),
                    "accepted_yaw": stance_plan.get("accepted_yaw"),
                    "candidate_count": len(stance_plan.get("candidates", [])),
                    "blockers": stance_plan.get("blockers", []),
                }
            if placement_validation_manifest is not None:
                outcome["placement_validation"] = {
                    "status": placement_validation_manifest.get("status"),
                    "path": str(placement_validation_path),
                    "blockers": placement_validation_manifest.get("blockers", []),
                    "ground_truth_xy_error_m": (
                        placement_validation_manifest.get("ground_truth_placement") or {}
                    ).get("xy_error_m"),
                    "visual_qc_status": (
                        placement_validation_manifest.get("visual_qc") or {}
                    ).get("status"),
                    "topdown_debug_frame": placement_validation_manifest.get("topdown_debug_frame"),
                }
            if pov_geometry_report is not None:
                outcome["manipulation_pov_geometry"] = {
                    "status": pov_geometry_report.get("status"),
                    "path": str(pov_geometry_path),
                    "blockers": pov_geometry_report.get("blockers", []),
                    "frames_checked": pov_geometry_report.get("frames_checked"),
                }
            outcomes.append(outcome)  # record BEFORE MP4 — MP4 is optional, frames already uploaded
            for name in ("overview", "robot_pov", "placement_topdown"):
                glob = str(sdir / "frames" / f"{name}_*.png")
                try:
                    subprocess.call(mp4_command(glob, fps, str(sdir / f"{name}.mp4")))
                except Exception as e:  # noqa: BLE001
                    _log(f"mp4 assembly for {name} failed ({e!r}); frames preserved for local assembly")
            _log(f"scenario {sid}: done")
            return outcome

        if serve:
            try:  # worker: flat module in the bundle dir (sys.path has it); repo/tests: package
                from warm_render_server import (  # type: ignore
                    FileJobSource, SignedUrlJobSource, serve_render_loop)
            except ImportError:
                from blueprint_pipeline.warm_render_server import (  # type: ignore
                    FileJobSource, SignedUrlJobSource, serve_render_loop)
            inbox_get_url = os.environ.get("BLUEPRINT_WARM_INBOX_GET_URL", "").strip()
            if inbox_get_url:
                warm_source = SignedUrlJobSource(inbox_get_url, out_dir)
                source_label = "signed_url_inbox"
            else:
                serve_root = Path(serve_dir) if serve_dir is not None else (out_dir / "warm_jobs")
                warm_source = FileJobSource(serve_root)
                source_label = f"file:{serve_root}"

            def _serve_render_one(job_scenario):
                normalized = parse_scenarios({"scenarios": [dict(job_scenario)]})
                if not normalized:
                    return {"status": "blocked", "blockers": ["scenario_unparseable"]}
                produced = _render_scenario(normalized[0])
                return {
                    "status": "completed" if produced is not None else "skipped",
                    "scenario_id": normalized[0].get("scenario_id"),
                    "outcome": produced,
                }

            # Readiness marker: the control plane polls the output zip for this so it knows Isaac is
            # booted + the scene is loaded + the loop is accepting jobs (so it can start submitting).
            (out_dir / "warm_serve_ready.json").write_text(json.dumps(
                {"status": "serving", "source": source_label,
                 "idle_timeout_s": serve_idle_timeout_s, "max_jobs": serve_max_jobs}))
            _log(f"warm serve: setup complete; serving jobs from {source_label} "
                 f"(idle_timeout={serve_idle_timeout_s}s, max_jobs={serve_max_jobs})")
            serve_summary = serve_render_loop(
                render_one=_serve_render_one, job_source=warm_source,
                idle_timeout_s=serve_idle_timeout_s, max_jobs=serve_max_jobs, log=_log)
            (out_dir / "warm_serve_summary.json").write_text(json.dumps(serve_summary, indent=2))
            _log(f"warm serve: exit {serve_summary}")
        else:
            for sc in scenarios:
                _render_scenario(sc)
    finally:
        try:
            # SimulationApp.close() can terminate or stall the worker process on some
            # remote runtimes, so persist the collector-visible result before closing Isaac.
            if dynamic_standing_contact_steps > 0 and not outcomes and not blockers:
                blockers.append("physics_articulation_dynamic_standing_contact_stopped_before_outcome")
            result = build_result(scenarios=scenarios, outcomes=outcomes, policy_id=policy_id,
                                  kitchen_usd=kitchen_usd, g1_usd=g1_usd, blockers=blockers,
                                  physics_articulation_contact_reports=physics_contact_reports)
            (out_dir / "isaac_g1_kitchen_parity_result.json").write_text(json.dumps(result, indent=2))
        finally:
            try:
                sim.close()
            except Exception:  # noqa: BLE001
                pass
    assert result is not None
    return result


# --------------------------------------------------------------------------------------------------
# Local dry-render preview (NO GPU, NO Isaac) — catch placement/camera/POV-framing bugs in <1s
# before spending a cloud render. It reproduces the SAME stance + camera + arm-skeleton math the GPU
# path runs (plan_task_stance / manipulation_cam_pose / compute_arm_reach_skeleton / project_point_to_
# pixel), so wrong-side stance, a camera that crops the reaching arm, or a camera aimed into an
# appliance are all visible locally. The arm skeleton uses a NOMINAL (Isaac-free) G1, so the arm is
# approximate; the stance, camera pose, and projection are exact.
# --------------------------------------------------------------------------------------------------

# Nominal G1 link offsets from the pelvis root (robot frame: +x forward, +y left, +z up). Approximate
# but dimensionally faithful enough that camera framing of the reaching arm transfers to the GPU run.
# Link names mirror the real USD ("...link") so compute_arm_reach_skeleton recognizes the arm chain.
_NOMINAL_G1_REST_OFFSETS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("pelvis_link", (0.0, 0.0, 0.0)),
    ("torso_link", (0.0, 0.0, 0.20)),
    ("head_link", (0.06, 0.0, 0.45)),
    ("right_shoulder_link", (0.0, -0.16, 0.34)),
    ("right_elbow_link", (0.06, -0.20, 0.10)),
    ("right_wrist_link", (0.12, -0.22, -0.06)),
    ("right_hand_link", (0.16, -0.23, -0.14)),
    ("left_shoulder_link", (0.0, 0.16, 0.34)),
    ("left_elbow_link", (0.06, 0.20, 0.10)),
    ("left_wrist_link", (0.12, 0.22, -0.06)),
    ("left_hand_link", (0.16, 0.23, -0.14)),
)


def _open_stage_local(usd_path: str):
    """Open a USD stage with plain pxr (NO omni/Isaac) for the local dry-render's geometry reads."""
    from pxr import Usd  # type: ignore
    return Usd.Stage.Open(str(usd_path))


def _bind_proxy_robot(
    stage,
    prim_path: str = "/World/G1",
    *,
    footprint_xy: tuple[float, float] = (0.50, 0.36),
    height: float = 1.55,
) -> str:
    """Bind a footprint-sized proxy robot box at ``prim_path`` so the geometric placement validator can
    place + bbox a 'robot' locally (the real G1 USD is an Isaac asset not present off-GPU). The proxy
    reproduces the standing footprint the validator checks for clip/standoff, no Isaac needed."""
    from pxr import UsdGeom, Gf  # type: ignore
    UsdGeom.Xform.Define(stage, prim_path)
    cube = UsdGeom.Cube.Define(stage, prim_path + "/proxy_body")
    cube.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddScaleOp().Set(Gf.Vec3d(float(footprint_xy[0]), float(footprint_xy[1]), float(height)))
    return prim_path


def nominal_g1_rest_offsets() -> list[tuple[str, tuple[float, float, float]]]:
    """Isaac-free nominal G1 rest-pose link offsets (pelvis frame), for the local dry-render skeleton.

    Same shape as :func:`_g1_link_rest_offsets` (which reads the real USD), so it feeds straight into
    :func:`_rest_skeleton_world` + :func:`compute_arm_reach_skeleton`. Approximate dimensions — used to
    preview whether the reaching arm lands in the camera frame, not to claim exact joint geometry.
    """
    return [(name, (float(o[0]), float(o[1]), float(o[2]))) for name, o in _NOMINAL_G1_REST_OFFSETS]


def _facing_error_deg(root_pose, yaw: float, target) -> float | None:
    """Angle (deg) between where the robot faces and the XY direction from the root to the target."""
    if target is None:
        return None
    dx = float(target[0]) - float(root_pose[0])
    dy = float(target[1]) - float(root_pose[1])
    if math.hypot(dx, dy) < 1e-6:
        return 0.0
    want = math.atan2(dy, dx)
    err = math.degrees(abs((want - float(yaw) + math.pi) % (2 * math.pi) - math.pi))
    return round(err, 3)


def _dry_render_checks(summary) -> dict[str, bool]:
    """Boolean pass/fail for the dry-render checklist. Centralized so the drawn labels and any callers
    agree — and so a perfect ``0.0`` facing error is not swallowed by a ``0.0 or default`` falsy trap."""
    chk = summary.get("checks", {}) or {}
    pf = summary.get("pov_framing", {}) or {}
    st = summary.get("stance", {}) or {}
    fe = chk.get("facing_error_deg")
    return {
        "faces_target": fe is not None and float(fe) < 8.0,
        "target_in_frame": bool(pf.get("target_in_frame")),
        "arm_in_frame": int(pf.get("arm_landmarks_in_frame") or 0) >= 1,
        "no_blockers": not st.get("blockers"),
    }


def _draw_dry_render_preview(
    path,
    *,
    scenario,
    stance_plan,
    root_pose,
    yaw: float,
    look_at,
    eye,
    target,
    pov_vfov_deg: float,
    width: int,
    height: int,
    skeleton_world,
    scene_objects,
    arm: str,
    summary,
) -> None:
    """Draw the 3-panel dry-render preview PNG: top-down placement, egocentric POV framing, summary."""
    from PIL import Image, ImageDraw  # local: only the dry-render path needs PIL

    PW, PH, GUT, TOP = 460, 420, 18, 34
    W = GUT + 3 * (PW + GUT)
    H = TOP + PH + GUT
    img = Image.new("RGB", (W, H), (22, 24, 28))
    d = ImageDraw.Draw(img)
    panels = [GUT + i * (PW + GUT) for i in range(3)]
    d.text((GUT, 8), f"DRY RENDER (no GPU) - {scenario.get('description') or scenario.get('instruction') or ''}"[:120],
           fill=(235, 235, 235))
    for x0 in panels:
        d.rectangle([x0, TOP, x0 + PW, TOP + PH], outline=(70, 74, 82))

    # ---- Panel A: top-down placement ----
    ax0 = panels[0]
    d.text((ax0 + 6, TOP + 4), "top-down placement", fill=(180, 200, 220))
    xs, ys = [float(root_pose[0]), float(eye[0])], [float(root_pose[1]), float(eye[1])]
    tb = stance_plan.get("task_target_bounds") if isinstance(stance_plan, dict) else None
    if tb:
        bmin, bmax = tb["bbox_min_xyz"], tb["bbox_max_xyz"]
        xs += [bmin[0], bmax[0]]; ys += [bmin[1], bmax[1]]
    for obj in scene_objects[:60]:
        try:
            xs += [obj.bbox_min[0], obj.bbox_max[0]]; ys += [obj.bbox_min[1], obj.bbox_max[1]]
        except Exception:  # noqa: BLE001
            continue
    if look_at is not None:
        xs.append(float(look_at[0])); ys.append(float(look_at[1]))
    pad = 0.6
    minx, maxx, miny, maxy = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    span = max(maxx - minx, maxy - miny, 1e-3)
    inner = PH - 36

    def w2s(wx, wy):
        sx = ax0 + 14 + (float(wx) - minx) / span * inner
        sy = TOP + 24 + (maxy - float(wy)) / span * inner  # +y world is up on screen
        return (sx, sy)

    for obj in scene_objects[:60]:
        try:
            p0 = w2s(obj.bbox_min[0], obj.bbox_min[1]); p1 = w2s(obj.bbox_max[0], obj.bbox_max[1])
            d.rectangle([min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])],
                        outline=(95, 100, 110))
        except Exception:  # noqa: BLE001
            continue
    if tb:
        p0 = w2s(bmin[0], bmin[1]); p1 = w2s(bmax[0], bmax[1])
        d.rectangle([min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])],
                    outline=(240, 150, 40), width=2)
    # robot footprint (rotated rectangle) + facing arrow
    hx, hy = 0.25, 0.18
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    poly = [w2s(root_pose[0] + cx * cyaw - cy * syaw, root_pose[1] + cx * syaw + cy * cyaw)
            for cx, cy in corners]
    d.polygon(poly, outline=(90, 170, 250))
    rc = w2s(root_pose[0], root_pose[1])
    fa = w2s(root_pose[0] + cyaw * 0.5, root_pose[1] + syaw * 0.5)
    d.line([rc, fa], fill=(230, 80, 80), width=2)
    d.ellipse([rc[0] - 3, rc[1] - 3, rc[0] + 3, rc[1] + 3], fill=(90, 170, 250))
    # camera eye + look-at + frustum edges
    ec = w2s(eye[0], eye[1])
    d.ellipse([ec[0] - 3, ec[1] - 3, ec[0] + 3, ec[1] + 3], fill=(80, 220, 130))
    if look_at is not None:
        lc = w2s(look_at[0], look_at[1])
        d.line([ec[0] - 5, ec[1], ec[0] + 5, ec[1]], fill=(240, 150, 40))
        d.line([lc[0] - 5, lc[1], lc[0] + 5, lc[1]], fill=(240, 150, 40))
        d.line([lc[0], lc[1] - 5, lc[0], lc[1] + 5], fill=(240, 150, 40))
        fdx, fdy = float(look_at[0]) - float(eye[0]), float(look_at[1]) - float(eye[1])
        ang = math.atan2(fdy, fdx)
        half = math.radians(pov_vfov_deg * (width / max(height, 1)) / 2.0)
        for s in (-1.0, 1.0):
            a = ang + s * half
            fp = w2s(eye[0] + math.cos(a) * span * 0.5, eye[1] + math.sin(a) * span * 0.5)
            d.line([ec, fp], fill=(60, 130, 90))

    # ---- Panel B: egocentric POV framing ----
    bx0 = panels[1]
    d.text((bx0 + 6, TOP + 4), f"egocentric POV  vfov={pov_vfov_deg:.0f}deg  (arm=APPROX)", fill=(180, 200, 220))
    fx0, fy0 = bx0 + 18, TOP + 28
    fw = PW - 36
    fh = int(fw * height / max(width, 1))
    if fh > PH - 46:
        fh = PH - 46; fw = int(fh * width / max(height, 1))
    d.rectangle([fx0, fy0, fx0 + fw, fy0 + fh], outline=(120, 125, 135))

    def proj(wp):
        px = project_point_to_pixel(wp, eye, target, (0.0, 0.0, 1.0), pov_vfov_deg, width, height)
        if px is None:
            return None
        return (fx0 + px[0] / width * fw, fy0 + px[1] / height * fh)

    for obj in scene_objects[:60]:
        try:
            sp = proj(obj.footprint_center() + (float(0.5 * (obj.bbox_min[2] + obj.bbox_max[2])),))
        except Exception:  # noqa: BLE001
            sp = None
        if sp:
            d.ellipse([sp[0] - 1.5, sp[1] - 1.5, sp[0] + 1.5, sp[1] + 1.5], fill=(110, 115, 125))
    sk = {n: p for n, p in skeleton_world}
    sides = ("right", "left") if str(arm) == "both" else (str(arm),)
    for side in sides:
        chain = [f"{side}_shoulder_link", f"{side}_elbow_link", f"{side}_wrist_link", f"{side}_hand_link"]
        prev = None
        for i, name in enumerate(chain):
            wp = sk.get(name)
            sp = proj(wp) if wp is not None else None
            if sp is None:
                prev = None
                continue
            if prev is not None:
                d.line([prev, sp], fill=(90, 170, 250), width=2)
            r = 5 if "hand" in name else 3
            d.ellipse([sp[0] - r, sp[1] - r, sp[0] + r, sp[1] + r], fill=(120, 200, 255))
            prev = sp
    if look_at is not None:
        lp = proj((float(look_at[0]), float(look_at[1]), float(look_at[2])))
        if lp:
            d.line([lp[0] - 7, lp[1], lp[0] + 7, lp[1]], fill=(240, 150, 40), width=2)
            d.line([lp[0], lp[1] - 7, lp[0], lp[1] + 7], fill=(240, 150, 40), width=2)

    # ---- Panel C: summary + checklist ----
    cx0 = panels[2] + 8
    d.text((cx0, TOP + 4), "summary", fill=(180, 200, 220))
    chk = summary.get("checks", {})
    pf = summary.get("pov_framing", {})
    st = summary.get("stance", {})
    ok = _dry_render_checks(summary)

    def mark(flag):
        return "OK" if flag else "XX"

    lines = [
        f"stance: {st.get('status')}",
        f"pose:  {['%.2f' % v for v in (st.get('accepted_pose') or [])]}",
        f"yaw:   {st.get('accepted_yaw')}",
        f"look_at: {['%.2f' % v for v in (summary.get('look_at') or [])]}",
        f"standoff_gap_m: {chk.get('standoff_gap_m')}",
        "",
        f"[{mark(ok['faces_target'])}] faces target  ({chk.get('facing_error_deg')} deg)",
        f"[{mark(ok['target_in_frame'])}] target in POV frame",
        f"[{mark(ok['arm_in_frame'])}] arm in POV frame  ({pf.get('arm_landmarks_in_frame')})",
        f"[{mark(ok['no_blockers'])}] no stance blockers",
    ]
    for blk in (st.get("blockers") or [])[:4]:
        lines.append(f"   blocker: {blk}")
    yy = TOP + 26
    for ln in lines:
        d.text((cx0, yy), ln[:60], fill=(225, 225, 225))
        yy += 18

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))


def render_local_preview(
    *,
    stage,
    scenario,
    out_dir,
    manipulation_reach_arm: str = "right",
    camera_vfov_deg: float = 50.0,
    width: int = 1280,
    height: int = 960,
    manipulation_look_at=None,
    robot_prim_path: str | None = None,
) -> dict[str, Any]:
    """Produce a no-GPU dry-render preview (PNG + summary JSON) for a task on an open USD stage.

    Reproduces the GPU runner's stance plan, manipulation camera, and arms-forward seed skeleton, then draws
    a top-down + egocentric-POV + checklist preview so wrong-side stance / cropped-arm / mis-aimed
    camera show up locally in <1s. ``robot_prim_path`` (when a placeable robot proxy is bound) enables
    the geometric placement validator; otherwise stance is planned without obstacle rejection.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = str(scenario.get("scenario_id") or scenario.get("episode_id") or "scenario")

    stance_plan = _plan_task_stance_for_stage(
        stage=stage,
        scenario=scenario,
        manipulation_look_at=manipulation_look_at,
        probe=lambda pose, yaw: 0,
        no_collision_probe=False,
        robot_prim_path=robot_prim_path,
    )

    summary: dict[str, Any] = {
        "scenario_id": sid,
        "task": scenario.get("description") or scenario.get("instruction") or scenario.get("task"),
        "manipulation_reach_arm": manipulation_reach_arm,
        "stance": {
            "status": stance_plan.get("status"),
            "blockers": stance_plan.get("blockers"),
            "accepted_pose": stance_plan.get("accepted_pose"),
            "accepted_yaw": stance_plan.get("accepted_yaw"),
        },
        "claim_boundary": (
            "Local dry-render preview: stance/camera/projection are exact; the reaching arm uses a "
            "nominal (Isaac-free) G1, so arm framing is approximate, not a manipulation-success claim."
        ),
    }

    try:
        scene_objects = _scene_objects_for_stage(stage)
    except Exception:  # noqa: BLE001 - the preview must not hard-fail on scene enumeration
        scene_objects = []

    if stance_plan.get("status") != "accepted":
        summary["pov_framing"] = {"target_in_frame": False, "arm_landmarks_in_frame": 0,
                                  "projected_landmark_count": 0}
        summary["checks"] = {"facing_error_deg": None, "standoff_gap_m": None}
        (out_dir / "dry_render_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    root = tuple(float(v) for v in stance_plan["accepted_pose"])
    yaw = float(stance_plan["accepted_yaw"])
    look_at = manipulation_look_at
    if look_at is None:
        look_at = _surface_affordance_point_for_stance(stance_plan, root) or stance_plan.get("task_target_xyz")
    look_at = tuple(float(v) for v in look_at) if look_at is not None else None

    pov_vfov_deg = max(float(camera_vfov_deg), 68.0)  # manipulation widen — mirrors the runner
    eye, tgt = manipulation_cam_pose(root, yaw, look_at=look_at, reach_arm=manipulation_reach_arm)

    skeleton_world = _rest_skeleton_world(nominal_g1_rest_offsets(), root, yaw)
    if look_at is not None:
        skeleton_world = compute_arm_reach_skeleton(skeleton_world, look_at, 1.0, arm=manipulation_reach_arm)
    pov_lms = _project_skeleton(skeleton_world, eye=eye, target=tgt, up=(0.0, 0.0, 1.0),
                                vfov_deg=pov_vfov_deg, width=width, height=height)

    active = ("right", "left") if str(manipulation_reach_arm) == "both" else (str(manipulation_reach_arm),)
    arm_in_frame = sum(
        1 for lm in pov_lms
        if any(lm["landmark_id"].startswith(f"{s}_") and any(k in lm["landmark_id"] for k in ("hand", "wrist", "elbow"))
               for s in active)
    )
    target_px = (project_point_to_pixel(look_at, eye, tgt, (0.0, 0.0, 1.0), pov_vfov_deg, width, height)
                 if look_at is not None else None)

    summary["look_at"] = list(look_at) if look_at is not None else None
    _target_center = stance_plan.get("task_target_xyz")
    _target_bounds = stance_plan.get("task_target_bounds")
    summary["target"] = {
        "center_xyz": list(_target_center) if _target_center is not None else None,
        "bbox_min_xyz": (_target_bounds or {}).get("bbox_min_xyz"),
        "bbox_max_xyz": (_target_bounds or {}).get("bbox_max_xyz"),
        "resolution_source": (stance_plan.get("target_resolution") or {}).get("source"),
    }
    summary["camera"] = {
        "pov_eye": [round(float(v), 4) for v in eye],
        "pov_target": [round(float(v), 4) for v in tgt],
        "pov_vfov_deg": round(float(pov_vfov_deg), 3),
    }
    summary["pov_framing"] = {
        "target_in_frame": target_px is not None,
        "arm_landmarks_in_frame": int(arm_in_frame),
        "projected_landmark_count": len(pov_lms),
    }
    summary["checks"] = {
        "facing_error_deg": _facing_error_deg(root, yaw, look_at),
        "standoff_gap_m": (stance_plan.get("candidates") or [{}])[stance_plan.get("selected_candidate_index", 0)]
        .get("standoff_from_target_surface_m"),
    }

    _draw_dry_render_preview(
        out_dir / "dry_render_preview.png",
        scenario=scenario, stance_plan=stance_plan, root_pose=root, yaw=yaw, look_at=look_at,
        eye=eye, target=tgt, pov_vfov_deg=pov_vfov_deg, width=width, height=height,
        skeleton_world=skeleton_world, scene_objects=scene_objects, arm=manipulation_reach_arm,
        summary=summary,
    )
    (out_dir / "dry_render_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Isaac G1 kitchen parity eval (GPU)")
    ap.add_argument("--request", help="execution request JSON (scenarios + asset hints)")
    ap.add_argument("--kitchen-usd", help="path/URI to Collected_KitchenRoom/KitchenRoom.usd")
    ap.add_argument("--g1-usd", help="path/URI to the official Isaac G1 USD")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--policy", default="blueprint_default_walk_to_target_smoke_policy")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--warmup-frames", type=int, default=6)
    ap.add_argument("--capture-every", type=int, default=1)
    ap.add_argument("--no-collision-probe", action="store_true",
                    help="skip the PhysX overlap probe (policy goes direct) — decouples render from physics")
    ap.add_argument("--per-scenario-seconds", type=int, default=480,
                    help="wall-clock cap per scenario so a slow/hung render cannot run forever")
    ap.add_argument("--focus-radius", type=float, default=0.0,
                    help="task-aware scene subset: keep only objects within N m of the route (0=full scene)")
    ap.add_argument("--keep-objects", default="room,floor,wall,ground,ceiling,light",
                    help="comma substrings of object names to always keep (structural shell)")
    ap.add_argument("--settle-seconds", type=int, default=0,
                    help="wait N s after scene load (PhysX on) for cooking to drain before rendering")
    ap.add_argument("--disable-physx", action="store_true",
                    help="(experiment only) disable physx cooking — known to break the renderer")
    ap.add_argument("--cheap-collision", action="store_true",
                    help="force bounding-box collision on all meshes (fast cooking; keeps full scene)")
    ap.add_argument("--articulated", action="store_true",
                    help="drive the G1 joints with the walk gait + emit g1_projected_skeleton_trace.jsonl (for OSCAR)")
    ap.add_argument("--camera-vfov", type=float, default=50.0, help="POV camera vertical FOV (deg) for skeleton projection")
    ap.add_argument("--manipulation-cam", action="store_true",
                    help="egocentric manipulation POV (head looking down-forward at the sink/hands) "
                         "instead of the behind-and-above follow cam — for WAM-ing the task, not navigation")
    ap.add_argument("--manipulation-look-at", default=None,
                    help="fixed world 'x,y,z' the manipulation cam aims at (e.g. the faucet) — pins the "
                         "framing to the known workspace instead of the policy's noisy final yaw")
    ap.add_argument("--render-subframes", type=int, default=1,
                    help="RTX orchestrator steps accumulated per captured frame to denoise grain (e.g. 16)")
    ap.add_argument("--no-software-denoise", action="store_true",
                    help="disable CPU-side PNG denoise fallback for saved frames")
    ap.add_argument("--manipulation-reach", action="store_true",
                    help="pose the visible G1 arms into the workspace for manipulation POV review; "
                         "this is posed simulator media, not manipulation-success proof")
    ap.add_argument("--manipulation-reach-arm", default="both", choices=["right", "left", "both"],
                    help="which arm is posed for the task")
    ap.add_argument("--fill-light-intensity", type=float, default=0.0,
                    help="add a sphere fill light over the manipulation workspace (the faucet) at this "
                         "intensity to lift the dark sink basin; 0 disables")
    ap.add_argument("--physics-articulation-drive", action="store_true",
                    help="(opt-in, default off) drive the G1 via the physics articulation tensor view. "
                         "All root seeds stay on the articulation API; the pure-USD root xform fallback "
                         "is used only when this is off.")
    ap.add_argument("--dynamic-standing-contact-steps", type=int, default=0,
                    help="opt-in PhysX standing/contact settle steps per sampled placement. This "
                         "forces --articulated, enables gravity, avoids the SingleArticulation "
                         "tensor view, and records physics_articulation_standing_contact_reports.json. "
                         "It is standing/contact evidence, not full dynamic walking.")
    ap.add_argument("--neutral-environment", action="store_true",
                    help="replace the kitchen asset's outdoor-HDRI dome light with a neutral bright "
                         "environment (no cityscape through the windows + lifts shadowed surfaces)")
    ap.add_argument("--kinematic-arm-pose", action="store_true",
                    help="pose the RENDERED arm(s) into a forward manipulation-ready seed via pure-USD "
                         "shoulder rotation (no physics tensor -> crash-safe); needs --manipulation-reach")
    ap.add_argument("--collision-approximation", default="boundingCube",
                    choices=["boundingCube", "convexHull", "convexDecomposition"],
                    help="mesh collision shape: boundingCube (fast, coarse) vs convexHull (shape-"
                         "accurate enough to stand centered + close at the sink, still fast)")
    ap.add_argument("--verify-cam", action="store_true",
                    help="render a 3rd-person verify_*.png that frames the whole robot at the workspace "
                         "(proves where it stands vs the egocentric POV)")
    ap.add_argument("--manipulation-stand", action="store_true",
                    help="place the robot AT the scenario target facing --manipulation-look-at every "
                         "step (task start pose; no navigation/redirect) — for manipulation, not locomotion")
    ap.add_argument("--dry-render", action="store_true",
                    help="NO-GPU local preview: reproduce stance + camera + arms-forward framing from the "
                         "kitchen USD + task string and write a preview PNG, so placement/POV bugs are caught "
                         "locally before a cloud render. Needs --kitchen-usd + --request (no --g1-usd, no GPU).")
    ap.add_argument("--dry-render-out", default=None,
                    help="output dir for --dry-render artifacts (default: <out-dir>/dry_render)")
    ap.add_argument("--serve", action="store_true",
                    help="PERSISTENT WARM MODE: boot Isaac + load the scene ONCE, then serve a stream of "
                         "task scenarios dropped into --serve-dir (each rerun skips image pull + Isaac boot "
                         "+ stage load + most settle). Exits on a stop sentinel, idle timeout, or max jobs.")
    ap.add_argument("--serve-dir", default=None,
                    help="job/result directory the warm worker polls (default: <out-dir>/warm_jobs)")
    ap.add_argument("--serve-idle-timeout", type=float, default=600.0,
                    help="seconds with no new job before the warm worker exits (default 600)")
    ap.add_argument("--serve-max-jobs", type=int, default=None,
                    help="optional cap on jobs served before the warm worker exits (default: unlimited)")
    return ap


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    manip_look_at = None
    if args.manipulation_look_at:
        parts = [float(v) for v in str(args.manipulation_look_at).replace(" ", "").split(",") if v]
        if len(parts) == 3:
            manip_look_at = (parts[0], parts[1], parts[2])

    request = load_request(args.request) if args.request else {}
    scenarios = parse_scenarios(request)
    kitchen_usd = args.kitchen_usd or request.get("kitchen_usd") or request.get("scene_usd")
    g1_usd = args.g1_usd or request.get("g1_usd")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_render:
        dry_out = Path(args.dry_render_out) if args.dry_render_out else (out_dir / "dry_render")
        if not scenarios or not kitchen_usd:
            res = {"status": "blocked", "blockers": ["missing_scenarios_or_kitchen_usd"],
                   "have_scenarios": bool(scenarios), "have_kitchen_usd": bool(kitchen_usd)}
            print(json.dumps(res))
            return 1
        stage = _open_stage_local(kitchen_usd)
        _bind_proxy_robot(stage, "/World/G1")
        reach_arm = args.manipulation_reach_arm
        summaries = []
        for sc in scenarios:
            sid = str(sc.get("scenario_id") or sc.get("episode_id") or f"scenario_{len(summaries)}")
            summ = render_local_preview(
                stage=stage, scenario=sc, out_dir=dry_out / sid,
                manipulation_reach_arm=reach_arm, camera_vfov_deg=args.camera_vfov,
                width=args.width, height=args.height, manipulation_look_at=manip_look_at,
                robot_prim_path="/World/G1",
            )
            summaries.append(summ)
        (dry_out / "dry_render_index.json").write_text(json.dumps(summaries, indent=2))
        accepted = sum(1 for s in summaries if s.get("stance", {}).get("status") == "accepted")
        print(json.dumps({"status": "dry_render_complete", "scenarios": len(summaries),
                          "accepted": accepted, "out_dir": str(dry_out)}))
        return 0

    put_url = os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL", "")

    # Warm serve mode boots Isaac with NO initial scenarios — jobs arrive at runtime via --serve-dir —
    # so it requires the assets but not a pre-supplied scenario list.
    if not kitchen_usd or not g1_usd or (not scenarios and not args.serve):
        res = {"schema_version": RESULT_SCHEMA_VERSION, "status": "blocked",
               "blockers": ["missing_scenarios_or_kitchen_usd_or_g1_usd"],
               "have_scenarios": bool(scenarios), "have_kitchen_usd": bool(kitchen_usd),
               "have_g1_usd": bool(g1_usd)}
        (out_dir / "isaac_g1_kitchen_parity_result.json").write_text(json.dumps(res, indent=2))
        upload_zip(out_dir, put_url)
        print(json.dumps(res))
        return 1
    result = run_scenarios(
        kitchen_usd=kitchen_usd, g1_usd=g1_usd, scenarios=scenarios, out_dir=out_dir,
        policy_id=args.policy, steps=args.steps, width=args.width, height=args.height,
        fps=args.fps, warmup_frames=args.warmup_frames, capture_every=args.capture_every,
        no_collision_probe=args.no_collision_probe, per_scenario_seconds=args.per_scenario_seconds,
        focus_radius=args.focus_radius,
        keep_substrings=tuple(s for s in args.keep_objects.split(",") if s.strip()),
        disable_physx=args.disable_physx, settle_seconds=args.settle_seconds,
        cheap_collision=args.cheap_collision, articulated=args.articulated,
        camera_vfov_deg=args.camera_vfov, manipulation_cam=args.manipulation_cam,
        manipulation_look_at=manip_look_at, render_subframes=args.render_subframes,
        manipulation_reach=args.manipulation_reach, manipulation_reach_arm=args.manipulation_reach_arm,
        fill_light_intensity=args.fill_light_intensity,
        physics_articulation_drive=args.physics_articulation_drive,
        dynamic_standing_contact_steps=args.dynamic_standing_contact_steps,
        neutral_environment=args.neutral_environment,
        kinematic_arm_pose=args.kinematic_arm_pose,
        collision_approximation=args.collision_approximation, verify_cam=args.verify_cam,
        manipulation_stand=args.manipulation_stand,
        software_denoise=not args.no_software_denoise,
        serve=args.serve,
        serve_dir=(Path(args.serve_dir) if args.serve_dir else None),
        serve_idle_timeout_s=args.serve_idle_timeout, serve_max_jobs=args.serve_max_jobs)
    upload_zip(out_dir, put_url)
    print(json.dumps({"status": result["status"], "passed": result["scenarios_passed"],
                      "executed": result["scenarios_executed"]}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
