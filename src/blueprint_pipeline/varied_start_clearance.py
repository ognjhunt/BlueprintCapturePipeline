"""A varied try of a rigid pick/place task must never start inside the area it is scored on.

Beta journey step: L5 team policy evaluation and the website screening. A scored cell moves the object's
start on the table (``team_evaluation_plan.variation_bounds``: at most a quarter of the start-to-goal
distance, and 3 cm, per axis). The scorer (``adp_rigid_task_scoring``) counts a success when the object
ends inside ``destination_position_bounds_world_m`` after a lift of ``minimum_lift_m`` and a planar
translation of ``minimum_translation_m`` from the try's own first sample. A start inside that area, or at
its edge, would let a small nudge score as a placement, and the larger areas a sentence like "the other end
of the table" seals make that more likely. So every varied start lies at least ``START_CLEARANCE_M`` (in
the table plane) outside the scored area:

- the cell draw re-draws an offending start shift, deterministically, and never drops the try
  (``clear_start_parameters``);
- ``uncleared_cells`` is the $0 plan-time check over every varied cell, not only the nominal one.

A task whose own nominal start is that close to its area is not refused: when the task is sealed, its
scored area is shrunk about its target until the start is clear by the margin, never below the 5 cm
disc (``shrink_area_off_start``), and the plan carries ``scene_estimate:destination_near_start`` if the
start is still not clear. The destination work upstream (``website_destination_area``) already keeps
the area off the start footprint; this is the guard if it ever does not.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

START_CLEARANCE_M = 0.02
REDRAWS = 16
START_PARAMETERS = ("object_start_x_delta_m", "object_start_y_delta_m")
#: The smallest scored area: the 5 cm disc (``website_surface_target_admission.WEBSITE_TARGET_TOLERANCE_M``).
MINIMUM_HALF_WIDTH_M = 0.05
DESTINATION_NEAR_START = "scene_estimate:destination_near_start"
INSIDE_SCORED_AREA = "varied_start_inside_scored_area"


def _area(task_spec: Mapping[str, Any]) -> tuple[list[float], list[float], list[float]] | None:
    start = task_spec.get("start_pose_world")
    box = task_spec.get("destination_position_bounds_world_m")
    if not isinstance(start, Sequence) or len(start) < 2 or not isinstance(box, Mapping):
        return None
    low, high = box.get("minimum"), box.get("maximum")
    if not isinstance(low, Sequence) or not isinstance(high, Sequence) or len(low) < 2 or len(high) < 2:
        return None
    try:
        return [float(v) for v in start[:2]], [float(v) for v in low[:2]], [float(v) for v in high[:2]]
    except (TypeError, ValueError):
        return None


def start_clearance_m(task_spec: Mapping[str, Any], parameters: Mapping[str, Any]) -> float | None:
    """The varied start's distance, in the table plane, outside the scored area (0 inside); None if unknown."""
    area = _area(task_spec)
    if area is None:
        return None
    start, low, high = area
    try:
        point = [start[axis] + float(parameters.get(name, 0.0)) for axis, name in enumerate(START_PARAMETERS)]
    except (AttributeError, TypeError, ValueError):
        return None  # a shift that is not a number: not assessed, never a stop
    distance = math.hypot(*(max(low[axis] - point[axis], 0.0, point[axis] - high[axis]) for axis in range(2)))
    return distance if math.isfinite(distance) else None


def clear(task_spec: Mapping[str, Any], parameters: Mapping[str, Any]) -> bool:
    clearance = start_clearance_m(task_spec, parameters)
    return clearance is None or clearance >= START_CLEARANCE_M


def _unit(*parts: Any) -> float:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).hexdigest()
    return (int(digest[:15], 16) + 1) / (16**15 + 1)


def clear_start_parameters(task_spec: Mapping[str, Any], parameters: Mapping[str, Any],
                           bounds: Mapping[str, tuple[float, float]], *, key: Sequence[Any]) -> dict[str, Any]:
    """``parameters`` with its start shift re-drawn inside ``bounds`` until the start is clear; never dropped.

    Deterministic in ``key``. Up to ``REDRAWS`` uniform draws; then the bounds' corner (or no shift) that
    lies farthest outside the area, which is clear whenever any start within the bounds is.
    """
    try:
        return _clear_start_parameters(task_spec, parameters, bounds, key=key)
    except (ArithmeticError, KeyError, IndexError, TypeError, ValueError):
        return dict(parameters)  # bounds it cannot read: the draw stands as drawn, never a stop


def _clear_start_parameters(task_spec: Mapping[str, Any], parameters: Mapping[str, Any],
                            bounds: Mapping[str, tuple[float, float]], *, key: Sequence[Any]) -> dict[str, Any]:
    shifted = [name for name in START_PARAMETERS if name in parameters and name in bounds]
    if not shifted or clear(task_spec, parameters):
        return dict(parameters)
    if not all(math.isfinite(float(value)) for name in shifted for value in bounds[name]):
        raise ValueError("varied_start_bounds_not_finite")
    for attempt in range(REDRAWS):
        value = dict(parameters)
        for name in shifted:
            low, high = bounds[name]
            value[name] = round(low + _unit(*key, "redraw", attempt, name) * (high - low), 4)
        if clear(task_spec, value):
            return value
    options = [dict(parameters, **{name: 0.0 for name in shifted})]
    for corner in ((0, 0), (0, 1), (1, 0), (1, 1)):
        options.append(dict(parameters, **{name: round(bounds[name][side], 4) for name, side in zip(shifted, corner)}))
    return max(options, key=lambda value: start_clearance_m(task_spec, value) or 0.0)


def shrink_area_off_start(task_spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``task_spec`` with its scored area shrunk about its target until the nominal start is clear.

    Returns the spec and a record of the change (None when nothing changed). Never below
    ``MINIMUM_HALF_WIDTH_M`` per axis; the record says whether the start is clear after it.
    """
    area, target = _area(task_spec), task_spec.get("target_position_world_m")
    if area is None or not isinstance(target, Sequence) or len(target) < 2 or clear(task_spec, {}):
        return task_spec, None
    start, low, high = area
    centre = [float(target[0]), float(target[1])]
    half = [min(centre[axis] - low[axis], high[axis] - centre[axis]) for axis in range(2)]
    if min(half) < 0:
        # The target lies outside its own area: no box about it is a sub-area. Left exactly as sealed, flagged.
        return task_spec, {"area_shrunk_off_start": False, "reason": "target_outside_area", "half_width_m_before": half,
                           "half_width_m": half, "start_clearance_m": start_clearance_m(task_spec, {}),
                           "start_clear": False, "flag": DESTINATION_NEAR_START}
    steps = 200
    for step in range(steps + 1):
        scale = 1.0 - step / steps
        # Only ever smaller, never below the 5 cm disc; an area already smaller than it is kept as it is.
        trial = [min(half[axis], max(MINIMUM_HALF_WIDTH_M, half[axis] * scale)) for axis in range(2)]
        box = {"minimum": [centre[0] - trial[0], centre[1] - trial[1], *task_spec["destination_position_bounds_world_m"]["minimum"][2:]],
               "maximum": [centre[0] + trial[0], centre[1] + trial[1], *task_spec["destination_position_bounds_world_m"]["maximum"][2:]]}
        candidate = {**task_spec, "destination_position_bounds_world_m": box}
        if clear(candidate, {}):
            break
    if not clear(candidate, {}) and all(trial[axis] >= half[axis] for axis in range(2)):
        candidate = task_spec  # nothing could be taken off: the sealed area stands exactly as it was
    # A box that clears is kept whatever step found it (the first is the box about the target itself). The
    # record says what happened: shrunk, or left exactly as sealed because nothing could be taken off.
    shrunk = candidate is not task_spec
    record = {"area_shrunk_off_start": shrunk, "half_width_m_before": half, "half_width_m": trial if shrunk else half,
              "start_clearance_m": start_clearance_m(candidate, {}), "start_clear": clear(candidate, {}),
              "flag": DESTINATION_NEAR_START, **({} if shrunk else {"reason": "area_already_at_minimum"})}
    return candidate, record


def uncleared_cells(task_spec: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> list[str]:
    """The $0 plan-time check: the ids of cells whose varied (or nominal) start is not clear of the area.

    A cell it cannot read is not assessed (left out), never a stop.
    """
    uncleared = []
    for cell in cells:
        try:
            parameters = (cell.get("resolved_scenario") or {}).get("parameters") or {}
            if not clear(task_spec, parameters):
                uncleared.append(str(cell.get("cell_id")))
        except (AttributeError, TypeError, ValueError):
            continue
    return uncleared


__all__ = ["DESTINATION_NEAR_START", "INSIDE_SCORED_AREA", "MINIMUM_HALF_WIDTH_M", "START_CLEARANCE_M", "clear",
           "clear_start_parameters", "shrink_area_off_start", "start_clearance_m", "uncleared_cells"]
