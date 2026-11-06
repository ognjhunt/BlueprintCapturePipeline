"""One scorer for any task: typed success predicates over measured simulator state.

A compiled task spec (``task_spec_compiler``) lists what success means as data:
a subject reaching a region and holding there, staying clear of named geometry,
a joint inside an interval, a height that stays level, a time limit. This module
evaluates that list against a retained state trace and nothing else. It never
runs a policy, reads a policy's reward or self-report, or claims a physical
outcome. Adding a task kind means writing its predicates, not a new scorer.

Missing readback is never success: a predicate whose required channel is absent
from any sample it needs is ``unscored``, and so is the episode.

Beside the verdict, every report carries what the robot actually did
(``behaviour``), read from the same trace: for a mobile robot its path, its
closest approach to each goal and its minimum clearance to obstacles; for a
gripper each real close (width falling below a threshold), where it happened and
how near the subject it was. Operators and the website read those instead of
inferring behaviour from criterion booleans.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest

SPEC_SCHEMA = "blueprint.task_predicates.v1"
#: A whole task spec compiled from a brief (``task_spec_compiler``); its predicates are under ``success``.
COMPILED_TASK_SPEC_SCHEMA = "blueprint.compiled_task_spec.v1"
REPORT_SCHEMA = "blueprint.task_predicate_score.v1"
BEHAVIOUR_SCHEMA = "blueprint.episode_behaviour.v1"

#: Each kind and the fields its definition must carry. Readback channels are per sample.
PREDICATE_KINDS: dict[str, frozenset[str]] = {
    # A subject's position inside a region, held for ``hold_samples`` consecutive samples.
    # ``when``: "terminal" (the last samples) or "any" (any window). Optional
    # ``max_height_drift_m`` bounds the subject's height change from its reset inside the window.
    "region_reached": frozenset({"id", "kind", "subject", "region", "hold_samples", "when"}),
    # The subject's height stays within ``max_drift_m`` of its reset height at every sample.
    "height_stable": frozenset({"id", "kind", "subject", "max_drift_m"}),
    # The robot touches none of the named geometry: no sample reports contact with it,
    # and, when ``minimum_clearance_m`` is set, every sample's measured clearance meets it.
    "no_contact": frozenset({"id", "kind", "geometry", "contact_force_threshold_n"}),
    # A joint's position inside ``interval``, held like ``region_reached``.
    "joint_in_interval": frozenset({"id", "kind", "joint", "interval", "hold_samples", "when"}),
    # The episode ends within ``max_steps`` samples (the runner's cut-off is not a success).
    "time_limit": frozenset({"id", "kind", "max_steps"}),
}
_OPTIONAL = {
    "region_reached": frozenset({"max_height_drift_m", "require_start_outside", "label"}),
    "height_stable": frozenset({"label"}),
    "no_contact": frozenset({"minimum_clearance_m", "label"}),
    "joint_in_interval": frozenset({"label"}),
    "time_limit": frozenset({"label"}),
}
REGION_SHAPES = ("disc", "box")
DEFAULT_GRIPPER_CLOSE_WIDTH_M = 0.04


class TaskPredicateError(ValueError):
    """A malformed predicate set or an unbound trace cannot be scored."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise TaskPredicateError(code)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _vector(value: Any, size: int) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        return None
    items = [_finite(item) for item in value]
    return None if any(item is None for item in items) else items  # type: ignore[return-value]


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TaskPredicateError("task_predicates_not_finite_json") from exc


def _validate_subject(subject: Any, code: str) -> dict[str, Any]:
    _require(isinstance(subject, Mapping), code)
    kind = subject.get("kind")
    if kind == "robot_root":
        _require(set(subject) == {"kind"}, code)
    elif kind == "object":
        _require(set(subject) == {"kind", "object_id"} and isinstance(subject.get("object_id"), str)
                 and bool(subject["object_id"]), code)
    else:
        raise TaskPredicateError(code)
    return dict(subject)


def validate_region(region: Any) -> dict[str, Any]:
    _require(isinstance(region, Mapping), "task_predicate_region_invalid")
    shape = region.get("shape")
    center = _vector(region.get("center_world_m"), 3)
    _require(shape in REGION_SHAPES and center is not None, "task_predicate_region_invalid")
    if shape == "disc":
        radius = _finite(region.get("radius_m"))
        _require(set(region) == {"shape", "center_world_m", "radius_m"} and radius is not None
                 and 0.0 < radius <= 50.0, "task_predicate_region_invalid")
    else:
        half = _vector(region.get("half_extents_m"), 3)
        yaw = _finite(region.get("yaw_rad", 0.0))
        _require(set(region) <= {"shape", "center_world_m", "half_extents_m", "yaw_rad"} and half is not None
                 and all(0.0 < item <= 50.0 for item in half) and yaw is not None,
                 "task_predicate_region_invalid")
    return _clone(region)


def _hold(value: Any) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 10_000,
             "task_predicate_hold_samples_invalid")
    return value


def validate_predicates(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical predicate set; refuse anything the scorer cannot read exactly."""
    spec = _clone(value)
    _require(isinstance(spec, dict) and spec.get("schema_version") == SPEC_SCHEMA,
             "task_predicates_schema_invalid")
    _require(set(spec) == {"schema_version", "predicates", "predicates_digest"}, "task_predicates_fields_invalid")
    rows = spec["predicates"]
    _require(isinstance(rows, list) and bool(rows), "task_predicates_empty")
    ids = [row.get("id") if isinstance(row, Mapping) else None for row in rows]
    _require(all(isinstance(key, str) and key for key in ids) and len(set(ids)) == len(ids),
             "task_predicate_ids_invalid")
    for row in rows:
        kind = row.get("kind")
        _require(kind in PREDICATE_KINDS, "task_predicate_kind_unknown")
        _require(PREDICATE_KINDS[kind] <= set(row) <= PREDICATE_KINDS[kind] | _OPTIONAL[kind],
                 f"task_predicate_fields_invalid:{row['id']}")
        if kind == "region_reached":
            _validate_subject(row["subject"], f"task_predicate_subject_invalid:{row['id']}")
            validate_region(row["region"])
            _hold(row["hold_samples"])
            _require(row["when"] in ("terminal", "any"), f"task_predicate_when_invalid:{row['id']}")
            if "max_height_drift_m" in row:
                drift = _finite(row["max_height_drift_m"])
                _require(drift is not None and 0.0 < drift <= 5.0, f"task_predicate_drift_invalid:{row['id']}")
            _require(type(row.get("require_start_outside", False)) is bool, f"task_predicate_fields_invalid:{row['id']}")
        elif kind == "height_stable":
            _validate_subject(row["subject"], f"task_predicate_subject_invalid:{row['id']}")
            drift = _finite(row["max_drift_m"])
            _require(drift is not None and 0.0 < drift <= 5.0, f"task_predicate_drift_invalid:{row['id']}")
        elif kind == "no_contact":
            geometry = row["geometry"]
            _require(geometry == "scene_obstacles" or (
                isinstance(geometry, list) and bool(geometry)
                and all(isinstance(item, str) and item for item in geometry)
                and len(set(geometry)) == len(geometry)), f"task_predicate_geometry_invalid:{row['id']}")
            threshold = _finite(row["contact_force_threshold_n"])
            _require(threshold is not None and threshold > 0.0, f"task_predicate_threshold_invalid:{row['id']}")
            if "minimum_clearance_m" in row:
                clearance = _finite(row["minimum_clearance_m"])
                _require(clearance is not None and 0.0 <= clearance <= 5.0,
                         f"task_predicate_clearance_invalid:{row['id']}")
        elif kind == "joint_in_interval":
            interval = _vector(row["interval"], 2)
            _require(isinstance(row["joint"], str) and bool(row["joint"]) and interval is not None
                     and interval[0] <= interval[1], f"task_predicate_interval_invalid:{row['id']}")
            _hold(row["hold_samples"])
            _require(row["when"] in ("terminal", "any"), f"task_predicate_when_invalid:{row['id']}")
        else:
            steps = row["max_steps"]
            _require(isinstance(steps, int) and not isinstance(steps, bool) and steps >= 1,
                     f"task_predicate_max_steps_invalid:{row['id']}")
    _require(spec["predicates_digest"] == canonical_digest(spec, digest_field="predicates_digest"),
             "task_predicates_digest_invalid")
    return spec


def seal_predicates(predicates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    spec = {"schema_version": SPEC_SCHEMA, "predicates": _clone(list(predicates))}
    spec["predicates_digest"] = canonical_digest(spec, digest_field="predicates_digest")
    return validate_predicates(spec)


def _yaw_rotate(x: float, y: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x + s * y, -s * x + c * y


def region_distance(region: Mapping[str, Any], position: Sequence[float]) -> float:
    """Horizontal distance from a point to a region (0 inside); boxes also bound height."""
    center = region["center_world_m"]
    if region["shape"] == "disc":
        return max(0.0, math.dist(position[:2], center[:2]) - region["radius_m"])
    half = region["half_extents_m"]
    local = _yaw_rotate(position[0] - center[0], position[1] - center[1], region.get("yaw_rad", 0.0))
    dx = max(0.0, abs(local[0]) - half[0])
    dy = max(0.0, abs(local[1]) - half[1])
    dz = max(0.0, abs(position[2] - center[2]) - half[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _subject_position(sample: Mapping[str, Any], subject: Mapping[str, Any]) -> list[float] | None:
    if subject["kind"] == "robot_root":
        value = sample.get("robot_root_position_world_m", sample.get("root_position_world_m"))
        return _vector(value, 3)
    poses = sample.get("object_poses_world")
    pose = poses.get(subject["object_id"]) if isinstance(poses, Mapping) else None
    if pose is None and sample.get("task_object_id") == subject["object_id"]:
        pose = sample.get("task_object_pose_world")
    vector = _vector(pose, 7) or _vector(pose, 3)
    return None if vector is None else vector[:3]


def _normalize_trace(samples: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    _require(isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)) and len(samples) >= 2,
             "task_predicate_samples_missing")
    for index, sample in enumerate(samples):
        _require(isinstance(sample, Mapping) and sample.get("step_index") == index,
                 f"task_predicate_sample_step_invalid:{index}")
    return list(samples)


def _windows(flags: list[bool], hold: int, when: str) -> tuple[bool, int | None]:
    first = next((index for index in range(hold - 1, len(flags)) if all(flags[index - hold + 1:index + 1])), None)
    if when == "terminal":
        return len(flags) >= hold and all(flags[-hold:]), first
    return first is not None, first


def _region_reached(row: Mapping[str, Any], trace: list[Mapping[str, Any]]) -> dict[str, Any]:
    positions = [_subject_position(sample, row["subject"]) for sample in trace]
    if any(position is None for position in positions):
        return {"status": "unscored", "missing_readback": "subject_position"}
    distances = [region_distance(row["region"], position) for position in positions]  # type: ignore[arg-type]
    drift_limit = row.get("max_height_drift_m")
    drifts = [abs(position[2] - positions[0][2]) for position in positions]  # type: ignore[index]
    inside = [distance == 0.0 and (drift_limit is None or drift <= drift_limit)
              for distance, drift in zip(distances, drifts, strict=True)]
    if row.get("require_start_outside") and distances[0] == 0.0:
        return {"status": "unscored", "invalid_start": "subject_starts_inside_region",
                "initial_distance_m": 0.0}
    satisfied, first = _windows(inside, row["hold_samples"], row["when"])
    return {"status": "scored", "satisfied": satisfied, "first_held_step": first,
            "initial_distance_m": distances[0], "minimum_distance_m": min(distances),
            "terminal_distance_m": distances[-1],
            **({"maximum_height_drift_m": max(drifts)} if drift_limit is not None else {})}


def _height_stable(row: Mapping[str, Any], trace: list[Mapping[str, Any]]) -> dict[str, Any]:
    positions = [_subject_position(sample, row["subject"]) for sample in trace]
    if any(position is None for position in positions):
        return {"status": "unscored", "missing_readback": "subject_position"}
    drift = max(abs(position[2] - positions[0][2]) for position in positions)  # type: ignore[index]
    return {"status": "scored", "satisfied": drift <= row["max_drift_m"], "maximum_height_drift_m": drift}


def _contact_with(sample: Mapping[str, Any], geometry: Any, threshold: float) -> bool | None:
    """True/False when the sample says whether the robot touched that geometry; None when it cannot."""
    if geometry == "scene_obstacles":
        force = _finite(sample.get("robot_scene_contact_peak_force_n"))
        if force is None and isinstance(sample.get("native_readback"), Mapping):
            force = _finite(sample["native_readback"].get("robot_scene_contact_peak_force_n"))
        return None if force is None else force >= threshold
    touched = sample.get("robot_contact_geometry_ids")
    if not isinstance(touched, list):
        return None
    forces = sample.get("robot_contact_peak_force_n_by_geometry")
    hits = {item for item in touched if isinstance(item, str)} & set(geometry)
    if isinstance(forces, Mapping):
        # A reported touch without its force cannot be compared with the threshold: unscored, never "clear".
        measured = [_finite(forces.get(item)) for item in sorted(hits)]
        if any(force is None for force in measured):
            return None
        return any(force >= threshold for force in measured)  # type: ignore[operator]
    return bool(hits)


def _no_contact(row: Mapping[str, Any], trace: list[Mapping[str, Any]]) -> dict[str, Any]:
    contacts = [_contact_with(sample, row["geometry"], row["contact_force_threshold_n"]) for sample in trace]
    if any(item is None for item in contacts):
        return {"status": "unscored", "missing_readback": "robot_contact"}
    result: dict[str, Any] = {"contact_steps": [index for index, hit in enumerate(contacts) if hit]}
    clear = not result["contact_steps"]
    if "minimum_clearance_m" in row:
        clearances = [_finite(sample.get("robot_obstacle_clearance_m")) for sample in trace]
        if any(item is None for item in clearances):
            return {"status": "unscored", "missing_readback": "robot_obstacle_clearance"}
        result["minimum_clearance_observed_m"] = min(clearances)  # type: ignore[type-var]
        clear = clear and result["minimum_clearance_observed_m"] >= row["minimum_clearance_m"]
    return {"status": "scored", "satisfied": clear, **result}


def _joint_in_interval(row: Mapping[str, Any], trace: list[Mapping[str, Any]]) -> dict[str, Any]:
    values = []
    for sample in trace:
        joints = sample.get("joint_positions")
        values.append(_finite(joints.get(row["joint"])) if isinstance(joints, Mapping) else None)
    if any(value is None for value in values):
        return {"status": "unscored", "missing_readback": "joint_position"}
    low, high = row["interval"]
    inside = [low <= value <= high for value in values]  # type: ignore[operator]
    satisfied, first = _windows(inside, row["hold_samples"], row["when"])
    return {"status": "scored", "satisfied": satisfied, "first_held_step": first,
            "initial_position": values[0], "terminal_position": values[-1]}


def _time_limit(row: Mapping[str, Any], trace: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {"status": "scored", "satisfied": len(trace) <= row["max_steps"], "sample_count": len(trace)}


_EVALUATORS = {"region_reached": _region_reached, "height_stable": _height_stable,
               "no_contact": _no_contact, "joint_in_interval": _joint_in_interval, "time_limit": _time_limit}


def score_episode(*, predicates: Mapping[str, Any], samples: Sequence[Mapping[str, Any]],
                  behaviour_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate every predicate on one trace; success only when all are scored and satisfied."""
    spec = validate_predicates(predicates)
    trace = _normalize_trace(samples)
    rows = []
    for row in spec["predicates"]:
        result = _EVALUATORS[row["kind"]](row, trace)
        rows.append({"id": row["id"], "kind": row["kind"], **result})
    unscored = [row["id"] for row in rows if row["status"] != "scored"]
    failed = [row["id"] for row in rows if row["status"] == "scored" and not row["satisfied"]]
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "unscored" if unscored else "scored",
        "outcome": None if unscored else ("success" if not failed else "failure"),
        "predicates_digest": spec["predicates_digest"],
        "predicate_results": rows,
        "failed_predicates": failed,
        "unscored_predicates": unscored,
        "sample_count": len(trace),
        "behaviour": episode_behaviour(samples=trace, predicates=spec, context=behaviour_context),
        "measurement_authority": "retained_native_simulator_state_trace",
        "policy_self_report_consulted": False,
        "evidence_level": "development_only",
        "physical_outcome_claimed": False,
    }
    report["score_digest"] = canonical_digest(report, digest_field="score_digest")
    return report


def _box_distance(point: Sequence[float], pose: Sequence[float], minimum: Sequence[float],
                  maximum: Sequence[float]) -> float:
    """Distance from a world point to a box given in its pose's own frame (0 inside)."""
    local = [point[index] - pose[index] for index in range(3)]
    x, y, z, w = pose[3:7]
    # Rotate by the inverse quaternion.
    qx, qy, qz = -x, -y, -z
    tx, ty, tz = (2 * (qy * local[2] - qz * local[1]), 2 * (qz * local[0] - qx * local[2]),
                  2 * (qx * local[1] - qy * local[0]))
    local = [local[0] + w * tx + (qy * tz - qz * ty), local[1] + w * ty + (qz * tx - qx * tz),
             local[2] + w * tz + (qx * ty - qy * tx)]
    excess = [max(0.0, minimum[index] - local[index], local[index] - maximum[index]) for index in range(3)]
    return math.sqrt(sum(item * item for item in excess))


def _subject_box(context: Mapping[str, Any]) -> tuple[list[float], list[float]] | None:
    bounds = context.get("subject_bounds_local_m")
    if isinstance(bounds, Mapping):
        low, high = _vector(bounds.get("minimum"), 3), _vector(bounds.get("maximum"), 3)
        if low is not None and high is not None and all(a <= b for a, b in zip(low, high, strict=True)):
            return low, high
        return None
    half = _vector(context.get("subject_half_extents_m"), 3)
    return ([-item for item in half], list(half)) if half is not None else None


def episode_behaviour(*, samples: Sequence[Mapping[str, Any]], predicates: Mapping[str, Any] | None = None,
                      context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What the robot did, measured from the trace; never a success claim.

    ``context`` may carry the manipulated object's box in its own pose frame
    (``subject_bounds_local_m`` as ``minimum``/``maximum``, or symmetric
    ``subject_half_extents_m``), the per-sample pose field that frame belongs to
    (``subject_pose_field``, default ``task_object_pose_world``) and
    ``gripper_close_width_m``. A channel that is absent from every sample is left
    out rather than guessed.
    """
    context = dict(context or {})
    result: dict[str, Any] = {"schema_version": BEHAVIOUR_SCHEMA}
    roots = [_vector(sample.get("robot_root_position_world_m", sample.get("root_position_world_m")), 3)
             for sample in samples]
    if all(root is not None for root in roots):
        path = sum(math.dist(a[:2], b[:2]) for a, b in pairwise(roots))  # type: ignore[index]
        navigation: dict[str, Any] = {"path_length_m": path, "start_world_m": roots[0], "end_world_m": roots[-1],
                                      "net_displacement_m": math.dist(roots[0][:2], roots[-1][:2])}  # type: ignore[index]
        goals = []
        for row in (predicates or {}).get("predicates", []):
            if row["kind"] == "region_reached" and row["subject"]["kind"] == "robot_root":
                distances = [region_distance(row["region"], root) for root in roots]  # type: ignore[arg-type]
                closest = min(range(len(distances)), key=distances.__getitem__)
                goals.append({"predicate_id": row["id"], "initial_distance_m": distances[0],
                              "closest_distance_m": distances[closest], "closest_step": closest,
                              "terminal_distance_m": distances[-1]})
        if goals:
            navigation["goals"] = goals
        clearances = [_finite(sample.get("robot_obstacle_clearance_m")) for sample in samples]
        if all(item is not None for item in clearances):
            low = min(range(len(clearances)), key=lambda index: clearances[index])  # type: ignore[arg-type,return-value]
            navigation["minimum_obstacle_clearance_m"] = clearances[low]
            navigation["minimum_obstacle_clearance_step"] = low
        forces = [_contact_with(sample, "scene_obstacles", context.get("contact_force_threshold_n", 1.0))
                  for sample in samples]
        if all(item is not None for item in forces):
            navigation["obstacle_contact_steps"] = [index for index, hit in enumerate(forces) if hit]
        result["navigation"] = navigation
    widths = [_finite(sample.get("gripper_width_m")) for sample in samples]
    if any(width is not None for width in widths):
        threshold = float(context.get("gripper_close_width_m", DEFAULT_GRIPPER_CLOSE_WIDTH_M))
        box = _subject_box(context)
        pose_field = str(context.get("subject_pose_field") or "task_object_pose_world")
        subject_id = context.get("subject_object_id")
        closes, approach = [], None
        previous_open = True
        for index, (sample, width) in enumerate(zip(samples, widths, strict=True)):
            grasp = _vector(sample.get("grasp_frame_position_world_m"), 3)
            pose = None
            if box is not None:
                poses = sample.get("object_poses_world")
                pose = (_vector(poses.get(subject_id), 7) if isinstance(poses, Mapping) and subject_id else None) \
                    or _vector(sample.get(pose_field), 7)
            distance = (_box_distance(grasp, pose, *box) if grasp is not None and pose is not None
                        and box is not None else None)
            if distance is not None and (approach is None or distance < approach["distance_m"]):
                approach = {"distance_m": distance, "step": index}
            if width is None:
                continue
            closed = width < threshold
            if closed and previous_open:
                closes.append({"step": index, "width_m": width, "grasp_frame_world_m": grasp,
                               **({"distance_to_subject_bounds_m": distance} if distance is not None else {})})
            previous_open = not closed
        measured = [width for width in widths if width is not None]
        result["gripper"] = {"close_width_threshold_m": threshold, "close_events": closes,
                             "ever_closed": bool(closes), "minimum_width_m": min(measured),
                             "width_readback_gaps": sum(width is None for width in widths)}
        if approach is not None:
            result["gripper"]["closest_approach_to_subject_bounds"] = approach
    return result


def g1_navigation_goal_predicates(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    """The published G1 navigation criterion as predicates; it scores the same as its own scorer.

    ``native_g1_navigation_goal.score_g1_navigation_episode`` is this set: the root inside
    the goal disc with its height near reset for the settle window at the end, and its
    height near reset throughout. Obstacle contact is outside that criterion, so it is
    outside this set; a compiled navigation spec adds ``no_contact``.
    """
    from .native_g1_navigation_goal import validate_g1_navigation_goal

    goal = validate_g1_navigation_goal(task_spec)
    region = {"shape": "disc", "center_world_m": goal["center_world_m"], "radius_m": goal["acceptance_radius_m"]}
    return seal_predicates([
        {"id": "goal", "kind": "region_reached", "subject": {"kind": "robot_root"}, "region": region,
         "hold_samples": goal["settle_window_samples"], "when": "terminal",
         "max_height_drift_m": goal["max_root_height_drift_m"], "require_start_outside": True},
        {"id": "upright", "kind": "height_stable", "subject": {"kind": "robot_root"},
         "max_drift_m": goal["max_root_height_drift_m"]},
    ])


__all__ = [
    "BEHAVIOUR_SCHEMA", "PREDICATE_KINDS", "REPORT_SCHEMA", "SPEC_SCHEMA", "TaskPredicateError",
    "episode_behaviour", "g1_navigation_goal_predicates", "region_distance", "score_episode", "seal_predicates", "validate_predicates",
    "validate_region",
]
