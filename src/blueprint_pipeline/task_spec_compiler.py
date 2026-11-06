"""Compile a confirmed task brief and a reconstructed scene into one task spec.

The spec is the single description every later step reads: what the robot must
be able to do (``requirements``: locomote, manipulate, reach, payload), where it
starts, the success predicates over simulator state (``task_predicates``), the
markers a policy can see, and the instruction each policy is given. Robot
choice, scene placement, the per-site robot packet and scoring all derive from
it, so a new kind of job needs a new spec, not new routing code.

Nothing is invented. Every position comes from the scene facts passed in, and
each one records its source. A value the scene does not establish is the
compiler's best supported guess, listed under ``flags`` so the job keeps going
and the guess stays visible:

- a goal taken from the filmed path's end instead of a reviewed anchor;
- a tolerance widened because metric scale is not validated (the scale
  evidence is carried through);
- a goal that would cover the subject's start, moved off it.

Markers are sized to the goal tolerance, never to the object, and a goal region
never overlaps where its subject starts.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_predicates import region_distance, seal_predicates, validate_predicates

SCHEMA = "blueprint.compiled_task_spec.v1"
CLAIM_CEILING = "development_only"

#: The scene's metric-scale reading, as the placement and reconstruction steps record it.
SCALE_STATUSES = ("validated", "provider_declared_not_independently_validated", "unverified")
#: Relative scale error assumed when scale is not validated: a 10% error moves a goal
#: 10% of its distance, so the goal tolerance grows by that much (flagged, never hidden).
UNVALIDATED_SCALE_FRACTION = 0.10
#: Bounds on a locomotion goal's acceptance radius. The lower bound is a robot's own
#: footprint; the upper keeps a goal a goal (and inside every published marker contract).
GOAL_RADIUS_MIN_M = 0.25
GOAL_RADIUS_MAX_M = 0.5
START_RADIUS_M = 0.25
#: A fall, not a crouch: a mobile robot's root may not drop or rise more than this.
ROOT_HEIGHT_DRIFT_M = 0.25
CONTACT_FORCE_THRESHOLD_N = 1.0
#: Samples the subject must stay in its goal at the end, at the policy's control rate.
DEFAULT_HOLD_SAMPLES = 10
MARKER_COLOURS = ("green", "yellow")

_RELATIONS = frozenset({"on", "in", "into", "inside", "onto", "at", "from", "near", "under", "beside", "by",
                        "next", "with", "behind", "above", "below", "against", "over", "to", "for", "that",
                        "which", "and", "or"})


class TaskSpecCompileError(ValueError):
    """The brief or scene facts cannot produce a well-formed spec."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise TaskSpecCompileError(code)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _point(value: Any, code: str) -> list[float]:
    _require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3, code)
    items = [_finite(item) for item in value]
    _require(all(item is not None for item in items), code)
    return items  # type: ignore[return-value]


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TaskSpecCompileError("task_spec_input_not_finite_json") from exc


def requirements(*, locomote: bool, manipulate: bool, reach_m: float | None = None,
                 payload_kg: float | None = None) -> dict[str, Any]:
    """What a robot must be able to do for this job; robot choice reads only this."""
    _require(type(locomote) is bool and type(manipulate) is bool and (locomote or manipulate),
             "task_spec_requirements_invalid")
    for value in (reach_m, payload_kg):
        _require(value is None or (_finite(value) is not None and value > 0), "task_spec_requirements_invalid")
    return {"locomote": locomote, "manipulate": manipulate, "reach_m": reach_m, "payload_kg": payload_kg}


def object_label(label: Any) -> str | None:
    """A label's object noun phrase, without its location ("tray on the table" -> "tray")."""
    if not isinstance(label, str) or not label.strip():
        return None
    words = label.strip().split()
    end = next((index for index, word in enumerate(words) if word.lower().strip(",.;:") in _RELATIONS
                and index > 0), len(words))
    phrase = " ".join(words[:end]).strip(" ,.;:")
    phrase = re.sub(r"^(the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
    return phrase or None


def locomotion_instruction(*, colour: str, goal_label: str | None, obstacle_label: str | None) -> str:
    goal = f"the {colour} marked area" + (f" {goal_label.strip()}" if goal_label and goal_label.strip() else "")
    obstacles = obstacle_label.strip() if obstacle_label and obstacle_label.strip() else "anything"
    if obstacles != "anything" and not re.match(r"^(the|a|an)\s", obstacles, flags=re.IGNORECASE):
        obstacles = "the " + obstacles
    return f"Move to {goal} without touching {obstacles}."


def placement_instruction(*, subject_label: Any, destination_label: Any, relation: str, colour: str) -> str:
    """Name the subject once, the destination once, and the marked area the policy can see.

    The subject's own location never leaks into it ("tray on the table" is the tray), so
    a destination on the surface the subject already stands on is not named twice.
    """
    subject = object_label(subject_label)
    destination = object_label(destination_label)
    _require(subject is not None, "task_spec_subject_label_missing")
    _require(relation in ("on", "inside"), "task_spec_destination_relation_invalid")
    if relation == "inside":
        _require(destination is not None, "task_spec_destination_label_missing")
        return f"Pick up the {subject} and put it inside the {destination}."
    marked = f"the {colour} marked area" + (f" on the {destination}" if destination else "")
    return f"Pick up the {subject} and place it on {marked}."


def goal_radius(*, base_m: float, start: Sequence[float], goal: Sequence[float],
                scale_status: str) -> tuple[float, list[str]]:
    """The goal tolerance: a footprint, widened by the scale error a goal's distance can carry."""
    flags = []
    radius = max(GOAL_RADIUS_MIN_M, base_m)
    if scale_status != "validated":
        widened = radius + UNVALIDATED_SCALE_FRACTION * math.dist(start[:2], goal[:2])
        if widened > radius:
            flags.append("goal_tolerance_widened_for_unvalidated_scale")
        radius = widened
    if radius > GOAL_RADIUS_MAX_M:
        flags.append("goal_tolerance_capped")
        radius = GOAL_RADIUS_MAX_M
    return radius, flags


def _start_footprint_radius(footprint: Mapping[str, Any] | None) -> float:
    if not footprint:
        return 0.0
    half = footprint.get("half_extents_m")
    if isinstance(half, Sequence) and len(half) >= 2:
        return math.hypot(float(half[0]), float(half[1]))
    return float(footprint.get("radius_m") or 0.0)


def goal_clear_of_start(*, goal: Sequence[float], radius: float, start: Sequence[float],
                        start_footprint_m: float, direction_hint: Sequence[float] | None = None
                        ) -> tuple[list[float], bool]:
    """Move a goal region off its subject's start footprint along the start-to-goal line.

    Returns the (possibly moved) centre and whether it moved. A coincident goal moves
    along ``direction_hint`` (or +x), which is a guess the caller flags.
    """
    gap = radius + start_footprint_m
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance >= gap:
        return list(goal), False
    if distance < 1e-9:
        hx, hy = (direction_hint[0], direction_hint[1]) if direction_hint else (1.0, 0.0)
        norm = math.hypot(hx, hy) or 1.0
        dx, dy, distance = hx / norm, hy / norm, 1.0
    scale = (gap + 1e-3) / distance
    return [start[0] + dx * scale, start[1] + dy * scale, goal[2]], True


def _locomotion(brief: Mapping[str, Any], scene: Mapping[str, Any], spec: dict[str, Any], flags: list[str]
                ) -> list[dict[str, Any]]:
    anchors = scene.get("anchors") or {}
    floor = _finite(scene.get("floor_height_world_m"))
    _require(floor is not None, "task_spec_floor_height_missing")
    path = scene.get("filmed_path_world_m")
    filmed = [_point(row, "task_spec_filmed_path_invalid") for row in path] if path else []
    if anchors.get("start_world_m") is not None:
        start, start_source = _point(anchors["start_world_m"], "task_spec_start_invalid"), "reviewed_anchor"
    else:
        _require(len(filmed) >= 2, "task_spec_start_unresolved")
        start, start_source = filmed[0], "filmed_path_start"
        flags.append("start_from_filmed_path")
    if anchors.get("goal_world_m") is not None:
        goal, goal_source = _point(anchors["goal_world_m"], "task_spec_goal_invalid"), "reviewed_anchor"
    else:
        _require(len(filmed) >= 2, "task_spec_goal_unresolved")
        goal, goal_source = filmed[-1], "filmed_path_endpoint"
        flags.append("goal_from_filmed_path")
    start, goal = [start[0], start[1], floor], [goal[0], goal[1], floor]
    scale = spec["metric_scale"]["status"]
    radius, radius_flags = goal_radius(base_m=float(brief.get("goal_tolerance_m") or GOAL_RADIUS_MIN_M),
                                       start=start, goal=goal, scale_status=scale)
    flags.extend(radius_flags)
    heading = [goal[0] - start[0], goal[1] - start[1]]
    goal, moved = goal_clear_of_start(goal=goal, radius=radius, start=start, start_footprint_m=START_RADIUS_M,
                                      direction_hint=heading)
    if moved:
        flags.append("goal_moved_off_start")
    spec["start"] = {"subject": {"kind": "robot_root"},
                     "region": {"shape": "disc", "center_world_m": start, "radius_m": START_RADIUS_M},
                     "facing_world_m": goal, "source": start_source}
    goal_region = {"shape": "disc", "center_world_m": goal, "radius_m": radius}
    colour = brief.get("marker_colour") or "yellow"
    _require(colour in MARKER_COLOURS, "task_spec_marker_colour_invalid")
    spec["markers"] = [{"predicate_id": "goal", "shape": "flat_disc", "colour": colour,
                        "position_world_m": goal, "radius_m": radius, "non_colliding": True,
                        "policy_visible": True, "sized_from": "goal_tolerance"}]
    spec["goal_source"] = goal_source
    hold = int(brief.get("hold_samples") or DEFAULT_HOLD_SAMPLES)
    obstacles = scene.get("obstacle_geometry") or "scene_obstacles"
    rows = [
        {"id": "goal", "kind": "region_reached", "subject": {"kind": "robot_root"}, "region": goal_region,
         "hold_samples": hold, "when": "terminal", "max_height_drift_m": ROOT_HEIGHT_DRIFT_M,
         "require_start_outside": True, "label": brief.get("goal_label") or "goal"},
        {"id": "upright", "kind": "height_stable", "subject": {"kind": "robot_root"},
         "max_drift_m": ROOT_HEIGHT_DRIFT_M, "label": "did not fall"},
    ]
    if brief.get("avoid_contact", True):
        rows.append({"id": "no_obstacle_contact", "kind": "no_contact", "geometry": obstacles,
                     "contact_force_threshold_n": CONTACT_FORCE_THRESHOLD_N,
                     "label": brief.get("obstacle_label") or "obstacles"})
    spec["instruction"] = {"canonical": locomotion_instruction(colour=colour, goal_label=brief.get("goal_label"),
                                                               obstacle_label=brief.get("obstacle_label"))}
    return rows


def _placement(brief: Mapping[str, Any], scene: Mapping[str, Any], spec: dict[str, Any], flags: list[str]
               ) -> list[dict[str, Any]]:
    objects = {row.get("object_id"): row for row in scene.get("objects") or [] if isinstance(row, Mapping)}
    subject_id = brief.get("subject_object_id")
    subject = objects.get(subject_id)
    _require(isinstance(subject, Mapping), "task_spec_subject_object_missing")
    pose = subject.get("pose_world")
    _require(isinstance(pose, Sequence) and len(pose) == 7, "task_spec_subject_pose_invalid")
    start = _point(pose[:3], "task_spec_subject_pose_invalid")
    anchors = scene.get("anchors") or {}
    goal = _point(anchors.get("goal_world_m"), "task_spec_goal_unresolved")
    tolerance = _finite(brief.get("goal_tolerance_m"))
    _require(tolerance is not None and 0.01 <= tolerance <= 1.0, "task_spec_goal_tolerance_invalid")
    footprint = _start_footprint_radius(subject)
    goal, moved = goal_clear_of_start(goal=goal, radius=tolerance, start=start, start_footprint_m=footprint)
    if moved:
        flags.append("goal_moved_off_start")
    region = {"shape": "box", "center_world_m": goal,
              "half_extents_m": [tolerance, tolerance, max(0.02, float((subject.get("half_extents_m") or [0, 0, 0.05])[2]))]}
    colour = brief.get("marker_colour") or "green"
    _require(colour in MARKER_COLOURS, "task_spec_marker_colour_invalid")
    spec["start"] = {"subject": {"kind": "object", "object_id": subject_id},
                     "region": {"shape": "disc", "center_world_m": start, "radius_m": max(footprint, 0.01)},
                     "source": "registered_object_pose"}
    spec["markers"] = [{"predicate_id": "goal", "shape": "flat_disc", "colour": colour,
                        "position_world_m": goal, "radius_m": tolerance, "non_colliding": True,
                        "policy_visible": True, "sized_from": "goal_tolerance"}]
    spec["goal_source"] = "reviewed_anchor"
    spec["instruction"] = {"canonical": placement_instruction(
        subject_label=subject.get("label"), destination_label=brief.get("destination_label"),
        relation=brief.get("destination_relation") or "on", colour=colour)}
    hold = int(brief.get("hold_samples") or DEFAULT_HOLD_SAMPLES)
    return [{"id": "goal", "kind": "region_reached", "subject": {"kind": "object", "object_id": subject_id},
             "region": region, "hold_samples": hold, "when": "terminal", "require_start_outside": True,
             "label": brief.get("destination_label") or "destination"}]


def compile_task_spec(*, brief: Mapping[str, Any], scene: Mapping[str, Any],
                      task_requirements: Mapping[str, Any]) -> dict[str, Any]:
    """One task spec from the confirmed brief and the reconstructed scene's facts.

    ``brief``: ``site_id``, ``task_id``, ``sentence`` and the readings the brief
    supports (labels for the goal, obstacles, subject and destination; a tolerance).
    ``scene``: ``scene_revision_digest``, ``floor_height_world_m``, ``metric_scale``
    (``status`` and ``evidence``), ``anchors`` and, for locomotion, the filmed path.
    ``task_requirements``: from ``requirements``.
    """
    brief, scene = _clone(dict(brief)), _clone(dict(scene))
    need = requirements(**task_requirements)
    for key in ("site_id", "task_id", "sentence"):
        _require(isinstance(brief.get(key), str) and bool(brief[key].strip()), f"task_spec_{key}_missing")
    _require(isinstance(scene.get("scene_revision_digest"), str), "task_spec_scene_revision_missing")
    scale = scene.get("metric_scale") or {}
    _require(scale.get("status") in SCALE_STATUSES, "task_spec_metric_scale_status_invalid")
    flags: list[str] = []
    if scale["status"] != "validated":
        flags.append("metric_scale_" + scale["status"])
    spec: dict[str, Any] = {
        "schema_version": SCHEMA, "claim_ceiling": CLAIM_CEILING,
        "site_id": brief["site_id"], "task_id": brief["task_id"], "sentence": brief["sentence"].strip(),
        "scene_revision_digest": scene["scene_revision_digest"], "requirements": need,
        "metric_scale": {"status": scale["status"], "evidence": scale.get("evidence") or []},
    }
    if need["manipulate"]:
        rows = _placement(brief, scene, spec, flags)
        if need["locomote"]:
            flags.append("mobile_manipulation_scored_on_object_goal_only")
    else:
        rows = _locomotion(brief, scene, spec, flags)
    max_steps = brief.get("max_steps")
    if max_steps is not None:
        rows.append({"id": "time_limit", "kind": "time_limit", "max_steps": int(max_steps)})
    spec["success"] = seal_predicates(rows)
    spec["flags"] = sorted(set(flags))
    spec["spec_digest"] = canonical_digest(spec, digest_field="spec_digest")
    return validate_task_spec(spec)


def validate_task_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    spec = _clone(dict(value))
    _require(spec.get("schema_version") == SCHEMA and spec.get("claim_ceiling") == CLAIM_CEILING,
             "task_spec_schema_invalid")
    _require(set(spec) == {"schema_version", "claim_ceiling", "site_id", "task_id", "sentence",
                           "scene_revision_digest", "requirements", "metric_scale", "start", "markers",
                           "goal_source", "instruction", "success", "flags", "spec_digest"},
             "task_spec_fields_invalid")
    requirements(**spec["requirements"])
    validate_predicates(spec["success"])
    goals = {row["id"]: row for row in spec["success"]["predicates"] if row["kind"] == "region_reached"}
    start = spec["start"]["region"]["center_world_m"]
    for marker in spec["markers"]:
        goal = goals.get(marker["predicate_id"])
        _require(goal is not None, "task_spec_marker_predicate_missing")
        region = goal["region"]
        tolerance = region["radius_m"] if region["shape"] == "disc" else max(region["half_extents_m"][:2])
        _require(marker["sized_from"] == "goal_tolerance" and math.isclose(marker["radius_m"], tolerance)
                 and marker["position_world_m"] == region["center_world_m"], "task_spec_marker_not_goal_sized")
        _require(region_distance(region, start) > 0.0 or region_distance(
            {**region, "center_world_m": [*region["center_world_m"][:2], start[2]]}, start) > 0.0,
                 "task_spec_goal_covers_start")
    _require(spec["spec_digest"] == canonical_digest(spec, digest_field="spec_digest"), "task_spec_digest_invalid")
    return spec


__all__ = [
    "SCHEMA", "TaskSpecCompileError", "compile_task_spec", "goal_clear_of_start", "goal_radius",
    "locomotion_instruction", "object_label", "placement_instruction", "requirements", "validate_task_spec",
]
