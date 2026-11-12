"""Can each AI robot's hand hold the task object: the physics screen's grasp tests, once per robot row.

``website_task_preparation.screen_physics`` tests grasp width and grip force only for a gripper
it is given, and the website path gives none. So on capture-447ae4f4 a 31 x 52 cm tray went to
the DROID robots, whose Robotiq 2F-85 opens 85 mm, with no note. This module runs the same
test for every robot row a job plans, with that row's own gripper:

- a row whose robot was registered with a manifest uses that manifest's tools
  (``website_task_preparation.embodiment_grippers``: opening declared, grip force not);
- a row on a robot profile whose gripper the spec catalog records uses that figure
  (``PROFILE_TOOLS``: the DROID Franka's Robotiq 2F-85, 85 mm, from Robotiq's product sheet;
  no grip force is recorded);
- any other row is "not assessed": nothing is invented.

The result is per robot and is only ever a flag on that robot's result. It is never a stop and
never a reason to skip the robot's test: the owner's rule is that the test runs anyway. The
flags reach the site with the row's ``scene_check_flags`` (``website_candidate_feasibility.scene_check``):

- ``task_object_wider_than_gripper``: even the object's smallest side, plus the size estimate's
  error, is wider than the hand opens;
- ``task_object_heavier_than_grip``: the grip force, where one is recorded, cannot hold even the
  lightest weight estimate;
- ``task_object_grip_depends_on_weight``: the hold succeeds at the light end of the weight
  estimate and fails at the heavy end.

- ``task_object_needs_edge_grip``: the grasp the robot will use cannot close across the object,
  only on a thinner edge. Every grasp the carry check plans (``robot_arm_path_clearance``:
  top-down first, then tilted) closes its jaws horizontally, so the narrowest thing the hand can
  close across is the object's smallest horizontal extent at its resting pose. When that, plus
  the size estimate's error, is wider than the hand opens but the screen's own test (its smallest
  side, often its height) passes, the hand can only hold it by pinching an edge or rim. On
  capture-447ae4f4 (approach "top_down") the 31 x 52 x 4.5 cm tray is 31 cm across against an
  85 mm opening: flagged.

The screen's own test (the smallest side of any orientation) is kept unchanged beside it.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = "website_candidate_grasp_screen.v1"
FLAG_WIDER = "task_object_wider_than_gripper"
FLAG_HEAVIER = "task_object_heavier_than_grip"
FLAG_DEPENDS = "task_object_grip_depends_on_weight"
FLAG_EDGE = "task_object_needs_edge_grip"
FLAGS = frozenset({FLAG_WIDER, FLAG_HEAVIER, FLAG_DEPENDS, FLAG_EDGE})
#: Robot profiles whose gripper the spec catalog records, by the catalog's ``tool_id``. The DROID
#: profile names its gripper, and its arena robot is built with that Robotiq 2F-85.
PROFILE_TOOLS = {"droid_franka_panda_robotiq_2f85_v1": "robotiq_2f85"}


def _catalog_tool(tool_id: str) -> dict[str, Any] | None:
    from .robot_spec_catalog import CATALOG_PATH
    import json

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    tool = catalog.get("default_tool") or {}
    if tool.get("tool_id") != tool_id:
        return None
    stroke = (tool.get("figures") or {}).get("stroke_m") or {}
    if not isinstance(stroke.get("value"), (int, float)):
        return None
    return {"model": tool_id, "name": tool.get("name"), "full_stroke_m": float(stroke["value"]),
            "grip_force_n": None, "source": "robot_spec_catalog", "source_url": stroke.get("source_url")}


def candidate_grippers(candidate: Mapping[str, Any], settings: Any) -> list[dict[str, Any]] | None:
    """The grippers one catalog row's robot has, or None when none is recorded."""
    from .website_task_preparation import embodiment_grippers

    registration = settings.get("robot_policy") if isinstance(settings, Mapping) else None
    if isinstance(registration, Mapping) and isinstance(registration.get("manifest"), Mapping):
        try:
            from .beta_policy_screening_launch import _bound_path, _read
            from .team_embodiment import validate_embodiment_manifest

            grippers = embodiment_grippers(validate_embodiment_manifest(_read(_bound_path(registration["manifest"]))))
        except (ValueError, OSError, KeyError, TypeError):
            return None
        return grippers or None
    tool_id = PROFILE_TOOLS.get(str((candidate.get("interface") or {}).get("robot_profile_id")))
    tool = _catalog_tool(tool_id) if tool_id else None
    return [tool] if tool else None


def _flags(screen: Mapping[str, Any]) -> list[str]:
    escalation = screen.get("measurement_escalation") or {}
    if screen.get("sensitivity") == "blocked_by_estimate":
        return [FLAG_WIDER if escalation.get("property") == "smallest_dimension_m" else FLAG_HEAVIER]
    if screen.get("sensitivity") == "outcome_depends_on_estimate":
        return [FLAG_DEPENDS]
    return []


def grasp_screen(dimensions_m: Any, grippers: list[Mapping[str, Any]] | None, *,
                 approach: str | None = None) -> dict[str, Any]:
    """One robot row's record: each gripper's screen and the row's flags.

    ``dimensions_m`` is the object's box at rest: two horizontal extents, then its height.
    ``approach`` is the grasp the carry check recorded (``"top_down"`` on 447), kept with the record.
    """
    from .website_task_preparation import DIMENSION_RELATIVE_ERROR, screen_physics

    if not grippers:
        return {"schema_version": SCHEMA_VERSION, "status": "not_assessed", "reason": "no_gripper_recorded",
                "flags": []}
    rows = []
    for gripper in grippers:
        screen = screen_physics(dimensions_m, gripper=gripper)
        flags = _flags(screen)
        # The grasp it will use closes horizontally: across the smaller horizontal extent at rest.
        across = min(float(dimensions_m[0]), float(dimensions_m[1])) * (1 + DIMENSION_RELATIVE_ERROR)
        closes_across = across <= float(gripper["full_stroke_m"])
        if not closes_across and FLAG_WIDER not in flags:
            flags.append(FLAG_EDGE)
        rows.append({"model": str(gripper["model"]), "full_stroke_m": float(gripper["full_stroke_m"]),
                     "grip_force_recorded": gripper.get("grip_force_n") is not None,
                     "sensitivity": screen["sensitivity"], "horizontal_width_upper_m": across,
                     "closes_across_object": closes_across, "flags": sorted(flags)})
    # A robot with several hands is flagged only when none of them can take the object.
    flags = sorted(set.intersection(*(set(row["flags"]) for row in rows)))
    record = {"schema_version": SCHEMA_VERSION, "status": "assessed", "grippers": rows, "flags": flags,
              "dimensions_m": [float(value) for value in dimensions_m], "approach": approach}
    numbers = [*record["dimensions_m"], *(value for row in rows
                                          for value in (row["full_stroke_m"], row["horizontal_width_upper_m"]))]
    if not all(math.isfinite(value) for value in numbers):
        # A figure that is not a number cannot be published or sealed in the plan: not assessed (grasp_screens).
        raise ValueError("grasp_screen_figure_not_finite")
    return record


def object_dimensions(task_spec: Mapping[str, Any]) -> list[float] | None:
    """The task object's collision-box size from a compiled rigid task spec."""
    bounds = task_spec.get("subject_collision_bounds_scoring_frame_m")
    try:
        size = [float(high) - float(low) for low, high in zip(bounds["minimum"], bounds["maximum"], strict=True)]
    except (KeyError, TypeError, ValueError):
        return None
    return size if len(size) == 3 and all(value > 0 for value in size) else None


def recorded_approach(capture_root: Any) -> str | None:
    """The grasp the job's carry check found (``capability_verdict.json``'s ``reach.grasp``), or None."""
    import json
    from pathlib import Path

    try:
        path = Path(capture_root) / "pipeline" / "website_scene_preparation" / "native" / "capability_verdict.json"
        grasp = (json.loads(path.read_text(encoding="utf-8")).get("reach") or {}).get("grasp")
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return grasp if isinstance(grasp, str) and grasp else None


def grasp_screens(*, catalog: Mapping[str, Any], profile: Mapping[str, Any], task_spec: Mapping[str, Any],
                  approach: str | None = None) -> dict[str, dict[str, Any]] | None:
    """Every catalog row's grasp record for this job, or None when the object has no box."""
    dimensions = object_dimensions(task_spec)
    if dimensions is None:
        return None
    configured = profile.get("screening_candidates") or {}
    records: dict[str, dict[str, Any]] = {}
    for row in catalog.get("candidates") or []:
        try:
            records[str(row["candidate_id"])] = grasp_screen(
                dimensions, candidate_grippers(row, configured.get(row["candidate_id"])), approach=approach)
        except Exception as exc:  # noqa: BLE001 - a flag-only check must never stop planning
            if isinstance(row, Mapping) and isinstance(row.get("candidate_id"), str):
                records[row["candidate_id"]] = not_assessed(exc)
    return records


def not_assessed(exc: BaseException) -> dict[str, Any]:
    """A row whose grasp could not be screened: recorded with its reason, never a stop."""
    return {"schema_version": SCHEMA_VERSION, "status": "not_assessed",
            "reason": "grasp_screen_failed:" + type(exc).__name__, "flags": []}


def validate_grasp_screen(value: Any) -> dict[str, Any]:
    """A plan row's grasp record, as the plan validator reads it."""
    if (not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION
            or value.get("status") not in {"assessed", "not_assessed"} or not isinstance(value.get("flags"), list)
            or not set(value["flags"]) <= FLAGS or value["flags"] != sorted(set(value["flags"]))):
        raise ValueError("beta_screening_grasp_screen_invalid")
    return value


__all__ = ["FLAGS", "FLAG_DEPENDS", "FLAG_EDGE", "FLAG_HEAVIER", "FLAG_WIDER", "PROFILE_TOOLS", "SCHEMA_VERSION",
           "candidate_grippers", "grasp_screen", "grasp_screens", "not_assessed", "object_dimensions", "recorded_approach",
           "validate_grasp_screen"]
