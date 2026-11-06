"""What a G1 navigation run needs, made from a website job's compiled navigation spec.

Owner request 2026-11-06 (~18:00Z): a website navigation job should reach a scored G1 run,
not hold at the scene handoff (``website_locomotion_task.HOLD_CODE``). The plan and its order
are in ``docs/G1_NAVIGATION_FROM_COMPILED_SPEC.md``. This module is the controller-side part
that needs no simulator:

- ``g1_navigation_goal_from_spec`` turns the compiled goal predicate into the G1 checkpoints'
  goal: the compiled centre, the acceptance radius kept inside what the checkpoints were
  validated for, and the flat yellow disc. Their trained prompt is "Avoid obstacles and move to
  the yellow marked area.", recorded as the checkpoints' prompt and never as the site's sentence.
- ``navigation_goal_basis`` records what that goal was made from: the site, the compiled spec and
  the brief. Each clamp and restyle is a disclosed assumption. The machine-composed authority
  (``native_g1_navigation_goal.compose_g1_navigation_goal_authority``, owner decision 4) binds it
  and refuses an unconfirmed brief.
- ``navigation_task_spec`` is the scene task: kind ``goal_region_navigation``, the robot's start
  pose facing the goal, the goal, the basis and the compiled spec itself, so the scorer reads
  the same predicates the compiler sealed.

Nothing here spends, books a machine or claims a result. Everything stays ``development_only``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_navigation_goal import (
    GOAL_BASIS_SCHEMA_VERSION,
    NAVIGATION_TASK_KIND,
    PUBLISHED_TASK_INSTRUCTION,
    SCHEMA_VERSION as GOAL_SCHEMA_VERSION,
    validate_g1_navigation_goal,
)

TASK_SPEC_SCHEMA = "blueprint.website_navigation_task_spec.v1"
#: The G1 navigation checkpoints' validated goal envelope (``validate_g1_navigation_goal``).
ACCEPTANCE_RADIUS_BOUNDS_M = (0.1, 0.5)
HEIGHT_DRIFT_BOUNDS_M = (0.05, 0.5)
SETTLE_SAMPLE_BOUNDS = (2, 100)
#: The G1 lane's control rate (``beta_policy_screening_g1_prepare``: 50 Hz policy, 200 Hz physics).
G1_CONTROL_FREQUENCY_HZ = 50.0
#: Without a time limit in the brief, an episode may last this long.
DEFAULT_EPISODE_SECONDS = 60.0


class WebsiteNavigationPacketError(ValueError):
    """A typed reason a compiled spec cannot become a G1 navigation task."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise WebsiteNavigationPacketError(code)


def _clamp(value: float, bounds: Sequence[float]) -> float:
    return float(min(max(value, bounds[0]), bounds[1]))


def _predicate(spec: Mapping[str, Any], kind: str, *, predicate_id: str | None = None) -> dict[str, Any]:
    rows = [row for row in spec["success"]["predicates"] if row.get("kind") == kind
            and (predicate_id is None or row.get("id") == predicate_id)]
    _require(len(rows) == 1, f"website_navigation_spec_{kind}_missing")
    return dict(rows[0])


def _compiled(spec: Mapping[str, Any]) -> dict[str, Any]:
    from .task_spec_compiler import validate_task_spec

    try:
        value = validate_task_spec(spec)
    except ValueError as exc:
        raise WebsiteNavigationPacketError("website_navigation_spec_invalid") from exc
    need = value["requirements"]
    _require(need.get("locomote") is True and need.get("manipulate") is False,
             "website_navigation_spec_not_navigation")
    return value


def g1_navigation_goal_from_spec(spec: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The G1 checkpoints' goal for a compiled navigation spec, and what had to be assumed for it."""
    spec = _compiled(spec)
    goal_row = _predicate(spec, "region_reached", predicate_id="goal")
    region = goal_row["region"]
    _require(region.get("shape") == "disc" and goal_row.get("subject") == {"kind": "robot_root"},
             "website_navigation_goal_region_unsupported")
    centre = [float(value) for value in region["center_world_m"]]
    compiled_radius = float(region["radius_m"])
    radius = _clamp(compiled_radius, ACCEPTANCE_RADIUS_BOUNDS_M)
    upright = _predicate(spec, "height_stable")
    drift = _clamp(float(upright["max_drift_m"]), HEIGHT_DRIFT_BOUNDS_M)
    hold = int(goal_row.get("hold_samples") or SETTLE_SAMPLE_BOUNDS[0])
    settle = int(_clamp(hold, SETTLE_SAMPLE_BOUNDS))
    assumptions = [
        "goal_marker_restyled_flat_yellow_disc_for_the_checkpoints_trained_prompt",
        "instruction_is_the_checkpoints_trained_prompt_not_the_site_sentence",
        f"settle_window_is_{settle}_samples_at_{G1_CONTROL_FREQUENCY_HZ:g}_hz",
    ]
    if not math.isclose(radius, compiled_radius):
        assumptions.append(f"acceptance_radius_clamped_from_{compiled_radius:.3f}m_to_{radius:.3f}m")
    if not math.isclose(drift, float(upright["max_drift_m"])):
        assumptions.append(f"root_height_drift_clamped_to_{drift:.3f}m")
    if settle != hold:
        assumptions.append(f"settle_window_clamped_from_{hold}_samples")
    assumptions.extend(f"compiled:{flag}" for flag in spec["flags"])
    goal = {
        "schema_version": GOAL_SCHEMA_VERSION,
        "center_world_m": centre,
        "acceptance_radius_m": radius,
        "max_root_height_drift_m": drift,
        "settle_window_samples": settle,
        "task_instruction": PUBLISHED_TASK_INSTRUCTION,
        "visible_target_marker": {
            "schema_version": "native_task_target_marker.v1", "shape": "flat_yellow_disc", "non_colliding": True,
            "surface_position_world_m": list(centre), "radius_m": radius,
        },
    }
    validate_g1_navigation_goal({"task_kind": NAVIGATION_TASK_KIND, "g1_navigation_goal": goal})
    return goal, assumptions


def navigation_goal_basis(spec: Mapping[str, Any], *, task_context: Mapping[str, Any],
                          assumptions: Sequence[str]) -> dict[str, Any]:
    """What the goal was made from, for the machine-composed authority to bind."""
    spec = _compiled(spec)
    _require(task_context.get("scene_id") == spec["site_id"], "website_navigation_site_mismatch")
    _require(isinstance(task_context.get("context_digest"), str), "website_navigation_task_context_invalid")
    return {"schema_version": GOAL_BASIS_SCHEMA_VERSION, "site_id": spec["site_id"],
            "compiled_task_spec_digest": spec["spec_digest"], "task_context_digest": task_context["context_digest"],
            "disclosed_assumptions": list(assumptions)}


def _facing(start: Sequence[float], target: Sequence[float]) -> list[float]:
    """A yaw-only quaternion (xyzw) turning +x towards ``target``."""
    yaw = math.atan2(float(target[1]) - float(start[1]), float(target[0]) - float(start[0]))
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def navigation_task_spec(spec: Mapping[str, Any], *, task_context: Mapping[str, Any]) -> dict[str, Any]:
    """The scene task a G1 navigation plan carries: start, goal, basis and the compiled spec."""
    spec = _compiled(spec)
    goal, assumptions = g1_navigation_goal_from_spec(spec)
    start = [float(value) for value in spec["start"]["region"]["center_world_m"]]
    facing = spec["start"].get("facing_world_m") or goal["center_world_m"]
    limits = [row for row in spec["success"]["predicates"] if row.get("kind") == "time_limit"]
    seconds = (int(limits[0]["max_steps"]) / G1_CONTROL_FREQUENCY_HZ) if limits else DEFAULT_EPISODE_SECONDS
    if not limits:
        assumptions = [*assumptions, f"episode_limit_defaulted_to_{DEFAULT_EPISODE_SECONDS:g}s"]
    value = {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_kind": NAVIGATION_TASK_KIND,
        "task_id": spec["task_id"],
        "site_id": spec["site_id"],
        "claim_ceiling": "development_only",
        "start_pose_world": [*start, *_facing(start, facing)],
        "start_source": spec["start"].get("source"),
        "g1_navigation_goal": goal,
        "navigation_goal_basis": navigation_goal_basis(spec, task_context=task_context, assumptions=assumptions),
        "compiled_task_spec": spec,
        "compiled_task_spec_digest": spec["spec_digest"],
        "control_frequency_hz": G1_CONTROL_FREQUENCY_HZ,
        "maximum_episode_seconds": seconds,
        "maximum_action_steps": int(math.floor(seconds * G1_CONTROL_FREQUENCY_HZ)),
    }
    value["task_spec_digest"] = canonical_digest(value, digest_field="task_spec_digest")
    return value


__all__ = ["ACCEPTANCE_RADIUS_BOUNDS_M", "TASK_SPEC_SCHEMA", "WebsiteNavigationPacketError",
           "g1_navigation_goal_from_spec", "navigation_goal_basis", "navigation_task_spec"]
