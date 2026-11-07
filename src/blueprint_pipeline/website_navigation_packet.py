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
  and discloses whether the site confirmed its brief; an unconfirmed brief is never a stop
  (owner, 2026-11-07).
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
    ACCEPTANCE_RADIUS_BOUNDS_M,
    DEFAULT_EPISODE_SECONDS,
    G1_CONTROL_FREQUENCY_HZ,
    GOAL_BASIS_SCHEMA_VERSION,
    NAVIGATION_TASK_KIND,
    goal_from_compiled_spec,
)

TASK_SPEC_SCHEMA = "blueprint.website_navigation_task_spec.v1"


class WebsiteNavigationPacketError(ValueError):
    """A typed reason a compiled spec cannot become a G1 navigation task."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise WebsiteNavigationPacketError(code)


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
    """The G1 checkpoints' goal for a compiled navigation spec, and what had to be assumed for it.

    The spec is validated here; the derivation itself is ``goal_from_compiled_spec``, the same one
    the authority validator re-runs, so the two can never disagree.
    """
    try:
        return goal_from_compiled_spec(_compiled(spec))
    except ValueError as exc:
        if isinstance(exc, WebsiteNavigationPacketError):
            raise
        raise WebsiteNavigationPacketError(str(exc)) from exc


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
