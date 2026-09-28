"""Score a G1 navigation policy against a visible goal in the same task scene.

This is a development simulator criterion for the pinned HSI vision-navigation
checkpoint. It measures root position from Isaac, not a policy self-report.
Obstacle avoidance and physical transfer are outside this score.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .adp_task_scoring import TaskNeutralScoringError, validate_rigid_task_success_contract
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest


SCHEMA_VERSION = "native_g1_navigation_goal.v1"
SCORE_SCHEMA_VERSION = "native_g1_navigation_goal_score.v1"
AUTHORITY_SCHEMA_VERSION = "native_g1_navigation_goal_authority.v1"
PUBLISHED_TASK_INSTRUCTION = "Avoid obstacles and move to the yellow marked area."


def validate_g1_navigation_goal(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    goal = task_spec.get("g1_navigation_goal")
    marker = goal.get("visible_target_marker") if isinstance(goal, Mapping) else None
    if not isinstance(goal, Mapping) or not isinstance(marker, Mapping):
        raise ValueError("g1_navigation_goal_or_visible_marker_missing")
    try:
        center = [float(value) for value in goal["center_world_m"]]
        surface = [float(value) for value in marker["surface_position_world_m"]]
        radius = float(goal["acceptance_radius_m"])
        marker_radius = float(marker["radius_m"])
        max_height_drift = float(goal["max_root_height_drift_m"])
        settle = goal["settle_window_samples"]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("g1_navigation_goal_invalid") from exc
    if (
        task_spec.get("task_kind") != "rigid_pick_place"
        or goal.get("schema_version") != SCHEMA_VERSION
        or goal.get("task_instruction") != PUBLISHED_TASK_INSTRUCTION
        or marker.get("schema_version") != "native_task_target_marker.v1"
        or marker.get("shape") != "flat_yellow_disc"
        or marker.get("non_colliding") is not True
        or len(center) != 3 or len(surface) != 3
        or not all(math.isfinite(value) for value in (*center, *surface, radius, marker_radius, max_height_drift))
        or center != surface
        or not 0.1 <= radius <= marker_radius <= 0.5
        or not 0.05 <= max_height_drift <= 0.5
        or isinstance(settle, bool) or not isinstance(settle, int)
        or not 2 <= settle <= 100
    ):
        raise ValueError("g1_navigation_goal_invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "center_world_m": center,
        "acceptance_radius_m": radius,
        "max_root_height_drift_m": max_height_drift,
        "settle_window_samples": settle,
        "task_instruction": PUBLISHED_TASK_INSTRUCTION,
        "visible_target_marker": {
            "schema_version": "native_task_target_marker.v1",
            "shape": "flat_yellow_disc",
            "non_colliding": True,
            "surface_position_world_m": surface,
            "radius_m": marker_radius,
        },
    }


def validate_g1_navigation_goal_authority(
    value: Mapping[str, Any] | None, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind team-confirmed goal arrival to one sealed G1 task/site scene."""

    try:
        authority = json.loads(json.dumps(value, allow_nan=False))
        if (
            plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
            or plan.get("task_kind") != "rigid_pick_place"
            or (plan.get("robot") or {}).get("robot_id") != "unitree_g1"
        ):
            raise ValueError("scene_plan_mismatch")
        task_spec = plan["task_spec"]
        goal = validate_g1_navigation_goal(task_spec)
        task_contract = validate_rigid_task_success_contract(
            task_spec["task_success_contract"],
            expected_task_id=plan["task_id"],
        )
        if task_spec.get("task_success_contract_digest") != task_contract["contract_digest"]:
            raise ValueError("task_contract_digest_mismatch")
        team_id = task_contract["provenance"]["confirmed_by_team_id"]
        expected_scope = {
            "site_id": task_contract["scope"]["site_id"],
            "task_id": plan["task_id"],
            "scene_plan_digest": plan["plan_digest"],
            "task_success_contract_digest": task_contract["contract_digest"],
        }
    except (AttributeError, KeyError, TypeError, ValueError, TaskNeutralScoringError) as exc:
        raise ValueError("g1_navigation_goal_authority_invalid") from exc
    if (
        not isinstance(authority, dict)
        or set(authority) != {
            "schema_version", "status", "scope", "goal", "goal_digest",
            "confirmed_by_team_id", "human_reviewer", "reviewed_criterion",
            "obstacle_clearance_scored", "physical_outcome_claimed", "authority_digest",
        }
        or authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or authority.get("status") != "confirmed_for_development_simulation"
        or authority.get("scope") != expected_scope
        or authority.get("goal") != goal
        or authority.get("goal_digest") != cross_runtime_canonical_digest(goal)
        or not isinstance(team_id, str) or not team_id.strip()
        or authority.get("confirmed_by_team_id") != team_id
        or not isinstance(authority.get("human_reviewer"), str)
        or not authority["human_reviewer"].strip()
        or authority.get("reviewed_criterion") != "root_xy_goal_arrival_and_terminal_hold"
        or authority.get("obstacle_clearance_scored") is not False
        or authority.get("physical_outcome_claimed") is not False
        or authority.get("authority_digest") != cross_runtime_canonical_digest(
            authority, digest_field="authority_digest"
        )
    ):
        raise ValueError("g1_navigation_goal_authority_invalid")
    return authority


def seal_g1_navigation_goal_authority(
    *, plan: Mapping[str, Any], confirmed_by_team_id: str, human_reviewer: str
) -> dict[str, Any]:
    """Seal a recorded human decision after the caller has obtained it."""

    task_spec = plan["task_spec"]
    task_contract = validate_rigid_task_success_contract(
        task_spec["task_success_contract"], expected_task_id=plan["task_id"]
    )
    authority = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "status": "confirmed_for_development_simulation",
        "scope": {
            "site_id": task_contract["scope"]["site_id"],
            "task_id": plan["task_id"],
            "scene_plan_digest": plan["plan_digest"],
            "task_success_contract_digest": task_contract["contract_digest"],
        },
        "goal": validate_g1_navigation_goal(task_spec),
        "confirmed_by_team_id": confirmed_by_team_id,
        "human_reviewer": human_reviewer,
        "reviewed_criterion": "root_xy_goal_arrival_and_terminal_hold",
        "obstacle_clearance_scored": False,
        "physical_outcome_claimed": False,
    }
    authority["goal_digest"] = cross_runtime_canonical_digest(authority["goal"])
    authority["authority_digest"] = cross_runtime_canonical_digest(
        authority, digest_field="authority_digest"
    )
    return validate_g1_navigation_goal_authority(authority, plan=plan)


def score_g1_navigation_episode(
    *, task_spec: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    goal = validate_g1_navigation_goal(task_spec)
    if len(samples) < 2:
        raise ValueError("g1_navigation_samples_missing")
    distances: list[float] = []
    heights: list[float] = []
    for index, sample in enumerate(samples):
        try:
            position = [float(value) for value in sample["root_position_world_m"]]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("g1_navigation_root_readback_invalid") from exc
        if (
            sample.get("step_index") != index
            or len(position) != 3
            or not all(math.isfinite(value) for value in position)
        ):
            raise ValueError("g1_navigation_root_readback_invalid")
        distances.append(math.dist(position[:2], goal["center_world_m"][:2]))
        heights.append(position[2])
    radius = goal["acceptance_radius_m"]
    if distances[0] <= radius:
        raise ValueError("g1_navigation_start_already_at_goal")
    height_drifts = [abs(height - heights[0]) for height in heights]
    height_stable_throughout = all(
        drift <= goal["max_root_height_drift_m"] for drift in height_drifts
    )
    within = [
        distance <= radius
        and drift <= goal["max_root_height_drift_m"]
        for distance, drift in zip(distances, height_drifts, strict=True)
    ]
    settle = goal["settle_window_samples"]
    first_settled_step = next(
        (index for index in range(settle - 1, len(within))
         if all(within[index - settle + 1:index + 1])),
        None,
    )
    terminal_hold = len(within) >= settle and all(within[-settle:])
    result = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "status": "scored",
        "outcome": "success" if terminal_hold and height_stable_throughout else "failure",
        "first_settled_step": first_settled_step,
        "terminal_goal_hold": terminal_hold,
        "root_height_stable_throughout": height_stable_throughout,
        "maximum_root_height_drift_observed_m": max(height_drifts),
        "initial_distance_m": distances[0],
        "minimum_distance_m": min(distances),
        "terminal_distance_m": distances[-1],
        "sample_count": len(samples),
        "goal": goal,
        "criterion": "root_xy_inside_visible_goal_with_stable_height",
        "obstacle_clearance_scored": False,
        "evidence_level": "development_only",
    }
    result["score_digest"] = canonical_digest(result, digest_field="score_digest")
    return result
