"""Score a G1 navigation policy against a visible goal in the same task scene.

This is a development simulator criterion for the pinned HSI vision-navigation
checkpoint. It measures root position from Isaac, not a policy self-report.
Obstacle avoidance and physical transfer are outside this score.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "native_g1_navigation_goal.v1"
SCORE_SCHEMA_VERSION = "native_g1_navigation_goal_score.v1"
PUBLISHED_TASK_INSTRUCTION = "Avoid obstacles and move to the yellow marked area."


def validate_g1_navigation_goal(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    goal = task_spec.get("g1_navigation_goal")
    marker = task_spec.get("visible_target_marker")
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
    }


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
    within = [
        distance <= radius
        and abs(height - heights[0]) <= goal["max_root_height_drift_m"]
        for distance, height in zip(distances, heights, strict=True)
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
        "outcome": "success" if terminal_hold else "failure",
        "first_settled_step": first_settled_step,
        "terminal_goal_hold": terminal_hold,
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
