from __future__ import annotations

import pytest

from blueprint_pipeline.native_g1_navigation_goal import (
    PUBLISHED_TASK_INSTRUCTION,
    score_g1_navigation_episode,
    validate_g1_navigation_goal,
)


def _task_spec() -> dict:
    return {
        "task_kind": "rigid_pick_place",
        "prompt": "Pick and place the box",
        "visible_target_marker": {
            "schema_version": "native_task_target_marker.v1",
            "shape": "flat_yellow_disc",
            "non_colliding": True,
            "surface_position_world_m": [2.0, 0.0, 0.0],
            "radius_m": 0.4,
        },
        "g1_navigation_goal": {
            "schema_version": "native_g1_navigation_goal.v1",
            "center_world_m": [2.0, 0.0, 0.0],
            "acceptance_radius_m": 0.3,
            "max_root_height_drift_m": 0.2,
            "settle_window_samples": 2,
            "task_instruction": PUBLISHED_TASK_INSTRUCTION,
        },
    }


def _samples(xs: list[float]) -> list[dict]:
    return [
        {"step_index": index, "root_position_world_m": [x, 0.0, 0.85]}
        for index, x in enumerate(xs)
    ]


def test_navigation_score_requires_settled_measured_root_at_visible_goal() -> None:
    score = score_g1_navigation_episode(
        task_spec=_task_spec(), samples=_samples([0.0, 1.5, 1.8, 1.85])
    )
    assert score["outcome"] == "success"
    assert score["first_settled_step"] == 3
    assert score["obstacle_clearance_scored"] is False
    assert score["score_digest"].startswith("sha256:")


def test_navigation_goal_approach_without_settle_is_failure() -> None:
    score = score_g1_navigation_episode(
        task_spec=_task_spec(), samples=_samples([0.0, 1.8, 1.4])
    )
    assert score["outcome"] == "failure"
    assert score["first_settled_step"] is None


def test_navigation_goal_must_still_be_held_at_episode_end() -> None:
    score = score_g1_navigation_episode(
        task_spec=_task_spec(), samples=_samples([0.0, 1.8, 1.9, 1.4])
    )
    assert score["first_settled_step"] == 2
    assert score["terminal_goal_hold"] is False
    assert score["outcome"] == "failure"


def test_navigation_goal_rejects_mismatched_marker_and_unchanged_start() -> None:
    task = _task_spec()
    task["visible_target_marker"]["surface_position_world_m"][0] = 3.0
    with pytest.raises(ValueError, match="g1_navigation_goal_invalid"):
        validate_g1_navigation_goal(task)
    with pytest.raises(ValueError, match="start_already_at_goal"):
        score_g1_navigation_episode(
            task_spec=_task_spec(), samples=_samples([2.0, 2.0])
        )


def test_navigation_goal_rejects_height_collapse_as_success() -> None:
    samples = _samples([0.0, 1.8, 1.9])
    samples[2]["root_position_world_m"][2] = 0.1
    score = score_g1_navigation_episode(task_spec=_task_spec(), samples=samples)
    assert score["outcome"] == "failure"
