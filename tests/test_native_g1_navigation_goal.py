from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.adp_task_scoring import seal_rigid_task_success_contract
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_navigation_goal import (
    PUBLISHED_TASK_INSTRUCTION,
    score_g1_navigation_episode,
    seal_g1_navigation_goal_authority,
    validate_g1_navigation_goal,
    validate_g1_navigation_goal_authority,
)
from tests.test_task_evaluation_policy_canary_setup import _setup as canary_setup


def _task_spec() -> dict:
    return {
        "task_kind": "rigid_pick_place",
        "prompt": "Pick and place the box",
        "visible_target_marker": {
            "schema_version": "native_task_target_marker.v1",
            "shape": "flat_green_disc",
            "non_colliding": True,
            "surface_position_world_m": [0.5, 0.0, 0.7],
            "radius_m": 0.06,
        },
        "g1_navigation_goal": {
            "schema_version": "native_g1_navigation_goal.v1",
            "center_world_m": [2.0, 0.0, 0.0],
            "acceptance_radius_m": 0.3,
            "max_root_height_drift_m": 0.2,
            "settle_window_samples": 2,
            "task_instruction": PUBLISHED_TASK_INSTRUCTION,
            "visible_target_marker": {
                "schema_version": "native_task_target_marker.v1",
                "shape": "flat_yellow_disc",
                "non_colliding": True,
                "surface_position_world_m": [2.0, 0.0, 0.0],
                "radius_m": 0.4,
            },
        },
    }


def _samples(xs: list[float]) -> list[dict]:
    return [
        {"step_index": index, "root_position_world_m": [x, 0.0, 0.85]}
        for index, x in enumerate(xs)
    ]


def _authority_plan() -> dict:
    task = _task_spec()
    criteria = canary_setup()["task_success_contract"]["criteria"]
    contract = seal_rigid_task_success_contract(
        task_spec=task,
        site_id="site-a", task_id="task-a",
        author_source="task_owner", author_id="owner-a",
        confirmation_status="confirmed", confirmed_by_team_id="team-a",
        criteria=criteria,
    )
    task["task_success_contract"] = contract
    task["task_success_contract_digest"] = contract["contract_digest"]
    plan = {
        "scene_id": "scene-a", "task_id": "task-a",
        "task_kind": "rigid_pick_place", "robot": {"robot_id": "unitree_g1"},
        "task_spec": task,
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def test_navigation_authority_binds_team_task_scene_and_exact_goal() -> None:
    plan = _authority_plan()
    authority = seal_g1_navigation_goal_authority(
        plan=plan, confirmed_by_team_id="team-a", human_reviewer="owner-a"
    )
    assert validate_g1_navigation_goal_authority(authority, plan=plan) == authority
    assert authority["obstacle_clearance_scored"] is False
    assert authority["physical_outcome_claimed"] is False
    for changed in (
        {"scope": {**authority["scope"], "site_id": "other-site"}},
        {"goal": {**authority["goal"], "acceptance_radius_m": 0.5}},
        {"confirmed_by_team_id": "other-team"},
    ):
        tampered = copy.deepcopy(authority)
        tampered.update(changed)
        with pytest.raises(ValueError, match="g1_navigation_goal_authority_invalid"):
            validate_g1_navigation_goal_authority(tampered, plan=plan)
    plan["task_spec"]["g1_navigation_goal"]["acceptance_radius_m"] = 0.2
    with pytest.raises(ValueError, match="g1_navigation_goal_authority_invalid"):
        validate_g1_navigation_goal_authority(authority, plan=plan)


def test_navigation_authority_requires_confirmed_team_task_contract() -> None:
    plan = _authority_plan()
    plan["task_spec"]["task_success_contract_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="g1_navigation_goal_authority_invalid"):
        seal_g1_navigation_goal_authority(
            plan=plan, confirmed_by_team_id="team-a", human_reviewer="owner-a"
        )


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
    task["g1_navigation_goal"]["visible_target_marker"]["surface_position_world_m"][0] = 3.0
    with pytest.raises(ValueError, match="g1_navigation_goal_invalid"):
        validate_g1_navigation_goal(task)
    with pytest.raises(ValueError, match="start_already_at_goal"):
        score_g1_navigation_episode(
            task_spec=_task_spec(), samples=_samples([2.0, 2.0])
        )


def test_navigation_goal_never_substitutes_manipulation_marker() -> None:
    task = _task_spec()
    del task["g1_navigation_goal"]["visible_target_marker"]
    with pytest.raises(ValueError, match="goal_or_visible_marker_missing"):
        validate_g1_navigation_goal(task)
    assert task["visible_target_marker"]["shape"] == "flat_green_disc"


def test_navigation_goal_rejects_height_collapse_as_success() -> None:
    samples = _samples([0.0, 1.8, 1.9])
    samples[2]["root_position_world_m"][2] = 0.1
    score = score_g1_navigation_episode(task_spec=_task_spec(), samples=samples)
    assert score["outcome"] == "failure"
