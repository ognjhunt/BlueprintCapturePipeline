from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_robot_placement_trajectory import (
    RobotPlacementTrajectoryError,
    placement_trajectory_from_native_plan,
    placement_trajectory_from_native_result,
    validate_robot_placement_trajectory,
)


def _plan() -> dict:
    value = {
        "schema_version": "native_rigid_construction_phase_plan.v1",
        "task_kind": "rigid_pick_place",
        "manipulation_strategy": "planar_push",
        "phase_count": 2,
        "execution_parameters": {
            "arrival_tolerance_m": 0.02,
            "arrival_orientation_tolerance_rad": 0.08,
        },
        "phases": [
            {
                "phase_id": "precontact",
                "position_world_m": [2.790353636, -6.7605156, 0.818319],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                "gripper_state": "open",
                "gate_ids": ["precontact_reachability"],
            },
            {
                "phase_id": "push_contact",
                "position_world_m": [2.910353636, -6.7605156, 0.818319],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                "gripper_state": "closed",
                "gate_ids": ["push_contact"],
            },
        ],
        "plan_digest": "",
    }
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


def test_projects_every_native_phase_and_binds_exact_plan() -> None:
    plan = _plan()

    trajectory = placement_trajectory_from_native_plan(plan)

    assert trajectory["source_plan_digest"] == plan["plan_digest"]
    assert [row["phase_id"] for row in trajectory["phases"]] == [
        "precontact",
        "push_contact",
    ]
    assert trajectory["model_may_modify_trajectory"] is False
    assert validate_robot_placement_trajectory(trajectory) == trajectory


def test_tampered_native_plan_or_projected_trajectory_fails_closed() -> None:
    plan = _plan()
    plan["phases"][0]["position_world_m"][0] += 0.1
    with pytest.raises(
        RobotPlacementTrajectoryError,
        match="robot_placement_native_trajectory_plan_invalid",
    ):
        placement_trajectory_from_native_plan(plan)

    trajectory = placement_trajectory_from_native_plan(_plan())
    tampered = copy.deepcopy(trajectory)
    tampered["phases"][0]["position_world_m"][0] += 0.1
    with pytest.raises(
        RobotPlacementTrajectoryError, match="robot_placement_trajectory_invalid"
    ):
        validate_robot_placement_trajectory(tampered)


def test_prior_native_result_supplies_exact_next_round_trajectory() -> None:
    plan = _plan()
    result = {
        "schema_version": "native_task_arena_construction_result.v1",
        "status": "blocked",
        "construction_phase_plan": plan,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")

    trajectory = placement_trajectory_from_native_result(result)

    assert trajectory["source_plan_digest"] == plan["plan_digest"]
    result["status"] = "completed"
    with pytest.raises(
        RobotPlacementTrajectoryError,
        match="robot_placement_native_construction_result_invalid",
    ):
        placement_trajectory_from_native_result(result)
