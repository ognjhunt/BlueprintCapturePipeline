from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.adp009d_task_scoring import CAN_START_POSITION_M
from blueprint_pipeline.adp_task_scoring import (
    OUTCOME_NEVER_MOVED,
    OUTCOME_NON_TASK_JOINT_MOVED,
    OUTCOME_OPENED_AND_SETTLED,
    OUTCOME_OPENED_THEN_REBOUNDED,
    OUTCOME_RELEASE_OR_RETREAT_INCOMPLETE,
    TaskNeutralScoringError,
    score_task_episode_from_spec,
)


def _articulated_spec() -> dict:
    return {
        "schema_version": "adp_task_spec.v1",
        "task_kind": "articulated_open_close",
        "target_joint_id": "right_door",
        "joint_reset_positions_rad": {"right_door": 0.0, "left_door": 0.0},
        "target_success_interval_rad": [0.785398163, 1.396263402],
        "joint_hard_limits_rad": {
            "right_door": [0.0, 1.919862177],
            "left_door": [-0.01, 1.919862177],
        },
        "settle_window_samples": 3,
        "maximum_settled_target_speed_rad_s": 0.05,
        "non_task_joint_motion_tolerance_rad": 0.001,
        "movement_epsilon_rad": 0.0001,
        "reset_tolerance_rad": 0.0001,
    }


def _sample(step: int, target: float, *, other: float = 0.0, speed: float = 0.0) -> dict:
    return {
        "step_index": step,
        "joint_positions_rad": {"right_door": target, "left_door": other},
        "joint_velocities_rad_s": {"right_door": speed, "left_door": 0.0},
        "task_contact_active": False,
        "joint_limit_violation": False,
        "containment_violation": False,
        "robot_collision_failure": False,
        "scene_collision_failure": False,
        "retreat_completed": step >= 3,
    }


def test_task_neutral_dispatch_preserves_original_rigid_fixture() -> None:
    samples = [
        {
            "step_index": index,
            "can_pose_world": [*CAN_START_POSITION_M, 0.0, 0.0, 0.0, 1.0],
        }
        for index in range(40)
    ]
    report = score_task_episode_from_spec(
        task_spec={
            "schema_version": "adp_task_spec.v1",
            "task_kind": "rigid_pick_place",
            "destination_position_world_m": [
                CAN_START_POSITION_M[0] + 0.2,
                CAN_START_POSITION_M[1],
                CAN_START_POSITION_M[2],
            ],
            "support_plane_z_m": CAN_START_POSITION_M[2],
            "settle_window_samples": 40,
            "require_sealed_start_pose": True,
        },
        samples=samples,
    )

    assert report["outcome"] == OUTCOME_NEVER_MOVED
    assert report["task_succeeded"] is False


def test_articulated_zero_action_is_a_deterministic_negative() -> None:
    report = score_task_episode_from_spec(
        task_spec=_articulated_spec(),
        samples=[_sample(index, 0.0) for index in range(4)],
    )

    assert report["outcome"] == OUTCOME_NEVER_MOVED
    assert report["task_succeeded"] is False
    assert report["judgement_source"] == "deterministic_native_simulator_joint_state"


def test_articulated_scripted_positive_requires_release_retreat_and_settle() -> None:
    samples = [
        _sample(0, 0.0),
        _sample(1, 0.4, speed=0.4),
        _sample(2, 0.9, speed=0.2),
        _sample(3, 0.9),
        _sample(4, 0.9),
        _sample(5, 0.9),
    ]

    report = score_task_episode_from_spec(task_spec=_articulated_spec(), samples=samples)

    assert report["outcome"] == OUTCOME_OPENED_AND_SETTLED
    assert report["outcome_rank"] == 4
    assert report["task_succeeded"] is True
    assert all(report["predicates"].values())


def test_opened_then_rebounded_cannot_pass() -> None:
    samples = [
        _sample(0, 0.0),
        _sample(1, 0.9),
        _sample(2, 0.4),
        _sample(3, 0.3),
        _sample(4, 0.2),
    ]

    report = score_task_episode_from_spec(task_spec=_articulated_spec(), samples=samples)

    assert report["outcome"] == OUTCOME_OPENED_THEN_REBOUNDED
    assert report["outcome_rank"] == 2
    assert report["task_succeeded"] is False


@pytest.mark.parametrize(("contact_active", "retreat_completed"), [(True, True), (False, False)])
def test_stable_open_without_release_or_retreat_has_its_frozen_failure_rung(
    contact_active: bool, retreat_completed: bool
) -> None:
    samples = [
        _sample(0, 0.0),
        _sample(1, 0.9),
        _sample(2, 0.9),
        _sample(3, 0.9),
        _sample(4, 0.9),
    ]
    for sample in samples[-3:]:
        sample["task_contact_active"] = contact_active
        sample["retreat_completed"] = retreat_completed

    report = score_task_episode_from_spec(task_spec=_articulated_spec(), samples=samples)

    assert report["outcome"] == OUTCOME_RELEASE_OR_RETREAT_INCOMPLETE
    assert report["outcome_rank"] == 3
    assert report["task_succeeded"] is False


def test_non_task_joint_motion_blocks_otherwise_successful_open() -> None:
    samples = [
        _sample(0, 0.0),
        _sample(1, 0.9, other=0.002),
        _sample(2, 0.9, other=0.002),
        _sample(3, 0.9, other=0.002),
    ]

    report = score_task_episode_from_spec(task_spec=_articulated_spec(), samples=samples)

    assert report["outcome"] == OUTCOME_NON_TASK_JOINT_MOVED
    assert report["task_succeeded"] is False


def test_missing_native_velocity_fails_closed() -> None:
    sample = _sample(0, 0.0)
    del sample["joint_velocities_rad_s"]

    with pytest.raises(TaskNeutralScoringError) as caught:
        score_task_episode_from_spec(
            task_spec=_articulated_spec(), samples=[sample, copy.deepcopy(sample)]
        )

    assert any("joint_velocities_invalid" in error for error in caught.value.errors)
