from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.policy_ranking_droid_kinematics import (
    RESET_JOINTS,
    integrate_joint_velocity_chunk,
)


def test_integrates_discrete_droid_joint_delta_and_gripper() -> None:
    actions = np.zeros((2, 8), dtype=np.float64)
    actions[0, 0] = 0.5
    actions[0, 7] = 0.6
    actions[1, 0] = -0.25
    result = integrate_joint_velocity_chunk(actions)
    assert result["status"] == "completed"
    assert result["states"][1]["joint_position_rad"][0] == pytest.approx(0.1)
    assert result["states"][1]["gripper_position"] == 1.0
    assert result["states"][2]["joint_position_rad"][0] == pytest.approx(0.05)
    assert result["states"][2]["gripper_position"] == 0.0
    assert result["claim_boundary"]["dynamics_or_contact_simulated"] is False


def test_action_and_joint_limits_are_recorded() -> None:
    initial = RESET_JOINTS.copy()
    initial[0] = 2.89
    actions = np.zeros((1, 8), dtype=np.float64)
    actions[0, 0] = 3.0
    result = integrate_joint_velocity_chunk(actions, initial_joint_position=initial)
    assert result["action_value_clip_count"] == 1
    assert result["joint_limit_clip_count"] == 1
    assert result["states"][1]["joint_position_rad"][0] == pytest.approx(2.8973)


@pytest.mark.parametrize(
    "actions",
    [np.zeros((0, 8)), np.zeros((16, 8)), np.zeros((2, 7)), [[float("nan")] * 8]],
)
def test_invalid_action_chunks_fail_closed(actions) -> None:
    with pytest.raises(ValueError, match="invalid_action_chunk"):
        integrate_joint_velocity_chunk(actions)
