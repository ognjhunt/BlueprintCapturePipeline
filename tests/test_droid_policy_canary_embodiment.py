from __future__ import annotations

import math

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.droid_policy_canary_embodiment import (
    DROID_NATIVE_RESET_JOINTS_RAD,
    DROID_POLICY_CANARY_PRESET_ID,
    apply_droid_policy_canary_profile,
    concrete_droid_task_instruction,
)
from blueprint_pipeline.native_task_arena_policy_canary_worker import (
    _episode_embodiment_parity_diagnostic,
)


def test_droid_profile_binds_official_reset_cameras_and_visible_target() -> None:
    plan = {
        "robot": {
            "joint_reset_positions_rad": {
                f"panda_joint{index}": 0.25 for index in range(1, 8)
            }
        },
        "task_spec": {
            "manipulation_strategy": "planar_push",
            "source_subject_identity": "scene-839873-mug-replacement",
            "target_position_world_m": [3.09, -6.76, 0.818],
            "prompt": "Move the configured rigid object.",
        },
        "plan_digest": "",
    }

    resolved = apply_droid_policy_canary_profile(plan)

    assert resolved["robot"]["joint_reset_positions_rad"] == (
        DROID_NATIVE_RESET_JOINTS_RAD
    )
    assert math.isclose(
        resolved["robot"]["joint_reset_positions_rad"]["panda_joint4"],
        -4 * math.pi / 5,
    )
    assert resolved["task_spec"]["prompt"] == (
        "Push the mug onto the green target marker."
    )
    profile = resolved["policy_canary_embodiment_profile"]
    assert profile["robot_preset_id"] == DROID_POLICY_CANARY_PRESET_ID
    assert profile["preserve_official_policy_camera_calibration"] is True
    assert profile["visible_target_marker"]["non_colliding"] is True
    assert profile["profile_digest"] == canonical_digest(
        profile, digest_field="profile_digest"
    )
    assert resolved["plan_digest"] == canonical_digest(
        resolved, digest_field="plan_digest"
    )


def test_concrete_instruction_falls_back_to_dynamic_subject_label() -> None:
    assert concrete_droid_task_instruction(
        {
            "manipulation_strategy": "planar_push",
            "subject_asset_id": "customer_site_blue_bottle_replacement",
        }
    ) == "Push the blue bottle onto the green target marker."


def test_embodiment_parity_requires_real_approach_without_joint_clamping() -> None:
    episode = {
        "state_trace": {
            "task_state_samples": [
                {
                    "grasp_frame_position_world_m": [0.0, 0.0, 0.0],
                    "task_object_pose_world": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                },
                {
                    "grasp_frame_position_world_m": [0.1, 0.0, 0.0],
                    "task_object_pose_world": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                },
            ]
        },
        "queries": [{"any_joint_limit_clamped": False}],
        "motion_evidence": {"actions_reached_robot": True, "arm_moved": True},
    }

    passed = _episode_embodiment_parity_diagnostic(
        episode, observation_support_qualified=True
    )
    assert passed["status"] == "passed"
    assert passed["approach_distance_m"] == pytest.approx(0.1)

    episode["state_trace"]["task_state_samples"][1][
        "grasp_frame_position_world_m"
    ] = [-0.1, 0.0, 0.0]
    blocked = _episode_embodiment_parity_diagnostic(
        episode, observation_support_qualified=True
    )
    assert blocked["status"] == "blocked"
    assert "droid_gripper_did_not_approach_task" in blocked["blockers"]
