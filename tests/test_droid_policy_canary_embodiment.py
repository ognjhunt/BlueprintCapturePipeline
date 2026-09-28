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


def test_pick_place_instruction_uses_configured_relation_and_labels() -> None:
    assert concrete_droid_task_instruction(
        {
            "manipulation_strategy": "pick_and_place",
            "destination_relation": "inside",
            "instruction_subject_label": "open book",
            "visible_target_label": "blue document tray",
        }
    ) == "Pick up and place the open book into the blue document tray."


def test_articulated_instruction_names_the_part_and_opening_action() -> None:
    spec = {
        "task_kind": "articulated_open_close",
        "manipulation_strategy": "articulated_open_close",
        "instruction_subject_label": "three-drawer wood-front mobile cabinet",
        "visible_target_label": "middle drawer",
        "configured_success_criteria": {"joint_type": "prismatic", "closing_required": False},
    }
    assert concrete_droid_task_instruction(spec) == (
        "Pull open the middle drawer of the three-drawer wood-front mobile cabinet."
    )
    del spec["visible_target_label"]
    with pytest.raises(ValueError, match="articulated_instruction_invalid"):
        concrete_droid_task_instruction(spec)


def test_embodiment_parity_records_failed_approach_without_blocking_valid_wiring() -> None:
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
    assert passed["policy_approach_observed"] is True

    episode["state_trace"]["task_state_samples"][1][
        "grasp_frame_position_world_m"
    ] = [-0.1, 0.0, 0.0]
    failed_policy = _episode_embodiment_parity_diagnostic(
        episode, observation_support_qualified=True
    )
    assert failed_policy["status"] == "passed"
    assert failed_policy["policy_approach_observed"] is False
    assert failed_policy["approach_distance_m"] == 0.0
    assert failed_policy["blockers"] == []
    legacy = _episode_embodiment_parity_diagnostic(
        episode, observation_support_qualified=True, legacy_approach_gate=True
    )
    assert legacy["status"] == "blocked"
    assert "droid_gripper_did_not_approach_task" in legacy["blockers"]
    assert "policy_approach_observed" not in legacy

    for row in episode["state_trace"]["task_state_samples"]:
        row["handle_reference_position_world_m"] = row.pop("task_object_pose_world")[:3]
    drawer = _episode_embodiment_parity_diagnostic(
        episode, observation_support_qualified=True
    )
    assert drawer["status"] == "passed"
    assert drawer["initial_gripper_to_task_distance_m"] == pytest.approx(1.0)

    del episode["state_trace"]["task_state_samples"][0]["handle_reference_position_world_m"]
    del episode["state_trace"]["task_state_samples"][1]["handle_reference_position_world_m"]
    missing_native_geometry = _episode_embodiment_parity_diagnostic(
        episode, observation_support_qualified=True
    )
    assert missing_native_geometry["status"] == "blocked"
    assert "droid_gripper_task_distance_unavailable" in missing_native_geometry["blockers"]
