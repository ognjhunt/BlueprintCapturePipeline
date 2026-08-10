from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.native_articulated_task_state import (
    NativeArticulatedTaskStateError,
    compile_native_articulated_task_sample,
)


ROOT = Path(__file__).parents[1]
TASK = json.loads(
    (
        ROOT
        / "docs/arm_decision_proof_v1/manifests"
        / "second_scene_840796_scene_task_freeze.v1.json"
    ).read_text(encoding="utf-8")
)["task_spec"]
BINDING = {
    "joint_ids": [
        "refrigerator_lower_door_hinge",
        "refrigerator_upper_door_hinge",
    ],
    "joint_prim_paths": {
        "refrigerator_lower_door_hinge": "/Asset/joints/lower_door_hinge",
        "refrigerator_upper_door_hinge": "/Asset/joints/upper_door_hinge",
    },
}
STATE = {
    "task_contact_minimum_force_n": 0.5,
    "collision_failure_minimum_force_n": 1.0,
    "retreat_minimum_separation_m": 0.10,
    "root_translation_tolerance_m": 0.002,
    "root_orientation_tolerance_rad": 0.01,
}


def _sample(**overrides):
    arguments = {
        "task_spec": TASK,
        "task_sample_binding": BINDING,
        "task_state_binding": STATE,
        "native_joint_names": ["upper_door_hinge", "lower_door_hinge"],
        "native_joint_positions_rad": [0.872664626, 0.0],
        "native_joint_velocities_rad_s": [0.0, 0.0],
        "task_robot_contact_forces_w_n": [[0.0, 0.0, 0.0]],
        "task_scene_contact_forces_w_n": [[0.0, 0.0, 0.0]],
        "robot_scene_contact_forces_w_n": [[0.0, 0.0, 0.0]],
        "task_root_pose_world": [1.9742142, 1.4792181, 0.0, 0.0, 0.0, 0.0, 1.0],
        "task_root_reset_pose_world": [
            1.9742142,
            1.4792181,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "grasp_frame_position_world_m": [2.10, 2.30, 1.02],
        "handle_reference_position_world_m": [1.94, 2.18, 1.02],
    }
    arguments.update(overrides)
    return compile_native_articulated_task_sample(**arguments)


def test_exact_frozen_task_sample_is_derived_from_numeric_readback() -> None:
    sample = _sample()

    assert sample["joint_positions_rad"] == {
        "refrigerator_lower_door_hinge": 0.0,
        "refrigerator_upper_door_hinge": pytest.approx(0.872664626),
    }
    assert sample["task_contact_active"] is False
    assert sample["retreat_completed"] is True
    assert sample["joint_limit_violation"] is False
    assert sample["containment_violation"] is False
    assert sample["robot_collision_failure"] is False
    assert sample["scene_collision_failure"] is False
    assert sample["native_readback"]["caller_asserted_booleans_used"] is False


def test_contact_and_collision_are_thresholded_from_independent_sensors() -> None:
    sample = _sample(
        task_robot_contact_forces_w_n=[[0.0, 0.0, 0.6]],
        task_scene_contact_forces_w_n=[[0.0, 0.0, 1.1]],
        robot_scene_contact_forces_w_n=[[0.0, 1.2, 0.0]],
    )

    assert sample["task_contact_active"] is True
    assert sample["retreat_completed"] is False
    assert sample["scene_collision_failure"] is True
    assert sample["robot_collision_failure"] is True


def test_missing_contact_sensor_readback_never_defaults_to_no_contact() -> None:
    with pytest.raises(
        NativeArticulatedTaskStateError,
        match="contact_readback_missing:task_robot_contact",
    ):
        _sample(task_robot_contact_forces_w_n=None)


def test_wrong_native_joint_name_fails_before_scoring() -> None:
    with pytest.raises(
        NativeArticulatedTaskStateError,
        match="joint_unresolved:refrigerator_upper_door_hinge",
    ):
        _sample(native_joint_names=["wrong_upper", "lower_door_hinge"])


def test_root_drift_and_joint_limit_violation_are_measured() -> None:
    sample = _sample(
        native_joint_positions_rad=[1.7, 0.0],
        task_root_pose_world=[1.98, 1.4792181, 0.0, 0.0, 0.0, 0.0, 1.0],
    )

    assert sample["joint_limit_violation"] is True
    assert sample["containment_violation"] is True


def test_retreat_requires_both_release_and_measured_separation() -> None:
    sample = _sample(
        grasp_frame_position_world_m=[1.95, 2.18, 1.02],
        handle_reference_position_world_m=[1.94, 2.18, 1.02],
    )

    assert sample["task_contact_active"] is False
    assert sample["retreat_completed"] is False
