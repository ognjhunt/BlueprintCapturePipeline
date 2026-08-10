from __future__ import annotations

from types import SimpleNamespace

import pytest

from blueprint_pipeline.native_task_arena_readback import (
    NativeArticulatedTaskArenaReadback,
    NativeTaskArenaReadbackError,
)
from blueprint_pipeline.native_task_arena_runtime import NativeTaskArenaEnvironment


def _built(*, include_forces: bool = True) -> NativeTaskArenaEnvironment:
    task_object = SimpleNamespace(
        joint_names=["upper_door_hinge", "lower_door_hinge"],
        data=SimpleNamespace(
            joint_pos=[[0.872664626, 0.0]],
            joint_vel=[[0.0, 0.0]],
            root_pose_w=[[1.9742142, 1.4792181, 0.0, 0.0, 0.0, 0.0, 1.0]],
            body_names=["cabinet", "upper_door", "lower_door"],
            body_pose_w=[
                [
                    [1.9742142, 1.4792181, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.9742142, 1.4792181, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.9742142, 1.4792181, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ],
        ),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=["left_inner_finger", "right_inner_finger"],
            body_pose_w=[
                [
                    [2.25, 2.25, 1.02, 0.0, 0.0, 0.0, 1.0],
                    [2.27, 2.25, 1.02, 0.0, 0.0, 0.0, 1.0],
                ]
            ],
        )
    )
    force = [[[[0.0, 0.0, 0.0]]]] if include_forces else None
    scene = {
        "task_object": task_object,
        "robot": robot,
        "task_robot_contact": SimpleNamespace(
            data=SimpleNamespace(force_matrix_w=force)
        ),
        "task_scene_contact": SimpleNamespace(
            data=SimpleNamespace(force_matrix_w=force)
        ),
        "robot_scene_contact": SimpleNamespace(
            data=SimpleNamespace(force_matrix_w=force)
        ),
        "robot_scene_contact_2": SimpleNamespace(
            data=SimpleNamespace(force_matrix_w=force)
        ),
    }
    task_spec = {
        "task_kind": "articulated_open_close",
        "joint_reset_positions_rad": {
            "refrigerator_upper_door_hinge": 0.0,
            "refrigerator_lower_door_hinge": 0.0,
        },
        "joint_hard_limits_rad": {
            "refrigerator_upper_door_hinge": [0.0, 1.57],
            "refrigerator_lower_door_hinge": [0.0, 1.57],
        },
    }
    plan = {
        "task_kind": "articulated_open_close",
        "task_spec": task_spec,
        "task_sample_binding": {
            "joint_ids": [
                "refrigerator_lower_door_hinge",
                "refrigerator_upper_door_hinge",
            ],
            "native_joint_names": {
                "refrigerator_upper_door_hinge": "upper_door_hinge",
                "refrigerator_lower_door_hinge": "lower_door_hinge",
            },
        },
        "task_state_binding": {
            "task_contact_minimum_force_n": 0.5,
            "collision_failure_minimum_force_n": 1.0,
            "retreat_minimum_separation_m": 0.10,
            "root_translation_tolerance_m": 0.002,
            "root_orientation_tolerance_rad": 0.01,
        },
        "robot": {
            "grasp_frame": {
                "kind": "body_midpoint",
                "body_names": ["left_inner_finger", "right_inner_finger"],
            }
        },
        "articulation": {
            "moving_link_native_body_name": "upper_door",
            "handle_grasp_point_link_m": [0.119962, 0.327634, 1.022997],
        },
        "objects": [
            {
                "semantic_role": "task_object",
                "pose_world": {
                    "position_world_m": [1.9742142, 1.4792181, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ],
    }
    return NativeTaskArenaEnvironment(
        env=SimpleNamespace(unwrapped=SimpleNamespace(scene=scene)),
        cfg=None,
        plan=plan,
        scene_asset_names={"task_object": "task_object"},
        contact_sensor_names={
            "task_robot_contact": ("task_robot_contact",),
            "task_scene_contact": ("task_scene_contact",),
            "robot_scene_contact": (
                "robot_scene_contact",
                "robot_scene_contact_2",
            ),
        },
        camera_scene_names={},
    )


def test_live_numeric_readback_compiles_the_exact_scorer_sample() -> None:
    sample = NativeArticulatedTaskArenaReadback(_built()).read_task_sample()

    assert sample["joint_positions_rad"] == {
        "refrigerator_lower_door_hinge": 0.0,
        "refrigerator_upper_door_hinge": pytest.approx(0.872664626),
    }
    assert sample["task_contact_active"] is False
    assert sample["robot_collision_failure"] is False
    assert sample["scene_collision_failure"] is False
    assert sample["retreat_completed"] is True
    assert sample["native_readback"]["caller_asserted_booleans_used"] is False


def test_missing_force_matrix_never_defaults_to_no_collision() -> None:
    with pytest.raises(NativeTaskArenaReadbackError) as excinfo:
        NativeArticulatedTaskArenaReadback(
            _built(include_forces=False)
        ).read_task_sample()

    assert excinfo.value.errors == (
        "native_task_arena_force_matrix_missing:task_robot_contact:task_robot_contact",
    )


def test_multiple_native_sensor_instances_aggregate_into_one_logical_channel() -> None:
    built = _built()
    built.env.unwrapped.scene["robot_scene_contact_2"].data.force_matrix_w = [
        [[[2.0, 0.0, 0.0]]]
    ]

    sample = NativeArticulatedTaskArenaReadback(built).read_task_sample()

    assert sample["robot_collision_failure"] is True
    assert sample["native_readback"]["robot_scene_contact_peak_force_n"] == (
        pytest.approx(2.0)
    )


def test_missing_native_grasp_body_fails_closed() -> None:
    built = _built()
    built.plan["robot"]["grasp_frame"]["body_names"][1] = "guessed_finger"

    with pytest.raises(NativeTaskArenaReadbackError) as excinfo:
        NativeArticulatedTaskArenaReadback(built).read_task_sample()

    assert excinfo.value.errors == (
        "native_task_arena_grasp_body_missing:guessed_finger",
    )
