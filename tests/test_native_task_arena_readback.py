from __future__ import annotations

from types import SimpleNamespace

import pytest

from blueprint_pipeline.native_task_arena_readback import (
    NativeArticulatedTaskArenaReadback,
    NativeTaskArenaReadbackError,
    read_native_task_arena_object_reset_state,
)
from blueprint_pipeline.native_task_arena_runtime import NativeTaskArenaEnvironment


def _built(*, include_forces: bool = True) -> NativeTaskArenaEnvironment:
    task_object = SimpleNamespace(
        joint_names=["upper_door_hinge", "lower_door_hinge"],
        data=SimpleNamespace(
            joint_pos=[[0.872664626, 0.0]],
            joint_vel=[[0.0, 0.0]],
            # Isaac Lab exposes native quaternions in WXYZ order.
            root_pose_w=[[1.9742142, 1.4792181, 0.0, 1.0, 0.0, 0.0, 0.0]],
            body_names=["cabinet", "upper_door", "lower_door"],
            body_pose_w=[
                [
                    [1.9742142, 1.4792181, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.9742142, 1.4792181, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.9742142, 1.4792181, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ],
        ),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=["left_inner_finger", "right_inner_finger"],
            body_pose_w=[
                [
                    [2.25, 2.25, 1.02, 1.0, 0.0, 0.0, 0.0],
                    [2.27, 2.25, 1.02, 1.0, 0.0, 0.0, 0.0],
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
                "asset_id": "legacy_task_object",
                "name": "task_object",
                "task_subject": True,
                "object_type": "ARTICULATION",
                "pose_world": {
                    "position_world_m": [1.9742142, 1.4792181, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "reset_state": {
                    "root_pose_world": {
                        "position_world_m": [1.9742142, 1.4792181, 0.0],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "joint_positions": {
                        "upper_door_hinge": 0.0,
                        "lower_door_hinge": 0.0,
                    },
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


def test_reset_readback_covers_active_and_inactive_replacements() -> None:
    built = _built()
    built.env.unwrapped.scene["task_object"].data.joint_pos = [[0.0, 0.0]]
    inactive_name = "replacement__inactive_drawer"
    built.plan["objects"].append(
        {
            "semantic_role": "replacement",
            "asset_id": "inactive_drawer",
            "name": inactive_name,
            "task_subject": False,
            "object_type": "ARTICULATION",
            "pose_world": {
                "position_world_m": [2.0, 3.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "reset_state": {
                "root_pose_world": {
                    "position_world_m": [2.0, 3.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "joint_positions": {"drawer_slide": 0.25},
            },
        }
    )
    built.scene_asset_names[inactive_name] = inactive_name
    built.env.unwrapped.scene[inactive_name] = SimpleNamespace(
        joint_names=["drawer_slide"],
        data=SimpleNamespace(
            root_pose_w=[[2.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            joint_pos=[[0.25]],
        ),
    )

    report = read_native_task_arena_object_reset_state(built)

    assert report["passed"] is True
    assert [row["asset_id"] for row in report["objects"]] == [
        "legacy_task_object",
        "inactive_drawer",
    ]
    assert all(row["passed"] for row in report["objects"])


def test_inactive_replacement_mutation_fails_reset_replay() -> None:
    built = _built()
    inactive_name = "replacement__inactive_rigid"
    built.plan["objects"].append(
        {
            "semantic_role": "replacement",
            "asset_id": "inactive_rigid",
            "name": inactive_name,
            "task_subject": False,
            "object_type": "RIGID",
            "pose_world": {
                "position_world_m": [2.0, 3.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "reset_state": {
                "root_pose_world": {
                    "position_world_m": [2.0, 3.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "joint_positions": {},
            },
        }
    )
    built.scene_asset_names[inactive_name] = inactive_name
    built.env.unwrapped.scene[inactive_name] = SimpleNamespace(
        data=SimpleNamespace(
            root_pose_w=[[2.02, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        )
    )

    report = read_native_task_arena_object_reset_state(built)

    assert report["passed"] is False
    inactive = report["objects"][1]
    assert inactive["asset_id"] == "inactive_rigid"
    assert inactive["root_translation_error_m"] == pytest.approx(0.02)
    assert inactive["passed"] is False
