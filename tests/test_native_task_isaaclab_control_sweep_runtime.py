from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from blueprint_pipeline.native_task_isaaclab_control_sweep_runtime import (
    NativeIsaacLabControlSweepTraceReader,
)


class _Scene(dict):
    env_origins = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


def _built():
    task = SimpleNamespace(
        data=SimpleNamespace(
            root_pose_w=np.asarray(
                [
                    [1.1, 2.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                    [3.2, 2.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
    )
    robot = SimpleNamespace(
        joint_names=[f"panda_joint{index}" for index in range(1, 8)] + ["gripper"],
        data=SimpleNamespace(
            joint_pos=np.asarray(
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
                    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                ]
            )
        ),
    )
    task_contact = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=np.asarray(
                [
                    [[[0.2, 0.0, 0.0]], [[0.6, 0.0, 0.0]]],
                    [[[0.1, 0.0, 0.0]], [[0.3, 0.4, 0.0]]],
                ]
            )
        )
    )
    robot_scene = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=np.asarray(
                [
                    [[[0.0, 0.0, 0.1]]],
                    [[[0.0, 0.0, 0.8]]],
                ]
            )
        )
    )
    scene = _Scene(
        task_object=task,
        robot=robot,
        task_contact=task_contact,
        robot_scene=robot_scene,
    )
    return SimpleNamespace(
        env=SimpleNamespace(unwrapped=SimpleNamespace(scene=scene)),
        plan={
            "task_spec": {
                "interaction_affordance": {
                    "asset_root_from_scoring_frame": {
                        "position_m": [0.1, 0.0, 0.0],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                }
            }
        },
        scene_asset_names={"task_object": "task_object"},
        contact_sensor_names={
            "task_robot_contact": ("task_contact",),
            "robot_scene_contact": ("robot_scene",),
        },
    )


def test_trace_reader_returns_clone_local_task_joint_and_contact_state() -> None:
    reader = NativeIsaacLabControlSweepTraceReader(_built())

    assert reader.environment_count == 2
    assert reader.scoring_positions_world_m() == [
        [1.0, 2.0, 0.8],
        [1.1, 2.0, 0.8],
    ]
    assert reader.arm_joint_positions_rad(
        arm_joint_names=[f"panda_joint{index}" for index in range(1, 8)]
    ) == [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    ]
    assert reader.peak_contact_force_vectors_w_n(
        logical_sensor_ids=("task_robot_contact", "robot_scene_contact")
    ) == [
        [0.6, 0.0, 0.0],
        [0.0, 0.0, 0.8],
    ]
