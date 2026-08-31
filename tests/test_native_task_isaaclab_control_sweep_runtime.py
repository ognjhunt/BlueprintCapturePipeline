from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_isaaclab_control_sweep_runtime import (
    NativeIsaacLabControlSweepTraceReader,
    NativeIsaacLabControlSweepWaveRunner,
)
from blueprint_pipeline.task_evaluation_control_search_funnel import (
    build_control_search_funnel_plan,
)
from blueprint_pipeline.task_evaluation_isaaclab_control_sweep import (
    build_isaaclab_control_sweep_schedule,
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


def _wave_candidate(index: int) -> dict:
    joints = {f"panda_joint{joint}": 0.01 * joint for joint in range(1, 8)}
    candidate = {
        "candidate_id": f"candidate-{index:03d}",
        "deterministic_rank": index,
        "robot_base_pose_world": {
            "position_world_m": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "reset_variant": {"robot_joint_reset_positions_rad": joints},
        "entry_trajectory_variant": {
            "waypoints": [
                {
                    "waypoint_id": "entry-0",
                    "stage_kind": "entry",
                    "robot_joint_positions_rad": joints,
                }
            ]
        },
        "interaction_trajectory_variant": {
            "waypoints": [
                {
                    "waypoint_id": "contact-0",
                    "stage_kind": "contact",
                    "robot_joint_positions_rad": joints,
                },
                {
                    "waypoint_id": "release-0",
                    "stage_kind": "release",
                    "robot_joint_positions_rad": joints,
                },
            ]
        },
        "candidate_digest": "",
    }
    candidate["candidate_digest"] = canonical_digest(
        candidate, digest_field="candidate_digest"
    )
    return candidate


def _wave_inventory() -> dict:
    inventory = {
        "schema_version": (
            "task_evaluation_native_construction_candidate_inventory.v1"
        ),
        "run_id": "scene-839873-vector-native",
        "round_index": 0,
        "source_native_feedback_digest": "sha256:" + "0" * 64,
        "model_authored_candidates": False,
        "candidates": [_wave_candidate(index) for index in range(8)],
        "inventory_digest": "",
    }
    inventory["inventory_digest"] = canonical_digest(
        inventory, digest_field="inventory_digest"
    )
    return inventory


def test_wave_runner_applies_clone_targets_and_reduces_native_traces() -> None:
    torch = pytest.importorskip("torch")
    inventory = _wave_inventory()
    plan = build_control_search_funnel_plan(
        run_id="scene-839873-vector-native",
        source_commit="a" * 40,
        packet_request_digest="sha256:" + "1" * 64,
        candidate_inventory=inventory,
        runtime_source_packet_digest="sha256:" + "2" * 64,
        scene_collision_digest="sha256:" + "3" * 64,
        task_object_asset_digest="sha256:" + "4" * 64,
        robot_configuration_digest="sha256:" + "5" * 64,
        task_scoring_digest="sha256:" + "6" * 64,
        requested_vector_env_count=8,
        maximum_vector_env_count=1_024,
        seeds_per_candidate=1,
        shortlist_size=8,
    )
    schedule = build_isaaclab_control_sweep_schedule(
        plan=plan,
        candidate_inventory=inventory,
        base_seed=839873104,
    )

    class Robot:
        joint_names = [f"panda_joint{index}" for index in range(1, 8)] + [
            "gripper"
        ]

        def __init__(self):
            self.data = SimpleNamespace(
                root_pose_w=torch.tensor(
                    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 8
                ),
                joint_pos=torch.zeros((8, 8)),
            )

        def write_root_pose_to_sim(self, value, *, env_ids):
            self.data.root_pose_w[env_ids] = value

        def write_joint_state_to_sim(self, position, velocity, *, env_ids):
            assert torch.all(velocity == 0.0)
            self.data.joint_pos[env_ids] = position

    robot = Robot()
    task = SimpleNamespace(
        data=SimpleNamespace(
            root_pose_w=torch.tensor(
                [[0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]] * 8
            )
        )
    )
    task_contact = SimpleNamespace(
        data=SimpleNamespace(force_matrix_w=torch.zeros((8, 1, 1, 3)))
    )
    robot_scene = SimpleNamespace(
        data=SimpleNamespace(force_matrix_w=torch.zeros((8, 1, 1, 3)))
    )

    class Scene(dict):
        env_origins = torch.tensor([[2.0 * index, 0.0, 0.0] for index in range(8)])

    scene = Scene(
        robot=robot,
        task_object=task,
        task_contact=task_contact,
        robot_scene=robot_scene,
    )

    class Env:
        device = "cpu"

        def __init__(self):
            self.unwrapped = self
            self.scene = scene

        def reset(self, *, seed):
            assert seed == 839873104
            task.data.root_pose_w[:, :3] = scene.env_origins + torch.tensor(
                [0.0, 0.0, 0.8]
            )
            task.data.root_pose_w[:, 3:7] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0]
            )
            task_contact.data.force_matrix_w.zero_()
            robot_scene.data.force_matrix_w.zero_()

        def step(self, action):
            robot.data.joint_pos[:, :7] = action[:, :7]
            closed = action[:, 7] == 0.0
            task.data.root_pose_w[closed, 0] += 0.03
            task_contact.data.force_matrix_w.zero_()
            task_contact.data.force_matrix_w[closed, 0, 0, 0] = 0.6
            robot_scene.data.force_matrix_w[:, 0, 0, 2] = 0.1

    built = SimpleNamespace(
        env=Env(),
        plan={
            "task_spec": {
                "start_pose_world": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                "target_position_world_m": [0.03, 0.0, 0.8],
                "reset_translation_tolerance_m": 0.002,
                "task_contact_minimum_force_n": 0.5,
                "interaction_affordance": {
                    "asset_root_from_scoring_frame": {
                        "position_m": [0.0, 0.0, 0.0],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                },
            }
        },
        scene_asset_names={"task_object": "task_object"},
        contact_sensor_names={
            "task_robot_contact": ("task_contact",),
            "robot_scene_contact": ("robot_scene",),
        },
    )
    runner = NativeIsaacLabControlSweepWaveRunner(
        plan=plan,
        schedule=schedule,
        gripper_open_command=1.0,
        gripper_closed_command=0.0,
        steps_per_waypoint=1,
        settle_steps=2,
        torch_module=torch,
        peak_gpu_memory_probe=lambda: 1024,
    )

    result = runner(
        built=built,
        wave=schedule["waves"][0],
        candidate_inventory=inventory,
        plan=plan,
    )

    assert len(result["outcomes"]) == 8
    assert result["peak_gpu_memory_bytes"] == 1024
    assert all(row["reset_readback_passed"] is True for row in result["outcomes"])
    assert all(
        row["required_task_contact_coverage_fraction"] == 1.0
        for row in result["outcomes"]
    )
    assert [row["task_displacement_m"] for row in result["outcomes"]] == pytest.approx(
        [0.03] * 8, abs=1.0e-6
    )
