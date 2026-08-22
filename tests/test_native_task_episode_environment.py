from __future__ import annotations

from types import SimpleNamespace

import pytest

from blueprint_pipeline.native_task_episode_environment import (
    NativeTaskEpisodeEnvironmentError,
    build_native_task_episode_environment,
)


class _Servo:
    def __init__(self):
        self.reset_count = 0
        self.calls = []

    def current_body_pose_world(self):
        return [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]

    def current_grasp_frame_pose_world(self):
        return [1.0, 2.0, 3.1, 0.0, 0.0, 0.0, 1.0]

    def current_gripper_pad_readback(self):
        return {
            "measured": {
                "pad_separation_m": 0.06,
                "controlled_body_position_world_m": [1.0, 2.0, 3.0],
                "controlled_body_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "pad_midpoint_world_m": [1.0, 2.0, 3.1],
                "finger_body_positions_world_m": {
                    "left": [0.96, 2.0, 3.1],
                    "right": [1.04, 2.0, 3.1],
                },
                "pad_centers_world_m": {
                    "left": [0.97, 2.0, 3.1],
                    "right": [1.03, 2.0, 3.1],
                },
            }
        }

    def reset_command_state(self):
        self.reset_count += 1

    def action_for_grasp_target(self, **kwargs):
        self.calls.append(kwargs)
        return [0.0] * 7 + [float(kwargs["gripper_command"])], {"ok": True}

    def action_for_grasp_target_physx_dls(self, **kwargs):
        self.calls.append({"backend": "physx_dls", **kwargs})
        return [0.5] * 7 + [float(kwargs["gripper_command"])], {"ok": True}

    def action_for_joint_target(self, **kwargs):
        self.calls.append(kwargs)
        return [0.25] * 7 + [float(kwargs["gripper_command"])], {"ok": True}


class _Readback:
    def read_task_sample(self):
        return {"joint_positions_rad": {"joint": 0.0}}


class _Adapter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _built(task_kind: str):
    scene = {
        "robot": object(),
        "bound_task_asset": object(),
        "robot_joint_wrench": object(),
    }
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene=scene,
            action_manager=SimpleNamespace(total_action_dim=8),
        ),
        reset=lambda *, seed: None,
    )
    return SimpleNamespace(
        env=env,
        plan={
            "task_kind": task_kind,
            "scenario": {"seed": 17},
            "cadence": {"control_frequency_hz": 15.0},
        },
        scene_asset_names={"task_object": "bound_task_asset"},
        camera_scene_names={
            "external": "arena_external_sensor",
            "wrist": "robot_wrist_sensor",
            "overview": "review_sensor",
        },
    )


@pytest.mark.parametrize(
    ("task_kind", "readback", "expected_source"),
    (
        ("rigid_pick_place", None, "native_rigid_body_readback"),
        (
            "articulated_open_close",
            _Readback(),
            "native_articulated_task_readback",
        ),
    ),
)
def test_factory_binds_original_and_articulated_fixtures_without_scene_names(
    monkeypatch: pytest.MonkeyPatch,
    task_kind: str,
    readback,
    expected_source: str,
) -> None:
    from blueprint_pipeline import native_task_episode_environment as module

    monkeypatch.setattr(module, "IsaacEpisodeAdapter", _Adapter)
    servo = _Servo()
    built = _built(task_kind)
    adapter, receipt = build_native_task_episode_environment(
        built=built,
        gripper_convention={
            "closed_command": 1.0,
            "open_command": 0.0,
            "finger_separation_m": {"0.0": 0.08, "1.0": 0.01},
        },
        servo=servo,
        task_readback=readback,
        to_tensor=lambda value: value,
    )

    assert receipt["task_kind"] == task_kind
    assert receipt["task_state_source"] == expected_source
    assert receipt["camera_scene_names"] == built.camera_scene_names
    assert adapter.kwargs["camera_scene_names"] == receipt["camera_scene_names"]
    assert (adapter.kwargs["rigid_task_object"] is None) == (
        task_kind == "articulated_open_close"
    )
    assert (adapter.kwargs["task_sample_callback"] is None) == (
        task_kind == "rigid_pick_place"
    )
    assert adapter.kwargs["simulation_step_seconds"] == pytest.approx(1.0 / 15.0)
    assert adapter.kwargs["joint_wrench_sensor"] is (
        built.env.unwrapped.scene["robot_joint_wrench"]
    )
    assert adapter.kwargs["grasp_frame_pose_callback"] == (
        servo.current_grasp_frame_pose_world
    )
    assert receipt["joint_wrench_source"] == (
        "IsaacLab JointWrenchSensor force+torque"
    )
    assert receipt["joint_wrench_convention"] == "incoming_joint_frame"
    assert receipt["grasp_frame_pose_source"] == (
        "native_franka_pose_servo.measured_controlled_body_to_grasp_frame"
    )
    if task_kind == "articulated_open_close":
        sample = adapter.kwargs["task_sample_callback"]()
        assert sample["gripper_width_m"] == pytest.approx(0.06)
        assert sample["gripper_controlled_body_quaternion_world_xyzw"] == [
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        assert sample["gripper_finger_body_positions_world_m"] == {
            "left": [0.96, 2.0, 3.1],
            "right": [1.04, 2.0, 3.1],
        }
        assert sample["gripper_pad_centers_world_m"] == {
            "left": [0.97, 2.0, 3.1],
            "right": [1.03, 2.0, 3.1],
        }
        assert receipt["gripper_state_source"] == (
            "native_finger_body_pose_plus_probe_sealed_pad_offset_each_sample"
        )
    else:
        assert receipt["gripper_state_source"] is None

    action = adapter.kwargs["scripted_pose_action_callback"](
        target_position_world_m=[1.0, 2.0, 3.0],
        target_quaternion_world_xyzw=None,
        gripper_command=1.0,
        max_joint_delta_rad=0.03,
        max_joint_setpoint_lead_rad=0.2,
    )
    assert action == [0.0] * 7 + [1.0]
    assert servo.calls[-1]["target_grasp_frame_quaternion_world_xyzw"] == [
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def test_articulated_factory_requires_native_task_readback() -> None:
    with pytest.raises(
        NativeTaskEpisodeEnvironmentError,
        match="native_task_episode_task_readback_missing",
    ):
        build_native_task_episode_environment(
            built=_built("articulated_open_close"),
            gripper_convention={
                "closed_command": 1.0,
                "open_command": 0.0,
                "finger_separation_m": {"0.0": 0.08, "1.0": 0.01},
            },
            servo=_Servo(),
            task_readback=None,
            to_tensor=lambda value: value,
        )


def test_factory_replays_construction_global_ik_joint_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import native_task_episode_environment as module

    monkeypatch.setattr(module, "IsaacEpisodeAdapter", _Adapter)
    servo = _Servo()
    adapter, receipt = build_native_task_episode_environment(
        built=_built("articulated_open_close"),
        gripper_convention={
            "closed_command": 1.0,
            "open_command": 0.0,
            "finger_separation_m": {"0.0": 0.08, "1.0": 0.01},
        },
        servo=servo,
        task_readback=_Readback(),
        to_tensor=lambda value: value,
        scripted_pose_joint_targets=[
            {
                "phase_id": "prealign",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "joint_positions_rad": [0.1] * 7,
            }
        ],
    )

    action = adapter.kwargs["scripted_pose_action_callback"](
        target_position_world_m=[1.0, 2.0, 3.0],
        target_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        gripper_command=1.0,
        max_joint_delta_rad=0.03,
        max_joint_setpoint_lead_rad=0.2,
    )

    assert action == [0.25] * 7 + [1.0]
    assert servo.calls[-1]["target_joint_positions_rad"] == [0.1] * 7
    assert receipt["scripted_pose_source"] == (
        "construction_global_ik_joint_target_with_native_pose_fallback"
    )
    assert receipt["scripted_pose_joint_targets"][0]["phase_id"] == "prealign"


def test_factory_uses_cartesian_servo_before_precision_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reachable endpoint must not turn contact into a joint-space arc."""

    from blueprint_pipeline import native_task_episode_environment as module

    monkeypatch.setattr(module, "IsaacEpisodeAdapter", _Adapter)
    servo = _Servo()
    adapter, receipt = build_native_task_episode_environment(
        built=_built("articulated_open_close"),
        gripper_convention={
            "closed_command": 1.0,
            "open_command": 0.0,
            "finger_separation_m": {"0.0": 0.08, "1.0": 0.01},
        },
        servo=servo,
        task_readback=_Readback(),
        to_tensor=lambda value: value,
        scripted_pose_joint_targets=[
            {
                "phase_id": "approach",
                "target_position_world_m": [1.0, 2.0, 2.9],
                "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "joint_positions_rad": [0.1] * 7,
            },
            {
                "phase_id": "contact_open",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "joint_positions_rad": [0.2] * 7,
            },
        ],
    )

    approach_action = adapter.kwargs["scripted_pose_action_callback"](
        target_position_world_m=[1.0, 2.0, 2.9],
        target_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        gripper_command=0.0,
        max_joint_delta_rad=0.03,
        max_joint_setpoint_lead_rad=0.2,
    )
    assert approach_action == [0.5] * 7 + [0.0]
    assert servo.calls[-1]["backend"] == "physx_dls"
    assert servo.calls[-1]["preferred_posture_joint_positions_rad"] == [0.1] * 7

    action = adapter.kwargs["scripted_pose_action_callback"](
        target_position_world_m=[1.0, 2.0, 3.0],
        target_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        gripper_command=0.0,
        max_joint_delta_rad=0.03,
        max_joint_setpoint_lead_rad=0.2,
    )

    assert action == [0.5] * 7 + [0.0]
    assert servo.calls[-1]["backend"] == "physx_dls"
    assert servo.calls[-1]["target_position_world_m"] == [1.0, 2.0, 3.0]
    assert servo.calls[-1]["preferred_posture_joint_positions_rad"] == [0.2] * 7
    assert "target_joint_positions_rad" not in servo.calls[-1]
    assert receipt["scripted_pose_source"] == (
        "global_ik_free_space_with_live_physx_jacobian_precision_servo_"
        "and_full_pose_nullspace_joint_limit_avoidance"
    )
    assert receipt["cartesian_contact_posture_source"] == (
        "selected_global_ik_joint_target_projected_through_live_physx_"
        "full_pose_jacobian_nullspace"
    )
    assert receipt["cartesian_precision_joint_limit_avoidance_source"] == (
        "isaaclab_pink_combined_task_jacobian_nullspace_projection"
    )
    assert receipt["cartesian_contact_posture_nullspace_gain"] == pytest.approx(0.20)
    assert receipt["cartesian_precision_joint_limit_avoidance_gain"] == pytest.approx(0.20)
    assert receipt["cartesian_precision_joint_limit_avoidance_margin_rad"] == pytest.approx(0.30)
    assert receipt["cartesian_contact_phase_ids"] == ["contact_open"]
    assert receipt["cartesian_contact_physx_dls_phase_ids"] == [
        "approach",
        "contact_open",
    ]


def test_factory_rejects_malformed_construction_joint_target() -> None:
    with pytest.raises(
        NativeTaskEpisodeEnvironmentError,
        match="native_task_episode_scripted_joint_target_invalid:0",
    ):
        build_native_task_episode_environment(
            built=_built("articulated_open_close"),
            gripper_convention={
                "closed_command": 1.0,
                "open_command": 0.0,
                "finger_separation_m": {"0.0": 0.08, "1.0": 0.01},
            },
            servo=_Servo(),
            task_readback=_Readback(),
            to_tensor=lambda value: value,
            scripted_pose_joint_targets=[
                {
                    "phase_id": "approach",
                    "target_position_world_m": [1.0, 2.0, 3.0],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "joint_positions_rad": [0.1] * 6,
                }
            ],
        )


def test_factory_rejects_an_ambiguous_gripper_probe() -> None:
    with pytest.raises(
        NativeTaskEpisodeEnvironmentError,
        match="native_task_episode_gripper_convention_invalid",
    ):
        build_native_task_episode_environment(
            built=_built("rigid_pick_place"),
            gripper_convention={
                "closed_command": 0.0,
                "open_command": 0.0,
                "finger_separation_m": {"0.0": 0.08},
            },
            servo=_Servo(),
            task_readback=None,
            to_tensor=lambda value: value,
        )
