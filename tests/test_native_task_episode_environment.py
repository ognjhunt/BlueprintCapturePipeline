from __future__ import annotations

from types import SimpleNamespace

import pytest

from blueprint_pipeline.native_task_episode_environment import (
    NativeRigidScoringEnvironment,
    NativeTaskEpisodeEnvironmentError,
    build_native_task_episode_environment,
)


class _RigidEpisodeEnvironment:
    def read_object_sample(self):
        return {
            "task_object_pose_world": [1.0, 2.0, 0.8, 0.0, 0.0, 0.0, 1.0],
            "gripper_width_m": 0.08,
        }

    def reset(self):
        return None


class _RigidNativeReadback:
    def __init__(self, **overrides):
        self.overrides = overrides

    def read_task_sample(self):
        return {
            "task_scoring_pose_world": [
                1.1,
                2.1,
                0.8,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "task_robot_contact_peak_force_n": 0.0,
            "task_support_contact_peak_force_n": 1.0,
            "task_scene_collision_peak_force_n": 0.0,
            "robot_scene_contact_peak_force_n": 0.0,
            "robot_task_forbidden_collision_peak_force_n": 0.0,
            "locked_joint_containment_violation": False,
            **self.overrides,
        }


def _rigid_scoring_task_spec():
    return {
        "task_contact_minimum_force_n": 0.5,
        "collision_failure_minimum_force_n": 5.0,
        "workspace_position_bounds_world_m": {
            "minimum": [0.0, 1.0, 0.7],
            "maximum": [2.0, 3.0, 1.2],
        },
    }


def test_rigid_scoring_environment_joins_native_safety_and_support_readback():
    base = _RigidEpisodeEnvironment()
    environment = NativeRigidScoringEnvironment(
        environment=base,
        task_readback=_RigidNativeReadback(),
        task_spec=_rigid_scoring_task_spec(),
    )

    sample = environment.read_object_sample()

    assert sample["task_object_pose_world"] == [
        1.1,
        2.1,
        0.8,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert sample["task_contact_active"] is False
    assert sample["support_contact_active"] is True
    assert sample["robot_collision_failure"] is False
    assert sample["forbidden_robot_task_collision_failure"] is False
    assert sample["scene_collision_failure"] is False
    assert sample["containment_violation"] is False
    assert sample["locked_joint_containment_violation"] is False
    assert sample["collision_failure_minimum_force_n"] == 5.0
    assert environment.reset() is None


def test_rigid_scoring_environment_retains_measured_contact_pair_identity():
    pairs = [
        {
            "robot_link_id": "panda_link7",
            "task_link_id": "mug_body",
            "sensor_instance_id": "forbidden__panda_link7__mug_body",
        }
    ]
    environment = NativeRigidScoringEnvironment(
        environment=_RigidEpisodeEnvironment(),
        task_readback=_RigidNativeReadback(
            robot_task_forbidden_contact_pairs=pairs
        ),
        task_spec=_rigid_scoring_task_spec(),
    )

    sample = environment.read_object_sample()

    assert sample["robot_task_forbidden_contact_pairs"] == pairs


def test_rigid_scoring_environment_refuses_missing_native_safety_channel():
    environment = NativeRigidScoringEnvironment(
        environment=_RigidEpisodeEnvironment(),
        task_readback=_RigidNativeReadback(
            robot_task_forbidden_collision_peak_force_n=None
        ),
        task_spec=_rigid_scoring_task_spec(),
    )

    with pytest.raises(
        NativeTaskEpisodeEnvironmentError,
        match="native_task_rigid_scoring_sample_invalid",
    ):
        environment.read_object_sample()


def _initial_support_environment():
    spec = _rigid_scoring_task_spec()
    spec.update(start_pose_world=[1.1, 2.1, 0.8, 0., 0., 0., 1.],
        minimum_lift_m=0.05, reset_translation_tolerance_m=0.002,
        maximum_task_contact_force_n=10., initial_source_support={
            "scene_prim_paths": ["/Scene/cabinet"],
            "support_plane_digest": "sha256:" + "a" * 64,
            "contact_permission": "initial_pickup_until_first_separation_or_lift"})
    readback = _RigidNativeReadback(task_initial_support_contact_peak_force_n=0.,
                                   task_support_contact_peak_force_n=0.)
    return NativeRigidScoringEnvironment(environment=_RigidEpisodeEnvironment(),
        task_readback=readback, task_spec=spec), readback


def test_initial_source_support_allows_pickup_but_not_return_after_separation():
    environment, readback = _initial_support_environment()
    # An initialized zero-force reset sample does not consume pickup permission.
    assert environment.read_object_sample()["initial_source_support_contact_permitted"]
    readback.overrides["task_initial_support_contact_peak_force_n"] = 6.
    initial = environment.read_object_sample()
    assert initial["initial_source_support_contact_active"]
    assert not initial["scene_collision_failure"]
    assert not initial["support_contact_active"]  # Destination remains tray-only.
    assert initial["task_initial_support_contact_peak_force_n"] == 6.
    readback.overrides["task_initial_support_contact_peak_force_n"] = 0.
    assert not environment.read_object_sample()["initial_source_support_contact_permitted"]
    readback.overrides["task_initial_support_contact_peak_force_n"] = 6.
    returned = environment.read_object_sample()
    assert returned["initial_source_support_collision_failure"]
    assert returned["scene_collision_failure"]
    assert returned["task_scene_collision_peak_force_n"] == 6.
    assert returned["task_non_support_scene_collision_peak_force_n"] == 0.
    environment.reset()
    assert not environment.read_object_sample()["scene_collision_failure"]


@pytest.mark.parametrize("mutation", ["other_background", "excess_force", "lift", "missing", "nan"])
def test_initial_support_never_hides_forbidden_or_unmeasured_contacts(mutation):
    environment, readback = _initial_support_environment()
    readback.overrides["task_initial_support_contact_peak_force_n"] = 6.
    if mutation == "other_background":
        readback.overrides["task_scene_collision_peak_force_n"] = 6.
    elif mutation == "excess_force":
        readback.overrides["task_initial_support_contact_peak_force_n"] = 11.
    elif mutation == "lift":
        readback.overrides["task_scoring_pose_world"] = [1.1, 2.1, .86, 0., 0., 0., 1.]
    else:
        readback.overrides["task_initial_support_contact_peak_force_n"] = None if mutation == "missing" else float("nan")
        with pytest.raises(NativeTaskEpisodeEnvironmentError, match="initial_support_readback"):
            environment.read_object_sample()
        return
    assert environment.read_object_sample()["scene_collision_failure"]


class _Servo:
    def __init__(self):
        self.reset_count = 0
        self.calls = []
        self.body_pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        self.grasp_frame_pose = [1.0, 2.0, 3.1, 0.0, 0.0, 0.0, 1.0]

    def current_body_pose_world(self):
        return list(self.body_pose)

    def current_grasp_frame_pose_world(self):
        return list(self.grasp_frame_pose)

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


class _MutableState:
    def __init__(self, values):
        self.values = [list(row) for row in values]

    def clone(self):
        return _MutableState(self.values)

    def zero_(self):
        for row in self.values:
            for index in range(len(row)):
                row[index] = 0.0

    def __getitem__(self, key):
        row, column = key
        return self.values[row][column]

    def __setitem__(self, key, value):
        row, column = key
        self.values[row][column] = float(value)


class _WritableArticulation:
    def __init__(self, joint_names):
        self.joint_names = list(joint_names)
        self.data = SimpleNamespace(
            joint_pos=_MutableState([[0.0] * len(joint_names)]),
            joint_vel=_MutableState([[1.0] * len(joint_names)]),
        )
        self.writes = []

    def write_joint_state_to_sim(self, position, velocity):
        self.writes.append((position.values, velocity.values))


def _built(task_kind: str):
    camera = lambda position: SimpleNamespace(  # noqa: E731
        data=SimpleNamespace(
            pos_w=[list(position)],
            quat_w_opengl=[[0.0, 0.0, 0.0, 1.0]],
        )
    )
    scene = {
        "robot": object(),
        "bound_task_asset": object(),
        "robot_joint_wrench": object(),
        "arena_external_sensor": camera([4.0, 5.0, 6.0]),
        "robot_wrist_sensor": camera([1.2, 2.0, 3.2]),
        "review_sensor": camera([7.0, 8.0, 9.0]),
    }
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene=scene,
            action_manager=SimpleNamespace(total_action_dim=8),
        ),
        reset=lambda *, seed: None,
    )
    plan = {
        "task_kind": task_kind,
        "scenario": {"seed": 17},
        "cadence": {"control_frequency_hz": 15.0},
    }
    if task_kind == "rigid_pick_place":
        plan["task_spec"] = {
            "interaction_affordance": {
                "asset_root_from_scoring_frame": {
                    "position_m": [0.0, 0.0, 0.06400000303983688],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        }
    return SimpleNamespace(
        env=env,
        plan=plan,
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
    if task_kind == "rigid_pick_place":
        expected_offset = {
            "position_m": [0.0, 0.0, 0.06400000303983688],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        assert adapter.kwargs["rigid_task_scoring_frame_offset"] == expected_offset
        assert receipt["rigid_task_pose_binding"] == {
            "asset_root_pose_retained": True,
            "task_object_pose_world_source": (
                "asset_root_pose_world_composed_with_interaction_affordance_"
                "asset_root_from_scoring_frame"
            ),
            "scoring_frame_offset": expected_offset,
        }
    else:
        assert adapter.kwargs["rigid_task_scoring_frame_offset"] is None
        assert receipt["rigid_task_pose_binding"] is None
    camera_pose = adapter.kwargs["camera_pose_callback"]
    assert camera_pose("arena_external_sensor") is None
    reset_camera_position, reset_camera_quaternion = camera_pose(
        "robot_wrist_sensor"
    )
    assert reset_camera_position == pytest.approx([1.2, 2.0, 3.2])
    assert reset_camera_quaternion == pytest.approx([0.0, 0.0, 0.0, 1.0])
    servo.body_pose = [2.0, 2.5, 3.0, 0.0, 0.0, 0.0, 1.0]
    servo.grasp_frame_pose = [2.0, 2.5, 3.1, 0.0, 0.0, 0.0, 1.0]
    moved_position, moved_quaternion = camera_pose("robot_wrist_sensor")
    assert moved_position == pytest.approx([2.2, 2.5, 3.2])
    assert moved_quaternion == pytest.approx([0.0, 0.0, 0.0, 1.0])
    half_sqrt = 2**-0.5
    servo.body_pose = [
        2.0,
        2.5,
        3.0,
        0.0,
        0.0,
        half_sqrt,
        half_sqrt,
    ]
    rotated_position, rotated_quaternion = camera_pose("robot_wrist_sensor")
    assert rotated_position == pytest.approx([2.0, 2.7, 3.2])
    assert rotated_quaternion == pytest.approx(
        [0.0, 0.0, half_sqrt, half_sqrt]
    )
    assert receipt["camera_world_pose_bindings"]["wrist"] == {
        "scene_name": "robot_wrist_sensor",
        "source": (
            "live_controlled_body_plus_reset_measured_rigid_mount_offset"
        ),
        "recomputed_each_observation": True,
        "sensor_buffer_static_pose_workaround": True,
        "mount_offset_position_controlled_body_m": pytest.approx([0.2, 0.0, 0.2]),
        "mount_offset_quaternion_controlled_body_xyzw": pytest.approx(
            [0.0, 0.0, 0.0, 1.0]
        ),
    }
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
    diagnostic_action = adapter.kwargs["scripted_pose_action_callback"](
        target_position_world_m=[1.1, 2.0, 3.0],
        target_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        gripper_command=1.0,
        max_joint_delta_rad=0.03,
        max_joint_setpoint_lead_rad=0.2,
        preferred_posture_joint_positions_rad=[0.1] * 7,
    )
    assert diagnostic_action == [0.5] * 7 + [1.0]
    assert servo.calls[-1]["backend"] == "physx_dls"
    assert servo.calls[-1]["preferred_posture_joint_positions_rad"] == [
        0.1
    ] * 7


def test_rigid_factory_refuses_a_missing_scoring_frame_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import native_task_episode_environment as module

    monkeypatch.setattr(module, "IsaacEpisodeAdapter", _Adapter)
    built = _built("rigid_pick_place")
    del built.plan["task_spec"]["interaction_affordance"][
        "asset_root_from_scoring_frame"
    ]

    with pytest.raises(
        NativeTaskEpisodeEnvironmentError,
        match="native_task_episode_rigid_scoring_frame_transform_invalid",
    ):
        build_native_task_episode_environment(
            built=built,
            gripper_convention={
                "closed_command": 1.0,
                "open_command": 0.0,
                "finger_separation_m": {"0.0": 0.08, "1.0": 0.01},
            },
            servo=_Servo(),
            task_readback=None,
            to_tensor=lambda value: value,
        )


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


def test_factory_checkpoint_writer_sets_arm_and_native_task_joint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import native_task_episode_environment as module

    monkeypatch.setattr(module, "IsaacEpisodeAdapter", _Adapter)
    built = _built("articulated_open_close")
    robot = _WritableArticulation(
        [*[f"panda_joint{index}" for index in range(1, 8)], "finger_joint"]
    )
    task = _WritableArticulation(["door_hinge", "locked_hinge"])
    built.env.unwrapped.scene["robot"] = robot
    built.env.unwrapped.scene["bound_task_asset"] = task
    built.plan["task_sample_binding"] = {
        "native_joint_names": {
            "door": "door_hinge",
            "locked": "locked_hinge",
        }
    }
    adapter, _receipt = build_native_task_episode_environment(
        built=built,
        gripper_convention={
            "closed_command": 1.0,
            "open_command": 0.0,
            "finger_separation_m": {"0.0": 0.08, "1.0": 0.01},
        },
        servo=_Servo(),
        task_readback=_Readback(),
        to_tensor=lambda value: value,
    )

    adapter.kwargs["diagnostic_checkpoint_reset_callback"](
        [0.1] * 7, {"door": 0.4}
    )

    assert robot.writes[-1][0][0][:7] == [0.1] * 7
    assert robot.writes[-1][1][0] == [0.0] * 8
    assert task.writes[-1][0][0] == [0.4, 0.0]
    assert task.writes[-1][1][0] == [0.0, 0.0]


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
    assert receipt["cartesian_contact_phase_controller_bindings"] == [
        {
            "phase_id": "approach",
            "controller": "live_physx_full_pose_dls",
            "preferred_posture_source": "selected_global_ik_joint_target",
            "recovery_target_bias_preserves_controller": True,
        },
        {
            "phase_id": "contact_open",
            "controller": "live_physx_full_pose_dls",
            "preferred_posture_source": "selected_global_ik_joint_target",
            "recovery_target_bias_preserves_controller": True,
        },
    ]


def test_factory_uses_physx_dls_for_precision_pose_without_offsim_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing optional PINK seed must not switch the live controller to PINK."""

    from blueprint_pipeline import native_task_episode_environment as module

    monkeypatch.setattr(module, "IsaacEpisodeAdapter", _Adapter)
    servo = _Servo()
    target = {
        "phase_id": "contact_open",
        "target_position_world_m": [1.0, 2.0, 3.0],
        "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
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
        scripted_pose_joint_targets=[],
        scripted_pose_phase_targets=[target],
    )

    action = adapter.kwargs["scripted_pose_action_callback"](
        phase_id="contact_open",
        # Retry calibration deliberately biases the authored target.  Phase
        # identity, not exact float equality with the original waypoint, must
        # keep the live controller authoritative.
        target_position_world_m=[1.01, 2.0, 3.0],
        target_quaternion_world_xyzw=target[
            "target_quaternion_world_xyzw"
        ],
        gripper_command=0.0,
        max_joint_delta_rad=0.03,
        max_joint_setpoint_lead_rad=0.2,
    )

    assert action == [0.5] * 7 + [0.0]
    assert servo.calls[-1]["backend"] == "physx_dls"
    assert servo.calls[-1]["preferred_posture_joint_positions_rad"] is None
    assert receipt["scripted_pose_source"] == (
        "live_physx_jacobian_precision_servo_without_offsim_posture_seed"
    )
    assert receipt["cartesian_contact_phase_ids"] == ["contact_open"]
    assert receipt["cartesian_contact_physx_dls_phase_ids"] == ["contact_open"]
    assert receipt["cartesian_contact_posture_source"] == (
        "no_offsim_posture_seed_live_physx_full_pose_dls"
    )
    assert receipt["cartesian_contact_phase_controller_bindings"] == [
        {
            "phase_id": "contact_open",
            "controller": "live_physx_full_pose_dls",
            "preferred_posture_source": None,
            "recovery_target_bias_preserves_controller": True,
        }
    ]
    assert receipt["cartesian_contact_posture_nullspace_gain"] is None
    assert receipt["cartesian_precision_joint_limit_avoidance_gain"] == pytest.approx(
        0.20
    )
    assert receipt["cartesian_precision_joint_limit_avoidance_margin_rad"] == (
        pytest.approx(0.30)
    )


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
