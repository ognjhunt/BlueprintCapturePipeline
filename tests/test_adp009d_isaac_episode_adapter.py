from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.adp009d_droid_observation import (
    DROID_EXTERIOR_VIEW_1,
    DROID_WRIST_VIEW,
)
from blueprint_pipeline.adp009d_contact_envelope import canonical_contact_envelope
from blueprint_pipeline.adp009d_isaac_episode_adapter import (
    CAMERA_VIEW_BINDING,
    FINGER_BODIES,
    FINGER_TOOL_FRAME_LOCAL_OFFSET_M,
    FINGER_TOOL_FRAME_SOURCE,
    GRIPPER_PHYSICAL_FULL_OPENING_M,
    IsaacEpisodeAdapter,
    IsaacEpisodeAdapterError,
    bounded_grasp_frame_target_for_task_orientation,
    controlled_body_pose_for_grasp_frame_target,
    describe_adapter,
    grasp_frame_target_for_task_space_strategy,
    rgb_from_camera_output,
    rotation_row_major_from_quaternion_xyzw,
    semantic_finger_tool_midpoint_world_m,
    signed_point_to_vertical_cylinder_clearance_m,
    validate_adapter_bindings,
)


def test_signed_cylinder_clearance_is_pose_aware_and_signed() -> None:
    pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    assert signed_point_to_vertical_cylinder_clearance_m(
        point_world_m=[0.04, 0.0, 0.0],
        cylinder_pose_world_xyzw=pose,
        radius_m=0.03,
        height_m=0.1,
    ) == pytest.approx(0.01)
    assert signed_point_to_vertical_cylinder_clearance_m(
        point_world_m=[0.02, 0.0, 0.0],
        cylinder_pose_world_xyzw=pose,
        radius_m=0.03,
        height_m=0.1,
    ) == pytest.approx(-0.01)


def test_semantic_finger_midpoint_applies_pinned_local_tool_offsets() -> None:
    midpoint = semantic_finger_tool_midpoint_world_m(
        left_finger_pose_world_xyzw=[
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ],
        right_finger_pose_world_xyzw=[
            0.06, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )

    assert midpoint == pytest.approx([0.03, 0.0, 1.046])


def test_semantic_finger_midpoint_rotates_each_local_tool_offset() -> None:
    midpoint = semantic_finger_tool_midpoint_world_m(
        left_finger_pose_world_xyzw=[
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        ],
        right_finger_pose_world_xyzw=[
            0.06, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        ],
    )

    # A 180 degree rotation about X maps each +local-Z tool offset to world -Z.
    assert midpoint == pytest.approx([0.03, 0.0, 0.954])


def test_semantic_finger_midpoint_refuses_nonrigid_body_orientations() -> None:
    with pytest.raises(
        IsaacEpisodeAdapterError,
        match="isaac_episode_finger_tool_frame_pose_invalid",
    ):
        semantic_finger_tool_midpoint_world_m(
            left_finger_pose_world_xyzw=[
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0,
            ],
            right_finger_pose_world_xyzw=[
                0.06, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            ],
        )


def test_grasp_frame_target_retains_the_measured_full_tool_offset() -> None:
    target_body, target_quaternion = controlled_body_pose_for_grasp_frame_target(
        current_body_position_world_m=[1.0, 2.0, 3.0],
        current_body_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        current_grasp_frame_position_world_m=[1.2, 1.9, 2.7],
        target_grasp_frame_position_world_m=[4.0, 5.0, 6.0],
        target_body_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
    )

    assert target_body == pytest.approx([3.8, 5.1, 6.3])
    assert target_quaternion == [0.0, 0.0, 0.0, 1.0]


def test_grasp_frame_target_rejects_a_nonrigid_orientation() -> None:
    with pytest.raises(
        IsaacEpisodeAdapterError,
        match="isaac_episode_grasp_frame_transform_invalid",
    ):
        controlled_body_pose_for_grasp_frame_target(
            current_body_position_world_m=[1.0, 2.0, 3.0],
            current_body_quaternion_world_xyzw=[0.0, 0.0, 0.0, 2.0],
            current_grasp_frame_position_world_m=[1.2, 1.9, 2.7],
            target_grasp_frame_position_world_m=[4.0, 5.0, 6.0],
            target_body_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        )


def test_bounded_task_space_target_holds_translation_until_orientation_is_safe() -> None:
    half_angle = math.radians(9.1716) / 2.0

    result = bounded_grasp_frame_target_for_task_orientation(
        current_position_world_m=[3.468, -3.310, 0.946],
        current_quaternion_world_xyzw=[
            0.0,
            0.0,
            math.sin(half_angle),
            math.cos(half_angle),
        ],
        target_position_world_m=[3.468, -3.310, 0.611],
        target_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        max_translation_step_m=0.01,
        orientation_tolerance_deg=2.0,
    )

    assert result["position_world_m"] == [3.468, -3.310, 0.946]
    assert result["orientation_error_deg"] == pytest.approx(9.1716)
    assert result["translation_requested_m"] == pytest.approx(0.335)
    assert result["translation_step_m"] == 0.0
    assert result["translation_held_for_orientation"] is True


def test_bounded_task_space_target_steps_locally_toward_large_descent() -> None:
    result = bounded_grasp_frame_target_for_task_orientation(
        current_position_world_m=[3.468, -3.310, 0.946],
        current_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        target_position_world_m=[3.468, -3.310, 0.611],
        target_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        max_translation_step_m=0.01,
        orientation_tolerance_deg=2.0,
    )

    assert result["position_world_m"] == pytest.approx(
        [3.468, -3.310, 0.936]
    )
    assert result["orientation_error_deg"] == 0.0
    assert result["translation_step_m"] == 0.01
    assert result["translation_held_for_orientation"] is False


def test_bounded_task_space_target_rejects_unbounded_step_contract() -> None:
    with pytest.raises(
        IsaacEpisodeAdapterError,
        match="isaac_episode_bounded_task_space_target_invalid",
    ):
        bounded_grasp_frame_target_for_task_orientation(
            current_position_world_m=[0.0, 0.0, 0.0],
            current_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
            target_position_world_m=[0.0, 0.0, 1.0],
            target_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
            max_translation_step_m=0.0,
            orientation_tolerance_deg=2.0,
        )


def test_direct_pregrasp_strategy_preserves_the_proven_coupled_pose_target() -> None:
    orientation_error_deg = 151.8895149465545
    half_angle = math.radians(orientation_error_deg) / 2.0
    current = [3.5542179326306393, -3.0661433904184823, 0.8331015124076399]
    target = [3.4681748, -3.3100837, 0.9464650138348478]

    result = grasp_frame_target_for_task_space_strategy(
        current_position_world_m=current,
        current_quaternion_world_xyzw=[
            math.cos(half_angle),
            math.sin(half_angle),
            0.0,
            0.0,
        ],
        target_position_world_m=target,
        target_quaternion_world_xyzw=[1.0, 0.0, 0.0, 0.0],
        max_translation_step_m=0.01,
        orientation_tolerance_deg=2.0,
        task_space_translation_strategy="direct_global_pose_target",
    )

    assert result["orientation_error_deg"] == pytest.approx(
        orientation_error_deg
    )
    assert result["position_world_m"] == target
    assert result["translation_step_m"] == pytest.approx(math.dist(current, target))
    assert result["translation_held_for_orientation"] is False
    assert result["task_space_translation_strategy"] == "direct_global_pose_target"


def test_task_space_strategy_rejects_an_unregistered_runtime_override() -> None:
    with pytest.raises(
        IsaacEpisodeAdapterError,
        match="isaac_episode_task_space_translation_strategy_invalid",
    ):
        grasp_frame_target_for_task_space_strategy(
            current_position_world_m=[0.0, 0.0, 0.0],
            current_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
            target_position_world_m=[0.0, 0.0, 1.0],
            target_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
            max_translation_step_m=0.01,
            orientation_tolerance_deg=2.0,
            task_space_translation_strategy="runtime_guess",
        )


def test_grasp_frame_target_rotates_the_measured_full_offset_into_task_orientation() -> None:
    target_body, target_quaternion = controlled_body_pose_for_grasp_frame_target(
        current_body_position_world_m=[1.0, 2.0, 3.0],
        current_body_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        current_grasp_frame_position_world_m=[1.2, 1.9, 2.7],
        target_grasp_frame_position_world_m=[4.0, 5.0, 6.0],
        target_body_quaternion_world_xyzw=[1.0, 0.0, 0.0, 0.0],
    )

    # A 180 degree rotation about body X preserves X and reverses Y/Z.
    assert target_body == pytest.approx([3.8, 4.9, 5.7])
    assert target_quaternion == [1.0, 0.0, 0.0, 0.0]


def test_v89_camera_orientation_was_unreachable_but_task_orientation_is_reachable() -> None:
    body = [3.468174695968628, -3.1697866916656494, 0.7484458684921265]
    camera_aim_quaternion = [
        -0.2917867997481312,
        -0.23988355674504283,
        0.4887041767341441,
        -0.7864378998615812,
    ]
    finger_midpoint = [
        3.3785593509674072,
        -2.9842019081115723,
        0.7934765517711639,
    ]
    pregrasp = [3.4681748, -3.3100837, 0.9464650138348478]
    robot_base = [3.4681748, -2.8100837, 0.2766791]

    held_body, _ = controlled_body_pose_for_grasp_frame_target(
        current_body_position_world_m=body,
        current_body_quaternion_world_xyzw=camera_aim_quaternion,
        current_grasp_frame_position_world_m=finger_midpoint,
        target_grasp_frame_position_world_m=pregrasp,
        target_body_quaternion_world_xyzw=camera_aim_quaternion,
    )
    task_body, _ = controlled_body_pose_for_grasp_frame_target(
        current_body_position_world_m=body,
        current_body_quaternion_world_xyzw=camera_aim_quaternion,
        current_grasp_frame_position_world_m=finger_midpoint,
        target_grasp_frame_position_world_m=pregrasp,
        target_body_quaternion_world_xyzw=[1.0, 0.0, 0.0, 0.0],
    )

    assert math.dist(held_body, robot_base) > 0.855
    assert math.dist(task_body, robot_base) < 0.855


def test_runtime_applies_the_preregistered_task_orientation_before_native_ik() -> None:
    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    callback = source[source.index("def _scripted_pose_action_callback") :]
    assert "scripted_control_task_orientation_missing" in callback
    assert "target_body_quaternion_world_xyzw=(" in callback
    assert "target_quaternion_world_xyzw" in callback
    assert "semantic_finger_tool_midpoint_world_m(" in callback
    assert '"raw_finger_body_midpoint_world_m"' in callback
    assert "grasp_frame_target_for_task_space_strategy(" in callback
    assert "task_space_translation_strategy" in callback
    assert "CONTROL_PLAN_FILENAME" in callback


class _Tensor(list):
    """Enough of a tensor for the adapter's indexing, without torch."""

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def _to_torch(value):
    return value


class _Data:
    def __init__(self, body_names, joint_pos, body_pose_w, joint_limits):
        self.body_names = body_names
        self.joint_pos = joint_pos
        self.body_pose_w = body_pose_w
        self.joint_limits = joint_limits
        if joint_pos is not None:
            shape = np.asarray(joint_pos).shape
            self.joint_vel = np.zeros(shape)
            self.joint_pos_target = np.asarray(joint_pos).copy()
            self.computed_torque = np.zeros(shape)
            self.applied_torque = np.zeros(shape)
            self.joint_effort_limits = np.tile(
                np.array([[87.0] * 4 + [12.0] * (shape[1] - 4)]),
                (shape[0], 1),
            )
            self.body_incoming_joint_wrench_b = np.zeros(
                (shape[0], len(body_names), 6)
            )


class _Robot:
    def __init__(self):
        bodies = [
            "panda_link0", "panda_link7", "base_link",
            "left_inner_finger", "right_inner_finger",
        ]
        # Fingers 0.06 m apart in x, so separation is exactly 0.06.
        poses = np.zeros((1, len(bodies), 7), dtype=float)
        poses[0, bodies.index("base_link"), :7] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        poses[0, bodies.index("left_inner_finger"), :7] = [
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ]
        poses[0, bodies.index("right_inner_finger"), :7] = [
            0.06, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ]
        self.data = _Data(
            body_names=bodies,
            joint_pos=np.linspace(0.0, 0.6, 13).reshape(1, 13),
            body_pose_w=poses,
            joint_limits=np.tile(np.array([[-2.9, 2.9]]), (1, 13, 1)),
        )


class _Can:
    def __init__(self):
        self.data = _Data(None, None, None, None)
        self.data.root_pose_w = np.array(
            [[3.468, -3.310, 0.526, 0.0, 0.0, 0.0, 1.0]], dtype=float
        )


class _ContactSensor:
    def __init__(self, body_names):
        self.body_names = list(body_names)
        forces = np.zeros((1, len(body_names), 3), dtype=float)
        forces[0, body_names.index("left_inner_finger")] = [3.0, 4.0, 0.0]
        forces[0, body_names.index("right_inner_finger")] = [0.0, 0.0, 6.0]
        self.data = type("ContactData", (), {"net_forces_w": forces})()


class _Camera:
    def __init__(self, channels=4):
        frame = np.zeros((1, 720, 1280, channels), dtype=np.uint8)
        frame[..., 0] = 200
        if channels == 4:
            frame[..., 3] = 255
        self.data = type(
            "D",
            (),
            {
                "output": {"rgb": _Tensor(frame)},
                "intrinsic_matrices": np.array(
                    [[[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]]]
                ),
                "pos_w": np.array([[1.0, 2.0, 3.0]]),
                "quat_w_opengl": np.array([[0.0, 0.0, 0.0, 1.0]]),
            },
        )()
        self.cfg = type(
            "Cfg",
            (),
            {"spawn": type("Spawn", (), {"clipping_range": (0.01, 100.0)})()},
        )()


class _Scene(dict):
    pass


class _Env:
    def __init__(self, channels=4):
        self.reset_calls: list[int] = []
        self.stepped: list[list[float]] = []
        scene = _Scene(
            {
                "external_camera": _Camera(channels),
                "wrist_camera": _Camera(channels),
                "external_camera_2": _Camera(channels),
            }
        )
        self.unwrapped = type("U", (), {"scene": scene, "device": "cpu"})()

    def reset(self, seed=None):
        self.reset_calls.append(seed)

    def step(self, tensor):
        self.stepped.append([float(v) for v in np.asarray(tensor)[0]])


def _adapter(env=None, *, with_contact_sensor=False):
    robot = _Robot()
    return IsaacEpisodeAdapter(
        env=env or _Env(),
        robot=robot,
        approved_can=_Can(),
        action_dim=8,
        reset_seed=20260806,
        to_torch=_to_torch,
        gripper_closed_width_m=0.0,
        gripper_open_width_m=0.06,
        contact_envelope=canonical_contact_envelope(),
        contact_sensor=(
            _ContactSensor(robot.data.body_names) if with_contact_sensor else None
        ),
    )


def test_arm_dynamics_observation_retains_actuator_and_contact_readback() -> None:
    adapter = _adapter(with_contact_sensor=True)

    dynamics = adapter.read_arm_dynamics_observation()
    sample = adapter.read_object_sample()

    assert dynamics["schema_version"] == "adp009d_arm_dynamics_observation.v2"
    assert dynamics["contact_envelope"] == canonical_contact_envelope()
    assert dynamics["joint_effort_limit_nm"] == [87.0] * 4 + [12.0] * 3
    assert dynamics["body_contact_force_world_n"]["left_inner_finger"] == [
        3.0,
        4.0,
        0.0,
    ]
    assert len(dynamics["body_incoming_joint_wrench_body"]) == len(
        adapter._robot.data.body_names
    )
    assert sample["finger_contact_forces_n"] == [5.0, 6.0]


def test_arm_dynamics_observation_retains_typed_contact_gap_off_gpu() -> None:
    dynamics = _adapter().read_arm_dynamics_observation()

    assert dynamics["body_contact_force_world_n"] is None


def test_alpha_is_dropped_at_the_boundary_not_downstream() -> None:
    """A constant alpha channel has already once made a black render look alive."""

    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 3] = 255

    rgb = rgb_from_camera_output(rgba)

    assert rgb.shape == (4, 4, 3)
    assert rgb.dtype == np.uint8
    assert not rgb.any()
    # Already-RGB input passes through unchanged.
    assert rgb_from_camera_output(np.zeros((4, 4, 3), np.uint8)).shape == (4, 4, 3)


def test_malformed_camera_frames_are_refused() -> None:
    for bad in (np.zeros((4, 4)), np.zeros((4, 4, 2)), np.zeros((4, 4, 5))):
        with pytest.raises(IsaacEpisodeAdapterError):
            rgb_from_camera_output(bad)


def test_policy_inputs_carry_both_views_as_uint8_rgb() -> None:
    adapter = _adapter()

    inputs = adapter.read_policy_inputs()

    for view in (DROID_EXTERIOR_VIEW_1, DROID_WRIST_VIEW):
        assert inputs[view].shape == (720, 1280, 3)
        assert inputs[view].dtype == np.uint8
    assert len(inputs["joint_position"]) == 7
    assert inputs["gripper_position"] == pytest.approx(0.0, abs=1e-9)
    assert inputs["eef_9d"] == pytest.approx(
        [1.0, 2.0, 3.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0]
    )
    assert "overview" not in inputs
    assert set(adapter.read_evaluation_camera_inputs()) == {
        "external",
        "wrist",
        "overview",
    }


def test_isaac_xyzw_quaternion_is_converted_before_nvidia_frame_correction() -> None:
    """Regression for decoding pinned IsaacLab body poses in the wrong order."""

    half_sqrt = 2**-0.5

    rotation = rotation_row_major_from_quaternion_xyzw(
        [0.0, 0.0, half_sqrt, half_sqrt]
    )

    assert rotation == pytest.approx([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_gripper_width_is_probe_calibrated_physical_opening() -> None:
    """Link-origin separation is calibrated onto the 2F-85 jaw stroke."""

    adapter = _adapter()

    sample = adapter.read_object_sample()

    assert sample["gripper_width_m"] == pytest.approx(
        GRIPPER_PHYSICAL_FULL_OPENING_M, abs=1e-9
    )
    assert sample["gripper_body_separation_m"] == pytest.approx(0.06, abs=1e-9)
    assert sample["gripper_width_open_fraction_unclamped"] == pytest.approx(1.0)
    assert sample["gripper_width_calibration_clamped"] is False
    assert sample["gripper_body_midpoint_world_m"] == pytest.approx(
        [0.03, 0.0, 1.0], abs=1e-9
    )
    # The grasp frame is the midpoint between the semantic finger tools, not
    # the raw inner-finger body origins.
    assert sample["grasp_frame_position_world_m"] == pytest.approx(
        [0.03, 0.0, 1.046], abs=1e-9
    )
    assert sample["grasp_frame_calibration"] == {
        "frame_id": "probe_calibrated_finger_midpoint",
        "finger_tool_frame_local_offset_m": [0.0, 0.0, 0.046],
        "source": FINGER_TOOL_FRAME_SOURCE,
        "raw_body_midpoint_retained": True,
    }
    assert sample["controlled_body_name"] == "base_link"
    assert sample["controlled_body_pose_world"] == pytest.approx(
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    )


def test_linkage_overtravel_is_bounded_and_raw_measurement_is_retained() -> None:
    """Regression for v74 samples whose link origins exceeded the 85 mm stroke."""

    from blueprint_pipeline.adp009d_task_scoring import normalize_object_samples

    robot = _Robot()
    bodies = robot.data.body_names
    robot.data.body_pose_w[0, bodies.index("right_inner_finger"), :3] = [
        0.10,
        0.0,
        1.0,
    ]
    adapter = IsaacEpisodeAdapter(
        env=_Env(),
        robot=robot,
        approved_can=_Can(),
        action_dim=8,
        reset_seed=20260806,
        to_torch=_to_torch,
        gripper_closed_width_m=0.0,
        gripper_open_width_m=0.0831756591796875,
        contact_envelope=canonical_contact_envelope(),
    )

    sample = dict(adapter.read_object_sample())
    assert sample["gripper_body_separation_m"] == pytest.approx(0.10)
    assert sample["gripper_width_m"] == pytest.approx(
        GRIPPER_PHYSICAL_FULL_OPENING_M
    )
    assert sample["gripper_width_open_fraction_unclamped"] > 1.0
    assert sample["gripper_width_calibration_clamped"] is True
    sample["step_index"] = 0
    normalized = normalize_object_samples([sample], require_sealed_start_pose=False)
    assert normalized[0]["gripper_width_m"] == pytest.approx(
        GRIPPER_PHYSICAL_FULL_OPENING_M
    )


def test_droid_gripper_state_is_closed_fraction_from_the_measured_probe_span() -> None:
    robot = _Robot()
    bodies = robot.data.body_names
    robot.data.body_pose_w[0, bodies.index("right_inner_finger"), :3] = [
        0.03,
        0.0,
        1.0,
    ]
    adapter = IsaacEpisodeAdapter(
        env=_Env(),
        robot=robot,
        approved_can=_Can(),
        action_dim=8,
        reset_seed=20260806,
        to_torch=_to_torch,
        gripper_closed_width_m=0.0,
        gripper_open_width_m=0.06,
        contact_envelope=canonical_contact_envelope(),
    )

    assert adapter.read_policy_inputs()["gripper_position"] == pytest.approx(0.5)


def test_object_sample_matches_the_scorer_schema() -> None:
    from blueprint_pipeline.adp009d_task_scoring import normalize_object_samples

    adapter = _adapter()
    sample = dict(adapter.read_object_sample())
    sample["step_index"] = 0

    normalized = normalize_object_samples([sample], require_sealed_start_pose=False)
    assert normalized[0]["step_index"] == 0
    assert len(sample["can_pose_world"]) == 7


def test_a_wrong_width_action_never_reaches_the_simulator() -> None:
    env = _Env()
    adapter = _adapter(env)

    with pytest.raises(IsaacEpisodeAdapterError):
        adapter.step([0.0] * 7)
    assert env.stepped == []


def test_reset_uses_the_pinned_seed() -> None:
    env = _Env()
    adapter = _adapter(env)

    adapter.reset()

    assert env.reset_calls == [20260806]


def test_control_hold_metadata_and_injected_scripted_pose_share_native_action_seam() -> None:
    env = _Env()
    calls = []

    def scripted(**kwargs):
        calls.append(kwargs)
        return [0.2] * 7 + [kwargs["gripper_command"]]

    adapter = IsaacEpisodeAdapter(
        env=env,
        robot=_Robot(),
        approved_can=_Can(),
        action_dim=8,
        reset_seed=20260806,
        to_torch=_to_torch,
        gripper_closed_width_m=0.0,
        gripper_open_width_m=0.06,
        simulation_step_seconds=1.0 / 15.0,
        scripted_pose_action_callback=scripted,
        contact_envelope=canonical_contact_envelope(),
    )
    adapter.reset()

    hold = adapter.hold_action(gripper_command=1.0)
    assert hold[:7] == adapter.read_arm_joint_positions()
    assert hold[7] == 1.0
    scripted_action = adapter.scripted_action_for_pose(
        target_position_world_m=[1.0, 2.0, 3.0],
        target_quaternion_world_xyzw=[1.0, 0.0, 0.0, 0.0],
        gripper_command=0.0,
        max_joint_delta_rad=0.03,
        max_task_space_translation_step_m=0.01,
        orientation_tolerance_deg=2.0,
        task_space_translation_strategy=(
            "orientation_first_bounded_local_increment"
        ),
    )
    assert scripted_action == [0.2] * 7 + [0.0]
    assert calls[0]["max_joint_delta_rad"] == 0.03
    assert calls[0]["max_task_space_translation_step_m"] == 0.01
    assert calls[0]["orientation_tolerance_deg"] == 2.0
    assert calls[0]["task_space_translation_strategy"] == (
        "orientation_first_bounded_local_increment"
    )
    metadata = adapter.read_control_observation_metadata()
    assert metadata["simulation_time_s"] == 0.0
    assert metadata["timestamp_ns"] == 0
    assert set(metadata["calibrations"]) == {"external", "wrist", "overview"}
    assert metadata["calibrations"]["external"]["resolution"] == [1280, 720]
    assert metadata["calibrations"]["external"]["world_from_camera"] == [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_control_metadata_uses_live_wrist_mount_pose_callback() -> None:
    """A moving render mount must not retain Isaac's initialization-only pose."""

    calls: list[str] = []

    def camera_pose(camera_name: str):
        calls.append(camera_name)
        if camera_name == "wrist_camera":
            return [4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 1.0]
        return None

    adapter = IsaacEpisodeAdapter(
        env=_Env(),
        robot=_Robot(),
        approved_can=_Can(),
        action_dim=8,
        reset_seed=20260806,
        to_torch=_to_torch,
        gripper_closed_width_m=0.0,
        gripper_open_width_m=0.06,
        simulation_step_seconds=1.0 / 15.0,
        camera_pose_callback=camera_pose,
        contact_envelope=canonical_contact_envelope(),
    )

    metadata = adapter.read_control_observation_metadata()

    assert calls == ["external_camera", "wrist_camera", "external_camera_2"]
    assert metadata["calibrations"]["wrist"]["world_pose_source"] == (
        "runtime_camera_pose_callback"
    )
    assert metadata["calibrations"]["wrist"]["world_from_camera"] == [
        [1.0, 0.0, 0.0, 4.0],
        [0.0, 1.0, 0.0, 5.0],
        [0.0, 0.0, 1.0, 6.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert metadata["calibrations"]["external"]["world_pose_source"] == (
        "isaac_sensor_buffer"
    )


def test_reset_callback_can_restore_a_wrist_observable_episode_start() -> None:
    env = _Env()
    calls: list[str] = []
    adapter = IsaacEpisodeAdapter(
        env=env,
        robot=_Robot(),
        approved_can=_Can(),
        action_dim=8,
        reset_seed=20260806,
        to_torch=_to_torch,
        gripper_closed_width_m=0.0,
        gripper_open_width_m=0.06,
        reset_callback=lambda: calls.append("observable_start_restored"),
        contact_envelope=canonical_contact_envelope(),
    )

    adapter.reset()

    assert calls == ["observable_start_restored"]
    assert env.reset_calls == []


def test_a_missing_finger_body_is_refused_at_construction() -> None:
    """Better to fail building the adapter than to score on a wrong width."""

    class _NoFingers(_Robot):
        def __init__(self):
            super().__init__()
            self.data.body_names = ["panda_link0", "panda_link7"]

    with pytest.raises(IsaacEpisodeAdapterError):
        IsaacEpisodeAdapter(
            env=_Env(),
            robot=_NoFingers(),
            approved_can=_Can(),
            action_dim=8,
            reset_seed=1,
            to_torch=_to_torch,
            gripper_closed_width_m=0.0,
            gripper_open_width_m=0.06,
            contact_envelope=canonical_contact_envelope(),
        )


def test_unmeasured_gripper_width_span_is_refused() -> None:
    with pytest.raises(IsaacEpisodeAdapterError, match="width_calibration_invalid"):
        IsaacEpisodeAdapter(
            env=_Env(),
            robot=_Robot(),
            approved_can=_Can(),
            action_dim=8,
            reset_seed=1,
            to_torch=_to_torch,
            gripper_closed_width_m=0.04,
            gripper_open_width_m=0.04,
            contact_envelope=canonical_contact_envelope(),
        )


def test_a_camera_without_rgb_is_refused() -> None:
    env = _Env()
    env.unwrapped.scene["wrist_camera"].data.output = {}
    adapter = _adapter(env)

    with pytest.raises(IsaacEpisodeAdapterError):
        adapter.read_policy_inputs()


def test_bindings_are_reported_and_drift_is_caught() -> None:
    bindings = describe_adapter()

    assert validate_adapter_bindings(bindings) == []
    assert bindings["camera_view_binding"] == dict(CAMERA_VIEW_BINDING)
    assert bindings["finger_bodies"] == list(FINGER_BODIES)
    assert bindings["finger_tool_frame_local_offset_m"] == list(
        FINGER_TOOL_FRAME_LOCAL_OFFSET_M
    )
    assert bindings["finger_tool_frame_source"] == FINGER_TOOL_FRAME_SOURCE
    assert bindings["gripper_physical_full_opening_m"] == pytest.approx(0.085)
    assert bindings["raw_gripper_body_separation_retained"] is True
    assert bindings["isaaclab_pose_quaternion_order"] == "xyzw"
    assert bindings["scripted_control_physx_jacobian_frame"] == "world"
    assert bindings["scripted_control_controller_error_frame"] == "robot_root"
    assert bindings["scripted_control_jacobian_frame_transform"] == (
        "rotate_linear_and_angular_rows_world_to_robot_root"
    )
    assert bindings["arm_dynamics_observation_schema_version"] == (
        "adp009d_arm_dynamics_observation.v2"
    )
    assert bindings["contact_envelope_runtime_validation_required"] is True
    assert bindings["contact_envelope_retained_in_arm_dynamics_observation"] is True
    assert bindings["contact_force_source"] == (
        "IsaacLab ContactSensor.data.net_forces_w"
    )
    assert bindings["scripted_control_task_space_translation_strategies"] == [
        "direct_global_pose_target",
        "orientation_first_bounded_local_increment",
    ]

    drifted = dict(bindings)
    drifted["camera_view_binding"] = {"external_camera": DROID_WRIST_VIEW}
    assert "isaac_episode_adapter_camera_binding_drifted" in validate_adapter_bindings(
        drifted
    )

    drifted = dict(bindings)
    drifted["gripper_width_source"] = "raw_link_origin_distance"
    assert "isaac_episode_adapter_gripper_width_source_drifted" in (
        validate_adapter_bindings(drifted)
    )

    drifted = dict(bindings)
    drifted["scripted_control_target_frame"] = "panda_hand_origin"
    assert "isaac_episode_adapter_scripted_control_target_frame_drifted" in (
        validate_adapter_bindings(drifted)
    )

    drifted = dict(bindings)
    drifted["finger_tool_frame_local_offset_m"] = [0.0, 0.0, 0.0]
    assert "isaac_episode_adapter_finger_tool_frame_offset_drifted" in (
        validate_adapter_bindings(drifted)
    )

    drifted = dict(bindings)
    drifted["scripted_control_body_pose_resolution"] = "guessed_z_offset"
    assert (
        "isaac_episode_adapter_scripted_control_body_pose_resolution_drifted"
        in validate_adapter_bindings(drifted)
    )

    drifted = dict(bindings)
    drifted["isaaclab_pose_quaternion_order"] = "wxyz"
    assert "isaac_episode_adapter_quaternion_order_drifted" in (
        validate_adapter_bindings(drifted)
    )

    drifted = dict(bindings)
    drifted["scripted_control_physx_jacobian_frame"] = "robot_root"
    assert "isaac_episode_adapter_physx_jacobian_frame_drifted" in (
        validate_adapter_bindings(drifted)
    )

    drifted = dict(bindings)
    drifted["scripted_control_jacobian_frame_transform"] = "none"
    assert "isaac_episode_adapter_jacobian_frame_transform_drifted" in (
        validate_adapter_bindings(drifted)
    )

    drifted = dict(bindings)
    drifted["contact_force_source"] = "inferred_from_motion"
    assert "isaac_episode_adapter_contact_force_source_drifted" in (
        validate_adapter_bindings(drifted)
    )

    drifted = dict(bindings)
    drifted["scripted_control_task_space_translation_strategies"] = [
        "runtime_guess"
    ]
    assert "isaac_episode_adapter_task_space_strategy_drifted" in (
        validate_adapter_bindings(drifted)
    )


def test_adapter_gripper_stroke_matches_deterministic_scorer() -> None:
    from blueprint_pipeline.adp009d_task_scoring import GRIPPER_FULL_OPENING_M

    assert GRIPPER_PHYSICAL_FULL_OPENING_M == GRIPPER_FULL_OPENING_M


def test_the_adapter_satisfies_the_episode_loop_seam() -> None:
    """Whatever the loop calls, the adapter must provide."""

    adapter = _adapter()
    for method in (
        "reset",
        "joint_limits",
        "read_policy_inputs",
        "read_arm_joint_positions",
        "step",
        "read_object_sample",
    ):
        assert callable(getattr(adapter, method))
    limits = adapter.joint_limits()
    assert len(limits) == 7
    assert all(len(row) == 2 for row in limits)


def test_module_imports_without_isaac_or_torch() -> None:
    """It must stay testable off-GPU, so simulator imports live inside methods."""

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_episode_adapter as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    header = source[: source.index("ADAPTER_SCHEMA_VERSION")]
    assert "import torch" not in header
    assert "import isaaclab" not in header
    assert "from pxr" not in header


def test_partner_sensor_covering_two_bodies_is_rejected_not_trusted() -> None:
    """A filtered sensor spanning both fingers is the unsupported PhysX shape and
    reports unreliable values, so it must fail closed rather than be summarized
    as attribution evidence."""
    import pytest

    from blueprint_pipeline.adp009d_isaac_episode_adapter import (
        IsaacEpisodeAdapter,
        IsaacEpisodeAdapterError,
    )

    class _Data:
        force_matrix_w = [[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 2.0]]]]

    class _Sensor:
        body_names = ["left_inner_finger", "right_inner_finger"]
        data = _Data()

    adapter = IsaacEpisodeAdapter.__new__(IsaacEpisodeAdapter)
    adapter._partner_contact_sensors = {"robot_contact_can_left": _Sensor()}
    with pytest.raises(IsaacEpisodeAdapterError) as excinfo:
        adapter._body_contact_partner_forces_n()
    assert "not_one_to_many" in str(excinfo.value)

    adapter._partner_contact_sensors = {}
    assert adapter._body_contact_partner_forces_n() is None


def test_resolved_filter_shape_count_is_reported_with_partner_forces() -> None:
    """A zero partner force means "not touching" only if the filter actually
    resolved. Without the resolved shape count, a filter expression that matched
    nothing is indistinguishable from a partner that is not in contact."""
    from blueprint_pipeline.adp009d_isaac_episode_adapter import IsaacEpisodeAdapter

    class _Data:
        def __init__(self, matrix):
            self.force_matrix_w = matrix

    class _Sensor:
        def __init__(self, body, matrix):
            self.body_names = [body]
            self.data = _Data(matrix)

    adapter = IsaacEpisodeAdapter.__new__(IsaacEpisodeAdapter)
    adapter._to_torch = lambda value: value
    adapter._partner_filter_shapes = {}

    # One env, one body, one resolved filter shape, zero force.
    import numpy as np

    resolved = np.zeros((1, 1, 1, 3))
    adapter._partner_contact_sensors = {"s": _Sensor("left_inner_finger", resolved)}
    forces = adapter._body_contact_partner_forces_n()
    assert forces == {"left_inner_finger": [0.0, 0.0, 0.0]}
    assert adapter._partner_filter_shapes == {"left_inner_finger": 1}

    # Zero filter shapes: the expression matched nothing, so withhold rather
    # than report a zero that would read as "the partner is not touching".
    unmatched = np.zeros((1, 1, 0, 3))
    adapter._partner_filter_shapes = {}
    adapter._partner_contact_sensors = {"s": _Sensor("left_inner_finger", unmatched)}
    assert adapter._body_contact_partner_forces_n() is None
    assert adapter._partner_filter_shapes == {}

    # The SAGE collision filter is a separate category.  It must retain its own
    # resolved-shape count so a zero can filter cannot be misread as evidence
    # about the static collision scene.
    adapter._sage_collision_filter_shapes = {}
    adapter._sage_collision_contact_sensors = {
        "sage": _Sensor("left_inner_finger", resolved)
    }
    sage_forces = adapter._body_contact_sage_collision_forces_n()
    assert sage_forces == {"left_inner_finger": [0.0, 0.0, 0.0]}
    assert adapter._sage_collision_filter_shapes == {"left_inner_finger": 1}

    adapter._sage_collision_filter_shapes = {}
    adapter._sage_collision_contact_sensors = {
        "sage": _Sensor("left_inner_finger", unmatched)
    }
    assert adapter._body_contact_sage_collision_forces_n() is None
    assert adapter._sage_collision_filter_shapes == {}
