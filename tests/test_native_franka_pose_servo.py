from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline.native_franka_global_seed_search import (
    high_margin_joint_seeds,
)
from blueprint_pipeline.native_franka_pose_servo import (
    NativeFrankaDifferentialIkServo,
    NativeFrankaPoseServoError,
    PHYSX_DLS_JOINT_LIMIT_AVOIDANCE_GAIN,
    PHYSX_DLS_JOINT_LIMIT_AVOIDANCE_MARGIN_RAD,
    PHYSX_DLS_POSTURE_NULLSPACE_GAIN,
    PINK_CONFIGURATION_LIMIT_MARGIN_RAD,
    PINK_INTEGRATION_DT_SECONDS,
    PINK_ORIENTATION_COST,
    PINK_POSITION_COST,
    PINK_POSTURE_COST,
    contract_xyzw_to_native_xyzw,
    contract_xyzw_to_pink_wxyz,
    deterministic_pink_joint_seeds,
    native_xyzw_to_contract_xyzw,
    pink_configuration_joint_positions,
    pose_nullspace_posture_bias,
    pose_nullspace_joint_limit_avoidance,
    resolve_native_franka_pose_binding,
    shift_physx_com_jacobian_to_link_origin,
)


def test_physx_com_jacobian_is_shifted_to_the_controlled_link_origin() -> None:
    torch = pytest.importorskip(
        "torch",
        reason="the pinned Isaac runtime consumes PhysX tensors",
    )
    # A unit angular velocity about +Z at a COM 0.2 m along +X has
    # omega x c = +0.2 m/s along Y. The actor origin therefore moves at the
    # COM velocity minus that term.
    jacobian_com = torch.zeros((1, 6, 1), dtype=torch.float64)
    jacobian_com[0, :3, 0] = torch.tensor([0.3, 0.4, 0.5])
    jacobian_com[0, 5, 0] = 1.0

    shifted = shift_physx_com_jacobian_to_link_origin(
        jacobian_world=jacobian_com,
        com_offset_from_link_world_m=torch.tensor(
            [[0.2, 0.0, 0.0]], dtype=torch.float64
        ),
    )

    assert shifted[0, :3, 0] == pytest.approx([0.3, 0.2, 0.5])
    assert shifted[0, 3:, 0] == pytest.approx([0.0, 0.0, 1.0])
    # The raw engine view is evidence and must not be mutated in place.
    assert jacobian_com[0, :3, 0] == pytest.approx([0.3, 0.4, 0.5])


def test_physx_com_jacobian_shift_refuses_mismatched_shapes() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(
        NativeFrankaPoseServoError,
        match="native_franka_pose_servo_jacobian_reference_invalid",
    ):
        shift_physx_com_jacobian_to_link_origin(
            jacobian_world=torch.zeros((1, 6, 7)),
            com_offset_from_link_world_m=torch.zeros((2, 3)),
        )


def test_pose_nullspace_posture_bias_preserves_full_pose_task() -> None:
    torch = pytest.importorskip(
        "torch",
        reason="tensor projection executes inside the Isaac GPU runtime",
    )

    jacobian = torch.zeros((1, 6, 7), dtype=torch.float64)
    jacobian[0, :6, :6] = torch.eye(6, dtype=torch.float64)
    jacobian[0, :, 6] = 1.0
    current = torch.zeros((1, 7), dtype=torch.float64)
    preferred = torch.tensor(
        [[1.0, -1.0, 0.5, 0.4, -0.3, 0.2, -0.1]], dtype=torch.float64
    )

    bias = pose_nullspace_posture_bias(
        joint_positions=current,
        preferred_joint_positions=preferred,
        task_jacobian=jacobian,
        gain=PHYSX_DLS_POSTURE_NULLSPACE_GAIN,
    )

    assert torch.matmul(jacobian, bias.unsqueeze(-1)).squeeze(-1) == (
        pytest.approx(torch.zeros((1, 6), dtype=torch.float64), abs=1e-12)
    )
    assert torch.linalg.vector_norm(bias).item() > 0.0


def test_pose_nullspace_joint_limit_avoidance_preserves_full_pose_task() -> None:
    torch = pytest.importorskip(
        "torch",
        reason="tensor projection executes inside the Isaac GPU runtime",
    )
    jacobian = torch.zeros((1, 6, 7), dtype=torch.float64)
    jacobian[0, :6, :6] = torch.eye(6, dtype=torch.float64)
    jacobian[0, :, 6] = 1.0
    current = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99]], dtype=torch.float64
    )
    lower = torch.full((1, 7), -1.0, dtype=torch.float64)
    upper = torch.full((1, 7), 1.0, dtype=torch.float64)

    bias = pose_nullspace_joint_limit_avoidance(
        joint_positions=current,
        lower_joint_limits=lower,
        upper_joint_limits=upper,
        task_jacobian=jacobian,
        gain=PHYSX_DLS_JOINT_LIMIT_AVOIDANCE_GAIN,
        margin=PHYSX_DLS_JOINT_LIMIT_AVOIDANCE_MARGIN_RAD,
    )

    assert torch.matmul(jacobian, bias.unsqueeze(-1)).squeeze(-1) == (
        pytest.approx(torch.zeros((1, 6), dtype=torch.float64), abs=1e-12)
    )
    assert bias[0, 6] < 0.0


def test_grasp_tcp_interpolates_between_measured_gripper_endpoints() -> None:
    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._body_to_grasp_position = [9.0, 9.0, 9.0]
    servo._body_to_grasp_positions_by_command = {
        0.0: [0.10, 0.00, 0.02],
        1.0: [0.10, 0.02, 0.00],
    }
    servo._last_gripper_command = 0.0

    assert servo._grasp_position_for_command(0.0) == pytest.approx(
        [0.10, 0.00, 0.02]
    )
    assert servo._grasp_position_for_command(1.0) == pytest.approx(
        [0.10, 0.02, 0.00]
    )
    assert servo._grasp_position_for_command(0.5) == pytest.approx(
        [0.10, 0.01, 0.01]
    )


def test_live_pad_readback_uses_moving_finger_bodies_and_local_offsets() -> None:
    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo.binding = {
        "controlled_body_name": "base_link",
        "controlled_body_index": 0,
    }
    servo._finger_body_indices = {"left": 1, "right": 2}
    servo._pad_center_offsets_in_finger_body = {
        "left": [0.01, 0.0, 0.0],
        "right": [-0.01, 0.0, 0.0],
    }
    servo._robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pose_w=np.array(
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [-0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ],
                dtype=float,
            )
        )
    )
    servo._to_torch = lambda value: value

    readback = servo.current_gripper_pad_readback()

    assert readback["measured"]["finger_body_positions_world_m"] == {
        "left": [-0.05, 0.0, 0.0],
        "right": [0.05, 0.0, 0.0],
    }
    assert readback["measured"]["pad_centers_world_m"]["left"] == pytest.approx(
        [-0.04, 0.0, 0.0]
    )
    assert readback["measured"]["pad_centers_world_m"]["right"] == pytest.approx(
        [0.04, 0.0, 0.0]
    )
    assert readback["measured"]["pad_midpoint_world_m"] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert readback["measured"]["pad_separation_m"] == pytest.approx(0.08)


def _bodies() -> list[str]:
    return [
        *[f"panda_link{index}" for index in range(9)],
        "panda_hand",
        "base_link",
        "left_inner_finger",
        "right_inner_finger",
    ]


def test_fixed_base_binding_uses_the_physx_jacobian_row_offset() -> None:
    result = resolve_native_franka_pose_binding(
        body_names=_bodies(),
        joint_names=[f"panda_joint{index}" for index in range(1, 8)],
        fixed_base=True,
    )

    assert result["controlled_body_name"] == "panda_hand"
    assert result["controlled_body_index"] == 9
    assert result["jacobian_body_index"] == 8
    assert result["finger_body_indices"] == [11, 12]


def test_missing_semantic_finger_fails_instead_of_guessing_last_bodies() -> None:
    with pytest.raises(NativeFrankaPoseServoError) as excinfo:
        resolve_native_franka_pose_binding(
            body_names=[name for name in _bodies() if name != "left_inner_finger"],
            joint_names=[f"panda_joint{index}" for index in range(1, 8)],
            fixed_base=True,
        )

    assert excinfo.value.errors == (
        "native_franka_pose_servo_finger_body_missing:left_inner_finger",
    )


def test_canned_beverage_joint_order_cannot_hide_a_wrong_robot_binding() -> None:
    with pytest.raises(NativeFrankaPoseServoError) as excinfo:
        resolve_native_franka_pose_binding(
            body_names=_bodies(),
            joint_names=["legacy_joint"] * 7,
            fixed_base=True,
        )

    assert excinfo.value.errors == (
        "native_franka_pose_servo_arm_joint_binding_invalid",
    )


def test_nonidentity_native_quaternion_preserves_beta2_xyzw_order() -> None:
    native_xyzw = [0.5, 0.5, -0.5, 0.5]

    contract_xyzw = native_xyzw_to_contract_xyzw(native_xyzw)

    assert contract_xyzw == pytest.approx(native_xyzw)
    assert contract_xyzw_to_native_xyzw(contract_xyzw) == pytest.approx(
        native_xyzw
    )


def test_pink_spatial_state_boundary_converts_xyzw_to_wxyz() -> None:
    assert contract_xyzw_to_pink_wxyz([0.1, -0.2, 0.3, 0.9]) == pytest.approx(
        [0.9233805169, 0.1025978352, -0.2051956704, 0.3077935056]
    )


def test_pose_servo_uses_pinned_pink_manipulation_weights() -> None:
    assert PINK_POSITION_COST == 5.0
    assert PINK_ORIENTATION_COST == 1.0
    assert PINK_POSTURE_COST == 5.0e-3
    assert PINK_INTEGRATION_DT_SECONDS == pytest.approx(1.0 / 20.0)


def test_pink_configuration_clamps_float_roundtrip_just_inside_limits() -> None:
    result = pink_configuration_joint_positions(
        measured_joint_positions_rad=[0.0, 3.7525010108947754],
        lower_joint_position_limits_rad=[-2.0, -0.0175],
        upper_joint_position_limits_rad=[2.0, 3.7525],
    )

    assert result[0] == 0.0
    assert result[1] == pytest.approx(
        3.7525 - PINK_CONFIGURATION_LIMIT_MARGIN_RAD
    )
    assert -0.0175 < result[1] < 3.7525


def test_pink_configuration_limit_contract_refuses_collapsed_margin() -> None:
    with pytest.raises(
        NativeFrankaPoseServoError,
        match="native_franka_pose_servo_pink_configuration_limits_invalid",
    ):
        pink_configuration_joint_positions(
            measured_joint_positions_rad=[0.0],
            lower_joint_position_limits_rad=[-1.0e-6],
            upper_joint_position_limits_rad=[1.0e-6],
        )


def test_global_pink_seeds_are_deterministic_bounded_and_preferred_first() -> None:
    lower = [-2.0] * 7
    upper = [2.0] * 7
    preferred = [[0.0] * 7, [1.0] * 7]

    first = deterministic_pink_joint_seeds(
        lower_joint_position_limits_rad=lower,
        upper_joint_position_limits_rad=upper,
        preferred_seeds=preferred,
        seed_count=12,
    )
    second = deterministic_pink_joint_seeds(
        lower_joint_position_limits_rad=lower,
        upper_joint_position_limits_rad=upper,
        preferred_seeds=preferred,
        seed_count=12,
    )

    assert first == second
    assert first[:2] == preferred
    assert len(first) == 12
    assert len({tuple(row) for row in first}) == 12
    assert all(
        -2.0 < value < 2.0 for row in first for value in row
    )


def test_global_pink_seeds_reject_invalid_joint_boxes() -> None:
    with pytest.raises(
        NativeFrankaPoseServoError,
        match="native_franka_pose_servo_global_seeds_invalid",
    ):
        deterministic_pink_joint_seeds(
            lower_joint_position_limits_rad=[-1.0] * 6,
            upper_joint_position_limits_rad=[1.0] * 6,
        )


def test_multistart_prefers_continuous_solved_configuration() -> None:
    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._joint_position_lower = [-2.0] * 7
    servo._joint_position_upper = [2.0] * 7
    calls = []

    def solve(**kwargs):
        calls.append(kwargs)
        joints = list(kwargs["seed_joint_positions_rad"])
        return {
            "solved": True,
            "joint_positions_rad": joints,
            "position_error_m": 0.001,
            "orientation_error_rad": 0.01,
            "iterations": 4,
        }

    servo.solve_grasp_target_from_joint_seed = solve
    result = servo.solve_grasp_target_multistart(
        target_position_world_m=[0.5, 0.0, 0.4],
        target_grasp_frame_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        preferred_seeds=[[1.0] * 7, [0.2] * 7],
        reference_joint_positions_rad=[0.0] * 7,
        seed_count=2,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
    )

    assert result["solved"] is True
    assert all(call["position_tolerance_m"] == 0.005 for call in calls)
    assert all(call["orientation_tolerance_rad"] == 0.08 for call in calls)
    assert result["position_tolerance_m"] == 0.005
    assert result["orientation_tolerance_rad"] == 0.08
    assert result["selected"]["seed_index"] == 1
    assert result["selected"]["joint_positions_rad"] == pytest.approx(
        [0.2] * 7
    )
    assert result["selected"]["joint_space_reference_distance_rad"] == pytest.approx(
        (7 * 0.2**2) ** 0.5
    )


def test_multistart_preserves_closest_whole_arm_branch_before_maximum_delta() -> None:
    """A tiny max-joint advantage must not select much larger total motion."""

    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._joint_position_lower = [-2.0] * 7
    servo._joint_position_upper = [2.0] * 7
    servo.solve_grasp_target_from_joint_seed = lambda **kwargs: {
        "solved": True,
        "joint_positions_rad": list(kwargs["seed_joint_positions_rad"]),
        "position_error_m": 0.001,
        "orientation_error_rad": 0.01,
        "iterations": 4,
    }

    result = servo.solve_grasp_target_multistart(
        target_position_world_m=[0.5, 0.0, 0.4],
        target_grasp_frame_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        preferred_seeds=[
            [0.90] * 7,
            [0.91, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        reference_joint_positions_rad=[0.0] * 7,
        seed_count=2,
    )

    assert result["attempts"][0]["maximum_reference_joint_delta_rad"] < result[
        "attempts"
    ][1]["maximum_reference_joint_delta_rad"]
    assert result["selected"]["seed_index"] == 1
    assert result["selected"]["joint_space_reference_distance_rad"] == pytest.approx(
        0.91
    )


def test_global_pink_iteration_clamps_integrated_float_overshoot() -> None:
    """The next Pink iteration must never receive its own out-of-limit output."""

    class _Rotation:
        @staticmethod
        def from_matrix(_matrix):
            return SimpleNamespace(as_quat=lambda: np.array([0.0, 0.0, 0.0, 1.0]))

    class _Configuration:
        def __init__(self, controller):
            self._controller = controller

        def get_transform_frame_to_world(self, _frame):
            return SimpleNamespace(
                translation=np.array(
                    [1.0, 0.0, 0.0]
                    if self._controller.forward_calls == 1
                    else [0.0, 0.0, 0.0]
                ),
                rotation=np.eye(3),
            )

    class _Controller:
        def __init__(self):
            self.forward_calls = 0
            self._configuration = _Configuration(self)

        def reset(self, _state, _setpoint, _time):
            return True

        def forward(self, state, _setpoint, _time):
            self.forward_calls += 1
            if self.forward_calls == 2:
                assert state[0] == pytest.approx(
                    1.0 - PINK_CONFIGURATION_LIMIT_MARGIN_RAD
                )
                assert -1.0 < state[0] < 1.0
            return [1.0008, *([0.0] * 6)]

    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._joint_position_lower = [-1.0] * 7
    servo._joint_position_upper = [1.0] * 7
    servo._pink_time_seconds = 0.0
    servo._pink_controller = _Controller()
    servo._Rotation = _Rotation
    servo._pink_hand_target_for_grasp_world = lambda **_kwargs: (
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )
    # A zero tool offset: the grasp frame coincides with the hand, so this
    # test measures the iteration clamp and nothing else.
    servo._pink_hand_pose_at_binding_base = [0.0] * 3 + [0.0, 0.0, 0.0, 1.0]
    servo._pink_grasp_pose_at_binding_base = [0.0] * 3 + [0.0, 0.0, 0.0, 1.0]
    servo._pink_setpoint = lambda **_kwargs: object()
    servo._pink_state_from_joint_positions = lambda joints: list(joints)
    servo._joint_positions_from_pink_state = lambda desired: list(desired)
    servo._reset_pink_controller = lambda: None

    result = servo.solve_grasp_target_from_joint_seed(
        target_position_world_m=[0.0, 0.0, 0.0],
        target_grasp_frame_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        seed_joint_positions_rad=[0.0] * 7,
        maximum_iterations=2,
    )

    assert result["solved"] is True
    assert result["iterations"] == 2
    assert result["iteration_feedback_clamp_count"] == 2
    assert result["maximum_iteration_feedback_clamp_rad"] == pytest.approx(
        0.0008 + PINK_CONFIGURATION_LIMIT_MARGIN_RAD
    )


def test_predicted_grasp_frame_fk_uses_pink_to_tcp_binding() -> None:
    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo.binding = {"arm_joint_names": [f"joint_{index}" for index in range(7)]}
    servo._base_pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    servo._pinocchio_frame_probes = lambda: (
        lambda _joints: ([0.2, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        lambda _joints: [],
    )
    observed: dict[str, list[float]] = {}

    def grasp_for_hand_candidate(**kwargs):
        observed.update(kwargs)
        command = float(kwargs.get("gripper_command") or 0.0)
        return [0.4, 0.1 + 0.02 * command, 0.2], [0.0, 0.0, 0.0, 1.0]

    servo._pink_grasp_frame_for_hand_candidate = grasp_for_hand_candidate

    predicted = servo.predicted_grasp_frame_pose_world(
        [0.0] * 7, gripper_command=1.0
    )

    assert observed["candidate_body_position_base_m"] == pytest.approx(
        [0.2, 0.0, 0.0]
    )
    assert observed["gripper_command"] == pytest.approx(1.0)
    assert predicted == pytest.approx(
        [1.4, 2.12, 3.2, 0.0, 0.0, 0.0, 1.0]
    )


def test_pinocchio_pose_jacobian_pair_reuses_fk_without_changing_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"forward": 0, "placements": 0, "jacobian": 0}
    data = SimpleNamespace(
        configuration=np.zeros(1, dtype=float),
        oMf=[
            SimpleNamespace(
                translation=np.zeros(3, dtype=float),
                rotation=np.eye(3, dtype=float),
            )
        ],
    )
    model = SimpleNamespace(getFrameId=lambda name: 0 if name == "panda_hand" else -1)

    def forward_kinematics(_model, frame_data, vector):
        calls["forward"] += 1
        frame_data.configuration = np.array(vector, dtype=float)

    def update_frame_placements(_model, frame_data):
        calls["placements"] += 1
        frame_data.oMf[0] = SimpleNamespace(
            translation=np.array(
                [frame_data.configuration[0], 0.0, 0.0], dtype=float
            ),
            rotation=np.eye(3, dtype=float),
        )

    def compute_frame_jacobian(_model, frame_data, vector, _frame_id, _frame):
        calls["jacobian"] += 1
        assert frame_data.configuration == pytest.approx(vector)
        return np.array(
            [[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
            dtype=float,
        )

    class _Quaternion:
        def __init__(self, _rotation):
            pass

        @staticmethod
        def coeffs():
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    pinocchio = SimpleNamespace(
        LOCAL_WORLD_ALIGNED=object(),
        Quaternion=_Quaternion,
        neutral=lambda _model: np.zeros(1, dtype=float),
        forwardKinematics=forward_kinematics,
        updateFramePlacements=update_frame_placements,
        computeFrameJacobian=compute_frame_jacobian,
    )
    monkeypatch.setitem(sys.modules, "pinocchio", pinocchio)

    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo.binding = {"arm_joint_names": ["panda_joint1"]}
    servo._pink_controller = SimpleNamespace(
        _configuration=SimpleNamespace(model=model, data=data)
    )
    probes = servo._pinocchio_frame_probes()
    assert probes is not None
    cached_pose, cached_jacobian = probes
    pose_calls = 0
    jacobian_calls = 0

    def counted_pose(joints):
        nonlocal pose_calls
        pose_calls += 1
        return cached_pose(joints)

    def counted_jacobian(joints):
        nonlocal jacobian_calls
        jacobian_calls += 1
        return cached_jacobian(joints)

    kwargs = {
        "seeds": [[0.0], [0.8]],
        "target_position_m": [0.4, 0.0, 0.0],
        "target_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "lower_joint_position_limits_rad": [-1.0],
        "upper_joint_position_limits_rad": [1.0],
        "position_tolerance_m": 0.001,
        "orientation_tolerance_rad": 0.01,
        "max_iterations": 20,
    }
    cached_report = high_margin_joint_seeds(
        frame_pose=counted_pose,
        frame_jacobian=counted_jacobian,
        **kwargs,
    )

    def reference_pose(joints):
        return [float(joints[0]), 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]

    def reference_jacobian(_joints):
        return [[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

    reference_report = high_margin_joint_seeds(
        frame_pose=reference_pose,
        frame_jacobian=reference_jacobian,
        **kwargs,
    )

    assert cached_report == reference_report
    assert jacobian_calls > 0
    assert calls["jacobian"] == jacobian_calls
    assert calls["forward"] == calls["placements"] == pose_calls
    assert calls["forward"] < pose_calls + jacobian_calls


def test_global_joint_solution_replay_stays_under_native_command_bounds() -> None:
    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._joint_position_lower = [-2.0] * 7
    servo._joint_position_upper = [2.0] * 7
    servo._last_command = None
    servo._last_gripper_command = 0.0
    servo._control_period_seconds = 1.0 / 15.0
    servo.read_arm_joint_positions = lambda: [0.0] * 7
    observed_velocity = []
    servo._write_joint_velocity_target = observed_velocity.extend

    action, diagnostic = servo.action_for_joint_target(
        target_joint_positions_rad=[1.0] * 7,
        gripper_command=0.25,
        max_joint_delta_rad=0.05,
        max_joint_setpoint_lead_rad=0.2,
        velocity_feedforward_scale=1.0,
    )

    assert action[:7] == pytest.approx([0.05] * 7)
    assert action[7] == 0.25
    assert observed_velocity == pytest.approx([0.75] * 7)
    assert diagnostic["bounded_joint_positions_rad"] == pytest.approx(
        [0.05] * 7
    )
    assert diagnostic["ik_backend"].endswith("_multistart_replay")


def test_multistart_avoids_a_joint_limit_solution_before_continuity() -> None:
    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._joint_position_lower = [-2.0] * 7
    servo._joint_position_upper = [2.0] * 7
    servo.solve_grasp_target_from_joint_seed = lambda **kwargs: {
        "solved": True,
        "joint_positions_rad": list(kwargs["seed_joint_positions_rad"]),
        "position_error_m": 0.001,
        "orientation_error_rad": 0.01,
        "iterations": 4,
    }

    result = servo.solve_grasp_target_multistart(
        target_position_world_m=[0.5, 0.0, 0.4],
        target_grasp_frame_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        preferred_seeds=[[1.99] * 7, [0.5] * 7],
        reference_joint_positions_rad=[1.99] * 7,
        seed_count=2,
    )

    assert result["selected"]["seed_index"] == 1
    assert result["selected"]["minimum_joint_limit_margin_rad"] > 0.05


def test_multistart_rejects_solved_pose_below_required_joint_margin() -> None:
    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._joint_position_lower = [-2.0] * 7
    servo._joint_position_upper = [2.0] * 7
    servo.solve_grasp_target_from_joint_seed = lambda **kwargs: {
        "solved": True,
        "joint_positions_rad": list(kwargs["seed_joint_positions_rad"]),
        "position_error_m": 0.001,
        "orientation_error_rad": 0.01,
        "iterations": 4,
    }

    result = servo.solve_grasp_target_multistart(
        target_position_world_m=[0.5, 0.0, 0.4],
        target_grasp_frame_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        preferred_seeds=[[1.999] * 7, [1.99] * 7],
        reference_joint_positions_rad=[1.999] * 7,
        seed_count=2,
        preferred_minimum_joint_limit_margin_rad=0.05,
        required_minimum_joint_limit_margin_rad=0.005,
    )

    assert result["solved"] is True
    assert result["selected"]["seed_index"] == 1
    assert result["selected"]["minimum_joint_limit_margin_rad"] == pytest.approx(
        0.01
    )
    assert result["required_minimum_joint_limit_margin_rad"] == pytest.approx(
        0.005
    )


def test_pose_servo_uses_pink_for_free_space_and_physx_dls_for_contact() -> None:
    import inspect

    from blueprint_pipeline.native_franka_pose_servo import (
        NativeFrankaDifferentialIkServo,
    )

    source = inspect.getsource(NativeFrankaDifferentialIkServo)
    assert "PinkIKController" in source
    assert 'load_pink_supported_robot("franka")' in source
    assert 'tool_frame="panda_hand"' in source
    assert "dt=PINK_INTEGRATION_DT_SECONDS" in source
    assert "self._pink_time_seconds += PINK_INTEGRATION_DT_SECONDS" in source
    assert "DifferentialIKController" in source
    assert "self._physx_dls_controller.compute" in source
    assert "robot.root_view.get_jacobians:physx_com_referenced" in source
    assert "self._robot.data.body_com_pose_w" in source
    assert "shift_physx_com_jacobian_to_link_origin(" in source
    assert "pink_hand_pose_at_binding" in source
    assert "current_grasp_frame_pose_world" in source
    reset_source = inspect.getsource(
        NativeFrankaDifferentialIkServo.reset_command_state
    )
    assert "_reset_pink_controller" not in reset_source


def test_pink_setpoint_combines_cartesian_target_and_preferred_posture() -> None:
    class _RobotState:
        def __init__(self, **kwargs):
            self.joints = kwargs.get("joints")
            self.sites = kwargs.get("sites")

    class _SpatialState:
        @staticmethod
        def from_name(**kwargs):
            return kwargs

    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._np = np
    servo._wp = SimpleNamespace(
        float32="float32",
        from_numpy=lambda value, **_kwargs: value,
    )
    servo._mg = SimpleNamespace(RobotState=_RobotState, SpatialState=_SpatialState)
    posture = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    posture_joints = object()
    servo._pink_state_from_joint_positions = lambda values: SimpleNamespace(
        joints=(posture_joints if list(values) == posture else None)
    )

    setpoint = servo._pink_setpoint(
        target_position_base=[0.4, 0.1, 0.2],
        target_quaternion_base_xyzw=[0.0, 0.0, 0.0, 1.0],
        preferred_posture_joint_positions_rad=posture,
    )

    assert setpoint.joints is posture_joints
    assert setpoint.sites["spatial_space"] == ["panda_hand"]
    assert setpoint.sites["positions"][1].tolist()[0] == pytest.approx(
        [0.4, 0.1, 0.2]
    )
    assert setpoint.sites["orientations"][1].tolist()[0] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0]
    )


@pytest.mark.parametrize("value", ([0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0]))
def test_quaternion_boundary_rejects_zero_or_wrong_length(value) -> None:
    with pytest.raises(
        NativeFrankaPoseServoError,
        match="native_franka_pose_servo_quaternion_invalid",
    ):
        native_xyzw_to_contract_xyzw(value)


def test_a_solve_is_accepted_on_the_frame_the_arrival_gate_measures() -> None:
    """C32-C37's recurring 11-15 mm miss was the acceptance envelope.

    The solve converts a grasp-frame target into a controlled-body target and
    used to score its candidates *at the body*.  The arrival gate measures the
    grasp frame, roughly 0.21 m down the tool, so an accepted body-orientation
    residual of 0.08 rad reached the gate as ~17 mm of position error while the
    solver reported millimetres.  A candidate that is perfect at the body and
    wrong at the grasp frame must now be refused.
    """

    import math

    from blueprint_pipeline.native_franka_pose_servo import (
        NativeFrankaDifferentialIkServo,
    )

    tool_offset_m = 0.21
    # 0.062 rad of yaw: comfortably inside the old 0.08 rad acceptance, and
    # exactly the 13 mm of grasp-frame error the paid runs kept measuring.
    yaw_rad = 0.062

    yaw_quaternion = np.array(
        [0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)]
    )

    class _Rotation:
        @staticmethod
        def from_matrix(_matrix):
            return SimpleNamespace(as_quat=lambda: yaw_quaternion)

    class _Configuration:
        def get_transform_frame_to_world(self, _frame):
            # The body lands exactly on its target; only its yaw is off.
            return SimpleNamespace(
                translation=np.array([0.0, 0.0, 0.0]), rotation=np.eye(3)
            )

    class _Controller:
        def __init__(self):
            self._configuration = _Configuration()

        def reset(self, _state, _setpoint, _time):
            return True

        def forward(self, _state, _setpoint, _time):
            return [0.0] * 7

    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._joint_position_lower = [-3.0] * 7
    servo._joint_position_upper = [3.0] * 7
    servo._pink_time_seconds = 0.0
    servo._pink_controller = _Controller()
    servo._Rotation = _Rotation
    servo._pink_hand_target_for_grasp_world = lambda **_kwargs: (
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )
    servo._pink_setpoint = lambda **_kwargs: object()
    servo._pink_state_from_joint_positions = lambda joints: list(joints)
    servo._joint_positions_from_pink_state = lambda desired: list(desired)
    servo._reset_pink_controller = lambda: None
    # The grasp frame sits `tool_offset_m` down the tool from the body.
    servo._pink_hand_pose_at_binding_base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    servo._pink_grasp_pose_at_binding_base = [
        tool_offset_m, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]

    result = servo.solve_grasp_target_from_joint_seed(
        target_position_world_m=[0.0, 0.0, 0.0],
        target_grasp_frame_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        seed_joint_positions_rad=[0.0] * 7,
        maximum_iterations=2,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
    )

    # The body is exactly on target, so the old scoring would have accepted.
    assert result["controlled_body_position_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert result["controlled_body_orientation_error_rad"] == pytest.approx(
        yaw_rad, abs=1e-6
    )
    # The grasp frame the gate measures is a lever-arm away, and refused.
    assert result["scored_frame"] == "grasp_frame"
    assert result["grasp_frame_position_error_m"] == pytest.approx(
        2.0 * tool_offset_m * math.sin(yaw_rad / 2.0), rel=1e-6
    )
    assert result["grasp_frame_position_error_m"] > 0.005
    assert result["solved"] is False


def test_a_settled_joint_is_charged_only_for_the_torque_it_is_using() -> None:
    """C36's flat gain sweep: every cell was held by the same 6 N-m.

    An implicit PD joint spends one budget on both terms, so reserving half of
    it for damping at all times makes the achievable holding torque
    ``stiffness * (0.5 * effort / stiffness)`` -- half the effort limit, and
    independent of stiffness. That is why sweeping wrist stiffness over a
    tenfold range moved the fingertip 0.3 mm. A settled joint is not paying the
    damping term and should get the rest of its budget back; a slewing one
    still pays.
    """

    from blueprint_pipeline.native_franka_pose_servo import (
        ACTUATOR_FEASIBLE_LEAD_FRACTION,
        NativeFrankaDifferentialIkServo,
    )

    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._joint_stiffness = [400.0]
    servo._joint_effort_limit = [12.0]
    servo._joint_damping = [80.0]
    static = [ACTUATOR_FEASIBLE_LEAD_FRACTION * 12.0 / 400.0]
    servo._actuator_feasible_lead_rad = list(static)

    # Settled: the damping term costs nothing, so the whole 12 N-m is available
    # and the lead doubles from the reserved 0.015 rad to 0.03 rad.
    settled = servo.actuator_feasible_lead_rad([0.0])
    assert settled == pytest.approx([12.0 / 400.0])
    assert settled[0] == pytest.approx(2.0 * static[0])

    # Slewing at 0.075 rad/s the damping term costs 6 N-m, leaving exactly the
    # static reservation -- the old behaviour, at the velocity it assumed.
    assert servo.actuator_feasible_lead_rad([0.075]) == pytest.approx(static)

    # Faster still, and the floor holds: this may return unspent budget, never
    # take more away than the reservation already did.
    assert servo.actuator_feasible_lead_rad([5.0]) == pytest.approx(static)

    # No velocity readback, or a malformed one, is no worse than before.
    assert servo.actuator_feasible_lead_rad(None) == pytest.approx(static)
    assert servo.actuator_feasible_lead_rad([float("nan")]) == pytest.approx(static)
    assert servo.actuator_feasible_lead_rad([0.0, 0.0]) == pytest.approx(static)


def test_the_arm_reports_the_joint_limits_it_is_actually_held_to() -> None:
    """Off-sim margin claims were being checked against the wrong robot.

    The simulator's soft joint limits were read at binding and never written
    down, so analysis fell back to a stock Franka description: against it a
    30 degree grasp roll appeared to buy 0.62 rad of joint-limit margin, while
    the live solver measured 0.024 rad for the same roll.  A factor of
    twenty-five, and the difference between a fix and a wasted run.
    """

    from blueprint_pipeline.native_franka_pose_servo import (
        NativeFrankaDifferentialIkServo,
    )

    servo = object.__new__(NativeFrankaDifferentialIkServo)
    assert servo.joint_position_limits_rad() is None

    servo._joint_position_lower = [-2.8, -1.7, -2.8, -3.0, -2.8, -0.01, -2.8]
    servo._joint_position_upper = [2.8, 1.7, 2.8, -0.07, 2.8, 3.7, 2.8]

    limits = servo.joint_position_limits_rad()

    assert limits["lower_rad"][4] == pytest.approx(-2.8)
    assert limits["upper_rad"][4] == pytest.approx(2.8)
    assert limits["source"] == "simulator_soft_joint_position_limits_at_binding"
    # Ragged limits are refused rather than half-reported.
    servo._joint_position_upper = [2.8]
    assert servo.joint_position_limits_rad() is None


def test_global_seeds_are_prepended_refined_and_recorded() -> None:
    """The search's answer has to reach the solver, not just a receipt.

    The algorithm having found a high-margin configuration proves nothing on
    its own: this lane has already shipped one rolled pose that was sealed in
    a receipt and never commanded.  What matters is that the configuration is
    prepended to the seeds the multistart actually solves from, that the
    solver's own refinement and scoring still decide the outcome, and that the
    search is recorded beside what the solver did with it.
    """

    from blueprint_pipeline.native_franka_pose_servo import (
        NativeFrankaDifferentialIkServo,
    )

    roomy = [1.1, 0.9, -1.4, -1.9, 2.3, 2.3, -2.1]
    solved_from: list[list[float]] = []

    class _Servo(NativeFrankaDifferentialIkServo):
        def __init__(self, search):
            self._joint_position_lower = [-2.9, -1.7, -2.9, -3.0, -2.9, -0.01, -2.9]
            self._joint_position_upper = [2.9, 1.7, 2.9, -0.07, 2.9, 3.7, 2.9]
            self._search = search

        def _pink_hand_target_for_grasp_world(self, **_kwargs):
            return [0.4, 0.0, 0.4], [0.0, 0.0, 0.0, 1.0]

        def global_margin_seeds(self, **_kwargs):
            return self._search

        def solve_grasp_target_from_joint_seed(
            self, *, seed_joint_positions_rad, **_kwargs
        ):
            solved_from.append(list(seed_joint_positions_rad))
            # The solver still decides: the roomy seed refines to real margin.
            roomy_seed = all(
                abs(a - b) < 1e-9 for a, b in zip(seed_joint_positions_rad, roomy)
            )
            return {
                "solved": True,
                "joint_positions_rad": list(seed_joint_positions_rad),
                "position_error_m": 0.001,
                "orientation_error_rad": 0.01,
                "iterations": 3,
            } | ({"minimum_joint_limit_margin_rad": 0.45} if roomy_seed else {})

    search = {"status": "searched", "seeds": [roomy], "best_margin_rad": 0.45}
    servo = _Servo(search)
    result = servo.solve_grasp_target_multistart(
        target_position_world_m=[0.4, 0.0, 0.4],
        target_grasp_frame_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        preferred_seeds=[[0.0] * 7],
        reference_joint_positions_rad=[0.0] * 7,
    )

    # The searched configuration is the first thing the solver tries...
    assert solved_from and solved_from[0] == pytest.approx(roomy)
    # ...and it is not the only thing: the solver's own seeds still run.
    assert len(solved_from) > 1
    # The search is recorded beside what the solver did with it.
    assert result["global_margin_seed_search"]["status"] == "searched"
    assert result["global_margin_seed_search"]["best_margin_rad"] == 0.45


def test_a_failed_seed_search_never_fails_the_solve() -> None:
    """The seeds are additions to a list that already works."""

    from blueprint_pipeline.native_franka_pose_servo import (
        NativeFrankaDifferentialIkServo,
    )

    solved_from: list[list[float]] = []

    class _Servo(NativeFrankaDifferentialIkServo):
        def __init__(self):
            self._joint_position_lower = [-2.9] * 7
            self._joint_position_upper = [2.9] * 7

        def _pink_hand_target_for_grasp_world(self, **_kwargs):
            return [0.4, 0.0, 0.4], [0.0, 0.0, 0.0, 1.0]

        def global_margin_seeds(self, **_kwargs):
            raise RuntimeError("pinocchio exploded")

        def solve_grasp_target_from_joint_seed(
            self, *, seed_joint_positions_rad, **_kwargs
        ):
            solved_from.append(list(seed_joint_positions_rad))
            return {
                "solved": True,
                "joint_positions_rad": list(seed_joint_positions_rad),
                "minimum_joint_limit_margin_rad": 0.2,
                "position_error_m": 0.001,
                "orientation_error_rad": 0.01,
                "iterations": 3,
            }

    result = _Servo().solve_grasp_target_multistart(
        target_position_world_m=[0.4, 0.0, 0.4],
        target_grasp_frame_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        preferred_seeds=[[0.0] * 7],
        reference_joint_positions_rad=[0.0] * 7,
    )

    # The solve proceeds on the solver's own seeds, and says why it had no others.
    assert solved_from
    assert result["selected"] is not None
    search = result["global_margin_seed_search"]
    assert search["status"] == "unavailable"
    assert "pinocchio exploded" in search["reason"]


def test_near_feasible_global_seed_is_refined_before_local_seeds() -> None:
    """A global near miss must reach the solver without becoming a solution."""

    from blueprint_pipeline.native_franka_pose_servo import (
        NativeFrankaDifferentialIkServo,
    )

    near = [1.0, 0.8, -1.2, -1.7, 2.1, 2.0, -2.0]
    solved_from: list[list[float]] = []

    class _Servo(NativeFrankaDifferentialIkServo):
        def __init__(self):
            self._joint_position_lower = [-2.9] * 7
            self._joint_position_upper = [2.9] * 7

        def _pink_hand_target_for_grasp_world(self, **_kwargs):
            return [0.4, 0.0, 0.4], [0.0, 0.0, 0.0, 1.0]

        def global_margin_seeds(self, **_kwargs):
            return {
                "status": "no_configuration_converged",
                "seeds": [],
                "near_feasible_seeds": [near],
                "near_feasible_orientation_errors_rad": [0.087],
            }

        def solve_grasp_target_from_joint_seed(
            self, *, seed_joint_positions_rad, **_kwargs
        ):
            solved_from.append(list(seed_joint_positions_rad))
            return {
                "solved": True,
                "joint_positions_rad": list(seed_joint_positions_rad),
                "position_error_m": 0.001,
                "orientation_error_rad": 0.01,
                "iterations": 2,
            }

    result = _Servo().solve_grasp_target_multistart(
        target_position_world_m=[0.4, 0.0, 0.4],
        target_grasp_frame_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        preferred_seeds=[[0.0] * 7],
        reference_joint_positions_rad=[0.0] * 7,
    )

    assert solved_from[0] == near
    assert result["global_margin_seed_search"]["status"] == (
        "no_configuration_converged"
    )
    assert result["selected"] is not None
