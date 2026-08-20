"""Native joint-limited PINK pose servo shared by construction and controls.

This is the task-neutral extraction of the controller first qualified by the
ADP-009D rigid rehearsal.  It controls the measured midpoint between the two
finger bodies, rotates PhysX's world-frame Jacobian into the robot root frame,
and bounds both command slew and lead before emitting the same absolute 8-D
Arena action consumed by learned policies.

Isaac imports occur only when the class is instantiated, so binding and helper
contracts remain hermetically testable on a CPU host.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .native_franka_action_math import (
    bounded_absolute_joint_setpoint,
    bounded_cartesian_pose_target,
    clip_joint_positions_to_limits,
    controlled_body_pose_for_rigid_grasp_frame_target,
    implicit_pd_torque_terms,
    joint_velocity_feedforward_rad_s,
)
from .native_franka_grasp_geometry import (
    NativeFrankaGraspGeometryError,
    measure_live_robotiq_grasp_geometry,
    validate_measured_grasp_geometry,
)
from .native_pose_transforms import (
    NativePoseTransformError,
    pose_world_to_base,
    world_to_base_rotation_row_major_xyzw,
)
from .rigid_frame_transforms import (
    quaternion_multiply_xyzw,
    rotate_vector_xyzw,
)


SCHEMA_VERSION = "native_franka_pose_servo.v1"
GRIPPER_FRAME_READBACK_SCHEMA_VERSION = "native_franka_gripper_frame_readback.v1"
ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
FINGER_BODY_NAMES = ("left_inner_finger", "right_inner_finger")
CONTROLLED_BODY_CANDIDATES = ("panda_hand", "base_link", "panda_link7")

# The jaw axis is a line, not an arrow: the two fingers are interchangeable, so
# the sign of the measured direction is a label this module chooses, not
# something the asset states.  The ordering is named in the receipt so the sign
# can be read rather than assumed.
JAW_AXIS_ORDERING = ("right_inner_finger", "left_inner_finger")

# Below this the two points coincide to within float noise and no direction
# exists.  Refusing is the only honest outcome: a zero-length difference
# normalised anyway would publish a unit vector nobody measured.
GRIPPER_FRAME_DEGENERACY_TOLERANCE_M = 1.0e-9

# The three candidate approach axes below are mutually orthogonal, so they are
# 90 degrees (1.5708 rad) apart.  A measured axis within this cone of one of
# them is unambiguously nearer to it than to either other candidate.
GRIPPER_FRAME_HYPOTHESIS_TOLERANCE_RAD = 0.25

BODY_FRAME_AXES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("+x", (1.0, 0.0, 0.0)),
    ("-x", (-1.0, 0.0, 0.0)),
    ("+y", (0.0, 1.0, 0.0)),
    ("-y", (0.0, -1.0, 0.0)),
    ("+z", (0.0, 0.0, 1.0)),
    ("-z", (0.0, 0.0, -1.0)),
)

# Three sealed artifacts disagree about which frame the controlled body is in,
# and each predicts a different body-frame approach axis.  They are recorded as
# predictions here so a completed run compares a measured axis against them
# arithmetically instead of by argument.  Each hypothesis is named for the
# artifact that asserts it; none of them is privileged by this module.
GRIPPER_FRAME_APPROACH_HYPOTHESES: tuple[dict[str, Any], ...] = (
    {
        "hypothesis_id": "tool_frame_convention_holds_for_controlled_body",
        "predicted_approach_axis_body": [0.0, 0.0, 1.0],
        "predicted_jaw_axis_body_up_to_sign": [0.0, 1.0, 0.0],
        "asserted_by": (
            "native_franka_action_math.grasp_orientation_contact_xyzw: "
            "+Z_ee is the approach axis and +Y_ee the jaw axis"
        ),
    },
    {
        "hypothesis_id": "reset_body_quaternion_implies_a_coupler_rotation",
        "predicted_approach_axis_body": [0.0, -1.0, 0.0],
        "predicted_jaw_axis_body_up_to_sign": None,
        "asserted_by": (
            "the measured reset body quaternion (0.5, 0.5, 0.5, 0.5) has no "
            "axis pointing down except -Y, while this repository's own forward "
            "kinematics puts the flange +Z straight down at the same joints"
        ),
    },
    {
        "hypothesis_id": "wrist_camera_optical_axis_is_the_approach_axis",
        "predicted_approach_axis_body": [1.0, 0.0, 0.0],
        "predicted_jaw_axis_body_up_to_sign": None,
        "asserted_by": (
            "the sealed wrist-camera extrinsic, expressed in base_link, has "
            "its optical axis at (0.9498, -0.3130, 0.0022) ~ +X_body"
        ),
    },
)
# Fraction of the commanded setpoint advance rate declared as a joint velocity
# target.  Defined here rather than in the plan compiler because the runtime
# bundle ships this module and not the compiler, and both sides must agree.
DEFAULT_VELOCITY_FEEDFORWARD_SCALE = 1.0
# Differential IK is a local linearisation. Keep each Cartesian command local
# before independently bounding the resulting joint target. At the Arena's
# 20 Hz control cadence these ceilings are 0.4 m/s and 2 rad/s respectively.
MAX_CARTESIAN_TRANSLATION_STEP_M = 0.02
MAX_CARTESIAN_ORIENTATION_STEP_RAD = 0.10
# Exact values from Isaac Sim 6.0.1's shipped Franka PINK example. Position is
# intentionally weighted 100x above orientation during reactive reaching, and
# the posture task keeps the redundant seventh joint away from a limit branch.
PINK_POSITION_COST = 5.0
PINK_ORIENTATION_COST = 0.05
PINK_POSTURE_COST = 5.0e-3


class NativeFrankaPoseServoError(RuntimeError):
    """Stable binding/controller failures at the native action seam."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def native_xyzw_to_contract_xyzw(value: Sequence[float]) -> list[float]:
    """Normalize Isaac Lab Beta2's native XYZW quaternion contract."""

    try:
        qx, qy, qz, qw = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise NativeFrankaPoseServoError(
            ["native_franka_pose_servo_quaternion_invalid"]
        ) from exc
    quaternion = [qx, qy, qz, qw]
    norm = math.sqrt(sum(item * item for item in quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        raise NativeFrankaPoseServoError(
            ["native_franka_pose_servo_quaternion_invalid"]
        )
    return [item / norm for item in quaternion]


def contract_xyzw_to_native_xyzw(value: Sequence[float]) -> list[float]:
    """Normalize a contract quaternion for Beta2 spawn and differential IK."""

    try:
        qx, qy, qz, qw = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise NativeFrankaPoseServoError(
            ["native_franka_pose_servo_quaternion_invalid"]
        ) from exc
    quaternion = [qx, qy, qz, qw]
    norm = math.sqrt(sum(item * item for item in quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        raise NativeFrankaPoseServoError(
            ["native_franka_pose_servo_quaternion_invalid"]
        )
    return [item / norm for item in quaternion]


def resolve_native_franka_pose_binding(
    *, body_names: Sequence[str], joint_names: Sequence[str], fixed_base: bool
) -> dict[str, Any]:
    """Resolve semantic finger/body names and the fixed-base Jacobian row."""

    bodies = [str(value) for value in body_names]
    joints = [str(value) for value in joint_names]
    errors: list[str] = []
    for name in FINGER_BODY_NAMES:
        if name not in bodies:
            errors.append(f"native_franka_pose_servo_finger_body_missing:{name}")
    controlled = next(
        (name for name in CONTROLLED_BODY_CANDIDATES if name in bodies), None
    )
    if controlled is None:
        errors.append("native_franka_pose_servo_controlled_body_missing")
    if tuple(joints[:7]) != ARM_JOINT_NAMES:
        errors.append("native_franka_pose_servo_arm_joint_binding_invalid")
    if errors:
        raise NativeFrankaPoseServoError(errors)
    assert controlled is not None
    body_index = bodies.index(controlled)
    jacobian_index = body_index - 1 if fixed_base else body_index
    if jacobian_index < 0:
        raise NativeFrankaPoseServoError(
            ["native_franka_pose_servo_jacobian_body_invalid"]
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "arm_joint_names": list(ARM_JOINT_NAMES),
        "arm_joint_ids": list(range(7)),
        "finger_body_names": list(FINGER_BODY_NAMES),
        "finger_body_indices": [bodies.index(name) for name in FINGER_BODY_NAMES],
        "controlled_body_name": controlled,
        "controlled_body_index": body_index,
        "jacobian_body_index": jacobian_index,
        "fixed_base": bool(fixed_base),
    }


def _direction_world_to_body(
    *, direction_world: Sequence[float], body_quaternion_world_xyzw: Sequence[float]
) -> list[float]:
    """Express a world-frame direction in controlled-body coordinates.

    This reuses the sealed ``pose_world_to_base`` rotation rather than restating
    a quaternion rotation here.  A direction is the offset between two points,
    so placing the base origin at zero makes the translation cancel and leaves
    exactly the rotation into body axes.
    """

    try:
        body, _ = pose_world_to_base(
            position_world=direction_world,
            quaternion_world_xyzw=(0.0, 0.0, 0.0, 1.0),
            base_position_world=(0.0, 0.0, 0.0),
            base_quaternion_world_xyzw=body_quaternion_world_xyzw,
        )
    except NativePoseTransformError as exc:
        raise NativeFrankaPoseServoError(
            ["native_franka_pose_servo_quaternion_invalid"]
        ) from exc
    return body


def _unit(
    vector: Sequence[float], *, degeneracy_error: str
) -> tuple[list[float], float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if not math.isfinite(norm) or norm <= GRIPPER_FRAME_DEGENERACY_TOLERANCE_M:
        raise NativeFrankaPoseServoError([degeneracy_error])
    return [float(value) / norm for value in vector], norm


def _nearest_body_axis(unit_body: Sequence[float]) -> dict[str, Any]:
    rows = []
    for label, axis in BODY_FRAME_AXES:
        dot = sum(
            float(value) * component
            for value, component in zip(unit_body, axis, strict=True)
        )
        rows.append((label, dot))
    label, dot = max(rows, key=lambda row: row[1])
    return {
        "body_axis_projections": {name: value for name, value in rows},
        "nearest_body_axis": label,
        "nearest_body_axis_angle_rad": math.acos(max(-1.0, min(1.0, dot))),
    }


def gripper_frame_axis_readback(
    *,
    controlled_body_name: str,
    body_position_world_m: Sequence[float],
    body_quaternion_world_xyzw: Sequence[float],
    finger_positions_world_m: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Measure the controlled body's jaw and approach axes in its own frame.

    Three sealed artifacts disagree about which frame the controlled body is
    in, and none of them can settle it because they are the things that
    disagree.  Reading the reset pose back does settle it, and needs no
    convention at all:

      * the direction between the two ``*_inner_finger`` bodies is the jaw
        axis, because that is what a parallel jaw separates along;
      * the direction from the controlled body's own origin to the finger
        midpoint is the direction the tool extends, which for a parallel jaw is
        the approach axis.

    Both are already read every control tick and thrown away.  This retains
    them, rotated into controlled-body coordinates, so a completed run carries
    the answer instead of a log line that scrolled past.

    What is returned is a measurement, not a verdict.  ``measured`` holds the
    raw readback -- the body pose and both finger world positions -- so every
    derived number can be recomputed from the receipt.  ``assessment`` compares
    the measured approach axis against the three recorded hypotheses
    arithmetically; it is placed alongside the numbers it came from and never
    in place of them.

    Absence fails closed.  A finger body that cannot be resolved, a coincident
    finger pair, or a finger midpoint sitting on the body origin each refuse by
    name rather than defaulting to an axis nobody measured.
    """

    errors = [
        f"native_franka_pose_servo_finger_body_missing:{name}"
        for name in FINGER_BODY_NAMES
        if name not in finger_positions_world_m
    ]
    if errors:
        raise NativeFrankaPoseServoError(errors)

    def _position(label: str, value: Sequence[float]) -> list[float]:
        try:
            row = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise NativeFrankaPoseServoError(
                [f"native_franka_pose_servo_gripper_frame_position_invalid:{label}"]
            ) from exc
        if len(row) != 3 or not all(math.isfinite(item) for item in row):
            raise NativeFrankaPoseServoError(
                [f"native_franka_pose_servo_gripper_frame_position_invalid:{label}"]
            )
        return row

    body_position = _position("controlled_body", body_position_world_m)
    fingers = {
        name: _position(name, finger_positions_world_m[name])
        for name in FINGER_BODY_NAMES
    }
    # Normalising here is what makes the recorded body axes comparable with the
    # hypotheses; the raw values stay in ``measured`` unchanged.
    body_quaternion = contract_xyzw_to_native_xyzw(
        body_quaternion_world_xyzw
    )

    origin, tip = JAW_AXIS_ORDERING
    jaw_world = [
        fingers[tip][index] - fingers[origin][index] for index in range(3)
    ]
    jaw_unit_world, separation = _unit(
        jaw_world,
        degeneracy_error="native_franka_pose_servo_gripper_frame_jaw_degenerate",
    )
    midpoint = [
        (fingers[origin][index] + fingers[tip][index]) / 2.0 for index in range(3)
    ]
    approach_world = [midpoint[index] - body_position[index] for index in range(3)]
    approach_unit_world, tool_offset = _unit(
        approach_world,
        degeneracy_error=(
            "native_franka_pose_servo_gripper_frame_approach_degenerate"
        ),
    )

    jaw_unit_body = _direction_world_to_body(
        direction_world=jaw_unit_world,
        body_quaternion_world_xyzw=body_quaternion,
    )
    approach_unit_body = _direction_world_to_body(
        direction_world=approach_unit_world,
        body_quaternion_world_xyzw=body_quaternion,
    )
    midpoint_offset_body = _direction_world_to_body(
        direction_world=approach_world,
        body_quaternion_world_xyzw=body_quaternion,
    )
    orthogonality_dot = sum(
        left * right
        for left, right in zip(jaw_unit_body, approach_unit_body, strict=True)
    )

    hypotheses = []
    for candidate in GRIPPER_FRAME_APPROACH_HYPOTHESES:
        axis = candidate["predicted_approach_axis_body"]
        dot = sum(
            value * component
            for value, component in zip(approach_unit_body, axis, strict=True)
        )
        row = dict(candidate)
        row["approach_dot"] = dot
        row["approach_angle_rad"] = math.acos(max(-1.0, min(1.0, dot)))
        row["within_tolerance"] = (
            row["approach_angle_rad"] <= GRIPPER_FRAME_HYPOTHESIS_TOLERANCE_RAD
        )
        hypotheses.append(row)
    supported = [row["hypothesis_id"] for row in hypotheses if row["within_tolerance"]]

    return {
        "schema_version": GRIPPER_FRAME_READBACK_SCHEMA_VERSION,
        "controlled_body_name": str(controlled_body_name),
        "measured": {
            "controlled_body_position_world_m": body_position,
            "controlled_body_quaternion_world_xyzw": body_quaternion,
            "finger_body_positions_world_m": fingers,
            "finger_midpoint_world_m": midpoint,
            "finger_separation_m": separation,
            "body_origin_to_finger_midpoint_m": tool_offset,
        },
        "derived": {
            "jaw_axis_ordering": list(JAW_AXIS_ORDERING),
            "jaw_axis_sign_is_a_label_not_a_measurement": True,
            "jaw_unit_world": jaw_unit_world,
            "jaw_unit_body": jaw_unit_body,
            "approach_axis_source": (
                "controlled_body_origin_to_measured_finger_midpoint"
            ),
            "approach_unit_world": approach_unit_world,
            "approach_unit_body": approach_unit_body,
            "body_origin_to_finger_midpoint_body_m": midpoint_offset_body,
            "jaw_approach_orthogonality_dot": orthogonality_dot,
            "jaw_approach_angle_rad": math.acos(
                max(-1.0, min(1.0, orthogonality_dot))
            ),
        },
        "assessment": {
            "hypothesis_tolerance_rad": GRIPPER_FRAME_HYPOTHESIS_TOLERANCE_RAD,
            "approach": _nearest_body_axis(approach_unit_body),
            "jaw": _nearest_body_axis(jaw_unit_body),
            "approach_hypotheses": hypotheses,
            "supported_hypothesis_ids": supported,
            "supported_hypothesis_id": supported[0] if len(supported) == 1 else None,
            "resolution": (
                "supported"
                if len(supported) == 1
                else ("ambiguous" if supported else "none_within_tolerance")
            ),
        },
    }


class NativeFrankaDifferentialIkServo:
    """One deterministic PINK/PhysX pose-servo action per control tick.

    The historical class name is retained for bundle/API compatibility; its IK
    backend is the pinned Isaac Sim 6.0.1 PINK constrained solver.
    """

    def __init__(
        self, *, env: Any, robot: Any, grasp_geometry_factory: Any = None
    ):
        import numpy as np
        import torch
        import warp as wp
        import isaacsim.robot_motion.experimental.motion_generation as mg
        from isaacsim.robot_motion.pink import (
            PinkIKController,
            load_pink_supported_robot,
        )
        from scipy.spatial.transform import Rotation

        self._env = env
        self._robot = robot
        self._mg = mg
        self._np = np
        self._wp = wp
        self._Rotation = Rotation
        self._torch = torch
        self._to_torch = lambda value: (
            value if hasattr(value, "detach") else torch.as_tensor(value)
        )
        self.binding = resolve_native_franka_pose_binding(
            body_names=list(robot.data.body_names),
            joint_names=list(robot.joint_names),
            fixed_base=bool(robot.is_fixed_base),
        )
        base_pose = self._to_torch(robot.data.root_pose_w)[0, :7]
        self._base_pose = [
            *[float(value) for value in base_pose[:3]],
            *native_xyzw_to_contract_xyzw(base_pose[3:7]),
        ]
        rotation = world_to_base_rotation_row_major_xyzw(self._base_pose[3:7])
        self._world_to_root = torch.tensor(
            [rotation], device=env.unwrapped.device, dtype=torch.float32
        ).reshape(1, 3, 3)
        self._last_command: list[float] | None = None
        unwrapped = env.unwrapped
        step_dt = getattr(unwrapped, "step_dt", None)
        if step_dt is None:
            cfg = getattr(unwrapped, "cfg", None)
            sim = getattr(cfg, "sim", None)
            dt = getattr(sim, "dt", None)
            decimation = getattr(cfg, "decimation", None)
            step_dt = None if dt is None or decimation is None else dt * decimation
        try:
            self._control_period_seconds = float(step_dt)
        except (TypeError, ValueError) as exc:
            raise NativeFrankaPoseServoError(
                ["native_franka_pose_servo_control_period_unresolved"]
            ) from exc
        if (
            not math.isfinite(self._control_period_seconds)
            or self._control_period_seconds <= 0.0
        ):
            raise NativeFrankaPoseServoError(
                ["native_franka_pose_servo_control_period_unresolved"]
            )
        # Read the drive gains once so the retained torque can be split into the
        # two terms that produce it.  Absence is recorded, never fatal.
        self._joint_stiffness = self._arm_gain("joint_stiffness")
        self._joint_damping = self._arm_gain("joint_damping")
        try:
            limits = self._to_torch(robot.data.soft_joint_pos_limits)[
                0, self.binding["arm_joint_ids"], :
            ]
            self._joint_position_lower = [float(value) for value in limits[:, 0]]
            self._joint_position_upper = [float(value) for value in limits[:, 1]]
        except Exception as exc:  # noqa: BLE001 - fail closed at the command seam
            raise NativeFrankaPoseServoError(
                ["native_franka_pose_servo_joint_limits_unresolved"]
            ) from exc
        if any(
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower >= upper
            for lower, upper in zip(
                self._joint_position_lower,
                self._joint_position_upper,
                strict=True,
            )
        ):
            raise NativeFrankaPoseServoError(
                ["native_franka_pose_servo_joint_limits_unresolved"]
            )
        body_pose = self.current_body_pose_world()
        try:
            if grasp_geometry_factory is None:
                import omni.usd

                grasp_geometry = measure_live_robotiq_grasp_geometry(
                    stage=omni.usd.get_context().get_stage(),
                    controlled_body_position_world_m=body_pose[:3],
                    controlled_body_quaternion_world_xyzw=body_pose[3:7],
                )
            else:
                grasp_geometry = grasp_geometry_factory(body_pose)
            self.grasp_geometry = validate_measured_grasp_geometry(
                grasp_geometry
            )
        except (NativeFrankaGraspGeometryError, TypeError) as exc:
            raise NativeFrankaPoseServoError(
                ["native_franka_pose_servo_grasp_geometry_invalid"]
            ) from exc
        transform = self.grasp_geometry["controlled_body_to_grasp_frame"]
        self._body_to_grasp_position = list(
            transform["position_controlled_body_m"]
        )
        self._body_to_grasp_quaternion = list(transform["orientation_xyzw"])
        self._pad_centers_body = {
            side: list(position)
            for side, position in self.grasp_geometry[
                "pad_centers_controlled_body_m"
            ].items()
        }
        try:
            pink_robot = load_pink_supported_robot("franka")
            # The bundled Franka URDF may include Panda fingers, while the
            # DROID articulation replaces them with one Robotiq mimic chain.
            # The seven arm joints and kinematics are unchanged; constrain the
            # solver to that exact common ordered subset.
            pink_robot.controlled_joint_names = list(
                self.binding["arm_joint_names"]
            )
            self._pink_controller = PinkIKController(
                pink_robot=pink_robot,
                robot_joint_space=list(self.binding["arm_joint_names"]),
                robot_site_space=["panda_hand"],
                tool_frame="panda_hand",
                position_cost=PINK_POSITION_COST,
                orientation_cost=PINK_ORIENTATION_COST,
                posture_cost=PINK_POSTURE_COST,
                solver="osqp",
                dt=self._control_period_seconds,
            )
            self._pink_time_seconds = 0.0
            self._reset_pink_controller()
            configuration = self._pink_controller._configuration
            if configuration is None:
                raise RuntimeError("pink_configuration_missing")
            hand = configuration.get_transform_frame_to_world("panda_hand")
            hand_quaternion = self._Rotation.from_matrix(hand.rotation).as_quat()
            self._pink_hand_pose_at_binding_base = [
                *[float(value) for value in hand.translation],
                *[float(value) for value in hand_quaternion],
            ]
            grasp_world = self.current_grasp_frame_pose_world()
            grasp_position_base, grasp_quaternion_base = pose_world_to_base(
                position_world=grasp_world[:3],
                quaternion_world_xyzw=grasp_world[3:7],
                base_position_world=self._base_pose[:3],
                base_quaternion_world_xyzw=self._base_pose[3:7],
            )
            self._pink_grasp_pose_at_binding_base = [
                *grasp_position_base,
                *grasp_quaternion_base,
            ]
        except Exception as exc:  # noqa: BLE001 - one stable runtime boundary
            raise NativeFrankaPoseServoError(
                ["native_franka_pose_servo_pink_initialization_failed"]
            ) from exc

    def _arm_gain(self, attribute: str) -> list[float] | None:
        data = getattr(self._robot, "data", None)
        value = getattr(data, attribute, None) if data is not None else None
        if value is None:
            return None
        try:
            row = self._to_torch(value)[0]
            return [
                float(row[index]) for index in self.binding["arm_joint_ids"]
            ]
        except Exception:  # noqa: BLE001 - diagnostic only
            return None

    def read_arm_joint_velocities(self) -> list[float]:
        values = self._to_torch(self._robot.data.joint_vel)[
            0, self.binding["arm_joint_ids"]
        ]
        return [float(value) for value in values]

    def _write_joint_velocity_target(self, velocities: Sequence[float]) -> None:
        """Declare the intended joint velocity through the stock Isaac Lab API.

        For implicit actuators ``write_data_to_sim`` sends the position *and*
        velocity target to PhysX, and the stock ``JointPositionAction`` never
        touches the velocity target, so this is an addition to the command
        rather than a competing control path.  The plain
        ``set_joint_velocity_target`` is deprecated in the pinned revision, so
        the indexed form is preferred.
        """

        target = self._torch.tensor(
            [[float(value) for value in velocities]],
            device=self._env.unwrapped.device,
            dtype=self._torch.float32,
        )
        joint_ids = list(self.binding["arm_joint_ids"])
        for name in (
            "set_joint_velocity_target_index",
            "set_joint_velocity_target",
        ):
            setter = getattr(self._robot, name, None)
            if setter is None:
                continue
            setter(target=target, joint_ids=joint_ids)
            return
        raise NativeFrankaPoseServoError(
            ["native_franka_pose_servo_velocity_target_api_missing"]
        )

    def reset_command_state(self) -> None:
        self._last_command = None
        # Keep PINK's posture target at the episode's bent reset posture across
        # phase boundaries. Resetting PINK here would redefine its posture task
        # to the current pose -- including a joint-limit pose at the start of
        # the next waypoint -- and erase the secondary objective this backend
        # was selected to provide. ``forward`` refreshes measured joints every
        # tick, so only the actuator command/feedforward state is phase-local.
        # A stale feedforward must not survive into the next phase or episode.
        self._write_joint_velocity_target([0.0] * len(self.binding["arm_joint_ids"]))

    def _pink_estimated_state(self) -> Any:
        positions = self._to_torch(self._robot.data.joint_pos)[
            :, self.binding["arm_joint_ids"]
        ].contiguous()
        velocities = self._to_torch(self._robot.data.joint_vel)[
            :, self.binding["arm_joint_ids"]
        ].contiguous()
        names = list(self.binding["arm_joint_names"])
        return self._mg.RobotState(
            joints=self._mg.JointState.from_name(
                robot_joint_space=names,
                positions=(names, self._wp.from_torch(positions)),
                velocities=(names, self._wp.from_torch(velocities)),
            )
        )

    def _reset_pink_controller(self) -> None:
        reset = self._pink_controller.reset(
            self._pink_estimated_state(), None, self._pink_time_seconds
        )
        if reset is not True:
            raise NativeFrankaPoseServoError(
                ["native_franka_pose_servo_pink_reset_failed"]
            )

    def _pink_desired_joint_positions(
        self, *, target_position_base: Sequence[float], target_quaternion_base_xyzw: Sequence[float]
    ) -> list[float]:
        names = list(self.binding["arm_joint_names"])
        position = self._wp.from_numpy(
            self._np.asarray([target_position_base], dtype=self._np.float32),
            dtype=self._wp.float32,
        )
        orientation = self._wp.from_numpy(
            self._np.asarray([target_quaternion_base_xyzw], dtype=self._np.float32),
            dtype=self._wp.float32,
        )
        setpoint = self._mg.RobotState(
            sites=self._mg.SpatialState.from_name(
                spatial_space=["panda_hand"],
                positions=(["panda_hand"], position),
                orientations=(["panda_hand"], orientation),
            )
        )
        desired = self._pink_controller.forward(
            self._pink_estimated_state(), setpoint, self._pink_time_seconds
        )
        self._pink_time_seconds += self._control_period_seconds
        if desired is None or desired.joints is None or desired.joints.positions is None:
            raise NativeFrankaPoseServoError(
                ["native_franka_pose_servo_pink_solution_missing"]
            )
        desired_names = list(desired.joints.position_names)
        raw = self._np.asarray(desired.joints.positions.numpy()).reshape(-1)
        if not set(names).issubset(desired_names):
            raise NativeFrankaPoseServoError(
                ["native_franka_pose_servo_pink_joint_mapping_invalid"]
            )
        return [float(raw[desired_names.index(name)]) for name in names]

    def current_body_pose_world(self) -> list[float]:
        pose = self._to_torch(self._robot.data.body_pose_w)[
            0, self.binding["controlled_body_index"], :7
        ]
        return [
            *[float(value) for value in pose[:3]],
            *native_xyzw_to_contract_xyzw(pose[3:7]),
        ]

    def current_grasp_frame_position_world(self) -> list[float]:
        return self.current_grasp_frame_pose_world()[:3]

    def current_grasp_frame_pose_world(self) -> list[float]:
        """Return the measured TCP pose, not the coincident finger body origins."""

        body = self.current_body_pose_world()
        offset = rotate_vector_xyzw(body[3:7], self._body_to_grasp_position)
        position = [body[index] + offset[index] for index in range(3)]
        quaternion = quaternion_multiply_xyzw(
            body[3:7], self._body_to_grasp_quaternion
        )
        return [*position, *quaternion]

    def current_gripper_frame_axis_readback(self) -> dict[str, Any]:
        """Retain the jaw and approach axes in controlled-body coordinates.

        Every buffer this reads is already read on the control path -- the
        controlled body's world pose by ``current_body_pose_world`` above, and
        both finger world positions by ``current_grasp_frame_position_world``,
        which averages them and discards the direction.  This costs one extra
        indexing of ``body_pose_w`` and settles, by measurement, which frame the
        controlled body is actually in.
        """

        body_pose = self.current_body_pose_world()
        fingers = {
            f"{side}_inner_finger": [
                body_pose[axis]
                + rotate_vector_xyzw(
                    body_pose[3:7], self._pad_centers_body[side]
                )[axis]
                for axis in range(3)
            ]
            for side in ("left", "right")
        }
        return gripper_frame_axis_readback(
            controlled_body_name=self.binding["controlled_body_name"],
            body_position_world_m=body_pose[:3],
            body_quaternion_world_xyzw=body_pose[3:7],
            finger_positions_world_m=fingers,
        )

    def read_arm_joint_positions(self) -> list[float]:
        values = self._to_torch(self._robot.data.joint_pos)[0, :7]
        return [float(value) for value in values]

    def _jacobians_world_and_root(self) -> tuple[Any, Any]:
        world = self._to_torch(self._robot.root_view.get_jacobians())[
            :,
            self.binding["jacobian_body_index"],
            :,
            self.binding["arm_joint_ids"],
        ]
        root = world.clone()
        root[:, :3, :] = self._torch.bmm(self._world_to_root, world[:, :3, :])
        root[:, 3:, :] = self._torch.bmm(self._world_to_root, world[:, 3:, :])
        return world, root

    def action_for_grasp_target(
        self,
        *,
        target_position_world_m: Sequence[float],
        target_grasp_frame_quaternion_world_xyzw: Sequence[float],
        gripper_command: float,
        # Required, not defaulted. These carried 0.03/0.20 as DEFAULTS, and a
        # default is invisible at the call site: the construction worker simply
        # omitted both arguments and inherited the pre-#786 pair, so a merged,
        # deployed limit raise was inert across r17/r19/r20 (PR #793).
        #
        # #793 fixed that call site. Removing the defaults closes the door
        # instead of the instance: with no default there is nothing to inherit,
        # so the next caller cannot reintroduce the same silence. Both current
        # callers already pass them explicitly.
        #
        # velocity_feedforward_scale below KEEPS its default on purpose: it is
        # bound to a named constant in this module, not a second copy of a
        # number defined elsewhere, and #797 designed 0.0 as the A/B baseline.
        max_joint_delta_rad: float,
        max_joint_setpoint_lead_rad: float,
        velocity_feedforward_scale: float = DEFAULT_VELOCITY_FEEDFORWARD_SCALE,
    ) -> tuple[list[float], dict[str, Any]]:
        body_pose = self.current_body_pose_world()
        grasp_pose = self.current_grasp_frame_pose_world()
        local_target_position, local_target_quaternion = (
            bounded_cartesian_pose_target(
                current_position_world_m=grasp_pose[:3],
                current_quaternion_world_xyzw=grasp_pose[3:7],
                target_position_world_m=target_position_world_m,
                target_quaternion_world_xyzw=(
                    target_grasp_frame_quaternion_world_xyzw
                ),
                max_translation_step_m=MAX_CARTESIAN_TRANSLATION_STEP_M,
                max_orientation_step_rad=MAX_CARTESIAN_ORIENTATION_STEP_RAD,
            )
        )
        target_body_position, target_body_quaternion = (
            controlled_body_pose_for_rigid_grasp_frame_target(
                current_body_position_world_m=body_pose[:3],
                current_body_quaternion_world_xyzw=body_pose[3:7],
                current_grasp_frame_position_world_m=grasp_pose[:3],
                current_grasp_frame_quaternion_world_xyzw=grasp_pose[3:7],
                target_grasp_frame_position_world_m=local_target_position,
                target_grasp_frame_quaternion_world_xyzw=(
                    local_target_quaternion
                ),
            )
        )
        position_root, quaternion_root = pose_world_to_base(
            position_world=target_body_position,
            quaternion_world_xyzw=target_body_quaternion,
            base_position_world=self._base_pose[:3],
            base_quaternion_world_xyzw=self._base_pose[3:7],
        )
        quaternion_root_native = contract_xyzw_to_native_xyzw(quaternion_root)
        target_grasp_position_root, target_grasp_quaternion_root = (
            pose_world_to_base(
                position_world=local_target_position,
                quaternion_world_xyzw=local_target_quaternion,
                base_position_world=self._base_pose[:3],
                base_quaternion_world_xyzw=self._base_pose[3:7],
            )
        )
        target_pink_hand_position_root, target_pink_hand_quaternion_root = (
            controlled_body_pose_for_rigid_grasp_frame_target(
                current_body_position_world_m=(
                    self._pink_hand_pose_at_binding_base[:3]
                ),
                current_body_quaternion_world_xyzw=(
                    self._pink_hand_pose_at_binding_base[3:7]
                ),
                current_grasp_frame_position_world_m=(
                    self._pink_grasp_pose_at_binding_base[:3]
                ),
                current_grasp_frame_quaternion_world_xyzw=(
                    self._pink_grasp_pose_at_binding_base[3:7]
                ),
                target_grasp_frame_position_world_m=(
                    target_grasp_position_root
                ),
                target_grasp_frame_quaternion_world_xyzw=(
                    target_grasp_quaternion_root
                ),
            )
        )
        jacobian_world, jacobian_root = self._jacobians_world_and_root()
        current = self._to_torch(self._robot.data.joint_pos)[
            :, self.binding["arm_joint_ids"]
        ]
        desired_values = self._pink_desired_joint_positions(
            target_position_base=target_pink_hand_position_root,
            target_quaternion_base_xyzw=target_pink_hand_quaternion_root,
        )
        current_values = [float(value) for value in current[0]]
        desired_within_joint_limits = clip_joint_positions_to_limits(
            desired_joint_positions_rad=desired_values,
            lower_joint_position_limits_rad=self._joint_position_lower,
            upper_joint_position_limits_rad=self._joint_position_upper,
        )
        previous = current_values if self._last_command is None else self._last_command
        bounded = bounded_absolute_joint_setpoint(
            measured_joint_positions_rad=current_values,
            desired_joint_positions_rad=desired_within_joint_limits,
            previous_commanded_joint_positions_rad=previous,
            max_command_slew_per_step_rad=float(max_joint_delta_rad),
            max_setpoint_lead_rad=float(max_joint_setpoint_lead_rad),
        )
        # Declare the rate the setpoint is advancing at, so the damping term
        # stops braking the motion we just commanded.  It is zero whenever the
        # setpoint holds, which leaves the joint damped at rest.
        feedforward = joint_velocity_feedforward_rad_s(
            commanded_joint_positions_rad=bounded,
            previous_commanded_joint_positions_rad=previous,
            control_period_seconds=self._control_period_seconds,
            scale=float(velocity_feedforward_scale),
        )
        self._write_joint_velocity_target(feedforward)
        measured_velocities = self.read_arm_joint_velocities()
        torque_terms: dict[str, Any] = {
            "available": False,
            "reason": "joint_gains_unavailable",
        }
        if self._joint_stiffness is not None and self._joint_damping is not None:
            torque_terms = implicit_pd_torque_terms(
                commanded_joint_positions_rad=bounded,
                measured_joint_positions_rad=current_values,
                commanded_joint_velocities_rad_s=feedforward,
                measured_joint_velocities_rad_s=measured_velocities,
                joint_stiffness=self._joint_stiffness,
                joint_damping=self._joint_damping,
            )
        self._last_command = list(bounded)
        action = [*bounded, float(gripper_command)]
        diagnostics = {
            "target_grasp_frame_position_world_m": [
                float(value) for value in target_position_world_m
            ],
            "current_grasp_frame_position_world_m": grasp_pose[:3],
            "current_grasp_frame_quaternion_world_xyzw": grasp_pose[3:7],
            "target_grasp_frame_quaternion_world_xyzw": list(
                target_grasp_frame_quaternion_world_xyzw
            ),
            "local_ik_target_grasp_frame_position_world_m": local_target_position,
            "local_ik_target_grasp_frame_quaternion_world_xyzw": (
                local_target_quaternion
            ),
            "max_cartesian_translation_step_m": (
                MAX_CARTESIAN_TRANSLATION_STEP_M
            ),
            "max_cartesian_orientation_step_rad": (
                MAX_CARTESIAN_ORIENTATION_STEP_RAD
            ),
            "target_controlled_body_position_world_m": target_body_position,
            "current_controlled_body_position_world_m": body_pose[:3],
            "target_controlled_body_quaternion_world_xyzw": target_body_quaternion,
            "controller_target_quaternion_root_xyzw": quaternion_root_native,
            "ik_backend": "isaacsim.robot_motion.pink.PinkIKController",
            "pink_position_cost": PINK_POSITION_COST,
            "pink_orientation_cost": PINK_ORIENTATION_COST,
            "pink_posture_cost": PINK_POSTURE_COST,
            "pink_target_hand_position_root_m": (
                target_pink_hand_position_root
            ),
            "pink_target_hand_quaternion_root_xyzw": (
                target_pink_hand_quaternion_root
            ),
            "pink_hand_pose_at_binding_root": (
                self._pink_hand_pose_at_binding_base
            ),
            "pink_grasp_pose_at_binding_root": (
                self._pink_grasp_pose_at_binding_base
            ),
            "jacobian_world_frobenius_norm": float(
                self._torch.linalg.vector_norm(jacobian_world[0])
            ),
            "jacobian_root_frobenius_norm": float(
                self._torch.linalg.vector_norm(jacobian_root[0])
            ),
            "jacobian_root_rank": int(
                self._torch.linalg.matrix_rank(jacobian_root[0])
            ),
            "desired_joint_positions_rad": desired_values,
            "desired_joint_positions_clipped_to_limits_rad": (
                desired_within_joint_limits
            ),
            "joint_position_lower_limits_rad": self._joint_position_lower,
            "joint_position_upper_limits_rad": self._joint_position_upper,
            "bounded_joint_positions_rad": bounded,
            "measured_joint_positions_rad": current_values,
            # An implicit-actuator torque is the sum of two terms that cancel at
            # steady state.  Retain them apart so a near-zero reading can be
            # told from gains that never took effect.
            "commanded_joint_velocity_feedforward_rad_s": feedforward,
            "measured_joint_velocity_rad_s": measured_velocities,
            "velocity_feedforward_scale": float(velocity_feedforward_scale),
            "control_period_seconds": self._control_period_seconds,
            "joint_stiffness": self._joint_stiffness,
            "joint_damping": self._joint_damping,
            "implicit_pd_torque_terms": torque_terms,
        }
        return action, diagnostics


__all__ = [
    "ARM_JOINT_NAMES",
    "BODY_FRAME_AXES",
    "DEFAULT_VELOCITY_FEEDFORWARD_SCALE",
    "CONTROLLED_BODY_CANDIDATES",
    "FINGER_BODY_NAMES",
    "GRIPPER_FRAME_APPROACH_HYPOTHESES",
    "GRIPPER_FRAME_DEGENERACY_TOLERANCE_M",
    "GRIPPER_FRAME_HYPOTHESIS_TOLERANCE_RAD",
    "GRIPPER_FRAME_READBACK_SCHEMA_VERSION",
    "JAW_AXIS_ORDERING",
    "NativeFrankaDifferentialIkServo",
    "NativeFrankaPoseServoError",
    "SCHEMA_VERSION",
    "contract_xyzw_to_native_xyzw",
    "gripper_frame_axis_readback",
    "native_xyzw_to_contract_xyzw",
    "resolve_native_franka_pose_binding",
]
