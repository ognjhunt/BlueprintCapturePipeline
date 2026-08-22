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
# The pinned Isaac Lab controller predates its upstream null-space joint-limit
# avoidance.  C20c reached the contact target's depth but saturated panda_joint5
# while the remaining redundant joints stayed on the local approach branch.
# Bias contact DLS toward the already solved, receipt-bound PINK posture through
# the *position* null space.  This preserves fingertip translation while using
# the seventh DOF to escape the saturated local branch.  The gain is still
# independently bounded by the existing per-tick slew and measured-state lead.
PHYSX_DLS_POSTURE_NULLSPACE_GAIN = 0.20
PHYSX_DLS_JOINT_LIMIT_AVOIDANCE_GAIN = 0.20
PHYSX_DLS_JOINT_LIMIT_AVOIDANCE_MARGIN_RAD = 0.30
# Position and posture retain Isaac Sim 6.0.1's shipped Franka PINK example.
# Orientation uses the pinned IsaacLab manipulation setting rather than the
# demo's 0.05: r37 proved that a 100:1 position/orientation ratio reached every
# Cartesian position to 6-10 mm while leaving 1.59-2.19 rad of rotation error.
# IsaacLab's exact pick/place configs use orientation cost 1.0 (or 4.0) with
# position cost 8.0; 1.0 here preserves position priority while making this a
# manipulation controller rather than a position-dominant reaching demo.
PINK_POSITION_COST = 5.0
PINK_ORIENTATION_COST = 1.0
PINK_POSTURE_COST = 5.0e-3
# PINK's ``dt`` is its QP integration timestep: it changes velocity limits and
# the next configuration the solver returns. It is not merely a timestamp.
# r40 qualified this controller at 20 Hz. r41 changed the task/policy cadence
# to 15 Hz and, because this value was copied from ``env.step_dt``, PINK chose a
# different joint-limit branch: Cartesian position converged to millimetres but
# orientation remained 0.54--0.85 rad wrong for all 400 steps. Keep the proven
# numerical integrator while the environment and learned policies run at their
# required 15 Hz. NVIDIA's own 6.0.1 examples likewise set PINK ``dt`` as an
# explicit controller integration parameter.
PINK_INTEGRATION_DT_SECONDS = 1.0 / 20.0
# PINK validates every measured configuration against the URDF limits before it
# solves.  The DROID reset writes Panda joint 6 at the exact URDF upper limit;
# after the float32 -> Python conversion observed in r35 that value was
# 3.7525010109 for a 3.7525000000 limit.  Keep the solver's internal measured
# configuration a negligible distance inside the same live articulation limits
# while leaving the real PhysX readback and every commanded target untouched.
PINK_CONFIGURATION_LIMIT_MARGIN_RAD = 1.0e-5
PINK_GLOBAL_SEED_COUNT = 16
PINK_GLOBAL_MAX_ITERATIONS = 192
PINK_GLOBAL_POSITION_TOLERANCE_M = 0.005
PINK_GLOBAL_ORIENTATION_TOLERANCE_RAD = 0.04
PINK_GLOBAL_MINIMUM_JOINT_MARGIN_RAD = 0.05
_HALTON_PRIMES = (2, 3, 5, 7, 11, 13, 17)
# Deterministic, publisher-authored starts before the low-discrepancy sweep:
# exact IsaacLab Arena DROID, IsaacLab Franka+Robotiq ready, and IsaacLab's
# cabinet-manipulation posture. They are IK hypotheses only; native execution
# still decides collision, contact, and arrival.
# Sources: IsaacLab-Arena droid.py@8b4a3a47 and IsaacLab robot/cabinet configs
# @ffff603e, the exact revisions already carried by the runtime source packet.
PINK_GLOBAL_REFERENCE_SEEDS: tuple[tuple[float, ...], ...] = (
    (
        0.0,
        -math.pi / 5.0,
        0.0,
        -4.0 * math.pi / 5.0,
        0.0,
        3.0 * math.pi / 5.0,
        0.0,
    ),
    (0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741),
    (1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469),
)


class NativeFrankaPoseServoError(RuntimeError):
    """Stable binding/controller failures at the native action seam."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    denominator = 1.0
    while index:
        index, digit = divmod(index, base)
        denominator *= base
        result += digit / denominator
    return result


def deterministic_pink_joint_seeds(
    *,
    lower_joint_position_limits_rad: Sequence[float],
    upper_joint_position_limits_rad: Sequence[float],
    preferred_seeds: Sequence[Sequence[float]] = (),
    seed_count: int = PINK_GLOBAL_SEED_COUNT,
) -> list[list[float]]:
    """Return reproducible multi-start configurations inside live limits.

    Pink is explicitly a local solver. A low-discrepancy Halton set covers the
    seven-dimensional Panda joint box without introducing a random seed or a
    new planner dependency. Caller-supplied configurations remain first so a
    continuous previous solution is always tried before global alternatives.
    """

    try:
        lower = [float(value) for value in lower_joint_position_limits_rad]
        upper = [float(value) for value in upper_joint_position_limits_rad]
        count = int(seed_count)
    except (TypeError, ValueError) as exc:
        raise NativeFrankaPoseServoError(
            ["native_franka_pose_servo_global_seeds_invalid"]
        ) from exc
    if (
        len(lower) != 7
        or len(upper) != 7
        or isinstance(seed_count, bool)
        or count <= 0
        or count > 256
        or any(
            not math.isfinite(lo)
            or not math.isfinite(hi)
            or hi - lo <= 2.0 * PINK_CONFIGURATION_LIMIT_MARGIN_RAD
            for lo, hi in zip(lower, upper, strict=True)
        )
    ):
        raise NativeFrankaPoseServoError(
            ["native_franka_pose_servo_global_seeds_invalid"]
        )

    seeds: list[list[float]] = []

    def append_unique(raw: Sequence[float]) -> None:
        seed = pink_configuration_joint_positions(
            measured_joint_positions_rad=raw,
            lower_joint_position_limits_rad=lower,
            upper_joint_position_limits_rad=upper,
        )
        if not any(
            max(abs(a - b) for a, b in zip(seed, existing, strict=True))
            <= 1.0e-9
            for existing in seeds
        ):
            seeds.append(seed)

    for seed in preferred_seeds:
        append_unique(seed)
        if len(seeds) >= count:
            return seeds
    sample_index = 1
    while len(seeds) < count:
        append_unique(
            [
                lo
                + PINK_CONFIGURATION_LIMIT_MARGIN_RAD
                + _radical_inverse(sample_index, _HALTON_PRIMES[axis])
                * (
                    hi
                    - lo
                    - 2.0 * PINK_CONFIGURATION_LIMIT_MARGIN_RAD
                )
                for axis, (lo, hi) in enumerate(
                    zip(lower, upper, strict=True)
                )
            ]
        )
        sample_index += 1
    return seeds


def pink_configuration_joint_positions(
    *,
    measured_joint_positions_rad: Sequence[float],
    lower_joint_position_limits_rad: Sequence[float],
    upper_joint_position_limits_rad: Sequence[float],
    margin_rad: float = PINK_CONFIGURATION_LIMIT_MARGIN_RAD,
) -> list[float]:
    """Clamp only PINK's measured configuration just inside its limits.

    This is deliberately distinct from command clipping: PINK refuses to solve
    when a float-roundtripped reset lies even one ULP outside a URDF limit.
    The returned values seed the constrained optimizer; PhysX measurements,
    terminal errors, and actuator commands continue to use the unmodified live
    joint positions.
    """

    try:
        measured = [float(value) for value in measured_joint_positions_rad]
        lower = [float(value) for value in lower_joint_position_limits_rad]
        upper = [float(value) for value in upper_joint_position_limits_rad]
        margin = float(margin_rad)
    except (TypeError, ValueError) as exc:
        raise NativeFrankaPoseServoError(
            ["native_franka_pose_servo_pink_configuration_limits_invalid"]
        ) from exc
    if (
        not measured
        or len(measured) != len(lower)
        or len(measured) != len(upper)
        or not math.isfinite(margin)
        or margin <= 0.0
        or not all(math.isfinite(value) for value in (*measured, *lower, *upper))
        or any(
            low + margin >= high - margin
            for low, high in zip(lower, upper, strict=True)
        )
    ):
        raise NativeFrankaPoseServoError(
            ["native_franka_pose_servo_pink_configuration_limits_invalid"]
        )
    return [
        min(max(value, low + margin), high - margin)
        for value, low, high in zip(measured, lower, upper, strict=True)
    ]


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


def pose_nullspace_posture_bias(
    *,
    joint_positions: Any,
    preferred_joint_positions: Any,
    task_jacobian: Any,
    gain: float,
) -> Any:
    """Project a preferred posture update away from the full fingertip pose.

    The scripted contact phases command both position and orientation.  Their
    secondary posture motion must therefore lie in the null space of the full
    six-dimensional task Jacobian.  Projecting through only the linear rows can
    preserve XYZ while rotating the gripper away from the handle.
    """

    import torch

    try:
        resolved_gain = float(gain)
    except (TypeError, ValueError) as exc:
        raise NativeFrankaPoseServoError(
            ["native_franka_pose_servo_posture_nullspace_invalid"]
        ) from exc
    if (
        not math.isfinite(resolved_gain)
        or resolved_gain <= 0.0
        or joint_positions.ndim != 2
        or preferred_joint_positions.shape != joint_positions.shape
        or task_jacobian.ndim != 3
        or task_jacobian.shape[0] != joint_positions.shape[0]
        or task_jacobian.shape[1] != 6
        or task_jacobian.shape[2] != joint_positions.shape[1]
    ):
        raise NativeFrankaPoseServoError(
            ["native_franka_pose_servo_posture_nullspace_invalid"]
        )
    task_pseudoinverse = torch.linalg.pinv(task_jacobian)
    joint_count = task_jacobian.shape[2]
    identity = torch.eye(
        joint_count,
        device=task_jacobian.device,
        dtype=task_jacobian.dtype,
    ).expand(task_jacobian.shape[0], -1, -1)
    nullspace_projection = identity - torch.bmm(
        task_pseudoinverse, task_jacobian
    )
    posture_delta = resolved_gain * (
        preferred_joint_positions - joint_positions
    )
    return torch.bmm(
        nullspace_projection, posture_delta.unsqueeze(-1)
    ).squeeze(-1)


def pose_nullspace_joint_limit_avoidance(
    *,
    joint_positions: Any,
    lower_joint_limits: Any,
    upper_joint_limits: Any,
    task_jacobian: Any,
    gain: float,
    margin: float,
) -> Any:
    """Move away from joint limits without changing the commanded grasp pose.

    Activate only near a limit, seek the joint-range center, and project the
    correction through the full task nullspace.  This follows Isaac Lab PINK's
    higher-priority task projection: all six pose rows remain authoritative.
    """

    import torch

    try:
        resolved_gain = float(gain)
        resolved_margin = float(margin)
    except (TypeError, ValueError) as exc:
        raise NativeFrankaPoseServoError(
            ["native_franka_pose_servo_joint_limit_nullspace_invalid"]
        ) from exc
    if (
        not math.isfinite(resolved_gain)
        or resolved_gain <= 0.0
        or not math.isfinite(resolved_margin)
        or resolved_margin <= 0.0
        or joint_positions.ndim != 2
        or lower_joint_limits.shape != joint_positions.shape
        or upper_joint_limits.shape != joint_positions.shape
        or task_jacobian.ndim != 3
        or task_jacobian.shape[0] != joint_positions.shape[0]
        or task_jacobian.shape[1] != 6
        or task_jacobian.shape[2] != joint_positions.shape[1]
    ):
        raise NativeFrankaPoseServoError(
            ["native_franka_pose_servo_joint_limit_nullspace_invalid"]
        )
    midpoint = 0.5 * (lower_joint_limits + upper_joint_limits)
    nearest_limit_distance = torch.minimum(
        joint_positions - lower_joint_limits,
        upper_joint_limits - joint_positions,
    )
    activation = 1.0 - (nearest_limit_distance / resolved_margin).clamp(
        0.0, 1.0
    )
    center_delta = (
        -resolved_gain * activation * (joint_positions - midpoint)
    )
    task_pseudoinverse = torch.linalg.pinv(task_jacobian)
    joint_count = task_jacobian.shape[2]
    identity = torch.eye(
        joint_count,
        device=task_jacobian.device,
        dtype=task_jacobian.dtype,
    ).expand(task_jacobian.shape[0], -1, -1)
    nullspace_projection = identity - torch.bmm(
        task_pseudoinverse, task_jacobian
    )
    return torch.bmm(
        nullspace_projection, center_delta.unsqueeze(-1)
    ).squeeze(-1)


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


def contract_xyzw_to_pink_wxyz(value: Sequence[float]) -> list[float]:
    """Convert Blueprint/Isaac Lab XYZW to PINK SpatialState WXYZ.

    Isaac Sim 6.0.1's ``PinkIKController`` explicitly decodes its site
    quaternion as ``[w, x, y, z]``.  The Arena and every Blueprint pose
    contract use ``[x, y, z, w]``. Keep that exceptional conversion at the
    single PINK boundary.
    """

    qx, qy, qz, qw = contract_xyzw_to_native_xyzw(value)
    return [qw, qx, qy, qz]


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
        self,
        *,
        env: Any,
        robot: Any,
        grasp_geometry_factory: Any = None,
        gripper_convention: Mapping[str, Any] | None = None,
    ):
        import numpy as np
        import torch
        import warp as wp
        import isaacsim.robot_motion.experimental.motion_generation as mg
        from isaaclab.controllers import (
            DifferentialIKController,
            DifferentialIKControllerCfg,
        )
        from isaaclab.utils.math import subtract_frame_transforms
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
        self._subtract_frame_transforms = subtract_frame_transforms
        self._to_torch = lambda value: (
            value if hasattr(value, "detach") else torch.as_tensor(value)
        )
        self.binding = resolve_native_franka_pose_binding(
            body_names=list(robot.data.body_names),
            joint_names=list(robot.joint_names),
            fixed_base=bool(robot.is_fixed_base),
        )
        # PINK remains the constrained/global controller for free-space motion.
        # The exact contact phases use this stock Isaac Lab DLS controller with
        # the live PhysX articulation Jacobian.  That closes error against the
        # body actually being simulated instead of a second Franka model.
        self._physx_dls_controller = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            num_envs=1,
            device=env.unwrapped.device,
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
        except NativeFrankaGraspGeometryError as exc:
            raise NativeFrankaPoseServoError(
                [f"native_franka_pose_servo_grasp_geometry_invalid:{exc}"]
            ) from exc
        except TypeError as exc:
            raise NativeFrankaPoseServoError(
                ["native_franka_pose_servo_grasp_geometry_invalid:type_error"]
            ) from exc
        transform = self.grasp_geometry["controlled_body_to_grasp_frame"]
        self._body_to_grasp_position = list(
            transform["position_controlled_body_m"]
        )
        self._body_to_grasp_quaternion = list(transform["orientation_xyzw"])
        self._body_to_grasp_positions_by_command: dict[float, list[float]] = {}
        self._open_gripper_command: float | None = None
        self._last_gripper_command: float | None = None
        if gripper_convention is not None:
            try:
                endpoint_positions = gripper_convention[
                    "pad_midpoint_controlled_body_m"
                ]
                open_command = float(gripper_convention["open_command"])
                closed_command = float(gripper_convention["closed_command"])
                resolved_positions = {
                    float(command): [float(value) for value in position]
                    for command, position in endpoint_positions.items()
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise NativeFrankaPoseServoError(
                    ["native_franka_pose_servo_gripper_endpoint_tcp_invalid"]
                ) from exc
            if (
                open_command == closed_command
                or set(resolved_positions) != {open_command, closed_command}
                or any(
                    len(position) != 3
                    or not all(math.isfinite(value) for value in position)
                    for position in resolved_positions.values()
                )
            ):
                raise NativeFrankaPoseServoError(
                    ["native_franka_pose_servo_gripper_endpoint_tcp_invalid"]
                )
            self._body_to_grasp_positions_by_command = resolved_positions
            self._open_gripper_command = open_command
            self._last_gripper_command = open_command
        self._pad_centers_body = {
            side: list(position)
            for side, position in self.grasp_geometry[
                "pad_centers_controlled_body_m"
            ].items()
        }
        self._finger_body_indices: dict[str, int] = {}
        self._pad_center_offsets_in_finger_body: dict[str, list[float]] = {}
        if gripper_convention is not None:
            try:
                body_names = list(self._robot.data.body_names)
                offsets = gripper_convention[
                    "pad_center_offsets_in_finger_body_m"
                ]
                for side in ("left", "right"):
                    body_name = f"{side}_inner_finger"
                    offset = [float(value) for value in offsets[side]]
                    if body_name not in body_names or len(offset) != 3 or not all(
                        math.isfinite(value) for value in offset
                    ):
                        raise ValueError(side)
                    self._finger_body_indices[side] = body_names.index(body_name)
                    self._pad_center_offsets_in_finger_body[side] = offset
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise NativeFrankaPoseServoError(
                    ["native_franka_pose_servo_live_pad_binding_invalid"]
                ) from exc
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
                dt=PINK_INTEGRATION_DT_SECONDS,
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
        self._last_gripper_command = self._open_gripper_command
        self._physx_dls_controller.reset()
        # Keep PINK's posture target at the episode's bent reset posture across
        # phase boundaries. Resetting PINK here would redefine its posture task
        # to the current pose -- including a joint-limit pose at the start of
        # the next waypoint -- and erase the secondary objective this backend
        # was selected to provide. ``forward`` refreshes measured joints every
        # tick, so only the actuator command/feedforward state is phase-local.
        # A stale feedforward must not survive into the next phase or episode.
        self._write_joint_velocity_target([0.0] * len(self.binding["arm_joint_ids"]))

    def _pink_state_from_joint_positions(
        self,
        joint_positions_rad: Sequence[float],
        joint_velocities_rad_s: Sequence[float] | None = None,
    ) -> Any:
        values = [float(value) for value in joint_positions_rad]
        velocities_values = (
            [0.0] * len(values)
            if joint_velocities_rad_s is None
            else [float(value) for value in joint_velocities_rad_s]
        )
        if len(values) != len(self.binding["arm_joint_names"]) or not all(
            math.isfinite(value) for value in values
        ) or len(velocities_values) != len(values) or not all(
            math.isfinite(value) for value in velocities_values
        ):
            raise NativeFrankaPoseServoError(
                ["native_franka_pose_servo_pink_state_invalid"]
            )
        positions = self._torch.tensor(
            [values],
            device=self._env.unwrapped.device,
            dtype=self._torch.float32,
        )
        velocities = self._torch.tensor(
            [velocities_values],
            device=self._env.unwrapped.device,
            dtype=self._torch.float32,
        )
        names = list(self.binding["arm_joint_names"])
        return self._mg.RobotState(
            joints=self._mg.JointState.from_name(
                robot_joint_space=names,
                positions=(names, self._wp.from_torch(positions)),
                velocities=(names, self._wp.from_torch(velocities)),
            )
        )

    def _pink_estimated_state(self) -> Any:
        measured_positions = self._to_torch(self._robot.data.joint_pos)[
            :, self.binding["arm_joint_ids"]
        ].contiguous()
        measured_values = [float(value) for value in measured_positions[0]]
        pink_values = pink_configuration_joint_positions(
            measured_joint_positions_rad=measured_values,
            lower_joint_position_limits_rad=self._joint_position_lower,
            upper_joint_position_limits_rad=self._joint_position_upper,
        )
        self._last_pink_measured_joint_positions_rad = measured_values
        self._last_pink_configuration_joint_positions_rad = pink_values
        measured_velocities = self._to_torch(self._robot.data.joint_vel)[
            :, self.binding["arm_joint_ids"]
        ].contiguous()
        return self._pink_state_from_joint_positions(
            pink_values,
            [float(value) for value in measured_velocities[0]],
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
        self,
        *,
        target_position_base: Sequence[float],
        target_quaternion_base_xyzw: Sequence[float],
        preferred_posture_joint_positions_rad: Sequence[float] | None = None,
    ) -> list[float]:
        setpoint = self._pink_setpoint(
            target_position_base=target_position_base,
            target_quaternion_base_xyzw=target_quaternion_base_xyzw,
            preferred_posture_joint_positions_rad=(
                preferred_posture_joint_positions_rad
            ),
        )
        desired = self._pink_controller.forward(
            self._pink_estimated_state(), setpoint, self._pink_time_seconds
        )
        self._pink_time_seconds += PINK_INTEGRATION_DT_SECONDS
        return self._joint_positions_from_pink_state(desired)

    def _pink_setpoint(
        self,
        *,
        target_position_base: Sequence[float],
        target_quaternion_base_xyzw: Sequence[float],
        preferred_posture_joint_positions_rad: Sequence[float] | None = None,
    ) -> Any:
        target_quaternion_base_wxyz = contract_xyzw_to_pink_wxyz(
            target_quaternion_base_xyzw
        )
        self._last_pink_target_hand_quaternion_root_wxyz = (
            target_quaternion_base_wxyz
        )
        position = self._wp.from_numpy(
            self._np.asarray([target_position_base], dtype=self._np.float32),
            dtype=self._wp.float32,
        )
        orientation = self._wp.from_numpy(
            self._np.asarray([target_quaternion_base_wxyz], dtype=self._np.float32),
            dtype=self._wp.float32,
        )
        posture_joints = None
        if preferred_posture_joint_positions_rad is not None:
            # Pink updates its PostureTask only when the setpoint carries joint
            # positions.  A site-only setpoint leaves the posture target at the
            # controller reset configuration, which can pin the live Cartesian
            # servo to a joint-limit branch even after multistart found a valid
            # endpoint.  Keep the site target authoritative for the fingertip
            # path and use the solved joints only as the redundant-arm posture
            # preference.
            posture_state = self._pink_state_from_joint_positions(
                preferred_posture_joint_positions_rad
            )
            posture_joints = posture_state.joints
        return self._mg.RobotState(
            joints=posture_joints,
            sites=self._mg.SpatialState.from_name(
                spatial_space=["panda_hand"],
                positions=(["panda_hand"], position),
                orientations=(["panda_hand"], orientation),
            )
        )

    def _joint_positions_from_pink_state(self, desired: Any) -> list[float]:
        names = list(self.binding["arm_joint_names"])
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

    @staticmethod
    def _quaternion_error_rad(
        observed_xyzw: Sequence[float], target_xyzw: Sequence[float]
    ) -> float:
        observed = [float(value) for value in observed_xyzw]
        target = [float(value) for value in target_xyzw]
        observed_norm = math.sqrt(sum(value * value for value in observed))
        target_norm = math.sqrt(sum(value * value for value in target))
        if observed_norm <= 1.0e-12 or target_norm <= 1.0e-12:
            return math.inf
        dot = abs(
            sum(
                left * right
                for left, right in zip(observed, target, strict=True)
            )
            / (observed_norm * target_norm)
        )
        return 2.0 * math.acos(max(-1.0, min(1.0, dot)))

    def _pink_hand_target_for_grasp_world(
        self,
        *,
        target_position_world_m: Sequence[float],
        target_grasp_frame_quaternion_world_xyzw: Sequence[float],
    ) -> tuple[list[float], list[float]]:
        target_grasp_position_root, target_grasp_quaternion_root = (
            pose_world_to_base(
                position_world=target_position_world_m,
                quaternion_world_xyzw=(
                    target_grasp_frame_quaternion_world_xyzw
                ),
                base_position_world=self._base_pose[:3],
                base_quaternion_world_xyzw=self._base_pose[3:7],
            )
        )
        return controlled_body_pose_for_rigid_grasp_frame_target(
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
            target_grasp_frame_position_world_m=target_grasp_position_root,
            target_grasp_frame_quaternion_world_xyzw=(
                target_grasp_quaternion_root
            ),
        )

    def solve_grasp_target_from_joint_seed(
        self,
        *,
        target_position_world_m: Sequence[float],
        target_grasp_frame_quaternion_world_xyzw: Sequence[float],
        seed_joint_positions_rad: Sequence[float],
        maximum_iterations: int = PINK_GLOBAL_MAX_ITERATIONS,
        position_tolerance_m: float = PINK_GLOBAL_POSITION_TOLERANCE_M,
        orientation_tolerance_rad: float = PINK_GLOBAL_ORIENTATION_TOLERANCE_RAD,
    ) -> dict[str, Any]:
        """Solve one exact pose off-sim from one seed, then restore live PINK.

        The estimated state supplied to ``PinkIKController.forward`` is an
        in-memory RobotState. No articulation target is written and PhysX is
        never stepped. This turns Pink into a bounded multi-start preflight;
        native execution remains the collision and dynamics authority.
        """

        seed = pink_configuration_joint_positions(
            measured_joint_positions_rad=seed_joint_positions_rad,
            lower_joint_position_limits_rad=self._joint_position_lower,
            upper_joint_position_limits_rad=self._joint_position_upper,
        )
        target_position, target_quaternion = (
            self._pink_hand_target_for_grasp_world(
                target_position_world_m=target_position_world_m,
                target_grasp_frame_quaternion_world_xyzw=(
                    target_grasp_frame_quaternion_world_xyzw
                ),
            )
        )
        setpoint = self._pink_setpoint(
            target_position_base=target_position,
            target_quaternion_base_xyzw=target_quaternion,
        )
        original_time = self._pink_time_seconds
        best = {
            "solved": False,
            "joint_positions_rad": list(seed),
            "position_error_m": math.inf,
            "orientation_error_rad": math.inf,
            "iterations": 0,
            "iteration_feedback_clamp_count": 0,
            "maximum_iteration_feedback_clamp_rad": 0.0,
        }
        iteration_feedback_clamp_count = 0
        maximum_iteration_feedback_clamp_rad = 0.0
        try:
            state = self._pink_state_from_joint_positions(seed)
            if self._pink_controller.reset(state, None, 0.0) is not True:
                raise NativeFrankaPoseServoError(
                    ["native_franka_pose_servo_global_seed_reset_failed"]
                )
            for iteration in range(1, int(maximum_iterations) + 1):
                desired = self._pink_controller.forward(
                    state,
                    setpoint,
                    iteration * PINK_INTEGRATION_DT_SECONDS,
                )
                raw_joints = self._joint_positions_from_pink_state(desired)
                # Pink integrates the QP velocity in float32 before returning
                # the next configuration.  At a constrained optimum that can
                # cross a URDF bound by one or more float ULPs.  Feeding that
                # result straight back into ``forward`` makes Pink reject the
                # next iteration before it can solve (observed for every
                # controls multistart seed in c7).  Clamp only this off-sim
                # iteration state just inside the same limits, exactly as the
                # live measured-state boundary already does.  PhysX readback
                # and native commands remain untouched.
                joints = pink_configuration_joint_positions(
                    measured_joint_positions_rad=raw_joints,
                    lower_joint_position_limits_rad=self._joint_position_lower,
                    upper_joint_position_limits_rad=self._joint_position_upper,
                )
                clamp_delta = max(
                    abs(raw - bounded)
                    for raw, bounded in zip(raw_joints, joints, strict=True)
                )
                if clamp_delta > 0.0:
                    iteration_feedback_clamp_count += 1
                    maximum_iteration_feedback_clamp_rad = max(
                        maximum_iteration_feedback_clamp_rad,
                        clamp_delta,
                    )
                state = self._pink_state_from_joint_positions(joints)
                configuration = self._pink_controller._configuration
                if configuration is None:
                    break
                hand = configuration.get_transform_frame_to_world(
                    "panda_hand"
                )
                hand_quaternion = self._Rotation.from_matrix(
                    hand.rotation
                ).as_quat()
                position_error = math.dist(
                    [float(value) for value in hand.translation],
                    target_position,
                )
                orientation_error = self._quaternion_error_rad(
                    [float(value) for value in hand_quaternion],
                    target_quaternion,
                )
                if (
                    position_error + orientation_error
                    < best["position_error_m"]
                    + best["orientation_error_rad"]
                ):
                    best = {
                        "solved": False,
                        "joint_positions_rad": joints,
                        "position_error_m": position_error,
                        "orientation_error_rad": orientation_error,
                        "iterations": iteration,
                        "iteration_feedback_clamp_count": (
                            iteration_feedback_clamp_count
                        ),
                        "maximum_iteration_feedback_clamp_rad": (
                            maximum_iteration_feedback_clamp_rad
                        ),
                    }
                if (
                    position_error <= float(position_tolerance_m)
                    and orientation_error <= float(
                        orientation_tolerance_rad
                    )
                ):
                    best["solved"] = True
                    break
        finally:
            self._pink_time_seconds = original_time
            self._reset_pink_controller()
        for field in ("position_error_m", "orientation_error_rad"):
            if not math.isfinite(float(best[field])):
                best[field] = None
        return best

    def solve_grasp_target_multistart(
        self,
        *,
        target_position_world_m: Sequence[float],
        target_grasp_frame_quaternion_world_xyzw: Sequence[float],
        preferred_seeds: Sequence[Sequence[float]],
        reference_joint_positions_rad: Sequence[float],
        seed_count: int = PINK_GLOBAL_SEED_COUNT,
        position_tolerance_m: float = PINK_GLOBAL_POSITION_TOLERANCE_M,
        orientation_tolerance_rad: float = PINK_GLOBAL_ORIENTATION_TOLERANCE_RAD,
    ) -> dict[str, Any]:
        seeds = deterministic_pink_joint_seeds(
            lower_joint_position_limits_rad=self._joint_position_lower,
            upper_joint_position_limits_rad=self._joint_position_upper,
            preferred_seeds=preferred_seeds,
            seed_count=seed_count,
        )
        reference = [float(value) for value in reference_joint_positions_rad]
        attempts = []
        for index, seed in enumerate(seeds):
            try:
                attempt = self.solve_grasp_target_from_joint_seed(
                    target_position_world_m=target_position_world_m,
                    target_grasp_frame_quaternion_world_xyzw=(
                        target_grasp_frame_quaternion_world_xyzw
                    ),
                    seed_joint_positions_rad=seed,
                    position_tolerance_m=position_tolerance_m,
                    orientation_tolerance_rad=orientation_tolerance_rad,
                )
            except NativeFrankaPoseServoError as exc:
                attempt = {
                    "solved": False,
                    "joint_positions_rad": list(seed),
                    "position_error_m": None,
                    "orientation_error_rad": None,
                    "iterations": 0,
                    "blockers": list(exc.errors),
                }
            margins = [
                min(value - lower, upper - value)
                for value, lower, upper in zip(
                    attempt["joint_positions_rad"],
                    self._joint_position_lower,
                    self._joint_position_upper,
                    strict=True,
                )
            ]
            attempt.update(
                {
                    "seed_index": index,
                    "seed_joint_positions_rad": seed,
                    "minimum_joint_limit_margin_rad": min(margins),
                    "joint_space_reference_distance_rad": math.sqrt(
                        sum(
                            (value - prior) ** 2
                            for value, prior in zip(
                                attempt["joint_positions_rad"],
                                reference,
                                strict=True,
                            )
                        )
                    ),
                    "maximum_reference_joint_delta_rad": max(
                        abs(value - prior)
                        for value, prior in zip(
                            attempt["joint_positions_rad"],
                            reference,
                            strict=True,
                        )
                    ),
                }
            )
            attempts.append(attempt)
        solved = [row for row in attempts if row["solved"]]
        selected = (
            min(
                solved,
                key=lambda row: (
                    row["minimum_joint_limit_margin_rad"]
                    < PINK_GLOBAL_MINIMUM_JOINT_MARGIN_RAD,
                    # Differential IK is local. Preserve the closest whole-arm
                    # branch, not merely the candidate with the smallest
                    # single-joint maximum. C18's maximum-delta comparison
                    # preferred one endpoint by 0.0015 rad even though its
                    # Euclidean joint travel was 19% larger; the live Cartesian
                    # servo then converged to the intervening constrained
                    # optimum and stopped 13.2 mm short of contact.
                    row["joint_space_reference_distance_rad"],
                    row["maximum_reference_joint_delta_rad"],
                    -row["minimum_joint_limit_margin_rad"],
                    row["position_error_m"] + row["orientation_error_rad"],
                    row["seed_index"],
                ),
            )
            if solved
            else None
        )
        return {
            "solved": selected is not None,
            "selected": None if selected is None else dict(selected),
            "attempts": attempts,
            "seed_count": len(seeds),
            "solver": "isaacsim.robot_motion.pink.PinkIKController_multistart",
            "position_tolerance_m": float(position_tolerance_m),
            "orientation_tolerance_rad": float(orientation_tolerance_rad),
            "preferred_minimum_joint_limit_margin_rad": (
                PINK_GLOBAL_MINIMUM_JOINT_MARGIN_RAD
            ),
        }

    def current_body_pose_world(self) -> list[float]:
        pose = self._to_torch(self._robot.data.body_pose_w)[
            0, self.binding["controlled_body_index"], :7
        ]
        return [
            *[float(value) for value in pose[:3]],
            *native_xyzw_to_contract_xyzw(pose[3:7]),
        ]

    def _grasp_position_for_command(
        self, gripper_command: float | None
    ) -> list[float]:
        if not self._body_to_grasp_positions_by_command:
            return list(self._body_to_grasp_position)
        command = (
            self._last_gripper_command
            if gripper_command is None
            else float(gripper_command)
        )
        if command is None or not math.isfinite(command):
            raise NativeFrankaPoseServoError(
                ["native_franka_pose_servo_gripper_endpoint_tcp_invalid"]
            )
        endpoints = sorted(self._body_to_grasp_positions_by_command)
        low, high = endpoints[0], endpoints[-1]
        if abs(high - low) <= 1.0e-9:
            raise NativeFrankaPoseServoError(
                ["native_franka_pose_servo_gripper_endpoint_tcp_invalid"]
            )
        alpha = max(0.0, min(1.0, (command - low) / (high - low)))
        return [
            self._body_to_grasp_positions_by_command[low][axis] * (1.0 - alpha)
            + self._body_to_grasp_positions_by_command[high][axis] * alpha
            for axis in range(3)
        ]

    def current_grasp_frame_position_world(self) -> list[float]:
        return self.current_grasp_frame_pose_world()[:3]

    def current_grasp_frame_pose_world(
        self, *, gripper_command: float | None = None
    ) -> list[float]:
        """Return the measured TCP pose, not the coincident finger body origins."""

        body = self.current_body_pose_world()
        offset = rotate_vector_xyzw(
            body[3:7], self._grasp_position_for_command(gripper_command)
        )
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

        pad_readback = self.current_gripper_pad_readback()
        measured = pad_readback["measured"]
        body_pose = [
            *measured["controlled_body_position_world_m"],
            *measured["controlled_body_quaternion_world_xyzw"],
        ]
        fingers = {
            f"{side}_inner_finger": measured["pad_centers_world_m"][side]
            for side in ("left", "right")
        }
        return gripper_frame_axis_readback(
            controlled_body_name=self.binding["controlled_body_name"],
            body_position_world_m=body_pose[:3],
            body_quaternion_world_xyzw=body_pose[3:7],
            finger_positions_world_m=fingers,
        )

    def current_gripper_pad_readback(self) -> dict[str, Any]:
        """Measure the moving fingertip pad centers from live body poses."""

        if set(self._finger_body_indices) != {"left", "right"} or set(
            self._pad_center_offsets_in_finger_body
        ) != {"left", "right"}:
            raise NativeFrankaPoseServoError(
                ["native_franka_pose_servo_live_pad_binding_invalid"]
            )
        poses = self._to_torch(self._robot.data.body_pose_w)
        body_pose = self.current_body_pose_world()
        finger_body_positions: dict[str, list[float]] = {}
        pad_centers: dict[str, list[float]] = {}
        for side in ("left", "right"):
            pose = poses[0, self._finger_body_indices[side], :7]
            finger_position = [float(value) for value in pose[:3]]
            finger_quaternion = native_xyzw_to_contract_xyzw(pose[3:7])
            offset_world = rotate_vector_xyzw(
                finger_quaternion,
                self._pad_center_offsets_in_finger_body[side],
            )
            finger_body_positions[side] = finger_position
            pad_centers[side] = [
                finger_position[axis] + offset_world[axis]
                for axis in range(3)
            ]
        midpoint = [
            (pad_centers["left"][axis] + pad_centers["right"][axis]) / 2.0
            for axis in range(3)
        ]
        return {
            "schema_version": "native_franka_gripper_pad_readback.v1",
            "measurement_authority": (
                "native_finger_body_pose_plus_probe_sealed_pad_center_offset"
            ),
            "measured": {
                "controlled_body_position_world_m": body_pose[:3],
                "controlled_body_quaternion_world_xyzw": body_pose[3:7],
                "finger_body_positions_world_m": finger_body_positions,
                "pad_center_offsets_in_finger_body_m": {
                    side: list(
                        self._pad_center_offsets_in_finger_body[side]
                    )
                    for side in ("left", "right")
                },
                "pad_centers_world_m": pad_centers,
                "pad_midpoint_world_m": midpoint,
                "pad_separation_m": math.dist(
                    pad_centers["left"], pad_centers["right"]
                ),
            },
        }

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

    def action_for_joint_target(
        self,
        *,
        target_joint_positions_rad: Sequence[float],
        gripper_command: float,
        max_joint_delta_rad: float,
        max_joint_setpoint_lead_rad: float,
        velocity_feedforward_scale: float = DEFAULT_VELOCITY_FEEDFORWARD_SCALE,
    ) -> tuple[list[float], dict[str, Any]]:
        """Track one globally solved pose as bounded native joint targets."""

        self._last_gripper_command = float(gripper_command)
        current = self.read_arm_joint_positions()
        desired = clip_joint_positions_to_limits(
            desired_joint_positions_rad=target_joint_positions_rad,
            lower_joint_position_limits_rad=self._joint_position_lower,
            upper_joint_position_limits_rad=self._joint_position_upper,
        )
        previous = current if self._last_command is None else self._last_command
        bounded = bounded_absolute_joint_setpoint(
            measured_joint_positions_rad=current,
            desired_joint_positions_rad=desired,
            previous_commanded_joint_positions_rad=previous,
            max_command_slew_per_step_rad=float(max_joint_delta_rad),
            max_setpoint_lead_rad=float(max_joint_setpoint_lead_rad),
        )
        feedforward = joint_velocity_feedforward_rad_s(
            commanded_joint_positions_rad=bounded,
            previous_commanded_joint_positions_rad=previous,
            control_period_seconds=self._control_period_seconds,
            scale=float(velocity_feedforward_scale),
        )
        self._write_joint_velocity_target(feedforward)
        self._last_command = list(bounded)
        return [*bounded, float(gripper_command)], {
            "ik_backend": (
                "isaacsim.robot_motion.pink.PinkIKController_multistart_replay"
            ),
            "measured_joint_positions_rad": current,
            "desired_joint_positions_rad": desired,
            "bounded_joint_positions_rad": bounded,
            "commanded_joint_velocity_feedforward_rad_s": feedforward,
            "joint_position_lower_limits_rad": list(
                self._joint_position_lower
            ),
            "joint_position_upper_limits_rad": list(
                self._joint_position_upper
            ),
        }

    def action_for_grasp_target_physx_dls(
        self,
        *,
        target_position_world_m: Sequence[float],
        target_grasp_frame_quaternion_world_xyzw: Sequence[float],
        gripper_command: float,
        max_joint_delta_rad: float,
        max_joint_setpoint_lead_rad: float,
        velocity_feedforward_scale: float = DEFAULT_VELOCITY_FEEDFORWARD_SCALE,
        preferred_posture_joint_positions_rad: Sequence[float] | None = None,
    ) -> tuple[list[float], dict[str, Any]]:
        """Close exact contact error with the live PhysX articulation Jacobian.

        PINK remains authoritative for constrained free-space motion and the
        off-sim reachability preflight.  At contact, however, the gate measures
        the live fingertip-pad midpoint.  Drive the corresponding controlled
        body pose with Isaac Lab's stock damped-least-squares controller so the
        Jacobian and measured pose come from the same PhysX articulation as the
        success measurement.
        """

        self._last_gripper_command = float(gripper_command)
        body_pose = self.current_body_pose_world()
        grasp_pose = self.current_grasp_frame_pose_world(
            gripper_command=gripper_command
        )
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
        command = self._torch.tensor(
            [position_root + quaternion_root_native],
            device=self._env.unwrapped.device,
            dtype=self._torch.float32,
        )
        self._physx_dls_controller.reset()
        self._physx_dls_controller.set_command(command)
        jacobian_world, jacobian_root = self._jacobians_world_and_root()
        body_pose_tensor = self._to_torch(self._robot.data.body_pose_w)[
            :, self.binding["controlled_body_index"]
        ]
        root_pose = self._to_torch(self._robot.data.root_pose_w)
        body_position_root, body_quaternion_root = self._subtract_frame_transforms(
            root_pose[:, :3],
            root_pose[:, 3:7],
            body_pose_tensor[:, :3],
            body_pose_tensor[:, 3:7],
        )
        current = self._to_torch(self._robot.data.joint_pos)[
            :, self.binding["arm_joint_ids"]
        ]
        desired = self._physx_dls_controller.compute(
            body_position_root,
            body_quaternion_root,
            jacobian_root,
            current,
        )
        lower_limits = self._torch.tensor(
            [self._joint_position_lower],
            device=current.device,
            dtype=current.dtype,
        )
        upper_limits = self._torch.tensor(
            [self._joint_position_upper],
            device=current.device,
            dtype=current.dtype,
        )
        joint_limit_bias = pose_nullspace_joint_limit_avoidance(
            joint_positions=current,
            lower_joint_limits=lower_limits,
            upper_joint_limits=upper_limits,
            task_jacobian=jacobian_root,
            gain=PHYSX_DLS_JOINT_LIMIT_AVOIDANCE_GAIN,
            margin=PHYSX_DLS_JOINT_LIMIT_AVOIDANCE_MARGIN_RAD,
        )
        desired = desired + joint_limit_bias
        posture_bias = self._torch.zeros_like(current)
        if preferred_posture_joint_positions_rad is not None:
            try:
                preferred_posture = self._torch.tensor(
                    [[float(value) for value in preferred_posture_joint_positions_rad]],
                    device=current.device,
                    dtype=current.dtype,
                )
            except (TypeError, ValueError) as exc:
                raise NativeFrankaPoseServoError(
                    ["native_franka_pose_servo_posture_nullspace_invalid"]
                ) from exc
            posture_bias = pose_nullspace_posture_bias(
                joint_positions=current,
                preferred_joint_positions=preferred_posture,
                task_jacobian=jacobian_root,
                gain=PHYSX_DLS_POSTURE_NULLSPACE_GAIN,
            )
            desired = desired + posture_bias
        current_values = [float(value) for value in current[0]]
        desired_values = [float(value) for value in desired[0]]
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
            "max_cartesian_translation_step_m": MAX_CARTESIAN_TRANSLATION_STEP_M,
            "max_cartesian_orientation_step_rad": MAX_CARTESIAN_ORIENTATION_STEP_RAD,
            "target_controlled_body_position_world_m": target_body_position,
            "current_controlled_body_position_world_m": body_pose[:3],
            "target_controlled_body_quaternion_world_xyzw": target_body_quaternion,
            "controller_target_quaternion_root_xyzw": quaternion_root_native,
            "ik_backend": "isaaclab.controllers.DifferentialIKController:physx_dls",
            "jacobian_authority": "robot.root_view.get_jacobians:physx_articulation",
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
            "preferred_posture_joint_positions_rad": (
                None
                if preferred_posture_joint_positions_rad is None
                else [float(value) for value in preferred_posture_joint_positions_rad]
            ),
            "pose_nullspace_posture_bias_rad": [
                float(value) for value in posture_bias[0]
            ],
            "pose_nullspace_posture_gain": PHYSX_DLS_POSTURE_NULLSPACE_GAIN,
            "pose_nullspace_joint_limit_avoidance_rad": [
                float(value) for value in joint_limit_bias[0]
            ],
            "pose_nullspace_joint_limit_avoidance_gain": (
                PHYSX_DLS_JOINT_LIMIT_AVOIDANCE_GAIN
            ),
            "pose_nullspace_joint_limit_avoidance_margin_rad": (
                PHYSX_DLS_JOINT_LIMIT_AVOIDANCE_MARGIN_RAD
            ),
            "desired_joint_positions_clipped_to_limits_rad": (
                desired_within_joint_limits
            ),
            "joint_position_lower_limits_rad": self._joint_position_lower,
            "joint_position_upper_limits_rad": self._joint_position_upper,
            "bounded_joint_positions_rad": bounded,
            "measured_joint_positions_rad": current_values,
            "commanded_joint_velocity_feedforward_rad_s": feedforward,
            "measured_joint_velocity_rad_s": measured_velocities,
            "velocity_feedforward_scale": float(velocity_feedforward_scale),
            "control_period_seconds": self._control_period_seconds,
            "joint_stiffness": self._joint_stiffness,
            "joint_damping": self._joint_damping,
            "implicit_pd_torque_terms": torque_terms,
        }
        return action, diagnostics

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
        preferred_posture_joint_positions_rad: Sequence[float] | None = None,
        velocity_feedforward_scale: float = DEFAULT_VELOCITY_FEEDFORWARD_SCALE,
    ) -> tuple[list[float], dict[str, Any]]:
        self._last_gripper_command = float(gripper_command)
        body_pose = self.current_body_pose_world()
        grasp_pose = self.current_grasp_frame_pose_world(
            gripper_command=gripper_command
        )
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
            preferred_posture_joint_positions_rad=(
                preferred_posture_joint_positions_rad
            ),
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
            "pink_preferred_posture_joint_positions_rad": (
                None
                if preferred_posture_joint_positions_rad is None
                else [
                    float(value)
                    for value in preferred_posture_joint_positions_rad
                ]
            ),
            "pink_integration_dt_seconds": PINK_INTEGRATION_DT_SECONDS,
            "pink_target_hand_position_root_m": (
                target_pink_hand_position_root
            ),
            "pink_target_hand_quaternion_root_xyzw": (
                target_pink_hand_quaternion_root
            ),
            "pink_spatial_state_hand_quaternion_root_wxyz": (
                self._last_pink_target_hand_quaternion_root_wxyz
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
            "pink_configuration_limit_margin_rad": (
                PINK_CONFIGURATION_LIMIT_MARGIN_RAD
            ),
            "pink_measured_joint_positions_rad": (
                self._last_pink_measured_joint_positions_rad
            ),
            "pink_configuration_joint_positions_rad": (
                self._last_pink_configuration_joint_positions_rad
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
    "PINK_GLOBAL_REFERENCE_SEEDS",
    "PINK_GLOBAL_SEED_COUNT",
    "PINK_INTEGRATION_DT_SECONDS",
    "SCHEMA_VERSION",
    "contract_xyzw_to_native_xyzw",
    "contract_xyzw_to_pink_wxyz",
    "deterministic_pink_joint_seeds",
    "gripper_frame_axis_readback",
    "native_xyzw_to_contract_xyzw",
    "resolve_native_franka_pose_binding",
]
