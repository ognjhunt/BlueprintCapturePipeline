"""Franka Panda kinematics from published parameters, with no asset to fetch.

The lane needs to resolve a scripted trajectory before it can spend anything on
a GPU, and the obvious way - load the vendor MJCF and use a physics engine's
solver - makes an offline planning step depend on a checkout that was staged
into a temp directory months ago and no longer exists. The kinematics are
published, so they are written down here instead: modified Denavit-Hartenberg
parameters and joint limits from Franka's own robot-parameters documentation.
No meshes, no inertias, no download, and it runs on a laptop.

What this buys beyond reach is the Jacobian, and the Jacobian is the part that
matters for a door. Reach feasibility is a yes/no about geometry; a door needs
sustained radial force, and radial is precisely the direction an arm loses as
it straightens. A placement can clear the reach check by millimetres and still
be unable to pull, so force capability is computed rather than assumed.

Torque limits are the published A1-A4 +/-87 N*m and A5-A7 +/-12 N*m. The force
figure derived from them is a static ceiling from the transposed Jacobian - it
ignores gravity load, dynamics, and the collision thresholds that in practice
stop a real arm long before its motors do, so it is an upper bound and the
receipt says as much.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


FRANKA_KINEMATICS_SCHEMA_VERSION = "franka_kinematics.v1"
ARM_JOINT_COUNT = 7

# Modified DH: (a, d, alpha) per joint, then the fixed flange offset.
FRANKA_MODIFIED_DH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -math.pi / 2),
    (0.0, 0.316, math.pi / 2),
    (0.0825, 0.0, math.pi / 2),
    (-0.0825, 0.384, -math.pi / 2),
    (0.0, 0.0, math.pi / 2),
    (0.088, 0.0, math.pi / 2),
)
FRANKA_FLANGE_OFFSET_M = 0.107
FRANKA_JOINT_LIMITS_RAD = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)
FRANKA_JOINT_TORQUE_LIMITS_N_M = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
IK_MAX_ITERATIONS = 400
IK_POSITION_TOLERANCE_M = 1e-4
IK_DAMPING = 0.05
IK_MAX_DAMPING = 1.0
IK_MAX_STEP_RAD = 0.2
JACOBIAN_EPSILON_RAD = 1e-6


class FrankaKinematicsError(ValueError):
    """Stable, sorted Franka kinematics failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _checked(joint_positions: Sequence[float]) -> list[float]:
    try:
        joints = [float(value) for value in joint_positions]
    except (TypeError, ValueError) as exc:
        raise FrankaKinematicsError(["franka_kinematics_joint_vector_invalid"]) from exc
    if len(joints) != ARM_JOINT_COUNT or not all(
        math.isfinite(value) for value in joints
    ):
        raise FrankaKinematicsError(["franka_kinematics_joint_vector_invalid"])
    return joints


def _multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[row][k] * b[k][column] for k in range(4)) for column in range(4)]
        for row in range(4)
    ]


def _link_transform(a: float, d: float, alpha: float, theta: float):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return [
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _chain(joints: Sequence[float]) -> list[list[list[float]]]:
    frames = [[[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]]
    for index, (a, d, alpha) in enumerate(FRANKA_MODIFIED_DH):
        frames.append(_multiply(frames[-1], _link_transform(a, d, alpha, joints[index])))
    flange = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    flange[2][3] = FRANKA_FLANGE_OFFSET_M
    frames.append(_multiply(frames[-1], flange))
    return frames


def forward_kinematics(
    joint_positions: Sequence[float],
) -> tuple[list[float], list[list[float]]]:
    """Flange position and orientation in the arm's base frame."""

    joints = _checked(joint_positions)
    transform = _chain(joints)[-1]
    position = [transform[row][3] for row in range(3)]
    rotation = [[transform[row][column] for column in range(3)] for row in range(3)]
    return position, rotation


def position_jacobian(joint_positions: Sequence[float]) -> list[list[float]]:
    """Three-by-seven linear-velocity Jacobian, by central differences.

    Derived from the forward kinematics rather than written out analytically.
    Modified-DH conventions disagree about which frame a joint axis belongs to,
    and picking the wrong one produces a Jacobian that looks entirely plausible
    and points the solver somewhere slightly wrong - which shows up as an IK
    that stalls a few centimetres out rather than as anything obviously broken.
    Differencing cannot disagree with the kinematics it is differencing.
    """

    joints = _checked(joint_positions)
    columns: list[list[float]] = []
    for index in range(ARM_JOINT_COUNT):
        forward = list(joints)
        backward = list(joints)
        forward[index] += JACOBIAN_EPSILON_RAD
        backward[index] -= JACOBIAN_EPSILON_RAD
        ahead, _ = forward_kinematics(forward)
        behind, _ = forward_kinematics(backward)
        columns.append(
            [
                (ahead[row] - behind[row]) / (2.0 * JACOBIAN_EPSILON_RAD)
                for row in range(3)
            ]
        )
    return [
        [columns[column][row] for column in range(ARM_JOINT_COUNT)] for row in range(3)
    ]


def manipulability(joint_positions: Sequence[float]) -> float:
    """Yoshikawa's measure: how freely the tip can move in any direction."""

    jacobian = position_jacobian(joint_positions)
    gram = [
        [
            sum(jacobian[row][k] * jacobian[column][k] for k in range(ARM_JOINT_COUNT))
            for column in range(3)
        ]
        for row in range(3)
    ]
    determinant = (
        gram[0][0] * (gram[1][1] * gram[2][2] - gram[1][2] * gram[2][1])
        - gram[0][1] * (gram[1][0] * gram[2][2] - gram[1][2] * gram[2][0])
        + gram[0][2] * (gram[1][0] * gram[2][1] - gram[1][1] * gram[2][0])
    )
    return math.sqrt(max(determinant, 0.0))


def radial_force_capability_n(
    joint_positions: Sequence[float], *, direction_world: Sequence[float]
) -> float:
    """Static force the arm can hold along one direction before a joint saturates.

    Transposed-Jacobian bound only: the first joint to hit its torque limit sets
    the ceiling. Gravity load, dynamics, and the collision thresholds that stop
    a real arm well short of its motors are all excluded, so this is an upper
    bound on what the hardware could do and not a prediction of what it will.
    """

    jacobian = position_jacobian(joint_positions)
    length = math.sqrt(sum(float(value) ** 2 for value in direction_world))
    if not math.isfinite(length) or length <= 0.0:
        raise FrankaKinematicsError(["franka_kinematics_direction_degenerate"])
    unit = [float(value) / length for value in direction_world]
    limiting = math.inf
    for index in range(ARM_JOINT_COUNT):
        # tau_i = (J^T f)_i, so a unit force costs this much at joint i.
        cost = abs(sum(jacobian[row][index] * unit[row] for row in range(3)))
        if cost > 1e-12:
            limiting = min(limiting, FRANKA_JOINT_TORQUE_LIMITS_N_M[index] / cost)
    return 0.0 if not math.isfinite(limiting) else limiting


def solve_position_ik(
    *,
    target_position_world_m: Sequence[float],
    seed_joint_positions: Sequence[float] | None = None,
    quaternion_world_xyzw: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Damped least-squares position IK, clamped to the published joint limits."""

    try:
        target = [float(value) for value in target_position_world_m]
    except (TypeError, ValueError) as exc:
        raise FrankaKinematicsError(["franka_kinematics_target_invalid"]) from exc
    if len(target) != 3 or not all(math.isfinite(value) for value in target):
        raise FrankaKinematicsError(["franka_kinematics_target_invalid"])

    if seed_joint_positions is None:
        # A fixed, mid-range seed keeps the solver deterministic run to run.
        joints = [0.0, -0.3, 0.0, -1.8, 0.0, 1.6, 0.785]
    else:
        joints = _checked(seed_joint_positions)

    # Levenberg-Marquardt rather than a fixed-damping pseudo-inverse. With
    # damping small enough to converge quickly the step overshoots, and the
    # joint-limit clamp then turns the overshoot into a permanent detour: the
    # arm parks against a limit and the residual keeps pushing it there. Only
    # accepting steps that reduce the error, and raising damping when one does
    # not, makes the clamp harmless.
    def _error(candidate: list[float]) -> float:
        reached, _ = forward_kinematics(candidate)
        return math.dist(reached, target)

    damping = IK_DAMPING
    best = list(joints)
    best_error = _error(best)
    for _ in range(IK_MAX_ITERATIONS):
        if best_error < IK_POSITION_TOLERANCE_M:
            break
        position, _ = forward_kinematics(joints)
        residual = [target[row] - position[row] for row in range(3)]
        jacobian = position_jacobian(joints)
        gram = [
            [
                sum(
                    jacobian[row][k] * jacobian[column][k]
                    for k in range(ARM_JOINT_COUNT)
                )
                + (damping**2 if row == column else 0.0)
                for column in range(3)
            ]
            for row in range(3)
        ]
        solved = _solve_3x3(gram, residual)
        if solved is None:
            damping = min(damping * 4.0, 10.0)
            continue
        delta = [
            sum(jacobian[row][index] * solved[row] for row in range(3))
            for index in range(ARM_JOINT_COUNT)
        ]
        step = max(abs(value) for value in delta)
        scale = 1.0 if step <= IK_MAX_STEP_RAD else IK_MAX_STEP_RAD / step
        candidate = [
            min(
                FRANKA_JOINT_LIMITS_RAD[index][1],
                max(
                    FRANKA_JOINT_LIMITS_RAD[index][0],
                    joints[index] + delta[index] * scale,
                ),
            )
            for index in range(ARM_JOINT_COUNT)
        ]
        candidate_error = _error(candidate)
        if candidate_error < best_error:
            joints, best, best_error = candidate, list(candidate), candidate_error
            damping = max(damping * 0.7, 1e-3)
        else:
            damping = min(damping * 2.0, IK_MAX_DAMPING)
    joints = best

    position, _ = forward_kinematics(joints)
    error_norm = math.dist(position, target)
    return {
        "solved": error_norm < 1e-3,
        "joint_positions": joints,
        "position_error_m": error_norm,
        "reached_position_world_m": position,
        "manipulability": manipulability(joints),
    }


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    rows = [list(matrix[index]) + [vector[index]] for index in range(3)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1e-15:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for row in range(3):
            if row == column:
                continue
            factor = rows[row][column] / rows[column][column]
            for k in range(column, 4):
                rows[row][k] -= factor * rows[column][k]
    return [rows[index][3] / rows[index][index] for index in range(3)]


__all__ = [
    "ARM_JOINT_COUNT",
    "FRANKA_FLANGE_OFFSET_M",
    "FRANKA_JOINT_LIMITS_RAD",
    "FRANKA_JOINT_TORQUE_LIMITS_N_M",
    "FRANKA_KINEMATICS_SCHEMA_VERSION",
    "FRANKA_MODIFIED_DH",
    "FrankaKinematicsError",
    "forward_kinematics",
    "manipulability",
    "position_jacobian",
    "radial_force_capability_n",
    "solve_position_ik",
]
