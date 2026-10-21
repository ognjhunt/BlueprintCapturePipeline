"""Named-arm TCP observations and measured, bounded native pose-servo steps.

The input pose is absolute in the named arm's base frame, metres and XYZW.
Isaac Lab Beta2's native pose buffers are already XYZW. PhysX's Jacobian is
at the body's centre of mass; it is shifted to the declared TCP before IK.
No task object, scorer state, reference robot or assumed FK enters this path.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

POSE_CHUNK_SCHEMA = "blueprint.named_arm_pose_chunk.v1"


def _finite_array(value: Any, shape: tuple[int, ...], code: str) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=object)
        if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, numbers.Real) for item in raw.flat):
            raise ValueError(code)
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(code) from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(code)
    return result


def rotation_xyzw(value: Sequence[float]) -> np.ndarray:
    q = _finite_array(value, (4,), "team_arm_quaternion_invalid")
    if not math.isclose(float(q @ q), 1.0, abs_tol=1e-4):
        raise ValueError("team_arm_quaternion_invalid")
    x, y, z, w = q
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                     [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                     [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


def quaternion_xyzw(rotation: Any) -> list[float]:
    matrix = _finite_array(rotation, (3, 3), "team_arm_rotation_invalid")
    if not (np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5)
            and math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-5)):
        raise ValueError("team_arm_rotation_invalid")
    # Largest component construction remains stable at a half turn.
    candidates = np.array([1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2],
                           1 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2],
                           1 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2],
                           1 + np.trace(matrix)])
    index = int(np.argmax(candidates))
    component = math.sqrt(max(0.0, float(candidates[index]))) / 2
    denominator = 4 * component
    if denominator < 1e-10:
        raise ValueError("team_arm_rotation_invalid")
    if index == 0:
        q = [component, (matrix[0, 1]+matrix[1, 0])/denominator,
             (matrix[0, 2]+matrix[2, 0])/denominator, (matrix[2, 1]-matrix[1, 2])/denominator]
    elif index == 1:
        q = [(matrix[0, 1]+matrix[1, 0])/denominator, component,
             (matrix[1, 2]+matrix[2, 1])/denominator, (matrix[0, 2]-matrix[2, 0])/denominator]
    elif index == 2:
        q = [(matrix[0, 2]+matrix[2, 0])/denominator, (matrix[1, 2]+matrix[2, 1])/denominator,
             component, (matrix[1, 0]-matrix[0, 1])/denominator]
    else:
        q = [(matrix[2, 1]-matrix[1, 2])/denominator, (matrix[0, 2]-matrix[2, 0])/denominator,
             (matrix[1, 0]-matrix[0, 1])/denominator, component]
    result = np.asarray(q)
    result /= np.linalg.norm(result)
    if result[3] < 0:
        result = -result
    return result.tolist()


def pose_matrix(position: Sequence[float], orientation_xyzw: Sequence[float]) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation_xyzw(orientation_xyzw)
    result[:3, 3] = _finite_array(position, (3,), "team_arm_position_invalid")
    return result


def relative_tcp_pose(*, base_pose: Sequence[float], tool_pose: Sequence[float],
                      tool_from_tcp: Mapping[str, Any]) -> dict[str, list[float]]:
    base = _finite_array(base_pose, (7,), "team_arm_native_pose_invalid")
    tool = _finite_array(tool_pose, (7,), "team_arm_native_pose_invalid")
    tcp = (np.linalg.inv(pose_matrix(base[:3], base[3:])) @ pose_matrix(tool[:3], tool[3:])
           @ pose_matrix(tool_from_tcp["position_m"], tool_from_tcp["orientation_xyzw"]))
    return {"position_m": tcp[:3, 3].tolist(), "orientation_xyzw": quaternion_xyzw(tcp[:3, :3])}


def _skew(value: np.ndarray) -> np.ndarray:
    x, y, z = value
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def pose_servo_step(*, joint_positions: Sequence[float], jacobian_world_com: Any,
                    body_pose_world: Sequence[float], body_com_position_world: Sequence[float],
                    base_pose_world: Sequence[float], tool_from_tcp: Mapping[str, Any],
                    target_position_m: Sequence[float], target_orientation_xyzw: Sequence[float],
                    position_limits: Sequence[Sequence[float]], velocity_limits: Sequence[float],
                    step_seconds: float, damping: float = 0.05, gain_per_second: float = 4.0
                    ) -> tuple[list[float], dict[str, Any]]:
    """One DLS target step, bounded by position and per-tick velocity limits.

    Rate/position limiting is explicit in the receipt, including its exact
    scale. This is a commanded servo step, never an achieved-pose/IK success.
    """
    q = np.asarray(joint_positions, dtype=float)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all():
        raise ValueError("team_arm_joint_state_invalid")
    n = len(q)
    jacobian = _finite_array(jacobian_world_com, (6, n), "team_arm_jacobian_invalid").copy()
    bounds = _finite_array(position_limits, (n, 2), "team_arm_limits_invalid")
    velocities = _finite_array(velocity_limits, (n,), "team_arm_limits_invalid")
    if (not math.isfinite(step_seconds) or step_seconds <= 0 or not math.isfinite(damping) or damping <= 0
            or not math.isfinite(gain_per_second) or gain_per_second <= 0
            or np.any(bounds[:, 0] >= bounds[:, 1]) or np.any(velocities <= 0)
            or np.any(q < bounds[:, 0] - 1e-6) or np.any(q > bounds[:, 1] + 1e-6)):
        raise ValueError("team_arm_limits_invalid")
    tool = _finite_array(body_pose_world, (7,), "team_arm_native_pose_invalid")
    base = _finite_array(base_pose_world, (7,), "team_arm_native_pose_invalid")
    com = _finite_array(body_com_position_world, (3,), "team_arm_native_pose_invalid")
    world_tool = pose_matrix(tool[:3], tool[3:])
    world_tcp = world_tool @ pose_matrix(tool_from_tcp["position_m"], tool_from_tcp["orientation_xyzw"])
    target = pose_matrix(base[:3], base[3:]) @ pose_matrix(target_position_m, target_orientation_xyzw)
    # v_TCP = v_COM + omega cross (TCP - COM).
    jacobian[:3] -= _skew(world_tcp[:3, 3] - com) @ jacobian[3:]
    error_rotation = target[:3, :3] @ world_tcp[:3, :3].T
    error_q = np.asarray(quaternion_xyzw(error_rotation))
    sine = float(np.linalg.norm(error_q[:3]))
    angle = 2 * math.atan2(sine, float(error_q[3]))
    rotational_error = np.zeros(3) if sine < 1e-12 else error_q[:3] * (angle / sine)
    translation_error = target[:3, 3] - world_tcp[:3, 3]
    error = np.concatenate((translation_error, rotational_error))
    delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping*damping*np.eye(6),
                                         error * gain_per_second * step_seconds)
    if not np.isfinite(delta).all():
        raise ValueError("team_arm_ik_nonfinite")
    rate_scale = min(1.0, float(np.min(velocities * step_seconds / np.maximum(np.abs(delta), 1e-30))))
    bounded = delta * rate_scale
    position_scale = 1.0
    for index, change in enumerate(bounded):
        if change > 0:
            position_scale = min(position_scale, max(0.0, float((bounds[index, 1] - q[index]) / change)))
        elif change < 0:
            position_scale = min(position_scale, max(0.0, float((bounds[index, 0] - q[index]) / change)))
    command = q + bounded * position_scale
    if np.any(command < bounds[:, 0] - 1e-6) or np.any(command > bounds[:, 1] + 1e-6):
        raise ValueError("team_arm_ik_joint_bounds_invalid")
    return command.tolist(), {
        "method": "measured_com_jacobian_tcp_shift_damped_least_squares",
        "quaternion_order": "xyzw", "target_frame": "arm_base", "step_seconds": step_seconds,
        "position_error_before_m": float(np.linalg.norm(translation_error)),
        "orientation_error_before_rad": angle, "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
        "damping": damping, "gain_per_second": gain_per_second,
        "velocity_limit_scale": rate_scale, "position_limit_scale": position_scale,
        "joint_position_before": q.tolist(), "unconstrained_delta": delta.tolist(),
        "joint_targets": command.tolist(), "target_achievement": "not_measured_by_this_command",
    }


def validate_pose_chunk(value: Mapping[str, Any], *, arm_names: Sequence[str], maximum_rows: int) -> dict[str, Any]:
    expected = {"schema_version", "frame", "arm_names", "positions_m", "orientations_xyzw", "gripper_closures"}
    if (set(value) != expected or value.get("schema_version") != POSE_CHUNK_SCHEMA
            or value.get("frame") != "arm_base" or value.get("arm_names") != list(arm_names)):
        raise ValueError("team_arm_pose_chunk_contract_invalid")
    rows = value.get("positions_m")
    if not isinstance(rows, list) or not 1 <= len(rows) <= maximum_rows:
        raise ValueError("team_arm_pose_chunk_shape_invalid")
    n, count = len(rows), len(arm_names)
    positions = _finite_array(rows, (n, count, 3), "team_arm_pose_chunk_shape_invalid")
    orientations = _finite_array(value.get("orientations_xyzw"), (n, count, 4), "team_arm_pose_chunk_shape_invalid")
    grippers = _finite_array(value.get("gripper_closures"), (n, count), "team_arm_pose_chunk_shape_invalid")
    if np.any(grippers < 0) or np.any(grippers > 1):
        raise ValueError("team_arm_pose_chunk_gripper_bounds_invalid")
    for row in orientations:
        for orientation in row:
            rotation_xyzw(orientation)
    return {"positions_m": positions, "orientations_xyzw": orientations, "gripper_closures": grippers}


class LiveArmKinematics:
    """Bind names once, but read native poses, q and COM Jacobians on every tick."""

    def __init__(self, *, robot: Any, row: Mapping[str, Any], as_array: Callable[[Any], Any]):
        if row.get("base_mode") != "fixed_at_station" or not row.get("arms"):
            raise ValueError("team_arm_named_fixed_binding_required")
        self.robot, self.row, self.array = robot, row, as_array
        self.arms = list(row["arms"])
        self.joints = list(robot.joint_names)
        self.bodies = list(robot.data.body_names)
        if len(set(self.joints)) != len(self.joints) or len(set(self.bodies)) != len(self.bodies):
            raise ValueError("team_arm_native_names_ambiguous")
        for arm in self.arms:
            if any(name not in self.joints for name in arm["joint_names"]) or any(
                    name not in self.bodies for name in (arm["base_link"], arm["tool_link"])):
                raise ValueError("team_arm_native_binding_missing")
            if self.bodies.index(arm["tool_link"]) == 0:
                raise ValueError("team_arm_native_jacobian_body_invalid")

    def state(self) -> dict[str, Any]:
        self._live_jacobians()  # Prove the declared fixed, independent bases before disclosure.
        poses = self.array(self.robot.data.body_pose_w)[0]
        if poses.shape != (len(self.bodies), 7) or not np.isfinite(poses).all():
            raise ValueError("team_arm_native_pose_invalid")
        return {"arm_names": [arm["name"] for arm in self.arms],
                "joint_names": list(self.row["joint_names"]),
                "arm_joint_names": {arm["name"]: list(arm["joint_names"]) for arm in self.arms},
                "arm_gripper_names": {arm["name"]: arm["gripper_name"] for arm in self.arms},
                "arm_eef_pose": {arm["name"]: relative_tcp_pose(
                    base_pose=poses[self.bodies.index(arm["base_link"])],
                    tool_pose=poses[self.bodies.index(arm["tool_link"])], tool_from_tcp=arm["tool_from_tcp"])
                    for arm in self.arms}}

    def _live_jacobians(self) -> Any:
        # Pinned Beta2 uses root_view and returns world-space COM Jacobians.
        jacobians = self.array(self.robot.root_view.get_jacobians())[0]
        if (not bool(self.robot.is_fixed_base) or jacobians.shape != (len(self.bodies)-1, 6, len(self.joints))
                or not np.isfinite(jacobians).all()):
            raise ValueError("team_arm_native_jacobian_shape_invalid")
        for arm in self.arms:
            base, tool = (self.bodies.index(arm[key]) for key in ("base_link", "tool_link"))
            if base and np.any(np.abs(jacobians[base-1]) > 1e-8):
                raise ValueError("team_arm_native_base_not_fixed")
            undeclared = [i for i, name in enumerate(self.joints) if name not in arm["joint_names"]]
            if np.any(np.abs(jacobians[tool-1][:, undeclared]) > 1e-8):
                raise ValueError("team_arm_native_joint_influence_undeclared")
        return jacobians

    def joint_targets(self, *, positions: Any, orientations: Any) -> tuple[dict[str, float], list[dict[str, Any]]]:
        q = self.array(self.robot.data.joint_pos)[0]
        poses = self.array(self.robot.data.body_pose_w)[0]
        com = self.array(self.robot.data.body_com_pose_w)[0]
        jacobians = self._live_jacobians()
        result, receipts = {}, []
        for index, arm in enumerate(self.arms):
            ids = [self.joints.index(name) for name in arm["joint_names"]]
            body = self.bodies.index(arm["tool_link"])
            target, receipt = pose_servo_step(
                joint_positions=q[ids], jacobian_world_com=jacobians[body-1][:, ids],
                body_pose_world=poses[body], body_com_position_world=com[body, :3],
                base_pose_world=poses[self.bodies.index(arm["base_link"])], tool_from_tcp=arm["tool_from_tcp"],
                target_position_m=positions[index], target_orientation_xyzw=orientations[index],
                position_limits=[self.row["joint_position_limits_rad"][name] for name in arm["joint_names"]],
                velocity_limits=[self.row["joint_velocity_limits"][name] for name in arm["joint_names"]],
                step_seconds=1.0 / float(self.row["control_hz"]))
            result.update(zip(arm["joint_names"], target, strict=True))
            receipts.append({"arm_name": arm["name"], "jacobian_provider": "physx", **receipt})
        return result, receipts
