"""Native differential-IK pose servo shared by construction and controls.

This is the task-neutral extraction of the controller first qualified by the
ADP-009D rigid rehearsal.  It controls the measured midpoint between the two
finger bodies, rotates PhysX's world-frame Jacobian into the robot root frame,
and bounds both command slew and lead before emitting the same absolute 8-D
Arena action consumed by learned policies.

Isaac imports occur only when the class is instantiated, so binding and helper
contracts remain hermetically testable on a CPU host.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .native_franka_action_math import (
    bounded_absolute_joint_setpoint,
    controlled_body_pose_for_grasp_frame_target,
)
from .native_pose_transforms import (
    pose_world_to_base,
    world_to_base_rotation_row_major_xyzw,
)


SCHEMA_VERSION = "native_franka_pose_servo.v1"
ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
FINGER_BODY_NAMES = ("left_inner_finger", "right_inner_finger")
CONTROLLED_BODY_CANDIDATES = ("panda_hand", "base_link", "panda_link7")


class NativeFrankaPoseServoError(RuntimeError):
    """Stable binding/controller failures at the native action seam."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


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


class NativeFrankaDifferentialIkServo:
    """One deterministic native pose-servo action per control tick."""

    def __init__(self, *, env: Any, robot: Any):
        import torch
        from isaaclab.controllers import DifferentialIKController
        from isaaclab.controllers import DifferentialIKControllerCfg
        from isaaclab.utils.math import subtract_frame_transforms

        self._env = env
        self._robot = robot
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
        self._controller = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            num_envs=1,
            device=env.unwrapped.device,
        )
        base_pose = self._to_torch(robot.data.root_pose_w)[0, :7]
        self._base_pose = [float(value) for value in base_pose]
        rotation = world_to_base_rotation_row_major_xyzw(self._base_pose[3:7])
        self._world_to_root = torch.tensor(
            [rotation], device=env.unwrapped.device, dtype=torch.float32
        ).reshape(1, 3, 3)
        self._last_command: list[float] | None = None

    def reset_command_state(self) -> None:
        self._last_command = None
        self._controller.reset()

    def current_body_pose_world(self) -> list[float]:
        pose = self._to_torch(self._robot.data.body_pose_w)[
            0, self.binding["controlled_body_index"], :7
        ]
        return [float(value) for value in pose]

    def current_grasp_frame_position_world(self) -> list[float]:
        poses = self._to_torch(self._robot.data.body_pose_w)[
            0, self.binding["finger_body_indices"], :3
        ]
        midpoint = (poses[0] + poses[1]) / 2.0
        return [float(value) for value in midpoint]

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
        target_body_quaternion_world_xyzw: Sequence[float],
        gripper_command: float,
        max_joint_delta_rad: float = 0.03,
        max_joint_setpoint_lead_rad: float = 0.20,
    ) -> tuple[list[float], dict[str, Any]]:
        body_pose = self.current_body_pose_world()
        grasp = self.current_grasp_frame_position_world()
        target_body_position, target_body_quaternion = (
            controlled_body_pose_for_grasp_frame_target(
                current_body_position_world_m=body_pose[:3],
                current_body_quaternion_world_xyzw=body_pose[3:7],
                current_grasp_frame_position_world_m=grasp,
                target_grasp_frame_position_world_m=target_position_world_m,
                target_body_quaternion_world_xyzw=target_body_quaternion_world_xyzw,
            )
        )
        position_root, quaternion_root = pose_world_to_base(
            position_world=target_body_position,
            quaternion_world_xyzw=target_body_quaternion,
            base_position_world=self._base_pose[:3],
            base_quaternion_world_xyzw=self._base_pose[3:7],
        )
        command = self._torch.tensor(
            [position_root + quaternion_root],
            device=self._env.unwrapped.device,
            dtype=self._torch.float32,
        )
        self._controller.reset()
        self._controller.set_command(command)
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
        desired = self._controller.compute(
            body_position_root,
            body_quaternion_root,
            jacobian_root,
            current,
        )
        current_values = [float(value) for value in current[0]]
        desired_values = [float(value) for value in desired[0]]
        previous = current_values if self._last_command is None else self._last_command
        bounded = bounded_absolute_joint_setpoint(
            measured_joint_positions_rad=current_values,
            desired_joint_positions_rad=desired_values,
            previous_commanded_joint_positions_rad=previous,
            max_command_slew_per_step_rad=float(max_joint_delta_rad),
            max_setpoint_lead_rad=float(max_joint_setpoint_lead_rad),
        )
        self._last_command = list(bounded)
        action = [*bounded, float(gripper_command)]
        diagnostics = {
            "target_grasp_frame_position_world_m": [
                float(value) for value in target_position_world_m
            ],
            "current_grasp_frame_position_world_m": grasp,
            "target_controlled_body_position_world_m": target_body_position,
            "current_controlled_body_position_world_m": body_pose[:3],
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
            "bounded_joint_positions_rad": bounded,
            "measured_joint_positions_rad": current_values,
        }
        return action, diagnostics


__all__ = [
    "ARM_JOINT_NAMES",
    "CONTROLLED_BODY_CANDIDATES",
    "FINGER_BODY_NAMES",
    "NativeFrankaDifferentialIkServo",
    "NativeFrankaPoseServoError",
    "SCHEMA_VERSION",
    "resolve_native_franka_pose_binding",
]
