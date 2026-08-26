"""Bind a task-neutral Arena build to the shared episode environment seam.

The Arena builder owns native scene names and the articulated readback owns
task state.  This module only joins those measured bindings to the existing
Isaac episode adapter; it does not select a scene, object class, joint name, or
task outcome.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .adp009d_isaac_episode_adapter import IsaacEpisodeAdapter
from .native_franka_pose_servo import (
    DEFAULT_VELOCITY_FEEDFORWARD_SCALE,
    PHYSX_DLS_JOINT_LIMIT_AVOIDANCE_GAIN,
    PHYSX_DLS_JOINT_LIMIT_AVOIDANCE_MARGIN_RAD,
    PHYSX_DLS_POSTURE_NULLSPACE_GAIN,
)
from .rigid_frame_transforms import apply_rigid_offset, rigid_offset_in_body_frame


SCHEMA_VERSION = "native_task_episode_environment.v3"

# A globally solved endpoint proves that the pose is reachable, but replaying
# that endpoint as a bounded joint-space setpoint does not preserve the path of
# the fingertips.  In C9 the contact endpoint was valid while the interpolated
# joint path swept the measured pad midpoint 94 mm sideways into the door.
# Contact-seeking motion must therefore remain on the live Cartesian servo;
# free-space poses can still reuse the globally selected branch that makes
# their arrival deterministic.
CARTESIAN_CONTACT_PHASE_IDS = frozenset({"contact_open", "contact_close"})
# NVIDIA's shipped manipulation guidance uses differential IK for precision
# approach and contact, while reserving joint-space interpolation for long
# free-space transport.  C21 proved why the boundary matters: replaying the
# global approach posture drove panda_joint5 onto its lower limit before the
# contact controller got a chance to avoid it.
PHYSX_DLS_PRECISION_PHASE_IDS = frozenset(
    {"approach", *CARTESIAN_CONTACT_PHASE_IDS}
)


class NativeTaskEpisodeEnvironmentError(ValueError):
    """Stable failures while binding one native construction to episodes."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _gripper_endpoint(
    convention: Mapping[str, Any], *, command_field: str
) -> tuple[float, float]:
    try:
        command = float(convention[command_field])
        separation = float(convention["finger_separation_m"][str(command)])
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_gripper_convention_invalid"]
        ) from exc
    if not math.isfinite(command) or not math.isfinite(separation):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_gripper_convention_invalid"]
        )
    return command, separation


def build_native_task_episode_environment(
    *,
    built: Any,
    gripper_convention: Mapping[str, Any],
    servo: Any,
    task_readback: Any | None,
    to_tensor: Any,
    scripted_pose_joint_targets: Sequence[Mapping[str, Any]] | None = None,
    scripted_pose_phase_targets: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[IsaacEpisodeAdapter, dict[str, Any]]:
    """Create the shared control/policy adapter from native Arena readbacks."""

    plan = getattr(built, "plan", None)
    env = getattr(built, "env", None)
    scene_asset_names = getattr(built, "scene_asset_names", None)
    camera_scene_names = getattr(built, "camera_scene_names", None)
    if (
        not isinstance(plan, Mapping)
        or env is None
        or not isinstance(scene_asset_names, Mapping)
        or not isinstance(camera_scene_names, Mapping)
    ):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_arena_build_invalid"]
        )
    task_kind = str(plan.get("task_kind") or "")
    if task_kind not in {"rigid_pick_place", "articulated_open_close"}:
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_task_kind_unsupported"]
        )
    try:
        seed = int(plan["scenario"]["seed"])
        control_frequency_hz = float(plan["cadence"]["control_frequency_hz"])
        action_dim = int(env.unwrapped.action_manager.total_action_dim)
        scene = env.unwrapped.scene
        robot = scene["robot"]
        task_object = scene[scene_asset_names["task_object"]]
        joint_wrench_sensor = scene["robot_joint_wrench"]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_native_binding_missing"]
        ) from exc
    if (
        not math.isfinite(control_frequency_hz)
        or control_frequency_hz <= 0.0
        or action_dim != 8
    ):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_action_or_cadence_invalid"]
        )
    closed_command, closed_separation = _gripper_endpoint(
        gripper_convention, command_field="closed_command"
    )
    open_command, open_separation = _gripper_endpoint(
        gripper_convention, command_field="open_command"
    )
    if (
        closed_command == open_command
        or open_separation - closed_separation <= 1.0e-6
    ):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_gripper_convention_invalid"]
        )
    if task_kind == "articulated_open_close" and (
        task_readback is None
        or not callable(getattr(task_readback, "read_task_sample", None))
        or not callable(
            getattr(servo, "current_gripper_pad_readback", None)
        )
    ):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_task_readback_missing"]
        )
    if (
        not callable(getattr(servo, "action_for_grasp_target", None))
        or not callable(
            getattr(servo, "action_for_grasp_target_physx_dls", None)
        )
        or not callable(getattr(servo, "reset_command_state", None))
    ):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_pose_servo_invalid"]
        )
    try:
        reset_controlled_body_pose = [
            float(value) for value in servo.current_body_pose_world()
        ]
        reset_grasp_frame_pose = [
            float(value) for value in servo.current_grasp_frame_pose_world()
        ]
    except (AttributeError, TypeError, ValueError) as exc:
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_controlled_body_pose_missing"]
        ) from exc
    if any(
        len(pose) != 7 or not all(math.isfinite(value) for value in pose)
        for pose in (reset_controlled_body_pose, reset_grasp_frame_pose)
    ):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_controlled_body_pose_missing"]
        )
    reset_orientation = reset_grasp_frame_pose[3:7]

    # Isaac's rendered wrist camera follows the articulation through Fabric,
    # while its sensor-buffer world pose can remain frozen at initialization.
    # That exact mismatch appeared in the retained GR00T episode: RGB changed
    # substantially while every wrist calibration digest stayed byte-identical.
    # Measure the rigid camera-to-grasp-frame mount once at reset, then rebuild
    # the evidence pose from the live measured grasp frame on every observation.
    try:
        wrist_camera_scene_name = str(camera_scene_names["wrist"])
        wrist_camera = scene[wrist_camera_scene_name]
        reset_wrist_position = [
            float(value) for value in to_tensor(wrist_camera.data.pos_w)[0]
        ]
        reset_wrist_quaternion = [
            float(value)
            for value in to_tensor(wrist_camera.data.quat_w_opengl)[0]
        ]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_wrist_camera_binding_missing"]
        ) from exc
    quaternion_norm = math.sqrt(
        sum(value * value for value in reset_wrist_quaternion)
    )
    if (
        not wrist_camera_scene_name
        or len(reset_wrist_position) != 3
        or len(reset_wrist_quaternion) != 4
        or not all(
            math.isfinite(value)
            for value in [*reset_wrist_position, *reset_wrist_quaternion]
        )
        or abs(quaternion_norm - 1.0) > 1.0e-5
    ):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_wrist_camera_pose_invalid"]
        )
    wrist_mount_position_controlled_body, wrist_mount_quaternion_controlled_body = (
        rigid_offset_in_body_frame(
            body_position_world=reset_controlled_body_pose[:3],
            body_quaternion_world_xyzw=reset_controlled_body_pose[3:7],
            child_position_world=reset_wrist_position,
            child_quaternion_world_xyzw=reset_wrist_quaternion,
        )
    )

    def live_camera_pose(
        camera_name: str,
    ) -> tuple[list[float], list[float]] | None:
        if str(camera_name) != wrist_camera_scene_name:
            return None
        try:
            live_controlled_body_pose = [
                float(value) for value in servo.current_body_pose_world()
            ]
        except (AttributeError, TypeError, ValueError) as exc:
            raise NativeTaskEpisodeEnvironmentError(
                ["native_task_episode_live_wrist_camera_pose_missing"]
            ) from exc
        if len(live_controlled_body_pose) != 7 or not all(
            math.isfinite(value) for value in live_controlled_body_pose
        ):
            raise NativeTaskEpisodeEnvironmentError(
                ["native_task_episode_live_wrist_camera_pose_missing"]
            )
        return apply_rigid_offset(
            body_position_world=live_controlled_body_pose[:3],
            body_quaternion_world_xyzw=live_controlled_body_pose[3:7],
            offset_position_body=wrist_mount_position_controlled_body,
            offset_quaternion_body_xyzw=wrist_mount_quaternion_controlled_body,
        )

    joint_target_rows: list[dict[str, Any]] = []
    joint_targets_by_pose: dict[
        tuple[tuple[float, ...], tuple[float, ...]], dict[str, Any]
    ] = {}
    for index, raw in enumerate(scripted_pose_joint_targets or []):
        try:
            phase_id = str(raw["phase_id"])
            position = [float(value) for value in raw["target_position_world_m"]]
            quaternion = [
                float(value) for value in raw["target_quaternion_world_xyzw"]
            ]
            joints = [float(value) for value in raw["joint_positions_rad"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeTaskEpisodeEnvironmentError(
                [f"native_task_episode_scripted_joint_target_invalid:{index}"]
            ) from exc
        if (
            not phase_id
            or len(position) != 3
            or len(joints) != 7
            or len(quaternion) != 4
            or not all(
                math.isfinite(value)
                for value in [
                    *position,
                    *joints,
                    *quaternion,
                ]
            )
        ):
            raise NativeTaskEpisodeEnvironmentError(
                [f"native_task_episode_scripted_joint_target_invalid:{index}"]
            )
        key = (tuple(position), tuple(quaternion))
        if key in joint_targets_by_pose:
            raise NativeTaskEpisodeEnvironmentError(
                ["native_task_episode_scripted_joint_target_duplicate"]
            )
        row = {
            "phase_id": phase_id,
            "target_position_world_m": position,
            "target_quaternion_world_xyzw": quaternion,
            "joint_positions_rad": joints,
        }
        joint_targets_by_pose[key] = row
        joint_target_rows.append(row)
    if joint_target_rows and not callable(
        getattr(servo, "action_for_joint_target", None)
    ):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_joint_target_servo_invalid"]
        )

    # A precision phase remains a live-PhysX-DLS phase even when the optional
    # off-sim PINK preflight could not supply a posture seed.  C79 introduced
    # that fail-open preflight policy, but pose dispatch previously inferred
    # precision authority only from *solved* joint-target rows.  With no row,
    # the callback silently fell back to PINK -- the exact veto C79 intended
    # to remove.  Bind phase authority independently from optional seeds.
    precision_phase_target_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(scripted_pose_phase_targets or []):
        try:
            phase_id = str(raw["phase_id"])
            position = [float(value) for value in raw["target_position_world_m"]]
            quaternion = [
                float(value) for value in raw["target_quaternion_world_xyzw"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeTaskEpisodeEnvironmentError(
                [f"native_task_episode_scripted_phase_target_invalid:{index}"]
            ) from exc
        if (
            not phase_id
            or len(position) != 3
            or len(quaternion) != 4
            or not all(math.isfinite(value) for value in [*position, *quaternion])
        ):
            raise NativeTaskEpisodeEnvironmentError(
                [f"native_task_episode_scripted_phase_target_invalid:{index}"]
            )
        if phase_id not in PHYSX_DLS_PRECISION_PHASE_IDS:
            continue
        precision_phase_target_rows.append(
            {
                "phase_id": phase_id,
                "target_position_world_m": position,
                "target_quaternion_world_xyzw": quaternion,
            }
        )
    declared_precision_phase_ids = {
        row["phase_id"]
        for row in [*joint_target_rows, *precision_phase_target_rows]
        if row["phase_id"] in PHYSX_DLS_PRECISION_PHASE_IDS
    }
    seeded_precision_phase_ids = {
        row["phase_id"]
        for row in joint_target_rows
        if row["phase_id"] in PHYSX_DLS_PRECISION_PHASE_IDS
    }

    def reset() -> None:
        env.reset(seed=seed)

    def diagnostic_checkpoint_reset(
        arm_joint_positions_rad: Sequence[float],
        task_joint_positions_rad: Mapping[str, float],
    ) -> None:
        """Inject robot/task joints after a normal deterministic reset."""

        try:
            robot_joint_names = list(
                getattr(robot, "joint_names", None)
                or robot.data.joint_names
            )
            robot_position = robot.data.joint_pos.clone()
            robot_velocity = robot.data.joint_vel.clone()
            robot_velocity.zero_()
            arm_indices = [
                robot_joint_names.index(f"panda_joint{index}")
                for index in range(1, 8)
            ]
            for index, value in zip(
                arm_indices, arm_joint_positions_rad, strict=True
            ):
                robot_position[0, index] = float(value)
            robot.write_joint_state_to_sim(robot_position, robot_velocity)

            if task_kind == "articulated_open_close":
                task_joint_names = list(
                    getattr(task_object, "joint_names", None)
                    or task_object.data.joint_names
                )
                native_names = dict(
                    plan["task_sample_binding"].get("native_joint_names") or {}
                )
                task_position = task_object.data.joint_pos.clone()
                task_velocity = task_object.data.joint_vel.clone()
                task_velocity.zero_()
                for logical_name, value in task_joint_positions_rad.items():
                    native_name = str(native_names.get(logical_name, logical_name))
                    task_position[0, task_joint_names.index(native_name)] = float(value)
                task_object.write_joint_state_to_sim(
                    task_position, task_velocity
                )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise NativeTaskEpisodeEnvironmentError(
                ["native_task_episode_diagnostic_checkpoint_write_failed"]
            ) from exc

    def articulated_task_sample() -> dict[str, Any]:
        raw = task_readback.read_task_sample()
        frame = servo.current_gripper_pad_readback()
        measured = frame.get("measured") if isinstance(frame, Mapping) else None
        if not isinstance(raw, Mapping) or not isinstance(measured, Mapping):
            raise NativeTaskEpisodeEnvironmentError(
                ["native_task_episode_gripper_measurement_invalid"]
            )
        try:
            separation = float(measured["pad_separation_m"])
            body_position = [
                float(value)
                for value in measured["controlled_body_position_world_m"]
            ]
            body_quaternion = [
                float(value)
                for value in measured[
                    "controlled_body_quaternion_world_xyzw"
                ]
            ]
            midpoint = [
                float(value) for value in measured["pad_midpoint_world_m"]
            ]
            finger_positions = {
                str(name): [float(value) for value in position]
                for name, position in measured[
                    "finger_body_positions_world_m"
                ].items()
            }
            pad_centers = {
                str(name): [float(value) for value in position]
                for name, position in measured["pad_centers_world_m"].items()
            }
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise NativeTaskEpisodeEnvironmentError(
                ["native_task_episode_gripper_measurement_invalid"]
            ) from exc
        finger_values = [
            value
            for position in finger_positions.values()
            for value in position
        ]
        pad_values = [
            value for position in pad_centers.values() for value in position
        ]
        if (
            not math.isfinite(separation)
            or separation < 0.0
            or len(body_position) != 3
            or len(body_quaternion) != 4
            or len(midpoint) != 3
            or len(finger_positions) != 2
            or len(pad_centers) != 2
            or any(len(position) != 3 for position in finger_positions.values())
            or any(len(position) != 3 for position in pad_centers.values())
            or not all(
                math.isfinite(value)
                for value in [
                    *body_position,
                    *body_quaternion,
                    *midpoint,
                    *finger_values,
                    *pad_values,
                ]
            )
        ):
            raise NativeTaskEpisodeEnvironmentError(
                ["native_task_episode_gripper_measurement_invalid"]
            )
        sample = dict(raw)
        sample.update(
            {
                "gripper_width_m": separation,
                "gripper_controlled_body_position_world_m": body_position,
                "gripper_controlled_body_quaternion_world_xyzw": body_quaternion,
                "gripper_pad_midpoint_world_m": midpoint,
                "gripper_finger_body_positions_world_m": finger_positions,
                "gripper_pad_centers_world_m": pad_centers,
                "gripper_measurement_authority": (
                    "native_finger_body_pose_plus_probe_sealed_pad_center_offset"
                ),
            }
        )
        return sample

    def bounded_joint_action(**kwargs: Any) -> list[float]:
        """Command a solved joint posture under the servo's actuator bounds.

        Plan rows carrying raw joint positions bypass the servo entirely, so
        nothing held them to what the actuator can pull.  C33 measured the
        cost: its 118-row entry ramp ran the command ahead of a lagging wrist,
        spent 37% of those rows saturated, and ended further from the handle
        than the shorter ramp before it.  Routing the same targets through the
        servo applies the slew and per-joint feasible-lead bounds, so the
        command can never outrun the joint that has to follow it.
        """

        action, _diagnostic = servo.action_for_joint_target(
            target_joint_positions_rad=kwargs["target_joint_positions_rad"],
            gripper_command=kwargs["gripper_command"],
            max_joint_delta_rad=kwargs["max_joint_delta_rad"],
            max_joint_setpoint_lead_rad=kwargs["max_joint_setpoint_lead_rad"],
        )
        return [float(value) for value in action]

    def scripted_pose_action(**kwargs: Any) -> list[float]:
        phase_id = str(kwargs.get("phase_id") or "")
        quaternion = kwargs.get("target_quaternion_world_xyzw")
        resolved_quaternion = reset_orientation if quaternion is None else quaternion
        pose_key = (
            tuple(float(value) for value in kwargs["target_position_world_m"]),
            tuple(float(value) for value in resolved_quaternion),
        )
        joint_target = joint_targets_by_pose.get(pose_key)
        common = {
            "gripper_command": kwargs["gripper_command"],
            "max_joint_delta_rad": kwargs["max_joint_delta_rad"],
            "max_joint_setpoint_lead_rad": kwargs["max_joint_setpoint_lead_rad"],
            # Controls replay the dynamics construction qualified, so the
            # feedforward has to be the same on both sides.
            "velocity_feedforward_scale": kwargs.get(
                "velocity_feedforward_scale", DEFAULT_VELOCITY_FEEDFORWARD_SCALE
            ),
        }
        diagnostic_preferred_posture = kwargs.get(
            "preferred_posture_joint_positions_rad"
        )
        if diagnostic_preferred_posture is not None:
            # Reset-isolated contact diagnostics must be able to compare the
            # joint-replay controller with the live PhysX TCP controller for
            # the *same* preferred IK branch.  Routing this through the normal
            # pose lookup would silently replace the tested branch with the
            # plan-selected one, defeating the A/B measurement.
            action, _diagnostic = servo.action_for_grasp_target_physx_dls(
                target_position_world_m=kwargs["target_position_world_m"],
                target_grasp_frame_quaternion_world_xyzw=resolved_quaternion,
                preferred_posture_joint_positions_rad=[
                    float(value) for value in diagnostic_preferred_posture
                ],
                **common,
            )
        elif joint_target is not None and joint_target[
            "phase_id"
        ] in PHYSX_DLS_PRECISION_PHASE_IDS:
            action, _diagnostic = servo.action_for_grasp_target_physx_dls(
                target_position_world_m=kwargs["target_position_world_m"],
                target_grasp_frame_quaternion_world_xyzw=resolved_quaternion,
                # C24 proved the off-sim solver can reach contact to 4.8 mm,
                # while the live DLS controller converges to a different
                # redundant-arm posture and stalls 15.0 mm away.  Preserve
                # the Cartesian path, but guide its one redundant DOF toward
                # the exact receipt-bound whole-arm solution.  The correction
                # is projected through the full six-row task null space, so it
                # cannot replace or weaken measured pose arrival.
                preferred_posture_joint_positions_rad=joint_target[
                    "joint_positions_rad"
                ],
                **common,
            )
        elif phase_id in PHYSX_DLS_PRECISION_PHASE_IDS:
            action, _diagnostic = servo.action_for_grasp_target_physx_dls(
                target_position_world_m=kwargs["target_position_world_m"],
                target_grasp_frame_quaternion_world_xyzw=resolved_quaternion,
                preferred_posture_joint_positions_rad=None,
                **common,
            )
        elif joint_target is not None:
            action, _diagnostic = servo.action_for_joint_target(
                target_joint_positions_rad=joint_target["joint_positions_rad"],
                **common,
            )
        else:
            action, _diagnostic = servo.action_for_grasp_target(
                target_position_world_m=kwargs["target_position_world_m"],
                target_grasp_frame_quaternion_world_xyzw=resolved_quaternion,
                **common,
            )
        return [float(value) for value in action]

    adapter = IsaacEpisodeAdapter(
        env=env,
        robot=robot,
        rigid_task_object=(task_object if task_kind == "rigid_pick_place" else None),
        action_dim=action_dim,
        reset_seed=seed,
        to_torch=to_tensor,
        gripper_closed_width_m=closed_separation,
        gripper_open_width_m=open_separation,
        reset_callback=reset,
        diagnostic_checkpoint_reset_callback=diagnostic_checkpoint_reset,
        scripted_pose_controller_reset_callback=servo.reset_command_state,
        simulation_step_seconds=1.0 / control_frequency_hz,
        scripted_pose_action_callback=scripted_pose_action,
        bounded_joint_action_callback=bounded_joint_action,
        task_sample_callback=(
            articulated_task_sample
            if task_kind == "articulated_open_close"
            else None
        ),
        grasp_frame_pose_callback=servo.current_grasp_frame_pose_world,
        grasp_frame_fk_callback=(
            servo.predicted_grasp_frame_pose_world
            if callable(
                getattr(servo, "predicted_grasp_frame_pose_world", None)
            )
            else None
        ),
        camera_scene_names=camera_scene_names,
        camera_pose_callback=live_camera_pose,
        joint_wrench_sensor=joint_wrench_sensor,
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "task_kind": task_kind,
        "action_dimension": action_dim,
        "reset_seed": seed,
        "control_frequency_hz": control_frequency_hz,
        "camera_scene_names": dict(camera_scene_names),
        "camera_world_pose_bindings": {
            "external": {
                "scene_name": str(camera_scene_names.get("external") or ""),
                "source": "isaac_sensor_buffer_static_camera",
                "recomputed_each_observation": False,
            },
            "wrist": {
                "scene_name": wrist_camera_scene_name,
                "source": (
                    "live_controlled_body_plus_reset_measured_rigid_mount_offset"
                ),
                "recomputed_each_observation": True,
                "sensor_buffer_static_pose_workaround": True,
                "mount_offset_position_controlled_body_m": (
                    wrist_mount_position_controlled_body
                ),
                "mount_offset_quaternion_controlled_body_xyzw": (
                    wrist_mount_quaternion_controlled_body
                ),
            },
            "overview": {
                "scene_name": str(camera_scene_names.get("overview") or ""),
                "source": "isaac_sensor_buffer_static_camera",
                "recomputed_each_observation": False,
                "policy_input": False,
            },
        },
        "task_state_source": (
            "native_articulated_task_readback"
            if task_kind == "articulated_open_close"
            else "native_rigid_body_readback"
        ),
        "diagnostic_checkpoint_reset": {
            "available": True,
            "state_components": [
                "arm_joint_positions_rad",
                "task_joint_positions_rad",
            ],
            "claim_boundary": (
                "reset_isolated_diagnostic_initialization_only;not_phase_"
                "admission_or_task_success"
            ),
        },
        "scripted_pose_source": (
            "global_ik_free_space_with_live_physx_jacobian_precision_servo_"
            "and_full_pose_nullspace_joint_limit_avoidance"
            if any(
                row["phase_id"] in CARTESIAN_CONTACT_PHASE_IDS
                for row in joint_target_rows
            )
            else "live_physx_jacobian_precision_servo_without_offsim_posture_seed"
            if precision_phase_target_rows
            else "construction_global_ik_joint_target_with_native_pose_fallback"
            if joint_target_rows
            else "native_franka_differential_ik_servo"
        ),
        "cartesian_contact_phase_ids": sorted(
            {
                row["phase_id"]
                for row in [*joint_target_rows, *precision_phase_target_rows]
                if row["phase_id"] in CARTESIAN_CONTACT_PHASE_IDS
            }
        ),
        "cartesian_contact_physx_dls_phase_ids": sorted(
            declared_precision_phase_ids
        ),
        "cartesian_contact_phase_controller_bindings": [
            {
                "phase_id": phase_id,
                "controller": "live_physx_full_pose_dls",
                "preferred_posture_source": (
                    "selected_global_ik_joint_target"
                    if phase_id in seeded_precision_phase_ids
                    else None
                ),
                "recovery_target_bias_preserves_controller": True,
            }
            for phase_id in sorted(declared_precision_phase_ids)
        ],
        "cartesian_contact_posture_source": (
            "selected_global_ik_joint_target_projected_through_live_physx_"
            "full_pose_jacobian_nullspace"
            if any(
                row["phase_id"] in PHYSX_DLS_PRECISION_PHASE_IDS
                for row in joint_target_rows
            )
            else "no_offsim_posture_seed_live_physx_full_pose_dls"
            if precision_phase_target_rows
            else None
        ),
        "cartesian_precision_joint_limit_avoidance_source": (
            "isaaclab_pink_combined_task_jacobian_nullspace_projection"
            if any(
                row["phase_id"] in PHYSX_DLS_PRECISION_PHASE_IDS
                for row in [*joint_target_rows, *precision_phase_target_rows]
            )
            else None
        ),
        "cartesian_contact_posture_nullspace_gain": (
            PHYSX_DLS_POSTURE_NULLSPACE_GAIN
            if seeded_precision_phase_ids
            else None
        ),
        "cartesian_precision_joint_limit_avoidance_gain": (
            PHYSX_DLS_JOINT_LIMIT_AVOIDANCE_GAIN
            if any(
                row["phase_id"] in PHYSX_DLS_PRECISION_PHASE_IDS
                for row in [*joint_target_rows, *precision_phase_target_rows]
            )
            else None
        ),
        "cartesian_precision_joint_limit_avoidance_margin_rad": (
            PHYSX_DLS_JOINT_LIMIT_AVOIDANCE_MARGIN_RAD
            if declared_precision_phase_ids
            else None
        ),
        "scripted_pose_joint_targets": joint_target_rows,
        "scripted_pose_precision_phase_targets": precision_phase_target_rows,
        "joint_wrench_source": "IsaacLab JointWrenchSensor force+torque",
        "joint_wrench_convention": "incoming_joint_frame",
        "controlled_body_orientation_source": "native_body_pose_readback",
        "grasp_frame_pose_source": (
            "native_franka_pose_servo.measured_controlled_body_to_grasp_frame"
        ),
        "gripper_state_source": (
            "native_finger_body_pose_plus_probe_sealed_pad_offset_each_sample"
            if task_kind == "articulated_open_close"
            else None
        ),
        "gripper_command_mapping": {
            "closed_command": closed_command,
            "open_command": open_command,
            "closed_finger_separation_m": closed_separation,
            "open_finger_separation_m": open_separation,
        },
    }
    return adapter, receipt


__all__ = [
    "NativeTaskEpisodeEnvironmentError",
    "SCHEMA_VERSION",
    "build_native_task_episode_environment",
]
