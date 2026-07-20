"""Build a truth-bounded G1 right-hand reach/pull seed for qualification.

The seed is derived from the pinned GEAR-SONIC MuJoCo model, an attested
standing joint state, and the exact live-Isaac pelvis/target geometry.  It is
useful as owned fine-tuning or controller-qualification input.  The optional
pull follows the exact door hinge arc kinematically; it is not a
physics-validated grasp, contact-driven articulation transition, or successful
episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .g1_sonic_motion_token_conversion import SOURCE_ACTION_JOINT_NAMES
from .isaac_task_review_renderer import ARTICULATED_HANDLE_FOCUS_SCHEMA_VERSION
from .gear_sonic_joint_order_contract import (
    PINNED_WBC_SOURCE_REVISION,
    PROTOCOL_V4_FULL_JOINT_ORDER,
    validate_model_joint_names,
)


SCHEMA_VERSION = "g1_microwave_reach_seed.v2"
PINNED_G1_MODEL_SHA256 = (
    "8b68d8f06674c5c10cd2cd89764b3cfba9fabba5080b55ea67ee1dd12cf630cd"
)
RIGHT_ARM_JOINT_NAMES = SOURCE_ACTION_JOINT_NAMES[29:36]
DEFAULT_EFFECTOR = "right_hand_index_0_link"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value: Any, *, size: int, name: str) -> np.ndarray:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, (Sequence, np.ndarray)
    ):
        raise ValueError(f"{name}_missing")
    result = np.asarray([float(item) for item in value], dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name}_invalid")
    return result


def _rotation_wxyz(value: Any) -> np.ndarray:
    quaternion = _finite_vector(value, size=4, name="live_isaac_pelvis_quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("live_isaac_pelvis_quaternion_norm_invalid")
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def minimum_jerk_trajectory(
    initial: Any,
    target: Any,
    *,
    frame_count: int,
) -> np.ndarray:
    """Interpolate joint targets with zero endpoint velocity/acceleration."""

    start = np.asarray(initial, dtype=np.float64)
    end = np.asarray(target, dtype=np.float64)
    if start.ndim != 1 or end.shape != start.shape or not np.isfinite(start).all() or not np.isfinite(end).all():
        raise ValueError("g1_microwave_reach_seed_joint_targets_invalid")
    if int(frame_count) < 2:
        raise ValueError("g1_microwave_reach_seed_frame_count_invalid")
    t = np.linspace(0.0, 1.0, int(frame_count), dtype=np.float64)
    blend = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    return start[None, :] + blend[:, None] * (end - start)[None, :]


def rotate_point_around_axis(
    point: Any,
    *,
    origin: Any,
    axis: Any,
    angle_rad: float,
) -> np.ndarray:
    """Rotate a world point around a normalized hinge axis with Rodrigues' rule."""

    target = _finite_vector(point, size=3, name="door_arc_point")
    center = _finite_vector(origin, size=3, name="door_arc_origin")
    direction = _finite_vector(axis, size=3, name="door_arc_axis")
    norm = float(np.linalg.norm(direction))
    angle = float(angle_rad)
    if norm <= 1e-12 or not math.isfinite(angle):
        raise ValueError("g1_microwave_reach_seed_door_arc_invalid")
    direction /= norm
    offset = target - center
    rotated = (
        offset * math.cos(angle)
        + np.cross(direction, offset) * math.sin(angle)
        + direction * float(np.dot(direction, offset)) * (1.0 - math.cos(angle))
    )
    return center + rotated


def _load_mapping(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}_not_object")
    return dict(value)


def solve_reach_seed(
    *,
    model_path: str | Path,
    standing_initialization_path: str | Path,
    initial_policy_observation_path: str | Path,
    target_focus_report_path: str | Path | None = None,
    frame_count: int = 101,
    pull_frame_count: int = 0,
    door_open_angle_rad: float = 0.0,
    effector_name: str = DEFAULT_EFFECTOR,
    minimum_progress_m: float = 0.015,
    maximum_final_distance_m: float = 0.01,
    expected_model_sha256: str = PINNED_G1_MODEL_SHA256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a right-arm reach and optional hinge-arc pull as a 43-joint seed."""

    try:
        import mujoco  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise RuntimeError("g1_microwave_reach_seed_mujoco_missing") from exc

    model = Path(model_path).expanduser().resolve()
    standing_path = Path(standing_initialization_path).expanduser().resolve()
    observation_path = Path(initial_policy_observation_path).expanduser().resolve()
    if not model.is_file() or _sha256(model) != str(expected_model_sha256).lower():
        raise ValueError("g1_microwave_reach_seed_model_missing_or_sha256_mismatch")
    standing = _load_mapping(standing_path, name="standing_initialization")
    observation = _load_mapping(observation_path, name="initial_policy_observation")
    if standing.get("status") != "passed" or standing.get("surrogate") is not False:
        raise ValueError("g1_microwave_reach_seed_standing_attestation_invalid")
    if standing.get("pinned_wbc_source_revision") != PINNED_WBC_SOURCE_REVISION:
        raise ValueError("g1_microwave_reach_seed_wbc_revision_mismatch")
    measured = _finite_vector(
        standing.get("measured_full_joint_positions"),
        size=len(PROTOCOL_V4_FULL_JOINT_ORDER),
        name="standing_joint_positions",
    )
    context = observation.get("camera_projection_context")
    if not isinstance(context, Mapping) or context.get("status") != (
        "captured_from_live_persistent_isaac_session"
    ):
        raise ValueError("g1_microwave_reach_seed_live_projection_context_invalid")
    pelvis = context.get("live_isaac_pelvis_world_pose")
    camera = context.get("camera_contract")
    if not isinstance(pelvis, Mapping) or not isinstance(camera, Mapping):
        raise ValueError("g1_microwave_reach_seed_geometry_missing")
    pelvis_world = _finite_vector(
        pelvis.get("position_xyz"), size=3, name="live_isaac_pelvis_position"
    )
    target_source: dict[str, Any] = {
        "source": "live_isaac_camera_calibration_target",
        "artifact": {
            "path": str(observation_path),
            "sha256": _sha256(observation_path),
        },
    }
    target_world_value: Any = camera.get("calibration_target_world_xyz_m")
    focus: dict[str, Any] | None = None
    if target_focus_report_path is not None:
        focus_path = Path(target_focus_report_path).expanduser().resolve()
        focus = _load_mapping(focus_path, name="target_focus_report")
        if (
            focus.get("schema_version") != ARTICULATED_HANDLE_FOCUS_SCHEMA_VERSION
            or focus.get("status")
            != "resolved_disconnected_articulated_handle"
            or focus.get("target_prim_path") != observation.get("target_prim_path")
        ):
            raise ValueError("g1_microwave_reach_seed_target_focus_report_invalid")
        target_world_value = focus.get("target_world_xyz_m")
        target_source = {
            "source": "disconnected_articulated_handle_focus",
            "artifact": {"path": str(focus_path), "sha256": _sha256(focus_path)},
            "joint_prim_path": focus.get("joint_prim_path"),
            "selected_component_count": focus.get("selected_component_count"),
        }
    pull_count = int(pull_frame_count)
    requested_open_angle = float(door_open_angle_rad)
    if pull_count or requested_open_angle:
        if pull_count < 2 or not 0.0 < requested_open_angle <= math.pi / 2.0:
            raise ValueError("g1_microwave_reach_seed_pull_contract_invalid")
        if focus is None:
            raise ValueError("g1_microwave_reach_seed_pull_requires_handle_focus")
    target_world = _finite_vector(
        target_world_value, size=3, name="microwave_target_world_position"
    )
    target_root_relative = _rotation_wxyz(pelvis.get("quaternion_wxyz")).T @ (
        target_world - pelvis_world
    )

    mj_model = mujoco.MjModel.from_xml_path(str(model))
    data = mujoco.MjData(mj_model)
    model_joint_names: list[str] = []
    for index in range(mj_model.njnt):
        if int(mj_model.jnt_type[index]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        model_joint_names.append(
            str(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, index) or "")
        )
    validate_model_joint_names(model_joint_names)
    for joint_name, value in zip(PROTOCOL_V4_FULL_JOINT_ORDER, measured):
        joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        data.qpos[int(mj_model.jnt_qposadr[joint_id])] = float(value)
    mujoco.mj_forward(mj_model, data)
    pelvis_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    effector_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, str(effector_name)
    )
    if pelvis_id < 0 or effector_id < 0:
        raise ValueError("g1_microwave_reach_seed_required_body_missing")
    target_model = np.asarray(data.xpos[pelvis_id], dtype=np.float64) + target_root_relative
    initial_effector = np.asarray(data.xpos[effector_id], dtype=np.float64).copy()

    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    joint_limits: list[tuple[float, float]] = []
    for joint_name in RIGHT_ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0 or not bool(mj_model.jnt_limited[joint_id]):
            raise ValueError("g1_microwave_reach_seed_arm_joint_missing_or_unlimited")
        qpos_addresses.append(int(mj_model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(mj_model.jnt_dofadr[joint_id]))
        joint_limits.append(
            tuple(float(value) for value in mj_model.jnt_range[joint_id])
        )
    initial_arm = np.asarray([data.qpos[address] for address in qpos_addresses]).copy()

    def solve_target(
        requested_target_model: np.ndarray, *, max_iterations: int
    ) -> tuple[int, float]:
        iterations_used = 0
        for iterations_used in range(1, max_iterations + 1):
            mujoco.mj_forward(mj_model, data)
            error = requested_target_model - np.asarray(
                data.xpos[effector_id], dtype=np.float64
            )
            distance = float(np.linalg.norm(error))
            if distance <= 0.003:
                return iterations_used, distance
            jacobian_position = np.zeros((3, mj_model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros((3, mj_model.nv), dtype=np.float64)
            mujoco.mj_jacBody(
                mj_model,
                data,
                jacobian_position,
                jacobian_rotation,
                effector_id,
            )
            jacobian = jacobian_position[:, dof_addresses]
            damping = 0.025
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * damping * np.eye(3), error
            )
            delta -= 0.002 * (
                np.asarray([data.qpos[address] for address in qpos_addresses])
                - initial_arm
            )
            delta = np.clip(delta, -0.05, 0.05)
            for index, (address, (lower, upper)) in enumerate(
                zip(qpos_addresses, joint_limits)
            ):
                data.qpos[address] = np.clip(
                    data.qpos[address] + 0.8 * delta[index],
                    lower + 1e-5,
                    upper - 1e-5,
                )
        mujoco.mj_forward(mj_model, data)
        distance = float(
            np.linalg.norm(
                requested_target_model
                - np.asarray(data.xpos[effector_id], dtype=np.float64)
            )
        )
        return iterations_used, distance

    reach_iterations, _ = solve_target(target_model, max_iterations=600)
    mujoco.mj_forward(mj_model, data)
    final_effector = np.asarray(data.xpos[effector_id], dtype=np.float64).copy()
    initial_distance = float(np.linalg.norm(initial_effector - target_model))
    final_distance = float(np.linalg.norm(final_effector - target_model))
    progress = initial_distance - final_distance
    if progress < float(minimum_progress_m):
        raise RuntimeError("g1_microwave_reach_seed_progress_gate_failed")
    if final_distance > float(maximum_final_distance_m):
        raise RuntimeError("g1_microwave_reach_seed_final_distance_gate_failed")

    initial_by_name = dict(zip(PROTOCOL_V4_FULL_JOINT_ORDER, measured))
    initial_source = np.asarray(
        [initial_by_name[name] for name in SOURCE_ACTION_JOINT_NAMES], dtype=np.float64
    )
    final_source = initial_source.copy()
    source_index = {name: index for index, name in enumerate(SOURCE_ACTION_JOINT_NAMES)}
    solved_arm = np.asarray([data.qpos[address] for address in qpos_addresses])
    for joint_name, value in zip(RIGHT_ARM_JOINT_NAMES, solved_arm):
        final_source[source_index[joint_name]] = float(value)
    trajectory = minimum_jerk_trajectory(
        initial_source, final_source, frame_count=int(frame_count)
    )
    pull_report: dict[str, Any] | None = None
    if pull_count:
        assert focus is not None
        hinge_world = _finite_vector(
            focus.get("hinge_world_xyz_m"), size=3, name="door_hinge_world_position"
        )
        joint_world_axis = _finite_vector(
            focus.get("joint_world_axis_xyz"), size=3, name="door_joint_world_axis"
        )
        lower_limit_degrees = float(focus.get("joint_lower_limit_degrees"))
        upper_limit_degrees = float(focus.get("joint_upper_limit_degrees"))
        if (
            not math.isfinite(lower_limit_degrees)
            or not math.isfinite(upper_limit_degrees)
            or lower_limit_degrees >= upper_limit_degrees
            or requested_open_angle
            > math.radians(upper_limit_degrees - lower_limit_degrees) + 1e-9
        ):
            raise ValueError("g1_microwave_reach_seed_door_joint_limits_invalid")
        # This asset is closed at its upper limit (0 deg) and opens toward its
        # lower limit (-90 deg), so the requested opening is a negative angle.
        signed_open_angle = -requested_open_angle
        angle_samples = minimum_jerk_trajectory(
            [0.0], [signed_open_angle], frame_count=pull_count
        )[:, 0]
        pull_rows: list[np.ndarray] = []
        pull_distances: list[float] = []
        pull_iterations: list[int] = []
        final_handle_world = target_world.copy()
        for angle in angle_samples[1:]:
            handle_world = rotate_point_around_axis(
                target_world,
                origin=hinge_world,
                axis=joint_world_axis,
                angle_rad=float(angle),
            )
            handle_root_relative = _rotation_wxyz(
                pelvis.get("quaternion_wxyz")
            ).T @ (handle_world - pelvis_world)
            handle_model = (
                np.asarray(data.xpos[pelvis_id], dtype=np.float64)
                + handle_root_relative
            )
            iterations_used, distance = solve_target(
                handle_model, max_iterations=300
            )
            if distance > float(maximum_final_distance_m):
                raise RuntimeError(
                    "g1_microwave_reach_seed_pull_distance_gate_failed"
                )
            row = final_source.copy()
            for joint_name, address in zip(RIGHT_ARM_JOINT_NAMES, qpos_addresses):
                row[source_index[joint_name]] = float(data.qpos[address])
            pull_rows.append(row)
            pull_distances.append(distance)
            pull_iterations.append(iterations_used)
            final_handle_world = handle_world
        trajectory = np.concatenate(
            (trajectory, np.asarray(pull_rows, dtype=np.float64)), axis=0
        )
        pull_report = {
            "requested_door_open_observable_transition_rad": requested_open_angle,
            "signed_joint_angle_rad": signed_open_angle,
            "hinge_world_xyz_m": hinge_world.tolist(),
            "joint_world_axis_xyz": (joint_world_axis / np.linalg.norm(joint_world_axis)).tolist(),
            "joint_lower_limit_degrees": lower_limit_degrees,
            "joint_upper_limit_degrees": upper_limit_degrees,
            "final_handle_target_world_xyz_m": final_handle_world.tolist(),
            "frame_count_including_reach_endpoint": pull_count,
            "appended_frame_count": len(pull_rows),
            "maximum_ik_error_m": max(pull_distances),
            "final_ik_error_m": pull_distances[-1],
            "maximum_solver_iterations": max(pull_iterations),
            "kinematic_arc_following_proven": True,
            "physics_validated_contact": False,
        }
    trajectory = trajectory.astype(np.float32)
    trajectory_sha256 = hashlib.sha256(trajectory.tobytes(order="C")).hexdigest()
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "qualified_right_arm_reach_and_kinematic_pull_seed"
            if pull_report is not None
            else "qualified_right_arm_reach_seed"
        ),
        "blockers": [],
        "model": {
            "path": str(model),
            "sha256": _sha256(model),
            "wbc_source_revision": PINNED_WBC_SOURCE_REVISION,
        },
        "inputs": {
            "standing_initialization": {
                "path": str(standing_path),
                "sha256": _sha256(standing_path),
            },
            "initial_policy_observation": {
                "path": str(observation_path),
                "sha256": _sha256(observation_path),
            },
        },
        "solver": {
            "method": "right_arm_only_damped_least_squares_mujoco_body_jacobian",
            "iterations": reach_iterations,
            "effector": str(effector_name),
            "minimum_progress_m": float(minimum_progress_m),
            "maximum_final_distance_m": float(maximum_final_distance_m),
        },
        "geometry": {
            "live_isaac_pelvis_world_xyz_m": pelvis_world.tolist(),
            "microwave_target_world_xyz_m": target_world.tolist(),
            "target_model_root_relative_xyz_m": target_root_relative.tolist(),
            "target_source": target_source,
            "initial_distance_m": initial_distance,
            "final_distance_m": final_distance,
            "progress_m": progress,
        },
        "joint_targets": {
            name: float(value) for name, value in zip(RIGHT_ARM_JOINT_NAMES, solved_arm)
        },
        "joint_limits": {
            name: [lower, upper]
            for name, (lower, upper) in zip(RIGHT_ARM_JOINT_NAMES, joint_limits)
        },
        "pull": pull_report,
        "trajectory": {
            "frame_count": int(trajectory.shape[0]),
            "fps": 50.0,
            "duration_seconds": float(trajectory.shape[0] / 50.0),
            "joint_count": int(trajectory.shape[1]),
            "joint_names": list(SOURCE_ACTION_JOINT_NAMES),
            "sha256": trajectory_sha256,
            "only_right_arm_changes": bool(
                np.allclose(
                    trajectory[:, :29],
                    np.tile(initial_source[:29], (trajectory.shape[0], 1)),
                    rtol=0.0,
                    atol=1e-7,
                )
                and np.allclose(
                    trajectory[:, 36:],
                    np.tile(initial_source[36:], (trajectory.shape[0], 1)),
                    rtol=0.0,
                    atol=1e-7,
                )
            ),
        },
        "fixed_base_upright_attestation": {
            "status": "passed_for_generated_reach_seed",
            "fixed_base": True,
            "upright": True,
            "basis": (
                "live_upright_standing_input_and_only_right_arm_joint_columns_change;"
                " conversion_must_hold_the_root_anchor_identity"
            ),
            "trajectory_sha256": trajectory_sha256,
        },
        "claim_boundary": {
            "kinematic_reach_gate_passed": True,
            "kinematic_door_arc_following_proven": pull_report is not None,
            "physics_validated_grasp_proven": False,
            "microwave_door_articulation_transition_proven": False,
            "task_success_proven": False,
            "trained_checkpoint_produced": False,
        },
    }
    if not report["trajectory"]["only_right_arm_changes"]:
        raise AssertionError("g1_microwave_reach_seed_non_arm_joint_drift")
    return trajectory, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a pinned-model G1 microwave right-arm reach/pull seed."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--standing-initialization", required=True)
    parser.add_argument("--initial-policy-observation", required=True)
    parser.add_argument("--target-focus-report")
    parser.add_argument("--trajectory-out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--frame-count", type=int, default=101)
    parser.add_argument("--pull-frame-count", type=int, default=0)
    parser.add_argument("--door-open-angle-rad", type=float, default=0.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    trajectory_out = Path(args.trajectory_out).expanduser().resolve()
    report_out = Path(args.report_out).expanduser().resolve()
    trajectory_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        trajectory, report = solve_reach_seed(
            model_path=args.model,
            standing_initialization_path=args.standing_initialization,
            initial_policy_observation_path=args.initial_policy_observation,
            target_focus_report_path=args.target_focus_report,
            frame_count=args.frame_count,
            pull_frame_count=args.pull_frame_count,
            door_open_angle_rad=args.door_open_angle_rad,
        )
        np.save(trajectory_out, trajectory, allow_pickle=False)
        report["trajectory"]["path"] = str(trajectory_out)
        report["trajectory"]["file_sha256"] = _sha256(trajectory_out)
        exit_code = 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": [f"reach_seed_failed:{type(exc).__name__}:{exc}"],
        }
        exit_code = 1
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
