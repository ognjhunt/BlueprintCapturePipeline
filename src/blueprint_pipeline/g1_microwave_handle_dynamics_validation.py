"""Validate contact-driven microwave articulation from a G1 grasp seed.

The validator deliberately removes two common false-positive paths: the door
panel cannot collide and the door body is gravity compensated.  The handle is
assigned an isolated collision category that only right-hand geoms accept.
The robot is replayed as a kinematic driver while the door hinge remains a
free simulated degree of freedom.  Consequently, any hinge motion must be
transmitted through right-hand-to-handle contact.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .g1_microwave_grasp_arc_seed import SCHEMA_VERSION as GRASP_SCHEMA_VERSION
from .g1_microwave_grasp_arc_seed import _proxy_model
from .g1_microwave_reach_seed import (
    PINNED_G1_MODEL_SHA256,
    _finite_vector,
    _load_mapping,
    _rotation_wxyz,
    _sha256,
)
from .g1_sonic_motion_token_conversion import SOURCE_ACTION_JOINT_NAMES
from .gear_sonic_joint_order_contract import PROTOCOL_V4_FULL_JOINT_ORDER


SCHEMA_VERSION = "g1_microwave_handle_only_dynamic_validation.v2"


def _positive_handle_contact_proven(
    *,
    contact_step_count: int,
    positive_force_contact_count: int,
    peak_normal_force_n: float,
) -> bool:
    return bool(
        contact_step_count > 0
        and positive_force_contact_count > 0
        and math.isfinite(peak_normal_force_n)
        and peak_normal_force_n > 0.0
    )


def _right_hand_geom_ids(mujoco: Any, model: Any) -> set[int]:
    ids: set[int] = set()
    for geom_id in range(model.ngeom):
        body_name = str(
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[geom_id]),
            )
            or ""
        )
        if body_name.startswith("right_hand_"):
            ids.add(geom_id)
    if not ids:
        raise ValueError("g1_microwave_handle_dynamics_right_hand_geoms_missing")
    return ids


def validate_handle_only_dynamics(
    *,
    model_path: str | Path,
    standing_initialization_path: str | Path,
    initial_policy_observation_path: str | Path,
    target_focus_report_path: str | Path,
    grasp_report_path: str | Path,
    trajectory_path: str | Path,
    closure_complete_frame_index: int = 125,
    simulation_substeps_per_frame: int = 20,
    simulation_timestep_seconds: float = 0.001,
    opening_threshold_rad: float = 0.01,
    door_angle_feedback_lead_rad: float = 0.02,
    requested_opening_tolerance_rad: float = 0.02,
    maximum_simulation_seconds: float = 8.0,
    no_progress_timeout_seconds: float = 2.0,
    meaningful_progress_delta_rad: float = 0.0001,
    expected_model_sha256: str = PINNED_G1_MODEL_SHA256,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run a hinge-feedback grasp pull and return its dynamics report/trace."""

    try:
        import mujoco  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("g1_microwave_handle_dynamics_mujoco_missing") from exc

    model_file = Path(model_path).expanduser().resolve()
    standing_file = Path(standing_initialization_path).expanduser().resolve()
    observation_file = Path(initial_policy_observation_path).expanduser().resolve()
    focus_file = Path(target_focus_report_path).expanduser().resolve()
    grasp_file = Path(grasp_report_path).expanduser().resolve()
    trajectory_file = Path(trajectory_path).expanduser().resolve()
    if not model_file.is_file() or _sha256(model_file) != str(
        expected_model_sha256
    ).lower():
        raise ValueError("g1_microwave_handle_dynamics_model_sha256_mismatch")
    if int(simulation_substeps_per_frame) < 1:
        raise ValueError("g1_microwave_handle_dynamics_substeps_invalid")
    timestep = float(simulation_timestep_seconds)
    threshold = float(opening_threshold_rad)
    feedback_lead = float(door_angle_feedback_lead_rad)
    completion_tolerance = float(requested_opening_tolerance_rad)
    maximum_duration = float(maximum_simulation_seconds)
    no_progress_timeout = float(no_progress_timeout_seconds)
    meaningful_progress = float(meaningful_progress_delta_rad)
    if not math.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("g1_microwave_handle_dynamics_timestep_invalid")
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("g1_microwave_handle_dynamics_threshold_invalid")
    if not math.isfinite(feedback_lead) or feedback_lead <= 0.0:
        raise ValueError("g1_microwave_handle_dynamics_feedback_lead_invalid")
    if not 0.0 <= completion_tolerance < threshold + 0.35:
        raise ValueError("g1_microwave_handle_dynamics_tolerance_invalid")
    if not math.isfinite(maximum_duration) or maximum_duration <= 0.0:
        raise ValueError("g1_microwave_handle_dynamics_maximum_duration_invalid")
    if not math.isfinite(no_progress_timeout) or no_progress_timeout <= 0.0:
        raise ValueError("g1_microwave_handle_dynamics_progress_timeout_invalid")
    if not math.isfinite(meaningful_progress) or meaningful_progress <= 0.0:
        raise ValueError("g1_microwave_handle_dynamics_progress_delta_invalid")

    standing = _load_mapping(standing_file, name="standing_initialization")
    observation = _load_mapping(observation_file, name="initial_policy_observation")
    focus = _load_mapping(focus_file, name="target_focus_report")
    grasp = _load_mapping(grasp_file, name="grasp_arc_seed_report")
    if grasp.get("schema_version") != GRASP_SCHEMA_VERSION:
        raise ValueError("g1_microwave_handle_dynamics_grasp_schema_mismatch")
    grasp_trajectory = grasp.get("trajectory")
    if not isinstance(grasp_trajectory, Mapping) or grasp_trajectory.get(
        "file_sha256"
    ) != _sha256(trajectory_file):
        raise ValueError("g1_microwave_handle_dynamics_trajectory_binding_failed")
    trajectory = np.load(trajectory_file, allow_pickle=False)
    if trajectory.ndim != 2 or trajectory.shape[1] != len(
        SOURCE_ACTION_JOINT_NAMES
    ):
        raise ValueError("g1_microwave_handle_dynamics_trajectory_shape_invalid")
    start = int(closure_complete_frame_index)
    if start < 0 or start >= trajectory.shape[0] - 1:
        raise ValueError("g1_microwave_handle_dynamics_start_frame_invalid")
    pose_solver = grasp.get("pose_solver")
    pose_frames = pose_solver.get("frames") if isinstance(pose_solver, Mapping) else None
    if not isinstance(pose_frames, list):
        raise ValueError("g1_microwave_handle_dynamics_pose_frames_missing")
    try:
        prescribed_openings = -np.asarray(
            [float(row["door_angle_rad"]) for row in pose_frames],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "g1_microwave_handle_dynamics_pose_angles_invalid"
        ) from exc
    pull_trajectory = trajectory[start:].astype(np.float64)
    if (
        prescribed_openings.shape != (pull_trajectory.shape[0],)
        or prescribed_openings[0] != 0.0
        or not np.all(np.diff(prescribed_openings) > 0.0)
    ):
        raise ValueError("g1_microwave_handle_dynamics_pull_binding_invalid")
    requested_opening = float(prescribed_openings[-1])
    if completion_tolerance >= requested_opening:
        raise ValueError("g1_microwave_handle_dynamics_tolerance_too_large")

    context = observation.get("camera_projection_context")
    if not isinstance(context, Mapping):
        raise ValueError("g1_microwave_handle_dynamics_camera_context_missing")
    pelvis = context.get("live_isaac_pelvis_world_pose")
    if not isinstance(pelvis, Mapping):
        raise ValueError("g1_microwave_handle_dynamics_pelvis_pose_missing")
    model, _ = _proxy_model(
        mujoco=mujoco,
        model_path=model_file,
        pelvis_world=_finite_vector(
            pelvis.get("position_xyz"), size=3, name="live_pelvis_world"
        ),
        pelvis_rotation=_rotation_wxyz(pelvis.get("quaternion_wxyz")),
        handle_world=_finite_vector(
            focus.get("target_world_xyz_m"), size=3, name="handle_world"
        ),
        hinge_world=_finite_vector(
            focus.get("hinge_world_xyz_m"), size=3, name="hinge_world"
        ),
        hinge_axis_world=_finite_vector(
            focus.get("joint_world_axis_xyz"), size=3, name="hinge_axis_world"
        ),
        focus=focus,
    )
    model.opt.timestep = timestep
    panel_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "microwave_door_panel"
    )
    handle_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "microwave_handle"
    )
    door_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "microwave_door_hinge"
    )
    if min(panel_geom, handle_geom, door_joint) < 0:
        raise ValueError("g1_microwave_handle_dynamics_proxy_component_missing")
    hand_geom_ids = _right_hand_geom_ids(mujoco, model)

    # Disable the full panel and isolate the handle to a collision bit accepted
    # only by the right hand.  This excludes the floor, wrist, forearm, torso,
    # and all other robot bodies as possible sources of door torque.
    collision_bit = 1 << 20
    model.geom_contype[panel_geom] = 0
    model.geom_conaffinity[panel_geom] = 0
    model.geom_contype[handle_geom] = collision_bit
    model.geom_conaffinity[handle_geom] = 0
    for geom_id in hand_geom_ids:
        model.geom_conaffinity[geom_id] = int(
            model.geom_conaffinity[geom_id]
        ) | collision_bit
    door_body = int(model.jnt_bodyid[door_joint])
    model.body_gravcomp[door_body] = 1.0

    data = mujoco.MjData(model)
    measured = _finite_vector(
        standing.get("measured_full_joint_positions"),
        size=len(PROTOCOL_V4_FULL_JOINT_ORDER),
        name="standing_joint_positions",
    )
    for name, value in zip(PROTOCOL_V4_FULL_JOINT_ORDER, measured):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)

    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    for name in SOURCE_ACTION_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
    door_qpos_address = int(model.jnt_qposadr[door_joint])
    door_dof_address = int(model.jnt_dofadr[door_joint])
    for address, value in zip(qpos_addresses, trajectory[start]):
        data.qpos[address] = float(value)
    data.qpos[door_qpos_address] = 0.0
    data.qvel[:] = 0.0
    root_qpos = data.qpos[:7].copy()
    mujoco.mj_forward(model, data)

    # Columns: simulation_time_s, door_angle_rad, door_velocity_rad_s,
    # right_hand_handle_contact (0/1), summed_normal_force_n.
    trace_rows: list[list[float]] = [
        [0.0, float(data.qpos[door_qpos_address]), 0.0, 0.0, 0.0]
    ]
    contact_step_count = 0
    positive_force_contact_count = 0
    peak_normal_force_n = 0.0
    elapsed = 0.0
    previous = pull_trajectory[0].copy()
    substeps = int(simulation_substeps_per_frame)
    control_period = substeps * timestep
    maximum_control_frames = max(1, int(math.ceil(maximum_duration / control_period)))
    no_progress_control_frames = max(
        1, int(math.ceil(no_progress_timeout / control_period))
    )
    best_opening = 0.0
    last_meaningful_progress_frame = 0
    termination_reason = "maximum_duration_reached"
    executed_control_frames = 0
    for control_frame in range(maximum_control_frames):
        measured_opening = max(0.0, -float(data.qpos[door_qpos_address]))
        desired_opening = min(requested_opening, measured_opening + feedback_lead)
        target = np.asarray(
            [
                np.interp(desired_opening, prescribed_openings, pull_trajectory[:, index])
                for index in range(pull_trajectory.shape[1])
            ],
            dtype=np.float64,
        )
        velocity = (target - previous) / (substeps * timestep)
        for substep in range(substeps):
            alpha = float(substep + 1) / float(substeps)
            command = previous + (target - previous) * alpha
            door_angle = float(data.qpos[door_qpos_address])
            door_velocity = float(data.qvel[door_dof_address])
            data.qpos[:7] = root_qpos
            data.qvel[:6] = 0.0
            for address, value in zip(qpos_addresses, command):
                data.qpos[address] = float(value)
            for address, value in zip(dof_addresses, velocity):
                data.qvel[address] = float(value)
            data.qpos[door_qpos_address] = door_angle
            data.qvel[door_dof_address] = door_velocity
            mujoco.mj_step(model, data)
            normal_force = 0.0
            contact_found = False
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                pair = {int(contact.geom1), int(contact.geom2)}
                if handle_geom not in pair or not pair.intersection(hand_geom_ids):
                    continue
                contact_found = True
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, contact_index, force)
                normal_force += max(0.0, float(force[0]))
                positive_force_contact_count += int(force[0] > 0.0)
            contact_step_count += int(contact_found)
            peak_normal_force_n = max(peak_normal_force_n, normal_force)
            elapsed += timestep
            trace_rows.append(
                [
                    elapsed,
                    float(data.qpos[door_qpos_address]),
                    float(data.qvel[door_dof_address]),
                    float(contact_found),
                    normal_force,
                ]
            )
        previous = target
        executed_control_frames = control_frame + 1
        observed_opening = max(0.0, -float(data.qpos[door_qpos_address]))
        if observed_opening >= best_opening + meaningful_progress:
            best_opening = observed_opening
            last_meaningful_progress_frame = executed_control_frames
        else:
            best_opening = max(best_opening, observed_opening)
        if best_opening >= requested_opening - completion_tolerance:
            termination_reason = "requested_opening_within_tolerance_reached"
            break
        if (
            executed_control_frames - last_meaningful_progress_frame
            >= no_progress_control_frames
        ):
            termination_reason = "no_meaningful_progress_timeout"
            break

    trace = np.asarray(trace_rows, dtype=np.float64)
    initial_angle = float(trace[0, 1])
    minimum_angle = float(np.min(trace[:, 1]))
    final_angle = float(trace[-1, 1])
    opening_delta = initial_angle - minimum_angle
    positive_handle_contact_proven = _positive_handle_contact_proven(
        contact_step_count=contact_step_count,
        positive_force_contact_count=positive_force_contact_count,
        peak_normal_force_n=peak_normal_force_n,
    )
    proven = bool(
        opening_delta >= threshold and positive_handle_contact_proven
    )
    requested_transition_proven = bool(
        opening_delta >= requested_opening - completion_tolerance
        and positive_handle_contact_proven
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "qualified_contact_driven_handle_only_requested_transition"
            if requested_transition_proven
            else (
                "qualified_contact_driven_handle_only_partial_articulation"
                if proven
                else "blocked_no_contact_driven_articulation"
            )
        ),
        "model": {"path": str(model_file), "sha256": _sha256(model_file)},
        "inputs": {
            "standing_initialization": {
                "path": str(standing_file),
                "sha256": _sha256(standing_file),
            },
            "initial_policy_observation": {
                "path": str(observation_file),
                "sha256": _sha256(observation_file),
            },
            "target_focus_report": {
                "path": str(focus_file),
                "sha256": _sha256(focus_file),
            },
            "grasp_report": {
                "path": str(grasp_file),
                "sha256": _sha256(grasp_file),
            },
            "trajectory": {
                "path": str(trajectory_file),
                "sha256": _sha256(trajectory_file),
            },
        },
        "simulation_contract": {
            "door_started_fully_closed": initial_angle == 0.0,
            "door_panel_collision_enabled": False,
            "door_gravity_compensation": 1.0,
            "handle_collision_acceptor": "right_hand_geoms_only",
            "handle_collision_category_bit": collision_bit,
            "robot_driver": (
                "door_angle_feedback_kinematic_joint_replay_with_pinned_floating_root"
            ),
            "door_hinge_driver": "unactuated_free_simulation_dof",
            "closure_complete_frame_index": start,
            "prescribed_pull_frame_count": int(pull_trajectory.shape[0]),
            "door_angle_feedback_lead_rad": feedback_lead,
            "requested_opening_rad": requested_opening,
            "requested_opening_tolerance_rad": completion_tolerance,
            "maximum_simulation_seconds": maximum_duration,
            "no_progress_timeout_seconds": no_progress_timeout,
            "meaningful_progress_delta_rad": meaningful_progress,
            "executed_control_frame_count": executed_control_frames,
            "termination_reason": termination_reason,
            "substeps_per_frame": substeps,
            "timestep_seconds": timestep,
            "simulated_duration_seconds": elapsed,
        },
        "observations": {
            "initial_door_angle_rad": initial_angle,
            "final_door_angle_rad": final_angle,
            "minimum_door_angle_rad": minimum_angle,
            "maximum_door_angle_rad": float(np.max(trace[:, 1])),
            "maximum_contact_driven_opening_delta_rad": opening_delta,
            "peak_abs_door_velocity_rad_s": float(np.max(np.abs(trace[:, 2]))),
            "right_hand_handle_contact_step_count": contact_step_count,
            "positive_normal_force_contact_count": positive_force_contact_count,
            "peak_summed_contact_normal_force_n": peak_normal_force_n,
        },
        "qualification": {
            "opening_threshold_rad": threshold,
            "right_hand_handle_positive_force_contact_proven": (
                positive_handle_contact_proven
            ),
            "contact_driven_door_articulation_proven": proven,
            "requested_opening_within_tolerance_proven": (
                requested_transition_proven
            ),
            "exact_requested_opening_angle_reached": bool(
                opening_delta >= requested_opening
            ),
            "semantic_task_success_proven": False,
        },
        "blockers": [
            *(
                []
                if positive_handle_contact_proven
                else ["right_hand_handle_positive_force_contact_not_proven"]
            ),
            *(
                []
                if requested_transition_proven
                else ["requested_handle_only_opening_transition_not_proven"]
            ),
            "isaac_usd_dynamic_episode_qualification_not_proven",
            "trained_microwave_checkpoint_not_produced",
            "semantic_episode_success_not_proven",
        ],
        "claim_boundary": (
            "This proves the requested hinge transition within the declared "
            "tolerance using only right-hand-to-handle contact in the MuJoCo "
            "proxy. It does not prove an Isaac USD result, checkpoint competence, "
            "or semantic episode success."
        ),
    }
    json.dumps(report, sort_keys=True)
    return report, trace


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate handle-only contact-driven microwave articulation."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--standing-initialization", required=True)
    parser.add_argument("--initial-policy-observation", required=True)
    parser.add_argument("--target-focus-report", required=True)
    parser.add_argument("--grasp-report", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--trace-out", required=True)
    parser.add_argument("--opening-threshold-rad", type=float, default=0.01)
    parser.add_argument("--door-angle-feedback-lead-rad", type=float, default=0.02)
    parser.add_argument(
        "--requested-opening-tolerance-rad", type=float, default=0.02
    )
    parser.add_argument("--maximum-simulation-seconds", type=float, default=8.0)
    parser.add_argument("--no-progress-timeout-seconds", type=float, default=2.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report_out = Path(args.report_out).expanduser().resolve()
    trace_out = Path(args.trace_out).expanduser().resolve()
    report_out.parent.mkdir(parents=True, exist_ok=True)
    trace_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        report, trace = validate_handle_only_dynamics(
            model_path=args.model,
            standing_initialization_path=args.standing_initialization,
            initial_policy_observation_path=args.initial_policy_observation,
            target_focus_report_path=args.target_focus_report,
            grasp_report_path=args.grasp_report,
            trajectory_path=args.trajectory,
            opening_threshold_rad=args.opening_threshold_rad,
            door_angle_feedback_lead_rad=args.door_angle_feedback_lead_rad,
            requested_opening_tolerance_rad=args.requested_opening_tolerance_rad,
            maximum_simulation_seconds=args.maximum_simulation_seconds,
            no_progress_timeout_seconds=args.no_progress_timeout_seconds,
        )
        np.save(trace_out, trace, allow_pickle=False)
        report["trace"] = {
            "path": str(trace_out),
            "sha256": _sha256(trace_out),
            "columns": [
                "simulation_time_s",
                "door_angle_rad",
                "door_velocity_rad_s",
                "right_hand_handle_contact",
                "summed_normal_force_n",
            ],
            "row_count": int(trace.shape[0]),
        }
        exit_code = int(
            not report["qualification"][
                "requested_opening_within_tolerance_proven"
            ]
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": [
                f"handle_dynamics_validation_failed:{type(exc).__name__}:{exc}"
            ],
        }
        exit_code = 1
    report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
