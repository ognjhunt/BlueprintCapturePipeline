"""Rebuild the microwave fine-tune episode from the live Isaac start state.

The original owned demonstration was rendered in a lightweight MuJoCo proxy and
started from the generic GEAR-SONIC standing pose.  The qualification runtime
starts from a task-facing manipulation-ready pose in the textured Isaac kitchen.
Training directly on the proxy episode therefore leaves both the visual
observation and the first controller horizon out of distribution.

This module is embedded into the retained-worker fine-tune component.  It:

1. solves a new owned grasp trajectory from the exact live Isaac proprioception;
2. encodes that trajectory with the pinned official GEAR-SONIC encoder;
3. renders the trajectory from the same rigid head camera in the exact kitchen;
4. replaces the one-episode LeRobot rows and recomputes their numeric statistics.

Its render phase proves only controller-target physics replay, measured
articulation, camera, and contact-report facts for this owned trajectory. It
does not claim learned-policy execution or semantic task success; those remain
later fail-closed qualification gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "g1_microwave_live_aligned_finetune.v1"
TASK_DESCRIPTION = "Stand at the microwave and open the microwave door."
FRAME_COUNT = 176
FPS = 50
TARGET_PRIM_PATH = "/root/Microwave017/Microwave017_Door"
# The same-session manipulation-ready pose uses the palm-down grasp convention
# seen by the rigid head camera.  The older proxy demonstration used the
# opposite wrist axis plus a 45-degree yaw; carrying that transform into this
# pose leaves the first handle target 49 mm out of reach.  These values were
# qualified against the exact live G1 model and all 51 pull/contact frames.
LIVE_ALIGNED_HAND_AXIS_POLARITY = -1.0
LIVE_ALIGNED_GRASP_YAW_RAD = 0.0
ISAAC_RENDER_JOINT_TRACKING_MAX_ERROR_RAD = 0.5
ISAAC_RENDER_ACTIVE_JOINT_MEAN_TRACKING_MAX_ERROR_RAD = 0.2
ISAAC_RENDER_REQUIRED_ACTIVE_JOINT_SPAN_RAD = 0.25
ISAAC_RENDER_REQUIRED_ACTIVE_LINK_WORLD_MOTION_M = 0.01
ISAAC_RENDER_REQUIRED_ACTIVE_LINK_PIXEL_MOTION_PX = 4.0
ISAAC_RENDER_MAX_BASE_DISPLACEMENT_M = 0.2
ISAAC_RENDER_MAX_BASE_HEIGHT_DROP_M = 0.15
ISAAC_RENDER_MAX_UPRIGHT_TILT_DEG = 30.0
ISAAC_RENDER_MEANINGFUL_DOOR_MOTION_RAD = 0.01
ISAAC_RENDER_ALLOWED_SUPPORT_ROBOT_TERMS = ("foot", "ankle", "toe")
ISAAC_RENDER_ALLOWED_SUPPORT_SCENE_TERMS = ("floor", "ground", "room", "kitchen")
ISAAC_RENDER_ALLOWED_TARGET_CONTACT_ROBOT_TERMS = (
    "right_wrist",
    "right_hand",
    "right_palm",
    "right_thumb",
    "right_index",
    "right_middle",
)
ISAAC_RENDER_ACTIVE_ARM_LINK_NAMES = (
    "right_elbow_link",
    "right_wrist_yaw_link",
)
ISAAC_RENDER_ACTIVE_JOINT_PREFIXES = ("waist_", "right_shoulder_", "right_elbow_")
ISAAC_RENDER_ACTIVE_JOINT_NAMES = {
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label}_missing_or_unsafe")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label}_not_object")
    return value


def _canonical_joint_positions(initial_state: dict[str, Any]) -> list[float]:
    from blueprint_pipeline.gear_sonic_joint_order_contract import (
        PROTOCOL_V4_FULL_JOINT_ORDER,
    )

    mapping = dict(initial_state.get("proprioception_mapping") or {})
    inventory = list(mapping.get("observed_dof_inventory") or [])
    positions = {
        str(row.get("normalized_name") or row.get("observed_name") or ""): float(
            row["position"]
        )
        for row in inventory
        if isinstance(row, dict)
    }
    if set(positions) != set(PROTOCOL_V4_FULL_JOINT_ORDER):
        raise ValueError("live_aligned_initial_joint_inventory_mismatch")
    result = [positions[name] for name in PROTOCOL_V4_FULL_JOINT_ORDER]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("live_aligned_initial_joint_positions_nonfinite")
    return result


def prepare_actions(
    *,
    initial_state_path: str | Path,
    standing_report_path: str | Path,
    initial_observation_path: str | Path,
    robot_model_path: str | Path,
    encoder_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Solve and encode one trajectory from the exact live start state."""

    from blueprint_pipeline.g1_microwave_grasp_arc_seed import solve_grasp_arc_seed
    from blueprint_pipeline.g1_sonic_motion_token_conversion import (
        FIXED_UPRIGHT_PROJECTED_GRAVITY,
        SOURCE_ACTION_JOINT_NAMES,
        convert_to_sonic_actions,
        fixed_upright_root_anchor_rotations,
        load_onnx_encoder,
    )

    initial_state_file = Path(initial_state_path).expanduser().resolve()
    standing_file = Path(standing_report_path).expanduser().resolve()
    observation_file = Path(initial_observation_path).expanduser().resolve()
    model_file = Path(robot_model_path).expanduser().resolve()
    encoder_file = Path(encoder_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    initial_state = _load_object(initial_state_file, label="live_aligned_initial_state")
    standing = _load_object(standing_file, label="live_aligned_standing_report")
    observation = _load_object(
        observation_file, label="live_aligned_initial_observation"
    )
    live_positions = _canonical_joint_positions(initial_state)
    aligned_standing = dict(standing)
    aligned_standing.update(
        {
            "measured_full_joint_positions": live_positions,
            "source": "same_session_live_isaac_manipulation_ready_proprioception",
            "claim_boundary": (
                "This derived seed binds the owned IK start to the same-session "
                "live Isaac manipulation-ready state. It is not controller or "
                "task-success proof."
            ),
        }
    )
    camera_context = dict(observation.get("camera_projection_context") or {})
    camera_contract = dict(camera_context.get("camera_contract") or {})
    focus = dict(camera_contract.get("task_target_focus") or {})
    if (
        focus.get("target_prim_path") != observation.get("target_prim_path")
        or focus.get("target_prim_path") != TARGET_PRIM_PATH
    ):
        raise ValueError("live_aligned_task_focus_binding_invalid")

    destination.mkdir(parents=True, exist_ok=True)
    aligned_standing_path = destination / "aligned_standing_initialization.json"
    focus_path = destination / "live_task_focus_report.json"
    aligned_standing_path.write_text(
        json.dumps(aligned_standing, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    focus_path.write_text(
        json.dumps(focus, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trajectory, grasp_report = solve_grasp_arc_seed(
        model_path=model_file,
        standing_initialization_path=aligned_standing_path,
        initial_policy_observation_path=observation_file,
        target_focus_report_path=focus_path,
        reach_frame_count=101,
        closure_frame_count=26,
        pull_frame_count=51,
        door_open_angle_rad=0.35,
        hand_axis_polarity=LIVE_ALIGNED_HAND_AXIS_POLARITY,
        grasp_yaw_rad=LIVE_ALIGNED_GRASP_YAW_RAD,
    )
    if trajectory.shape != (FRAME_COUNT, len(SOURCE_ACTION_JOINT_NAMES)):
        raise RuntimeError("live_aligned_trajectory_shape_invalid")
    anchors = fixed_upright_root_anchor_rotations(
        FRAME_COUNT,
        fixed_base_upright_attested=True,
    )
    encoder = load_onnx_encoder(encoder_file)
    sonic_actions, conversion_report = convert_to_sonic_actions(
        trajectory,
        action_joint_names=SOURCE_ACTION_JOINT_NAMES,
        root_anchor_rotations_6d=anchors,
        encoder=encoder,
        source_provenance={
            "source_type": "same_session_live_aligned_owned_isaac_seed",
            "initial_state_sha256": _sha256(initial_state_file),
            "initial_observation_sha256": _sha256(observation_file),
            "robot_model_sha256": _sha256(model_file),
        },
        fps=float(FPS),
    )
    paths = {
        "observation_state": destination / "observation_state_43d.npy",
        "projected_gravity": destination / "observation_projected_gravity.npy",
        "motion_token": destination / "action_motion_token_64d.npy",
        "left_hand": destination / "teleop_left_hand_joints_7d.npy",
        "right_hand": destination / "teleop_right_hand_joints_7d.npy",
    }
    np.save(paths["observation_state"], trajectory.astype(np.float32), allow_pickle=False)
    np.save(
        paths["projected_gravity"],
        np.tile(
            np.asarray(
                initial_state.get(
                    "projected_gravity", FIXED_UPRIGHT_PROJECTED_GRAVITY
                ),
                dtype=np.float32,
            ),
            (FRAME_COUNT, 1),
        ),
        allow_pickle=False,
    )
    np.save(paths["motion_token"], sonic_actions[:, :64], allow_pickle=False)
    np.save(paths["left_hand"], sonic_actions[:, 64:71], allow_pickle=False)
    np.save(paths["right_hand"], sonic_actions[:, 71:78], allow_pickle=False)
    grasp_path = destination / "live_aligned_grasp_report.json"
    conversion_path = destination / "live_aligned_sonic_conversion_report.json"
    grasp_path.write_text(
        json.dumps(grasp_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    conversion_path.write_text(
        json.dumps(conversion_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "live_start_aligned_actions_prepared",
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "task_target_prim_path": TARGET_PRIM_PATH,
        "initial_state_sha256": _sha256(initial_state_file),
        "initial_observation_sha256": _sha256(observation_file),
        "robot_model_sha256": _sha256(model_file),
        "encoder_sha256": _sha256(encoder_file),
        "grasp_geometry": dict(grasp_report.get("geometry") or {}),
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "blockers": [],
        "claim_boundary": {
            "same_session_live_start_bound": True,
            "owned_kinematic_reach_only": True,
            "controller_execution_not_proven": True,
            "isaac_contact_not_proven": True,
            "semantic_success_not_proven": True,
        },
    }
    report_path = destination / "live_aligned_action_preparation.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _load_runtime_backend_overlay() -> Any:
    """Load the hash-verified task backend materialized by the prior episode."""

    import blueprint_pipeline  # noqa: F401

    source = Path("/workspace/runtime_overlay/isaac_runtime_task_backend.py")
    if source.is_symlink() or not source.is_file():
        raise RuntimeError("live_aligned_isaac_backend_overlay_missing")
    name = "blueprint_pipeline.isaac_runtime_task_backend"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("live_aligned_isaac_backend_overlay_spec_missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _active_joint_motion(
    trajectory: np.ndarray, joint_names: Sequence[str]
) -> dict[str, Any]:
    values = np.asarray(trajectory, dtype=np.float64)
    spans = np.ptp(values, axis=0)
    velocities = np.diff(values, axis=0) * float(FPS)
    accelerations = np.diff(velocities, axis=0) * float(FPS)
    active_indices = [
        index
        for index, name in enumerate(joint_names)
        if str(name).startswith(ISAAC_RENDER_ACTIVE_JOINT_PREFIXES)
        or str(name) in ISAAC_RENDER_ACTIVE_JOINT_NAMES
    ]
    active = {
        str(name): float(spans[index])
        for index, name in enumerate(joint_names)
        if index in active_indices
    }
    maximum = max(active.values(), default=0.0)
    if maximum < ISAAC_RENDER_REQUIRED_ACTIVE_JOINT_SPAN_RAD:
        raise RuntimeError("live_aligned_isaac_planned_active_arm_motion_too_low")
    return {
        "active_joint_names": [str(joint_names[index]) for index in active_indices],
        "joint_span_rad": active,
        "maximum_active_joint_span_rad": maximum,
        "maximum_active_joint_velocity_rad_s": float(
            np.max(np.abs(velocities[:, active_indices]), initial=0.0)
        ),
        "maximum_active_joint_acceleration_rad_s2": float(
            np.max(np.abs(accelerations[:, active_indices]), initial=0.0)
        ),
        "minimum_required_active_joint_span_rad": (
            ISAAC_RENDER_REQUIRED_ACTIVE_JOINT_SPAN_RAD
        ),
        "passed": True,
    }


def _robot_joint_readback(robot: Any, indices: np.ndarray) -> np.ndarray:
    getter = getattr(robot, "get_joint_positions", None)
    if not callable(getter):
        raise RuntimeError("live_aligned_isaac_joint_readback_api_missing")
    values = np.asarray(getter(), dtype=np.float64).reshape(-1)
    if (
        values.size == 0
        or not np.isfinite(values).all()
        or int(np.max(indices, initial=-1)) >= values.size
    ):
        raise RuntimeError("live_aligned_isaac_joint_readback_invalid")
    return values[indices]


def _robot_joint_velocity_readback(robot: Any, indices: np.ndarray) -> np.ndarray:
    getter = getattr(robot, "get_joint_velocities", None)
    if not callable(getter):
        raise RuntimeError("live_aligned_isaac_joint_velocity_readback_api_missing")
    values = np.asarray(getter(), dtype=np.float64).reshape(-1)
    if (
        values.size == 0
        or not np.isfinite(values).all()
        or int(np.max(indices, initial=-1)) >= values.size
    ):
        raise RuntimeError("live_aligned_isaac_joint_velocity_readback_invalid")
    return values[indices]


def _project_active_arm_landmarks(
    *,
    camera_contract: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    from blueprint_pipeline.isaac_task_review_renderer import project_world_point

    landmarks = {
        str(row.get("landmark_id") or ""): dict(row)
        for row in list(registration.get("landmarks") or [])
        if isinstance(row, Mapping)
    }
    if set(ISAAC_RENDER_ACTIVE_ARM_LINK_NAMES) - set(landmarks):
        raise RuntimeError("live_aligned_isaac_active_arm_landmarks_missing")
    result: dict[str, dict[str, Any]] = {}
    for name in ISAAC_RENDER_ACTIVE_ARM_LINK_NAMES:
        world = list(landmarks[name].get("world_position_xyz") or [])
        projection = project_world_point(camera_contract, world)
        if (
            len(world) != 3
            or not all(math.isfinite(float(value)) for value in world)
            or projection.get("in_frame") is not True
        ):
            raise RuntimeError(f"live_aligned_isaac_active_arm_not_in_frame:{name}")
        result[name] = {
            "world_position_xyz_m": [float(value) for value in world],
            "u_px": float(projection["u_px"]),
            "v_px": float(projection["v_px"]),
            "in_frame": True,
        }
    return result


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(
        sum(
            (float(second[index]) - float(first[index])) ** 2
            for index in range(len(first))
        )
    )


def _path_within(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def _classify_robot_contact_events(
    events: Sequence[Mapping[str, Any]],
    *,
    robot_prim_path: str,
    target_prim_path: str,
) -> dict[str, Any]:
    """Separate allowed support/manipulator contacts from invalid collisions."""

    allowed_support: list[dict[str, Any]] = []
    target_manipulator: list[dict[str, Any]] = []
    unexpected: list[dict[str, Any]] = []
    active_events: list[dict[str, Any]] = []
    for raw_event in events:
        event = dict(raw_event)
        if int(event.get("contact_point_count") or 0) <= 0:
            continue
        active_events.append(event)
        collider_paths = [
            str(event.get("collider0_prim_path") or ""),
            str(event.get("collider1_prim_path") or ""),
        ]
        actor_paths = [
            str(event.get("actor0_prim_path") or ""),
            str(event.get("actor1_prim_path") or ""),
        ]
        robot_paths = [
            path for path in collider_paths if _path_within(path, robot_prim_path)
        ]
        other_paths = [
            path for path in collider_paths if not _path_within(path, robot_prim_path)
        ]
        if not robot_paths:
            robot_paths = [
                path for path in actor_paths if _path_within(path, robot_prim_path)
            ]
        if not other_paths:
            other_paths = [
                path for path in actor_paths if not _path_within(path, robot_prim_path)
            ]
        if not robot_paths:
            continue
        lowered_others = [path.lower() for path in other_paths]
        support_robot_contact = any(
            any(
                term in path.lower()
                for term in ISAAC_RENDER_ALLOWED_SUPPORT_ROBOT_TERMS
            )
            for path in robot_paths
        )
        support_scene_contact = bool(other_paths) and all(
            any(
                term in path
                for term in ISAAC_RENDER_ALLOWED_SUPPORT_SCENE_TERMS
            )
            for path in lowered_others
        )
        if support_robot_contact and support_scene_contact:
            allowed_support.append(event)
            continue
        touches_target = any(
            _path_within(path, target_prim_path)
            or _path_within(target_prim_path, path)
            for path in other_paths
        )
        manipulator_contact = any(
            any(term in path.lower() for term in ISAAC_RENDER_ALLOWED_TARGET_CONTACT_ROBOT_TERMS)
            for path in robot_paths
        )
        if touches_target and manipulator_contact:
            target_manipulator.append(event)
        else:
            unexpected.append(event)
    return {
        "active_contact_event_count": len(active_events),
        "allowed_support_contact_events": allowed_support,
        "target_manipulator_contact_events": target_manipulator,
        "unexpected_robot_collision_events": unexpected,
    }


def _summarize_render_motion(
    *,
    records: Sequence[Mapping[str, Any]],
    planned_motion: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if len(records) != FRAME_COUNT:
        blockers.append("live_aligned_isaac_motion_record_horizon_invalid")
    max_tracking_error = max(
        (float(row.get("target_joint_max_error_rad") or 0.0) for row in records),
        default=math.inf,
    )
    max_active_mean_tracking_error = max(
        (
            float(row.get("active_joint_mean_tracking_error_rad") or 0.0)
            for row in records
        ),
        default=math.inf,
    )
    if (
        not math.isfinite(max_tracking_error)
        or max_tracking_error > ISAAC_RENDER_JOINT_TRACKING_MAX_ERROR_RAD
    ):
        blockers.append("live_aligned_isaac_joint_target_tracking_error_too_high")
    if (
        not math.isfinite(max_active_mean_tracking_error)
        or max_active_mean_tracking_error
        > ISAAC_RENDER_ACTIVE_JOINT_MEAN_TRACKING_MAX_ERROR_RAD
    ):
        blockers.append("live_aligned_isaac_active_joint_tracking_error_too_high")
    physics_step_deltas = [
        int(row["physics_step_delta"])
        for row in records
        if row.get("physics_step_delta") is not None
    ]
    simulation_time_deltas = [
        float(row["simulation_time_delta_seconds"])
        for row in records
        if row.get("simulation_time_delta_seconds") is not None
    ]
    render_physics_step_deltas = [
        int(row["render_physics_step_delta"])
        for row in records
        if row.get("render_physics_step_delta") is not None
    ]
    if len(physics_step_deltas) != len(records) or any(
        value != 1 for value in physics_step_deltas
    ):
        blockers.append("live_aligned_isaac_controller_target_physics_step_invalid")
    if len(simulation_time_deltas) != len(records) or any(
        not math.isclose(value, 1.0 / float(FPS), rel_tol=0.0, abs_tol=1e-9)
        for value in simulation_time_deltas
    ):
        blockers.append("live_aligned_isaac_controller_target_simulation_time_invalid")
    if len(render_physics_step_deltas) != len(records) or any(
        value != 0 for value in render_physics_step_deltas
    ):
        blockers.append("live_aligned_isaac_render_added_hidden_physics_step")

    active_joint_names = [
        str(name) for name in planned_motion.get("active_joint_names") or []
    ]
    active_joint_spans: dict[str, float] = {}
    for name in active_joint_names:
        values = [
            float(dict(row.get("active_joint_positions") or {})[name])
            for row in records
            if name in dict(row.get("active_joint_positions") or {})
        ]
        active_joint_spans[name] = (
            max(values) - min(values) if len(values) == len(records) and values else 0.0
        )
    maximum_measured_active_joint_span = max(
        active_joint_spans.values(), default=0.0
    )
    if maximum_measured_active_joint_span < (
        ISAAC_RENDER_REQUIRED_ACTIVE_JOINT_SPAN_RAD * 0.5
    ):
        blockers.append("live_aligned_isaac_measured_active_joint_motion_too_low")

    active_velocity_rows = [
        [
            float(dict(row.get("active_joint_velocities_rad_s") or {})[name])
            for name in active_joint_names
        ]
        for row in records
        if all(
            name in dict(row.get("active_joint_velocities_rad_s") or {})
            for name in active_joint_names
        )
    ]
    active_velocities = np.asarray(active_velocity_rows, dtype=np.float64)
    maximum_measured_velocity = float(
        np.max(np.abs(active_velocities), initial=0.0)
    )
    measured_accelerations = (
        np.diff(active_velocities, axis=0) * float(FPS)
        if active_velocities.shape[0] >= 2
        else np.zeros((0, len(active_joint_names)), dtype=np.float64)
    )
    maximum_measured_acceleration = float(
        np.max(np.abs(measured_accelerations), initial=0.0)
    )
    planned_max_velocity = float(
        planned_motion.get("maximum_active_joint_velocity_rad_s") or 0.0
    )
    planned_max_acceleration = float(
        planned_motion.get("maximum_active_joint_acceleration_rad_s2") or 0.0
    )
    velocity_guardrail = max(4.0, planned_max_velocity * 3.0 + 1.0)
    acceleration_guardrail = max(50.0, planned_max_acceleration * 3.0 + 10.0)
    if active_velocities.shape[0] != len(records):
        blockers.append("live_aligned_isaac_active_joint_velocity_horizon_invalid")
    if maximum_measured_velocity > velocity_guardrail:
        blockers.append("live_aligned_isaac_active_joint_velocity_exploded")
    if maximum_measured_acceleration > acceleration_guardrail:
        blockers.append("live_aligned_isaac_active_joint_acceleration_exploded")

    base_positions = [
        [float(value) for value in row.get("base_position_xyz_m") or []]
        for row in records
    ]
    base_positions_valid = bool(base_positions) and all(
        len(value) == 3 and all(math.isfinite(item) for item in value)
        for value in base_positions
    )
    maximum_base_displacement = (
        max(_distance(base_positions[0], value) for value in base_positions)
        if base_positions_valid
        else math.inf
    )
    maximum_base_height_drop = (
        max(0.0, base_positions[0][2] - min(value[2] for value in base_positions))
        if base_positions_valid
        else math.inf
    )
    projected_gravity_rows = [
        [float(value) for value in row.get("projected_gravity") or []]
        for row in records
    ]
    gravity_valid = bool(projected_gravity_rows) and all(
        len(value) == 3 and all(math.isfinite(item) for item in value)
        for value in projected_gravity_rows
    )
    maximum_tilt_degrees = (
        max(
            math.degrees(
                math.atan2(
                    math.hypot(gravity[0], gravity[1]),
                    abs(gravity[2]),
                )
            )
            for gravity in projected_gravity_rows
        )
        if gravity_valid
        else math.inf
    )
    if maximum_base_displacement > ISAAC_RENDER_MAX_BASE_DISPLACEMENT_M:
        blockers.append("live_aligned_isaac_base_displacement_too_high")
    if maximum_base_height_drop > ISAAC_RENDER_MAX_BASE_HEIGHT_DROP_M:
        blockers.append("live_aligned_isaac_base_height_drop_too_high")
    if maximum_tilt_degrees > ISAAC_RENDER_MAX_UPRIGHT_TILT_DEG:
        blockers.append("live_aligned_isaac_upright_tilt_too_high")

    door_angles = [
        float(row.get("door_open_angle_rad"))
        for row in records
        if row.get("door_open_angle_rad") is not None
    ]
    door_angle_readback_valid = len(door_angles) == len(records) and all(
        math.isfinite(value) for value in door_angles
    )
    door_angle_span = (
        max(door_angles) - min(door_angles) if door_angle_readback_valid else math.inf
    )
    initial_door_angle = (
        float(records[0].get("door_open_angle_before_step_rad"))
        if records and records[0].get("door_open_angle_before_step_rad") is not None
        else math.nan
    )
    if not math.isfinite(initial_door_angle):
        door_angle_readback_valid = False
        door_angle_span = math.inf
    elif door_angle_readback_valid:
        door_angle_span = max(
            [initial_door_angle, *door_angles]
        ) - min([initial_door_angle, *door_angles])
    target_contact_frame_indices = [
        int(row["frame_index"])
        for row in records
        if bool(row.get("target_manipulator_contact"))
    ]
    unexpected_collision_frame_indices = [
        int(row["frame_index"])
        for row in records
        if list(row.get("unexpected_robot_collision_events") or [])
    ]
    contact_monitor_active = len(records) == FRAME_COUNT and all(
        row.get("contact_report_monitor_active") is True for row in records
    )
    if not contact_monitor_active:
        blockers.append("live_aligned_isaac_contact_report_monitor_inactive")
    if not door_angle_readback_valid:
        blockers.append("live_aligned_isaac_door_joint_readback_invalid")
    if unexpected_collision_frame_indices:
        blockers.append("live_aligned_isaac_unexpected_robot_collision")
    first_meaningful_door_motion_frame: int | None = None
    if door_angle_readback_valid and door_angles:
        first_meaningful_door_motion_frame = next(
            (
                index
                for index, value in enumerate(door_angles)
                if abs(value - initial_door_angle)
                >= ISAAC_RENDER_MEANINGFUL_DOOR_MOTION_RAD
            ),
            None,
        )
    door_motion_contact_gated = first_meaningful_door_motion_frame is None or any(
        index <= first_meaningful_door_motion_frame
        for index in target_contact_frame_indices
    )
    if not door_motion_contact_gated:
        blockers.append("live_aligned_isaac_door_motion_without_manipulator_contact")

    link_motion: dict[str, dict[str, Any]] = {}
    for name in ISAAC_RENDER_ACTIVE_ARM_LINK_NAMES:
        positions = [
            list(dict(row.get("active_arm_landmarks") or {}).get(name, {}).get(
                "world_position_xyz_m"
            ) or [])
            for row in records
        ]
        pixels = [
            [
                dict(row.get("active_arm_landmarks") or {}).get(name, {}).get("u_px"),
                dict(row.get("active_arm_landmarks") or {}).get(name, {}).get("v_px"),
            ]
            for row in records
        ]
        valid = bool(positions and all(len(value) == 3 for value in positions)) and bool(
            pixels
            and all(
                len(value) == 2
                and all(item is not None and math.isfinite(float(item)) for item in value)
                for value in pixels
            )
        )
        world_motion = (
            max(_distance(positions[0], value) for value in positions) if valid else 0.0
        )
        pixel_motion = (
            max(_distance(pixels[0], value) for value in pixels) if valid else 0.0
        )
        link_motion[name] = {
            "maximum_world_displacement_from_first_m": world_motion,
            "maximum_pixel_displacement_from_first_px": pixel_motion,
            "all_frames_in_robot_pov": valid,
        }
    maximum_world_motion = max(
        (
            float(row["maximum_world_displacement_from_first_m"])
            for row in link_motion.values()
        ),
        default=0.0,
    )
    maximum_pixel_motion = max(
        (
            float(row["maximum_pixel_displacement_from_first_px"])
            for row in link_motion.values()
        ),
        default=0.0,
    )
    if not all(row["all_frames_in_robot_pov"] is True for row in link_motion.values()):
        blockers.append("live_aligned_isaac_active_arm_visibility_incomplete")
    if maximum_world_motion < ISAAC_RENDER_REQUIRED_ACTIVE_LINK_WORLD_MOTION_M:
        blockers.append("live_aligned_isaac_active_arm_world_motion_too_low")
    if maximum_pixel_motion < ISAAC_RENDER_REQUIRED_ACTIVE_LINK_PIXEL_MOTION_PX:
        blockers.append("live_aligned_isaac_active_arm_pixel_motion_too_low")
    distinct_frame_count = len(
        {str(row.get("frame_sha256") or "") for row in records if row.get("frame_sha256")}
    )
    if distinct_frame_count < 2:
        blockers.append("live_aligned_isaac_rendered_frames_identical")
    return {
        "schema_version": "g1_microwave_live_aligned_isaac_motion_evidence.v2",
        "status": "passed" if not blockers else "blocked",
        "frame_count": len(records),
        "planned_motion": dict(planned_motion),
        "physics_execution": {
            "controller_target_api": "ArticulationAction",
            "maximum_joint_target_tracking_error_rad": max_tracking_error,
            "maximum_allowed_joint_target_tracking_error_rad": (
                ISAAC_RENDER_JOINT_TRACKING_MAX_ERROR_RAD
            ),
            "maximum_active_joint_mean_tracking_error_rad": (
                max_active_mean_tracking_error
            ),
            "maximum_allowed_active_joint_mean_tracking_error_rad": (
                ISAAC_RENDER_ACTIVE_JOINT_MEAN_TRACKING_MAX_ERROR_RAD
            ),
            "measured_active_joint_span_rad": active_joint_spans,
            "maximum_measured_active_joint_span_rad": (
                maximum_measured_active_joint_span
            ),
            "maximum_measured_active_joint_velocity_rad_s": (
                maximum_measured_velocity
            ),
            "velocity_explosion_guardrail_rad_s": velocity_guardrail,
            "maximum_measured_active_joint_acceleration_rad_s2": (
                maximum_measured_acceleration
            ),
            "acceleration_explosion_guardrail_rad_s2": acceleration_guardrail,
            "one_physics_step_per_controller_target": not any(
                "physics_step" in blocker or "simulation_time" in blocker
                for blocker in blockers
            ),
        },
        "render_synchronization": {
            "rendered_from_post_physics_measured_pose": True,
            "replicator_delta_time_seconds": 0.0,
            "replicator_pause_timeline": False,
            "hidden_render_physics_step_absent": not any(
                "hidden_physics_step" in blocker for blocker in blockers
            ),
        },
        "stability": {
            "maximum_base_displacement_m": maximum_base_displacement,
            "maximum_allowed_base_displacement_m": (
                ISAAC_RENDER_MAX_BASE_DISPLACEMENT_M
            ),
            "maximum_base_height_drop_m": maximum_base_height_drop,
            "maximum_allowed_base_height_drop_m": (
                ISAAC_RENDER_MAX_BASE_HEIGHT_DROP_M
            ),
            "maximum_upright_tilt_degrees": maximum_tilt_degrees,
            "maximum_allowed_upright_tilt_degrees": (
                ISAAC_RENDER_MAX_UPRIGHT_TILT_DEG
            ),
        },
        "contact_and_door_physics": {
            "contact_report_api": "PhysxContactReportAPI",
            "contact_report_monitor_active": contact_monitor_active,
            "contact_report_impulse_threshold": 0.0,
            "allowed_support_robot_terms": list(
                ISAAC_RENDER_ALLOWED_SUPPORT_ROBOT_TERMS
            ),
            "allowed_support_scene_terms": list(
                ISAAC_RENDER_ALLOWED_SUPPORT_SCENE_TERMS
            ),
            "allowed_target_robot_contact_terms": list(
                ISAAC_RENDER_ALLOWED_TARGET_CONTACT_ROBOT_TERMS
            ),
            "target_manipulator_contact_frame_indices": (
                target_contact_frame_indices
            ),
            "unexpected_robot_collision_frame_indices": (
                unexpected_collision_frame_indices
            ),
            "door_open_angle_span_rad": door_angle_span,
            "meaningful_door_motion_threshold_rad": (
                ISAAC_RENDER_MEANINGFUL_DOOR_MOTION_RAD
            ),
            "first_meaningful_door_motion_frame": (
                first_meaningful_door_motion_frame
            ),
            "door_motion_absent_or_preceded_by_manipulator_contact": (
                door_motion_contact_gated
            ),
            "door_joint_directly_authored": False,
        },
        "active_arm_motion": {
            "links": link_motion,
            "maximum_world_displacement_from_first_m": maximum_world_motion,
            "minimum_required_world_displacement_m": (
                ISAAC_RENDER_REQUIRED_ACTIVE_LINK_WORLD_MOTION_M
            ),
            "maximum_pixel_displacement_from_first_px": maximum_pixel_motion,
            "minimum_required_pixel_displacement_px": (
                ISAAC_RENDER_REQUIRED_ACTIVE_LINK_PIXEL_MOTION_PX
            ),
        },
        "rendered_frame_identity": {
            "distinct_sha256_count": distinct_frame_count,
            "frame_count": len(records),
            "byte_difference_is_not_motion_proof": True,
        },
        "records": [dict(row) for row in records],
        "blockers": blockers,
        "claim_boundary": {
            "controller_target_physics_replay_proven": not blockers,
            "one_physics_step_per_target_proven": not blockers,
            "rendered_post_physics_measured_pose_proven": not blockers,
            "active_arm_link_motion_in_robot_pov_proven": not blockers,
            "learned_policy_execution_not_proven": True,
            "hardware_joint_velocity_limits_not_proven": True,
            "hardware_joint_acceleration_limits_not_proven": True,
            "door_joint_directly_authored": False,
            "contact_reports_captured": contact_monitor_active,
            "unexpected_robot_collision_absent": not (
                unexpected_collision_frame_indices
            ),
            "door_motion_contact_gated_if_present": door_motion_contact_gated,
            "door_articulation_transition_proven": (
                door_angle_readback_valid
                and door_angle_span >= ISAAC_RENDER_MEANINGFUL_DOOR_MOTION_RAD
                and door_motion_contact_gated
            ),
            "semantic_success_not_proven": True,
        },
    }


def _write_render_motion_evidence(seed: Path, evidence: Mapping[str, Any]) -> Path:
    path = seed / "live_aligned_isaac_motion_evidence.json"
    path.write_text(
        json.dumps(dict(evidence), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _finalize_isaac_head_render(
    *,
    seed: Path,
    frames_dir: Path,
    stage_path: Path,
    motion_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Encode and attest frames before SimulationApp shutdown can exit Python."""

    if (
        motion_evidence.get("schema_version")
        != "g1_microwave_live_aligned_isaac_motion_evidence.v2"
        or motion_evidence.get("status") != "passed"
        or list(motion_evidence.get("blockers") or [])
        or dict(motion_evidence.get("physics_execution") or {}).get(
            "one_physics_step_per_controller_target"
        )
        is not True
        or dict(motion_evidence.get("render_synchronization") or {}).get(
            "rendered_from_post_physics_measured_pose"
        )
        is not True
        or dict(motion_evidence.get("render_synchronization") or {}).get(
            "hidden_render_physics_step_absent"
        )
        is not True
        or dict(motion_evidence.get("contact_and_door_physics") or {}).get(
            "contact_report_monitor_active"
        )
        is not True
        or dict(motion_evidence.get("contact_and_door_physics") or {}).get(
            "door_motion_absent_or_preceded_by_manipulator_contact"
        )
        is not True
    ):
        raise RuntimeError("live_aligned_isaac_motion_evidence_not_passed")
    motion_evidence_path = _write_render_motion_evidence(seed, motion_evidence)
    video = seed / "ego_view.mp4"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "17",
            str(video),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not video.is_file() or video.stat().st_size <= 0:
        raise RuntimeError("live_aligned_isaac_video_encode_failed")
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "exact_isaac_rigid_head_episode_rendered",
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "video_path": str(video),
        "video_sha256": _sha256(video),
        "stage_path": str(stage_path),
        "stage_sha256": _sha256(stage_path),
        "camera_role": "robot_pov",
        "camera_motion_model": "rigid_head_local_transform",
        "third_person_used_for_policy": False,
        "door_motion_in_training_render": (
            "physics_only_no_direct_door_authoring_contact_gated_if_present"
        ),
        "motion_evidence_path": str(motion_evidence_path),
        "motion_evidence_sha256": _sha256(motion_evidence_path),
        "articulation_pose_sequence_verified": True,
        "physics_replay_verified": True,
        "one_physics_step_per_target_verified": True,
        "rendered_post_physics_measured_pose_verified": True,
        "active_arm_motion_in_robot_pov_verified": True,
        "contact_report_monitor_verified": True,
        "unexpected_robot_collision_absent": True,
        "door_motion_contact_gated_if_present": True,
        "blockers": [],
        "claim_boundary": {
            "exact_isaac_visual_domain_rendered": True,
            "render_is_owned_training_support": True,
            "controller_target_physics_replay_proven": True,
            "one_physics_step_per_target_proven": True,
            "rendered_post_physics_measured_pose_proven": True,
            "active_arm_link_motion_in_robot_pov_proven": True,
            "learned_policy_execution_not_proven": True,
            "door_joint_directly_authored": False,
            "contact_reports_captured": True,
            "unexpected_robot_collision_absent": True,
            "door_motion_contact_gated_if_present": True,
            "door_articulation_transition_proven": bool(
                dict(motion_evidence.get("claim_boundary") or {}).get(
                    "door_articulation_transition_proven"
                )
            ),
            "semantic_success_not_proven": True,
        },
    }
    report_path = seed / "live_aligned_isaac_render_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def render_isaac(
    *,
    seed_dir: str | Path,
    stage_path: str | Path,
    g1_usd_path: str | Path,
    route_file: str | Path,
    evidence_dir: str | Path,
) -> dict[str, Any]:
    """Render the aligned expert states from the exact rigid head camera."""

    seed = Path(seed_dir).expanduser().resolve()
    trajectory = np.load(seed / "observation_state_43d.npy", allow_pickle=False)
    if trajectory.shape != (FRAME_COUNT, 43) or not np.isfinite(trajectory).all():
        raise ValueError("live_aligned_isaac_trajectory_invalid")
    from blueprint_pipeline.g1_sonic_motion_token_conversion import (
        SOURCE_ACTION_JOINT_NAMES,
    )

    planned_motion = _active_joint_motion(trajectory, SOURCE_ACTION_JOINT_NAMES)
    backend_module = _load_runtime_backend_overlay()
    backend = backend_module.create_backend(
        stage_path=str(Path(stage_path).expanduser().resolve()),
        robot_prim_path="/World/G1",
        evidence_dir=str(Path(evidence_dir).expanduser().resolve()),
        g1_usd_path=str(Path(g1_usd_path).expanduser().resolve()),
        route_file=str(Path(route_file).expanduser().resolve()),
    )
    frames_dir = seed / "isaac_head_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=False)
    try:
        registration = backend._live_robot_registration_link_poses()
        backend.review_renderer.set_initial_robot_pov_calibration_landmarks(
            registration["landmarks"]
        )
        indices = np.asarray(
            [int(backend.robot.get_dof_index(name)) for name in SOURCE_ACTION_JOINT_NAMES],
            dtype=np.int64,
        )
        if np.any(indices < 0) or len(set(indices.tolist())) != 43:
            raise RuntimeError("live_aligned_isaac_joint_mapping_invalid")
        timeline = getattr(backend, "timeline", None)
        play = getattr(timeline, "play", None)
        is_playing = getattr(timeline, "is_playing", None)
        if not callable(play) or not callable(is_playing):
            raise RuntimeError("live_aligned_isaac_timeline_play_api_missing")
        if not bool(is_playing()):
            play()
        commit = getattr(timeline, "commit", None)
        if callable(commit):
            commit()
        if not bool(is_playing()):
            raise RuntimeError("live_aligned_isaac_physics_timeline_not_playing")
        simulation_manager = getattr(backend, "_simulation_manager", None)
        physics_counter = getattr(simulation_manager, "get_num_physics_steps", None)
        simulation_clock = getattr(simulation_manager, "get_simulation_time", None)
        if not callable(physics_counter) or not callable(simulation_clock):
            raise RuntimeError("live_aligned_isaac_physics_clock_missing")
        apply_controller_state = getattr(backend, "_apply_controller_state", None)
        if not callable(apply_controller_state):
            raise RuntimeError("live_aligned_isaac_controller_target_api_missing")
        render_measured_pose = getattr(
            backend.review_renderer, "render_measured_pose", None
        )
        if not callable(render_measured_pose):
            raise RuntimeError("live_aligned_isaac_measured_pose_renderer_missing")
        app_update = getattr(getattr(backend, "app", None), "update", None)
        if not callable(app_update):
            raise RuntimeError("live_aligned_isaac_app_update_api_missing")
        contact_event_cursor = getattr(backend, "contact_event_cursor", None)
        contact_events_since = getattr(backend, "contact_events_since", None)
        measure_door_open_angle = getattr(
            backend, "measure_revolute_task_open_angle", None
        )
        if not callable(contact_event_cursor) or not callable(contact_events_since):
            raise RuntimeError("live_aligned_isaac_contact_report_api_missing")
        if not callable(measure_door_open_angle):
            raise RuntimeError("live_aligned_isaac_door_joint_readback_api_missing")
        active_joint_names = list(planned_motion["active_joint_names"])
        active_joint_indices = [
            SOURCE_ACTION_JOINT_NAMES.index(name) for name in active_joint_names
        ]
        records: list[dict[str, Any]] = []
        for frame_index, positions in enumerate(trajectory):
            commanded = np.asarray(positions, dtype=np.float32)
            door_sample_before = dict(measure_door_open_angle(TARGET_PRIM_PATH))
            door_open_angle_before = float(door_sample_before.get("value_rad"))
            if not math.isfinite(door_open_angle_before):
                raise RuntimeError("live_aligned_isaac_door_joint_readback_invalid")
            contact_cursor = int(contact_event_cursor())
            apply_controller_state(
                {
                    "joint_names": list(SOURCE_ACTION_JOINT_NAMES),
                    "joint_positions": commanded.tolist(),
                    "joint_order_schema_version": str(
                        backend_module.JOINT_ORDER_SCHEMA_VERSION
                    ),
                    "mapping_digest": str(backend_module.PROTOCOL_V4_MAPPING_DIGEST),
                }
            )
            physics_before = int(physics_counter())
            simulation_time_before = float(simulation_clock())
            app_update()
            physics_after = int(physics_counter())
            simulation_time_after = float(simulation_clock())
            physics_step_delta = physics_after - physics_before
            simulation_time_delta = simulation_time_after - simulation_time_before
            if physics_step_delta != 1 or not math.isclose(
                simulation_time_delta,
                1.0 / float(FPS),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "live_aligned_isaac_controller_target_physics_step_invalid:"
                    f"{frame_index}:{physics_step_delta}:{simulation_time_delta:.12f}"
                )
            contact_classification = _classify_robot_contact_events(
                list(contact_events_since(contact_cursor)),
                robot_prim_path=str(backend.robot_prim_path),
                target_prim_path=TARGET_PRIM_PATH,
            )
            door_sample = dict(measure_door_open_angle(TARGET_PRIM_PATH))
            door_open_angle = float(door_sample.get("value_rad"))
            if not math.isfinite(door_open_angle):
                raise RuntimeError("live_aligned_isaac_door_joint_readback_invalid")
            measured_positions = _robot_joint_readback(backend.robot, indices)
            measured_velocities = _robot_joint_velocity_readback(
                backend.robot, indices
            )
            tracking_errors = np.abs(
                measured_positions - commanded.astype(np.float64)
            )
            target_max_error = float(np.max(tracking_errors, initial=0.0))
            active_mean_error = float(
                np.mean(tracking_errors[active_joint_indices])
            )
            world_position, _world_orientation = backend.robot.get_world_pose()
            base_position = np.asarray(world_position, dtype=np.float64).reshape(-1)
            if base_position.shape != (3,) or not np.isfinite(base_position).all():
                raise RuntimeError("live_aligned_isaac_base_pose_invalid")
            projected_gravity = [
                float(value) for value in backend._live_projected_gravity()
            ]
            render_physics_before = int(physics_counter())
            rows = list(
                render_measured_pose(
                    step_index=frame_index,
                    target_prim_path=TARGET_PRIM_PATH,
                )
            )
            render_physics_after = int(physics_counter())
            render_physics_step_delta = (
                render_physics_after - render_physics_before
            )
            if render_physics_step_delta != 0:
                raise RuntimeError(
                    "live_aligned_isaac_render_added_hidden_physics_step:"
                    f"{frame_index}:{render_physics_step_delta}"
                )
            if not bool(is_playing()):
                raise RuntimeError(
                    "live_aligned_isaac_renderer_paused_physics_timeline:"
                    f"{frame_index}"
                )
            robot_rows = [
                dict(row)
                for row in rows
                if isinstance(row, dict) and row.get("camera_role") == "robot_pov"
            ]
            if len(robot_rows) != 1:
                raise RuntimeError("live_aligned_isaac_robot_pov_not_unique")
            source = Path(str(robot_rows[0].get("path") or "")).resolve()
            if not source.is_file() or _sha256(source) != robot_rows[0].get("sha256"):
                raise RuntimeError("live_aligned_isaac_robot_pov_digest_invalid")
            destination = frames_dir / f"frame_{frame_index:06d}.png"
            shutil.copyfile(source, destination)
            registration = backend._live_robot_registration_link_poses()
            landmarks = _project_active_arm_landmarks(
                camera_contract=dict(robot_rows[0].get("camera_contract") or {}),
                registration=registration,
            )
            records.append(
                {
                    "frame_index": frame_index,
                    "commanded_pose_sha256": hashlib.sha256(
                        np.ascontiguousarray(commanded).tobytes()
                    ).hexdigest(),
                    "target_joint_max_error_rad": target_max_error,
                    "active_joint_mean_tracking_error_rad": active_mean_error,
                    "active_joint_positions": {
                        name: float(measured_positions[index])
                        for name, index in zip(
                            active_joint_names,
                            active_joint_indices,
                            strict=True,
                        )
                    },
                    "active_joint_velocities_rad_s": {
                        name: float(measured_velocities[index])
                        for name, index in zip(
                            active_joint_names,
                            active_joint_indices,
                            strict=True,
                        )
                    },
                    "physics_step_count_before": physics_before,
                    "physics_step_count_after": physics_after,
                    "physics_step_delta": physics_step_delta,
                    "simulation_time_before_seconds": simulation_time_before,
                    "simulation_time_after_seconds": simulation_time_after,
                    "simulation_time_delta_seconds": simulation_time_delta,
                    "render_physics_step_delta": render_physics_step_delta,
                    "contact_report_monitor_active": True,
                    "active_contact_event_count": int(
                        contact_classification["active_contact_event_count"]
                    ),
                    "allowed_support_contact_event_count": len(
                        contact_classification["allowed_support_contact_events"]
                    ),
                    "target_manipulator_contact": bool(
                        contact_classification[
                            "target_manipulator_contact_events"
                        ]
                    ),
                    "target_manipulator_contact_events": list(
                        contact_classification[
                            "target_manipulator_contact_events"
                        ]
                    ),
                    "unexpected_robot_collision_events": list(
                        contact_classification[
                            "unexpected_robot_collision_events"
                        ]
                    ),
                    "door_open_angle_rad": door_open_angle,
                    "door_open_angle_before_step_rad": door_open_angle_before,
                    "door_joint_measurement": door_sample,
                    "base_position_xyz_m": base_position.tolist(),
                    "projected_gravity": projected_gravity,
                    "active_arm_landmarks": landmarks,
                    "frame_sha256": _sha256(destination),
                }
            )
        motion_evidence = _summarize_render_motion(
            records=records,
            planned_motion=planned_motion,
        )
        _write_render_motion_evidence(seed, motion_evidence)
        if motion_evidence["status"] != "passed":
            raise RuntimeError(
                "live_aligned_isaac_render_motion_validation_failed:"
                + ",".join(motion_evidence["blockers"])
            )
        # Isaac's SimulationApp.close() may terminate the interpreter as part
        # of Kit shutdown.  Finalize the durable media while the app is still
        # alive so successful frame rendering cannot be discarded on exit.
        return _finalize_isaac_head_render(
            seed=seed,
            frames_dir=frames_dir,
            stage_path=Path(stage_path).expanduser().resolve(),
            motion_evidence=motion_evidence,
        )
    finally:
        backend.close()


def _numeric_stats(array: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(array, dtype=np.float64)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def patch_dataset(*, seed_dir: str | Path, dataset_dir: str | Path) -> dict[str, Any]:
    """Replace the one LeRobot episode and all action/state statistics."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    seed = Path(seed_dir).expanduser().resolve()
    dataset = Path(dataset_dir).expanduser().resolve()
    render_report = _load_object(
        seed / "live_aligned_isaac_render_report.json",
        label="live_aligned_isaac_render_report",
    )
    motion_evidence = _load_object(
        seed / "live_aligned_isaac_motion_evidence.json",
        label="live_aligned_isaac_motion_evidence",
    )
    if (
        render_report.get("status") != "exact_isaac_rigid_head_episode_rendered"
        or render_report.get("articulation_pose_sequence_verified") is not True
        or render_report.get("physics_replay_verified") is not True
        or render_report.get("one_physics_step_per_target_verified") is not True
        or render_report.get("rendered_post_physics_measured_pose_verified") is not True
        or render_report.get("active_arm_motion_in_robot_pov_verified") is not True
        or render_report.get("contact_report_monitor_verified") is not True
        or render_report.get("unexpected_robot_collision_absent") is not True
        or render_report.get("door_motion_contact_gated_if_present") is not True
        or list(render_report.get("blockers") or [])
        or motion_evidence.get("status") != "passed"
        or list(motion_evidence.get("blockers") or [])
        or render_report.get("motion_evidence_sha256")
        != _sha256(seed / "live_aligned_isaac_motion_evidence.json")
    ):
        raise ValueError("live_aligned_isaac_render_motion_not_qualified")
    arrays = {
        "observation.state": np.load(
            seed / "observation_state_43d.npy", allow_pickle=False
        ).astype(np.float64),
        "observation.projected_gravity": np.load(
            seed / "observation_projected_gravity.npy", allow_pickle=False
        ).astype(np.float64),
        "action.motion_token": np.load(
            seed / "action_motion_token_64d.npy", allow_pickle=False
        ).astype(np.float64),
        "teleop.left_hand_joints": np.load(
            seed / "teleop_left_hand_joints_7d.npy", allow_pickle=False
        ).astype(np.float32),
        "teleop.right_hand_joints": np.load(
            seed / "teleop_right_hand_joints_7d.npy", allow_pickle=False
        ).astype(np.float32),
    }
    if any(value.shape[0] != FRAME_COUNT for value in arrays.values()):
        raise ValueError("live_aligned_dataset_array_horizon_invalid")
    parquet_path = dataset / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(parquet_path)
    for name, values in arrays.items():
        index = table.schema.get_field_index(name)
        if index < 0:
            raise ValueError(f"live_aligned_dataset_column_missing:{name}")
        table = table.set_column(
            index,
            name,
            pa.array(values.tolist(), type=table.schema.field(index).type),
        )
    pq.write_table(table, parquet_path, compression="snappy")
    video_target = (
        dataset
        / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4"
    )
    shutil.copyfile(seed / "ego_view.mp4", video_target)

    timestamps = np.arange(FRAME_COUNT, dtype=np.float64) / float(FPS)
    stats = {name: _numeric_stats(values) for name, values in arrays.items()}
    stats["timestamp"] = _numeric_stats(timestamps[:, None])
    fingerprints = {
        name: "sha256:" + hashlib.sha256(
            np.ascontiguousarray(values).tobytes()
        ).hexdigest()
        for name, values in {**arrays, "timestamp": timestamps[:, None]}.items()
    }
    stats["__fingerprints__"] = fingerprints
    stats_path = dataset / "meta/stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    episode_stats = {key: value for key, value in stats.items() if key != "__fingerprints__"}
    (dataset / "meta/episodes_stats.jsonl").write_text(
        json.dumps({"episode_index": 0, "stats": episode_stats}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    materialization_path = dataset / "materialization_report.json"
    materialization = _load_object(
        materialization_path, label="live_aligned_materialization_report"
    )
    materialization["live_alignment"] = {
        "schema_version": SCHEMA_VERSION,
        "status": "same_session_live_start_and_exact_isaac_head_view_bound",
        "preparation_report_sha256": _sha256(
            seed / "live_aligned_action_preparation.json"
        ),
        "render_report_sha256": _sha256(seed / "live_aligned_isaac_render_report.json"),
        "motion_evidence_sha256": _sha256(
            seed / "live_aligned_isaac_motion_evidence.json"
        ),
        "stats_sha256": _sha256(stats_path),
        "parquet_sha256": _sha256(parquet_path),
        "video_sha256": _sha256(video_target),
    }
    materialization_path.write_text(
        json.dumps(materialization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    preflight_path = dataset / "groot_n17_finetune_preflight.json"
    preflight = _load_object(
        preflight_path, label="live_aligned_groot_finetune_preflight"
    )
    loader = dict(preflight.get("training_loader") or {})
    loader["stats_sha256"] = _sha256(stats_path)
    loader["live_aligned_isaac_head_view"] = True
    preflight["training_loader"] = loader
    preflight["live_alignment"] = dict(materialization["live_alignment"])
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "live_aligned_lerobot_episode_patched",
        "dataset_dir": str(dataset),
        "parquet_sha256": _sha256(parquet_path),
        "video_sha256": _sha256(video_target),
        "stats_sha256": _sha256(stats_path),
        "frame_count": FRAME_COUNT,
        "blockers": [],
    }
    report_path = seed / "live_aligned_dataset_patch_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-actions")
    prepare.add_argument("--initial-state", required=True)
    prepare.add_argument("--standing-report", required=True)
    prepare.add_argument("--initial-observation", required=True)
    prepare.add_argument("--robot-model", required=True)
    prepare.add_argument("--encoder", required=True)
    prepare.add_argument("--output-dir", required=True)
    render = subparsers.add_parser("render-isaac")
    render.add_argument("--seed-dir", required=True)
    render.add_argument("--stage", required=True)
    render.add_argument("--g1-usd", required=True)
    render.add_argument("--route-file", required=True)
    render.add_argument("--evidence-dir", required=True)
    patch = subparsers.add_parser("patch-dataset")
    patch.add_argument("--seed-dir", required=True)
    patch.add_argument("--dataset-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "prepare-actions":
        prepare_actions(
            initial_state_path=args.initial_state,
            standing_report_path=args.standing_report,
            initial_observation_path=args.initial_observation,
            robot_model_path=args.robot_model,
            encoder_path=args.encoder,
            output_dir=args.output_dir,
        )
    elif args.command == "render-isaac":
        render_isaac(
            seed_dir=args.seed_dir,
            stage_path=args.stage,
            g1_usd_path=args.g1_usd,
            route_file=args.route_file,
            evidence_dir=args.evidence_dir,
        )
    else:
        patch_dataset(seed_dir=args.seed_dir, dataset_dir=args.dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
