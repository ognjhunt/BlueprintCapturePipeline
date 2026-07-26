"""One-process Isaac backend for controller application and task measurement.

Imports are intentionally lazy: this module is shipped in the worker image and
is not importable as an Isaac runtime on ordinary CI hosts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .g1_proprioception_map import (
    G1_PROPRIOCEPTION_MAP_SCHEMA_VERSION,
    resolve_g1_proprioception_map,
)
from .isaac_task_review_renderer import project_world_point
from .gear_sonic_joint_order_contract import (
    JOINT_ORDER_SCHEMA_VERSION,
    PINNED_WBC_SOURCE_REVISION,
    PROTOCOL_V4_BODY_JOINT_NAMES,
    PROTOCOL_V4_FULL_JOINT_ORDER,
    PROTOCOL_V4_LEFT_HAND_JOINT_NAMES,
    PROTOCOL_V4_MAPPING_DIGEST,
    PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES,
    validate_model_joint_names,
    validate_full_joint_order,
)
from .task_episode_baseline import (
    build_task_episode_baseline,
    canonical_task_contract_sha256,
    evaluate_task_criterion,
    verify_task_episode_baseline,
)


def normalize_physx_contact_reports(
    contact_headers: Sequence[Any],
    contact_data: Sequence[Any],
    *,
    path_decoder: Callable[[Any], Any],
) -> list[dict[str, Any]]:
    """Convert PhysX callback objects into durable, JSON-safe contact evidence."""

    normalized: list[dict[str, Any]] = []

    def decode(value: Any) -> str:
        path = str(path_decoder(value))
        if not path.startswith("/"):
            raise RuntimeError("persistent_isaac_contact_report_path_invalid")
        return path

    for header in contact_headers:
        offset = int(header.contact_data_offset)
        count = int(header.num_contact_data)
        if offset < 0 or count < 0 or offset + count > len(contact_data):
            raise RuntimeError("persistent_isaac_contact_report_range_invalid")
        impulses: list[float] = []
        minimum_separation = math.inf
        for item in contact_data[offset : offset + count]:
            raw_impulse = item.impulse
            try:
                impulse_values = [float(value) for value in raw_impulse]
            except TypeError:
                impulse_values = [float(raw_impulse)]
            if not impulse_values or not all(
                math.isfinite(value) for value in impulse_values
            ):
                raise RuntimeError("persistent_isaac_contact_impulse_invalid")
            impulses.append(math.sqrt(sum(value * value for value in impulse_values)))
            separation = float(item.separation)
            if not math.isfinite(separation):
                raise RuntimeError("persistent_isaac_contact_separation_invalid")
            minimum_separation = min(minimum_separation, separation)
        normalized.append(
            {
                "event_type": str(header.type),
                "actor0_prim_path": decode(header.actor0),
                "actor1_prim_path": decode(header.actor1),
                "collider0_prim_path": decode(header.collider0),
                "collider1_prim_path": decode(header.collider1),
                "contact_point_count": count,
                "maximum_impulse": max(impulses, default=0.0),
                "total_impulse": sum(impulses),
                "minimum_separation_m": minimum_separation if count else None,
            }
        )
    return normalized


GEAR_SONIC_STANDING_INITIALIZATION_SCHEMA_VERSION = "gear_sonic_standing_initialization.v1"
GEAR_SONIC_MANIPULATION_READY_INITIALIZATION_SCHEMA_VERSION = (
    "gear_sonic_manipulation_ready_initialization.v1"
)
GEAR_SONIC_ISAAC_STATE_SNAPSHOT_SCHEMA_VERSION = "gear_sonic_isaac_state_snapshot.v1"
GEAR_SONIC_ISAAC_STATE_SNAPSHOT_ENV = "BLUEPRINT_GEAR_SONIC_ISAAC_STATE_SNAPSHOT"
GEAR_SONIC_ISAAC_STATE_SNAPSHOT_DEFAULT_PATH = (
    "/workspace/closed_loop_out/gear_sonic_isaac_state_snapshot.json"
)
GEAR_SONIC_ISAAC_STATE_FRESHNESS_WINDOW_NS = 500_000_000
GEAR_SONIC_STANDING_JOINT_POSITION_TOLERANCE_RAD = 0.05
GEAR_SONIC_STANDING_JOINT_VELOCITY_TOLERANCE_RAD_S = 0.05
# Restored from the validated late-June manipulation camera lane.  The
# official standing pose is still applied, measured, and persisted first; this
# second pose only raises the task-side arm into a forward manipulation seed so
# a true head camera can see the active forearm without pitching down into an
# overhead view.
GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_DELTAS_RAD: dict[str, float] = {
    "right_shoulder_pitch_joint": -0.85,
    "right_shoulder_roll_joint": -0.15,
    "right_shoulder_yaw_joint": -0.10,
    "right_elbow_joint": -0.23,
    "right_wrist_roll_joint": 0.10,
    "right_wrist_pitch_joint": -0.15,
}
GEAR_SONIC_MANIPULATION_READY_MIN_REQUESTED_DELTA_FRACTION = 0.9
GEAR_SONIC_MANIPULATION_READY_MAX_SETTLE_STEPS = 120
# The task-ready seed is accepted on the joints it intentionally moves.  The
# live G1 hand and floating-base dynamics can remain active after official
# standing verification; treating those unrelated velocities as arm-seed
# failure rejected an otherwise exact late-June pose.  Half a radian/second is
# still a tight one-control-frame bound, and the immediately following rigid
# head-camera render must independently prove the elbow and wrist are in frame.
GEAR_SONIC_MANIPULATION_READY_JOINT_VELOCITY_TOLERANCE_RAD_S = 0.5
ROBOT_POV_REQUIRED_ACTIVE_ARM_LINK_NAMES: tuple[str, ...] = (
    "right_elbow_link",
    "right_wrist_yaw_link",
)
CONTROLLER_FK_CAMERA_PROJECTION_SCHEMA_VERSION = (
    "controller_fk_camera_projection_context.v1"
)
CONTROLLER_FK_CAMERA_PROJECTION_TRANSFORM = (
    "mujoco_pelvis_relative_to_live_isaac_pelvis_wxyz"
)
CONTROLLER_FK_REGISTRATION_LANDMARK_NAMES: tuple[str, ...] = (
    "left_shoulder_pitch_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
CONTROLLER_FK_REGISTRATION_MAX_RESIDUAL_M = 0.025
ISAAC_PHYSX_LIVE_RIGID_BODY_POSE_SOURCE = (
    "omni_physx_get_rigidbody_transformation_world_pose_xyzw_reordered_wxyz"
)
CONTROLLER_HORIZON_EXECUTION_SCHEMA_VERSION = (
    "gear_sonic_controller_horizon_execution.v1"
)
CONTROLLER_ACTION_SEQUENCE_SCHEMA_VERSION = "gear_sonic_controller_action_sequence.v1"
GEAR_SONIC_CONTROL_HZ = 50.0
ISAAC_REVIEW_SAMPLE_HZ = 10.0
ISAAC_REVIEW_CONTROLLER_FRAME_STRIDE = int(
    GEAR_SONIC_CONTROL_HZ / ISAAC_REVIEW_SAMPLE_HZ
)
GEAR_SONIC_CONTROL_DT_SECONDS = 1.0 / GEAR_SONIC_CONTROL_HZ
# Match the validated G1 footprint used by the late-June kitchen stance
# planner. The previous generic 0.45 x 0.45 x 0.90 m half-extents engulfed the
# appliance the robot was intentionally standing near and marked every
# manipulation stance as colliding.
G1_LIVE_COLLISION_HALF_EXTENT_M = (0.12, 0.23, 0.62)
ISAAC_REVIEW_RENDERER_ARTICULATION_LIFECYCLE_SCHEMA_VERSION = (
    "persistent_isaac_review_renderer_articulation_lifecycle.v1"
)
ISAAC_ARTICULATION_STARTUP_UPDATE_COUNT = 8


def configure_and_verify_simulation_control_clock(
    *,
    stage: Any,
    timeline: Any,
    app: Any | None = None,
    simulation_manager: Any | None = None,
    rendering_manager: Any | None = None,
) -> dict[str, Any]:
    """Configure and read back one coherent 50 Hz physics/render/controller clock."""

    if simulation_manager is None:
        from isaacsim.core.simulation_manager import SimulationManager  # type: ignore

        simulation_manager = SimulationManager
    if rendering_manager is None:
        from isaacsim.core.rendering_manager import RenderingManager  # type: ignore

        rendering_manager = RenderingManager
    timeline.stop()
    # Timeline mutations are pending until committed.  Commit the stopped
    # state before touching physics, then commit the timing settings before
    # reading them back; otherwise a valid setter call can be compared against
    # the preceding stage's values.
    timeline.commit()
    # Kitchen stages may author their own timeCodesPerSecond (commonly 24 or
    # 60).  The controller clock contract cannot merely read back 50 Hz and
    # assume the stage already matches it: author the same 50 Hz clock before
    # configuring the independent Isaac physics/render clocks.
    stage.SetTimeCodesPerSecond(GEAR_SONIC_CONTROL_HZ)
    simulation_manager.setup_simulation(dt=GEAR_SONIC_CONTROL_DT_SECONDS)
    physics_scenes = list(simulation_manager.get_physics_scenes())
    if len(physics_scenes) != 1:
        raise RuntimeError("persistent_isaac_single_physics_scene_required")
    # ``PhysxScene.set_dt`` converts the requested float dt to an integer
    # steps-per-second value.  Isaac documents that conversion as truncating,
    # so the integer controller rate is the authoritative value to author and
    # verify.  This avoids a float -> int -> float round trip changing a 50 Hz
    # controller contract before the first live step.
    for scene in physics_scenes:
        scene.set_steps_per_second(int(GEAR_SONIC_CONTROL_HZ))
    rendering_manager.set_dt(GEAR_SONIC_CONTROL_DT_SECONDS)
    timeline.set_target_framerate(GEAR_SONIC_CONTROL_HZ)
    # Isaac's ``play every frame`` switch enables fast-forward/useFastMode.  A
    # controller evidence clock must instead prove real one-tick progression.
    timeline.set_play_every_frame(False)
    timeline.commit()
    physics_dts = [float(scene.get_dt()) for scene in physics_scenes]
    physics_steps_per_second = [
        float(scene.get_steps_per_second()) for scene in physics_scenes
    ]
    render_dt = float(rendering_manager.get_dt())
    stage_time_codes_per_second = float(stage.GetTimeCodesPerSecond())
    target_framerate = float(timeline.get_target_framerate())
    play_every_frame = bool(timeline.get_play_every_frame())
    mismatched_fields = [
        *[
            f"physics_dt_seconds[{index}]"
            for index, value in enumerate(physics_dts)
            if not math.isclose(
                value,
                GEAR_SONIC_CONTROL_DT_SECONDS,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ],
        *[
            f"physics_steps_per_second[{index}]"
            for index, value in enumerate(physics_steps_per_second)
            if not math.isclose(
                value,
                GEAR_SONIC_CONTROL_HZ,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ],
    ]
    if not math.isclose(
        render_dt,
        GEAR_SONIC_CONTROL_DT_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        mismatched_fields.append("render_loop_dt_seconds")
    if not math.isclose(
        stage_time_codes_per_second,
        GEAR_SONIC_CONTROL_HZ,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        mismatched_fields.append("time_codes_per_second")
    if not math.isclose(
        target_framerate,
        GEAR_SONIC_CONTROL_HZ,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        mismatched_fields.append("target_framerate")
    if play_every_frame:
        mismatched_fields.append("play_every_frame")
    if mismatched_fields:
        diagnostic = {
            "expected": {
                "physics_dt_seconds": GEAR_SONIC_CONTROL_DT_SECONDS,
                "physics_steps_per_second": GEAR_SONIC_CONTROL_HZ,
                "render_loop_dt_seconds": GEAR_SONIC_CONTROL_DT_SECONDS,
                "time_codes_per_second": GEAR_SONIC_CONTROL_HZ,
                "target_framerate": GEAR_SONIC_CONTROL_HZ,
                "play_every_frame": False,
            },
            "mismatched_fields": mismatched_fields,
            "observed": {
                "physics_dt_seconds": physics_dts,
                "physics_steps_per_second": physics_steps_per_second,
                "render_loop_dt_seconds": render_dt,
                "time_codes_per_second": stage_time_codes_per_second,
                "target_framerate": target_framerate,
                "play_every_frame": play_every_frame,
            },
        }
        raise RuntimeError(
            "persistent_isaac_controller_clock_readback_mismatch:"
            + json.dumps(diagnostic, sort_keys=True, separators=(",", ":"))
        )

    preflight: dict[str, Any] = {
        "performed": False,
        "physics_step_delta": None,
        "simulation_time_delta_seconds": None,
    }
    if app is not None:
        timeline.play()
        timeline.commit()
        # Playing can initialize physics.  Establish the one-update baseline
        # only after that lifecycle transition has been committed so the
        # measured delta belongs solely to the following controller tick.
        physics_steps_before = int(simulation_manager.get_num_physics_steps())
        simulation_time_before = float(simulation_manager.get_simulation_time())
        app.update()
        physics_steps_after = int(simulation_manager.get_num_physics_steps())
        simulation_time_after = float(simulation_manager.get_simulation_time())
        timeline.stop()
        timeline.commit()
        physics_step_delta = physics_steps_after - physics_steps_before
        simulation_time_delta = simulation_time_after - simulation_time_before
        if physics_step_delta != 1 or not math.isclose(
            simulation_time_delta,
            GEAR_SONIC_CONTROL_DT_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            diagnostic = {
                "expected": {
                    "physics_step_delta": 1,
                    "simulation_time_delta_seconds": GEAR_SONIC_CONTROL_DT_SECONDS,
                },
                "observed": {
                    "physics_step_count_before": physics_steps_before,
                    "physics_step_count_after": physics_steps_after,
                    "physics_step_delta": physics_step_delta,
                    "simulation_time_before_seconds": simulation_time_before,
                    "simulation_time_after_seconds": simulation_time_after,
                    "simulation_time_delta_seconds": simulation_time_delta,
                },
            }
            raise RuntimeError(
                "persistent_isaac_controller_clock_preflight_failed:"
                + json.dumps(diagnostic, sort_keys=True, separators=(",", ":"))
            )
        preflight = {
            "performed": True,
            "physics_step_count_before": physics_steps_before,
            "physics_step_count_after": physics_steps_after,
            "physics_step_delta": physics_step_delta,
            "simulation_time_before_seconds": simulation_time_before,
            "simulation_time_after_seconds": simulation_time_after,
            "simulation_time_delta_seconds": simulation_time_delta,
            "timeline_stopped_after_preflight": True,
        }
    return {
        "schema_version": "persistent_isaac_controller_timing.v2",
        "physics_dt_seconds": physics_dts,
        "physics_steps_per_second": physics_steps_per_second,
        "render_loop_dt_seconds": render_dt,
        "time_codes_per_second": stage_time_codes_per_second,
        "target_framerate": target_framerate,
        "play_every_frame": play_every_frame,
        "one_physics_update_per_controller_frame": True,
        "clock_configuration": (
            "UsdStage.SetTimeCodesPerSecond_and_"
            "SimulationManager.setup_simulation_and_RenderingManager.set_dt"
        ),
        "clock_readback_verified": True,
        "preflight": preflight,
        "authored_usd_saved": False,
    }
GEAR_SONIC_ACTION_DIMENSION = 78

# Pinned from gear_sonic_deploy/g1/include/policy_parameters.hpp at
# PINNED_WBC_SOURCE_REVISION.  The tuple is in the exact protocol-v4 body
# order above; hands are deliberately initialized to zero.
GEAR_SONIC_DEFAULT_BODY_STANDING_POSITIONS: tuple[float, ...] = (
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
)
GEAR_SONIC_DEFAULT_FULL_STANDING_POSITIONS: tuple[float, ...] = (
    GEAR_SONIC_DEFAULT_BODY_STANDING_POSITIONS
    + (0.0,) * len(PROTOCOL_V4_LEFT_HAND_JOINT_NAMES)
    + (0.0,) * len(PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES)
)


def _right_arm_manipulation_ready_positions() -> tuple[float, ...]:
    positions = list(GEAR_SONIC_DEFAULT_FULL_STANDING_POSITIONS)
    index_by_name = {
        name: index for index, name in enumerate(PROTOCOL_V4_FULL_JOINT_ORDER)
    }
    for name, delta in GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_DELTAS_RAD.items():
        positions[index_by_name[name]] += float(delta)
    return tuple(positions)


GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_POSITIONS = (
    _right_arm_manipulation_ready_positions()
)
GEAR_SONIC_DEFAULT_STANDING_POSE_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "body_joint_names": list(PROTOCOL_V4_BODY_JOINT_NAMES),
            "body_joint_positions": list(GEAR_SONIC_DEFAULT_BODY_STANDING_POSITIONS),
            "left_hand_joint_names": list(PROTOCOL_V4_LEFT_HAND_JOINT_NAMES),
            "left_hand_joint_positions": [0.0 for _ in PROTOCOL_V4_LEFT_HAND_JOINT_NAMES],
            "right_hand_joint_names": list(PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES),
            "right_hand_joint_positions": [0.0 for _ in PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES],
            "joint_order_schema_version": JOINT_ORDER_SCHEMA_VERSION,
            "mapping_digest": PROTOCOL_V4_MAPPING_DIGEST,
            "pinned_wbc_source_revision": PINNED_WBC_SOURCE_REVISION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_POSE_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "joint_names": list(PROTOCOL_V4_FULL_JOINT_ORDER),
            "joint_positions": list(
                GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_POSITIONS
            ),
            "source_standing_pose_sha256": GEAR_SONIC_DEFAULT_STANDING_POSE_SHA256,
            "task_side": "right",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _finite_float_vector(value: Any, *, expected_length: int, error_code: str) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(error_code) from exc
    if len(result) != expected_length or not all(math.isfinite(item) for item in result):
        raise RuntimeError(error_code)
    return result


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _validated_controller_execution_sequence(
    *,
    state: Mapping[str, Any],
    action: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    """Validate a complete GEAR horizon before any frame reaches PhysX.

    Legacy callers without a sequence remain one-frame receding-horizon
    actions. An explicit sequence is all-or-nothing: its action frames, FK
    rows, timing, hashes, order, and final legacy state must agree before the
    first target is applied.
    """

    raw_sequence = state.get("controller_fk_sequence")
    if raw_sequence is None:
        return [dict(state)], {
            "schema_version": "persistent_isaac_legacy_single_controller_frame.v1",
            "execution_mode": "legacy_single_frame_receding_horizon",
            "execution_frame_count": 1,
            "control_hz": None,
            "sample_period_seconds": None,
        }, False
    if isinstance(raw_sequence, (str, bytes, bytearray, Mapping)) or not isinstance(
        raw_sequence, Sequence
    ):
        raise RuntimeError("persistent_isaac_controller_fk_sequence_invalid")
    sequence = [dict(row) for row in raw_sequence if isinstance(row, Mapping)]
    if not sequence or len(sequence) != len(raw_sequence):
        raise RuntimeError("persistent_isaac_controller_fk_sequence_invalid")
    if int(state.get("executed_control_frame_count") or 0) != len(sequence):
        raise RuntimeError("persistent_isaac_controller_fk_sequence_count_mismatch")
    sequence_sha256 = _canonical_sha256(sequence)
    if str(state.get("controller_fk_sequence_sha256") or "").lower() != sequence_sha256:
        raise RuntimeError("persistent_isaac_controller_fk_sequence_sha256_mismatch")

    contract = dict(state.get("controller_execution_contract") or {})
    try:
        control_hz = float(contract.get("control_hz"))
        sample_period = float(contract.get("sample_period_seconds"))
        duration = float(contract.get("declared_execution_duration_seconds"))
        execution_count = int(contract.get("execution_frame_count") or 0)
        source_count = int(contract.get("source_horizon_frame_count") or 0)
        frame_dimension = int(contract.get("frame_dimension") or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("persistent_isaac_controller_execution_contract_invalid") from exc
    if (
        contract.get("schema_version") != CONTROLLER_HORIZON_EXECUTION_SCHEMA_VERSION
        or str(contract.get("execution_mode") or "")
        not in {"single_frame_receding_horizon", "bounded_model_horizon_prefix"}
        or int(contract.get("controller_session_count") or 0) != 1
        or execution_count != len(sequence)
        or source_count < execution_count
        or frame_dimension != GEAR_SONIC_ACTION_DIMENSION
        or not math.isclose(control_hz, GEAR_SONIC_CONTROL_HZ, rel_tol=0.0, abs_tol=1e-9)
        or not math.isclose(sample_period, 1.0 / control_hz, rel_tol=0.0, abs_tol=1e-9)
        or not math.isclose(
            duration,
            execution_count / control_hz,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or str(contract.get("controller_fk_sequence_sha256") or "").lower()
        != sequence_sha256
        or str(contract.get("final_controller_fk_frame_sha256") or "").lower()
        != _canonical_sha256(sequence[-1])
        or str(contract.get("controller_state_sequence_sha256") or "").lower()
        != _canonical_sha256(
            [str(row.get("controller_state_sha256") or "") for row in sequence]
        )
    ):
        raise RuntimeError("persistent_isaac_controller_execution_contract_invalid")

    controller_action = dict(action.get("controller_action") or {})
    raw_action_frames = controller_action.get("frames")
    if (
        controller_action.get("schema_version")
        != CONTROLLER_ACTION_SEQUENCE_SCHEMA_VERSION
        or isinstance(raw_action_frames, (str, bytes, bytearray, Mapping))
        or not isinstance(raw_action_frames, Sequence)
    ):
        raise RuntimeError("persistent_isaac_controller_action_sequence_invalid")
    action_frames = [
        _finite_float_vector(
            frame,
            expected_length=GEAR_SONIC_ACTION_DIMENSION,
            error_code="persistent_isaac_controller_action_frame_invalid",
        )
        for frame in raw_action_frames
    ]
    if (
        int(controller_action.get("execution_frame_count") or 0) != len(sequence)
        or int(controller_action.get("source_horizon_frame_count") or 0) != source_count
        or int(controller_action.get("frame_dimension") or 0)
        != GEAR_SONIC_ACTION_DIMENSION
        or str(controller_action.get("execution_mode") or "")
        != str(contract.get("execution_mode") or "")
        or not math.isclose(
            float(controller_action.get("control_hz") or 0.0),
            control_hz,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(controller_action.get("sample_period_seconds") or 0.0),
            sample_period,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or len(action_frames) != len(sequence)
        or _canonical_sha256(action_frames)
        != str(controller_action.get("frames_sha256") or "").lower()
        or str(contract.get("input_action_frames_sha256") or "").lower()
        != _canonical_sha256(action_frames)
        or str(contract.get("source_action_frames_sha256") or "").lower()
        != str(controller_action.get("source_frames_sha256") or "").lower()
    ):
        raise RuntimeError("persistent_isaac_controller_action_sequence_mismatch")

    for index, (row, action_frame) in enumerate(zip(sequence, action_frames, strict=True)):
        names = [str(item) for item in row.get("joint_names") or []]
        positions = _finite_float_vector(
            row.get("joint_positions"),
            expected_length=len(PROTOCOL_V4_FULL_JOINT_ORDER),
            error_code="persistent_isaac_controller_fk_sequence_joint_state_invalid",
        )
        try:
            validate_full_joint_order(names, source=f"persistent_isaac_controller_frame_{index}")
        except ValueError as exc:
            raise RuntimeError(
                "persistent_isaac_controller_fk_sequence_joint_mapping_invalid"
            ) from exc
        applied_mapping = list(row.get("applied_dof_mapping") or [])
        controller_state_sha256 = str(row.get("controller_state_sha256") or "").lower()
        if (
            int(row.get("horizon_frame_index", -1)) != index
            or str(row.get("source_action_frame_sha256") or "").lower()
            != _canonical_sha256(action_frame)
            or len(applied_mapping) != len(names)
            or any(
                not isinstance(mapping, Mapping)
                or str(mapping.get("joint_name") or "") != names[mapping_index]
                for mapping_index, mapping in enumerate(applied_mapping)
            )
            or len(controller_state_sha256) != 64
            or any(character not in "0123456789abcdef" for character in controller_state_sha256)
        ):
            raise RuntimeError("persistent_isaac_controller_fk_sequence_binding_invalid")
        row["joint_names"] = names
        row["joint_positions"] = positions

    if (
        list(state.get("joint_names") or []) != sequence[-1]["joint_names"]
        or [float(item) for item in state.get("joint_positions") or []]
        != sequence[-1]["joint_positions"]
    ):
        raise RuntimeError("persistent_isaac_controller_final_state_sequence_mismatch")
    return sequence, contract, True


def _standing_articulation_action(
    *, joint_positions: Any, joint_velocities: Any, joint_indices: Any
) -> Any:
    """Build Isaac's supported position-and-velocity target action."""

    from isaacsim.core.utils.types import ArticulationAction  # type: ignore

    return ArticulationAction(
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        joint_indices=joint_indices,
    )


def _protocol_v4_articulation_indices(live_joint_names: Any) -> list[int]:
    """Return canonical protocol indices into one exact live G1 inventory."""

    if isinstance(live_joint_names, (str, bytes, bytearray)):
        raise RuntimeError("persistent_isaac_protocol_v4_dof_inventory_invalid")
    try:
        names = [str(item) for item in live_joint_names]
        validate_model_joint_names(names, source="isaac_articulation")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("persistent_isaac_protocol_v4_dof_inventory_invalid") from exc
    index_by_name = {name: index for index, name in enumerate(names)}
    return [index_by_name[name] for name in PROTOCOL_V4_FULL_JOINT_ORDER]


def build_gear_sonic_isaac_state_snapshot(
    *,
    live_joint_names: Any,
    live_joint_positions: Any,
    live_joint_velocities: Any,
    base_quaternion_wxyz: Any,
    base_angular_velocity_xyz: Any,
    simulator_session_id: str,
    stage_id: str,
    heartbeat_sequence: int,
    captured_at_ns: int,
    source: str,
    source_action_sha256: str = "",
    source_step_index: int | None = None,
    snapshot_path: str = "",
    freshness_window_ns: int = GEAR_SONIC_ISAAC_STATE_FRESHNESS_WINDOW_NS,
) -> dict[str, Any]:
    """Build the exact state record consumed by the native Unitree DDS bridge."""

    names = [str(item) for item in live_joint_names]
    canonical_indices = _protocol_v4_articulation_indices(names)
    positions = _finite_float_vector(
        live_joint_positions,
        expected_length=len(names),
        error_code="persistent_isaac_state_snapshot_joint_positions_invalid",
    )
    velocities = _finite_float_vector(
        live_joint_velocities,
        expected_length=len(names),
        error_code="persistent_isaac_state_snapshot_joint_velocities_invalid",
    )
    quaternion = list(
        _normalized_quaternion_wxyz(
            base_quaternion_wxyz,
            error_code="persistent_isaac_state_snapshot_base_quaternion_invalid",
        )
    )
    angular_velocity = _finite_float_vector(
        base_angular_velocity_xyz,
        expected_length=3,
        error_code="persistent_isaac_state_snapshot_base_angular_velocity_invalid",
    )
    session = str(simulator_session_id or "").strip()
    stage = str(stage_id or "").strip()
    state_source = str(source or "").strip()
    try:
        sequence = int(heartbeat_sequence)
        timestamp = int(captured_at_ns)
        freshness = int(freshness_window_ns)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("persistent_isaac_state_snapshot_freshness_invalid") from exc
    if not session or not stage or not state_source or sequence < 1 or timestamp < 1 or freshness < 1:
        raise RuntimeError("persistent_isaac_state_snapshot_identity_invalid")
    if source_step_index is not None and int(source_step_index) < 0:
        raise RuntimeError("persistent_isaac_state_snapshot_source_step_invalid")
    canonical_positions = [positions[index] for index in canonical_indices]
    canonical_velocities = [velocities[index] for index in canonical_indices]
    body_count = len(PROTOCOL_V4_BODY_JOINT_NAMES)
    w, x, y, z = quaternion
    projected_gravity = [
        -(2.0 * x * z - 2.0 * y * w),
        -(2.0 * y * z + 2.0 * x * w),
        -(1.0 - 2.0 * x * x - 2.0 * y * y),
    ]
    accelerometer = [-9.81 * value for value in projected_gravity]
    payload: dict[str, Any] = {
        "schema_version": GEAR_SONIC_ISAAC_STATE_SNAPSHOT_SCHEMA_VERSION,
        "status": "live",
        "ready_for_native_dds_bridge": True,
        "simulator_session_id": session,
        "stage_id": stage,
        "source": "live_isaac_articulation",
        "capture_reason": state_source,
        "surrogate": False,
        "captured_at_ns": timestamp,
        "fresh_until_ns": timestamp + freshness,
        "freshness_window_ns": freshness,
        "heartbeat_sequence": sequence,
        "joint_order_schema_version": JOINT_ORDER_SCHEMA_VERSION,
        "mapping_digest": PROTOCOL_V4_MAPPING_DIGEST,
        "pinned_wbc_source_revision": PINNED_WBC_SOURCE_REVISION,
        "body_joint_names": list(PROTOCOL_V4_BODY_JOINT_NAMES),
        "body_q": canonical_positions[:body_count],
        "body_dq": canonical_velocities[:body_count],
        "body_joint_positions": canonical_positions[:body_count],
        "body_joint_velocities": canonical_velocities[:body_count],
        "left_hand_joint_names": list(PROTOCOL_V4_LEFT_HAND_JOINT_NAMES),
        "left_hand_q": canonical_positions[body_count : body_count + 7],
        "left_hand_dq": canonical_velocities[body_count : body_count + 7],
        "right_hand_joint_names": list(PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES),
        "right_hand_q": canonical_positions[body_count + 7 : body_count + 14],
        "right_hand_dq": canonical_velocities[body_count + 7 : body_count + 14],
        "base_quaternion_wxyz": quaternion,
        "base_angular_velocity": angular_velocity,
        "base_angular_velocity_xyz": angular_velocity,
        "accelerometer_mps2": accelerometer,
        "accelerometer_source": (
            "base_orientation_derived_specific_force_gravity_only_no_linear_acceleration"
        ),
        "name_order_metadata": {
            "live_articulation_dof_names": names,
            "protocol_v4_full_joint_names": list(PROTOCOL_V4_FULL_JOINT_ORDER),
            "protocol_v4_articulation_dof_indices": canonical_indices,
            "body_joint_count": body_count,
            "left_hand_joint_count": len(PROTOCOL_V4_LEFT_HAND_JOINT_NAMES),
            "right_hand_joint_count": len(PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES),
        },
        "claim_boundary": (
            "This is a fresh live Isaac articulation sample for DDS publication; "
            "it does not by itself prove controller readiness or task success."
        ),
    }
    if source_action_sha256:
        action_sha = str(source_action_sha256).strip().lower()
        if len(action_sha) != 64 or any(char not in "0123456789abcdef" for char in action_sha):
            raise RuntimeError("persistent_isaac_state_snapshot_source_action_sha256_invalid")
        payload["source_action_sha256"] = action_sha
    if source_step_index is not None:
        payload["source_step_index"] = int(source_step_index)
    if str(snapshot_path or "").strip():
        payload["snapshot_path"] = str(Path(snapshot_path).expanduser().resolve())
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> dict[str, str]:
    """Durably replace one JSON record without exposing a partial heartbeat."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _write_bytes_atomic(path: str | Path, payload: bytes) -> dict[str, Any]:
    """Durably replace one binary artifact without exposing a partial PNG."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _yaw_from_quaternion_wxyz(value: Any) -> float:
    w, x, y, z = _normalized_quaternion_wxyz(
        value,
        error_code="persistent_isaac_projection_root_quaternion_invalid",
    )
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    if not math.isfinite(yaw):
        raise RuntimeError("persistent_isaac_projection_root_yaw_invalid")
    return yaw


def bind_post_action_policy_state_measurement(
    state: Mapping[str, Any],
    *,
    simulator_session_id: str,
    stage_id: str,
    source_action_sha256: str,
    source_step_index: int,
    captured_at_ns: int,
    state_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind live policy proprioception to the action that produced it."""

    bound = dict(state)
    measurement = dict(bound.get("measurement") or {})
    measurement.update(
        {
            "simulator_session_id": str(simulator_session_id),
            "stage_id": str(stage_id),
            "source": "post_action_live_isaac_articulation",
            "source_action_sha256": str(source_action_sha256),
            "source_step_index": int(source_step_index),
            "captured_at_ns": str(int(captured_at_ns)),
            "surrogate": False,
        }
    )
    snapshot = dict(state_snapshot or {})
    if snapshot:
        measurement.update(
            {
                "snapshot_path": str(snapshot.get("snapshot_path") or ""),
                "state_snapshot_captured_at_ns": str(snapshot["captured_at_ns"]),
                "state_snapshot_heartbeat_sequence": snapshot["heartbeat_sequence"],
                "state_snapshot_payload_sha256": snapshot["payload_sha256"],
                "state_snapshot_fresh_until_ns": str(snapshot["fresh_until_ns"]),
                "state_snapshot_freshness_window_ns": str(snapshot["freshness_window_ns"]),
            }
        )
    bound["measurement"] = measurement
    return bound


def _physx_overlap_hit_prim_path(hit: Any) -> str:
    """Return the rigid-body or collider path from a PhysX overlap hit.

    Older test/runtime shims exposed overlap hits as mapping-like values, while
    Isaac 6 passes an ``OverlapHit`` object with ``rigid_body`` and
    ``collision`` attributes.  Keep the rigid-body-first selection used by the
    live-geometry filter, and reject undecodable hits instead of silently
    treating an incomplete overlap query as collision-free.
    """

    getter = getattr(hit, "get", None)
    for field in ("rigid_body", "collision"):
        value = getter(field) if callable(getter) else None
        if value is None or not str(value).strip():
            value = getattr(hit, field, None)
        path = str(value).strip() if value is not None else ""
        if not path:
            continue
        if not path.startswith("/"):
            raise RuntimeError("persistent_isaac_overlap_hit_prim_path_invalid")
        return path
    raise RuntimeError("persistent_isaac_overlap_hit_prim_path_missing")


@dataclass(frozen=True)
class RevoluteTaskJointBinding:
    """Attempt-bound mapping from a semantic affordance to its physics joint.

    The Palatial kitchen binds the task contract to the microwave door rigid
    body.  Its actual hinge is a child ``PhysicsRevoluteJoint`` and the asset is
    intentionally a regular rigid-body joint graph, not an articulation.  Keep
    those two paths separate instead of rewriting the contract or changing the
    authored physics by injecting an articulation root.
    """

    contracted_prim_path: str
    joint_prim_path: str
    body0_prim_path: str
    body1_prim_path: str
    axis: str
    lower_limit_degrees: float
    upper_limit_degrees: float
    measurement_convention: str = "upper_limit_minus_signed_angle_radians"

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def resolve_revolute_task_joint(
    stage: Any, *, contracted_prim_path: str
) -> RevoluteTaskJointBinding:
    """Resolve one exact rigid-body affordance to its authored hinge.

    No stage-wide fuzzy scan is allowed.  A body-bound task must have exactly
    one descendant revolute joint whose ``body1`` relationship targets that
    exact body.  A contract that already names a revolute joint is also
    accepted, but its body relationships remain mandatory.
    """

    from pxr import Usd, UsdPhysics  # type: ignore

    path = str(contracted_prim_path or "").strip()
    prim = stage.GetPrimAtPath(path)
    if not path.startswith("/") or not prim or not prim.IsValid():
        raise RuntimeError(f"persistent_isaac_task_prim_missing:{path}")

    if prim.IsA(UsdPhysics.RevoluteJoint):
        candidates = [prim]
    else:
        candidates = []
        for candidate in Usd.PrimRange(prim):
            if not candidate.IsA(UsdPhysics.RevoluteJoint):
                continue
            joint = UsdPhysics.Joint(candidate)
            body1_targets = [str(item) for item in joint.GetBody1Rel().GetTargets()]
            if body1_targets == [path]:
                candidates.append(candidate)
    if len(candidates) != 1:
        joined = ",".join(str(item.GetPath()) for item in candidates)
        raise RuntimeError(
            f"persistent_isaac_task_revolute_joint_resolution_not_unique:{path}:{joined}"
        )

    joint_prim = candidates[0]
    joint = UsdPhysics.RevoluteJoint(joint_prim)
    body0_targets = [str(item) for item in joint.GetBody0Rel().GetTargets()]
    body1_targets = [str(item) for item in joint.GetBody1Rel().GetTargets()]
    if len(body0_targets) != 1 or len(body1_targets) != 1:
        raise RuntimeError("persistent_isaac_task_joint_body_binding_invalid")
    if not prim.IsA(UsdPhysics.RevoluteJoint) and body1_targets[0] != path:
        raise RuntimeError("persistent_isaac_task_joint_body1_mismatch")
    for body_path in (*body0_targets, *body1_targets):
        body = stage.GetPrimAtPath(body_path)
        if not body or not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"persistent_isaac_task_joint_rigid_body_missing:{body_path}")

    axis = str(joint.GetAxisAttr().Get() or "")
    if axis not in {"X", "Y", "Z"}:
        raise RuntimeError("persistent_isaac_task_joint_axis_invalid")
    try:
        lower = float(joint.GetLowerLimitAttr().Get())
        upper = float(joint.GetUpperLimitAttr().Get())
    except (TypeError, ValueError) as exc:
        raise RuntimeError("persistent_isaac_task_joint_limits_invalid") from exc
    if not all(math.isfinite(item) for item in (lower, upper)) or lower >= upper:
        raise RuntimeError("persistent_isaac_task_joint_limits_invalid")

    return RevoluteTaskJointBinding(
        contracted_prim_path=path,
        joint_prim_path=str(joint_prim.GetPath()),
        body0_prim_path=body0_targets[0],
        body1_prim_path=body1_targets[0],
        axis=axis,
        lower_limit_degrees=lower,
        upper_limit_degrees=upper,
    )


def _normalized_quaternion_wxyz(
    values: Any, *, error_code: str
) -> tuple[float, float, float, float]:
    try:
        items = tuple(float(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(error_code) from exc
    if len(items) != 4 or not all(math.isfinite(item) for item in items):
        raise RuntimeError(error_code)
    norm = math.sqrt(sum(item * item for item in items))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError(error_code)
    return tuple(item / norm for item in items)  # type: ignore[return-value]


def _multiply_quaternion_wxyz(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def measure_revolute_joint_signed_angle_rad(
    stage: Any,
    binding: RevoluteTaskJointBinding,
    *,
    body_rotation_reader: Callable[[str], Any] | None = None,
) -> float:
    """Read a regular PhysX hinge coordinate from its two rigid-body poses.

    Production supplies a reader backed by the current PhysX rigid-body world
    transform.  The USD transform fallback exists only for static, offline
    inspection and tests.  In either case, the relative rotation between the
    two authored joint frames is the physical revolute coordinate; no drive
    target, fixture, or policy output is used as a surrogate.
    """

    from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore

    joint = UsdPhysics.RevoluteJoint.Get(stage, binding.joint_prim_path)
    if not joint or not joint.GetPrim().IsValid():
        raise RuntimeError("persistent_isaac_task_joint_disappeared")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default()) if body_rotation_reader is None else None

    def frame_rotation(body_path: str, local_rotation: Any) -> tuple[float, float, float, float]:
        body = stage.GetPrimAtPath(body_path)
        if not body or not body.IsValid():
            raise RuntimeError(f"persistent_isaac_task_joint_rigid_body_missing:{body_path}")
        if body_rotation_reader is None:
            from pxr import Gf  # type: ignore

            if cache is None:  # pragma: no cover - type narrowing
                raise RuntimeError("persistent_isaac_task_pose_reader_missing")
            world_quat = Gf.Transform(cache.GetLocalToWorldTransform(body)).GetRotation().GetQuat()
            body_rotation = (
                float(world_quat.GetReal()),
                *[float(item) for item in world_quat.GetImaginary()],
            )
        else:
            body_rotation = body_rotation_reader(body_path)
        body_quat = _normalized_quaternion_wxyz(
            body_rotation,
            error_code="persistent_isaac_task_body_rotation_invalid",
        )
        local_quat = _normalized_quaternion_wxyz(
            (
                float(local_rotation.GetReal()),
                *[float(item) for item in local_rotation.GetImaginary()],
            ),
            error_code="persistent_isaac_task_joint_local_rotation_invalid",
        )
        return _normalized_quaternion_wxyz(
            _multiply_quaternion_wxyz(body_quat, local_quat),
            error_code="persistent_isaac_task_joint_frame_rotation_invalid",
        )

    local0 = joint.GetLocalRot0Attr().Get()
    local1 = joint.GetLocalRot1Attr().Get()
    if local0 is None or local1 is None:
        raise RuntimeError("persistent_isaac_task_joint_local_rotation_missing")
    frame0 = frame_rotation(binding.body0_prim_path, local0)
    frame1 = frame_rotation(binding.body1_prim_path, local1)
    frame0_inverse = (frame0[0], -frame0[1], -frame0[2], -frame0[3])
    relative = _normalized_quaternion_wxyz(
        _multiply_quaternion_wxyz(frame0_inverse, frame1),
        error_code="persistent_isaac_task_joint_relative_rotation_invalid",
    )
    # q and -q encode the same rotation.  Canonicalize to the positive-real
    # hemisphere so equivalent PhysX quaternion signs yield one joint angle.
    if relative[0] < 0.0:
        relative = tuple(-item for item in relative)  # type: ignore[assignment]
    axis_index = {"X": 0, "Y": 1, "Z": 2}[binding.axis]
    real = relative[0]
    projected = relative[axis_index + 1]
    twist_norm = math.hypot(real, projected)
    if not math.isfinite(twist_norm) or twist_norm <= 1e-12:
        raise RuntimeError("persistent_isaac_task_joint_twist_invalid")
    angle = 2.0 * math.atan2(projected / twist_norm, real / twist_norm)
    angle = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if not math.isfinite(angle):
        raise RuntimeError("persistent_isaac_task_joint_position_nonfinite")
    return angle


def measurement_backend_source_sha256() -> str:
    """Bind task measurements to the exact backend source loaded by Isaac."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_robot_start_pose(route_file: str | Path) -> tuple[list[float], float]:
    """Load the attempt-bound stance used by the proven kitchen runner."""
    path = Path(route_file).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = list(payload.get("route_points") or [])
    if not points or len(points[0]) != 3:
        raise RuntimeError("persistent_isaac_route_start_pose_missing")
    pose = [float(value) for value in points[0]]
    yaw = float(payload.get("accepted_stance_yaw_rad"))
    if not all(math.isfinite(value) for value in [*pose, yaw]):
        raise RuntimeError("persistent_isaac_route_start_pose_invalid")
    return pose, yaw


def compose_g1_for_episode(
    stage: Any,
    *,
    robot_prim_path: str,
    g1_usd_path: str | Path,
    route_file: str | Path,
) -> dict[str, Any]:
    """Compose and place G1 exactly once when the raw kitchen lacks it."""
    from pxr import Gf, UsdGeom, UsdPhysics  # type: ignore

    try:
        from pxr import PhysxSchema  # type: ignore
    except ImportError:
        # usd-core supports CPU composition tests but does not ship NVIDIA's
        # PhysX schemas. The live backend checks this flag and fails before
        # simulation; only offline scene composition may continue without it.
        PhysxSchema = None

    asset = Path(g1_usd_path).expanduser().resolve()
    if not asset.is_file():
        raise RuntimeError("persistent_isaac_g1_asset_missing")
    existing = stage.GetPrimAtPath(robot_prim_path)
    existing_valid = bool(existing and existing.IsValid())
    if not existing_valid:
        robot = stage.DefinePrim(robot_prim_path, "Xform")
        robot.GetReferences().AddReference(str(asset))
        stage.Load(robot_prim_path)
    robot = stage.GetPrimAtPath(robot_prim_path)
    if not robot or not robot.IsValid():
        raise RuntimeError("persistent_isaac_g1_composition_failed")

    pose, yaw = load_robot_start_pose(route_file)
    xform = UsdGeom.Xformable(robot)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pose))
    xform.AddRotateZOp().Set(math.degrees(yaw))

    articulation_roots = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(robot_prim_path)
        and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if not articulation_roots:
        raise RuntimeError("persistent_isaac_g1_articulation_missing_after_composition")
    contact_report_roots: list[str] = []
    if PhysxSchema is not None:
        for root_path in articulation_roots:
            root_prim = stage.GetPrimAtPath(root_path)
            if root_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                report_api = PhysxSchema.PhysxContactReportAPI(root_prim)
            else:
                report_api = PhysxSchema.PhysxContactReportAPI.Apply(root_prim)
            if not report_api:
                raise RuntimeError(
                    f"persistent_isaac_contact_report_api_apply_failed:{root_path}"
                )
            report_api.CreateThresholdAttr(0.0)
            contact_report_roots.append(root_path)
    return {
        "schema_version": "persistent_isaac_g1_composition.v1",
        "status": "passed",
        "robot_prim_path": robot_prim_path,
        "g1_usd_path": str(asset),
        "route_file": str(Path(route_file).expanduser().resolve()),
        "start_pose_xyz": pose,
        "start_yaw_rad": yaw,
        "robot_was_already_present": existing_valid,
        "articulation_root_paths": articulation_roots,
        "contact_report_articulation_root_paths": contact_report_roots,
        "contact_report_schema_available": PhysxSchema is not None,
        "contact_report_impulse_threshold": 0.0,
        "claim_boundary": (
            "Composition proves a controllable G1 is present at the attempt-bound stance; "
            "it does not prove policy actions or task success."
        ),
    }


class IsaacPersistentTaskBackend:
    def __init__(
        self,
        *,
        stage_path: str,
        robot_prim_path: str,
        evidence_dir: str | Path,
        g1_usd_path: str | Path,
        route_file: str | Path,
        state_snapshot_path: str | Path | None = None,
        headless: bool = True,
    ) -> None:
        from isaacsim import SimulationApp  # type: ignore

        self.app = SimulationApp(
            {
                "headless": bool(headless),
                "renderer": "RayTracedLighting",
                "anti_aliasing": 3,
            }
        )
        import omni.timeline  # type: ignore
        import omni.usd  # type: ignore

        self.timeline = omni.timeline.get_timeline_interface()
        self.stage_path = str(Path(stage_path).expanduser().resolve())
        self.robot_prim_path = str(robot_prim_path)
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = f"isaac-task-session-{uuid.uuid4().hex}"
        self.episode_baseline: dict[str, Any] | None = None
        self.episode_baseline_attestation: dict[str, Any] | None = None
        self.live_geometry_results: dict[str, dict[str, Any]] = {}
        self._task_joint_bindings: dict[str, RevoluteTaskJointBinding] = {}
        self.measurement_backend_source_sha256 = measurement_backend_source_sha256()
        self.attempt_id = ""
        self.launch_nonce = ""
        self.allocation_launch_session_id = ""
        self.qualification_attempt_bound = False
        self.qualification_attempt_sequence: int | None = None
        self.qualification_attempt_nonce_sha256: str | None = None
        self.review_execution_frame_index = 0
        self.controller_global_frame_index = 0
        self.stage_id = hashlib.sha256(Path(self.stage_path).read_bytes()).hexdigest()
        omni.usd.get_context().open_stage(self.stage_path)
        self.stage = omni.usd.get_context().get_stage()
        self.robot_composition = compose_g1_for_episode(
            self.stage,
            robot_prim_path=self.robot_prim_path,
            g1_usd_path=g1_usd_path,
            route_file=route_file,
        )
        if (
            self.robot_composition.get("contact_report_schema_available") is not True
            or not self.robot_composition.get("contact_report_articulation_root_paths")
        ):
            raise RuntimeError("persistent_isaac_contact_report_schema_unavailable")
        self._contact_events: list[dict[str, Any]] = []
        self._contact_report_error: str | None = None
        self._contact_report_subscription = self._subscribe_contact_reports()
        # SONIC/GEAR emits control targets at 50 Hz. Configure both independent
        # Isaac clocks while stopped, then fail closed unless their readbacks
        # prove one coherent 0.02 s physics/render/controller tick.
        from isaacsim.core.simulation_manager import SimulationManager  # type: ignore

        self._simulation_manager = SimulationManager
        self.simulation_control_timing = configure_and_verify_simulation_control_clock(
            stage=self.stage,
            timeline=self.timeline,
            app=self.app,
            simulation_manager=SimulationManager,
        )
        self.robot_composition["simulation_control_timing"] = dict(
            self.simulation_control_timing
        )
        (self.evidence_dir / "robot_stage_composition.json").write_text(
            json.dumps(self.robot_composition, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        configured_snapshot_path = (
            str(state_snapshot_path or "").strip()
            or str(os.environ.get(GEAR_SONIC_ISAAC_STATE_SNAPSHOT_ENV) or "").strip()
            or GEAR_SONIC_ISAAC_STATE_SNAPSHOT_DEFAULT_PATH
        )
        self.live_state_snapshot_path = Path(configured_snapshot_path).expanduser().resolve()
        self._live_state_snapshot_sequence = 0
        self._articulations: dict[str, Any] = {}
        from .isaac_task_review_renderer import IsaacTaskReviewRenderer

        self.review_renderer = IsaacTaskReviewRenderer(
            stage=self.stage,
            app=self.app,
            robot_prim_path=self.robot_prim_path,
            output_dir=self.evidence_dir,
            heartbeat_callback=None,
        )
        self.review_renderer_articulation_lifecycle = (
            self._prewarm_review_renderer_and_initialize_robot()
        )

    def _subscribe_contact_reports(self) -> Any:
        """Subscribe before simulation so every robot contact is evidence-bound."""

        import omni.physx  # type: ignore
        from pxr import PhysicsSchemaTools  # type: ignore

        def callback(contact_headers: Sequence[Any], contact_data: Sequence[Any]) -> None:
            try:
                self._contact_events.extend(
                    normalize_physx_contact_reports(
                        contact_headers,
                        contact_data,
                        path_decoder=PhysicsSchemaTools.intToSdfPath,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - fail closed at readback boundary
                self._contact_report_error = f"{type(exc).__name__}:{exc}"

        interface = omni.physx.get_physx_simulation_interface()
        subscription = interface.subscribe_contact_report_events(callback)
        if subscription is None:
            raise RuntimeError("persistent_isaac_contact_report_subscription_failed")
        return subscription

    def contact_event_cursor(self) -> int:
        if self._contact_report_error:
            raise RuntimeError(
                "persistent_isaac_contact_report_callback_failed:"
                f"{self._contact_report_error}"
            )
        return len(self._contact_events)

    def contact_events_since(self, cursor: int) -> list[dict[str, Any]]:
        if self._contact_report_error:
            raise RuntimeError(
                "persistent_isaac_contact_report_callback_failed:"
                f"{self._contact_report_error}"
            )
        start = int(cursor)
        if start < 0 or start > len(self._contact_events):
            raise RuntimeError("persistent_isaac_contact_event_cursor_invalid")
        return [dict(item) for item in self._contact_events[start:]]

    def measure_revolute_task_open_angle(self, prim_path: str) -> dict[str, Any]:
        criterion = {
            "criterion_id": "microwave_door_open_angle",
            "observable_transition": "articulation_angle_rad",
            "articulation_prim_path": str(prim_path),
            "comparison": "increase_at_least",
            "unit": "rad",
        }
        return self._task_joint_sample(
            self._task_joint_binding(criterion),
            criterion,
        )

    def _prewarm_review_renderer_and_initialize_robot(self) -> dict[str, Any]:
        """Finish lazy render setup before creating the robot tensor view.

        Replicator may edit the stage while realizing a render product.  Any
        such edit beneath an already-initialized articulation invalidates its
        shared PhysX tensor ``SimulationView``.  This lifecycle runs once,
        before a signed episode baseline can exist, and intentionally offers no
        reset/reinitialize path after the baseline.
        """

        artifact_path = (
            self.evidence_dir / "review_renderer_articulation_lifecycle.json"
        )
        started_at_ns = time.time_ns()
        observations: dict[str, Any] = {
            "timeline_stopped_during_prewarm": False,
            "physics_step_count_before_prewarm": None,
            "physics_step_count_after_prewarm": None,
            "physics_step_delta_during_prewarm": None,
            "physics_step_counter_reset_during_prewarm": False,
            "physics_steps_advanced_during_prewarm": None,
            "articulation_count_before_prewarm": len(self._articulations),
            "no_articulation_tensor_view_before_prewarm": False,
            "renderer_prewarm": None,
            "articulation_initialized_after_prewarm": False,
            "standing_applied_and_verified_after_articulation_initialize": False,
            "manipulation_ready_pose_applied_after_standing_verification": False,
            "initial_live_state_snapshot_written_after_manipulation_ready_pose": False,
            "heartbeat_attached_after_manipulation_ready_verification": False,
            "heartbeat_callback_attached_during_prewarm": None,
        }
        common: dict[str, Any] = {
            "schema_version": (
                ISAAC_REVIEW_RENDERER_ARTICULATION_LIFECYCLE_SCHEMA_VERSION
            ),
            "simulator_session_id": self.session_id,
            "stage_id": self.stage_id,
            "robot_prim_path": self.robot_prim_path,
            "started_at_ns": started_at_ns,
            "signed_episode_baseline_present_during_lifecycle": bool(
                getattr(self, "episode_baseline", None)
                or getattr(self, "episode_baseline_attestation", None)
            ),
            "signed_episode_baseline_must_follow_lifecycle": True,
            "post_baseline_reset_or_reinitialize_supported": False,
            "startup_update_count_before_articulation_initialize": (
                ISAAC_ARTICULATION_STARTUP_UPDATE_COUNT
            ),
            "claim_boundary": (
                "This proves Replicator render products were realized while physics "
                "was stopped and before the G1 SingleArticulation tensor view was "
                "created, then official standing state, the reviewed late-June "
                "right-arm manipulation-ready seed, and heartbeat attachment "
                "completed in that order. It does not prove policy execution or "
                "task success."
            ),
        }
        try:
            if common["signed_episode_baseline_present_during_lifecycle"]:
                raise RuntimeError(
                    "persistent_isaac_renderer_lifecycle_after_episode_baseline_forbidden"
                )
            if self._articulations or hasattr(self, "robot"):
                raise RuntimeError(
                    "persistent_isaac_articulation_exists_before_renderer_prewarm"
                )
            observations["no_articulation_tensor_view_before_prewarm"] = True
            is_playing = getattr(self.timeline, "is_playing", None)
            if not callable(is_playing):
                raise RuntimeError("persistent_isaac_timeline_state_api_missing")
            if bool(is_playing()):
                raise RuntimeError(
                    "persistent_isaac_timeline_playing_before_renderer_prewarm"
                )
            observations["timeline_stopped_during_prewarm"] = True
            physics_steps_before = int(
                self._simulation_manager.get_num_physics_steps()
            )
            observations["physics_step_count_before_prewarm"] = physics_steps_before
            renderer_prewarm = self.review_renderer.prewarm()
            observations["renderer_prewarm"] = dict(renderer_prewarm)
            observations["heartbeat_callback_attached_during_prewarm"] = bool(
                renderer_prewarm.get("heartbeat_callback_attached_during_prewarm")
            )
            physics_steps_after = int(
                self._simulation_manager.get_num_physics_steps()
            )
            physics_step_delta = physics_steps_after - physics_steps_before
            observations["physics_step_count_after_prewarm"] = physics_steps_after
            observations["physics_step_delta_during_prewarm"] = physics_step_delta
            # Realizing the Replicator graph can invalidate and recreate the
            # not-yet-bound PhysX scene, which resets this process-global
            # counter (attempt 029 observed 3 -> 0).  A reset before any
            # articulation or signed baseline is safe; a positive delta is an
            # actual physics advance and remains forbidden.  The renderer's
            # explicit capture also binds ``delta_time=0.0`` and waits for the
            # completed frame.
            observations["physics_step_counter_reset_during_prewarm"] = (
                physics_step_delta < 0
            )
            observations["physics_steps_advanced_during_prewarm"] = (
                physics_step_delta > 0
            )
            if physics_step_delta > 0 or bool(is_playing()):
                raise RuntimeError(
                    "persistent_isaac_renderer_prewarm_advanced_physics"
                )
            if renderer_prewarm.get("status") != "passed":
                raise RuntimeError("persistent_isaac_renderer_prewarm_not_passed")
            if observations["heartbeat_callback_attached_during_prewarm"]:
                raise RuntimeError(
                    "persistent_isaac_renderer_prewarm_heartbeat_attached"
                )
            if self.review_renderer.heartbeat_callback is not None:
                raise RuntimeError(
                    "persistent_isaac_renderer_heartbeat_attached_before_articulation"
                )

            self.timeline.play()
            self.timeline.commit()
            for _ in range(ISAAC_ARTICULATION_STARTUP_UPDATE_COUNT):
                self.app.update()
            self.robot = self._articulation(self.robot_prim_path)
            if not bool(getattr(self.robot, "handles_initialized", False)):
                raise RuntimeError("persistent_isaac_robot_articulation_not_found")
            observations["articulation_initialized_after_prewarm"] = True
            self.standing_initialization = (
                self._initialize_official_gear_sonic_standing_pose()
            )
            if self.standing_initialization.get("status") != "passed":
                raise RuntimeError(
                    "persistent_isaac_standing_initialization_not_passed"
                )
            observations[
                "standing_applied_and_verified_after_articulation_initialize"
            ] = True
            self.manipulation_ready_initialization = (
                self._initialize_right_arm_manipulation_ready_pose()
            )
            if self.manipulation_ready_initialization.get("status") != "passed":
                raise RuntimeError(
                    "persistent_isaac_manipulation_ready_initialization_not_passed"
                )
            observations[
                "manipulation_ready_pose_applied_after_standing_verification"
            ] = True
            self.initial_live_state_snapshot = self._write_live_state_snapshot(
                source="initial_manipulation_ready_pose_live_isaac_articulation",
            )
            if not isinstance(self.initial_live_state_snapshot, Mapping):
                raise RuntimeError(
                    "persistent_isaac_initial_live_state_snapshot_missing"
                )
            observations[
                "initial_live_state_snapshot_written_after_manipulation_ready_pose"
            ] = True
            renderer_contract = self.review_renderer.attach_heartbeat_callback(
                self._refresh_live_state_if_configured
            )
            if (
                renderer_contract.get(
                    "heartbeat_callback_attached_after_prewarm"
                )
                is not True
            ):
                raise RuntimeError(
                    "persistent_isaac_renderer_heartbeat_attachment_unproven"
                )
            observations[
                "heartbeat_attached_after_manipulation_ready_verification"
            ] = True
            evidence = {
                **common,
                **observations,
                "status": "passed",
                "completed_at_ns": time.time_ns(),
                "renderer_contract_after_heartbeat_attachment": renderer_contract,
                "blockers": [],
            }
            artifact = _write_json_atomic(artifact_path, evidence)
            self.review_renderer_articulation_lifecycle_artifact = artifact
            return evidence
        except Exception as exc:
            if observations["physics_step_count_before_prewarm"] is not None:
                try:
                    physics_steps_after = int(
                        self._simulation_manager.get_num_physics_steps()
                    )
                    observations["physics_step_count_after_prewarm"] = (
                        physics_steps_after
                    )
                    observations["physics_step_delta_during_prewarm"] = (
                        physics_steps_after
                        - int(observations["physics_step_count_before_prewarm"])
                    )
                except Exception:  # noqa: BLE001 - preserve the root blocker
                    pass
            blocked = {
                **common,
                **observations,
                "status": "blocked",
                "completed_at_ns": time.time_ns(),
                "blockers": [f"{type(exc).__name__}:{exc}"],
            }
            self.review_renderer_articulation_lifecycle = blocked
            self.review_renderer_articulation_lifecycle_artifact = _write_json_atomic(
                artifact_path, blocked
            )
            raise

    def _articulation(self, prim_path: str):
        cached = self._articulations.get(prim_path)
        if cached is not None:
            return cached
        if (
            getattr(self, "episode_baseline", None) is not None
            or getattr(self, "episode_baseline_attestation", None) is not None
        ):
            raise RuntimeError(
                "persistent_isaac_articulation_initialize_after_episode_baseline_forbidden"
            )
        try:
            renderer_prewarm = self.review_renderer.prewarm_contract()
        except Exception as exc:
            raise RuntimeError(
                "persistent_isaac_articulation_before_review_renderer_prewarm"
            ) from exc
        if renderer_prewarm.get("status") != "passed":
            raise RuntimeError(
                "persistent_isaac_articulation_before_review_renderer_prewarm"
            )
        if self.review_renderer.heartbeat_callback is not None:
            raise RuntimeError(
                "persistent_isaac_articulation_after_renderer_heartbeat_attachment"
            )
        from isaacsim.core.prims import SingleArticulation  # type: ignore

        articulation = SingleArticulation(
            prim_path=prim_path,
            name=f"blueprint_articulation_{len(self._articulations)}",
        )
        articulation.initialize()
        if not bool(getattr(articulation, "handles_initialized", False)):
            raise RuntimeError(f"persistent_isaac_articulation_not_initialized:{prim_path}")
        self._articulations[prim_path] = articulation
        return articulation

    def _live_robot_joint_state(
        self,
    ) -> tuple[list[str], list[float], list[float], list[int]]:
        names = [str(item) for item in (self.robot.dof_names or [])]
        canonical_indices = _protocol_v4_articulation_indices(names)
        positions = _finite_float_vector(
            self.robot.get_joint_positions(),
            expected_length=len(names),
            error_code="persistent_isaac_live_joint_positions_invalid",
        )
        velocities = _finite_float_vector(
            self.robot.get_joint_velocities(),
            expected_length=len(names),
            error_code="persistent_isaac_live_joint_velocities_invalid",
        )
        return names, positions, velocities, canonical_indices

    def _initialize_official_gear_sonic_standing_pose(self) -> dict[str, Any]:
        """Reset one live G1 to the exact official GEAR standing posture."""

        artifact_path = self.evidence_dir / "gear_sonic_standing_initialization.json"
        started_at_ns = time.time_ns()
        joint_state_application: dict[str, Any] = {
            "mode": "not_applied",
            "required_state_apis": [
                "set_joint_positions",
                "set_joint_velocities",
            ],
            "required_state_apis_available": False,
            "optional_target_apis_available": [],
            "applied_apis": [],
            "protocol_v4_articulation_dof_indices": [],
            "target_joint_count": len(PROTOCOL_V4_FULL_JOINT_ORDER),
            "standing_pose_sha256": GEAR_SONIC_DEFAULT_STANDING_POSE_SHA256,
            "surrogate": False,
        }
        common: dict[str, Any] = {
            "schema_version": GEAR_SONIC_STANDING_INITIALIZATION_SCHEMA_VERSION,
            "simulator_session_id": self.session_id,
            "stage_id": self.stage_id,
            "robot_prim_path": self.robot_prim_path,
            "joint_order_schema_version": JOINT_ORDER_SCHEMA_VERSION,
            "mapping_digest": PROTOCOL_V4_MAPPING_DIGEST,
            "pinned_wbc_source_revision": PINNED_WBC_SOURCE_REVISION,
            "standing_pose_sha256": GEAR_SONIC_DEFAULT_STANDING_POSE_SHA256,
            "body_joint_names": list(PROTOCOL_V4_BODY_JOINT_NAMES),
            "body_joint_target_positions": list(GEAR_SONIC_DEFAULT_BODY_STANDING_POSITIONS),
            "left_hand_joint_names": list(PROTOCOL_V4_LEFT_HAND_JOINT_NAMES),
            "right_hand_joint_names": list(PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES),
            "hand_target_position": 0.0,
            "target_joint_velocity": 0.0,
            "started_at_ns": started_at_ns,
            "source": "official_gear_sonic_policy_parameters_default_body_angles",
            "surrogate": False,
            "claim_boundary": (
                "This proves the live Isaac G1 was initialized and measured in the "
                "official standing posture; it does not prove DDS or controller readiness."
            ),
        }
        try:
            names = [str(item) for item in (self.robot.dof_names or [])]
            canonical_indices = _protocol_v4_articulation_indices(names)
            import numpy as np

            target_positions = np.asarray(
                GEAR_SONIC_DEFAULT_FULL_STANDING_POSITIONS,
                dtype=np.float32,
            )
            zero_velocities = np.zeros(len(PROTOCOL_V4_FULL_JOINT_ORDER), dtype=np.float32)
            joint_indices = np.asarray(canonical_indices, dtype=np.int64)
            start_xyz = _finite_float_vector(
                self.robot_composition.get("start_pose_xyz"),
                expected_length=3,
                error_code="persistent_isaac_standing_start_pose_invalid",
            )
            start_yaw = float(self.robot_composition.get("start_yaw_rad"))
            if not math.isfinite(start_yaw):
                raise RuntimeError("persistent_isaac_standing_start_pose_invalid")
            start_quaternion_wxyz = [
                math.cos(start_yaw / 2.0),
                0.0,
                0.0,
                math.sin(start_yaw / 2.0),
            ]
            set_joint_positions = getattr(self.robot, "set_joint_positions", None)
            set_joint_velocities = getattr(self.robot, "set_joint_velocities", None)
            set_joint_position_targets = getattr(
                self.robot, "set_joint_position_targets", None
            )
            set_joint_velocity_targets = getattr(
                self.robot, "set_joint_velocity_targets", None
            )
            apply_action = getattr(self.robot, "apply_action", None)
            available_target_apis = [
                api_name
                for api_name, api in (
                    ("apply_action", apply_action),
                    ("set_joint_position_targets", set_joint_position_targets),
                    ("set_joint_velocity_targets", set_joint_velocity_targets),
                )
                if callable(api)
            ]
            joint_state_application.update(
                {
                    "required_state_apis_available": bool(
                        callable(set_joint_positions)
                        and callable(set_joint_velocities)
                    ),
                    "optional_target_apis_available": available_target_apis,
                    "protocol_v4_articulation_dof_indices": canonical_indices,
                }
            )
            if not callable(set_joint_positions) or not callable(set_joint_velocities):
                raise RuntimeError("persistent_isaac_standing_joint_state_api_missing")
            self.robot.set_world_pose(
                position=np.asarray(start_xyz, dtype=np.float32),
                orientation=np.asarray(start_quaternion_wxyz, dtype=np.float32),
            )
            set_joint_positions(
                target_positions,
                joint_indices=joint_indices,
            )
            set_joint_velocities(
                zero_velocities,
                joint_indices=joint_indices,
            )
            applied_apis = ["set_joint_positions", "set_joint_velocities"]
            joint_state_application.update(
                {
                    "mode": "state_setters_applied_target_pending",
                    "applied_apis": applied_apis,
                }
            )
            if callable(apply_action):
                standing_action = _standing_articulation_action(
                    joint_positions=target_positions,
                    joint_velocities=zero_velocities,
                    joint_indices=joint_indices,
                )
                apply_action(standing_action)
                applied_apis.append("apply_action")
                application_mode = "state_plus_articulation_action"
                target_binding = {
                    "api": "apply_action",
                    "action_type": "isaacsim.core.utils.types.ArticulationAction",
                    "joint_positions_bound": True,
                    "joint_velocities_bound": True,
                    "joint_indices_bound": True,
                    "surrogate": False,
                }
            else:
                if callable(set_joint_position_targets):
                    set_joint_position_targets(
                        target_positions,
                        joint_indices=joint_indices,
                    )
                    applied_apis.append("set_joint_position_targets")
                if callable(set_joint_velocity_targets):
                    set_joint_velocity_targets(
                        zero_velocities,
                        joint_indices=joint_indices,
                    )
                    applied_apis.append("set_joint_velocity_targets")
                legacy_target_apis = tuple(
                    api
                    for api in available_target_apis
                    if api != "apply_action"
                )
                application_mode = {
                    (): "state_setters_only",
                    ("set_joint_position_targets",): "state_plus_position_targets",
                    ("set_joint_velocity_targets",): "state_plus_velocity_targets",
                    (
                        "set_joint_position_targets",
                        "set_joint_velocity_targets",
                    ): "state_plus_position_velocity_targets",
                }[legacy_target_apis]
                target_binding = {
                    "api": "legacy_target_setters" if legacy_target_apis else None,
                    "applied_target_apis": list(legacy_target_apis),
                    "joint_positions_bound": "set_joint_position_targets"
                    in legacy_target_apis,
                    "joint_velocities_bound": "set_joint_velocity_targets"
                    in legacy_target_apis,
                    "joint_indices_bound": bool(legacy_target_apis),
                    "surrogate": False,
                }
            joint_state_application.update(
                {
                    "mode": application_mode,
                    "applied_apis": applied_apis,
                    "target_binding": target_binding,
                }
            )
            set_linear_velocity = getattr(self.robot, "set_linear_velocity", None)
            set_angular_velocity = getattr(self.robot, "set_angular_velocity", None)
            if not callable(set_linear_velocity) or not callable(set_angular_velocity):
                raise RuntimeError("persistent_isaac_standing_base_velocity_api_missing")
            set_linear_velocity(np.zeros(3, dtype=np.float32))
            set_angular_velocity(np.zeros(3, dtype=np.float32))
            # Commit the reset to PhysX before accepting measurements from it.
            self.app.update()
            (
                measured_names,
                measured_positions,
                measured_velocities,
                measured_indices,
            ) = self._live_robot_joint_state()
            canonical_measured_positions = [measured_positions[index] for index in measured_indices]
            canonical_measured_velocities = [
                measured_velocities[index] for index in measured_indices
            ]
            position_errors = [
                abs(actual - expected)
                for actual, expected in zip(
                    canonical_measured_positions,
                    GEAR_SONIC_DEFAULT_FULL_STANDING_POSITIONS,
                    strict=True,
                )
            ]
            velocity_errors = [abs(value) for value in canonical_measured_velocities]
            if max(position_errors, default=math.inf) > (
                GEAR_SONIC_STANDING_JOINT_POSITION_TOLERANCE_RAD
            ):
                raise RuntimeError("persistent_isaac_standing_joint_position_verification_failed")
            if max(velocity_errors, default=math.inf) > (
                GEAR_SONIC_STANDING_JOINT_VELOCITY_TOLERANCE_RAD_S
            ):
                raise RuntimeError("persistent_isaac_standing_joint_velocity_verification_failed")
            projected_gravity = self._live_projected_gravity()
            if (
                abs(projected_gravity[0]) > 0.2
                or abs(projected_gravity[1]) > 0.2
                or projected_gravity[2] > -0.9
            ):
                raise RuntimeError("persistent_isaac_standing_base_orientation_verification_failed")
            completed_at_ns = time.time_ns()
            evidence = {
                **common,
                "status": "passed",
                "completed_at_ns": completed_at_ns,
                "live_articulation_dof_names": measured_names,
                "protocol_v4_articulation_dof_indices": measured_indices,
                "measured_full_joint_positions": canonical_measured_positions,
                "measured_full_joint_velocities": canonical_measured_velocities,
                "maximum_joint_position_error_rad": max(position_errors),
                "maximum_joint_velocity_rad_s": max(velocity_errors),
                "joint_position_tolerance_rad": (GEAR_SONIC_STANDING_JOINT_POSITION_TOLERANCE_RAD),
                "joint_velocity_tolerance_rad_s": (
                    GEAR_SONIC_STANDING_JOINT_VELOCITY_TOLERANCE_RAD_S
                ),
                "start_pose_xyz": start_xyz,
                "start_quaternion_wxyz": start_quaternion_wxyz,
                "projected_gravity": projected_gravity,
                "joint_state_application": dict(joint_state_application),
                "blockers": [],
            }
            artifact = _write_json_atomic(artifact_path, evidence)
            self.standing_initialization_artifact = artifact
            return evidence
        except Exception as exc:  # noqa: BLE001 - evidence then fail closed
            blocked = {
                **common,
                "status": "blocked",
                "completed_at_ns": time.time_ns(),
                "joint_state_application": dict(joint_state_application),
                "blockers": [
                    f"persistent_isaac_standing_initialization_failed:{type(exc).__name__}:{exc}"
                ],
            }
            _write_json_atomic(artifact_path, blocked)
            raise RuntimeError(
                f"persistent_isaac_standing_initialization_failed:{type(exc).__name__}:{exc}"
            ) from exc

    def _initialize_right_arm_manipulation_ready_pose(self) -> dict[str, Any]:
        """Raise the task-side arm while preserving separate standing proof."""

        artifact_path = (
            self.evidence_dir / "gear_sonic_manipulation_ready_initialization.json"
        )
        started_at_ns = time.time_ns()
        common: dict[str, Any] = {
            "schema_version": (
                GEAR_SONIC_MANIPULATION_READY_INITIALIZATION_SCHEMA_VERSION
            ),
            "simulator_session_id": self.session_id,
            "stage_id": self.stage_id,
            "robot_prim_path": self.robot_prim_path,
            "joint_order_schema_version": JOINT_ORDER_SCHEMA_VERSION,
            "mapping_digest": PROTOCOL_V4_MAPPING_DIGEST,
            "source_standing_pose_sha256": GEAR_SONIC_DEFAULT_STANDING_POSE_SHA256,
            "manipulation_ready_pose_sha256": (
                GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_POSE_SHA256
            ),
            "task_side": "right",
            "joint_position_deltas_rad": dict(
                GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_DELTAS_RAD
            ),
            "target_full_joint_positions": list(
                GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_POSITIONS
            ),
            "started_at_ns": started_at_ns,
            "surrogate": False,
            "claim_boundary": (
                "This proves the live Isaac G1 entered the reviewed late-June "
                "right-arm manipulation-ready seed after separate official-standing "
                "verification. It proves neither learned-policy competence nor task "
                "success."
            ),
        }
        measured_diagnostics: dict[str, Any] = {}
        try:
            names = [str(item) for item in (self.robot.dof_names or [])]
            canonical_indices = _protocol_v4_articulation_indices(names)
            import numpy as np

            target_positions = np.asarray(
                GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_POSITIONS,
                dtype=np.float32,
            )
            zero_velocities = np.zeros(
                len(PROTOCOL_V4_FULL_JOINT_ORDER), dtype=np.float32
            )
            joint_indices = np.asarray(canonical_indices, dtype=np.int64)
            set_joint_positions = getattr(self.robot, "set_joint_positions", None)
            set_joint_velocities = getattr(self.robot, "set_joint_velocities", None)
            if not callable(set_joint_positions) or not callable(set_joint_velocities):
                raise RuntimeError(
                    "persistent_isaac_manipulation_ready_joint_state_api_missing"
                )
            set_joint_positions(target_positions, joint_indices=joint_indices)
            set_joint_velocities(zero_velocities, joint_indices=joint_indices)
            apply_action = getattr(self.robot, "apply_action", None)
            set_joint_position_targets = getattr(
                self.robot, "set_joint_position_targets", None
            )
            set_joint_velocity_targets = getattr(
                self.robot, "set_joint_velocity_targets", None
            )
            applied_apis = ["set_joint_positions", "set_joint_velocities"]
            if callable(apply_action):
                apply_action(
                    _standing_articulation_action(
                        joint_positions=target_positions,
                        joint_velocities=zero_velocities,
                        joint_indices=joint_indices,
                    )
                )
                applied_apis.append("apply_action")
            else:
                if callable(set_joint_position_targets):
                    set_joint_position_targets(
                        target_positions, joint_indices=joint_indices
                    )
                    applied_apis.append("set_joint_position_targets")
                if callable(set_joint_velocity_targets):
                    set_joint_velocity_targets(
                        zero_velocities, joint_indices=joint_indices
                    )
                    applied_apis.append("set_joint_velocity_targets")
            standing_by_name = dict(
                zip(
                    PROTOCOL_V4_FULL_JOINT_ORDER,
                    GEAR_SONIC_DEFAULT_FULL_STANDING_POSITIONS,
                    strict=True,
                )
            )
            insufficient_delta_joints: list[str] = []
            velocity_errors: list[float] = []
            for settle_step in range(
                1, GEAR_SONIC_MANIPULATION_READY_MAX_SETTLE_STEPS + 1
            ):
                # Mirror the validated late-June dynamic-standing path: keep
                # the articulation target asserted while PhysX resolves the
                # one-frame teleport transient instead of judging that first
                # high-velocity readback as the settled arm pose.
                if callable(apply_action):
                    apply_action(
                        _standing_articulation_action(
                            joint_positions=target_positions,
                            joint_velocities=zero_velocities,
                            joint_indices=joint_indices,
                        )
                    )
                else:
                    if callable(set_joint_position_targets):
                        set_joint_position_targets(
                            target_positions, joint_indices=joint_indices
                        )
                    if callable(set_joint_velocity_targets):
                        set_joint_velocity_targets(
                            zero_velocities, joint_indices=joint_indices
                        )
                self.app.update()
                (
                    measured_names,
                    measured_positions,
                    measured_velocities,
                    measured_indices,
                ) = self._live_robot_joint_state()
                canonical_positions = [
                    measured_positions[index] for index in measured_indices
                ]
                canonical_velocities = [
                    measured_velocities[index] for index in measured_indices
                ]
                position_errors = [
                    abs(actual - expected)
                    for actual, expected in zip(
                        canonical_positions,
                        GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_POSITIONS,
                        strict=True,
                    )
                ]
                velocity_errors = [abs(value) for value in canonical_velocities]
                measured_by_name = dict(
                    zip(
                        PROTOCOL_V4_FULL_JOINT_ORDER,
                        canonical_positions,
                        strict=True,
                    )
                )
                measured_velocities_by_name = dict(
                    zip(
                        PROTOCOL_V4_FULL_JOINT_ORDER,
                        canonical_velocities,
                        strict=True,
                    )
                )
                achieved_deltas = {
                    name: measured_by_name[name] - standing_by_name[name]
                    for name in GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_DELTAS_RAD
                }
                achieved_fractions = {
                    name: achieved_deltas[name] / requested_delta
                    for name, requested_delta in (
                        GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_DELTAS_RAD.items()
                    )
                }
                insufficient_delta_joints = sorted(
                    name
                    for name, fraction in achieved_fractions.items()
                    if fraction
                    < GEAR_SONIC_MANIPULATION_READY_MIN_REQUESTED_DELTA_FRACTION
                )
                manipulation_ready_joint_velocities = {
                    name: measured_velocities_by_name[name]
                    for name in GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_DELTAS_RAD
                }
                manipulation_ready_velocity_errors = [
                    abs(value)
                    for value in manipulation_ready_joint_velocities.values()
                ]
                measured_diagnostics = {
                    "live_articulation_dof_names": measured_names,
                    "protocol_v4_articulation_dof_indices": measured_indices,
                    "measured_full_joint_positions": canonical_positions,
                    "measured_full_joint_velocities": canonical_velocities,
                    "maximum_joint_position_error_rad": max(position_errors),
                    "maximum_joint_velocity_rad_s": max(velocity_errors),
                    "manipulation_ready_joint_velocities_rad_s": (
                        manipulation_ready_joint_velocities
                    ),
                    "maximum_manipulation_ready_joint_velocity_rad_s": max(
                        manipulation_ready_velocity_errors
                    ),
                    "maximum_manipulation_ready_joint_velocity_tolerance_rad_s": (
                        GEAR_SONIC_MANIPULATION_READY_JOINT_VELOCITY_TOLERANCE_RAD_S
                    ),
                    "achieved_joint_position_deltas_rad": achieved_deltas,
                    "achieved_requested_delta_fraction_by_joint": achieved_fractions,
                    "minimum_requested_delta_fraction": (
                        GEAR_SONIC_MANIPULATION_READY_MIN_REQUESTED_DELTA_FRACTION
                    ),
                    "insufficient_delta_joints": insufficient_delta_joints,
                    "settle_steps_executed": settle_step,
                    "maximum_settle_steps": (
                        GEAR_SONIC_MANIPULATION_READY_MAX_SETTLE_STEPS
                    ),
                    "applied_apis": applied_apis,
                }
                if not insufficient_delta_joints and max(
                    manipulation_ready_velocity_errors, default=math.inf
                ) <= GEAR_SONIC_MANIPULATION_READY_JOINT_VELOCITY_TOLERANCE_RAD_S:
                    break
            # The late-June pose is an initial forward-arm seed, not a rigid
            # calibration pose.  PhysX may move a directly seeded joint toward
            # its drive target during the committing update.  Require every
            # task-side joint to retain at least ninety percent of its
            # requested signed displacement; accepting the earlier halfway
            # pose left the elbow just below the 640x480 head-camera crop.  The
            # immediately following head-camera gate remains authoritative for
            # actual elbow/wrist visibility.
            if insufficient_delta_joints:
                raise RuntimeError(
                    "persistent_isaac_manipulation_ready_joint_position_verification_failed"
                )
            if max(manipulation_ready_velocity_errors, default=math.inf) > (
                GEAR_SONIC_MANIPULATION_READY_JOINT_VELOCITY_TOLERANCE_RAD_S
            ):
                raise RuntimeError(
                    "persistent_isaac_manipulation_ready_joint_velocity_verification_failed"
                )
            # The settle loop above proves the drive can attain and retain the
            # requested arm seed, but a free-standing G1 can translate or lean
            # while PhysX resolves those targets.  Reassert the already
            # attested standing base and the verified arm state once, before
            # the renderer's three initial updates.  This is initialization,
            # not an episode action; subsequent policy frames remain fully
            # dynamic.
            start_xyz = _finite_float_vector(
                self.robot_composition.get("start_pose_xyz"),
                expected_length=3,
                error_code="persistent_isaac_manipulation_ready_start_pose_invalid",
            )
            start_yaw = float(self.robot_composition.get("start_yaw_rad"))
            if not math.isfinite(start_yaw):
                raise RuntimeError(
                    "persistent_isaac_manipulation_ready_start_pose_invalid"
                )
            start_quaternion_wxyz = [
                math.cos(start_yaw / 2.0),
                0.0,
                0.0,
                math.sin(start_yaw / 2.0),
            ]
            set_linear_velocity = getattr(self.robot, "set_linear_velocity", None)
            set_angular_velocity = getattr(self.robot, "set_angular_velocity", None)
            if not callable(set_linear_velocity) or not callable(set_angular_velocity):
                raise RuntimeError(
                    "persistent_isaac_manipulation_ready_base_velocity_api_missing"
                )
            self.robot.set_world_pose(
                position=np.asarray(start_xyz, dtype=np.float32),
                orientation=np.asarray(start_quaternion_wxyz, dtype=np.float32),
            )
            set_linear_velocity(np.zeros(3, dtype=np.float32))
            set_angular_velocity(np.zeros(3, dtype=np.float32))
            set_joint_positions(target_positions, joint_indices=joint_indices)
            set_joint_velocities(zero_velocities, joint_indices=joint_indices)
            if callable(apply_action):
                apply_action(
                    _standing_articulation_action(
                        joint_positions=target_positions,
                        joint_velocities=zero_velocities,
                        joint_indices=joint_indices,
                    )
                )
            else:
                if callable(set_joint_position_targets):
                    set_joint_position_targets(
                        target_positions, joint_indices=joint_indices
                    )
                if callable(set_joint_velocity_targets):
                    set_joint_velocity_targets(
                        zero_velocities, joint_indices=joint_indices
                    )
            (
                final_names,
                final_positions,
                final_velocities,
                final_indices,
            ) = self._live_robot_joint_state()
            final_canonical_positions = [
                final_positions[index] for index in final_indices
            ]
            final_canonical_velocities = [
                final_velocities[index] for index in final_indices
            ]
            final_position_errors = [
                abs(actual - expected)
                for actual, expected in zip(
                    final_canonical_positions,
                    GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_POSITIONS,
                    strict=True,
                )
            ]
            final_velocity_errors = [
                abs(value) for value in final_canonical_velocities
            ]
            if max(final_position_errors, default=math.inf) > (
                GEAR_SONIC_STANDING_JOINT_POSITION_TOLERANCE_RAD
            ):
                raise RuntimeError(
                    "persistent_isaac_manipulation_ready_final_state_position_verification_failed"
                )
            if max(final_velocity_errors, default=math.inf) > (
                GEAR_SONIC_STANDING_JOINT_VELOCITY_TOLERANCE_RAD_S
            ):
                raise RuntimeError(
                    "persistent_isaac_manipulation_ready_final_state_velocity_verification_failed"
                )
            measured_diagnostics.update(
                {
                    "final_state_reassertion": {
                        "status": "passed",
                        "purpose": "pre_render_attested_base_and_arm_initialization",
                        "start_pose_xyz": start_xyz,
                        "start_quaternion_wxyz": start_quaternion_wxyz,
                        "physics_updates_after_reassertion": 0,
                        "episode_action": False,
                        "surrogate": False,
                    },
                    "final_live_articulation_dof_names": final_names,
                    "final_protocol_v4_articulation_dof_indices": final_indices,
                    "final_measured_full_joint_positions": (
                        final_canonical_positions
                    ),
                    "final_measured_full_joint_velocities": (
                        final_canonical_velocities
                    ),
                    "final_maximum_joint_position_error_rad": max(
                        final_position_errors
                    ),
                    "final_maximum_joint_velocity_rad_s": max(
                        final_velocity_errors
                    ),
                }
            )
            evidence = {
                **common,
                "status": "passed",
                "completed_at_ns": time.time_ns(),
                **measured_diagnostics,
                "blockers": [],
            }
            artifact = _write_json_atomic(artifact_path, evidence)
            self.manipulation_ready_initialization_artifact = artifact
            return evidence
        except Exception as exc:  # noqa: BLE001 - persist exact blocker first
            blocked = {
                **common,
                "status": "blocked",
                "completed_at_ns": time.time_ns(),
                **measured_diagnostics,
                "blockers": [
                    "persistent_isaac_manipulation_ready_initialization_failed:"
                    f"{type(exc).__name__}:{exc}"
                ],
            }
            _write_json_atomic(artifact_path, blocked)
            raise RuntimeError(blocked["blockers"][0]) from exc

    def _write_live_state_snapshot(
        self,
        *,
        source: str,
        source_action_sha256: str = "",
        source_step_index: int | None = None,
        captured_at_ns: int | None = None,
    ) -> dict[str, Any] | None:
        """Atomically publish one live state/heartbeat for the DDS bridge.

        Real backends always configure the path in ``__init__``.  The missing
        path branch exists only for hermetic tests that instantiate this class
        with ``__new__`` and do not exercise bridge publication.
        """

        path = getattr(self, "live_state_snapshot_path", None)
        if path is None:
            return None
        names, positions, velocities, _ = self._live_robot_joint_state()
        _, base_quaternion = self.robot.get_world_pose()
        get_angular_velocity = getattr(self.robot, "get_angular_velocity", None)
        if not callable(get_angular_velocity):
            raise RuntimeError("persistent_isaac_state_snapshot_base_angular_velocity_unavailable")
        sequence = int(getattr(self, "_live_state_snapshot_sequence", 0)) + 1
        payload = build_gear_sonic_isaac_state_snapshot(
            live_joint_names=names,
            live_joint_positions=positions,
            live_joint_velocities=velocities,
            base_quaternion_wxyz=base_quaternion,
            base_angular_velocity_xyz=get_angular_velocity(),
            simulator_session_id=self.session_id,
            stage_id=self.stage_id,
            heartbeat_sequence=sequence,
            captured_at_ns=int(captured_at_ns or time.time_ns()),
            source=source,
            source_action_sha256=source_action_sha256,
            source_step_index=source_step_index,
            snapshot_path=str(path),
        )
        artifact = _write_json_atomic(path, payload)
        self._live_state_snapshot_sequence = sequence
        self.last_live_state_snapshot = payload
        self.last_live_state_snapshot_artifact = artifact
        return payload

    def refresh_live_state_snapshot(self) -> dict[str, Any]:
        """Resample the idle persistent articulation for the native DDS bridge.

        The HTTP service calls this from the same thread that owns Isaac.  It
        advances neither the episode nor the kitchen physics; it only reads the
        current articulation and refreshes the sub-500 ms state attestation.
        """

        payload = self._write_live_state_snapshot(
            source="idle_service_heartbeat_live_isaac_articulation",
        )
        if payload is None:  # Real backends always configure this path.
            raise RuntimeError("persistent_isaac_state_snapshot_path_missing")
        return payload

    def _refresh_live_state_if_configured(self) -> dict[str, Any] | None:
        """Refresh bridge state without breaking hermetic backend-only tests."""

        return self._write_live_state_snapshot(
            source="thread_affine_operation_heartbeat_live_isaac_articulation",
        )

    def _live_rigid_body_world_pose(self, prim_path: str) -> dict[str, Any]:
        """Read one current PhysX world pose without owning a tensor view.

        Isaac 6 invalidates Python ``SimulationView`` objects across hard
        stop/play lifecycle transitions.  ``SingleRigidPrim`` owns such a view,
        so constructing and caching one after the clock preflight can still
        leave registration and task sampling attached to an invalid generation.
        The documented ``IPhysX.get_rigidbody_transformation`` query reads the
        current simulated rigid body directly and owns no Python tensor-view
        lifetime.  A failed PhysX lookup is terminal: authored USD transforms
        are not accepted as live episode state.
        """

        from pxr import UsdPhysics  # type: ignore

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"persistent_isaac_task_joint_rigid_body_missing:{prim_path}")
        reader = getattr(self, "_physx_rigid_body_transform_reader", None)
        if reader is None:
            import omni.physx  # type: ignore

            reader = omni.physx.get_physx_interface().get_rigidbody_transformation
        try:
            state = reader(prim_path)
        except Exception as exc:  # noqa: BLE001 - normalize Isaac runtime errors
            raise RuntimeError(
                f"persistent_isaac_live_rigid_body_transform_unavailable:{prim_path}"
            ) from exc
        if not isinstance(state, Mapping) or state.get("ret_val") is not True:
            raise RuntimeError(
                f"persistent_isaac_live_rigid_body_transform_unavailable:{prim_path}"
            )
        position = _finite_float_vector(
            state.get("position"),
            expected_length=3,
            error_code=(
                f"persistent_isaac_live_rigid_body_position_invalid:{prim_path}"
            ),
        )
        rotation_xyzw = _finite_float_vector(
            state.get("rotation"),
            expected_length=4,
            error_code=(
                f"persistent_isaac_live_rigid_body_rotation_invalid:{prim_path}"
            ),
        )
        orientation_wxyz = _normalized_quaternion_wxyz(
            (
                rotation_xyzw[3],
                rotation_xyzw[0],
                rotation_xyzw[1],
                rotation_xyzw[2],
            ),
            error_code=(
                f"persistent_isaac_live_rigid_body_rotation_invalid:{prim_path}"
            ),
        )
        return {
            "position": position,
            "orientation": orientation_wxyz,
            "pose_source": ISAAC_PHYSX_LIVE_RIGID_BODY_POSE_SOURCE,
            "surrogate": False,
        }

    def _live_robot_registration_link_poses(self) -> dict[str, Any]:
        """Measure the exact live Isaac links used to register MuJoCo FK.

        Named-link agreement is checked by the official executor before any
        projected landmark can condition OSCAR.  This prevents an assumed
        articulation-root transform from silently standing in for a proven
        cross-simulator registration.
        """

        from pxr import UsdPhysics  # type: ignore

        requested = {"pelvis", *CONTROLLER_FK_REGISTRATION_LANDMARK_NAMES}
        paths_by_name: dict[str, list[str]] = {name: [] for name in requested}
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            name = str(prim.GetName())
            if (
                path.startswith(self.robot_prim_path + "/")
                and name in requested
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                paths_by_name[name].append(path)
        invalid = sorted(
            name for name, paths in paths_by_name.items() if len(paths) != 1
        )
        if invalid:
            raise RuntimeError(
                "persistent_isaac_projection_registration_links_missing_or_ambiguous:"
                + ",".join(invalid)
            )

        def live_pose(name: str) -> dict[str, Any]:
            prim_path = paths_by_name[name][0]
            try:
                state = self._live_rigid_body_world_pose(prim_path)
            except Exception as exc:  # noqa: BLE001 - normalize Isaac runtime errors
                raise RuntimeError(
                    f"persistent_isaac_projection_link_dynamic_state_unavailable:{name}"
                ) from exc
            return {
                "landmark_id": name,
                "prim_path": prim_path,
                "world_position_xyz": _finite_float_vector(
                    state["position"],
                    expected_length=3,
                    error_code=(
                        f"persistent_isaac_projection_link_position_invalid:{name}"
                    ),
                ),
                "world_quaternion_wxyz": list(
                    _normalized_quaternion_wxyz(
                        state["orientation"],
                        error_code=(
                            f"persistent_isaac_projection_link_orientation_invalid:{name}"
                        ),
                    )
                ),
                "pose_source": state["pose_source"],
                "surrogate": False,
            }

        return {
            "pelvis": live_pose("pelvis"),
            "landmarks": [
                live_pose(name)
                for name in CONTROLLER_FK_REGISTRATION_LANDMARK_NAMES
            ],
        }

    def _task_body_world_rotation(self, prim_path: str) -> tuple[float, ...]:
        try:
            orientation = self._live_rigid_body_world_pose(prim_path)["orientation"]
        except Exception as exc:  # noqa: BLE001 - normalize Isaac runtime errors
            raise RuntimeError(
                f"persistent_isaac_task_body_dynamic_state_unavailable:{prim_path}"
            ) from exc
        return _normalized_quaternion_wxyz(
            orientation,
            error_code="persistent_isaac_task_body_rotation_invalid",
        )

    def _task_joint_binding(self, criterion: Mapping[str, Any]) -> RevoluteTaskJointBinding:
        prim_path = self._resolve_task_prim(criterion)
        cached = getattr(self, "_task_joint_bindings", {}).get(prim_path)
        if cached is not None:
            return cached
        binding = resolve_revolute_task_joint(self.stage, contracted_prim_path=prim_path)
        if not hasattr(self, "_task_joint_bindings"):
            self._task_joint_bindings = {}
        self._task_joint_bindings[prim_path] = binding
        return binding

    def _task_joint_sample(
        self,
        binding: RevoluteTaskJointBinding,
        criterion: Mapping[str, Any],
    ) -> dict[str, Any]:
        if (
            str(criterion.get("criterion_id") or "") != "microwave_door_open_angle"
            or str(criterion.get("observable_transition") or "") != "articulation_angle_rad"
            or str(criterion.get("comparison") or "") != "increase_at_least"
            or str(criterion.get("unit") or "") != "rad"
        ):
            raise RuntimeError("persistent_isaac_task_joint_measurement_contract_unsupported")
        raw_signed = measure_revolute_joint_signed_angle_rad(
            self.stage,
            binding,
            body_rotation_reader=self._task_body_world_rotation,
        )
        # The exact Palatial hinge is limited to [-90, 0] degrees.  Opening is
        # therefore a negative signed coordinate, while the registered
        # semantic criterion is an increase in open angle.  Distance below the
        # authored closed upper limit is the physical opening angle.
        lower = math.radians(binding.lower_limit_degrees)
        upper = math.radians(binding.upper_limit_degrees)
        if binding.lower_limit_degrees >= 0.0 or abs(binding.upper_limit_degrees) > 1e-6:
            raise RuntimeError("persistent_isaac_task_joint_opening_limit_convention_unsupported")
        tolerance = math.radians(2.0)
        if raw_signed < lower - tolerance or raw_signed > upper + tolerance:
            raise RuntimeError("persistent_isaac_task_joint_position_outside_limits")
        bounded_signed = min(upper, max(lower, raw_signed))
        value = upper - bounded_signed
        return {
            "value_rad": value,
            "raw_signed_angle_rad": raw_signed,
            "bounded_signed_angle_rad": bounded_signed,
            "measurement_convention": binding.measurement_convention,
            "pose_source": ISAAC_PHYSX_LIVE_RIGID_BODY_POSE_SOURCE,
            "physics_joint_prim_path": binding.joint_prim_path,
            "joint_axis": binding.axis,
            "joint_lower_limit_degrees": binding.lower_limit_degrees,
            "joint_upper_limit_degrees": binding.upper_limit_degrees,
            "measurement_backend_source_sha256": getattr(
                self,
                "measurement_backend_source_sha256",
                measurement_backend_source_sha256(),
            ),
            "surrogate": False,
        }

    def _resolve_task_prim(self, criterion: Mapping[str, Any]) -> str:
        exact = str(criterion.get("articulation_prim_path") or "").strip()
        if exact:
            return exact
        resolution = dict(criterion.get("articulation_prim_path_resolution") or {})
        root_term = str(resolution.get("required_target_root") or "").lower()
        terms = [str(item).lower() for item in resolution.get("required_affordance_terms") or []]
        from pxr import UsdPhysics  # type: ignore
        import omni.usd  # type: ignore

        stage = omni.usd.get_context().get_stage()
        matches = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            lower = path.lower()
            if root_term and root_term not in lower:
                continue
            if terms and not any(term in lower for term in terms):
                continue
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                matches.append(path)
        if len(matches) != 1:
            raise RuntimeError(
                "persistent_isaac_task_prim_resolution_not_unique:" + ",".join(matches)
            )
        return matches[0]

    def _apply_controller_state(self, state: Mapping[str, Any]) -> None:
        names = [str(item) for item in state.get("joint_names") or []]
        positions = [float(item) for item in state.get("joint_positions") or []]
        if (
            str(state.get("joint_order_schema_version") or "") != JOINT_ORDER_SCHEMA_VERSION
            or str(state.get("mapping_digest") or "") != PROTOCOL_V4_MAPPING_DIGEST
        ):
            raise RuntimeError("persistent_isaac_controller_joint_mapping_invalid")
        try:
            validate_full_joint_order(names, source="persistent_isaac_controller")
        except ValueError as exc:
            raise RuntimeError("persistent_isaac_controller_joint_mapping_invalid") from exc
        if len(names) != len(positions) or len(names) != len(PROTOCOL_V4_FULL_JOINT_ORDER):
            raise RuntimeError("persistent_isaac_controller_joint_state_invalid")
        joint_indices = []
        for name in names:
            try:
                joint_index = int(self.robot.get_dof_index(name))
            except Exception as exc:  # noqa: BLE001 - normalize Isaac lookup errors
                raise RuntimeError(f"persistent_isaac_robot_dof_missing:{name}") from exc
            if joint_index < 0:
                raise RuntimeError(f"persistent_isaac_robot_dof_missing:{name}")
            joint_indices.append(joint_index)
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction  # type: ignore

        self.robot.apply_action(
            ArticulationAction(
                joint_positions=np.asarray(positions, dtype=np.float32),
                joint_indices=np.asarray(joint_indices, dtype=np.int64),
            )
        )

    def _live_projected_gravity(self) -> list[float]:
        """Measure base orientation and express world gravity in the base frame."""
        _, rotation = self.robot.get_world_pose()
        w, x, y, z = (float(value) for value in rotation)
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError("persistent_isaac_base_orientation_invalid")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        projected = [
            -(2.0 * x * z - 2.0 * y * w),
            -(2.0 * y * z + 2.0 * x * w),
            -(1.0 - 2.0 * x * x - 2.0 * y * y),
        ]
        if not all(math.isfinite(value) for value in projected):
            raise RuntimeError("persistent_isaac_projected_gravity_invalid")
        return projected

    def _single_registered_criterion(self, contract: Mapping[str, Any]) -> dict[str, Any]:
        criteria = [
            dict(item)
            for item in contract.get("registered_criteria") or contract.get("criteria") or []
        ]
        if len(criteria) != 1:
            raise RuntimeError("persistent_isaac_requires_one_registered_criterion")
        return criteria[0]

    def capture_episode_baseline(
        self,
        *,
        task_success_contract: Mapping[str, Any],
        attempt_id: str,
        launch_nonce: str,
        task_contract_artifact_sha256: str | None = None,
        settle_steps: int = 8,
    ) -> dict[str, Any]:
        """Capture the immutable episode baseline after settle, before action zero."""
        if getattr(self, "episode_baseline", None) is not None:
            raise RuntimeError("persistent_isaac_episode_baseline_already_captured")
        contract = dict(task_success_contract or {})
        criterion = self._single_registered_criterion(contract)
        prim_path = self._resolve_task_prim(criterion)
        task_joint = self._task_joint_binding(criterion)
        for _ in range(max(1, int(settle_steps))):
            self.app.update()
        self._write_live_state_snapshot(
            source="episode_baseline_settled_live_isaac_articulation",
        )
        initial_sample = self._task_joint_sample(task_joint, criterion)
        baseline = build_task_episode_baseline(
            episode_initial_value=float(initial_sample["value_rad"]),
            attempt_id=str(attempt_id),
            launch_nonce=str(launch_nonce),
            simulator_session_id=self.session_id,
            stage_id=self.stage_id,
            articulation_prim_path=prim_path,
            task_contract_sha256=canonical_task_contract_sha256(contract),
            task_contract_artifact_sha256=task_contract_artifact_sha256,
            criterion_id=str(criterion.get("criterion_id") or ""),
            unit=str(criterion.get("unit") or ""),
            captured_timestamp=str(time.time_ns()),
        )
        baseline["physics_joint_binding"] = task_joint.to_payload()
        baseline["initial_physics_measurement"] = dict(initial_sample)
        baseline["measurement_backend_source_sha256"] = getattr(
            self,
            "measurement_backend_source_sha256",
            measurement_backend_source_sha256(),
        )
        artifact = self.evidence_dir / "task_episode_baseline.json"
        try:
            with artifact.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        except FileExistsError as exc:
            raise RuntimeError("persistent_isaac_episode_baseline_artifact_already_exists") from exc
        self.episode_baseline = dict(baseline)
        self.attempt_id = str(attempt_id)
        self.launch_nonce = str(launch_nonce)
        self.episode_baseline_artifact = {
            "path": str(artifact),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
        self.live_geometry_results = self._measure_live_geometry(
            target_prim_path=prim_path,
            task_success_contract=contract,
        )
        return dict(baseline)

    def _measure_live_geometry(
        self, *, target_prim_path: str, task_success_contract: Mapping[str, Any]
    ) -> dict[str, dict[str, Any]]:
        from .isaac_live_geometry_validation import build_live_geometry_results

        try:
            position, orientation = self.robot.get_world_pose()
            robot_xyz = [float(value) for value in position]
            # Isaac Core returns WXYZ; the geometry validator consumes XYZW.
            quat = [
                float(orientation[1]),
                float(orientation[2]),
                float(orientation[3]),
                float(orientation[0]),
            ]
            renderer = self.review_renderer
            target_xyz = renderer._center(target_prim_path)
            import omni.physx  # type: ignore

            overlaps: list[str] = []
            overlap_decode_errors: list[str] = []

            def on_hit(hit):
                try:
                    overlaps.append(_physx_overlap_hit_prim_path(hit))
                except RuntimeError as exc:
                    overlap_decode_errors.append(str(exc))
                    return False
                return True

            query = omni.physx.get_physx_scene_query_interface()
            query.overlap_box(
                G1_LIVE_COLLISION_HALF_EXTENT_M,
                tuple(robot_xyz),
                tuple(quat),
                on_hit,
                False,
            )
            if overlap_decode_errors:
                raise RuntimeError(overlap_decode_errors[0])
            max_reach = float(task_success_contract.get("max_reach_distance_m") or 1.5)
            return build_live_geometry_results(
                robot_xyz=robot_xyz,
                robot_quaternion_xyzw=quat,
                target_xyz=target_xyz,
                overlapping_prim_paths=overlaps,
                robot_prim_path=self.robot_prim_path,
                max_reach_distance_m=max_reach,
            )
        except Exception as exc:  # noqa: BLE001 - unsupported runtime must fail closed
            blocker = f"live_isaac_geometry_measurement_failed:{type(exc).__name__}"
            return {
                "stance": {
                    "schema_version": "g1_kitchen_live_stance_validation.v1",
                    "stance_valid": False,
                    "reach_valid": False,
                    "facing_valid": False,
                    "blockers": [blocker],
                },
                "collision": {
                    "schema_version": "g1_kitchen_live_collision_validation.v1",
                    "collision_free": False,
                    "clearance_valid": False,
                    "blockers": [blocker],
                },
            }

    def install_episode_baseline_attestation(self, attestation: Mapping[str, Any]) -> None:
        if self.episode_baseline is None:
            raise RuntimeError("persistent_isaac_episode_baseline_missing")
        if getattr(self, "episode_baseline_attestation", None) is not None:
            raise RuntimeError("persistent_isaac_episode_baseline_attestation_already_installed")
        self.episode_baseline_attestation = dict(attestation)

    def capture_initial_policy_observation(
        self,
        *,
        target_prim_path: str,
        frame_output_path: str | Path,
        projection_context_output_path: str | Path,
    ) -> dict[str, Any]:
        """Render and bind the actual standing-state RGB used by GR00T/OSCAR.

        This runs after the attempt baseline is captured and before the HTTP
        service becomes ready.  Consequently, a ready service proves the stale
        bundled fallback has been replaced by a frame and camera contract from
        this exact persistent Isaac session.
        """

        if not str(target_prim_path or "").startswith("/"):
            raise RuntimeError("persistent_isaac_initial_policy_target_prim_invalid")
        renderer = getattr(self, "review_renderer", None)
        if renderer is None:
            raise RuntimeError("persistent_isaac_initial_policy_renderer_missing")
        live_registration = self._live_robot_registration_link_poses()
        set_calibration_landmarks = getattr(
            renderer, "set_initial_robot_pov_calibration_landmarks", None
        )
        if callable(set_calibration_landmarks):
            set_calibration_landmarks(live_registration["landmarks"])
        frames = list(
            renderer.render(
                step_index=0,
                target_prim_path=str(target_prim_path),
            )
        )
        robot_pov_rows = [
            dict(row)
            for row in frames
            if isinstance(row, Mapping) and row.get("camera_role") == "robot_pov"
        ]
        if len(robot_pov_rows) != 1:
            raise RuntimeError("persistent_isaac_initial_robot_pov_not_unique")
        source = robot_pov_rows[0]
        source_path = Path(str(source.get("path") or "")).expanduser().resolve()
        if source_path.is_symlink() or not source_path.is_file():
            raise RuntimeError("persistent_isaac_initial_robot_pov_missing_or_unsafe")
        source_bytes = source_path.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if source_sha256 != str(source.get("sha256") or "").strip().lower():
            raise RuntimeError("persistent_isaac_initial_robot_pov_sha256_mismatch")
        width = int(source.get("width") or 0)
        height = int(source.get("height") or 0)
        if (width, height) != (640, 480):
            raise RuntimeError("persistent_isaac_initial_robot_pov_resolution_invalid")
        visual_signal = dict(source.get("visual_signal") or {})
        if (
            visual_signal.get("status") != "completed"
            or visual_signal.get("non_uniform") is not True
        ):
            raise RuntimeError("persistent_isaac_initial_robot_pov_visual_signal_invalid")
        frame_artifact = _write_bytes_atomic(frame_output_path, source_bytes)
        frame_artifact.update(
            {
                "width": width,
                "height": height,
                "camera_role": "robot_pov",
                "captured_from_path": str(source_path),
            }
        )
        camera_contract = dict(source.get("camera_contract") or {})
        if not camera_contract:
            camera_contract_method = getattr(renderer, "camera_contract", None)
            if not callable(camera_contract_method):
                raise RuntimeError(
                    "persistent_isaac_initial_robot_pov_camera_contract_missing"
                )
            camera_contract = dict(camera_contract_method("robot_pov") or {})
        if (
            camera_contract.get("available") is not True
            or camera_contract.get("projection_token") != "perspective"
            or list(camera_contract.get("resolution") or []) != [width, height]
            or camera_contract.get("viewpoint_mode")
            != "robot_head_mounted_egocentric"
            or camera_contract.get("robot_mounted") is not True
            or camera_contract.get("policy_observation_eligible") is not True
            or camera_contract.get("mount_motion_model")
            != "rigid_head_local_transform"
            or camera_contract.get("gaze_motion_model")
            != "inherits_head_orientation_no_task_reaim"
        ):
            raise RuntimeError(
                "persistent_isaac_initial_robot_pov_camera_contract_invalid"
            )
        (
            live_joint_names,
            live_joint_positions,
            live_joint_velocities,
            canonical_joint_indices,
        ) = self._live_robot_joint_state()
        standing_landmark_projections = {
            str(row["landmark_id"]): project_world_point(
                camera_contract,
                row["world_position_xyz"],
            )
            for row in live_registration["landmarks"]
        }
        if set(standing_landmark_projections) != set(
            CONTROLLER_FK_REGISTRATION_LANDMARK_NAMES
        ):
            raise RuntimeError(
                "persistent_isaac_initial_robot_pov_registration_landmarks_incomplete"
            )
        in_frame_registration_landmark_names = sorted(
            name
            for name, projection in standing_landmark_projections.items()
            if projection.get("in_frame") is True
        )
        missing_active_arm_links = sorted(
            set(ROBOT_POV_REQUIRED_ACTIVE_ARM_LINK_NAMES)
            - set(in_frame_registration_landmark_names)
        )
        pelvis_pose = dict(live_registration["pelvis"])
        pelvis_xyz = list(pelvis_pose["world_position_xyz"])
        pelvis_quaternion_wxyz = list(pelvis_pose["world_quaternion_wxyz"])
        standing_positions = [
            live_joint_positions[index] for index in canonical_joint_indices
        ]
        standing_velocities = [
            live_joint_velocities[index] for index in canonical_joint_indices
        ]
        captured_at_ns = time.time_ns()
        context = {
            "schema_version": CONTROLLER_FK_CAMERA_PROJECTION_SCHEMA_VERSION,
            "status": "captured_from_live_persistent_isaac_session",
            "attempt_id": self.attempt_id,
            "launch_nonce": self.launch_nonce,
            "allocation_launch_session_id": getattr(
                self, "allocation_launch_session_id", self.launch_nonce
            ),
            "qualification_attempt_bound": getattr(
                self, "qualification_attempt_bound", False
            ),
            "qualification_attempt_sequence": getattr(
                self, "qualification_attempt_sequence", None
            ),
            "qualification_attempt_nonce_sha256": (
                getattr(self, "qualification_attempt_nonce_sha256", None)
            ),
            "simulator_session_id": self.session_id,
            "stage_id": self.stage_id,
            "captured_at_ns": captured_at_ns,
            "source_frame_artifact": frame_artifact,
            "source_render_artifact": source,
            "camera_contract": camera_contract,
            "live_isaac_pelvis_world_pose": {
                "prim_path": pelvis_pose["prim_path"],
                "position_xyz": pelvis_xyz,
                "quaternion_wxyz": pelvis_quaternion_wxyz,
                "yaw_radians": _yaw_from_quaternion_wxyz(
                    pelvis_quaternion_wxyz
                ),
                "source": "live_isaac_pelvis_rigid_body_at_initial_policy_frame_capture",
            },
            "standing_cross_simulator_registration": {
                "status": "pending_official_mujoco_named_link_residual_verification",
                "required_landmark_names": list(
                    CONTROLLER_FK_REGISTRATION_LANDMARK_NAMES
                ),
                "isaac_named_link_world_poses": list(
                    live_registration["landmarks"]
                ),
                "standing_joint_names": list(PROTOCOL_V4_FULL_JOINT_ORDER),
                "standing_joint_positions": standing_positions,
                "standing_joint_velocities": standing_velocities,
                "live_articulation_dof_names": live_joint_names,
                "joint_and_link_pose_captured_at_ns": captured_at_ns,
                "camera_projection_validation": {
                    "status": "captured",
                    "active_forearm_visibility_passed": not missing_active_arm_links,
                    "missing_active_arm_link_names": missing_active_arm_links,
                    "all_required_landmarks_in_frame": len(
                        in_frame_registration_landmark_names
                    )
                    == len(CONTROLLER_FK_REGISTRATION_LANDMARK_NAMES),
                    "in_frame_landmark_count": len(
                        in_frame_registration_landmark_names
                    ),
                    "in_frame_landmark_names": (
                        in_frame_registration_landmark_names
                    ),
                    "visibility_required_for_named_link_registration": False,
                    "active_forearm_visibility_required_for_policy_observation": True,
                    "required_active_arm_link_names": list(
                        ROBOT_POV_REQUIRED_ACTIVE_ARM_LINK_NAMES
                    ),
                    "projections": standing_landmark_projections,
                },
                "maximum_residual_tolerance_m": (
                    CONTROLLER_FK_REGISTRATION_MAX_RESIDUAL_M
                ),
                "surrogate": False,
            },
            "coordinate_transform": CONTROLLER_FK_CAMERA_PROJECTION_TRANSFORM,
            "camera_calibration_mode": (
                "same_session_live_isaac_robot_head_mounted_egocentric"
            ),
            "visual_signal": visual_signal,
            "claim_boundary": {
                "projection_is_bound_to_exact_live_initial_rgb_camera": True,
                "policy_observation_is_robot_head_mounted_egocentric": True,
                "policy_camera_inherits_head_translation_and_rotation": True,
                "policy_camera_reaims_at_task_each_frame": False,
                "third_person_overview_is_review_only": True,
                "bundled_seed_frame_reused": False,
                "named_link_cross_simulator_registration_required": True,
                "projection_is_action_derived_controller_fk_support": True,
                "projection_is_not_task_success_or_contact_proof": True,
            },
        }
        context_artifact = _write_json_atomic(projection_context_output_path, context)
        if missing_active_arm_links:
            raise RuntimeError(
                "persistent_isaac_initial_robot_pov_active_forearm_not_in_frame:"
                + ",".join(missing_active_arm_links)
            )
        initial_frame_rows: dict[str, Any] = {}
        for row in frames:
            role = str(row.get("camera_role") or "")
            row_path = Path(str(row.get("path") or "")).expanduser().resolve()
            if role not in {"overview", "robot_pov"} or row.get("frame_index") != 0:
                raise RuntimeError("persistent_isaac_initial_frame_binding_invalid")
            if row_path.is_symlink() or not row_path.is_file():
                raise RuntimeError("persistent_isaac_initial_frame_binding_missing")
            row_sha256 = hashlib.sha256(row_path.read_bytes()).hexdigest()
            if row_sha256 != str(row.get("sha256") or "").lower():
                raise RuntimeError("persistent_isaac_initial_frame_binding_sha256_mismatch")
            initial_frame_rows[row_path.name] = {
                "camera_role": role,
                "step_index": 0,
                "review_frame_index": 0,
                "control_frame_global_index": 0,
                "initial_frame": True,
                "camera_motion_model": (
                    "rigid_head_local_transform"
                    if role == "robot_pov"
                    else "task_framed_third_person_review"
                ),
                "sha256": row_sha256,
                "simulator_session_id": self.session_id,
                "stage_id": self.stage_id,
                "attempt_id": self.attempt_id,
                "launch_nonce": self.launch_nonce,
                "allocation_launch_session_id": getattr(
                    self, "allocation_launch_session_id", self.launch_nonce
                ),
                "qualification_attempt_bound": getattr(
                    self, "qualification_attempt_bound", False
                ),
                "qualification_attempt_sequence": getattr(
                    self, "qualification_attempt_sequence", None
                ),
                "qualification_attempt_nonce_sha256": getattr(
                    self, "qualification_attempt_nonce_sha256", None
                ),
                "episode_baseline_digest": str(
                    dict(getattr(self, "episode_baseline", {}) or {}).get(
                        "baseline_digest"
                    )
                    or ""
                ),
                "captured_at_ns": captured_at_ns,
            }
        if set(initial_frame_rows) != {"overview_0000.png", "robot_pov_0000.png"}:
            raise RuntimeError("persistent_isaac_initial_camera_roles_incomplete")
        initial_bindings_artifact = _write_json_atomic(
            self.evidence_dir / "frames" / "initial_frame_bindings.json",
            {
                "schema_version": "isaac_initial_review_frame_bindings.v1",
                "frames": initial_frame_rows,
            },
        )
        evidence = {
            "schema_version": "persistent_isaac_initial_policy_observation.v1",
            "status": "completed",
            "simulator_session_id": self.session_id,
            "stage_id": self.stage_id,
            "target_prim_path": str(target_prim_path),
            "source_frame_artifact": frame_artifact,
            "camera_projection_context_artifact": context_artifact,
            "camera_projection_context": context,
            "initial_frame_bindings_artifact": initial_bindings_artifact,
            "blockers": [],
        }
        evidence_artifact = _write_json_atomic(
            self.evidence_dir / "initial_policy_observation.json",
            evidence,
        )
        evidence["evidence_artifact"] = evidence_artifact
        return evidence

    def verify_initial_observation_preserved_episode_baseline(
        self,
        *,
        task_success_contract: Mapping[str, Any],
        tolerance_rad: float = 1e-4,
    ) -> dict[str, Any]:
        """Fail if rendering advanced the semantic task past its signed baseline."""

        baseline = dict(getattr(self, "episode_baseline", {}) or {})
        if not baseline:
            raise RuntimeError("persistent_isaac_episode_baseline_missing")
        tolerance = float(tolerance_rad)
        if not math.isfinite(tolerance) or not 0.0 < tolerance <= 1e-3:
            raise RuntimeError("persistent_isaac_initial_observation_baseline_tolerance_invalid")
        criterion = self._single_registered_criterion(task_success_contract)
        binding = self._task_joint_binding(criterion)
        sample = self._task_joint_sample(binding, criterion)
        expected = float(baseline["episode_initial_value"])
        observed = float(sample["value_rad"])
        drift = abs(observed - expected)
        result = {
            "schema_version": "persistent_isaac_initial_observation_baseline_guard.v1",
            "status": "passed" if drift <= tolerance else "blocked",
            "simulator_session_id": self.session_id,
            "stage_id": self.stage_id,
            "criterion_id": criterion.get("criterion_id"),
            "episode_initial_value_rad": expected,
            "post_render_value_rad": observed,
            "absolute_drift_rad": drift,
            "maximum_allowed_drift_rad": tolerance,
            "post_render_measurement": sample,
            "surrogate": False,
            "blockers": (
                []
                if drift <= tolerance
                else ["initial_policy_observation_render_changed_episode_baseline"]
            ),
        }
        artifact = _write_json_atomic(
            self.evidence_dir / "initial_policy_observation_baseline_guard.json",
            result,
        )
        result["evidence_artifact"] = artifact
        if drift > tolerance:
            raise RuntimeError(
                "persistent_isaac_initial_observation_changed_episode_baseline:"
                f"{drift:.9f}>{tolerance:.9f}"
            )
        return result

    def apply_and_measure(self, request: Mapping[str, Any]) -> dict[str, Any]:
        action = dict(request.get("action") or {})
        wam_output = dict(request.get("wam_output") or {})
        state = dict(wam_output.get("generated_robot_state") or {})
        source_action_sha = hashlib.sha256(
            json.dumps(action, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        if str(state.get("source_action_sha256") or "") != source_action_sha:
            raise RuntimeError("persistent_isaac_controller_state_action_mismatch")
        contract = dict(request.get("task_success_contract") or {})
        criterion = self._single_registered_criterion(contract)
        prim_path = self._resolve_task_prim(criterion)
        task_joint = self._task_joint_binding(criterion)
        baseline = getattr(self, "episode_baseline", None)
        baseline_blockers = verify_task_episode_baseline(
            baseline,
            simulator_session_id=self.session_id,
            stage_id=self.stage_id,
            articulation_prim_path=prim_path,
            task_contract_sha256=canonical_task_contract_sha256(contract),
            attempt_id=getattr(self, "attempt_id", ""),
            launch_nonce=getattr(self, "launch_nonce", ""),
        )
        if baseline_blockers:
            raise RuntimeError(
                "persistent_isaac_episode_baseline_invalid:" + ",".join(baseline_blockers)
            )
        if not getattr(self, "episode_baseline_attestation", None):
            raise RuntimeError("persistent_isaac_episode_baseline_attestation_missing")
        if dict(baseline.get("physics_joint_binding") or {}) != task_joint.to_payload():
            raise RuntimeError("persistent_isaac_episode_baseline_joint_binding_mismatch")
        episode_initial = float(baseline["episode_initial_value"])
        step = int(request.get("step_index") or 0)
        before_timestamp = time.time_ns()
        before_sample = self._task_joint_sample(task_joint, criterion)
        before = float(before_sample["value_rad"])
        (
            controller_sequence,
            controller_execution_contract,
            explicit_controller_sequence,
        ) = _validated_controller_execution_sequence(state=state, action=action)
        requested_control_frame_count = len(controller_sequence)
        executed_control_frame_count = 0
        controller_frame_measurements: list[dict[str, Any]] = []
        controller_horizon_terminated_on_semantic_success = False
        renderer = getattr(self, "review_renderer", None)
        if renderer is None:
            raise RuntimeError("persistent_isaac_attempt_review_renderer_missing")
        review_frames: list[dict[str, Any]] = []
        controller_execution_started_at_ns = time.time_ns()
        for frame_index, controller_frame in enumerate(controller_sequence):
            replay_timeline = getattr(self, "timeline", None)
            if frame_index == 0 and explicit_controller_sequence and replay_timeline is not None and not bool(replay_timeline.is_playing()):
                # The executor idles stopped after its startup calibration probe;
                # replaying frames needs live physics or delta==1 reads 0/False.
                replay_timeline.play()
                self.app.update()
            frame_state = {
                **state,
                **controller_frame,
                "joint_order_schema_version": JOINT_ORDER_SCHEMA_VERSION,
                "mapping_digest": PROTOCOL_V4_MAPPING_DIGEST,
            }
            self._apply_controller_state(frame_state)
            self._refresh_live_state_if_configured()
            # The live stage is configured to 50 time codes/target frames per
            # second before first play, so one update consumes one exact SONIC
            # control sample without interpolating or discarding the horizon.
            simulation_manager = getattr(self, "_simulation_manager", None)
            if explicit_controller_sequence and simulation_manager is None:
                raise RuntimeError("persistent_isaac_physics_step_counter_missing")
            physics_step_count_before = int(simulation_manager.get_num_physics_steps()) if simulation_manager is not None else None
            simulation_time_before = (
                float(simulation_manager.get_simulation_time())
                if simulation_manager is not None
                else None
            )
            follow_live_robot = getattr(renderer, "follow_live_robot", None)
            if not callable(follow_live_robot):
                raise RuntimeError("persistent_isaac_head_camera_follow_missing")
            # Author the head-mounted camera from the latest live articulation
            # before the one allowed physics/render update.  Replicator then
            # captures the newly rendered view without introducing a hidden
            # second physics step.
            follow_live_robot()
            self.app.update()
            physics_step_count_after = (
                int(simulation_manager.get_num_physics_steps())
                if simulation_manager is not None
                else None
            )
            simulation_time_after = (
                float(simulation_manager.get_simulation_time())
                if simulation_manager is not None
                else None
            )
            physics_step_delta = (
                physics_step_count_after - physics_step_count_before
                if physics_step_count_before is not None
                and physics_step_count_after is not None
                else None
            )
            simulation_time_delta_seconds = (
                simulation_time_after - simulation_time_before
                if simulation_time_before is not None
                and simulation_time_after is not None
                else None
            )
            if explicit_controller_sequence and (
                physics_step_delta != 1
                or simulation_time_delta_seconds is None
                or not math.isclose(
                    simulation_time_delta_seconds,
                    GEAR_SONIC_CONTROL_DT_SECONDS,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise RuntimeError(f"persistent_isaac_controller_frame_physics_step_delta_invalid:delta={physics_step_delta}:dt={simulation_time_delta_seconds}:playing={bool(self.timeline.is_playing())}:frame={frame_index}")
            self._refresh_live_state_if_configured()
            executed_control_frame_count += 1
            frame_sample = self._task_joint_sample(task_joint, criterion)
            frame_value = float(frame_sample["value_rad"])
            frame_evaluation = evaluate_task_criterion(
                criterion,
                episode_initial_value=episode_initial,
                step_before=before,
                step_after=frame_value,
            )
            control_frame_global_index = int(
                getattr(self, "controller_global_frame_index", 0)
            ) + 1
            self.controller_global_frame_index = control_frame_global_index
            terminal_frame = bool(frame_evaluation.get("passed"))
            scheduled_review_frame = (
                not explicit_controller_sequence
                or control_frame_global_index % ISAAC_REVIEW_CONTROLLER_FRAME_STRIDE == 0
            )
            sampled_for_review = scheduled_review_frame or terminal_frame
            frame_artifacts: list[dict[str, Any]] = []
            review_frame_index: int | None = None
            review_frame_captured_at_ns: int | None = None
            if sampled_for_review:
                review_frame_index = int(
                    getattr(self, "review_execution_frame_index", 0)
                ) + 1
                self.review_execution_frame_index = review_frame_index
                capture_current = getattr(renderer, "capture_current", None)
                if not callable(capture_current):
                    raise RuntimeError(
                        "persistent_isaac_zero_update_review_capture_missing"
                    )
                review_frame_captured_at_ns = time.time_ns()
                captured_frames = list(capture_current(step_index=review_frame_index))
                if not captured_frames:
                    raise RuntimeError("persistent_isaac_controller_review_frame_missing")
                for captured in captured_frames:
                    captured.update(
                        {
                            "control_frame_global_index": control_frame_global_index,
                            "physics_step_count_before": physics_step_count_before,
                            "physics_step_count_after": physics_step_count_after,
                            "physics_step_delta": physics_step_delta,
                            "simulation_time_before_seconds": simulation_time_before,
                            "simulation_time_after_seconds": simulation_time_after,
                            "simulation_time_delta_seconds": simulation_time_delta_seconds,
                            "outer_source_step_index": step,
                            "horizon_frame_index": frame_index,
                            "controller_frame_index": controller_frame.get(
                                "controller_frame_index"
                            ),
                            "source_action_frame_sha256": controller_frame.get(
                                "source_action_frame_sha256"
                            ),
                            "task_joint_value_rad": frame_value,
                            "registered_transition_passed": terminal_frame,
                            "semantic_terminal_frame": terminal_frame,
                            "captured_at_ns": review_frame_captured_at_ns,
                            "camera_motion_model": (
                                "rigid_head_local_transform"
                                if captured.get("camera_role") == "robot_pov"
                                else "task_framed_third_person_review"
                            ),
                        }
                    )
                    frame_artifacts.append(
                        {
                            "camera_role": captured.get("camera_role"),
                            "frame_index": review_frame_index,
                            "control_frame_global_index": control_frame_global_index,
                            "path": captured.get("path"),
                            "sha256": captured.get("sha256"),
                        }
                    )
                review_frames.extend(captured_frames)
            controller_frame_measurements.append(
                {
                    "control_frame_global_index": control_frame_global_index,
                    "physics_step_count_before": physics_step_count_before,
                    "physics_step_count_after": physics_step_count_after,
                    "physics_step_delta": physics_step_delta,
                    "simulation_time_before_seconds": simulation_time_before,
                    "simulation_time_after_seconds": simulation_time_after,
                    "simulation_time_delta_seconds": simulation_time_delta_seconds,
                    "horizon_frame_index": frame_index,
                    "controller_frame_index": controller_frame.get(
                        "controller_frame_index"
                    ),
                    "source_action_frame_sha256": controller_frame.get(
                        "source_action_frame_sha256"
                    ),
                    "value_rad": frame_value,
                    "episode_delta_rad": frame_value - episode_initial,
                    "registered_transition_passed": bool(
                        frame_evaluation.get("passed")
                    ),
                    "scheduled_review_frame": scheduled_review_frame,
                    "sampled_for_review": sampled_for_review,
                    "review_frame_index": review_frame_index,
                    "review_frame_artifacts": frame_artifacts,
                    "review_frame_captured_at_ns": review_frame_captured_at_ns,
                    "semantic_terminal_frame": terminal_frame,
                    "sampled_at_ns": time.time_ns(),
                }
            )
            if explicit_controller_sequence and frame_evaluation.get("passed") is True:
                controller_horizon_terminated_on_semantic_success = True
                break

        if not explicit_controller_sequence:
            # Preserve the established legacy settle behavior for callers that
            # intentionally submit one frame without the signed horizon contract.
            for _ in range(
                max(1, int(request.get("physics_steps_per_action") or 4)) - 1
            ):
                follow_live_robot = getattr(renderer, "follow_live_robot", None)
                if not callable(follow_live_robot):
                    raise RuntimeError("persistent_isaac_head_camera_follow_missing")
                follow_live_robot()
                self.app.update()
                self._refresh_live_state_if_configured()
        controller_execution_completed_at_ns = time.time_ns()
        evidence_step = int(request.get("evidence_step_index", step))
        if evidence_step != step:
            raise RuntimeError("persistent_isaac_action_evidence_step_mismatch")
        # Renderer heartbeats intentionally use a generic source while a
        # thread-affine update or PNG write is in flight. Re-sample after all
        # rendering so the action-bound state and the on-disk bridge snapshot
        # are the exact final state returned to the next GR00T query.
        post_action_captured_at_ns = time.time_ns()
        post_action_state_snapshot = self._write_live_state_snapshot(
            source="post_action_live_isaac_articulation",
            source_action_sha256=source_action_sha,
            source_step_index=step,
            captured_at_ns=post_action_captured_at_ns,
        )
        post_action_policy_state = bind_post_action_policy_state_measurement(
            self.initial_policy_state(),
            simulator_session_id=self.session_id,
            stage_id=self.stage_id,
            source_action_sha256=source_action_sha,
            source_step_index=step,
            captured_at_ns=post_action_captured_at_ns,
            state_snapshot=post_action_state_snapshot,
        )
        after_sample = self._task_joint_sample(task_joint, criterion)
        after = float(after_sample["value_rad"])
        after_timestamp = time.time_ns()
        measurement = {
            "schema_version": "task_transition_measurement.v1",
            "criterion_id": criterion.get("criterion_id"),
            "observable_transition": criterion.get("observable_transition"),
            "before_value": before,
            "after_value": after,
            "episode_initial_value": episode_initial,
            "step_before": before,
            "step_after": after,
            "step_delta": after - before,
            "episode_delta": after - episode_initial,
            "episode_baseline_digest": str(baseline["baseline_digest"]),
            "episode_baseline": dict(baseline),
            "episode_baseline_artifact": dict(self.episode_baseline_artifact),
            "episode_baseline_attestation": dict(self.episode_baseline_attestation or {}),
            "attempt_id": getattr(self, "attempt_id", ""),
            "launch_nonce": getattr(self, "launch_nonce", ""),
            "allocation_launch_session_id": getattr(
                self, "allocation_launch_session_id", ""
            ),
            "qualification_attempt_bound": getattr(
                self, "qualification_attempt_bound", False
            ),
            "qualification_attempt_sequence": getattr(
                self, "qualification_attempt_sequence", None
            ),
            "qualification_attempt_nonce_sha256": getattr(
                self, "qualification_attempt_nonce_sha256", None
            ),
            "unit": criterion.get("unit"),
            "source_step_index": step,
            "evidence_step_index": evidence_step,
            "source_action_sha256": source_action_sha,
            "controller_execution_contract": controller_execution_contract,
            "controller_fk_sequence_sha256": (
                str(state.get("controller_fk_sequence_sha256") or "") or None
            ),
            "controller_horizon_requested_frame_count": requested_control_frame_count,
            "controller_horizon_executed_frame_count": executed_control_frame_count,
            "controller_review_frame_count": sum(
                1
                for row in controller_frame_measurements
                if row.get("review_frame_index") is not None
            ),
            "controller_review_frame_indices": [
                int(row["review_frame_index"])
                for row in controller_frame_measurements
                if row.get("review_frame_index") is not None
            ],
            "controller_terminal_review_frame_index": (
                int(controller_frame_measurements[-1]["review_frame_index"])
                if controller_horizon_terminated_on_semantic_success
                and controller_frame_measurements
                and controller_frame_measurements[-1].get("review_frame_index")
                is not None
                else None
            ),
            "controller_review_sampling": {
                "source_control_hz": GEAR_SONIC_CONTROL_HZ,
                "target_review_hz": ISAAC_REVIEW_SAMPLE_HZ,
                "controller_frame_stride": ISAAC_REVIEW_CONTROLLER_FRAME_STRIDE,
                "semantic_terminal_frame_always_included": True,
            },
            "controller_horizon_fully_executed": (
                executed_control_frame_count == requested_control_frame_count
            ),
            "controller_horizon_terminated_on_semantic_success": (
                controller_horizon_terminated_on_semantic_success
            ),
            "controller_frame_measurements": controller_frame_measurements,
            "controller_execution_started_at_ns": controller_execution_started_at_ns,
            "controller_execution_completed_at_ns": controller_execution_completed_at_ns,
            "controller_execution_wall_seconds": (
                controller_execution_completed_at_ns - controller_execution_started_at_ns
            )
            / 1_000_000_000.0,
            "simulation_control_hz": (
                float(controller_execution_contract.get("control_hz"))
                if explicit_controller_sequence
                else None
            ),
            "one_physics_update_per_controller_frame": explicit_controller_sequence,
            "articulation_prim_path": prim_path,
            "physics_joint_prim_path": task_joint.joint_prim_path,
            "physics_joint_binding": task_joint.to_payload(),
            "measurement_convention": task_joint.measurement_convention,
            "pose_source": after_sample["pose_source"],
            "raw_signed_before_angle_rad": before_sample["raw_signed_angle_rad"],
            "raw_signed_after_angle_rad": after_sample["raw_signed_angle_rad"],
            "bounded_signed_before_angle_rad": before_sample["bounded_signed_angle_rad"],
            "bounded_signed_after_angle_rad": after_sample["bounded_signed_angle_rad"],
            "measurement_backend_source_sha256": getattr(
                self,
                "measurement_backend_source_sha256",
                measurement_backend_source_sha256(),
            ),
            "physics_measurement_surrogate": False,
            "post_action_state_snapshot_payload_sha256": str(
                (post_action_state_snapshot or {}).get("payload_sha256") or ""
            ),
            "post_action_state_snapshot_heartbeat_sequence": (
                (post_action_state_snapshot or {}).get("heartbeat_sequence")
            ),
            "simulator_session_id": self.session_id,
            "stage_id": self.stage_id,
            "before_timestamp": str(before_timestamp),
            "after_timestamp": str(after_timestamp),
        }
        artifact = self.evidence_dir / f"task_measurement_{step:04d}.json"
        artifact.write_text(
            json.dumps(measurement, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        for frame in review_frames:
            frame.update(
                {
                    "source_action_sha256": source_action_sha,
                    "simulator_session_id": self.session_id,
                    "stage_id": self.stage_id,
                    "before_timestamp": str(before_timestamp),
                    "after_timestamp": str(after_timestamp),
                    "attempt_id": getattr(self, "attempt_id", ""),
                    "launch_nonce": getattr(self, "launch_nonce", ""),
                    "allocation_launch_session_id": getattr(
                        self, "allocation_launch_session_id", ""
                    ),
                    "qualification_attempt_bound": getattr(
                        self, "qualification_attempt_bound", False
                    ),
                    "qualification_attempt_sequence": getattr(
                        self, "qualification_attempt_sequence", None
                    ),
                    "qualification_attempt_nonce_sha256": getattr(
                        self, "qualification_attempt_nonce_sha256", None
                    ),
                }
            )
        from .isaac_review_media import record_frame_step_bindings

        record_frame_step_bindings(
            frames_dir=getattr(renderer, "frames_dir", self.evidence_dir / "frames"),
            artifacts=review_frames,
        )
        return {
            **measurement,
            "runtime_result_id": f"{self.session_id}-step-{step:04d}",
            "persistent_simulator_state_applied": True,
            "official_controller_action_applied": True,
            "post_action_policy_state": post_action_policy_state,
            "post_action_state_snapshot": dict(post_action_state_snapshot or {}),
            "simulator_backend": "isaac",
            # The task-transition evaluator parses every evidence_artifact as
            # typed JSON. Review PNGs have their own hash-bound media channel
            # and must never be mixed into the semantic measurement set.
            "evidence_artifacts": [
                {"path": str(artifact), "sha256": artifact_sha},
            ],
            "review_media_artifacts": review_frames,
            "review_frames": review_frames,
            "live_stance_validation": dict(
                getattr(self, "live_geometry_results", {}).get("stance") or {}
            ),
            "live_collision_validation": dict(
                getattr(self, "live_geometry_results", {}).get("collision") or {}
            ),
        }

    def initial_policy_state(self) -> dict[str, Any]:
        """Return attempt-bound proprioception measured from the live articulation."""
        names = list(self.robot.dof_names or [])
        positions = self.robot.get_joint_positions()
        if positions is None or len(names) != len(positions):
            raise RuntimeError("persistent_isaac_initial_proprioception_unavailable")
        observed = [
            {"name": str(name), "position": float(position)}
            for name, position in zip(names, positions, strict=True)
        ]
        resolution = resolve_g1_proprioception_map(observed, require_hands=True)
        if resolution["status"] != "passed":
            raise RuntimeError(
                "persistent_isaac_initial_proprio_mapping_blocked:"
                + ",".join(resolution["blockers"])
            )
        return {
            **resolution["group_values"],
            "projected_gravity": self._live_projected_gravity(),
            "proprioception_mapping": {
                "schema_version": G1_PROPRIOCEPTION_MAP_SCHEMA_VERSION,
                "observed_dof_inventory": resolution["observed_dof_inventory"],
                "resolved_map": resolution["resolved_map"],
                "dimensions": resolution["dimensions"],
                "unmapped_observed_dofs": resolution["unmapped_observed_dofs"],
                "mapping_digest": resolution["mapping_digest"],
            },
            "measurement": {
                "simulator_session_id": self.session_id,
                "stage_id": self.stage_id,
                "source": "live_isaac_articulation_dof_positions_and_base_orientation",
                "surrogate": False,
                "mapping_digest": resolution["mapping_digest"],
            },
        }

    def close(self) -> None:
        self._contact_report_subscription = None
        self.timeline.stop()
        self.app.close()


def create_backend(**kwargs) -> IsaacPersistentTaskBackend:
    return IsaacPersistentTaskBackend(**kwargs)
