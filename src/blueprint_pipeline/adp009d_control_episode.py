"""Deterministic negative and scripted-positive controls for ADP-009D cells.

The scenario suite is policy neutral.  Before either learned candidate may run
on one resolved cell, this module proves two properties through the same native
8D action seam: holding the current joints cannot complete the task, and a
fixed differential-IK pick/place program can.  A failed positive blocks the
cell; it is never counted as a learned-policy failure.

Isaac remains injected.  The plan, sequencing, scoring, retained state/action
trace, and dual-camera evidence are all hermetically testable off GPU.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

try:  # flat provider-bundle layout
    from adp009d_task_scoring import (
        OUTCOME_NEVER_MOVED,
        OUTCOME_PLACED,
        SETTLE_WINDOW_SAMPLES,
        score_task_episode,
    )
except ModuleNotFoundError:  # repository package
    from .adp009d_task_scoring import (
        OUTCOME_NEVER_MOVED,
        OUTCOME_PLACED,
        SETTLE_WINDOW_SAMPLES,
        score_task_episode,
    )
try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest
try:  # flat provider-bundle layout
    from episode_visual_evidence import (
        finalize_manipulation_evaluation_visual_evidence,
        persist_multicamera_observation,
    )
except ModuleNotFoundError:  # repository package
    from .episode_visual_evidence import (
        finalize_manipulation_evaluation_visual_evidence,
        persist_multicamera_observation,
    )
try:  # flat provider-bundle layout
    from adp009d_contact_envelope import (
        ContactEnvelopeError,
        canonical_contact_envelope,
        validate_contact_envelope,
    )
except ModuleNotFoundError:  # repository package
    from .adp009d_contact_envelope import (
        ContactEnvelopeError,
        canonical_contact_envelope,
        validate_contact_envelope,
    )


CONTROL_PLAN_SCHEMA_VERSION = "adp009d_control_plan.v12"
CONTROL_PLAN_FILENAME = f"{CONTROL_PLAN_SCHEMA_VERSION}.json"
CONTROL_EPISODE_SCHEMA_VERSION = "adp009d_control_episode.v4"
CONTROL_PAIR_SCHEMA_VERSION = "adp009d_control_pair.v1"
SCENARIO_INSTANCE_SCHEMA_VERSION = "adp009d_scenario_instance.v1"
ARM_DYNAMICS_OBSERVATION_SCHEMA_VERSION = "adp009d_arm_dynamics_observation.v2"
ARM_DYNAMICS_SUMMARY_SCHEMA_VERSION = "adp009d_arm_dynamics_summary.v2"

ZERO_ACTION_NEGATIVE = "zero_action_negative"
SCRIPTED_POSITIVE = "deterministic_scripted_positive"
REQUIRED_CONTROLS = (ZERO_ACTION_NEGATIVE, SCRIPTED_POSITIVE)

# IK controls an articulation body while the task is grasped at the midpoint
# between the finger bodies.  Those frames are not coincident: v88 proved that
# treating them as one left the fingers 0.39 m from the can while the arm moved
# by 0.81 rad.  The plan therefore targets the semantic grasp frame and the
# live adapter measures the body-to-grasp transform; no asset-specific scalar
# tool offset is allowed here.
GRASP_TARGET_FRAME = "probe_calibrated_finger_midpoint"
CONTROLLED_BODY_ORIENTATION_STRATEGY = "horizontal_support_top_down_task_orientation"
CONTROLLED_BODY_QUATERNION_WORLD_XYZW = [1.0, 0.0, 0.0, 0.0]
PREGRASP_CLEARANCE_ABOVE_SUPPORT_M = 0.42
MAX_JOINT_DELTA_PER_STEP_RAD = 0.03
PHASE_ARRIVAL_TOLERANCE_M = 0.02
PHASE_ORIENTATION_TOLERANCE_DEG = 2.0
# Production v8 proved the 10 mm local Cartesian waypoint preserved lateral
# and orientation coherence, but recomputing an absolute joint target after a
# single environment step let the loaded position actuator settle 119 mm above
# the grasp.  Production v9 raised the waypoint to 30 mm and reached the same
# height while oscillating laterally, falsifying waypoint magnitude as the
# remedy.  Production v10 then proved that a four-step hold must not be applied
# to the direct-global pregrasp phase: it caused that phase to drift past its
# feedback-corrected solution.  Keep per-step feedback for the direct target,
# while holding only bounded-local phase targets long enough for the native
# actuator to track them before differential IK recomputes the next waypoint.
MAX_TASK_SPACE_TRANSLATION_STEP_M = 0.01
DIRECT_TARGET_ACTION_HOLD_STEPS = 1
BOUNDED_LOCAL_ACTION_HOLD_STEPS = 4
DIRECT_GLOBAL_POSE_TARGET = "direct_global_pose_target"
ORIENTATION_FIRST_BOUNDED_LOCAL_INCREMENT = (
    "orientation_first_bounded_local_increment"
)
# The DROID Robotiq 2F-85 is the frozen ADP-009D embodiment.  A generic pose
# tolerance is not enough before descending around an object: the tool can be
# "at" pregrasp while one open finger is already over the object.  Preserve a
# small geometric clearance on each side.  The SDF and finger contact envelope
# is not geometric clearance: it is explicitly subtracted below so a solver
# standoff cannot be mistaken for an admissible open-jaw approach.
GRIPPER_FULL_OPENING_M = 0.085
MOTION_PHASE_MINIMUM_STEPS = 1
MOTION_PHASE_MAXIMUM_STEPS = 240
GRIPPER_DWELL_MINIMUM_STEPS = 30
GRIPPER_DWELL_MAXIMUM_STEPS = 120
PHASE_ARRIVAL_STABILITY_STEPS = 3
ZERO_ACTION_STEPS = 80
# Retain a calibrated overview sequence throughout motion, not only at phase
# boundaries.  Arena advances at roughly 30 Hz, so eight native steps yields a
# human-review stream close to the platform's 4 fps portable-video contract.
CONTROL_REVIEW_FRAME_STRIDE_STEPS = 8
STALL_TRACKING_ERROR_THRESHOLD_RAD = 0.005
STALL_JOINT_VELOCITY_THRESHOLD_RAD_S = 0.002

BLOCKER_ZERO_COMPLETED_TASK = "zero_action_negative_completed_task"
BLOCKER_POSITIVE_FAILED = "deterministic_scripted_positive_failed"
BLOCKER_PHASE_NOT_REACHED = "scripted_control_phase_not_reached"
BLOCKER_MEDIA_INCOMPLETE = "control_episode_media_incomplete"


class ControlEpisodeError(ValueError):
    """Stable fail-closed control-contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


class ControlEnvironment(Protocol):
    """The native simulator surface required by both controls."""

    def reset(self) -> None: ...

    def read_policy_inputs(self) -> Mapping[str, Any]: ...

    def read_evaluation_camera_inputs(self) -> Mapping[str, Any]: ...

    def read_control_observation_metadata(self) -> Mapping[str, Any]: ...

    def read_arm_joint_positions(self) -> Sequence[float]: ...

    def read_arm_dynamics_observation(self) -> Mapping[str, Any]: ...

    def read_object_sample(self) -> Mapping[str, Any]: ...

    def step(self, isaac_action: Sequence[float]) -> None: ...

    def hold_action(self, *, gripper_command: float) -> Sequence[float]: ...

    def scripted_action_for_pose(
        self,
        *,
        target_position_world_m: Sequence[float],
        target_quaternion_world_xyzw: Sequence[float] | None,
        gripper_command: float,
        max_joint_delta_rad: float,
        max_task_space_translation_step_m: float,
        orientation_tolerance_deg: float,
        task_space_translation_strategy: str,
    ) -> Sequence[float]: ...


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ControlEpisodeError(["control_plan_parameter_non_numeric"]) from exc
    if not math.isfinite(number):
        raise ControlEpisodeError(["control_plan_parameter_nonfinite"])
    return number


def _position(parameters: Mapping[str, Any], prefix: str) -> list[float]:
    return [
        _finite(parameters[f"{prefix}_x_m"]),
        _finite(parameters[f"{prefix}_y_m"]),
        _finite(parameters[f"{prefix}_z_m"]),
    ]


def materialize_control_plan(instance: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a fixed native control plan from one digest-bound scenario cell."""

    try:
        value = json.loads(json.dumps(dict(instance), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ControlEpisodeError(["control_plan_instance_not_json_mapping"]) from exc
    errors: list[str] = []
    if value.get("schema_version") != SCENARIO_INSTANCE_SCHEMA_VERSION:
        errors.append("control_plan_instance_schema_invalid")
    if value.get("instance_digest") != canonical_digest(
        value, digest_field="instance_digest"
    ):
        errors.append("control_plan_instance_digest_mismatch")
    if set(value.get("required_controls") or []) != set(REQUIRED_CONTROLS):
        errors.append("control_plan_required_controls_invalid")
    if value.get("policy_neutral") is not True:
        errors.append("control_plan_instance_not_policy_neutral")
    parameters = value.get("resolved_parameters")
    if not isinstance(parameters, Mapping):
        errors.append("control_plan_resolved_parameters_missing")
    if errors:
        raise ControlEpisodeError(errors)

    assert isinstance(parameters, Mapping)
    start = _position(parameters, "object_start")
    target = _position(parameters, "target")
    object_height = _finite(parameters.get("object_height_m"))
    if "object_radius_m" not in parameters:
        raise ControlEpisodeError(["control_plan_object_radius_missing"])
    object_radius = _finite(parameters.get("object_radius_m"))
    if object_height <= 0.0:
        raise ControlEpisodeError(["control_plan_object_height_invalid"])
    if object_radius <= 0.0:
        raise ControlEpisodeError(["control_plan_object_radius_invalid"])
    contact_envelope = canonical_contact_envelope()
    open_jaw_radial_clearance = GRIPPER_FULL_OPENING_M / 2.0 - object_radius
    effective_contact_envelope = float(
        contact_envelope["effective_contact_envelope_m"]
    )
    open_jaw_effective_radial_clearance = (
        open_jaw_radial_clearance - effective_contact_envelope
    )
    if open_jaw_effective_radial_clearance <= 0.0:
        raise ControlEpisodeError(
            ["control_plan_object_open_jaw_effective_clearance_insufficient"]
        )
    aperture_safe_arrival_tolerance = min(
        PHASE_ARRIVAL_TOLERANCE_M,
        open_jaw_effective_radial_clearance,
    )
    grasp_frame_z = start[2] + object_height / 2.0
    place_frame_z = target[2] + object_height / 2.0
    phases = [
        {
            "phase_id": "pregrasp",
            "mode": "ik_pose",
            "target_position_world_m": [
                start[0],
                start[1],
                start[2] + PREGRASP_CLEARANCE_ABOVE_SUPPORT_M,
            ],
            "gripper": "open",
            "minimum_steps": MOTION_PHASE_MINIMUM_STEPS,
            "maximum_steps": MOTION_PHASE_MAXIMUM_STEPS,
        },
        {
            "phase_id": "descend",
            "mode": "ik_pose",
            "target_position_world_m": [start[0], start[1], grasp_frame_z],
            "gripper": "open",
            "minimum_steps": MOTION_PHASE_MINIMUM_STEPS,
            "maximum_steps": MOTION_PHASE_MAXIMUM_STEPS,
        },
        {
            "phase_id": "grasp",
            "mode": "ik_pose",
            "target_position_world_m": [start[0], start[1], grasp_frame_z],
            "gripper": "closed",
            "minimum_steps": GRIPPER_DWELL_MINIMUM_STEPS,
            "maximum_steps": GRIPPER_DWELL_MAXIMUM_STEPS,
        },
        {
            "phase_id": "lift",
            "mode": "ik_pose",
            "target_position_world_m": [
                start[0],
                start[1],
                start[2] + PREGRASP_CLEARANCE_ABOVE_SUPPORT_M,
            ],
            "gripper": "closed",
            "minimum_steps": MOTION_PHASE_MINIMUM_STEPS,
            "maximum_steps": MOTION_PHASE_MAXIMUM_STEPS,
        },
        {
            "phase_id": "transport",
            "mode": "ik_pose",
            "target_position_world_m": [
                target[0],
                target[1],
                target[2] + PREGRASP_CLEARANCE_ABOVE_SUPPORT_M,
            ],
            "gripper": "closed",
            "minimum_steps": MOTION_PHASE_MINIMUM_STEPS,
            "maximum_steps": MOTION_PHASE_MAXIMUM_STEPS,
        },
        {
            "phase_id": "place",
            "mode": "ik_pose",
            "target_position_world_m": [target[0], target[1], place_frame_z],
            "gripper": "closed",
            "minimum_steps": MOTION_PHASE_MINIMUM_STEPS,
            "maximum_steps": MOTION_PHASE_MAXIMUM_STEPS,
        },
        {
            "phase_id": "release",
            "mode": "ik_pose",
            "target_position_world_m": [target[0], target[1], place_frame_z],
            "gripper": "open",
            "minimum_steps": GRIPPER_DWELL_MINIMUM_STEPS,
            "maximum_steps": GRIPPER_DWELL_MAXIMUM_STEPS,
        },
        {
            "phase_id": "retreat",
            "mode": "ik_pose",
            "target_position_world_m": [
                target[0],
                target[1],
                target[2] + PREGRASP_CLEARANCE_ABOVE_SUPPORT_M,
            ],
            "gripper": "open",
            "minimum_steps": MOTION_PHASE_MINIMUM_STEPS,
            "maximum_steps": MOTION_PHASE_MAXIMUM_STEPS,
        },
        {
            "phase_id": "settle",
            "mode": "hold_current_joint_positions",
            "target_position_world_m": None,
            "gripper": "open",
            "steps": SETTLE_WINDOW_SAMPLES,
        },
    ]
    for phase in phases:
        if phase["target_position_world_m"] is not None:
            phase["target_frame"] = GRASP_TARGET_FRAME
            phase["orientation_strategy"] = CONTROLLED_BODY_ORIENTATION_STRATEGY
            phase["target_quaternion_world_xyzw"] = list(
                CONTROLLED_BODY_QUATERNION_WORLD_XYZW
            )
            if phase["phase_id"] in {"pregrasp", "descend", "grasp"}:
                phase["arrival_tolerance_m"] = aperture_safe_arrival_tolerance
                phase["arrival_tolerance_basis"] = (
                    "open_jaw_radial_clearance_minus_effective_contact_envelope"
                )
            else:
                phase["arrival_tolerance_m"] = PHASE_ARRIVAL_TOLERANCE_M
                phase["arrival_tolerance_basis"] = "generic_pose_tolerance"
            phase["arrival_stability_steps"] = PHASE_ARRIVAL_STABILITY_STEPS
            phase["orientation_tolerance_deg"] = PHASE_ORIENTATION_TOLERANCE_DEG
            phase["orientation_tolerance_basis"] = (
                "top_down_task_orientation_angular_distance"
            )
            phase["max_task_space_translation_step_m"] = (
                MAX_TASK_SPACE_TRANSLATION_STEP_M
            )
            phase["task_space_translation_strategy"] = (
                DIRECT_GLOBAL_POSE_TARGET
                if phase["phase_id"] == "pregrasp"
                else ORIENTATION_FIRST_BOUNDED_LOCAL_INCREMENT
            )
            phase["action_hold_steps"] = (
                DIRECT_TARGET_ACTION_HOLD_STEPS
                if phase["task_space_translation_strategy"]
                == DIRECT_GLOBAL_POSE_TARGET
                else BOUNDED_LOCAL_ACTION_HOLD_STEPS
            )
        phase["max_joint_delta_rad"] = MAX_JOINT_DELTA_PER_STEP_RAD

    plan: dict[str, Any] = {
        "schema_version": CONTROL_PLAN_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "instance_digest": value["instance_digest"],
        "suite_digest": value.get("suite_digest"),
        "cell_id": value.get("cell_id"),
        "family": value.get("family"),
        "seed": value.get("seed"),
        "resolved_start_position_world_m": start,
        "resolved_destination_position_world_m": target,
        "object_height_m": object_height,
        "object_radius_m": object_radius,
        "contact_envelope": contact_envelope,
        "open_gripper_geometry": {
            "full_opening_m": GRIPPER_FULL_OPENING_M,
            "object_diameter_m": object_radius * 2.0,
            "radial_clearance_m": open_jaw_radial_clearance,
            "effective_contact_envelope_m": effective_contact_envelope,
            "radial_clearance_after_effective_contact_envelope_m": (
                open_jaw_effective_radial_clearance
            ),
            "clearance_accounting": (
                "radial_clearance_m_minus_effective_contact_envelope_m"
            ),
            "aperture_safe_arrival_tolerance_m": (
                aperture_safe_arrival_tolerance
            ),
        },
        "grasp_target_frame": GRASP_TARGET_FRAME,
        "controlled_body_orientation_strategy": CONTROLLED_BODY_ORIENTATION_STRATEGY,
        "controlled_body_quaternion_world_xyzw": list(
            CONTROLLED_BODY_QUATERNION_WORLD_XYZW
        ),
        "zero_action": {
            "semantics": (
                "zero_joint_velocity_realized_as_hold_current_absolute_joint_positions"
            ),
            "gripper": "open",
            "steps": ZERO_ACTION_STEPS,
        },
        "scripted_positive_phases": phases,
        "caller_asserted_success_accepted": False,
        "candidate_policy_queried": False,
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def _sample(environment: ControlEnvironment, step_index: int) -> dict[str, Any]:
    sample = dict(environment.read_object_sample())
    if "can_pose_world" not in sample:
        raise ControlEpisodeError(["control_episode_can_pose_world_missing"])
    sample["step_index"] = int(step_index)
    return sample


def _phase_arrival(
    *,
    phase: Mapping[str, Any],
    start_sample: Mapping[str, Any],
    terminal_sample: Mapping[str, Any],
    steps_executed: int,
    stability_steps_observed: int,
    termination_reason: str,
) -> dict[str, Any]:
    """Retain and gate the semantic grasp-frame error for one IK phase."""

    target = [float(value) for value in phase["target_position_world_m"]]
    start = [float(value) for value in start_sample["grasp_frame_position_world_m"]]
    achieved = [
        float(value) for value in terminal_sample["grasp_frame_position_world_m"]
    ]
    tolerance = float(phase["arrival_tolerance_m"])
    error = math.dist(achieved, target)
    lateral_error = math.dist(achieved[:2], target[:2])
    orientation_error_deg = _orientation_error_degrees(
        achieved_quaternion_xyzw=_controlled_body_quaternion(terminal_sample),
        target_quaternion_xyzw=phase["target_quaternion_world_xyzw"],
    )
    orientation_tolerance_deg = float(phase["orientation_tolerance_deg"])
    stability_steps_required = int(phase["arrival_stability_steps"])
    return {
        "phase_id": str(phase["phase_id"]),
        "target_frame": str(phase["target_frame"]),
        "target_position_world_m": target,
        "start_position_world_m": start,
        "achieved_position_world_m": achieved,
        "terminal_position_error_m": error,
        "terminal_lateral_error_m": lateral_error,
        "arrival_tolerance_m": tolerance,
        "arrival_tolerance_basis": str(phase["arrival_tolerance_basis"]),
        "terminal_position_within_tolerance": error <= tolerance,
        "terminal_orientation_error_deg": orientation_error_deg,
        "orientation_tolerance_deg": orientation_tolerance_deg,
        "orientation_tolerance_basis": str(
            phase["orientation_tolerance_basis"]
        ),
        "terminal_orientation_within_tolerance": (
            orientation_error_deg <= orientation_tolerance_deg
        ),
        "terminal_within_tolerance": (
            error <= tolerance
            and orientation_error_deg <= orientation_tolerance_deg
        ),
        "minimum_steps": int(phase["minimum_steps"]),
        "maximum_steps": int(phase["maximum_steps"]),
        "steps_executed": int(steps_executed),
        "arrival_stability_steps_required": stability_steps_required,
        "arrival_stability_steps_observed": int(stability_steps_observed),
        "termination_reason": str(termination_reason),
        "target_reached": termination_reason == "stable_arrival",
    }


def _within_phase_arrival_tolerance(
    *, phase: Mapping[str, Any], sample: Mapping[str, Any]
) -> bool:
    target = [float(value) for value in phase["target_position_world_m"]]
    achieved = [
        float(value) for value in sample["grasp_frame_position_world_m"]
    ]
    orientation_error_deg = _orientation_error_degrees(
        achieved_quaternion_xyzw=_controlled_body_quaternion(sample),
        target_quaternion_xyzw=phase["target_quaternion_world_xyzw"],
    )
    return (
        math.dist(achieved, target) <= float(phase["arrival_tolerance_m"])
        and orientation_error_deg <= float(phase["orientation_tolerance_deg"])
    )


def _controlled_body_quaternion(sample: Mapping[str, Any]) -> list[float]:
    pose = sample.get("controlled_body_pose_world")
    if not isinstance(pose, Sequence) or isinstance(pose, (str, bytes)):
        raise ControlEpisodeError(["control_episode_controlled_body_pose_missing"])
    try:
        values = [float(value) for value in pose]
    except (TypeError, ValueError) as exc:
        raise ControlEpisodeError(
            ["control_episode_controlled_body_pose_invalid"]
        ) from exc
    if len(values) != 7 or not all(math.isfinite(value) for value in values):
        raise ControlEpisodeError(["control_episode_controlled_body_pose_invalid"])
    quaternion = values[3:7]
    if abs(math.sqrt(sum(value * value for value in quaternion)) - 1.0) > 1.0e-5:
        raise ControlEpisodeError(["control_episode_controlled_body_pose_invalid"])
    return quaternion


def _orientation_error_degrees(
    *,
    achieved_quaternion_xyzw: Sequence[float],
    target_quaternion_xyzw: Sequence[float],
) -> float:
    """Return the shortest angular distance between equivalent quaternions."""

    try:
        achieved = [float(value) for value in achieved_quaternion_xyzw]
        target = [float(value) for value in target_quaternion_xyzw]
    except (TypeError, ValueError) as exc:
        raise ControlEpisodeError(
            ["control_episode_target_orientation_invalid"]
        ) from exc
    if (
        len(achieved) != 4
        or len(target) != 4
        or not all(math.isfinite(value) for value in (*achieved, *target))
        or abs(math.sqrt(sum(value * value for value in achieved)) - 1.0) > 1.0e-5
        or abs(math.sqrt(sum(value * value for value in target)) - 1.0) > 1.0e-5
    ):
        raise ControlEpisodeError(["control_episode_target_orientation_invalid"])
    dot = abs(sum(a * b for a, b in zip(achieved, target, strict=True)))
    dot = min(1.0, max(0.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _persist_observation(
    environment: ControlEnvironment,
    *,
    output_dir: Path,
    episode_id: str,
    observation_index: int,
    kind: str,
) -> dict[str, Any]:
    images = dict(environment.read_evaluation_camera_inputs())
    missing = {"external", "wrist", "overview"} - set(images)
    if missing:
        raise ControlEpisodeError(
            [f"control_episode_required_camera_missing:{camera_id}" for camera_id in missing]
        )
    metadata = dict(environment.read_control_observation_metadata())
    return persist_multicamera_observation(
        images,
        output_dir=output_dir,
        episode_id=episode_id,
        observation_index=observation_index,
        kind=kind,
        timestamp_ns=int(metadata["timestamp_ns"]),
        simulation_time_s=float(metadata["simulation_time_s"]),
        calibrations=metadata["calibrations"],
        source_devices=metadata["source_devices"],
        synchronizations=metadata["synchronizations"],
    )


def _record_action(
    *,
    step_index: int,
    phase_id: str,
    action: Sequence[float],
    observed_before: Sequence[float],
    observed_after: Sequence[float],
    dynamics_before: Mapping[str, Any],
    dynamics_after: Mapping[str, Any],
    action_recomputed: bool,
    action_hold_index: int,
) -> dict[str, Any]:
    values = [float(value) for value in action]
    if len(values) != 8 or not all(math.isfinite(value) for value in values):
        raise ControlEpisodeError(["control_episode_action_invalid"])
    return {
        "step_index": int(step_index),
        "phase_id": phase_id,
        "isaac_action": values,
        "action_recomputed": bool(action_recomputed),
        "action_hold_index": int(action_hold_index),
        "observed_joint_position_before_rad": [float(v) for v in observed_before],
        "observed_joint_position_after_rad": [float(v) for v in observed_after],
        "arm_dynamics_before": _canonical_dynamics_observation(dynamics_before),
        "arm_dynamics_after": _canonical_dynamics_observation(dynamics_after),
    }


def _canonical_dynamics_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Retain finite simulator readback without accepting caller-authored gaps."""

    if not isinstance(value, Mapping):
        raise ControlEpisodeError(["control_episode_arm_dynamics_not_mapping"])
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ControlEpisodeError(["control_episode_arm_dynamics_invalid"]) from exc
    if result.get("schema_version") != ARM_DYNAMICS_OBSERVATION_SCHEMA_VERSION:
        raise ControlEpisodeError(["control_episode_arm_dynamics_schema_invalid"])
    required = {
        "joint_position_rad",
        "joint_velocity_rad_s",
        "joint_position_target_rad",
        "computed_torque_nm",
        "applied_torque_nm",
        "joint_effort_limit_nm",
        "joint_effort_utilization",
        "body_contact_force_world_n",
        "body_incoming_joint_wrench_body",
        "contact_envelope",
    }
    if required - set(result):
        raise ControlEpisodeError(["control_episode_arm_dynamics_field_missing"])
    vector_fields = (
        "joint_position_rad",
        "joint_velocity_rad_s",
        "joint_position_target_rad",
        "computed_torque_nm",
        "applied_torque_nm",
        "joint_effort_limit_nm",
        "joint_effort_utilization",
    )
    if any(
        not isinstance(result.get(field), list) or len(result[field]) != 7
        for field in vector_fields
    ):
        raise ControlEpisodeError(["control_episode_arm_dynamics_vector_invalid"])
    for field, width, allow_none in (
        ("body_contact_force_world_n", 3, True),
        ("body_contact_partner_force_world_n", 3, True),
        ("body_contact_sage_collision_force_world_n", 3, True),
        ("body_incoming_joint_wrench_body", 6, False),
    ):
        body_values = result.get(field)
        if body_values is None and allow_none:
            continue
        if not isinstance(body_values, dict) or any(
            not isinstance(vector, list) or len(vector) != width
            for vector in body_values.values()
        ):
            raise ControlEpisodeError(["control_episode_arm_dynamics_body_vector_invalid"])
    try:
        result["contact_envelope"] = validate_contact_envelope(
            result["contact_envelope"]
        )
    except ContactEnvelopeError as exc:
        raise ControlEpisodeError([str(exc)]) from exc
    return result


def _plan_contact_envelope(plan: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return validate_contact_envelope(plan.get("contact_envelope"))
    except ContactEnvelopeError as exc:
        raise ControlEpisodeError([str(exc)]) from exc


def _require_dynamics_contact_envelope(
    dynamics: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> None:
    if dynamics.get("contact_envelope") != dict(expected):
        raise ControlEpisodeError(["control_episode_contact_envelope_plan_mismatch"])


def _vector_norm(values: Sequence[Any]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _summarize_arm_dynamics(actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize tracking, saturation, and contact without declaring a cause."""

    phases: dict[str, dict[str, Any]] = {}
    contact_envelope: dict[str, Any] | None = None
    for action in actions:
        phase_id = str(action["phase_id"])
        dynamics = dict(action["arm_dynamics_after"])
        observed_contact_envelope = _canonical_dynamics_observation(dynamics)[
            "contact_envelope"
        ]
        if contact_envelope is None:
            contact_envelope = observed_contact_envelope
        elif observed_contact_envelope != contact_envelope:
            raise ControlEpisodeError(["control_episode_contact_envelope_drifted"])
        positions = dynamics["joint_position_rad"]
        targets = dynamics["joint_position_target_rad"]
        velocities = dynamics["joint_velocity_rad_s"]
        tracking_error = max(
            abs(float(target) - float(position))
            for target, position in zip(targets, positions, strict=True)
        )
        maximum_velocity = max(abs(float(value)) for value in velocities)
        maximum_effort_utilization = max(
            abs(float(value)) for value in dynamics["joint_effort_utilization"]
        )
        maximum_clip_residual = max(
            (
                abs(float(value))
                for value in dynamics.get("torque_clip_residual_nm", [])
            ),
            default=0.0,
        )
        contact_forces = dynamics.get("body_contact_force_world_n") or {}
        contact_magnitudes = {
            str(name): _vector_norm(vector)
            for name, vector in contact_forces.items()
        }
        peak_contact_body = (
            max(contact_magnitudes, key=contact_magnitudes.get)
            if contact_magnitudes
            else None
        )
        maximum_contact_force = (
            contact_magnitudes[peak_contact_body]
            if peak_contact_body is not None
            else 0.0
        )
        incoming_wrenches = dynamics.get("body_incoming_joint_wrench_body") or {}
        maximum_incoming_force = max(
            (_vector_norm(wrench[:3]) for wrench in incoming_wrenches.values()),
            default=0.0,
        )
        partner_forces = dynamics.get("body_contact_partner_force_world_n")
        partner_available = partner_forces is not None
        maximum_partner_force = max(
            (_vector_norm(vector) for vector in (partner_forces or {}).values()),
            default=0.0,
        )
        sage_collision_forces = dynamics.get(
            "body_contact_sage_collision_force_world_n"
        )
        sage_collision_available = sage_collision_forces is not None
        maximum_sage_collision_force = max(
            (
                _vector_norm(vector)
                for vector in (sage_collision_forces or {}).values()
            ),
            default=0.0,
        )
        # These filters have separate, non-overlapping scopes.  Preserve the
        # per-scope force as well as the total accounted force: one scope cannot
        # make a zero in the other mean anything, and an unresolved filter is
        # deliberately represented as unavailable rather than as zero.
        accounted_force_by_body: dict[str, float] = {}
        for filtered_forces in (partner_forces or {}, sage_collision_forces or {}):
            for body_name, vector in filtered_forces.items():
                accounted_force_by_body[str(body_name)] = (
                    accounted_force_by_body.get(str(body_name), 0.0)
                    + _vector_norm(vector)
                )
        maximum_accounted_filtered_force = max(
            accounted_force_by_body.values(), default=0.0
        )
        # Force the explicit filters do not account for.  A large residual names
        # a non-can/non-SAGE contact investigation; it never asserts which prim
        # is responsible.
        unattributed_contact_force = max(
            maximum_contact_force - maximum_accounted_filtered_force, 0.0
        )
        phase = phases.setdefault(
            phase_id,
            {
                "step_count": 0,
                "stalled_tracking_step_count": 0,
                "maximum_joint_position_tracking_error_rad": 0.0,
                "maximum_absolute_joint_velocity_rad_s": 0.0,
                "maximum_joint_effort_utilization": 0.0,
                "maximum_torque_clip_residual_nm": 0.0,
                "maximum_body_contact_force_n": 0.0,
                "peak_contact_body": None,
                "maximum_incoming_joint_force_n": 0.0,
                "contact_partner_matrix_available": partner_available,
                "contact_sage_collision_matrix_available": sage_collision_available,
                "maximum_filtered_partner_contact_force_n": 0.0,
                "maximum_sage_collision_contact_force_n": 0.0,
                "maximum_accounted_filtered_contact_force_n": 0.0,
                "maximum_unattributed_contact_force_n": 0.0,
                "final": None,
            },
        )
        phase["step_count"] += 1
        if (
            tracking_error >= STALL_TRACKING_ERROR_THRESHOLD_RAD
            and maximum_velocity <= STALL_JOINT_VELOCITY_THRESHOLD_RAD_S
        ):
            phase["stalled_tracking_step_count"] += 1
        phase["maximum_joint_position_tracking_error_rad"] = max(
            phase["maximum_joint_position_tracking_error_rad"], tracking_error
        )
        phase["maximum_absolute_joint_velocity_rad_s"] = max(
            phase["maximum_absolute_joint_velocity_rad_s"], maximum_velocity
        )
        phase["maximum_joint_effort_utilization"] = max(
            phase["maximum_joint_effort_utilization"], maximum_effort_utilization
        )
        phase["maximum_torque_clip_residual_nm"] = max(
            phase["maximum_torque_clip_residual_nm"], maximum_clip_residual
        )
        if maximum_contact_force > phase["maximum_body_contact_force_n"]:
            phase["maximum_body_contact_force_n"] = maximum_contact_force
            phase["peak_contact_body"] = peak_contact_body
        phase["maximum_incoming_joint_force_n"] = max(
            phase["maximum_incoming_joint_force_n"], maximum_incoming_force
        )
        phase["contact_partner_matrix_available"] = (
            phase["contact_partner_matrix_available"] and partner_available
        )
        phase["contact_sage_collision_matrix_available"] = (
            phase["contact_sage_collision_matrix_available"]
            and sage_collision_available
        )
        phase["maximum_filtered_partner_contact_force_n"] = max(
            phase["maximum_filtered_partner_contact_force_n"], maximum_partner_force
        )
        phase["maximum_sage_collision_contact_force_n"] = max(
            phase["maximum_sage_collision_contact_force_n"],
            maximum_sage_collision_force,
        )
        phase["maximum_accounted_filtered_contact_force_n"] = max(
            phase["maximum_accounted_filtered_contact_force_n"],
            maximum_accounted_filtered_force,
        )
        phase["maximum_unattributed_contact_force_n"] = max(
            phase["maximum_unattributed_contact_force_n"], unattributed_contact_force
        )
        phase["final"] = {
            "step_index": int(action["step_index"]),
            "joint_position_tracking_error_rad": tracking_error,
            "maximum_absolute_joint_velocity_rad_s": maximum_velocity,
            "maximum_joint_effort_utilization": maximum_effort_utilization,
            "maximum_body_contact_force_n": maximum_contact_force,
            "peak_contact_body": peak_contact_body,
            "filtered_partner_contact_force_n": maximum_partner_force,
            "sage_collision_contact_force_n": maximum_sage_collision_force,
            "accounted_filtered_contact_force_n": maximum_accounted_filtered_force,
            "unattributed_contact_force_n": unattributed_contact_force,
        }
    if contact_envelope is None:
        raise ControlEpisodeError(["control_episode_arm_dynamics_missing"])
    return {
        "schema_version": ARM_DYNAMICS_SUMMARY_SCHEMA_VERSION,
        "contact_envelope": contact_envelope,
        "stall_tracking_error_threshold_rad": STALL_TRACKING_ERROR_THRESHOLD_RAD,
        "stall_joint_velocity_threshold_rad_s": (
            STALL_JOINT_VELOCITY_THRESHOLD_RAD_S
        ),
        "phases": phases,
        "claim_boundary": (
            "Readback distinguishes tracking, saturation, and contact evidence; "
            "it does not by itself assign root cause or task success."
        ),
    }


def run_control_episode(
    *,
    environment: ControlEnvironment,
    plan: Mapping[str, Any],
    control_id: str,
    gripper_open_command: float,
    gripper_closed_command: float,
    media_output_dir: str | Path,
    episode_id: str,
) -> dict[str, Any]:
    """Execute one control and seal its deterministic evidence."""

    if control_id not in REQUIRED_CONTROLS:
        raise ControlEpisodeError([f"control_episode_control_unknown:{control_id}"])
    if plan.get("schema_version") != CONTROL_PLAN_SCHEMA_VERSION:
        raise ControlEpisodeError(["control_episode_plan_schema_invalid"])
    if plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest"):
        raise ControlEpisodeError(["control_episode_plan_digest_mismatch"])
    plan_contact_envelope = _plan_contact_envelope(plan)
    output = Path(media_output_dir).expanduser().resolve()
    if not episode_id.strip():
        raise ControlEpisodeError(["control_episode_id_missing"])

    environment.reset()
    initial_arm_dynamics = _canonical_dynamics_observation(
        environment.read_arm_dynamics_observation()
    )
    _require_dynamics_contact_envelope(
        initial_arm_dynamics,
        expected=plan_contact_envelope,
    )
    samples = [_sample(environment, 0)]
    actions: list[dict[str, Any]] = []
    policy_inputs: list[dict[str, Any]] = [
        _persist_observation(
            environment,
            output_dir=output,
            episode_id=episode_id,
            observation_index=0,
            kind="policy-input",
        )
    ]
    review_observations: list[dict[str, Any]] = []
    observation_index = 1
    step_index = 0
    phase_arrivals: list[dict[str, Any]] = []
    phase_execution_blocker: str | None = None

    if control_id == ZERO_ACTION_NEGATIVE:
        phases = [
            {
                "phase_id": ZERO_ACTION_NEGATIVE,
                "mode": "hold_current_joint_positions",
                "gripper": "open",
                "steps": int(plan["zero_action"]["steps"]),
                "target_position_world_m": None,
                "target_quaternion_world_xyzw": None,
                "max_joint_delta_rad": MAX_JOINT_DELTA_PER_STEP_RAD,
            }
        ]
    else:
        phases = [dict(phase) for phase in plan["scripted_positive_phases"]]

    for phase_index, phase in enumerate(phases):
        phase_start_sample = samples[-1]
        phase_steps_executed = 0
        stability_steps_observed = 0
        termination_reason = "fixed_steps_completed"
        gripper_command = (
            float(gripper_open_command)
            if phase["gripper"] == "open"
            else float(gripper_closed_command)
        )
        phase_step_limit = int(
            phase.get("maximum_steps", phase.get("steps", 0))
        )
        held_action: Sequence[float] | None = None
        for phase_step_index in range(phase_step_limit):
            before = [float(v) for v in environment.read_arm_joint_positions()]
            dynamics_before = _canonical_dynamics_observation(
                environment.read_arm_dynamics_observation()
            )
            _require_dynamics_contact_envelope(
                dynamics_before,
                expected=plan_contact_envelope,
            )
            if phase["mode"] == "hold_current_joint_positions":
                action = environment.hold_action(gripper_command=gripper_command)
                action_recomputed = True
                action_hold_index = 0
            else:
                action_hold_steps = int(phase["action_hold_steps"])
                action_hold_index = phase_step_index % action_hold_steps
                action_recomputed = held_action is None or action_hold_index == 0
                if action_recomputed:
                    held_action = environment.scripted_action_for_pose(
                        target_position_world_m=phase["target_position_world_m"],
                        target_quaternion_world_xyzw=phase[
                            "target_quaternion_world_xyzw"
                        ],
                        gripper_command=gripper_command,
                        max_joint_delta_rad=float(phase["max_joint_delta_rad"]),
                        max_task_space_translation_step_m=float(
                            phase["max_task_space_translation_step_m"]
                        ),
                        orientation_tolerance_deg=float(
                            phase["orientation_tolerance_deg"]
                        ),
                        task_space_translation_strategy=str(
                            phase["task_space_translation_strategy"]
                        ),
                    )
                assert held_action is not None
                action = held_action
            environment.step(action)
            step_index += 1
            phase_steps_executed += 1
            after = [float(v) for v in environment.read_arm_joint_positions()]
            dynamics_after = _canonical_dynamics_observation(
                environment.read_arm_dynamics_observation()
            )
            _require_dynamics_contact_envelope(
                dynamics_after,
                expected=plan_contact_envelope,
            )
            actions.append(
                _record_action(
                    step_index=step_index,
                    phase_id=str(phase["phase_id"]),
                    action=action,
                    observed_before=before,
                    observed_after=after,
                    dynamics_before=dynamics_before,
                    dynamics_after=dynamics_after,
                    action_recomputed=action_recomputed,
                    action_hold_index=action_hold_index,
                )
            )
            samples.append(_sample(environment, step_index))
            if phase["mode"] == "ik_pose":
                if _within_phase_arrival_tolerance(
                    phase=phase, sample=samples[-1]
                ):
                    stability_steps_observed += 1
                else:
                    stability_steps_observed = 0
            if step_index % CONTROL_REVIEW_FRAME_STRIDE_STEPS == 0:
                review_observations.append(
                    _persist_observation(
                        environment,
                        output_dir=output,
                        episode_id=episode_id,
                        observation_index=observation_index,
                        kind="review-sample",
                    )
                )
                observation_index += 1
            if (
                phase["mode"] == "ik_pose"
                and phase_steps_executed >= int(phase["minimum_steps"])
                and stability_steps_observed
                >= int(phase["arrival_stability_steps"])
            ):
                termination_reason = "stable_arrival"
                break
        if phase["mode"] == "ik_pose" and termination_reason != "stable_arrival":
            termination_reason = "maximum_steps_exhausted"
        if (
            phase_index < len(phases) - 1
            and step_index % CONTROL_REVIEW_FRAME_STRIDE_STEPS != 0
        ):
            review_observations.append(
                _persist_observation(
                    environment,
                    output_dir=output,
                    episode_id=episode_id,
                    observation_index=observation_index,
                    kind="review-sample",
                )
            )
            observation_index += 1
        if phase["mode"] == "ik_pose":
            arrival = _phase_arrival(
                phase=phase,
                start_sample=phase_start_sample,
                terminal_sample=samples[-1],
                steps_executed=phase_steps_executed,
                stability_steps_observed=stability_steps_observed,
                termination_reason=termination_reason,
            )
            phase_arrivals.append(arrival)
            if not arrival["target_reached"]:
                phase_execution_blocker = (
                    f"{BLOCKER_PHASE_NOT_REACHED}:{phase['phase_id']}:"
                    f"error_m={arrival['terminal_position_error_m']:.6f}:"
                    "lateral_error_m="
                    f"{arrival['terminal_lateral_error_m']:.6f}:"
                    f"tolerance_m={arrival['arrival_tolerance_m']:.6f}:"
                    "orientation_error_deg="
                    f"{arrival['terminal_orientation_error_deg']:.6f}:"
                    "orientation_tolerance_deg="
                    f"{arrival['orientation_tolerance_deg']:.6f}:"
                    "stability_steps="
                    f"{arrival['arrival_stability_steps_observed']}/"
                    f"{arrival['arrival_stability_steps_required']}"
                )
                break

    terminal = _persist_observation(
        environment,
        output_dir=output,
        episode_id=episode_id,
        observation_index=observation_index,
        kind="terminal-observation",
    )
    visual, artifacts = finalize_manipulation_evaluation_visual_evidence(
        output_dir=output,
        episode_id=episode_id,
        identity={
            "control_id": control_id,
            "instance_digest": plan["instance_digest"],
            "control_plan_digest": plan["plan_digest"],
            "candidate_policy_queried": False,
            "observation_consumer": "deterministic_control_monitor",
        },
        policy_input_observations=policy_inputs,
        review_observations=review_observations,
        terminal_observation=terminal,
    )
    score = score_task_episode(
        samples=samples,
        destination_position_world_m=plan["resolved_destination_position_world_m"],
        settle_window_samples=SETTLE_WINDOW_SAMPLES,
        require_sealed_start_pose=(
            plan.get("family") == "canonical"
            or plan["resolved_start_position_world_m"]
            == [3.4681748, -3.3100837, 0.5264650138348479]
        ),
    )
    if control_id == ZERO_ACTION_NEGATIVE:
        passed = score.get("status") == "scored" and score.get("task_succeeded") is False
        blockers = [] if passed else [BLOCKER_ZERO_COMPLETED_TASK]
    else:
        passed = score.get("status") == "scored" and score.get("outcome") == OUTCOME_PLACED
        blockers = (
            []
            if passed
            else [f"{BLOCKER_POSITIVE_FAILED}:{score.get('outcome')}"]
        )
        if phase_execution_blocker is not None:
            blockers.append(phase_execution_blocker)
            passed = False
    if visual.get("status") != "complete":
        blockers.append(BLOCKER_MEDIA_INCOMPLETE)
        passed = False

    receipt: dict[str, Any] = {
        "schema_version": CONTROL_EPISODE_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "control_id": control_id,
        "episode_id": episode_id,
        "instance_digest": plan["instance_digest"],
        "control_plan_digest": plan["plan_digest"],
        "contact_envelope": plan_contact_envelope,
        "initial_arm_dynamics": initial_arm_dynamics,
        "control_passed": passed,
        "blockers": sorted(set(blockers)),
        "environment_steps": step_index,
        "score": score,
        "observed_outcome": score.get("outcome"),
        "expected_outcome": (
            "task_failure" if control_id == ZERO_ACTION_NEGATIVE else OUTCOME_PLACED
        ),
        "zero_action_remained_never_moved": (
            score.get("outcome") == OUTCOME_NEVER_MOVED
            if control_id == ZERO_ACTION_NEGATIVE
            else None
        ),
        "state_trace": samples,
        "state_trace_digest": canonical_digest({"samples": samples}),
        "action_trace": actions,
        "action_trace_digest": canonical_digest({"actions": actions}),
        "arm_dynamics_summary": _summarize_arm_dynamics(actions),
        "phase_arrivals": phase_arrivals,
        "phase_execution_blocker": phase_execution_blocker,
        "contact_trace": [
            {
                "step_index": sample["step_index"],
                "finger_contact_forces_n": sample.get("finger_contact_forces_n"),
                "gripper_width_m": sample.get("gripper_width_m"),
            }
            for sample in samples
        ],
        "contact_sensor_gap": (
            None
            if any(sample.get("finger_contact_forces_n") is not None for sample in samples)
            else "native_finger_contact_sensor_not_instrumented_grasp_inferred_from_width_and_lift"
        ),
        "visual_evidence": visual,
        "media_artifacts": artifacts,
        "review_sampling": {
            "stride_environment_steps": CONTROL_REVIEW_FRAME_STRIDE_STEPS,
            "review_observation_count": len(review_observations),
            "full_motion_temporal_coverage": True,
        },
        "grader_authority": "deterministic_simulator_state",
        "candidate_policy_queried": False,
        "policy_self_grading_used": False,
        "caller_asserted_success_accepted": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ControlEpisodeError([f"control_receipt_overwrite_forbidden:{path.name}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_required_controls(
    *,
    environment: ControlEnvironment,
    scenario_instance: Mapping[str, Any],
    expected_control_plan: Mapping[str, Any] | None = None,
    gripper_open_command: float,
    gripper_closed_command: float,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run both controls in order and emit the policy-admission receipt."""

    plan = materialize_control_plan(scenario_instance)
    if expected_control_plan is not None:
        expected = json.loads(json.dumps(dict(expected_control_plan), allow_nan=False))
        if (
            expected.get("plan_digest")
            != canonical_digest(expected, digest_field="plan_digest")
            or expected != plan
        ):
            raise ControlEpisodeError(["control_plan_bundle_binding_mismatch"])
    output = Path(output_dir).expanduser().resolve()
    _write_json(output / CONTROL_PLAN_FILENAME, plan)
    controls: list[dict[str, Any]] = []
    for control_id in REQUIRED_CONTROLS:
        receipt = run_control_episode(
            environment=environment,
            plan=plan,
            control_id=control_id,
            gripper_open_command=gripper_open_command,
            gripper_closed_command=gripper_closed_command,
            media_output_dir=output,
            episode_id=f"{plan['cell_id']}-{control_id}",
        )
        controls.append(receipt)
        _write_json(output / f"adp009d_control_episode.{control_id}.json", receipt)

    blockers: list[str] = []
    for receipt in controls:
        blockers.extend(receipt.get("blockers") or [])
    plan_contact_envelope = _plan_contact_envelope(plan)
    for receipt in controls:
        dynamics_summary = receipt.get("arm_dynamics_summary")
        if (
            receipt.get("contact_envelope") != plan_contact_envelope
            or not isinstance(dynamics_summary, Mapping)
            or dynamics_summary.get("contact_envelope") != plan_contact_envelope
        ):
            raise ControlEpisodeError(["control_pair_contact_envelope_unretained"])
    pair: dict[str, Any] = {
        "schema_version": CONTROL_PAIR_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "cell_id": plan["cell_id"],
        "family": plan["family"],
        "suite_digest": plan["suite_digest"],
        "instance_digest": plan["instance_digest"],
        "control_plan_digest": plan["plan_digest"],
        "contact_envelope": plan_contact_envelope,
        "execution_order": list(REQUIRED_CONTROLS),
        "controls": [
            {
                "control_id": receipt["control_id"],
                "control_passed": receipt["control_passed"],
                "observed_outcome": receipt["observed_outcome"],
                "contact_envelope": receipt["contact_envelope"],
                "receipt_digest": receipt["receipt_digest"],
            }
            for receipt in controls
        ],
        "cell_admitted_for_policy_execution": not blockers,
        "policy_execution_blockers": sorted(set(blockers)),
        "positive_failure_is_policy_failure": False,
        "candidate_policy_queried": False,
        "pair_digest": "",
    }
    pair["pair_digest"] = canonical_digest(pair, digest_field="pair_digest")
    _write_json(output / "adp009d_control_pair.v1.json", pair)
    return pair


__all__ = [
    "BLOCKER_POSITIVE_FAILED",
    "BLOCKER_PHASE_NOT_REACHED",
    "BLOCKER_ZERO_COMPLETED_TASK",
    "ARM_DYNAMICS_OBSERVATION_SCHEMA_VERSION",
    "ARM_DYNAMICS_SUMMARY_SCHEMA_VERSION",
    "CONTROL_EPISODE_SCHEMA_VERSION",
    "CONTROL_PLAN_FILENAME",
    "CONTROL_PAIR_SCHEMA_VERSION",
    "CONTROL_PLAN_SCHEMA_VERSION",
    "ControlEpisodeError",
    "SCRIPTED_POSITIVE",
    "ZERO_ACTION_NEGATIVE",
    "materialize_control_plan",
    "run_control_episode",
    "run_required_controls",
]
