"""Deterministic negative and scripted-positive controls for ADP-009D cells.

The scenario suite is policy neutral.  Before either learned candidate may run
on one resolved cell, this module proves two properties through the same native
8D action seam: holding the current joints cannot complete the task, and a
frozen joint or Cartesian differential-IK program can.  A failed positive
blocks the cell; it is never counted as a learned-policy failure.

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
    from adp_task_scoring import (
        OUTCOME_NEVER_MOVED as TASK_NEUTRAL_OUTCOME_NEVER_MOVED,
        TASK_KIND_ARTICULATED_OPEN_CLOSE,
        TASK_KIND_DEFORMABLE_TRANSFER,
        TASK_KIND_RIGID_PICK_PLACE,
        TaskNeutralScoringError,
        score_task_episode_from_spec,
        validate_articulated_task_spec,
        validate_deformable_task_spec,
    )
except ModuleNotFoundError:  # repository package
    from .adp_task_scoring import (
        OUTCOME_NEVER_MOVED as TASK_NEUTRAL_OUTCOME_NEVER_MOVED,
        TASK_KIND_ARTICULATED_OPEN_CLOSE,
        TASK_KIND_DEFORMABLE_TRANSFER,
        TASK_KIND_RIGID_PICK_PLACE,
        TaskNeutralScoringError,
        score_task_episode_from_spec,
        validate_articulated_task_spec,
        validate_deformable_task_spec,
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


CONTROL_PLAN_SCHEMA_VERSION = "adp009d_control_plan.v5"
CONTROL_EPISODE_SCHEMA_VERSION = "adp009d_control_episode.v2"
CONTROL_PAIR_SCHEMA_VERSION = "adp009d_control_pair.v1"
SCENARIO_INSTANCE_SCHEMA_VERSION = "adp009d_scenario_instance.v1"
TASK_CONTROL_PLAN_SCHEMA_VERSION = "adp_task_control_plan.v1"
TASK_CONTROL_EPISODE_SCHEMA_VERSION = "adp_task_control_episode.v1"
TASK_CONTROL_PAIR_SCHEMA_VERSION = "adp_task_control_pair.v1"

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
# Absolute-position actuators need a target that can accumulate ahead of a
# slowly moving measured state.  This is a separate ceiling from command slew:
# v98 used the 0.03-rad slew as the lead ceiling and starved the controller.
MAX_JOINT_SETPOINT_LEAD_RAD = 0.20
PHASE_ARRIVAL_TOLERANCE_M = 0.02
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
        max_joint_setpoint_lead_rad: float,
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
    if value.get("instance_digest") != canonical_digest(value, digest_field="instance_digest"):
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
    if object_height <= 0.0:
        raise ControlEpisodeError(["control_plan_object_height_invalid"])
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
            phase["target_quaternion_world_xyzw"] = list(CONTROLLED_BODY_QUATERNION_WORLD_XYZW)
            phase["arrival_tolerance_m"] = PHASE_ARRIVAL_TOLERANCE_M
            phase["arrival_stability_steps"] = PHASE_ARRIVAL_STABILITY_STEPS
        phase["max_joint_delta_rad"] = MAX_JOINT_DELTA_PER_STEP_RAD
        phase["max_joint_setpoint_lead_rad"] = MAX_JOINT_SETPOINT_LEAD_RAD

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
        "grasp_target_frame": GRASP_TARGET_FRAME,
        "controlled_body_orientation_strategy": CONTROLLED_BODY_ORIENTATION_STRATEGY,
        "controlled_body_quaternion_world_xyzw": list(CONTROLLED_BODY_QUATERNION_WORLD_XYZW),
        "zero_action": {
            "semantics": ("zero_joint_velocity_realized_as_hold_current_absolute_joint_positions"),
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
    achieved = [float(value) for value in terminal_sample["grasp_frame_position_world_m"]]
    tolerance = float(phase["arrival_tolerance_m"])
    error = math.dist(achieved, target)
    stability_steps_required = int(phase["arrival_stability_steps"])
    return {
        "phase_id": str(phase["phase_id"]),
        "target_frame": str(phase["target_frame"]),
        "target_position_world_m": target,
        "start_position_world_m": start,
        "achieved_position_world_m": achieved,
        "terminal_position_error_m": error,
        "arrival_tolerance_m": tolerance,
        "terminal_within_tolerance": error <= tolerance,
        "minimum_steps": int(phase["minimum_steps"]),
        "maximum_steps": int(phase["maximum_steps"]),
        "steps_executed": int(steps_executed),
        "arrival_stability_steps_required": stability_steps_required,
        "arrival_stability_steps_observed": int(stability_steps_observed),
        "termination_reason": str(termination_reason),
        "target_reached": termination_reason == "stable_arrival",
    }


def _within_phase_arrival_tolerance(*, phase: Mapping[str, Any], sample: Mapping[str, Any]) -> bool:
    target = [float(value) for value in phase["target_position_world_m"]]
    achieved = [float(value) for value in sample["grasp_frame_position_world_m"]]
    return math.dist(achieved, target) <= float(phase["arrival_tolerance_m"])


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
) -> dict[str, Any]:
    values = [float(value) for value in action]
    if len(values) != 8 or not all(math.isfinite(value) for value in values):
        raise ControlEpisodeError(["control_episode_action_invalid"])
    return {
        "step_index": int(step_index),
        "phase_id": phase_id,
        "isaac_action": values,
        "observed_joint_position_before_rad": [float(v) for v in observed_before],
        "observed_joint_position_after_rad": [float(v) for v in observed_after],
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
    output = Path(media_output_dir).expanduser().resolve()
    if not episode_id.strip():
        raise ControlEpisodeError(["control_episode_id_missing"])

    environment.reset()
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
                "max_joint_setpoint_lead_rad": MAX_JOINT_SETPOINT_LEAD_RAD,
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
        phase_step_limit = int(phase.get("maximum_steps", phase.get("steps", 0)))
        for _ in range(phase_step_limit):
            before = [float(v) for v in environment.read_arm_joint_positions()]
            if phase["mode"] == "hold_current_joint_positions":
                action = environment.hold_action(gripper_command=gripper_command)
            else:
                action = environment.scripted_action_for_pose(
                    target_position_world_m=phase["target_position_world_m"],
                    target_quaternion_world_xyzw=phase["target_quaternion_world_xyzw"],
                    gripper_command=gripper_command,
                    max_joint_delta_rad=float(phase["max_joint_delta_rad"]),
                    max_joint_setpoint_lead_rad=float(phase["max_joint_setpoint_lead_rad"]),
                )
            environment.step(action)
            step_index += 1
            phase_steps_executed += 1
            after = [float(v) for v in environment.read_arm_joint_positions()]
            actions.append(
                _record_action(
                    step_index=step_index,
                    phase_id=str(phase["phase_id"]),
                    action=action,
                    observed_before=before,
                    observed_after=after,
                )
            )
            samples.append(_sample(environment, step_index))
            if phase["mode"] == "ik_pose":
                if _within_phase_arrival_tolerance(phase=phase, sample=samples[-1]):
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
                and stability_steps_observed >= int(phase["arrival_stability_steps"])
            ):
                termination_reason = "stable_arrival"
                break
        if phase["mode"] == "ik_pose" and termination_reason != "stable_arrival":
            termination_reason = "maximum_steps_exhausted"
        if phase_index < len(phases) - 1 and step_index % CONTROL_REVIEW_FRAME_STRIDE_STEPS != 0:
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
        blockers = [] if passed else [f"{BLOCKER_POSITIVE_FAILED}:{score.get('outcome')}"]
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
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ControlEpisodeError([f"control_receipt_overwrite_forbidden:{path.name}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            expected.get("plan_digest") != canonical_digest(expected, digest_field="plan_digest")
            or expected != plan
        ):
            raise ControlEpisodeError(["control_plan_bundle_binding_mismatch"])
    output = Path(output_dir).expanduser().resolve()
    _write_json(output / "adp009d_control_plan.v5.json", plan)
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
    pair: dict[str, Any] = {
        "schema_version": CONTROL_PAIR_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "cell_id": plan["cell_id"],
        "family": plan["family"],
        "suite_digest": plan["suite_digest"],
        "instance_digest": plan["instance_digest"],
        "control_plan_digest": plan["plan_digest"],
        "execution_order": list(REQUIRED_CONTROLS),
        "controls": [
            {
                "control_id": receipt["control_id"],
                "control_passed": receipt["control_passed"],
                "observed_outcome": receipt["observed_outcome"],
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


def _task_neutral_sample(
    environment: ControlEnvironment, *, task_kind: str, step_index: int
) -> dict[str, Any]:
    if task_kind == TASK_KIND_RIGID_PICK_PLACE:
        raw = environment.read_object_sample()
    else:
        reader = getattr(environment, "read_task_sample", None)
        if not callable(reader):
            raise ControlEpisodeError(["task_control_native_task_sample_missing"])
        raw = reader()
    if not isinstance(raw, Mapping):
        raise ControlEpisodeError(["task_control_native_task_sample_invalid"])
    sample = dict(raw)
    if "step_index" in sample and sample["step_index"] != step_index:
        raise ControlEpisodeError(["task_control_sample_step_mismatch"])
    sample["step_index"] = int(step_index)
    return sample


def validate_task_control_plan(
    plan: Mapping[str, Any], *, task_spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate an immutable task-neutral control trajectory before reset."""

    try:
        checked = json.loads(json.dumps(dict(plan), allow_nan=False))
        task = json.loads(json.dumps(dict(task_spec), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ControlEpisodeError(["task_control_plan_invalid"]) from exc
    errors: list[str] = []
    if checked.get("schema_version") != TASK_CONTROL_PLAN_SCHEMA_VERSION:
        errors.append("task_control_plan_schema_invalid")
    if checked.get("plan_digest") != canonical_digest(checked, digest_field="plan_digest"):
        errors.append("task_control_plan_digest_mismatch")
    if checked.get("task_spec_digest") != canonical_digest(task):
        errors.append("task_control_plan_task_spec_mismatch")
    if checked.get("trajectory_source") != "native_ik_preflight":
        errors.append("task_control_trajectory_source_invalid")
    planner_receipt_digest = str(checked.get("planner_receipt_digest") or "")
    if not planner_receipt_digest.startswith("sha256:") or len(planner_receipt_digest) != 71:
        errors.append("task_control_planner_receipt_digest_invalid")
    zero_steps = checked.get("zero_action_steps")
    if isinstance(zero_steps, bool) or not isinstance(zero_steps, int) or zero_steps < 1:
        errors.append("task_control_zero_action_steps_invalid")
    actions = checked.get("scripted_positive_actions")
    if not isinstance(actions, list) or not actions:
        errors.append("task_control_scripted_actions_missing")
        actions = []
    normalized_actions: list[dict[str, Any]] = []
    maximum_scripted_steps = 0
    for index, raw in enumerate(actions):
        if not isinstance(raw, Mapping):
            errors.append(f"task_control_scripted_action_invalid:{index}")
            continue
        phase_id = str(raw.get("phase_id") or "")
        if raw.get("mode") == "ik_pose" or "target_position_world_m" in raw:
            try:
                position = [float(value) for value in raw["target_position_world_m"]]
                quaternion_raw = raw.get("target_quaternion_world_xyzw")
                quaternion = (
                    None if quaternion_raw is None else [float(value) for value in quaternion_raw]
                )
                minimum_steps = int(raw["minimum_steps"])
                maximum_steps = int(raw["maximum_steps"])
                arrival_tolerance_m = float(raw["arrival_tolerance_m"])
                arrival_stability_steps = int(raw["arrival_stability_steps"])
                max_joint_delta_rad = float(raw["max_joint_delta_rad"])
                max_joint_setpoint_lead_rad = float(raw["max_joint_setpoint_lead_rad"])
            except (KeyError, TypeError, ValueError):
                position = []
                quaternion = []
                minimum_steps = 0
                maximum_steps = 0
                arrival_tolerance_m = 0.0
                arrival_stability_steps = 0
                max_joint_delta_rad = 0.0
                max_joint_setpoint_lead_rad = 0.0
            gripper_state = str(raw.get("gripper_state") or "")
            quaternion_valid = quaternion is None or (
                len(quaternion) == 4
                and all(math.isfinite(value) for value in quaternion)
                and math.isclose(
                    math.sqrt(sum(value * value for value in quaternion)),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-5,
                )
            )
            if (
                not phase_id
                or len(position) != 3
                or not all(math.isfinite(value) for value in position)
                or not quaternion_valid
                or gripper_state not in {"open", "closed"}
                or minimum_steps < 1
                or maximum_steps < minimum_steps
                or arrival_tolerance_m <= 0.0
                or not math.isfinite(arrival_tolerance_m)
                or arrival_stability_steps < 1
                or max_joint_delta_rad <= 0.0
                or not math.isfinite(max_joint_delta_rad)
                or max_joint_setpoint_lead_rad < max_joint_delta_rad
                or not math.isfinite(max_joint_setpoint_lead_rad)
            ):
                errors.append(f"task_control_scripted_pose_invalid:{index}")
            else:
                normalized_actions.append(
                    {
                        "phase_id": phase_id,
                        "mode": "ik_pose",
                        "target_position_world_m": position,
                        "target_quaternion_world_xyzw": quaternion,
                        "gripper_state": gripper_state,
                        "minimum_steps": minimum_steps,
                        "maximum_steps": maximum_steps,
                        "arrival_tolerance_m": arrival_tolerance_m,
                        "arrival_stability_steps": arrival_stability_steps,
                        "max_joint_delta_rad": max_joint_delta_rad,
                        "max_joint_setpoint_lead_rad": max_joint_setpoint_lead_rad,
                    }
                )
                maximum_scripted_steps += maximum_steps
            continue
        if "isaac_action" in raw:
            values = raw.get("isaac_action")
            try:
                action = [float(value) for value in values]
            except (TypeError, ValueError):
                action = []
            if (
                not phase_id
                or len(action) != 8
                or not all(math.isfinite(value) for value in action)
            ):
                errors.append(f"task_control_scripted_action_invalid:{index}")
            else:
                normalized_actions.append({"phase_id": phase_id, "isaac_action": action})
                maximum_scripted_steps += 1
            continue
        values = raw.get("arm_joint_positions")
        gripper_state = str(raw.get("gripper_state") or "")
        try:
            arm = [float(value) for value in values]
        except (TypeError, ValueError):
            arm = []
        if (
            not phase_id
            or len(arm) != 7
            or not all(math.isfinite(value) for value in arm)
            or gripper_state not in {"open", "closed"}
        ):
            errors.append(f"task_control_scripted_action_invalid:{index}")
        else:
            normalized_actions.append(
                {
                    "phase_id": phase_id,
                    "arm_joint_positions": arm,
                    "gripper_state": gripper_state,
                }
            )
            maximum_scripted_steps += 1
    kind = task.get("task_kind")
    if kind == TASK_KIND_ARTICULATED_OPEN_CLOSE:
        try:
            validate_articulated_task_spec(task)
        except TaskNeutralScoringError as exc:
            errors.extend(exc.errors)
    elif kind == TASK_KIND_DEFORMABLE_TRANSFER:
        try:
            validate_deformable_task_spec(task)
        except TaskNeutralScoringError as exc:
            errors.extend(exc.errors)
    elif kind != TASK_KIND_RIGID_PICK_PLACE:
        errors.append("task_control_task_kind_unsupported")
    settle_steps = task.get("settle_window_samples")
    if isinstance(settle_steps, bool) or not isinstance(settle_steps, int) or settle_steps < 1:
        errors.append("task_control_settle_window_invalid")
        settle_steps = 0
    maximum_steps = task.get("maximum_action_steps")
    if maximum_steps is not None and (
        isinstance(maximum_steps, bool)
        or not isinstance(maximum_steps, int)
        or maximum_scripted_steps + int(settle_steps) > maximum_steps
        or int(zero_steps or 0) > maximum_steps
    ):
        errors.append("task_control_action_budget_exceeds_task_spec")
    if errors:
        raise ControlEpisodeError(errors)
    checked["scripted_positive_actions"] = normalized_actions
    return checked


def _run_task_control_episode(
    *,
    environment: ControlEnvironment,
    task_spec: Mapping[str, Any],
    plan: Mapping[str, Any],
    control_id: str,
    gripper_open_command: float,
    gripper_closed_command: float | None,
    output: Path,
    episode_id: str,
) -> dict[str, Any]:
    task_kind = str(task_spec["task_kind"])
    environment.reset()
    samples = [_task_neutral_sample(environment, task_kind=task_kind, step_index=0)]
    policy_inputs = [
        _persist_observation(
            environment,
            output_dir=output,
            episode_id=episode_id,
            observation_index=0,
            kind="policy-input",
        )
    ]
    review_observations: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    phase_arrivals: list[dict[str, Any]] = []
    phase_execution_blocker: str | None = None
    observation_index = 1
    step_index = 0
    if control_id == ZERO_ACTION_NEGATIVE:
        trajectory = [
            {
                "phase_id": ZERO_ACTION_NEGATIVE,
                "mode": "hold_current_joint_positions",
            }
            for _ in range(int(plan["zero_action_steps"]))
        ]
    else:
        trajectory = [dict(row) for row in plan["scripted_positive_actions"]]
        trajectory.extend(
            {
                "phase_id": "release_settle",
                "mode": "hold_current_joint_positions",
            }
            for _ in range(int(task_spec["settle_window_samples"]))
        )
    for row in trajectory:
        pose_mode = row.get("mode") == "ik_pose"
        phase_steps = int(row["maximum_steps"]) if pose_mode else 1
        phase_steps_executed = 0
        stable_steps = 0
        termination_reason = "fixed_steps_completed"
        start_sample = samples[-1]
        for _ in range(phase_steps):
            before = [float(value) for value in environment.read_arm_joint_positions()]
            if row.get("mode") == "hold_current_joint_positions":
                action = environment.hold_action(gripper_command=float(gripper_open_command))
            elif pose_mode:
                state = str(row["gripper_state"])
                if state == "closed" and gripper_closed_command is None:
                    raise ControlEpisodeError(["task_control_gripper_closed_command_missing"])
                command = (
                    float(gripper_open_command)
                    if state == "open"
                    else float(gripper_closed_command)
                )
                action = environment.scripted_action_for_pose(
                    target_position_world_m=row["target_position_world_m"],
                    target_quaternion_world_xyzw=row["target_quaternion_world_xyzw"],
                    gripper_command=command,
                    max_joint_delta_rad=float(row["max_joint_delta_rad"]),
                    max_joint_setpoint_lead_rad=float(row["max_joint_setpoint_lead_rad"]),
                )
            elif "isaac_action" in row:
                action = row["isaac_action"]
            else:
                state = str(row["gripper_state"])
                if state == "closed" and gripper_closed_command is None:
                    raise ControlEpisodeError(["task_control_gripper_closed_command_missing"])
                command = (
                    float(gripper_open_command)
                    if state == "open"
                    else float(gripper_closed_command)
                )
                action = [*row["arm_joint_positions"], command]
            action = [float(value) for value in action]
            environment.step(action)
            step_index += 1
            phase_steps_executed += 1
            after = [float(value) for value in environment.read_arm_joint_positions()]
            actions.append(
                _record_action(
                    step_index=step_index,
                    phase_id=str(row["phase_id"]),
                    action=action,
                    observed_before=before,
                    observed_after=after,
                )
            )
            samples.append(
                _task_neutral_sample(environment, task_kind=task_kind, step_index=step_index)
            )
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
            if pose_mode:
                measured = samples[-1].get("grasp_frame_position_world_m")
                if not isinstance(measured, Sequence) or isinstance(measured, (str, bytes)):
                    raise ControlEpisodeError(["task_control_grasp_frame_readback_missing"])
                try:
                    error = math.dist(
                        [float(value) for value in measured],
                        row["target_position_world_m"],
                    )
                except (TypeError, ValueError) as exc:
                    raise ControlEpisodeError(
                        ["task_control_grasp_frame_readback_invalid"]
                    ) from exc
                stable_steps = stable_steps + 1 if error <= float(row["arrival_tolerance_m"]) else 0
                if phase_steps_executed >= int(row["minimum_steps"]) and stable_steps >= int(
                    row["arrival_stability_steps"]
                ):
                    termination_reason = "stable_arrival"
                    break
        if pose_mode:
            measured = samples[-1].get("grasp_frame_position_world_m")
            terminal_error = math.dist(
                [float(value) for value in measured], row["target_position_world_m"]
            )
            arrival = {
                "phase_id": str(row["phase_id"]),
                "start_position_world_m": start_sample.get("grasp_frame_position_world_m"),
                "target_position_world_m": row["target_position_world_m"],
                "terminal_position_world_m": measured,
                "terminal_position_error_m": terminal_error,
                "arrival_tolerance_m": float(row["arrival_tolerance_m"]),
                "arrival_stability_steps_required": int(row["arrival_stability_steps"]),
                "arrival_stability_steps_observed": stable_steps,
                "termination_reason": termination_reason,
                "target_reached": termination_reason == "stable_arrival",
            }
            phase_arrivals.append(arrival)
            if not arrival["target_reached"]:
                phase_execution_blocker = (
                    f"{BLOCKER_PHASE_NOT_REACHED}:{row['phase_id']}:"
                    f"error_m={terminal_error:.6f}:stability_steps="
                    f"{stable_steps}/{row['arrival_stability_steps']}"
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
            "task_spec_digest": plan["task_spec_digest"],
            "control_plan_digest": plan["plan_digest"],
            "candidate_policy_queried": False,
            "observation_consumer": "deterministic_control_monitor",
        },
        policy_input_observations=policy_inputs,
        review_observations=review_observations,
        terminal_observation=terminal,
    )
    try:
        score = score_task_episode_from_spec(task_spec=task_spec, samples=samples)
    except TaskNeutralScoringError as exc:
        raise ControlEpisodeError(exc.errors) from exc
    if control_id == ZERO_ACTION_NEGATIVE:
        passed = (
            score.get("status") == "scored"
            and score.get("task_succeeded") is False
            and score.get("outcome") == TASK_NEUTRAL_OUTCOME_NEVER_MOVED
        )
        blockers = [] if passed else [BLOCKER_ZERO_COMPLETED_TASK]
    else:
        passed = score.get("status") == "scored" and score.get("task_succeeded") is True
        blockers = [] if passed else [f"{BLOCKER_POSITIVE_FAILED}:{score.get('outcome')}"]
        if phase_execution_blocker is not None:
            blockers.append(phase_execution_blocker)
            passed = False
    if visual.get("status") != "complete":
        blockers.append(BLOCKER_MEDIA_INCOMPLETE)
        passed = False
    receipt: dict[str, Any] = {
        "schema_version": TASK_CONTROL_EPISODE_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "control_id": control_id,
        "episode_id": episode_id,
        "task_kind": task_kind,
        "task_spec_digest": plan["task_spec_digest"],
        "control_plan_digest": plan["plan_digest"],
        "control_passed": passed,
        "blockers": sorted(set(blockers)),
        "score": score,
        "observed_outcome": score.get("outcome"),
        "state_trace": samples,
        "state_trace_digest": canonical_digest({"samples": samples}),
        "action_trace": actions,
        "action_trace_digest": canonical_digest({"actions": actions}),
        "environment_steps": step_index,
        "phase_arrivals": phase_arrivals,
        "phase_execution_blocker": phase_execution_blocker,
        "visual_evidence": visual,
        "media_artifacts": artifacts,
        "grader_authority": "deterministic_simulator_state",
        "candidate_policy_queried": False,
        "caller_asserted_success_accepted": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def run_task_neutral_controls(
    *,
    environment: ControlEnvironment,
    task_spec: Mapping[str, Any],
    control_plan: Mapping[str, Any],
    gripper_open_command: float,
    gripper_closed_command: float | None = None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run zero then scripted controls for rigid or articulated task state."""

    task = json.loads(json.dumps(dict(task_spec), allow_nan=False))
    plan = validate_task_control_plan(control_plan, task_spec=task)
    output = Path(output_dir).expanduser().resolve()
    _write_json(output / "adp_task_control_plan.v1.json", plan)
    receipts = []
    for control_id in REQUIRED_CONTROLS:
        receipt = _run_task_control_episode(
            environment=environment,
            task_spec=task,
            plan=plan,
            control_id=control_id,
            gripper_open_command=float(gripper_open_command),
            gripper_closed_command=(
                None if gripper_closed_command is None else float(gripper_closed_command)
            ),
            output=output,
            episode_id=f"{plan['cell_id']}-{control_id}",
        )
        receipts.append(receipt)
        _write_json(output / f"adp_task_control_episode.{control_id}.json", receipt)
    blockers = [blocker for receipt in receipts for blocker in receipt.get("blockers", [])]
    pair: dict[str, Any] = {
        "schema_version": TASK_CONTROL_PAIR_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "cell_id": plan["cell_id"],
        "task_kind": task["task_kind"],
        "task_spec_digest": plan["task_spec_digest"],
        "control_plan_digest": plan["plan_digest"],
        "execution_order": list(REQUIRED_CONTROLS),
        "controls": [
            {
                "control_id": receipt["control_id"],
                "control_passed": receipt["control_passed"],
                "observed_outcome": receipt["observed_outcome"],
                "receipt_digest": receipt["receipt_digest"],
            }
            for receipt in receipts
        ],
        "cell_admitted_for_policy_execution": not blockers,
        "policy_execution_blockers": sorted(set(blockers)),
        "candidate_policy_queried": False,
        "pair_digest": "",
    }
    pair["pair_digest"] = canonical_digest(pair, digest_field="pair_digest")
    _write_json(output / "adp_task_control_pair.v1.json", pair)
    return pair


__all__ = [
    "BLOCKER_POSITIVE_FAILED",
    "BLOCKER_PHASE_NOT_REACHED",
    "BLOCKER_ZERO_COMPLETED_TASK",
    "CONTROL_EPISODE_SCHEMA_VERSION",
    "CONTROL_PAIR_SCHEMA_VERSION",
    "CONTROL_PLAN_SCHEMA_VERSION",
    "MAX_JOINT_DELTA_PER_STEP_RAD",
    "MAX_JOINT_SETPOINT_LEAD_RAD",
    "ControlEpisodeError",
    "SCRIPTED_POSITIVE",
    "TASK_CONTROL_EPISODE_SCHEMA_VERSION",
    "TASK_CONTROL_PAIR_SCHEMA_VERSION",
    "TASK_CONTROL_PLAN_SCHEMA_VERSION",
    "ZERO_ACTION_NEGATIVE",
    "materialize_control_plan",
    "run_control_episode",
    "run_required_controls",
    "run_task_neutral_controls",
    "validate_task_control_plan",
]
