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


CONTROL_PLAN_SCHEMA_VERSION = "adp009d_control_plan.v3"
CONTROL_EPISODE_SCHEMA_VERSION = "adp009d_control_episode.v1"
CONTROL_PAIR_SCHEMA_VERSION = "adp009d_control_pair.v1"
SCENARIO_INSTANCE_SCHEMA_VERSION = "adp009d_scenario_instance.v1"

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
            "steps": 80,
        },
        {
            "phase_id": "descend",
            "mode": "ik_pose",
            "target_position_world_m": [start[0], start[1], grasp_frame_z],
            "gripper": "open",
            "steps": 80,
        },
        {
            "phase_id": "grasp",
            "mode": "ik_pose",
            "target_position_world_m": [start[0], start[1], grasp_frame_z],
            "gripper": "closed",
            "steps": 30,
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
            "steps": 80,
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
            "steps": 100,
        },
        {
            "phase_id": "place",
            "mode": "ik_pose",
            "target_position_world_m": [target[0], target[1], place_frame_z],
            "gripper": "closed",
            "steps": 80,
        },
        {
            "phase_id": "release",
            "mode": "ik_pose",
            "target_position_world_m": [target[0], target[1], place_frame_z],
            "gripper": "open",
            "steps": 30,
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
            "steps": 60,
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
            phase["arrival_tolerance_m"] = PHASE_ARRIVAL_TOLERANCE_M
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
) -> dict[str, Any]:
    """Retain and gate the semantic grasp-frame error for one IK phase."""

    target = [float(value) for value in phase["target_position_world_m"]]
    start = [float(value) for value in start_sample["grasp_frame_position_world_m"]]
    achieved = [
        float(value) for value in terminal_sample["grasp_frame_position_world_m"]
    ]
    tolerance = float(phase["arrival_tolerance_m"])
    error = math.dist(achieved, target)
    return {
        "phase_id": str(phase["phase_id"]),
        "target_frame": str(phase["target_frame"]),
        "target_position_world_m": target,
        "start_position_world_m": start,
        "achieved_position_world_m": achieved,
        "terminal_position_error_m": error,
        "arrival_tolerance_m": tolerance,
        "target_reached": error <= tolerance,
    }


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
            }
        ]
    else:
        phases = [dict(phase) for phase in plan["scripted_positive_phases"]]

    for phase_index, phase in enumerate(phases):
        phase_start_sample = samples[-1]
        gripper_command = (
            float(gripper_open_command)
            if phase["gripper"] == "open"
            else float(gripper_closed_command)
        )
        for _ in range(int(phase["steps"])):
            before = [float(v) for v in environment.read_arm_joint_positions()]
            if phase["mode"] == "hold_current_joint_positions":
                action = environment.hold_action(gripper_command=gripper_command)
            else:
                action = environment.scripted_action_for_pose(
                    target_position_world_m=phase["target_position_world_m"],
                    target_quaternion_world_xyzw=phase[
                        "target_quaternion_world_xyzw"
                    ],
                    gripper_command=gripper_command,
                    max_joint_delta_rad=float(phase["max_joint_delta_rad"]),
                )
            environment.step(action)
            step_index += 1
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
            )
            phase_arrivals.append(arrival)
            if not arrival["target_reached"]:
                phase_execution_blocker = (
                    f"{BLOCKER_PHASE_NOT_REACHED}:{phase['phase_id']}:"
                    f"error_m={arrival['terminal_position_error_m']:.6f}"
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
    _write_json(output / "adp009d_control_plan.v3.json", plan)
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


__all__ = [
    "BLOCKER_POSITIVE_FAILED",
    "BLOCKER_PHASE_NOT_REACHED",
    "BLOCKER_ZERO_COMPLETED_TASK",
    "CONTROL_EPISODE_SCHEMA_VERSION",
    "CONTROL_PAIR_SCHEMA_VERSION",
    "CONTROL_PLAN_SCHEMA_VERSION",
    "ControlEpisodeError",
    "SCRIPTED_POSITIVE",
    "ZERO_ACTION_NEGATIVE",
    "materialize_control_plan",
    "run_control_episode",
    "run_required_controls",
]
