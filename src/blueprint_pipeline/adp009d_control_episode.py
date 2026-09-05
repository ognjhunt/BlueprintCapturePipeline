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
from collections.abc import Callable, Mapping, Sequence
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
        TASK_KIND_RIGID_PICK_PLACE,
        TaskNeutralScoringError,
        score_task_episode_from_spec,
        validate_articulated_task_spec,
    )
except ModuleNotFoundError:  # repository package
    from .adp_task_scoring import (
        OUTCOME_NEVER_MOVED as TASK_NEUTRAL_OUTCOME_NEVER_MOVED,
        TASK_KIND_ARTICULATED_OPEN_CLOSE,
        TASK_KIND_RIGID_PICK_PLACE,
        TaskNeutralScoringError,
        score_task_episode_from_spec,
        validate_articulated_task_spec,
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
try:  # flat provider-bundle layout
    from adp009d_physics_backend_comparison import (
        build_backend_contact_configuration,
        normalize_physics_backend,
        validate_backend_contact_configuration,
    )
except ModuleNotFoundError:  # repository package
    from .adp009d_physics_backend_comparison import (
        build_backend_contact_configuration,
        normalize_physics_backend,
        validate_backend_contact_configuration,
    )
try:  # flat provider-bundle layout
    from task_control_diagnostic_boundary import (
        apply_diagnostic_receipt_boundary,
        build_task_control_pair,
        control_plan_boundary_errors,
        copy_diagnostic_annotations,
        diagnostic_receipt_annotations,
    )
except ModuleNotFoundError:  # repository package
    from .task_control_diagnostic_boundary import (
        apply_diagnostic_receipt_boundary,
        build_task_control_pair,
        control_plan_boundary_errors,
        copy_diagnostic_annotations,
        diagnostic_receipt_annotations,
    )


CONTROL_PLAN_SCHEMA_VERSION = "adp009d_control_plan.v12"
CONTROL_PLAN_FILENAME = f"{CONTROL_PLAN_SCHEMA_VERSION}.json"
CONTROL_EPISODE_SCHEMA_VERSION = "adp009d_control_episode.v4"
CONTROL_PAIR_SCHEMA_VERSION = "adp009d_control_pair.v1"
SCENARIO_INSTANCE_SCHEMA_VERSION = "adp009d_scenario_instance.v1"
TASK_CONTROL_PLAN_SCHEMA_VERSION = "adp_task_control_plan.v1"
TASK_CONTROL_EPISODE_SCHEMA_VERSION = "adp_task_control_episode.v1"
TASK_CONTROL_PAIR_SCHEMA_VERSION = "adp_task_control_pair.v1"
TASK_DOWNSTREAM_DIAGNOSTIC_SCHEMA_VERSION = (
    "adp_task_synthetic_post_phase5_downstream_diagnostic.v1"
)
ARM_DYNAMICS_OBSERVATION_SCHEMA_VERSION = "adp009d_arm_dynamics_observation.v2"
ARM_DYNAMICS_SUMMARY_SCHEMA_VERSION = "adp009d_arm_dynamics_summary.v2"

ZERO_ACTION_NEGATIVE = "zero_action_negative"
SCRIPTED_POSITIVE = "deterministic_scripted_positive"
REQUIRED_CONTROLS = (ZERO_ACTION_NEGATIVE, SCRIPTED_POSITIVE)
DOWNSTREAM_DIAGNOSTIC_CONTROL_ID = (
    "development_only_synthetic_post_phase5_downstream"
)
DOWNSTREAM_DIAGNOSTIC_PHASE_IDS = (
    "joint_path_01",
    "joint_path_02",
    "joint_path_03",
    "joint_path_04",
    "release",
    "retreat",
)

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
# MAX_JOINT_SETPOINT_LEAD_RAD was defined and exported here at 0.20 and read by
# nothing -- repo-wide, including scripts and profiles. Its comment claimed it
# was "retained for current-main task-neutral controls"; those read the value
# off the plan row, not from here.
#
# A dead binding of a live name is a landmine: native_task_construction_plan
# binds the same name to 1.00, so anything importing it from this module would
# have silently taken the pre-#786 throttle -- the exact shape that made #786
# inert for three paid runs.
#
# The neighbouring slew constant does NOT shadow: it is deliberately named
# MAX_JOINT_DELTA_PER_STEP_RAD. That is the pattern to follow here.
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

# Paid runs C20c and C25 each burned a full GPU cycle to learn only that the
# live arm parks ~14-15 mm from the commanded contact pose.  A failed pose
# phase therefore retries inside the same episode: retreat toward the phase's
# already-achieved entry pose, then re-command the pose biased by the measured
# miss.  The arrival gate always measures against the original sealed target
# -- only the command is biased -- and every attempt seals its own arrival
# row, so a run either passes honestly or pins whether the miss is a
# constant offset (compensation converges) or a saturation (it does not).
# An attempt costs about 88 simulator steps -- the phase plus its bounded
# retreat -- which at 15 Hz is roughly six seconds of a run that is already
# paying minutes for scene construction.  Thirty attempts is therefore about
# three minutes of GPU time, while a fresh run to resume the same search costs
# a full cold start.  The count was never the thing worth conserving; the
# projection below is what stops a dead strategy from consuming the budget,
# and a converging one should be allowed to finish inside the run that found
# it.  C29 was ended by a cap of three while it was still closing the gap.
TASK_CONTROL_MAX_POSE_PHASE_ATTEMPTS = 30
TASK_CONTROL_RECOVERY_RETREAT_MAXIMUM_STEPS = 24

# C28 sealed why a single strategy repeated blindly is the wrong budget: its
# three measured-miss attempts diverged 15.4 -> 28.5 -> 38.6 mm.  Repeating a
# diverging strategy only produces a worse number, while stopping a
# *converging* one at a fixed count throws away a run that was about to land.
# So the count is not the control -- the trend is.  After each failed attempt
# the executor reads its own sealed error against the previous attempt and
# either repeats a strategy that improved, or escalates to the next rung.
# Each rung is a distinct physical hypothesis, ordered cheapest-first:
#   * measured_miss_compensation -- the miss is a constant offset.
#   * damped_half_miss_compensation -- compensation overshoots or oscillates.
#   * clean_entry_pose_reentry -- the miss was path-dependent, so re-approach
#     from the qualified entry pose with no bias at all.
#   * extended_standoff_reentry -- the servo passes through a degenerate
#     direction near the target, so re-enter along a longer straight line.
# The arrival gate is identical on every rung: measured fingertip against the
# original sealed target.  Only the command and the re-entry pose vary.
TASK_CONTROL_RECOVERY_LADDER = (
    "measured_miss_compensation",
    "damped_half_miss_compensation",
    "clean_entry_pose_reentry",
    "extended_standoff_reentry",
)
TASK_CONTROL_RECOVERY_EXTENDED_STANDOFF_SCALE = 2.0


def recovery_ladder_for_plan(plan: Mapping[str, Any]) -> tuple[str, ...]:
    """Use the declared implemented recovery rungs, or the default ladder.

    A malformed or empty declaration must not silently disable recovery.
    """

    declared = plan.get("recovery_strategy_ladder")
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
        return TASK_CONTROL_RECOVERY_LADDER
    ladder = tuple(
        str(value)
        for value in declared
        if str(value) in TASK_CONTROL_RECOVERY_LADDER
    )
    if not ladder or len(ladder) != len(set(ladder)) or len(ladder) != len(declared):
        return TASK_CONTROL_RECOVERY_LADDER
    return ladder


def _next_recovery_strategy(
    attempt_history: Sequence[Mapping[str, Any]],
    *,
    ladder: Sequence[str] = TASK_CONTROL_RECOVERY_LADDER,
    arrival_tolerance_m: float = 0.0,
    remaining_attempts: int = 0,
) -> str | None:
    """Keep a strategy that will land in time; escalate one that will not.

    A sign check is not enough, and C29 proved both halves of that.  Its
    branch-replay entry turned C28's divergence (15.4 -> 28.5 -> 38.6 mm) into
    convergence (15.5 -> 13.6 -> 12.9 mm), so "is it improving?" now says
    repeat -- but the improvement was also *decelerating* (1.84 mm, then
    0.69 mm), and at that rate it needed roughly eleven more attempts to
    reach a 5 mm gate with five left.  Repeating it would have burned the
    whole budget on a strategy that could not finish.

    So the test is a projection rather than a sign: at the rate this strategy
    is actually closing the gap, does it land inside the attempts that remain?
    If yes, keep going -- that is a run about to succeed, and stopping it only
    to resume the same trend on a fresh GPU is pure waste.  If no, spend the
    remaining attempts on a different hypothesis instead.

    ``attempt_history`` is this phase's failed attempts in order, each with
    the strategy that produced it (``None`` for the nominal attempt) and its
    measured terminal error.  Returns the strategy for the next attempt, or
    ``None`` when the ladder is exhausted.
    """

    rungs = tuple(ladder) or TASK_CONTROL_RECOVERY_LADDER
    if not attempt_history:
        return rungs[0]
    last = attempt_history[-1]
    strategy = last.get("strategy")
    if strategy is None:
        return rungs[0]
    if strategy not in rungs:
        return None
    previous = attempt_history[-2] if len(attempt_history) > 1 else None
    if previous is not None:
        try:
            last_error = float(last["error_m"])
            improvement = float(previous["error_m"]) - last_error
            excess = last_error - float(arrival_tolerance_m)
        except (KeyError, TypeError, ValueError):
            improvement = 0.0
            excess = 0.0
        if improvement > 0.0 and excess > 0.0:
            projected_attempts = excess / improvement
            if projected_attempts <= max(0, int(remaining_attempts)):
                return strategy
        elif improvement > 0.0:
            return strategy
    rung = rungs.index(strategy)
    if rung + 1 >= len(rungs):
        return None
    return rungs[rung + 1]


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
        phase_id: str | None = None,
        target_position_world_m: Sequence[float],
        target_quaternion_world_xyzw: Sequence[float] | None,
        gripper_command: float,
        max_joint_delta_rad: float,
        max_task_space_translation_step_m: float,
        orientation_tolerance_deg: float,
        task_space_translation_strategy: str,
    ) -> Sequence[float]: ...


def _quaternion_angle_xyzw(a: Sequence[float], b: Sequence[float]) -> float:
    try:
        qa = [float(value) for value in a]
        qb = [float(value) for value in b]
    except (TypeError, ValueError) as exc:
        raise ControlEpisodeError(
            ["task_control_grasp_frame_orientation_invalid"]
        ) from exc
    if len(qa) != 4 or len(qb) != 4:
        raise ControlEpisodeError(
            ["task_control_grasp_frame_orientation_invalid"]
        )
    norm_a = math.sqrt(sum(value * value for value in qa))
    norm_b = math.sqrt(sum(value * value for value in qb))
    if not all(math.isfinite(value) for value in (*qa, *qb)) or min(
        norm_a, norm_b
    ) <= 0.0:
        raise ControlEpisodeError(
            ["task_control_grasp_frame_orientation_invalid"]
        )
    dot = abs(sum(x * y for x, y in zip(qa, qb, strict=True)) / (norm_a * norm_b))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))

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


def materialize_control_plan(
    instance: Mapping[str, Any], *, physics_backend: str = "physx"
) -> dict[str, Any]:
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
    backend = normalize_physics_backend(physics_backend)
    backend_contact_configuration = build_backend_contact_configuration(backend)
    contact_envelope = canonical_contact_envelope() if backend == "physx" else None
    open_jaw_radial_clearance = GRIPPER_FULL_OPENING_M / 2.0 - object_radius
    effective_contact_envelope = float(
        backend_contact_configuration["planner_contact_allowance_m"]
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
        "physics_backend": backend,
        "backend_contact_configuration": backend_contact_configuration,
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
            "aperture_safe_arrival_tolerance_m": aperture_safe_arrival_tolerance,
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
        "semantic_plan_digest": "",
        "plan_digest": "",
    }
    semantic_plan = {
        "instance_digest": plan["instance_digest"],
        "cell_id": plan["cell_id"],
        "seed": plan["seed"],
        "resolved_start_position_world_m": plan["resolved_start_position_world_m"],
        "resolved_destination_position_world_m": plan[
            "resolved_destination_position_world_m"
        ],
        "object_height_m": object_height,
        "object_radius_m": object_radius,
        "grasp_target_frame": plan["grasp_target_frame"],
        "controlled_body_orientation_strategy": plan[
            "controlled_body_orientation_strategy"
        ],
        "controlled_body_quaternion_world_xyzw": plan[
            "controlled_body_quaternion_world_xyzw"
        ],
        "phase_semantics": [
            {
                key: phase.get(key)
                for key in (
                    "phase_id",
                    "mode",
                    "target_position_world_m",
                    "target_frame",
                    "target_quaternion_world_xyzw",
                    "gripper",
                )
            }
            for phase in phases
        ],
    }
    plan["semantic_plan_digest"] = canonical_digest(semantic_plan)
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
    if result["contact_envelope"] is not None:
        try:
            result["contact_envelope"] = validate_contact_envelope(
                result["contact_envelope"]
            )
        except ContactEnvelopeError as exc:
            raise ControlEpisodeError([str(exc)]) from exc
    contact_configuration = result.get("backend_contact_configuration")
    if contact_configuration is None and result["contact_envelope"] is not None:
        contact_configuration = build_backend_contact_configuration("physx")
    if not isinstance(contact_configuration, Mapping):
        raise ControlEpisodeError(
            ["control_episode_backend_contact_configuration_missing"]
        )
    configuration_blockers = validate_backend_contact_configuration(
        contact_configuration
    )
    if configuration_blockers:
        raise ControlEpisodeError(configuration_blockers)
    result["backend_contact_configuration"] = dict(contact_configuration)
    return result


def _plan_contact_envelope(plan: Mapping[str, Any]) -> dict[str, Any] | None:
    if plan.get("physics_backend", "physx") == "newton":
        if plan.get("contact_envelope") is not None:
            raise ControlEpisodeError(["adp009d_newton_physx_contact_envelope_forbidden"])
        return None
    try:
        return validate_contact_envelope(plan.get("contact_envelope"))
    except ContactEnvelopeError as exc:
        raise ControlEpisodeError([str(exc)]) from exc


def _require_dynamics_contact_envelope(
    dynamics: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None,
) -> None:
    expected_value = None if expected is None else dict(expected)
    if dynamics.get("contact_envelope") != expected_value:
        raise ControlEpisodeError(["control_episode_contact_envelope_plan_mismatch"])


def _plan_backend_contact_configuration(plan: Mapping[str, Any]) -> dict[str, Any]:
    configuration = plan.get("backend_contact_configuration")
    if configuration is None and plan.get("physics_backend", "physx") == "physx":
        configuration = build_backend_contact_configuration("physx")
    if not isinstance(configuration, Mapping):
        raise ControlEpisodeError(
            ["control_plan_backend_contact_configuration_missing"]
        )
    blockers = validate_backend_contact_configuration(configuration)
    if blockers:
        raise ControlEpisodeError(blockers)
    if configuration.get("physics_backend") != plan.get("physics_backend", "physx"):
        raise ControlEpisodeError(["control_plan_physics_backend_mismatch"])
    return dict(configuration)


def _require_dynamics_contact_configuration(
    dynamics: Mapping[str, Any], *, expected: Mapping[str, Any]
) -> None:
    if dynamics.get("backend_contact_configuration") != dict(expected):
        raise ControlEpisodeError(
            ["control_episode_backend_contact_configuration_plan_mismatch"]
        )


def _vector_norm(values: Sequence[Any]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _summarize_arm_dynamics(actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize tracking, saturation, and contact without declaring a cause."""
    phases: dict[str, dict[str, Any]] = {}
    contact_envelope: dict[str, Any] | None = None
    contact_configuration: dict[str, Any] | None = None
    contact_observation_seen = False
    for action in actions:
        phase_id = str(action["phase_id"])
        dynamics = dict(action["arm_dynamics_after"])
        canonical_dynamics = _canonical_dynamics_observation(dynamics)
        observed_contact_envelope = canonical_dynamics["contact_envelope"]
        observed_contact_configuration = canonical_dynamics[
            "backend_contact_configuration"
        ]
        if not contact_observation_seen:
            contact_envelope = observed_contact_envelope
            contact_configuration = observed_contact_configuration
            contact_observation_seen = True
        elif observed_contact_envelope != contact_envelope:
            raise ControlEpisodeError(["control_episode_contact_envelope_drifted"])
        elif observed_contact_configuration != contact_configuration:
            raise ControlEpisodeError(
                ["control_episode_backend_contact_configuration_drifted"]
            )
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
    if not contact_observation_seen or contact_configuration is None:
        raise ControlEpisodeError(["control_episode_arm_dynamics_missing"])
    return {
        "schema_version": ARM_DYNAMICS_SUMMARY_SCHEMA_VERSION,
        "backend_contact_configuration": contact_configuration,
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
    plan_contact_configuration = _plan_backend_contact_configuration(plan)
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
    _require_dynamics_contact_configuration(
        initial_arm_dynamics, expected=plan_contact_configuration
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
            dynamics_before = _canonical_dynamics_observation(environment.read_arm_dynamics_observation())
            _require_dynamics_contact_envelope(
                dynamics_before,
                expected=plan_contact_envelope,
            )
            _require_dynamics_contact_configuration(
                dynamics_before, expected=plan_contact_configuration
            )
            if phase["mode"] == "hold_current_joint_positions":
                action = environment.hold_action(gripper_command=gripper_command)
                action_recomputed = True
                action_hold_index = 0
            else:
                action_hold_steps = int(phase["action_hold_steps"])
                action_hold_index = phase_step_index % action_hold_steps
                action_recomputed = held_action is None or action_hold_index == 0
                solved_hold = phase.get("hold_solved_arm_joint_positions_rad")
                bounded = getattr(environment, "bounded_joint_action", None)
                if solved_hold is not None:
                    if not callable(bounded):
                        raise ControlEpisodeError(
                            [
                                "control_episode_solved_joint_dispatch_unavailable:"
                                f"{phase['phase_id']}"
                            ]
                        )
                    # Command the posture the preflight solved for this pose
                    # rather than letting the Cartesian controller re-derive
                    # one.  The arrival gate is untouched: it still measures
                    # the real fingertip against the sealed target, so a solved
                    # vector that does not put it there still fails honestly.
                    if action_recomputed:
                        held_action = [
                            float(value)
                            for value in bounded(
                                target_joint_positions_rad=list(solved_hold),
                                gripper_command=gripper_command,
                                max_joint_delta_rad=float(
                                    phase["max_joint_delta_rad"]
                                ),
                                max_joint_setpoint_lead_rad=float(
                                    phase["max_joint_setpoint_lead_rad"]
                                ),
                            )
                        ]
                elif action_recomputed:
                    held_action = environment.scripted_action_for_pose(
                        phase_id=str(phase["phase_id"]),
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
            dynamics_after = _canonical_dynamics_observation(environment.read_arm_dynamics_observation())
            _require_dynamics_contact_envelope(
                dynamics_after,
                expected=plan_contact_envelope,
            )
            _require_dynamics_contact_configuration(
                dynamics_after, expected=plan_contact_configuration
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
        "control_plan_semantic_digest": plan["semantic_plan_digest"],
        "physics_backend": plan.get("physics_backend", "physx"),
        "backend_contact_configuration": plan_contact_configuration,
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
    plan_contact_configuration = _plan_backend_contact_configuration(plan)
    for receipt in controls:
        dynamics_summary = receipt.get("arm_dynamics_summary")
        if (
            receipt.get("contact_envelope") != plan_contact_envelope
            or receipt.get("backend_contact_configuration")
            != plan_contact_configuration
            or not isinstance(dynamics_summary, Mapping)
            or dynamics_summary.get("contact_envelope") != plan_contact_envelope
            or dynamics_summary.get("backend_contact_configuration")
            != plan_contact_configuration
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
        "control_plan_semantic_digest": plan["semantic_plan_digest"],
        "physics_backend": plan.get("physics_backend", "physx"),
        "backend_contact_configuration": plan_contact_configuration,
        "contact_envelope": plan_contact_envelope,
        "execution_order": list(REQUIRED_CONTROLS),
        "controls": [
            {
                "control_id": receipt["control_id"],
                "control_passed": receipt["control_passed"],
                "observed_outcome": receipt["observed_outcome"],
                "backend_contact_configuration": receipt[
                    "backend_contact_configuration"
                ],
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
    "DOWNSTREAM_DIAGNOSTIC_PHASE_IDS",
    "ControlEpisodeError",
    "SCRIPTED_POSITIVE",
    "ZERO_ACTION_NEGATIVE",
    "materialize_control_plan",
    "run_control_episode",
    "run_required_controls",
]


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


def _task_contact_pad_forces_n(sample: Mapping[str, Any]) -> dict[str, float]:
    """Return attributed inner-finger contact peaks from native readback."""

    native = sample.get("native_readback")
    if not isinstance(native, Mapping):
        return {}
    instance_readback = native.get("contact_sensor_instance_readback")
    if not isinstance(instance_readback, Mapping):
        return {}
    instances = instance_readback.get("task_robot_contact")
    if not isinstance(instances, Sequence) or isinstance(instances, (str, bytes)):
        return {}
    peaks: dict[str, float] = {}
    for instance in instances:
        if not isinstance(instance, Mapping):
            continue
        forces = instance.get("nonzero_filter_forces")
        if not isinstance(forces, Sequence) or isinstance(forces, (str, bytes)):
            continue
        for force in forces:
            if not isinstance(force, Mapping):
                continue
            path = str(force.get("filter_prim_path_expr") or "")
            side = next(
                (
                    candidate
                    for candidate in ("left_inner_finger", "right_inner_finger")
                    if candidate in path
                ),
                None,
            )
            if side is None:
                continue
            try:
                magnitude = float(force["force_magnitude_n"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(magnitude) and magnitude >= 0.0:
                peaks[side] = max(peaks.get(side, 0.0), magnitude)
    return peaks


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
    if checked.get("plan_digest") != canonical_digest(
        checked, digest_field="plan_digest"
    ):
        errors.append("task_control_plan_digest_mismatch")
    if checked.get("task_spec_digest") != canonical_digest(task):
        errors.append("task_control_plan_task_spec_mismatch")
    diagnostic_plan, boundary_errors = control_plan_boundary_errors(checked)
    errors.extend(boundary_errors)
    attempt_limit = checked.get("maximum_pose_phase_attempts", TASK_CONTROL_MAX_POSE_PHASE_ATTEMPTS)
    if type(attempt_limit) is not int or not 1 <= attempt_limit <= TASK_CONTROL_MAX_POSE_PHASE_ATTEMPTS:
        errors.append("task_control_phase_attempt_limit_invalid")
    planner_receipt_digest = str(checked.get("planner_receipt_digest") or "")
    if not planner_receipt_digest.startswith("sha256:") or len(
        planner_receipt_digest
    ) != 71:
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
                arrival_position_raw = raw.get("arrival_target_position_world_m")
                arrival_position = (
                    None
                    if arrival_position_raw is None
                    else [float(value) for value in arrival_position_raw]
                )
                quaternion_raw = raw.get("target_quaternion_world_xyzw")
                quaternion = (
                    None
                    if quaternion_raw is None
                    else [float(value) for value in quaternion_raw]
                )
                minimum_steps = int(raw["minimum_steps"])
                maximum_steps = int(raw["maximum_steps"])
                arrival_tolerance_m = float(raw["arrival_tolerance_m"])
                arrival_stability_steps = int(raw["arrival_stability_steps"])
                orientation_tolerance_raw = raw.get(
                    "arrival_orientation_tolerance_rad"
                )
                arrival_orientation_tolerance_rad = (
                    None
                    if orientation_tolerance_raw is None
                    else float(orientation_tolerance_raw)
                )
                max_joint_delta_rad = float(raw["max_joint_delta_rad"])
                max_joint_setpoint_lead_rad = float(
                    raw["max_joint_setpoint_lead_rad"]
                )
            except (KeyError, TypeError, ValueError):
                position = []
                arrival_position = []
                quaternion = []
                minimum_steps = 0
                maximum_steps = 0
                arrival_tolerance_m = 0.0
                arrival_stability_steps = 0
                arrival_orientation_tolerance_rad = None
                max_joint_delta_rad = 0.0
                max_joint_setpoint_lead_rad = 0.0
            gripper_state = str(raw.get("gripper_state") or "")
            position_only_arrival = raw.get("position_only_arrival") is True
            # A phase may carry the joint vector its own preflight solved for
            # this pose.  C42 and C43 measured why that matters: the Cartesian
            # controller re-derives a posture from scratch and walked 0.19 to
            # 0.53 rad away from the solved vector, whose own forward
            # kinematics sat inside the arrival gate, onto one whose kinematics
            # were already 20 mm outside it.  Tracking was never the problem --
            # the arm reached what it was told to within 0.008 rad.  It was
            # told the wrong thing.
            held_raw = raw.get("hold_solved_arm_joint_positions_rad")
            try:
                held_joints = (
                    None
                    if held_raw is None
                    else [float(value) for value in held_raw]
                )
            except (TypeError, ValueError):
                held_joints = []
            held_valid = held_joints is None or (
                len(held_joints) == 7
                and all(math.isfinite(value) for value in held_joints)
            )
            physx_preferred_raw = raw.get(
                "physx_dls_preferred_posture_joint_positions_rad"
            )
            try:
                physx_preferred_joints = (
                    None
                    if physx_preferred_raw is None
                    else [float(value) for value in physx_preferred_raw]
                )
            except (TypeError, ValueError):
                physx_preferred_joints = []
            physx_preferred_valid = physx_preferred_joints is None or (
                len(physx_preferred_joints) == 7
                and all(math.isfinite(value) for value in physx_preferred_joints)
            )
            expected_task_joints_raw = raw.get("expected_joint_positions")
            try:
                expected_task_joints = (
                    None
                    if expected_task_joints_raw is None
                    else {
                        str(name): float(value)
                        for name, value in expected_task_joints_raw.items()
                    }
                )
            except (AttributeError, TypeError, ValueError):
                expected_task_joints = {}
            expected_task_joints_valid = expected_task_joints is None or (
                bool(expected_task_joints)
                and all(
                    name and math.isfinite(value)
                    for name, value in expected_task_joints.items()
                )
            )
            hold_arm_during_gripper_transition = (
                raw.get("hold_arm_joint_positions_during_gripper_transition")
                is True
            )
            require_bilateral_task_contact = (
                raw.get("require_bilateral_task_contact") is True
            )
            bilateral_threshold_raw = raw.get(
                "bilateral_task_contact_minimum_force_n"
            )
            try:
                bilateral_threshold = (
                    None
                    if bilateral_threshold_raw is None
                    else float(bilateral_threshold_raw)
                )
            except (TypeError, ValueError):
                bilateral_threshold = math.nan
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
                or not (
                    arrival_position is None
                    or (
                        len(arrival_position) == 3
                        and all(math.isfinite(value) for value in arrival_position)
                    )
                )
                or not quaternion_valid
                or gripper_state not in {"open", "closed"}
                or minimum_steps < 1
                or maximum_steps < minimum_steps
                or arrival_tolerance_m <= 0.0
                or not math.isfinite(arrival_tolerance_m)
                or arrival_stability_steps < 1
                or (
                    quaternion is not None
                    and task.get("schema_version") == "adp_task_spec.v2"
                    and arrival_orientation_tolerance_rad is None
                    and not position_only_arrival
                )
                or (
                    position_only_arrival
                    and (phase_id != "prealign" or gripper_state != "open")
                )
                or (
                    hold_arm_during_gripper_transition
                    and (phase_id, gripper_state)
                    not in {("contact_close", "closed"), ("release", "open")}
                )
                or (
                    require_bilateral_task_contact
                    and (
                        (phase_id, gripper_state) != ("contact_close", "closed")
                        or bilateral_threshold is None
                        or not math.isfinite(bilateral_threshold)
                        or bilateral_threshold <= 0.0
                    )
                )
                or (
                    not require_bilateral_task_contact
                    and bilateral_threshold_raw is not None
                )
                or (
                    arrival_orientation_tolerance_rad is not None
                    and (
                        arrival_orientation_tolerance_rad <= 0.0
                        or not math.isfinite(arrival_orientation_tolerance_rad)
                    )
                )
                or max_joint_delta_rad <= 0.0
                or not math.isfinite(max_joint_delta_rad)
                or max_joint_setpoint_lead_rad < max_joint_delta_rad
                or not math.isfinite(max_joint_setpoint_lead_rad)
                or not held_valid
                or not physx_preferred_valid
                or not expected_task_joints_valid
                or (
                    held_joints is not None
                    and physx_preferred_joints is not None
                )
                or (
                    physx_preferred_joints is not None
                    and (phase_id, gripper_state)
                    != ("contact_close", "closed")
                )
            ):
                errors.append(f"task_control_scripted_pose_invalid:{index}")
            else:
                normalized_actions.append(
                    {
                        "phase_id": phase_id,
                        "mode": "ik_pose",
                        "target_position_world_m": position,
                        "arrival_target_position_world_m": arrival_position,
                        "target_quaternion_world_xyzw": quaternion,
                        "hold_solved_arm_joint_positions_rad": held_joints,
                        "physx_dls_preferred_posture_joint_positions_rad": (
                            physx_preferred_joints
                        ),
                        "gripper_state": gripper_state,
                        "minimum_steps": minimum_steps,
                        "maximum_steps": maximum_steps,
                        "arrival_tolerance_m": arrival_tolerance_m,
                        "arrival_stability_steps": arrival_stability_steps,
                        "arrival_orientation_tolerance_rad": (
                            arrival_orientation_tolerance_rad
                        ),
                        "position_only_arrival": position_only_arrival,
                        "hold_arm_joint_positions_during_gripper_transition": (
                            hold_arm_during_gripper_transition
                        ),
                        "require_bilateral_task_contact": (
                            require_bilateral_task_contact
                        ),
                        "bilateral_task_contact_minimum_force_n": (
                            bilateral_threshold
                        ),
                        "max_joint_delta_rad": max_joint_delta_rad,
                        "max_joint_setpoint_lead_rad": max_joint_setpoint_lead_rad,
                        **(
                            {
                                "expected_joint_positions": (
                                    expected_task_joints
                                )
                            }
                            if expected_task_joints is not None
                            else {}
                        ),
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
                normalized_actions.append(
                    {"phase_id": phase_id, "isaac_action": action}
                )
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
            normalized = {
                "phase_id": phase_id,
                "arm_joint_positions": arm,
                "gripper_state": gripper_state,
            }
            # A joint row may carry the same actuator bounds a pose row does.
            # Dropping them here would silently return the row to the
            # unbounded command path that saturated C33's entry ramp, so they
            # are preserved when both are present and well formed.
            try:
                slew = float(raw["max_joint_delta_rad"])
                lead = float(raw["max_joint_setpoint_lead_rad"])
            except (KeyError, TypeError, ValueError):
                slew = lead = None
            if (
                slew is not None
                and lead is not None
                and math.isfinite(slew)
                and math.isfinite(lead)
                and slew > 0.0
                and lead >= slew
            ):
                normalized["max_joint_delta_rad"] = slew
                normalized["max_joint_setpoint_lead_rad"] = lead
            normalized_actions.append(normalized)
            maximum_scripted_steps += 1
    for index, action in enumerate(normalized_actions):
        if not action.get(
            "hold_arm_joint_positions_during_gripper_transition"
        ):
            continue
        if index == 0:
            errors.append("task_control_gripper_transition_predecessor_invalid")
            continue
        previous = normalized_actions[index - 1]
        phase_id = action["phase_id"]
        same_pose = (
            previous.get("mode") == "ik_pose"
            and previous.get("target_position_world_m")
            == action.get("target_position_world_m")
            and previous.get("target_quaternion_world_xyzw")
            == action.get("target_quaternion_world_xyzw")
        )
        predecessor_valid = (
            phase_id == "contact_close"
            and previous.get("phase_id") == "contact_open"
            and previous.get("gripper_state") == "open"
        ) or (
            phase_id == "release"
            and previous.get("gripper_state") == "closed"
        )
        if not same_pose or not predecessor_valid:
            errors.append("task_control_gripper_transition_predecessor_invalid")
    kind = task.get("task_kind")
    if kind == TASK_KIND_ARTICULATED_OPEN_CLOSE:
        try:
            validate_articulated_task_spec(task)
        except TaskNeutralScoringError as exc:
            errors.extend(exc.errors)
    elif kind != TASK_KIND_RIGID_PICK_PLACE:
        errors.append("task_control_task_kind_unsupported")
    settle_steps = task.get("settle_window_samples")
    if (
        isinstance(settle_steps, bool)
        or not isinstance(settle_steps, int)
        or settle_steps < 1
    ):
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
    initial_reset_callback: Callable[[], None] | None = None,
    trajectory_override: Sequence[Mapping[str, Any]] | None = None,
    qualification_allowed: bool = True,
    receipt_annotations: Mapping[str, Any] | None = None,
    initial_sample_validator: Callable[[Mapping[str, Any]], str | None]
    | None = None,
) -> dict[str, Any]:
    task_kind = str(task_spec["task_kind"])
    if initial_reset_callback is None:
        environment.reset()
    else:
        initial_reset_callback()
    if callable(getattr(environment, "begin_episode", None)):
        environment.begin_episode()
    samples = [
        _task_neutral_sample(environment, task_kind=task_kind, step_index=0)
    ]
    initial_state_blocker = (
        initial_sample_validator(samples[0])
        if initial_sample_validator is not None
        else None
    )
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
    phase_execution_blocker: str | None = initial_state_blocker
    observation_index = 1
    step_index = 0
    if trajectory_override is not None:
        trajectory = (
            []
            if initial_state_blocker is not None
            else [dict(row) for row in trajectory_override]
        )
    elif control_id == ZERO_ACTION_NEGATIVE:
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
    row_index = 0
    attempt_number = 1
    commanded_position_bias = [0.0, 0.0, 0.0]
    solved_joint_command_bias = [0.0] * 7
    current_strategy: str | None = None
    attempt_history: list[dict[str, Any]] = []
    recovery_ladder = recovery_ladder_for_plan(plan)
    phase_attempt_limit = plan.get("maximum_pose_phase_attempts", TASK_CONTROL_MAX_POSE_PHASE_ATTEMPTS)
    while row_index < len(trajectory):
        row = trajectory[row_index]
        pose_mode = row.get("mode") == "ik_pose"
        hold_arm_during_gripper_transition = bool(
            pose_mode
            and row.get(
                "hold_arm_joint_positions_during_gripper_transition"
            )
            is True
        )
        require_bilateral_task_contact = bool(
            pose_mode and row.get("require_bilateral_task_contact") is True
        )
        bilateral_contact_threshold_n = (
            float(row["bilateral_task_contact_minimum_force_n"])
            if require_bilateral_task_contact
            else None
        )
        phase_steps = int(row["maximum_steps"]) if pose_mode else 1
        phase_steps_executed = 0
        stable_steps = 0
        bilateral_contact_stability_steps = 0
        terminal_pad_forces_n: dict[str, float] = {}
        termination_reason = "fixed_steps_completed"
        start_sample = samples[-1]
        held_arm_joint_positions = (
            [float(value) for value in environment.read_arm_joint_positions()]
            if hold_arm_during_gripper_transition
            else None
        )
        explicit_arrival_target = (
            row.get("arrival_target_position_world_m") is not None
        )
        arrival_target_position = row.get("arrival_target_position_world_m")
        if not explicit_arrival_target:
            arrival_target_position = row.get("target_position_world_m")
        arrival_target_orientation = row.get("target_quaternion_world_xyzw")
        arrival_target_source = (
            "sealed_arrival_pose_override"
            if row.get("arrival_target_position_world_m") is not None
            else "commanded_phase_pose"
        )
        # Holding the reached arm joints is a command policy, not permission
        # to move the scientific arrival gate.  An explicit sealed target
        # remains authoritative through gripper motion; only legacy hold rows
        # without one fall back to the qualified entry pose.
        if hold_arm_during_gripper_transition and not explicit_arrival_target:
            arrival_target_position = start_sample.get(
                "grasp_frame_position_world_m"
            )
            arrival_target_orientation = start_sample.get(
                "grasp_frame_orientation_world_xyzw"
            )
            if (
                not isinstance(arrival_target_position, Sequence)
                or isinstance(arrival_target_position, (str, bytes))
                or not isinstance(arrival_target_orientation, Sequence)
                or isinstance(arrival_target_orientation, (str, bytes))
            ):
                raise ControlEpisodeError(
                    ["task_control_gripper_transition_entry_pose_missing"]
                )
            arrival_target_position = [
                float(value) for value in arrival_target_position
            ]
            arrival_target_orientation = [
                float(value) for value in arrival_target_orientation
            ]
            arrival_target_source = (
                "previous_phase_qualified_entry_pose_held_during_gripper_"
                "transition"
            )
        # The command may carry a recovery bias; the arrival gate never does.
        commanded_position = row.get("target_position_world_m")
        if pose_mode and not hold_arm_during_gripper_transition:
            commanded_position = [
                float(value) + float(bias)
                for value, bias in zip(
                    row["target_position_world_m"], commanded_position_bias
                )
            ]
        for _ in range(phase_steps):
            before = [float(value) for value in environment.read_arm_joint_positions()]
            dynamics_before = _canonical_dynamics_observation(
                environment.read_arm_dynamics_observation()
            )
            if row.get("mode") == "hold_current_joint_positions":
                action = environment.hold_action(
                    gripper_command=float(gripper_open_command)
                )
            elif pose_mode:
                state = str(row["gripper_state"])
                if state == "closed" and gripper_closed_command is None:
                    raise ControlEpisodeError(
                        ["task_control_gripper_closed_command_missing"]
                    )
                command = (
                    float(gripper_open_command)
                    if state == "open"
                    else float(gripper_closed_command)
                )
                solved_hold = row.get("hold_solved_arm_joint_positions_rad")
                if held_arm_joint_positions is not None:
                    action = [*held_arm_joint_positions, command]
                elif solved_hold is not None:
                    bounded = getattr(environment, "bounded_joint_action", None)
                    if not callable(bounded):
                        raise ControlEpisodeError(
                            [
                                "task_control_solved_joint_dispatch_unavailable:"
                                f"{row['phase_id']}"
                            ]
                        )
                    action = bounded(
                        target_joint_positions_rad=[
                            float(value) + float(bias)
                            for value, bias in zip(
                                solved_hold, solved_joint_command_bias
                            )
                        ],
                        gripper_command=command,
                        max_joint_delta_rad=float(row["max_joint_delta_rad"]),
                        max_joint_setpoint_lead_rad=float(
                            row["max_joint_setpoint_lead_rad"]
                        ),
                    )
                else:
                    pose_action_kwargs = {
                        "phase_id": str(row["phase_id"]),
                        "target_position_world_m": commanded_position,
                        "target_quaternion_world_xyzw": row[
                            "target_quaternion_world_xyzw"
                        ],
                        "gripper_command": command,
                        "max_joint_delta_rad": float(
                            row["max_joint_delta_rad"]
                        ),
                        "max_joint_setpoint_lead_rad": float(
                            row["max_joint_setpoint_lead_rad"]
                        ),
                    }
                    physx_preferred = row.get(
                        "physx_dls_preferred_posture_joint_positions_rad"
                    )
                    if physx_preferred is not None:
                        pose_action_kwargs[
                            "preferred_posture_joint_positions_rad"
                        ] = physx_preferred
                    action = environment.scripted_action_for_pose(
                        **pose_action_kwargs
                    )
            elif "isaac_action" in row:
                action = row["isaac_action"]
            else:
                state = str(row["gripper_state"])
                if state == "closed" and gripper_closed_command is None:
                    raise ControlEpisodeError(
                        ["task_control_gripper_closed_command_missing"]
                    )
                command = (
                    float(gripper_open_command)
                    if state == "open"
                    else float(gripper_closed_command)
                )
                bounded = getattr(environment, "bounded_joint_action", None)
                lead = row.get("max_joint_setpoint_lead_rad")
                slew = row.get("max_joint_delta_rad")
                if callable(bounded) and lead is not None and slew is not None:
                    # Raw joint rows bypassed the servo, so nothing held them
                    # to what the actuator can pull.  C33's entry ramp ran the
                    # command ahead of a lagging wrist for 37% of its rows and
                    # ended further from the handle than the shorter ramp
                    # before it.  Same targets, now under the servo's bounds,
                    # which is also what turns a repeated target into a ramp
                    # at exactly the rate the joint can follow.
                    action = bounded(
                        target_joint_positions_rad=row["arm_joint_positions"],
                        gripper_command=command,
                        max_joint_delta_rad=float(slew),
                        max_joint_setpoint_lead_rad=float(lead),
                    )
                else:
                    action = [*row["arm_joint_positions"], command]
            action = [float(value) for value in action]
            environment.step(action)
            step_index += 1
            phase_steps_executed += 1
            after = [float(value) for value in environment.read_arm_joint_positions()]
            dynamics_after = _canonical_dynamics_observation(
                environment.read_arm_dynamics_observation()
            )
            actions.append(
                _record_action(
                    step_index=step_index,
                    phase_id=str(row["phase_id"]),
                    action=action,
                    observed_before=before,
                    observed_after=after,
                    dynamics_before=dynamics_before, dynamics_after=dynamics_after,
                    action_recomputed=(
                        not hold_arm_during_gripper_transition
                    ),
                    action_hold_index=(
                        phase_steps_executed - 1
                        if hold_arm_during_gripper_transition
                        else 0
                    ),
                )
            )
            samples.append(
                _task_neutral_sample(
                    environment, task_kind=task_kind, step_index=step_index
                )
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
                if not isinstance(measured, Sequence) or isinstance(
                    measured, (str, bytes)
                ):
                    raise ControlEpisodeError(
                        ["task_control_grasp_frame_readback_missing"]
                    )
                try:
                    error = math.dist(
                        [float(value) for value in measured],
                        arrival_target_position,
                    )
                except (TypeError, ValueError) as exc:
                    raise ControlEpisodeError(
                        ["task_control_grasp_frame_readback_invalid"]
                    ) from exc
                orientation_error = None
                orientation_tolerance = row.get(
                    "arrival_orientation_tolerance_rad"
                )
                if orientation_tolerance is not None:
                    measured_orientation = samples[-1].get(
                        "grasp_frame_orientation_world_xyzw"
                    )
                    if not isinstance(measured_orientation, Sequence) or isinstance(
                        measured_orientation, (str, bytes)
                    ):
                        raise ControlEpisodeError(
                            ["task_control_grasp_frame_orientation_missing"]
                        )
                    orientation_error = _quaternion_angle_xyzw(
                        measured_orientation,
                        arrival_target_orientation,
                    )
                arrived = error <= float(row["arrival_tolerance_m"]) and (
                    orientation_error is None
                    or orientation_error <= float(orientation_tolerance)
                )
                if require_bilateral_task_contact:
                    terminal_pad_forces_n = _task_contact_pad_forces_n(samples[-1])
                    bilateral_active = all(
                        terminal_pad_forces_n.get(side, 0.0)
                        >= float(bilateral_contact_threshold_n)
                        for side in ("left_inner_finger", "right_inner_finger")
                    )
                    bilateral_contact_stability_steps = (
                        bilateral_contact_stability_steps + 1
                        if bilateral_active
                        else 0
                    )
                    arrived = arrived and bilateral_active
                stable_steps = (
                    stable_steps + 1 if arrived else 0
                )
                if (
                    phase_steps_executed >= int(row["minimum_steps"])
                    and stable_steps >= int(row["arrival_stability_steps"])
                ):
                    termination_reason = "stable_arrival"
                    break
        if pose_mode:
            measured = samples[-1].get("grasp_frame_position_world_m")
            terminal_error = math.dist(
                [float(value) for value in measured], arrival_target_position
            )
            orientation_tolerance = row.get("arrival_orientation_tolerance_rad")
            measured_orientation = samples[-1].get(
                "grasp_frame_orientation_world_xyzw"
            )
            terminal_orientation_error = (
                None
                if orientation_tolerance is None
                else _quaternion_angle_xyzw(
                    measured_orientation,
                    arrival_target_orientation,
                )
            )
            terminal_reached_joints = [
                float(value) for value in environment.read_arm_joint_positions()
            ]
            terminal_commanded_joints = [
                float(value)
                for value in actions[-1]["arm_dynamics_after"][
                    "joint_position_target_rad"
                ]
            ]
            selected_joints = row.get("hold_solved_arm_joint_positions_rad")
            selected_joints = (
                [float(value) for value in selected_joints]
                if selected_joints is not None
                else None
            )
            predictor = getattr(
                environment, "predict_grasp_frame_pose_world", None
            )
            predicted_fk_pose = None
            if callable(predictor):
                try:
                    candidate = predictor(
                        terminal_reached_joints, gripper_command=command
                    )
                    if candidate is not None:
                        candidate = [float(value) for value in candidate]
                        if len(candidate) == 7 and all(
                            math.isfinite(value) for value in candidate
                        ):
                            predicted_fk_pose = candidate
                except Exception:  # noqa: BLE001 - retain an explicit gap
                    predicted_fk_pose = None
            arrival = {
                "phase_id": str(row["phase_id"]),
                "start_position_world_m": start_sample.get(
                    "grasp_frame_position_world_m"
                ),
                "target_position_world_m": arrival_target_position,
                "commanded_target_position_world_m": row[
                    "target_position_world_m"
                ],
                "arrival_target_source": arrival_target_source,
                "terminal_position_world_m": measured,
                "terminal_position_error_m": terminal_error,
                "arrival_tolerance_m": float(row["arrival_tolerance_m"]),
                "target_orientation_world_xyzw": arrival_target_orientation,
                "commanded_target_orientation_world_xyzw": row[
                    "target_quaternion_world_xyzw"
                ],
                "terminal_orientation_world_xyzw": measured_orientation,
                "terminal_orientation_error_rad": terminal_orientation_error,
                "arrival_orientation_tolerance_rad": orientation_tolerance,
                "arrival_stability_steps_required": int(
                    row["arrival_stability_steps"]
                ),
                "arrival_stability_steps_observed": stable_steps,
                "bilateral_task_contact_required": (
                    require_bilateral_task_contact
                ),
                "bilateral_task_contact_minimum_force_n": (
                    bilateral_contact_threshold_n
                ),
                "terminal_task_contact_pad_forces_n": terminal_pad_forces_n,
                "terminal_bilateral_task_contact_active": (
                    require_bilateral_task_contact
                    and all(
                        terminal_pad_forces_n.get(side, 0.0)
                        >= float(bilateral_contact_threshold_n)
                        for side in ("left_inner_finger", "right_inner_finger")
                    )
                ),
                "bilateral_task_contact_stability_steps_observed": (
                    bilateral_contact_stability_steps
                ),
                "termination_reason": termination_reason,
                "target_reached": termination_reason == "stable_arrival",
                "selected_joint_positions_rad": selected_joints,
                "arm_command_source": (
                    "solved_joint_target"
                    if selected_joints is not None and current_strategy is None
                    else "joint_tracking_recovery_from_solved_branch"
                    if selected_joints is not None
                    else "live_physx_dls_with_preferred_posture"
                    if row.get(
                        "physx_dls_preferred_posture_joint_positions_rad"
                    )
                    is not None
                    else "cartesian_pose_servo"
                ),
                "solved_joint_command_bias_rad": (
                    list(solved_joint_command_bias)
                    if selected_joints is not None
                    else None
                ),
                "terminal_commanded_joint_positions_rad": terminal_commanded_joints,
                "terminal_reached_joint_positions_rad": terminal_reached_joints,
                "selected_to_commanded_joint_l2_rad": (
                    math.dist(selected_joints, terminal_commanded_joints)
                    if selected_joints is not None
                    else None
                ),
                "commanded_to_reached_joint_l2_rad": math.dist(
                    terminal_commanded_joints, terminal_reached_joints
                ),
                "terminal_fk_grasp_frame_position_world_m": (
                    predicted_fk_pose[:3] if predicted_fk_pose is not None else None
                ),
                "terminal_fk_to_measured_tcp_error_m": (
                    math.dist(predicted_fk_pose[:3], measured)
                    if predicted_fk_pose is not None
                    else None
                ),
                "terminal_fk_status": (
                    "measured" if predicted_fk_pose is not None else "unavailable"
                ),
            }
            arrival["attempt"] = attempt_number
            arrival["recovery_strategy"] = current_strategy
            arrival["commanded_position_bias_m"] = [
                float(value) for value in commanded_position_bias
            ]
            phase_arrivals.append(arrival)
            if not arrival["target_reached"]:
                attempt_history.append(
                    {"strategy": current_strategy, "error_m": terminal_error}
                )
                selected_tracking_error = (
                    math.dist(selected_joints, terminal_reached_joints)
                    if selected_joints is not None
                    else None
                )
                if (
                    selected_joints is not None
                    and not hold_arm_during_gripper_transition
                    and attempt_number < phase_attempt_limit
                    and selected_tracking_error is not None
                    and selected_tracking_error > 1.0e-4
                ):
                    # C60 proved the selected posture's own FK is inside the
                    # contact gate while the actuator settles about 0.011 rad
                    # away. C61 then proved a Cartesian retry abandons that
                    # good branch by 0.56--1.62 rad and worsens the TCP miss.
                    # Compensate the measured actuator residual in joint
                    # space instead: if command q settles at q+e, command
                    # q-e next. The bounded joint seam still clips every
                    # target to the live limits and lead budget.
                    solved_joint_command_bias = [
                        float(bias) + float(selected) - float(reached)
                        for bias, selected, reached in zip(
                            solved_joint_command_bias,
                            selected_joints,
                            terminal_reached_joints,
                        )
                    ]
                    current_strategy = "measured_joint_tracking_compensation"
                    attempt_number += 1
                    continue
                next_strategy = (
                    None
                    if selected_joints is not None
                    else _next_recovery_strategy(
                        attempt_history,
                        ladder=recovery_ladder,
                        arrival_tolerance_m=float(row["arrival_tolerance_m"]),
                        remaining_attempts=(
                            phase_attempt_limit - attempt_number
                        ),
                    )
                )
                if (
                    attempt_number < phase_attempt_limit
                    and not hold_arm_during_gripper_transition
                    # A position bias cannot repair an orientation-only miss;
                    # retry only when the position gate itself failed.
                    and terminal_error > float(row["arrival_tolerance_m"])
                    and next_strategy is not None
                ):
                    # Bounded retreat toward the phase's already-achieved
                    # entry pose so a limit-saturated arm regains room, then
                    # re-enter under the strategy the trend selected.  Every
                    # retreat step is recorded in the same traces.
                    retreat_position = start_sample.get(
                        "grasp_frame_position_world_m"
                    )
                    retreat_orientation = (
                        start_sample.get("grasp_frame_orientation_world_xyzw")
                        or row.get("target_quaternion_world_xyzw")
                    )
                    if isinstance(
                        retreat_position, Sequence
                    ) and not isinstance(retreat_position, (str, bytes)):
                        standoff_scale = (
                            TASK_CONTROL_RECOVERY_EXTENDED_STANDOFF_SCALE
                            if next_strategy == "extended_standoff_reentry"
                            else 1.0
                        )
                        # scale 1.0 is exactly the qualified entry pose; a
                        # larger scale backs further out along the same line
                        # so re-entry travels a longer straight approach.
                        retreat_target = [
                            float(target)
                            + standoff_scale * (float(entry) - float(target))
                            for target, entry in zip(
                                arrival_target_position, retreat_position
                            )
                        ]
                        for _ in range(
                            TASK_CONTROL_RECOVERY_RETREAT_MAXIMUM_STEPS
                        ):
                            before = [
                                float(value)
                                for value in environment.read_arm_joint_positions()
                            ]
                            dynamics_before = _canonical_dynamics_observation(
                                environment.read_arm_dynamics_observation()
                            )
                            action = environment.scripted_action_for_pose(
                                phase_id=str(row["phase_id"]),
                                target_position_world_m=retreat_target,
                                target_quaternion_world_xyzw=retreat_orientation,
                                gripper_command=command,
                                max_joint_delta_rad=float(
                                    row["max_joint_delta_rad"]
                                ),
                                max_joint_setpoint_lead_rad=float(
                                    row["max_joint_setpoint_lead_rad"]
                                ),
                            )
                            action = [float(value) for value in action]
                            environment.step(action)
                            step_index += 1
                            after = [
                                float(value)
                                for value in environment.read_arm_joint_positions()
                            ]
                            dynamics_after = _canonical_dynamics_observation(
                                environment.read_arm_dynamics_observation()
                            )
                            actions.append(
                                _record_action(
                                    step_index=step_index,
                                    phase_id=str(row["phase_id"]),
                                    action=action,
                                    observed_before=before,
                                    observed_after=after,
                                    dynamics_before=dynamics_before,
                                    dynamics_after=dynamics_after,
                                    action_recomputed=True,
                                    action_hold_index=0,
                                )
                            )
                            samples.append(
                                _task_neutral_sample(
                                    environment,
                                    task_kind=task_kind,
                                    step_index=step_index,
                                )
                            )
                            reached_back = samples[-1].get(
                                "grasp_frame_position_world_m"
                            )
                            if (
                                isinstance(reached_back, Sequence)
                                and not isinstance(reached_back, (str, bytes))
                                and math.dist(
                                    [float(value) for value in reached_back],
                                    retreat_target,
                                )
                                <= 4.0 * float(row["arrival_tolerance_m"])
                            ):
                                break
                    measured_miss = [
                        float(target) - float(value)
                        for target, value in zip(
                            arrival_target_position, measured
                        )
                    ]
                    if next_strategy == "measured_miss_compensation":
                        commanded_position_bias = [
                            float(bias) + miss
                            for bias, miss in zip(
                                commanded_position_bias, measured_miss
                            )
                        ]
                    elif next_strategy == "damped_half_miss_compensation":
                        commanded_position_bias = [
                            float(bias) + 0.5 * miss
                            for bias, miss in zip(
                                commanded_position_bias, measured_miss
                            )
                        ]
                    else:
                        # Both re-entry rungs test the unbiased command from a
                        # cleaner starting pose, so an accumulated bias would
                        # confound exactly what they are meant to isolate.
                        commanded_position_bias = [0.0, 0.0, 0.0]
                    current_strategy = next_strategy
                    attempt_number += 1
                    continue
                # Report the phase's best attempt, not its last.  C32 ended
                # claiming 15.39 mm after its own first attempt had reached
                # 11.63 mm: later rungs made it worse, and the run threw the
                # better result away.  The blocker is the evidence a human or
                # an agent reads to choose the next hypothesis, so it has to
                # carry what this phase actually achieved.
                best = min(
                    attempt_history, key=lambda row_: float(row_["error_m"])
                )
                best_attempt = attempt_history.index(best) + 1
                phase_execution_blocker = (
                    f"{BLOCKER_PHASE_NOT_REACHED}:{row['phase_id']}:"
                    f"error_m={float(best['error_m']):.6f}"
                    f":best_attempt={best_attempt}"
                    f":best_strategy={best.get('strategy')}"
                    f":final_error_m={terminal_error:.6f}"
                    f":orientation_error_rad={terminal_orientation_error}"
                    f":stability_steps="
                    f"{stable_steps}/{row['arrival_stability_steps']}"
                    f":attempts={attempt_number}"
                    f":strategies_exhausted={next_strategy is None}"
                    + (
                        ":bilateral_task_contact=false"
                        f":pad_forces_n={terminal_pad_forces_n}"
                        f":contact_threshold_n={bilateral_contact_threshold_n}"
                        if require_bilateral_task_contact
                        else ""
                    )
                )
                break
        row_index += 1
        attempt_number = 1
        commanded_position_bias = [0.0, 0.0, 0.0]
        solved_joint_command_bias = [0.0] * 7
        current_strategy = None
        attempt_history = []
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
    if not qualification_allowed:
        blockers.append("synthetic_checkpoint_diagnostic_not_qualifying")
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
    apply_diagnostic_receipt_boundary(
        receipt, qualification_allowed=qualification_allowed
    )
    copy_diagnostic_annotations(receipt, receipt_annotations)
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def run_task_neutral_controls(
    *,
    environment: ControlEnvironment,
    task_spec: Mapping[str, Any],
    control_plan: Mapping[str, Any],
    gripper_open_command: float,
    gripper_closed_command: float | None = None,
    output_dir: str | Path,
    qualification_allowed: bool = True,
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
                None
                if gripper_closed_command is None
                else float(gripper_closed_command)
            ),
            output=output,
            episode_id=f"{plan['cell_id']}-{control_id}",
            qualification_allowed=bool(qualification_allowed),
            receipt_annotations=diagnostic_receipt_annotations(
                plan, qualification_allowed=qualification_allowed
            ),
        )
        receipts.append(receipt)
        _write_json(output / f"adp_task_control_episode.{control_id}.json", receipt)
    pair = build_task_control_pair(
        plan=plan,
        task=task,
        receipts=receipts,
        qualification_allowed=bool(qualification_allowed),
        required_controls=REQUIRED_CONTROLS,
        canonical_digest=canonical_digest,
    )
    _write_json(output / "adp_task_control_pair.v1.json", pair)
    return pair


def run_task_neutral_control(
    *,
    environment: ControlEnvironment,
    task_spec: Mapping[str, Any],
    control_plan: Mapping[str, Any],
    control_id: str,
    gripper_open_command: float,
    gripper_closed_command: float | None = None,
    output_dir: str | Path,
    qualification_allowed: bool = True,
) -> dict[str, Any]:
    """Run exactly one preregistered task-neutral control episode.

    A production Task Evaluation Run may need the negative and positive
    controls to terminalize under separate provider authorities.  This entry
    point preserves the same episode implementation and receipt schema as the
    paired helper while refusing any unrecognized control selector.
    """

    if control_id not in REQUIRED_CONTROLS:
        raise ControlEpisodeError(
            [f"task_control_selection_invalid:{control_id or 'missing'}"]
        )
    task = json.loads(json.dumps(dict(task_spec), allow_nan=False))
    plan = validate_task_control_plan(control_plan, task_spec=task)
    output = Path(output_dir).expanduser().resolve()
    _write_json(output / "adp_task_control_plan.v1.json", plan)
    receipt = _run_task_control_episode(
        environment=environment,
        task_spec=task,
        plan=plan,
        control_id=control_id,
        gripper_open_command=float(gripper_open_command),
        gripper_closed_command=(
            None
            if gripper_closed_command is None
            else float(gripper_closed_command)
        ),
        output=output,
        episode_id=f"{plan['cell_id']}-{control_id}",
        qualification_allowed=bool(qualification_allowed),
        receipt_annotations=diagnostic_receipt_annotations(
            plan, qualification_allowed=qualification_allowed
        ),
    )
    _write_json(output / f"adp_task_control_episode.{control_id}.json", receipt)
    return receipt


def run_synthetic_post_phase5_downstream_diagnostic(
    *,
    environment: ControlEnvironment,
    task_spec: Mapping[str, Any],
    control_plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    gripper_open_command: float,
    gripper_closed_command: float,
    output_dir: str | Path,
    checkpoint_settle_steps: int = 8,
) -> dict[str, Any]:
    """Execute phases 6--11 continuously from a synthetic Phase-5 boundary.

    The checkpoint deliberately does *not* assert that a grasp was achieved.
    It injects a digest-bound, gate-filtered arm and task state, then physically
    commands the closed gripper through the same bounded seam before replaying
    the unchanged downstream suffix.  It therefore breaks the debugging
    dependency on Phase 5 without qualifying Phase 5, a policy cell, any
    downstream phase, or task success.
    """

    task = json.loads(json.dumps(dict(task_spec), allow_nan=False))
    plan = validate_task_control_plan(control_plan, task_spec=task)
    try:
        frozen_checkpoint = json.loads(
            json.dumps(dict(checkpoint), allow_nan=False)
        )
        arm = [
            float(value)
            for value in frozen_checkpoint["arm_joint_positions_rad"]
        ]
        task_joints = {
            str(name): float(value)
            for name, value in frozen_checkpoint[
                "task_joint_positions_rad"
            ].items()
        }
        open_command = float(gripper_open_command)
        closed_command = float(gripper_closed_command)
    except (KeyError, TypeError, ValueError) as exc:
        raise ControlEpisodeError(
            ["downstream_diagnostic_checkpoint_invalid"]
        ) from exc
    errors: list[str] = []
    if (
        frozen_checkpoint.get("schema_version")
        != "adp_task_synthetic_post_phase5_checkpoint.v1"
        or frozen_checkpoint.get("source_phase_id") != "contact_close"
        or frozen_checkpoint.get("gripper_state") != "closed"
        or frozen_checkpoint.get("phase5_qualified") is not False
        or frozen_checkpoint.get("initialization_authority")
        != "runtime_derived_from_gate_qualified_offsim_contact_close"
        or frozen_checkpoint.get("checkpoint_digest")
        != canonical_digest(
            frozen_checkpoint, digest_field="checkpoint_digest"
        )
    ):
        errors.append("downstream_diagnostic_checkpoint_contract_invalid")
    if len(arm) != 7 or not all(math.isfinite(value) for value in arm):
        errors.append("downstream_diagnostic_checkpoint_arm_invalid")
    if not task_joints or not all(
        name and math.isfinite(value) for name, value in task_joints.items()
    ):
        errors.append("downstream_diagnostic_checkpoint_task_joints_invalid")
    if (
        not all(math.isfinite(value) for value in (open_command, closed_command))
        or open_command == closed_command
    ):
        errors.append("downstream_diagnostic_gripper_commands_invalid")
    if (
        isinstance(checkpoint_settle_steps, bool)
        or not isinstance(checkpoint_settle_steps, int)
        or checkpoint_settle_steps < 1
    ):
        errors.append("downstream_diagnostic_settle_steps_invalid")
    resetter = getattr(environment, "reset_to_diagnostic_checkpoint", None)
    bounded = getattr(environment, "bounded_joint_action", None)
    limits_reader = getattr(environment, "joint_limits", None)
    if not callable(resetter):
        errors.append("downstream_diagnostic_checkpoint_reset_unavailable")
    if not callable(bounded):
        errors.append("downstream_diagnostic_bounded_action_unavailable")
    arm_limits: list[list[float]] = []
    if not callable(limits_reader):
        errors.append("downstream_diagnostic_arm_joint_limits_unavailable")
    else:
        try:
            arm_limits = [
                [float(bound) for bound in row]
                for row in limits_reader()
            ]
        except (TypeError, ValueError):
            arm_limits = []
        if (
            len(arm_limits) != 7
            or any(
                len(row) != 2
                or not all(math.isfinite(bound) for bound in row)
                or row[0] > row[1]
                for row in arm_limits
            )
        ):
            errors.append("downstream_diagnostic_arm_joint_limits_invalid")
        elif any(
            value < lower or value > upper
            for value, (lower, upper) in zip(arm, arm_limits, strict=True)
        ):
            errors.append("downstream_diagnostic_arm_checkpoint_out_of_bounds")
    task_limits = task.get("joint_hard_limits_rad")
    if not isinstance(task_limits, Mapping):
        errors.append("downstream_diagnostic_task_joint_limits_unavailable")
    else:
        for name, value in task_joints.items():
            try:
                lower, upper = [float(bound) for bound in task_limits[name]]
            except (KeyError, TypeError, ValueError):
                errors.append(
                    f"downstream_diagnostic_task_joint_limit_invalid:{name}"
                )
                continue
            if (
                not all(math.isfinite(bound) for bound in (lower, upper))
                or lower > upper
                or value < lower
                or value > upper
            ):
                errors.append(
                    f"downstream_diagnostic_task_checkpoint_out_of_bounds:{name}"
                )

    actions = plan.get("scripted_positive_actions")
    rows_by_id = {
        str(row.get("phase_id") or ""): row
        for row in actions or []
        if isinstance(row, Mapping)
    }
    downstream_rows = [
        dict(rows_by_id.get(phase_id) or {})
        for phase_id in DOWNSTREAM_DIAGNOSTIC_PHASE_IDS
    ]
    if any(not row for row in downstream_rows):
        errors.append("downstream_diagnostic_phase_suffix_missing")
    else:
        ordered_phase_ids = [
            str(row.get("phase_id") or "")
            for row in actions or []
            if isinstance(row, Mapping)
            and str(row.get("phase_id") or "")
            in DOWNSTREAM_DIAGNOSTIC_PHASE_IDS
        ]
        if ordered_phase_ids != list(DOWNSTREAM_DIAGNOSTIC_PHASE_IDS):
            errors.append("downstream_diagnostic_phase_suffix_order_invalid")
    if errors:
        raise ControlEpisodeError(errors)

    first = downstream_rows[0]
    checkpoint_rows = [
        {
            "phase_id": "synthetic_post_phase5_checkpoint_settle",
            "arm_joint_positions": list(arm),
            "gripper_state": "closed",
            "max_joint_delta_rad": float(first["max_joint_delta_rad"]),
            "max_joint_setpoint_lead_rad": float(
                first["max_joint_setpoint_lead_rad"]
            ),
        }
        for _ in range(checkpoint_settle_steps)
    ]
    release_settle_rows = [
        {
            "phase_id": "release_settle",
            "mode": "hold_current_joint_positions",
        }
        for _ in range(int(task["settle_window_samples"]))
    ]
    trajectory = [*checkpoint_rows, *downstream_rows, *release_settle_rows]
    output = Path(output_dir).expanduser().resolve()
    _write_json(output / "adp_task_control_plan.v1.json", plan)

    def _initial_sample_blocker(sample: Mapping[str, Any]) -> str | None:
        if any(
            sample.get(name) is True
            for name in (
                "joint_limit_violation",
                "containment_violation",
                "robot_collision_failure",
                "scene_collision_failure",
            )
        ):
            return "synthetic_checkpoint_initial_state_unsafe"
        try:
            reached_arm = [
                float(value) for value in environment.read_arm_joint_positions()
            ]
        except (TypeError, ValueError):
            return "synthetic_checkpoint_arm_readback_invalid"
        if len(reached_arm) != 7 or any(
            abs(expected - reached) > 1.0e-5
            for expected, reached in zip(arm, reached_arm, strict=True)
        ):
            return "synthetic_checkpoint_arm_readback_mismatch"
        observed_task = sample.get("joint_positions_rad")
        if not isinstance(observed_task, Mapping):
            return "synthetic_checkpoint_task_readback_missing"
        reset_tolerance = float(task.get("reset_tolerance_rad") or 1.0e-4)
        try:
            mismatch = any(
                abs(float(observed_task[name]) - expected) > reset_tolerance
                for name, expected in task_joints.items()
            )
        except (KeyError, TypeError, ValueError):
            return "synthetic_checkpoint_task_readback_invalid"
        return "synthetic_checkpoint_task_readback_mismatch" if mismatch else None

    receipt = _run_task_control_episode(
        environment=environment,
        task_spec=task,
        plan=plan,
        control_id=DOWNSTREAM_DIAGNOSTIC_CONTROL_ID,
        gripper_open_command=open_command,
        gripper_closed_command=closed_command,
        output=output,
        episode_id=(
            f"{plan['cell_id']}-{DOWNSTREAM_DIAGNOSTIC_CONTROL_ID}"
        ),
        initial_reset_callback=lambda: resetter(
            arm_joint_positions_rad=arm,
            task_joint_positions_rad=task_joints,
        ),
        trajectory_override=trajectory,
        qualification_allowed=False,
        receipt_annotations={
            "checkpoint": frozen_checkpoint,
            "checkpoint_settle_steps": checkpoint_settle_steps,
            "requested_phase_ids": list(DOWNSTREAM_DIAGNOSTIC_PHASE_IDS),
        },
        initial_sample_validator=_initial_sample_blocker,
    )
    executed_phase_ids: list[str] = []
    for row in receipt["action_trace"]:
        phase_id = str(row.get("phase_id") or "")
        if (
            phase_id in DOWNSTREAM_DIAGNOSTIC_PHASE_IDS
            and phase_id not in executed_phase_ids
        ):
            executed_phase_ids.append(phase_id)
    arrivals = {
        str(row.get("phase_id") or ""): row
        for row in receipt["phase_arrivals"]
        if isinstance(row, Mapping)
    }
    pose_phase_ids = [
        str(row["phase_id"])
        for row in downstream_rows
        if row.get("mode") == "ik_pose"
    ]
    unsafe_state_observed = any(
        sample.get(name) is True
        for sample in receipt["state_trace"]
        for name in (
            "joint_limit_violation",
            "containment_violation",
            "robot_collision_failure",
            "scene_collision_failure",
        )
    )
    receipt.update(
        {
            "schema_version": TASK_DOWNSTREAM_DIAGNOSTIC_SCHEMA_VERSION,
            "status": "measured",
            "phase5_qualified": False,
            "synthetic_checkpoint": frozen_checkpoint,
            "checkpoint_settle_steps": checkpoint_settle_steps,
            "requested_phase_ids": list(DOWNSTREAM_DIAGNOSTIC_PHASE_IDS),
            "executed_phase_ids": executed_phase_ids,
            "continuous_suffix_executed": executed_phase_ids
            == list(DOWNSTREAM_DIAGNOSTIC_PHASE_IDS),
            "all_pose_gates_reached": bool(pose_phase_ids)
            and all(
                arrivals.get(phase_id, {}).get("target_reached") is True
                for phase_id in pose_phase_ids
            ),
            "unsafe_state_observed": unsafe_state_observed,
            "synthetic_checkpoint_task_outcome": receipt["score"].get(
                "outcome"
            ),
            "synthetic_checkpoint_task_succeeded": receipt["score"].get(
                "task_succeeded"
            ),
            "retained_evidence": [
                "per_step_commanded_and_reached_arm_joints",
                "per_step_arm_dynamics",
                "per_phase_fk_and_measured_tcp",
                "per_step_task_door_contact_state",
                "lossless_multicamera_observations_and_review_media",
            ],
            "qualification_effect": "none",
            "receipt_digest": "",
        }
    )
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    _write_json(
        output / "adp_task_synthetic_post_phase5_downstream_diagnostic.v1.json",
        receipt,
    )
    return receipt


__all__ = [
    "BLOCKER_POSITIVE_FAILED",
    "BLOCKER_PHASE_NOT_REACHED",
    "BLOCKER_ZERO_COMPLETED_TASK",
    "CONTROL_EPISODE_SCHEMA_VERSION",
    "CONTROL_PAIR_SCHEMA_VERSION",
    "CONTROL_PLAN_SCHEMA_VERSION",
    "MAX_JOINT_DELTA_PER_STEP_RAD",
    "ControlEpisodeError",
    "SCRIPTED_POSITIVE",
    "DOWNSTREAM_DIAGNOSTIC_PHASE_IDS",
    "TASK_CONTROL_EPISODE_SCHEMA_VERSION",
    "TASK_CONTROL_PAIR_SCHEMA_VERSION",
    "TASK_CONTROL_PLAN_SCHEMA_VERSION",
    "TASK_DOWNSTREAM_DIAGNOSTIC_SCHEMA_VERSION",
    "ZERO_ACTION_NEGATIVE",
    "materialize_control_plan",
    "run_control_episode",
    "run_required_controls",
    "run_task_neutral_control",
    "run_task_neutral_controls",
    "run_synthetic_post_phase5_downstream_diagnostic",
    "validate_task_control_plan",
]
