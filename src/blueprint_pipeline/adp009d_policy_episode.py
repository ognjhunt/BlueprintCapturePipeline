"""Run one frozen ADP-009D task episode against a learned DROID policy.

This is the orchestration the five ADP-009D adapters were built for, and it is
the only place they meet: observation formatting, policy query, action-chunk
execution, and deterministic scoring, in that order, for one episode.

The simulator is injected rather than imported.  Everything here is arithmetic
and sequencing, so the whole loop -- including its failure paths -- is testable
without a GPU, and the Isaac-side adapter stays a thin, reviewable shim that
only reads and writes simulator state.

Three properties are load-bearing and enforced rather than assumed:

* **The episode ends with a settle window the gripper is absent from.**  The
  place predicate is judged on a can at rest after release; without a settle
  phase ``placed`` could never be decided, and an episode would silently score
  one rung lower than it earned.
* **Step indices strictly increase across the whole episode**, including across
  policy queries, because the scorer treats a repeated index as malformed
  evidence rather than reordering it.
* **The policy is queried only through the injected client**, and every query
  and chunk is retained, so a receipt can be re-derived without a simulator.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

try:  # flat provider-bundle layout
    from adp009d_droid_action_execution import (
        ACTION_SPACE_JOINT_VELOCITY,
        ARM_JOINT_COUNT,
        DROID_CONTROL_HZ,
        DROID_OPEN_LOOP_HORIZON,
        DroidActionExecutionError,
        GripperConvention,
        droid_row_to_isaac_action,
        plan_chunk_execution,
    )
except ModuleNotFoundError:  # repository package
    from .adp009d_droid_action_execution import (
        ACTION_SPACE_JOINT_VELOCITY,
        ARM_JOINT_COUNT,
        DROID_CONTROL_HZ,
        DROID_OPEN_LOOP_HORIZON,
        DroidActionExecutionError,
        GripperConvention,
        droid_row_to_isaac_action,
        plan_chunk_execution,
    )
try:  # flat provider-bundle layout
    from adp009d_droid_observation import (
        CANDIDATE_REQUIRED_VIEWS,
        DROID_OBSERVATION_SCHEMA_VERSION,
        GROOT_HISTORICAL_VIEW_KEYS,
        GROOT_HISTORY_STEPS,
        DroidObservationError,
        build_droid_observation,
        describe_observation_conversion,
    )
except ModuleNotFoundError:  # repository package
    from .adp009d_droid_observation import (
        CANDIDATE_REQUIRED_VIEWS,
        DROID_OBSERVATION_SCHEMA_VERSION,
        GROOT_HISTORICAL_VIEW_KEYS,
        GROOT_HISTORY_STEPS,
        DroidObservationError,
        build_droid_observation,
        describe_observation_conversion,
    )
try:  # flat provider-bundle layout
    from adp009d_task_scoring import (
        SUPPORT_PLANE_Z_M,
        SETTLE_WINDOW_SAMPLES,
        TaskScoringError,
    )
except ModuleNotFoundError:  # repository package
    from .adp009d_task_scoring import (
        SUPPORT_PLANE_Z_M,
        SETTLE_WINDOW_SAMPLES,
        TaskScoringError,
    )
try:  # flat provider-bundle layout
    from adp_task_scoring import (
        TASK_KIND_ARTICULATED_OPEN_CLOSE,
        TASK_KIND_RIGID_PICK_PLACE,
        TASK_SPEC_SCHEMA_VERSION,
        TaskNeutralScoringError,
        score_task_episode_from_spec,
        validate_articulated_task_spec,
    )
except ModuleNotFoundError:  # repository package
    from .adp_task_scoring import (
        TASK_KIND_ARTICULATED_OPEN_CLOSE,
        TASK_KIND_RIGID_PICK_PLACE,
        TASK_SPEC_SCHEMA_VERSION,
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
        finalize_visual_evidence,
        persist_multicamera_observation,
        persist_observation_frame,
    )
except ModuleNotFoundError:  # repository package
    from .episode_visual_evidence import (
        finalize_manipulation_evaluation_visual_evidence,
        finalize_visual_evidence,
        persist_multicamera_observation,
        persist_observation_frame,
    )

EPISODE_SCHEMA_VERSION = "adp009d_policy_episode.v3"

# This is a numerical-motion threshold, not a task-success threshold.  It only
# separates a changing simulator joint state from float noise so a can outcome
# is never attributed to a policy whose commands were not observed at the arm.
ARM_MOTION_EPSILON_RAD = 1e-6

# A policy that has not moved the can within this many queries has failed the
# episode; the cap bounds paid GPU time and is recorded rather than implicit.
DEFAULT_MAX_POLICY_QUERIES = 60
EVALUATION_REVIEW_FRAME_STRIDE_STEPS = 8

BLOCKER_NO_SETTLE_WINDOW = "policy_episode_settle_window_not_reached"
BLOCKER_GRIPPER_PRESENT_IN_SETTLE = "policy_episode_gripper_present_during_settle"
BLOCKER_STEP_INDEX_NOT_INCREASING = "policy_episode_step_index_not_increasing"
BLOCKER_CLIENT_RETURNED_NOTHING = "policy_episode_client_returned_no_chunk"
BLOCKER_QUERY_BUDGET_EXHAUSTED = "policy_episode_query_budget_exhausted"
BLOCKER_ENVIRONMENT_CONTRACT = "policy_episode_environment_contract_violated"


class PolicyEpisodeError(ValueError):
    """Fail-closed episode contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


class EpisodeEnvironment(Protocol):
    """The exact simulator surface this loop needs, and nothing more."""

    def reset(self) -> None:
        """Return to the sealed canonical start pose."""

    def read_policy_inputs(self) -> Mapping[str, Any]:
        """Camera RGB by DROID view name, plus ``joint_position`` and ``gripper_position``."""

    def read_arm_joint_positions(self) -> Sequence[float]:
        """Seven observed arm joint positions, without rendering policy cameras."""

    def step(self, isaac_action: Sequence[float]) -> None:
        """Apply one 8-dimensional Arena action for one environment step."""

    def read_object_sample(self) -> Mapping[str, Any]:
        """Deterministic object state: ``can_pose_world`` and optional grasp evidence."""

    def read_task_sample(self) -> Mapping[str, Any]:
        """Task-neutral deterministic state for a non-rigid task spec."""

    def joint_limits(self) -> Sequence[Sequence[float]]:
        """Seven ``(lower, upper)`` arm joint limits, in radians."""


class DroidPolicyClient(Protocol):
    """The policy seam.  Implementations talk to a server; this never does."""

    def infer(self, observation: Mapping[str, Any]) -> Any:
        """Return an action chunk of shape ``(rows, 8)`` with ``rows >= 8``."""


def _sample_with_index(
    raw: Mapping[str, Any],
    step_index: int,
    previous_index: int | None,
    *,
    required_field: str | None = "can_pose_world",
) -> dict[str, Any]:
    if previous_index is not None and step_index <= previous_index:
        raise PolicyEpisodeError(
            [f"{BLOCKER_STEP_INDEX_NOT_INCREASING}:{step_index}<={previous_index}"]
        )
    sample = dict(raw)
    sample["step_index"] = step_index
    if required_field is not None and required_field not in sample:
        raise PolicyEpisodeError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:{required_field}_missing"]
        )
    return sample


def _resolved_task_spec(
    *,
    task_spec: Mapping[str, Any] | None,
    destination_position_world_m: Sequence[float] | None,
    settle_window_samples: int,
    max_policy_queries: int,
    open_loop_horizon: int,
) -> dict[str, Any]:
    """Resolve the legacy rigid call or validate an explicit task-neutral spec."""

    if task_spec is None:
        if destination_position_world_m is None:
            raise PolicyEpisodeError(["policy_episode_destination_missing"])
        return {
            "schema_version": TASK_SPEC_SCHEMA_VERSION,
            "task_kind": TASK_KIND_RIGID_PICK_PLACE,
            "destination_position_world_m": [
                float(value) for value in destination_position_world_m
            ],
            "support_plane_z_m": SUPPORT_PLANE_Z_M,
            "settle_window_samples": int(settle_window_samples),
            "require_sealed_start_pose": True,
        }
    try:
        resolved = json.loads(json.dumps(dict(task_spec), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise PolicyEpisodeError(["policy_episode_task_spec_invalid"]) from exc
    kind = resolved.get("task_kind")
    if kind == TASK_KIND_ARTICULATED_OPEN_CLOSE:
        try:
            validate_articulated_task_spec(resolved)
        except TaskNeutralScoringError as exc:
            raise PolicyEpisodeError(exc.errors) from exc
    elif kind != TASK_KIND_RIGID_PICK_PLACE:
        raise PolicyEpisodeError(["policy_episode_task_kind_unsupported"])
    if int(resolved.get("settle_window_samples", -1)) != int(
        settle_window_samples
    ):
        raise PolicyEpisodeError(["policy_episode_settle_window_task_spec_mismatch"])
    expected_hz = resolved.get("control_frequency_hz")
    if expected_hz is not None and float(expected_hz) != float(DROID_CONTROL_HZ):
        raise PolicyEpisodeError(["policy_episode_control_frequency_task_spec_mismatch"])
    maximum_steps = resolved.get("maximum_action_steps")
    if maximum_steps is not None and int(max_policy_queries) * int(
        open_loop_horizon
    ) > int(maximum_steps):
        raise PolicyEpisodeError(["policy_episode_action_budget_exceeds_task_spec"])
    return resolved


def _read_task_sample(
    environment: EpisodeEnvironment, *, task_kind: str
) -> Mapping[str, Any]:
    if task_kind == TASK_KIND_RIGID_PICK_PLACE:
        return environment.read_object_sample()
    reader = getattr(environment, "read_task_sample", None)
    if not callable(reader):
        raise PolicyEpisodeError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:read_task_sample_missing"]
        )
    sample = reader()
    if not isinstance(sample, Mapping):
        raise PolicyEpisodeError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:task_sample_invalid"]
        )
    return sample


def _policy_view_composite(
    observation: Mapping[str, Any], *, candidate_id: str
) -> Any:
    """One lossless RGB canvas containing every exact image shown to a policy."""

    import numpy as np

    view_order = list(CANDIDATE_REQUIRED_VIEWS[candidate_id])
    if candidate_id == "groot_n17_droid":
        view_order = [
            GROOT_HISTORICAL_VIEW_KEYS[view]
            for view in CANDIDATE_REQUIRED_VIEWS[candidate_id]
        ] + view_order
    views = [np.asarray(observation[name]) for name in view_order]
    if not views or any(
        view.dtype != np.uint8 or view.ndim != 3 or view.shape[2] != 3
        for view in views
    ):
        raise PolicyEpisodeError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:policy_media_view_invalid"]
        )
    if len({view.shape[0] for view in views}) != 1:
        raise PolicyEpisodeError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:policy_media_view_height_mismatch"]
        )
    return np.ascontiguousarray(np.concatenate(views, axis=1))


def _retain_policy_input_sample(
    history: deque[tuple[int, Mapping[str, Any]]],
    *,
    step_index: int,
    inputs: Mapping[str, Any],
) -> None:
    if history and history[-1][0] == int(step_index):
        history.pop()
    history.append((int(step_index), dict(inputs)))


def _persist_evaluation_camera_observation(
    environment: EpisodeEnvironment,
    *,
    output_dir: Path,
    episode_id: str,
    observation_index: int,
    kind: str,
) -> dict[str, Any]:
    image_reader = getattr(environment, "read_evaluation_camera_inputs", None)
    metadata_reader = getattr(environment, "read_control_observation_metadata", None)
    if not callable(image_reader) or not callable(metadata_reader):
        raise PolicyEpisodeError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:overview_camera_contract_missing"]
        )
    images = dict(image_reader())
    missing = {"external", "wrist", "overview"} - set(images)
    if missing:
        raise PolicyEpisodeError(
            [
                f"{BLOCKER_ENVIRONMENT_CONTRACT}:evaluation_camera_missing:{camera_id}"
                for camera_id in missing
            ]
        )
    metadata = dict(metadata_reader())
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


def _historical_camera_rgb(
    history: deque[tuple[int, Mapping[str, Any]]],
    *,
    candidate_id: str,
    step_index: int,
) -> dict[str, Any] | None:
    if candidate_id != "groot_n17_droid":
        return None
    target_index = max(0, int(step_index) - GROOT_HISTORY_STEPS)
    sample = next(
        (inputs for index, inputs in history if index == target_index),
        None,
    )
    if sample is None:
        raise PolicyEpisodeError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:groot_history_step_missing:{target_index}"]
        )
    return {
        view: sample[view]
        for view in CANDIDATE_REQUIRED_VIEWS[candidate_id]
        if view in sample
    }


def _read_arm_joint_positions(environment: EpisodeEnvironment) -> list[float]:
    reader = getattr(environment, "read_arm_joint_positions", None)
    if not callable(reader):
        raise PolicyEpisodeError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:read_arm_joint_positions_missing"]
        )
    raw = reader()
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise PolicyEpisodeError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:arm_joint_positions_invalid"]
        ) from exc
    if len(values) != ARM_JOINT_COUNT or not all(math.isfinite(value) for value in values):
        raise PolicyEpisodeError(
            [
                f"{BLOCKER_ENVIRONMENT_CONTRACT}:"
                f"arm_joint_positions_invalid:{len(values)}"
            ]
        )
    return values


def _motion_and_command_evidence(
    *,
    joint_trace: Sequence[Sequence[float]],
    commanded_actions: Sequence[Mapping[str, Any]],
    command_response_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reset = [float(value) for value in joint_trace[0]]
    end = [float(value) for value in joint_trace[-1]]
    max_delta = [
        max(abs(float(sample[index]) - reset[index]) for sample in joint_trace)
        for index in range(ARM_JOINT_COUNT)
    ]
    end_delta = [end[index] - reset[index] for index in range(ARM_JOINT_COUNT)]

    arm_targets = [
        float(value)
        for action in commanded_actions
        for value in action["joint_position_target_rad"]
    ]
    velocity_commands = [
        float(value)
        for action in commanded_actions
        for value in action["joint_velocity_command_rad_s"]
    ]
    clipped_velocity_commands = [
        float(value)
        for action in commanded_actions
        for value in action["clipped_droid_action"][:ARM_JOINT_COUNT]
        if action["joint_velocity_command_rad_s"]
    ]
    source_arm_commands = [
        float(value)
        for action in commanded_actions
        for value in action["source_arm_command"]
    ]
    source_action_spaces = {
        str(action["source_action_space"]) for action in commanded_actions
    }
    if len(source_action_spaces) != 1:
        raise PolicyEpisodeError(["policy_episode_source_action_space_inconsistent"])
    source_action_space = next(iter(source_action_spaces))
    target_deltas = [
        abs(float(target) - float(observed))
        for action in commanded_actions
        for target, observed in zip(
            action["joint_position_target_rad"],
            action["observed_before_rad"],
            strict=True,
        )
    ]
    full_action_l2 = [
        math.sqrt(sum(float(value) ** 2 for value in action["isaac_action"]))
        for action in commanded_actions
    ]
    gripper_commands = [
        abs(float(action["isaac_action"][ARM_JOINT_COUNT]))
        for action in commanded_actions
    ]
    nontrivial_rows = sum(
        any(
            abs(float(target) - float(observed)) > ARM_MOTION_EPSILON_RAD
            for target, observed in zip(
                action["joint_position_target_rad"],
                action["observed_before_rad"],
                strict=True,
            )
        )
        for action in commanded_actions
    )

    arm_moved = max(max_delta, default=0.0) > ARM_MOTION_EPSILON_RAD
    actions_reached_robot = command_response_rows > 0
    if actions_reached_robot:
        interpretation = "policy_task_outcome_interpretable"
    elif arm_moved:
        interpretation = "arm_motion_without_command_response_harness_fault"
    elif nontrivial_rows:
        interpretation = "nontrivial_actions_not_observed_at_robot_harness_fault"
    else:
        interpretation = "no_arm_motion_and_no_nontrivial_command_harness_fault"

    action_summary = {
        "policy_action_rows_submitted": len(commanded_actions),
        "source_action_space": source_action_space,
        "source_arm_command_max_abs": max(
            (abs(value) for value in source_arm_commands), default=0.0
        ),
        "source_arm_command_mean_abs": (
            sum(abs(value) for value in source_arm_commands) / len(source_arm_commands)
            if source_arm_commands
            else 0.0
        ),
        "joint_velocity_command_max_abs_rad_s": max(
            (abs(value) for value in velocity_commands), default=0.0
        ),
        "joint_velocity_command_mean_abs_rad_s": (
            sum(abs(value) for value in velocity_commands) / len(velocity_commands)
            if velocity_commands
            else 0.0
        ),
        "joint_velocity_command_clipped_value_count": sum(
            abs(raw - clipped) > 1e-12
            for raw, clipped in zip(
                velocity_commands, clipped_velocity_commands, strict=True
            )
        ),
        "arm_target_max_abs_rad": max((abs(value) for value in arm_targets), default=0.0),
        "arm_target_mean_abs_rad": (
            sum(abs(value) for value in arm_targets) / len(arm_targets)
            if arm_targets
            else 0.0
        ),
        "arm_target_delta_from_observed_max_abs_rad": max(target_deltas, default=0.0),
        "arm_target_delta_from_observed_mean_abs_rad": (
            sum(target_deltas) / len(target_deltas) if target_deltas else 0.0
        ),
        "full_action_l2_max": max(full_action_l2, default=0.0),
        "gripper_command_max_abs": max(gripper_commands, default=0.0),
        "nontrivial_arm_target_rows": nontrivial_rows,
    }
    motion_evidence = {
        "joint_position_reset_rad": reset,
        "joint_position_end_rad": end,
        "joint_position_end_delta_rad": end_delta,
        "max_abs_joint_delta_from_reset_rad": max_delta,
        "joint_position_samples": len(joint_trace),
        "arm_motion_epsilon_rad": ARM_MOTION_EPSILON_RAD,
        "command_response_rows": int(command_response_rows),
        "arm_moved": arm_moved,
        "actions_reached_robot": actions_reached_robot,
        "policy_outcome_interpretable": actions_reached_robot,
        "interpretation": interpretation,
    }
    return motion_evidence, action_summary


def run_policy_episode(
    *,
    environment: EpisodeEnvironment,
    policy: DroidPolicyClient,
    candidate_id: str,
    destination_position_world_m: Sequence[float] | None = None,
    prompt: str,
    gripper: GripperConvention,
    task_spec: Mapping[str, Any] | None = None,
    max_policy_queries: int = DEFAULT_MAX_POLICY_QUERIES,
    settle_window_samples: int = SETTLE_WINDOW_SAMPLES,
    open_loop_horizon: int = DROID_OPEN_LOOP_HORIZON,
    media_output_dir: str | Path | None = None,
    episode_id: str | None = None,
) -> dict[str, Any]:
    """Run one episode end to end and return a digest-bound receipt.

    The loop resets, then repeatedly formats an observation for this candidate,
    asks the policy for a chunk, executes exactly the open-loop horizon of it,
    and samples deterministic object state after every environment step.  When
    the query budget is spent it holds the arm still for a settle window so the
    placed predicate can be decided on a can at rest.

    Raises :class:`PolicyEpisodeError` when the environment, the client, or the
    episode's own shape violates its contract.  Scoring errors surface as
    :class:`~blueprint_pipeline.adp009d_task_scoring.TaskScoringError`.
    """

    if candidate_id not in CANDIDATE_REQUIRED_VIEWS:
        raise PolicyEpisodeError([f"policy_episode_unknown_candidate:{candidate_id}"])
    if int(max_policy_queries) < 1:
        raise PolicyEpisodeError(["policy_episode_query_budget_invalid"])
    if int(settle_window_samples) < 1:
        raise PolicyEpisodeError(["policy_episode_settle_window_invalid"])
    if (media_output_dir is None) != (episode_id is None):
        raise PolicyEpisodeError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:policy_media_binding_incomplete"]
        )
    policy_action_space = str(
        getattr(policy, "action_space", ACTION_SPACE_JOINT_VELOCITY)
    )
    resolved_task_spec = _resolved_task_spec(
        task_spec=task_spec,
        destination_position_world_m=destination_position_world_m,
        settle_window_samples=int(settle_window_samples),
        max_policy_queries=int(max_policy_queries),
        open_loop_horizon=int(open_loop_horizon),
    )
    task_kind = str(resolved_task_spec["task_kind"])

    media_root = (
        Path(media_output_dir).expanduser().resolve()
        if media_output_dir is not None
        else None
    )
    retained_policy_frames: list[dict[str, Any]] = []
    retained_multicamera_observations: list[dict[str, Any]] = []
    retained_review_observations: list[dict[str, Any]] = []
    media_observation_index = 0
    multicamera_evaluation_available = callable(
        getattr(environment, "read_evaluation_camera_inputs", None)
    ) and callable(getattr(environment, "read_control_observation_metadata", None))
    policy_input_history: deque[tuple[int, Mapping[str, Any]]] = deque(
        maxlen=GROOT_HISTORY_STEPS + 1
    )

    episode_started = time.monotonic()
    timings_seconds = {
        "reset_and_initial_state": 0.0,
        "policy_input_read": 0.0,
        "observation_preprocessing": 0.0,
        "policy_inference": 0.0,
        "action_planning": 0.0,
        "environment_step_including_render": 0.0,
        "joint_state_read": 0.0,
        "object_state_sample": 0.0,
        "settle_steps_including_render": 0.0,
        "deterministic_scoring": 0.0,
        "media_persistence": 0.0,
    }
    phase_started = time.monotonic()
    environment.reset()
    joint_limits = environment.joint_limits()
    joint_trace = [_read_arm_joint_positions(environment)]

    samples: list[dict[str, Any]] = []
    previous_index: int | None = None
    step_index = 0
    samples.append(
        _sample_with_index(
            _read_task_sample(environment, task_kind=task_kind),
            step_index,
            previous_index,
            required_field=(
                "can_pose_world"
                if task_kind == TASK_KIND_RIGID_PICK_PLACE
                else "joint_positions_rad"
            ),
        )
    )
    previous_index = step_index
    timings_seconds["reset_and_initial_state"] += time.monotonic() - phase_started

    queries: list[dict[str, Any]] = []
    last_action: list[float] | None = None
    commanded_actions: list[dict[str, Any]] = []
    command_response_rows = 0

    for query_index in range(int(max_policy_queries)):
        phase_started = time.monotonic()
        inputs = environment.read_policy_inputs()
        _retain_policy_input_sample(
            policy_input_history, step_index=step_index, inputs=inputs
        )
        timings_seconds["policy_input_read"] += time.monotonic() - phase_started
        camera_rgb = {
            view: inputs[view] for view in CANDIDATE_REQUIRED_VIEWS[candidate_id] if view in inputs
        }
        phase_started = time.monotonic()
        try:
            observation = build_droid_observation(
                candidate_id=candidate_id,
                camera_rgb=camera_rgb,
                joint_position=inputs["joint_position"],
                gripper_position=inputs["gripper_position"],
                prompt=prompt,
                eef_9d=inputs.get("eef_9d"),
                historical_camera_rgb=_historical_camera_rgb(
                    policy_input_history,
                    candidate_id=candidate_id,
                    step_index=step_index,
                ),
            )
        except KeyError as exc:
            raise PolicyEpisodeError(
                [f"{BLOCKER_ENVIRONMENT_CONTRACT}:{exc.args[0]}_missing"]
            ) from exc
        except DroidObservationError:
            raise
        timings_seconds["observation_preprocessing"] += (
            time.monotonic() - phase_started
        )

        if media_root is not None and episode_id is not None:
            phase_started = time.monotonic()
            if multicamera_evaluation_available:
                retained_multicamera_observations.append(
                    _persist_evaluation_camera_observation(
                        environment,
                        output_dir=media_root,
                        episode_id=episode_id,
                        observation_index=media_observation_index,
                        kind="policy-input",
                    )
                )
                media_observation_index += 1
            else:
                retained_policy_frames.append(
                    persist_observation_frame(
                        _policy_view_composite(
                            observation, candidate_id=candidate_id
                        ),
                        output_dir=media_root,
                        episode_id=episode_id,
                        frame_index=query_index,
                        kind="policy-input",
                    )
                )
            timings_seconds["media_persistence"] += time.monotonic() - phase_started

        phase_started = time.monotonic()
        chunk = policy.infer(observation)
        timings_seconds["policy_inference"] += time.monotonic() - phase_started
        if chunk is None:
            raise PolicyEpisodeError([BLOCKER_CLIENT_RETURNED_NOTHING])
        inference_evidence_reader = getattr(policy, "last_inference_evidence", None)
        policy_inference_evidence = (
            inference_evidence_reader()
            if callable(inference_evidence_reader)
            else None
        )

        phase_started = time.monotonic()
        plan = plan_chunk_execution(
            chunk,
            horizon=int(open_loop_horizon),
            action_space=policy_action_space,
        )
        timings_seconds["action_planning"] += time.monotonic() - phase_started
        query_clamped_rows = 0
        for planned_action in plan["actions"]:
            before = list(joint_trace[-1])
            action = droid_row_to_isaac_action(
                planned_action["droid_action"],
                current_joint_position=before,
                joint_limits=joint_limits,
                gripper=gripper,
                action_space=policy_action_space,
            )
            query_clamped_rows += int(action["joint_limit_clamped"])
            phase_started = time.monotonic()
            environment.step(action["isaac_action"])
            timings_seconds["environment_step_including_render"] += (
                time.monotonic() - phase_started
            )
            phase_started = time.monotonic()
            after = _read_arm_joint_positions(environment)
            timings_seconds["joint_state_read"] += time.monotonic() - phase_started
            joint_trace.append(after)
            target = [float(value) for value in action["joint_position_target_rad"]]
            response_observed = any(
                abs(after[index] - before[index]) > ARM_MOTION_EPSILON_RAD
                and (target[index] - before[index]) * (after[index] - before[index]) > 0.0
                for index in range(ARM_JOINT_COUNT)
            )
            command_response_rows += int(response_observed)
            commanded_actions.append(
                {
                    "joint_position_target_rad": target,
                    "joint_velocity_command_rad_s": list(
                        action["joint_velocity_command_rad_s"]
                    ),
                    "source_arm_command": list(action["source_arm_command"]),
                    "source_action_space": action["source_action_space"],
                    "clipped_droid_action": list(action["clipped_droid_action"]),
                    "observed_before_rad": before,
                    "isaac_action": [float(value) for value in action["isaac_action"]],
                }
            )
            step_index += 1
            if (
                media_root is not None
                and episode_id is not None
                and multicamera_evaluation_available
                and step_index % EVALUATION_REVIEW_FRAME_STRIDE_STEPS == 0
            ):
                phase_started = time.monotonic()
                retained_review_observations.append(
                    _persist_evaluation_camera_observation(
                        environment,
                        output_dir=media_root,
                        episode_id=episode_id,
                        observation_index=media_observation_index,
                        kind="review-sample",
                    )
                )
                media_observation_index += 1
                timings_seconds["media_persistence"] += (
                    time.monotonic() - phase_started
                )
            if candidate_id == "groot_n17_droid":
                phase_started = time.monotonic()
                post_step_inputs = environment.read_policy_inputs()
                _retain_policy_input_sample(
                    policy_input_history,
                    step_index=step_index,
                    inputs=post_step_inputs,
                )
                timings_seconds["policy_input_read"] += (
                    time.monotonic() - phase_started
                )
            phase_started = time.monotonic()
            samples.append(
                _sample_with_index(
                    _read_task_sample(environment, task_kind=task_kind),
                    step_index,
                    previous_index,
                    required_field=(
                        "can_pose_world"
                        if task_kind == TASK_KIND_RIGID_PICK_PLACE
                        else "joint_positions_rad"
                    ),
                )
            )
            timings_seconds["object_state_sample"] += time.monotonic() - phase_started
            previous_index = step_index
            last_action = list(action["isaac_action"])

        queries.append(
            {
                "query_index": query_index,
                "chunk_shape": plan["chunk_shape"],
                "executed_rows": plan["executed_rows"],
                "discarded_rows": plan["discarded_rows"],
                "source_action_space": plan["source_action_space"],
                "position_adapter": plan["position_adapter"],
                "position_adapter_max_joint_delta_rad": plan[
                    "position_adapter_max_joint_delta_rad"
                ],
                "droid_source_revision": plan["droid_source_revision"],
                "openpi_source_revision": plan["openpi_source_revision"],
                "any_joint_limit_clamped": query_clamped_rows > 0,
                "joint_limit_clamped_rows": query_clamped_rows,
                "final_step_index": step_index,
                "policy_inference_evidence": policy_inference_evidence,
            }
        )

    if last_action is None:
        raise PolicyEpisodeError([BLOCKER_QUERY_BUDGET_EXHAUSTED])

    # Settle: hold the arm where the policy left it, but with the gripper open,
    # so the place predicate is judged on a released can at rest.  Holding the
    # commanded joints keeps this a settle rather than a retreat, which would
    # itself disturb the object being judged.
    release_action = list(last_action)
    release_action[7] = gripper.open_command
    settle_start_index = step_index
    for _ in range(int(settle_window_samples)):
        phase_started = time.monotonic()
        environment.step(release_action)
        joint_trace.append(_read_arm_joint_positions(environment))
        timings_seconds["settle_steps_including_render"] += (
            time.monotonic() - phase_started
        )
        step_index += 1
        if (
            media_root is not None
            and episode_id is not None
            and multicamera_evaluation_available
            and step_index % EVALUATION_REVIEW_FRAME_STRIDE_STEPS == 0
        ):
            phase_started = time.monotonic()
            retained_review_observations.append(
                _persist_evaluation_camera_observation(
                    environment,
                    output_dir=media_root,
                    episode_id=episode_id,
                    observation_index=media_observation_index,
                    kind="review-sample",
                )
            )
            media_observation_index += 1
            timings_seconds["media_persistence"] += (
                time.monotonic() - phase_started
            )
        if candidate_id == "groot_n17_droid":
            phase_started = time.monotonic()
            post_step_inputs = environment.read_policy_inputs()
            _retain_policy_input_sample(
                policy_input_history,
                step_index=step_index,
                inputs=post_step_inputs,
            )
            timings_seconds["policy_input_read"] += time.monotonic() - phase_started
        phase_started = time.monotonic()
        samples.append(
            _sample_with_index(
                _read_task_sample(environment, task_kind=task_kind),
                step_index,
                previous_index,
                required_field=(
                    "can_pose_world"
                    if task_kind == TASK_KIND_RIGID_PICK_PLACE
                    else "joint_positions_rad"
                ),
            )
        )
        timings_seconds["object_state_sample"] += time.monotonic() - phase_started
        previous_index = step_index

    if step_index - settle_start_index < int(settle_window_samples):
        raise PolicyEpisodeError([BLOCKER_NO_SETTLE_WINDOW])

    phase_started = time.monotonic()
    score = score_task_episode_from_spec(
        task_spec=resolved_task_spec,
        samples=samples,
    )
    timings_seconds["deterministic_scoring"] += time.monotonic() - phase_started
    motion_evidence, commanded_action_magnitudes = _motion_and_command_evidence(
        joint_trace=joint_trace,
        commanded_actions=commanded_actions,
        command_response_rows=command_response_rows,
    )

    visual_evidence = None
    media_artifacts: list[dict[str, Any]] = []
    if media_root is not None and episode_id is not None:
        phase_started = time.monotonic()
        if multicamera_evaluation_available:
            terminal_observation = _persist_evaluation_camera_observation(
                environment,
                output_dir=media_root,
                episode_id=episode_id,
                observation_index=media_observation_index,
                kind="terminal-observation",
            )
            visual_evidence, media_artifacts = (
                finalize_manipulation_evaluation_visual_evidence(
                    output_dir=media_root,
                    episode_id=episode_id,
                    identity={
                        "candidate_id": candidate_id,
                        "prompt": str(prompt),
                        "policy_input_camera_ids": ["external", "wrist"],
                        "review_only_camera_ids": ["overview"],
                        "overview_camera_used_by_policy": False,
                        "overview_camera_used_by_grader": False,
                    },
                    policy_input_observations=retained_multicamera_observations,
                    review_observations=retained_review_observations,
                    terminal_observation=terminal_observation,
                )
            )
        else:
            terminal_inputs = environment.read_policy_inputs()
            _retain_policy_input_sample(
                policy_input_history, step_index=step_index, inputs=terminal_inputs
            )
            terminal_camera_rgb = {
                view: terminal_inputs[view]
                for view in CANDIDATE_REQUIRED_VIEWS[candidate_id]
                if view in terminal_inputs
            }
            try:
                terminal_policy_observation = build_droid_observation(
                    candidate_id=candidate_id,
                    camera_rgb=terminal_camera_rgb,
                    joint_position=terminal_inputs["joint_position"],
                    gripper_position=terminal_inputs["gripper_position"],
                    prompt=prompt,
                    eef_9d=terminal_inputs.get("eef_9d"),
                    historical_camera_rgb=_historical_camera_rgb(
                        policy_input_history,
                        candidate_id=candidate_id,
                        step_index=step_index,
                    ),
                )
            except KeyError as exc:
                raise PolicyEpisodeError(
                    [f"{BLOCKER_ENVIRONMENT_CONTRACT}:{exc.args[0]}_missing"]
                ) from exc
            terminal_frame = persist_observation_frame(
                _policy_view_composite(
                    terminal_policy_observation, candidate_id=candidate_id
                ),
                output_dir=media_root,
                episode_id=episode_id,
                frame_index=len(retained_policy_frames),
                kind="terminal-observation",
            )
            visual_evidence, media_artifacts = finalize_visual_evidence(
                output_dir=media_root,
                episode_id=episode_id,
                identity={
                    "candidate_id": candidate_id,
                    "prompt": str(prompt),
                    "policy_input_view_order": (
                        [
                            GROOT_HISTORICAL_VIEW_KEYS[view]
                            for view in CANDIDATE_REQUIRED_VIEWS[candidate_id]
                        ]
                        if candidate_id == "groot_n17_droid"
                        else []
                    )
                    + list(CANDIDATE_REQUIRED_VIEWS[candidate_id]),
                    "legacy_media_profile": True,
                },
                policy_input_frames=retained_policy_frames,
                terminal_observation=terminal_frame,
            )
        timings_seconds["media_persistence"] += time.monotonic() - phase_started

    timings_seconds = {
        key: round(float(value), 6) for key, value in timings_seconds.items()
    }
    timings_seconds["total"] = round(time.monotonic() - episode_started, 6)

    receipt: dict[str, Any] = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "task_kind": task_kind,
        "task_spec": resolved_task_spec,
        "task_spec_digest": canonical_digest(resolved_task_spec),
        "prompt": str(prompt),
        "policy_queries": len(queries),
        "max_policy_queries": int(max_policy_queries),
        "environment_steps": step_index,
        "settle_window_samples": int(settle_window_samples),
        "open_loop_horizon": int(open_loop_horizon),
        "control_hz": DROID_CONTROL_HZ,
        "observation_adapter_schema_version": DROID_OBSERVATION_SCHEMA_VERSION,
        "action_space": commanded_action_magnitudes["source_action_space"],
        "observation_conversion": describe_observation_conversion(candidate_id),
        "destination_position_world_m": (
            [float(v) for v in destination_position_world_m]
            if destination_position_world_m is not None
            else None
        ),
        "queries": queries,
        "motion_evidence": motion_evidence,
        "commanded_action_magnitudes": commanded_action_magnitudes,
        "score": score,
        "candidate_policy_queried": True,
        "episode_id": episode_id,
        "visual_evidence": visual_evidence,
        "media_artifacts": media_artifacts,
        "observation_trace_digest": (
            canonical_digest(
                {
                    "observations": [
                        row["raw_rgb_sha256"] for row in retained_policy_frames
                    ]
                }
            )
            if retained_policy_frames
            else None
        ),
        "performance_diagnostics": {
            "clock": "time.monotonic",
            "claim_scope": "cycle_time_diagnostic_not_scientific_metric",
            "environment_step_bucket_includes_renderer_when_enabled": True,
            "timings_seconds": timings_seconds,
        },
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = [
    "ARM_MOTION_EPSILON_RAD",
    "DEFAULT_MAX_POLICY_QUERIES",
    "EPISODE_SCHEMA_VERSION",
    "DroidActionExecutionError",
    "DroidPolicyClient",
    "EpisodeEnvironment",
    "PolicyEpisodeError",
    "TaskScoringError",
    "run_policy_episode",
]
