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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

try:  # flat provider-bundle layout
    from adp009d_droid_action_execution import (
        ACTION_SPACE_JOINT_VELOCITY,
        ARM_JOINT_COUNT,
        BLOCKER_CHUNK_NONFINITE,
        BLOCKER_CHUNK_SHAPE,
        DROID_ACTION_WIDTH,
        DROID_CONTROL_HZ,
        DROID_OPEN_LOOP_HORIZON,
        DroidActionExecutionError,
        GripperConvention,
        droid_row_to_isaac_action,
        plan_chunk_execution,
        validate_candidate_action_bounds,
    )
except ModuleNotFoundError:  # repository package
    from .adp009d_droid_action_execution import (
        ACTION_SPACE_JOINT_VELOCITY,
        ARM_JOINT_COUNT,
        BLOCKER_CHUNK_NONFINITE,
        BLOCKER_CHUNK_SHAPE,
        DROID_ACTION_WIDTH,
        DROID_CONTROL_HZ,
        DROID_OPEN_LOOP_HORIZON,
        DroidActionExecutionError,
        GripperConvention,
        droid_row_to_isaac_action,
        plan_chunk_execution,
        validate_candidate_action_bounds,
    )
try:  # flat provider-bundle layout
    from adp009d_droid_observation import (
        CANDIDATE_REQUIRED_VIEWS,
        DROID_EXTERIOR_VIEW_1,
        DROID_OBSERVATION_SCHEMA_VERSION,
        DROID_WRIST_VIEW,
        DroidObservationError,
        build_droid_observation,
        describe_observation_conversion,
    )
except ModuleNotFoundError:  # repository package
    from .adp009d_droid_observation import (
        CANDIDATE_REQUIRED_VIEWS,
        DROID_EXTERIOR_VIEW_1,
        DROID_OBSERVATION_SCHEMA_VERSION,
        DROID_WRIST_VIEW,
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
        TASK_SPEC_GRAPH_SCHEMA_VERSION,
        TASK_SPEC_SCHEMA_VERSION,
        TaskNeutralScoringError,
        score_task_episode_from_spec,
        validate_articulated_task_spec,
    )
except ModuleNotFoundError:  # repository package
    from .adp_task_scoring import (
        TASK_KIND_ARTICULATED_OPEN_CLOSE,
        TASK_KIND_RIGID_PICK_PLACE,
        TASK_SPEC_GRAPH_SCHEMA_VERSION,
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
        finalize_failed_policy_visual_evidence,
        finalize_manipulation_evaluation_visual_evidence,
        finalize_visual_evidence,
        persist_multicamera_observation,
        persist_observation_frame,
    )
except ModuleNotFoundError:  # repository package
    from .episode_visual_evidence import (
        finalize_failed_policy_visual_evidence,
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
BLOCKER_SOURCE_RESOLUTION_UNMEASURED = (
    "policy_episode_source_resolution_unmeasured_or_mixed"
)


class PolicyEpisodeError(ValueError):
    """Fail-closed episode contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def _measured_source_hw(observed: set[tuple[int, int]]) -> tuple[int, int]:
    """The one camera size every policy-input frame was observed to have.

    The receipt seals which conversion was applied; a defaulted or ambiguous
    source size would describe a conversion that did not happen, so anything
    but exactly one measured size refuses.
    """

    if len(observed) != 1:
        raise PolicyEpisodeError([BLOCKER_SOURCE_RESOLUTION_UNMEASURED])
    return next(iter(observed))


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


def maximum_policy_queries_for_task_spec(
    task_spec: Mapping[str, Any], *, open_loop_horizon: int
) -> int:
    """Return the shared full-step query budget for one frozen task.

    Both candidates execute the same number of actions from each returned
    chunk. Taking the integer number of complete chunks that fit inside the
    task's action ceiling gives them the same maximum interaction time without
    permitting either policy to overrun the preregistered episode.
    """

    raw_steps = task_spec.get("maximum_action_steps")
    if (
        isinstance(raw_steps, bool)
        or not isinstance(raw_steps, int)
        or raw_steps <= 0
        or isinstance(open_loop_horizon, bool)
        or not isinstance(open_loop_horizon, int)
        or open_loop_horizon <= 0
    ):
        raise PolicyEpisodeError(["policy_episode_action_budget_invalid"])
    queries = raw_steps // open_loop_horizon
    if queries <= 0:
        raise PolicyEpisodeError(["policy_episode_action_budget_invalid"])
    return queries


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


def _persist_evaluation_camera_observation(
    environment: EpisodeEnvironment,
    *,
    output_dir: Path,
    episode_id: str,
    observation_index: int,
    kind: str,
    exact_policy_input_camera_rgb: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    image_reader = getattr(environment, "read_evaluation_camera_inputs", None)
    metadata_reader = getattr(environment, "read_control_observation_metadata", None)
    if not callable(image_reader) or not callable(metadata_reader):
        raise PolicyEpisodeError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:overview_camera_contract_missing"]
        )
    images = dict(image_reader())
    if exact_policy_input_camera_rgb is not None:
        if kind != "policy-input":
            raise PolicyEpisodeError(
                [f"{BLOCKER_ENVIRONMENT_CONTRACT}:policy_camera_override_kind_invalid"]
            )
        exact_policy_inputs = dict(exact_policy_input_camera_rgb)
        if set(exact_policy_inputs) != {"external", "wrist"}:
            raise PolicyEpisodeError(
                [f"{BLOCKER_ENVIRONMENT_CONTRACT}:policy_camera_override_invalid"]
            )
        # These are the exact raw arrays used to build the observation passed
        # to the policy.  The independent evaluation-camera read is retained
        # only for the overview stream; it must not silently substitute a
        # second external/wrist read for the policy's actual input bytes.
        images.update(exact_policy_inputs)
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


def _json_safe_policy_action(value: Any) -> Any:
    """Retain candidate output before numerical validation, including NaN tags."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe_policy_action(item) for key, item in value.items()}
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"nonfinite_float": repr(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe_policy_action(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe_policy_action(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe_policy_action(item())
    return {"unsupported_type": f"{type(value).__module__}.{type(value).__name__}"}


def _raw_policy_action_evidence(chunk: Any, *, query_index: int) -> dict[str, Any]:
    retained = _json_safe_policy_action(chunk)
    record = {
        "query_index": int(query_index),
        "wire_value_type": f"{type(chunk).__module__}.{type(chunk).__name__}",
        "raw_action_chunk": retained,
        "shape_validated": False,
        "finite_values_validated": False,
        "raw_bounds_validated": False,
        "chunk_contract_validated": False,
    }
    record["raw_action_chunk_digest"] = canonical_digest(
        {"raw_action_chunk": retained}
    )
    return record


def _prevalidation_vendor_action_evidence(
    inference_evidence: Any, *, query_index: int
) -> dict[str, Any] | None:
    """Project a digest-verified vendor response into the episode query log."""

    if not isinstance(inference_evidence, Mapping):
        return None
    retained = inference_evidence.get("raw_vendor_action_response")
    observed_digest = inference_evidence.get(
        "raw_vendor_action_response_digest"
    )
    if (
        inference_evidence.get("server_response_received") is not True
        or retained is None
        or observed_digest
        != canonical_digest({"raw_vendor_action_response": retained})
    ):
        return None
    return {
        "query_index": int(query_index),
        "wire_value_type": str(
            inference_evidence.get("wire_response_type") or "unknown"
        ),
        "raw_vendor_action_response": retained,
        "raw_vendor_action_response_digest": observed_digest,
        "raw_vendor_action_response_role": inference_evidence.get(
            "raw_vendor_action_response_role"
        ),
        "action_payload_returned": (
            inference_evidence.get("action_payload_returned") is True
        ),
        "shape_validated": False,
        "finite_values_validated": False,
        "raw_bounds_validated": False,
        "chunk_contract_validated": False,
    }


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
    scoring_authorized: bool = True,
    require_complete_multicamera_media: bool = False,
    progress: dict[str, Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
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
    rigid_pose_field = (
        "task_object_pose_world"
        if resolved_task_spec.get("schema_version") == TASK_SPEC_GRAPH_SCHEMA_VERSION
        else "can_pose_world"
    )

    media_root = (
        Path(media_output_dir).expanduser().resolve()
        if media_output_dir is not None
        else None
    )
    retained_policy_frames: list[dict[str, Any]] = []
    retained_multicamera_observations: list[dict[str, Any]] = []
    retained_review_observations: list[dict[str, Any]] = []
    observed_source_resolutions_hw: set[tuple[int, int]] = set()
    media_observation_index = 0
    multicamera_evaluation_available = callable(
        getattr(environment, "read_evaluation_camera_inputs", None)
    ) and callable(getattr(environment, "read_control_observation_metadata", None))
    if require_complete_multicamera_media and (
        media_root is None
        or episode_id is None
        or not multicamera_evaluation_available
    ):
        raise PolicyEpisodeError(
            [
                f"{BLOCKER_ENVIRONMENT_CONTRACT}:"
                "complete_multicamera_media_contract_missing"
            ]
        )
    episode_progress = progress if progress is not None else {}
    if progress is not None:
        progress.clear()
    episode_progress.update(
        {
            "first_observation_retained": False,
            "exact_policy_observation_retained": False,
            "multicamera_policy_observation_retained": False,
            "candidate_policy_query_attempted": False,
            "candidate_policy_queried": False,
            "candidate_action_returned": False,
            "candidate_action_shape_validated": False,
            "candidate_action_finite_validated": False,
            "candidate_action_bounds_validated": False,
            "candidate_action_validated": False,
            "candidate_native_command_validated": False,
            "candidate_joint_state_validated": False,
            "candidate_action_applied": False,
            "episode_running": False,
        }
    )

    def _emit_progress(phase: str) -> None:
        episode_progress["phase"] = phase
        if progress_callback is not None:
            progress_callback(
                {
                    key: episode_progress[key]
                    for key in (
                        "phase",
                        "first_observation_retained",
                        "exact_policy_observation_retained",
                        "multicamera_policy_observation_retained",
                        "candidate_policy_query_attempted",
                        "candidate_policy_queried",
                        "candidate_action_returned",
                        "candidate_action_shape_validated",
                        "candidate_action_finite_validated",
                        "candidate_action_bounds_validated",
                        "candidate_action_validated",
                        "candidate_native_command_validated",
                        "candidate_joint_state_validated",
                        "candidate_action_applied",
                        "episode_running",
                    )
                }
                | {
                    key: episode_progress[key]
                    for key in (
                        "candidate_policy_action_queries",
                        "commanded_actions",
                        "candidate_exact_policy_input_frames",
                    )
                    if key in episode_progress
                }
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
                rigid_pose_field
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
    candidate_policy_action_queries: list[dict[str, Any]] = []
    command_response_rows = 0
    episode_progress["commanded_actions"] = commanded_actions
    episode_progress["candidate_policy_action_queries"] = (
        candidate_policy_action_queries
    )

    media_sealed = False
    sealed_visual_evidence: dict[str, Any] | None = None
    sealed_media_artifacts: list[dict[str, Any]] = []
    terminal_observation_record: dict[str, Any] | None = None

    def _seal_terminal_visual_evidence(
        *, failure_reason: str | None = None
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Seal retained media while the simulator is still readable.

        The worker invokes this closure if the episode raises after its first
        observation.  That keeps calibration, timestamps, exact RGB digests,
        manifests, and H.264 review videos from disappearing merely because a
        later policy/action/scoring step failed.
        """

        nonlocal media_observation_index
        nonlocal media_sealed
        nonlocal sealed_visual_evidence
        nonlocal sealed_media_artifacts
        nonlocal terminal_observation_record
        if media_sealed:
            return sealed_visual_evidence, sealed_media_artifacts
        if (
            media_root is None
            or episode_id is None
            or not retained_policy_frames
        ):
            return None, []
        phase_started = time.monotonic()
        native_multicamera_incomplete = bool(
            multicamera_evaluation_available
            and len(retained_multicamera_observations) != len(retained_policy_frames)
        )
        if failure_reason is not None:
            # Failure sealing is observation-only. Never issue a fresh camera,
            # joint, or simulator read and never manufacture a scientific
            # terminal frame after the exception that stopped the episode.
            visual_evidence, media_artifacts = (
                finalize_failed_policy_visual_evidence(
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
                    exact_policy_input_frames=retained_policy_frames,
                    multicamera_policy_input_observations=(
                        retained_multicamera_observations
                    ),
                    review_observations=retained_review_observations,
                    failure_reason=str(failure_reason),
                )
            )
        elif multicamera_evaluation_available and retained_multicamera_observations:
            if terminal_observation_record is None:
                terminal_observation_record = _persist_evaluation_camera_observation(
                    environment,
                    output_dir=media_root,
                    episode_id=episode_id,
                    observation_index=media_observation_index,
                    kind="terminal-observation",
                )
                media_observation_index += 1
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
                    terminal_observation=terminal_observation_record,
                )
            )
        else:
            if terminal_observation_record is None:
                terminal_inputs = environment.read_policy_inputs()
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
                    )
                except KeyError as exc:
                    raise PolicyEpisodeError(
                        [f"{BLOCKER_ENVIRONMENT_CONTRACT}:{exc.args[0]}_missing"]
                    ) from exc
                terminal_observation_record = persist_observation_frame(
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
                    "policy_input_view_order": list(
                        CANDIDATE_REQUIRED_VIEWS[candidate_id]
                    ),
                    "legacy_media_profile": True,
                },
                policy_input_frames=retained_policy_frames,
                terminal_observation=terminal_observation_record,
            )
        visual_evidence = dict(visual_evidence)
        if native_multicamera_incomplete:
            # The exact lossless composite was already retained before native
            # evaluation-camera persistence was attempted. Preserve its
            # digest-bound manifest and review video, but do not promote that
            # useful subset into a complete media claim: external/wrist raw
            # frames, calibration, and timestamps are still required.
            visual_evidence["status"] = "incomplete_after_first_observation"
            visual_evidence["media_gap"] = {
                "type": "after_first_observation_evidence_incomplete",
                "reason": (
                    str(failure_reason)
                    if failure_reason
                    else "native_multicamera_policy_observation_not_retained"
                ),
            }
            visual_evidence["exact_policy_observation_retained"] = True
            visual_evidence["multicamera_policy_observation_retained"] = bool(
                retained_multicamera_observations
            )
            visual_evidence["multicamera_policy_observation_complete"] = False
            visual_evidence["exact_policy_observation_count"] = len(
                retained_policy_frames
            )
            visual_evidence["multicamera_policy_observation_count"] = len(
                retained_multicamera_observations
            )
            visual_evidence["missing_required_evidence"] = [
                "native_external_camera_frames",
                "native_wrist_camera_frames",
                "native_camera_calibration_and_timestamps",
                "multicamera_frame_manifest",
                "per_camera_review_videos",
            ]
        visual_evidence["episode_terminal_status"] = (
            "failed_after_first_observation" if failure_reason else "completed"
        )
        if failure_reason:
            visual_evidence["episode_failure_reason"] = str(failure_reason)
        timings_seconds["media_persistence"] += time.monotonic() - phase_started
        sealed_visual_evidence = visual_evidence
        sealed_media_artifacts = media_artifacts
        # A typed derived-video gap is a durable failure result, but the
        # immutable frame manifest can still be re-entered later in the same
        # worker if the encoder becomes available before teardown.
        media_sealed = not bool(visual_evidence.get("video_gaps"))
        episode_progress["visual_evidence"] = visual_evidence
        episode_progress["media_artifacts"] = media_artifacts
        if failure_reason is None:
            _emit_progress("episode_media_sealed")
        return visual_evidence, media_artifacts

    episode_progress["_failure_media_finalizer"] = _seal_terminal_visual_evidence
    episode_progress["candidate_exact_policy_input_frames"] = retained_policy_frames

    for query_index in range(int(max_policy_queries)):
        phase_started = time.monotonic()
        inputs = environment.read_policy_inputs()
        timings_seconds["policy_input_read"] += time.monotonic() - phase_started
        camera_rgb = {
            view: inputs[view] for view in CANDIDATE_REQUIRED_VIEWS[candidate_id] if view in inputs
        }
        # The conversion receipt must report the size the cameras actually
        # rendered, so measure it from the frames the policy receives rather
        # than trusting the module's 1280x720 default.  One source size cannot
        # truthfully describe two differently sized views, so a mix refuses
        # immediately instead of sealing a conversion that did not happen.
        for frame in camera_rgb.values():
            shape = getattr(frame, "shape", None)
            if shape is not None and len(shape) >= 2:
                observed_source_resolutions_hw.add((int(shape[0]), int(shape[1])))
        if len(observed_source_resolutions_hw) > 1:
            raise PolicyEpisodeError([BLOCKER_SOURCE_RESOLUTION_UNMEASURED])
        phase_started = time.monotonic()
        try:
            observation = build_droid_observation(
                candidate_id=candidate_id,
                camera_rgb=camera_rgb,
                joint_position=inputs["joint_position"],
                gripper_position=inputs["gripper_position"],
                prompt=prompt,
                eef_9d=inputs.get("eef_9d"),
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
            view_order = list(CANDIDATE_REQUIRED_VIEWS[candidate_id])
            exact_frame = persist_observation_frame(
                _policy_view_composite(observation, candidate_id=candidate_id),
                output_dir=media_root,
                episode_id=episode_id,
                frame_index=query_index,
                kind="policy-input",
            )
            exact_frame.update(
                {
                    "candidate_id": candidate_id,
                    "candidate_exact_policy_input": True,
                    "view_order": view_order,
                    "view_shapes": {
                        view: list(observation[view].shape)
                        for view in view_order
                    },
                }
            )
            exact_frame["frame_manifest_digest"] = canonical_digest(exact_frame)
            retained_policy_frames.append(exact_frame)
            # This exact lossless composite is the authority for whether a first
            # policy observation was retained. Native evaluation-camera capture
            # is additional evidence and may fail independently after this byte
            # sequence is already durable.
            episode_progress["first_observation_retained"] = True
            episode_progress["exact_policy_observation_retained"] = True
            _emit_progress("first_observation")
            if multicamera_evaluation_available:
                multicamera_observation = _persist_evaluation_camera_observation(
                    environment,
                    output_dir=media_root,
                    episode_id=episode_id,
                    observation_index=media_observation_index,
                    kind="policy-input",
                    exact_policy_input_camera_rgb={
                        "external": camera_rgb[DROID_EXTERIOR_VIEW_1],
                        "wrist": camera_rgb[DROID_WRIST_VIEW],
                    },
                )
                retained_multicamera_observations.append(multicamera_observation)
                exact_frame["multicamera_observation_digest"] = (
                    multicamera_observation["observation_digest"]
                )
                exact_frame["raw_policy_input_camera_bindings"] = {
                    camera_id: {
                        "frame_digest": multicamera_observation["views"][camera_id][
                            "frame_digest"
                        ],
                        "raw_rgb_sha256": multicamera_observation["views"][camera_id][
                            "raw_rgb_sha256"
                        ],
                    }
                    for camera_id in ("external", "wrist")
                }
                exact_frame["frame_manifest_digest"] = canonical_digest(
                    exact_frame, digest_field="frame_manifest_digest"
                )
                media_observation_index += 1
                episode_progress["multicamera_policy_observation_retained"] = True
                _emit_progress("multicamera_observation_retained")
            timings_seconds["media_persistence"] += time.monotonic() - phase_started

        phase_started = time.monotonic()
        episode_progress["candidate_policy_query_attempted"] = True
        _emit_progress("policy_query_started")
        try:
            chunk = policy.infer(observation)
        except BaseException as exc:
            timings_seconds["policy_inference"] += time.monotonic() - phase_started
            inference_evidence_reader = getattr(
                policy, "last_inference_evidence", None
            )
            try:
                failure_inference_evidence = (
                    inference_evidence_reader()
                    if callable(inference_evidence_reader)
                    else None
                )
            except (TypeError, ValueError):
                failure_inference_evidence = None
            vendor_action_evidence = _prevalidation_vendor_action_evidence(
                failure_inference_evidence, query_index=query_index
            )
            if vendor_action_evidence is not None:
                episode_progress["candidate_policy_queried"] = True
                episode_progress["policy_inference_evidence"] = (
                    failure_inference_evidence
                )
                _emit_progress("policy_response_received")
                if vendor_action_evidence["action_payload_returned"]:
                    episode_progress["candidate_action_returned"] = True
                    candidate_policy_action_queries.append(vendor_action_evidence)
                    if (
                        str(exc) == "groot_policy_action_shape_mismatch"
                        or str(exc).startswith("openpi_inference_response_")
                    ):
                        _emit_progress("policy_action_shape_refused")
                    elif str(exc) == "groot_policy_action_nonfinite":
                        vendor_action_evidence["shape_validated"] = True
                        episode_progress["candidate_action_shape_validated"] = True
                        _emit_progress("policy_action_shape_validated")
                        _emit_progress("policy_action_finite_refused")
            raise
        timings_seconds["policy_inference"] += time.monotonic() - phase_started
        episode_progress["candidate_policy_queried"] = True
        if chunk is None:
            raise PolicyEpisodeError([BLOCKER_CLIENT_RETURNED_NOTHING])
        inference_evidence_reader = getattr(policy, "last_inference_evidence", None)
        policy_inference_evidence = (
            inference_evidence_reader()
            if callable(inference_evidence_reader)
            else None
        )
        episode_progress["candidate_action_returned"] = True
        episode_progress["policy_inference_evidence"] = policy_inference_evidence
        raw_action_evidence = _prevalidation_vendor_action_evidence(
            policy_inference_evidence, query_index=query_index
        ) or _raw_policy_action_evidence(
            chunk, query_index=query_index
        )
        candidate_policy_action_queries.append(raw_action_evidence)
        _emit_progress("policy_response_received")

        phase_started = time.monotonic()
        import numpy as np

        try:
            action_values = np.asarray(chunk, dtype=float)
        except (TypeError, ValueError) as exc:
            raw_action_evidence["shape_validation_error"] = f"{type(exc).__name__}:{exc}"
            _emit_progress("policy_action_shape_refused")
            raise DroidActionExecutionError(
                [f"{BLOCKER_CHUNK_SHAPE}:not_numeric"]
            ) from exc
        raw_action_evidence["observed_shape"] = list(action_values.shape)
        if action_values.ndim != 2 or action_values.shape[1] != DROID_ACTION_WIDTH:
            _emit_progress("policy_action_shape_refused")
            raise DroidActionExecutionError(
                [f"{BLOCKER_CHUNK_SHAPE}:{tuple(action_values.shape)}"]
            )
        raw_action_evidence["shape_validated"] = True
        episode_progress["candidate_action_shape_validated"] = True
        _emit_progress("policy_action_shape_validated")
        if not np.isfinite(action_values).all():
            _emit_progress("policy_action_finite_refused")
            raise DroidActionExecutionError([BLOCKER_CHUNK_NONFINITE])
        raw_action_evidence["finite_values_validated"] = True
        episode_progress["candidate_action_finite_validated"] = True
        _emit_progress("policy_action_finite_validated")
        try:
            raw_bound_contract = validate_candidate_action_bounds(
                action_values,
                action_space=policy_action_space,
                joint_limits=joint_limits,
            )
        except DroidActionExecutionError as exc:
            raw_action_evidence["raw_bound_validation_errors"] = list(exc.errors)
            _emit_progress("policy_action_bounds_refused")
            raise
        raw_action_evidence["raw_bounds_validated"] = True
        raw_action_evidence["raw_bound_contract"] = raw_bound_contract
        episode_progress["candidate_action_bounds_validated"] = True
        _emit_progress("policy_action_bounds_validated")
        plan = plan_chunk_execution(
            action_values,
            horizon=int(open_loop_horizon),
            action_space=policy_action_space,
            candidate_id=candidate_id,
        )
        timings_seconds["action_planning"] += time.monotonic() - phase_started
        raw_action_evidence["chunk_contract_validated"] = True
        episode_progress["candidate_action_validated"] = True
        _emit_progress("first_policy_action")
        query_clamped_rows = 0
        for action_index, planned_action in enumerate(plan["actions"]):
            before = list(joint_trace[-1])
            action = droid_row_to_isaac_action(
                planned_action["droid_action"],
                current_joint_position=before,
                joint_limits=joint_limits,
                gripper=gripper,
                action_space=policy_action_space,
                candidate_id=candidate_id,
            )
            query_clamped_rows += int(action["joint_limit_clamped"])
            action_record = {
                "joint_position_target_rad": [
                    float(value) for value in action["joint_position_target_rad"]
                ],
                "joint_velocity_command_rad_s": list(
                    action["joint_velocity_command_rad_s"]
                ),
                "source_arm_command": list(action["source_arm_command"]),
                "source_action_space": action["source_action_space"],
                "clipped_droid_action": list(action["clipped_droid_action"]),
                "observed_before_rad": before,
                "observed_after_rad": None,
                "isaac_action": [float(value) for value in action["isaac_action"]],
                "query_index": query_index,
                "action_index_within_query": action_index,
                "step_index": step_index + 1,
                "native_command_validated": True,
                "joint_state_before_validated": True,
                "environment_step_attempted": False,
                "environment_step_applied": False,
                "joint_state_after_validated": False,
            }
            commanded_actions.append(action_record)
            episode_progress["candidate_native_command_validated"] = True
            _emit_progress("native_command_validated")
            phase_started = time.monotonic()
            action_record["environment_step_attempted"] = True
            _emit_progress("environment_step_started")
            environment.step(action["isaac_action"])
            timings_seconds["environment_step_including_render"] += (
                time.monotonic() - phase_started
            )
            step_index += 1
            action_record["environment_step_applied"] = True
            episode_progress["candidate_action_applied"] = True
            episode_progress["episode_running"] = True
            _emit_progress("episode_running")
            phase_started = time.monotonic()
            after = _read_arm_joint_positions(environment)
            timings_seconds["joint_state_read"] += time.monotonic() - phase_started
            action_record["observed_after_rad"] = after
            action_record["joint_state_after_validated"] = True
            episode_progress["candidate_joint_state_validated"] = True
            _emit_progress("joint_state_validated")
            joint_trace.append(after)
            target = [float(value) for value in action["joint_position_target_rad"]]
            response_observed = any(
                abs(after[index] - before[index]) > ARM_MOTION_EPSILON_RAD
                and (target[index] - before[index]) * (after[index] - before[index]) > 0.0
                for index in range(ARM_JOINT_COUNT)
            )
            command_response_rows += int(response_observed)
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
            phase_started = time.monotonic()
            samples.append(
                _sample_with_index(
                    _read_task_sample(environment, task_kind=task_kind),
                    step_index,
                    previous_index,
                    required_field=(
                        rigid_pose_field
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
                "returned_chunk": plan["returned_chunk"],
                "returned_chunk_digest": canonical_digest(
                    {"returned_chunk": plan["returned_chunk"]}
                ),
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
        phase_started = time.monotonic()
        samples.append(
            _sample_with_index(
                _read_task_sample(environment, task_kind=task_kind),
                step_index,
                previous_index,
                required_field=(
                    rigid_pose_field
                    if task_kind == TASK_KIND_RIGID_PICK_PLACE
                    else "joint_positions_rad"
                ),
            )
        )
        timings_seconds["object_state_sample"] += time.monotonic() - phase_started
        previous_index = step_index

    if step_index - settle_start_index < int(settle_window_samples):
        raise PolicyEpisodeError([BLOCKER_NO_SETTLE_WINDOW])

    if scoring_authorized:
        phase_started = time.monotonic()
        score = score_task_episode_from_spec(
            task_spec=resolved_task_spec,
            samples=samples,
        )
        timings_seconds["deterministic_scoring"] += time.monotonic() - phase_started
    else:
        score = {
            "status": "not_scored",
            "blockers": ["unqualified_controls_policy_diagnostic"],
            "claim_boundary": (
                "policy actions and simulator observations retained; task outcome "
                "not scored, ranked, or qualified"
            ),
        }
    motion_evidence, commanded_action_magnitudes = _motion_and_command_evidence(
        joint_trace=joint_trace,
        commanded_actions=commanded_actions,
        command_response_rows=command_response_rows,
    )

    visual_evidence, media_artifacts = _seal_terminal_visual_evidence()
    if require_complete_multicamera_media:
        required_cameras = {"external", "wrist", "overview"}
        visual = visual_evidence if isinstance(visual_evidence, Mapping) else {}
        videos = visual.get("videos")
        if (
            visual.get("status") != "complete"
            or visual.get("episode_terminal_status") != "completed"
            or set(visual.get("required_camera_ids") or ()) != required_cameras
            or set(visual.get("review_only_camera_ids") or ()) != {"overview"}
            or visual.get("terminal_observation_present") is not True
            or not isinstance(videos, Mapping)
            or set(videos) != required_cameras
            or len(retained_policy_frames) != len(queries)
            or len(retained_multicamera_observations) != len(retained_policy_frames)
            or visual.get("policy_input_observation_count") != len(queries)
            or visual.get("policy_input_frame_count") != 2 * len(queries)
            or any(
                frame.get("frame_manifest_digest")
                != canonical_digest(frame, digest_field="frame_manifest_digest")
                or frame.get("multicamera_observation_digest")
                != observation.get("observation_digest")
                or any(
                    (frame.get("raw_policy_input_camera_bindings") or {}).get(
                        camera_id
                    )
                    != {
                        "frame_digest": observation["views"][camera_id][
                            "frame_digest"
                        ],
                        "raw_rgb_sha256": observation["views"][camera_id][
                            "raw_rgb_sha256"
                        ],
                    }
                    for camera_id in ("external", "wrist")
                )
                for frame, observation in zip(
                    retained_policy_frames,
                    retained_multicamera_observations,
                    strict=True,
                )
            )
        ):
            raise PolicyEpisodeError(
                [
                    f"{BLOCKER_ENVIRONMENT_CONTRACT}:"
                    "complete_multicamera_media_invalid"
                ]
            )

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
        "observation_conversion": describe_observation_conversion(
            candidate_id, source_hw=_measured_source_hw(observed_source_resolutions_hw)
        ),
        "destination_position_world_m": (
            [float(v) for v in destination_position_world_m]
            if destination_position_world_m is not None
            else None
        ),
        "queries": queries,
        "candidate_policy_action_queries": candidate_policy_action_queries,
        "commanded_actions": commanded_actions,
        "motion_evidence": motion_evidence,
        "commanded_action_magnitudes": commanded_action_magnitudes,
        "score": score,
        "candidate_policy_queried": True,
        "scoring_authorized": bool(scoring_authorized),
        "episode_id": episode_id,
        "visual_evidence": visual_evidence,
        "media_artifacts": media_artifacts,
        "candidate_exact_policy_input_frames": retained_policy_frames,
        "candidate_exact_policy_input_manifest_digest": (
            canonical_digest({"frames": retained_policy_frames})
            if retained_policy_frames
            else None
        ),
        "observation_trace_digest": (
            canonical_digest({"observations": retained_policy_frames})
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
    _emit_progress("episode_complete")
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
    "maximum_policy_queries_for_task_spec",
    "run_policy_episode",
]
