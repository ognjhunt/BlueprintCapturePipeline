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
import shutil
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
    from policy_episode_lifecycle import (
        TERMINAL_PLANNED_DURATION,
        TERMINAL_POLICY_SAFETY,
        build_lifecycle,
        seal_prestart_readiness,
        validate_policy_episode_lifecycle,
    )
except ModuleNotFoundError:  # repository package
    from .policy_episode_lifecycle import (
        TERMINAL_PLANNED_DURATION,
        TERMINAL_POLICY_SAFETY,
        build_lifecycle,
        seal_prestart_readiness,
        validate_policy_episode_lifecycle,
    )
try:  # flat provider-bundle layout
    from adp009d_policy_episode_evidence import (
        ARM_MOTION_EPSILON_RAD,
        BLOCKER_CLIENT_RETURNED_NOTHING,
        PolicyEpisodeEvidenceError,
        motion_and_command_evidence as _build_motion_and_command_evidence,
        prevalidation_vendor_action_evidence as _prevalidation_vendor_action_evidence,
        raw_policy_action_evidence as _raw_policy_action_evidence,
        terminal_class_for_policy_exception as _terminal_class_for_policy_exception,
    )
except ModuleNotFoundError:  # repository package
    from .adp009d_policy_episode_evidence import (
        ARM_MOTION_EPSILON_RAD,
        BLOCKER_CLIENT_RETURNED_NOTHING,
        PolicyEpisodeEvidenceError,
        motion_and_command_evidence as _build_motion_and_command_evidence,
        prevalidation_vendor_action_evidence as _prevalidation_vendor_action_evidence,
        raw_policy_action_evidence as _raw_policy_action_evidence,
        terminal_class_for_policy_exception as _terminal_class_for_policy_exception,
    )
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
try:  # flat provider-bundle layout
    from policy_episode_trace_evidence import episode_trace_evidence
except ModuleNotFoundError:  # repository package
    from .policy_episode_trace_evidence import episode_trace_evidence

EPISODE_SCHEMA_VERSION = "adp009d_policy_episode.v4"

DEFAULT_MAX_POLICY_QUERIES = 60
EVALUATION_REVIEW_FRAME_STRIDE_STEPS = 8
BLOCKER_NO_SETTLE_WINDOW = "policy_episode_settle_window_not_reached"
BLOCKER_GRIPPER_PRESENT_IN_SETTLE = "policy_episode_gripper_present_during_settle"
BLOCKER_STEP_INDEX_NOT_INCREASING = "policy_episode_step_index_not_increasing"
BLOCKER_QUERY_BUDGET_EXHAUSTED = "policy_episode_query_budget_exhausted"
BLOCKER_ENVIRONMENT_CONTRACT = "policy_episode_environment_contract_violated"
BLOCKER_SOURCE_RESOLUTION_UNMEASURED = (
    "policy_episode_source_resolution_unmeasured_or_mixed"
)
BLOCKER_PRESTART_READINESS = "policy_episode_prestart_readiness_failed"
BLOCKER_POST_START_INFRASTRUCTURE = (
    "policy_episode_post_start_infrastructure_invariant_violation"
)
# Provider workers reserve space before they cross the scientific start
# boundary.  The raw-frame projection below is deliberately padded by this
# fixed floor for PNG/container overhead and atomic-write headroom.
PRESTART_MEDIA_RESERVE_FLOOR_BYTES = 64 * 1024 * 1024
PRESTART_MEDIA_RESERVE_MULTIPLIER = 3


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


def _policy_prestart_evidence(policy: DroidPolicyClient) -> dict[str, Any]:
    """Reconfirm the live control plane without performing inference."""

    preflight = getattr(policy, "preflight_readiness", None)
    if callable(preflight):
        raw = preflight()
    else:
        summary = getattr(policy, "evidence_summary", None)
        if not callable(summary):
            raise PolicyEpisodeError(
                [f"{BLOCKER_PRESTART_READINESS}:policy_readiness_method_missing"]
            )
        raw = summary()
    if not isinstance(raw, Mapping):
        raise PolicyEpisodeError(
            [f"{BLOCKER_PRESTART_READINESS}:policy_readiness_invalid"]
        )
    evidence = json.loads(json.dumps(dict(raw), allow_nan=False))
    if (
        evidence.get("identity_verified") is not True
        or evidence.get("candidate_policy_queried") not in {None, False}
        or evidence.get("candidate_inference_performed") not in {None, False}
        or evidence.get("policy_state_advanced") not in {None, False}
        or evidence.get("last_inference_evidence") is not None
    ):
        raise PolicyEpisodeError(
            [f"{BLOCKER_PRESTART_READINESS}:policy_control_plane_unready"]
        )
    return evidence


def _project_media_reserve_bytes(
    *,
    camera_rgb: Mapping[str, Any],
    evaluation_images: Mapping[str, Any],
    max_policy_queries: int,
    open_loop_horizon: int,
    settle_window_samples: int,
) -> int:
    """Conservatively reserve lossless-frame, review-video, and atomic-write space."""

    import numpy as np

    policy_raw = sum(int(np.asarray(frame).nbytes) for frame in camera_rgb.values())
    evaluation_raw = sum(
        int(np.asarray(frame).nbytes) for frame in evaluation_images.values()
    )
    action_steps = int(max_policy_queries) * int(open_loop_horizon)
    review_observations = (
        action_steps + int(settle_window_samples)
    ) // EVALUATION_REVIEW_FRAME_STRIDE_STEPS
    # Each query retains the exact composite plus native external/wrist PNGs.
    # Review/terminal observations retain all three cameras.  Multiplying the
    # raw projection covers lossless codec overhead, derived videos, manifests,
    # and the temporary bytes used by atomic writes/encoders.
    projected_raw = (
        2 * policy_raw * int(max_policy_queries)
        + evaluation_raw * (review_observations + 1)
    )
    return max(
        PRESTART_MEDIA_RESERVE_FLOOR_BYTES,
        PRESTART_MEDIA_RESERVE_MULTIPLIER * projected_raw,
    )


def _prestart_episode_readiness(
    *,
    environment: EpisodeEnvironment,
    policy: DroidPolicyClient,
    candidate_id: str,
    prompt: str,
    gripper: GripperConvention,
    task_kind: str,
    media_root: Path,
    episode_id: str,
    max_policy_queries: int,
    open_loop_horizon: int,
    settle_window_samples: int,
) -> dict[str, Any]:
    """Exercise every predictable runtime seam, then restore canonical reset.

    This rehearsal is outcome-blind: it never calls policy inference and its
    no-op robot command is discarded by a second canonical reset.  Its media
    lives under a distinct readiness episode id and therefore cannot be
    mistaken for scientific policy input.
    """

    environment.reset()
    joint_limits = environment.joint_limits()
    if (
        len(joint_limits) != ARM_JOINT_COUNT
        or any(len(row) != 2 for row in joint_limits)
        or any(
            not all(math.isfinite(float(value)) for value in row)
            or float(row[0]) >= float(row[1])
            for row in joint_limits
        )
    ):
        raise PolicyEpisodeError(
            [f"{BLOCKER_PRESTART_READINESS}:joint_limits_invalid"]
        )
    reset_joints = _read_arm_joint_positions(environment)
    if any(
        not float(lower) <= joint <= float(upper)
        for joint, (lower, upper) in zip(
            reset_joints, joint_limits, strict=True
        )
    ):
        raise PolicyEpisodeError(
            [f"{BLOCKER_PRESTART_READINESS}:reset_joint_state_out_of_bounds"]
        )
    required_task_field = (
        "can_pose_world"
        if task_kind == TASK_KIND_RIGID_PICK_PLACE
        else "joint_positions_rad"
    )
    initial_task_sample = dict(_read_task_sample(environment, task_kind=task_kind))
    if required_task_field not in initial_task_sample:
        raise PolicyEpisodeError(
            [f"{BLOCKER_PRESTART_READINESS}:task_state_invalid"]
        )
    inputs = environment.read_policy_inputs()
    camera_rgb = {
        view: inputs[view]
        for view in CANDIDATE_REQUIRED_VIEWS[candidate_id]
        if view in inputs
    }
    try:
        observation = build_droid_observation(
            candidate_id=candidate_id,
            camera_rgb=camera_rgb,
            joint_position=inputs["joint_position"],
            gripper_position=inputs["gripper_position"],
            prompt=prompt,
            eef_9d=inputs.get("eef_9d"),
        )
    except (KeyError, DroidObservationError) as exc:
        raise PolicyEpisodeError(
            [f"{BLOCKER_PRESTART_READINESS}:policy_observation_invalid:{exc}"]
        ) from exc

    image_reader = getattr(environment, "read_evaluation_camera_inputs", None)
    metadata_reader = getattr(environment, "read_control_observation_metadata", None)
    if not callable(image_reader) or not callable(metadata_reader):
        raise PolicyEpisodeError(
            [f"{BLOCKER_PRESTART_READINESS}:multicamera_contract_missing"]
        )
    evaluation_images = dict(image_reader())
    if set(evaluation_images) != {"external", "wrist", "overview"}:
        raise PolicyEpisodeError(
            [f"{BLOCKER_PRESTART_READINESS}:multicamera_set_invalid"]
        )
    reserve_bytes = _project_media_reserve_bytes(
        camera_rgb=camera_rgb,
        evaluation_images=evaluation_images,
        max_policy_queries=max_policy_queries,
        open_loop_horizon=open_loop_horizon,
        settle_window_samples=settle_window_samples,
    )
    media_root.mkdir(parents=True, exist_ok=True)
    free_bytes = int(shutil.disk_usage(media_root).free)
    if free_bytes < reserve_bytes:
        raise PolicyEpisodeError(
            [
                f"{BLOCKER_PRESTART_READINESS}:evidence_storage_insufficient:"
                f"{free_bytes}<{reserve_bytes}"
            ]
        )

    policy_evidence = _policy_prestart_evidence(policy)
    readiness_id = f"{episode_id}--prestart-readiness"
    exact_frame = persist_observation_frame(
        _policy_view_composite(observation, candidate_id=candidate_id),
        output_dir=media_root,
        episode_id=readiness_id,
        frame_index=0,
        kind="policy-input",
    )
    policy_observation = _persist_evaluation_camera_observation(
        environment,
        output_dir=media_root,
        episode_id=readiness_id,
        observation_index=0,
        kind="policy-input",
        exact_policy_input_camera_rgb={
            "external": camera_rgb[DROID_EXTERIOR_VIEW_1],
            "wrist": camera_rgb[DROID_WRIST_VIEW],
        },
    )

    # Exercise the same step/readback seam the episode will use, while holding
    # the reset joint targets.  This is not a learned-policy action.
    environment.step([*reset_joints, float(gripper.open_command)])
    probe_joints = _read_arm_joint_positions(environment)
    probe_task_sample = dict(_read_task_sample(environment, task_kind=task_kind))
    if required_task_field not in probe_task_sample:
        raise PolicyEpisodeError(
            [f"{BLOCKER_PRESTART_READINESS}:probe_task_state_invalid"]
        )
    terminal_observation = _persist_evaluation_camera_observation(
        environment,
        output_dir=media_root,
        episode_id=readiness_id,
        observation_index=1,
        kind="terminal-observation",
    )
    visual, artifacts = finalize_manipulation_evaluation_visual_evidence(
        output_dir=media_root,
        episode_id=readiness_id,
        identity={
            "candidate_id": candidate_id,
            "purpose": "outcome_blind_prestart_readiness",
            "candidate_policy_queried": False,
            "policy_input_camera_ids": ["external", "wrist"],
            "review_only_camera_ids": ["overview"],
        },
        policy_input_observations=[policy_observation],
        terminal_observation=terminal_observation,
    )
    if (
        visual.get("status") != "complete"
        or set(visual.get("videos") or {}) != {"external", "wrist", "overview"}
    ):
        raise PolicyEpisodeError(
            [f"{BLOCKER_PRESTART_READINESS}:review_media_roundtrip_failed"]
        )

    environment.reset()
    restored_joints = _read_arm_joint_positions(environment)
    restored_task_sample = dict(_read_task_sample(environment, task_kind=task_kind))
    if required_task_field not in restored_task_sample:
        raise PolicyEpisodeError(
            [f"{BLOCKER_PRESTART_READINESS}:restored_task_state_invalid"]
        )
    reset_restored = all(
        abs(left - right) <= ARM_MOTION_EPSILON_RAD
        for left, right in zip(reset_joints, restored_joints, strict=True)
    )
    if not reset_restored:
        raise PolicyEpisodeError(
            [f"{BLOCKER_PRESTART_READINESS}:canonical_reset_state_mismatch"]
        )
    queried = bool(getattr(policy, "candidate_policy_queried", False))
    readiness = seal_prestart_readiness(
        {
            "candidate_id": candidate_id,
            "episode_id": episode_id,
            "readiness_episode_id": readiness_id,
            "outcome_blind": True,
            "candidate_policy_queried": queried,
            "policy_state_advanced": False,
            "canonical_reset_restored": reset_restored,
            "checks": {
                "environment_reset": True,
                "joint_limits_readback": True,
                "joint_state_readback": True,
                "task_state_readback": True,
                "policy_observation_built": True,
                "policy_control_plane_ready": True,
                "evidence_storage_reserved": True,
                "exact_media_write_readback": bool(exact_frame.get("png_sha256")),
                "multicamera_write_readback": True,
                "review_video_encode_readback": True,
                "environment_step_readback": True,
                "canonical_reset_restored": reset_restored,
            },
            "storage_reservation": {
                "required_free_bytes": reserve_bytes,
                "observed_free_bytes": free_bytes,
                "projection_is_conservative": True,
            },
            "policy_control_plane": policy_evidence,
            "reset_joint_positions_rad": reset_joints,
            "probe_joint_positions_rad": probe_joints,
            "restored_joint_positions_rad": restored_joints,
            "initial_task_sample_digest": canonical_digest(initial_task_sample),
            "probe_task_sample_digest": canonical_digest(probe_task_sample),
            "restored_task_sample_digest": canonical_digest(restored_task_sample),
            "exact_media_frame": exact_frame,
            "visual_evidence": visual,
            "media_artifacts": artifacts,
        }
    )
    return readiness


def _motion_and_command_evidence(
    *,
    joint_trace: Sequence[Sequence[float]],
    commanded_actions: Sequence[Mapping[str, Any]],
    command_response_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return _build_motion_and_command_evidence(
            joint_trace=joint_trace,
            commanded_actions=commanded_actions,
            command_response_rows=command_response_rows,
        )
    except PolicyEpisodeEvidenceError as exc:
        raise PolicyEpisodeError(exc.errors) from exc


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
    require_prestart_readiness: bool = False,
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
            "candidate_discarded_tail_bounds_validated": None,
            "candidate_action_validated": False,
            "candidate_native_command_validated": False,
            "candidate_joint_state_validated": False,
            "candidate_action_applied": False,
            "episode_running": False,
            "episode_readiness_verified": False,
            "episode_started": False,
            "episode_terminal_class": None,
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
                        "candidate_discarded_tail_bounds_validated",
                        "candidate_action_validated",
                        "candidate_native_command_validated",
                        "candidate_joint_state_validated",
                        "candidate_action_applied",
                        "episode_running",
                        "episode_readiness_verified",
                        "episode_started",
                    )
                }
                | {
                    key: episode_progress[key]
                    for key in (
                        "candidate_policy_action_queries",
                        "commanded_actions",
                        "candidate_exact_policy_input_frames",
                        "prestart_readiness",
                        "episode_terminal_class",
                    )
                    if key in episode_progress
                }
            )

    prestart_readiness: dict[str, Any] | None = None
    if require_prestart_readiness:
        if media_root is None or episode_id is None:
            raise PolicyEpisodeError(
                [f"{BLOCKER_PRESTART_READINESS}:media_binding_required"]
            )
        _emit_progress("episode_readiness_started")
        prestart_readiness = _prestart_episode_readiness(
            environment=environment,
            policy=policy,
            candidate_id=candidate_id,
            prompt=prompt,
            gripper=gripper,
            task_kind=task_kind,
            media_root=media_root,
            episode_id=episode_id,
            max_policy_queries=int(max_policy_queries),
            open_loop_horizon=int(open_loop_horizon),
            settle_window_samples=int(settle_window_samples),
        )
        episode_progress["prestart_readiness"] = prestart_readiness
        episode_progress["episode_readiness_verified"] = True
        _emit_progress("episode_readiness_verified")
    episode_started_monotonic = time.monotonic()
    if require_prestart_readiness:
        episode_progress["episode_started"] = True
        _emit_progress("episode_started")
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

    def _seal_legitimate_early_terminal(exc: BaseException) -> dict[str, Any]:
        """Turn only a typed policy/scientific boundary into an episode result.

        The worker calls this after ``run_policy_episode`` unwinds.  Transport,
        renderer, disk, process, and environment exceptions are intentionally
        unclassified and therefore remain post-start infrastructure invariant
        violations rather than being disguised as scientific outcomes.
        """

        terminal_class = _terminal_class_for_policy_exception(
            exc,
            policy_inference_evidence=episode_progress.get(
                "policy_inference_evidence"
            ),
        )
        if terminal_class is None:
            raise PolicyEpisodeError(
                [f"{BLOCKER_POST_START_INFRASTRUCTURE}:{type(exc).__name__}:{exc}"]
            ) from exc
        if (
            not require_prestart_readiness
            or prestart_readiness is None
            or episode_progress.get("episode_started") is not True
        ):
            raise PolicyEpisodeError(
                [f"{BLOCKER_PRESTART_READINESS}:early_terminal_without_start_proof"]
            ) from exc
        visual_evidence, media_artifacts = _seal_terminal_visual_evidence()
        visual = (
            visual_evidence if isinstance(visual_evidence, Mapping) else {}
        )
        if (
            visual.get("status") != "complete"
            or set(visual.get("required_camera_ids") or ())
            != {"external", "wrist", "overview"}
            or set(visual.get("review_only_camera_ids") or ()) != {"overview"}
            or visual.get("terminal_observation_present") is not True
            or len(retained_multicamera_observations) != len(retained_policy_frames)
        ):
            raise PolicyEpisodeError(
                [
                    f"{BLOCKER_POST_START_INFRASTRUCTURE}:"
                    "early_terminal_media_incomplete"
                ]
            ) from exc
        visual = dict(visual)
        visual["episode_terminal_status"] = terminal_class
        visual["episode_terminal_reason"] = f"{type(exc).__name__}:{exc}"
        episode_progress["visual_evidence"] = visual

        motion_evidence, action_magnitudes = _motion_and_command_evidence(
            joint_trace=joint_trace,
            commanded_actions=commanded_actions,
            command_response_rows=command_response_rows,
        )
        actual_action_steps = sum(
            record.get("environment_step_applied") is True
            for record in commanded_actions
        )
        rounded_timings = {
            key: round(float(value), 6) for key, value in timings_seconds.items()
        }
        rounded_timings["total"] = round(
            time.monotonic() - episode_started_monotonic, 6
        )
        terminal_reason = f"{type(exc).__name__}:{exc}"
        score = {
            "status": "not_scored",
            "blockers": [terminal_class],
            "claim_boundary": (
                "typed candidate safety/scientific terminal retained; no task "
                "success, ranking, or superiority claim"
            ),
        }
        state_trace, contact_force_evidence, task_object_trajectory = (
            episode_trace_evidence(
                joint_trace=joint_trace,
                task_samples=samples,
                task_pose_field=rigid_pose_field,
            )
        )
        lifecycle = build_lifecycle(
            readiness=prestart_readiness,
            terminal_class=terminal_class,
            planned_policy_queries=int(max_policy_queries),
            planned_action_steps=int(max_policy_queries) * int(open_loop_horizon),
            planned_settle_steps=int(settle_window_samples),
            actual_policy_queries=len(candidate_policy_action_queries),
            actual_action_steps=actual_action_steps,
            actual_settle_steps=0,
            terminal_reason=terminal_reason,
            retained_terminal_result=True,
        )
        receipt: dict[str, Any] = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "task_kind": task_kind,
            "task_spec": resolved_task_spec,
            "task_spec_digest": canonical_digest(resolved_task_spec),
            "prompt": str(prompt),
            "policy_queries": len(candidate_policy_action_queries),
            "policy_observations_retained": len(retained_policy_frames),
            "max_policy_queries": int(max_policy_queries),
            "environment_steps": actual_action_steps,
            "settle_window_samples": int(settle_window_samples),
            "open_loop_horizon": int(open_loop_horizon),
            "control_hz": DROID_CONTROL_HZ,
            "observation_adapter_schema_version": DROID_OBSERVATION_SCHEMA_VERSION,
            "action_space": action_magnitudes["source_action_space"],
            "observation_conversion": describe_observation_conversion(
                candidate_id,
                source_hw=_measured_source_hw(observed_source_resolutions_hw),
            ),
            "destination_position_world_m": (
                [float(value) for value in destination_position_world_m]
                if destination_position_world_m is not None
                else None
            ),
            "queries": queries,
            "candidate_policy_action_queries": candidate_policy_action_queries,
            "commanded_actions": commanded_actions,
            "motion_evidence": motion_evidence,
            "state_trace": state_trace,
            "contact_force_evidence": contact_force_evidence,
            "task_object_trajectory": task_object_trajectory,
            "commanded_action_magnitudes": action_magnitudes,
            "score": score,
            "candidate_policy_queried": bool(
                episode_progress.get("candidate_policy_queried")
            ),
            "scoring_authorized": bool(scoring_authorized),
            "episode_id": episode_id,
            "visual_evidence": visual,
            "media_artifacts": media_artifacts,
            "candidate_exact_policy_input_frames": retained_policy_frames,
            "candidate_exact_policy_input_manifest_digest": canonical_digest(
                {"frames": retained_policy_frames}
            ),
            "observation_trace_digest": canonical_digest(
                {"observations": retained_policy_frames}
            ),
            "terminal_policy_exception": {
                "type": type(exc).__name__,
                "message": str(exc),
                "classification": terminal_class,
            },
            "prestart_readiness": prestart_readiness,
            "lifecycle": lifecycle,
            "performance_diagnostics": {
                "clock": "time.monotonic",
                "claim_scope": "cycle_time_diagnostic_not_scientific_metric",
                "environment_step_bucket_includes_renderer_when_enabled": True,
                "timings_seconds": rounded_timings,
            },
        }
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        validate_policy_episode_lifecycle(receipt)
        episode_progress["episode_terminal_class"] = terminal_class
        _emit_progress(
            "episode_policy_safety_terminal"
            if terminal_class == TERMINAL_POLICY_SAFETY
            else "episode_scientific_terminal"
        )
        return receipt

    episode_progress["_legitimate_early_terminal_finalizer"] = (
        _seal_legitimate_early_terminal
    )

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
        plan = plan_chunk_execution(
            action_values,
            horizon=int(open_loop_horizon),
            action_space=policy_action_space,
            candidate_id=candidate_id,
        )
        executed_prefix_rows = int(plan["executed_rows"])
        discarded_tail_rows = int(plan["discarded_rows"])
        raw_action_evidence.update(
            {
                "bounds_validation_scope": "executed_open_loop_prefix",
                "returned_rows": int(action_values.shape[0]),
                "executed_prefix_rows": executed_prefix_rows,
                "discarded_tail_rows": discarded_tail_rows,
                "executed_prefix_bounds_validated": False,
                "discarded_tail_bounds_validated": None,
                "nonexecuted_tail_scientific_output_retained": True,
            }
        )
        try:
            executed_prefix_bound_contract = validate_candidate_action_bounds(
                action_values[:executed_prefix_rows],
                action_space=policy_action_space,
                joint_limits=joint_limits,
                candidate_id=candidate_id,
            )
        except DroidActionExecutionError as exc:
            raw_action_evidence["raw_bound_validation_errors"] = list(exc.errors)
            raw_action_evidence["executed_prefix_bound_validation_errors"] = list(
                exc.errors
            )
            _emit_progress("policy_action_bounds_refused")
            raise
        raw_action_evidence["executed_prefix_bounds_validated"] = True
        episode_progress["candidate_action_bounds_validated"] = True

        full_response_bound_contract = None
        full_response_bound_errors: list[str] = []
        try:
            full_response_bound_contract = validate_candidate_action_bounds(
                action_values,
                action_space=policy_action_space,
                joint_limits=joint_limits,
                candidate_id=candidate_id,
            )
        except DroidActionExecutionError as exc:
            full_response_bound_errors = list(exc.errors)

        full_response_bounds_validated = not full_response_bound_errors
        discarded_tail_bounds_validated = (
            full_response_bounds_validated if discarded_tail_rows else None
        )
        raw_action_evidence["raw_bounds_validated"] = full_response_bounds_validated
        raw_action_evidence["discarded_tail_bounds_validated"] = (
            discarded_tail_bounds_validated
        )
        if full_response_bound_errors:
            # These indexes deliberately remain relative to the original full
            # response so retained evidence points at the exact vendor row.
            raw_action_evidence["raw_bound_validation_errors"] = (
                full_response_bound_errors
            )
            raw_action_evidence["discarded_tail_bound_validation_errors"] = (
                full_response_bound_errors
            )
        raw_bound_contract = {
            "validation_scope": "executed_open_loop_prefix",
            "returned_rows": int(action_values.shape[0]),
            "executed_prefix_rows": executed_prefix_rows,
            "discarded_tail_rows": discarded_tail_rows,
            "executed_prefix_bounds_validated": True,
            "executed_prefix_contract": executed_prefix_bound_contract,
            "discarded_tail_bounds_validated": discarded_tail_bounds_validated,
            "discarded_tail_bound_validation_errors": full_response_bound_errors,
            "full_raw_response_bounds_validated": full_response_bounds_validated,
            "full_raw_response_contract": full_response_bound_contract,
            "nonexecuted_tail_scientific_output_retained": True,
        }
        raw_action_evidence["raw_bound_contract"] = raw_bound_contract
        episode_progress["candidate_discarded_tail_bounds_validated"] = (
            discarded_tail_bounds_validated
        )
        _emit_progress("policy_action_bounds_validated")
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
                "position_adapter": action["position_adapter"],
                "joint_limit_clamped": bool(action["joint_limit_clamped"]),
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
                "raw_bound_contract": raw_bound_contract,
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
    state_trace, contact_force_evidence, task_object_trajectory = (
        episode_trace_evidence(
            joint_trace=joint_trace,
            task_samples=samples,
            task_pose_field=rigid_pose_field,
        )
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
    timings_seconds["total"] = round(
        time.monotonic() - episode_started_monotonic, 6
    )

    lifecycle = None
    if require_prestart_readiness:
        if prestart_readiness is None:
            raise PolicyEpisodeError(
                [f"{BLOCKER_PRESTART_READINESS}:receipt_missing"]
            )
        lifecycle = build_lifecycle(
            readiness=prestart_readiness,
            terminal_class=TERMINAL_PLANNED_DURATION,
            planned_policy_queries=int(max_policy_queries),
            planned_action_steps=int(max_policy_queries) * int(open_loop_horizon),
            planned_settle_steps=int(settle_window_samples),
            actual_policy_queries=len(queries),
            actual_action_steps=len(commanded_actions),
            actual_settle_steps=step_index - len(commanded_actions),
            terminal_reason="planned_policy_control_duration_completed",
            retained_terminal_result=True,
        )

    receipt: dict[str, Any] = {
        "schema_version": (
            EPISODE_SCHEMA_VERSION
            if require_prestart_readiness
            else "adp009d_policy_episode.v3"
        ),
        "candidate_id": candidate_id,
        "task_kind": task_kind,
        "task_spec": resolved_task_spec,
        "task_spec_digest": canonical_digest(resolved_task_spec),
        "prompt": str(prompt),
        "policy_queries": len(queries),
        "policy_observations_retained": len(retained_policy_frames),
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
        "state_trace": state_trace,
        "contact_force_evidence": contact_force_evidence,
        "task_object_trajectory": task_object_trajectory,
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
    if prestart_readiness is not None:
        receipt["prestart_readiness"] = prestart_readiness
    if lifecycle is not None:
        receipt["lifecycle"] = lifecycle
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    if require_prestart_readiness:
        validate_policy_episode_lifecycle(receipt)
        episode_progress["episode_terminal_class"] = TERMINAL_PLANNED_DURATION
        _emit_progress("episode_planned_duration_complete")
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
