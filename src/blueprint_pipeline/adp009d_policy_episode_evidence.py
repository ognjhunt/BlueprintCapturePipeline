"""Policy-action and motion evidence helpers for one ADP-009D episode.

This module is deliberately simulator-neutral.  It classifies typed policy
terminal conditions and reduces already-retained commands/readbacks into
digest-bound evidence without querying a policy or mutating an environment.
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

try:  # flat provider-bundle layout
    from adp009d_droid_action_execution import (
        ARM_JOINT_COUNT,
        BLOCKER_CHUNK_NONFINITE,
        BLOCKER_CHUNK_SHAPE,
        BLOCKER_GRIPPER_BOUNDS,
        BLOCKER_JOINT_POSITION_BOUNDS,
        BLOCKER_JOINT_VELOCITY_BOUNDS,
        DroidActionExecutionError,
    )
except ModuleNotFoundError:  # repository package
    from .adp009d_droid_action_execution import (
        ARM_JOINT_COUNT,
        BLOCKER_CHUNK_NONFINITE,
        BLOCKER_CHUNK_SHAPE,
        BLOCKER_GRIPPER_BOUNDS,
        BLOCKER_JOINT_POSITION_BOUNDS,
        BLOCKER_JOINT_VELOCITY_BOUNDS,
        DroidActionExecutionError,
    )
try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest
try:  # flat provider-bundle layout
    from policy_episode_lifecycle import TERMINAL_POLICY_SAFETY, TERMINAL_SCIENTIFIC
except ModuleNotFoundError:  # repository package
    from .policy_episode_lifecycle import TERMINAL_POLICY_SAFETY, TERMINAL_SCIENTIFIC


ARM_MOTION_EPSILON_RAD = 1e-6
BLOCKER_CLIENT_RETURNED_NOTHING = "policy_episode_client_returned_no_chunk"
PRESTART_READINESS_BLOCKER = "policy_episode_prestart_readiness_failed"
BLOCKER_POLICY_INPUT_SATURATION_GATE_UNAVAILABLE = (
    f"{PRESTART_READINESS_BLOCKER}:policy_input_saturation_gate_unavailable"
)


class PolicyEpisodeEvidenceError(ValueError):
    """Retained action evidence was internally inconsistent."""

    def __init__(self, errors: Sequence[str]):
        self.errors = [str(error) for error in errors]
        super().__init__(";".join(self.errors))


def policy_input_saturation_evidence(*, camera_rgb: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse clipped policy-input frames before any candidate query.

    The scene-839873 r13 cells fed both candidates frames whose captured site
    was a per-channel clamp of radiance far above display white (white blobs
    with chromatic fringes), while every retained review PNG had been
    display-encoded from the HDR buffer, so nothing upstream could see it.
    This reads the exact arrays the observation is built from.  A bundle that
    cannot import the gate refuses rather than proceeding blind.
    """

    try:  # flat provider-bundle layout
        from native_task_camera_observability import (
            NativeTaskCameraObservabilityError,
            validate_native_task_policy_input_frames,
        )
    except ModuleNotFoundError:  # repository package / arena bundle
        try:
            from .native_task_camera_observability import (
                NativeTaskCameraObservabilityError,
                validate_native_task_policy_input_frames,
            )
        except ImportError as exc:
            raise PolicyEpisodeEvidenceError(
                [BLOCKER_POLICY_INPUT_SATURATION_GATE_UNAVAILABLE]
            ) from exc
    try:
        return validate_native_task_policy_input_frames(camera_rgb)
    except NativeTaskCameraObservabilityError as exc:
        raise PolicyEpisodeEvidenceError(
            [f"{PRESTART_READINESS_BLOCKER}:{error}" for error in exc.errors]
        ) from exc


def prepolicy_visual_readiness_evidence(
    *,
    camera_rgb: Mapping[str, Any],
    candidate_policy_loaded: bool,
    candidate_policy_queried: bool = False,
    observation_integrity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure the exact three-camera reset domain before policy inference.

    ``candidate_policy_loaded`` is what the caller knows to be true: the
    paired session loads a candidate before its first episode, so an
    episode-level gate reports ``True`` even though no query has happened.
    ``observation_integrity`` carries the sealed
    ``policy_observation_integrity_authority.v1`` (``authority``) and the
    session's ``appearance_render_backend_receipt_digest``; without both the
    gate refuses before the first query.
    """

    binding = dict(observation_integrity or {})
    runtime_gate = binding.get("runtime_gate")
    runtime_gate = dict(runtime_gate) if isinstance(runtime_gate, Mapping) else None
    runtime_gate_passed = bool(
        runtime_gate
        and runtime_gate.get("schema_version")
        == "policy_canary_runtime_observation_integrity_gate.v1"
        and runtime_gate.get("status") == "passed"
        and runtime_gate.get("run_kind") == "internal_policy_canary"
        and runtime_gate.get("claim_ceiling") == "diagnostic_policy_execution"
        and runtime_gate.get("frame_structure_passed") is True
        and runtime_gate.get("target_semantic_visibility_passed") is True
        and runtime_gate.get("candidate_policy_loaded") is False
        and runtime_gate.get("candidate_policy_queried") is False
        and runtime_gate.get("official_ranking_permitted") is False
        and runtime_gate.get("scene_promotion_permitted") is False
        and runtime_gate.get("blockers") == []
        and runtime_gate.get("policy_observation_integrity_passed") is True
        and runtime_gate.get("appearance_render_backend_receipt_digest")
        == binding.get("appearance_render_backend_receipt_digest")
        and runtime_gate.get("gate_digest")
        == canonical_digest(runtime_gate, digest_field="gate_digest")
    )

    try:  # flat provider-bundle layout
        from native_task_camera_observability import (
            NativeTaskCameraObservabilityError,
            measure_native_task_prepolicy_visual_frames,
        )
    except ModuleNotFoundError:  # repository package / arena bundle
        try:
            from .native_task_camera_observability import (
                NativeTaskCameraObservabilityError,
                measure_native_task_prepolicy_visual_frames,
            )
        except ImportError as exc:
            raise PolicyEpisodeEvidenceError(
                [f"{PRESTART_READINESS_BLOCKER}:prepolicy_visual_gate_unavailable"]
            ) from exc
    try:
        receipt = measure_native_task_prepolicy_visual_frames(
            camera_rgb,
            candidate_policy_loaded=candidate_policy_loaded,
            candidate_policy_queried=candidate_policy_queried,
            observation_integrity_authority=binding.get("authority"),
            appearance_render_backend_receipt_digest=binding.get(
                "appearance_render_backend_receipt_digest"
            ),
        )
    except NativeTaskCameraObservabilityError as exc:
        raise PolicyEpisodeEvidenceError(
            [f"{PRESTART_READINESS_BLOCKER}:{error}" for error in exc.errors]
        ) from exc
    if runtime_gate_passed and receipt["frame_structure_passed"] is True:
        receipt = dict(receipt)
        receipt.update(
            policy_observation_integrity_passed=True,
            policy_observation_integrity_blockers=[],
            target_semantic_visibility_passed=True,
            appearance_reference_parity_passed=False,
            human_visual_review_status=(
                "not_required_for_internal_diagnostic_policy_execution"
            ),
            runtime_observation_gate={
                "gate_digest": runtime_gate["gate_digest"],
                "wrist_camera_mount_selection_digest": runtime_gate.get(
                    "wrist_camera_mount_selection_digest"
                ),
                "appearance_render_backend_receipt_digest": runtime_gate[
                    "appearance_render_backend_receipt_digest"
                ],
            },
            quality_boundary=(
                "internal diagnostic policy execution only; a digest-bound native "
                "camera-mount sweep, semantic task visibility, and exact reset-frame "
                "structure may unlock the unqualified canary without controls, "
                "reference-render parity, ranking, or scene promotion"
            ),
        )
    # A structural pass is not observation integrity: Scene 839873's frames
    # passed every structural check and were visibly corrupt.  Only sealed
    # same-pose parity bound to this backend plus human approval may unlock
    # the first query.
    blockers = (
        receipt["blockers"]
        if not receipt["frame_structure_passed"]
        else (
            []
            if receipt["policy_observation_integrity_passed"] is True
            else receipt["policy_observation_integrity_blockers"]
        )
    )
    if blockers:
        raise PolicyEpisodeEvidenceError(
            [f"{PRESTART_READINESS_BLOCKER}:{blocker}" for blocker in blockers]
        )
    return receipt


def json_safe_policy_action(value: Any) -> Any:
    """Retain candidate output before numerical validation, including NaN tags."""

    if isinstance(value, Mapping):
        return {
            str(key): json_safe_policy_action(item)
            for key, item in value.items()
        }
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"nonfinite_float": repr(value)}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [json_safe_policy_action(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return json_safe_policy_action(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return json_safe_policy_action(item())
    return {
        "unsupported_type": f"{type(value).__module__}.{type(value).__name__}"
    }


def raw_policy_action_evidence(chunk: Any, *, query_index: int) -> dict[str, Any]:
    """Retain the exact pre-validation policy return for scientific audit."""

    retained = json_safe_policy_action(chunk)
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


def terminal_class_for_policy_exception(
    exc: BaseException,
    *,
    policy_inference_evidence: Any,
) -> str | None:
    """Classify only predeclared candidate-result boundaries as legitimate."""

    message = str(exc)
    safety_prefixes = (
        BLOCKER_CHUNK_SHAPE,
        BLOCKER_CHUNK_NONFINITE,
        BLOCKER_JOINT_POSITION_BOUNDS,
        BLOCKER_JOINT_VELOCITY_BOUNDS,
        BLOCKER_GRIPPER_BOUNDS,
        BLOCKER_CLIENT_RETURNED_NOTHING,
        "groot_policy_response_invalid",
        "groot_policy_actions_invalid",
        "groot_policy_action_shape_mismatch",
        "groot_policy_action_nonfinite",
        "openpi_inference_response_",
    )
    if isinstance(exc, DroidActionExecutionError) and any(
        str(error).startswith(safety_prefixes) for error in exc.errors
    ):
        return TERMINAL_POLICY_SAFETY
    evidence = (
        policy_inference_evidence
        if isinstance(policy_inference_evidence, Mapping)
        else {}
    )
    if (
        evidence.get("server_response_received") is True
        and evidence.get("action_payload_returned") is True
        and message.startswith(safety_prefixes)
    ):
        return TERMINAL_POLICY_SAFETY
    if message == "groot_droid_eef_position_outside_checkpoint_observed_support":
        return TERMINAL_SCIENTIFIC
    return None


def prevalidation_vendor_action_evidence(
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


def motion_and_command_evidence(
    *,
    joint_trace: Sequence[Sequence[float]],
    commanded_actions: Sequence[Mapping[str, Any]],
    command_response_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reduce retained commands and joint readbacks into bounded evidence."""

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
    if len(source_action_spaces) > 1:
        raise PolicyEpisodeEvidenceError(
            ["policy_episode_source_action_space_inconsistent"]
        )
    source_action_space = (
        next(iter(source_action_spaces))
        if source_action_spaces
        else "none_no_executable_action"
    )
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


try:
    from adp009d_droid_observation import CANDIDATE_REQUIRED_VIEWS
except ModuleNotFoundError:
    from .adp009d_droid_observation import CANDIDATE_REQUIRED_VIEWS

BLOCKER_ENVIRONMENT_CONTRACT = "policy_episode_environment_contract_violated"
BLOCKER_SOURCE_RESOLUTION_UNMEASURED = "policy_episode_source_resolution_unmeasured_or_mixed"
BLOCKER_PRESTART_READINESS = PRESTART_READINESS_BLOCKER

def measured_source_hw(observed: set[tuple[int, int]]) -> tuple[int, int]:
    """Return the unique measured input size; refuse absent or mixed sizes."""

    if len(observed) != 1:
        raise PolicyEpisodeEvidenceError([BLOCKER_SOURCE_RESOLUTION_UNMEASURED])
    return next(iter(observed))


def policy_view_composite(
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
        raise PolicyEpisodeEvidenceError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:policy_media_view_invalid"]
        )
    if len({view.shape[0] for view in views}) != 1:
        raise PolicyEpisodeEvidenceError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:policy_media_view_height_mismatch"]
        )
    return np.ascontiguousarray(np.concatenate(views, axis=1))


def persist_evaluation_camera_observation(
    environment: Any,
    *,
    output_dir: Path,
    episode_id: str,
    observation_index: int,
    kind: str,
    exact_policy_input_camera_rgb: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from episode_visual_evidence import persist_multicamera_observation
    except ModuleNotFoundError:
        from .episode_visual_evidence import persist_multicamera_observation

    image_reader = getattr(environment, "read_evaluation_camera_inputs", None)
    metadata_reader = getattr(environment, "read_control_observation_metadata", None)
    if not callable(image_reader) or not callable(metadata_reader):
        raise PolicyEpisodeEvidenceError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:overview_camera_contract_missing"]
        )
    images = dict(image_reader())
    if exact_policy_input_camera_rgb is not None:
        if kind != "policy-input":
            raise PolicyEpisodeEvidenceError(
                [f"{BLOCKER_ENVIRONMENT_CONTRACT}:policy_camera_override_kind_invalid"]
            )
        exact_policy_inputs = dict(exact_policy_input_camera_rgb)
        if set(exact_policy_inputs) != {"external", "wrist"}:
            raise PolicyEpisodeEvidenceError(
                [f"{BLOCKER_ENVIRONMENT_CONTRACT}:policy_camera_override_invalid"]
            )
        # These are the exact raw arrays used to build the observation passed
        # to the policy.  The independent evaluation-camera read is retained
        # only for the overview stream; it must not silently substitute a
        # second external/wrist read for the policy's actual input bytes.
        images.update(exact_policy_inputs)
    missing = {"external", "wrist", "overview"} - set(images)
    if missing:
        raise PolicyEpisodeEvidenceError(
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


def read_arm_joint_positions(environment: Any) -> list[float]:
    reader = getattr(environment, "read_arm_joint_positions", None)
    if not callable(reader):
        raise PolicyEpisodeEvidenceError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:read_arm_joint_positions_missing"]
        )
    raw = reader()
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise PolicyEpisodeEvidenceError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:arm_joint_positions_invalid"]
        ) from exc
    if len(values) != ARM_JOINT_COUNT or not all(math.isfinite(value) for value in values):
        raise PolicyEpisodeEvidenceError(
            [
                f"{BLOCKER_ENVIRONMENT_CONTRACT}:"
                f"arm_joint_positions_invalid:{len(values)}"
            ]
        )
    return values


def policy_prestart_evidence(policy: Any) -> dict[str, Any]:
    """Reconfirm the live control plane without performing inference."""

    preflight = getattr(policy, "preflight_readiness", None)
    if callable(preflight):
        raw = preflight()
    else:
        summary = getattr(policy, "evidence_summary", None)
        if not callable(summary):
            raise PolicyEpisodeEvidenceError(
                [f"{BLOCKER_PRESTART_READINESS}:policy_readiness_method_missing"]
            )
        raw = summary()
    if not isinstance(raw, Mapping):
        raise PolicyEpisodeEvidenceError(
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
        raise PolicyEpisodeEvidenceError(
            [f"{BLOCKER_PRESTART_READINESS}:policy_control_plane_unready"]
        )
    return evidence


def request_storage_bytes(observation: Mapping[str, Any]) -> int:
    try:
        from policy_request_evidence import snapshot_request
    except ModuleNotFoundError:
        from .policy_request_evidence import snapshot_request
    return len(json.dumps(snapshot_request(observation), sort_keys=True).encode("utf-8")) + 4096


def project_media_reserve_bytes(
    *,
    camera_rgb: Mapping[str, Any],
    evaluation_images: Mapping[str, Any],
    max_policy_queries: int,
    open_loop_horizon: int,
    settle_window_samples: int,
    frame_stride: int,
    reserve_floor: int,
    reserve_multiplier: int,
    policy_request_bytes: int = 0,
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
    ) // frame_stride
    # Each query retains the exact composite plus native external/wrist PNGs.
    # Review/terminal observations retain all three cameras.  Multiplying the
    # raw projection covers lossless codec overhead, derived videos, manifests,
    # and the temporary bytes used by atomic writes/encoders.
    projected_raw = (
        (2 * policy_raw + 2 * int(policy_request_bytes)) * int(max_policy_queries)
        + evaluation_raw * (review_observations + 1)
    )
    return max(
        reserve_floor,
        reserve_multiplier * projected_raw,
    )


__all__ = [
    "measured_source_hw",
    "policy_view_composite",
    "persist_evaluation_camera_observation",
    "read_arm_joint_positions",
    "policy_prestart_evidence",
    "request_storage_bytes",
    "project_media_reserve_bytes",

    "BLOCKER_POLICY_INPUT_SATURATION_GATE_UNAVAILABLE",
    "PRESTART_READINESS_BLOCKER",
    "policy_input_saturation_evidence",
    "prepolicy_visual_readiness_evidence",
    "ARM_MOTION_EPSILON_RAD",
    "BLOCKER_CLIENT_RETURNED_NOTHING",
    "PolicyEpisodeEvidenceError",
    "json_safe_policy_action",
    "motion_and_command_evidence",
    "prevalidation_vendor_action_evidence",
    "raw_policy_action_evidence",
    "terminal_class_for_policy_exception",
]
