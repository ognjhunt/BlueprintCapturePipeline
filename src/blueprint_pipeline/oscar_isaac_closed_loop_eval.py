"""Per-step policy <-> OSCAR WAM <-> perception closed-loop evaluation.

Each policy action conditions the next generated observation, the perception
harness analyzes that frame, and its derived observation feeds the next step.
Injected WAM and perception backends keep hermetic and real-model runs on the
same structural path without treating either as semantic task proof.
"""

from __future__ import annotations

import argparse
import base64
import csv
import errno
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .gpu_residency_attribution import (
    ATTRIBUTION_MODE_DEVICE_HANDLE_FALLBACK,
    ATTRIBUTION_MODE_HOST_PID_NAMESPACE,
    compute_app_attribution_unavailable,
    device_handle_residency_fallback,
    linux_nvidia_host_to_local_pid_map as _linux_nvidia_host_to_local_pid_map,
    pid_ancestor_chain as _pid_ancestor_chain,
)
from .initial_policy_observation_contract import (
    resolve_start_frame_evidence_path,
    validated_initial_policy_observation as _validated_initial_policy_observation,
)
from .closed_loop_consistency_scoring import (
    _score_closed_loop_step_episode_consistency as _score_closed_loop_step_episode_consistency_impl,
)
from .g1_kitchen_worker_proof_emission import (
    emit_rows_from_closed_loop_state,
    legacy_worker_proof_rows,
)
from .generated_episode_authority import (
    bind_generated_episode_to_authoritative_loop_status as _bind_generated_episode_to_authoritative_loop_status,
)
from .sc3_fidelity_contracts import (
    SC3_TASK_COMPLETION_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    validate_checkpoint_attestation,
    validate_synchronized_multiview,
    validate_trusted_ed25519_attestation,
)
from .isaac_g1_policy import (
    DeterministicWalkToTargetPolicy,
    StepContext,
    action_record,
    interpolate_route,
)
from .oscar_wam_provider_command_adapter import run as run_oscar_wam_provider_adapter
from .oscar_wam_command_adapter import DEFAULT_NUM_FRAMES as DEFAULT_OSCAR_NUM_FRAMES
from .oscar_wam_provider_bundle import _normalize_oscar_robot_action_prompt
from .oscar_official_release import (
    OFFICIAL_OSCAR_HF_REPO,
    official_release_blockers,
    official_release_contract,
)
from .oscar_cosmos_wam_evaluator import (  # noqa: F401 - preserve test/caller seams
    WAM_CONSISTENCY_COMMAND_ENV,
    WAM_CONSISTENCY_COMMAND_OUTPUT,
    WAM_CONSISTENCY_GATE_ENV,
    WAM_SUCCESS_LABEL_COMMAND_ENV,
    WAM_SUCCESS_LABEL_COMMAND_OUTPUT,
    WAM_SUCCESS_LABEL_GATE_ENV,
    _env_truthy as _wam_consistency_env_truthy,
    _normalize_wam_episode_consistency,
    _normalize_wam_success_labels,
    _run_wam_consistency_command,
    _run_wam_success_label_command,
    _unscored_wam_episode_consistency,
    _wam_consistency_blockers,
    success_label_inference_input_sha256,
)
from .wam_backend_strategy import get_wam_backend_strategy
from .wam_derived_observation_harness import (
    GENERATED_RGB_POLICY_OBSERVATION_BACKEND_KIND,
    run_wam_derived_observation_harness_step,
)
from .wam_generated_video_review import visual_smoke_generated_rollouts_for_review
from .wam_provider_runtime import WAM_PROVIDER_COMMAND_ENV_BY_SUBSTRATE
from .wam_action_consistency_contract import cross_step_action_motion_replay_blockers

LOOP_SCHEMA_VERSION = "oscar_isaac_closed_loop_eval.v1"
NEXT_OBSERVATION_SELECTION_SCHEMA_VERSION = "oscar_next_observation_selection.v1"
CLOSED_LOOP_WAM_BACKEND_READINESS_SCHEMA_VERSION = "closed_loop_wam_backend_readiness.v1"
SUPPORTED_CLOSED_LOOP_WAM_BACKENDS = ("oscar_wam", "cosmos3_wam")
BUILT_IN_CLOSED_LOOP_WAM_BACKENDS = frozenset({"oscar_wam", "cosmos3_wam"})
VAST_API_GATE_ENV = "BLUEPRINT_ALLOW_VAST_API_CALLS"
VAST_INSTANCE_LAUNCH_GATE_ENV = "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH"
VAST_PAID_WAM_GATE_ENV = "BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH"
VAST_API_KEY_FILE_ENV = "VAST_API_KEY_FILE"
PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST_ENV = (
    "BLUEPRINT_PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST"
)
PERSISTENT_WAM_CLEAN_FRAME_REANCHOR_INTERVAL_ENV = (
    "BLUEPRINT_PERSISTENT_WAM_CLEAN_FRAME_REANCHOR_INTERVAL_STEPS"
)
ALLOW_EXPERIMENTAL_OSCAR_VERSION_ENV = "BLUEPRINT_ALLOW_EXPERIMENTAL_OSCAR_WAM_VERSION"
SC3_LEARNED_POLICY_RUNTIME_TRUSTED_PUBLIC_KEY_SHA256_ENV = (
    "BLUEPRINT_SC3_LEARNED_POLICY_RUNTIME_TRUSTED_PUBLIC_KEY_SHA256"
)
SC3_FK_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV = (
    "BLUEPRINT_SC3_FK_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256"
)
CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_ENV = "BLUEPRINT_CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT"
CONTROLLER_FK_CAMERA_PROJECTION_SCHEMA_VERSION = "controller_fk_camera_projection_context.v1"
CONTROLLER_FK_CAMERA_PROJECTION_LIVE_STATUS = "captured_from_live_persistent_isaac_session"
CONTROLLER_FK_CAMERA_PROJECTION_TRANSFORM = "mujoco_pelvis_relative_to_live_isaac_pelvis_wxyz"
SC3_COSMOS3_RUNTIME_TRUSTED_PUBLIC_KEY_SHA256_ENV = (
    "BLUEPRINT_SC3_COSMOS3_RUNTIME_TRUSTED_PUBLIC_KEY_SHA256"
)
UNITREE_G1_SONIC_STATE_DIMS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
    "projected_gravity": 3,
}
POST_ACTION_POLICY_STATE_SOURCE = "post_action_live_isaac_articulation"
MANIPULATION_EFFECTOR_PROGRESS_MINIMUM_M = 0.015
APPROACH_MEASUREMENT_SOURCE = "chunk_fk_from_canonical_initial_state"
MANIPULATION_EFFECTOR_PROJECTED_MOTION_MINIMUM_PX = 8.0
UNSAFE_STANCE_MAX_HORIZONTAL_PROJECTED_GRAVITY = 0.5
UNSAFE_STANCE_MIN_UPRIGHT_PROJECTED_GRAVITY_Z = -0.7
# A SONIC action chunk is 40 frames (about 0.8 seconds at 50 Hz). Three stale
# decisions allow one short reach/grasp attempt, then terminate on the fourth
# observation instead of paying for the historical eight stale WAM generations.
DEFAULT_NO_PROGRESS_PATIENCE_STEPS = 3
OSCAR_GPU_RESIDENCY_SAMPLE_SCHEMA_VERSION = "oscar_gpu_residency_sample.v1"
OSCAR_GPU_RESIDENCY_REPORT_SCHEMA_VERSION = "oscar_gpu_residency_report.v1"
OSCAR_GPU_RESIDENCY_REQUIRED_ROLES = ("groot", "gear_sonic", "isaac_task", "oscar")
OSCAR_GPU_RESIDENCY_PID_ENV_BY_ROLE = {
    "groot": "GROOT_PID",
    "gear_sonic": "GEAR_SONIC_PID",
    "isaac_task": "ISAAC_TASK_PID",
}
OSCAR_GPU_RESIDENCY_SAMPLE_INTERVAL_SECONDS = 0.75
OSCAR_GPU_RESIDENCY_MAX_SAMPLES = 4800
OSCAR_SUBPROCESS_TERMINATION_GRACE_SECONDS = 10.0

# A WAM generation backend: given the current observation frame, the policy action, the step
# index, and the action history, produce the next-observation frame path (and optional video).
# Returns a mapping with at least {"generated_frame_path": <path>}.
WamGenerateNext = Callable[
    [str, Mapping[str, Any], int, Sequence[Mapping[str, Any]]], Mapping[str, Any]
]

# A learned policy endpoint: given the harness-adapted WAM-generated observation,
# prior action history, and step index, return the next action dict.
PolicyEndpoint = Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]], int], Mapping[str, Any]]
TaskCompletionEvaluator = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _score_closed_loop_step_episode_consistency(**kwargs: Any) -> dict[str, Any]:
    """Keep the renderer seam injectable through this module's established interface."""
    return _score_closed_loop_step_episode_consistency_impl(
        **kwargs,
        visual_smoke_fn=visual_smoke_generated_rollouts_for_review,
    )


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in (_string(item).strip() for item in value) if item]
    return []


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sc3_runtime_attestation(
    signed_payload: Mapping[str, Any],
    *,
    private_key_file: str | Path,
    report_path: str | Path,
    signer_key_id: str,
    verifier_id: str,
) -> dict[str, Any]:
    """Sign a typed runtime result for a separately trusted process boundary."""

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = serialization.load_pem_private_key(
        Path(private_key_file).expanduser().read_bytes(), password=None
    )
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("sc3_runtime_signing_key_must_be_ed25519")
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signed_bytes = json.dumps(dict(signed_payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    signed_payload_sha256 = hashlib.sha256(signed_bytes).hexdigest()
    report = Path(report_path).expanduser()
    ensure_dir(report.parent)
    report.write_text(
        json.dumps(
            {
                "schema_version": "sc3_signature_verification_report.v1",
                "algorithm": "Ed25519",
                "verification_status": "verified",
                "public_key_sha256": hashlib.sha256(public_key).hexdigest(),
                "signed_payload_sha256": signed_payload_sha256,
                "signer_key_id": signer_key_id,
                "verifier_id": verifier_id,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "algorithm": "Ed25519",
        "signature_verified": True,
        "signer_key_id": signer_key_id,
        "verifier_id": verifier_id,
        "public_key_base64": base64.b64encode(public_key).decode("ascii"),
        "public_key_sha256": hashlib.sha256(public_key).hexdigest(),
        "signature_base64": base64.b64encode(private_key.sign(signed_bytes)).decode("ascii"),
        "signed_payload_sha256": signed_payload_sha256,
        "verification_report_artifact": {
            "path": str(report.resolve()),
            "sha256": _file_sha256(report),
        },
    }


def _is_sha256(value: Any) -> bool:
    text = _string(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _load_live_controller_fk_camera_projection_context(path: str | Path) -> dict[str, Any]:
    context_path = Path(path).expanduser().resolve()
    if not context_path.is_file() or context_path.is_symlink():
        raise ValueError("controller_fk_camera_projection_context_missing_or_unsafe")
    try:
        value = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("controller_fk_camera_projection_context_unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError("controller_fk_camera_projection_context_not_object")
    context = dict(value)
    if context.get("schema_version") != CONTROLLER_FK_CAMERA_PROJECTION_SCHEMA_VERSION:
        raise ValueError("controller_fk_camera_projection_context_schema_invalid")
    if context.get("status") != CONTROLLER_FK_CAMERA_PROJECTION_LIVE_STATUS:
        raise ValueError("controller_fk_camera_projection_context_not_live_capture")
    if context.get("coordinate_transform") != CONTROLLER_FK_CAMERA_PROJECTION_TRANSFORM:
        raise ValueError("controller_fk_camera_projection_context_transform_invalid")
    for field in ("attempt_id", "launch_nonce", "simulator_session_id", "stage_id"):
        if not _string(context.get(field)).strip():
            raise ValueError(f"controller_fk_camera_projection_context_{field}_missing")
    frame = _mapping(context.get("source_frame_artifact"))
    frame_path = Path(_string(frame.get("path"))).expanduser().resolve()
    frame_sha256 = _string(frame.get("sha256")).lower()
    if (
        frame_path.is_symlink()
        or not frame_path.is_file()
        or not _is_sha256(frame_sha256)
        or _file_sha256(frame_path) != frame_sha256
        or int(frame.get("width") or 0) != 640
        or int(frame.get("height") or 0) != 480
    ):
        raise ValueError("controller_fk_camera_projection_context_frame_invalid")
    camera = _mapping(context.get("camera_contract"))
    intrinsics = _mapping(camera.get("intrinsics"))
    clipping_range = camera.get("clipping_range_m")
    try:
        near_m, far_m = [float(value) for value in clipping_range]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "controller_fk_camera_projection_context_camera_clipping_range_invalid"
        ) from exc
    if (
        camera.get("available") is not True
        or camera.get("projection_token") != "perspective"
        or list(camera.get("resolution") or []) != [640, 480]
        or camera.get("viewpoint_mode") != "robot_head_mounted_egocentric"
        or camera.get("mount_motion_model") != "rigid_head_local_transform"
        or camera.get("gaze_motion_model") != "inherits_head_orientation_no_task_reaim"
        or intrinsics.get("available") is not True
        or int(intrinsics.get("image_width") or 0) != 640
        or int(intrinsics.get("image_height") or 0) != 480
    ):
        raise ValueError("controller_fk_camera_projection_context_camera_invalid")
    if (
        not isinstance(clipping_range, Sequence)
        or isinstance(clipping_range, (str, bytes, bytearray))
        or len(clipping_range) != 2
        or not all(math.isfinite(value) for value in (near_m, far_m))
        or not 0.0 < near_m < far_m
    ):
        raise ValueError("controller_fk_camera_projection_context_camera_clipping_range_invalid")
    registration = _mapping(context.get("standing_cross_simulator_registration"))
    if (
        registration.get("status") != "pending_official_mujoco_named_link_residual_verification"
        or registration.get("surrogate") is not False
    ):
        raise ValueError("controller_fk_camera_projection_context_registration_invalid")
    return context


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_numeric_sequence(value: Any, *, minimum_length: int = 1) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    numbers = [_finite_float(item) for item in value]
    return len(numbers) >= minimum_length and all(item is not None for item in numbers)


def _validated_post_action_policy_state(
    value: Any,
    *,
    simulator_session_id: str,
    stage_id: str,
    source_action_sha256: str,
    source_step_index: int,
) -> dict[str, Any]:
    """Validate same-session Isaac proprioception before a learned-policy requery.

    A WAM/controller-FK state is useful diagnostic evidence, but it is not the
    articulation state after the action was applied.  This contract therefore
    accepts only the exact G1 grouped state measured by the persistent Isaac
    session and bound to this action and loop step.
    """

    state = _mapping(value)
    if not state:
        raise RuntimeError("post_action_policy_state_missing_or_not_object")
    normalized: dict[str, Any] = dict(state)
    for field, dimension in UNITREE_G1_SONIC_STATE_DIMS.items():
        values = state.get(field)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise RuntimeError(f"post_action_policy_state_{field}_not_sequence")
        if len(values) != dimension:
            raise RuntimeError(
                f"post_action_policy_state_{field}_dimension_{len(values)}_expected_{dimension}"
            )
        finite_values = [_finite_float(item) for item in values]
        if any(item is None for item in finite_values):
            raise RuntimeError(f"post_action_policy_state_{field}_nonfinite")
        normalized[field] = [float(item) for item in finite_values if item is not None]

    measurement = _mapping(state.get("measurement"))
    if not measurement:
        raise RuntimeError("post_action_policy_state_measurement_missing_or_not_object")
    if measurement.get("surrogate") is not False:
        raise RuntimeError("post_action_policy_state_surrogate_not_false")
    if _string(measurement.get("source")).strip() != POST_ACTION_POLICY_STATE_SOURCE:
        raise RuntimeError("post_action_policy_state_source_not_live_post_action_isaac")
    expected_session_id = _string(simulator_session_id).strip()
    expected_stage_id = _string(stage_id).strip()
    expected_action_sha256 = _string(source_action_sha256).strip().lower()
    if not expected_session_id:
        raise RuntimeError("post_action_policy_state_expected_simulator_session_id_missing")
    if not expected_stage_id:
        raise RuntimeError("post_action_policy_state_expected_stage_id_missing")
    if not _is_sha256(expected_action_sha256):
        raise RuntimeError("post_action_policy_state_expected_action_sha256_invalid")
    if _string(measurement.get("simulator_session_id")).strip() != expected_session_id:
        raise RuntimeError("post_action_policy_state_simulator_session_id_mismatch")
    if _string(measurement.get("stage_id")).strip() != expected_stage_id:
        raise RuntimeError("post_action_policy_state_stage_id_mismatch")
    if _string(measurement.get("source_action_sha256")).strip().lower() != (expected_action_sha256):
        raise RuntimeError("post_action_policy_state_source_action_sha256_mismatch")
    observed_step = measurement.get("source_step_index")
    if (
        isinstance(observed_step, bool)
        or not isinstance(observed_step, int)
        or observed_step != int(source_step_index)
    ):
        raise RuntimeError("post_action_policy_state_source_step_index_mismatch")
    captured_at_ns = measurement.get("captured_at_ns")
    if isinstance(captured_at_ns, bool):
        raise RuntimeError("post_action_policy_state_captured_at_ns_invalid")
    try:
        captured_at_ns_int = int(captured_at_ns)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("post_action_policy_state_captured_at_ns_invalid") from exc
    if captured_at_ns_int <= 0:
        raise RuntimeError("post_action_policy_state_captured_at_ns_invalid")
    normalized["measurement"] = dict(measurement)
    return normalized


def _post_action_stance_report(state: Mapping[str, Any]) -> dict[str, Any]:
    """Classify live Isaac stance from the controller's projected-gravity vector.

    The upright convention used by the G1 policy is ``[0, 0, -1]``.  The
    deliberately loose running threshold catches a robot that is plainly
    falling or down without treating ordinary gait lean as a terminal event.
    """

    gravity = state.get("projected_gravity")
    if not _finite_numeric_sequence(gravity, minimum_length=3) or len(gravity) != 3:
        return {
            "schema_version": "post_action_stance_report.v1",
            "status": "invalid",
            "unsafe_stance_detected": True,
            "blockers": ["post_action_projected_gravity_missing_or_invalid"],
        }
    gx, gy, gz = (float(value) for value in gravity)
    unsafe = bool(
        abs(gx) > UNSAFE_STANCE_MAX_HORIZONTAL_PROJECTED_GRAVITY
        or abs(gy) > UNSAFE_STANCE_MAX_HORIZONTAL_PROJECTED_GRAVITY
        or gz > UNSAFE_STANCE_MIN_UPRIGHT_PROJECTED_GRAVITY_Z
    )
    return {
        "schema_version": "post_action_stance_report.v1",
        "status": "unsafe" if unsafe else "upright",
        "unsafe_stance_detected": unsafe,
        "projected_gravity": [gx, gy, gz],
        "thresholds": {
            "maximum_absolute_horizontal_component": (
                UNSAFE_STANCE_MAX_HORIZONTAL_PROJECTED_GRAVITY
            ),
            "maximum_z_for_upright": UNSAFE_STANCE_MIN_UPRIGHT_PROJECTED_GRAVITY_Z,
        },
        "blockers": ["unsafe_post_action_robot_stance"] if unsafe else [],
    }


def _task_progress_report(
    completion_result: Mapping[str, Any],
    *,
    minimum_progress_fraction: float,
) -> dict[str, Any]:
    """Normalize live task-transition measurements for an online stall watchdog.

    This is a resource-control signal, not a success verdict. Success still
    requires the registered, attested task-transition contract.
    """

    comparison = _string(completion_result.get("comparison")).strip()
    initial = _finite_float(completion_result.get("episode_initial_value"))
    current = _finite_float(completion_result.get("after_value"))
    tolerance = _finite_float(completion_result.get("tolerance"))
    target = _finite_float(completion_result.get("target_value"))
    fraction = float(minimum_progress_fraction)
    if (
        comparison
        not in {
            "increase_at_least",
            "decrease_at_least",
            "absolute_change_at_least",
            "within_tolerance",
            "at_or_above",
            "at_or_below",
        }
        or initial is None
        or current is None
        or tolerance is None
        or not math.isfinite(fraction)
        or fraction <= 0.0
    ):
        return {
            "schema_version": "online_task_progress_report.v1",
            "status": "unavailable",
            "resource_control_only": True,
            "blockers": ["online_task_progress_measurement_unavailable"],
        }
    if comparison in {"increase_at_least", "at_or_above"}:
        progress = current - initial
    elif comparison in {"decrease_at_least", "at_or_below"}:
        progress = initial - current
    elif comparison == "absolute_change_at_least":
        progress = abs(current - initial)
    elif target is not None:
        progress = abs(initial - target) - abs(current - target)
    else:
        return {
            "schema_version": "online_task_progress_report.v1",
            "status": "unavailable",
            "resource_control_only": True,
            "blockers": ["online_task_progress_target_value_unavailable"],
        }
    minimum_delta = max(1e-6, abs(tolerance) * fraction)
    return {
        "schema_version": "online_task_progress_report.v1",
        "status": "measured",
        "resource_control_only": True,
        "criterion_id": completion_result.get("criterion_id"),
        "comparison": comparison,
        "episode_initial_value": initial,
        "current_value": current,
        "target_value": target,
        "success_tolerance": tolerance,
        "progress_toward_criterion": float(progress),
        "minimum_meaningful_progress_delta": minimum_delta,
        "registered_transition_passed": bool(
            completion_result.get("registered_transition_passed") is True
        ),
        "blockers": [],
        "claim_boundary": (
            "This online signal may stop a stalled episode to conserve runtime; "
            "it cannot prove task success or replace the registered transition judge."
        ),
    }


def _manipulation_effector_progress_report(
    projection: Mapping[str, Any],
    *,
    minimum_progress_m: float = MANIPULATION_EFFECTOR_PROGRESS_MINIMUM_M,
    minimum_projected_motion_px: float = (MANIPULATION_EFFECTOR_PROJECTED_MOTION_MINIMUM_PX),
) -> dict[str, Any]:
    """Measure whether a controller FK horizon moves a hand/wrist toward the task.

    This is a capability/sanity gate, not task-success or contact proof.  It
    prevents an upper-body horizon whose end effectors remain static from being
    handed to an expensive WAM as though it were a manipulation action.
    """

    target = projection.get("task_target_world_xyz_m")
    if not _finite_numeric_sequence(target, minimum_length=3) or len(target) != 3:
        return {
            "schema_version": "manipulation_effector_progress_report.v1",
            "status": "blocked",
            "capability_gate_passed": False,
            "blockers": ["manipulation_task_target_world_xyz_missing_or_invalid"],
        }
    target_xyz = tuple(float(value) for value in target)
    sequence = projection.get("controller_fk_sequence")
    if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes, bytearray)):
        sequence = _mapping(projection.get("generated_robot_state")).get("controller_fk_sequence")
    if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes, bytearray)):
        sequence = []

    distance_by_effector: dict[str, list[float]] = {}
    position_by_effector: dict[str, list[tuple[float, float, float]]] = {}
    projected_position_by_effector: dict[str, list[tuple[float, float]]] = {}
    in_frame_count_by_effector: dict[str, int] = {}
    for frame in sequence:
        landmarks = _mapping(frame).get("landmarks")
        if not isinstance(landmarks, Sequence) or isinstance(landmarks, (str, bytes, bytearray)):
            continue
        observed_this_frame: set[str] = set()
        for landmark in landmarks:
            row = _mapping(landmark)
            name = _string(row.get("name") or row.get("landmark_id")).strip().lower()
            if not name or ("wrist" not in name and "hand" not in name):
                continue
            world_xyz = row.get("world_xyz") or row.get("world_xyz_m")
            if not _finite_numeric_sequence(world_xyz, minimum_length=3) or len(world_xyz) != 3:
                continue
            if name in observed_this_frame:
                continue
            observed_this_frame.add(name)
            xyz = tuple(float(value) for value in world_xyz)
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(xyz, target_xyz, strict=True)))
            distance_by_effector.setdefault(name, []).append(distance)
            position_by_effector.setdefault(name, []).append(xyz)
            image_projection = _mapping(row.get("image_projection"))
            u_px = _finite_float(image_projection.get("u_px"))
            v_px = _finite_float(image_projection.get("v_px"))
            if u_px is not None and v_px is not None:
                projected_position_by_effector.setdefault(name, []).append((u_px, v_px))
                if image_projection.get("available") is True:
                    in_frame_count_by_effector[name] = in_frame_count_by_effector.get(name, 0) + 1

    effector_rows: list[dict[str, Any]] = []
    for name, distances in sorted(distance_by_effector.items()):
        if len(distances) < 2:
            continue
        first_distance = distances[0]
        minimum_distance = min(distances)
        positions = position_by_effector[name]
        first_position = positions[0]
        maximum_displacement = max(math.dist(first_position, position) for position in positions)
        projected_positions = projected_position_by_effector.get(name, [])
        maximum_projected_displacement = (
            max(math.dist(projected_positions[0], position) for position in projected_positions)
            if len(projected_positions) >= 2
            else 0.0
        )
        effector_rows.append(
            {
                "effector": name,
                "frame_count": len(distances),
                "first_distance_m": round(first_distance, 9),
                "minimum_distance_m": round(minimum_distance, 9),
                "final_distance_m": round(distances[-1], 9),
                "maximum_progress_toward_target_m": round(
                    max(0.0, first_distance - minimum_distance), 9
                ),
                "maximum_displacement_from_first_frame_m": round(maximum_displacement, 9),
                "projected_frame_count": len(projected_positions),
                "in_frame_projection_count": in_frame_count_by_effector.get(name, 0),
                "maximum_projected_displacement_from_first_frame_px": round(
                    maximum_projected_displacement, 6
                ),
            }
        )
    best_progress = max(
        (float(row["maximum_progress_toward_target_m"]) for row in effector_rows),
        default=0.0,
    )
    best_displacement = max(
        (float(row["maximum_displacement_from_first_frame_m"]) for row in effector_rows),
        default=0.0,
    )
    directional_progress_passed = bool(effector_rows and best_progress >= float(minimum_progress_m))
    motion_capability_passed = bool(
        effector_rows and best_displacement >= float(minimum_progress_m)
    )
    best_projected_displacement = max(
        (
            float(row["maximum_projected_displacement_from_first_frame_px"])
            for row in effector_rows
            if int(row["in_frame_projection_count"]) >= 2
        ),
        default=0.0,
    )
    projected_motion_capability_passed = bool(
        effector_rows and best_projected_displacement >= float(minimum_projected_motion_px)
    )
    # For a registered manipulation target, arbitrary arm motion is not a
    # sufficient basis for a goal-conditioned WAM transition. Otherwise the
    # language goal can overpower an action that visibly moves away and cause
    # the object to animate without contact. Require measurable target-directed
    # progress before the expensive learned transition; later Isaac
    # apply/readback still remains the authority for contact and task success.
    passed = (
        directional_progress_passed
        and motion_capability_passed
        and projected_motion_capability_passed
    )
    blockers: list[str] = []
    if not directional_progress_passed:
        blockers.append("manipulation_controller_fk_no_directional_effector_progress")
    if not motion_capability_passed:
        blockers.append("manipulation_controller_fk_no_meaningful_effector_motion")
    if not projected_motion_capability_passed:
        blockers.append("manipulation_controller_fk_no_visible_projected_effector_motion")
    warnings: list[str] = []
    return {
        "schema_version": "manipulation_effector_progress_report.v1",
        "status": "passed" if passed else "blocked",
        "capability_gate_passed": passed,
        "task_target_world_xyz_m": list(target_xyz),
        "minimum_required_progress_m": float(minimum_progress_m),
        "minimum_required_projected_motion_px": float(minimum_projected_motion_px),
        "best_progress_toward_target_m": round(best_progress, 9),
        "best_effector_displacement_m": round(best_displacement, 9),
        "best_visible_projected_effector_displacement_px": round(best_projected_displacement, 6),
        "directional_progress_passed": directional_progress_passed,
        "motion_capability_passed": motion_capability_passed,
        "projected_motion_capability_passed": (projected_motion_capability_passed),
        "effectors": effector_rows,
        "blockers": blockers,
        "warnings": warnings,
        "claim_boundary": (
            "Controller-FK end-effector motion is a pre-generation action-conditioning "
            "capability check. Directional progress is recorded separately and does not "
            "replace live Isaac task progress, contact, articulation transition, or "
            "task-success proof."
        ),
    }


def _validated_post_action_egocentric_frame(
    response: Mapping[str, Any], *, source_step_index: int
) -> dict[str, Any]:
    """Select the latest hash-bound head-mounted RGB from an Isaac action."""
    raw_rows = response.get("review_frames") or response.get("review_media_artifacts")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes, bytearray)):
        return {}
    candidates: list[dict[str, Any]] = []
    for value in raw_rows:
        row = _mapping(value)
        if row.get("camera_role") != "robot_pov" or row.get("outer_source_step_index") != int(
            source_step_index
        ):
            continue
        camera = _mapping(row.get("camera_contract"))
        if (
            camera.get("viewpoint_mode") != "robot_head_mounted_egocentric"
            or camera.get("robot_mounted") is not True
            or camera.get("policy_observation_eligible") is not True
            or camera.get("mount_motion_model") != "rigid_head_local_transform"
            or camera.get("gaze_motion_model") != "inherits_head_orientation_no_task_reaim"
        ):
            raise RuntimeError("post_action_robot_pov_not_egocentric")
        path = Path(_string(row.get("path"))).expanduser().resolve()
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("post_action_robot_pov_frame_missing_or_unsafe")
        observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_sha256 = _string(row.get("sha256")).strip().lower()
        if not _is_sha256(expected_sha256) or observed_sha256 != expected_sha256:
            raise RuntimeError("post_action_robot_pov_frame_sha256_mismatch")
        if (int(row.get("width") or 0), int(row.get("height") or 0)) != (640, 480):
            raise RuntimeError("post_action_robot_pov_frame_resolution_invalid")
        visual_signal = _mapping(row.get("visual_signal"))
        candidates.append(
            {
                "path": str(path),
                "sha256": observed_sha256,
                "width": 640,
                "height": 480,
                "camera_role": "robot_pov",
                "viewpoint_mode": "robot_head_mounted_egocentric",
                "mount_motion_model": "rigid_head_local_transform",
                "gaze_motion_model": "inherits_head_orientation_no_task_reaim",
                "source_step_index": int(source_step_index),
                "frame_index": int(row.get("frame_index") or 0),
                "control_frame_global_index": int(row.get("control_frame_global_index") or 0),
                "captured_at_ns": int(row.get("captured_at_ns") or 0),
                "visual_signal_valid": not visual_signal
                or visual_signal.get("status") == "completed"
                and visual_signal.get("non_uniform") is True,
            }
        )
    if not candidates:
        return {}
    latest = max(
        candidates,
        key=lambda row: (
            row["control_frame_global_index"],
            row["frame_index"],
            row["captured_at_ns"],
        ),
    )
    return latest if latest.pop("visual_signal_valid") is True else {}


def _landmark_has_numeric_evidence(landmark: Mapping[str, Any]) -> bool:
    for key in ("world_xyz", "camera_xyz", "position_xyz", "xyz", "position"):
        if _finite_numeric_sequence(landmark.get(key), minimum_length=3):
            return True
    if (
        _finite_float(landmark.get("x")) is not None
        and _finite_float(landmark.get("y")) is not None
    ):
        return True
    projection = _mapping(landmark.get("image_projection"))
    return bool(
        projection.get("available") is True
        and _finite_float(projection.get("u_px")) is not None
        and _finite_float(projection.get("v_px")) is not None
    )


def _landmark_evidence_blockers(landmarks: Sequence[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    if not landmarks:
        return ["fresh_action_projected_skeleton_missing"]
    image_space_landmark_count = 0
    for index, landmark in enumerate(landmarks):
        landmark_id = _string(
            landmark.get("landmark_id") or landmark.get("name") or landmark.get("label")
        ).strip()
        if not landmark_id:
            blockers.append(f"fresh_action_skeleton_landmark_id_missing:{index}")
        if not _landmark_has_numeric_evidence(landmark):
            blockers.append(f"fresh_action_skeleton_landmark_numeric_evidence_missing:{index}")
        # OSCAR consumes an image-space skeleton video.  Controller/FK XYZ is
        # useful kinematic evidence, but it cannot be drawn truthfully without
        # the camera projection that produced the current RGB observation.
        # Accepting bare x/y here previously let an all-black conditioning
        # video pass the action-identity contract.
        projection = _mapping(landmark.get("image_projection"))
        available = projection.get("available")
        if available is True:
            if (
                _finite_float(projection.get("u_px")) is None
                or _finite_float(projection.get("v_px")) is None
            ):
                blockers.append(
                    f"fresh_action_skeleton_landmark_pixel_projection_missing_or_invalid:{index}"
                )
            else:
                image_space_landmark_count += 1
        elif (
            available is False
            and projection.get("unavailable_reason") == "outside_live_camera_viewport"
            and _finite_float(projection.get("u_px")) is not None
            and _finite_float(projection.get("v_px")) is not None
        ):
            # A true head camera may place an official arm action entirely
            # below the RGB viewport.  Finite out-of-view projections remain
            # exact camera-space action evidence and are rendered as explicit
            # edge indicators in the separate conditioning stream.
            image_space_landmark_count += 1
        elif not (available is False and _string(projection.get("unavailable_reason")).strip()):
            blockers.append(
                f"fresh_action_skeleton_landmark_pixel_projection_missing_or_invalid:{index}"
            )
    if image_space_landmark_count < 1:
        blockers.append("fresh_action_skeleton_no_in_frame_pixel_projections")
    return blockers


def _numeric_state_values(value: Any) -> tuple[list[float], bool]:
    if isinstance(value, bool) or value is None:
        return [], False
    if isinstance(value, (int, float)):
        number = _finite_float(value)
        return ([number] if number is not None else []), number is None
    if isinstance(value, Mapping):
        values: list[float] = []
        invalid = False
        for child in value.values():
            child_values, child_invalid = _numeric_state_values(child)
            values.extend(child_values)
            invalid = invalid or child_invalid
        return values, invalid
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values: list[float] = []
        invalid = False
        for child in value:
            child_values, child_invalid = _numeric_state_values(child)
            values.extend(child_values)
            invalid = invalid or child_invalid
        return values, invalid
    return [], True


def _generated_state_evidence_blockers(generated_state: Mapping[str, Any]) -> list[str]:
    state_fields = (
        "joint_positions",
        "joint_velocities",
        "joint_state",
        "proprioception",
        "state_vector",
        "robot_state_vector",
        "unitree_g1_sonic_state",
    )
    evidence_present = False
    blockers: list[str] = []
    for field in state_fields:
        if field not in generated_state:
            continue
        values, invalid = _numeric_state_values(generated_state.get(field))
        if invalid or not values:
            blockers.append(f"generated_robot_state_{field}_nonfinite_or_empty")
        else:
            evidence_present = True
    if not evidence_present:
        blockers.append("generated_robot_state_numeric_evidence_missing")
    state_payload = {key: value for key, value in generated_state.items() if key != "state_sha256"}
    state_sha256 = _string(generated_state.get("state_sha256")).strip().lower()
    if not _is_sha256(state_sha256):
        blockers.append("generated_robot_state_sha256_missing_or_invalid")
    elif state_sha256 != _canonical_sha256(state_payload):
        blockers.append("generated_robot_state_sha256_mismatch")
    return blockers


def _with_action_conditioning_digests(projection: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(projection)
    landmarks = [
        dict(row) for row in normalized.get("landmarks", []) or [] if isinstance(row, Mapping)
    ]
    normalized["landmarks"] = landmarks
    normalized["landmarks_sha256"] = _canonical_sha256(landmarks)
    generated_state = _mapping(normalized.get("generated_robot_state"))
    if generated_state:
        generated_state = {
            key: value for key, value in generated_state.items() if key != "state_sha256"
        }
        generated_state["state_sha256"] = _canonical_sha256(generated_state)
        normalized["generated_robot_state"] = generated_state
    normalized["action_conditioning_sha256"] = _canonical_sha256(
        {
            "source_action_sha256": normalized.get("source_action_sha256"),
            "landmarks_sha256": normalized.get("landmarks_sha256"),
            "generated_state_sha256": generated_state.get("state_sha256")
            if generated_state
            else None,
        }
    )
    return normalized


def _conditioning_evidence_sha256(wam_output: Mapping[str, Any]) -> str:
    conditioning = _mapping(wam_output.get("skeleton_conditioning"))
    state = _mapping(wam_output.get("generated_robot_state"))
    evidence_state = {
        key: value
        for key, value in state.items()
        if key not in {"source_action_sha256", "state_sha256"}
    }
    return _canonical_sha256(
        {
            "landmarks": conditioning.get("landmarks"),
            "generated_robot_state": evidence_state,
        }
    )


def _action_conditioning_blockers(
    *,
    action: Mapping[str, Any],
    wam_output: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if action.get("not_a_learned_robot_policy_action") is True:
        blockers.append("not_a_learned_robot_policy_action")
    if action.get("out_of_distribution_action_projection") is True:
        blockers.append("surrogate_policy_action_projection_not_allowed")
    action_sha256 = _canonical_sha256(action)
    conditioning = _mapping(wam_output.get("skeleton_conditioning"))
    landmarks = [
        dict(row) for row in conditioning.get("landmarks", []) or [] if isinstance(row, Mapping)
    ]
    blockers.extend(_landmark_evidence_blockers(landmarks))
    landmarks_sha256 = _string(conditioning.get("landmarks_sha256")).strip().lower()
    if not _is_sha256(landmarks_sha256):
        blockers.append("fresh_action_skeleton_landmarks_sha256_missing_or_invalid")
    elif landmarks_sha256 != _canonical_sha256(landmarks):
        blockers.append("fresh_action_skeleton_landmarks_sha256_mismatch")
    if conditioning.get("derived_via_controller_fk") is not True:
        blockers.append("fresh_action_skeleton_not_derived_via_controller_fk")
    if _string(conditioning.get("source_action_sha256")).strip() != action_sha256:
        blockers.append("fresh_action_skeleton_identity_mismatch")
    for required_key in ("controller_id", "controller_sha256", "robot_model_sha256"):
        value = _string(conditioning.get(required_key)).strip().lower()
        if not value:
            blockers.append(f"fresh_action_skeleton_{required_key}_missing")
        elif required_key.endswith("sha256") and not _is_sha256(value):
            blockers.append(f"fresh_action_skeleton_{required_key}_invalid")
    generated_state = _mapping(wam_output.get("generated_robot_state"))
    if not generated_state:
        blockers.append("generated_robot_state_missing")
    elif _string(generated_state.get("source_action_sha256")).strip() != action_sha256:
        blockers.append("generated_robot_state_action_identity_mismatch")
    elif generated_state.get("proxy_or_surrogate") is not False:
        blockers.append("generated_robot_state_proxy_not_allowed")
    else:
        blockers.extend(_generated_state_evidence_blockers(generated_state))
    expected_conditioning_sha256 = _canonical_sha256(
        {
            "source_action_sha256": action_sha256,
            "landmarks_sha256": landmarks_sha256,
            "generated_state_sha256": generated_state.get("state_sha256"),
        }
    )
    if _string(conditioning.get("action_conditioning_sha256")).strip().lower() != (
        expected_conditioning_sha256
    ):
        blockers.append("fresh_action_conditioning_digest_mismatch")
    return blockers


def _registered_task_criteria(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("registered_criteria", "success_criteria", "criteria"):
        value = contract.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [dict(row) for row in value if isinstance(row, Mapping)]
    return [dict(contract)] if _string(contract.get("criterion_id")).strip() else []


def _validate_hashed_evidence_artifacts(value: Any) -> tuple[list[dict[str, Any]], list[str]]:
    rows = (
        [dict(row) for row in value if isinstance(row, Mapping)]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
        else []
    )
    blockers: list[str] = []
    if not rows:
        return [], ["task_transition_hashed_evidence_artifacts_missing"]
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        path_text = _string(row.get("path")).strip()
        digest = _string(row.get("sha256")).strip().lower()
        if not path_text:
            blockers.append(f"task_transition_evidence_path_missing:{index}")
            continue
        path = Path(path_text).expanduser()
        if not path.is_file():
            blockers.append(f"task_transition_evidence_file_missing:{index}")
            continue
        if not _is_sha256(digest):
            blockers.append(f"task_transition_evidence_sha256_invalid:{index}")
            continue
        if _file_sha256(path) != digest:
            blockers.append(f"task_transition_evidence_sha256_mismatch:{index}")
            continue
        validated.append({**row, "path": str(path.resolve()), "sha256": digest})
    return validated, blockers


TASK_TRANSITION_MEASUREMENT_SCHEMA_VERSION = "task_transition_measurement.v1"


def _validate_transition_measurement_artifacts(
    value: Any,
    *,
    criterion_id: str,
    observable_transition: str,
    before_value: float | None,
    after_value: float | None,
    unit: str,
    source_step_index: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate that hashed evidence contains the exact measured transition.

    File presence and a matching digest prove only artifact integrity.  Task
    completion additionally requires a typed JSON measurement whose identity,
    values, unit, and source loop step exactly match the evaluator result.
    """

    artifacts, blockers = _validate_hashed_evidence_artifacts(value)
    bound: list[dict[str, Any]] = []
    for index, artifact in enumerate(artifacts):
        path = Path(_string(artifact.get("path"))).expanduser()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append(f"task_transition_measurement_json_invalid:{index}")
            continue
        if not isinstance(payload, Mapping):
            blockers.append(f"task_transition_measurement_not_object:{index}")
            continue
        measurement = dict(payload)
        if measurement.get("schema_version") != TASK_TRANSITION_MEASUREMENT_SCHEMA_VERSION:
            blockers.append(f"task_transition_measurement_schema_invalid:{index}")
        expected_strings = {
            "criterion_id": criterion_id,
            "observable_transition": observable_transition,
            "unit": unit,
        }
        for field, expected in expected_strings.items():
            if not expected or _string(measurement.get(field)).strip() != expected:
                blockers.append(f"task_transition_measurement_binding_mismatch:{field}:{index}")
        for field, expected in (
            ("before_value", before_value),
            ("after_value", after_value),
        ):
            observed = _finite_float(measurement.get(field))
            if expected is None or observed is None or abs(observed - expected) > 1e-12:
                blockers.append(f"task_transition_measurement_binding_mismatch:{field}:{index}")
        observed_step = measurement.get("source_step_index")
        if (
            source_step_index is None
            or isinstance(observed_step, bool)
            or not isinstance(observed_step, int)
            or observed_step != source_step_index
        ):
            blockers.append(
                f"task_transition_measurement_binding_mismatch:source_step_index:{index}"
            )
        artifact_blockers = [blocker for blocker in blockers if blocker.endswith(f":{index}")]
        if not artifact_blockers:
            bound.append({**artifact, "measurement": measurement})
    if artifacts and not bound:
        blockers.append("task_transition_measurement_artifact_not_bound")
    return bound, sorted(set(blockers))


def _computed_transition_passed(
    *,
    comparison: str,
    before_value: float,
    after_value: float,
    tolerance: float,
    target_value: float | None,
    episode_initial_value: float | None,
) -> bool | None:
    if episode_initial_value is None:
        return None
    from .task_episode_baseline import evaluate_task_criterion

    try:
        evaluation = evaluate_task_criterion(
            {"comparison": comparison, "tolerance": tolerance, "target_value": target_value},
            episode_initial_value=episode_initial_value,
            step_before=before_value,
            step_after=after_value,
        )
    except ValueError:
        return None
    return bool(evaluation["passed"])


def _task_completion_attested_payload(
    result: Mapping[str, Any],
    *,
    validator_derived_fields: set[str],
) -> dict[str, Any]:
    """Recover the exact task-completion payload covered by the live signature.

    ``comparison`` and ``articulation_prim_path`` can originate in either place:
    the persistent Isaac service supplies and signs both for live measurements,
    while older/fixture evaluators omit them and the first validation pass derives
    them from the registered criterion.  A proof leaf from either source contains
    ``registered_transition_passed``, so presence of that field alone cannot tell
    the second-pass validator whether the two ambiguous fields were signed.

    Try the bounded four possible signed-field combinations and accept only the
    one whose canonical digest is already committed by the Ed25519 attestation.
    The subsequent attestation validator still verifies the signature itself.
    """

    payload = {
        key: value
        for key, value in result.items()
        if key != "evaluator_attestation" and key not in validator_derived_fields
    }
    ambiguous_fields = tuple(
        field for field in ("comparison", "articulation_prim_path") if field in payload
    )
    committed_sha256 = _string(
        _mapping(result.get("evaluator_attestation")).get("signed_payload_sha256")
    ).lower()
    if re.fullmatch(r"[0-9a-f]{64}", committed_sha256):
        for exclusion_mask in range(1 << len(ambiguous_fields)):
            candidate = dict(payload)
            for index, field in enumerate(ambiguous_fields):
                if exclusion_mask & (1 << index):
                    candidate.pop(field, None)
            if _canonical_sha256(candidate) == committed_sha256:
                return candidate

    # Preserve the historical fail-closed fallback when the claimed digest is
    # malformed or matches none of the bounded candidates.  Signature validation
    # will reject that payload; this branch must never manufacture a passing hash.
    if "registered_transition_passed" in result:
        for field in ambiguous_fields:
            payload.pop(field, None)
    return payload


def _validate_task_completion_transition(
    *,
    completion_result: Mapping[str, Any],
    task_success_contract: Mapping[str, Any],
    expected_source_step_index: int | None = None,
) -> dict[str, Any]:
    result = dict(completion_result)
    blockers: list[str] = []
    # A validated transition is persisted as a proof leaf and can be validated
    # again by the semantic judge or host collector.  Do not accidentally feed
    # validator-derived fields back into the signed payload on that second
    # pass: the persistent Isaac service signed the original result before
    # these fields existed.
    validator_derived_fields = {
        "registered_criterion",
        "target_value",
        "computed_transition_passed",
        "reported_transition_passed",
        "validated_evidence_artifacts",
        "evaluator_attestation_validation",
        "registered_transition_passed",
        "validation_blockers",
    }
    evaluator_attestation_validation = validate_trusted_ed25519_attestation(
        _mapping(result.get("evaluator_attestation")),
        signed_payload=_task_completion_attested_payload(
            result,
            validator_derived_fields=validator_derived_fields,
        ),
        prefix="task_transition_evaluator_attestation",
        trusted_public_key_sha256_env=(SC3_TASK_COMPLETION_TRUSTED_PUBLIC_KEY_SHA256_ENV),
    )
    blockers.extend(_string_list(evaluator_attestation_validation.get("blockers")))
    criterion_id = _string(result.get("criterion_id")).strip()
    registered = {
        _string(row.get("criterion_id")).strip(): dict(row)
        for row in _registered_task_criteria(task_success_contract)
        if _string(row.get("criterion_id")).strip()
    }
    criterion = _mapping(registered.get(criterion_id))
    if not criterion_id or not criterion:
        blockers.append("task_transition_criterion_not_registered")
    observable_transition = _string(result.get("observable_transition")).strip()
    registered_transition = _string(criterion.get("observable_transition")).strip()
    if not registered_transition or observable_transition != registered_transition:
        blockers.append("task_transition_observable_transition_not_registered")
    comparison = _string(criterion.get("comparison")).strip()
    if comparison not in {
        "increase_at_least",
        "decrease_at_least",
        "absolute_change_at_least",
        "within_tolerance",
        "at_or_above",
        "at_or_below",
    }:
        blockers.append("task_transition_comparison_missing_or_unsupported")
    tolerance = _finite_float(result.get("tolerance"))
    registered_tolerance = _finite_float(criterion.get("tolerance"))
    if (
        tolerance is None
        or registered_tolerance is None
        or tolerance < 0.0
        or abs(tolerance - registered_tolerance) > 1e-12
    ):
        blockers.append("task_transition_tolerance_missing_nonfinite_or_unregistered")
    if comparison in {
        "increase_at_least",
        "decrease_at_least",
        "absolute_change_at_least",
    } and (tolerance is None or tolerance <= 0.0):
        blockers.append("task_transition_change_tolerance_must_be_positive")
    before_value = _finite_float(result.get("before_value"))
    after_value = _finite_float(result.get("after_value"))
    if before_value is None or after_value is None:
        blockers.append("task_transition_before_after_values_missing_or_nonfinite")
    unit = _string(result.get("unit")).strip()
    registered_unit = _string(criterion.get("unit")).strip()
    if not unit or not registered_unit or unit != registered_unit:
        blockers.append("task_transition_unit_missing_or_unregistered")
    articulation_prim_path = _string(result.get("articulation_prim_path")).strip()
    registered_prim_path = _string(criterion.get("articulation_prim_path")).strip()
    prim_resolution = _mapping(criterion.get("articulation_prim_path_resolution"))
    if registered_prim_path and articulation_prim_path != registered_prim_path:
        blockers.append("task_transition_articulation_prim_path_mismatch")
    if prim_resolution:
        root_term = _string(prim_resolution.get("required_target_root")).lower()
        affordance_terms = [
            _string(item).lower()
            for item in prim_resolution.get("required_affordance_terms") or []
            if _string(item).strip()
        ]
        path_lower = articulation_prim_path.lower()
        if (
            not articulation_prim_path.startswith("/")
            or (root_term and root_term not in path_lower)
            or (affordance_terms and not any(term in path_lower for term in affordance_terms))
            or result.get("attempt_input_manifest_sha256") is None
        ):
            blockers.append("task_transition_attempt_bound_articulation_prim_path_not_resolved")
    source_step_value = result.get("source_step_index")
    source_step_index = (
        source_step_value
        if isinstance(source_step_value, int) and not isinstance(source_step_value, bool)
        else None
    )
    if source_step_index is None:
        blockers.append("task_transition_source_step_index_missing_or_invalid")
    elif expected_source_step_index is not None and source_step_index != expected_source_step_index:
        blockers.append("task_transition_source_step_index_mismatch")
    target_value = _finite_float(
        result.get("target_value")
        if result.get("target_value") is not None
        else criterion.get("target_value")
    )
    evidence_artifacts, evidence_blockers = _validate_transition_measurement_artifacts(
        result.get("evidence_artifacts") or result.get("evidence_refs"),
        criterion_id=criterion_id,
        observable_transition=observable_transition,
        before_value=before_value,
        after_value=after_value,
        unit=unit,
        source_step_index=source_step_index,
    )
    blockers.extend(evidence_blockers)
    episode_initial_value = _finite_float(result.get("episode_initial_value"))
    if episode_initial_value is None:
        blockers.append("task_transition_episode_initial_value_missing_or_nonfinite")
    computed_passed = (
        _computed_transition_passed(
            comparison=comparison,
            before_value=before_value,
            after_value=after_value,
            tolerance=tolerance,
            target_value=target_value,
            episode_initial_value=episode_initial_value,
        )
        if comparison
        and before_value is not None
        and after_value is not None
        and tolerance is not None
        else None
    )
    reported_passed = result.get("passed")
    if not isinstance(reported_passed, bool):
        blockers.append("task_transition_passed_not_strict_boolean")
    elif computed_passed is None or reported_passed is not computed_passed:
        blockers.append("task_transition_reported_verdict_mismatch")
    if result.get("status") != "completed":
        blockers.append("task_transition_evaluator_not_completed")
    blockers = sorted(set(blockers))
    return {
        **result,
        "registered_criterion": criterion or None,
        "comparison": comparison or None,
        "before_value": before_value,
        "after_value": after_value,
        "target_value": target_value,
        "tolerance": tolerance,
        "unit": unit or None,
        "articulation_prim_path": articulation_prim_path or None,
        "source_step_index": source_step_index,
        "computed_transition_passed": computed_passed,
        "reported_transition_passed": reported_passed
        if isinstance(reported_passed, bool)
        else None,
        "validated_evidence_artifacts": evidence_artifacts,
        "evaluator_attestation_validation": evaluator_attestation_validation,
        "registered_transition_passed": bool(
            not blockers and computed_passed is True and reported_passed is True
        ),
        "validation_blockers": blockers,
    }


def _callable_label(value: Any) -> str:
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if module and name:
        return f"{module}.{name}"
    return type(value).__name__


def _xyz_list(value: Any, fallback: Sequence[float] | None = None) -> list[float]:
    source: Sequence[Any] | None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        source = value
    elif fallback is not None and len(fallback) >= 3:
        source = fallback
    else:
        source = (0.0, 0.0, 0.793)
    return [round(float(source[0]), 6), round(float(source[1]), 6), round(float(source[2]), 6)]


def _endpoint_action_signature(action: Mapping[str, Any]) -> str:
    root = action.get("root_position") or action.get("root_pose")
    yaw = action.get("root_yaw_radians", action.get("yaw"))
    signature = {
        "root_position": _xyz_list(root),
        "root_yaw_radians": yaw,
        "policy_action": _string(action.get("policy_action") or action.get("motion_token")),
        "joint_targets": action.get("joint_targets"),
        "action_chunk": action.get("action_chunk"),
        "sonic_action_chunk": action.get("sonic_action_chunk"),
        "controller_action": action.get("controller_action"),
    }
    return json.dumps(signature, sort_keys=True, default=str)


def _action_record_from_policy_endpoint(
    *,
    base_action: Mapping[str, Any],
    endpoint_action: Mapping[str, Any],
    requery_source_step_index: int,
    source_observation_kind: str = "wam_generated_observation",
) -> dict[str, Any]:
    action = dict(base_action)
    root = _xyz_list(
        endpoint_action.get("root_position") or endpoint_action.get("root_pose"),
        action.get("root_position"),
    )
    action["root_position"] = root
    action["desired_root_position"] = _xyz_list(
        endpoint_action.get("desired_root_position"),
        root,
    )
    yaw = endpoint_action.get("root_yaw_radians", endpoint_action.get("yaw"))
    if yaw is not None:
        action["root_yaw_radians"] = round(float(yaw), 6)
    action["policy_action"] = _string(
        endpoint_action.get("policy_action")
        or endpoint_action.get("motion_token")
        or "learned_policy_action"
    )
    if "joint_targets" in endpoint_action:
        action["joint_targets"] = endpoint_action.get("joint_targets")
    for key in (
        "action_chunk",
        "sonic_action_chunk",
        "action_units",
        "action_timing",
        "action_horizon",
        "controller_action",
        "not_a_learned_robot_policy_action",
        "out_of_distribution_action_projection",
    ):
        if key in endpoint_action:
            action[key] = endpoint_action.get(key)
    action["learned_policy_endpoint_action"] = dict(endpoint_action)
    action["policy_requeried_on_generated_observation"] = (
        source_observation_kind == "wam_generated_observation"
    )
    action["policy_requeried_on_initial_real_observation"] = (
        source_observation_kind == "initial_real_observation"
    )
    action["policy_requeried_fresh"] = True
    action["policy_requery_source_step_index"] = int(requery_source_step_index)
    action["policy_action_source"] = f"policy_endpoint_requery_on_{source_observation_kind}"
    return action


def make_learned_policy_command_endpoint(
    *,
    command: str,
    work_dir: str | Path,
    timeout_seconds: float = 120.0,
) -> PolicyEndpoint:
    """Expose a real learned-policy process boundary to the strict CLI lane."""

    argv = shlex.split(_string(command).strip())
    if not argv:
        raise ValueError("learned_policy_command_missing")
    root = Path(work_dir).expanduser().resolve()
    ensure_dir(root)
    seen_runtime_result_ids: set[str] = set()

    def endpoint(
        observation: Mapping[str, Any],
        action_history: Sequence[Mapping[str, Any]],
        step_index: int,
    ) -> Mapping[str, Any]:
        step_dir = root / f"step_{int(step_index):04d}"
        ensure_dir(step_dir)
        request_path = step_dir / "learned_policy_request.json"
        output_path = step_dir / "learned_policy_output.json"
        if output_path.exists():
            output_path.unlink()
        request = {
            "schema_version": "oscar_learned_policy_endpoint_request.v1",
            "step_index": int(step_index),
            "observation": dict(observation),
            "action_history": [dict(row) for row in action_history],
        }
        write_json(request_path, request)
        request_sha256 = _canonical_sha256(request)
        completed = subprocess.run(
            argv,
            input=json.dumps(request, sort_keys=True),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1.0, float(timeout_seconds)),
            cwd=str(step_dir),
            env={
                **os.environ,
                "BLUEPRINT_LEARNED_POLICY_INPUT": str(request_path),
                "BLUEPRINT_LEARNED_POLICY_OUTPUT": str(output_path),
                "BLUEPRINT_LEARNED_POLICY_STEP_INDEX": str(int(step_index)),
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(f"learned_policy_command_returncode_{completed.returncode}")
        payload: Any = None
        if output_path.is_file():
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        elif completed.stdout.strip():
            payload = json.loads(completed.stdout)
        response = _mapping(payload)
        if response.get("status") != "completed":
            raise RuntimeError("learned_policy_command_status_not_completed")
        if response.get("learned_policy_action_proven") is not True:
            raise RuntimeError("learned_policy_command_action_not_proven")
        action = _mapping(response.get("action") or response.get("policy_action"))
        if not action:
            raise RuntimeError("learned_policy_command_action_missing")
        if action.get("not_a_learned_robot_policy_action") is not False:
            raise RuntimeError("learned_policy_command_proxy_action_rejected")
        if action.get("out_of_distribution_action_projection") is not False:
            raise RuntimeError("learned_policy_command_surrogate_projection_rejected")
        if not _is_sha256(response.get("checkpoint_sha256")):
            raise RuntimeError("learned_policy_command_checkpoint_sha256_invalid")
        if not _is_sha256(response.get("model_code_sha256")):
            raise RuntimeError("learned_policy_command_model_code_sha256_invalid")
        action_chunk = action.get("action_chunk")
        if not (
            isinstance(action_chunk, Sequence)
            and not isinstance(action_chunk, (str, bytes, bytearray))
            and len(action_chunk) == 7
            and all(_finite_float(value) is not None for value in action_chunk)
        ):
            raise RuntimeError("learned_policy_command_action_chunk_not_finite_7d")
        runtime_result_id = _string(response.get("runtime_result_id")).strip()
        if not runtime_result_id or runtime_result_id in seen_runtime_result_ids:
            raise RuntimeError("learned_policy_command_runtime_result_id_missing_or_replayed")
        endpoint_id = _string(response.get("policy_endpoint_id")).strip()
        if not endpoint_id:
            raise RuntimeError("learned_policy_command_endpoint_id_missing")
        checkpoint_ref = _mapping(response.get("checkpoint_artifact"))
        model_code_ref = _mapping(response.get("model_code_artifact"))
        _, checkpoint_blockers = _validate_hashed_evidence_artifacts([checkpoint_ref])
        _, model_code_blockers = _validate_hashed_evidence_artifacts([model_code_ref])
        if (
            checkpoint_blockers
            or _string(checkpoint_ref.get("sha256")).lower()
            != _string(response.get("checkpoint_sha256")).lower()
        ):
            raise RuntimeError("learned_policy_command_checkpoint_artifact_invalid")
        if (
            model_code_blockers
            or _string(model_code_ref.get("sha256")).lower()
            != _string(response.get("model_code_sha256")).lower()
        ):
            raise RuntimeError("learned_policy_command_model_code_artifact_invalid")
        signed_result = {
            "schema_version": "sc3_learned_policy_runtime_result.v1",
            "request_sha256": request_sha256,
            "step_index": int(step_index),
            "policy_endpoint_id": endpoint_id,
            "runtime_result_id": runtime_result_id,
            "checkpoint_sha256": _string(response.get("checkpoint_sha256")).lower(),
            "model_code_sha256": _string(response.get("model_code_sha256")).lower(),
            "checkpoint_artifact": checkpoint_ref,
            "model_code_artifact": model_code_ref,
            "action": action,
        }
        attestation = validate_trusted_ed25519_attestation(
            _mapping(response.get("runtime_attestation")),
            signed_payload=signed_result,
            prefix="learned_policy_runtime_attestation",
            trusted_public_key_sha256_env=(
                SC3_LEARNED_POLICY_RUNTIME_TRUSTED_PUBLIC_KEY_SHA256_ENV
            ),
        )
        if attestation.get("status") != "validated":
            raise RuntimeError(
                "learned_policy_command_runtime_attestation_invalid:"
                + ",".join(_string_list(attestation.get("blockers")))
            )
        seen_runtime_result_ids.add(runtime_result_id)
        return {
            **action,
            "learned_policy_endpoint_id": endpoint_id,
            "learned_policy_runtime_result_id": runtime_result_id,
            "learned_policy_request_sha256": request_sha256,
            "learned_policy_checkpoint_sha256": _string(response.get("checkpoint_sha256")),
            "learned_policy_model_code_sha256": _string(response.get("model_code_sha256")),
        }

    return endpoint


def make_task_completion_command_evaluator(
    *,
    command: str,
    work_dir: str | Path,
    timeout_seconds: float = 120.0,
) -> TaskCompletionEvaluator:
    """Expose a per-step task-transition measurement process to the CLI lane."""

    argv = shlex.split(_string(command).strip())
    if not argv:
        raise ValueError("task_completion_command_missing")
    root = Path(work_dir).expanduser().resolve()
    ensure_dir(root)
    persistent_simulator_session_id: str | None = None
    seen_runtime_result_ids: set[str] = set()

    def evaluator(context: Mapping[str, Any]) -> Mapping[str, Any]:
        nonlocal persistent_simulator_session_id
        step_value = context.get("step_index")
        if isinstance(step_value, bool) or not isinstance(step_value, int):
            raise ValueError("task_completion_step_index_missing_or_invalid")
        step_dir = root / f"step_{step_value:04d}"
        ensure_dir(step_dir)
        request_path = step_dir / "task_completion_request.json"
        output_path = step_dir / "task_completion_output.json"
        if output_path.exists():
            output_path.unlink()
        request = {
            "schema_version": "oscar_task_completion_evaluator_request.v1",
            **dict(context),
        }
        write_json(request_path, request)
        completed = subprocess.run(
            argv,
            input=json.dumps(request, sort_keys=True),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1.0, float(timeout_seconds)),
            cwd=str(step_dir),
            env={
                **os.environ,
                "BLUEPRINT_TASK_COMPLETION_INPUT": str(request_path),
                "BLUEPRINT_TASK_COMPLETION_OUTPUT": str(output_path),
                "BLUEPRINT_TASK_COMPLETION_STEP_INDEX": str(step_value),
            },
        )
        stdout_path = step_dir / "task_completion_stdout.log"
        stderr_path = step_dir / "task_completion_stderr.log"
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        write_json(
            step_dir / "task_completion_command_result.json",
            {
                "schema_version": "task_completion_command_result.v1",
                "status": "completed" if completed.returncode == 0 else "blocked",
                "returncode": int(completed.returncode),
                "stdout_log": str(stdout_path.resolve()),
                "stderr_log": str(stderr_path.resolve()),
                "output_path": str(output_path),
                "output_present": output_path.is_file(),
                "timeout_seconds": max(1.0, float(timeout_seconds)),
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(f"task_completion_command_returncode_{completed.returncode}")
        payload: Any = None
        if output_path.is_file():
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        elif completed.stdout.strip():
            payload = json.loads(completed.stdout)
        response = _mapping(payload)
        if not response:
            raise RuntimeError("task_completion_command_output_missing_or_invalid")
        expected_action_sha256 = _canonical_sha256(_mapping(context.get("action")))
        if _string(response.get("source_action_sha256")).strip() != expected_action_sha256:
            raise RuntimeError("task_completion_action_sha256_mismatch")
        session_id = _string(response.get("simulator_session_id")).strip()
        if not session_id:
            raise RuntimeError("task_completion_simulator_session_id_missing")
        if persistent_simulator_session_id is None:
            persistent_simulator_session_id = session_id
        elif session_id != persistent_simulator_session_id:
            raise RuntimeError("task_completion_simulator_session_changed")
        runtime_result_id = _string(response.get("runtime_result_id")).strip()
        if not runtime_result_id or runtime_result_id in seen_runtime_result_ids:
            raise RuntimeError("task_completion_runtime_result_id_missing_or_replayed")
        seen_runtime_result_ids.add(runtime_result_id)
        if response.get("persistent_simulator_state_applied") is not True:
            raise RuntimeError("task_completion_persistent_simulator_state_not_applied")
        if response.get("official_controller_action_applied") is not True:
            raise RuntimeError("task_completion_official_controller_action_not_applied")
        if _string(response.get("simulator_backend")).lower() != "isaac":
            raise RuntimeError("task_completion_simulator_backend_not_isaac")
        if not _string(response.get("stage_id")).strip():
            raise RuntimeError("task_completion_stage_id_missing")
        if not _string(response.get("articulation_prim_path")).startswith("/"):
            raise RuntimeError("task_completion_articulation_prim_path_invalid")
        before_timestamp = _string(response.get("before_timestamp")).strip()
        after_timestamp = _string(response.get("after_timestamp")).strip()
        if not before_timestamp or not after_timestamp or before_timestamp == after_timestamp:
            raise RuntimeError("task_completion_before_after_timestamps_invalid")
        response["post_action_policy_state"] = _validated_post_action_policy_state(
            response.get("post_action_policy_state"),
            simulator_session_id=session_id,
            stage_id=_string(response.get("stage_id")).strip(),
            source_action_sha256=expected_action_sha256,
            source_step_index=step_value,
        )
        post_action_egocentric_frame = _validated_post_action_egocentric_frame(
            response,
            source_step_index=step_value,
        )
        if post_action_egocentric_frame:
            response["post_action_egocentric_frame"] = post_action_egocentric_frame
        return response

    return evaluator


def _neutral_unitree_g1_sonic_state() -> dict[str, list[float]]:
    state = {key: [0.0] * int(dim) for key, dim in UNITREE_G1_SONIC_STATE_DIMS.items()}
    state["projected_gravity"] = [0.0, 0.0, -1.0]
    return state


def _policy_observation(
    frame_path: str,
    target: Sequence[float],
    step_index: int,
    *,
    task_prompt: str | None = None,
) -> dict[str, Any]:
    """The policy observation handed to the harness for this step (evaluator-controlled state
    kept separate from the WAM's pixel-inferred fields)."""
    return {
        "schema_version": "oscar_isaac_closed_loop_observation.v1",
        "step_index": step_index,
        "camera_role": "robot_pov",
        "viewpoint_mode": "robot_head_mounted_egocentric",
        "policy_observation_eligible": True,
        "third_person_overview_included": False,
        "visual_observation": {
            "generated_frame_path": _string(frame_path),
            "camera_frame_path": _string(frame_path),
            "camera_role": "robot_pov",
            "viewpoint_mode": "robot_head_mounted_egocentric",
            "policy_observation_eligible": True,
            "third_person_overview_included": False,
            "wam_viewpoint_inherits_source_robot_head_pov": True,
        },
        "task_target_position_xyz": [round(float(c), 6) for c in target],
        "task_prompt": _string(task_prompt).strip() or None,
    }


def build_oscar_per_step_request(
    *,
    current_frame_path: str,
    action: Mapping[str, Any],
    step_index: int,
    task_prompt: str,
    num_frames: int,
    output_dir: str | Path,
    skeleton_landmarks: Sequence[Mapping[str, Any]] | None = None,
    skeleton_trace_rows: Sequence[Mapping[str, Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Shape one per-step OSCAR-2B next-observation generation request.

    OSCAR generates a short clip forward from the current observation (``current_frame_path``),
    conditioned on the task prompt and the projected G1 skeleton for this step's action. The
    NEXT observation is selected from usable future frames of that clip. This is pure request
    shaping with no GPU or OSCAR import, so it is fully unit-testable; the actual inference is the
    injected callable in :func:`make_oscar_per_step_wam_backend`.
    """
    return {
        "schema_version": "oscar_per_step_generation_request.v1",
        "step_index": int(step_index),
        "reference_frame_path": _string(current_frame_path),
        "task_prompt": _string(task_prompt),
        "num_frames": max(1, int(num_frames)),
        "seed": int(seed) + int(step_index),
        "output_dir": str(Path(output_dir).expanduser() / f"oscar_step_{step_index:04d}"),
        "policy_action": dict(action),
        "root_position": list(action.get("root_position") or []),
        "root_yaw_radians": action.get("root_yaw_radians"),
        "projected_landmark_count": len(skeleton_landmarks or []),
        "skeleton_landmarks": [dict(landmark) for landmark in (skeleton_landmarks or [])],
        "skeleton_trace_row_count": len(skeleton_trace_rows or []),
        "skeleton_trace_rows": [dict(row) for row in (skeleton_trace_rows or [])],
    }


def build_wam_generation_step_input(
    *,
    current_frame_path: str | Path,
    action: Mapping[str, Any],
    step_index: int,
    output_dir: str | Path,
    task_prompt: str,
    next_observation_frame_path: str | Path | None = None,
    target_object_id: str = "task_target",
    projected_skeleton_trace_path: str | Path | None = None,
    rgb_context_frame_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Build the provider-bundle input for one per-step OSCAR WAM call."""
    frame = Path(current_frame_path).expanduser().resolve()
    visual = {
        "camera_id": "head_pov",
        "camera_frame_path": str(frame),
        "wam_generated_observation": step_index > 1,
    }
    if projected_skeleton_trace_path:
        trace_path = Path(projected_skeleton_trace_path).expanduser().resolve()
        visual["g1_projected_skeleton_trace_jsonl"] = str(trace_path)
        visual["projected_skeleton_trace_path"] = str(trace_path)
    out = Path(output_dir).expanduser().resolve()
    requested_next = (
        Path(next_observation_frame_path).expanduser()
        if next_observation_frame_path
        else out / "generated_next_observation.png"
    )
    payload = {
        "schema_version": "wam_generation_step_input.v1",
        "step_index": int(step_index),
        "source_policy_observation_frame_path": str(frame),
        "source_policy_action": {
            **dict(action),
            "task_prompt": _string(task_prompt),
            "action_type": _string(action.get("action_type"))
            or _string(action.get("policy_action"))
            or "isaac_g1_policy_action",
        },
        "current_policy_observation": {
            "schema_version": "blueprint_policy_observation.v1",
            "task_id": "isaac_g1_oscar_per_step_closed_loop",
            "task_prompt": _string(task_prompt),
            "target_object_id": target_object_id,
            "robot_profile_id": "unitree_g1",
            "policy_source": "isaac_g1_policy",
            "camera_frame_path": str(frame),
            "visual_observation": visual,
            "unitree_g1_sonic_state": _neutral_unitree_g1_sonic_state(),
            "unitree_g1_sonic_state_source": "neutral_unitree_g1_sonic_contract_state",
            "unitree_g1_sonic_state_metadata": {
                "complete": True,
                "robot_profile_id": "unitree_g1",
                "state_vector_dims": dict(UNITREE_G1_SONIC_STATE_DIMS),
                "neutral_state_for_initial_sim_observation": True,
                "scene_or_task_specific_coordinates_hardcoded": False,
            },
            "claim_boundary": {
                "simulator_generated_world_observation_only": True,
                "generated_wam_frame_is_support_artifact": step_index > 1,
                "physical_robot_sensor_proof": False,
                "deployment_readiness_proven": False,
            },
        },
        "requested_output": {
            "next_observation_frame_path": str(requested_next),
            "action_conditioned_generation_required": True,
        },
        "claim_boundary": {
            "isaac_policy_action_is_sim_policy_action": True,
            "wam_generation_is_not_robot_policy": True,
            "physical_robot_sensor_proof": False,
        },
    }
    context_paths: list[str] = []
    seen_context_paths: set[str] = set()
    for value in rgb_context_frame_paths or []:
        text = _string(value)
        if not text:
            continue
        path = Path(text).expanduser()
        if not path.is_file():
            continue
        resolved = str(path.resolve())
        if resolved in seen_context_paths:
            continue
        seen_context_paths.add(resolved)
        context_paths.append(resolved)
    if context_paths:
        payload["rgb_context_frame_paths"] = context_paths
        payload["claim_boundary"]["rgb_context_frame_paths_are_real_observation_history"] = True
    return payload


@contextmanager
def _temporary_environ(updates: Mapping[str, str | None]) -> Iterator[None]:
    previous: dict[str, str | None] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _provider_video_path(payload: Mapping[str, Any]) -> str:
    for rollout in payload.get("rollouts", []) or []:
        if not isinstance(rollout, Mapping):
            continue
        video = _string(rollout.get("generated_video_path"))
        if video and Path(video).expanduser().is_file():
            return video
    return ""


def _provider_payload_proves_fresh_model(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("status") == "completed"
        and payload.get("fresh_provider_model_run_claimed")
        and payload.get("provider_learned_wam_model_ran")
        and payload.get("provider_generated_video_is_model_output")
    )


def _provider_payload_visual_acceptance_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    status = _string(payload.get("status"))
    if status and status != "completed":
        blockers.append(f"oscar_provider_status_{status}")
        blockers.extend(_string_list(payload.get("blockers")))
    visual_status = _string(payload.get("generated_rollout_visual_smoke_status"))
    if visual_status and visual_status != "passed_visual_quality_smoke":
        blockers.append("provider_generated_rollout_visual_smoke_not_passed")
        blockers.append(f"provider_generated_rollout_visual_smoke_status:{visual_status}")
        blockers.extend(_string_list(payload.get("generated_rollout_visual_quality_blockers")))
        visual_smoke = _mapping(payload.get("generated_rollout_visual_smoke"))
        blockers.extend(_string_list(visual_smoke.get("blockers")))
    return sorted(set(item for item in blockers if item))


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _git_config_value(path: str | Path | None, key: str) -> str:
    if not _string(path):
        return ""
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "-C", str(Path(_string(path)).expanduser()), "config", "--get", key],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return (completed.stdout or "").strip().splitlines()[-1] if completed.stdout.strip() else ""


def _git_head_commit(path: str | Path | None) -> str:
    if not _string(path):
        return ""
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "-C", str(Path(_string(path)).expanduser()), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return (completed.stdout or "").strip().splitlines()[-1] if completed.stdout.strip() else ""


def _checkpoint_revision_from_path(path: str | Path | None) -> str:
    if not _string(path):
        return ""
    hex_chars = set("0123456789abcdef")
    checkpoint = Path(_string(path)).expanduser()
    for item in (checkpoint, *checkpoint.parents):
        name = item.name.strip().lower()
        if len(name) == 40 and all(char in hex_chars for char in name):
            return name
    return ""


def _float_env(name: str, default: float) -> float:
    try:
        return float(_string(os.getenv(name)) or default)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(float(_string(os.getenv(name)) or default))
    except ValueError:
        return default


def _vast_session_budget_path() -> Path:
    explicit = _string(os.getenv("VAST_SESSION_BUDGET_LEDGER_FILE"))
    if explicit:
        return Path(explicit).expanduser()
    key_file = Path(
        _string(os.getenv(VAST_API_KEY_FILE_ENV)) or "~/.blueprint-secrets/vast_api_key"
    ).expanduser()
    return key_file.parent / "vast_session_cost_summary.json"


def _vast_paid_provider_preflight(
    *,
    allow_paid_provider_launch: bool,
    max_hourly_rate_usd: float,
    max_live_minutes: int,
    session_max_live_minutes: int,
    hard_cap_usd: float,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if not allow_paid_provider_launch:
        blockers.append("closed_loop_paid_provider_launch_not_authorized")
    if not _env_truthy(VAST_PAID_WAM_GATE_ENV):
        blockers.append(f"missing_env_{VAST_PAID_WAM_GATE_ENV}")
    if not _env_truthy(VAST_API_GATE_ENV):
        blockers.append(f"missing_env_{VAST_API_GATE_ENV}")
    if not _env_truthy(VAST_INSTANCE_LAUNCH_GATE_ENV):
        blockers.append(f"missing_env_{VAST_INSTANCE_LAUNCH_GATE_ENV}")
    key_file = Path(
        _string(os.getenv(VAST_API_KEY_FILE_ENV)) or "~/.blueprint-secrets/vast_api_key"
    ).expanduser()
    if not key_file.is_file():
        blockers.append(f"missing_file_based_secret_{VAST_API_KEY_FILE_ENV}")
    budget_path = _vast_session_budget_path()
    prior_cost = 0.0
    prior_live_seconds = 0.0
    budget_present = budget_path.is_file()
    budget_parse_error = None
    attempt_count = 0
    if budget_present:
        try:
            budget = json.loads(budget_path.read_text(encoding="utf-8"))
            attempts = budget.get("attempts")
            if isinstance(attempts, list):
                attempt_count = len(attempts)
                for row in attempts:
                    if not isinstance(row, Mapping):
                        continue
                    try:
                        prior_cost += float(row.get("estimated_cost_usd") or 0.0)
                    except (TypeError, ValueError):
                        pass
                    try:
                        prior_live_seconds += float(
                            row.get("actual_live_runtime_seconds_observed_by_adapter") or 0.0
                        )
                    except (TypeError, ValueError):
                        pass
        except Exception as exc:  # pragma: no cover - type surfaced in artifact
            budget_parse_error = type(exc).__name__
            blockers.append("vast_session_budget_ledger_parse_failed")
    projected_incremental_cost = max_hourly_rate_usd * (max_live_minutes / 60.0)
    prior_live_minutes = prior_live_seconds / 60.0
    if session_max_live_minutes >= 0 and prior_live_minutes >= float(session_max_live_minutes):
        blockers.append("session_live_runtime_limit_exhausted")
    if prior_cost + projected_incremental_cost > hard_cap_usd:
        blockers.append("session_estimated_spend_hard_cap_exhausted")
    elif prior_cost >= hard_cap_usd:
        blockers.append("session_estimated_spend_hard_cap_already_exceeded")
    if prior_cost > 0.0 and not budget_present:
        warnings.append("vast_budget_prior_cost_inferred_without_ledger")
    return {
        "schema_version": "closed_loop_vast_paid_provider_preflight.v1",
        "status": "ready" if not blockers else "blocked",
        "provider": "vast",
        "gate_env": {
            VAST_PAID_WAM_GATE_ENV: _env_truthy(VAST_PAID_WAM_GATE_ENV),
            VAST_API_GATE_ENV: _env_truthy(VAST_API_GATE_ENV),
            VAST_INSTANCE_LAUNCH_GATE_ENV: _env_truthy(VAST_INSTANCE_LAUNCH_GATE_ENV),
        },
        "vast_api_key_file_present": key_file.is_file(),
        "vast_api_key_file_path": str(key_file),
        "budget_path": str(budget_path),
        "budget_ledger_present": budget_present,
        "budget_parse_error": budget_parse_error,
        "attempt_count": attempt_count,
        "prior_estimated_cost_usd": round(prior_cost, 6),
        "prior_live_runtime_minutes": round(prior_live_minutes, 6),
        "max_hourly_rate_usd": float(max_hourly_rate_usd),
        "requested_max_live_runtime_minutes": int(max_live_minutes),
        "session_max_live_runtime_minutes": int(session_max_live_minutes),
        "projected_max_incremental_cost_usd": round(projected_incremental_cost, 6),
        "hard_cap_usd": float(hard_cap_usd),
        "blockers": sorted(set(blockers)),
        "warnings": warnings,
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "preflight_is_no_spend": True,
            "preflight_does_not_call_vast_api": True,
            "secret_values_not_read_into_artifact": True,
        },
    }


def _runpod_paid_provider_preflight(
    *,
    allow_paid_provider_launch: bool,
    max_hourly_rate_usd: float,
    max_live_minutes: int,
    hard_cap_usd: float,
) -> dict[str, Any]:
    """Cost-enveloped RunPod gate mirroring the Vast preflight's discipline.

    RunPod is the funded provider (Vast may carry no credits); a bounded OSCAR
    generation (~10 min on a <=$0.80/hr pod) must be launchable under the same
    no-spend preflight rules: explicit env gates, file-based secret, projected
    cost under a hard cap. This preflight makes NO provider API calls.
    """
    blockers: list[str] = []
    if not allow_paid_provider_launch:
        blockers.append("closed_loop_paid_provider_launch_not_authorized")
    if not _env_truthy("BLUEPRINT_ALLOW_RUNPOD_API_CALLS"):
        blockers.append("missing_env_BLUEPRINT_ALLOW_RUNPOD_API_CALLS")
    if not _env_truthy("BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH"):
        blockers.append("missing_env_BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH")
    key_file = Path(
        _string(os.getenv("RUNPOD_API_KEY_FILE")) or "~/.blueprint-secrets/runpod_api_key"
    ).expanduser()
    if not key_file.is_file():
        blockers.append("missing_file_based_secret_RUNPOD_API_KEY_FILE")
    projected_incremental_cost = float(max_hourly_rate_usd) * (float(max_live_minutes) / 60.0)
    if projected_incremental_cost > float(hard_cap_usd):
        blockers.append("closed_loop_runpod_projected_cost_exceeds_hard_cap_usd")
    return {
        "schema_version": "closed_loop_paid_provider_preflight.v1",
        "status": "ready" if not blockers else "blocked",
        "provider": "runpod",
        "max_hourly_rate_usd": float(max_hourly_rate_usd),
        "max_live_minutes": int(max_live_minutes),
        "projected_max_incremental_cost_usd": round(projected_incremental_cost, 6),
        "hard_cap_usd": float(hard_cap_usd),
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "preflight_is_no_spend": True,
            "preflight_does_not_call_runpod_api": True,
            "secret_values_not_read_into_artifact": True,
        },
    }


def _closed_loop_paid_provider_preflight(
    *,
    provider: str,
    allow_paid_provider_launch: bool,
) -> dict[str, Any]:
    provider_id = _string(provider).strip().lower()
    if provider_id == "runpod":
        return _runpod_paid_provider_preflight(
            allow_paid_provider_launch=allow_paid_provider_launch,
            max_hourly_rate_usd=_float_env("BLUEPRINT_RUNPOD_WAM_MAX_HOURLY_RATE", 0.80),
            max_live_minutes=_int_env("BLUEPRINT_RUNPOD_WAM_MAX_LIVE_MINUTES", 30),
            hard_cap_usd=_float_env("BLUEPRINT_RUNPOD_WAM_HARD_CAP_USD", 3.0),
        )
    if provider_id == "vast":
        return _vast_paid_provider_preflight(
            allow_paid_provider_launch=allow_paid_provider_launch,
            max_hourly_rate_usd=_float_env("BLUEPRINT_VAST_WAM_MAX_HOURLY_RATE", 0.35),
            max_live_minutes=_int_env("BLUEPRINT_VAST_WAM_MAX_LIVE_MINUTES", 30),
            session_max_live_minutes=_int_env("BLUEPRINT_VAST_WAM_SESSION_MAX_LIVE_MINUTES", 35),
            hard_cap_usd=_float_env("BLUEPRINT_VAST_WAM_HARD_CAP_USD", 3.0),
        )
    return {
        "schema_version": "closed_loop_paid_provider_preflight.v1",
        "status": "blocked",
        "provider": provider_id or provider,
        "blockers": [f"closed_loop_paid_provider_{provider_id or 'unknown'}_disabled_use_vast"],
        "claim_boundary": {
            "preflight_is_no_spend": True,
            "paid_closed_loop_provider_default_is_vast": True,
        },
    }


def build_closed_loop_wam_backend_readiness(
    *,
    selected_backend: str,
    use_provider_command: bool,
    oscar_repo: str | None = None,
    checkpoint: str | None = None,
    oscar_provider: str = "vast",
    allow_paid_provider_launch: bool = False,
) -> dict[str, Any]:
    """Describe which WAM backend the closed-loop runner can actually execute.

    Cosmos3 is executable only through its explicit per-step command contract;
    no configured command means no Cosmos3 runtime claim.
    """

    backend = _string(selected_backend).strip() or "oscar_wam"
    command_env_var = WAM_PROVIDER_COMMAND_ENV_BY_SUBSTRATE.get(backend)
    backend_command = _string(os.environ.get(command_env_var or "")) or _string(
        os.environ.get("BLUEPRINT_WAM_PROVIDER_COMMAND")
    )
    local_oscar_configured = bool(_string(oscar_repo) and _string(checkpoint))
    built_in_oscar_provider_configured = bool(backend == "oscar_wam" and use_provider_command)
    local_official_release = (
        official_release_contract(
            source_url=(
                _string(os.environ.get("BLUEPRINT_OSCAR_WAM_SOURCE_URL"))
                or _git_config_value(oscar_repo, "remote.origin.url")
            ),
            source_ref=(
                _string(os.environ.get("BLUEPRINT_OSCAR_WAM_SOURCE_REF"))
                or _git_head_commit(oscar_repo)
            ),
            hf_repo=_string(os.environ.get("BLUEPRINT_OSCAR_WAM_HF_REPO"))
            or OFFICIAL_OSCAR_HF_REPO,
            hf_revision=(
                _string(os.environ.get("BLUEPRINT_OSCAR_WAM_HF_REVISION"))
                or _checkpoint_revision_from_path(checkpoint)
            ),
        )
        if local_oscar_configured
        else None
    )
    experimental_oscar_version_allowed = _env_truthy(ALLOW_EXPERIMENTAL_OSCAR_VERSION_ENV)
    paid_provider_preflight = (
        _closed_loop_paid_provider_preflight(
            provider=oscar_provider,
            allow_paid_provider_launch=allow_paid_provider_launch,
        )
        if built_in_oscar_provider_configured and allow_paid_provider_launch
        else {
            "schema_version": "closed_loop_paid_provider_preflight.v1",
            "status": "not_requested",
            "provider": oscar_provider,
            "blockers": [],
            "claim_boundary": {"preflight_is_no_spend": True},
        }
    )
    explicit_provider_command_configured = bool(backend_command)
    supported_by_this_runner = backend in BUILT_IN_CLOSED_LOOP_WAM_BACKENDS
    blockers: list[str] = []
    if backend not in SUPPORTED_CLOSED_LOOP_WAM_BACKENDS:
        blockers.append("unsupported_closed_loop_wam_backend")
    if backend == "oscar_wam":
        if not (built_in_oscar_provider_configured or local_oscar_configured):
            blockers.append("blocked_missing_oscar_provider_or_local_checkpoint")
        if (
            local_oscar_configured
            and not built_in_oscar_provider_configured
            and not experimental_oscar_version_allowed
            and local_official_release is not None
        ):
            blockers.extend(official_release_blockers(local_official_release))
        blockers.extend(str(item) for item in paid_provider_preflight.get("blockers") or [])
    elif backend == "cosmos3_wam":
        if not explicit_provider_command_configured:
            blockers.append("blocked_cosmos3_wam_requires_explicit_provider_command")
    elif backend in SUPPORTED_CLOSED_LOOP_WAM_BACKENDS:
        blockers.append("blocked_selected_wam_backend_not_supported_by_runner")
    return {
        "schema_version": CLOSED_LOOP_WAM_BACKEND_READINESS_SCHEMA_VERSION,
        "selected_wam_backend": backend,
        "status": "ready" if not blockers else "blocked",
        "supported_backend_ids": list(SUPPORTED_CLOSED_LOOP_WAM_BACKENDS),
        "built_in_closed_loop_backend_ids": sorted(BUILT_IN_CLOSED_LOOP_WAM_BACKENDS),
        "supported_by_this_runner": supported_by_this_runner,
        "provider_adapter_kind": (
            "built_in_oscar_provider_adapter"
            if built_in_oscar_provider_configured
            else "local_oscar_subprocess"
            if local_oscar_configured
            else "explicit_provider_command"
            if explicit_provider_command_configured
            else "not_configured"
        ),
        "oscar_provider": oscar_provider,
        "allow_paid_provider_launch": bool(allow_paid_provider_launch),
        "local_oscar_repo_configured": bool(_string(oscar_repo)),
        "local_oscar_checkpoint_configured": bool(_string(checkpoint)),
        "official_oscar_release": local_official_release,
        "experimental_oscar_version_allowed": experimental_oscar_version_allowed,
        "explicit_provider_command_configured": explicit_provider_command_configured,
        "provider_command_env_var": command_env_var,
        "generic_provider_command_env_var": "BLUEPRINT_WAM_PROVIDER_COMMAND",
        "paid_provider_preflight": paid_provider_preflight,
        "strategy": get_wam_backend_strategy(backend),
        "blockers": blockers,
        "claim_boundary": {
            "readiness_manifest_is_no_spend": True,
            "readiness_manifest_is_not_model_execution_proof": True,
            "cosmos3_strategy_preference_does_not_imply_runtime_execution": True,
            "cosmos3_per_step_command_contract_wired": bool(
                backend == "cosmos3_wam" and explicit_provider_command_configured
            ),
            "oscar_provider_path_is_not_cosmos3_runtime": backend == "oscar_wam",
            "official_oscar_source_and_checkpoint_pinned": bool(
                local_official_release
                and local_official_release.get("official_release_match") is True
            )
            if local_oscar_configured
            else False,
        },
    }


def build_closed_loop_seed_conditioning_preflight(
    *,
    selected_backend: str,
    use_provider_command: bool,
    allow_paid_provider_launch: bool,
    steps: int,
    projected_skeleton_trace_path: str | Path | None,
) -> dict[str, Any]:
    """Fail closed when a paid multi-step provider loop would run without skeleton conditioning."""

    backend = _string(selected_backend).strip() or "oscar_wam"
    blockers: list[str] = []
    trace_path = (
        Path(projected_skeleton_trace_path).expanduser() if projected_skeleton_trace_path else None
    )
    required = bool(
        backend == "oscar_wam"
        and use_provider_command
        and allow_paid_provider_launch
        and int(steps) > 1
    )
    if required:
        if trace_path is None:
            blockers.append(
                "closed_loop_projected_skeleton_trace_missing_for_paid_multi_step_provider_wam"
            )
        elif not trace_path.is_file():
            blockers.append(
                "closed_loop_projected_skeleton_trace_file_missing_for_paid_multi_step_provider_wam"
            )
    return {
        "schema_version": "closed_loop_seed_conditioning_preflight.v1",
        "status": "ready" if not blockers else "blocked",
        "required": required,
        "selected_wam_backend": backend,
        "use_provider_command": bool(use_provider_command),
        "allow_paid_provider_launch": bool(allow_paid_provider_launch),
        "steps": int(steps),
        "projected_skeleton_trace_path": str(trace_path) if trace_path else None,
        "projected_skeleton_trace_present": bool(trace_path and trace_path.is_file()),
        "blockers": blockers,
        "claim_boundary": {
            "preflight_is_no_spend": True,
            "projected_skeleton_trace_is_conditioning_not_task_success_proof": True,
            "scene_or_task_specific_coordinates_hardcoded": False,
        },
    }


def _first_closed_loop_policy_action(
    *,
    route_points: Sequence[Sequence[float]],
    steps: int,
) -> dict[str, Any]:
    route = [tuple(float(c) for c in point) for point in route_points]
    target = route[-1]
    policy = DeterministicWalkToTargetPolicy()
    policy.reset({"route_points": list(route), "start": route[0], "target": target})
    decision = policy.step(
        StepContext(step=0, num_steps=max(1, int(steps)), probe_collision=lambda pose, yaw: 0)
    )
    return action_record(decision=decision, step=0, sim_time_s=0.0, target=target)


def build_closed_loop_provider_input_contract_preflight(
    *,
    start_frame_path: str | Path,
    route_points: Sequence[Sequence[float]],
    output_dir: str | Path,
    task_prompt: str,
    selected_backend: str,
    use_provider_command: bool,
    steps: int,
    num_frames: int,
    num_steps: int,
    guidance: float,
    seed: int,
    height: int,
    width: int,
    fps: float,
    projected_skeleton_trace_path: str | Path | None,
) -> dict[str, Any]:
    """Materialize the first-step provider input contract without launching a provider."""

    backend = _string(selected_backend).strip() or "oscar_wam"
    required = bool(backend == "oscar_wam" and use_provider_command)
    if not required:
        return {
            "schema_version": "closed_loop_provider_input_contract_preflight.v1",
            "status": "not_requested",
            "required": False,
            "selected_wam_backend": backend,
            "use_provider_command": bool(use_provider_command),
            "blockers": [],
            "claim_boundary": {"preflight_is_no_spend": True},
        }
    out = Path(output_dir).expanduser().resolve()
    ensure_dir(out)
    blockers: list[str] = []
    contract: dict[str, Any] = {}
    bundle_manifest: dict[str, Any] = {}
    step_input_path = out / "wam_generation_step_input.json"
    bundle_manifest_path = out / "oscar_wam_provider_bundle_manifest.json"
    try:
        if not route_points:
            raise ValueError("empty_route_points")
        action = _first_closed_loop_policy_action(route_points=route_points, steps=steps)
        step_input = build_wam_generation_step_input(
            current_frame_path=start_frame_path,
            action=action,
            step_index=1,
            output_dir=out / "step_0001",
            task_prompt=task_prompt,
            projected_skeleton_trace_path=projected_skeleton_trace_path,
        )
        write_json(step_input_path, step_input)
        from .oscar_wam_provider_bundle import build_oscar_wam_provider_bundle

        bundle_manifest = build_oscar_wam_provider_bundle(
            job_dir=out,
            wam_rollout_input_manifest=step_input_path,
            num_frames=int(num_frames),
            height=int(height),
            width=int(width),
            fps=float(fps),
            num_steps=int(num_steps),
            guidance=float(guidance),
            seed=int(seed) + 1,
        )
        contract = _mapping(bundle_manifest.get("input_package_contract_diagnostic"))
        blockers.extend(str(item) for item in bundle_manifest.get("blockers") or [])
        if contract.get("status") == "blocked":
            blockers.extend(str(item) for item in contract.get("blockers") or [])
    except Exception as exc:
        blockers.append(f"provider_input_contract_preflight_failed:{type(exc).__name__}")
    return {
        "schema_version": "closed_loop_provider_input_contract_preflight.v1",
        "status": "ready" if not blockers else "blocked",
        "required": True,
        "selected_wam_backend": backend,
        "use_provider_command": bool(use_provider_command),
        "step_input_path": str(step_input_path),
        "bundle_manifest_path": str(bundle_manifest_path),
        "bundle_status": bundle_manifest.get("status"),
        "contract_status": contract.get("status"),
        "contract_warnings": contract.get("warnings") or [],
        "contract_blockers": contract.get("blockers") or [],
        "autoregressive_risk_flags": contract.get("autoregressive_risk_flags") or [],
        "high_risk_flags": contract.get("high_risk_flags") or [],
        "ranking_risk_flags": contract.get("ranking_risk_flags") or [],
        "autoregressive_risk_level": contract.get("autoregressive_risk_level"),
        "policy_ranking_risk_level": contract.get("policy_ranking_risk_level"),
        "policy_ranking_claim_safe": contract.get("policy_ranking_claim_safe"),
        "short_rollout_sanity_recommended_before_scale_up": bool(
            contract.get("short_rollout_sanity_recommended_before_scale_up")
        ),
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "preflight_is_no_spend": True,
            "provider_input_contract_is_not_model_execution_proof": True,
            "provider_input_contract_is_not_generated_rollout_quality_proof": True,
            "scene_or_task_specific_coordinates_hardcoded": False,
        },
    }


def build_closed_loop_short_rollout_sanity_gate(
    *,
    selected_backend: str,
    use_provider_command: bool,
    allow_paid_provider_launch: bool,
    steps: int,
    provider_input_contract_preflight: Mapping[str, Any],
    short_visual_sanity_manifest_path: str | Path | None = None,
    expected_policy_observation_path: str | Path | None = None,
) -> dict[str, Any]:
    """Require a passed short visual sanity run before paid long WAM scale-up."""

    backend = _string(selected_backend).strip() or "oscar_wam"
    risk_recommends_short_sanity = bool(
        provider_input_contract_preflight.get("short_rollout_sanity_recommended_before_scale_up")
    )
    required = bool(
        backend == "oscar_wam"
        and use_provider_command
        and allow_paid_provider_launch
        and int(steps) > 2
        and risk_recommends_short_sanity
    )
    manifest_path_text = _string(short_visual_sanity_manifest_path) or _string(
        os.getenv(PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST_ENV)
    )
    if not required:
        return {
            "schema_version": "closed_loop_short_rollout_sanity_gate.v1",
            "status": "not_required",
            "required": False,
            "selected_wam_backend": backend,
            "steps": int(steps),
            "risk_recommends_short_sanity": risk_recommends_short_sanity,
            "blockers": [],
            "claim_boundary": {"gate_is_no_spend": True},
        }
    blockers = ["closed_loop_paid_long_wam_requires_passed_short_rollout_sanity"]
    validation: dict[str, Any] = {}
    if not manifest_path_text:
        blockers.append("short_visual_sanity_manifest_env_missing")
    else:
        try:
            from .unitree_groot_n17_sonic_vast_persistent_session import (
                validate_persistent_wam_short_visual_sanity_manifest,
            )

            validation = validate_persistent_wam_short_visual_sanity_manifest(
                manifest_path_text,
                policy_observation_path=expected_policy_observation_path,
            )
            if validation.get("status") == "passed_short_visual_sanity":
                blockers = []
            else:
                blockers.extend(str(item) for item in validation.get("blockers") or [])
        except Exception as exc:
            blockers.append(f"short_visual_sanity_manifest_validation_failed:{type(exc).__name__}")
    return {
        "schema_version": "closed_loop_short_rollout_sanity_gate.v1",
        "status": "ready" if not blockers else "blocked",
        "required": True,
        "selected_wam_backend": backend,
        "steps": int(steps),
        "risk_recommends_short_sanity": risk_recommends_short_sanity,
        "short_visual_sanity_manifest_path": manifest_path_text or None,
        "expected_policy_observation_path": _string(expected_policy_observation_path) or None,
        "short_visual_sanity_validation": validation,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "gate_is_no_spend": True,
            "short_visual_sanity_is_not_task_success_proof": True,
            "short_visual_sanity_is_scale_up_gate_only": True,
            "generated_world_rank_fidelity_result_proven": False,
        },
    }


def _short_visual_sanity_provider_for_oscar_provider(provider: str) -> tuple[str, str]:
    provider_text = _string(provider).strip().lower()
    if provider_text in {"vast", "runpod"}:
        return provider_text, "explicit_provider"
    return "vast", "auto_defaults_to_vast_provider_command_order"


def build_closed_loop_short_visual_sanity_launch_plan(
    *,
    selected_backend: str,
    use_provider_command: bool,
    allow_paid_provider_launch: bool,
    steps: int,
    provider_input_contract_preflight: Mapping[str, Any],
    output_dir: str | Path,
    oscar_provider: str,
    task_prompt: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Materialize the exact short-sanity command needed before long paid rollout."""

    backend = _string(selected_backend).strip() or "oscar_wam"
    risk_recommends_short_sanity = bool(
        provider_input_contract_preflight.get("short_rollout_sanity_recommended_before_scale_up")
    )
    required = bool(
        backend == "oscar_wam"
        and use_provider_command
        and allow_paid_provider_launch
        and int(steps) > 2
        and risk_recommends_short_sanity
    )
    root = Path(output_dir).expanduser().resolve()
    policy_observation_path = root / "short_visual_sanity_policy_observation.json"
    job_dir = root / "short_visual_sanity_job"
    expected_manifest_path = job_dir / "persistent_wam_short_visual_sanity_manifest.json"
    if not required:
        return {
            "schema_version": "closed_loop_short_visual_sanity_launch_plan.v1",
            "status": "not_required",
            "required": False,
            "selected_wam_backend": backend,
            "steps": int(steps),
            "risk_recommends_short_sanity": risk_recommends_short_sanity,
            "blockers": [],
            "claim_boundary": {"plan_is_no_spend": True},
        }

    blockers: list[str] = []
    step_input_path = Path(
        _string(provider_input_contract_preflight.get("step_input_path"))
    ).expanduser()
    step_input: dict[str, Any] = {}
    policy_observation: dict[str, Any] = {}
    if not step_input_path.is_file():
        blockers.append("closed_loop_short_sanity_step_input_missing")
    else:
        try:
            step_input = json.loads(step_input_path.read_text(encoding="utf-8"))
            policy_observation = _mapping(step_input.get("current_policy_observation"))
        except Exception as exc:
            blockers.append(f"closed_loop_short_sanity_step_input_unreadable:{type(exc).__name__}")
    if not policy_observation and step_input_path.is_file():
        blockers.append("closed_loop_short_sanity_policy_observation_missing")
    if policy_observation:
        write_json(policy_observation_path, policy_observation)

    short_provider, provider_resolution = _short_visual_sanity_provider_for_oscar_provider(
        oscar_provider
    )
    paid_provider_preflight = _closed_loop_paid_provider_preflight(
        provider=short_provider,
        allow_paid_provider_launch=allow_paid_provider_launch,
    )
    provider_launch_blockers = [str(item) for item in paid_provider_preflight.get("blockers") or []]
    command_argv = [
        sys.executable,
        "-m",
        "blueprint_pipeline.persistent_wam_short_visual_sanity",
        "--policy-observation",
        str(policy_observation_path),
        "--job-dir",
        str(job_dir),
        "--provider",
        short_provider,
        "--transition-count",
        "2",
        "--task-prompt",
        _string(task_prompt),
        "--timeout-seconds",
        str(float(timeout_seconds)),
    ]
    command_materialized = bool(policy_observation and not blockers)
    launch_allowed_now = bool(command_materialized and not provider_launch_blockers)
    if blockers:
        status = "blocked"
    elif provider_launch_blockers:
        status = "blocked_provider_authorization"
    else:
        status = "ready"
    return {
        "schema_version": "closed_loop_short_visual_sanity_launch_plan.v1",
        "status": status,
        "required": True,
        "selected_wam_backend": backend,
        "steps": int(steps),
        "risk_recommends_short_sanity": risk_recommends_short_sanity,
        "source_step_input_path": str(step_input_path),
        "policy_observation_path": str(policy_observation_path) if policy_observation else None,
        "policy_observation_materialized": bool(policy_observation),
        "command_materialized": command_materialized,
        "job_dir": str(job_dir),
        "provider": short_provider,
        "provider_resolution": provider_resolution,
        "paid_provider_preflight": paid_provider_preflight,
        "provider_launch_allowed_now": launch_allowed_now,
        "provider_launch_blockers": sorted(set(provider_launch_blockers)),
        "command_argv": command_argv,
        "command_display": shlex.join(command_argv),
        "expected_manifest_path": str(expected_manifest_path),
        "unlock_env": {
            PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST_ENV: str(expected_manifest_path)
        },
        "followup_closed_loop_arg": [
            "--short-visual-sanity-manifest",
            str(expected_manifest_path),
        ],
        "blockers": sorted(set([*blockers, *provider_launch_blockers])),
        "claim_boundary": {
            "plan_is_no_spend": True,
            "short_visual_sanity_is_scale_up_gate_only": True,
            "short_visual_sanity_is_not_task_success_proof": True,
            "generated_world_rank_fidelity_result_proven": False,
        },
    }


def make_oscar_provider_command_wam_backend(
    *,
    work_dir: str | Path,
    task_prompt: str,
    num_frames: int = DEFAULT_OSCAR_NUM_FRAMES,
    num_steps: int = 35,
    guidance: float = 6.0,
    seed: int = 42,
    height: int = 480,
    width: int = 640,
    fps: float = 15.0,
    provider: str = "vast",
    allow_paid_provider_launch: bool = False,
    timeout_seconds: float = 3600.0,
    adapter_run: Callable[
        [Sequence[str] | None], Mapping[str, Any]
    ] = run_oscar_wam_provider_adapter,
    extract_next_frame: Callable[[str | Path, str | Path], Path | None] | None = None,
    projected_skeleton_trace_path: str | Path | None = None,
) -> WamGenerateNext:
    """Drive one fresh OSCAR provider run per closed-loop step.

    When a projected skeleton trace is available, keep it attached for every
    autoregressive step. The generated observation changes per step, but the
    provider still needs the action/robot visual-conditioning stream; dropping it
    after the first frame makes later WAM calls much more likely to drift or
    collapse while still looking structurally "wired."
    """
    resolved_work = Path(work_dir).expanduser().resolve()
    ensure_dir(resolved_work)
    rgb_context_history: list[str] = []

    def _generate_next(
        current_frame: str,
        action: Mapping[str, Any],
        step_index: int,
        history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        step_dir = resolved_work / f"step_{step_index:04d}"
        ensure_dir(step_dir)
        current_frame_path = Path(current_frame).expanduser()
        if current_frame_path.is_file():
            current_resolved = str(current_frame_path.resolve())
            if current_resolved not in rgb_context_history:
                rgb_context_history.append(current_resolved)
        step_input = build_wam_generation_step_input(
            current_frame_path=current_frame,
            action=action,
            step_index=step_index,
            output_dir=step_dir,
            task_prompt=task_prompt,
            projected_skeleton_trace_path=projected_skeleton_trace_path,
            rgb_context_frame_paths=rgb_context_history[-max(2, int(num_frames)) :],
        )
        step_input_path = step_dir / "wam_generation_step_input.json"
        write_json(step_input_path, step_input)
        output_path = step_dir / "wam_provider_output.json"
        adapter_args = [
            "--mode",
            "auto",
            "--provider",
            provider,
            "--work-dir",
            str(step_dir / "provider_workspace"),
            "--timeout-seconds",
            str(float(timeout_seconds)),
        ]
        if allow_paid_provider_launch:
            adapter_args.append("--allow-paid-provider-launch")
        with _temporary_environ(
            {
                "BLUEPRINT_WAM_ROLLOUT_INPUT": str(step_input_path),
                "BLUEPRINT_WAM_ROLLOUT_OUTPUT": str(output_path),
                "BLUEPRINT_OSCAR_WAM_NUM_FRAMES": str(max(1, int(num_frames))),
                "BLUEPRINT_OSCAR_WAM_NUM_STEPS": str(max(1, int(num_steps))),
                "BLUEPRINT_OSCAR_WAM_GUIDANCE": str(float(guidance)),
                "BLUEPRINT_OSCAR_WAM_SEED": str(int(seed) + int(step_index)),
                "BLUEPRINT_OSCAR_WAM_HEIGHT": str(int(height)),
                "BLUEPRINT_OSCAR_WAM_WIDTH": str(int(width)),
                "BLUEPRINT_OSCAR_WAM_FPS": str(float(fps)),
            }
        ):
            payload = dict(adapter_run(adapter_args) or {})
        if not output_path.is_file():
            write_json(output_path, payload)
        video = _provider_video_path(payload)
        if not video:
            return {
                "status": "blocked",
                "wam_backend": "oscar_2b_per_step_provider",
                "generated_frame_path": "",
                "generated_video_path": "",
                "provider_payload": payload,
                "provider_output_path": str(output_path),
                "fresh_provider_model_run_claimed": False,
                "blockers": payload.get("blockers") or ["oscar_provider_video_missing"],
            }
        visual_acceptance_blockers = _provider_payload_visual_acceptance_blockers(payload)
        if visual_acceptance_blockers:
            return {
                "status": "blocked",
                "wam_backend": "oscar_2b_per_step_provider",
                "generated_frame_path": "",
                "generated_video_path": video,
                "provider_payload": payload,
                "provider_output_path": str(output_path),
                "fresh_provider_model_run_claimed": _provider_payload_proves_fresh_model(payload),
                "blockers": visual_acceptance_blockers,
            }
        extractor = extract_next_frame or extract_next_observation_frame_from_video
        next_frame = extractor(video, step_dir / "next_observation")
        if next_frame is None or not Path(next_frame).is_file():
            return {
                "status": "blocked",
                "wam_backend": "oscar_2b_per_step_provider",
                "generated_frame_path": "",
                "generated_video_path": video,
                "provider_payload": payload,
                "provider_output_path": str(output_path),
                "fresh_provider_model_run_claimed": _provider_payload_proves_fresh_model(payload),
                "blockers": ["oscar_provider_next_observation_frame_extraction_failed"],
            }
        next_frame_resolved = str(Path(next_frame).expanduser().resolve())
        if next_frame_resolved not in rgb_context_history:
            rgb_context_history.append(next_frame_resolved)
        return {
            "status": "completed" if payload.get("status") == "completed" else "blocked",
            "wam_backend": "oscar_2b_per_step_provider",
            "generated_frame_path": str(next_frame),
            "generated_video_path": video,
            "provider_payload": payload,
            "provider_output_path": str(output_path),
            "fresh_provider_model_run_claimed": _provider_payload_proves_fresh_model(payload),
            "blockers": payload.get("blockers") or [],
        }

    return _generate_next


def make_oscar_per_step_wam_backend(
    *,
    oscar_generate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    work_dir: str | Path,
    task_prompt: str,
    num_frames: int = DEFAULT_OSCAR_NUM_FRAMES,
    skeleton_for_action: Callable[[Mapping[str, Any], int], Any] | None = None,
    seed: int = 42,
    require_manipulation_effector_progress: bool = False,
    minimum_manipulation_effector_progress_m: float = (MANIPULATION_EFFECTOR_PROGRESS_MINIMUM_M),
) -> WamGenerateNext:
    """A ``wam_generate_next`` backend that drives real per-step OSCAR-2B generation.

    ``oscar_generate`` is the injected inference call — it receives a per-step request (see
    :func:`build_oscar_per_step_request`) and must return a mapping with a ``generated_frame_path``
    (the next observation) and optionally ``generated_video_path``. On GPU this is a thin call into
    a persistent OSCAR-2B pod; in tests it is mocked. ``skeleton_for_action`` projects the G1
    skeleton landmarks for an action (the Isaac projector at run time; ``None`` omits conditioning).
    """
    resolved_work = Path(work_dir).expanduser().resolve()

    def _generate_next(
        current_frame: str,
        action: Mapping[str, Any],
        step_index: int,
        history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        projection = skeleton_for_action(action, step_index) if skeleton_for_action else None
        if isinstance(projection, Mapping):
            skeleton = [
                dict(row)
                for row in projection.get("landmarks", []) or []
                if isinstance(row, Mapping)
            ]
            skeleton_trace_rows = []
            for frame_index, raw_frame in enumerate(
                projection.get("controller_fk_sequence", []) or []
            ):
                frame = _mapping(raw_frame)
                frame_landmarks = [
                    dict(row)
                    for row in frame.get("landmarks", []) or []
                    if isinstance(row, Mapping)
                ]
                if frame_landmarks:
                    skeleton_trace_rows.append(
                        {
                            "frame_index": frame_index,
                            "source_controller_horizon_frame_index": (
                                frame.get("horizon_frame_index", frame_index)
                            ),
                            "projected_landmarks": frame_landmarks,
                        }
                    )
            projection_metadata = _with_action_conditioning_digests(
                {**dict(projection), "landmarks": skeleton}
            )
        else:
            skeleton = list(projection or [])
            skeleton_trace_rows = []
            projection_metadata = {}
        projection_blockers = (
            _action_conditioning_blockers(
                action=action,
                wam_output={
                    "skeleton_conditioning": projection_metadata,
                    "generated_robot_state": projection_metadata.get("generated_robot_state"),
                },
            )
            if isinstance(projection, Mapping)
            else []
        )
        if projection_blockers:
            return {
                "status": "blocked",
                "blockers": [
                    f"oscar_controller_fk_projection_invalid:{blocker}"
                    for blocker in projection_blockers
                ],
                "generated_frame_path": "",
                "generated_video_path": None,
                "skeleton_conditioning": projection_metadata or None,
                "generated_robot_state": projection_metadata.get("generated_robot_state"),
                "wam_backend": "oscar_2b_per_step",
                "wam_generation_status": "blocked",
                "wam_generation_blockers": [
                    f"oscar_controller_fk_projection_invalid:{blocker}"
                    for blocker in projection_blockers
                ],
            }
        manipulation_progress_report: dict[str, Any] = {}
        if require_manipulation_effector_progress:
            manipulation_progress_report = _manipulation_effector_progress_report(
                projection_metadata,
                minimum_progress_m=float(minimum_manipulation_effector_progress_m),
            )
            report_dir = resolved_work / f"oscar_step_{step_index:04d}"
            ensure_dir(report_dir)
            report_path = report_dir / "manipulation_effector_progress_report.json"
            write_json(report_path, manipulation_progress_report)
            if manipulation_progress_report.get("capability_gate_passed") is not True:
                progress_blockers = _string_list(manipulation_progress_report.get("blockers")) or [
                    "manipulation_controller_fk_effector_progress_not_proven"
                ]
                return {
                    "status": "blocked",
                    "blockers": progress_blockers,
                    "generated_frame_path": "",
                    "generated_video_path": None,
                    "skeleton_conditioning": projection_metadata or None,
                    "generated_robot_state": projection_metadata.get("generated_robot_state"),
                    "wam_backend": "oscar_2b_per_step",
                    "wam_generation_status": "blocked",
                    "wam_generation_blockers": progress_blockers,
                    "manipulation_effector_progress_report": (manipulation_progress_report),
                    "manipulation_effector_progress_report_path": str(report_path),
                }
        request = build_oscar_per_step_request(
            current_frame_path=current_frame,
            action=action,
            step_index=step_index,
            task_prompt=task_prompt,
            num_frames=num_frames,
            output_dir=resolved_work,
            skeleton_landmarks=skeleton,
            skeleton_trace_rows=skeleton_trace_rows,
            seed=seed,
        )
        result = dict(oscar_generate(request) or {})
        generated_frame = _string(result.get("generated_frame_path"))
        return {
            "status": result.get("status"),
            "blockers": list(result.get("blockers") or []),
            "generated_frame_path": generated_frame,
            "generated_video_path": result.get("generated_video_path"),
            "skeleton_conditioning": {
                **projection_metadata,
                "landmarks": skeleton,
            }
            if skeleton
            else None,
            "generated_robot_state": projection_metadata.get("generated_robot_state"),
            "manipulation_effector_progress_report": manipulation_progress_report or None,
            "wam_backend": "oscar_2b_per_step",
            "wam_generation_status": result.get("status"),
            "wam_generation_blockers": list(result.get("blockers") or []),
            "oscar_gpu_residency_report_path": result.get("oscar_gpu_residency_report_path"),
            "oscar_gpu_residency_samples_path": result.get("oscar_gpu_residency_samples_path"),
            "oscar_gpu_residency": result.get("oscar_gpu_residency"),
        }

    return _generate_next


def make_cosmos3_per_step_command_wam_backend(
    *,
    command: str | Sequence[str],
    work_dir: str | Path,
    task_prompt: str,
    skeleton_for_action: Callable[[Mapping[str, Any], int], Any],
    timeout_seconds: float = 3600.0,
) -> WamGenerateNext:
    """Drive a configured Cosmos3 forward/inverse/cross-view model each loop step."""

    argv = shlex.split(command) if isinstance(command, str) else [str(item) for item in command]
    if not argv:
        raise ValueError("cosmos3_per_step_command_missing")
    resolved_work = Path(work_dir).expanduser().resolve()
    ensure_dir(resolved_work)
    runtime_session_id = (
        "cosmos3-runtime-" + hashlib.sha256(str(resolved_work).encode("utf-8")).hexdigest()[:24]
    )
    seen_runtime_result_ids: set[str] = set()

    def _generate_next(
        current_frame: str,
        action: Mapping[str, Any],
        step_index: int,
        history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        step_dir = resolved_work / f"step_{step_index:04d}"
        ensure_dir(step_dir)
        projection = skeleton_for_action(action, step_index)
        if not isinstance(projection, Mapping):
            return {
                "status": "blocked",
                "wam_generation_status": "blocked",
                "blockers": ["cosmos3_controller_fk_projection_missing"],
            }
        projection_metadata = _with_action_conditioning_digests(dict(projection))
        projection_blockers = _action_conditioning_blockers(
            action=action,
            wam_output={
                "skeleton_conditioning": projection_metadata,
                "generated_robot_state": projection_metadata.get("generated_robot_state"),
            },
        )
        if projection_blockers:
            return {
                "status": "blocked",
                "wam_generation_status": "blocked",
                "blockers": [
                    f"cosmos3_controller_fk_projection_invalid:{blocker}"
                    for blocker in projection_blockers
                ],
            }
        source_frame = Path(current_frame).expanduser().resolve()
        action_sha256 = _canonical_sha256(action)
        source_observation_sha256 = (
            hashlib.sha256(source_frame.read_bytes()).hexdigest() if source_frame.is_file() else ""
        )
        request_path = step_dir / "cosmos3_closed_loop_request.json"
        output_path = step_dir / "cosmos3_closed_loop_output.json"
        request = {
            "schema_version": "cosmos3_closed_loop_step_request.v1",
            "runtime_session_id": runtime_session_id,
            "step_index": step_index,
            "task_prompt": task_prompt,
            "source_observation_artifact": {
                "path": str(source_frame),
                "sha256": source_observation_sha256,
            },
            "action": dict(action),
            "action_sha256": action_sha256,
            "action_history": [dict(row) for row in history],
            "skeleton_conditioning": projection_metadata,
            "required_mode_outputs": [
                "forward_dynamics",
                "inverse_dynamics",
                "cross_view",
            ],
        }
        write_json(request_path, request)
        request_sha256 = _canonical_sha256(request)
        if output_path.exists():
            output_path.unlink()
        completed = subprocess.run(
            argv,
            input=json.dumps(request, sort_keys=True),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1.0, float(timeout_seconds)),
            cwd=str(step_dir),
            env={
                **os.environ,
                "BLUEPRINT_COSMOS3_CLOSED_LOOP_INPUT": str(request_path),
                "BLUEPRINT_COSMOS3_CLOSED_LOOP_OUTPUT": str(output_path),
                "BLUEPRINT_COSMOS3_CLOSED_LOOP_STEP_INDEX": str(step_index),
            },
        )
        payload: dict[str, Any] = {}
        if output_path.is_file():
            payload = _mapping(json.loads(output_path.read_text(encoding="utf-8")))
        elif completed.stdout.strip():
            payload = _mapping(json.loads(completed.stdout))
        blockers: list[str] = []
        runtime_result_id = _string(payload.get("runtime_result_id")).strip()
        if completed.returncode != 0:
            blockers.append(f"cosmos3_closed_loop_command_returncode_{completed.returncode}")
        if not (
            payload.get("schema_version") == "cosmos3_closed_loop_step_output.v1"
            and payload.get("status") == "completed"
            and payload.get("fresh_model_command_executed_this_invocation") is True
            and payload.get("learned_wam_model_ran") is True
            and payload.get("consumed_action_sha256") == action_sha256
            and payload.get("source_observation_sha256") == source_observation_sha256
            and payload.get("runtime_session_id") == runtime_session_id
            and payload.get("request_sha256") == request_sha256
            and runtime_result_id
            and runtime_result_id not in seen_runtime_result_ids
        ):
            blockers.append("cosmos3_closed_loop_output_contract_invalid")
        checkpoint_validation = validate_checkpoint_attestation(
            _mapping(payload.get("sc3_checkpoint_attestation"))
        )
        if checkpoint_validation.get("status") != "validated":
            blockers.append("cosmos3_sc3_checkpoint_attestation_not_validated")
        checkpoint_sha256 = _string(
            _mapping(payload.get("sc3_checkpoint_attestation")).get("checkpoint_sha256")
        ).lower()
        mode_outputs = _mapping(payload.get("mode_outputs"))
        mode_artifact_sha256s: set[str] = set()
        for mode in ("forward_dynamics", "inverse_dynamics", "cross_view"):
            mode_output = _mapping(mode_outputs.get(mode))
            _, evidence_blockers = _validate_hashed_evidence_artifacts(
                [mode_output.get("artifact")]
            )
            artifact_ref = _mapping(mode_output.get("artifact"))
            artifact_sha256 = _string(artifact_ref.get("sha256")).lower()
            artifact_path = Path(_string(artifact_ref.get("path"))).expanduser()
            artifact_payload: dict[str, Any] = {}
            if artifact_path.is_file():
                try:
                    artifact_payload = _mapping(
                        json.loads(artifact_path.read_text(encoding="utf-8"))
                    )
                except (OSError, json.JSONDecodeError):
                    pass
            if (
                mode_output.get("status") != "completed"
                or mode_output.get("fresh_mode_execution_proven") is not True
                or evidence_blockers
                or not artifact_sha256
                or artifact_sha256 in mode_artifact_sha256s
                or artifact_payload.get("schema_version") != "cosmos3_sc3_mode_output.v1"
                or artifact_payload.get("status") != "completed"
                or artifact_payload.get("mode") != mode
                or artifact_payload.get("runtime_session_id") != runtime_session_id
                or artifact_payload.get("runtime_result_id") != runtime_result_id
                or artifact_payload.get("request_sha256") != request_sha256
                or artifact_payload.get("action_sha256") != action_sha256
                or artifact_payload.get("source_observation_sha256") != source_observation_sha256
                or artifact_payload.get("checkpoint_sha256") != checkpoint_sha256
            ):
                blockers.append(f"cosmos3_{mode}_output_not_proven")
            mode_artifact_sha256s.add(artifact_sha256)
        generated_frame = Path(_string(payload.get("generated_frame_path"))).expanduser()
        generated_frame_ref = _mapping(payload.get("generated_frame_artifact"))
        _, generated_frame_blockers = _validate_hashed_evidence_artifacts([generated_frame_ref])
        if (
            not generated_frame.is_file()
            or generated_frame_blockers
            or Path(_string(generated_frame_ref.get("path"))).expanduser().resolve()
            != generated_frame.resolve()
        ):
            blockers.append("cosmos3_generated_frame_missing")
        signed_result = {
            "schema_version": "sc3_cosmos3_runtime_result.v1",
            "runtime_session_id": runtime_session_id,
            "runtime_result_id": runtime_result_id,
            "request_sha256": request_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "source_observation_sha256": source_observation_sha256,
            "action_sha256": action_sha256,
            "generated_frame_artifact": generated_frame_ref,
            "mode_outputs": mode_outputs,
        }
        runtime_attestation = validate_trusted_ed25519_attestation(
            _mapping(payload.get("runtime_attestation")),
            signed_payload=signed_result,
            prefix="cosmos3_runtime_attestation",
            trusted_public_key_sha256_env=(SC3_COSMOS3_RUNTIME_TRUSTED_PUBLIC_KEY_SHA256_ENV),
        )
        blockers.extend(_string_list(runtime_attestation.get("blockers")))
        blockers = sorted(set(blockers))
        if not blockers:
            seen_runtime_result_ids.add(runtime_result_id)
        return {
            "status": "completed" if not blockers else "blocked",
            "generated_frame_path": str(generated_frame) if not blockers else None,
            "generated_video_path": payload.get("generated_video_path"),
            "skeleton_conditioning": projection_metadata,
            "generated_robot_state": projection_metadata.get("generated_robot_state"),
            "wam_backend": "cosmos3_nano_per_step",
            "wam_generation_status": "completed" if not blockers else "blocked",
            "wam_generation_blockers": blockers,
            "sc3_checkpoint_attestation_validation": checkpoint_validation,
            "sc3_mode_outputs": mode_outputs,
            "cosmos3_command_output_path": str(output_path),
        }

    return _generate_next


def make_controller_fk_skeleton_projector(
    *,
    command: str | Sequence[str],
    work_dir: str | Path,
    timeout_seconds: float = 120.0,
) -> Callable[[Mapping[str, Any], int], Mapping[str, Any]]:
    """Wrap a real controller/FK converter behind a strict JSON contract."""

    import subprocess

    argv = shlex.split(command) if isinstance(command, str) else [str(item) for item in command]
    if not argv:
        raise ValueError("controller_fk_skeleton_command_missing")
    resolved_work = Path(work_dir).expanduser().resolve()
    ensure_dir(resolved_work)
    seen_runtime_result_ids: set[str] = set()
    camera_projection_context: dict[str, Any] | None = None
    camera_projection_context_path = _string(
        os.environ.get(CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_ENV)
    ).strip()
    if camera_projection_context_path:
        camera_projection_context = _load_live_controller_fk_camera_projection_context(
            camera_projection_context_path
        )

    def _project(action: Mapping[str, Any], step_index: int) -> Mapping[str, Any]:
        step_dir = resolved_work / f"step_{step_index:04d}"
        ensure_dir(step_dir)
        action_sha256 = _canonical_sha256(action)
        input_path = step_dir / "controller_fk_input.json"
        output_path = step_dir / "controller_fk_output.json"
        request = {
            "schema_version": "controller_fk_skeleton_request.v1",
            "step_index": int(step_index),
            "source_action_sha256": action_sha256,
            "action": dict(action),
        }
        if camera_projection_context is not None:
            request["camera_projection_context"] = dict(camera_projection_context)
        write_json(input_path, request)
        request_sha256 = _canonical_sha256(request)
        result = subprocess.run(
            argv,
            cwd=str(step_dir),
            env={
                **os.environ,
                "BLUEPRINT_CONTROLLER_FK_INPUT": str(input_path),
                "BLUEPRINT_CONTROLLER_FK_OUTPUT": str(output_path),
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=float(timeout_seconds),
        )
        (step_dir / "controller_fk_stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (step_dir / "controller_fk_stderr.log").write_text(result.stderr or "", encoding="utf-8")
        write_json(
            step_dir / "controller_fk_command_result.json",
            {
                "schema_version": "controller_fk_command_result.v1",
                "returncode": int(result.returncode),
                "stdout_log": str((step_dir / "controller_fk_stdout.log").resolve()),
                "stderr_log": str((step_dir / "controller_fk_stderr.log").resolve()),
                "output_path": str(output_path),
                "output_present": output_path.is_file(),
            },
        )
        if result.returncode != 0:
            raise RuntimeError(f"controller_fk_skeleton_command_nonzero:{int(result.returncode)}")
        if output_path.is_file():
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(result.stdout or "{}")
        if not isinstance(payload, Mapping):
            raise RuntimeError("controller_fk_skeleton_output_not_object")
        projection = dict(payload)
        if projection.get("status") != "completed":
            raise RuntimeError("controller_fk_skeleton_output_not_completed")
        if _string(projection.get("source_action_sha256")).strip() != action_sha256:
            raise RuntimeError("controller_fk_skeleton_action_identity_mismatch")
        runtime_result_id = _string(projection.get("runtime_result_id")).strip()
        if not runtime_result_id or runtime_result_id in seen_runtime_result_ids:
            raise RuntimeError("controller_fk_skeleton_runtime_result_id_missing_or_replayed")
        controller_code_ref = _mapping(projection.get("controller_code_artifact"))
        robot_model_ref = _mapping(projection.get("robot_model_artifact"))
        _, controller_code_blockers = _validate_hashed_evidence_artifacts([controller_code_ref])
        _, robot_model_blockers = _validate_hashed_evidence_artifacts([robot_model_ref])
        if (
            controller_code_blockers
            or _string(controller_code_ref.get("sha256")).lower()
            != _string(projection.get("controller_sha256")).lower()
        ):
            raise RuntimeError("controller_fk_skeleton_controller_code_invalid")
        if (
            robot_model_blockers
            or _string(robot_model_ref.get("sha256")).lower()
            != _string(projection.get("robot_model_sha256")).lower()
        ):
            raise RuntimeError("controller_fk_skeleton_robot_model_invalid")
        projection_context_sha256 = _string(
            projection.get("camera_projection_context_sha256")
        ).lower()
        source_frame_sha256 = _string(projection.get("camera_source_frame_sha256")).lower()
        registration = _mapping(projection.get("cross_simulator_registration"))
        if not _is_sha256(projection_context_sha256):
            raise RuntimeError("controller_fk_projection_context_sha256_invalid")
        if not _is_sha256(source_frame_sha256):
            raise RuntimeError("controller_fk_projection_source_frame_sha256_invalid")
        if registration.get("status") != "passed" or registration.get("surrogate") is not False:
            raise RuntimeError("controller_fk_cross_simulator_registration_not_proven")
        signed_result = {
            "schema_version": "sc3_controller_fk_runtime_result.v1",
            "request_sha256": request_sha256,
            "step_index": int(step_index),
            "source_action_sha256": action_sha256,
            "runtime_result_id": runtime_result_id,
            "controller_id": projection.get("controller_id"),
            "controller_sha256": projection.get("controller_sha256"),
            "robot_model_sha256": projection.get("robot_model_sha256"),
            "controller_code_artifact": controller_code_ref,
            "robot_model_artifact": robot_model_ref,
            "derived_via_controller_fk": projection.get("derived_via_controller_fk"),
            "landmarks": projection.get("landmarks"),
            "camera_projection_context_sha256": projection.get("camera_projection_context_sha256"),
            "camera_source_frame_sha256": projection.get("camera_source_frame_sha256"),
            "cross_simulator_registration": projection.get("cross_simulator_registration"),
            "generated_robot_state": projection.get("generated_robot_state"),
        }
        attestation = validate_trusted_ed25519_attestation(
            _mapping(projection.get("executor_attestation")),
            signed_payload=signed_result,
            prefix="controller_fk_executor_attestation",
            trusted_public_key_sha256_env=(SC3_FK_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV),
        )
        if attestation.get("status") != "validated":
            raise RuntimeError(
                "controller_fk_skeleton_executor_attestation_invalid:"
                + ",".join(_string_list(attestation.get("blockers")))
            )
        projection["derived_via_controller_fk"] = (
            projection.get("derived_via_controller_fk") is True
        )
        action_target = action.get("target")
        if _finite_numeric_sequence(action_target, minimum_length=3) and len(list(action_target)) == 3:
            # Progress is judged toward the manipulation target the action was
            # conditioned on. The camera framing-validation point below exists to
            # prove the appliance is IN FRAME and sits 0.76 m from the handle on
            # the live bundle -- measuring hand progress toward it falsely
            # rejected a +134 mm reach (attempt 067, runner_done-9631481e).
            projection["task_target_world_xyz_m"] = [float(value) for value in action_target]
            projection["task_target_binding"] = {
                "source": "action_manipulation_target",
                "camera_projection_context_sha256": projection_context_sha256,
                "source_frame_sha256": source_frame_sha256,
            }
        if camera_projection_context is not None:
            required_points = _mapping(
                _mapping(
                    _mapping(camera_projection_context.get("camera_contract")).get(
                        "framing_validation"
                    )
                ).get("required_world_points")
            )
            task_target = _mapping(required_points.get("task_target")).get("world_xyz_m")
            if _finite_numeric_sequence(task_target, minimum_length=3) and len(task_target) == 3:
                projection["camera_framing_task_target_world_xyz_m"] = [
                    float(value) for value in task_target
                ]
                if "task_target_world_xyz_m" not in projection:
                    projection["task_target_world_xyz_m"] = [
                        float(value) for value in task_target
                    ]
                    projection["task_target_binding"] = {
                        "source": "live_isaac_robot_pov_camera_framing_validation",
                        "camera_projection_context_sha256": projection_context_sha256,
                        "source_frame_sha256": source_frame_sha256,
                    }
        normalized_projection = _with_action_conditioning_digests(projection)
        evidence_blockers = _action_conditioning_blockers(
            action=action,
            wam_output={
                "skeleton_conditioning": normalized_projection,
                "generated_robot_state": normalized_projection.get("generated_robot_state"),
            },
        )
        if evidence_blockers:
            raise RuntimeError(
                "controller_fk_skeleton_evidence_invalid:" + ",".join(evidence_blockers)
            )
        seen_runtime_result_ids.add(runtime_result_id)
        return normalized_projection

    return _project


def _provider_completed(provider_statuses: Sequence[Any], provider: str) -> bool:
    for status_value in provider_statuses:
        if not isinstance(status_value, Mapping):
            continue
        if status_value.get("provider") != provider:
            continue
        return bool(status_value.get("ran")) and not bool(status_value.get("blockers") or [])
    return False


def _da3_completed(provider_statuses: Sequence[Any]) -> bool:
    for status_value in provider_statuses:
        if not isinstance(status_value, Mapping):
            continue
        if status_value.get("provider") != "depth":
            continue
        kind = _string(status_value.get("kind")).lower()
        return bool(
            status_value.get("ran")
            and kind in {"depth_anything_3", "da3", "depth-anything-3"}
            and not bool(status_value.get("blockers") or [])
        )
    return False


def _step_backend_status(step_record: Mapping[str, Any]) -> dict[str, Any]:
    backend = step_record.get("harness_backend")
    if not isinstance(backend, Mapping):
        backend = (
            step_record.get("backend") if isinstance(step_record.get("backend"), Mapping) else {}
        )
    provider_statuses = (
        list(backend.get("provider_statuses") or []) if isinstance(backend, Mapping) else []
    )
    backend_kind = _string(backend.get("kind")) if isinstance(backend, Mapping) else ""
    generated_rgb_validated = bool(
        backend_kind == GENERATED_RGB_POLICY_OBSERVATION_BACKEND_KIND
        and backend.get("generated_rgb_policy_observation_validated")
        and backend.get("built_in_generated_rgb_validation_ran")
    )
    return {
        "backend_status": backend.get("status") if isinstance(backend, Mapping) else None,
        "backend_kind": backend_kind or None,
        "real_model_ran": bool(
            isinstance(backend, Mapping) and backend.get("real_sam_or_depth_model_ran")
        ),
        "generated_rgb_policy_observation_validated": generated_rgb_validated,
        "perception_model_ran": bool(
            isinstance(backend, Mapping)
            and (
                backend.get("real_perception_model_ran")
                or backend.get("real_sam_or_depth_model_ran")
            )
        ),
        "no_perception_model_ran": bool(
            generated_rgb_validated
            and not backend.get("real_perception_model_ran")
            and not backend.get("real_sam_or_depth_model_ran")
            and not backend.get("sam3_ran")
            and not backend.get("da3_ran")
        ),
        "provider_statuses": provider_statuses,
        "sam3_completed": _provider_completed(provider_statuses, "sam3"),
        "depth_completed": _provider_completed(provider_statuses, "depth"),
        "da3_completed": _da3_completed(provider_statuses),
    }


def _frame_signal_stats(frame: Any, cv2: Any) -> dict[str, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    edges = cv2.Canny(gray, 50, 150)
    return {
        "mean_luma": round(float(gray.mean()), 3),
        "std_luma": round(float(gray.std()), 3),
        "luma_min": int(gray.min()),
        "luma_max": int(gray.max()),
        "luma_range": int(gray.max()) - int(gray.min()),
        "dark_pixel_ratio": round(float((gray < 32).mean()), 6),
        "bright_pixel_ratio": round(float((gray > 224).mean()), 6),
        "edge_density": round(float((edges > 0).mean()), 6),
    }


def _next_observation_signal_blockers(stats: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    mean_luma = float(stats.get("mean_luma") or 0.0)
    std_luma = float(stats.get("std_luma") or 0.0)
    luma_range = float(stats.get("luma_range") or 0.0)
    dark_ratio = float(stats.get("dark_pixel_ratio") or 0.0)
    bright_ratio = float(stats.get("bright_pixel_ratio") or 0.0)
    edge_density = float(stats.get("edge_density") or 0.0)
    if mean_luma < 25.0 or dark_ratio > 0.78:
        blockers.append("next_observation_candidate_too_dark")
    if mean_luma > 245.0 and bright_ratio > 0.90:
        blockers.append("next_observation_candidate_overexposed")
    if std_luma < 8.0 or luma_range < 32.0:
        blockers.append("next_observation_candidate_flat_or_low_contrast")
    if edge_density < 0.002:
        blockers.append("next_observation_candidate_low_scene_structure")
    if edge_density > 0.12 and std_luma < 28.0:
        blockers.append("next_observation_candidate_static_noise_artifact")
    return blockers


def _write_selection_manifest(
    out_dir: Path,
    *,
    status: str,
    video_path: Path,
    candidates: Sequence[Mapping[str, Any]],
    selected_frame_index: int | None,
    blockers: Sequence[str],
    extraction_method: str,
) -> None:
    write_json(
        out_dir / "next_observation_selection.json",
        {
            "schema_version": NEXT_OBSERVATION_SELECTION_SCHEMA_VERSION,
            "status": status,
            "video_path": str(video_path),
            "selected_frame_index": selected_frame_index,
            "extraction_method": extraction_method,
            "candidate_count": len(candidates),
            "candidates": list(candidates),
            "blockers": list(blockers),
            "claim_boundary": {
                "selected_frame_is_generated_next_observation_candidate": status == "completed",
                "visual_signal_gate_is_not_task_success_evidence": True,
                "scene_or_task_specific_pixels_used": False,
            },
        },
    )


def extract_next_observation_frame_from_video(
    video_path: str | Path, out_dir: str | Path
) -> Path | None:
    """Default ``extract_next_frame`` for OSCAR clips.

    The first video frame is treated as the seed/current observation. The next observation is the
    earliest future frame with enough generic visual signal to feed the harness. This avoids
    advancing the closed loop with late frames that have collapsed to dark/flat artifacts while
    keeping the gate task- and scene-neutral.
    """
    resolved_out = Path(out_dir).expanduser()
    resolved_out.mkdir(parents=True, exist_ok=True)
    resolved_video = Path(video_path).expanduser()
    try:
        import cv2  # local import: only needed where a real clip is produced
    except ImportError:
        cv2 = None
    if cv2 is not None:
        capture = cv2.VideoCapture(str(resolved_video))
        candidates: list[dict[str, Any]] = []
        selected_index: int | None = None
        selected_frame = None
        try:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                stats = _frame_signal_stats(frame, cv2)
                blockers = (
                    ["next_observation_candidate_is_seed_frame"]
                    if frame_index == 0
                    else _next_observation_signal_blockers(stats)
                )
                candidates.append(
                    {
                        "frame_index": frame_index,
                        **stats,
                        "blockers": blockers,
                    }
                )
                if frame_index > 0 and not blockers:
                    selected_index = frame_index
                    selected_frame = frame.copy()
                    break
                frame_index += 1
        finally:
            capture.release()
        if selected_frame is None:
            _write_selection_manifest(
                resolved_out,
                status="blocked",
                video_path=resolved_video,
                candidates=candidates,
                selected_frame_index=None,
                blockers=["no_usable_future_next_observation_frame"],
                extraction_method="opencv_signal_gate",
            )
            return None
        frame_path = resolved_out / "next_observation.png"
        if not cv2.imwrite(str(frame_path), selected_frame):
            _write_selection_manifest(
                resolved_out,
                status="blocked",
                video_path=resolved_video,
                candidates=candidates,
                selected_frame_index=selected_index,
                blockers=["next_observation_frame_write_failed"],
                extraction_method="opencv_signal_gate",
            )
            return None
        _write_selection_manifest(
            resolved_out,
            status="completed",
            video_path=resolved_video,
            candidates=candidates,
            selected_frame_index=selected_index,
            blockers=[],
            extraction_method="opencv_signal_gate",
        )
        return frame_path

    frame_path = resolved_out / "next_observation.png"
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(resolved_video),
                "-vf",
                "select=gte(n\\,1)",
                "-frames:v",
                "1",
                str(frame_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not frame_path.is_file():
        _write_selection_manifest(
            resolved_out,
            status="blocked",
            video_path=resolved_video,
            candidates=[],
            selected_frame_index=None,
            blockers=["ffmpeg_first_future_frame_extraction_failed"],
            extraction_method="ffmpeg_first_future_frame",
        )
        return None
    _write_selection_manifest(
        resolved_out,
        status="completed",
        video_path=resolved_video,
        candidates=[],
        selected_frame_index=1,
        blockers=[],
        extraction_method="ffmpeg_first_future_frame",
    )
    return frame_path


def extract_last_frame_via_opencv(video_path: str | Path, out_dir: str | Path) -> Path | None:
    """Compatibility wrapper for older callers.

    Despite the historical name, the closed-loop now extracts the earliest usable future frame
    rather than blindly taking the last frame.
    """
    return extract_next_observation_frame_from_video(video_path, out_dir)


def _geometry_sidecar_from_route(
    route_payload: Mapping[str, Any],
    *,
    start_frame_path: str | Path,
) -> Path | None:
    candidates: list[Any] = [
        route_payload.get("manipulation_pov_geometry_path"),
        route_payload.get("seed_geometry_path"),
        route_payload.get("geometry_path"),
    ]
    source_trace = _string(route_payload.get("source_trace"))
    if source_trace:
        candidates.append(Path(source_trace).expanduser().parent / "manipulation_pov_geometry.json")
    start = Path(start_frame_path).expanduser()
    candidates.append(start.parent / "manipulation_pov_geometry.json")
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()
    return None


def _infer_skeleton_segments(landmarks: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    ids = {_string(item.get("landmark_id")) for item in landmarks}
    segments: list[dict[str, str]] = []
    for prefix in ("left", "right"):
        wrist = f"{prefix}_wrist_link"
        hand = f"{prefix}_hand_link"
        if wrist in ids and hand in ids:
            segments.append({"from": wrist, "to": hand})
    return segments


def _scaled_projection_xy(
    projection: Mapping[str, Any],
    *,
    x_scale: float,
    y_scale: float,
) -> tuple[float, float] | None:
    if projection.get("available") is not True:
        return None
    try:
        return float(projection.get("u_px")) * x_scale, float(projection.get("v_px")) * y_scale
    except (TypeError, ValueError):
        return None


def _landmark_temporal_reach_fraction(landmark: Mapping[str, Any]) -> float:
    text = f"{_string(landmark.get('landmark_id'))} {_string(landmark.get('link_role'))}".lower()
    if "hand" in text or "gripper" in text:
        return 0.70
    if "wrist" in text:
        return 0.45
    if "elbow" in text or "forearm" in text:
        return 0.25
    if "shoulder" in text:
        return 0.08
    return 0.35


def _temporal_projected_landmarks(
    landmarks: Sequence[Mapping[str, Any]],
    *,
    target_projection_xy: tuple[float, float] | None,
    progress: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    bounded_progress = max(0.0, min(1.0, float(progress)))
    for landmark in landmarks:
        item = dict(landmark)
        projection = dict(_mapping(item.get("image_projection")))
        if target_projection_xy is not None and projection.get("available") is True:
            try:
                u_px = float(projection.get("u_px"))
                v_px = float(projection.get("v_px"))
            except (TypeError, ValueError):
                output.append(item)
                continue
            reach = _landmark_temporal_reach_fraction(item) * bounded_progress
            projection["u_px"] = round(u_px + (target_projection_xy[0] - u_px) * reach, 3)
            projection["v_px"] = round(v_px + (target_projection_xy[1] - v_px) * reach, 3)
        item["image_projection"] = projection
        output.append(item)
    return output


def materialize_projected_skeleton_trace_from_seed_geometry(
    *,
    route_payload: Mapping[str, Any],
    start_frame_path: str | Path,
    output_dir: str | Path,
    num_frames: int = DEFAULT_OSCAR_NUM_FRAMES,
) -> Path | None:
    """Convert a seed-render geometry sidecar into OSCAR's projected-skeleton trace format.

    This uses route/seed metadata only. It does not know about kitchens, refrigerators, or fixed
    coordinates; if no geometry sidecar is present, the caller simply proceeds without the trace.
    """
    geometry_path = _geometry_sidecar_from_route(route_payload, start_frame_path=start_frame_path)
    if geometry_path is None:
        return None
    try:
        payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    frames = payload.get("frames") if isinstance(payload.get("frames"), list) else [payload]
    if not frames:
        return None
    row = next((item for item in frames if isinstance(item, Mapping)), None)
    if row is None:
        return None
    raw_landmarks = row.get("projected_landmarks")
    if not isinstance(raw_landmarks, Sequence) or isinstance(raw_landmarks, (str, bytes)):
        return None
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(Path(start_frame_path).expanduser()) as image:
            target_width, target_height = image.size
    except Exception:
        return None
    seed_quality = (
        row.get("seed_frame_quality") if isinstance(row.get("seed_frame_quality"), Mapping) else {}
    )
    image_size = (
        seed_quality.get("image_size_px")
        if isinstance(seed_quality.get("image_size_px"), Sequence)
        else []
    )
    source_width = float(image_size[0]) if len(image_size) >= 2 else float(target_width)
    source_height = float(image_size[1]) if len(image_size) >= 2 else float(target_height)
    if source_width <= 0.0 or source_height <= 0.0:
        return None
    x_scale = float(target_width) / source_width
    y_scale = float(target_height) / source_height
    landmarks: list[dict[str, Any]] = []
    for landmark in raw_landmarks:
        if not isinstance(landmark, Mapping):
            continue
        projection = landmark.get("image_projection")
        if not isinstance(projection, Mapping) or projection.get("available") is not True:
            continue
        projected_xy = _scaled_projection_xy(projection, x_scale=x_scale, y_scale=y_scale)
        if projected_xy is None:
            continue
        landmarks.append(
            {
                "landmark_id": _string(landmark.get("landmark_id")),
                "link_role": _string(landmark.get("link_role")),
                "image_projection": {
                    "available": True,
                    "u_px": round(projected_xy[0], 3),
                    "v_px": round(projected_xy[1], 3),
                    "depth_m": projection.get("depth_m"),
                },
            }
        )
    if not landmarks:
        return None
    raw_segments = row.get("segments") if isinstance(row.get("segments"), Sequence) else []
    segments = [
        {"from": _string(segment.get("from")), "to": _string(segment.get("to"))}
        for segment in raw_segments
        if isinstance(segment, Mapping) and segment.get("from") and segment.get("to")
    ]
    if not segments:
        segments = _infer_skeleton_segments(landmarks)
    target_projection = row.get("target_projection")
    target_projection_xy = (
        _scaled_projection_xy(target_projection, x_scale=x_scale, y_scale=y_scale)
        if isinstance(target_projection, Mapping)
        else None
    )
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "g1_projected_skeleton_trace.jsonl"
    frame_count = max(1, int(num_frames)) if target_projection_xy is not None else 1
    base_frame_index = int(row.get("frame_index") or row.get("step") or 0)
    lines: list[str] = []
    for trace_index in range(frame_count):
        progress = trace_index / max(frame_count - 1, 1)
        trace_landmarks = _temporal_projected_landmarks(
            landmarks,
            target_projection_xy=target_projection_xy,
            progress=progress,
        )
        trace_row = {
            "schema_version": "blueprint.g1.projected_upper_body_skeleton.v1",
            "status": "completed",
            "source_geometry_path": str(geometry_path),
            "frame_index": base_frame_index + trace_index,
            "temporal_progress": round(progress, 6),
            "camera": _string(row.get("camera")) or "head_pov",
            "image_size_px": [int(target_width), int(target_height)],
            "source_image_size_px": [int(source_width), int(source_height)],
            "target_projection": {
                "available": target_projection_xy is not None,
                "u_px": round(target_projection_xy[0], 3) if target_projection_xy else None,
                "v_px": round(target_projection_xy[1], 3) if target_projection_xy else None,
            },
            "projected_landmark_count": len(trace_landmarks),
            "landmarks": trace_landmarks,
            "segments": segments,
            "claim_boundary": {
                "projected_skeleton_trace_derived_from_seed_render_geometry": True,
                "temporal_rows_are_target_conditioning_from_resolved_affordance_projection": bool(
                    target_projection_xy
                ),
                "not_a_learned_robot_policy_action": True,
                "simulated_state_not_physical_robot_sensor_evidence": True,
                "not_task_success_or_contact_proof": True,
            },
        }
        lines.append(json.dumps(trace_row, sort_keys=True))
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return trace_path


def _positive_pid(value: Any) -> int | None:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _oscar_process_group_absent(process_group_id: int) -> bool:
    """Return true only when the isolated OSCAR process group no longer exists."""

    try:
        os.killpg(int(process_group_id), 0)
    except ProcessLookupError:
        return True
    except OSError as exc:
        return exc.errno == errno.ESRCH
    return False


def _wait_for_oscar_process_group_absence(
    process_group_id: int,
    *,
    timeout_seconds: float,
    process_group_absent: Callable[[int], bool],
) -> bool:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        if process_group_absent(process_group_id):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


def _linux_process_parent_pid(pid: int) -> int | None:
    """Read one Linux parent PID without invoking another process.

    ``/proc/<pid>/stat`` puts the executable name in parentheses and that name may itself
    contain spaces or parentheses, so split after the final closing parenthesis rather than
    splitting the entire record on whitespace.
    """

    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        fields_after_name = stat.rsplit(")", 1)[1].strip().split()
        return _positive_pid(fields_after_name[1])
    except (IndexError, OSError, TypeError, ValueError):
        return None


def _nvidia_smi_query(
    *,
    run: Callable[..., Any],
    argv: Sequence[str],
) -> dict[str, Any]:
    try:
        completed = run(
            list(argv),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}"[:2000],
        }
    returncode = getattr(completed, "returncode", 1)
    stdout = _string(getattr(completed, "stdout", ""))
    stderr = _string(getattr(completed, "stderr", ""))
    return {
        "status": "completed" if returncode == 0 else "failed",
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr[:2000],
    }


def _csv_rows(text: str) -> list[list[str]]:
    return [
        [cell.strip() for cell in row]
        for row in csv.reader(text.splitlines())
        if row and any(cell.strip() for cell in row)
    ]


def _nvidia_memory_mib(value: Any) -> float | None:
    cleaned = re.sub(r"\s*(?:MiB|MB)\s*$", "", _string(value).strip(), flags=re.IGNORECASE)
    if not cleaned or cleaned.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        return round(float(cleaned), 3)
    except ValueError:
        return None


def _valid_gpu_uuid(value: Any) -> str | None:
    uuid = _string(value).strip()
    if not uuid or uuid.upper() in {"N/A", "[N/A]", "UNKNOWN", "NONE"}:
        return None
    return uuid


def collect_oscar_gpu_residency_sample(
    *,
    query_run: Callable[..., Any],
    role_root_pids: Mapping[str, int | None],
    parent_pid: Callable[[int], int | None] = _linux_process_parent_pid,
    host_to_local_pid_map: Callable[[], Mapping[int, int]] = _linux_nvidia_host_to_local_pid_map,
    proc_root: Path = Path("/proc"),
    sample_index: int = 0,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    """Capture one simultaneous GPU/process snapshot for the four runtime roles.

    A compute application is attributed to a role when its PID is that role's exported root PID
    or a descendant of it. This matters for torchrun and Isaac, whose CUDA-owning child process is
    commonly not the PID exported by the launch shell.
    """

    gpu_query = _nvidia_smi_query(
        run=query_run,
        argv=(
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ),
    )
    compute_query = _nvidia_smi_query(
        run=query_run,
        argv=(
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ),
    )
    blockers: list[str] = []
    if gpu_query["status"] != "completed":
        blockers.append("nvidia_smi_gpu_query_failed")
    if compute_query["status"] != "completed":
        blockers.append("nvidia_smi_compute_apps_query_failed")

    gpus: list[dict[str, Any]] = []
    if gpu_query["status"] == "completed":
        for row in _csv_rows(_string(gpu_query.get("stdout"))):
            if len(row) != 5:
                blockers.append("nvidia_smi_gpu_query_row_malformed")
                continue
            uuid = _valid_gpu_uuid(row[1])
            if uuid is None:
                blockers.append("nvidia_smi_gpu_uuid_missing")
            gpus.append(
                {
                    "index": row[0],
                    "uuid": uuid,
                    "memory_total_mib": _nvidia_memory_mib(row[2]),
                    "memory_used_mib": _nvidia_memory_mib(row[3]),
                    "memory_free_mib": _nvidia_memory_mib(row[4]),
                }
            )
        if not gpus:
            blockers.append("nvidia_smi_gpu_inventory_empty")
    inventory_uuids = {item["uuid"] for item in gpus if item.get("uuid")}
    if len(inventory_uuids) != len([item for item in gpus if item.get("uuid")]):
        blockers.append("nvidia_smi_gpu_uuid_duplicate")

    normalized_roots = {
        role: _positive_pid(role_root_pids.get(role)) for role in OSCAR_GPU_RESIDENCY_REQUIRED_ROLES
    }
    for role, root_pid in normalized_roots.items():
        if root_pid is None:
            blockers.append(f"gpu_residency_{role}_root_pid_missing")

    try:
        namespace_pid_map = {
            host_pid: local_pid
            for raw_host_pid, raw_local_pid in host_to_local_pid_map().items()
            if (host_pid := _positive_pid(raw_host_pid)) is not None
            and (local_pid := _positive_pid(raw_local_pid)) is not None
        }
    except Exception:
        namespace_pid_map = {}

    compute_apps: list[dict[str, Any]] = []
    if compute_query["status"] == "completed":
        for row in _csv_rows(_string(compute_query.get("stdout"))):
            if len(row) != 4:
                blockers.append("nvidia_smi_compute_apps_query_row_malformed")
                continue
            uuid = _valid_gpu_uuid(row[0])
            pid = _positive_pid(row[1])
            if uuid is None:
                blockers.append("nvidia_smi_compute_app_gpu_uuid_missing")
            elif uuid not in inventory_uuids:
                blockers.append("nvidia_smi_compute_app_gpu_uuid_not_in_inventory")
            if pid is None:
                blockers.append("nvidia_smi_compute_app_pid_invalid")
                ancestor_chain: list[int] = []
                local_pid = None
            else:
                local_pid = namespace_pid_map.get(pid, pid)
                ancestor_chain = _pid_ancestor_chain(local_pid, parent_pid=parent_pid)
            roles = [
                role
                for role, root_pid in normalized_roots.items()
                if root_pid is not None and root_pid in ancestor_chain
            ]
            compute_apps.append(
                {
                    "gpu_uuid": uuid,
                    "pid": pid,
                    "local_pid": local_pid,
                    "pid_namespace_translation": (
                        "nspid_host_to_local"
                        if pid is not None and local_pid != pid
                        else "identity"
                    ),
                    "process_name": row[2],
                    "used_gpu_memory_mib": _nvidia_memory_mib(row[3]),
                    "ancestor_chain": ancestor_chain,
                    "roles": roles,
                }
            )

    roles_by_gpu_uuid: dict[str, set[str]] = {uuid: set() for uuid in inventory_uuids}
    for app in compute_apps:
        uuid = app.get("gpu_uuid")
        if uuid in roles_by_gpu_uuid:
            roles_by_gpu_uuid[uuid].update(app.get("roles") or [])
    required = set(OSCAR_GPU_RESIDENCY_REQUIRED_ROLES)
    simultaneous_uuids = sorted(
        uuid for uuid, roles in roles_by_gpu_uuid.items() if required.issubset(roles)
    )
    attribution_mode = ATTRIBUTION_MODE_HOST_PID_NAMESPACE
    fallback: dict[str, Any] = {}
    # Hosts whose container runtime never exposes the outer NSpid chain cannot
    # attribute host PIDs at all, which the host-PID path reports identically to
    # "the roles are not resident" (attempt 068 failed a sealed run this way).
    # Re-derive residency from our own processes' device handles before ruling.
    if not simultaneous_uuids and any(
        compute_app_attribution_unavailable(app) for app in compute_apps
    ):
        fallback = device_handle_residency_fallback(
            role_root_pids=normalized_roots,
            required_roles=sorted(required),
            inventory_uuids=inventory_uuids,
            parent_pid=parent_pid,
            proc_root=proc_root,
        )
        # Fallback failures are diagnostic, never per-sample blockers: early
        # samples legitimately precede the roles' first CUDA context, and a
        # union-of-sample-blockers summary would make that transient permanent.
        if fallback.get("applied"):
            attribution_mode = ATTRIBUTION_MODE_DEVICE_HANDLE_FALLBACK
            roles_by_gpu_uuid[str(fallback["gpu_uuid"])].update(fallback["roles"])
            simultaneous_uuids = sorted(
                uuid for uuid, roles in roles_by_gpu_uuid.items() if required.issubset(roles)
            )
    same_gpu_role_sets = {uuid: sorted(roles) for uuid, roles in sorted(roles_by_gpu_uuid.items())}
    diagnostic_text = "\n".join(
        [
            _string(gpu_query.get("stdout")),
            _string(gpu_query.get("stderr")),
            _string(compute_query.get("stdout")),
            _string(compute_query.get("stderr")),
        ]
    )
    cuda_oom_detected = bool(
        re.search(r"(?:CUDA\s+out\s+of\s+memory|CUDNN_STATUS_ALLOC_FAILED)", diagnostic_text, re.I)
    )
    xid_detected = bool(re.search(r"\bXid\b", diagnostic_text, re.I))
    if cuda_oom_detected:
        blockers.append("cuda_out_of_memory_detected")
    if xid_detected:
        blockers.append("nvidia_xid_detected")
    blockers = sorted(set(blockers))
    return {
        "schema_version": OSCAR_GPU_RESIDENCY_SAMPLE_SCHEMA_VERSION,
        "sample_index": int(sample_index),
        "sampled_at": utc_now_iso(),
        "elapsed_seconds": round(max(0.0, float(elapsed_seconds)), 3),
        "query_status": "completed" if not blockers else "blocked",
        "gpu_query": {
            "status": gpu_query["status"],
            "returncode": gpu_query["returncode"],
            "stderr": gpu_query["stderr"],
        },
        "compute_apps_query": {
            "status": compute_query["status"],
            "returncode": compute_query["returncode"],
            "stderr": compute_query["stderr"],
        },
        "role_root_pids": normalized_roots,
        "gpus": gpus,
        "compute_apps": compute_apps,
        "roles_by_gpu_uuid": same_gpu_role_sets,
        "role_attribution_mode": attribution_mode,
        "device_handle_attribution": fallback,
        "simultaneous_required_roles_gpu_uuids": simultaneous_uuids,
        "all_required_roles_simultaneously_resident_on_same_gpu": bool(simultaneous_uuids),
        "cuda_oom_detected": cuda_oom_detected,
        "nvidia_xid_detected": xid_detected,
        "blockers": blockers,
    }


def summarize_oscar_gpu_residency_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    role_root_pids: Mapping[str, int | None],
    runtime_diagnostics: str = "",
    extra_blockers: Sequence[str] = (),
) -> dict[str, Any]:
    """Reduce timestamped snapshots without turning free memory into an admission cutoff."""

    blockers = {str(item) for item in extra_blockers if str(item)}
    same_gpu_uuids: set[str] = set()
    simultaneous_sample_indices: list[int] = []
    peak_used: float | None = None
    minimum_free: float | None = None
    role_observed_sample_counts = {role: 0 for role in OSCAR_GPU_RESIDENCY_REQUIRED_ROLES}
    for sample in samples:
        blockers.update(_string_list(sample.get("blockers")))
        simultaneous = _string_list(sample.get("simultaneous_required_roles_gpu_uuids"))
        if simultaneous:
            same_gpu_uuids.update(simultaneous)
            simultaneous_sample_indices.append(int(sample.get("sample_index") or 0))
        observed_roles: set[str] = set()
        for roles in _mapping(sample.get("roles_by_gpu_uuid")).values():
            observed_roles.update(_string_list(roles))
        for role in observed_roles:
            if role in role_observed_sample_counts:
                role_observed_sample_counts[role] += 1
        for gpu in sample.get("gpus") or []:
            if not isinstance(gpu, Mapping):
                continue
            used = gpu.get("memory_used_mib")
            free = gpu.get("memory_free_mib")
            if isinstance(used, (int, float)):
                peak_used = float(used) if peak_used is None else max(peak_used, float(used))
            if isinstance(free, (int, float)):
                minimum_free = (
                    float(free) if minimum_free is None else min(minimum_free, float(free))
                )
    if not samples:
        blockers.add("oscar_gpu_residency_samples_absent")
    if not simultaneous_sample_indices:
        blockers.add("required_roles_not_simultaneously_resident_on_same_gpu")
        # Name the cause when the host, not the workload, defeated attribution:
        # "no roles resident" and "no roles attributable" are opposite defects
        # with opposite remedies, and attempt 068 could not tell them apart.
        for sample in reversed(list(samples)):
            attribution = sample.get("device_handle_attribution")
            if isinstance(attribution, Mapping) and attribution.get("blockers"):
                blockers.update(str(item) for item in attribution["blockers"])
                break

    cuda_oom_detected = any(sample.get("cuda_oom_detected") is True for sample in samples) or bool(
        re.search(
            r"(?:CUDA\s+out\s+of\s+memory|CUDNN_STATUS_ALLOC_FAILED)",
            runtime_diagnostics,
            re.I,
        )
    )
    xid_detected = any(sample.get("nvidia_xid_detected") is True for sample in samples) or bool(
        re.search(r"\bXid\b", runtime_diagnostics, re.I)
    )
    if cuda_oom_detected:
        blockers.add("cuda_out_of_memory_detected")
    if xid_detected:
        blockers.add("nvidia_xid_detected")
    sorted_blockers = sorted(blockers)
    proof_passed = bool(simultaneous_sample_indices) and not sorted_blockers
    return {
        "schema_version": OSCAR_GPU_RESIDENCY_REPORT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if proof_passed else "blocked",
        "proof_passed": proof_passed,
        "required_roles": list(OSCAR_GPU_RESIDENCY_REQUIRED_ROLES),
        "role_root_pids": {
            role: _positive_pid(role_root_pids.get(role))
            for role in OSCAR_GPU_RESIDENCY_REQUIRED_ROLES
        },
        "sample_count": len(samples),
        "simultaneous_sample_count": len(simultaneous_sample_indices),
        "simultaneous_sample_indices": simultaneous_sample_indices,
        "same_gpu_uuids": sorted(same_gpu_uuids),
        "role_observed_sample_counts": role_observed_sample_counts,
        "peak_gpu_memory_used_mib": peak_used,
        "minimum_gpu_memory_free_mib": minimum_free,
        "nvidia_smi_query_failure_detected": any(
            blocker.startswith("nvidia_smi_") and blocker.endswith("_failed")
            for blocker in sorted_blockers
        ),
        "gpu_uuid_failure_detected": any("gpu_uuid" in blocker for blocker in sorted_blockers),
        "cuda_oom_detected": cuda_oom_detected,
        "nvidia_xid_detected": xid_detected,
        "blockers": sorted_blockers,
        "claim_boundary": {
            "same_gpu_simultaneous_residency_requires_one_atomic_nvidia_smi_sample": True,
            "process_role_classification_uses_exported_root_pid_ancestry": True,
            "gpu_free_memory_is_observed_not_an_admission_cutoff": True,
            "residency_does_not_claim_semantic_task_success": True,
        },
    }


class _OscarGpuResidencySampler:
    """A subprocess-scoped, bounded sampler that streams evidence instead of buffering forever."""

    def __init__(
        self,
        *,
        output_dir: Path,
        oscar_pid: int | None,
        query_run: Callable[..., Any],
        parent_pid: Callable[[int], int | None] = _linux_process_parent_pid,
        host_to_local_pid_map: Callable[
            [], Mapping[int, int]
        ] = _linux_nvidia_host_to_local_pid_map,
        interval_seconds: float = OSCAR_GPU_RESIDENCY_SAMPLE_INTERVAL_SECONDS,
        max_samples: int = OSCAR_GPU_RESIDENCY_MAX_SAMPLES,
    ) -> None:
        self.samples_path = output_dir / "oscar_gpu_residency.jsonl"
        self.report_path = output_dir / "oscar_gpu_residency_report.json"
        self.query_run = query_run
        self.parent_pid = parent_pid
        self.host_to_local_pid_map = host_to_local_pid_map
        self.interval_seconds = min(1.0, max(0.5, float(interval_seconds)))
        self.max_samples = max(1, int(max_samples))
        self.role_root_pids = {
            role: _positive_pid(os.environ.get(env_name))
            for role, env_name in OSCAR_GPU_RESIDENCY_PID_ENV_BY_ROLE.items()
        }
        self.role_root_pids["oscar"] = _positive_pid(oscar_pid)
        self._samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._started_monotonic = time.monotonic()
        self._sample_limit_reached = False
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="oscar-gpu-residency-sampler",
            daemon=True,
        )

    def start(self) -> None:
        self.samples_path.unlink(missing_ok=True)
        self.report_path.unlink(missing_ok=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.is_set() and len(self._samples) < self.max_samples:
            sample = collect_oscar_gpu_residency_sample(
                query_run=self.query_run,
                role_root_pids=self.role_root_pids,
                parent_pid=self.parent_pid,
                host_to_local_pid_map=self.host_to_local_pid_map,
                sample_index=len(self._samples),
                elapsed_seconds=time.monotonic() - self._started_monotonic,
            )
            self._samples.append(sample)
            with self.samples_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
                handle.flush()
            if self._stop.wait(self.interval_seconds):
                break
        if len(self._samples) >= self.max_samples and not self._stop.is_set():
            self._sample_limit_reached = True

    def finalize(self, *, runtime_diagnostics: str = "") -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=12)
        extra_blockers: list[str] = []
        if self._thread.is_alive():
            extra_blockers.append("oscar_gpu_residency_sampler_stop_timeout")
        if self._sample_limit_reached:
            extra_blockers.append("oscar_gpu_residency_sample_limit_reached")
        report = summarize_oscar_gpu_residency_samples(
            self._samples,
            role_root_pids=self.role_root_pids,
            runtime_diagnostics=runtime_diagnostics,
            extra_blockers=extra_blockers,
        )
        report["samples_path"] = str(self.samples_path)
        report["report_path"] = str(self.report_path)
        report["sample_interval_seconds"] = self.interval_seconds
        report["maximum_sample_count"] = self.max_samples
        write_json(self.report_path, report)
        return report


def build_oscar_inference_argv(
    *,
    python: str,
    oscar_repo: str | Path,
    checkpoint: str | Path,
    first_frame_path: str,
    prompt: str,
    num_frames: int,
    num_steps: int,
    guidance: float,
    seed: int,
    height: int,
    width: int,
    fps: float,
    output_video: str | Path,
    skeleton_video: str | Path | None = None,
) -> list[str]:
    """The OSCAR inference argv for one per-step next-observation generation.

    Mirrors oscar_wam_command_adapter's invocation: torch.distributed.run inference_oscar.py with
    the current observation as --first-frame and (optionally) the action's projected skeleton as
    --skeleton-video. Pure argv construction so the real backend below stays unit-testable.
    """
    repo = Path(oscar_repo).expanduser()
    argv = [
        python,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        str(repo / "inference" / "inference_oscar.py"),
        "--checkpoint",
        str(checkpoint),
        "--first-frame",
        _string(first_frame_path),
        "--start-frame",
        "0",
        "--prompt",
        _string(prompt),
        "--num-steps",
        str(int(num_steps)),
        "--guidance",
        str(float(guidance)),
        "--seed",
        str(int(seed)),
        "--num-frames",
        str(max(1, int(num_frames))),
        "--height",
        str(int(height)),
        "--width",
        str(int(width)),
        "--fps",
        str(float(fps)),
        "--output",
        str(output_video),
    ]
    if skeleton_video is not None:
        argv.extend(["--skeleton-video", str(skeleton_video)])
    return argv


def make_local_oscar_subprocess_generate(
    *,
    oscar_repo: str | Path,
    checkpoint: str | Path,
    python: str = "python",
    num_steps: int = 35,
    guidance: float = 6.0,
    height: int = 480,
    width: int = 640,
    fps: float = 15.0,
    timeout_seconds: float = 3600.0,
    run: Callable[..., Any],
    popen: Callable[..., Any] | None = None,
    process_group_signal: Callable[[int, int], None] = os.killpg,
    process_group_absent: Callable[[int], bool] = _oscar_process_group_absent,
    termination_grace_seconds: float = OSCAR_SUBPROCESS_TERMINATION_GRACE_SECONDS,
    gpu_query_run: Callable[..., Any] | None = None,
    gpu_parent_pid: Callable[[int], int | None] = _linux_process_parent_pid,
    gpu_host_to_local_pid_map: Callable[
        [], Mapping[int, int]
    ] = _linux_nvidia_host_to_local_pid_map,
    build_skeleton_video: Callable[[Sequence[Mapping[str, Any]], Path], Path | None] | None = None,
    extract_next_frame: Callable[[Path, Path], Path | None],
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Real per-step OSCAR-2B inference, for running ON a GPU pod that has the OSCAR repo +
    checkpoint. ``run`` (subprocess.run), ``build_skeleton_video`` (landmarks -> conditioning
    video), and ``extract_next_frame`` (output clip -> next-observation frame, e.g. via ffmpeg)
    are injected, so the whole wrapper is unit-testable without GPU or OSCAR installed.

    The real worker injects ``popen`` so the OSCAR process has an attributable root PID. The
    process starts in its own session so a timeout owns and terminates the entire torchrun process
    group, not just its launcher. While that process runs, GPU inventory and compute applications
    are sampled every 0.75 seconds. ``timeout_seconds`` bounds that real subprocess using the same
    provider-call timeout supplied by the CLI. Legacy/fake ``run`` injection remains synchronous
    and does not pretend to produce residency evidence.
    """
    repo = Path(oscar_repo).expanduser()

    def _oscar_generate(request: Mapping[str, Any]) -> dict[str, Any]:
        out_dir = Path(_string(request.get("output_dir"))).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        output_video = out_dir / "oscar_next_observation.mp4"
        landmarks = request.get("skeleton_landmarks") or []
        skeleton_trace_rows = request.get("skeleton_trace_rows") or []
        skeleton_input = skeleton_trace_rows or landmarks
        skeleton_video = (
            build_skeleton_video(skeleton_input, out_dir) if build_skeleton_video else None
        )
        stdout_log = out_dir / "oscar_subprocess_stdout.log"
        stderr_log = out_dir / "oscar_subprocess_stderr.log"
        if build_skeleton_video is not None and skeleton_video is None:
            stdout_log.write_text("", encoding="utf-8")
            stderr_log.write_text(
                "OSCAR inference skipped: projected skeleton conditioning is unavailable.\n",
                encoding="utf-8",
            )
            return {
                "status": "blocked",
                "blockers": ["oscar_per_step_projected_skeleton_conditioning_unavailable"],
                "generated_frame_path": "",
                "generated_video_path": "",
                "stdout_log_path": str(stdout_log),
                "stderr_log_path": str(stderr_log),
            }
        argv = build_oscar_inference_argv(
            python=python,
            oscar_repo=repo,
            checkpoint=checkpoint,
            first_frame_path=_string(request.get("reference_frame_path")),
            prompt=_normalize_oscar_robot_action_prompt(_string(request.get("task_prompt")))[0],
            num_frames=int(request.get("num_frames") or 8),
            num_steps=num_steps,
            guidance=guidance,
            seed=int(request.get("seed") or 42),
            height=height,
            width=width,
            fps=fps,
            output_video=output_video,
            skeleton_video=skeleton_video,
        )
        runtime_env = os.environ.copy()
        cudnn_lib_dir_value = _string(runtime_env.get("BLUEPRINT_OSCAR_CUDNN_LIB_DIR"))
        if cudnn_lib_dir_value:
            cudnn_lib_dir = Path(cudnn_lib_dir_value).expanduser()
            if (
                not cudnn_lib_dir.is_absolute()
                or not cudnn_lib_dir.is_dir()
                or not (cudnn_lib_dir / "libcudnn_graph.so.9").is_file()
            ):
                stdout_log.write_text("", encoding="utf-8")
                stderr_log.write_text(
                    "OSCAR inference skipped: configured cuDNN runtime is unavailable.\n",
                    encoding="utf-8",
                )
                return {
                    "status": "blocked",
                    "blockers": ["oscar_cudnn_runtime_directory_invalid"],
                    "generated_frame_path": "",
                    "generated_video_path": "",
                    "stdout_log_path": str(stdout_log),
                    "stderr_log_path": str(stderr_log),
                }
            existing_ld_library_path = _string(runtime_env.get("LD_LIBRARY_PATH"))
            runtime_env["LD_LIBRARY_PATH"] = ":".join(
                value for value in (str(cudnn_lib_dir), existing_ld_library_path) if value
            )
        residency_report: dict[str, Any] | None = None
        subprocess_timed_out = False
        timeout_cleanup_failed = False
        process_group_id: int | None = None
        process_group_term_sent = False
        process_group_kill_sent = False
        process_group_absent_verified = False
        subprocess_timeout_seconds = max(1.0, float(timeout_seconds))
        cleanup_grace_seconds = max(0.1, float(termination_grace_seconds))
        if popen is None:
            completed = run(
                argv,
                cwd=str(repo),
                capture_output=True,
                text=True,
                check=False,
                env=runtime_env,
            )
            stdout = str(getattr(completed, "stdout", "") or "")
            stderr = str(getattr(completed, "stderr", "") or "")
            returncode = getattr(completed, "returncode", 1)
        else:
            process = popen(
                argv,
                cwd=str(repo),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=runtime_env,
                start_new_session=True,
            )
            process_group_id = _positive_pid(getattr(process, "pid", None))
            sampler = _OscarGpuResidencySampler(
                output_dir=out_dir,
                oscar_pid=process_group_id,
                query_run=gpu_query_run or subprocess.run,
                parent_pid=gpu_parent_pid,
                host_to_local_pid_map=gpu_host_to_local_pid_map,
            )
            sampler.start()
            stdout = ""
            stderr = ""
            try:
                stdout, stderr = process.communicate(timeout=subprocess_timeout_seconds)
            except subprocess.TimeoutExpired as timeout_error:
                subprocess_timed_out = True

                def _timeout_output(value: Any) -> str:
                    if isinstance(value, bytes):
                        return value.decode("utf-8", errors="replace")
                    return str(value or "")

                stdout = _timeout_output(timeout_error.stdout)
                stderr = _timeout_output(timeout_error.stderr)

                if process_group_id is None:
                    timeout_cleanup_failed = True
                else:
                    try:
                        process_group_signal(process_group_id, signal.SIGTERM)
                        process_group_term_sent = True
                    except ProcessLookupError:
                        process_group_absent_verified = True
                    except OSError:
                        timeout_cleanup_failed = True

                terminate_communicate_timed_out = False
                try:
                    terminated_stdout, terminated_stderr = process.communicate(
                        timeout=cleanup_grace_seconds
                    )
                    stdout = str(terminated_stdout or stdout)
                    stderr = str(terminated_stderr or stderr)
                except subprocess.TimeoutExpired as terminate_timeout:
                    terminate_communicate_timed_out = True
                    stdout = _timeout_output(terminate_timeout.stdout) or stdout
                    stderr = _timeout_output(terminate_timeout.stderr) or stderr

                if process_group_id is not None and not terminate_communicate_timed_out:
                    try:
                        process_group_absent_verified = _wait_for_oscar_process_group_absence(
                            process_group_id,
                            timeout_seconds=cleanup_grace_seconds,
                            process_group_absent=process_group_absent,
                        )
                    except Exception:
                        timeout_cleanup_failed = True

                process_group_needs_kill = process_group_id is not None and (
                    terminate_communicate_timed_out or not process_group_absent_verified
                )
                if process_group_needs_kill:
                    try:
                        process_group_signal(process_group_id, signal.SIGKILL)
                        process_group_kill_sent = True
                    except ProcessLookupError:
                        process_group_absent_verified = True
                    except OSError:
                        timeout_cleanup_failed = True
                    try:
                        killed_stdout, killed_stderr = process.communicate(
                            timeout=cleanup_grace_seconds
                        )
                        stdout = str(killed_stdout or stdout)
                        stderr = str(killed_stderr or stderr)
                    except subprocess.TimeoutExpired as kill_timeout:
                        stdout = _timeout_output(kill_timeout.stdout) or stdout
                        stderr = _timeout_output(kill_timeout.stderr) or stderr
                        timeout_cleanup_failed = True
                    try:
                        process_group_absent_verified = _wait_for_oscar_process_group_absence(
                            process_group_id,
                            timeout_seconds=cleanup_grace_seconds,
                            process_group_absent=process_group_absent,
                        )
                    except Exception:
                        timeout_cleanup_failed = True
                if process_group_id is not None and not process_group_absent_verified:
                    timeout_cleanup_failed = True
                timeout_diagnostic = (
                    f"OSCAR inference timed out after {subprocess_timeout_seconds:.3f} seconds."
                )
                stderr = (
                    f"{stderr.rstrip()}\n{timeout_diagnostic}\n"
                    if stderr
                    else (f"{timeout_diagnostic}\n")
                )
            finally:
                stdout = str(stdout or "")
                stderr = str(stderr or "")
                residency_report = sampler.finalize(
                    runtime_diagnostics=f"{stdout}\n{stderr}",
                )
            returncode = getattr(process, "returncode", 1)
        stdout_log.write_text(stdout, encoding="utf-8")
        stderr_log.write_text(stderr, encoding="utf-8")
        residency_fields: dict[str, Any] = {}
        if residency_report is not None:
            residency_fields = {
                "oscar_gpu_residency_report_path": str(residency_report.get("report_path") or ""),
                "oscar_gpu_residency_samples_path": str(residency_report.get("samples_path") or ""),
                "oscar_gpu_residency": residency_report,
            }
        residency_blockers: list[str] = []
        first_real_transition = popen is not None and int(request.get("step_index") or 0) == 1
        if first_real_transition and not bool(
            residency_report and residency_report.get("proof_passed") is True
        ):
            residency_blockers.append("oscar_first_transition_gpu_residency_proof_absent")
            residency_blockers.extend(
                f"oscar_gpu_residency:{blocker}"
                for blocker in _string_list((residency_report or {}).get("blockers"))
            )
        if residency_report and residency_report.get("cuda_oom_detected") is True:
            residency_blockers.append("oscar_cuda_out_of_memory_detected")
        if residency_report and residency_report.get("nvidia_xid_detected") is True:
            residency_blockers.append("oscar_nvidia_xid_detected")
        if subprocess_timed_out:
            timeout_blockers = ["oscar_per_step_inference_timeout"]
            if timeout_cleanup_failed:
                timeout_blockers.append("oscar_per_step_inference_timeout_cleanup_failed")
            return {
                "status": "blocked",
                "blockers": sorted(set(timeout_blockers + residency_blockers)),
                "generated_frame_path": "",
                "generated_video_path": str(output_video) if output_video.is_file() else "",
                "stdout_log_path": str(stdout_log),
                "stderr_log_path": str(stderr_log),
                "oscar_subprocess_timed_out": True,
                "oscar_subprocess_timeout_seconds": subprocess_timeout_seconds,
                "oscar_subprocess_timeout_cleanup_failed": timeout_cleanup_failed,
                "oscar_subprocess_process_group_id": process_group_id,
                "oscar_subprocess_process_group_term_sent": process_group_term_sent,
                "oscar_subprocess_process_group_kill_sent": process_group_kill_sent,
                "oscar_subprocess_process_group_absent_verified": (process_group_absent_verified),
                **residency_fields,
            }
        if returncode != 0 or not output_video.is_file():
            return {
                "status": "blocked",
                "blockers": sorted(
                    set([f"oscar_per_step_inference_returncode_{returncode}"] + residency_blockers)
                ),
                "generated_frame_path": "",
                "generated_video_path": str(output_video) if output_video.is_file() else "",
                "stdout_log_path": str(stdout_log),
                "stderr_log_path": str(stderr_log),
                **residency_fields,
            }
        if residency_blockers:
            return {
                "status": "blocked",
                "blockers": sorted(set(residency_blockers)),
                "generated_frame_path": "",
                "generated_video_path": str(output_video),
                "stdout_log_path": str(stdout_log),
                "stderr_log_path": str(stderr_log),
                **residency_fields,
            }
        next_frame = extract_next_frame(output_video, out_dir)
        if not next_frame or not Path(next_frame).is_file():
            return {
                "status": "blocked",
                "blockers": ["oscar_per_step_next_frame_extraction_failed"],
                "generated_frame_path": "",
                "generated_video_path": str(output_video),
                "stdout_log_path": str(stdout_log),
                "stderr_log_path": str(stderr_log),
                **residency_fields,
            }
        return {
            "status": "completed",
            "generated_frame_path": str(next_frame),
            "generated_video_path": str(output_video),
            "stdout_log_path": str(stdout_log),
            "stderr_log_path": str(stderr_log),
            **residency_fields,
        }

    return _oscar_generate


def evaluate_isaac_manipulation_success(
    *,
    generated_at: str,
    status: str,
    proof: Mapping[str, Any],
    trace_rows: Sequence[Mapping[str, Any]],
    task_target_reached: bool,
    perception_target_prompts: Sequence[str],
) -> dict[str, Any]:
    def _count(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    learned_policy_requery_steps = _count(
        proof.get("fresh_learned_policy_requery_steps")
        or proof.get("learned_policy_requery_steps")
        or proof.get("learned_policy_requery_count")
        or proof.get("policy_requery_steps")
    )
    action_conditioned_steps = sum(
        1
        for row in trace_rows
        if (
            row.get("policy_action_conditioned_on_wam_generated_observation")
            or row.get("policy_action_from_wam_requery")
            or row.get("policy_requeried_fresh")
        )
    )
    fresh_oscar_provider_steps = _count(proof.get("fresh_oscar_provider_model_run_steps"))
    real_perception_steps = _count(proof.get("real_perception_backend_steps"))
    structural_loop_completed = status == "completed"
    registered_transition = _mapping(proof.get("registered_task_completion_transition"))
    transition_evidence, transition_evidence_blockers = _validate_transition_measurement_artifacts(
        registered_transition.get("validated_evidence_artifacts"),
        criterion_id=_string(registered_transition.get("criterion_id")).strip(),
        observable_transition=_string(registered_transition.get("observable_transition")).strip(),
        before_value=_finite_float(registered_transition.get("before_value")),
        after_value=_finite_float(registered_transition.get("after_value")),
        unit=_string(registered_transition.get("unit")).strip(),
        source_step_index=(
            registered_transition.get("source_step_index")
            if isinstance(registered_transition.get("source_step_index"), int)
            and not isinstance(registered_transition.get("source_step_index"), bool)
            else None
        ),
    )
    registered_transition_proven = bool(
        registered_transition.get("registered_transition_passed") is True
        and registered_transition.get("computed_transition_passed") is True
        and _mapping(registered_transition.get("registered_criterion"))
        and _finite_float(registered_transition.get("before_value")) is not None
        and _finite_float(registered_transition.get("after_value")) is not None
        and _finite_float(registered_transition.get("tolerance")) is not None
        and _string(registered_transition.get("unit")).strip()
        and isinstance(registered_transition.get("source_step_index"), int)
        and not isinstance(registered_transition.get("source_step_index"), bool)
        and not _string_list(registered_transition.get("validation_blockers"))
        and transition_evidence
        and not transition_evidence_blockers
    )
    success_proven = bool(structural_loop_completed and registered_transition_proven)
    if success_proven:
        reason = (
            "A registered task criterion with finite before/after measurements and "
            "hash-verified evidence proved the simulated manipulation transition."
        )
    elif learned_policy_requery_steps > 0 or action_conditioned_steps > 0:
        reason = (
            "Loop ran with live learned-policy requeries on WAM-generated observations, "
            "but no task-success signal proved the manipulation."
        )
    elif structural_loop_completed:
        reason = (
            "Loop completed structurally (deterministic/no learned requery); "
            "no manipulation success proven."
        )
    else:
        reason = "Loop did not complete a learned-policy requery or produce a manipulation-success signal."
    prompt = next((str(item) for item in perception_target_prompts if str(item).strip()), "")
    return {
        "schema_version": "isaac_manipulation_success_evaluator_results.v1",
        "generated_at": generated_at,
        "status": "completed",
        "simulator_backend": "isaac",
        "question": prompt or "Did the target manipulation succeed?",
        "answer": "yes" if success_proven else "not_proven",
        "did_target_manipulation_succeed": bool(success_proven),
        "manipulation_success_proven": bool(success_proven),
        "success_proof_separate_from_structural_loop_proof": True,
        "structural_loop_completed": structural_loop_completed,
        "kinematic_route_reached_is_not_manipulation_success": True,
        "task_target_reached": bool(task_target_reached),
        "learned_policy_requery_steps": learned_policy_requery_steps,
        "action_conditioned_steps": action_conditioned_steps,
        "fresh_oscar_provider_model_run_steps": fresh_oscar_provider_steps,
        "real_perception_backend_steps": real_perception_steps,
        "registered_task_completion_transition": registered_transition or None,
        "registered_task_completion_transition_proven": registered_transition_proven,
        "registered_task_completion_evidence": transition_evidence,
        "registered_task_completion_evidence_blockers": transition_evidence_blockers,
        "reason": reason,
        "raw_secret_values_recorded": False,
    }


def _wam_consistency_command(explicit_command: str | None) -> str:
    return _string(explicit_command) or _string(os.getenv(WAM_CONSISTENCY_COMMAND_ENV))


def _wam_episode_consistency_requested(
    *,
    explicit_command: str | None,
    allow_wam_consistency_scoring: bool,
) -> bool:
    return bool(allow_wam_consistency_scoring or _wam_consistency_command(explicit_command))


def _wam_success_label_command(explicit_command: str | None) -> str:
    return _string(explicit_command) or _string(os.getenv(WAM_SUCCESS_LABEL_COMMAND_ENV))


def _closed_loop_generated_episode_artifacts(
    *,
    output_dir: Path,
    generated_at: str,
    trace_rows: Sequence[Mapping[str, Any]],
    initial_frame_path: str,
    policy_id: str,
    task_prompts: Sequence[str],
    target: Sequence[float],
) -> dict[str, Any]:
    step_videos: list[dict[str, Any]] = []
    blockers: list[str] = []
    for trace_position, row in enumerate(trace_rows, start=1):
        video_path = _string(row.get("wam_generated_video"))
        resolved_video = Path(video_path).expanduser() if video_path else None
        video_present = bool(resolved_video and resolved_video.is_file())
        video_sha256 = _file_sha256(resolved_video) if video_present and resolved_video else None
        if not video_path:
            blockers.append(f"closed_loop_step_video_path_missing:{trace_position}")
        elif not video_present:
            blockers.append(f"closed_loop_step_video_file_missing:{trace_position}")
        step_videos.append(
            {
                "step_index": row.get("step_index"),
                "generated_video_path": video_path,
                "generated_video_present": video_present,
                "generated_video_sha256": video_sha256,
                "generated_frame_path": row.get("wam_generated_frame"),
                "source_observation_frame_path": row.get("source_observation_frame"),
                "policy_action": row.get("policy_action"),
                "policy_action_source": row.get("policy_action_source"),
                "policy_requeried_fresh": bool(row.get("policy_requeried_fresh")),
            }
        )
    present_videos = [row for row in step_videos if row.get("generated_video_present")]
    ordered_step_videos = sorted(
        present_videos,
        key=lambda row: int(row.get("step_index") or 0),
    )
    ordered_step_indices = [int(row.get("step_index") or 0) for row in ordered_step_videos]
    expected_step_indices = list(range(1, len(trace_rows) + 1))
    episode_order_verified = bool(
        ordered_step_videos
        and len(step_videos) == len(trace_rows)
        and len(ordered_step_videos) == len(step_videos)
        and ordered_step_indices == expected_step_indices
        and all(_is_sha256(row.get("generated_video_sha256")) for row in ordered_step_videos)
    )
    selected = ordered_step_videos[-1] if episode_order_verified else {}
    selected_video = _string(selected.get("generated_video_path"))
    if not selected_video:
        blockers.append("missing_generated_video_for_closed_loop_success_review")
    if trace_rows and not episode_order_verified:
        blockers.append("closed_loop_episode_order_not_verified")
    blockers = sorted(set(blockers))
    rollouts = (
        [
            {
                "rollout_id": "oscar_isaac_closed_loop_episode",
                "scenario_eval_run_id": "isaac_closed_loop_episode",
                "policy_id": policy_id,
                "model_candidate": "oscar_2b_per_step",
                "generated_video_path": selected_video,
                "generated_frame_path": selected.get("generated_frame_path"),
                "source_observation_frame_path": initial_frame_path,
                "final_generated_frame_path": selected.get("generated_frame_path"),
                "selected_review_video_step_index": selected.get("step_index"),
                "step_video_count": len(present_videos),
                "ordered_step_videos": ordered_step_videos,
                "ordered_step_indices": ordered_step_indices,
                "episode_order_verified": episode_order_verified,
                "review_media_scope": "full_ordered_episode",
                "task_target_position_xyz": [round(float(c), 6) for c in target],
                "task_prompt": next((prompt for prompt in task_prompts if prompt), ""),
                "generated_step_videos": step_videos,
            }
        ]
        if selected_video and episode_order_verified
        else []
    )
    manifest = {
        "schema_version": "closed_loop_generated_episode_manifest.v1",
        "generated_at": generated_at,
        "status": "completed" if selected_video and episode_order_verified else "blocked",
        "source_initial_site_capture_frame_path": initial_frame_path,
        "step_video_count": len(present_videos),
        "ordered_step_videos": ordered_step_videos,
        "ordered_step_indices": ordered_step_indices,
        "expected_step_indices": expected_step_indices,
        "episode_order_verified": episode_order_verified,
        "review_media_scope": "full_ordered_episode",
        "selected_review_video_path": selected_video or None,
        "selected_review_video_step_index": selected.get("step_index"),
        "generated_step_videos": step_videos,
        "rollouts": rollouts,
        "blockers": blockers,
        "sim_only_constraint": {
            "real_world_data_allowed": "site_capture_only",
            "source_initial_frame_is_site_capture_input": True,
            "generated_videos_are_model_outputs": True,
            "physical_robot_rollout_used": False,
        },
        "claim_boundary": {
            "generated_episode_video_is_model_derived_support_media": True,
            "generated_episode_video_is_not_raw_robot_evidence": True,
            "selected_review_video_is_not_task_success_without_success_label": True,
            "real_world_task_success_proven": False,
            "physical_robot_readiness_proven": False,
        },
        "raw_secret_values_recorded": False,
    }
    manifest_path = output_dir / "closed_loop_generated_episode_manifest.json"
    results_path = output_dir / "closed_loop_generated_episode_results.json"
    write_json(manifest_path, manifest)
    write_json(
        results_path,
        {
            "schema_version": "closed_loop_generated_episode_results.v1",
            "generated_at": generated_at,
            "status": manifest["status"],
            "rollouts": rollouts,
            "blockers": blockers,
            "claim_boundary": manifest["claim_boundary"],
        },
    )
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "results_path": str(results_path),
    }


def _score_closed_loop_generated_video_success(
    *,
    output_dir: Path,
    generated_at: str,
    episode_artifacts: Mapping[str, Any],
    task_prompts: Sequence[str],
    command: str | None,
    allow_wam_success_labeling: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    success_dir = output_dir / "generated_video_success"
    ensure_dir(success_dir)
    rollouts = [
        dict(item)
        for item in episode_artifacts.get("rollouts", []) or []
        if isinstance(item, Mapping)
    ]
    visual_smoke = visual_smoke_generated_rollouts_for_review(
        rollouts=rollouts,
        output_dir=success_dir / "visual_smoke",
        generated_at=generated_at,
        require_review_quality_profile=False,
    )
    visual_rollout_useful = bool(
        _mapping(visual_smoke.get("claim_boundary")).get(
            "visual_rollout_useful_for_task_success_review"
        )
    )
    visual_smoke_path = success_dir / "wam_generated_rollout_visual_smoke.json"
    write_json(visual_smoke_path, visual_smoke)
    request_path = success_dir / "wam_success_label_request.json"
    output_path = success_dir / WAM_SUCCESS_LABEL_COMMAND_OUTPUT
    task_prompt = next((prompt for prompt in task_prompts if prompt), "")
    request = {
        "schema_version": "wam_success_label_request.v1",
        "generated_at": generated_at,
        "status": "ready_for_vlm_judge"
        if rollouts and visual_rollout_useful
        else "blocked_generated_rollout_visual_quality"
        if rollouts
        else "blocked_missing_generated_rollout",
        "source_isaac_closed_loop_output_dir": str(output_dir),
        "closed_loop_generated_episode_manifest": episode_artifacts.get("manifest_path"),
        "closed_loop_generated_episode_results": episode_artifacts.get("results_path"),
        "generated_rollout_visual_smoke": str(visual_smoke_path),
        "generated_rollout_visual_smoke_status": _string(visual_smoke.get("status")),
        "generated_rollout_visually_useful_for_success_review": visual_rollout_useful,
        "rollouts": rollouts,
        "inference_input_manifest_sha256": success_label_inference_input_sha256(
            rollouts,
            criterion_ids=(
                "end_effector_reaches_target",
                "target_state_change_visible",
                "robot_caused_target_motion",
            ),
        ),
        "task_prompts": [
            {
                "scenario_eval_run_id": "isaac_closed_loop_episode",
                "task_prompt": task_prompt,
                "task_id": "isaac_g1_oscar_per_step_closed_loop",
            }
        ],
        "success_label_contract": {
            "expected_output_path": str(output_path),
            "required_top_level_keys": ["labels"],
            "label_required_keys": [
                "rollout_id",
                "success",
                "confidence",
                "rationale",
                "criterion_results",
            ],
            "minimum_calibrated_confidence": 0.8,
            "full_ordered_episode_required": True,
            "success_requires": [
                "The visible robot end effector reaches the task-relevant target.",
                "The target object or control visibly changes into the requested state.",
                "The state change is causally plausible from robot motion.",
                "Ambiguous, occluded, or prompt-only evidence fails closed.",
            ],
        },
        "sim_only_constraint": {
            "real_world_data_allowed": "site_capture_only",
            "source_initial_frame_is_site_capture_input": True,
            "judge_input_is_generated_video": True,
            "physical_robot_rollout_used": False,
        },
        "claim_boundary": {
            "judge_input_is_generated_video_not_raw_robot_evidence": True,
            "judge_success_label_does_not_prove_forward_inverse_consistency": True,
            "judge_success_label_does_not_prove_real_world_task_success": True,
            "judge_success_label_does_not_prove_physical_robot_readiness": True,
            "raw_credentials_written_to_artifacts": False,
        },
    }
    write_json(request_path, request)

    configured_command = _wam_success_label_command(command)
    label_blockers: list[str] = []
    command_result: dict[str, Any] | None = None
    command_payload: dict[str, Any] = {}
    if not rollouts:
        label_blockers = ["missing_generated_video_for_success_label"]
    elif not visual_rollout_useful:
        label_blockers = _string_list(visual_smoke.get("blockers")) or [
            "generated_rollout_not_visually_useful_for_success_review"
        ]
    elif allow_wam_success_labeling or configured_command:
        if not _wam_consistency_env_truthy(WAM_SUCCESS_LABEL_GATE_ENV):
            label_blockers.append(f"missing_env_{WAM_SUCCESS_LABEL_GATE_ENV}")
        if not allow_wam_success_labeling:
            label_blockers.append("missing_cli_allow_wam_success_labeling")
        if not configured_command:
            label_blockers.append("missing_wam_success_label_command")
        if not label_blockers:
            command_payload, command_result = _run_wam_success_label_command(
                command=configured_command,
                input_path=request_path,
                output_path=output_path,
                timeout_seconds=timeout_seconds,
            )
            if command_result.get("status") != "completed":
                label_blockers.extend(
                    _string_list(command_result.get("blockers"))
                    or ["wam_success_label_command_blocked"]
                )
    else:
        label_blockers = ["requires_wam_success_review"]

    if command_payload and not label_blockers:
        success_labels = _normalize_wam_success_labels(
            command_payload=command_payload,
            rollouts=rollouts,
            generated_at=generated_at,
            visual_smoke_status=_string(visual_smoke.get("status")),
            visual_rollout_useful=visual_rollout_useful,
        )
        label_blockers = _string_list(success_labels.get("blockers"))
    else:
        success_labels = {
            "schema_version": "wam_success_labels.v1",
            "generated_at": generated_at,
            "status": "blocked" if not rollouts or not visual_rollout_useful else "requires_review",
            "wam_success_label_from_generated_video": False,
            "visual_smoke_status": _string(visual_smoke.get("status")),
            "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
            "review_grade_visual_evidence_available": visual_rollout_useful,
            "review_grade_success_labels": False,
            "label_count": 0,
            "labels": [],
            "blockers": label_blockers,
            "command_result": command_result,
            "human_review_required": bool(rollouts and visual_rollout_useful),
            "claim_boundary": {
                "success_label_is_from_generated_video_not_physical_robot": True,
                "success_label_requires_passed_visual_smoke": True,
                "success_label_does_not_prove_forward_inverse_consistency": True,
                "success_label_does_not_prove_real_world_task_success": True,
                "raw_credentials_written_to_artifacts": False,
                "secret_hashes_written_to_artifacts": False,
            },
        }
    if command_result is not None:
        success_labels["command_result"] = command_result
    labels = [
        dict(item) for item in success_labels.get("labels", []) or [] if isinstance(item, Mapping)
    ]
    generated_video_success_label_passed = bool(
        success_labels.get("status") == "completed"
        and success_labels.get("review_grade_success_labels")
        and labels
        and all(label.get("review_task_success") is True for label in labels)
    )
    write_json(success_dir / "wam_success_labels.json", success_labels)
    return {
        "schema_version": "closed_loop_generated_video_success.v1",
        "generated_at": generated_at,
        "status": "completed" if generated_video_success_label_passed else "not_proven",
        "request_path": str(request_path),
        "success_labels_path": str(success_dir / "wam_success_labels.json"),
        "visual_smoke_path": str(visual_smoke_path),
        "command_output_path": str(output_path),
        "success_label_judge_configured": bool(configured_command),
        "success_label_judge_ran": bool(
            isinstance(command_result, Mapping) and command_result.get("status") == "completed"
        ),
        "wam_success_label_from_generated_video": bool(
            success_labels.get("wam_success_label_from_generated_video")
        ),
        "generated_video_success_label_passed": generated_video_success_label_passed,
        "simulated_manipulation_success_shown": generated_video_success_label_passed,
        "real_world_task_success_proven": False,
        "physical_robot_readiness_proven": False,
        "success_labels": success_labels,
        "blockers": label_blockers
        if not generated_video_success_label_passed
        else _string_list(success_labels.get("blockers")),
        "claim_boundary": {
            "simulated_manipulation_success_shown_requires_generated_video_label": True,
            "generated_video_success_label_is_sim_only_support": True,
            "real_world_task_success_proven": False,
            "physical_robot_readiness_proven": False,
            "forward_inverse_consistency_proven_by_success_label": False,
        },
    }


GENERATED_CLIP_SEED_CORRELATION_FLOOR = 0.5
GENERATED_CLIP_PATCH_CORRELATION_FLOOR = 0.5
GENERATED_CLIP_MAX_LAPLACIAN_VARIANCE_RATIO = 3.0


def generated_clip_coherence(
    video_path: str | Path | None,
    *,
    correlation_floor: float = GENERATED_CLIP_SEED_CORRELATION_FLOOR,
) -> dict[str, Any]:
    """Measure how long a generated clip stays visually anchored to its seed.

    2026-07-06 T4 finding: OSCAR clips collapsed to noise within 4-9 of 81
    frames, and the contrast/edge signal gate could not see it (noise has
    plenty of edges). Normalized correlation of each frame against frame 0
    catches drift-to-garbage cheaply and scene-neutrally. ``coherent_horizon``
    is 1 + the count of leading frames with correlation >= floor; a horizon of
    1 means not even the frame the loop feeds forward is anchored to the seed.
    Fail-open to ``not_measured`` (never fabricates a score) when cv2 or the
    clip is unavailable.
    """
    resolved = Path(video_path).expanduser() if video_path else None
    if resolved is None or not resolved.is_file():
        return {"status": "not_measured", "blockers": ["generated_video_missing"]}
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {"status": "not_measured", "blockers": ["cv2_unavailable"]}
    capture = cv2.VideoCapture(str(resolved))
    seed = None
    seed_full = None
    seed_laplacian_variance = None
    correlations: list[float] = []
    patch_correlation_medians: list[float] = []
    laplacian_variance_ratios: list[float] = []

    def _patch_correlation_median(seed_gray, current_gray) -> float:
        height, width = seed_gray.shape
        rows = 6 if height >= 240 else 3
        columns = 8 if width >= 320 else 4
        values: list[float] = []
        for row in range(rows):
            y0, y1 = row * height // rows, (row + 1) * height // rows
            for column in range(columns):
                x0, x1 = column * width // columns, (column + 1) * width // columns
                left = seed_gray[y0:y1, x0:x1].reshape(-1)
                right = current_gray[y0:y1, x0:x1].reshape(-1)
                left_centered = left - left.mean()
                right_centered = right - right.mean()
                denominator = float(
                    np.sqrt(
                        (left_centered * left_centered).sum()
                        * (right_centered * right_centered).sum()
                    )
                )
                if denominator > 1e-6:
                    values.append(float((left_centered * right_centered).sum() / denominator))
        return float(np.median(values)) if values else 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32")
            gray = cv2.resize(gray_full, (80, 60))
            if seed is None:
                seed = gray - gray.mean()
                seed_full = gray_full
                seed_laplacian_variance = float(cv2.Laplacian(gray_full, cv2.CV_32F).var())
                continue
            centered = gray - gray.mean()
            denominator = float(np.sqrt((centered * centered).sum() * (seed * seed).sum()))
            correlations.append(
                float((centered * seed).sum() / denominator) if denominator else 0.0
            )
            patch_correlation_medians.append(
                _patch_correlation_median(seed_full, gray_full) if seed_full is not None else 0.0
            )
            current_laplacian_variance = float(cv2.Laplacian(gray_full, cv2.CV_32F).var())
            laplacian_variance_ratios.append(
                current_laplacian_variance / max(float(seed_laplacian_variance or 0.0), 1e-6)
            )
    finally:
        capture.release()
    if seed is None or not correlations:
        return {
            "status": "not_measured",
            "blockers": ["generated_video_unreadable_or_single_frame"],
        }
    horizon = 1
    for correlation, patch_median, laplacian_ratio in zip(
        correlations,
        patch_correlation_medians,
        laplacian_variance_ratios,
        strict=True,
    ):
        if (
            correlation < float(correlation_floor)
            or patch_median < GENERATED_CLIP_PATCH_CORRELATION_FLOOR
            or laplacian_ratio > GENERATED_CLIP_MAX_LAPLACIAN_VARIANCE_RATIO
        ):
            break
        horizon += 1
    return {
        "status": "measured",
        "frame_count": len(correlations) + 1,
        "seed_correlation_floor": float(correlation_floor),
        "patch_correlation_floor": GENERATED_CLIP_PATCH_CORRELATION_FLOOR,
        "maximum_laplacian_variance_ratio": (GENERATED_CLIP_MAX_LAPLACIAN_VARIANCE_RATIO),
        "coherent_horizon_frames": horizon,
        "first_frame_correlation": round(correlations[0], 6),
        "first_frame_patch_correlation_median": round(patch_correlation_medians[0], 6),
        "first_frame_laplacian_variance_ratio": round(laplacian_variance_ratios[0], 6),
        "final_frame_correlation": round(correlations[-1], 6),
        "min_correlation": round(min(correlations), 6),
        "min_patch_correlation_median": round(min(patch_correlation_medians), 6),
        "max_laplacian_variance_ratio": round(max(laplacian_variance_ratios), 6),
        "early_frame_artifact_detected": horizon == 1,
        "blockers": [],
        "claim_boundary": (
            "Global and tiled seed correlation plus high-frequency growth "
            "measure visual drift and artifact collapse only; they are not "
            "semantic fidelity, physics plausibility, or task-success evidence."
        ),
    }


def run_oscar_isaac_closed_loop(
    *,
    output_dir: str | Path,
    start_frame_path: str | Path,
    route_points: Sequence[Sequence[float]],
    wam_generate_next: WamGenerateNext,
    steps: int,
    probe_collision: Callable[[Sequence[float], float], int] | None = None,
    harness_backend_kind: str = "fixture",
    harness_backend_command: str | Sequence[str] | None = None,
    allow_external_backend: bool = False,
    backend_timeout_seconds: int = 600,
    policy_id: str = "blueprint_default_walk_to_target_smoke_policy",
    task_prompt: str | None = None,
    generated_at: str | None = None,
    require_fresh_oscar_provider: bool = False,
    require_real_perception_backend: bool = False,
    require_sam3_completed: bool = False,
    require_da3_completed: bool = False,
    perception_target_prompts: Sequence[str] | None = None,
    wam_backend_id: str = "oscar_wam",
    wam_backend_readiness: Mapping[str, Any] | None = None,
    require_fresh_learned_policy_requery: bool = False,
    require_action_derived_skeleton_conditioning: bool = False,
    clean_frame_reanchor_interval: int = 0,
    policy_endpoint: PolicyEndpoint | None = None,
    initial_observation_evidence: Mapping[str, Any] | None = None,
    wam_consistency_command: str | None = None,
    allow_wam_consistency_scoring: bool = False,
    wam_consistency_timeout_seconds: float | None = None,
    require_forward_inverse_consistency: bool = False,
    require_synchronized_calibrated_multiview: bool = False,
    wam_success_label_command: str | None = None,
    allow_wam_success_labeling: bool = False,
    require_generated_video_success_label: bool = False,
    wam_success_label_timeout_seconds: float | None = None,
    min_coherent_horizon_frames: int = 0,
    stop_on_task_completion: bool = False,
    stop_on_unsafe_stance: bool = False,
    stop_on_no_progress: bool = False,
    no_progress_patience_steps: int = DEFAULT_NO_PROGRESS_PATIENCE_STEPS,
    minimum_task_progress_fraction: float = 0.02,
    min_steps: int = 1,
    task_success_contract: Mapping[str, Any] | None = None,
    task_completion_evaluator: TaskCompletionEvaluator | None = None,
    attempt_input_manifest: str | Path | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    resolved_out = Path(output_dir).expanduser().resolve()
    ensure_dir(resolved_out)
    harness_dir = resolved_out / "wam_derived_observation_harness"
    route = [tuple(float(c) for c in point) for point in route_points]
    if not route:
        return {
            "schema_version": LOOP_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "blockers": ["blocked_empty_route"],
        }
    target = route[-1]
    resolved_task_prompt = _string(task_prompt).strip()
    cleaned_target_prompts = [
        prompt for prompt in (_string(item) for item in (perception_target_prompts or [])) if prompt
    ]
    completion_contract = _mapping(task_success_contract)
    learned_endpoint_configured = policy_endpoint is not None
    strict_fresh_learned_policy_requery = bool(
        require_fresh_learned_policy_requery or learned_endpoint_configured
    )
    strict_action_derived_skeleton_conditioning = bool(
        require_action_derived_skeleton_conditioning or learned_endpoint_configured
    )
    task_kind = _string(completion_contract.get("task_kind")).strip().lower() or (
        "manipulation" if cleaned_target_prompts else "navigation_smoke"
    )
    manipulation_task = task_kind not in {"navigation", "navigation_smoke"}
    task_completion_results: list[dict[str, Any]] = []
    proven_task_completion_transition: dict[str, Any] = {}
    bounded_steps = max(1, int(steps))

    policy = DeterministicWalkToTargetPolicy()
    policy.reset({"route_points": list(route), "start": route[0], "target": target})
    oracle = probe_collision or (lambda pose, yaw: 0)

    current_frame = str(Path(start_frame_path).expanduser().resolve())
    initial_clean_frame = current_frame
    try:
        configured_reanchor_interval = int(clean_frame_reanchor_interval)
    except (TypeError, ValueError):
        configured_reanchor_interval = 0
    effective_reanchor_interval = max(
        0,
        configured_reanchor_interval
        or _int_env(PERSISTENT_WAM_CLEAN_FRAME_REANCHOR_INTERVAL_ENV, 0),
    )
    clean_frame_reanchor_events: list[dict[str, Any]] = []
    action_history: list[dict[str, Any]] = []
    clip_coherence_rows: list[dict[str, Any]] = []
    episode_termination_reason: str | None = None
    task_completed_early = False
    task_progress_history: list[dict[str, Any]] = []
    best_task_progress: float | None = None
    last_meaningful_approach_step: int | None = None
    last_meaningful_task_progress_step: int | None = None
    step_records: list[dict[str, Any]] = []
    adapter_reports: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    proof_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    pending_policy_endpoint_action: dict[str, Any] | None = None
    pending_policy_endpoint_source_step: int | None = None
    previous_endpoint_action_signature: str | None = None
    learned_policy_requery_count = 0
    terminal_observation_policy_requery_count = 0
    policy_action_changed_count = 0
    action_derived_skeleton_conditioned_steps = 0
    conditioning_action_by_evidence_sha256: dict[str, str] = {}
    action_conditioning_evidence_rows: list[dict[str, Any]] = []
    current_generated_robot_state: dict[str, Any] = {}
    consistency_results: list[dict[str, Any]] = []
    consistency_requested = bool(
        require_forward_inverse_consistency
        or _wam_episode_consistency_requested(
            explicit_command=wam_consistency_command,
            allow_wam_consistency_scoring=allow_wam_consistency_scoring,
        )
    )

    # The sealed manipulation path must query GR00T on the initial real frame.
    # Waiting until after the first WAM transition would make action zero a
    # deterministic fixture action and poison every downstream identity binding.
    if policy_endpoint is not None:
        try:
            validated_initial = _validated_initial_policy_observation(
                initial_observation_evidence,
                start_frame_path=current_frame,
            )
            initial_observation = {
                "schema_version": "oscar_initial_real_policy_observation.v1",
                "frame_path": validated_initial["frame_path"],
                "frame_sha256": validated_initial["sha256"],
                "camera_frame_path": validated_initial["frame_path"],
                "source_observation_frame": validated_initial["frame_path"],
                "observation_kind": "initial_real_observation",
                "camera_role": validated_initial["camera_role"],
                "viewpoint_mode": validated_initial["viewpoint_mode"],
                "mount_motion_model": validated_initial["mount_motion_model"],
                "gaze_motion_model": validated_initial["gaze_motion_model"],
                "policy_observation_eligible": True,
                "third_person_overview_included": False,
                "visual_observation": {
                    "camera_frame_path": validated_initial["frame_path"],
                    "camera_frame_sha256": validated_initial["sha256"],
                    "camera_role": validated_initial["camera_role"],
                    "viewpoint_mode": validated_initial["viewpoint_mode"],
                    "mount_motion_model": validated_initial["mount_motion_model"],
                    "gaze_motion_model": validated_initial["gaze_motion_model"],
                    "policy_observation_eligible": True,
                    "third_person_overview_included": False,
                    "camera_contract": validated_initial["camera_contract"],
                },
                "task_prompt": resolved_task_prompt or None,
                "route_target_xyz": list(target),
                "generated_robot_state": {},
            }
            pending_policy_endpoint_action = dict(policy_endpoint(initial_observation, [], 0) or {})
            if not pending_policy_endpoint_action:
                raise RuntimeError("initial_policy_action_empty")
            pending_policy_endpoint_source_step = 0
            previous_endpoint_action_signature = _endpoint_action_signature(
                pending_policy_endpoint_action
            )
            learned_policy_requery_count += 1
        except Exception as exc:  # noqa: BLE001
            detail = _string(exc).strip()
            if detail.startswith("initial_policy_observation_"):
                blockers.append(detail)
            else:
                blockers.append(f"initial_learned_policy_query_failed:{type(exc).__name__}")

    if policy_endpoint is not None and pending_policy_endpoint_action is None:
        # Fail closed: substituting DeterministicWalkToTargetPolicy for the sealed
        # manipulation policy poisons identity bindings and, in production,
        # crashes FK conditioning (unitree_g1_sonic_action_missing). No step runs.
        blockers.append("initial_learned_policy_action_unavailable_fail_closed")
        bounded_steps = 0

    for step_index in range(1, bounded_steps + 1):
        # 1. policy acts
        decision = policy.step(
            StepContext(step=step_index - 1, num_steps=bounded_steps, probe_collision=oracle)
        )
        sim_time_s = round((step_index - 1) * 0.02, 9)
        action = action_record(
            decision=decision, step=step_index - 1, sim_time_s=sim_time_s, target=target
        )
        policy_action_from_wam_requery = False
        if pending_policy_endpoint_action is not None:
            action = _action_record_from_policy_endpoint(
                base_action=action,
                endpoint_action=pending_policy_endpoint_action,
                requery_source_step_index=pending_policy_endpoint_source_step or (step_index - 1),
                source_observation_kind=(
                    "initial_real_observation"
                    if step_index == 1 and pending_policy_endpoint_source_step == 0
                    else "wam_generated_observation"
                ),
            )
            policy_action_from_wam_requery = True
            pending_policy_endpoint_action = None
            pending_policy_endpoint_source_step = None
        else:
            action.setdefault("policy_requeried_on_generated_observation", False)
            action.setdefault(
                "policy_action_source", "isaac_g1_policy.DeterministicWalkToTargetPolicy"
            )
        policy_requeried_fresh = bool(
            action.get("policy_requeried_fresh")
            or action.get("policy_requeried_on_generated_observation")
        )
        action_history.append(action)

        # 2. WAM generates the NEXT observation conditioned on the action
        wam_output = dict(
            wam_generate_next(current_frame, action, step_index, list(action_history)) or {}
        )
        multiview_validation = validate_synchronized_multiview(
            _mapping(wam_output.get("synchronized_multiview"))
        )
        if (
            require_synchronized_calibrated_multiview
            and multiview_validation.get("status") != "validated"
        ):
            blockers.extend(
                f"multiview_step_{step_index}:{blocker}"
                for blocker in _string_list(multiview_validation.get("blockers"))
            )
            break
        action_conditioning_blockers: list[str] = []
        if strict_action_derived_skeleton_conditioning and policy_action_from_wam_requery:
            action_conditioning_blockers = _action_conditioning_blockers(
                action=action,
                wam_output=wam_output,
            )
            if action_conditioning_blockers:
                blockers.extend(
                    f"action_conditioning_step_{step_index}:{blocker}"
                    for blocker in action_conditioning_blockers
                )
                break
            learned_action_signature = _endpoint_action_signature(
                _mapping(action.get("learned_policy_endpoint_action")) or action
            )
            conditioning_evidence_sha256 = _conditioning_evidence_sha256(wam_output)
            prior_action_signature = conditioning_action_by_evidence_sha256.get(
                conditioning_evidence_sha256
            )
            if (
                prior_action_signature is not None
                and prior_action_signature != learned_action_signature
            ):
                blockers.append(
                    f"action_conditioning_step_{step_index}:"
                    "fresh_action_conditioning_not_action_differentiated"
                )
                break
            conditioning_action_by_evidence_sha256[conditioning_evidence_sha256] = (
                learned_action_signature
            )
            action_conditioning_evidence_rows.append(
                {
                    "step_index": step_index,
                    "source_action_sha256": _canonical_sha256(action),
                    "learned_action_signature_sha256": hashlib.sha256(
                        learned_action_signature.encode("utf-8")
                    ).hexdigest(),
                    "conditioning_evidence_sha256": conditioning_evidence_sha256,
                }
            )
            action_derived_skeleton_conditioned_steps += 1
            current_generated_robot_state = _mapping(wam_output.get("generated_robot_state"))
        wam_status = _string(wam_output.get("status"))
        if wam_status and wam_status != "completed":
            step_blockers = _string_list(wam_output.get("blockers")) or [
                f"wam_generation_status_{wam_status}"
            ]
            blockers.extend(
                f"blocked_wam_generation_at_step_{step_index}:{blocker}"
                for blocker in step_blockers
            )
            break
        generated_frame = _string(wam_output.get("generated_frame_path"))
        if not generated_frame or not Path(generated_frame).is_file():
            blockers.append(f"blocked_wam_generation_missing_frame_at_step_{step_index}")
            break
        clip_coherence = generated_clip_coherence(wam_output.get("generated_video_path"))
        clip_coherence_rows.append({"step_index": step_index, **clip_coherence})
        if (
            int(min_coherent_horizon_frames or 0) > 0
            and clip_coherence.get("status") == "measured"
            and int(clip_coherence.get("coherent_horizon_frames") or 0)
            < int(min_coherent_horizon_frames)
        ):
            blockers.append(
                "blocked_generated_clip_coherence_below_floor_at_step_"
                f"{step_index}:horizon_"
                f"{clip_coherence.get('coherent_horizon_frames')}"
                f"_lt_{int(min_coherent_horizon_frames)}"
            )
            break
        wam_provider_payload = (
            wam_output.get("provider_payload")
            if isinstance(wam_output.get("provider_payload"), Mapping)
            else {}
        )
        fresh_oscar_provider = bool(
            wam_output.get("fresh_provider_model_run_claimed")
            or _provider_payload_proves_fresh_model(wam_provider_payload)
        )
        if require_fresh_oscar_provider and not fresh_oscar_provider:
            blockers.append(f"fresh_oscar_provider_model_run_not_proven_at_step_{step_index}")
        consistency_result: dict[str, Any] = {}
        consistency_early_termination = False
        if consistency_requested:
            consistency_result = _score_closed_loop_step_episode_consistency(
                output_dir=resolved_out,
                generated_at=generated,
                step_index=step_index,
                policy_id=policy_id,
                source_frame_path=current_frame,
                generated_frame_path=generated_frame,
                wam_output=wam_output,
                action=action,
                action_history=action_history,
                task_prompts=cleaned_target_prompts,
                wam_consistency_command=wam_consistency_command,
                allow_wam_consistency_scoring=allow_wam_consistency_scoring,
                require_strict_action_aware_consistency=bool(require_forward_inverse_consistency),
                timeout_seconds=(
                    float(wam_consistency_timeout_seconds)
                    if wam_consistency_timeout_seconds is not None
                    else float(backend_timeout_seconds)
                ),
            )
            replay_blockers = cross_step_action_motion_replay_blockers(
                [*consistency_results, consistency_result]
            )
            if replay_blockers:
                consistency_result["blockers"] = sorted(
                    set(_string_list(consistency_result.get("blockers")) + replay_blockers)
                )
                consistency_result["forward_inverse_consistency_proven"] = False
                consistency_result["early_termination_recommended"] = True
            consistency_results.append(consistency_result)
            consistency_early_termination = bool(
                consistency_result.get("early_termination_recommended")
            )
        clean_frame_reanchor_applied = bool(
            effective_reanchor_interval > 0 and step_index % effective_reanchor_interval == 0
        )
        next_policy_frame = generated_frame
        clean_frame_reanchor_source_kind: str | None = None

        # 3. The selected observation backend validates or analyses the generated
        # frame immediately. The direct RGB-only path runs no perception model.
        result = run_wam_derived_observation_harness_step(
            output_dir=harness_dir,
            generated_at=generated,
            step_index=step_index,
            source_generated_frame_path=generated_frame,
            source_generated_video_path=wam_output.get("generated_video_path"),
            source_wam_rollout_id=f"oscar_isaac_closed_loop_step_{step_index:04d}",
            transition_id=f"oscar_isaac_transition_{step_index:04d}",
            source_policy_action={
                **action,
                **({"task_prompt": cleaned_target_prompts[0]} if cleaned_target_prompts else {}),
            },
            action_history=action_history,
            current_policy_observation=_policy_observation(
                current_frame,
                target,
                step_index,
                task_prompt=resolved_task_prompt,
            ),
            skeleton_conditioning=wam_output.get("skeleton_conditioning"),
            eval_ready_task_grounding={
                "schema_version": "eval_ready_task_grounding.v1",
                "status": "prompt_only_for_generated_frame_perception"
                if cleaned_target_prompts
                else "not_supplied",
                "task": {
                    "task_id": "isaac_g1_oscar_per_step_closed_loop",
                    "target_prompts_for_object_index_backends": cleaned_target_prompts,
                },
                "selected_task_target": {
                    "object_id": "perception_target",
                    "label": cleaned_target_prompts[0],
                    "source_prompt": cleaned_target_prompts[0],
                    "source": "closed_loop_cli_target_prompt",
                }
                if cleaned_target_prompts
                else {},
            },
            previous_steps=step_records,
            previous_adapter_reports=adapter_reports,
            backend_kind=harness_backend_kind,
            external_consistency=_mapping(consistency_result.get("consistency")),
            backend_command=harness_backend_command,
            allow_external_backend=allow_external_backend,
            backend_timeout_seconds=backend_timeout_seconds,
            policy_id=policy_id,
        )
        step_record = dict(result.get("step_record") or {})
        adapter_report = dict(result.get("policy_adapter_report") or {})
        backend_status = _step_backend_status(step_record)
        if require_real_perception_backend and not backend_status["real_model_ran"]:
            blockers.append(f"real_perception_backend_not_proven_at_step_{step_index}")
        if require_sam3_completed and not backend_status["sam3_completed"]:
            blockers.append(f"sam3_provider_not_completed_at_step_{step_index}")
        if require_da3_completed and not backend_status["da3_completed"]:
            blockers.append(f"da3_provider_not_completed_at_step_{step_index}")
        step_records.append(step_record)
        adapter_reports.append(adapter_report)

        adapted_observation = _mapping(result.get("adapted_policy_observation"))
        adapted_observation["task_prompt"] = resolved_task_prompt or None
        if current_generated_robot_state:
            adapted_observation["generated_robot_state"] = dict(current_generated_robot_state)
            adapted_observation["generated_robot_state_carried_forward"] = True
            adapted_observation["controller_fk_generated_robot_state"] = dict(
                current_generated_robot_state
            )

        # Apply the action to the persistent Isaac session and measure the
        # resulting articulation state before asking GR00T for another action.
        # Controller FK remains separate diagnostic evidence; it must never be
        # substituted for the same-session post-action proprioception below.
        completion_result: dict[str, Any] = {}
        post_action_policy_state: dict[str, Any] = {}
        post_action_stance_report: dict[str, Any] = {}
        raw_completion_result: dict[str, Any] = {}
        task_completion_evaluation_failed = False
        task_completion_evaluation_status = (
            "not_configured" if task_completion_evaluator is None else "pending"
        )
        if task_completion_evaluator is not None:
            try:
                raw_completion_result = dict(
                    task_completion_evaluator(
                        {
                            "step_index": step_index,
                            # Frame 0000 is the immutable initial observation.
                            # Action evidence therefore uses its source step
                            # number (0001..N) and can never overwrite it.
                            "evidence_step_index": step_index,
                            "action": action,
                            "wam_output": wam_output,
                            "harness_step_record": step_record,
                            "adapted_observation": adapted_observation,
                            "task_success_contract": completion_contract,
                        }
                    )
                    or {}
                )
                completion_result = _validate_task_completion_transition(
                    completion_result=raw_completion_result,
                    task_success_contract=completion_contract,
                    expected_source_step_index=step_index,
                )
                task_completion_results.append(completion_result)
                expected_action_sha256 = _canonical_sha256(action)
                if _string(raw_completion_result.get("source_action_sha256")).strip() != (
                    expected_action_sha256
                ):
                    raise RuntimeError("post_action_policy_state_response_action_sha256_mismatch")
                post_action_policy_state = _validated_post_action_policy_state(
                    raw_completion_result.get("post_action_policy_state"),
                    simulator_session_id=_string(
                        raw_completion_result.get("simulator_session_id")
                    ).strip(),
                    stage_id=_string(raw_completion_result.get("stage_id")).strip(),
                    source_action_sha256=expected_action_sha256,
                    source_step_index=step_index,
                )
                adapted_observation["unitree_g1_sonic_state"] = dict(post_action_policy_state)
                adapted_observation["unitree_g1_sonic_state_source"] = (
                    POST_ACTION_POLICY_STATE_SOURCE
                )
                state_measurement = _mapping(post_action_policy_state.get("measurement"))
                adapted_observation["unitree_g1_sonic_state_metadata"] = {
                    "complete": True,
                    "surrogate": False,
                    "measured_proprio_available": True,
                    "simulator_session_id": state_measurement.get("simulator_session_id"),
                    "stage_id": state_measurement.get("stage_id"),
                    "source_action_sha256": state_measurement.get("source_action_sha256"),
                    "source_step_index": state_measurement.get("source_step_index"),
                    "captured_at_ns": state_measurement.get("captured_at_ns"),
                }
                task_completion_evaluation_status = "completed_with_live_post_action_state"
            except Exception as exc:  # noqa: BLE001 - fail closed before policy requery
                task_completion_evaluation_failed = True
                task_completion_evaluation_status = "failed"
                blockers.append(
                    f"task_completion_evaluation_failed_at_step_{step_index}:"
                    f"{type(exc).__name__}:{exc}"
                )

        if post_action_policy_state:
            post_action_stance_report = _post_action_stance_report(post_action_policy_state)
        unsafe_stance_detected = bool(
            post_action_stance_report.get("unsafe_stance_detected") is True
        )
        unsafe_stance_terminal_now = bool(stop_on_unsafe_stance and unsafe_stance_detected)

        if clean_frame_reanchor_applied:
            post_action_egocentric_frame = _mapping(
                raw_completion_result.get("post_action_egocentric_frame")
            )
            post_action_frame_path = _string(post_action_egocentric_frame.get("path")).strip()
            if post_action_frame_path:
                next_policy_frame = post_action_frame_path
                clean_frame_reanchor_source_kind = (
                    "post_action_live_isaac_robot_head_mounted_egocentric"
                )
            elif task_completion_evaluator is not None and not (task_completion_evaluation_failed):
                blockers.append(f"post_action_egocentric_reanchor_missing_at_step_{step_index}")
                task_completion_evaluation_failed = True
                task_completion_evaluation_status = "failed_missing_post_action_egocentric_reanchor"
                next_policy_frame = initial_clean_frame
                clean_frame_reanchor_source_kind = (
                    "initial_policy_observation_clean_frame_fail_closed_fallback"
                )
            else:
                next_policy_frame = initial_clean_frame
                clean_frame_reanchor_source_kind = "initial_policy_observation_clean_frame"
            clean_frame_reanchor_events.append(
                {
                    "step_index": step_index,
                    "generated_next_observation_frame_path": generated_frame,
                    "next_policy_observation_frame_path": next_policy_frame,
                    "source_frame_kind": clean_frame_reanchor_source_kind,
                    "post_action_egocentric_frame": (
                        dict(post_action_egocentric_frame) if post_action_egocentric_frame else None
                    ),
                    "interval_steps": effective_reanchor_interval,
                }
            )

        transition_passed = bool(
            completion_result.get("registered_transition_passed") is True
            and _string(completion_result.get("criterion_id")).strip()
            not in {"root_proximity", "robot_root_proximity"}
        )
        if transition_passed:
            proven_task_completion_transition = dict(completion_result)
        step_root_position = action.get("root_position") or []
        target_reached_now = bool(
            len(step_root_position) >= len(target)
            and sum((float(a) - float(b)) ** 2 for a, b in zip(step_root_position, target)) ** 0.5
            < 0.25
        )
        completion_now = bool(
            transition_passed if manipulation_task else transition_passed or target_reached_now
        )
        task_progress_report = _task_progress_report(
            completion_result,
            minimum_progress_fraction=minimum_task_progress_fraction,
        )
        task_progress_value = _finite_float(task_progress_report.get("progress_toward_criterion"))
        minimum_progress_delta = _finite_float(
            task_progress_report.get("minimum_meaningful_progress_delta")
        )
        meaningful_progress_now = False
        if task_progress_value is not None and minimum_progress_delta is not None:
            if best_task_progress is None:
                best_task_progress = task_progress_value
                last_meaningful_task_progress_step = step_index
            elif task_progress_value >= best_task_progress + minimum_progress_delta:
                best_task_progress = task_progress_value
                last_meaningful_task_progress_step = step_index
                meaningful_progress_now = True
        steps_since_meaningful_progress = (
            step_index - last_meaningful_task_progress_step
            if last_meaningful_task_progress_step is not None
            else 0
        )
        # Phase-scoped progress (attempt 067 run 6): while the effector still
        # closes on the target, approach IS progress -- the task joint cannot
        # move before contact. Stall requires BOTH streams dead for patience.
        patience_steps = max(1, int(no_progress_patience_steps))
        effector_report = _mapping(wam_output.get("manipulation_effector_progress_report"))
        approach_progress_m = _finite_float(effector_report.get("best_progress_toward_target_m"))
        approach_minimum_m = _finite_float(effector_report.get("minimum_required_progress_m"))
        if approach_progress_m is not None and approach_minimum_m is not None and approach_progress_m >= approach_minimum_m:
            last_meaningful_approach_step = step_index
        approach_stream_stalled = last_meaningful_approach_step is None or (
            step_index - last_meaningful_approach_step) >= patience_steps
        no_progress_terminal_now = bool(
            stop_on_no_progress
            and manipulation_task
            and not completion_now
            and task_progress_value is not None
            and step_index >= max(max(1, int(min_steps)), patience_steps + 1)
            and steps_since_meaningful_progress >= patience_steps
            and approach_stream_stalled
        )
        task_progress_report["approach_progress_m"] = approach_progress_m
        task_progress_report["approach_measurement_source"] = APPROACH_MEASUREMENT_SOURCE
        task_progress_report["last_meaningful_approach_step"] = last_meaningful_approach_step
        task_progress_report.update(
            {
                "step_index": step_index,
                "best_progress_toward_criterion": best_task_progress,
                "meaningful_progress_this_step": meaningful_progress_now,
                "last_meaningful_progress_step": last_meaningful_task_progress_step,
                "steps_since_meaningful_progress": steps_since_meaningful_progress,
                "patience_steps": patience_steps,
                "terminal_enabled": bool(stop_on_no_progress),
                "terminal_now": no_progress_terminal_now,
            }
        )
        task_progress_history.append(dict(task_progress_report))
        completion_terminal_now = bool(
            stop_on_task_completion and completion_now and step_index >= max(1, int(min_steps))
        )
        adapter_safe_for_policy_requery = bool(adapter_report.get("safe_for_policy_requery"))
        safe_for_terminal_observation_requery = bool(
            adapter_safe_for_policy_requery
            and not consistency_early_termination
            and not task_completion_evaluation_failed
            and not unsafe_stance_terminal_now
            and not no_progress_terminal_now
        )
        safe_for_policy_requery = bool(
            safe_for_terminal_observation_requery and not completion_terminal_now
        )
        policy_requeried_on_wam_observation = False
        policy_action_changed_vs_previous = False
        terminal_observation_policy_requery = False
        terminal_observation_policy_requery_action_sha256: str | None = None
        terminal_observation_policy_action_execution_status: str | None = None
        requery_status = "absent"
        if policy_endpoint is not None and (step_index < bounded_steps or completion_terminal_now):
            if completion_terminal_now:
                if safe_for_terminal_observation_requery:
                    try:
                        # A semantically terminal transition still needs one
                        # GR00T query on OSCAR's generated observation to prove
                        # that the learned WAM observation reached the policy.
                        # The resulting action is evidence only: the episode is
                        # already terminal, so it is never scheduled, applied to
                        # Isaac, or sent back through OSCAR.
                        terminal_action = dict(
                            policy_endpoint(
                                adapted_observation,
                                list(action_history),
                                step_index,
                            )
                            or {}
                        )
                        if not terminal_action:
                            raise RuntimeError("terminal_policy_action_empty")
                        learned_policy_requery_count += 1
                        terminal_observation_policy_requery_count += 1
                        terminal_signature = _endpoint_action_signature(terminal_action)
                        policy_action_changed_vs_previous = bool(
                            previous_endpoint_action_signature is not None
                            and terminal_signature != previous_endpoint_action_signature
                        )
                        terminal_observation_policy_requery = True
                        terminal_observation_policy_requery_action_sha256 = _canonical_sha256(
                            terminal_action
                        )
                        terminal_observation_policy_action_execution_status = (
                            "not_executed_semantic_terminal"
                        )
                        policy_requeried_on_wam_observation = True
                        requery_status = "completed_terminal_observation_not_executed"
                    except Exception as exc:  # noqa: BLE001
                        blockers.append(
                            "learned_policy_terminal_observation_requery_failed_at_step_"
                            f"{step_index}:{type(exc).__name__}"
                        )
                        requery_status = "failed_terminal_observation_requery"
                else:
                    requery_status = "skipped_terminal_observation_unsafe"
            elif unsafe_stance_terminal_now:
                requery_status = "skipped_unsafe_stance"
            elif task_completion_evaluation_failed:
                requery_status = "skipped_task_completion_evaluation_failed"
            elif no_progress_terminal_now:
                requery_status = "skipped_no_task_progress"
            elif safe_for_policy_requery:
                try:
                    learned_action = dict(
                        policy_endpoint(adapted_observation, list(action_history), step_index) or {}
                    )
                    learned_policy_requery_count += 1
                    signature = _endpoint_action_signature(learned_action)
                    policy_action_changed_vs_previous = (
                        previous_endpoint_action_signature is not None
                        and signature != previous_endpoint_action_signature
                    )
                    if policy_action_changed_vs_previous:
                        policy_action_changed_count += 1
                    previous_endpoint_action_signature = signature
                    pending_policy_endpoint_action = learned_action
                    pending_policy_endpoint_source_step = step_index
                    policy_requeried_on_wam_observation = True
                    requery_status = "completed"
                except Exception as exc:  # noqa: BLE001
                    blockers.append(
                        f"learned_policy_requery_failed_at_step_{step_index}:{type(exc).__name__}"
                    )
                    requery_status = "failed"
            else:
                requery_status = "skipped_unsafe"

        trace_row = {
            "step_index": step_index,
            "policy_action": action.get("policy_action"),
            "root_position": action.get("root_position"),
            "policy_action_source": action.get("policy_action_source"),
            "policy_action_from_wam_requery": policy_action_from_wam_requery,
            "policy_requeried_fresh": policy_requeried_fresh,
            "source_observation_frame": current_frame,
            "wam_generated_frame": generated_frame,
            "wam_generated_video": wam_output.get("generated_video_path"),
            "wam_backend": wam_output.get("wam_backend"),
            "wam_generation_status": wam_output.get("status")
            or wam_output.get("wam_generation_status"),
            "oscar_gpu_residency_report_path": wam_output.get("oscar_gpu_residency_report_path"),
            "oscar_gpu_residency_samples_path": wam_output.get("oscar_gpu_residency_samples_path"),
            "oscar_gpu_residency": wam_output.get("oscar_gpu_residency"),
            "fresh_oscar_provider_model_run_claimed": fresh_oscar_provider,
            "provider_output_path": wam_output.get("provider_output_path"),
            "harness_step_status": step_record.get("status"),
            "harness_backend_kind": harness_backend_kind,
            "real_perception_backend_model_ran": backend_status["real_model_ran"],
            "generated_rgb_policy_observation_validated": backend_status[
                "generated_rgb_policy_observation_validated"
            ],
            "no_perception_model_ran": backend_status["no_perception_model_ran"],
            "sam3_completed": backend_status["sam3_completed"],
            "depth_completed": backend_status["depth_completed"],
            "da3_completed": backend_status["da3_completed"],
            "safe_for_policy_requery": safe_for_policy_requery,
            "safe_for_terminal_observation_requery": (safe_for_terminal_observation_requery),
            "policy_adapter_safe_for_policy_requery": adapter_safe_for_policy_requery,
            "policy_requeried_on_wam_observation": policy_requeried_on_wam_observation,
            "policy_action_changed_vs_previous": policy_action_changed_vs_previous,
            "terminal_observation_policy_requery": terminal_observation_policy_requery,
            "terminal_observation_policy_requery_action_sha256": (
                terminal_observation_policy_requery_action_sha256
            ),
            "terminal_observation_policy_action_execution_status": (
                terminal_observation_policy_action_execution_status
            ),
            "requery_status": requery_status,
            "task_completion_evaluation_status": task_completion_evaluation_status,
            "post_action_policy_state_validated": bool(post_action_policy_state),
            "post_action_policy_state_source": (
                POST_ACTION_POLICY_STATE_SOURCE if post_action_policy_state else None
            ),
            "post_action_policy_state_simulator_session_id": _mapping(
                post_action_policy_state.get("measurement")
            ).get("simulator_session_id"),
            "post_action_policy_state_stage_id": _mapping(
                post_action_policy_state.get("measurement")
            ).get("stage_id"),
            "post_action_stance_report": post_action_stance_report or None,
            "unsafe_stance_detected": unsafe_stance_detected,
            "unsafe_stance_terminal_enabled": bool(stop_on_unsafe_stance),
            "online_task_progress": task_progress_report,
            "no_progress_terminal_enabled": bool(stop_on_no_progress),
            "no_progress_terminal_now": no_progress_terminal_now,
            "clean_frame_reanchor_applied": clean_frame_reanchor_applied,
            "clean_frame_reanchor_source_kind": clean_frame_reanchor_source_kind,
            "next_policy_observation_frame": next_policy_frame,
            "wam_episode_consistency_request": consistency_result.get("request_path"),
            "wam_consistency_checks": consistency_result.get("checks_path"),
            "external_episode_consistency_scorer_ran": bool(
                consistency_result.get("external_episode_consistency_scorer_ran")
            ),
            "forward_inverse_consistency_proven": bool(
                consistency_result.get("forward_inverse_consistency_proven")
            ),
            "wam_episode_consistency_early_termination_recommended": (
                consistency_early_termination
            ),
            "wam_episode_consistency_blockers": _string_list(consistency_result.get("blockers")),
            "action_conditioning_blockers": action_conditioning_blockers,
            "action_derived_skeleton_conditioning_proven": bool(
                policy_action_from_wam_requery
                and not action_conditioning_blockers
                and strict_action_derived_skeleton_conditioning
            ),
            "synchronized_multiview_validation": multiview_validation,
        }
        trace_rows.append(trace_row)
        proof_rows.append(
            {
                "step_index": step_index,
                "policy_action_recorded": bool(action.get("policy_action")),
                "policy_action_from_wam_requery": policy_action_from_wam_requery,
                "policy_requeried_fresh": policy_requeried_fresh,
                "source_observation_frame": current_frame,
                "wam_generated_frame": generated_frame,
                "wam_generated_video": wam_output.get("generated_video_path"),
                "oscar_per_step_backend": wam_output.get("wam_backend"),
                "oscar_gpu_residency_report_path": wam_output.get(
                    "oscar_gpu_residency_report_path"
                ),
                "oscar_gpu_residency_proof_passed": _mapping(
                    wam_output.get("oscar_gpu_residency")
                ).get("proof_passed"),
                "fresh_oscar_provider_model_run_claimed": fresh_oscar_provider,
                "real_perception_backend_model_ran": backend_status["real_model_ran"],
                "generated_rgb_policy_observation_validated": backend_status[
                    "generated_rgb_policy_observation_validated"
                ],
                "no_perception_model_ran": backend_status["no_perception_model_ran"],
                "sam3_completed": backend_status["sam3_completed"],
                "depth_completed": backend_status["depth_completed"],
                "da3_completed": backend_status["da3_completed"],
                "safe_for_policy_requery": safe_for_policy_requery,
                "safe_for_terminal_observation_requery": (safe_for_terminal_observation_requery),
                "policy_adapter_safe_for_policy_requery": adapter_safe_for_policy_requery,
                "policy_requeried_on_wam_observation": policy_requeried_on_wam_observation,
                "policy_action_changed_vs_previous": policy_action_changed_vs_previous,
                "terminal_observation_policy_requery": (terminal_observation_policy_requery),
                "terminal_observation_policy_requery_action_sha256": (
                    terminal_observation_policy_requery_action_sha256
                ),
                "terminal_observation_policy_action_execution_status": (
                    terminal_observation_policy_action_execution_status
                ),
                "requery_status": requery_status,
                "task_completion_evaluation_status": task_completion_evaluation_status,
                "post_action_policy_state_validated": bool(post_action_policy_state),
                "unsafe_stance_detected": unsafe_stance_detected,
                "post_action_stance_status": post_action_stance_report.get("status"),
                "online_task_progress": task_progress_report,
                "no_progress_terminal_now": no_progress_terminal_now,
                "external_episode_consistency_scorer_ran": bool(
                    consistency_result.get("external_episode_consistency_scorer_ran")
                ),
                "forward_inverse_consistency_proven": bool(
                    consistency_result.get("forward_inverse_consistency_proven")
                ),
                "wam_episode_consistency_early_termination_recommended": (
                    consistency_early_termination
                ),
            }
        )
        if consistency_early_termination:
            step_blockers = _string_list(consistency_result.get("blockers")) or [
                "wam_episode_consistency_not_proven"
            ]
            blockers.extend(
                f"wam_episode_consistency_step_{step_index}:{blocker}" for blocker in step_blockers
            )
            episode_termination_reason = (
                f"wam_episode_consistency_early_termination_at_step_{step_index}"
            )
            break

        if unsafe_stance_terminal_now:
            blockers.append(f"unsafe_post_action_robot_stance_at_step_{step_index}")
            episode_termination_reason = f"unsafe_stance_detected_at_step_{step_index}"
            current_frame = next_policy_frame
            break

        if task_completion_evaluation_failed:
            episode_termination_reason = f"task_completion_evaluation_failed_at_step_{step_index}"
            current_frame = next_policy_frame
            break

        # Dynamic episode length: task completion is measured above, before
        # any next-policy query. `steps` remains only the hard safety cap.
        if completion_terminal_now:
            criterion_id = (
                _string(completion_result.get("criterion_id")).strip() or "navigation_goal"
            )
            episode_termination_reason = (
                f"task_criterion_{criterion_id}_passed_at_step_{step_index}"
            )
            task_completed_early = True
            # 4. feed forward still records the generated frame as consumed.
            current_frame = next_policy_frame
            break

        if no_progress_terminal_now:
            blockers.append(f"online_task_no_progress_at_step_{step_index}")
            episode_termination_reason = f"no_task_progress_at_step_{step_index}"
            current_frame = next_policy_frame
            break

        # 4. feed forward: the generated frame becomes the next step's observation
        current_frame = next_policy_frame

    if episode_termination_reason is None:
        if blockers:
            episode_termination_reason = f"blocked:{blockers[-1]}"
        else:
            episode_termination_reason = "steps_cap_reached"

    trace_path = resolved_out / "oscar_isaac_closed_loop_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in trace_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    generated_episode_artifacts = _closed_loop_generated_episode_artifacts(
        output_dir=resolved_out,
        generated_at=generated,
        trace_rows=trace_rows,
        initial_frame_path=initial_clean_frame,
        policy_id=policy_id,
        task_prompts=cleaned_target_prompts,
        target=target,
    )
    generated_video_success = _score_closed_loop_generated_video_success(
        output_dir=resolved_out,
        generated_at=generated,
        episode_artifacts=generated_episode_artifacts,
        task_prompts=cleaned_target_prompts,
        command=wam_success_label_command,
        allow_wam_success_labeling=allow_wam_success_labeling,
        timeout_seconds=(
            float(wam_success_label_timeout_seconds)
            if wam_success_label_timeout_seconds is not None
            else float(backend_timeout_seconds)
        ),
    )

    final_pose = trace_rows[-1]["root_position"] if trace_rows else list(route[0])
    reached = bool(
        trace_rows
        and interpolate_route(route, 1.0)[0]
        and sum((a - b) ** 2 for a, b in zip(final_pose, target)) ** 0.5 < 0.25
    )
    fresh_learned_policy_requery_steps = sum(
        1 for row in proof_rows if row.get("policy_requeried_fresh")
    )
    generated_observation_count = sum(1 for row in proof_rows if row.get("wam_generated_frame"))
    multi_action_policy_requery_contract_proven = bool(
        learned_policy_requery_count >= 2 and policy_action_changed_count >= 1
    )
    terminal_observation_policy_requery_proven = bool(
        manipulation_task
        and task_completed_early
        and len(action_history) == 1
        and proven_task_completion_transition
        and terminal_observation_policy_requery_count == 1
        and sum(
            1
            for row in proof_rows
            if row.get("terminal_observation_policy_requery") is True
            and row.get("policy_requeried_on_wam_observation") is True
            and row.get("requery_status") == "completed_terminal_observation_not_executed"
            and row.get("terminal_observation_policy_action_execution_status")
            == "not_executed_semantic_terminal"
            and row.get("terminal_observation_policy_requery_action_sha256")
        )
        == 1
    )
    policy_endpoint_requery_contract_proven = bool(
        multi_action_policy_requery_contract_proven or terminal_observation_policy_requery_proven
    )
    policy_observes_wam_generated_next_observation = bool(
        generated_observation_count >= 1
        and (
            fresh_learned_policy_requery_steps >= 2
            or multi_action_policy_requery_contract_proven
            or terminal_observation_policy_requery_proven
        )
    )
    wam_evaluator_in_control_loop = bool(generated_observation_count >= 1)
    consistency_scorer_ran_steps = sum(
        1 for row in proof_rows if row.get("external_episode_consistency_scorer_ran")
    )
    consistency_proven_steps = sum(
        1 for row in proof_rows if row.get("forward_inverse_consistency_proven")
    )
    forward_consistency_proven = bool(consistency_results) and all(
        item.get("forward_dynamics_consistency_proven") is True for item in consistency_results
    )
    inverse_consistency_proven = bool(consistency_results) and all(
        item.get("inverse_dynamics_consistency_proven") is True for item in consistency_results
    )
    consistency_early_termination_recommended = any(
        row.get("wam_episode_consistency_early_termination_recommended") for row in proof_rows
    )
    generated_video_success_label_passed = bool(
        generated_video_success.get("generated_video_success_label_passed")
    )
    simulated_manipulation_success_shown = bool(
        generated_video_success.get("simulated_manipulation_success_shown")
    )
    if policy_endpoint is not None and not policy_endpoint_requery_contract_proven:
        blockers.append("blocked_learned_policy_requery_not_proven")
    if strict_fresh_learned_policy_requery and fresh_learned_policy_requery_steps < 1:
        blockers.append("fresh_learned_policy_requery_not_proven")
    if strict_action_derived_skeleton_conditioning and (
        fresh_learned_policy_requery_steps < 1
        or action_derived_skeleton_conditioned_steps != fresh_learned_policy_requery_steps
    ):
        blockers.append("fresh_learned_actions_not_all_controller_fk_conditioned")
    unique_conditioned_action_count = len(
        {row["learned_action_signature_sha256"] for row in action_conditioning_evidence_rows}
    )
    unique_conditioning_evidence_count = len(
        {row["conditioning_evidence_sha256"] for row in action_conditioning_evidence_rows}
    )
    action_conditioning_differentiation_proven = bool(
        unique_conditioned_action_count >= 2
        and unique_conditioning_evidence_count == unique_conditioned_action_count
    )
    single_executed_action_semantic_terminal = bool(
        terminal_observation_policy_requery_proven
        and len(action_history) == 1
        and action_derived_skeleton_conditioned_steps == 1
    )
    action_conditioning_differentiation_applicable = bool(
        strict_action_derived_skeleton_conditioning and not single_executed_action_semantic_terminal
    )
    action_conditioning_differentiation_requirement_satisfied = bool(
        not strict_action_derived_skeleton_conditioning
        or single_executed_action_semantic_terminal
        or action_conditioning_differentiation_proven
    )
    action_conditioning_differentiation_status = (
        "not_required"
        if not strict_action_derived_skeleton_conditioning
        else "not_applicable_single_executed_action_semantic_terminal"
        if single_executed_action_semantic_terminal
        else "proven"
        if action_conditioning_differentiation_proven
        else "not_proven"
    )
    if (
        action_conditioning_differentiation_applicable
        and not action_conditioning_differentiation_proven
    ):
        blockers.append("fresh_action_conditioning_differentiation_not_proven")
    if require_forward_inverse_consistency and (
        not consistency_results
        or consistency_scorer_ran_steps < len(proof_rows)
        or consistency_proven_steps < len(proof_rows)
        or consistency_early_termination_recommended
    ):
        blockers.append("forward_inverse_consistency_not_proven")
    if require_generated_video_success_label and not simulated_manipulation_success_shown:
        blockers.append("generated_video_success_label_not_proven")
        blockers.extend(
            f"generated_video_success_label:{blocker}"
            for blocker in _string_list(generated_video_success.get("blockers"))
        )
    if stop_on_task_completion and manipulation_task and not proven_task_completion_transition:
        blockers.append("registered_manipulation_task_transition_not_proven")
    status = "completed" if trace_rows and not blockers else "blocked"
    generated_episode_artifacts = _bind_generated_episode_to_authoritative_loop_status(
        generated_episode_artifacts,
        authoritative_status=status,
    )
    feed_forward_verified = all(
        trace_rows[index]["source_observation_frame"]
        == trace_rows[index - 1]["next_policy_observation_frame"]
        for index in range(1, len(trace_rows))
    )
    proof = {
        "policy_source": (
            f"policy_endpoint:{_callable_label(policy_endpoint)}"
            if policy_endpoint is not None
            else "isaac_g1_policy.DeterministicWalkToTargetPolicy"
        ),
        "simulator_backend": "isaac",
        "selected_wam_backend": _string(wam_backend_id) or "oscar_wam",
        "isaac_policy_actions_recorded": len(action_history),
        "learned_policy_requery_count": learned_policy_requery_count,
        "terminal_observation_policy_requery_count": (terminal_observation_policy_requery_count),
        "terminal_observation_policy_requery_proven": (terminal_observation_policy_requery_proven),
        "terminal_observation_policy_action_execution_status": (
            "not_executed_semantic_terminal"
            if terminal_observation_policy_requery_proven
            else "not_applicable"
        ),
        "multi_action_policy_requery_contract_proven": (
            multi_action_policy_requery_contract_proven
        ),
        "policy_endpoint_requery_contract_proven": (policy_endpoint_requery_contract_proven),
        "fresh_learned_policy_requery_steps": fresh_learned_policy_requery_steps,
        "policy_action_changed_count": policy_action_changed_count,
        "action_derived_skeleton_conditioned_steps": (action_derived_skeleton_conditioned_steps),
        "fresh_actions_all_controller_fk_conditioned": bool(
            fresh_learned_policy_requery_steps > 0
            and action_derived_skeleton_conditioned_steps == fresh_learned_policy_requery_steps
        ),
        "fresh_action_conditioning_differentiation_proven": bool(
            action_conditioning_differentiation_proven
        ),
        "fresh_action_conditioning_differentiation_applicable": (
            action_conditioning_differentiation_applicable
        ),
        "fresh_action_conditioning_differentiation_requirement_satisfied": (
            action_conditioning_differentiation_requirement_satisfied
        ),
        "fresh_action_conditioning_differentiation_status": (
            action_conditioning_differentiation_status
        ),
        "action_conditioning_evidence": action_conditioning_evidence_rows,
        "registered_task_completion_transition": (proven_task_completion_transition or None),
        "manipulation_success_signal": (
            "success" if manipulation_task and proven_task_completion_transition else "not_proven"
        ),
        "generated_observation_count": generated_observation_count,
        "policy_observes_wam_generated_next_observation": (
            policy_observes_wam_generated_next_observation
        ),
        "wam_evaluator_in_control_loop": wam_evaluator_in_control_loop,
        "oscar_per_step_generation_calls": sum(
            1 for row in proof_rows if row.get("oscar_per_step_backend")
        ),
        "fresh_oscar_provider_model_run_steps": sum(
            1 for row in proof_rows if row.get("fresh_oscar_provider_model_run_claimed")
        ),
        "real_perception_backend_steps": sum(
            1 for row in proof_rows if row.get("real_perception_backend_model_ran")
        ),
        "generated_rgb_policy_observation_validated_steps": sum(
            1 for row in proof_rows if row.get("generated_rgb_policy_observation_validated")
        ),
        "no_perception_model_steps": sum(
            1 for row in proof_rows if row.get("no_perception_model_ran")
        ),
        "sam3_completed_steps": sum(1 for row in proof_rows if row.get("sam3_completed")),
        "depth_completed_steps": sum(1 for row in proof_rows if row.get("depth_completed")),
        "da3_completed_steps": sum(1 for row in proof_rows if row.get("da3_completed")),
        "feed_forward_verified": feed_forward_verified,
        "wam_episode_consistency_requested": consistency_requested,
        "external_episode_consistency_scorer_ran_steps": consistency_scorer_ran_steps,
        "forward_inverse_consistency_proven_steps": consistency_proven_steps,
        "forward_inverse_consistency_proven": bool(
            consistency_results
            and consistency_proven_steps == len(consistency_results)
            and not consistency_early_termination_recommended
        ),
        "wam_episode_consistency_early_termination_recommended": (
            consistency_early_termination_recommended
        ),
        "wam_episode_consistency_request_paths": [
            str(item.get("request_path"))
            for item in consistency_results
            if item.get("request_path")
        ],
        "wam_consistency_checks_paths": [
            str(item.get("checks_path")) for item in consistency_results if item.get("checks_path")
        ],
        "wam_episode_consistency_blockers": sorted(
            {
                str(blocker)
                for item in consistency_results
                for blocker in _string_list(item.get("blockers"))
                if str(blocker)
            }
        ),
        "closed_loop_generated_episode_manifest_path": generated_episode_artifacts["manifest_path"],
        "closed_loop_generated_episode_results_path": generated_episode_artifacts["results_path"],
        "generated_video_success_label_requested": bool(
            require_generated_video_success_label
            or allow_wam_success_labeling
            or _wam_success_label_command(wam_success_label_command)
        ),
        "generated_video_success_label_judge_configured": bool(
            generated_video_success.get("success_label_judge_configured")
        ),
        "generated_video_success_label_judge_ran": bool(
            generated_video_success.get("success_label_judge_ran")
        ),
        "generated_video_success_label_passed": generated_video_success_label_passed,
        "simulated_manipulation_success_shown": simulated_manipulation_success_shown,
        "real_world_task_success_proven": False,
        "requirements": {
            "fresh_oscar_provider_required": bool(require_fresh_oscar_provider),
            "real_perception_backend_required": bool(require_real_perception_backend),
            "sam3_completed_required": bool(require_sam3_completed),
            "da3_completed_required": bool(require_da3_completed),
            "fresh_learned_policy_requery_required": strict_fresh_learned_policy_requery,
            "action_derived_skeleton_conditioning_required": bool(
                strict_action_derived_skeleton_conditioning
            ),
            "forward_inverse_consistency_required": bool(require_forward_inverse_consistency),
            "synchronized_calibrated_multiview_required": bool(
                require_synchronized_calibrated_multiview
            ),
            "generated_video_success_label_required": bool(require_generated_video_success_label),
        },
        "per_step": proof_rows,
    }
    manipulation_success_judge = evaluate_isaac_manipulation_success(
        generated_at=generated,
        status=status,
        proof=proof,
        trace_rows=trace_rows,
        task_target_reached=reached,
        perception_target_prompts=cleaned_target_prompts,
    )
    manipulation_success_judge_path = resolved_out / "manipulation_success_evaluator_results.json"
    write_json(manipulation_success_judge_path, manipulation_success_judge)
    manifest = {
        "schema_version": LOOP_SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "loop_kind": (
            "per_step_policy_wam_generated_rgb_closed_loop"
            if harness_backend_kind == GENERATED_RGB_POLICY_OBSERVATION_BACKEND_KIND
            else "per_step_policy_wam_perception_closed_loop"
        ),
        "steps_executed": len(trace_rows),
        "steps_requested": bounded_steps,
        "harness_backend_kind": harness_backend_kind,
        "real_perception_backend_used": bool(proof["real_perception_backend_steps"]),
        "generated_rgb_policy_observation_backend_used": bool(
            harness_backend_kind == GENERATED_RGB_POLICY_OBSERVATION_BACKEND_KIND
        ),
        "generated_rgb_policy_observation_validated_steps": proof[
            "generated_rgb_policy_observation_validated_steps"
        ],
        "sam3_da3_or_other_perception_model_ran": bool(proof["real_perception_backend_steps"]),
        "task_target_position_xyz": [round(float(c), 6) for c in target],
        "perception_target_prompts": cleaned_target_prompts,
        "final_root_position": final_pose,
        "task_target_reached": reached,
        "trace_path": str(trace_path),
        "harness_dir": str(harness_dir),
        "selected_wam_backend": _string(wam_backend_id) or "oscar_wam",
        "wam_backend_readiness": dict(wam_backend_readiness or {}),
        "policy_observes_wam_generated_next_observation": proof[
            "policy_observes_wam_generated_next_observation"
        ],
        "wam_evaluator_in_control_loop": proof["wam_evaluator_in_control_loop"],
        "clean_frame_reanchoring": {
            "enabled": bool(effective_reanchor_interval > 0),
            "interval_steps": int(effective_reanchor_interval) or None,
            "source_frame_kind": (
                sorted(
                    {
                        _string(event.get("source_frame_kind"))
                        for event in clean_frame_reanchor_events
                        if _string(event.get("source_frame_kind"))
                    }
                )[0]
                if len(
                    {
                        _string(event.get("source_frame_kind"))
                        for event in clean_frame_reanchor_events
                        if _string(event.get("source_frame_kind"))
                    }
                )
                == 1
                else "mixed_reanchor_sources"
                if clean_frame_reanchor_events
                else "initial_policy_observation_clean_frame"
            ),
        },
        "episode_termination": {
            "reason": episode_termination_reason,
            "steps_executed": len(trace_rows),
            "steps_cap": int(steps),
            "stop_on_task_completion": bool(stop_on_task_completion),
            "stop_on_unsafe_stance": bool(stop_on_unsafe_stance),
            "stop_on_no_progress": bool(stop_on_no_progress),
            "no_progress_patience_steps": max(1, int(no_progress_patience_steps)),
            "minimum_task_progress_fraction": float(minimum_task_progress_fraction),
            "task_progress_history": task_progress_history,
            "unsafe_stance_detected": any(
                row.get("unsafe_stance_detected") is True for row in trace_rows
            ),
            "min_steps": max(1, int(min_steps)),
            "task_completed_early": bool(task_completed_early),
            "task_kind": task_kind,
            "manipulation_requires_registered_observable_transition": True,
            "registered_task_completion_evaluator_configured": bool(
                task_completion_evaluator is not None
            ),
            "task_completion_results": task_completion_results,
            "task_completion_evidence_status": (
                "passed"
                if task_completed_early
                else "blocked_missing_registered_task_completion_evaluator"
                if stop_on_task_completion
                and manipulation_task
                and task_completion_evaluator is None
                else "not_satisfied"
            ),
            "claim_boundary": (
                "Navigation smoke may terminate on a root goal. Manipulation "
                "requires a registered task-specific observable transition; "
                "robot-root proximity never proves manipulation success. A live "
                "unsafe stance terminates the configured manipulation episode. "
                "The no-progress watchdog is resource control only and cannot "
                "prove success."
            ),
        },
        "generated_clip_coherence": {
            "per_step": clip_coherence_rows,
            "seed_correlation_floor": GENERATED_CLIP_SEED_CORRELATION_FLOOR,
            "min_coherent_horizon_frames_required": int(min_coherent_horizon_frames or 0),
            "min_measured_coherent_horizon_frames": min(
                (
                    int(row.get("coherent_horizon_frames") or 0)
                    for row in clip_coherence_rows
                    if row.get("status") == "measured"
                ),
                default=None,
            ),
            "claim_boundary": (
                "Coherence horizons quantify visual drift of generated clips; "
                "they are quality floors, never task-success evidence."
            ),
        },
        "clean_frame_reanchor_event_count": len(clean_frame_reanchor_events),
        "clean_frame_reanchor_events": clean_frame_reanchor_events,
        "periodic_clean_frame_reanchoring_used": bool(clean_frame_reanchor_events),
        "manipulation_success_evaluator_results_path": str(manipulation_success_judge_path),
        "manipulation_success_proven": bool(
            manipulation_success_judge.get("manipulation_success_proven")
        ),
        "closed_loop_generated_episode_manifest_path": generated_episode_artifacts["manifest_path"],
        "closed_loop_generated_episode_results_path": generated_episode_artifacts["results_path"],
        "generated_video_success": generated_video_success,
        "generated_video_success_label_request_path": generated_video_success["request_path"],
        "generated_video_success_labels_path": generated_video_success["success_labels_path"],
        "generated_video_success_label_passed": generated_video_success_label_passed,
        "simulated_manipulation_success_shown": simulated_manipulation_success_shown,
        "real_world_task_success_proven": False,
        "success_proof": {
            "manipulation_success_proven": bool(
                manipulation_success_judge.get("manipulation_success_proven")
            ),
            "simulated_manipulation_success_shown": simulated_manipulation_success_shown,
            "generated_video_success_label_passed": generated_video_success_label_passed,
            "generated_video_success_label_is_sim_only": True,
            "real_world_task_success_proven": False,
            "did_target_manipulation_succeed": bool(
                manipulation_success_judge.get("did_target_manipulation_succeed")
            ),
            "success_proof_separate_from_structural_loop_proof": True,
            "structural_loop_completed": bool(
                manipulation_success_judge.get("structural_loop_completed")
            ),
            "answer": manipulation_success_judge.get("answer"),
        },
        "forward_inverse_consistency_proven": bool(proof["forward_inverse_consistency_proven"]),
        "external_episode_consistency_scorer_ran": bool(consistency_scorer_ran_steps),
        "wam_episode_consistency_early_termination_recommended": bool(
            consistency_early_termination_recommended
        ),
        "wam_episode_consistency_request_paths": list(
            proof["wam_episode_consistency_request_paths"]
        ),
        "wam_consistency_checks_paths": list(proof["wam_consistency_checks_paths"]),
        "proof": proof,
        "blockers": blockers,
        "claim_boundary": (
            "Per-step closed loop: policy action -> WAM-generated next observation -> SAM3/DA3 "
            "perception harness, repeated. Harness derives support observations from WAM pixels; "
            "policy_observes_wam_generated_next_observation is true only when fresh learned-policy "
            "requery evidence is present. Clean-frame reanchoring, when enabled, feeds the initial "
            "clean policy observation back into the loop as drift control; it is not raw capture "
            "truth or task-success proof. Task success is judged in-process by the "
            "manipulation_success_evaluator and kept separate from structural loop proof; it is "
            "not_proven unless a learned-policy task-success signal fired. Forward/inverse "
            "episode consistency, when configured, is an external reliability/abstention "
            "signal only and can block feed-forward policy requery without proving task success. "
            "Generated-video success labels are sim-only support labels over model-derived media; "
            "they do not prove real-world task success, physical robot readiness, or "
            "forward/inverse episode consistency."
        ),
        "raw_secret_values_recorded": False,
    }
    legacy_proof_rows = legacy_worker_proof_rows(
        proof=proof,
        task_completion_results=task_completion_results,
        manipulation_success_judge=manipulation_success_judge,
        proven_task_completion_transition=proven_task_completion_transition,
        consistency_results=consistency_results,
        forward_consistency_proven=forward_consistency_proven,
        inverse_consistency_proven=inverse_consistency_proven,
    )
    if attempt_input_manifest:
        manifest["g1_kitchen_proof_rows"] = emit_rows_from_closed_loop_state(locals())
    else:
        manifest["g1_kitchen_proof_rows"] = legacy_proof_rows
    write_json(resolved_out / "oscar_isaac_closed_loop_manifest.json", manifest)
    return manifest


DEFAULT_SAM3_HARNESS_BACKEND_COMMAND = [
    sys.executable,
    "-m",
    "blueprint_pipeline.wam_real_provider_validation_probe",
    "backend",
]


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser, exposed so the sealed bundle argv contract is hermetically testable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-frame", required=True, help="initial robot-POV observation frame")
    parser.add_argument(
        "--start-frame-evidence",
        help=(
            "JSON controller/FK camera-projection context that hash-binds --start-frame "
            "to the live Isaac rigid-head robot POV. Required before any learned-policy query."
        ),
    )
    parser.add_argument("--route-file", required=True, help='JSON: {"route_points": [[x,y,z],...]}')
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--task-prompt", default="walk to the sink")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_OSCAR_NUM_FRAMES,
        help="OSCAR clip length per step",
    )
    parser.add_argument("--oscar-resident-worker", action="store_true")
    parser.add_argument("--oscar-resident-worker-max-restarts", type=int, default=0)
    parser.add_argument("--oscar-num-steps", type=int, default=35)
    parser.add_argument("--oscar-guidance", type=float, default=6.0)
    parser.add_argument("--oscar-seed", type=int, default=42)
    parser.add_argument("--oscar-height", type=int, default=480)
    parser.add_argument("--oscar-width", type=int, default=640)
    parser.add_argument("--oscar-fps", type=float, default=15.0)
    parser.add_argument(
        "--allow-non-native-oscar-resolution",
        action="store_true",
        help=(
            "Explicitly permit generation below OSCAR's native 480x640. "
            "Sub-native generation degrades quality and does NOT reduce "
            "weight-residency OOM (2026-07-06 lesson); without this flag the "
            "run blocks before any GPU work."
        ),
    )
    parser.add_argument(
        "--stop-on-task-completion",
        action="store_true",
        help=(
            "Make episode length task-adaptive: end the episode when the "
            "target-reached criterion fires (same test the final judge uses); "
            "--steps then acts only as the hard cap. Without this the episode "
            "always runs exactly --steps generations regardless of the task."
        ),
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=1,
        help="Minimum steps before task-completion early termination may fire.",
    )
    parser.add_argument(
        "--no-progress-patience-steps",
        type=int,
        default=DEFAULT_NO_PROGRESS_PATIENCE_STEPS,
        help=(
            "For registered manipulation tasks, stop after this many closed-loop "
            "decisions without meaningful live task-state progress."
        ),
    )
    parser.add_argument(
        "--minimum-task-progress-fraction",
        type=float,
        default=0.02,
        help=(
            "Meaningful online progress as a fraction of the registered success "
            "tolerance; this is a resource watchdog, not success proof."
        ),
    )
    parser.add_argument(
        "--task-success-contract",
        default=None,
        help=(
            "JSON contract registering manipulation criteria, observable transitions, "
            "comparisons, tolerances, and units. Required with a manipulation "
            "--task-completion-command."
        ),
    )
    parser.add_argument(
        "--attempt-input-manifest",
        default=None,
        help="Immutable attempt manifest used to bind and sign worker proof leaves.",
    )
    parser.add_argument(
        "--task-completion-command",
        default=None,
        help=(
            "Per-step evaluator command. Receives BLUEPRINT_TASK_COMPLETION_INPUT "
            "and must write BLUEPRINT_TASK_COMPLETION_OUTPUT with a typed, hashed "
            "task-transition measurement."
        ),
    )
    parser.add_argument(
        "--min-coherent-horizon-frames",
        type=int,
        default=2,
        help=(
            "Fail a step whose generated clip is not seed-anchored for at "
            "least this many frames (2 = the frame fed forward must be "
            "coherent). 0 disables the gate."
        ),
    )
    parser.add_argument(
        "--wam-backend",
        choices=SUPPORTED_CLOSED_LOOP_WAM_BACKENDS,
        default="oscar_wam",
        help=(
            "WAM backend requested for the closed-loop. cosmos3_wam requires an "
            "explicit per-step command and controller/FK action conditioning."
        ),
    )
    parser.add_argument("--oscar-repo")
    parser.add_argument("--checkpoint")
    parser.add_argument("--use-provider-command", action="store_true")
    parser.add_argument("--oscar-provider", choices=("auto", "vast", "runpod"), default="vast")
    parser.add_argument("--provider-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--allow-paid-provider-launch", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--harness-backend-kind", default="real_provider_probe")
    parser.add_argument("--harness-backend-command", default=None)
    parser.add_argument("--perception-target-prompt", action="append", default=[])
    parser.add_argument("--require-fresh-oscar-provider", action="store_true")
    parser.add_argument("--require-real-perception-backend", action="store_true")
    parser.add_argument("--require-sam3-completed", action="store_true")
    parser.add_argument("--require-da3-completed", action="store_true")
    parser.add_argument("--require-fresh-learned-policy-requery", action="store_true")
    parser.add_argument(
        "--learned-policy-command",
        default=None,
        help=(
            "Command endpoint for a real learned policy. Receives a phase-local JSON "
            "request via stdin and BLUEPRINT_LEARNED_POLICY_INPUT, and must write "
            "BLUEPRINT_LEARNED_POLICY_OUTPUT with strict action/checkpoint proof."
        ),
    )
    parser.add_argument(
        "--groot-sonic-policy-server-url",
        default=None,
        help=(
            "Wire a live GR00T N1.7+SONIC ZMQ policy server as the learned "
            "policy_endpoint (real per-step requery on the WAM-generated "
            "observation). Example: tcp://127.0.0.1:5550"
        ),
    )
    parser.add_argument(
        "--groot-sonic-execution-frame-count",
        type=int,
        default=1,
        help=(
            "Number of frames from each GR00T SONIC action horizon to send to "
            "the controller/FK path. The default preserves the historical "
            "single-frame behavior; direct full-horizon evaluations may opt in "
            "up to the model-provided horizon."
        ),
    )
    parser.add_argument("--groot-root", default=None)
    parser.add_argument(
        "--groot-policy-initial-state",
        default=None,
        help="Attempt-bound JSON object containing the initial Isaac UNITREE_G1_SONIC proprioception.",
    )
    parser.add_argument(
        "--action-skeleton-command",
        default=None,
        help=(
            "Controller/FK command that converts each exact learned action into "
            "per-frame skeleton landmarks and the generated robot state. Required "
            "for strict learned-policy evaluation."
        ),
    )
    parser.add_argument("--clean-frame-reanchor-interval", type=int, default=0)
    parser.add_argument("--wam-consistency-command")
    parser.add_argument("--allow-wam-consistency-scoring", action="store_true")
    parser.add_argument("--wam-consistency-timeout-seconds", type=float)
    parser.add_argument("--require-forward-inverse-consistency", action="store_true")
    parser.add_argument("--wam-success-label-command")
    parser.add_argument("--allow-wam-success-labeling", action="store_true")
    parser.add_argument("--require-generated-video-success-label", action="store_true")
    parser.add_argument("--wam-success-label-timeout-seconds", type=float)
    parser.add_argument(
        "--short-visual-sanity-manifest",
        default=None,
        help=(
            "Passed persistent_wam_short_visual_sanity_manifest.json required before "
            "paid long WAM scale-up when provider input risk flags recommend it."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the per-step OSCAR-2B <-> SAM3 closed loop. Intended to run ON a GPU pod that has the
    OSCAR repo + checkpoint and the SAM3/DA3 perception backend. ``--dry-run`` validates the full
    assembly (paths, backends, route) and writes the plan without any inference, so the wiring is
    verifiable with zero GPU.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(out_dir)
    route_payload = json.loads(Path(args.route_file).read_text(encoding="utf-8"))
    route = list(route_payload.get("route_points") or [])
    task_success_contract: dict[str, Any] = {}
    initial_groot_policy_state: dict[str, Any] = {}
    initial_observation_evidence: dict[str, Any] = {}
    start_frame_evidence_path = resolve_start_frame_evidence_path(args.start_frame_evidence)
    if start_frame_evidence_path:
        try:
            initial_observation_evidence = _mapping(
                json.loads(
                    Path(start_frame_evidence_path).expanduser().read_text(encoding="utf-8")
                )
            )
        except (OSError, json.JSONDecodeError):
            initial_observation_evidence = {}
    task_success_contract_load_blocker: str | None = None
    if args.task_success_contract:
        try:
            loaded_contract = json.loads(
                Path(args.task_success_contract).expanduser().read_text(encoding="utf-8")
            )
            task_success_contract = _mapping(loaded_contract)
            if not task_success_contract:
                task_success_contract_load_blocker = (
                    "blocked_task_success_contract_not_a_json_object"
                )
        except (OSError, json.JSONDecodeError):
            task_success_contract_load_blocker = (
                "blocked_task_success_contract_missing_or_invalid_json"
            )
    if args.groot_policy_initial_state:
        try:
            initial_groot_policy_state = _mapping(
                json.loads(
                    Path(args.groot_policy_initial_state).expanduser().read_text(encoding="utf-8")
                )
            )
        except (OSError, json.JSONDecodeError):
            initial_groot_policy_state = {}
    projected_skeleton_trace_path = materialize_projected_skeleton_trace_from_seed_geometry(
        route_payload=route_payload,
        start_frame_path=args.start_frame,
        output_dir=out_dir / "seed_conditioning",
        num_frames=int(args.num_frames),
    )
    harness_command = (
        shlex.split(args.harness_backend_command)
        if args.harness_backend_command
        else DEFAULT_SAM3_HARNESS_BACKEND_COMMAND
    )
    wam_backend_readiness = build_closed_loop_wam_backend_readiness(
        selected_backend=args.wam_backend,
        use_provider_command=bool(args.use_provider_command),
        oscar_repo=args.oscar_repo,
        checkpoint=args.checkpoint,
        oscar_provider=args.oscar_provider,
        allow_paid_provider_launch=bool(args.allow_paid_provider_launch),
    )
    seed_conditioning_preflight = build_closed_loop_seed_conditioning_preflight(
        selected_backend=args.wam_backend,
        use_provider_command=bool(args.use_provider_command),
        allow_paid_provider_launch=bool(args.allow_paid_provider_launch),
        steps=int(args.steps),
        projected_skeleton_trace_path=projected_skeleton_trace_path,
    )
    provider_input_contract_preflight = build_closed_loop_provider_input_contract_preflight(
        start_frame_path=args.start_frame,
        route_points=route,
        output_dir=out_dir / "provider_input_contract_preflight",
        task_prompt=args.task_prompt,
        selected_backend=args.wam_backend,
        use_provider_command=bool(args.use_provider_command),
        steps=int(args.steps),
        num_frames=int(args.num_frames),
        num_steps=int(args.oscar_num_steps),
        guidance=float(args.oscar_guidance),
        seed=int(args.oscar_seed),
        height=int(args.oscar_height),
        width=int(args.oscar_width),
        fps=float(args.oscar_fps),
        projected_skeleton_trace_path=projected_skeleton_trace_path,
    )
    wam_backend_readiness["provider_input_contract_preflight"] = provider_input_contract_preflight
    if provider_input_contract_preflight.get("blockers"):
        wam_backend_readiness["blockers"] = list(wam_backend_readiness.get("blockers") or []) + [
            str(item) for item in provider_input_contract_preflight.get("blockers") or []
        ]
        wam_backend_readiness["status"] = "blocked"
    short_visual_sanity_launch_plan = build_closed_loop_short_visual_sanity_launch_plan(
        selected_backend=args.wam_backend,
        use_provider_command=bool(args.use_provider_command),
        allow_paid_provider_launch=bool(args.allow_paid_provider_launch),
        steps=int(args.steps),
        provider_input_contract_preflight=provider_input_contract_preflight,
        output_dir=out_dir / "short_visual_sanity_launch_plan",
        oscar_provider=args.oscar_provider,
        task_prompt=args.task_prompt,
        timeout_seconds=float(args.provider_timeout_seconds),
    )
    wam_backend_readiness["short_visual_sanity_launch_plan"] = short_visual_sanity_launch_plan
    short_rollout_sanity_gate = build_closed_loop_short_rollout_sanity_gate(
        selected_backend=args.wam_backend,
        use_provider_command=bool(args.use_provider_command),
        allow_paid_provider_launch=bool(args.allow_paid_provider_launch),
        steps=int(args.steps),
        provider_input_contract_preflight=provider_input_contract_preflight,
        short_visual_sanity_manifest_path=args.short_visual_sanity_manifest,
        expected_policy_observation_path=short_visual_sanity_launch_plan.get(
            "policy_observation_path"
        ),
    )
    wam_backend_readiness["short_rollout_sanity_gate"] = short_rollout_sanity_gate
    if short_visual_sanity_launch_plan.get("blockers"):
        wam_backend_readiness["blockers"] = list(wam_backend_readiness.get("blockers") or []) + [
            str(item) for item in short_visual_sanity_launch_plan.get("blockers") or []
        ]
        wam_backend_readiness["status"] = "blocked"
    if short_rollout_sanity_gate.get("blockers"):
        wam_backend_readiness["blockers"] = list(wam_backend_readiness.get("blockers") or []) + [
            str(item) for item in short_rollout_sanity_gate.get("blockers") or []
        ]
        wam_backend_readiness["status"] = "blocked"
    if seed_conditioning_preflight.get("blockers"):
        wam_backend_readiness["seed_conditioning_preflight"] = seed_conditioning_preflight
        wam_backend_readiness["blockers"] = list(wam_backend_readiness.get("blockers") or []) + [
            str(item) for item in seed_conditioning_preflight.get("blockers") or []
        ]
        wam_backend_readiness["status"] = "blocked"
    else:
        wam_backend_readiness["seed_conditioning_preflight"] = seed_conditioning_preflight
    native_resolution = int(args.oscar_height) == 480 and int(args.oscar_width) == 640
    wam_backend_readiness["oscar_generation_resolution_contract"] = {
        "requested_height": int(args.oscar_height),
        "requested_width": int(args.oscar_width),
        "native_height": 480,
        "native_width": 640,
        "native_match": native_resolution,
        "override_used": bool(args.allow_non_native_oscar_resolution),
    }
    if not native_resolution and not args.allow_non_native_oscar_resolution:
        wam_backend_readiness["blockers"] = list(wam_backend_readiness.get("blockers") or []) + [
            "blocked_non_native_oscar_resolution_requires_explicit_override"
        ]
        wam_backend_readiness["status"] = "blocked"
    learned_policy_endpoint_configured = bool(
        args.learned_policy_command or args.groot_sonic_policy_server_url
    )
    strict_learned_policy_requested = bool(
        args.require_fresh_learned_policy_requery or learned_policy_endpoint_configured
    )
    strict_action_conditioning_requested = bool(
        strict_learned_policy_requested or args.wam_backend == "cosmos3_wam"
    )
    if strict_action_conditioning_requested and not _string(args.action_skeleton_command).strip():
        wam_backend_readiness["blockers"] = list(wam_backend_readiness.get("blockers") or []) + [
            "blocked_strict_evaluation_requires_action_skeleton_controller_fk_command"
        ]
        wam_backend_readiness["status"] = "blocked"
    if args.groot_sonic_policy_server_url and not initial_groot_policy_state:
        wam_backend_readiness["blockers"] = list(wam_backend_readiness.get("blockers") or []) + [
            "blocked_groot_policy_initial_attempt_bound_proprioception_missing"
        ]
        wam_backend_readiness["status"] = "blocked"
    if strict_learned_policy_requested and args.use_provider_command:
        wam_backend_readiness["blockers"] = list(wam_backend_readiness.get("blockers") or []) + [
            "blocked_provider_command_backend_lacks_fresh_action_controller_fk_hook"
        ]
        wam_backend_readiness["status"] = "blocked"
    task_kind = _string(task_success_contract.get("task_kind")).strip().lower() or (
        "manipulation" if list(args.perception_target_prompt or []) else "navigation_smoke"
    )
    manipulation_completion_requested = bool(
        args.stop_on_task_completion and task_kind not in {"navigation", "navigation_smoke"}
    )
    completion_contract_blockers: list[str] = []
    if task_success_contract_load_blocker:
        completion_contract_blockers.append(task_success_contract_load_blocker)
    if bool(args.task_success_contract) != bool(args.task_completion_command):
        completion_contract_blockers.append(
            "blocked_task_success_contract_and_completion_command_must_be_configured_together"
        )
    if args.task_completion_command and not _registered_task_criteria(task_success_contract):
        completion_contract_blockers.append(
            "blocked_task_success_contract_has_no_registered_criteria"
        )
    if manipulation_completion_requested:
        if not args.task_success_contract:
            completion_contract_blockers.append(
                "blocked_manipulation_completion_requires_task_success_contract"
            )
        if not args.task_completion_command:
            completion_contract_blockers.append(
                "blocked_manipulation_completion_requires_task_completion_command"
            )
        if int(args.no_progress_patience_steps) <= 0:
            completion_contract_blockers.append(
                "blocked_no_progress_patience_steps_must_be_positive"
            )
        if (
            not math.isfinite(float(args.minimum_task_progress_fraction))
            or float(args.minimum_task_progress_fraction) <= 0.0
        ):
            completion_contract_blockers.append(
                "blocked_minimum_task_progress_fraction_must_be_positive"
            )
    if completion_contract_blockers:
        wam_backend_readiness["blockers"] = (
            list(wam_backend_readiness.get("blockers") or []) + completion_contract_blockers
        )
        wam_backend_readiness["status"] = "blocked"
    wam_backend_readiness["blockers"] = list(
        dict.fromkeys(str(item) for item in wam_backend_readiness.get("blockers") or [])
    )
    write_json(out_dir / "closed_loop_wam_backend_readiness.json", wam_backend_readiness)
    oscar_ready = wam_backend_readiness["status"] == "ready"

    if args.dry_run or not oscar_ready:
        plan = {
            "schema_version": "oscar_isaac_closed_loop_plan.v1",
            "generated_at": utc_now_iso(),
            "status": "prepared" if oscar_ready else "blocked",
            "mode": "dry_run" if args.dry_run else "prepared",
            "start_frame": args.start_frame,
            "start_frame_present": Path(args.start_frame).expanduser().is_file(),
            "route_point_count": len(route),
            "projected_skeleton_trace_path": str(projected_skeleton_trace_path)
            if projected_skeleton_trace_path
            else None,
            "steps": int(args.steps),
            "task_prompt": args.task_prompt,
            "num_frames_per_step": int(args.num_frames),
            "oscar_runtime_settings": {
                "num_frames": int(args.num_frames),
                "num_steps": int(args.oscar_num_steps),
                "guidance": float(args.oscar_guidance),
                "seed": int(args.oscar_seed),
                "height": int(args.oscar_height),
                "width": int(args.oscar_width),
                "fps": float(args.oscar_fps),
            },
            "selected_wam_backend": args.wam_backend,
            "wam_backend_readiness_path": str(out_dir / "closed_loop_wam_backend_readiness.json"),
            "wam_backend_readiness": wam_backend_readiness,
            "seed_conditioning_preflight": seed_conditioning_preflight,
            "provider_input_contract_preflight": provider_input_contract_preflight,
            "short_rollout_sanity_gate": short_rollout_sanity_gate,
            "short_visual_sanity_launch_plan": short_visual_sanity_launch_plan,
            "use_provider_command": bool(args.use_provider_command),
            "oscar_provider": args.oscar_provider,
            "allow_paid_provider_launch": bool(args.allow_paid_provider_launch),
            "oscar_repo": args.oscar_repo,
            "checkpoint_configured": bool(args.checkpoint),
            "harness_backend_kind": args.harness_backend_kind,
            "harness_backend_command_argv0": harness_command[0] if harness_command else None,
            "perception_target_prompts": list(args.perception_target_prompt or []),
            "requirements": {
                "fresh_oscar_provider_required": bool(args.require_fresh_oscar_provider),
                "real_perception_backend_required": bool(args.require_real_perception_backend),
                "sam3_completed_required": bool(args.require_sam3_completed),
                "da3_completed_required": bool(args.require_da3_completed),
                "fresh_learned_policy_requery_required": bool(strict_learned_policy_requested),
                "fresh_action_controller_fk_conditioning_required": bool(
                    strict_action_conditioning_requested
                ),
                "action_skeleton_controller_fk_command_configured": bool(
                    _string(args.action_skeleton_command).strip()
                ),
                "task_success_contract_configured": bool(task_success_contract),
                "task_completion_command_configured": bool(args.task_completion_command),
                "manipulation_completion_requested": manipulation_completion_requested,
                "online_no_progress_termination_enabled": manipulation_completion_requested,
                "no_progress_patience_steps": int(args.no_progress_patience_steps),
                "minimum_task_progress_fraction": float(args.minimum_task_progress_fraction),
                "clean_frame_reanchor_interval": int(args.clean_frame_reanchor_interval),
                "wam_episode_consistency_scoring_configured": bool(
                    args.allow_wam_consistency_scoring or args.wam_consistency_command
                ),
                "wam_episode_consistency_command_configured": bool(args.wam_consistency_command),
                "forward_inverse_consistency_required": bool(
                    args.require_forward_inverse_consistency
                ),
                "synchronized_calibrated_multiview_required": bool(
                    args.require_forward_inverse_consistency
                ),
                "wam_success_labeling_configured": bool(
                    args.allow_wam_success_labeling or args.wam_success_label_command
                ),
                "wam_success_label_command_configured": bool(args.wam_success_label_command),
                "generated_video_success_label_required": bool(
                    args.require_generated_video_success_label
                ),
            },
            "blockers": list(wam_backend_readiness.get("blockers") or []),
        }
        write_json(out_dir / "oscar_isaac_closed_loop_plan.json", plan)
        print(json.dumps({"status": plan["status"], "mode": plan["mode"]}, sort_keys=True))
        return 0 if plan["status"] in {"prepared"} else 2

    skeleton_projector = (
        make_controller_fk_skeleton_projector(
            command=args.action_skeleton_command,
            work_dir=out_dir / "controller_fk_skeleton",
            timeout_seconds=float(args.provider_timeout_seconds),
        )
        if _string(args.action_skeleton_command).strip()
        else None
    )
    # Bound before the backend branches so teardown below is always reachable.
    resident_worker = None
    if args.wam_backend == "cosmos3_wam":
        cosmos3_command_env = WAM_PROVIDER_COMMAND_ENV_BY_SUBSTRATE.get("cosmos3_wam")
        cosmos3_command = _string(
            os.environ.get(cosmos3_command_env or "")
            or os.environ.get("BLUEPRINT_WAM_PROVIDER_COMMAND")
        ).strip()
        if skeleton_projector is None:
            raise RuntimeError("cosmos3_closed_loop_requires_controller_fk_projector")
        backend = make_cosmos3_per_step_command_wam_backend(
            command=cosmos3_command,
            work_dir=out_dir / "cosmos3_generation",
            task_prompt=args.task_prompt,
            skeleton_for_action=skeleton_projector,
            timeout_seconds=float(args.provider_timeout_seconds),
        )
    elif args.use_provider_command:
        backend = make_oscar_provider_command_wam_backend(
            work_dir=out_dir / "oscar_generation",
            task_prompt=args.task_prompt,
            num_frames=int(args.num_frames),
            num_steps=int(args.oscar_num_steps),
            guidance=float(args.oscar_guidance),
            seed=int(args.oscar_seed),
            height=int(args.oscar_height),
            width=int(args.oscar_width),
            fps=float(args.oscar_fps),
            provider=args.oscar_provider,
            allow_paid_provider_launch=bool(args.allow_paid_provider_launch),
            timeout_seconds=float(args.provider_timeout_seconds),
            projected_skeleton_trace_path=projected_skeleton_trace_path,
        )
    else:
        import subprocess

        def _step_skeleton_video_builder(
            trace_or_landmarks: Sequence[Mapping[str, Any]],
            step_out_dir: Path,
        ) -> Path | None:
            """Render this step's projected-skeleton conditioning clip.

            OSCAR's inference CLI requires --skeleton-video; without landmarks
            there is nothing truthful to render, so return None and let the
            step fail closed rather than fabricating conditioning.
            """
            from .oscar_wam_provider_bundle import (
                _render_projected_skeleton_conditioning_video,
            )

            trace_rows = [
                dict(row)
                for row in trace_or_landmarks
                if isinstance(row, Mapping)
                and (
                    isinstance(row.get("projected_landmarks"), Sequence)
                    or isinstance(row.get("landmarks"), Sequence)
                )
            ]
            if trace_rows:
                trace_path = Path(step_out_dir) / "step_skeleton_trace.jsonl"
                output_frame_count = max(1, int(args.num_frames))
                with trace_path.open("w", encoding="utf-8") as handle:
                    for frame_index in range(output_frame_count):
                        source_index = (
                            0
                            if output_frame_count == 1
                            else round(
                                frame_index * (len(trace_rows) - 1) / (output_frame_count - 1)
                            )
                        )
                        source_row = trace_rows[source_index]
                        handle.write(
                            json.dumps(
                                {
                                    "frame_index": frame_index,
                                    "source_controller_horizon_frame_index": (
                                        source_row.get(
                                            "source_controller_horizon_frame_index",
                                            source_index,
                                        )
                                    ),
                                    "projected_landmarks": [
                                        dict(landmark)
                                        for landmark in (
                                            source_row.get("projected_landmarks")
                                            or source_row.get("landmarks")
                                            or []
                                        )
                                        if isinstance(landmark, Mapping)
                                    ],
                                }
                            )
                            + "\n"
                        )
            elif trace_or_landmarks:
                # Compatibility path for non-horizon fixtures. The live
                # manipulation lane supplies the full controller FK sequence
                # above and cannot take this branch.
                trace_path = Path(step_out_dir) / "step_skeleton_trace.jsonl"
                with trace_path.open("w", encoding="utf-8") as handle:
                    for frame_index in range(max(1, int(args.num_frames))):
                        handle.write(
                            json.dumps(
                                {
                                    "frame_index": frame_index,
                                    "projected_landmarks": [
                                        dict(landmark) for landmark in trace_or_landmarks
                                    ],
                                }
                            )
                            + "\n"
                        )
            elif projected_skeleton_trace_path and Path(projected_skeleton_trace_path).is_file():
                # No per-step landmarks: condition on the materialized SEED
                # skeleton trace (truthfully the seed pose, not this step's
                # action). Recorded via the trace filename.
                trace_path = Path(projected_skeleton_trace_path)
            else:
                return None
            output_path = Path(step_out_dir) / "step_skeleton_conditioning.mp4"
            try:
                render_report, _ = _render_projected_skeleton_conditioning_video(
                    trace_path=trace_path,
                    output_path=output_path,
                    width=int(args.oscar_width),
                    height=int(args.oscar_height),
                    fps=float(args.oscar_fps),
                    num_frames=max(1, int(args.num_frames)),
                    conditioning_mode="controller_fk_action_horizon",
                )
            except Exception:
                return None
            visual_signal = _mapping(render_report.get("visual_signal"))
            if visual_signal.get("status") != "completed" or list(
                visual_signal.get("blockers") or []
            ):
                return None
            return output_path if output_path.is_file() else None

        if args.oscar_resident_worker:
            from .oscar_resident_worker import start_resident_oscar_generate_from_args

            resident_worker, oscar_generate = start_resident_oscar_generate_from_args(
                args,
                python=sys.executable,
                build_skeleton_video=_step_skeleton_video_builder,
                extract_next_frame=extract_next_observation_frame_from_video,
            )
        else:
            oscar_generate = make_local_oscar_subprocess_generate(
                oscar_repo=args.oscar_repo,
                checkpoint=args.checkpoint,
                # The sealed worker deliberately keeps GR00T, OSCAR, and Isaac in
                # separate interpreters.  The closed-loop process itself is
                # launched with /opt/oscar-venv/bin/python, but that venv is not on
                # PATH; a bare `python` here can therefore select the system
                # interpreter and fail at the first OSCAR import.
                python=sys.executable,
                num_steps=int(args.oscar_num_steps),
                guidance=float(args.oscar_guidance),
                height=int(args.oscar_height),
                width=int(args.oscar_width),
                fps=float(args.oscar_fps),
                timeout_seconds=float(args.provider_timeout_seconds),
                run=subprocess.run,
                popen=subprocess.Popen,
                gpu_query_run=subprocess.run,
                build_skeleton_video=_step_skeleton_video_builder,
                extract_next_frame=extract_next_observation_frame_from_video,
            )
        backend = make_oscar_per_step_wam_backend(
            oscar_generate=oscar_generate,
            work_dir=out_dir / "oscar_generation",
            task_prompt=args.task_prompt,
            num_frames=int(args.num_frames),
            skeleton_for_action=skeleton_projector,
            seed=int(args.oscar_seed),
            require_manipulation_effector_progress=manipulation_completion_requested,
        )
    policy_endpoint = None
    if args.learned_policy_command and args.groot_sonic_policy_server_url:
        parser.error(
            "choose exactly one of --learned-policy-command or --groot-sonic-policy-server-url"
        )
    if args.learned_policy_command:
        policy_endpoint = make_learned_policy_command_endpoint(
            command=args.learned_policy_command,
            work_dir=out_dir / "learned_policy_endpoint",
            timeout_seconds=float(args.provider_timeout_seconds),
        )
    elif args.groot_sonic_policy_server_url:
        from .groot_sonic_policy_endpoint import make_groot_sonic_zmq_policy_endpoint

        policy_endpoint = make_groot_sonic_zmq_policy_endpoint(
            policy_server_url=args.groot_sonic_policy_server_url,
            groot_root=args.groot_root,
            sonic_state=initial_groot_policy_state,
            execution_frame_count=int(args.groot_sonic_execution_frame_count),
        )
    task_completion_evaluator = (
        make_task_completion_command_evaluator(
            command=args.task_completion_command,
            work_dir=out_dir / "task_completion_evaluator",
            timeout_seconds=float(args.provider_timeout_seconds),
        )
        if args.task_completion_command
        else None
    )
    try:
        manifest = run_oscar_isaac_closed_loop(
            output_dir=out_dir,
            start_frame_path=args.start_frame,
            route_points=route,
            wam_generate_next=backend,
            policy_endpoint=policy_endpoint,
            initial_observation_evidence=initial_observation_evidence,
            steps=int(args.steps),
            task_prompt=args.task_prompt,
            harness_backend_kind=args.harness_backend_kind,
            harness_backend_command=harness_command,
            allow_external_backend=args.harness_backend_kind
            not in {"fixture", GENERATED_RGB_POLICY_OBSERVATION_BACKEND_KIND},
            require_fresh_oscar_provider=bool(args.require_fresh_oscar_provider),
            require_real_perception_backend=bool(args.require_real_perception_backend),
            require_sam3_completed=bool(args.require_sam3_completed),
            require_da3_completed=bool(args.require_da3_completed),
            require_fresh_learned_policy_requery=strict_learned_policy_requested,
            require_action_derived_skeleton_conditioning=(strict_action_conditioning_requested),
            clean_frame_reanchor_interval=int(args.clean_frame_reanchor_interval),
            perception_target_prompts=list(args.perception_target_prompt or []),
            wam_backend_id=args.wam_backend,
            wam_backend_readiness=wam_backend_readiness,
            wam_consistency_command=args.wam_consistency_command,
            allow_wam_consistency_scoring=bool(args.allow_wam_consistency_scoring),
            wam_consistency_timeout_seconds=args.wam_consistency_timeout_seconds,
            require_forward_inverse_consistency=bool(args.require_forward_inverse_consistency),
            require_synchronized_calibrated_multiview=bool(args.require_forward_inverse_consistency),
            wam_success_label_command=args.wam_success_label_command,
            allow_wam_success_labeling=bool(args.allow_wam_success_labeling),
            require_generated_video_success_label=bool(args.require_generated_video_success_label),
            wam_success_label_timeout_seconds=args.wam_success_label_timeout_seconds,
            min_coherent_horizon_frames=int(args.min_coherent_horizon_frames),
            stop_on_task_completion=bool(args.stop_on_task_completion),
            stop_on_unsafe_stance=manipulation_completion_requested,
            stop_on_no_progress=manipulation_completion_requested,
            no_progress_patience_steps=int(args.no_progress_patience_steps),
            minimum_task_progress_fraction=float(args.minimum_task_progress_fraction),
            min_steps=int(args.min_steps),
            task_success_contract=task_success_contract,
            task_completion_evaluator=task_completion_evaluator,
            attempt_input_manifest=args.attempt_input_manifest,
        )
    finally:
        # Holds the GPU for the whole rollout, so it is torn down on the failure
        # path too, after its throughput report is written.
        if resident_worker is not None:
            resident_worker.close_and_report(out_dir)
    print(
        json.dumps(
            {"status": manifest["status"], "steps_executed": manifest.get("steps_executed")},
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
