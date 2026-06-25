"""Run a persistent Vast Unitree GR00T/SONIC + WAM session.

This runner exists to avoid the fragile pattern of allocating a fresh GPU
provider instance for each policy or WAM step. It stages one provider bundle
whose remote entrypoint starts a local policy worker and a local WAM worker,
calls their ``/infer`` endpoints repeatedly, and lets the Vast adapter tear the
single instance down after the session output is uploaded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .unitree_groot_n17_sonic_policy_runtime import POLICY_ID
from .unitree_groot_n17_sonic_vast_policy_command import (
    ALLOWED_MACHINE_ID_ENVS,
    ALLOW_UNPINNED_FALLBACK_ENV,
    EXCLUDED_MACHINE_ID_ENVS,
    INNER_POLICY_COMMAND_ENV,
    OBJECT_STORE_KEY_PREFIX_ENV,
    PUBLIC_IMAGE_ENV as UNITREE_PUBLIC_IMAGE_ENV,
    VAST_LAUNCH_MODE_ENV,
)
from .vast_provider_adapter import (
    DEFAULT_PUBLIC_CUDA_IMAGE,
    VAST_IMAGE_LOGIN_MODE_ENV,
    run_vast_provider_adapter,
)
from .vast_wam_authorized_runner import DEFAULT_WAM_PUBLIC_IMAGE
from .wam_provider_object_store import stage_wam_provider_bundle_object_store
from .runpod_wam_async_runner import create_runpod_wam_async_run, poll_runpod_wam_async_run
from .image_model_render_remediation import (
    ENABLE_ENV as IMAGE_MODEL_RENDER_REMEDIATION_ENABLE_ENV,
    image_model_render_remediation_enabled,
    run_image_model_render_remediation,
)
from .wam_auxiliary_observation import (
    build_wam_auxiliary_observation_manifest,
    summarize_wam_auxiliary_observation_manifest,
)
from .wam_generated_video_review import (
    REVIEW_QUALITY_MIN_FPS,
    REVIEW_QUALITY_MIN_HEIGHT,
    REVIEW_QUALITY_MIN_NUM_FRAMES,
    REVIEW_QUALITY_MIN_WIDTH,
    assess_source_policy_observation_visual_qa,
    write_persistent_wam_visual_quality_artifacts,
)


SCHEMA_VERSION = "unitree_groot_n17_sonic_vast_persistent_session.v1"
BUNDLE_SCHEMA_VERSION = "unitree_groot_n17_sonic_wam_persistent_session_bundle.v1"
OUTPUT_SCHEMA_VERSION = "unitree_groot_n17_sonic_wam_persistent_session_output.v1"
DEFAULT_BUNDLE_FILENAME = "unitree_groot_n17_sonic_wam_persistent_session_bundle.zip"
DEFAULT_OBJECT_STORE_KEY_PREFIX = "blueprint/unitree-groot-sonic-persistent-session"
PERSISTENT_SESSION_JOB_ROOT_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PERSISTENT_SESSION_JOB_ROOT"
PERSISTENT_SESSION_PUBLIC_IMAGE_ENV = "BLUEPRINT_VAST_UNITREE_WAM_PERSISTENT_SESSION_PUBLIC_IMAGE"
PERSISTENT_SESSION_ALLOW_STRUCTURAL_WAM_FALLBACK_ENV = (
    "BLUEPRINT_ALLOW_PERSISTENT_SESSION_STRUCTURAL_WAM_FALLBACK"
)
PERSISTENT_SESSION_USE_LIVE_WAM_ENV = "BLUEPRINT_PERSISTENT_SESSION_USE_LIVE_WAM"
PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV = (
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PERSISTENT_INNER_POLICY_COMMAND"
)
RUNPOD_FULL_LOOP_OVERRIDE_ENV = "BLUEPRINT_ALLOW_UNITREE_GROOT_N17_SONIC_RUNPOD_FULL_LOOP"
OSCAR_WAM_VISUAL_PROFILE_ENV = "BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE"
PERSISTENT_WAM_LONG_REVIEW_ROLLOUT_ENV = "BLUEPRINT_ALLOW_PERSISTENT_WAM_LONG_REVIEW_ROLLOUT"
PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST_ENV = (
    "BLUEPRINT_PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST"
)
PERSISTENT_WAM_SHORT_VISUAL_SANITY_SCHEMA_VERSION = "persistent_wam_short_visual_sanity.v1"
PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_WIDTH = 320
PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_HEIGHT = 256
PERSISTENT_WAM_REVIEW_QUALITY_MAX_UNGATED_LOOP_STEPS_ENV = (
    "BLUEPRINT_PERSISTENT_WAM_REVIEW_QUALITY_MAX_UNGATED_LOOP_STEPS"
)
PERSISTENT_WAM_CLEAN_FRAME_REANCHOR_INTERVAL_ENV = (
    "BLUEPRINT_PERSISTENT_WAM_CLEAN_FRAME_REANCHOR_INTERVAL_STEPS"
)
PERSISTENT_WAM_AUTOREGRESSIVE_DRIFT_BLOCKER_MANIFEST_ENV = (
    "BLUEPRINT_PERSISTENT_WAM_AUTOREGRESSIVE_DRIFT_BLOCKER_MANIFEST"
)
PERSISTENT_WAM_LONG_REVIEW_ROLLOUT_MIN_STEPS = 12
PERSISTENT_WAM_LONG_REVIEW_QUALITY_GATE_SCHEMA_VERSION = (
    "persistent_wam_long_review_rollout_quality_gate.v1"
)
SYNTHETIC_FALLBACK_WAM_LAUNCH_EXPERIMENT_ENV = (
    "BLUEPRINT_ALLOW_SYNTHETIC_FALLBACK_WAM_LAUNCH_EXPERIMENT"
)
SYNTHETIC_FALLBACK_WAM_SOURCE_KINDS = {
    "synthetic_fallback",
    "synthetic_gpt_image_2_seed",
}
DEFAULT_INNER_POLICY_COMMAND = (
    "python -m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
)
RUNPOD_WAM_CARRIER_SMOKE_DEFAULT_ENV = {
    OSCAR_WAM_VISUAL_PROFILE_ENV: "smoke",
    "BLUEPRINT_OSCAR_WAM_NUM_STEPS": "2",
    "BLUEPRINT_OSCAR_WAM_NUM_FRAMES": "9",
    "BLUEPRINT_OSCAR_WAM_HEIGHT": "128",
    "BLUEPRINT_OSCAR_WAM_WIDTH": "128",
    "BLUEPRINT_OSCAR_WAM_FPS": "4",
    "BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS": "1200",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE": "system_python_minimal",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT": "true",
}
RUNPOD_WAM_CARRIER_REVIEW_QUALITY_DEFAULT_ENV = {
    OSCAR_WAM_VISUAL_PROFILE_ENV: "review_quality",
    "BLUEPRINT_OSCAR_WAM_NUM_STEPS": "2",
    "BLUEPRINT_OSCAR_WAM_NUM_FRAMES": "24",
    "BLUEPRINT_OSCAR_WAM_HEIGHT": "480",
    "BLUEPRINT_OSCAR_WAM_WIDTH": "640",
    "BLUEPRINT_OSCAR_WAM_FPS": "15",
    "BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS": "1200",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE": "system_python_minimal",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT": "true",
}
RUNPOD_WAM_CARRIER_DEFAULT_ENV = RUNPOD_WAM_CARRIER_SMOKE_DEFAULT_ENV
RUNPOD_WAM_CARRIER_ENV_KEYS = tuple(
    sorted(
        set(RUNPOD_WAM_CARRIER_SMOKE_DEFAULT_ENV)
        | set(RUNPOD_WAM_CARRIER_REVIEW_QUALITY_DEFAULT_ENV)
    )
)
RUNPOD_WAM_CARRIER_MIN_OSCAR_NUM_FRAMES = 5


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_env(name: str, default: float) -> float:
    try:
        return float(_string(os.getenv(name)) or default)
    except ValueError:
        return float(default)


def _int_env(name: str, default: int) -> int:
    try:
        return int(_string(os.getenv(name)) or default)
    except ValueError:
        return int(default)


def _normalized_wam_visual_profile(value: str | None = None) -> str:
    profile = _string(value if value is not None else os.getenv(OSCAR_WAM_VISUAL_PROFILE_ENV))
    return profile if profile in {"smoke", "review_quality"} else "smoke"


def _runpod_wam_carrier_defaults_for_profile(profile: str) -> dict[str, str]:
    return dict(
        RUNPOD_WAM_CARRIER_REVIEW_QUALITY_DEFAULT_ENV
        if profile == "review_quality"
        else RUNPOD_WAM_CARRIER_SMOKE_DEFAULT_ENV
    )


def _current_wam_visual_profile_settings() -> dict[str, Any]:
    profile = _normalized_wam_visual_profile()
    defaults = _runpod_wam_carrier_defaults_for_profile(profile)
    return {
        "schema_version": "persistent_wam_visual_profile_settings.v1",
        "visual_profile": profile,
        "num_steps": _int_env(
            "BLUEPRINT_OSCAR_WAM_NUM_STEPS",
            int(defaults["BLUEPRINT_OSCAR_WAM_NUM_STEPS"]),
        ),
        "num_frames": _int_env(
            "BLUEPRINT_OSCAR_WAM_NUM_FRAMES",
            int(defaults["BLUEPRINT_OSCAR_WAM_NUM_FRAMES"]),
        ),
        "height": _int_env(
            "BLUEPRINT_OSCAR_WAM_HEIGHT",
            int(defaults["BLUEPRINT_OSCAR_WAM_HEIGHT"]),
        ),
        "width": _int_env(
            "BLUEPRINT_OSCAR_WAM_WIDTH",
            int(defaults["BLUEPRINT_OSCAR_WAM_WIDTH"]),
        ),
        "fps": _float_env(
            "BLUEPRINT_OSCAR_WAM_FPS",
            float(defaults["BLUEPRINT_OSCAR_WAM_FPS"]),
        ),
        "review_quality_minimum": {
            "width": REVIEW_QUALITY_MIN_WIDTH,
            "height": REVIEW_QUALITY_MIN_HEIGHT,
            "fps": REVIEW_QUALITY_MIN_FPS,
            "num_frames": REVIEW_QUALITY_MIN_NUM_FRAMES,
        },
        "smoke_only": profile != "review_quality",
    }


def _resolve_optional_path(value: Any) -> Path | None:
    text = _string(value)
    return Path(text).expanduser().resolve() if text else None


def _existing_artifact_path_blocker(
    payload: Mapping[str, Any],
    key: str,
    blocker: str,
) -> str | None:
    path = _resolve_optional_path(payload.get(key))
    if path is None or not path.is_file():
        return blocker
    return None


def _first_ffprobe_video_stream(metadata: Mapping[str, Any]) -> dict[str, Any]:
    streams = metadata.get("streams")
    if not isinstance(streams, Sequence) or isinstance(streams, (str, bytes, bytearray)):
        return {}
    for stream in streams:
        if isinstance(stream, Mapping):
            return dict(stream)
    return {}


def _intish(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_persistent_wam_short_visual_sanity_manifest(
    path: str | Path | None,
    *,
    policy_observation_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the short review-quality sanity pass required before long WAM rollouts."""
    manifest_path = _resolve_optional_path(path)
    blockers: list[str] = []
    payload: dict[str, Any] = {}
    if manifest_path is None:
        blockers.append("short_visual_sanity_manifest_env_missing")
    elif not manifest_path.is_file():
        blockers.append("short_visual_sanity_manifest_missing")
    else:
        try:
            payload = _read_json(manifest_path)
        except Exception as exc:
            blockers.append(f"short_visual_sanity_manifest_unreadable:{type(exc).__name__}")
            payload = {}

    if payload:
        if payload.get("schema_version") != PERSISTENT_WAM_SHORT_VISUAL_SANITY_SCHEMA_VERSION:
            blockers.append("short_visual_sanity_manifest_schema_mismatch")
        if payload.get("status") != "passed_short_visual_sanity":
            blockers.append("short_visual_sanity_status_not_passed")
        if payload.get("short_visual_sanity_passed") is not True:
            blockers.append("short_visual_sanity_pass_flag_missing")
        if payload.get("visual_profile") != "review_quality":
            blockers.append("short_visual_sanity_not_review_quality_profile")
        try:
            requested_transition_count = int(payload.get("requested_transition_count") or 0)
        except (TypeError, ValueError):
            requested_transition_count = 0
        if requested_transition_count not in {1, 2}:
            blockers.append("short_visual_sanity_transition_count_not_1_or_2")
        try:
            live_wam_generation_success_count = int(
                payload.get("live_wam_generation_success_count") or 0
            )
        except (TypeError, ValueError):
            live_wam_generation_success_count = 0
        try:
            learned_wam_model_success_count = int(
                payload.get("learned_wam_model_success_count") or 0
            )
        except (TypeError, ValueError):
            learned_wam_model_success_count = 0
        if live_wam_generation_success_count < requested_transition_count:
            blockers.append("short_visual_sanity_live_wam_transition_count_not_passed")
        if learned_wam_model_success_count < requested_transition_count:
            blockers.append("short_visual_sanity_learned_wam_transition_count_not_passed")
        if payload.get("structural_fallback_used") is True:
            blockers.append("short_visual_sanity_structural_fallback_cannot_unlock_long_rollout")
        if (
            payload.get("source_policy_observation_visual_qa_status")
            != "passed_visual_quality_gate"
        ):
            blockers.append("short_visual_sanity_source_observation_qa_not_passed")
        if payload.get("wam_rollout_visual_success") is not True:
            blockers.append("short_visual_sanity_wam_visual_success_not_passed")
        if payload.get("ffprobe_command_ran") is not True:
            blockers.append("short_visual_sanity_ffprobe_command_not_ran")
        ffprobe_returncode = _intish(payload.get("ffprobe_returncode"))
        if ffprobe_returncode != 0:
            blockers.append("short_visual_sanity_ffprobe_returncode_not_zero")
        ffprobe_metadata = _mapping(payload.get("ffprobe_metadata"))
        if not ffprobe_metadata:
            blockers.append("short_visual_sanity_ffprobe_metadata_missing")
        else:
            stream = _first_ffprobe_video_stream(ffprobe_metadata)
            width = _intish(stream.get("width")) or 0
            height = _intish(stream.get("height")) or 0
            if (
                width < PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_WIDTH
                or height < PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_HEIGHT
            ):
                blockers.append("short_visual_sanity_review_video_below_minimum_resolution")
        for key, blocker in (
            (
                "source_policy_observation_visual_qa_path",
                "short_visual_sanity_source_qa_artifact_missing",
            ),
            (
                "wam_rollout_visual_quality_report_path",
                "short_visual_sanity_quality_report_missing",
            ),
            ("wam_rollout_contact_sheet_path", "short_visual_sanity_contact_sheet_missing"),
            ("video_review_status_path", "short_visual_sanity_video_status_missing"),
            ("review_video_path", "short_visual_sanity_review_video_missing"),
        ):
            artifact_blocker = _existing_artifact_path_blocker(payload, key, blocker)
            if artifact_blocker:
                blockers.append(artifact_blocker)
        if policy_observation_path is not None:
            expected = Path(policy_observation_path).expanduser().resolve()
            observed = _resolve_optional_path(payload.get("policy_observation_path"))
            if observed != expected:
                blockers.append("short_visual_sanity_policy_observation_mismatch")
        paid_provider = _mapping(payload.get("paid_provider"))
        if paid_provider.get("used") is True:
            if paid_provider.get("continuing_spend_from_this_run") is not False:
                blockers.append("short_visual_sanity_paid_provider_teardown_not_zero_spend")
            if paid_provider.get("teardown_status") not in {
                "completed",
                "not_required_prelaunch_blocked",
                "not_required_no_paid_provider",
            }:
                blockers.append("short_visual_sanity_paid_provider_teardown_status_not_completed")
            teardown_path = _resolve_optional_path(paid_provider.get("teardown_manifest_path"))
            if teardown_path is None or not teardown_path.is_file():
                blockers.append("short_visual_sanity_paid_provider_teardown_manifest_missing")

    return {
        "schema_version": "persistent_wam_short_visual_sanity_gate_validation.v1",
        "status": "passed_short_visual_sanity" if not blockers else "blocked",
        "manifest_path": str(manifest_path) if manifest_path else None,
        "blockers": sorted(set(blockers)),
        "short_visual_sanity_passed": not blockers,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def validate_persistent_wam_autoregressive_drift_blocker(
    path: str | Path | None,
) -> dict[str, Any]:
    """Validate a prior visual report that proves long autoregressive drift."""
    manifest_path = _resolve_optional_path(path)
    blockers: list[str] = []
    payload: dict[str, Any] = {}
    if manifest_path is None:
        blockers.append("autoregressive_drift_blocker_manifest_env_missing")
    elif not manifest_path.is_file():
        blockers.append("autoregressive_drift_blocker_manifest_missing")
    else:
        try:
            payload = _read_json(manifest_path)
        except Exception as exc:
            blockers.append(f"autoregressive_drift_blocker_manifest_unreadable:{type(exc).__name__}")
            payload = {}

    concrete_drift_blocker = False
    if payload:
        report_blockers = {str(item) for item in payload.get("blockers") or [] if str(item)}
        guard = _mapping(payload.get("autoregressive_chain_guard"))
        try:
            generated_frame_count = int(
                payload.get("generated_frame_count") or guard.get("generated_frame_count") or 0
            )
        except (TypeError, ValueError):
            generated_frame_count = 0
        concrete_drift_blocker = bool(
            payload.get("visual_success") is False
            and generated_frame_count >= 3
            and (
                "autoregressive_chain_visual_drift_or_quality_blocked_long_rollout"
                in report_blockers
                or guard.get("long_horizon_visual_drift_blocker") is True
                or guard.get("long_rollout_should_not_be_overclaimed") is True
            )
        )
        if payload.get("visual_success") is not False:
            blockers.append("autoregressive_drift_blocker_visual_success_not_false")
        if generated_frame_count < 3:
            blockers.append("autoregressive_drift_blocker_needs_multi_transition_evidence")
        if not concrete_drift_blocker:
            blockers.append("autoregressive_drift_blocker_not_concrete")

    return {
        "schema_version": "persistent_wam_autoregressive_drift_blocker_validation.v1",
        "status": "confirmed_autoregressive_drift_blocker"
        if payload and concrete_drift_blocker and not blockers
        else "blocked",
        "manifest_path": str(manifest_path) if manifest_path else None,
        "concrete_autoregressive_drift_blocker_proven": bool(
            payload and concrete_drift_blocker and not blockers
        ),
        "blockers": sorted(set(blockers)),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def _clean_frame_reanchoring_settings(
    *,
    loop_step_count: int,
    max_unanchored_steps: int,
) -> dict[str, Any]:
    interval = _int_env(PERSISTENT_WAM_CLEAN_FRAME_REANCHOR_INTERVAL_ENV, 0)
    enabled = interval > 0
    blockers: list[str] = []
    expected_reanchors: list[int] = []
    if enabled:
        if interval > max(1, int(max_unanchored_steps)):
            blockers.append("clean_frame_reanchor_interval_exceeds_short_sanity_horizon")
        expected_reanchors = [
            transition_index
            for transition_index in range(interval, max(1, int(loop_step_count)), interval)
        ]
        if not expected_reanchors:
            blockers.append("clean_frame_reanchor_interval_produces_no_pre_final_reanchor")
    return {
        "schema_version": "persistent_wam_clean_frame_reanchoring.v1",
        "enabled": bool(enabled),
        "interval_steps": int(interval) if enabled else None,
        "max_unanchored_autoregressive_steps": int(max_unanchored_steps),
        "source_frame_kind": "initial_policy_observation_clean_frame",
        "reanchor_policy": (
            "reset_policy_observation_frame_after_completed_transition"
            if enabled
            else "disabled"
        ),
        "expected_reanchor_transition_indices": expected_reanchors,
        "periodic_clean_frame_reanchoring_proven": bool(enabled and not blockers),
        "blockers": sorted(set(blockers)),
    }


def _persistent_wam_long_review_rollout_quality_gate(
    *,
    settings: Mapping[str, Any],
    loop_step_count: int,
) -> dict[str, Any]:
    max_unanchored_steps = _int_env(
        PERSISTENT_WAM_REVIEW_QUALITY_MAX_UNGATED_LOOP_STEPS_ENV,
        3,
    )
    reanchoring = _clean_frame_reanchoring_settings(
        loop_step_count=loop_step_count,
        max_unanchored_steps=max_unanchored_steps,
    )
    required = bool(
        settings.get("visual_profile") == "review_quality"
        and int(loop_step_count) >= PERSISTENT_WAM_LONG_REVIEW_ROLLOUT_MIN_STEPS
    )
    drift_validation = validate_persistent_wam_autoregressive_drift_blocker(
        os.getenv(PERSISTENT_WAM_AUTOREGRESSIVE_DRIFT_BLOCKER_MANIFEST_ENV)
    )
    blockers: list[str] = []
    status = "not_required"
    paid_launch_allowed = True
    if required:
        if reanchoring.get("periodic_clean_frame_reanchoring_proven") is True:
            status = "passed_periodic_clean_frame_reanchoring"
        elif drift_validation.get("status") == "confirmed_autoregressive_drift_blocker":
            status = "blocked_autoregressive_drift_confirmed"
            paid_launch_allowed = False
            blockers.append(
                "autoregressive_chain_drift_blocker_present_before_12_step_paid_rollout"
            )
        else:
            status = "blocked_missing_long_rollout_quality_proof"
            paid_launch_allowed = False
            blockers.append(
                "review_quality_12_step_paid_rollout_requires_clean_frame_reanchoring_or_drift_blocker"
            )
            blockers.extend(str(item) for item in reanchoring.get("blockers") or [])
            blockers.extend(str(item) for item in drift_validation.get("blockers") or [])

    return {
        "schema_version": PERSISTENT_WAM_LONG_REVIEW_QUALITY_GATE_SCHEMA_VERSION,
        "status": status,
        "required_before_12_step_paid_review_quality_rollout": required,
        "visual_profile": settings.get("visual_profile"),
        "loop_step_count": int(loop_step_count),
        "min_long_review_rollout_steps": PERSISTENT_WAM_LONG_REVIEW_ROLLOUT_MIN_STEPS,
        "paid_rollout_launch_allowed": bool(paid_launch_allowed and not blockers),
        "clean_frame_reanchoring": reanchoring,
        "drift_blocker_validation": drift_validation,
        "periodic_clean_frame_reanchoring_proven": bool(
            reanchoring.get("periodic_clean_frame_reanchoring_proven")
        ),
        "concrete_autoregressive_drift_blocker_proven": bool(
            drift_validation.get("concrete_autoregressive_drift_blocker_proven")
        ),
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "clean_frame_reanchoring_is_quality_control_not_task_success": True,
            "autoregressive_drift_blocker_prevents_paid_long_rollout": bool(
                status == "blocked_autoregressive_drift_confirmed"
            ),
            "long_rollout_quality_gate_is_not_physical_robot_readiness": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "real_world_manipulation_success_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def _persistent_wam_visual_profile_blockers(
    *,
    settings: Mapping[str, Any],
    source_visual_qa: Mapping[str, Any],
    loop_step_count: int,
    policy_observation_path: str | Path | None = None,
) -> list[str]:
    blockers: list[str] = []
    if settings.get("visual_profile") != "review_quality":
        return blockers
    if source_visual_qa.get("status") != "passed_visual_quality_gate":
        blockers.append("source_policy_observation_visual_qa_failed_for_review_quality")
    if int(settings.get("width") or 0) < REVIEW_QUALITY_MIN_WIDTH:
        blockers.append("review_quality_profile_width_below_minimum")
    if int(settings.get("height") or 0) < REVIEW_QUALITY_MIN_HEIGHT:
        blockers.append("review_quality_profile_height_below_minimum")
    if float(settings.get("fps") or 0.0) < REVIEW_QUALITY_MIN_FPS:
        blockers.append("review_quality_profile_fps_below_minimum")
    if int(settings.get("num_frames") or 0) < REVIEW_QUALITY_MIN_NUM_FRAMES:
        blockers.append("review_quality_profile_num_frames_below_minimum")
    max_ungated_steps = _int_env(PERSISTENT_WAM_REVIEW_QUALITY_MAX_UNGATED_LOOP_STEPS_ENV, 3)
    if loop_step_count > max_ungated_steps:
        validation = validate_persistent_wam_short_visual_sanity_manifest(
            os.getenv(PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST_ENV),
            policy_observation_path=policy_observation_path,
        )
        if validation.get("status") != "passed_short_visual_sanity":
            blockers.append("review_quality_long_rollout_requires_passed_short_visual_sanity")
            blockers.extend(str(item) for item in validation.get("blockers") or [])
            if _truthy(os.getenv(PERSISTENT_WAM_LONG_REVIEW_ROLLOUT_ENV)):
                blockers.append(
                    "review_quality_long_rollout_env_override_requires_short_visual_sanity_manifest"
                )
    return sorted(set(blockers))


def _policy_observation_source_kind(observation: Mapping[str, Any]) -> str:
    visual = _mapping(observation.get("visual_observation"))
    provenance = _mapping(observation.get("provenance"))
    source_candidate = _mapping(observation.get("source_candidate"))
    for value in (
        observation.get("source_kind"),
        observation.get("selection_source_kind"),
        visual.get("source_kind"),
        provenance.get("source_kind"),
        source_candidate.get("source_kind"),
    ):
        text = _string(value)
        if text:
            return text
    claim_boundary = {
        **_mapping(visual.get("claim_boundary")),
        **_mapping(observation.get("claim_boundary")),
    }
    if (
        visual.get("synthetic_fallback") is True
        or provenance.get("synthetic_fallback") is True
        or claim_boundary.get("selected_synthetic_fallback") is True
    ):
        return "synthetic_fallback"
    return ""


def _policy_observation_truth_value(
    observation: Mapping[str, Any],
    *,
    key: str,
    default: bool = False,
) -> bool:
    visual = _mapping(observation.get("visual_observation"))
    provenance = _mapping(observation.get("provenance"))
    claim_boundary = {
        **_mapping(visual.get("claim_boundary")),
        **_mapping(observation.get("claim_boundary")),
    }
    for container in (claim_boundary, visual, provenance, observation):
        if key in container:
            return bool(container.get(key))
    return default


def _synthetic_fallback_wam_launch_gate(
    *,
    observation: Mapping[str, Any],
    original_source_kind: str,
    visual_profile: str,
    use_live_wam: bool,
) -> dict[str, Any]:
    effective_source_kind = _policy_observation_source_kind(observation)
    source_kinds = [
        source_kind
        for source_kind in (original_source_kind, effective_source_kind)
        if source_kind
    ]
    visual = _mapping(observation.get("visual_observation"))
    provenance = _mapping(observation.get("provenance"))
    claim_boundary = {
        **_mapping(visual.get("claim_boundary")),
        **_mapping(observation.get("claim_boundary")),
    }
    synthetic_fallback_used = bool(
        any(source_kind in SYNTHETIC_FALLBACK_WAM_SOURCE_KINDS for source_kind in source_kinds)
        or visual.get("synthetic_fallback") is True
        or provenance.get("synthetic_fallback") is True
        or claim_boundary.get("selected_synthetic_fallback") is True
    )
    launch_path = bool(use_live_wam or visual_profile == "review_quality")
    env_enabled = _truthy(os.getenv(SYNTHETIC_FALLBACK_WAM_LAUNCH_EXPERIMENT_ENV))
    blockers: list[str] = []
    if synthetic_fallback_used and launch_path and not env_enabled:
        blockers.append(
            "synthetic_fallback_live_or_review_wam_launch_requires_experimental_env"
        )
    return {
        "schema_version": "synthetic_fallback_wam_launch_gate.v1",
        "synthetic_fallback_initial_observation_used": synthetic_fallback_used,
        "original_source_kind": original_source_kind or None,
        "effective_source_kind": effective_source_kind or None,
        "source_kinds": sorted(set(source_kinds)),
        "launch_path_requires_gate": launch_path,
        "use_live_wam": bool(use_live_wam),
        "visual_profile": visual_profile,
        "experimental_env": SYNTHETIC_FALLBACK_WAM_LAUNCH_EXPERIMENT_ENV,
        "experimental_env_enabled": env_enabled,
        "capture_truth": False
        if synthetic_fallback_used
        else _policy_observation_truth_value(observation, key="capture_truth"),
        "geometry_truth": False
        if synthetic_fallback_used
        else _policy_observation_truth_value(observation, key="geometry_truth"),
        "collision_truth": False,
        "provider_success_separate_from_visually_useful_rollout": True,
        "visually_useful_rollout": False,
        "visually_useful_rollout_pending_review": True,
        "blockers": blockers,
    }


def _apply_synthetic_fallback_truth_labels(
    observation: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    labeled = json.loads(json.dumps(dict(observation)))
    if not gate.get("synthetic_fallback_initial_observation_used"):
        return labeled
    visual = _mapping(labeled.get("visual_observation"))
    visual.update(
        {
            "capture_truth": False,
            "geometry_truth": False,
            "collision_truth": False,
            "synthetic_fallback_wam_launch_experiment_env": gate.get("experimental_env"),
            "synthetic_fallback_wam_launch_experiment_enabled": bool(
                gate.get("experimental_env_enabled")
            ),
        }
    )
    labeled["visual_observation"] = visual
    provenance = _mapping(labeled.get("provenance"))
    provenance.update(
        {
            "capture_truth": False,
            "geometry_truth": False,
            "collision_truth": False,
            "synthetic_fallback_wam_launch_experiment_enabled": bool(
                gate.get("experimental_env_enabled")
            ),
        }
    )
    labeled["provenance"] = provenance
    claim_boundary = _mapping(labeled.get("claim_boundary"))
    claim_boundary.update(
        {
            "synthetic_fallback_initial_observation_used": True,
            "synthetic_fallback_wam_launch_experiment_env": gate.get("experimental_env"),
            "synthetic_fallback_wam_launch_experiment_enabled": bool(
                gate.get("experimental_env_enabled")
            ),
            "capture_truth": False,
            "geometry_truth": False,
            "collision_truth": False,
            "provider_success_separate_from_visually_useful_rollout": True,
            "visually_useful_rollout": False,
            "visual_rollout_quality_must_be_judged_separately": True,
        }
    )
    labeled["claim_boundary"] = claim_boundary
    return labeled


def _machine_ids_from_env(env_names: Sequence[str]) -> list[int]:
    values: list[int] = []
    for env_name in env_names:
        for chunk in _string(os.getenv(env_name)).replace(",", " ").split():
            try:
                machine_id = int(chunk)
            except ValueError:
                continue
            if machine_id > 0 and machine_id not in values:
                values.append(machine_id)
    return values


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _load_policy_observation(path: str | Path) -> dict[str, Any]:
    payload = _read_json(Path(path).expanduser())
    observation = (
        payload.get("observation") if isinstance(payload.get("observation"), Mapping) else payload
    )
    if not isinstance(observation, Mapping):
        raise ValueError("policy_observation_json_must_contain_object")
    return dict(observation)


def _camera_frame_path(observation: Mapping[str, Any]) -> Path | None:
    visual = _mapping(observation.get("visual_observation"))
    for candidate in (
        visual.get("camera_frame_path"),
        _mapping(observation.get("sensor_surrogates")).get("camera_frame_path"),
        observation.get("camera_frame_path"),
    ):
        text = _string(candidate)
        if not text:
            continue
        path = Path(text).expanduser()
        if path.is_file():
            return path.resolve()
    return None


def _resolve_local_json_path(value: Any, *, base_dir: Path) -> Path | None:
    text = _string(value)
    if not text or "://" in text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _load_local_json_ref(value: Any, *, base_dir: Path) -> tuple[Any | None, Path | None]:
    path = _resolve_local_json_path(value, base_dir=base_dir)
    if path is None or not path.is_file():
        return None, path
    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except (OSError, json.JSONDecodeError):
        return None, path


def _policy_observation_semantic_visual_evidence(
    observation: Mapping[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    visual = _mapping(observation.get("visual_observation"))
    eval_ready: Any = observation.get("eval_ready_task_grounding") or visual.get(
        "eval_ready_task_grounding"
    )
    eval_ready_path: Path | None = None
    for key in (
        "eval_ready_task_grounding_path",
        "eval_ready_task_grounding_json_path",
        "task_grounding_path",
    ):
        if isinstance(eval_ready, Mapping):
            break
        loaded, path = _load_local_json_ref(
            observation.get(key) or visual.get(key), base_dir=base_dir
        )
        if loaded is not None:
            eval_ready = loaded
            eval_ready_path = path
            break
    if not isinstance(eval_ready, Mapping):
        for candidate in (
            base_dir / "eval_ready_task_grounding.json",
            base_dir / "simulation_automation" / "eval_ready_task_grounding.json",
            base_dir.parent / "simulation_automation" / "eval_ready_task_grounding.json",
        ):
            loaded, path = _load_local_json_ref(str(candidate), base_dir=base_dir)
            if isinstance(loaded, Mapping):
                eval_ready = loaded
                eval_ready_path = path
                break
    object_index: Any = observation.get("object_index") or visual.get("object_index")
    object_index_path: Path | None = None
    object_index_path_values = [
        observation.get("object_index_path"),
        visual.get("object_index_path"),
        _mapping(eval_ready).get("object_index", {}).get("path")
        if isinstance(_mapping(eval_ready).get("object_index"), Mapping)
        else None,
    ]
    for value in object_index_path_values:
        if isinstance(object_index, (Mapping, list)):
            break
        loaded, path = _load_local_json_ref(value, base_dir=base_dir)
        if loaded is not None:
            object_index = loaded
            object_index_path = path
            break
    if not isinstance(object_index, (Mapping, list)):
        for candidate in (
            base_dir / "object_index.json",
            base_dir / "raw" / "object_index.json",
            base_dir.parent / "raw" / "object_index.json",
        ):
            loaded, path = _load_local_json_ref(str(candidate), base_dir=base_dir)
            if loaded is not None:
                object_index = loaded
                object_index_path = path
                break
    artifact_base_dir = object_index_path.parent if object_index_path else base_dir
    return {
        "object_index": object_index if isinstance(object_index, (Mapping, list)) else None,
        "object_index_path": str(object_index_path) if object_index_path else None,
        "eval_ready_task_grounding": dict(eval_ready) if isinstance(eval_ready, Mapping) else None,
        "eval_ready_task_grounding_path": str(eval_ready_path) if eval_ready_path else None,
        "semantic_artifact_base_dir": str(artifact_base_dir),
    }


def _write_executable(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


PERSISTENT_SESSION_RUNNER = r"""#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib import request as urllib_request
from urllib import error as urllib_error

OUTPUT_SCHEMA_VERSION = "unitree_groot_n17_sonic_wam_persistent_session_output.v1"
POLICY_ID = "unitree_groot_n17_sonic_policy"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _command_available(command: str | None) -> bool:
    text = _string(command)
    if not text:
        return False
    try:
        parts = shlex.split(text)
    except ValueError:
        return False
    if not parts:
        return False
    executable = parts[0]
    return bool(shutil.which(executable) or Path(executable).expanduser().is_file())


def _command_uses_policy_server_client(command: str | None) -> bool:
    return "unitree_groot_n17_sonic_policy_server_command" in _string(command)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _phase(name: str, **fields: Any) -> None:
    payload = {
        "phase": name,
        "observed_at_epoch": round(time.time(), 3),
        "raw_secret_values_recorded": False,
        **fields,
    }
    print(
        "BLUEPRINT_PERSISTENT_SESSION_PHASE:"
        + json.dumps(payload, sort_keys=True),
        flush=True,
    )
    _upload_phase_heartbeat(payload)


def _upload_phase_heartbeat(phase_payload: Mapping[str, Any]) -> None:
    upload_enabled = _truthy(
        os.environ.get("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_UPLOAD_PHASE_HEARTBEATS")
        or "true"
    )
    put_url = _string(os.environ.get("OUTPUT_PUT_URL"))
    work_dir = _string(os.environ.get("WORK_DIR"))
    output_dir_text = _string(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR"))
    if not upload_enabled or not put_url or not work_dir or not output_dir_text:
        return
    try:
        output_dir = Path(output_dir_text).expanduser().resolve()
        output_path = Path(
            _string(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT"))
            or output_dir / "unitree_groot_n17_sonic_policy_provider_output.json"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        heartbeat = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "running",
            "policy_id": POLICY_ID,
            "persistent_provider_session_used": True,
            "runtime_phase": phase_payload.get("phase"),
            "runtime_phase_details": dict(phase_payload),
            "runpod_unitree_groot_sonic_remote_heartbeat": True,
            "blockers": [],
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        output_path.write_text(
            json.dumps(heartbeat, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        zip_path = (
            Path(work_dir).expanduser().resolve()
            / "unitree_groot_n17_sonic_provider_phase_heartbeat.zip"
        )
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(output_path, output_path.relative_to(output_dir).as_posix())
        if not zip_path.stat().st_size or not zipfile.is_zipfile(zip_path):
            raise RuntimeError("invalid_or_empty_phase_heartbeat_zip")
        request = urllib_request.Request(
            put_url,
            data=zip_path.read_bytes(),
            method="PUT",
            headers={"Content-Type": "application/zip"},
        )
        timeout_seconds = int(
            os.environ.get(
                "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_PHASE_HEARTBEAT_TIMEOUT_SECONDS"
            )
            or "20"
        )
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            response.read()
        print(
            "BLUEPRINT_PERSISTENT_SESSION_PHASE_HEARTBEAT_UPLOAD_OK:"
            + json.dumps(
                {
                    "phase": phase_payload.get("phase"),
                    "raw_secret_values_recorded": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as exc:
        print(
            "BLUEPRINT_PERSISTENT_SESSION_PHASE_HEARTBEAT_UPLOAD_BLOCKED:"
            + json.dumps(
                {
                    "phase": phase_payload.get("phase"),
                    "error_type": type(exc).__name__,
                    "raw_secret_values_recorded": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _read_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    value = json.loads(raw.decode("utf-8") or "{}")
    return dict(value) if isinstance(value, Mapping) else {}


def _send(handler: BaseHTTPRequestHandler, status: int, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(payload), sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _http_post_json(url: str, payload: Mapping[str, Any], timeout_seconds: float) -> dict[str, Any]:
    data = json.dumps(dict(payload)).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
            parsed = json.loads(response.read().decode("utf-8") or "{}")
            status_code = int(getattr(response, "status", 200) or 200)
    except urllib_error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw or "{}")
        except Exception:
            parsed = {
                "status": "blocked",
                "blockers": [f"persistent_worker_http_error:{exc.code}"],
                "error_message_redacted": raw[-1000:],
            }
        status_code = int(exc.code)
        if isinstance(parsed, Mapping):
            parsed = dict(parsed)
            parsed.setdefault("status", "blocked")
            parsed.setdefault("blockers", [f"persistent_worker_http_error:{exc.code}"])
            parsed["http_status_code"] = status_code
            parsed["http_error"] = True
            return parsed
        parsed = {"status": "blocked", "blockers": [f"persistent_worker_http_error:{exc.code}"], "http_status_code": status_code}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _http_post_json_with_retries(
    url: str,
    payload: Mapping[str, Any],
    *,
    timeout_seconds: float,
    attempts: int = 3,
    sleep_seconds: float = 5.0,
) -> dict[str, Any]:
    response: dict[str, Any] = {}
    for attempt in range(1, max(1, attempts) + 1):
        response = _http_post_json(url, payload, timeout_seconds=timeout_seconds)
        response["persistent_http_attempt_index"] = attempt
        if not response.get("http_error"):
            return response
        if attempt < attempts:
            time.sleep(sleep_seconds)
    return response


def _extract_action(response: Mapping[str, Any]) -> dict[str, Any]:
    action = response.get("action") or response.get("policy_action") or response.get("normalized_action")
    return dict(action) if isinstance(action, Mapping) else {}


def _copy_or_extract_wam_frame(payload: Mapping[str, Any], target_frame: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    for key in ("generated_next_observation_frame_path", "camera_frame_path", "frame_path", "image_path"):
        value = _string(payload.get(key))
        if value:
            candidates.append(Path(value).expanduser())
    visual = _mapping(payload.get("visual_observation"))
    if _string(visual.get("camera_frame_path")):
        candidates.append(Path(_string(visual.get("camera_frame_path"))).expanduser())
    for rollout in payload.get("rollouts") or []:
        if isinstance(rollout, Mapping):
            for key in ("generated_video_path", "video_path", "output_video_path"):
                value = _string(rollout.get(key))
                if value:
                    candidates.append(Path(value).expanduser())
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            target_frame.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target_frame)
            return {
                "status": "completed",
                "source_kind": "image",
                "source_path": str(candidate),
                "materialized_frame_path": str(target_frame),
            }
        if candidate.is_file() and candidate.suffix.lower() in {".mp4", ".mov", ".m4v"}:
            try:
                import cv2
            except Exception as exc:
                return {
                    "status": "blocked",
                    "blockers": [f"opencv_import_failed:{type(exc).__name__}"],
                    "source_path": str(candidate),
                }
            cap = cv2.VideoCapture(str(candidate))
            try:
                ok, frame = cap.read()
            finally:
                cap.release()
            if ok and frame is not None:
                target_frame.parent.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(target_frame), frame):
                    return {
                        "status": "completed",
                        "source_kind": "video_first_frame",
                        "source_path": str(candidate),
                        "materialized_frame_path": str(target_frame),
                    }
    return {
        "status": "blocked",
        "blockers": ["wam_output_missing_materializable_frame_or_video"],
    }


def _structural_wam_frame(source_frame: Path, target_frame: Path, step_index: int) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        return {"status": "blocked", "blockers": [f"pillow_import_failed:{type(exc).__name__}"]}
    try:
        image = Image.open(source_frame).convert("RGB")
    except Exception:
        image = Image.new("RGB", (640, 480), (32, 35, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 42), fill=(14, 22, 34))
    try:
        font = ImageFont.load_default()
        draw.text((12, 14), f"structural WAM fallback step {step_index}", fill=(240, 246, 250), font=font)
    except Exception:
        pass
    target_frame.parent.mkdir(parents=True, exist_ok=True)
    image.save(target_frame, quality=92)
    return {
        "status": "completed",
        "source_kind": "structural_fallback_image",
        "source_path": str(source_frame),
        "materialized_frame_path": str(target_frame),
    }


class PolicyWorker(BaseHTTPRequestHandler):
    policy_command = ""
    command_source = ""
    command_available = False
    command_uses_policy_server_client = False
    policy_server_url = ""
    timeout_seconds = 240.0
    output_dir = Path(".")

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/") not in {"/readyz", "/healthz"}:
            _send(self, 404, {"status": "not_found"})
            return
        _send(
            self,
            200,
            {
                "schema_version": "persistent_policy_worker_ready.v1",
                "status": "ready" if self.policy_command else "blocked",
                "ready_for_inference": bool(self.policy_command),
                "policy_id": POLICY_ID,
                "persistent_policy_worker_command_source": self.command_source,
                "persistent_policy_worker_command_available": bool(self.command_available),
                "persistent_policy_worker_command_uses_policy_server_client": bool(
                    self.command_uses_policy_server_client
                ),
                "raw_secret_values_recorded": False,
            },
        )

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/infer":
            _send(self, 404, {"status": "not_found"})
            return
        started = time.monotonic()
        try:
            payload = _read_body(self)
            observation = _mapping(payload.get("observation")) or _mapping(payload)
            from blueprint_pipeline.unitree_groot_n17_sonic_policy_command_adapter import run_unitree_groot_n17_sonic_policy

            uses_policy_server_client = bool(self.command_uses_policy_server_client)
            response, exit_code = run_unitree_groot_n17_sonic_policy(
                payload={"observation": observation},
                command=self.policy_command,
                n17_checkpoint=os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT"),
                sonic_checkpoint=(
                    None
                    if uses_policy_server_client
                    else os.environ.get("BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT")
                ),
                groot_root=(
                    None
                    if uses_policy_server_client
                    else os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT")
                ),
                wbc_root=(
                    None
                    if uses_policy_server_client
                    else os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT")
                ),
                policy_server_url=self.policy_server_url,
                sim2sim_command=os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_COMMAND"),
                timeout_seconds=self.timeout_seconds,
            )
            _send(
                self,
                200 if exit_code == 0 else 502,
                {
                    **dict(response),
                    "persistent_worker_infer": True,
                    "persistent_worker_duration_seconds": round(time.monotonic() - started, 6),
                    "provider_instance_reused_for_policy_infer": True,
                    "persistent_policy_worker_command_configured": bool(self.policy_command),
                    "persistent_policy_worker_command_source": self.command_source,
                    "persistent_policy_worker_command_available": bool(self.command_available),
                    "persistent_policy_worker_command_uses_policy_server_client": uses_policy_server_client,
                    "raw_secret_values_recorded": False,
                },
            )
        except Exception as exc:
            _send(
                self,
                500,
                {
                    "status": "blocked",
                    "policy_id": POLICY_ID,
                    "blockers": [f"persistent_policy_worker_infer_failed:{type(exc).__name__}"],
                    "error": str(exc)[:800],
                    "persistent_policy_worker_command_configured": bool(self.policy_command),
                    "persistent_policy_worker_command_source": self.command_source,
                    "persistent_policy_worker_command_available": bool(self.command_available),
                    "persistent_policy_worker_command_uses_policy_server_client": bool(
                        self.command_uses_policy_server_client
                    ),
                    "raw_secret_values_recorded": False,
                },
            )


class WamWorker(BaseHTTPRequestHandler):
    output_dir = Path(".")
    use_live_wam = True
    allow_structural_fallback = False
    timeout_seconds = 3600.0

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/") not in {"/readyz", "/healthz"}:
            _send(self, 404, {"status": "not_found"})
            return
        _send(
            self,
            200,
            {
                "schema_version": "persistent_wam_worker_ready.v1",
                "status": "ready",
                "ready_for_inference": True,
                "use_live_wam": self.use_live_wam,
                "allow_structural_fallback": self.allow_structural_fallback,
                "raw_secret_values_recorded": False,
            },
        )

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/infer":
            _send(self, 404, {"status": "not_found"})
            return
        started = time.monotonic()
        payload = _read_body(self)
        step_index = int(payload.get("step_index") or 0)
        source_frame = Path(_string(payload.get("source_frame"))).expanduser()
        step_dir = self.output_dir / "wam_worker_steps" / f"step_{step_index:04d}"
        target_frame = self.output_dir / "generated_next_observations" / f"wam_generated_next_observation_step_{step_index:04d}.jpg"
        step_dir.mkdir(parents=True, exist_ok=True)
        current_policy_observation = _mapping(payload.get("current_policy_observation"))
        auxiliary_observation = _mapping(current_policy_observation.get("wam_auxiliary_observation"))
        auxiliary_manifest_path = _string(
            current_policy_observation.get("wam_auxiliary_observation_manifest_path")
        ) or _string(
            _mapping(current_policy_observation.get("visual_observation")).get(
                "wam_auxiliary_observation_manifest_path"
            )
        )
        if auxiliary_manifest_path and not auxiliary_observation:
            candidate_auxiliary_path = Path(auxiliary_manifest_path).expanduser()
            if candidate_auxiliary_path.is_file():
                try:
                    auxiliary_observation = json.loads(
                        candidate_auxiliary_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    auxiliary_observation = {}
        step_input = {
            "schema_version": "wam_generation_step_input.v1",
            "generated_at": payload.get("generated_at"),
            "step_index": step_index,
            "wam_evaluator_backend": "persistent_oscar_wam_worker",
            "source_policy_observation_frame_path": str(source_frame),
            "source_policy_action": _mapping(payload.get("source_policy_action")),
            "current_policy_observation": current_policy_observation,
            "wam_auxiliary_observation_manifest_path": auxiliary_manifest_path or None,
            "auxiliary_observation": auxiliary_observation,
            "requested_output": {
                "next_observation_frame_path": str(target_frame),
                "action_conditioned_generation_required": True,
            },
            "claim_boundary": {
                "wam_generation_is_not_robot_policy": True,
                "physical_robot_sensor_proof": False,
            },
        }
        step_input_path = step_dir / "wam_generation_step_input.json"
        _write_json(step_input_path, step_input)
        live_payload: dict[str, Any] = {}
        live_blockers: list[str] = []
        materialization: dict[str, Any] = {}
        live_ran = False
        if self.use_live_wam:
            try:
                from blueprint_pipeline.oscar_wam_provider_bundle import build_oscar_wam_provider_bundle

                bundle = build_oscar_wam_provider_bundle(
                    job_dir=step_dir / "oscar_wam_worker_bundle",
                    wam_rollout_input_manifest=step_input_path,
                    timeout_seconds=int(self.timeout_seconds),
                    num_steps=int(os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_STEPS", "12")),
                    num_frames=int(os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_FRAMES", "24")),
                    height=int(os.environ.get("BLUEPRINT_OSCAR_WAM_HEIGHT", "480")),
                    width=int(os.environ.get("BLUEPRINT_OSCAR_WAM_WIDTH", "640")),
                    fps=float(os.environ.get("BLUEPRINT_OSCAR_WAM_FPS", "15")),
                )
                if bundle.get("status") != "completed":
                    live_blockers.extend(bundle.get("blockers") or ["oscar_wam_provider_bundle_blocked"])
                else:
                    bundle_root = Path(str(bundle["job_dir"])) / "oscar_wam_provider_bundle"
                    output_dir = step_dir / "oscar_runtime_output"
                    env = os.environ.copy()
                    env["BLUEPRINT_WAM_PROVIDER_BUNDLE_DIR"] = str(bundle_root)
                    env["BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR"] = str(output_dir)
                    env["BLUEPRINT_WAM_PROVIDER_WORK_DIR"] = str(self.output_dir / "persistent_wam_shared_work")
                    env["BLUEPRINT_WAM_ROLLOUT_INPUT"] = str(bundle_root / "provider_runtime" / "wam_rollout_input_manifest.json")
                    runtime_log_path = step_dir / "persistent_wam_worker_runtime_stdout_stderr.log"
                    timeout_expired = False
                    completed_returncode: int | None = None
                    process_group_terminated = False
                    process_group_killed = False
                    process_group_id: int | None = None
                    with runtime_log_path.open("a", encoding="utf-8") as runtime_log:
                        runtime_log.write(
                            json.dumps(
                                {
                                    "event": "persistent_wam_worker_oscar_runtime_started",
                                    "timeout_seconds": self.timeout_seconds,
                                    "step_index": step_index,
                                    "raw_secret_values_recorded": False,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        runtime_log.flush()
                        proc: subprocess.Popen[str] | None = None
                        try:
                            proc = subprocess.Popen(
                                [
                                    "bash",
                                    str(bundle_root / "provider_runtime" / "run_wam_provider_runtime.sh"),
                                ],
                                cwd=str(bundle_root),
                                env=env,
                                text=True,
                                stdout=runtime_log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                            process_group_id = os.getpgid(proc.pid)
                            wait_started = time.monotonic()
                            timeout_deadline = wait_started + float(self.timeout_seconds)
                            last_wait_log = wait_started
                            while True:
                                completed_returncode = proc.poll()
                                if completed_returncode is not None:
                                    break
                                now = time.monotonic()
                                if now >= timeout_deadline:
                                    raise subprocess.TimeoutExpired(proc.args, self.timeout_seconds)
                                if now - last_wait_log >= 60:
                                    runtime_log.write(
                                        json.dumps(
                                            {
                                                "event": "persistent_wam_worker_oscar_runtime_waiting",
                                                "elapsed_seconds": round(now - wait_started, 3),
                                                "timeout_seconds": self.timeout_seconds,
                                                "process_group_id": process_group_id,
                                                "raw_secret_values_recorded": False,
                                            },
                                            sort_keys=True,
                                        )
                                        + "\n"
                                    )
                                    runtime_log.flush()
                                    last_wait_log = now
                                time.sleep(min(5.0, max(0.1, timeout_deadline - now)))
                        except subprocess.TimeoutExpired:
                            timeout_expired = True
                            live_blockers.append("persistent_wam_worker_oscar_runtime_timeout")
                            if proc is not None:
                                try:
                                    os.killpg(process_group_id or os.getpgid(proc.pid), signal.SIGTERM)
                                    process_group_terminated = True
                                    completed_returncode = proc.wait(timeout=20)
                                except ProcessLookupError:
                                    completed_returncode = proc.poll()
                                except subprocess.TimeoutExpired:
                                    os.killpg(process_group_id or os.getpgid(proc.pid), signal.SIGKILL)
                                    process_group_killed = True
                                    completed_returncode = proc.wait(timeout=20)
                            runtime_log.write(
                                json.dumps(
                                    {
                                        "event": "persistent_wam_worker_oscar_runtime_timeout",
                                        "timeout_seconds": self.timeout_seconds,
                                        "process_group_terminated": process_group_terminated,
                                        "process_group_killed": process_group_killed,
                                        "process_group_id": process_group_id,
                                        "returncode": completed_returncode,
                                        "raw_secret_values_recorded": False,
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            runtime_log.flush()
                    live_ran = True
                    provider_output_path = output_dir / "wam_provider_output.json"
                    if provider_output_path.is_file():
                        live_payload = json.loads(provider_output_path.read_text(encoding="utf-8"))
                        live_payload = dict(live_payload) if isinstance(live_payload, Mapping) else {}
                    if completed_returncode not in (0, None):
                        live_blockers.append("persistent_wam_worker_oscar_runtime_nonzero_exit")
                    if not live_payload:
                        live_blockers.append("persistent_wam_worker_missing_oscar_provider_output")
                    materialization = _copy_or_extract_wam_frame(live_payload, target_frame)
                    if materialization.get("status") != "completed":
                        live_blockers.extend(materialization.get("blockers") or ["persistent_wam_frame_materialization_failed"])
                    _write_json(
                        step_dir / "persistent_wam_worker_command_execution.json",
                        {
                            "schema_version": "persistent_wam_worker_command_execution.v1",
                            "status": "completed"
                            if completed_returncode == 0 and not timeout_expired
                            else "blocked",
                            "returncode": completed_returncode,
                            "timed_out": timeout_expired,
                            "timeout_seconds": self.timeout_seconds,
                            "process_group_id": process_group_id,
                            "process_group_terminated": process_group_terminated,
                            "process_group_killed": process_group_killed,
                            "runtime_stdout_stderr_log_path": str(runtime_log_path),
                            "runtime_stdout_stderr_log_size_bytes": runtime_log_path.stat().st_size
                            if runtime_log_path.is_file()
                            else 0,
                            "stdout_stderr_streamed_to_log": True,
                            "bundle_manifest": bundle,
                            "raw_secret_values_recorded": False,
                        },
                    )
            except Exception as exc:
                live_blockers.append(f"persistent_wam_worker_live_infer_failed:{type(exc).__name__}")
                _write_json(
                    step_dir / "persistent_wam_worker_exception.json",
                    {
                        "status": "blocked",
                        "error_type": type(exc).__name__,
                        "traceback_tail": traceback.format_exc()[-4000:],
                        "raw_secret_values_recorded": False,
                    },
                )
        fallback_used = False
        structural_fallback_requested = (
            self.allow_structural_fallback and (bool(live_blockers) or not self.use_live_wam)
        )
        if structural_fallback_requested:
            fallback_used = True
            materialization = _structural_wam_frame(source_frame, target_frame, step_index)
            if materialization.get("status") != "completed":
                live_blockers.extend(
                    materialization.get("blockers")
                    or ["persistent_structural_wam_fallback_materialization_failed"]
                )
        completed = target_frame.is_file() and (
            (not live_blockers and materialization.get("status") == "completed")
            or (fallback_used and materialization.get("status") == "completed")
        )
        if not completed and not live_blockers and not self.use_live_wam:
            live_blockers.append("persistent_wam_live_disabled_without_structural_fallback")
        response = {
            "schema_version": "persistent_wam_worker_infer_response.v1",
            "status": "completed" if completed else "blocked",
            "step_index": step_index,
            "wam_evaluator_backend": "persistent_oscar_wam_worker" if not fallback_used else "persistent_structural_wam_fallback",
            "provider_instance_reused_for_wam_infer": True,
            "persistent_wam_worker_infer": True,
            "persistent_worker_duration_seconds": round(time.monotonic() - started, 6),
            "live_wam_generation_command_ran": bool(live_ran and not fallback_used),
            "learned_oscar_or_cosmos_model_ran": bool(
                not fallback_used and live_payload.get("status") == "completed"
            ),
            "wam_model_checkpoint_used": bool(
                not fallback_used and _mapping(live_payload.get("model_provenance")).get("checkpoint_path")
            ),
            "action_conditioned_generation_ran": bool(completed),
            "generated_next_observation_frame_path": str(target_frame) if target_frame.is_file() else None,
            "materialization": materialization,
            "live_wam_payload_redacted": live_payload,
            "structural_fallback_used": fallback_used,
            "blockers": [] if completed else sorted(set(live_blockers)),
            "claim_boundary": {
                "wam_is_next_observation_generator_not_robot_policy": True,
                "generated_observation_is_not_raw_capture": True,
                "capture_truth": False,
                "geometry_truth": False,
                "collision_truth": False,
                "structural_fallback_is_not_live_wam_model": fallback_used,
                "provider_success_separate_from_visually_useful_rollout": True,
                "visually_useful_rollout": False,
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
                "safety_validation_proven": False,
            },
            "raw_secret_values_recorded": False,
        }
        _write_json(step_dir / "persistent_wam_worker_infer_response.json", response)
        _send(self, 200 if completed else 502, response)


def _start_server(port: int, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _side_by_side_html(path: Path, rows: list[Mapping[str, Any]]) -> None:
    cards = []
    for row in rows:
        cards.append(
            "<section><h2>Step {}</h2><pre>{}</pre></section>".format(
                row.get("transition_index"),
                json.dumps(dict(row), indent=2, sort_keys=True),
            )
        )
    path.write_text(
        "\n".join(
            [
                "<!doctype html><html><head><meta charset='utf-8'><title>Persistent Policy WAM Trace</title>",
                "<style>body{font-family:sans-serif;margin:24px;background:#f7f7f7}section{background:white;border:1px solid #ddd;border-radius:8px;padding:16px;margin:0 0 16px}pre{white-space:pre-wrap;font-size:12px}</style>",
                "</head><body><h1>Persistent Policy WAM Trace</h1>",
                *cards,
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    runtime_dir = Path(__file__).resolve().parent
    output_dir = Path(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR", runtime_dir / "runtime_output")).resolve()
    output_path = Path(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT", output_dir / "unitree_groot_n17_sonic_policy_provider_output.json")).resolve()
    session_input_path = Path(os.environ.get("BLUEPRINT_PERSISTENT_SESSION_INPUT", runtime_dir / "persistent_session_input.json")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    policy_server_process = None
    policy_server = None
    wam_server = None
    try:
        session_input = json.loads(session_input_path.read_text(encoding="utf-8"))
        observation = _mapping(session_input.get("initial_observation"))
        loop_step_count = max(1, int(session_input.get("loop_step_count") or 12))
        policy_port = int(session_input.get("policy_worker_port") or 8765)
        wam_port = int(session_input.get("wam_worker_port") or 8766)
        use_live_wam = bool(session_input.get("use_live_wam") is not False)
        allow_structural_fallback = bool(session_input.get("allow_structural_wam_fallback"))
        timeout_seconds = float(session_input.get("timeout_seconds") or 3600.0)
        initial_frame = runtime_dir / "initial_policy_frame.png"
        clean_frame_reanchoring = _mapping(session_input.get("clean_frame_reanchoring"))
        clean_frame_reanchoring_enabled = bool(clean_frame_reanchoring.get("enabled"))
        try:
            clean_frame_reanchor_interval = int(
                clean_frame_reanchoring.get("interval_steps") or 0
            )
        except (TypeError, ValueError):
            clean_frame_reanchor_interval = 0
        visual = _mapping(observation.get("visual_observation"))
        visual["camera_frame_path"] = str(initial_frame)
        runtime_auxiliary_observation_manifest = runtime_dir / "wam_auxiliary_observation" / "wam_auxiliary_observation_manifest.json"
        if runtime_auxiliary_observation_manifest.is_file():
            try:
                runtime_auxiliary_payload = json.loads(
                    runtime_auxiliary_observation_manifest.read_text(encoding="utf-8")
                )
                if isinstance(runtime_auxiliary_payload, Mapping):
                    runtime_auxiliary_payload = dict(runtime_auxiliary_payload)
                    runtime_auxiliary_payload["manifest_path"] = str(
                        runtime_auxiliary_observation_manifest
                    )
                    runtime_auxiliary_payload["source_image_path"] = str(initial_frame)
                    runtime_auxiliary_payload["source_image_path_exists"] = initial_frame.is_file()
                    runtime_auxiliary_payload["runtime_paths_rewritten_for_provider_runtime"] = True
                    _write_json(runtime_auxiliary_observation_manifest, runtime_auxiliary_payload)
            except Exception:
                pass
            visual["wam_auxiliary_observation_manifest_path"] = str(runtime_auxiliary_observation_manifest)
            observation["wam_auxiliary_observation_manifest_path"] = str(runtime_auxiliary_observation_manifest)
            auxiliary_summary = _mapping(observation.get("wam_auxiliary_observation"))
            if auxiliary_summary:
                auxiliary_summary["manifest_path"] = str(runtime_auxiliary_observation_manifest)
                auxiliary_summary["runtime_path_rewritten_for_provider_runtime"] = True
                observation["wam_auxiliary_observation"] = auxiliary_summary
        observation["visual_observation"] = visual
        observation["camera_frame_path"] = str(initial_frame)

        _phase("bootstrap_policy_server_started")
        from blueprint_pipeline import unitree_groot_n17_sonic_provider_smoke as provider_smoke

        bootstrap_namespace: dict[str, Any] = {
            "__name__": "blueprint_persistent_session_bootstrap",
            "__file__": str(runtime_dir / "unitree_groot_n17_sonic_provider_runner.py"),
            "_blueprint_outer_phase_callback": _phase,
        }
        exec(provider_smoke.PROVIDER_RUNNER, bootstrap_namespace)
        _bootstrap_gr00t_policy_server = bootstrap_namespace["_bootstrap_gr00t_policy_server"]

        os.environ.setdefault("BLUEPRINT_UNITREE_GROOT_N17_SONIC_AUTO_START_POLICY_SERVER", "true")
        policy_server_url = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL", "tcp://127.0.0.1:5550")
        policy_server_bootstrap, policy_server_process = _bootstrap_gr00t_policy_server(
            output_dir=output_dir,
            policy_server_url=policy_server_url,
            model_path=os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT") or "LucaFrat/groot-bs16",
        )
        _write_json(output_dir / "groot_policy_server_bootstrap.json", policy_server_bootstrap)
        _phase("bootstrap_policy_server_completed", status=policy_server_bootstrap.get("status"))
        if policy_server_bootstrap.get("status") != "completed":
            raise RuntimeError("persistent_session_policy_server_bootstrap_blocked")

        configured_command = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PERSISTENT_INNER_POLICY_COMMAND", "")
        configured_command_source = "persistent_inner_policy_command_env" if configured_command else ""
        if not configured_command:
            configured_command = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_VAST_INNER_POLICY_COMMAND", "")
            configured_command_source = "vast_inner_policy_command_env" if configured_command else configured_command_source
        if not configured_command:
            configured_command = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND", "")
            configured_command_source = "policy_command_env" if configured_command else configured_command_source
        command = configured_command
        command_source = configured_command_source or "unset"
        repo_root = _mapping(policy_server_bootstrap.get("checkout")).get("repo_root")
        venv_python = policy_server_bootstrap.get("venv_python")
        if not repo_root and venv_python:
            derived_repo_root = Path(str(venv_python)).expanduser().resolve().parent.parent.parent / "Isaac-GR00T"
            if derived_repo_root.is_dir():
                repo_root = str(derived_repo_root)
        venv_python_path = Path(str(venv_python)).expanduser() if venv_python else None
        venv_python_available = bool(venv_python_path and venv_python_path.is_file())
        if policy_server_bootstrap.get("status") == "completed" and venv_python_available:
            command = f"{shlex.quote(str(venv_python))} -m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
            command_source = "bootstrap_venv_policy_server_client_for_persistent_session"
            if repo_root:
                os.environ["PYTHONPATH"] = str(repo_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
        if not command:
            command = "python3 -m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
            command_source = "python3_policy_server_client_fallback"
        command_available = _command_available(command)
        command_uses_policy_server_client = _command_uses_policy_server_client(command)
        _phase(
            "policy_worker_command_selected",
            command_source=command_source,
            command_available=command_available,
            command_uses_policy_server_client=command_uses_policy_server_client,
            repo_root_configured=bool(repo_root),
            venv_python_available=venv_python_available,
            configured_command_source=configured_command_source or None,
        )

        PolicyWorker.policy_command = command
        PolicyWorker.command_source = command_source
        PolicyWorker.command_available = command_available
        PolicyWorker.command_uses_policy_server_client = command_uses_policy_server_client
        PolicyWorker.policy_server_url = policy_server_url
        PolicyWorker.timeout_seconds = timeout_seconds
        PolicyWorker.output_dir = output_dir
        WamWorker.output_dir = output_dir
        WamWorker.use_live_wam = use_live_wam
        WamWorker.allow_structural_fallback = allow_structural_fallback
        WamWorker.timeout_seconds = float(
            os.environ.get("BLUEPRINT_PERSISTENT_SESSION_WAM_STEP_TIMEOUT_SECONDS")
            or timeout_seconds
        )
        policy_server = _start_server(policy_port, PolicyWorker)
        wam_server = _start_server(wam_port, WamWorker)
        _phase("workers_started", policy_port=policy_port, wam_port=wam_port)

        policy_calls: list[dict[str, Any]] = []
        wam_calls: list[dict[str, Any]] = []
        side_rows: list[dict[str, Any]] = []
        current_observation = observation
        current_frame = initial_frame
        current_action: dict[str, Any] = {}
        clean_frame_reanchor_events: list[dict[str, Any]] = []
        blockers: list[str] = []
        for step_index in range(loop_step_count):
            _phase("policy_infer_started", step_index=step_index)
            policy_response = _http_post_json_with_retries(
                f"http://127.0.0.1:{policy_port}/infer",
                {"observation": current_observation},
                timeout_seconds=timeout_seconds,
                attempts=3,
                sleep_seconds=5.0,
            )
            action = _extract_action(policy_response)
            policy_row = {
                "step_index": step_index,
                "status": "completed" if policy_response.get("status") == "completed" and action else "blocked",
                "policy_id": policy_response.get("policy_id"),
                "policy_observation_frame_path": str(current_frame),
                "action": action,
                "unitree_groot_n17_sonic_policy_action_command_ran": bool(
                    policy_response.get("unitree_groot_n17_sonic_policy_action_command_ran")
                ),
                "unitree_policy_action_command_ran": bool(policy_response.get("unitree_policy_action_command_ran")),
                "provider_output_replay_used": bool(policy_response.get("provider_output_replay_used")),
                "worker_response_redacted": policy_response,
            }
            policy_calls.append(policy_row)
            _write_json(output_dir / "policy_calls" / f"policy_call_{step_index:04d}.json", policy_row)
            if policy_row["status"] != "completed":
                blockers.extend(policy_response.get("blockers") or ["persistent_policy_infer_blocked"])
                break
            current_action = action
            if step_index >= loop_step_count - 1:
                break
            _phase("wam_infer_started", step_index=step_index + 1)
            wam_response = _http_post_json_with_retries(
                f"http://127.0.0.1:{wam_port}/infer",
                {
                    "generated_at": session_input.get("generated_at"),
                    "step_index": step_index + 1,
                    "source_frame": str(current_frame),
                    "current_policy_observation": current_observation,
                    "source_policy_action": current_action,
                },
                timeout_seconds=timeout_seconds,
                attempts=2,
                sleep_seconds=5.0,
            )
            wam_calls.append(wam_response)
            _write_json(output_dir / "wam_calls" / f"wam_call_{step_index + 1:04d}.json", wam_response)
            if wam_response.get("status") != "completed":
                blockers.extend(wam_response.get("blockers") or ["persistent_wam_infer_blocked"])
                break
            transition_index = step_index + 1
            next_frame = Path(
                _string(wam_response.get("generated_next_observation_frame_path"))
            ).expanduser()
            clean_frame_reanchor_applied = bool(
                clean_frame_reanchoring_enabled
                and clean_frame_reanchor_interval > 0
                and transition_index % clean_frame_reanchor_interval == 0
            )
            next_policy_frame = initial_frame if clean_frame_reanchor_applied else next_frame
            if clean_frame_reanchor_applied:
                clean_frame_reanchor_events.append(
                    {
                        "transition_index": transition_index,
                        "generated_next_observation_frame_path": str(next_frame),
                        "next_policy_observation_frame_path": str(next_policy_frame),
                        "source_frame_kind": clean_frame_reanchoring.get("source_frame_kind"),
                        "interval_steps": clean_frame_reanchor_interval,
                    }
                )
            generated_observation = {
                "schema_version": "wam_generated_next_observation.v1",
                "generated_observation_index": transition_index,
                "observation_source": "persistent_wam_worker_next_observation",
                "wam_evaluator_backend": wam_response.get("wam_evaluator_backend"),
                "wam_model_checkpoint_used": bool(wam_response.get("wam_model_checkpoint_used")),
                "action_conditioned_generation_ran": bool(wam_response.get("action_conditioned_generation_ran")),
                "live_wam_generation_command_ran": bool(wam_response.get("live_wam_generation_command_ran")),
                "learned_oscar_or_cosmos_model_ran": bool(wam_response.get("learned_oscar_or_cosmos_model_ran")),
                "generated_next_observation_frame_path": str(next_frame),
                "next_policy_observation_frame_path": str(next_policy_frame),
                "clean_frame_reanchor_applied_for_next_policy_call": clean_frame_reanchor_applied,
                "visual_observation": {
                    **_mapping(current_observation.get("visual_observation")),
                    "camera_frame_path": str(next_policy_frame),
                    "wam_generated_next_observation_frame_path": str(next_frame),
                    "wam_generated_observation": True,
                    "clean_frame_reanchor_applied": clean_frame_reanchor_applied,
                    "clean_frame_reanchor_source_path": str(initial_frame)
                    if clean_frame_reanchor_applied
                    else None,
                    "physical_robot_sensor_proof": False,
                },
            }
            side_rows.append(
                {
                    "schema_version": "persistent_policy_wam_side_by_side_trace_row.v1",
                    "transition_index": transition_index,
                    "policy_pov_frame_path": str(current_frame),
                    "policy_action_summary": {
                        "action_type": current_action.get("action_type"),
                        "action_chunk_length": len(current_action.get("action_chunk") or []),
                    },
                    "wam_generated_next_observation_frame_path": str(next_frame),
                    "next_policy_observation_frame_path": str(next_policy_frame),
                    "clean_frame_reanchor_applied_for_next_policy_call": clean_frame_reanchor_applied,
                    "wam_evaluator_backend": wam_response.get("wam_evaluator_backend"),
                    "live_wam_generation_command_ran": bool(wam_response.get("live_wam_generation_command_ran")),
                    "learned_oscar_or_cosmos_model_ran": bool(wam_response.get("learned_oscar_or_cosmos_model_ran")),
                    "next_policy_call_expected": True,
                }
            )
            current_observation = {
                **current_observation,
                **generated_observation,
                "camera_frame_path": str(next_policy_frame),
                "visual_observation": generated_observation["visual_observation"],
            }
            current_frame = next_policy_frame
        repeated_policy_calls = sum(
            1
            for row in policy_calls
            if row.get("status") == "completed"
            and row.get("unitree_policy_action_command_ran")
            and not row.get("provider_output_replay_used")
        )
        generated_count = sum(1 for row in wam_calls if row.get("status") == "completed")
        live_wam_count = sum(1 for row in wam_calls if row.get("live_wam_generation_command_ran"))
        learned_wam_count = sum(1 for row in wam_calls if row.get("learned_oscar_or_cosmos_model_ran"))
        _write_jsonl(output_dir / "robot_policy_wam_loop_trace.jsonl", policy_calls)
        _write_jsonl(output_dir / "wam_generated_next_observations.jsonl", wam_calls)
        _write_jsonl(output_dir / "robot_policy_wam_side_by_side_trace.jsonl", side_rows)
        _side_by_side_html(output_dir / "robot_policy_wam_side_by_side_trace.html", side_rows)
        required_wam_transitions = max(0, loop_step_count - 1)
        policy_only_session = required_wam_transitions == 0
        completed = bool(
            repeated_policy_calls >= loop_step_count
            and generated_count >= required_wam_transitions
            and not blockers
            and (
                policy_only_session
                or live_wam_count >= required_wam_transitions
                or (allow_structural_fallback and generated_count >= required_wam_transitions)
            )
        )
        result = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "completed" if completed else "blocked",
            "policy_id": POLICY_ID,
            "selected_candidate_id": POLICY_ID,
            "persistent_provider_session_used": True,
            "provider_instance_reused_for_policy_and_wam_loop": True,
            "policy_worker_url_redacted": f"http://127.0.0.1:{policy_port}/infer",
            "wam_worker_url_redacted": f"http://127.0.0.1:{wam_port}/infer",
            "policy_server_bootstrap": policy_server_bootstrap,
            "requested_loop_step_count": loop_step_count,
            "required_policy_call_count": loop_step_count,
            "required_wam_transition_count": required_wam_transitions,
            "policy_only_session": policy_only_session,
            "repeated_policy_calls_count": repeated_policy_calls,
            "generated_next_observation_count": generated_count,
            "live_wam_generation_success_count": live_wam_count,
            "learned_wam_model_success_count": learned_wam_count,
            "policy_observes_wam_generated_next_observation": repeated_policy_calls >= 2 and generated_count >= 1,
            "wam_evaluator_in_control_loop": generated_count >= 1,
            "clean_frame_reanchoring": clean_frame_reanchoring,
            "clean_frame_reanchor_event_count": len(clean_frame_reanchor_events),
            "clean_frame_reanchor_events": clean_frame_reanchor_events,
            "periodic_clean_frame_reanchoring_used": bool(clean_frame_reanchor_events),
            "unitree_groot_n17_sonic_model_executed": repeated_policy_calls >= 1,
            "unitree_groot_n17_sonic_policy_action_command_ran": repeated_policy_calls >= 1,
            "unitree_policy_action_command_ran": repeated_policy_calls >= 1,
            "policy_action_model_command_ran": repeated_policy_calls >= 1,
            "provider_output_replay_used": False,
            "trace_path": str(output_dir / "robot_policy_wam_loop_trace.jsonl"),
            "wam_generated_next_observations_jsonl": str(output_dir / "wam_generated_next_observations.jsonl"),
            "side_by_side_trace_path": str(output_dir / "robot_policy_wam_side_by_side_trace.jsonl"),
            "side_by_side_trace_html_path": str(output_dir / "robot_policy_wam_side_by_side_trace.html"),
            "blockers": sorted(set(str(item) for item in blockers if str(item))),
            "duration_seconds": round(time.monotonic() - started, 6),
            "claim_boundary": {
                "simulator_generated_world_proof_only": True,
                "persistent_provider_session_is_runtime_proof_not_task_success": True,
                "wam_is_next_observation_generator_not_robot_policy": True,
                "generated_observations_are_not_raw_capture": True,
                "capture_truth": False,
                "geometry_truth": False,
                "collision_truth": False,
                "provider_success": completed,
                "provider_success_separate_from_visually_useful_rollout": True,
                "periodic_clean_frame_reanchoring_is_quality_control_not_task_success": True,
                "visually_useful_rollout": False,
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
                "safety_validation_proven": False,
                "real_world_manipulation_success_proven": False,
            },
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
            "action": policy_calls[-1].get("action") if policy_calls else None,
        }
        _write_json(output_path, result)
        _write_json(output_dir / "unitree_groot_n17_sonic_wam_persistent_session_output.json", result)
        return 0 if completed else 2
    except Exception as exc:
        result = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "failed",
            "policy_id": POLICY_ID,
            "selected_candidate_id": POLICY_ID,
            "persistent_provider_session_used": True,
            "unitree_groot_n17_sonic_model_executed": False,
            "unitree_groot_n17_sonic_policy_action_command_ran": False,
            "unitree_policy_action_command_ran": False,
            "policy_action_model_command_ran": False,
            "provider_output_replay_used": False,
            "policy_server_bootstrap": locals().get("policy_server_bootstrap", {}),
            "traceback_tail": traceback.format_exc()[-4000:],
            "blockers": [f"persistent_session_runner_failed:{type(exc).__name__}"],
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
            "claim_boundary": {
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
                "safety_validation_proven": False,
                "real_world_manipulation_success_proven": False,
            },
        }
        _write_json(output_path, result)
        _write_json(output_dir / "unitree_groot_n17_sonic_wam_persistent_session_output.json", result)
        return 1
    finally:
        if policy_server is not None:
            policy_server.shutdown()
        if wam_server is not None:
            wam_server.shutdown()
        if policy_server_process is not None and policy_server_process.poll() is None:
            policy_server_process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
"""


RUN_SCRIPT = """#!/usr/bin/env bash
set +e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
RUNTIME_PY="${RUNTIME_PY:-python3}"
blueprint_phase_heartbeat() {
  phase="$1"
  PHASE="$phase" "$RUNTIME_PY" - <<'PY'
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

def _string(value: object) -> str:
    return str(value).strip() if value is not None else ""

if _truthy(os.environ.get("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_UPLOAD_PHASE_HEARTBEATS") or "true"):
    put_url = _string(os.environ.get("OUTPUT_PUT_URL"))
    work_dir = _string(os.environ.get("WORK_DIR"))
    output_dir_text = _string(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR"))
    if put_url and work_dir and output_dir_text:
        try:
            phase = _string(os.environ.get("PHASE")) or "unknown"
            output_dir = Path(output_dir_text).expanduser().resolve()
            output_path = Path(
                _string(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT"))
                or output_dir / "unitree_groot_n17_sonic_policy_provider_output.json"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": "unitree_groot_n17_sonic_wam_persistent_session_output.v1",
                "status": "running",
                "policy_id": "unitree_groot_n17_sonic_policy",
                "persistent_provider_session_used": True,
                "runtime_phase": phase,
                "runtime_phase_details": {
                    "phase": phase,
                    "observed_at_epoch": round(time.time(), 3),
                    "source": "runpod_entrypoint_script",
                    "raw_secret_values_recorded": False,
                },
                "runpod_unitree_groot_sonic_remote_heartbeat": True,
                "blockers": [],
                "raw_credentials_written_to_artifacts": False,
                "secret_hashes_written_to_artifacts": False,
            }
            output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
            zip_path = Path(work_dir) / "unitree_groot_n17_sonic_provider_phase_heartbeat.zip"
            tmp_zip_path = zip_path.with_suffix(zip_path.suffix + ".tmp")
            tmp_zip_path.unlink(missing_ok=True)
            with zipfile.ZipFile(tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(output_path, output_path.name)
            if not tmp_zip_path.stat().st_size or not zipfile.is_zipfile(tmp_zip_path):
                raise RuntimeError("invalid_or_empty_phase_heartbeat_zip")
            tmp_zip_path.replace(zip_path)
            request = urllib.request.Request(
                put_url,
                data=zip_path.read_bytes(),
                method="PUT",
                headers={"Content-Type": "application/zip"},
            )
            timeout_seconds = int(
                os.environ.get("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_PHASE_HEARTBEAT_TIMEOUT_SECONDS")
                or "20"
            )
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                response.read()
            print("BLUEPRINT_RUNPOD_ENTRYPOINT_PHASE_HEARTBEAT_UPLOAD_OK:%s" % phase, flush=True)
        except Exception as exc:
            print(
                "BLUEPRINT_RUNPOD_ENTRYPOINT_PHASE_HEARTBEAT_UPLOAD_BLOCKED:%s:%s"
                % (_string(os.environ.get("PHASE")) or "unknown", type(exc).__name__),
                flush=True,
            )
PY
}
blueprint_phase_heartbeat runpod_entrypoint_dependency_probe_started
if ! "$RUNTIME_PY" - <<'PY' >/tmp/blueprint_persistent_session_deps_probe.log 2>&1
import importlib.util
missing=[m for m in ['numpy','PIL','zmq','msgpack','msgpack_numpy','cv2'] if importlib.util.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
then
  blueprint_phase_heartbeat runpod_entrypoint_python_dependency_install_started
  "$RUNTIME_PY" -m pip install --quiet --only-binary=:all: --timeout 60 --retries 1 --break-system-packages numpy pillow pyzmq msgpack msgpack-numpy opencv-python-headless >/tmp/blueprint_persistent_session_pip_install.log 2>&1
  pip_install_rc=$?
  if [ $pip_install_rc -eq 0 ]; then
    blueprint_phase_heartbeat runpod_entrypoint_python_dependency_install_completed
  else
    blueprint_phase_heartbeat runpod_entrypoint_python_dependency_install_failed
  fi
else
  blueprint_phase_heartbeat runpod_entrypoint_python_dependencies_present
fi
blueprint_phase_heartbeat runpod_entrypoint_runner_starting
"$RUNTIME_PY" unitree_groot_n17_sonic_wam_persistent_session_runner.py
runner_rc=$?
if [ $runner_rc -ne 0 ] && [ ! -f "${BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT:-}" ]; then
"$RUNTIME_PY" - <<'PY'
import json
import os
from pathlib import Path
out = Path(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT", "unitree_groot_n17_sonic_policy_provider_output.json"))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "schema_version": "unitree_groot_n17_sonic_wam_persistent_session_output.v1",
    "status": "failed",
    "policy_id": "unitree_groot_n17_sonic_policy",
    "persistent_provider_session_used": True,
    "unitree_groot_n17_sonic_model_executed": False,
    "unitree_groot_n17_sonic_policy_action_command_ran": False,
    "policy_action_model_command_ran": False,
    "provider_output_replay_used": False,
    "blockers": ["persistent_session_runner_failed_without_runtime_result"],
    "legacy_blockers": [
        "unitree_groot_n17_sonic_provider_runner_failed_without_runtime_result",
        "blocked_unitree_groot_n17_sonic_process_exited_without_result"
    ],
    "raw_credentials_written_to_artifacts": False,
    "secret_hashes_written_to_artifacts": False
}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
fi
exit $runner_rc
	"""


RUNPOD_WAM_CARRIER_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
runtime_dir="$(cd "$(dirname "$0")" && pwd)"
bundle_dir="$(cd "$runtime_dir/.." && pwd)"
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_BUNDLE_DIR="$bundle_dir"
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR="${BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR:-${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-$bundle_dir/runtime_output}}"
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT="${BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT:-$BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR/unitree_groot_n17_sonic_policy_provider_output.json}"
mkdir -p "$BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR"
bash "$runtime_dir/run_unitree_groot_n17_sonic_runpod_wrapper.sh"
"""


RUNPOD_WRAPPER_SCRIPT = r"""#!/usr/bin/env bash
set -euo pipefail
echo BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_BUNDLE_WRAPPER_STARTED
WORK_DIR="${BLUEPRINT_RUNPOD_PROVIDER_WORK_DIR:-${WORK_DIR:-/workspace/blueprint_unitree_groot_sonic_persistent_provider}}"
PROVIDER_BUNDLE_DIR="${BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_BUNDLE_DIR:-$WORK_DIR/unitree_groot_n17_sonic_provider_bundle}"
OUTPUT_PUT_URL="${BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL:-${OUTPUT_PUT_URL:-}}"
export WORK_DIR PROVIDER_BUNDLE_DIR OUTPUT_PUT_URL
mkdir -p "$WORK_DIR/runtime_output"
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR="$WORK_DIR/runtime_output"
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT="$WORK_DIR/runtime_output/unitree_groot_n17_sonic_policy_provider_output.json"
export BLUEPRINT_PERSISTENT_SESSION_INPUT="$PROVIDER_BUNDLE_DIR/provider_runtime/persistent_session_input.json"
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_AUTO_START_POLICY_SERVER="${BLUEPRINT_UNITREE_GROOT_N17_SONIC_AUTO_START_POLICY_SERVER:-true}"
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_REMOTE_ROOT="${BLUEPRINT_UNITREE_GROOT_N17_SONIC_REMOTE_ROOT:-$WORK_DIR/groot_runtime/Isaac-GR00T}"
if [ -z "${BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT:-}" ] || [ ! -d "${BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT:-}" ]; then
  export BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT="$BLUEPRINT_UNITREE_GROOT_N17_SONIC_REMOTE_ROOT"
fi
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$WORK_DIR/hf_hub_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$WORK_DIR/transformers_cache}"
WRAPPER_PID="$$"
ENTRYPOINT_TIMEOUT_SECONDS="${BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_ENTRYPOINT_TIMEOUT_SECONDS:-7200}"
WRAPPER_WATCHDOG_SECONDS="${BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_WRAPPER_WATCHDOG_SECONDS:-}"
if [ -z "$WRAPPER_WATCHDOG_SECONDS" ]; then
  WRAPPER_WATCHDOG_SECONDS=$((ENTRYPOINT_TIMEOUT_SECONDS + 300))
fi
export WRAPPER_PID ENTRYPOINT_TIMEOUT_SECONDS WRAPPER_WATCHDOG_SECONDS
wrapper_watchdog_pid=""
entrypoint_heartbeat_pid=""
upload_unitree_groot_sonic_output() {
  shell_rc="${BLUEPRINT_WRAPPER_EXIT_RC:-$?}"
  set +e
  mkdir -p "$BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR"
  if [ ! -f "$BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT" ]; then
SHELL_EXIT_RC="$shell_rc" python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT"])
out.parent.mkdir(parents=True, exist_ok=True)
entrypoint_rc = os.environ.get("entrypoint_rc")
out.write_text(json.dumps({
    "schema_version": "unitree_groot_n17_sonic_wam_persistent_session_output.v1",
    "status": "blocked",
    "policy_id": "unitree_groot_n17_sonic_policy",
    "persistent_provider_session_used": True,
    "unitree_groot_n17_sonic_model_executed": False,
    "unitree_groot_n17_sonic_policy_action_command_ran": False,
    "policy_action_model_command_ran": False,
    "provider_output_replay_used": False,
    "blockers": ["runpod_unitree_groot_sonic_bundle_wrapper_exited_before_runtime_result"],
    "shell_exit_returncode": int(os.environ.get("SHELL_EXIT_RC", "1") or 1),
    "entrypoint_returncode": int(entrypoint_rc) if entrypoint_rc else None,
    "raw_credentials_written_to_artifacts": False,
    "secret_hashes_written_to_artifacts": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  fi
  for log in \
    /tmp/blueprint_runpod_apt_update.log \
    /tmp/blueprint_runpod_apt_install.log \
    /tmp/blueprint_persistent_session_deps_probe.log \
    /tmp/blueprint_persistent_session_pip_install.log
  do
    cp "$log" "$BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR/$(basename "$log")" 2>/dev/null || true
  done
  python - <<'PY'
import json
import os
import zipfile
from pathlib import Path

output_dir = Path(os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR"])
zip_path = Path(os.environ["WORK_DIR"]) / "unitree_groot_n17_sonic_provider_runtime_output.zip"
tmp_zip_path = zip_path.with_suffix(zip_path.suffix + ".tmp")
excluded_dirs = {
    ".git",
    ".venv",
    "__pycache__",
    "checkpoints",
    "groot_runtime",
    "hf_home",
    "hf_hub_cache",
    "transformers_cache",
}
max_file_bytes = 8 * 1024 * 1024
omitted = []
try:
    tmp_zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if output_dir.is_dir():
            for root, dirs, files in os.walk(output_dir):
                dirs[:] = sorted(item for item in dirs if item not in excluded_dirs)
                current_root = Path(root)
                for filename in sorted(files):
                    path = current_root / filename
                    relative = path.relative_to(output_dir)
                    if path.stat().st_size > max_file_bytes:
                        omitted.append(str(relative))
                        continue
                    archive.write(path, relative.as_posix())
            if omitted:
                archive.writestr(
                    "runpod_unitree_groot_sonic_runtime_output_omissions.json",
                    json.dumps(
                        {
                            "schema_version": "runpod_runtime_output_omissions.v1",
                            "omitted_count": len(omitted),
                            "omitted_paths_sample": omitted[:200],
                            "max_file_bytes": max_file_bytes,
                            "raw_secret_values_recorded": False,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                )
        else:
            archive.writestr(
                "unitree_groot_n17_sonic_wam_persistent_session_output.json",
                json.dumps({"status": "blocked", "blockers": ["runtime_output_directory_missing"]}, indent=2),
            )
    if not tmp_zip_path.stat().st_size or not zipfile.is_zipfile(tmp_zip_path):
        raise RuntimeError("invalid_or_empty_runtime_output_zip")
    tmp_zip_path.replace(zip_path)
except Exception as exc:
    tmp_zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_wam_persistent_session_output.json",
            json.dumps(
                {
                    "schema_version": "unitree_groot_n17_sonic_wam_persistent_session_output.v1",
                    "status": "blocked",
                    "policy_id": "unitree_groot_n17_sonic_policy",
                    "persistent_provider_session_used": True,
                    "unitree_groot_n17_sonic_model_executed": False,
                    "unitree_groot_n17_sonic_policy_action_command_ran": False,
                    "policy_action_model_command_ran": False,
                    "provider_output_replay_used": False,
                    "blockers": ["runpod_runtime_output_zip_creation_failed"],
                    "zip_creation_error_type": type(exc).__name__,
                    "zip_creation_error_preview": str(exc)[:400],
                    "raw_credentials_written_to_artifacts": False,
                    "secret_hashes_written_to_artifacts": False,
                },
                indent=2,
                sort_keys=True,
            ),
        )
print("BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_OUTPUT_ZIP_WRITTEN:%d" % zip_path.stat().st_size)
PY
  if [ -n "${OUTPUT_PUT_URL:-}" ]; then
python - <<'PY'
import json
import os
import urllib.request
import zipfile
from pathlib import Path

zip_path = Path(os.environ["WORK_DIR"]) / "unitree_groot_n17_sonic_provider_runtime_output.zip"
timeout_seconds = int(os.environ.get("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_UPLOAD_TIMEOUT_SECONDS", "60") or "60")
status_path = Path(os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR"]) / "runpod_unitree_groot_sonic_output_upload_status.json"
try:
    if not zip_path.stat().st_size or not zipfile.is_zipfile(zip_path):
        raise RuntimeError("invalid_or_empty_runtime_output_zip")
    request = urllib.request.Request(
        os.environ["OUTPUT_PUT_URL"],
        data=zip_path.read_bytes(),
        method="PUT",
        headers={"Content-Type": "application/zip"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response.read()
    status_path.write_text(
        json.dumps(
            {
                "schema_version": "runpod_unitree_groot_sonic_output_upload_status.v1",
                "status": "completed",
                "timeout_seconds": timeout_seconds,
                "raw_secret_values_recorded": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_OUTPUT_UPLOAD_OK")
except Exception as exc:
    status_path.write_text(
        json.dumps(
            {
                "schema_version": "runpod_unitree_groot_sonic_output_upload_status.v1",
                "status": "blocked",
                "timeout_seconds": timeout_seconds,
                "error_type": type(exc).__name__,
                "error_preview": str(exc)[:400],
                "raw_secret_values_recorded": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_OUTPUT_UPLOAD_BLOCKED:%s" % type(exc).__name__)
PY
  else
    echo BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_OUTPUT_UPLOAD_SKIPPED:output_put_url_missing
  fi
	}
	trap 'BLUEPRINT_WRAPPER_EXIT_RC=$?; if [ -n "${wrapper_watchdog_pid:-}" ]; then kill "$wrapper_watchdog_pid" 2>/dev/null || true; fi; if [ -n "${entrypoint_heartbeat_pid:-}" ]; then kill "$entrypoint_heartbeat_pid" 2>/dev/null || true; fi; upload_unitree_groot_sonic_output; exit "$BLUEPRINT_WRAPPER_EXIT_RC"' EXIT
	python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "schema_version": "unitree_groot_n17_sonic_wam_persistent_session_output.v1",
    "status": "running",
    "policy_id": "unitree_groot_n17_sonic_policy",
    "persistent_provider_session_used": True,
    "runtime_phase": "runpod_bundle_wrapper_started",
    "runpod_unitree_groot_sonic_remote_heartbeat": True,
    "blockers": [],
    "raw_credentials_written_to_artifacts": False,
    "secret_hashes_written_to_artifacts": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
		if [ "${BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_UPLOAD_BOOTSTRAP_HEARTBEAT:-true}" = "true" ]; then
	  upload_unitree_groot_sonic_output
	fi
	rm -f "$BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT"
	write_unitree_groot_sonic_phase_heartbeat() {
	  phase="$1"
	  if [ -f "$BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT" ]; then
	    if python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT"])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "running" else 1)
PY
	    then
	      if [ "${BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_UPLOAD_PHASE_HEARTBEATS:-true}" = "true" ]; then
	        upload_unitree_groot_sonic_output
	      fi
	    fi
	    return 0
	  fi
PHASE="$phase" python - <<'PY'
import json
import os
import time
from pathlib import Path

out = Path(os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT"])
out.parent.mkdir(parents=True, exist_ok=True)
phase = os.environ.get("PHASE", "unknown")
entrypoint_log = out.parent / "runpod_unitree_groot_sonic_entrypoint.log"
entrypoint_log_tail = None
if entrypoint_log.is_file():
    entrypoint_log_tail = entrypoint_log.read_text(encoding="utf-8", errors="replace")[-4000:]
out.write_text(json.dumps({
    "schema_version": "unitree_groot_n17_sonic_wam_persistent_session_output.v1",
    "status": "running",
    "policy_id": "unitree_groot_n17_sonic_policy",
    "persistent_provider_session_used": True,
    "runtime_phase": phase,
    "runtime_phase_details": {
        "phase": phase,
        "observed_at_epoch": round(time.time(), 3),
        "source": "runpod_outer_wrapper",
        "entrypoint_log_tail": entrypoint_log_tail,
        "raw_secret_values_recorded": False,
    },
    "runpod_unitree_groot_sonic_remote_heartbeat": True,
    "blockers": [],
    "raw_credentials_written_to_artifacts": False,
    "secret_hashes_written_to_artifacts": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
	  if [ "${BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_UPLOAD_PHASE_HEARTBEATS:-true}" = "true" ]; then
	    upload_unitree_groot_sonic_output
	  fi
	  rm -f "$BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT"
	}
	set -euo pipefail
	(
	  sleep "$WRAPPER_WATCHDOG_SECONDS"
	  if [ ! -f "$BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT" ]; then
WRAPPER_WATCHDOG_SECONDS="$WRAPPER_WATCHDOG_SECONDS" python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "schema_version": "unitree_groot_n17_sonic_wam_persistent_session_output.v1",
    "status": "blocked",
    "policy_id": "unitree_groot_n17_sonic_policy",
    "persistent_provider_session_used": True,
    "unitree_groot_n17_sonic_model_executed": False,
    "unitree_groot_n17_sonic_policy_action_command_ran": False,
    "policy_action_model_command_ran": False,
    "provider_output_replay_used": False,
    "blockers": ["runpod_unitree_groot_sonic_wrapper_watchdog_timeout_before_runtime_result"],
    "wrapper_watchdog_timeout_seconds": int(os.environ.get("WRAPPER_WATCHDOG_SECONDS", "0") or 0),
    "raw_credentials_written_to_artifacts": False,
    "secret_hashes_written_to_artifacts": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    upload_unitree_groot_sonic_output
    kill -TERM "$WRAPPER_PID" 2>/dev/null || true
  fi
) &
wrapper_watchdog_pid=$!
if command -v apt-get >/dev/null 2>&1; then
  write_unitree_groot_sonic_phase_heartbeat runpod_system_dependency_check_started
  if ! command -v git >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
    write_unitree_groot_sonic_phase_heartbeat runpod_system_dependency_install_started
    set +e
    timeout 300 apt-get update >/tmp/blueprint_runpod_apt_update.log 2>&1
    apt_update_rc=$?
    DEBIAN_FRONTEND=noninteractive timeout 600 apt-get install -y git ffmpeg curl ca-certificates >/tmp/blueprint_runpod_apt_install.log 2>&1
    apt_install_rc=$?
    set -e
    if [ $apt_update_rc -eq 0 ] && [ $apt_install_rc -eq 0 ]; then
      write_unitree_groot_sonic_phase_heartbeat runpod_system_dependency_install_completed
    else
      write_unitree_groot_sonic_phase_heartbeat runpod_system_dependency_install_failed
    fi
  else
    write_unitree_groot_sonic_phase_heartbeat runpod_system_dependencies_present
  fi
fi
export PYTHONPATH="$PROVIDER_BUNDLE_DIR/provider_runtime:${PYTHONPATH:-}"
write_unitree_groot_sonic_phase_heartbeat runpod_entrypoint_subprocess_starting
(
  while true; do
    sleep "${BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_ENTRYPOINT_HEARTBEAT_SECONDS:-60}"
    write_unitree_groot_sonic_phase_heartbeat runpod_entrypoint_subprocess_running
  done
) &
entrypoint_heartbeat_pid=$!
echo BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_ENTRYPOINT_STARTED
set +e
python - <<'PY'
import json
import os
import signal
import subprocess
import time
from pathlib import Path

output_dir = Path(os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)
entrypoint = (
    Path(os.environ["PROVIDER_BUNDLE_DIR"])
    / "provider_runtime"
    / "run_unitree_groot_n17_sonic_provider_runtime.sh"
)
timeout_seconds = int(
    os.environ.get("ENTRYPOINT_TIMEOUT_SECONDS")
    or os.environ.get("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_ENTRYPOINT_TIMEOUT_SECONDS")
    or "7200"
)
log_path = output_dir / "runpod_unitree_groot_sonic_entrypoint.log"
execution_path = output_dir / "runpod_unitree_groot_sonic_entrypoint_execution.json"
started = time.time()
timed_out = False
returncode = None
with log_path.open("ab") as handle:
    handle.write(
        (
            "BLUEPRINT_ENTRYPOINT_STARTED:"
            + json.dumps(
                {
                    "entrypoint": str(entrypoint),
                    "timeout_seconds": timeout_seconds,
                    "raw_secret_values_recorded": False,
                },
                sort_keys=True,
            )
            + "\n"
        ).encode()
    )
    handle.flush()
    proc = subprocess.Popen(
        ["bash", str(entrypoint)],
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        start_new_session=True,
    )
    try:
        returncode = proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            returncode = proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            returncode = proc.wait(timeout=20)
        handle.write(b"\nBLUEPRINT_ENTRYPOINT_TIMED_OUT\n")
duration = round(time.time() - started, 3)
execution_path.write_text(
    json.dumps(
        {
            "schema_version": "runpod_unitree_groot_sonic_entrypoint_execution.v1",
            "status": "timed_out" if timed_out else ("completed" if returncode == 0 else "failed"),
            "returncode": returncode,
            "timed_out": timed_out,
            "timeout_seconds": timeout_seconds,
            "duration_seconds": duration,
            "entrypoint_log_path": str(log_path),
            "raw_secret_values_recorded": False,
            "secret_hashes_recorded": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
raise SystemExit(returncode if returncode is not None else 124)
PY
entrypoint_rc=$?
export entrypoint_rc
set -e
if [ ! -f "$BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT" ]; then
python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT"])
out.parent.mkdir(parents=True, exist_ok=True)
rc = int(os.environ.get("entrypoint_rc", "124") or 124)
out.write_text(json.dumps({
    "schema_version": "unitree_groot_n17_sonic_wam_persistent_session_output.v1",
    "status": "blocked",
    "policy_id": "unitree_groot_n17_sonic_policy",
    "persistent_provider_session_used": True,
    "unitree_groot_n17_sonic_model_executed": False,
    "unitree_groot_n17_sonic_policy_action_command_ran": False,
    "policy_action_model_command_ran": False,
    "provider_output_replay_used": False,
    "blockers": ["persistent_session_entrypoint_exited_without_runtime_result"],
    "entrypoint_returncode": rc,
    "raw_credentials_written_to_artifacts": False,
    "secret_hashes_written_to_artifacts": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi
echo BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_PROVIDER_COMPLETED_OR_BLOCKED
kill "$entrypoint_heartbeat_pid" 2>/dev/null || true
kill "$wrapper_watchdog_pid" 2>/dev/null || true
"""


def _copy_blueprint_runtime(runtime_dir: Path) -> list[str]:
    package_dir = runtime_dir / "blueprint_pipeline"
    ensure_dir(package_dir)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    copied = ["provider_runtime/blueprint_pipeline/__init__.py"]
    source_dir = Path(__file__).resolve().parent
    for filename in (
        "common.py",
        "unitree_groot_n17_sonic_policy_command_adapter.py",
        "unitree_groot_n17_sonic_policy_runtime.py",
        "unitree_groot_n17_sonic_policy_server_command.py",
        "unitree_groot_n17_sonic_provider_smoke.py",
        "oscar_wam_provider_bundle.py",
        "oscar_wam_command_adapter.py",
        "wam_generated_video_review.py",
    ):
        shutil.copy2(source_dir / filename, package_dir / filename)
        copied.append(f"provider_runtime/blueprint_pipeline/{filename}")
    return copied


def build_persistent_session_provider_bundle(
    *,
    job_dir: str | Path,
    policy_observation_path: str | Path,
    loop_step_count: int = 12,
    task_prompt: str | None = None,
    timeout_seconds: float = 3600.0,
    use_live_wam: bool = True,
    allow_structural_wam_fallback: bool = False,
    bundle_filename: str = DEFAULT_BUNDLE_FILENAME,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    runtime_dir = job / "provider_runtime"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    ensure_dir(runtime_dir)
    observation = _load_policy_observation(policy_observation_path)
    original_observation_source_kind = _policy_observation_source_kind(observation)
    if task_prompt and not any(
        observation.get(key) for key in ("task_prompt", "prompt", "task_description")
    ):
        observation["task_prompt"] = task_prompt
    frame_path = _camera_frame_path(observation)
    visual_profile_settings = _current_wam_visual_profile_settings()
    visual_profile = str(visual_profile_settings["visual_profile"])
    semantic_visual_evidence = _policy_observation_semantic_visual_evidence(
        observation,
        base_dir=Path(policy_observation_path).expanduser().parent,
    )
    source_visual_qa = assess_source_policy_observation_visual_qa(
        frame_path,
        generated_at=generated,
        target_object_id=_string(observation.get("target_object_id")) or None,
        task_id=_string(observation.get("task_id")) or None,
        object_index=semantic_visual_evidence.get("object_index"),
        eval_ready_task_grounding=semantic_visual_evidence.get("eval_ready_task_grounding"),
        semantic_artifact_base_dir=semantic_visual_evidence.get("semantic_artifact_base_dir"),
        visual_profile=visual_profile,
        review_quality_required=visual_profile == "review_quality",
    )
    original_source_visual_qa = dict(source_visual_qa)
    original_frame_path = frame_path
    original_source_visual_qa_path: Path | None = None
    remediation_manifest: dict[str, Any] | None = None
    remediation_manifest_path: Path | None = None
    remediation_applied = False
    if (
        visual_profile == "review_quality"
        and source_visual_qa.get("status") != "passed_visual_quality_gate"
        and image_model_render_remediation_enabled()
    ):
        original_source_visual_qa_path = job / "original_source_policy_observation_visual_qa.json"
        write_json(original_source_visual_qa_path, original_source_visual_qa)
        remediation_dir = job / "image_model_render_remediation"
        remediation_manifest = run_image_model_render_remediation(
            original_frame_path=frame_path,
            source_visual_qa=source_visual_qa,
            output_dir=remediation_dir,
            generated_at=generated,
            task_id=_string(observation.get("task_id")) or None,
            target_object_id=_string(observation.get("target_object_id")) or None,
            object_index=semantic_visual_evidence.get("object_index"),
            eval_ready_task_grounding=semantic_visual_evidence.get("eval_ready_task_grounding"),
            semantic_artifact_base_dir=semantic_visual_evidence.get("semantic_artifact_base_dir"),
            visual_profile=visual_profile,
            review_quality_required=True,
        )
        remediation_manifest_path = remediation_dir / "image_model_render_remediation_manifest.json"
        enhanced_frame_text = _string(remediation_manifest.get("enhanced_frame_path"))
        enhanced_qa_path_text = _string(remediation_manifest.get("enhanced_source_visual_qa_path"))
        if remediation_manifest.get("status") == "completed" and enhanced_frame_text:
            enhanced_frame_path = Path(enhanced_frame_text).expanduser()
            if enhanced_frame_path.is_file():
                frame_path = enhanced_frame_path.resolve()
                if enhanced_qa_path_text and Path(enhanced_qa_path_text).is_file():
                    source_visual_qa = _read_json(Path(enhanced_qa_path_text))
                else:
                    source_visual_qa = assess_source_policy_observation_visual_qa(
                        frame_path,
                        generated_at=generated,
                        target_object_id=_string(observation.get("target_object_id")) or None,
                        task_id=_string(observation.get("task_id")) or None,
                        object_index=semantic_visual_evidence.get("object_index"),
                        eval_ready_task_grounding=semantic_visual_evidence.get(
                            "eval_ready_task_grounding"
                        ),
                        semantic_artifact_base_dir=semantic_visual_evidence.get(
                            "semantic_artifact_base_dir"
                        ),
                        visual_profile=visual_profile,
                        review_quality_required=True,
                    )
                visual = _mapping(observation.get("visual_observation"))
                visual["camera_frame_path"] = str(frame_path)
                visual["image_model_render_remediation_applied"] = True
                visual["original_3d_render_frame_path"] = (
                    str(original_frame_path) if original_frame_path else None
                )
                visual["image_model_render_remediation_manifest_path"] = str(
                    remediation_manifest_path
                )
                observation["visual_observation"] = visual
                observation["camera_frame_path"] = str(frame_path)
                observation["source_kind"] = "image_model_enhanced_3d_render_seed"
                observation["image_model_render_remediation"] = {
                    "status": "completed",
                    "manifest_path": str(remediation_manifest_path),
                    "original_frame_path": str(original_frame_path)
                    if original_frame_path
                    else None,
                    "enhanced_frame_path": str(frame_path),
                    "source_visual_qa_before_remediation_path": str(original_source_visual_qa_path),
                }
                claim_boundary = _mapping(observation.get("claim_boundary"))
                claim_boundary.update(
                    {
                        "image_model_enhanced_policy_observation_used": True,
                        "enhanced_policy_observation_is_not_capture_truth": True,
                        "enhanced_policy_observation_is_not_geometry_truth": True,
                        "enhanced_policy_observation_is_not_collision_truth": True,
                        "capture_truth": False,
                        "geometry_truth": False,
                        "collision_truth": False,
                    }
                )
                observation["claim_boundary"] = claim_boundary
                remediation_applied = True
    synthetic_launch_gate = _synthetic_fallback_wam_launch_gate(
        observation=observation,
        original_source_kind=original_observation_source_kind,
        visual_profile=visual_profile,
        use_live_wam=use_live_wam,
    )
    observation = _apply_synthetic_fallback_truth_labels(observation, synthetic_launch_gate)
    source_visual_qa_path = job / "source_policy_observation_visual_qa.json"
    write_json(source_visual_qa_path, source_visual_qa)
    blockers: list[str] = []
    blockers.extend(str(item) for item in synthetic_launch_gate.get("blockers") or [])
    auxiliary_observation_manifest: dict[str, Any] = {}
    runtime_auxiliary_observation_manifest: dict[str, Any] = {}
    auxiliary_observation_manifest_path: Path | None = None
    runtime_auxiliary_observation_manifest_path: Path | None = None
    if frame_path is None:
        blockers.append("blocked_missing_policy_visual_observation_frame")
    else:
        shutil.copy2(frame_path, runtime_dir / "initial_policy_frame.png")
        shutil.copy2(frame_path, runtime_dir / "input_frame.png")
        write_json(runtime_dir / "source_policy_observation_visual_qa.json", source_visual_qa)
        auxiliary_observation_manifest = build_wam_auxiliary_observation_manifest(
            output_dir=job / "wam_auxiliary_observation",
            source_image_path=frame_path,
            policy_observation=observation,
            generated_at=generated,
            source_kind=_string(observation.get("source_kind")) or None,
            camera_id=_string(_mapping(observation.get("visual_observation")).get("camera_id"))
            or _string(observation.get("camera_id"))
            or None,
            robot_profile_id=_string(observation.get("robot_profile_id")) or None,
            task_id=_string(observation.get("task_id")) or None,
            target_object_id=_string(observation.get("target_object_id")) or None,
        )
        auxiliary_observation_manifest_path = Path(
            str(auxiliary_observation_manifest["manifest_path"])
        )
        runtime_observation = json.loads(json.dumps(observation))
        runtime_visual = _mapping(runtime_observation.get("visual_observation"))
        runtime_visual["camera_frame_path"] = str(runtime_dir / "initial_policy_frame.png")
        runtime_visual["source_image_path"] = str(runtime_dir / "initial_policy_frame.png")
        runtime_observation["visual_observation"] = runtime_visual
        runtime_observation["camera_frame_path"] = str(runtime_dir / "initial_policy_frame.png")
        runtime_auxiliary_observation_manifest = build_wam_auxiliary_observation_manifest(
            output_dir=runtime_dir / "wam_auxiliary_observation",
            source_image_path=runtime_dir / "initial_policy_frame.png",
            policy_observation=runtime_observation,
            generated_at=generated,
            source_kind=_string(runtime_observation.get("source_kind")) or None,
            camera_id=_string(runtime_visual.get("camera_id"))
            or _string(runtime_observation.get("camera_id"))
            or None,
            robot_profile_id=_string(runtime_observation.get("robot_profile_id")) or None,
            task_id=_string(runtime_observation.get("task_id")) or None,
            target_object_id=_string(runtime_observation.get("target_object_id")) or None,
        )
        runtime_auxiliary_observation_manifest_path = Path(
            str(runtime_auxiliary_observation_manifest["manifest_path"])
        )
        runtime_auxiliary_observation_manifest["manifest_path"] = (
            "provider_runtime/wam_auxiliary_observation/wam_auxiliary_observation_manifest.json"
        )
        runtime_auxiliary_observation_manifest["source_image_path"] = (
            "provider_runtime/initial_policy_frame.png"
        )
        runtime_auxiliary_observation_manifest["source_image_path_exists"] = True
        runtime_auxiliary_observation_manifest["runtime_paths_rewritten_for_provider_bundle"] = True
        write_json(
            runtime_auxiliary_observation_manifest_path,
            runtime_auxiliary_observation_manifest,
        )
        auxiliary_summary = summarize_wam_auxiliary_observation_manifest(
            runtime_auxiliary_observation_manifest
        )
        visual = _mapping(observation.get("visual_observation"))
        visual["wam_auxiliary_observation_manifest_path"] = str(
            runtime_auxiliary_observation_manifest_path
        )
        observation["visual_observation"] = visual
        observation["wam_auxiliary_observation_manifest_path"] = str(
            runtime_auxiliary_observation_manifest_path
        )
        observation["wam_auxiliary_observation"] = auxiliary_summary
    if original_source_visual_qa_path and original_source_visual_qa_path.is_file():
        shutil.copy2(
            original_source_visual_qa_path,
            runtime_dir / "original_source_policy_observation_visual_qa.json",
        )
    if remediation_manifest_path and remediation_manifest_path.is_file():
        runtime_remediation_dir = runtime_dir / "image_model_render_remediation"
        if runtime_remediation_dir.exists():
            shutil.rmtree(runtime_remediation_dir)
        shutil.copytree(remediation_manifest_path.parent, runtime_remediation_dir)
        if remediation_manifest and remediation_manifest.get("status") != "completed":
            blockers.extend(
                str(item)
                for item in remediation_manifest.get("blockers")
                or ["image_model_render_remediation_blocked"]
            )
    profile_blockers = _persistent_wam_visual_profile_blockers(
        settings=visual_profile_settings,
        source_visual_qa=source_visual_qa,
        loop_step_count=int(loop_step_count),
        policy_observation_path=policy_observation_path,
    )
    long_review_quality_gate = _persistent_wam_long_review_rollout_quality_gate(
        settings=visual_profile_settings,
        loop_step_count=int(loop_step_count),
    )
    long_review_quality_gate_path = job / "long_review_rollout_quality_gate.json"
    write_json(long_review_quality_gate_path, long_review_quality_gate)
    blockers.extend(profile_blockers)
    blockers.extend(str(item) for item in long_review_quality_gate.get("blockers") or [])
    copied = _copy_blueprint_runtime(runtime_dir)
    _write_executable(
        runtime_dir / "unitree_groot_n17_sonic_wam_persistent_session_runner.py",
        PERSISTENT_SESSION_RUNNER,
    )
    _write_executable(
        runtime_dir / "unitree_groot_n17_sonic_provider_runner.py",
        PERSISTENT_SESSION_RUNNER,
    )
    _write_executable(runtime_dir / "run_unitree_groot_n17_sonic_provider_runtime.sh", RUN_SCRIPT)
    _write_executable(runtime_dir / "run_wam_provider_runtime.sh", RUNPOD_WAM_CARRIER_SCRIPT)
    _write_executable(
        runtime_dir / "run_unitree_groot_n17_sonic_runpod_wrapper.sh",
        RUNPOD_WRAPPER_SCRIPT,
    )
    session_input = {
        "schema_version": "unitree_groot_n17_sonic_wam_persistent_session_input.v1",
        "generated_at": generated,
        "initial_observation": observation,
        "loop_step_count": int(loop_step_count),
        "timeout_seconds": float(timeout_seconds),
        "visual_profile": visual_profile,
        "wam_visual_profile_settings": visual_profile_settings,
        "clean_frame_reanchoring": _mapping(
            long_review_quality_gate.get("clean_frame_reanchoring")
        ),
        "long_review_rollout_quality_gate": {
            "status": long_review_quality_gate.get("status"),
            "required_before_12_step_paid_review_quality_rollout": long_review_quality_gate.get(
                "required_before_12_step_paid_review_quality_rollout"
            ),
            "periodic_clean_frame_reanchoring_proven": long_review_quality_gate.get(
                "periodic_clean_frame_reanchoring_proven"
            ),
            "concrete_autoregressive_drift_blocker_proven": long_review_quality_gate.get(
                "concrete_autoregressive_drift_blocker_proven"
            ),
            "blockers": list(long_review_quality_gate.get("blockers") or []),
        },
        "use_live_wam": bool(use_live_wam),
        "allow_structural_wam_fallback": bool(allow_structural_wam_fallback),
        "synthetic_fallback_wam_launch_gate": synthetic_launch_gate,
        "policy_worker_port": 8765,
        "wam_worker_port": 8766,
        "image_model_render_remediation": {
            "enabled_env": IMAGE_MODEL_RENDER_REMEDIATION_ENABLE_ENV,
            "enabled": image_model_render_remediation_enabled(),
            "applied": remediation_applied,
            "status": remediation_manifest.get("status")
            if remediation_manifest
            else "not_attempted",
            "manifest_path": str(remediation_manifest_path) if remediation_manifest_path else None,
            "original_frame_path": str(original_frame_path) if original_frame_path else None,
            "effective_frame_path": str(frame_path) if frame_path else None,
        },
        "wam_auxiliary_observation": {
            "status": runtime_auxiliary_observation_manifest.get("status")
            if runtime_auxiliary_observation_manifest
            else "not_attempted",
            "local_manifest_path": str(auxiliary_observation_manifest_path)
            if auxiliary_observation_manifest_path
            else None,
            "runtime_manifest_path": str(runtime_auxiliary_observation_manifest_path)
            if runtime_auxiliary_observation_manifest_path
            else None,
            "modalities_available": _mapping(
                runtime_auxiliary_observation_manifest.get("modalities_available")
            ),
            "claim_boundary": _mapping(
                runtime_auxiliary_observation_manifest.get("claim_boundary")
            ),
        },
        "claim_boundary": {
            "simulator_generated_world_proof_only": True,
            "capture_truth": False,
            "geometry_truth": False,
            "collision_truth": False,
            "initial_policy_observation_capture_truth": synthetic_launch_gate.get(
                "capture_truth"
            ),
            "initial_policy_observation_geometry_truth": synthetic_launch_gate.get(
                "geometry_truth"
            ),
            "synthetic_fallback_initial_observation_used": synthetic_launch_gate.get(
                "synthetic_fallback_initial_observation_used"
            ),
            "synthetic_fallback_wam_launch_experiment_enabled": synthetic_launch_gate.get(
                "experimental_env_enabled"
            ),
            "provider_success_separate_from_visually_useful_rollout": True,
            "visually_useful_rollout": False,
            "image_model_enhanced_policy_observation_used": remediation_applied,
            "enhanced_policy_observation_is_not_capture_truth": remediation_applied,
            "enhanced_policy_observation_is_not_geometry_truth": remediation_applied,
            "enhanced_policy_observation_is_not_collision_truth": remediation_applied,
            "wam_auxiliary_observation_is_conditioning_support": bool(
                runtime_auxiliary_observation_manifest
            ),
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
        },
    }
    write_json(runtime_dir / "persistent_session_input.json", session_input)
    write_json(runtime_dir / "policy_input.json", {"observation": observation})
    write_json(
        runtime_dir / "unitree_groot_n17_sonic_policy_provider_manifest.json",
        {
            "schema_version": "unitree_groot_n17_sonic_policy_provider_bundle.v1",
            "generated_at": generated,
            "status": "bundle_ready" if not blockers else "blocked",
            "local_bundle_ready_for_remote_staging": not blockers,
            "persistent_session_bundle": True,
            "visual_profile": visual_profile,
            "wam_visual_profile_settings": visual_profile_settings,
            "source_policy_observation_visual_qa_path": str(source_visual_qa_path),
            "source_policy_observation_visual_qa_status": source_visual_qa.get("status"),
            "long_review_rollout_quality_gate_path": str(long_review_quality_gate_path),
            "long_review_rollout_quality_gate_status": long_review_quality_gate.get("status"),
            "clean_frame_reanchoring": _mapping(
                long_review_quality_gate.get("clean_frame_reanchoring")
            ),
            "original_initial_frame_path": str(original_frame_path)
            if original_frame_path
            else None,
            "original_source_policy_observation_visual_qa_path": str(original_source_visual_qa_path)
            if original_source_visual_qa_path
            else None,
            "image_model_render_remediation_enabled": image_model_render_remediation_enabled(),
            "image_model_render_remediation_applied": remediation_applied,
            "image_model_render_remediation_status": remediation_manifest.get("status")
            if remediation_manifest
            else "not_attempted",
            "image_model_render_remediation_manifest_path": str(remediation_manifest_path)
            if remediation_manifest_path
            else None,
            "synthetic_fallback_wam_launch_gate": synthetic_launch_gate,
            "wam_auxiliary_observation_manifest_path": str(auxiliary_observation_manifest_path)
            if auxiliary_observation_manifest_path
            else None,
            "runtime_wam_auxiliary_observation_manifest_path": str(
                runtime_auxiliary_observation_manifest_path
            )
            if runtime_auxiliary_observation_manifest_path
            else None,
            "wam_auxiliary_observation_modalities_available": _mapping(
                runtime_auxiliary_observation_manifest.get("modalities_available")
            ),
            "semantic_visual_qa_source_paths": {
                "object_index": semantic_visual_evidence.get("object_index_path"),
                "eval_ready_task_grounding": semantic_visual_evidence.get(
                    "eval_ready_task_grounding_path"
                ),
            },
            "runtime_entrypoint": "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh",
            "runpod_wam_carrier_entrypoint": "provider_runtime/run_wam_provider_runtime.sh",
            "runpod_runtime_wrapper": "provider_runtime/run_unitree_groot_n17_sonic_runpod_wrapper.sh",
            "runner_path": "provider_runtime/unitree_groot_n17_sonic_wam_persistent_session_runner.py",
            "legacy_runner_path_for_vast_preflight": "provider_runtime/unitree_groot_n17_sonic_provider_runner.py",
            "policy_id": POLICY_ID,
            "blockers": blockers,
            "claim_boundary": {
                "bundle_build_is_not_model_execution": True,
                "capture_truth": False,
                "geometry_truth": False,
                "collision_truth": False,
                "initial_policy_observation_capture_truth": synthetic_launch_gate.get(
                    "capture_truth"
                ),
                "initial_policy_observation_geometry_truth": synthetic_launch_gate.get(
                    "geometry_truth"
                ),
                "synthetic_fallback_initial_observation_used": synthetic_launch_gate.get(
                    "synthetic_fallback_initial_observation_used"
                ),
                "synthetic_fallback_wam_launch_experiment_enabled": synthetic_launch_gate.get(
                    "experimental_env_enabled"
                ),
                "provider_success_separate_from_visually_useful_rollout": True,
                "visually_useful_rollout": False,
                "image_model_enhanced_policy_observation_used": remediation_applied,
                "enhanced_policy_observation_is_not_capture_truth": remediation_applied,
                "enhanced_policy_observation_is_not_geometry_truth": remediation_applied,
                "enhanced_policy_observation_is_not_collision_truth": remediation_applied,
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
                "safety_validation_proven": False,
            },
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        },
    )
    bundle_path = job / bundle_filename
    if bundle_path.exists():
        bundle_path.unlink()
    zip_entries: list[str] = []
    if not blockers:
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(runtime_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(job).as_posix())
        with zipfile.ZipFile(bundle_path) as archive:
            zip_entries = sorted(archive.namelist())
            if archive.testzip() is not None:
                blockers.append("persistent_session_bundle_zip_integrity_failed")
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "bundle_ready" if not blockers else "blocked",
        "job_dir": str(job),
        "bundle_path": str(bundle_path),
        "bundle_present": bundle_path.is_file(),
        "bundle_size_bytes": bundle_path.stat().st_size if bundle_path.is_file() else 0,
        "runtime_entrypoint": "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh",
        "runpod_runtime_wrapper": "provider_runtime/run_unitree_groot_n17_sonic_runpod_wrapper.sh",
        "runtime_runner": "provider_runtime/unitree_groot_n17_sonic_wam_persistent_session_runner.py",
        "policy_observation_path": str(Path(policy_observation_path).expanduser()),
        "initial_frame_path": str(frame_path) if frame_path else None,
        "original_initial_frame_path": str(original_frame_path) if original_frame_path else None,
        "visual_profile": visual_profile,
        "wam_visual_profile_settings": visual_profile_settings,
        "source_policy_observation_visual_qa_path": str(source_visual_qa_path),
        "source_policy_observation_visual_qa_status": source_visual_qa.get("status"),
        "long_review_rollout_quality_gate_path": str(long_review_quality_gate_path),
        "long_review_rollout_quality_gate_status": long_review_quality_gate.get("status"),
        "clean_frame_reanchoring": _mapping(
            long_review_quality_gate.get("clean_frame_reanchoring")
        ),
        "original_source_policy_observation_visual_qa_path": str(original_source_visual_qa_path)
        if original_source_visual_qa_path
        else None,
        "image_model_render_remediation_enabled": image_model_render_remediation_enabled(),
        "image_model_render_remediation_applied": remediation_applied,
        "image_model_render_remediation_status": remediation_manifest.get("status")
        if remediation_manifest
        else "not_attempted",
        "image_model_render_remediation_manifest_path": str(remediation_manifest_path)
        if remediation_manifest_path
        else None,
        "synthetic_fallback_wam_launch_gate": synthetic_launch_gate,
        "semantic_visual_qa_source_paths": {
            "object_index": semantic_visual_evidence.get("object_index_path"),
            "eval_ready_task_grounding": semantic_visual_evidence.get(
                "eval_ready_task_grounding_path"
            ),
        },
        "loop_step_count": int(loop_step_count),
        "use_live_wam": bool(use_live_wam),
        "allow_structural_wam_fallback": bool(allow_structural_wam_fallback),
        "zip_entry_count": len(zip_entries),
        "zip_entries": zip_entries,
        "copied_blueprint_runtime_files": copied,
        "provider_bundle_kind": "unitree_groot_n17_sonic",
        "blockers": blockers,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": {
            "bundle_build_is_not_model_execution": True,
            "persistent_session_reuses_provider_instance_after_launch": True,
            "capture_truth": False,
            "geometry_truth": False,
            "collision_truth": False,
            "initial_policy_observation_capture_truth": synthetic_launch_gate.get(
                "capture_truth"
            ),
            "initial_policy_observation_geometry_truth": synthetic_launch_gate.get(
                "geometry_truth"
            ),
            "synthetic_fallback_initial_observation_used": synthetic_launch_gate.get(
                "synthetic_fallback_initial_observation_used"
            ),
            "synthetic_fallback_wam_launch_experiment_enabled": synthetic_launch_gate.get(
                "experimental_env_enabled"
            ),
            "provider_success_separate_from_visually_useful_rollout": True,
            "visually_useful_rollout": False,
            "image_model_enhanced_policy_observation_used": remediation_applied,
            "enhanced_policy_observation_is_not_capture_truth": remediation_applied,
            "enhanced_policy_observation_is_not_geometry_truth": remediation_applied,
            "enhanced_policy_observation_is_not_collision_truth": remediation_applied,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
        },
    }
    write_json(job / "persistent_session_provider_bundle_manifest.json", manifest)
    return manifest


def _job_dir(root: str | Path | None = None) -> Path:
    root_path = (
        Path(root).expanduser()
        if root
        else Path(
            _string(os.getenv(PERSISTENT_SESSION_JOB_ROOT_ENV))
            or Path.cwd() / "unitree_groot_n17_sonic_vast_persistent_session"
        )
    )
    job = root_path / utc_now_iso().replace(":", "").replace("+", "_").replace("-", "")
    ensure_dir(job)
    return job.resolve()


def _blocked_payload(
    *,
    generated_at: str,
    job_dir: Path,
    blockers: Sequence[str],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked",
        "policy_id": POLICY_ID,
        "selected_candidate_id": POLICY_ID,
        "job_dir": str(job_dir),
        "blockers": sorted({str(item) for item in blockers if str(item)}),
        "details": dict(details or {}),
        "persistent_provider_session_used": False,
        "unitree_groot_n17_sonic_model_executed": False,
        "unitree_groot_n17_sonic_policy_action_command_ran": False,
        "unitree_policy_action_command_ran": False,
        "policy_action_model_command_ran": False,
        "provider_output_replay_used": False,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "safety_validation_proven": False,
        "real_world_manipulation_success_proven": False,
    }


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _oscar_temporal_tokenizer_failure_detected(
    *,
    extraction_dir: Path | None,
    imported_payload: Mapping[str, Any],
    wam_calls: Sequence[Mapping[str, Any]],
) -> bool:
    payload_text = json.dumps(
        {
            "imported": dict(imported_payload),
            "wam_calls": [dict(row) for row in wam_calls],
        },
        sort_keys=True,
        default=str,
    ).lower()
    if (
        "kernel size can't be greater than actual input size" in payload_text
        and "worldsim/_src/tokenizers/wan2pt1.py" in payload_text
    ):
        return True
    if extraction_dir is None:
        return False
    for path in sorted(extraction_dir.rglob("wam_runtime_result.json")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        if (
            "kernel size can't be greater than actual input size" in text
            and "worldsim/_src/tokenizers/wan2pt1.py" in text
        ):
            return True
        if (
            "kernel size can't be greater than actual input size" in text
            and "calculated padded input size per channel" in text
        ):
            return True
    return False


def _runpod_persistent_session_wait_seconds(
    *,
    explicit_max_wait_seconds: int | None,
    timeout_seconds: float,
    loop_step_count: int,
) -> int:
    default_wait = _int_env(
        "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_MAX_WAIT_SECONDS",
        max(7200, int(timeout_seconds) * max(1, loop_step_count)),
    )
    requested_wait = int(explicit_max_wait_seconds or default_wait)
    entrypoint_timeout = _int_env(
        "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_ENTRYPOINT_TIMEOUT_SECONDS",
        int(timeout_seconds),
    )
    wrapper_watchdog = _int_env(
        "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_WRAPPER_WATCHDOG_SECONDS",
        entrypoint_timeout + 300,
    )
    wait_buffer = _int_env("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_WAIT_BUFFER_SECONDS", 300)
    remote_runtime_floor = max(entrypoint_timeout, wrapper_watchdog) + max(0, wait_buffer)
    return max(requested_wait, remote_runtime_floor)


def _write_runpod_live_wam_blocker_classification(
    *,
    job: Path,
    generated_at: str,
    poll_manifest: Mapping[str, Any],
    extraction_dir: Path | None = None,
    imported: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    imported_payload = _mapping(imported)
    last_nonterminal_output = _mapping(poll_manifest.get("last_nonterminal_output"))
    last_nonterminal_runtime_result = _mapping(last_nonterminal_output.get("runtime_result"))
    last_nonterminal_runtime_phase = _string(last_nonterminal_runtime_result.get("runtime_phase"))
    if not last_nonterminal_runtime_phase:
        nonterminal_zip = Path(
            _string(last_nonterminal_output.get("nonterminal_zip_path"))
        ).expanduser()
        if nonterminal_zip.is_file() and zipfile.is_zipfile(nonterminal_zip):
            try:
                with zipfile.ZipFile(nonterminal_zip) as archive:
                    output_name = "unitree_groot_n17_sonic_policy_provider_output.json"
                    if output_name in set(archive.namelist()):
                        nonterminal_payload = json.loads(
                            archive.read(output_name).decode("utf-8") or "{}"
                        )
                        last_nonterminal_runtime_phase = _string(
                            _mapping(nonterminal_payload).get("runtime_phase")
                        )
            except (OSError, ValueError, zipfile.BadZipFile):
                last_nonterminal_runtime_phase = ""
    has_output_zip = bool(
        poll_manifest.get("output_zip_present")
        or poll_manifest.get("provider_command_status") == "completed"
        or imported_payload
    )
    policy_calls = (
        _load_json_rows(sorted((extraction_dir / "policy_calls").glob("policy_call_*.json")))
        if extraction_dir is not None
        else []
    )
    wam_calls = (
        _load_json_rows(sorted((extraction_dir / "wam_calls").glob("wam_call_*.json")))
        if extraction_dir is not None
        else []
    )
    side_rows = (
        _jsonl_rows(extraction_dir / "robot_policy_wam_side_by_side_trace.jsonl")
        if extraction_dir is not None
        else []
    )
    generated_frames = (
        sorted((extraction_dir / "generated_next_observations").glob("*"))
        if extraction_dir is not None and (extraction_dir / "generated_next_observations").is_dir()
        else []
    )
    requested_policy_calls = _as_int(
        imported_payload.get("required_policy_call_count")
        or imported_payload.get("requested_loop_step_count"),
        default=2,
    )
    required_wam_transitions = _as_int(
        imported_payload.get("required_wam_transition_count"),
        default=max(0, requested_policy_calls - 1),
    )
    repeated_policy_calls = _as_int(imported_payload.get("repeated_policy_calls_count"))
    generated_next_observations = _as_int(imported_payload.get("generated_next_observation_count"))
    live_wam_successes = _as_int(imported_payload.get("live_wam_generation_success_count"))
    learned_wam_successes = _as_int(imported_payload.get("learned_wam_model_success_count"))
    all_blockers = [
        str(item)
        for item in (
            list(imported_payload.get("blockers") or [])
            + list(poll_manifest.get("provider_command_blockers") or [])
            + [
                blocker
                for row in wam_calls
                for blocker in list(_mapping(row).get("blockers") or [])
            ]
        )
        if str(item)
    ]
    blocker_text = " ".join(all_blockers).lower()
    first_policy_blocked = bool(policy_calls and policy_calls[0].get("status") != "completed")
    blocked_wam_calls = [row for row in wam_calls if row.get("status") != "completed"]
    materialization_blocked = any(
        _mapping(row.get("materialization")).get("status") == "blocked" for row in wam_calls
    ) or any("materializ" in blocker.lower() for blocker in all_blockers)
    oscar_temporal_tokenizer_blocked = _oscar_temporal_tokenizer_failure_detected(
        extraction_dir=extraction_dir,
        imported_payload=imported_payload,
        wam_calls=wam_calls,
    )
    entrypoint_execution = (
        _read_json(extraction_dir / "runpod_unitree_groot_sonic_entrypoint_execution.json")
        if extraction_dir is not None
        and (extraction_dir / "runpod_unitree_groot_sonic_entrypoint_execution.json").is_file()
        else {}
    )

    poll_status_running = (
        poll_manifest.get("status") == "running"
        or poll_manifest.get("provider_command_status") == "running"
        or bool(poll_manifest.get("continuing_spend_from_this_run"))
    )
    if poll_status_running:
        classified_blocker = "runpod_persistent_session_still_running"
    elif not has_output_zip and last_nonterminal_output:
        if poll_manifest.get("pod_status") == "RUNNING" and poll_manifest.get("teardown_performed"):
            classified_blocker = (
                "runpod_remote_runtime_still_running_after_heartbeat_until_local_timeout"
            )
        elif poll_manifest.get("pod_status") == "not_found" and last_nonterminal_runtime_phase:
            if last_nonterminal_runtime_phase in {
                "gr00t_model_snapshot_completed",
                "gr00t_policy_server_process_starting",
            }:
                classified_blocker = (
                    "runpod_pod_disappeared_after_gr00t_model_snapshot_before_policy_server_ready"
                )
            elif last_nonterminal_runtime_phase in {
                "gr00t_policy_server_process_started",
                "gr00t_policy_server_waiting_for_listen",
            }:
                classified_blocker = "runpod_pod_disappeared_during_gr00t_policy_server_process_start_after_heartbeat"
            elif last_nonterminal_runtime_phase == "gr00t_uv_sync_started":
                classified_blocker = "runpod_pod_disappeared_during_gr00t_uv_sync_after_heartbeat"
            elif (
                last_nonterminal_runtime_phase == "gr00t_system_python_minimal_deps_install_started"
            ):
                classified_blocker = (
                    "runpod_pod_disappeared_during_gr00t_system_python_minimal_deps_install"
                    "_after_heartbeat"
                )
            elif last_nonterminal_runtime_phase.startswith("gr00t_system_python_"):
                classified_blocker = (
                    "runpod_pod_disappeared_during_gr00t_system_python_bootstrap_after_heartbeat"
                )
            elif last_nonterminal_runtime_phase.startswith(("bootstrap_", "gr00t_")):
                classified_blocker = (
                    "runpod_pod_disappeared_during_policy_server_bootstrap_after_heartbeat"
                )
            elif last_nonterminal_runtime_phase == "wam_infer_started":
                classified_blocker = "runpod_pod_disappeared_during_live_wam_after_heartbeat"
            else:
                classified_blocker = "runpod_pod_disappeared_after_nonterminal_heartbeat"
        else:
            classified_blocker = "runpod_terminal_output_upload_failed_after_remote_heartbeat"
    elif not has_output_zip or not imported_payload:
        classified_blocker = "runpod_wrapper_or_upload_watchdog_no_valid_provider_artifact"
    elif (
        "persistent_session_entrypoint_exited_without_runtime_result" in all_blockers
        or entrypoint_execution.get("timed_out") is True
    ):
        classified_blocker = "policy_runtime_bootstrap_timeout"
    elif first_policy_blocked or repeated_policy_calls < 1:
        classified_blocker = "policy_initial_call_blocked"
    elif required_wam_transitions and not wam_calls:
        classified_blocker = "live_wam_runtime_not_invoked_after_policy"
    elif required_wam_transitions and generated_next_observations < required_wam_transitions:
        if oscar_temporal_tokenizer_blocked:
            classified_blocker = "oscar_wam_temporal_window_too_short"
        elif materialization_blocked:
            classified_blocker = "wam_frame_materialization_blocked"
        elif (
            blocked_wam_calls
            or live_wam_successes < required_wam_transitions
            or learned_wam_successes < required_wam_transitions
            or "oscar" in blocker_text
            or "live_infer" in blocker_text
        ):
            classified_blocker = "live_wam_runtime_blocked"
        else:
            classified_blocker = "wam_frame_materialization_blocked"
    elif repeated_policy_calls < requested_policy_calls:
        classified_blocker = "policy_requery_blocked"
    elif imported_payload.get("status") != "completed":
        classified_blocker = "persistent_session_provider_output_blocked"
    else:
        classified_blocker = "none"

    status = "completed" if classified_blocker == "none" else "blocked"
    classification = {
        "schema_version": "runpod_live_wam_blocker_classification.v1",
        "generated_at": generated_at,
        "status": status,
        "classified_blocker": classified_blocker,
        "provider": "runpod",
        "job_dir": str(job),
        "runpod_poll_manifest_path": str(
            job / "runpod_persistent_session_run" / "runpod_wam_async_poll_manifest.json"
        ),
        "imported_provider_output_dir": str(extraction_dir) if extraction_dir is not None else None,
        "evidence": {
            "output_zip_present": has_output_zip,
            "runtime_result_status": poll_manifest.get("runtime_result_status"),
            "entrypoint_execution_status": entrypoint_execution.get("status"),
            "entrypoint_timed_out": entrypoint_execution.get("timed_out"),
            "entrypoint_timeout_seconds": entrypoint_execution.get("timeout_seconds"),
            "last_nonterminal_runtime_result_status": last_nonterminal_output.get(
                "runtime_result_status"
            ),
            "last_nonterminal_runtime_phase": last_nonterminal_runtime_phase or None,
            "last_nonterminal_zip_path": last_nonterminal_output.get("nonterminal_zip_path"),
            "provider_command_status": poll_manifest.get("provider_command_status"),
            "pod_status": poll_manifest.get("pod_status"),
            "teardown_performed": bool(poll_manifest.get("teardown_performed")),
            "continuing_spend_from_this_run": bool(
                poll_manifest.get("continuing_spend_from_this_run")
            ),
            "requested_policy_call_count": requested_policy_calls,
            "required_wam_transition_count": required_wam_transitions,
            "policy_call_artifact_count": len(policy_calls),
            "wam_call_artifact_count": len(wam_calls),
            "side_by_side_trace_row_count": len(side_rows),
            "generated_frame_artifact_count": len(generated_frames),
            "repeated_policy_calls_count": repeated_policy_calls,
            "generated_next_observation_count": generated_next_observations,
            "live_wam_generation_success_count": live_wam_successes,
            "learned_wam_model_success_count": learned_wam_successes,
            "policy_observes_wam_generated_next_observation": bool(
                imported_payload.get("policy_observes_wam_generated_next_observation")
            ),
            "oscar_temporal_tokenizer_blocked": oscar_temporal_tokenizer_blocked,
            "blockers": sorted(set(all_blockers)),
        },
        "claim_boundary": {
            "classification_is_runtime_diagnostic_not_physical_robot_readiness": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "real_world_manipulation_success_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(job / "runpod_live_wam_blocker_classification.json", classification)
    return classification


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def _load_json_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        try:
            value = _read_json(path)
        except Exception:
            continue
        rows.append(value)
    return rows


def _action_summary(action: Mapping[str, Any]) -> dict[str, Any]:
    chunk = action.get("action_chunk")
    return {
        "action_type": action.get("action_type"),
        "action_chunk_present": isinstance(chunk, Sequence)
        and not isinstance(chunk, (str, bytes, bytearray)),
        "action_chunk_length": len(chunk)
        if isinstance(chunk, Sequence) and not isinstance(chunk, (str, bytes, bytearray))
        else None,
        "source_action_key": action.get("source_action_key"),
        "control_fields": action.get("unitree_g1_sonic_control_fields"),
    }


def _concat_file_line(path: Path) -> str:
    return "file '{}'\n".format(str(path).replace("'", "'\\''"))


def _write_review_video(
    *,
    job: Path,
    extraction_dir: Path,
    generated_at: str,
    fps: float = 2.0,
    structural_fallback_used: bool = False,
) -> dict[str, Any]:
    review_dir = job / "review_video"
    ensure_dir(review_dir)
    live_rollout_videos = sorted(
        path.resolve()
        for path in (extraction_dir / "wam_worker_steps").glob("step_*/oscar_runtime_output/*.mp4")
        if path.is_file()
    )
    live_rollout_videos.extend(
        path.resolve()
        for path in (extraction_dir / "wam_worker_steps").glob(
            "step_*/oscar_wam_worker_bundle/provider_runtime/*.mp4"
        )
        if path.is_file()
    )
    live_rollout_videos = sorted(dict.fromkeys(live_rollout_videos))
    output_path = review_dir / (
        "persistent_policy_wam_live_rollout_review.mp4"
        if live_rollout_videos
        else "persistent_policy_wam_review.mp4"
    )
    initial_frame = job / "provider_bundle" / "provider_runtime" / "initial_policy_frame.png"
    frames = [initial_frame] if initial_frame.is_file() else []
    frames.extend(sorted((extraction_dir / "generated_next_observations").glob("*.jpg")))
    frames = [path.resolve() for path in frames if path.is_file()]
    concat_path = review_dir / (
        "persistent_policy_wam_live_rollout_frames.ffconcat"
        if live_rollout_videos
        else "persistent_policy_wam_review_frames.ffconcat"
    )
    status = {
        "schema_version": "persistent_policy_wam_video_review_status.v1",
        "generated_at": generated_at,
        "status": "blocked",
        "review_video_path": str(output_path),
        "frame_count": len(frames),
        "live_rollout_video_count": len(live_rollout_videos),
        "review_video_source": "live_wam_generated_rollout_videos"
        if live_rollout_videos
        else "still_policy_and_next_observation_frames",
        "fps_requested": fps,
        "ffmpeg_command_ran": False,
        "ffprobe_command_ran": False,
        "ffprobe_metadata": {},
        "blockers": [],
        "claim_boundary": {
            "video_is_review_artifact_not_task_success_proof": True,
            "structural_fallback_video_is_not_live_wam_model_proof": bool(
                structural_fallback_used or not live_rollout_videos
            ),
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "real_world_manipulation_success_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    if live_rollout_videos:
        with concat_path.open("w", encoding="utf-8") as handle:
            handle.write("ffconcat version 1.0\n")
            for video in live_rollout_videos:
                handle.write(_concat_file_line(video))
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        if shutil.which("ffmpeg") is None:
            status["blockers"] = ["ffmpeg_not_available_for_review_video"]
            write_json(job / "video_review_status.json", status)
            return status
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        status["ffmpeg_command_ran"] = True
        status["ffmpeg_returncode"] = completed.returncode
        status["ffmpeg_stdout_size_bytes"] = len(completed.stdout or "")
        status["ffmpeg_stderr_size_bytes"] = len(completed.stderr or "")
        if completed.returncode != 0 or not output_path.is_file():
            status["blockers"] = ["ffmpeg_live_wam_rollout_review_video_failed"]
            write_json(job / "video_review_status.json", status)
            return status
        if shutil.which("ffprobe") is not None:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
                    "-show_entries",
                    "format=duration,size",
                    "-of",
                    "json",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            status["ffprobe_command_ran"] = True
            status["ffprobe_returncode"] = probe.returncode
            if probe.returncode == 0:
                try:
                    parsed = json.loads(probe.stdout or "{}")
                except json.JSONDecodeError:
                    parsed = {}
                status["ffprobe_metadata"] = parsed if isinstance(parsed, Mapping) else {}
        status["status"] = "completed"
        status["blockers"] = []
        write_json(job / "video_review_status.json", status)
        return status
    if len(frames) < 2:
        status["blockers"] = ["not_enough_frames_for_review_video"]
        write_json(job / "video_review_status.json", status)
        return status
    if shutil.which("ffmpeg") is None:
        status["blockers"] = ["ffmpeg_not_available_for_review_video"]
        write_json(job / "video_review_status.json", status)
        return status
    duration = 1.0 / float(fps)
    with concat_path.open("w", encoding="utf-8") as handle:
        handle.write("ffconcat version 1.0\n")
        for frame in frames:
            handle.write(_concat_file_line(frame))
            handle.write(f"duration {duration:.6f}\n")
        handle.write(_concat_file_line(frames[-1]))
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-movflags",
        "+faststart",
        "-r",
        str(fps),
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    status["ffmpeg_command_ran"] = True
    status["ffmpeg_returncode"] = completed.returncode
    status["ffmpeg_stdout_size_bytes"] = len(completed.stdout or "")
    status["ffmpeg_stderr_size_bytes"] = len(completed.stderr or "")
    if completed.returncode != 0 or not output_path.is_file():
        status["blockers"] = ["ffmpeg_review_video_failed"]
        write_json(job / "video_review_status.json", status)
        return status
    if shutil.which("ffprobe") is not None:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
                "-show_entries",
                "format=duration,size",
                "-of",
                "json",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        status["ffprobe_command_ran"] = True
        status["ffprobe_returncode"] = probe.returncode
        if probe.returncode == 0:
            try:
                parsed = json.loads(probe.stdout or "{}")
            except json.JSONDecodeError:
                parsed = {}
            status["ffprobe_metadata"] = parsed if isinstance(parsed, Mapping) else {}
    status["status"] = "completed"
    status["blockers"] = []
    write_json(job / "video_review_status.json", status)
    return status


def _postprocess_imported_persistent_session_artifacts(
    *,
    job: Path,
    extraction_dir: Path,
    imported: Mapping[str, Any],
    generated_at: str,
    policy_observation_path: str | Path,
    vast_result: Mapping[str, Any],
    vast_run_dir: Path,
) -> dict[str, Any]:
    policy_calls = _load_json_rows(
        sorted((extraction_dir / "policy_calls").glob("policy_call_*.json"))
    )
    wam_calls = _load_json_rows(sorted((extraction_dir / "wam_calls").glob("wam_call_*.json")))
    trace_rows = _jsonl_rows(extraction_dir / "robot_policy_wam_loop_trace.jsonl")
    side_rows = _jsonl_rows(extraction_dir / "robot_policy_wam_side_by_side_trace.jsonl")
    wam_rows = _jsonl_rows(extraction_dir / "wam_generated_next_observations.jsonl")
    for filename in (
        "robot_policy_wam_side_by_side_trace.jsonl",
        "robot_policy_wam_side_by_side_trace.html",
        "wam_generated_next_observations.jsonl",
        "robot_policy_wam_loop_trace.jsonl",
    ):
        source = extraction_dir / filename
        if source.is_file():
            shutil.copy2(source, job / filename)
    first_policy = policy_calls[0] if policy_calls else {}
    first_action = _mapping(first_policy.get("action"))
    policy_completed_count = sum(1 for row in policy_calls if row.get("status") == "completed")
    wam_completed_count = sum(1 for row in wam_rows if row.get("status") == "completed")
    structural_wam_count = sum(1 for row in wam_rows if row.get("structural_fallback_used") is True)
    live_wam_count = int(imported.get("live_wam_generation_success_count") or 0)
    learned_wam_count = int(imported.get("learned_wam_model_success_count") or 0)
    write_json(
        job / "policy_action_model_command_discovery.json",
        {
            "schema_version": "policy_action_model_command_discovery.v1",
            "generated_at": generated_at,
            "status": "completed" if policy_completed_count else "blocked",
            "selected_candidate_id": POLICY_ID,
            "candidate_checkpoint": "LucaFrat/groot-bs16",
            "candidate_priority": "default_experimental_unitree_g1_sonic",
            "trusted_for_production": False,
            "policy_server_client_used": True,
            "provider_output_replay_used": False,
            "raw_credentials_written_to_artifacts": False,
        },
    )
    write_json(
        job / "policy_action_model_command_execution.json",
        {
            "schema_version": "policy_action_model_command_execution.v1",
            "generated_at": generated_at,
            "status": "completed" if policy_completed_count else "blocked",
            "policy_call_count": len(policy_calls),
            "completed_policy_call_count": policy_completed_count,
            "persistent_policy_worker_command_source": _mapping(
                first_policy.get("worker_response_redacted")
            ).get("persistent_policy_worker_command_source"),
            "policy_server_bootstrap_status": _mapping(imported.get("policy_server_bootstrap")).get(
                "status"
            ),
            "provider_instance_reused_for_policy_and_wam_loop": bool(
                imported.get("provider_instance_reused_for_policy_and_wam_loop")
            ),
            "provider_output_replay_used": False,
            "raw_credentials_written_to_artifacts": False,
        },
    )
    write_json(
        job / "policy_action_model_command_output.json",
        {
            "schema_version": "policy_action_model_command_output.v1",
            "generated_at": generated_at,
            "status": "completed" if first_action else "blocked",
            "selected_candidate_id": POLICY_ID,
            "policy_calls_dir": str(extraction_dir / "policy_calls"),
            "first_action_summary": _action_summary(first_action),
            "full_action_payloads_are_in_policy_call_artifacts": True,
            "provider_output_replay_used": False,
            "raw_credentials_written_to_artifacts": False,
        },
    )
    write_json(
        job / "wam_generation_command_discovery.json",
        {
            "schema_version": "wam_generation_command_discovery.v1",
            "generated_at": generated_at,
            "status": "completed" if wam_completed_count else "blocked",
            "wam_evaluator_backend": "persistent_structural_wam_fallback"
            if structural_wam_count
            else "persistent_oscar_wam_worker",
            "live_wam_generation_success_count": live_wam_count,
            "learned_wam_model_success_count": learned_wam_count,
            "structural_fallback_count": structural_wam_count,
            "live_wam_model_configured": live_wam_count > 0,
            "raw_credentials_written_to_artifacts": False,
        },
    )
    write_json(
        job / "wam_generation_command_execution.json",
        {
            "schema_version": "wam_generation_command_execution.v1",
            "generated_at": generated_at,
            "status": "completed" if wam_completed_count else "blocked",
            "wam_call_count": len(wam_calls) or len(wam_rows),
            "completed_wam_call_count": wam_completed_count,
            "action_conditioned_generation_ran": bool(wam_completed_count),
            "live_wam_generation_command_ran": live_wam_count > 0,
            "structural_fallback_used": structural_wam_count > 0,
            "raw_credentials_written_to_artifacts": False,
        },
    )
    write_json(
        job / "wam_generation_command_output.json",
        {
            "schema_version": "wam_generation_command_output.v1",
            "generated_at": generated_at,
            "status": "completed" if wam_completed_count else "blocked",
            "wam_generated_next_observations_jsonl": str(
                job / "wam_generated_next_observations.jsonl"
            ),
            "generated_next_observations_dir": str(extraction_dir / "generated_next_observations"),
            "generated_next_observation_count": wam_completed_count,
            "live_wam_generation_success_count": live_wam_count,
            "learned_wam_model_success_count": learned_wam_count,
            "structural_fallback_used": structural_wam_count > 0,
            "raw_credentials_written_to_artifacts": False,
        },
    )
    write_json(
        job / "robot_policy_wam_loop_manifest.json",
        {
            "schema_version": "robot_policy_wam_loop_manifest.v1",
            "generated_at": generated_at,
            "status": imported.get("status"),
            "policy_observation_path": str(Path(policy_observation_path).expanduser()),
            "persistent_provider_session_used": bool(
                imported.get("persistent_provider_session_used")
            ),
            "provider_instance_reused_for_policy_and_wam_loop": bool(
                imported.get("provider_instance_reused_for_policy_and_wam_loop")
            ),
            "repeated_policy_calls_count": int(imported.get("repeated_policy_calls_count") or 0),
            "generated_next_observation_count": int(
                imported.get("generated_next_observation_count") or 0
            ),
            "policy_observes_wam_generated_next_observation": bool(
                imported.get("policy_observes_wam_generated_next_observation")
            ),
            "trace_row_count": len(trace_rows),
            "side_by_side_trace_row_count": len(side_rows),
            "policy_calls_dir": str(extraction_dir / "policy_calls"),
            "wam_calls_dir": str(extraction_dir / "wam_calls"),
            "provider_output_replay_used": False,
            "raw_credentials_written_to_artifacts": False,
        },
    )
    success_proven = bool(
        live_wam_count > 0
        and learned_wam_count > 0
        and imported.get("manipulation_success_evaluator_result") == "success"
    )
    if success_proven:
        success_reason = "A live evaluator reported sink-handle success."
    elif live_wam_count > 0:
        success_reason = (
            "The loop completed with live learned WAM generations, but no task-success "
            "evaluator or physics state proved a sink-handle state transition."
        )
    elif structural_wam_count > 0:
        success_reason = (
            "The loop completed with structural WAM fallback only; no live learned WAM or "
            "physics state proved a sink-handle state transition."
        )
    else:
        success_reason = (
            "The loop did not produce a completed WAM generation or physics state proving a "
            "sink-handle state transition."
        )
    judge = {
        "schema_version": "manipulation_success_evaluator_results.v1",
        "generated_at": generated_at,
        "status": "completed",
        "question": "Did the sink handle end up turned on?",
        "answer": "not_proven" if not success_proven else "yes",
        "did_sink_handle_end_up_turned_on": bool(success_proven),
        "sink_handle_turned_on_proven": bool(success_proven),
        "success_proof_separate_from_structural_loop_proof": True,
        "structural_loop_completed": imported.get("status") == "completed",
        "live_wam_generation_success_count": live_wam_count,
        "learned_wam_model_success_count": learned_wam_count,
        "structural_fallback_used": structural_wam_count > 0,
        "reason": success_reason,
        "raw_credentials_written_to_artifacts": False,
    }
    write_json(job / "manipulation_success_evaluator_results.json", judge)
    video_status = _write_review_video(
        job=job,
        extraction_dir=extraction_dir,
        generated_at=generated_at,
        fps=float(os.getenv("BLUEPRINT_PERSISTENT_SESSION_REVIEW_FPS", "2.0")),
        structural_fallback_used=structural_wam_count > 0,
    )
    try:
        policy_observation = _load_policy_observation(policy_observation_path)
    except Exception:
        policy_observation = {}
    visual_profile_settings = _current_wam_visual_profile_settings()
    source_frame = job / "provider_bundle" / "provider_runtime" / "initial_policy_frame.png"
    generated_frame_paths = sorted((extraction_dir / "generated_next_observations").glob("*.jpg"))
    visual_quality_report = write_persistent_wam_visual_quality_artifacts(
        job_dir=job,
        generated_at=generated_at,
        source_frame_path=source_frame if source_frame.is_file() else None,
        generated_frame_paths=generated_frame_paths,
        review_video_path=_mapping(video_status).get("review_video_path"),
        video_status=video_status,
        visual_profile=str(visual_profile_settings["visual_profile"]),
        requested_settings=visual_profile_settings,
        provider_status=_string(imported.get("status")) or None,
        live_wam_generation_success_count=live_wam_count,
        learned_wam_model_success_count=learned_wam_count,
        structural_fallback_used=structural_wam_count > 0,
        target_object_id=_string(policy_observation.get("target_object_id")) or None,
        task_id=_string(policy_observation.get("task_id")) or None,
    )
    claim_boundary = {
        "schema_version": "persistent_policy_wam_claim_boundary.v1",
        "generated_at": generated_at,
        "simulator_generated_world_proof_only": True,
        "capture_truth": False,
        "geometry_truth": False,
        "collision_truth": False,
        "structural_loop_proof_completed": imported.get("status") == "completed",
        "success_proof_completed": success_proven,
        "provider_success": imported.get("status") == "completed",
        "provider_success_separate_from_visually_useful_rollout": True,
        "live_wam_generation_success": live_wam_count > 0,
        "learned_wam_model_success": learned_wam_count > 0,
        "visually_useful_rollout": bool(visual_quality_report.get("visual_success")),
        "visual_success": bool(visual_quality_report.get("visual_success")),
        "live_wam_generation_success_can_coexist_with_visually_useful_rollout_false": True,
        "valid_mp4_or_provider_completed_is_not_visual_success": True,
        "wam_rollout_visual_quality_report": str(job / "wam_rollout_visual_quality_report.json"),
        "local_structural_wam_generator_is_not_live_oscar_or_cosmos_model": structural_wam_count
        > 0,
        "frame_copy_placeholder_until_live_wam_model_configured": structural_wam_count > 0,
        "wam_evaluator_is_not_robot_policy": True,
        "provider_output_replay_used": False,
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "safety_validation_proven": False,
        "real_world_manipulation_success_proven": False,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(job / "claim_boundary.json", claim_boundary)
    if not success_proven:
        wam_generation_label = (
            "live_wam_success_not_task_success_proof"
            if live_wam_count > 0
            else "structural_wam_fallback_only"
            if structural_wam_count
            else "wam_generation_missing"
        )
        labels = [
            "task_success_not_proven",
            "live_wam_not_run" if live_wam_count == 0 else "live_wam_success_not_judged",
            wam_generation_label,
            "physics_contact_not_validated",
        ]
        if not visual_quality_report.get("visual_success"):
            labels.append("wam_rollout_visual_quality_failed")
        visual_blockers = {str(item) for item in visual_quality_report.get("blockers") or []}
        if (
            "source_policy_observation_visual_qa_failed_for_review_quality" in visual_blockers
            or any(item.startswith("source_policy_observation_") for item in visual_blockers)
        ):
            labels.append("source_policy_observation_visual_qa_failed")
        if "autoregressive_chain_visual_drift_or_quality_blocked_long_rollout" in visual_blockers:
            labels.append("autoregressive_chain_visual_drift_or_quality_blocked")
        write_json(
            job / "failure_labels.json",
            {
                "schema_version": "persistent_policy_wam_failure_labels.v1",
                "generated_at": generated_at,
                "status": "completed",
                "labels": sorted(set(labels)),
                "raw_credentials_written_to_artifacts": False,
            },
        )
    return {
        "schema_version": "persistent_session_postprocess_artifacts.v1",
        "generated_at": generated_at,
        "status": "completed",
        "policy_action_model_command_discovery": str(
            job / "policy_action_model_command_discovery.json"
        ),
        "policy_action_model_command_execution": str(
            job / "policy_action_model_command_execution.json"
        ),
        "policy_action_model_command_output": str(job / "policy_action_model_command_output.json"),
        "wam_generation_command_discovery": str(job / "wam_generation_command_discovery.json"),
        "wam_generation_command_execution": str(job / "wam_generation_command_execution.json"),
        "wam_generation_command_output": str(job / "wam_generation_command_output.json"),
        "robot_policy_wam_loop_manifest": str(job / "robot_policy_wam_loop_manifest.json"),
        "manipulation_success_evaluator_results": str(
            job / "manipulation_success_evaluator_results.json"
        ),
        "video_review_status": str(job / "video_review_status.json"),
        "review_video_path": _mapping(video_status).get("review_video_path"),
        "source_policy_observation_visual_qa": str(
            job / "source_policy_observation_visual_qa.json"
        ),
        "wam_rollout_visual_quality_report": str(job / "wam_rollout_visual_quality_report.json"),
        "wam_rollout_contact_sheet": str(job / "wam_rollout_contact_sheet.jpg")
        if (job / "wam_rollout_contact_sheet.jpg").is_file()
        else None,
        "wam_rollout_frame_stats": str(job / "wam_rollout_frame_stats.jsonl"),
        "wam_rollout_visual_success": bool(visual_quality_report.get("visual_success")),
        "claim_boundary": str(job / "claim_boundary.json"),
        "failure_labels": str(job / "failure_labels.json")
        if (job / "failure_labels.json").is_file()
        else None,
        "vast_provider_adapter_result_path": str(
            vast_run_dir / "vast_provider_adapter_result.json"
        ),
        "estimated_cost_usd": vast_result.get("estimated_cost_usd"),
        "raw_credentials_written_to_artifacts": False,
    }


def run_persistent_session(
    *,
    policy_observation_path: str | Path,
    job_dir: str | Path | None = None,
    loop_step_count: int = 12,
    task_prompt: str | None = None,
    timeout_seconds: float = 3600.0,
    use_live_wam: bool = True,
    allow_structural_wam_fallback: bool | None = None,
) -> tuple[dict[str, Any], int]:
    generated_at = utc_now_iso()
    job = _job_dir(job_dir)
    requested_loop_step_count = max(1, int(loop_step_count))
    allow_fallback = (
        _truthy(os.getenv(PERSISTENT_SESSION_ALLOW_STRUCTURAL_WAM_FALLBACK_ENV))
        if allow_structural_wam_fallback is None
        else bool(allow_structural_wam_fallback)
    )
    inner_policy_command = (
        _string(os.getenv(INNER_POLICY_COMMAND_ENV)) or DEFAULT_INNER_POLICY_COMMAND
    )
    previous_policy_command = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND")
    previous_persistent_inner_policy_command = os.environ.get(
        PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV
    )
    previous_vast_inner_policy_command = os.environ.get(INNER_POLICY_COMMAND_ENV)
    os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"] = inner_policy_command
    os.environ[PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV] = inner_policy_command
    os.environ[INNER_POLICY_COMMAND_ENV] = inner_policy_command
    try:
        bundle = build_persistent_session_provider_bundle(
            job_dir=job / "provider_bundle",
            policy_observation_path=policy_observation_path,
            loop_step_count=requested_loop_step_count,
            task_prompt=task_prompt,
            timeout_seconds=timeout_seconds,
            use_live_wam=use_live_wam,
            allow_structural_wam_fallback=allow_fallback,
            generated_at=generated_at,
        )
        if bundle.get("status") != "bundle_ready":
            output = _blocked_payload(
                generated_at=generated_at,
                job_dir=job,
                blockers=bundle.get("blockers") or ["persistent_session_provider_bundle_blocked"],
                details={
                    "bundle_manifest_path": str(
                        job / "provider_bundle" / "persistent_session_provider_bundle_manifest.json"
                    )
                },
            )
            write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
            return output, 2
        bundle_path = Path(str(bundle["bundle_path"])).expanduser().resolve()
        staging = stage_wam_provider_bundle_object_store(
            job_dir=job / "object_store_staging",
            bundle_path=bundle_path,
            key_prefix=_string(os.getenv(OBJECT_STORE_KEY_PREFIX_ENV))
            or DEFAULT_OBJECT_STORE_KEY_PREFIX,
            expiration_seconds=_int_env(
                "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIGNED_URL_SECONDS", 21600
            ),
            generated_at=generated_at,
        )
        if staging.get("status") != "completed":
            output = _blocked_payload(
                generated_at=generated_at,
                job_dir=job,
                blockers=staging.get("blockers")
                or ["persistent_session_object_store_staging_blocked"],
                details={
                    "object_store_staging_manifest_path": str(
                        job
                        / "object_store_staging"
                        / "wam_provider_object_store_staging_manifest.json"
                    )
                },
            )
            write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
            return output, 2
        staging_dir = job / "object_store_staging"
        bundle_url = (staging_dir / "provider_bundle_url.txt").read_text(encoding="utf-8").strip()
        output_put_url = (
            (staging_dir / "provider_output_put_url.txt").read_text(encoding="utf-8").strip()
        )
        output_get_url = (
            (staging_dir / "provider_output_get_url.txt").read_text(encoding="utf-8").strip()
        )
        excluded_machine_ids = _machine_ids_from_env(EXCLUDED_MACHINE_ID_ENVS)
        allowed_machine_ids = _machine_ids_from_env(ALLOWED_MACHINE_ID_ENVS)
        machine_avoidlist_path = job / "vast_machine_avoidlist.json"
        if excluded_machine_ids:
            write_json(
                machine_avoidlist_path,
                {
                    "schema_version": "vast_machine_avoidlist.v1",
                    "generated_at": generated_at,
                    "status": "loaded_from_env",
                    "machine_ids": sorted(excluded_machine_ids),
                    "raw_secret_values_recorded": False,
                },
            )

        def run_remote_attempt(
            run_dir: Path, attempt_allowed_machine_ids: Sequence[int]
        ) -> tuple[dict[str, Any], Path]:
            output_zip = run_dir / "vast_provider_runtime_output.zip"
            result = run_vast_provider_adapter(
                job_dir=run_dir,
                mode="live-startup-probe",
                allow_vast_api_call=_truthy(os.getenv("BLUEPRINT_ALLOW_VAST_API_CALLS")),
                allow_instance_launch=_truthy(os.getenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH")),
                max_hourly_rate=_float_env(
                    "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_MAX_HOURLY_RATE", 0.60
                ),
                target_spend_usd=_float_env(
                    "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_TARGET_SPEND_USD", 3.0
                ),
                hard_cap_usd=_float_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_HARD_CAP_USD", 3.0),
                max_live_minutes=_int_env(
                    "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_MAX_LIVE_MINUTES", 55
                ),
                session_max_live_minutes=_int_env(
                    "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_SESSION_MAX_LIVE_MINUTES", 420
                ),
                public_image=(
                    _string(os.getenv(PERSISTENT_SESSION_PUBLIC_IMAGE_ENV))
                    or _string(os.getenv("BLUEPRINT_VAST_WAM_PUBLIC_IMAGE"))
                    or _string(os.getenv(UNITREE_PUBLIC_IMAGE_ENV))
                    or DEFAULT_WAM_PUBLIC_IMAGE
                    or DEFAULT_PUBLIC_CUDA_IMAGE
                ),
                provider_bundle=bundle_path,
                provider_bundle_url=bundle_url,
                provider_output_put_url=output_put_url,
                provider_output_get_url=output_get_url,
                provider_runtime_output_zip=output_zip,
                enable_blueprint_bundle=True,
                provider_bundle_kind="unitree_groot_n17_sonic",
                vast_launch_mode=_string(os.getenv(VAST_LAUNCH_MODE_ENV)) or "ssh_direct",
                ngc_image_login_mode=os.getenv(VAST_IMAGE_LOGIN_MODE_ENV),
                disk_gb=_int_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_DISK_GB", 120),
                min_gpu_ram_mb=_int_env(
                    "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_MIN_GPU_RAM_MB", 48000
                ),
                poll_interval_seconds=_int_env(
                    "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_POLL_SECONDS", 15
                ),
                startup_timeout_seconds=_int_env(
                    "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_STARTUP_TIMEOUT_SECONDS", 1800
                ),
                machine_avoidlist_path=machine_avoidlist_path,
                allowed_machine_ids=attempt_allowed_machine_ids,
                verify_staging_urls=True,
            )
            return result, output_zip

        run_dir = job / "vast_persistent_session_run"
        effective_run_dir = run_dir
        vast_result, output_zip = run_remote_attempt(run_dir, allowed_machine_ids)
        fallback_result: dict[str, Any] | None = None
        if (
            allowed_machine_ids
            and _truthy(os.getenv(ALLOW_UNPINNED_FALLBACK_ENV))
            and vast_result.get("status") != "completed"
            and "no_vast_offer_matching_allowed_machine_ids"
            in {str(item) for item in (vast_result.get("blockers") or [])}
        ):
            effective_run_dir = job / "vast_persistent_session_run_unpinned_fallback"
            fallback_result, output_zip = run_remote_attempt(effective_run_dir, [])
            vast_result = fallback_result
        if vast_result.get("status") != "completed" or not output_zip.is_file():
            output = _blocked_payload(
                generated_at=generated_at,
                job_dir=job,
                blockers=vast_result.get("blockers")
                or ["persistent_session_vast_provider_blocked"],
                details={
                    "vast_provider_adapter_result_path": str(
                        effective_run_dir / "vast_provider_adapter_result.json"
                    ),
                    "vast_teardown_manifest_path": str(
                        effective_run_dir / "vast_teardown_manifest.json"
                    ),
                    "fallback_vast_provider_adapter_result_path": str(
                        job
                        / "vast_persistent_session_run_unpinned_fallback"
                        / "vast_provider_adapter_result.json"
                    )
                    if fallback_result is not None
                    else None,
                },
            )
            write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
            return output, 2
        extraction_dir = job / "imported_persistent_session_output"
        ensure_dir(extraction_dir)
        with zipfile.ZipFile(output_zip) as archive:
            archive.extractall(extraction_dir)
        imported_path = (
            extraction_dir / "unitree_groot_n17_sonic_wam_persistent_session_output.json"
        )
        if not imported_path.is_file():
            imported_path = extraction_dir / "unitree_groot_n17_sonic_policy_provider_output.json"
        imported = _read_json(imported_path) if imported_path.is_file() else {}
        postprocess = _postprocess_imported_persistent_session_artifacts(
            job=job,
            extraction_dir=extraction_dir,
            imported=imported,
            generated_at=generated_at,
            policy_observation_path=policy_observation_path,
            vast_result=vast_result,
            vast_run_dir=effective_run_dir,
        )
        completed = imported.get("status") == "completed"
        output = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "completed" if completed else "blocked",
            "policy_id": POLICY_ID,
            "selected_candidate_id": POLICY_ID,
            "job_dir": str(job),
            "persistent_provider_session_used": bool(
                imported.get("persistent_provider_session_used")
            ),
            "provider_instance_reused_for_policy_and_wam_loop": bool(
                imported.get("provider_instance_reused_for_policy_and_wam_loop")
            ),
            "repeated_policy_calls_count": int(imported.get("repeated_policy_calls_count") or 0),
            "generated_next_observation_count": int(
                imported.get("generated_next_observation_count") or 0
            ),
            "live_wam_generation_success_count": int(
                imported.get("live_wam_generation_success_count") or 0
            ),
            "learned_wam_model_success_count": int(
                imported.get("learned_wam_model_success_count") or 0
            ),
            "unitree_groot_n17_sonic_model_executed": bool(
                imported.get("unitree_groot_n17_sonic_model_executed")
            ),
            "unitree_groot_n17_sonic_policy_action_command_ran": bool(
                imported.get("unitree_groot_n17_sonic_policy_action_command_ran")
            ),
            "unitree_policy_action_command_ran": bool(
                imported.get("unitree_policy_action_command_ran")
            ),
            "policy_action_model_command_ran": bool(
                imported.get("policy_action_model_command_ran")
            ),
            "provider_output_replay_used": bool(imported.get("provider_output_replay_used")),
            "blockers": []
            if completed
            else imported.get("blockers") or ["persistent_session_provider_output_blocked"],
            "imported_provider_output_dir": str(extraction_dir),
            "imported_provider_output_path": str(imported_path)
            if imported_path.is_file()
            else None,
            "vast_provider_adapter_result_path": str(
                effective_run_dir / "vast_provider_adapter_result.json"
            ),
            "vast_teardown_manifest_path": str(effective_run_dir / "vast_teardown_manifest.json"),
            "estimated_cost_usd": vast_result.get("estimated_cost_usd"),
            "postprocess_artifacts": postprocess,
            "review_video_path": postprocess.get("review_video_path"),
            "video_review_status_path": postprocess.get("video_review_status"),
            "source_policy_observation_visual_qa_path": postprocess.get(
                "source_policy_observation_visual_qa"
            ),
            "wam_rollout_visual_quality_report_path": postprocess.get(
                "wam_rollout_visual_quality_report"
            ),
            "wam_rollout_contact_sheet_path": postprocess.get("wam_rollout_contact_sheet"),
            "wam_rollout_visual_success": bool(postprocess.get("wam_rollout_visual_success")),
            "clean_frame_reanchoring": _mapping(imported.get("clean_frame_reanchoring")),
            "clean_frame_reanchor_event_count": int(
                imported.get("clean_frame_reanchor_event_count") or 0
            ),
            "periodic_clean_frame_reanchoring_used": bool(
                imported.get("periodic_clean_frame_reanchoring_used")
            ),
            "manipulation_success_evaluator_results_path": postprocess.get(
                "manipulation_success_evaluator_results"
            ),
            "claim_boundary_path": postprocess.get("claim_boundary"),
            "claim_boundary": {
                "simulator_generated_world_proof_only": True,
                "capture_truth": False,
                "geometry_truth": False,
                "collision_truth": False,
                "persistent_provider_session_is_runtime_proof_not_task_success": True,
                "valid_mp4_or_provider_completed_is_not_visual_success": True,
                "provider_success": completed,
                "provider_success_separate_from_visually_useful_rollout": True,
                "periodic_clean_frame_reanchoring_used": bool(
                    imported.get("periodic_clean_frame_reanchoring_used")
                ),
                "live_wam_generation_success_can_coexist_with_visually_useful_rollout_false": True,
                "visually_useful_rollout": bool(postprocess.get("wam_rollout_visual_success")),
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
                "safety_validation_proven": False,
                "real_world_manipulation_success_proven": False,
            },
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
        return output, 0 if completed else 2
    finally:
        if previous_policy_command is None:
            os.environ.pop("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND", None)
        else:
            os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"] = previous_policy_command
        if previous_persistent_inner_policy_command is None:
            os.environ.pop(PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV, None)
        else:
            os.environ[PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV] = (
                previous_persistent_inner_policy_command
            )
        if previous_vast_inner_policy_command is None:
            os.environ.pop(INNER_POLICY_COMMAND_ENV, None)
        else:
            os.environ[INNER_POLICY_COMMAND_ENV] = previous_vast_inner_policy_command


def run_persistent_session_runpod(
    *,
    policy_observation_path: str | Path,
    job_dir: str | Path | None = None,
    loop_step_count: int = 12,
    task_prompt: str | None = None,
    timeout_seconds: float = 3600.0,
    use_live_wam: bool = True,
    allow_structural_wam_fallback: bool | None = None,
    max_wait_seconds: int | None = None,
) -> tuple[dict[str, Any], int]:
    generated_at = utc_now_iso()
    job = _job_dir(job_dir)
    requested_loop_step_count = max(1, int(loop_step_count))
    allow_fallback = (
        _truthy(os.getenv(PERSISTENT_SESSION_ALLOW_STRUCTURAL_WAM_FALLBACK_ENV))
        if allow_structural_wam_fallback is None
        else bool(allow_structural_wam_fallback)
    )
    inner_policy_command = (
        _string(os.getenv(INNER_POLICY_COMMAND_ENV)) or DEFAULT_INNER_POLICY_COMMAND
    )
    previous_policy_command = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND")
    previous_persistent_inner_policy_command = os.environ.get(
        PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV
    )
    previous_vast_inner_policy_command = os.environ.get(INNER_POLICY_COMMAND_ENV)
    previous_wam_carrier_unitree = os.environ.get(
        "BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC"
    )
    previous_wam_default_env = {key: os.environ.get(key) for key in RUNPOD_WAM_CARRIER_ENV_KEYS}
    os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"] = inner_policy_command
    os.environ[PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV] = inner_policy_command
    os.environ[INNER_POLICY_COMMAND_ENV] = inner_policy_command
    runpod_provider_bundle_kind = (
        _string(os.getenv("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_PROVIDER_BUNDLE_KIND")) or "wam"
    )
    if runpod_provider_bundle_kind == "wam":
        os.environ["BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC"] = "true"
        os.environ.setdefault(OSCAR_WAM_VISUAL_PROFILE_ENV, "smoke")
        visual_profile = _normalized_wam_visual_profile()
        wam_carrier_defaults = _runpod_wam_carrier_defaults_for_profile(visual_profile)
        for key, value in wam_carrier_defaults.items():
            os.environ.setdefault(key, value)
        if visual_profile == "smoke":
            os.environ["BLUEPRINT_OSCAR_WAM_NUM_FRAMES"] = str(
                max(
                    RUNPOD_WAM_CARRIER_MIN_OSCAR_NUM_FRAMES,
                    _int_env(
                        "BLUEPRINT_OSCAR_WAM_NUM_FRAMES",
                        int(RUNPOD_WAM_CARRIER_SMOKE_DEFAULT_ENV["BLUEPRINT_OSCAR_WAM_NUM_FRAMES"]),
                    ),
                )
            )
    try:
        bundle = build_persistent_session_provider_bundle(
            job_dir=job / "provider_bundle",
            policy_observation_path=policy_observation_path,
            loop_step_count=requested_loop_step_count,
            task_prompt=task_prompt,
            timeout_seconds=timeout_seconds,
            use_live_wam=use_live_wam,
            allow_structural_wam_fallback=allow_fallback,
            generated_at=generated_at,
        )
        if bundle.get("status") != "bundle_ready":
            output = _blocked_payload(
                generated_at=generated_at,
                job_dir=job,
                blockers=bundle.get("blockers") or ["persistent_session_provider_bundle_blocked"],
                details={
                    "bundle_manifest_path": str(
                        job / "provider_bundle" / "persistent_session_provider_bundle_manifest.json"
                    ),
                    "provider": "runpod",
                },
            )
            write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
            return output, 2
        bundle_path = Path(str(bundle["bundle_path"])).expanduser().resolve()
        staging = stage_wam_provider_bundle_object_store(
            job_dir=job / "object_store_staging",
            bundle_path=bundle_path,
            key_prefix=_string(os.getenv(OBJECT_STORE_KEY_PREFIX_ENV))
            or DEFAULT_OBJECT_STORE_KEY_PREFIX,
            expiration_seconds=_int_env(
                "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIGNED_URL_SECONDS",
                21600,
            ),
            generated_at=generated_at,
        )
        if staging.get("status") != "completed":
            output = _blocked_payload(
                generated_at=generated_at,
                job_dir=job,
                blockers=staging.get("blockers")
                or ["persistent_session_object_store_staging_blocked"],
                details={
                    "object_store_staging_manifest_path": str(
                        job
                        / "object_store_staging"
                        / "wam_provider_object_store_staging_manifest.json"
                    ),
                    "provider": "runpod",
                },
            )
            write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
            return output, 2
        staging_dir = job / "object_store_staging"
        runpod_dir = job / "runpod_persistent_session_run"
        output_zip = runpod_dir / "runpod_provider_runtime_output.zip"
        create_manifest = create_runpod_wam_async_run(
            job_dir=runpod_dir,
            bundle_path=bundle_path,
            provider_bundle_url_file=staging_dir / "provider_bundle_url.txt",
            provider_output_put_url_file=staging_dir / "provider_output_put_url.txt",
            provider_output_get_url_file=staging_dir / "provider_output_get_url.txt",
            output_path=output_zip,
            allow_paid_runpod_launch=True,
            skip_public_staging_verification=True,
            image_name=(
                _string(os.getenv("BLUEPRINT_RUNPOD_WAM_PUBLIC_IMAGE"))
                or _string(os.getenv(PERSISTENT_SESSION_PUBLIC_IMAGE_ENV))
                or _string(os.getenv("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_PUBLIC_IMAGE"))
                or _string(os.getenv("BLUEPRINT_VAST_WAM_PUBLIC_IMAGE"))
                or DEFAULT_WAM_PUBLIC_IMAGE
            ),
            provider_bundle_kind=runpod_provider_bundle_kind,
            container_disk_gb=_int_env("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_DISK_GB", 240),
            volume_gb=_int_env("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_VOLUME_GB", 120),
            min_vcpu_per_gpu=_int_env("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_MIN_VCPU", 8),
            min_ram_per_gpu=_int_env("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_MIN_RAM_GB", 40),
            generated_at=generated_at,
        )
        if create_manifest.get("status") != "pod_created":
            output = _blocked_payload(
                generated_at=generated_at,
                job_dir=job,
                blockers=create_manifest.get("blockers")
                or ["persistent_session_runpod_create_blocked"],
                details={
                    "runpod_create_manifest_path": str(
                        runpod_dir / "runpod_wam_async_create_manifest.json"
                    ),
                    "provider": "runpod",
                },
            )
            write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
            return output, 2
        wait_seconds = _runpod_persistent_session_wait_seconds(
            explicit_max_wait_seconds=max_wait_seconds,
            timeout_seconds=timeout_seconds,
            loop_step_count=requested_loop_step_count,
        )
        poll_manifest = poll_runpod_wam_async_run(
            job_dir=runpod_dir,
            max_wait_seconds=wait_seconds,
            retry_interval_seconds=_int_env(
                "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_POLL_SECONDS",
                20,
            ),
            teardown=True,
            generated_at=generated_at,
        )
        if poll_manifest.get("status") != "completed" or not output_zip.is_file():
            poll_manifest_summary = {
                "status": poll_manifest.get("status"),
                "provider_command_status": poll_manifest.get("provider_command_status"),
                "pod_status": poll_manifest.get("pod_status"),
                "output_zip_present": bool(poll_manifest.get("output_zip_present")),
                "nonterminal_running_output": bool(poll_manifest.get("nonterminal_running_output")),
                "remote_runtime_running_without_terminal_output": bool(
                    poll_manifest.get("remote_runtime_running_without_terminal_output")
                ),
                "continuing_spend_from_this_run": bool(
                    poll_manifest.get("continuing_spend_from_this_run")
                ),
                "teardown_performed": bool(poll_manifest.get("teardown_performed")),
            }
            classification = _write_runpod_live_wam_blocker_classification(
                job=job,
                generated_at=generated_at,
                poll_manifest=poll_manifest,
            )
            output = _blocked_payload(
                generated_at=generated_at,
                job_dir=job,
                blockers=poll_manifest.get("provider_command_blockers")
                or [
                    "runpod_persistent_session_still_running"
                    if poll_manifest_summary["status"] == "running"
                    or poll_manifest_summary["provider_command_status"] == "running"
                    or poll_manifest_summary["continuing_spend_from_this_run"]
                    else "persistent_session_runpod_provider_blocked"
                ],
                details={
                    "runpod_create_manifest_path": str(
                        runpod_dir / "runpod_wam_async_create_manifest.json"
                    ),
                    "runpod_poll_manifest_path": str(
                        runpod_dir / "runpod_wam_async_poll_manifest.json"
                    ),
                    "runpod_delete_manifest_path": str(
                        runpod_dir / "runpod_wam_async_delete_manifest.json"
                    ),
                    "provider": "runpod",
                    "continuing_spend_from_this_run": poll_manifest.get(
                        "continuing_spend_from_this_run"
                    ),
                    "poll_manifest": poll_manifest_summary,
                    "runpod_live_wam_blocker_classification_path": str(
                        job / "runpod_live_wam_blocker_classification.json"
                    ),
                    "classified_blocker": classification.get("classified_blocker"),
                },
            )
            write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
            return output, 2
        extraction_dir = job / "imported_persistent_session_output"
        ensure_dir(extraction_dir)
        with zipfile.ZipFile(output_zip) as archive:
            archive.extractall(extraction_dir)
        imported_path = (
            extraction_dir / "unitree_groot_n17_sonic_wam_persistent_session_output.json"
        )
        if not imported_path.is_file():
            imported_path = extraction_dir / "unitree_groot_n17_sonic_policy_provider_output.json"
        imported = _read_json(imported_path) if imported_path.is_file() else {}
        postprocess = _postprocess_imported_persistent_session_artifacts(
            job=job,
            extraction_dir=extraction_dir,
            imported=imported,
            generated_at=generated_at,
            policy_observation_path=policy_observation_path,
            vast_result=poll_manifest,
            vast_run_dir=runpod_dir,
        )
        completed = imported.get("status") == "completed"
        classification = (
            _write_runpod_live_wam_blocker_classification(
                job=job,
                generated_at=generated_at,
                poll_manifest=poll_manifest,
                extraction_dir=extraction_dir,
                imported=imported,
            )
            if not completed
            else {"status": "completed", "classified_blocker": "none"}
        )
        output = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "completed" if completed else "blocked",
            "provider": "runpod",
            "policy_id": POLICY_ID,
            "selected_candidate_id": POLICY_ID,
            "job_dir": str(job),
            "persistent_provider_session_used": bool(
                imported.get("persistent_provider_session_used")
            ),
            "provider_instance_reused_for_policy_and_wam_loop": bool(
                imported.get("provider_instance_reused_for_policy_and_wam_loop")
            ),
            "repeated_policy_calls_count": int(imported.get("repeated_policy_calls_count") or 0),
            "generated_next_observation_count": int(
                imported.get("generated_next_observation_count") or 0
            ),
            "live_wam_generation_success_count": int(
                imported.get("live_wam_generation_success_count") or 0
            ),
            "learned_wam_model_success_count": int(
                imported.get("learned_wam_model_success_count") or 0
            ),
            "unitree_groot_n17_sonic_model_executed": bool(
                imported.get("unitree_groot_n17_sonic_model_executed")
            ),
            "unitree_groot_n17_sonic_policy_action_command_ran": bool(
                imported.get("unitree_groot_n17_sonic_policy_action_command_ran")
            ),
            "unitree_policy_action_command_ran": bool(
                imported.get("unitree_policy_action_command_ran")
            ),
            "policy_action_model_command_ran": bool(
                imported.get("policy_action_model_command_ran")
            ),
            "provider_output_replay_used": bool(imported.get("provider_output_replay_used")),
            "blockers": []
            if completed
            else imported.get("blockers") or ["persistent_session_provider_output_blocked"],
            "imported_provider_output_dir": str(extraction_dir),
            "imported_provider_output_path": str(imported_path)
            if imported_path.is_file()
            else None,
            "runpod_create_manifest_path": str(
                runpod_dir / "runpod_wam_async_create_manifest.json"
            ),
            "runpod_poll_manifest_path": str(runpod_dir / "runpod_wam_async_poll_manifest.json"),
            "runpod_teardown_manifest_path": str(
                runpod_dir / "runpod_wam_async_delete_manifest.json"
            ),
            "provider_runtime_output_zip_path": str(output_zip),
            "runpod_live_wam_blocker_classification_path": str(
                job / "runpod_live_wam_blocker_classification.json"
            )
            if not completed
            else None,
            "classified_blocker": classification.get("classified_blocker"),
            "continuing_spend_from_this_run": bool(
                poll_manifest.get("continuing_spend_from_this_run")
            ),
            "postprocess_artifacts": postprocess,
            "review_video_path": postprocess.get("review_video_path"),
            "video_review_status_path": postprocess.get("video_review_status"),
            "source_policy_observation_visual_qa_path": postprocess.get(
                "source_policy_observation_visual_qa"
            ),
            "wam_rollout_visual_quality_report_path": postprocess.get(
                "wam_rollout_visual_quality_report"
            ),
            "wam_rollout_contact_sheet_path": postprocess.get("wam_rollout_contact_sheet"),
            "wam_rollout_visual_success": bool(postprocess.get("wam_rollout_visual_success")),
            "clean_frame_reanchoring": _mapping(imported.get("clean_frame_reanchoring")),
            "clean_frame_reanchor_event_count": int(
                imported.get("clean_frame_reanchor_event_count") or 0
            ),
            "periodic_clean_frame_reanchoring_used": bool(
                imported.get("periodic_clean_frame_reanchoring_used")
            ),
            "manipulation_success_evaluator_results_path": postprocess.get(
                "manipulation_success_evaluator_results"
            ),
            "claim_boundary_path": postprocess.get("claim_boundary"),
            "claim_boundary": {
                "simulator_generated_world_proof_only": True,
                "capture_truth": False,
                "geometry_truth": False,
                "collision_truth": False,
                "persistent_provider_session_is_runtime_proof_not_task_success": True,
                "valid_mp4_or_provider_completed_is_not_visual_success": True,
                "provider_success": completed,
                "provider_success_separate_from_visually_useful_rollout": True,
                "periodic_clean_frame_reanchoring_used": bool(
                    imported.get("periodic_clean_frame_reanchoring_used")
                ),
                "live_wam_generation_success_can_coexist_with_visually_useful_rollout_false": True,
                "visually_useful_rollout": bool(postprocess.get("wam_rollout_visual_success")),
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
                "safety_validation_proven": False,
                "real_world_manipulation_success_proven": False,
            },
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
        return output, 0 if completed else 2
    finally:
        if previous_policy_command is None:
            os.environ.pop("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND", None)
        else:
            os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"] = previous_policy_command
        if previous_persistent_inner_policy_command is None:
            os.environ.pop(PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV, None)
        else:
            os.environ[PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV] = (
                previous_persistent_inner_policy_command
            )
        if previous_vast_inner_policy_command is None:
            os.environ.pop(INNER_POLICY_COMMAND_ENV, None)
        else:
            os.environ[INNER_POLICY_COMMAND_ENV] = previous_vast_inner_policy_command
        if previous_wam_carrier_unitree is None:
            os.environ.pop("BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC", None)
        else:
            os.environ["BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC"] = (
                previous_wam_carrier_unitree
            )
        for key, previous in previous_wam_default_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-observation", required=True)
    parser.add_argument("--job-dir")
    parser.add_argument("--provider", choices=("runpod", "vast"), default="runpod")
    parser.add_argument("--loop-step-count", type=int, default=12)
    parser.add_argument("--task-prompt")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--disable-live-wam", action="store_true")
    parser.add_argument("--allow-structural-wam-fallback", action="store_true")
    args = parser.parse_args(argv)
    runner = run_persistent_session_runpod if args.provider == "runpod" else run_persistent_session
    result, exit_code = runner(
        policy_observation_path=args.policy_observation,
        job_dir=args.job_dir,
        loop_step_count=args.loop_step_count,
        task_prompt=args.task_prompt,
        timeout_seconds=args.timeout_seconds,
        use_live_wam=not args.disable_live_wam,
        allow_structural_wam_fallback=args.allow_structural_wam_fallback,
    )
    print(json.dumps(result, sort_keys=True))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
