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

from . import launch_provenance
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
    DEFAULT_HEARTBEAT_NO_PROGRESS_SECONDS,
    DEFAULT_PUBLIC_CUDA_IMAGE,
    VAST_IMAGE_LOGIN_MODE_ENV,
    run_vast_provider_adapter,
)
from .vast_wam_authorized_runner import DEFAULT_WAM_PUBLIC_IMAGE
from .wam_provider_object_store import stage_wam_provider_bundle_object_store
from .runpod_wam_async_runner import (
    RUNPOD_WAM_TEARDOWN_ACTION_ENV,
    create_runpod_wam_async_run,
    poll_runpod_wam_async_run,
)
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
from .oscar_cosmos_wam_evaluator import (
    WAM_CONSISTENCY_COMMAND_ENV,
    WAM_CONSISTENCY_COMMAND_OUTPUT,
    WAM_CONSISTENCY_GATE_ENV,
    _env_truthy as _wam_consistency_env_truthy,
    _normalize_wam_episode_consistency,
    _run_wam_consistency_command,
    _unscored_wam_episode_consistency,
    _wam_consistency_blockers,
)
from .oscar_official_release import OFFICIAL_OSCAR_WAM_IMAGE_REF


SCHEMA_VERSION = "unitree_groot_n17_sonic_vast_persistent_session.v1"
BUNDLE_SCHEMA_VERSION = "unitree_groot_n17_sonic_wam_persistent_session_bundle.v1"
OUTPUT_SCHEMA_VERSION = "unitree_groot_n17_sonic_wam_persistent_session_output.v1"
DEFAULT_BUNDLE_FILENAME = "unitree_groot_n17_sonic_wam_persistent_session_bundle.zip"
DEFAULT_OBJECT_STORE_KEY_PREFIX = "blueprint/unitree-groot-sonic-persistent-session"
DEFAULT_RUNPOD_UNITREE_GROOT_SONIC_WAM_PUBLIC_IMAGE = (
    OFFICIAL_OSCAR_WAM_IMAGE_REF
)
RUNTIME_PROJECTED_SKELETON_TRACE_BUNDLE_PATH = (
    "provider_runtime/seed_conditioning/g1_projected_skeleton_trace.jsonl"
)
RUNTIME_ISAAC_SCENE_CONTEXT_BUNDLE_DIR = "provider_runtime/isaac_scene_context"
EXPLICIT_ISAAC_MANIPULATION_POV_GEOMETRY_ENV = (
    "BLUEPRINT_PERSISTENT_SESSION_ISAAC_MANIPULATION_POV_GEOMETRY"
)
EXPLICIT_ISAAC_PLACEMENT_VALIDATION_ENV = (
    "BLUEPRINT_PERSISTENT_SESSION_ISAAC_PLACEMENT_VALIDATION"
)
EXPLICIT_ISAAC_TASK_STANCE_PLAN_ENV = "BLUEPRINT_PERSISTENT_SESSION_ISAAC_TASK_STANCE_PLAN"
PERSISTENT_SESSION_JOB_ROOT_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PERSISTENT_SESSION_JOB_ROOT"
PERSISTENT_SESSION_PUBLIC_IMAGE_ENV = "BLUEPRINT_VAST_UNITREE_WAM_PERSISTENT_SESSION_PUBLIC_IMAGE"
PERSISTENT_SESSION_ALLOW_STRUCTURAL_WAM_FALLBACK_ENV = (
    "BLUEPRINT_ALLOW_PERSISTENT_SESSION_STRUCTURAL_WAM_FALLBACK"
)
PERSISTENT_SESSION_USE_LIVE_WAM_ENV = "BLUEPRINT_PERSISTENT_SESSION_USE_LIVE_WAM"
PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV = (
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PERSISTENT_INNER_POLICY_COMMAND"
)
ALLOW_DIRTY_PAID_LAUNCH_ENV = "BLUEPRINT_ALLOW_DIRTY_PAID_LAUNCH"
RUNPOD_FULL_LOOP_OVERRIDE_ENV = "BLUEPRINT_ALLOW_UNITREE_GROOT_N17_SONIC_RUNPOD_FULL_LOOP"
OSCAR_WAM_VISUAL_PROFILE_ENV = "BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE"
PERSISTENT_WAM_LONG_REVIEW_ROLLOUT_ENV = "BLUEPRINT_ALLOW_PERSISTENT_WAM_LONG_REVIEW_ROLLOUT"
PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST_ENV = (
    "BLUEPRINT_PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST"
)
PERSISTENT_WAM_SHORT_VISUAL_SANITY_SCHEMA_VERSION = "persistent_wam_short_visual_sanity.v1"
PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_WIDTH = 320
PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_HEIGHT = 256
PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_FPS = REVIEW_QUALITY_MIN_FPS
PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_NUM_FRAMES = (
    REVIEW_QUALITY_MIN_NUM_FRAMES
)
PERSISTENT_WAM_REVIEW_QUALITY_MAX_UNGATED_LOOP_STEPS_ENV = (
    "BLUEPRINT_PERSISTENT_WAM_REVIEW_QUALITY_MAX_UNGATED_LOOP_STEPS"
)
PERSISTENT_WAM_CLEAN_FRAME_REANCHOR_INTERVAL_ENV = (
    "BLUEPRINT_PERSISTENT_WAM_CLEAN_FRAME_REANCHOR_INTERVAL_STEPS"
)
PERSISTENT_WAM_AUTOREGRESSIVE_DRIFT_BLOCKER_MANIFEST_ENV = (
    "BLUEPRINT_PERSISTENT_WAM_AUTOREGRESSIVE_DRIFT_BLOCKER_MANIFEST"
)
PERSISTENT_WAM_MATERIALIZATION_BLOCKER_MANIFEST_ENV = (
    "BLUEPRINT_PERSISTENT_WAM_MATERIALIZATION_BLOCKER_MANIFEST"
)
PERSISTENT_WAM_LONG_REVIEW_ROLLOUT_MIN_STEPS = 12
REVIEW_QUALITY_MIN_OSCAR_NUM_STEPS = 35
REVIEW_QUALITY_MIN_OSCAR_GUIDANCE = 6.0
REVIEW_QUALITY_MIN_OSCAR_NUM_FRAMES = 81
PERSISTENT_WAM_LONG_REVIEW_QUALITY_GATE_SCHEMA_VERSION = (
    "persistent_wam_long_review_rollout_quality_gate.v1"
)
PERSISTENT_WAM_RANK_FIDELITY_CALIBRATION_REQUIREMENT_SCHEMA_VERSION = (
    "persistent_wam_rank_fidelity_calibration_requirement.v1"
)
PERSISTENT_WAM_RANK_FIDELITY_CALIBRATION_ANCHOR_REQUEST_SCHEMA_VERSION = (
    "persistent_wam_rank_fidelity_calibration_anchor_request.v1"
)
PERSISTENT_WAM_RANK_FIDELITY_SMALL_CALIBRATION_SET_SCHEMA_VERSION = (
    "persistent_wam_rank_fidelity_small_calibration_set.v1"
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
    "BLUEPRINT_OSCAR_WAM_GUIDANCE": "3.5",
    "BLUEPRINT_OSCAR_WAM_NUM_FRAMES": "9",
    "BLUEPRINT_OSCAR_WAM_HEIGHT": "128",
    "BLUEPRINT_OSCAR_WAM_WIDTH": "128",
    "BLUEPRINT_OSCAR_WAM_FPS": "4",
    "BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS": "1200",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE": "system_python_minimal",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT": "true",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SERVER_STARTUP_TIMEOUT_SECONDS": "2400",
}
RUNPOD_WAM_CARRIER_REVIEW_QUALITY_DEFAULT_ENV = {
    OSCAR_WAM_VISUAL_PROFILE_ENV: "review_quality",
    "BLUEPRINT_OSCAR_WAM_NUM_STEPS": str(REVIEW_QUALITY_MIN_OSCAR_NUM_STEPS),
    "BLUEPRINT_OSCAR_WAM_GUIDANCE": str(REVIEW_QUALITY_MIN_OSCAR_GUIDANCE),
    "BLUEPRINT_OSCAR_WAM_NUM_FRAMES": str(REVIEW_QUALITY_MIN_OSCAR_NUM_FRAMES),
    "BLUEPRINT_OSCAR_WAM_HEIGHT": "480",
    "BLUEPRINT_OSCAR_WAM_WIDTH": "640",
    "BLUEPRINT_OSCAR_WAM_FPS": "15",
    "BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS": "1200",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE": "system_python_minimal",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT": "true",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SERVER_STARTUP_TIMEOUT_SECONDS": "2400",
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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [text for item in value if (text := _string(item))]


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
        "guidance": _float_env(
            "BLUEPRINT_OSCAR_WAM_GUIDANCE",
            float(defaults["BLUEPRINT_OSCAR_WAM_GUIDANCE"]),
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
            "oscar_num_frames": REVIEW_QUALITY_MIN_OSCAR_NUM_FRAMES,
            "num_steps": REVIEW_QUALITY_MIN_OSCAR_NUM_STEPS,
            "guidance": REVIEW_QUALITY_MIN_OSCAR_GUIDANCE,
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
    empty_blocker: str | None = None,
) -> str | None:
    path = _resolve_optional_path(payload.get(key))
    if path is None or not path.is_file():
        return blocker
    if path.stat().st_size <= 0:
        return empty_blocker or blocker
    return None


def _read_manifest_artifact_json(
    payload: Mapping[str, Any],
    key: str,
    unreadable_blocker: str,
) -> tuple[dict[str, Any], list[str]]:
    path = _resolve_optional_path(payload.get(key))
    if path is None or not path.is_file():
        return {}, []
    try:
        return _read_json(path), []
    except Exception as exc:
        return {}, [f"{unreadable_blocker}:{type(exc).__name__}"]


def _first_ffprobe_video_stream(metadata: Mapping[str, Any]) -> dict[str, Any]:
    streams = metadata.get("streams")
    if not isinstance(streams, Sequence) or isinstance(streams, (str, bytes, bytearray)):
        return {}
    for stream in streams:
        if isinstance(stream, Mapping):
            return dict(stream)
    return {}


def _rationalish_float(value: Any) -> float:
    text = _string(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            return float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _ffprobe_review_media_profile(metadata: Mapping[str, Any]) -> dict[str, Any]:
    stream = _first_ffprobe_video_stream(metadata)
    width = _intish(stream.get("width")) or 0
    height = _intish(stream.get("height")) or 0
    fps = _rationalish_float(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    frame_count = _intish(stream.get("nb_frames")) or 0
    resolution_passed = bool(
        width >= PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_WIDTH
        and height >= PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_HEIGHT
    )
    fps_passed = bool(fps >= PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_FPS)
    frame_count_passed = bool(
        frame_count >= PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_NUM_FRAMES
    )
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "minimum_width": PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_WIDTH,
        "minimum_height": PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_HEIGHT,
        "minimum_fps": PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_FPS,
        "minimum_num_frames": PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_NUM_FRAMES,
        "resolution_passed": resolution_passed,
        "fps_passed": fps_passed,
        "frame_count_passed": frame_count_passed,
        "passed": bool(resolution_passed and fps_passed and frame_count_passed),
    }


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
            media_profile = _ffprobe_review_media_profile(ffprobe_metadata)
            if not media_profile["resolution_passed"]:
                blockers.append("short_visual_sanity_review_video_below_minimum_resolution")
            if not media_profile["fps_passed"]:
                blockers.append(
                    "short_visual_sanity_review_video_fps_below_review_quality_minimum"
                )
            if not media_profile["frame_count_passed"]:
                blockers.append(
                    "short_visual_sanity_review_video_frame_count_below_review_quality_minimum"
                )
        for key, blocker, empty_blocker in (
            (
                "source_policy_observation_visual_qa_path",
                "short_visual_sanity_source_qa_artifact_missing",
                "short_visual_sanity_source_qa_artifact_empty",
            ),
            (
                "wam_rollout_visual_quality_report_path",
                "short_visual_sanity_quality_report_missing",
                "short_visual_sanity_quality_report_empty",
            ),
            (
                "wam_rollout_contact_sheet_path",
                "short_visual_sanity_contact_sheet_missing",
                "short_visual_sanity_contact_sheet_empty",
            ),
            (
                "video_review_status_path",
                "short_visual_sanity_video_status_missing",
                "short_visual_sanity_video_status_empty",
            ),
            (
                "review_video_path",
                "short_visual_sanity_review_video_missing",
                "short_visual_sanity_review_video_empty",
            ),
        ):
            artifact_blocker = _existing_artifact_path_blocker(
                payload,
                key,
                blocker,
                empty_blocker,
            )
            if artifact_blocker:
                blockers.append(artifact_blocker)
        source_qa_artifact, source_qa_artifact_blockers = _read_manifest_artifact_json(
            payload,
            "source_policy_observation_visual_qa_path",
            "short_visual_sanity_source_qa_artifact_unreadable",
        )
        blockers.extend(source_qa_artifact_blockers)
        if source_qa_artifact:
            if source_qa_artifact.get("status") != "passed_visual_quality_gate":
                blockers.append("short_visual_sanity_source_qa_artifact_not_passed")
                blockers.extend(str(item) for item in source_qa_artifact.get("blockers") or [])
        visual_report_artifact, visual_report_artifact_blockers = _read_manifest_artifact_json(
            payload,
            "wam_rollout_visual_quality_report_path",
            "short_visual_sanity_quality_report_unreadable",
        )
        blockers.extend(visual_report_artifact_blockers)
        if visual_report_artifact:
            if visual_report_artifact.get("status") not in {
                None,
                "passed_visual_quality_gate",
            }:
                blockers.append("short_visual_sanity_quality_report_status_not_passed")
            profile_contract = _mapping(visual_report_artifact.get("profile_contract"))
            if _string(visual_report_artifact.get("visual_profile")) != "review_quality":
                blockers.append("short_visual_sanity_quality_report_not_review_quality_profile")
            if profile_contract.get("smoke_only") is True:
                blockers.append("short_visual_sanity_quality_report_smoke_only")
            if (
                profile_contract
                and profile_contract.get("review_quality_minimum_satisfied") is not True
            ):
                blockers.append(
                    "short_visual_sanity_quality_report_review_quality_minimum_not_satisfied"
                )
            if visual_report_artifact.get("visual_success") is not True:
                blockers.append("short_visual_sanity_quality_report_visual_success_not_passed")
                blockers.extend(
                    str(item) for item in visual_report_artifact.get("blockers") or []
                )
            if visual_report_artifact.get("structural_fallback_used") is True:
                blockers.append("short_visual_sanity_quality_report_structural_fallback_used")
        video_status_artifact, video_status_artifact_blockers = _read_manifest_artifact_json(
            payload,
            "video_review_status_path",
            "short_visual_sanity_video_status_unreadable",
        )
        blockers.extend(video_status_artifact_blockers)
        if video_status_artifact:
            if video_status_artifact.get("status") != "completed":
                blockers.append("short_visual_sanity_video_status_not_completed")
            if video_status_artifact.get("ffprobe_command_ran") is not True:
                blockers.append("short_visual_sanity_video_status_ffprobe_command_not_ran")
            if _intish(video_status_artifact.get("ffprobe_returncode")) != 0:
                blockers.append("short_visual_sanity_video_status_ffprobe_returncode_not_zero")
            video_ffprobe_metadata = _mapping(video_status_artifact.get("ffprobe_metadata"))
            if not video_ffprobe_metadata:
                blockers.append("short_visual_sanity_video_status_ffprobe_metadata_missing")
            else:
                media_profile = _ffprobe_review_media_profile(video_ffprobe_metadata)
                if not media_profile["resolution_passed"]:
                    blockers.append(
                        "short_visual_sanity_video_status_review_video_below_minimum_resolution"
                    )
                if not media_profile["fps_passed"]:
                    blockers.append(
                        "short_visual_sanity_video_status_review_video_fps_below_review_quality_minimum"
                    )
                if not media_profile["frame_count_passed"]:
                    blockers.append(
                        "short_visual_sanity_video_status_review_video_frame_count_below_review_quality_minimum"
                    )
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
            elif teardown_path.stat().st_size <= 0:
                blockers.append("short_visual_sanity_paid_provider_teardown_manifest_empty")
            else:
                try:
                    teardown_artifact = _read_json(teardown_path)
                except Exception as exc:
                    blockers.append(
                        "short_visual_sanity_paid_provider_teardown_manifest_unreadable:"
                        f"{type(exc).__name__}"
                    )
                else:
                    teardown_status = _string(teardown_artifact.get("status"))
                    teardown_completed = bool(
                        teardown_status == "completed"
                        or teardown_artifact.get("runner_gpu_teardown_completed") is True
                    )
                    if not teardown_completed:
                        blockers.append(
                            "short_visual_sanity_paid_provider_teardown_artifact_not_completed"
                        )
                    if teardown_artifact.get("continuing_spend_from_this_run") is not False:
                        blockers.append(
                            "short_visual_sanity_paid_provider_teardown_artifact_not_zero_spend"
                        )

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


def validate_persistent_wam_materialization_quality_blocker(
    path: str | Path | None,
) -> dict[str, Any]:
    """Validate prior materialization evidence that makes another paid run redundant."""
    manifest_path = _resolve_optional_path(path)
    blockers: list[str] = []
    payload: dict[str, Any] = {}
    materialization: dict[str, Any] = {}
    if manifest_path is None:
        return {
            "schema_version": "persistent_wam_materialization_quality_blocker_validation.v1",
            "status": "not_configured",
            "manifest_path": None,
            "concrete_materialization_quality_blocker_proven": False,
            "blockers": [],
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
    if not manifest_path.is_file():
        blockers.append("materialization_quality_blocker_manifest_missing")
    else:
        try:
            payload = _read_json(manifest_path)
        except Exception as exc:
            blockers.append(
                f"materialization_quality_blocker_manifest_unreadable:{type(exc).__name__}"
            )
            payload = {}

    if payload:
        materialization = _mapping(payload.get("materialization_quality")) or payload
        future_frame_quality_status = _string(materialization.get("future_frame_quality_status"))
        future_frame_quality_blockers = set(
            _string_list(materialization.get("future_frame_quality_blockers"))
        )
        try:
            degraded_future_frame_count = int(materialization.get("degraded_future_frame_count") or 0)
        except (TypeError, ValueError):
            degraded_future_frame_count = 0
        try:
            video_first_frame_materialization_count = int(
                materialization.get("video_first_frame_materialization_count") or 0
            )
        except (TypeError, ValueError):
            video_first_frame_materialization_count = 0
        concrete_materialization_blocker = bool(
            future_frame_quality_status == "failed"
            and (
                degraded_future_frame_count > 0
                or video_first_frame_materialization_count > 0
                or "wam_generated_next_observation_future_frame_degraded_visual_signal"
                in future_frame_quality_blockers
                or "wam_generated_next_observation_used_video_first_frame_fallback"
                in future_frame_quality_blockers
            )
        )
        if future_frame_quality_status != "failed":
            blockers.append("materialization_quality_blocker_status_not_failed")
        if not concrete_materialization_blocker:
            blockers.append("materialization_quality_blocker_not_concrete")
    else:
        concrete_materialization_blocker = False

    return {
        "schema_version": "persistent_wam_materialization_quality_blocker_validation.v1",
        "status": "confirmed_materialization_quality_blocker"
        if payload and concrete_materialization_blocker and not blockers
        else "blocked",
        "manifest_path": str(manifest_path),
        "concrete_materialization_quality_blocker_proven": bool(
            payload and concrete_materialization_blocker and not blockers
        ),
        "future_frame_quality_status": materialization.get("future_frame_quality_status"),
        "future_frame_quality_blockers": _string_list(
            materialization.get("future_frame_quality_blockers")
        ),
        "degraded_future_frame_count": materialization.get("degraded_future_frame_count"),
        "video_first_frame_materialization_count": materialization.get(
            "video_first_frame_materialization_count"
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
    materialization_validation = validate_persistent_wam_materialization_quality_blocker(
        os.getenv(PERSISTENT_WAM_MATERIALIZATION_BLOCKER_MANIFEST_ENV)
    )
    blockers: list[str] = []
    status = "not_required"
    paid_launch_allowed = True
    if (
        settings.get("visual_profile") == "review_quality"
        and materialization_validation.get("status")
        == "confirmed_materialization_quality_blocker"
    ):
        status = "blocked_materialization_quality_confirmed"
        paid_launch_allowed = False
        blockers.append("future_frame_materialization_quality_blocker_present_before_paid_rollout")
    elif required:
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
        "materialization_quality_blocker_validation": materialization_validation,
        "periodic_clean_frame_reanchoring_proven": bool(
            reanchoring.get("periodic_clean_frame_reanchoring_proven")
        ),
        "concrete_autoregressive_drift_blocker_proven": bool(
            drift_validation.get("concrete_autoregressive_drift_blocker_proven")
        ),
        "concrete_materialization_quality_blocker_proven": bool(
            materialization_validation.get("concrete_materialization_quality_blocker_proven")
        ),
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "clean_frame_reanchoring_is_quality_control_not_task_success": True,
            "autoregressive_drift_blocker_prevents_paid_long_rollout": bool(
                status == "blocked_autoregressive_drift_confirmed"
            ),
            "long_rollout_quality_gate_is_not_generated_world_rank_fidelity": True,
            "materialization_quality_blocker_prevents_same_config_paid_rollout": bool(
                status == "blocked_materialization_quality_confirmed"
            ),
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
            "accepted_anchor_manipulation_success_proven": False,
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
    if int(settings.get("num_frames") or 0) < REVIEW_QUALITY_MIN_OSCAR_NUM_FRAMES:
        blockers.append("review_quality_profile_num_frames_below_oscar_default")
    if int(settings.get("num_steps") or 0) < REVIEW_QUALITY_MIN_OSCAR_NUM_STEPS:
        blockers.append("review_quality_profile_num_steps_below_oscar_default")
    if float(settings.get("guidance") or 0.0) < REVIEW_QUALITY_MIN_OSCAR_GUIDANCE:
        blockers.append("review_quality_profile_guidance_below_oscar_default")
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


def _attach_explicit_isaac_scene_context(
    observation: dict[str, Any],
    *,
    base_dir: Path,
    manipulation_pov_geometry_path: str | Path | None = None,
    placement_validation_path: str | Path | None = None,
    task_stance_plan_path: str | Path | None = None,
) -> dict[str, Any]:
    raw_paths = {
        "manipulation_pov_geometry": _string(
            manipulation_pov_geometry_path
            or os.getenv(EXPLICIT_ISAAC_MANIPULATION_POV_GEOMETRY_ENV)
        ),
        "placement_validation": _string(
            placement_validation_path or os.getenv(EXPLICIT_ISAAC_PLACEMENT_VALIDATION_ENV)
        ),
        "task_stance_plan": _string(
            task_stance_plan_path or os.getenv(EXPLICIT_ISAAC_TASK_STANCE_PLAN_ENV)
        ),
    }
    requested_paths = {key: value for key, value in raw_paths.items() if value}
    if not requested_paths:
        return {
            "schema_version": "persistent_session_explicit_isaac_scene_context.v1",
            "status": "not_requested",
            "requested": False,
            "requested_source_paths": raw_paths,
            "resolved_source_paths": {},
            "blockers": [],
            "claim_boundary": {
                "explicit_isaac_scene_context_is_optional_runtime_conditioning": True,
                "explicit_isaac_scene_context_is_not_capture_truth": True,
                "explicit_isaac_scene_context_is_not_task_success_proof": True,
            },
        }

    visual = _mapping(observation.get("visual_observation"))
    resolved_paths: dict[str, str] = {}
    blockers: list[str] = []

    def resolve_required(label: str, value: str) -> Path | None:
        path = _resolve_local_json_path(value, base_dir=base_dir)
        if path is None:
            blockers.append(f"blocked_explicit_isaac_{label}_path_unresolvable")
            return None
        if not path.is_file():
            blockers.append(f"blocked_explicit_isaac_{label}_path_missing")
            resolved_paths[label] = str(path)
            return None
        resolved = path.resolve()
        resolved_paths[label] = str(resolved)
        return resolved

    geometry_path = resolve_required(
        "manipulation_pov_geometry",
        requested_paths["manipulation_pov_geometry"],
    ) if requested_paths.get("manipulation_pov_geometry") else None
    placement_path = resolve_required(
        "placement_validation",
        requested_paths["placement_validation"],
    ) if requested_paths.get("placement_validation") else None
    stance_path = resolve_required(
        "task_stance_plan",
        requested_paths["task_stance_plan"],
    ) if requested_paths.get("task_stance_plan") else None

    if geometry_path is not None:
        geometry_text = str(geometry_path)
        observation["manipulation_pov_geometry_path"] = geometry_text
        observation["isaac_manipulation_pov_geometry_path"] = geometry_text
        visual["manipulation_pov_geometry_path"] = geometry_text
        visual["isaac_manipulation_pov_geometry_path"] = geometry_text
    if placement_path is not None:
        placement_text = str(placement_path)
        observation["placement_validation_path"] = placement_text
        observation["isaac_scene_manifest_path"] = placement_text
        visual["placement_validation_path"] = placement_text
        visual["isaac_scene_manifest_path"] = placement_text
    if stance_path is not None:
        stance_text = str(stance_path)
        observation["task_stance_plan_path"] = stance_text
        visual["task_stance_plan_path"] = stance_text
    observation["visual_observation"] = visual

    claim_boundary = _mapping(observation.get("claim_boundary"))
    claim_boundary.update(
        {
            "explicit_isaac_scene_context_attached": True,
            "explicit_isaac_scene_context_is_not_capture_truth": True,
            "explicit_isaac_scene_context_is_not_task_success_proof": True,
            "scene_or_task_specific_coordinates_hardcoded": False,
        }
    )
    observation["claim_boundary"] = claim_boundary
    return {
        "schema_version": "persistent_session_explicit_isaac_scene_context.v1",
        "status": "blocked" if blockers else "attached",
        "requested": True,
        "requested_source_paths": raw_paths,
        "resolved_source_paths": resolved_paths,
        "blockers": blockers,
        "claim_boundary": {
            "explicit_isaac_scene_context_is_optional_runtime_conditioning": True,
            "explicit_isaac_scene_context_is_not_capture_truth": True,
            "explicit_isaac_scene_context_is_not_task_success_proof": True,
            "explicit_sidecars_copied_into_provider_bundle_before_wam_conditioning": not blockers,
        },
    }


def _source_geometry_path_from_projected_skeleton_trace(
    trace_path: Path | None,
    *,
    base_dir: Path,
) -> Path | None:
    if trace_path is None or not trace_path.is_file():
        return None
    try:
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            for key in (
                "source_geometry_path",
                "manipulation_pov_geometry_path",
                "seed_geometry_path",
                "geometry_path",
            ):
                path = _resolve_local_json_path(
                    row.get(key),
                    base_dir=trace_path.parent if trace_path.parent else base_dir,
                )
                if path and path.is_file():
                    return path.resolve()
            break
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _first_existing_local_json_path(
    values: Sequence[Any],
    *,
    base_dir: Path,
) -> tuple[Path | None, Path | None]:
    unresolved: Path | None = None
    for value in values:
        path = _resolve_local_json_path(value, base_dir=base_dir)
        if path is None:
            continue
        unresolved = path
        if path.is_file():
            return path.resolve(), unresolved
    return None, unresolved


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
    projected_skeleton_trace_path: Path | None = None
    unresolved_projected_skeleton_trace_path: Path | None = None
    projected_skeleton_trace_path, unresolved_projected_skeleton_trace_path = (
        _first_existing_local_json_path(
            (
                observation.get("projected_skeleton_trace_path"),
                observation.get("g1_projected_skeleton_trace_jsonl"),
                visual.get("projected_skeleton_trace_path"),
                visual.get("g1_projected_skeleton_trace_jsonl"),
            ),
            base_dir=base_dir,
        )
    )
    trace_source_geometry_path = _source_geometry_path_from_projected_skeleton_trace(
        projected_skeleton_trace_path,
        base_dir=base_dir,
    )
    manipulation_pov_geometry_path, unresolved_manipulation_pov_geometry_path = (
        _first_existing_local_json_path(
            (
                observation.get("manipulation_pov_geometry_path"),
                observation.get("isaac_manipulation_pov_geometry_path"),
                observation.get("seed_geometry_path"),
                observation.get("geometry_path"),
                visual.get("manipulation_pov_geometry_path"),
                visual.get("isaac_manipulation_pov_geometry_path"),
                visual.get("seed_geometry_path"),
                visual.get("geometry_path"),
                trace_source_geometry_path,
                base_dir / "manipulation_pov_geometry.json",
                base_dir.parent / "manipulation_pov_geometry.json",
            ),
            base_dir=base_dir,
        )
    )
    isaac_scene_manifest_path, unresolved_isaac_scene_manifest_path = (
        _first_existing_local_json_path(
            (
                observation.get("isaac_scene_manifest_path"),
                observation.get("placement_validation_path"),
                observation.get("placement_validation_json_path"),
                visual.get("isaac_scene_manifest_path"),
                visual.get("placement_validation_path"),
                visual.get("placement_validation_json_path"),
                (
                    manipulation_pov_geometry_path.parent / "placement_validation.json"
                    if manipulation_pov_geometry_path
                    else None
                ),
                base_dir / "placement_validation.json",
                base_dir.parent / "placement_validation.json",
            ),
            base_dir=base_dir,
        )
    )
    task_stance_plan_path, unresolved_task_stance_plan_path = _first_existing_local_json_path(
        (
            observation.get("task_stance_plan_path"),
            visual.get("task_stance_plan_path"),
            manipulation_pov_geometry_path.parent / "task_stance_plan.json"
            if manipulation_pov_geometry_path
            else None,
            base_dir / "task_stance_plan.json",
            base_dir.parent / "task_stance_plan.json",
        ),
        base_dir=base_dir,
    )
    artifact_base_dir = object_index_path.parent if object_index_path else base_dir
    return {
        "object_index": object_index if isinstance(object_index, (Mapping, list)) else None,
        "object_index_path": str(object_index_path) if object_index_path else None,
        "eval_ready_task_grounding": dict(eval_ready) if isinstance(eval_ready, Mapping) else None,
        "eval_ready_task_grounding_path": str(eval_ready_path) if eval_ready_path else None,
        "projected_skeleton_trace_path": str(
            projected_skeleton_trace_path or unresolved_projected_skeleton_trace_path
        )
        if (projected_skeleton_trace_path or unresolved_projected_skeleton_trace_path)
        else None,
        "manipulation_pov_geometry_path": str(
            manipulation_pov_geometry_path or unresolved_manipulation_pov_geometry_path
        )
        if (manipulation_pov_geometry_path or unresolved_manipulation_pov_geometry_path)
        else None,
        "isaac_scene_manifest_path": str(
            isaac_scene_manifest_path or unresolved_isaac_scene_manifest_path
        )
        if (isaac_scene_manifest_path or unresolved_isaac_scene_manifest_path)
        else None,
        "task_stance_plan_path": str(task_stance_plan_path or unresolved_task_stance_plan_path)
        if (task_stance_plan_path or unresolved_task_stance_plan_path)
        else None,
        "semantic_artifact_base_dir": str(artifact_base_dir),
    }


def _copy_projected_skeleton_trace_for_runtime(
    source_path: Any,
    *,
    runtime_dir: Path,
) -> Path | None:
    source_text = _string(source_path)
    if not source_text:
        return None
    source = Path(source_text).expanduser()
    if not source.is_file():
        return None
    destination = runtime_dir / "seed_conditioning" / "g1_projected_skeleton_trace.jsonl"
    ensure_dir(destination.parent)
    shutil.copy2(source, destination)
    return destination


def _copy_isaac_scene_context_for_runtime(
    semantic_visual_evidence: Mapping[str, Any],
    *,
    runtime_dir: Path,
) -> dict[str, str]:
    copied: dict[str, str] = {}
    context_dir = runtime_dir / "isaac_scene_context"
    for key, filename in (
        ("manipulation_pov_geometry_path", "manipulation_pov_geometry.json"),
        ("isaac_scene_manifest_path", "placement_validation.json"),
        ("task_stance_plan_path", "task_stance_plan.json"),
    ):
        source_text = _string(semantic_visual_evidence.get(key))
        if not source_text:
            continue
        source = Path(source_text).expanduser()
        if not source.is_file():
            continue
        destination = context_dir / filename
        ensure_dir(destination.parent)
        shutil.copy2(source, destination)
        copied[key] = str(destination)
    return copied


def _write_executable(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


PERSISTENT_SESSION_RUNNER = r"""#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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
from typing import Any, Mapping, Sequence
from urllib import request as urllib_request
from urllib import error as urllib_error

OUTPUT_SCHEMA_VERSION = "unitree_groot_n17_sonic_wam_persistent_session_output.v1"
POLICY_ID = "unitree_groot_n17_sonic_policy"
REVIEW_QUALITY_MIN_OSCAR_NUM_FRAMES = 81


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


def _copy_runtime_sidecar(source: Path, destination: Path) -> str | None:
    if not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


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


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _materialize_future_frame_from_video(candidate: Path, target_frame: Path) -> dict[str, Any]:
    selection_dir = target_frame.parent / f"{target_frame.stem}_future_frame_selection"
    selection_manifest_path = selection_dir / "next_observation_selection.json"
    try:
        from blueprint_pipeline.wam_generated_video_review import (
            extract_next_observation_frame_from_video,
        )
    except Exception as exc:
        return {
            "status": "blocked",
            "blockers": [f"future_frame_selector_import_failed:{type(exc).__name__}"],
            "source_path": str(candidate),
        }

    selected_frame = extract_next_observation_frame_from_video(candidate, selection_dir)
    selection = _read_optional_json(selection_manifest_path)
    if selected_frame is None or not selected_frame.is_file():
        return {
            "status": "blocked",
            "blockers": list(selection.get("blockers") or ["no_usable_future_next_observation_frame"]),
            "source_path": str(candidate),
            "future_frame_selected": False,
            "frame_selection_policy": "prefer_signal_valid_else_earliest_decodable_future_frame",
            "selection_manifest_path": str(selection_manifest_path)
            if selection_manifest_path.is_file()
            else None,
            "selection_status": selection.get("status") or "blocked",
        }

    target_frame.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_frame, target_frame)
    return {
        "status": "completed",
        "source_kind": "video_future_frame",
        "source_path": str(candidate),
        "selected_frame_index": selection.get("selected_frame_index"),
        "future_frame_selected": True,
        "frame_selection_policy": "prefer_signal_valid_else_earliest_decodable_future_frame",
        "selection_manifest_path": str(selection_manifest_path)
        if selection_manifest_path.is_file()
        else None,
        "selection_status": selection.get("status") or "completed",
        "selection_quality_status": selection.get("selection_quality_status"),
        "selected_frame_signal_blockers": list(
            selection.get("selected_frame_signal_blockers") or []
        ),
        "extraction_method": selection.get("extraction_method"),
        "materialized_frame_path": str(target_frame),
        "claim_boundary": {
            "selected_frame_is_generated_next_observation_candidate": True,
            "visual_signal_gate_is_not_task_success_evidence": True,
            "scene_or_task_specific_pixels_used": False,
        },
    }


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
            future_materialization = _materialize_future_frame_from_video(candidate, target_frame)
            if future_materialization.get("status") == "completed":
                return future_materialization
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
                        "selected_frame_index": 0,
                        "future_frame_selected": False,
                        "frame_selection_policy": "video_first_frame",
                        "future_frame_selection_status": future_materialization.get("status"),
                        "future_frame_selection_blockers": list(
                            future_materialization.get("blockers") or []
                        ),
                        "future_frame_selection_manifest_path": future_materialization.get(
                            "selection_manifest_path"
                        ),
                        "materialized_frame_path": str(target_frame),
                        "claim_boundary": {
                            "video_first_frame_may_be_seed_or_minimal_motion_frame": True,
                            "future_frame_rollout_quality_not_proven_by_this_materialization": True,
                        },
                    }
    return {
        "status": "blocked",
        "blockers": ["wam_output_missing_materializable_frame_or_video"],
    }


def _frame_history_append_unique(history: list[str], frame_path: Path) -> list[str]:
    if not frame_path.is_file():
        return history
    resolved = str(frame_path.expanduser().resolve())
    if resolved not in history:
        history.append(resolved)
    return history


def _frame_history_window(history: Sequence[str], *, max_frames: int) -> list[str]:
    limit = max(2, int(max_frames or 2))
    return [str(item) for item in list(history)[-limit:] if _string(item)]


def _generated_next_observation_visual_gate(
    *,
    source_frame: Path,
    generated_frame: Path,
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "persistent_wam_generated_next_observation_visual_gate.v1",
        "status": "blocked",
        "source_frame_path": str(source_frame),
        "generated_frame_path": str(generated_frame),
        "materialization_source_kind": materialization.get("source_kind"),
        "materialization_selection_quality_status": materialization.get("selection_quality_status"),
        "blockers": [],
    }
    try:
        from blueprint_pipeline.wam_generated_video_review import (
            _frame_visual_stats,
            _generated_frame_quality_blockers,
        )
    except Exception as exc:
        result["blockers"] = [f"generated_frame_visual_gate_import_failed:{type(exc).__name__}"]
        return result

    source_stats = _frame_visual_stats(source_frame, role="source_policy_observation")
    generated_stats = _frame_visual_stats(
        generated_frame,
        role="generated_next_observation",
        source_frame_stats=source_stats if source_stats.get("status") == "completed" else None,
    )
    blockers: list[str] = []
    if source_stats.get("status") != "completed":
        blockers.extend(
            str(item) for item in source_stats.get("blockers") or ["source_frame_visual_stats_blocked"]
        )
    if generated_stats.get("status") != "completed":
        blockers.extend(
            str(item)
            for item in generated_stats.get("blockers") or ["generated_frame_visual_stats_blocked"]
        )
    else:
        blockers.extend(_generated_frame_quality_blockers([generated_stats]))

    source_kind = _string(materialization.get("source_kind"))
    if source_kind == "video_first_frame":
        blockers.append("wam_generated_next_observation_used_video_first_frame_fallback")
    if _string(materialization.get("selection_quality_status")) == "degraded_visual_signal":
        blockers.append("wam_generated_next_observation_future_frame_degraded_visual_signal")
    for blocker in materialization.get("selected_frame_signal_blockers") or []:
        if _string(blocker):
            blockers.append(_string(blocker))

    result.update(
        {
            "status": "passed_visual_quality_gate" if not blockers else "failed_visual_quality_gate",
            "source_frame_stats": source_stats,
            "generated_frame_stats": generated_stats,
            "blockers": sorted(set(blockers)),
            "claim_boundary": {
                "visual_gate_is_source_relative_sanity_not_task_success": True,
                "visual_gate_blocks_autoregressive_policy_feedback": True,
                "scene_or_task_specific_pixels_hardcoded": False,
            },
        }
    )
    return result


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


def _float_config(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return float(default)


ALLOW_SEED_DERIVED_SKELETON_FOR_ACTION_WAM_ENV = (
    "BLUEPRINT_ALLOW_SEED_DERIVED_SKELETON_FOR_ACTION_CONDITIONED_WAM"
)
PROJECTED_SKELETON_TRACE_KEYS = (
    "g1_projected_skeleton_trace_jsonl",
    "projected_skeleton_trace_path",
    "policy_derived_projected_skeleton_trace_path",
    "policy_action_projected_skeleton_trace_path",
)


def _action_payload_present(action: Mapping[str, Any]) -> bool:
    if not action:
        return False
    for key in ("action_chunk", "sonic_latent_action", "motion_token"):
        value = action.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return bool(value)
        if _string(value):
            return True
    if action.get("unitree_groot_n17_sonic_action_chunk_present"):
        return True
    if action.get("unitree_groot_n17_sonic_action_payload_present"):
        return True
    for key in ("hand_targets", "left_hand_joints", "right_hand_joints"):
        if action.get(key):
            return True
    return False


def _projected_skeleton_trace_claim_boundary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, Mapping):
                return _mapping(row.get("claim_boundary"))
    except Exception:
        return {}
    return {}


def _policy_derived_projected_skeleton_trace(path: Path) -> bool:
    claim_boundary = _projected_skeleton_trace_claim_boundary(path)
    policy_action_delta_applied_to_seed_geometry = bool(
        claim_boundary.get("policy_action_delta_applied_to_seed_geometry")
        or claim_boundary.get("isaac_seed_geometry_action_projection")
    )
    return bool(
        claim_boundary.get("policy_derived_action_conditioning")
        and not claim_boundary.get("not_a_learned_robot_policy_action")
        and (
            not claim_boundary.get("projected_skeleton_trace_derived_from_seed_render_geometry")
            or policy_action_delta_applied_to_seed_geometry
        )
        and not claim_boundary.get(
            "temporal_rows_are_target_conditioning_from_resolved_affordance_projection"
        )
    )


def _ranking_safe_policy_projected_skeleton_trace(path: Path) -> bool:
    claim_boundary = _projected_skeleton_trace_claim_boundary(path)
    scene_faithful_bridge_used = bool(
        claim_boundary.get("scene_faithful_isaac_policy_action_projection_bridge_used")
        or claim_boundary.get("blueprint_simulator_only_isaac_action_projection_bridge_used")
        or claim_boundary.get("simulator_only_mujoco_action_trace_bridge_used")
        or claim_boundary.get("official_wbc_or_sim_bridge_used")
    )
    return bool(
        _policy_derived_projected_skeleton_trace(path)
        and not claim_boundary.get("nominal_kinematic_projection_without_scene_or_wbc_bridge")
        and scene_faithful_bridge_used
    )


SONIC_ACTION_FRAME_DIM = 78
SONIC_LATENT_FRAME_DIM = 64
SONIC_SIM2SIM_UPPER_BODY_SLOT_COUNT = 28
NOMINAL_POLICY_ACTION_TRACE_ENV = "BLUEPRINT_ENABLE_NOMINAL_POLICY_ACTION_SKELETON_TRACE"


def _flatten_numbers(value: Any) -> list[float]:
    numbers: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            value = float(item)
            if value == value and value not in {float("inf"), float("-inf")}:
                numbers.append(value)

    visit(value)
    return numbers


def _mean_abs(values: Sequence[float]) -> float:
    return sum(abs(float(item)) for item in values) / len(values) if values else 0.0


def _clip(value: float, low: float, high: float) -> float:
    return min(float(high), max(float(low), float(value)))


def _vec3(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    result: list[float] = []
    for item in list(value)[:3]:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            return None
    if len(result) != 3:
        return None
    return result


def _vsub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(a[index]) - float(b[index]) for index in range(3)]


def _vadd(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(a[index]) + float(b[index]) for index in range(3)]


def _vscale(a: Sequence[float], scale: float) -> list[float]:
    return [float(a[index]) * float(scale) for index in range(3)]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[index]) * float(b[index]) for index in range(3))


def _cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in a[:3]))


def _normalize(a: Sequence[float], fallback: Sequence[float] | None = None) -> list[float]:
    length = _norm(a)
    if length <= 1e-9:
        return [float(item) for item in (fallback or [1.0, 0.0, 0.0])[:3]]
    return [float(item) / length for item in a[:3]]


def _rotate_about_axis(vector: Sequence[float], axis: Sequence[float], radians: float) -> list[float]:
    unit_axis = _normalize(axis)
    cos_t = math.cos(float(radians))
    sin_t = math.sin(float(radians))
    cross = _cross(unit_axis, vector)
    dot = _dot(unit_axis, vector)
    return [
        float(vector[index]) * cos_t
        + cross[index] * sin_t
        + unit_axis[index] * dot * (1.0 - cos_t)
        for index in range(3)
    ]


def _rotate_many(
    vector: Sequence[float],
    rotations: Sequence[tuple[Sequence[float], float]],
) -> list[float]:
    result = [float(item) for item in vector[:3]]
    for axis, radians in rotations:
        result = _rotate_about_axis(result, axis, radians)
    return result


def _visual_dimension(observation: Mapping[str, Any], key: str, default: int) -> int:
    visual = _mapping(observation.get("visual_observation"))
    for value in (visual.get(key), observation.get(key), os.environ.get(f"BLUEPRINT_OSCAR_WAM_{key.upper()}")):
        try:
            parsed = int(float(value))
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return int(default)


def _geometry_sidecar_path_from_observation(observation: Mapping[str, Any]) -> Path | None:
    visual = _mapping(observation.get("visual_observation"))
    for value in (
        observation.get("manipulation_pov_geometry_path"),
        observation.get("isaac_manipulation_pov_geometry_path"),
        observation.get("seed_geometry_path"),
        observation.get("geometry_path"),
        visual.get("manipulation_pov_geometry_path"),
        visual.get("isaac_manipulation_pov_geometry_path"),
        visual.get("seed_geometry_path"),
        visual.get("geometry_path"),
    ):
        text = _string(value)
        if not text:
            continue
        path = Path(text).expanduser()
        if path.is_file():
            return path.resolve()
    return None


def _first_isaac_geometry_frame(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    frames = payload.get("frames") if isinstance(payload, Mapping) else None
    if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes, bytearray)):
        for frame in frames:
            if isinstance(frame, Mapping):
                return dict(frame)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _camera_projection_from_isaac_geometry(
    *,
    geometry_frame: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    camera_meta = _mapping(geometry_frame.get("camera_meta"))
    eye = _vec3(camera_meta.get("camera_eye_xyz"))
    target = _vec3(camera_meta.get("camera_target_xyz"))
    if eye is None or target is None:
        return {"available": False, "blockers": ["isaac_geometry_camera_pose_missing"]}
    viewport = camera_meta.get("viewport_size_px")
    width = _visual_dimension(observation, "width", 640)
    height = _visual_dimension(observation, "height", 480)
    if isinstance(viewport, Sequence) and not isinstance(viewport, (str, bytes, bytearray)):
        try:
            if int(viewport[0]) > 0 and int(viewport[1]) > 0:
                width = int(viewport[0])
                height = int(viewport[1])
        except (TypeError, ValueError, IndexError):
            pass
    try:
        vfov_deg = float(camera_meta.get("camera_vfov_deg") or 90.0)
    except (TypeError, ValueError):
        vfov_deg = 90.0
    forward = _normalize(_vsub(target, eye), fallback=[1.0, 0.0, 0.0])
    world_up = [0.0, 0.0, 1.0]
    right = _normalize(_cross(forward, world_up), fallback=[0.0, -1.0, 0.0])
    up = _normalize(_cross(right, forward), fallback=world_up)
    focal_y = 0.5 * float(height) / math.tan(math.radians(vfov_deg) / 2.0)
    return {
        "available": True,
        "camera_eye_xyz": eye,
        "camera_target_xyz": target,
        "forward": forward,
        "right": right,
        "up": up,
        "width": width,
        "height": height,
        "vfov_deg": vfov_deg,
        "focal_y_px": focal_y,
    }


def _project_isaac_world_point(
    world_xyz: Sequence[float],
    *,
    camera: Mapping[str, Any],
) -> dict[str, Any]:
    if not camera.get("available"):
        return {"available": False, "blockers": list(camera.get("blockers") or [])}
    eye = camera.get("camera_eye_xyz")
    if not isinstance(eye, Sequence):
        return {"available": False, "blockers": ["isaac_geometry_camera_eye_missing"]}
    delta = _vsub(world_xyz, eye)
    depth = _dot(delta, camera.get("forward") or [1.0, 0.0, 0.0])
    if depth <= 1e-6:
        return {
            "available": False,
            "depth_m": round(depth, 6),
            "blockers": ["isaac_geometry_projection_behind_camera"],
        }
    width = int(camera.get("width") or 640)
    height = int(camera.get("height") or 480)
    focal_y = float(camera.get("focal_y_px") or 0.0)
    u = width * 0.5 + focal_y * _dot(delta, camera.get("right") or [0.0, -1.0, 0.0]) / depth
    v = height * 0.5 - focal_y * _dot(delta, camera.get("up") or [0.0, 0.0, 1.0]) / depth
    return {
        "available": True,
        "u_px": round(u, 2),
        "v_px": round(v, 2),
        "image_width_px": width,
        "image_height_px": height,
        "depth_m": round(depth, 6),
        "inside_image": bool(0.0 <= u < width and 0.0 <= v < height),
    }


def _isaac_seed_arm_points_by_arm(geometry_frame: Mapping[str, Any]) -> dict[str, dict[str, list[float]]]:
    camera_meta = _mapping(geometry_frame.get("camera_meta"))
    raw_by_arm = _mapping(camera_meta.get("arm_link_points_by_arm_xyz"))
    result: dict[str, dict[str, list[float]]] = {}
    for arm in ("left", "right"):
        raw_points = _mapping(raw_by_arm.get(arm))
        points: dict[str, list[float]] = {}
        for role in ("shoulder", "elbow", "wrist", "hand"):
            value = _vec3(raw_points.get(role))
            if value is not None:
                points[role] = value
        if points:
            result[arm] = points
    return result


def _sonic_value_to_joint_delta(value: float, scale_rad: float) -> float:
    return _clip(float(value), -1.0, 1.0) * float(scale_rad)


def _sidecar_arm_chain_fk_points(
    *,
    arm: str,
    points: Mapping[str, Sequence[float]],
    action_values: Sequence[float],
    camera: Mapping[str, Any],
) -> dict[str, list[float]]:
    shoulder = points.get("shoulder")
    elbow = points.get("elbow")
    wrist = points.get("wrist")
    hand = points.get("hand")
    if shoulder is None or elbow is None or wrist is None or hand is None:
        return {}
    side = -1.0 if arm == "left" else 1.0
    upper_seed = _vsub(elbow, shoulder)
    forearm_seed = _vsub(wrist, elbow)
    hand_seed = _vsub(hand, wrist)
    arm_axis = _normalize(_vsub(hand, shoulder), fallback=camera.get("forward") or [1.0, 0.0, 0.0])
    camera_right = camera.get("right") or [0.0, -1.0, 0.0]
    camera_up = camera.get("up") or [0.0, 0.0, 1.0]
    camera_forward = camera.get("forward") or [1.0, 0.0, 0.0]
    values = list(action_values) + [0.0] * max(0, 7 - len(action_values))
    shoulder_rotations = [
        (camera_right, _sonic_value_to_joint_delta(values[0], 0.38)),
        (camera_up, _sonic_value_to_joint_delta(values[1] * side, 0.32)),
        (arm_axis, _sonic_value_to_joint_delta(values[2], 0.22)),
    ]
    elbow_axis = _normalize(_cross(upper_seed, camera_forward), fallback=camera_right)
    elbow_rotations = [
        (elbow_axis, _sonic_value_to_joint_delta(values[3], 0.55)),
    ]
    wrist_rotations = [
        (camera_right, _sonic_value_to_joint_delta(values[4], 0.24)),
        (camera_up, _sonic_value_to_joint_delta(values[5] * side, 0.20)),
        (arm_axis, _sonic_value_to_joint_delta(values[6], 0.18)),
    ]
    upper = _rotate_many(upper_seed, shoulder_rotations)
    elbow_world = _vadd(shoulder, upper)
    forearm = _rotate_many(
        _rotate_many(forearm_seed, shoulder_rotations),
        elbow_rotations,
    )
    wrist_world = _vadd(elbow_world, forearm)
    hand_segment = _rotate_many(
        _rotate_many(
            _rotate_many(hand_seed, shoulder_rotations),
            elbow_rotations,
        ),
        wrist_rotations,
    )
    hand_world = _vadd(wrist_world, hand_segment)
    return {
        "shoulder": [float(item) for item in shoulder[:3]],
        "elbow": elbow_world,
        "wrist": wrist_world,
        "hand": hand_world,
    }


def _materialize_isaac_geometry_policy_action_projected_skeleton_trace(
    *,
    work_dir: Path | None,
    observation: Mapping[str, Any],
    source_policy_action: Mapping[str, Any],
) -> Path | None:
    if work_dir is None:
        return None
    geometry_path = _geometry_sidecar_path_from_observation(observation)
    if geometry_path is None:
        return None
    values = _flatten_numbers(source_policy_action.get("action_chunk"))
    if not values or len(values) % SONIC_ACTION_FRAME_DIM != 0:
        return None
    geometry_frame = _first_isaac_geometry_frame(geometry_path)
    camera = _camera_projection_from_isaac_geometry(
        geometry_frame=geometry_frame,
        observation=observation,
    )
    arm_points = _isaac_seed_arm_points_by_arm(geometry_frame)
    if not camera.get("available") or not arm_points:
        return None
    frames = [
        values[index : index + SONIC_ACTION_FRAME_DIM]
        for index in range(0, len(values), SONIC_ACTION_FRAME_DIM)
    ]
    if not any(
        abs(item) > 1e-9
        for frame in frames
        for item in frame[:SONIC_SIM2SIM_UPPER_BODY_SLOT_COUNT]
    ):
        return None
    trace_path = work_dir / "policy_action_isaac_geometry_projected_skeleton_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    segments = [
        {"from": "left_shoulder", "to": "left_elbow"},
        {"from": "left_elbow", "to": "left_wrist"},
        {"from": "left_wrist", "to": "left_hand"},
        {"from": "right_shoulder", "to": "right_elbow"},
        {"from": "right_elbow", "to": "right_wrist"},
        {"from": "right_wrist", "to": "right_hand"},
    ]
    with trace_path.open("w", encoding="utf-8") as handle:
        for index, frame in enumerate(frames):
            landmarks: list[dict[str, Any]] = []
            for arm, offset in (("left", 0), ("right", 14)):
                points = arm_points.get(arm, {})
                if not points:
                    continue
                arm_slice = frame[offset : offset + 14]
                fk_points = _sidecar_arm_chain_fk_points(
                    arm=arm,
                    points=points,
                    action_values=arm_slice[:7],
                    camera=camera,
                )
                for role in ("shoulder", "elbow", "wrist", "hand"):
                    seed_point = points.get(role)
                    if seed_point is None:
                        continue
                    world = fk_points.get(role) or [float(value) for value in seed_point[:3]]
                    landmarks.append(
                        {
                            "landmark_id": f"{arm}_{role}",
                            "arm": arm,
                            "link_role": role,
                            "seed_world_xyz_m": [round(value, 6) for value in seed_point],
                            "world_xyz_m": [round(value, 6) for value in world],
                            "image_projection": _project_isaac_world_point(world, camera=camera),
                        }
                    )
            projected_count = sum(
                1
                for landmark in landmarks
                if _mapping(landmark.get("image_projection")).get("available")
            )
            handle.write(
                json.dumps(
                    {
                        "schema_version": "blueprint.g1.isaac_geometry_policy_action_projected_skeleton.v1",
                        "status": "completed" if projected_count else "warning_no_projected_landmarks",
                        "frame_index": index,
                        "step": index,
                        "camera": _string(_mapping(observation.get("visual_observation")).get("camera_id"))
                        or _string(geometry_frame.get("camera"))
                        or "head_pov",
                        "image_width_px": int(camera.get("width") or 0),
                        "image_height_px": int(camera.get("height") or 0),
                        "source_geometry_path": str(geometry_path),
                        "landmarks": landmarks,
                        "segments": segments,
                        "projected_landmark_count": projected_count,
                        "kinematic_chain": {
                            "source": "isaac_manipulation_pov_geometry_arm_link_points",
                            "projection_method": "isaac_camera_sidecar_pinhole_projection",
                            "action_delta_method": "sonic_action_chunk_sidecar_upper_body_fk_joint_deltas",
                            "sidecar_kinematic_chain_fk_executed": True,
                            "urdf_fk_solver_executed": False,
                            "full_g1_urdf_fk_executed": False,
                            "official_groot_wholebodycontrol_sim2sim_executed": False,
                        },
                        "claim_boundary": {
                            "policy_derived_action_conditioning": True,
                            "not_a_learned_robot_policy_action": False,
                            "policy_action_delta_applied_to_seed_geometry": True,
                            "isaac_seed_geometry_action_projection": True,
                            "isaac_policy_action_projection_bridge_used": True,
                            "scene_faithful_isaac_policy_action_projection_bridge_used": True,
                            "projected_skeleton_trace_derived_from_seed_render_geometry": True,
                            "temporal_rows_are_target_conditioning_from_resolved_affordance_projection": False,
                            "nominal_kinematic_projection_without_scene_or_wbc_bridge": False,
                            "official_wbc_or_sim_bridge_used": False,
                            "blueprint_simulator_only_isaac_action_projection_bridge_used": True,
                            "official_groot_wholebodycontrol_sim2sim_used": False,
                            "uses_isaac_seed_arm_link_geometry": True,
                            "uses_isaac_sidecar_link_landmarks_not_hand_drawn_screen_axes": True,
                            "sidecar_kinematic_chain_fk_solver_used": True,
                            "full_g1_urdf_fk_solver_used": False,
                            "sonic_action_delta_is_heuristic_reach_lift_not_official_wbc": False,
                            "sonic_action_delta_is_heuristic_joint_delta_not_official_wbc": True,
                            "dynamic_scene_coordinates_from_artifact_not_source_code": True,
                            "simulated_state_not_physical_robot_sensor_evidence": True,
                            "not_task_success_proof": True,
                            "not_physical_robot_sensor_proof": True,
                            "scene_or_task_specific_pixels_used": True,
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    if isinstance(source_policy_action, dict):
        source_policy_action["policy_action_projected_skeleton_trace_path"] = str(trace_path)
        source_policy_action["isaac_geometry_policy_action_projected_skeleton_trace_path"] = str(
            trace_path
        )
    return trace_path


def _nominal_policy_action_projected_landmarks(
    frame: Sequence[float],
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    left = list(frame[:14])
    right = list(frame[14:28])
    left_reach = _clip(0.18 + _mean_abs(left[:7]) * 0.75, 0.18, 0.58)
    right_reach = _clip(0.18 + _mean_abs(right[:7]) * 0.75, 0.18, 0.58)
    left_lift = _clip(_mean_abs(left[7:14]) * 0.35, 0.0, 0.22)
    right_lift = _clip(_mean_abs(right[7:14]) * 0.35, 0.0, 0.22)

    def point(landmark_id: str, x_frac: float, y_frac: float) -> dict[str, Any]:
        return {
            "landmark_id": landmark_id,
            "image_projection": {
                "available": True,
                "u_px": round(_clip(x_frac, 0.02, 0.98) * width, 2),
                "v_px": round(_clip(y_frac, 0.02, 0.98) * height, 2),
                "image_width_px": width,
                "image_height_px": height,
                "inside_image": True,
                "coordinate_space": "source_policy_observation_pixels",
            },
        }

    return [
        point("left_shoulder", 0.40, 0.94),
        point("left_elbow", 0.38 - left_reach * 0.05, 0.82 - left_reach * 0.18 - left_lift),
        point("left_wrist", 0.36 - left_reach * 0.08, 0.70 - left_reach * 0.26 - left_lift),
        point("left_hand", 0.34 - left_reach * 0.10, 0.59 - left_reach * 0.32 - left_lift),
        point("right_shoulder", 0.60, 0.94),
        point("right_elbow", 0.62 + right_reach * 0.05, 0.82 - right_reach * 0.18 - right_lift),
        point("right_wrist", 0.64 + right_reach * 0.08, 0.70 - right_reach * 0.26 - right_lift),
        point("right_hand", 0.66 + right_reach * 0.10, 0.59 - right_reach * 0.32 - right_lift),
    ]


def _materialize_nominal_policy_action_projected_skeleton_trace(
    *,
    work_dir: Path | None,
    observation: Mapping[str, Any],
    source_policy_action: Mapping[str, Any],
) -> Path | None:
    if work_dir is None:
        return None
    if not _truthy(os.environ.get(NOMINAL_POLICY_ACTION_TRACE_ENV, "1")):
        return None
    values = _flatten_numbers(source_policy_action.get("action_chunk"))
    if not values or len(values) % SONIC_ACTION_FRAME_DIM != 0:
        return None
    frames = [
        values[index : index + SONIC_ACTION_FRAME_DIM]
        for index in range(0, len(values), SONIC_ACTION_FRAME_DIM)
    ]
    if not any(
        abs(item) > 1e-9
        for frame in frames
        for item in frame[:SONIC_SIM2SIM_UPPER_BODY_SLOT_COUNT]
    ):
        return None
    width = _visual_dimension(observation, "width", 640)
    height = _visual_dimension(observation, "height", 480)
    trace_path = work_dir / "policy_action_nominal_projected_skeleton_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    segments = [
        {"from": "left_shoulder", "to": "left_elbow"},
        {"from": "left_elbow", "to": "left_wrist"},
        {"from": "left_wrist", "to": "left_hand"},
        {"from": "right_shoulder", "to": "right_elbow"},
        {"from": "right_elbow", "to": "right_wrist"},
        {"from": "right_wrist", "to": "right_hand"},
    ]
    with trace_path.open("w", encoding="utf-8") as handle:
        for index, frame in enumerate(frames):
            landmarks = _nominal_policy_action_projected_landmarks(
                frame,
                width=width,
                height=height,
            )
            handle.write(
                json.dumps(
                    {
                        "schema_version": "blueprint.g1.nominal_policy_action_projected_skeleton.v1",
                        "status": "completed",
                        "frame_index": index,
                        "step": index,
                        "camera": _string(_mapping(observation.get("visual_observation")).get("camera_id"))
                        or "head_pov",
                        "image_width_px": width,
                        "image_height_px": height,
                        "source_image_width_px": width,
                        "source_image_height_px": height,
                        "coordinate_space": "source_policy_observation_pixels",
                        "landmarks": landmarks,
                        "segments": segments,
                        "projected_landmark_count": len(landmarks),
                        "claim_boundary": {
                            "policy_derived_action_conditioning": True,
                            "not_a_learned_robot_policy_action": False,
                            "nominal_kinematic_projection_without_scene_or_wbc_bridge": True,
                            "official_wbc_or_sim_bridge_used": False,
                            "simulated_state_not_physical_robot_sensor_evidence": True,
                            "not_task_success_proof": True,
                            "not_physical_robot_sensor_proof": True,
                            "scene_or_task_specific_pixels_used": False,
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    if isinstance(source_policy_action, dict):
        source_policy_action["policy_action_projected_skeleton_trace_path"] = str(trace_path)
    return trace_path


def _projected_skeleton_trace_candidates(
    observation: Mapping[str, Any],
    auxiliary_observation: Mapping[str, Any],
    source_policy_action: Mapping[str, Any],
) -> list[Path]:
    visual = _mapping(observation.get("visual_observation"))
    action_conditioning = _mapping(auxiliary_observation.get("action_conditioning"))
    candidates: list[Path] = []
    for value in (
        *(source_policy_action.get(key) for key in PROJECTED_SKELETON_TRACE_KEYS),
        source_policy_action.get("projected_hand_keypoint_trace_path"),
        *(observation.get(key) for key in PROJECTED_SKELETON_TRACE_KEYS),
        *(visual.get(key) for key in PROJECTED_SKELETON_TRACE_KEYS),
        action_conditioning.get("projected_skeleton_trace_path"),
        action_conditioning.get("projected_hand_keypoint_trace_path"),
    ):
        text = _string(value)
        if text:
            candidates.append(Path(text).expanduser())
    return candidates


def _strip_projected_skeleton_conditioning(
    observation: Mapping[str, Any],
    auxiliary_observation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sanitized_observation = json.loads(json.dumps(dict(observation)))
    sanitized_visual = _mapping(sanitized_observation.get("visual_observation"))
    for key in PROJECTED_SKELETON_TRACE_KEYS:
        sanitized_observation.pop(key, None)
        sanitized_visual.pop(key, None)
    sanitized_observation["visual_observation"] = sanitized_visual

    sanitized_auxiliary = json.loads(json.dumps(dict(auxiliary_observation)))
    action_conditioning = _mapping(sanitized_auxiliary.get("action_conditioning"))
    projected_trace_removed = False
    for key in ("projected_skeleton_trace_path", "projected_hand_keypoint_trace_path"):
        projected_trace_removed = projected_trace_removed or key in action_conditioning
        action_conditioning.pop(key, None)
        action_conditioning.pop(f"local_{key}_omitted_from_runtime_manifest", None)
    if projected_trace_removed:
        action_conditioning["projected_trace_removed_for_policy_ranking_safety"] = True
    if action_conditioning:
        sanitized_auxiliary["action_conditioning"] = action_conditioning
    return sanitized_observation, sanitized_auxiliary


def _prepare_action_conditioned_wam_inputs(
    *,
    observation: Mapping[str, Any],
    auxiliary_observation: Mapping[str, Any],
    auxiliary_manifest_path: str,
    source_policy_action: Mapping[str, Any],
    work_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    action_present = _action_payload_present(source_policy_action)
    geometry_trace = (
        _materialize_isaac_geometry_policy_action_projected_skeleton_trace(
            work_dir=work_dir,
            observation=observation,
            source_policy_action=source_policy_action,
        )
        if action_present
        else None
    )
    nominal_trace = (
        _materialize_nominal_policy_action_projected_skeleton_trace(
            work_dir=work_dir,
            observation=observation,
            source_policy_action=source_policy_action,
        )
        if action_present and geometry_trace is None
        else None
    )
    candidates = _projected_skeleton_trace_candidates(
        observation,
        auxiliary_observation,
        source_policy_action,
    )
    candidate_text = [str(path) for path in candidates]
    policy_derived_candidates = [
        path for path in candidates if path.is_file() and _policy_derived_projected_skeleton_trace(path)
    ]
    ranking_safe_candidates = [
        path for path in policy_derived_candidates if _ranking_safe_policy_projected_skeleton_trace(path)
    ]
    contract = {
        "schema_version": "persistent_wam_policy_action_to_skeleton_contract.v1",
        "status": "not_required" if not action_present else "ready",
        "source_policy_action_present": action_present,
        "projected_skeleton_trace_candidate_count": len(candidates),
        "projected_skeleton_trace_candidates": candidate_text,
        "policy_derived_projected_skeleton_trace_present": bool(policy_derived_candidates),
        "ranking_safe_projected_skeleton_trace_present": bool(ranking_safe_candidates),
        "geometry_anchored_policy_action_projected_skeleton_trace_path": str(geometry_trace)
        if geometry_trace
        else None,
        "nominal_policy_action_projected_skeleton_trace_path": str(nominal_trace) if nominal_trace else None,
        "seed_or_target_projected_skeleton_allowed": _truthy(
            os.environ.get(ALLOW_SEED_DERIVED_SKELETON_FOR_ACTION_WAM_ENV)
        ),
        "policy_ranking_claim_safe": not action_present or bool(ranking_safe_candidates),
        "blockers": [],
        "claim_boundary": {
            "policy_ranking_claim_safe_requires_policy_derived_action_conditioning": True,
            "geometry_anchored_policy_action_projection_is_wam_conditioning_not_ranking_proof": bool(
                geometry_trace
            ),
            "scene_faithful_isaac_policy_action_projection_bridge_is_sim_only_ranking_conditioning": bool(
                geometry_trace
            ),
            "nominal_policy_action_projection_is_wam_conditioning_not_ranking_proof": True,
            "seed_or_target_skeleton_is_visual_smoke_only": True,
            "scene_or_task_specific_pixels_used": bool(geometry_trace),
        },
    }
    if not action_present:
        return dict(observation), dict(auxiliary_observation), auxiliary_manifest_path, contract
    if policy_derived_candidates:
        selected = policy_derived_candidates[0]
        selected_ranking_safe = _ranking_safe_policy_projected_skeleton_trace(selected)
        selected_claim_boundary = _projected_skeleton_trace_claim_boundary(selected)
        selected_geometry_anchored = bool(
            selected_claim_boundary.get("policy_action_delta_applied_to_seed_geometry")
            or selected_claim_boundary.get("isaac_seed_geometry_action_projection")
        )
        contract["status"] = (
            "policy_derived_projected_skeleton_trace_available"
            if selected_ranking_safe
            else "isaac_geometry_anchored_policy_action_projected_skeleton_trace_available"
            if selected_geometry_anchored
            else "nominal_policy_action_projected_skeleton_trace_available"
        )
        contract["selected_projected_skeleton_trace_path"] = str(policy_derived_candidates[0])
        contract["selected_projected_skeleton_trace_policy_ranking_safe"] = selected_ranking_safe
        if not selected_ranking_safe:
            contract["blockers"] = [
                "isaac_seed_geometry_policy_action_projection_without_official_wbc_or_joint_bridge"
                if selected_geometry_anchored
                else "nominal_policy_action_projection_without_scene_or_wbc_bridge"
            ]
            contract["selected_projected_skeleton_trace_claim_boundary"] = selected_claim_boundary
        elif selected_geometry_anchored:
            contract["selected_projected_skeleton_trace_claim_boundary"] = selected_claim_boundary
            claim_boundary = _mapping(contract.get("claim_boundary"))
            claim_boundary[
                "geometry_anchored_policy_action_projection_is_wam_conditioning_not_ranking_proof"
            ] = False
            claim_boundary[
                "scene_faithful_isaac_policy_action_projection_bridge_is_sim_only_ranking_conditioning"
            ] = True
            contract["claim_boundary"] = claim_boundary
        return dict(observation), dict(auxiliary_observation), auxiliary_manifest_path, contract
    if contract["seed_or_target_projected_skeleton_allowed"]:
        contract["status"] = "visual_smoke_seed_or_target_skeleton_allowed"
        contract["policy_ranking_claim_safe"] = False
        contract["blockers"] = [
            "seed_or_target_projected_skeleton_used_for_visual_smoke_not_policy_ranking"
        ]
        return dict(observation), dict(auxiliary_observation), auxiliary_manifest_path, contract
    sanitized_observation, sanitized_auxiliary = _strip_projected_skeleton_conditioning(
        observation,
        auxiliary_observation,
    )
    contract["status"] = (
        "stripped_seed_or_target_projected_skeleton_for_policy_action_conditioning"
        if candidates
        else "no_policy_derived_projected_skeleton_trace_available"
    )
    contract["blockers"] = [
        "policy_action_to_projected_skeleton_decoder_missing_for_ranking_safe_wam"
    ]
    contract["auxiliary_manifest_path_cleared_to_prevent_unsanitized_reload"] = bool(
        auxiliary_manifest_path
    )
    return sanitized_observation, sanitized_auxiliary, "", contract


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
        rgb_context_frame_paths: list[str] = []
        seen_rgb_context_frame_paths: set[str] = set()
        for value in payload.get("rgb_context_frame_paths") or []:
            text = _string(value)
            if not text:
                continue
            path = Path(text).expanduser()
            if not path.is_file():
                continue
            resolved = str(path.resolve())
            if resolved in seen_rgb_context_frame_paths:
                continue
            seen_rgb_context_frame_paths.add(resolved)
            rgb_context_frame_paths.append(resolved)
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
        source_policy_action = _mapping(payload.get("source_policy_action"))
        (
            current_policy_observation,
            auxiliary_observation,
            auxiliary_manifest_path,
            policy_action_to_skeleton_contract,
        ) = _prepare_action_conditioned_wam_inputs(
            observation=current_policy_observation,
            auxiliary_observation=auxiliary_observation,
            auxiliary_manifest_path=auxiliary_manifest_path,
            source_policy_action=source_policy_action,
            work_dir=step_dir,
        )
        step_input = {
            "schema_version": "wam_generation_step_input.v1",
            "generated_at": payload.get("generated_at"),
            "step_index": step_index,
            "wam_evaluator_backend": "persistent_oscar_wam_worker",
            "source_policy_observation_frame_path": str(source_frame),
            "rgb_context_frame_paths": rgb_context_frame_paths,
            "source_policy_action": source_policy_action,
            "current_policy_observation": current_policy_observation,
            "wam_auxiliary_observation_manifest_path": auxiliary_manifest_path or None,
            "auxiliary_observation": auxiliary_observation,
            "policy_action_to_skeleton_contract": policy_action_to_skeleton_contract,
            "requested_output": {
                "next_observation_frame_path": str(target_frame),
                "action_conditioned_generation_required": True,
            },
            "claim_boundary": {
                "wam_generation_is_not_robot_policy": True,
                "physical_robot_sensor_proof": False,
                "policy_action_to_skeleton_contract_is_input_provenance_not_task_success": True,
                "rgb_context_frame_paths_are_real_observation_history": bool(
                    rgb_context_frame_paths
                ),
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
                    num_steps=int(os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_STEPS", "35")),
                    guidance=_float_config("BLUEPRINT_OSCAR_WAM_GUIDANCE", 6.0),
                    num_frames=int(
                        os.environ.get(
                            "BLUEPRINT_OSCAR_WAM_NUM_FRAMES",
                            str(REVIEW_QUALITY_MIN_OSCAR_NUM_FRAMES),
                        )
                    ),
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
        generated_visual_gate: dict[str, Any] = {}
        if completed:
            generated_visual_gate = _generated_next_observation_visual_gate(
                source_frame=source_frame,
                generated_frame=target_frame,
                materialization=materialization,
            )
            _write_json(step_dir / "generated_next_observation_visual_gate.json", generated_visual_gate)
            visual_gate_blockers = [
                str(item) for item in generated_visual_gate.get("blockers") or []
            ]
            if visual_gate_blockers:
                live_blockers.extend(
                    [
                        "persistent_wam_generated_next_observation_visual_quality_failed",
                        *visual_gate_blockers,
                    ]
                )
                completed = False
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
            "generated_next_observation_visual_gate": generated_visual_gate,
            "accepted_for_next_policy_observation": bool(completed),
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
                "generated_world_rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
                "non_ranking_operational_claim_proven": False,
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
        runtime_projected_skeleton_trace = runtime_dir / "seed_conditioning" / "g1_projected_skeleton_trace.jsonl"
        runtime_isaac_scene_context_dir = runtime_dir / "isaac_scene_context"
        runtime_manipulation_pov_geometry = runtime_isaac_scene_context_dir / "manipulation_pov_geometry.json"
        runtime_placement_validation = runtime_isaac_scene_context_dir / "placement_validation.json"
        runtime_task_stance_plan = runtime_isaac_scene_context_dir / "task_stance_plan.json"
        output_isaac_scene_context_dir = output_dir / "isaac_scene_context"
        isaac_scene_context_output_paths = {
            "manipulation_pov_geometry": _copy_runtime_sidecar(
                runtime_manipulation_pov_geometry,
                output_isaac_scene_context_dir / "manipulation_pov_geometry.json",
            ),
            "placement_validation": _copy_runtime_sidecar(
                runtime_placement_validation,
                output_isaac_scene_context_dir / "placement_validation.json",
            ),
            "task_stance_plan": _copy_runtime_sidecar(
                runtime_task_stance_plan,
                output_isaac_scene_context_dir / "task_stance_plan.json",
            ),
        }
        isaac_scene_context_output_paths = {
            key: value for key, value in isaac_scene_context_output_paths.items() if value
        }
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
        if runtime_projected_skeleton_trace.is_file():
            visual["g1_projected_skeleton_trace_jsonl"] = str(runtime_projected_skeleton_trace)
            visual["projected_skeleton_trace_path"] = str(runtime_projected_skeleton_trace)
            observation["g1_projected_skeleton_trace_jsonl"] = str(runtime_projected_skeleton_trace)
            observation["projected_skeleton_trace_path"] = str(runtime_projected_skeleton_trace)
        if runtime_manipulation_pov_geometry.is_file():
            visual["manipulation_pov_geometry_path"] = str(runtime_manipulation_pov_geometry)
            visual["isaac_manipulation_pov_geometry_path"] = str(runtime_manipulation_pov_geometry)
            observation["manipulation_pov_geometry_path"] = str(runtime_manipulation_pov_geometry)
            observation["isaac_manipulation_pov_geometry_path"] = str(runtime_manipulation_pov_geometry)
        if runtime_placement_validation.is_file():
            visual["placement_validation_path"] = str(runtime_placement_validation)
            visual["isaac_scene_manifest_path"] = str(runtime_placement_validation)
            observation["placement_validation_path"] = str(runtime_placement_validation)
            observation["isaac_scene_manifest_path"] = str(runtime_placement_validation)
        if runtime_task_stance_plan.is_file():
            visual["task_stance_plan_path"] = str(runtime_task_stance_plan)
            observation["task_stance_plan_path"] = str(runtime_task_stance_plan)
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
                    if runtime_projected_skeleton_trace.is_file():
                        action_conditioning = _mapping(
                            runtime_auxiliary_payload.get("action_conditioning")
                        )
                        action_conditioning["projected_skeleton_trace_path"] = str(
                            runtime_projected_skeleton_trace
                        )
                        action_conditioning["projected_hand_keypoint_trace_path"] = str(
                            runtime_projected_skeleton_trace
                        )
                        action_conditioning[
                            "projected_trace_runtime_path_rewritten_for_provider_runtime"
                        ] = True
                        runtime_auxiliary_payload["action_conditioning"] = action_conditioning
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
        rgb_context_history: list[str] = []
        _frame_history_append_unique(rgb_context_history, initial_frame)
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
                    "rgb_context_frame_paths": _frame_history_window(
                        rgb_context_history,
                        max_frames=int(
                            os.environ.get(
                                "BLUEPRINT_OSCAR_WAM_NUM_FRAMES",
                                str(REVIEW_QUALITY_MIN_OSCAR_NUM_FRAMES),
                            )
                        ),
                    ),
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
            if not bool(wam_response.get("accepted_for_next_policy_observation", True)):
                blockers.extend(
                    wam_response.get("blockers")
                    or ["persistent_wam_generated_next_observation_not_accepted_for_policy_feedback"]
                )
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
            _frame_history_append_unique(rgb_context_history, current_frame)
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
            "isaac_scene_context_output_paths": isaac_scene_context_output_paths,
            "isaac_scene_context_sidecars_packaged": bool(isaac_scene_context_output_paths),
            "blockers": sorted(set(str(item) for item in blockers if str(item))),
            "duration_seconds": round(time.monotonic() - started, 6),
            "claim_boundary": {
                "simulator_generated_world_proof_only": True,
                "isaac_scene_context_sidecars_are_seed_geometry_metadata": bool(
                    isaac_scene_context_output_paths
                ),
                "isaac_scene_context_sidecars_are_not_policy_action_projection": bool(
                    isaac_scene_context_output_paths
                ),
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
                "generated_world_rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
                "non_ranking_operational_claim_proven": False,
                "accepted_anchor_manipulation_success_proven": False,
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
                "generated_world_rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
                "non_ranking_operational_claim_proven": False,
                "accepted_anchor_manipulation_success_proven": False,
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
	    if ! python - <<'PY'
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
	      return 0
	    fi
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
provider_status="$(python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT"])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    payload = {}
print(str(payload.get("status") or ""))
PY
)"
case "${BLUEPRINT_RUNPOD_KEEPALIVE_AFTER_SUCCESS:-}" in
  1|true|TRUE|yes|YES|on|ON)
    if [ "${entrypoint_rc:-1}" = "0" ] && [ "$provider_status" = "completed" ]; then
      upload_unitree_groot_sonic_output
      python - <<'PY'
import json
import os
import time
from pathlib import Path

status_path = Path(os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR"]) / "runpod_keepalive_after_success_status.json"
status_path.write_text(
    json.dumps(
        {
            "schema_version": "runpod_keepalive_after_success_status.v1",
            "status": "running_after_success",
            "started_at_epoch": round(time.time(), 3),
            "reason": "keep_on_success_requested",
            "raw_secret_values_recorded": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
      echo BLUEPRINT_RUNPOD_KEEPALIVE_AFTER_SUCCESS_STARTED
      while true; do
        sleep "${BLUEPRINT_RUNPOD_KEEPALIVE_AFTER_SUCCESS_HEARTBEAT_SECONDS:-300}"
      done
    fi
    ;;
esac
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
        "oscar_official_release.py",
        "wam_auxiliary_observation.py",
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
    manipulation_pov_geometry_path: str | Path | None = None,
    placement_validation_path: str | Path | None = None,
    task_stance_plan_path: str | Path | None = None,
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
    policy_observation_base_dir = Path(policy_observation_path).expanduser().parent
    explicit_isaac_scene_context = _attach_explicit_isaac_scene_context(
        observation,
        base_dir=policy_observation_base_dir,
        manipulation_pov_geometry_path=manipulation_pov_geometry_path,
        placement_validation_path=placement_validation_path,
        task_stance_plan_path=task_stance_plan_path,
    )
    frame_path = _camera_frame_path(observation)
    visual_profile_settings = _current_wam_visual_profile_settings()
    visual_profile = str(visual_profile_settings["visual_profile"])
    semantic_visual_evidence = _policy_observation_semantic_visual_evidence(
        observation,
        base_dir=policy_observation_base_dir,
    )
    source_visual_qa = assess_source_policy_observation_visual_qa(
        frame_path,
        generated_at=generated,
        target_object_id=_string(observation.get("target_object_id")) or None,
        task_id=_string(observation.get("task_id")) or None,
        object_index=semantic_visual_evidence.get("object_index"),
        eval_ready_task_grounding=semantic_visual_evidence.get("eval_ready_task_grounding"),
        semantic_artifact_base_dir=semantic_visual_evidence.get("semantic_artifact_base_dir"),
        projected_skeleton_trace_path=semantic_visual_evidence.get("projected_skeleton_trace_path"),
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
            projected_skeleton_trace_path=semantic_visual_evidence.get(
                "projected_skeleton_trace_path"
            ),
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
                        projected_skeleton_trace_path=semantic_visual_evidence.get(
                            "projected_skeleton_trace_path"
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
    blockers.extend(str(item) for item in explicit_isaac_scene_context.get("blockers") or [])
    auxiliary_observation_manifest: dict[str, Any] = {}
    runtime_auxiliary_observation_manifest: dict[str, Any] = {}
    auxiliary_observation_manifest_path: Path | None = None
    runtime_auxiliary_observation_manifest_path: Path | None = None
    runtime_projected_skeleton_trace_path: Path | None = None
    runtime_isaac_scene_context_paths: dict[str, str] = {}
    if frame_path is None:
        blockers.append("blocked_missing_policy_visual_observation_frame")
    else:
        shutil.copy2(frame_path, runtime_dir / "initial_policy_frame.png")
        shutil.copy2(frame_path, runtime_dir / "input_frame.png")
        runtime_projected_skeleton_trace_path = _copy_projected_skeleton_trace_for_runtime(
            semantic_visual_evidence.get("projected_skeleton_trace_path"),
            runtime_dir=runtime_dir,
        )
        runtime_isaac_scene_context_paths = _copy_isaac_scene_context_for_runtime(
            semantic_visual_evidence,
            runtime_dir=runtime_dir,
        )
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
            projected_skeleton_trace_path=semantic_visual_evidence.get(
                "projected_skeleton_trace_path"
            ),
        )
        auxiliary_observation_manifest_path = Path(
            str(auxiliary_observation_manifest["manifest_path"])
        )
        runtime_observation = json.loads(json.dumps(observation))
        runtime_visual = _mapping(runtime_observation.get("visual_observation"))
        runtime_visual["camera_frame_path"] = str(runtime_dir / "initial_policy_frame.png")
        runtime_visual["source_image_path"] = str(runtime_dir / "initial_policy_frame.png")
        if runtime_projected_skeleton_trace_path:
            runtime_visual["g1_projected_skeleton_trace_jsonl"] = str(
                runtime_projected_skeleton_trace_path
            )
            runtime_visual["projected_skeleton_trace_path"] = str(
                runtime_projected_skeleton_trace_path
            )
            runtime_observation["g1_projected_skeleton_trace_jsonl"] = str(
                runtime_projected_skeleton_trace_path
            )
            runtime_observation["projected_skeleton_trace_path"] = str(
                runtime_projected_skeleton_trace_path
            )
        if runtime_isaac_scene_context_paths.get("manipulation_pov_geometry_path"):
            runtime_geometry_path = runtime_isaac_scene_context_paths[
                "manipulation_pov_geometry_path"
            ]
            runtime_visual["manipulation_pov_geometry_path"] = runtime_geometry_path
            runtime_visual["isaac_manipulation_pov_geometry_path"] = runtime_geometry_path
            runtime_observation["manipulation_pov_geometry_path"] = runtime_geometry_path
            runtime_observation["isaac_manipulation_pov_geometry_path"] = runtime_geometry_path
        if runtime_isaac_scene_context_paths.get("isaac_scene_manifest_path"):
            runtime_scene_manifest_path = runtime_isaac_scene_context_paths[
                "isaac_scene_manifest_path"
            ]
            runtime_visual["placement_validation_path"] = runtime_scene_manifest_path
            runtime_visual["isaac_scene_manifest_path"] = runtime_scene_manifest_path
            runtime_observation["placement_validation_path"] = runtime_scene_manifest_path
            runtime_observation["isaac_scene_manifest_path"] = runtime_scene_manifest_path
        if runtime_isaac_scene_context_paths.get("task_stance_plan_path"):
            runtime_task_stance_plan_path = runtime_isaac_scene_context_paths[
                "task_stance_plan_path"
            ]
            runtime_visual["task_stance_plan_path"] = runtime_task_stance_plan_path
            runtime_observation["task_stance_plan_path"] = runtime_task_stance_plan_path
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
            projected_skeleton_trace_path=runtime_projected_skeleton_trace_path,
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
        if runtime_projected_skeleton_trace_path:
            action_conditioning = _mapping(
                runtime_auxiliary_observation_manifest.get("action_conditioning")
            )
            action_conditioning["projected_skeleton_trace_path"] = (
                RUNTIME_PROJECTED_SKELETON_TRACE_BUNDLE_PATH
            )
            action_conditioning["projected_hand_keypoint_trace_path"] = (
                RUNTIME_PROJECTED_SKELETON_TRACE_BUNDLE_PATH
            )
            action_conditioning[
                "projected_trace_runtime_path_rewritten_for_provider_bundle"
            ] = True
            runtime_auxiliary_observation_manifest["action_conditioning"] = action_conditioning
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
        if runtime_projected_skeleton_trace_path:
            visual["g1_projected_skeleton_trace_jsonl"] = (
                RUNTIME_PROJECTED_SKELETON_TRACE_BUNDLE_PATH
            )
            visual["projected_skeleton_trace_path"] = (
                RUNTIME_PROJECTED_SKELETON_TRACE_BUNDLE_PATH
            )
            observation["g1_projected_skeleton_trace_jsonl"] = (
                RUNTIME_PROJECTED_SKELETON_TRACE_BUNDLE_PATH
            )
            observation["projected_skeleton_trace_path"] = (
                RUNTIME_PROJECTED_SKELETON_TRACE_BUNDLE_PATH
            )
        if runtime_isaac_scene_context_paths.get("manipulation_pov_geometry_path"):
            bundle_geometry_path = (
                f"{RUNTIME_ISAAC_SCENE_CONTEXT_BUNDLE_DIR}/manipulation_pov_geometry.json"
            )
            visual["manipulation_pov_geometry_path"] = bundle_geometry_path
            visual["isaac_manipulation_pov_geometry_path"] = bundle_geometry_path
            observation["manipulation_pov_geometry_path"] = bundle_geometry_path
            observation["isaac_manipulation_pov_geometry_path"] = bundle_geometry_path
        if runtime_isaac_scene_context_paths.get("isaac_scene_manifest_path"):
            bundle_scene_manifest_path = (
                f"{RUNTIME_ISAAC_SCENE_CONTEXT_BUNDLE_DIR}/placement_validation.json"
            )
            visual["placement_validation_path"] = bundle_scene_manifest_path
            visual["isaac_scene_manifest_path"] = bundle_scene_manifest_path
            observation["placement_validation_path"] = bundle_scene_manifest_path
            observation["isaac_scene_manifest_path"] = bundle_scene_manifest_path
        if runtime_isaac_scene_context_paths.get("task_stance_plan_path"):
            bundle_task_stance_plan_path = (
                f"{RUNTIME_ISAAC_SCENE_CONTEXT_BUNDLE_DIR}/task_stance_plan.json"
            )
            visual["task_stance_plan_path"] = bundle_task_stance_plan_path
            observation["task_stance_plan_path"] = bundle_task_stance_plan_path
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
    runtime_isaac_scene_context_bundle_paths = {
        "manipulation_pov_geometry": (
            f"{RUNTIME_ISAAC_SCENE_CONTEXT_BUNDLE_DIR}/manipulation_pov_geometry.json"
            if runtime_isaac_scene_context_paths.get("manipulation_pov_geometry_path")
            else None
        ),
        "placement_validation": (
            f"{RUNTIME_ISAAC_SCENE_CONTEXT_BUNDLE_DIR}/placement_validation.json"
            if runtime_isaac_scene_context_paths.get("isaac_scene_manifest_path")
            else None
        ),
        "task_stance_plan": (
            f"{RUNTIME_ISAAC_SCENE_CONTEXT_BUNDLE_DIR}/task_stance_plan.json"
            if runtime_isaac_scene_context_paths.get("task_stance_plan_path")
            else None
        ),
    }
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
            "concrete_materialization_quality_blocker_proven": long_review_quality_gate.get(
                "concrete_materialization_quality_blocker_proven"
            ),
            "blockers": list(long_review_quality_gate.get("blockers") or []),
            "materialization_quality_blocker_validation": _mapping(
                long_review_quality_gate.get("materialization_quality_blocker_validation")
            ),
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
        "isaac_scene_context": {
            "status": "available" if runtime_isaac_scene_context_paths else "not_available",
            "explicit_request": explicit_isaac_scene_context,
            "local_source_paths": {
                "manipulation_pov_geometry": semantic_visual_evidence.get(
                    "manipulation_pov_geometry_path"
                ),
                "placement_validation": semantic_visual_evidence.get(
                    "isaac_scene_manifest_path"
                ),
                "task_stance_plan": semantic_visual_evidence.get("task_stance_plan_path"),
            },
            "runtime_paths": runtime_isaac_scene_context_bundle_paths,
            "claim_boundary": {
                "isaac_scene_context_is_geometry_metadata_not_policy_action_projection": True,
                "isaac_scene_context_is_not_task_success_proof": True,
                "scene_or_task_specific_coordinates_hardcoded": False,
            },
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
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
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
            "materialization_quality_blocker_validation": _mapping(
                long_review_quality_gate.get("materialization_quality_blocker_validation")
            ),
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
            "explicit_isaac_scene_context": explicit_isaac_scene_context,
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
                "projected_skeleton_trace": semantic_visual_evidence.get(
                    "projected_skeleton_trace_path"
                ),
                "manipulation_pov_geometry": semantic_visual_evidence.get(
                    "manipulation_pov_geometry_path"
                ),
                "placement_validation": semantic_visual_evidence.get(
                    "isaac_scene_manifest_path"
                ),
                "task_stance_plan": semantic_visual_evidence.get("task_stance_plan_path"),
            },
            "runtime_isaac_scene_context_paths": runtime_isaac_scene_context_bundle_paths,
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
                "generated_world_rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
                "non_ranking_operational_claim_proven": False,
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
        "materialization_quality_blocker_validation": _mapping(
            long_review_quality_gate.get("materialization_quality_blocker_validation")
        ),
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
        "explicit_isaac_scene_context": explicit_isaac_scene_context,
        "semantic_visual_qa_source_paths": {
            "object_index": semantic_visual_evidence.get("object_index_path"),
            "eval_ready_task_grounding": semantic_visual_evidence.get(
                "eval_ready_task_grounding_path"
            ),
            "projected_skeleton_trace": semantic_visual_evidence.get(
                "projected_skeleton_trace_path"
            ),
            "manipulation_pov_geometry": semantic_visual_evidence.get(
                "manipulation_pov_geometry_path"
            ),
            "placement_validation": semantic_visual_evidence.get("isaac_scene_manifest_path"),
            "task_stance_plan": semantic_visual_evidence.get("task_stance_plan_path"),
        },
        "runtime_isaac_scene_context_paths": runtime_isaac_scene_context_bundle_paths,
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
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
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


def _completed_runpod_resume_job(root: str | Path | None) -> Path | None:
    if not root:
        return None
    root_path = Path(root).expanduser()
    candidates = [root_path]
    if root_path.is_dir():
        candidates.extend(sorted((p for p in root_path.iterdir() if p.is_dir()), reverse=True))
    for candidate in candidates:
        runpod_dir = candidate / "runpod_persistent_session_run"
        output_zip = runpod_dir / "runpod_provider_runtime_output.zip"
        poll_manifest_path = runpod_dir / "runpod_wam_async_poll_manifest.json"
        if not output_zip.is_file() or not poll_manifest_path.is_file():
            continue
        poll_manifest = _read_json(poll_manifest_path)
        if poll_manifest.get("status") == "completed":
            return candidate.resolve()
    return None


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
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "non_ranking_operational_claim_proven": False,
        "accepted_anchor_manipulation_success_proven": False,
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
    elif not has_output_zip and poll_manifest.get("pod_status") == "not_found":
        classified_blocker = "runpod_pod_disappeared_before_first_heartbeat"
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
            "classification_is_runtime_diagnostic_not_generated_world_rank_fidelity": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
            "accepted_anchor_manipulation_success_proven": False,
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


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_values(value: Any) -> list[float]:
    if _is_numeric_scalar(value):
        return [float(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values: list[float] = []
        for item in value:
            values.extend(_numeric_values(item))
        return values
    return []


def _numeric_shape(value: Any) -> list[int] | None:
    if _is_numeric_scalar(value):
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    children = [_numeric_shape(item) for item in value]
    if not children:
        return [0]
    if any(child is None for child in children):
        return [len(value)]
    first = children[0]
    if all(child == first for child in children):
        return [len(value), *list(first or [])]
    return [len(value)]


def _numeric_tensor_summary(name: str, value: Any) -> dict[str, Any]:
    values = _numeric_values(value)
    nonzero_count = sum(1 for item in values if abs(item) > 1e-9)
    return {
        "name": name,
        "present": value is not None,
        "shape": _numeric_shape(value),
        "numeric_count": len(values),
        "nonzero_count": nonzero_count,
        "all_zero": bool(values and nonzero_count == 0),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean_abs": sum(abs(item) for item in values) / len(values) if values else None,
    }


SONIC_ACTION_FRAME_DIM = 78
SONIC_LATENT_FRAME_DIM = 64
SONIC_HAND_CONTROL_FRAME_DIM = SONIC_ACTION_FRAME_DIM - SONIC_LATENT_FRAME_DIM
SONIC_SIM2SIM_UPPER_BODY_SLOT_COUNT = 28


def _sonic_action_frame_summary(action_chunk: Any) -> dict[str, Any]:
    values = _numeric_values(action_chunk)
    frame_count = (
        len(values) // SONIC_ACTION_FRAME_DIM
        if values and len(values) % SONIC_ACTION_FRAME_DIM == 0
        else 0
    )
    summary = {
        "present": bool(values),
        "value_count": len(values),
        "expected_frame_dim": SONIC_ACTION_FRAME_DIM,
        "frame_count": frame_count,
        "bridgeable_sonic_action_chunk": frame_count > 0,
        "latent_prefix_dim": SONIC_LATENT_FRAME_DIM,
        "hand_control_tail_dim": SONIC_HAND_CONTROL_FRAME_DIM,
        "hand_control_tail_value_count": 0,
        "hand_control_tail_nonzero_count": 0,
        "hand_control_tail_mean_abs": None,
        "sim2sim_upper_body_slot_count_per_frame": SONIC_SIM2SIM_UPPER_BODY_SLOT_COUNT,
        "sim2sim_upper_body_slot_value_count": 0,
        "sim2sim_upper_body_slot_nonzero_count": 0,
        "sim2sim_upper_body_slot_mean_abs": None,
        "claim_boundary": {
            "sonic_action_chunk_is_policy_output": bool(values),
            "sonic_action_chunk_is_not_a_projected_skeleton": True,
            "sonic_action_chunk_requires_scene_or_wbc_bridge_for_pose_trace": bool(values),
            "sim2sim_upper_body_slot_mapping_is_blueprint_bridge_not_official_wbc": True,
        },
    }
    if frame_count <= 0:
        blockers = []
        if values:
            blockers.append("sonic_action_chunk_not_evenly_divisible_by_78")
        summary["blockers"] = blockers
        return summary
    frames = [
        values[index : index + SONIC_ACTION_FRAME_DIM]
        for index in range(0, len(values), SONIC_ACTION_FRAME_DIM)
    ]
    hand_tail_values = [
        item for frame in frames for item in frame[SONIC_LATENT_FRAME_DIM:SONIC_ACTION_FRAME_DIM]
    ]
    upper_body_values = [
        item for frame in frames for item in frame[:SONIC_SIM2SIM_UPPER_BODY_SLOT_COUNT]
    ]
    hand_tail_nonzero = sum(1 for item in hand_tail_values if abs(item) > 1e-9)
    upper_body_nonzero = sum(1 for item in upper_body_values if abs(item) > 1e-9)
    summary.update(
        {
            "hand_control_tail_value_count": len(hand_tail_values),
            "hand_control_tail_nonzero_count": hand_tail_nonzero,
            "hand_control_tail_mean_abs": (
                sum(abs(item) for item in hand_tail_values) / len(hand_tail_values)
                if hand_tail_values
                else None
            ),
            "sim2sim_upper_body_slot_value_count": len(upper_body_values),
            "sim2sim_upper_body_slot_nonzero_count": upper_body_nonzero,
            "sim2sim_upper_body_slot_mean_abs": (
                sum(abs(item) for item in upper_body_values) / len(upper_body_values)
                if upper_body_values
                else None
            ),
            "blockers": [],
        }
    )
    return summary


def _policy_action_decoding_contract(
    action: Mapping[str, Any],
    *,
    generated_at: str,
) -> dict[str, Any]:
    hand_targets = _mapping(action.get("hand_targets"))
    summaries = {
        "action_chunk": _numeric_tensor_summary("action_chunk", action.get("action_chunk")),
        "sonic_latent_action": _numeric_tensor_summary(
            "sonic_latent_action",
            action.get("sonic_latent_action"),
        ),
        "motion_token": _numeric_tensor_summary("motion_token", action.get("motion_token")),
        "joint_targets": _numeric_tensor_summary("joint_targets", action.get("joint_targets")),
        "arm_targets": _numeric_tensor_summary("arm_targets", action.get("arm_targets")),
        "left_hand_joints": _numeric_tensor_summary(
            "left_hand_joints",
            action.get("left_hand_joints") or hand_targets.get("left_hand_joints"),
        ),
        "right_hand_joints": _numeric_tensor_summary(
            "right_hand_joints",
            action.get("right_hand_joints") or hand_targets.get("right_hand_joints"),
        ),
    }
    sonic_frame_summary = _sonic_action_frame_summary(action.get("action_chunk"))
    latent_present = any(
        summaries[key]["numeric_count"] > 0
        for key in ("action_chunk", "sonic_latent_action", "motion_token")
    )
    bridgeable_sonic_action_chunk = bool(
        sonic_frame_summary.get("bridgeable_sonic_action_chunk")
        and (
            int(sonic_frame_summary.get("hand_control_tail_nonzero_count") or 0) > 0
            or int(sonic_frame_summary.get("sim2sim_upper_body_slot_nonzero_count") or 0) > 0
        )
    )
    decoded_target_keys = ("joint_targets", "arm_targets", "left_hand_joints", "right_hand_joints")
    decoded_target_present = any(summaries[key]["numeric_count"] > 0 for key in decoded_target_keys)
    decoded_target_nonzero = any(
        summaries[key]["nonzero_count"] > 0 for key in decoded_target_keys
    )
    hand_targets_all_zero = bool(
        (summaries["left_hand_joints"]["numeric_count"] or 0) > 0
        and (summaries["right_hand_joints"]["numeric_count"] or 0) > 0
        and summaries["left_hand_joints"]["all_zero"]
        and summaries["right_hand_joints"]["all_zero"]
    )
    if decoded_target_nonzero:
        status = "decoded_control_targets_available"
        blockers: list[str] = []
    elif bridgeable_sonic_action_chunk:
        status = "sonic_action_chunk_available_requires_bridge"
        blockers = ["policy_action_requires_scene_or_wbc_bridge_for_projected_skeleton"]
    elif latent_present:
        status = "blocked_latent_action_without_pose_decoder"
        blockers = ["policy_action_latent_without_decoded_pose_targets"]
    elif decoded_target_present:
        status = "blocked_decoded_control_targets_all_zero"
        blockers = ["policy_action_decoded_control_targets_all_zero"]
    else:
        status = "blocked_no_policy_action_tensor"
        blockers = ["policy_action_tensor_missing"]
    warnings = ["policy_hand_targets_all_zero"] if hand_targets_all_zero else []
    return {
        "schema_version": "persistent_policy_action_decoding_contract.v1",
        "generated_at": generated_at,
        "status": status,
        "action_type": action.get("action_type"),
        "control_fields": action.get("unitree_g1_sonic_control_fields"),
        "latent_action_present": latent_present,
        "decoded_control_target_present": decoded_target_present,
        "decoded_control_target_nonzero": decoded_target_nonzero,
        "bridgeable_sonic_action_chunk": bridgeable_sonic_action_chunk,
        "policy_derived_projected_skeleton_trace_present": False,
        "policy_ranking_claim_safe": False,
        "tensor_summaries": summaries,
        "sonic_action_frame_summary": sonic_frame_summary,
        "warnings": warnings,
        "blockers": blockers,
        "claim_boundary": {
            "policy_action_decoding_contract_is_payload_introspection_only": True,
            "latent_action_is_not_a_decoded_robot_pose_or_skeleton": latent_present,
            "sonic_action_chunk_is_not_a_decoded_pose_or_projected_skeleton": bool(
                sonic_frame_summary.get("present")
            ),
            "sonic_action_chunk_requires_bridge_before_wam_ranking_claim": bool(
                bridgeable_sonic_action_chunk
            ),
            "decoded_control_targets_are_not_task_success_proof": True,
            "policy_ranking_claim_safe_requires_policy_derived_projected_skeleton": True,
            "scene_or_task_specific_pixels_used": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


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


def _write_policy_action_decoding_contract(
    *,
    job: Path,
    action: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    contract = _policy_action_decoding_contract(action, generated_at=generated_at)
    write_json(job / "policy_action_decoding_contract.json", contract)
    return contract


def _mujoco_scene_manifest_candidates(job: Path, extraction_dir: Path) -> list[Path]:
    return [
        job / "mujoco_scene_manifest.json",
        job / "provider_bundle" / "provider_runtime" / "mujoco_scene_manifest.json",
        extraction_dir / "mujoco_scene_manifest.json",
    ]


def _mujoco_scene_manifest_path(job: Path, extraction_dir: Path) -> Path | None:
    for path in _mujoco_scene_manifest_candidates(job, extraction_dir):
        if path.is_file():
            return path
    return None


def _isaac_scene_manifest_candidates(job: Path, extraction_dir: Path) -> list[Path]:
    return [
        job / "isaac_scene_manifest.json",
        job / "placement_validation.json",
        job / "provider_bundle" / "provider_runtime" / "isaac_scene_manifest.json",
        job / "provider_bundle" / "provider_runtime" / "placement_validation.json",
        job
        / "provider_bundle"
        / "provider_runtime"
        / "isaac_scene_context"
        / "placement_validation.json",
        extraction_dir / "isaac_scene_manifest.json",
        extraction_dir / "placement_validation.json",
        extraction_dir / "isaac_scene_context" / "placement_validation.json",
        extraction_dir / "provider_runtime" / "isaac_scene_context" / "placement_validation.json",
    ]


def _isaac_scene_manifest_path(job: Path, extraction_dir: Path) -> Path | None:
    for path in _isaac_scene_manifest_candidates(job, extraction_dir):
        if path.is_file():
            return path
    return None


def _isaac_manipulation_pov_geometry_candidates(job: Path, extraction_dir: Path) -> list[Path]:
    return [
        job / "manipulation_pov_geometry.json",
        job / "provider_bundle" / "provider_runtime" / "manipulation_pov_geometry.json",
        job
        / "provider_bundle"
        / "provider_runtime"
        / "isaac_scene_context"
        / "manipulation_pov_geometry.json",
        extraction_dir / "manipulation_pov_geometry.json",
        extraction_dir / "isaac_scene_context" / "manipulation_pov_geometry.json",
        extraction_dir
        / "provider_runtime"
        / "isaac_scene_context"
        / "manipulation_pov_geometry.json",
    ]


def _isaac_manipulation_pov_geometry_path(job: Path, extraction_dir: Path) -> Path | None:
    for path in _isaac_manipulation_pov_geometry_candidates(job, extraction_dir):
        if path.is_file():
            return path
    return None


def _task_stance_plan_candidates(job: Path, extraction_dir: Path) -> list[Path]:
    return [
        job / "task_stance_plan.json",
        job / "provider_bundle" / "provider_runtime" / "task_stance_plan.json",
        job
        / "provider_bundle"
        / "provider_runtime"
        / "isaac_scene_context"
        / "task_stance_plan.json",
        extraction_dir / "task_stance_plan.json",
        extraction_dir / "isaac_scene_context" / "task_stance_plan.json",
        extraction_dir / "provider_runtime" / "isaac_scene_context" / "task_stance_plan.json",
    ]


def _task_stance_plan_path(job: Path, extraction_dir: Path) -> Path | None:
    for path in _task_stance_plan_candidates(job, extraction_dir):
        if path.is_file():
            return path
    return None


def _scene_bridge_blockers(
    *,
    mujoco_scene_manifest: Path | None,
    isaac_scene_manifest: Path | None,
    isaac_manipulation_pov_geometry: Path | None = None,
) -> list[str]:
    blockers = ["blocked_missing_scene_faithful_policy_action_projection_bridge"]
    if mujoco_scene_manifest is None and isaac_scene_manifest is None:
        blockers.append("blocked_missing_scene_manifest_for_policy_action_bridge")
    if isaac_scene_manifest is not None:
        if isaac_manipulation_pov_geometry is None:
            blockers.append("blocked_missing_isaac_manipulation_pov_geometry_for_action_bridge")
    if mujoco_scene_manifest is None:
        blockers.append("blocked_no_available_mujoco_sim2sim_manifest_for_legacy_bridge")
    return blockers


def _write_policy_action_bridge_readiness(
    *,
    job: Path,
    extraction_dir: Path,
    action_contract: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    scene_manifest = _mujoco_scene_manifest_path(job, extraction_dir)
    isaac_scene_manifest = _isaac_scene_manifest_path(job, extraction_dir)
    isaac_manipulation_pov_geometry = _isaac_manipulation_pov_geometry_path(
        job,
        extraction_dir,
    )
    task_stance_plan = _task_stance_plan_path(job, extraction_dir)
    latent_action_present = bool(action_contract.get("latent_action_present"))
    decoded_targets_nonzero = bool(action_contract.get("decoded_control_target_nonzero"))
    bridgeable_sonic_action_chunk = bool(action_contract.get("bridgeable_sonic_action_chunk"))
    isaac_projection_bridge_available = bool(
        bridgeable_sonic_action_chunk
        and isaac_scene_manifest is not None
        and isaac_manipulation_pov_geometry is not None
    )
    sim2sim_execution = job / "unitree_groot_n17_sonic_sim2sim_execution.json"
    if decoded_targets_nonzero:
        status = "decoded_control_targets_available"
        blockers: list[str] = []
    elif isaac_projection_bridge_available:
        status = "ready_for_isaac_sonic_action_projection_bridge"
        blockers = []
    elif bridgeable_sonic_action_chunk and scene_manifest is not None:
        status = "ready_for_sim2sim_sonic_action_trace_bridge"
        blockers = []
    elif bridgeable_sonic_action_chunk:
        status = "blocked_missing_scene_bridge_for_sonic_action_chunk"
        blockers = _scene_bridge_blockers(
            mujoco_scene_manifest=scene_manifest,
            isaac_scene_manifest=isaac_scene_manifest,
            isaac_manipulation_pov_geometry=isaac_manipulation_pov_geometry,
        )
    elif latent_action_present and scene_manifest is not None:
        status = "ready_for_sim2sim_latent_action_trace_bridge"
        blockers = []
    elif latent_action_present:
        status = "blocked_missing_scene_bridge_for_latent_action"
        blockers = _scene_bridge_blockers(
            mujoco_scene_manifest=scene_manifest,
            isaac_scene_manifest=isaac_scene_manifest,
            isaac_manipulation_pov_geometry=isaac_manipulation_pov_geometry,
        )
    else:
        status = "blocked_no_policy_action_for_bridge"
        blockers = ["policy_action_tensor_missing"]
    payload = {
        "schema_version": "persistent_policy_action_bridge_readiness.v1",
        "generated_at": generated_at,
        "status": status,
        "latent_action_present": latent_action_present,
        "decoded_control_target_nonzero": decoded_targets_nonzero,
        "bridgeable_sonic_action_chunk": bridgeable_sonic_action_chunk,
        "scene_bridge_manifest_path": str(scene_manifest or isaac_scene_manifest)
        if scene_manifest or isaac_scene_manifest
        else None,
        "scene_bridge_manifest_kind": (
            "mujoco" if scene_manifest is not None else "isaac" if isaac_scene_manifest else None
        ),
        "scene_bridge_manifest_candidates": [
            {
                "kind": "mujoco",
                "path": str(path),
                "exists": path.is_file(),
                "supports_current_bridge": True,
            }
            for path in _mujoco_scene_manifest_candidates(job, extraction_dir)
        ]
        + [
            {
                "kind": "isaac",
                "path": str(path),
                "exists": path.is_file(),
                "supports_current_bridge": bool(
                    path.is_file() and isaac_manipulation_pov_geometry is not None
                ),
                "implementation_status": (
                    "implemented"
                    if path.is_file() and isaac_manipulation_pov_geometry is not None
                    else "needs_isaac_manipulation_pov_geometry_sidecar"
                ),
            }
            for path in _isaac_scene_manifest_candidates(job, extraction_dir)
        ],
        "mujoco_scene_manifest_path": str(scene_manifest) if scene_manifest else None,
        "mujoco_scene_manifest_candidates": [
            str(path) for path in _mujoco_scene_manifest_candidates(job, extraction_dir)
        ],
        "isaac_scene_manifest_path": str(isaac_scene_manifest) if isaac_scene_manifest else None,
        "isaac_scene_manifest_candidates": [
            str(path) for path in _isaac_scene_manifest_candidates(job, extraction_dir)
        ],
        "isaac_manipulation_pov_geometry_path": (
            str(isaac_manipulation_pov_geometry) if isaac_manipulation_pov_geometry else None
        ),
        "isaac_manipulation_pov_geometry_candidates": [
            str(path) for path in _isaac_manipulation_pov_geometry_candidates(job, extraction_dir)
        ],
        "task_stance_plan_path": str(task_stance_plan) if task_stance_plan else None,
        "task_stance_plan_candidates": [
            str(path) for path in _task_stance_plan_candidates(job, extraction_dir)
        ],
        "sim2sim_execution_path": str(sim2sim_execution)
        if sim2sim_execution.is_file()
        else None,
        "bridge_candidates": [
            {
                "id": "isaac_g1_policy_action_projection_bridge",
                "kind": "simulator_only_isaac_action_trace_bridge",
                "requires": [
                    "policy_action_to_isaac_g1_articulation_mapping",
                    "isaac_scene_manifest_or_render_state_with_camera_contract",
                    "isaac_manipulation_pov_geometry_with_projectable_g1_arm_links",
                ],
                "available": bool(isaac_projection_bridge_available),
                "implementation_status": "implemented",
                "claim_boundary": (
                    "Required for the Isaac kitchen/fridge lane before policy ranking claims; "
                    "the MuJoCo sim2sim bridge is not a substitute for Isaac task truth."
                ),
            },
            {
                "id": "unitree_groot_n17_sonic_sim2sim_command",
                "kind": "simulator_only_mujoco_action_trace_bridge",
                "requires": [
                    "policy_action_40x78_sonic_action_chunk",
                    "mujoco_scene_manifest_with_unitree_upper_body_actuators",
                ],
                "available": bool(bridgeable_sonic_action_chunk and scene_manifest is not None),
            },
            {
                "id": "official_groot_wholebodycontrol_sim2sim",
                "kind": "official_wbc_sim2sim_launcher",
                "requires": [
                    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT",
                    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_COMMAND",
                ],
                "available": False,
            },
        ],
        "blockers": blockers,
        "claim_boundary": {
            "bridge_readiness_is_not_bridge_execution": True,
            "simulator_only_bridge_is_not_official_wbc_mapping": True,
            "mujoco_bridge_is_legacy_action_trace_support_not_isaac_scene_truth": True,
            "isaac_scene_bridge_required_for_isaac_task_ranking_claim": True,
            "decoded_or_bridged_joint_trace_is_not_task_success_proof": True,
            "policy_ranking_claim_safe_requires_policy_derived_wam_conditioning": True,
            "scene_or_task_specific_pixels_used": bool(isaac_projection_bridge_available),
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(job / "policy_action_bridge_readiness.json", payload)
    return payload


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
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
            "accepted_anchor_manipulation_success_proven": False,
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


def _first_existing_file(*paths: Path) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _step_index_from_wam_step_dir(step_dir: Path) -> int | None:
    try:
        return int(step_dir.name.rsplit("_", 1)[-1])
    except ValueError:
        return None


def _contact_sheet_tile(path: Path, *, label: str, tile_size: tuple[int, int]) -> Any:
    from PIL import Image, ImageDraw

    tile_width, tile_height = tile_size
    label_height = 24
    canvas = Image.new("RGB", tile_size, (245, 245, 245))
    try:
        image = Image.open(path).convert("RGB")
    except OSError:
        image = Image.new("RGB", (tile_width, tile_height - label_height), (30, 30, 30))
    image.thumbnail((tile_width, tile_height - label_height))
    x = (tile_width - image.width) // 2
    y = label_height + (tile_height - label_height - image.height) // 2
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, tile_width, label_height), fill=(20, 20, 20))
    draw.text((8, 6), label[:44], fill=(255, 255, 255))
    return canvas


def _write_wam_input_review_contact_sheet(
    *,
    output_path: Path,
    rows: Sequence[Mapping[str, Any]],
    max_tiles: int = 12,
) -> dict[str, Any]:
    image_items: list[tuple[Path, str]] = []
    for row in rows:
        step_label = f"step {row.get('step_index') or '?'}"
        first_frame_path = Path(_string(row.get("first_frame_path"))).expanduser()
        if first_frame_path.is_file():
            image_items.append((first_frame_path, f"{step_label} WAM first frame"))
        preview_paths = [
            Path(path).expanduser()
            for path in _string_list(row.get("action_conditioning_preview_frame_paths"))
        ]
        for index, preview_path in enumerate(preview_paths[:2]):
            if preview_path.is_file():
                image_items.append(
                    (preview_path, f"{step_label} action conditioning {index + 1}")
                )
    image_items = image_items[:max_tiles]
    if not image_items:
        return {
            "status": "blocked",
            "blockers": ["wam_input_review_no_readable_image_inputs"],
            "contact_sheet_path": None,
            "tile_count": 0,
        }
    try:
        from PIL import Image
    except ImportError:
        return {
            "status": "blocked",
            "blockers": ["pil_not_available_for_wam_input_review_contact_sheet"],
            "contact_sheet_path": None,
            "tile_count": 0,
        }
    tile_size = (360, 292)
    columns = 2
    rows_count = (len(image_items) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_size[0], rows_count * tile_size[1]),
        (230, 230, 230),
    )
    for index, (path, label) in enumerate(image_items):
        tile = _contact_sheet_tile(path, label=label, tile_size=tile_size)
        sheet.paste(tile, ((index % columns) * tile_size[0], (index // columns) * tile_size[1]))
    ensure_dir(output_path.parent)
    sheet.save(output_path, format="JPEG", quality=92)
    return {
        "status": "completed",
        "blockers": [],
        "contact_sheet_path": str(output_path),
        "tile_count": len(image_items),
        "tile_labels": [label for _, label in image_items],
    }


def _wam_input_review_row(step_dir: Path) -> dict[str, Any]:
    bundle_dir = step_dir / "oscar_wam_worker_bundle"
    local_materialization_dir = bundle_dir / "local_input_materialization"
    runtime_dir = bundle_dir / "oscar_wam_provider_bundle" / "provider_runtime"
    local_input_dir = local_materialization_dir / "oscar_input"
    runtime_input_dir = runtime_dir / "oscar_input"
    first_frame = _first_existing_file(
        runtime_input_dir / "first_frame.png",
        local_input_dir / "first_frame.png",
    )
    rgb_context = _first_existing_file(
        runtime_input_dir / "rgb_context.mp4",
        local_input_dir / "rgb_context.mp4",
    )
    action_conditioning = _first_existing_file(
        runtime_input_dir / "blueprint_proxy_skeleton_conditioning.mp4",
        local_input_dir / "blueprint_proxy_skeleton_conditioning.mp4",
    )
    input_package_manifest = _first_existing_file(
        local_materialization_dir / "oscar_wam_input_package_manifest.json",
    )
    runtime_input_manifest = _first_existing_file(runtime_dir / "wam_rollout_input_manifest.json")
    auxiliary_manifest = _first_existing_file(
        runtime_input_dir / "wam_auxiliary_observation_manifest.json",
        local_input_dir / "wam_auxiliary_observation_manifest.json",
    )
    input_package = _read_optional_json(input_package_manifest)
    runtime_manifest = _read_optional_json(runtime_input_manifest)
    contract = _mapping(
        input_package.get("policy_action_to_skeleton_contract")
        or runtime_manifest.get("policy_action_to_skeleton_contract")
    )
    preview_dir = (
        local_materialization_dir
        / "oscar_input_conditioning_visual_review"
        / "generated_rollout_frame_review"
        / "frames"
    )
    preview_frames = sorted(path for path in preview_dir.glob("*.jpg") if path.is_file())
    return {
        "step_dir": str(step_dir),
        "step_index": _step_index_from_wam_step_dir(step_dir),
        "status": "completed" if first_frame or rgb_context or action_conditioning else "blocked",
        "first_frame_path": str(first_frame) if first_frame else None,
        "rgb_context_video_path": str(rgb_context) if rgb_context else None,
        "action_conditioning_video_path": str(action_conditioning)
        if action_conditioning
        else None,
        "action_conditioning_preview_frame_paths": [str(path) for path in preview_frames[:6]],
        "input_package_manifest_path": str(input_package_manifest)
        if input_package_manifest
        else None,
        "runtime_input_manifest_path": str(runtime_input_manifest)
        if runtime_input_manifest
        else None,
        "auxiliary_observation_manifest_path": str(auxiliary_manifest)
        if auxiliary_manifest
        else None,
        "policy_action_to_skeleton_contract_status": contract.get("status"),
        "policy_ranking_claim_safe": contract.get("policy_ranking_claim_safe"),
        "policy_action_to_skeleton_contract_blockers": _string_list(contract.get("blockers")),
        "claim_boundary": {
            "wam_input_review_is_input_provenance_only": True,
            "review_media_is_not_wam_output_quality_or_task_success": True,
            "scene_or_task_specific_pixels_used": False,
        },
    }


def _write_wam_input_review_artifacts(
    *,
    job: Path,
    extraction_dir: Path,
    generated_at: str,
) -> dict[str, Any]:
    step_root = extraction_dir / "wam_worker_steps"
    rows = [
        _wam_input_review_row(step_dir)
        for step_dir in sorted(step_root.glob("step_*"))
        if step_dir.is_dir()
    ] if step_root.is_dir() else []
    contact_sheet = _write_wam_input_review_contact_sheet(
        output_path=job / "wam_input_review_contact_sheet.jpg",
        rows=rows,
    )
    blockers: list[str] = []
    if not rows:
        blockers.append("wam_input_review_no_wam_worker_steps")
    blockers.extend(_string_list(contact_sheet.get("blockers")))
    manifest = {
        "schema_version": "persistent_wam_input_review_manifest.v1",
        "generated_at": generated_at,
        "status": "completed" if rows else "blocked",
        "wam_step_count": len(rows),
        "input_media_row_count": sum(1 for row in rows if row.get("status") == "completed"),
        "rows": rows,
        "contact_sheet": contact_sheet,
        "contact_sheet_path": contact_sheet.get("contact_sheet_path"),
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "wam_input_review_is_input_provenance_only": True,
            "review_media_is_not_wam_output_quality_or_task_success": True,
            "policy_ranking_claim_safe_not_inferred_from_review_media": True,
            "scene_or_task_specific_pixels_used": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(job / "wam_input_review_manifest.json", manifest)
    return manifest


def _persistent_episode_consistency_visual_status(
    visual_quality_report: Mapping[str, Any],
) -> str:
    if visual_quality_report.get("visual_success"):
        return "passed_visual_quality_smoke"
    return _string(visual_quality_report.get("status")) or "failed_visual_quality_gate"


def _write_persistent_episode_consistency_artifacts(
    *,
    job: Path,
    extraction_dir: Path,
    imported: Mapping[str, Any],
    generated_at: str,
    policy_observation: Mapping[str, Any],
    visual_quality_report: Mapping[str, Any],
    video_status: Mapping[str, Any],
    policy_calls: Sequence[Mapping[str, Any]],
    wam_rows: Sequence[Mapping[str, Any]],
    side_rows: Sequence[Mapping[str, Any]],
    timeout_seconds: float,
) -> dict[str, Any]:
    review_video_path = _string(
        video_status.get("review_video_path")
        or visual_quality_report.get("review_video_path")
    )
    visual_rollout_useful = bool(visual_quality_report.get("visual_success"))
    has_review_video = bool(review_video_path and Path(review_video_path).expanduser().is_file())
    rollout_id = "persistent_g1_wam_rollout_0001"
    scenario_eval_run_id = "persistent_g1_wam_episode"
    task_prompt = _string(policy_observation.get("task_prompt"))
    task_id = _string(policy_observation.get("task_id")) or "unitree_groot_n17_sonic_persistent_session"
    rollouts = (
        [
            {
                "rollout_id": rollout_id,
                "scenario_eval_run_id": scenario_eval_run_id,
                "policy_id": POLICY_ID,
                "task_id": task_id,
                "model_candidate": "oscar_wam",
                "generated_video_path": review_video_path,
                "live_wam_generation_success_count": int(
                    imported.get("live_wam_generation_success_count") or 0
                ),
                "learned_wam_model_success_count": int(
                    imported.get("learned_wam_model_success_count") or 0
                ),
                "structural_fallback_used": any(
                    bool(row.get("structural_fallback_used")) for row in wam_rows
                ),
            }
        ]
        if has_review_video
        else []
    )
    request_path = job / "wam_episode_consistency_request.json"
    output_path = job / WAM_CONSISTENCY_COMMAND_OUTPUT
    checks_path = job / "wam_consistency_checks.json"
    generated_results_path = job / "wam_generated_rollout_results.json"
    visual_smoke_path = job / "wam_rollout_visual_quality_report.json"
    visual_smoke_status = _persistent_episode_consistency_visual_status(visual_quality_report)
    request = {
        "schema_version": "wam_episode_consistency_request.v1",
        "generated_at": generated_at,
        "status": "ready_for_external_episode_scorer"
        if rollouts and visual_rollout_useful
        else "blocked_generated_rollout_visual_quality"
        if rollouts
        else "blocked_missing_generated_rollout",
        "source_persistent_session_output_dir": str(extraction_dir),
        "generated_rollout_results": str(generated_results_path),
        "generated_rollout_visual_smoke": str(visual_smoke_path),
        "generated_rollout_visual_smoke_status": visual_smoke_status,
        "generated_rollout_visually_useful_for_success_review": visual_rollout_useful,
        "rollouts": rollouts,
        "task_prompts": [
            {
                "scenario_eval_run_id": scenario_eval_run_id,
                "task_prompt": task_prompt,
                "task_id": task_id,
            }
        ],
        "source_trace_paths": {
            "robot_policy_wam_loop_trace_jsonl": str(job / "robot_policy_wam_loop_trace.jsonl"),
            "wam_generated_next_observations_jsonl": str(
                job / "wam_generated_next_observations.jsonl"
            ),
            "robot_policy_wam_side_by_side_trace_jsonl": str(
                job / "robot_policy_wam_side_by_side_trace.jsonl"
            ),
            "robot_policy_wam_loop_manifest": str(job / "robot_policy_wam_loop_manifest.json"),
        },
        "trace_summary": {
            "policy_call_count": len(policy_calls),
            "wam_transition_count": len(wam_rows),
            "side_by_side_trace_row_count": len(side_rows),
            "live_wam_generation_success_count": int(
                imported.get("live_wam_generation_success_count") or 0
            ),
            "learned_wam_model_success_count": int(
                imported.get("learned_wam_model_success_count") or 0
            ),
        },
        "expected_output_path": str(output_path),
        "consistency_label_contract": {
            "required_top_level_keys": ["rollout_checks"],
            "rollout_check_required_keys": [
                "rollout_id",
                "forward_consistent",
                "inverse_consistent",
                "confidence",
                "rationale",
            ],
        },
        "claim_boundary": {
            "scorer_is_separate_from_wam_execution_and_evaluator": True,
            "scorer_input_is_generated_video_and_trace_context_not_physical_robot": True,
            "consistency_label_does_not_prove_task_success": True,
            "consistency_label_does_not_prove_generated_world_rank_fidelity": True,
            "raw_credentials_written_to_artifacts": False,
        },
    }
    write_json(request_path, request)
    write_json(
        generated_results_path,
        {
            "schema_version": "persistent_wam_generated_rollout_results.v1",
            "generated_at": generated_at,
            "status": "completed" if rollouts else "blocked_missing_generated_rollout",
            "rollouts": rollouts,
            "blockers": [] if rollouts else ["missing_review_video_for_wam_episode_consistency"],
            "claim_boundary": {
                "generated_review_video_is_not_task_success_proof": True,
                "generated_review_video_is_not_forward_inverse_consistency": True,
            },
        },
    )
    command = _string(os.getenv(WAM_CONSISTENCY_COMMAND_ENV))
    allow_scoring = _wam_consistency_env_truthy(WAM_CONSISTENCY_GATE_ENV)
    consistency_blockers: list[str] = []
    command_result: dict[str, Any] | None = None
    command_payload: dict[str, Any] = {}
    if not rollouts:
        consistency_blockers = ["missing_review_video_for_wam_episode_consistency"]
    elif not visual_rollout_useful:
        consistency_blockers = _string_list(visual_quality_report.get("blockers")) or [
            "generated_rollout_not_visually_useful_for_consistency_proof"
        ]
    elif allow_scoring or command:
        if not _wam_consistency_env_truthy(WAM_CONSISTENCY_GATE_ENV):
            consistency_blockers.append(f"missing_env_{WAM_CONSISTENCY_GATE_ENV}")
        if not command:
            consistency_blockers.append("missing_wam_episode_consistency_command")
        if not consistency_blockers:
            command_payload, command_result = _run_wam_consistency_command(
                command=command,
                input_path=request_path,
                output_path=output_path,
                timeout_seconds=timeout_seconds,
            )
            if command_result.get("status") != "completed":
                consistency_blockers.extend(
                    _string_list(command_result.get("blockers"))
                    or ["wam_episode_consistency_command_blocked"]
                )
    else:
        consistency_blockers = ["requires_external_wam_episode_consistency_scorer"]

    if command_payload and not consistency_blockers:
        consistency = _normalize_wam_episode_consistency(
            command_payload=command_payload,
            rollouts=rollouts,
            generated_at=generated_at,
            action_conditioned_video_rollout_generated=bool(rollouts),
            action_conditioned_video_rollout_available=bool(rollouts),
            provider_output_replay_used=False,
            success_label_generated=False,
            visual_smoke_status=visual_smoke_status,
            visual_rollout_useful=visual_rollout_useful,
            command_result=command_result,
        )
    else:
        consistency = _unscored_wam_episode_consistency(
            generated_at=generated_at,
            rollouts=rollouts,
            action_conditioned_video_rollout_generated=bool(rollouts),
            action_conditioned_video_rollout_available=bool(rollouts),
            provider_output_replay_used=False,
            success_label_generated=False,
            visual_smoke_status=visual_smoke_status,
            visual_rollout_useful=visual_rollout_useful,
            blockers=consistency_blockers,
            blocked_reason="blocked_missing_generated_rollout"
            if not rollouts
            else "blocked_generated_rollout_visual_quality"
            if not visual_rollout_useful
            else None,
        )
        if command_result is not None:
            consistency["command_result"] = command_result
    scoring_requested = bool(allow_scoring or command)
    consistency["scoring_requested"] = scoring_requested
    consistency["early_termination_recommended"] = bool(
        scoring_requested and not consistency.get("forward_inverse_consistency_proven")
    )
    consistency["request_path"] = str(request_path)
    write_json(checks_path, consistency)
    return {
        "schema_version": "persistent_wam_episode_consistency_summary.v1",
        "generated_at": generated_at,
        "status": consistency.get("status"),
        "scoring_requested": scoring_requested,
        "wam_episode_consistency_request": str(request_path),
        "wam_episode_consistency_command": str(output_path) if output_path.is_file() else None,
        "wam_consistency_checks": str(checks_path),
        "external_episode_consistency_scorer_ran": bool(
            consistency.get("external_episode_consistency_scorer_ran")
        ),
        "external_episode_consistency_scorer_required": bool(
            consistency.get("external_episode_consistency_scorer_required")
        ),
        "forward_inverse_consistency_proven": bool(
            consistency.get("forward_inverse_consistency_proven")
        ),
        "early_termination_recommended": bool(
            consistency.get("early_termination_recommended")
        ),
        "blockers": _wam_consistency_blockers(consistency),
        "claim_boundary": {
            "forward_inverse_consistency_is_external_episode_label_not_wam_execution": True,
            "forward_inverse_consistency_is_reliability_review_signal_only": True,
            "forward_inverse_consistency_does_not_prove_task_success": True,
            "forward_inverse_consistency_does_not_prove_generated_world_rank_fidelity": True,
        },
    }


def _write_rank_fidelity_calibration_requirement(
    *,
    job: Path,
    generated_at: str,
    imported: Mapping[str, Any],
    policy_observation: Mapping[str, Any],
    visual_quality_report: Mapping[str, Any],
    consistency_summary: Mapping[str, Any],
    success_proven: bool,
) -> dict[str, Any]:
    task_id = _string(policy_observation.get("task_id")) or "unknown_task"
    target_object_id = _string(policy_observation.get("target_object_id")) or None
    scenario_eval_run_id = _string(imported.get("scenario_eval_run_id")) or "persistent_g1_wam_episode"
    scenario_variation_instance_id = (
        _string(imported.get("scenario_variation_instance_id"))
        or _string(policy_observation.get("scenario_variation_instance_id"))
        or "persistent_wam_single_variation"
    )
    candidate_records = [
        {
            "record_id": "persistent_wam_prediction_0001",
            "scenario_eval_run_id": scenario_eval_run_id,
            "policy_id": POLICY_ID,
            "task_id": task_id,
            "scenario_variation_instance_id": scenario_variation_instance_id,
            "target_object_id": target_object_id,
            "predicted_visual_success": bool(visual_quality_report.get("visual_success")),
            "predicted_episode_consistency": bool(
                consistency_summary.get("forward_inverse_consistency_proven")
            ),
            "predicted_task_success": bool(success_proven),
            "actual_status": "needs_accepted_anchor_outcome",
            "source": "persistent_wam_visual_review_prediction_record",
        }
    ]
    anchor_join_keys = [
        "scenario_eval_run_id",
        "policy_id",
        "task_id",
        "scenario_variation_instance_id",
    ]
    required_anchor_evidence = {
        "actual_success": "required_boolean",
        "review_status": "accepted",
        "operator_attestation.status": "signed",
        "owner_or_reviewer_evidence_refs": "required_nonempty_list",
        "physical_run_evidence_refs": "required_when_physical_evidence_requested",
    }
    anchor_request_rows = [
        {
            "anchor_request_id": "persistent_wam_anchor_request_0001",
            "row_role": "current_policy_current_variation",
            "candidate_prediction_record_id": "persistent_wam_prediction_0001",
            "scenario_eval_run_id": scenario_eval_run_id,
            "policy_id": POLICY_ID,
            "task_id": task_id,
            "scenario_variation_instance_id": scenario_variation_instance_id,
            "target_object_id": target_object_id,
            "prediction_status": "available",
            "actual_status": "needs_accepted_anchor_outcome",
            "exact_join_keys_status": "ready_for_actual_join",
            "required_anchor_evidence": required_anchor_evidence,
            "accepted_for_calibration": False,
        },
        {
            "anchor_request_id": "persistent_wam_anchor_request_0002",
            "row_role": "current_policy_second_variation",
            "candidate_prediction_record_id": None,
            "scenario_eval_run_id": "to_be_assigned_by_calibration_run",
            "policy_id": POLICY_ID,
            "task_id": task_id,
            "scenario_variation_instance_id": "second_variation_required",
            "target_object_id": target_object_id,
            "prediction_status": "needs_matching_prediction_record",
            "actual_status": "needs_accepted_anchor_outcome",
            "exact_join_keys_status": "requires_prediction_and_actual_rows",
            "required_anchor_evidence": required_anchor_evidence,
            "accepted_for_calibration": False,
        },
        {
            "anchor_request_id": "persistent_wam_anchor_request_0003",
            "row_role": "comparison_policy_first_variation",
            "candidate_prediction_record_id": None,
            "scenario_eval_run_id": "to_be_assigned_by_calibration_run",
            "policy_id": "comparison_policy_required",
            "task_id": task_id,
            "scenario_variation_instance_id": "comparison_variation_001_required",
            "target_object_id": target_object_id,
            "prediction_status": "needs_matching_prediction_record",
            "actual_status": "needs_accepted_anchor_outcome",
            "exact_join_keys_status": "requires_prediction_and_actual_rows",
            "required_anchor_evidence": required_anchor_evidence,
            "accepted_for_calibration": False,
        },
        {
            "anchor_request_id": "persistent_wam_anchor_request_0004",
            "row_role": "comparison_policy_second_variation",
            "candidate_prediction_record_id": None,
            "scenario_eval_run_id": "to_be_assigned_by_calibration_run",
            "policy_id": "comparison_policy_required",
            "task_id": task_id,
            "scenario_variation_instance_id": "comparison_variation_002_required",
            "target_object_id": target_object_id,
            "prediction_status": "needs_matching_prediction_record",
            "actual_status": "needs_accepted_anchor_outcome",
            "exact_join_keys_status": "requires_prediction_and_actual_rows",
            "required_anchor_evidence": required_anchor_evidence,
            "accepted_for_calibration": False,
        },
    ]
    blockers = [
        "missing_accepted_calibration_anchor_outcomes",
        "insufficient_anchor_count",
        "insufficient_policy_group_count",
        "real_world_rank_correlation_not_measured",
    ]
    small_calibration_set_path = job / "rank_fidelity_small_calibration_set.json"
    small_calibration_set_rows = [
        {
            "calibration_set_row_id": row["anchor_request_id"].replace(
                "anchor_request",
                "set_row",
            ),
            "anchor_request_id": row["anchor_request_id"],
            "row_role": row["row_role"],
            "scenario_eval_run_id": row["scenario_eval_run_id"],
            "policy_id": row["policy_id"],
            "task_id": row["task_id"],
            "scenario_variation_instance_id": row["scenario_variation_instance_id"],
            "target_object_id": row["target_object_id"],
            "candidate_prediction_record_id": row["candidate_prediction_record_id"],
            "prediction_status": row["prediction_status"],
            "actual_status": row["actual_status"],
            "exact_join_keys_status": row["exact_join_keys_status"],
            "accepted_for_calibration": False,
        }
        for row in anchor_request_rows
    ]
    small_calibration_set_payload = {
        "schema_version": PERSISTENT_WAM_RANK_FIDELITY_SMALL_CALIBRATION_SET_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "draft_pending_prediction_and_accepted_outcome_rows",
        "policy_id": POLICY_ID,
        "task_id": task_id,
        "target_object_id": target_object_id,
        "set_row_count": len(small_calibration_set_rows),
        "minimum_rows": 4,
        "minimum_policy_groups": 2,
        "minimum_variations_per_policy": 2,
        "current_prediction_row_count": 1,
        "missing_prediction_row_count": 3,
        "accepted_anchor_count": 0,
        "exact_join_keys": anchor_join_keys,
        "set_rows": small_calibration_set_rows,
        "required_anchor_evidence": required_anchor_evidence,
        "blockers": blockers,
        "claim_boundary": {
            "small_calibration_set_is_collection_plan_until_actuals_are_accepted": True,
            "small_calibration_set_rows_are_not_accepted_anchors": True,
            "exact_prediction_vs_actual_join_required": True,
            "loose_or_inferred_matches_allowed_for_calibration": False,
            "rank_fidelity_result_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
            "raw_credentials_written_to_artifacts": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(small_calibration_set_path, small_calibration_set_payload)
    anchor_request_path = job / "rank_fidelity_calibration_anchor_request.json"
    anchor_request_payload = {
        "schema_version": PERSISTENT_WAM_RANK_FIDELITY_CALIBRATION_ANCHOR_REQUEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked_awaiting_accepted_anchor_outcomes",
        "policy_id": POLICY_ID,
        "task_id": task_id,
        "target_object_id": target_object_id,
        "requested_anchor_count": len(anchor_request_rows),
        "minimum_accepted_anchor_count": 4,
        "minimum_policy_group_count": 2,
        "current_prediction_anchor_request_count": 1,
        "missing_prediction_anchor_request_count": 3,
        "accepted_anchor_count": 0,
        "exact_join_keys": anchor_join_keys,
        "small_calibration_set": str(small_calibration_set_path),
        "small_calibration_set_status": small_calibration_set_payload["status"],
        "anchor_request_rows": anchor_request_rows,
        "blockers": blockers,
        "claim_boundary": {
            "anchor_request_rows_are_not_accepted_anchors": True,
            "small_calibration_set_rows_are_not_accepted_anchors": True,
            "request_artifact_does_not_prove_rank_fidelity": True,
            "exact_prediction_vs_actual_join_required": True,
            "loose_or_inferred_matches_allowed_for_calibration": False,
            "rank_fidelity_result_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
            "raw_credentials_written_to_artifacts": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(anchor_request_path, anchor_request_payload)
    payload = {
        "schema_version": PERSISTENT_WAM_RANK_FIDELITY_CALIBRATION_REQUIREMENT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked_missing_calibration_anchors",
        "policy_id": POLICY_ID,
        "task_id": task_id,
        "target_object_id": target_object_id,
        "candidate_prediction_record_count": len(candidate_records),
        "candidate_prediction_records": candidate_records,
        "calibration_anchor_request": str(anchor_request_path),
        "calibration_anchor_request_status": anchor_request_payload["status"],
        "small_calibration_set": str(small_calibration_set_path),
        "small_calibration_set_status": small_calibration_set_payload["status"],
        "small_calibration_set_row_count": small_calibration_set_payload["set_row_count"],
        "requested_anchor_count": anchor_request_payload["requested_anchor_count"],
        "accepted_anchor_count": 0,
        "policy_group_count": 1,
        "minimum_accepted_anchor_count": 4,
        "minimum_policy_group_count": 2,
        "accepted_anchor_schema": {
            "join_keys": anchor_join_keys,
            "required_fields": [
                "actual_success",
                "review_status",
                "operator_attestation.status",
            ],
            "accepted_review_status": "accepted",
            "accepted_operator_attestation_status": "signed",
        },
        "recommended_small_calibration_set": {
            "minimum_rows": 4,
            "minimum_policy_groups": 2,
            "minimum_variations_per_policy": 2,
            "requires_exact_join_key_match": True,
            "requires_accepted_reviewer_or_operator_anchor": True,
        },
        "sim_vs_real_calibration_score": None,
        "spearman_rank_correlation": None,
        "pearson_success_rate_correlation": None,
        "deployment_accuracy_claim_allowed": False,
        "rank_fidelity_result_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "blockers": blockers,
        "claim_boundary": {
            "visual_quality_pass_is_not_rank_fidelity": True,
            "episode_consistency_label_is_not_rank_fidelity": True,
            "candidate_prediction_records_are_not_accepted_anchors": True,
            "small_calibration_set_rows_are_not_accepted_anchors": True,
            "rank_fidelity_requires_accepted_prediction_vs_actual_anchor_set": True,
            "rank_fidelity_result_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
            "raw_credentials_written_to_artifacts": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(job / "rank_fidelity_calibration_requirement.json", payload)
    return payload


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
    materialization_source_kind_counts: dict[str, int] = {}
    selection_quality_status_counts: dict[str, int] = {}
    selected_frame_signal_blocker_counts: dict[str, int] = {}
    materialized_future_frame_count = 0
    video_first_frame_materialization_count = 0
    degraded_future_frame_count = 0
    input_contract_status_counts: dict[str, int] = {}
    input_contract_warning_counts: dict[str, int] = {}
    input_contract_risk_flag_counts: dict[str, int] = {}
    input_contract_high_risk_flag_counts: dict[str, int] = {}
    input_contract_ranking_risk_flag_counts: dict[str, int] = {}
    input_contract_conditioning_mode_counts: dict[str, int] = {}
    input_contract_rgb_context_mode_counts: dict[str, int] = {}
    input_contract_high_risk_count = 0
    input_contract_policy_ranking_risk_count = 0
    input_contract_projected_skeleton_used_count = 0
    input_contract_policy_action_proxy_count = 0
    input_contract_scene_faithful_bridge_count = 0
    input_contract_safe_sim_ranking_bridge_count = 0
    for row in wam_calls:
        materialization = _mapping(row.get("materialization"))
        source_kind = _string(materialization.get("source_kind")) or "unknown"
        materialization_source_kind_counts[source_kind] = (
            materialization_source_kind_counts.get(source_kind, 0) + 1
        )
        if materialization.get("future_frame_selected") is True:
            materialized_future_frame_count += 1
        selection_quality_status = _string(materialization.get("selection_quality_status"))
        if selection_quality_status:
            selection_quality_status_counts[selection_quality_status] = (
                selection_quality_status_counts.get(selection_quality_status, 0) + 1
            )
            if (
                source_kind == "video_future_frame"
                and selection_quality_status != "passed_signal_gate"
            ):
                degraded_future_frame_count += 1
        for blocker in _string_list(materialization.get("selected_frame_signal_blockers")):
            selected_frame_signal_blocker_counts[blocker] = (
                selected_frame_signal_blocker_counts.get(blocker, 0) + 1
            )
        if source_kind == "video_first_frame":
            video_first_frame_materialization_count += 1
        live_payload = _mapping(row.get("live_wam_payload_redacted"))
        input_package = _mapping(live_payload.get("input_package"))
        input_contract = _mapping(input_package.get("oscar_input_contract_diagnostic"))
        if not input_contract:
            input_contract = _mapping(live_payload.get("oscar_input_contract_diagnostic"))
        skeleton_contract = _mapping(input_contract.get("skeleton_video"))
        projected_contract = _mapping(input_contract.get("projected_skeleton_trace"))
        rgb_contract = _mapping(input_contract.get("rgb_context"))
        input_contract_claim_boundary = _mapping(input_contract.get("claim_boundary"))
        input_status = _string(input_contract.get("status"))
        if input_status:
            input_contract_status_counts[input_status] = (
                input_contract_status_counts.get(input_status, 0) + 1
            )
        conditioning_mode = _string(skeleton_contract.get("conditioning_mode")) or _string(
            _mapping(input_package.get("skeleton_video")).get("conditioning_mode")
        )
        if conditioning_mode:
            input_contract_conditioning_mode_counts[conditioning_mode] = (
                input_contract_conditioning_mode_counts.get(conditioning_mode, 0) + 1
            )
        rgb_context_mode = _string(rgb_contract.get("rgb_context_mode")) or _string(
            _mapping(input_package.get("rgb_video")).get("rgb_context_mode")
        )
        if rgb_context_mode:
            input_contract_rgb_context_mode_counts[rgb_context_mode] = (
                input_contract_rgb_context_mode_counts.get(rgb_context_mode, 0) + 1
            )
        for warning in _string_list(input_contract.get("warnings")):
            input_contract_warning_counts[warning] = (
                input_contract_warning_counts.get(warning, 0) + 1
            )
        for flag in _string_list(input_contract.get("autoregressive_risk_flags")):
            input_contract_risk_flag_counts[flag] = (
                input_contract_risk_flag_counts.get(flag, 0) + 1
            )
        row_high_risk_flags = _string_list(input_contract.get("high_risk_flags"))
        for flag in row_high_risk_flags:
            input_contract_high_risk_flag_counts[flag] = (
                input_contract_high_risk_flag_counts.get(flag, 0) + 1
            )
        row_ranking_risk_flags = _string_list(input_contract.get("ranking_risk_flags"))
        for flag in row_ranking_risk_flags:
            input_contract_ranking_risk_flag_counts[flag] = (
                input_contract_ranking_risk_flag_counts.get(flag, 0) + 1
            )
        if input_contract.get("policy_ranking_claim_safe") is False or row_ranking_risk_flags:
            input_contract_policy_ranking_risk_count += 1
        projected_used = bool(
            projected_contract.get("used_for_conditioning")
            or _mapping(input_package.get("projected_skeleton_trace")).get("used_for_conditioning")
        )
        if projected_used:
            input_contract_projected_skeleton_used_count += 1
        scene_faithful_bridge_used = bool(
            projected_contract.get("scene_faithful_isaac_policy_action_projection_bridge_used")
            or projected_contract.get("official_wbc_or_sim_bridge_used")
            or input_contract.get("scene_faithful_isaac_policy_action_projection_bridge_used")
            or input_contract_claim_boundary.get("scene_or_task_specific_pixels_used")
        )
        safe_sim_ranking_bridge_used = bool(
            projected_contract.get("policy_action_bridge_safe_for_sim_ranking")
            or input_contract.get("policy_action_bridge_safe_for_sim_ranking")
            or (
                scene_faithful_bridge_used
                and input_contract.get("policy_ranking_claim_safe") is True
            )
        )
        if scene_faithful_bridge_used:
            input_contract_scene_faithful_bridge_count += 1
        if safe_sim_ranking_bridge_used:
            input_contract_safe_sim_ranking_bridge_count += 1
        policy_action_proxy_used = bool(
            skeleton_contract.get("policy_action_proxy_used")
            or _mapping(input_package.get("claim_boundary")).get(
                "policy_action_conditioning_proxy_video_used"
            )
            or conditioning_mode == "unitree_sonic_policy_action_proxy_over_scene_frame"
        )
        if policy_action_proxy_used:
            input_contract_policy_action_proxy_count += 1
        inferred_high_risk = bool(
            input_status == "warning_high_risk"
            or input_contract.get("autoregressive_risk_level") == "high"
            or row_high_risk_flags
            or (
                policy_action_proxy_used
                and not projected_used
                and rgb_context_mode == "single_frame_repeat"
            )
        )
        if inferred_high_risk:
            input_contract_high_risk_count += 1
            if (
                policy_action_proxy_used
                and not projected_used
                and rgb_context_mode == "single_frame_repeat"
            ):
                input_contract_high_risk_flag_counts[
                    "policy_action_proxy_single_frame_repeat_without_projected_skeleton"
                ] = (
                    input_contract_high_risk_flag_counts.get(
                        "policy_action_proxy_single_frame_repeat_without_projected_skeleton",
                        0,
                    )
                    + 1
                )
    future_frame_quality_blockers: list[str] = []
    if video_first_frame_materialization_count > 0:
        future_frame_quality_blockers.append(
            "wam_generated_next_observation_used_video_first_frame_fallback"
        )
    if degraded_future_frame_count > 0:
        future_frame_quality_blockers.append(
            "wam_generated_next_observation_future_frame_degraded_visual_signal"
        )
    materialization_summary = {
        "schema_version": "persistent_wam_materialization_summary.v1",
        "generated_at": generated_at,
        "status": "completed",
        "wam_call_count": len(wam_calls),
        "source_kind_counts": materialization_source_kind_counts,
        "selection_quality_status_counts": selection_quality_status_counts,
        "selected_frame_signal_blocker_counts": selected_frame_signal_blocker_counts,
        "video_first_frame_materialization_count": video_first_frame_materialization_count,
        "materialized_future_frame_count": materialized_future_frame_count,
        "degraded_future_frame_count": degraded_future_frame_count,
        "all_materialized_frames_are_video_first_frames": bool(
            wam_calls and video_first_frame_materialization_count == len(wam_calls)
        ),
        "all_materialized_future_frames_passed_signal_gate": bool(
            materialized_future_frame_count > 0
            and degraded_future_frame_count == 0
            and video_first_frame_materialization_count == 0
        ),
        "future_frame_quality_status": "failed" if future_frame_quality_blockers else "passed",
        "future_frame_quality_blockers": future_frame_quality_blockers,
        "claim_boundary": {
            "video_first_frame_materialization_is_not_future_rollout_quality_proof": True,
            "degraded_future_frame_materialization_is_not_visual_rollout_quality_proof": True,
            "source_kind_summary_is_not_task_success_evidence": True,
            "scene_or_task_specific_pixels_used": bool(input_contract_scene_faithful_bridge_count),
        },
        "raw_credentials_written_to_artifacts": False,
    }
    write_json(job / "wam_materialization_summary.json", materialization_summary)
    input_contract_blockers: list[str] = []
    if input_contract_high_risk_count:
        if (
            input_contract_policy_action_proxy_count
            and input_contract_projected_skeleton_used_count == 0
        ):
            input_contract_blockers.append(
                "wam_input_contract_high_risk_policy_action_proxy_without_projected_skeleton"
            )
        if (
            input_contract_high_risk_flag_counts.get(
                "projected_skeleton_nominal_action_projection_high_risk", 0
            )
            > 0
        ):
            input_contract_blockers.append(
                "wam_input_contract_high_risk_projected_skeleton_nominal_action_projection"
            )
        if (
            input_contract_high_risk_flag_counts.get(
                "projected_skeleton_not_scene_faithful_policy_action_high_risk", 0
            )
            > 0
        ):
            input_contract_blockers.append(
                "wam_input_contract_high_risk_projected_skeleton_not_scene_faithful_policy_action_bridge"
            )
        if not input_contract_blockers:
            input_contract_blockers.append("wam_input_contract_high_risk")
    if input_contract_policy_ranking_risk_count:
        input_contract_blockers.append("wam_input_contract_policy_ranking_claim_not_safe")
        if (
            input_contract_ranking_risk_flag_counts.get(
                "projected_skeleton_missing_scene_faithful_policy_action_bridge", 0
            )
            > 0
        ):
            input_contract_blockers.append(
                "wam_input_contract_missing_scene_faithful_policy_action_bridge"
            )

    input_contract_summary = {
        "schema_version": "persistent_wam_input_contract_summary.v1",
        "generated_at": generated_at,
        "status": "warning_high_risk"
        if input_contract_high_risk_count
        else "completed"
        if input_contract_status_counts
        else "not_available",
        "wam_call_count": len(wam_calls),
        "contract_status_counts": input_contract_status_counts,
        "contract_warning_counts": input_contract_warning_counts,
        "contract_autoregressive_risk_flag_counts": input_contract_risk_flag_counts,
        "contract_high_risk_flag_counts": input_contract_high_risk_flag_counts,
        "contract_ranking_risk_flag_counts": input_contract_ranking_risk_flag_counts,
        "conditioning_mode_counts": input_contract_conditioning_mode_counts,
        "rgb_context_mode_counts": input_contract_rgb_context_mode_counts,
        "high_risk_input_contract_count": input_contract_high_risk_count,
        "policy_ranking_risk_input_contract_count": input_contract_policy_ranking_risk_count,
        "policy_ranking_claim_safe": input_contract_policy_ranking_risk_count == 0,
        "projected_skeleton_conditioning_count": input_contract_projected_skeleton_used_count,
        "policy_action_proxy_conditioning_count": input_contract_policy_action_proxy_count,
        "scene_faithful_isaac_policy_action_projection_bridge_count": (
            input_contract_scene_faithful_bridge_count
        ),
        "policy_action_bridge_safe_for_sim_ranking_count": (
            input_contract_safe_sim_ranking_bridge_count
        ),
        "blockers": input_contract_blockers,
        "claim_boundary": {
            "input_contract_summary_is_not_model_execution_proof": True,
            "input_contract_summary_is_not_rollout_quality_proof": True,
            "high_risk_input_contract_can_explain_but_not_prove_model_failure": True,
            "policy_ranking_claim_safe_requires_policy_derived_action_conditioning": True,
            "scene_or_task_specific_pixels_used": bool(input_contract_scene_faithful_bridge_count),
        },
        "raw_credentials_written_to_artifacts": False,
    }
    write_json(job / "wam_input_contract_summary.json", input_contract_summary)
    wam_input_review = _write_wam_input_review_artifacts(
        job=job,
        extraction_dir=extraction_dir,
        generated_at=generated_at,
    )
    policy_action_decoding_contract = _write_policy_action_decoding_contract(
        job=job,
        action=first_action,
        generated_at=generated_at,
    )
    policy_action_bridge_readiness = _write_policy_action_bridge_readiness(
        job=job,
        extraction_dir=extraction_dir,
        action_contract=policy_action_decoding_contract,
        generated_at=generated_at,
    )
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
            "policy_action_decoding_contract": str(job / "policy_action_decoding_contract.json"),
            "policy_action_decoding_status": policy_action_decoding_contract.get("status"),
            "policy_action_bridge_readiness": str(job / "policy_action_bridge_readiness.json"),
            "policy_action_bridge_readiness_status": policy_action_bridge_readiness.get("status"),
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
    try:
        policy_observation = _load_policy_observation(policy_observation_path)
    except Exception:
        policy_observation = {}
    postprocess_visual_evidence = (
        _policy_observation_semantic_visual_evidence(
            policy_observation,
            base_dir=Path(policy_observation_path).expanduser().parent,
        )
        if policy_observation
        else {}
    )
    task_prompt = _string(policy_observation.get("task_prompt"))
    target_object_id = _string(policy_observation.get("target_object_id"))
    target_label = _string(policy_observation.get("target_label")) or target_object_id
    success_proven = bool(
        live_wam_count > 0
        and learned_wam_count > 0
        and imported.get("manipulation_success_evaluator_result") == "success"
    )
    if success_proven:
        success_reason = "A live evaluator reported requested manipulation success."
    elif live_wam_count > 0:
        success_reason = (
            "The loop completed with live learned WAM generations, but no task-success "
            "evaluator or physics state proved the requested manipulation state transition."
        )
    elif structural_wam_count > 0:
        success_reason = (
            "The loop completed with structural WAM fallback only; no live learned WAM or "
            "physics state proved the requested manipulation state transition."
        )
    else:
        success_reason = (
            "The loop did not produce a completed WAM generation or physics state proving a "
            "requested manipulation state transition."
        )
    judge = {
        "schema_version": "manipulation_success_evaluator_results.v1",
        "generated_at": generated_at,
        "status": "completed",
        "question": "Did the requested manipulation succeed?",
        "task_prompt": task_prompt or None,
        "target_object_id": target_object_id or None,
        "target_label": target_label or None,
        "answer": "not_proven" if not success_proven else "yes",
        "did_target_manipulation_succeed": bool(success_proven),
        "manipulation_success_proven": bool(success_proven),
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
        projected_skeleton_trace_path=postprocess_visual_evidence.get(
            "projected_skeleton_trace_path"
        ),
    )
    future_frame_quality_blockers = _string_list(
        materialization_summary.get("future_frame_quality_blockers")
    )
    input_contract_quality_blockers = _string_list(input_contract_summary.get("blockers"))
    combined_quality_blockers = sorted(
        set(future_frame_quality_blockers + input_contract_quality_blockers)
    )
    if combined_quality_blockers:
        visual_quality_report = dict(visual_quality_report)
        frame_visual_status = _string(visual_quality_report.get("status")) or "unknown"
        frame_visual_success = bool(visual_quality_report.get("visual_success"))
        visual_quality_report["blockers"] = sorted(
            set(_string_list(visual_quality_report.get("blockers")) + combined_quality_blockers)
        )
        visual_quality_report["status"] = "failed_visual_quality_gate"
        visual_quality_report["visual_success"] = False
        visual_quality_report["frame_visual_status_before_contract_gate"] = frame_visual_status
        visual_quality_report["frame_visual_success_before_contract_gate"] = frame_visual_success
        visual_quality_report["materialization_gate_failed"] = bool(future_frame_quality_blockers)
        visual_quality_report["input_contract_gate_failed"] = bool(input_contract_quality_blockers)
        visual_quality_report["overall_gate_success"] = False
        materialization_quality = dict(visual_quality_report.get("materialization_quality") or {})
        materialization_quality.update(
            {
                "schema_version": "persistent_wam_materialization_quality.v1",
                "future_frame_quality_status": materialization_summary.get(
                    "future_frame_quality_status"
                ),
                "future_frame_quality_blockers": future_frame_quality_blockers,
                "materialized_future_frame_count": materialized_future_frame_count,
                "video_first_frame_materialization_count": video_first_frame_materialization_count,
                "degraded_future_frame_count": degraded_future_frame_count,
                "selection_quality_status_counts": selection_quality_status_counts,
                "selected_frame_signal_blocker_counts": selected_frame_signal_blocker_counts,
                "video_first_frame_materialization_is_not_future_rollout_quality_proof": True,
                "degraded_future_frame_materialization_is_not_visual_rollout_quality_proof": True,
            }
        )
        visual_quality_report["materialization_quality"] = materialization_quality
        visual_quality_report["input_contract_quality"] = input_contract_summary
        visual_quality_report["claim_boundary"] = {
            **_mapping(visual_quality_report.get("claim_boundary")),
            "video_first_frame_materialization_is_not_future_rollout_quality_proof": True,
            "degraded_future_frame_materialization_is_not_visual_rollout_quality_proof": True,
            "future_frame_materialization_required_for_visual_success": True,
            "high_risk_input_contract_is_not_visual_rollout_quality_proof": bool(
                input_contract_high_risk_count
            ),
        }
        write_json(job / "wam_rollout_visual_quality_report.json", visual_quality_report)
    consistency_summary = _write_persistent_episode_consistency_artifacts(
        job=job,
        extraction_dir=extraction_dir,
        imported=imported,
        generated_at=generated_at,
        policy_observation=policy_observation,
        visual_quality_report=visual_quality_report,
        video_status=video_status,
        policy_calls=policy_calls,
        wam_rows=wam_rows,
        side_rows=side_rows,
        timeout_seconds=_float_env("BLUEPRINT_WAM_EPISODE_CONSISTENCY_TIMEOUT_SECONDS", 60.0),
    )
    rank_fidelity_calibration_requirement = _write_rank_fidelity_calibration_requirement(
        job=job,
        generated_at=generated_at,
        imported=imported,
        policy_observation=policy_observation,
        visual_quality_report=visual_quality_report,
        consistency_summary=consistency_summary,
        success_proven=success_proven,
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
        "forward_inverse_consistency_proven": bool(
            consistency_summary.get("forward_inverse_consistency_proven")
        ),
        "external_episode_consistency_scorer_ran": bool(
            consistency_summary.get("external_episode_consistency_scorer_ran")
        ),
        "wam_episode_consistency_early_termination_recommended": bool(
            consistency_summary.get("early_termination_recommended")
        ),
        "wam_episode_consistency_request": consistency_summary.get(
            "wam_episode_consistency_request"
        ),
        "wam_consistency_checks": consistency_summary.get("wam_consistency_checks"),
        "forward_inverse_consistency_is_reliability_review_signal_only": True,
        "forward_inverse_consistency_does_not_prove_task_success": True,
        "forward_inverse_consistency_does_not_prove_generated_world_rank_fidelity": True,
        "rank_fidelity_calibration_required": True,
        "rank_fidelity_calibration_requirement": str(
            job / "rank_fidelity_calibration_requirement.json"
        ),
        "rank_fidelity_calibration_anchor_request": str(
            job / "rank_fidelity_calibration_anchor_request.json"
        ),
        "rank_fidelity_small_calibration_set": str(
            job / "rank_fidelity_small_calibration_set.json"
        ),
        "rank_fidelity_calibration_status": rank_fidelity_calibration_requirement.get("status"),
        "rank_fidelity_calibration_blockers": _string_list(
            rank_fidelity_calibration_requirement.get("blockers")
        ),
        "visual_review_ranking_is_not_real_world_rank_fidelity": True,
        "video_first_frame_materialization_is_not_future_rollout_quality_proof": (
            video_first_frame_materialization_count > 0
        ),
        "degraded_future_frame_materialization_is_not_visual_rollout_quality_proof": (
            degraded_future_frame_count > 0
        ),
        "materialized_future_frame_count": materialized_future_frame_count,
        "video_first_frame_materialization_count": video_first_frame_materialization_count,
        "degraded_future_frame_count": degraded_future_frame_count,
        "wam_input_contract_high_risk_count": input_contract_high_risk_count,
        "high_risk_input_contract_is_not_visual_rollout_quality_proof": bool(
            input_contract_high_risk_count
        ),
        "wam_rollout_visual_quality_report": str(job / "wam_rollout_visual_quality_report.json"),
        "local_structural_wam_generator_is_not_live_oscar_or_cosmos_model": structural_wam_count
        > 0,
        "frame_copy_placeholder_until_live_wam_model_configured": structural_wam_count > 0,
        "wam_evaluator_is_not_robot_policy": True,
        "provider_output_replay_used": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "non_ranking_operational_claim_proven": False,
        "accepted_anchor_manipulation_success_proven": False,
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
        if input_contract_high_risk_count:
            labels.append("wam_input_contract_high_risk")
        if consistency_summary.get("early_termination_recommended"):
            labels.append("wam_episode_consistency_early_termination_recommended")
            labels.append("forward_inverse_consistency_not_proven")
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
    elif consistency_summary.get("early_termination_recommended"):
        write_json(
            job / "failure_labels.json",
            {
                "schema_version": "persistent_policy_wam_failure_labels.v1",
                "generated_at": generated_at,
                "status": "completed",
                "labels": [
                    "wam_episode_consistency_early_termination_recommended",
                    "forward_inverse_consistency_not_proven",
                ],
                "task_success_not_failed_by_consistency_label": True,
                "consistency_label_is_reliability_abstention_only": True,
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
        "policy_action_decoding_contract": str(job / "policy_action_decoding_contract.json"),
        "policy_action_bridge_readiness": str(job / "policy_action_bridge_readiness.json"),
        "wam_generation_command_discovery": str(job / "wam_generation_command_discovery.json"),
        "wam_generation_command_execution": str(job / "wam_generation_command_execution.json"),
        "wam_generation_command_output": str(job / "wam_generation_command_output.json"),
        "wam_materialization_summary": str(job / "wam_materialization_summary.json"),
        "wam_input_contract_summary": str(job / "wam_input_contract_summary.json"),
        "wam_input_review_manifest": str(job / "wam_input_review_manifest.json"),
        "wam_input_review_contact_sheet": _string(wam_input_review.get("contact_sheet_path"))
        or None,
        "rank_fidelity_calibration_requirement": str(
            job / "rank_fidelity_calibration_requirement.json"
        ),
        "rank_fidelity_calibration_anchor_request": str(
            job / "rank_fidelity_calibration_anchor_request.json"
        ),
        "rank_fidelity_small_calibration_set": str(
            job / "rank_fidelity_small_calibration_set.json"
        ),
        "rank_fidelity_calibration_status": rank_fidelity_calibration_requirement.get("status"),
        "rank_fidelity_calibration_blockers": _string_list(
            rank_fidelity_calibration_requirement.get("blockers")
        ),
        "rank_fidelity_result_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
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
        "wam_episode_consistency_request": consistency_summary.get(
            "wam_episode_consistency_request"
        ),
        "wam_episode_consistency_command": consistency_summary.get(
            "wam_episode_consistency_command"
        ),
        "wam_consistency_checks": consistency_summary.get("wam_consistency_checks"),
        "forward_inverse_consistency_proven": bool(
            consistency_summary.get("forward_inverse_consistency_proven")
        ),
        "external_episode_consistency_scorer_ran": bool(
            consistency_summary.get("external_episode_consistency_scorer_ran")
        ),
        "external_episode_consistency_scorer_required": bool(
            consistency_summary.get("external_episode_consistency_scorer_required")
        ),
        "wam_episode_consistency_early_termination_recommended": bool(
            consistency_summary.get("early_termination_recommended")
        ),
        "wam_episode_consistency_blockers": _string_list(
            consistency_summary.get("blockers")
        ),
        "blockers": _string_list(consistency_summary.get("blockers"))
        if consistency_summary.get("early_termination_recommended")
        else [],
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


def _runpod_teardown_manifest_path(runpod_dir: Path, poll_manifest: Mapping[str, Any]) -> Path:
    teardown_action = _string(poll_manifest.get("teardown_action"))
    if teardown_action == "stop":
        return runpod_dir / "runpod_wam_async_stop_manifest.json"
    if teardown_action == "delete":
        return runpod_dir / "runpod_wam_async_delete_manifest.json"
    stop_path = runpod_dir / "runpod_wam_async_stop_manifest.json"
    if stop_path.is_file():
        return stop_path
    return runpod_dir / "runpod_wam_async_delete_manifest.json"


def _runpod_teardown_completed(runpod_dir: Path) -> bool:
    for filename in (
        "runpod_wam_async_stop_manifest.json",
        "runpod_wam_async_delete_manifest.json",
    ):
        path = runpod_dir / filename
        if not path.is_file():
            continue
        try:
            payload = _read_json(path)
        except (OSError, ValueError):
            continue
        if _string(payload.get("status")) == "completed":
            return True
    return False


def _runpod_keepalive_summary(
    *,
    runpod_dir: Path,
    poll_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    summary = {
        "teardown_requested": bool(poll_manifest.get("teardown_requested")),
        "teardown_action": poll_manifest.get("teardown_action"),
        "teardown_performed": bool(poll_manifest.get("teardown_performed")),
        "requested_keep_running_on_success": bool(
            poll_manifest.get("requested_keep_running_on_success")
        ),
        "keep_running_on_success": bool(poll_manifest.get("keep_running_on_success")),
        "keepalive_runtime_health": _mapping(poll_manifest.get("keepalive_runtime_health")),
        "keepalive_runtime_unhealthy_on_success": bool(
            poll_manifest.get("keepalive_runtime_unhealthy_on_success")
        ),
        "keepalive_performed": bool(poll_manifest.get("keepalive_performed")),
        "keepalive_manifest_path": poll_manifest.get("keepalive_manifest_path"),
        "warm_candidate_path": _mapping(poll_manifest.get("warm_candidate")).get("path")
        or poll_manifest.get("warm_candidate_path"),
        "continuing_spend_from_this_run": bool(
            poll_manifest.get("continuing_spend_from_this_run")
        ),
        "preserved_from_existing_keepalive_manifest": False,
        "raw_secret_values_recorded": False,
    }
    if (
        summary["keep_running_on_success"]
        or summary["keepalive_performed"]
        or summary["continuing_spend_from_this_run"]
        or _runpod_teardown_completed(runpod_dir)
    ):
        return summary

    keepalive_path = runpod_dir / "runpod_wam_async_keepalive_manifest.json"
    if not keepalive_path.is_file():
        return summary
    try:
        keepalive = _read_json(keepalive_path)
    except (OSError, ValueError):
        return summary
    if (
        _string(keepalive.get("status")) != "completed"
        or not bool(keepalive.get("continuing_spend_from_this_run"))
    ):
        return summary
    return {
        **summary,
        "teardown_action": keepalive.get("teardown_action") or "keep_on_success",
        "requested_keep_running_on_success": True,
        "keep_running_on_success": True,
        "keepalive_performed": True,
        "keepalive_manifest_path": str(keepalive_path),
        "warm_candidate_path": _mapping(keepalive.get("warm_candidate")).get("path")
        or keepalive.get("warm_candidate_path")
        or summary.get("warm_candidate_path"),
        "continuing_spend_from_this_run": True,
        "continuing_spend_evidence_source": "existing_keepalive_manifest",
        "preserved_from_existing_keepalive_manifest": True,
    }


def _finalize_runpod_persistent_session_output(
    *,
    job: Path,
    generated_at: str,
    policy_observation_path: str | Path,
    git_evidence: Mapping[str, Any],
    poll_manifest: Mapping[str, Any],
    runpod_dir: Path,
    output_zip: Path,
    provider_output_resume_used: bool = False,
) -> tuple[dict[str, Any], int]:
    extraction_dir = job / "imported_persistent_session_output"
    if extraction_dir.exists():
        shutil.rmtree(extraction_dir)
    ensure_dir(extraction_dir)
    with zipfile.ZipFile(output_zip) as archive:
        archive.extractall(extraction_dir)
    imported_path = extraction_dir / "unitree_groot_n17_sonic_wam_persistent_session_output.json"
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
    visual_report_path = _string(postprocess.get("wam_rollout_visual_quality_report"))
    visual_report: dict[str, Any] = {}
    if visual_report_path:
        visual_report_candidate = Path(visual_report_path).expanduser()
        if visual_report_candidate.is_file():
            try:
                visual_report = _read_json(visual_report_candidate)
            except (OSError, ValueError):
                visual_report = {}
    visual_success = bool(postprocess.get("wam_rollout_visual_success"))
    visual_quality_blockers = _string_list(visual_report.get("blockers"))
    learned_wam_success_count = int(imported.get("learned_wam_model_success_count") or 0)
    live_wam_success_count = int(imported.get("live_wam_generation_success_count") or 0)
    provider_inference_completed = bool(
        learned_wam_success_count > 0 or live_wam_success_count > 0
    )
    provider_output_failed_visual_quality = bool(
        provider_inference_completed
        and not visual_success
        and (
            visual_quality_blockers
            or _string(visual_report.get("status")) == "failed_visual_quality_gate"
        )
    )
    policy_ranking_blockers: list[str] = []
    if provider_output_failed_visual_quality:
        policy_ranking_blockers.append(
            "completed_provider_output_failed_wam_visual_quality_gate"
            if completed
            else "provider_inference_output_failed_wam_visual_quality_gate"
        )
        policy_ranking_blockers.extend(visual_quality_blockers)
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
    runpod_keepalive = _runpod_keepalive_summary(
        runpod_dir=runpod_dir,
        poll_manifest=poll_manifest,
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if completed else "blocked",
        "provider": "runpod",
        "policy_id": POLICY_ID,
        "selected_candidate_id": POLICY_ID,
        "job_dir": str(job),
        "git_evidence": dict(git_evidence),
        "persistent_provider_session_used": bool(imported.get("persistent_provider_session_used")),
        "provider_instance_reused_for_policy_and_wam_loop": bool(
            imported.get("provider_instance_reused_for_policy_and_wam_loop")
        ),
        "repeated_policy_calls_count": int(imported.get("repeated_policy_calls_count") or 0),
        "generated_next_observation_count": int(
            imported.get("generated_next_observation_count") or 0
        ),
        "live_wam_generation_success_count": live_wam_success_count,
        "learned_wam_model_success_count": learned_wam_success_count,
        "unitree_groot_n17_sonic_model_executed": bool(
            imported.get("unitree_groot_n17_sonic_model_executed")
        ),
        "unitree_groot_n17_sonic_policy_action_command_ran": bool(
            imported.get("unitree_groot_n17_sonic_policy_action_command_ran")
        ),
        "unitree_policy_action_command_ran": bool(imported.get("unitree_policy_action_command_ran")),
        "policy_action_model_command_ran": bool(imported.get("policy_action_model_command_ran")),
        "provider_output_replay_used": bool(imported.get("provider_output_replay_used")),
        "provider_output_resume_used": bool(provider_output_resume_used),
        "blockers": []
        if completed
        else imported.get("blockers") or ["persistent_session_provider_output_blocked"],
        "imported_provider_output_dir": str(extraction_dir),
        "imported_provider_output_path": str(imported_path) if imported_path.is_file() else None,
        "runpod_create_manifest_path": str(runpod_dir / "runpod_wam_async_create_manifest.json"),
        "runpod_poll_manifest_path": str(runpod_dir / "runpod_wam_async_poll_manifest.json"),
        "runpod_teardown_manifest_path": str(
            _runpod_teardown_manifest_path(runpod_dir, poll_manifest)
        ),
        "runpod_keepalive": runpod_keepalive,
        "provider_runtime_output_zip_path": str(output_zip),
        "runpod_live_wam_blocker_classification_path": str(
            job / "runpod_live_wam_blocker_classification.json"
        )
        if not completed
        else None,
        "classified_blocker": classification.get("classified_blocker"),
        "continuing_spend_from_this_run": bool(
            runpod_keepalive.get("continuing_spend_from_this_run")
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
        "policy_action_decoding_contract_path": postprocess.get(
            "policy_action_decoding_contract"
        ),
        "policy_action_bridge_readiness_path": postprocess.get(
            "policy_action_bridge_readiness"
        ),
        "wam_input_review_manifest_path": postprocess.get("wam_input_review_manifest"),
        "wam_input_review_contact_sheet_path": postprocess.get(
            "wam_input_review_contact_sheet"
        ),
        "rank_fidelity_calibration_requirement_path": postprocess.get(
            "rank_fidelity_calibration_requirement"
        ),
        "rank_fidelity_calibration_anchor_request_path": postprocess.get(
            "rank_fidelity_calibration_anchor_request"
        ),
        "rank_fidelity_small_calibration_set_path": postprocess.get(
            "rank_fidelity_small_calibration_set"
        ),
        "rank_fidelity_calibration_status": postprocess.get(
            "rank_fidelity_calibration_status"
        ),
        "rank_fidelity_calibration_blockers": _string_list(
            postprocess.get("rank_fidelity_calibration_blockers")
        ),
        "rank_fidelity_result_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "wam_rollout_visual_success": visual_success,
        "visual_quality_report_status": visual_report.get("status"),
        "visual_quality_blockers": visual_quality_blockers,
        "provider_completed_but_visual_quality_failed": provider_output_failed_visual_quality,
        "policy_evaluation_ranking_ready": bool(completed and visual_success),
        "policy_evaluation_ranking_status": "ready_for_visual_review"
        if completed and visual_success
        else "blocked_wam_visual_quality"
        if completed or provider_output_failed_visual_quality
        else "blocked_provider_runtime",
        "policy_evaluation_ranking_blockers": sorted(set(policy_ranking_blockers)),
        "wam_episode_consistency_request_path": postprocess.get("wam_episode_consistency_request"),
        "wam_consistency_checks_path": postprocess.get("wam_consistency_checks"),
        "forward_inverse_consistency_proven": bool(
            postprocess.get("forward_inverse_consistency_proven")
        ),
        "external_episode_consistency_scorer_ran": bool(
            postprocess.get("external_episode_consistency_scorer_ran")
        ),
        "external_episode_consistency_scorer_required": bool(
            postprocess.get("external_episode_consistency_scorer_required")
        ),
        "wam_episode_consistency_early_termination_recommended": bool(
            postprocess.get("wam_episode_consistency_early_termination_recommended")
        ),
        "wam_episode_consistency_blockers": _string_list(
            postprocess.get("wam_episode_consistency_blockers")
        ),
        "clean_frame_reanchoring": _mapping(imported.get("clean_frame_reanchoring")),
        "clean_frame_reanchor_event_count": int(imported.get("clean_frame_reanchor_event_count") or 0),
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
            "visually_useful_rollout": visual_success,
            "provider_completed_but_visual_quality_failed": provider_output_failed_visual_quality,
            "forward_inverse_consistency_proven": bool(
                postprocess.get("forward_inverse_consistency_proven")
            ),
            "external_episode_consistency_scorer_ran": bool(
                postprocess.get("external_episode_consistency_scorer_ran")
            ),
            "wam_episode_consistency_early_termination_recommended": bool(
                postprocess.get("wam_episode_consistency_early_termination_recommended")
            ),
            "forward_inverse_consistency_is_reliability_review_signal_only": True,
            "forward_inverse_consistency_does_not_prove_task_success": True,
            "rank_fidelity_calibration_required": True,
            "visual_review_ranking_is_not_real_world_rank_fidelity": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
            "accepted_anchor_manipulation_success_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
    return output, 0 if completed else 2


def run_persistent_session(
    *,
    policy_observation_path: str | Path,
    job_dir: str | Path | None = None,
    loop_step_count: int = 12,
    task_prompt: str | None = None,
    timeout_seconds: float = 3600.0,
    use_live_wam: bool = True,
    allow_structural_wam_fallback: bool | None = None,
    manipulation_pov_geometry_path: str | Path | None = None,
    placement_validation_path: str | Path | None = None,
    task_stance_plan_path: str | Path | None = None,
) -> tuple[dict[str, Any], int]:
    generated_at = utc_now_iso()
    job = _job_dir(job_dir)
    requested_loop_step_count = max(1, int(loop_step_count))
    allow_fallback = (
        _truthy(os.getenv(PERSISTENT_SESSION_ALLOW_STRUCTURAL_WAM_FALLBACK_ENV))
        if allow_structural_wam_fallback is None
        else bool(allow_structural_wam_fallback)
    )
    git_evidence = launch_provenance.git_worktree_evidence()
    launch_gate = launch_provenance.evaluate_dirty_tree_paid_launch_gate(
        git_evidence=git_evidence,
        allow_paid=_truthy(os.getenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH")),
        allow_dirty_paid_launch=_truthy(os.getenv(ALLOW_DIRTY_PAID_LAUNCH_ENV)),
    )
    if not launch_gate["launch_allowed"]:
        output = _blocked_payload(
            generated_at=generated_at,
            job_dir=job,
            blockers=launch_gate["blockers"],
            details={
                "git_evidence": git_evidence,
                "note": launch_gate["note"],
                "provider": "vast",
            },
        )
        write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
        return output, 2
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
            manipulation_pov_geometry_path=manipulation_pov_geometry_path,
            placement_validation_path=placement_validation_path,
            task_stance_plan_path=task_stance_plan_path,
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
                    "git_evidence": git_evidence,
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
                    ),
                    "git_evidence": git_evidence,
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
                min_compute_cap=_int_env(
                    "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_MIN_COMPUTE_CAP", 800
                ),
                poll_interval_seconds=_int_env(
                    "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_POLL_SECONDS", 15
                ),
                startup_timeout_seconds=_int_env(
                    "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_STARTUP_TIMEOUT_SECONDS", 1800
                ),
                heartbeat_no_progress_seconds=_int_env(
                    "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_HEARTBEAT_NO_PROGRESS_SECONDS",
                    _int_env(
                        "BLUEPRINT_VAST_HEARTBEAT_NO_PROGRESS_SECONDS",
                        DEFAULT_HEARTBEAT_NO_PROGRESS_SECONDS,
                    ),
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
                    "git_evidence": git_evidence,
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
            "git_evidence": git_evidence,
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
            "wam_evaluator_in_control_loop": bool(imported.get("wam_evaluator_in_control_loop")),
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
            "policy_action_decoding_contract_path": postprocess.get(
                "policy_action_decoding_contract"
            ),
            "policy_action_bridge_readiness_path": postprocess.get(
                "policy_action_bridge_readiness"
            ),
            "wam_input_review_manifest_path": postprocess.get("wam_input_review_manifest"),
            "wam_input_review_contact_sheet_path": postprocess.get(
                "wam_input_review_contact_sheet"
            ),
            "rank_fidelity_calibration_requirement_path": postprocess.get(
                "rank_fidelity_calibration_requirement"
            ),
            "rank_fidelity_calibration_anchor_request_path": postprocess.get(
                "rank_fidelity_calibration_anchor_request"
            ),
            "rank_fidelity_small_calibration_set_path": postprocess.get(
                "rank_fidelity_small_calibration_set"
            ),
            "rank_fidelity_calibration_status": postprocess.get(
                "rank_fidelity_calibration_status"
            ),
            "rank_fidelity_calibration_blockers": _string_list(
                postprocess.get("rank_fidelity_calibration_blockers")
            ),
            "rank_fidelity_result_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
            "wam_rollout_visual_success": bool(postprocess.get("wam_rollout_visual_success")),
            "wam_materialization_summary_path": postprocess.get("wam_materialization_summary"),
            "wam_episode_consistency_request_path": postprocess.get(
                "wam_episode_consistency_request"
            ),
            "wam_consistency_checks_path": postprocess.get("wam_consistency_checks"),
            "forward_inverse_consistency_proven": bool(
                postprocess.get("forward_inverse_consistency_proven")
            ),
            "external_episode_consistency_scorer_ran": bool(
                postprocess.get("external_episode_consistency_scorer_ran")
            ),
            "external_episode_consistency_scorer_required": bool(
                postprocess.get("external_episode_consistency_scorer_required")
            ),
            "wam_episode_consistency_early_termination_recommended": bool(
                postprocess.get("wam_episode_consistency_early_termination_recommended")
            ),
            "wam_episode_consistency_blockers": _string_list(
                postprocess.get("wam_episode_consistency_blockers")
            ),
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
                "forward_inverse_consistency_proven": bool(
                    postprocess.get("forward_inverse_consistency_proven")
                ),
                "external_episode_consistency_scorer_ran": bool(
                    postprocess.get("external_episode_consistency_scorer_ran")
                ),
                "wam_episode_consistency_early_termination_recommended": bool(
                    postprocess.get("wam_episode_consistency_early_termination_recommended")
                ),
                "forward_inverse_consistency_is_reliability_review_signal_only": True,
                "forward_inverse_consistency_does_not_prove_task_success": True,
                "rank_fidelity_calibration_required": True,
                "visual_review_ranking_is_not_real_world_rank_fidelity": True,
                "generated_world_rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
                "non_ranking_operational_claim_proven": False,
                "accepted_anchor_manipulation_success_proven": False,
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
    manipulation_pov_geometry_path: str | Path | None = None,
    placement_validation_path: str | Path | None = None,
    task_stance_plan_path: str | Path | None = None,
    max_wait_seconds: int | None = None,
) -> tuple[dict[str, Any], int]:
    generated_at = utc_now_iso()
    job = _completed_runpod_resume_job(job_dir) or _job_dir(job_dir)
    requested_loop_step_count = max(1, int(loop_step_count))
    allow_fallback = (
        _truthy(os.getenv(PERSISTENT_SESSION_ALLOW_STRUCTURAL_WAM_FALLBACK_ENV))
        if allow_structural_wam_fallback is None
        else bool(allow_structural_wam_fallback)
    )
    git_evidence = launch_provenance.git_worktree_evidence()
    resume_runpod_dir = job / "runpod_persistent_session_run"
    resume_output_zip = resume_runpod_dir / "runpod_provider_runtime_output.zip"
    resume_poll_manifest_path = resume_runpod_dir / "runpod_wam_async_poll_manifest.json"
    if resume_output_zip.is_file() and resume_poll_manifest_path.is_file():
        resume_poll_manifest = _read_json(resume_poll_manifest_path)
        if resume_poll_manifest.get("status") == "completed":
            return _finalize_runpod_persistent_session_output(
                job=job,
                generated_at=generated_at,
                policy_observation_path=policy_observation_path,
                git_evidence=git_evidence,
                poll_manifest=resume_poll_manifest,
                runpod_dir=resume_runpod_dir,
                output_zip=resume_output_zip,
                provider_output_resume_used=True,
            )
    launch_gate = launch_provenance.evaluate_dirty_tree_paid_launch_gate(
        git_evidence=git_evidence,
        allow_paid=True,
        allow_dirty_paid_launch=_truthy(os.getenv(ALLOW_DIRTY_PAID_LAUNCH_ENV)),
    )
    if not launch_gate["launch_allowed"]:
        output = _blocked_payload(
            generated_at=generated_at,
            job_dir=job,
            blockers=launch_gate["blockers"],
            details={
                "git_evidence": git_evidence,
                "note": launch_gate["note"],
                "provider": "runpod",
            },
        )
        write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
        return output, 2
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
    previous_runpod_teardown_action = os.environ.get(RUNPOD_WAM_TEARDOWN_ACTION_ENV)
    os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"] = inner_policy_command
    os.environ[PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV] = inner_policy_command
    os.environ[INNER_POLICY_COMMAND_ENV] = inner_policy_command
    os.environ.setdefault(RUNPOD_WAM_TEARDOWN_ACTION_ENV, "keep_on_success")
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
            manipulation_pov_geometry_path=manipulation_pov_geometry_path,
            placement_validation_path=placement_validation_path,
            task_stance_plan_path=task_stance_plan_path,
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
                    "git_evidence": git_evidence,
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
                    "git_evidence": git_evidence,
                },
            )
            write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
            return output, 2
        staging_dir = job / "object_store_staging"
        runpod_dir = job / "runpod_persistent_session_run"
        output_zip = runpod_dir / "runpod_provider_runtime_output.zip"
        default_runpod_image = (
            DEFAULT_RUNPOD_UNITREE_GROOT_SONIC_WAM_PUBLIC_IMAGE
            if runpod_provider_bundle_kind == "wam"
            else DEFAULT_WAM_PUBLIC_IMAGE
        )
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
                or default_runpod_image
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
                    "git_evidence": git_evidence,
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
            classified_blocker = _string(classification.get("classified_blocker"))
            provider_command_blockers = _string_list(
                poll_manifest.get("provider_command_blockers")
            )
            fallback_blockers = [
                "runpod_persistent_session_still_running"
                if poll_manifest_summary["status"] == "running"
                or poll_manifest_summary["provider_command_status"] == "running"
                or poll_manifest_summary["continuing_spend_from_this_run"]
                else "persistent_session_runpod_provider_blocked"
            ]
            blocked_result_blockers = [
                item
                for item in [classified_blocker, *provider_command_blockers]
                if item and item != "none"
            ] or fallback_blockers
            output = _blocked_payload(
                generated_at=generated_at,
                job_dir=job,
                blockers=blocked_result_blockers,
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
                    "provider_command_blockers": provider_command_blockers,
                    "poll_manifest": poll_manifest_summary,
                    "runpod_live_wam_blocker_classification_path": str(
                        job / "runpod_live_wam_blocker_classification.json"
                    ),
                    "classified_blocker": classification.get("classified_blocker"),
                    "git_evidence": git_evidence,
                },
            )
            write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
            return output, 2
        return _finalize_runpod_persistent_session_output(
            job=job,
            generated_at=generated_at,
            policy_observation_path=policy_observation_path,
            git_evidence=git_evidence,
            poll_manifest=poll_manifest,
            runpod_dir=runpod_dir,
            output_zip=output_zip,
        )
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
        if previous_runpod_teardown_action is None:
            os.environ.pop(RUNPOD_WAM_TEARDOWN_ACTION_ENV, None)
        else:
            os.environ[RUNPOD_WAM_TEARDOWN_ACTION_ENV] = previous_runpod_teardown_action


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-observation", required=True)
    parser.add_argument("--job-dir")
    parser.add_argument("--provider", choices=("runpod", "vast"), default="vast")
    parser.add_argument("--loop-step-count", type=int, default=12)
    parser.add_argument("--task-prompt")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--disable-live-wam", action="store_true")
    parser.add_argument("--allow-structural-wam-fallback", action="store_true")
    parser.add_argument("--manipulation-pov-geometry")
    parser.add_argument("--placement-validation")
    parser.add_argument("--task-stance-plan")
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
        manipulation_pov_geometry_path=args.manipulation_pov_geometry,
        placement_validation_path=args.placement_validation,
        task_stance_plan_path=args.task_stance_plan,
    )
    print(json.dumps(result, sort_keys=True))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
