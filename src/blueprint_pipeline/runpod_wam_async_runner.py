"""Short-lived RunPod runner for WAM and Unitree policy provider bundles."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from .common import ensure_dir, parse_bool, utc_now_iso, write_json
from .groot_oscar_runpod_carrier_volume import verify_carrier_volume_admission
from .paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .paid_lane_guard import (
    PreSpendPreflightBlocked,
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    close_pending_teardown,
    image_contract_from_ref,
    open_pending_teardown,
    require_pre_spend_preflight,
)
from .provider_reliability_manifest import (
    TEARDOWN_STATUS_SOURCE_PROVIDER_API,
    build_provider_reliability_manifest,
    build_teardown_proof,
    evaluate_post_marker_stall,
)
from .runpod_provider_adapter import (
    RUNPOD_API_GATE_ENV,
    RUNPOD_EXISTING_POD_ID_ENV,
    RUNPOD_API_KEY_FILE_ENV,
    RUNPOD_API_KEY_ENV,
    RUNPOD_REST_API_BASE,
)
from .runpod_wam_launch_contract import (
    build_pod_payload as _build_pod_payload,
    confirm_provider_lane_handoff_no_allocation as _confirm_provider_lane_handoff_no_allocation,
    extract_pod_id as _extract_pod_id,
    read_compatible_warm_candidate as _read_compatible_warm_candidate_contract,
    redacted_payload_summary as _redacted_payload_summary,
    selected_existing_pod_id as _selected_existing_pod_id,
    update_provider_lane_handoff_receipt as _update_provider_lane_handoff_receipt,
)
from .secret_artifact_policy import (
    SECRET_PATH_DISCLOSURE_POLICY,
    redacted_secret_file_status,
    secret_path_disclosure_policy,
)
from .vast_bundle_staging import (
    BUNDLE_ROUTE,
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_SECRET_ENV_FILE,
    DEFAULT_TOKEN_FILE,
    OUTPUT_ROUTE,
    _read_or_create_token,
    _url_with_token,
    prepare_vast_bundle_staging,
    run_local_staging_self_test,
    verify_public_staging_urls,
)
from .vast_provider_adapter import _inspect_provider_runtime_output_zip
from .vast_wam_authorized_runner import DEFAULT_WAM_PUBLIC_IMAGE
from .wam_async_runner_common import (
    download_url_to_file,
    read_json_mapping as _read_json,
    read_sensitive_url_file as _read_sensitive_url_file,
    redact_provider_url as _redact_provider_url,
)


RUNPOD_WAM_STATE_SCHEMA_VERSION = "runpod_wam_async_state.v1"
RUNPOD_WAM_CREATE_SCHEMA_VERSION = "runpod_wam_async_create_manifest.v1"
RUNPOD_WAM_POLL_SCHEMA_VERSION = "runpod_wam_async_poll_manifest.v1"
RUNPOD_WAM_DELETE_SCHEMA_VERSION = "runpod_wam_async_delete_manifest.v1"
RUNPOD_WAM_STOP_SCHEMA_VERSION = "runpod_wam_async_stop_manifest.v1"
RUNPOD_WAM_WARM_CANDIDATE_SCHEMA_VERSION = "runpod_wam_warm_candidate.v1"
RUNPOD_WAM_PROVIDER_RELIABILITY_MANIFEST_NAME = "provider_reliability_manifest.json"
RUNPOD_WAM_POST_MARKER_NO_PROGRESS_TIMEOUT_ENV = (
    "BLUEPRINT_RUNPOD_WAM_POST_MARKER_NO_PROGRESS_TIMEOUT_SECONDS"
)
GENERIC_POST_MARKER_NO_PROGRESS_TIMEOUT_ENV = "BLUEPRINT_POST_MARKER_NO_PROGRESS_TIMEOUT_SECONDS"
DEFAULT_RUNPOD_WAM_POST_MARKER_NO_PROGRESS_TIMEOUT_SECONDS = 900
RUNPOD_WAM_TEARDOWN_ACTION_ENV = "BLUEPRINT_RUNPOD_WAM_TEARDOWN_ACTION"
RUNPOD_WAM_EXISTING_POD_ID_ENV = "BLUEPRINT_RUNPOD_WAM_EXISTING_POD_ID"
RUNPOD_WAM_WARM_CANDIDATE_FILE_ENV = "BLUEPRINT_RUNPOD_WAM_WARM_CANDIDATE_FILE"
RUNPOD_WAM_DISABLE_WARM_CANDIDATE_ENV = "BLUEPRINT_RUNPOD_WAM_DISABLE_WARM_CANDIDATE"
RUNPOD_WAM_RUNNING_CANDIDATE_RUNTIME_ABSENT_MAX_SECONDS_ENV = (
    "BLUEPRINT_RUNPOD_WAM_RUNNING_CANDIDATE_RUNTIME_ABSENT_MAX_SECONDS"
)
RUNPOD_POD_LAUNCH_GATE_ENV = "BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH"
RUNPOD_WAM_MAX_SPEND_USD_ENV = "BLUEPRINT_RUNPOD_WAM_MAX_SPEND_USD"
RUNPOD_UNITREE_GROOT_SONIC_FULL_LOOP_OVERRIDE_ENV = (
    "BLUEPRINT_ALLOW_UNITREE_GROOT_N17_SONIC_RUNPOD_FULL_LOOP"
)
RUNPOD_UNITREE_GROOT_SONIC_MAX_UNGATED_LOOP_STEPS = 2
RUNPOD_PROVIDER_BUNDLE_KINDS = ("wam", "unitree_unifolm", "unitree_groot_n17_sonic")
RUNPOD_WAM_LANE = "runpod_wam_async"
RUNPOD_TERMINAL_POD_STATUSES = {"not_found", "EXITED", "TERMINATED", "FAILED", "STOPPED"}
RUNPOD_ACTIVE_POD_STATUSES = {
    "CREATED",
    "PENDING",
    "QUEUED",
    "STARTING",
    "INITIALIZING",
    "PROVISIONING",
    "RUNNING",
    "RESTARTING",
    "pending_api_visibility",
}
RUNPOD_WAM_PROVIDER_OUTPUT_NONTERMINAL_STATUSES = {"running", "starting", "in_progress"}
RUNPOD_WAM_PROVIDER_OUTPUT_SUCCESS_STATUSES = {"completed", "success", "succeeded"}
RUNPOD_WAM_PROVIDER_OUTPUT_FAILURE_STATUSES = {
    "blocked",
    "failed",
    "error",
    "timeout",
    "timed_out",
}
RUNPOD_WAM_PROVIDER_OUTPUT_RUNTIME_RESULT_NAMES = {
    "wam": (
        "wam_runtime_result.json",
        "isaac_runtime_result.json",
    ),
    "unitree_unifolm": ("unitree_unifolm_policy_provider_output.json",),
    "unitree_groot_n17_sonic": (
        "unitree_groot_n17_sonic_policy_provider_output.json",
        "unitree_groot_n17_sonic_wam_persistent_session_output.json",
        "wam_runtime_result.json",
    ),
}
RUNPOD_WAM_PROVIDER_ENTRYPOINT_MANIFEST_NAME = "runpod_wam_provider_entrypoint_execution.json"
RUNPOD_WAM_PROVIDER_HEARTBEAT_MANIFEST_NAME = "wam_provider_output.json"
DEFAULT_GPU_TYPE_IDS = (
    "NVIDIA A40",
    "NVIDIA RTX A5000",
    "NVIDIA RTX A6000",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
)
DEFAULT_HF_TOKEN_FILES = (
    "~/.blueprint-secrets/hf_token",
    "~/.blueprint-secrets/hf_token.txt",
    "~/.blueprint-secrets/huggingface_token",
    "~/.blueprint-secrets/huggingface_token.txt",
)
DEFAULT_RUNPOD_WAM_WARM_CANDIDATE_FILE = "~/.blueprint-cache/runpod_wam_warm_candidate.json"
PROVIDER_RUNTIME_CONFIG_ENV_KEYS = (
    "BLUEPRINT_OSCAR_WAM_ATTEMPT_TRANSFORMER_ENGINE_INSTALL",
    "BLUEPRINT_OSCAR_WAM_SKIP_RUNTIME_PIP_INSTALL",
    "BLUEPRINT_OSCAR_WAM_TRANSFORMER_ENGINE_STRATEGY",
    "BLUEPRINT_OSCAR_WAM_NUM_STEPS",
    "BLUEPRINT_OSCAR_WAM_GUIDANCE",
    "BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE",
    "BLUEPRINT_OSCAR_WAM_NUM_FRAMES",
    "BLUEPRINT_OSCAR_WAM_HEIGHT",
    "BLUEPRINT_OSCAR_WAM_WIDTH",
    "BLUEPRINT_OSCAR_WAM_FPS",
    "BLUEPRINT_OSCAR_WAM_CONDITIONING_MODE",
    "BLUEPRINT_OSCAR_WAM_RGB_CONTEXT_MODE",
    "BLUEPRINT_OSCAR_WAM_OFFICIAL_CASE_SMOKE",
    "BLUEPRINT_OSCAR_WAM_OFFICIAL_CASE_RGB_VIDEO",
    "BLUEPRINT_OSCAR_WAM_OFFICIAL_CASE_USE_SCRIPT",
    "BLUEPRINT_OSCAR_WAM_CONDITIONING_BACKGROUND_ALPHA",
    "BLUEPRINT_OSCAR_WAM_CONDITIONING_VOID_THRESHOLD",
    "BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS",
    "BLUEPRINT_OSCAR_WAM_ENABLE_HF_TRANSFER",
    "BLUEPRINT_WAM_PROVIDER_ALLOW_BREAK_SYSTEM_PACKAGES",
    "BLUEPRINT_WAM_PROVIDER_DISABLE_VENV",
    "BLUEPRINT_RUNPOD_WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS",
)
UNITREE_GROOT_SONIC_RUNTIME_CONFIG_ENV_KEYS = (
    *PROVIDER_RUNTIME_CONFIG_ENV_KEYS,
    "BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC",
    "BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT",
    "BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_COMMAND",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PERSISTENT_INNER_POLICY_COMMAND",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_AUTO_START_POLICY_SERVER",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON_INSTALL_REQUIREMENTS",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SKIP_SYSTEM_PYTHON_DEPS_INSTALL",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON_DEPS_TIMEOUT_SECONDS",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SERVER_STARTUP_TIMEOUT_SECONDS",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_UV_SYNC_TIMEOUT_SECONDS",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_MODEL_SNAPSHOT_TIMEOUT_SECONDS",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_MODEL_SNAPSHOT_ATTEMPTS",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_MODEL_SNAPSHOT_MAX_WORKERS",
    "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_ENTRYPOINT_TIMEOUT_SECONDS",
    "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_WRAPPER_WATCHDOG_SECONDS",
    "BLUEPRINT_PERSISTENT_SESSION_WAM_STEP_TIMEOUT_SECONDS",
)
UNITREE_UNIFOLM_RUNTIME_CONFIG_ENV_KEYS = (
    "BLUEPRINT_UNITREE_UNIFOLM_MODE",
    "BLUEPRINT_UNITREE_UNIFOLM_COMMAND",
    "BLUEPRINT_UNITREE_UNIFOLM_CHECKPOINT",
    "BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT",
    "BLUEPRINT_UNITREE_UNIFOLM_SOURCE_ROOT",
    "BLUEPRINT_UNITREE_UNIFOLM_TIMEOUT_SECONDS",
    "BLUEPRINT_UNITREE_UNIFOLM_VLA_ATTENTION_IMPLEMENTATION",
    "BLUEPRINT_UNITREE_UNIFOLM_ALLOW_HF_DOWNLOAD",
    "BLUEPRINT_UNITREE_UNIFOLM_MODEL_CACHE_ROOT",
    "BLUEPRINT_UNITREE_UNIFOLM_VLA_REPO",
    "BLUEPRINT_UNITREE_UNIFOLM_VLM_REPO",
    "BLUEPRINT_UNITREE_UNIFOLM_VLA_SERVER_STARTUP_TIMEOUT_SECONDS",
)
UNITREE_UNIFOLM_RUNTIME_CONFIG_ALIASES = {
    "BLUEPRINT_UNITREE_UNIFOLM_COMMAND": ("BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",),
    "BLUEPRINT_UNITREE_UNIFOLM_CHECKPOINT": (
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT",
        "BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT",
    ),
    "BLUEPRINT_UNITREE_UNIFOLM_SOURCE_ROOT": (
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_SOURCE_ROOT",
    ),
}
UNITREE_UNIFOLM_RUNTIME_CONFIG_DEFAULTS = {
    "BLUEPRINT_UNITREE_UNIFOLM_MODE": "vla",
    "BLUEPRINT_UNITREE_UNIFOLM_COMMAND": "/usr/local/bin/run_unitree_unifolm_vla_policy_once",
    "BLUEPRINT_UNITREE_UNIFOLM_CHECKPOINT": "unitreerobotics/UnifoLM-VLA-Base",
    "BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT": "unitreerobotics/UnifoLM-VLM-Base",
    "BLUEPRINT_UNITREE_UNIFOLM_SOURCE_ROOT": "/opt/unifolm-vla",
    "BLUEPRINT_UNITREE_UNIFOLM_TIMEOUT_SECONDS": "1800",
    "BLUEPRINT_UNITREE_UNIFOLM_VLA_ATTENTION_IMPLEMENTATION": "sdpa",
    "BLUEPRINT_UNITREE_UNIFOLM_ALLOW_HF_DOWNLOAD": "true",
}


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(_string(os.getenv(name)) or default)
    except ValueError:
        return int(default)


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _env_truthy(name: str) -> bool:
    return _string(os.getenv(name)).lower() in {"1", "true", "yes", "on"}


def _runpod_wam_post_marker_timeout_seconds(explicit: int | None = None) -> int:
    if explicit is not None:
        return max(0, int(explicit))
    for env_name in (
        RUNPOD_WAM_POST_MARKER_NO_PROGRESS_TIMEOUT_ENV,
        GENERIC_POST_MARKER_NO_PROGRESS_TIMEOUT_ENV,
    ):
        raw = _string(os.getenv(env_name))
        if raw:
            try:
                return max(0, int(raw))
            except ValueError:
                return DEFAULT_RUNPOD_WAM_POST_MARKER_NO_PROGRESS_TIMEOUT_SECONDS
    return DEFAULT_RUNPOD_WAM_POST_MARKER_NO_PROGRESS_TIMEOUT_SECONDS


def _read_unitree_groot_sonic_bundle_input(bundle_path: Path) -> dict[str, Any]:
    if not bundle_path.is_file() or not zipfile.is_zipfile(bundle_path):
        return {}
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            input_name = "provider_runtime/persistent_session_input.json"
            if input_name not in set(archive.namelist()):
                return {}
            payload = json.loads(archive.read(input_name).decode("utf-8") or "{}")
    except (OSError, ValueError, zipfile.BadZipFile):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _unitree_groot_sonic_full_loop_create_guard(
    *,
    bundle_path: Path,
    provider_bundle_kind: str,
) -> dict[str, Any]:
    payload = _read_unitree_groot_sonic_bundle_input(bundle_path)
    schema_version = _string(payload.get("schema_version"))
    is_unitree_groot_sonic_bundle = bool(
        provider_bundle_kind == "unitree_groot_n17_sonic"
        or schema_version == "unitree_groot_n17_sonic_wam_persistent_session_input.v1"
    )
    if not is_unitree_groot_sonic_bundle:
        return {"status": "not_applicable", "raw_secret_values_recorded": False}
    try:
        loop_step_count = max(1, int(payload.get("loop_step_count") or 1))
    except (TypeError, ValueError):
        loop_step_count = 1
    return {
        "status": "allowed",
        "requested_loop_step_count": loop_step_count,
        "previous_max_loop_step_count_without_override": (
            RUNPOD_UNITREE_GROOT_SONIC_MAX_UNGATED_LOOP_STEPS
        ),
        "override_env": RUNPOD_UNITREE_GROOT_SONIC_FULL_LOOP_OVERRIDE_ENV,
        "provider_bundle_kind": provider_bundle_kind,
        "bundle_input_schema_version": schema_version,
        "full_loop_launch_is_default": True,
        "raw_secret_values_recorded": False,
    }


def _runpod_wam_prelaunch_spend_guard(
    *,
    max_spend_usd: float | None,
    allow_paid_runpod_launch: bool,
    gpu_type_ids: Sequence[str],
    container_disk_gb: int,
    volume_gb: int,
) -> dict[str, Any]:
    env_budget = _float_or_none(os.getenv(RUNPOD_WAM_MAX_SPEND_USD_ENV))
    requested_budget = max_spend_usd if max_spend_usd is not None else env_budget
    api_gate_present = _env_truthy(RUNPOD_API_GATE_ENV)
    pod_gate_present = _env_truthy(RUNPOD_POD_LAUNCH_GATE_ENV)
    blockers: list[str] = []
    if not allow_paid_runpod_launch:
        blockers.append("paid_runpod_launch_not_authorized_by_runner_flag")
    if not api_gate_present:
        blockers.append(f"missing_env_{RUNPOD_API_GATE_ENV}")
    if not pod_gate_present:
        blockers.append(f"missing_env_{RUNPOD_POD_LAUNCH_GATE_ENV}")
    if requested_budget is None:
        blockers.append("runpod_wam_max_spend_usd_missing")
    elif requested_budget <= 0:
        blockers.append("runpod_wam_max_spend_usd_must_be_positive")
    if not gpu_type_ids:
        blockers.append("runpod_wam_gpu_type_ids_missing")
    if container_disk_gb <= 0:
        blockers.append("runpod_wam_container_disk_gb_invalid")
    if volume_gb < 0:
        blockers.append("runpod_wam_volume_gb_invalid")
    can_launch = not blockers
    return {
        "schema_version": "runpod_wam_prelaunch_spend_guard.v1",
        "status": "passed" if can_launch else "blocked",
        "required_before_provider_launch": True,
        "can_launch": can_launch,
        "requested_budget_usd": requested_budget,
        "budget_source": "argument"
        if max_spend_usd is not None
        else ("env" if env_budget is not None else "missing"),
        "single_pod_launch": True,
        "max_active_workers": 1,
        "gpu_type_ids": list(gpu_type_ids),
        "container_disk_gb": container_disk_gb,
        "volume_gb": volume_gb,
        "checks": {
            "allow_paid_runpod_launch_flag_present": allow_paid_runpod_launch,
            f"env_{RUNPOD_API_GATE_ENV}_present": api_gate_present,
            f"env_{RUNPOD_POD_LAUNCH_GATE_ENV}_present": pod_gate_present,
            "requested_budget_declared": requested_budget is not None,
            "requested_budget_positive": requested_budget is not None and requested_budget > 0,
            "single_pod_launch": True,
            "teardown_command_written_after_create": True,
        },
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "can_launch_is_spend_gate_only": True,
            "can_launch_is_not_provider_runtime_success": True,
            "can_launch_is_not_task_success": True,
            "no_runpod_api_call_before_can_launch": True,
        },
    }


def _state_path(job_dir: Path) -> Path:
    return job_dir / "runpod_wam_async_state.json"


def _read_runpod_api_key() -> tuple[str, dict[str, Any]]:
    env_value = _string(os.getenv(RUNPOD_API_KEY_ENV))
    if env_value:
        return env_value, {
            "api_key_configured": True,
            "api_key_source": RUNPOD_API_KEY_ENV,
            "api_key_file_configured": False,
            "raw_secret_values_recorded": False,
        }
    key_file = Path(
        _string(os.getenv(RUNPOD_API_KEY_FILE_ENV))
        or "~/.blueprint-secrets/runpod_api_key"
    ).expanduser()
    mode = oct(key_file.stat().st_mode & 0o777) if key_file.exists() else None
    try:
        key = key_file.read_text(encoding="utf-8").strip() if key_file.is_file() else ""
    except OSError as exc:
        status = redacted_secret_file_status(
            key_file,
            env_name=RUNPOD_API_KEY_FILE_ENV,
            raw_secret_field="raw_secret_values_recorded",
        )
        status.update({
            "api_key_configured": False,
            "api_key_source": RUNPOD_API_KEY_FILE_ENV,
            "api_key_file_configured": True,
            "api_key_file_mode": mode,
            "api_key_file_read_error": type(exc).__name__,
        })
        return "", status
    status = redacted_secret_file_status(
        key_file,
        env_name=RUNPOD_API_KEY_FILE_ENV,
        raw_secret_field="raw_secret_values_recorded",
    )
    status.update({
        "api_key_configured": bool(key),
        "api_key_source": RUNPOD_API_KEY_FILE_ENV if key else None,
        "api_key_file_configured": True,
        "api_key_file_mode": mode,
        "api_key_file_mode_is_0600": mode == "0o600",
    })
    return key, status


def _secret_file_meta(path: Path, *, label: str, source: str) -> dict[str, Any]:
    status = redacted_secret_file_status(
        path,
        path_source=source,
        raw_secret_field="raw_secret_values_recorded",
    )
    value_present = False
    read_error = None
    if path.is_file():
        try:
            value_present = bool(path.read_text(encoding="utf-8").strip())
        except OSError as exc:
            read_error = type(exc).__name__
    status.update({
        "label": label,
        "source": source,
        "value_present": value_present,
        "read_error": read_error,
    })
    return status


def _read_model_secret_env() -> tuple[dict[str, str], dict[str, Any]]:
    env_values: dict[str, str] = {}
    candidates: list[dict[str, Any]] = []
    explicit_files = [
        ("HF_TOKEN_FILE", _string(os.getenv("HF_TOKEN_FILE"))),
        ("HUGGING_FACE_HUB_TOKEN_FILE", _string(os.getenv("HUGGING_FACE_HUB_TOKEN_FILE"))),
        ("BLUEPRINT_HF_TOKEN_FILE", _string(os.getenv("BLUEPRINT_HF_TOKEN_FILE"))),
    ]
    file_candidates: list[tuple[str, Path]] = [
        (label, Path(value).expanduser())
        for label, value in explicit_files
        if value
    ]
    file_candidates.extend(
        ("default_secret_file", Path(item).expanduser()) for item in DEFAULT_HF_TOKEN_FILES
    )
    selected: dict[str, Any] | None = None
    for source, path in file_candidates:
        meta = _secret_file_meta(path, label="hf_token", source=source)
        candidates.append(meta)
        if selected is None and meta.get("present") and meta.get("value_present"):
            selected = meta
    configured_selected_file = selected
    configured_env = _string(os.getenv("HF_TOKEN"))
    if configured_env:
        env_values["HF_TOKEN"] = configured_env
        env_values["HUGGING_FACE_HUB_TOKEN"] = configured_env
        return env_values, {
            "schema_version": "runpod_wam_model_secret_env.v1",
            "status": "configured",
            "env_keys_forwarded": sorted(env_values),
            "source": configured_selected_file.get("source")
            if configured_selected_file
            else "HF_TOKEN",
            "selected_file": configured_selected_file,
            "candidate_files": candidates,
            "raw_secret_values_recorded": False,
            "secret_hashes_recorded": False,
        }
    selected = None
    for source, path in file_candidates:
        meta = next(
            (
                row
                for row in candidates
                if row.get("source") == source and row.get("path") == str(path)
            ),
            _secret_file_meta(path, label="hf_token", source=source),
        )
        if meta.get("present") and meta.get("value_present"):
            try:
                token = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            env_values["HF_TOKEN"] = token
            env_values["HUGGING_FACE_HUB_TOKEN"] = token
            selected = meta
            break
    return env_values, {
        "schema_version": "runpod_wam_model_secret_env.v1",
        "status": "configured" if env_values else "not_configured",
        "env_keys_forwarded": sorted(env_values),
        "source": selected.get("source") if selected else None,
        "selected_file": selected,
        "candidate_files": candidates,
        "raw_secret_values_recorded": False,
        "secret_hashes_recorded": False,
    }


def _provider_runtime_config_keys(provider_bundle_kind: str) -> tuple[str, ...]:
    if provider_bundle_kind == "unitree_unifolm":
        return UNITREE_UNIFOLM_RUNTIME_CONFIG_ENV_KEYS
    if provider_bundle_kind == "unitree_groot_n17_sonic":
        return UNITREE_GROOT_SONIC_RUNTIME_CONFIG_ENV_KEYS
    if (
        provider_bundle_kind == "wam"
        and os.getenv("BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC", "")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    ):
        return UNITREE_GROOT_SONIC_RUNTIME_CONFIG_ENV_KEYS
    return PROVIDER_RUNTIME_CONFIG_ENV_KEYS


def _read_provider_runtime_config_env(
    provider_bundle_kind: str = "wam",
) -> tuple[dict[str, str], dict[str, Any]]:
    if provider_bundle_kind not in RUNPOD_PROVIDER_BUNDLE_KINDS:
        raise ValueError(f"unsupported_provider_bundle_kind:{provider_bundle_kind}")
    env_values: dict[str, str] = {}
    value_sources: dict[str, str] = {}
    for key in _provider_runtime_config_keys(provider_bundle_kind):
        value = _string(os.getenv(key))
        if not value:
            for alias in UNITREE_UNIFOLM_RUNTIME_CONFIG_ALIASES.get(key, ()):
                alias_value = _string(os.getenv(alias))
                if alias_value:
                    value = alias_value
                    value_sources[key] = alias
                    break
        if not value and provider_bundle_kind == "unitree_unifolm":
            value = UNITREE_UNIFOLM_RUNTIME_CONFIG_DEFAULTS.get(key, "")
            if value:
                value_sources[key] = "image_default"
        if value:
            env_values[key] = value
            value_sources.setdefault(key, key)
    return env_values, {
        "schema_version": "runpod_wam_provider_runtime_config_env.v1",
        "provider_bundle_kind": provider_bundle_kind,
        "status": "configured" if env_values else "not_configured",
        "env_keys_forwarded": sorted(env_values),
        "values": dict(sorted(env_values.items())),
        "value_sources": dict(sorted(value_sources.items())),
        "raw_secret_values_recorded": False,
        "secret_hashes_recorded": False,
    }


def _runpod_request(
    *,
    method: str,
    path: str,
    api_key: str,
    payload: Mapping[str, Any] | None = None,
    timeout_seconds: int = 45,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(dict(payload or {})).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{RUNPOD_REST_API_BASE}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "BlueprintRunPodWamAsyncRunner/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = int(getattr(response, "status", 200))
        text = response.read().decode("utf-8", errors="replace")
    if not text.strip():
        return status, {}
    parsed = json.loads(text)
    return status, dict(parsed) if isinstance(parsed, Mapping) else {"response": parsed}


def _provider_shell_script(provider_bundle_kind: str = "wam") -> str:
    if provider_bundle_kind == "unitree_groot_n17_sonic":
        return r"""
set -euo pipefail
echo BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_PERSISTENT_PROVIDER_STARTED
WORK_DIR="${BLUEPRINT_RUNPOD_PROVIDER_WORK_DIR:-/workspace/blueprint_unitree_groot_sonic_persistent_provider}"
BUNDLE_URL="${BLUEPRINT_EVAL_MANIFEST_URI:-}"
OUTPUT_PUT_URL="${BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL:-${BLUEPRINT_ARTIFACT_OUTPUT_URI:-}}"
export WORK_DIR BUNDLE_URL OUTPUT_PUT_URL
mkdir -p "$WORK_DIR"
mkdir -p "$WORK_DIR/runtime_output"
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR="$WORK_DIR/runtime_output"
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT="$WORK_DIR/runtime_output/unitree_groot_n17_sonic_policy_provider_output.json"
upload_unitree_outer_blocker() {
  set +e
  OUTER_RC="${1:-1}" python - <<'PY'
import json
import os
import urllib.request
import zipfile
from pathlib import Path

output_dir = Path(os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)
out = Path(os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT"])
out.write_text(json.dumps({
    "schema_version": "unitree_groot_n17_sonic_wam_persistent_session_output.v1",
    "status": "blocked",
    "policy_id": "unitree_groot_n17_sonic_policy",
    "persistent_provider_session_used": True,
    "unitree_groot_n17_sonic_model_executed": False,
    "unitree_groot_n17_sonic_policy_action_command_ran": False,
    "policy_action_model_command_ran": False,
    "provider_output_replay_used": False,
    "blockers": ["runpod_unitree_groot_sonic_outer_bootstrap_failed_before_inner_wrapper_result"],
    "outer_returncode": int(os.environ.get("OUTER_RC", "1") or 1),
    "raw_credentials_written_to_artifacts": False,
    "secret_hashes_written_to_artifacts": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
zip_path = Path(os.environ["WORK_DIR"]) / "unitree_groot_n17_sonic_provider_runtime_output.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(output_dir).as_posix())
print("BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_OUTER_BLOCKER_ZIP_WRITTEN:%d" % zip_path.stat().st_size)
put_url = os.environ.get("OUTPUT_PUT_URL", "")
if put_url:
    request = urllib.request.Request(
        put_url,
        data=zip_path.read_bytes(),
        method="PUT",
        headers={"Content-Type": "application/zip"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read()
    print("BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_OUTER_BLOCKER_UPLOAD_OK")
PY
}
trap 'rc=$?; if [ "$rc" -ne 0 ]; then upload_unitree_outer_blocker "$rc"; fi; exit "$rc"' EXIT
if [ -z "$BUNDLE_URL" ]; then echo BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_BLOCKED:bundle_url_missing; exit 20; fi
if [ -z "$OUTPUT_PUT_URL" ]; then echo BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_BLOCKED:output_put_url_missing; exit 21; fi
python - <<'PY'
import os
import urllib.request
from pathlib import Path

target = Path(os.environ["WORK_DIR"]) / "unitree_groot_n17_sonic_wam_persistent_session_bundle.zip"
with urllib.request.urlopen(os.environ["BUNDLE_URL"], timeout=300) as response:
    target.write_bytes(response.read())
print("BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_BUNDLE_DOWNLOADED:%d" % target.stat().st_size)
PY
rm -rf "$WORK_DIR/unitree_groot_n17_sonic_provider_bundle" "$WORK_DIR/unitree_groot_n17_sonic_provider_runtime_output.zip"
python -m zipfile -e "$WORK_DIR/unitree_groot_n17_sonic_wam_persistent_session_bundle.zip" "$WORK_DIR/unitree_groot_n17_sonic_provider_bundle"
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_BUNDLE_DIR="$WORK_DIR/unitree_groot_n17_sonic_provider_bundle"
bash "$WORK_DIR/unitree_groot_n17_sonic_provider_bundle/provider_runtime/run_unitree_groot_n17_sonic_runpod_wrapper.sh"
"""
    if provider_bundle_kind == "unitree_unifolm":
        return r"""
set -euo pipefail
echo BLUEPRINT_RUNPOD_UNITREE_UNIFOLM_PROVIDER_STARTED
WORK_DIR="${BLUEPRINT_RUNPOD_PROVIDER_WORK_DIR:-/workspace/blueprint_unitree_unifolm_provider}"
BUNDLE_URL="${BLUEPRINT_EVAL_MANIFEST_URI:-}"
OUTPUT_PUT_URL="${BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL:-}"
export WORK_DIR BUNDLE_URL OUTPUT_PUT_URL
if [ -z "$BUNDLE_URL" ]; then echo BLUEPRINT_RUNPOD_UNITREE_UNIFOLM_BLOCKED:bundle_url_missing; exit 20; fi
if [ -z "$OUTPUT_PUT_URL" ]; then echo BLUEPRINT_RUNPOD_UNITREE_UNIFOLM_BLOCKED:output_put_url_missing; exit 21; fi
mkdir -p "$WORK_DIR"
if command -v apt-get >/dev/null 2>&1; then
  if ! command -v git >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
    timeout 300 apt-get update >/tmp/blueprint_runpod_apt_update.log 2>&1 || true
    DEBIAN_FRONTEND=noninteractive timeout 600 apt-get install -y git ffmpeg curl ca-certificates >/tmp/blueprint_runpod_apt_install.log 2>&1 || true
  fi
fi
python - <<'PY'
import os
import urllib.request
from pathlib import Path
target = Path(os.environ["WORK_DIR"]) / "unitree_unifolm_policy_provider_runtime_bundle.zip"
with urllib.request.urlopen(os.environ["BUNDLE_URL"], timeout=300) as response:
    target.write_bytes(response.read())
print("BLUEPRINT_RUNPOD_UNITREE_UNIFOLM_BUNDLE_DOWNLOADED:%d" % target.stat().st_size)
PY
rm -rf "$WORK_DIR/unitree_unifolm_provider_bundle" "$WORK_DIR/runtime_output" "$WORK_DIR/unitree_unifolm_policy_provider_runtime_output.zip"
python -m zipfile -e "$WORK_DIR/unitree_unifolm_policy_provider_runtime_bundle.zip" "$WORK_DIR/unitree_unifolm_provider_bundle"
echo BLUEPRINT_RUNPOD_UNITREE_UNIFOLM_ENTRYPOINT_STARTED
export PYTHONPATH="$WORK_DIR/unitree_unifolm_provider_bundle/provider_runtime:${PYTHONPATH:-}"
export BLUEPRINT_UNITREE_UNIFOLM_PROVIDER_OUTPUT_DIR="$WORK_DIR/runtime_output"
export BLUEPRINT_UNITREE_UNIFOLM_PROVIDER_OUTPUT="$WORK_DIR/runtime_output/unitree_unifolm_policy_provider_output.json"
export BLUEPRINT_UNITREE_UNIFOLM_POLICY_INPUT="$WORK_DIR/unitree_unifolm_provider_bundle/provider_runtime/policy_input.json"
bash "$WORK_DIR/unitree_unifolm_provider_bundle/provider_runtime/run_unitree_unifolm_provider_runtime.sh" || true
python - <<'PY'
import json
import os
import zipfile
from pathlib import Path
output_dir = Path(os.environ["BLUEPRINT_UNITREE_UNIFOLM_PROVIDER_OUTPUT_DIR"])
zip_path = Path(os.environ["WORK_DIR"]) / "unitree_unifolm_policy_provider_runtime_output.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    if output_dir.is_dir():
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir).as_posix())
    else:
        archive.writestr(
            "unitree_unifolm_policy_provider_output.json",
            json.dumps({"status": "blocked", "blockers": ["runtime_output_directory_missing"]}, indent=2),
        )
print("BLUEPRINT_RUNPOD_UNITREE_UNIFOLM_OUTPUT_ZIP_WRITTEN:%d" % zip_path.stat().st_size)
PY
python - <<'PY'
import os
import urllib.request
from pathlib import Path
zip_path = Path(os.environ["WORK_DIR"]) / "unitree_unifolm_policy_provider_runtime_output.zip"
request = urllib.request.Request(
    os.environ["OUTPUT_PUT_URL"],
    data=zip_path.read_bytes(),
    method="PUT",
    headers={"Content-Type": "application/zip"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    response.read()
print("BLUEPRINT_RUNPOD_UNITREE_UNIFOLM_OUTPUT_UPLOAD_OK")
PY
echo BLUEPRINT_RUNPOD_UNITREE_UNIFOLM_PROVIDER_COMPLETED_OR_BLOCKED
"""
    if provider_bundle_kind != "wam":
        raise ValueError(f"unsupported_provider_bundle_kind:{provider_bundle_kind}")
    return r"""
set -euo pipefail
echo BLUEPRINT_RUNPOD_WAM_PROVIDER_STARTED
WORK_DIR="${BLUEPRINT_RUNPOD_WAM_WORK_DIR:-/workspace/blueprint_wam_provider}"
BUNDLE_URL="${BLUEPRINT_EVAL_MANIFEST_URI:-}"
OUTPUT_PUT_URL="${BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL:-}"
export WORK_DIR BUNDLE_URL OUTPUT_PUT_URL
export BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR="${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-$WORK_DIR/runtime_output}"
upload_wam_outer_blocker() {
  set +e
  mkdir -p "$BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR"
  OUTER_RC="${1:-1}" python - <<'PY'
import json
import os
import urllib.request
import zipfile
from pathlib import Path

output_dir = Path(os.environ["BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)
rc = int(os.environ.get("OUTER_RC", "1") or 1)
blockers = ["runpod_wam_outer_bootstrap_failed_before_runtime_result"]
(output_dir / "wam_provider_output.json").write_text(
    json.dumps(
        {
            "schema_version": "wam_provider_output.v1",
            "status": "blocked",
            "blockers": blockers,
            "outer_returncode": rc,
            "raw_credentials_written_to_artifacts": False,
        },
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
carrier = os.environ.get("BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC", "").lower()
if carrier in {"1", "true", "yes", "on"}:
    (output_dir / "unitree_groot_n17_sonic_policy_provider_output.json").write_text(
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
                "blockers": blockers,
                "outer_returncode": rc,
                "raw_credentials_written_to_artifacts": False,
                "secret_hashes_written_to_artifacts": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
zip_path = Path(os.environ["WORK_DIR"]) / "wam_provider_runtime_output.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(output_dir).as_posix())
put_url = os.environ.get("OUTPUT_PUT_URL", "")
if put_url:
    request = urllib.request.Request(
        put_url,
        data=zip_path.read_bytes(),
        method="PUT",
        headers={"Content-Type": "application/zip"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read()
PY
}
entrypoint_heartbeat_pid=""
trap 'rc=$?; if [ -n "${entrypoint_heartbeat_pid:-}" ]; then kill "$entrypoint_heartbeat_pid" 2>/dev/null || true; fi; if [ "$rc" -ne 0 ]; then upload_wam_outer_blocker "$rc"; fi; exit "$rc"' EXIT
upload_wam_running_heartbeat() {
  set +e
  phase="${1:-runpod_wam_outer_wrapper_running}"
  mkdir -p "$BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR"
  PHASE="$phase" python - <<'PY'
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

output_dir = Path(os.environ["BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)
phase = os.environ.get("PHASE", "runpod_wam_outer_wrapper_running")
entrypoint_log_tail = None
entrypoint_log = os.environ.get("BLUEPRINT_RUNPOD_WAM_ENTRYPOINT_LOG_PATH", "").strip()
if entrypoint_log:
    log_path = Path(entrypoint_log)
    if log_path.is_file():
        entrypoint_log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
payload = {
    "schema_version": "wam_provider_output.v1",
    "status": "running",
    "runtime_phase": phase,
    "runtime_phase_details": {
        "phase": phase,
        "observed_at_epoch": round(time.time(), 3),
        "source": "runpod_wam_outer_wrapper",
        "entrypoint_log_tail": entrypoint_log_tail,
        "raw_secret_values_recorded": False,
    },
    "blockers": [],
    "raw_credentials_written_to_artifacts": False,
}
(output_dir / "wam_provider_output.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
carrier = os.environ.get("BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC", "").lower()
if carrier in {"1", "true", "yes", "on"}:
    unitree_payload = {
        "schema_version": "unitree_groot_n17_sonic_wam_persistent_session_output.v1",
        "status": "running",
        "policy_id": "unitree_groot_n17_sonic_policy",
        "persistent_provider_session_used": True,
        "runtime_phase": phase,
        "runtime_phase_details": payload["runtime_phase_details"],
        "blockers": [],
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    (output_dir / "unitree_groot_n17_sonic_policy_provider_output.json").write_text(
        json.dumps(unitree_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
zip_path = Path(os.environ["WORK_DIR"]) / "wam_provider_runtime_output.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(output_dir).as_posix())
put_url = os.environ.get("OUTPUT_PUT_URL", "")
if put_url:
    request = urllib.request.Request(
        put_url,
        data=zip_path.read_bytes(),
        method="PUT",
        headers={"Content-Type": "application/zip"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read()
PY
  set -e
}
if [ -z "$BUNDLE_URL" ]; then echo BLUEPRINT_RUNPOD_WAM_BLOCKED:bundle_url_missing; exit 20; fi
if [ -z "$OUTPUT_PUT_URL" ]; then echo BLUEPRINT_RUNPOD_WAM_BLOCKED:output_put_url_missing; exit 21; fi
mkdir -p "$WORK_DIR"
upload_wam_running_heartbeat runpod_wam_outer_wrapper_started
if command -v apt-get >/dev/null 2>&1; then
  if ! command -v git >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
    upload_wam_running_heartbeat runpod_wam_system_dependency_install_started
    timeout 300 apt-get update >/tmp/blueprint_runpod_apt_update.log 2>&1 || true
    DEBIAN_FRONTEND=noninteractive timeout 600 apt-get install -y git ffmpeg curl ca-certificates >/tmp/blueprint_runpod_apt_install.log 2>&1 || true
    upload_wam_running_heartbeat runpod_wam_system_dependency_install_completed
  else
    upload_wam_running_heartbeat runpod_wam_system_dependencies_present
  fi
fi
python - <<'PY'
import os
import urllib.request
from pathlib import Path
target = Path(os.environ["WORK_DIR"]) / "wam_provider_runtime_bundle.zip"
with urllib.request.urlopen(os.environ["BUNDLE_URL"], timeout=300) as response:
    target.write_bytes(response.read())
print("BLUEPRINT_RUNPOD_WAM_BUNDLE_DOWNLOADED:%d" % target.stat().st_size)
PY
upload_wam_running_heartbeat runpod_wam_bundle_downloaded
rm -rf "$WORK_DIR/wam_provider_bundle" "$WORK_DIR/runtime_output" "$WORK_DIR/wam_provider_runtime_output.zip"
python -m zipfile -e "$WORK_DIR/wam_provider_runtime_bundle.zip" "$WORK_DIR/wam_provider_bundle"
echo BLUEPRINT_RUNPOD_WAM_ENTRYPOINT_STARTED
upload_wam_running_heartbeat runpod_wam_entrypoint_starting
export BLUEPRINT_WAM_PROVIDER_BUNDLE_DIR="$WORK_DIR/wam_provider_bundle"
export BLUEPRINT_WAM_ROLLOUT_INPUT="$WORK_DIR/wam_provider_bundle/provider_runtime/wam_rollout_input_manifest.json"
mkdir -p "$BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR"
WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS="${BLUEPRINT_RUNPOD_WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS:-}"
if [ -z "$WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS" ] && [ -n "${BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_WRAPPER_WATCHDOG_SECONDS:-}" ]; then
  WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS=$((BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_WRAPPER_WATCHDOG_SECONDS + 300))
fi
WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS="${WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS:-7200}"
export WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS
runtime_log="$BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR/runpod_wam_provider_entrypoint.log"
export BLUEPRINT_RUNPOD_WAM_ENTRYPOINT_LOG_PATH="$runtime_log"
echo "BLUEPRINT_RUNPOD_WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS=$WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS" > "$runtime_log"
(
  while true; do
    upload_wam_running_heartbeat runpod_wam_entrypoint_running
    sleep "${BLUEPRINT_RUNPOD_WAM_ENTRYPOINT_HEARTBEAT_SECONDS:-60}"
  done
) &
entrypoint_heartbeat_pid=$!
set +e
if command -v timeout >/dev/null 2>&1; then
  timeout "$WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS" bash "$WORK_DIR/wam_provider_bundle/provider_runtime/run_wam_provider_runtime.sh" >> "$runtime_log" 2>&1
else
  bash "$WORK_DIR/wam_provider_bundle/provider_runtime/run_wam_provider_runtime.sh" >> "$runtime_log" 2>&1
fi
wam_runtime_rc=$?
export wam_runtime_rc
set -e
kill "$entrypoint_heartbeat_pid" 2>/dev/null || true
entrypoint_heartbeat_pid=""
entrypoint_status="completed"
entrypoint_timed_out=false
entrypoint_blockers='[]'
if [ "$wam_runtime_rc" -ne 0 ]; then
  entrypoint_status="blocked"
  entrypoint_blockers='["runpod_wam_provider_entrypoint_nonzero_or_timeout"]'
  if [ "$wam_runtime_rc" = "124" ] || [ "$wam_runtime_rc" = "137" ]; then
    entrypoint_timed_out=true
    entrypoint_blockers='["runpod_wam_provider_entrypoint_nonzero_or_timeout","runpod_wam_provider_entrypoint_timeout"]'
  fi
fi
cat > "$BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR/runpod_wam_provider_entrypoint_execution.json" <<EOF
{"schema_version":"runpod_wam_provider_entrypoint_execution.v1","status":"$entrypoint_status","returncode":$wam_runtime_rc,"timed_out":$entrypoint_timed_out,"timeout_seconds":$WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS,"runtime_stdout_stderr_log_path":"$BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR/runpod_wam_provider_entrypoint.log","raw_secret_values_recorded":false}
EOF
if [ "$wam_runtime_rc" -ne 0 ]; then
  generic_output="$BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR/wam_provider_output.json"
  if [ ! -f "$generic_output" ]; then
    cat > "$generic_output" <<EOF
{"schema_version":"wam_provider_output.v1","status":"blocked","blockers":$entrypoint_blockers,"runpod_wam_provider_entrypoint_returncode":$wam_runtime_rc,"raw_credentials_written_to_artifacts":false}
EOF
  fi
  carrier_flag="$(printf '%s' "${BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC:-}" | tr '[:upper:]' '[:lower:]')"
  unitree_output="${BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT:-$BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR/unitree_groot_n17_sonic_policy_provider_output.json}"
  if [ "$carrier_flag" = "true" ] || [ "$carrier_flag" = "1" ] || [ "$carrier_flag" = "yes" ] || [ "$carrier_flag" = "on" ]; then
    if [ ! -f "$unitree_output" ]; then
      cat > "$unitree_output" <<EOF
{"schema_version":"unitree_groot_n17_sonic_wam_persistent_session_output.v1","status":"blocked","policy_id":"unitree_groot_n17_sonic_policy","persistent_provider_session_used":true,"unitree_groot_n17_sonic_model_executed":false,"unitree_groot_n17_sonic_policy_action_command_ran":false,"policy_action_model_command_ran":false,"provider_output_replay_used":false,"blockers":$entrypoint_blockers,"runpod_wam_provider_entrypoint_returncode":$wam_runtime_rc,"raw_credentials_written_to_artifacts":false,"secret_hashes_written_to_artifacts":false}
EOF
    fi
  fi
fi
python - <<'PY'
import json
import os
import zipfile
from pathlib import Path
output_dir = Path(os.environ["BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR"])
zip_path = Path(os.environ["WORK_DIR"]) / "wam_provider_runtime_output.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    if output_dir.is_dir():
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir).as_posix())
    else:
        archive.writestr(
            "runtime_output_missing.json",
            json.dumps({"status": "blocked", "blockers": ["runtime_output_directory_missing"]}, indent=2),
        )
print("BLUEPRINT_RUNPOD_WAM_OUTPUT_ZIP_WRITTEN:%d" % zip_path.stat().st_size)
PY
python - <<'PY'
import os
import urllib.request
from pathlib import Path
zip_path = Path(os.environ["WORK_DIR"]) / "wam_provider_runtime_output.zip"
request = urllib.request.Request(
    os.environ["OUTPUT_PUT_URL"],
    data=zip_path.read_bytes(),
    method="PUT",
    headers={"Content-Type": "application/zip"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    response.read()
print("BLUEPRINT_RUNPOD_WAM_OUTPUT_UPLOAD_OK")
PY
echo BLUEPRINT_RUNPOD_WAM_PROVIDER_COMPLETED_OR_BLOCKED
"""


def _pod_payload(
    *,
    job_name: str,
    image_name: str,
    gpu_type_ids: Sequence[str],
    provider_bundle_url: str,
    provider_output_put_url: str,
    provider_bundle_kind: str,
    model_secret_env: Mapping[str, str],
    provider_runtime_config_env: Mapping[str, str],
    container_disk_gb: int,
    volume_gb: int,
    cloud_type: str = "SECURE",
    allowed_cuda_versions: Sequence[str] = (),
    min_vcpu_per_gpu: int = 2,
    min_ram_per_gpu: int = 8,
    carrier_volume_admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _build_pod_payload(
        job_name=job_name,
        image_name=image_name,
        gpu_type_ids=gpu_type_ids,
        provider_bundle_url=provider_bundle_url,
        provider_output_put_url=provider_output_put_url,
        provider_bundle_kind=provider_bundle_kind,
        model_secret_env=model_secret_env,
        provider_runtime_config_env=provider_runtime_config_env,
        container_disk_gb=container_disk_gb,
        volume_gb=volume_gb,
        cloud_type=cloud_type,
        allowed_cuda_versions=allowed_cuda_versions,
        min_vcpu_per_gpu=min_vcpu_per_gpu,
        min_ram_per_gpu=min_ram_per_gpu,
        provider_script=_provider_shell_script(provider_bundle_kind),
        keep_on_success=_teardown_action() == "keep_on_success",
        carrier_volume_admission=carrier_volume_admission,
    )


def _warm_candidate_path() -> Path:
    return Path(
        _string(os.getenv(RUNPOD_WAM_WARM_CANDIDATE_FILE_ENV))
        or DEFAULT_RUNPOD_WAM_WARM_CANDIDATE_FILE
    ).expanduser()


def _read_compatible_warm_candidate(
    *,
    provider_bundle_kind: str,
    image_name: str,
    cloud_type: str,
) -> dict[str, Any]:
    candidate_path = _warm_candidate_path()
    return _read_compatible_warm_candidate_contract(
        candidate_path=candidate_path,
        disabled=_env_truthy(RUNPOD_WAM_DISABLE_WARM_CANDIDATE_ENV),
        disable_env=RUNPOD_WAM_DISABLE_WARM_CANDIDATE_ENV,
        provider_bundle_kind=provider_bundle_kind,
        image_name=image_name,
        cloud_type=cloud_type,
    )


def _retire_warm_candidate(
    *,
    warm_candidate: Mapping[str, Any],
    reason: str,
    generated_at: str,
) -> dict[str, Any]:
    candidate_path = _warm_candidate_path()
    candidate_pod_id = _string(warm_candidate.get("pod_id"))
    try:
        existing = _read_json(candidate_path)
    except (OSError, ValueError) as exc:
        return {
            "status": "not_retired",
            "path": str(candidate_path),
            "reason": "warm_candidate_state_unreadable",
            "error_type": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
    if _string(existing.get("pod_id")) != candidate_pod_id:
        return {
            "status": "not_retired",
            "path": str(candidate_path),
            "reason": "warm_candidate_changed",
            "candidate_pod_id": candidate_pod_id,
            "current_pod_id": _string(existing.get("pod_id")),
            "raw_secret_values_recorded": False,
        }
    retired = {
        "schema_version": RUNPOD_WAM_WARM_CANDIDATE_SCHEMA_VERSION,
        "status": "retired",
        "generated_at": generated_at,
        "retired_pod_id": candidate_pod_id,
        "reason": reason,
        "previous_source_job_dir": existing.get("source_job_dir"),
        "raw_secret_values_recorded": False,
    }
    try:
        write_json(candidate_path, retired)
    except OSError as exc:
        return {
            "status": "not_retired",
            "path": str(candidate_path),
            "reason": "warm_candidate_retire_write_failed",
            "error_type": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
    return {
        "status": "retired",
        "path": str(candidate_path),
        "retired_pod_id": candidate_pod_id,
        "reason": reason,
        "raw_secret_values_recorded": False,
    }


def _write_stopped_warm_candidate(
    *,
    job_dir: Path,
    pod_id: str,
    generated_at: str,
) -> dict[str, Any]:
    candidate_path = _warm_candidate_path()
    try:
        state = _read_json(_state_path(job_dir))
    except (OSError, ValueError) as exc:
        return {
            "status": "blocked",
            "path": str(candidate_path),
            "blockers": ["runpod_warm_candidate_state_unreadable"],
            "error_type": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
    candidate = {
        "schema_version": RUNPOD_WAM_WARM_CANDIDATE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "pod_id": pod_id,
        "provider_bundle_kind": _string(state.get("provider_bundle_kind")) or "wam",
        "image_name": _string(state.get("image_name")),
        "cloud_type": _string(state.get("cloud_type")) or "SECURE",
        "gpu_type_ids": list(state.get("gpu_type_ids") or []),
        "container_disk_gb": state.get("container_disk_gb"),
        "volume_gb": state.get("volume_gb"),
        "source_job_dir": str(job_dir),
        "source_stop_manifest_path": str(job_dir / "runpod_wam_async_stop_manifest.json"),
        "stopped_pod_preserved_for_warm_reuse": True,
        "raw_secret_values_recorded": False,
    }
    try:
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(candidate_path, candidate)
    except OSError as exc:
        return {
            "status": "blocked",
            "path": str(candidate_path),
            "blockers": ["runpod_warm_candidate_write_failed"],
            "error_type": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
    return {
        "status": "recorded",
        "path": str(candidate_path),
        "pod_id": pod_id,
        "provider_bundle_kind": candidate["provider_bundle_kind"],
        "image_name": candidate["image_name"],
        "cloud_type": candidate["cloud_type"],
        "raw_secret_values_recorded": False,
    }


def _write_running_warm_candidate(
    *,
    job_dir: Path,
    pod_id: str,
    generated_at: str,
) -> dict[str, Any]:
    candidate_path = _warm_candidate_path()
    try:
        state = _read_json(_state_path(job_dir))
    except (OSError, ValueError) as exc:
        return {
            "status": "blocked",
            "path": str(candidate_path),
            "blockers": ["runpod_warm_candidate_state_unreadable"],
            "error_type": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
    candidate = {
        "schema_version": RUNPOD_WAM_WARM_CANDIDATE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "pod_id": pod_id,
        "provider_bundle_kind": _string(state.get("provider_bundle_kind")) or "wam",
        "image_name": _string(state.get("image_name")),
        "cloud_type": _string(state.get("cloud_type")) or "SECURE",
        "gpu_type_ids": list(state.get("gpu_type_ids") or []),
        "container_disk_gb": state.get("container_disk_gb"),
        "volume_gb": state.get("volume_gb"),
        "source_job_dir": str(job_dir),
        "source_keepalive_poll_manifest_path": str(
            job_dir / "runpod_wam_async_poll_manifest.json"
        ),
        "running_pod_preserved_for_hot_reuse": True,
        "raw_secret_values_recorded": False,
    }
    try:
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(candidate_path, candidate)
    except OSError as exc:
        return {
            "status": "blocked",
            "path": str(candidate_path),
            "blockers": ["runpod_warm_candidate_write_failed"],
            "error_type": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
    return {
        "status": "recorded",
        "path": str(candidate_path),
        "pod_id": pod_id,
        "provider_bundle_kind": candidate["provider_bundle_kind"],
        "image_name": candidate["image_name"],
        "cloud_type": candidate["cloud_type"],
        "running_pod_preserved_for_hot_reuse": True,
        "raw_secret_values_recorded": False,
    }


def _existing_pod_update_payload(pod_payload: Mapping[str, Any]) -> dict[str, Any]:
    update_keys = {
        "containerDiskInGb",
        "dockerEntrypoint",
        "dockerStartCmd",
        "env",
        "imageName",
        "name",
        "ports",
        "volumeInGb",
        "volumeMountPath",
    }
    return {key: pod_payload[key] for key in update_keys if key in pod_payload}


def _pod_status(payload: Mapping[str, Any]) -> str:
    pod = _mapping(payload.get("pod")) or _mapping(payload.get("data")) or dict(payload)
    return (
        _string(pod.get("desiredStatus"))
        or _string(pod.get("runtimeStatus"))
        or _string(pod.get("status"))
        or _string(pod.get("machineStatus"))
        or "unknown"
    )


def _pod_runtime_present(payload: Mapping[str, Any]) -> bool:
    pod = _mapping(payload.get("pod")) or _mapping(payload.get("data")) or dict(payload)
    return bool(pod.get("runtime"))


def _pod_public_ip_present(payload: Mapping[str, Any]) -> bool:
    pod = _mapping(payload.get("pod")) or _mapping(payload.get("data")) or dict(payload)
    return bool(_string(pod.get("publicIp")))


def _iso_epoch_seconds(value: Any) -> float | None:
    text = _string(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _candidate_age_seconds(candidate: Mapping[str, Any], generated_at: str) -> float | None:
    recorded_epoch = _iso_epoch_seconds(candidate.get("recorded_at"))
    generated_epoch = _iso_epoch_seconds(generated_at)
    if recorded_epoch is None or generated_epoch is None:
        return None
    return max(0.0, generated_epoch - recorded_epoch)


def _validate_running_warm_candidate_runtime(
    candidate: Mapping[str, Any],
    *,
    api_key: str,
    generated_at: str,
) -> dict[str, Any]:
    """Reject stale running-hot candidates that no longer have a RunPod runtime."""
    selected = dict(candidate)
    if selected.get("status") != "selected" or not selected.get(
        "running_pod_preserved_for_hot_reuse"
    ):
        return selected
    pod_id = _string(selected.get("pod_id"))
    if not pod_id:
        return selected
    try:
        status_code, pod_payload = _runpod_request(
            method="GET",
            path=f"/pods/{pod_id}",
            api_key=api_key,
            timeout_seconds=20,
        )
    except urllib.error.HTTPError as exc:
        selected.update(
            {
                "status": "rejected",
                "reason": "running_warm_candidate_status_http_error",
                "pod_status_http_status_code": exc.code,
                "raw_secret_values_recorded": False,
            }
        )
        return selected
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        selected.update(
            {
                "status": "rejected",
                "reason": "running_warm_candidate_status_probe_failed",
                "probe_error_type": type(exc).__name__,
                "raw_secret_values_recorded": False,
            }
        )
        return selected
    pod_status = _pod_status(pod_payload)
    runtime_present = _pod_runtime_present(pod_payload)
    public_ip_present = _pod_public_ip_present(pod_payload)
    age_seconds = _candidate_age_seconds(selected, generated_at)
    max_runtime_absent_seconds = _env_int(
        RUNPOD_WAM_RUNNING_CANDIDATE_RUNTIME_ABSENT_MAX_SECONDS_ENV,
        480,
    )
    active_without_runtime = bool(
        not runtime_present
        and (
            pod_status in RUNPOD_ACTIVE_POD_STATUSES
            or pod_status.upper() in RUNPOD_ACTIVE_POD_STATUSES
        )
    )
    stale_runtime_absent = bool(
        active_without_runtime
        and age_seconds is not None
        and age_seconds >= max_runtime_absent_seconds
    )
    selected["reuse_probe"] = {
        "status": "stale_runtime_absent_rejected"
        if stale_runtime_absent
        else "passed",
        "pod_status_http_status_code": status_code,
        "pod_status": pod_status,
        "runtime_present": runtime_present,
        "public_ip_present": public_ip_present,
        "candidate_age_seconds": round(age_seconds, 6)
        if age_seconds is not None
        else None,
        "max_runtime_absent_seconds": max_runtime_absent_seconds,
        "raw_secret_values_recorded": False,
    }
    if stale_runtime_absent:
        selected.update(
            {
                "status": "rejected",
                "reason": "running_warm_candidate_runtime_absent_too_long",
                "raw_secret_values_recorded": False,
            }
        )
    return selected


def _staging_urls(public_base_url: str, token_file: Path) -> tuple[str, str, dict[str, Any]]:
    token, token_status = _read_or_create_token(token_file)
    return (
        _url_with_token(public_base_url, BUNDLE_ROUTE, token),
        _url_with_token(public_base_url, OUTPUT_ROUTE, token),
        token_status,
    )


def create_runpod_wam_async_run(
    *,
    job_dir: str | Path,
    bundle_path: str | Path,
    public_base_url: str = "",
    provider_bundle_url: str = "",
    provider_output_put_url: str = "",
    provider_output_get_url: str = "",
    provider_bundle_url_file: str | Path | None = None,
    provider_output_put_url_file: str | Path | None = None,
    provider_output_get_url_file: str | Path | None = None,
    token_file: str | Path | None = None,
    secret_env_file: str | Path | None = None,
    output_path: str | Path | None = None,
    max_spend_usd: float | None = None,
    allow_paid_runpod_launch: bool = False,
    skip_public_staging_verification: bool = False,
    verify_output_put_url: bool = False,
    gpu_type_ids: Sequence[str] = DEFAULT_GPU_TYPE_IDS,
    image_name: str = DEFAULT_WAM_PUBLIC_IMAGE,
    provider_bundle_kind: str = "wam",
    container_disk_gb: int = 80,
    volume_gb: int = 20,
    cloud_type: str = "SECURE",
    allowed_cuda_versions: Sequence[str] = (),
    min_vcpu_per_gpu: int = 2,
    min_ram_per_gpu: int = 8,
    existing_pod_id: str = "",
    generated_at: str | None = None,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None = None,
    carrier_volume_admission: Mapping[str, Any] | None = None,
    pod_name: str = "",
    provider_lane_handoff_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    if provider_bundle_kind not in RUNPOD_PROVIDER_BUNDLE_KINDS:
        raise ValueError(f"unsupported_provider_bundle_kind:{provider_bundle_kind}")
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    resolved_bundle = Path(bundle_path).expanduser().resolve()
    resolved_output = (
        Path(output_path).expanduser().resolve()
        if output_path
        else resolved_job_dir / DEFAULT_OUTPUT_FILENAME
    )
    resolved_token_file = (
        Path(token_file).expanduser().resolve()
        if token_file
        else Path(DEFAULT_TOKEN_FILE).expanduser().resolve()
    )
    resolved_secret_env_file = (
        Path(secret_env_file).expanduser().resolve()
        if secret_env_file
        else Path(DEFAULT_SECRET_ENV_FILE).expanduser().resolve()
    )
    ensure_dir(resolved_job_dir)
    # A fresh run must not inherit a prior run's output zip: the poll treats a pre-existing
    # output file as this run's result and short-circuits before the worker uploads anything,
    # so clear any stale terminal/nonterminal output left over from an earlier run in this dir.
    for _stale_output in (
        resolved_output,
        resolved_output.with_name(f"{resolved_output.stem}_nonterminal{resolved_output.suffix}"),
    ):
        _stale_output.unlink(missing_ok=True)
    full_loop_guard = _unitree_groot_sonic_full_loop_create_guard(
        bundle_path=resolved_bundle,
        provider_bundle_kind=provider_bundle_kind,
    )
    if full_loop_guard.get("status") == "blocked":
        manifest = {
            "schema_version": RUNPOD_WAM_CREATE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "provider_bundle_kind": provider_bundle_kind,
            "bundle_path": str(resolved_bundle),
            "blockers": list(full_loop_guard.get("blockers") or []),
            "full_loop_guard": full_loop_guard,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "runpod_wam_async_create_manifest.json", manifest)
        return manifest
    bundle_url_from_file, bundle_url_file_meta = _read_sensitive_url_file(
        str(provider_bundle_url_file or ""),
        label="provider_bundle_url_file",
    )
    output_url_from_file, output_url_file_meta = _read_sensitive_url_file(
        str(provider_output_put_url_file or ""),
        label="provider_output_put_url_file",
    )
    output_get_url_from_file, output_get_url_file_meta = _read_sensitive_url_file(
        str(provider_output_get_url_file or ""),
        label="provider_output_get_url_file",
    )
    if not _string(provider_bundle_url) and bundle_url_from_file:
        provider_bundle_url = bundle_url_from_file
    if not _string(provider_output_put_url) and output_url_from_file:
        provider_output_put_url = output_url_from_file
    if not _string(provider_output_get_url) and output_get_url_from_file:
        provider_output_get_url = output_get_url_from_file
    direct_provider_urls = bool(provider_bundle_url and provider_output_put_url)
    token_status: dict[str, Any] = {
        "present": False,
        "path_redacted": True,
        "path_disclosure_policy": SECRET_PATH_DISCLOSURE_POLICY,
        "token_recorded_in_manifest": False,
        "reason": "not_required_for_explicit_provider_urls"
        if direct_provider_urls
        else "pending_staging_token_resolution",
    }
    if direct_provider_urls:
        provider_bundle_url = _string(provider_bundle_url)
        provider_output_put_url = _string(provider_output_put_url)
        staging_manifest = {
            "schema_version": "runpod_wam_direct_provider_urls.v1",
            "generated_at": generated,
            "status": "ready",
            "job_dir": str(resolved_job_dir),
            "bundle_path": str(resolved_bundle),
            "output_path": str(resolved_output),
            "provider_bundle_url_redacted": _redact_provider_url(provider_bundle_url),
            "provider_output_put_url_redacted": _redact_provider_url(provider_output_put_url),
            "provider_output_get_url_redacted": _redact_provider_url(provider_output_get_url),
            "provider_bundle_url_file": bundle_url_file_meta,
            "provider_output_put_url_file": output_url_file_meta,
            "provider_output_get_url_file": output_get_url_file_meta,
            "explicit_provider_urls_used": True,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "runpod_wam_direct_provider_urls_manifest.json", staging_manifest)
        self_test = {
            "schema_version": "runpod_wam_local_staging_self_test.v1",
            "generated_at": generated,
            "status": "skipped",
            "reason": "explicit_provider_urls_supplied",
            "raw_secret_values_recorded": False,
        }
    else:
        provider_bundle_url, provider_output_put_url, token_status = _staging_urls(
            public_base_url,
            resolved_token_file,
        )
        staging_manifest = prepare_vast_bundle_staging(
            job_dir=resolved_job_dir,
            bundle_path=resolved_bundle,
            public_base_url=public_base_url,
            token_file=resolved_token_file,
            secret_env_file=resolved_secret_env_file,
            output_path=resolved_output,
            generated_at=generated,
        )
        self_test = run_local_staging_self_test(
            job_dir=resolved_job_dir,
            bundle_path=resolved_bundle,
            output_path=resolved_job_dir / "runpod_wam_staging_self_test_output.zip",
            token_file=resolved_token_file,
            generated_at=generated,
        )
    if skip_public_staging_verification:
        public_verification = {
            "schema_version": "vast_public_staging_verification.v1",
            "generated_at": generated,
            "completed_at": generated,
            "status": "skipped",
            "job_dir": str(resolved_job_dir),
            "reason": "skip_public_staging_verification_requested",
            "provider_bundle_url_redacted": _redact_provider_url(provider_bundle_url),
            "provider_output_put_url_redacted": _redact_provider_url(provider_output_put_url),
            "provider_output_get_url_redacted": _redact_provider_url(provider_output_get_url),
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_public_staging_verification.json", public_verification)
    else:
        public_verification = verify_public_staging_urls(
            job_dir=resolved_job_dir,
            provider_bundle_url=provider_bundle_url,
            provider_output_put_url=provider_output_put_url,
            bundle_path=resolved_bundle,
            output_path=resolved_output,
            max_wait_seconds=30,
            retry_interval_seconds=2,
            timeout_seconds=10,
            required_consecutive_successes=1 if direct_provider_urls else 2,
            allow_output_put_probe=verify_output_put_url or not direct_provider_urls,
            cleanup_output_probe=not direct_provider_urls,
            generated_at=generated,
        )
    api_key, api_key_meta = _read_runpod_api_key()
    model_secret_env, model_secret_env_status = _read_model_secret_env()
    provider_runtime_config_env, provider_runtime_config_env_status = (
        _read_provider_runtime_config_env(provider_bundle_kind)
    )
    prelaunch_spend_guard = _runpod_wam_prelaunch_spend_guard(
        max_spend_usd=max_spend_usd,
        allow_paid_runpod_launch=allow_paid_runpod_launch,
        gpu_type_ids=gpu_type_ids,
        container_disk_gb=container_disk_gb,
        volume_gb=volume_gb,
    )
    post_marker_timeout = _runpod_wam_post_marker_timeout_seconds()
    try:
        pre_spend_preflight = require_pre_spend_preflight(
            lane=RUNPOD_WAM_LANE,
            provider="runpod",
            credential_present=bool(api_key),
            capacity_evidence={
                "available": bool(api_key and gpu_type_ids),
                "detail": f"runpod_on_demand_pool_configured:{len(list(gpu_type_ids))}_gpu_types",
            },
            image_contract=image_contract_from_ref(image_name),
            runtime_contract={
                "startup_marker": "nonterminal_output_heartbeat",
                "progress_marker": "provider_output_zip",
                "startup_timeout_seconds": post_marker_timeout,
                "no_progress_timeout_seconds": post_marker_timeout,
            },
            spend_gate_open=prelaunch_spend_guard.get("can_launch") is True,
            record_dir=resolved_job_dir,
        )
    except PreSpendPreflightBlocked as blocked_preflight:
        pre_spend_preflight = blocked_preflight.preflight
    blockers: list[str] = []
    if pre_spend_preflight.get("status") != "PASS":
        blockers.append("runpod_wam_pre_spend_preflight_not_passed")
        blockers.extend(pre_spend_preflight.get("blockers") or [])
    if staging_manifest.get("status") != "ready":
        blockers.extend(staging_manifest.get("blockers") or ["runpod_wam_staging_not_ready"])
    if not direct_provider_urls and self_test.get("status") != "passed":
        blockers.append("runpod_wam_local_staging_self_test_failed")
    if public_verification.get("status") not in {"passed", "skipped"}:
        blockers.extend(public_verification.get("blockers") or ["runpod_wam_public_staging_not_verified"])
    if not direct_provider_urls and not _string(public_base_url).startswith("https://"):
        blockers.append("runpod_wam_public_base_url_must_be_https")
    if direct_provider_urls:
        bundle_scheme = urlparse(provider_bundle_url).scheme
        output_scheme = urlparse(provider_output_put_url).scheme
        if bundle_scheme not in {"http", "https"}:
            blockers.append("runpod_provider_bundle_url_scheme_not_http")
        if output_scheme not in {"http", "https"}:
            blockers.append("runpod_provider_output_put_url_scheme_not_http")
    elif not _string(public_base_url):
        blockers.append("runpod_public_base_url_or_explicit_provider_urls_required")
    if prelaunch_spend_guard.get("can_launch") is not True:
        blockers.append("runpod_wam_prelaunch_spend_guard_not_passed")
        blockers.extend(prelaunch_spend_guard.get("blockers") or [])
    if not allow_paid_runpod_launch:
        blockers.append("paid_runpod_launch_not_authorized_by_runner_flag")
    if os.getenv(RUNPOD_API_GATE_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
        blockers.append(f"missing_env_{RUNPOD_API_GATE_ENV}")
    if os.getenv(RUNPOD_POD_LAUNCH_GATE_ENV, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        blockers.append(f"missing_env_{RUNPOD_POD_LAUNCH_GATE_ENV}")
    if not api_key:
        blockers.append(f"missing_env_{RUNPOD_API_KEY_ENV}_or_{RUNPOD_API_KEY_FILE_ENV}")
    if not resolved_bundle.is_file():
        blockers.append("runpod_wam_provider_bundle_missing")
    if blockers:
        manifest = {
            "schema_version": RUNPOD_WAM_CREATE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "provider_bundle_kind": provider_bundle_kind,
            "blockers": sorted(set(blockers)),
            "staging_manifest_status": staging_manifest.get("status"),
            "self_test_status": self_test.get("status"),
            "public_staging_verification_status": public_verification.get("status"),
            "explicit_provider_urls_used": direct_provider_urls,
            "provider_bundle_url_redacted": _redact_provider_url(provider_bundle_url),
            "provider_output_put_url_redacted": _redact_provider_url(provider_output_put_url),
            "provider_output_get_url_redacted": _redact_provider_url(provider_output_get_url),
            "provider_bundle_url_file": bundle_url_file_meta,
            "provider_output_put_url_file": output_url_file_meta,
            "provider_output_get_url_file": output_get_url_file_meta,
            "model_secret_env_status": model_secret_env_status,
            "provider_runtime_config_env_status": provider_runtime_config_env_status,
            "prelaunch_spend_guard": prelaunch_spend_guard,
            "pre_spend_preflight": pre_spend_preflight,
            "api_key_status": api_key_meta,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "runpod_wam_async_create_manifest.json", manifest)
        return manifest
    try:
        require_paid_resource_admission_grant(
            paid_resource_admission_grant, resource_class="runpod_wam_async"
        )
    except PaidResourceAdmissionBlocked as exc:
        manifest = {
            "schema_version": RUNPOD_WAM_CREATE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "blockers": [
                "runpod_wam_shared_admission_missing_or_invalid",
                *exc.blockers,
            ],
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "runpod_wam_async_create_manifest.json", manifest)
        return manifest
    try:
        payload = _pod_payload(
            job_name=(
                _string(pod_name)
                or f"blueprint-{provider_bundle_kind.replace('_', '-')}-{int(time.time())}"
            ),
            image_name=image_name,
            gpu_type_ids=gpu_type_ids,
            provider_bundle_url=provider_bundle_url,
            provider_output_put_url=provider_output_put_url,
            provider_bundle_kind=provider_bundle_kind,
            model_secret_env=model_secret_env,
            provider_runtime_config_env=provider_runtime_config_env,
            container_disk_gb=container_disk_gb,
            volume_gb=volume_gb,
            cloud_type=cloud_type,
            allowed_cuda_versions=allowed_cuda_versions,
            min_vcpu_per_gpu=min_vcpu_per_gpu,
            min_ram_per_gpu=min_ram_per_gpu,
            carrier_volume_admission=carrier_volume_admission,
        )
    except ValueError as exc:
        manifest = {
            "schema_version": RUNPOD_WAM_CREATE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "provider_bundle_kind": provider_bundle_kind,
            "blockers": [str(exc)],
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "runpod_wam_async_create_manifest.json", manifest)
        return manifest
    explicit_existing_pod_id = _selected_existing_pod_id(
        existing_pod_id,
        wam_existing_pod_id_env=RUNPOD_WAM_EXISTING_POD_ID_ENV,
        provider_existing_pod_id_env=RUNPOD_EXISTING_POD_ID_ENV,
    )
    if carrier_volume_admission is not None and explicit_existing_pod_id:
        manifest = {
            "schema_version": RUNPOD_WAM_CREATE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "provider_bundle_kind": provider_bundle_kind,
            "blockers": ["carrier_network_volume_requires_fresh_pod_create"],
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "runpod_wam_async_create_manifest.json", manifest)
        return manifest
    warm_candidate = (
        {
            "status": "not_considered",
            "reason": "explicit_existing_pod_id_configured",
            "pod_id": explicit_existing_pod_id,
            "raw_secret_values_recorded": False,
        }
        if explicit_existing_pod_id
        else ({
            "status": "not_considered",
            "reason": "carrier_network_volume_requires_fresh_pod_create",
            "raw_secret_values_recorded": False,
        } if carrier_volume_admission is not None else _read_compatible_warm_candidate(
            provider_bundle_kind=provider_bundle_kind,
            image_name=image_name,
            cloud_type=cloud_type,
        ))
    )
    if not explicit_existing_pod_id:
        warm_candidate = _validate_running_warm_candidate_runtime(
            warm_candidate,
            api_key=api_key,
            generated_at=generated,
        )
    selected_existing_pod_id = explicit_existing_pod_id or (
        _string(warm_candidate.get("pod_id"))
        if warm_candidate.get("status") == "selected"
        else ""
    )
    warm_start_http_error: dict[str, Any] | None = None
    # Teardown obligation goes on disk BEFORE any billable RunPod call: if this
    # process dies between launch and collect, reap_orphans finds the record.
    resolved_pod_name = _string(payload.get("name"))
    pending_teardown = open_pending_teardown(
        provider="runpod",
        lane=RUNPOD_WAM_LANE,
        run_id=resolved_pod_name
        or f"wam-{provider_bundle_kind}-{int(time.time() * 1000)}",
        resource_kind="compute_instance",
        resource_name=resolved_pod_name,
        job_dir=str(resolved_job_dir),
        max_age_seconds=18_600 if carrier_volume_admission is not None else 7_200,
    )
    handoff_receipt_update: dict[str, Any] = {"status": "not_configured"}
    if provider_lane_handoff_receipt_path is not None:
        try:
            handoff_receipt_update = _update_provider_lane_handoff_receipt(
                provider_lane_handoff_receipt_path,
                pod_name=resolved_pod_name,
                pending_teardown_record=pending_teardown["path"],
            )
        except ValueError as exc:
            cancel_pending_teardown(
                pending_teardown["path"],
                reason="provider_lane_handoff_receipt_pre_create_update_failed",
                evidence={"blocker": str(exc)},
            )
            manifest = {
                "schema_version": RUNPOD_WAM_CREATE_SCHEMA_VERSION,
                "generated_at": generated,
                "status": "blocked",
                "job_dir": str(resolved_job_dir),
                "provider_bundle_kind": provider_bundle_kind,
                "blockers": [str(exc)],
                "provider_mutations_performed": 0,
                "raw_secret_values_recorded": False,
            }
            write_json(
                resolved_job_dir / "runpod_wam_async_create_manifest.json", manifest
            )
            return manifest

    def _cancel_unallocated_create(
        *, reason: str, evidence: Mapping[str, Any]
    ) -> dict[str, Any]:
        cancelled = cancel_pending_teardown(
            pending_teardown["path"], reason=reason, evidence=evidence
        )
        if provider_lane_handoff_receipt_path is None:
            return {"status": "not_configured"}
        if cancelled.get("status") != "cancelled_no_allocation":
            return {
                "status": "no_allocation_confirmation_blocked",
                "reason": "pending_teardown_cancellation_not_confirmed",
                "raw_secret_values_recorded": False,
            }
        try:
            return _confirm_provider_lane_handoff_no_allocation(
                provider_lane_handoff_receipt_path,
                pod_name=resolved_pod_name,
                pending_teardown_record=pending_teardown["path"],
            )
        except ValueError as exc:
            return {
                "status": "no_allocation_confirmation_blocked",
                "blocker": str(exc),
                "raw_secret_values_recorded": False,
            }
    try:
        if selected_existing_pod_id:
            update_payload = _existing_pod_update_payload(payload)
            update_status_code, update_response = _runpod_request(
                method="POST",
                path=f"/pods/{selected_existing_pod_id}/update",
                api_key=api_key,
                payload=update_payload,
                timeout_seconds=45,
            )
            status_code, response = _runpod_request(
                method="POST",
                path=f"/pods/{selected_existing_pod_id}/start",
                api_key=api_key,
                payload={},
                timeout_seconds=45,
            )
            pod_id = _extract_pod_id(response) or selected_existing_pod_id
            launch_mode = "existing_pod_start"
            warm_reuse_detail = {
                "requested": True,
                "existing_pod_id": selected_existing_pod_id,
                "selection_source": "explicit_existing_pod_id"
                if explicit_existing_pod_id
                else "dynamic_warm_candidate",
                "candidate_reuse_kind": _string(warm_candidate.get("reuse_kind"))
                or (
                    "explicit_existing_pod_id"
                    if explicit_existing_pod_id
                    else "existing_pod_candidate"
                ),
                "dynamic_warm_candidate": warm_candidate,
                "update_http_status_code": update_status_code,
                "start_http_status_code": status_code,
                "update_response_keys": sorted(update_response.keys()),
                "start_response_keys": sorted(response.keys()),
                "update_payload_keys": sorted(update_payload.keys()),
                "claim_boundary": {
                    "existing_pod_id_reused": True,
                    "existing_pod_update_start_path_used": True,
                    "running_hot_candidate_still_uses_update_start_path": bool(
                        warm_candidate.get("running_pod_preserved_for_hot_reuse")
                    ),
                    "resident_in_pod_job_queue_not_proven": bool(
                        warm_candidate.get("running_pod_preserved_for_hot_reuse")
                    ),
                },
                "raw_secret_values_recorded": False,
            }
        else:
            status_code, response = _runpod_request(
                method="POST",
                path="/pods",
                api_key=api_key,
                payload=payload,
                timeout_seconds=45,
            )
            pod_id = _extract_pod_id(response)
            launch_mode = "fresh_pod_create"
            warm_reuse_detail = {
                "requested": False,
                "existing_pod_id": "",
                "selection_source": "none",
                "dynamic_warm_candidate": warm_candidate,
                "raw_secret_values_recorded": False,
            }
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")[:500]
        stopped_dynamic_warm_candidate = bool(
            selected_existing_pod_id
            and not explicit_existing_pod_id
            and warm_candidate.get("status") == "selected"
            and warm_candidate.get("stopped_pod_preserved_for_warm_reuse")
        )
        if stopped_dynamic_warm_candidate:
            if exc.code in {404, 410}:
                warm_candidate_retirement = _retire_warm_candidate(
                    warm_candidate=warm_candidate,
                    reason=f"stopped_warm_candidate_start_http_{exc.code}",
                    generated_at=generated,
                )
            else:
                warm_candidate_retirement = {
                    "status": "not_retired",
                    "path": str(_warm_candidate_path()),
                    "reason": "stopped_warm_candidate_start_error_may_be_transient",
                    "http_status_code": exc.code,
                    "raw_secret_values_recorded": False,
                }
            warm_start_http_error = {
                "http_status_code": exc.code,
                "runpod_error_preview": "REDACTED_SECRET"
                if api_key in error_body
                else error_body,
                "fallback_reason": "stopped_warm_candidate_start_failed",
                "warm_candidate_retirement": warm_candidate_retirement,
                "raw_secret_values_recorded": False,
            }
            try:
                status_code, response = _runpod_request(
                    method="POST",
                    path="/pods",
                    api_key=api_key,
                    payload=payload,
                    timeout_seconds=45,
                )
                pod_id = _extract_pod_id(response)
                launch_mode = "fresh_pod_create_after_stopped_warm_start_failed"
                warm_reuse_detail = {
                    "requested": True,
                    "existing_pod_id": selected_existing_pod_id,
                    "selection_source": "dynamic_warm_candidate",
                    "candidate_reuse_kind": _string(warm_candidate.get("reuse_kind"))
                    or "stopped_warm_candidate",
                    "dynamic_warm_candidate": warm_candidate,
                    "stopped_warm_candidate_start_failed": True,
                    "stopped_warm_candidate_start_error": warm_start_http_error,
                    "warm_candidate_retirement": warm_candidate_retirement,
                    "fallback_fresh_create_attempted": True,
                    "fresh_create_http_status_code": status_code,
                    "fresh_create_response_keys": sorted(response.keys()),
                    "claim_boundary": {
                        "existing_pod_id_reused": False,
                        "stopped_warm_candidate_does_not_reserve_gpu_capacity": True,
                        "fallback_fresh_create_used_after_start_capacity_failure": True,
                    },
                    "raw_secret_values_recorded": False,
                }
            except urllib.error.HTTPError as fallback_exc:
                fallback_error_body = fallback_exc.read().decode(
                    "utf-8",
                    errors="replace",
                )[:500]
                manifest = {
                    "schema_version": RUNPOD_WAM_CREATE_SCHEMA_VERSION,
                    "generated_at": generated,
                    "status": "blocked",
                    "job_dir": str(resolved_job_dir),
                    "provider_bundle_kind": provider_bundle_kind,
                    "blockers": [
                        "runpod_stopped_warm_candidate_start_http_error",
                        "runpod_create_pod_http_error",
                    ],
                    "http_status_code": fallback_exc.code,
                    "runpod_error_preview": "REDACTED_SECRET"
                    if api_key in fallback_error_body
                    else fallback_error_body,
                    "model_secret_env_status": model_secret_env_status,
                    "provider_runtime_config_env_status": provider_runtime_config_env_status,
                    "pod_launch_mode": "fresh_pod_create_after_stopped_warm_start_failed",
                    "warm_existing_pod": {
                        "requested": True,
                        "existing_pod_id": selected_existing_pod_id,
                        "selection_source": "dynamic_warm_candidate",
                        "dynamic_warm_candidate": warm_candidate,
                        "stopped_warm_candidate_start_error": warm_start_http_error,
                        "fallback_fresh_create_attempted": True,
                        "raw_secret_values_recorded": False,
                    },
                    "raw_secret_values_recorded": False,
                }
                handoff_receipt_update = _cancel_unallocated_create(
                    reason="runpod_create_pod_http_error_no_allocation",
                    evidence={"http_status_code": fallback_exc.code},
                )
                manifest["provider_lane_handoff_receipt_update"] = handoff_receipt_update
                write_json(resolved_job_dir / "runpod_wam_async_create_manifest.json", manifest)
                return manifest
        else:
            manifest = {
                "schema_version": RUNPOD_WAM_CREATE_SCHEMA_VERSION,
                "generated_at": generated,
                "status": "blocked",
                "job_dir": str(resolved_job_dir),
                "provider_bundle_kind": provider_bundle_kind,
                "blockers": ["runpod_create_pod_http_error"],
                "http_status_code": exc.code,
                "runpod_error_preview": "REDACTED_SECRET" if api_key in error_body else error_body,
                "model_secret_env_status": model_secret_env_status,
                "provider_runtime_config_env_status": provider_runtime_config_env_status,
                "pod_launch_mode": "existing_pod_start" if selected_existing_pod_id else "fresh_pod_create",
                "warm_existing_pod": {
                    "requested": bool(selected_existing_pod_id),
                    "existing_pod_id": selected_existing_pod_id,
                    "selection_source": "explicit_existing_pod_id"
                    if explicit_existing_pod_id
                    else "dynamic_warm_candidate"
                    if selected_existing_pod_id
                    else "none",
                    "dynamic_warm_candidate": warm_candidate,
                    "raw_secret_values_recorded": False,
                },
                "raw_secret_values_recorded": False,
            }
            handoff_receipt_update = _cancel_unallocated_create(
                reason="runpod_create_pod_http_error_no_allocation",
                evidence={"http_status_code": exc.code},
            )
            manifest["provider_lane_handoff_receipt_update"] = handoff_receipt_update
            write_json(resolved_job_dir / "runpod_wam_async_create_manifest.json", manifest)
            return manifest
    if not pod_id:
        manifest = {
            "schema_version": RUNPOD_WAM_CREATE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "provider_bundle_kind": provider_bundle_kind,
            "blockers": ["runpod_create_response_missing_pod_id"],
            "http_status_code": status_code,
            "runpod_response_keys": sorted(response.keys()),
            "model_secret_env_status": model_secret_env_status,
            "provider_runtime_config_env_status": provider_runtime_config_env_status,
            "raw_secret_values_recorded": False,
        }
        handoff_receipt_update = _cancel_unallocated_create(
            reason="runpod_create_response_missing_pod_id",
            evidence={"http_status_code": status_code},
        )
        manifest["provider_lane_handoff_receipt_update"] = handoff_receipt_update
        write_json(resolved_job_dir / "runpod_wam_async_create_manifest.json", manifest)
        return manifest
    bind_pending_teardown_instance(pending_teardown["path"], pod_id)
    if provider_lane_handoff_receipt_path is not None:
        try:
            handoff_receipt_update = _update_provider_lane_handoff_receipt(
                provider_lane_handoff_receipt_path,
                pod_name=resolved_pod_name,
                pending_teardown_record=pending_teardown["path"],
                pod_id=pod_id,
            )
        except ValueError as exc:
            # The pre-create receipt already names the open pending record, whose
            # bound instance id is authoritative to the independent watchdog.
            handoff_receipt_update = {
                "status": "post_create_update_failed_pending_record_still_bound",
                "blocker": str(exc),
                "pod_id_present_in_pending_record": True,
                "raw_secret_values_recorded": False,
            }
    state = {
        "schema_version": RUNPOD_WAM_STATE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "pod_created",
        "job_dir": str(resolved_job_dir),
        "provider_bundle_kind": provider_bundle_kind,
        "pod_id": pod_id,
        "pod_launch_mode": launch_mode,
        "warm_existing_pod": warm_reuse_detail,
        "output_path": str(resolved_output),
        "public_base_url_present": bool(public_base_url),
        "explicit_provider_urls_used": direct_provider_urls,
        "provider_bundle_url_redacted": _redact_provider_url(provider_bundle_url),
        "provider_output_put_url_redacted": _redact_provider_url(provider_output_put_url),
        "provider_output_get_url_redacted": _redact_provider_url(provider_output_get_url),
        "provider_bundle_url_file": bundle_url_file_meta,
        "provider_output_put_url_file": output_url_file_meta,
        "provider_output_get_url_file": output_get_url_file_meta,
        "bundle_path": str(resolved_bundle),
        "full_loop_guard": full_loop_guard,
        "prelaunch_spend_guard": prelaunch_spend_guard,
        "pre_spend_preflight": pre_spend_preflight,
        "pending_teardown_record": pending_teardown["path"],
        "provider_lane_handoff_receipt_update": handoff_receipt_update,
        "token_file_path_redacted": True,
        "secret_env_file_path_redacted": True,
        "secret_artifact_policy": secret_path_disclosure_policy(),
        "image_name": image_name,
        "gpu_type_ids": list(gpu_type_ids),
        "cloud_type": cloud_type,
        "allowed_cuda_versions": list(allowed_cuda_versions),
        "min_vcpu_per_gpu": min_vcpu_per_gpu,
        "min_ram_per_gpu": min_ram_per_gpu,
        "container_disk_gb": container_disk_gb,
        "volume_gb": volume_gb,
        "carrier_volume_admission": (
            verify_carrier_volume_admission(
                carrier_volume_admission,
                expected_carrier_image_ref=image_name,
            )
            if carrier_volume_admission is not None
            else None
        ),
        "model_secret_env_status": model_secret_env_status,
        "provider_runtime_config_env_status": provider_runtime_config_env_status,
        "created_at_epoch": time.time(),
        "raw_secret_values_recorded": False,
    }
    write_json(_state_path(resolved_job_dir), state)
    manifest = {
        "schema_version": RUNPOD_WAM_CREATE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "pod_created",
        "job_dir": str(resolved_job_dir),
        "provider_bundle_kind": provider_bundle_kind,
        "pod_id": pod_id,
        "http_status_code": status_code,
        "output_path": str(resolved_output),
        "pod_launch_mode": launch_mode,
        "warm_existing_pod": warm_reuse_detail,
        "pod_request_summary": _redacted_payload_summary(payload),
        "explicit_provider_urls_used": direct_provider_urls,
        "provider_bundle_url_redacted": _redact_provider_url(provider_bundle_url),
        "provider_output_put_url_redacted": _redact_provider_url(provider_output_put_url),
        "provider_output_get_url_redacted": _redact_provider_url(provider_output_get_url),
        "provider_bundle_url_file": bundle_url_file_meta,
        "provider_output_put_url_file": output_url_file_meta,
        "provider_output_get_url_file": output_get_url_file_meta,
        "full_loop_guard": full_loop_guard,
        "prelaunch_spend_guard": prelaunch_spend_guard,
        "pre_spend_preflight": pre_spend_preflight,
        "pending_teardown_record": pending_teardown["path"],
        "provider_lane_handoff_receipt_update": handoff_receipt_update,
        "model_secret_env_status": model_secret_env_status,
        "provider_runtime_config_env_status": provider_runtime_config_env_status,
        "runpod_response_keys": sorted(response.keys()),
        "poll_command": f"python -m blueprint_pipeline.runpod_wam_async_runner poll --job-dir {resolved_job_dir}",
        "teardown_command": f"python -m blueprint_pipeline.runpod_wam_async_runner poll --job-dir {resolved_job_dir} --teardown",
        "raw_secret_values_recorded": False,
    }
    write_json(resolved_job_dir / "runpod_wam_async_create_manifest.json", manifest)
    return manifest


def _delete_pod(
    *,
    job_dir: Path,
    pod_id: str,
    api_key: str,
    generated_at: str,
) -> dict[str, Any]:
    try:
        status_code, response = _runpod_request(
            method="DELETE",
            path=f"/pods/{pod_id}",
            api_key=api_key,
            timeout_seconds=30,
        )
        status = "completed" if status_code in {200, 202, 204} else "blocked"
        blockers: list[str] = [] if status == "completed" else ["runpod_delete_pod_unexpected_status"]
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        response = {}
        status = "completed" if exc.code in {404, 410} else "blocked"
        blockers = [] if status == "completed" else ["runpod_delete_pod_http_error"]
    # The DELETE response is only a request acknowledgement. Teardown proof needs
    # the provider to report the pod terminal on a state query, so probe it.
    terminal_state_api_confirmed = False
    verified_pod_status: str | None = None
    terminal_state_verification: dict[str, Any] | None = None
    if status_code in {404, 410}:
        # The state API already says the allocation does not exist.
        terminal_state_api_confirmed = True
        verified_pod_status = "not_found"
    elif status == "completed":
        terminal_state_verification = _verify_pod_not_active_after_teardown_error(
            pod_id=pod_id,
            api_key=api_key,
            generated_at=generated_at,
        )
        probe_status = _string(terminal_state_verification.get("pod_status"))
        if probe_status and probe_status not in {"http_error", "status_probe_error"}:
            verified_pod_status = probe_status
        terminal_state_api_confirmed = bool(
            terminal_state_verification.get("spend_released")
        )
        if not terminal_state_api_confirmed:
            blockers = [
                *blockers,
                "runpod_delete_terminal_state_not_api_confirmed",
            ]
    manifest = {
        "schema_version": RUNPOD_WAM_DELETE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "job_dir": str(job_dir),
        "pod_id": pod_id,
        "http_status_code": status_code,
        "response_keys": sorted(response.keys()),
        "blockers": blockers,
        "terminal_state_api_confirmed": terminal_state_api_confirmed,
        "verified_pod_status": verified_pod_status,
        "terminal_state_verification": terminal_state_verification,
        "continuing_spend_from_this_run": status != "completed",
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "runpod_wam_async_delete_manifest.json", manifest)
    return manifest


def _verify_pod_not_active_after_teardown_error(
    *,
    pod_id: str,
    api_key: str,
    generated_at: str,
) -> dict[str, Any]:
    try:
        status_code, payload = _runpod_request(
            method="GET",
            path=f"/pods/{pod_id}",
            api_key=api_key,
            timeout_seconds=20,
        )
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 410}:
            return {
                "status": "completed",
                "generated_at": generated_at,
                "pod_id": pod_id,
                "http_status_code": exc.code,
                "pod_status": "not_found",
                "spend_released": True,
                "blockers": [],
                "raw_secret_values_recorded": False,
            }
        return {
            "status": "blocked",
            "generated_at": generated_at,
            "pod_id": pod_id,
            "http_status_code": exc.code,
            "pod_status": "http_error",
            "spend_released": False,
            "blockers": ["runpod_stop_error_status_probe_http_error"],
            "raw_secret_values_recorded": False,
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "status": "blocked",
            "generated_at": generated_at,
            "pod_id": pod_id,
            "pod_status": "status_probe_error",
            "spend_released": False,
            "blockers": ["runpod_stop_error_status_probe_failed"],
            "probe_error_type": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
    pod_status = _pod_status(payload)
    pod_status_upper = pod_status.upper()
    spend_released = bool(
        pod_status in RUNPOD_TERMINAL_POD_STATUSES
        or pod_status_upper in RUNPOD_TERMINAL_POD_STATUSES
    )
    return {
        "status": "completed" if spend_released else "blocked",
        "generated_at": generated_at,
        "pod_id": pod_id,
        "http_status_code": status_code,
        "pod_status": pod_status,
        "spend_released": spend_released,
        "blockers": []
        if spend_released
        else ["runpod_stop_error_pod_still_active_after_status_probe"],
        "raw_secret_values_recorded": False,
    }


def _stop_pod(
    *,
    job_dir: Path,
    pod_id: str,
    api_key: str,
    generated_at: str,
    record_warm_candidate: bool = True,
) -> dict[str, Any]:
    verification: dict[str, Any] | None = None
    stop_response_confirmed = False
    try:
        status_code, response = _runpod_request(
            method="POST",
            path=f"/pods/{pod_id}/stop",
            api_key=api_key,
            timeout_seconds=30,
        )
        status = "completed" if status_code in {200, 202, 204} else "blocked"
        stop_response_confirmed = status == "completed"
        blockers: list[str] = [] if status == "completed" else ["runpod_stop_pod_unexpected_status"]
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        response = {}
        status = "completed" if exc.code in {404, 410} else "blocked"
        blockers = [] if status == "completed" else ["runpod_stop_pod_http_error"]
        if status != "completed":
            verification = _verify_pod_not_active_after_teardown_error(
                pod_id=pod_id,
                api_key=api_key,
                generated_at=generated_at,
            )
            if verification.get("spend_released"):
                status = "completed"
                blockers = []
            else:
                blockers.extend(str(item) for item in verification.get("blockers") or [])
    warm_candidate = (
        _write_stopped_warm_candidate(
            job_dir=job_dir,
            pod_id=pod_id,
            generated_at=generated_at,
        )
        if status == "completed" and record_warm_candidate and stop_response_confirmed
        else {
            "status": "not_recorded",
            "reason": "runtime_output_not_successful_for_warm_reuse"
            if status == "completed" and stop_response_confirmed
            else "runpod_stop_completion_verified_without_reusable_stopped_pod"
            if status == "completed"
            else "runpod_stop_not_completed",
            "raw_secret_values_recorded": False,
        }
    )
    manifest = {
        "schema_version": RUNPOD_WAM_STOP_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "job_dir": str(job_dir),
        "pod_id": pod_id,
        "http_status_code": status_code,
        "stop_error_verification": verification,
        "response_keys": sorted(response.keys()),
        "blockers": blockers,
        "stopped_pod_preserved_for_warm_reuse": bool(
            status == "completed" and record_warm_candidate and stop_response_confirmed
        ),
        "warm_candidate_recording_requested": bool(record_warm_candidate),
        "warm_candidate": warm_candidate,
        "warm_candidate_path": warm_candidate.get("path"),
        "stop_response_confirmed": stop_response_confirmed,
        "gpu_spend_released_if_provider_honors_stop": status == "completed",
        "stopped_volume_storage_may_continue_billing": bool(
            status == "completed" and stop_response_confirmed
        ),
        "continuing_spend_from_this_run": status != "completed",
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "runpod_wam_async_stop_manifest.json", manifest)
    return manifest


def _teardown_action() -> str:
    action = _string(os.getenv(RUNPOD_WAM_TEARDOWN_ACTION_ENV)).lower()
    if action in {"stop", "stopped", "preserve", "warm"}:
        return "stop"
    if action in {"keep", "keep_running", "keep_on_success", "hot", "hot_reuse"}:
        return "keep_on_success"
    return "delete"


def _reliability_phase(passed: bool, blockers: Sequence[str] = (), **fields: Any) -> dict[str, Any]:
    return {
        "status": "PASS" if passed else "FAIL",
        "blockers": sorted({str(blocker) for blocker in blockers if str(blocker).strip()}),
        **fields,
    }


def _teardown_proof_from_runpod_poll(
    *,
    pod_id: str,
    teardown_requested: bool,
    teardown_manifest: Mapping[str, Any] | None,
    teardown_action: str,
    pod_status: str,
    keep_running_on_success: bool,
    generated_at: str,
) -> dict[str, Any]:
    terminal_status = ""
    terminate_requested = bool(teardown_requested)
    status_source: str | None = None
    teardown = _mapping(teardown_manifest)
    if keep_running_on_success:
        return build_teardown_proof(
            provider="runpod",
            allocation_id=pod_id,
            terminate_requested=False,
            provider_terminal_status=None,
            keep_alive_requested=True,
            keep_alive_reason="keep_on_success",
        )
    if teardown.get("status") == "completed":
        terminate_requested = True
        if teardown.get("terminal_state_api_confirmed") is True and _string(
            teardown.get("verified_pod_status")
        ):
            # The delete/stop manifest carries a provider state query result.
            terminal_status = _string(teardown.get("verified_pod_status")).lower()
            status_source = TEARDOWN_STATUS_SOURCE_PROVIDER_API
        else:
            # Legacy self-reported completion — recorded, but cannot prove teardown.
            terminal_status = "deleted" if teardown_action == "delete" else "stopped"
    elif pod_status in RUNPOD_TERMINAL_POD_STATUSES or pod_status.upper() in RUNPOD_TERMINAL_POD_STATUSES:
        # pod_status comes from the poll's own GET /pods/{id} query — API evidence.
        terminal_status = "not_found" if pod_status == "not_found" else pod_status.lower()
        terminate_requested = True
        status_source = TEARDOWN_STATUS_SOURCE_PROVIDER_API
    return build_teardown_proof(
        provider="runpod",
        allocation_id=pod_id,
        terminate_requested=terminate_requested,
        provider_terminal_status=terminal_status or None,
        verified_at=generated_at if terminal_status else None,
        status_source=status_source,
    )


def _write_wam_provider_reliability_manifest(
    *,
    job_dir: Path,
    state: Mapping[str, Any],
    poll_manifest: Mapping[str, Any],
    teardown_manifest: Mapping[str, Any] | None,
    generated_at: str,
) -> str:
    output_zip_present = poll_manifest.get("output_zip_present") is True
    provider_output_validation = _mapping(poll_manifest.get("provider_output_validation"))
    provider_output_validation_status = _string(provider_output_validation.get("status"))
    provider_output_terminal = poll_manifest.get("provider_output_terminal") is True
    provider_output_contract_valid = bool(
        output_zip_present and provider_output_validation_status == "completed"
    )
    output_present = bool(output_zip_present and provider_output_terminal)
    pod_id = _string(poll_manifest.get("pod_id"))
    pod_status = _string(poll_manifest.get("pod_status"))
    provider_bundle_kind = _string(poll_manifest.get("provider_bundle_kind")) or "wam"
    stall_evaluation = _mapping(poll_manifest.get("stall_evaluation"))
    stall_blockers = [str(b) for b in stall_evaluation.get("blockers") or []]
    runtime_result_status = _string(poll_manifest.get("runtime_result_status")).lower()
    runtime_result_blockers = [
        str(blocker) for blocker in poll_manifest.get("runtime_result_blockers") or []
    ]
    active_without_terminal_output = bool(
        poll_manifest.get("remote_runtime_running_without_terminal_output")
        or poll_manifest.get("nonterminal_running_output")
    )

    preflight = _reliability_phase(
        bool(pod_id),
        [] if pod_id else ["runpod_wam_state_missing_pod_id"],
        state_schema=state.get("schema_version"),
    )
    launch = _reliability_phase(
        bool(pod_id),
        [] if pod_id else ["provider_launch_failed:pod_id_missing"],
        pod_launch_mode=state.get("pod_launch_mode"),
        pod_id=pod_id or None,
    )
    startup_seen = bool(output_zip_present or poll_manifest.get("last_nonterminal_output"))
    startup_blockers = []
    if not startup_seen and not active_without_terminal_output:
        startup_blockers.append("startup_marker_timeout:no_runtime_or_heartbeat_observed")
    if stall_evaluation.get("stall_mode") == "container_startup":
        startup_blockers.extend(stall_blockers)
    startup = _reliability_phase(
        not startup_blockers,
        startup_blockers,
        startup_marker_seen=startup_seen,
        pod_status=pod_status or None,
    )

    runtime_blockers: list[str] = []
    if stall_evaluation.get("stall_mode") == "runtime_execution":
        runtime_blockers.extend(stall_blockers)
    if output_zip_present and provider_output_validation_status == "blocked":
        runtime_blockers.extend(
            str(blocker) for blocker in provider_output_validation.get("blockers") or []
        )
    if runtime_result_status in {"blocked", "failed", "error", "timeout", "timed_out"}:
        runtime_blockers.append(f"runner_failed:{runtime_result_status}")
        runtime_blockers.extend(f"runner_failed:{blocker}" for blocker in runtime_result_blockers)
    runtime = _reliability_phase(
        not runtime_blockers and (output_present or active_without_terminal_output),
        runtime_blockers or ([] if output_present or active_without_terminal_output else ["runtime_not_observed"]),
        provider_command_status=poll_manifest.get("provider_command_status"),
        runtime_result_status=runtime_result_status or None,
    )
    artifact_collection_blockers = (
        []
        if provider_output_contract_valid
        else [
            str(blocker)
            for blocker in (
                provider_output_validation.get("blockers")
                or ["artifact_collection_failed:runtime_output_zip_not_received"]
            )
        ]
    )
    collection = _reliability_phase(
        provider_output_contract_valid,
        artifact_collection_blockers,
        output_zip_path=poll_manifest.get("provider_runtime_output_zip_path"),
        output_zip_present=output_zip_present,
        provider_output_validation_status=provider_output_validation_status or None,
    )
    teardown = _teardown_proof_from_runpod_poll(
        pod_id=pod_id,
        teardown_requested=parse_bool(
            poll_manifest.get("teardown_requested"),
            default=False,
        ),
        teardown_manifest=teardown_manifest,
        teardown_action=_string(poll_manifest.get("teardown_action")) or "not_requested",
        pod_status=pod_status,
        keep_running_on_success=parse_bool(
            poll_manifest.get("keep_running_on_success"),
            default=False,
        ),
        generated_at=generated_at,
    )
    manifest = build_provider_reliability_manifest(
        run_id=pod_id or _string(state.get("generated_at")) or generated_at,
        provider="runpod",
        session_dir=str(job_dir),
        launched_at=_string(state.get("generated_at")) or None,
        pre_spend_preflight=preflight,
        provider_launch=launch,
        container_startup=startup,
        runtime_execution=runtime,
        artifact_collection=collection,
        teardown=teardown,
        not_applicable_phases=("artifact_quality", "task_evaluation"),
        spend={
            "provider_bundle_kind": provider_bundle_kind,
            "continuing_spend_from_this_run": parse_bool(
                poll_manifest.get("continuing_spend_from_this_run"),
                default=False,
            ),
            "teardown_action": poll_manifest.get("teardown_action"),
        },
    )
    # A PASSING teardown proof releases the crash-safety record; anything less
    # keeps it open for reap_orphans.
    pending_teardown_record = _string(state.get("pending_teardown_record"))
    if pending_teardown_record:
        close_pending_teardown(pending_teardown_record, teardown)
    path = job_dir / RUNPOD_WAM_PROVIDER_RELIABILITY_MANIFEST_NAME
    write_json(path, manifest)
    return str(path)


def _download_provider_output_zip(
    *,
    job_dir: Path,
    provider_output_get_url: str,
    output_path: Path,
    generated_at: str,
    provider_bundle_kind: str = "wam",
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": "runpod_wam_output_download.v1",
        "generated_at": generated_at,
        "status": "not_requested",
        "output_path": str(output_path),
        "provider_output_get_url_redacted": _redact_provider_url(provider_output_get_url),
        "raw_secret_values_recorded": False,
    }
    if not _string(provider_output_get_url):
        manifest.update({"status": "skipped", "reason": "provider_output_get_url_missing"})
        write_json(job_dir / "runpod_wam_output_download_manifest.json", manifest)
        return manifest
    transfer = download_url_to_file(
        url=provider_output_get_url,
        output_path=output_path,
        user_agent="BlueprintRunPodWamPoll/1.0",
        timeout_seconds=60,
    )
    if transfer["status"] == "completed":
        downloaded_size = int(transfer["downloaded_size_bytes"])
        valid_zip = downloaded_size > 0 and zipfile.is_zipfile(output_path)
        output_validation = (
            _validate_wam_provider_output_zip(
                output_path,
                provider_bundle_kind=provider_bundle_kind,
            )
            if valid_zip
            else {}
        )
        manifest.update(
            {
                "status": "completed" if valid_zip else "not_available",
                "downloaded_size_bytes": downloaded_size,
                "output_present": valid_zip,
                "valid_zip": valid_zip,
                "empty_download": downloaded_size == 0,
                "provider_output_validation": output_validation,
                "provider_output_validation_status": output_validation.get("status")
                if output_validation
                else None,
                "provider_output_terminal": output_validation.get("terminal_output_present")
                if output_validation
                else False,
                "provider_output_usable": output_validation.get("provider_output_usable")
                if output_validation
                else False,
            }
        )
        if not valid_zip:
            output_path.unlink(missing_ok=True)
    elif transfer["status"] == "http_error":
        manifest.update(
            {
                "status": "not_available",
                "http_status_code": transfer.get("http_status_code"),
                "error_type": "HTTPError",
                "output_present": output_path.is_file(),
            }
        )
    else:
        manifest.update(
            {
                "status": "blocked",
                "error_type": transfer.get("error_type"),
                "output_present": output_path.is_file(),
            }
        )
    write_json(job_dir / "runpod_wam_output_download_manifest.json", manifest)
    return manifest


def _zip_wam_provider_output_status(zip_path: Path) -> str:
    """Status from ``wam_provider_output.json`` inside a runtime-output zip, or '' if absent.

    The OSCAR provider's RunPod outer wrapper uploads an early heartbeat zip containing only
    ``wam_provider_output.json`` (status=running) before any ``wam_runtime_result.json`` exists.
    ``_inspect_provider_runtime_output_zip`` only reads the runtime-result files, so without this
    the poll mistakes that first heartbeat for completion and tears the pod down before the model
    can install deps, download the checkpoint, and run inference.
    """
    try:
        with zipfile.ZipFile(zip_path) as archive:
            if "wam_provider_output.json" not in archive.namelist():
                return ""
            payload = json.loads(archive.read("wam_provider_output.json").decode("utf-8"))
    except (OSError, ValueError, zipfile.BadZipFile):
        return ""
    if not isinstance(payload, Mapping):
        return ""
    return _string(payload.get("status"))


def _provider_output_runtime_result_candidates(provider_bundle_kind: str) -> tuple[str, ...]:
    return RUNPOD_WAM_PROVIDER_OUTPUT_RUNTIME_RESULT_NAMES.get(
        provider_bundle_kind,
        RUNPOD_WAM_PROVIDER_OUTPUT_RUNTIME_RESULT_NAMES["wam"],
    )


def _read_provider_output_json_member(
    archive: zipfile.ZipFile,
    member: str,
) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(archive.read(member).decode("utf-8") or "{}")
    except Exception as exc:
        return None, type(exc).__name__
    if not isinstance(payload, Mapping):
        return None, "not_object"
    return dict(payload), ""


def _provider_output_member(
    names: Sequence[str],
    suffixes: Sequence[str],
) -> str:
    for suffix in suffixes:
        for name in names:
            if name == suffix or name.endswith(f"/{suffix}"):
                return name
    return ""


def _provider_output_status_kind(status: str) -> str:
    normalized = status.lower()
    if normalized in RUNPOD_WAM_PROVIDER_OUTPUT_NONTERMINAL_STATUSES:
        return "nonterminal"
    if normalized in RUNPOD_WAM_PROVIDER_OUTPUT_SUCCESS_STATUSES:
        return "success"
    if normalized in RUNPOD_WAM_PROVIDER_OUTPUT_FAILURE_STATUSES:
        return "failure"
    return ""


def _provider_output_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers = payload.get("blockers")
    if not isinstance(blockers, Sequence) or isinstance(blockers, (str, bytes)):
        return []
    return [str(blocker) for blocker in blockers if str(blocker)]


def _validate_wam_provider_output_zip(
    zip_path: Path,
    *,
    provider_bundle_kind: str,
) -> dict[str, Any]:
    resolved = zip_path.expanduser()
    manifest: dict[str, Any] = {
        "schema_version": "runpod_wam_provider_output_validation.v1",
        "status": "missing",
        "zip_path": str(resolved),
        "zip_present": False,
        "valid_zip": False,
        "terminal_output_present": False,
        "provider_output_usable": False,
        "runtime_result_manifest_present": False,
        "entrypoint_execution_manifest_present": False,
        "heartbeat_manifest_present": False,
        "blockers": ["provider_output_zip_missing"],
        "raw_secret_values_recorded": False,
    }
    if not resolved.is_file():
        return manifest
    manifest.update(
        {
            "status": "blocked",
            "zip_present": True,
            "zip_size_bytes": resolved.stat().st_size,
            "blockers": [],
        }
    )
    if not zipfile.is_zipfile(resolved):
        manifest.update(
            {
                "valid_zip": False,
                "blockers": ["provider_output_zip_invalid"],
            }
        )
        return manifest

    try:
        with zipfile.ZipFile(resolved) as archive:
            names = sorted(archive.namelist())
            manifest.update(
                {
                    "valid_zip": True,
                    "zip_member_count": len(names),
                    "zip_members_preview": names[:50],
                }
            )
            runtime_member = _provider_output_member(
                names,
                _provider_output_runtime_result_candidates(provider_bundle_kind),
            )
            if runtime_member:
                payload, parse_error = _read_provider_output_json_member(archive, runtime_member)
                manifest.update(
                    {
                        "runtime_result_manifest_present": True,
                        "runtime_result_manifest_name": runtime_member,
                    }
                )
                if parse_error or payload is None:
                    manifest["blockers"] = [
                        f"provider_output_manifest_malformed:{runtime_member}:{parse_error}"
                    ]
                    return manifest
                runtime_status = _string(payload.get("status")).lower()
                runtime_blockers = _provider_output_blockers(payload)
                manifest.update(
                    {
                        "runtime_result_status": runtime_status or None,
                        "runtime_result_blockers": runtime_blockers,
                    }
                )
                status_kind = _provider_output_status_kind(runtime_status)
                if status_kind == "nonterminal":
                    manifest.update(
                        {
                            "status": "nonterminal",
                            "blockers": [],
                            "nonterminal_runtime_result_status": runtime_status,
                        }
                    )
                    return manifest
                if not runtime_status:
                    manifest["blockers"] = [
                        f"provider_output_manifest_status_missing:{runtime_member}"
                    ]
                    return manifest
                if not status_kind:
                    manifest["blockers"] = [
                        f"provider_output_manifest_status_unrecognized:{runtime_status}"
                    ]
                    return manifest
                provider_output_usable = bool(status_kind == "success" and not runtime_blockers)
                manifest.update(
                    {
                        "status": "completed",
                        "terminal_output_present": True,
                        "provider_output_usable": provider_output_usable,
                        "blockers": []
                        if provider_output_usable
                        else (
                            runtime_blockers
                            or [f"provider_output_manifest_status:{runtime_status}"]
                        ),
                    }
                )
                return manifest

            entrypoint_member = _provider_output_member(
                names,
                (RUNPOD_WAM_PROVIDER_ENTRYPOINT_MANIFEST_NAME,),
            )
            if entrypoint_member:
                payload, parse_error = _read_provider_output_json_member(archive, entrypoint_member)
                manifest.update(
                    {
                        "entrypoint_execution_manifest_present": True,
                        "entrypoint_execution_manifest_name": entrypoint_member,
                    }
                )
                if parse_error or payload is None:
                    manifest["blockers"] = [
                        f"provider_entrypoint_execution_manifest_malformed:"
                        f"{entrypoint_member}:{parse_error}"
                    ]
                    return manifest
                entrypoint_status = _string(payload.get("status")).lower()
                returncode = payload.get("returncode")
                entrypoint_blockers = _provider_output_blockers(payload)
                entrypoint_failed = bool(
                    _provider_output_status_kind(entrypoint_status) == "failure"
                    or (isinstance(returncode, int) and returncode != 0)
                )
                manifest.update(
                    {
                        "entrypoint_execution_status": entrypoint_status or None,
                        "entrypoint_execution_returncode": returncode
                        if isinstance(returncode, int)
                        else None,
                    }
                )
                if entrypoint_failed:
                    manifest.update(
                        {
                            "status": "completed",
                            "terminal_output_present": True,
                            "provider_output_usable": False,
                            "runtime_result_status": "blocked",
                            "runtime_result_blockers": entrypoint_blockers
                            or ["provider_entrypoint_failed_without_runtime_result"],
                            "blockers": entrypoint_blockers
                            or ["provider_entrypoint_failed_without_runtime_result"],
                        }
                    )
                    return manifest
                if _provider_output_status_kind(entrypoint_status) == "success":
                    manifest["blockers"] = [
                        "provider_runtime_result_manifest_missing_after_completed_entrypoint"
                    ]
                    return manifest

            heartbeat_member = _provider_output_member(
                names,
                (RUNPOD_WAM_PROVIDER_HEARTBEAT_MANIFEST_NAME,),
            )
            if heartbeat_member:
                payload, parse_error = _read_provider_output_json_member(archive, heartbeat_member)
                manifest.update(
                    {
                        "heartbeat_manifest_present": True,
                        "heartbeat_manifest_name": heartbeat_member,
                    }
                )
                if parse_error or payload is None:
                    manifest["blockers"] = [
                        f"provider_output_heartbeat_manifest_malformed:"
                        f"{heartbeat_member}:{parse_error}"
                    ]
                    return manifest
                heartbeat_status = _string(payload.get("status")).lower()
                heartbeat_blockers = _provider_output_blockers(payload)
                manifest["heartbeat_status"] = heartbeat_status or None
                status_kind = _provider_output_status_kind(heartbeat_status)
                if status_kind == "nonterminal":
                    manifest.update(
                        {
                            "status": "nonterminal",
                            "blockers": [],
                            "nonterminal_runtime_result_status": heartbeat_status,
                        }
                    )
                    return manifest
                if status_kind == "failure":
                    manifest.update(
                        {
                            "status": "completed",
                            "terminal_output_present": True,
                            "provider_output_usable": False,
                            "runtime_result_status": heartbeat_status,
                            "runtime_result_blockers": heartbeat_blockers
                            or [f"provider_output_heartbeat_status:{heartbeat_status}"],
                            "blockers": heartbeat_blockers
                            or [f"provider_output_heartbeat_status:{heartbeat_status}"],
                        }
                    )
                    return manifest
                if status_kind == "success":
                    manifest["blockers"] = [
                        "provider_runtime_result_manifest_missing_after_completed_heartbeat"
                    ]
                    return manifest

            manifest["blockers"] = ["provider_runtime_result_manifest_missing"]
            return manifest
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        manifest.update(
            {
                "status": "blocked",
                "valid_zip": False,
                "blockers": [f"provider_output_zip_invalid:{type(exc).__name__}"],
            }
        )
        return manifest


def poll_runpod_wam_async_run(
    *,
    job_dir: str | Path,
    max_wait_seconds: int = 60,
    retry_interval_seconds: int = 5,
    teardown: bool = False,
    post_marker_no_progress_timeout_seconds: int | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    state = _read_json(_state_path(resolved_job_dir))
    pod_id = _string(state.get("pod_id"))
    try:
        created_at_epoch = float(state.get("created_at_epoch") or time.time())
    except (TypeError, ValueError):
        created_at_epoch = time.time()
    not_found_grace_seconds = _env_int("BLUEPRINT_RUNPOD_POD_STATUS_NOT_FOUND_GRACE_SECONDS", 300)
    output_path = Path(_string(state.get("output_path"))).expanduser()
    provider_bundle_kind = _string(state.get("provider_bundle_kind")) or "wam"
    if provider_bundle_kind not in RUNPOD_PROVIDER_BUNDLE_KINDS:
        blockers = [f"unsupported_provider_bundle_kind:{provider_bundle_kind}"]
    else:
        blockers = []
    output_get_url = ""
    output_get_meta = _mapping(state.get("provider_output_get_url_file"))
    output_get_path = _string(output_get_meta.get("path"))
    if output_get_path:
        output_get_url, _output_get_meta = _read_sensitive_url_file(
            output_get_path,
            label="provider_output_get_url_file",
        )
    api_key, api_key_meta = _read_runpod_api_key()
    if not pod_id:
        blockers.append("runpod_wam_state_missing_pod_id")
    if not api_key:
        blockers.append(f"missing_env_{RUNPOD_API_KEY_ENV}_or_{RUNPOD_API_KEY_FILE_ENV}")
    status_code: int | None = None
    pod_payload: dict[str, Any] = {}
    pod_status = "unknown"
    started_monotonic = time.monotonic()
    deadline = time.monotonic() + max(0, max_wait_seconds)
    post_marker_timeout_seconds = _runpod_wam_post_marker_timeout_seconds(
        post_marker_no_progress_timeout_seconds
    )
    last_progress_monotonic: float | None = None
    stall_evaluation: dict[str, Any] = {}
    stall_teardown_requested = False
    output_present = output_path.is_file()
    last_nonterminal_output: dict[str, Any] | None = None
    transient_not_found_count = 0
    transient_pod_status_error_count = 0
    last_pod_status_error: dict[str, Any] | None = None
    existing_teardown_completed = False
    while not blockers and time.monotonic() <= deadline:
        output_present = output_path.is_file()
        if not output_present and output_get_url:
            download_manifest = _download_provider_output_zip(
                job_dir=resolved_job_dir,
                provider_output_get_url=output_get_url,
                output_path=output_path,
                generated_at=generated,
                provider_bundle_kind=provider_bundle_kind,
            )
            output_present = output_path.is_file()
            if download_manifest.get("status") == "completed":
                output_validation = _mapping(download_manifest.get("provider_output_validation"))
                downloaded_inspection = _inspect_provider_runtime_output_zip(
                    output_path,
                    expected_video_count=0,
                )
                runtime_status = _string(
                    output_validation.get("runtime_result_status")
                    or downloaded_inspection.get("runtime_result_status")
                )
                # The OSCAR outer wrapper heartbeats via wam_provider_output.json (status=running)
                # with no wam_runtime_result.json yet, so runtime_result_status is empty for it.
                # Fall back to the provider-output status so the first heartbeat is recognized as
                # nonterminal and the poll keeps waiting for the model to finish.
                heartbeat_status = _string(
                    output_validation.get("nonterminal_runtime_result_status")
                    or runtime_status
                    or output_validation.get("heartbeat_status")
                )
                if not heartbeat_status:
                    heartbeat_status = _zip_wam_provider_output_status(output_path)
                if (
                    output_validation.get("status") == "nonterminal"
                    or heartbeat_status in RUNPOD_WAM_PROVIDER_OUTPUT_NONTERMINAL_STATUSES
                ):
                    last_progress_monotonic = time.monotonic()
                    nonterminal_path = output_path.with_name(
                        f"{output_path.stem}_nonterminal{output_path.suffix}"
                    )
                    output_path.replace(nonterminal_path)
                    last_nonterminal_output = {
                        "schema_version": "runpod_wam_nonterminal_output.v1",
                        "generated_at": generated,
                        "status": "running",
                        "runtime_result_status": heartbeat_status,
                        "runtime_result": downloaded_inspection.get("runtime_result"),
                        "nonterminal_zip_path": str(nonterminal_path),
                        "nonterminal_zip_size_bytes": nonterminal_path.stat().st_size,
                        "provider_bundle_kind": provider_bundle_kind,
                        "provider_output_validation": output_validation,
                        "raw_secret_values_recorded": False,
                    }
                    write_json(
                        resolved_job_dir / "runpod_wam_nonterminal_output_manifest.json",
                        last_nonterminal_output,
                    )
                    download_manifest.update(
                        {
                            "status": "nonterminal",
                            "output_present": False,
                            "terminal_output_present": False,
                            "nonterminal_runtime_result_status": heartbeat_status,
                            "nonterminal_zip_path": str(nonterminal_path),
                            "nonterminal_zip_size_bytes": nonterminal_path.stat().st_size,
                        }
                    )
                    write_json(
                        resolved_job_dir / "runpod_wam_output_download_manifest.json",
                        download_manifest,
                    )
                    output_present = False
                else:
                    break
        if output_present:
            break
        try:
            status_code, pod_payload = _runpod_request(
                method="GET",
                path=f"/pods/{pod_id}",
                api_key=api_key,
                timeout_seconds=20,
            )
            pod_status = _pod_status(pod_payload)
        except urllib.error.HTTPError as exc:
            status_code = exc.code
            if exc.code in {404, 410}:
                try:
                    existing_delete_manifest = _read_json(
                        resolved_job_dir / "runpod_wam_async_delete_manifest.json"
                    )
                except (OSError, ValueError):
                    existing_delete_manifest = {}
                if (
                    existing_delete_manifest.get("status") == "completed"
                    and _string(existing_delete_manifest.get("pod_id")) == pod_id
                ):
                    existing_teardown_completed = True
                    pod_status = "not_found"
                    break
                elapsed_since_create = max(0.0, time.time() - created_at_epoch)
                if elapsed_since_create <= not_found_grace_seconds:
                    transient_not_found_count += 1
                    pod_status = "pending_api_visibility"
                    if time.monotonic() + retry_interval_seconds > deadline:
                        break
                    time.sleep(max(1, retry_interval_seconds))
                    continue
                pod_status = "not_found"
            else:
                pod_status = "http_error"
            if exc.code not in {404, 410}:
                blockers.append("runpod_pod_status_http_error")
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            status_code = None
            transient_pod_status_error_count += 1
            pod_status = "status_probe_error"
            last_pod_status_error = {
                "error_type": type(exc).__name__,
                "message_preview": str(exc)[:300],
                "raw_secret_values_recorded": False,
            }
            if time.monotonic() + retry_interval_seconds > deadline:
                break
            time.sleep(max(1, retry_interval_seconds))
            continue
        if output_present:
            break
        pod_status_is_active_now = (
            pod_status in RUNPOD_ACTIVE_POD_STATUSES
            or pod_status.upper() in RUNPOD_ACTIVE_POD_STATUSES
        )
        if post_marker_timeout_seconds > 0 and pod_status_is_active_now:
            now_monotonic = time.monotonic()
            startup_elapsed_seconds = max(0.0, now_monotonic - started_monotonic)
            last_progress_age_seconds = (
                max(0.0, now_monotonic - last_progress_monotonic)
                if last_progress_monotonic is not None
                else None
            )
            stall_evaluation = evaluate_post_marker_stall(
                startup_marker_seen=last_progress_monotonic is not None,
                startup_elapsed_seconds=startup_elapsed_seconds,
                startup_timeout_seconds=post_marker_timeout_seconds,
                last_progress_age_seconds=last_progress_age_seconds,
                no_progress_timeout_seconds=post_marker_timeout_seconds,
            )
            if stall_evaluation.get("should_terminate"):
                stall_teardown_requested = True
                break
        if time.monotonic() + retry_interval_seconds > deadline:
            break
        time.sleep(max(1, retry_interval_seconds))
    output_inspection = _inspect_provider_runtime_output_zip(
        output_path,
        video_extract_dir=resolved_job_dir / "runpod_wam_output_videos",
        expected_video_count=0 if provider_bundle_kind == "unitree_unifolm" else 1,
    )
    provider_output_validation = _validate_wam_provider_output_zip(
        output_path,
        provider_bundle_kind=provider_bundle_kind,
    )
    output_zip_present = output_inspection.get("zip_present") is True
    provider_output_validation_status = _string(provider_output_validation.get("status"))
    provider_output_terminal = provider_output_validation.get("terminal_output_present") is True
    provider_output_usable = provider_output_validation.get("provider_output_usable") is True
    provider_output_validation_failed = bool(
        output_zip_present and provider_output_validation_status == "blocked"
    )
    output_present = bool(
        output_zip_present and (provider_output_terminal or provider_output_validation_failed)
    )
    runtime_result_status = _string(
        provider_output_validation.get("runtime_result_status")
        or output_inspection.get("runtime_result_status")
    )
    # Provider-runtime layer only: the pod stayed alive and the output zip arrived
    # with a recognized successful terminal result. This is infrastructure health,
    # not task success.
    provider_runtime_operational = provider_output_usable
    # Task layer: only an explicit boolean task_success in the runtime result counts.
    # Absent or non-boolean stays None — provider runtime success never promotes to it.
    runtime_result_payload = output_inspection.get("runtime_result")
    runtime_task_success = (
        runtime_result_payload.get("task_success")
        if isinstance(runtime_result_payload, Mapping)
        and isinstance(runtime_result_payload.get("task_success"), bool)
        else None
    )
    runtime_output_success = provider_runtime_operational
    runtime_result_failed = (
        runtime_result_status in RUNPOD_WAM_PROVIDER_OUTPUT_FAILURE_STATUSES
    )
    elapsed_wait_seconds = max(0.0, time.monotonic() - started_monotonic)
    wait_deadline_expired = elapsed_wait_seconds >= max(0, max_wait_seconds)
    pod_status_is_active = (
        pod_status in RUNPOD_ACTIVE_POD_STATUSES
        or pod_status.upper() in RUNPOD_ACTIVE_POD_STATUSES
    )
    pod_status_is_terminal = (
        pod_status in RUNPOD_TERMINAL_POD_STATUSES
        or pod_status.upper() in RUNPOD_TERMINAL_POD_STATUSES
    )
    teardown_action = _teardown_action()
    requested_keep_running_on_success = bool(
        teardown
        and teardown_action == "keep_on_success"
        and runtime_output_success
        and pod_id
        and not pod_status_is_terminal
    )
    keepalive_runtime_health: dict[str, Any] | None = None
    if requested_keep_running_on_success:
        try:
            keepalive_status_code, keepalive_pod_payload = _runpod_request(
                method="GET",
                path=f"/pods/{pod_id}",
                api_key=api_key,
                timeout_seconds=20,
            )
            pod_status = _pod_status(keepalive_pod_payload)
            status_code = keepalive_status_code
            pod_status_is_active = (
                pod_status in RUNPOD_ACTIVE_POD_STATUSES
                or pod_status.upper() in RUNPOD_ACTIVE_POD_STATUSES
            )
            pod_status_is_terminal = (
                pod_status in RUNPOD_TERMINAL_POD_STATUSES
                or pod_status.upper() in RUNPOD_TERMINAL_POD_STATUSES
            )
            runtime_present = _pod_runtime_present(keepalive_pod_payload)
            public_ip_present = _pod_public_ip_present(keepalive_pod_payload)
            active_status_without_runtime_metadata = bool(
                pod_status_is_active and not runtime_present
            )
            runtime_healthy_for_hot_reuse = bool(
                pod_status_is_active
                and (runtime_present or active_status_without_runtime_metadata)
            )
            keepalive_runtime_health = {
                "status": "healthy_for_hot_reuse"
                if runtime_healthy_for_hot_reuse
                else "unhealthy_for_hot_reuse",
                "pod_status_http_status_code": keepalive_status_code,
                "pod_status": pod_status,
                "runtime_present": runtime_present,
                "public_ip_present": public_ip_present,
                "active_status_without_runtime_metadata": active_status_without_runtime_metadata,
                "health_basis": (
                    "runtime_metadata_present"
                    if runtime_present
                    else "active_pod_status_without_runtime_metadata"
                    if active_status_without_runtime_metadata
                    else "inactive_or_terminal_pod_status"
                ),
                "runtime_healthy_for_hot_reuse": runtime_healthy_for_hot_reuse,
                "raw_secret_values_recorded": False,
            }
        except urllib.error.HTTPError as exc:
            status_code = exc.code
            keepalive_runtime_health = {
                "status": "probe_http_error",
                "pod_status_http_status_code": exc.code,
                "runtime_healthy_for_hot_reuse": False,
                "raw_secret_values_recorded": False,
            }
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            keepalive_runtime_health = {
                "status": "probe_failed",
                "error_type": type(exc).__name__,
                "runtime_healthy_for_hot_reuse": False,
                "raw_secret_values_recorded": False,
            }
    remote_runtime_running_without_terminal_output = bool(
        not output_present and pod_status_is_active
    )
    runtime_stall_observed = bool(stall_evaluation.get("should_terminate"))
    auto_teardown_failure = bool(
        runtime_stall_observed or runtime_result_failed or provider_output_validation_failed
    )
    nonterminal_running_output = bool(
        last_nonterminal_output
        and not output_present
        and pod_status_is_active
    )
    keep_running_on_success = bool(
        requested_keep_running_on_success
        and _mapping(keepalive_runtime_health).get("runtime_healthy_for_hot_reuse") is True
    )
    keepalive_runtime_unhealthy_on_success = bool(
        requested_keep_running_on_success and not keep_running_on_success
    )
    should_teardown = bool(
        auto_teardown_failure
        or (
            teardown
            and (output_present or pod_status_is_terminal or runtime_stall_observed)
            and not keep_running_on_success
        )
    )
    effective_teardown_action = "delete" if auto_teardown_failure else teardown_action
    effective_teardown_requested = bool(teardown or auto_teardown_failure)
    teardown_pending = bool(
        not blockers and should_teardown and pod_id and api_key and pod_status != "not_found"
    )
    teardown_manifest: dict[str, Any] | None = None
    continuing_spend = bool(
        pod_id
        and not pod_status_is_terminal
        and (
            keep_running_on_success
            or (
                not output_present
                and (
                    nonterminal_running_output
                    or remote_runtime_running_without_terminal_output
                    or not should_teardown
                    or (teardown_manifest or {}).get("status") != "completed"
                )
            )
        )
    )
    provider_status = (
        "completed"
        if provider_output_usable
        else (
            "blocked"
            if output_present or runtime_stall_observed
            else ("running" if remote_runtime_running_without_terminal_output else "blocked")
        )
    )
    provider_blockers: list[str] = []
    if runtime_stall_observed:
        provider_blockers.extend(
            str(blocker) for blocker in stall_evaluation.get("blockers") or []
        )
    if output_zip_present and not provider_output_usable:
        provider_blockers.extend(
            str(blocker) for blocker in provider_output_validation.get("blockers") or []
        )
    if not output_present and not remote_runtime_running_without_terminal_output:
        provider_blockers.append("runpod_provider_runtime_output_zip_not_received_locally")
    if blockers:
        provider_blockers.extend(blockers)
    poll_status = (
        "blocked"
        if provider_output_validation_failed
        else ("completed" if output_present else ("running" if continuing_spend else "blocked"))
    )
    manifest = {
        "schema_version": RUNPOD_WAM_POLL_SCHEMA_VERSION,
        "generated_at": generated,
        "status": poll_status,
        "job_dir": str(resolved_job_dir),
        "provider_bundle_kind": provider_bundle_kind,
        "pod_id": pod_id,
        "pod_status": pod_status,
        "pod_status_is_active": pod_status_is_active,
        "pod_status_is_terminal": pod_status_is_terminal,
        "pod_status_http_status_code": status_code,
        "pod_status_not_found_grace_seconds": not_found_grace_seconds,
        "pod_status_transient_not_found_count": transient_not_found_count,
        "pod_status_transient_error_count": transient_pod_status_error_count,
        "last_pod_status_error": last_pod_status_error,
        "provider_command_status": provider_status,
        "provider_command_blockers": provider_blockers,
        "output_zip_present": output_zip_present,
        "provider_output_terminal": provider_output_terminal,
        "provider_output_usable": provider_output_usable,
        "provider_output_validation_status": provider_output_validation_status or None,
        "provider_output_validation": provider_output_validation,
        "runtime_output_success": runtime_output_success,
        "provider_runtime_operational": provider_runtime_operational,
        "runtime_task_success": runtime_task_success,
        "runtime_output_success_is_provider_runtime_only": True,
        "provider_runtime_success_is_not_task_success": True,
        "nonterminal_running_output": nonterminal_running_output,
        "remote_runtime_running_without_terminal_output": (
            remote_runtime_running_without_terminal_output
        ),
        "post_marker_no_progress_timeout_seconds": post_marker_timeout_seconds,
        "stall_evaluation": stall_evaluation,
        "runtime_stall_observed": runtime_stall_observed,
        "auto_teardown_failure": auto_teardown_failure,
        "stall_teardown_requested": stall_teardown_requested,
        "elapsed_wait_seconds": round(elapsed_wait_seconds, 6),
        "max_wait_seconds": max_wait_seconds,
        "wait_deadline_expired": wait_deadline_expired,
        "provider_runtime_output_zip_path": str(output_path),
        "runtime_result": output_inspection.get("runtime_result"),
        "runtime_result_status": runtime_result_status or None,
        "runtime_result_blockers": provider_output_validation.get("runtime_result_blockers")
        or output_inspection.get("runtime_result_blockers"),
        "last_nonterminal_output": last_nonterminal_output,
        "mp4_count": output_inspection.get("mp4_count"),
        "teardown_requested": effective_teardown_requested,
        "teardown_action": effective_teardown_action
        if effective_teardown_requested
        else "not_requested",
        "teardown_pending": teardown_pending,
        "teardown_performed": existing_teardown_completed,
        "requested_keep_running_on_success": requested_keep_running_on_success,
        "keep_running_on_success": keep_running_on_success,
        "keepalive_runtime_health": keepalive_runtime_health,
        "keepalive_runtime_unhealthy_on_success": keepalive_runtime_unhealthy_on_success,
        "continuing_spend_from_this_run": continuing_spend,
        "api_key_status": api_key_meta,
        "raw_secret_values_recorded": False,
    }
    if teardown_pending:
        write_json(
            resolved_job_dir / "runpod_wam_async_pre_teardown_poll_manifest.json",
            manifest,
        )
        stop_instead_of_delete = bool(
            not auto_teardown_failure
            and not runtime_stall_observed
            and (effective_teardown_action == "stop" or keepalive_runtime_unhealthy_on_success)
        )
        if stop_instead_of_delete:
            teardown_manifest = _stop_pod(
                job_dir=resolved_job_dir,
                pod_id=pod_id,
                api_key=api_key,
                generated_at=generated,
                record_warm_candidate=runtime_output_success,
            )
        else:
            teardown_manifest = _delete_pod(
                job_dir=resolved_job_dir,
                pod_id=pod_id,
                api_key=api_key,
                generated_at=generated,
            )
        teardown_completed = (teardown_manifest or {}).get("status") == "completed"
        continuing_spend = bool(
            pod_id
            and not teardown_completed
            and not output_present
            and (
                nonterminal_running_output
                or remote_runtime_running_without_terminal_output
                or not should_teardown
                or (teardown_manifest or {}).get("status") != "completed"
            )
            and not pod_status_is_terminal
        )
        manifest["teardown_performed"] = bool(
            teardown_manifest and teardown_completed
        )
        manifest["continuing_spend_from_this_run"] = continuing_spend
        manifest["status"] = (
            "blocked"
            if provider_output_validation_failed
            else "completed"
            if output_present
            else ("running" if continuing_spend and not runtime_stall_observed else "blocked")
        )
        manifest["teardown_manifest_path"] = str(
            resolved_job_dir
            / (
                "runpod_wam_async_stop_manifest.json"
                if stop_instead_of_delete
                else "runpod_wam_async_delete_manifest.json"
            )
        )
    elif keep_running_on_success:
        warm_candidate = _write_running_warm_candidate(
            job_dir=resolved_job_dir,
            pod_id=pod_id,
            generated_at=generated,
        )
        keepalive_manifest = {
            "schema_version": "runpod_wam_async_keepalive_manifest.v1",
            "generated_at": generated,
            "status": "completed",
            "job_dir": str(resolved_job_dir),
            "pod_id": pod_id,
            "teardown_action": "keep_on_success",
            "output_zip_present": output_present,
            "continuing_spend_from_this_run": True,
            "warm_candidate": warm_candidate,
            "warm_candidate_path": warm_candidate.get("path"),
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "runpod_wam_async_keepalive_manifest.json", keepalive_manifest)
        continuing_spend = True
        manifest["keepalive_performed"] = True
        manifest["continuing_spend_from_this_run"] = True
        manifest["warm_candidate"] = warm_candidate
        manifest["warm_candidate_path"] = warm_candidate.get("path")
        manifest["keepalive_manifest_path"] = str(
            resolved_job_dir / "runpod_wam_async_keepalive_manifest.json"
        )
    manifest["provider_reliability_manifest"] = _write_wam_provider_reliability_manifest(
        job_dir=resolved_job_dir,
        state=state,
        poll_manifest=manifest,
        teardown_manifest=teardown_manifest,
        generated_at=generated,
    )
    write_json(resolved_job_dir / "runpod_wam_async_poll_manifest.json", manifest)
    state_update = {
        **state,
        "last_polled_at": generated,
        "last_pod_status": pod_status,
        "provider_command_status": provider_status,
        "provider_command_blockers": provider_blockers,
        "continuing_spend_from_this_run": continuing_spend,
        "raw_secret_values_recorded": False,
    }
    write_json(_state_path(resolved_job_dir), state_update)
    return manifest


def stop_runpod_wam_async_run(
    *,
    job_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    state = _read_json(_state_path(resolved_job_dir))
    pod_id = _string(state.get("pod_id"))
    api_key, api_key_meta = _read_runpod_api_key()
    if not pod_id or not api_key:
        manifest = {
            "schema_version": RUNPOD_WAM_STOP_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "pod_id": pod_id,
            "blockers": [
                "runpod_wam_state_missing_pod_id"
                if not pod_id
                else f"missing_env_{RUNPOD_API_KEY_ENV}_or_{RUNPOD_API_KEY_FILE_ENV}"
            ],
            "api_key_status": api_key_meta,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "runpod_wam_async_stop_manifest.json", manifest)
        return manifest
    manifest = _stop_pod(
        job_dir=resolved_job_dir,
        pod_id=pod_id,
        api_key=api_key,
        generated_at=generated,
    )
    manifest["api_key_status"] = api_key_meta
    write_json(resolved_job_dir / "runpod_wam_async_stop_manifest.json", manifest)
    state_update = {
        **state,
        "last_polled_at": generated,
        "last_pod_status": "stop_requested",
        "continuing_spend_from_this_run": bool(
            manifest.get("continuing_spend_from_this_run")
        ),
        "raw_secret_values_recorded": False,
    }
    write_json(_state_path(resolved_job_dir), state_update)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--job-dir", required=True)
    create.add_argument("--bundle-path", required=True)
    create.add_argument("--public-base-url", default="")
    create.add_argument("--provider-bundle-url", default="")
    create.add_argument("--provider-output-put-url", default="")
    create.add_argument("--provider-output-get-url", default="")
    create.add_argument("--provider-bundle-url-file")
    create.add_argument("--provider-output-put-url-file")
    create.add_argument("--provider-output-get-url-file")
    create.add_argument("--token-file")
    create.add_argument("--secret-env-file")
    create.add_argument("--output-path")
    create.add_argument("--max-spend-usd", type=float)
    create.add_argument("--allow-paid-runpod-launch", action="store_true")
    create.add_argument("--skip-public-staging-verification", action="store_true")
    create.add_argument("--verify-output-put-url", action="store_true")
    create.add_argument("--gpu-type-id", action="append", default=[])
    create.add_argument("--image-name", default=DEFAULT_WAM_PUBLIC_IMAGE)
    create.add_argument("--provider-bundle-kind", choices=RUNPOD_PROVIDER_BUNDLE_KINDS, default="wam")
    create.add_argument("--container-disk-gb", type=int, default=80)
    create.add_argument("--volume-gb", type=int, default=20)
    create.add_argument("--cloud-type", choices=("SECURE", "COMMUNITY"), default="SECURE")
    create.add_argument("--allowed-cuda-version", action="append", default=[])
    create.add_argument("--min-vcpu-per-gpu", type=int, default=2)
    create.add_argument("--min-ram-per-gpu", type=int, default=8)
    poll = subparsers.add_parser("poll")
    poll.add_argument("--job-dir", required=True)
    poll.add_argument("--max-wait-seconds", type=int, default=60)
    poll.add_argument("--retry-interval-seconds", type=int, default=5)
    poll.add_argument("--post-marker-no-progress-timeout-seconds", type=int, default=None)
    poll.add_argument("--teardown", action="store_true")
    stop = subparsers.add_parser("stop")
    stop.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        print("legacy_runpod_wam_create_cli_disabled", file=sys.stderr)
        return 2
    if args.command == "poll":
        manifest = poll_runpod_wam_async_run(
            job_dir=args.job_dir,
            max_wait_seconds=args.max_wait_seconds,
            retry_interval_seconds=args.retry_interval_seconds,
            post_marker_no_progress_timeout_seconds=(
                args.post_marker_no_progress_timeout_seconds
            ),
            teardown=args.teardown,
        )
    else:
        manifest = stop_runpod_wam_async_run(job_dir=args.job_dir)
    print(json.dumps(manifest, sort_keys=True))
    return 0 if manifest.get("status") in {"pod_created", "running", "completed"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
