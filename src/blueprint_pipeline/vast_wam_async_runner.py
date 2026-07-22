"""Short-lived Vast WAM runner for OSCAR/Cosmos provider bundles.

The regular Vast WAM runner waits for the remote job in one local process. This
module splits the paid path into create and poll commands so Codex can resume or
teardown without relying on a long-lived local process.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from .common import ensure_dir, utc_now_iso, write_json
from .paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
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
from .vast_provider_adapter import (
    DEFAULT_HARD_CAP_USD,
    DEFAULT_MAX_HOURLY_RATE,
    DEFAULT_TARGET_SPEND_USD,
    DEFAULT_VAST_API_KEY_FILE,
    DEFAULT_WAM_ROLLOUT_VIDEO_COUNT,
    VAST_API_GATE_ENV as _VAST_API_GATE_ENV,
    VAST_MIN_RELIABILITY_ENV,
    VAST_PREFERRED_GEOLOCATION_REGEX_ENV,
    VAST_PREFERRED_GPU_KEYWORDS_ENV,
    VAST_INSTANCE_LAUNCH_GATE_ENV as _VAST_INSTANCE_LAUNCH_GATE_ENV,
    VAST_REQUIRE_DIRECT_PORT_ENV,
    VAST_API_KEY_FILE_ENV,
    VAST_FINAL_VALIDATION_SCHEMA_VERSION,
    VAST_GPU_SANITY_SCHEMA_VERSION,
    VAST_ISAAC_SMOKE_SCHEMA_VERSION,
    VAST_OFFER_SELECTION_SCHEMA_VERSION,
    VAST_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION,
    VAST_PROVIDER_COMMAND_SCHEMA_VERSION,
    VAST_STARTUP_PROBE_SCHEMA_VERSION,
    VAST_TEARDOWN_SCHEMA_VERSION,
    VAST_VIDEO_SMOKE_SCHEMA_VERSION,
    _api_gate_blockers,
    _api_json,
    _append_phase,
    _append_session_budget_attempt,
    _blueprint_bundle_preflight,
    _budget_ledger,
    _create_payload,
    _create_request_summary,
    _env_int,
    _env_float,
    _env_csv,
    _env_truthy,
    _fill_missing_phase_rows,
    _final_validation,
    _forwarded_secret_values,
    _inline_provider_bundle_payload,
    _instance_id_from_create_response,
    _instance_status,
    _isaac_image_startup_preflight,
    _mapping,
    _number,
    _offer_summary,
    _offers_from_response,
    _offer_artifact_summary,
    _poll_instance,
    _prelaunch_inventory_guard,
    _probe_env,
    _probe_shell_script,
    _provider_plan,
    _read_secret_file,
    _redact_runtime_value,
    _request_logs_and_fetch,
    _resolve_disk_gb,
    _resolve_image_login,
    _resolve_launch_mode,
    _resolve_probe_image,
    _runtime_discovery,
    _search_payload,
    _select_offer,
    _session_budget_guard,
    _string,
    _string_list,
    _try_acquire_vast_launch_lock,
    _url_secret_values,
    _vast_launch_lock_path,
    _vast_session_budget_ledger_path,
    _release_vast_launch_lock,
    _inspect_provider_runtime_output_zip,
)
from .vast_wam_authorized_runner import DEFAULT_WAM_PUBLIC_IMAGE, DEFAULT_WAM_VAST_LAUNCH_MODE
from .wam_async_runner_common import (
    download_url_to_file,
    read_json_mapping as _read_json,
    read_sensitive_url_file as _read_sensitive_url_file,
    redact_provider_url as _redact_provider_url,
)


ASYNC_STATE_SCHEMA_VERSION = "vast_wam_async_state.v1"
ASYNC_CREATE_SCHEMA_VERSION = "vast_wam_async_create_manifest.v1"
ASYNC_POLL_SCHEMA_VERSION = "vast_wam_async_poll_manifest.v1"
DEFAULT_DISK_GB = 80
DEFAULT_HEARTBEAT_URL = "https://example.com/"
# A missing container ("No such container") is normal during the image-pull/boot window,
# but a container that never appears is a dud offer that would otherwise idle for the entire
# live window before teardown. Cap the tolerance so a dud is detected and torn down quickly,
# letting the caller re-fire on a fresh offer. Default 12 minutes (override via env).
VAST_WAM_CONTAINER_MISSING_MAX_SECONDS_ENV = "BLUEPRINT_VAST_WAM_CONTAINER_MISSING_MAX_SECONDS"
DEFAULT_VAST_WAM_CONTAINER_MISSING_MAX_SECONDS = 720
VAST_API_GATE_ENV = _VAST_API_GATE_ENV
VAST_INSTANCE_LAUNCH_GATE_ENV = _VAST_INSTANCE_LAUNCH_GATE_ENV
DEFAULT_WAM_PREFERRED_GPU_KEYWORDS = (
    "RTX 6000",
    "RTX A6000",
    "A6000",
    "A40",
    "L40",
    "L40S",
    "A100",
    "H100",
    "H200",
)


def _state_path(job_dir: Path) -> Path:
    return job_dir / "vast_wam_async_state.json"


def _deadline_capped_log_wait_seconds(
    *,
    state: Mapping[str, Any],
    requested_max_wait_seconds: int,
    now_epoch: float,
) -> tuple[int, float | None, bool]:
    requested = max(0, int(requested_max_wait_seconds))
    deadline_epoch = float(_number(state.get("max_live_deadline_epoch")) or 0.0)
    if deadline_epoch <= 0.0:
        return requested, None, False
    seconds_until_deadline = deadline_epoch - now_epoch
    capped = min(requested, max(0, int(seconds_until_deadline)))
    return capped, seconds_until_deadline, capped < requested


def _url_file_path_from_meta(meta: Any) -> str:
    return _string(_mapping(meta).get("path"))


def _download_provider_output_zip(
    *,
    job_dir: Path,
    output_zip_path: Path,
    provider_output_get_url: str,
    provider_upload_marker_seen: bool,
    generated_at: str,
) -> dict[str, Any]:
    ensure_dir(output_zip_path.parent)
    manifest: dict[str, Any] = {
        "schema_version": "vast_provider_output_download.v1",
        "generated_at": generated_at,
        "status": "not_requested",
        "provider_output_get_url_present": bool(_string(provider_output_get_url)),
        "provider_upload_marker_seen": bool(provider_upload_marker_seen),
        "output_zip_path": str(output_zip_path),
        "raw_secret_values_recorded": False,
    }
    if provider_upload_marker_seen and _string(provider_output_get_url) and not output_zip_path.is_file():
        transfer = download_url_to_file(
            url=_string(provider_output_get_url),
            output_path=output_zip_path,
            user_agent="BlueprintVastAsyncWamRunner/1.0",
            timeout_seconds=90,
        )
        if transfer["status"] == "completed":
            manifest.update(
                {
                    "status": "completed",
                    "http_status_code": transfer.get("http_status_code"),
                    "output_zip_present_after_download": output_zip_path.is_file(),
                    "output_zip_size_bytes": transfer.get("downloaded_size_bytes", 0),
                }
            )
        else:
            manifest.update(
                {
                    "status": "blocked",
                    "error_type": transfer.get("error_type"),
                    "http_status_code": transfer.get("http_status_code"),
                    "blockers": ["provider_output_get_url_download_failed"],
                }
            )
    elif provider_upload_marker_seen and output_zip_path.is_file():
        manifest.update(
            {
                "status": "skipped",
                "reason": "provider_runtime_output_zip_already_present",
                "output_zip_present_after_download": True,
                "output_zip_size_bytes": output_zip_path.stat().st_size,
            }
        )
    elif provider_upload_marker_seen and not _string(provider_output_get_url):
        manifest.update(
            {
                "status": "blocked",
                "blockers": ["provider_output_get_url_missing"],
            }
        )
    write_json(job_dir / "vast_provider_output_download_manifest.json", manifest)
    return manifest


def _regex_field(text: str, field: str) -> str:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]*)"', text)
    return match.group(1) if match else ""


def _regex_number(text: str, field: str) -> float | None:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    return float(match.group(1)) if match else None


def _read_async_state(job_dir: Path) -> dict[str, Any]:
    path = _state_path(job_dir)
    try:
        return _read_json(path)
    except Exception as exc:
        state_text = path.read_text(encoding="utf-8") if path.is_file() else ""
        create_manifest = {}
        try:
            create_manifest = _read_json(job_dir / "vast_wam_async_create_manifest.json")
        except Exception:
            create_manifest = {}
        instance_id = (
            _number(create_manifest.get("instance_id"))
            or _regex_number(state_text, "instance_id")
        )
        selected_offer = _mapping(create_manifest.get("selected_offer"))
        recovered = {
            "schema_version": ASYNC_STATE_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "recovered_from_malformed_state",
            "state_parse_error": f"{type(exc).__name__}:{str(exc)[:200]}",
            "job_dir": str(job_dir),
            "bundle_path": _regex_field(state_text, "bundle_path"),
            "output_path": _string(create_manifest.get("output_path"))
            or _regex_field(state_text, "output_path"),
            "public_base_url": _regex_field(state_text, "public_base_url"),
            "token_file": _regex_field(state_text, "token_file"),
            "secret_env_file": _regex_field(state_text, "secret_env_file"),
            "session_budget_ledger": _regex_field(state_text, "session_budget_ledger"),
            "instance_id": int(instance_id) if instance_id is not None else None,
            "created_at_epoch": _regex_number(state_text, "created_at_epoch") or time.time(),
            "max_live_minutes": int(_regex_number(state_text, "max_live_minutes") or 0),
            "max_live_deadline_epoch": _regex_number(state_text, "max_live_deadline_epoch")
            or time.time(),
            "selected_offer": _offer_artifact_summary(selected_offer),
            "selected_hourly_rate_usd": _number(selected_offer.get("hourly_rate_usd"))
            or _regex_number(state_text, "selected_hourly_rate_usd"),
            "target_spend_usd": _regex_number(state_text, "target_spend_usd"),
            "hard_cap_usd": _regex_number(state_text, "hard_cap_usd"),
            "max_hourly_rate_usd": _regex_number(state_text, "max_hourly_rate_usd"),
            "raw_credentials_recorded": False,
        }
        write_json(
            job_dir / "vast_wam_async_state_recovery_manifest.json",
            {
                "schema_version": "vast_wam_async_state_recovery_manifest.v1",
                "generated_at": utc_now_iso(),
                "status": "completed",
                "state_path": str(path),
                "parse_error": recovered["state_parse_error"],
                "recovered_instance_id": recovered.get("instance_id"),
                "recovered_output_path_present": bool(recovered.get("output_path")),
                "raw_credentials_recorded": False,
            },
        )
        return recovered


def _write_blocked_result(
    job_dir: Path,
    *,
    generated_at: str,
    reason: str,
    blockers: Sequence[str],
) -> dict[str, Any]:
    result = {
        "schema_version": VAST_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_dir": str(job_dir),
        "status": "blocked",
        "reason": reason,
        "blockers": list(blockers),
        "api_call_performed": False,
        "vast_side_effects_may_have_occurred": False,
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "vast_provider_adapter_result.json", result)
    write_json(
        job_dir / "vast_teardown_manifest.json",
        {
            "schema_version": VAST_TEARDOWN_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "not_required_prelaunch_blocked",
            "vast_instance_ids": [],
            "teardown_actions_performed": [],
            "continuing_spend_from_this_run": False,
            "raw_credentials_recorded": False,
        },
    )
    _fill_missing_phase_rows(job_dir, reason=reason)
    write_json(
        job_dir / "vast_final_validation.json",
        {
            "schema_version": VAST_FINAL_VALIDATION_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "job_dir": str(job_dir),
            "blockers": list(blockers),
            "continuing_spend_from_this_run": False,
            "raw_secret_values_recorded": False,
        },
    )
    return result


def _destroy_vast_instance_with_retry(
    *,
    instance_id: int,
    api_key: str,
    attempts: int = 3,
    backoff_seconds: float = 3.0,
) -> tuple[bool, list[dict[str, Any]]]:
    """Destroy a Vast instance, retrying transient failures.

    A single failed DELETE must never leave an instance billing. In practice a destroy can be
    rejected transiently — e.g. the instance is still ``loading`` (observed: a dud offer whose
    destroy failed once and kept billing until a manual retry), or a momentary network/API
    error. Retries up to ``attempts`` times with linear backoff; a 404 means the instance is
    already gone (success). Returns ``(continuing_spend, teardown_actions)`` where
    ``continuing_spend`` is True only if *every* attempt failed.
    """
    teardown_actions: list[dict[str, Any]] = []
    total = max(1, int(attempts))
    for attempt in range(1, total + 1):
        try:
            delete_status, delete_response = _api_json(
                method="DELETE",
                path=f"/instances/{instance_id}/",
                api_key=api_key,
                timeout_seconds=30,
            )
            teardown_actions.append(
                {
                    "instance_id": instance_id,
                    "action": "destroy_instance",
                    "attempt": attempt,
                    "http_status_code": delete_status,
                    "response": _redact_runtime_value(delete_response, [api_key]),
                    "status": "completed",
                }
            )
            return False, teardown_actions
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                teardown_actions.append(
                    {
                        "instance_id": instance_id,
                        "action": "destroy_instance",
                        "attempt": attempt,
                        "http_status_code": exc.code,
                        "status": "completed",
                        "reason": "instance_already_absent",
                    }
                )
                return False, teardown_actions
            teardown_actions.append(
                {
                    "instance_id": instance_id,
                    "action": "destroy_instance",
                    "attempt": attempt,
                    "http_status_code": exc.code,
                    "status": "failed",
                }
            )
        except Exception as exc:
            teardown_actions.append(
                {
                    "instance_id": instance_id,
                    "action": "destroy_instance",
                    "attempt": attempt,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
            )
        if attempt < total:
            time.sleep(min(15.0, backoff_seconds * attempt))
    return True, teardown_actions


def destroy_async_vast_wam_run(
    *,
    job_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    ensure_dir(resolved_job_dir)
    state = _read_async_state(resolved_job_dir)
    instance_id = int(_number(state.get("instance_id")) or 0)
    api_key, vast_secret_status = _read_secret_file(
        VAST_API_KEY_FILE_ENV,
        DEFAULT_VAST_API_KEY_FILE,
    )
    blockers = _api_gate_blockers(
        allow_vast_api_call=True,
        allow_instance_launch=True,
        api_key=api_key,
    )
    if instance_id <= 0:
        blockers.append("missing_async_vast_instance_id")
    teardown_actions: list[dict[str, Any]] = []
    continuing_spend = bool(instance_id)
    if not blockers:
        _append_phase(
            resolved_job_dir,
            "vast_instance_teardown_started",
            "running",
            instance_ids=[instance_id],
        )
        continuing_spend, destroy_actions = _destroy_vast_instance_with_retry(
            instance_id=instance_id,
            api_key=api_key,
        )
        teardown_actions.extend(destroy_actions)
        _append_phase(
            resolved_job_dir,
            "vast_instance_teardown_completed",
            "completed" if not continuing_spend else "blocked",
            blockers=[] if not continuing_spend else ["vast_instance_destroy_failed"],
            instance_ids=[instance_id],
        )
    now_epoch = time.time()
    elapsed_seconds = max(
        0.0,
        now_epoch - float(_number(state.get("created_at_epoch")) or now_epoch),
    )
    hourly_rate = float(
        _number(state.get("selected_hourly_rate_usd"))
        or _number(_mapping(state.get("selected_offer")).get("hourly_rate_usd"))
        or 0.0
    )
    estimated_cost_usd = round(hourly_rate * elapsed_seconds / 3600.0, 6)
    status = "completed" if not blockers and not continuing_spend else "blocked"
    manifest = {
        "schema_version": VAST_TEARDOWN_SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "vast_instance_ids": [instance_id] if instance_id else [],
        "teardown_actions_performed": teardown_actions,
        "runner_gpu_teardown_completed": not continuing_spend,
        "continuing_spend_from_this_run": continuing_spend,
        "estimated_cost_usd": estimated_cost_usd,
        "vast_secret_status": vast_secret_status,
        "blockers": blockers if blockers else ([] if not continuing_spend else ["vast_instance_destroy_failed"]),
        "zero_continuing_spend_scope": "direct async destroy deleted instance or it was already absent"
        if not continuing_spend
        else "teardown failure requires manual Vast console/API verification",
        "raw_secret_values_recorded": False,
    }
    write_json(resolved_job_dir / "vast_teardown_manifest.json", manifest)
    if not continuing_spend:
        state["status"] = "teardown_completed"
        state["destroyed_at_epoch"] = now_epoch
        state["continuing_spend_from_this_run"] = False
        write_json(_state_path(resolved_job_dir), state)
    return manifest


def _provider_urls(public_base_url: str, token_file: Path) -> tuple[str, str, dict[str, Any]]:
    token, token_status = _read_or_create_token(token_file)
    return (
        _url_with_token(public_base_url, BUNDLE_ROUTE, token),
        _url_with_token(public_base_url, OUTPUT_ROUTE, token),
        token_status,
    )


def create_async_vast_wam_run(
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
    session_budget_ledger: str | Path | None = None,
    allow_paid_vast_launch: bool = False,
    max_hourly_rate: float = DEFAULT_MAX_HOURLY_RATE,
    target_spend_usd: float = DEFAULT_TARGET_SPEND_USD,
    hard_cap_usd: float = DEFAULT_HARD_CAP_USD,
    allow_target_spend_overrun: bool = False,
    max_live_minutes: int = 30,
    session_max_live_minutes: int | None = 45,
    min_gpu_ram_mb: int = 0,
    excluded_machine_ids: Sequence[int] = (),
    allowed_machine_ids: Sequence[int] = (),
    min_reliability: float | None = None,
    require_direct_port: bool | None = None,
    preferred_gpu_keywords: Sequence[str] = (),
    preferred_geolocation_regex: str = "",
    prefer_isaac_rt: bool = False,
    startup_poll_seconds: int = 90,
    public_staging_verify_max_wait_seconds: int = 120,
    public_staging_verify_retry_interval_seconds: float = 5.0,
    public_staging_verify_timeout_seconds: float = 20.0,
    public_staging_required_consecutive_successes: int = 2,
    verify_output_put_url: bool = False,
    public_image: str = DEFAULT_WAM_PUBLIC_IMAGE,
    vast_launch_mode: str = DEFAULT_WAM_VAST_LAUNCH_MODE,
    disk_gb: int = DEFAULT_DISK_GB,
    heartbeat_url: str = DEFAULT_HEARTBEAT_URL,
    generated_at: str | None = None,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    created_epoch = time.time()
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
    resolved_session_budget_ledger = (
        Path(session_budget_ledger).expanduser().resolve()
        if session_budget_ledger
        else _vast_session_budget_ledger_path()
    )
    ensure_dir(resolved_job_dir)
    launch_mode = _resolve_launch_mode(
        requested=vast_launch_mode,
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        provider_bundle_kind="wam",
    )
    resolved_disk_gb = _resolve_disk_gb(requested=disk_gb, enable_isaac_smoke=False)
    resolved_min_reliability = max(
        0.0,
        float(
            min_reliability
            if min_reliability is not None
            else _env_float(VAST_MIN_RELIABILITY_ENV, 0.0)
        ),
    )
    resolved_require_direct_port = bool(
        require_direct_port
        if require_direct_port is not None
        else _env_truthy(VAST_REQUIRE_DIRECT_PORT_ENV)
    )
    resolved_preferred_gpu_keywords = [
        _string(item) for item in preferred_gpu_keywords if _string(item)
    ] or _env_csv(VAST_PREFERRED_GPU_KEYWORDS_ENV, DEFAULT_WAM_PREFERRED_GPU_KEYWORDS)
    resolved_preferred_geolocation_regex = _string(
        preferred_geolocation_regex or os.getenv(VAST_PREFERRED_GEOLOCATION_REGEX_ENV)
    )
    selected_container_image = _resolve_probe_image(
        public_image=public_image,
        isaac_image="",
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        provider_bundle_kind="wam",
    )
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
        "path": str(resolved_token_file),
        "token_recorded_in_manifest": False,
        "reason": "not_required_for_explicit_provider_urls"
        if direct_provider_urls
        else "pending_staging_token_resolution",
    }
    if direct_provider_urls:
        provider_bundle_url = _string(provider_bundle_url)
        provider_output_put_url = _string(provider_output_put_url)
        provider_output_get_url = _string(provider_output_get_url)
    else:
        provider_bundle_url, provider_output_put_url, token_status = _provider_urls(
            public_base_url,
            resolved_token_file,
        )
    secret_values_for_urls = _url_secret_values(
        provider_bundle_url,
        provider_output_put_url,
        provider_output_get_url,
    )
    inline_bundle_transport = (
        {
            "inline_provider_bundle_transport_used": False,
            "inline_provider_bundle_transport_reason": "disabled_for_explicit_provider_urls",
            "inline_provider_bundle_size_bytes": resolved_bundle.stat().st_size
            if resolved_bundle.is_file()
            else 0,
            "inline_provider_bundle_base64_length": 0,
            "inline_provider_bundle_sha256_present": False,
            "raw_secret_values_recorded": False,
        }
        if direct_provider_urls
        else _inline_provider_bundle_payload(
            resolved_bundle,
            provider_bundle_kind="wam",
            enable_blueprint_bundle=True,
        )
    )

    if direct_provider_urls:
        staging_manifest = {
            "schema_version": "vast_wam_direct_provider_urls.v1",
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
        write_json(
            resolved_job_dir / "vast_wam_direct_provider_urls_manifest.json",
            staging_manifest,
        )
        self_test = {
            "schema_version": "vast_wam_local_staging_self_test.v1",
            "generated_at": generated,
            "status": "skipped",
            "reason": "explicit_provider_urls_supplied",
            "raw_secret_values_recorded": False,
        }
    else:
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
            output_path=resolved_job_dir / "vast_wam_async_staging_self_test_output.zip",
            token_file=resolved_token_file,
            generated_at=generated,
        )
    public_verification = verify_public_staging_urls(
        job_dir=resolved_job_dir,
        provider_bundle_url=provider_bundle_url,
        provider_output_put_url=provider_output_put_url,
        bundle_path=resolved_bundle,
        output_path=resolved_output,
        max_wait_seconds=public_staging_verify_max_wait_seconds,
        retry_interval_seconds=public_staging_verify_retry_interval_seconds,
        timeout_seconds=public_staging_verify_timeout_seconds,
        required_consecutive_successes=1
        if direct_provider_urls
        else public_staging_required_consecutive_successes,
        allow_output_put_probe=verify_output_put_url or not direct_provider_urls,
        cleanup_output_probe=not direct_provider_urls,
        require_bundle_fetch_probe=(
            not direct_provider_urls
            and inline_bundle_transport.get("inline_provider_bundle_transport_used") is not True
        ),
        generated_at=generated,
    )

    blockers: list[str] = []
    if staging_manifest.get("status") != "ready":
        blockers.extend(_string_list(staging_manifest.get("blockers")))
    if not direct_provider_urls and self_test.get("status") != "passed":
        blockers.append("local_staging_self_test_failed")
    if public_verification.get("status") != "passed":
        blockers.extend(
            _string_list(public_verification.get("blockers"))
            or ["public_staging_url_stability_not_proven"]
        )
    if not allow_paid_vast_launch:
        blockers.append("paid_vast_launch_not_authorized_by_runner_flag")
    if direct_provider_urls:
        bundle_scheme = urlparse(provider_bundle_url).scheme
        output_scheme = urlparse(provider_output_put_url).scheme
        if bundle_scheme not in {"http", "https"}:
            blockers.append("vast_provider_bundle_url_scheme_not_http")
        if output_scheme not in {"http", "https"}:
            blockers.append("vast_provider_output_put_url_scheme_not_http")
        output_get_scheme = urlparse(provider_output_get_url).scheme if provider_output_get_url else ""
        if output_get_scheme and output_get_scheme not in {"http", "https"}:
            blockers.append("vast_provider_output_get_url_scheme_not_http")
    elif not _string(public_base_url):
        blockers.append("vast_public_base_url_or_explicit_provider_urls_required")

    api_key, vast_secret_status = _read_secret_file(VAST_API_KEY_FILE_ENV, "~/.blueprint-secrets/vast_api_key")
    gate_blockers = _api_gate_blockers(
        allow_vast_api_call=allow_paid_vast_launch,
        allow_instance_launch=allow_paid_vast_launch,
        api_key=api_key,
    )
    blockers.extend(gate_blockers)

    _runtime_discovery(
        resolved_job_dir,
        generated_at=generated,
        launch_mode=launch_mode,
        disk_gb=resolved_disk_gb,
    )
    _append_phase(
        resolved_job_dir,
        "vast_docs_checked",
        "completed",
        proof_effect="vast_api_docs_and_launch_modes_recorded",
    )
    _append_phase(
        resolved_job_dir,
        "vast_secret_file_verified",
        "completed"
        if vast_secret_status["present"] and vast_secret_status.get("mode_is_0600")
        else "blocked",
        blockers=[] if vast_secret_status["present"] else [f"missing_file_based_secret_{VAST_API_KEY_FILE_ENV}"],
        proof_effect="secret_file_metadata_only",
    )
    _provider_plan(
        job_dir=resolved_job_dir,
        generated_at=generated,
        max_hourly_rate=max_hourly_rate,
        target_spend_usd=target_spend_usd,
        hard_cap_usd=hard_cap_usd,
        max_live_minutes=max_live_minutes,
        public_image=public_image,
        isaac_image="",
        selected_container_image=selected_container_image,
        previous_job_dir=None,
        provider_bundle=resolved_bundle,
        provider_bundle_kind="wam",
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        launch_mode=launch_mode,
        disk_gb=resolved_disk_gb,
        ngc_image_login_mode="never",
        vast_template_hash_id=None,
        use_vast_template_image=False,
        allow_cold_isaac_image_pull=False,
        min_cold_isaac_pull_live_minutes=0,
        provider_bundle_url=provider_bundle_url,
        provider_output_put_url=provider_output_put_url,
        provider_bundle_inline_transport=inline_bundle_transport,
    )
    bundle_preflight = _blueprint_bundle_preflight(
        job_dir=resolved_job_dir,
        generated_at=generated,
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="wam",
        bundle_path=resolved_bundle,
        provider_bundle_url=provider_bundle_url,
        provider_output_put_url=provider_output_put_url,
        verify_staging_urls=False,
        allow_staging_output_put_probe=False,
    )
    image_preflight = _isaac_image_startup_preflight(
        job_dir=resolved_job_dir,
        generated_at=generated,
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        provider_bundle_kind="wam",
        selected_container_image=selected_container_image,
        vast_template_hash_id=None,
        use_vast_template_image=False,
        max_live_minutes=max_live_minutes,
        allow_cold_isaac_image_pull=False,
        min_cold_isaac_pull_live_minutes=0,
    )
    blockers.extend(_string_list(bundle_preflight.get("blockers")))
    blockers.extend(_string_list(image_preflight.get("blockers")))
    session_guard = _session_budget_guard(
        job_dir=resolved_job_dir,
        generated_at=generated,
        budget_path=resolved_session_budget_ledger,
        session_max_live_minutes=session_max_live_minutes,
        requested_max_live_minutes=max_live_minutes,
        target_spend_usd=target_spend_usd,
        hard_cap_usd=hard_cap_usd,
        max_hourly_rate=max_hourly_rate,
    )
    blockers.extend(_string_list(session_guard.get("blockers")))
    if not allow_target_spend_overrun and "requested_max_spend_would_exceed_target" in _string_list(session_guard.get("warnings")):
        blockers.append("requested_max_spend_would_exceed_target")

    _budget_ledger(
        job_dir=resolved_job_dir,
        generated_at=generated,
        target_spend_usd=target_spend_usd,
        hard_cap_usd=hard_cap_usd,
        max_hourly_rate=max_hourly_rate,
        max_live_minutes=max_live_minutes,
        selected_offer=None,
    )
    if blockers:
        manifest = {
            "schema_version": ASYNC_CREATE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "blockers": sorted(set(blockers)),
            "allow_paid_vast_launch": allow_paid_vast_launch,
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
            "provider_bundle_inline_transport_used": (
                inline_bundle_transport.get("inline_provider_bundle_transport_used") is True
            ),
            "provider_bundle_inline_transport_reason": _string(
                inline_bundle_transport.get("inline_provider_bundle_transport_reason")
            ),
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_wam_async_create_manifest.json", manifest)
        _write_blocked_result(
            resolved_job_dir,
            generated_at=generated,
            reason="vast_wam_async_create_preflight_blocked",
            blockers=sorted(set(blockers)),
        )
        return manifest

    try:
        require_paid_resource_admission_grant(
            paid_resource_admission_grant,
            resource_class="vast_wam_async",
        )
    except PaidResourceAdmissionBlocked as exc:
        manifest = {
            "schema_version": ASYNC_CREATE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "blockers": [
                "vast_wam_shared_admission_missing_or_invalid",
                *exc.blockers,
            ],
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_wam_async_create_manifest.json", manifest)
        return manifest

    lock_handle, lock_manifest = _try_acquire_vast_launch_lock(
        job_dir=resolved_job_dir,
        generated_at=generated,
        lock_path=_vast_launch_lock_path(),
    )
    if lock_handle is None:
        lock_blockers = _string_list(lock_manifest.get("blockers")) or ["vast_paid_launch_lock_busy"]
        manifest = {
            "schema_version": ASYNC_CREATE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "blockers": lock_blockers,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_wam_async_create_manifest.json", manifest)
        _write_blocked_result(
            resolved_job_dir,
            generated_at=generated,
            reason="vast_paid_launch_lock_blocked",
            blockers=lock_blockers,
        )
        return manifest

    instance_id: int | None = None
    selected_offer: dict[str, Any] | None = None
    try:
        inventory_guard = _prelaunch_inventory_guard(
            job_dir=resolved_job_dir,
            generated_at=generated,
            api_key=api_key,
        )
        inventory_blockers = _string_list(inventory_guard.get("blockers"))
        if inventory_blockers:
            return _write_blocked_result(
                resolved_job_dir,
                generated_at=generated,
                reason="vast_prelaunch_inventory_guard_blocked",
                blockers=inventory_blockers,
            )

        _append_phase(resolved_job_dir, "vast_offer_search_started", "running")
        search_request = _search_payload(limit=100, max_hourly_rate=max_hourly_rate)
        search_status, search_response = _api_json(
            method="POST",
            path="/bundles/",
            api_key=api_key,
            payload=search_request,
            timeout_seconds=45,
        )
        offers = _offers_from_response(search_response)
        selected_offer = _select_offer(
            offers,
            max_hourly_rate=max_hourly_rate,
            min_gpu_ram_mb=min_gpu_ram_mb,
            excluded_machine_ids=excluded_machine_ids,
            allowed_machine_ids=allowed_machine_ids,
            require_known_supported_isaac_driver=False,
            min_reliability=resolved_min_reliability,
            require_direct_port=resolved_require_direct_port,
            preferred_gpu_keywords=resolved_preferred_gpu_keywords,
            preferred_geolocation_regex=resolved_preferred_geolocation_regex,
            prefer_isaac_rt=prefer_isaac_rt,
        )
        offer_blockers: list[str] = (
            []
            if selected_offer
            else ["no_vast_offer_matching_rate_and_gpu_memory_constraints"]
        )
        offer_manifest = {
            "schema_version": VAST_OFFER_SELECTION_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "selected" if selected_offer else "blocked",
            "offer_search_performed": True,
            "http_status_code": search_status,
            "offer_count": len(offers),
            "max_hourly_rate_usd": max_hourly_rate,
            "min_gpu_ram_mb": int(min_gpu_ram_mb),
            "min_reliability": resolved_min_reliability,
            "require_direct_port": resolved_require_direct_port,
            "preferred_gpu_keywords": list(resolved_preferred_gpu_keywords),
            "preferred_geolocation_regex": resolved_preferred_geolocation_regex,
            "prefer_isaac_rt": prefer_isaac_rt,
            "excluded_machine_ids": list(excluded_machine_ids),
            "allowed_machine_ids": list(allowed_machine_ids),
            "selected_offer": _offer_artifact_summary(selected_offer),
            "considered_offers": [
                _offer_artifact_summary(_offer_summary(offer)) for offer in offers
            ][:25],
            "blockers": offer_blockers,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_offer_selection_manifest.json", offer_manifest)
        _append_phase(
            resolved_job_dir,
            "vast_offer_selected",
            "completed" if selected_offer else "blocked",
            blockers=offer_blockers,
            proof_effect="vast_offer_selected_under_budget" if selected_offer else "none",
        )
        if not selected_offer:
            return _write_blocked_result(
                resolved_job_dir,
                generated_at=generated,
                reason="no_vast_offer_selected",
                blockers=offer_blockers,
            )
        projected_full_cost = float(selected_offer["hourly_rate_usd"]) * max_live_minutes / 60.0
        if projected_full_cost > hard_cap_usd:
            return _write_blocked_result(
                resolved_job_dir,
                generated_at=generated,
                reason="selected_offer_projected_max_runtime_exceeds_hard_cap",
                blockers=["selected_offer_projected_max_runtime_exceeds_hard_cap"],
            )
        image_login, image_login_summary = _resolve_image_login(
            image=selected_container_image,
            ngc_key="",
            mode="never",
        )
        create_payload = _create_payload(
            image=selected_container_image,
            label=f"blueprint-vast-wam-async-{int(time.time())}",
            launch_mode=launch_mode,
            probe_script=_probe_shell_script(
                heartbeat_url,
                enable_isaac_smoke=False,
                enable_blueprint_bundle=True,
                provider_bundle_kind="wam",
            ),
            disk_gb=resolved_disk_gb,
            env=_probe_env(
                job_dir=resolved_job_dir,
                enable_isaac_smoke=False,
                provider_bundle_url=provider_bundle_url,
                provider_output_put_url=provider_output_put_url,
                provider_bundle_inline_base64=_string(
                    inline_bundle_transport.get("inline_provider_bundle_base64")
                ),
                provider_bundle_inline_sha256=_string(
                    inline_bundle_transport.get("inline_provider_bundle_sha256")
                ),
            ),
            image_login=image_login,
            template_hash_id=None,
        )
        secret_values = [
            api_key,
            *_forwarded_secret_values(),
            *secret_values_for_urls,
            _string(inline_bundle_transport.get("inline_provider_bundle_base64")),
        ]
        _append_phase(
            resolved_job_dir,
            "vast_instance_create_requested",
            "running",
            offer_id=selected_offer["ask_contract_id"],
        )
        try:
            create_status, create_response = _api_json(
                method="PUT",
                path=f"/asks/{selected_offer['ask_contract_id']}/",
                api_key=api_key,
                payload=create_payload,
                timeout_seconds=45,
            )
        except urllib.error.HTTPError as exc:
            blockers = [f"vast_create_instance_http_error:{exc.code}"]
            manifest = {
                "schema_version": ASYNC_CREATE_SCHEMA_VERSION,
                "generated_at": generated,
                "status": "blocked",
                "job_dir": str(resolved_job_dir),
                "blockers": blockers,
                "selected_offer": _offer_artifact_summary(selected_offer),
                "create_http_status_code": exc.code,
                "explicit_provider_urls_used": direct_provider_urls,
                "provider_bundle_url_redacted": _redact_provider_url(provider_bundle_url),
                "provider_output_put_url_redacted": _redact_provider_url(provider_output_put_url),
                "provider_output_get_url_redacted": _redact_provider_url(provider_output_get_url),
                "raw_secret_values_recorded": False,
            }
            write_json(resolved_job_dir / "vast_wam_async_create_manifest.json", manifest)
            _append_phase(
                resolved_job_dir,
                "vast_instance_create_requested",
                "blocked",
                blockers=blockers,
                offer_id=selected_offer["ask_contract_id"],
                http_status_code=exc.code,
            )
            _write_blocked_result(
                resolved_job_dir,
                generated_at=generated,
                reason="vast_create_instance_api_error",
                blockers=blockers,
            )
            return manifest
        except Exception as exc:
            blockers = [f"vast_create_instance_api_exception:{type(exc).__name__}"]
            manifest = {
                "schema_version": ASYNC_CREATE_SCHEMA_VERSION,
                "generated_at": generated,
                "status": "blocked",
                "job_dir": str(resolved_job_dir),
                "blockers": blockers,
                "selected_offer": _offer_artifact_summary(selected_offer),
                "explicit_provider_urls_used": direct_provider_urls,
                "provider_bundle_url_redacted": _redact_provider_url(provider_bundle_url),
                "provider_output_put_url_redacted": _redact_provider_url(provider_output_put_url),
                "provider_output_get_url_redacted": _redact_provider_url(provider_output_get_url),
                "raw_secret_values_recorded": False,
            }
            write_json(resolved_job_dir / "vast_wam_async_create_manifest.json", manifest)
            _append_phase(
                resolved_job_dir,
                "vast_instance_create_requested",
                "blocked",
                blockers=blockers,
                offer_id=selected_offer["ask_contract_id"],
            )
            _write_blocked_result(
                resolved_job_dir,
                generated_at=generated,
                reason="vast_create_instance_api_error",
                blockers=blockers,
            )
            return manifest
        instance_id = _instance_id_from_create_response(create_response)
        if not instance_id:
            return _write_blocked_result(
                resolved_job_dir,
                generated_at=generated,
                reason="vast_create_response_missing_instance_id",
                blockers=["vast_create_response_missing_instance_id"],
            )
        _append_phase(
            resolved_job_dir,
            "vast_instance_create_requested",
            "completed",
            proof_effect="vast_instance_create_api_returned_instance_id",
            instance_id=instance_id,
            http_status_code=create_status,
        )
        status, observations, instance_payload = _poll_instance(
            instance_id=instance_id,
            api_key=api_key,
            timeout_seconds=max(0, startup_poll_seconds),
            poll_interval_seconds=10,
        )
        log_readable = status.lower() in {"running", "exited", "stopped"}
        _append_phase(
            resolved_job_dir,
            "vast_instance_started_or_blocked",
            "completed" if log_readable else "blocked",
            blockers=[] if log_readable else [f"vast_instance_status:{status}"],
            instance_id=instance_id,
            proof_effect="vast_instance_reached_running_or_log_readable_state"
            if log_readable
            else "none",
        )
        state = {
            "schema_version": ASYNC_STATE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "instance_created",
            "job_dir": str(resolved_job_dir),
            "bundle_path": str(resolved_bundle),
            "output_path": str(resolved_output),
            "public_base_url": public_base_url,
            "explicit_provider_urls_used": direct_provider_urls,
            "provider_bundle_url_redacted": _redact_provider_url(provider_bundle_url),
            "provider_output_put_url_redacted": _redact_provider_url(provider_output_put_url),
            "provider_output_get_url_redacted": _redact_provider_url(provider_output_get_url),
            "provider_bundle_url_file": bundle_url_file_meta,
            "provider_output_put_url_file": output_url_file_meta,
            "provider_output_get_url_file": output_get_url_file_meta,
            "token_file": str(resolved_token_file),
            "secret_env_file": str(resolved_secret_env_file),
            "session_budget_ledger": str(resolved_session_budget_ledger),
            "instance_id": instance_id,
            "created_at_epoch": created_epoch,
            "max_live_minutes": max_live_minutes,
            "max_live_deadline_epoch": created_epoch + max_live_minutes * 60.0,
            "selected_offer": _offer_artifact_summary(selected_offer),
            "selected_hourly_rate_usd": selected_offer.get("hourly_rate_usd"),
            "excluded_machine_ids": list(excluded_machine_ids),
            "allowed_machine_ids": list(allowed_machine_ids),
            "public_image": public_image,
            "selected_container_image": selected_container_image,
            "vast_launch_mode": launch_mode,
            "disk_gb": resolved_disk_gb,
            "provider_bundle_inline_transport_used": (
                inline_bundle_transport.get("inline_provider_bundle_transport_used") is True
            ),
            "provider_bundle_inline_transport_reason": _string(
                inline_bundle_transport.get("inline_provider_bundle_transport_reason")
            ),
            "provider_bundle_inline_transport_size_bytes": int(
                _number(inline_bundle_transport.get("inline_provider_bundle_size_bytes")) or 0
            ),
            "provider_bundle_inline_transport_base64_length": int(
                _number(inline_bundle_transport.get("inline_provider_bundle_base64_length")) or 0
            ),
            "provider_bundle_inline_transport_sha256_present": (
                inline_bundle_transport.get("inline_provider_bundle_sha256_present") is True
            ),
            "target_spend_usd": target_spend_usd,
            "hard_cap_usd": hard_cap_usd,
            "max_hourly_rate_usd": max_hourly_rate,
            "min_gpu_ram_mb": int(min_gpu_ram_mb),
            "min_reliability": resolved_min_reliability,
            "require_direct_port": resolved_require_direct_port,
            "preferred_gpu_keywords": list(resolved_preferred_gpu_keywords),
            "preferred_geolocation_regex": resolved_preferred_geolocation_regex,
            "prefer_isaac_rt": prefer_isaac_rt,
            "last_instance_status": status,
            "instance_observations": observations,
            "last_instance_payload_redacted": _redact_runtime_value(
                instance_payload,
                secret_values,
            ),
            "create_http_status_code": create_status,
            "create_response_redacted": _redact_runtime_value(create_response, secret_values),
            "create_request_summary": _create_request_summary(
                create_payload,
                secret_values=secret_values,
            ),
            "image_login_summary": image_login_summary,
            "raw_secret_values_recorded": False,
        }
        write_json(_state_path(resolved_job_dir), state)
        _budget_ledger(
            job_dir=resolved_job_dir,
            generated_at=utc_now_iso(),
            target_spend_usd=target_spend_usd,
            hard_cap_usd=hard_cap_usd,
            max_hourly_rate=max_hourly_rate,
            max_live_minutes=max_live_minutes,
            selected_offer=selected_offer,
            instance_ids=[instance_id],
            started_at_monotonic=0.0,
            ended_at_monotonic=0.0,
            status="live_instance_created_async",
            continuing_spend=True,
        )
        manifest = {
            "schema_version": ASYNC_CREATE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "instance_created",
            "job_dir": str(resolved_job_dir),
            "state_path": str(_state_path(resolved_job_dir)),
            "instance_id": instance_id,
            "last_instance_status": status,
            "output_path": str(resolved_output),
            "selected_offer": _offer_artifact_summary(selected_offer),
            "min_gpu_ram_mb": int(min_gpu_ram_mb),
            "min_reliability": resolved_min_reliability,
            "require_direct_port": resolved_require_direct_port,
            "preferred_gpu_keywords": list(resolved_preferred_gpu_keywords),
            "preferred_geolocation_regex": resolved_preferred_geolocation_regex,
            "prefer_isaac_rt": prefer_isaac_rt,
            "explicit_provider_urls_used": direct_provider_urls,
            "provider_bundle_url_redacted": _redact_provider_url(provider_bundle_url),
            "provider_output_put_url_redacted": _redact_provider_url(provider_output_put_url),
            "provider_output_get_url_redacted": _redact_provider_url(provider_output_get_url),
            "provider_bundle_url_file": bundle_url_file_meta,
            "provider_output_put_url_file": output_url_file_meta,
            "provider_output_get_url_file": output_get_url_file_meta,
            "provider_bundle_inline_transport_used": (
                inline_bundle_transport.get("inline_provider_bundle_transport_used") is True
            ),
            "provider_bundle_inline_transport_reason": _string(
                inline_bundle_transport.get("inline_provider_bundle_transport_reason")
            ),
            "poll_command": (
                "python -m blueprint_pipeline.vast_wam_async_runner poll "
                f"--job-dir {resolved_job_dir}"
            ),
            "teardown_command": (
                "python -m blueprint_pipeline.vast_wam_async_runner poll "
                f"--job-dir {resolved_job_dir} --teardown"
            ),
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_wam_async_create_manifest.json", manifest)
        return manifest
    finally:
        _release_vast_launch_lock(
            lock_handle,
            job_dir=resolved_job_dir,
            generated_at=utc_now_iso(),
        )


def _write_poll_phase_artifacts(
    *,
    job_dir: Path,
    generated_at: str,
    state: Mapping[str, Any],
    heartbeat_text: str,
    onstart_logs: Mapping[str, Any],
    output_download_manifest: Mapping[str, Any],
    output_zip_inspection: Mapping[str, Any],
) -> dict[str, Any]:
    instance_id = int(_number(state.get("instance_id")) or 0)
    heartbeat_ok = "BLUEPRINT_VAST_HEARTBEAT_OK" in heartbeat_text
    gpu_ok = "BLUEPRINT_VAST_GPU_SANITY_OK" in heartbeat_text and "nvidia-smi: command not found" not in heartbeat_text.lower()
    provider_started = "BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED" in heartbeat_text
    provider_downloaded = "BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED" in heartbeat_text
    provider_entrypoint_started = "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED" in heartbeat_text
    provider_upload_ok = "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK" in heartbeat_text
    provider_completed_or_blocked = "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED" in heartbeat_text
    provider_blocked_markers = []
    for item in heartbeat_text.split():
        if item.startswith("BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:"):
            provider_blocked_markers.append(item.split(":", 1)[1])
    provider_runtime_output_zip_received = output_zip_inspection.get("zip_present") is True
    runtime_result_present = output_zip_inspection.get("runtime_result_present") is True
    runtime_result = _mapping(output_zip_inspection.get("runtime_result"))
    runtime_result_status = _string(runtime_result.get("status"))
    runtime_result_blockers = _string_list(runtime_result.get("blockers"))
    mp4_validation = _mapping(output_zip_inspection.get("mp4_validation"))
    video_smoke_proven = output_zip_inspection.get("video_smoke_proven") is True
    remote_proven = (
        provider_started
        and provider_downloaded
        and provider_entrypoint_started
        and provider_completed_or_blocked
    )
    completion_blockers: list[str] = []
    if not provider_started:
        completion_blockers.append("provider_bundle_start_marker_missing")
    if not provider_downloaded:
        completion_blockers.append("provider_bundle_download_marker_missing")
    if not provider_entrypoint_started:
        completion_blockers.append("provider_entrypoint_start_marker_missing")
    if not provider_completed_or_blocked:
        completion_blockers.append("provider_bundle_completion_marker_missing")
    if not provider_upload_ok:
        completion_blockers.append("provider_output_upload_marker_missing")
    if not provider_runtime_output_zip_received:
        completion_blockers.append("provider_runtime_output_zip_not_received_locally")
    if provider_runtime_output_zip_received and not runtime_result_present:
        completion_blockers.append("provider_runtime_result_missing_from_output_zip")
    completion_blockers.extend(
        f"provider_remote_blocker:{marker}" for marker in provider_blocked_markers
    )
    provider_status = (
        "completed"
        if remote_proven and provider_runtime_output_zip_received and runtime_result_present
        else "blocked"
    )
    _append_phase(
        job_dir,
        "vast_heartbeat_completed_or_blocked",
        "completed" if heartbeat_ok else "blocked",
        blockers=[] if heartbeat_ok else ["vast_heartbeat_output_missing_success_marker"],
        instance_id=instance_id,
    )
    _append_phase(
        job_dir,
        "vast_gpu_sanity_completed_or_blocked",
        "completed" if gpu_ok else "blocked",
        blockers=[] if gpu_ok else ["vast_gpu_sanity_output_missing_or_nvidia_smi_failed"],
        instance_id=instance_id,
    )
    _append_phase(job_dir, "vast_isaac_smoke_started", "completed", proof_effect="isaac_smoke_not_required_for_wam_bundle")
    _append_phase(job_dir, "vast_isaac_smoke_completed_or_blocked", "completed", proof_effect="isaac_smoke_not_required_for_wam_bundle")
    _append_phase(
        job_dir,
        "vast_blueprint_bundle_started",
        "completed" if provider_started else "blocked",
        blockers=[] if provider_started else ["provider_bundle_start_marker_missing"],
        instance_id=instance_id,
    )
    _append_phase(
        job_dir,
        "vast_blueprint_bundle_completed_or_blocked",
        "completed" if provider_status == "completed" else "blocked",
        blockers=completion_blockers,
        instance_id=instance_id,
    )
    write_json(
        job_dir / "vast_startup_probe_manifest.json",
        {
            "schema_version": VAST_STARTUP_PROBE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "completed" if heartbeat_ok else "blocked",
            "instance_id": instance_id,
            "heartbeat_completed": heartbeat_ok,
            "startup_probe_proven": heartbeat_ok,
            "container_log_result": dict(onstart_logs),
            "blockers": [] if heartbeat_ok else ["vast_heartbeat_output_missing_success_marker"],
            "raw_secret_values_recorded": False,
        },
    )
    write_json(
        job_dir / "vast_gpu_sanity_report.json",
        {
            "schema_version": VAST_GPU_SANITY_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "completed" if gpu_ok else "blocked",
            "instance_id": instance_id,
            "nvidia_smi_visible": gpu_ok,
            "gpu_sanity_proven": gpu_ok,
            "blockers": [] if gpu_ok else ["vast_gpu_sanity_output_missing_or_nvidia_smi_failed"],
            "raw_secret_values_recorded": False,
        },
    )
    write_json(
        job_dir / "vast_isaac_smoke_result.json",
        {
            "schema_version": VAST_ISAAC_SMOKE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "not_required",
            "instance_id": instance_id,
            "provider_bundle_kind": "wam",
            "isaac_smoke_attempted": False,
            "blockers": [],
            "raw_secret_values_recorded": False,
        },
    )
    provider_command = {
        "schema_version": VAST_PROVIDER_COMMAND_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": provider_status,
        "instance_id": instance_id,
        "provider_bundle_started": provider_started,
        "provider_bundle_downloaded": provider_downloaded,
        "provider_entrypoint_started": provider_entrypoint_started,
        "provider_completed_or_blocked_marker_seen": provider_completed_or_blocked,
        "provider_output_upload_ok": provider_upload_ok,
        "provider_runtime_output_zip_received": provider_runtime_output_zip_received,
        "provider_command_path_remote_proven": remote_proven,
        "provider_bundle_kind": "wam",
        "provider_runtime_output_zip_path": state.get("output_path"),
        "provider_runtime_output_zip_inspection": dict(output_zip_inspection),
        "provider_output_download_manifest": dict(output_download_manifest),
        "runtime_result_status": runtime_result_status or None,
        "runtime_result_blockers": runtime_result_blockers,
        "blueprint_provider_bundle_execution_proven": provider_status == "completed",
        "blockers": completion_blockers,
        "raw_credentials_recorded": False,
    }
    write_json(job_dir / "vast_provider_command_result.json", provider_command)
    write_json(
        job_dir / "vast_video_smoke_result.json",
        {
            "schema_version": VAST_VIDEO_SMOKE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "completed" if video_smoke_proven else "blocked",
            "provider_bundle_kind": "wam",
            "provider_runtime_output_zip_path": state.get("output_path"),
            "provider_runtime_output_zip_received": provider_runtime_output_zip_received,
            "video_smoke_proven": video_smoke_proven,
            "expected_video_count": DEFAULT_WAM_ROLLOUT_VIDEO_COUNT,
            "mp4_count": output_zip_inspection.get("mp4_count"),
            "mp4_members": output_zip_inspection.get("mp4_members"),
            "mp4_validation": mp4_validation,
            "blockers": []
            if video_smoke_proven
            else (_string_list(mp4_validation.get("blockers")) or ["mp4_video_smoke_not_proven"]),
            "raw_secret_values_recorded": False,
        },
    )
    return provider_command


def poll_async_vast_wam_run(
    *,
    job_dir: str | Path,
    max_wait_seconds: int = 45,
    retry_interval_seconds: int = 10,
    teardown: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    state = _read_async_state(resolved_job_dir)
    instance_id = int(_number(state.get("instance_id")) or 0)
    if not instance_id:
        manifest = {
            "schema_version": ASYNC_POLL_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "blockers": ["vast_wam_async_state_missing_instance_id"],
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_wam_async_poll_manifest.json", manifest)
        return manifest
    api_key, _vast_secret_status = _read_secret_file(VAST_API_KEY_FILE_ENV, "~/.blueprint-secrets/vast_api_key")
    if not api_key:
        manifest = {
            "schema_version": ASYNC_POLL_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "instance_id": instance_id,
            "blockers": [f"missing_file_based_secret_{VAST_API_KEY_FILE_ENV}"],
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_wam_async_poll_manifest.json", manifest)
        return manifest
    provider_output_get_url = ""
    if state.get("explicit_provider_urls_used") is True:
        provider_bundle_url, _bundle_url_file_meta = _read_sensitive_url_file(
            _url_file_path_from_meta(state.get("provider_bundle_url_file")),
            label="provider_bundle_url_file",
        )
        provider_output_put_url, _output_put_url_file_meta = _read_sensitive_url_file(
            _url_file_path_from_meta(state.get("provider_output_put_url_file")),
            label="provider_output_put_url_file",
        )
        provider_output_get_url, _output_get_url_file_meta = _read_sensitive_url_file(
            _url_file_path_from_meta(state.get("provider_output_get_url_file")),
            label="provider_output_get_url_file",
        )
    else:
        token_file = Path(_string(state.get("token_file"))).expanduser().resolve()
        public_base_url = _string(state.get("public_base_url"))
        provider_bundle_url, provider_output_put_url, _token_status = _provider_urls(public_base_url, token_file)
    secret_values = [
        api_key,
        *_forwarded_secret_values(),
        *_url_secret_values(provider_bundle_url, provider_output_put_url, provider_output_get_url),
    ]
    _append_phase(resolved_job_dir, "vast_heartbeat_started", "running", instance_id=instance_id)
    poll_started_epoch = time.time()
    (
        effective_max_wait_seconds,
        seconds_until_max_live_deadline,
        log_wait_deadline_cap_applied,
    ) = _deadline_capped_log_wait_seconds(
        state=state,
        requested_max_wait_seconds=max_wait_seconds,
        now_epoch=poll_started_epoch,
    )
    # Tolerate a missing container only for a bounded boot/pull window; a dud offer whose
    # container never materializes is torn down quickly instead of idling the full deadline.
    container_missing_max_seconds = min(
        int(effective_max_wait_seconds),
        max(
            60,
            _env_int(
                VAST_WAM_CONTAINER_MISSING_MAX_SECONDS_ENV,
                DEFAULT_VAST_WAM_CONTAINER_MISSING_MAX_SECONDS,
            ),
        ),
    )
    container_missing_retry_attempts = max(
        1, int(container_missing_max_seconds / max(1, retry_interval_seconds))
    )
    onstart_logs = _request_logs_and_fetch(
        instance_id=instance_id,
        api_key=api_key,
        output_log_path=resolved_job_dir / "vast_onstart_container.log",
        secret_values=secret_values,
        wait_seconds=0,
        tail_lines=2000,
        success_markers=[
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED",
            "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK",
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED",
            "BLUEPRINT_VAST_ONSTART_DONE",
        ],
        max_wait_seconds=effective_max_wait_seconds,
        retry_interval_seconds=retry_interval_seconds,
        container_missing_retry_attempts=container_missing_retry_attempts,
    )
    heartbeat_text = Path(onstart_logs["output_log_path"]).read_text(encoding="utf-8")
    output_path = Path(_string(state.get("output_path"))).expanduser().resolve()
    provider_upload_marker_seen = "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK" in heartbeat_text
    output_download_manifest = _download_provider_output_zip(
        job_dir=resolved_job_dir,
        output_zip_path=output_path,
        provider_output_get_url=provider_output_get_url,
        provider_upload_marker_seen=provider_upload_marker_seen,
        generated_at=generated,
    )
    output_zip_inspection = _inspect_provider_runtime_output_zip(
        output_path,
        video_extract_dir=resolved_job_dir / "vast_provider_runtime_output_videos",
        expected_video_count=DEFAULT_WAM_ROLLOUT_VIDEO_COUNT,
    )
    provider_command = _write_poll_phase_artifacts(
        job_dir=resolved_job_dir,
        generated_at=generated,
        state=state,
        heartbeat_text=heartbeat_text,
        onstart_logs=onstart_logs,
        output_download_manifest=output_download_manifest,
        output_zip_inspection=output_zip_inspection,
    )
    status_payload: dict[str, Any] = {}
    instance_status = "unknown"
    try:
        _status_code, status_payload = _api_json(
            method="GET",
            path=f"/instances/{instance_id}/",
            api_key=api_key,
            timeout_seconds=30,
        )
        instance_status = _instance_status(status_payload)
    except Exception as exc:
        instance_status = f"status_probe_failed:{type(exc).__name__}"

    now_epoch = time.time()
    timed_out = now_epoch >= float(_number(state.get("max_live_deadline_epoch")) or 0.0)
    provider_done_or_blocked = (
        "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED" in heartbeat_text
        or "BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED" in heartbeat_text
        or provider_command.get("provider_runtime_output_zip_received") is True
    )
    should_teardown = bool(teardown or timed_out or provider_done_or_blocked)
    teardown_actions: list[dict[str, Any]] = []
    continuing_spend = not should_teardown
    if should_teardown:
        _append_phase(
            resolved_job_dir,
            "vast_instance_teardown_started",
            "running",
            instance_ids=[instance_id],
        )
        continuing_spend, teardown_actions = _destroy_vast_instance_with_retry(
            instance_id=instance_id,
            api_key=api_key,
        )
        write_json(
            resolved_job_dir / "vast_teardown_manifest.json",
            {
                "schema_version": VAST_TEARDOWN_SCHEMA_VERSION,
                "generated_at": utc_now_iso(),
                "status": "completed" if not continuing_spend else "blocked",
                "vast_instance_ids": [instance_id],
                "teardown_actions_performed": teardown_actions,
                "runner_gpu_teardown_completed": not continuing_spend,
                "continuing_spend_from_this_run": continuing_spend,
                "zero_continuing_spend_scope": "async poll destroyed instance or it was already absent"
                if not continuing_spend
                else "teardown failure requires manual Vast console/API verification",
                "raw_secret_values_recorded": False,
            },
        )
        _append_phase(
            resolved_job_dir,
            "vast_instance_teardown_completed",
            "completed" if not continuing_spend else "blocked",
            blockers=[] if not continuing_spend else ["vast_instance_destroy_failed"],
            instance_ids=[instance_id],
        )
    else:
        write_json(
            resolved_job_dir / "vast_teardown_manifest.json",
            {
                "schema_version": VAST_TEARDOWN_SCHEMA_VERSION,
                "generated_at": utc_now_iso(),
                "status": "deferred_async_run_still_active",
                "vast_instance_ids": [instance_id],
                "teardown_actions_performed": [],
                "runner_gpu_teardown_completed": False,
                "continuing_spend_from_this_run": True,
                "raw_secret_values_recorded": False,
            },
        )
    elapsed_seconds = max(0.0, now_epoch - float(_number(state.get("created_at_epoch")) or now_epoch))
    ledger = _budget_ledger(
        job_dir=resolved_job_dir,
        generated_at=utc_now_iso(),
        target_spend_usd=float(_number(state.get("target_spend_usd")) or DEFAULT_TARGET_SPEND_USD),
        hard_cap_usd=float(_number(state.get("hard_cap_usd")) or DEFAULT_HARD_CAP_USD),
        max_hourly_rate=float(_number(state.get("max_hourly_rate_usd")) or DEFAULT_MAX_HOURLY_RATE),
        max_live_minutes=int(_number(state.get("max_live_minutes")) or 0),
        selected_offer=_mapping(state.get("selected_offer")),
        instance_ids=[instance_id],
        started_at_monotonic=0.0,
        ended_at_monotonic=elapsed_seconds,
        status="completed" if not continuing_spend else "poll_running",
        continuing_spend=continuing_spend,
    )
    if not continuing_spend:
        _append_session_budget_attempt(
            budget_path=Path(_string(state.get("session_budget_ledger"))).expanduser().resolve(),
            job_dir=resolved_job_dir,
            generated_at=generated,
            ledger=ledger,
            selected_offer=_mapping(state.get("selected_offer")),
            result_status=_string(provider_command.get("status")) or "blocked",
            result_reason="vast_wam_async_poll_completed",
            blockers=_string_list(provider_command.get("blockers")),
        )
    _append_phase(
        resolved_job_dir,
        "vast_artifacts_exported",
        "completed",
        proof_effect="vast_wam_async_poll_artifacts_written",
    )
    _fill_missing_phase_rows(
        resolved_job_dir,
        reason="vast_wam_async_poll_no_observation_for_phase",
    )
    validation = _final_validation(
        job_dir=resolved_job_dir,
        generated_at=utc_now_iso(),
        instance_ids=[instance_id],
        continuing_spend=continuing_spend,
        estimated_cost_usd=float(ledger.get("estimated_cost_usd") or 0.0),
        hard_cap_usd=float(_number(state.get("hard_cap_usd")) or DEFAULT_HARD_CAP_USD),
    )
    result = {
        "schema_version": VAST_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION,
        "generated_at": generated,
        "job_dir": str(resolved_job_dir),
        "status": "completed"
        if provider_command.get("status") == "completed" and not continuing_spend
        else ("running" if continuing_spend else "blocked"),
        "reason": "vast_wam_async_poll_completed",
        "api_call_performed": True,
        "vast_side_effects_may_have_occurred": True,
        "vast_instance_ids": [instance_id],
        "adapter_result_mode": "async_poll",
        "provider_command_status": provider_command.get("status"),
        "provider_command_blockers": provider_command.get("blockers"),
        "continuing_spend_from_this_run": continuing_spend,
        "requested_log_fetch_max_wait_seconds": max_wait_seconds,
        "effective_log_fetch_max_wait_seconds": effective_max_wait_seconds,
        "seconds_until_max_live_deadline_at_poll_start": seconds_until_max_live_deadline,
        "log_wait_deadline_cap_applied": log_wait_deadline_cap_applied,
        "final_validation_status": validation.get("status"),
        "raw_secret_values_recorded": False,
    }
    write_json(resolved_job_dir / "vast_provider_adapter_result.json", result)
    state_update = {
        **state,
        "status": "teardown_completed" if not continuing_spend else "running",
        "last_polled_at": generated,
        "last_instance_status": instance_status,
        "last_instance_payload_redacted": _redact_runtime_value(status_payload, secret_values),
        "provider_command_status": provider_command.get("status"),
        "provider_command_blockers": provider_command.get("blockers"),
        "continuing_spend_from_this_run": continuing_spend,
        "raw_secret_values_recorded": False,
    }
    write_json(_state_path(resolved_job_dir), state_update)
    manifest = {
        "schema_version": ASYNC_POLL_SCHEMA_VERSION,
        "generated_at": generated,
        "status": result["status"],
        "job_dir": str(resolved_job_dir),
        "instance_id": instance_id,
        "instance_status": instance_status,
        "provider_command_status": provider_command.get("status"),
        "provider_command_blockers": provider_command.get("blockers"),
        "output_zip_present": output_zip_inspection.get("zip_present"),
        "runtime_result_status": output_zip_inspection.get("runtime_result_status"),
        "runtime_result_blockers": output_zip_inspection.get("runtime_result_blockers"),
        "mp4_count": output_zip_inspection.get("mp4_count"),
        "teardown_requested": teardown,
        "teardown_performed": should_teardown,
        "continuing_spend_from_this_run": continuing_spend,
        "requested_log_fetch_max_wait_seconds": max_wait_seconds,
        "effective_log_fetch_max_wait_seconds": effective_max_wait_seconds,
        "seconds_until_max_live_deadline_at_poll_start": seconds_until_max_live_deadline,
        "log_wait_deadline_cap_applied": log_wait_deadline_cap_applied,
        "estimated_cost_usd": ledger.get("estimated_cost_usd"),
        "final_validation_status": validation.get("status"),
        "raw_secret_values_recorded": False,
    }
    write_json(resolved_job_dir / "vast_wam_async_poll_manifest.json", manifest)
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
    create.add_argument("--session-budget-ledger")
    create.add_argument("--allow-paid-vast-launch", action="store_true")
    create.add_argument("--max-hourly-rate", type=float, default=DEFAULT_MAX_HOURLY_RATE)
    create.add_argument("--target-spend-usd", type=float, default=DEFAULT_TARGET_SPEND_USD)
    create.add_argument("--hard-cap-usd", type=float, default=DEFAULT_HARD_CAP_USD)
    create.add_argument("--allow-target-spend-overrun", action="store_true")
    create.add_argument("--max-live-minutes", type=int, default=30)
    create.add_argument("--session-max-live-minutes", type=int, default=45)
    create.add_argument(
        "--min-gpu-ram-mb",
        type=int,
        default=0,
        help="Minimum GPU memory in MiB required when selecting a Vast offer.",
    )
    create.add_argument(
        "--excluded-machine-id",
        action="append",
        type=int,
        default=[],
        help="Vast machine id to exclude from offer selection; repeatable.",
    )
    create.add_argument(
        "--allowed-machine-id",
        action="append",
        type=int,
        default=[],
        help="Restrict Vast offer selection to this machine id; repeatable.",
    )
    create.add_argument(
        "--min-reliability",
        type=float,
        help=(
            "Minimum Vast offer reliability. Defaults to "
            f"{VAST_MIN_RELIABILITY_ENV} when set, otherwise no hard floor."
        ),
    )
    create.add_argument(
        "--require-direct-port",
        action="store_true",
        help=f"Require direct_port_count > 0. Can also be enabled with {VAST_REQUIRE_DIRECT_PORT_ENV}=true.",
    )
    create.add_argument(
        "--preferred-gpu-keyword",
        action="append",
        default=[],
        help=(
            "Preferred GPU family keyword for WAM offer sorting; repeatable. "
            f"Defaults to {VAST_PREFERRED_GPU_KEYWORDS_ENV} or workstation/datacenter families."
        ),
    )
    create.add_argument(
        "--preferred-geolocation-regex",
        default="",
        help=f"Regex used to prefer Vast geolocations before price; {VAST_PREFERRED_GEOLOCATION_REGEX_ENV} can also set it.",
    )
    create.add_argument(
        "--prefer-isaac-rt",
        action="store_true",
        help="Use the Isaac rendering RTX-first pool for WAM offer sorting. Disabled by default for WAM.",
    )
    create.add_argument("--startup-poll-seconds", type=int, default=90)
    create.add_argument("--public-staging-verify-max-wait-seconds", type=int, default=120)
    create.add_argument("--public-staging-verify-retry-interval-seconds", type=float, default=5.0)
    create.add_argument("--public-staging-verify-timeout-seconds", type=float, default=20.0)
    create.add_argument("--public-staging-required-consecutive-successes", type=int, default=2)
    create.add_argument("--verify-output-put-url", action="store_true")
    create.add_argument("--public-image", default=DEFAULT_WAM_PUBLIC_IMAGE)
    create.add_argument("--vast-launch-mode", default=DEFAULT_WAM_VAST_LAUNCH_MODE)
    create.add_argument("--disk-gb", type=int, default=DEFAULT_DISK_GB)
    create.add_argument("--heartbeat-url", default=DEFAULT_HEARTBEAT_URL)
    poll = subparsers.add_parser("poll")
    poll.add_argument("--job-dir", required=True)
    poll.add_argument("--max-wait-seconds", type=int, default=45)
    poll.add_argument("--retry-interval-seconds", type=int, default=10)
    poll.add_argument("--teardown", action="store_true")
    destroy = subparsers.add_parser("destroy")
    destroy.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        print("legacy_vast_wam_create_cli_disabled", file=sys.stderr)
        return 2
    if args.command == "poll":
        manifest = poll_async_vast_wam_run(
            job_dir=args.job_dir,
            max_wait_seconds=args.max_wait_seconds,
            retry_interval_seconds=args.retry_interval_seconds,
            teardown=args.teardown,
        )
    else:
        manifest = destroy_async_vast_wam_run(job_dir=args.job_dir)
    print(json.dumps(manifest, sort_keys=True))
    return 0 if manifest.get("status") in {"instance_created", "running", "completed"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
