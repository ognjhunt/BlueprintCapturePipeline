"""Coordinate a gated Vast run for OSCAR/Cosmos-style WAM bundles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .provider_bundle_staging_common import (
    BUNDLE_ROUTE,
    OUTPUT_ROUTE,
    read_or_create_staging_token as _read_or_create_token,
    staging_url_with_token as _url_with_token,
)
from .vast_bundle_staging import (
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_SECRET_ENV_FILE,
    DEFAULT_TOKEN_FILE,
    prepare_vast_bundle_staging,
    run_local_staging_self_test,
    verify_public_staging_urls,
)
from .vast_independent_watchdog_control import (
    VastWatchdogHandle,
    arm_independent_vast_watchdog,
    close_independent_vast_watchdog,
)
from .vast_provider_adapter import (
    DEFAULT_HARD_CAP_USD,
    DEFAULT_MAX_HOURLY_RATE,
    DEFAULT_TARGET_SPEND_USD,
    VAST_API_GATE_ENV,
    VAST_INSTANCE_LAUNCH_GATE_ENV,
    run_vast_provider_adapter,
    _vast_session_budget_ledger_path,
)
from .vast_probe_guards import (
    staging_verification_guard as build_staging_verification_guard,
    target_spend_guard as build_target_spend_guard,
)
from .wam_async_runner_common import read_sensitive_url_file, redact_provider_url
from .oscar_official_release import OFFICIAL_OSCAR_WAM_IMAGE_REF
from .paid_resource_admission import PaidResourceAdmissionGrant
from .policy_ranking_successor_retained_session import create_retained_session_manifest
from .gpu_render_providers import get_render_provider
from .wam_async_runner_common import download_url_to_file
from .wam_provider_output import inspect_provider_runtime_output_zip


VAST_WAM_AUTHORIZED_RUNNER_SCHEMA_VERSION = "vast_wam_authorized_runner.v1"
DEFAULT_WAM_PUBLIC_IMAGE = OFFICIAL_OSCAR_WAM_IMAGE_REF
DEFAULT_WAM_VAST_LAUNCH_MODE = "auto"
RECOVERABLE_PROVIDER_OBSERVATION_BLOCKERS = frozenset(
    {
        "vast_heartbeat_container_missing",
        "vast_heartbeat_output_missing_success_marker",
        "vast_heartbeat_no_log_progress_timeout",
    }
)


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _redacted_path(route: str) -> str:
    return f"/{route.strip('/')}?token=<redacted-token>"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, Mapping) else {}


def _truth_boundaries() -> dict[str, Any]:
    return {
        "wam_vla_runtime_proven": False,
        "action_conditioned_video_rollout_generated": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "official_policy_execution_proven": False,
        "controller_grade_execution_proven": False,
    }


def _timestamp(value: Any) -> datetime | None:
    text = _string(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _recover_completed_provider_output(
    *,
    job_dir: Path,
    output_path: Path,
    provider_output_get_url: str,
    provider_bundle_kind: str,
    adapter_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover a fresh completed callback archive after a provider log-tail race.

    Vast can destroy a short-lived container between log polls after the callback
    upload succeeds. Recovery is deliberately narrow: only observation-layer
    blockers qualify, provider teardown must already be proven, the callback must
    have been modified during this allocation, and the archive must contain a
    completed runtime result. Scientific gates remain separate.
    """

    blockers = sorted(
        {
            str(item).strip()
            for item in adapter_result.get("blockers") or []
            if str(item).strip()
        }
    )
    manifest: dict[str, Any] = {
        "schema_version": "vast_provider_output_completion_recovery.v1",
        "generated_at": utc_now_iso(),
        "status": "not_eligible",
        "adapter_status": adapter_result.get("status"),
        "adapter_blockers": blockers,
        "provider_bundle_kind": provider_bundle_kind,
        "provider_output_get_url_present": bool(_string(provider_output_get_url)),
        "provider_instance_ids_present": bool(adapter_result.get("vast_instance_ids")),
        "provider_teardown_proven": (
            adapter_result.get("continuing_spend_from_this_run") is False
        ),
        "raw_secret_values_recorded": False,
    }
    eligibility_blockers: list[str] = []
    if adapter_result.get("status") == "completed":
        eligibility_blockers.append("adapter_already_completed")
    if not blockers or not set(blockers).issubset(RECOVERABLE_PROVIDER_OBSERVATION_BLOCKERS):
        eligibility_blockers.append("adapter_failure_not_limited_to_observation_layer")
    if adapter_result.get("continuing_spend_from_this_run") is not False:
        eligibility_blockers.append("provider_teardown_not_proven")
    if not adapter_result.get("vast_instance_ids"):
        eligibility_blockers.append("provider_instance_id_missing")
    if not _string(provider_output_get_url):
        eligibility_blockers.append("provider_output_get_url_missing")
    if eligibility_blockers:
        manifest["blockers"] = eligibility_blockers
        write_json(job_dir / "vast_provider_output_completion_recovery.json", manifest)
        return manifest

    transfer = download_url_to_file(
        url=_string(provider_output_get_url),
        output_path=output_path,
        user_agent="BlueprintVastAuthorizedWamRecovery/1.0",
        timeout_seconds=180,
    )
    manifest["download"] = {
        key: transfer.get(key)
        for key in (
            "status",
            "http_status_code",
            "downloaded_size_bytes",
            "output_present",
            "response_last_modified",
            "response_etag_present",
            "error_type",
        )
        if key in transfer
    }
    if transfer.get("status") != "completed":
        manifest.update(
            {
                "status": "blocked",
                "blockers": ["provider_output_recovery_download_failed"],
            }
        )
        write_json(job_dir / "vast_provider_output_completion_recovery.json", manifest)
        return manifest

    allocation_started = _timestamp(adapter_result.get("generated_at"))
    object_modified = _timestamp(transfer.get("response_last_modified"))
    fresh_for_allocation = bool(
        allocation_started and object_modified and object_modified >= allocation_started
    )
    manifest["freshness"] = {
        "allocation_started_at": (
            allocation_started.isoformat() if allocation_started else None
        ),
        "object_last_modified_at": object_modified.isoformat() if object_modified else None,
        "object_fresh_for_allocation": fresh_for_allocation,
    }
    inspection = inspect_provider_runtime_output_zip(
        output_path,
        expected_video_count=1 if provider_bundle_kind == "wam" else None,
    )
    manifest["output_inspection"] = inspection
    runtime_completed = (
        inspection.get("status") == "completed"
        and inspection.get("runtime_result_present") is True
        and inspection.get("runtime_result_status") == "completed"
        and not inspection.get("runtime_result_blockers")
    )
    wam_media_present = provider_bundle_kind != "wam" or int(
        inspection.get("mp4_count") or 0
    ) >= 1
    recovery_blockers: list[str] = []
    if not fresh_for_allocation:
        recovery_blockers.append("provider_output_object_not_fresh_for_allocation")
    if not runtime_completed:
        recovery_blockers.append("provider_runtime_result_not_completed")
    if not wam_media_present:
        recovery_blockers.append("provider_wam_output_missing_video")
    manifest.update(
        {
            "status": "completed" if not recovery_blockers else "blocked",
            "completion_recovered": not recovery_blockers,
            "blockers": recovery_blockers,
            "claim_boundary": {
                "provider_transport_completion_only": True,
                "scientific_validity_proven": False,
                "ranking_fidelity_proven": False,
            },
        }
    )
    write_json(job_dir / "vast_provider_output_completion_recovery.json", manifest)
    return manifest


def run_vast_wam_authorized_runner(
    *,
    job_dir: str | Path,
    bundle_path: str | Path,
    public_base_url: str | None = None,
    token_file: str | Path | None = None,
    secret_env_file: str | Path | None = None,
    provider_bundle_url_file: str | Path | None = None,
    provider_output_put_url_file: str | Path | None = None,
    provider_output_get_url_file: str | Path | None = None,
    output_path: str | Path | None = None,
    session_budget_ledger: str | Path | None = None,
    allow_paid_vast_launch: bool = False,
    max_hourly_rate: float = DEFAULT_MAX_HOURLY_RATE,
    target_spend_usd: float = DEFAULT_TARGET_SPEND_USD,
    hard_cap_usd: float = DEFAULT_HARD_CAP_USD,
    allow_target_spend_overrun: bool = False,
    max_live_minutes: int = 45,
    session_max_live_minutes: int | None = 45,
    startup_timeout_seconds: int = 1800,
    verify_staging_urls: bool = True,
    allow_unverified_public_staging_for_paid_launch: bool = False,
    public_staging_verify_max_wait_seconds: int = 180,
    public_staging_verify_retry_interval_seconds: float = 5.0,
    public_staging_verify_timeout_seconds: float = 20.0,
    public_staging_required_consecutive_successes: int = 3,
    allow_staging_output_put_probe: bool = True,
    public_image: str = DEFAULT_WAM_PUBLIC_IMAGE,
    vast_launch_mode: str = DEFAULT_WAM_VAST_LAUNCH_MODE,
    vast_template_hash_id: str | None = None,
    use_vast_template_image: bool = False,
    disk_gb: int = 80,
    min_gpu_ram_mb: int | None = None,
    min_compute_cap: int | None = None,
    max_compute_cap: int | None = None,
    min_reliability: float | None = None,
    preferred_gpu_keywords: Sequence[str] = (),
    prefer_isaac_rt: bool = False,
    gpu_selection_policy: str | Mapping[str, Any] | None = None,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None = None,
    pre_provider_mutation_hook: Callable[[], Mapping[str, Any]] | None = None,
    require_independent_watchdog: bool = False,
    retain_instance_on_runtime_failure: bool = False,
    retention_binding: Mapping[str, Any] | None = None,
    forward_hf_token: bool = True,
    provider_bundle_kind: str = "wam",
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
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
    direct_transport_configured = any(
        value is not None
        for value in (
            provider_bundle_url_file,
            provider_output_put_url_file,
            provider_output_get_url_file,
        )
    )

    token, token_status = _read_or_create_token(resolved_token_file)
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
        output_path=resolved_job_dir / "vast_wam_staging_self_test_output.zip",
        token_file=resolved_token_file,
        generated_at=generated,
    )

    blockers: list[str] = []
    if staging_manifest.get("status") != "ready" and not direct_transport_configured:
        blockers.extend(str(item) for item in staging_manifest.get("blockers") or [])
    if self_test.get("status") != "passed":
        blockers.append("local_staging_self_test_failed")
    provider_bundle_url = ""
    provider_output_put_url = ""
    provider_output_get_url = ""
    direct_url_files = {
        "provider_bundle_url": provider_bundle_url_file,
        "provider_output_put_url": provider_output_put_url_file,
        "provider_output_get_url": provider_output_get_url_file,
    }
    direct_url_status: dict[str, dict[str, Any]] = {}
    if direct_transport_configured:
        direct_urls: dict[str, str] = {}
        for label, path_value in direct_url_files.items():
            value, metadata = read_sensitive_url_file(str(path_value or ""), label=label)
            direct_url_status[label] = {
                "configured": metadata.get("configured") is True,
                "present": metadata.get("present") is True,
                "mode_is_0600": metadata.get("mode_is_0600") is True,
                "value_present": metadata.get("value_present") is True,
                "raw_secret_values_recorded": False,
            }
            if not value:
                blockers.append(f"{label}_missing")
            elif not value.startswith("https://"):
                blockers.append(f"{label}_not_https")
            elif metadata.get("mode_is_0600") is not True:
                blockers.append(f"{label}_file_permissions_not_0600")
            direct_urls[label] = value
        provider_bundle_url = direct_urls.get("provider_bundle_url", "")
        provider_output_put_url = direct_urls.get("provider_output_put_url", "")
        provider_output_get_url = direct_urls.get("provider_output_get_url", "")
    elif _string(public_base_url):
        provider_bundle_url = _url_with_token(_string(public_base_url), BUNDLE_ROUTE, token)
        provider_output_put_url = _url_with_token(_string(public_base_url), OUTPUT_ROUTE, token)
    else:
        blockers.append("public_staging_transport_missing_for_paid_vast_launch")
    public_staging_verification: dict[str, Any] = {"status": "not_requested"}
    if (
        allow_paid_vast_launch
        and verify_staging_urls
        and not allow_unverified_public_staging_for_paid_launch
        and provider_bundle_url
        and provider_output_put_url
    ):
        public_staging_verification = verify_public_staging_urls(
            job_dir=resolved_job_dir,
            provider_bundle_url=provider_bundle_url,
            provider_output_put_url=provider_output_put_url,
            bundle_path=resolved_bundle,
            output_path=resolved_output,
            max_wait_seconds=public_staging_verify_max_wait_seconds,
            retry_interval_seconds=public_staging_verify_retry_interval_seconds,
            timeout_seconds=public_staging_verify_timeout_seconds,
            required_consecutive_successes=public_staging_required_consecutive_successes,
            allow_output_put_probe=allow_staging_output_put_probe,
            cleanup_output_probe=True,
            bundle_probe_method=("GET" if direct_transport_configured else "HEAD"),
            generated_at=generated,
        )
        if public_staging_verification.get("status") != "passed":
            blockers.extend(
                str(item)
                for item in public_staging_verification.get("blockers")
                or ["public_staging_url_stability_not_proven"]
            )
    staging_guard_manifest = staging_manifest
    if direct_transport_configured:
        staging_guard_manifest = {
            **staging_manifest,
            "status": "ready",
            "provider_fetchable_bundle_uri_ready": bool(provider_bundle_url),
            "provider_output_callback_ready": bool(provider_output_put_url),
            "blockers": [],
        }
    staging_verification_guard = build_staging_verification_guard(
        verify_staging_urls=verify_staging_urls,
        allow_unverified_public_staging_for_paid_launch=(
            allow_unverified_public_staging_for_paid_launch
        ),
        staging_manifest=staging_guard_manifest,
        public_staging_verification=public_staging_verification,
    )
    if allow_paid_vast_launch:
        blockers.extend(str(item) for item in staging_verification_guard.get("blockers") or [])
    target_spend_guard = build_target_spend_guard(
        budget_path=resolved_session_budget_ledger,
        target_spend_usd=target_spend_usd,
        max_hourly_rate=max_hourly_rate,
        max_live_minutes=max_live_minutes,
        allow_target_spend_overrun=allow_target_spend_overrun,
    )
    if allow_paid_vast_launch:
        blockers.extend(str(item) for item in target_spend_guard.get("blockers") or [])

    adapter_result: dict[str, Any] | None = None
    watchdog_handle: VastWatchdogHandle | None = None
    watchdog_handoff: dict[str, Any] = {"status": "not_required"}
    watchdog_close: dict[str, Any] = {"status": "not_required"}
    retained_session: dict[str, Any] = {"status": "not_created"}
    completion_recovery: dict[str, Any] = {"status": "not_required"}
    paid_launch_attempted = False
    if allow_paid_vast_launch:
        if not provider_bundle_url or not provider_output_put_url:
            blockers.append("paid_vast_launch_requires_public_staging_urls")
        elif blockers:
            blockers.append("paid_vast_launch_preflight_blocked")
        else:
            if require_independent_watchdog:
                watchdog_handoff, watchdog_handle = arm_independent_vast_watchdog(
                    job_dir=resolved_job_dir,
                    max_live_minutes=max_live_minutes,
                    generated_at=generated,
                    pod_name_prefix="blueprint-groot-oscar-canary-vast-wam-",
                )
                blockers.extend(str(item) for item in watchdog_handoff.get("blockers") or [])
            if blockers:
                blockers.append("paid_vast_launch_watchdog_blocked")
            else:
                paid_launch_attempted = True
                adapter_result = run_vast_provider_adapter(
                    job_dir=resolved_job_dir,
                    mode="live-startup-probe",
                    allow_vast_api_call=True,
                    allow_instance_launch=True,
                    max_hourly_rate=max_hourly_rate,
                    target_spend_usd=target_spend_usd,
                    hard_cap_usd=hard_cap_usd,
                    max_live_minutes=max_live_minutes,
                    public_image=public_image,
                    provider_bundle=resolved_bundle,
                    provider_bundle_url=provider_bundle_url,
                    provider_output_put_url=provider_output_put_url,
                    provider_output_get_url=provider_output_get_url,
                    provider_runtime_output_zip=resolved_output,
                    enable_isaac_smoke=False,
                    enable_blueprint_bundle=True,
                    provider_bundle_kind=provider_bundle_kind,
                    vast_launch_mode=vast_launch_mode,
                    startup_timeout_seconds=startup_timeout_seconds,
                    session_budget_ledger_path=resolved_session_budget_ledger,
                    session_max_live_minutes=session_max_live_minutes,
                    verify_staging_urls=verify_staging_urls,
                    ngc_image_login_mode="never",
                    vast_template_hash_id=vast_template_hash_id,
                    use_vast_template_image=use_vast_template_image,
                    require_known_supported_isaac_driver=False,
                    disk_gb=disk_gb,
                    min_gpu_ram_mb=min_gpu_ram_mb,
                    min_compute_cap=min_compute_cap,
                    max_compute_cap=max_compute_cap,
                    min_reliability=min_reliability,
                    preferred_gpu_keywords=preferred_gpu_keywords,
                    prefer_isaac_rt=prefer_isaac_rt,
                    gpu_selection_policy=gpu_selection_policy,
                    instance_label_prefix=(
                        watchdog_handle.pod_name_prefix
                        if watchdog_handle
                        else "blueprint-vast-probe-"
                    ),
                    started_instance_id_path=(
                        watchdog_handle.started_instance_id_path if watchdog_handle else None
                    ),
                    retain_instance_on_runtime_failure=(
                        retain_instance_on_runtime_failure and watchdog_handle is not None
                    ),
                    retention_watchdog_handoff=watchdog_handoff,
                    forward_hf_token=forward_hf_token,
                    paid_resource_admission_grant=paid_resource_admission_grant,
                    pre_provider_mutation_hook=pre_provider_mutation_hook,
                )
                if watchdog_handle:
                    instance_ids = [
                        int(value) for value in adapter_result.get("vast_instance_ids") or []
                    ]
                    watchdog_close = close_independent_vast_watchdog(
                        job_dir=resolved_job_dir,
                        handle=watchdog_handle,
                        instance_ids=instance_ids,
                        provider_teardown_completed=(
                            adapter_result.get("continuing_spend_from_this_run") is False
                        ),
                        provider_allocation_impossible=(
                            adapter_result.get("provider_create_attempted") is False
                        ),
                    )
                    if watchdog_close.get("status") not in {
                        "provider_terminal",
                        "cancelled_no_allocation",
                        "retained_until_hard_ttl",
                    }:
                        blockers.append("independent_vast_watchdog_not_terminal")
                    if (
                        adapter_result.get("retained_owned") is True
                        and watchdog_close.get("status") == "retained_until_hard_ttl"
                    ):
                        binding = dict(retention_binding or {})
                        try:
                            retained_session = create_retained_session_manifest(
                                job_dir=resolved_job_dir,
                                adapter_result=adapter_result,
                                watchdog_handoff=watchdog_handoff,
                                source_commit=str(binding["source_commit"]),
                                dirty_state_declaration=str(binding["dirty_state_declaration"]),
                                bundle_sha256=str(binding["bundle_sha256"]),
                                authorization_receipt_sha256=str(
                                    binding["authorization_receipt_sha256"]
                                ),
                                image_digest=str(binding["image_digest"]),
                                checkpoint=str(binding["checkpoint"]),
                                checkpoint_revision=str(binding["checkpoint_revision"]),
                            )
                        except (KeyError, OSError, ValueError) as exc:
                            blockers.append(
                                f"successor_retained_session_handoff_failed:{type(exc).__name__}"
                            )
                            retained_ids = [
                                int(value)
                                for value in adapter_result.get("vast_instance_ids") or []
                            ]
                            teardown = (
                                get_render_provider("vast").terminate(str(retained_ids[-1]))
                                if retained_ids
                                else {"status": "blocked"}
                            )
                            retained_session = {
                                "status": "teardown_requested",
                                "reason": "retained_session_handoff_failed",
                                "provider_teardown": teardown,
                            }
                            if teardown.get("status") == "terminated":
                                adapter_result["retained_owned"] = False
                                adapter_result["continuing_spend_from_this_run"] = False
                                watchdog_close = close_independent_vast_watchdog(
                                    job_dir=resolved_job_dir,
                                    handle=watchdog_handle,
                                    instance_ids=retained_ids,
                                    provider_teardown_completed=True,
                                )
                if adapter_result.get("status") != "completed":
                    completion_recovery = _recover_completed_provider_output(
                        job_dir=resolved_job_dir,
                        output_path=resolved_output,
                        provider_output_get_url=provider_output_get_url,
                        provider_bundle_kind=provider_bundle_kind,
                        adapter_result=adapter_result,
                    )
                    if completion_recovery.get("status") != "completed":
                        blockers.extend(
                            str(item) for item in adapter_result.get("blockers") or []
                        )
    else:
        blockers.append("paid_vast_launch_not_authorized_by_runner_flag")

    retained_owned = bool(
        paid_launch_attempted
        and adapter_result
        and adapter_result.get("retained_owned") is True
        and watchdog_close.get("status") == "retained_until_hard_ttl"
    )
    status = (
        "retained_owned"
        if retained_owned
        else "completed"
        if paid_launch_attempted
        and adapter_result
        and (
            adapter_result.get("status") == "completed"
            or completion_recovery.get("status") == "completed"
        )
        and not blockers
        else "blocked"
    )
    output_inspection: dict[str, Any] = dict(
        completion_recovery.get("output_inspection") or {}
    )
    if resolved_output.is_file():
        try:
            output_inspection.update(
                {
                    "output_zip_present": True,
                    "output_zip_size_bytes": resolved_output.stat().st_size,
                }
            )
        except OSError:
            output_inspection["output_zip_present"] = True
    manifest = {
        "schema_version": VAST_WAM_AUTHORIZED_RUNNER_SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "job_dir": str(resolved_job_dir),
        "bundle_path": str(resolved_bundle),
        "bundle_present": resolved_bundle.is_file(),
        "bundle_size_bytes": resolved_bundle.stat().st_size if resolved_bundle.is_file() else 0,
        "output_path": str(resolved_output),
        "output_inspection": output_inspection,
        "token_file": token_status,
        "secret_env_file": str(resolved_secret_env_file),
        "public_base_url_present": bool(_string(public_base_url)),
        "transport_mode": (
            "direct_signed_url_files" if direct_transport_configured else "public_base_url"
        ),
        "direct_signed_url_files": direct_url_status,
        "provider_bundle_url_redacted": redact_provider_url(provider_bundle_url),
        "provider_output_put_url_redacted": redact_provider_url(provider_output_put_url),
        "provider_output_get_url_redacted": redact_provider_url(provider_output_get_url),
        "bundle_url_path": (
            None if direct_transport_configured else _redacted_path(BUNDLE_ROUTE)
        ),
        "output_put_url_path": (
            None if direct_transport_configured else _redacted_path(OUTPUT_ROUTE)
        ),
        "local_staging_self_test_status": self_test.get("status"),
        "staging_manifest_status": staging_guard_manifest.get("status"),
        "local_base_staging_manifest_status": staging_manifest.get("status"),
        "provider_bundle_kind": provider_bundle_kind,
        "authorized_lane_role": (
            "independent_evaluator" if provider_bundle_kind == "evaluator" else "wam"
        ),
        "public_image": public_image,
        "vast_launch_mode": vast_launch_mode,
        "allow_paid_vast_launch": allow_paid_vast_launch,
        "paid_launch_attempted": paid_launch_attempted,
        "independent_watchdog_required": require_independent_watchdog,
        "retention_requested": retain_instance_on_runtime_failure,
        "retained_owned": retained_owned,
        "retained_session": retained_session,
        "independent_watchdog_handoff": watchdog_handoff,
        "independent_watchdog_close": watchdog_close,
        "adapter_result_status": adapter_result.get("status") if adapter_result else None,
        "adapter_result_reason": adapter_result.get("reason") if adapter_result else None,
        "provider_output_completion_recovery": completion_recovery,
        "adapter_result_path": str(resolved_job_dir / "vast_provider_adapter_result.json")
        if adapter_result
        else None,
        "session_budget_ledger": str(resolved_session_budget_ledger),
        "max_live_minutes": max_live_minutes,
        "session_max_live_minutes": session_max_live_minutes,
        "startup_timeout_seconds": startup_timeout_seconds,
        "hardware_constraints": {
            "disk_gb": disk_gb,
            "min_gpu_ram_mb": min_gpu_ram_mb,
            "min_compute_cap": min_compute_cap,
            "max_compute_cap": max_compute_cap,
            "min_reliability": min_reliability,
            "preferred_gpu_keywords": list(preferred_gpu_keywords),
            "prefer_isaac_rt": prefer_isaac_rt,
            "gpu_selection_policy_present": gpu_selection_policy is not None,
        },
        "staging_verification_guard": staging_verification_guard,
        "public_staging_verification_status": public_staging_verification.get("status"),
        "public_staging_verification_path": str(
            resolved_job_dir / "vast_public_staging_verification.json"
        )
        if public_staging_verification.get("status") != "not_requested"
        else None,
        "public_staging_required_consecutive_successes": (
            public_staging_required_consecutive_successes
        ),
        "allow_staging_output_put_probe": allow_staging_output_put_probe,
        "target_spend_guard": target_spend_guard,
        "allow_target_spend_overrun": allow_target_spend_overrun,
        "allow_unverified_public_staging_for_paid_launch": (
            allow_unverified_public_staging_for_paid_launch
        ),
        "adapter_env_gates_required": [VAST_API_GATE_ENV, VAST_INSTANCE_LAUNCH_GATE_ENV],
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
        **_truth_boundaries(),
    }
    write_json(resolved_job_dir / "vast_wam_authorized_runner_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run fail-closed staging and optional paid Vast WAM bundle execution."
    )
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--public-base-url")
    parser.add_argument("--token-file")
    parser.add_argument("--secret-env-file")
    parser.add_argument("--provider-bundle-url-file")
    parser.add_argument("--provider-output-put-url-file")
    parser.add_argument("--provider-output-get-url-file")
    parser.add_argument("--output-path")
    parser.add_argument("--session-budget-ledger")
    parser.add_argument("--allow-paid-vast-launch", action="store_true")
    parser.add_argument("--max-hourly-rate", type=float, default=DEFAULT_MAX_HOURLY_RATE)
    parser.add_argument("--target-spend-usd", type=float, default=DEFAULT_TARGET_SPEND_USD)
    parser.add_argument("--hard-cap-usd", type=float, default=DEFAULT_HARD_CAP_USD)
    parser.add_argument(
        "--allow-target-spend-overrun",
        action="store_true",
        help="Allow paid launch even if the projected request exceeds target spend; hard cap is still enforced by the adapter.",
    )
    parser.add_argument("--max-live-minutes", type=int, default=45)
    parser.add_argument("--session-max-live-minutes", type=int, default=45)
    parser.add_argument("--startup-timeout-seconds", type=int, default=1800)
    parser.add_argument("--no-verify-staging-urls", action="store_true")
    parser.add_argument(
        "--allow-unverified-public-staging-for-paid-launch",
        action="store_true",
        help="Allow a paid launch even when public staging URLs are not probed first. Use only with an independently verified tunnel.",
    )
    parser.add_argument("--public-staging-verify-max-wait-seconds", type=int, default=180)
    parser.add_argument("--public-staging-verify-retry-interval-seconds", type=float, default=5.0)
    parser.add_argument("--public-staging-verify-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--public-staging-required-consecutive-successes", type=int, default=3)
    parser.add_argument("--no-staging-output-put-probe", action="store_true")
    parser.add_argument("--public-image", default=DEFAULT_WAM_PUBLIC_IMAGE)
    parser.add_argument("--vast-launch-mode", default=DEFAULT_WAM_VAST_LAUNCH_MODE)
    parser.add_argument("--vast-template-hash-id")
    parser.add_argument("--use-vast-template-image", action="store_true")
    args = parser.parse_args(argv)
    manifest = run_vast_wam_authorized_runner(
        job_dir=args.job_dir,
        bundle_path=args.bundle_path,
        public_base_url=args.public_base_url,
        token_file=args.token_file,
        secret_env_file=args.secret_env_file,
        provider_bundle_url_file=args.provider_bundle_url_file,
        provider_output_put_url_file=args.provider_output_put_url_file,
        provider_output_get_url_file=args.provider_output_get_url_file,
        output_path=args.output_path,
        session_budget_ledger=args.session_budget_ledger,
        allow_paid_vast_launch=args.allow_paid_vast_launch,
        max_hourly_rate=args.max_hourly_rate,
        target_spend_usd=args.target_spend_usd,
        hard_cap_usd=args.hard_cap_usd,
        allow_target_spend_overrun=args.allow_target_spend_overrun,
        max_live_minutes=args.max_live_minutes,
        session_max_live_minutes=args.session_max_live_minutes,
        startup_timeout_seconds=args.startup_timeout_seconds,
        verify_staging_urls=not args.no_verify_staging_urls,
        allow_unverified_public_staging_for_paid_launch=(
            args.allow_unverified_public_staging_for_paid_launch
        ),
        public_staging_verify_max_wait_seconds=args.public_staging_verify_max_wait_seconds,
        public_staging_verify_retry_interval_seconds=args.public_staging_verify_retry_interval_seconds,
        public_staging_verify_timeout_seconds=args.public_staging_verify_timeout_seconds,
        public_staging_required_consecutive_successes=(
            args.public_staging_required_consecutive_successes
        ),
        allow_staging_output_put_probe=not args.no_staging_output_put_probe,
        public_image=args.public_image,
        vast_launch_mode=args.vast_launch_mode,
        vast_template_hash_id=args.vast_template_hash_id,
        use_vast_template_image=args.use_vast_template_image,
        require_independent_watchdog=True,
    )
    print(
        "[vast-wam-authorized-runner] manifest="
        + str(Path(args.job_dir).resolve() / "vast_wam_authorized_runner_manifest.json")
    )
    print(f"[vast-wam-authorized-runner] status={manifest.get('status')}")
    blockers = manifest.get("blockers") or []
    if blockers:
        print("[vast-wam-authorized-runner] blockers=" + ",".join(str(item) for item in blockers))
    return 0 if manifest.get("status") == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
