"""Fail-closed retention decisions and lifecycle evidence for Vast GPU sessions."""

from __future__ import annotations

import json
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .retained_gpu_session_lifecycle import record_retained_gpu_state


VAST_RETENTION_SCHEMA_VERSION = "vast_retained_instance_decision.v1"
NATIVE_TASK_ARENA_WARM_RETENTION_MODE = "native_task_arena_warm_worker"
SCENE_CONFIGURATION_WARM_RETENTION_MODE = (
    "task_evaluation_scene_configuration_diagnostic_warm_worker"
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cosmos_runtime_retention_evidence(video_smoke: Mapping[str, Any]) -> dict[str, Any]:
    """Read bounded, non-secret Cosmos lifecycle evidence from the output zip."""

    if "cosmos_server_loaded" in video_smoke or "cosmos_runtime_status" in video_smoke:
        runtime_status = video_smoke.get("cosmos_runtime_status")
        return {
            "server_loaded": video_smoke.get("cosmos_server_loaded") is True,
            "runtime_terminal": runtime_status == "completed",
            "runtime_status": runtime_status,
            "reason": "embedded",
        }

    archive_value = video_smoke.get("provider_runtime_output_zip_path")
    if not isinstance(archive_value, str) or not archive_value:
        return {"server_loaded": False, "runtime_terminal": False, "reason": "output_zip_missing"}
    archive_path = Path(archive_value).expanduser()
    try:
        if not archive_path.is_file() or archive_path.stat().st_size > 128 * 1024 * 1024:
            return {
                "server_loaded": False,
                "runtime_terminal": False,
                "reason": "output_zip_invalid",
            }
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            required = {"cosmos_server_retention.json", "wam_runtime_result.json"}
            if not required.issubset(names):
                return {
                    "server_loaded": False,
                    "runtime_terminal": False,
                    "reason": "lifecycle_members_missing",
                }
            retention = json.loads(archive.read("cosmos_server_retention.json"))
            runtime = json.loads(archive.read("wam_runtime_result.json"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
        return {
            "server_loaded": False,
            "runtime_terminal": False,
            "reason": "lifecycle_evidence_invalid",
        }
    server_loaded = bool(
        isinstance(retention, Mapping)
        and retention.get("status") == "retained_loaded"
        and retention.get("process_alive") is True
        and retention.get("server_remained_loaded") is True
    )
    runtime_terminal = bool(isinstance(runtime, Mapping) and runtime.get("status") == "completed")
    return {
        "server_loaded": server_loaded,
        "runtime_terminal": runtime_terminal,
        "runtime_status": runtime.get("status") if isinstance(runtime, Mapping) else None,
        "reason": "observed",
    }


def retention_decision(
    *,
    requested: bool,
    watchdog_handoff: Mapping[str, Any] | None,
    instance_ids: Sequence[int],
    startup_probe: Mapping[str, Any],
    gpu_sanity: Mapping[str, Any],
    video_smoke: Mapping[str, Any],
    retention_mode: str = "cosmos_server",
    warm_worker_evidence: Mapping[str, Any] | None = None,
    observed_now_epoch: float | None = None,
) -> dict[str, Any]:
    """Admit retention only for a healthy host with an armed hard-TTL watchdog."""

    watchdog = _mapping(watchdog_handoff)
    now = time.time() if observed_now_epoch is None else float(observed_now_epoch)
    blockers: list[str] = []
    if not requested:
        blockers.append("retention_not_requested")
    if not instance_ids:
        blockers.append("retention_instance_identity_missing")
    if (
        watchdog.get("status") != "armed"
        or watchdog.get("independent_process") is not True
        or watchdog.get("watchdog_armed_before_allocation") is not True
    ):
        blockers.append("retention_independent_watchdog_not_armed")
    deadline = _number(watchdog.get("watchdog_deadline_epoch"))
    if deadline is None or deadline <= now:
        blockers.append("retention_watchdog_deadline_expired_or_missing")
    if (
        startup_probe.get("status") != "completed"
        or startup_probe.get("startup_probe_proven") is not True
    ):
        blockers.append("retention_container_health_not_proven")
    if gpu_sanity.get("status") != "completed" or gpu_sanity.get("gpu_sanity_proven") is not True:
        blockers.append("retention_gpu_health_not_proven")
    cosmos = _cosmos_runtime_retention_evidence(video_smoke)
    warm = _mapping(warm_worker_evidence)
    if retention_mode == NATIVE_TASK_ARENA_WARM_RETENTION_MODE:
        if warm.get("provider_bundle_kind") != "native_task_arena":
            blockers.append("retention_native_task_arena_bundle_kind_invalid")
        if warm.get("runtime_dependency_cache_ready") is not True:
            blockers.append("retention_native_task_arena_dependency_cache_not_ready")
        if warm.get("instance_running") is not True:
            blockers.append("retention_native_task_arena_instance_not_running")
        if warm.get("workload_independent_access_recorded") is not True:
            blockers.append("retention_native_task_arena_access_not_recorded")
        if not isinstance(warm.get("ssh_host"), str) or not warm.get("ssh_host"):
            blockers.append("retention_native_task_arena_ssh_host_missing")
        ssh_port = warm.get("ssh_port")
        if isinstance(ssh_port, bool) or not isinstance(ssh_port, int) or ssh_port <= 0:
            blockers.append("retention_native_task_arena_ssh_port_invalid")
    elif retention_mode == SCENE_CONFIGURATION_WARM_RETENTION_MODE:
        if warm.get("provider_bundle_kind") != "task_evaluation_scene_configuration":
            blockers.append("retention_scene_configuration_bundle_kind_invalid")
        if warm.get("scene_configuration_bundle_downloaded") is not True:
            blockers.append("retention_scene_configuration_bundle_not_downloaded")
        if warm.get("scene_configuration_bundle_sha256_verified") is not True:
            blockers.append(
                "retention_scene_configuration_bundle_sha256_not_verified"
            )
        if warm.get("scene_configuration_entrypoint_started") is not True:
            blockers.append("retention_scene_configuration_entrypoint_not_started")
        if warm.get("scene_configuration_runtime_root_ready") is not True:
            blockers.append("retention_scene_configuration_runtime_root_not_ready")
        if warm.get("scene_configuration_runtime_secrets_scrubbed") is not True:
            blockers.append("retention_scene_configuration_runtime_secrets_not_scrubbed")
        if warm.get("fresh_ssh_runtime_secret_environment_absent") is not True:
            blockers.append(
                "retention_scene_configuration_fresh_ssh_secret_environment_not_absent"
            )
        if warm.get("instance_running") is not True:
            blockers.append("retention_scene_configuration_instance_not_running")
        if warm.get("workload_independent_access_recorded") is not True:
            blockers.append("retention_scene_configuration_access_not_recorded")
        if not isinstance(warm.get("ssh_host"), str) or not warm.get("ssh_host"):
            blockers.append("retention_scene_configuration_ssh_host_missing")
        ssh_port = warm.get("ssh_port")
        if isinstance(ssh_port, bool) or not isinstance(ssh_port, int) or ssh_port <= 0:
            blockers.append("retention_scene_configuration_ssh_port_invalid")
    elif retention_mode == "cosmos_server":
        if cosmos.get("runtime_terminal") is True:
            blockers.append("retention_not_needed_after_terminal_bundle_success")
        elif cosmos.get("server_loaded") is not True:
            blockers.append("retention_cosmos_server_not_proven_loaded")
    else:
        blockers.append("retention_mode_unsupported")
    return {
        "schema_version": VAST_RETENTION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "retained_owned" if not blockers else "teardown_required",
        "requested": requested,
        "retention_mode": retention_mode,
        "instance_ids": list(instance_ids),
        "watchdog_pid": watchdog.get("watchdog_pid"),
        "watchdog_deadline_epoch": deadline,
        "watchdog_out_dir": watchdog.get("watchdog_out_dir"),
        "watchdog_pod_name_prefix": watchdog.get("pod_name_prefix"),
        "watchdog_started_instance_id_path": watchdog.get(
            "started_instance_id_path"
        ),
        "container_health_proven": startup_probe.get("startup_probe_proven") is True,
        "gpu_health_proven": gpu_sanity.get("gpu_sanity_proven") is True,
        "cosmos_server_loaded": cosmos.get("server_loaded") is True,
        "cosmos_runtime_status": cosmos.get("runtime_status"),
        "warm_worker_evidence": dict(warm),
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }


def record_initial_lifecycle(
    root: Path,
    *,
    instance_id: int,
    offer_id: Any,
    image: str,
    watchdog_handoff: Mapping[str, Any] | None,
) -> None:
    watchdog = _mapping(watchdog_handoff)
    evidence = {
        "provider": "vast",
        "provider_instance_id": instance_id,
        "offer_id": offer_id,
        "image": image,
        "watchdog_pid": watchdog.get("watchdog_pid"),
        "watchdog_deadline_epoch": watchdog.get("watchdog_deadline_epoch"),
    }
    for state in ("allocated", "container_starting", "image_pulling"):
        record_retained_gpu_state(root, state, evidence=evidence)


def record_terminal_lifecycle(
    root: Path,
    *,
    instance_id: int,
    decision: Mapping[str, Any],
    video_smoke_completed: bool,
) -> None:
    evidence = {
        "provider": "vast",
        "provider_instance_id": instance_id,
        "watchdog_pid": decision.get("watchdog_pid"),
        "watchdog_deadline_epoch": decision.get("watchdog_deadline_epoch"),
        "container_health_proven": decision.get("container_health_proven"),
        "gpu_health_proven": decision.get("gpu_health_proven"),
        "cosmos_server_loaded": decision.get("cosmos_server_loaded"),
    }
    if decision.get("status") == "retained_owned":
        states = ("healthy", "retained_owned")
    elif video_smoke_completed:
        states = ("healthy", "experiment_running", "terminal_success", "teardown_requested")
    else:
        states = ("terminal_failure", "teardown_requested")
    for state in states:
        record_retained_gpu_state(root, state, evidence=evidence)


def bind_all_in_cost(
    root: Path,
    *,
    selected_offer: dict[str, Any],
    instance_payload: Mapping[str, Any],
    instance_id: int,
    disk_gb: int,
    max_live_minutes: int,
    max_hourly_rate: float | None = None,
    hard_cap_usd: float,
    max_hourly_rate_usd: float | None = None,
    expected_provider_download_bytes: int = 0,
    expected_provider_upload_bytes: int = 0,
) -> dict[str, Any]:
    effective_max_hourly_rate = (
        max_hourly_rate_usd
        if max_hourly_rate_usd is not None
        else max_hourly_rate
    )
    if effective_max_hourly_rate is None or effective_max_hourly_rate <= 0:
        raise ValueError("vast_all_in_cost_max_hourly_rate_invalid")
    compute_rate = _number(selected_offer.get("compute_hourly_rate_usd"))
    if compute_rate is None:
        compute_rate = float(selected_offer["hourly_rate_usd"])
    compute_rate = float(compute_rate)
    # The single-instance endpoint wraps the live row in ``instances``.  Offer
    # search reports the compute ask, while the created row's ``dph_total`` is
    # the authoritative post-allocation rate including the requested disk.  A
    # top-level-only read silently discarded that storage surcharge.
    instance_row = _mapping(instance_payload.get("instances")) or instance_payload
    all_in_rate = _number(instance_row.get("dph_total"))
    storage_rate = _number(instance_row.get("storage_total_cost"))
    download_rate = _number(instance_row.get("inet_down_cost"))
    upload_rate = _number(instance_row.get("inet_up_cost"))
    all_in_rate_observed = all_in_rate is not None and all_in_rate > 0
    if all_in_rate_observed:
        selected_offer.update(
            compute_hourly_rate_usd=compute_rate,
            storage_hourly_rate_usd=storage_rate,
            hourly_rate_usd=all_in_rate,
            provider_download_cost_per_gb_usd=download_rate,
            provider_upload_cost_per_gb_usd=upload_rate,
        )
    transfer_rates_observed = (
        (not expected_provider_download_bytes or download_rate is not None)
        and (not expected_provider_upload_bytes or upload_rate is not None)
    )
    projected_transfer_cost = (
        expected_provider_download_bytes
        / 1_000_000_000.0
        * float(download_rate or 0.0)
        + expected_provider_upload_bytes
        / 1_000_000_000.0
        * float(upload_rate or 0.0)
        if transfer_rates_observed
        else None
    )
    projected_runtime_cost = (
        all_in_rate * max_live_minutes / 60.0 if all_in_rate_observed else None
    )
    projected_cost = (
        projected_runtime_cost + projected_transfer_cost
        if projected_runtime_cost is not None and projected_transfer_cost is not None
        else None
    )
    binding = {
        "schema_version": "vast_all_in_cost_binding.v1",
        "generated_at": utc_now_iso(),
        "instance_id": instance_id,
        "disk_gb": disk_gb,
        "compute_hourly_rate_usd": compute_rate,
        "storage_hourly_rate_usd": storage_rate,
        "all_in_hourly_rate_observed": all_in_rate_observed,
        "all_in_hourly_rate_usd": all_in_rate,
        "provider_download_cost_per_gb_usd": download_rate,
        "provider_upload_cost_per_gb_usd": upload_rate,
        "expected_provider_download_bytes": expected_provider_download_bytes,
        "expected_provider_upload_bytes": expected_provider_upload_bytes,
        "provider_transfer_rates_observed": transfer_rates_observed,
        "projected_provider_transfer_cost_usd": projected_transfer_cost,
        "max_hourly_rate_usd": effective_max_hourly_rate,
        "all_in_hourly_rate_under_max": (
            all_in_rate_observed and all_in_rate <= effective_max_hourly_rate
        ),
        "max_live_minutes": max_live_minutes,
        "projected_runtime_cost_usd": projected_runtime_cost,
        "projected_all_in_cost_usd": projected_cost,
        "hard_cap_usd": hard_cap_usd,
        "all_in_hourly_rate_under_max_hourly": (
            all_in_rate_observed
            and all_in_rate
            <= effective_max_hourly_rate
        ),
        "projected_all_in_cost_under_hard_cap": (
            projected_cost is not None and projected_cost <= hard_cap_usd
        ),
        "raw_secret_values_recorded": False,
    }
    write_json(root / "vast_all_in_cost_binding.json", binding)
    return binding


__all__ = [
    "SCENE_CONFIGURATION_WARM_RETENTION_MODE",
    "VAST_RETENTION_SCHEMA_VERSION",
    "bind_all_in_cost",
    "record_initial_lifecycle",
    "record_terminal_lifecycle",
    "retention_decision",
]
