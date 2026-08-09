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
    if cosmos.get("runtime_terminal") is True:
        blockers.append("retention_not_needed_after_terminal_bundle_success")
    elif cosmos.get("server_loaded") is not True:
        blockers.append("retention_cosmos_server_not_proven_loaded")
    return {
        "schema_version": VAST_RETENTION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "retained_owned" if not blockers else "teardown_required",
        "requested": requested,
        "instance_ids": list(instance_ids),
        "watchdog_pid": watchdog.get("watchdog_pid"),
        "watchdog_deadline_epoch": deadline,
        "container_health_proven": startup_probe.get("startup_probe_proven") is True,
        "gpu_health_proven": gpu_sanity.get("gpu_sanity_proven") is True,
        "cosmos_server_loaded": cosmos.get("server_loaded") is True,
        "cosmos_runtime_status": cosmos.get("runtime_status"),
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
    hard_cap_usd: float,
    max_hourly_rate_usd: float | None = None,
) -> dict[str, Any]:
    compute_rate = _number(selected_offer.get("compute_hourly_rate_usd"))
    if compute_rate is None:
        compute_rate = float(selected_offer["hourly_rate_usd"])
    projected_storage_rate = _number(selected_offer.get("storage_hourly_rate_usd"))
    provider_all_in_rate = _number(instance_payload.get("dph_total"))
    provider_storage_rate = _number(instance_payload.get("storage_total_cost"))
    storage_rate = provider_storage_rate
    if storage_rate is None:
        storage_rate = projected_storage_rate
    projected_all_in_rate = compute_rate + (projected_storage_rate or 0.0)
    all_in_rate = max(
        projected_all_in_rate,
        provider_all_in_rate or 0.0,
    )
    selected_offer.update(
        compute_hourly_rate_usd=compute_rate,
        storage_hourly_rate_usd=storage_rate,
        hourly_rate_usd=all_in_rate,
    )
    projected_cost = float(selected_offer["hourly_rate_usd"]) * max_live_minutes / 60.0
    binding = {
        "schema_version": "vast_all_in_cost_binding.v1",
        "generated_at": utc_now_iso(),
        "instance_id": instance_id,
        "disk_gb": disk_gb,
        "compute_hourly_rate_usd": compute_rate,
        "storage_hourly_rate_usd": storage_rate,
        "all_in_hourly_rate_usd": selected_offer["hourly_rate_usd"],
        "max_live_minutes": max_live_minutes,
        "projected_all_in_cost_usd": projected_cost,
        "hard_cap_usd": hard_cap_usd,
        "projected_all_in_cost_under_hard_cap": projected_cost <= hard_cap_usd,
        "max_hourly_rate_usd": max_hourly_rate_usd,
        "all_in_hourly_rate_under_max_hourly": (
            max_hourly_rate_usd is None or all_in_rate <= max_hourly_rate_usd
        ),
        "raw_secret_values_recorded": False,
    }
    write_json(root / "vast_all_in_cost_binding.json", binding)
    return binding


__all__ = [
    "VAST_RETENTION_SCHEMA_VERSION",
    "bind_all_in_cost",
    "record_initial_lifecycle",
    "record_terminal_lifecycle",
    "retention_decision",
]
