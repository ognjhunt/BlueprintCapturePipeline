"""Retained RunPod volume watchdog and disabled legacy GPU preparation seam.

The supported model-volume command is storage-only and lives in
``groot_oscar_runpod_storage_volume``. This module keeps the independent
deadline watchdog and compatibility admission helpers, but contains no RunPod
Pod create path. The legacy preparation callable always fails before provider
inventory or mutation.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .gpu_render_providers import _runpod_call, get_render_provider
from .groot_oscar_infrastructure_admission import (
    RUNPOD_NETWORK_VOLUME_DATA_CENTER_IDS,
)


SCHEMA_VERSION = "groot_oscar_runpod_model_volume_admission.v1"
RESULT_SCHEMA_VERSION = "groot_oscar_runpod_model_volume_result.v1"
WATCHDOG_SCHEMA_VERSION = "groot_oscar_runpod_model_volume_watchdog.v1"
WATCHDOG_HANDOFF_SCHEMA_VERSION = "groot_oscar_model_volume_watchdog_handoff.v1"
MODEL_CACHE_PATH = "/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1"
MIN_VOLUME_GIB = 30
MAX_VOLUME_GIB = 100
MAX_TTL_SECONDS = 3600
POD_NAME_PREFIX = "blueprint-groot-oscar-canary-model-"
VOLUME_NAME_PREFIX = "blueprint-groot-oscar-models-"
AUTHORIZED_MODEL_VOLUME_GPU_TYPES = frozenset(
    {
        "NVIDIA A40",
        "NVIDIA L40S",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    }
)
_RUNPOD_ID = re.compile(r"\A[A-Za-z0-9._-]{1,256}\Z")
_SECRET_ERROR_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+\S+"),
    re.compile(r"(?i)\b(?:api[_-]?key|authorization|token)\s*[=:]\s*\S+"),
    re.compile(r"\b(?:hf|rpa|rp)_[A-Za-z0-9._-]{8,}\b"),
)


def _safe_provider_error_summary(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    parts: list[str] = []
    for field in ("code", "statusCode", "error", "message", "detail", "title"):
        value = payload.get(field)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            text = str(value).strip()
            if text:
                parts.append(f"{field}={text}")
    if not parts:
        return None
    summary = "; ".join(parts)[:1000]
    for pattern in _SECRET_ERROR_PATTERNS:
        summary = pattern.sub("[REDACTED]", summary)
    return summary


def _single_gpu_capacity_verified(
    *,
    capacity: Mapping[str, Any],
    selected: Mapping[str, Any],
    data_center_id: str,
    required_cuda_version: str,
) -> bool:
    """Legacy admission compatibility only; storage preparation uses no GPU."""

    return bool(
        capacity.get("status") == "available"
        and capacity.get("capacity_confidence") == "advisory"
        and selected.get("capacity_confidence") == "advisory"
        and selected.get("single_gpu_offer_requested") is True
        and selected.get("single_gpu_offer_available") is True
        and selected.get("capacity_data_center_id") == data_center_id
        and required_cuda_version
        in (selected.get("capacity_allowed_cuda_versions") or [])
    )


def build_model_volume_admission(
    *,
    release_image_ref: str,
    data_center_id: str,
    gpu_type_id: str,
    required_cuda_version: str,
    volume_size_gib: int,
    hard_ttl_seconds: int,
    max_spend_usd: float,
    hourly_rate_usd: float,
    volume_hourly_rate_usd: float,
    capacity_verified: bool,
    inventory_verified_zero: bool,
    paid_mutation_authorized: bool,
    watchdog_armed_before_allocation: bool,
) -> dict[str, Any]:
    """Preserve the old evidence parser while permanently blocking allocation."""

    del release_image_ref
    blockers: list[str] = []
    if data_center_id not in RUNPOD_NETWORK_VOLUME_DATA_CENTER_IDS:
        blockers.append("model_volume_data_center_not_network_volume_capable")
    if gpu_type_id not in AUTHORIZED_MODEL_VOLUME_GPU_TYPES:
        blockers.append("model_volume_gpu_type_outside_authorized_campaign")
    if required_cuda_version != "12.8":
        blockers.append("model_volume_cuda_version_not_12_8")
    if type(volume_size_gib) is not int or not MIN_VOLUME_GIB <= volume_size_gib <= MAX_VOLUME_GIB:
        blockers.append("model_volume_size_outside_30_to_100_gib")
    if type(hard_ttl_seconds) is not int or not 60 < hard_ttl_seconds <= MAX_TTL_SECONDS:
        blockers.append("model_volume_ttl_outside_guardrail")
    if (
        not isinstance(volume_hourly_rate_usd, (int, float))
        or volume_hourly_rate_usd <= 0
    ):
        blockers.append("model_volume_storage_hourly_rate_missing")
    elif (
        isinstance(hourly_rate_usd, (int, float))
        and hourly_rate_usd > 0
        and (hourly_rate_usd + volume_hourly_rate_usd)
        * hard_ttl_seconds
        / 3600
        > max_spend_usd
    ):
        blockers.append("model_volume_ttl_cost_exceeds_max_spend")
    if capacity_verified is not True:
        blockers.append("model_volume_single_gpu_capacity_not_verified")
    if inventory_verified_zero is not True:
        blockers.append("model_volume_preallocation_inventory_not_zero")
    if paid_mutation_authorized is not True:
        blockers.append("model_volume_paid_mutation_not_authorized")
    if watchdog_armed_before_allocation is not True:
        blockers.append("model_volume_watchdog_not_armed_before_allocation")
    return {
        "schema_version": SCHEMA_VERSION,
        "resource_class": "model_volume",
        "status": "admitted" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
    }


def _delete_pod(*, key: str, pod_id: str) -> dict[str, Any]:
    """Cleanup-only compatibility; this module cannot create a Pod."""

    delete_http, _ = _runpod_call("DELETE", f"/pods/{pod_id}", None, key=key, timeout=30)
    verify_http = 0
    for _attempt in range(6):
        verify_http, _ = _runpod_call("GET", f"/pods/{pod_id}", None, key=key, timeout=30)
        if verify_http == 404:
            break
        time.sleep(2)
    return {
        "delete_http": delete_http,
        "verify_http": verify_http,
        "provider_absence_confirmed": verify_http == 404,
    }


def _delete_volume(*, key: str, volume_id: str) -> dict[str, Any]:
    delete_http, _ = _runpod_call(
        "DELETE", f"/networkvolumes/{volume_id}", None, key=key, timeout=30
    )
    verify_http = 0
    for _attempt in range(6):
        verify_http, _ = _runpod_call(
            "GET", f"/networkvolumes/{volume_id}", None, key=key, timeout=30
        )
        if verify_http == 404:
            break
        time.sleep(2)
    return {
        "delete_http": delete_http,
        "verify_http": verify_http,
        "provider_absence_confirmed": verify_http == 404,
    }


def _matching_resources(
    *, key: str, pod_prefix: str | None, volume_prefix: str | None
) -> tuple[list[str], list[str], bool]:
    pods_http, pods_payload = _runpod_call("GET", "/pods", None, key=key, timeout=30)
    volumes_http, volumes_payload = _runpod_call(
        "GET", "/networkvolumes", None, key=key, timeout=30
    )
    verified = (
        pods_http == 200
        and isinstance(pods_payload, list)
        and volumes_http == 200
        and isinstance(volumes_payload, list)
    )
    pod_rows = pods_payload if isinstance(pods_payload, list) else []
    volume_rows = volumes_payload if isinstance(volumes_payload, list) else []
    pods = [
        str(row.get("id"))
        for row in pod_rows
        if isinstance(row, Mapping)
        and (pod_prefix is None or str(row.get("name") or "").startswith(pod_prefix))
        and row.get("id")
    ]
    volumes = [
        str(row.get("id"))
        for row in volume_rows
        if isinstance(row, Mapping)
        and (
            volume_prefix is None
            or str(row.get("name") or "").startswith(volume_prefix)
        )
        and row.get("id")
    ]
    return pods, volumes, verified


def watchdog(*, state_path: Path) -> int:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    root = state_path.parent
    deadline = float(state["deadline_epoch"])
    write_json(
        root / "watchdog_armed.json",
        {
            "schema_version": WATCHDOG_SCHEMA_VERSION,
            "status": "armed",
            "pid": os.getpid(),
            "deadline_epoch": deadline,
            "pod_name_prefix": state.get("pod_name_prefix"),
            "volume_name": state.get("volume_name"),
            "watchdog_nonce": state.get("watchdog_nonce"),
            "raw_secret_values_recorded": False,
        },
    )
    handoff = root / "watchdog_handoff.json"
    while time.time() < deadline:
        if handoff.is_file():
            try:
                payload = json.loads(handoff.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                payload = {}
            if isinstance(payload, Mapping) and payload.get("status") in {
                "cancelled_before_provider_allocation",
                "failure_cleanup_provider_terminal",
            }:
                write_json(
                    root / "watchdog_result.json",
                    {
                        "schema_version": WATCHDOG_SCHEMA_VERSION,
                        "status": "handoff_after_supervisor_teardown",
                        "provider_mutations_performed": 0,
                        "raw_secret_values_recorded": False,
                    },
                )
                return 0
        time.sleep(10)
    provider = get_render_provider("runpod")
    key = provider._key()  # type: ignore[attr-defined]
    if not key:
        write_json(
            root / "watchdog_result.json",
            {
                "schema_version": WATCHDOG_SCHEMA_VERSION,
                "status": "teardown_unverified",
                "blockers": ["runpod_api_key_missing"],
                "raw_secret_values_recorded": False,
            },
        )
        return 2
    pod_ids, volume_ids, inventory_verified = _matching_resources(
        key=key,
        pod_prefix=str(state["pod_name_prefix"]),
        volume_prefix=str(state["volume_name"]),
    )
    pod_results = [_delete_pod(key=key, pod_id=item) for item in pod_ids]
    volume_results = [_delete_volume(key=key, volume_id=item) for item in volume_ids]
    final_pods, final_volumes, final_inventory_verified = _matching_resources(
        key=key,
        pod_prefix=str(state["pod_name_prefix"]),
        volume_prefix=str(state["volume_name"]),
    )
    terminal = bool(
        inventory_verified
        and final_inventory_verified
        and not final_pods
        and not final_volumes
    )
    ledger_close: dict[str, Any] = {}
    lease_release: dict[str, Any] = {}
    if terminal:
        try:
            refreshed_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            refreshed_state = {}
        global_pods, global_volumes, global_verified = _matching_resources(
            key=key,
            pod_prefix=None,
            volume_prefix=None,
        )
        terminal = bool(global_verified and not global_pods and not global_volumes)
        if terminal and isinstance(refreshed_state, Mapping):
            from .paid_lane_guard import close_pending_teardown, load_pending_teardowns
            from .paid_provider_lane_lease import (
                build_paid_provider_lane_reconciliation,
                release_transferred_paid_provider_lane_lease,
            )

            pending_path = str(refreshed_state.get("pending_teardown_record") or "")
            lane_handoff = refreshed_state.get("provider_lane_handoff")
            lane_handoff = lane_handoff if isinstance(lane_handoff, Mapping) else {}
            binding = lane_handoff.get("binding")
            binding = binding if isinstance(binding, Mapping) else {}
            if pending_path:
                ledger_close = close_pending_teardown(
                    pending_path,
                    {
                        "status": "PASS",
                        "provider_absence_confirmed": True,
                        "instance_id": refreshed_state.get("volume_id"),
                    },
                )
            reconciliation = build_paid_provider_lane_reconciliation(
                provider="runpod",
                lane=str(binding.get("lane") or "groot_oscar_model_volume"),
                provider_inventory={
                    "api_confirmed": True,
                    "live_resource_count": 0,
                    "resources": [],
                },
                open_pending_teardowns=load_pending_teardowns(),
            )
            lease_path_value = str(lane_handoff.get("lease_path") or "")
            if lease_path_value:
                lease_release = release_transferred_paid_provider_lane_lease(
                    lease_path_value=lease_path_value,
                    teardown_owner_pid=os.getpid(),
                    terminal_reconciliation=reconciliation,
                    reason="retained_volume_watchdog_provider_terminal",
                )
    write_json(
        root / "watchdog_result.json",
        {
            "schema_version": WATCHDOG_SCHEMA_VERSION,
            "status": "provider_terminal" if terminal else "teardown_unverified",
            "pod_terminations": pod_results,
            "volume_deletions": volume_results,
            "provider_absence_confirmed": terminal,
            "pending_teardown_close": ledger_close,
            "provider_lane_lease_release": lease_release,
            "raw_secret_values_recorded": False,
        },
    )
    return 0 if terminal else 2


def _extract_id(payload: Mapping[str, Any]) -> str:
    value = payload.get("id") or payload.get("podId")
    if type(value) is not str or value != value.strip() or not _RUNPOD_ID.fullmatch(value):
        return ""
    return value


def _watchdog_process_running(process: Any) -> bool:
    try:
        return process is not None and process.poll() is None
    except (OSError, ValueError):
        return False


def run_model_volume(*, output_dir: Path, **_legacy_arguments: Any) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    ensure_dir(output)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked_before_allocation",
        "blockers": [
            "legacy_gpu_model_volume_preparation_disabled_use_storage_only_allocator"
        ],
        "provider_mutation_attempted": False,
        "gpu_compute_allocated": False,
        "raw_secret_values_recorded": False,
    }
    write_json(output / "model_volume_result.json", result)
    return result


def launch_detached(*, output_dir: Path, run_arguments: Sequence[str]) -> dict[str, Any]:
    """Compatibility launcher is disabled; use the canonical allocator."""

    del run_arguments
    output = output_dir.expanduser().resolve()
    ensure_dir(output)
    result = {
        "schema_version": "groot_oscar_legacy_model_volume_launcher.v1",
        "status": "blocked",
        "blockers": [
            "legacy_gpu_model_volume_preparation_disabled_use_storage_only_allocator"
        ],
        "provider_mutation_attempted": False,
        "raw_secret_values_recorded": False,
    }
    write_json(output / "legacy_launcher_result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    watch = sub.add_parser("watchdog")
    watch.add_argument("--state", required=True)
    args = parser.parse_args(argv)
    if args.command == "watchdog":
        return watchdog(state_path=Path(args.state))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
