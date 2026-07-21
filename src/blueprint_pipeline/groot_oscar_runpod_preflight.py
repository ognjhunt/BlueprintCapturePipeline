"""Collect read-only RunPod evidence for the GR00T + OSCAR GPU canary.

This command performs no provider mutation.  It verifies an existing network
volume through RunPod REST, checks provider GPU stock through the existing
read-only capacity probe, confirms zero matching billable pods, and binds an
already-running independent watchdog into the spend envelope.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import write_json
from .gpu_render_providers import _runpod_call, get_render_provider
from .groot_oscar_infrastructure_admission import (
    build_runpod_gpu_runtime_evidence,
    build_runpod_network_volume_evidence,
)
from .paid_provider_lane_lease import read_process_argv

SCHEMA_VERSION = "groot_oscar_runpod_preflight_bundle.v1"
WATCHDOG_MODULE = "blueprint_pipeline.groot_oscar_runpod_watchdog"
MODEL_VOLUME_WATCHDOG_MODULE = (
    "blueprint_pipeline.groot_oscar_runpod_volume_watchdog"
)
MODEL_VOLUME_HANDOFF_SCHEMA_VERSION = "groot_oscar_model_volume_watchdog_handoff.v1"
MODEL_VOLUME_WATCHDOG_MARGIN_SECONDS = 60


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("expected_json_object")
    return dict(value)


def _process_alive(pid: Any) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _watchdog_process_matches(
    *,
    argv: Sequence[str],
    pod_name_prefix: str,
    deadline_epoch: float,
    watchdog_out_dir: str,
) -> bool:
    tokens = tuple(str(token) for token in argv)
    try:
        module_index = tokens.index(WATCHDOG_MODULE)
        if module_index <= 0 or tokens[module_index - 1] != "-m":
            return False
        prefix_index = tokens.index("--pod-name-prefix", module_index + 1) + 1
        deadline_index = tokens.index("--deadline-epoch", module_index + 1) + 1
        out_dir_index = tokens.index("--out-dir", module_index + 1) + 1
        observed_prefix = tokens[prefix_index]
        observed_deadline = float(tokens[deadline_index])
    except (ValueError, IndexError):
        return False
    return bool(
        out_dir_index < len(tokens)
        and observed_prefix == pod_name_prefix
        and observed_deadline == deadline_epoch
        and Path(tokens[out_dir_index]).expanduser().resolve()
        == Path(watchdog_out_dir).expanduser().resolve()
    )


def _model_volume_watchdog_process_matches(
    *, argv: Sequence[str], state_path: str
) -> bool:
    tokens = tuple(str(token) for token in argv)
    try:
        module_index = tokens.index(MODEL_VOLUME_WATCHDOG_MODULE)
        state_index = tokens.index("--state", module_index + 2) + 1
    except (ValueError, IndexError):
        return False
    return bool(
        module_index > 0
        and tokens[module_index - 1] == "-m"
        and module_index + 1 < len(tokens)
        and tokens[module_index + 1] == "watchdog"
        and state_index < len(tokens)
        and tokens[state_index] == state_path
    )


def build_watchdog_spend_evidence(
    *,
    watchdog: Mapping[str, Any],
    max_spend_usd: float,
    paid_mutation_authorized: bool,
    clock: Callable[[], float] = time.time,
    process_argv_probe: Callable[[int], Sequence[str]] = read_process_argv,
) -> dict[str, Any]:
    deadline = watchdog.get("deadline_epoch")
    pid = watchdog.get("pid")
    ttl = int(float(deadline) - clock()) if isinstance(deadline, (int, float)) else 0
    process_alive = _process_alive(pid)
    process_identity_verified = False
    pod_name_prefix = str(watchdog.get("pod_name_prefix") or "").strip()
    watchdog_out_dir = str(watchdog.get("watchdog_out_dir") or "").strip()
    if process_alive and isinstance(deadline, (int, float)):
        process_identity_verified = _watchdog_process_matches(
            argv=process_argv_probe(pid),
            pod_name_prefix=pod_name_prefix,
            deadline_epoch=float(deadline),
            watchdog_out_dir=watchdog_out_dir,
        )
    armed = bool(
        watchdog.get("schema_version") == "groot_oscar_runpod_canary_watchdog.v1"
        and watchdog.get("status") == "armed"
        and watchdog.get("independent_process") is True
        and process_alive
        and process_identity_verified
        and ttl > 60
        and bool(watchdog_out_dir)
    )
    return {
        "schema_version": "groot_oscar_runpod_spend_evidence.v1",
        "paid_mutation_authorized": paid_mutation_authorized,
        "max_spend_usd": max_spend_usd,
        "hard_ttl_seconds": ttl,
        "one_resource_limit": True,
        "independent_teardown_watchdog": armed,
        "watchdog_armed_before_allocation": armed,
        "watchdog_pid": pid if process_alive else None,
        "watchdog_process_identity_verified": process_identity_verified,
        "watchdog_deadline_epoch": deadline,
        "watchdog_pod_name_prefix": pod_name_prefix,
        "watchdog_out_dir": watchdog.get("watchdog_out_dir"),
        "raw_secret_values_recorded": False,
    }


def build_model_volume_watchdog_handoff_evidence(
    *,
    handoff: Mapping[str, Any],
    network_volume_id: str,
    canary_watchdog_deadline_epoch: float,
    clock: Callable[[], float] = time.time,
    process_argv_probe: Callable[[int], Sequence[str]] = read_process_argv,
) -> dict[str, Any]:
    """Prove the retained volume watchdog cannot delete cache during canary use."""

    deadline = handoff.get("watchdog_deadline_epoch")
    deadline_value = float(deadline) if isinstance(deadline, (int, float)) else 0.0
    remaining_ttl_seconds = max(0, int(deadline_value - clock()))
    watchdog_pid = handoff.get("watchdog_pid")
    watchdog_state_path = str(handoff.get("watchdog_state_path") or "").strip()
    process_alive = _process_alive(watchdog_pid)
    process_identity_verified = bool(
        process_alive
        and watchdog_state_path
        and _model_volume_watchdog_process_matches(
            argv=process_argv_probe(watchdog_pid),
            state_path=watchdog_state_path,
        )
    )
    blockers: list[str] = []
    if handoff.get("schema_version") != MODEL_VOLUME_HANDOFF_SCHEMA_VERSION:
        blockers.append("model_volume_watchdog_handoff_schema_invalid")
    if handoff.get("status") not in {"volume_ready_watchdog_retained", "verified"}:
        blockers.append("model_volume_watchdog_handoff_not_ready")
    if str(handoff.get("volume_id") or "") != network_volume_id:
        blockers.append("model_volume_watchdog_handoff_volume_mismatch")
    if handoff.get("preparation_pod_absence_confirmed") is not True:
        blockers.append("model_volume_preparation_pod_absence_not_confirmed")
    if handoff.get("volume_presence_confirmed") is not True:
        blockers.append("model_volume_presence_not_confirmed_at_handoff")
    if handoff.get("teardown_owner") != "independent_model_volume_watchdog":
        blockers.append("model_volume_watchdog_teardown_owner_invalid")
    if not process_alive:
        blockers.append("model_volume_watchdog_process_not_alive")
    elif not process_identity_verified:
        blockers.append("model_volume_watchdog_process_identity_invalid")
    if handoff.get("next_owner_must_arm_before_transfer") is not True:
        blockers.append("model_volume_watchdog_transfer_contract_missing")
    if deadline_value < (
        canary_watchdog_deadline_epoch + MODEL_VOLUME_WATCHDOG_MARGIN_SECONDS
    ):
        blockers.append("model_volume_watchdog_ttl_does_not_cover_canary")
    return {
        "schema_version": MODEL_VOLUME_HANDOFF_SCHEMA_VERSION,
        "status": "verified" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "volume_id": network_volume_id,
        "watchdog_deadline_epoch": deadline_value or None,
        "remaining_ttl_seconds": remaining_ttl_seconds,
        "required_canary_deadline_epoch": canary_watchdog_deadline_epoch,
        "required_margin_seconds": MODEL_VOLUME_WATCHDOG_MARGIN_SECONDS,
        "teardown_owner": handoff.get("teardown_owner"),
        "watchdog_pid": watchdog_pid if process_alive else None,
        "watchdog_state_path": watchdog_state_path or None,
        "watchdog_process_identity_verified": process_identity_verified,
        "preparation_pod_absence_confirmed": handoff.get(
            "preparation_pod_absence_confirmed"
        )
        is True,
        "volume_presence_confirmed": handoff.get("volume_presence_confirmed") is True,
        "next_owner_must_arm_before_transfer": handoff.get(
            "next_owner_must_arm_before_transfer"
        )
        is True,
        "provider_lane_handoff": _mapping(handoff.get("provider_lane_handoff")),
        "raw_secret_values_recorded": False,
    }


def collect_runpod_preflight(
    *,
    network_volume_id: str,
    model_cache_path: str,
    gpu_type_id: str,
    required_cuda_version: str,
    name_prefix: str,
    watchdog: Mapping[str, Any],
    model_volume_watchdog_handoff: Mapping[str, Any],
    max_spend_usd: float,
    paid_mutation_authorized: bool,
    volume_getter: Callable[[str], tuple[int, Mapping[str, Any]]],
    capacity_probe: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    inventory_probe: Callable[[str | None], Mapping[str, Any]],
    clock: Callable[[], float] = time.time,
    process_argv_probe: Callable[[int], Sequence[str]] = read_process_argv,
) -> dict[str, Any]:
    status, raw_volume = volume_getter(network_volume_id)
    volume = build_runpod_network_volume_evidence(
        provider_payload=raw_volume if status == 200 else {},
        expected_volume_id=network_volume_id,
        model_cache_path=model_cache_path,
    )
    capacity = dict(
        capacity_probe(
            {
                "cloudType": "SECURE",
                "gpuTypeIds": [gpu_type_id],
                "dataCenterIds": [str(volume.get("data_center_id") or "")],
                "allowedCudaVersions": [required_cuda_version],
                "requires_rtx": True,
            }
        )
    )
    attempt_inventory = dict(inventory_probe(name_prefix))
    inventory = dict(inventory_probe(None))
    observed_at_epoch = clock()
    zero_inventory = bool(
        inventory.get("api_confirmed") is True
        and inventory.get("live_resource_count") == 0
    )
    runtime = build_runpod_gpu_runtime_evidence(
        capacity=capacity,
        gpu_type_id=gpu_type_id,
        data_center_id=str(volume.get("data_center_id") or ""),
        required_cuda_version=required_cuda_version,
        provider_inventory_verified_zero=zero_inventory,
    )
    spend = build_watchdog_spend_evidence(
        watchdog=watchdog,
        max_spend_usd=max_spend_usd,
        paid_mutation_authorized=paid_mutation_authorized,
        clock=lambda: observed_at_epoch,
        process_argv_probe=process_argv_probe,
    )
    model_volume_handoff = build_model_volume_watchdog_handoff_evidence(
        handoff=model_volume_watchdog_handoff,
        network_volume_id=network_volume_id,
        canary_watchdog_deadline_epoch=float(spend.get("watchdog_deadline_epoch") or 0),
        clock=lambda: observed_at_epoch,
        process_argv_probe=process_argv_probe,
    )
    blockers: list[str] = []
    if volume.get("status") != "verified":
        blockers.extend(str(item) for item in volume.get("blockers") or [])
    if runtime.get("provider_api_verified") is not True:
        blockers.append("runpod_gpu_capacity_not_provider_verified")
    if runtime.get("single_gpu_available") is not True:
        blockers.append("runpod_single_gpu_not_available")
    if not zero_inventory:
        blockers.append("runpod_preallocation_inventory_not_zero")
    if spend.get("watchdog_armed_before_allocation") is not True:
        blockers.append("runpod_teardown_watchdog_not_armed_before_allocation")
    if model_volume_handoff.get("status") != "verified":
        blockers.extend(str(item) for item in model_volume_handoff.get("blockers") or [])
    watchdog_prefix = str(spend.get("watchdog_pod_name_prefix") or "").strip()
    if watchdog_prefix and watchdog_prefix != name_prefix:
        blockers.append("runpod_teardown_watchdog_name_prefix_mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "observed_at_epoch": observed_at_epoch,
        "status": "verified" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "volume": volume,
        "runtime": runtime,
        "spend": spend,
        "model_volume_watchdog_handoff": model_volume_handoff,
        "capacity_snapshot": capacity,
        "billable_inventory": inventory,
        "attempt_billable_inventory": attempt_inventory,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-volume-id", required=True)
    parser.add_argument("--model-cache-path", required=True)
    parser.add_argument("--gpu-type-id", required=True)
    parser.add_argument("--required-cuda-version", required=True)
    parser.add_argument("--name-prefix", default="blueprint-groot-oscar-canary-")
    parser.add_argument("--watchdog-evidence", required=True)
    parser.add_argument("--model-volume-watchdog-handoff", required=True)
    parser.add_argument("--max-spend-usd", type=float, required=True)
    parser.add_argument("--paid-mutation-authorized", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    provider = get_render_provider("runpod")
    key = provider._key()  # type: ignore[attr-defined]
    if not key:
        parser.error("RunPod API key file is required")

    def volume_getter(volume_id: str) -> tuple[int, Mapping[str, Any]]:
        status, payload = _runpod_call(
            "GET", f"/networkvolumes/{volume_id}", None, key=key, timeout=30
        )
        return status, _mapping(payload)

    result = collect_runpod_preflight(
        network_volume_id=args.network_volume_id,
        model_cache_path=args.model_cache_path,
        gpu_type_id=args.gpu_type_id,
        required_cuda_version=args.required_cuda_version,
        name_prefix=args.name_prefix,
        watchdog=_read_json(args.watchdog_evidence),
        model_volume_watchdog_handoff=_read_json(args.model_volume_watchdog_handoff),
        max_spend_usd=args.max_spend_usd,
        paid_mutation_authorized=args.paid_mutation_authorized,
        volume_getter=volume_getter,
        capacity_probe=provider.capacity_preflight,
        inventory_probe=lambda prefix: provider.billable_inventory(name_prefix=prefix),
    )
    write_json(Path(args.out), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
