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

SCHEMA_VERSION = "groot_oscar_runpod_preflight_bundle.v1"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("expected_json_object")
    return dict(value)


def build_watchdog_spend_evidence(
    *,
    watchdog: Mapping[str, Any],
    max_spend_usd: float,
    paid_mutation_authorized: bool,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    deadline = watchdog.get("deadline_epoch")
    pid = watchdog.get("pid")
    ttl = int(float(deadline) - clock()) if isinstance(deadline, (int, float)) else 0
    process_alive = False
    if type(pid) is int and pid > 0:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            process_alive = False
        except PermissionError:
            process_alive = True
        else:
            process_alive = True
    armed = bool(
        watchdog.get("schema_version")
        in {
            "production_gpu_warm_watchdog.v1",
            "groot_oscar_runpod_canary_watchdog.v1",
        }
        and watchdog.get("status") == "armed"
        and watchdog.get("independent_process") is True
        and process_alive
        and ttl > 60
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
        "watchdog_deadline_epoch": deadline,
        "watchdog_pod_name_prefix": watchdog.get("pod_name_prefix"),
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
    max_spend_usd: float,
    paid_mutation_authorized: bool,
    volume_getter: Callable[[str], tuple[int, Mapping[str, Any]]],
    capacity_probe: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    inventory_probe: Callable[[str], Mapping[str, Any]],
    clock: Callable[[], float] = time.time,
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
                "requires_rtx": True,
            }
        )
    )
    inventory = dict(inventory_probe(name_prefix))
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
        clock=clock,
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
    watchdog_prefix = str(spend.get("watchdog_pod_name_prefix") or "").strip()
    if watchdog_prefix and watchdog_prefix != name_prefix:
        blockers.append("runpod_teardown_watchdog_name_prefix_mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "volume": volume,
        "runtime": runtime,
        "spend": spend,
        "capacity_snapshot": capacity,
        "billable_inventory": inventory,
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
