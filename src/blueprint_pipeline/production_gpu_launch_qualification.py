"""Assemble authoritative live warm-pool evidence into the customer launch gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from .common import write_json
from .production_gpu_worker_pool import (
    REQUIRED_READY_CHECKS,
    build_production_startup_readiness,
    release_fingerprint,
)


SCHEMA_VERSION = "production_gpu_launch_qualification.v1"


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def build_launch_qualification(
    *,
    host_image_id: str,
    worker_image_ref: str,
    gpu_family: str,
    min_ready_workers: int,
    registrations: Sequence[Mapping[str, Any]],
    pool_snapshot: Mapping[str, Any],
    bind_probe: Mapping[str, Any],
    replenishment_probe: Mapping[str, Any],
    rollback_drill: Mapping[str, Any],
    current_provider_inventory: Mapping[str, Any],
    teardown_drill: Mapping[str, Any],
    bind_slo_seconds: float = 10.0,
    cold_replenishment_slo_seconds: float = 1800.0,
) -> dict[str, Any]:
    """Build one fail-closed packet; historical teardown never implies live capacity."""

    fingerprint = release_fingerprint(
        host_image_id=host_image_id,
        worker_image_ref=worker_image_ref,
        gpu_family=gpu_family,
    )
    rows = [_mapping(row.get("registration_payload", row)) for row in registrations]
    matching = [
        row
        for row in rows
        if row.get("host_image_id") == host_image_id
        and row.get("worker_image_ref") == worker_image_ref
        and row.get("gpu_family") == gpu_family
        and all(_mapping(row.get("readiness")).get(name) is True for name in REQUIRED_READY_CHECKS)
    ]
    worker_ids = {str(row.get("worker_id") or "") for row in matching}
    worker_ids.discard("")
    pool_counts = _mapping(pool_snapshot.get("worker_counts"))
    pool_ready = int(pool_counts.get("ready") or 0)
    inventory = _mapping(current_provider_inventory)
    inventory_rows = inventory.get("resources")
    inventory_ids = {
        str(_mapping(row).get("id") or _mapping(row).get("instance_id") or "")
        for row in inventory_rows
    } if isinstance(inventory_rows, list) else set()
    inventory_ids.discard("")
    current_live_count = int(inventory.get("live_resource_count") or 0)
    try:
        inventory_age_seconds = time.time() - float(
            cast(Any, inventory.get("observed_at_epoch"))
        )
    except (TypeError, ValueError):
        inventory_age_seconds = float("inf")
    bind = _mapping(bind_probe)
    replenish = _mapping(replenishment_probe)
    rollback = _mapping(rollback_drill)
    teardown = _mapping(teardown_drill)
    registration_ready = len(worker_ids) >= int(min_ready_workers)
    inventory_matches = bool(
        inventory.get("api_confirmed") is True
        and 0 <= inventory_age_seconds <= 120
        and current_live_count >= int(min_ready_workers)
        and worker_ids.issubset(inventory_ids)
    )
    live_evidence = {
        "release_fingerprint": fingerprint,
        "baked_host_image_verified": registration_ready,
        "worker_image_cached_on_host_verified": registration_ready,
        "all_required_ready_checks_observed": registration_ready,
        "ready_worker_count": min(len(worker_ids), pool_ready),
        "current_capacity_deployed": inventory_matches and pool_ready >= int(min_ready_workers),
        "current_provider_live_worker_count": current_live_count,
        "warm_bind_p95_seconds": (
            bind.get("warm_bind_p95_seconds")
            if bind.get("schema_version") == "production_gpu_warm_bind_probe.v1"
            and bind.get("status") == "passed"
            and int(bind.get("sample_count") or 0) >= 20
            else None
        ),
        "cold_replenishment_p95_seconds": replenish.get("cold_replenishment_p95_seconds"),
        "customer_request_provider_calls": bind.get("provider_calls_performed"),
        "async_scale_replenishment_proven": bool(
            replenish.get("status") == "passed"
            and replenish.get("schema_version")
            == "production_gpu_replenishment_probe.v1"
            and int(replenish.get("sample_count") or 0) >= 3
            and replenish.get("customer_request_provider_calls") == 0
            and replenish.get("provisioning_occurred_asynchronously") is True
            and replenish.get("release_fingerprint") == fingerprint
        ),
        "rollback_drill_passed": bool(
            rollback.get("schema_version") == "production_gpu_rollback_drill.v1"
            and rollback.get("status") == "passed"
            and rollback.get("release_fingerprint") == fingerprint
            and rollback.get("candidate_quarantined_before_provider_teardown") is True
            and rollback.get("customer_bind_routed_to_quarantined_worker") is False
        ),
        "provider_inventory_confirmed": inventory_matches,
        "teardown_and_absence_confirmed": bool(
            teardown.get("status") == "PASS"
            and teardown.get("api_confirmed_absent") is True
            and _mapping(teardown.get("teardown_proof")).get("status") == "PASS"
        ),
    }
    gate = build_production_startup_readiness(
        host_image_id=host_image_id,
        worker_image_ref=worker_image_ref,
        gpu_family=gpu_family,
        min_ready_workers=min_ready_workers,
        bind_slo_seconds=bind_slo_seconds,
        cold_replenishment_slo_seconds=cold_replenishment_slo_seconds,
        live_evidence=live_evidence,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": gate["status"],
        "release_fingerprint": fingerprint,
        "distinct_registered_worker_ids": sorted(worker_ids),
        "pool_snapshot": dict(pool_snapshot),
        "current_provider_inventory": inventory,
        "live_evidence": live_evidence,
        "launch_gate": gate,
        "blockers": list(gate["blockers"]),
        "claim_boundary": {
            "live_registration_required": True,
            "current_warm_capacity_required": True,
            "historical_teardown_does_not_prove_current_capacity": True,
            "local_bind_benchmark_is_not_accepted": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("qualification input must be a JSON object")
    result = build_launch_qualification(**raw)
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "customer_launch_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
