"""Measure asynchronous warm-capacity replenishment outside customer requests."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import write_json
from .production_gpu_worker_agent import _post_json, _read_token
from .production_gpu_worker_pool import release_fingerprint


CYCLE_SCHEMA_VERSION = "production_gpu_replenishment_cycle.v1"
PROBE_SCHEMA_VERSION = "production_gpu_replenishment_probe.v1"


def run_replenishment_cycle(
    *,
    pool_base_url: str,
    token: str,
    host_image_id: str,
    worker_image_ref: str,
    gpu_family: str,
    timeout_seconds: float = 1800.0,
    poll_interval_seconds: float = 5.0,
    sender: Callable[[str, str, Mapping[str, Any], str], dict[str, Any]] = _post_json,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    timeout = float(timeout_seconds)
    interval = float(poll_interval_seconds)
    if not 30 <= timeout <= 1800 or not 1 <= interval <= 60:
        raise ValueError("replenishment_probe_timing_out_of_range")
    fingerprint = release_fingerprint(
        host_image_id=host_image_id,
        worker_image_ref=worker_image_ref,
        gpu_family=gpu_family,
    )
    job_id = "replenishment-probe-" + uuid.uuid4().hex
    request = {
        "job_id": job_id,
        "host_image_id": host_image_id,
        "worker_image_ref": worker_image_ref,
        "gpu_family": gpu_family,
        "lease_seconds": 60,
    }
    started = clock()
    first = sender(pool_base_url, "/v1/customer-jobs/bind", request, token)
    blockers: list[str] = []
    if first.get("status") != "queued_waiting_for_warm_worker":
        blockers.append("probe_did_not_observe_initial_capacity_deficit")
    if first.get("customer_request_provider_calls") != 0:
        blockers.append("customer_request_performed_provider_call")
    bound: dict[str, Any] = {}
    while not blockers and clock() - started < timeout:
        sleeper(min(interval, max(0.0, timeout - (clock() - started))))
        candidate = sender(pool_base_url, "/v1/customer-jobs/bind", request, token)
        if candidate.get("status") == "bound_to_ready_worker":
            bound = candidate
            break
    elapsed = clock() - started
    released: dict[str, Any] = {"status": "not_bound"}
    if bound:
        released = sender(
            pool_base_url,
            f"/v1/workers/{bound['worker_id']}/release",
            {
                "job_id": job_id,
                "lease_token": bound.get("lease_token"),
                "healthy": True,
            },
            token,
        )
        if released.get("state") != "ready":
            blockers.append("replenished_worker_release_failed")
    else:
        blockers.append("replenishment_timeout")
    return {
        "schema_version": CYCLE_SCHEMA_VERSION,
        "status": "passed" if not blockers else "failed",
        "release_fingerprint": fingerprint,
        "replenishment_seconds": round(elapsed, 3),
        "customer_request_provider_calls": first.get("customer_request_provider_calls"),
        "provisioning_occurred_asynchronously": first.get("status")
        == "queued_waiting_for_warm_worker",
        "scale_request_id": first.get("scale_request_id"),
        "worker_id": bound.get("worker_id"),
        "blockers": blockers,
    }


def build_replenishment_probe(
    *,
    cycles: Sequence[Mapping[str, Any]],
    release_fingerprint_value: str,
    minimum_samples: int = 3,
    slo_seconds: float = 1800.0,
) -> dict[str, Any]:
    rows = [dict(row) for row in cycles]
    valid = [
        row
        for row in rows
        if row.get("schema_version") == CYCLE_SCHEMA_VERSION
        and row.get("status") == "passed"
        and row.get("release_fingerprint") == release_fingerprint_value
        and row.get("customer_request_provider_calls") == 0
        and row.get("provisioning_occurred_asynchronously") is True
    ]
    values = sorted(float(row["replenishment_seconds"]) for row in valid)
    p95 = (
        None
        if not values
        else values[0]
        if len(values) == 1
        else statistics.quantiles(values, n=100, method="inclusive")[94]
    )
    blockers: list[str] = []
    if len(valid) < int(minimum_samples):
        blockers.append("replenishment_sample_count_insufficient")
    if p95 is None or p95 > float(slo_seconds):
        blockers.append("cold_replenishment_p95_slo_failed")
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "passed" if not blockers else "failed",
        "release_fingerprint": release_fingerprint_value,
        "sample_count": len(valid),
        "minimum_samples": int(minimum_samples),
        "cold_replenishment_p95_seconds": p95,
        "cold_replenishment_slo_seconds": float(slo_seconds),
        "customer_request_provider_calls": 0 if valid else None,
        "provisioning_occurred_asynchronously": bool(valid),
        "cycles": rows,
        "blockers": blockers,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cycle = sub.add_parser("cycle")
    cycle.add_argument("--pool-base-url", required=True)
    cycle.add_argument("--pool-token-file", required=True, type=Path)
    cycle.add_argument("--host-image-id", required=True)
    cycle.add_argument("--worker-image-ref", required=True)
    cycle.add_argument("--gpu-family", required=True)
    cycle.add_argument("--output", required=True, type=Path)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--cycles", required=True, type=Path)
    aggregate.add_argument("--release-fingerprint", required=True)
    aggregate.add_argument("--minimum-samples", type=int, default=3)
    aggregate.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "cycle":
        result = run_replenishment_cycle(
            pool_base_url=args.pool_base_url,
            token=_read_token(args.pool_token_file),
            host_image_id=args.host_image_id,
            worker_image_ref=args.worker_image_ref,
            gpu_family=args.gpu_family,
        )
    else:
        raw = json.loads(args.cycles.read_text(encoding="utf-8"))
        rows = raw.get("cycles") if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            raise SystemExit("cycles must be a JSON list or {cycles:[...]}")
        result = build_replenishment_probe(
            cycles=rows,
            release_fingerprint_value=args.release_fingerprint,
            minimum_samples=args.minimum_samples,
        )
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
