"""Live control-plane rollback drill for a two-worker production GPU pool."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import write_json
from .production_gpu_worker_agent import _post_json, _read_token
from .production_gpu_worker_pool import release_fingerprint


SCHEMA_VERSION = "production_gpu_rollback_drill.v1"


def run_rollback_drill(
    *,
    pool_base_url: str,
    token: str,
    candidate_worker_id: str,
    host_image_id: str,
    worker_image_ref: str,
    gpu_family: str,
    sender: Callable[[str, str, Mapping[str, Any], str], dict[str, Any]] = _post_json,
) -> dict[str, Any]:
    """Quarantine one candidate and prove the other warm worker receives a bind."""

    candidate = str(candidate_worker_id or "").strip()
    fingerprint = release_fingerprint(
        host_image_id=host_image_id,
        worker_image_ref=worker_image_ref,
        gpu_family=gpu_family,
    )
    quarantine = sender(
        pool_base_url,
        f"/v1/workers/{candidate}/quarantine",
        {"reason": "release_rollback_drill"},
        token,
    )
    job_id = "rollback-drill-" + uuid.uuid4().hex
    binding = sender(
        pool_base_url,
        "/v1/customer-jobs/bind",
        {
            "job_id": job_id,
            "host_image_id": host_image_id,
            "worker_image_ref": worker_image_ref,
            "gpu_family": gpu_family,
            "lease_seconds": 60,
        },
        token,
    )
    routed_worker = str(binding.get("worker_id") or "")
    routed_to_candidate = bool(routed_worker and routed_worker == candidate)
    release: dict[str, Any] = {"status": "not_bound"}
    if binding.get("status") == "bound_to_ready_worker" and routed_worker:
        release = sender(
            pool_base_url,
            f"/v1/workers/{routed_worker}/release",
            {
                "job_id": job_id,
                "lease_token": binding.get("lease_token"),
                "healthy": True,
            },
            token,
        )
    passed = bool(
        quarantine.get("state") == "quarantined"
        and binding.get("status") == "bound_to_ready_worker"
        and routed_worker
        and not routed_to_candidate
        and release.get("state") == "ready"
        and binding.get("customer_request_provider_calls") == 0
    )
    blockers: list[str] = []
    if quarantine.get("state") != "quarantined":
        blockers.append("candidate_not_quarantined")
    if binding.get("status") != "bound_to_ready_worker":
        blockers.append("alternate_warm_worker_not_available")
    if routed_to_candidate:
        blockers.append("customer_bind_routed_to_quarantined_worker")
    if binding.get("status") == "bound_to_ready_worker" and release.get("state") != "ready":
        blockers.append("alternate_worker_release_failed")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "release_fingerprint": fingerprint,
        "candidate_worker_id": candidate,
        "candidate_quarantined_before_provider_teardown": quarantine.get("state")
        == "quarantined",
        "customer_bind_routed_to_quarantined_worker": routed_to_candidate,
        "alternate_worker_id": routed_worker or None,
        "customer_request_provider_calls": binding.get("customer_request_provider_calls"),
        "quarantine": quarantine,
        "binding": binding,
        "release": release,
        "blockers": blockers,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-base-url", required=True)
    parser.add_argument("--pool-token-file", required=True, type=Path)
    parser.add_argument("--candidate-worker-id", required=True)
    parser.add_argument("--host-image-id", required=True)
    parser.add_argument("--worker-image-ref", required=True)
    parser.add_argument("--gpu-family", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_rollback_drill(
        pool_base_url=args.pool_base_url,
        token=_read_token(args.pool_token_file),
        candidate_worker_id=args.candidate_worker_id,
        host_image_id=args.host_image_id,
        worker_image_ref=args.worker_image_ref,
        gpu_family=args.gpu_family,
    )
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
