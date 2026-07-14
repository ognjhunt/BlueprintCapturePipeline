#!/usr/bin/env python3
"""Measure warm bind/release latency without provisioning provider capacity."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from blueprint_pipeline.common import write_json


def _token(path: Path) -> str:
    source = path.expanduser().resolve()
    if not source.is_file() or source.is_symlink() or source.stat().st_mode & 0o077:
        raise ValueError("warm_bind_probe_token_file_invalid")
    value = source.read_text(encoding="utf-8").strip()
    if len(value.encode()) < 32:
        raise ValueError("warm_bind_probe_token_too_short")
    return value


def _post(base_url: str, route: str, payload: Mapping[str, Any], token: str) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("warm_bind_probe_requires_https_or_loopback")
    target = urllib.parse.urljoin(base_url.rstrip("/") + "/", route.lstrip("/"))
    if urllib.parse.urlparse(target).netloc != parsed.netloc:
        raise ValueError("warm_bind_probe_origin_escape")
    request = urllib.request.Request(
        target,
        method="POST",
        data=json.dumps(dict(payload)).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - origin constrained above
        body = response.read(1024 * 1024 + 1)
    if len(body) > 1024 * 1024:
        raise ValueError("warm_bind_probe_response_too_large")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("warm_bind_probe_response_invalid")
    return value


def run_probe(
    *,
    base_url: str,
    token: str,
    host_image_id: str,
    worker_image_ref: str,
    gpu_family: str,
    samples: int = 5,
    sender: Callable[[str, str, Mapping[str, Any], str], dict[str, Any]] = _post,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if not 1 <= int(samples) <= 50:
        raise ValueError("warm_bind_probe_samples_out_of_range")
    rows: list[float] = []
    blockers: list[str] = []
    for _ in range(int(samples)):
        job_id = "warm-bind-probe-" + uuid.uuid4().hex
        started = clock()
        bound = sender(
            base_url,
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
        elapsed = clock() - started
        if bound.get("status") != "bound_to_ready_worker":
            blockers.append("warm_worker_not_immediately_available")
            break
        rows.append(elapsed)
        released = sender(
            base_url,
            f"/v1/workers/{bound['worker_id']}/release",
            {
                "job_id": job_id,
                "lease_token": bound["lease_token"],
                "healthy": True,
            },
            token,
        )
        if released.get("state") != "ready":
            blockers.append("warm_worker_release_failed")
            break
    ordered = sorted(rows)
    p95 = None
    if ordered:
        p95 = (
            ordered[0]
            if len(ordered) == 1
            else statistics.quantiles(ordered, n=100, method="inclusive")[94]
        )
    if p95 is None or p95 > 10:
        blockers.append("warm_bind_p95_slo_failed")
    return {
        "schema_version": "production_gpu_warm_bind_probe.v1",
        "status": "passed" if not blockers and len(rows) == int(samples) else "failed",
        "release": {
            "host_image_id": host_image_id,
            "worker_image_ref": worker_image_ref,
            "gpu_family": gpu_family,
        },
        "sample_count": len(rows),
        "warm_bind_p95_seconds": p95,
        "warm_bind_slo_seconds": 10,
        "provider_calls_performed": 0,
        "lease_tokens_recorded": False,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--host-image-id", required=True)
    parser.add_argument("--worker-image-ref", required=True)
    parser.add_argument("--gpu-family", required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_probe(
        base_url=args.base_url,
        token=_token(args.token_file),
        host_image_id=args.host_image_id,
        worker_image_ref=args.worker_image_ref,
        gpu_family=args.gpu_family,
        samples=args.samples,
    )
    write_json(args.output, result)
    print(
        json.dumps(
            {"status": result["status"], "warm_bind_p95_seconds": result["warm_bind_p95_seconds"]}
        )
    )
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
