"""Fail-closed registration agent for an already-warm production GPU worker.

This process does not provision capacity or execute customer-supplied commands.  It
only joins three local, immutable evidence records (host boot, cached release, and
warm application state), registers the exact release with the private worker pool,
and maintains a heartbeat while the worker remains healthy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import safe_outbound_http

WORKER_AGENT_SCHEMA_VERSION = "production_gpu_worker_agent.v1"
HOST_BOOT_EVIDENCE_SCHEMA_VERSION = "production_gpu_host_boot_evidence.v1"
CACHE_EVIDENCE_SCHEMA_VERSION = "production_gpu_cache_evidence.v1"
WARM_SERVE_EVIDENCE_SCHEMA_VERSION = "production_gpu_warm_serve_ready.v2"
POOL_TOKEN_FILE_ENV = "BLUEPRINT_PRODUCTION_GPU_POOL_TOKEN_FILE"
_DIGEST_IMAGE = re.compile(r"\A[^\s@]+@sha256:[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
RUNPOD_GPU_POOL_CLASS = "runpod-secure-l40s-preferred-a40-fallback"
RUNPOD_ALLOWED_GPU_MODELS = frozenset({"NVIDIA L40S", "NVIDIA A40"})


class WorkerEvidenceError(ValueError):
    """A worker attempted to register without complete exact-release evidence."""


def _read_json_record(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise WorkerEvidenceError(f"{label}_evidence_file_invalid")
    source = candidate.resolve()
    if not source.is_file():
        raise WorkerEvidenceError(f"{label}_evidence_file_invalid")
    if source.stat().st_size > 1024 * 1024:
        raise WorkerEvidenceError(f"{label}_evidence_file_too_large")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerEvidenceError(f"{label}_evidence_json_invalid") from exc
    if not isinstance(value, dict):
        raise WorkerEvidenceError(f"{label}_evidence_must_be_object")
    return value


def _required_true(checks: object, names: Sequence[str], *, label: str) -> dict[str, bool]:
    if not isinstance(checks, Mapping):
        raise WorkerEvidenceError(f"{label}_checks_required")
    missing = [name for name in names if checks.get(name) is not True]
    if missing:
        raise WorkerEvidenceError(f"{label}_checks_incomplete:" + ",".join(missing))
    return {name: True for name in names}


def build_worker_registration_payload(
    *,
    worker_id: str,
    provider: str,
    host_image_id: str,
    worker_image_ref: str,
    gpu_family: str,
    endpoint_ref: str,
    launch_session_id: str,
    host_evidence: Mapping[str, Any],
    cache_evidence: Mapping[str, Any],
    warm_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Join independent evidence layers without promoting one layer into another."""

    if not _DIGEST_IMAGE.fullmatch(str(worker_image_ref or "")):
        raise WorkerEvidenceError("worker_image_ref_must_be_digest_pinned")
    expected = {
        "worker_id": str(worker_id or "").strip(),
        "provider": str(provider or "").strip(),
        "host_image_id": str(host_image_id or "").strip(),
        "worker_image_ref": str(worker_image_ref or "").strip(),
        "gpu_family": str(gpu_family or "").strip(),
        "endpoint_ref": str(endpoint_ref or "").strip(),
        "launch_session_id": str(launch_session_id or "").strip(),
    }
    missing = [name for name, value in expected.items() if not value]
    if missing:
        raise WorkerEvidenceError("registration_fields_required:" + ",".join(missing))
    endpoint = urllib.parse.urlparse(expected["endpoint_ref"])
    if (
        endpoint.scheme != "https"
        or not endpoint.hostname
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        raise WorkerEvidenceError("worker_endpoint_ref_requires_credential_free_https")

    if host_evidence.get("schema_version") != HOST_BOOT_EVIDENCE_SCHEMA_VERSION:
        raise WorkerEvidenceError("host_evidence_schema_unsupported")
    if str(host_evidence.get("host_image_id") or "") != expected["host_image_id"]:
        raise WorkerEvidenceError("host_evidence_release_mismatch")
    if expected["provider"] == "runpod":
        actual_gpu_model = str(host_evidence.get("actual_gpu_model") or "").strip()
        if actual_gpu_model not in RUNPOD_ALLOWED_GPU_MODELS:
            raise WorkerEvidenceError("runpod_actual_gpu_model_not_allowed")
        if expected["gpu_family"] not in {actual_gpu_model, RUNPOD_GPU_POOL_CLASS}:
            raise WorkerEvidenceError("runpod_gpu_pool_class_mismatch")
    host_checks = _required_true(
        host_evidence.get("checks"),
        ("host_image_booted", "nvidia_driver_ready", "container_runtime_ready"),
        label="host",
    )

    if cache_evidence.get("schema_version") != CACHE_EVIDENCE_SCHEMA_VERSION:
        raise WorkerEvidenceError("cache_evidence_schema_unsupported")
    if str(cache_evidence.get("worker_image_ref") or "") != expected["worker_image_ref"]:
        raise WorkerEvidenceError("cache_evidence_release_mismatch")
    if not _SHA256.fullmatch(str(cache_evidence.get("model_manifest_digest") or "")):
        raise WorkerEvidenceError("cache_model_manifest_digest_required")
    cache_checks = _required_true(
        cache_evidence.get("checks"),
        ("worker_image_cached", "models_cached_offline"),
        label="cache",
    )

    if warm_evidence.get("schema_version") != WARM_SERVE_EVIDENCE_SCHEMA_VERSION:
        raise WorkerEvidenceError("warm_evidence_schema_unsupported")
    if warm_evidence.get("status") != "serving":
        raise WorkerEvidenceError("warm_evidence_not_serving")
    if str(warm_evidence.get("launch_session_id") or "") != expected["launch_session_id"]:
        raise WorkerEvidenceError("warm_evidence_launch_session_mismatch")
    if str(warm_evidence.get("worker_image_ref") or "") != expected["worker_image_ref"]:
        raise WorkerEvidenceError("warm_evidence_release_mismatch")
    warm_checks = _required_true(
        warm_evidence.get("checks"),
        (
            "isaac_renderer_warm",
            "kitchen_scene_loaded",
            "policy_endpoint_ready",
            "worker_healthcheck_passed",
        ),
        label="warm",
    )

    readiness = {**host_checks, **cache_checks, **warm_checks}
    return {
        "worker_id": expected["worker_id"],
        "provider": expected["provider"],
        "host_image_id": expected["host_image_id"],
        "worker_image_ref": expected["worker_image_ref"],
        "gpu_family": expected["gpu_family"],
        "endpoint_ref": expected["endpoint_ref"],
        "readiness": readiness,
        "agent_evidence": {
            "schema_version": WORKER_AGENT_SCHEMA_VERSION,
            "launch_session_id": expected["launch_session_id"],
            "model_manifest_digest": cache_evidence["model_manifest_digest"],
            "actual_gpu_model": host_evidence.get("actual_gpu_model"),
            "evidence_layers_joined": ["host_boot", "release_cache", "warm_application"],
            "customer_command_executed": False,
        },
    }


def _read_token(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise WorkerEvidenceError("pool_token_file_invalid")
    token_path = candidate.resolve()
    if not token_path.is_file():
        raise WorkerEvidenceError("pool_token_file_invalid")
    if token_path.stat().st_mode & 0o077:
        raise WorkerEvidenceError("pool_token_file_permissions_too_open")
    token = token_path.read_text(encoding="utf-8").strip()
    if len(token.encode("utf-8")) < 32:
        raise WorkerEvidenceError("pool_token_too_short")
    return token


def _post_json(base_url: str, path: str, payload: Mapping[str, Any], token: str) -> dict[str, Any]:
    base = str(base_url or "").rstrip("/")
    policy = safe_outbound_http.service_endpoint_policy(base, max_response_bytes=1024 * 1024)
    parsed_base = urllib.parse.urlparse(base)
    target = urllib.parse.urljoin(base + "/", path.lstrip("/"))
    parsed_target = urllib.parse.urlparse(target)
    if (parsed_target.scheme, parsed_target.netloc) != (parsed_base.scheme, parsed_base.netloc):
        raise WorkerEvidenceError("pool_api_path_origin_escape")
    request = urllib.request.Request(
        target,
        data=json.dumps(dict(payload), sort_keys=True).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        response = safe_outbound_http.open_request(request, policy=policy, timeout_seconds=30)
    except urllib.error.HTTPError as exc:
        raise WorkerEvidenceError(f"pool_api_http_error:{exc.code}") from exc
    if not 200 <= response.status < 300:
        raise WorkerEvidenceError(f"pool_api_http_error:{response.status}")
    try:
        value = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerEvidenceError("pool_api_response_invalid") from exc
    if not isinstance(value, dict):
        raise WorkerEvidenceError("pool_api_response_invalid")
    return value


def run_worker_agent(
    *,
    registration_payload: Mapping[str, Any],
    pool_base_url: str,
    token: str,
    heartbeat_interval_seconds: float = 15.0,
    once: bool = False,
    sender: Callable[[str, str, Mapping[str, Any], str], dict[str, Any]] = _post_json,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Register then heartbeat; any rejected heartbeat terminates fail-closed."""

    interval = float(heartbeat_interval_seconds)
    if not 5 <= interval <= 300:
        raise WorkerEvidenceError("heartbeat_interval_out_of_range")
    registered = sender(pool_base_url, "/v1/workers/ready", registration_payload, token)
    if registered.get("ready_for_customer_binding") is not True:
        raise WorkerEvidenceError("pool_registration_not_ready")
    if once:
        return {"status": "registered", "worker_id": registration_payload["worker_id"]}
    heartbeats = 0
    while True:
        sleeper(interval)
        heartbeat = sender(
            pool_base_url,
            f"/v1/workers/{registration_payload['worker_id']}/heartbeat",
            {},
            token,
        )
        if heartbeat.get("heartbeat_recorded") is not True:
            raise WorkerEvidenceError("pool_heartbeat_rejected")
        heartbeats += 1
        if heartbeats >= 2**31 - 1:  # pragma: no cover - defensive rollover
            heartbeats = 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-base-url", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--host-image-id", required=True)
    parser.add_argument("--worker-image-ref", required=True)
    parser.add_argument("--gpu-family", required=True)
    parser.add_argument("--endpoint-ref", required=True)
    parser.add_argument("--launch-session-id", required=True)
    parser.add_argument("--host-evidence", required=True)
    parser.add_argument("--cache-evidence", required=True)
    parser.add_argument("--warm-evidence", required=True)
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=15)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    token_file = os.getenv(POOL_TOKEN_FILE_ENV, "").strip()
    if not token_file:
        raise SystemExit("production_gpu_pool_token_file_required")
    payload = build_worker_registration_payload(
        worker_id=args.worker_id,
        provider=args.provider,
        host_image_id=args.host_image_id,
        worker_image_ref=args.worker_image_ref,
        gpu_family=args.gpu_family,
        endpoint_ref=args.endpoint_ref,
        launch_session_id=args.launch_session_id,
        host_evidence=_read_json_record(args.host_evidence, label="host"),
        cache_evidence=_read_json_record(args.cache_evidence, label="cache"),
        warm_evidence=_read_json_record(args.warm_evidence, label="warm"),
    )
    result = run_worker_agent(
        registration_payload=payload,
        pool_base_url=args.pool_base_url,
        token=_read_token(token_file),
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        once=args.once,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
