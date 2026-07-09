#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "docs" / "beta_capacity_cost_storage_model_2026-07-08.json"


def _load_model(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _target_concurrency(model: dict[str, Any]) -> int:
    return int(model["beta_target"]["target_concurrent_uploaders"])


def _concurrency_blockers(model: dict[str, Any], concurrency: int) -> list[str]:
    target = _target_concurrency(model)
    if concurrency < target:
        return [f"concurrency_below_beta_target:{concurrency}<target:{target}"]
    return []


def _request_once(
    url: str,
    payload: bytes,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    request_headers = {
        "content-type": "application/json",
        "user-agent": "blueprint-beta-intake-soak/1",
    }
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers=request_headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(1024)
            status = response.status
            ok = 200 <= status < 300
    except urllib.error.HTTPError as error:
        body = error.read(1024)
        status = error.code
        ok = 200 <= status < 300
    except Exception as error:  # noqa: BLE001 - the report needs the concrete failure string.
        return {
            "ok": False,
            "status": None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": type(error).__name__,
            "message": str(error),
        }
    return {
        "ok": ok,
        "status": status,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "sample_body": body.decode("utf-8", errors="replace"),
    }


def build_capacity_cost_summary(model: dict[str, Any]) -> dict[str, Any]:
    cost_model = model.get("cost_per_capture_model")
    if isinstance(cost_model, dict) and cost_model:
        return dict(cost_model)
    target = model["beta_target"]
    budget = model.get("budget_guardrails") or {"cohort_hard_stop_threshold_usd": 5000}
    captures_per_month = int(target["modeled_captures_per_month"])
    hard_stop_usd = float(budget["cohort_hard_stop_threshold_usd"])
    monthly_projection = model.get("monthly_projection") or {
        "total_new_storage_gib_p50": captures_per_month * 4.2
    }
    storage_p50 = float(monthly_projection["total_new_storage_gib_p50"])
    return {
        "schema_version": "blueprint.beta_cost_per_capture_model.v1",
        "status": "derived_from_capacity_model_without_unit_costs",
        "modeled_captures_per_month": captures_per_month,
        "budget_cap_usd_per_capture": round(hard_stop_usd / captures_per_month, 2),
        "budget_cap_usd_per_100_user_month": hard_stop_usd,
        "storage_gib_per_capture_p50": round(storage_p50 / captures_per_month, 3),
        "claim_boundary": "derived_budget_ceiling_only_not_live_billing_or_vendor_pricing",
    }


def _firestore_hotspot_policy(model: dict[str, Any]) -> dict[str, Any]:
    runtime_capacity = model.get("runtime_capacity") if isinstance(model.get("runtime_capacity"), dict) else {}
    policy = (
        runtime_capacity.get("firestore_created_at_hotspot_policy")
        if isinstance(runtime_capacity.get("firestore_created_at_hotspot_policy"), dict)
        else {}
    )
    if policy:
        return dict(policy)
    return {
        "schema_version": "blueprint.firestore_created_at_hotspot_policy.v1",
        "collection": "captures",
        "shard_field": "createdAtShard",
        "monitoring_alert_policy": "google_monitoring_alert_policy.firestore_request_latency",
        "latency_metric": "serviceruntime.googleapis.com/api/request_latencies",
        "p99_alert_threshold_seconds": 0.25,
        "p99_alert_duration_seconds": 300,
        "soak_report_observation_field": "firestore_latency_observation",
        "claim_boundary": "fallback_policy_for_legacy_capacity_model",
    }


def build_firestore_latency_observation(
    model: dict[str, Any],
    *,
    p99_latency_seconds: float | None = None,
    source: str | None = None,
    required: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    policy = _firestore_hotspot_policy(model)
    threshold = float(policy.get("p99_alert_threshold_seconds", 0.25))
    blockers: list[str] = []
    status = "not_provided"
    if p99_latency_seconds is None:
        if required:
            status = "required_missing"
            blockers.append("firestore_latency_observation_missing")
    else:
        status = "passed"
        if p99_latency_seconds > threshold:
            status = "failed"
            blockers.append(f"firestore_p99_latency_exceeded:{p99_latency_seconds:.3f}>{threshold:.3f}")
        if required and not source:
            status = "failed"
            blockers.append("firestore_latency_source_missing")
    return (
        {
            "schema_version": "blueprint.firestore_latency_observation.v1",
            "status": status,
            "required": required,
            "metric": policy.get("latency_metric"),
            "monitoring_alert_policy": policy.get("monitoring_alert_policy"),
            "p99_latency_seconds": p99_latency_seconds,
            "p99_threshold_seconds": threshold,
            "duration_seconds": policy.get("p99_alert_duration_seconds", 300),
            "source": source,
            "claim_boundary": "operator_supplied_firestore_metric_observation_not_live_metric_collection",
        },
        blockers,
    )


def build_dry_run_report(model: dict[str, Any], concurrency: int, duration_seconds: int) -> dict[str, Any]:
    target = model["beta_target"]
    blockers = _concurrency_blockers(model, concurrency)
    firestore_observation, _ = build_firestore_latency_observation(model)
    return {
        "schema_version": "blueprint.beta_intake_soak_report.v1",
        "status": "blocked" if blockers else "dry_run",
        "claim_boundary": "dry_run_only_no_network_requests_were_sent",
        "target_external_users": target["external_users"],
        "modeled_captures_per_month": target["modeled_captures_per_month"],
        "target_concurrent_uploaders": target["target_concurrent_uploaders"],
        "planned_concurrency": concurrency,
        "target_concurrency_met": not blockers,
        "blockers": blockers,
        "planned_duration_seconds": duration_seconds,
        "per_capture_limits": model["per_capture_limits"],
        "cost_per_capture_model": build_capacity_cost_summary(model),
        "firestore_created_at_hotspot_policy": _firestore_hotspot_policy(model),
        "firestore_latency_observation": firestore_observation,
        "next_required_action": "rerun with --target-url against staging or production intake and archive the JSON report",
    }


def _latency_summary(latencies: list[float]) -> dict[str, float | None]:
    if not latencies:
        return {"min": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "min": min(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": max(ordered),
    }


def run_soak(
    url: str,
    model: dict[str, Any],
    concurrency: int,
    request_count: int | None,
    duration_seconds: int,
    timeout: float,
    *,
    headers: dict[str, str] | None = None,
    max_failure_rate: float = 0.0,
    firestore_p99_latency_seconds: float | None = None,
    firestore_latency_source: str | None = None,
    require_firestore_latency: bool = False,
) -> dict[str, Any]:
    concurrency_blockers = _concurrency_blockers(model, concurrency)
    blockers = list(concurrency_blockers)
    if request_count is not None and request_count < concurrency:
        blockers.append(f"request_count_below_concurrency:{request_count}<concurrency:{concurrency}")
    payload = json.dumps(
        {
            "probe": "beta_intake_soak",
            "model_schema_version": model["schema_version"],
            "capture_limits": model["per_capture_limits"],
            "cost_per_capture_model": build_capacity_cost_summary(model),
        },
        sort_keys=True,
    ).encode("utf-8")
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    next_request_index = 0
    lock = threading.Lock()
    deadline = started + max(0, duration_seconds)

    def worker(worker_index: int) -> list[dict[str, Any]]:
        nonlocal next_request_index
        worker_results: list[dict[str, Any]] = []
        while True:
            with lock:
                if request_count is not None and next_request_index >= request_count:
                    break
                if time.perf_counter() >= deadline:
                    break
                next_request_index += 1
                request_index = next_request_index
            result = _request_once(url, payload, timeout, headers=headers)
            result["worker_index"] = worker_index
            result["request_index"] = request_index
            worker_results.append(result)
        return worker_results

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, worker_index) for worker_index in range(concurrency)]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    elapsed = time.perf_counter() - started
    latencies = [float(item["latency_ms"]) for item in results if isinstance(item.get("latency_ms"), (int, float))]
    statuses: dict[str, int] = {}
    for item in results:
        key = str(item.get("status") or item.get("error") or "unknown")
        statuses[key] = statuses.get(key, 0) + 1
    ok_count = sum(1 for item in results if item.get("ok") is True)
    failure_count = len(results) - ok_count
    failure_rate = (failure_count / len(results)) if results else 1.0
    if not results:
        blockers.append("no_soak_requests_executed")
    if failure_rate > max_failure_rate:
        blockers.append(f"failure_rate_exceeded:{failure_rate:.4f}>{max_failure_rate:.4f}")
    firestore_observation, firestore_blockers = build_firestore_latency_observation(
        model,
        p99_latency_seconds=firestore_p99_latency_seconds,
        source=firestore_latency_source,
        required=require_firestore_latency,
    )
    blockers.extend(firestore_blockers)
    return {
        "schema_version": "blueprint.beta_intake_soak_report.v1",
        "status": "passed" if not blockers else "failed",
        "target_url": url,
        "request_count_cap": request_count,
        "requests_executed": len(results),
        "concurrency": concurrency,
        "planned_duration_seconds": duration_seconds,
        "target_concurrent_uploaders": _target_concurrency(model),
        "target_concurrency_met": not concurrency_blockers,
        "blockers": blockers,
        "elapsed_seconds": round(elapsed, 3),
        "ok_count": ok_count,
        "failure_count": failure_count,
        "failure_rate": round(failure_rate, 6),
        "max_failure_rate": max_failure_rate,
        "throughput_requests_per_second": round(len(results) / elapsed, 3) if elapsed > 0 else None,
        "status_counts": statuses,
        "latency_ms": _latency_summary(latencies),
        "cost_per_capture_model": build_capacity_cost_summary(model),
        "firestore_created_at_hotspot_policy": _firestore_hotspot_policy(model),
        "firestore_latency_observation": firestore_observation,
        "sample_results": results[: min(10, len(results))],
        "claim_boundary": "http_probe_only_no_provider_runtime_or_upload_media_was_executed",
    }


def _parse_headers(values: list[str], bearer_token_env: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        if ":" in value:
            key, raw = value.split(":", 1)
        elif "=" in value:
            key, raw = value.split("=", 1)
        else:
            raise ValueError(f"header must use name:value or name=value syntax: {value}")
        key = key.strip()
        raw = raw.strip()
        if not key:
            raise ValueError(f"header name is empty: {value}")
        headers[key] = raw
    if bearer_token_env:
        import os

        token = os.getenv(bearer_token_env, "").strip()
        if not token:
            raise ValueError(f"bearer token env var is empty or unset: {bearer_token_env}")
        headers["authorization"] = f"Bearer {token}"
    return headers


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or dry-run the beta intake soak harness.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--target-url", help="HTTP endpoint to POST the soak probe to.")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--duration-seconds", type=int)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-failure-rate", type=float, default=0.0)
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Additional HTTP header as name:value or name=value. Repeatable.",
    )
    parser.add_argument(
        "--bearer-token-env",
        help="Environment variable containing a bearer token for the target endpoint.",
    )
    parser.add_argument(
        "--firestore-p99-latency-seconds",
        type=float,
        help="Observed Firestore p99 request latency in seconds from the soak/load window.",
    )
    parser.add_argument(
        "--firestore-latency-source",
        help="Dashboard URL, MQL query reference, or archived evidence path for the Firestore latency observation.",
    )
    parser.add_argument(
        "--require-firestore-latency",
        action="store_true",
        help="Fail the live soak report unless a Firestore p99 latency observation and source are supplied.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model = _load_model(Path(args.model))
    default_concurrency = int(model["beta_target"]["target_concurrent_uploaders"])
    default_duration = int(model["beta_target"]["intake_soak_duration_seconds"])
    concurrency = args.concurrency or default_concurrency
    duration_seconds = args.duration_seconds or default_duration
    request_count = args.requests

    if args.dry_run or not args.target_url:
        report = build_dry_run_report(model, concurrency, duration_seconds)
    else:
        report = run_soak(
            args.target_url,
            model,
            concurrency,
            request_count,
            duration_seconds,
            args.timeout,
            headers=_parse_headers(args.header, args.bearer_token_env),
            max_failure_rate=args.max_failure_rate,
            firestore_p99_latency_seconds=args.firestore_p99_latency_seconds,
            firestore_latency_source=args.firestore_latency_source,
            require_firestore_latency=args.require_firestore_latency,
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"passed", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
