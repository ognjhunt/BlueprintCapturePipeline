#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
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


def _request_once(url: str, payload: bytes, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "content-type": "application/json",
            "user-agent": "blueprint-beta-intake-soak/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(1024)
            status = response.status
            ok = 200 <= status < 500
    except urllib.error.HTTPError as error:
        body = error.read(1024)
        status = error.code
        ok = 200 <= status < 500
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


def build_dry_run_report(model: dict[str, Any], concurrency: int, duration_seconds: int) -> dict[str, Any]:
    target = model["beta_target"]
    blockers = _concurrency_blockers(model, concurrency)
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
        "next_required_action": "rerun with --target-url against staging or production intake and archive the JSON report",
    }


def run_soak(url: str, model: dict[str, Any], concurrency: int, requests: int, timeout: float) -> dict[str, Any]:
    blockers = _concurrency_blockers(model, concurrency)
    payload = json.dumps(
        {
            "probe": "beta_intake_soak",
            "model_schema_version": model["schema_version"],
            "capture_limits": model["per_capture_limits"],
        },
        sort_keys=True,
    ).encode("utf-8")
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_request_once, url, payload, timeout) for _ in range(requests)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - started
    latencies = [float(item["latency_ms"]) for item in results if isinstance(item.get("latency_ms"), (int, float))]
    statuses: dict[str, int] = {}
    for item in results:
        key = str(item.get("status") or item.get("error") or "unknown")
        statuses[key] = statuses.get(key, 0) + 1
    ok_count = sum(1 for item in results if item.get("ok") is True)
    return {
        "schema_version": "blueprint.beta_intake_soak_report.v1",
        "status": "passed" if ok_count == len(results) and not blockers else "failed",
        "target_url": url,
        "requests": requests,
        "concurrency": concurrency,
        "target_concurrent_uploaders": _target_concurrency(model),
        "target_concurrency_met": not blockers,
        "blockers": blockers,
        "elapsed_seconds": round(elapsed, 3),
        "ok_count": ok_count,
        "failure_count": len(results) - ok_count,
        "status_counts": statuses,
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "p50": statistics.median(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "claim_boundary": "http_probe_only_no_provider_runtime_or_upload_media_was_executed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or dry-run the beta intake soak harness.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--target-url", help="HTTP endpoint to POST the soak probe to.")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--duration-seconds", type=int)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model = _load_model(Path(args.model))
    default_concurrency = int(model["beta_target"]["target_concurrent_uploaders"])
    default_duration = int(model["beta_target"]["intake_soak_duration_seconds"])
    concurrency = args.concurrency or default_concurrency
    duration_seconds = args.duration_seconds or default_duration
    requests = args.requests or max(concurrency, concurrency * 4)

    if args.dry_run or not args.target_url:
        report = build_dry_run_report(model, concurrency, duration_seconds)
    else:
        report = run_soak(args.target_url, model, concurrency, requests, args.timeout)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"passed", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
