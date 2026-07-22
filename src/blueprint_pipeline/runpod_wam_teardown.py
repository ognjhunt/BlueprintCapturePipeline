"""RunPod stop/delete execution and provider-state verification for WAM polls."""

from __future__ import annotations

import urllib.error
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .core.common import write_json


RunPodRequest = Callable[..., tuple[int, Mapping[str, Any]]]
PodStatusReader = Callable[[Mapping[str, Any]], str]


def verify_runpod_pod_inactive(
    *,
    pod_id: str,
    api_key: str,
    generated_at: str,
    request: RunPodRequest,
    pod_status_reader: PodStatusReader,
    terminal_statuses: Sequence[str],
) -> dict[str, Any]:
    """Query provider state after teardown uncertainty and fail closed if active."""

    try:
        status_code, payload = request(
            method="GET",
            path=f"/pods/{pod_id}",
            api_key=api_key,
            timeout_seconds=20,
        )
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 410}:
            return {
                "status": "completed",
                "generated_at": generated_at,
                "pod_id": pod_id,
                "http_status_code": exc.code,
                "pod_status": "not_found",
                "spend_released": True,
                "blockers": [],
                "raw_secret_values_recorded": False,
            }
        return {
            "status": "blocked",
            "generated_at": generated_at,
            "pod_id": pod_id,
            "http_status_code": exc.code,
            "pod_status": "http_error",
            "spend_released": False,
            "blockers": ["runpod_stop_error_status_probe_http_error"],
            "raw_secret_values_recorded": False,
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "status": "blocked",
            "generated_at": generated_at,
            "pod_id": pod_id,
            "pod_status": "status_probe_error",
            "spend_released": False,
            "blockers": ["runpod_stop_error_status_probe_failed"],
            "probe_error_type": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
    pod_status = pod_status_reader(payload)
    terminal = set(terminal_statuses)
    spend_released = bool(pod_status in terminal or pod_status.upper() in terminal)
    return {
        "status": "completed" if spend_released else "blocked",
        "generated_at": generated_at,
        "pod_id": pod_id,
        "http_status_code": status_code,
        "pod_status": pod_status,
        "spend_released": spend_released,
        "blockers": []
        if spend_released
        else ["runpod_stop_error_pod_still_active_after_status_probe"],
        "raw_secret_values_recorded": False,
    }


def delete_runpod_pod(
    *,
    job_dir: Path,
    pod_id: str,
    api_key: str,
    generated_at: str,
    schema_version: str,
    request: RunPodRequest,
    verify_inactive: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Delete one pod and separately verify provider-terminal state."""

    try:
        status_code, response = request(
            method="DELETE",
            path=f"/pods/{pod_id}",
            api_key=api_key,
            timeout_seconds=30,
        )
        status = "completed" if status_code in {200, 202, 204} else "blocked"
        blockers: list[str] = (
            [] if status == "completed" else ["runpod_delete_pod_unexpected_status"]
        )
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        response = {}
        status = "completed" if exc.code in {404, 410} else "blocked"
        blockers = [] if status == "completed" else ["runpod_delete_pod_http_error"]

    terminal_state_api_confirmed = False
    verified_pod_status: str | None = None
    terminal_state_verification: dict[str, Any] | None = None
    if status_code in {404, 410}:
        terminal_state_api_confirmed = True
        verified_pod_status = "not_found"
    elif status == "completed":
        terminal_state_verification = verify_inactive(
            pod_id=pod_id,
            api_key=api_key,
            generated_at=generated_at,
        )
        probe_status = str(terminal_state_verification.get("pod_status") or "").strip()
        if probe_status and probe_status not in {"http_error", "status_probe_error"}:
            verified_pod_status = probe_status
        terminal_state_api_confirmed = bool(
            terminal_state_verification.get("spend_released")
        )
        if not terminal_state_api_confirmed:
            blockers = [*blockers, "runpod_delete_terminal_state_not_api_confirmed"]
    manifest = {
        "schema_version": schema_version,
        "generated_at": generated_at,
        "status": status,
        "job_dir": str(job_dir),
        "pod_id": pod_id,
        "http_status_code": status_code,
        "response_keys": sorted(response.keys()),
        "blockers": blockers,
        "terminal_state_api_confirmed": terminal_state_api_confirmed,
        "verified_pod_status": verified_pod_status,
        "terminal_state_verification": terminal_state_verification,
        "continuing_spend_from_this_run": status != "completed",
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "runpod_wam_async_delete_manifest.json", manifest)
    return manifest


def stop_runpod_pod(
    *,
    job_dir: Path,
    pod_id: str,
    api_key: str,
    generated_at: str,
    schema_version: str,
    request: RunPodRequest,
    verify_inactive: Callable[..., dict[str, Any]],
    write_stopped_warm_candidate: Callable[..., dict[str, Any]],
    record_warm_candidate: bool = True,
) -> dict[str, Any]:
    """Stop one pod, preserving warm reuse only after an acknowledged stop."""

    verification: dict[str, Any] | None = None
    stop_response_confirmed = False
    try:
        status_code, response = request(
            method="POST",
            path=f"/pods/{pod_id}/stop",
            api_key=api_key,
            timeout_seconds=30,
        )
        status = "completed" if status_code in {200, 202, 204} else "blocked"
        stop_response_confirmed = status == "completed"
        blockers: list[str] = (
            [] if status == "completed" else ["runpod_stop_pod_unexpected_status"]
        )
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        response = {}
        status = "completed" if exc.code in {404, 410} else "blocked"
        blockers = [] if status == "completed" else ["runpod_stop_pod_http_error"]
        if status != "completed":
            verification = verify_inactive(
                pod_id=pod_id,
                api_key=api_key,
                generated_at=generated_at,
            )
            if verification.get("spend_released"):
                status = "completed"
                blockers = []
            else:
                blockers.extend(str(item) for item in verification.get("blockers") or [])
    warm_candidate = (
        write_stopped_warm_candidate(
            job_dir=job_dir,
            pod_id=pod_id,
            generated_at=generated_at,
        )
        if status == "completed" and record_warm_candidate and stop_response_confirmed
        else {
            "status": "not_recorded",
            "reason": "runtime_output_not_successful_for_warm_reuse"
            if status == "completed" and stop_response_confirmed
            else "runpod_stop_completion_verified_without_reusable_stopped_pod"
            if status == "completed"
            else "runpod_stop_not_completed",
            "raw_secret_values_recorded": False,
        }
    )
    manifest = {
        "schema_version": schema_version,
        "generated_at": generated_at,
        "status": status,
        "job_dir": str(job_dir),
        "pod_id": pod_id,
        "http_status_code": status_code,
        "stop_error_verification": verification,
        "response_keys": sorted(response.keys()),
        "blockers": blockers,
        "stopped_pod_preserved_for_warm_reuse": bool(
            status == "completed" and record_warm_candidate and stop_response_confirmed
        ),
        "warm_candidate_recording_requested": bool(record_warm_candidate),
        "warm_candidate": warm_candidate,
        "warm_candidate_path": warm_candidate.get("path"),
        "stop_response_confirmed": stop_response_confirmed,
        "gpu_spend_released_if_provider_honors_stop": status == "completed",
        "stopped_volume_storage_may_continue_billing": bool(
            status == "completed" and stop_response_confirmed
        ),
        "continuing_spend_from_this_run": status != "completed",
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "runpod_wam_async_stop_manifest.json", manifest)
    return manifest
