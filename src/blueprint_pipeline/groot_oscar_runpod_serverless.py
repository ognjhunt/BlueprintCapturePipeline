"""Canonical RunPod FlashBoot active-worker launcher for GR00T + OSCAR.

This is the provider-supported persistent-image alternative used when an
EBS-backed AWS AMI is unavailable.  It never creates an ordinary RunPod Pod.
One private Serverless template and one active-worker endpoint are created,
bound to the verified model-cache volume, and protected by an independent
hard-TTL watchdog before startup or policy work begins.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .gpu_render_providers import _runpod_call
from .groot_oscar_runpod_serverless_campaign_io import (
    cleanup_campaign_storage,
    retrieve_campaign_outputs,
    stage_campaign_inputs,
    validate_campaign_io_evidence,
)
from .groot_oscar_runpod_carrier_volume import (
    RUNTIME_ARCHIVE_ROOTS,
    runtime_bootstrap_shell_prefix,
    verify_carrier_volume_admission,
)
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    mark_pending_teardown_ambiguous,
    open_pending_teardown,
)
from .paid_provider_lane_lease import accept_paid_provider_lane_lease_handoff
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from .production_gpu_campaign_budget import (
    CampaignBudgetExceeded,
    ProductionGpuCampaignBudget,
)
from .runpod_provider_adapter import _http_json


SCHEMA_VERSION = "groot_oscar_runpod_serverless_active_worker.v1"
RUNPOD_REST_API = "https://rest.runpod.io/v1"
RUNPOD_SERVERLESS_API = "https://api.runpod.ai/v2"
DEFAULT_GPU_TYPES = ("NVIDIA A40", "NVIDIA L40S")
SUPPORTED_GPU_TYPES = frozenset(
    {
        "NVIDIA A40",
        "NVIDIA L40S",
        "NVIDIA RTX A6000",
        "NVIDIA RTX 6000 Ada Generation",
    }
)
SUPPORTED_NETWORK_VOLUME_DATA_CENTER_IDS = frozenset(
    {
        "EU-CZ-1",
        "EU-RO-1",
        "EUR-IS-1",
        "EUR-NO-1",
        "US-CA-2",
        "US-IL-1",
        "US-MO-2",
        "US-NC-1",
        "US-NE-1",
        "US-WA-1",
    }
)
# Conservative public Serverless prices as of 2026-07-16. The 48-GiB group
# covers L40/L40S/6000 Ada. We reserve the public flex ceiling even when
# workersMin=1 may qualify for a lower negotiated active-worker price.
ACTIVE_GPU_HOURLY_RATES_USD = {
    "NVIDIA A40": 1.22,
    "NVIDIA L40S": 1.75,
    "NVIDIA RTX A6000": 1.22,
    "NVIDIA RTX 6000 Ada Generation": 1.75,
}
DEFAULT_MAX_HOURLY_RATE_USD = ACTIVE_GPU_HOURLY_RATES_USD["NVIDIA L40S"]
DEFAULT_RESERVATION_SECONDS = 5_215
STARTUP_JOB_TIMEOUT_SECONDS = 1_200
STRICT_JOB_TIMEOUT_SECONDS = 300
STRICT_ATTEMPT_WALL_SECONDS = 480
STRICT_TEARDOWN_BUFFER_SECONDS = 120
CAMPAIGN_JOB_TIMEOUT_SECONDS = 3_500
CAMPAIGN_TEARDOWN_BUFFER_SECONDS = 120
MODEL_CACHE_PATH = "/runpod-volume/.blueprint-model-cache/blueprint-groot-oscar-v1"
CARRIER_CONTAINER_DISK_GIB = 160


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected_json_object:{path}")
    return dict(value)


def _digest_ref(value: Any) -> bool:
    text = str(value or "")
    if "@sha256:" not in text:
        return False
    digest = text.rsplit("@sha256:", 1)[-1]
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def _serverless_volume_path(path: str) -> str:
    prefix = "/workspace/"
    if not path.startswith(prefix):
        raise ValueError("serverless_carrier_volume_path_outside_workspace")
    return "/runpod-volume/" + path.removeprefix(prefix)


def compute_startup_wall_timeout_seconds(*, deadline_epoch: float, now_epoch: float) -> int:
    """Bound startup so strict execution and the full campaign remain reserved."""

    return min(
        STARTUP_JOB_TIMEOUT_SECONDS,
        int(
            deadline_epoch
            - now_epoch
            - STRICT_JOB_TIMEOUT_SECONDS
            - CAMPAIGN_JOB_TIMEOUT_SECONDS
            - CAMPAIGN_TEARDOWN_BUFFER_SECONDS
        ),
    )


def build_template_payload(
    *,
    name: str,
    image_ref: str,
    source_commit: str,
    model_manifest_digest: str,
    carrier_volume_admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not model_manifest_digest.startswith("sha256:") or len(model_manifest_digest) != 71:
        raise ValueError("serverless_model_manifest_digest_invalid")
    payload = {
        "category": "NVIDIA",
        "containerDiskInGb": 80,
        "dockerEntrypoint": ["/opt/blueprint/thin_release_entrypoint.sh"],
        "dockerStartCmd": [
            "/opt/runpod-serverless-venv/bin/python",
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_serverless_worker",
        ],
        "env": {
            "BLUEPRINT_SOURCE_COMMIT": source_commit,
            "BLUEPRINT_WORKER_IMAGE_DIGEST": image_ref,
            "BLUEPRINT_GROOT_OSCAR_EXPECTED_MODEL_MANIFEST_DIGEST": (model_manifest_digest),
            "BLUEPRINT_GROOT_OSCAR_MODEL_CACHE": MODEL_CACHE_PATH,
            "BLUEPRINT_GROOT_OSCAR_OSCAR_CHECKPOINT": f"{MODEL_CACHE_PATH}/oscar",
            "BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT": f"{MODEL_CACHE_PATH}/sonic",
            "BLUEPRINT_ROBOT_EVAL_PROVIDER_RUNTIME": "true",
            "BLUEPRINT_RUNPOD_SERVERLESS_NETWORK_VOLUME_RUNTIME": "true",
        },
        "imageName": image_ref,
        "isPublic": False,
        "isServerless": True,
        "name": f"{name}-template",
        "ports": [],
        "readme": "Private bounded Blueprint GR00T OSCAR Isaac queue worker.",
        "volumeInGb": 0,
        "volumeMountPath": "/runpod-volume",
    }
    if carrier_volume_admission is None:
        return payload
    carrier = verify_carrier_volume_admission(carrier_volume_admission)
    if carrier.get("status") != "verified":
        raise ValueError("serverless_carrier_volume_admission_invalid")
    if carrier.get("source_release_image_ref") != image_ref:
        raise ValueError("serverless_carrier_source_release_mismatch")
    if carrier.get("source_release_commit") != source_commit:
        raise ValueError("serverless_carrier_source_commit_mismatch")
    carrier_manifest_digest = str(carrier.get("model_manifest_digest") or "")
    if carrier_manifest_digest and carrier_manifest_digest != model_manifest_digest:
        raise ValueError("serverless_carrier_model_manifest_mismatch")
    model_root = _serverless_volume_path(str(carrier["model_cache_root"]))
    runtime_archive_path = _serverless_volume_path(str(carrier["runtime_archive_path"]))
    runtime_manifest_path = _serverless_volume_path(str(carrier["runtime_manifest_path"]))
    model_manifest_path = _serverless_volume_path(str(carrier["model_cache_manifest_path"]))
    payload.update(
        {
            "containerDiskInGb": CARRIER_CONTAINER_DISK_GIB,
            "dockerEntrypoint": ["/bin/bash", "-lc"],
            "dockerStartCmd": [
                runtime_bootstrap_shell_prefix()
                + "\nexec /opt/runpod-serverless-venv/bin/python -m "
                "blueprint_pipeline.groot_oscar_runpod_serverless_worker"
            ],
            "imageName": carrier["carrier_image_ref"],
        }
    )
    payload["env"].update(
        {
            "WORK_DIR": "/runpod-volume/.blueprint-serverless-bootstrap",
            "BLUEPRINT_RUNTIME_ARCHIVE_PATH": runtime_archive_path,
            "BLUEPRINT_RUNTIME_MANIFEST_PATH": runtime_manifest_path,
            "BLUEPRINT_RUNTIME_ARCHIVE_ROOTS": ":".join(RUNTIME_ARCHIVE_ROOTS),
            "BLUEPRINT_RUNTIME_ARCHIVE_SHA256": carrier["runtime_archive_sha256"],
            "BLUEPRINT_RUNTIME_MANIFEST_SHA256": carrier["runtime_manifest_sha256"],
            "BLUEPRINT_MODEL_CACHE_ROOT": model_root,
            "BLUEPRINT_MODEL_CACHE_MANIFEST_PATH": model_manifest_path,
            "BLUEPRINT_MODEL_CACHE_MANIFEST_SHA256": carrier["model_manifest_sha256"],
            "BLUEPRINT_GROOT_OSCAR_MODEL_CACHE": model_root,
            "BLUEPRINT_GROOT_OSCAR_OSCAR_CHECKPOINT": f"{model_root}/oscar",
            "BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT": f"{model_root}/sonic",
            "BLUEPRINT_RUNTIME_SOURCE_RELEASE_IMAGE_DIGEST": image_ref,
            "BLUEPRINT_RUNTIME_CARRIER_IMAGE_DIGEST": carrier["carrier_image_ref"],
        }
    )
    return payload


def build_endpoint_payload(
    *,
    name: str,
    template_id: str,
    network_volume_id: str,
    data_center_id: str,
    gpu_type_ids: Sequence[str] = DEFAULT_GPU_TYPES,
) -> dict[str, Any]:
    return {
        "templateId": template_id,
        "computeType": "GPU",
        "dataCenterIds": [data_center_id],
        "executionTimeoutMs": CAMPAIGN_JOB_TIMEOUT_SECONDS * 1000,
        "flashboot": True,
        "gpuCount": 1,
        "gpuTypeIds": list(gpu_type_ids),
        "idleTimeout": 5,
        "minCudaVersion": "12.8",
        "allowedCudaVersions": ["12.8", "12.9", "13.0"],
        "name": f"{name}-endpoint",
        "networkVolumeId": network_volume_id,
        "scalerType": "REQUEST_COUNT",
        "scalerValue": 1,
        "workersMax": 1,
        "workersMin": 1,
    }


def validate_serverless_inputs(
    *,
    release: Mapping[str, Any],
    model_cache: Mapping[str, Any],
    volume: Mapping[str, Any],
    provider_inventory: Mapping[str, Any],
    expected_source_commit: str,
    resource_name_prefix: str,
    reservation_seconds: int,
    initial_spent_usd: float,
    initial_gpu_seconds: int,
    max_hourly_rate_usd: float = DEFAULT_MAX_HOURLY_RATE_USD,
    gpu_type_ids: Sequence[str] = DEFAULT_GPU_TYPES,
    carrier_volume_admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    image_ref = str(release.get("resolved_digest_ref") or release.get("release_image_ref") or "")
    thin = release.get("thin_release_contract")
    thin = dict(thin) if isinstance(thin, Mapping) else {}
    if not _digest_ref(image_ref):
        blockers.append("serverless_release_image_not_digest_pinned")
    if thin.get("status") != "passed" or thin.get("models_externalized") is not True:
        blockers.append("serverless_thin_release_contract_not_passed")
    serverless_worker = release.get("serverless_worker_contract")
    serverless_worker = dict(serverless_worker) if isinstance(serverless_worker, Mapping) else {}
    if not (
        serverless_worker.get("status") == "passed"
        and serverless_worker.get("worker_source_packaged") is True
        and serverless_worker.get("worker_command_packaged") is True
        and serverless_worker.get("runpod_sdk_exactly_pinned") is True
    ):
        blockers.append("serverless_worker_not_proven_in_release")
    if release.get("runnable_platform") != "linux/amd64":
        blockers.append("serverless_release_platform_invalid")
    release_source_commit = str(release.get("source_commit") or "")
    if len(expected_source_commit) != 40 or any(
        char not in "0123456789abcdef" for char in expected_source_commit
    ):
        blockers.append("serverless_expected_source_commit_invalid")
    if release_source_commit != expected_source_commit:
        blockers.append("serverless_release_source_commit_mismatch")
    if (
        model_cache.get("status") != "passed"
        or model_cache.get("schema_version") != "groot_oscar_external_model_cache_verification.v2"
    ):
        blockers.append("serverless_model_cache_not_verified")
    manifest_digest = str(model_cache.get("model_manifest_digest") or "")
    if not manifest_digest.startswith("sha256:") or len(manifest_digest) != 71:
        blockers.append("serverless_model_manifest_digest_invalid")
    if str(model_cache.get("provider_volume_id") or "") != str(volume.get("id") or ""):
        blockers.append("serverless_model_cache_volume_mismatch")
    if volume.get("provider_api_verified") is not True:
        blockers.append("serverless_network_volume_not_provider_verified")
    volume_data_center_id = str(volume.get("data_center_id") or "")
    if volume_data_center_id not in SUPPORTED_NETWORK_VOLUME_DATA_CENTER_IDS:
        blockers.append("serverless_network_volume_datacenter_invalid")
    selected_gpu_types = tuple(str(item) for item in gpu_type_ids)
    if (
        not selected_gpu_types
        or len(set(selected_gpu_types)) != len(selected_gpu_types)
        or any(item not in SUPPORTED_GPU_TYPES for item in selected_gpu_types)
        or any("H100" in item.upper() for item in selected_gpu_types)
    ):
        blockers.append("serverless_gpu_types_invalid_or_h100_disallowed")
    if provider_inventory.get("api_confirmed") is not True:
        blockers.append("serverless_provider_inventory_unverified")
    if provider_inventory.get("matching_compute_count") != 0:
        blockers.append("serverless_matching_compute_already_present")
    if provider_inventory.get("matching_template_count") != 0:
        blockers.append("serverless_matching_template_already_present")
    if not resource_name_prefix.startswith("blueprint-groot-oscar-serverless-"):
        blockers.append("serverless_resource_prefix_invalid")
    remaining_wall_seconds = 21_000 - initial_gpu_seconds
    if reservation_seconds != remaining_wall_seconds:
        blockers.append("serverless_campaign_reservation_must_equal_remaining_wall_cap")
    minimum_reservation_seconds = (
        STRICT_JOB_TIMEOUT_SECONDS
        + CAMPAIGN_JOB_TIMEOUT_SECONDS
        + CAMPAIGN_TEARDOWN_BUFFER_SECONDS
        + 1
    )
    if reservation_seconds < minimum_reservation_seconds:
        blockers.append("serverless_remaining_wall_cap_cannot_preserve_campaign")
    if initial_gpu_seconds + reservation_seconds > 21_000:
        blockers.append("serverless_campaign_wall_cap_exceeded")
    selected_rate_ceiling = max(
        (ACTIVE_GPU_HOURLY_RATES_USD.get(item, 0.0) for item in selected_gpu_types),
        default=0.0,
    )
    if initial_spent_usd + max_hourly_rate_usd * reservation_seconds / 3600 > 20:
        blockers.append("serverless_campaign_spend_cap_exceeded")
    if max_hourly_rate_usd != selected_rate_ceiling:
        blockers.append("serverless_campaign_hourly_rate_must_equal_gpu_ceiling")
    carrier: dict[str, Any] = {}
    if carrier_volume_admission is None:
        blockers.append("serverless_carrier_volume_admission_required")
    else:
        carrier = verify_carrier_volume_admission(carrier_volume_admission)
        if carrier.get("status") != "verified":
            blockers.extend(list(carrier.get("blockers") or []))
            blockers.append("serverless_carrier_volume_not_verified")
        if carrier.get("network_volume_id") != str(volume.get("id") or ""):
            blockers.append("serverless_carrier_volume_id_mismatch")
        if carrier.get("data_center_id") != volume_data_center_id:
            blockers.append("serverless_carrier_volume_datacenter_mismatch")
        if carrier.get("source_release_image_ref") != image_ref:
            blockers.append("serverless_carrier_source_release_mismatch")
        if carrier.get("source_release_commit") != release_source_commit:
            blockers.append("serverless_carrier_source_commit_mismatch")
        carrier_manifest_digest = str(carrier.get("model_manifest_digest") or "")
        if carrier_manifest_digest and carrier_manifest_digest != manifest_digest:
            blockers.append("serverless_carrier_model_manifest_mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "admitted" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "release_image_ref": image_ref or None,
        "source_commit": release_source_commit or None,
        "expected_source_commit": expected_source_commit,
        "model_manifest_digest": manifest_digest or None,
        "network_volume_id": volume.get("id"),
        "data_center_id": volume.get("data_center_id"),
        "resource_name_prefix": resource_name_prefix,
        "reservation_seconds": reservation_seconds,
        "maximum_startup_seconds": max(
            0,
            reservation_seconds
            - STRICT_JOB_TIMEOUT_SECONDS
            - CAMPAIGN_JOB_TIMEOUT_SECONDS
            - CAMPAIGN_TEARDOWN_BUFFER_SECONDS,
        ),
        "gpu_type_ids": list(selected_gpu_types),
        "carrier_volume_verified": bool(carrier) and carrier.get("status") == "verified",
        "carrier_image_ref": carrier.get("carrier_image_ref"),
        "flashboot": True,
        "workers_min": 1,
        "workers_max": 1,
        "ordinary_runpod_pod_create_allowed": False,
        "semantic_task_success_proven": False,
    }


def collect_inventory(*, api_key: str, resource_name_prefix: str) -> dict[str, Any]:
    endpoint_http, endpoints = _runpod_call("GET", "/endpoints", None, key=api_key, timeout=30)
    template_http, templates = _runpod_call("GET", "/templates", None, key=api_key, timeout=30)
    pod_http, pods = _runpod_call("GET", "/pods", None, key=api_key, timeout=30)
    endpoint_rows = endpoints if isinstance(endpoints, list) else []
    template_rows = templates if isinstance(templates, list) else []
    pod_rows = pods if isinstance(pods, list) else []
    matching_endpoints = [
        row
        for row in endpoint_rows
        if isinstance(row, Mapping) and str(row.get("name") or "").startswith(resource_name_prefix)
    ]
    matching_templates = [
        row
        for row in template_rows
        if isinstance(row, Mapping) and str(row.get("name") or "").startswith(resource_name_prefix)
    ]
    matching_pods = [
        row
        for row in pod_rows
        if isinstance(row, Mapping) and str(row.get("name") or "").startswith(resource_name_prefix)
    ]
    return {
        "status": "observed" if {endpoint_http, template_http, pod_http} == {200} else "blocked",
        "api_confirmed": endpoint_http == template_http == pod_http == 200,
        "matching_compute_count": len(matching_endpoints) + len(matching_pods),
        "matching_endpoint_count": len(matching_endpoints),
        "matching_pod_count": len(matching_pods),
        "matching_template_count": len(matching_templates),
        "http": {
            "endpoints": endpoint_http,
            "templates": template_http,
            "pods": pod_http,
        },
        "raw_provider_response_recorded": False,
    }


def validate_model_volume_handoff_binding(
    binding: Mapping[str, Any], *, volume_id: str
) -> list[str]:
    """Bind the paid-lane handoff to the exact admitted model volume."""

    blockers: list[str] = []
    if str(binding.get("provider") or "").strip().lower() != "runpod":
        blockers.append("serverless_model_volume_handoff_provider_mismatch")
    if str(binding.get("lane") or "") != "groot_oscar_model_volume":
        blockers.append("serverless_model_volume_handoff_lane_mismatch")
    if not volume_id or str(binding.get("volume_id") or "") != volume_id:
        blockers.append("serverless_model_volume_handoff_volume_mismatch")
    return blockers


def _write_watchdog_state(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, dict(payload))
    os.chmod(path, 0o600)


def _arm_watchdog(
    *, output_dir: Path, state_path: Path, resource_name_prefix: str, deadline: float
) -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    command = [
        sys.executable,
        "-m",
        "blueprint_pipeline.groot_oscar_runpod_serverless_watchdog",
        "--state",
        str(state_path),
        "--resource-name-prefix",
        resource_name_prefix,
        "--deadline-epoch",
        str(deadline),
    ]
    log = (output_dir / "watchdog.log").open("ab")
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    armed_path = output_dir / "watchdog_armed.json"
    for _ in range(100):
        if armed_path.is_file():
            armed = _read(armed_path)
            if armed.get("pid") == process.pid and armed.get("status") == "armed":
                return process, armed
        if process.poll() is not None:
            break
        time.sleep(0.1)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    raise RuntimeError("serverless_watchdog_did_not_arm")


def _request_teardown(
    output_dir: Path,
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    write_json(output_dir / "teardown.request.json", {"status": "requested"})
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    proof_path = output_dir / "serverless_teardown_proof.json"
    return _read(proof_path) if proof_path.is_file() else {"status": "BLOCKED"}


def _submit_job(
    *,
    api_key: str,
    endpoint_id: str,
    operation: str,
    timeout_seconds: int,
    job_input: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    bounded_input = {"operation": operation}
    if job_input:
        bounded_input.update(dict(job_input))
    return _http_json(
        url=f"{RUNPOD_SERVERLESS_API}/{endpoint_id}/run",
        payload={
            "input": bounded_input,
            "policy": {
                "executionTimeout": timeout_seconds * 1000,
                "ttl": max(timeout_seconds + 120, STRICT_ATTEMPT_WALL_SECONDS) * 1000,
                "lowPriority": False,
            },
        },
        api_key=api_key,
        timeout_seconds=30,
    )


def _poll_job(
    *,
    api_key: str,
    endpoint_id: str,
    job_id: str,
    operation_label: str,
    wall_timeout_seconds: int,
    poll_interval_seconds: float = 5,
) -> dict[str, Any]:
    started = time.monotonic()
    trace: list[dict[str, Any]] = []
    last_status = ""
    while time.monotonic() - started < wall_timeout_seconds:
        remaining = wall_timeout_seconds - (time.monotonic() - started)
        http, payload = _http_json(
            url=f"{RUNPOD_SERVERLESS_API}/{endpoint_id}/status/{job_id}",
            payload=None,
            api_key=api_key,
            timeout_seconds=max(1, min(15, int(remaining))),
            method="GET",
        )
        status = str(payload.get("status") or "")
        if status != last_status or not trace:
            elapsed = round(time.monotonic() - started, 3)
            trace.append(
                {
                    "elapsed_seconds": elapsed,
                    "http": http,
                    "status": status,
                    "delay_time_ms": payload.get("delayTime"),
                    "execution_time_ms": payload.get("executionTime"),
                }
            )
            print(
                f"[runpod-serverless] {operation_label} status={status or 'UNKNOWN'} "
                f"elapsed={elapsed}s",
                file=sys.stderr,
                flush=True,
            )
            last_status = status
        if status in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            output = payload.get("output")
            return {
                "status": status,
                "provider_job_id": job_id,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "phase_trace": trace,
                "output": dict(output) if isinstance(output, Mapping) else {},
                "raw_secret_values_recorded": False,
            }
        remaining_after_request = wall_timeout_seconds - (time.monotonic() - started)
        if remaining_after_request > 0:
            time.sleep(min(poll_interval_seconds, remaining_after_request))
    return {
        "status": "WALL_TIMEOUT",
        "provider_job_id": job_id,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "phase_trace": trace,
        "output": {},
        "raw_secret_values_recorded": False,
    }


def _record_job_execution_state(state: dict[str, Any], result: Mapping[str, Any]) -> None:
    """Persist whether RunPod ever assigned a worker before teardown settles."""

    trace = result.get("phase_trace")
    trace = list(trace) if isinstance(trace, list) else []
    statuses = [str(row.get("status") or "") for row in trace if isinstance(row, Mapping)]
    execution_times = [row.get("execution_time_ms") for row in trace if isinstance(row, Mapping)]
    execution_observed = any(
        status in {"IN_PROGRESS", "RUNNING", "COMPLETED", "FAILED"} for status in statuses
    ) or any(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        for value in execution_times
    )
    previous = state.get("serverless_job_execution")
    previous = dict(previous) if isinstance(previous, Mapping) else {}
    state["serverless_job_execution"] = {
        "worker_execution_observed": bool(
            previous.get("worker_execution_observed") is True or execution_observed
        ),
        "provider_job_id": result.get("provider_job_id"),
        "provider_job_status": (statuses[-1] if statuses else str(result.get("status") or "")),
        "poll_result_status": result.get("status"),
        "execution_time_ms": next(
            (value for value in reversed(execution_times) if value is not None), None
        ),
        "phase_statuses": statuses,
    }


def run_active_worker(
    *,
    output_dir: str | Path,
    release_evidence: str | Path,
    model_cache_evidence: str | Path,
    watchdog_handoff_evidence: str | Path,
    api_key_file: str | Path,
    campaign_io_evidence: str | Path,
    runpod_s3_access_key_file: str | Path,
    runpod_s3_secret_key_file: str | Path,
    resource_name_prefix: str,
    expected_source_commit: str,
    execute: bool,
    campaign_budget_ledger: str | Path,
    initial_spent_usd: float,
    initial_gpu_seconds: int,
    reservation_seconds: int = DEFAULT_RESERVATION_SECONDS,
    max_hourly_rate_usd: float = DEFAULT_MAX_HOURLY_RATE_USD,
    carrier_volume_admission: str | Path | None = None,
    gpu_type_ids: Sequence[str] = DEFAULT_GPU_TYPES,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    key_path = Path(api_key_file).expanduser().resolve()
    if key_path.is_symlink() or not key_path.is_file() or key_path.stat().st_mode & 0o077:
        raise ValueError("runpod_api_key_file_unsafe")
    api_key = key_path.read_text(encoding="utf-8").strip()
    release = _read(release_evidence)
    model_cache = _read(model_cache_evidence)
    carrier_admission = (
        _read(carrier_volume_admission) if carrier_volume_admission is not None else None
    )
    handoff_evidence = _read(watchdog_handoff_evidence)
    handoff = handoff_evidence.get("provider_lane_handoff")
    handoff = dict(handoff) if isinstance(handoff, Mapping) else {}
    binding = handoff.get("binding")
    binding = dict(binding) if isinstance(binding, Mapping) else {}
    volume_id = str(model_cache.get("provider_volume_id") or "")
    volume_http, volume_payload = _runpod_call(
        "GET", f"/networkvolumes/{volume_id}", None, key=api_key, timeout=30
    )
    volume = {
        "id": volume_payload.get("id") if isinstance(volume_payload, Mapping) else None,
        "data_center_id": (
            volume_payload.get("dataCenterId") if isinstance(volume_payload, Mapping) else None
        ),
        "provider_api_verified": volume_http == 200,
    }
    inventory = collect_inventory(api_key=api_key, resource_name_prefix=resource_name_prefix)
    admission = validate_serverless_inputs(
        release=release,
        model_cache=model_cache,
        volume=volume,
        provider_inventory=inventory,
        expected_source_commit=expected_source_commit,
        resource_name_prefix=resource_name_prefix,
        reservation_seconds=reservation_seconds,
        initial_spent_usd=initial_spent_usd,
        initial_gpu_seconds=initial_gpu_seconds,
        max_hourly_rate_usd=max_hourly_rate_usd,
        gpu_type_ids=gpu_type_ids,
        carrier_volume_admission=carrier_admission,
    )
    campaign_io = validate_campaign_io_evidence(
        campaign_io_evidence,
        source_commit=str(admission.get("source_commit") or ""),
        image_ref=str(admission.get("release_image_ref") or ""),
        model_manifest_digest=str(admission.get("model_manifest_digest") or ""),
        volume_id=volume_id,
        data_center_id=str(volume.get("data_center_id") or ""),
    )
    if campaign_io.get("status") != "passed":
        admission["blockers"] = sorted(
            {
                *list(admission.get("blockers") or []),
                *list(campaign_io.get("blockers") or []),
            }
        )
        admission["status"] = "blocked"
    handoff_blockers = validate_model_volume_handoff_binding(binding, volume_id=volume_id)
    if handoff_blockers:
        admission["blockers"] = sorted({*list(admission.get("blockers") or []), *handoff_blockers})
        admission["status"] = "blocked"
    write_json(output / "serverless_admission.json", admission)
    write_json(output / "campaign_io_admission.json", campaign_io)
    if admission.get("status") != "admitted":
        write_json(
            output / "serverless_request_shapes.json",
            {
                "status": "blocked_before_request_shape",
                "template": None,
                "endpoint": None,
            },
        )
        return {
            **admission,
            "status": "blocked",
            "provider_mutations_performed": 0,
        }
    template_payload = build_template_payload(
        name=f"{resource_name_prefix}template",
        image_ref=str(admission.get("release_image_ref") or ""),
        source_commit=str(admission.get("source_commit") or ""),
        model_manifest_digest=str(admission.get("model_manifest_digest") or ""),
        carrier_volume_admission=carrier_admission,
    )
    endpoint_payload = build_endpoint_payload(
        name=f"{resource_name_prefix}endpoint",
        template_id="<allocated-template-id>",
        network_volume_id=volume_id,
        data_center_id=str(volume.get("data_center_id") or ""),
        gpu_type_ids=gpu_type_ids,
    )
    write_json(
        output / "serverless_request_shapes.json",
        {"template": template_payload, "endpoint": endpoint_payload},
    )
    if not execute:
        return {
            **admission,
            "status": "dry_run_ready",
            "provider_mutations_performed": 0,
        }
    grant = require_paid_resource_admission(
        build_paid_lane_admission(
            resource_class="runpod_serverless_active_worker",
            blockers=list(admission.get("blockers") or []),
        ),
        resource_class="runpod_serverless_active_worker",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
    del grant  # the checked grant is the create boundary
    budget = ProductionGpuCampaignBudget(
        campaign_budget_ledger,
        initial_spent_usd=initial_spent_usd,
        initial_used_gpu_seconds=initial_gpu_seconds,
        total_spend_cap_usd=20.0,
        combined_gpu_wall_cap_seconds=21_000,
    )
    reservation_id = resource_name_prefix.rstrip("-")
    try:
        reservation = budget.reserve(
            reservation_id=reservation_id,
            gpu_seconds=reservation_seconds,
            max_hourly_rate_usd=max_hourly_rate_usd,
        )
    except CampaignBudgetExceeded as exc:
        return {
            **admission,
            "status": "blocked",
            "blockers": [str(exc.admission.get("blocker"))],
            "provider_mutations_performed": 0,
        }
    pending = open_pending_teardown(
        provider="runpod",
        lane="groot_oscar_serverless_campaign",
        run_id=reservation_id,
        resource_kind="serverless_endpoint",
        resource_name=f"{reservation_id}-endpoint",
        job_dir=output,
        max_age_seconds=reservation_seconds + 600,
    )
    deadline = time.time() + reservation_seconds
    state_path = output / "watchdog_state.json"
    state = {
        "schema_version": SCHEMA_VERSION,
        "resource_name_prefix": resource_name_prefix,
        "api_key_file": str(key_path),
        "deadline_epoch": deadline,
        "pending_teardown_record": pending["path"],
        "campaign_budget": {
            **reservation,
            "reservation_id": reservation_id,
            "ledger_path": str(Path(campaign_budget_ledger).expanduser().resolve()),
            "initial_spent_usd": initial_spent_usd,
            "initial_used_gpu_seconds": initial_gpu_seconds,
            "total_spend_cap_usd": 20.0,
            "combined_gpu_wall_cap_seconds": 21_000,
        },
        "raw_secret_values_recorded": False,
    }
    _write_watchdog_state(state_path, state)
    try:
        watchdog, armed = _arm_watchdog(
            output_dir=output,
            state_path=state_path,
            resource_name_prefix=resource_name_prefix,
            deadline=deadline,
        )
    except Exception as exc:
        cancel_pending_teardown(
            pending["path"],
            reason="watchdog_failed_before_provider_mutation",
            evidence={"provider_inventory_api_confirmed": True},
        )
        budget.settle(
            reservation_id=reservation_id,
            charged_gpu_seconds=0,
            charged_usd=0.0,
            outcome="watchdog_failed_before_provider_mutation",
        )
        return {
            **admission,
            "status": "blocked",
            "blockers": [f"serverless_watchdog_arm_failed:{type(exc).__name__}"],
            "provider_mutations_performed": 0,
        }
    watchdog_contract = {
        "watchdog_pid": watchdog.pid,
        "watchdog_pod_name_prefix": resource_name_prefix,
        "watchdog_deadline_epoch": deadline,
        "watchdog_process_identity_verified": True,
        "independent_teardown_watchdog": True,
        "armed_evidence": armed,
    }
    acceptance = accept_paid_provider_lane_lease_handoff(
        handoff,
        canary_watchdog=watchdog_contract,
        expected_binding=binding,
    )
    write_json(output / "provider_lane_handoff_acceptance.json", acceptance)
    if acceptance.get("status") != "accepted":
        state["provider_lane_handoff_acceptance"] = acceptance
        _write_watchdog_state(state_path, state)
        proof = _request_teardown(output, watchdog)
        return {
            **admission,
            "status": "blocked",
            "blockers": [*acceptance.get("blockers", []), "provider_lane_handoff_not_accepted"],
            "teardown": proof,
            "provider_mutations_performed": 0,
        }
    state["provider_lane_handoff_acceptance"] = acceptance
    _write_watchdog_state(state_path, state)
    mutations = 0
    storage_mutations = 0
    template_id = ""
    endpoint_id = ""
    strict_started: float | None = None
    campaign_started = False
    campaign_storage_owned = False
    compute_teardown_proof: dict[str, Any] | None = None
    try:
        staged = stage_campaign_inputs(
            campaign_io,
            access_key_file=runpod_s3_access_key_file,
            secret_key_file=runpod_s3_secret_key_file,
        )
        campaign_storage_owned = True
        storage_mutations += int(staged.get("uploaded_file_count") or 0) + int(
            staged.get("deleted_stale_input_file_count") or 0
        )
        write_json(output / "campaign_input_staging.json", staged)
        template_http, template_response = _runpod_call(
            "POST", "/templates", template_payload, key=api_key, timeout=60
        )
        mutations += 1
        template_id = str(
            template_response.get("id") if isinstance(template_response, Mapping) else ""
        )
        if template_http not in {200, 201} or not template_id:
            raise RuntimeError("serverless_template_create_failed_or_ambiguous")
        print(
            f"[runpod-serverless] template_created id={template_id}",
            file=sys.stderr,
            flush=True,
        )
        endpoint_payload["templateId"] = template_id
        state["template_id"] = template_id
        state["endpoint_create_requested_at_epoch"] = time.time()
        _write_watchdog_state(state_path, state)
        endpoint_http, endpoint_response = _runpod_call(
            "POST", "/endpoints", endpoint_payload, key=api_key, timeout=60
        )
        mutations += 1
        endpoint_id = str(
            endpoint_response.get("id") if isinstance(endpoint_response, Mapping) else ""
        )
        if endpoint_http not in {200, 201} or not endpoint_id:
            raise RuntimeError("serverless_endpoint_create_failed_or_ambiguous")
        state.update(
            {
                "template_id": template_id,
                "endpoint_id": endpoint_id,
                "endpoint_allocated_at_epoch": time.time(),
            }
        )
        _write_watchdog_state(state_path, state)
        print(
            f"[runpod-serverless] endpoint_created id={endpoint_id}",
            file=sys.stderr,
            flush=True,
        )
        bind_pending_teardown_instance(pending["path"], endpoint_id)
        startup_wall_timeout_seconds = compute_startup_wall_timeout_seconds(
            deadline_epoch=deadline,
            now_epoch=time.time(),
        )
        if startup_wall_timeout_seconds < 1:
            raise RuntimeError("serverless_startup_budget_no_longer_available")
        state["startup_wall_timeout_seconds"] = startup_wall_timeout_seconds
        _write_watchdog_state(state_path, state)
        startup_http, startup_response = _submit_job(
            api_key=api_key,
            endpoint_id=endpoint_id,
            operation="startup",
            timeout_seconds=startup_wall_timeout_seconds,
        )
        startup_id = str(startup_response.get("id") or "")
        if startup_http not in {200, 201} or not startup_id:
            raise RuntimeError("serverless_startup_job_submit_failed")
        startup = _poll_job(
            api_key=api_key,
            endpoint_id=endpoint_id,
            job_id=startup_id,
            operation_label="startup",
            wall_timeout_seconds=startup_wall_timeout_seconds,
        )
        _record_job_execution_state(state, startup)
        _write_watchdog_state(state_path, state)
        write_json(output / "startup_job_result.json", startup)
        if (
            startup.get("status") != "COMPLETED"
            or startup.get("output", {}).get("status") != "completed"
            or startup.get("output", {}).get("runtime_present") is not True
        ):
            raise RuntimeError("serverless_startup_not_proven")
        startup_output = startup.get("output")
        startup_output = dict(startup_output) if isinstance(startup_output, Mapping) else {}
        selected_gpu_name = str(startup_output.get("gpu_name") or "")
        selected_rate = ACTIVE_GPU_HOURLY_RATES_USD.get(selected_gpu_name)
        if selected_rate is not None:
            state["campaign_budget"]["max_hourly_rate_usd"] = selected_rate
            state["campaign_budget"]["billing_rate_basis"] = (
                "runpod_public_serverless_selected_gpu_ceiling"
            )
        else:
            state["campaign_budget"]["billing_rate_basis"] = (
                "runpod_public_serverless_selected_gpu_ceiling"
            )
        state["campaign_budget"]["selected_gpu_name"] = selected_gpu_name or None
        _write_watchdog_state(state_path, state)
        strict_started = time.monotonic()
        strict_http, strict_response = _submit_job(
            api_key=api_key,
            endpoint_id=endpoint_id,
            operation="strict-policy-smoke",
            timeout_seconds=STRICT_JOB_TIMEOUT_SECONDS,
        )
        strict_id = str(strict_response.get("id") or "")
        if strict_http not in {200, 201} or not strict_id:
            raise RuntimeError("serverless_strict_job_submit_failed")
        strict = _poll_job(
            api_key=api_key,
            endpoint_id=endpoint_id,
            job_id=strict_id,
            operation_label="strict-policy-smoke",
            wall_timeout_seconds=max(
                1,
                int(
                    STRICT_ATTEMPT_WALL_SECONDS
                    - STRICT_TEARDOWN_BUFFER_SECONDS
                    - (time.monotonic() - strict_started)
                ),
            ),
        )
        _record_job_execution_state(state, strict)
        _write_watchdog_state(state_path, state)
        strict["strict_attempt_elapsed_seconds"] = round(time.monotonic() - strict_started, 3)
        strict["teardown_buffer_seconds_remaining"] = round(
            max(
                0.0,
                STRICT_ATTEMPT_WALL_SECONDS - float(strict["strict_attempt_elapsed_seconds"]),
            ),
            3,
        )
        write_json(output / "strict_policy_job_result.json", strict)
        strict_output = strict.get("output")
        strict_output = dict(strict_output) if isinstance(strict_output, Mapping) else {}
        if not (
            strict.get("status") == "COMPLETED"
            and strict_output.get("status") == "completed"
            and strict_output.get("completed_action_count") == 3
            and strict_output.get("model_execution_proven") is True
            and len(str(strict_output.get("runtime_worker_identity_sha256") or "")) == 64
        ):
            raise RuntimeError("serverless_strict_policy_probe_failed")
        campaign_seconds_remaining = deadline - time.time()
        if campaign_seconds_remaining < (
            CAMPAIGN_JOB_TIMEOUT_SECONDS + CAMPAIGN_TEARDOWN_BUFFER_SECONDS
        ):
            raise RuntimeError("serverless_campaign_full_budget_no_longer_available")
        campaign_started = True
        campaign_http, campaign_response = _submit_job(
            api_key=api_key,
            endpoint_id=endpoint_id,
            operation="kitchen-campaign",
            timeout_seconds=CAMPAIGN_JOB_TIMEOUT_SECONDS,
            job_input={
                "campaign_manifest_relative_path": campaign_io["campaign_manifest_relative_path"],
                "campaign_manifest_sha256": campaign_io["campaign_manifest_sha256"],
                "output_relative_path": campaign_io["output_relative_path"],
                "expected_runtime_worker_identity_sha256": strict_output[
                    "runtime_worker_identity_sha256"
                ],
            },
        )
        campaign_id = str(campaign_response.get("id") or "")
        if campaign_http not in {200, 201} or not campaign_id:
            raise RuntimeError("serverless_campaign_job_submit_failed")
        campaign = _poll_job(
            api_key=api_key,
            endpoint_id=endpoint_id,
            job_id=campaign_id,
            operation_label="kitchen-campaign",
            wall_timeout_seconds=CAMPAIGN_JOB_TIMEOUT_SECONDS,
        )
        _record_job_execution_state(state, campaign)
        _write_watchdog_state(state_path, state)
        write_json(output / "kitchen_campaign_job_result.json", campaign)
        campaign_output = campaign.get("output")
        campaign_output = dict(campaign_output) if isinstance(campaign_output, Mapping) else {}
        if not (
            campaign.get("status") == "COMPLETED"
            and campaign_output.get("status") == "completed"
            and campaign_output.get("smoke_passed") is True
            and campaign_output.get("all_dynamic_episodes_completed") is True
            and len(list(campaign_output.get("runs") or [])) == 4
        ):
            raise RuntimeError("serverless_kitchen_campaign_failed")
        teardown = _request_teardown(
            output,
            watchdog,
            timeout_seconds=max(
                1,
                min(
                    180,
                    int(max(1.0, deadline - time.time())),
                ),
            ),
        )
        if teardown.get("status") != "PASS":
            raise RuntimeError("serverless_terminal_teardown_not_proven")
        compute_teardown_proof = teardown
        retrieved = retrieve_campaign_outputs(
            campaign_io,
            destination=output / "retrieved_campaign_artifacts",
            access_key_file=runpod_s3_access_key_file,
            secret_key_file=runpod_s3_secret_key_file,
        )
        write_json(output / "campaign_artifact_retrieval.json", retrieved)
        if retrieved.get("transfer_status") == "completed":
            cleanup = cleanup_campaign_storage(
                campaign_io,
                access_key_file=runpod_s3_access_key_file,
                secret_key_file=runpod_s3_secret_key_file,
            )
            storage_mutations += int(cleanup.get("deleted_file_count") or 0)
        else:
            cleanup = {
                "status": "blocked",
                "blockers": ["campaign_remote_artifacts_retained_until_transfer_completes"],
                "deleted_file_count": 0,
            }
        write_json(output / "campaign_storage_cleanup.json", cleanup)
        terminal_blockers = []
        if retrieved.get("status") != "completed":
            terminal_blockers.append("serverless_campaign_artifact_verification_failed")
        if cleanup.get("status") != "completed":
            terminal_blockers.append("serverless_campaign_storage_cleanup_failed")
        semantic_by_attempt = campaign_output.get("semantic_task_success_by_attempt")
        semantic_by_attempt = (
            dict(semantic_by_attempt) if isinstance(semantic_by_attempt, Mapping) else {}
        )
        result = {
            **admission,
            "status": "completed" if not terminal_blockers else "blocked",
            "blockers": terminal_blockers,
            "endpoint_id": endpoint_id,
            "template_id": template_id,
            "watchdog_pid": watchdog.pid,
            "watchdog_deadline_epoch": deadline,
            "pending_teardown_record": pending["path"],
            "campaign_budget_reservation": reservation,
            "provider_mutations_performed": mutations,
            "storage_mutations_performed": storage_mutations,
            "startup": startup,
            "strict_policy_probe": strict,
            "kitchen_campaign": campaign,
            "teardown": teardown,
            "campaign_artifact_retrieval": retrieved,
            "campaign_storage_cleanup": cleanup,
            "structural_campaign_completed": True,
            "semantic_task_success_proven": bool(semantic_by_attempt)
            and all(value is True for value in semantic_by_attempt.values()),
            "semantic_task_success_by_attempt": semantic_by_attempt,
            "semantic_task_success_not_inferred_from_execution": True,
        }
        write_json(output / "serverless_active_worker_result.json", result)
        return result
    except Exception as exc:
        if compute_teardown_proof is None:
            mark_pending_teardown_ambiguous(
                pending["path"], reason=f"serverless_failure:{type(exc).__name__}"
            )
            state.update({"template_id": template_id, "endpoint_id": endpoint_id})
            _write_watchdog_state(state_path, state)
            teardown_timeout = STRICT_TEARDOWN_BUFFER_SECONDS - 10
            if campaign_started:
                teardown_timeout = max(
                    1,
                    min(
                        CAMPAIGN_TEARDOWN_BUFFER_SECONDS,
                        int(max(1.0, deadline - time.time())),
                    ),
                )
            elif strict_started is not None:
                teardown_timeout = max(
                    1,
                    int(STRICT_ATTEMPT_WALL_SECONDS - (time.monotonic() - strict_started) - 10),
                )
            proof = _request_teardown(
                output,
                watchdog,
                timeout_seconds=teardown_timeout,
            )
        else:
            proof = compute_teardown_proof
        if campaign_storage_owned:
            try:
                retrieved = retrieve_campaign_outputs(
                    campaign_io,
                    destination=output / "retrieved_campaign_artifacts",
                    access_key_file=runpod_s3_access_key_file,
                    secret_key_file=runpod_s3_secret_key_file,
                )
            except Exception as retrieve_exc:
                retrieved = {
                    "status": "blocked",
                    "transfer_status": "blocked",
                    "blockers": [f"campaign_retrieval_exception:{type(retrieve_exc).__name__}"],
                }
        else:
            retrieved = {
                "status": "not_attempted",
                "transfer_status": "not_attempted",
                "blockers": [],
            }
        write_json(output / "campaign_artifact_retrieval.json", retrieved)
        if campaign_storage_owned and retrieved.get("transfer_status") == "completed":
            try:
                cleanup = cleanup_campaign_storage(
                    campaign_io,
                    access_key_file=runpod_s3_access_key_file,
                    secret_key_file=runpod_s3_secret_key_file,
                )
                storage_mutations += int(cleanup.get("deleted_file_count") or 0)
            except Exception as cleanup_exc:
                cleanup = {
                    "status": "blocked",
                    "blockers": [f"campaign_cleanup_exception:{type(cleanup_exc).__name__}"],
                }
        else:
            cleanup = {
                "status": "not_attempted" if not campaign_storage_owned else "blocked",
                "blockers": (
                    []
                    if not campaign_storage_owned
                    else ["campaign_remote_artifacts_retained_until_transfer_completes"]
                ),
            }
        write_json(output / "campaign_storage_cleanup.json", cleanup)
        result = {
            **admission,
            "status": "blocked",
            "blockers": [str(exc)],
            "provider_mutations_performed": mutations,
            "storage_mutations_performed": storage_mutations,
            "teardown": proof,
            "campaign_artifact_retrieval": retrieved,
            "campaign_storage_cleanup": cleanup,
            "endpoint_id": endpoint_id or None,
            "template_id": template_id or None,
        }
        write_json(output / "serverless_active_worker_result.json", result)
        return result
