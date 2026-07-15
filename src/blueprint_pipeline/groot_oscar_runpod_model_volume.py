"""Canonical bounded preparation of the GR00T + OSCAR RunPod model volume.

The allocator creates one network volume and one temporary preparation Pod.
An independent deadline watchdog owns both names before either create call.
The Pod exposes only a token-authenticated verification response, and the
supervisor deletes the Pod before handing the verified volume to the canary.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import safe_outbound_http
from .common import ensure_dir, write_json
from .gpu_render_providers import _runpod_call, get_render_provider
from .groot_oscar_infrastructure_admission import (
    RUNPOD_NETWORK_VOLUME_DATA_CENTER_IDS,
    build_runpod_network_volume_evidence,
)
from .paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    require_paid_resource_admission,
)


SCHEMA_VERSION = "groot_oscar_runpod_model_volume_admission.v1"
RESULT_SCHEMA_VERSION = "groot_oscar_runpod_model_volume_result.v1"
WATCHDOG_SCHEMA_VERSION = "groot_oscar_runpod_model_volume_watchdog.v1"
MODEL_CACHE_PATH = "/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1"
MIN_VOLUME_GIB = 30
MAX_VOLUME_GIB = 100
MAX_TTL_SECONDS = 3600
EVIDENCE_PORT = 8765
POD_NAME_PREFIX = "blueprint-groot-oscar-canary-model-"
VOLUME_NAME_PREFIX = "blueprint-groot-oscar-models-"
# The user's standing 2026-07-14 authorization explicitly covers everything
# needed to complete this bounded campaign, including these qualified RTX
# datacenter fallbacks. Spend, CUDA, storage-datacenter, one-resource, and
# watchdog gates still apply independently.
AUTHORIZED_MODEL_VOLUME_GPU_TYPES = frozenset(
    {
        "NVIDIA A40",
        "NVIDIA L40S",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    }
)
_DIGEST_REF = re.compile(r"\A[^\s@]+@sha256:[0-9a-f]{64}\Z")
_RUNPOD_ID = re.compile(r"\A[A-Za-z0-9._-]{1,256}\Z")
_SECRET_ERROR_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+\S+"),
    re.compile(r"(?i)\b(?:api[_-]?key|authorization|token)\s*[=:]\s*\S+"),
    re.compile(r"\b(?:hf|rpa|rp)_[A-Za-z0-9._-]{8,}\b"),
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_secret(path: str | Path) -> str:
    value = Path(path).expanduser().read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError("model_volume_secret_file_empty")
    return value


def _safe_provider_error_summary(payload: Any) -> str | None:
    """Keep provider diagnosis without persisting request bodies or secrets."""

    if not isinstance(payload, Mapping):
        return None
    parts: list[str] = []
    for field in ("code", "statusCode", "error", "message", "detail", "title"):
        value = payload.get(field)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            text_value = str(value).strip()
            if text_value:
                parts.append(f"{field}={text_value}")
    if not parts:
        return None
    summary = "; ".join(parts)[:1000]
    for pattern in _SECRET_ERROR_PATTERNS:
        summary = pattern.sub("[REDACTED]", summary)
    return summary


def _single_gpu_capacity_verified(
    *,
    capacity: Mapping[str, Any],
    selected: Mapping[str, Any],
    data_center_id: str,
    required_cuda_version: str,
) -> bool:
    """Accept the exact one-GPU offer while keeping it advisory, not reserved.

    Network-volume support is gated separately by the provider-derived
    ``RUNPOD_NETWORK_VOLUME_DATA_CENTER_IDS`` set. RunPod's account
    ``myself.datacenters`` response can omit a datacenter where this account
    has just created a volume, so it is not a reliable duplicate launch gate.
    """

    return bool(
        capacity.get("status") == "available"
        and capacity.get("capacity_confidence") == "advisory"
        and selected.get("capacity_confidence") == "advisory"
        and selected.get("single_gpu_offer_requested") is True
        and selected.get("single_gpu_offer_available") is True
        and selected.get("capacity_data_center_id") == data_center_id
        and required_cuda_version
        in (selected.get("capacity_allowed_cuda_versions") or [])
    )


def build_model_volume_admission(
    *,
    release_image_ref: str,
    data_center_id: str,
    gpu_type_id: str,
    required_cuda_version: str,
    volume_size_gib: int,
    hard_ttl_seconds: int,
    max_spend_usd: float,
    hourly_rate_usd: float,
    volume_hourly_rate_usd: float,
    capacity_verified: bool,
    inventory_verified_zero: bool,
    paid_mutation_authorized: bool,
    watchdog_armed_before_allocation: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not _DIGEST_REF.fullmatch(release_image_ref):
        blockers.append("model_volume_release_image_not_digest_pinned")
    if not data_center_id:
        blockers.append("model_volume_data_center_missing")
    elif data_center_id not in RUNPOD_NETWORK_VOLUME_DATA_CENTER_IDS:
        blockers.append("model_volume_data_center_not_network_volume_capable")
    if not gpu_type_id:
        blockers.append("model_volume_gpu_type_missing")
    elif gpu_type_id not in AUTHORIZED_MODEL_VOLUME_GPU_TYPES:
        blockers.append("model_volume_gpu_type_outside_authorized_campaign")
    if required_cuda_version != "12.8":
        blockers.append("model_volume_cuda_version_not_12_8")
    if type(volume_size_gib) is not int or not MIN_VOLUME_GIB <= volume_size_gib <= MAX_VOLUME_GIB:
        blockers.append("model_volume_size_outside_30_to_100_gib")
    if type(hard_ttl_seconds) is not int or not 60 < hard_ttl_seconds <= MAX_TTL_SECONDS:
        blockers.append("model_volume_ttl_outside_guardrail")
    if not isinstance(max_spend_usd, (int, float)) or max_spend_usd <= 0:
        blockers.append("model_volume_max_spend_missing")
    if not isinstance(hourly_rate_usd, (int, float)) or hourly_rate_usd <= 0:
        blockers.append("model_volume_hourly_rate_missing")
    if (
        not isinstance(volume_hourly_rate_usd, (int, float))
        or volume_hourly_rate_usd <= 0
    ):
        blockers.append("model_volume_storage_hourly_rate_missing")
    elif (
        isinstance(hourly_rate_usd, (int, float))
        and hourly_rate_usd > 0
        and (hourly_rate_usd + volume_hourly_rate_usd)
        * hard_ttl_seconds
        / 3600
        > max_spend_usd
    ):
        blockers.append("model_volume_ttl_cost_exceeds_max_spend")
    if capacity_verified is not True:
        blockers.append("model_volume_single_gpu_capacity_not_verified")
    if inventory_verified_zero is not True:
        blockers.append("model_volume_preallocation_inventory_not_zero")
    if paid_mutation_authorized is not True:
        blockers.append("model_volume_paid_mutation_not_authorized")
    if watchdog_armed_before_allocation is not True:
        blockers.append("model_volume_watchdog_not_armed_before_allocation")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "admitted" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "release_image_ref": release_image_ref,
        "data_center_id": data_center_id,
        "gpu_type_id": gpu_type_id,
        "required_cuda_version": required_cuda_version,
        "volume_size_gib": volume_size_gib,
        "limits": {
            "hard_ttl_seconds": hard_ttl_seconds,
            "max_spend_usd": max_spend_usd,
            "gpu_hourly_rate_usd": hourly_rate_usd,
            "volume_hourly_rate_usd": volume_hourly_rate_usd,
            "one_volume_limit": True,
            "one_preparation_pod_limit": True,
        },
        "raw_secret_values_recorded": False,
    }


def _delete_pod(*, key: str, pod_id: str) -> dict[str, Any]:
    delete_http, _ = _runpod_call("DELETE", f"/pods/{pod_id}", None, key=key, timeout=30)
    verify_http = 0
    for _attempt in range(6):
        verify_http, _ = _runpod_call(
            "GET", f"/pods/{pod_id}", None, key=key, timeout=30
        )
        if verify_http == 404:
            break
        time.sleep(2)
    absent = verify_http == 404
    return {
        "delete_http": delete_http,
        "verify_http": verify_http,
        "provider_absence_confirmed": absent,
    }


def _delete_volume(*, key: str, volume_id: str) -> dict[str, Any]:
    delete_http, _ = _runpod_call(
        "DELETE", f"/networkvolumes/{volume_id}", None, key=key, timeout=30
    )
    verify_http = 0
    for _attempt in range(6):
        verify_http, _ = _runpod_call(
            "GET", f"/networkvolumes/{volume_id}", None, key=key, timeout=30
        )
        if verify_http == 404:
            break
        time.sleep(2)
    absent = verify_http == 404
    return {
        "delete_http": delete_http,
        "verify_http": verify_http,
        "provider_absence_confirmed": absent,
    }


def _matching_resources(
    *, key: str, pod_prefix: str | None, volume_prefix: str | None
) -> tuple[list[str], list[str], bool]:
    pods_http, pods_payload = _runpod_call("GET", "/pods", None, key=key, timeout=30)
    volumes_http, volumes_payload = _runpod_call(
        "GET", "/networkvolumes", None, key=key, timeout=30
    )
    inventory_verified = (
        pods_http == 200
        and isinstance(pods_payload, list)
        and volumes_http == 200
        and isinstance(volumes_payload, list)
    )
    pod_rows = pods_payload if pods_http == 200 and isinstance(pods_payload, list) else []
    volume_rows = (
        volumes_payload if volumes_http == 200 and isinstance(volumes_payload, list) else []
    )
    pod_ids = [
        str(row.get("id"))
        for row in pod_rows
        if isinstance(row, Mapping)
        and (pod_prefix is None or str(row.get("name") or "").startswith(pod_prefix))
        and str(row.get("id") or "")
    ]
    volume_ids = [
        str(row.get("id"))
        for row in volume_rows
        if isinstance(row, Mapping)
        and (
            volume_prefix is None
            or str(row.get("name") or "").startswith(volume_prefix)
        )
        and str(row.get("id") or "")
    ]
    return pod_ids, volume_ids, inventory_verified


def watchdog(*, state_path: Path) -> int:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    root = state_path.parent
    deadline = float(state["deadline_epoch"])
    write_json(
        root / "watchdog_armed.json",
        {
            "schema_version": WATCHDOG_SCHEMA_VERSION,
            "status": "armed",
            "pid": os.getpid(),
            "deadline_epoch": deadline,
            "pod_name_prefix": state.get("pod_name_prefix"),
            "volume_name": state.get("volume_name"),
            "watchdog_nonce": state.get("watchdog_nonce"),
            "raw_secret_values_recorded": False,
        },
    )
    handoff = root / "watchdog_handoff.json"
    while time.time() < deadline:
        if handoff.is_file():
            try:
                handoff_payload = json.loads(handoff.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                handoff_payload = {}
            terminal_statuses = {
                "cancelled_before_provider_allocation",
                "failure_cleanup_provider_terminal",
            }
            if (
                isinstance(handoff_payload, Mapping)
                and handoff_payload.get("status") in terminal_statuses
            ):
                write_json(
                    root / "watchdog_result.json",
                    {
                        "schema_version": WATCHDOG_SCHEMA_VERSION,
                        "status": "handoff_after_supervisor_teardown",
                        "provider_mutations_performed": 0,
                        "raw_secret_values_recorded": False,
                    },
                )
                return 0
        time.sleep(10)
    provider = get_render_provider("runpod")
    key = provider._key()  # type: ignore[attr-defined]
    if not key:
        result = {
            "schema_version": WATCHDOG_SCHEMA_VERSION,
            "status": "teardown_unverified",
            "blockers": ["runpod_api_key_missing"],
            "raw_secret_values_recorded": False,
        }
        write_json(root / "watchdog_result.json", result)
        return 2
    pod_ids, volume_ids, inventory_verified = _matching_resources(
        key=key,
        pod_prefix=str(state["pod_name_prefix"]),
        volume_prefix=str(state["volume_name"]),
    )
    pod_results = [_delete_pod(key=key, pod_id=item) for item in pod_ids]
    volume_results = [_delete_volume(key=key, volume_id=item) for item in volume_ids]
    final_pods, final_volumes, final_inventory_verified = _matching_resources(
        key=key,
        pod_prefix=str(state["pod_name_prefix"]),
        volume_prefix=str(state["volume_name"]),
    )
    terminal = bool(
        inventory_verified
        and final_inventory_verified
        and not final_pods
        and not final_volumes
    )
    result = {
        "schema_version": WATCHDOG_SCHEMA_VERSION,
        "status": "provider_terminal" if terminal else "teardown_unverified",
        "pod_terminations": pod_results,
        "volume_deletions": volume_results,
        "provider_absence_confirmed": terminal,
        "raw_secret_values_recorded": False,
    }
    write_json(root / "watchdog_result.json", result)
    return 0 if terminal else 2


def _worker_script() -> str:
    return r'''set -euo pipefail
export HF_HUB_DISABLE_TELEMETRY=1
export ROOT=/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1
export EVIDENCE=/workspace/.blueprint-model-cache/preparation-evidence
mkdir -p "$EVIDENCE"
/opt/gr00t-venv/bin/python -m blueprint_pipeline.groot_oscar_model_cache prepare --root "$ROOT" --out "$EVIDENCE/manifest.json"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/gr00t-venv/bin/python -m blueprint_pipeline.groot_oscar_model_cache verify --root "$ROOT" --provider-volume-id "$BLUEPRINT_GROOT_OSCAR_PROVIDER_VOLUME_ID" --out "$EVIDENCE/verification.json"
/opt/gr00t-venv/bin/python - <<'PY'
import http.server, json, os
from pathlib import Path
token = os.environ["BLUEPRINT_MODEL_VOLUME_EVIDENCE_TOKEN"]
evidence = Path(os.environ["EVIDENCE"])
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/verification" or self.headers.get("Authorization") != "Bearer " + token:
            self.send_response(404); self.end_headers(); return
        body = json.dumps({
            "verification": json.loads((evidence / "verification.json").read_text()),
            "manifest": json.loads((evidence / "manifest.json").read_text()),
        }, sort_keys=True).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *_args):
        return
http.server.ThreadingHTTPServer(("0.0.0.0", 8765), Handler).serve_forever()
PY'''


def _fetch_verification(*, pod_id: str, token: str, timeout_seconds: int = 20) -> dict[str, Any]:
    url = f"https://{pod_id}-{EVIDENCE_PORT}.proxy.runpod.net/verification"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    response = safe_outbound_http.open_request(
        request,
        policy=safe_outbound_http.service_endpoint_policy(url, max_response_bytes=4 * 1024 * 1024),
        timeout_seconds=timeout_seconds,
    )
    value = json.loads(response.body.decode("utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _extract_id(payload: Mapping[str, Any]) -> str:
    value = payload.get("id") or payload.get("podId")
    if type(value) is not str or value != value.strip() or not _RUNPOD_ID.fullmatch(value):
        return ""
    return value


def run_model_volume(
    *,
    output_dir: Path,
    release_image_ref: str,
    data_center_id: str,
    gpu_type_id: str,
    required_cuda_version: str,
    volume_size_gib: int,
    hard_ttl_seconds: int,
    max_spend_usd: float,
    volume_hourly_rate_usd: float,
    hf_token_file: Path,
    allow_paid: bool,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    ensure_dir(output)
    provider = get_render_provider("runpod")
    key = provider._key()  # type: ignore[attr-defined]
    if not key:
        raise ValueError("runpod_api_key_missing")
    try:
        hf_token = _read_secret(hf_token_file)
    except Exception as exc:  # noqa: BLE001 - stop before inventory or allocation
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked_before_allocation",
            "blockers": ["model_volume_hf_token_unavailable"],
            "provider_mutation_attempted": False,
            "maximum_compute_spend_usd": 0.0,
            "maximum_storage_spend_usd": 0.0,
            "error_type": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
        write_json(output / "model_volume_result.json", result)
        return result
    suffix = secrets.token_hex(5)
    pod_prefix = f"{POD_NAME_PREFIX}{suffix}"
    pod_name = pod_prefix
    volume_name = f"{VOLUME_NAME_PREFIX}{suffix}"
    existing_pods, existing_volumes, inventory_verified = _matching_resources(
        key=key,
        pod_prefix=None,
        volume_prefix=None,
    )
    capacity = provider.capacity_preflight(
        {
            "cloudType": "SECURE",
            "gpuTypeIds": [gpu_type_id],
            "dataCenterIds": [data_center_id],
            "allowedCudaVersions": [required_cuda_version],
            "requires_rtx": True,
        }
    )
    viable = capacity.get("viable_gpu_types")
    viable = viable if isinstance(viable, list) else []
    selected = next(
        (row for row in viable if isinstance(row, Mapping) and row.get("gpu_type_id") == gpu_type_id),
        {},
    )
    hourly_rate = float(selected.get("on_demand_price_usd_per_hour") or 0)
    capacity_verified = _single_gpu_capacity_verified(
        capacity=capacity,
        selected=selected,
        data_center_id=data_center_id,
        required_cuda_version=required_cuda_version,
    )
    deadline = time.time() + hard_ttl_seconds
    watchdog_nonce = secrets.token_hex(16)
    state_path = output / "watchdog_state.json"
    write_json(
        state_path,
        {
            "deadline_epoch": deadline,
            "pod_name_prefix": pod_prefix,
            "volume_name": volume_name,
            "watchdog_nonce": watchdog_nonce,
        },
    )
    watchdog_armed = False
    stale_watchdog_state = any(
        (output / name).exists()
        for name in ("watchdog_armed.json", "watchdog_handoff.json", "watchdog.pid")
    )
    if not stale_watchdog_state:
        try:
            with (output / "watchdog.log").open("ab") as log:
                watch = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "blueprint_pipeline.groot_oscar_runpod_model_volume",
                        "watchdog",
                        "--state",
                        str(state_path),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            (output / "watchdog.pid").write_text(f"{watch.pid}\n", encoding="utf-8")
            armed_path = output / "watchdog_armed.json"
            watchdog_start_deadline = time.time() + 10
            while time.time() < watchdog_start_deadline:
                if armed_path.is_file() and watch.poll() is None:
                    try:
                        armed = json.loads(armed_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError, json.JSONDecodeError):
                        armed = {}
                    watchdog_armed = bool(
                        isinstance(armed, Mapping)
                        and armed.get("schema_version") == WATCHDOG_SCHEMA_VERSION
                        and armed.get("status") == "armed"
                        and armed.get("pid") == watch.pid
                        and armed.get("watchdog_nonce") == watchdog_nonce
                        and armed.get("pod_name_prefix") == pod_prefix
                        and armed.get("volume_name") == volume_name
                    )
                    if watchdog_armed:
                        break
                if watch.poll() is not None:
                    break
                time.sleep(0.05)
        except Exception:  # noqa: BLE001 - admission remains blocked without handoff
            watchdog_armed = False
    admission = build_model_volume_admission(
        release_image_ref=release_image_ref,
        data_center_id=data_center_id,
        gpu_type_id=gpu_type_id,
        required_cuda_version=required_cuda_version,
        volume_size_gib=volume_size_gib,
        hard_ttl_seconds=hard_ttl_seconds,
        max_spend_usd=max_spend_usd,
        hourly_rate_usd=hourly_rate,
        volume_hourly_rate_usd=volume_hourly_rate_usd,
        capacity_verified=capacity_verified,
        inventory_verified_zero=(
            inventory_verified and not existing_pods and not existing_volumes
        ),
        paid_mutation_authorized=allow_paid,
        watchdog_armed_before_allocation=watchdog_armed,
    )
    write_json(output / "model_volume_admission.json", admission)
    try:
        require_paid_resource_admission(
            admission,
            resource_class="model_volume",
            expected_schema_version=SCHEMA_VERSION,
        )
    except PaidResourceAdmissionBlocked:
        write_json(
            output / "watchdog_handoff.json",
            {"status": "cancelled_before_provider_allocation"},
        )
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked_before_allocation",
            "blockers": admission.get("blockers") or [
                "model_volume_paid_resource_admission_blocked"
            ],
            "provider_mutation_attempted": False,
            "maximum_compute_spend_usd": 0.0,
            "maximum_storage_spend_usd": 0.0,
            "raw_secret_values_recorded": False,
        }
        write_json(output / "model_volume_result.json", result)
        return result
    started = time.time()
    volume_id = ""
    pod_id = ""
    success = False
    pod_teardown: dict[str, Any] = {"provider_absence_confirmed": False}
    volume_teardown: dict[str, Any] | None = None
    verification: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    error_type: str | None = None
    provider_failure: dict[str, Any] = {}
    compute_started = False
    try:
        volume_http, volume_response = _runpod_call(
            "POST",
            "/networkvolumes",
            {"dataCenterId": data_center_id, "name": volume_name, "size": volume_size_gib},
            key=key,
            timeout=45,
        )
        volume_id = _extract_id(_mapping(volume_response))
        if volume_http not in {200, 201} or not volume_id:
            provider_error_summary = _safe_provider_error_summary(volume_response)
            provider_failure = {
                "operation": "create_network_volume",
                "http_status": volume_http,
                "provider_error_summary": provider_error_summary,
                "provider_error_recorded": provider_error_summary is not None,
                "allocation_created": bool(volume_id),
                "spend_occurred": False if not volume_id else None,
                "raw_provider_response_recorded": False,
            }
            raise RuntimeError("runpod_network_volume_create_failed_or_ambiguous")
        get_http, volume_row = _runpod_call(
            "GET", f"/networkvolumes/{volume_id}", None, key=key, timeout=30
        )
        volume_evidence = build_runpod_network_volume_evidence(
            provider_payload=_mapping(volume_row),
            expected_volume_id=volume_id,
            model_cache_path=MODEL_CACHE_PATH,
            expected_name=volume_name,
            allocation_nonce=suffix,
        )
        if (
            get_http != 200
            or volume_evidence["status"] != "verified"
            or volume_evidence["data_center_id"] != data_center_id
            or volume_evidence["size_bytes"] != volume_size_gib * 1024**3
        ):
            raise RuntimeError("runpod_network_volume_post_create_verification_failed")
        write_json(output / "network_volume_evidence.json", volume_evidence)
        evidence_token = secrets.token_urlsafe(32)
        pod_body = {
            "cloudType": "SECURE",
            "computeType": "GPU",
            "gpuCount": 1,
            "gpuTypeIds": [gpu_type_id],
            "gpuTypePriority": "availability",
            "containerDiskInGb": 80,
            "minVCPUPerGPU": 4,
            "minRAMPerGPU": 16,
            "name": pod_name,
            "imageName": release_image_ref,
            "dockerEntrypoint": ["bash", "-lc"],
            "dockerStartCmd": [_worker_script()],
            "ports": [f"{EVIDENCE_PORT}/http"],
            "volumeMountPath": "/workspace",
            "networkVolumeId": volume_id,
            "dataCenterIds": [data_center_id],
            "allowedCudaVersions": [required_cuda_version],
            "env": {
                "HF_TOKEN": hf_token,
                "BLUEPRINT_MODEL_VOLUME_EVIDENCE_TOKEN": evidence_token,
                "BLUEPRINT_GROOT_OSCAR_PROVIDER_VOLUME_ID": volume_id,
            },
        }
        pod_http, pod_response = _runpod_call(
            "POST", "/pods", pod_body, key=key, timeout=90
        )
        pod_id = _extract_id(_mapping(pod_response))
        if pod_http not in {200, 201} or not pod_id:
            provider_error_summary = _safe_provider_error_summary(pod_response)
            provider_failure = {
                "operation": "create_preparation_pod",
                "http_status": pod_http,
                "provider_error_summary": provider_error_summary,
                "provider_error_recorded": provider_error_summary is not None,
                "allocation_created": bool(pod_id),
                "spend_occurred": False if not pod_id else None,
                "raw_provider_response_recorded": False,
            }
            raise RuntimeError("runpod_model_volume_pod_create_failed_or_ambiguous")
        compute_started = True
        write_json(
            output / "preparation_pod.json",
            {
                "status": "allocated",
                "pod_id": pod_id,
                "pod_name": pod_name,
                "volume_id": volume_id,
                "raw_secret_values_recorded": False,
            },
        )
        while time.time() < deadline - 60:
            try:
                response = _fetch_verification(pod_id=pod_id, token=evidence_token)
            except (OSError, ValueError, json.JSONDecodeError, urllib.error.HTTPError):
                time.sleep(10)
                continue
            verification = _mapping(response.get("verification"))
            manifest = _mapping(response.get("manifest"))
            if (
                verification.get("status") == "passed"
                and verification.get("schema_version") == "groot_oscar_external_model_cache_verification.v2"
                and verification.get("provider_volume_id") == volume_id
                and verification.get("cache_root") == MODEL_CACHE_PATH
                and str(verification.get("model_manifest_digest") or "")
                == str(manifest.get("manifest_digest") or "")
            ):
                success = True
                break
            raise RuntimeError("runpod_model_volume_verification_invalid")
        if not success:
            raise RuntimeError("runpod_model_volume_verification_timeout")
        write_json(output / "model_cache_verification.json", verification)
        write_json(output / "model_cache_manifest.json", manifest)
    except Exception as exc:  # noqa: BLE001 - terminal evidence and cleanup are mandatory
        error_type = type(exc).__name__
        write_json(
            output / "model_volume_error.json",
            {
                "error_type": error_type,
                "error": str(exc),
                "provider_failure": provider_failure or None,
                "raw_secret_values_recorded": False,
            },
        )
    finally:
        if pod_id:
            pod_teardown = _delete_pod(key=key, pod_id=pod_id)
        else:
            matching_pods, _, _ = _matching_resources(
                key=key, pod_prefix=pod_prefix, volume_prefix=volume_name
            )
            for item in matching_pods:
                pod_teardown = _delete_pod(key=key, pod_id=item)
        if not success:
            volume_ids = [volume_id] if volume_id else []
            if not volume_ids:
                _, volume_ids, _ = _matching_resources(
                    key=key, pod_prefix=pod_prefix, volume_prefix=volume_name
                )
            for item in volume_ids:
                volume_teardown = _delete_volume(key=key, volume_id=item)
        final_pods, final_volumes, final_inventory_verified = _matching_resources(
            key=key, pod_prefix=pod_prefix, volume_prefix=volume_name
        )
        cleanup_terminal = bool(
            final_inventory_verified
            and not final_pods
            and not final_volumes
        )
        if success and (
            final_inventory_verified
            and not final_pods
            and final_volumes == [volume_id]
            and pod_teardown.get("provider_absence_confirmed") is True
        ):
            write_json(
                output / "watchdog_handoff.json",
                {
                    "status": "volume_ready_watchdog_retained",
                    "volume_id": volume_id,
                    "preparation_pod_absence_confirmed": True,
                    "volume_presence_confirmed": True,
                    "teardown_owner": "independent_model_volume_watchdog",
                    "watchdog_deadline_epoch": deadline,
                    "next_owner_must_arm_before_transfer": True,
                },
            )
        elif not success and cleanup_terminal and (
            not pod_id or pod_teardown.get("provider_absence_confirmed") is True
        ):
            write_json(
                output / "watchdog_handoff.json",
                {
                    "status": "failure_cleanup_provider_terminal",
                    "volume_id": volume_id or None,
                    "preparation_pod_absence_confirmed": not final_pods,
                    "failure_volume_absence_confirmed": not final_volumes,
                },
            )
    elapsed = max(0.0, time.time() - started)
    handoff_path = output / "watchdog_handoff.json"
    result_success = bool(
        success
        and pod_teardown.get("provider_absence_confirmed") is True
        and final_inventory_verified
        and not final_pods
        and final_volumes == [volume_id]
        and handoff_path.is_file()
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed" if result_success else "failed",
        "blockers": [] if result_success else [
            "runpod_model_volume_preparation_failed"
            if not success
            else "runpod_model_volume_pod_teardown_unverified"
        ],
        "volume_id": volume_id or None,
        "data_center_id": data_center_id,
        "model_cache_path": MODEL_CACHE_PATH,
        "model_manifest_digest": verification.get("model_manifest_digest"),
        "preparation_pod_id": pod_id or None,
        "preparation_pod_teardown": pod_teardown,
        "failure_volume_teardown": volume_teardown,
        "elapsed_seconds": elapsed,
        "maximum_compute_spend_usd": (
            hourly_rate * elapsed / 3600 if compute_started else 0.0
        ),
        "maximum_total_spend_usd": (
            (hourly_rate + volume_hourly_rate_usd) * hard_ttl_seconds / 3600
        ),
        "watchdog_retained_until_epoch": deadline if result_success else None,
        "error_type": error_type,
        "raw_secret_values_recorded": False,
    }
    write_json(output / "model_volume_result.json", result)
    return result


def launch_detached(*, output_dir: Path, run_arguments: Sequence[str]) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    ensure_dir(output)
    if (output / "model_volume_result.json").exists():
        raise ValueError("model_volume_output_already_terminal")
    lock_path = output / "supervisor.lock"
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("model_volume_output_already_has_supervisor") from exc
    with os.fdopen(lock_fd, "w", encoding="utf-8") as lock:
        lock.write(f"created_by_pid={os.getpid()}\n")
    with (output / "supervisor.log").open("ab") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "blueprint_pipeline.paid_resource_allocator",
                "model-volume-run",
                *run_arguments,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    result = {
        "schema_version": "groot_oscar_runpod_model_volume_supervisor.v1",
        "status": "supervisor_started",
        "pid": process.pid,
        "start_new_session": True,
        "raw_secret_values_recorded": False,
    }
    write_json(output / "supervisor_launch.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    watch = sub.add_parser("watchdog")
    watch.add_argument("--state", required=True)
    args = parser.parse_args(argv)
    if args.command == "watchdog":
        return watchdog(state_path=Path(args.state))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
