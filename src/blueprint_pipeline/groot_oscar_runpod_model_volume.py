"""Canonical bounded preparation of the GR00T + OSCAR RunPod model volume.

The allocator creates one network volume and one temporary preparation Pod.
An independent deadline watchdog owns both names before either create call.
The Pod exposes only a token-authenticated verification response, and the
supervisor deletes the Pod before handing the verified volume to the canary.
"""

from __future__ import annotations

import json
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
from .paid_resource_admission import require_paid_resource_admission


SCHEMA_VERSION = "groot_oscar_runpod_model_volume_admission.v1"
RESULT_SCHEMA_VERSION = "groot_oscar_runpod_model_volume_result.v1"
WATCHDOG_SCHEMA_VERSION = "groot_oscar_runpod_model_volume_watchdog.v1"
MODEL_CACHE_PATH = "/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1"
MIN_VOLUME_GIB = 30
MAX_VOLUME_GIB = 100
MAX_TTL_SECONDS = 3600
EVIDENCE_PORT = 8765
_DIGEST_REF = re.compile(r"\A[^\s@]+@sha256:[0-9a-f]{64}\Z")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_secret(path: str | Path) -> str:
    value = Path(path).expanduser().read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError("model_volume_secret_file_empty")
    return value


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
    inventory_verified_zero: bool,
    paid_mutation_authorized: bool,
    watchdog_armed_before_allocation: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not _DIGEST_REF.fullmatch(release_image_ref):
        blockers.append("model_volume_release_image_not_digest_pinned")
    if not data_center_id:
        blockers.append("model_volume_data_center_missing")
    if not gpu_type_id:
        blockers.append("model_volume_gpu_type_missing")
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
    elif hourly_rate_usd * hard_ttl_seconds / 3600 > max_spend_usd:
        blockers.append("model_volume_ttl_cost_exceeds_max_spend")
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


def _matching_resources(*, key: str, pod_prefix: str, volume_name: str) -> tuple[list[str], list[str]]:
    pods_http, pods_payload = _runpod_call("GET", "/pods", None, key=key, timeout=30)
    volumes_http, volumes_payload = _runpod_call(
        "GET", "/networkvolumes", None, key=key, timeout=30
    )
    pod_rows = pods_payload if pods_http == 200 and isinstance(pods_payload, list) else []
    volume_rows = (
        volumes_payload if volumes_http == 200 and isinstance(volumes_payload, list) else []
    )
    pod_ids = [
        str(row.get("id"))
        for row in pod_rows
        if isinstance(row, Mapping)
        and str(row.get("name") or "").startswith(pod_prefix)
        and str(row.get("id") or "")
    ]
    volume_ids = [
        str(row.get("id"))
        for row in volume_rows
        if isinstance(row, Mapping)
        and str(row.get("name") or "") == volume_name
        and str(row.get("id") or "")
    ]
    return pod_ids, volume_ids


def watchdog(*, state_path: Path) -> int:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    root = state_path.parent
    deadline = float(state["deadline_epoch"])
    handoff = root / "watchdog_handoff.json"
    while time.time() < deadline:
        if handoff.is_file():
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
    pod_ids, volume_ids = _matching_resources(
        key=key,
        pod_prefix=str(state["pod_name_prefix"]),
        volume_name=str(state["volume_name"]),
    )
    pod_results = [_delete_pod(key=key, pod_id=item) for item in pod_ids]
    volume_results = [_delete_volume(key=key, volume_id=item) for item in volume_ids]
    final_pods, final_volumes = _matching_resources(
        key=key,
        pod_prefix=str(state["pod_name_prefix"]),
        volume_name=str(state["volume_name"]),
    )
    terminal = not final_pods and not final_volumes
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
    return str(payload.get("id") or payload.get("podId") or "").strip()


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
    hf_token_file: Path,
    allow_paid: bool,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    ensure_dir(output)
    provider = get_render_provider("runpod")
    key = provider._key()  # type: ignore[attr-defined]
    if not key:
        raise ValueError("runpod_api_key_missing")
    suffix = secrets.token_hex(5)
    pod_prefix = f"blueprint-groot-oscar-canary-model-{suffix}"
    pod_name = pod_prefix
    volume_name = f"blueprint-groot-oscar-models-{suffix}"
    existing_pods, existing_volumes = _matching_resources(
        key=key, pod_prefix=pod_prefix, volume_name=volume_name
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
    deadline = time.time() + hard_ttl_seconds
    state_path = output / "watchdog_state.json"
    write_json(
        state_path,
        {
            "deadline_epoch": deadline,
            "pod_name_prefix": pod_prefix,
            "volume_name": volume_name,
        },
    )
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
    admission = build_model_volume_admission(
        release_image_ref=release_image_ref,
        data_center_id=data_center_id,
        gpu_type_id=gpu_type_id,
        required_cuda_version=required_cuda_version,
        volume_size_gib=volume_size_gib,
        hard_ttl_seconds=hard_ttl_seconds,
        max_spend_usd=max_spend_usd,
        hourly_rate_usd=hourly_rate,
        inventory_verified_zero=not existing_pods and not existing_volumes,
        paid_mutation_authorized=allow_paid,
        watchdog_armed_before_allocation=watch.poll() is None,
    )
    write_json(output / "model_volume_admission.json", admission)
    try:
        require_paid_resource_admission(
            admission,
            resource_class="model_volume",
            expected_schema_version=SCHEMA_VERSION,
        )
    except Exception:
        write_json(
            output / "watchdog_handoff.json",
            {"status": "cancelled_before_provider_allocation"},
        )
        raise
    started = time.time()
    volume_id = ""
    pod_id = ""
    success = False
    pod_teardown: dict[str, Any] = {"provider_absence_confirmed": False}
    volume_teardown: dict[str, Any] | None = None
    verification: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    error_type: str | None = None
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
            raise RuntimeError("runpod_network_volume_create_failed_or_ambiguous")
        get_http, volume_row = _runpod_call(
            "GET", f"/networkvolumes/{volume_id}", None, key=key, timeout=30
        )
        if get_http != 200 or _mapping(volume_row).get("dataCenterId") != data_center_id:
            raise RuntimeError("runpod_network_volume_post_create_verification_failed")
        write_json(
            output / "network_volume_evidence.json",
            {
                "schema_version": "groot_oscar_runpod_network_volume_evidence.v1",
                "status": "verified",
                "provider": "runpod",
                "provider_api_verified": True,
                "id": volume_id,
                "data_center_id": data_center_id,
                "size_bytes": volume_size_gib * 1024**3,
                "model_cache_path": MODEL_CACHE_PATH,
                "raw_provider_response_recorded": False,
            },
        )
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
                "HF_TOKEN": _read_secret(hf_token_file),
                "BLUEPRINT_MODEL_VOLUME_EVIDENCE_TOKEN": evidence_token,
                "BLUEPRINT_GROOT_OSCAR_PROVIDER_VOLUME_ID": volume_id,
            },
        }
        pod_http, pod_response = _runpod_call(
            "POST", "/pods", pod_body, key=key, timeout=90
        )
        pod_id = _extract_id(_mapping(pod_response))
        if pod_http not in {200, 201} or not pod_id:
            raise RuntimeError("runpod_model_volume_pod_create_failed_or_ambiguous")
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
            {"error_type": error_type, "error": str(exc), "raw_secret_values_recorded": False},
        )
    finally:
        if pod_id:
            pod_teardown = _delete_pod(key=key, pod_id=pod_id)
        else:
            matching_pods, _ = _matching_resources(
                key=key, pod_prefix=pod_prefix, volume_name=volume_name
            )
            for item in matching_pods:
                pod_teardown = _delete_pod(key=key, pod_id=item)
        if not success:
            volume_ids = [volume_id] if volume_id else []
            if not volume_ids:
                _, volume_ids = _matching_resources(
                    key=key, pod_prefix=pod_prefix, volume_name=volume_name
                )
            for item in volume_ids:
                volume_teardown = _delete_volume(key=key, volume_id=item)
        if success and pod_teardown.get("provider_absence_confirmed") is True:
            write_json(
                output / "watchdog_handoff.json",
                {
                    "status": "verified_volume_handed_to_gpu_canary",
                    "volume_id": volume_id,
                    "preparation_pod_absence_confirmed": True,
                },
            )
    elapsed = max(0.0, time.time() - started)
    result_success = bool(success and pod_teardown.get("provider_absence_confirmed") is True)
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
        "maximum_compute_spend_usd": hourly_rate * elapsed / 3600,
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
