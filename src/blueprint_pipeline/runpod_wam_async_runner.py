"""Short-lived RunPod WAM runner for OSCAR/Cosmos provider bundles."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse, urlunparse

from .common import ensure_dir, utc_now_iso, write_json
from .runpod_provider_adapter import (
    RUNPOD_API_GATE_ENV,
    RUNPOD_API_KEY_FILE_ENV,
    RUNPOD_API_KEY_ENV,
    RUNPOD_REST_API_BASE,
)
from .vast_bundle_staging import (
    BUNDLE_ROUTE,
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_SECRET_ENV_FILE,
    DEFAULT_TOKEN_FILE,
    OUTPUT_ROUTE,
    _read_or_create_token,
    _url_with_token,
    prepare_vast_bundle_staging,
    run_local_staging_self_test,
    verify_public_staging_urls,
)
from .vast_provider_adapter import _inspect_provider_runtime_output_zip
from .vast_wam_authorized_runner import DEFAULT_WAM_PUBLIC_IMAGE


RUNPOD_WAM_STATE_SCHEMA_VERSION = "runpod_wam_async_state.v1"
RUNPOD_WAM_CREATE_SCHEMA_VERSION = "runpod_wam_async_create_manifest.v1"
RUNPOD_WAM_POLL_SCHEMA_VERSION = "runpod_wam_async_poll_manifest.v1"
RUNPOD_WAM_DELETE_SCHEMA_VERSION = "runpod_wam_async_delete_manifest.v1"
RUNPOD_POD_LAUNCH_GATE_ENV = "BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH"
DEFAULT_GPU_TYPE_IDS = (
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
)


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _redact_provider_url(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return "<redacted-url>" if value else ""
    query = "REDACTED_QUERY" if parsed.query else ""
    fragment = "REDACTED_FRAGMENT" if parsed.fragment else ""
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", query, fragment))


def _state_path(job_dir: Path) -> Path:
    return job_dir / "runpod_wam_async_state.json"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, Mapping) else {}


def _read_runpod_api_key() -> tuple[str, dict[str, Any]]:
    env_value = _string(os.getenv(RUNPOD_API_KEY_ENV))
    if env_value:
        return env_value, {
            "api_key_configured": True,
            "api_key_source": RUNPOD_API_KEY_ENV,
            "api_key_file_configured": False,
            "raw_secret_values_recorded": False,
        }
    key_file = Path(
        _string(os.getenv(RUNPOD_API_KEY_FILE_ENV))
        or "~/.blueprint-secrets/runpod_api_key"
    ).expanduser()
    mode = oct(key_file.stat().st_mode & 0o777) if key_file.exists() else None
    try:
        key = key_file.read_text(encoding="utf-8").strip() if key_file.is_file() else ""
    except OSError as exc:
        return "", {
            "api_key_configured": False,
            "api_key_source": RUNPOD_API_KEY_FILE_ENV,
            "api_key_file_configured": True,
            "api_key_file_path": str(key_file),
            "api_key_file_mode": mode,
            "api_key_file_read_error": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
    return key, {
        "api_key_configured": bool(key),
        "api_key_source": RUNPOD_API_KEY_FILE_ENV if key else None,
        "api_key_file_configured": True,
        "api_key_file_path": str(key_file),
        "api_key_file_mode": mode,
        "api_key_file_mode_is_0600": mode == "0o600",
        "raw_secret_values_recorded": False,
    }


def _read_sensitive_url_file(path_value: str, *, label: str) -> tuple[str, dict[str, Any]]:
    if not _string(path_value):
        return "", {
            "label": label,
            "configured": False,
            "present": False,
            "raw_secret_values_recorded": False,
        }
    path = Path(path_value).expanduser().resolve()
    mode = oct(path.stat().st_mode & 0o777) if path.exists() else None
    try:
        value = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    except OSError as exc:
        return "", {
            "label": label,
            "configured": True,
            "path": str(path),
            "present": path.exists(),
            "mode": mode,
            "read_error": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
    return value, {
        "label": label,
        "configured": True,
        "path": str(path),
        "present": path.is_file(),
        "mode": mode,
        "mode_is_0600": mode == "0o600",
        "value_present": bool(value),
        "raw_secret_values_recorded": False,
    }


def _runpod_request(
    *,
    method: str,
    path: str,
    api_key: str,
    payload: Mapping[str, Any] | None = None,
    timeout_seconds: int = 45,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(dict(payload or {})).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{RUNPOD_REST_API_BASE}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = int(getattr(response, "status", 200))
        text = response.read().decode("utf-8", errors="replace")
    if not text.strip():
        return status, {}
    parsed = json.loads(text)
    return status, dict(parsed) if isinstance(parsed, Mapping) else {"response": parsed}


def _redacted_payload_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = _mapping(payload)
    return {
        "cloudType": body.get("cloudType"),
        "computeType": body.get("computeType"),
        "gpuCount": body.get("gpuCount"),
        "gpuTypeIds": body.get("gpuTypeIds"),
        "gpuTypePriority": body.get("gpuTypePriority"),
        "volumeInGb": body.get("volumeInGb"),
        "containerDiskInGb": body.get("containerDiskInGb"),
        "minVCPUPerGPU": body.get("minVCPUPerGPU"),
        "minRAMPerGPU": body.get("minRAMPerGPU"),
        "name": body.get("name"),
        "imageName": body.get("imageName"),
        "dockerEntrypoint": body.get("dockerEntrypoint"),
        "dockerStartCmd_present": bool(body.get("dockerStartCmd")),
        "env_keys": sorted((_mapping(body.get("env")) or {}).keys()),
        "raw_secret_values_recorded": False,
    }


def _provider_shell_script() -> str:
    return r"""
set -euo pipefail
echo BLUEPRINT_RUNPOD_WAM_PROVIDER_STARTED
WORK_DIR="${BLUEPRINT_RUNPOD_WAM_WORK_DIR:-/workspace/blueprint_wam_provider}"
BUNDLE_URL="${BLUEPRINT_EVAL_MANIFEST_URI:-}"
OUTPUT_PUT_URL="${BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL:-}"
export WORK_DIR BUNDLE_URL OUTPUT_PUT_URL
if [ -z "$BUNDLE_URL" ]; then echo BLUEPRINT_RUNPOD_WAM_BLOCKED:bundle_url_missing; exit 20; fi
if [ -z "$OUTPUT_PUT_URL" ]; then echo BLUEPRINT_RUNPOD_WAM_BLOCKED:output_put_url_missing; exit 21; fi
mkdir -p "$WORK_DIR"
if command -v apt-get >/dev/null 2>&1; then
  if ! command -v git >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
    apt-get update >/tmp/blueprint_runpod_apt_update.log 2>&1 || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y git ffmpeg ca-certificates >/tmp/blueprint_runpod_apt_install.log 2>&1 || true
  fi
fi
python - <<'PY'
import os
import urllib.request
from pathlib import Path
target = Path(os.environ["WORK_DIR"]) / "wam_provider_runtime_bundle.zip"
with urllib.request.urlopen(os.environ["BUNDLE_URL"], timeout=300) as response:
    target.write_bytes(response.read())
print("BLUEPRINT_RUNPOD_WAM_BUNDLE_DOWNLOADED:%d" % target.stat().st_size)
PY
rm -rf "$WORK_DIR/wam_provider_bundle" "$WORK_DIR/runtime_output" "$WORK_DIR/wam_provider_runtime_output.zip"
python -m zipfile -e "$WORK_DIR/wam_provider_runtime_bundle.zip" "$WORK_DIR/wam_provider_bundle"
echo BLUEPRINT_RUNPOD_WAM_ENTRYPOINT_STARTED
export BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR="$WORK_DIR/runtime_output"
export BLUEPRINT_WAM_PROVIDER_BUNDLE_DIR="$WORK_DIR/wam_provider_bundle"
export BLUEPRINT_WAM_ROLLOUT_INPUT="$WORK_DIR/wam_provider_bundle/provider_runtime/wam_rollout_input_manifest.json"
bash "$WORK_DIR/wam_provider_bundle/provider_runtime/run_wam_provider_runtime.sh" || true
python - <<'PY'
import json
import os
import zipfile
from pathlib import Path
output_dir = Path(os.environ["BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR"])
zip_path = Path(os.environ["WORK_DIR"]) / "wam_provider_runtime_output.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    if output_dir.is_dir():
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir).as_posix())
    else:
        archive.writestr(
            "runtime_output_missing.json",
            json.dumps({"status": "blocked", "blockers": ["runtime_output_directory_missing"]}, indent=2),
        )
print("BLUEPRINT_RUNPOD_WAM_OUTPUT_ZIP_WRITTEN:%d" % zip_path.stat().st_size)
PY
python - <<'PY'
import os
import urllib.request
from pathlib import Path
zip_path = Path(os.environ["WORK_DIR"]) / "wam_provider_runtime_output.zip"
request = urllib.request.Request(
    os.environ["OUTPUT_PUT_URL"],
    data=zip_path.read_bytes(),
    method="PUT",
    headers={"Content-Type": "application/zip"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    response.read()
print("BLUEPRINT_RUNPOD_WAM_OUTPUT_UPLOAD_OK")
PY
echo BLUEPRINT_RUNPOD_WAM_PROVIDER_COMPLETED_OR_BLOCKED
"""


def _pod_payload(
    *,
    job_name: str,
    image_name: str,
    gpu_type_ids: Sequence[str],
    provider_bundle_url: str,
    provider_output_put_url: str,
    container_disk_gb: int,
    volume_gb: int,
) -> dict[str, Any]:
    return {
        "cloudType": "SECURE",
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": list(gpu_type_ids),
        "gpuTypePriority": "availability",
        "volumeInGb": volume_gb,
        "containerDiskInGb": container_disk_gb,
        "minVCPUPerGPU": 4,
        "minRAMPerGPU": 16,
        "name": job_name,
        "imageName": image_name,
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [_provider_shell_script()],
        "ports": [],
        "volumeMountPath": "/workspace",
        "env": {
            "BLUEPRINT_EVAL_MANIFEST_URI": provider_bundle_url,
            "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": provider_output_put_url,
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "WORK_DIR": "/workspace/blueprint_wam_provider",
        },
    }


def _extract_pod_id(response: Mapping[str, Any]) -> str:
    for key in ("id", "podId", "pod_id"):
        value = _string(response.get(key))
        if value:
            return value
    for key in ("pod", "data"):
        nested = _mapping(response.get(key))
        for nested_key in ("id", "podId", "pod_id"):
            value = _string(nested.get(nested_key))
            if value:
                return value
    return ""


def _pod_status(payload: Mapping[str, Any]) -> str:
    pod = _mapping(payload.get("pod")) or _mapping(payload.get("data")) or dict(payload)
    return (
        _string(pod.get("desiredStatus"))
        or _string(pod.get("runtimeStatus"))
        or _string(pod.get("status"))
        or _string(pod.get("machineStatus"))
        or "unknown"
    )


def _staging_urls(public_base_url: str, token_file: Path) -> tuple[str, str, dict[str, Any]]:
    token, token_status = _read_or_create_token(token_file)
    return (
        _url_with_token(public_base_url, BUNDLE_ROUTE, token),
        _url_with_token(public_base_url, OUTPUT_ROUTE, token),
        token_status,
    )


def create_runpod_wam_async_run(
    *,
    job_dir: str | Path,
    bundle_path: str | Path,
    public_base_url: str = "",
    provider_bundle_url: str = "",
    provider_output_put_url: str = "",
    provider_bundle_url_file: str | Path | None = None,
    provider_output_put_url_file: str | Path | None = None,
    token_file: str | Path | None = None,
    secret_env_file: str | Path | None = None,
    output_path: str | Path | None = None,
    allow_paid_runpod_launch: bool = False,
    skip_public_staging_verification: bool = False,
    verify_output_put_url: bool = False,
    gpu_type_ids: Sequence[str] = DEFAULT_GPU_TYPE_IDS,
    image_name: str = DEFAULT_WAM_PUBLIC_IMAGE,
    container_disk_gb: int = 80,
    volume_gb: int = 20,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    resolved_bundle = Path(bundle_path).expanduser().resolve()
    resolved_output = (
        Path(output_path).expanduser().resolve()
        if output_path
        else resolved_job_dir / DEFAULT_OUTPUT_FILENAME
    )
    resolved_token_file = (
        Path(token_file).expanduser().resolve()
        if token_file
        else Path(DEFAULT_TOKEN_FILE).expanduser().resolve()
    )
    resolved_secret_env_file = (
        Path(secret_env_file).expanduser().resolve()
        if secret_env_file
        else Path(DEFAULT_SECRET_ENV_FILE).expanduser().resolve()
    )
    ensure_dir(resolved_job_dir)
    bundle_url_from_file, bundle_url_file_meta = _read_sensitive_url_file(
        str(provider_bundle_url_file or ""),
        label="provider_bundle_url_file",
    )
    output_url_from_file, output_url_file_meta = _read_sensitive_url_file(
        str(provider_output_put_url_file or ""),
        label="provider_output_put_url_file",
    )
    if not _string(provider_bundle_url) and bundle_url_from_file:
        provider_bundle_url = bundle_url_from_file
    if not _string(provider_output_put_url) and output_url_from_file:
        provider_output_put_url = output_url_from_file
    direct_provider_urls = bool(provider_bundle_url and provider_output_put_url)
    token_status: dict[str, Any] = {
        "present": False,
        "path": str(resolved_token_file),
        "token_recorded_in_manifest": False,
        "reason": "not_required_for_explicit_provider_urls"
        if direct_provider_urls
        else "pending_staging_token_resolution",
    }
    if direct_provider_urls:
        provider_bundle_url = _string(provider_bundle_url)
        provider_output_put_url = _string(provider_output_put_url)
        staging_manifest = {
            "schema_version": "runpod_wam_direct_provider_urls.v1",
            "generated_at": generated,
            "status": "ready",
            "job_dir": str(resolved_job_dir),
            "bundle_path": str(resolved_bundle),
            "output_path": str(resolved_output),
            "provider_bundle_url_redacted": _redact_provider_url(provider_bundle_url),
            "provider_output_put_url_redacted": _redact_provider_url(provider_output_put_url),
            "provider_bundle_url_file": bundle_url_file_meta,
            "provider_output_put_url_file": output_url_file_meta,
            "explicit_provider_urls_used": True,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "runpod_wam_direct_provider_urls_manifest.json", staging_manifest)
        self_test = {
            "schema_version": "runpod_wam_local_staging_self_test.v1",
            "generated_at": generated,
            "status": "skipped",
            "reason": "explicit_provider_urls_supplied",
            "raw_secret_values_recorded": False,
        }
    else:
        provider_bundle_url, provider_output_put_url, token_status = _staging_urls(
            public_base_url,
            resolved_token_file,
        )
        staging_manifest = prepare_vast_bundle_staging(
            job_dir=resolved_job_dir,
            bundle_path=resolved_bundle,
            public_base_url=public_base_url,
            token_file=resolved_token_file,
            secret_env_file=resolved_secret_env_file,
            output_path=resolved_output,
            generated_at=generated,
        )
        self_test = run_local_staging_self_test(
            job_dir=resolved_job_dir,
            bundle_path=resolved_bundle,
            output_path=resolved_job_dir / "runpod_wam_staging_self_test_output.zip",
            token_file=resolved_token_file,
            generated_at=generated,
        )
    if skip_public_staging_verification:
        public_verification = {
            "schema_version": "vast_public_staging_verification.v1",
            "generated_at": generated,
            "completed_at": generated,
            "status": "skipped",
            "job_dir": str(resolved_job_dir),
            "reason": "skip_public_staging_verification_requested",
            "provider_bundle_url_redacted": _redact_provider_url(provider_bundle_url),
            "provider_output_put_url_redacted": _redact_provider_url(provider_output_put_url),
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_public_staging_verification.json", public_verification)
    else:
        public_verification = verify_public_staging_urls(
            job_dir=resolved_job_dir,
            provider_bundle_url=provider_bundle_url,
            provider_output_put_url=provider_output_put_url,
            bundle_path=resolved_bundle,
            output_path=resolved_output,
            max_wait_seconds=30,
            retry_interval_seconds=2,
            timeout_seconds=10,
            required_consecutive_successes=1 if direct_provider_urls else 2,
            allow_output_put_probe=verify_output_put_url or not direct_provider_urls,
            cleanup_output_probe=not direct_provider_urls,
            generated_at=generated,
        )
    api_key, api_key_meta = _read_runpod_api_key()
    blockers: list[str] = []
    if staging_manifest.get("status") != "ready":
        blockers.extend(staging_manifest.get("blockers") or ["runpod_wam_staging_not_ready"])
    if not direct_provider_urls and self_test.get("status") != "passed":
        blockers.append("runpod_wam_local_staging_self_test_failed")
    if public_verification.get("status") not in {"passed", "skipped"}:
        blockers.extend(public_verification.get("blockers") or ["runpod_wam_public_staging_not_verified"])
    if not direct_provider_urls and not _string(public_base_url).startswith("https://"):
        blockers.append("runpod_wam_public_base_url_must_be_https")
    if direct_provider_urls:
        bundle_scheme = urlparse(provider_bundle_url).scheme
        output_scheme = urlparse(provider_output_put_url).scheme
        if bundle_scheme not in {"http", "https"}:
            blockers.append("runpod_provider_bundle_url_scheme_not_http")
        if output_scheme not in {"http", "https"}:
            blockers.append("runpod_provider_output_put_url_scheme_not_http")
    elif not _string(public_base_url):
        blockers.append("runpod_public_base_url_or_explicit_provider_urls_required")
    if not allow_paid_runpod_launch:
        blockers.append("paid_runpod_launch_not_authorized_by_runner_flag")
    if not os.getenv(RUNPOD_API_GATE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        blockers.append(f"missing_env_{RUNPOD_API_GATE_ENV}")
    if not os.getenv(RUNPOD_POD_LAUNCH_GATE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        blockers.append(f"missing_env_{RUNPOD_POD_LAUNCH_GATE_ENV}")
    if not api_key:
        blockers.append(f"missing_env_{RUNPOD_API_KEY_ENV}_or_{RUNPOD_API_KEY_FILE_ENV}")
    if not resolved_bundle.is_file():
        blockers.append("runpod_wam_provider_bundle_missing")
    if blockers:
        manifest = {
            "schema_version": RUNPOD_WAM_CREATE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "blockers": sorted(set(blockers)),
            "staging_manifest_status": staging_manifest.get("status"),
            "self_test_status": self_test.get("status"),
            "public_staging_verification_status": public_verification.get("status"),
            "explicit_provider_urls_used": direct_provider_urls,
            "provider_bundle_url_redacted": _redact_provider_url(provider_bundle_url),
            "provider_output_put_url_redacted": _redact_provider_url(provider_output_put_url),
            "provider_bundle_url_file": bundle_url_file_meta,
            "provider_output_put_url_file": output_url_file_meta,
            "api_key_status": api_key_meta,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "runpod_wam_async_create_manifest.json", manifest)
        return manifest

    payload = _pod_payload(
        job_name=f"blueprint-wam-{int(time.time())}",
        image_name=image_name,
        gpu_type_ids=gpu_type_ids,
        provider_bundle_url=provider_bundle_url,
        provider_output_put_url=provider_output_put_url,
        container_disk_gb=container_disk_gb,
        volume_gb=volume_gb,
    )
    try:
        status_code, response = _runpod_request(
            method="POST",
            path="/pods",
            api_key=api_key,
            payload=payload,
            timeout_seconds=45,
        )
        pod_id = _extract_pod_id(response)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")[:500]
        manifest = {
            "schema_version": RUNPOD_WAM_CREATE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "blockers": ["runpod_create_pod_http_error"],
            "http_status_code": exc.code,
            "runpod_error_preview": "REDACTED_SECRET" if api_key in error_body else error_body,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "runpod_wam_async_create_manifest.json", manifest)
        return manifest
    if not pod_id:
        manifest = {
            "schema_version": RUNPOD_WAM_CREATE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "blockers": ["runpod_create_response_missing_pod_id"],
            "http_status_code": status_code,
            "runpod_response_keys": sorted(response.keys()),
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "runpod_wam_async_create_manifest.json", manifest)
        return manifest
    state = {
        "schema_version": RUNPOD_WAM_STATE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "pod_created",
        "job_dir": str(resolved_job_dir),
        "pod_id": pod_id,
        "output_path": str(resolved_output),
        "public_base_url_present": bool(public_base_url),
        "explicit_provider_urls_used": direct_provider_urls,
        "provider_bundle_url_redacted": _redact_provider_url(provider_bundle_url),
        "provider_output_put_url_redacted": _redact_provider_url(provider_output_put_url),
        "provider_bundle_url_file": bundle_url_file_meta,
        "provider_output_put_url_file": output_url_file_meta,
        "bundle_path": str(resolved_bundle),
        "token_file": str(resolved_token_file),
        "secret_env_file": str(resolved_secret_env_file),
        "image_name": image_name,
        "gpu_type_ids": list(gpu_type_ids),
        "container_disk_gb": container_disk_gb,
        "volume_gb": volume_gb,
        "created_at_epoch": time.time(),
        "raw_secret_values_recorded": False,
    }
    write_json(_state_path(resolved_job_dir), state)
    manifest = {
        "schema_version": RUNPOD_WAM_CREATE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "pod_created",
        "job_dir": str(resolved_job_dir),
        "pod_id": pod_id,
        "http_status_code": status_code,
        "output_path": str(resolved_output),
        "pod_request_summary": _redacted_payload_summary(payload),
        "explicit_provider_urls_used": direct_provider_urls,
        "provider_bundle_url_redacted": _redact_provider_url(provider_bundle_url),
        "provider_output_put_url_redacted": _redact_provider_url(provider_output_put_url),
        "provider_bundle_url_file": bundle_url_file_meta,
        "provider_output_put_url_file": output_url_file_meta,
        "runpod_response_keys": sorted(response.keys()),
        "poll_command": f"python -m blueprint_pipeline.runpod_wam_async_runner poll --job-dir {resolved_job_dir}",
        "teardown_command": f"python -m blueprint_pipeline.runpod_wam_async_runner poll --job-dir {resolved_job_dir} --teardown",
        "raw_secret_values_recorded": False,
    }
    write_json(resolved_job_dir / "runpod_wam_async_create_manifest.json", manifest)
    return manifest


def _delete_pod(
    *,
    job_dir: Path,
    pod_id: str,
    api_key: str,
    generated_at: str,
) -> dict[str, Any]:
    try:
        status_code, response = _runpod_request(
            method="DELETE",
            path=f"/pods/{pod_id}",
            api_key=api_key,
            timeout_seconds=30,
        )
        status = "completed" if status_code in {200, 202, 204} else "blocked"
        blockers: list[str] = [] if status == "completed" else ["runpod_delete_pod_unexpected_status"]
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        response = {}
        status = "completed" if exc.code in {404, 410} else "blocked"
        blockers = [] if status == "completed" else ["runpod_delete_pod_http_error"]
    manifest = {
        "schema_version": RUNPOD_WAM_DELETE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "job_dir": str(job_dir),
        "pod_id": pod_id,
        "http_status_code": status_code,
        "response_keys": sorted(response.keys()),
        "blockers": blockers,
        "continuing_spend_from_this_run": status != "completed",
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "runpod_wam_async_delete_manifest.json", manifest)
    return manifest


def poll_runpod_wam_async_run(
    *,
    job_dir: str | Path,
    max_wait_seconds: int = 60,
    retry_interval_seconds: int = 5,
    teardown: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    state = _read_json(_state_path(resolved_job_dir))
    pod_id = _string(state.get("pod_id"))
    output_path = Path(_string(state.get("output_path"))).expanduser()
    api_key, api_key_meta = _read_runpod_api_key()
    blockers: list[str] = []
    if not pod_id:
        blockers.append("runpod_wam_state_missing_pod_id")
    if not api_key:
        blockers.append(f"missing_env_{RUNPOD_API_KEY_ENV}_or_{RUNPOD_API_KEY_FILE_ENV}")
    status_code: int | None = None
    pod_payload: dict[str, Any] = {}
    pod_status = "unknown"
    deadline = time.monotonic() + max(0, max_wait_seconds)
    output_present = output_path.is_file()
    while not blockers and time.monotonic() <= deadline:
        output_present = output_path.is_file()
        try:
            status_code, pod_payload = _runpod_request(
                method="GET",
                path=f"/pods/{pod_id}",
                api_key=api_key,
                timeout_seconds=20,
            )
            pod_status = _pod_status(pod_payload)
        except urllib.error.HTTPError as exc:
            status_code = exc.code
            pod_status = "not_found" if exc.code in {404, 410} else "http_error"
            if exc.code not in {404, 410}:
                blockers.append("runpod_pod_status_http_error")
            break
        if output_present:
            break
        if time.monotonic() + retry_interval_seconds > deadline:
            break
        time.sleep(max(1, retry_interval_seconds))
    output_inspection = _inspect_provider_runtime_output_zip(
        output_path,
        video_extract_dir=resolved_job_dir / "runpod_wam_output_videos",
        expected_video_count=1,
    )
    output_present = output_inspection.get("zip_present") is True
    should_teardown = teardown or output_present or pod_status in {"not_found", "EXITED", "TERMINATED"}
    delete_manifest: dict[str, Any] | None = None
    if not blockers and should_teardown and pod_id and api_key and pod_status != "not_found":
        delete_manifest = _delete_pod(
            job_dir=resolved_job_dir,
            pod_id=pod_id,
            api_key=api_key,
            generated_at=generated,
        )
    continuing_spend = bool(
        pod_id
        and not output_present
        and (not should_teardown or (delete_manifest or {}).get("status") != "completed")
        and pod_status not in {"not_found", "TERMINATED", "EXITED"}
    )
    provider_status = "completed" if output_present else "blocked"
    provider_blockers: list[str] = []
    if not output_present:
        provider_blockers.append("runpod_provider_runtime_output_zip_not_received_locally")
    if blockers:
        provider_blockers.extend(blockers)
    manifest = {
        "schema_version": RUNPOD_WAM_POLL_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed" if output_present and not continuing_spend else ("running" if continuing_spend else "blocked"),
        "job_dir": str(resolved_job_dir),
        "pod_id": pod_id,
        "pod_status": pod_status,
        "pod_status_http_status_code": status_code,
        "provider_command_status": provider_status,
        "provider_command_blockers": provider_blockers,
        "output_zip_present": output_present,
        "provider_runtime_output_zip_path": str(output_path),
        "runtime_result_status": output_inspection.get("runtime_result_status"),
        "runtime_result_blockers": output_inspection.get("runtime_result_blockers"),
        "mp4_count": output_inspection.get("mp4_count"),
        "teardown_requested": teardown,
        "teardown_performed": bool(delete_manifest and delete_manifest.get("status") == "completed"),
        "continuing_spend_from_this_run": continuing_spend,
        "api_key_status": api_key_meta,
        "raw_secret_values_recorded": False,
    }
    write_json(resolved_job_dir / "runpod_wam_async_poll_manifest.json", manifest)
    state_update = {
        **state,
        "last_polled_at": generated,
        "last_pod_status": pod_status,
        "provider_command_status": provider_status,
        "provider_command_blockers": provider_blockers,
        "continuing_spend_from_this_run": continuing_spend,
        "raw_secret_values_recorded": False,
    }
    write_json(_state_path(resolved_job_dir), state_update)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--job-dir", required=True)
    create.add_argument("--bundle-path", required=True)
    create.add_argument("--public-base-url", default="")
    create.add_argument("--provider-bundle-url", default="")
    create.add_argument("--provider-output-put-url", default="")
    create.add_argument("--provider-bundle-url-file")
    create.add_argument("--provider-output-put-url-file")
    create.add_argument("--token-file")
    create.add_argument("--secret-env-file")
    create.add_argument("--output-path")
    create.add_argument("--allow-paid-runpod-launch", action="store_true")
    create.add_argument("--skip-public-staging-verification", action="store_true")
    create.add_argument("--verify-output-put-url", action="store_true")
    create.add_argument("--gpu-type-id", action="append", default=[])
    create.add_argument("--image-name", default=DEFAULT_WAM_PUBLIC_IMAGE)
    create.add_argument("--container-disk-gb", type=int, default=80)
    create.add_argument("--volume-gb", type=int, default=20)
    poll = subparsers.add_parser("poll")
    poll.add_argument("--job-dir", required=True)
    poll.add_argument("--max-wait-seconds", type=int, default=60)
    poll.add_argument("--retry-interval-seconds", type=int, default=5)
    poll.add_argument("--teardown", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "create":
        manifest = create_runpod_wam_async_run(
            job_dir=args.job_dir,
            bundle_path=args.bundle_path,
            public_base_url=args.public_base_url,
            provider_bundle_url=args.provider_bundle_url,
            provider_output_put_url=args.provider_output_put_url,
            provider_bundle_url_file=args.provider_bundle_url_file,
            provider_output_put_url_file=args.provider_output_put_url_file,
            token_file=args.token_file,
            secret_env_file=args.secret_env_file,
            output_path=args.output_path,
            allow_paid_runpod_launch=args.allow_paid_runpod_launch,
            skip_public_staging_verification=args.skip_public_staging_verification,
            verify_output_put_url=args.verify_output_put_url,
            gpu_type_ids=args.gpu_type_id or DEFAULT_GPU_TYPE_IDS,
            image_name=args.image_name,
            container_disk_gb=args.container_disk_gb,
            volume_gb=args.volume_gb,
        )
    else:
        manifest = poll_runpod_wam_async_run(
            job_dir=args.job_dir,
            max_wait_seconds=args.max_wait_seconds,
            retry_interval_seconds=args.retry_interval_seconds,
            teardown=args.teardown,
        )
    print(json.dumps(manifest, sort_keys=True))
    return 0 if manifest.get("status") in {"pod_created", "running", "completed"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
