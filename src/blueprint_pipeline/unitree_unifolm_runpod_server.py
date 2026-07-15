"""Launch a long-lived RunPod Unitree UnifoLM VLA server."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .runpod_provider_adapter import RUNPOD_API_GATE_ENV
from .runpod_wam_async_runner import (
    DEFAULT_GPU_TYPE_IDS,
    RUNPOD_POD_LAUNCH_GATE_ENV,
    _delete_pod,
    _extract_pod_id,
    _read_model_secret_env,
    _read_runpod_api_key,
    _runpod_request,
)


SCHEMA_VERSION = "unitree_unifolm_runpod_server.v1"
STATE_SCHEMA_VERSION = "unitree_unifolm_runpod_server_state.v1"
DEFAULT_IMAGE = "docker.io/nijelhunt/blueprint-unitree-unifolm:20260622-cu124-sdpa3"
DEFAULT_PORT = 8777
DEFAULT_BACKEND_PORT = 8778
DEFAULT_CONTAINER_DISK_GB = 160
DEFAULT_VOLUME_GB = 80
DEFAULT_VLA_REPO = "unitreerobotics/UnifoLM-VLA-Base"
DEFAULT_VLM_REPO = "unitreerobotics/UnifoLM-VLM-Base"
DEFAULT_UNNORM_KEY = "g1_stack_block"
RUNPOD_UNIFOLM_MAX_SPEND_USD_ENV = "BLUEPRINT_RUNPOD_UNITREE_UNIFOLM_MAX_SPEND_USD"


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _state_path(job_dir: Path) -> Path:
    return job_dir / "unitree_unifolm_runpod_server_state.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, Mapping) else {}


def _server_proxy_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{int(port)}.proxy.runpod.net/act"


def _redacted_payload_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = _mapping(payload)
    env = _mapping(body.get("env"))
    return {
        "cloudType": body.get("cloudType"),
        "computeType": body.get("computeType"),
        "gpuCount": body.get("gpuCount"),
        "gpuTypeIds": body.get("gpuTypeIds"),
        "containerDiskInGb": body.get("containerDiskInGb"),
        "volumeInGb": body.get("volumeInGb"),
        "imageName": body.get("imageName"),
        "ports": body.get("ports"),
        "dockerEntrypoint": body.get("dockerEntrypoint"),
        "dockerStartCmd": "<unitree_unifolm_status_proxy_wrapper>",
        "dockerStartCmdLength": len(str(body.get("dockerStartCmd") or "")),
        "env_keys": sorted(env),
        "raw_secret_values_recorded": False,
        "secret_hashes_recorded": False,
    }


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _unitree_unifolm_prelaunch_spend_guard(
    *,
    max_spend_usd: float | None,
    allow_paid_runpod_launch: bool,
    gpu_type_ids: Sequence[str],
    container_disk_gb: int,
    volume_gb: int,
) -> dict[str, Any]:
    env_budget = _float_or_none(os.getenv(RUNPOD_UNIFOLM_MAX_SPEND_USD_ENV))
    requested_budget = max_spend_usd if max_spend_usd is not None else env_budget
    api_gate_present = _env_truthy(RUNPOD_API_GATE_ENV)
    pod_gate_present = _env_truthy(RUNPOD_POD_LAUNCH_GATE_ENV)
    blockers: list[str] = []
    if not allow_paid_runpod_launch:
        blockers.append("paid_runpod_launch_not_authorized_by_runner_flag")
    if not api_gate_present:
        blockers.append(f"missing_env_{RUNPOD_API_GATE_ENV}")
    if not pod_gate_present:
        blockers.append(f"missing_env_{RUNPOD_POD_LAUNCH_GATE_ENV}")
    if requested_budget is None:
        blockers.append("unitree_unifolm_runpod_max_spend_usd_missing")
    elif requested_budget <= 0:
        blockers.append("unitree_unifolm_runpod_max_spend_usd_must_be_positive")
    if not gpu_type_ids:
        blockers.append("unitree_unifolm_runpod_gpu_type_ids_missing")
    if container_disk_gb <= 0:
        blockers.append("unitree_unifolm_runpod_container_disk_gb_invalid")
    if volume_gb < 0:
        blockers.append("unitree_unifolm_runpod_volume_gb_invalid")
    can_launch = not blockers
    return {
        "schema_version": "unitree_unifolm_runpod_prelaunch_spend_guard.v1",
        "status": "passed" if can_launch else "blocked",
        "required_before_provider_launch": True,
        "can_launch": can_launch,
        "requested_budget_usd": requested_budget,
        "budget_source": "argument"
        if max_spend_usd is not None
        else ("env" if env_budget is not None else "missing"),
        "single_pod_launch": True,
        "max_active_workers": 1,
        "gpu_type_ids": list(gpu_type_ids),
        "container_disk_gb": int(container_disk_gb),
        "volume_gb": int(volume_gb),
        "checks": {
            "allow_paid_runpod_launch_flag_present": allow_paid_runpod_launch,
            f"env_{RUNPOD_API_GATE_ENV}_present": api_gate_present,
            f"env_{RUNPOD_POD_LAUNCH_GATE_ENV}_present": pod_gate_present,
            "requested_budget_declared": requested_budget is not None,
            "requested_budget_positive": (
                requested_budget is not None and requested_budget > 0
            ),
            "single_pod_launch": True,
            "teardown_command_written_after_create": True,
        },
        "blockers": sorted(set(blockers)),
        "truth_boundary": {
            "budget_declared_before_provider_launch": True,
            "runpod_billing_cap_enforced_by_api": False,
            "spend_ledger_still_required_after_launch": True,
        },
    }


def _status_proxy_start_command(*, public_port: int, backend_port: int) -> str:
    return f"""cat >/tmp/blueprint_unitree_unifolm_proxy.py <<'PY'
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PUBLIC_PORT = int(os.getenv("BLUEPRINT_UNITREE_UNIFOLM_PROXY_PORT", "{int(public_port)}"))
BACKEND_PORT = int(os.getenv("BLUEPRINT_UNITREE_UNIFOLM_BACKEND_PORT", "{int(backend_port)}"))
LOG_PATH = Path(os.getenv("BLUEPRINT_UNITREE_UNIFOLM_SERVER_LOG", "/workspace/blueprint_unitree_unifolm_server.log"))
env = os.environ.copy()
env["BLUEPRINT_UNITREE_UNIFOLM_VLA_SERVER_PORT"] = str(BACKEND_PORT)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
log = LOG_PATH.open("ab", buffering=0)
proc = None
backend_start_error = ""

def start_backend() -> None:
    global proc, backend_start_error
    try:
        proc = subprocess.Popen(["run_unitree_unifolm_vla_server"], stdout=log, stderr=subprocess.STDOUT, env=env)
    except Exception as exc:
        backend_start_error = f"{{type(exc).__name__}}: {{exc}}"

def log_tail() -> str:
    try:
        data = LOG_PATH.read_bytes()[-12000:]
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")

def status_payload() -> dict[str, object]:
    running = proc is not None and proc.poll() is None
    return {{
        "schema_version": "unitree_unifolm_runpod_status_proxy.v1",
        "status": "starting" if running else "backend_unavailable",
        "backend_process_running": running,
        "backend_returncode": None if proc is None else proc.poll(),
        "backend_start_error": backend_start_error,
        "backend_url": f"http://127.0.0.1:{{BACKEND_PORT}}/act",
        "log_path": str(LOG_PATH),
        "log_tail": log_tail(),
        "raw_secret_values_recorded": False,
        "secret_hashes_recorded": False,
    }}

class Handler(BaseHTTPRequestHandler):
    def _write_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._write_json(200, status_payload())

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/act":
            self._write_json(404, {{"status": "blocked", "blockers": ["unknown_path"], "raw_secret_values_recorded": False}})
            return
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length)
        if proc is None or proc.poll() is not None:
            blockers = ["backend_not_running"]
            if backend_start_error:
                blockers.append("backend_start_error")
            self._write_json(
                503,
                {{
                    **status_payload(),
                    "status": "blocked",
                    "blockers": blockers,
                }},
            )
            return
        request = urllib.request.Request(
            f"http://127.0.0.1:{{BACKEND_PORT}}/act",
            data=body,
            method="POST",
            headers={{"Content-Type": self.headers.get("Content-Type", "application/json"), "User-Agent": "BlueprintUnitreeUnifoLMRunPodProxy/1.0"}},
        )
        try:
            with urllib.request.urlopen(request, timeout=95) as response:
                payload = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except Exception as exc:
            self._write_json(
                503,
                {{
                    **status_payload(),
                    "status": "blocked",
                    "blockers": [f"backend_not_ready:{{type(exc).__name__}}"],
                }},
            )

    def log_message(self, _format: str, *_args: object) -> None:
        return

threading.Thread(target=start_backend, daemon=True).start()
ThreadingHTTPServer(("0.0.0.0", PUBLIC_PORT), Handler).serve_forever()
PY
python3 /tmp/blueprint_unitree_unifolm_proxy.py"""


def _server_env(
    *,
    port: int,
    backend_port: int,
    vla_checkpoint: str,
    vlm_checkpoint: str,
    unnorm_key: str,
    attention_implementation: str,
    allow_hf_download: bool,
    model_cache_root: str,
    model_secret_env: Mapping[str, str],
) -> dict[str, str]:
    env = {
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_SERVER_PORT": str(int(port)),
        "BLUEPRINT_UNITREE_UNIFOLM_PROXY_PORT": str(int(port)),
        "BLUEPRINT_UNITREE_UNIFOLM_BACKEND_PORT": str(int(backend_port)),
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT": vla_checkpoint,
        "BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT": vla_checkpoint,
        "BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT": vlm_checkpoint,
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_REPO": DEFAULT_VLA_REPO,
        "BLUEPRINT_UNITREE_UNIFOLM_VLM_REPO": DEFAULT_VLM_REPO,
        "BLUEPRINT_UNITREE_UNIFOLM_UNNORM_KEY": unnorm_key,
        "BLUEPRINT_UNITREE_UNIFOLM_ALLOW_HF_DOWNLOAD": (
            "true" if allow_hf_download else "false"
        ),
        "BLUEPRINT_UNITREE_UNIFOLM_MODEL_CACHE_ROOT": model_cache_root,
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_ATTENTION_IMPLEMENTATION": attention_implementation,
    }
    env.update({key: value for key, value in model_secret_env.items() if _string(value)})
    return env


def _pod_payload(
    *,
    image_name: str,
    gpu_type_ids: Sequence[str],
    port: int,
    container_disk_gb: int,
    volume_gb: int,
    cloud_type: str,
    env: Mapping[str, str],
    backend_port: int,
) -> dict[str, Any]:
    return {
        "cloudType": cloud_type,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": list(gpu_type_ids),
        "gpuTypePriority": "availability",
        "volumeInGb": int(volume_gb),
        "containerDiskInGb": int(container_disk_gb),
        "minVCPUPerGPU": 4,
        "minRAMPerGPU": 16,
        "name": f"blueprint-unitree-unifolm-server-{int(time.time())}",
        "imageName": image_name,
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [
            _status_proxy_start_command(public_port=port, backend_port=backend_port)
        ],
        "ports": [f"{int(port)}/http"],
        "volumeMountPath": "/workspace",
        "env": dict(env),
    }


def launch_unitree_unifolm_runpod_server(
    *,
    job_dir: str | Path | None = None,
    image_name: str = DEFAULT_IMAGE,
    gpu_type_ids: Sequence[str] = DEFAULT_GPU_TYPE_IDS,
    port: int = DEFAULT_PORT,
    backend_port: int = DEFAULT_BACKEND_PORT,
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
    volume_gb: int = DEFAULT_VOLUME_GB,
    cloud_type: str = "SECURE",
    vla_checkpoint: str = DEFAULT_VLA_REPO,
    vlm_checkpoint: str = DEFAULT_VLM_REPO,
    unnorm_key: str = DEFAULT_UNNORM_KEY,
    attention_implementation: str = "sdpa",
    allow_hf_download: bool = True,
    model_cache_root: str = "/workspace/models",
    max_spend_usd: float | None = None,
    allow_paid_runpod_launch: bool = False,
    generated_at: str | None = None,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    output = Path(
        job_dir
        or _repo_root()
        / "robot_eval_jobs"
        / f"unitree_unifolm_runpod_server_{_timestamp()}"
    ).expanduser().resolve()
    ensure_dir(output)
    api_key, api_key_status = _read_runpod_api_key()
    model_secret_env, model_secret_env_status = _read_model_secret_env()
    env = _server_env(
        port=port,
        backend_port=backend_port,
        vla_checkpoint=vla_checkpoint,
        vlm_checkpoint=vlm_checkpoint,
        unnorm_key=unnorm_key,
        attention_implementation=attention_implementation,
        allow_hf_download=allow_hf_download,
        model_cache_root=model_cache_root,
        model_secret_env=model_secret_env,
    )
    payload = _pod_payload(
        image_name=image_name,
        gpu_type_ids=gpu_type_ids,
        port=port,
        container_disk_gb=container_disk_gb,
        volume_gb=volume_gb,
        cloud_type=cloud_type,
        env=env,
        backend_port=backend_port,
    )
    prelaunch_spend_guard = _unitree_unifolm_prelaunch_spend_guard(
        max_spend_usd=max_spend_usd,
        allow_paid_runpod_launch=allow_paid_runpod_launch,
        gpu_type_ids=gpu_type_ids,
        container_disk_gb=container_disk_gb,
        volume_gb=volume_gb,
    )
    blockers: list[str] = []
    if prelaunch_spend_guard.get("can_launch") is not True:
        blockers.append("unitree_unifolm_runpod_prelaunch_spend_guard_not_passed")
        blockers.extend(prelaunch_spend_guard.get("blockers") or [])
    if not allow_paid_runpod_launch:
        blockers.append("paid_runpod_launch_not_authorized_by_runner_flag")
    if not _env_truthy(RUNPOD_API_GATE_ENV):
        blockers.append(f"missing_env_{RUNPOD_API_GATE_ENV}")
    if not _env_truthy(RUNPOD_POD_LAUNCH_GATE_ENV):
        blockers.append(f"missing_env_{RUNPOD_POD_LAUNCH_GATE_ENV}")
    if not api_key:
        blockers.append("missing_runpod_api_key_or_file")
    if not image_name:
        blockers.append("missing_unitree_unifolm_image_name")
    if blockers:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(output),
            "blockers": sorted(set(blockers)),
            "prelaunch_spend_guard": prelaunch_spend_guard,
            "api_key_status": api_key_status,
            "model_secret_env_status": model_secret_env_status,
            "redacted_pod_payload": _redacted_payload_summary(payload),
            "raw_secret_values_recorded": False,
            "secret_hashes_recorded": False,
        }
        write_json(output / "unitree_unifolm_runpod_server_launch_manifest.json", manifest)
        return manifest
    try:
        require_paid_resource_admission_grant(
            paid_resource_admission_grant,
            resource_class="unitree_unifolm_runpod",
        )
    except PaidResourceAdmissionBlocked as exc:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(output),
            "blockers": [
                "unitree_unifolm_shared_admission_missing_or_invalid",
                *exc.blockers,
            ],
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
        write_json(output / "unitree_unifolm_runpod_server_launch_manifest.json", manifest)
        return manifest
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
        error_body = exc.read().decode("utf-8", errors="replace")[:800]
        error_preview = "REDACTED_SECRET" if api_key and api_key in error_body else error_body
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(output),
            "http_status_code": exc.code,
            "blockers": ["runpod_unitree_unifolm_server_create_http_error"],
            "runpod_error_preview": error_preview,
            "api_key_status": api_key_status,
            "model_secret_env_status": model_secret_env_status,
            "redacted_pod_payload": _redacted_payload_summary(payload),
            "raw_secret_values_recorded": False,
            "secret_hashes_recorded": False,
        }
        write_json(output / "unitree_unifolm_runpod_server_launch_manifest.json", manifest)
        return manifest
    if not pod_id:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(output),
            "http_status_code": status_code,
            "blockers": ["runpod_unitree_unifolm_server_create_response_missing_pod_id"],
            "response_keys": sorted(response),
            "api_key_status": api_key_status,
            "model_secret_env_status": model_secret_env_status,
            "redacted_pod_payload": _redacted_payload_summary(payload),
            "raw_secret_values_recorded": False,
            "secret_hashes_recorded": False,
        }
        write_json(output / "unitree_unifolm_runpod_server_launch_manifest.json", manifest)
        return manifest
    server_url = _server_proxy_url(pod_id, port)
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "pod_created",
        "job_dir": str(output),
        "pod_id": pod_id,
        "port": int(port),
        "backend_port": int(backend_port),
        "server_url": server_url,
        "image_name": image_name,
        "max_spend_usd": prelaunch_spend_guard.get("requested_budget_usd"),
        "continuing_spend_from_this_run": True,
        "delete_command": (
            f"python -m blueprint_pipeline.unitree_unifolm_runpod_server delete "
            f"--job-dir {output}"
        ),
        "raw_secret_values_recorded": False,
        "secret_hashes_recorded": False,
    }
    write_json(_state_path(output), state)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "status": "pod_created",
        "job_dir": str(output),
        "pod_id": pod_id,
        "server_url": server_url,
        "http_status_code": status_code,
        "api_key_status": api_key_status,
        "model_secret_env_status": model_secret_env_status,
        "redacted_pod_payload": _redacted_payload_summary(payload),
        "prelaunch_spend_guard": prelaunch_spend_guard,
        "state_path": str(_state_path(output)),
        "local_bridge_command": (
            "python -m blueprint_pipeline.unitree_unifolm_vla_server_bridge "
            f"--server-url {server_url}"
        ),
        "endpoint_policy_command": (
            "python -m blueprint_pipeline.unitree_unifolm_policy_command_adapter "
            "--mode vla "
            "--command 'python -m blueprint_pipeline.unitree_unifolm_vla_server_bridge "
            f"--server-url {server_url}' "
            f"--checkpoint {vla_checkpoint} --vlm-checkpoint {vlm_checkpoint}"
        ),
        "truth_boundary": {
            "pod_created_is_not_model_loaded": True,
            "server_url_must_answer_act_before_policy_proof": True,
            "generated_world_rank_fidelity_result_proven": False,
            "raw_secret_values_recorded": False,
        },
        "raw_secret_values_recorded": False,
        "secret_hashes_recorded": False,
    }
    write_json(output / "unitree_unifolm_runpod_server_launch_manifest.json", manifest)
    return manifest


def poll_unitree_unifolm_runpod_server(
    *,
    job_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    job = Path(job_dir).expanduser().resolve()
    state = _read_json(_state_path(job))
    api_key, api_key_status = _read_runpod_api_key()
    pod_id = _string(state.get("pod_id"))
    if not pod_id or not api_key:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(job),
            "blockers": [
                "missing_unitree_unifolm_runpod_server_state_or_api_key",
            ],
            "api_key_status": api_key_status,
            "raw_secret_values_recorded": False,
            "secret_hashes_recorded": False,
        }
        write_json(job / "unitree_unifolm_runpod_server_poll_manifest.json", manifest)
        return manifest
    try:
        status_code, response = _runpod_request(
            method="GET",
            path=f"/pods/{pod_id}",
            api_key=api_key,
            timeout_seconds=30,
        )
        status = "running" if status_code == 200 else "blocked"
        blockers: list[str] = [] if status == "running" else ["runpod_unitree_server_poll_failed"]
    except urllib.error.HTTPError as exc:
        response = {}
        status_code = exc.code
        status = "blocked"
        blockers = ["runpod_unitree_server_poll_http_error"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "job_dir": str(job),
        "pod_id": pod_id,
        "server_url": state.get("server_url"),
        "http_status_code": status_code,
        "response_status": response.get("status") or response.get("desiredStatus"),
        "ports": response.get("ports"),
        "port_mappings_present": bool(response.get("portMappings")),
        "blockers": blockers,
        "api_key_status": api_key_status,
        "raw_secret_values_recorded": False,
        "secret_hashes_recorded": False,
    }
    write_json(job / "unitree_unifolm_runpod_server_poll_manifest.json", manifest)
    return manifest


def _server_status_url(server_url: str) -> str:
    if server_url.rstrip("/").endswith("/act"):
        return server_url.rstrip("/")[: -len("/act")] + "/status"
    return server_url.rstrip("/") + "/status"


def probe_unitree_unifolm_runpod_server(
    *,
    job_dir: str | Path,
    server_url: str | None = None,
    timeout_seconds: float = 20.0,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    job = Path(job_dir).expanduser().resolve()
    state = _read_json(_state_path(job))
    resolved_server_url = _string(server_url) or _string(state.get("server_url"))
    blockers: list[str] = []
    if not resolved_server_url:
        blockers.append("missing_unitree_unifolm_runpod_server_url")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(job),
            "blockers": blockers,
            "raw_secret_values_recorded": False,
            "secret_hashes_recorded": False,
        }
        write_json(job / "unitree_unifolm_runpod_server_proxy_probe.json", manifest)
        return manifest
    status_url = _server_status_url(resolved_server_url)
    payload: dict[str, Any] = {}
    http_status_code: int | None = None
    try:
        request = urllib.request.Request(
            status_url,
            headers={"User-Agent": "BlueprintUnitreeUnifoLMRunPodProbe/1.0"},
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            http_status_code = response.status
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        http_status_code = exc.code
        error_preview = exc.read().decode("utf-8", errors="replace")[:800]
        blockers.append("blocked_runpod_status_proxy_http_error")
        payload = {
            "error_type": "HTTPError",
            "error_preview": error_preview,
        }
    except Exception as exc:
        blockers.append(f"blocked_runpod_status_proxy_probe_failed:{type(exc).__name__}")
        payload = {"error_type": type(exc).__name__, "error_preview": str(exc)[:300]}
    backend_running = bool(payload.get("backend_process_running"))
    backend_error = _string(payload.get("backend_start_error"))
    if not blockers and backend_error:
        blockers.append("blocked_unitree_unifolm_backend_start_error")
    if not blockers and not backend_running:
        blockers.append("blocked_unitree_unifolm_backend_not_running")
    status = "running" if not blockers else "blocked"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "job_dir": str(job),
        "server_url": resolved_server_url,
        "status_url": status_url,
        "http_status_code": http_status_code,
        "backend_process_running": backend_running,
        "backend_start_error_present": bool(backend_error),
        "status_payload": payload,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
        "secret_hashes_recorded": False,
    }
    write_json(job / "unitree_unifolm_runpod_server_proxy_probe.json", manifest)
    return manifest


def delete_unitree_unifolm_runpod_server(
    *,
    job_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    job = Path(job_dir).expanduser().resolve()
    state = _read_json(_state_path(job))
    api_key, api_key_status = _read_runpod_api_key()
    pod_id = _string(state.get("pod_id"))
    if not pod_id or not api_key:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(job),
            "blockers": [
                "missing_unitree_unifolm_runpod_server_state_or_api_key",
            ],
            "api_key_status": api_key_status,
            "raw_secret_values_recorded": False,
            "secret_hashes_recorded": False,
        }
        write_json(job / "unitree_unifolm_runpod_server_delete_manifest.json", manifest)
        return manifest
    manifest = _delete_pod(job_dir=job, pod_id=pod_id, api_key=api_key, generated_at=generated)
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["server_state_path"] = str(_state_path(job))
    write_json(job / "unitree_unifolm_runpod_server_delete_manifest.json", manifest)
    return manifest


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--job-dir")
    launch.add_argument("--image-name", default=os.getenv("BLUEPRINT_UNITREE_UNIFOLM_GPU_IMAGE_REF", DEFAULT_IMAGE))
    launch.add_argument("--gpu-type-id", action="append", default=[])
    launch.add_argument("--port", type=int, default=DEFAULT_PORT)
    launch.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT)
    launch.add_argument("--container-disk-gb", type=int, default=DEFAULT_CONTAINER_DISK_GB)
    launch.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB)
    launch.add_argument("--cloud-type", choices=("SECURE", "COMMUNITY"), default="SECURE")
    launch.add_argument("--vla-checkpoint", default=DEFAULT_VLA_REPO)
    launch.add_argument("--vlm-checkpoint", default=DEFAULT_VLM_REPO)
    launch.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    launch.add_argument("--attention-implementation", default="sdpa")
    launch.add_argument("--model-cache-root", default="/workspace/models")
    launch.add_argument(
        "--max-spend-usd",
        type=float,
        default=_float_or_none(os.getenv(RUNPOD_UNIFOLM_MAX_SPEND_USD_ENV)),
        help=(
            "Required positive prelaunch budget declaration before creating "
            "a RunPod pod. Can also be set with "
            f"{RUNPOD_UNIFOLM_MAX_SPEND_USD_ENV}."
        ),
    )
    launch.add_argument("--disable-hf-download", action="store_true")
    launch.add_argument("--allow-paid-runpod-launch", action="store_true")
    poll = subparsers.add_parser("poll")
    poll.add_argument("--job-dir", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--job-dir", required=True)
    probe.add_argument("--server-url")
    probe.add_argument("--timeout-seconds", type=float, default=20.0)
    delete = subparsers.add_parser("delete")
    delete.add_argument("--job-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    if args.command == "launch":
        print("legacy_unitree_unifolm_runpod_launch_cli_disabled", file=sys.stderr)
        return 2
    if args.command == "poll":
        manifest = poll_unitree_unifolm_runpod_server(job_dir=args.job_dir)
    elif args.command == "probe":
        manifest = probe_unitree_unifolm_runpod_server(
            job_dir=args.job_dir,
            server_url=args.server_url,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        manifest = delete_unitree_unifolm_runpod_server(job_dir=args.job_dir)
    print(json.dumps(manifest, sort_keys=True))
    return 0 if manifest.get("status") in {"pod_created", "running", "completed"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
