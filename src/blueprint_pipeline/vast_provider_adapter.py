"""Vast.ai startup probe for Blueprint robot-eval GPU lanes.

The adapter is intentionally separate from the RunPod adapter. It can build a
dry-run request plan and, behind explicit gates, run a bounded Vast instance
startup probe that records heartbeat, GPU sanity, optional Isaac smoke, and
teardown artifacts without promoting any of those into rank-fidelity proof.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import ipaddress
import json
import logging
import os
import re
import shlex
import signal
import shutil
import socket
import subprocess
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence
from urllib.parse import parse_qs, quote, urlencode, urlparse, urlunparse

from .common import ensure_dir, utc_now_iso, write_json
from .lane_hardware_requirements import KNOWN_GPU_VRAM_GB
from .isaac_driver_support import (
    driver_newer_branch_sort_rank as _driver_newer_branch_sort_rank,
    driver_sort_rank as _driver_sort_rank,
    isaac_driver_support_status as _isaac_driver_support_status,
)
from . import vast_compute_capability as vcc
from .gpu_selection_policy import (
    GPU_SELECTION_POLICIES,
    _is_disallowed_for_isaac,
    _is_isaac_rt_candidate,
    gpu_allowed_by_policy,
    policy_manifest,
    resolve_gpu_selection_policy,
)
from .logging_utils import log_event
from .paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .provider_worker_endpoint_manifest import write_provider_worker_endpoint_manifest
from .wam_provider_output import (
    inspect_provider_runtime_output_zip,
    probe_mp4_video,
    summarize_runtime_result,
)


VAST_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION = "vast_provider_adapter_result.v1"
VAST_RUNTIME_DISCOVERY_SCHEMA_VERSION = "vast_runtime_discovery.v1"
VAST_PROVIDER_PLAN_SCHEMA_VERSION = "vast_provider_plan.v1"
VAST_OFFER_SELECTION_SCHEMA_VERSION = "vast_offer_selection_manifest.v1"
VAST_BUDGET_LEDGER_SCHEMA_VERSION = "vast_budget_ledger.v1"
VAST_PHASE_LOG_SCHEMA_VERSION = "vast_runtime_phase_log.v1"
VAST_STARTUP_PROBE_SCHEMA_VERSION = "vast_startup_probe_manifest.v1"
VAST_GPU_SANITY_SCHEMA_VERSION = "vast_gpu_sanity_report.v1"
VAST_ISAAC_SMOKE_SCHEMA_VERSION = "vast_isaac_smoke_result.v1"
VAST_PROVIDER_COMMAND_SCHEMA_VERSION = "vast_provider_command_result.v1"
VAST_VIDEO_SMOKE_SCHEMA_VERSION = "vast_video_smoke_result.v1"
VAST_BLUEPRINT_BUNDLE_PREFLIGHT_SCHEMA_VERSION = "vast_blueprint_bundle_preflight.v1"
VAST_ISAAC_IMAGE_STARTUP_PREFLIGHT_SCHEMA_VERSION = "vast_isaac_image_startup_preflight.v1"
VAST_TEMPLATE_DISCOVERY_SCHEMA_VERSION = "vast_template_discovery.v1"
VAST_TEARDOWN_SCHEMA_VERSION = "vast_teardown_manifest.v1"
VAST_FINAL_VALIDATION_SCHEMA_VERSION = "vast_final_validation.v1"

VAST_API_BASE = "https://console.vast.ai/api/v0"
VAST_API_KEY_FILE_ENV = "VAST_API_KEY_FILE"
VAST_LAUNCH_LOCK_FILE_ENV = "VAST_LAUNCH_LOCK_FILE"
VAST_SESSION_BUDGET_LEDGER_FILE_ENV = "VAST_SESSION_BUDGET_LEDGER_FILE"
VAST_API_GATE_ENV = "BLUEPRINT_ALLOW_VAST_API_CALLS"
VAST_INSTANCE_LAUNCH_GATE_ENV = "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH"
VAST_FORWARD_SECRET_ENV_VARS_ENV = "BLUEPRINT_VAST_FORWARD_SECRET_ENV_VARS"
VAST_WAM_MIN_GPU_RAM_MB_ENV = "BLUEPRINT_VAST_WAM_MIN_GPU_RAM_MB"
VAST_MIN_COMPUTE_CAP_ENV = "BLUEPRINT_VAST_MIN_COMPUTE_CAP"
VAST_MIN_RELIABILITY_ENV = "BLUEPRINT_VAST_MIN_RELIABILITY"
VAST_REQUIRE_DIRECT_PORT_ENV = "BLUEPRINT_VAST_REQUIRE_DIRECT_PORT"
VAST_PREFERRED_GPU_KEYWORDS_ENV = "BLUEPRINT_VAST_PREFERRED_GPU_KEYWORDS"
VAST_PREFERRED_GEOLOCATION_REGEX_ENV = "BLUEPRINT_VAST_PREFERRED_GEOLOCATION_REGEX"
VAST_WAM_NO_PROGRESS_SECONDS_ENV = "BLUEPRINT_VAST_WAM_NO_PROGRESS_SECONDS"
VAST_HEARTBEAT_NO_PROGRESS_SECONDS_ENV = "BLUEPRINT_VAST_HEARTBEAT_NO_PROGRESS_SECONDS"
VAST_ALLOW_COMMAND_EXECUTE_SCRIPT_FALLBACK_ENV = (
    "BLUEPRINT_VAST_ALLOW_COMMAND_EXECUTE_SCRIPT_FALLBACK"
)
VAST_CONTAINER_MISSING_RETRY_ATTEMPTS_ENV = "BLUEPRINT_VAST_CONTAINER_MISSING_RETRY_ATTEMPTS"
VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS_ENV = "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"
VAST_INLINE_PROVIDER_BUNDLE_BASE64_ENV = "BLUEPRINT_VAST_PROVIDER_BUNDLE_BASE64"
VAST_INLINE_PROVIDER_BUNDLE_SHA256_ENV = "BLUEPRINT_VAST_PROVIDER_BUNDLE_SHA256"
VAST_INLINE_PROVIDER_BUNDLE_MAX_RAW_BYTES = 96_000
VAST_INLINE_PROVIDER_BUNDLE_MAX_BASE64_CHARS = 130_000
VAST_IMAGE_LOGIN_MODE_ENV = "BLUEPRINT_VAST_IMAGE_LOGIN_MODE"
DEFAULT_VAST_API_KEY_FILE = "~/.blueprint-secrets/vast_api_key"
DEFAULT_VAST_LAUNCH_LOCK_FILENAME = "vast_paid_launch.lock"
DEFAULT_VAST_SESSION_BUDGET_FILENAME = "vast_session_cost_summary.json"
DEFAULT_NGC_API_KEY_FILE = "~/.blueprint-secrets/ngc_api_key"
DOCKER_USERNAME_FILE_ENV = "DOCKER_USERNAME_FILE"
DOCKER_PAT_FILE_ENV = "DOCKER_PAT_FILE"
DEFAULT_DOCKER_USERNAME_FILE = "~/.blueprint-secrets/docker_username"
DEFAULT_DOCKER_PAT_FILE = "~/.blueprint-secrets/docker_pat"
HF_TOKEN_FILE_ENV = "HF_TOKEN_FILE"
HF_TOKEN_FILE_ENV_NAMES = (
    HF_TOKEN_FILE_ENV,
    "HUGGINGFACE_TOKEN_FILE",
    "HUGGING_FACE_HUB_TOKEN_FILE",
    "BLUEPRINT_HF_TOKEN_FILE",
)
DEFAULT_HF_TOKEN_FILE = "~/.blueprint-secrets/huggingface_token"
DEFAULT_PUBLIC_CUDA_IMAGE = "nvidia/cuda:12.4.1-runtime-ubuntu22.04"
DEFAULT_ISAAC_IMAGE = "nvcr.io/nvidia/isaac-sim:6.0.0"
DEFAULT_HEARTBEAT_URL = "https://example.com/"
DEFAULT_PUBLIC_DISK_GB = 32
DEFAULT_ISAAC_DISK_GB = 100
DEFAULT_VAST_LAUNCH_MODE = "auto"
VAST_LAUNCH_MODES = ("auto", "ssh_direct", "jupyter_direct", "args")
VAST_PROVIDER_BUNDLE_KINDS = ("isaac", "wam", "unitree_unifolm", "unitree_groot_n17_sonic")
DEFAULT_NGC_IMAGE_LOGIN_MODE = "auto"
NGC_IMAGE_LOGIN_MODES = ("auto", "always", "never")
DEFAULT_MAX_HOURLY_RATE = 0.60
DEFAULT_TARGET_SPEND_USD = 0.35
DEFAULT_HARD_CAP_USD = 0.75
DEFAULT_MAX_LIVE_MINUTES = 45
DEFAULT_SESSION_MAX_LIVE_MINUTES = 45
DEFAULT_HEARTBEAT_NO_PROGRESS_SECONDS = 600
DEFAULT_VIDEO_SMOKE_CAMERA_COUNT = 6
DEFAULT_WAM_ROLLOUT_VIDEO_COUNT = 1
DEFAULT_UNITREE_UNIFOLM_VIDEO_COUNT = 0
DEFAULT_UNITREE_GROOT_N17_SONIC_VIDEO_COUNT = 0
DEFAULT_ARGS_LOG_HOLD_SECONDS = 180
DEFAULT_MIN_COLD_ISAAC_PULL_LIVE_MINUTES = 18
VAST_LIVE_ATTEMPT_ARTIFACT_NAMES = (
    "vast_runtime_phase_log.jsonl",
    "vast_offer_selection_manifest.json",
    "vast_budget_ledger.json",
    "vast_startup_probe_manifest.json",
    "vast_gpu_sanity_report.json",
    "vast_isaac_smoke_result.json",
    "vast_provider_command_result.json",
    "vast_video_smoke_result.json",
    "vast_teardown_manifest.json",
    "vast_final_validation.json",
    "vast_provider_adapter_result.json",
    "vast_session_budget_guard.json",
    "vast_blueprint_bundle_preflight.json",
    "vast_launch_lock_manifest.json",
    "vast_prelaunch_inventory_guard.json",
    "provider_worker_endpoint_manifest.json",
    "vast_provider_runtime_output.zip",
    "vast_onstart_container.log",
)
VAST_DOC_SOURCES = [
    {
        "label": "Vast create instance API",
        "url": "https://docs.vast.ai/api-reference/instances/create-instance",
        "notes": [
            "create accepts an ask/offer id with PUT /api/v0/asks/{id}/",
            "runtype args preserves image entrypoint",
            "ssh/jupyter launch modes replace entrypoint and use onstart",
        ],
    },
    {
        "label": "Vast search offers API",
        "url": "https://docs.vast.ai/api-reference/search/search-offers",
        "notes": ["search offers uses POST /api/v0/bundles/ with filter operators"],
    },
    {
        "label": "Vast launch modes",
        "url": "https://docs.vast.ai/guides/instances/connect/overview",
        "notes": ["launch modes are entrypoint, ssh, and jupyter"],
    },
    {
        "label": "Isaac Sim 6.0 container installation",
        "url": "https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_container.html",
        "notes": [
            "Isaac container runs headless in Docker",
            "ACCEPT_EULA=Y is required",
            "larger persistent cache/disk mounts are recommended",
        ],
    },
    {
        "label": "Vast execute API",
        "url": "https://docs.vast.ai/api-reference/instances/execute",
        "notes": ["execute queues a constrained command and returns a result_url"],
    },
    {
        "label": "Vast destroy instance API",
        "url": "https://docs.vast.ai/api-reference/instances/destroy-instance",
        "notes": ["destroy deletes the instance and stops continuing spend for it"],
    },
]
VAST_REQUIRED_PHASES = (
    "vast_docs_checked",
    "vast_secret_file_verified",
    "vast_offer_search_started",
    "vast_offer_selected",
    "vast_instance_create_requested",
    "vast_instance_started_or_blocked",
    "vast_heartbeat_started",
    "vast_heartbeat_completed_or_blocked",
    "vast_gpu_sanity_started",
    "vast_gpu_sanity_completed_or_blocked",
    "vast_isaac_smoke_started",
    "vast_isaac_smoke_completed_or_blocked",
    "vast_blueprint_bundle_started",
    "vast_blueprint_bundle_completed_or_blocked",
    "vast_artifacts_exported",
    "vast_instance_teardown_started",
    "vast_instance_teardown_completed",
)
SENSITIVE_KEY_MARKERS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "LOGIN",
    "JUPYTER",
)
REDACTED_SECRET = "REDACTED_SECRET"
REDACTED_SECRET_FIELD = "REDACTED_SECRET_FIELD"
REDACTED_INLINE_PROVIDER_BUNDLE = "REDACTED_INLINE_PROVIDER_BUNDLE"
ISAAC_KNOWN_UNSUPPORTED_DRIVER_FLOOR = (570, 0, 0)
ISAAC_KNOWN_UNSUPPORTED_DRIVER_CEILING_EXCLUSIVE = (570, 158, 1)
VAST_TERMINAL_INSTANCE_STATUSES = (
    "stopped",
    "exited",
    "failed",
    "destroyed",
    "deleted",
    "inactive",
    "completed",
)
logger = logging.getLogger(__name__)


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _provider_expected_video_count(provider_bundle_kind: str) -> int:
    if provider_bundle_kind == "isaac":
        return DEFAULT_VIDEO_SMOKE_CAMERA_COUNT
    if provider_bundle_kind == "wam":
        return DEFAULT_WAM_ROLLOUT_VIDEO_COUNT
    if provider_bundle_kind == "unitree_unifolm":
        return DEFAULT_UNITREE_UNIFOLM_VIDEO_COUNT
    if provider_bundle_kind == "unitree_groot_n17_sonic":
        return DEFAULT_UNITREE_GROOT_N17_SONIC_VIDEO_COUNT
    return 0


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _normalized_binary_capability(value: Any) -> bool | None:
    """Normalize provider 0/1 capability fields without treating unknown as false."""
    if isinstance(value, bool):
        return value
    number = _number(value)
    if number == 1:
        return True
    if number == 0:
        return False
    text = _string(value).strip().lower()
    if text in {"true", "yes"}:
        return True
    if text in {"false", "no"}:
        return False
    return None


def _content_range_total_bytes(value: Any) -> int | None:
    text = _string(value)
    if "/" not in text:
        return None
    total = text.rsplit("/", 1)[-1].strip()
    if not total or total == "*":
        return None
    try:
        parsed = int(total)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _version_tuple(value: Any) -> tuple[int, int, int] | None:
    text = _string(value)
    if not text:
        return None
    parts = re.findall(r"\d+", text)
    if not parts:
        return None
    numbers = [int(item) for item in parts[:3]]
    while len(numbers) < 3:
        numbers.append(0)
    return numbers[0], numbers[1], numbers[2]


def _driver_version(offer: Mapping[str, Any]) -> str:
    for key in (
        "driver_version",
        "driverVersion",
        "cuda_driver_version",
        "nvidia_driver_version",
        "driver",
    ):
        value = _string(offer.get(key))
        if value:
            return value
    return ""


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [item for item in (_string(item) for item in value) if item]
    return []


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_string(os.getenv(name)) or default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_string(os.getenv(name)) or default)
    except ValueError:
        return default


def _env_csv(name: str, default: Sequence[str] = ()) -> list[str]:
    text = _string(os.getenv(name))
    if not text:
        return list(default)
    return [item.strip() for item in text.split(",") if item.strip()]


def _write_jsonl_row(path: Path, row: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _append_phase(
    job_dir: Path,
    phase: str,
    status: str,
    *,
    blockers: Sequence[str] | None = None,
    proof_effect: str = "none",
    **extra: Any,
) -> None:
    _write_jsonl_row(
        job_dir / "vast_runtime_phase_log.jsonl",
        {
            "schema_version": VAST_PHASE_LOG_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "phase": phase,
            "status": status,
            "blockers": list(blockers or []),
            "proof_effect": proof_effect,
            **extra,
        },
    )


def _secret_file_status(env_var: str, default_path: str) -> dict[str, Any]:
    explicit = _string(os.getenv(env_var))
    selected = explicit or default_path
    path = Path(selected).expanduser()
    mode = None
    if path.exists():
        mode = oct(path.stat().st_mode & 0o777)
    return {
        "env_var": env_var,
        "path": str(path),
        "path_source": "env" if explicit else "default_blueprint_secret_file_path",
        "present": path.is_file(),
        "mode": mode,
        "mode_is_0600": mode == "0o600",
        "raw_secret_value_recorded": False,
    }


def _read_secret_file(env_var: str, default_path: str) -> tuple[str, dict[str, Any]]:
    status = _secret_file_status(env_var, default_path)
    if not status["present"]:
        path = Path(status["path"])
        if path.exists():
            try:
                path.read_text(encoding="utf-8")
            except OSError as exc:
                status.update({"read_error": type(exc).__name__})
        return "", status
    try:
        key = Path(status["path"]).read_text(encoding="utf-8").strip()
    except OSError as exc:
        status.update({"read_error": type(exc).__name__, "present": False})
        return "", status
    status["secret_nonempty"] = bool(key)
    return key, status


def _read_hf_token_file() -> tuple[str, dict[str, Any]]:
    selected_env = HF_TOKEN_FILE_ENV
    for env_name in HF_TOKEN_FILE_ENV_NAMES:
        if _string(os.getenv(env_name)):
            selected_env = env_name
            break
    token, status = _read_secret_file(selected_env, DEFAULT_HF_TOKEN_FILE)
    status["accepted_env_vars"] = list(HF_TOKEN_FILE_ENV_NAMES)
    return token, status


def _hf_token_secret_values() -> list[str]:
    token, _status = _read_hf_token_file()
    return [token] if token else []


def _redact_text(text: str, secret_values: Sequence[str]) -> str:
    redacted = text
    for value in sorted((item for item in secret_values if item), key=len, reverse=True):
        redacted = redacted.replace(value, REDACTED_SECRET)
    return _redact_signed_url_queries(redacted)


def _redact_signed_url_queries(text: str) -> str:
    """Collapse signed URL query strings so credential-bearing keys do not persist."""
    if not text or not re.search(r"(?:X-Amz-|x-amz-|X-Goog-|x-goog-|token=|signature=)", text):
        return text

    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        parsed = urlparse(value)
        if not parsed.scheme or not parsed.netloc:
            return REDACTED_SECRET
        query = "REDACTED_QUERY" if parsed.query else ""
        fragment = "REDACTED_FRAGMENT" if parsed.fragment else ""
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", query, fragment))

    return re.sub(r"https?://[^\s'\"<>]+", replace, text)


def _redact_result_url(value: str) -> str:
    text = _string(value)
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc and (parsed.query or parsed.fragment):
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                "",
                "REDACTED_QUERY" if parsed.query else "",
                "REDACTED_FRAGMENT" if parsed.fragment else "",
            )
        )
    return _redact_signed_url_queries(text)


def _url_secret_values(*urls: str | None) -> list[str]:
    secret_values: list[str] = []
    for url in urls:
        text = _string(url)
        if not text:
            continue
        parsed = urlparse(text)
        query = parse_qs(parsed.query)
        for key, values in query.items():
            key_lower = key.lower()
            if not (
                "token" in key_lower
                or "signature" in key_lower
                or "credential" in key_lower
                or "access" in key_lower
            ):
                continue
            for value in values:
                if value:
                    secret_values.append(value)
                    secret_values.append(quote(value, safe=""))
    return _dedupe(secret_values)


def _inline_provider_bundle_payload(
    bundle_path: Path | None,
    *,
    provider_bundle_kind: str,
    enable_blueprint_bundle: bool,
    max_raw_bytes: int = VAST_INLINE_PROVIDER_BUNDLE_MAX_RAW_BYTES,
    max_base64_chars: int = VAST_INLINE_PROVIDER_BUNDLE_MAX_BASE64_CHARS,
) -> dict[str, Any]:
    """Return an artifact-safe inline transport descriptor for small provider bundles.

    The raw base64 payload is returned for request construction only. Callers
    must redact it before writing artifacts.
    """

    if not enable_blueprint_bundle:
        return {
            "inline_provider_bundle_transport_used": False,
            "inline_provider_bundle_transport_reason": "blueprint_bundle_disabled",
        }
    if provider_bundle_kind not in {"wam", "unitree_unifolm", "unitree_groot_n17_sonic"}:
        return {
            "inline_provider_bundle_transport_used": False,
            "inline_provider_bundle_transport_reason": "inline_transport_provider_kind_not_supported",
            "inline_provider_bundle_supported_kinds": [
                "wam",
                "unitree_unifolm",
                "unitree_groot_n17_sonic",
            ],
        }
    if not bundle_path or not bundle_path.is_file():
        return {
            "inline_provider_bundle_transport_used": False,
            "inline_provider_bundle_transport_reason": "provider_bundle_missing",
        }
    raw = bundle_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    if len(raw) > max_raw_bytes or len(encoded) > max_base64_chars:
        return {
            "inline_provider_bundle_transport_used": False,
            "inline_provider_bundle_transport_reason": "provider_bundle_too_large_for_inline_env",
            "inline_provider_bundle_size_bytes": len(raw),
            "inline_provider_bundle_base64_length": len(encoded),
            "inline_provider_bundle_max_raw_bytes": max_raw_bytes,
            "inline_provider_bundle_max_base64_chars": max_base64_chars,
        }
    return {
        "inline_provider_bundle_transport_used": True,
        "inline_provider_bundle_transport_reason": f"small_{provider_bundle_kind}_bundle_inline_env",
        "inline_provider_bundle_size_bytes": len(raw),
        "inline_provider_bundle_base64_length": len(encoded),
        "inline_provider_bundle_sha256_present": True,
        "inline_provider_bundle_base64": encoded,
        "inline_provider_bundle_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _redact_runtime_value(value: Any, secret_values: Sequence[str] = ()) -> Any:
    if isinstance(value, str):
        return _redact_text(value, secret_values)
    if isinstance(value, list):
        if (
            len(value) == 2
            and isinstance(value[0], str)
            and value[0].upper() == value[0]
            and any(marker in value[0].upper() for marker in SENSITIVE_KEY_MARKERS)
        ):
            return [value[0], REDACTED_SECRET_FIELD]
        return [_redact_runtime_value(item, secret_values) for item in value]
    if isinstance(value, tuple):
        if (
            len(value) == 2
            and isinstance(value[0], str)
            and value[0].upper() == value[0]
            and any(marker in value[0].upper() for marker in SENSITIVE_KEY_MARKERS)
        ):
            return [value[0], REDACTED_SECRET_FIELD]
        return [_redact_runtime_value(item, secret_values) for item in value]
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text == VAST_INLINE_PROVIDER_BUNDLE_BASE64_ENV and item:
                result[key_text] = REDACTED_INLINE_PROVIDER_BUNDLE
            elif (
                isinstance(item, str)
                and item
                and any(marker in key_text.upper() for marker in SENSITIVE_KEY_MARKERS)
            ):
                result[key_text] = REDACTED_SECRET_FIELD
            else:
                result[key_text] = _redact_runtime_value(item, secret_values)
        return result
    return value


def _provider_url_public_blocker(url: str | None, role: str) -> str | None:
    text = _string(url)
    if not text:
        return None
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        return f"{role}_url_scheme_not_http"
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return f"{role}_url_host_missing"
    if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local"):
        return f"{role}_url_not_publicly_reachable"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_unspecified or ip.is_multicast:
        return f"{role}_url_not_publicly_reachable"
    return None


def _resolve_public_dns_a_records(host: str, *, timeout_seconds: int = 10) -> list[str]:
    dig_path = shutil.which("dig")
    if not dig_path:
        return []
    try:
        completed = subprocess.run(
            [dig_path, "+short", host, "@1.1.1.1"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception:
        return []
    ips: list[str] = []
    for line in completed.stdout.splitlines():
        candidate = line.strip()
        try:
            ip = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if ip.version != 4:
            continue
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_unspecified
            or ip.is_multicast
        ):
            continue
        ips.append(candidate)
    return _dedupe(ips)


def _read_http_headers_from_socket(sock: socket.socket) -> tuple[int | None, dict[str, str]]:
    chunks: list[bytes] = []
    deadline = time.time() + 20
    while time.time() < deadline:
        data = sock.recv(4096)
        if not data:
            break
        chunks.append(data)
        if b"\r\n\r\n" in b"".join(chunks):
            break
    header_text = b"".join(chunks).split(b"\r\n\r\n", 1)[0].decode("iso-8859-1", errors="replace")
    lines = header_text.splitlines()
    status_code: int | None = None
    if lines:
        parts = lines[0].split()
        if len(parts) >= 2:
            try:
                status_code = int(parts[1])
            except ValueError:
                status_code = None
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip()] = value.strip()
    return status_code, headers


def _head_with_public_dns_fallback(
    url: str,
    *,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").strip()
    if not host or parsed.scheme not in {"http", "https"}:
        return {
            "status": "blocked",
            "method": "HEAD_WITH_PUBLIC_DNS_FALLBACK",
            "blockers": ["provider_bundle_fetch_url_public_dns_fallback_unsupported_url"],
        }
    deadline = time.time() + max(1, timeout_seconds)
    ips: list[str] = []
    dns_attempt_count = 0
    while time.time() < deadline and not ips:
        dns_attempt_count += 1
        ips = _resolve_public_dns_a_records(host, timeout_seconds=min(10, timeout_seconds))
        if not ips:
            time.sleep(3)
    errors: list[dict[str, str]] = []
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    for ip in ips:
        try:
            remaining = max(1, int(deadline - time.time()))
            raw_sock = socket.create_connection((ip, port), timeout=min(20, remaining))
            try:
                if parsed.scheme == "https":
                    context = ssl.create_default_context()
                    sock = context.wrap_socket(raw_sock, server_hostname=host)
                else:
                    sock = raw_sock
                try:
                    sock.settimeout(min(20, max(1, int(deadline - time.time()))))
                    request = (
                        f"HEAD {path} HTTP/1.1\r\n"
                        f"Host: {host}\r\n"
                        "User-Agent: BlueprintVastStagingProbe/1.0\r\n"
                        "Connection: close\r\n"
                        "\r\n"
                    )
                    sock.sendall(request.encode("ascii"))
                    status_code, headers = _read_http_headers_from_socket(sock)
                finally:
                    sock.close()
            except Exception:
                raw_sock.close()
                raise
            content_length = _number(headers.get("Content-Length"))
            return {
                "status": "passed"
                if status_code is not None and 200 <= status_code < 300
                else "blocked",
                "method": "HEAD_WITH_PUBLIC_DNS_FALLBACK",
                "http_status_code": status_code,
                "content_type": headers.get("Content-Type"),
                "content_length": int(content_length) if content_length is not None else None,
                "public_dns_resolver": "dig @1.1.1.1",
                "public_dns_attempt_count": dns_attempt_count,
                "resolved_ip_count": len(ips),
                "resolved_ip_used": ip,
            }
        except Exception as exc:
            errors.append({"ip": ip, "error_type": type(exc).__name__})
    return {
        "status": "blocked",
        "method": "HEAD_WITH_PUBLIC_DNS_FALLBACK",
        "public_dns_resolver": "dig @1.1.1.1",
        "public_dns_attempt_count": dns_attempt_count,
        "resolved_ip_count": len(ips),
        "connection_errors": errors[-5:],
        "blockers": ["provider_bundle_fetch_url_public_dns_fallback_failed"],
    }


def _api_json(
    *,
    method: str,
    path: str,
    api_key: str,
    payload: Mapping[str, Any] | None = None,
    timeout_seconds: int = 30,
) -> tuple[int, dict[str, Any]]:
    url = (
        path
        if path.startswith("http://") or path.startswith("https://")
        else f"{VAST_API_BASE}{path}"
    )
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status_code = int(getattr(response, "status", 200))
        response_text = response.read().decode("utf-8", errors="replace")
    if not response_text.strip():
        return status_code, {}
    parsed = json.loads(response_text)
    return status_code, dict(parsed) if isinstance(parsed, Mapping) else {"response": parsed}


def _fetch_text(url: str, timeout_seconds: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8", errors="replace")


def _vast_template_summary(template: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "id": template.get("id"),
        "hash_id": template.get("hash_id"),
        "name": template.get("name"),
        "image": template.get("image"),
        "tag": template.get("tag"),
        "default_tag": template.get("default_tag"),
        "recommended": template.get("recommended"),
        "recommended_disk_space": template.get("recommended_disk_space"),
        "count_created": template.get("count_created"),
        "recent_create_date": template.get("recent_create_date"),
        "docker_login_repo": template.get("docker_login_repo"),
        "ssh_direct": template.get("ssh_direct"),
        "jup_direct": template.get("jup_direct"),
        "use_ssh": template.get("use_ssh"),
    }
    haystack = " ".join(str(value or "") for value in fields.values()).lower()
    isaac_terms = ("isaac", "omniverse", "nvidia/isaac", "isaac-sim", "simulationapp")
    gpu_render_terms = ("rtx", "render", "opengl", "vulkan", "egl")
    fields["isaac_template_candidate"] = any(term in haystack for term in isaac_terms)
    fields["rendering_template_candidate"] = any(term in haystack for term in gpu_render_terms)
    fields["requires_registry_login"] = bool(_string(fields.get("docker_login_repo")))
    return fields


def _discover_vast_templates(
    *,
    job_dir: Path,
    generated_at: str,
    api_key: str,
    max_templates_inspected: int = 5000,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    blockers: list[str] = []
    query = {
        "select_filters": json.dumps({}),
        "select_cols": json.dumps(
            [
                "id",
                "hash_id",
                "name",
                "image",
                "tag",
                "default_tag",
                "recommended",
                "recommended_disk_space",
                "count_created",
                "recent_create_date",
                "docker_login_repo",
                "ssh_direct",
                "jup_direct",
                "use_ssh",
            ]
        ),
    }
    url = f"{VAST_API_BASE}/template/?{urlencode(query)}"
    status_code = 0
    payload: dict[str, Any] = {}
    templates: list[dict[str, Any]] = []
    try:
        status_code, payload = _api_json(
            method="GET",
            path=url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        raw_templates = payload.get("templates") if isinstance(payload, Mapping) else []
        if isinstance(raw_templates, list):
            templates = [dict(item) for item in raw_templates if isinstance(item, Mapping)]
        else:
            blockers.append("vast_template_response_missing_templates_list")
    except urllib.error.HTTPError as exc:
        blockers.append(f"vast_template_search_http_error:{exc.code}")
    except Exception as exc:
        blockers.append(f"vast_template_search_failed:{type(exc).__name__}")
    templates.sort(
        key=lambda item: _number(item.get("count_created")) or 0.0,
        reverse=True,
    )
    inspected = templates[: max(0, max_templates_inspected)]
    summaries = [_vast_template_summary(template) for template in inspected]
    isaac_candidates = [item for item in summaries if item.get("isaac_template_candidate")]
    rendering_candidates = [
        item
        for item in summaries
        if item.get("isaac_template_candidate") or item.get("rendering_template_candidate")
    ]
    manifest = {
        "schema_version": VAST_TEMPLATE_DISCOVERY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if not blockers else "blocked",
        "job_dir": str(job_dir),
        "docs_source": "https://docs.vast.ai/api-reference/search/search-templates",
        "api_endpoint": "GET /api/v0/template/",
        "http_status_code": status_code or None,
        "templates_found_reported": payload.get("templates_found")
        if isinstance(payload, Mapping)
        else None,
        "templates_returned": len(templates),
        "templates_inspected": len(inspected),
        "max_templates_inspected": max_templates_inspected,
        "isaac_candidate_count": len(isaac_candidates),
        "rendering_or_isaac_candidate_count": len(rendering_candidates),
        "isaac_template_candidates": isaac_candidates[:25],
        "rendering_or_isaac_template_candidates": rendering_candidates[:25],
        "candidate_selection_status": "candidate_found"
        if isaac_candidates
        else "no_isaac_template_candidate_found",
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "vast_template_discovery.json", manifest)
    return manifest


def _vastai_version() -> dict[str, Any]:
    executable = shutil.which("vastai")
    if not executable:
        return {"present": False, "path": None}
    try:
        completed = subprocess.run(
            [executable, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception as exc:  # pragma: no cover - environment dependent.
        return {"present": True, "path": executable, "probe_error": type(exc).__name__}
    output = (completed.stdout or completed.stderr or "").strip()
    return {
        "present": True,
        "path": executable,
        "returncode": completed.returncode,
        "first_line": output.splitlines()[0] if output else None,
    }


def _runtime_discovery(
    job_dir: Path,
    *,
    generated_at: str,
    launch_mode: str,
    disk_gb: int,
) -> dict[str, Any]:
    discovery = {
        "schema_version": VAST_RUNTIME_DISCOVERY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_dir": str(job_dir),
        "vast_docs_checked": True,
        "docs_checked_at": generated_at,
        "docs_sources": VAST_DOC_SOURCES,
        "api_surfaces": {
            "search_offers": "POST /api/v0/bundles/",
            "create_instance": "PUT /api/v0/asks/{id}/",
            "show_instance": "GET /api/v0/instances/{id}/",
            "execute_command": "PUT /api/v0/instances/command/{id}/",
            "show_logs": "PUT /api/v0/instances/request_logs/{id}",
            "destroy_instance": "DELETE /api/v0/instances/{id}/",
        },
        "launch_mode_notes": {
            "entrypoint_args": "preserves image entrypoint and does not inject SSH/Jupyter",
            "ssh_jupyter": "replaces image entrypoint; use onstart for startup commands",
            "launch_mode_used_for_probe": launch_mode,
        },
        "disk_gb_requested_for_probe": disk_gb,
        "vastai_cli_probe": _vastai_version(),
        "proof_boundary": (
            "Runtime discovery only records current Vast API/CLI surfaces. It does not "
            "prove provider allocation, GPU visibility, Isaac execution, or generated-world rank fidelity."
        ),
    }
    write_json(job_dir / "vast_runtime_discovery.json", discovery)
    return discovery


def _provider_plan(
    *,
    job_dir: Path,
    generated_at: str,
    max_hourly_rate: float,
    target_spend_usd: float,
    hard_cap_usd: float,
    max_live_minutes: int,
    public_image: str,
    isaac_image: str,
    selected_container_image: str,
    previous_job_dir: Path | None,
    provider_bundle: Path | None,
    provider_bundle_kind: str,
    enable_isaac_smoke: bool,
    enable_blueprint_bundle: bool,
    launch_mode: str,
    disk_gb: int,
    ngc_image_login_mode: str,
    vast_template_hash_id: str | None,
    use_vast_template_image: bool,
    allow_cold_isaac_image_pull: bool,
    min_cold_isaac_pull_live_minutes: int,
    provider_bundle_url: str | None = None,
    provider_output_put_url: str | None = None,
    provider_bundle_inline_transport: Mapping[str, Any] | None = None,
    require_known_supported_isaac_driver: bool = False,
) -> dict[str, Any]:
    inline_transport = _mapping(provider_bundle_inline_transport)
    plan = {
        "schema_version": VAST_PROVIDER_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "ready_for_explicit_vast_probe",
        "job_dir": str(job_dir),
        "previous_staged_job_dir": str(previous_job_dir) if previous_job_dir else None,
        "provider_bundle_path": str(provider_bundle) if provider_bundle else None,
        "provider_bundle_kind": provider_bundle_kind,
        "provider_bundle_present": bool(provider_bundle and provider_bundle.is_file()),
        "budget": {
            "target_spend_usd": target_spend_usd,
            "hard_cap_usd": hard_cap_usd,
            "max_hourly_rate_usd": max_hourly_rate,
            "max_live_minutes": max_live_minutes,
        },
        "launch_strategy": {
            "provider": "vast",
            "heartbeat_image": public_image,
            "heartbeat_launch_mode": launch_mode,
            "gpu_sanity_launch_mode": "same_instance_log_stream",
            "isaac_image": isaac_image,
            "selected_container_image": selected_container_image,
            "isaac_smoke_enabled": enable_isaac_smoke,
            "isaac_launch_mode": launch_mode if enable_isaac_smoke else None,
            "disk_gb": disk_gb,
            "ngc_image_login_mode": ngc_image_login_mode,
            "vast_template_hash_present": bool(_string(vast_template_hash_id)),
            "use_vast_template_image": use_vast_template_image,
            "allow_cold_isaac_image_pull": allow_cold_isaac_image_pull,
            "min_cold_isaac_pull_live_minutes": min_cold_isaac_pull_live_minutes,
            "isaac_required_env": {
                "ACCEPT_EULA": "present" if enable_isaac_smoke else "not_required_for_public_probe",
                "PRIVACY_CONSENT": "present"
                if enable_isaac_smoke
                else "not_required_for_public_probe",
                "NVIDIA_DRIVER_CAPABILITIES": "present"
                if enable_isaac_smoke
                else "not_required_for_public_probe",
            },
            "blueprint_bundle_enabled": enable_blueprint_bundle,
            "blueprint_bundle_kind": provider_bundle_kind,
            "blueprint_bundle_fetch_url_present": bool(_string(provider_bundle_url)),
            "blueprint_bundle_output_put_url_present": bool(_string(provider_output_put_url)),
            "blueprint_bundle_inline_transport_used": (
                inline_transport.get("inline_provider_bundle_transport_used") is True
            ),
            "blueprint_bundle_inline_transport_reason": _string(
                inline_transport.get("inline_provider_bundle_transport_reason")
            ),
            "blueprint_bundle_inline_transport_size_bytes": int(
                _number(inline_transport.get("inline_provider_bundle_size_bytes")) or 0
            ),
            "blueprint_bundle_inline_transport_base64_length": int(
                _number(inline_transport.get("inline_provider_bundle_base64_length")) or 0
            ),
            "blueprint_bundle_inline_transport_sha256_present": (
                inline_transport.get("inline_provider_bundle_sha256_present") is True
            ),
            "require_known_supported_isaac_driver": require_known_supported_isaac_driver,
        },
        "truth_boundaries": _truth_boundaries(),
        "raw_secret_values_recorded": False,
        "proof_boundary": (
            "This plan authorizes a bounded Vast startup probe only. Heartbeat, GPU "
            "sanity, and Isaac smoke do not prove generated-world rank fidelity, deployment "
            "readiness, official G1 policy execution, or real WAM/VLA execution."
        ),
    }
    write_json(job_dir / "vast_provider_plan.json", plan)
    return plan


def _truth_boundaries() -> dict[str, Any]:
    return {
        "isaac_sim_does_not_make_spz_or_3dgs_physical": True,
        "direct_splat_collision_proven": False,
        "collider_source_if_used": "metadata_derived_collider_proxy_required",
        "splat_visuals_if_used": "splat_rendered_visual_evidence_synchronized_with_isaac_state",
        "real_wam_vla_runtime_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "dexterous_hand_policy_proven": False,
        "controller_grade_execution_proven": False,
        "official_policy_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
    }


def _search_payload(*, limit: int, max_hourly_rate: float | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "limit": limit,
        "type": "on-demand",
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "num_gpus": {"eq": 1},
    }
    if max_hourly_rate is not None:
        payload["dph_total"] = {"lte": max_hourly_rate}
    return payload


def _offers_from_response(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    offers = response.get("offers")
    if isinstance(offers, list):
        return [dict(item) for item in offers if isinstance(item, Mapping)]
    if isinstance(offers, Mapping):
        if "id" in offers or "ask_contract_id" in offers:
            return [dict(offers)]
        return [dict(item) for item in offers.values() if isinstance(item, Mapping)]
    response_value = response.get("response")
    if isinstance(response_value, list):
        return [dict(item) for item in response_value if isinstance(item, Mapping)]
    return []


def _offer_id(offer: Mapping[str, Any]) -> int | None:
    for key in ("ask_contract_id", "id"):
        value = _number(offer.get(key))
        if value:
            return int(value)
    return None


def _offer_hourly_rate(offer: Mapping[str, Any]) -> float | None:
    direct_keys = (
        "hourly_rate_usd",
        "dph_total",
        "discounted_dph_total",
        "totalHour",
        "discountedTotalPerHour",
        "min_bid",
    )
    for key in direct_keys:
        value = _number(offer.get(key))
        if value is not None:
            return value
    pricing = _mapping(offer.get("pricing"))
    for section_name in ("instance", "machine"):
        section = _mapping(pricing.get(section_name))
        for key in ("discountedTotalPerHour", "totalHour", "discountTotalHour"):
            value = _number(section.get(key))
            if value is not None:
                return value
    return None


def _gpu_name(offer: Mapping[str, Any]) -> str:
    return (
        _string(offer.get("gpu_name"))
        or _string(offer.get("gpu_name_str"))
        or _string(offer.get("gpu_display_name"))
        or _string(offer.get("gpu_model_slug")).replace("_", " ")
    )


def _normalized_gpu_model_key(value: Any) -> str:
    """Normalize provider/vendor spelling without weakening model identity."""

    text = _string(value).upper()
    for vendor_token in ("NVIDIA", "GEFORCE"):
        text = re.sub(rf"\b{vendor_token}\b", "", text)
    return re.sub(r"[^A-Z0-9]+", "", text)


def _known_gpu_vram_cap_mb(gpu_name: str) -> int | None:
    """Return the measured/model VRAM cap for an exact normalized GPU model.

    Vast can report host-total ``gpu_ram`` on a one-GPU slice.  The model cap
    prevents a 24 GB RTX 4090 on a multi-GPU host from masquerading as a 48 GB
    card while preserving provider-reported memory for models not yet in the
    authoritative hardware registry.
    """

    model_key = _normalized_gpu_model_key(gpu_name)
    if not model_key:
        return None
    for known_name, vram_gb in KNOWN_GPU_VRAM_GB.items():
        if _normalized_gpu_model_key(known_name) == model_key:
            return int(round(float(vram_gb) * 1024))
    return None


def _offer_summary(offer: Mapping[str, Any]) -> dict[str, Any]:
    gpu = _gpu_name(offer)
    driver = _driver_version(offer)
    driver_status = _isaac_driver_support_status(driver)
    compute_cap = offer.get("compute_cap")
    provider_reported_gpu_ram_mb = _number(
        offer.get("gpu_ram") or offer.get("gpu_totalram") or offer.get("gpu_ram_mb")
    )
    known_model_vram_cap_mb = _known_gpu_vram_cap_mb(gpu)
    effective_gpu_ram_mb = (
        min(int(provider_reported_gpu_ram_mb), known_model_vram_cap_mb)
        if provider_reported_gpu_ram_mb is not None and known_model_vram_cap_mb is not None
        else (
            int(provider_reported_gpu_ram_mb)
            if provider_reported_gpu_ram_mb is not None
            else known_model_vram_cap_mb
        )
    )
    return {
        "ask_contract_id": _offer_id(offer),
        "gpu_name": gpu,
        "hourly_rate_usd": _offer_hourly_rate(offer),
        "driver_version": driver or None,
        "isaac_driver_support_status": driver_status,
        "isaac_driver_preferred_for_rtx": driver_status
        == "outside_known_unsupported_omniverse_rtx_driver_range",
        "cuda_max_good": offer.get("cuda_max_good"),
        "compute_cap": compute_cap,
        "compute_cap_normalized": vcc.normalized_compute_cap(compute_cap),
        "gpu_ram_mb": effective_gpu_ram_mb,
        "provider_reported_gpu_ram_mb": (
            int(provider_reported_gpu_ram_mb) if provider_reported_gpu_ram_mb is not None else None
        ),
        "known_model_vram_cap_mb": known_model_vram_cap_mb,
        "gpu_ram_normalization": (
            "known_model_cap_applied"
            if known_model_vram_cap_mb is not None
            and provider_reported_gpu_ram_mb is not None
            and int(provider_reported_gpu_ram_mb) > known_model_vram_cap_mb
            else (
                "known_model_cap_not_needed"
                if known_model_vram_cap_mb is not None
                else "provider_reported_model_not_registered"
            )
        ),
        "num_gpus": offer.get("num_gpus"),
        "reliability": offer.get("reliability"),
        "verified": offer.get("verified"),
        "rentable": offer.get("rentable"),
        "direct_port_count": offer.get("direct_port_count"),
        "geolocation": offer.get("geolocation"),
        "machine_id": offer.get("machine_id"),
        "has_avx": _normalized_binary_capability(offer.get("has_avx")),
        "isaac_rt_candidate": _is_isaac_rt_candidate(gpu),
        "disallowed_for_isaac_rendering": _is_disallowed_for_isaac(gpu),
    }


def _safe_slug(value: Any) -> str | None:
    text = _string(value)
    if not text:
        return None
    slug = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return slug or None


def _offer_artifact_summary(offer: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not offer:
        return None
    gpu = _gpu_name(offer) or _string(offer.get("gpu_model_slug"))
    hourly = _number(offer.get("hourly_rate_usd"))
    ask_contract_id = _number(offer.get("ask_contract_id"))
    machine_id = _number(offer.get("machine_id"))
    return {
        "ask_contract_id": int(ask_contract_id) if ask_contract_id is not None else None,
        "gpu_model_slug": _safe_slug(gpu),
        "hourly_rate_usd": hourly,
        "driver_version": _string(offer.get("driver_version")) or None,
        "isaac_driver_support_status": _string(offer.get("isaac_driver_support_status")) or None,
        "cuda_max_good": offer.get("cuda_max_good"),
        "compute_cap": offer.get("compute_cap"),
        "compute_cap_normalized": offer.get("compute_cap_normalized"),
        "gpu_ram_mb": offer.get("gpu_ram_mb"),
        "provider_reported_gpu_ram_mb": offer.get("provider_reported_gpu_ram_mb"),
        "known_model_vram_cap_mb": offer.get("known_model_vram_cap_mb"),
        "gpu_ram_normalization": offer.get("gpu_ram_normalization"),
        "num_gpus": offer.get("num_gpus"),
        "reliability": offer.get("reliability"),
        "verified": offer.get("verified"),
        "rentable": offer.get("rentable"),
        "direct_port_count": offer.get("direct_port_count"),
        "geolocation": _safe_slug(offer.get("geolocation")),
        "machine_id": int(machine_id) if machine_id is not None else None,
        "has_avx": _normalized_binary_capability(offer.get("has_avx")),
        "isaac_rt_candidate": bool(offer.get("isaac_rt_candidate")),
        "disallowed_for_isaac_rendering": bool(offer.get("disallowed_for_isaac_rendering")),
        "raw_secret_values_recorded": False,
    }


def _keyword_match_rank(value: Any, keywords: Sequence[str]) -> int:
    if not keywords:
        return 0
    haystack = _string(value).lower()
    return 0 if any(_string(keyword).lower() in haystack for keyword in keywords) else 1


def _regex_match_rank(value: Any, pattern: str) -> int:
    text = _string(value)
    regex = _string(pattern)
    if not regex:
        return 0
    try:
        return 0 if re.search(regex, text, flags=re.IGNORECASE) else 1
    except re.error:
        return 1


def _machine_id_set(values: Iterable[Any]) -> set[int]:
    result: set[int] = set()
    for value in values:
        number = _number(value)
        if number is not None:
            result.add(int(number))
    return result


def _load_machine_avoidlist(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": "vast_machine_avoidlist.v1",
            "status": "empty",
            "machine_ids": [],
            "entries": [],
            "raw_secret_values_recorded": False,
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "schema_version": "vast_machine_avoidlist.v1",
            "status": "blocked_parse_failed",
            "machine_ids": [],
            "entries": [],
            "parse_error": f"{type(exc).__name__}:{str(exc)[:200]}",
            "raw_secret_values_recorded": False,
        }
    return (
        dict(data)
        if isinstance(data, Mapping)
        else {
            "schema_version": "vast_machine_avoidlist.v1",
            "status": "blocked_invalid_shape",
            "machine_ids": [],
            "entries": [],
            "raw_secret_values_recorded": False,
        }
    )


def _avoidlist_machine_ids(path: Path) -> set[int]:
    data = _load_machine_avoidlist(path)
    ids = _machine_id_set(data.get("machine_ids") or [])
    for entry in data.get("entries") or []:
        if isinstance(entry, Mapping):
            ids.update(_machine_id_set([entry.get("machine_id")]))
    return ids


def _record_machine_avoidlist_entry(
    *,
    path: Path,
    generated_at: str,
    selected_offer: Mapping[str, Any] | None,
    instance_id: int | None,
    blockers: Sequence[str],
    reason: str,
) -> dict[str, Any]:
    machine_id = _number((selected_offer or {}).get("machine_id"))
    data = _load_machine_avoidlist(path)
    entries = [entry for entry in data.get("entries") or [] if isinstance(entry, Mapping)]
    if machine_id is not None:
        entry = {
            "generated_at": generated_at,
            "machine_id": int(machine_id),
            "instance_id": instance_id,
            "offer_id": (selected_offer or {}).get("ask_contract_id"),
            "gpu_model_slug": _safe_slug((selected_offer or {}).get("gpu_name")),
            "driver_version": (selected_offer or {}).get("driver_version"),
            "reason": reason,
            "blockers": list(blockers),
            "retry_policy": "exclude_for_current_job_until_manual_review",
        }
        entries.append(entry)
    machine_ids = sorted(_avoidlist_machine_ids(path) | _machine_id_set([machine_id]))
    payload = {
        "schema_version": "vast_machine_avoidlist.v1",
        "generated_at": generated_at,
        "status": "completed",
        "machine_ids": machine_ids,
        "entries": entries,
        "raw_secret_values_recorded": False,
    }
    write_json(path, payload)
    return payload


def _attempt_preservation_slug(generated_at: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "", generated_at)
    return slug[:32] or str(int(time.time()))


def _preserve_existing_live_attempt_artifacts(
    *,
    job_dir: Path,
    generated_at: str,
    reason: str,
    artifact_names: Sequence[str] = VAST_LIVE_ATTEMPT_ARTIFACT_NAMES,
) -> dict[str, Any] | None:
    present_paths = [job_dir / name for name in artifact_names if (job_dir / name).is_file()]
    if not present_paths:
        return None
    preserve_dir = job_dir / f"attempt_preserved_{_attempt_preservation_slug(generated_at)}"
    suffix = 1
    while preserve_dir.exists():
        suffix += 1
        preserve_dir = (
            job_dir / f"attempt_preserved_{_attempt_preservation_slug(generated_at)}_{suffix}"
        )
    ensure_dir(preserve_dir)
    copied: list[str] = []
    copy_errors: list[dict[str, Any]] = []
    for source in present_paths:
        target = preserve_dir / source.name
        try:
            shutil.copy2(source, target)
            copied.append(source.name)
        except Exception as exc:
            copy_errors.append(
                {
                    "artifact": source.name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:300],
                }
            )
    manifest = {
        "schema_version": "vast_live_attempt_preservation_manifest.v1",
        "generated_at": generated_at,
        "status": "completed" if not copy_errors else "blocked_copy_errors",
        "reason": reason,
        "source_job_dir": str(job_dir),
        "preserve_dir": str(preserve_dir),
        "copied_artifacts": copied,
        "copy_errors": copy_errors,
        "artifact_count": len(copied),
        "raw_secret_values_recorded": False,
    }
    write_json(preserve_dir / "vast_attempt_preservation_manifest.json", manifest)
    write_json(job_dir / "vast_latest_attempt_preservation_manifest.json", manifest)
    return manifest


def _select_offer(
    offers: Sequence[Mapping[str, Any]],
    *,
    max_hourly_rate: float,
    min_gpu_ram_mb: int = 0,
    min_compute_cap: int = 0,
    max_compute_cap: int = vcc.TENSORRT_MAX_SUPPORTED_COMPUTE_CAP,
    excluded_machine_ids: Iterable[Any] = (),
    allowed_machine_ids: Iterable[Any] = (),
    require_avx: bool = False,
    require_known_supported_isaac_driver: bool = False,
    min_reliability: float = 0.0,
    require_direct_port: bool = False,
    preferred_gpu_keywords: Sequence[str] = (),
    preferred_geolocation_regex: str = "",
    prefer_isaac_rt: bool = True,
    gpu_selection_policy: str | Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    excluded = _machine_id_set(excluded_machine_ids)
    allowed = _machine_id_set(allowed_machine_ids)
    policy = resolve_gpu_selection_policy(gpu_selection_policy, prefer_isaac_rt=prefer_isaac_rt)
    summaries = [_offer_summary(offer) for offer in offers]
    candidates = [
        item
        for item in summaries
        if item["ask_contract_id"]
        and _number(item["hourly_rate_usd"]) is not None
        and float(item["hourly_rate_usd"]) <= max_hourly_rate
        and int(_number(item.get("gpu_ram_mb")) or 0) >= int(min_gpu_ram_mb)
        and vcc.meets_min_compute_cap(item, min_compute_cap)
        and vcc.meets_max_compute_cap(item, max_compute_cap)
        and (
            not min_reliability
            or (
                _number(item.get("reliability")) is not None
                and float(_number(item.get("reliability")) or 0.0) >= min_reliability
            )
        )
        and (not require_direct_port or int(_number(item.get("direct_port_count")) or 0) > 0)
        and gpu_allowed_by_policy(_string(item.get("gpu_name")), policy)
        and int(_number(item.get("machine_id")) or -1) not in excluded
        and (not allowed or int(_number(item.get("machine_id")) or -1) in allowed)
        and (not require_avx or item.get("has_avx") is True)
    ]
    if require_known_supported_isaac_driver:
        candidates = [
            item
            for item in candidates
            if item["isaac_driver_support_status"]
            == "outside_known_unsupported_omniverse_rtx_driver_range"
        ]
    if not candidates:
        return None
    rt_candidates = [item for item in candidates if item["isaac_rt_candidate"]]
    selected_pool = (rt_candidates or candidates) if prefer_isaac_rt else candidates
    return sorted(
        selected_pool,
        key=lambda item: (
            _driver_sort_rank(item),
            _driver_newer_branch_sort_rank(item),
            _keyword_match_rank(item.get("gpu_name"), preferred_gpu_keywords),
            _regex_match_rank(item.get("geolocation"), preferred_geolocation_regex),
            -float(_number(item.get("reliability")) or 0.0),
            -int(_number(item.get("direct_port_count")) or 0),
            float(item["hourly_rate_usd"] or 999),
            0 if item["isaac_rt_candidate"] else 1,
            str(item["gpu_name"]),
        ),
    )[0]


def _vast_stale_offer_create_retry_attempts() -> int:
    text = _string(os.getenv(VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS_ENV))
    if not text:
        return 2
    try:
        return max(0, int(text))
    except ValueError:
        return 2


def _is_stale_offer_create_http_error(exc: urllib.error.HTTPError) -> bool:
    return int(getattr(exc, "code", 0) or 0) in {404, 409, 410}


def _offer_selection_manifest(
    *,
    generated_at: str,
    status_code: int,
    offers: Sequence[Mapping[str, Any]],
    selected_offer: Mapping[str, Any] | None,
    max_hourly_rate: float,
    min_gpu_ram_mb: int,
    min_compute_cap: int = 0,
    max_compute_cap: int = vcc.TENSORRT_MAX_SUPPORTED_COMPUTE_CAP,
    require_known_supported_isaac_driver: bool,
    excluded_machine_ids: Iterable[Any],
    allowed_machine_ids: Iterable[Any],
    machine_avoidlist_path: Path,
    avoidlist_status: str | None,
    blockers: Sequence[str],
    min_reliability: float = 0.0,
    require_direct_port: bool = False,
    preferred_gpu_keywords: Sequence[str] = (),
    preferred_geolocation_regex: str = "",
    prefer_isaac_rt: bool = True,
    gpu_selection_policy: str | Mapping[str, Any] | None = None,
    create_retry_attempts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    policy = resolve_gpu_selection_policy(gpu_selection_policy, prefer_isaac_rt=prefer_isaac_rt)
    summaries = [_offer_summary(offer) for offer in offers]
    known_supported_offer_count = sum(
        1
        for item in summaries
        if item.get("isaac_driver_support_status")
        == "outside_known_unsupported_omniverse_rtx_driver_range"
    )
    known_unsupported_driver_offer_count = sum(
        1
        for item in summaries
        if item.get("isaac_driver_support_status") == "known_unsupported_omniverse_rtx_driver_range"
    )
    excluded = _machine_id_set(excluded_machine_ids)
    allowed = _machine_id_set(allowed_machine_ids)
    excluded_offer_count = sum(
        1 for item in summaries if int(_number(item.get("machine_id")) or -1) in excluded
    )
    allowed_offer_count = sum(
        1 for item in summaries if int(_number(item.get("machine_id")) or -1) in allowed
    )
    quality_filtered_offer_count = sum(
        1
        for item in summaries
        if item["ask_contract_id"]
        and _number(item["hourly_rate_usd"]) is not None
        and float(item["hourly_rate_usd"]) <= max_hourly_rate
        and int(_number(item.get("gpu_ram_mb")) or 0) >= int(min_gpu_ram_mb)
        and vcc.meets_min_compute_cap(item, min_compute_cap)
        and vcc.meets_max_compute_cap(item, max_compute_cap)
        and (
            not min_reliability
            or (
                _number(item.get("reliability")) is not None
                and float(_number(item.get("reliability")) or 0.0) >= min_reliability
            )
        )
        and (not require_direct_port or int(_number(item.get("direct_port_count")) or 0) > 0)
        and gpu_allowed_by_policy(_string(item.get("gpu_name")), policy)
        and int(_number(item.get("machine_id")) or -1) not in excluded
        and (not allowed or int(_number(item.get("machine_id")) or -1) in allowed)
    )
    return {
        "schema_version": VAST_OFFER_SELECTION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "selected" if selected_offer else "blocked",
        "offer_search_performed": True,
        "http_status_code": status_code,
        "offer_count": len(offers),
        "max_hourly_rate_usd": max_hourly_rate,
        "min_gpu_ram_mb": min_gpu_ram_mb,
        "min_compute_cap": min_compute_cap,
        "max_compute_cap": max_compute_cap,
        "architecture_excluded_offer_count": vcc.architecture_excluded_count(
            summaries, max_compute_cap
        ),
        "require_known_supported_isaac_driver": require_known_supported_isaac_driver,
        "min_reliability": min_reliability,
        "require_direct_port": require_direct_port,
        "preferred_gpu_keywords": list(preferred_gpu_keywords),
        "preferred_geolocation_regex": preferred_geolocation_regex,
        "prefer_isaac_rt": prefer_isaac_rt,
        "gpu_selection_policy": policy_manifest(policy),
        "quality_filtered_offer_count": quality_filtered_offer_count,
        "known_supported_driver_offer_count": known_supported_offer_count,
        "known_unsupported_driver_offer_count": known_unsupported_driver_offer_count,
        "selected_offer": _offer_artifact_summary(selected_offer),
        "selected_offer_isaac_rt_candidate": bool(
            selected_offer and selected_offer.get("isaac_rt_candidate")
        ),
        "machine_avoidlist_path": str(machine_avoidlist_path),
        "excluded_machine_ids": sorted(excluded),
        "excluded_offer_count": excluded_offer_count,
        "allowed_machine_ids": sorted(allowed),
        "allowlist_active": bool(allowed),
        "allowed_offer_count": allowed_offer_count if allowed else None,
        "avoidlist_status": avoidlist_status,
        "considered_offers": [_offer_artifact_summary(item) for item in summaries[:25]],
        "create_retry_attempts": [dict(item) for item in create_retry_attempts],
        "blockers": list(blockers),
        "raw_secret_values_recorded": False,
    }


def _budget_ledger(
    *,
    job_dir: Path,
    generated_at: str,
    target_spend_usd: float,
    hard_cap_usd: float,
    max_hourly_rate: float,
    max_live_minutes: int,
    selected_offer: Mapping[str, Any] | None,
    instance_ids: Sequence[int] = (),
    started_at_monotonic: float | None = None,
    ended_at_monotonic: float | None = None,
    status: str = "planned",
    continuing_spend: bool = False,
) -> dict[str, Any]:
    hourly = _number((selected_offer or {}).get("hourly_rate_usd")) or 0.0
    elapsed_seconds = 0.0
    if started_at_monotonic is not None and ended_at_monotonic is not None:
        elapsed_seconds = max(0.0, ended_at_monotonic - started_at_monotonic)
    estimated_cost = hourly * elapsed_seconds / 3600.0
    ledger = {
        "schema_version": VAST_BUDGET_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "target_spend_usd": target_spend_usd,
        "hard_cap_usd": hard_cap_usd,
        "max_hourly_rate_usd": max_hourly_rate,
        "max_live_runtime_minutes": max_live_minutes,
        "selected_hourly_rate_usd": hourly or None,
        "vast_instance_ids": list(instance_ids),
        "actual_live_runtime_seconds_observed_by_adapter": elapsed_seconds,
        "estimated_cost_usd": round(estimated_cost, 6),
        "actual_cost_usd": None,
        "actual_cost_source": "not_available_from_instance_probe_api",
        "estimated_spend_under_target": estimated_cost <= target_spend_usd,
        "estimated_spend_under_hard_cap": estimated_cost <= hard_cap_usd,
        "continuing_spend_from_this_run": continuing_spend,
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "vast_budget_ledger.json", ledger)
    return ledger


def _vast_session_budget_ledger_path() -> Path:
    override = _string(os.environ.get(VAST_SESSION_BUDGET_LEDGER_FILE_ENV))
    if override:
        return Path(override).expanduser().resolve()
    api_key_path = Path(
        os.environ.get(VAST_API_KEY_FILE_ENV, DEFAULT_VAST_API_KEY_FILE)
    ).expanduser()
    return api_key_path.parent / DEFAULT_VAST_SESSION_BUDGET_FILENAME


def _append_session_budget_attempt(
    *,
    budget_path: Path,
    job_dir: Path,
    generated_at: str,
    ledger: Mapping[str, Any],
    selected_offer: Mapping[str, Any] | None,
    result_status: str,
    result_reason: str | None,
    blockers: Sequence[str],
) -> dict[str, Any]:
    attempts: list[Mapping[str, Any]] = []
    parse_error: str | None = None
    if budget_path.is_file():
        try:
            payload = json.loads(budget_path.read_text(encoding="utf-8"))
            attempts = [item for item in payload.get("attempts") or [] if isinstance(item, Mapping)]
        except Exception as exc:
            parse_error = f"{type(exc).__name__}:{str(exc)[:200]}"
    attempt = {
        "generated_at": generated_at,
        "job_dir": str(job_dir),
        "status": result_status,
        "reason": result_reason,
        "blockers": list(blockers),
        "vast_instance_ids": list(ledger.get("vast_instance_ids") or []),
        "selected_hourly_rate_usd": ledger.get("selected_hourly_rate_usd"),
        "actual_live_runtime_seconds_observed_by_adapter": ledger.get(
            "actual_live_runtime_seconds_observed_by_adapter"
        ),
        "estimated_cost_usd": ledger.get("estimated_cost_usd"),
        "continuing_spend_from_this_run": ledger.get("continuing_spend_from_this_run"),
        "gpu_model_slug": _safe_slug((selected_offer or {}).get("gpu_name")),
        "machine_id": (selected_offer or {}).get("machine_id"),
        "offer_id": (selected_offer or {}).get("ask_contract_id"),
    }
    attempts = [
        item
        for item in attempts
        if str(item.get("job_dir")) != str(job_dir)
        or item.get("generated_at") != attempt["generated_at"]
    ]
    attempts.append(attempt)
    summary = {
        "schema_version": "vast_session_cost_summary.v4",
        "generated_at": generated_at,
        "status": "completed" if parse_error is None else "completed_after_parse_reset",
        "parse_error_recovered": parse_error,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "estimated_cost_usd": round(sum(_attempt_estimated_cost(item) for item in attempts), 6),
        "live_runtime_seconds": round(sum(_attempt_runtime_seconds(item) for item in attempts), 6),
        "raw_secret_values_recorded": False,
    }
    ensure_dir(budget_path.parent)
    write_json(budget_path, summary)
    return summary


def _attempt_runtime_seconds(attempt: Mapping[str, Any]) -> float:
    for key in (
        "runtime_seconds_observed_by_adapter",
        "actual_live_runtime_seconds_observed_by_adapter",
        "runtime_seconds_estimated_from_teardown_artifact_mtime",
    ):
        value = _number(attempt.get(key))
        if value is not None:
            return max(0.0, value)
    cost = _number(
        attempt.get("estimated_cost_usd_using_observed_rate")
        if attempt.get("estimated_cost_usd_using_observed_rate") is not None
        else attempt.get("estimated_cost_usd")
    )
    hourly = _number(attempt.get("observed_hourly_rate_usd")) or _number(
        attempt.get("selected_hourly_rate_usd")
    )
    if cost is not None and hourly and hourly > 0:
        return max(0.0, cost * 3600.0 / hourly)
    return 0.0


def _attempt_estimated_cost(attempt: Mapping[str, Any]) -> float:
    for key in ("estimated_cost_usd_using_observed_rate", "estimated_cost_usd"):
        value = _number(attempt.get(key))
        if value is not None:
            return max(0.0, value)
    return 0.0


def _session_budget_guard(
    *,
    job_dir: Path,
    generated_at: str,
    budget_path: Path,
    session_max_live_minutes: int | None,
    requested_max_live_minutes: int,
    target_spend_usd: float,
    hard_cap_usd: float,
    max_hourly_rate: float,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    attempts: list[Mapping[str, Any]] = []
    budget_parse_error = None
    if budget_path.is_file():
        try:
            payload = json.loads(budget_path.read_text(encoding="utf-8"))
            attempts = [item for item in payload.get("attempts") or [] if isinstance(item, Mapping)]
        except Exception as exc:
            budget_parse_error = f"{type(exc).__name__}:{str(exc)[:200]}"
            blockers.append("session_budget_ledger_parse_failed")
    prior_live_seconds = sum(_attempt_runtime_seconds(attempt) for attempt in attempts)
    prior_estimated_cost = sum(_attempt_estimated_cost(attempt) for attempt in attempts)
    requested_max_seconds = max(0, requested_max_live_minutes) * 60.0
    projected_max_cost = max(0.0, max_hourly_rate) * max(0, requested_max_live_minutes) / 60.0
    session_max_seconds = (
        max(0, session_max_live_minutes) * 60.0 if session_max_live_minutes is not None else None
    )
    if session_max_seconds is not None:
        if prior_live_seconds >= session_max_seconds:
            blockers.append("session_live_runtime_limit_exhausted")
        elif prior_live_seconds + requested_max_seconds > session_max_seconds:
            blockers.append("requested_live_runtime_would_exceed_session_limit")
    if prior_estimated_cost >= hard_cap_usd:
        blockers.append("session_estimated_spend_hard_cap_exhausted")
    elif prior_estimated_cost + projected_max_cost > hard_cap_usd:
        blockers.append("requested_max_spend_would_exceed_hard_cap")
    if prior_estimated_cost >= target_spend_usd:
        warnings.append("session_estimated_spend_target_already_exceeded")
    elif prior_estimated_cost + projected_max_cost > target_spend_usd:
        warnings.append("requested_max_spend_would_exceed_target")
    guard = {
        "schema_version": "vast_session_budget_guard.v1",
        "generated_at": generated_at,
        "status": "blocked" if blockers else "passed",
        "budget_path": str(budget_path),
        "budget_ledger_present": budget_path.is_file(),
        "budget_parse_error": budget_parse_error,
        "attempt_count": len(attempts),
        "prior_live_runtime_seconds": round(prior_live_seconds, 6),
        "prior_live_runtime_minutes": round(prior_live_seconds / 60.0, 6),
        "requested_max_live_runtime_minutes": requested_max_live_minutes,
        "session_max_live_runtime_minutes": session_max_live_minutes,
        "prior_estimated_cost_usd": round(prior_estimated_cost, 6),
        "projected_max_incremental_cost_usd": round(projected_max_cost, 6),
        "target_spend_usd": target_spend_usd,
        "hard_cap_usd": hard_cap_usd,
        "blockers": blockers,
        "warnings": warnings,
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "vast_session_budget_guard.json", guard)
    return guard


def _blueprint_bundle_preflight(
    *,
    job_dir: Path,
    generated_at: str,
    enable_blueprint_bundle: bool,
    enable_isaac_smoke: bool,
    provider_bundle_kind: str,
    bundle_path: Path | None,
    provider_bundle_url: str | None,
    provider_output_put_url: str | None,
    verify_staging_urls: bool = False,
    allow_staging_output_put_probe: bool = False,
) -> dict[str, Any]:
    if provider_bundle_kind not in VAST_PROVIDER_BUNDLE_KINDS:
        raise ValueError(f"unsupported_provider_bundle_kind:{provider_bundle_kind}")
    isaac_required_entries = {
        "provider_runtime/isaac_realistic_runtime_runner.py",
        "provider_runtime/run_isaac_realistic_runtime.sh",
        "provider_runtime/isaac_provider_eval_manifest.json",
        "provider_runtime/generated_site_scene.usda",
        "provider_runtime/generated_site_scene.usd",
        "provider_runtime/scenario_eval_matrix.json",
        "provider_runtime/camera_manifest.json",
        "provider_runtime/episode_spec_manifest.json",
    }
    wam_required_entries = {
        "provider_runtime/wam_provider_runtime_runner.py",
        "provider_runtime/run_wam_provider_runtime.sh",
        "provider_runtime/wam_provider_runtime_manifest.json",
        "provider_runtime/wam_rollout_input_manifest.json",
        "provider_runtime/oscar_input/first_frame.png",
        "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4",
    }
    unitree_unifolm_required_entries = {
        "provider_runtime/unitree_unifolm_provider_runner.py",
        "provider_runtime/run_unitree_unifolm_provider_runtime.sh",
        "provider_runtime/unitree_unifolm_policy_provider_manifest.json",
        "provider_runtime/policy_input.json",
        "provider_runtime/input_frame.png",
        "provider_runtime/blueprint_pipeline/__init__.py",
        "provider_runtime/blueprint_pipeline/unitree_unifolm_policy_command_adapter.py",
        "provider_runtime/blueprint_pipeline/unitree_unifolm_vla_server_bridge.py",
    }
    unitree_groot_required_entries = {
        "provider_runtime/unitree_groot_n17_sonic_provider_runner.py",
        "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh",
        "provider_runtime/unitree_groot_n17_sonic_policy_provider_manifest.json",
        "provider_runtime/policy_input.json",
        "provider_runtime/input_frame.png",
        "provider_runtime/blueprint_pipeline/__init__.py",
        "provider_runtime/blueprint_pipeline/common.py",
        "provider_runtime/blueprint_pipeline/unitree_groot_n17_sonic_policy_command_adapter.py",
        "provider_runtime/blueprint_pipeline/unitree_groot_n17_sonic_policy_runtime.py",
        "provider_runtime/blueprint_pipeline/unitree_groot_n17_sonic_policy_server_command.py",
    }
    if provider_bundle_kind == "isaac":
        required_entries = isaac_required_entries
        entrypoint_member = "provider_runtime/run_isaac_realistic_runtime.sh"
        runner_member = "provider_runtime/isaac_realistic_runtime_runner.py"
        readiness_name = "isaac_provider_bundle_readiness.json"
    elif provider_bundle_kind == "unitree_unifolm":
        required_entries = unitree_unifolm_required_entries
        entrypoint_member = "provider_runtime/run_unitree_unifolm_provider_runtime.sh"
        runner_member = "provider_runtime/unitree_unifolm_provider_runner.py"
        readiness_name = "unitree_unifolm_policy_provider_manifest.json"
    elif provider_bundle_kind == "unitree_groot_n17_sonic":
        required_entries = unitree_groot_required_entries
        entrypoint_member = "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh"
        runner_member = "provider_runtime/unitree_groot_n17_sonic_provider_runner.py"
        readiness_name = "unitree_groot_n17_sonic_policy_provider_manifest.json"
    else:
        required_entries = wam_required_entries
        entrypoint_member = "provider_runtime/run_wam_provider_runtime.sh"
        runner_member = "provider_runtime/wam_provider_runtime_runner.py"
        readiness_name = "oscar_wam_provider_bundle_manifest.json"
    blockers: list[str] = []
    warnings: list[str] = []
    zip_entries: list[str] = []
    missing_entries: list[str] = []
    zip_parse_error = None
    zip_testzip_result: str | None = None
    json_member_parse_errors: list[str] = []
    entrypoint_text = ""
    runner_text = ""
    eval_manifest: dict[str, Any] = {}
    eval_manifest_parse_error: str | None = None
    readiness_path = (
        bundle_path.parent / readiness_name if bundle_path else job_dir / readiness_name
    )
    if (
        provider_bundle_kind in {"unitree_unifolm", "unitree_groot_n17_sonic"}
        and not readiness_path.is_file()
    ):
        readiness_path = (
            bundle_path.parent / "provider_runtime" / readiness_name
            if bundle_path
            else job_dir / "provider_runtime" / readiness_name
        )
    readiness: dict[str, Any] = {}
    readiness_parse_error = None
    bundle_url_probe: dict[str, Any] = {"status": "not_requested"}
    output_put_probe: dict[str, Any] = {"status": "not_requested"}
    bundle_url_blocker = _provider_url_public_blocker(
        provider_bundle_url,
        "provider_bundle_fetch",
    )
    output_put_url_blocker = _provider_url_public_blocker(
        provider_output_put_url,
        "provider_output_put",
    )

    if enable_blueprint_bundle:
        if provider_bundle_kind == "isaac" and not enable_isaac_smoke:
            blockers.append("blueprint_bundle_execution_requires_isaac_smoke_path")
        if not bundle_path or not bundle_path.is_file():
            blockers.append(
                "isaac_provider_runtime_bundle_missing"
                if provider_bundle_kind == "isaac"
                else "provider_runtime_bundle_missing"
            )
        else:
            try:
                with zipfile.ZipFile(bundle_path) as archive:
                    zip_entries = sorted(archive.namelist())
                    zip_testzip_result = archive.testzip()
                    for member in zip_entries:
                        if member.endswith(".json"):
                            try:
                                json.loads(archive.read(member).decode("utf-8", errors="replace"))
                            except Exception as exc:
                                json_member_parse_errors.append(f"{member}:{type(exc).__name__}")
                    if entrypoint_member in zip_entries:
                        entrypoint_text = archive.read(entrypoint_member).decode(
                            "utf-8", errors="replace"
                        )
                    if runner_member in zip_entries:
                        runner_text = archive.read(runner_member).decode("utf-8", errors="replace")
                    if (
                        provider_bundle_kind == "isaac"
                        and "provider_runtime/isaac_provider_eval_manifest.json" in zip_entries
                    ):
                        try:
                            eval_payload = json.loads(
                                archive.read(
                                    "provider_runtime/isaac_provider_eval_manifest.json"
                                ).decode("utf-8", errors="replace")
                            )
                            eval_manifest = (
                                dict(eval_payload) if isinstance(eval_payload, Mapping) else {}
                            )
                        except Exception as exc:
                            eval_manifest_parse_error = f"{type(exc).__name__}:{str(exc)[:300]}"
                            blockers.append("provider_eval_manifest_parse_failed")
            except Exception as exc:
                zip_parse_error = f"{type(exc).__name__}:{str(exc)[:300]}"
                blockers.append(
                    f"provider_runtime_bundle_zip_inspection_failed:{type(exc).__name__}"
                )
            missing_entries = sorted(required_entries - set(zip_entries))
            if missing_entries:
                blockers.append("provider_runtime_bundle_required_entries_missing")
            if zip_testzip_result is not None:
                blockers.append("provider_runtime_bundle_zip_integrity_failed")
            if json_member_parse_errors:
                blockers.append("provider_runtime_bundle_json_parse_failed")
            if zip_entries:
                if provider_bundle_kind == "isaac":
                    entrypoint_has_crash_fallback = (
                        "write_missing_result" in entrypoint_text
                        and "isaac_runner_process_exited_without_runtime_result" in entrypoint_text
                        and "blocked_isaac_process_exited_without_result" in entrypoint_text
                    )
                    runner_has_required_runtime = "SimulationApp" in runner_text
                    missing_runtime_blocker = "provider_runner_missing_isaac_simulation_app_smoke"
                elif provider_bundle_kind == "unitree_unifolm":
                    entrypoint_has_crash_fallback = (
                        "unitree_unifolm_provider_runner_failed_without_runtime_result"
                        in entrypoint_text
                        and "blocked_unitree_unifolm_process_exited_without_result"
                        in entrypoint_text
                    )
                    runner_has_required_runtime = (
                        "unitree_unifolm_policy_provider_output.json" in runner_text
                        and "unitree_unifolm_model_executed" in runner_text
                        and "unitree_unifolm_policy_action_command_ran" in runner_text
                    )
                    missing_runtime_blocker = (
                        "provider_runner_missing_unitree_unifolm_runtime_contract"
                    )
                elif provider_bundle_kind == "unitree_groot_n17_sonic":
                    entrypoint_has_crash_fallback = (
                        "unitree_groot_n17_sonic_provider_runner_failed_without_runtime_result"
                        in entrypoint_text
                        and "blocked_unitree_groot_n17_sonic_process_exited_without_result"
                        in entrypoint_text
                    )
                    runner_has_required_runtime = (
                        "unitree_groot_n17_sonic_policy_provider_output.json" in runner_text
                        and "unitree_groot_n17_sonic_model_executed" in runner_text
                        and "unitree_groot_n17_sonic_policy_action_command_ran" in runner_text
                    )
                    missing_runtime_blocker = (
                        "provider_runner_missing_unitree_groot_n17_sonic_runtime_contract"
                    )
                else:
                    entrypoint_has_crash_fallback = (
                        "write_missing_result" in entrypoint_text
                        and "wam_runner_process_exited_without_runtime_result" in entrypoint_text
                        and "blocked_wam_process_exited_without_result" in entrypoint_text
                    )
                    runner_has_required_runtime = (
                        "wam_runtime_result.json" in runner_text
                        and "OSCAR-2B" in runner_text
                        and "action_conditioned_video_rollout_generated" in runner_text
                    )
                    missing_runtime_blocker = "provider_runner_missing_wam_runtime_contract"
                if not entrypoint_has_crash_fallback:
                    blockers.append("provider_entrypoint_missing_runtime_result_crash_fallback")
                if not runner_has_required_runtime:
                    blockers.append(missing_runtime_blocker)
                if provider_bundle_kind == "isaac":
                    relative_paths = _mapping(eval_manifest.get("relative_paths"))
                    prefixed_relative_paths = sorted(
                        key
                        for key, value in relative_paths.items()
                        if _string(value).startswith("provider_runtime/")
                    )
                    runner_has_bundle_path_resolver = (
                        "_resolve_bundle_relative_path" in runner_text
                        and "_bundle_relative_path_candidates" in runner_text
                    )
                    if prefixed_relative_paths and not runner_has_bundle_path_resolver:
                        blockers.append(
                            "provider_runtime_bundle_stale_prefixed_paths_without_resolver"
                        )
        if readiness_path.is_file():
            try:
                readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
            except Exception as exc:
                readiness_parse_error = f"{type(exc).__name__}:{str(exc)[:300]}"
                blockers.append("provider_bundle_readiness_parse_failed")
            local_ready = readiness.get("local_bundle_ready_for_remote_staging")
            if local_ready is False:
                blockers.append("provider_bundle_readiness_local_failed")
                for blocker in _string_list(readiness.get("blockers")):
                    if not blocker.startswith("provider_launch_request_blocked:"):
                        blockers.append(f"provider_bundle_readiness:{blocker}")
        else:
            warnings.append("provider_bundle_readiness_manifest_missing")
        if not _string(provider_bundle_url):
            blockers.append("provider_bundle_fetch_url_missing")
        elif bundle_url_blocker:
            blockers.append(bundle_url_blocker)
        if not _string(provider_output_put_url):
            blockers.append("provider_output_put_url_missing")
        elif output_put_url_blocker:
            blockers.append(output_put_url_blocker)
        if verify_staging_urls and _string(provider_bundle_url) and not bundle_url_blocker:
            request = urllib.request.Request(_string(provider_bundle_url), method="HEAD")
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    status_code = int(getattr(response, "status", 200))
                    headers = dict(response.headers.items())
                content_length = _number(headers.get("Content-Length"))
                bundle_url_probe = {
                    "status": "passed" if 200 <= status_code < 300 else "blocked",
                    "method": "HEAD",
                    "http_status_code": status_code,
                    "content_type": headers.get("Content-Type"),
                    "content_length": int(content_length) if content_length is not None else None,
                }
                if not (200 <= status_code < 300):
                    blockers.append("provider_bundle_fetch_url_unreachable")
                elif (
                    bundle_path
                    and bundle_path.is_file()
                    and content_length is not None
                    and int(content_length) != bundle_path.stat().st_size
                ):
                    blockers.append("provider_bundle_fetch_url_size_mismatch")
            except urllib.error.HTTPError as exc:
                if exc.code in {403, 405}:
                    fallback_request = urllib.request.Request(
                        _string(provider_bundle_url),
                        method="GET",
                        headers={"Range": "bytes=0-0"},
                    )
                    try:
                        with urllib.request.urlopen(fallback_request, timeout=20) as response:
                            status_code = int(getattr(response, "status", 200))
                            headers = dict(response.headers.items())
                            response.read(1)
                        content_length = _content_range_total_bytes(headers.get("Content-Range"))
                        if content_length is None:
                            content_length = _number(headers.get("Content-Length"))
                        bundle_url_probe = {
                            "status": "passed" if 200 <= status_code < 300 else "blocked",
                            "method": "GET",
                            "range_request": "bytes=0-0",
                            "head_http_status_code": exc.code,
                            "http_status_code": status_code,
                            "content_type": headers.get("Content-Type"),
                            "content_range": headers.get("Content-Range"),
                            "content_length": int(content_length)
                            if content_length is not None
                            else None,
                            "head_error_type": type(exc).__name__,
                        }
                        if not (200 <= status_code < 300):
                            blockers.append("provider_bundle_fetch_url_unreachable")
                        elif (
                            bundle_path
                            and bundle_path.is_file()
                            and content_length is not None
                            and int(content_length) != bundle_path.stat().st_size
                        ):
                            blockers.append("provider_bundle_fetch_url_size_mismatch")
                    except urllib.error.HTTPError as fallback_exc:
                        bundle_url_probe = {
                            "status": "blocked",
                            "method": "GET",
                            "range_request": "bytes=0-0",
                            "head_http_status_code": exc.code,
                            "http_status_code": fallback_exc.code,
                            "head_error_type": type(exc).__name__,
                            "error_type": type(fallback_exc).__name__,
                        }
                        blockers.append("provider_bundle_fetch_url_unreachable")
                    except urllib.error.URLError as fallback_exc:
                        bundle_url_probe = {
                            "status": "blocked",
                            "method": "GET",
                            "range_request": "bytes=0-0",
                            "head_http_status_code": exc.code,
                            "head_error_type": type(exc).__name__,
                            "error_type": type(fallback_exc).__name__,
                            "reason_type": type(fallback_exc.reason).__name__,
                        }
                        blockers.append("provider_bundle_fetch_url_unreachable")
                    except Exception as fallback_exc:
                        bundle_url_probe = {
                            "status": "blocked",
                            "method": "GET",
                            "range_request": "bytes=0-0",
                            "head_http_status_code": exc.code,
                            "head_error_type": type(exc).__name__,
                            "error_type": type(fallback_exc).__name__,
                        }
                        blockers.append("provider_bundle_fetch_url_unreachable")
                else:
                    bundle_url_probe = {
                        "status": "blocked",
                        "method": "HEAD",
                        "http_status_code": exc.code,
                        "error_type": type(exc).__name__,
                    }
                    blockers.append("provider_bundle_fetch_url_unreachable")
            except urllib.error.URLError as exc:
                fallback_probe = _head_with_public_dns_fallback(
                    _string(provider_bundle_url),
                    timeout_seconds=90,
                )
                bundle_url_probe = {
                    **fallback_probe,
                    "normal_head_error_type": type(exc).__name__,
                    "normal_head_reason_type": type(exc.reason).__name__,
                }
                content_length = _number(fallback_probe.get("content_length"))
                if fallback_probe.get("status") != "passed":
                    blockers.append("provider_bundle_fetch_url_unreachable")
                elif (
                    bundle_path
                    and bundle_path.is_file()
                    and content_length is not None
                    and int(content_length) != bundle_path.stat().st_size
                ):
                    blockers.append("provider_bundle_fetch_url_size_mismatch")
            except Exception as exc:
                bundle_url_probe = {
                    "status": "blocked",
                    "method": "HEAD",
                    "error_type": type(exc).__name__,
                }
                blockers.append("provider_bundle_fetch_url_unreachable")
        elif verify_staging_urls:
            bundle_url_probe = {
                "status": "blocked",
                "method": "HEAD",
                "blockers": [bundle_url_blocker or "provider_bundle_fetch_url_missing"],
            }

        if verify_staging_urls and _string(provider_output_put_url) and not output_put_url_blocker:
            if allow_staging_output_put_probe:
                probe_zip = b"PK\x05\x06" + (b"\x00" * 18)
                request = urllib.request.Request(
                    _string(provider_output_put_url),
                    data=probe_zip,
                    method="PUT",
                    headers={"Content-Type": "application/zip"},
                )
                try:
                    with urllib.request.urlopen(request, timeout=20) as response:
                        status_code = int(getattr(response, "status", 200))
                        response_text = response.read().decode("utf-8", errors="replace")
                    output_put_probe = {
                        "status": "passed" if 200 <= status_code < 300 else "blocked",
                        "method": "PUT",
                        "http_status_code": status_code,
                        "probe_bytes": len(probe_zip),
                        "response_preview": response_text[:200],
                    }
                    if not (200 <= status_code < 300):
                        blockers.append("provider_output_put_url_unwritable")
                except urllib.error.HTTPError as exc:
                    output_put_probe = {
                        "status": "blocked",
                        "method": "PUT",
                        "http_status_code": exc.code,
                        "probe_bytes": len(probe_zip),
                        "error_type": type(exc).__name__,
                    }
                    blockers.append("provider_output_put_url_unwritable")
                except urllib.error.URLError as exc:
                    output_put_probe = {
                        "status": "blocked",
                        "method": "PUT",
                        "probe_bytes": len(probe_zip),
                        "error_type": type(exc).__name__,
                        "reason_type": type(exc.reason).__name__,
                    }
                    blockers.append("provider_output_put_url_unwritable")
                except Exception as exc:
                    output_put_probe = {
                        "status": "blocked",
                        "method": "PUT",
                        "probe_bytes": len(probe_zip),
                        "error_type": type(exc).__name__,
                    }
                    blockers.append("provider_output_put_url_unwritable")
            else:
                output_put_probe = {
                    "status": "skipped",
                    "reason": "output_put_probe_requires_explicit_allow",
                    "method": "PUT",
                }
                warnings.append("provider_output_put_url_not_mutation_probed")
        elif verify_staging_urls:
            output_put_probe = {
                "status": "blocked",
                "method": "PUT",
                "blockers": [output_put_url_blocker or "provider_output_put_url_missing"],
            }

    manifest = {
        "schema_version": VAST_BLUEPRINT_BUNDLE_PREFLIGHT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked"
        if blockers
        else ("passed" if enable_blueprint_bundle else "not_required"),
        "blueprint_bundle_enabled": enable_blueprint_bundle,
        "provider_bundle_kind": provider_bundle_kind,
        "isaac_smoke_enabled": enable_isaac_smoke,
        "bundle_path": str(bundle_path) if bundle_path else None,
        "bundle_present": bool(bundle_path and bundle_path.is_file()),
        "bundle_size_bytes": bundle_path.stat().st_size
        if bundle_path and bundle_path.is_file()
        else 0,
        "provider_bundle_fetch_url_present": bool(_string(provider_bundle_url)),
        "provider_output_put_url_present": bool(_string(provider_output_put_url)),
        "staging_url_verification_requested": verify_staging_urls,
        "staging_output_put_probe_allowed": allow_staging_output_put_probe,
        "bundle_url_probe": bundle_url_probe,
        "output_put_probe": output_put_probe,
        "zip_entry_count": len(zip_entries),
        "zip_required_entries_present": bool(zip_entries) and not missing_entries,
        "zip_integrity_test_passed": bool(zip_entries)
        and zip_parse_error is None
        and zip_testzip_result is None,
        "zip_testzip_result": zip_testzip_result,
        "json_member_parse_errors": json_member_parse_errors,
        "missing_zip_entries": missing_entries,
        "zip_parse_error": zip_parse_error,
        "provider_eval_manifest_parse_error": eval_manifest_parse_error,
        "provider_eval_manifest_relative_paths": _mapping(eval_manifest.get("relative_paths")),
        "provider_bundle_readiness_path": str(readiness_path),
        "provider_bundle_readiness_present": readiness_path.is_file(),
        "provider_bundle_readiness_parse_error": readiness_parse_error,
        "provider_bundle_local_ready_for_remote_staging": readiness.get(
            "local_bundle_ready_for_remote_staging"
        ),
        "blockers": sorted(set(blockers)),
        "warnings": warnings,
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "vast_blueprint_bundle_preflight.json", manifest)
    return manifest


def _isaac_image_startup_preflight(
    *,
    job_dir: Path,
    generated_at: str,
    enable_isaac_smoke: bool,
    enable_blueprint_bundle: bool,
    provider_bundle_kind: str = "isaac",
    selected_container_image: str,
    vast_template_hash_id: str | None,
    use_vast_template_image: bool,
    max_live_minutes: int,
    allow_cold_isaac_image_pull: bool,
    min_cold_isaac_pull_live_minutes: int,
) -> dict[str, Any]:
    if provider_bundle_kind not in VAST_PROVIDER_BUNDLE_KINDS:
        raise ValueError(f"unsupported_provider_bundle_kind:{provider_bundle_kind}")
    blockers: list[str] = []
    warnings: list[str] = []
    isaac_path_requested = enable_isaac_smoke or (
        enable_blueprint_bundle and provider_bundle_kind == "isaac"
    )
    template_hash = _string(vast_template_hash_id)
    direct_official_isaac_image = (
        isaac_path_requested
        and not use_vast_template_image
        and selected_container_image == DEFAULT_ISAAC_IMAGE
    )
    template_image_cache_proven = False
    template_image_cache_evidence = (
        "not_requested" if not use_vast_template_image else "not_proven_by_vast_template_hash"
    )
    custom_or_template_image = bool(
        isaac_path_requested
        and (
            use_vast_template_image
            or (selected_container_image and selected_container_image != DEFAULT_ISAAC_IMAGE)
        )
    )
    if use_vast_template_image and not template_hash:
        blockers.append("vast_template_hash_required_when_using_template_image")
    if direct_official_isaac_image and not allow_cold_isaac_image_pull:
        blockers.append("cold_official_isaac_image_pull_not_authorized")
    if (
        direct_official_isaac_image
        and allow_cold_isaac_image_pull
        and max_live_minutes < min_cold_isaac_pull_live_minutes
    ):
        blockers.append("cold_official_isaac_image_pull_live_window_too_short")
    if (
        isaac_path_requested
        and use_vast_template_image
        and max_live_minutes < min_cold_isaac_pull_live_minutes
    ):
        blockers.append("vast_template_image_cache_not_proven_for_short_live_window")
    if direct_official_isaac_image:
        warnings.append(
            "previous_live_attempts_showed_nvcr_isaac_image_cold_pull_can_exceed_short_startup_windows"
        )
    if isaac_path_requested and use_vast_template_image:
        warnings.append("vast_template_hash_is_launch_configuration_not_image_cache_proof")
    if isaac_path_requested and custom_or_template_image:
        warnings.append(
            "custom_or_template_isaac_image_path_must_still_prove_onstart_heartbeat_before_bundle_execution"
        )
    manifest = {
        "schema_version": VAST_ISAAC_IMAGE_STARTUP_PREFLIGHT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked" if blockers else ("passed" if isaac_path_requested else "not_required"),
        "provider_bundle_kind": provider_bundle_kind,
        "isaac_path_requested": isaac_path_requested,
        "selected_container_image": selected_container_image,
        "default_official_isaac_image": DEFAULT_ISAAC_IMAGE,
        "direct_official_isaac_image": direct_official_isaac_image,
        "vast_template_hash_present": bool(template_hash),
        "use_vast_template_image": use_vast_template_image,
        "template_image_cache_proven": template_image_cache_proven,
        "template_image_cache_evidence": template_image_cache_evidence,
        "custom_or_template_image_path": custom_or_template_image,
        "allow_cold_isaac_image_pull": allow_cold_isaac_image_pull,
        "max_live_minutes": max_live_minutes,
        "min_cold_isaac_pull_live_minutes": min_cold_isaac_pull_live_minutes,
        "blockers": blockers,
        "warnings": warnings,
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "vast_isaac_image_startup_preflight.json", manifest)
    return manifest


def _make_env_string(env: Mapping[str, str]) -> str:
    parts: list[str] = []
    for key, value in env.items():
        if not key or value is None:
            continue
        parts.append("-e")
        parts.append(f"{key}={value}")
    return " ".join(shlex.quote(item) for item in parts)


def _resolve_launch_mode(
    *,
    requested: str,
    enable_isaac_smoke: bool,
    enable_blueprint_bundle: bool = False,
    provider_bundle_kind: str = "isaac",
) -> str:
    if requested not in VAST_LAUNCH_MODES:
        raise ValueError(f"unsupported_vast_launch_mode:{requested}")
    if provider_bundle_kind not in VAST_PROVIDER_BUNDLE_KINDS:
        raise ValueError(f"unsupported_provider_bundle_kind:{provider_bundle_kind}")
    if requested == "auto":
        if enable_blueprint_bundle and provider_bundle_kind in {"wam", "unitree_unifolm"}:
            return "ssh_direct"
        return "args" if enable_isaac_smoke else "ssh_direct"
    return requested


def _resolve_disk_gb(*, requested: int | None, enable_isaac_smoke: bool) -> int:
    if requested is not None:
        return requested
    return DEFAULT_ISAAC_DISK_GB if enable_isaac_smoke else DEFAULT_PUBLIC_DISK_GB


def _resolve_probe_image(
    *,
    public_image: str,
    isaac_image: str,
    enable_isaac_smoke: bool,
    enable_blueprint_bundle: bool,
    provider_bundle_kind: str = "isaac",
) -> str:
    if provider_bundle_kind not in VAST_PROVIDER_BUNDLE_KINDS:
        raise ValueError(f"unsupported_provider_bundle_kind:{provider_bundle_kind}")
    if enable_isaac_smoke or (enable_blueprint_bundle and provider_bundle_kind == "isaac"):
        return isaac_image
    return public_image


def _probe_env(
    *,
    job_dir: Path,
    enable_isaac_smoke: bool,
    provider_bundle_url: str | None = None,
    provider_output_put_url: str | None = None,
    provider_bundle_inline_base64: str | None = None,
    provider_bundle_inline_sha256: str | None = None,
) -> dict[str, str]:
    env = {
        "BLUEPRINT_VAST_PROBE": "true",
        "BLUEPRINT_VAST_PROBE_JOB_DIR_BASENAME": job_dir.name,
    }
    if enable_isaac_smoke:
        env.update(
            {
                "ACCEPT_EULA": "Y",
                "PRIVACY_CONSENT": "Y",
                "NVIDIA_DRIVER_CAPABILITIES": "all",
            }
        )
    if _string(provider_bundle_url):
        env["BLUEPRINT_EVAL_MANIFEST_URI"] = _string(provider_bundle_url)
    if _string(provider_output_put_url):
        env["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"] = _string(provider_output_put_url)
    if _string(provider_bundle_inline_base64):
        env[VAST_INLINE_PROVIDER_BUNDLE_BASE64_ENV] = _string(provider_bundle_inline_base64)
    if _string(provider_bundle_inline_sha256):
        env[VAST_INLINE_PROVIDER_BUNDLE_SHA256_ENV] = _string(provider_bundle_inline_sha256)
    hf_token, _hf_token_status = _read_hf_token_file()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    for env_name in (
        "BLUEPRINT_OSCAR_WAM_TRANSFORMER_ENGINE_STRATEGY",
        "BLUEPRINT_OSCAR_WAM_SKIP_RUNTIME_PIP_INSTALL",
        "BLUEPRINT_OSCAR_WAM_ATTEMPT_TRANSFORMER_ENGINE_INSTALL",
        "BLUEPRINT_OSCAR_WAM_OMIT_FPS_ARG",
        "BLUEPRINT_OSCAR_WAM_NUM_STEPS",
        "BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE",
        "BLUEPRINT_OSCAR_WAM_NUM_FRAMES",
        "BLUEPRINT_OSCAR_WAM_HEIGHT",
        "BLUEPRINT_OSCAR_WAM_WIDTH",
        "BLUEPRINT_OSCAR_WAM_FPS",
        "BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS",
        "BLUEPRINT_OSCAR_WAM_ENABLE_HF_TRANSFER",
        "BLUEPRINT_WAM_PROVIDER_DISABLE_VENV",
        "BLUEPRINT_WAM_PROVIDER_ALLOW_BREAK_SYSTEM_PACKAGES",
        "BLUEPRINT_OSCAR_WAM_HF_REPO",
        "BLUEPRINT_OSCAR_WAM_SOURCE_URL",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PERSISTENT_INNER_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_VAST_INNER_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT",
        "BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_TIMEOUT_SECONDS",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_AUTO_START_POLICY_SERVER",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON_INSTALL_REQUIREMENTS",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SKIP_SYSTEM_PYTHON_DEPS_INSTALL",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON_DEPS_TIMEOUT_SECONDS",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_REPO_URL",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_REPO_REF",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_REMOTE_ROOT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_UV_SYNC_TIMEOUT_SECONDS",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SERVER_STARTUP_TIMEOUT_SECONDS",
    ):
        value = os.getenv(env_name)
        if value:
            env[env_name] = value
    for env_name in (
        item.strip() for item in _string(os.getenv(VAST_FORWARD_SECRET_ENV_VARS_ENV)).split(",")
    ):
        if not any(marker in env_name.upper() for marker in SENSITIVE_KEY_MARKERS):
            continue
        value = os.getenv(env_name)
        if value:
            env[env_name] = value
    return env


def _forwarded_secret_values() -> list[str]:
    values: list[str] = _hf_token_secret_values()
    for env_name in (
        item.strip() for item in _string(os.getenv(VAST_FORWARD_SECRET_ENV_VARS_ENV)).split(",")
    ):
        if not any(marker in env_name.upper() for marker in SENSITIVE_KEY_MARKERS):
            continue
        value = os.getenv(env_name)
        if value:
            values.append(value)
    return _dedupe(values)


def _official_public_isaac_image(image: str) -> bool:
    return image.startswith("nvcr.io/nvidia/isaac-sim:")


def _docker_hub_image_requires_login(image: str, docker_username: str) -> bool:
    normalized = _string(image)
    username = _string(docker_username)
    if not normalized or not username:
        return False
    namespace_prefixes = (
        f"docker.io/{username}/",
        f"index.docker.io/{username}/",
        f"registry-1.docker.io/{username}/",
        f"{username}/",
    )
    return any(normalized.startswith(prefix) for prefix in namespace_prefixes)


def _resolve_image_login(
    *,
    image: str,
    ngc_key: str,
    docker_username: str = "",
    docker_pat: str = "",
    mode: str,
) -> tuple[str | None, dict[str, Any]]:
    if mode not in NGC_IMAGE_LOGIN_MODES:
        raise ValueError(f"unsupported_ngc_image_login_mode:{mode}")
    if not image.startswith("nvcr.io/"):
        if mode == "never":
            return None, {
                "mode": mode,
                "reason": "docker_hub_image_login_disabled",
                "image_login_supplied": False,
                "docker_secret_file_present": bool(docker_pat),
                "docker_username_present": bool(docker_username),
            }
        if _docker_hub_image_requires_login(image, docker_username):
            if not docker_pat:
                return None, {
                    "mode": mode,
                    "reason": "docker_pat_missing",
                    "image_login_supplied": False,
                    "docker_secret_file_present": False,
                    "docker_username_present": bool(docker_username),
                }
            return f"-u {docker_username} -p {docker_pat} docker.io", {
                "mode": mode,
                "reason": "docker_hub_image_login_supplied",
                "image_login_supplied": True,
                "docker_secret_file_present": True,
                "docker_username_present": True,
                "docker_registry": "docker.io",
            }
        return None, {
            "mode": mode,
            "reason": "non_ngc_image",
            "image_login_supplied": False,
            "docker_secret_file_present": bool(docker_pat),
            "docker_username_present": bool(docker_username),
        }
    if mode == "never" or (mode == "auto" and _official_public_isaac_image(image)):
        return None, {
            "mode": mode,
            "reason": "public_official_isaac_image_without_registry_login"
            if _official_public_isaac_image(image)
            else "ngc_image_login_disabled",
            "image_login_supplied": False,
            "ngc_secret_file_present": bool(ngc_key),
        }
    if not ngc_key:
        return None, {
            "mode": mode,
            "reason": "ngc_key_missing",
            "image_login_supplied": False,
        }
    # Vast's image_login field takes docker-login-style args, not a shell
    # command line. Quote the whole field at the CLI/API boundary, not the
    # username inside the value.
    return f"-u $oauthtoken -p {ngc_key} nvcr.io", {
        "mode": mode,
        "reason": "ngc_image_login_supplied",
        "image_login_supplied": True,
    }


def _create_payload(
    *,
    image: str | None,
    label: str,
    launch_mode: str,
    probe_script: str,
    disk_gb: int,
    env: Mapping[str, str] | None = None,
    image_login: str | None = None,
    template_hash_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": label,
        "disk": disk_gb,
        "runtype": launch_mode,
        "target_state": "running",
        "cancel_unavail": True,
        "env": dict(env or {}),
    }
    if image:
        payload["image"] = image
    if _string(template_hash_id):
        payload["template_hash_id"] = _string(template_hash_id)
    if launch_mode == "args":
        # Vast API args mode uses args_str, not onstart. Run through bash so
        # images whose entrypoint execs CMD still receive an executable command.
        # Keep the container alive briefly after emitting markers; otherwise
        # request_logs can race an already-exited args container and return
        # "No such container" with no useful stdout.
        # Run the probe in a subshell while the outer wrapper has errexit off.
        # This captures failures even when the probe enables ``set -e``.  A
        # newline before ``)`` also keeps a terminal heredoc delimiter valid;
        # appending ``;`` directly after it produces a leading-semicolon syntax
        # error in ``bash -c``.
        wrapped_script = (
            "set +e\n"
            "(\n"
            f"{probe_script.rstrip()}\n"
            ")\n"
            "script_rc=$?\n"
            "echo BLUEPRINT_VAST_ARGS_LOG_HOLD_STARTED\n"
            f"sleep ${{BLUEPRINT_VAST_ARGS_LOG_HOLD_SECONDS:-{DEFAULT_ARGS_LOG_HOLD_SECONDS}}}\n"
            "echo BLUEPRINT_VAST_ARGS_LOG_HOLD_DONE\n"
            'exit "$script_rc"'
        )
        payload["args_str"] = "bash -lc " + shlex.quote(wrapped_script)
    else:
        payload["onstart"] = probe_script
        if launch_mode == "jupyter_direct":
            payload["use_jupyter_lab"] = True
            payload["jupyter_dir"] = "/workspace"
    if image_login:
        payload["image_login"] = image_login
    return payload


def _probe_shell_script(
    heartbeat_url: str,
    *,
    enable_isaac_smoke: bool = False,
    enable_blueprint_bundle: bool = False,
    provider_bundle_kind: str = "isaac",
) -> str:
    if provider_bundle_kind not in VAST_PROVIDER_BUNDLE_KINDS:
        raise ValueError(f"unsupported_provider_bundle_kind:{provider_bundle_kind}")
    quoted_url = shlex.quote(heartbeat_url)
    script = (
        "set +e; WORK_DIR=/workspace; "
        'mkdir -p "$WORK_DIR/blueprint_vast_probe" 2>/dev/null || '
        '{ WORK_DIR=/tmp/blueprint_vast_work; mkdir -p "$WORK_DIR/blueprint_vast_probe"; }; '
        'export BLUEPRINT_VAST_WORK_DIR="$WORK_DIR"; '
        "echo BLUEPRINT_VAST_WORK_DIR:$WORK_DIR; "
        "PY_NET=''; "
        "if command -v python3 >/dev/null 2>&1; then PY_NET=$(command -v python3); "
        "elif command -v python >/dev/null 2>&1; then PY_NET=$(command -v python); fi; "
        "echo BLUEPRINT_VAST_ONSTART_STARTED; date -u; "
        "blueprint_http_get() { "
        'blueprint_get_url="$1"; '
        'if command -v curl >/dev/null 2>&1; then curl -fsSL "$blueprint_get_url"; return $?; fi; '
        'if command -v wget >/dev/null 2>&1; then wget -qO- "$blueprint_get_url"; return $?; fi; '
        'blueprint_get_py="${PY_NET:-${RUNTIME_PY:-}}"; '
        'if [ -n "$blueprint_get_py" ]; then '
        'BLUEPRINT_HTTP_GET_URL="$blueprint_get_url" "$blueprint_get_py" - <<\'PY\'\n'
        "import os\n"
        "import sys\n"
        "import urllib.request\n"
        "url = os.environ.get('BLUEPRINT_HTTP_GET_URL', '')\n"
        "try:\n"
        "    request = urllib.request.Request(url, headers={'User-Agent': 'BlueprintVastProbe/1.0'})\n"
        "    with urllib.request.urlopen(request, timeout=30) as response:\n"
        "        sys.stdout.buffer.write(response.read())\n"
        "except Exception as exc:\n"
        "    print('BLUEPRINT_VAST_PY_HTTP_GET_ERROR:%s' % type(exc).__name__)\n"
        "    raise SystemExit(1)\n"
        "PY\n"
        "return $?; "
        "fi; "
        "return 127; "
        "}; "
        "blueprint_download_url() { "
        'blueprint_download_src="$1"; blueprint_download_dst="$2"; '
        'if [ -n "${BLUEPRINT_VAST_PROVIDER_BUNDLE_BASE64:-}" ]; then '
        'blueprint_download_py="${PY_NET:-${RUNTIME_PY:-}}"; '
        'if [ -z "$blueprint_download_py" ]; then echo BLUEPRINT_VAST_INLINE_BUNDLE_DECODE_ERROR:python_missing; return 127; fi; '
        'BLUEPRINT_DOWNLOAD_PATH="$blueprint_download_dst" "$blueprint_download_py" - <<\'PY\'\n'
        "import base64\n"
        "import hashlib\n"
        "import os\n"
        "payload = os.environ.get('BLUEPRINT_VAST_PROVIDER_BUNDLE_BASE64', '')\n"
        "expected_sha256 = os.environ.get('BLUEPRINT_VAST_PROVIDER_BUNDLE_SHA256', '')\n"
        "dst = os.environ.get('BLUEPRINT_DOWNLOAD_PATH', '')\n"
        "try:\n"
        "    data = base64.b64decode(payload.encode('ascii'), validate=True)\n"
        "    actual_sha256 = hashlib.sha256(data).hexdigest()\n"
        "    if expected_sha256 and actual_sha256 != expected_sha256:\n"
        "        print('BLUEPRINT_VAST_INLINE_BUNDLE_SHA256_MISMATCH')\n"
        "        raise SystemExit(42)\n"
        "    with open(dst, 'wb') as handle:\n"
        "        handle.write(data)\n"
        "    print('BLUEPRINT_VAST_INLINE_BUNDLE_DECODED:%d' % len(data))\n"
        "except SystemExit:\n"
        "    raise\n"
        "except Exception as exc:\n"
        "    print('BLUEPRINT_VAST_INLINE_BUNDLE_DECODE_ERROR:%s' % type(exc).__name__)\n"
        "    raise SystemExit(43)\n"
        "PY\n"
        "return $?; "
        "fi; "
        'if command -v curl >/dev/null 2>&1; then curl -fL "$blueprint_download_src" -o "$blueprint_download_dst"; return $?; fi; '
        'if command -v wget >/dev/null 2>&1; then wget -O "$blueprint_download_dst" "$blueprint_download_src"; return $?; fi; '
        'blueprint_download_py="${PY_NET:-${RUNTIME_PY:-}}"; '
        'if [ -n "$blueprint_download_py" ]; then '
        'BLUEPRINT_DOWNLOAD_URL="$blueprint_download_src" BLUEPRINT_DOWNLOAD_PATH="$blueprint_download_dst" "$blueprint_download_py" - <<\'PY\'\n'
        "import os\n"
        "import shutil\n"
        "import urllib.request\n"
        "url = os.environ.get('BLUEPRINT_DOWNLOAD_URL', '')\n"
        "dst = os.environ.get('BLUEPRINT_DOWNLOAD_PATH', '')\n"
        "try:\n"
        "    request = urllib.request.Request(url, headers={'User-Agent': 'BlueprintVastProbe/1.0'})\n"
        "    with urllib.request.urlopen(request, timeout=120) as response, open(dst, 'wb') as handle:\n"
        "        shutil.copyfileobj(response, handle)\n"
        "except Exception as exc:\n"
        "    print('BLUEPRINT_VAST_PY_DOWNLOAD_ERROR:%s' % type(exc).__name__)\n"
        "    raise SystemExit(1)\n"
        "PY\n"
        "return $?; "
        "fi; "
        "return 127; "
        "}; "
        "blueprint_upload_put() { "
        'blueprint_upload_url="$1"; blueprint_upload_path="$2"; '
        'if command -v curl >/dev/null 2>&1; then curl -fsS -X PUT -H \'Content-Type: application/zip\' --data-binary @"$blueprint_upload_path" "$blueprint_upload_url" >/tmp/blueprint_provider_upload_response.json; return $?; fi; '
        'blueprint_upload_py="${PY_NET:-${RUNTIME_PY:-}}"; '
        'if [ -n "$blueprint_upload_py" ]; then '
        'BLUEPRINT_UPLOAD_URL="$blueprint_upload_url" BLUEPRINT_UPLOAD_PATH="$blueprint_upload_path" "$blueprint_upload_py" - <<\'PY\' >/tmp/blueprint_provider_upload_response.json\n'
        "import os\n"
        "import sys\n"
        "import urllib.request\n"
        "url = os.environ.get('BLUEPRINT_UPLOAD_URL', '')\n"
        "path = os.environ.get('BLUEPRINT_UPLOAD_PATH', '')\n"
        "try:\n"
        "    with open(path, 'rb') as handle:\n"
        "        data = handle.read()\n"
        "    request = urllib.request.Request(url, data=data, method='PUT', headers={'Content-Type': 'application/zip', 'User-Agent': 'BlueprintVastProbe/1.0'})\n"
        "    with urllib.request.urlopen(request, timeout=120) as response:\n"
        "        sys.stdout.buffer.write(response.read())\n"
        "except Exception as exc:\n"
        "    print('BLUEPRINT_VAST_PY_UPLOAD_ERROR:%s' % type(exc).__name__)\n"
        "    raise SystemExit(1)\n"
        "PY\n"
        "return $?; "
        "fi; "
        "return 127; "
        "}; "
        f"blueprint_http_get {quoted_url}; hb=$?; "
        "if [ $hb -eq 0 ]; then echo BLUEPRINT_VAST_HEARTBEAT_OK; "
        "else echo BLUEPRINT_VAST_HEARTBEAT_BLOCKED:$hb; fi; "
        "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; smi=$?; "
        "if [ $smi -eq 0 ]; then echo BLUEPRINT_VAST_GPU_SANITY_OK; "
        "else echo BLUEPRINT_VAST_GPU_SANITY_BLOCKED:$smi; fi; "
        'echo BLUEPRINT_VAST_DF_START; df -h "$WORK_DIR"; '
    )
    if enable_isaac_smoke:
        script += (
            "ISAAC_PY=''; "
            "if [ -x /isaac-sim/python.sh ]; then ISAAC_PY=/isaac-sim/python.sh; "
            "elif [ -x /isaac-sim/python ]; then ISAAC_PY=/isaac-sim/python; "
            "elif command -v python3 >/dev/null 2>&1; then ISAAC_PY=$(command -v python3); fi; "
            'if [ -n "$ISAAC_PY" ]; then '
            '$ISAAC_PY -c \'from isaacsim import SimulationApp; app=SimulationApp({"headless": True}); print("BLUEPRINT_VAST_ISAAC_SMOKE_OK", flush=True); import os; os._exit(0)\'; irc=$?; '
            "if [ $irc -eq 0 ]; then echo BLUEPRINT_VAST_ISAAC_SMOKE_COMPLETED; "
            "else echo BLUEPRINT_VAST_ISAAC_SMOKE_BLOCKED:$irc; fi; "
            "else echo BLUEPRINT_VAST_ISAAC_SMOKE_BLOCKED:python_missing; fi; "
        )
    else:
        script += "echo BLUEPRINT_VAST_ISAAC_SMOKE_SKIPPED; "
    if enable_blueprint_bundle:
        common_start = (
            "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED; "
            'BUNDLE_URL="${BLUEPRINT_EVAL_MANIFEST_URI:-}"; '
            'OUTPUT_PUT_URL="${BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL:-}"; '
            'if [ -z "$BUNDLE_URL" ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:bundle_url_missing; '
            'elif [ -z "$OUTPUT_PUT_URL" ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:output_put_url_missing; '
            "else "
        )
        if provider_bundle_kind == "isaac":
            script += (
                common_start
                + 'rm -rf "$WORK_DIR/isaac_provider_bundle" "$WORK_DIR/isaac_provider_runtime_bundle.zip" "$WORK_DIR/isaac_provider_runtime_output.zip"; '
                'blueprint_download_url "$BUNDLE_URL" "$WORK_DIR/isaac_provider_runtime_bundle.zip"; dl=$?; '
                "if [ $dl -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:download_failed:$dl; "
                'elif [ -z "$ISAAC_PY" ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:isaac_python_missing; '
                "else "
                "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED; "
                '$ISAAC_PY -m zipfile -e "$WORK_DIR/isaac_provider_runtime_bundle.zip" "$WORK_DIR/isaac_provider_bundle"; unzip_rc=$?; '
                "if [ $unzip_rc -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:unzip_failed:$unzip_rc; "
                'elif [ ! -f "$WORK_DIR/isaac_provider_bundle/provider_runtime/run_isaac_realistic_runtime.sh" ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:entrypoint_missing; '
                "else "
                'export BLUEPRINT_ISAAC_PYTHON="$ISAAC_PY"; '
                'export BLUEPRINT_ISAAC_OUTPUT_DIR="$WORK_DIR/isaac_provider_bundle/runtime_output"; '
                'export BLUEPRINT_ISAAC_EVAL_MANIFEST="$WORK_DIR/isaac_provider_bundle/provider_runtime/isaac_provider_eval_manifest.json"; '
                "echo BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED; "
                'bash "$WORK_DIR/isaac_provider_bundle/provider_runtime/run_isaac_realistic_runtime.sh"; provider_rc=$?; '
                "echo BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_EXIT_CODE:$provider_rc; "
                "$ISAAC_PY - <<'PY'\n"
                "import json\n"
                "import os\n"
                "import zipfile\n"
                "from pathlib import Path\n"
                "output_dir = Path(os.environ.get('BLUEPRINT_ISAAC_OUTPUT_DIR', '/workspace/isaac_provider_bundle/runtime_output'))\n"
                "work_dir = Path(os.environ.get('BLUEPRINT_VAST_WORK_DIR', '/tmp/blueprint_vast_work'))\n"
                "output_zip = work_dir / 'isaac_provider_runtime_output.zip'\n"
                "with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:\n"
                "    if output_dir.is_dir():\n"
                "        skipped = []\n"
                "        for path in sorted(output_dir.rglob('*')):\n"
                "            if path.is_file():\n"
                "                rel = path.relative_to(output_dir)\n"
                "                if rel.parts and rel.parts[0] in {'groot_runtime', 'hf_cache'}:\n"
                "                    skipped.append({'path': rel.as_posix(), 'reason': 'large_runtime_or_model_cache_excluded_from_provider_output_zip'})\n"
                "                    continue\n"
                "                size = path.stat().st_size\n"
                "                if size > 20_000_000:\n"
                "                    skipped.append({'path': rel.as_posix(), 'reason': 'large_file_excluded_from_provider_output_zip', 'bytes': size})\n"
                "                    continue\n"
                "                archive.write(path, rel.as_posix())\n"
                "        if skipped:\n"
                "            archive.writestr('provider_output_zip_exclusions.json', json.dumps({'schema_version': 'unitree_groot_n17_sonic_provider_output_zip_exclusions.v1', 'skipped': skipped[:5000]}, indent=2, sort_keys=True))\n"
                "    else:\n"
                "        archive.writestr('runtime_output_missing.json', json.dumps({'status': 'blocked', 'blockers': ['runtime_output_directory_missing']}, indent=2))\n"
                "print('BLUEPRINT_VAST_PROVIDER_OUTPUT_ZIP_WRITTEN:%d' % output_zip.stat().st_size)\n"
                "PY\n"
                "zip_rc=$?; "
                "if [ $zip_rc -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:output_zip_failed:$zip_rc; "
                'elif blueprint_upload_put "$OUTPUT_PUT_URL" "$WORK_DIR/isaac_provider_runtime_output.zip"; then '
                "echo BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK; cat /tmp/blueprint_provider_upload_response.json; "
                "else upload_rc=$?; echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:output_upload_failed:$upload_rc; fi; "
                "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED; "
                "fi; fi; fi; fi; "
            )
        elif provider_bundle_kind == "unitree_unifolm":
            script += (
                common_start + "RUNTIME_PY=''; "
                "if command -v apt-get >/dev/null 2>&1 && "
                "{ ! command -v python3 >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || "
                "! command -v curl >/dev/null 2>&1 || ! command -v unzip >/dev/null 2>&1; }; then "
                "apt-get update >/tmp/blueprint_vast_apt_update.log 2>&1 && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip curl unzip git >/tmp/blueprint_vast_apt_install.log 2>&1; "
                "fi; "
                "if [ -x /opt/conda/bin/python ]; then RUNTIME_PY=/opt/conda/bin/python; "
                "elif [ -x /usr/local/bin/python ]; then RUNTIME_PY=/usr/local/bin/python; "
                "elif command -v python3 >/dev/null 2>&1; then RUNTIME_PY=$(command -v python3); "
                "elif command -v python >/dev/null 2>&1; then RUNTIME_PY=$(command -v python); fi; "
                'if [ -z "$RUNTIME_PY" ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:python_missing; '
                "else "
                'rm -rf "$WORK_DIR/unitree_unifolm_provider_bundle" "$WORK_DIR/unitree_unifolm_policy_provider_runtime_bundle.zip" "$WORK_DIR/unitree_unifolm_policy_provider_runtime_output.zip"; '
                'blueprint_download_url "$BUNDLE_URL" "$WORK_DIR/unitree_unifolm_policy_provider_runtime_bundle.zip"; dl=$?; '
                "if [ $dl -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:download_failed:$dl; "
                "else "
                "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED; "
                '$RUNTIME_PY -m zipfile -e "$WORK_DIR/unitree_unifolm_policy_provider_runtime_bundle.zip" "$WORK_DIR/unitree_unifolm_provider_bundle"; unzip_rc=$?; '
                "if [ $unzip_rc -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:unzip_failed:$unzip_rc; "
                'elif [ ! -f "$WORK_DIR/unitree_unifolm_provider_bundle/provider_runtime/run_unitree_unifolm_provider_runtime.sh" ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:entrypoint_missing; '
                "else "
                'export PYTHONPATH="$WORK_DIR/unitree_unifolm_provider_bundle/provider_runtime:${PYTHONPATH:-}"; '
                'export BLUEPRINT_UNITREE_UNIFOLM_PROVIDER_OUTPUT_DIR="$WORK_DIR/unitree_unifolm_provider_bundle/runtime_output"; '
                'export BLUEPRINT_UNITREE_UNIFOLM_PROVIDER_OUTPUT="$WORK_DIR/unitree_unifolm_provider_bundle/runtime_output/unitree_unifolm_policy_provider_output.json"; '
                'export BLUEPRINT_UNITREE_UNIFOLM_POLICY_INPUT="$WORK_DIR/unitree_unifolm_provider_bundle/provider_runtime/policy_input.json"; '
                'mkdir -p "$BLUEPRINT_UNITREE_UNIFOLM_PROVIDER_OUTPUT_DIR"; '
                "echo BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED; "
                'bash "$WORK_DIR/unitree_unifolm_provider_bundle/provider_runtime/run_unitree_unifolm_provider_runtime.sh"; provider_rc=$?; '
                "echo BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_EXIT_CODE:$provider_rc; "
                "$RUNTIME_PY - <<'PY'\n"
                "import json\n"
                "import os\n"
                "import zipfile\n"
                "from pathlib import Path\n"
                "output_dir = Path(os.environ.get('BLUEPRINT_UNITREE_UNIFOLM_PROVIDER_OUTPUT_DIR', '/workspace/unitree_unifolm_provider_bundle/runtime_output'))\n"
                "work_dir = Path(os.environ.get('BLUEPRINT_VAST_WORK_DIR', '/tmp/blueprint_vast_work'))\n"
                "output_zip = work_dir / 'unitree_unifolm_policy_provider_runtime_output.zip'\n"
                "with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:\n"
                "    if output_dir.is_dir():\n"
                "        for path in sorted(output_dir.rglob('*')):\n"
                "            if path.is_file():\n"
                "                archive.write(path, path.relative_to(output_dir).as_posix())\n"
                "    else:\n"
                "        archive.writestr('runtime_output_missing.json', json.dumps({'status': 'blocked', 'blockers': ['runtime_output_directory_missing']}, indent=2))\n"
                "print('BLUEPRINT_VAST_PROVIDER_OUTPUT_ZIP_WRITTEN:%d' % output_zip.stat().st_size)\n"
                "PY\n"
                "zip_rc=$?; "
                "if [ $zip_rc -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:output_zip_failed:$zip_rc; "
                'elif blueprint_upload_put "$OUTPUT_PUT_URL" "$WORK_DIR/unitree_unifolm_policy_provider_runtime_output.zip"; then '
                "echo BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK; cat /tmp/blueprint_provider_upload_response.json; "
                "else upload_rc=$?; echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:output_upload_failed:$upload_rc; fi; "
                "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED; "
                "fi; fi; fi; fi; "
            )
        elif provider_bundle_kind == "unitree_groot_n17_sonic":
            script += (
                common_start + "RUNTIME_PY=''; "
                "if command -v apt-get >/dev/null 2>&1 && "
                "{ ! command -v python3 >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || "
                "! command -v curl >/dev/null 2>&1 || ! command -v unzip >/dev/null 2>&1; }; then "
                "apt-get update >/tmp/blueprint_vast_apt_update.log 2>&1 && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip curl unzip git >/tmp/blueprint_vast_apt_install.log 2>&1; "
                "fi; "
                "if [ -x /opt/conda/bin/python ]; then RUNTIME_PY=/opt/conda/bin/python; "
                "elif [ -x /usr/local/bin/python ]; then RUNTIME_PY=/usr/local/bin/python; "
                "elif command -v python3 >/dev/null 2>&1; then RUNTIME_PY=$(command -v python3); "
                "elif command -v python >/dev/null 2>&1; then RUNTIME_PY=$(command -v python); fi; "
                'if [ -z "$RUNTIME_PY" ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:python_missing; '
                "else "
                'rm -rf "$WORK_DIR/unitree_groot_n17_sonic_provider_bundle" "$WORK_DIR/unitree_groot_n17_sonic_policy_provider_runtime_bundle.zip" "$WORK_DIR/unitree_groot_n17_sonic_policy_provider_runtime_output.zip"; '
                'blueprint_download_url "$BUNDLE_URL" "$WORK_DIR/unitree_groot_n17_sonic_policy_provider_runtime_bundle.zip"; dl=$?; '
                "if [ $dl -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:download_failed:$dl; "
                "else "
                "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED; "
                '$RUNTIME_PY -m zipfile -e "$WORK_DIR/unitree_groot_n17_sonic_policy_provider_runtime_bundle.zip" "$WORK_DIR/unitree_groot_n17_sonic_provider_bundle"; unzip_rc=$?; '
                "if [ $unzip_rc -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:unzip_failed:$unzip_rc; "
                'elif [ ! -f "$WORK_DIR/unitree_groot_n17_sonic_provider_bundle/provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh" ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:entrypoint_missing; '
                "else "
                "echo BLUEPRINT_VAST_PROVIDER_PYTHON_DEPS_CHECK_STARTED; "
                "$RUNTIME_PY - <<'PY'\n"
                "import importlib.util\n"
                "import sys\n"
                "packages = ['numpy', 'PIL', 'zmq', 'msgpack', 'msgpack_numpy']\n"
                "missing = [name for name in packages if importlib.util.find_spec(name) is None]\n"
                "if missing:\n"
                "    print('BLUEPRINT_VAST_PROVIDER_PYTHON_DEPS_MISSING:' + ','.join(missing))\n"
                "    sys.exit(1)\n"
                "print('BLUEPRINT_VAST_PROVIDER_PYTHON_DEPS_READY')\n"
                "PY\n"
                "deps_rc=$?; "
                "if [ $deps_rc -ne 0 ]; then "
                'if [ "${BLUEPRINT_UNITREE_GROOT_N17_SONIC_AUTO_START_POLICY_SERVER:-}" = "true" ] || [ "${BLUEPRINT_UNITREE_GROOT_N17_SONIC_AUTO_START_POLICY_SERVER:-}" = "1" ]; then '
                "echo BLUEPRINT_VAST_PROVIDER_PYTHON_DEPS_DELEGATED_TO_GROOT_UV; "
                "else "
                "echo BLUEPRINT_VAST_PROVIDER_PIP_INSTALL_STARTED; "
                "$RUNTIME_PY -m pip install --quiet --only-binary=:all: --timeout 60 --retries 1 --break-system-packages numpy pillow pyzmq msgpack msgpack-numpy >/tmp/blueprint_unitree_groot_pip_install.log 2>&1; pip_rc=$?; "
                "if [ $pip_rc -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:python_dependency_install_failed:$pip_rc; "
                "else echo BLUEPRINT_VAST_PROVIDER_PIP_INSTALL_COMPLETED; fi; "
                "fi; "
                "fi; "
                'export PYTHONPATH="$WORK_DIR/unitree_groot_n17_sonic_provider_bundle/provider_runtime:${PYTHONPATH:-}"; '
                'export BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR="$WORK_DIR/unitree_groot_n17_sonic_provider_bundle/runtime_output"; '
                'export BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT="$WORK_DIR/unitree_groot_n17_sonic_provider_bundle/runtime_output/unitree_groot_n17_sonic_policy_provider_output.json"; '
                'export BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_INPUT="$WORK_DIR/unitree_groot_n17_sonic_provider_bundle/provider_runtime/policy_input.json"; '
                'mkdir -p "$BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR"; '
                "echo BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED; "
                'bash "$WORK_DIR/unitree_groot_n17_sonic_provider_bundle/provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh"; provider_rc=$?; '
                "echo BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_EXIT_CODE:$provider_rc; "
                "$RUNTIME_PY - <<'PY'\n"
                "import json\n"
                "import os\n"
                "import zipfile\n"
                "from pathlib import Path\n"
                "output_dir = Path(os.environ.get('BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR', '/workspace/unitree_groot_n17_sonic_provider_bundle/runtime_output'))\n"
                "work_dir = Path(os.environ.get('BLUEPRINT_VAST_WORK_DIR', '/tmp/blueprint_vast_work'))\n"
                "output_zip = work_dir / 'unitree_groot_n17_sonic_policy_provider_runtime_output.zip'\n"
                "with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:\n"
                "    if output_dir.is_dir():\n"
                "        skipped = []\n"
                "        for path in sorted(output_dir.rglob('*')):\n"
                "            if path.is_file():\n"
                "                rel = path.relative_to(output_dir)\n"
                "                if rel.parts and rel.parts[0] in {'groot_runtime', 'hf_cache'}:\n"
                "                    skipped.append({'path': rel.as_posix(), 'reason': 'large_runtime_or_model_cache_excluded_from_provider_output_zip'})\n"
                "                    continue\n"
                "                size = path.stat().st_size\n"
                "                if size > 20_000_000:\n"
                "                    skipped.append({'path': rel.as_posix(), 'reason': 'large_file_excluded_from_provider_output_zip', 'bytes': size})\n"
                "                    continue\n"
                "                archive.write(path, rel.as_posix())\n"
                "        if skipped:\n"
                "            archive.writestr('provider_output_zip_exclusions.json', json.dumps({'schema_version': 'unitree_groot_n17_sonic_provider_output_zip_exclusions.v1', 'skipped': skipped[:5000]}, indent=2, sort_keys=True))\n"
                "    else:\n"
                "        archive.writestr('runtime_output_missing.json', json.dumps({'status': 'blocked', 'blockers': ['runtime_output_directory_missing']}, indent=2))\n"
                "print('BLUEPRINT_VAST_PROVIDER_OUTPUT_ZIP_WRITTEN:%d' % output_zip.stat().st_size)\n"
                "PY\n"
                "zip_rc=$?; "
                "if [ $zip_rc -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:output_zip_failed:$zip_rc; "
                'elif blueprint_upload_put "$OUTPUT_PUT_URL" "$WORK_DIR/unitree_groot_n17_sonic_policy_provider_runtime_output.zip"; then '
                "echo BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK; cat /tmp/blueprint_provider_upload_response.json; "
                "else upload_rc=$?; echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:output_upload_failed:$upload_rc; fi; "
                "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED; "
                "fi; fi; fi; fi; "
            )
        else:
            script += (
                common_start + "RUNTIME_PY=''; "
                "if command -v apt-get >/dev/null 2>&1 && "
                "{ ! command -v python3 >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || "
                "! command -v curl >/dev/null 2>&1 || ! command -v unzip >/dev/null 2>&1 || "
                "! command -v ffmpeg >/dev/null 2>&1; }; then "
                "apt-get update >/tmp/blueprint_vast_apt_update.log 2>&1 && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip curl unzip ffmpeg git >/tmp/blueprint_vast_apt_install.log 2>&1; "
                "fi; "
                "if [ -x /opt/conda/bin/python ]; then RUNTIME_PY=/opt/conda/bin/python; "
                "elif [ -x /usr/local/bin/python ]; then RUNTIME_PY=/usr/local/bin/python; "
                "elif command -v python3 >/dev/null 2>&1; then RUNTIME_PY=$(command -v python3); "
                "elif command -v python >/dev/null 2>&1; then RUNTIME_PY=$(command -v python); fi; "
                'if [ -z "$RUNTIME_PY" ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:python_missing; '
                "else "
                'rm -rf "$WORK_DIR/wam_provider_bundle" "$WORK_DIR/wam_provider_runtime_bundle.zip" "$WORK_DIR/wam_provider_runtime_output.zip"; '
                'blueprint_download_url "$BUNDLE_URL" "$WORK_DIR/wam_provider_runtime_bundle.zip"; dl=$?; '
                "if [ $dl -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:download_failed:$dl; "
                "else "
                "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED; "
                '$RUNTIME_PY -m zipfile -e "$WORK_DIR/wam_provider_runtime_bundle.zip" "$WORK_DIR/wam_provider_bundle"; unzip_rc=$?; '
                "if [ $unzip_rc -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:unzip_failed:$unzip_rc; "
                'elif [ ! -f "$WORK_DIR/wam_provider_bundle/provider_runtime/run_wam_provider_runtime.sh" ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:entrypoint_missing; '
                "else "
                'export BLUEPRINT_WAM_PROVIDER_PYTHON="$RUNTIME_PY"; '
                'export BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR="$WORK_DIR/wam_provider_bundle/runtime_output"; '
                'export BLUEPRINT_WAM_PROVIDER_BUNDLE_DIR="$WORK_DIR/wam_provider_bundle"; '
                'export BLUEPRINT_WAM_ROLLOUT_INPUT="$WORK_DIR/wam_provider_bundle/provider_runtime/wam_rollout_input_manifest.json"; '
                "echo BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED; "
                'bash "$WORK_DIR/wam_provider_bundle/provider_runtime/run_wam_provider_runtime.sh"; provider_rc=$?; '
                "echo BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_EXIT_CODE:$provider_rc; "
                "$RUNTIME_PY - <<'PY'\n"
                "import json\n"
                "import os\n"
                "import zipfile\n"
                "from pathlib import Path\n"
                "output_dir = Path(os.environ.get('BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR', '/workspace/wam_provider_bundle/runtime_output'))\n"
                "work_dir = Path(os.environ.get('BLUEPRINT_VAST_WORK_DIR', '/tmp/blueprint_vast_work'))\n"
                "output_zip = work_dir / 'wam_provider_runtime_output.zip'\n"
                "with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:\n"
                "    if output_dir.is_dir():\n"
                "        for path in sorted(output_dir.rglob('*')):\n"
                "            if path.is_file():\n"
                "                archive.write(path, path.relative_to(output_dir).as_posix())\n"
                "    else:\n"
                "        archive.writestr('runtime_output_missing.json', json.dumps({'status': 'blocked', 'blockers': ['runtime_output_directory_missing']}, indent=2))\n"
                "print('BLUEPRINT_VAST_PROVIDER_OUTPUT_ZIP_WRITTEN:%d' % output_zip.stat().st_size)\n"
                "PY\n"
                "zip_rc=$?; "
                "if [ $zip_rc -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:output_zip_failed:$zip_rc; "
                'elif blueprint_upload_put "$OUTPUT_PUT_URL" "$WORK_DIR/wam_provider_runtime_output.zip"; then '
                "echo BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK; cat /tmp/blueprint_provider_upload_response.json; "
                "else upload_rc=$?; echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:output_upload_failed:$upload_rc; fi; "
                "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED; "
                "fi; fi; fi; fi; "
            )
    else:
        script += "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_SKIPPED; "
    script += "echo BLUEPRINT_VAST_ONSTART_DONE"
    return script


def _create_request_summary(
    payload: Mapping[str, Any],
    *,
    secret_values: Sequence[str],
) -> dict[str, Any]:
    env = _mapping(payload.get("env"))
    inline_payload = _string(env.get(VAST_INLINE_PROVIDER_BUNDLE_BASE64_ENV))
    return {
        "image": payload.get("image"),
        "label": payload.get("label"),
        "disk_gb": payload.get("disk"),
        "runtype": payload.get("runtype"),
        "target_state": payload.get("target_state"),
        "cancel_unavail": payload.get("cancel_unavail"),
        "template_hash_present": bool(_string(payload.get("template_hash_id"))),
        "image_override_present": bool(_string(payload.get("image"))),
        "entrypoint": payload.get("entrypoint"),
        "args_present": "args" in payload,
        "args_count": len(payload.get("args", []))
        if isinstance(payload.get("args"), list)
        else None,
        "args_str_present": "args_str" in payload,
        "args_str_length": len(str(payload.get("args_str", ""))) if "args_str" in payload else 0,
        "onstart_present": "onstart" in payload,
        "onstart_length": len(str(payload.get("onstart", ""))) if "onstart" in payload else 0,
        "env_keys": sorted(str(key) for key in env.keys()),
        "inline_provider_bundle_transport_present": bool(inline_payload),
        "inline_provider_bundle_base64_length": len(inline_payload),
        "inline_provider_bundle_sha256_present": bool(
            _string(env.get(VAST_INLINE_PROVIDER_BUNDLE_SHA256_ENV))
        ),
        "isaac_required_env_present": {
            "ACCEPT_EULA": env.get("ACCEPT_EULA") == "Y",
            "PRIVACY_CONSENT": env.get("PRIVACY_CONSENT") == "Y",
            "NVIDIA_DRIVER_CAPABILITIES": bool(env.get("NVIDIA_DRIVER_CAPABILITIES")),
        },
        "image_login_supplied": bool(payload.get("image_login")),
        "raw_payload_redacted": _redact_runtime_value(payload, secret_values),
    }


def _instance_id_from_create_response(response: Mapping[str, Any]) -> int | None:
    for key in ("new_contract", "id", "instance_id", "contract_id"):
        value = _number(response.get(key))
        if value:
            return int(value)
    nested = _mapping(response.get("instance"))
    value = _number(nested.get("id"))
    return int(value) if value else None


def _instance_status(instance_payload: Mapping[str, Any]) -> str:
    instances = instance_payload.get("instances")
    if isinstance(instances, Mapping):
        data = instances
    else:
        data = instance_payload
    return (
        _string(data.get("actual_status"))
        or _string(data.get("cur_state"))
        or _string(data.get("status"))
        or _string(data.get("intended_status"))
        or "unknown"
    )


def _instance_list_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    instances = payload.get("instances")
    if isinstance(instances, list):
        return [item for item in instances if isinstance(item, Mapping)]
    if isinstance(instances, Mapping):
        if any(isinstance(value, Mapping) for value in instances.values()):
            return [value for value in instances.values() if isinstance(value, Mapping)]
        return [instances]
    for key in ("results", "data", "response"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
        if isinstance(value, Mapping):
            return [item for item in value.values() if isinstance(item, Mapping)]
    if any(key in payload for key in ("actual_status", "cur_state", "status", "intended_status")):
        return [payload]
    return []


def _sanitized_instance_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id") or row.get("instance_id") or row.get("contract_id"),
        "machine_id": row.get("machine_id"),
        "has_avx": _normalized_binary_capability(row.get("has_avx")),
        "gpu_name": row.get("gpu_name") or row.get("gpu_display_name") or row.get("gpu_names"),
        "actual_status": row.get("actual_status"),
        "cur_state": row.get("cur_state"),
        "status": row.get("status"),
        "intended_status": row.get("intended_status"),
        "dph_total": row.get("dph_total") or row.get("min_bid") or row.get("price_per_hour"),
        "raw_status_normalized": _instance_status(row).lower(),
    }


def _active_instance_rows_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    active_rows: list[dict[str, Any]] = []
    for row in _instance_list_rows(payload):
        sanitized = _sanitized_instance_row(row)
        status = _string(sanitized.get("raw_status_normalized")).lower()
        if status and status not in set(VAST_TERMINAL_INSTANCE_STATUSES):
            active_rows.append(sanitized)
    return active_rows


def _prelaunch_inventory_guard(
    *,
    job_dir: Path,
    generated_at: str,
    api_key: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    active_instances: list[dict[str, Any]] = []
    status_code: int | None = None
    query_error: str | None = None
    query_attempt_count = 0
    for attempt in range(1, 4):
        query_attempt_count = attempt
        try:
            status_code, payload = _api_json(
                method="GET",
                path="/instances/",
                api_key=api_key,
                timeout_seconds=30,
            )
            active_instances = _active_instance_rows_from_payload(payload)
            query_error = None
            break
        except urllib.error.HTTPError as exc:
            status_code = exc.code
            query_error = f"{type(exc).__name__}:{str(exc)[:300]}"
            if exc.code == 429 and attempt < 3:
                time.sleep(min(8, 2 * attempt))
                continue
            blockers.append("vast_prelaunch_inventory_query_failed")
            break
        except Exception as exc:
            query_error = f"{type(exc).__name__}:{str(exc)[:300]}"
            blockers.append("vast_prelaunch_inventory_query_failed")
            break
    if active_instances:
        blockers.append("active_vast_instances_detected_before_new_launch")
    manifest = {
        "schema_version": "vast_prelaunch_inventory_guard.v1",
        "generated_at": generated_at,
        "status": "passed" if not blockers else "blocked",
        "api_endpoint_checked": "GET /api/v0/instances/",
        "api_http_status_code": status_code,
        "active_instance_count": len(active_instances),
        "active_instances": active_instances,
        "continuing_spend_detected_before_new_launch": bool(active_instances),
        "query_error": query_error,
        "query_attempt_count": query_attempt_count,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "vast_prelaunch_inventory_guard.json", manifest)
    return manifest


def _poll_instance(
    *,
    instance_id: int,
    api_key: str,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    observations: list[dict[str, Any]] = []
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status_code, payload = _api_json(
            method="GET",
            path=f"/instances/{instance_id}/",
            api_key=api_key,
            timeout_seconds=30,
        )
        last_payload = payload
        status = _instance_status(payload)
        observations.append(
            {
                "observed_at": utc_now_iso(),
                "http_status_code": status_code,
                "status": status,
                "actual_status": _mapping(payload.get("instances")).get("actual_status")
                if isinstance(payload.get("instances"), Mapping)
                else payload.get("actual_status"),
                "cur_state": _mapping(payload.get("instances")).get("cur_state")
                if isinstance(payload.get("instances"), Mapping)
                else payload.get("cur_state"),
            }
        )
        if status.lower() in {"running", "exited", "stopped", "failed"}:
            return status, observations, last_payload
        time.sleep(poll_interval_seconds)
    return _instance_status(last_payload), observations, last_payload


def _execute_and_fetch(
    *,
    instance_id: int,
    api_key: str,
    command: str,
    output_log_path: Path,
    secret_values: Sequence[str],
    wait_seconds: int = 8,
) -> dict[str, Any]:
    response: dict[str, Any] = {}
    status_code = 0
    result_url = ""
    output_text = ""
    fetch_error = None
    api_request_error = None
    try:
        status_code, response = _api_json(
            method="PUT",
            path=f"/instances/command/{instance_id}/",
            api_key=api_key,
            payload={"command": command},
            timeout_seconds=30,
        )
        result_url = _string(response.get("result_url"))
    except Exception as exc:  # pragma: no cover - live network dependent.
        fetch_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        api_request_error = fetch_error
    if result_url:
        time.sleep(wait_seconds)
        try:
            output_text = _fetch_text(result_url, timeout_seconds=30)
        except Exception as exc:  # pragma: no cover - live network dependent.
            fetch_error = f"{type(exc).__name__}: {str(exc)[:300]}"
    ensure_dir(output_log_path.parent)
    output_log_path.write_text(_redact_text(output_text, secret_values), encoding="utf-8")
    return {
        "http_status_code": status_code,
        "response": _redact_runtime_value(response, secret_values),
        "result_url_present": bool(result_url),
        "result_url": _redact_result_url(result_url),
        "output_log_path": str(output_log_path),
        "output_fetch_error": fetch_error,
        "api_request_error": api_request_error,
        "output_size_bytes": output_log_path.stat().st_size,
    }


def _request_logs_and_fetch(
    *,
    instance_id: int,
    api_key: str,
    output_log_path: Path,
    secret_values: Sequence[str],
    wait_seconds: int = 20,
    tail_lines: int = 1000,
    max_wait_seconds: int = 420,
    retry_interval_seconds: int = 30,
    success_markers: Sequence[str] = (),
    container_missing_retry_attempts: int = 5,
    no_progress_seconds: int | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0, max_wait_seconds)
    attempts: list[dict[str, Any]] = []
    output_text = ""
    response: dict[str, Any] = {}
    status_code = 0
    result_url = ""
    fetch_error = None
    time.sleep(max(0, wait_seconds))
    no_progress_limit_seconds = max(
        0,
        int(
            no_progress_seconds
            if no_progress_seconds is not None
            else _env_int(VAST_WAM_NO_PROGRESS_SECONDS_ENV, 900)
        ),
    )
    last_progress_monotonic = time.monotonic()
    previous_output_text: str | None = None
    previous_runtime_phase_count = 0
    no_progress_timeout_reached = False
    break_reason = ""
    attempt_index = 0
    container_missing_count = 0
    while True:
        attempt_index += 1
        attempt_started = time.monotonic()
        api_request_error = None
        fetch_error = None
        try:
            status_code, response = _api_json(
                method="PUT",
                path=f"/instances/request_logs/{instance_id}",
                api_key=api_key,
                payload={"tail": str(tail_lines), "daemon_logs": "false"},
                timeout_seconds=30,
            )
            result_url = _string(response.get("result_url") or response.get("temp_download_url"))
        except Exception as exc:  # pragma: no cover - live network dependent.
            response = {}
            result_url = ""
            fetch_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            api_request_error = fetch_error
        attempt_text = ""
        if result_url:
            time.sleep(5)
            try:
                attempt_text = _fetch_text(result_url, timeout_seconds=30)
            except Exception as exc:  # pragma: no cover - live network dependent.
                fetch_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        output_text = attempt_text
        marker_found = any(marker and marker in attempt_text for marker in success_markers)
        container_missing = "No such container" in attempt_text
        runtime_phase_count = attempt_text.count("BLUEPRINT_WAM_RUNTIME_PHASE:")
        output_changed = previous_output_text is None or attempt_text != previous_output_text
        runtime_phase_progress = runtime_phase_count > previous_runtime_phase_count
        # A container that never materializes often flickers between empty logs and a Docker
        # "No such container" / daemon error. That text changing between polls must NOT count
        # as progress, or it keeps the no-progress watchdog alive for the entire live window on
        # a dud offer (observed: a dud idled the full deadline instead of bailing). Genuine
        # worker phase markers always count as progress regardless of any error text.
        container_or_daemon_error_only = container_missing or (
            "Error response from daemon" in attempt_text
        )
        progress_observed = bool(attempt_text.strip()) and (
            runtime_phase_progress or (output_changed and not container_or_daemon_error_only)
        )
        if progress_observed:
            last_progress_monotonic = time.monotonic()
        previous_output_text = attempt_text
        previous_runtime_phase_count = max(previous_runtime_phase_count, runtime_phase_count)
        no_progress_elapsed_seconds = max(0.0, time.monotonic() - last_progress_monotonic)
        no_progress_timeout_reached = bool(
            no_progress_limit_seconds and no_progress_elapsed_seconds >= no_progress_limit_seconds
        )
        if container_missing:
            container_missing_count += 1
        attempts.append(
            {
                "attempt": attempt_index,
                "observed_at": utc_now_iso(),
                "attempt_elapsed_seconds": round(time.monotonic() - attempt_started, 6),
                "http_status_code": status_code,
                "result_url_present": bool(result_url),
                "output_size_bytes": len(attempt_text.encode("utf-8")),
                "output_changed": output_changed,
                "runtime_phase_marker_count": runtime_phase_count,
                "runtime_phase_progress_observed": runtime_phase_progress,
                "progress_observed": progress_observed,
                "no_progress_elapsed_seconds": round(no_progress_elapsed_seconds, 6),
                "no_progress_timeout_reached": no_progress_timeout_reached,
                "fetch_error": fetch_error,
                "api_request_error": api_request_error,
                "success_marker_found": marker_found,
                "container_missing_marker_observed": container_missing,
                "container_missing_observed_count": container_missing_count,
            }
        )
        terminal_container_missing = container_missing and container_missing_count >= max(
            1, int(container_missing_retry_attempts)
        )
        deadline_reached = time.monotonic() >= deadline
        if marker_found:
            break_reason = "success_marker_found"
        elif terminal_container_missing:
            break_reason = "terminal_container_missing"
        elif no_progress_timeout_reached:
            break_reason = "no_log_progress_timeout"
        elif deadline_reached:
            break_reason = "max_wait_deadline_reached"
        if break_reason:
            break
        time.sleep(max(1, retry_interval_seconds))
    ensure_dir(output_log_path.parent)
    output_log_path.write_text(_redact_text(output_text, secret_values), encoding="utf-8")
    return {
        "http_status_code": status_code,
        "response": _redact_runtime_value(response, secret_values),
        "result_url_present": bool(result_url),
        "result_url": _redact_result_url(result_url),
        "output_log_path": str(output_log_path),
        "output_fetch_error": fetch_error,
        "output_size_bytes": output_log_path.stat().st_size,
        "log_poll_attempts": attempts,
        "no_progress_timeout_seconds": no_progress_limit_seconds,
        "no_progress_timeout_reached": no_progress_timeout_reached,
        "break_reason": break_reason,
    }


def _log_result_has_container_missing(log_result: Mapping[str, Any]) -> bool:
    attempts = log_result.get("log_poll_attempts")
    if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
        return False
    return any(
        isinstance(item, Mapping) and bool(item.get("container_missing_marker_observed"))
        for item in attempts
    )


def _log_text_has_success_marker(text: str, markers: Sequence[str]) -> bool:
    return any(marker and marker in text for marker in markers)


def _ffprobe_video(path: Path) -> dict[str, Any]:
    """Compatibility wrapper for the provider-neutral video probe."""

    return probe_mp4_video(path)


def _inspect_provider_runtime_output_zip(
    path: Path | None,
    *,
    video_extract_dir: Path | None = None,
    expected_video_count: int | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper retaining Vast's monkeypatch surface."""

    return inspect_provider_runtime_output_zip(
        path,
        video_extract_dir=video_extract_dir,
        expected_video_count=expected_video_count,
        video_probe=_ffprobe_video,
    )


def _runtime_result_artifact_summary(
    runtime_result: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Compatibility wrapper for provider-neutral runtime-result summaries."""

    return summarize_runtime_result(runtime_result)


def _write_blocked_phase_artifacts(
    *,
    job_dir: Path,
    generated_at: str,
    heartbeat_reason: str | None = None,
    gpu_reason: str | None = None,
    isaac_reason: str | None = None,
    provider_reason: str | None = None,
) -> None:
    if not (job_dir / "vast_startup_probe_manifest.json").exists():
        write_json(
            job_dir / "vast_startup_probe_manifest.json",
            {
                "schema_version": VAST_STARTUP_PROBE_SCHEMA_VERSION,
                "generated_at": generated_at,
                "status": "blocked",
                "blockers": [heartbeat_reason or "vast_heartbeat_not_run"],
                "heartbeat_completed": False,
                "startup_probe_proven": False,
                **_truth_boundaries(),
            },
        )
    video_smoke_path = job_dir / "vast_video_smoke_result.json"
    write_blocked_video_smoke = True
    if video_smoke_path.exists():
        try:
            existing_video_smoke = _mapping(
                json.loads(video_smoke_path.read_text(encoding="utf-8"))
            )
        except Exception:
            existing_video_smoke = {}
        write_blocked_video_smoke = (
            existing_video_smoke.get("status") != "completed"
            and existing_video_smoke.get("generated_at") != generated_at
        )
    if write_blocked_video_smoke:
        write_json(
            video_smoke_path,
            {
                "schema_version": VAST_VIDEO_SMOKE_SCHEMA_VERSION,
                "generated_at": generated_at,
                "status": "blocked",
                "blockers": [provider_reason or "vast_blueprint_video_smoke_not_run"],
                "video_smoke_proven": False,
                "expected_video_count": DEFAULT_VIDEO_SMOKE_CAMERA_COUNT,
                "mp4_count": 0,
                "proof_boundary": (
                    "Video smoke requires returned MP4 files plus ffprobe duration/frame "
                    "validation. Provider startup or output zip upload alone is not video proof."
                ),
                **_truth_boundaries(),
            },
        )
    if not (job_dir / "vast_gpu_sanity_report.json").exists():
        write_json(
            job_dir / "vast_gpu_sanity_report.json",
            {
                "schema_version": VAST_GPU_SANITY_SCHEMA_VERSION,
                "generated_at": generated_at,
                "status": "blocked",
                "blockers": [gpu_reason or "vast_gpu_sanity_not_run"],
                "nvidia_smi_visible": False,
                "gpu_sanity_proven": False,
                **_truth_boundaries(),
            },
        )
    if not (job_dir / "vast_isaac_smoke_result.json").exists():
        write_json(
            job_dir / "vast_isaac_smoke_result.json",
            {
                "schema_version": VAST_ISAAC_SMOKE_SCHEMA_VERSION,
                "generated_at": generated_at,
                "status": "blocked",
                "blockers": [isaac_reason or "vast_isaac_smoke_not_run"],
                "isaac_simulation_app_started": False,
                "isaac_smoke_proven": False,
                **_truth_boundaries(),
            },
        )
    if not (job_dir / "vast_provider_command_result.json").exists():
        write_json(
            job_dir / "vast_provider_command_result.json",
            {
                "schema_version": VAST_PROVIDER_COMMAND_SCHEMA_VERSION,
                "generated_at": generated_at,
                "status": "blocked",
                "blockers": [provider_reason or "vast_blueprint_provider_bundle_not_run"],
                "provider_runtime_output_zip_produced": False,
                "provider_command_path_remote_proven": False,
                **_truth_boundaries(),
            },
        )


def _fill_missing_phase_rows(job_dir: Path, *, reason: str) -> None:
    phase_path = job_dir / "vast_runtime_phase_log.jsonl"
    seen: set[str] = set()
    if phase_path.exists():
        for line in phase_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen.add(str(row.get("phase")))
    for phase in VAST_REQUIRED_PHASES:
        if phase not in seen:
            _append_phase(job_dir, phase, "blocked", blockers=[reason])


def _append_preallocation_blocked_phases(
    job_dir: Path,
    *,
    blockers: Sequence[str],
    start_index: int = 2,
) -> None:
    for phase in VAST_REQUIRED_PHASES[start_index:]:
        if phase == "vast_artifacts_exported":
            _append_phase(
                job_dir,
                phase,
                "completed",
                proof_effect="blocked_preallocation_artifacts_written",
            )
        elif phase in {"vast_instance_teardown_started", "vast_instance_teardown_completed"}:
            _append_phase(
                job_dir,
                phase,
                "completed",
                proof_effect="no_vast_instance_created_no_teardown_required",
                instance_ids=[],
            )
        else:
            _append_phase(job_dir, phase, "blocked", blockers=blockers)


def _ensure_offer_manifest(job_dir: Path, *, generated_at: str, blockers: Sequence[str]) -> None:
    path = job_dir / "vast_offer_selection_manifest.json"
    if path.exists():
        return
    write_json(
        path,
        {
            "schema_version": VAST_OFFER_SELECTION_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "offer_search_performed": False,
            "selected_offer": None,
            "blockers": list(blockers),
            "raw_secret_values_recorded": False,
        },
    )


def _final_validation(
    *,
    job_dir: Path,
    generated_at: str,
    instance_ids: Sequence[int],
    continuing_spend: bool,
    estimated_cost_usd: float,
    hard_cap_usd: float,
) -> dict[str, Any]:
    required = [
        "vast_runtime_discovery.json",
        "vast_provider_plan.json",
        "vast_offer_selection_manifest.json",
        "vast_budget_ledger.json",
        "vast_runtime_phase_log.jsonl",
        "vast_startup_probe_manifest.json",
        "vast_gpu_sanity_report.json",
        "vast_isaac_smoke_result.json",
        "vast_provider_command_result.json",
        "vast_video_smoke_result.json",
        "vast_teardown_manifest.json",
    ]
    missing = [name for name in required if not (job_dir / name).exists()]
    json_errors: list[str] = []
    for path in job_dir.glob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            json_errors.append(f"{path.name}:{type(exc).__name__}")
    phase_lines = []
    phase_path = job_dir / "vast_runtime_phase_log.jsonl"
    if phase_path.exists():
        for line_number, line in enumerate(
            phase_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except Exception as exc:
                json_errors.append(f"{phase_path.name}:{line_number}:{type(exc).__name__}")
                continue
            if isinstance(parsed, Mapping):
                phase_lines.append(dict(parsed))
    phases = {str(row.get("phase")) for row in phase_lines}
    missing_phases = [phase for phase in VAST_REQUIRED_PHASES if phase not in phases]
    blockers: list[str] = []
    if missing:
        blockers.append("missing_required_vast_artifacts")
    if json_errors:
        blockers.append("json_parse_errors")
    if missing_phases:
        blockers.append("missing_required_vast_runtime_phases")
    if continuing_spend:
        blockers.append("continuing_vast_spend_detected")
    if estimated_cost_usd > hard_cap_usd:
        blockers.append("vast_estimated_spend_exceeded_hard_cap")
    video_smoke: dict[str, Any] = {}
    video_path = job_dir / "vast_video_smoke_result.json"
    if video_path.is_file():
        try:
            video_smoke = _mapping(json.loads(video_path.read_text(encoding="utf-8")))
        except Exception:
            video_smoke = {}
    validation = {
        "schema_version": VAST_FINAL_VALIDATION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed" if not blockers else "blocked",
        "job_dir": str(job_dir),
        "required_artifacts": required,
        "missing_required_artifacts": missing,
        "json_parse_errors": json_errors,
        "required_phases": list(VAST_REQUIRED_PHASES),
        "missing_required_phases": missing_phases,
        "vast_instance_ids": list(instance_ids),
        "estimated_cost_usd": estimated_cost_usd,
        "spend_hard_cap_usd": hard_cap_usd,
        "continuing_spend_from_this_run": continuing_spend,
        "all_vast_instances_destroyed_by_adapter": not continuing_spend,
        "video_smoke_proven": video_smoke.get("video_smoke_proven") is True,
        "video_smoke_status": video_smoke.get("status"),
        "raw_secret_values_recorded": False,
        "blockers": blockers,
        **_truth_boundaries(),
    }
    write_json(job_dir / "vast_final_validation.json", validation)
    return validation


def _api_gate_blockers(
    *, allow_vast_api_call: bool, allow_instance_launch: bool, api_key: str
) -> list[str]:
    blockers: list[str] = []
    if not _env_truthy(VAST_API_GATE_ENV):
        blockers.append(f"missing_env_{VAST_API_GATE_ENV}")
    if not _env_truthy(VAST_INSTANCE_LAUNCH_GATE_ENV):
        blockers.append(f"missing_env_{VAST_INSTANCE_LAUNCH_GATE_ENV}")
    if not allow_vast_api_call:
        blockers.append("missing_cli_allow_vast_api_call")
    if not allow_instance_launch:
        blockers.append("missing_cli_allow_vast_instance_launch")
    if not api_key:
        blockers.append(f"missing_file_based_secret_{VAST_API_KEY_FILE_ENV}")
    return blockers


def _vast_launch_lock_path() -> Path:
    configured = _string(os.environ.get(VAST_LAUNCH_LOCK_FILE_ENV))
    if configured:
        return Path(configured).expanduser().resolve()
    api_key_path = Path(
        os.environ.get(VAST_API_KEY_FILE_ENV, DEFAULT_VAST_API_KEY_FILE)
    ).expanduser()
    return (api_key_path.parent / DEFAULT_VAST_LAUNCH_LOCK_FILENAME).resolve()


def _try_acquire_vast_launch_lock(
    *,
    job_dir: Path,
    generated_at: str,
    lock_path: Path | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    lock_path = lock_path or _vast_launch_lock_path()
    ensure_dir(lock_path.parent)
    handle = lock_path.open("a+", encoding="utf-8")
    lock_path.chmod(0o600)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        existing_holder = handle.read()[:1000]
        handle.close()
        manifest = {
            "schema_version": "vast_launch_lock_manifest.v1",
            "generated_at": generated_at,
            "status": "blocked",
            "lock_path": str(lock_path),
            "lock_acquired": False,
            "blockers": ["vast_paid_launch_lock_busy"],
            "existing_lock_record_prefix": existing_holder,
            "raw_secret_values_recorded": False,
        }
        write_json(job_dir / "vast_launch_lock_manifest.json", manifest)
        return None, manifest
    record = {
        "pid": os.getpid(),
        "job_dir": str(job_dir),
        "acquired_at": generated_at,
        "purpose": "vast_paid_instance_launch_single_flight_guard",
    }
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    manifest = {
        "schema_version": "vast_launch_lock_manifest.v1",
        "generated_at": generated_at,
        "status": "acquired",
        "lock_path": str(lock_path),
        "lock_acquired": True,
        "lock_record": record,
        "blockers": [],
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "vast_launch_lock_manifest.json", manifest)
    return handle, manifest


def _release_vast_launch_lock(
    handle: Any | None,
    *,
    job_dir: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any] | None:
    if handle is None:
        return None
    lock_path = Path(handle.name).expanduser()
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
    manifest = {
        "schema_version": "vast_launch_lock_manifest.v1",
        "generated_at": generated_at or utc_now_iso(),
        "status": "released",
        "lock_path": str(lock_path),
        "lock_released": True,
        "raw_secret_values_recorded": False,
    }
    if job_dir is not None:
        write_json(job_dir / "vast_launch_lock_manifest.json", manifest)
    return manifest


def run_vast_provider_adapter(
    *,
    job_dir: str | Path,
    mode: str = "dry-run",
    allow_vast_api_call: bool = False,
    allow_instance_launch: bool = False,
    max_hourly_rate: float = DEFAULT_MAX_HOURLY_RATE,
    target_spend_usd: float = DEFAULT_TARGET_SPEND_USD,
    hard_cap_usd: float = DEFAULT_HARD_CAP_USD,
    max_live_minutes: int = DEFAULT_MAX_LIVE_MINUTES,
    public_image: str = DEFAULT_PUBLIC_CUDA_IMAGE,
    isaac_image: str = DEFAULT_ISAAC_IMAGE,
    heartbeat_url: str = DEFAULT_HEARTBEAT_URL,
    previous_job_dir: str | Path | None = None,
    provider_bundle: str | Path | None = None,
    provider_bundle_url: str | None = None,
    provider_output_put_url: str | None = None,
    provider_output_get_url: str | None = None,
    provider_runtime_output_zip: str | Path | None = None,
    enable_isaac_smoke: bool = False,
    enable_blueprint_bundle: bool = False,
    provider_bundle_kind: str = "isaac",
    vast_launch_mode: str = DEFAULT_VAST_LAUNCH_MODE,
    ngc_image_login_mode: str | None = None,
    vast_template_hash_id: str | None = None,
    use_vast_template_image: bool = False,
    allow_cold_isaac_image_pull: bool = True,
    min_cold_isaac_pull_live_minutes: int = DEFAULT_MIN_COLD_ISAAC_PULL_LIVE_MINUTES,
    disk_gb: int | None = None,
    min_gpu_ram_mb: int | None = None,
    min_compute_cap: int | None = None,
    max_compute_cap: int | None = None,
    poll_interval_seconds: int = 10,
    startup_timeout_seconds: int = 420,
    heartbeat_no_progress_seconds: int | None = None,
    machine_avoidlist_path: str | Path | None = None,
    allowed_machine_ids: Iterable[Any] = (),
    session_budget_ledger_path: str | Path | None = None,
    session_max_live_minutes: int | None = DEFAULT_SESSION_MAX_LIVE_MINUTES,
    verify_staging_urls: bool = False,
    allow_staging_output_put_probe: bool = False,
    require_known_supported_isaac_driver: bool = False,
    min_reliability: float | None = None,
    require_direct_port: bool | None = None,
    preferred_gpu_keywords: Sequence[str] = (),
    preferred_geolocation_regex: str = "",
    prefer_isaac_rt: bool | None = None,
    gpu_selection_policy: str | None = None,
    vast_launch_lock_file: str | Path | None = None,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None = None,
) -> dict[str, Any]:
    if provider_bundle_kind not in VAST_PROVIDER_BUNDLE_KINDS:
        raise ValueError(f"unsupported_provider_bundle_kind:{provider_bundle_kind}")
    resolved_image_login_mode = (
        _string(ngc_image_login_mode)
        or _string(os.getenv(VAST_IMAGE_LOGIN_MODE_ENV))
        or DEFAULT_NGC_IMAGE_LOGIN_MODE
    )
    if resolved_image_login_mode not in NGC_IMAGE_LOGIN_MODES:
        raise ValueError(f"unsupported_ngc_image_login_mode:{resolved_image_login_mode}")
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    ensure_dir(resolved_job_dir)
    generated_at = utc_now_iso()
    resolved_machine_avoidlist_path = (
        Path(machine_avoidlist_path).expanduser().resolve()
        if machine_avoidlist_path
        else resolved_job_dir / "vast_machine_avoidlist.json"
    )
    resolved_session_budget_ledger_path = (
        Path(session_budget_ledger_path).expanduser().resolve()
        if session_budget_ledger_path
        else _vast_session_budget_ledger_path()
    )
    resolved_vast_launch_lock_path = (
        Path(vast_launch_lock_file).expanduser().resolve()
        if vast_launch_lock_file
        else _vast_launch_lock_path()
    )
    resolved_min_gpu_ram_mb = max(
        0,
        int(
            _number(min_gpu_ram_mb)
            if min_gpu_ram_mb is not None
            else (_number(os.getenv(VAST_WAM_MIN_GPU_RAM_MB_ENV)) or 0)
        ),
    )
    resolved_min_compute_cap = max(
        0,
        int(
            _number(min_compute_cap)
            if min_compute_cap is not None
            else (_number(os.getenv(VAST_MIN_COMPUTE_CAP_ENV)) or 0)
        ),
    )
    resolved_max_compute_cap = vcc.resolve_max_compute_cap(max_compute_cap)
    resolved_min_reliability = max(
        0.0,
        float(
            _number(min_reliability)
            if min_reliability is not None
            else _env_float(VAST_MIN_RELIABILITY_ENV, 0.0)
        ),
    )
    resolved_require_direct_port = bool(
        require_direct_port
        if require_direct_port is not None
        else _env_truthy(VAST_REQUIRE_DIRECT_PORT_ENV)
    )
    resolved_preferred_gpu_keywords = [
        _string(item) for item in preferred_gpu_keywords if _string(item)
    ] or _env_csv(VAST_PREFERRED_GPU_KEYWORDS_ENV)
    resolved_preferred_geolocation_regex = _string(
        preferred_geolocation_regex or os.getenv(VAST_PREFERRED_GEOLOCATION_REGEX_ENV)
    )
    resolved_prefer_isaac_rt = (
        provider_bundle_kind == "isaac" if prefer_isaac_rt is None else bool(prefer_isaac_rt)
    )
    avoidlist = _load_machine_avoidlist(resolved_machine_avoidlist_path)
    excluded_machine_ids = _avoidlist_machine_ids(resolved_machine_avoidlist_path)
    resolved_allowed_machine_ids = _machine_id_set(allowed_machine_ids)
    launch_mode = _resolve_launch_mode(
        requested=vast_launch_mode,
        enable_isaac_smoke=enable_isaac_smoke,
        enable_blueprint_bundle=enable_blueprint_bundle,
        provider_bundle_kind=provider_bundle_kind,
    )
    resolved_disk_gb = _resolve_disk_gb(
        requested=disk_gb,
        enable_isaac_smoke=enable_isaac_smoke,
    )
    selected_container_image = _resolve_probe_image(
        public_image=public_image,
        isaac_image=isaac_image,
        enable_isaac_smoke=enable_isaac_smoke,
        enable_blueprint_bundle=enable_blueprint_bundle,
        provider_bundle_kind=provider_bundle_kind,
    )
    template_hash = _string(vast_template_hash_id)
    create_request_image = None if use_vast_template_image else selected_container_image
    result_path = resolved_job_dir / "vast_provider_adapter_result.json"
    phase_path = resolved_job_dir / "vast_runtime_phase_log.jsonl"
    if mode == "live-startup-probe":
        _preserve_existing_live_attempt_artifacts(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            reason="preserve_existing_live_attempt_before_new_vast_run",
        )
    if phase_path.exists():
        phase_path.unlink()
    if mode == "live-startup-probe":
        for stale_name in (
            "vast_offer_selection_manifest.json",
            "vast_budget_ledger.json",
            "vast_startup_probe_manifest.json",
            "vast_gpu_sanity_report.json",
            "vast_isaac_smoke_result.json",
            "vast_provider_command_result.json",
            "vast_teardown_manifest.json",
            "vast_final_validation.json",
            "vast_provider_adapter_result.json",
            "vast_session_budget_guard.json",
            "vast_blueprint_bundle_preflight.json",
            "vast_launch_lock_manifest.json",
            "vast_prelaunch_inventory_guard.json",
        ):
            stale_path = resolved_job_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
    instance_ids: list[int] = []
    selected_offer: dict[str, Any] | None = None
    started_at_monotonic: float | None = None
    api_key, vast_secret_status = _read_secret_file(
        VAST_API_KEY_FILE_ENV, DEFAULT_VAST_API_KEY_FILE
    )
    ngc_key, ngc_secret_status = _read_secret_file("NGC_API_KEY_FILE", DEFAULT_NGC_API_KEY_FILE)
    docker_username, docker_username_status = _read_secret_file(
        DOCKER_USERNAME_FILE_ENV, DEFAULT_DOCKER_USERNAME_FILE
    )
    docker_pat, docker_pat_status = _read_secret_file(DOCKER_PAT_FILE_ENV, DEFAULT_DOCKER_PAT_FILE)
    hf_token, hf_token_status = _read_hf_token_file()
    previous_path = Path(previous_job_dir).expanduser().resolve() if previous_job_dir else None
    bundle_path = Path(provider_bundle).expanduser().resolve() if provider_bundle else None
    output_zip_path = (
        Path(provider_runtime_output_zip).expanduser().resolve()
        if provider_runtime_output_zip
        else resolved_job_dir / "vast_provider_runtime_output.zip"
    )
    inline_bundle_transport = _inline_provider_bundle_payload(
        bundle_path,
        provider_bundle_kind=provider_bundle_kind,
        enable_blueprint_bundle=enable_blueprint_bundle,
    )
    if (
        provider_bundle_kind in {"wam", "unitree_unifolm", "unitree_groot_n17_sonic"}
        and _string(provider_bundle_url)
        and inline_bundle_transport.get("inline_provider_bundle_transport_used") is True
    ):
        inline_bundle_transport = {
            "inline_provider_bundle_transport_used": False,
            "inline_provider_bundle_transport_reason": "disabled_for_vast_env_size_with_fetch_url",
            "inline_provider_bundle_size_bytes": inline_bundle_transport.get(
                "inline_provider_bundle_size_bytes"
            ),
            "inline_provider_bundle_base64_length": inline_bundle_transport.get(
                "inline_provider_bundle_base64_length"
            ),
            "inline_provider_bundle_max_raw_bytes": VAST_INLINE_PROVIDER_BUNDLE_MAX_RAW_BYTES,
            "inline_provider_bundle_max_base64_chars": VAST_INLINE_PROVIDER_BUNDLE_MAX_BASE64_CHARS,
        }

    _runtime_discovery(
        resolved_job_dir,
        generated_at=generated_at,
        launch_mode=launch_mode,
        disk_gb=resolved_disk_gb,
    )
    _append_phase(
        resolved_job_dir,
        "vast_docs_checked",
        "completed",
        proof_effect="vast_api_docs_and_launch_modes_recorded",
    )
    _append_phase(
        resolved_job_dir,
        "vast_secret_file_verified",
        "completed"
        if vast_secret_status["present"] and vast_secret_status.get("mode_is_0600")
        else "blocked",
        blockers=[]
        if vast_secret_status["present"]
        else [f"missing_file_based_secret_{VAST_API_KEY_FILE_ENV}"],
        proof_effect="secret_file_metadata_only",
    )
    _provider_plan(
        job_dir=resolved_job_dir,
        generated_at=generated_at,
        max_hourly_rate=max_hourly_rate,
        target_spend_usd=target_spend_usd,
        hard_cap_usd=hard_cap_usd,
        max_live_minutes=max_live_minutes,
        public_image=public_image,
        isaac_image=isaac_image,
        selected_container_image=selected_container_image,
        previous_job_dir=previous_path,
        provider_bundle=bundle_path,
        provider_bundle_kind=provider_bundle_kind,
        enable_isaac_smoke=enable_isaac_smoke,
        enable_blueprint_bundle=enable_blueprint_bundle,
        launch_mode=launch_mode,
        disk_gb=resolved_disk_gb,
        ngc_image_login_mode=resolved_image_login_mode,
        vast_template_hash_id=template_hash,
        use_vast_template_image=use_vast_template_image,
        allow_cold_isaac_image_pull=allow_cold_isaac_image_pull,
        min_cold_isaac_pull_live_minutes=min_cold_isaac_pull_live_minutes,
        provider_bundle_url=provider_bundle_url,
        provider_output_put_url=provider_output_put_url,
        provider_bundle_inline_transport=inline_bundle_transport,
        require_known_supported_isaac_driver=require_known_supported_isaac_driver,
    )
    _budget_ledger(
        job_dir=resolved_job_dir,
        generated_at=generated_at,
        target_spend_usd=target_spend_usd,
        hard_cap_usd=hard_cap_usd,
        max_hourly_rate=max_hourly_rate,
        max_live_minutes=max_live_minutes,
        selected_offer=None,
    )

    base_result: dict[str, Any] = {
        "schema_version": VAST_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_dir": str(resolved_job_dir),
        "mode": mode,
        "api_call_performed": False,
        "vast_side_effects_may_have_occurred": False,
        "vast_instance_ids": instance_ids,
        "vast_launch_mode": launch_mode,
        "provider_bundle_kind": provider_bundle_kind,
        "ngc_image_login_mode": resolved_image_login_mode,
        "vast_template_hash_present": bool(template_hash),
        "use_vast_template_image": use_vast_template_image,
        "allow_cold_isaac_image_pull": allow_cold_isaac_image_pull,
        "min_cold_isaac_pull_live_minutes": min_cold_isaac_pull_live_minutes,
        "disk_gb": resolved_disk_gb,
        "public_image": public_image,
        "isaac_image": isaac_image,
        "selected_container_image": selected_container_image,
        "require_known_supported_isaac_driver": require_known_supported_isaac_driver,
        "provider_bundle_inline_transport_used": (
            inline_bundle_transport.get("inline_provider_bundle_transport_used") is True
        ),
        "provider_bundle_inline_transport_reason": _string(
            inline_bundle_transport.get("inline_provider_bundle_transport_reason")
        ),
        "provider_bundle_inline_transport_size_bytes": int(
            _number(inline_bundle_transport.get("inline_provider_bundle_size_bytes")) or 0
        ),
        "provider_bundle_inline_transport_base64_length": int(
            _number(inline_bundle_transport.get("inline_provider_bundle_base64_length")) or 0
        ),
        "provider_bundle_inline_transport_sha256_present": (
            inline_bundle_transport.get("inline_provider_bundle_sha256_present") is True
        ),
        "machine_avoidlist_path": str(resolved_machine_avoidlist_path),
        "excluded_machine_ids": sorted(excluded_machine_ids),
        "allowed_machine_ids": sorted(resolved_allowed_machine_ids),
        "machine_allowlist_active": bool(resolved_allowed_machine_ids),
        "min_gpu_ram_mb": resolved_min_gpu_ram_mb,
        "min_compute_cap": resolved_min_compute_cap,
        "max_compute_cap": resolved_max_compute_cap,
        "session_budget_ledger_path": str(resolved_session_budget_ledger_path),
        "vast_launch_lock_path": str(resolved_vast_launch_lock_path),
        "vast_launch_lock_manifest_path": str(resolved_job_dir / "vast_launch_lock_manifest.json"),
        "secret_values_in_artifact": False,
        "raw_api_key_stored": False,
        "vast_secret_file_status": vast_secret_status,
        "ngc_secret_file_status": ngc_secret_status,
        "docker_username_file_status": docker_username_status,
        "docker_pat_file_status": docker_pat_status,
        "hf_token_file_status": hf_token_status,
        **_truth_boundaries(),
    }
    provider_worker_endpoint_manifest = write_provider_worker_endpoint_manifest(
        output_dir=resolved_job_dir,
        provider="vast",
        mode=mode,
        job_id=resolved_job_dir.name,
        provider_request_shape={
            "operation": "vast_provider_worker_startup_probe",
            "command": launch_mode,
            "image": {"configured_image_ref": selected_container_image},
            "inputs": {
                "provider_bundle_kind": provider_bundle_kind,
                "provider_bundle_url_present": bool(_string(provider_bundle_url)),
                "provider_output_get_url_present": bool(_string(provider_output_get_url)),
                "provider_output_put_url_present": bool(_string(provider_output_put_url)),
            },
            "vast_launch_mode": launch_mode,
            "provider_bundle_kind": provider_bundle_kind,
            "selected_container_image": selected_container_image,
        },
    )
    base_result.update(
        {
            "provider_worker_endpoint_manifest_path": str(
                resolved_job_dir / "provider_worker_endpoint_manifest.json"
            ),
            "provider_worker_endpoint_manifest": provider_worker_endpoint_manifest,
        }
    )

    if mode == "dry-run":
        dry_offer_request = _search_payload(limit=100, max_hourly_rate=max_hourly_rate)
        offer_manifest = {
            "schema_version": VAST_OFFER_SELECTION_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "dry_run_ready",
            "offer_search_performed": False,
            "offer_search_request": dry_offer_request,
            "selected_offer": None,
            "selected_hourly_rate_usd": None,
            "machine_avoidlist_path": str(resolved_machine_avoidlist_path),
            "excluded_machine_ids": sorted(excluded_machine_ids),
            "allowed_machine_ids": sorted(resolved_allowed_machine_ids),
            "machine_allowlist_active": bool(resolved_allowed_machine_ids),
            "blockers": [],
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_offer_selection_manifest.json", offer_manifest)
        _write_blocked_phase_artifacts(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            heartbeat_reason="dry_run_no_vast_instance_started",
            gpu_reason="dry_run_no_vast_instance_started",
            isaac_reason="dry_run_no_vast_instance_started",
            provider_reason="dry_run_no_vast_instance_started",
        )
        teardown = {
            "schema_version": VAST_TEARDOWN_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "not_required_dry_run",
            "vast_instance_ids": [],
            "teardown_actions_performed": [],
            "continuing_spend_from_this_run": False,
            "zero_continuing_spend_scope": "dry run made no Vast API calls",
        }
        write_json(resolved_job_dir / "vast_teardown_manifest.json", teardown)
        _append_phase(
            resolved_job_dir,
            "vast_offer_search_started",
            "blocked",
            blockers=["dry_run_no_offer_search"],
        )
        _append_phase(
            resolved_job_dir,
            "vast_offer_selected",
            "blocked",
            blockers=["dry_run_no_offer_selected"],
        )
        _append_phase(
            resolved_job_dir,
            "vast_instance_create_requested",
            "blocked",
            blockers=["dry_run_no_instance_create"],
        )
        _append_phase(
            resolved_job_dir,
            "vast_instance_started_or_blocked",
            "blocked",
            blockers=["dry_run_no_instance_started"],
        )
        for phase in VAST_REQUIRED_PHASES[6:]:
            _append_phase(
                resolved_job_dir, phase, "blocked", blockers=["dry_run_no_instance_started"]
            )
        validation = _final_validation(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            instance_ids=[],
            continuing_spend=False,
            estimated_cost_usd=0.0,
            hard_cap_usd=hard_cap_usd,
        )
        base_result.update(
            {
                "status": "dry_run_ready",
                "reason": "vast_probe_request_shape_validated_without_api_call",
                "blockers": [],
                "offer_search_request": dry_offer_request,
                "final_validation_status": validation["status"],
            }
        )
        write_json(result_path, base_result)
        return base_result

    if mode != "live-startup-probe":
        if mode == "template-discovery":
            blockers: list[str] = []
            if os.environ.get(VAST_API_GATE_ENV) != "true" or not allow_vast_api_call:
                blockers.append("missing_read_only_vast_api_gate")
            if not api_key:
                blockers.append("missing_file_based_vast_api_key")
            if blockers:
                reason = "template_discovery_no_vast_instance_started"
                template_discovery = {
                    "schema_version": VAST_TEMPLATE_DISCOVERY_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "status": "blocked",
                    "job_dir": str(resolved_job_dir),
                    "blockers": blockers,
                    "raw_secret_values_recorded": False,
                }
                write_json(resolved_job_dir / "vast_template_discovery.json", template_discovery)
                _ensure_offer_manifest(
                    resolved_job_dir,
                    generated_at=generated_at,
                    blockers=[reason, *blockers],
                )
                _write_blocked_phase_artifacts(
                    job_dir=resolved_job_dir,
                    generated_at=generated_at,
                    heartbeat_reason=reason,
                    gpu_reason=reason,
                    isaac_reason=reason,
                    provider_reason=reason,
                )
                write_json(
                    resolved_job_dir / "vast_teardown_manifest.json",
                    {
                        "schema_version": VAST_TEARDOWN_SCHEMA_VERSION,
                        "generated_at": generated_at,
                        "status": "not_required_template_discovery_blocked",
                        "vast_instance_ids": [],
                        "teardown_actions_performed": [],
                        "continuing_spend_from_this_run": False,
                        "zero_continuing_spend_scope": "template discovery blocked before Vast API call",
                    },
                )
                _append_preallocation_blocked_phases(
                    resolved_job_dir,
                    blockers=[reason, *blockers],
                    start_index=2,
                )
                validation = _final_validation(
                    job_dir=resolved_job_dir,
                    generated_at=generated_at,
                    instance_ids=[],
                    continuing_spend=False,
                    estimated_cost_usd=0.0,
                    hard_cap_usd=hard_cap_usd,
                )
                base_result.update(
                    {
                        "status": "blocked",
                        "reason": "vast_template_discovery_gate_blocked",
                        "blockers": blockers,
                        "template_discovery": template_discovery,
                        "final_validation_status": validation["status"],
                    }
                )
                write_json(result_path, base_result)
                return base_result
            template_discovery = _discover_vast_templates(
                job_dir=resolved_job_dir,
                generated_at=generated_at,
                api_key=api_key,
            )
            reason = "template_discovery_no_vast_instance_started"
            _ensure_offer_manifest(
                resolved_job_dir,
                generated_at=generated_at,
                blockers=[reason],
            )
            _write_blocked_phase_artifacts(
                job_dir=resolved_job_dir,
                generated_at=generated_at,
                heartbeat_reason=reason,
                gpu_reason=reason,
                isaac_reason=reason,
                provider_reason=reason,
            )
            write_json(
                resolved_job_dir / "vast_teardown_manifest.json",
                {
                    "schema_version": VAST_TEARDOWN_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "status": "not_required_template_discovery",
                    "vast_instance_ids": [],
                    "teardown_actions_performed": [],
                    "continuing_spend_from_this_run": False,
                    "zero_continuing_spend_scope": "template discovery does not allocate Vast instances",
                },
            )
            _append_preallocation_blocked_phases(
                resolved_job_dir,
                blockers=[reason],
                start_index=2,
            )
            validation = _final_validation(
                job_dir=resolved_job_dir,
                generated_at=generated_at,
                instance_ids=[],
                continuing_spend=False,
                estimated_cost_usd=0.0,
                hard_cap_usd=hard_cap_usd,
            )
            base_result.update(
                {
                    "status": "completed"
                    if template_discovery.get("status") == "completed"
                    else "blocked",
                    "reason": "vast_template_discovery_completed"
                    if template_discovery.get("status") == "completed"
                    else "vast_template_discovery_blocked",
                    "api_call_performed": True,
                    "vast_side_effects_may_have_occurred": False,
                    "blockers": _string_list(template_discovery.get("blockers")),
                    "template_discovery": template_discovery,
                    "final_validation_status": validation["status"],
                }
            )
            write_json(result_path, base_result)
            return base_result
        base_result.update(
            {
                "status": "blocked",
                "reason": "unsupported_vast_adapter_mode",
                "blockers": [f"unsupported_vast_adapter_mode:{mode}"],
            }
        )
        write_json(result_path, base_result)
        return base_result

    gate_blockers = _api_gate_blockers(
        allow_vast_api_call=allow_vast_api_call,
        allow_instance_launch=allow_instance_launch,
        api_key=api_key,
    )
    if gate_blockers:
        offer_manifest = {
            "schema_version": VAST_OFFER_SELECTION_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "offer_search_performed": False,
            "selected_offer": None,
            "machine_avoidlist_path": str(resolved_machine_avoidlist_path),
            "excluded_machine_ids": sorted(excluded_machine_ids),
            "blockers": gate_blockers,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_offer_selection_manifest.json", offer_manifest)
        _write_blocked_phase_artifacts(job_dir=resolved_job_dir, generated_at=generated_at)
        write_json(
            resolved_job_dir / "vast_teardown_manifest.json",
            {
                "schema_version": VAST_TEARDOWN_SCHEMA_VERSION,
                "generated_at": generated_at,
                "status": "not_required_gate_blocked",
                "vast_instance_ids": [],
                "teardown_actions_performed": [],
                "continuing_spend_from_this_run": False,
                "zero_continuing_spend_scope": "Vast API and instance launch gates blocked before allocation",
            },
        )
        _append_preallocation_blocked_phases(
            resolved_job_dir,
            blockers=gate_blockers,
            start_index=2,
        )
        validation = _final_validation(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            instance_ids=[],
            continuing_spend=False,
            estimated_cost_usd=0.0,
            hard_cap_usd=hard_cap_usd,
        )
        base_result.update(
            {
                "status": "blocked",
                "reason": "vast_api_gate_blocked",
                "blockers": gate_blockers,
                "final_validation_status": validation["status"],
            }
        )
        write_json(result_path, base_result)
        return base_result

    bundle_preflight = _blueprint_bundle_preflight(
        job_dir=resolved_job_dir,
        generated_at=generated_at,
        enable_blueprint_bundle=enable_blueprint_bundle,
        enable_isaac_smoke=enable_isaac_smoke,
        provider_bundle_kind=provider_bundle_kind,
        bundle_path=bundle_path,
        provider_bundle_url=provider_bundle_url,
        provider_output_put_url=provider_output_put_url,
        verify_staging_urls=verify_staging_urls,
        allow_staging_output_put_probe=allow_staging_output_put_probe,
    )
    bundle_preflight_blockers = _string_list(bundle_preflight.get("blockers"))
    image_startup_preflight = _isaac_image_startup_preflight(
        job_dir=resolved_job_dir,
        generated_at=generated_at,
        enable_isaac_smoke=enable_isaac_smoke,
        enable_blueprint_bundle=enable_blueprint_bundle,
        provider_bundle_kind=provider_bundle_kind,
        selected_container_image=selected_container_image,
        vast_template_hash_id=template_hash,
        use_vast_template_image=use_vast_template_image,
        max_live_minutes=max_live_minutes,
        allow_cold_isaac_image_pull=allow_cold_isaac_image_pull,
        min_cold_isaac_pull_live_minutes=min_cold_isaac_pull_live_minutes,
    )
    image_startup_blockers = _string_list(image_startup_preflight.get("blockers"))

    session_guard = _session_budget_guard(
        job_dir=resolved_job_dir,
        generated_at=generated_at,
        budget_path=resolved_session_budget_ledger_path,
        session_max_live_minutes=session_max_live_minutes,
        requested_max_live_minutes=max_live_minutes,
        target_spend_usd=target_spend_usd,
        hard_cap_usd=hard_cap_usd,
        max_hourly_rate=max_hourly_rate,
    )
    session_blockers = _string_list(session_guard.get("blockers"))
    if session_blockers:
        offer_manifest = {
            "schema_version": VAST_OFFER_SELECTION_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "offer_search_performed": False,
            "selected_offer": None,
            "machine_avoidlist_path": str(resolved_machine_avoidlist_path),
            "excluded_machine_ids": sorted(excluded_machine_ids),
            "session_budget_guard_path": str(resolved_job_dir / "vast_session_budget_guard.json"),
            "blueprint_bundle_preflight_path": str(
                resolved_job_dir / "vast_blueprint_bundle_preflight.json"
            ),
            "blueprint_bundle_preflight_blockers": bundle_preflight_blockers,
            "isaac_image_startup_preflight_path": str(
                resolved_job_dir / "vast_isaac_image_startup_preflight.json"
            ),
            "isaac_image_startup_preflight_blockers": image_startup_blockers,
            "blockers": session_blockers,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_offer_selection_manifest.json", offer_manifest)
        _write_blocked_phase_artifacts(job_dir=resolved_job_dir, generated_at=generated_at)
        write_json(
            resolved_job_dir / "vast_teardown_manifest.json",
            {
                "schema_version": VAST_TEARDOWN_SCHEMA_VERSION,
                "generated_at": generated_at,
                "status": "not_required_session_budget_blocked",
                "vast_instance_ids": [],
                "teardown_actions_performed": [],
                "continuing_spend_from_this_run": False,
                "zero_continuing_spend_scope": "session budget guard blocked before Vast API offer search",
            },
        )
        _append_preallocation_blocked_phases(
            resolved_job_dir,
            blockers=session_blockers,
            start_index=2,
        )
        validation = _final_validation(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            instance_ids=[],
            continuing_spend=False,
            estimated_cost_usd=0.0,
            hard_cap_usd=hard_cap_usd,
        )
        base_result.update(
            {
                "status": "blocked",
                "reason": "vast_session_budget_guard_blocked",
                "blockers": session_blockers,
                "session_budget_guard": session_guard,
                "blueprint_bundle_preflight": bundle_preflight,
                "isaac_image_startup_preflight": image_startup_preflight,
                "final_validation_status": validation["status"],
            }
        )
        write_json(result_path, base_result)
        return base_result

    if image_startup_blockers:
        offer_manifest = {
            "schema_version": VAST_OFFER_SELECTION_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "offer_search_performed": False,
            "selected_offer": None,
            "machine_avoidlist_path": str(resolved_machine_avoidlist_path),
            "excluded_machine_ids": sorted(excluded_machine_ids),
            "isaac_image_startup_preflight_path": str(
                resolved_job_dir / "vast_isaac_image_startup_preflight.json"
            ),
            "blockers": image_startup_blockers,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_offer_selection_manifest.json", offer_manifest)
        _write_blocked_phase_artifacts(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            heartbeat_reason="isaac_image_startup_preflight_blocked",
            gpu_reason="isaac_image_startup_preflight_blocked",
            isaac_reason="isaac_image_startup_preflight_blocked",
            provider_reason="isaac_image_startup_preflight_blocked",
        )
        write_json(
            resolved_job_dir / "vast_teardown_manifest.json",
            {
                "schema_version": VAST_TEARDOWN_SCHEMA_VERSION,
                "generated_at": generated_at,
                "status": "not_required_isaac_image_startup_preflight_blocked",
                "vast_instance_ids": [],
                "teardown_actions_performed": [],
                "continuing_spend_from_this_run": False,
                "zero_continuing_spend_scope": "Isaac image startup preflight blocked before Vast API offer search",
            },
        )
        _append_preallocation_blocked_phases(
            resolved_job_dir,
            blockers=image_startup_blockers,
            start_index=2,
        )
        validation = _final_validation(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            instance_ids=[],
            continuing_spend=False,
            estimated_cost_usd=0.0,
            hard_cap_usd=hard_cap_usd,
        )
        base_result.update(
            {
                "status": "blocked",
                "reason": "vast_isaac_image_startup_preflight_blocked",
                "blockers": image_startup_blockers,
                "isaac_image_startup_preflight": image_startup_preflight,
                "blueprint_bundle_preflight": bundle_preflight,
                "final_validation_status": validation["status"],
            }
        )
        write_json(result_path, base_result)
        return base_result

    if enable_blueprint_bundle and bundle_preflight_blockers:
        offer_manifest = {
            "schema_version": VAST_OFFER_SELECTION_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "offer_search_performed": False,
            "selected_offer": None,
            "machine_avoidlist_path": str(resolved_machine_avoidlist_path),
            "excluded_machine_ids": sorted(excluded_machine_ids),
            "blueprint_bundle_preflight_path": str(
                resolved_job_dir / "vast_blueprint_bundle_preflight.json"
            ),
            "blockers": bundle_preflight_blockers,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_offer_selection_manifest.json", offer_manifest)
        write_json(
            resolved_job_dir / "vast_provider_command_result.json",
            {
                "schema_version": VAST_PROVIDER_COMMAND_SCHEMA_VERSION,
                "generated_at": generated_at,
                "status": "blocked",
                "provider_runtime_output_zip_produced": False,
                "provider_command_path_remote_proven": False,
                "provider_bundle_path": str(bundle_path) if bundle_path else None,
                "provider_bundle_fetch_url_present": bool(_string(provider_bundle_url)),
                "provider_output_put_url_present": bool(_string(provider_output_put_url)),
                "blueprint_bundle_preflight_path": str(
                    resolved_job_dir / "vast_blueprint_bundle_preflight.json"
                ),
                "blockers": bundle_preflight_blockers,
                **_truth_boundaries(),
            },
        )
        _write_blocked_phase_artifacts(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            provider_reason="blueprint_bundle_preflight_blocked",
        )
        write_json(
            resolved_job_dir / "vast_teardown_manifest.json",
            {
                "schema_version": VAST_TEARDOWN_SCHEMA_VERSION,
                "generated_at": generated_at,
                "status": "not_required_blueprint_bundle_preflight_blocked",
                "vast_instance_ids": [],
                "teardown_actions_performed": [],
                "continuing_spend_from_this_run": False,
                "zero_continuing_spend_scope": "Blueprint bundle preflight blocked before Vast API offer search",
            },
        )
        _append_preallocation_blocked_phases(
            resolved_job_dir,
            blockers=bundle_preflight_blockers,
            start_index=2,
        )
        validation = _final_validation(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            instance_ids=[],
            continuing_spend=False,
            estimated_cost_usd=0.0,
            hard_cap_usd=hard_cap_usd,
        )
        base_result.update(
            {
                "status": "blocked",
                "reason": "vast_blueprint_bundle_preflight_blocked",
                "blockers": bundle_preflight_blockers,
                "blueprint_bundle_preflight": bundle_preflight,
                "final_validation_status": validation["status"],
            }
        )
        write_json(result_path, base_result)
        return base_result

    launch_lock_handle, launch_lock_manifest = _try_acquire_vast_launch_lock(
        job_dir=resolved_job_dir,
        generated_at=generated_at,
        lock_path=resolved_vast_launch_lock_path,
    )
    base_result.update(
        {
            "vast_launch_lock_status": launch_lock_manifest.get("status"),
            "vast_launch_lock_acquired": launch_lock_manifest.get("lock_acquired") is True,
            "vast_launch_lock_manifest": launch_lock_manifest,
        }
    )
    if launch_lock_handle is None:
        lock_blockers = _string_list(launch_lock_manifest.get("blockers")) or [
            "vast_paid_launch_lock_busy"
        ]
        offer_manifest = {
            "schema_version": VAST_OFFER_SELECTION_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "offer_search_performed": False,
            "selected_offer": None,
            "machine_avoidlist_path": str(resolved_machine_avoidlist_path),
            "excluded_machine_ids": sorted(excluded_machine_ids),
            "launch_lock_manifest_path": str(resolved_job_dir / "vast_launch_lock_manifest.json"),
            "blockers": lock_blockers,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_offer_selection_manifest.json", offer_manifest)
        _write_blocked_phase_artifacts(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            heartbeat_reason="vast_paid_launch_lock_busy",
            gpu_reason="vast_paid_launch_lock_busy",
            isaac_reason="vast_paid_launch_lock_busy",
            provider_reason="vast_paid_launch_lock_busy",
        )
        write_json(
            resolved_job_dir / "vast_teardown_manifest.json",
            {
                "schema_version": VAST_TEARDOWN_SCHEMA_VERSION,
                "generated_at": generated_at,
                "status": "not_required_launch_lock_blocked",
                "vast_instance_ids": [],
                "teardown_actions_performed": [],
                "continuing_spend_from_this_run": False,
                "zero_continuing_spend_scope": (
                    "Vast paid launch lock blocked before offer search or allocation"
                ),
                "raw_secret_values_recorded": False,
            },
        )
        _append_preallocation_blocked_phases(
            resolved_job_dir,
            blockers=lock_blockers,
            start_index=2,
        )
        validation = _final_validation(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            instance_ids=[],
            continuing_spend=False,
            estimated_cost_usd=0.0,
            hard_cap_usd=hard_cap_usd,
        )
        base_result.update(
            {
                "status": "blocked",
                "reason": "vast_paid_launch_lock_blocked",
                "blockers": lock_blockers,
                "api_call_performed": False,
                "vast_side_effects_may_have_occurred": False,
                "final_validation_status": validation["status"],
            }
        )
        write_json(result_path, base_result)
        return base_result

    prelaunch_inventory_guard = _prelaunch_inventory_guard(
        job_dir=resolved_job_dir,
        generated_at=generated_at,
        api_key=api_key,
    )
    prelaunch_inventory_blockers = _string_list(prelaunch_inventory_guard.get("blockers"))
    base_result.update(
        {
            "prelaunch_inventory_guard_status": prelaunch_inventory_guard.get("status"),
            "prelaunch_inventory_guard": prelaunch_inventory_guard,
        }
    )
    if prelaunch_inventory_blockers:
        launch_lock_release_manifest = _release_vast_launch_lock(
            launch_lock_handle,
            job_dir=resolved_job_dir,
            generated_at=utc_now_iso(),
        )
        launch_lock_handle = None
        offer_manifest = {
            "schema_version": VAST_OFFER_SELECTION_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "offer_search_performed": False,
            "selected_offer": None,
            "machine_avoidlist_path": str(resolved_machine_avoidlist_path),
            "excluded_machine_ids": sorted(excluded_machine_ids),
            "allowed_machine_ids": sorted(resolved_allowed_machine_ids),
            "machine_allowlist_active": bool(resolved_allowed_machine_ids),
            "prelaunch_inventory_guard_path": str(
                resolved_job_dir / "vast_prelaunch_inventory_guard.json"
            ),
            "blockers": prelaunch_inventory_blockers,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_offer_selection_manifest.json", offer_manifest)
        _write_blocked_phase_artifacts(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            heartbeat_reason="vast_prelaunch_inventory_guard_blocked",
            gpu_reason="vast_prelaunch_inventory_guard_blocked",
            isaac_reason="vast_prelaunch_inventory_guard_blocked",
            provider_reason="vast_prelaunch_inventory_guard_blocked",
        )
        write_json(
            resolved_job_dir / "vast_teardown_manifest.json",
            {
                "schema_version": VAST_TEARDOWN_SCHEMA_VERSION,
                "generated_at": generated_at,
                "status": "not_required_prelaunch_inventory_guard_blocked",
                "vast_instance_ids": [],
                "teardown_actions_performed": [],
                "continuing_spend_from_this_run": False,
                "zero_continuing_spend_scope": (
                    "Vast prelaunch inventory guard blocked before offer search or allocation"
                ),
                "raw_secret_values_recorded": False,
            },
        )
        _append_preallocation_blocked_phases(
            resolved_job_dir,
            blockers=prelaunch_inventory_blockers,
            start_index=2,
        )
        validation = _final_validation(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            instance_ids=[],
            continuing_spend=False,
            estimated_cost_usd=0.0,
            hard_cap_usd=hard_cap_usd,
        )
        base_result.update(
            {
                "status": "blocked",
                "reason": "vast_prelaunch_inventory_guard_blocked",
                "blockers": prelaunch_inventory_blockers,
                "api_call_performed": True,
                "vast_side_effects_may_have_occurred": False,
                "vast_launch_lock_status": (
                    launch_lock_release_manifest or launch_lock_manifest
                ).get("status"),
                "vast_launch_lock_manifest": launch_lock_release_manifest or launch_lock_manifest,
                "final_validation_status": validation["status"],
            }
        )
        write_json(result_path, base_result)
        return base_result

    try:
        require_paid_resource_admission_grant(
            paid_resource_admission_grant,
            resource_class="vast_provider_adapter",
        )
    except PaidResourceAdmissionBlocked as exc:
        base_result.update(
            {
                "status": "blocked",
                "reason": "shared_paid_resource_admission_blocked",
                "blockers": [
                    "vast_provider_shared_admission_missing_or_invalid",
                    *exc.blockers,
                ],
                "api_call_performed": True,
                "vast_side_effects_may_have_occurred": False,
            }
        )
        write_json(result_path, base_result)
        return base_result

    secret_values = [
        api_key,
        ngc_key,
        docker_username,
        docker_pat,
        hf_token,
        *_forwarded_secret_values(),
        *_url_secret_values(provider_bundle_url, provider_output_put_url, provider_output_get_url),
        _string(inline_bundle_transport.get("inline_provider_bundle_base64")),
    ]
    teardown_actions: list[dict[str, Any]] = []
    continuing_spend = False
    estimated_cost_usd = 0.0
    launch_lock_release_manifest: dict[str, Any] | None = None
    previous_signal_handlers: dict[int, Any] = {}

    ignore_local_probe_signals = _env_truthy(
        "BLUEPRINT_VAST_IGNORE_LOCAL_SIGTERM_DURING_PROVIDER_RUN"
    )
    ignored_signal_counts: dict[str, int] = {}

    def _raise_probe_interrupt(signum: int, _frame: Any) -> None:
        if signum in {signal.SIGINT, signal.SIGTERM} and ignore_local_probe_signals:
            key = str(signum)
            ignored_signal_counts[key] = ignored_signal_counts.get(key, 0) + 1
            write_json(
                resolved_job_dir / "vast_signal_handling_manifest.json",
                {
                    "schema_version": "vast_signal_handling_manifest.v1",
                    "generated_at": utc_now_iso(),
                    "status": "ignored_local_probe_signal",
                    "ignored_signal_counts": ignored_signal_counts,
                    "ignore_local_sigterm_during_provider_run": True,
                    "ignored_signal": signum,
                    "reason": (
                        "Local Codex runner may send SIGINT/SIGTERM to long-running local "
                        "provider orchestration; adapter keeps running so Vast can "
                        "reach its own timeout/teardown path."
                    ),
                    "raw_secret_values_recorded": False,
                },
            )
            return
        raise KeyboardInterrupt(f"vast_probe_signal:{signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _raise_probe_interrupt)
        except (ValueError, OSError, AttributeError):
            # Signal registration is only available from the main thread and on
            # platforms that expose the given signal. Teardown still runs for
            # regular Python exceptions below.
            previous_signal_handlers.pop(signum, None)
    try:
        search_request = _search_payload(limit=100, max_hourly_rate=max_hourly_rate)
        create_retry_attempts: list[dict[str, Any]] = []
        max_stale_offer_retries = _vast_stale_offer_create_retry_attempts()
        create_status = 0
        create_response: dict[str, Any] = {}
        create_payload: dict[str, Any] = {}
        image_login_summary: dict[str, Any] = {}
        for create_attempt_index in range(max_stale_offer_retries + 1):
            _append_phase(
                resolved_job_dir,
                "vast_offer_search_started",
                "running",
                create_attempt_index=create_attempt_index,
            )
            status_code, search_response = _api_json(
                method="POST",
                path="/bundles/",
                api_key=api_key,
                payload=search_request,
                timeout_seconds=45,
            )
            offers = _offers_from_response(search_response)
            selected_offer = _select_offer(
                offers,
                max_hourly_rate=max_hourly_rate,
                min_gpu_ram_mb=resolved_min_gpu_ram_mb,
                min_compute_cap=resolved_min_compute_cap,
                max_compute_cap=resolved_max_compute_cap,
                excluded_machine_ids=excluded_machine_ids,
                allowed_machine_ids=resolved_allowed_machine_ids,
                require_known_supported_isaac_driver=require_known_supported_isaac_driver,
                min_reliability=resolved_min_reliability,
                require_direct_port=resolved_require_direct_port,
                preferred_gpu_keywords=resolved_preferred_gpu_keywords,
                preferred_geolocation_regex=resolved_preferred_geolocation_regex,
                prefer_isaac_rt=resolved_prefer_isaac_rt,
                gpu_selection_policy=gpu_selection_policy,
            )
            offer_blockers: list[str] = []
            if not selected_offer:
                if resolved_allowed_machine_ids:
                    offer_blockers.append("no_vast_offer_matching_allowed_machine_ids")
                if resolved_min_compute_cap:
                    offer_blockers.append("no_vast_offer_meeting_min_compute_cap")
                if vcc.any_offer_exceeds_ceiling(offers, resolved_max_compute_cap):
                    offer_blockers.append("no_vast_offer_at_or_below_max_compute_cap")
                offer_blockers.append(
                    "no_vast_offer_with_known_supported_isaac_driver_at_or_below_max_hourly_rate"
                    if require_known_supported_isaac_driver
                    else "no_vast_offer_at_or_below_max_hourly_rate"
                )
            offer_manifest = _offer_selection_manifest(
                generated_at=generated_at,
                status_code=status_code,
                offers=offers,
                selected_offer=selected_offer,
                max_hourly_rate=max_hourly_rate,
                min_gpu_ram_mb=resolved_min_gpu_ram_mb,
                min_compute_cap=resolved_min_compute_cap,
                max_compute_cap=resolved_max_compute_cap,
                require_known_supported_isaac_driver=require_known_supported_isaac_driver,
                excluded_machine_ids=excluded_machine_ids,
                allowed_machine_ids=resolved_allowed_machine_ids,
                machine_avoidlist_path=resolved_machine_avoidlist_path,
                avoidlist_status=_string(avoidlist.get("status")) or None,
                blockers=offer_blockers,
                min_reliability=resolved_min_reliability,
                require_direct_port=resolved_require_direct_port,
                preferred_gpu_keywords=resolved_preferred_gpu_keywords,
                preferred_geolocation_regex=resolved_preferred_geolocation_regex,
                prefer_isaac_rt=resolved_prefer_isaac_rt,
                gpu_selection_policy=gpu_selection_policy,
                create_retry_attempts=create_retry_attempts,
            )
            write_json(resolved_job_dir / "vast_offer_selection_manifest.json", offer_manifest)
            _append_phase(
                resolved_job_dir,
                "vast_offer_selected",
                "completed" if selected_offer else "blocked",
                blockers=offer_blockers,
                proof_effect="vast_offer_selected_under_budget" if selected_offer else "none",
                create_attempt_index=create_attempt_index,
            )
            if not selected_offer:
                raise RuntimeError("no_vast_offer_selected")

            projected_full_cost = float(selected_offer["hourly_rate_usd"]) * max_live_minutes / 60.0
            if projected_full_cost > hard_cap_usd:
                raise RuntimeError("selected_offer_projected_max_runtime_exceeds_hard_cap")

            image_login, image_login_summary = _resolve_image_login(
                image=create_request_image or "",
                ngc_key=ngc_key,
                docker_username=docker_username,
                docker_pat=docker_pat,
                mode=resolved_image_login_mode,
            )
            probe_script = _probe_shell_script(
                heartbeat_url,
                enable_isaac_smoke=enable_isaac_smoke,
                enable_blueprint_bundle=enable_blueprint_bundle,
                provider_bundle_kind=provider_bundle_kind,
            )
            create_payload = _create_payload(
                image=create_request_image,
                label=f"blueprint-vast-probe-{int(time.time())}",
                launch_mode=launch_mode,
                probe_script=probe_script,
                disk_gb=resolved_disk_gb,
                env=_probe_env(
                    job_dir=resolved_job_dir,
                    enable_isaac_smoke=enable_isaac_smoke,
                    provider_bundle_url=provider_bundle_url,
                    provider_output_put_url=provider_output_put_url,
                    provider_bundle_inline_base64=_string(
                        inline_bundle_transport.get("inline_provider_bundle_base64")
                    ),
                    provider_bundle_inline_sha256=_string(
                        inline_bundle_transport.get("inline_provider_bundle_sha256")
                    ),
                ),
                image_login=image_login,
                template_hash_id=template_hash,
            )
            _append_phase(
                resolved_job_dir,
                "vast_instance_create_requested",
                "running",
                offer_id=selected_offer["ask_contract_id"],
                create_attempt_index=create_attempt_index,
            )
            try:
                create_status, create_response = _api_json(
                    method="PUT",
                    path=f"/asks/{selected_offer['ask_contract_id']}/",
                    api_key=api_key,
                    payload=create_payload,
                    timeout_seconds=45,
                )
                break
            except urllib.error.HTTPError as exc:
                if (
                    not _is_stale_offer_create_http_error(exc)
                    or create_attempt_index >= max_stale_offer_retries
                ):
                    raise
                error_text = exc.read().decode("utf-8", errors="replace")
                selected_machine_id = _number(selected_offer.get("machine_id"))
                if selected_machine_id is not None:
                    excluded_machine_ids.add(int(selected_machine_id))
                retry_attempt = {
                    "attempt": create_attempt_index,
                    "status": "stale_offer_retry",
                    "http_status_code": exc.code,
                    "offer_id": selected_offer.get("ask_contract_id"),
                    "machine_id": int(selected_machine_id)
                    if selected_machine_id is not None
                    else None,
                    "error_preview": _redact_text(error_text[:500], secret_values),
                    "raw_secret_values_recorded": False,
                }
                create_retry_attempts.append(retry_attempt)
                _append_phase(
                    resolved_job_dir,
                    "vast_instance_create_requested",
                    "blocked",
                    blockers=["vast_create_stale_offer_retry"],
                    proof_effect="stale_offer_excluded_before_instance_allocation",
                    create_retry_status=_string(retry_attempt.get("status")),
                    create_retry_attempt=retry_attempt.get("attempt"),
                    http_status_code=retry_attempt.get("http_status_code"),
                    offer_id=retry_attempt.get("offer_id"),
                    machine_id=retry_attempt.get("machine_id"),
                )
                offer_manifest = _offer_selection_manifest(
                    generated_at=generated_at,
                    status_code=status_code,
                    offers=offers,
                    selected_offer=None,
                    max_hourly_rate=max_hourly_rate,
                    min_gpu_ram_mb=resolved_min_gpu_ram_mb,
                    require_known_supported_isaac_driver=require_known_supported_isaac_driver,
                    excluded_machine_ids=excluded_machine_ids,
                    allowed_machine_ids=resolved_allowed_machine_ids,
                    machine_avoidlist_path=resolved_machine_avoidlist_path,
                    avoidlist_status=_string(avoidlist.get("status")) or None,
                    blockers=["vast_create_stale_offer_retry"],
                    min_reliability=resolved_min_reliability,
                    require_direct_port=resolved_require_direct_port,
                    preferred_gpu_keywords=resolved_preferred_gpu_keywords,
                    preferred_geolocation_regex=resolved_preferred_geolocation_regex,
                    prefer_isaac_rt=resolved_prefer_isaac_rt,
                    gpu_selection_policy=gpu_selection_policy,
                    create_retry_attempts=create_retry_attempts,
                )
                write_json(
                    resolved_job_dir / "vast_offer_selection_manifest.json",
                    offer_manifest,
                )
                time.sleep(1)
                continue
        instance_id = _instance_id_from_create_response(create_response)
        if not instance_id:
            raise RuntimeError("vast_create_response_missing_instance_id")
        started_at_monotonic = time.monotonic()
        instance_ids.append(instance_id)
        _budget_ledger(
            job_dir=resolved_job_dir,
            generated_at=utc_now_iso(),
            target_spend_usd=target_spend_usd,
            hard_cap_usd=hard_cap_usd,
            max_hourly_rate=max_hourly_rate,
            max_live_minutes=max_live_minutes,
            selected_offer=selected_offer,
            instance_ids=instance_ids,
            started_at_monotonic=started_at_monotonic,
            ended_at_monotonic=started_at_monotonic,
            status="live_instance_created",
            continuing_spend=True,
        )
        _append_phase(
            resolved_job_dir,
            "vast_instance_create_requested",
            "completed",
            proof_effect="vast_instance_create_api_returned_instance_id",
            instance_id=instance_id,
            http_status_code=create_status,
        )
        status, observations, instance_payload = _poll_instance(
            instance_id=instance_id,
            api_key=api_key,
            timeout_seconds=min(startup_timeout_seconds, max_live_minutes * 60),
            poll_interval_seconds=poll_interval_seconds,
        )
        instance_running = status.lower() == "running"
        instance_log_readable = instance_running or (
            launch_mode == "args" and status.lower() in {"exited", "stopped"}
        )
        _append_phase(
            resolved_job_dir,
            "vast_instance_started_or_blocked",
            "completed" if instance_log_readable else "blocked",
            blockers=[] if instance_log_readable else [f"vast_instance_status:{status}"],
            proof_effect="vast_instance_reached_running_or_log_readable_state"
            if instance_log_readable
            else "none",
            instance_id=instance_id,
            launch_mode=launch_mode,
        )
        if not instance_log_readable:
            write_json(
                resolved_job_dir / "vast_startup_probe_manifest.json",
                {
                    "schema_version": VAST_STARTUP_PROBE_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "status": "blocked",
                    "instance_id": instance_id,
                    "launch_mode_used": launch_mode,
                    "disk_gb": resolved_disk_gb,
                    "create_request_summary": _create_request_summary(
                        create_payload,
                        secret_values=secret_values,
                    ),
                    "image_login_summary": image_login_summary,
                    "create_http_status_code": create_status,
                    "create_response": _redact_runtime_value(create_response, secret_values),
                    "instance_observations": observations,
                    "last_instance_payload": _redact_runtime_value(instance_payload, secret_values),
                    "blockers": [f"vast_instance_status:{status}"],
                    "heartbeat_completed": False,
                    **_truth_boundaries(),
                },
            )
            raise RuntimeError(f"vast_instance_not_running:{status}")

        _append_phase(
            resolved_job_dir, "vast_heartbeat_started", "running", instance_id=instance_id
        )
        log_success_markers = (
            [
                "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED",
                "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK",
                "BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED",
                "BLUEPRINT_VAST_ONSTART_DONE",
            ]
            if enable_blueprint_bundle
            else [
                "BLUEPRINT_VAST_ISAAC_SMOKE_OK",
                "BLUEPRINT_VAST_ISAAC_SMOKE_BLOCKED",
                "BLUEPRINT_VAST_ONSTART_DONE",
            ]
            if enable_isaac_smoke
            else [
                "BLUEPRINT_VAST_HEARTBEAT_OK",
                "BLUEPRINT_VAST_HEARTBEAT_BLOCKED",
                "BLUEPRINT_VAST_GPU_SANITY_OK",
                "BLUEPRINT_VAST_GPU_SANITY_BLOCKED",
            ]
        )
        resolved_heartbeat_no_progress_seconds = (
            _env_int(
                VAST_HEARTBEAT_NO_PROGRESS_SECONDS_ENV,
                DEFAULT_HEARTBEAT_NO_PROGRESS_SECONDS,
            )
            if heartbeat_no_progress_seconds is None
            else heartbeat_no_progress_seconds
        )
        onstart_logs = _request_logs_and_fetch(
            instance_id=instance_id,
            api_key=api_key,
            output_log_path=resolved_job_dir / "vast_onstart_container.log",
            secret_values=secret_values,
            max_wait_seconds=min(startup_timeout_seconds, max_live_minutes * 60),
            retry_interval_seconds=30,
            success_markers=log_success_markers,
            container_missing_retry_attempts=int(
                os.environ.get(VAST_CONTAINER_MISSING_RETRY_ATTEMPTS_ENV, "2")
            ),
            no_progress_seconds=resolved_heartbeat_no_progress_seconds,
        )
        heartbeat_text = Path(onstart_logs["output_log_path"]).read_text(encoding="utf-8")
        if not _log_text_has_success_marker(
            heartbeat_text, log_success_markers
        ) and _log_result_has_container_missing(onstart_logs):
            if _env_truthy(VAST_ALLOW_COMMAND_EXECUTE_SCRIPT_FALLBACK_ENV):
                execute_logs = _execute_and_fetch(
                    instance_id=instance_id,
                    api_key=api_key,
                    command="bash -lc " + shlex.quote(probe_script),
                    output_log_path=resolved_job_dir / "vast_command_execute_probe.log",
                    secret_values=secret_values,
                    wait_seconds=20,
                )
                execute_text = Path(execute_logs["output_log_path"]).read_text(encoding="utf-8")
                onstart_logs["command_execute_fallback"] = execute_logs
                if execute_text:
                    heartbeat_text = execute_text
                    onstart_logs["effective_log_source"] = "command_execute_fallback"
                    onstart_logs["effective_output_log_path"] = execute_logs["output_log_path"]
                else:
                    onstart_logs["effective_log_source"] = "request_logs"
            else:
                onstart_logs["command_execute_fallback"] = {
                    "status": "skipped",
                    "reason": "vast_execute_api_is_constrained_to_simple_commands",
                    "enable_env": VAST_ALLOW_COMMAND_EXECUTE_SCRIPT_FALLBACK_ENV,
                    "raw_secret_values_recorded": False,
                }
                onstart_logs["effective_log_source"] = "request_logs"
        heartbeat_ok = "BLUEPRINT_VAST_HEARTBEAT_OK" in heartbeat_text
        downstream_marker_seen = any(
            marker in heartbeat_text
            for marker in (
                "BLUEPRINT_VAST_GPU_SANITY_OK",
                "BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED",
                "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED",
                "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK",
            )
        )
        startup_probe_ok = heartbeat_ok or downstream_marker_seen
        heartbeat_warnings = (
            ["vast_heartbeat_url_failed_but_downstream_provider_marker_seen"]
            if downstream_marker_seen and not heartbeat_ok
            else []
        )
        heartbeat_log_break_reason = _string(onstart_logs.get("break_reason"))
        heartbeat_no_progress_timeout = bool(
            onstart_logs.get("no_progress_timeout_reached")
            or heartbeat_log_break_reason == "no_log_progress_timeout"
        )
        heartbeat_blockers = []
        if not startup_probe_ok:
            if heartbeat_no_progress_timeout:
                heartbeat_blockers.append("vast_heartbeat_no_log_progress_timeout")
            if _log_result_has_container_missing(onstart_logs):
                heartbeat_blockers.append("vast_heartbeat_container_missing")
            heartbeat_blockers.append("vast_heartbeat_output_missing_success_marker")
            heartbeat_blockers = _dedupe(heartbeat_blockers)
        heartbeat_manifest = {
            "schema_version": VAST_STARTUP_PROBE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "completed" if startup_probe_ok else "blocked",
            "instance_id": instance_id,
            "heartbeat_completed": heartbeat_ok,
            "startup_probe_proven": startup_probe_ok,
            "startup_probe_proof_source": "heartbeat_url"
            if heartbeat_ok
            else ("downstream_provider_marker" if downstream_marker_seen else "none"),
            "downstream_provider_marker_seen": downstream_marker_seen,
            "heartbeat_url_kind": "public_echo_endpoint",
            "heartbeat_no_progress_timeout_seconds": resolved_heartbeat_no_progress_seconds,
            "launch_mode_used": launch_mode,
            "disk_gb": resolved_disk_gb,
            "container_image": selected_container_image,
            "create_http_status_code": create_status,
            "create_request_summary": _create_request_summary(
                create_payload,
                secret_values=secret_values,
            ),
            "image_login_summary": image_login_summary,
            "instance_observations": observations,
            "last_instance_payload": _redact_runtime_value(instance_payload, secret_values),
            "container_log_result": onstart_logs,
            "blockers": [] if startup_probe_ok else heartbeat_blockers,
            "warnings": heartbeat_warnings,
            "proof_boundary": "Heartbeat proves Vast instance startup plus command/log retrieval only.",
            **_truth_boundaries(),
        }
        write_json(resolved_job_dir / "vast_startup_probe_manifest.json", heartbeat_manifest)
        _append_phase(
            resolved_job_dir,
            "vast_heartbeat_completed_or_blocked",
            "completed" if startup_probe_ok else "blocked",
            blockers=[] if startup_probe_ok else heartbeat_blockers,
            proof_effect="vast_container_command_heartbeat_completed"
            if heartbeat_ok
            else (
                "vast_downstream_provider_marker_seen_after_heartbeat_url_failed"
                if downstream_marker_seen
                else "none"
            ),
            instance_id=instance_id,
            warnings=heartbeat_warnings,
        )
        if not heartbeat_ok and not downstream_marker_seen:
            raise RuntimeError(
                heartbeat_blockers[0] if heartbeat_blockers else "vast_heartbeat_blocked"
            )

        _append_phase(
            resolved_job_dir, "vast_gpu_sanity_started", "running", instance_id=instance_id
        )
        gpu_text = heartbeat_text
        nvidia_visible = "BLUEPRINT_VAST_GPU_SANITY_OK" in gpu_text and "NVIDIA-SMI" not in gpu_text
        gpu_ok = "BLUEPRINT_VAST_GPU_SANITY_OK" in gpu_text and not re.search(
            r"nvidia-smi: command not found|failed because it couldn't communicate",
            gpu_text,
            flags=re.IGNORECASE,
        )
        gpu_report = {
            "schema_version": VAST_GPU_SANITY_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "completed" if gpu_ok else "blocked",
            "instance_id": instance_id,
            "selected_offer": _offer_artifact_summary(selected_offer),
            "nvidia_smi_visible": gpu_ok,
            "gpu_sanity_proven": gpu_ok,
            "driver_cuda_visibility_checked": True,
            "disk_space_checked": True,
            "network_egress_checked": True,
            "bundle_download_ability_checked": False,
            "launch_mode_used": launch_mode,
            "disk_gb": resolved_disk_gb,
            "container_log_result": onstart_logs,
            "blockers": [] if gpu_ok else ["vast_gpu_sanity_output_missing_or_nvidia_smi_failed"],
            "proof_boundary": "GPU sanity proves provider GPU visibility only, not simulator execution.",
            **_truth_boundaries(),
        }
        gpu_report["nvidia_smi_marker_absent_from_error"] = nvidia_visible
        write_json(resolved_job_dir / "vast_gpu_sanity_report.json", gpu_report)
        _append_phase(
            resolved_job_dir,
            "vast_gpu_sanity_completed_or_blocked",
            "completed" if gpu_ok else "blocked",
            blockers=[] if gpu_ok else ["vast_gpu_sanity_output_missing_or_nvidia_smi_failed"],
            proof_effect="vast_gpu_nvidia_smi_completed" if gpu_ok else "none",
            instance_id=instance_id,
        )

        isaac_blockers: list[str] = []
        if provider_bundle_kind != "isaac" and not enable_isaac_smoke:
            isaac_not_required_effect = (
                f"isaac_smoke_not_required_for_{provider_bundle_kind}_bundle"
            )
            _append_phase(
                resolved_job_dir,
                "vast_isaac_smoke_started",
                "completed",
                proof_effect=isaac_not_required_effect,
            )
            _append_phase(
                resolved_job_dir,
                "vast_isaac_smoke_completed_or_blocked",
                "completed",
                proof_effect=isaac_not_required_effect,
            )
            write_json(
                resolved_job_dir / "vast_isaac_smoke_result.json",
                {
                    "schema_version": VAST_ISAAC_SMOKE_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "status": "not_required",
                    "instance_id": instance_id,
                    "provider_bundle_kind": provider_bundle_kind,
                    "isaac_smoke_attempted": False,
                    "isaac_image": isaac_image,
                    "selected_container_image": selected_container_image,
                    "launch_mode_used": launch_mode,
                    "disk_gb": resolved_disk_gb,
                    "isaac_simulation_app_started": False,
                    "blockers": [],
                    **_truth_boundaries(),
                },
            )
        elif not enable_isaac_smoke:
            isaac_blockers.append("isaac_smoke_disabled_for_this_bounded_probe")
        if (
            provider_bundle_kind == "isaac"
            and selected_offer
            and not selected_offer.get("isaac_rt_candidate")
        ):
            isaac_blockers.append("selected_gpu_not_in_isaac_rt_candidate_allowlist")
        if (
            provider_bundle_kind == "isaac"
            and image_login_summary.get("reason") == "ngc_key_missing"
        ):
            isaac_blockers.append("ngc_api_key_file_missing_or_empty_for_required_ngc_login")
        if provider_bundle_kind == "isaac" and (isaac_blockers or not gpu_ok):
            if not gpu_ok:
                isaac_blockers.append("gpu_sanity_not_proven")
            _append_phase(
                resolved_job_dir,
                "vast_isaac_smoke_started",
                "blocked",
                blockers=isaac_blockers,
            )
            _append_phase(
                resolved_job_dir,
                "vast_isaac_smoke_completed_or_blocked",
                "blocked",
                blockers=isaac_blockers,
            )
            write_json(
                resolved_job_dir / "vast_isaac_smoke_result.json",
                {
                    "schema_version": VAST_ISAAC_SMOKE_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "status": "blocked",
                    "instance_id": instance_id,
                    "isaac_smoke_attempted": False,
                    "isaac_image": isaac_image,
                    "selected_container_image": selected_container_image,
                    "launch_mode_used": launch_mode,
                    "disk_gb": resolved_disk_gb,
                    "isaac_simulation_app_started": False,
                    "blockers": isaac_blockers,
                    **_truth_boundaries(),
                },
            )
        elif provider_bundle_kind == "isaac":
            _append_phase(resolved_job_dir, "vast_isaac_smoke_started", "running")
            isaac_text = heartbeat_text
            isaac_ok = "BLUEPRINT_VAST_ISAAC_SMOKE_OK" in isaac_text
            write_json(
                resolved_job_dir / "vast_isaac_smoke_result.json",
                {
                    "schema_version": VAST_ISAAC_SMOKE_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "status": "completed" if isaac_ok else "blocked",
                    "instance_id": instance_id,
                    "isaac_smoke_attempted": True,
                    "isaac_image": isaac_image,
                    "selected_container_image": selected_container_image,
                    "launch_mode_used": launch_mode,
                    "disk_gb": resolved_disk_gb,
                    "isaac_simulation_app_started": isaac_ok,
                    "isaac_python_import_probe_completed": "BLUEPRINT_VAST_ISAAC_SMOKE_OK"
                    in isaac_text,
                    "container_log_result": onstart_logs,
                    "blockers": []
                    if isaac_ok
                    else ["isaac_simulation_app_marker_missing_or_blocked"],
                    **_truth_boundaries(),
                },
            )
            _append_phase(
                resolved_job_dir,
                "vast_isaac_smoke_completed_or_blocked",
                "completed" if isaac_ok else "blocked",
                blockers=[] if isaac_ok else ["isaac_simulation_app_marker_missing_or_blocked"],
                proof_effect="isaac_simulation_app_headless_started" if isaac_ok else "none",
            )

        provider_blockers: list[str] = []
        if not enable_blueprint_bundle:
            provider_blockers.append("blueprint_bundle_execution_disabled_for_this_probe")
        if enable_blueprint_bundle and provider_bundle_kind == "isaac" and not enable_isaac_smoke:
            provider_blockers.append("blueprint_bundle_execution_requires_isaac_smoke_path")
        if not bundle_path or not bundle_path.is_file():
            provider_blockers.append(
                "isaac_provider_runtime_bundle_missing"
                if provider_bundle_kind == "isaac"
                else "provider_runtime_bundle_missing"
            )
        if enable_blueprint_bundle and not _string(provider_bundle_url):
            provider_blockers.append("provider_bundle_fetch_url_missing")
        if enable_blueprint_bundle and not _string(provider_output_put_url):
            provider_blockers.append("provider_output_put_url_missing")
        if provider_blockers:
            _append_phase(
                resolved_job_dir,
                "vast_blueprint_bundle_started",
                "blocked",
                blockers=provider_blockers,
            )
            _append_phase(
                resolved_job_dir,
                "vast_blueprint_bundle_completed_or_blocked",
                "blocked",
                blockers=provider_blockers,
            )
            write_json(
                resolved_job_dir / "vast_provider_command_result.json",
                {
                    "schema_version": VAST_PROVIDER_COMMAND_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "status": "blocked",
                    "instance_id": instance_id,
                    "provider_runtime_output_zip_produced": False,
                    "provider_command_path_remote_proven": False,
                    "provider_bundle_path": str(bundle_path) if bundle_path else None,
                    "provider_bundle_kind": provider_bundle_kind,
                    "provider_bundle_fetch_url_present": bool(_string(provider_bundle_url)),
                    "provider_output_put_url_present": bool(_string(provider_output_put_url)),
                    "blockers": provider_blockers,
                    **_truth_boundaries(),
                },
            )
        else:
            provider_started = "BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED" in heartbeat_text
            provider_downloaded = "BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED" in heartbeat_text
            provider_entrypoint_started = (
                "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED" in heartbeat_text
            )
            provider_upload_ok = "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK" in heartbeat_text
            provider_completed_or_blocked = (
                "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED" in heartbeat_text
            )
            provider_exit_match = re.search(
                r"BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_EXIT_CODE:(\d+)",
                heartbeat_text,
            )
            provider_entrypoint_exit_code = (
                int(provider_exit_match.group(1)) if provider_exit_match else None
            )
            # Vast's request_logs endpoint returns a tail. Isaac crash output can be
            # large enough to push early phase markers out of that tail, while later
            # markers such as entrypoint exit or output upload remain. Those later
            # markers are impossible unless the earlier provider phases happened, so
            # infer them instead of reporting misleading startup blockers.
            if provider_entrypoint_exit_code is not None:
                provider_started = True
                provider_downloaded = True
                provider_entrypoint_started = True
            if provider_upload_ok:
                provider_started = True
                provider_downloaded = True
                provider_entrypoint_started = True
                provider_completed_or_blocked = True
            output_download_manifest: dict[str, Any] = {
                "schema_version": "vast_provider_output_download.v1",
                "generated_at": utc_now_iso(),
                "status": "not_requested",
                "provider_output_get_url_present": bool(_string(provider_output_get_url)),
                "provider_upload_marker_seen": provider_upload_ok,
                "output_zip_path": str(output_zip_path),
                "raw_secret_values_recorded": False,
            }
            if (
                provider_upload_ok
                and _string(provider_output_get_url)
                and not output_zip_path.is_file()
            ):
                try:
                    with urllib.request.urlopen(
                        _string(provider_output_get_url),
                        timeout=60,
                    ) as response:
                        output_zip_path.write_bytes(response.read())
                    output_download_manifest.update(
                        {
                            "status": "completed",
                            "http_status_code": int(getattr(response, "status", 200)),
                            "output_zip_present_after_download": output_zip_path.is_file(),
                            "output_zip_size_bytes": output_zip_path.stat().st_size
                            if output_zip_path.is_file()
                            else 0,
                        }
                    )
                except Exception as exc:
                    output_download_manifest.update(
                        {
                            "status": "blocked",
                            "error_type": type(exc).__name__,
                            "blockers": ["provider_output_get_url_download_failed"],
                        }
                    )
            elif provider_upload_ok and output_zip_path.is_file():
                output_download_manifest.update(
                    {
                        "status": "skipped",
                        "reason": "provider_runtime_output_zip_already_present",
                        "output_zip_present_after_download": True,
                        "output_zip_size_bytes": output_zip_path.stat().st_size,
                    }
                )
            elif provider_upload_ok and not _string(provider_output_get_url):
                output_download_manifest.update(
                    {
                        "status": "blocked",
                        "blockers": ["provider_output_get_url_missing"],
                    }
                )
            write_json(
                resolved_job_dir / "vast_provider_output_download_manifest.json",
                output_download_manifest,
            )
            provider_blocked_markers = re.findall(
                r"BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:([^\s]+)",
                heartbeat_text,
            )
            output_zip_inspection = _inspect_provider_runtime_output_zip(
                output_zip_path,
                video_extract_dir=resolved_job_dir / "vast_provider_runtime_output_videos",
                expected_video_count=_provider_expected_video_count(provider_bundle_kind),
            )
            output_zip_received = output_zip_inspection.get("zip_present") is True
            provider_runtime_output_zip_produced = provider_upload_ok and output_zip_received
            runtime_result_present = output_zip_inspection.get("runtime_result_present") is True
            video_smoke_proven = output_zip_inspection.get("video_smoke_proven") is True
            mp4_validation = _mapping(output_zip_inspection.get("mp4_validation"))
            remote_proven = (
                provider_started
                and provider_downloaded
                and provider_entrypoint_started
                and provider_completed_or_blocked
            )
            completion_blockers: list[str] = []
            if not provider_started:
                completion_blockers.append("provider_bundle_start_marker_missing")
            if not provider_downloaded:
                completion_blockers.append("provider_bundle_download_marker_missing")
            if not provider_entrypoint_started:
                completion_blockers.append("provider_entrypoint_start_marker_missing")
            if not provider_completed_or_blocked:
                completion_blockers.append("provider_bundle_completion_marker_missing")
            if not provider_upload_ok:
                completion_blockers.append("provider_output_upload_marker_missing")
            if not output_zip_received:
                completion_blockers.append("provider_runtime_output_zip_not_received_locally")
            if provider_runtime_output_zip_produced and not runtime_result_present:
                completion_blockers.append("provider_runtime_result_missing_from_output_zip")
            completion_blockers.extend(
                f"provider_remote_blocker:{marker}" for marker in provider_blocked_markers
            )
            video_blockers = _string_list(mp4_validation.get("blockers"))
            runtime_result = _mapping(output_zip_inspection.get("runtime_result"))
            runtime_result_status = _string(runtime_result.get("status"))
            expected_provider_video_count = _provider_expected_video_count(provider_bundle_kind)
            provider_status = (
                "completed"
                if remote_proven and provider_runtime_output_zip_produced and runtime_result_present
                else "blocked"
            )
            _append_phase(
                resolved_job_dir,
                "vast_blueprint_bundle_started",
                "completed" if provider_started else "blocked",
                blockers=[] if provider_started else ["provider_bundle_start_marker_missing"],
                proof_effect="provider_bundle_remote_start_marker_observed"
                if provider_started
                else "none",
            )
            _append_phase(
                resolved_job_dir,
                "vast_blueprint_bundle_completed_or_blocked",
                "completed" if provider_status == "completed" else "blocked",
                blockers=completion_blockers,
                proof_effect="blueprint_provider_bundle_remote_entrypoint_and_output_zip_proven"
                if provider_status == "completed"
                else "none",
            )
            write_json(
                resolved_job_dir / "vast_provider_command_result.json",
                {
                    "schema_version": VAST_PROVIDER_COMMAND_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "status": provider_status,
                    "instance_id": instance_id,
                    "provider_bundle_started": provider_started,
                    "provider_bundle_downloaded": provider_downloaded,
                    "provider_entrypoint_started": provider_entrypoint_started,
                    "provider_entrypoint_exit_code": provider_entrypoint_exit_code,
                    "provider_remote_blocked_markers": provider_blocked_markers,
                    "provider_completed_or_blocked_marker_seen": provider_completed_or_blocked,
                    "provider_output_upload_ok": provider_upload_ok,
                    "provider_runtime_output_zip_produced": provider_runtime_output_zip_produced,
                    "provider_runtime_output_zip_received": output_zip_received,
                    "provider_command_path_remote_proven": remote_proven,
                    "provider_bundle_path": str(bundle_path),
                    "provider_bundle_kind": provider_bundle_kind,
                    "provider_runtime_output_zip_path": str(output_zip_path),
                    "provider_bundle_fetch_url_present": True,
                    "provider_output_put_url_present": True,
                    "provider_output_get_url_present": bool(_string(provider_output_get_url)),
                    "provider_output_download_manifest": output_download_manifest,
                    "provider_runtime_output_zip_inspection": output_zip_inspection,
                    "runtime_result_status": runtime_result_status or None,
                    "runtime_result_blockers": _string_list(runtime_result.get("blockers")),
                    "blueprint_provider_bundle_execution_proven": provider_status == "completed",
                    "video_smoke_proven": video_smoke_proven,
                    "video_smoke_expected_video_count": expected_provider_video_count,
                    "blockers": completion_blockers,
                    "proof_boundary": (
                        "Provider bundle execution proves remote bundle download, entrypoint "
                        "start, and output return only. A runtime result blocker still means "
                        "controller-grade or official policy execution was not proven."
                    ),
                    **_truth_boundaries(),
                },
            )
            write_json(
                resolved_job_dir / "vast_video_smoke_result.json",
                {
                    "schema_version": VAST_VIDEO_SMOKE_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "status": (
                        "not_required"
                        if expected_provider_video_count == 0
                        else ("completed" if video_smoke_proven else "blocked")
                    ),
                    "provider_bundle_kind": provider_bundle_kind,
                    "provider_runtime_output_zip_path": str(output_zip_path),
                    "provider_runtime_output_zip_received": output_zip_received,
                    "video_smoke_proven": video_smoke_proven,
                    "expected_video_count": expected_provider_video_count,
                    "mp4_count": output_zip_inspection.get("mp4_count"),
                    "mp4_members": output_zip_inspection.get("mp4_members"),
                    "mp4_validation": mp4_validation,
                    "blockers": []
                    if video_smoke_proven or expected_provider_video_count == 0
                    else (video_blockers or ["mp4_video_smoke_not_proven"]),
                    "proof_boundary": (
                        "Video smoke is proven only when the provider output zip contains "
                        "the expected MP4 camera outputs and ffprobe confirms non-empty "
                        "duration and frame counts."
                    ),
                    **_truth_boundaries(),
                },
            )

    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        base_result.update(
            {
                "status": "failed",
                "reason": "vast_api_http_error",
                "blockers": ["vast_api_http_error"],
                "api_call_performed": True,
                "http_status_code": exc.code,
                "vast_error": _redact_text(
                    error_text, secret_values if "secret_values" in locals() else []
                ),
            }
        )
        _write_blocked_phase_artifacts(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            heartbeat_reason="vast_api_http_error",
            gpu_reason="vast_api_http_error",
            isaac_reason="vast_api_http_error",
            provider_reason="vast_api_http_error",
        )
    except KeyboardInterrupt as exc:
        interrupt_detail = str(exc)[:300] or type(exc).__name__
        interrupt_blocker = "vast_probe_interrupted_before_completion"
        base_result.update(
            {
                "status": "blocked",
                "reason": "vast_probe_interrupted",
                "blockers": [interrupt_blocker],
                "interrupt_detail": interrupt_detail,
                "api_call_performed": True,
            }
        )
        _write_blocked_phase_artifacts(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            heartbeat_reason=interrupt_blocker,
            gpu_reason=interrupt_blocker,
            isaac_reason=interrupt_blocker,
            provider_reason=interrupt_blocker,
        )
    except Exception as exc:
        exc_text = str(exc)[:300] or type(exc).__name__
        if exc_text.startswith("no_vast_offer"):
            exception_status = "blocked"
            exception_reason = "vast_offer_selection_blocked"
            exception_blockers = []
            if resolved_allowed_machine_ids:
                exception_blockers.append("no_vast_offer_matching_allowed_machine_ids")
            if resolved_min_compute_cap:
                exception_blockers.append("no_vast_offer_meeting_min_compute_cap")
            exception_blockers.append(
                "no_vast_offer_with_known_supported_isaac_driver_at_or_below_max_hourly_rate"
                if require_known_supported_isaac_driver
                else "no_vast_offer_at_or_below_max_hourly_rate"
            )
        else:
            exception_status = "failed"
            exception_reason = "vast_probe_failed"
            exception_blockers = [exc_text]
        base_result.update(
            {
                "status": exception_status,
                "reason": exception_reason,
                "blockers": exception_blockers,
                "api_call_performed": True,
            }
        )
        _write_blocked_phase_artifacts(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            heartbeat_reason=exception_blockers[0],
            gpu_reason=exception_blockers[0],
            isaac_reason=exception_blockers[0],
            provider_reason=exception_blockers[0],
        )
    finally:
        if instance_ids:
            _append_phase(
                resolved_job_dir,
                "vast_instance_teardown_started",
                "running",
                instance_ids=instance_ids,
            )
        else:
            _append_phase(
                resolved_job_dir, "vast_instance_teardown_started", "completed", instance_ids=[]
            )
        _write_blocked_phase_artifacts(
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            heartbeat_reason=_string(base_result.get("reason"))
            or "vast_probe_ended_before_phase_artifacts",
            gpu_reason=_string(base_result.get("reason"))
            or "vast_probe_ended_before_phase_artifacts",
            isaac_reason=_string(base_result.get("reason"))
            or "vast_probe_ended_before_phase_artifacts",
            provider_reason=_string(base_result.get("reason"))
            or "vast_probe_ended_before_phase_artifacts",
        )
        for instance_id in list(instance_ids):
            try:
                status_code, response = _api_json(
                    method="DELETE",
                    path=f"/instances/{instance_id}/",
                    api_key=api_key,
                    timeout_seconds=30,
                )
                teardown_actions.append(
                    {
                        "instance_id": instance_id,
                        "action": "destroy_instance",
                        "http_status_code": status_code,
                        "response": _redact_runtime_value(response, [api_key]),
                        "status": "completed",
                    }
                )
            except urllib.error.HTTPError as exc:  # pragma: no cover - live network dependent.
                if exc.code == 404:
                    teardown_actions.append(
                        {
                            "instance_id": instance_id,
                            "action": "destroy_instance",
                            "http_status_code": exc.code,
                            "status": "completed",
                            "reason": "instance_already_absent",
                        }
                    )
                else:
                    continuing_spend = True
                    teardown_actions.append(
                        {
                            "instance_id": instance_id,
                            "action": "destroy_instance",
                            "status": "failed",
                            "http_status_code": exc.code,
                            "error": _redact_text(
                                f"{type(exc).__name__}: {str(exc)[:300]}", [api_key]
                            ),
                        }
                    )
            except Exception as exc:  # pragma: no cover - live network dependent.
                continuing_spend = True
                teardown_actions.append(
                    {
                        "instance_id": instance_id,
                        "action": "destroy_instance",
                        "status": "failed",
                        "error": _redact_text(f"{type(exc).__name__}: {str(exc)[:300]}", [api_key]),
                    }
                )
        teardown_status = "completed" if not continuing_spend else "blocked"
        write_json(
            resolved_job_dir / "vast_teardown_manifest.json",
            {
                "schema_version": VAST_TEARDOWN_SCHEMA_VERSION,
                "generated_at": utc_now_iso(),
                "status": teardown_status,
                "vast_instance_ids": instance_ids,
                "teardown_actions_performed": teardown_actions,
                "runner_gpu_teardown_completed": not continuing_spend,
                "continuing_spend_from_this_run": continuing_spend,
                "zero_continuing_spend_scope": "all Vast instances created by this adapter were destroyed"
                if not continuing_spend
                else "teardown failure requires manual Vast console/API verification",
                "raw_secret_values_recorded": False,
            },
        )
        _append_phase(
            resolved_job_dir,
            "vast_instance_teardown_completed",
            "completed" if not continuing_spend else "blocked",
            blockers=[] if not continuing_spend else ["vast_instance_destroy_failed"],
            proof_effect="vast_instances_destroyed_by_adapter" if not continuing_spend else "none",
            instance_ids=instance_ids,
        )
        ended_at_monotonic = time.monotonic()
        ledger = _budget_ledger(
            job_dir=resolved_job_dir,
            generated_at=utc_now_iso(),
            target_spend_usd=target_spend_usd,
            hard_cap_usd=hard_cap_usd,
            max_hourly_rate=max_hourly_rate,
            max_live_minutes=max_live_minutes,
            selected_offer=selected_offer,
            instance_ids=instance_ids,
            started_at_monotonic=started_at_monotonic,
            ended_at_monotonic=ended_at_monotonic,
            status="completed" if not continuing_spend else "blocked_teardown",
            continuing_spend=continuing_spend,
        )
        estimated_cost_usd = float(ledger["estimated_cost_usd"])
        session_budget_summary = _append_session_budget_attempt(
            budget_path=resolved_session_budget_ledger_path,
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            ledger=ledger,
            selected_offer=selected_offer,
            result_status=_string(base_result.get("status"))
            or "teardown_completed_before_result_classification",
            result_reason=_string(base_result.get("reason")),
            blockers=_string_list(base_result.get("blockers")),
        )
        for signum, previous_handler in previous_signal_handlers.items():
            try:
                signal.signal(signum, previous_handler)
            except (ValueError, OSError, AttributeError):
                pass
        launch_lock_release_manifest = _release_vast_launch_lock(
            launch_lock_handle,
            job_dir=resolved_job_dir,
            generated_at=utc_now_iso(),
        )
        launch_lock_handle = None
        current_blockers = _string_list(base_result.get("blockers"))
        startup_control_plane_blocked = any(
            blocker
            in {
                "vast_heartbeat_blocked",
                "vast_heartbeat_no_log_progress_timeout",
                "vast_heartbeat_container_missing",
                "vast_heartbeat_output_missing_success_marker",
                "vast_probe_interrupted_before_completion",
            }
            or "No such container" in blocker
            for blocker in current_blockers
        )
        if selected_offer and startup_control_plane_blocked:
            avoidlist = _record_machine_avoidlist_entry(
                path=resolved_machine_avoidlist_path,
                generated_at=utc_now_iso(),
                selected_offer=selected_offer,
                instance_id=instance_ids[-1] if instance_ids else None,
                blockers=current_blockers,
                reason="vast_startup_control_plane_did_not_reach_onstart_heartbeat",
            )
            excluded_machine_ids = _machine_id_set(avoidlist.get("machine_ids") or [])
        _append_phase(
            resolved_job_dir,
            "vast_artifacts_exported",
            "completed",
            proof_effect="vast_probe_artifacts_written",
        )
        _ensure_offer_manifest(
            resolved_job_dir,
            generated_at=utc_now_iso(),
            blockers=_string_list(base_result.get("blockers"))
            or ["vast_probe_ended_before_offer_manifest"],
        )
        _fill_missing_phase_rows(
            resolved_job_dir,
            reason=_string(base_result.get("reason")) or "vast_probe_ended_before_phase",
        )
        validation = _final_validation(
            job_dir=resolved_job_dir,
            generated_at=utc_now_iso(),
            instance_ids=instance_ids,
            continuing_spend=continuing_spend,
            estimated_cost_usd=estimated_cost_usd,
            hard_cap_usd=hard_cap_usd,
        )
        if "status" not in base_result or base_result.get("status") not in {"failed", "blocked"}:
            heartbeat = _mapping(
                json.loads(
                    (resolved_job_dir / "vast_startup_probe_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
            gpu = _mapping(
                json.loads(
                    (resolved_job_dir / "vast_gpu_sanity_report.json").read_text(encoding="utf-8")
                )
            )
            provider = _mapping(
                json.loads(
                    (resolved_job_dir / "vast_provider_command_result.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
            video_smoke = _mapping(
                json.loads(
                    (resolved_job_dir / "vast_video_smoke_result.json").read_text(encoding="utf-8")
                )
            )
            provider_blockers = _string_list(provider.get("blockers"))
            video_smoke_blockers = _string_list(video_smoke.get("blockers"))
            provider_status = _string(provider.get("status"))
            expected_video_count = int(_number(video_smoke.get("expected_video_count")) or 0)
            video_smoke_required = enable_blueprint_bundle and expected_video_count > 0
            video_smoke_blocked = video_smoke_required and video_smoke.get("status") != "completed"
            requested_provider_blocked = enable_blueprint_bundle and (
                provider_status != "completed" or video_smoke_blocked
            )
            base_result.update(
                {
                    "status": "blocked"
                    if requested_provider_blocked
                    else ("completed" if heartbeat.get("status") == "completed" else "blocked"),
                    "reason": "vast_blueprint_video_smoke_blocked"
                    if video_smoke_blocked
                    else "vast_blueprint_bundle_blocked"
                    if requested_provider_blocked
                    else (
                        "vast_startup_probe_completed"
                        if heartbeat.get("status") == "completed"
                        else "vast_startup_probe_blocked"
                    ),
                    "blockers": _string_list(heartbeat.get("blockers"))
                    + _string_list(gpu.get("blockers"))
                    + provider_blockers
                    + video_smoke_blockers,
                    "api_call_performed": True,
                    "vast_side_effects_may_have_occurred": bool(instance_ids),
                }
            )
        base_result.update(
            {
                "vast_instance_ids": instance_ids,
                "machine_avoidlist_path": str(resolved_machine_avoidlist_path),
                "excluded_machine_ids": sorted(excluded_machine_ids),
                "session_budget_summary_path": str(resolved_session_budget_ledger_path),
                "session_budget_summary": session_budget_summary,
                "estimated_cost_usd": estimated_cost_usd,
                "continuing_spend_from_this_run": continuing_spend,
                "final_validation_status": validation["status"],
                "vast_launch_lock_status": (
                    launch_lock_release_manifest or launch_lock_manifest
                ).get("status"),
                "vast_launch_lock_acquired": True,
                "vast_launch_lock_manifest": launch_lock_release_manifest or launch_lock_manifest,
                "artifacts": {
                    "vast_runtime_discovery": str(resolved_job_dir / "vast_runtime_discovery.json"),
                    "vast_provider_plan": str(resolved_job_dir / "vast_provider_plan.json"),
                    "vast_offer_selection_manifest": str(
                        resolved_job_dir / "vast_offer_selection_manifest.json"
                    ),
                    "vast_budget_ledger": str(resolved_job_dir / "vast_budget_ledger.json"),
                    "vast_runtime_phase_log": str(
                        resolved_job_dir / "vast_runtime_phase_log.jsonl"
                    ),
                    "vast_startup_probe_manifest": str(
                        resolved_job_dir / "vast_startup_probe_manifest.json"
                    ),
                    "vast_gpu_sanity_report": str(resolved_job_dir / "vast_gpu_sanity_report.json"),
                    "vast_isaac_smoke_result": str(
                        resolved_job_dir / "vast_isaac_smoke_result.json"
                    ),
                    "vast_provider_command_result": str(
                        resolved_job_dir / "vast_provider_command_result.json"
                    ),
                    "vast_video_smoke_result": str(
                        resolved_job_dir / "vast_video_smoke_result.json"
                    ),
                    "vast_teardown_manifest": str(resolved_job_dir / "vast_teardown_manifest.json"),
                    "vast_final_validation": str(resolved_job_dir / "vast_final_validation.json"),
                    "vast_launch_lock_manifest": str(
                        resolved_job_dir / "vast_launch_lock_manifest.json"
                    ),
                    "vast_prelaunch_inventory_guard": str(
                        resolved_job_dir / "vast_prelaunch_inventory_guard.json"
                    ),
                },
            }
        )
        session_budget_summary = _append_session_budget_attempt(
            budget_path=resolved_session_budget_ledger_path,
            job_dir=resolved_job_dir,
            generated_at=generated_at,
            ledger=ledger,
            selected_offer=selected_offer,
            result_status=_string(base_result.get("status")) or "unknown",
            result_reason=_string(base_result.get("reason")),
            blockers=_string_list(base_result.get("blockers")),
        )
        base_result["session_budget_summary"] = session_budget_summary
        write_json(result_path, base_result)
        log_event(
            logger,
            logging.INFO if base_result.get("status") == "completed" else logging.WARNING,
            "vast_provider_adapter.completed",
            job_dir=str(resolved_job_dir),
            status=base_result.get("status"),
            instance_ids=instance_ids,
            continuing_spend=continuing_spend,
            estimated_cost_usd=estimated_cost_usd,
        )
    return base_result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a gated Vast.ai startup probe for Blueprint robot-eval GPU lanes."
    )
    parser.add_argument("--job-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["dry-run", "template-discovery", "live-startup-probe"],
        default="dry-run",
    )
    parser.add_argument(
        "--gpu-selection-policy",
        choices=sorted(GPU_SELECTION_POLICIES),
        default=None,
        help="workload GPU policy; see gpu_selection_policy.GPU_SELECTION_POLICIES",
    )
    parser.add_argument("--max-hourly-rate", type=float, default=DEFAULT_MAX_HOURLY_RATE)
    parser.add_argument("--target-spend-usd", type=float, default=DEFAULT_TARGET_SPEND_USD)
    parser.add_argument("--hard-cap-usd", type=float, default=DEFAULT_HARD_CAP_USD)
    parser.add_argument("--max-live-minutes", type=int, default=DEFAULT_MAX_LIVE_MINUTES)
    parser.add_argument("--public-image", default=DEFAULT_PUBLIC_CUDA_IMAGE)
    parser.add_argument("--isaac-image", default=DEFAULT_ISAAC_IMAGE)
    parser.add_argument("--heartbeat-url", default=DEFAULT_HEARTBEAT_URL)
    parser.add_argument("--previous-job-dir")
    parser.add_argument("--provider-bundle")
    parser.add_argument(
        "--provider-bundle-url",
        help="Provider-fetchable URL for isaac_provider_runtime_bundle.zip; token values are redacted from artifacts.",
    )
    parser.add_argument(
        "--provider-output-put-url",
        help="Provider-writable PUT URL for vast_provider_runtime_output.zip; token values are redacted from artifacts.",
    )
    parser.add_argument(
        "--provider-output-get-url",
        help="Provider-readable GET URL for downloading the uploaded runtime output zip; token values are redacted from artifacts.",
    )
    parser.add_argument(
        "--provider-runtime-output-zip",
        help="Local path expected to contain the uploaded provider runtime output zip.",
    )
    parser.add_argument("--enable-isaac-smoke", action="store_true")
    parser.add_argument("--enable-blueprint-bundle", action="store_true")
    parser.add_argument(
        "--provider-bundle-kind",
        choices=VAST_PROVIDER_BUNDLE_KINDS,
        default="isaac",
        help="Provider bundle runtime contract to execute. Defaults to the existing Isaac path.",
    )
    parser.add_argument(
        "--vast-launch-mode",
        choices=VAST_LAUNCH_MODES,
        default=DEFAULT_VAST_LAUNCH_MODE,
        help="Use auto to select args for Isaac smoke and ssh_direct otherwise.",
    )
    parser.add_argument(
        "--ngc-image-login-mode",
        choices=NGC_IMAGE_LOGIN_MODES,
        default=os.getenv(VAST_IMAGE_LOGIN_MODE_ENV, DEFAULT_NGC_IMAGE_LOGIN_MODE),
        help="Use auto to avoid login for the official public Isaac image; always forces NGC credentials.",
    )
    parser.add_argument(
        "--vast-template-hash-id",
        help=(
            "Optional Vast template hash for launch configuration reuse. "
            "A template hash alone is not image-cache or prewarm proof."
        ),
    )
    parser.add_argument(
        "--use-vast-template-image",
        action="store_true",
        help="Omit the direct image override and use the image configured on --vast-template-hash-id.",
    )
    parser.add_argument(
        "--allow-cold-isaac-image-pull",
        action="store_true",
        dest="allow_cold_isaac_image_pull",
        help="Allow direct cold pulls of the official Isaac image. The authorized wrapper disables this by default.",
    )
    parser.add_argument(
        "--block-cold-isaac-image-pull",
        action="store_false",
        dest="allow_cold_isaac_image_pull",
        help="Block paid live probes that would directly cold-pull the official Isaac image.",
    )
    parser.set_defaults(allow_cold_isaac_image_pull=True)
    parser.add_argument(
        "--min-cold-isaac-pull-live-minutes",
        type=int,
        default=DEFAULT_MIN_COLD_ISAAC_PULL_LIVE_MINUTES,
        help="Minimum live window required when allowing a direct cold pull of the official Isaac image.",
    )
    parser.add_argument(
        "--disk-gb",
        type=int,
        help=f"Override Vast disk GB. Defaults to {DEFAULT_ISAAC_DISK_GB} for Isaac smoke and {DEFAULT_PUBLIC_DISK_GB} otherwise.",
    )
    parser.add_argument("--poll-interval-seconds", type=int, default=10)
    parser.add_argument("--startup-timeout-seconds", type=int, default=420)
    parser.add_argument(
        "--heartbeat-no-progress-seconds",
        type=int,
        default=None,
        help=(
            "Maximum seconds to wait with no onstart/request_logs progress before "
            f"blocking startup. Defaults to {VAST_HEARTBEAT_NO_PROGRESS_SECONDS_ENV} "
            f"or {DEFAULT_HEARTBEAT_NO_PROGRESS_SECONDS}."
        ),
    )
    parser.add_argument(
        "--machine-avoidlist",
        help="Optional JSON avoidlist of Vast machine IDs to exclude from offer selection. Defaults to <job-dir>/vast_machine_avoidlist.json.",
    )
    parser.add_argument(
        "--allowed-machine-id",
        action="append",
        default=[],
        help=(
            "Restrict offer selection to this Vast machine ID. Can be repeated; "
            "use after a host-specific canary has passed."
        ),
    )
    parser.add_argument(
        "--session-budget-ledger",
        help=(
            "Optional session cost summary JSON used to block paid launches before Vast API calls. "
            f"Defaults to {DEFAULT_VAST_SESSION_BUDGET_FILENAME} beside {VAST_API_KEY_FILE_ENV}; "
            f"{VAST_SESSION_BUDGET_LEDGER_FILE_ENV} can override the default."
        ),
    )
    parser.add_argument(
        "--vast-launch-lock-file",
        help=(
            "Optional single-flight lock file for paid Vast launches. Defaults to "
            "vast_paid_launch.lock beside VAST_API_KEY_FILE."
        ),
    )
    parser.add_argument(
        "--session-max-live-minutes",
        type=int,
        default=DEFAULT_SESSION_MAX_LIVE_MINUTES,
        help=f"Maximum cumulative live Vast runtime allowed for this session. Defaults to {DEFAULT_SESSION_MAX_LIVE_MINUTES}.",
    )
    parser.add_argument(
        "--verify-staging-urls",
        action="store_true",
        help="Verify provider bundle URL reachability before Vast offer search.",
    )
    parser.add_argument(
        "--allow-staging-output-put-probe",
        action="store_true",
        help="Allow a small pre-allocation PUT probe to the provider output URL. This is intentionally opt-in because some signed URLs are one-shot or overwrite targets.",
    )
    parser.add_argument(
        "--require-known-supported-isaac-driver",
        action="store_true",
        help="Exclude offers in the known unsupported Omniverse RTX driver range; recommended for Blueprint bundle/video proof.",
    )
    parser.add_argument(
        "--allow-vast-api-call",
        action="store_true",
        help=f"Required with {VAST_API_GATE_ENV}=true for live Vast API calls.",
    )
    parser.add_argument(
        "--allow-vast-instance-launch",
        action="store_true",
        help=f"Required with {VAST_INSTANCE_LAUNCH_GATE_ENV}=true for paid Vast instance launch.",
    )
    args = parser.parse_args(argv)
    if args.mode == "live-startup-probe":
        print("legacy_vast_provider_mutation_cli_disabled", file=sys.stderr)
        return 2
    adapter_kwargs = vars(args).copy()
    for cli_name, adapter_name in (
        ("allow_vast_instance_launch", "allow_instance_launch"),
        ("machine_avoidlist", "machine_avoidlist_path"),
        ("allowed_machine_id", "allowed_machine_ids"),
        ("session_budget_ledger", "session_budget_ledger_path"),
    ):
        adapter_kwargs[adapter_name] = adapter_kwargs.pop(cli_name)
    result = run_vast_provider_adapter(**adapter_kwargs)
    print(
        f"[vast-provider-adapter] result={Path(args.job_dir).resolve() / 'vast_provider_adapter_result.json'}"
    )
    print(f"[vast-provider-adapter] status={result.get('status')}")
    print(
        f"[vast-provider-adapter] instance_ids={','.join(str(item) for item in result.get('vast_instance_ids', []))}"
    )
    blockers = _string_list(result.get("blockers"))
    if blockers:
        print("[vast-provider-adapter] blockers=" + ",".join(blockers))
    return 0 if result.get("status") in {"completed", "dry_run_ready"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
