"""RunPod adapter for prepared robot-eval GPU provider launch requests."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .logging_utils import log_event
from .provider_worker_endpoint_manifest import write_provider_worker_endpoint_manifest


RUNPOD_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION = "runpod_provider_adapter_result.v1"
RUNPOD_API_KEY_ENV = "RUNPOD_API_KEY"
RUNPOD_API_KEY_FILE_ENV = "RUNPOD_API_KEY_FILE"
RUNPOD_CONFIG_FILE_ENV = "RUNPOD_CONFIG_FILE"
DEFAULT_RUNPOD_CONFIG_FILE = "~/.runpod/config.toml"
SIGNED_URL_SIGNATURE_PARAM = "x-goog-" + "signature="
RUNPOD_ENDPOINT_ID_ENV = "BLUEPRINT_RUNPOD_ENDPOINT_ID"
RUNPOD_API_GATE_ENV = "BLUEPRINT_ALLOW_RUNPOD_API_CALLS"
RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"
RUNPOD_REST_API_BASE_ENV = "RUNPOD_REST_API_BASE"
RUNPOD_REST_API_BASE = os.getenv(RUNPOD_REST_API_BASE_ENV, "https://rest.runpod.io/v1").rstrip("/")
RUNPOD_SERVERLESS_API_BASE = "https://api.runpod.ai/v2"
RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV = "BLUEPRINT_RUNPOD_CONTAINER_REGISTRY_AUTH_ID"
PROVIDER_LAUNCH_REQUEST_ENV = "BLUEPRINT_GPU_PROVIDER_LAUNCH_REQUEST"
PROVIDER_ADAPTER_OUTPUT_ENV = "BLUEPRINT_GPU_PROVIDER_ADAPTER_OUTPUT"
RUNPOD_FORWARD_SECRET_ENV_VARS_ENV = "BLUEPRINT_RUNPOD_FORWARD_SECRET_ENV_VARS"
GENERIC_WORKER_IMAGE_REF_ENV = "BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF"
SENSITIVE_ENV_NAME_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
logger = logging.getLogger(__name__)
WORKER_IMAGE_REF_ENV_BY_SIMULATOR = {
    "isaac_sim": "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
    "isaac_lab_arena": "BLUEPRINT_ISAAC_ARENA_EVAL_WORKER_IMAGE_REF",
    "mujoco": "BLUEPRINT_MUJOCO_EVAL_WORKER_IMAGE_REF",
    "pybullet": "BLUEPRINT_PYBULLET_EVAL_WORKER_IMAGE_REF",
    "newton": "BLUEPRINT_NEWTON_EVAL_WORKER_IMAGE_REF",
}
SIGNED_URL_QUERY_PATTERN = re.compile(
    rf"([?&]){SIGNED_URL_SIGNATURE_PARAM}[^\s\"'&]+",
    flags=re.IGNORECASE,
)
SIGNED_URL_SIGNATURE_REPLACEMENT = "x-goog-redacted-signature-param=<redacted:signed-url-signature>"


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


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


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


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


def _read_runpod_api_key() -> tuple[str, dict[str, Any]]:
    env_value = _string(os.getenv(RUNPOD_API_KEY_ENV))
    if env_value:
        return env_value, {
            "api_key_configured": True,
            "api_key_source": RUNPOD_API_KEY_ENV,
            "api_key_file_configured": False,
        }
    key_file = _string(os.getenv(RUNPOD_API_KEY_FILE_ENV))
    if key_file:
        try:
            key = Path(key_file).expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:
            return "", {
                "api_key_configured": False,
                "api_key_source": RUNPOD_API_KEY_FILE_ENV,
                "api_key_file_configured": True,
                "api_key_file_read_error": type(exc).__name__,
            }
        return key, {
            "api_key_configured": bool(key),
            "api_key_source": RUNPOD_API_KEY_FILE_ENV if key else None,
            "api_key_file_configured": True,
        }

    config_file = Path(
        _string(os.getenv(RUNPOD_CONFIG_FILE_ENV)) or DEFAULT_RUNPOD_CONFIG_FILE
    ).expanduser()
    if not config_file.is_file():
        return "", {
            "api_key_configured": False,
            "api_key_source": None,
            "api_key_file_configured": False,
            "api_key_config_file": str(config_file),
            "api_key_config_file_configured": False,
        }
    try:
        payload = tomllib.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return "", {
            "api_key_configured": False,
            "api_key_source": RUNPOD_CONFIG_FILE_ENV,
            "api_key_file_configured": False,
            "api_key_config_file": str(config_file),
            "api_key_config_file_configured": True,
            "api_key_config_file_read_error": type(exc).__name__,
        }
    default_section = _mapping(payload.get("default"))
    key = _string(default_section.get("api_key") or payload.get("api_key"))
    return key, {
        "api_key_configured": bool(key),
        "api_key_source": RUNPOD_CONFIG_FILE_ENV if key else None,
        "api_key_file_configured": False,
        "api_key_config_file": str(config_file),
        "api_key_config_file_configured": True,
    }


def _provider_shape(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(request.get("provider_request_shape"))


def _limits(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(_provider_shape(request).get("limits"))


def _inputs(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(_provider_shape(request).get("inputs"))


def _image(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(_provider_shape(request).get("image"))


def _gpu(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(_provider_shape(request).get("gpu"))


def _cache(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(_provider_shape(request).get("cache"))


def _environment(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(_provider_shape(request).get("environment"))


def _timeout_ms(seconds: Any, default_seconds: int = 600) -> int:
    value = _number(seconds)
    return int((value if value and value > 0 else default_seconds) * 1000)


def _cost_control_policy(request: Mapping[str, Any]) -> Dict[str, Any]:
    limits = _limits(request)
    warm_pool = _mapping(limits.get("warm_pool_policy"))
    hard_timeout_seconds = int(_number(limits.get("hard_timeout_seconds")) or 600)
    idle_timeout_seconds = int(_number(limits.get("idle_timeout_seconds")) or 60)
    watchdog_ttl_seconds = int(
        _number(limits.get("external_watchdog_ttl_seconds")) or max(900, hard_timeout_seconds)
    )
    max_active_workers = int(_number(limits.get("max_active_workers")) or 1)
    return {
        "source": "gpu_provider_launch_request.provider_request_shape.limits",
        "hard_timeout_seconds": hard_timeout_seconds,
        "idle_timeout_seconds": idle_timeout_seconds,
        "external_watchdog_ttl_seconds": watchdog_ttl_seconds,
        "max_active_workers": max_active_workers,
        "warm_pool_policy": {
            "decision": warm_pool.get("decision") or "scale_to_zero_on_demand",
            "active_worker_target": int(_number(limits.get("active_worker_target")) or 0),
            "warm_worker_recommended": bool(warm_pool.get("warm_worker_recommended")),
            "scale_to_zero_default": bool(
                limits.get("scale_to_zero_default") is not False
                and not warm_pool.get("warm_worker_recommended")
            ),
            "decision_reasons": _string_list(warm_pool.get("decision_reasons")),
        },
        "serverless_endpoint_controls": {
            "per_request_policy_fields": ["executionTimeout", "ttl", "lowPriority"],
            "endpoint_level_settings_required": [
                "active_workers",
                "max_workers",
                "idle_timeout",
                "execution_timeout",
                "job_ttl",
            ],
            "idle_timeout_set_by_run_request": False,
            "max_workers_set_by_run_request": False,
            "recommended_active_workers": 0
            if limits.get("scale_to_zero_default") is not False
            else 1,
            "recommended_max_workers": max_active_workers,
            "recommended_idle_timeout_seconds": idle_timeout_seconds,
        },
        "on_demand_pod_controls": {
            "pod_idle_timeout_is_not_provider_native": True,
            "external_watchdog_or_owner_terminator_required": True,
            "external_watchdog_owner": _string(limits.get("external_watchdog_owner"))
            or "provider_launcher_or_owner_control_plane",
            "worker_env_shutdown_controls": [
                "BLUEPRINT_GPU_PROVIDER_HARD_TIMEOUT_SECONDS",
                "BLUEPRINT_GPU_PROVIDER_IDLE_TIMEOUT_SECONDS",
                "BLUEPRINT_GPU_PROVIDER_EXTERNAL_WATCHDOG_TTL_SECONDS",
            ],
        },
        "proof_boundary": {
            "policy_documents_cost_controls_only": True,
            "provider_idle_shutdown_configured": False,
            "provider_allocation_proven": False,
            "simulator_execution_proven": False,
        },
    }


def _redact_signed_url_text(text: str) -> str:
    if SIGNED_URL_SIGNATURE_PARAM not in text.lower():
        return text
    return SIGNED_URL_QUERY_PATTERN.sub(rf"\1{SIGNED_URL_SIGNATURE_REPLACEMENT}", text)


def _redact_text(text: str, api_key: str) -> str:
    redacted = text.replace(api_key, "<redacted:RUNPOD_API_KEY>") if api_key else text
    return _redact_signed_url_text(redacted)


def _redact_runtime_value(value: Any) -> Any:
    if isinstance(value, str):
        if SIGNED_URL_SIGNATURE_PARAM in value.lower():
            return _redact_signed_url_text(value)
        return value
    if isinstance(value, list):
        return [_redact_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_runtime_value(item) for item in value]
    if isinstance(value, Mapping):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if (
                isinstance(item, str)
                and item
                and any(marker in key_text.upper() for marker in SENSITIVE_ENV_NAME_MARKERS)
            ):
                redacted[key_text] = "<redacted:secret-env>"
            elif (
                key_text == "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"
                and isinstance(item, str)
                and item
            ):
                redacted[key_text] = "<redacted:signed-url>"
            else:
                redacted[key_text] = _redact_runtime_value(item)
        return redacted
    return value


def _request_summary(request: Mapping[str, Any]) -> Dict[str, Any]:
    inputs = _inputs(request)
    image = _image(request)
    limits = _limits(request)
    return {
        "job_id": _string(request.get("job_id")),
        "provider": _string(request.get("provider")),
        "provider_launch_request_status": _string(request.get("status")),
        "operation": _string(request.get("operation"))
        or _string(_provider_shape(request).get("operation")),
        "worker_image_ref_present": bool(_string(image.get("configured_image_ref"))),
        "worker_image_ref_is_versioned": image.get("configured_image_ref_is_versioned")
        is True,
        "worker_image_ref_fetchable_by_provider": image.get(
            "configured_image_ref_fetchable_by_provider"
        )
        is not False,
        "container_registry_auth_id_present": bool(
            _string(image.get("container_registry_auth_id"))
            or _string(os.getenv(RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV))
        ),
        "manifest_uri_present": bool(_string(inputs.get("manifest_uri"))),
        "manifest_uri_fetchable_by_provider": inputs.get("manifest_uri_fetchable_by_provider")
        is True,
        "artifact_output_uri_present": bool(_string(inputs.get("artifact_output_uri"))),
        "hard_timeout_seconds": limits.get("hard_timeout_seconds"),
        "idle_timeout_seconds": limits.get("idle_timeout_seconds"),
        "external_watchdog_ttl_seconds": limits.get("external_watchdog_ttl_seconds"),
        "max_active_workers": limits.get("max_active_workers"),
    }


def _base_result(
    *,
    request_path: Path,
    output_path: Path,
    request: Mapping[str, Any],
    mode: str,
) -> Dict[str, Any]:
    return {
        "schema_version": RUNPOD_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "provider_launch_request_path": str(request_path),
        "output_path": str(output_path),
        "mode": mode,
        "job_id": _string(request.get("job_id")),
        "provider": _string(request.get("provider")) or "runpod",
        "api_call_performed": False,
        "runpod_side_effects_may_have_occurred": False,
        "live_provider_call_proven": False,
        "provider_allocation_proven": False,
        "provider_job_submitted": False,
        "secret_values_in_artifact": False,
        "raw_api_key_stored": False,
        "signed_url_values_in_artifact": False,
        "simulator_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
        "request_summary": _request_summary(request),
        "cost_control_policy": _cost_control_policy(request),
    }


def _adapter_event_name(status: str) -> str:
    if status == "blocked":
        return "runpod_provider_adapter.blocked"
    if status == "failed":
        return "runpod_provider_adapter.failed"
    return "runpod_provider_adapter.completed"


def _persist_result(output_path: Path, result: Mapping[str, Any]) -> Dict[str, Any]:
    persisted = dict(result)
    write_json(output_path, persisted)
    blockers = _string_list(persisted.get("blockers"))
    status = _string(persisted.get("status"))
    log_event(
        logger,
        logging.WARNING if status in {"blocked", "failed"} else logging.INFO,
        _adapter_event_name(status),
        output_path=str(output_path),
        provider_launch_request_path=persisted.get("provider_launch_request_path"),
        job_id=persisted.get("job_id"),
        provider=persisted.get("provider"),
        mode=persisted.get("mode"),
        status=status,
        reason=persisted.get("reason"),
        blocker_count=len(blockers),
        blockers=blockers,
        api_call_performed=persisted.get("api_call_performed"),
        runpod_side_effects_may_have_occurred=persisted.get(
            "runpod_side_effects_may_have_occurred"
        ),
        http_status_code=persisted.get("http_status_code"),
        provider_job_submitted=persisted.get("provider_job_submitted"),
    )
    return persisted


def _serverless_payload(
    request: Mapping[str, Any],
    *,
    endpoint_id: str,
) -> Dict[str, Any]:
    inputs = _inputs(request)
    limits = _limits(request)
    cost_control = _cost_control_policy(request)
    hard_timeout = _timeout_ms(limits.get("hard_timeout_seconds"))
    ttl = _timeout_ms(limits.get("external_watchdog_ttl_seconds"), default_seconds=900)
    return {
        "url": f"{RUNPOD_SERVERLESS_API_BASE}/{endpoint_id}/run",
        "method": "POST",
        "body": {
            "input": {
                "job_id": _string(request.get("job_id")),
                "worker_manifest_uri": _string(inputs.get("manifest_uri")),
                "artifact_output_uri": _string(inputs.get("artifact_output_uri")),
                "provider_launch_request_status": _string(request.get("status")),
                "cost_control_policy": {
                    "hard_timeout_seconds": cost_control["hard_timeout_seconds"],
                    "idle_timeout_seconds": cost_control["idle_timeout_seconds"],
                    "external_watchdog_ttl_seconds": cost_control[
                        "external_watchdog_ttl_seconds"
                    ],
                    "serverless_idle_timeout_requires_endpoint_setting": True,
                },
            },
            "policy": {
                "executionTimeout": hard_timeout,
                "ttl": max(ttl, hard_timeout),
                "lowPriority": False,
            },
        },
    }


def _pod_env(request: Mapping[str, Any]) -> list[dict[str, str]]:
    inputs = _inputs(request)
    limits = _limits(request)
    image_ref = _string(_image(request).get("configured_image_ref"))
    environment = _environment(request)
    runtime_preflight = _mapping(_provider_shape(request).get("runtime_preflight"))
    artifact_output_required = _bool(inputs.get("artifact_output_uri_required"))
    simulator = (
        _string(runtime_preflight.get("simulator"))
        or _string(request.get("simulator"))
        or _string(inputs.get("simulator"))
    )
    env = {
        "BLUEPRINT_EVAL_MANIFEST_URI": _string(inputs.get("manifest_uri")),
        "BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI": _string(inputs.get("capture_root_bundle_uri")),
        "BLUEPRINT_ROBOT_EVAL_JOB_ID": _string(request.get("job_id")),
        "BLUEPRINT_ROBOT_EVAL_PROVIDER_RUNTIME": "true",
        "BLUEPRINT_GPU_PROVIDER_HARD_TIMEOUT_SECONDS": str(
            int(_number(limits.get("hard_timeout_seconds")) or 600)
        ),
        "BLUEPRINT_GPU_PROVIDER_IDLE_TIMEOUT_SECONDS": str(
            int(_number(limits.get("idle_timeout_seconds")) or 60)
        ),
        "BLUEPRINT_GPU_PROVIDER_EXTERNAL_WATCHDOG_TTL_SECONDS": str(
            int(_number(limits.get("external_watchdog_ttl_seconds")) or 900)
        ),
        "NVIDIA_DRIVER_CAPABILITIES": os.getenv(
            "BLUEPRINT_RUNPOD_NVIDIA_DRIVER_CAPABILITIES",
            "all",
        ),
    }
    if artifact_output_required is not False:
        env["BLUEPRINT_ARTIFACT_OUTPUT_URI"] = _string(inputs.get("artifact_output_uri"))
    signed_put_url = _string(os.getenv("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"))
    if signed_put_url:
        env["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"] = signed_put_url
    if _env_truthy("BLUEPRINT_ALLOW_GPU_PROVISIONING"):
        env["BLUEPRINT_ALLOW_GPU_PROVISIONING"] = "true"
    if _env_truthy("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"):
        env["BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"] = "true"
    if image_ref:
        env[GENERIC_WORKER_IMAGE_REF_ENV] = image_ref
        simulator_image_env = WORKER_IMAGE_REF_ENV_BY_SIMULATOR.get(simulator)
    if simulator_image_env:
        env[simulator_image_env] = image_ref
    plaintext_names = set(_string_list(environment.get("plaintext_env_var_names")))
    secret_names = set(_string_list(environment.get("secret_env_var_names")))
    plaintext_values = _mapping(
        environment.get("plaintext_env_values") or environment.get("plaintext_env")
    )
    for key, value in plaintext_values.items():
        env_key = _string(key)
        env_value = _string(value)
        if not env_key or not env_value or env_key in secret_names:
            continue
        if plaintext_names and env_key not in plaintext_names:
            continue
        if SIGNED_URL_SIGNATURE_PARAM in env_value.lower():
            continue
        env[env_key] = env_value
    forward_secret_names = _dedupe(
        name.strip()
        for name in _string(os.getenv(RUNPOD_FORWARD_SECRET_ENV_VARS_ENV)).split(",")
    )
    for env_key in forward_secret_names:
        if not any(marker in env_key.upper() for marker in SENSITIVE_ENV_NAME_MARKERS):
            continue
        env_value = os.getenv(env_key)
        if env_value:
            env[env_key] = env_value
    cache_paths = _mapping(_cache(request).get("paths"))
    cache_env_by_key = {
        "mujoco_assets": "BLUEPRINT_MUJOCO_ASSET_CACHE",
        "policy_files": "BLUEPRINT_POLICY_CACHE",
        "converted_scenes": "BLUEPRINT_CONVERTED_SCENE_CACHE",
        "worker_deps": "BLUEPRINT_WORKER_DEPS_CACHE",
    }
    for cache_key, env_name in cache_env_by_key.items():
        value = _string(cache_paths.get(cache_key))
        if value:
            env[env_name] = value
    return [{"key": key, "value": value} for key, value in env.items() if value]


def _pod_payload(
    request: Mapping[str, Any],
    *,
    pod_name: str | None = None,
    gpu_type_id: str | None = None,
) -> Dict[str, Any]:
    image = _image(request)
    gpu = _gpu(request)
    limits = _limits(request)
    priority = _string_list(
        gpu.get("provider_gpu_priority")
        or gpu.get("priority_fallback_list")
        or gpu.get("gpu_type_priority")
    )
    image_ref = _string(image.get("configured_image_ref"))
    container_registry_auth_id = _string(
        image.get("container_registry_auth_id")
        or _provider_shape(request).get("container_registry_auth_id")
        or os.getenv(RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV)
    )
    selected_gpu = (
        _string(gpu_type_id)
        or _string(gpu.get("preferred_gpu_type_id"))
        or (priority[0] if priority else "")
        or _string(gpu.get("preferred_gpu_class"))
        or "NVIDIA RTX A6000"
    )
    name = _string(pod_name) or f"blueprint-robot-eval-{_string(request.get('job_id'))}"
    env = {item["key"]: item["value"] for item in _pod_env(request)}
    provider_shape = _provider_shape(request)
    command = _string(provider_shape.get("command"))
    docker_entrypoint = _string_list(
        provider_shape.get("docker_entrypoint") or provider_shape.get("dockerEntrypoint")
    )
    docker_start_cmd_override = _string_list(
        provider_shape.get("docker_start_cmd") or provider_shape.get("dockerStartCmd")
    )
    start_cmd = []
    if docker_start_cmd_override:
        start_cmd = docker_start_cmd_override
    elif command:
        if docker_entrypoint:
            start_cmd = [command]
        else:
            parts = shlex.split(command)
            start_cmd = parts[1:] if parts and parts[0] == "blueprint-run-robot-eval-worker" else parts
    if start_cmd == ["--manifest", "${BLUEPRINT_EVAL_MANIFEST_URI}"]:
            start_cmd = []
    input_payload = {
        "cloudType": _string(os.getenv("BLUEPRINT_RUNPOD_CLOUD_TYPE")) or "SECURE",
        "computeType": "GPU",
        "gpuCount": int(_number(gpu.get("gpu_count")) or 1),
        "gpuTypeIds": [selected_gpu],
        "gpuTypePriority": "availability",
        "volumeInGb": int(_number(gpu.get("volume_in_gb")) or 40),
        "containerDiskInGb": int(_number(gpu.get("container_disk_in_gb")) or 60),
        "minVCPUPerGPU": int(_number(gpu.get("min_vcpu_count")) or 4),
        "minRAMPerGPU": int(_number(gpu.get("min_memory_in_gb")) or 16),
        "name": name,
        "imageName": image_ref,
        "dockerStartCmd": start_cmd,
        "ports": [],
        "volumeMountPath": "/workspace",
        "env": env,
    }
    if docker_entrypoint:
        input_payload["dockerEntrypoint"] = docker_entrypoint
    if container_registry_auth_id:
        input_payload["containerRegistryAuthId"] = container_registry_auth_id
    return {
        "url": f"{RUNPOD_REST_API_BASE}/pods",
        "method": "POST",
        "body": input_payload,
        "api_surface": "rest_pods",
        "idle_shutdown_expected_seconds": limits.get("idle_timeout_seconds"),
    }


def _request_blockers(
    *,
    request: Mapping[str, Any],
    mode: str,
    endpoint_id: str,
) -> list[str]:
    blockers: list[str] = []
    limits = _limits(request)
    hard_timeout_seconds = _number(limits.get("hard_timeout_seconds"))
    idle_timeout_seconds = _number(limits.get("idle_timeout_seconds"))
    watchdog_ttl_seconds = _number(limits.get("external_watchdog_ttl_seconds"))
    max_active_workers = _number(limits.get("max_active_workers"))
    if request.get("schema_version") != "robot_eval_gpu_provider_launch_request.v1":
        blockers.append("invalid_provider_launch_request_schema")
    if _string(request.get("provider")) != "runpod":
        blockers.append("provider_launch_request_not_runpod")
    if request.get("status") != "request_manifest_ready":
        blockers.append("provider_launch_request_not_ready")
    provider_input_setup = request.get("provider_input_setup")
    if isinstance(provider_input_setup, Mapping):
        setup_blockers = _string_list(provider_input_setup.get("blockers"))
        if setup_blockers:
            blockers.append("provider_input_setup_blocked")
            blockers.extend(setup_blockers)
    inputs = _inputs(request)
    image = _image(request)
    artifact_output_uri = _string(inputs.get("artifact_output_uri"))
    artifact_output_required = _bool(inputs.get("artifact_output_uri_required"))
    artifact_output_scheme = urlparse(artifact_output_uri).scheme or "local"
    signed_put_url = _string(os.getenv("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"))
    if mode == "serverless-run" and not endpoint_id:
        blockers.append(f"missing_env_{RUNPOD_ENDPOINT_ID_ENV}")
    if mode == "on-demand-pod" and not _string(image.get("configured_image_ref")):
        blockers.append("missing_provider_worker_image_ref")
    if _string(image.get("configured_image_ref")) and (
        image.get("configured_image_ref_is_versioned") is not True
    ):
        blockers.append("prebuilt_worker_image_ref_not_versioned")
    if image.get("configured_image_ref_fetchable_by_provider") is False:
        blockers.append("prebuilt_worker_image_ref_not_provider_fetchable")
    if not _string(inputs.get("manifest_uri")):
        blockers.append("missing_provider_worker_manifest_uri")
    if inputs.get("manifest_uri_fetchable_by_provider") is not True:
        blockers.append("provider_worker_manifest_uri_not_fetchable")
    if not artifact_output_uri and artifact_output_required is not False:
        blockers.append("missing_provider_artifact_output_uri")
    if artifact_output_required is False and not signed_put_url:
        blockers.append("missing_runtime_manifest_signed_put_url_for_artifact_output_optional")
    if artifact_output_uri and artifact_output_scheme not in {"gs", "s3", "r2", "file", "local"}:
        blockers.append("provider_artifact_output_uri_not_writable")
    if not hard_timeout_seconds or hard_timeout_seconds <= 0:
        blockers.append("missing_provider_hard_timeout_seconds")
    if not idle_timeout_seconds or idle_timeout_seconds <= 0:
        blockers.append("missing_provider_idle_timeout_seconds")
    if not watchdog_ttl_seconds or watchdog_ttl_seconds <= 0:
        blockers.append("missing_provider_external_watchdog_ttl_seconds")
    elif hard_timeout_seconds and watchdog_ttl_seconds <= hard_timeout_seconds:
        blockers.append("provider_external_watchdog_ttl_must_exceed_hard_timeout")
    if not max_active_workers or max_active_workers <= 0:
        blockers.append("missing_provider_max_active_workers")
    if _environment(request).get("secret_values_in_artifact") is not False:
        blockers.append("provider_launch_request_secret_values_in_artifact")
    return _dedupe(blockers)


def _api_gate_blockers(*, allow_runpod_api_call: bool, api_key: str) -> list[str]:
    blockers: list[str] = []
    if not _env_truthy(RUNPOD_API_GATE_ENV):
        blockers.append(f"missing_env_{RUNPOD_API_GATE_ENV}")
    if not allow_runpod_api_call:
        blockers.append("missing_cli_allow_runpod_api_call")
    if not api_key:
        blockers.append(
            f"missing_env_{RUNPOD_API_KEY_ENV}_or_{RUNPOD_API_KEY_FILE_ENV}_or_{RUNPOD_CONFIG_FILE_ENV}"
        )
    return blockers


def _http_json(
    *,
    url: str,
    payload: Mapping[str, Any],
    api_key: str,
    timeout_seconds: int,
) -> tuple[int, Dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
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


def run_runpod_provider_adapter(
    *,
    provider_launch_request_path: str | Path,
    output_path: str | Path | None = None,
    mode: str = "dry-run",
    allow_runpod_api_call: bool = False,
    endpoint_id: str | None = None,
    pod_name: str | None = None,
    gpu_type_id: str | None = None,
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    request_path = Path(provider_launch_request_path).resolve()
    resolved_output = (
        Path(output_path).resolve()
        if output_path
        else Path(
            os.getenv(PROVIDER_ADAPTER_OUTPUT_ENV)
            or request_path.parent / "runpod_provider_adapter_result.json"
        ).resolve()
    )
    ensure_dir(resolved_output.parent)
    payload = read_json_any(request_path)
    request = dict(payload) if isinstance(payload, Mapping) else {}
    result = _base_result(
        request_path=request_path,
        output_path=resolved_output,
        request=request,
        mode=mode,
    )
    log_event(
        logger,
        logging.INFO,
        "runpod_provider_adapter.started",
        provider_launch_request_path=str(request_path),
        output_path=str(resolved_output),
        job_id=request.get("job_id"),
        provider=request.get("provider"),
        mode=mode,
        allow_runpod_api_call=allow_runpod_api_call,
    )
    if not request:
        result.update(
            {
                "status": "blocked",
                "reason": "invalid_provider_launch_request_json",
                "blockers": ["invalid_provider_launch_request_json"],
            }
        )
        return _persist_result(resolved_output, result)

    api_key, api_key_meta = _read_runpod_api_key()
    result.update(api_key_meta)
    selected_endpoint_id = _string(endpoint_id) or _string(os.getenv(RUNPOD_ENDPOINT_ID_ENV))
    if mode == "auto":
        mode = "serverless-run" if selected_endpoint_id else "on-demand-pod"
        result["mode"] = mode
    endpoint_manifest_mode = (
        "serverless-run" if mode == "dry-run" and selected_endpoint_id else mode
    )
    provider_worker_endpoint_manifest = write_provider_worker_endpoint_manifest(
        output_dir=request_path.parent,
        provider="runpod",
        mode=endpoint_manifest_mode,
        job_id=_string(request.get("job_id")),
        serverless_endpoint_id=selected_endpoint_id,
        provider_request_shape=_provider_shape(request),
    )
    result.update(
        {
            "provider_worker_endpoint_manifest_path": str(
                request_path.parent / "provider_worker_endpoint_manifest.json"
            ),
            "provider_worker_endpoint_manifest": provider_worker_endpoint_manifest,
        }
    )
    request_blockers = _request_blockers(
        request=request,
        mode=mode,
        endpoint_id=selected_endpoint_id,
    )
    if mode == "serverless-run":
        runpod_request = _serverless_payload(request, endpoint_id=selected_endpoint_id)
    elif mode == "on-demand-pod":
        runpod_request = _pod_payload(
            request,
            pod_name=pod_name,
            gpu_type_id=gpu_type_id,
        )
    elif mode == "dry-run":
        runpod_request = {
            "serverless_run": _serverless_payload(
                request,
                endpoint_id=selected_endpoint_id or "<set-BLUEPRINT_RUNPOD_ENDPOINT_ID>",
            ),
            "on_demand_pod": _pod_payload(
                request,
                pod_name=pod_name,
                gpu_type_id=gpu_type_id,
            ),
        }
    else:
        request_blockers.append(f"unsupported_runpod_adapter_mode:{mode}")
        runpod_request = {}

    result["runpod_request"] = _redact_runtime_value(runpod_request)
    result["request_blockers"] = request_blockers
    if request_blockers:
        result.update(
            {
                "status": "blocked",
                "reason": "runpod_request_not_launchable",
                "blockers": request_blockers,
            }
        )
        return _persist_result(resolved_output, result)

    if mode == "dry-run":
        result.update(
            {
                "status": "dry_run_ready",
                "reason": "runpod_request_shape_validated_without_api_call",
                "blockers": [],
            }
        )
        return _persist_result(resolved_output, result)

    gate_blockers = _api_gate_blockers(
        allow_runpod_api_call=allow_runpod_api_call,
        api_key=api_key,
    )
    if gate_blockers:
        result.update(
            {
                "status": "blocked",
                "reason": "runpod_api_gate_blocked",
                "blockers": gate_blockers,
                **api_key_meta,
            }
        )
        return _persist_result(resolved_output, result)

    try:
        if mode == "serverless-run":
            status_code, response = _http_json(
                url=str(runpod_request["url"]),
                payload=dict(runpod_request["body"]),
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
        else:
            status_code, response = _http_json(
                url=str(runpod_request["url"]),
                payload=dict(runpod_request["body"]),
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
        response_text = _redact_text(json.dumps(response, sort_keys=True), api_key)
        redacted_response = _redact_runtime_value(json.loads(response_text))
        result.update(
            {
                "status": "submitted",
                "reason": "runpod_api_call_completed",
                "blockers": [],
                "api_call_performed": True,
                "runpod_side_effects_may_have_occurred": True,
                "provider_job_submitted": True,
                "http_status_code": status_code,
                "runpod_response": redacted_response,
            }
        )
    except urllib.error.HTTPError as exc:
        error_body = _redact_text(
            exc.read().decode("utf-8", errors="replace"),
            api_key,
        )
        result.update(
            {
                "status": "failed",
                "reason": "runpod_api_http_error",
                "blockers": ["runpod_api_http_error"],
                "api_call_performed": True,
                "runpod_side_effects_may_have_occurred": True,
                "http_status_code": exc.code,
                "runpod_error": error_body,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive around network/runtime failures
        result.update(
            {
                "status": "failed",
                "reason": "runpod_api_call_failed",
                "blockers": ["runpod_api_call_failed"],
                "api_call_performed": True,
                "runpod_side_effects_may_have_occurred": True,
                "error_type": type(exc).__name__,
                "error": _redact_text(str(exc), api_key),
            }
        )
    return _persist_result(resolved_output, result)


def _request_path_from_args(args: argparse.Namespace) -> Path:
    if args.provider_launch_request:
        return Path(args.provider_launch_request)
    env_path = _string(os.getenv(PROVIDER_LAUNCH_REQUEST_ENV))
    if env_path:
        return Path(env_path)
    raise ValueError(
        f"Provide --provider-launch-request or {PROVIDER_LAUNCH_REQUEST_ENV}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or submit a gated RunPod request for a robot-eval worker."
    )
    parser.add_argument("--provider-launch-request")
    parser.add_argument("--output-path")
    parser.add_argument(
        "--mode",
        choices=["dry-run", "auto", "serverless-run", "on-demand-pod"],
        default="dry-run",
    )
    parser.add_argument("--endpoint-id")
    parser.add_argument("--pod-name")
    parser.add_argument("--gpu-type-id")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument(
        "--allow-runpod-api-call",
        action="store_true",
        help=f"Required with {RUNPOD_API_GATE_ENV}=true for live RunPod API calls.",
    )
    args = parser.parse_args(argv)
    try:
        request_path = _request_path_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=args.output_path,
        mode=args.mode,
        allow_runpod_api_call=args.allow_runpod_api_call,
        endpoint_id=args.endpoint_id,
        pod_name=args.pod_name,
        gpu_type_id=args.gpu_type_id,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"[runpod-provider-adapter] result={result['output_path']}")
    print(f"[runpod-provider-adapter] status={result['status']}")
    print(f"[runpod-provider-adapter] mode={result.get('mode')}")
    blockers = result.get("blockers")
    if blockers:
        print("[runpod-provider-adapter] blockers=" + ",".join(blockers))
    return 0 if result["status"] in {"dry_run_ready", "submitted"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
