"""RunPod adapter for prepared robot-eval GPU provider launch requests."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence
from urllib.parse import urlparse

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by Python 3.10 CI
    import tomli as tomllib

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .logging_utils import log_event
from .paid_resource_admission import (
    PaidResourceAdmissionGrant,
    PaidResourceAdmissionBlocked,
    require_paid_resource_admission_grant,
)
from .provider_worker_endpoint_manifest import write_provider_worker_endpoint_manifest
from . import safe_outbound_http


RUNPOD_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION = "runpod_provider_adapter_result.v1"
RUNPOD_PROVIDER_READINESS_MANIFEST_SCHEMA_VERSION = "runpod_provider_readiness_manifest.v1"
RUNPOD_PROVIDER_READINESS_MANIFEST_NAME = "runpod_provider_readiness_manifest.json"
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


def _runpod_provider_api_policy() -> safe_outbound_http.OutboundHttpPolicy:
    """Allow only RunPod's fixed hosts and the configured HTTPS REST origin."""

    configured = urlparse(RUNPOD_REST_API_BASE)
    configured_host = configured.hostname
    if (
        configured.scheme != "https"
        or not configured_host
        or configured.username is not None
        or configured.password is not None
    ):
        raise ValueError(f"{RUNPOD_REST_API_BASE_ENV}_must_be_credential_free_https_origin")
    return safe_outbound_http.OutboundHttpPolicy(
        allowed_hosts=frozenset(
            {"rest.runpod.io", "api.runpod.ai", configured_host.lower()}
        ),
        max_response_bytes=8 * 1024 * 1024,
    )
RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV = "BLUEPRINT_RUNPOD_CONTAINER_REGISTRY_AUTH_ID"
RUNPOD_EXISTING_POD_ID_ENV = "BLUEPRINT_RUNPOD_EXISTING_POD_ID"
PROVIDER_LAUNCH_REQUEST_ENV = "BLUEPRINT_GPU_PROVIDER_LAUNCH_REQUEST"
PROVIDER_ADAPTER_OUTPUT_ENV = "BLUEPRINT_GPU_PROVIDER_ADAPTER_OUTPUT"
RUNPOD_FORWARD_SECRET_ENV_VARS_ENV = "BLUEPRINT_RUNPOD_FORWARD_SECRET_ENV_VARS"
GENERIC_WORKER_IMAGE_REF_ENV = "BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF"
RUNPOD_IMAGE_STARTUP_CANARY_MODE = "image-startup-canary-pod"
RUNPOD_IMAGE_STARTUP_CANARY_HOLD_SECONDS_ENV = (
    "BLUEPRINT_RUNPOD_IMAGE_STARTUP_CANARY_HOLD_SECONDS"
)
RUNPOD_ALLOW_LARGE_IMAGE_FRESH_START_ENV = (
    "BLUEPRINT_ALLOW_LARGE_RUNPOD_IMAGE_FRESH_START"
)
RUNPOD_LARGE_IMAGE_TOTAL_WARN_BYTES = 12_000_000_000
RUNPOD_LARGE_IMAGE_LAYER_WARN_BYTES = 8_000_000_000
SENSITIVE_ENV_NAME_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
REMOTE_PROVIDER_ARTIFACT_OUTPUT_URI_SCHEMES = {"gs", "s3", "r2"}
SECRET_ENV_NAME_LIST_KEYS = {
    "env_names_declared",
    "secret_env_var_names",
}
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


def _local_sim_only_prerequisite(request: Mapping[str, Any]) -> Dict[str, Any]:
    provider_shape = _provider_shape(request)
    return _mapping(
        provider_shape.get("local_sim_only_prerequisite")
        or request.get("local_sim_only_prerequisite")
    )


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
    startup_artifact_timeout_seconds = int(
        _number(limits.get("startup_artifact_timeout_seconds"))
        or min(360, max(60, hard_timeout_seconds // 2))
    )
    watchdog_ttl_seconds = int(
        _number(limits.get("external_watchdog_ttl_seconds")) or max(900, hard_timeout_seconds)
    )
    max_active_workers = int(_number(limits.get("max_active_workers")) or 1)
    return {
        "source": "gpu_provider_launch_request.provider_request_shape.limits",
        "hard_timeout_seconds": hard_timeout_seconds,
        "idle_timeout_seconds": idle_timeout_seconds,
        "startup_artifact_timeout_seconds": startup_artifact_timeout_seconds,
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
            "recommended_startup_artifact_timeout_seconds": startup_artifact_timeout_seconds,
        },
        "on_demand_pod_controls": {
            "pod_idle_timeout_is_not_provider_native": True,
            "external_watchdog_or_owner_terminator_required": True,
            "startup_artifact_watchdog_required": True,
            "startup_artifact_timeout_seconds": startup_artifact_timeout_seconds,
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


def _redact_secret_env_name_lists(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in SECRET_ENV_NAME_LIST_KEYS:
                redacted[key_text] = [
                    "<redacted:secret-env-var-name>" for _ in _string_list(item)
                ]
                continue
            redacted[key_text] = _redact_secret_env_name_lists(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secret_env_name_lists(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_secret_env_name_lists(item) for item in value]
    return value


def _bytes_value(value: Any) -> int | None:
    number = _number(value)
    if number is None or number < 0:
        return None
    return int(number)


def _image_size_metadata(image: Mapping[str, Any]) -> Dict[str, Any]:
    for key in (
        "image_size_diagnostic",
        "image_size_metadata",
        "image_manifest",
        "registry_manifest",
        "manifest_inspection",
    ):
        metadata = _mapping(image.get(key))
        if metadata:
            return metadata
    return {}


def _image_layer_sizes_bytes(image: Mapping[str, Any]) -> list[int]:
    metadata = _image_size_metadata(image)
    sizes: list[int] = []
    for source in (image, metadata):
        for key in (
            "compressed_layer_sizes_bytes",
            "layer_sizes_bytes",
            "layers_size_bytes",
            "layer_size_bytes",
        ):
            value = source.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for item in value:
                    size = _bytes_value(item)
                    if size is not None:
                        sizes.append(size)
            else:
                size = _bytes_value(value)
                if size is not None:
                    sizes.append(size)
        layers = source.get("layers")
        if isinstance(layers, Sequence) and not isinstance(layers, (str, bytes, bytearray)):
            for layer in layers:
                layer_mapping = _mapping(layer)
                for key in ("compressed_size_bytes", "size_bytes", "size"):
                    size = _bytes_value(layer_mapping.get(key))
                    if size is not None:
                        sizes.append(size)
                        break
    return sizes


def _image_startup_diagnostic(request: Mapping[str, Any]) -> Dict[str, Any]:
    image = _image(request)
    limits = _limits(request)
    metadata = _image_size_metadata(image)
    layer_sizes = _image_layer_sizes_bytes(image)
    explicit_largest = next(
        (
            _bytes_value(source.get(key))
            for source in (image, metadata)
            for key in (
                "largest_compressed_layer_size_bytes",
                "largest_layer_size_bytes",
                "max_layer_size_bytes",
            )
            if _bytes_value(source.get(key)) is not None
        ),
        None,
    )
    largest_layer_size = explicit_largest if explicit_largest is not None else (
        max(layer_sizes) if layer_sizes else None
    )
    explicit_total = next(
        (
            _bytes_value(source.get(key))
            for source in (image, metadata)
            for key in (
                "total_compressed_size_bytes",
                "compressed_size_bytes",
                "manifest_total_size_bytes",
                "total_layer_size_bytes",
            )
            if _bytes_value(source.get(key)) is not None
        ),
        None,
    )
    total_size = explicit_total if explicit_total is not None else (
        sum(layer_sizes) if layer_sizes else None
    )
    large_total = (
        total_size is not None and total_size >= RUNPOD_LARGE_IMAGE_TOTAL_WARN_BYTES
    )
    large_layer = (
        largest_layer_size is not None
        and largest_layer_size >= RUNPOD_LARGE_IMAGE_LAYER_WARN_BYTES
    )
    warnings: list[str] = []
    if large_total:
        warnings.append("large_worker_image_total_size_may_exceed_startup_watchdog")
    if large_layer:
        warnings.append("large_worker_image_layer_may_exceed_startup_watchdog")
    startup_timeout = int(
        _number(limits.get("startup_artifact_timeout_seconds"))
        or _number(_cost_control_policy(request).get("startup_artifact_timeout_seconds"))
        or 0
    )
    return {
        "image_ref": _string(image.get("configured_image_ref")) or None,
        "metadata_present": bool(metadata or layer_sizes),
        "total_compressed_size_bytes": total_size,
        "largest_layer_size_bytes": largest_layer_size,
        "large_image_pull_risk": bool(large_total or large_layer),
        "warnings": warnings,
        "startup_artifact_timeout_seconds": startup_timeout,
        "startup_artifact_watchdog_required": _bool(
            limits.get("startup_artifact_watchdog_required")
        )
        is True,
        "same_image_canary_recommended": bool(large_total or large_layer),
        "warm_existing_pod_mode_available": True,
        "image_startup_canary_mode_available": True,
        "canary_hold_seconds_env": RUNPOD_IMAGE_STARTUP_CANARY_HOLD_SECONDS_ENV,
        "large_image_fresh_start_override_env": RUNPOD_ALLOW_LARGE_IMAGE_FRESH_START_ENV,
        "diagnostic_blocker_if_canary_times_out": (
            "prebuilt_isaac_image_layer_pull_exceeded_watchdog"
            if large_total or large_layer
            else "provider_pod_startup_or_image_pull_timeout"
        ),
        "proof_boundary": (
            "Image metadata and canary artifacts only diagnose container startup. "
            "They do not prove Isaac Sim execution, policy execution, safety, or "
            "robot readiness."
        ),
    }


def _request_summary(request: Mapping[str, Any]) -> Dict[str, Any]:
    inputs = _inputs(request)
    image = _image(request)
    limits = _limits(request)
    startup_diagnostic = _image_startup_diagnostic(request)
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
        "capture_root_bundle_uri_present": bool(
            _string(inputs.get("capture_root_bundle_uri"))
        ),
        "capture_root_bundle_uri_fetchable_by_provider": inputs.get(
            "capture_root_bundle_uri_fetchable_by_provider"
        )
        is True,
        "artifact_output_uri_present": bool(_string(inputs.get("artifact_output_uri"))),
        "local_sim_only_prerequisite_status": _local_sim_only_prerequisite(
            request
        ).get("status"),
        "local_sim_only_evidence_clean": _local_sim_only_prerequisite(request).get(
            "local_sim_only_evidence_clean"
        )
        is True,
        "hard_timeout_seconds": limits.get("hard_timeout_seconds"),
        "idle_timeout_seconds": limits.get("idle_timeout_seconds"),
        "startup_artifact_timeout_seconds": limits.get(
            "startup_artifact_timeout_seconds"
        ),
        "worker_image_startup_large_image_pull_risk": startup_diagnostic.get(
            "large_image_pull_risk"
        ),
        "external_watchdog_ttl_seconds": limits.get("external_watchdog_ttl_seconds"),
        "max_active_workers": limits.get("max_active_workers"),
    }


def _readiness_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(RUNPOD_PROVIDER_READINESS_MANIFEST_NAME)


def _provider_readiness_manifest(
    *,
    request_path: Path,
    output_path: Path,
    request: Mapping[str, Any],
    mode: str,
    request_blockers: Sequence[str],
    endpoint_manifest_path: Path,
    api_key_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    provider_shape = _provider_shape(request)
    inputs = _inputs(request)
    local_sim_only_prerequisite = _local_sim_only_prerequisite(request)
    limits = _limits(request)
    environment = _environment(request)
    artifact_finalizer = _mapping(provider_shape.get("artifact_finalizer"))
    cost_policy = _cost_control_policy(request)
    artifact_output_uri = _string(inputs.get("artifact_output_uri"))
    artifact_output_scheme = urlparse(artifact_output_uri).scheme if artifact_output_uri else ""
    hard_timeout_seconds = int(_number(limits.get("hard_timeout_seconds")) or 0)
    idle_timeout_seconds = int(_number(limits.get("idle_timeout_seconds")) or 0)
    startup_artifact_timeout_seconds = int(
        _number(limits.get("startup_artifact_timeout_seconds"))
        or _number(cost_policy.get("startup_artifact_timeout_seconds"))
        or 0
    )
    external_watchdog_ttl_seconds = int(
        _number(limits.get("external_watchdog_ttl_seconds")) or 0
    )
    requested_budget_usd = _number(limits.get("requested_budget_usd"))
    artifact_output_uri_provider_writable = bool(
        inputs.get("artifact_output_uri_provider_writable")
    )
    artifact_output_write_auth = _mapping(inputs.get("artifact_output_write_auth"))
    artifact_output_write_auth_ready = bool(
        inputs.get("artifact_output_write_auth_contract_ready")
        or artifact_output_write_auth.get("write_auth_contract_ready")
    )
    image_startup_diagnostic = _image_startup_diagnostic(request)
    signed_put_url_present = bool(
        _string(os.getenv("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"))
    )
    readiness_blockers = _dedupe(request_blockers)
    return {
        "schema_version": RUNPOD_PROVIDER_READINESS_MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": (
            "ready_for_explicit_paid_provider_attempt"
            if not readiness_blockers
            else "blocked_before_paid_provider_attempt"
        ),
        "provider": _string(request.get("provider")) or "runpod",
        "mode": mode,
        "job_id": _string(request.get("job_id")),
        "source_artifacts": {
            "provider_launch_request_path": str(request_path),
            "runpod_provider_adapter_result_path": str(output_path),
            "provider_worker_endpoint_manifest_path": str(endpoint_manifest_path),
        },
        "api_call_performed": False,
        "live_provider_call_authorized": False,
        "spend_limits": {
            "requested_budget_usd": requested_budget_usd,
            "requested_budget_declared": requested_budget_usd is not None,
            "max_active_workers": cost_policy.get("max_active_workers"),
            "bounded_single_worker_attempt": cost_policy.get("max_active_workers") == 1,
            "hard_timeout_seconds": hard_timeout_seconds,
            "idle_timeout_seconds": idle_timeout_seconds,
            "startup_artifact_timeout_seconds": startup_artifact_timeout_seconds,
            "external_watchdog_ttl_seconds": external_watchdog_ttl_seconds,
            "external_watchdog_ttl_exceeds_hard_timeout": (
                external_watchdog_ttl_seconds > hard_timeout_seconds > 0
            ),
            "scale_to_zero_default": _mapping(
                cost_policy.get("warm_pool_policy")
            ).get("scale_to_zero_default"),
            "warm_pool_policy": cost_policy.get("warm_pool_policy"),
        },
        "provider_inputs": {
            "manifest_uri_present": bool(_string(inputs.get("manifest_uri"))),
            "manifest_uri": _redact_runtime_value(_string(inputs.get("manifest_uri"))),
            "manifest_uri_fetchable_by_provider": inputs.get(
                "manifest_uri_fetchable_by_provider"
            )
            is True,
            "capture_root_bundle_uri_present": bool(
                _string(inputs.get("capture_root_bundle_uri"))
            ),
            "capture_root_bundle_uri": _redact_runtime_value(
                _string(inputs.get("capture_root_bundle_uri"))
            ),
            "capture_root_bundle_uri_fetchable_by_provider": inputs.get(
                "capture_root_bundle_uri_fetchable_by_provider"
            )
            is True,
        },
        "local_sim_only_prerequisite": {
            "present": bool(local_sim_only_prerequisite),
            **local_sim_only_prerequisite,
        },
        "artifact_output": {
            "artifact_output_uri_required": _bool(
                inputs.get("artifact_output_uri_required")
            )
            is not False,
            "artifact_output_uri_present": bool(artifact_output_uri),
            "artifact_output_uri": _redact_runtime_value(artifact_output_uri),
            "artifact_output_uri_scheme": artifact_output_scheme or None,
            "artifact_output_uri_scheme_provider_writable": artifact_output_scheme
            in REMOTE_PROVIDER_ARTIFACT_OUTPUT_URI_SCHEMES,
            "artifact_output_uri_provider_writable": artifact_output_uri_provider_writable,
            "artifact_output_write_auth_contract_ready": artifact_output_write_auth_ready,
            "runtime_manifest_signed_put_url_present": signed_put_url_present,
            "runtime_manifest_signed_put_url_value_stored": False,
        },
        "watchdog_and_teardown": {
            "idle_shutdown_required": _bool(limits.get("idle_shutdown_required")) is True,
            "idle_timeout_seconds": idle_timeout_seconds,
            "startup_artifact_watchdog_required": _bool(
                limits.get("startup_artifact_watchdog_required")
            )
            is True,
            "startup_artifact_timeout_seconds": startup_artifact_timeout_seconds,
            "external_watchdog_ttl_required": _bool(
                limits.get("external_watchdog_ttl_required")
            )
            is True,
            "external_watchdog_ttl_seconds": external_watchdog_ttl_seconds,
            "external_watchdog_owner": _string(limits.get("external_watchdog_owner"))
            or None,
            "external_watchdog_ttl_exceeds_hard_timeout": (
                external_watchdog_ttl_seconds > hard_timeout_seconds > 0
            ),
            "upload_before_shutdown_required": artifact_finalizer.get(
                "upload_before_shutdown_required"
            )
            is True,
            "record_actual_gpu_time_required": artifact_finalizer.get(
                "record_actual_gpu_time_required"
            )
            is True,
            "provider_shutdown_evidence_required_after_live_attempt": True,
            "continuing_spend_from_this_run_must_be_false_after_teardown": True,
            "expected_post_run_artifacts": [
                "provider_runtime_finalizer_proof.json",
                "worker_runtime_manifest.json",
                "provider_shutdown_proof.json or provider lifecycle zero-active-worker evidence",
            ],
        },
        "image_startup_diagnostic": image_startup_diagnostic,
        "no_secret_artifact_policy": {
            "secret_values_in_artifact": environment.get("secret_values_in_artifact"),
            "customer_visible_secret_values_allowed": environment.get(
                "customer_visible_secret_values_allowed"
            ),
            "secret_env_var_names_declared_count": len(
                _string_list(environment.get("secret_env_var_names"))
            ),
            "secret_env_var_names_stored": False,
            "api_key_configured": api_key_meta.get("api_key_configured"),
            "api_key_source": api_key_meta.get("api_key_source"),
            "raw_api_key_stored": False,
            "signed_url_values_in_artifact": False,
            "secret_values_forwarded_only_by_explicit_allowlist": True,
        },
        "blockers": readiness_blockers,
        "claim_boundary": {
            "optional_provider_runtime_evidence_only": True,
            "not_sim_only_launch_proof_until_artifacts_imported_and_reviewed": True,
            "runpod_api_called": False,
            "provider_allocation_proven": False,
            "provider_job_submitted": False,
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _write_provider_readiness_manifest(
    *,
    request_path: Path,
    output_path: Path,
    request: Mapping[str, Any],
    mode: str,
    request_blockers: Sequence[str],
    endpoint_manifest_path: Path,
    api_key_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest_path = _readiness_manifest_path(output_path)
    manifest = _provider_readiness_manifest(
        request_path=request_path,
        output_path=output_path,
        request=request,
        mode=mode,
        request_blockers=request_blockers,
        endpoint_manifest_path=endpoint_manifest_path,
        api_key_meta=api_key_meta,
    )
    write_json(manifest_path, manifest)
    return manifest


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
        "image_startup_diagnostic": _image_startup_diagnostic(request),
    }


def _adapter_event_name(status: str) -> str:
    if status == "blocked":
        return "runpod_provider_adapter.blocked"
    if status == "failed":
        return "runpod_provider_adapter.failed"
    return "runpod_provider_adapter.completed"


def _persist_result(output_path: Path, result: Mapping[str, Any]) -> Dict[str, Any]:
    persisted = _redact_secret_env_name_lists(dict(result))
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
                "capture_root_bundle_uri": _string(inputs.get("capture_root_bundle_uri")),
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
    simulator_image_env = ""
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
        if env_key in env:
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
        # RunPod exposes its persistent network volume at /workspace. Thin
        # GR00T+OSCAR releases point this at a pre-populated subdirectory;
        # the image entrypoint verifies the immutable byte manifest offline.
        "groot_oscar_models": "BLUEPRINT_GROOT_OSCAR_MODEL_CACHE",
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
            if parts and parts[0] == "blueprint-run-robot-eval-worker":
                start_cmd = parts[1:]
            else:
                start_cmd = parts
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
    network_volume_id = _string(provider_shape.get("network_volume_id"))
    data_center_id = _string(provider_shape.get("data_center_id"))
    allowed_cuda_versions = _string_list(
        provider_shape.get("allowed_cuda_versions")
    )
    if network_volume_id:
        input_payload["networkVolumeId"] = network_volume_id
    if data_center_id:
        input_payload["dataCenterIds"] = [data_center_id]
    if allowed_cuda_versions:
        input_payload["allowedCudaVersions"] = allowed_cuda_versions
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


def _image_startup_canary_command() -> str:
    return r'''set -euo pipefail
OUT_DIR="${BLUEPRINT_CANARY_OUTPUT_DIR:-/workspace/blueprint_canary_output}"
mkdir -p "$OUT_DIR"
PYTHON_BIN="${BLUEPRINT_CANARY_PYTHON:-/isaac-sim/python.sh}"
if [ -z "${BLUEPRINT_CANARY_PYTHON:-}" ]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
  if [ -z "$PYTHON_BIN" ] && [ -x /isaac-sim/python.sh ]; then
    PYTHON_BIN="/isaac-sim/python.sh"
  fi
elif [ ! -x "$PYTHON_BIN" ]; then
  RESOLVED_PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
  if [ -n "$RESOLVED_PYTHON_BIN" ]; then
    PYTHON_BIN="$RESOLVED_PYTHON_BIN"
  else
    echo "blueprint canary blocked: BLUEPRINT_CANARY_PYTHON is not executable: $PYTHON_BIN" >&2
    exit 127
  fi
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "blueprint canary blocked: no python runtime" >&2
  exit 127
fi
"$PYTHON_BIN" - <<'PY'
import json
import os
import platform
import shutil
import socket
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

out_dir = Path(os.environ.get("BLUEPRINT_CANARY_OUTPUT_DIR", "/workspace/blueprint_canary_output"))
out_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "schema_version": "runpod_image_startup_canary.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": "container_started",
    "job_id": os.environ.get("BLUEPRINT_ROBOT_EVAL_JOB_ID"),
    "hostname": socket.gethostname(),
    "python_executable": sys.executable,
    "python_selection": "explicit_BLUEPRINT_CANARY_PYTHON"
    if os.environ.get("BLUEPRINT_CANARY_PYTHON")
    else "python3_or_python_before_isaac_python",
    "platform": platform.platform(),
    "image_ref": os.environ.get("BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF")
    or os.environ.get("BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF"),
    "isaac_python_exists": Path("/isaac-sim/python.sh").exists(),
    "isaac_root_exists": Path("/isaac-sim").exists(),
    "python3_path": shutil.which("python3"),
    "python_path": shutil.which("python"),
    "curl_path": shutil.which("curl"),
    "proof_boundary": (
        "This canary proves the RunPod image container reached user command execution "
        "and uploaded an artifact. It does not prove Isaac Sim boot, scenario execution, "
        "policy execution, safety validation, or robot readiness."
    ),
}
json_path = out_dir / "runpod_image_startup_canary.json"
json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
zip_path = out_dir / "runpod_image_startup_canary_output.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(json_path, json_path.name)
upload_url = os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL")
if not upload_url:
    raise SystemExit("missing_BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL")
data = zip_path.read_bytes()
request = urllib.request.Request(
    upload_url,
    data=data,
    method="PUT",
    headers={"Content-Type": "application/zip", "Content-Length": str(len(data))},
)
with urllib.request.urlopen(request, timeout=120) as response:
    upload_status = int(getattr(response, "status", 200))
(out_dir / "runpod_image_startup_canary_upload_status.json").write_text(
    json.dumps(
        {
            "schema_version": "runpod_image_startup_canary_upload_status.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "uploaded",
            "upload_status": upload_status,
            "uploaded_bytes": len(data),
            "zip_path": str(zip_path),
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
time.sleep(float(os.environ.get("BLUEPRINT_CANARY_POST_UPLOAD_SLEEP_SECONDS", "5")))
PY'''


def _image_startup_canary_pod_payload(
    request: Mapping[str, Any],
    *,
    pod_name: str | None = None,
    gpu_type_id: str | None = None,
) -> Dict[str, Any]:
    pod_payload = _pod_payload(
        request,
        pod_name=pod_name,
        gpu_type_id=gpu_type_id,
    )
    body = _mapping(pod_payload.get("body"))
    env = _mapping(body.get("env"))
    env["BLUEPRINT_RUNPOD_IMAGE_STARTUP_CANARY"] = "true"
    canary_hold_seconds = _string(os.getenv(RUNPOD_IMAGE_STARTUP_CANARY_HOLD_SECONDS_ENV))
    if canary_hold_seconds:
        env["BLUEPRINT_CANARY_POST_UPLOAD_SLEEP_SECONDS"] = canary_hold_seconds
    else:
        env.setdefault("BLUEPRINT_CANARY_POST_UPLOAD_SLEEP_SECONDS", "5")
    thin_entrypoint = ["/opt/blueprint/thin_release_entrypoint.sh"]
    use_thin_entrypoint = body.get("dockerEntrypoint") == thin_entrypoint
    body.update(
        {
            "dockerEntrypoint": thin_entrypoint if use_thin_entrypoint else ["bash"],
            "dockerStartCmd": (
                ["bash", "-lc", _image_startup_canary_command()]
                if use_thin_entrypoint
                else ["-lc", _image_startup_canary_command()]
            ),
            "env": env,
        }
    )
    return {
        **pod_payload,
        "body": body,
        "api_surface": "rest_pods_image_startup_canary",
        "proof_boundary": {
            "image_container_startup_only": True,
            "simulator_execution_proven": False,
            "policy_execution_proven": False,
        },
    }


def _existing_pod_start_payload(
    request: Mapping[str, Any],
    *,
    pod_id: str,
    pod_name: str | None = None,
    gpu_type_id: str | None = None,
) -> Dict[str, Any]:
    pod_payload = _pod_payload(
        request,
        pod_name=pod_name,
        gpu_type_id=gpu_type_id,
    )
    pod_body = _mapping(pod_payload.get("body"))
    update_keys = {
        "containerDiskInGb",
        "containerRegistryAuthId",
        "dockerEntrypoint",
        "dockerStartCmd",
        "env",
        "imageName",
        "name",
        "networkVolumeId",
        "dataCenterIds",
        "allowedCudaVersions",
        "ports",
        "volumeInGb",
        "volumeMountPath",
    }
    update_body = {key: pod_body[key] for key in update_keys if key in pod_body}
    return {
        "update": {
            "url": f"{RUNPOD_REST_API_BASE}/pods/{pod_id}/update",
            "method": "POST",
            "body": update_body,
            "api_surface": "rest_pods_update_existing",
        },
        "start": {
            "url": f"{RUNPOD_REST_API_BASE}/pods/{pod_id}/start",
            "method": "POST",
            "body": {},
            "api_surface": "rest_pods_start_existing",
        },
        "existing_pod_id": pod_id,
        "idle_shutdown_expected_seconds": pod_payload.get(
            "idle_shutdown_expected_seconds"
        ),
    }


def _request_blockers(
    *,
    request: Mapping[str, Any],
    mode: str,
    endpoint_id: str,
    existing_pod_id: str = "",
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
    provider_shape = _provider_shape(request)
    prelaunch_spend_guard = _mapping(request.get("prelaunch_spend_guard"))
    if (
        prelaunch_spend_guard
        and prelaunch_spend_guard.get("required_before_provider_launch") is True
        and prelaunch_spend_guard.get("can_launch") is not True
    ):
        blockers.append("provider_prelaunch_spend_guard_not_passed")
        blockers.extend(_string_list(prelaunch_spend_guard.get("blockers")))
    local_sim_only_prerequisite = _local_sim_only_prerequisite(request)
    artifact_finalizer = _mapping(provider_shape.get("artifact_finalizer"))
    artifact_output_uri = _string(inputs.get("artifact_output_uri"))
    artifact_output_required = _bool(inputs.get("artifact_output_uri_required"))
    artifact_output_scheme = urlparse(artifact_output_uri).scheme or "local"
    artifact_output_write_auth = _mapping(inputs.get("artifact_output_write_auth"))
    artifact_output_write_auth_ready = bool(
        inputs.get("artifact_output_write_auth_contract_ready")
        or artifact_output_write_auth.get("write_auth_contract_ready")
    )
    signed_put_url = _string(os.getenv("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"))
    if mode == "serverless-run" and not endpoint_id:
        blockers.append(f"missing_env_{RUNPOD_ENDPOINT_ID_ENV}")
    if mode in {
        "on-demand-pod",
        "existing-pod-start",
        RUNPOD_IMAGE_STARTUP_CANARY_MODE,
    } and not _string(image.get("configured_image_ref")):
        blockers.append("missing_provider_worker_image_ref")
    if mode == "existing-pod-start" and not (
        _string(existing_pod_id) or _string(os.getenv(RUNPOD_EXISTING_POD_ID_ENV))
    ):
        blockers.append(f"missing_env_{RUNPOD_EXISTING_POD_ID_ENV}")
    if _string(image.get("configured_image_ref")) and (
        image.get("configured_image_ref_is_versioned") is not True
    ):
        blockers.append("prebuilt_worker_image_ref_not_versioned")
    if image.get("configured_image_ref_fetchable_by_provider") is False:
        blockers.append("prebuilt_worker_image_ref_not_provider_fetchable")
    image_startup_diagnostic = _image_startup_diagnostic(request)
    if (
        mode == "on-demand-pod"
        and image_startup_diagnostic.get("large_image_pull_risk") is True
        and not _env_truthy(RUNPOD_ALLOW_LARGE_IMAGE_FRESH_START_ENV)
    ):
        blockers.append("large_worker_image_requires_canary_or_warm_provider")
    if not _string(inputs.get("manifest_uri")):
        blockers.append("missing_provider_worker_manifest_uri")
    if inputs.get("manifest_uri_fetchable_by_provider") is not True:
        blockers.append("provider_worker_manifest_uri_not_fetchable")
    if not _string(inputs.get("capture_root_bundle_uri")):
        blockers.append("missing_provider_capture_root_bundle_uri")
    if inputs.get("capture_root_bundle_uri_fetchable_by_provider") is not True:
        blockers.append("provider_capture_root_bundle_uri_not_fetchable")
    if not artifact_output_uri and artifact_output_required is not False:
        blockers.append("missing_provider_artifact_output_uri")
    if artifact_output_required is False and not signed_put_url:
        blockers.append("missing_runtime_manifest_signed_put_url_for_artifact_output_optional")
    if mode == RUNPOD_IMAGE_STARTUP_CANARY_MODE and not signed_put_url:
        blockers.append("missing_runtime_manifest_signed_put_url_for_image_startup_canary")
    if (
        artifact_output_uri
        and artifact_output_required is not False
        and artifact_output_scheme not in REMOTE_PROVIDER_ARTIFACT_OUTPUT_URI_SCHEMES
    ):
        blockers.append("provider_artifact_output_uri_not_writable")
    if (
        artifact_output_uri
        and artifact_output_required is not False
        and artifact_output_scheme in REMOTE_PROVIDER_ARTIFACT_OUTPUT_URI_SCHEMES
        and inputs.get("artifact_output_uri_provider_writable") is not True
    ):
        blockers.append("provider_artifact_output_uri_not_marked_writable")
    if (
        artifact_output_uri
        and artifact_output_required is not False
        and inputs.get("artifact_output_uri_provider_writable") is True
        and not artifact_output_write_auth_ready
    ):
        blockers.append("provider_artifact_output_write_auth_contract_missing")
    canary_startup_probe = mode == RUNPOD_IMAGE_STARTUP_CANARY_MODE
    if _string(request.get("provider")) == "runpod" and not canary_startup_probe:
        if not local_sim_only_prerequisite:
            blockers.append("missing_local_sim_only_provider_prerequisite")
        elif (
            local_sim_only_prerequisite.get("status") != "passed"
            or local_sim_only_prerequisite.get("local_sim_only_evidence_clean") is not True
        ):
            blockers.append("local_sim_only_provider_prerequisite_not_passed")
            blockers.extend(_string_list(local_sim_only_prerequisite.get("blockers")))
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
    if limits.get("requested_budget_usd") is None:
        blockers.append("missing_provider_requested_budget_usd")
    elif _number(limits.get("requested_budget_usd")) is None or (
        _number(limits.get("requested_budget_usd")) or 0
    ) < 0:
        blockers.append("invalid_provider_requested_budget_usd")
    if _bool(limits.get("idle_shutdown_required")) is not True:
        blockers.append("provider_idle_shutdown_not_required")
    if not _string(limits.get("external_watchdog_owner")):
        blockers.append("provider_external_watchdog_owner_missing")
    if artifact_finalizer.get("upload_before_shutdown_required") is not True:
        blockers.append("provider_artifact_upload_before_shutdown_not_required")
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
            f"missing_env_{RUNPOD_API_KEY_ENV}_or_{RUNPOD_API_KEY_FILE_ENV}_or_"
            f"{RUNPOD_CONFIG_FILE_ENV}"
        )
    return blockers


def _http_json(
    *,
    url: str,
    payload: Mapping[str, Any] | None,
    api_key: str,
    timeout_seconds: int,
    method: str = "POST",
) -> tuple[int, Dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    response = safe_outbound_http.open_request(
        request,
        policy=_runpod_provider_api_policy(),
        timeout_seconds=timeout_seconds,
    )
    status_code = response.status
    response_text = response.body.decode("utf-8", errors="replace")
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
    existing_pod_id: str | None = None,
    gpu_type_id: str | None = None,
    timeout_seconds: int = 30,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None = None,
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
    selected_existing_pod_id = _string(existing_pod_id) or _string(
        os.getenv(RUNPOD_EXISTING_POD_ID_ENV)
    )
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
    endpoint_manifest_path = request_path.parent / "provider_worker_endpoint_manifest.json"
    request_blockers = _request_blockers(
        request=request,
        mode=mode,
        endpoint_id=selected_endpoint_id,
        existing_pod_id=selected_existing_pod_id,
    )
    if mode == "serverless-run":
        runpod_request = _serverless_payload(request, endpoint_id=selected_endpoint_id)
    elif mode == "on-demand-pod":
        runpod_request = _pod_payload(
            request,
            pod_name=pod_name,
            gpu_type_id=gpu_type_id,
        )
    elif mode == RUNPOD_IMAGE_STARTUP_CANARY_MODE:
        runpod_request = _image_startup_canary_pod_payload(
            request,
            pod_name=pod_name,
            gpu_type_id=gpu_type_id,
        )
    elif mode == "existing-pod-start":
        runpod_request = _existing_pod_start_payload(
            request,
            pod_id=selected_existing_pod_id,
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
    provider_readiness_manifest = _write_provider_readiness_manifest(
        request_path=request_path,
        output_path=resolved_output,
        request=request,
        mode=mode,
        request_blockers=request_blockers,
        endpoint_manifest_path=endpoint_manifest_path,
        api_key_meta=api_key_meta,
    )
    result["provider_readiness_manifest_path"] = str(
        _readiness_manifest_path(resolved_output)
    )
    result["provider_readiness_manifest"] = provider_readiness_manifest
    if request_blockers:
        result.update(
            {
                "status": "blocked",
                "reason": "runpod_request_not_launchable",
                "blockers": request_blockers,
            }
        )
        return _persist_result(resolved_output, result)

    if mode == "dry-run" or (
        mode == RUNPOD_IMAGE_STARTUP_CANARY_MODE and not allow_runpod_api_call
    ):
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
        require_paid_resource_admission_grant(
            paid_resource_admission_grant,
            resource_class="runpod_provider_adapter",
        )
    except PaidResourceAdmissionBlocked as exc:
        result.update(
            {
                "status": "blocked",
                "reason": "shared_paid_resource_admission_blocked",
                "blockers": [
                    "runpod_provider_shared_admission_missing_or_invalid",
                    *exc.blockers,
                ],
                "api_call_performed": False,
                "runpod_side_effects_may_have_occurred": False,
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
        elif mode == "existing-pod-start":
            update_request = _mapping(runpod_request.get("update"))
            start_request = _mapping(runpod_request.get("start"))
            update_status_code, update_response = _http_json(
                url=str(update_request["url"]),
                payload=dict(_mapping(update_request.get("body"))),
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                method=_string(update_request.get("method")) or "POST",
            )
            start_status_code, start_response = _http_json(
                url=str(start_request["url"]),
                payload=dict(_mapping(start_request.get("body"))),
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                method=_string(start_request.get("method")) or "POST",
            )
            status_code = start_status_code
            response = {
                "id": selected_existing_pod_id,
                "existing_pod_id": selected_existing_pod_id,
                "update_http_status_code": update_status_code,
                "start_http_status_code": start_status_code,
                "update_response": update_response,
                "start_response": start_response,
            }
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
        choices=[
            "dry-run",
            "auto",
            "serverless-run",
            "on-demand-pod",
            "existing-pod-start",
            RUNPOD_IMAGE_STARTUP_CANARY_MODE,
        ],
        default="dry-run",
    )
    parser.add_argument("--endpoint-id")
    parser.add_argument("--pod-name")
    parser.add_argument("--existing-pod-id")
    parser.add_argument("--gpu-type-id")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument(
        "--allow-runpod-api-call",
        action="store_true",
        help=f"Required with {RUNPOD_API_GATE_ENV}=true for live RunPod API calls.",
    )
    args = parser.parse_args(argv)
    if args.mode != "dry-run":
        print("legacy_runpod_provider_mutation_cli_disabled", file=sys.stderr)
        return 2
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
        existing_pod_id=args.existing_pod_id,
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
