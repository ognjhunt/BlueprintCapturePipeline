"""Lambda Cloud adapter for prepared robot-eval GPU provider launch requests.

The adapter consumes ``robot_eval_gpu_provider_launch_request.v1`` artifacts and
prepares Lambda Cloud On-Demand Cloud API calls. Dry-run mode validates the
Blueprint provider envelope and writes the Lambda request shape without calling
Lambda. Live API modes are fail-closed behind both ``BLUEPRINT_ALLOW_LAMBDA_API_CALLS``
and ``--allow-lambda-api-call``.

This adapter proves request consumability and, in live modes, that a Lambda API
request was made. It does not prove worker readiness, simulator execution,
artifact upload, teardown, generated-world rank fidelity, safety, or physical
robot readiness.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .logging_utils import log_event
from .paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .provider_worker_endpoint_manifest import write_provider_worker_endpoint_manifest


LAMBDA_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION = "lambda_provider_adapter_result.v1"
LAMBDA_PROVIDER_READINESS_MANIFEST_SCHEMA_VERSION = (
    "lambda_provider_readiness_manifest.v1"
)
LAMBDA_PROVIDER_READINESS_MANIFEST_NAME = "lambda_provider_readiness_manifest.json"
LAMBDA_TEARDOWN_MANIFEST_SCHEMA_VERSION = "lambda_provider_teardown_manifest.v1"
LAMBDA_TEARDOWN_MANIFEST_NAME = "lambda_provider_teardown_manifest.json"
LAMBDA_PROVIDER_NAME = "lambda_cloud"
LAMBDA_API_KEY_ENV = "LAMBDA_API_KEY"
LAMBDA_API_KEY_FILE_ENV = "LAMBDA_API_KEY_FILE"
DEFAULT_LAMBDA_API_KEY_FILE = "~/.blueprint-secrets/lambda_api_key"
LAMBDA_API_GATE_ENV = "BLUEPRINT_ALLOW_LAMBDA_API_CALLS"
LAMBDA_API_BASE_ENV = "LAMBDA_API_BASE"
DEFAULT_LAMBDA_API_BASE = "https://cloud.lambda.ai/api/v1"
LAMBDA_API_USER_AGENT = "curl/8.7.1"
LAMBDA_REGION_NAME_ENV = "BLUEPRINT_LAMBDA_REGION_NAME"
LAMBDA_INSTANCE_TYPE_NAME_ENV = "BLUEPRINT_LAMBDA_INSTANCE_TYPE_NAME"
LAMBDA_SSH_KEY_NAME_ENV = "BLUEPRINT_LAMBDA_SSH_KEY_NAME"
LAMBDA_FILE_SYSTEM_NAMES_ENV = "BLUEPRINT_LAMBDA_FILE_SYSTEM_NAMES"
LAMBDA_FILE_SYSTEM_MOUNTS_JSON_ENV = "BLUEPRINT_LAMBDA_FILE_SYSTEM_MOUNTS_JSON"
LAMBDA_IMAGE_ID_ENV = "BLUEPRINT_LAMBDA_IMAGE_ID"
LAMBDA_IMAGE_FAMILY_ENV = "BLUEPRINT_LAMBDA_IMAGE_FAMILY"
LAMBDA_FIREWALL_RULESET_IDS_ENV = "BLUEPRINT_LAMBDA_FIREWALL_RULESET_IDS"
LAMBDA_TAGS_JSON_ENV = "BLUEPRINT_LAMBDA_TAGS_JSON"
LAMBDA_INSTANCE_NAME_ENV = "BLUEPRINT_LAMBDA_INSTANCE_NAME"
LAMBDA_HOSTNAME_ENV = "BLUEPRINT_LAMBDA_HOSTNAME"
LAMBDA_USER_DATA_FILE_ENV = "BLUEPRINT_LAMBDA_USER_DATA_FILE"
LAMBDA_FORWARD_SECRET_ENV_VARS_ENV = "BLUEPRINT_LAMBDA_FORWARD_SECRET_ENV_VARS"
LAMBDA_INSTANCE_IDS_ENV = "BLUEPRINT_LAMBDA_INSTANCE_IDS"
GENERIC_WORKER_IMAGE_REF_ENV = "BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF"
PROVIDER_LAUNCH_REQUEST_ENV = "BLUEPRINT_GPU_PROVIDER_LAUNCH_REQUEST"
PROVIDER_ADAPTER_OUTPUT_ENV = "BLUEPRINT_GPU_PROVIDER_ADAPTER_OUTPUT"
SIGNED_URL_SIGNATURE_PARAM = "x-goog-" + "signature="
SIGNED_URL_QUERY_KEY_MARKERS = ("signature", "credential", "security-token")
LAMBDA_CLOUD_INIT_USER_DATA_MAX_BYTES = 1_000_000
REMOTE_PROVIDER_ARTIFACT_OUTPUT_URI_SCHEMES = {"gs", "s3", "r2"}
SENSITIVE_ENV_NAME_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
LIVE_LAUNCH_MODES = {"auto", "allocate", "launch-instance"}
READ_ONLY_API_MODES = {
    "list-instances",
    "list-instance-types",
    "list-ssh-keys",
    "list-images",
    "list-regions",
}
TERMINATE_MODE = "terminate-instances"
MUTATING_API_MODES = LIVE_LAUNCH_MODES | {TERMINATE_MODE}
API_MODES = MUTATING_API_MODES | READ_ONLY_API_MODES
LAMBDA_TERMINAL_INSTANCE_STATUSES = {
    "deleted",
    "destroyed",
    "exited",
    "not_found",
    "stopped",
    "terminated",
}
LAMBDA_DOC_SOURCES = [
    {
        "label": "Lambda Cloud API OpenAPI spec",
        "url": "https://docs-api.lambda.ai/api/cloud/spec.json",
        "notes": [
            "production server is https://cloud.lambda.ai/",
            "Bearer API key auth is preferred",
            "launch endpoint is POST /api/v1/instance-operations/launch",
            "termination endpoint is POST /api/v1/instance-operations/terminate",
        ],
    },
    {
        "label": "Lambda Cloud instance management docs",
        "url": "https://docs.lambda.ai/public-cloud/on-demand/creating-managing-instances/",
        "notes": [
            "instances can be listed, launched, restarted, and terminated through the API",
            "Lambda warns OS-level shutdown leaves billing active",
        ],
    },
    {
        "label": "Lambda Cloud SSH access docs",
        "url": "https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/",
        "notes": [
            "one SSH key is selected when launching an On-Demand Cloud instance",
            "instances are reached as ubuntu@instance-ip with the selected key",
        ],
    },
]
logger = logging.getLogger(__name__)


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
            return float(value.strip())
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


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable) and not isinstance(
        value, (bytes, bytearray, Mapping)
    ):
        return [item for item in (_string(item) for item in value) if item]
    return []


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _json_env(name: str) -> Any:
    value = _string(os.getenv(name))
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _read_runtime_file_value(value: Any) -> str:
    path_text = _string(value)
    if not path_text:
        return ""
    path = Path(path_text).expanduser()
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _input_value(inputs: Mapping[str, Any], key: str) -> str:
    return _string(inputs.get(key)) or _read_runtime_file_value(inputs.get(f"{key}_file"))


def _signed_put_url_from_inputs(inputs: Mapping[str, Any]) -> str:
    return (
        _string(os.getenv("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"))
        or _read_runtime_file_value(inputs.get("artifact_output_signed_put_url_file"))
        or _read_runtime_file_value(inputs.get("runtime_manifest_signed_put_url_file"))
    )


def _provider_shape(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(request.get("provider_request_shape"))


def _provider_lambda_shape(request: Mapping[str, Any]) -> Dict[str, Any]:
    provider_shape = _provider_shape(request)
    return _mapping(provider_shape.get("lambda_cloud") or provider_shape.get("lambda"))


def _inputs(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(_provider_shape(request).get("inputs"))


def _image(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(_provider_shape(request).get("image"))


def _gpu(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(_provider_shape(request).get("gpu"))


def _limits(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(_provider_shape(request).get("limits"))


def _environment(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(_provider_shape(request).get("environment"))


def _cache(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(_provider_shape(request).get("cache"))


def _local_sim_only_prerequisite(request: Mapping[str, Any]) -> Dict[str, Any]:
    provider_shape = _provider_shape(request)
    return _mapping(
        provider_shape.get("local_sim_only_prerequisite")
        or request.get("local_sim_only_prerequisite")
    )


def _read_lambda_api_key() -> tuple[str, dict[str, Any]]:
    """Resolve the Lambda Cloud API key from env, then the file pointer."""
    env_value = _string(os.getenv(LAMBDA_API_KEY_ENV))
    if env_value:
        return env_value, {
            "api_key_configured": True,
            "api_key_source": LAMBDA_API_KEY_ENV,
            "api_key_file_configured": False,
        }
    key_file = _string(os.getenv(LAMBDA_API_KEY_FILE_ENV)) or DEFAULT_LAMBDA_API_KEY_FILE
    path = Path(key_file).expanduser()
    if not path.is_file():
        return "", {
            "api_key_configured": False,
            "api_key_source": None,
            "api_key_file_configured": False,
            "api_key_file": str(path),
        }
    try:
        key = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return "", {
            "api_key_configured": False,
            "api_key_source": LAMBDA_API_KEY_FILE_ENV,
            "api_key_file_configured": True,
            "api_key_file": str(path),
            "api_key_file_read_error": type(exc).__name__,
        }
    return key, {
        "api_key_configured": bool(key),
        "api_key_source": LAMBDA_API_KEY_FILE_ENV if key else None,
        "api_key_file_configured": True,
        "api_key_file": str(path),
    }


def _lambda_api_base() -> str:
    return (_string(os.getenv(LAMBDA_API_BASE_ENV)) or DEFAULT_LAMBDA_API_BASE).rstrip("/")


def _clean_instance_name(job_id: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "-", job_id).strip("-._")
    return f"blueprint-{base or 'robot-eval'}"[:64]


def _split_ids(values: Sequence[str] | None) -> list[str]:
    explicit = list(values or [])
    env_values = _csv(_string(os.getenv(LAMBDA_INSTANCE_IDS_ENV)))
    return _dedupe([*explicit, *env_values])


def _tag_entries(request: Mapping[str, Any]) -> list[dict[str, str]]:
    tags: list[dict[str, str]] = []
    job_id = _string(request.get("job_id"))
    if job_id:
        tags.append({"key": "blueprint:job-id", "value": job_id[:128]})
    tags.append({"key": "blueprint:provider", "value": LAMBDA_PROVIDER_NAME})
    env_tags = _json_env(LAMBDA_TAGS_JSON_ENV)
    if isinstance(env_tags, Mapping):
        for key, value in env_tags.items():
            key_text = _string(key)
            value_text = _string(value)
            if key_text and value_text:
                tags.append({"key": key_text[:55], "value": value_text[:128]})
    elif isinstance(env_tags, Sequence) and not isinstance(env_tags, (str, bytes)):
        for item in env_tags:
            item_map = _mapping(item)
            key_text = _string(item_map.get("key"))
            value_text = _string(item_map.get("value"))
            if key_text and value_text:
                tags.append({"key": key_text[:55], "value": value_text[:128]})
    deduped: dict[str, str] = {}
    for tag in tags:
        key = tag["key"]
        if re.match(r"^[a-z][a-z0-9-:]+$", key):
            deduped[key] = tag["value"]
    return [{"key": key, "value": value} for key, value in deduped.items()]


def _optional_json_list_env(name: str) -> list[dict[str, Any]]:
    value = _json_env(name)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_mapping(item) for item in value if _mapping(item)]
    return []


def _launch_config(
    request: Mapping[str, Any],
    *,
    region_name: str | None = None,
    instance_type_name: str | None = None,
    ssh_key_name: str | None = None,
    file_system_names: Sequence[str] | None = None,
    image_id: str | None = None,
    image_family: str | None = None,
    firewall_ruleset_ids: Sequence[str] | None = None,
    instance_name: str | None = None,
    hostname: str | None = None,
    user_data_file: str | None = None,
) -> dict[str, Any]:
    lambda_shape = _provider_lambda_shape(request)
    ssh_key_names = _string_list(lambda_shape.get("ssh_key_names"))
    file_system_names_from_shape = _string_list(lambda_shape.get("file_system_names"))
    firewall_rulesets_from_shape = _string_list(lambda_shape.get("firewall_ruleset_ids"))
    selected_image_id = (
        _string(image_id)
        or _string(os.getenv(LAMBDA_IMAGE_ID_ENV))
        or _string(lambda_shape.get("image_id"))
    )
    selected_image_family = (
        _string(image_family)
        or _string(os.getenv(LAMBDA_IMAGE_FAMILY_ENV))
        or _string(lambda_shape.get("image_family"))
    )
    selected_user_data_file = (
        _string(user_data_file)
        or _string(os.getenv(LAMBDA_USER_DATA_FILE_ENV))
        or _string(lambda_shape.get("user_data_file"))
    )
    selected_instance_name = (
        _string(instance_name)
        or _string(os.getenv(LAMBDA_INSTANCE_NAME_ENV))
        or _string(lambda_shape.get("name"))
        or _clean_instance_name(_string(request.get("job_id")))
    )
    return {
        "region_name": (
            _string(region_name)
            or _string(os.getenv(LAMBDA_REGION_NAME_ENV))
            or _string(lambda_shape.get("region_name"))
        ),
        "instance_type_name": (
            _string(instance_type_name)
            or _string(os.getenv(LAMBDA_INSTANCE_TYPE_NAME_ENV))
            or _string(lambda_shape.get("instance_type_name"))
            or _string(_gpu(request).get("lambda_instance_type_name"))
        ),
        "ssh_key_names": _dedupe(
            [
                _string(ssh_key_name)
                or _string(os.getenv(LAMBDA_SSH_KEY_NAME_ENV))
                or _string(lambda_shape.get("ssh_key_name")),
                *(ssh_key_names[:1] if ssh_key_names else []),
            ]
        ),
        "file_system_names": _dedupe(
            [
                *_string_list(file_system_names),
                *_csv(_string(os.getenv(LAMBDA_FILE_SYSTEM_NAMES_ENV))),
                *file_system_names_from_shape,
            ]
        ),
        "file_system_mounts": _optional_json_list_env(LAMBDA_FILE_SYSTEM_MOUNTS_JSON_ENV)
        or [
            _mapping(item)
            for item in (
                lambda_shape.get("file_system_mounts")
                if isinstance(lambda_shape.get("file_system_mounts"), Sequence)
                and not isinstance(lambda_shape.get("file_system_mounts"), (str, bytes))
                else []
            )
            if _mapping(item)
        ],
        "firewall_rulesets": [
            {"id": value}
            for value in _dedupe(
                [
                    *_string_list(firewall_ruleset_ids),
                    *_csv(_string(os.getenv(LAMBDA_FIREWALL_RULESET_IDS_ENV))),
                    *firewall_rulesets_from_shape,
                ]
            )
        ],
        "image": {"id": selected_image_id}
        if selected_image_id
        else {"family": selected_image_family}
        if selected_image_family
        else None,
        "name": selected_instance_name,
        "hostname": (
            _string(hostname)
            or _string(os.getenv(LAMBDA_HOSTNAME_ENV))
            or _string(lambda_shape.get("hostname"))
        ),
        "tags": _tag_entries(request),
        "user_data_file": selected_user_data_file,
    }


def _worker_env(request: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    inputs = _inputs(request)
    limits = _limits(request)
    image_ref = _string(_image(request).get("configured_image_ref"))
    environment = _environment(request)
    runtime_preflight = _mapping(_provider_shape(request).get("runtime_preflight"))
    simulator = (
        _string(runtime_preflight.get("simulator"))
        or _string(request.get("simulator"))
        or _string(inputs.get("simulator"))
    )
    env = {
        "BLUEPRINT_EVAL_MANIFEST_URI": _input_value(inputs, "manifest_uri"),
        "BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI": _input_value(
            inputs,
            "capture_root_bundle_uri",
        ),
        "BLUEPRINT_ROBOT_EVAL_JOB_ID": _string(request.get("job_id")),
        "BLUEPRINT_ROBOT_EVAL_PROVIDER_RUNTIME": "true",
        "BLUEPRINT_GPU_PROVIDER": LAMBDA_PROVIDER_NAME,
        "BLUEPRINT_GPU_PROVIDER_HARD_TIMEOUT_SECONDS": str(
            int(_number(limits.get("hard_timeout_seconds")) or 600)
        ),
        "BLUEPRINT_GPU_PROVIDER_IDLE_TIMEOUT_SECONDS": str(
            int(_number(limits.get("idle_timeout_seconds")) or 60)
        ),
        "BLUEPRINT_GPU_PROVIDER_EXTERNAL_WATCHDOG_TTL_SECONDS": str(
            int(_number(limits.get("external_watchdog_ttl_seconds")) or 900)
        ),
    }
    artifact_output_required = _bool(inputs.get("artifact_output_uri_required"))
    if artifact_output_required is not False:
        env["BLUEPRINT_ARTIFACT_OUTPUT_URI"] = _input_value(
            inputs,
            "artifact_output_uri",
        )
    signed_put_url = _signed_put_url_from_inputs(inputs)
    if signed_put_url:
        env["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"] = signed_put_url
    if _env_truthy("BLUEPRINT_ALLOW_GPU_PROVISIONING"):
        env["BLUEPRINT_ALLOW_GPU_PROVISIONING"] = "true"
    if _env_truthy("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"):
        env["BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"] = "true"
    if image_ref:
        env[GENERIC_WORKER_IMAGE_REF_ENV] = image_ref
    simulator_image_env = {
        "isaac_sim": "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
        "isaac_lab_arena": "BLUEPRINT_ISAAC_ARENA_EVAL_WORKER_IMAGE_REF",
        "mujoco": "BLUEPRINT_MUJOCO_EVAL_WORKER_IMAGE_REF",
        "pybullet": "BLUEPRINT_PYBULLET_EVAL_WORKER_IMAGE_REF",
        "newton": "BLUEPRINT_NEWTON_EVAL_WORKER_IMAGE_REF",
    }.get(simulator)
    if simulator_image_env and image_ref:
        env[simulator_image_env] = image_ref
    plaintext_names = set(_string_list(environment.get("plaintext_env_var_names")))
    secret_names = set(_string_list(environment.get("secret_env_var_names")))
    plaintext_values = _mapping(
        environment.get("plaintext_env_values") or environment.get("plaintext_env")
    )
    plaintext_value_files = _mapping(environment.get("plaintext_env_value_files"))
    for key, value in plaintext_values.items():
        env_key = _string(key)
        env_value = _string(value)
        if not env_key or not env_value or env_key in secret_names or env_key in env:
            continue
        if plaintext_names and env_key not in plaintext_names:
            continue
        if SIGNED_URL_SIGNATURE_PARAM in env_value.lower():
            continue
        env[env_key] = env_value
    for key, value in plaintext_value_files.items():
        env_key = _string(key)
        env_value = _read_runtime_file_value(value)
        if not env_key or not env_value or env_key in secret_names or env_key in env:
            continue
        if plaintext_names and env_key not in plaintext_names:
            continue
        env[env_key] = env_value
    forwarded_secret_names: list[str] = []
    for env_key in _csv(_string(os.getenv(LAMBDA_FORWARD_SECRET_ENV_VARS_ENV))):
        if not any(marker in env_key.upper() for marker in SENSITIVE_ENV_NAME_MARKERS):
            continue
        env_value = os.getenv(env_key)
        if env_value:
            env[env_key] = env_value
            forwarded_secret_names.append(env_key)
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
    return {key: value for key, value in env.items() if value}, forwarded_secret_names


def _generated_user_data(request: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    image_ref = _string(_image(request).get("configured_image_ref"))
    provider_shape = _provider_shape(request)
    command = _string(provider_shape.get("command"))
    if not command:
        command = "blueprint-run-robot-eval-worker --manifest ${BLUEPRINT_EVAL_MANIFEST_URI}"
    env, forwarded_secret_names = _worker_env(request)
    export_lines = [
        f"export {key}={shlex.quote(value)}" for key, value in sorted(env.items())
    ]
    docker_env_flags = " ".join(f"-e {shlex.quote(key)}" for key in sorted(env))
    job_id = _string(request.get("job_id")) or "robot-eval"
    container_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"blueprint-{job_id}")[:64]
    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "mkdir -p /opt/blueprint-runtime",
        *export_lines,
        "cd /opt/blueprint-runtime",
    ]
    if image_ref:
        script_lines.extend(
            [
                'DOCKER_CMD="docker"',
                'if command -v sudo >/dev/null 2>&1; then DOCKER_CMD="sudo docker"; fi',
                "mkdir -p /workspace/out",
                "chmod 777 /workspace /workspace/out || true",
                f"$DOCKER_CMD pull {shlex.quote(image_ref)}",
                (
                    "$DOCKER_CMD run --rm --gpus all "
                    f"--name {shlex.quote(container_name)} "
                    "--entrypoint bash "
                    "-v /workspace:/workspace "
                    f"{docker_env_flags} "
                    f"{shlex.quote(image_ref)} "
                    f"-lc {shlex.quote(command)}"
                ),
            ]
        )
    else:
        script_lines.append(command)
    user_data = "\n".join(script_lines) + "\n"
    metadata = {
        "generated_by_adapter": True,
        "format": "cloud-init-compatible-shell-script",
        "worker_image_ref_present": bool(image_ref),
        "worker_command_present": bool(command),
        "plaintext_env_var_names": sorted(
            key for key in env if key not in forwarded_secret_names
        ),
        "forwarded_secret_env_var_names": sorted(forwarded_secret_names),
        "secret_values_in_artifact": False,
        "user_data_size_bytes": len(user_data.encode("utf-8")),
        "lambda_user_data_size_limit_bytes": LAMBDA_CLOUD_INIT_USER_DATA_MAX_BYTES,
    }
    return user_data, metadata


def _user_data_for_request(
    request: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str, dict[str, Any], list[str]]:
    blockers: list[str] = []
    user_data_file = _string(config.get("user_data_file"))
    if user_data_file:
        path = Path(user_data_file).expanduser()
        if not path.is_file():
            return "", {
                "generated_by_adapter": False,
                "user_data_file": str(path),
                "user_data_file_configured": False,
            }, ["lambda_user_data_file_missing"]
        try:
            user_data = path.read_text(encoding="utf-8")
        except OSError as exc:
            return "", {
                "generated_by_adapter": False,
                "user_data_file": str(path),
                "user_data_file_configured": True,
                "user_data_file_read_error": type(exc).__name__,
            }, ["lambda_user_data_file_unreadable"]
        metadata = {
            "generated_by_adapter": False,
            "user_data_file": str(path),
            "user_data_file_configured": True,
            "user_data_size_bytes": len(user_data.encode("utf-8")),
            "lambda_user_data_size_limit_bytes": LAMBDA_CLOUD_INIT_USER_DATA_MAX_BYTES,
        }
    else:
        user_data, metadata = _generated_user_data(request)
    if len(user_data.encode("utf-8")) > LAMBDA_CLOUD_INIT_USER_DATA_MAX_BYTES:
        blockers.append("lambda_user_data_exceeds_1mb_limit")
    return user_data, metadata, blockers


def _launch_payload(
    request: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    blockers: list[str] = []
    if not _string(config.get("region_name")):
        blockers.append(f"missing_env_{LAMBDA_REGION_NAME_ENV}")
    if not _string(config.get("instance_type_name")):
        blockers.append(f"missing_env_{LAMBDA_INSTANCE_TYPE_NAME_ENV}")
    ssh_key_names = _string_list(config.get("ssh_key_names"))
    if len(ssh_key_names) != 1:
        blockers.append(f"missing_or_invalid_env_{LAMBDA_SSH_KEY_NAME_ENV}")
    user_data, user_data_meta, user_data_blockers = _user_data_for_request(
        request,
        config,
    )
    blockers.extend(user_data_blockers)
    body: dict[str, Any] = {
        "region_name": _string(config.get("region_name")),
        "instance_type_name": _string(config.get("instance_type_name")),
        "ssh_key_names": ssh_key_names,
        "file_system_names": _string_list(config.get("file_system_names")),
        "name": _string(config.get("name")) or _clean_instance_name(_string(request.get("job_id"))),
        "user_data": user_data,
        "tags": list(config.get("tags") or []),
    }
    for optional_key in ("hostname", "image", "file_system_mounts", "firewall_rulesets"):
        value = config.get(optional_key)
        if value:
            body[optional_key] = value
    return {
        "url": f"{_lambda_api_base()}/instance-operations/launch",
        "method": "POST",
        "body": body,
        "api_surface": "lambda_cloud_instance_launch",
        "rate_limit_note": "launch endpoint is limited to one request per 12 seconds or five per minute",
    }, user_data_meta, blockers


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
        "manifest_uri_present": bool(_input_value(inputs, "manifest_uri")),
        "manifest_uri_fetchable_by_provider": inputs.get("manifest_uri_fetchable_by_provider")
        is True,
        "capture_root_bundle_uri_present": bool(
            _input_value(inputs, "capture_root_bundle_uri")
        ),
        "capture_root_bundle_uri_fetchable_by_provider": inputs.get(
            "capture_root_bundle_uri_fetchable_by_provider"
        )
        is True,
        "artifact_output_uri_present": bool(_input_value(inputs, "artifact_output_uri")),
        "artifact_output_uri_provider_writable": inputs.get(
            "artifact_output_uri_provider_writable"
        )
        is True,
        "local_sim_only_prerequisite_status": _local_sim_only_prerequisite(
            request
        ).get("status"),
        "hard_timeout_seconds": limits.get("hard_timeout_seconds"),
        "idle_timeout_seconds": limits.get("idle_timeout_seconds"),
        "external_watchdog_ttl_seconds": limits.get("external_watchdog_ttl_seconds"),
        "max_active_workers": limits.get("max_active_workers"),
    }


def _request_blockers(request: Mapping[str, Any], *, mode: str) -> list[str]:
    blockers: list[str] = []
    if request.get("schema_version") != "robot_eval_gpu_provider_launch_request.v1":
        blockers.append("invalid_provider_launch_request_schema")
    if _string(request.get("provider")) != LAMBDA_PROVIDER_NAME:
        blockers.append("provider_launch_request_not_lambda_cloud")
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
    limits = _limits(request)
    provider_shape = _provider_shape(request)
    local_sim_only_prerequisite = _local_sim_only_prerequisite(request)
    artifact_finalizer = _mapping(provider_shape.get("artifact_finalizer"))
    artifact_output_uri = _input_value(inputs, "artifact_output_uri")
    artifact_output_required = _bool(inputs.get("artifact_output_uri_required"))
    artifact_output_scheme = urllib.parse.urlparse(artifact_output_uri).scheme or "local"
    artifact_output_write_auth = _mapping(inputs.get("artifact_output_write_auth"))
    artifact_output_write_auth_ready = bool(
        inputs.get("artifact_output_write_auth_contract_ready")
        or artifact_output_write_auth.get("write_auth_contract_ready")
    )
    if mode in LIVE_LAUNCH_MODES and not _string(image.get("configured_image_ref")):
        blockers.append("missing_provider_worker_image_ref")
    if _string(image.get("configured_image_ref")) and (
        image.get("configured_image_ref_is_versioned") is not True
    ):
        blockers.append("prebuilt_worker_image_ref_not_versioned")
    if image.get("configured_image_ref_fetchable_by_provider") is False:
        blockers.append("prebuilt_worker_image_ref_not_provider_fetchable")
    if not _input_value(inputs, "manifest_uri"):
        blockers.append("missing_provider_worker_manifest_uri")
    if inputs.get("manifest_uri_fetchable_by_provider") is not True:
        blockers.append("provider_worker_manifest_uri_not_fetchable")
    if not _input_value(inputs, "capture_root_bundle_uri"):
        blockers.append("missing_provider_capture_root_bundle_uri")
    if inputs.get("capture_root_bundle_uri_fetchable_by_provider") is not True:
        blockers.append("provider_capture_root_bundle_uri_not_fetchable")
    if not artifact_output_uri and artifact_output_required is not False:
        blockers.append("missing_provider_artifact_output_uri")
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
    if not local_sim_only_prerequisite:
        blockers.append("missing_local_sim_only_provider_prerequisite")
    elif (
        local_sim_only_prerequisite.get("status") != "passed"
        or local_sim_only_prerequisite.get("local_sim_only_evidence_clean") is not True
    ):
        blockers.append("local_sim_only_provider_prerequisite_not_passed")
        blockers.extend(_string_list(local_sim_only_prerequisite.get("blockers")))
    hard_timeout_seconds = _number(limits.get("hard_timeout_seconds"))
    idle_timeout_seconds = _number(limits.get("idle_timeout_seconds"))
    watchdog_ttl_seconds = _number(limits.get("external_watchdog_ttl_seconds"))
    max_active_workers = _number(limits.get("max_active_workers"))
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


def _redact_signed_url_text(text: str) -> str:
    parsed = urllib.parse.urlsplit(text)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    if query and any(
        any(marker in key.lower() for marker in SIGNED_URL_QUERY_KEY_MARKERS)
        for key, _value in query
    ):
        return urllib.parse.urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                "REDACTED_QUERY",
                parsed.fragment,
            )
        )
    redacted_query = [
        (key, "<redacted:signed-url-signature>")
        if key.lower() == SIGNED_URL_SIGNATURE_PARAM.rstrip("=").lower()
        else (key, value)
        for key, value in query
    ]
    if query != redacted_query:
        return urllib.parse.urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urllib.parse.urlencode(redacted_query),
                parsed.fragment,
            )
        )
    return text


def _redact_text(text: str, api_key: str = "") -> str:
    value = text.replace(api_key, "<redacted:LAMBDA_API_KEY>") if api_key else text
    value = _redact_signed_url_text(value)
    value = re.sub(r"([?&]token=)[^&]+", r"\1<redacted:jupyter-token>", value)
    return value


def _redact_runtime_value(value: Any, *, api_key: str = "") -> Any:
    if isinstance(value, str):
        return _redact_text(value, api_key=api_key)
    if isinstance(value, list):
        return [_redact_runtime_value(item, api_key=api_key) for item in value]
    if isinstance(value, tuple):
        return [_redact_runtime_value(item, api_key=api_key) for item in value]
    if isinstance(value, Mapping):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            upper = key_text.upper()
            if key_text in {"user_data", "private_key"} and item:
                redacted[key_text] = f"<redacted:{key_text}>"
            elif key_text in {"jupyter_token", "Authorization"} and item:
                redacted[key_text] = f"<redacted:{key_text}>"
            elif (
                isinstance(item, str)
                and item
                and any(marker in upper for marker in SENSITIVE_ENV_NAME_MARKERS)
            ):
                redacted[key_text] = "<redacted:secret-field>"
            else:
                redacted[key_text] = _redact_runtime_value(item, api_key=api_key)
        return redacted
    return value


def _api_gate_blockers(*, allow_lambda_api_call: bool, api_key: str) -> list[str]:
    blockers: list[str] = []
    if not _env_truthy(LAMBDA_API_GATE_ENV):
        blockers.append(f"missing_env_{LAMBDA_API_GATE_ENV}")
    if not allow_lambda_api_call:
        blockers.append("missing_cli_allow_lambda_api_call")
    if not api_key:
        blockers.append(f"missing_env_{LAMBDA_API_KEY_ENV}_or_{LAMBDA_API_KEY_FILE_ENV}")
    return blockers


def _http_json(
    *,
    url: str,
    payload: Mapping[str, Any] | None,
    api_key: str,
    timeout_seconds: int,
    method: str,
) -> tuple[int, Dict[str, Any]]:
    from .provider_transport import provider_json_request

    return provider_json_request(
        url=url,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": LAMBDA_API_USER_AGENT,
        },
        body_json=payload,
        timeout_seconds=timeout_seconds,
    )


def _lambda_instance_records(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = response.get("data")
    raw_instances: Any = []
    if isinstance(data, Mapping):
        raw_instances = (
            data.get("instances")
            or data.get("instance_list")
            or data.get("items")
            or data.get("results")
            or []
        )
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        raw_instances = data
    elif isinstance(response.get("instances"), Sequence) and not isinstance(
        response.get("instances"),
        (str, bytes, bytearray),
    ):
        raw_instances = response.get("instances")
    if not isinstance(raw_instances, Sequence) or isinstance(
        raw_instances,
        (str, bytes, bytearray),
    ):
        return []
    return [dict(item) for item in raw_instances if isinstance(item, Mapping)]


def _lambda_instance_id(record: Mapping[str, Any]) -> str:
    return _string(
        record.get("id")
        or record.get("instance_id")
        or record.get("instanceId")
        or record.get("name")
    )


def _lambda_instance_status(record: Mapping[str, Any]) -> str:
    return _string(
        record.get("status")
        or record.get("state")
        or record.get("instance_status")
        or record.get("lifecycle_state")
    ).lower()


def _lambda_teardown_verification_from_list_response(
    *,
    target_instance_ids: Sequence[str],
    response: Mapping[str, Any],
    http_status_code: int,
) -> dict[str, Any]:
    targets = [item for item in (_string(item) for item in target_instance_ids) if item]
    records = _lambda_instance_records(response)
    by_id = {_lambda_instance_id(record): record for record in records if _lambda_instance_id(record)}
    active: list[dict[str, Any]] = []
    terminal: list[dict[str, Any]] = []
    missing: list[str] = []
    for instance_id in targets:
        record = by_id.get(instance_id)
        if record is None:
            missing.append(instance_id)
            terminal.append({"id": instance_id, "status": "not_found"})
            continue
        status = _lambda_instance_status(record)
        summary = {"id": instance_id, "status": status or "unknown"}
        if status in LAMBDA_TERMINAL_INSTANCE_STATUSES:
            terminal.append(summary)
        else:
            active.append(summary)
    api_confirmed = bool(targets) and not active
    return {
        "schema_version": "lambda_provider_teardown_verification.v1",
        "checked_at": utc_now_iso(),
        "status": "completed" if api_confirmed else "blocked",
        "api_confirmed": api_confirmed,
        "status_source": "provider_api",
        "http_status_code": http_status_code,
        "target_instance_ids": targets,
        "terminal_instances": terminal,
        "active_instances": active,
        "missing_instance_ids_treated_as_terminal": missing,
        "instance_count_returned": len(records),
        "blockers": [] if api_confirmed else ["lambda_instances_still_active_after_terminate"],
    }


def _verify_lambda_teardown(
    *,
    instance_ids: Sequence[str],
    api_key: str,
    timeout_seconds: int,
    attempts: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    max_attempts = max(1, int(attempts))
    for attempt in range(max_attempts):
        try:
            status_code, response = _http_json(
                url=f"{_lambda_api_base()}/instances",
                payload=None,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                method="GET",
            )
            verification = _lambda_teardown_verification_from_list_response(
                target_instance_ids=instance_ids,
                response=response,
                http_status_code=status_code,
            )
            verification["attempt"] = attempt + 1
            observations.append(verification)
            if verification.get("api_confirmed") is True:
                return {
                    **verification,
                    "attempts": observations,
                }
        except Exception as exc:  # noqa: BLE001 - teardown verification must fail closed
            observations.append(
                {
                    "schema_version": "lambda_provider_teardown_verification.v1",
                    "checked_at": utc_now_iso(),
                    "status": "blocked",
                    "api_confirmed": False,
                    "status_source": "provider_api",
                    "attempt": attempt + 1,
                    "error_type": type(exc).__name__,
                    "error": _redact_text(str(exc), api_key=api_key),
                    "blockers": ["lambda_list_instances_followup_failed"],
                }
            )
        if attempt < max_attempts - 1 and poll_interval_seconds > 0:
            time.sleep(float(poll_interval_seconds))
    last = observations[-1] if observations else {}
    blockers = _string_list(last.get("blockers")) or [
        "lambda_teardown_verification_missing"
    ]
    return {
        **{k: v for k, v in last.items() if k != "attempts"},
        "schema_version": "lambda_provider_teardown_verification.v1",
        "checked_at": utc_now_iso(),
        "status": "blocked",
        "api_confirmed": False,
        "status_source": "provider_api",
        "target_instance_ids": [
            item for item in (_string(item) for item in instance_ids) if item
        ],
        "blockers": blockers,
        "attempts": observations,
    }


def _readiness_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(LAMBDA_PROVIDER_READINESS_MANIFEST_NAME)


def _teardown_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(LAMBDA_TEARDOWN_MANIFEST_NAME)


def _provider_readiness_manifest(
    *,
    request_path: Path,
    output_path: Path,
    request: Mapping[str, Any],
    mode: str,
    request_blockers: Sequence[str],
    launch_config_blockers: Sequence[str],
    provider_worker_endpoint_manifest_path: Path,
    api_key_meta: Mapping[str, Any],
    user_data_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    launch_blockers = _dedupe([*request_blockers, *launch_config_blockers])
    limits = _limits(request)
    inputs = _inputs(request)
    return {
        "schema_version": LAMBDA_PROVIDER_READINESS_MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": (
            "ready_for_explicit_paid_provider_attempt"
            if not launch_blockers
            else "blocked_before_paid_provider_attempt"
        ),
        "provider": LAMBDA_PROVIDER_NAME,
        "mode": mode,
        "job_id": _string(request.get("job_id")),
        "source_artifacts": {
            "provider_launch_request_path": str(request_path),
            "lambda_provider_adapter_result_path": str(output_path),
            "provider_worker_endpoint_manifest_path": str(
                provider_worker_endpoint_manifest_path
            ),
        },
        "api_call_performed": False,
        "live_provider_call_authorized": False,
        "api_key_readiness": dict(api_key_meta),
        "lambda_launch_contract": {
            "api_base": _lambda_api_base(),
            "launch_endpoint": "/instance-operations/launch",
            "terminate_endpoint": "/instance-operations/terminate",
            "auth_scheme": "bearer",
            "ssh_key_names_required_count": 1,
            "launch_rate_limit_note": (
                "official spec: one launch request per 12 seconds or five per minute"
            ),
        },
        "provider_inputs": {
            "manifest_uri_present": bool(_input_value(inputs, "manifest_uri")),
            "manifest_uri_fetchable_by_provider": inputs.get(
                "manifest_uri_fetchable_by_provider"
            )
            is True,
            "capture_root_bundle_uri_present": bool(
                _input_value(inputs, "capture_root_bundle_uri")
            ),
            "capture_root_bundle_uri_fetchable_by_provider": inputs.get(
                "capture_root_bundle_uri_fetchable_by_provider"
            )
            is True,
        },
        "artifact_output": {
            "artifact_output_uri_present": bool(
                _input_value(inputs, "artifact_output_uri")
            ),
            "artifact_output_uri_scheme": urllib.parse.urlparse(
                _input_value(inputs, "artifact_output_uri")
            ).scheme
            or None,
            "artifact_output_uri_provider_writable": inputs.get(
                "artifact_output_uri_provider_writable"
            )
            is True,
            "artifact_output_write_auth_contract_ready": inputs.get(
                "artifact_output_write_auth_contract_ready"
            )
            is True,
        },
        "spend_limits": {
            "requested_budget_usd": _number(limits.get("requested_budget_usd")),
            "hard_timeout_seconds": _number(limits.get("hard_timeout_seconds")),
            "idle_timeout_seconds": _number(limits.get("idle_timeout_seconds")),
            "external_watchdog_ttl_seconds": _number(
                limits.get("external_watchdog_ttl_seconds")
            ),
            "max_active_workers": _number(limits.get("max_active_workers")),
            "bounded_single_worker_attempt": _number(limits.get("max_active_workers"))
            == 1,
            "idle_shutdown_required": _bool(limits.get("idle_shutdown_required"))
            is True,
        },
        "user_data": dict(user_data_metadata),
        "blockers": launch_blockers,
        "claim_boundary": {
            "lambda_launch_request_is_not_worker_ready_proof": True,
            "lambda_launch_request_is_not_simulator_execution_proof": True,
            "lambda_termination_request_is_not_full_cost_reconciliation": True,
            "generated_world_rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _adapter_event_name(status: str) -> str:
    if status in {"blocked", "failed"}:
        return f"lambda_provider_adapter.{status}"
    return "lambda_provider_adapter.completed"


def _persist_result(output_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    persisted = _redact_runtime_value(dict(result), api_key=_string(result.get("_api_key")))
    persisted.pop("_api_key", None)
    write_json(output_path, persisted)
    blockers = [b for b in persisted.get("blockers", []) if isinstance(b, str)]
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
        lambda_side_effects_may_have_occurred=persisted.get(
            "lambda_side_effects_may_have_occurred"
        ),
        http_status_code=persisted.get("http_status_code"),
        provider_job_submitted=persisted.get("provider_job_submitted"),
    )
    return persisted


def run_lambda_provider_adapter(
    *,
    provider_launch_request_path: str | Path,
    output_path: str | Path | None = None,
    mode: str = "dry-run",
    allow_lambda_api_call: bool = False,
    region_name: str | None = None,
    instance_type_name: str | None = None,
    ssh_key_name: str | None = None,
    file_system_names: Sequence[str] | None = None,
    image_id: str | None = None,
    image_family: str | None = None,
    firewall_ruleset_ids: Sequence[str] | None = None,
    instance_name: str | None = None,
    hostname: str | None = None,
    user_data_file: str | None = None,
    instance_ids: Sequence[str] | None = None,
    timeout_seconds: int = 30,
    teardown_poll_attempts: int = 3,
    teardown_poll_interval_seconds: float = 2.0,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None = None,
) -> Dict[str, Any]:
    request_path = Path(provider_launch_request_path).resolve()
    resolved_output = (
        Path(output_path).resolve()
        if output_path
        else Path(
            os.getenv(PROVIDER_ADAPTER_OUTPUT_ENV)
            or request_path.parent / "lambda_provider_adapter_result.json"
        ).resolve()
    )
    ensure_dir(resolved_output.parent)
    payload = read_json_any(request_path)
    request = dict(payload) if isinstance(payload, Mapping) else {}
    if mode == "auto":
        mode = "launch-instance"
    api_key, api_key_meta = _read_lambda_api_key()
    config = _launch_config(
        request,
        region_name=region_name,
        instance_type_name=instance_type_name,
        ssh_key_name=ssh_key_name,
        file_system_names=file_system_names,
        image_id=image_id,
        image_family=image_family,
        firewall_ruleset_ids=firewall_ruleset_ids,
        instance_name=instance_name,
        hostname=hostname,
        user_data_file=user_data_file,
    )
    lambda_request: dict[str, Any] = {}
    user_data_meta: dict[str, Any] = {}
    launch_config_blockers: list[str] = []
    if request:
        lambda_request, user_data_meta, launch_config_blockers = _launch_payload(
            request,
            config,
        )
    result: Dict[str, Any] = {
        "schema_version": LAMBDA_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "provider_launch_request_path": str(request_path),
        "output_path": str(resolved_output),
        "mode": mode,
        "job_id": _string(request.get("job_id")),
        "provider": _string(request.get("provider")) or LAMBDA_PROVIDER_NAME,
        "api_call_performed": False,
        "lambda_side_effects_may_have_occurred": False,
        "live_provider_call_proven": False,
        "provider_allocation_proven": False,
        "provider_job_submitted": False,
        "simulator_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
        "raw_api_key_stored": False,
        "secret_values_in_artifact": False,
        "api_key_readiness": api_key_meta,
        "lambda_api_base": _lambda_api_base(),
        "lambda_cloud_api_version": "1.10.0",
        "lambda_doc_sources": LAMBDA_DOC_SOURCES,
        "request_summary": _request_summary(request),
        "lambda_launch_config": {
            "region_name_present": bool(_string(config.get("region_name"))),
            "instance_type_name_present": bool(_string(config.get("instance_type_name"))),
            "ssh_key_names_count": len(_string_list(config.get("ssh_key_names"))),
            "file_system_names_count": len(_string_list(config.get("file_system_names"))),
            "image_configured": bool(config.get("image")),
            "firewall_rulesets_count": len(config.get("firewall_rulesets") or []),
            "name": config.get("name"),
            "hostname_present": bool(_string(config.get("hostname"))),
        },
        "lambda_request": _redact_runtime_value(lambda_request, api_key=api_key),
        "user_data": user_data_meta,
        "proof_boundary": (
            "This adapter validates and optionally submits Lambda Cloud API requests. "
            "It does not prove GPU allocation, worker readiness, simulator execution, "
            "artifact upload, teardown, safety, or rank fidelity by itself."
        ),
        "_api_key": api_key,
    }
    log_event(
        logger,
        logging.INFO,
        "lambda_provider_adapter.started",
        provider_launch_request_path=str(request_path),
        output_path=str(resolved_output),
        job_id=request.get("job_id"),
        provider=request.get("provider"),
        mode=mode,
        allow_lambda_api_call=allow_lambda_api_call,
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

    endpoint_manifest_mode = "launch-instance" if mode == "dry-run" else mode
    provider_worker_endpoint_manifest = write_provider_worker_endpoint_manifest(
        output_dir=request_path.parent,
        provider=LAMBDA_PROVIDER_NAME,
        mode=endpoint_manifest_mode,
        job_id=_string(request.get("job_id")),
        provider_request_shape=_provider_shape(request),
    )
    endpoint_manifest_path = request_path.parent / "provider_worker_endpoint_manifest.json"
    result.update(
        {
            "provider_worker_endpoint_manifest_path": str(endpoint_manifest_path),
            "provider_worker_endpoint_manifest": provider_worker_endpoint_manifest,
        }
    )
    request_blockers = _request_blockers(request, mode=mode)
    result["request_blockers"] = request_blockers
    result["launch_config_blockers"] = launch_config_blockers
    readiness_manifest = _provider_readiness_manifest(
        request_path=request_path,
        output_path=resolved_output,
        request=request,
        mode=mode,
        request_blockers=request_blockers,
        launch_config_blockers=launch_config_blockers,
        provider_worker_endpoint_manifest_path=endpoint_manifest_path,
        api_key_meta=api_key_meta,
        user_data_metadata=user_data_meta,
    )
    write_json(_readiness_manifest_path(resolved_output), readiness_manifest)
    result["provider_readiness_manifest_path"] = str(
        _readiness_manifest_path(resolved_output)
    )
    result["provider_readiness_manifest"] = readiness_manifest

    if request_blockers:
        result.update(
            {
                "status": "blocked",
                "reason": "lambda_request_not_launchable",
                "blockers": request_blockers,
            }
        )
        return _persist_result(resolved_output, result)
    if mode == "dry-run":
        result.update(
            {
                "status": "dry_run_ready",
                "reason": "lambda_cloud_request_shape_validated_without_api_call",
                "blockers": [],
            }
        )
        return _persist_result(resolved_output, result)
    if mode not in API_MODES:
        result.update(
            {
                "status": "blocked",
                "reason": "unsupported_lambda_adapter_mode",
                "blockers": [f"unsupported_lambda_adapter_mode:{mode}"],
            }
        )
        return _persist_result(resolved_output, result)
    if mode in LIVE_LAUNCH_MODES and launch_config_blockers:
        result.update(
            {
                "status": "blocked",
                "reason": "lambda_launch_config_blocked",
                "blockers": launch_config_blockers,
            }
        )
        return _persist_result(resolved_output, result)
    terminate_ids = _split_ids(instance_ids)
    if mode == TERMINATE_MODE and not terminate_ids:
        result.update(
            {
                "status": "blocked",
                "reason": "lambda_termination_instance_ids_missing",
                "blockers": [f"missing_env_{LAMBDA_INSTANCE_IDS_ENV}"],
            }
        )
        return _persist_result(resolved_output, result)

    gate_blockers = _api_gate_blockers(
        allow_lambda_api_call=allow_lambda_api_call,
        api_key=api_key,
    )
    if gate_blockers:
        result.update(
            {
                "status": "blocked",
                "reason": "lambda_api_gate_blocked",
                "blockers": gate_blockers,
            }
        )
        return _persist_result(resolved_output, result)

    if mode in MUTATING_API_MODES:
        try:
            require_paid_resource_admission_grant(
                paid_resource_admission_grant,
                resource_class="lambda_provider_adapter",
            )
        except PaidResourceAdmissionBlocked as exc:
            result.update(
                {
                    "status": "blocked",
                    "reason": "shared_paid_resource_admission_blocked",
                    "blockers": [
                        "lambda_provider_shared_admission_missing_or_invalid",
                        *exc.blockers,
                    ],
                    "provider_mutations_performed": 0,
                }
            )
            return _persist_result(resolved_output, result)

    if mode in READ_ONLY_API_MODES:
        api_request = {
            "list-instances": {"url": f"{_lambda_api_base()}/instances", "method": "GET"},
            "list-instance-types": {
                "url": f"{_lambda_api_base()}/instance-types",
                "method": "GET",
            },
            "list-ssh-keys": {"url": f"{_lambda_api_base()}/ssh-keys", "method": "GET"},
            "list-images": {"url": f"{_lambda_api_base()}/images", "method": "GET"},
            "list-regions": {"url": f"{_lambda_api_base()}/regions", "method": "GET"},
        }[mode]
        payload_for_api: Mapping[str, Any] | None = None
    elif mode == TERMINATE_MODE:
        api_request = {
            "url": f"{_lambda_api_base()}/instance-operations/terminate",
            "method": "POST",
        }
        payload_for_api = {"instance_ids": terminate_ids}
        result["lambda_termination_request"] = {
            "url": api_request["url"],
            "method": api_request["method"],
            "body": {"instance_ids": terminate_ids},
        }
    else:
        api_request = {
            "url": lambda_request["url"],
            "method": lambda_request["method"],
        }
        payload_for_api = _mapping(lambda_request.get("body"))
    try:
        status_code, response = _http_json(
            url=api_request["url"],
            payload=payload_for_api,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            method=api_request["method"],
        )
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed_error: Any = json.loads(response_body)
        except json.JSONDecodeError:
            parsed_error = {"body": response_body}
        result.update(
            {
                "status": "failed",
                "reason": "lambda_api_http_error",
                "blockers": ["lambda_api_http_error"],
                "api_call_performed": True,
                "lambda_side_effects_may_have_occurred": mode in MUTATING_API_MODES,
                "http_status_code": exc.code,
                "lambda_api_error": _redact_runtime_value(parsed_error, api_key=api_key),
            }
        )
        return _persist_result(resolved_output, result)
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        result.update(
            {
                "status": "failed",
                "reason": "lambda_api_request_failed",
                "blockers": ["lambda_api_request_failed"],
                "api_call_performed": True,
                "lambda_side_effects_may_have_occurred": mode in MUTATING_API_MODES,
                "error_type": type(exc).__name__,
                "error": _redact_text(str(exc), api_key=api_key),
            }
        )
        return _persist_result(resolved_output, result)

    redacted_response = _redact_runtime_value(response, api_key=api_key)
    result.update(
        {
            "api_call_performed": True,
            "http_status_code": status_code,
            "lambda_response": redacted_response,
        }
    )
    if mode in READ_ONLY_API_MODES:
        result.update(
            {
                "status": "completed",
                "reason": "lambda_read_only_api_call_completed",
                "blockers": [],
                "lambda_side_effects_may_have_occurred": False,
            }
        )
        return _persist_result(resolved_output, result)
    if mode == TERMINATE_MODE:
        verification = _verify_lambda_teardown(
            instance_ids=terminate_ids,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            attempts=teardown_poll_attempts,
            poll_interval_seconds=teardown_poll_interval_seconds,
        )
        teardown_proven = verification.get("api_confirmed") is True
        teardown = {
            "schema_version": LAMBDA_TEARDOWN_MANIFEST_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "completed" if teardown_proven else "termination_unverified",
            "provider": LAMBDA_PROVIDER_NAME,
            "job_id": _string(request.get("job_id")),
            "instance_ids": terminate_ids,
            "api_call_performed": True,
            "http_status_code": status_code,
            "termination_response": redacted_response,
            "teardown_verification": verification,
            "provider_api_terminal_status_confirmed": teardown_proven,
            "continuing_spend_requires_followup_list_instances": not teardown_proven,
            "open_billing_risk": not teardown_proven,
            "claim_boundary": {
                "termination_request_is_not_full_cost_reconciliation": not teardown_proven,
                "list_instances_followup_required_for_zero_live_instance_proof": not teardown_proven,
                "provider_api_terminal_status_required_for_teardown_proof": True,
            },
        }
        write_json(_teardown_manifest_path(resolved_output), teardown)
        result.update(
            {
                "status": "completed" if teardown_proven else "termination_unverified",
                "reason": (
                    "lambda_termination_verified"
                    if teardown_proven
                    else "lambda_termination_request_completed_without_terminal_proof"
                ),
                "blockers": [] if teardown_proven else _string_list(verification.get("blockers")),
                "lambda_side_effects_may_have_occurred": True,
                "provider_teardown_requested": True,
                "provider_teardown_proven": teardown_proven,
                "open_billing_risk": not teardown_proven,
                "provider_teardown_manifest_path": str(
                    _teardown_manifest_path(resolved_output)
                ),
                "provider_teardown_manifest": teardown,
            }
        )
        return _persist_result(resolved_output, result)

    instance_ids_launched = _string_list(_mapping(response.get("data")).get("instance_ids"))
    result.update(
        {
            "status": "submitted",
            "reason": "lambda_launch_request_submitted",
            "blockers": [],
            "lambda_side_effects_may_have_occurred": True,
            "provider_job_submitted": bool(instance_ids_launched),
            "provider_allocation_proven": False,
            "live_provider_call_proven": True,
            "lambda_instance_ids": instance_ids_launched,
            "followup_required": [
                "poll Lambda instance details until running",
                "probe provider worker /readyz before using policy endpoint",
                "collect artifact upload proof before shutdown",
                "terminate instance with lambda provider adapter terminate-instances",
                "list instances after termination for zero-live-instance proof",
            ],
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
        description="Validate or submit a gated Lambda Cloud robot-eval provider request."
    )
    parser.add_argument("--provider-launch-request")
    parser.add_argument("--output-path")
    parser.add_argument(
        "--mode",
        choices=[
            "dry-run",
            "auto",
            "allocate",
            "launch-instance",
            "terminate-instances",
            "list-instances",
            "list-instance-types",
            "list-ssh-keys",
            "list-images",
            "list-regions",
        ],
        default="dry-run",
    )
    parser.add_argument("--lambda-region-name")
    parser.add_argument("--lambda-instance-type-name")
    parser.add_argument("--lambda-ssh-key-name")
    parser.add_argument("--lambda-file-system-name", action="append", default=[])
    parser.add_argument("--lambda-image-id")
    parser.add_argument("--lambda-image-family")
    parser.add_argument("--lambda-firewall-ruleset-id", action="append", default=[])
    parser.add_argument("--lambda-instance-name")
    parser.add_argument("--lambda-hostname")
    parser.add_argument("--lambda-user-data-file")
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--teardown-poll-attempts", type=int, default=3)
    parser.add_argument("--teardown-poll-interval-seconds", type=float, default=2.0)
    parser.add_argument(
        "--allow-lambda-api-call",
        action="store_true",
        help=f"Required with {LAMBDA_API_GATE_ENV}=true for Lambda Cloud API calls.",
    )
    args = parser.parse_args(argv)
    if args.mode in MUTATING_API_MODES:
        print("legacy_lambda_provider_mutation_cli_disabled", file=sys.stderr)
        return 2
    try:
        request_path = _request_path_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    result = run_lambda_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=args.output_path,
        mode=args.mode,
        allow_lambda_api_call=args.allow_lambda_api_call,
        region_name=args.lambda_region_name,
        instance_type_name=args.lambda_instance_type_name,
        ssh_key_name=args.lambda_ssh_key_name,
        file_system_names=args.lambda_file_system_name,
        image_id=args.lambda_image_id,
        image_family=args.lambda_image_family,
        firewall_ruleset_ids=args.lambda_firewall_ruleset_id,
        instance_name=args.lambda_instance_name,
        hostname=args.lambda_hostname,
        user_data_file=args.lambda_user_data_file,
        instance_ids=args.instance_id,
        timeout_seconds=args.timeout_seconds,
        teardown_poll_attempts=args.teardown_poll_attempts,
        teardown_poll_interval_seconds=args.teardown_poll_interval_seconds,
    )
    print(f"[lambda-provider-adapter] result={result['output_path']}")
    print(f"[lambda-provider-adapter] status={result['status']}")
    print(f"[lambda-provider-adapter] mode={result.get('mode')}")
    blockers = result.get("blockers")
    if blockers:
        print("[lambda-provider-adapter] blockers=" + ",".join(blockers))
    return 0 if result["status"] in {"dry_run_ready", "completed", "submitted", "termination_requested"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
