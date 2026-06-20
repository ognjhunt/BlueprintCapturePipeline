"""Prepared worker entrypoint for robot-eval jobs.

The website/control plane should enqueue a manifest URI. A prepared worker image
can run this module, load that manifest, execute the existing deterministic job
orchestrator, and copy artifacts to a configured destination before shutdown.
Live GPU/provider execution still requires the same explicit env and CLI gates as
the orchestrator.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .common import (
    ensure_dir,
    ensure_local_uri_path,
    parse_gs_uri,
    read_json_any,
    utc_now_iso,
    write_json,
)
from .cpu_simulator_preflight import CPU_BACKENDS
from .robot_eval_job_orchestrator import (
    ISAAC_SIMULATORS,
    PROVISIONERS,
    SIMULATORS,
    _artifact_output_write_auth_contract,
    _remote_cloud_execution_closure_manifest,
    _run_command_simulator,
    build_robot_eval_job,
)
from .wam_provider_runtime import parse_wam_provider_commands


WORKER_RUNTIME_MANIFEST_SCHEMA_VERSION = "robot_eval_worker_runtime_manifest.v1"
WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION = "robot_eval_worker_runtime_preflight.v1"
WORKER_INPUT_MANIFEST_SCHEMA_VERSION = "robot_eval_worker_manifest.v1"
RUNTIME_PREFLIGHT_COMMAND_ENV = "BLUEPRINT_RUNTIME_PREFLIGHT_COMMAND"
RUNTIME_PREFLIGHT_DETAIL_OUTPUT_ENV = "BLUEPRINT_RUNTIME_PREFLIGHT_DETAIL_OUTPUT"
SENSITIVE_ENV_NAME_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
SIGNED_URL_SIGNATURE_PARAM = "x-goog-" + "signature="
SIGNED_URL_QUERY_PATTERN = re.compile(
    rf"([?&]{SIGNED_URL_SIGNATURE_PARAM})[^\s\"'&]+",
    flags=re.IGNORECASE,
)
REMOTE_ARTIFACT_OUTPUT_URI_SCHEMES = {"gs", "s3", "r2"}


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [item for item in (_string(item) for item in value) if item]
    return []


def _dedupe(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
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


def _artifact_output_uri_is_provider_writable(
    uri: str, *, live_provider: bool
) -> bool:
    if not uri:
        return False
    scheme = urllib.parse.urlparse(uri).scheme or "local"
    if live_provider:
        return scheme in REMOTE_ARTIFACT_OUTPUT_URI_SCHEMES
    return scheme in {"local", "file", *REMOTE_ARTIFACT_OUTPUT_URI_SCHEMES}


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _secret_env_var_names(payload: Mapping[str, Any]) -> List[str]:
    secret_policy = _mapping(payload.get("secret_policy"))
    explicit_names = [
        *_string_list(payload.get("secret_env_var_names")),
        *_string_list(payload.get("runtime_preflight_secret_env_var_names")),
        *_string_list(secret_policy.get("provider_credential_env_vars")),
        *_string_list(secret_policy.get("storage_credential_env_vars")),
    ]
    ambient_names = [
        name
        for name in os.environ
        if any(marker in name.upper() for marker in SENSITIVE_ENV_NAME_MARKERS)
    ]
    return _dedupe([*explicit_names, *ambient_names])


def _secret_values_from_env(env: Mapping[str, str], names: Sequence[str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for name in names:
        value = env.get(name)
        if value and len(value) >= 4:
            values.setdefault(value, name)
    return values


def _output_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _redact_text(value: Any, secret_values: Mapping[str, str]) -> str:
    text = _output_text(value)
    for secret_value, env_name in sorted(
        secret_values.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        text = text.replace(secret_value, f"<redacted:{env_name}>")
    return _redact_signed_url_text(text)


def _redact_signed_url_text(text: str) -> str:
    if SIGNED_URL_SIGNATURE_PARAM not in text.lower():
        return text
    return SIGNED_URL_QUERY_PATTERN.sub(r"\1<redacted:signed-url-signature>", text)


def _redact_runtime_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_signed_url_text(value)
    if isinstance(value, list):
        return [_redact_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_runtime_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _redact_runtime_value(item) for key, item in value.items()}
    return value


def _write_redacted_json(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    redacted = _redact_runtime_value(payload)
    if not isinstance(redacted, Mapping):  # pragma: no cover - defensive
        redacted = dict(payload)
    redacted_dict = dict(redacted)
    write_json(path, redacted_dict)
    return redacted_dict


def _optional_json_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = read_json_any(path)
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _log_redaction_summary(secret_values: Mapping[str, str]) -> Dict[str, Any]:
    return {
        "stdout_stderr_secret_redaction_enabled": True,
        "redacted_secret_env_var_names": sorted(set(secret_values.values())),
        "redacted_secret_value_count": len(secret_values),
    }


def _command_summary(argv: Sequence[str]) -> Dict[str, Any]:
    return {
        "shell": False,
        "executable": Path(argv[0]).name if argv else "",
        "argv_count": len(argv),
        "argument_count": max(len(argv) - 1, 0),
        "arguments_redacted": max(len(argv) - 1, 0),
        "raw_command_stored": False,
    }


def _runtime_preflight_contract_blockers(
    *,
    simulator: str,
    contract: Mapping[str, Any],
) -> List[str]:
    blockers: List[str] = []
    required_checks = set(_string_list(contract.get("required_checks")))
    if contract.get("required_before_scene_load") is not True:
        blockers.append("runtime_preflight_not_required_before_scene_load")
    if contract.get("worker_blocks_scene_load_on_failed_preflight") is not True:
        blockers.append("runtime_preflight_does_not_block_scene_load")
    if contract.get("run_before") != "scene_load_and_policy_execution":
        blockers.append("runtime_preflight_order_not_before_scene_load")
    if contract.get("result_artifact") != "worker_runtime_preflight.json":
        blockers.append("runtime_preflight_result_artifact_missing")
    if not required_checks:
        blockers.append("runtime_preflight_required_checks_missing")
    if contract.get("runtime_preflight_is_not_simulator_proof") is not True:
        blockers.append("runtime_preflight_proof_boundary_missing")
    if simulator in ISAAC_SIMULATORS:
        for check in (
            "nvidia_smi_gpu_inventory",
            "driver_version",
            "vulkan_device",
            "rtx_renderer_available",
            "isaac_headless_launch",
            "blank_scene_load",
            "test_frame_render",
        ):
            if check not in required_checks:
                blockers.append(f"runtime_preflight_missing_check:{check}")
        if contract.get("vulkan_required") is not True:
            blockers.append("runtime_preflight_vulkan_not_required_for_isaac")
        if contract.get("test_frame_render_required") is not True:
            blockers.append("runtime_preflight_frame_render_not_required_for_isaac")
    return _dedupe(blockers)


def _runtime_preflight_command(payload: Mapping[str, Any], simulator: str) -> str:
    command = _string(payload.get("runtime_preflight_command"))
    if command:
        return command
    command_map = _mapping(payload.get("runtime_preflight_commands"))
    command = _string(command_map.get(simulator))
    if command:
        return command
    return _string(os.getenv(RUNTIME_PREFLIGHT_COMMAND_ENV))


def _write_worker_runtime_preflight(
    *,
    work_dir: Path,
    manifest_uri: str,
    job_id: str,
    capture_root: str,
    provisioner: str,
    simulator: str,
    contract: Mapping[str, Any],
    allow_simulator_execution: bool,
    timeout_seconds: int,
    generated_at: str,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    ensure_dir(work_dir)
    output_path = work_dir / "worker_runtime_preflight.json"
    if simulator == "fixture":
        result = {
            "schema_version": WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "not_required_for_fixture",
            "manifest_uri": manifest_uri,
            "job_id": job_id,
            "capture_root": capture_root,
            "provisioner": provisioner,
            "simulator": simulator,
            "runtime_preflight_contract": dict(contract),
            "execution_performed": False,
            "secret_values_in_artifact": False,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": [],
        }
        result = _write_redacted_json(output_path, result)
        return result
    if not allow_simulator_execution:
        result = {
            "schema_version": WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "not_run_missing_simulator_execution_gate",
            "manifest_uri": manifest_uri,
            "job_id": job_id,
            "capture_root": capture_root,
            "provisioner": provisioner,
            "simulator": simulator,
            "runtime_preflight_contract": dict(contract),
            "execution_performed": False,
            "secret_values_in_artifact": False,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": ["missing_simulator_execution_gate_for_runtime_preflight"],
        }
        result = _write_redacted_json(output_path, result)
        return result
    command_text = _runtime_preflight_command(payload, simulator)
    if not command_text:
        result = {
            "schema_version": WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "manifest_uri": manifest_uri,
            "job_id": job_id,
            "capture_root": capture_root,
            "provisioner": provisioner,
            "simulator": simulator,
            "runtime_preflight_contract": dict(contract),
            "execution_performed": False,
            "secret_values_in_artifact": False,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": ["missing_runtime_preflight_command"],
        }
        result = _write_redacted_json(output_path, result)
        return result
    try:
        argv = shlex.split(command_text)
    except ValueError as exc:
        result = {
            "schema_version": WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "manifest_uri": manifest_uri,
            "job_id": job_id,
            "capture_root": capture_root,
            "provisioner": provisioner,
            "simulator": simulator,
            "runtime_preflight_contract": dict(contract),
            "execution_performed": False,
            "secret_values_in_artifact": False,
            "command_parse_error": str(exc),
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": ["invalid_runtime_preflight_command"],
        }
        result = _write_redacted_json(output_path, result)
        return result
    stdout_path = work_dir / "worker_runtime_preflight.stdout.log"
    stderr_path = work_dir / "worker_runtime_preflight.stderr.log"
    detail_path = work_dir / "worker_runtime_preflight_detail.json"
    env = os.environ.copy()
    env["BLUEPRINT_RUNTIME_PREFLIGHT_OUTPUT"] = str(output_path)
    env[RUNTIME_PREFLIGHT_DETAIL_OUTPUT_ENV] = str(detail_path)
    env["BLUEPRINT_RUNTIME_PREFLIGHT_STDOUT"] = str(stdout_path)
    env["BLUEPRINT_RUNTIME_PREFLIGHT_STDERR"] = str(stderr_path)
    env["BLUEPRINT_CAPTURE_ROOT"] = capture_root
    env["BLUEPRINT_ROBOT_EVAL_JOB_ID"] = job_id
    env["BLUEPRINT_SIMULATOR_FRAMEWORK"] = simulator
    env["BLUEPRINT_WORKER_DIR"] = str(work_dir)
    command_summary = _command_summary(argv)
    secret_values = _secret_values_from_env(env, _secret_env_var_names(payload))
    redaction_summary = _log_redaction_summary(secret_values)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
        stdout_path.write_text(_redact_text(completed.stdout, secret_values), encoding="utf-8")
        stderr_path.write_text(_redact_text(completed.stderr, secret_values), encoding="utf-8")
        detail_payload = _optional_json_mapping(detail_path)
        success = completed.returncode == 0
        result = {
            "schema_version": WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "passed" if success else "blocked",
            "manifest_uri": manifest_uri,
            "job_id": job_id,
            "capture_root": capture_root,
            "provisioner": provisioner,
            "simulator": simulator,
            "runtime_preflight_contract": dict(contract),
            "command": command_summary,
            "execution_performed": True,
            "exit_code": completed.returncode,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "detail_path": str(detail_path) if detail_payload else None,
            "detail_status": detail_payload.get("status") if detail_payload else None,
            "detail_blockers": _string_list(detail_payload.get("blockers"))
            if detail_payload
            else [],
            "detail_artifact": detail_payload or None,
            "secret_values_in_artifact": False,
            **redaction_summary,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": [] if success else ["runtime_preflight_command_failed"],
        }
    except FileNotFoundError:
        result = {
            "schema_version": WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "manifest_uri": manifest_uri,
            "job_id": job_id,
            "capture_root": capture_root,
            "provisioner": provisioner,
            "simulator": simulator,
            "runtime_preflight_contract": dict(contract),
            "command": command_summary,
            "execution_performed": False,
            "secret_values_in_artifact": False,
            **redaction_summary,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": ["missing_runtime_preflight_command_dependency"],
        }
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(_redact_text(exc.stdout, secret_values), encoding="utf-8")
        stderr_path.write_text(_redact_text(exc.stderr, secret_values), encoding="utf-8")
        result = {
            "schema_version": WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "manifest_uri": manifest_uri,
            "job_id": job_id,
            "capture_root": capture_root,
            "provisioner": provisioner,
            "simulator": simulator,
            "runtime_preflight_contract": dict(contract),
            "command": command_summary,
            "execution_performed": True,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "secret_values_in_artifact": False,
            **redaction_summary,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": ["runtime_preflight_command_timeout"],
        }
    return _write_redacted_json(output_path, result)


def _parse_s3_compatible_uri(uri: str) -> Tuple[str, str, str]:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme not in {"s3", "r2"}:
        raise ValueError(f"Expected s3:// or r2:// URI, got: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid {parsed.scheme} URI: {uri}")
    return parsed.scheme, bucket, key


def _s3_compatible_endpoint_url(scheme: str) -> str | None:
    endpoint_url = (
        os.getenv("BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL")
        or os.getenv("BLUEPRINT_S3_ENDPOINT_URL")
        or os.getenv("AWS_ENDPOINT_URL")
        or os.getenv("R2_ENDPOINT_URL")
    )
    if scheme == "r2" and not endpoint_url:
        raise RuntimeError(
            "r2:// storage requires BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL or R2_ENDPOINT_URL"
        )
    return endpoint_url or None


def _s3_compatible_client(uri: str) -> Any:
    scheme, _, _ = _parse_s3_compatible_uri(uri)
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("boto3 is required for s3:// or r2:// worker storage") from exc
    kwargs: Dict[str, Any] = {}
    endpoint_url = _s3_compatible_endpoint_url(scheme)
    region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if region_name:
        kwargs["region_name"] = region_name
    return boto3.client("s3", **kwargs)


def _download_s3_compatible_uri(uri: str, target: Path) -> Path:
    _, bucket, key = _parse_s3_compatible_uri(uri)
    ensure_dir(target.parent)
    client = _s3_compatible_client(uri)
    client.download_file(bucket, key, str(target))
    return target


def _uri_to_local_path(uri: str, work_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme in {"", "file"}:
        return Path(urllib.request.url2pathname(parsed.path if parsed.scheme else uri))
    if parsed.scheme in {"http", "https"}:
        target = work_dir / "downloads" / "worker_manifest.json"
        ensure_dir(target.parent)
        with urllib.request.urlopen(uri, timeout=30) as response:
            target.write_bytes(response.read())
        return target
    if parsed.scheme == "gs":
        gcs_root = Path(os.getenv("BLUEPRINT_GCS_MOUNT_ROOT") or "/mnt/gcs")
        return ensure_local_uri_path(uri, gcs_root=gcs_root, scratch_dir=work_dir / "downloads")
    if parsed.scheme in {"s3", "r2"}:
        return _download_s3_compatible_uri(uri, work_dir / "downloads" / "worker_manifest.json")
    raise ValueError(f"Unsupported worker manifest URI scheme: {parsed.scheme}")


def _uri_to_local_file(uri: str, work_dir: Path, *, filename: str) -> Path:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme in {"", "file"}:
        return Path(urllib.request.url2pathname(parsed.path if parsed.scheme else uri))
    target = work_dir / "downloads" / filename
    ensure_dir(target.parent)
    if parsed.scheme in {"http", "https"}:
        with urllib.request.urlopen(uri, timeout=120) as response:
            target.write_bytes(response.read())
        return target
    if parsed.scheme == "gs":
        gcs_root = Path(os.getenv("BLUEPRINT_GCS_MOUNT_ROOT") or "/mnt/gcs")
        return ensure_local_uri_path(uri, gcs_root=gcs_root, scratch_dir=target.parent)
    if parsed.scheme in {"s3", "r2"}:
        return _download_s3_compatible_uri(uri, target)
    raise ValueError(f"Unsupported capture root bundle URI scheme: {parsed.scheme}")


def _safe_archive_member(name: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def _find_extracted_capture_root(extract_dir: Path) -> Path | None:
    if (extract_dir / "capture_descriptor.json").is_file():
        return extract_dir
    direct_children = [path for path in extract_dir.iterdir() if path.is_dir()]
    for child in direct_children:
        if (child / "capture_descriptor.json").is_file():
            return child
    for descriptor in extract_dir.rglob("capture_descriptor.json"):
        return descriptor.parent
    return None


def _extract_capture_root_bundle(uri: str, work_dir: Path) -> Dict[str, Any]:
    archive_name = Path(urllib.parse.urlparse(uri).path).name or "capture-root-bundle.zip"
    archive_path = _uri_to_local_file(uri, work_dir, filename=archive_name)
    extract_dir = work_dir / "capture_root_bundle"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    ensure_dir(extract_dir)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            unsafe = [name for name in archive.namelist() if not _safe_archive_member(name)]
            if unsafe:
                raise ValueError("capture root bundle contains unsafe zip paths")
            archive.extractall(extract_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            members = archive.getmembers()
            unsafe = [member.name for member in members if not _safe_archive_member(member.name)]
            if unsafe:
                raise ValueError("capture root bundle contains unsafe tar paths")
            archive.extractall(extract_dir)
    else:
        raise ValueError("capture root bundle must be a .zip or .tar archive")
    capture_root = _find_extracted_capture_root(extract_dir)
    if capture_root is None:
        raise ValueError("capture root bundle did not contain capture_descriptor.json")
    manifest = {
        "schema_version": "robot_eval_worker_capture_root_bundle.v1",
        "source_uri": uri,
        "archive_path": str(archive_path),
        "extract_dir": str(extract_dir),
        "capture_root": str(capture_root),
        "status": "extracted",
    }
    return _write_redacted_json(work_dir / "capture_root_bundle_manifest.json", manifest)


def _replace_capture_root_prefix(value: Any, *, old_root: str, new_root: str) -> Any:
    if isinstance(value, str):
        return new_root + value[len(old_root) :] if old_root and value.startswith(old_root) else value
    if isinstance(value, list):
        return [
            _replace_capture_root_prefix(item, old_root=old_root, new_root=new_root)
            for item in value
        ]
    if isinstance(value, Mapping):
        return {
            key: _replace_capture_root_prefix(item, old_root=old_root, new_root=new_root)
            for key, item in value.items()
        }
    return value


def _rebase_job_request_capture_root(
    job_request: Mapping[str, Any],
    *,
    selected_capture_root: str,
) -> Dict[str, Any]:
    request = dict(job_request)
    site_package = _mapping(request.get("site_package"))
    original_root = (
        _string(site_package.get("capture_root"))
        or _string(request.get("capture_root"))
    )
    if original_root and original_root != selected_capture_root:
        request = _replace_capture_root_prefix(
            request,
            old_root=original_root,
            new_root=selected_capture_root,
        )
        site_package = _mapping(request.get("site_package"))
    request["capture_root"] = selected_capture_root
    if site_package:
        site_package["capture_root"] = selected_capture_root
        request["site_package"] = site_package
    return request


def _load_manifest(uri: str, work_dir: Path) -> Dict[str, Any]:
    path = _uri_to_local_path(uri, work_dir)
    payload = read_json_any(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected worker manifest object at {uri}")
    return dict(payload)


def _copy_directory_contents(source_dir: Path, destination_dir: Path) -> None:
    ensure_dir(destination_dir)
    for source in source_dir.iterdir():
        destination = destination_dir / source.name
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            ensure_dir(destination.parent)
            shutil.copy2(source, destination)


def _upload_directory_to_gs(source_dir: Path, destination_uri: str) -> int:
    parsed = parse_gs_uri(destination_uri)
    try:
        from google.cloud import storage as gcs_storage  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("google-cloud-storage is required for gs:// artifact upload") from exc

    client = gcs_storage.Client()
    bucket = client.bucket(parsed.bucket)
    uploaded = 0
    prefix = parsed.key.rstrip("/")
    for source in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        relative = source.relative_to(source_dir).as_posix()
        bucket.blob(f"{prefix}/{relative}").upload_from_filename(str(source))
        uploaded += 1
    return uploaded


def _upload_directory_to_s3_compatible(source_dir: Path, destination_uri: str) -> int:
    _, bucket, key = _parse_s3_compatible_uri(destination_uri)
    client = _s3_compatible_client(destination_uri)
    uploaded = 0
    prefix = key.rstrip("/")
    for source in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        relative = source.relative_to(source_dir).as_posix()
        client.upload_file(str(source), bucket, f"{prefix}/{relative}")
        uploaded += 1
    return uploaded


def _copy_runtime_manifest_to_artifact_output(
    *,
    runtime_manifest_path: Path,
    artifact_output_uri: str,
) -> Dict[str, Any]:
    relative_path = "worker_runtime_manifest.json"
    parsed = urllib.parse.urlparse(artifact_output_uri)
    if parsed.scheme in {"", "file"}:
        destination_root = Path(
            urllib.request.url2pathname(
                parsed.path if parsed.scheme else artifact_output_uri
            )
        )
        destination = destination_root / relative_path
        ensure_dir(destination.parent)
        shutil.copy2(runtime_manifest_path, destination)
        return {
            "status": "completed",
            "destination_uri": artifact_output_uri,
            "relative_path": relative_path,
            "destination_path": str(destination),
        }
    if parsed.scheme == "gs":
        gs_uri = parse_gs_uri(artifact_output_uri)
        try:
            from google.cloud import storage as gcs_storage  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("google-cloud-storage is required for gs:// artifact upload") from exc
        client = gcs_storage.Client()
        key = f"{gs_uri.key.rstrip('/')}/{relative_path}"
        client.bucket(gs_uri.bucket).blob(key).upload_from_filename(
            str(runtime_manifest_path)
        )
        return {
            "status": "completed",
            "destination_uri": artifact_output_uri,
            "relative_path": relative_path,
            "object_key": key,
        }
    if parsed.scheme in {"s3", "r2"}:
        scheme, bucket, key = _parse_s3_compatible_uri(artifact_output_uri)
        client = _s3_compatible_client(artifact_output_uri)
        object_key = f"{key.rstrip('/')}/{relative_path}"
        client.upload_file(str(runtime_manifest_path), bucket, object_key)
        return {
            "status": "completed",
            "destination_uri": artifact_output_uri,
            "relative_path": relative_path,
            "object_key": object_key,
            "storage_scheme": scheme,
        }
    return {
        "status": "blocked",
        "destination_uri": artifact_output_uri,
        "relative_path": relative_path,
        "blockers": [f"unsupported_artifact_output_uri_scheme:{parsed.scheme or 'local'}"],
    }


def _upload_runtime_manifest_to_signed_put_url(runtime_manifest_path: Path) -> Dict[str, Any]:
    signed_put_url = os.getenv("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL", "").strip()
    if not signed_put_url:
        return {"status": "not_configured"}
    request = urllib.request.Request(
        signed_put_url,
        data=runtime_manifest_path.read_bytes(),
        method="PUT",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        status_code = int(getattr(response, "status", 200))
    return {
        "status": "completed" if 200 <= status_code < 300 else "blocked",
        "http_status_code": status_code,
        "destination": "signed_put_url",
        "signed_url_stored": False,
    }


def _write_runtime_manifest_with_signed_put(
    runtime_manifest_path: Path,
    runtime_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    runtime_manifest = _write_redacted_json(runtime_manifest_path, runtime_manifest)
    if not os.getenv("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL", "").strip():
        return runtime_manifest
    try:
        signed_put_upload = _upload_runtime_manifest_to_signed_put_url(runtime_manifest_path)
    except Exception as exc:
        signed_put_upload = {
            "status": "blocked",
            "destination": "signed_put_url",
            "blockers": [f"signed_put_upload_failed:{type(exc).__name__}"],
        }
    runtime_manifest["signed_put_runtime_manifest_upload"] = signed_put_upload
    if signed_put_upload.get("status") == "blocked":
        runtime_manifest["status"] = "blocked"
        runtime_manifest["blockers"] = _dedupe(
            [
                *_string_list(runtime_manifest.get("blockers")),
                *_string_list(signed_put_upload.get("blockers")),
            ]
        )
    runtime_manifest = _write_redacted_json(runtime_manifest_path, runtime_manifest)
    if signed_put_upload.get("status") == "completed":
        try:
            final_upload = _upload_runtime_manifest_to_signed_put_url(runtime_manifest_path)
            runtime_manifest["signed_put_runtime_manifest_upload"] = {
                **signed_put_upload,
                "final_manifest_reupload_status": final_upload.get("status"),
                "final_manifest_reupload_http_status_code": final_upload.get(
                    "http_status_code"
                ),
            }
        except Exception as exc:
            runtime_manifest["signed_put_runtime_manifest_upload"] = {
                **signed_put_upload,
                "final_manifest_reupload_status": "blocked",
                "final_manifest_reupload_blockers": [
                    f"signed_put_final_upload_failed:{type(exc).__name__}"
                ],
            }
        runtime_manifest = _write_redacted_json(runtime_manifest_path, runtime_manifest)
    return runtime_manifest


def _copy_worker_runtime_files_to_artifact_output(
    *,
    work_dir: Path,
    artifact_output_uri: str,
    relative_paths: Sequence[str],
) -> Dict[str, Any]:
    existing = [
        (relative_path, work_dir / relative_path)
        for relative_path in relative_paths
        if (work_dir / relative_path).is_file()
    ]
    parsed = urllib.parse.urlparse(artifact_output_uri)
    if parsed.scheme in {"", "file"}:
        destination_root = Path(
            urllib.request.url2pathname(
                parsed.path if parsed.scheme else artifact_output_uri
            )
        )
        for relative_path, source in existing:
            destination = destination_root / relative_path
            ensure_dir(destination.parent)
            shutil.copy2(source, destination)
        return {
            "status": "completed",
            "destination_uri": artifact_output_uri,
            "copied_file_count": len(existing),
            "relative_paths": [relative_path for relative_path, _ in existing],
        }
    if parsed.scheme == "gs":
        gs_uri = parse_gs_uri(artifact_output_uri)
        try:
            from google.cloud import storage as gcs_storage  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("google-cloud-storage is required for gs:// artifact upload") from exc
        client = gcs_storage.Client()
        uploaded: List[str] = []
        for relative_path, source in existing:
            key = f"{gs_uri.key.rstrip('/')}/{relative_path}"
            client.bucket(gs_uri.bucket).blob(key).upload_from_filename(str(source))
            uploaded.append(key)
        return {
            "status": "completed",
            "destination_uri": artifact_output_uri,
            "uploaded_file_count": len(uploaded),
            "object_keys": uploaded,
        }
    if parsed.scheme in {"s3", "r2"}:
        scheme, bucket, key = _parse_s3_compatible_uri(artifact_output_uri)
        client = _s3_compatible_client(artifact_output_uri)
        uploaded = []
        for relative_path, source in existing:
            object_key = f"{key.rstrip('/')}/{relative_path}"
            client.upload_file(str(source), bucket, object_key)
            uploaded.append(object_key)
        return {
            "status": "completed",
            "destination_uri": artifact_output_uri,
            "uploaded_file_count": len(uploaded),
            "object_keys": uploaded,
            "storage_scheme": scheme,
        }
    return {
        "status": "blocked",
        "destination_uri": artifact_output_uri,
        "blockers": [f"unsupported_artifact_output_uri_scheme:{parsed.scheme or 'local'}"],
    }


def _copy_worker_runtime_files_to_job_dir(*, worker_dir: Path, job_dir: Path) -> None:
    downloaded_worker_manifest = worker_dir / "downloads" / "worker_manifest.json"
    if downloaded_worker_manifest.is_file():
        payload = read_json_any(downloaded_worker_manifest)
        if isinstance(payload, Mapping):
            _write_redacted_json(job_dir / "worker_manifest.json", payload)

    for relative_path in (
        "capture_root_bundle_manifest.json",
        "worker_runtime_preflight.json",
        "worker_runtime_preflight_detail.json",
        "worker_runtime_preflight.stdout.log",
        "worker_runtime_preflight.stderr.log",
    ):
        source = worker_dir / relative_path
        if source.is_file():
            shutil.copy2(source, job_dir / relative_path)


def _record_provider_runtime_gpu_time(
    *,
    job_dir: Path,
    started_monotonic: float,
    generated_at: str,
) -> Dict[str, Any]:
    if not _env_truthy("BLUEPRINT_ROBOT_EVAL_PROVIDER_RUNTIME"):
        return {"status": "not_provider_runtime"}
    ledger_path = job_dir / "gpu_cost_control_ledger.json"
    if not ledger_path.is_file():
        return {"status": "blocked", "blockers": ["missing_gpu_cost_control_ledger"]}
    payload = read_json_any(ledger_path)
    if not isinstance(payload, Mapping):
        return {"status": "blocked", "blockers": ["invalid_gpu_cost_control_ledger"]}
    ledger = dict(payload)
    gpu_time = _mapping(ledger.get("gpu_time"))
    elapsed_seconds = max(0.0, time.monotonic() - started_monotonic)
    gpu_time.update(
        {
            "actual_gpu_seconds": round(elapsed_seconds, 3),
            "actual_gpu_time_source": "worker_runtime_wall_clock_seconds",
            "actual_gpu_time_record_required": True,
            "actual_gpu_time_record_present": True,
            "actual_gpu_time_recorded_at": generated_at,
            "actual_gpu_time_is_provider_billing_record": False,
            "actual_gpu_time_claim_boundary": (
                "worker wall-clock runtime inside provider pod; use provider invoice "
                "or pod lifecycle records for billing reconciliation"
            ),
        }
    )
    ledger["gpu_time"] = gpu_time
    ledger["status"] = "provider_runtime_observed"
    ledger["live_provider_calls_performed"] = True
    ledger["execution_performed"] = True
    ledger["provider_runtime_accounting"] = {
        "status": "recorded",
        "source": "blueprint-run-robot-eval-worker",
        "generated_at": generated_at,
        "wall_clock_seconds": round(elapsed_seconds, 3),
        "provider_billing_record": False,
    }
    write_json(ledger_path, ledger)
    return dict(ledger["provider_runtime_accounting"])


def _provider_shutdown_evidence_from_runtime(job_dir: Path) -> Dict[str, Any]:
    configured = _string(os.getenv("BLUEPRINT_PROVIDER_SHUTDOWN_PROOF"))
    candidates = [job_dir / "provider_shutdown_proof.json"]
    if configured:
        candidates.insert(0, Path(configured))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        payload = read_json_any(candidate)
        if isinstance(payload, Mapping):
            return {
                "status": "provided",
                "source_path": str(candidate),
                **dict(payload),
            }
    return {
        "status": "missing",
        "proof_boundary": (
            "Worker finalizer completion is not provider deallocation proof. "
            "Record provider pod/worker termination or zero-active-worker evidence "
            "in provider_shutdown_proof.json before claiming clean shutdown."
        ),
    }


def _provider_shutdown_proven(evidence: Mapping[str, Any]) -> bool:
    return bool(
        evidence.get("provider_shutdown_proven")
        or evidence.get("clean_shutdown_proven")
        or evidence.get("zero_active_workers_after_run")
        or evidence.get("pod_terminated")
        or evidence.get("worker_terminated")
    )


def _record_provider_runtime_finalizer_proof(
    *,
    job_dir: Path,
    artifact_output_uri: str,
    artifact_upload: Mapping[str, Any],
    finalizer_refresh_upload: Mapping[str, Any],
    runtime_manifest_upload: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    if not _env_truthy("BLUEPRINT_ROBOT_EVAL_PROVIDER_RUNTIME"):
        return {"status": "not_provider_runtime"}
    ledger_path = job_dir / "gpu_cost_control_ledger.json"
    payload = read_json_any(ledger_path) if ledger_path.is_file() else {}
    if not isinstance(payload, Mapping):
        return {"status": "blocked", "blockers": ["invalid_gpu_cost_control_ledger"]}
    ledger = dict(payload)
    artifact_finalizer = _mapping(ledger.get("artifact_finalizer"))
    provider_shutdown_evidence = _provider_shutdown_evidence_from_runtime(job_dir)
    provider_shutdown_proven = _provider_shutdown_proven(provider_shutdown_evidence)
    artifact_upload_completed = artifact_upload.get("status") == "completed"
    finalizer_refresh_completed = finalizer_refresh_upload.get("status") == "completed"
    runtime_manifest_uploaded = runtime_manifest_upload.get("status") == "completed"

    def upload_evidence(payload: Mapping[str, Any]) -> Dict[str, Any]:
        keys = (
            "status",
            "destination_uri",
            "uploaded_file_count",
            "copied_file_count",
            "object_keys",
            "relative_paths",
            "object_key",
            "destination_path",
            "storage_scheme",
            "blockers",
        )
        return {key: payload.get(key) for key in keys if payload.get(key) not in (None, [])}

    artifact_upload_evidence = upload_evidence(artifact_upload)
    finalizer_refresh_upload_evidence = upload_evidence(finalizer_refresh_upload)
    runtime_manifest_upload_evidence = upload_evidence(runtime_manifest_upload)
    worker_finalizer_proven = bool(
        artifact_upload_completed and finalizer_refresh_completed and runtime_manifest_uploaded
    )
    blockers: list[str] = []
    if not artifact_upload_completed:
        blockers.append("artifact_upload_not_completed_before_shutdown")
    if not finalizer_refresh_completed:
        blockers.append("finalizer_refresh_upload_not_completed_before_shutdown")
    if not runtime_manifest_uploaded:
        blockers.append("worker_runtime_manifest_upload_not_completed_before_shutdown")
    if not provider_shutdown_proven:
        blockers.append("provider_shutdown_evidence_missing")
    proof = {
        "schema_version": "robot_eval_provider_runtime_finalizer_proof.v1",
        "generated_at": generated_at,
        "status": "completed" if worker_finalizer_proven else "blocked",
        "artifact_output_uri": artifact_output_uri,
        "artifact_upload_completed_before_shutdown": artifact_upload_completed,
        "finalizer_refresh_upload_completed_before_shutdown": finalizer_refresh_completed,
        "worker_runtime_manifest_upload_completed_before_shutdown": runtime_manifest_uploaded,
        "artifact_upload_evidence": artifact_upload_evidence,
        "finalizer_refresh_upload_evidence": finalizer_refresh_upload_evidence,
        "runtime_manifest_upload_evidence": runtime_manifest_upload_evidence,
        "worker_artifacts_finalized_before_shutdown": worker_finalizer_proven,
        "provider_shutdown_proven": provider_shutdown_proven,
        "provider_shutdown_evidence": provider_shutdown_evidence,
        "clean_shutdown_proven": worker_finalizer_proven and provider_shutdown_proven,
        "blockers": blockers,
        "claim_boundary": (
            "Worker finalizer proof shows Blueprint artifacts were copied before worker "
            "exit. Clean provider shutdown additionally requires provider lifecycle "
            "evidence such as pod termination or zero active workers after the run."
        ),
    }
    artifact_finalizer.update(
        {
            "worker_artifacts_finalized_before_shutdown": worker_finalizer_proven,
            "worker_finalizer_completed_before_shutdown": worker_finalizer_proven,
            "provider_shutdown_proven": provider_shutdown_proven,
            "provider_shutdown_evidence": provider_shutdown_evidence,
            "clean_shutdown_proven": proof["clean_shutdown_proven"],
            "finalizer_proof_path": "provider_runtime_finalizer_proof.json",
            "artifact_upload_evidence": artifact_upload_evidence,
            "finalizer_refresh_upload_evidence": finalizer_refresh_upload_evidence,
            "runtime_manifest_upload_evidence": runtime_manifest_upload_evidence,
        }
    )
    ledger["artifact_finalizer"] = artifact_finalizer
    ledger["provider_runtime_finalizer"] = proof
    write_json(job_dir / "provider_runtime_finalizer_proof.json", proof)
    write_json(ledger_path, ledger)
    return proof


def _refresh_remote_cloud_closure_after_worker_runtime(
    *,
    job_dir: Path,
    generated_at: str,
) -> Dict[str, Any]:
    run_manifest_path = job_dir / "job_run_manifest.json"
    run_manifest = _optional_json_mapping(run_manifest_path)
    worker_launch_plan = _optional_json_mapping(job_dir / "worker_launch_plan.json")
    worker_manifest = _optional_json_mapping(job_dir / "worker_manifest.json")
    provider_launch_request = _optional_json_mapping(job_dir / "gpu_provider_launch_request.json")
    gpu_result = _optional_json_mapping(job_dir / "gpu_provisioning_result.json")
    gpu_cost_ledger = _optional_json_mapping(job_dir / "gpu_cost_control_ledger.json")
    sim_result = _optional_json_mapping(job_dir / "simulator_service_result.json")
    if not gpu_cost_ledger:
        return {"status": "not_available", "reason": "missing_gpu_cost_control_ledger"}
    closure = _remote_cloud_execution_closure_manifest(
        job_id=_string(run_manifest.get("job_id")) or job_dir.name,
        provisioner=_string(run_manifest.get("provisioner"))
        or _string(gpu_cost_ledger.get("provider"))
        or _string(provider_launch_request.get("provider"))
        or "fixture_local",
        simulator=_string(run_manifest.get("simulator"))
        or _string(worker_manifest.get("simulator"))
        or "fixture",
        worker_launch_plan=worker_launch_plan,
        worker_manifest=worker_manifest,
        provider_launch_request=provider_launch_request,
        gpu_result=gpu_result,
        gpu_cost_ledger=gpu_cost_ledger,
        sim_result=sim_result,
        generated_at=generated_at,
    )
    write_json(job_dir / "remote_cloud_execution_closure_manifest.json", closure)
    if run_manifest:
        artifacts = dict(_mapping(run_manifest.get("artifacts")))
        artifacts["remote_cloud_execution_closure_manifest"] = (
            "remote_cloud_execution_closure_manifest.json"
        )
        artifacts["provider_runtime_finalizer_proof"] = "provider_runtime_finalizer_proof.json"
        run_manifest.update(
            {
                "remote_cloud_execution_closure_status": closure.get("status"),
                "remote_cloud_execution_proven": bool(
                    closure.get("remote_cloud_execution_proven")
                ),
                "remote_cloud_clean_shutdown_proven": bool(
                    closure.get("clean_shutdown_proven")
                ),
                "remote_cloud_runtime_blockers": _string_list(
                    closure.get("runtime_blockers")
                ),
                "artifacts": artifacts,
            }
        )
        write_json(run_manifest_path, run_manifest)
    return closure


def _refresh_job_startup_audit_with_worker_runtime(job_dir: Path) -> Dict[str, Any]:
    from .robot_eval_startup_architecture_audit import (
        build_robot_eval_startup_architecture_audit,
    )

    audit = build_robot_eval_startup_architecture_audit(
        job_dir=job_dir,
        output_path=job_dir / "startup_architecture_audit.json",
    )
    run_manifest_path = job_dir / "job_run_manifest.json"
    if not run_manifest_path.is_file():
        return audit
    payload = read_json_any(run_manifest_path)
    if not isinstance(payload, Mapping):
        return audit
    run_manifest = dict(payload)
    artifacts = dict(_mapping(run_manifest.get("artifacts")))
    for relative_path in (
        "capture_root_bundle_manifest.json",
        "startup_architecture_audit.json",
        "worker_runtime_manifest.json",
        "worker_runtime_preflight.json",
        "worker_runtime_preflight_detail.json",
        "worker_runtime_preflight.stdout.log",
        "worker_runtime_preflight.stderr.log",
    ):
        if (job_dir / relative_path).is_file():
            artifacts[Path(relative_path).stem] = relative_path
    run_manifest["startup_architecture_audit_status"] = audit.get("status")
    run_manifest["startup_architecture_audit_path"] = "startup_architecture_audit.json"
    run_manifest["startup_architecture_compliant"] = bool(
        audit.get("architecture_compliant")
    )
    run_manifest["artifacts"] = artifacts
    write_json(run_manifest_path, run_manifest)
    return audit


def _attach_worker_failure_artifact_upload(
    *,
    runtime_manifest: Dict[str, Any],
    work_dir: Path,
    artifact_output_uri: str | None,
) -> Dict[str, Any]:
    if not artifact_output_uri:
        return _write_runtime_manifest_with_signed_put(
            work_dir / "worker_runtime_manifest.json",
            runtime_manifest,
        )
    try:
        artifact_upload = _copy_worker_runtime_files_to_artifact_output(
            work_dir=work_dir,
            artifact_output_uri=artifact_output_uri,
            relative_paths=[
                "worker_runtime_preflight.json",
                "worker_runtime_preflight_detail.json",
                "worker_runtime_preflight.stdout.log",
                "worker_runtime_preflight.stderr.log",
                "worker_runtime_manifest.json",
            ],
        )
    except Exception as exc:
        artifact_upload = {
            "status": "blocked",
            "destination_uri": artifact_output_uri,
            "blockers": [f"artifact_upload_failed:{type(exc).__name__}"],
        }
    runtime_manifest["artifact_upload"] = artifact_upload
    runtime_manifest["blockers"] = _dedupe(
        [
            *_string_list(runtime_manifest.get("blockers")),
            *_string_list(artifact_upload.get("blockers")),
        ]
    )
    runtime_manifest = _write_runtime_manifest_with_signed_put(
        work_dir / "worker_runtime_manifest.json",
        runtime_manifest,
    )
    try:
        _copy_worker_runtime_files_to_artifact_output(
            work_dir=work_dir,
            artifact_output_uri=artifact_output_uri,
            relative_paths=["worker_runtime_manifest.json"],
        )
    except Exception:
        pass
    return runtime_manifest


def _copy_artifacts(*, job_dir: Path, artifact_output_uri: str) -> Dict[str, Any]:
    parsed = urllib.parse.urlparse(artifact_output_uri)
    if parsed.scheme in {"", "file"}:
        destination = Path(
            urllib.request.url2pathname(parsed.path if parsed.scheme else artifact_output_uri)
        )
        _copy_directory_contents(job_dir, destination)
        return {
            "status": "completed",
            "destination_uri": artifact_output_uri,
            "destination_path": str(destination),
            "uploaded_file_count": len([path for path in destination.rglob("*") if path.is_file()]),
        }
    if parsed.scheme == "gs":
        uploaded = _upload_directory_to_gs(job_dir, artifact_output_uri)
        return {
            "status": "completed",
            "destination_uri": artifact_output_uri,
            "uploaded_file_count": uploaded,
        }
    if parsed.scheme in {"s3", "r2"}:
        uploaded = _upload_directory_to_s3_compatible(job_dir, artifact_output_uri)
        return {
            "status": "completed",
            "destination_uri": artifact_output_uri,
            "uploaded_file_count": uploaded,
            "storage_scheme": parsed.scheme,
            "s3_compatible_endpoint_configured": bool(_s3_compatible_endpoint_url(parsed.scheme)),
        }
    return {
        "status": "blocked",
        "destination_uri": artifact_output_uri,
        "blockers": [f"unsupported_artifact_output_uri_scheme:{parsed.scheme or 'local'}"],
    }


def _parse_simulator_commands(values: Iterable[str]) -> Dict[str, str]:
    commands: Dict[str, str] = {}
    for value in values:
        simulator, separator, command = value.partition("=")
        if (
            not separator
            or simulator not in SIMULATORS
            or simulator == "fixture"
            or not command.strip()
        ):
            raise ValueError(
                "--simulator-command must be formatted as "
                "<mujoco|pybullet|newton|isaac_sim|isaac_lab_arena>=<command>"
            )
        commands[simulator] = command.strip()
    return commands


def _resolve_capture_root_relative_path(
    value: str,
    *,
    capture_root: Path,
) -> Path | None:
    text = _string(value)
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else capture_root / path


def _blocked_runtime_manifest(
    *,
    work_dir: Path,
    manifest_uri: str,
    blockers: Sequence[str],
    generated_at: str,
    context: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    runtime_manifest = {
        "schema_version": WORKER_RUNTIME_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked",
        "manifest_uri": manifest_uri,
        "blockers": list(blockers),
        "live_provider_calls_performed": False,
        "simulator_execution_proven": False,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
    }
    runtime_manifest.update(dict(context or {}))
    return _write_runtime_manifest_with_signed_put(
        work_dir / "worker_runtime_manifest.json",
        runtime_manifest,
    )


def run_robot_eval_worker(
    *,
    manifest_uri: str,
    work_dir: str | Path | None = None,
    capture_root: str | Path | None = None,
    job_id: str | None = None,
    provisioner: str | None = None,
    simulator: str | None = None,
    evaluation_substrate: str | None = None,
    allow_wam_provider: bool = False,
    wam_provider_commands: Mapping[str, str] | None = None,
    wam_artifact_output_uri: str | None = None,
    wam_provider_max_retries: int = 0,
    wam_provider_timeout_seconds: int | None = None,
    allow_gpu_provisioning: bool = False,
    allow_simulator_execution: bool = False,
    allowed_simulators: Sequence[str] = (),
    simulator_commands: Mapping[str, str] | None = None,
    allow_cpu_simulator_preflight: bool = False,
    cpu_preflight_backends: Sequence[str] = CPU_BACKENDS,
    cpu_preflight_smoke_steps: int = 10,
    allow_cpu_preflight_render: bool = False,
    timeout_seconds: int | None = None,
    budget_usd: float | None = None,
    artifact_output_uri: str | None = None,
    artifact_output_uri_required: bool | None = None,
) -> Dict[str, Any]:
    generated_at = utc_now_iso()
    worker_started_monotonic = time.monotonic()
    worker_dir = Path(work_dir or os.getenv("BLUEPRINT_WORKER_DIR") or "/tmp/blueprint-worker")
    ensure_dir(worker_dir)
    try:
        payload = _load_manifest(manifest_uri, worker_dir)
    except Exception as exc:
        return _blocked_runtime_manifest(
            work_dir=worker_dir,
            manifest_uri=manifest_uri,
            blockers=[f"worker_manifest_load_failed:{type(exc).__name__}"],
            generated_at=generated_at,
        )

    job_request = _mapping(payload.get("job_request")) or _mapping(payload)
    input_bundle = _mapping(payload.get("input_bundle"))
    capture_root_bundle_uri = _string(
        payload.get("capture_root_bundle_uri")
        or input_bundle.get("capture_root_bundle_uri")
    )
    capture_root_bundle: Dict[str, Any] = {"status": "not_configured"}
    bundle_capture_root = ""
    if capture_root_bundle_uri and not capture_root:
        try:
            capture_root_bundle = _extract_capture_root_bundle(capture_root_bundle_uri, worker_dir)
            bundle_capture_root = _string(capture_root_bundle.get("capture_root"))
        except Exception as exc:
            return _blocked_runtime_manifest(
                work_dir=worker_dir,
                manifest_uri=manifest_uri,
                blockers=[f"capture_root_bundle_extract_failed:{type(exc).__name__}"],
                generated_at=generated_at,
                context={
                    "capture_root_bundle_uri": capture_root_bundle_uri,
                    "capture_root_bundle": {
                        "status": "blocked",
                        "source_uri": capture_root_bundle_uri,
                    },
                },
            )
    selected_capture_root = (
        _string(capture_root)
        or bundle_capture_root
        or _string(payload.get("capture_root"))
        or _string(job_request.get("capture_root"))
    )
    if not selected_capture_root:
        return _blocked_runtime_manifest(
            work_dir=worker_dir,
            manifest_uri=manifest_uri,
            blockers=["missing_capture_root"],
            generated_at=generated_at,
        )

    selected_job_id = (
        job_id
        or _string(payload.get("job_id"))
        or _string(job_request.get("job_id"))
        or _string(job_request.get("jobId"))
    )
    if not selected_job_id:
        return _blocked_runtime_manifest(
            work_dir=worker_dir,
            manifest_uri=manifest_uri,
            blockers=["missing_job_id"],
            generated_at=generated_at,
        )

    selected_provisioner = provisioner or _string(payload.get("provisioner")) or "fixture_local"
    selected_simulator = simulator or _string(payload.get("simulator")) or "fixture"
    selected_evaluation_substrate = (
        evaluation_substrate
        or _string(payload.get("evaluation_substrate") or payload.get("evaluationSubstrate"))
        or _string(job_request.get("evaluation_substrate") or job_request.get("evaluationSubstrate"))
        or None
    )
    selected_allow_wam_provider = bool(
        allow_wam_provider
        or _bool(payload.get("allow_wam_provider") or payload.get("allowWamProvider"))
    )
    selected_wam_provider_commands = {
        **_mapping(payload.get("wam_provider_commands") or payload.get("wamProviderCommands")),
        **dict(wam_provider_commands or {}),
    }
    selected_wam_artifact_output_uri = (
        wam_artifact_output_uri
        or _string(payload.get("wam_artifact_output_uri") or payload.get("wamArtifactOutputUri"))
        or None
    )
    selected_wam_provider_max_retries = int(
        wam_provider_max_retries
        or _number(payload.get("wam_provider_max_retries") or payload.get("wamProviderMaxRetries"))
        or 0
    )
    selected_wam_provider_timeout_seconds = (
        int(wam_provider_timeout_seconds)
        if wam_provider_timeout_seconds is not None
        else int(
            _number(
                payload.get("wam_provider_timeout_seconds")
                or payload.get("wamProviderTimeoutSeconds")
            )
            or 0
        )
        or None
    )
    selected_robot = (
        _string(payload.get("robot"))
        or _string(payload.get("robot_model"))
        or _string(job_request.get("robot"))
        or _string(job_request.get("robot_model"))
        or None
    )
    live_provider_manifest_required = selected_provisioner != "fixture_local"
    payload_schema = _string(payload.get("schema_version"))
    if live_provider_manifest_required and payload_schema != WORKER_INPUT_MANIFEST_SCHEMA_VERSION:
        return _blocked_runtime_manifest(
            work_dir=worker_dir,
            manifest_uri=manifest_uri,
            blockers=["invalid_or_missing_worker_manifest_schema"],
            generated_at=generated_at,
            context={
                "job_id": selected_job_id,
                "capture_root": selected_capture_root,
                "capture_root_bundle_uri": capture_root_bundle_uri or None,
                "capture_root_bundle": capture_root_bundle,
                "provisioner": selected_provisioner,
                "simulator": selected_simulator,
                "expected_worker_manifest_schema": WORKER_INPUT_MANIFEST_SCHEMA_VERSION,
                "actual_worker_manifest_schema": payload_schema or None,
            },
        )
    if live_provider_manifest_required and not isinstance(payload.get("job_request"), Mapping):
        return _blocked_runtime_manifest(
            work_dir=worker_dir,
            manifest_uri=manifest_uri,
            blockers=["missing_worker_manifest_job_request"],
            generated_at=generated_at,
            context={
                "job_id": selected_job_id,
                "capture_root": selected_capture_root,
                "capture_root_bundle_uri": capture_root_bundle_uri or None,
                "capture_root_bundle": capture_root_bundle,
                "provisioner": selected_provisioner,
                "simulator": selected_simulator,
                "expected_worker_manifest_schema": WORKER_INPUT_MANIFEST_SCHEMA_VERSION,
            },
        )
    runtime_preflight_contract = _mapping(payload.get("runtime_preflight_contract"))
    selected_timeout = int(
        timeout_seconds
        or _number(payload.get("timeout_seconds"))
        or _number(_mapping(job_request.get("budget")).get("timeout_seconds"))
        or 120
    )
    payload_budget = _number(payload.get("budget_usd"))
    request_budget = _number(_mapping(job_request.get("budget")).get("budget_usd"))
    selected_budget = (
        budget_usd
        if budget_usd is not None
        else payload_budget
        if payload_budget is not None
        else request_budget
    )
    selected_allowed_simulators = list(allowed_simulators) or _string_list(
        payload.get("allowed_simulators")
    )
    selected_simulator_commands = {
        **_mapping(payload.get("simulator_commands")),
        **dict(simulator_commands or {}),
    }
    selected_scenario_eval_matrix_path = _resolve_capture_root_relative_path(
        _string(payload.get("scenario_eval_matrix_path")),
        capture_root=Path(selected_capture_root),
    )
    selected_artifact_output_uri = artifact_output_uri or _string(
        payload.get("artifact_output_uri")
    )
    selected_wam_artifact_output_uri = (
        selected_wam_artifact_output_uri or selected_artifact_output_uri or None
    )
    selected_artifact_output_uri_scheme = None
    if selected_artifact_output_uri:
        selected_artifact_output_uri_scheme = (
            urllib.parse.urlparse(selected_artifact_output_uri).scheme or "local"
        )
    payload_artifact_required = _bool(payload.get("artifact_output_uri_required"))
    selected_artifact_output_uri_required = (
        artifact_output_uri_required
        if artifact_output_uri_required is not None
        else payload_artifact_required
        if payload_artifact_required is not None
        else selected_provisioner != "fixture_local"
    )
    payload_artifact_output_uri_provider_writable = _bool(
        payload.get("artifact_output_uri_provider_writable")
    )
    artifact_output_uri_has_explicit_scheme = bool(
        urllib.parse.urlparse(selected_artifact_output_uri).scheme
    )
    inferred_artifact_output_uri_provider_writable = (
        _artifact_output_uri_is_provider_writable(
            selected_artifact_output_uri,
            live_provider=live_provider_manifest_required,
        )
        if selected_artifact_output_uri
        and (
            artifact_output_uri_has_explicit_scheme
            or payload_artifact_output_uri_provider_writable is not None
        )
        else None
    )
    artifact_output_uri_provider_writable_flags = [
        flag
        for flag in (
            payload_artifact_output_uri_provider_writable,
            inferred_artifact_output_uri_provider_writable,
        )
        if flag is not None
    ]
    selected_artifact_output_uri_provider_writable = (
        all(artifact_output_uri_provider_writable_flags)
        if artifact_output_uri_provider_writable_flags
        else None
    )
    artifact_output_write_auth = _mapping(payload.get("artifact_output_write_auth"))
    payload_artifact_output_write_auth_ready = _bool(
        payload.get("artifact_output_write_auth_contract_ready")
    )
    inferred_artifact_output_write_auth = _artifact_output_write_auth_contract(
        selected_artifact_output_uri,
        external_provider=live_provider_manifest_required,
        provider_writable=bool(selected_artifact_output_uri_provider_writable),
    )
    selected_artifact_output_write_auth = (
        artifact_output_write_auth or inferred_artifact_output_write_auth
    )
    artifact_output_write_auth_contract_ready = (
        payload_artifact_output_write_auth_ready
        if payload_artifact_output_write_auth_ready is not None
        else bool(selected_artifact_output_write_auth.get("write_auth_contract_ready"))
    )
    artifact_output_write_auth_contract_present = bool(
        artifact_output_write_auth or payload_artifact_output_write_auth_ready is not None
    )
    selected_artifact_output_write_auth_contract_ready = bool(
        artifact_output_write_auth_contract_ready
        and (
            not live_provider_manifest_required
            or artifact_output_write_auth_contract_present
        )
    )
    if selected_artifact_output_uri_required and not selected_artifact_output_uri:
        return _blocked_runtime_manifest(
            work_dir=worker_dir,
            manifest_uri=manifest_uri,
            blockers=["missing_artifact_output_uri"],
            generated_at=generated_at,
            context={
                "job_id": selected_job_id,
                "capture_root": selected_capture_root,
                "capture_root_bundle_uri": capture_root_bundle_uri or None,
                "capture_root_bundle": capture_root_bundle,
                "provisioner": selected_provisioner,
                "simulator": selected_simulator,
                "artifact_output_uri_required": True,
                "artifact_upload": {
                    "status": "blocked",
                    "reason": "missing_artifact_output_uri",
                },
            },
        )
    if (
        live_provider_manifest_required
        and selected_artifact_output_uri
        and selected_artifact_output_uri_provider_writable is False
    ):
        return _blocked_runtime_manifest(
            work_dir=worker_dir,
            manifest_uri=manifest_uri,
            blockers=["artifact_output_uri_not_provider_writable"],
            generated_at=generated_at,
            context={
                "job_id": selected_job_id,
                "capture_root": selected_capture_root,
                "capture_root_bundle_uri": capture_root_bundle_uri or None,
                "capture_root_bundle": capture_root_bundle,
                "provisioner": selected_provisioner,
                "simulator": selected_simulator,
                "artifact_output_uri_required": selected_artifact_output_uri_required,
                "artifact_output_uri": selected_artifact_output_uri,
                "artifact_output_uri_scheme": selected_artifact_output_uri_scheme,
                "artifact_output_uri_provider_writable": False,
                "artifact_upload": {
                    "status": "blocked",
                    "reason": "artifact_output_uri_not_provider_writable",
                },
            },
        )
    if (
        live_provider_manifest_required
        and selected_artifact_output_uri
        and selected_artifact_output_uri_provider_writable is True
        and (
            not artifact_output_write_auth_contract_present
            or not selected_artifact_output_write_auth_contract_ready
        )
    ):
        return _blocked_runtime_manifest(
            work_dir=worker_dir,
            manifest_uri=manifest_uri,
            blockers=["artifact_output_write_auth_contract_missing"],
            generated_at=generated_at,
            context={
                "job_id": selected_job_id,
                "capture_root": selected_capture_root,
                "capture_root_bundle_uri": capture_root_bundle_uri or None,
                "capture_root_bundle": capture_root_bundle,
                "provisioner": selected_provisioner,
                "simulator": selected_simulator,
                "artifact_output_uri_required": selected_artifact_output_uri_required,
                "artifact_output_uri": selected_artifact_output_uri,
                "artifact_output_uri_scheme": selected_artifact_output_uri_scheme,
                "artifact_output_uri_provider_writable": (
                    selected_artifact_output_uri_provider_writable
                ),
                "artifact_output_write_auth": selected_artifact_output_write_auth,
                "artifact_output_write_auth_contract_ready": (
                    selected_artifact_output_write_auth_contract_ready
                ),
                "artifact_upload": {
                    "status": "blocked",
                    "reason": "artifact_output_write_auth_contract_missing",
                },
            },
        )
    if live_provider_manifest_required and selected_simulator != "fixture":
        runtime_preflight_blockers = _runtime_preflight_contract_blockers(
            simulator=selected_simulator,
            contract=runtime_preflight_contract,
        )
        if runtime_preflight_blockers:
            return _blocked_runtime_manifest(
                work_dir=worker_dir,
                manifest_uri=manifest_uri,
                blockers=["invalid_worker_runtime_preflight_contract"],
                generated_at=generated_at,
                context={
                    "job_id": selected_job_id,
                    "capture_root": selected_capture_root,
                    "capture_root_bundle_uri": capture_root_bundle_uri or None,
                    "capture_root_bundle": capture_root_bundle,
                    "provisioner": selected_provisioner,
                    "simulator": selected_simulator,
                    "runtime_preflight_contract": runtime_preflight_contract,
                    "runtime_preflight_contract_blockers": runtime_preflight_blockers,
                    "simulator_execution_proven": False,
                    "robot_readiness_proven": False,
                },
            )
    runtime_preflight = _write_worker_runtime_preflight(
        work_dir=worker_dir,
        manifest_uri=manifest_uri,
        job_id=selected_job_id,
        capture_root=selected_capture_root,
        provisioner=selected_provisioner,
        simulator=selected_simulator,
        contract=runtime_preflight_contract,
        allow_simulator_execution=allow_simulator_execution,
        timeout_seconds=selected_timeout,
        generated_at=generated_at,
        payload=payload,
    )
    runtime_preflight_blockers = _string_list(runtime_preflight.get("blockers"))
    if selected_simulator != "fixture" and allow_simulator_execution and runtime_preflight_blockers:
        runtime_manifest = _blocked_runtime_manifest(
            work_dir=worker_dir,
            manifest_uri=manifest_uri,
            blockers=["worker_runtime_preflight_blocked"],
            generated_at=generated_at,
            context={
                "job_id": selected_job_id,
                "capture_root": selected_capture_root,
                "capture_root_bundle_uri": capture_root_bundle_uri or None,
                "capture_root_bundle": capture_root_bundle,
                "provisioner": selected_provisioner,
                "simulator": selected_simulator,
                "runtime_preflight_manifest_path": str(
                    worker_dir / "worker_runtime_preflight.json"
                ),
                "runtime_preflight_status": runtime_preflight.get("status"),
                "runtime_preflight_blockers": runtime_preflight_blockers,
                "simulator_execution_proven": False,
                "robot_readiness_proven": False,
            },
        )
        return _attach_worker_failure_artifact_upload(
            runtime_manifest=runtime_manifest,
            work_dir=worker_dir,
            artifact_output_uri=selected_artifact_output_uri,
        )

    job_request = _rebase_job_request_capture_root(
        job_request,
        selected_capture_root=selected_capture_root,
    )
    request_path = worker_dir / "job_request.json"
    write_json(request_path, job_request)
    try:
        result = build_robot_eval_job(
            capture_root=selected_capture_root,
            job_request=request_path,
            job_id=selected_job_id,
            provisioner=selected_provisioner,
            simulator=selected_simulator,
            evaluation_substrate=selected_evaluation_substrate,
            allow_wam_provider=selected_allow_wam_provider,
            wam_provider_commands=selected_wam_provider_commands,
            wam_artifact_output_uri=selected_wam_artifact_output_uri,
            wam_provider_max_retries=selected_wam_provider_max_retries,
            wam_provider_timeout_seconds=selected_wam_provider_timeout_seconds,
            allow_gpu_provisioning=allow_gpu_provisioning,
            allow_simulator_execution=allow_simulator_execution,
            allowed_simulators=selected_allowed_simulators,
            simulator_commands=selected_simulator_commands,
            allow_cpu_simulator_preflight=allow_cpu_simulator_preflight,
            cpu_preflight_backends=cpu_preflight_backends,
            cpu_preflight_smoke_steps=cpu_preflight_smoke_steps,
            allow_cpu_preflight_render=allow_cpu_preflight_render,
            timeout_seconds=selected_timeout,
            budget_usd=selected_budget,
        )
    except Exception as exc:
        return _blocked_runtime_manifest(
            work_dir=worker_dir,
            manifest_uri=manifest_uri,
            blockers=[f"worker_orchestrator_failed:{type(exc).__name__}"],
            generated_at=generated_at,
        )

    job_dir = Path(_string(result.get("job_dir")))
    if job_dir:
        _copy_worker_runtime_files_to_job_dir(worker_dir=worker_dir, job_dir=job_dir)
    provider_runtime_accounting = (
        _record_provider_runtime_gpu_time(
            job_dir=job_dir,
            started_monotonic=worker_started_monotonic,
            generated_at=generated_at,
        )
        if job_dir
        else {"status": "not_available_missing_job_dir"}
    )
    job_result_manifest = _optional_json_mapping(
        Path(_string(result.get("manifest_path")) or job_dir / "job_run_manifest.json")
    )
    job_result = {**dict(result), **job_result_manifest}

    artifact_upload = {"status": "not_requested"}
    worker_blockers: List[str] = []
    if selected_artifact_output_uri:
        try:
            artifact_upload = _copy_artifacts(
                job_dir=job_dir,
                artifact_output_uri=selected_artifact_output_uri,
            )
        except Exception as exc:
            artifact_upload = {
                "status": "blocked",
                "destination_uri": selected_artifact_output_uri,
                "blockers": [f"artifact_upload_failed:{type(exc).__name__}"],
            }
        worker_blockers.extend(_string_list(artifact_upload.get("blockers")))

    job_status = _string(job_result.get("status"))
    job_blockers = _string_list(job_result.get("blockers"))
    job_missing_inputs = _string_list(job_result.get("missing_inputs"))
    simulator_execution_proven = bool(job_result.get("simulator_execution_proven"))
    robot_readiness_proven = bool(job_result.get("robot_readiness_proven"))
    public_claim_upgrade_allowed = bool(job_result.get("public_claim_upgrade_allowed"))
    scenario_eval_matrix_status = job_result.get("scenario_eval_matrix_status")
    simulator_service_status = job_result.get("simulator_service_status")
    evaluation_status = job_result.get("evaluation_status")
    full_job_status = job_status
    full_job_blockers = list(job_blockers)
    full_job_missing_inputs = list(job_missing_inputs)
    provider_runtime_simulator_command_result: Dict[str, Any] = {
        "status": "not_attempted",
        "reason": "full_job_result_used",
    }
    if (
        not simulator_execution_proven
        and job_status == "blocked"
        and selected_simulator != "fixture"
        and _string(selected_simulator_commands.get(selected_simulator))
    ):
        matrix_path = selected_scenario_eval_matrix_path or job_dir / "scenario_eval_matrix.json"
        fallback_blockers: List[str] = []
        if not _env_truthy("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"):
            fallback_blockers.append("missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION")
        if not allow_simulator_execution:
            fallback_blockers.append("missing_cli_allow_simulator_execution")
        if selected_simulator not in set(selected_allowed_simulators):
            fallback_blockers.append(f"missing_cli_allow_simulator_{selected_simulator}")
        if not matrix_path.is_file():
            fallback_blockers.append("missing_provider_runtime_scenario_eval_matrix")
        if fallback_blockers:
            provider_runtime_simulator_command_result = {
                "status": "blocked",
                "reason": "provider_runtime_simulator_command_blocked",
                "blockers": fallback_blockers,
                "scenario_eval_matrix_path": str(matrix_path),
            }
        else:
            simulator_output_path = (
                job_dir / f"{selected_simulator}_provider_runtime_simulator_output.json"
            )
            provider_runtime_simulator_command_result = _run_command_simulator(
                simulator=selected_simulator,
                command_text=_string(selected_simulator_commands.get(selected_simulator)),
                timeout_seconds=selected_timeout,
                generated_at=generated_at,
                output_path=simulator_output_path,
                capture_root=Path(selected_capture_root),
                scenario_eval_matrix_path=matrix_path,
            )
            provider_runtime_simulator_command_result[
                "scenario_eval_matrix_path"
            ] = str(matrix_path)
            write_json(
                job_dir / "provider_runtime_simulator_command_result.json",
                provider_runtime_simulator_command_result,
            )
            matrix_payload = _optional_json_mapping(matrix_path)
            if (
                provider_runtime_simulator_command_result.get("status") == "completed"
                and provider_runtime_simulator_command_result.get("simulator_execution_proven")
                is True
            ):
                simulator_execution_proven = True
                job_status = "simulator_command_completed"
                job_blockers = []
                job_missing_inputs = []
                scenario_eval_matrix_status = matrix_payload.get("status") or "completed"
                simulator_service_status = provider_runtime_simulator_command_result.get(
                    "status"
                )
                evaluation_status = "completed"
                public_claim_upgrade_allowed = False
                robot_readiness_proven = False
    if worker_blockers:
        status = "blocked"
    elif job_status == "blocked":
        status = "blocked"
    else:
        status = "completed"

    runtime_manifest = {
        "schema_version": WORKER_RUNTIME_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "manifest_uri": manifest_uri,
        "work_dir": str(worker_dir),
        "job_id": selected_job_id,
        "capture_root": selected_capture_root,
        "capture_root_bundle_uri": capture_root_bundle_uri or None,
        "capture_root_bundle": capture_root_bundle,
        "provisioner": selected_provisioner,
        "simulator": selected_simulator,
        "evaluation_substrate": selected_evaluation_substrate,
        "robot": selected_robot,
        "job_status": job_status,
        "full_job_status": full_job_status,
        "job_dir": job_result.get("job_dir"),
        "job_run_manifest_uri": job_result.get("manifest_path"),
        "job_blockers": job_blockers,
        "full_job_blockers": full_job_blockers,
        "job_missing_inputs": job_missing_inputs,
        "full_job_missing_inputs": full_job_missing_inputs,
        "scenario_eval_matrix_status": scenario_eval_matrix_status,
        "simulator_service_status": simulator_service_status,
        "evaluation_status": evaluation_status,
        "provider_runtime_simulator_command_result": provider_runtime_simulator_command_result,
        "artifact_upload": artifact_upload,
        "provider_runtime_accounting": provider_runtime_accounting,
        "artifact_output_uri_required": selected_artifact_output_uri_required,
        "artifact_output_uri": selected_artifact_output_uri or None,
        "artifact_output_uri_scheme": selected_artifact_output_uri_scheme,
        "artifact_output_uri_provider_writable": selected_artifact_output_uri_provider_writable,
        "artifact_output_write_auth": selected_artifact_output_write_auth,
        "artifact_output_write_auth_contract_ready": (
            selected_artifact_output_write_auth_contract_ready
        ),
        "runtime_preflight_contract": runtime_preflight_contract,
        "runtime_preflight_manifest_path": "worker_runtime_preflight.json",
        "runtime_preflight_status": runtime_preflight.get("status"),
        "runtime_preflight_blockers": runtime_preflight_blockers,
        "runtime_preflight_artifact": _string(runtime_preflight_contract.get("result_artifact"))
        or "worker_runtime_preflight.json",
        "runtime_preflight_required_before_scene_load": (
            runtime_preflight_contract.get("required_before_scene_load") is True
        ),
        "blockers": worker_blockers,
        "live_provider_calls_performed": False,
        "simulator_execution_proven": simulator_execution_proven,
        "robot_readiness_proven": robot_readiness_proven,
        "public_claim_upgrade_allowed": public_claim_upgrade_allowed,
    }
    runtime_manifest_path = worker_dir / "worker_runtime_manifest.json"
    runtime_manifest = _write_redacted_json(runtime_manifest_path, runtime_manifest)
    job_runtime_manifest_path = job_dir / "worker_runtime_manifest.json"
    runtime_manifest = _write_redacted_json(job_runtime_manifest_path, runtime_manifest)
    startup_audit = _refresh_job_startup_audit_with_worker_runtime(job_dir)
    runtime_manifest["startup_architecture_audit_status"] = startup_audit.get("status")
    runtime_manifest["startup_architecture_compliant"] = bool(
        startup_audit.get("architecture_compliant")
    )
    runtime_manifest["startup_architecture_blockers"] = _string_list(
        startup_audit.get("blockers")
    )
    runtime_manifest = _write_redacted_json(runtime_manifest_path, runtime_manifest)
    runtime_manifest = _write_redacted_json(job_runtime_manifest_path, runtime_manifest)
    if selected_artifact_output_uri and artifact_upload.get("status") == "completed":
        finalizer_refresh_upload: Dict[str, Any] = {"status": "not_attempted"}
        try:
            finalizer_refresh_upload = _copy_worker_runtime_files_to_artifact_output(
                work_dir=job_dir,
                artifact_output_uri=selected_artifact_output_uri,
                relative_paths=[
                    "job_run_manifest.json",
                    "startup_architecture_audit.json",
                    "worker_runtime_manifest.json",
                    "worker_runtime_preflight_detail.json",
                ],
            )
            runtime_manifest_upload = _copy_runtime_manifest_to_artifact_output(
                runtime_manifest_path=job_runtime_manifest_path,
                artifact_output_uri=selected_artifact_output_uri,
            )
        except Exception as exc:
            runtime_manifest_upload = {
                "status": "blocked",
                "destination_uri": selected_artifact_output_uri,
                "relative_path": "worker_runtime_manifest.json",
                "blockers": [f"worker_runtime_manifest_upload_failed:{type(exc).__name__}"],
            }
        artifact_upload["worker_runtime_manifest_upload"] = runtime_manifest_upload
        artifact_upload["finalizer_refresh_upload"] = finalizer_refresh_upload
        artifact_upload["worker_runtime_manifest_included"] = (
            runtime_manifest_upload.get("status") == "completed"
        )
        provider_runtime_finalizer = _record_provider_runtime_finalizer_proof(
            job_dir=job_dir,
            artifact_output_uri=selected_artifact_output_uri,
            artifact_upload=artifact_upload,
            finalizer_refresh_upload=finalizer_refresh_upload,
            runtime_manifest_upload=runtime_manifest_upload,
            generated_at=generated_at,
        )
        remote_cloud_closure = _refresh_remote_cloud_closure_after_worker_runtime(
            job_dir=job_dir,
            generated_at=generated_at,
        )
        artifact_upload["provider_runtime_finalizer"] = provider_runtime_finalizer
        artifact_upload["remote_cloud_execution_closure"] = remote_cloud_closure
        if provider_runtime_finalizer.get("status") != "not_provider_runtime":
            _copy_worker_runtime_files_to_artifact_output(
                work_dir=job_dir,
                artifact_output_uri=selected_artifact_output_uri,
                relative_paths=[
                    "gpu_cost_control_ledger.json",
                    "provider_runtime_finalizer_proof.json",
                    "remote_cloud_execution_closure_manifest.json",
                    "job_run_manifest.json",
                ],
            )
        runtime_manifest["artifact_upload"] = artifact_upload
        if runtime_manifest_upload.get("status") != "completed":
            runtime_manifest["status"] = "blocked"
            runtime_manifest["blockers"] = _dedupe(
                [
                    *_string_list(runtime_manifest.get("blockers")),
                    *_string_list(runtime_manifest_upload.get("blockers")),
                ]
            )
        runtime_manifest = _write_redacted_json(runtime_manifest_path, runtime_manifest)
        runtime_manifest = _write_redacted_json(job_runtime_manifest_path, runtime_manifest)
        if runtime_manifest_upload.get("status") == "completed":
            _copy_runtime_manifest_to_artifact_output(
                runtime_manifest_path=job_runtime_manifest_path,
                artifact_output_uri=selected_artifact_output_uri,
            )
    runtime_manifest = _write_runtime_manifest_with_signed_put(
        job_runtime_manifest_path,
        runtime_manifest,
    )
    runtime_manifest = _write_redacted_json(runtime_manifest_path, runtime_manifest)
    return runtime_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a queued Blueprint robot-eval job worker")
    parser.add_argument(
        "--manifest",
        default=os.getenv("BLUEPRINT_EVAL_MANIFEST_URI"),
        help="Worker manifest URI or local path. Defaults to BLUEPRINT_EVAL_MANIFEST_URI.",
    )
    parser.add_argument("--work-dir", default=os.getenv("BLUEPRINT_WORKER_DIR"))
    parser.add_argument("--capture-root")
    parser.add_argument("--job-id")
    parser.add_argument("--provisioner", choices=PROVISIONERS)
    parser.add_argument("--simulator", choices=SIMULATORS)
    parser.add_argument("--evaluation-substrate")
    parser.add_argument("--allow-wam-provider", action="store_true")
    parser.add_argument("--wam-provider-command", action="append", default=[])
    parser.add_argument("--wam-artifact-output-uri")
    parser.add_argument("--wam-provider-max-retries", type=int, default=0)
    parser.add_argument("--wam-provider-timeout-seconds", type=int)
    parser.add_argument("--allow-gpu-provisioning", action="store_true")
    parser.add_argument("--allow-simulator-execution", action="store_true")
    parser.add_argument("--allowed-simulator", action="append", default=[])
    parser.add_argument("--simulator-command", action="append", default=[])
    parser.add_argument("--allow-cpu-simulator-preflight", action="store_true")
    parser.add_argument("--cpu-preflight-backend", action="append", default=[])
    parser.add_argument("--cpu-preflight-smoke-steps", type=int, default=10)
    parser.add_argument("--allow-cpu-preflight-render", action="store_true")
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--artifact-output-uri", default=os.getenv("BLUEPRINT_ARTIFACT_OUTPUT_URI"))
    parser.add_argument("--require-artifact-output-uri", action="store_true")
    parser.add_argument("--allow-missing-artifact-output-uri", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.manifest:
        parser.error("--manifest or BLUEPRINT_EVAL_MANIFEST_URI is required")
    result = run_robot_eval_worker(
        manifest_uri=args.manifest,
        work_dir=args.work_dir,
        capture_root=args.capture_root,
        job_id=args.job_id,
        provisioner=args.provisioner,
        simulator=args.simulator,
        evaluation_substrate=args.evaluation_substrate,
        allow_wam_provider=args.allow_wam_provider,
        wam_provider_commands=parse_wam_provider_commands(args.wam_provider_command),
        wam_artifact_output_uri=args.wam_artifact_output_uri,
        wam_provider_max_retries=args.wam_provider_max_retries,
        wam_provider_timeout_seconds=args.wam_provider_timeout_seconds,
        allow_gpu_provisioning=args.allow_gpu_provisioning,
        allow_simulator_execution=args.allow_simulator_execution,
        allowed_simulators=args.allowed_simulator,
        simulator_commands=_parse_simulator_commands(args.simulator_command),
        allow_cpu_simulator_preflight=args.allow_cpu_simulator_preflight,
        cpu_preflight_backends=args.cpu_preflight_backend or CPU_BACKENDS,
        cpu_preflight_smoke_steps=args.cpu_preflight_smoke_steps,
        allow_cpu_preflight_render=args.allow_cpu_preflight_render,
        timeout_seconds=args.timeout_seconds,
        budget_usd=args.budget_usd,
        artifact_output_uri=args.artifact_output_uri,
        artifact_output_uri_required=(
            False
            if args.allow_missing_artifact_output_uri
            else True
            if args.require_artifact_output_uri
            else None
        ),
    )
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
