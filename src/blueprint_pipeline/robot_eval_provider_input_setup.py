"""Prepare provider-fetchable inputs for a robot-eval worker launch.

This command packages the current capture inputs for a remote worker, computes
the worker manifest/artifact URIs, optionally uploads provider inputs to object
storage, and reruns the job orchestrator with the resulting environment. It does
not call RunPod or any other GPU provider.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import os
import shlex
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from .common import ensure_dir, parse_gs_uri, read_json_any, utc_now_iso, write_json
from .robot_eval_job_orchestrator import (
    WORKER_ARTIFACT_OUTPUT_URI_ENV,
    WORKER_CAPTURE_ROOT_BUNDLE_URI_ENV,
    WORKER_IMAGE_REF_ENV_BY_SIMULATOR,
    WORKER_MANIFEST_URI_ENV,
    _remote_cloud_execution_closure_manifest,
)
from .robot_eval_evaluation_run_adapter import (
    execute_robot_eval_request_as_evaluation_run,
)


PROVIDER_INPUT_SETUP_SCHEMA_VERSION = "robot_eval_provider_input_setup.v1"
DEFAULT_INCLUDE_PATHS = (
    "capture_descriptor.json",
    "raw/manifest.json",
    "raw/capture_context.json",
    "raw/capture_upload_complete.json",
    "pipeline/robot_eval_dataset",
    "pipeline/simulation_automation",
    "pipeline/sim_only_beta_rehearsal",
    "pipeline/worldlabs_assets",
    "pipeline/provider_run_manifest.json",
    "pipeline/worldlabs_input_audit.json",
    "pipeline/worldlabs_operation_manifest.json",
)
RAW_MEDIA_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
REMOTE_PROVIDER_INPUT_URI_SCHEMES = {"gs", "s3", "r2", "http", "https"}
REMOTE_PROVIDER_ARTIFACT_OUTPUT_URI_SCHEMES = {"gs", "s3", "r2"}
R2_ENDPOINT_ENV_VAR_ALTERNATIVES = (
    "BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL",
    "R2_ENDPOINT_URL",
    "AWS_ENDPOINT_URL",
)
S3_SECRET_ENV_VARS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
S3_REGION_ENV_VAR_ALTERNATIVES = ("AWS_REGION", "AWS_DEFAULT_REGION")


def _file_env_name(env_name: str) -> str:
    return f"{env_name}_FILE"


def _read_file_env_value(env_name: str) -> str:
    path_text = os.getenv(_file_env_name(env_name))
    if not path_text:
        return ""
    path = Path(path_text).expanduser()
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _env_or_file_value(env_name: str) -> str:
    return os.getenv(env_name) or _read_file_env_value(env_name)


def _env_or_file_present(env_name: str) -> bool:
    return bool(_env_or_file_value(env_name))


def _first_env_or_file_value(env_names: Sequence[str]) -> str:
    for env_name in env_names:
        value = _env_or_file_value(env_name)
        if value:
            return value
    return ""


def _present_file_env_vars(env_names: Sequence[str]) -> list[str]:
    return [
        _file_env_name(env_name)
        for env_name in env_names
        if _read_file_env_value(env_name)
    ]


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = _string(value)
        return [text] if text else []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return [_string(item) for item in value if _string(item)]
    return []


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uri_join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *(part.strip("/") for part in parts if part)])


def _git_short_sha(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(cwd),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return "unknownsha"
    return result.stdout.strip() or "unknownsha"


def default_image_ref(*, simulator: str, repo_root: Path) -> str:
    date = utc_now_iso()[:10].replace("-", "")
    sha = _git_short_sha(repo_root)
    family = "mujoco-eval-worker" if simulator == "mujoco" else f"{simulator}-eval-worker"
    return f"ghcr.io/ognjhunt/blueprint-{family}:{date}-{sha}"


def _iter_bundle_files(
    capture_root: Path,
    *,
    include_raw_media: bool,
    include_paths: Iterable[str] = DEFAULT_INCLUDE_PATHS,
) -> Iterable[Path]:
    seen: set[Path] = set()
    for relative in include_paths:
        candidate = capture_root / relative
        if candidate.is_file():
            sources = [candidate]
        elif candidate.is_dir():
            sources = [path for path in sorted(candidate.rglob("*")) if path.is_file()]
        else:
            continue
        for source in sources:
            if not include_raw_media and source.suffix.lower() in RAW_MEDIA_SUFFIXES:
                continue
            if source not in seen:
                seen.add(source)
                yield source


def _bundle_arcname(capture_root: Path, source: Path) -> Path:
    parts = capture_root.parts
    try:
        scenes_index = parts.index("scenes")
    except ValueError:
        return Path("capture-root") / source.relative_to(capture_root)
    if scenes_index <= 0:
        return Path("capture-root") / source.relative_to(capture_root)
    archive_root = Path(*parts[: scenes_index - 1])
    return source.relative_to(archive_root)


def build_capture_root_bundle(
    *,
    capture_root: str | Path,
    output_path: str | Path,
    include_raw_media: bool = False,
    include_paths: Iterable[str] = DEFAULT_INCLUDE_PATHS,
) -> Dict[str, Any]:
    root = Path(capture_root).resolve()
    path = Path(output_path).resolve()
    ensure_dir(path.parent)
    files = list(
        _iter_bundle_files(
            root,
            include_raw_media=include_raw_media,
            include_paths=include_paths,
        )
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in files:
            archive.write(source, _bundle_arcname(root, source))
    return {
        "status": "created",
        "path": str(path),
        "capture_root": str(root),
        "file_count": len(files),
        "include_raw_media": include_raw_media,
        "raw_media_excluded_by_default": not include_raw_media,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "format": "zip",
    }


def _upload_file_to_gs(source: Path, destination_uri: str) -> Dict[str, Any]:
    parsed = parse_gs_uri(destination_uri)
    try:
        from google.cloud import storage as gcs_storage  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("google-cloud-storage is required for gs:// uploads") from exc
    client = gcs_storage.Client()
    client.bucket(parsed.bucket).blob(parsed.key).upload_from_filename(str(source))
    return {
        "status": "uploaded",
        "source": str(source),
        "destination_uri": destination_uri,
        "storage_scheme": "gs",
    }


def _upload_file_to_s3_compatible(source: Path, destination_uri: str) -> Dict[str, Any]:
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("boto3 is required for s3:// or r2:// uploads") from exc
    parsed = urlparse(destination_uri)
    endpoint_url = _first_env_or_file_value(R2_ENDPOINT_ENV_VAR_ALTERNATIVES) or None
    kwargs: Dict[str, Any] = {}
    if parsed.scheme == "r2" and endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    region = _first_env_or_file_value(S3_REGION_ENV_VAR_ALTERNATIVES)
    if region:
        kwargs["region_name"] = region
    access_key = _env_or_file_value("AWS_ACCESS_KEY_ID")
    secret_key = _env_or_file_value("AWS_SECRET_ACCESS_KEY")
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    client = boto3.client("s3", **kwargs)
    client.upload_file(str(source), parsed.netloc, parsed.path.lstrip("/"))
    return {
        "status": "uploaded",
        "source": str(source),
        "destination_uri": destination_uri,
        "storage_scheme": parsed.scheme,
        "endpoint_configured": bool(endpoint_url),
        "file_based_secret_env_vars_present": _present_file_env_vars(S3_SECRET_ENV_VARS),
        "file_based_endpoint_env_vars_present": _present_file_env_vars(
            R2_ENDPOINT_ENV_VAR_ALTERNATIVES
        ),
        "secret_values_recorded": False,
    }


def _validate_uploaded_object(source: Path, destination_uri: str) -> Dict[str, Any]:
    scheme = urlparse(destination_uri).scheme
    expected_size_bytes = source.stat().st_size if source.is_file() else None
    try:
        if scheme == "gs":
            parsed = parse_gs_uri(destination_uri)
            from google.cloud import storage as gcs_storage  # type: ignore[import-untyped]

            client = gcs_storage.Client()
            blob = client.bucket(parsed.bucket).blob(parsed.key)
            blob.reload()
            object_size_bytes = int(blob.size or 0)
        elif scheme in {"s3", "r2"}:
            try:
                import boto3  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("boto3 is required for s3:// or r2:// validation") from exc
            parsed = urlparse(destination_uri)
            endpoint_url = _first_env_or_file_value(R2_ENDPOINT_ENV_VAR_ALTERNATIVES) or None
            kwargs: Dict[str, Any] = {}
            if parsed.scheme == "r2" and endpoint_url:
                kwargs["endpoint_url"] = endpoint_url
            region = _first_env_or_file_value(S3_REGION_ENV_VAR_ALTERNATIVES)
            if region:
                kwargs["region_name"] = region
            access_key = _env_or_file_value("AWS_ACCESS_KEY_ID")
            secret_key = _env_or_file_value("AWS_SECRET_ACCESS_KEY")
            if access_key and secret_key:
                kwargs["aws_access_key_id"] = access_key
                kwargs["aws_secret_access_key"] = secret_key
            response = boto3.client("s3", **kwargs).head_object(
                Bucket=parsed.netloc,
                Key=parsed.path.lstrip("/"),
            )
            object_size_bytes = int(response.get("ContentLength") or 0)
        elif scheme in {"", "file"}:
            destination = Path(urlparse(destination_uri).path if scheme else destination_uri).resolve()
            if not destination.is_file():
                return {
                    "status": "blocked",
                    "destination_uri": destination_uri,
                    "storage_scheme": scheme or "file",
                    "blockers": ["uploaded_object_missing_after_copy"],
                    "secret_values_recorded": False,
                }
            object_size_bytes = destination.stat().st_size
        else:
            return {
                "status": "blocked",
                "destination_uri": destination_uri,
                "storage_scheme": scheme or "local",
                "blockers": [f"unsupported_validation_uri_scheme:{scheme}"],
                "secret_values_recorded": False,
            }
    except Exception as exc:
        return {
            "status": "blocked",
            "destination_uri": destination_uri,
            "storage_scheme": scheme or "file",
            "blockers": [f"uploaded_object_validation_failed:{type(exc).__name__}"],
            "error": str(exc),
            "secret_values_recorded": False,
        }
    size_matches = (
        expected_size_bytes is None or int(object_size_bytes) == int(expected_size_bytes)
    )
    return {
        "status": "validated" if size_matches else "blocked",
        "destination_uri": destination_uri,
        "storage_scheme": scheme or "file",
        "object_size_bytes": object_size_bytes,
        "expected_size_bytes": expected_size_bytes,
        "size_matches_source": size_matches,
        "provider_fetchable_object_probe": size_matches,
        "blockers": [] if size_matches else ["uploaded_object_size_mismatch"],
        "secret_values_recorded": False,
    }


def _classify_upload_error(*, scheme: str, error: BaseException) -> str:
    text = str(error).lower()
    if scheme == "gs" and "billing account" in text and (
        "disabled" in text or "absent" in text or "accountdisabled" in text
    ):
        return "upload_failed:gs_billing_account_disabled"
    if "accessdenied" in text or "access denied" in text:
        return f"upload_failed:{scheme}_access_denied"
    if "forbidden" in text or "403" in text:
        return f"upload_failed:{scheme}_forbidden"
    return f"upload_failed:{type(error).__name__}"


def upload_file(source: str | Path, destination_uri: str) -> Dict[str, Any]:
    path = Path(source).resolve()
    scheme = urlparse(destination_uri).scheme
    try:
        if scheme == "gs":
            result = _upload_file_to_gs(path, destination_uri)
            validation = _validate_uploaded_object(path, destination_uri)
            if validation.get("status") != "validated":
                return {
                    **result,
                    "status": "blocked",
                    "blockers": validation.get("blockers") or ["uploaded_object_validation_failed"],
                    "post_upload_validation": validation,
                }
            return {**result, "post_upload_validation": validation}
        if scheme in {"s3", "r2"}:
            result = _upload_file_to_s3_compatible(path, destination_uri)
            validation = _validate_uploaded_object(path, destination_uri)
            if validation.get("status") != "validated":
                return {
                    **result,
                    "status": "blocked",
                    "blockers": validation.get("blockers") or ["uploaded_object_validation_failed"],
                    "post_upload_validation": validation,
                }
            return {**result, "post_upload_validation": validation}
        if scheme in {"", "file"}:
            destination = Path(urlparse(destination_uri).path if scheme else destination_uri).resolve()
            ensure_dir(destination.parent)
            destination.write_bytes(path.read_bytes())
            validation = _validate_uploaded_object(path, destination_uri)
            if validation.get("status") != "validated":
                return {
                    "status": "blocked",
                    "source": str(path),
                    "destination_uri": str(destination),
                    "storage_scheme": "file",
                    "blockers": validation.get("blockers") or ["uploaded_object_validation_failed"],
                    "post_upload_validation": validation,
                }
            return {
                "status": "uploaded",
                "source": str(path),
                "destination_uri": str(destination),
                "storage_scheme": "file",
                "post_upload_validation": validation,
            }
    except Exception as exc:
        blocker = _classify_upload_error(scheme=scheme or "file", error=exc)
        return {
            "status": "blocked",
            "source": str(path),
            "destination_uri": destination_uri,
            "storage_scheme": scheme or "file",
            "blockers": [blocker],
            "error": str(exc),
        }
    return {
        "status": "blocked",
        "source": str(path),
        "destination_uri": destination_uri,
        "blockers": [f"unsupported_upload_uri_scheme:{scheme}"],
    }


@contextlib.contextmanager
def _patched_env(values: Mapping[str, str]) -> Iterable[None]:
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_env_file(path: Path, values: Mapping[str, str]) -> Dict[str, Any]:
    lines = [
        "# Source this file to run the explicit provider launcher. It contains no raw secrets.",
    ]
    for key, value in values.items():
        lines.append(f"export {key}={value!r}")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "written", "path": str(path)}


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _storage_upload_commands(*, source: str, destination_uri: str) -> list[str]:
    parsed = urlparse(destination_uri)
    scheme = parsed.scheme
    quoted_source = f'"{source}"'
    quoted_destination = f'"{destination_uri}"'
    if scheme == "gs":
        return [
            'test -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" || { echo "missing GOOGLE_APPLICATION_CREDENTIALS" >&2; exit 2; }',
            f"gcloud storage cp {quoted_source} {quoted_destination}",
            f"gcloud storage ls {quoted_destination} >/dev/null",
        ]
    if scheme == "s3":
        return [
            'test -n "${AWS_ACCESS_KEY_ID:-}" || { echo "missing AWS_ACCESS_KEY_ID" >&2; exit 2; }',
            'test -n "${AWS_SECRET_ACCESS_KEY:-}" || { echo "missing AWS_SECRET_ACCESS_KEY" >&2; exit 2; }',
            f"aws s3 cp {quoted_source} {quoted_destination}",
            f"aws s3 ls {quoted_destination} >/dev/null",
        ]
    if scheme == "r2":
        s3_destination = f"s3://{parsed.netloc}{parsed.path}"
        return [
            "# Configure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and an R2 endpoint env in the shell.",
            'test -n "${AWS_ACCESS_KEY_ID:-}" || { echo "missing AWS_ACCESS_KEY_ID" >&2; exit 2; }',
            'test -n "${AWS_SECRET_ACCESS_KEY:-}" || { echo "missing AWS_SECRET_ACCESS_KEY" >&2; exit 2; }',
            'BLUEPRINT_R2_ENDPOINT_URL="${BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL:-${R2_ENDPOINT_URL:-${AWS_ENDPOINT_URL:-}}}"',
            'test -n "$BLUEPRINT_R2_ENDPOINT_URL" || { echo "missing BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL or R2_ENDPOINT_URL or AWS_ENDPOINT_URL" >&2; exit 2; }',
            f'aws s3 cp {quoted_source} "{s3_destination}" --endpoint-url "$BLUEPRINT_R2_ENDPOINT_URL"',
            f'aws s3 ls "{s3_destination}" --endpoint-url "$BLUEPRINT_R2_ENDPOINT_URL" >/dev/null',
        ]
    if scheme in {"", "file"}:
        return [f"cp {quoted_source} {quoted_destination}"]
    return [f"# Upload {quoted_source} to provider-readable URI {quoted_destination}."]


def _worker_entrypoint_command(
    *,
    allow_simulator_execution: bool,
    allowed_simulators: Sequence[str],
    simulator_commands: Mapping[str, str] | None,
) -> str:
    parts = ["blueprint-run-robot-eval-worker"]
    if allow_simulator_execution:
        parts.append("--allow-simulator-execution")
    for simulator in allowed_simulators:
        parts.extend(["--allowed-simulator", simulator])
    for simulator, command in (simulator_commands or {}).items():
        parts.extend(["--simulator-command", f"{simulator}={command}"])
    return " ".join(shlex.quote(part) for part in parts)


def _rewrite_provider_launch_worker_command(
    *,
    path: Path,
    allow_simulator_execution: bool,
    allowed_simulators: Sequence[str],
    simulator_commands: Mapping[str, str] | None,
) -> Dict[str, Any]:
    if not path.is_file():
        return {"status": "not_available", "path": str(path)}
    payload = read_json_any(path)
    if not isinstance(payload, Mapping):
        return {"status": "blocked", "path": str(path), "blockers": ["invalid_provider_launch_json"]}
    updated = dict(payload)
    provider_shape = dict(updated.get("provider_request_shape") or {})
    command = _worker_entrypoint_command(
        allow_simulator_execution=allow_simulator_execution,
        allowed_simulators=allowed_simulators,
        simulator_commands=simulator_commands,
    )
    provider_shape["command"] = command
    updated["provider_request_shape"] = provider_shape
    write_json(path, updated)
    return {"status": "updated", "path": str(path), "command": command}


def _write_publish_resolution_script(
    *,
    path: Path,
    image_ref: str,
    simulator: str,
    bundle_path: str,
    bundle_uri: str,
    worker_manifest_path: str,
    worker_manifest_uri: str,
) -> Dict[str, Any]:
    dockerfile = (
        "deploy/docker/robot_eval_worker/mujoco/Dockerfile"
        if simulator == "mujoco"
        else f"deploy/docker/robot_eval_worker/{simulator}/Dockerfile"
    )
    bundle_upload_commands = _storage_upload_commands(source=bundle_path, destination_uri=bundle_uri)
    worker_manifest_upload_commands = _storage_upload_commands(
        source=worker_manifest_path,
        destination_uri=worker_manifest_uri,
    )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Contains no secrets. Export registry/object-storage credentials in the shell.",
        f'docker build -f "{dockerfile}" -t "{image_ref}" .',
        f'docker push "{image_ref}"',
        "",
        *bundle_upload_commands,
        *worker_manifest_upload_commands,
        "",
        "# Rerun blueprint-prepare-robot-eval-provider-inputs with --image-ref set to the pushed image.",
        "# Keep --upload enabled only after object-storage credentials and billing are valid.",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return {
        "status": "written",
        "path": str(path),
        "secret_values_in_artifact": False,
        "r2_cli_destinations_use_s3_scheme_with_endpoint": (
            urlparse(bundle_uri).scheme == "r2" or urlparse(worker_manifest_uri).scheme == "r2"
        ),
        "upload_command_count": sum(
            1
            for command in [*bundle_upload_commands, *worker_manifest_upload_commands]
            if command.startswith(("aws s3 cp", "gcloud storage cp", "cp "))
        ),
        "post_upload_verification_command_count": sum(
            1
            for command in [*bundle_upload_commands, *worker_manifest_upload_commands]
            if " ls " in command
        ),
        "requires_object_store_credentials_in_shell": True,
        "rerun_with_upload_required_for_provider_fetchability": True,
    }


def _write_artifact_output_write_probe(
    *,
    path: Path,
    job_id: str,
    artifact_output_uri: str,
) -> Dict[str, Any]:
    payload = {
        "schema_version": "robot_eval_artifact_output_write_probe.v1",
        "generated_at": utc_now_iso(),
        "status": "prepared",
        "job_id": job_id,
        "artifact_output_uri": artifact_output_uri,
        "purpose": "pre_spend_object_store_write_probe",
        "claim_boundary": {
            "object_store_output_write_probe_only": True,
            "provider_runtime_wrote_artifacts": False,
            "remote_cloud_execution_proven": False,
            "clean_shutdown_proven": False,
            "public_claim_upgrade_allowed": False,
        },
        "secret_values_recorded": False,
    }
    write_json(path, payload)
    return {
        "status": "prepared",
        "path": str(path),
        "destination_uri": _uri_join(
            artifact_output_uri,
            "_blueprint_provider_output_write_probe.json",
        ),
        "secret_values_recorded": False,
    }


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _local_sim_only_closure_source(job_dir: Path) -> tuple[Path, str]:
    pipeline_dir = job_dir.parent.parent
    local_gate_report_path = (
        pipeline_dir
        / "live_pipeline_control_plane"
        / "sim_only_beta_local_gate"
        / "sim_only_beta_local_gate_report.json"
    )
    local_gate_report = _read_optional_mapping(local_gate_report_path)
    route_proof_job_id = _string(local_gate_report.get("route_proof_job_id")) or _string(
        local_gate_report.get("job_id")
    )
    if route_proof_job_id:
        route_proof_path = (
            pipeline_dir
            / "robot_eval_jobs"
            / route_proof_job_id
            / "robot_team_grade_eval_closure_manifest.json"
        )
        if route_proof_path.is_file():
            return route_proof_path, "sim_only_beta_local_gate_route_proof_job"
    return job_dir / "robot_team_grade_eval_closure_manifest.json", "current_job"


def _local_sim_only_provider_prerequisite(job_dir: Path) -> Dict[str, Any]:
    path, source_kind = _local_sim_only_closure_source(job_dir)
    base = {
        "schema_version": "robot_eval_provider_local_sim_only_prerequisite.v1",
        "required_before_provider_spend": True,
        "source_artifact": "robot_team_grade_eval_closure_manifest.json",
        "source_kind": source_kind,
        "source_path": str(path),
        "claim_boundary": {
            "provider_spend_requires_local_sim_only_evidence_clean": True,
            "local_sim_only_clean_does_not_prove_remote_provider_execution": True,
            "local_sim_only_clean_does_not_prove_launch_approval": True,
        },
    }
    if not path.is_file():
        return {
            **base,
            "status": "missing",
            "local_sim_only_evidence_clean": False,
            "sim_only_beta_core_complete": None,
            "sim_only_beta_blocked_requirement_ids": [],
            "blockers": ["local_sim_only_closure_manifest_missing"],
        }
    try:
        payload = read_json_any(path)
    except (OSError, ValueError) as exc:
        return {
            **base,
            "status": "unreadable",
            "local_sim_only_evidence_clean": False,
            "sim_only_beta_core_complete": None,
            "sim_only_beta_blocked_requirement_ids": [],
            "blockers": [
                f"local_sim_only_closure_manifest_unreadable:{type(exc).__name__}"
            ],
        }
    closure = _mapping(payload)
    if not closure:
        return {
            **base,
            "status": "invalid",
            "local_sim_only_evidence_clean": False,
            "sim_only_beta_core_complete": None,
            "sim_only_beta_blocked_requirement_ids": [],
            "blockers": ["local_sim_only_closure_manifest_not_json_object"],
        }
    requirements = [
        _mapping(item)
        for item in closure.get("requirements") or []
        if isinstance(item, Mapping)
    ]
    explicit_blocked_ids = _string_list(
        closure.get("sim_only_beta_blocked_requirement_ids")
    )
    requirement_blocked_ids = [
        _string(requirement.get("requirement_id"))
        for requirement in requirements
        if requirement.get("sim_only_beta_required") is True
        and requirement.get("passed") is not True
        and _string(requirement.get("requirement_id"))
    ]
    blocked_ids = sorted({*explicit_blocked_ids, *requirement_blocked_ids})
    blockers_by_requirement = {
        requirement_id: _string_list(requirement.get("blockers"))
        for requirement in requirements
        for requirement_id in [_string(requirement.get("requirement_id"))]
        if requirement_id in blocked_ids
    }
    core_complete = closure.get("sim_only_beta_core_complete") is True
    clean = bool(core_complete and not blocked_ids)
    blockers = (
        []
        if clean
        else [
            "local_sim_only_evidence_not_clean",
            *(
                f"sim_only_beta_requirement_{requirement_id}_not_complete"
                for requirement_id in blocked_ids
            ),
        ]
    )
    if not clean and not blocked_ids and closure.get("sim_only_beta_core_complete") is not True:
        blockers.append("sim_only_beta_core_complete_not_true")
    return {
        **base,
        "status": "passed" if clean else "blocked",
        "local_sim_only_evidence_clean": clean,
        "sim_only_beta_core_complete": core_complete,
        "sim_only_beta_blocked_requirement_ids": blocked_ids,
        "sim_only_beta_requirement_blockers": blockers_by_requirement,
        "blockers": blockers,
    }


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    return {
        "status": "present",
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _uri_scheme(uri: str) -> str:
    return urlparse(uri).scheme or "local"


def _list_count(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value else 0
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return len([item for item in value if item])
    return 0


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _upload_readiness_preflight(
    *,
    destination_uri: str,
    artifact_write_auth: Mapping[str, Any],
) -> Dict[str, Any]:
    scheme = _uri_scheme(destination_uri)
    required_secret_env_vars = _string_values(
        artifact_write_auth.get("required_secret_env_vars")
    )
    required_plaintext_env_vars = _string_values(
        artifact_write_auth.get("required_plaintext_env_vars")
    )
    missing_secret_env_vars = [
        name for name in required_secret_env_vars if not _env_or_file_present(name)
    ]
    present_secret_file_env_vars = _present_file_env_vars(required_secret_env_vars)
    present_plaintext_env_vars = [
        name for name in required_plaintext_env_vars if _env_or_file_present(name)
    ]
    missing_plaintext_env_vars = [
        name for name in required_plaintext_env_vars if not _env_or_file_present(name)
    ]
    present_plaintext_file_env_vars = _present_file_env_vars(required_plaintext_env_vars)
    endpoint_env_var_alternatives = (
        list(R2_ENDPOINT_ENV_VAR_ALTERNATIVES) if scheme == "r2" else []
    )
    endpoint_file_env_var_alternatives = [
        _file_env_name(name) for name in endpoint_env_var_alternatives
    ]
    endpoint_env_var_present = (
        any(_env_or_file_present(name) for name in endpoint_env_var_alternatives)
        if endpoint_env_var_alternatives
        else True
    )
    if scheme == "gs":
        upload_tool = "google-cloud-storage"
        upload_tool_available = _module_available("google.cloud.storage")
    elif scheme in {"s3", "r2"}:
        upload_tool = "boto3"
        upload_tool_available = _module_available("boto3")
    elif scheme in {"local", "file"}:
        upload_tool = "local-filesystem"
        upload_tool_available = True
    else:
        upload_tool = None
        upload_tool_available = False
    blockers: list[str] = []
    if scheme not in {"gs", "s3", "r2", "local", "file"}:
        blockers.append("upload_uri_scheme_not_supported")
    if not upload_tool_available:
        blockers.append(f"upload_tool_missing:{upload_tool or scheme}")
    if missing_secret_env_vars:
        blockers.append("upload_secret_env_vars_missing")
    if scheme == "r2" and not endpoint_env_var_present:
        blockers.append("upload_r2_endpoint_env_missing")
    status = "ready_for_upload_attempt" if not blockers else "blocked_upload_preflight"
    return {
        "schema_version": "robot_eval_provider_input_upload_preflight.v1",
        "status": status,
        "storage_scheme": scheme,
        "upload_tool": upload_tool,
        "upload_tool_available": upload_tool_available,
        "required_secret_env_var_count": len(required_secret_env_vars),
        "missing_secret_env_vars": missing_secret_env_vars,
        "accepted_secret_file_env_vars": [
            _file_env_name(name) for name in required_secret_env_vars
        ],
        "present_secret_file_env_vars": present_secret_file_env_vars,
        "secret_values_recorded": False,
        "required_plaintext_env_vars": required_plaintext_env_vars,
        "present_plaintext_env_var_count": len(present_plaintext_env_vars),
        "missing_plaintext_env_vars": missing_plaintext_env_vars,
        "accepted_plaintext_file_env_vars": [
            _file_env_name(name) for name in required_plaintext_env_vars
        ],
        "present_plaintext_file_env_vars": present_plaintext_file_env_vars,
        "r2_endpoint_env_var_alternatives": endpoint_env_var_alternatives,
        "r2_endpoint_file_env_var_alternatives": endpoint_file_env_var_alternatives,
        "r2_endpoint_file_env_vars_present": _present_file_env_vars(
            endpoint_env_var_alternatives
        ),
        "r2_endpoint_env_var_present": endpoint_env_var_present,
        "provider_input_objects_fetchable_proven": False,
        "provider_output_writability_proven": False,
        "blockers": blockers,
    }


def _image_ref_is_pinned(image_ref: str) -> bool:
    image_tag = image_ref.rsplit("/", 1)[-1]
    return bool(image_ref and ":" in image_tag and not image_tag.endswith(":latest"))


def _provider_package_validation(
    *,
    job_id: str,
    capture_root: Path,
    simulator: str,
    provisioner: str,
    bundle: Mapping[str, Any],
    capture_root_bundle_uri: str,
    worker_manifest_path: Path,
    worker_manifest_uri: str,
    artifact_output_uri: str,
    image_ref: str,
    provider_launch_request: Mapping[str, Any],
    worker_launch_plan: Mapping[str, Any],
    timeout_seconds: int,
    budget_usd: float,
    upload_requested: bool,
    provider_inputs_uploaded: bool,
    upload_results: Sequence[Mapping[str, Any]],
    artifact_output_write_probe: Mapping[str, Any],
    setup_manifest_path: Path,
) -> Dict[str, Any]:
    worker_manifest = _read_optional_mapping(worker_manifest_path)
    provider_shape = dict(provider_launch_request.get("provider_request_shape") or {})
    provider_inputs = dict(provider_shape.get("inputs") or {})
    provider_limits = dict(provider_shape.get("limits") or {})
    artifact_contract = dict(worker_launch_plan.get("artifact_upload_contract") or {})
    artifact_write_auth = dict(
        provider_inputs.get("artifact_output_write_auth")
        or artifact_contract.get("artifact_output_write_auth")
        or {}
    )
    launch_mode = dict(worker_launch_plan.get("launch_mode") or {})
    hard_timeout_seconds = int(
        provider_limits.get("hard_timeout_seconds")
        or launch_mode.get("hard_timeout_seconds")
        or timeout_seconds
        or 0
    )
    external_watchdog_ttl_seconds = int(
        provider_limits.get("external_watchdog_ttl_seconds")
        or launch_mode.get("external_watchdog_ttl_seconds")
        or 0
    )
    idle_timeout_seconds = int(
        provider_limits.get("idle_timeout_seconds")
        or launch_mode.get("idle_timeout_seconds")
        or 0
    )
    max_active_workers = int(
        provider_limits.get("max_active_workers")
        or launch_mode.get("max_active_workers")
        or 0
    )
    bundle_scheme = _uri_scheme(capture_root_bundle_uri)
    worker_manifest_scheme = _uri_scheme(worker_manifest_uri)
    artifact_output_scheme = _uri_scheme(artifact_output_uri)
    provider_artifact_output_writable = bool(
        provider_inputs.get("artifact_output_uri_provider_writable")
        or artifact_contract.get("artifact_output_uri_provider_writable")
    )
    artifact_output_write_auth_contract_ready = bool(
        provider_inputs.get("artifact_output_write_auth_contract_ready")
        or artifact_contract.get("artifact_output_write_auth_contract_ready")
        or artifact_write_auth.get("write_auth_contract_ready")
    )
    upload_preflight = _upload_readiness_preflight(
        destination_uri=artifact_output_uri,
        artifact_write_auth=artifact_write_auth,
    )
    upload_before_shutdown_required = bool(
        dict(provider_shape.get("artifact_finalizer") or {}).get(
            "upload_before_shutdown_required"
        )
        or artifact_contract.get("upload_before_shutdown_required")
    )
    idle_shutdown_required = bool(
        provider_limits.get("idle_shutdown_required")
        or launch_mode.get("idle_shutdown_required")
    )
    blockers: list[str] = []
    external_blockers: list[str] = []
    if provider_launch_request.get("job_id") != job_id:
        blockers.append("provider_launch_request_job_id_mismatch")
    if worker_manifest.get("job_id") != job_id:
        blockers.append("worker_manifest_job_id_mismatch")
    if worker_manifest.get("capture_root") != str(capture_root):
        blockers.append("worker_manifest_capture_root_mismatch")
    if bundle.get("status") != "created" or not bundle.get("size_bytes"):
        blockers.append("capture_root_bundle_missing_or_empty")
    if _file_fingerprint(worker_manifest_path).get("status") != "present":
        blockers.append("worker_manifest_file_missing")
    if not _image_ref_is_pinned(image_ref):
        blockers.append("worker_image_ref_not_pinned")
    if bundle_scheme not in REMOTE_PROVIDER_INPUT_URI_SCHEMES:
        blockers.append("capture_root_bundle_uri_not_provider_fetchable_scheme")
    if worker_manifest_scheme not in REMOTE_PROVIDER_INPUT_URI_SCHEMES:
        blockers.append("worker_manifest_uri_not_provider_fetchable_scheme")
    if artifact_output_scheme not in REMOTE_PROVIDER_ARTIFACT_OUTPUT_URI_SCHEMES:
        blockers.append("artifact_output_uri_not_remote_provider_writable_scheme")
    if not provider_artifact_output_writable:
        blockers.append("artifact_output_uri_not_marked_provider_writable")
    if provider_artifact_output_writable and not artifact_output_write_auth_contract_ready:
        blockers.append("artifact_output_write_auth_contract_missing")
    if hard_timeout_seconds <= 0:
        blockers.append("hard_timeout_seconds_missing")
    if external_watchdog_ttl_seconds <= hard_timeout_seconds:
        blockers.append("external_watchdog_ttl_must_exceed_hard_timeout")
    if budget_usd < 0:
        blockers.append("max_spend_usd_invalid")
    if max_active_workers != 1:
        blockers.append("max_active_workers_must_be_one_for_bounded_attempt")
    if not idle_shutdown_required:
        blockers.append("idle_shutdown_not_required")
    if not upload_before_shutdown_required:
        blockers.append("artifact_upload_before_shutdown_not_required")
    if upload_requested and not provider_inputs_uploaded:
        external_blockers.append("provider_inputs_upload_failed_or_incomplete")
    artifact_output_write_probe_proven = bool(
        artifact_output_write_probe.get("status") == "uploaded"
        and not artifact_output_write_probe.get("blockers")
        and _mapping(artifact_output_write_probe.get("post_upload_validation")).get(
            "status"
        )
        == "validated"
    )
    if upload_requested and not artifact_output_write_probe_proven:
        external_blockers.append("artifact_output_write_probe_not_proven")
    if not upload_requested:
        external_blockers.append("provider_inputs_upload_not_requested")
    status = (
        "blocked_package_contract"
        if blockers
        else "validated_uploaded_provider_inputs"
        if provider_inputs_uploaded
        else "validated_pre_spend_package_pending_upload"
    )
    return {
        "schema_version": "robot_eval_provider_package_validation.v1",
        "generated_at": utc_now_iso(),
        "status": status,
        "job_id": job_id,
        "capture_root": str(capture_root),
        "provisioner": provisioner,
        "simulator": simulator,
        "setup_manifest_path": str(setup_manifest_path),
        "pinned_inputs": {
            "capture_root_bundle": {
                "uri": capture_root_bundle_uri,
                "uri_scheme": bundle_scheme,
                "uri_scheme_provider_fetchable": bundle_scheme
                in REMOTE_PROVIDER_INPUT_URI_SCHEMES,
                "path": bundle.get("path"),
                "file_count": bundle.get("file_count"),
                "size_bytes": bundle.get("size_bytes"),
                "sha256": bundle.get("sha256"),
                "raw_media_excluded_by_default": bundle.get(
                    "raw_media_excluded_by_default"
                ),
            },
            "worker_manifest": {
                "uri": worker_manifest_uri,
                "uri_scheme": worker_manifest_scheme,
                "uri_scheme_provider_fetchable": worker_manifest_scheme
                in REMOTE_PROVIDER_INPUT_URI_SCHEMES,
                **_file_fingerprint(worker_manifest_path),
            },
            "worker_image": {
                "image_ref": image_ref,
                "pinned_by_tag": _image_ref_is_pinned(image_ref),
            },
        },
        "exact_ids": {
            "job_id": job_id,
            "provider_launch_request_job_id": provider_launch_request.get("job_id"),
            "worker_manifest_job_id": worker_manifest.get("job_id"),
            "capture_root": str(capture_root),
            "worker_manifest_capture_root": worker_manifest.get("capture_root"),
        },
        "artifact_output": {
            "uri": artifact_output_uri,
            "uri_scheme": artifact_output_scheme,
            "uri_scheme_provider_writable": artifact_output_scheme
            in REMOTE_PROVIDER_ARTIFACT_OUTPUT_URI_SCHEMES,
            "artifact_output_uri_provider_writable": provider_artifact_output_writable,
            "artifact_output_write_auth_contract_ready": (
                artifact_output_write_auth_contract_ready
            ),
            "artifact_output_write_auth": {
                "authorization_mode": artifact_write_auth.get("authorization_mode"),
                "write_auth_contract_ready": bool(
                    artifact_write_auth.get("write_auth_contract_ready")
                ),
                "required_secret_env_var_count": _list_count(
                    artifact_write_auth.get("required_secret_env_vars")
                ),
                "required_plaintext_env_var_count": _list_count(
                    artifact_write_auth.get("required_plaintext_env_vars")
                ),
                "secret_values_in_artifact": artifact_write_auth.get(
                    "secret_values_in_artifact"
                ),
            },
            "provider_input_objects_fetchable_proven": provider_inputs_uploaded,
            "provider_output_write_test_performed": bool(
                artifact_output_write_probe.get("status") != "not_requested"
            ),
            "object_store_output_write_probe": dict(artifact_output_write_probe),
            "object_store_output_writability_proven": artifact_output_write_probe_proven,
            "provider_output_writability_proven": False,
            "live_provider_runtime_still_required_for_output_writability_proof": True,
            "upload_preflight": upload_preflight,
        },
        "runtime_limits": {
            "hard_timeout_seconds": hard_timeout_seconds,
            "idle_timeout_seconds": idle_timeout_seconds,
            "external_watchdog_ttl_seconds": external_watchdog_ttl_seconds,
            "max_active_workers": max_active_workers,
            "max_spend_usd": budget_usd,
        },
        "teardown_behavior": {
            "idle_shutdown_required": idle_shutdown_required,
            "upload_before_shutdown_required": upload_before_shutdown_required,
            "scale_to_zero_default": bool(launch_mode.get("scale_to_zero_default")),
            "external_watchdog_owner": launch_mode.get("external_watchdog_owner")
            or provider_limits.get("external_watchdog_owner"),
        },
        "upload_validation": {
            "upload_requested": upload_requested,
            "provider_inputs_uploaded": provider_inputs_uploaded,
            "upload_result_count": len(upload_results),
            "upload_results": [dict(result) for result in upload_results],
            "artifact_output_write_probe": dict(artifact_output_write_probe),
            "upload_preflight": upload_preflight,
            "upload_preflight_ready": upload_preflight.get("status")
            == "ready_for_upload_attempt",
            "live_runtime_still_requires_fetchable_uploaded_inputs": True,
        },
        "blockers": blockers,
        "external_blockers": external_blockers,
        "claim_boundary": {
            "provider_package_validated_without_provider_spend": True,
            "runpod_api_called": False,
            "remote_cloud_execution_proven": False,
            "clean_shutdown_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _refresh_remote_cloud_execution_closure_manifest(
    *,
    job_dir: Path,
    job_id: str,
    provisioner: str,
    simulator: str,
) -> None:
    provider_request = _read_optional_mapping(job_dir / "gpu_provider_launch_request.json")
    if not provider_request:
        return
    closure = _remote_cloud_execution_closure_manifest(
        job_id=job_id,
        provisioner=provisioner,
        simulator=simulator,
        worker_launch_plan=_read_optional_mapping(job_dir / "worker_launch_plan.json"),
        worker_manifest=_read_optional_mapping(job_dir / "worker_manifest.json"),
        provider_launch_request=provider_request,
        gpu_result=_read_optional_mapping(job_dir / "gpu_provisioning_result.json"),
        gpu_cost_ledger=_read_optional_mapping(job_dir / "gpu_cost_control_ledger.json"),
        sim_result=_read_optional_mapping(job_dir / "simulator_service_result.json"),
        generated_at=utc_now_iso(),
    )
    write_json(job_dir / "remote_cloud_execution_closure_manifest.json", closure)


def _annotate_provider_launch_request(
    *,
    job_dir: Path,
    job_id: str,
    provisioner: str,
    simulator: str,
    setup_manifest: Mapping[str, Any],
    setup_manifest_path: Path,
) -> None:
    request_path = job_dir / "gpu_provider_launch_request.json"
    if not request_path.is_file():
        return
    request = read_json_any(request_path)
    if not isinstance(request, dict):
        return
    blockers = _dedupe(str(blocker) for blocker in setup_manifest.get("blockers") or [])
    local_sim_only_prerequisite = _local_sim_only_provider_prerequisite(job_dir)
    local_prereq_blockers = _string_list(local_sim_only_prerequisite.get("blockers"))
    provider_shape = _mapping(request.get("provider_request_shape"))
    provider_shape["local_sim_only_prerequisite"] = local_sim_only_prerequisite
    request["provider_request_shape"] = provider_shape
    request["provider_input_setup"] = {
        "status": setup_manifest.get("status"),
        "manifest_path": str(setup_manifest_path),
        "blockers": blockers,
        "provider_inputs_uploaded": bool(
            (setup_manifest.get("proof_boundary") or {}).get("provider_inputs_uploaded")
            if isinstance(setup_manifest.get("proof_boundary"), Mapping)
            else False
        ),
        "image_ref_published_proven": bool(
            (setup_manifest.get("proof_boundary") or {}).get("image_ref_published_proven")
            if isinstance(setup_manifest.get("proof_boundary"), Mapping)
            else False
        ),
        "capture_root_bundle_uri": setup_manifest.get("capture_root_bundle_uri"),
        "worker_manifest_uri": setup_manifest.get("worker_manifest_uri"),
        "artifact_output_uri": setup_manifest.get("artifact_output_uri"),
        "publish_resolution": setup_manifest.get("publish_resolution"),
    }
    existing_blockers = _string_list(request.get("blockers"))
    provider_setup_blocker_values = {"provider_input_setup_blocked"}
    local_prereq_blocker_values = {
        "local_sim_only_prerequisite_blocked",
        "local_sim_only_evidence_not_clean",
        "local_sim_only_closure_manifest_missing",
        "sim_only_beta_core_complete_not_true",
    }
    preserved_blockers = [
        blocker
        for blocker in existing_blockers
        if not (not blockers and blocker in provider_setup_blocker_values)
        and not (
            not local_prereq_blockers
            and (
                blocker in local_prereq_blocker_values
                or blocker.startswith("sim_only_beta_requirement_")
            )
        )
    ]
    if blockers or local_prereq_blockers:
        existing_status = _string(request.get("status"))
        if blockers:
            request["status"] = "blocked_provider_input_setup"
        elif existing_status == "request_manifest_ready":
            request["status"] = "blocked_local_sim_only_prerequisite"
        request["blockers"] = _dedupe(
            [
                *preserved_blockers,
                *(["provider_input_setup_blocked"] if blockers else []),
                *blockers,
                *(["local_sim_only_prerequisite_blocked"] if local_prereq_blockers else []),
                *local_prereq_blockers,
            ]
        )
    else:
        request["blockers"] = preserved_blockers
        if _string(request.get("status")) in {
            "blocked_provider_input_setup",
            "blocked_local_sim_only_prerequisite",
        }:
            request["status"] = "blocked" if preserved_blockers else "request_manifest_ready"
    package_validation = setup_manifest.get("provider_package_validation")
    if isinstance(package_validation, Mapping):
        request["provider_input_setup"]["provider_package_validation"] = dict(
            package_validation
        )
    write_json(request_path, request)
    _refresh_remote_cloud_execution_closure_manifest(
        job_dir=job_dir,
        job_id=job_id,
        provisioner=provisioner,
        simulator=simulator,
    )


def prepare_robot_eval_provider_inputs(
    *,
    capture_root: str | Path,
    job_request: str | Path,
    job_id: str,
    artifact_root_uri: str,
    simulator: str = "mujoco",
    provisioner: str = "runpod",
    image_ref: str | None = None,
    output_dir: str | Path | None = None,
    include_raw_media: bool = False,
    upload: bool = False,
    allow_gpu_provisioning: bool = True,
    allow_simulator_execution: bool = False,
    allowed_simulators: Sequence[str] = (),
    simulator_commands: Mapping[str, str] | None = None,
    timeout_seconds: int = 600,
    budget_usd: float = 10.0,
) -> Dict[str, Any]:
    root = Path(capture_root).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(output_dir or root / "pipeline" / "robot_eval_provider_inputs" / job_id)
    ensure_dir(out_dir)
    selected_image_ref = image_ref or _string(
        os.getenv(WORKER_IMAGE_REF_ENV_BY_SIMULATOR.get(simulator, ""))
    )
    candidate_image_ref = selected_image_ref or default_image_ref(
        simulator=simulator,
        repo_root=repo_root,
    )
    bundle_path = out_dir / "capture-root.zip"
    bundle_uri = _uri_join(artifact_root_uri, "capture-root.zip")
    worker_manifest_uri = _uri_join(artifact_root_uri, "worker_manifest.json")
    artifact_output_uri = _uri_join(artifact_root_uri, "artifacts")
    input_upload_results: list[Dict[str, Any]] = []

    image_env_var = WORKER_IMAGE_REF_ENV_BY_SIMULATOR.get(simulator, "BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF")
    env_values = {
        image_env_var: candidate_image_ref,
        WORKER_MANIFEST_URI_ENV: worker_manifest_uri,
        WORKER_ARTIFACT_OUTPUT_URI_ENV: artifact_output_uri,
        WORKER_CAPTURE_ROOT_BUNDLE_URI_ENV: bundle_uri,
        "BLUEPRINT_ALLOW_GPU_PROVISIONING": "true" if allow_gpu_provisioning else "",
        "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION": "true" if allow_simulator_execution else "",
    }
    with _patched_env(env_values):
        job_result = execute_robot_eval_request_as_evaluation_run(
            capture_root=root,
            job_request=Path(job_request).resolve(),
            job_id=job_id,
            provisioner=provisioner,
            simulator=simulator,
            allow_gpu_provisioning=allow_gpu_provisioning,
            allow_simulator_execution=allow_simulator_execution,
            allowed_simulators=list(allowed_simulators),
            simulator_commands=dict(simulator_commands or {}),
            timeout_seconds=timeout_seconds,
            budget_usd=budget_usd,
        )

    job_dir = Path(_string(job_result.get("job_dir")) or root / "pipeline" / "robot_eval_jobs" / job_id)
    provider_launch_command = _rewrite_provider_launch_worker_command(
        path=job_dir / "gpu_provider_launch_request.json",
        allow_simulator_execution=allow_simulator_execution,
        allowed_simulators=list(allowed_simulators),
        simulator_commands=simulator_commands,
    )
    bundle_include_paths = (
        *DEFAULT_INCLUDE_PATHS,
        f"pipeline/robot_eval_jobs/{job_id}",
    )
    bundle = build_capture_root_bundle(
        capture_root=root,
        output_path=bundle_path,
        include_raw_media=include_raw_media,
        include_paths=bundle_include_paths,
    )
    if upload:
        input_upload_results.append(upload_file(bundle_path, bundle_uri))
    worker_manifest_path = job_dir / "worker_manifest.json"
    if upload and worker_manifest_path.is_file():
        input_upload_results.append(upload_file(worker_manifest_path, worker_manifest_uri))
    artifact_output_write_probe: Dict[str, Any] = {
        "schema_version": "robot_eval_artifact_output_write_probe_upload.v1",
        "status": "not_requested",
        "reason": "upload_not_requested",
        "artifact_output_uri": artifact_output_uri,
        "secret_values_recorded": False,
    }
    if upload:
        probe = _write_artifact_output_write_probe(
            path=out_dir / "artifact_output_write_probe.json",
            job_id=job_id,
            artifact_output_uri=artifact_output_uri,
        )
        artifact_output_write_probe = {
            **probe,
            **upload_file(probe["path"], str(probe["destination_uri"])),
        }

    env_file = _write_env_file(out_dir / "provider_input_env.sh", env_values)
    publish_resolution = _write_publish_resolution_script(
        path=out_dir / "provider_publish_resolution.sh",
        image_ref=candidate_image_ref,
        simulator=simulator,
        bundle_path=str(bundle_path),
        bundle_uri=bundle_uri,
        worker_manifest_path=str(worker_manifest_path),
        worker_manifest_uri=worker_manifest_uri,
    )
    blockers: list[str] = []
    if not selected_image_ref:
        blockers.append("worker_image_ref_is_candidate_until_built_and_pushed")
    for result in input_upload_results:
        blockers.extend(result.get("blockers", []))
    if upload and len(input_upload_results) < 2:
        blockers.append("worker_manifest_upload_missing")
    external_provider = provisioner != "fixture_local"
    provider_inputs_uploaded = upload and len(input_upload_results) >= 2 and not any(
        result.get("blockers") for result in input_upload_results
    )
    artifact_output_write_probe_proven = bool(
        artifact_output_write_probe.get("status") == "uploaded"
        and not artifact_output_write_probe.get("blockers")
        and _mapping(artifact_output_write_probe.get("post_upload_validation")).get(
            "status"
        )
        == "validated"
    )
    if external_provider and not provider_inputs_uploaded:
        blockers.append("provider_inputs_upload_not_proven")
    if upload and not artifact_output_write_probe_proven:
        blockers.append("artifact_output_write_probe_not_proven")
    blockers = _dedupe(blockers)
    status = "ready_for_provider_launcher_inputs" if not blockers else "prepared_with_external_blockers"
    manifest_path = out_dir / "provider_input_setup_manifest.json"
    provider_launch_request = _read_optional_mapping(job_dir / "gpu_provider_launch_request.json")
    worker_launch_plan = _read_optional_mapping(job_dir / "worker_launch_plan.json")
    package_validation = _provider_package_validation(
        job_id=job_id,
        capture_root=root,
        simulator=simulator,
        provisioner=provisioner,
        bundle=bundle,
        capture_root_bundle_uri=bundle_uri,
        worker_manifest_path=worker_manifest_path,
        worker_manifest_uri=worker_manifest_uri,
        artifact_output_uri=artifact_output_uri,
        image_ref=candidate_image_ref,
        provider_launch_request=provider_launch_request,
        worker_launch_plan=worker_launch_plan,
        timeout_seconds=timeout_seconds,
        budget_usd=budget_usd,
        upload_requested=upload,
        provider_inputs_uploaded=provider_inputs_uploaded,
        upload_results=input_upload_results,
        artifact_output_write_probe=artifact_output_write_probe,
        setup_manifest_path=manifest_path,
    )
    manifest = {
        "schema_version": PROVIDER_INPUT_SETUP_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "capture_root": str(root),
        "job_id": job_id,
        "simulator": simulator,
        "provisioner": provisioner,
        "artifact_root_uri": artifact_root_uri,
        "bundle": bundle,
        "capture_root_bundle_uri": bundle_uri,
        "worker_manifest_uri": worker_manifest_uri,
        "artifact_output_uri": artifact_output_uri,
        "image_ref": {
            "env_var": image_env_var,
            "configured_or_candidate": candidate_image_ref,
            "provided_by_user_or_env": bool(selected_image_ref),
            "candidate_build_command": (
                "docker build -f deploy/docker/robot_eval_worker/mujoco/Dockerfile "
                f"-t {candidate_image_ref} ."
                if simulator == "mujoco"
                else None
            ),
            "candidate_push_command": f"docker push {candidate_image_ref}",
        },
        "job_result": job_result,
        "provider_launch_command": provider_launch_command,
        "worker_manifest_path": str(worker_manifest_path),
        "upload_requested": upload,
        "upload_results": input_upload_results,
        "artifact_output_write_probe": artifact_output_write_probe,
        "provider_package_validation": package_validation,
        "env_file": env_file,
        "publish_resolution": publish_resolution,
        "blockers": blockers,
        "proof_boundary": {
            "provider_inputs_prepared": True,
            "provider_inputs_uploaded": provider_inputs_uploaded,
            "provider_package_validated": str(package_validation.get("status", "")).startswith(
                "validated"
            ),
            "image_ref_published_proven": bool(selected_image_ref),
            "runpod_api_called": False,
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    write_json(manifest_path, manifest)
    _annotate_provider_launch_request(
        job_dir=job_dir,
        job_id=job_id,
        provisioner=provisioner,
        simulator=simulator,
        setup_manifest=manifest,
        setup_manifest_path=manifest_path,
    )
    return manifest


def _parse_simulator_commands(values: Sequence[str]) -> Dict[str, str]:
    commands: Dict[str, str] = {}
    for value in values:
        simulator, sep, command = value.partition("=")
        if not sep or not simulator.strip() or not command.strip():
            raise ValueError("--simulator-command must be formatted as <simulator>=<command>")
        commands[simulator.strip()] = command.strip()
    return commands


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare robot-eval provider worker inputs.")
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--job-request", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--artifact-root-uri", required=True)
    parser.add_argument("--simulator", default="mujoco")
    parser.add_argument("--provisioner", default="runpod")
    parser.add_argument("--image-ref")
    parser.add_argument("--output-dir")
    parser.add_argument("--include-raw-media", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--allow-simulator-execution", action="store_true")
    parser.add_argument("--allowed-simulator", action="append", default=[])
    parser.add_argument("--simulator-command", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--budget-usd", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = prepare_robot_eval_provider_inputs(
        capture_root=args.capture_root,
        job_request=args.job_request,
        job_id=args.job_id,
        artifact_root_uri=args.artifact_root_uri,
        simulator=args.simulator,
        provisioner=args.provisioner,
        image_ref=args.image_ref,
        output_dir=args.output_dir,
        include_raw_media=args.include_raw_media,
        upload=args.upload,
        allow_simulator_execution=args.allow_simulator_execution,
        allowed_simulators=args.allowed_simulator,
        simulator_commands=_parse_simulator_commands(args.simulator_command),
        timeout_seconds=args.timeout_seconds,
        budget_usd=args.budget_usd,
    )
    print(
        "[robot-eval-provider-input-setup] manifest="
        + str(Path(result["env_file"]["path"]).with_name("provider_input_setup_manifest.json"))
    )
    print(f"[robot-eval-provider-input-setup] status={result['status']}")
    blockers = result.get("blockers") or []
    if blockers:
        print("[robot-eval-provider-input-setup] blockers=" + ",".join(blockers))
    return 0 if result.get("status") == "ready_for_provider_launcher_inputs" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
