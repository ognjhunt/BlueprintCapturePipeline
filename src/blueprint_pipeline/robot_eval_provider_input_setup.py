"""Prepare provider-fetchable inputs for a robot-eval worker launch.

This command packages the current capture inputs for a remote worker, computes
the worker manifest/artifact URIs, optionally uploads provider inputs to object
storage, and reruns the job orchestrator with the resulting environment. It does
not call RunPod or any other GPU provider.
"""

from __future__ import annotations

import argparse
import contextlib
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
    build_robot_eval_job,
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


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


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
    endpoint_url = (
        os.getenv("BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL")
        or os.getenv("R2_ENDPOINT_URL")
        or os.getenv("AWS_ENDPOINT_URL")
        or None
    )
    kwargs: Dict[str, Any] = {}
    if parsed.scheme == "r2" and endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"):
        kwargs["region_name"] = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    client = boto3.client("s3", **kwargs)
    client.upload_file(str(source), parsed.netloc, parsed.path.lstrip("/"))
    return {
        "status": "uploaded",
        "source": str(source),
        "destination_uri": destination_uri,
        "storage_scheme": parsed.scheme,
        "endpoint_configured": bool(endpoint_url),
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
            return _upload_file_to_gs(path, destination_uri)
        if scheme in {"s3", "r2"}:
            return _upload_file_to_s3_compatible(path, destination_uri)
        if scheme in {"", "file"}:
            destination = Path(urlparse(destination_uri).path if scheme else destination_uri).resolve()
            ensure_dir(destination.parent)
            destination.write_bytes(path.read_bytes())
            return {
                "status": "uploaded",
                "source": str(path),
                "destination_uri": str(destination),
                "storage_scheme": "file",
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
        return [f"gcloud storage cp {quoted_source} {quoted_destination}"]
    if scheme == "s3":
        return [f"aws s3 cp {quoted_source} {quoted_destination}"]
    if scheme == "r2":
        return [
            "# Configure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, and endpoint in the shell.",
            f"aws s3 cp {quoted_source} {quoted_destination} --endpoint-url \"$BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL\"",
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
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Contains no secrets. Export registry/object-storage credentials in the shell.",
        f'docker build -f "{dockerfile}" -t "{image_ref}" .',
        f'docker push "{image_ref}"',
        "",
        *_storage_upload_commands(source=bundle_path, destination_uri=bundle_uri),
        *_storage_upload_commands(source=worker_manifest_path, destination_uri=worker_manifest_uri),
        "",
        "# Rerun blueprint-prepare-robot-eval-provider-inputs with --image-ref set to the pushed image.",
        "# Keep --upload enabled only after object-storage credentials and billing are valid.",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return {"status": "written", "path": str(path)}


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


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
        gpu_result=_read_optional_mapping(job_dir / "gpu_provisioner_result.json"),
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
    if blockers:
        request["status"] = "blocked_provider_input_setup"
        request["blockers"] = _dedupe(
            [
                *(str(blocker) for blocker in request.get("blockers") or []),
                "provider_input_setup_blocked",
                *blockers,
            ]
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
    upload_results: list[Dict[str, Any]] = []

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
        job_result = build_robot_eval_job(
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
        upload_results.append(upload_file(bundle_path, bundle_uri))
    worker_manifest_path = job_dir / "worker_manifest.json"
    if upload and worker_manifest_path.is_file():
        upload_results.append(upload_file(worker_manifest_path, worker_manifest_uri))

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
    for result in upload_results:
        blockers.extend(result.get("blockers", []))
    if upload and len(upload_results) < 2:
        blockers.append("worker_manifest_upload_missing")
    external_provider = provisioner != "fixture_local"
    provider_inputs_uploaded = upload and len(upload_results) >= 2 and not any(
        result.get("blockers") for result in upload_results
    )
    if external_provider and not provider_inputs_uploaded:
        blockers.append("provider_inputs_upload_not_proven")
    blockers = _dedupe(blockers)
    status = "ready_for_provider_launcher_inputs" if not blockers else "prepared_with_external_blockers"
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
        "upload_results": upload_results,
        "env_file": env_file,
        "publish_resolution": publish_resolution,
        "blockers": blockers,
        "proof_boundary": {
            "provider_inputs_prepared": True,
            "provider_inputs_uploaded": provider_inputs_uploaded,
            "image_ref_published_proven": bool(selected_image_ref),
            "runpod_api_called": False,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    manifest_path = out_dir / "provider_input_setup_manifest.json"
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
