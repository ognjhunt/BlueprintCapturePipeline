"""Small shared CLI helpers for the canonical paid resource allocator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .common import ensure_dir, write_json


DETACHED_GPU_CANARY_SUPERVISOR_ENV = "BLUEPRINT_DETACHED_GPU_CANARY_SUPERVISOR"
DETACHED_MODEL_VOLUME_SUPERVISOR_ENV = "BLUEPRINT_DETACHED_MODEL_VOLUME_SUPERVISOR"
DETACHED_CPU_BUILD_SUPERVISOR_ENV = "BLUEPRINT_DETACHED_CPU_BUILD_SUPERVISOR"
LAUNCH_DETACHED_GPU_CANARY_SUPERVISOR_DIR_ENV = (
    "BLUEPRINT_LAUNCH_DETACHED_GPU_CANARY_SUPERVISOR_DIR"
)
DETACHED_GPU_CANARY_MANIFEST = "detached_gpu_canary_supervisor.json"
DETACHED_GPU_CANARY_LOG = "detached_gpu_canary_supervisor.log"
DETACHED_GPU_CANARY_LOCK = "detached_gpu_canary_supervisor.lock"


def maybe_launch_detached_gpu_canary(
    *,
    command: str,
    execute: bool,
    supervisor_dir: str | None,
    argv: Sequence[str],
    repo_root: Path,
) -> dict[str, Any] | None:
    """Detach the canonical allocator without exposing its arguments or secrets."""

    if command != "gpu-canary" or not supervisor_dir:
        return None
    if os.getenv(DETACHED_GPU_CANARY_SUPERVISOR_ENV) == "1":
        return None
    if not execute:
        return {
            "status": "blocked",
            "blockers": ["detached_gpu_canary_requires_execute"],
            "provider_mutations_performed": 0,
        }
    declared = Path(supervisor_dir).expanduser()
    if declared.exists() and declared.is_symlink():
        return {
            "status": "blocked",
            "blockers": ["detached_gpu_canary_supervisor_dir_symlink_forbidden"],
            "provider_mutations_performed": 0,
        }
    root = declared.resolve()
    ensure_dir(root)
    os.chmod(root, 0o700)
    manifest_path = root / DETACHED_GPU_CANARY_MANIFEST
    log_path = root / DETACHED_GPU_CANARY_LOG
    lock_path = root / DETACHED_GPU_CANARY_LOCK
    if manifest_path.exists() or log_path.exists() or lock_path.exists():
        return {
            "status": "blocked",
            "blockers": ["detached_gpu_canary_supervisor_artifact_exists"],
            "provider_mutations_performed": 0,
        }
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return {
            "status": "blocked",
            "blockers": ["detached_gpu_canary_supervisor_already_starting"],
            "provider_mutations_performed": 0,
        }
    os.close(lock_fd)
    raw_arguments = [str(value) for value in argv]
    argument_shape = [
        value.split("=", 1)[0]
        for index, value in enumerate(raw_arguments)
        if index == 0 or value.startswith("--")
    ]
    argument_shape_digest = (
        "sha256:" + hashlib.sha256("\0".join(argument_shape).encode("utf-8")).hexdigest()
    )
    pending = {
        "schema_version": "detached_gpu_canary_supervisor.v1",
        "status": "launch_pending",
        "argument_shape_digest": argument_shape_digest,
        "argument_count": len(raw_arguments),
        "raw_arguments_recorded": False,
        "raw_argument_values_hashed": False,
        "raw_secret_values_recorded": False,
        "provider_mutations_performed_by_launcher": 0,
    }
    write_json(manifest_path, pending)
    os.chmod(manifest_path, 0o600)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    child_env = dict(os.environ)
    child_env[DETACHED_GPU_CANARY_SUPERVISOR_ENV] = "1"
    try:
        with os.fdopen(log_fd, "ab", closefd=True) as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "blueprint_pipeline.paid_resource_allocator",
                    *raw_arguments,
                ],
                cwd=str(repo_root),
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except BaseException:
        write_json(
            manifest_path,
            {
                **pending,
                "status": "launch_failed",
                "blockers": ["detached_gpu_canary_supervisor_launch_failed"],
            },
        )
        os.chmod(manifest_path, 0o600)
        raise
    result = {
        **pending,
        "status": "supervisor_started",
        "pid": process.pid,
        "independent_process": True,
        "start_new_session": True,
        "sigint_ignored_by_child": True,
        "log_path": str(log_path),
    }
    write_json(manifest_path, result)
    os.chmod(manifest_path, 0o600)
    return result


def configure_or_launch_detached_gpu_canary(
    command: str,
    *,
    execute: bool,
    argv: Sequence[str],
    repo_root: Path,
) -> int | None:
    """Detach a requested GPU canary, otherwise configure child signal policy."""

    supervisor_dir = os.getenv(LAUNCH_DETACHED_GPU_CANARY_SUPERVISOR_DIR_ENV)
    detached = maybe_launch_detached_gpu_canary(
        command=command,
        execute=execute,
        supervisor_dir=supervisor_dir,
        argv=argv,
        repo_root=repo_root,
    )
    if detached is not None:
        print(json.dumps(detached, sort_keys=True))
        return 0 if detached.get("status") == "supervisor_started" else 2
    configure_detached_supervisor_signal_policy(
        command,
        detached_model_volume_env=DETACHED_MODEL_VOLUME_SUPERVISOR_ENV,
        detached_cpu_build_env=DETACHED_CPU_BUILD_SUPERVISOR_ENV,
    )
    return None


def configure_detached_supervisor_signal_policy(
    command: str,
    *,
    detached_model_volume_env: str,
    detached_cpu_build_env: str,
) -> bool:
    """Keep an explicitly detached paid supervisor alive through local SIGINT.

    SIGTERM remains available for an intentional stop. Provider resources also
    remain bounded by their independent deadline watchdogs.
    """

    detached_model_volume = (
        command == "model-volume-run" and os.getenv(detached_model_volume_env) == "1"
    )
    detached_cpu_build = command == "cpu-build-run" and os.getenv(detached_cpu_build_env) == "1"
    detached_gpu_canary = (
        command == "gpu-canary" and os.getenv(DETACHED_GPU_CANARY_SUPERVISOR_ENV) == "1"
    )
    if not (detached_model_volume or detached_cpu_build or detached_gpu_canary):
        return False
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, OSError, ValueError):
        return False
    return True


def add_cpu_arguments(parser: argparse.ArgumentParser, *, require_provider: bool = True) -> None:
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--builder-evidence", required=True)
    parser.add_argument("--spend", required=True)
    parser.add_argument("--token-file", default="~/.blueprint-secrets/digitalocean_api_token")
    parser.add_argument("--docker-username-file", default="~/.blueprint-secrets/docker_username")
    parser.add_argument("--docker-password-file", default="~/.blueprint-secrets/docker_pat")
    parser.add_argument("--hf-token-file", default="~/.blueprint-secrets/hf_token")
    parser.add_argument(
        "--runpod-s3-access-key-file",
        default="~/.blueprint-secrets/runpod_s3_access_key",
    )
    parser.add_argument(
        "--runpod-s3-secret-key-file",
        default="~/.blueprint-secrets/runpod_s3_secret_key",
    )
    parser.add_argument("--login-private-key", required=require_provider)
    parser.add_argument("--host-private-key", required=require_provider)
    parser.add_argument("--ssh-key-id", required=require_provider, type=int)
    parser.add_argument("--region", default="sfo3")
    parser.add_argument("--allow-paid", action="store_true")


def cpu_vector(args: argparse.Namespace) -> list[str]:
    values = [
        "--output-dir",
        args.output_dir,
        "--packet-manifest",
        args.packet_manifest,
        "--builder-evidence",
        args.builder_evidence,
        "--spend",
        args.spend,
        "--token-file",
        args.token_file,
        "--docker-username-file",
        args.docker_username_file,
        "--docker-password-file",
        args.docker_password_file,
        "--hf-token-file",
        args.hf_token_file,
        "--runpod-s3-access-key-file",
        args.runpod_s3_access_key_file,
        "--runpod-s3-secret-key-file",
        args.runpod_s3_secret_key_file,
        "--login-private-key",
        args.login_private_key,
        "--host-private-key",
        args.host_private_key,
        "--ssh-key-id",
        str(args.ssh_key_id),
        "--region",
        args.region,
    ]
    if args.allow_paid:
        values.append("--allow-paid")
    return values


def missing_cpu_provider_arguments(args: argparse.Namespace) -> list[str]:
    return [
        name
        for name in ("login_private_key", "host_private_key", "ssh_key_id")
        if getattr(args, name, None) in (None, "")
    ]


def cpu_prerequisite_blocked_result(prerequisite: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "blueprint.cpu_build_allocator_result.v1",
        "status": "blocked_before_allocation",
        "blockers": prerequisite.get("blockers", ["groot_oscar_live_prerequisites_not_ready"]),
        "provider_mutation_attempted": False,
    }


def cpu_builder_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "output_dir": Path(args.output_dir),
        "packet_manifest_path": Path(args.packet_manifest),
        "builder_evidence_path": Path(args.builder_evidence),
        "spend_path": Path(args.spend),
        "token_file": Path(args.token_file),
        "docker_username_file": Path(args.docker_username_file),
        "docker_password_file": Path(args.docker_password_file),
        "login_private_key": Path(args.login_private_key),
        "host_private_key": Path(args.host_private_key),
        "ssh_key_id": args.ssh_key_id,
        "region": args.region,
        "allow_paid": args.allow_paid,
        "hf_token_file": Path(args.hf_token_file),
        "runpod_s3_access_key_file": Path(args.runpod_s3_access_key_file),
        "runpod_s3_secret_key_file": Path(args.runpod_s3_secret_key_file),
    }


__all__ = [
    "DETACHED_GPU_CANARY_LOG",
    "DETACHED_GPU_CANARY_LOCK",
    "DETACHED_GPU_CANARY_MANIFEST",
    "DETACHED_GPU_CANARY_SUPERVISOR_ENV",
    "DETACHED_MODEL_VOLUME_SUPERVISOR_ENV",
    "DETACHED_CPU_BUILD_SUPERVISOR_ENV",
    "LAUNCH_DETACHED_GPU_CANARY_SUPERVISOR_DIR_ENV",
    "add_cpu_arguments",
    "cpu_builder_kwargs",
    "cpu_prerequisite_blocked_result",
    "configure_detached_supervisor_signal_policy",
    "configure_or_launch_detached_gpu_canary",
    "cpu_vector",
    "missing_cpu_provider_arguments",
    "maybe_launch_detached_gpu_canary",
]
