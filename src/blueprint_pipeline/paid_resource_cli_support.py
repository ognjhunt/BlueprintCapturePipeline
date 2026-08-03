"""Small shared CLI helpers for the canonical paid resource allocator."""

from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path
from typing import Any


DETACHED_GPU_CANARY_SUPERVISOR_ENV = "BLUEPRINT_DETACHED_GPU_CANARY_SUPERVISOR"


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
    "DETACHED_GPU_CANARY_SUPERVISOR_ENV",
    "add_cpu_arguments",
    "cpu_builder_kwargs",
    "cpu_prerequisite_blocked_result",
    "configure_detached_supervisor_signal_policy",
    "cpu_vector",
    "missing_cpu_provider_arguments",
]
