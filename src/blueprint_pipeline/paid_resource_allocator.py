"""Canonical paid allocators for CPU image builds and GPU canaries.

No other public command may allocate these resource classes.  Provider-specific
modules are adapters behind this interface and their mutation CLIs are disabled.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .common import ensure_dir, write_json
from .groot_oscar_digitalocean_builder import (
    launch_detached_builder,
    observe_local_machine,
    run_builder,
)
from .groot_oscar_infrastructure_admission import (
    BUILD_SCHEMA_VERSION,
    build_build_plane_admission,
    build_cpu_build_execution_admission,
)
from .groot_oscar_runpod_canary import run_canary
from .groot_oscar_runpod_model_volume import launch_detached as launch_detached_model_volume
from .groot_oscar_runpod_model_volume import run_model_volume
from .paid_resource_admission import require_paid_resource_admission


ROOT = Path(__file__).resolve().parents[2]
CPU_BUILD_PREREQUISITE_EVIDENCE = "groot_oscar_live_prerequisites.json"


def _add_cpu_arguments(parser: argparse.ArgumentParser, *, require_provider: bool = True) -> None:
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--builder-evidence", required=True)
    parser.add_argument("--spend", required=True)
    parser.add_argument("--token-file", default="~/.blueprint-secrets/digitalocean_api_token")
    parser.add_argument("--docker-username-file", default="~/.blueprint-secrets/docker_username")
    parser.add_argument("--docker-password-file", default="~/.blueprint-secrets/docker_pat")
    parser.add_argument("--login-private-key", required=require_provider)
    parser.add_argument("--host-private-key", required=require_provider)
    parser.add_argument("--ssh-key-id", required=require_provider, type=int)
    parser.add_argument("--region", default="sfo3")
    parser.add_argument("--allow-paid", action="store_true")


def _cpu_vector(args: argparse.Namespace) -> list[str]:
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


def _missing_cpu_provider_arguments(args: argparse.Namespace) -> list[str]:
    return [
        name
        for name in ("login_private_key", "host_private_key", "ssh_key_id")
        if getattr(args, name, None) in (None, "")
    ]


def _run_cpu(args: argparse.Namespace) -> dict:
    prerequisite = _run_cpu_prerequisite_gate(Path(args.output_dir))
    if prerequisite.get("status") != "ready":
        return {
            "schema_version": "blueprint.cpu_build_allocator_result.v1",
            "status": "blocked_before_allocation",
            "blockers": prerequisite.get(
                "blockers", ["groot_oscar_live_prerequisites_not_ready"]
            ),
            "provider_mutation_attempted": False,
        }
    return run_builder(
        output_dir=Path(args.output_dir),
        packet_manifest_path=Path(args.packet_manifest),
        builder_evidence_path=Path(args.builder_evidence),
        spend_path=Path(args.spend),
        token_file=Path(args.token_file),
        docker_username_file=Path(args.docker_username_file),
        docker_password_file=Path(args.docker_password_file),
        login_private_key=Path(args.login_private_key),
        host_private_key=Path(args.host_private_key),
        ssh_key_id=args.ssh_key_id,
        region=args.region,
        allow_paid=args.allow_paid,
    )


def _run_cpu_prerequisite_gate(output_dir: Path) -> dict:
    """Run the read-only upstream gate before a provider create call."""
    output = output_dir.expanduser().resolve()
    ensure_dir(output)
    evidence = output / CPU_BUILD_PREREQUISITE_EVIDENCE
    verifier = ROOT / "scripts/verify_groot_oscar_live_prerequisites.py"
    completed = subprocess.run(
        [sys.executable, str(verifier), "--live", "--output", str(evidence)],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if evidence.is_file():
        result = _load(evidence)
    else:
        result = {
            "schema": "groot_oscar_live_prerequisites.v1",
            "status": "blocked",
            "live": True,
            "blockers": [
                "groot_oscar_live_prerequisite_verifier_failed_without_evidence"
            ],
            "checks": {},
            "verifier_exit_code": completed.returncode,
        }
        write_json(evidence, result)
    if completed.returncode != 0 and result.get("status") == "ready":
        result["status"] = "blocked"
        result["blockers"] = ["groot_oscar_live_prerequisite_verifier_exit_nonzero"]
        result["verifier_exit_code"] = completed.returncode
        write_json(evidence, result)
    return result


def _load(path: str | Path) -> dict:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _run_local_cpu_build(args: argparse.Namespace) -> dict:
    output = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output)
    allocation = build_build_plane_admission(
        packet=_load(args.packet_manifest),
        builder=_load(args.builder_evidence),
        spend=_load(args.spend),
    )
    write_json(output / "build_plane_admission.json", allocation)
    require_paid_resource_admission(
        allocation,
        resource_class="cpu_build",
        expected_schema_version=BUILD_SCHEMA_VERSION,
    )
    live = observe_local_machine(mount_path=args.mount_path)
    write_json(output / "live_machine_capability.json", live)
    execution = build_cpu_build_execution_admission(
        allocation_admission=allocation, live_machine=live
    )
    write_json(output / "cpu_build_execution_admission.json", execution)
    require_paid_resource_admission(
        execution,
        resource_class="cpu_build",
        expected_schema_version=execution["schema_version"],
    )
    environment = dict(os.environ)
    environment["BLUEPRINT_CANONICAL_CPU_BUILD_CONTEXT"] = "true"
    completed = subprocess.run(
        [str(Path(args.build_script).expanduser().resolve())],
        cwd=str(Path(args.build_workdir).expanduser().resolve()),
        env=environment,
        check=False,
    )
    result = {
        "schema_version": "blueprint.local_cpu_build_allocator_result.v1",
        "status": "completed" if completed.returncode == 0 else "failed",
        "build_exit_code": completed.returncode,
        "build_process_started": True,
        "registry_mutation_possible": True,
        "live_machine_capability_status": live["status"],
    }
    write_json(output / "local_cpu_build_allocator_result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("cpu-build")
    _add_cpu_arguments(cpu, require_provider=False)
    cpu.add_argument(
        "--execution-plane", choices=("digitalocean", "local"), default="digitalocean"
    )
    cpu.add_argument("--mount-path")
    cpu.add_argument("--build-workdir")
    cpu.add_argument("--build-script")
    cpu_run = commands.add_parser("cpu-build-run", help=argparse.SUPPRESS)
    _add_cpu_arguments(cpu_run)
    cpu_local = commands.add_parser("cpu-build-local", help=argparse.SUPPRESS)
    cpu_local.add_argument("--output-dir", required=True)
    cpu_local.add_argument("--packet-manifest", required=True)
    cpu_local.add_argument("--builder-evidence", required=True)
    cpu_local.add_argument("--spend", required=True)
    cpu_local.add_argument("--mount-path", required=True)
    cpu_local.add_argument("--build-workdir", required=True)
    cpu_local.add_argument("--build-script", required=True)
    gpu = commands.add_parser("gpu-canary")
    gpu.add_argument("--provider-launch-request", required=True)
    gpu.add_argument("--release-evidence", required=True)
    gpu.add_argument("--model-cache-evidence", required=True)
    gpu.add_argument("--preflight-bundle", required=True)
    gpu.add_argument("--admission-out", required=True)
    gpu.add_argument("--bound-request-out", required=True)
    gpu.add_argument("--adapter-output", required=True)
    gpu.add_argument("--pod-name", required=True)
    gpu.add_argument("--execute", action="store_true")
    for name, hidden in (("model-volume", False), ("model-volume-run", True)):
        model = commands.add_parser(name, help=argparse.SUPPRESS if hidden else None)
        model.add_argument("--output-dir", required=True)
        model.add_argument("--release-image-ref", required=True)
        model.add_argument("--data-center-id", required=True)
        model.add_argument("--gpu-type-id", required=True)
        model.add_argument("--required-cuda-version", default="12.8")
        model.add_argument("--volume-size-gib", type=int, default=50)
        model.add_argument("--volume-hourly-rate-usd", type=float, required=True)
        model.add_argument("--hard-ttl-seconds", type=int, default=2700)
        model.add_argument("--max-spend-usd", type=float, default=0.40)
        model.add_argument("--hf-token-file", default="~/.blueprint-secrets/hf_token")
        model.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "cpu-build":
        if args.execution_plane == "local":
            missing = [
                name
                for name in ("mount_path", "build_workdir", "build_script")
                if not getattr(args, name, None)
            ]
            if missing:
                success = False
            else:
                prerequisite = _run_cpu_prerequisite_gate(Path(args.output_dir))
                if prerequisite.get("status") != "ready":
                    success = False
                else:
                    result = _run_local_cpu_build(args)
                    success = result.get("status") == "completed"
        elif missing := _missing_cpu_provider_arguments(args):
            success = False
        else:
            prerequisite = _run_cpu_prerequisite_gate(Path(args.output_dir))
            if prerequisite.get("status") != "ready":
                result = {
                    "status": "blocked_before_supervisor",
                    "blockers": prerequisite.get(
                        "blockers", ["groot_oscar_live_prerequisites_not_ready"]
                    ),
                    "provider_mutation_attempted": False,
                }
            else:
                result = launch_detached_builder(
                    output_dir=Path(args.output_dir), run_arguments=_cpu_vector(args)
                )
            success = result.get("status") == "supervisor_started"
    elif args.command == "cpu-build-run":
        result = _run_cpu(args)
        success = result.get("status") == "completed"
    elif args.command == "cpu-build-local":
        result = _run_local_cpu_build(args)
        success = result.get("status") == "completed"
    elif args.command == "gpu-canary":
        result = run_canary(
            provider_launch_request=args.provider_launch_request,
            release_evidence=args.release_evidence,
            model_cache_evidence=args.model_cache_evidence,
            preflight_bundle=args.preflight_bundle,
            admission_out=args.admission_out,
            bound_request_out=args.bound_request_out,
            adapter_output=args.adapter_output,
            pod_name=args.pod_name,
            execute=args.execute,
        )
        success = result.get("status") in {"dry_run_ready", "submitted"}
    elif args.command == "model-volume":
        vector = [
            "--output-dir", args.output_dir,
            "--release-image-ref", args.release_image_ref,
            "--data-center-id", args.data_center_id,
            "--gpu-type-id", args.gpu_type_id,
            "--required-cuda-version", args.required_cuda_version,
            "--volume-size-gib", str(args.volume_size_gib),
            "--volume-hourly-rate-usd", str(args.volume_hourly_rate_usd),
            "--hard-ttl-seconds", str(args.hard_ttl_seconds),
            "--max-spend-usd", str(args.max_spend_usd),
            "--hf-token-file", args.hf_token_file,
        ]
        if args.allow_paid:
            vector.append("--allow-paid")
        result = launch_detached_model_volume(
            output_dir=Path(args.output_dir), run_arguments=vector
        )
        success = result.get("status") == "supervisor_started"
    else:
        result = run_model_volume(
            output_dir=Path(args.output_dir),
            release_image_ref=args.release_image_ref,
            data_center_id=args.data_center_id,
            gpu_type_id=args.gpu_type_id,
            required_cuda_version=args.required_cuda_version,
            volume_size_gib=args.volume_size_gib,
            volume_hourly_rate_usd=args.volume_hourly_rate_usd,
            hard_ttl_seconds=args.hard_ttl_seconds,
            max_spend_usd=args.max_spend_usd,
            hf_token_file=Path(args.hf_token_file),
            allow_paid=args.allow_paid,
        )
        success = result.get("status") == "completed"
    # Provider results can contain credential-bearing request fields. Persisted
    # evidence remains in the explicitly selected output paths; stdout exposes
    # only the derived success bit so CI and operators cannot leak the payload.
    print(json.dumps({"success": success}, sort_keys=True))
    return 0 if success else 2


def cpu_build_main(argv: Sequence[str] | None = None) -> int:
    return main(["cpu-build", *(list(argv) if argv is not None else sys.argv[1:])])


def gpu_canary_main(argv: Sequence[str] | None = None) -> int:
    return main(["gpu-canary", *(list(argv) if argv is not None else sys.argv[1:])])


def model_volume_main(argv: Sequence[str] | None = None) -> int:
    return main(["model-volume", *(list(argv) if argv is not None else sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
