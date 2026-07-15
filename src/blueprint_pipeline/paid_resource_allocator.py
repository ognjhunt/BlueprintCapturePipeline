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
from .groot_oscar_runpod_storage_volume import (
    launch_detached as launch_detached_model_volume,
)
from .groot_oscar_runpod_storage_volume import run_storage_model_volume
from .paid_resource_admission import require_paid_resource_admission


ROOT = Path(__file__).resolve().parents[2]
CPU_BUILD_PREREQUISITE_EVIDENCE = "groot_oscar_live_prerequisites.json"
MIN_RECONCILED_CAMPAIGN_SPEND_USD = 11.57
MIN_RECONCILED_GPU_SECONDS = 11_619
GPU_CANARY_RESERVATION_SECONDS = 1_200
FUTURE_CAMPAIGN_ALLOWANCE_SECONDS = 3_900
COMBINED_GPU_PLAN_SECONDS = (
    GPU_CANARY_RESERVATION_SECONDS + FUTURE_CAMPAIGN_ALLOWANCE_SECONDS
)


def _add_cpu_arguments(parser: argparse.ArgumentParser, *, require_provider: bool = True) -> None:
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
        hf_token_file=Path(args.hf_token_file),
        runpod_s3_access_key_file=Path(args.runpod_s3_access_key_file),
        runpod_s3_secret_key_file=Path(args.runpod_s3_secret_key_file),
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
    gpu.add_argument("--campaign-budget-ledger")
    gpu.add_argument("--campaign-initial-spent-usd", type=float)
    gpu.add_argument("--campaign-initial-used-gpu-seconds", type=int)
    gpu.add_argument("--campaign-total-spend-cap-usd", type=float, default=20.0)
    gpu.add_argument("--campaign-wall-cap-seconds", type=int, default=16_800)
    gpu.add_argument(
        "--campaign-reservation-seconds",
        type=int,
        default=GPU_CANARY_RESERVATION_SECONDS,
    )
    gpu.add_argument(
        "--future-campaign-allowance-seconds",
        type=int,
        default=FUTURE_CAMPAIGN_ALLOWANCE_SECONDS,
    )
    gpu.add_argument("--authorize-reduced-canary-timeout", action="store_true")
    gpu.add_argument("--campaign-max-hourly-rate-usd", type=float)
    for name, hidden in (("model-volume", False), ("model-volume-run", True)):
        model = commands.add_parser(name, help=argparse.SUPPRESS if hidden else None)
        model.add_argument("--output-dir", required=True)
        model.add_argument("--repo-root", default=str(ROOT))
        model.add_argument("--data-center-id", required=True)
        model.add_argument("--volume-size-gib", type=int, default=50)
        model.add_argument("--storage-hourly-rate-usd", type=float, required=True)
        model.add_argument("--storage-ttl-seconds", type=int, default=14_400)
        model.add_argument("--max-storage-spend-usd", type=float, default=0.05)
        model.add_argument("--builder-evidence", required=True)
        model.add_argument("--builder-spend", required=True)
        model.add_argument(
            "--digitalocean-token-file",
            default="~/.blueprint-secrets/digitalocean_api_token",
        )
        model.add_argument("--hf-token-file", default="~/.blueprint-secrets/hf_token")
        model.add_argument(
            "--runpod-s3-access-key-file",
            default="~/.blueprint-secrets/runpod_s3_access_key",
        )
        model.add_argument(
            "--runpod-s3-secret-key-file",
            default="~/.blueprint-secrets/runpod_s3_secret_key",
        )
        model.add_argument("--login-private-key", required=True)
        model.add_argument("--host-private-key", required=True)
        model.add_argument("--ssh-key-id", required=True, type=int)
        model.add_argument("--region", default="sfo3")
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
        budget_arguments = (
            args.campaign_budget_ledger,
            args.campaign_initial_spent_usd,
            args.campaign_initial_used_gpu_seconds,
            args.campaign_max_hourly_rate_usd,
        )
        if args.execute and any(value is None for value in budget_arguments):
            result = {
                "status": "blocked",
                "blockers": ["gpu_canary_cumulative_budget_arguments_missing"],
                "provider_mutations_performed": 0,
            }
            write_json(Path(args.admission_out), result)
        else:
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
                campaign_budget=(
                    {
                        "ledger_path": args.campaign_budget_ledger,
                        "initial_spent_usd": args.campaign_initial_spent_usd,
                        "initial_used_gpu_seconds": args.campaign_initial_used_gpu_seconds,
                        "total_spend_cap_usd": args.campaign_total_spend_cap_usd,
                        "combined_gpu_wall_cap_seconds": args.campaign_wall_cap_seconds,
                        "reservation_gpu_seconds": args.campaign_reservation_seconds,
                        "campaign_stage": "gpu_canary",
                        "maximum_canary_reservation_gpu_seconds": GPU_CANARY_RESERVATION_SECONDS,
                        "future_campaign_allowance_gpu_seconds": (
                            args.future_campaign_allowance_seconds
                        ),
                        "maximum_future_campaign_allowance_gpu_seconds": (
                            FUTURE_CAMPAIGN_ALLOWANCE_SECONDS
                        ),
                        "maximum_combined_plan_gpu_seconds": COMBINED_GPU_PLAN_SECONDS,
                        "reduced_canary_timeout_acknowledged": (
                            args.authorize_reduced_canary_timeout
                        ),
                        "max_hourly_rate_usd": args.campaign_max_hourly_rate_usd,
                        "minimum_reconciled_spend_usd": MIN_RECONCILED_CAMPAIGN_SPEND_USD,
                        "minimum_reconciled_gpu_seconds": MIN_RECONCILED_GPU_SECONDS,
                    }
                    if args.execute
                    else None
                ),
            )
        success = result.get("status") in {"dry_run_ready", "submitted"}
    elif args.command == "model-volume":
        vector = [
            "--output-dir", args.output_dir,
            "--repo-root", args.repo_root,
            "--data-center-id", args.data_center_id,
            "--volume-size-gib", str(args.volume_size_gib),
            "--storage-hourly-rate-usd", str(args.storage_hourly_rate_usd),
            "--storage-ttl-seconds", str(args.storage_ttl_seconds),
            "--max-storage-spend-usd", str(args.max_storage_spend_usd),
            "--builder-evidence", args.builder_evidence,
            "--builder-spend", args.builder_spend,
            "--digitalocean-token-file", args.digitalocean_token_file,
            "--hf-token-file", args.hf_token_file,
            "--runpod-s3-access-key-file", args.runpod_s3_access_key_file,
            "--runpod-s3-secret-key-file", args.runpod_s3_secret_key_file,
            "--login-private-key", args.login_private_key,
            "--host-private-key", args.host_private_key,
            "--ssh-key-id", str(args.ssh_key_id),
            "--region", args.region,
        ]
        if args.allow_paid:
            vector.append("--allow-paid")
        result = launch_detached_model_volume(
            output_dir=Path(args.output_dir), run_arguments=vector
        )
        success = result.get("status") == "supervisor_started"
    else:
        result = run_storage_model_volume(
            output_dir=Path(args.output_dir),
            repo_root=Path(args.repo_root),
            data_center_id=args.data_center_id,
            volume_size_gib=args.volume_size_gib,
            storage_ttl_seconds=args.storage_ttl_seconds,
            storage_hourly_rate_usd=args.storage_hourly_rate_usd,
            max_storage_spend_usd=args.max_storage_spend_usd,
            builder_evidence_path=Path(args.builder_evidence),
            builder_spend_path=Path(args.builder_spend),
            digitalocean_token_file=Path(args.digitalocean_token_file),
            hf_token_file=Path(args.hf_token_file),
            runpod_s3_access_key_file=Path(args.runpod_s3_access_key_file),
            runpod_s3_secret_key_file=Path(args.runpod_s3_secret_key_file),
            login_private_key=Path(args.login_private_key),
            host_private_key=Path(args.host_private_key),
            ssh_key_id=args.ssh_key_id,
            region=args.region,
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
