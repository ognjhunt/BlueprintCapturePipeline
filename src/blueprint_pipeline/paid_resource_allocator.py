"""Canonical paid allocators for CPU image builds and GPU canaries.

No other public command may allocate these resource classes.  Provider-specific
modules are adapters behind this interface and their mutation CLIs are disabled.
"""

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
from .groot_oscar_digitalocean_builder import (
    DETACHED_CPU_BUILD_SUPERVISOR_ENV,
    launch_detached_builder,
    observe_local_machine,
    run_builder,
)
from .groot_oscar_digitalocean_prebaked_host import (
    DEFAULT_REGION as DIGITALOCEAN_PREBAKE_DEFAULT_REGION,
    DEFAULT_SIZE as DIGITALOCEAN_PREBAKE_DEFAULT_SIZE,
    DEFAULT_SOURCE_IMAGE as DIGITALOCEAN_PREBAKE_DEFAULT_SOURCE_IMAGE,
    DEFAULT_VOLUME_GIB as DIGITALOCEAN_PREBAKE_DEFAULT_VOLUME_GIB,
    run_prebake as run_digitalocean_prebake,
)
from .groot_oscar_infrastructure_admission import (
    BUILD_SCHEMA_VERSION,
    build_build_plane_admission,
    build_cpu_build_execution_admission,
)
from .groot_oscar_runpod_canary import run_canary
from .groot_oscar_runpod_persistent_carrier import (
    PERSISTENT_CARRIER_PROBE_KIND,
    PERSISTENT_WATCHDOG_MAX_TTL_SECONDS,
)
from .groot_oscar_runpod_persistent_carrier_campaign import (
    run_persistent_carrier_campaign,
)
from .groot_oscar_runpod_serverless import (
    DEFAULT_MAX_HOURLY_RATE_USD,
    DEFAULT_RESERVATION_SECONDS,
    run_active_worker,
)
from .groot_oscar_runpod_storage_volume import (
    launch_detached as launch_detached_model_volume,
)
from .groot_oscar_runpod_storage_volume import (
    retain_verified_model_cache,
    run_storage_model_volume,
)
from .g1_microwave_finetune_provider_job import (
    PROBE_KIND as G1_MICROWAVE_FINETUNE_PROBE_KIND,
    run_finetune_job as run_g1_microwave_finetune_job,
)
from .openai_candidate_paid_admission import (
    OPENAI_API_CANDIDATE_ADMISSION_SCHEMA_VERSION,
    OPENAI_API_CANDIDATE_RESOURCE_CLASS,
    prepare_openai_api_candidate_admission,
    prepare_pigey_candidate_runtime_admission,
)
from .paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    require_paid_resource_admission,
)
from .openpi_policy_ranking_gpu_admission import (
    CURRENT_REFERENCE_POLICY_CANARY_PROBE_KIND,
    NEW_SITE_CANARY_PROBE_KIND,
    PROBE_KIND as OPENPI_POLICY_RANKING_PROBE_KIND,
)
from .openpi_policy_ranking_runpod import run_openpi_policy_ranking_campaign
from .nvidia_warehouse_native_camera_gpu_admission import (
    PROBE_KIND as NVIDIA_WAREHOUSE_NATIVE_CAMERA_PROBE_KIND,
    run_native_camera_gpu_lane,
)
from .policy_ranking_successor_gpu_admission import (
    CTRL_WORLD_CURRENT_REFERENCE_PROBE_KIND,
    PROBE_KIND as POLICY_RANKING_SUCCESSOR_COSMOS_PROBE_KIND,
    run_successor_gpu_lane,
)
from .policy_ranking_cosmos_reasoner_gpu_admission import (
    PROBE_KIND as POLICY_RANKING_COSMOS_REASONER_PROBE_KIND,
    run_gpu_lane as run_cosmos_reasoner_gpu_lane,
)
from .policy_ranking_successor_retained_session import refresh_retained_session
from .single_g1_kitchen_episode_runpod import (
    PROBE_KIND as SINGLE_KITCHEN_EPISODE_PROBE_KIND,
    run_single_episode,
)
from .single_g1_kitchen_qualification_session import (
    COMPONENT_ALIASES as QUALIFICATION_COMPONENT_ALIASES,
    PROBE_KIND as SINGLE_KITCHEN_QUALIFICATION_PROBE_KIND,
    SESSION_ACTIONS as QUALIFICATION_SESSION_ACTIONS,
    run_qualification_session,
)


ROOT = Path(__file__).resolve().parents[2]
CPU_BUILD_PREREQUISITE_EVIDENCE = "groot_oscar_live_prerequisites.json"
MIN_RECONCILED_CAMPAIGN_SPEND_USD = 14.557003
MIN_RECONCILED_GPU_SECONDS = 15_624
GPU_CANARY_RESERVATION_SECONDS = 1_200
STRICT_POLICY_SMOKE_RESERVATION_SECONDS = 480
FUTURE_CAMPAIGN_ALLOWANCE_SECONDS = 3_500
COMBINED_GPU_PLAN_SECONDS = GPU_CANARY_RESERVATION_SECONDS + FUTURE_CAMPAIGN_ALLOWANCE_SECONDS
PERSISTENT_CAMPAIGN_WALL_CAP_SECONDS = 36_000
DETACHED_MODEL_VOLUME_SUPERVISOR_ENV = "BLUEPRINT_DETACHED_MODEL_VOLUME_SUPERVISOR"
AdmissionResult = tuple[dict[str, Any], PaidResourceAdmissionGrant | None]


def admit_openai_api_candidate(**kwargs: Any) -> AdmissionResult:
    """Canonical source-bound issuer for one paid OpenAI candidate capability."""
    admission = prepare_openai_api_candidate_admission(
        source_checkout_validator=_source_checkout_blockers,
        checkout_state_reader=_current_checkout_source_state,
        **kwargs,
    )
    if kwargs.get("execute") is not True:
        return admission, None
    grant = require_paid_resource_admission(
        admission,
        resource_class=OPENAI_API_CANDIDATE_RESOURCE_CLASS,
        expected_schema_version=OPENAI_API_CANDIDATE_ADMISSION_SCHEMA_VERSION,
    )
    return admission, grant


def admit_pigey_candidate_runtime(*, runtime: Any, **kwargs: Any) -> AdmissionResult:
    try:
        admission = prepare_pigey_candidate_runtime_admission(
            runtime=runtime,
            source_checkout_validator=_source_checkout_blockers,
            checkout_state_reader=_current_checkout_source_state,
            **kwargs,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        blockers = ["openai_candidate_runtime_execution_profile_invalid"]
        raise PaidResourceAdmissionBlocked(blockers) from exc
    if kwargs.get("execute") is not True:
        return admission, None
    grant = require_paid_resource_admission(
        admission,
        resource_class=OPENAI_API_CANDIDATE_RESOURCE_CLASS,
        expected_schema_version=OPENAI_API_CANDIDATE_ADMISSION_SCHEMA_VERSION,
    )
    runtime.paid_resource_admission_grant = grant
    return admission, grant


def _current_checkout_source_state() -> tuple[str, bool]:
    try:
        commit_result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--verify", "HEAD^{commit}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status_result = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "", False
    commit = commit_result.stdout.strip().lower()
    commit_valid = bool(
        commit_result.returncode == 0
        and len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit)
    )
    clean = bool(status_result.returncode == 0 and not status_result.stdout.strip())
    return (commit if commit_valid else ""), clean


def _current_origin_main_commit() -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "rev-parse",
                "--verify",
                "origin/main^{commit}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    commit = result.stdout.strip().lower()
    if (
        result.returncode != 0
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        return ""
    return commit


def _current_remote_main_commit() -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "ls-remote",
                "--exit-code",
                "origin",
                "refs/heads/main",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    fields = result.stdout.strip().split()
    commit = fields[0].lower() if len(fields) == 2 else ""
    if (
        result.returncode != 0
        or fields[1:] != ["refs/heads/main"]
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        return ""
    return commit


def _current_branch_name() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "branch", "--show-current"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _current_remote_branch_commit(branch: str) -> str:
    if not branch or not branch.startswith("codex/"):
        return ""
    reference = f"refs/heads/{branch}"
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-remote", "--exit-code", "origin", reference],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    fields = result.stdout.strip().split()
    commit = fields[0].lower() if len(fields) == 2 else ""
    if (
        result.returncode != 0
        or fields[1:] != [reference]
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        return ""
    return commit


def _source_checkout_blockers(
    expected_source_commit: str, *, allow_pushed_branch_diagnostic: bool = False
) -> tuple[list[str], str]:
    checkout_commit, checkout_clean = _current_checkout_source_state()
    blockers: list[str] = []
    if not checkout_commit:
        blockers.append("gpu_canary_checkout_source_commit_unavailable")
    elif expected_source_commit.strip().lower() != checkout_commit:
        blockers.append("gpu_canary_expected_source_commit_not_current_checkout")
    if allow_pushed_branch_diagnostic:
        branch = _current_branch_name()
        remote_branch_commit = _current_remote_branch_commit(branch)
        if not branch.startswith("codex/"):
            blockers.append("gpu_canary_experimental_branch_not_codex_namespaced")
        if not remote_branch_commit:
            blockers.append("gpu_canary_remote_experimental_branch_commit_unavailable")
        elif checkout_commit != remote_branch_commit:
            blockers.append("gpu_canary_checkout_not_pushed_experimental_branch")
    else:
        origin_main_commit = _current_origin_main_commit()
        remote_main_commit = _current_remote_main_commit()
        if not origin_main_commit:
            blockers.append("gpu_canary_origin_main_commit_unavailable")
        elif checkout_commit != origin_main_commit:
            blockers.append("gpu_canary_checkout_not_origin_main")
        if not remote_main_commit:
            blockers.append("gpu_canary_remote_main_commit_unavailable")
        elif checkout_commit != remote_main_commit:
            blockers.append("gpu_canary_checkout_not_remote_main")
    if not checkout_clean:
        blockers.append("gpu_canary_checkout_not_clean")
    return blockers, checkout_commit


def _control_plane_checkout_blockers() -> tuple[list[str], dict[str, object]]:
    """Pin the clean allocator identity without coupling it to the runtime image.

    Main pointers remain recorded diagnostics.  A clean detached commit or
    reviewed branch is a valid orchestrator identity and does not force an
    immutable runtime-image rebuild.
    """

    checkout_commit, checkout_clean = _current_checkout_source_state()
    origin_main_commit = _current_origin_main_commit()
    remote_main_commit = _current_remote_main_commit()
    blockers: list[str] = []
    if not checkout_commit:
        blockers.append("gpu_canary_orchestrator_source_commit_unavailable")
    if not checkout_clean:
        blockers.append("gpu_canary_orchestrator_checkout_not_clean")
    identity: dict[str, object] = {
        "schema_version": "blueprint.gpu_canary_control_plane_identity.v1",
        "orchestrator_source_commit": checkout_commit or None,
        "checkout_clean": checkout_clean,
        "origin_main_commit": origin_main_commit or None,
        "remote_main_commit": remote_main_commit or None,
        "orchestrator_equals_origin_main": bool(
            checkout_commit and checkout_commit == origin_main_commit
        ),
        "orchestrator_equals_remote_main": bool(
            checkout_commit and checkout_commit == remote_main_commit
        ),
        "main_parity_is_diagnostic_not_runtime_identity": True,
        "raw_secret_values_recorded": False,
    }
    return blockers, identity


def _write_blocked_qualification_allocation_outputs(args: argparse.Namespace, result: dict) -> None:
    """Persist every promised allocation output when source admission blocks."""

    for attribute in (
        "provider_launch_request",
        "preflight_bundle",
        "admission_out",
        "bound_request_out",
        "adapter_output",
    ):
        value = getattr(args, attribute, None)
        if value:
            write_json(Path(value), result)


def _configure_detached_supervisor_signal_policy(command: str) -> bool:
    """Keep an explicitly detached paid supervisor alive through local SIGINT.

    SIGTERM remains available for an intentional stop. Provider resources also
    remain bounded by their independent deadline watchdogs.
    """

    detached_model_volume = (
        command == "model-volume-run" and os.getenv(DETACHED_MODEL_VOLUME_SUPERVISOR_ENV) == "1"
    )
    detached_cpu_build = (
        command == "cpu-build-run" and os.getenv(DETACHED_CPU_BUILD_SUPERVISOR_ENV) == "1"
    )
    if not (detached_model_volume or detached_cpu_build):
        return False
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, OSError, ValueError):
        return False
    return True


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
            "blockers": prerequisite.get("blockers", ["groot_oscar_live_prerequisites_not_ready"]),
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
            "blockers": ["groot_oscar_live_prerequisite_verifier_failed_without_evidence"],
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
    cpu.add_argument("--execution-plane", choices=("digitalocean", "local"), default="digitalocean")
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
    gpu.add_argument("--expected-source-commit")
    gpu.add_argument(
        "--experimental-branch-diagnostic",
        action="store_true",
        help=(
            "Allow a clean exact commit on its pushed codex/ branch for diagnostic canaries. "
            "Release and production evidence still require exact main."
        ),
    )
    gpu.add_argument(
        "--expected-image-source-commit",
        help=(
            "Exact source commit represented by the runtime image. "
            "--expected-source-commit remains a compatibility alias."
        ),
    )
    gpu.add_argument(
        "--provider",
        choices=("runpod", "vast", "digitalocean"),
        default="runpod",
    )
    gpu.add_argument(
        "--probe-kind",
        choices=(
            "startup",
            "strict-policy-smoke",
            "persistent-host-bake",
            PERSISTENT_CARRIER_PROBE_KIND,
            SINGLE_KITCHEN_EPISODE_PROBE_KIND,
            SINGLE_KITCHEN_QUALIFICATION_PROBE_KIND,
            G1_MICROWAVE_FINETUNE_PROBE_KIND,
            OPENPI_POLICY_RANKING_PROBE_KIND,
            NEW_SITE_CANARY_PROBE_KIND,
            CURRENT_REFERENCE_POLICY_CANARY_PROBE_KIND,
            NVIDIA_WAREHOUSE_NATIVE_CAMERA_PROBE_KIND,
            POLICY_RANKING_SUCCESSOR_COSMOS_PROBE_KIND,
            CTRL_WORLD_CURRENT_REFERENCE_PROBE_KIND,
            POLICY_RANKING_COSMOS_REASONER_PROBE_KIND,
        ),
        default="strict-policy-smoke",
    )
    gpu.add_argument("--carrier-volume-admission")
    gpu.add_argument("--policy-observation")
    gpu.add_argument("--persistent-job-dir")
    gpu.add_argument("--task-prompt")
    gpu.add_argument("--episode-bundle")
    gpu.add_argument("--provider-bundle-url-file")
    gpu.add_argument("--provider-output-put-url-file")
    gpu.add_argument("--provider-output-get-url-file")
    gpu.add_argument("--provider-bootstrap-url-file")
    gpu.add_argument("--finetune-provider-bundle")
    gpu.add_argument("--openpi-input-bundle-receipt")
    gpu.add_argument("--native-camera-input-bundle-receipt")
    gpu.add_argument("--openpi-input-secret-url-file")
    gpu.add_argument("--openpi-output-secret-put-url-file")
    gpu.add_argument("--openpi-output-secret-get-url-file")
    gpu.add_argument(
        "--openpi-provider",
        choices=("vast", "runpod"),
        default="vast",
        help="Policy-ranking GPU provider; Vast is the frozen default.",
    )
    gpu.add_argument("--openpi-hard-ttl-seconds", type=int, default=14_400)
    gpu.add_argument("--openpi-max-spend-usd", type=float, default=3.0)
    gpu.add_argument("--successor-public-base-url")
    gpu.add_argument("--successor-token-file")
    gpu.add_argument("--successor-secret-env-file")
    gpu.add_argument("--successor-output-path")
    gpu.add_argument("--successor-session-budget-ledger")
    gpu.add_argument("--successor-bundle-receipt")
    gpu.add_argument(
        "--successor-profile-freeze",
        help=(
            "Tracked exact-HEAD GPU profile freeze for a request-bound successor bundle. "
            "Required for Blueprint Ctrl-World current-reference execution."
        ),
    )
    gpu.add_argument("--reasoner-bundle-receipt")
    gpu.add_argument("--reasoner-public-base-url")
    gpu.add_argument("--reasoner-token-file")
    gpu.add_argument("--reasoner-secret-env-file")
    gpu.add_argument("--reasoner-output-path")
    gpu.add_argument("--reasoner-session-budget-ledger")
    gpu.add_argument(
        "--successor-action",
        choices=("launch", "refresh"),
        default="launch",
    )
    gpu.add_argument("--successor-session-manifest")
    gpu.add_argument(
        "--successor-dirty-state-declaration",
        choices=("clean_exact_commit", "declared_dirty_overlay"),
        default="clean_exact_commit",
    )
    gpu.add_argument("--finetune-object-store-stage-dir")
    gpu.add_argument("--finetune-checkpoint-object-store-stage-dir")
    gpu.add_argument(
        "--finetune-checkpoint-part-stage-dir",
        action="append",
        default=[],
    )
    gpu.add_argument("--finetune-checkpoint-vast-session-manifest")
    gpu.add_argument(
        "--qualification-action",
        choices=QUALIFICATION_SESSION_ACTIONS,
    )
    gpu.add_argument("--qualification-session-manifest")
    gpu.add_argument("--qualification-training-dataset")
    gpu.add_argument("--qualification-trained-checkpoint")
    gpu.add_argument("--qualification-checkpoint-report")
    gpu.add_argument(
        "--qualification-checkpoint-part-stage-dir",
        action="append",
        default=[],
    )
    gpu.add_argument(
        "--qualification-component",
        choices=tuple(QUALIFICATION_COMPONENT_ALIASES),
        default="episode",
    )
    gpu.add_argument("--qualification-tail-lines", type=int, default=200)
    gpu.add_argument("--qualification-watchdog-extension-seconds", type=int)
    gpu.add_argument("--qualification-watchdog-extension-spend-cap-usd", type=float)
    gpu.add_argument(
        "--qualification-identity-file",
        default="~/.ssh/id_ed25519",
    )
    gpu.add_argument("--execute", action="store_true")
    gpu.add_argument("--campaign-budget-ledger")
    gpu.add_argument("--campaign-initial-spent-usd", type=float)
    gpu.add_argument("--campaign-initial-used-gpu-seconds", type=int)
    gpu.add_argument("--campaign-total-spend-cap-usd", type=float, default=20.0)
    gpu.add_argument("--campaign-wall-cap-seconds", type=int)
    gpu.add_argument(
        "--campaign-reservation-seconds",
        type=int,
        default=None,
    )
    gpu.add_argument(
        "--future-campaign-allowance-seconds",
        type=int,
        default=FUTURE_CAMPAIGN_ALLOWANCE_SECONDS,
    )
    gpu.add_argument("--authorize-reduced-canary-timeout", action="store_true")
    gpu.add_argument("--authorize-persistent-carrier-campaign", action="store_true")
    gpu.add_argument("--campaign-max-hourly-rate-usd", type=float)
    gpu.add_argument(
        "--digitalocean-token-file",
        default="~/.blueprint-secrets/digitalocean_api_token",
    )
    gpu.add_argument("--docker-username-file", default="~/.blueprint-secrets/docker_username")
    gpu.add_argument("--docker-password-file", default="~/.blueprint-secrets/docker_pat")
    gpu.add_argument(
        "--runpod-s3-access-key-file",
        default="~/.blueprint-secrets/runpod_s3_access_key",
    )
    gpu.add_argument(
        "--runpod-s3-secret-key-file",
        default="~/.blueprint-secrets/runpod_s3_secret_key",
    )
    gpu.add_argument("--login-private-key")
    gpu.add_argument("--host-private-key")
    gpu.add_argument("--ssh-key-id", type=int)
    gpu.add_argument("--digitalocean-region", default=DIGITALOCEAN_PREBAKE_DEFAULT_REGION)
    gpu.add_argument("--digitalocean-size", default=DIGITALOCEAN_PREBAKE_DEFAULT_SIZE)
    gpu.add_argument(
        "--digitalocean-source-image",
        default=DIGITALOCEAN_PREBAKE_DEFAULT_SOURCE_IMAGE,
    )
    gpu.add_argument(
        "--digitalocean-volume-size-gib",
        type=int,
        default=DIGITALOCEAN_PREBAKE_DEFAULT_VOLUME_GIB,
    )
    warm = commands.add_parser("gpu-warm-worker")
    warm.add_argument("--output-dir", required=True)
    warm.add_argument("--release-evidence", required=True)
    warm.add_argument("--model-cache-evidence", required=True)
    warm.add_argument("--watchdog-handoff-evidence", required=True)
    warm.add_argument(
        "--runpod-api-key-file",
        default="~/.blueprint-secrets/runpod_api_key",
    )
    warm.add_argument("--resource-name-prefix", required=True)
    warm.add_argument("--expected-source-commit", required=True)
    warm.add_argument("--campaign-budget-ledger", required=True)
    warm.add_argument("--campaign-initial-spent-usd", required=True, type=float)
    warm.add_argument("--campaign-initial-used-gpu-seconds", required=True, type=int)
    warm.add_argument(
        "--campaign-reservation-seconds",
        type=int,
        default=DEFAULT_RESERVATION_SECONDS,
    )
    warm.add_argument(
        "--campaign-max-hourly-rate-usd",
        type=float,
        default=DEFAULT_MAX_HOURLY_RATE_USD,
    )
    warm.add_argument("--campaign-io-evidence", required=True)
    warm.add_argument("--carrier-volume-admission")
    warm.add_argument(
        "--gpu-type-id",
        action="append",
        dest="gpu_type_ids",
        default=None,
    )
    warm.add_argument(
        "--runpod-s3-access-key-file",
        default="~/.blueprint-secrets/runpod_s3_access_key",
    )
    warm.add_argument(
        "--runpod-s3-secret-key-file",
        default="~/.blueprint-secrets/runpod_s3_secret_key",
    )
    warm.add_argument("--execute", action="store_true")
    for name, hidden in (("model-volume", False), ("model-volume-run", True)):
        model = commands.add_parser(name, help=argparse.SUPPRESS if hidden else None)
        model.add_argument("--output-dir", required=True)
        model.add_argument("--repo-root", default=str(ROOT))
        model.add_argument("--data-center-id")
        model.add_argument("--volume-size-gib", type=int, default=50)
        model.add_argument("--runtime-source-release-image-ref", default="")
        model.add_argument("--runtime-source-release-evidence")
        model.add_argument("--carrier-image-ref", default="")
        model.add_argument("--replacement-source-output")
        model.add_argument("--storage-hourly-rate-usd", type=float, required=True)
        model.add_argument("--storage-ttl-seconds", type=int, default=14_400)
        model.add_argument("--max-storage-spend-usd", type=float, default=0.05)
        model.add_argument("--builder-evidence")
        model.add_argument("--builder-spend")
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
        model.add_argument("--login-private-key")
        model.add_argument("--host-private-key")
        model.add_argument("--ssh-key-id", type=int)
        model.add_argument("--region", default="sfo3")
        model.add_argument("--allow-paid", action="store_true")
        model.add_argument("--retain-existing-output")
        model.add_argument("--retention-ttl-seconds", type=int, default=7 * 24 * 60 * 60)
        model.add_argument("--retention-max-spend-usd", type=float, default=1.0)
        model.add_argument("--campaign-spent-to-date-usd", type=float)
        model.add_argument("--campaign-total-spend-cap-usd", type=float, default=20.0)
    args = parser.parse_args(argv)
    _configure_detached_supervisor_signal_policy(args.command)
    if args.command in {"model-volume", "model-volume-run"} and not (
        args.command == "model-volume" and args.retain_existing_output
    ):
        missing_model_volume_inputs = [
            name
            for name in (
                "data_center_id",
                "builder_evidence",
                "builder_spend",
                "login_private_key",
                "host_private_key",
                "ssh_key_id",
            )
            if getattr(args, name, None) in {None, ""}
        ]
        if missing_model_volume_inputs:
            print(json.dumps({"success": False}, sort_keys=True))
            return 2
        runtime_bundle_requested = bool(
            args.runtime_source_release_image_ref or args.carrier_image_ref
        )
        if runtime_bundle_requested and not args.runtime_source_release_evidence:
            print(json.dumps({"success": False}, sort_keys=True))
            return 2
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
        if args.probe_kind == NVIDIA_WAREHOUSE_NATIVE_CAMERA_PROBE_KIND:
            missing = [
                name
                for name in (
                    "release_evidence",
                    "native_camera_input_bundle_receipt",
                    "preflight_bundle",
                    "admission_out",
                    "bound_request_out",
                    "adapter_output",
                    "pod_name",
                    "expected_source_commit",
                )
                if not getattr(args, name, None)
            ]
            if args.provider != "vast":
                missing.append("provider_must_be_vast")
            if args.execute:
                missing.extend(
                    name
                    for name in (
                        "provider_bundle_url_file",
                        "provider_output_put_url_file",
                        "provider_output_get_url_file",
                        "campaign_budget_ledger",
                        "campaign_initial_spent_usd",
                        "campaign_initial_used_gpu_seconds",
                    )
                    if getattr(args, name, None) is None
                )
            if missing:
                result = {
                    "status": "blocked",
                    "blockers": [
                        "native_camera_gpu_required_arguments_missing:"
                        + ",".join(sorted(set(missing)))
                    ],
                    "provider_mutations_performed": 0,
                }
                write_json(Path(args.admission_out), result)
            else:
                source_blockers, checkout_commit = _source_checkout_blockers(
                    args.expected_source_commit or "",
                    allow_pushed_branch_diagnostic=args.experimental_branch_diagnostic,
                )
                if source_blockers:
                    result = {
                        "status": "blocked",
                        "blockers": source_blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.admission_out), result)
                else:
                    result = run_native_camera_gpu_lane(
                        release_evidence=args.release_evidence,
                        input_bundle_receipt=args.native_camera_input_bundle_receipt,
                        preflight_bundle=args.preflight_bundle,
                        admission_out=args.admission_out,
                        bound_request_out=args.bound_request_out,
                        adapter_output=args.adapter_output,
                        pod_name=args.pod_name,
                        expected_source_commit=checkout_commit,
                        execute=args.execute,
                        hard_ttl_seconds=args.openpi_hard_ttl_seconds,
                        max_spend_usd=args.openpi_max_spend_usd,
                        input_secret_url_file=args.provider_bundle_url_file,
                        output_secret_put_url_file=args.provider_output_put_url_file,
                        output_secret_get_url_file=args.provider_output_get_url_file,
                        campaign_budget_ledger=args.campaign_budget_ledger,
                        campaign_initial_spent_usd=args.campaign_initial_spent_usd,
                        campaign_initial_used_gpu_seconds=(args.campaign_initial_used_gpu_seconds),
                        campaign_total_spend_cap_usd=(args.campaign_total_spend_cap_usd),
                        campaign_wall_cap_seconds=(
                            args.campaign_wall_cap_seconds
                            if args.campaign_wall_cap_seconds is not None
                            else 36_000
                        ),
                        provider_name=args.provider,
                    )
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == POLICY_RANKING_COSMOS_REASONER_PROBE_KIND:
            required = (
                "provider_launch_request",
                "preflight_bundle",
                "episode_bundle",
                "reasoner_bundle_receipt",
                "admission_out",
                "bound_request_out",
                "adapter_output",
                "pod_name",
                "expected_source_commit",
            )
            missing = [name for name in required if not getattr(args, name, None)]
            if args.execute and not args.reasoner_session_budget_ledger:
                missing.append("reasoner_session_budget_ledger")
            if (
                args.execute
                and not args.reasoner_public_base_url
                and not all(
                    getattr(args, name, None)
                    for name in (
                        "provider_bundle_url_file",
                        "provider_output_put_url_file",
                        "provider_output_get_url_file",
                    )
                )
            ):
                missing.append("reasoner_staging_transport")
            if missing:
                result = {
                    "status": "blocked",
                    "blockers": [
                        "policy_ranking_cosmos_reasoner_required_arguments_missing:"
                        + ",".join(sorted(set(missing)))
                    ],
                    "provider_mutations_performed": 0,
                }
                write_json(Path(args.admission_out), result)
            else:
                source_blockers, checkout_commit = _source_checkout_blockers(
                    args.expected_source_commit or "",
                    allow_pushed_branch_diagnostic=args.experimental_branch_diagnostic,
                )
                if source_blockers:
                    result = {
                        "status": "blocked",
                        "blockers": source_blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.admission_out), result)
                else:
                    result = run_cosmos_reasoner_gpu_lane(
                        authorization_path=args.provider_launch_request,
                        preflight_path=args.preflight_bundle,
                        bundle_path=args.episode_bundle,
                        bundle_receipt_path=args.reasoner_bundle_receipt,
                        admission_out=args.admission_out,
                        bound_request_out=args.bound_request_out,
                        adapter_output=args.adapter_output,
                        job_dir=args.pod_name,
                        expected_source_commit=checkout_commit,
                        execute=args.execute,
                        public_base_url=args.reasoner_public_base_url,
                        token_file=args.reasoner_token_file,
                        secret_env_file=args.reasoner_secret_env_file,
                        provider_bundle_url_file=args.provider_bundle_url_file,
                        provider_output_put_url_file=args.provider_output_put_url_file,
                        provider_output_get_url_file=args.provider_output_get_url_file,
                        output_path=args.reasoner_output_path,
                        session_budget_ledger=args.reasoner_session_budget_ledger,
                    )
            success = result.get("status") in {
                "dry_run_ready",
                "completed",
                "retained_owned",
            }
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind in {
            POLICY_RANKING_SUCCESSOR_COSMOS_PROBE_KIND,
            CTRL_WORLD_CURRENT_REFERENCE_PROBE_KIND,
        }:
            if args.successor_action == "refresh":
                missing = [
                    name
                    for name in (
                        "provider_launch_request",
                        "episode_bundle",
                        "successor_public_base_url",
                        "successor_token_file",
                        "successor_session_manifest",
                        "adapter_output",
                        "expected_source_commit",
                    )
                    if not getattr(args, name, None)
                ]
                control_blockers, control_identity = _control_plane_checkout_blockers()
                if (
                    args.expected_source_commit
                    and args.expected_source_commit.strip().lower()
                    != control_identity.get("orchestrator_source_commit")
                ):
                    control_blockers.append(
                        "gpu_canary_expected_source_commit_not_current_checkout"
                    )
                if missing or control_blockers or not args.execute:
                    result = {
                        "status": "blocked",
                        "blockers": [
                            *control_blockers,
                            *(
                                [
                                    "policy_ranking_successor_refresh_required_arguments_missing:"
                                    + ",".join(sorted(set(missing)))
                                ]
                                if missing
                                else []
                            ),
                            *(
                                ["policy_ranking_successor_refresh_execute_required"]
                                if not args.execute
                                else []
                            ),
                        ],
                        "provider_mutations_performed": 0,
                    }
                else:
                    authorization_path = Path(args.provider_launch_request).expanduser().resolve()
                    authorization_receipt_sha256 = hashlib.sha256(
                        authorization_path.read_bytes()
                    ).hexdigest()
                    result = refresh_retained_session(
                        session_manifest=args.successor_session_manifest,
                        bundle_path=args.episode_bundle,
                        public_base_url=args.successor_public_base_url,
                        token_file=args.successor_token_file,
                        source_commit=str(control_identity["orchestrator_source_commit"]),
                        dirty_state_declaration=args.successor_dirty_state_declaration,
                        authorization_receipt_sha256=authorization_receipt_sha256,
                        identity_file=args.qualification_identity_file,
                    )
                write_json(Path(args.adapter_output), result)
                success = result.get("status") == "provider_absent"
                print(json.dumps({"success": success}, sort_keys=True))
                return 0 if success else 2
            missing = [
                name
                for name in (
                    "provider_launch_request",
                    "release_evidence",
                    "model_cache_evidence",
                    "preflight_bundle",
                    "episode_bundle",
                    "successor_bundle_receipt",
                    "admission_out",
                    "bound_request_out",
                    "adapter_output",
                    "pod_name",
                    "expected_source_commit",
                )
                if not getattr(args, name, None)
            ]
            if args.execute:
                if not args.successor_public_base_url and not all(
                    getattr(args, name, None)
                    for name in (
                        "provider_bundle_url_file",
                        "provider_output_put_url_file",
                        "provider_output_get_url_file",
                    )
                ):
                    missing.append("successor_staging_transport")
                if not args.successor_session_budget_ledger:
                    missing.append("successor_session_budget_ledger")
            if missing:
                result = {
                    "status": "blocked",
                    "blockers": [
                        "policy_ranking_successor_required_arguments_missing:"
                        + ",".join(sorted(set(missing)))
                    ],
                    "provider_mutations_performed": 0,
                }
                write_json(Path(args.admission_out), result)
            else:
                source_blockers, checkout_commit = _source_checkout_blockers(
                    args.expected_source_commit or "",
                    allow_pushed_branch_diagnostic=args.experimental_branch_diagnostic,
                )
                if source_blockers:
                    result = {
                        "status": "blocked",
                        "blockers": source_blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.admission_out), result)
                else:
                    result = run_successor_gpu_lane(
                        authorization_path=args.provider_launch_request,
                        environment_path=args.release_evidence,
                        smoke_inventory_path=args.model_cache_evidence,
                        provider_preflight_path=args.preflight_bundle,
                        provider_bundle_path=args.episode_bundle,
                        provider_bundle_receipt_path=args.successor_bundle_receipt,
                        admission_out=args.admission_out,
                        bound_request_out=args.bound_request_out,
                        adapter_output=args.adapter_output,
                        job_dir=args.pod_name,
                        public_base_url=args.successor_public_base_url,
                        token_file=args.successor_token_file,
                        secret_env_file=args.successor_secret_env_file,
                        provider_bundle_url_file=args.provider_bundle_url_file,
                        provider_output_put_url_file=args.provider_output_put_url_file,
                        provider_output_get_url_file=args.provider_output_get_url_file,
                        output_path=args.successor_output_path,
                        session_budget_ledger=args.successor_session_budget_ledger,
                        expected_source_commit=checkout_commit,
                        execute=args.execute,
                        expected_probe_kind=args.probe_kind,
                        current_reference_profile_freeze_path=args.successor_profile_freeze,
                    )
            success = result.get("status") in {
                "dry_run_ready",
                "completed",
                "retained_owned",
            }
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind in {
            OPENPI_POLICY_RANKING_PROBE_KIND,
            NEW_SITE_CANARY_PROBE_KIND,
            CURRENT_REFERENCE_POLICY_CANARY_PROBE_KIND,
        }:
            missing = [
                name
                for name in (
                    "openpi_input_bundle_receipt",
                    "openpi_input_secret_url_file",
                    "openpi_output_secret_put_url_file",
                    "expected_source_commit",
                )
                if not getattr(args, name, None)
            ]
            if args.execute:
                missing.extend(
                    name
                    for name in (
                        "campaign_budget_ledger",
                        "campaign_initial_spent_usd",
                        "campaign_initial_used_gpu_seconds",
                        "openpi_output_secret_get_url_file",
                    )
                    if getattr(args, name, None) is None
                )
            if missing:
                result = {
                    "status": "blocked",
                    "blockers": [
                        "openpi_policy_ranking_required_arguments_missing:"
                        + ",".join(sorted(set(missing)))
                    ],
                    "provider_mutations_performed": 0,
                }
                write_json(Path(args.admission_out), result)
            else:
                source_blockers, checkout_commit = _source_checkout_blockers(
                    args.expected_source_commit or "",
                    allow_pushed_branch_diagnostic=args.experimental_branch_diagnostic,
                )
                if source_blockers:
                    result = {
                        "status": "blocked",
                        "blockers": source_blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.admission_out), result)
                else:
                    result = run_openpi_policy_ranking_campaign(
                        release_evidence=args.release_evidence,
                        input_bundle_receipt=args.openpi_input_bundle_receipt,
                        preflight_bundle=args.preflight_bundle,
                        admission_out=args.admission_out,
                        bound_request_out=args.bound_request_out,
                        adapter_output=args.adapter_output,
                        input_secret_url_file=args.openpi_input_secret_url_file,
                        output_secret_put_url_file=(args.openpi_output_secret_put_url_file),
                        pod_name=args.pod_name,
                        expected_source_commit=checkout_commit,
                        execute=args.execute,
                        hard_ttl_seconds=args.openpi_hard_ttl_seconds,
                        max_spend_usd=args.openpi_max_spend_usd,
                        campaign_budget_ledger=args.campaign_budget_ledger,
                        campaign_initial_spent_usd=(args.campaign_initial_spent_usd),
                        campaign_initial_used_gpu_seconds=(args.campaign_initial_used_gpu_seconds),
                        campaign_total_spend_cap_usd=(args.campaign_total_spend_cap_usd),
                        campaign_wall_cap_seconds=(
                            args.campaign_wall_cap_seconds
                            if args.campaign_wall_cap_seconds is not None
                            else 36_000
                        ),
                        output_secret_get_url_file=(args.openpi_output_secret_get_url_file),
                        provider_name=args.openpi_provider,
                        compute_authorization_path=args.provider_launch_request,
                    )
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == G1_MICROWAVE_FINETUNE_PROBE_KIND:
            missing = [
                name
                for name in (
                    "finetune_provider_bundle",
                    "finetune_object_store_stage_dir",
                    "finetune_checkpoint_object_store_stage_dir",
                )
                if not getattr(args, name, None)
            ]
            if args.provider not in {"runpod", "vast"}:
                missing.append("provider_must_be_runpod_or_vast")
            if missing:
                result = {
                    "status": "blocked",
                    "blockers": [
                        "g1_microwave_finetune_required_arguments_missing:"
                        + ",".join(sorted(set(missing)))
                    ],
                    "provider_mutations_performed": 0,
                }
                write_json(Path(args.adapter_output), result)
            else:
                result = run_g1_microwave_finetune_job(
                    provider_name=args.provider,
                    provider_bundle=args.finetune_provider_bundle,
                    object_store_stage_dir=args.finetune_object_store_stage_dir,
                    checkpoint_object_store_stage_dir=(
                        args.finetune_checkpoint_object_store_stage_dir
                    ),
                    checkpoint_object_store_part_stage_dirs=(
                        args.finetune_checkpoint_part_stage_dir
                    ),
                    release_evidence=args.release_evidence,
                    provider_launch_request=args.provider_launch_request,
                    preflight_bundle=args.preflight_bundle,
                    admission_out=args.admission_out,
                    bound_request_out=args.bound_request_out,
                    adapter_output=args.adapter_output,
                    pod_name=args.pod_name,
                    execute=args.execute,
                    checkpoint_vast_session_manifest=(
                        args.finetune_checkpoint_vast_session_manifest
                    ),
                    qualification_identity_file=args.qualification_identity_file,
                )
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == SINGLE_KITCHEN_QUALIFICATION_PROBE_KIND:
            missing = [
                name
                for name in (
                    "qualification_action",
                    "qualification_session_manifest",
                )
                if not getattr(args, name, None)
            ]
            if args.provider != "vast":
                missing.append("provider_must_be_vast")
            if args.qualification_action == "allocate":
                expected_image_source_commit = (
                    args.expected_image_source_commit or args.expected_source_commit
                )
                missing.extend(
                    name
                    for name in (
                        "episode_bundle",
                        "provider_bundle_url_file",
                        "provider_output_put_url_file",
                        "provider_output_get_url_file",
                        "expected_image_source_commit",
                    )
                    if not (
                        expected_image_source_commit
                        if name == "expected_image_source_commit"
                        else getattr(args, name, None)
                    )
                )
                if (
                    args.expected_image_source_commit
                    and args.expected_source_commit
                    and args.expected_image_source_commit != args.expected_source_commit
                ):
                    missing.append("conflicting_image_source_commit_arguments")
            if args.qualification_action == "refresh-bootstrap" and not getattr(
                args, "episode_bundle", None
            ):
                missing.append("episode_bundle")
            if args.qualification_action == "restart-component" and (
                args.qualification_component not in {"groot", "controller", "isaac", "bridge"}
            ):
                missing.append("restartable_qualification_component")
            if args.qualification_action == "stop-component" and (
                args.qualification_component
                not in {
                    "episode",
                    "groot",
                    "controller",
                    "isaac",
                    "bridge",
                    "finetune",
                }
            ):
                missing.append("stoppable_qualification_component")
            if args.qualification_action == "extend-watchdog":
                missing.extend(
                    name
                    for name in (
                        "qualification_watchdog_extension_seconds",
                        "qualification_watchdog_extension_spend_cap_usd",
                    )
                    if getattr(args, name, None) is None
                )
            if missing:
                result = {
                    "status": "blocked",
                    "blockers": [
                        "single_kitchen_qualification_required_arguments_missing:"
                        + ",".join(sorted(set(missing)))
                    ],
                    "provider_mutations_performed": 0,
                }
                if args.qualification_action == "allocate":
                    _write_blocked_qualification_allocation_outputs(args, result)
                else:
                    write_json(Path(args.adapter_output), result)
            else:
                if args.qualification_action == "allocate":
                    source_blockers, control_plane_identity = _control_plane_checkout_blockers()
                    if source_blockers:
                        result = {
                            "status": "blocked",
                            "blockers": source_blockers,
                            "control_plane_identity": control_plane_identity,
                            "provider_mutations_performed": 0,
                        }
                        _write_blocked_qualification_allocation_outputs(args, result)
                        print(json.dumps({"success": False}, sort_keys=True))
                        return 2
                try:
                    result = run_qualification_session(
                        action=args.qualification_action,
                        session_manifest=args.qualification_session_manifest,
                        provider_name=args.provider,
                        component=args.qualification_component,
                        tail_lines=args.qualification_tail_lines,
                        identity_file=args.qualification_identity_file,
                        episode_bundle=args.episode_bundle,
                        training_dataset=args.qualification_training_dataset,
                        trained_checkpoint_path=args.qualification_trained_checkpoint,
                        provider_bundle_url_file=args.provider_bundle_url_file,
                        provider_output_put_url_file=args.provider_output_put_url_file,
                        provider_output_get_url_file=args.provider_output_get_url_file,
                        provider_bootstrap_url_file=args.provider_bootstrap_url_file,
                        release_evidence=args.release_evidence,
                        model_cache_evidence=args.model_cache_evidence,
                        expected_source_commit=(
                            args.expected_image_source_commit or args.expected_source_commit
                        ),
                        orchestrator_source_commit=(
                            str(control_plane_identity["orchestrator_source_commit"])
                            if args.qualification_action == "allocate"
                            else None
                        ),
                        provider_launch_request=args.provider_launch_request,
                        preflight_bundle=args.preflight_bundle,
                        admission_out=args.admission_out,
                        bound_request_out=args.bound_request_out,
                        adapter_output=args.adapter_output,
                        pod_name=args.pod_name,
                        watchdog_extension_seconds=(args.qualification_watchdog_extension_seconds),
                        watchdog_extension_spend_cap_usd=(
                            args.qualification_watchdog_extension_spend_cap_usd
                        ),
                        execute=args.execute,
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    result = {
                        "status": "blocked",
                        "blockers": [str(exc)],
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.adapter_output), result)
            success = result.get("status") in {
                "bootstrap_staging_required",
                "refresh_bootstrap_staging_required_continuing_spend",
                "dry_run_bound",
                "dry_run_ready",
                "dry_run_refresh_bound_continuing_spend",
                "allocated_ready_continuing_spend",
                "bootstrap_refreshed_continuing_spend",
                "episode_dispatched_continuing_spend",
                "episode_snapshot_collected_continuing_spend",
                "episode_collected_passed_continuing_spend",
                "episode_collected_blocked_continuing_spend",
                "status_observed_continuing_spend",
                "tail_collected_continuing_spend",
                "gpu_status_collected_continuing_spend",
                "component_restarted_continuing_spend",
                "component_stopped_continuing_spend",
                "watchdog_extended_continuing_spend",
                "teardown_completed_provider_zero",
            }
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == SINGLE_KITCHEN_EPISODE_PROBE_KIND:
            missing = [
                name
                for name in (
                    "episode_bundle",
                    "provider_bundle_url_file",
                    "provider_output_put_url_file",
                    "provider_output_get_url_file",
                )
                if not getattr(args, name, None)
            ]
            if args.provider not in {"runpod", "vast"}:
                missing.append("provider_must_be_runpod_or_vast")
            if missing:
                result = {
                    "status": "blocked",
                    "blockers": [
                        "single_kitchen_episode_required_arguments_missing:"
                        + ",".join(sorted(missing))
                    ],
                    "provider_mutations_performed": 0,
                }
                write_json(Path(args.adapter_output), result)
            else:
                result = run_single_episode(
                    provider_name=args.provider,
                    episode_bundle=args.episode_bundle,
                    provider_bundle_url_file=args.provider_bundle_url_file,
                    provider_output_put_url_file=args.provider_output_put_url_file,
                    provider_output_get_url_file=args.provider_output_get_url_file,
                    provider_bootstrap_url_file=args.provider_bootstrap_url_file,
                    release_evidence=args.release_evidence,
                    provider_launch_request=args.provider_launch_request,
                    preflight_bundle=args.preflight_bundle,
                    admission_out=args.admission_out,
                    bound_request_out=args.bound_request_out,
                    adapter_output=args.adapter_output,
                    pod_name=args.pod_name,
                    execute=args.execute,
                    qualification_checkpoint_report=(args.qualification_checkpoint_report),
                    qualification_checkpoint_part_stage_dirs=tuple(
                        args.qualification_checkpoint_part_stage_dir
                    ),
                )
            success = result.get("status") in {
                "bootstrap_staging_required",
                "dry_run_ready",
                "completed",
            }
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == "persistent-host-bake":
            missing = [
                name
                for name in (
                    "campaign_budget_ledger",
                    "campaign_initial_spent_usd",
                    "campaign_initial_used_gpu_seconds",
                    "campaign_reservation_seconds",
                    "campaign_max_hourly_rate_usd",
                    "login_private_key",
                    "host_private_key",
                    "ssh_key_id",
                )
                if getattr(args, name, None) in {None, ""}
            ]
            if args.provider != "digitalocean":
                missing.append("provider_must_be_digitalocean")
            if missing:
                result = {
                    "status": "blocked",
                    "blockers": [
                        "digitalocean_prebake_required_arguments_missing:"
                        + ",".join(sorted(missing))
                    ],
                    "provider_mutations_performed": 0,
                }
                write_json(Path(args.adapter_output), result)
            else:
                output_dir = Path(args.adapter_output).expanduser().resolve().parent
                result = run_digitalocean_prebake(
                    output_dir=output_dir,
                    release_evidence_path=Path(args.release_evidence),
                    model_cache_evidence_path=Path(args.model_cache_evidence),
                    token_file=Path(args.digitalocean_token_file),
                    docker_username_file=Path(args.docker_username_file),
                    docker_password_file=Path(args.docker_password_file),
                    runpod_s3_access_key_file=Path(args.runpod_s3_access_key_file),
                    runpod_s3_secret_key_file=Path(args.runpod_s3_secret_key_file),
                    login_private_key=Path(args.login_private_key),
                    host_private_key=Path(args.host_private_key),
                    ssh_key_id=args.ssh_key_id,
                    region=args.digitalocean_region,
                    size=args.digitalocean_size,
                    source_image=args.digitalocean_source_image,
                    volume_size_gib=args.digitalocean_volume_size_gib,
                    reservation_seconds=args.campaign_reservation_seconds,
                    future_gpu_seconds=args.future_campaign_allowance_seconds,
                    campaign_budget_ledger=Path(args.campaign_budget_ledger),
                    initial_spent_usd=args.campaign_initial_spent_usd,
                    initial_gpu_seconds=args.campaign_initial_used_gpu_seconds,
                    total_spend_cap_usd=args.campaign_total_spend_cap_usd,
                    gpu_wall_cap_seconds=(
                        args.campaign_wall_cap_seconds
                        if args.campaign_wall_cap_seconds is not None
                        else 21_000
                    ),
                    max_hourly_rate_usd=args.campaign_max_hourly_rate_usd,
                    execute=args.execute,
                )
                generated = {
                    Path(args.preflight_bundle): output_dir / "digitalocean_prebake_preflight.json",
                    Path(args.admission_out): output_dir / "digitalocean_prebake_admission.json",
                    Path(args.bound_request_out): output_dir / "bound_provider_request.json",
                    Path(args.adapter_output): output_dir
                    / "digitalocean_prebaked_host_result.json",
                }
                for destination, source in generated.items():
                    if destination.expanduser().resolve() == source.resolve():
                        continue
                    if source.is_file():
                        write_json(
                            destination,
                            json.loads(source.read_text(encoding="utf-8")),
                        )
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.provider != "runpod":
            result = {
                "status": "blocked",
                "blockers": ["digitalocean_gpu_canary_requires_persistent_host_bake"],
                "provider_mutations_performed": 0,
            }
            write_json(Path(args.adapter_output), result)
            print(json.dumps({"success": False}, sort_keys=True))
            return 2
        if args.probe_kind == PERSISTENT_CARRIER_PROBE_KIND:
            missing = [
                name
                for name in (
                    "carrier_volume_admission",
                    "policy_observation",
                    "persistent_job_dir",
                    "expected_source_commit",
                )
                if not getattr(args, name, None)
            ]
            budget_arguments = (
                args.campaign_budget_ledger,
                args.campaign_initial_spent_usd,
                args.campaign_initial_used_gpu_seconds,
                args.campaign_max_hourly_rate_usd,
            )
            if args.execute and any(value is None for value in budget_arguments):
                missing.append("persistent_campaign_budget_arguments")
            if args.execute and not args.authorize_persistent_carrier_campaign:
                missing.append("persistent_campaign_authorization")
            checkout_commit = ""
            if not missing:
                source_blockers, checkout_commit = _source_checkout_blockers(
                    args.expected_source_commit or "",
                    allow_pushed_branch_diagnostic=args.experimental_branch_diagnostic,
                )
                if source_blockers:
                    result = {
                        "status": "blocked",
                        "blockers": source_blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.admission_out), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
            if missing:
                result = {
                    "status": "blocked",
                    "blockers": [
                        "persistent_carrier_required_arguments_missing:" + ",".join(sorted(missing))
                    ],
                    "provider_mutations_performed": 0,
                }
                write_json(Path(args.admission_out), result)
            else:
                result = run_persistent_carrier_campaign(
                    provider_launch_request=args.provider_launch_request,
                    release_evidence=args.release_evidence,
                    model_cache_evidence=args.model_cache_evidence,
                    preflight_bundle=args.preflight_bundle,
                    carrier_volume_admission=args.carrier_volume_admission,
                    policy_observation_path=args.policy_observation,
                    persistent_job_dir=args.persistent_job_dir,
                    admission_out=args.admission_out,
                    bound_request_out=args.bound_request_out,
                    adapter_output=args.adapter_output,
                    pod_name=args.pod_name,
                    execute=args.execute,
                    expected_source_commit=checkout_commit,
                    task_prompt=args.task_prompt,
                    campaign_budget=(
                        {
                            "ledger_path": args.campaign_budget_ledger,
                            "initial_spent_usd": args.campaign_initial_spent_usd,
                            "initial_used_gpu_seconds": (args.campaign_initial_used_gpu_seconds),
                            "total_spend_cap_usd": args.campaign_total_spend_cap_usd,
                            "combined_gpu_wall_cap_seconds": (
                                args.campaign_wall_cap_seconds
                                if args.campaign_wall_cap_seconds is not None
                                else PERSISTENT_CAMPAIGN_WALL_CAP_SECONDS
                            ),
                            "reservation_gpu_seconds": (PERSISTENT_WATCHDOG_MAX_TTL_SECONDS),
                            "campaign_stage": "persistent_carrier_campaign",
                            "maximum_canary_reservation_gpu_seconds": (
                                PERSISTENT_WATCHDOG_MAX_TTL_SECONDS
                            ),
                            "future_campaign_allowance_gpu_seconds": 0,
                            "maximum_future_campaign_allowance_gpu_seconds": 0,
                            "maximum_combined_plan_gpu_seconds": (
                                PERSISTENT_WATCHDOG_MAX_TTL_SECONDS
                            ),
                            "reduced_canary_timeout_acknowledged": (
                                args.authorize_persistent_carrier_campaign
                            ),
                            "max_hourly_rate_usd": args.campaign_max_hourly_rate_usd,
                            "minimum_reconciled_spend_usd": (MIN_RECONCILED_CAMPAIGN_SPEND_USD),
                            "minimum_reconciled_gpu_seconds": (MIN_RECONCILED_GPU_SECONDS),
                        }
                        if args.execute
                        else None
                    ),
                )
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        generic_canary_missing = [
            name
            for name in (
                "expected_source_commit",
                "provider_output_put_url_file",
            )
            if not getattr(args, name, None)
        ]
        if generic_canary_missing:
            result = {
                "status": "blocked",
                "blockers": [
                    "gpu_canary_required_arguments_missing:"
                    + ",".join(sorted(generic_canary_missing))
                ],
                "provider_mutations_performed": 0,
            }
            write_json(Path(args.admission_out), result)
            print(json.dumps({"success": False}, sort_keys=True))
            return 2
        source_blockers, checkout_commit = _source_checkout_blockers(
            args.expected_source_commit or "",
            allow_pushed_branch_diagnostic=args.experimental_branch_diagnostic,
        )
        if source_blockers:
            result = {
                "status": "blocked",
                "blockers": source_blockers,
                "provider_mutations_performed": 0,
            }
            write_json(Path(args.admission_out), result)
            print(json.dumps({"success": False}, sort_keys=True))
            return 2
        strict_policy_smoke = args.probe_kind == "strict-policy-smoke"
        maximum_canary_reservation_seconds = (
            STRICT_POLICY_SMOKE_RESERVATION_SECONDS
            if strict_policy_smoke
            else GPU_CANARY_RESERVATION_SECONDS
        )
        reservation_seconds = (
            args.campaign_reservation_seconds
            if args.campaign_reservation_seconds is not None
            else maximum_canary_reservation_seconds
        )
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
                expected_source_commit=checkout_commit,
                provider_output_put_url_file=args.provider_output_put_url_file,
                probe_kind=args.probe_kind,
                campaign_budget=(
                    {
                        "ledger_path": args.campaign_budget_ledger,
                        "initial_spent_usd": args.campaign_initial_spent_usd,
                        "initial_used_gpu_seconds": args.campaign_initial_used_gpu_seconds,
                        "total_spend_cap_usd": args.campaign_total_spend_cap_usd,
                        "combined_gpu_wall_cap_seconds": (
                            args.campaign_wall_cap_seconds
                            if args.campaign_wall_cap_seconds is not None
                            else 21_000
                        ),
                        "reservation_gpu_seconds": reservation_seconds,
                        "campaign_stage": "gpu_canary",
                        "maximum_canary_reservation_gpu_seconds": (
                            maximum_canary_reservation_seconds
                        ),
                        "future_campaign_allowance_gpu_seconds": (
                            args.future_campaign_allowance_seconds
                        ),
                        "maximum_future_campaign_allowance_gpu_seconds": (
                            FUTURE_CAMPAIGN_ALLOWANCE_SECONDS
                        ),
                        "maximum_combined_plan_gpu_seconds": (
                            STRICT_POLICY_SMOKE_RESERVATION_SECONDS
                            + FUTURE_CAMPAIGN_ALLOWANCE_SECONDS
                            if strict_policy_smoke
                            else COMBINED_GPU_PLAN_SECONDS
                        ),
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
    elif args.command == "gpu-warm-worker":
        result = run_active_worker(
            output_dir=args.output_dir,
            release_evidence=args.release_evidence,
            model_cache_evidence=args.model_cache_evidence,
            watchdog_handoff_evidence=args.watchdog_handoff_evidence,
            api_key_file=args.runpod_api_key_file,
            campaign_io_evidence=args.campaign_io_evidence,
            runpod_s3_access_key_file=args.runpod_s3_access_key_file,
            runpod_s3_secret_key_file=args.runpod_s3_secret_key_file,
            resource_name_prefix=args.resource_name_prefix,
            expected_source_commit=args.expected_source_commit,
            execute=args.execute,
            campaign_budget_ledger=args.campaign_budget_ledger,
            initial_spent_usd=args.campaign_initial_spent_usd,
            initial_gpu_seconds=args.campaign_initial_used_gpu_seconds,
            reservation_seconds=args.campaign_reservation_seconds,
            max_hourly_rate_usd=args.campaign_max_hourly_rate_usd,
            carrier_volume_admission=args.carrier_volume_admission,
            gpu_type_ids=(
                tuple(args.gpu_type_ids) if args.gpu_type_ids else ("NVIDIA A40", "NVIDIA L40S")
            ),
        )
        success = result.get("status") in {
            "dry_run_ready",
            "completed",
        }
    elif args.command == "model-volume":
        if args.retain_existing_output:
            if args.campaign_spent_to_date_usd is None:
                result = {
                    "status": "blocked",
                    "blockers": ["bounded_cache_retention_campaign_spend_missing"],
                }
            else:
                result = retain_verified_model_cache(
                    output_dir=Path(args.output_dir),
                    source_output_dir=Path(args.retain_existing_output),
                    retention_ttl_seconds=args.retention_ttl_seconds,
                    storage_hourly_rate_usd=args.storage_hourly_rate_usd,
                    max_retention_spend_usd=args.retention_max_spend_usd,
                    campaign_spent_to_date_usd=args.campaign_spent_to_date_usd,
                    campaign_total_spend_cap_usd=args.campaign_total_spend_cap_usd,
                    runpod_s3_access_key_file=Path(args.runpod_s3_access_key_file),
                    runpod_s3_secret_key_file=Path(args.runpod_s3_secret_key_file),
                    allow_paid=args.allow_paid,
                )
            success = result.get("status") == "retained"
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        vector = [
            "--output-dir",
            args.output_dir,
            "--repo-root",
            args.repo_root,
            "--data-center-id",
            args.data_center_id,
            "--volume-size-gib",
            str(args.volume_size_gib),
            "--storage-hourly-rate-usd",
            str(args.storage_hourly_rate_usd),
            "--storage-ttl-seconds",
            str(args.storage_ttl_seconds),
            "--max-storage-spend-usd",
            str(args.max_storage_spend_usd),
            "--builder-evidence",
            args.builder_evidence,
            "--builder-spend",
            args.builder_spend,
            "--digitalocean-token-file",
            args.digitalocean_token_file,
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
            vector.append("--allow-paid")
        if args.runtime_source_release_image_ref:
            vector.extend(
                [
                    "--runtime-source-release-image-ref",
                    args.runtime_source_release_image_ref,
                ]
            )
        if args.runtime_source_release_evidence:
            vector.extend(
                [
                    "--runtime-source-release-evidence",
                    args.runtime_source_release_evidence,
                ]
            )
        if args.carrier_image_ref:
            vector.extend(["--carrier-image-ref", args.carrier_image_ref])
        if args.replacement_source_output:
            vector.extend(["--replacement-source-output", args.replacement_source_output])
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
            runtime_source_release_image_ref=args.runtime_source_release_image_ref,
            carrier_image_ref=args.carrier_image_ref,
            runtime_source_release_evidence_path=(
                Path(args.runtime_source_release_evidence)
                if args.runtime_source_release_evidence
                else None
            ),
            replacement_source_output_dir=(
                Path(args.replacement_source_output) if args.replacement_source_output else None
            ),
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
