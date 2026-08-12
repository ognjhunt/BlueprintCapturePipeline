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
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .decision_evidence_contracts import canonical_digest
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
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from .paid_resource_cli_arguments import add_cpu_arguments as _add_cpu_arguments
from .hosted_model_inference_preflight import (
    BACKENDS as HOSTED_MODEL_BACKENDS,
    PROBE_PROFILE as HOSTED_MODEL_PROBE_PROFILE,
    REQUIRED_CAPABILITIES as HOSTED_MODEL_REQUIRED_CAPABILITIES,
    SCHEMA_VERSION as HOSTED_MODEL_PREFLIGHT_SCHEMA_VERSION,
)
from .openpi_policy_ranking_gpu_admission import (
    NEW_SITE_CANARY_PROBE_KIND,
    PROBE_KIND as OPENPI_POLICY_RANKING_PROBE_KIND,
)
from .openpi_policy_ranking_runpod import run_openpi_policy_ranking_campaign
from .nvidia_warehouse_native_camera_gpu_admission import (
    PROBE_KIND as NVIDIA_WAREHOUSE_NATIVE_CAMERA_PROBE_KIND,
    run_native_camera_gpu_lane,
)
from .policy_ranking_successor_gpu_admission import (
    PROBE_KIND as POLICY_RANKING_SUCCESSOR_COSMOS_PROBE_KIND,
    run_successor_gpu_lane,
)
from .policy_ranking_cosmos_reasoner_gpu_admission import (
    PROBE_KIND as POLICY_RANKING_COSMOS_REASONER_PROBE_KIND,
    run_gpu_lane as run_cosmos_reasoner_gpu_lane,
)
from .policy_ranking_successor_retained_session import refresh_retained_session
from .reconstruction_gpu_admission import (
    PROBE_KIND as RECONSTRUCTION_WORKER_SMOKE_PROBE_KIND,
    collect_reconstruction_vast_preflight,
    prepare_reconstruction_gpu_canary,
    select_reconstruction_execution_adapter_id,
)
from .reconstruction_isaac_vast_operation import run_reconstruction_isaac_vast_operation
from . import measurement_dlo_lab_paid_allocator
from .reconstruction_paid_transport import prepare_reconstruction_paid_transport
from .reconstruction_vast_operation import run_reconstruction_vast_operation
from .reconstruction_vast_worker_smoke import run_reconstruction_vast_worker_smoke
from .gpu_render_providers import get_render_provider
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
from .simpler_public_vast import (
    PROBE_KIND as ADP_SIMPLER_PUBLIC_REFERENCE_PROBE_KIND,
    build_simpler_public_vast_bundle,
    run_simpler_public_vast,
)
from .adp_isaac_lab_arena_vast import (
    PROBE_KIND as ADP_ISAAC_LAB_ARENA_PROBE_KIND,
    build_arena_native_control_bundle,
    run_arena_native_control_vast,
)
from .adp009d_franka_vast import run_adp009d_native_microcheck_vast
from .native_task_arena_construction_bundle import (
    PROBE_KIND as NATIVE_TASK_ARENA_CONSTRUCTION_PROBE_KIND,
    build_native_task_arena_construction_bundle,
    load_verified_native_task_arena_construction_bundle,
)
from .native_task_arena_controls_bundle import (
    PROBE_KIND as NATIVE_TASK_ARENA_CONTROLS_PROBE_KIND,
    build_native_task_arena_controls_bundle,
    load_verified_native_task_arena_controls_bundle,
)
from .native_task_arena_policy_bundle import (
    PROBE_KIND as NATIVE_TASK_ARENA_POLICY_PROBE_KIND,
    build_native_task_arena_policy_bundle,
    load_verified_native_task_arena_policy_bundle,
)
from .native_task_arena_vast import (
    run_native_task_arena_controls_vast,
    run_native_task_arena_policy_vast,
    run_native_task_arena_vast,
)
from .native_task_runtime_source_packet import (
    verify_native_task_runtime_source_packet,
)
from .adp009d_gated_backbone import probe_gated_backbone_access
from .adp009d_native_microcheck_bundle import (
    PROBE_KIND as ADP009D_NATIVE_MICROCHECK_PROBE_KIND,
    build_native_microcheck_bundle_isolated as build_native_microcheck_bundle,
)
from .articulated_native_diagnostic_bundle import (
    build_articulated_native_diagnostic_bundle,
)
from .model_access_env import model_access_secret_status, normalize_model_access_env
from .adp009d_ovrtx_vast import (
    PROBE_KIND as ADP009D_OVRTX_LIVE_CAMERA_PROBE_KIND,
    build_ovrtx_live_camera_bundle,
    run_ovrtx_live_camera_vast,
)
from .adp009d_aura_native_vast import (
    PROBE_KIND as ADP009D_AURA_NATIVE_LIVE_CAMERA_PROBE_KIND,
    build_aura_native_live_camera_bundle,
    run_aura_native_live_camera_vast,
)
from .public_scene_simready_isaac_bundle import DEFAULT_IMAGE as ADP_SIMREADY_ISAAC_IMAGE
from .public_scene_simready_isaac_vast import (
    PROBE_KIND as ADP_SIMREADY_ISAAC_PROBE_KIND,
    run_simready_isaac_vast,
    validate_simready_isaac_paid_attempt_authority,
)
from .adp_content_agents_vast import (
    DEFAULT_IMAGE as ADP_CONTENT_AGENTS_IMAGE,
    PROBE_KIND as ADP_CONTENT_AGENTS_PROBE_KIND,
    SOURCE_COMMIT as ADP_CONTENT_AGENTS_SOURCE_COMMIT,
    SOURCE_TREE as ADP_CONTENT_AGENTS_SOURCE_TREE,
    consume_content_agents_paid_attempt_authority_once,
    run_content_agents_vast,
    validate_content_agents_paid_attempt_authority,
)
from .adp_content_agents_bundle_preflight import validate_bundle_config_preflight
from .adp_joint_agent_vast import (
    DEFAULT_IMAGE as ADP_JOINT_AGENT_IMAGE,
    PROBE_KIND as ADP_JOINT_AGENT_PROBE_KIND,
    PROVIDER_BUNDLE_KIND as ADP_JOINT_AGENT_PROVIDER_BUNDLE_KIND,
    SOURCE_TREE as ADP_JOINT_AGENT_SOURCE_TREE,
    run_joint_agent_vast,
)
from .adp_gaussian_excision_vast import (
    DEFAULT_IMAGE as ADP_GAUSSIAN_EXCISION_IMAGE,
    PROBE_KIND as ADP_GAUSSIAN_EXCISION_PROBE_KIND,
    PROVIDER_BUNDLE_KIND as ADP_GAUSSIAN_EXCISION_PROVIDER_BUNDLE_KIND,
    SOURCE_TREE as ADP_GAUSSIAN_EXCISION_SOURCE_TREE,
    run_gaussian_excision_vast,
    validate_gaussian_excision_paid_attempt_authority,
)
from .adp_retained_scene_render_vast import (
    PROBE_KIND as ADP_RETAINED_SCENE_RENDER_PROBE_KIND,
    run_retained_scene_render_vast,
    validate_retained_scene_render_bundle,
    validate_retained_scene_render_paid_attempt_authority,
)
from .adp_aura_author_smoke_vast import (
    DEFAULT_IMAGE as ADP_AURA_SMOKE_IMAGE,
    PREREQUISITE_RECEIPT_DIGEST as ADP_AURA_PREREQUISITE_RECEIPT_DIGEST,
    PROBE_KIND as ADP_AURA_SMOKE_PROBE_KIND,
    SOURCE_COMMIT as ADP_AURA_SOURCE_COMMIT,
    SOURCE_TREE as ADP_AURA_SOURCE_TREE,
    run_aura_author_smoke_vast,
)
from .adp_aura_interiorgs_vast import (
    AURA_RUNTIME_PREREQUISITE_RECEIPT_DIGEST as ADP_AURA_RUNTIME_PREREQUISITE_RECEIPT_DIGEST,
    PROBE_KIND as ADP_AURA_INTERIORGS_PROBE_KIND,
    run_aura_interiorgs_vast,
)
from .public_scene_aura_exact_residual_vast import (
    MAX_HARD_CAP_USD as AURA_EXACT_RESIDUAL_MAX_HARD_CAP_USD,
    MAX_TTL_SECONDS as AURA_EXACT_RESIDUAL_MAX_TTL_SECONDS,
    PROBE_KIND as ADP_AURA_EXACT_RESIDUAL_PROBE_KIND,
    run_aura_exact_residual_vast,
    validate_aura_exact_residual_paid_attempt_authority,
    validate_aura_exact_residual_bundle,
)
from .public_scene_artifixer3d_vast import (
    MAX_HARD_CAP_USD as ARTIFIXER3D_MAX_HARD_CAP_USD,
    MAX_TTL_SECONDS as ARTIFIXER3D_MAX_TTL_SECONDS,
    PROBE_KIND as ADP_ARTIFIXER3D_PROBE_KIND,
    run_artifixer3d_vast,
    validate_artifixer3d_bundle,
    validate_artifixer3d_paid_attempt_authority,
)
from .public_scene_execution_authority import (
    PublicSceneExecutionAuthorityError,
    validate_aura_retry_authority_binding,
)
from .adp_inpaint360_interiorgs_vast import (
    DEFAULT_IMAGE as ADP_INPAINT360_INTERIORGS_IMAGE,
    LAMA_SOURCE_COMMIT as ADP_INPAINT360_LAMA_SOURCE_COMMIT,
    LAMA_SOURCE_TREE as ADP_INPAINT360_LAMA_SOURCE_TREE,
    PREREQUISITE_RECEIPT_DIGEST as ADP_INPAINT360_PREREQUISITE_RECEIPT_DIGEST,
    PROBE_KIND as ADP_INPAINT360_INTERIORGS_PROBE_KIND,
    SOURCE_COMMIT as ADP_INPAINT360_SOURCE_COMMIT,
    SOURCE_TREE as ADP_INPAINT360_SOURCE_TREE,
    run_inpaint360_interiorgs_vast,
)
from .teleport_paid_allocator import (
    add_teleport_provider_arguments,
    load_teleport_credentials,
    run_teleport_provider,
)
from .task_evaluation_profile_preflight import (
    PROBE_KIND as TASK_EVALUATION_PROFILE_PREFLIGHT_PROBE_KIND,
    run_task_evaluation_profile_preflight,
)
from .task_evaluation_terminal_resource_release import dispatch_terminal_resource_release


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
DETACHED_GPU_CANARY_SUPERVISOR_ENV = "BLUEPRINT_DETACHED_GPU_CANARY_SUPERVISOR"
LAUNCH_DETACHED_GPU_CANARY_SUPERVISOR_DIR_ENV = (
    "BLUEPRINT_LAUNCH_DETACHED_GPU_CANARY_SUPERVISOR_DIR"
)
DETACHED_GPU_CANARY_MANIFEST = "detached_gpu_canary_supervisor.json"
DETACHED_GPU_CANARY_LOG = "detached_gpu_canary_supervisor.log"
DETACHED_GPU_CANARY_LOCK = "detached_gpu_canary_supervisor.lock"
TERMINAL_RESOURCE_RELEASE_WORKER_ENV = (
    "BLUEPRINT_TASK_EVALUATION_TERMINAL_RESOURCE_RELEASE_WORKER"
)
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


CHECKOUT_IDENTITY_PROBE_TIMEOUT_SECONDS = 20
CHECKOUT_IDENTITY_PROBE_ATTEMPTS = 3
CHECKOUT_IDENTITY_PROBE_BACKOFF_SECONDS = 2.0


def _checkout_git_command(*arguments: str) -> list[str]:
    """Trust only the physical immutable checkout used by this allocator."""

    return ["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT), *arguments]


def _run_checkout_probe(argv: Sequence[str]) -> subprocess.CompletedProcess | None:
    """Probe immutable checkout identity, retrying a transient Git lock."""

    result: subprocess.CompletedProcess | None = None
    for attempt in range(CHECKOUT_IDENTITY_PROBE_ATTEMPTS):
        try:
            result = subprocess.run(
                list(argv), check=False, capture_output=True, text=True,
                timeout=CHECKOUT_IDENTITY_PROBE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0:
            return result
        if attempt + 1 < CHECKOUT_IDENTITY_PROBE_ATTEMPTS:
            time.sleep(CHECKOUT_IDENTITY_PROBE_BACKOFF_SECONDS)
    return result


def _current_checkout_source_state() -> tuple[str, bool, bool]:
    """Separate a failed Git observation from a genuinely dirty checkout."""

    commit_result = _run_checkout_probe(
        _checkout_git_command("rev-parse", "--verify", "HEAD^{commit}")
    )
    status_result = _run_checkout_probe(
        _checkout_git_command("status", "--porcelain", "--untracked-files=no")
    )
    if commit_result is None or status_result is None:
        return "", False, False
    if commit_result.returncode != 0 or status_result.returncode != 0:
        return "", False, False
    commit = commit_result.stdout.strip().lower()
    commit_valid = bool(
        len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit)
    )
    clean = not status_result.stdout.strip()
    return (commit if commit_valid else ""), clean, True


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
    checkout_commit, checkout_clean, probe_ran = _current_checkout_source_state()
    blockers: list[str] = []
    if not probe_ran:
        return ["gpu_canary_checkout_identity_probe_failed"], ""
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

    checkout_commit, checkout_clean, probe_ran = _current_checkout_source_state()
    origin_main_commit = _current_origin_main_commit()
    remote_main_commit = _current_remote_main_commit()
    blockers: list[str] = []
    if not probe_ran:
        blockers.append("gpu_canary_orchestrator_identity_probe_failed")
    else:
        if not checkout_commit:
            blockers.append("gpu_canary_orchestrator_source_commit_unavailable")
        if not checkout_clean:
            blockers.append("gpu_canary_orchestrator_checkout_not_clean")
    identity = {
        "schema_version": "blueprint.gpu_canary_control_plane_identity.v1",
        "orchestrator_source_commit": checkout_commit or None,
        "checkout_clean": checkout_clean,
        "identity_probe_ran": probe_ran,
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


def _adp_expected_source_commit_blockers(
    expected_source_commit: str, control_identity: Mapping[str, object]
) -> tuple[list[str], str]:
    """Bind an ADP paid run to the exact clean orchestrator commit."""

    expected = expected_source_commit.strip().lower()
    blockers: list[str] = []
    if not expected:
        blockers.append("adp_gpu_canary_expected_source_commit_missing")
    elif len(expected) != 40 or any(character not in "0123456789abcdef" for character in expected):
        blockers.append("adp_gpu_canary_expected_source_commit_invalid")
    elif expected != str(control_identity.get("orchestrator_source_commit") or "").lower():
        blockers.append("adp_gpu_canary_expected_source_commit_not_control_plane_checkout")
    return blockers, expected


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


def maybe_launch_detached_gpu_canary(
    *,
    command: str,
    execute: bool,
    supervisor_dir: str | None,
    argv: Sequence[str],
    repo_root: Path,
) -> dict[str, Any] | None:
    """Detach the canonical allocator without recording raw arguments or secrets."""

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

    detached = maybe_launch_detached_gpu_canary(
        command=command,
        execute=execute,
        supervisor_dir=os.getenv(LAUNCH_DETACHED_GPU_CANARY_SUPERVISOR_DIR_ENV),
        argv=argv,
        repo_root=repo_root,
    )
    if detached is not None:
        print(json.dumps(detached, sort_keys=True))
        return 0 if detached.get("status") == "supervisor_started" else 2
    _configure_detached_supervisor_signal_policy(command)
    return None


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


def _run_reconstruction_gpu_canary(
    args: argparse.Namespace, *, checkout_commit: str
) -> dict[str, Any]:
    """Admit and optionally execute the Vast-first worker smoke."""

    if getattr(args, "reconstruction_refresh_preflight", False):
        seed = _load(args.preflight_bundle)
        provider = get_render_provider(args.provider)
        capacity_request = seed.get("capacity_request")
        capacity_request = capacity_request if isinstance(capacity_request, dict) else {}
        max_hourly_rate = getattr(args, "reconstruction_max_hourly_rate_usd", None)
        if max_hourly_rate is None:
            max_hourly_rate = capacity_request.get("max_hourly_rate_usd")
        container_disk_bytes = getattr(args, "reconstruction_container_disk_bytes", None)
        if container_disk_bytes is None:
            container_disk_bytes = seed.get("container_disk_bytes")
        if (
            isinstance(max_hourly_rate, bool)
            or not isinstance(max_hourly_rate, (int, float))
            or float(max_hourly_rate) <= 0
        ):
            max_hourly_rate = 0.0
        if (
            isinstance(container_disk_bytes, bool)
            or not isinstance(container_disk_bytes, int)
            or container_disk_bytes < 0
        ):
            container_disk_bytes = 0
        refreshed = collect_reconstruction_vast_preflight(
            name_prefix=getattr(
                args,
                "reconstruction_name_prefix",
                "blueprint-reconstruction-",
            ),
            container_disk_bytes=container_disk_bytes,
            watchdog=(seed.get("watchdog") if isinstance(seed.get("watchdog"), dict) else {}),
            conflicting_owner_present=(seed.get("conflicting_owner_present") is not False),
            capacity_probe=provider.capacity_preflight,
            inventory_probe=lambda prefix: provider.billable_inventory(name_prefix=prefix),
            max_hourly_rate_usd=float(max_hourly_rate),
        )
        write_json(Path(args.preflight_bundle), refreshed)
    admission = prepare_reconstruction_gpu_canary(
        request_path=args.provider_launch_request,
        preflight_path=args.preflight_bundle,
        admission_out=args.admission_out,
        bound_request_out=args.bound_request_out,
        adapter_output=args.adapter_output,
        provider=args.provider,
        expected_source_commit=args.expected_source_commit or "",
        checkout_source_commit=checkout_commit,
        checkout_clean=True,
        max_spend_usd=args.reconstruction_max_spend_usd,
        hard_ttl_seconds=args.reconstruction_hard_ttl_seconds,
        retry_cap=args.reconstruction_retry_cap,
        authority_id=args.reconstruction_authority_id,
        execute=args.execute,
        execution_adapter_id=select_reconstruction_execution_adapter_id(
            args.provider_launch_request, execute=args.execute
        ),
        image_release_path=getattr(args, "reconstruction_isaac_image_release", None),
        measurement_isaac_runtime_release_path=getattr(
            args, "measurement_isaac_runtime_release", None
        ),
        measurement_dlo_lab_runtime_release_path=getattr(
            args, "measurement_dlo_lab_runtime_release", None
        ),
        measurement_chrono_dem_runtime_release_path=getattr(
            args, "measurement_chrono_dem_runtime_release", None
        ),
    )
    if not args.execute or admission.get("status") != "execute_ready":
        return admission
    operation = str(admission.get("operation") or "worker_smoke")
    operation_bundle_receipt, resolved_urls, transport_blockers = (
        prepare_reconstruction_paid_transport(args=args, admission=admission, load_json=_load)
    )
    paid_admission = build_paid_lane_admission(
        resource_class="gpu_render",
        blockers=[*list(admission.get("blockers") or []), *transport_blockers],
    )
    adapter_path = Path(args.adapter_output).expanduser().resolve()
    ensure_dir(adapter_path.parent)
    write_json(adapter_path.parent / "reconstruction_paid_lane_admission.json", paid_admission)
    try:
        grant = require_paid_resource_admission(
            paid_admission,
            resource_class="gpu_render",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
    except PaidResourceAdmissionBlocked as exc:
        result = {
            "schema_version": "reconstruction_gpu_canary_adapter_result.v1",
            "status": "blocked",
            "blockers": sorted(set(exc.blockers + transport_blockers)),
            "provider_mutations_performed": 0,
            "cost_usd": 0.0,
            "scientific_qualification_inferred": False,
            "proof_effect": "none",
            "claim_ceiling": "no_execution_evidence",
        }
        write_json(adapter_path, result)
        return result
    if operation in {
        "measurement_dlo_lab_canary",
        "measurement_isaac_canary",
        "measurement_chrono_dem_canary",
    }:
        result = measurement_dlo_lab_paid_allocator.run_measurement_canary_from_canonical_allocator(
            operation=operation,
            args=args,
            bundle_receipt=operation_bundle_receipt,
            resolved_urls=resolved_urls,
            adapter_path=adapter_path,
            paid_resource_admission_grant=grant,
            load_json=_load,
        )
    elif operation == "worker_smoke":
        result = run_reconstruction_vast_worker_smoke(
            bound_request=_load(args.bound_request_out),
            preflight=_load(args.preflight_bundle),
            job_dir=adapter_path.parent / "reconstruction_vast_worker_smoke",
            output_put_url=resolved_urls["provider_output_put_url"],
            output_get_url=resolved_urls["provider_output_get_url"],
            provider=get_render_provider(args.provider),
            paid_resource_admission_grant=grant,
        )
    elif operation in {"pose_canary", "trainer_canary"}:
        result = run_reconstruction_vast_operation(
            bound_request=_load(args.bound_request_out),
            bundle_receipt=operation_bundle_receipt,
            preflight=_load(args.preflight_bundle),
            job_dir=adapter_path.parent / "reconstruction_vast_operation",
            input_bundle_get_url=resolved_urls["provider_bundle_url"],
            input_receipt_get_url=resolved_urls["operation_receipt_get_url"],
            output_bundle_put_url=resolved_urls["provider_output_put_url"],
            output_bundle_get_url=resolved_urls["provider_output_get_url"],
            provider=get_render_provider(args.provider),
            paid_resource_admission_grant=grant,
            allocator_admission=admission,
        )
    else:
        result = run_reconstruction_isaac_vast_operation(
            bound_request=_load(args.bound_request_out),
            bundle_receipt=operation_bundle_receipt,
            preflight=_load(args.preflight_bundle),
            job_dir=adapter_path.parent / "reconstruction_isaac_vast_operation",
            input_bundle_get_url=resolved_urls["provider_bundle_url"],
            input_receipt_get_url=resolved_urls["operation_receipt_get_url"],
            output_bundle_put_url=resolved_urls["provider_output_put_url"],
            output_bundle_get_url=resolved_urls["provider_output_get_url"],
            provider=get_render_provider(args.provider),
            paid_resource_admission_grant=grant,
        )
    write_json(adapter_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_teleport_provider_arguments(commands, root=ROOT)
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
    # These are legacy generic-canary inputs.  Specialized paid lanes bind
    # their own immutable bundles and must not need meaningless placeholders.
    # The generic branch below still rejects their absence before any mutation.
    gpu.add_argument("--provider-launch-request")
    gpu.add_argument("--release-evidence")
    gpu.add_argument("--model-cache-evidence")
    gpu.add_argument("--preflight-bundle")
    gpu.add_argument("--admission-out")
    gpu.add_argument("--bound-request-out")
    gpu.add_argument("--adapter-output")
    gpu.add_argument("--pod-name")
    gpu.add_argument(
        "--terminal-resource-release",
        help="Immutable release-only request; cannot be combined with a launch profile.",
    )
    gpu.add_argument(
        "--terminal-resource-release-output",
        help="Receipt destination for an exact stopped-provider-record release.",
    )
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
            NVIDIA_WAREHOUSE_NATIVE_CAMERA_PROBE_KIND,
            POLICY_RANKING_SUCCESSOR_COSMOS_PROBE_KIND,
            POLICY_RANKING_COSMOS_REASONER_PROBE_KIND,
            RECONSTRUCTION_WORKER_SMOKE_PROBE_KIND,
            ADP_SIMPLER_PUBLIC_REFERENCE_PROBE_KIND,
            ADP_ISAAC_LAB_ARENA_PROBE_KIND,
            ADP009D_NATIVE_MICROCHECK_PROBE_KIND,
            NATIVE_TASK_ARENA_CONSTRUCTION_PROBE_KIND,
            NATIVE_TASK_ARENA_CONTROLS_PROBE_KIND,
            NATIVE_TASK_ARENA_POLICY_PROBE_KIND,
            ADP009D_OVRTX_LIVE_CAMERA_PROBE_KIND,
            ADP009D_AURA_NATIVE_LIVE_CAMERA_PROBE_KIND,
            ADP_SIMREADY_ISAAC_PROBE_KIND,
            ADP_CONTENT_AGENTS_PROBE_KIND,
            ADP_JOINT_AGENT_PROBE_KIND,
            ADP_GAUSSIAN_EXCISION_PROBE_KIND,
            ADP_RETAINED_SCENE_RENDER_PROBE_KIND,
            ADP_AURA_SMOKE_PROBE_KIND,
            ADP_AURA_INTERIORGS_PROBE_KIND,
            ADP_AURA_EXACT_RESIDUAL_PROBE_KIND,
            ADP_ARTIFIXER3D_PROBE_KIND,
            ADP_INPAINT360_INTERIORGS_PROBE_KIND,
            TASK_EVALUATION_PROFILE_PREFLIGHT_PROBE_KIND,
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
    gpu.add_argument("--reconstruction-operation-bundle-receipt")
    gpu.add_argument("--reconstruction-operation-receipt-url-file")
    gpu.add_argument("--reconstruction-isaac-image-release")
    measurement_dlo_lab_paid_allocator.add_measurement_allocator_arguments(gpu)
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
    gpu.add_argument("--reconstruction-max-spend-usd", type=float)
    gpu.add_argument("--reconstruction-hard-ttl-seconds", type=int)
    gpu.add_argument("--reconstruction-retry-cap", type=int)
    gpu.add_argument("--reconstruction-authority-id")
    gpu.add_argument("--adp-public-reference-manifest")
    gpu.add_argument("--adp-arena-approval")
    gpu.add_argument("--adp009d-approved-can")
    gpu.add_argument(
        "--native-task-arena-packet",
        help="Exact sealed native_task_arena_packet directory for a task-neutral construction canary.",
    )
    gpu.add_argument(
        "--native-task-arena-bundle-receipt",
        help=(
            "Previously dry-run native task bundle receipt. Required for execute so "
            "the paid launch cannot rebuild different bytes."
        ),
    )
    gpu.add_argument(
        "--native-task-arena-runtime-source-packet",
        help=(
            "Receipt for the exact released Isaac Lab/Arena source packet installed "
            "before a task-neutral native construction canary."
        ),
    )
    gpu.add_argument(
        "--native-task-arena-construction-result",
        help=(
            "Qualified native construction result used to compile the frozen "
            "zero-action and scripted-positive control plan."
        ),
    )
    gpu.add_argument("--native-task-arena-control-result")
    gpu.add_argument("--native-task-arena-policy-execution-spec")
    gpu.add_argument("--adp009d-sage-collision")
    gpu.add_argument("--adp009d-harness-manifest")
    gpu.add_argument(
        "--adp009d-articulated-diagnostic-asset",
        help=(
            "Exact articulated USD for the blank-stage native import/drive/reset "
            "diagnostic. Requires --adp009d-articulated-diagnostic-request and "
            "--adp009d-diagnostic-only; replaces the canned-object inputs."
        ),
    )
    gpu.add_argument(
        "--adp009d-articulated-diagnostic-request",
        help="Frozen articulated_native_diagnostic_request.v1 JSON.",
    )
    gpu.add_argument(
        "--adp009d-aura-particlefield",
        default=None,
        help=(
            "Aura ParticleField USD to render inside the Isaac scene.  Omit for "
            "a micro-check with no appearance layer."
        ),
    )
    gpu.add_argument(
        "--adp009d-policy-candidate",
        default=None,
        help=(
            "Frozen candidate whose checkpoint and policy environment the worker "
            "should provision.  Omit for a micro-check with no policy."
        ),
    )
    gpu.add_argument(
        "--adp009d-controls",
        action="store_true",
        help=(
            "Run the zero-action and deterministic scripted-positive controls "
            "before admitting policy execution."
        ),
    )
    gpu.add_argument(
        "--adp009d-diagnostic-only",
        action="store_true",
        help=(
            "Explicitly admit a camera/physics diagnostic with neither controls "
            "nor a learned-policy candidate. Paid ADP-009D requests must select "
            "this mode, controls, or at least one candidate."
        ),
    )
    gpu.add_argument("--adp009d-scenario-instance", default=None)
    gpu.add_argument(
        "--adp009d-authorize-gated-backbone",
        action="store_true",
        help=(
            "Explicitly authorize forwarding the canonical Hugging Face credential "
            "only to materialize GR00T N1.7's pinned Cosmos-Reason2 backbone."
        ),
    )
    gpu.add_argument("--adp009d-ovrtx-probe-manifest")
    gpu.add_argument("--adp009d-aura-native-probe-manifest")
    gpu.add_argument("--adp009d-aura-source-root")
    gpu.add_argument("--adp-simready-isaac-bundle-receipt")
    gpu.add_argument("--adp-simready-isaac-attempt-authority")
    gpu.add_argument("--adp-job-dir")
    gpu.add_argument("--adp-max-hourly-rate-usd", type=float, default=0.80)
    gpu.add_argument("--adp-max-spend-usd", type=float, default=2.00)
    gpu.add_argument("--adp-hard-ttl-seconds", type=int, default=7200)
    gpu.add_argument("--adp-machine-avoidlist")
    gpu.add_argument(
        "--adp-allowed-active-vast-instance-id",
        action="append",
        type=int,
        default=[],
    )
    gpu.add_argument("--adp-content-agents-bundle-receipt")
    gpu.add_argument("--adp-content-agents-config-preflight-receipt")
    gpu.add_argument("--adp-content-agents-attempt-authority")
    gpu.add_argument("--adp-joint-agent-bundle-receipt")
    gpu.add_argument("--adp-gaussian-excision-bundle-receipt")
    gpu.add_argument("--adp-gaussian-excision-attempt-authority")
    gpu.add_argument("--adp-gaussian-excision-previous-attempt-receipt")
    gpu.add_argument("--adp-retained-scene-render-bundle-receipt")
    gpu.add_argument("--adp-retained-scene-render-attempt-authority")
    gpu.add_argument("--adp-retained-scene-render-job-dir")
    gpu.add_argument("--adp-retained-scene-render-max-hourly-rate-usd", type=float, default=2.0)
    gpu.add_argument("--adp-retained-scene-render-hard-ttl-seconds", type=int, default=10_800)
    gpu.add_argument("--adp-aura-bundle-receipt")
    gpu.add_argument("--adp-aura-interiorgs-bundle-receipt")
    gpu.add_argument("--adp-aura-exact-residual-bundle-receipt")
    gpu.add_argument("--adp-aura-attempt-authority")
    gpu.add_argument("--adp-artifixer3d-bundle-receipt")
    gpu.add_argument("--adp-artifixer3d-attempt-authority")
    gpu.add_argument("--adp-inpaint360-bundle-receipt")
    gpu.add_argument(
        "--reconstruction-refresh-preflight",
        action="store_true",
        help=(
            "Refresh mutation-free Vast capacity and provider inventory in the "
            "preflight bundle before reconstruction admission."
        ),
    )
    gpu.add_argument(
        "--reconstruction-name-prefix",
        default="blueprint-reconstruction-",
    )
    gpu.add_argument("--reconstruction-container-disk-bytes", type=int)
    gpu.add_argument("--reconstruction-max-hourly-rate-usd", type=float)
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
    if args.command == "gpu-canary":
        normal_required = (
            "provider_launch_request", "release_evidence", "model_cache_evidence",
            "preflight_bundle", "admission_out", "bound_request_out", "adapter_output", "pod_name",
        )
        if args.terminal_resource_release:
            if not args.terminal_resource_release_output or not args.execute:
                parser.error(
                    "--terminal-resource-release requires --terminal-resource-release-output and --execute"
                )
            if str(os.getenv(TERMINAL_RESOURCE_RELEASE_WORKER_ENV) or "").strip().lower() not in {
                "1", "true", "yes", "on"
            }:
                parser.error("--terminal-resource-release is restricted to the queue worker")
            if any(getattr(args, name, None) for name in normal_required):
                parser.error("--terminal-resource-release cannot be combined with launch arguments")
    detached_exit = configure_or_launch_detached_gpu_canary(
        args.command,
        execute=bool(getattr(args, "execute", False)),
        argv=list(argv) if argv is not None else sys.argv[1:],
        repo_root=ROOT,
    )
    if detached_exit is not None:
        return detached_exit
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
    if args.command == "provider-reconstruction":
        result = run_teleport_provider(
            args,
            load_json=_load,
            source_checkout_blockers=_source_checkout_blockers,
            credential_loader=load_teleport_credentials,
        )
        success = result.get("status") in {"dry_run_ready", "succeeded_unqualified"}
    elif args.command == "cpu-build":
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
        if args.terminal_resource_release:
            output = Path(args.terminal_resource_release_output).expanduser().resolve()
            result = dispatch_terminal_resource_release(
                request_path=args.terminal_resource_release,
                state_root=output.parent,
            )
            write_json(output, result)
            success = result.get("status") == "completed"
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == TASK_EVALUATION_PROFILE_PREFLIGHT_PROBE_KIND:
            control_blockers, control_identity = _control_plane_checkout_blockers()
            source_blockers, expected_source_commit = _adp_expected_source_commit_blockers(
                args.expected_source_commit or "", control_identity
            )
            if control_blockers or source_blockers:
                result = {
                    "schema_version": "task_evaluation_allocator_preflight_result.v1",
                    "status": "blocked",
                    "blockers": sorted(set([*control_blockers, *source_blockers])),
                    "provider_mutation_attempted": False,
                    "provider_mutations_performed": 0,
                    "continuing_spend_from_this_run": False,
                    "retry_cap": 0,
                }
            else:
                result = run_task_evaluation_profile_preflight(
                    request_path=args.provider_launch_request,
                    release_evidence_path=args.release_evidence,
                    readiness_receipt_path=args.model_cache_evidence,
                    provider_guard_path=args.preflight_bundle,
                    expected_source_commit=expected_source_commit,
                    observed_source_commit=str(
                        control_identity.get("orchestrator_source_commit") or ""
                    ),
                    execute=args.execute,
                )
            write_json(Path(args.admission_out), result)
            write_json(Path(args.adapter_output), result)
            success = result.get("status") == "dry_run_ready"
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == ADP_RETAINED_SCENE_RENDER_PROBE_KIND:
            missing = [
                name
                for name in (
                    "adp_retained_scene_render_bundle_receipt",
                    "adp_retained_scene_render_job_dir",
                )
                if not getattr(args, name, None)
            ]
            control_blockers, control_identity = _control_plane_checkout_blockers()
            source_blockers, expected_source_commit = _adp_expected_source_commit_blockers(
                args.expected_source_commit or "", control_identity
            )
            blockers = [*missing, *control_blockers, *source_blockers]
            if args.provider != "vast":
                blockers.append("retained_scene_render_provider_must_be_vast")
            prepared_bundle: dict[str, Any] | None = None
            receipt_path: Path | None = None
            if args.adp_retained_scene_render_bundle_receipt:
                receipt_path = (
                    Path(args.adp_retained_scene_render_bundle_receipt).expanduser().resolve()
                )
                if not receipt_path.is_file():
                    blockers.append("retained_scene_render_bundle_receipt_missing")
                else:
                    try:
                        prepared_bundle = _load(receipt_path)
                        validate_retained_scene_render_bundle(
                            prepared_bundle, expected_commit=expected_source_commit or None
                        )
                    except (OSError, ValueError, json.JSONDecodeError):
                        blockers.append("retained_scene_render_bundle_binding_invalid")
            execution_authority: dict[str, Any] | None = None
            allowed_active_instance_ids: list[int] = []
            if prepared_bundle is not None:
                record = prepared_bundle.get("execution_authority")
                if not isinstance(record, Mapping):
                    blockers.append("retained_scene_render_execution_authority_record_invalid")
                else:
                    authority_path = Path(str(record.get("path") or "")).expanduser().resolve()
                    observed = (
                        "sha256:" + hashlib.sha256(authority_path.read_bytes()).hexdigest()
                        if authority_path.is_file()
                        else ""
                    )
                    try:
                        execution_authority = _load(authority_path)
                    except (OSError, ValueError, json.JSONDecodeError):
                        execution_authority = None
                    if (
                        execution_authority is None
                        or record.get("sha256") != observed
                        or record.get("size_bytes") != authority_path.stat().st_size
                        or record.get("authority_digest")
                        != canonical_digest(execution_authority, digest_field="authority_digest")
                    ):
                        blockers.append("retained_scene_render_execution_authority_invalid")
                    else:
                        paid = execution_authority.get("paid_compute")
                        rows = (
                            paid.get("external_instance_allowlist")
                            if isinstance(paid, Mapping)
                            else None
                        )
                        if not isinstance(rows, list) or any(
                            isinstance(value, bool) or not isinstance(value, int) or value <= 0
                            for value in rows
                        ):
                            blockers.append(
                                "retained_scene_render_external_instance_allowlist_invalid"
                            )
                        else:
                            allowed_active_instance_ids = sorted(set(rows))
            if sorted(set(args.adp_allowed_active_vast_instance_id)) != allowed_active_instance_ids:
                blockers.append("retained_scene_render_external_instance_allowlist_mismatch")
            hard_cap_usd = (
                prepared_bundle.get("hard_total_spend_cap_usd")
                if prepared_bundle is not None
                else None
            )
            rate = args.adp_retained_scene_render_max_hourly_rate_usd
            ttl = args.adp_retained_scene_render_hard_ttl_seconds
            if (
                not isinstance(hard_cap_usd, (int, float))
                or isinstance(hard_cap_usd, bool)
                or rate <= 0
                or not 1800 <= ttl <= 10_800
                or rate * ttl / 3600.0 > float(hard_cap_usd)
            ):
                blockers.append("retained_scene_render_budget_or_ttl_invalid")
            machine_avoidlist_path: Path | None = None
            machine_avoidlist_sha256: str | None = None
            if args.adp_machine_avoidlist:
                machine_avoidlist_path = Path(args.adp_machine_avoidlist).expanduser().resolve()
                if not machine_avoidlist_path.is_file():
                    blockers.append("retained_scene_render_machine_avoidlist_missing")
                else:
                    machine_avoidlist_sha256 = (
                        "sha256:" + hashlib.sha256(machine_avoidlist_path.read_bytes()).hexdigest()
                    )
            paid_attempt_authority: dict[str, Any] | None = None
            if args.adp_retained_scene_render_attempt_authority:
                try:
                    paid_attempt_authority = _load(
                        Path(args.adp_retained_scene_render_attempt_authority)
                        .expanduser()
                        .resolve()
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    blockers.append("retained_scene_render_paid_attempt_authority_invalid")
            elif args.execute:
                blockers.append("retained_scene_render_paid_attempt_authority_missing")
            if paid_attempt_authority is not None and prepared_bundle is not None:
                try:
                    validate_retained_scene_render_paid_attempt_authority(
                        paid_attempt_authority,
                        prepared_bundle=prepared_bundle,
                        max_hourly_rate_usd=rate,
                        hard_ttl_seconds=ttl,
                        allowed_active_instance_ids=allowed_active_instance_ids,
                    )
                except ValueError as exc:
                    blockers.append(str(exc))
            receipt_sha256 = (
                "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                if receipt_path and receipt_path.is_file()
                else None
            )
            allocation_binding = {
                "program_id": "arm-decision-proof-v1",
                "probe_kind": ADP_RETAINED_SCENE_RENDER_PROBE_KIND,
                "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
                "expected_source_commit": expected_source_commit or None,
                "bundle_receipt_sha256": receipt_sha256,
                "bundle_sha256": prepared_bundle.get("bundle_sha256") if prepared_bundle else None,
                "execution_authority_digest": (
                    (prepared_bundle.get("execution_authority") or {}).get("authority_digest")
                    if prepared_bundle
                    else None
                ),
                "paid_attempt_authority_digest": (
                    paid_attempt_authority.get("authorization_digest")
                    if paid_attempt_authority
                    else None
                ),
                "max_hourly_rate_usd": rate,
                "hard_cap_usd": hard_cap_usd,
                "hard_ttl_seconds": ttl,
                "allowed_active_vast_instance_ids": allowed_active_instance_ids,
                "machine_avoidlist_sha256": machine_avoidlist_sha256,
                "retry_cap": 0,
            }
            binding_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(allocation_binding, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            )
            admission = build_paid_lane_admission(
                resource_class="vast_provider_adapter", blockers=sorted(set(blockers))
            )
            admission.update(
                {
                    "program_id": "arm-decision-proof-v1",
                    "probe_kind": ADP_RETAINED_SCENE_RENDER_PROBE_KIND,
                    "control_plane_identity": control_identity,
                    "authority": "user_authorized_retained_scene_gpu_render",
                    "private_scene_derived_input_only": True,
                    "raw_interiorgs_downloaded_bytes_uploaded": False,
                    "provider_training_authorized": False,
                    "publication_authorized": False,
                    "retry_cap": 0,
                    "allocation_binding": allocation_binding,
                    "allocation_binding_digest": binding_digest,
                }
            )
            write_json(Path(args.admission_out), admission)
            grant: PaidResourceAdmissionGrant | None = None
            if args.execute:
                try:
                    grant = require_paid_resource_admission(
                        admission,
                        resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                    )
                except PaidResourceAdmissionBlocked as exc:
                    result = {
                        "status": "blocked",
                        "blockers": exc.blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.adapter_output), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
            if blockers or prepared_bundle is None:
                result = {
                    "status": "blocked",
                    "blockers": sorted(set(blockers)),
                    "provider_mutations_performed": 0,
                }
            else:
                result = run_retained_scene_render_vast(
                    job_dir=args.adp_retained_scene_render_job_dir,
                    paid_resource_admission_grant=grant,
                    execute=args.execute,
                    prepared_bundle=prepared_bundle,
                    paid_attempt_authority=paid_attempt_authority,
                    max_hourly_rate_usd=rate,
                    hard_ttl_seconds=ttl,
                    allowed_active_instance_ids=allowed_active_instance_ids,
                    machine_avoidlist_path=machine_avoidlist_path,
                )
            write_json(Path(args.adapter_output), result)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == ADP_GAUSSIAN_EXCISION_PROBE_KIND:
            missing = [
                name
                for name in (
                    "adp_gaussian_excision_bundle_receipt",
                    "adp_job_dir",
                )
                if not getattr(args, name, None)
            ]
            control_blockers, control_identity = _control_plane_checkout_blockers()
            source_blockers, expected_source_commit = _adp_expected_source_commit_blockers(
                args.expected_source_commit or "", control_identity
            )
            blockers = [*missing, *control_blockers, *source_blockers]
            if args.provider != "vast":
                blockers.append("gaussian_excision_provider_must_be_vast")
            if args.adp_max_hourly_rate_usd <= 0 or args.adp_max_hourly_rate_usd > 0.60:
                blockers.append("gaussian_excision_max_hourly_rate_invalid")
            if args.adp_max_spend_usd != 1.50:
                blockers.append("gaussian_excision_hard_cap_mismatch")
            if args.adp_hard_ttl_seconds != 3600:
                blockers.append("gaussian_excision_hard_ttl_mismatch")
            if any(value <= 0 for value in args.adp_allowed_active_vast_instance_id):
                blockers.append("gaussian_excision_allowed_active_instance_id_invalid")
            machine_avoidlist_path: Path | None = None
            machine_avoidlist_sha256: str | None = None
            if args.adp_machine_avoidlist:
                machine_avoidlist_path = Path(args.adp_machine_avoidlist).expanduser().resolve()
                if not machine_avoidlist_path.is_file():
                    blockers.append("gaussian_excision_machine_avoidlist_missing")
                else:
                    machine_avoidlist_sha256 = (
                        "sha256:" + hashlib.sha256(machine_avoidlist_path.read_bytes()).hexdigest()
                    )
            prepared_bundle: dict[str, Any] | None = None
            receipt_path: Path | None = None
            if args.adp_gaussian_excision_bundle_receipt:
                receipt_path = (
                    Path(args.adp_gaussian_excision_bundle_receipt).expanduser().resolve()
                )
                if not receipt_path.is_file():
                    blockers.append("gaussian_excision_bundle_receipt_missing")
                else:
                    try:
                        prepared_bundle = _load(receipt_path)
                    except (OSError, ValueError, json.JSONDecodeError):
                        blockers.append("gaussian_excision_bundle_receipt_invalid")
            bundle_path: Path | None = None
            if prepared_bundle is not None:
                bundle_path = (
                    Path(str(prepared_bundle.get("bundle_path") or "")).expanduser().resolve()
                )
                observed_bundle_sha256 = (
                    "sha256:" + hashlib.sha256(bundle_path.read_bytes()).hexdigest()
                    if bundle_path.is_file()
                    else ""
                )
                released = prepared_bundle.get("released_code") or {}
                if (
                    prepared_bundle.get("status") != "ready"
                    or prepared_bundle.get("provider_bundle_kind")
                    != ADP_GAUSSIAN_EXCISION_PROVIDER_BUNDLE_KIND
                    or prepared_bundle.get("container_image") != ADP_GAUSSIAN_EXCISION_IMAGE
                    or prepared_bundle.get("blueprint_commit") != expected_source_commit
                    or released.get("tree") != ADP_GAUSSIAN_EXCISION_SOURCE_TREE
                    or released.get("source_modified") is not False
                    or prepared_bundle.get("hard_cap_usd") != 1.50
                    or prepared_bundle.get("hard_ttl_seconds") != 3600
                    or prepared_bundle.get("maximum_paid_attempts") != 1
                    or prepared_bundle.get("automatic_paid_retry_allowed") is not False
                    or prepared_bundle.get("provider_zero_required_after_return") is not True
                    or prepared_bundle.get("provider_network_dependency_install_required")
                    is not False
                    or not str(
                        prepared_bundle.get("dependency_wheelhouse_manifest_digest") or ""
                    ).startswith("sha256:")
                    or prepared_bundle.get("raw_interiorgs_downloaded_bytes_included") is not False
                    or prepared_bundle.get("private_scene_derived_standard_splat_included")
                    is not True
                    or prepared_bundle.get("blockers") not in ([], None)
                    or not bundle_path.is_file()
                    or observed_bundle_sha256 != prepared_bundle.get("bundle_sha256")
                ):
                    blockers.append("gaussian_excision_bundle_binding_invalid")
            receipt_sha256 = (
                "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                if receipt_path and receipt_path.is_file()
                else None
            )
            paid_attempt_authority: dict[str, Any] | None = None
            previous_attempt_receipt: dict[str, Any] | None = None
            if args.adp_gaussian_excision_attempt_authority:
                try:
                    paid_attempt_authority = _load(
                        Path(args.adp_gaussian_excision_attempt_authority).expanduser().resolve()
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    blockers.append("gaussian_excision_paid_attempt_authority_invalid")
            elif args.execute:
                blockers.append("gaussian_excision_paid_attempt_authority_missing")
            if args.adp_gaussian_excision_previous_attempt_receipt:
                try:
                    previous_attempt_receipt = _load(
                        Path(args.adp_gaussian_excision_previous_attempt_receipt)
                        .expanduser()
                        .resolve()
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    blockers.append("gaussian_excision_previous_attempt_receipt_invalid")
            if paid_attempt_authority is not None and prepared_bundle is not None:
                try:
                    validate_gaussian_excision_paid_attempt_authority(
                        paid_attempt_authority,
                        prepared_bundle=prepared_bundle,
                        previous_attempt_receipt=previous_attempt_receipt,
                        allowed_active_instance_ids=args.adp_allowed_active_vast_instance_id,
                    )
                except ValueError as exc:
                    blockers.append(str(exc))
            allocation_binding = {
                "program_id": "arm-decision-proof-v1",
                "probe_kind": ADP_GAUSSIAN_EXCISION_PROBE_KIND,
                "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
                "expected_source_commit": expected_source_commit or None,
                "bundle_receipt_sha256": receipt_sha256,
                "bundle_sha256": (
                    prepared_bundle.get("bundle_sha256") if prepared_bundle else None
                ),
                "freeze_digest": (
                    prepared_bundle.get("freeze_digest") if prepared_bundle else None
                ),
                "execution_authority_digest": (
                    prepared_bundle.get("execution_authority_digest") if prepared_bundle else None
                ),
                "paid_attempt_authority_digest": (
                    paid_attempt_authority.get("authorization_digest")
                    if paid_attempt_authority
                    else None
                ),
                "previous_attempt_receipt_digest": (
                    previous_attempt_receipt.get("receipt_digest")
                    if previous_attempt_receipt
                    else None
                ),
                "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                "hard_cap_usd": args.adp_max_spend_usd,
                "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                "allowed_active_vast_instance_ids": sorted(
                    set(args.adp_allowed_active_vast_instance_id)
                ),
                "machine_avoidlist_sha256": machine_avoidlist_sha256,
                "retry_cap": 0,
            }
            allocation_binding_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(allocation_binding, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()
            )
            paid_admission = build_paid_lane_admission(
                resource_class="vast_provider_adapter",
                blockers=sorted(set(blockers)),
            )
            paid_admission.update(
                {
                    "program_id": "arm-decision-proof-v1",
                    "probe_kind": ADP_GAUSSIAN_EXCISION_PROBE_KIND,
                    "control_plane_identity": control_identity,
                    "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                    "hard_cap_usd": args.adp_max_spend_usd,
                    "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                    "retry_cap": 0,
                    "authority": "user_authorized_gaussian_ownership_gpu_audit",
                    "private_scene_derived_input_only": True,
                    "raw_interiorgs_downloaded_bytes_uploaded": False,
                    "heldout_cameras_accessed_for_classification": False,
                    "allocation_binding": allocation_binding,
                    "allocation_binding_digest": allocation_binding_digest,
                }
            )
            write_json(Path(args.admission_out), paid_admission)
            grant = None
            if args.execute:
                try:
                    grant = require_paid_resource_admission(
                        paid_admission,
                        resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                    )
                except PaidResourceAdmissionBlocked as exc:
                    result = {
                        "status": "blocked",
                        "blockers": exc.blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.adapter_output), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
            if blockers or prepared_bundle is None:
                result = {
                    "status": "blocked",
                    "blockers": sorted(set(blockers)),
                    "provider_mutations_performed": 0,
                }
            else:
                result = run_gaussian_excision_vast(
                    job_dir=args.adp_job_dir,
                    paid_resource_admission_grant=grant,
                    execute=args.execute,
                    prepared_bundle=prepared_bundle,
                    paid_attempt_authority=paid_attempt_authority,
                    previous_attempt_receipt=previous_attempt_receipt,
                    max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                    hard_cap_usd=args.adp_max_spend_usd,
                    hard_ttl_seconds=args.adp_hard_ttl_seconds,
                    public_image=ADP_GAUSSIAN_EXCISION_IMAGE,
                    allowed_active_instance_ids=args.adp_allowed_active_vast_instance_id,
                    machine_avoidlist_path=machine_avoidlist_path,
                )
            write_json(Path(args.adapter_output), result)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == ADP_JOINT_AGENT_PROBE_KIND:
            missing = [
                name
                for name in ("adp_joint_agent_bundle_receipt", "adp_job_dir")
                if not getattr(args, name, None)
            ]
            control_blockers, control_identity = _control_plane_checkout_blockers()
            source_blockers, expected_source_commit = _adp_expected_source_commit_blockers(
                args.expected_source_commit or "", control_identity
            )
            blockers = [*missing, *control_blockers, *source_blockers]
            if args.provider != "vast":
                blockers.append("adp_joint_agent_provider_must_be_vast")
            if not 0 < args.adp_max_hourly_rate_usd <= args.adp_max_spend_usd:
                blockers.append("adp_joint_agent_budget_invalid")
            if not 5400 <= args.adp_hard_ttl_seconds <= 14_400:
                blockers.append("adp_joint_agent_hard_ttl_invalid")
            if any(value <= 0 for value in args.adp_allowed_active_vast_instance_id):
                blockers.append("adp_joint_agent_allowed_active_vast_instance_id_invalid")
            allowed_active_instance_ids = sorted(set(args.adp_allowed_active_vast_instance_id))
            prepared_bundle: dict[str, Any] | None = None
            receipt_path: Path | None = None
            if args.adp_joint_agent_bundle_receipt:
                receipt_path = Path(args.adp_joint_agent_bundle_receipt).expanduser().resolve()
                if not receipt_path.is_file():
                    blockers.append("adp_joint_agent_bundle_receipt_missing")
                else:
                    try:
                        prepared_bundle = _load(receipt_path)
                    except (OSError, ValueError, json.JSONDecodeError):
                        blockers.append("adp_joint_agent_bundle_receipt_invalid")
            bundle_path: Path | None = None
            if prepared_bundle is not None:
                bundle_path = (
                    Path(str(prepared_bundle.get("bundle_path") or "")).expanduser().resolve()
                )
                observed_bundle_sha256 = (
                    "sha256:" + hashlib.sha256(bundle_path.read_bytes()).hexdigest()
                    if bundle_path.is_file()
                    else ""
                )
                blueprint_source = prepared_bundle.get("blueprint_source") or {}
                scope_digest = str(prepared_bundle.get("scope_amendment_digest") or "")
                model_preflight_digest = str(
                    prepared_bundle.get("model_preflight_receipt_digest")
                    or prepared_bundle.get("nim_preflight_receipt_digest")
                    or ""
                )
                model = prepared_bundle.get("model") or {}
                model_backend = str(model.get("backend") or "")
                capability = model.get("capability_preflight") or {}
                capability_verified = capability.get("verified_capabilities")
                capability_preflight_valid = (
                    capability.get("schema_version") == HOSTED_MODEL_PREFLIGHT_SCHEMA_VERSION
                    and capability.get("probe_profile") == HOSTED_MODEL_PROBE_PROFILE
                    and isinstance(capability_verified, list)
                    and set(HOSTED_MODEL_REQUIRED_CAPABILITIES).issubset(set(capability_verified))
                    and capability.get("receipt_digest") == model_preflight_digest
                )
                legacy_nim_bundle = not prepared_bundle.get("model_preflight_receipt_digest")
                if (
                    prepared_bundle.get("status") != "ready"
                    or prepared_bundle.get("provider_bundle_kind")
                    != ADP_JOINT_AGENT_PROVIDER_BUNDLE_KIND
                    or prepared_bundle.get("container_image") != ADP_JOINT_AGENT_IMAGE
                    or prepared_bundle.get("source_tree") != ADP_JOINT_AGENT_SOURCE_TREE
                    or blueprint_source.get("commit") != expected_source_commit
                    or blueprint_source.get("dirty") is not False
                    or prepared_bundle.get("completion_retries") != 0
                    or prepared_bundle.get("automatic_paid_retry_allowed") is not False
                    or prepared_bundle.get("provider_zero_required_after_return") is not True
                    or prepared_bundle.get("blockers") not in ([], None)
                    or len(scope_digest) != 71
                    or not scope_digest.startswith("sha256:")
                    or any(character not in "0123456789abcdef" for character in scope_digest[7:])
                    or len(model_preflight_digest) != 71
                    or not model_preflight_digest.startswith("sha256:")
                    or any(
                        character not in "0123456789abcdef"
                        for character in model_preflight_digest[7:]
                    )
                    or (
                        not legacy_nim_bundle
                        and (
                            model_backend not in HOSTED_MODEL_BACKENDS
                            or not str(model.get("id") or "")
                            or not capability_preflight_valid
                        )
                    )
                    or not bundle_path.is_file()
                    or observed_bundle_sha256 != prepared_bundle.get("bundle_sha256")
                ):
                    blockers.append("adp_joint_agent_bundle_binding_invalid")
                if allowed_active_instance_ids and (
                    len(allowed_active_instance_ids) != 1
                    or prepared_bundle.get("one_instance_at_a_time") is not False
                    or prepared_bundle.get("maximum_concurrent_paid_instances") != 2
                ):
                    blockers.append("adp_joint_agent_concurrent_authority_binding_invalid")
            joint_avoidlist_path: Path | None = None
            joint_avoidlist_sha256: str | None = None
            if args.adp_machine_avoidlist:
                joint_avoidlist_path = Path(args.adp_machine_avoidlist).expanduser().resolve()
                try:
                    joint_avoidlist = _load(joint_avoidlist_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    blockers.append("adp_joint_agent_machine_avoidlist_invalid")
                else:
                    if (
                        joint_avoidlist.get("schema_version") != "vast_machine_avoidlist.v1"
                        or not isinstance(joint_avoidlist.get("machine_ids"), list)
                        or any(
                            not isinstance(machine_id, int) or machine_id <= 0
                            for machine_id in joint_avoidlist["machine_ids"]
                        )
                    ):
                        blockers.append("adp_joint_agent_machine_avoidlist_invalid")
                    joint_avoidlist_sha256 = (
                        "sha256:" + hashlib.sha256(joint_avoidlist_path.read_bytes()).hexdigest()
                    )
            receipt_sha256 = (
                "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                if receipt_path and receipt_path.is_file()
                else None
            )
            allocation_binding = {
                "program_id": "arm-decision-proof-v1",
                "probe_kind": ADP_JOINT_AGENT_PROBE_KIND,
                "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
                "expected_source_commit": expected_source_commit or None,
                "bundle_receipt_sha256": receipt_sha256,
                "bundle_sha256": (
                    prepared_bundle.get("bundle_sha256") if prepared_bundle else None
                ),
                "scope_amendment_digest": (
                    prepared_bundle.get("scope_amendment_digest") if prepared_bundle else None
                ),
                "model_preflight_receipt_digest": (
                    prepared_bundle.get("model_preflight_receipt_digest")
                    or prepared_bundle.get("nim_preflight_receipt_digest")
                    if prepared_bundle
                    else None
                ),
                "model": prepared_bundle.get("model") if prepared_bundle else None,
                "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                "hard_cap_usd": args.adp_max_spend_usd,
                "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                "allowed_active_vast_instance_ids": allowed_active_instance_ids,
                "machine_avoidlist_sha256": joint_avoidlist_sha256,
                "retry_cap": 0,
            }
            allocation_binding_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(allocation_binding, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()
            )
            paid_admission = build_paid_lane_admission(
                resource_class="vast_provider_adapter", blockers=sorted(set(blockers))
            )
            paid_admission.update(
                {
                    "program_id": "arm-decision-proof-v1",
                    "probe_kind": ADP_JOINT_AGENT_PROBE_KIND,
                    "control_plane_identity": control_identity,
                    "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                    "hard_cap_usd": args.adp_max_spend_usd,
                    "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                    "retry_cap": 0,
                    "explicit_concurrent_gpu_authority_bound": bool(
                        args.adp_allowed_active_vast_instance_id
                    ),
                    "authority": "user_authorized_bounded_joint_agent_gpu_compute",
                    "private_scene_derived_input_only": True,
                    "raw_interiorgs_downloaded_bytes_uploaded": False,
                    "candidate_policy_queried": False,
                    "allocation_binding": allocation_binding,
                    "allocation_binding_digest": allocation_binding_digest,
                }
            )
            write_json(Path(args.admission_out), paid_admission)
            grant = None
            if args.execute:
                try:
                    grant = require_paid_resource_admission(
                        paid_admission,
                        resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                    )
                except PaidResourceAdmissionBlocked as exc:
                    result = {
                        "status": "blocked",
                        "blockers": exc.blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.adapter_output), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
            if blockers or prepared_bundle is None:
                result = {
                    "status": "blocked",
                    "blockers": sorted(set(blockers)),
                    "provider_mutations_performed": 0,
                }
            else:
                result = run_joint_agent_vast(
                    job_dir=args.adp_job_dir,
                    paid_resource_admission_grant=grant,
                    execute=args.execute,
                    prepared_bundle=prepared_bundle,
                    max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                    hard_cap_usd=args.adp_max_spend_usd,
                    hard_ttl_seconds=args.adp_hard_ttl_seconds,
                    public_image=ADP_JOINT_AGENT_IMAGE,
                    allowed_active_instance_ids=args.adp_allowed_active_vast_instance_id,
                    machine_avoidlist_path=joint_avoidlist_path,
                )
            write_json(Path(args.adapter_output), result)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == ADP_INPAINT360_INTERIORGS_PROBE_KIND:
            missing = [
                name
                for name in ("adp_inpaint360_bundle_receipt", "adp_job_dir")
                if not getattr(args, name, None)
            ]
            control_blockers, control_identity = _control_plane_checkout_blockers()
            source_blockers, expected_source_commit = _adp_expected_source_commit_blockers(
                args.expected_source_commit or "", control_identity
            )
            blockers = [*missing, *control_blockers, *source_blockers]
            if args.provider != "vast":
                blockers.append("adp_inpaint360_provider_must_be_vast")
            if not 0 < args.adp_max_hourly_rate_usd <= args.adp_max_spend_usd:
                blockers.append("adp_inpaint360_budget_invalid")
            if any(value <= 0 for value in args.adp_allowed_active_vast_instance_id):
                blockers.append("adp_inpaint360_allowed_active_vast_instance_id_invalid")
            if not 7200 <= args.adp_hard_ttl_seconds <= 14_400:
                blockers.append("adp_inpaint360_hard_ttl_invalid")
            prepared_bundle: dict[str, Any] | None = None
            receipt_path: Path | None = None
            if args.adp_inpaint360_bundle_receipt:
                receipt_path = Path(args.adp_inpaint360_bundle_receipt).expanduser().resolve()
                if not receipt_path.is_file():
                    blockers.append("adp_inpaint360_bundle_receipt_missing")
                else:
                    try:
                        prepared_bundle = _load(receipt_path)
                    except (OSError, ValueError, json.JSONDecodeError):
                        blockers.append("adp_inpaint360_bundle_receipt_invalid")
            bundle_path: Path | None = None
            if prepared_bundle is not None:
                bundle_path = (
                    Path(str(prepared_bundle.get("bundle_path") or "")).expanduser().resolve()
                )
                observed_bundle_sha256 = (
                    "sha256:" + hashlib.sha256(bundle_path.read_bytes()).hexdigest()
                    if bundle_path.is_file()
                    else ""
                )
                if (
                    prepared_bundle.get("status") != "ready"
                    or prepared_bundle.get("source_commit") != ADP_INPAINT360_SOURCE_COMMIT
                    or prepared_bundle.get("source_tree") != ADP_INPAINT360_SOURCE_TREE
                    or prepared_bundle.get("lama_source_commit")
                    != ADP_INPAINT360_LAMA_SOURCE_COMMIT
                    or prepared_bundle.get("lama_source_tree") != ADP_INPAINT360_LAMA_SOURCE_TREE
                    or prepared_bundle.get("prerequisite_receipt_digest")
                    != ADP_INPAINT360_PREREQUISITE_RECEIPT_DIGEST
                    or prepared_bundle.get("container_image") != ADP_INPAINT360_INTERIORGS_IMAGE
                    or prepared_bundle.get("retry_cap") != 0
                    or prepared_bundle.get("blockers") not in ([], None)
                    or not prepared_bundle.get("adapter_receipt_digest")
                    or not isinstance(prepared_bundle.get("target_binding"), Mapping)
                    or not str(
                        (prepared_bundle.get("target_binding") or {}).get("scene_id") or ""
                    ).isdigit()
                    or not str(
                        (prepared_bundle.get("target_binding") or {}).get("target_instance_id")
                        or ""
                    ).isdigit()
                    or not str(
                        (prepared_bundle.get("target_binding") or {}).get("method_config_id") or ""
                    )
                    or prepared_bundle.get("blueprint_repository_tracked_state") != "clean"
                    or prepared_bundle.get("blueprint_repository_commit") != expected_source_commit
                    or not bundle_path.is_file()
                    or observed_bundle_sha256 != prepared_bundle.get("bundle_sha256")
                ):
                    blockers.append("adp_inpaint360_bundle_binding_invalid")
            receipt_sha256 = (
                "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                if receipt_path and receipt_path.is_file()
                else None
            )
            avoidlist_path: Path | None = None
            avoidlist_sha256: str | None = None
            if args.adp_machine_avoidlist:
                avoidlist_path = Path(args.adp_machine_avoidlist).expanduser().resolve()
                try:
                    avoidlist = _load(avoidlist_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    blockers.append("adp_inpaint360_machine_avoidlist_invalid")
                else:
                    if (
                        avoidlist.get("schema_version") != "vast_machine_avoidlist.v1"
                        or not isinstance(avoidlist.get("machine_ids"), list)
                        or any(
                            not isinstance(machine_id, int) or machine_id <= 0
                            for machine_id in avoidlist["machine_ids"]
                        )
                    ):
                        blockers.append("adp_inpaint360_machine_avoidlist_invalid")
                    avoidlist_sha256 = (
                        "sha256:" + hashlib.sha256(avoidlist_path.read_bytes()).hexdigest()
                    )
            allocation_binding = {
                "program_id": "arm-decision-proof-v1",
                "probe_kind": ADP_INPAINT360_INTERIORGS_PROBE_KIND,
                "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
                "expected_source_commit": expected_source_commit or None,
                "bundle_receipt_sha256": receipt_sha256,
                "bundle_sha256": (
                    prepared_bundle.get("bundle_sha256") if prepared_bundle else None
                ),
                "blueprint_repository_commit": (
                    prepared_bundle.get("blueprint_repository_commit") if prepared_bundle else None
                ),
                "inpaint360_source_commit": ADP_INPAINT360_SOURCE_COMMIT,
                "inpaint360_source_tree": ADP_INPAINT360_SOURCE_TREE,
                "lama_source_commit": ADP_INPAINT360_LAMA_SOURCE_COMMIT,
                "lama_source_tree": ADP_INPAINT360_LAMA_SOURCE_TREE,
                "prerequisite_receipt_digest": ADP_INPAINT360_PREREQUISITE_RECEIPT_DIGEST,
                "adapter_receipt_digest": (
                    prepared_bundle.get("adapter_receipt_digest") if prepared_bundle else None
                ),
                "target_binding": (
                    prepared_bundle.get("target_binding") if prepared_bundle else None
                ),
                "container_image": ADP_INPAINT360_INTERIORGS_IMAGE,
                "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                "hard_cap_usd": args.adp_max_spend_usd,
                "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                "machine_avoidlist_sha256": avoidlist_sha256,
                "allowed_active_vast_instance_ids": sorted(
                    set(args.adp_allowed_active_vast_instance_id)
                ),
                "retry_cap": 0,
            }
            allocation_binding_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(allocation_binding, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()
            )
            paid_admission = build_paid_lane_admission(
                resource_class="vast_provider_adapter", blockers=sorted(set(blockers))
            )
            paid_admission.update(
                {
                    "program_id": "arm-decision-proof-v1",
                    "probe_kind": ADP_INPAINT360_INTERIORGS_PROBE_KIND,
                    "control_plane_identity": control_identity,
                    "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                    "hard_cap_usd": args.adp_max_spend_usd,
                    "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                    "retry_cap": 0,
                    "explicit_concurrent_gpu_authority_bound": bool(
                        args.adp_allowed_active_vast_instance_id
                    ),
                    "authority": (
                        "user_authorized_all_in_scope_goal_resources_including_gpu_usage"
                    ),
                    "publisher_source_rights_bound": True,
                    "dataset_internal_use_only": True,
                    "rendered_frames_have_no_hidden_background_truth": True,
                    "replacement_or_physics_result_claimed": False,
                    "allocation_binding": allocation_binding,
                    "allocation_binding_digest": allocation_binding_digest,
                }
            )
            write_json(Path(args.admission_out), paid_admission)
            grant = None
            if args.execute:
                try:
                    grant = require_paid_resource_admission(
                        paid_admission,
                        resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                    )
                except PaidResourceAdmissionBlocked as exc:
                    result = {
                        "status": "blocked",
                        "blockers": exc.blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.adapter_output), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
            if blockers or prepared_bundle is None:
                result = {
                    "status": "blocked",
                    "blockers": sorted(set(blockers)),
                    "provider_mutations_performed": 0,
                }
            else:
                result = run_inpaint360_interiorgs_vast(
                    job_dir=args.adp_job_dir,
                    paid_resource_admission_grant=grant,
                    execute=args.execute,
                    prepared_bundle=prepared_bundle,
                    max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                    hard_cap_usd=args.adp_max_spend_usd,
                    hard_ttl_seconds=args.adp_hard_ttl_seconds,
                    public_image=ADP_INPAINT360_INTERIORGS_IMAGE,
                    machine_avoidlist_path=avoidlist_path,
                    allowed_active_instance_ids=args.adp_allowed_active_vast_instance_id,
                )
            write_json(Path(args.adapter_output), result)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind in {
            ADP_AURA_SMOKE_PROBE_KIND,
            ADP_AURA_INTERIORGS_PROBE_KIND,
        }:
            aura_interiorgs = args.probe_kind == ADP_AURA_INTERIORGS_PROBE_KIND
            aura_receipt_argument = (
                args.adp_aura_interiorgs_bundle_receipt
                if aura_interiorgs
                else args.adp_aura_bundle_receipt
            )
            missing = [
                name
                for name, value in (
                    (
                        "adp_aura_interiorgs_bundle_receipt"
                        if aura_interiorgs
                        else "adp_aura_bundle_receipt",
                        aura_receipt_argument,
                    ),
                    ("adp_job_dir", args.adp_job_dir),
                    (
                        "adp_aura_attempt_authority",
                        args.adp_aura_attempt_authority if aura_interiorgs else True,
                    ),
                )
                if not value
            ]
            control_blockers, control_identity = _control_plane_checkout_blockers()
            source_blockers, expected_source_commit = _adp_expected_source_commit_blockers(
                args.expected_source_commit or "", control_identity
            )
            blockers = [*missing, *control_blockers, *source_blockers]
            if args.provider != "vast":
                blockers.append("adp_aura_provider_must_be_vast")
            if not 0 < args.adp_max_hourly_rate_usd <= args.adp_max_spend_usd:
                blockers.append("adp_aura_budget_invalid")
            if any(value <= 0 for value in args.adp_allowed_active_vast_instance_id):
                blockers.append("adp_aura_allowed_active_vast_instance_id_invalid")
            minimum_aura_ttl = 7200 if aura_interiorgs else 5400
            if not minimum_aura_ttl <= args.adp_hard_ttl_seconds <= 14_400:
                blockers.append("adp_aura_hard_ttl_invalid")
            prepared_bundle: dict[str, Any] | None = None
            receipt_path: Path | None = None
            if aura_receipt_argument:
                receipt_path = Path(aura_receipt_argument).expanduser().resolve()
                if not receipt_path.is_file():
                    blockers.append("adp_aura_bundle_receipt_missing")
                else:
                    try:
                        prepared_bundle = _load(receipt_path)
                    except (OSError, ValueError, json.JSONDecodeError):
                        blockers.append("adp_aura_bundle_receipt_invalid")
            bundle_path: Path | None = None
            if prepared_bundle is not None:
                bundle_path = (
                    Path(str(prepared_bundle.get("bundle_path") or "")).expanduser().resolve()
                )
                observed_bundle_sha256 = (
                    "sha256:" + hashlib.sha256(bundle_path.read_bytes()).hexdigest()
                    if bundle_path.is_file()
                    else ""
                )
                if (
                    prepared_bundle.get("status") != "ready"
                    or prepared_bundle.get("source_commit") != ADP_AURA_SOURCE_COMMIT
                    or prepared_bundle.get("source_tree") != ADP_AURA_SOURCE_TREE
                    or prepared_bundle.get("prerequisite_receipt_digest")
                    != ADP_AURA_PREREQUISITE_RECEIPT_DIGEST
                    or prepared_bundle.get("container_image") != ADP_AURA_SMOKE_IMAGE
                    or prepared_bundle.get("retry_cap") != 0
                    or prepared_bundle.get("blockers") not in ([], None)
                    or (aura_interiorgs and not prepared_bundle.get("adapter_receipt_digest"))
                    or (
                        aura_interiorgs
                        and prepared_bundle.get("runtime_prerequisite_receipt_digest")
                        != ADP_AURA_RUNTIME_PREREQUISITE_RECEIPT_DIGEST
                    )
                    or (
                        aura_interiorgs
                        and prepared_bundle.get("blueprint_commit") != expected_source_commit
                    )
                    or not bundle_path.is_file()
                    or observed_bundle_sha256 != prepared_bundle.get("bundle_sha256")
                ):
                    blockers.append("adp_aura_bundle_binding_invalid")
            execution_authority: dict[str, Any] | None = None
            execution_authority_path: Path | None = None
            execution_authority_sha256: str | None = None
            if aura_interiorgs and args.adp_aura_attempt_authority:
                execution_authority_path = (
                    Path(args.adp_aura_attempt_authority).expanduser().resolve()
                )
                if not execution_authority_path.is_file():
                    blockers.append("adp_aura_attempt_authority_missing")
                elif prepared_bundle is None:
                    blockers.append("adp_aura_attempt_authority_bundle_missing")
                else:
                    try:
                        execution_authority = validate_aura_retry_authority_binding(
                            _load(execution_authority_path),
                            adapter_receipt_digest=str(
                                prepared_bundle.get("adapter_receipt_digest") or ""
                            ),
                            hard_cap_usd=args.adp_max_spend_usd,
                            hard_ttl_seconds=args.adp_hard_ttl_seconds,
                        )
                    except (
                        OSError,
                        ValueError,
                        json.JSONDecodeError,
                        PublicSceneExecutionAuthorityError,
                    ):
                        blockers.append("adp_aura_attempt_authority_invalid")
                    else:
                        execution_authority_sha256 = (
                            "sha256:"
                            + hashlib.sha256(execution_authority_path.read_bytes()).hexdigest()
                        )
            receipt_sha256 = (
                "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                if receipt_path and receipt_path.is_file()
                else None
            )
            avoidlist_path: Path | None = None
            avoidlist_sha256: str | None = None
            if args.adp_machine_avoidlist:
                avoidlist_path = Path(args.adp_machine_avoidlist).expanduser().resolve()
                try:
                    avoidlist = _load(avoidlist_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    blockers.append("adp_aura_machine_avoidlist_invalid")
                else:
                    if (
                        avoidlist.get("schema_version") != "vast_machine_avoidlist.v1"
                        or not isinstance(avoidlist.get("machine_ids"), list)
                        or any(
                            not isinstance(machine_id, int) or machine_id <= 0
                            for machine_id in avoidlist["machine_ids"]
                        )
                    ):
                        blockers.append("adp_aura_machine_avoidlist_invalid")
                    avoidlist_sha256 = (
                        "sha256:" + hashlib.sha256(avoidlist_path.read_bytes()).hexdigest()
                    )
            allocation_binding = {
                "program_id": "arm-decision-proof-v1",
                "probe_kind": args.probe_kind,
                "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
                "expected_source_commit": expected_source_commit or None,
                "bundle_receipt_sha256": receipt_sha256,
                "bundle_sha256": (
                    prepared_bundle.get("bundle_sha256") if prepared_bundle else None
                ),
                "aura_source_commit": ADP_AURA_SOURCE_COMMIT,
                "aura_source_tree": ADP_AURA_SOURCE_TREE,
                "prerequisite_receipt_digest": ADP_AURA_PREREQUISITE_RECEIPT_DIGEST,
                "runtime_prerequisite_receipt_digest": (
                    ADP_AURA_RUNTIME_PREREQUISITE_RECEIPT_DIGEST if aura_interiorgs else None
                ),
                "adapter_receipt_digest": (
                    prepared_bundle.get("adapter_receipt_digest")
                    if prepared_bundle and aura_interiorgs
                    else None
                ),
                "execution_authority_digest": (
                    execution_authority.get("authorization_digest") if execution_authority else None
                ),
                "execution_authority_file_sha256": execution_authority_sha256,
                "attempt_ordinal": (
                    execution_authority["attempt_authority"]["attempt_ordinal"]
                    if execution_authority
                    else None
                ),
                "container_image": ADP_AURA_SMOKE_IMAGE,
                "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                "hard_cap_usd": args.adp_max_spend_usd,
                "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                "machine_avoidlist_sha256": avoidlist_sha256,
                "allowed_active_vast_instance_ids": sorted(
                    set(args.adp_allowed_active_vast_instance_id)
                ),
                "retry_cap": 0,
            }
            allocation_binding_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(allocation_binding, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()
            )
            paid_admission = build_paid_lane_admission(
                resource_class="vast_provider_adapter", blockers=sorted(set(blockers))
            )
            paid_admission.update(
                {
                    "program_id": "arm-decision-proof-v1",
                    "probe_kind": args.probe_kind,
                    "control_plane_identity": control_identity,
                    "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                    "hard_cap_usd": args.adp_max_spend_usd,
                    "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                    "retry_cap": 0,
                    "explicit_concurrent_gpu_authority_bound": bool(
                        args.adp_allowed_active_vast_instance_id
                    ),
                    "authority": (
                        "user_authorized_all_in_scope_goal_resources_including_gpu_usage"
                    ),
                    "publisher_data_and_checkpoint_rights_bound": True,
                    "full_author_workflow_claimed": False,
                    "aura_interiorgs_full_native_workflow_authorized": aura_interiorgs,
                    "aura_inpaint_init_author_smoke_only": not aura_interiorgs,
                    "rendered_frames_have_no_hidden_background_truth": aura_interiorgs,
                    "output_claim_ceiling": (
                        "visual_candidate_only" if aura_interiorgs else "author_smoke_only"
                    ),
                    "allocation_binding": allocation_binding,
                    "allocation_binding_digest": allocation_binding_digest,
                }
            )
            write_json(Path(args.admission_out), paid_admission)
            grant = None
            if args.execute:
                try:
                    grant = require_paid_resource_admission(
                        paid_admission,
                        resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                    )
                except PaidResourceAdmissionBlocked as exc:
                    result = {
                        "status": "blocked",
                        "blockers": exc.blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.adapter_output), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
            if blockers or prepared_bundle is None:
                result = {
                    "status": "blocked",
                    "blockers": sorted(set(blockers)),
                    "provider_mutations_performed": 0,
                }
            else:
                aura_runner = (
                    run_aura_interiorgs_vast if aura_interiorgs else run_aura_author_smoke_vast
                )
                result = aura_runner(
                    job_dir=args.adp_job_dir,
                    paid_resource_admission_grant=grant,
                    execute=args.execute,
                    prepared_bundle=prepared_bundle,
                    max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                    hard_cap_usd=args.adp_max_spend_usd,
                    hard_ttl_seconds=args.adp_hard_ttl_seconds,
                    public_image=ADP_AURA_SMOKE_IMAGE,
                    machine_avoidlist_path=avoidlist_path,
                    allowed_active_instance_ids=args.adp_allowed_active_vast_instance_id,
                )
            write_json(Path(args.adapter_output), result)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == ADP_AURA_EXACT_RESIDUAL_PROBE_KIND:
            missing = [
                name
                for name, value in (
                    ("adp_aura_exact_residual_bundle_receipt", args.adp_aura_exact_residual_bundle_receipt),
                    ("adp_job_dir", args.adp_job_dir),
                )
                if not value
            ]
            control_blockers, control_identity = _control_plane_checkout_blockers()
            source_blockers, expected_source_commit = _adp_expected_source_commit_blockers(
                args.expected_source_commit or "", control_identity
            )
            blockers = [*missing, *control_blockers, *source_blockers]
            if args.provider != "vast":
                blockers.append("aura_exact_residual_provider_must_be_vast")
            if not 0 < args.adp_max_hourly_rate_usd <= args.adp_max_spend_usd:
                blockers.append("aura_exact_residual_budget_invalid")
            if args.adp_max_spend_usd > AURA_EXACT_RESIDUAL_MAX_HARD_CAP_USD:
                blockers.append("aura_exact_residual_hard_cap_exceeds_authority")
            if not 7200 <= args.adp_hard_ttl_seconds <= AURA_EXACT_RESIDUAL_MAX_TTL_SECONDS:
                blockers.append("aura_exact_residual_hard_ttl_invalid")
            if args.adp_hard_ttl_seconds * args.adp_max_hourly_rate_usd / 3600 > args.adp_max_spend_usd:
                blockers.append("aura_exact_residual_runtime_cost_exceeds_hard_cap")
            if any(value <= 0 for value in args.adp_allowed_active_vast_instance_id):
                blockers.append("aura_exact_residual_allowed_active_vast_instance_id_invalid")
            prepared_bundle: dict[str, Any] | None = None
            paid_attempt_authority: dict[str, Any] | None = None
            paid_attempt_authority_path: Path | None = None
            paid_attempt_authority_sha256: str | None = None
            if args.adp_aura_exact_residual_bundle_receipt:
                try:
                    prepared_bundle = validate_aura_exact_residual_bundle(
                        args.adp_aura_exact_residual_bundle_receipt
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    blockers.append("aura_exact_residual_bundle_binding_invalid")
            if prepared_bundle is not None:
                expected_allowed = prepared_bundle["allowed_active_instance_ids"]
                observed_allowed = sorted(set(args.adp_allowed_active_vast_instance_id))
                if observed_allowed != expected_allowed:
                    blockers.append("aura_exact_residual_external_instance_allowlist_mismatch")
            if args.execute:
                if not args.adp_aura_attempt_authority:
                    blockers.append("aura_exact_residual_paid_attempt_authority_missing")
                elif prepared_bundle is None:
                    blockers.append("aura_exact_residual_paid_attempt_authority_bundle_missing")
                else:
                    paid_attempt_authority_path = (
                        Path(args.adp_aura_attempt_authority).expanduser().resolve()
                    )
                    try:
                        raw_authority = _load(paid_attempt_authority_path)
                        paid_attempt_authority = (
                            validate_aura_exact_residual_paid_attempt_authority(
                                raw_authority,
                                prepared_bundle=prepared_bundle,
                                max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                                hard_cap_usd=args.adp_max_spend_usd,
                                hard_ttl_seconds=args.adp_hard_ttl_seconds,
                                allowed_active_instance_ids=expected_allowed,
                            )
                        )
                        paid_attempt_authority_sha256 = (
                            "sha256:"
                            + hashlib.sha256(paid_attempt_authority_path.read_bytes()).hexdigest()
                        )
                    except (OSError, ValueError, json.JSONDecodeError):
                        blockers.append("aura_exact_residual_paid_attempt_authority_invalid")
            avoidlist_path: Path | None = None
            avoidlist_sha256: str | None = None
            if args.adp_machine_avoidlist:
                avoidlist_path = Path(args.adp_machine_avoidlist).expanduser().resolve()
                try:
                    avoidlist = _load(avoidlist_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    blockers.append("aura_exact_residual_machine_avoidlist_invalid")
                else:
                    if (
                        avoidlist.get("schema_version") != "vast_machine_avoidlist.v1"
                        or not isinstance(avoidlist.get("machine_ids"), list)
                        or any(
                            not isinstance(machine_id, int) or machine_id <= 0
                            for machine_id in avoidlist["machine_ids"]
                        )
                    ):
                        blockers.append("aura_exact_residual_machine_avoidlist_invalid")
                    avoidlist_sha256 = "sha256:" + hashlib.sha256(avoidlist_path.read_bytes()).hexdigest()
            allocation_binding = {
                "program_id": "arm-decision-proof-v1",
                "probe_kind": args.probe_kind,
                "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
                "expected_source_commit": expected_source_commit or None,
                "bundle_receipt_sha256": prepared_bundle.get("receipt_sha256") if prepared_bundle else None,
                "bundle_sha256": prepared_bundle.get("bundle_sha256") if prepared_bundle else None,
                "preflight_digest": prepared_bundle.get("preflight_digest") if prepared_bundle else None,
                "execution_authority_digest": prepared_bundle.get("execution_authority_digest") if prepared_bundle else None,
                "paid_attempt_authority_digest": (
                    paid_attempt_authority.get("authorization_digest") if paid_attempt_authority else None
                ),
                "paid_attempt_authority_file_sha256": paid_attempt_authority_sha256,
                "container_image": prepared_bundle.get("container_image") if prepared_bundle else None,
                "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                "hard_cap_usd": args.adp_max_spend_usd,
                "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                "allowed_active_vast_instance_ids": sorted(set(args.adp_allowed_active_vast_instance_id)),
                "machine_avoidlist_sha256": avoidlist_sha256,
                "retry_cap": 0,
            }
            allocation_binding_digest = "sha256:" + hashlib.sha256(
                json.dumps(allocation_binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            paid_admission = build_paid_lane_admission(
                resource_class="vast_provider_adapter", blockers=sorted(set(blockers))
            )
            paid_admission.update(
                {
                    "program_id": "arm-decision-proof-v1",
                    "probe_kind": args.probe_kind,
                    "control_plane_identity": control_identity,
                    "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                    "hard_cap_usd": args.adp_max_spend_usd,
                    "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                    "retry_cap": 0,
                    "private_derived_upload_only": True,
                    "raw_interiorgs_upload_authorized": False,
                    "provider_training_authorized": False,
                    "inpaint360_code_or_author_data_included": False,
                    "exact_mask_only_edits_required": True,
                    "allocation_binding": allocation_binding,
                    "allocation_binding_digest": allocation_binding_digest,
                }
            )
            write_json(Path(args.admission_out), paid_admission)
            grant = None
            if args.execute:
                try:
                    grant = require_paid_resource_admission(
                        paid_admission,
                        resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                    )
                except PaidResourceAdmissionBlocked as exc:
                    result = {"status": "blocked", "blockers": exc.blockers, "provider_mutations_performed": 0}
                    write_json(Path(args.adapter_output), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
            if blockers or prepared_bundle is None:
                result = {"status": "blocked", "blockers": sorted(set(blockers)), "provider_mutations_performed": 0}
            else:
                result = run_aura_exact_residual_vast(
                    job_dir=args.adp_job_dir,
                    paid_resource_admission_grant=grant,
                    execute=args.execute,
                    prepared_bundle=prepared_bundle,
                    max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                    hard_cap_usd=args.adp_max_spend_usd,
                    hard_ttl_seconds=args.adp_hard_ttl_seconds,
                    machine_avoidlist_path=avoidlist_path,
                    paid_attempt_authority=paid_attempt_authority,
                )
            write_json(Path(args.adapter_output), result)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == ADP_ARTIFIXER3D_PROBE_KIND:
            missing = [
                name
                for name, value in (
                    (
                        "adp_artifixer3d_bundle_receipt",
                        args.adp_artifixer3d_bundle_receipt,
                    ),
                    ("adp_job_dir", args.adp_job_dir),
                    ("admission_out", args.admission_out),
                    ("adapter_output", args.adapter_output),
                )
                if not value
            ]
            control_blockers, control_identity = _control_plane_checkout_blockers()
            source_blockers, expected_source_commit = _adp_expected_source_commit_blockers(
                args.expected_source_commit or "", control_identity
            )
            blockers = [*missing, *control_blockers, *source_blockers]
            if args.provider != "vast":
                blockers.append("artifixer3d_provider_must_be_vast")
            if not 0 < args.adp_max_hourly_rate_usd <= args.adp_max_spend_usd:
                blockers.append("artifixer3d_budget_invalid")
            if args.adp_max_spend_usd > ARTIFIXER3D_MAX_HARD_CAP_USD:
                blockers.append("artifixer3d_hard_cap_exceeds_authority")
            if not 7200 <= args.adp_hard_ttl_seconds <= ARTIFIXER3D_MAX_TTL_SECONDS:
                blockers.append("artifixer3d_hard_ttl_invalid")
            if (
                args.adp_hard_ttl_seconds
                * args.adp_max_hourly_rate_usd
                / 3600
                > args.adp_max_spend_usd
            ):
                blockers.append("artifixer3d_runtime_cost_exceeds_hard_cap")
            if any(value <= 0 for value in args.adp_allowed_active_vast_instance_id):
                blockers.append("artifixer3d_allowed_active_vast_instance_id_invalid")
            prepared_bundle: dict[str, Any] | None = None
            paid_attempt_authority: dict[str, Any] | None = None
            authority_path: Path | None = None
            authority_sha256: str | None = None
            if args.adp_artifixer3d_bundle_receipt:
                try:
                    prepared_bundle = validate_artifixer3d_bundle(
                        args.adp_artifixer3d_bundle_receipt
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    blockers.append("artifixer3d_bundle_binding_invalid")
            if prepared_bundle is not None:
                observed_allowed = sorted(
                    set(args.adp_allowed_active_vast_instance_id)
                )
                if observed_allowed != prepared_bundle["allowed_active_instance_ids"]:
                    blockers.append("artifixer3d_external_instance_allowlist_mismatch")
                if set(observed_allowed) & set(
                    prepared_bundle["forbidden_external_instance_ids"]
                ):
                    blockers.append("artifixer3d_forbidden_external_instance_allowed")
                if (
                    expected_source_commit
                    and prepared_bundle["blueprint_source_identity"]["commit"]
                    != expected_source_commit
                ):
                    blockers.append("artifixer3d_bundle_source_commit_mismatch")
            if args.execute:
                if not args.adp_artifixer3d_attempt_authority:
                    blockers.append("artifixer3d_paid_attempt_authority_missing")
                elif prepared_bundle is None:
                    blockers.append("artifixer3d_paid_attempt_authority_bundle_missing")
                else:
                    authority_path = Path(
                        args.adp_artifixer3d_attempt_authority
                    ).expanduser().resolve()
                    try:
                        raw_authority = _load(authority_path)
                        paid_attempt_authority = (
                            validate_artifixer3d_paid_attempt_authority(
                                raw_authority,
                                prepared_bundle=prepared_bundle,
                                max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                                hard_cap_usd=args.adp_max_spend_usd,
                                hard_ttl_seconds=args.adp_hard_ttl_seconds,
                                allowed_active_instance_ids=prepared_bundle[
                                    "allowed_active_instance_ids"
                                ],
                            )
                        )
                        authority_sha256 = "sha256:" + hashlib.sha256(
                            authority_path.read_bytes()
                        ).hexdigest()
                    except (OSError, ValueError, json.JSONDecodeError):
                        blockers.append("artifixer3d_paid_attempt_authority_invalid")
            avoidlist_path: Path | None = None
            avoidlist_sha256: str | None = None
            if args.adp_machine_avoidlist:
                avoidlist_path = Path(args.adp_machine_avoidlist).expanduser().resolve()
                try:
                    avoidlist = _load(avoidlist_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    blockers.append("artifixer3d_machine_avoidlist_invalid")
                else:
                    if (
                        avoidlist.get("schema_version") != "vast_machine_avoidlist.v1"
                        or not isinstance(avoidlist.get("machine_ids"), list)
                        or any(
                            not isinstance(machine_id, int) or machine_id <= 0
                            for machine_id in avoidlist["machine_ids"]
                        )
                    ):
                        blockers.append("artifixer3d_machine_avoidlist_invalid")
                    avoidlist_sha256 = "sha256:" + hashlib.sha256(
                        avoidlist_path.read_bytes()
                    ).hexdigest()
            allocation_binding = {
                "program_id": "arm-decision-proof-v1",
                "probe_kind": args.probe_kind,
                "orchestrator_source_commit": control_identity.get(
                    "orchestrator_source_commit"
                ),
                "expected_source_commit": expected_source_commit or None,
                "bundle_receipt_sha256": (
                    prepared_bundle.get("receipt_sha256")
                    if prepared_bundle
                    else None
                ),
                "bundle_sha256": (
                    prepared_bundle.get("bundle_sha256")
                    if prepared_bundle
                    else None
                ),
                "manifest_digest": (
                    prepared_bundle.get("manifest_digest")
                    if prepared_bundle
                    else None
                ),
                "runtime_request_digest": (
                    prepared_bundle.get("runtime_request_digest")
                    if prepared_bundle
                    else None
                ),
                "paid_attempt_authority_digest": (
                    paid_attempt_authority.get("authorization_digest")
                    if paid_attempt_authority
                    else None
                ),
                "paid_attempt_authority_file_sha256": authority_sha256,
                "container_image": (
                    prepared_bundle.get("container_image")
                    if prepared_bundle
                    else None
                ),
                "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                "hard_cap_usd": args.adp_max_spend_usd,
                "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                "allowed_active_vast_instance_ids": sorted(
                    set(args.adp_allowed_active_vast_instance_id)
                ),
                "forbidden_external_instance_ids": (
                    prepared_bundle.get("forbidden_external_instance_ids")
                    if prepared_bundle
                    else None
                ),
                "machine_avoidlist_sha256": avoidlist_sha256,
                "retry_cap": 0,
            }
            allocation_binding_digest = "sha256:" + hashlib.sha256(
                json.dumps(
                    allocation_binding, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            paid_admission = build_paid_lane_admission(
                resource_class="vast_provider_adapter",
                blockers=sorted(set(blockers)),
            )
            paid_admission.update(
                {
                    "program_id": "arm-decision-proof-v1",
                    "probe_kind": args.probe_kind,
                    "control_plane_identity": control_identity,
                    "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                    "hard_cap_usd": args.adp_max_spend_usd,
                    "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                    "retry_cap": 0,
                    "private_derived_upload_only": True,
                    "raw_interiorgs_upload_authorized": False,
                    "raw_dataset_bytes_upload_authorized": False,
                    "provider_training_authorized": False,
                    "commercial_use_authorized": False,
                    "internal_noncommercial_research_only": True,
                    "exact_mask_only_edits_required": True,
                    "source_object_restoration_authorized": False,
                    "generated_output_is_physical_evidence": False,
                    "allocation_binding": allocation_binding,
                    "allocation_binding_digest": allocation_binding_digest,
                }
            )
            if args.admission_out:
                write_json(Path(args.admission_out), paid_admission)
            grant = None
            if args.execute:
                try:
                    grant = require_paid_resource_admission(
                        paid_admission,
                        resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                    )
                except PaidResourceAdmissionBlocked as exc:
                    result = {
                        "status": "blocked",
                        "blockers": exc.blockers,
                        "provider_mutations_performed": 0,
                    }
                    if args.adapter_output:
                        write_json(Path(args.adapter_output), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
            if blockers or prepared_bundle is None:
                result = {
                    "status": "blocked",
                    "blockers": sorted(set(blockers)),
                    "provider_mutations_performed": 0,
                }
            else:
                result = run_artifixer3d_vast(
                    job_dir=args.adp_job_dir,
                    paid_resource_admission_grant=grant,
                    execute=args.execute,
                    prepared_bundle=prepared_bundle,
                    max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                    hard_cap_usd=args.adp_max_spend_usd,
                    hard_ttl_seconds=args.adp_hard_ttl_seconds,
                    machine_avoidlist_path=avoidlist_path,
                    paid_attempt_authority=paid_attempt_authority,
                )
            if args.adapter_output:
                write_json(Path(args.adapter_output), result)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == ADP_CONTENT_AGENTS_PROBE_KIND:
            missing = [
                name
                for name in (
                    "adp_content_agents_bundle_receipt",
                    "adp_content_agents_config_preflight_receipt",
                    "adp_job_dir",
                )
                if not getattr(args, name, None)
            ]
            control_blockers, control_identity = _control_plane_checkout_blockers()
            blockers = [*missing, *control_blockers]
            if args.provider != "vast":
                blockers.append("adp_content_agents_provider_must_be_vast")
            if not 0 < args.adp_max_hourly_rate_usd <= args.adp_max_spend_usd:
                blockers.append("adp_content_agents_budget_invalid")
            if not 2700 <= args.adp_hard_ttl_seconds <= 14_400:
                blockers.append("adp_content_agents_hard_ttl_invalid")
            prepared_bundle: dict[str, Any] | None = None
            receipt_path: Path | None = None
            if args.adp_content_agents_bundle_receipt:
                receipt_path = Path(args.adp_content_agents_bundle_receipt).expanduser().resolve()
                if not receipt_path.is_file():
                    blockers.append("adp_content_agents_bundle_receipt_missing")
                else:
                    try:
                        prepared_bundle = _load(receipt_path)
                    except (OSError, ValueError, json.JSONDecodeError):
                        blockers.append("adp_content_agents_bundle_receipt_invalid")
            bundle_path: Path | None = None
            if prepared_bundle is not None:
                raw_bundle_path = str(prepared_bundle.get("bundle_path") or "")
                bundle_path = Path(raw_bundle_path).expanduser().resolve()
                expected_bundle_sha256 = str(prepared_bundle.get("bundle_sha256") or "")
                observed_bundle_sha256 = (
                    "sha256:" + hashlib.sha256(bundle_path.read_bytes()).hexdigest()
                    if bundle_path.is_file()
                    else ""
                )
                if (
                    prepared_bundle.get("status") != "ready"
                    or prepared_bundle.get("source_commit") != ADP_CONTENT_AGENTS_SOURCE_COMMIT
                    or prepared_bundle.get("source_tree") != ADP_CONTENT_AGENTS_SOURCE_TREE
                    or prepared_bundle.get("container_image") != ADP_CONTENT_AGENTS_IMAGE
                    or prepared_bundle.get("retry_cap") != 0
                    or prepared_bundle.get("blockers") not in ([], None)
                    or not bundle_path.is_file()
                    or observed_bundle_sha256 != expected_bundle_sha256
                ):
                    blockers.append("adp_content_agents_bundle_binding_invalid")
            receipt_sha256 = (
                "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                if receipt_path and receipt_path.is_file()
                else None
            )
            config_preflight: dict[str, Any] | None = None
            config_preflight_path: Path | None = None
            if args.adp_content_agents_config_preflight_receipt:
                config_preflight_path = (
                    Path(args.adp_content_agents_config_preflight_receipt).expanduser().resolve()
                )
                if not config_preflight_path.is_file():
                    blockers.append("adp_content_agents_config_preflight_receipt_missing")
                else:
                    try:
                        config_preflight = _load(config_preflight_path)
                    except (OSError, ValueError, json.JSONDecodeError):
                        blockers.append("adp_content_agents_config_preflight_receipt_invalid")
            if config_preflight is not None and prepared_bundle is not None:
                blockers.extend(
                    validate_bundle_config_preflight(
                        preflight=config_preflight,
                        prepared_bundle=prepared_bundle,
                        preflight_receipt_path=config_preflight_path,
                        expected_orchestrator_source_commit=str(
                            control_identity.get("orchestrator_source_commit") or ""
                        ),
                    )
                )
                if config_preflight.get("bundle_receipt_sha256") != receipt_sha256:
                    blockers.append("adp_content_agents_config_preflight_bundle_receipt_mismatch")
            config_preflight_receipt_sha256 = (
                "sha256:" + hashlib.sha256(config_preflight_path.read_bytes()).hexdigest()
                if config_preflight_path and config_preflight_path.is_file()
                else None
            )
            paid_attempt_authority: dict[str, Any] | None = None
            if args.adp_content_agents_attempt_authority:
                try:
                    paid_attempt_authority = _load(
                        Path(args.adp_content_agents_attempt_authority).expanduser().resolve()
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    blockers.append("adp_content_agents_paid_attempt_authority_invalid")
            elif args.execute:
                blockers.append("adp_content_agents_paid_attempt_authority_missing")
            if (
                paid_attempt_authority is not None
                and prepared_bundle is not None
                and config_preflight is not None
            ):
                try:
                    validate_content_agents_paid_attempt_authority(
                        paid_attempt_authority,
                        prepared_bundle=prepared_bundle,
                        bundle_receipt_sha256=receipt_sha256,
                        config_preflight=config_preflight,
                        config_preflight_receipt_sha256=(config_preflight_receipt_sha256),
                        max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                        hard_cap_usd=args.adp_max_spend_usd,
                        hard_ttl_seconds=args.adp_hard_ttl_seconds,
                        allowed_active_instance_ids=args.adp_allowed_active_vast_instance_id,
                    )
                except ValueError as exc:
                    blockers.append(str(exc))
            allocation_binding = {
                "program_id": "arm-decision-proof-v1",
                "probe_kind": ADP_CONTENT_AGENTS_PROBE_KIND,
                "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
                "bundle_receipt_sha256": receipt_sha256,
                "config_preflight_receipt_sha256": config_preflight_receipt_sha256,
                "config_preflight_receipt_digest": (
                    config_preflight.get("receipt_digest") if config_preflight else None
                ),
                "paid_attempt_authority_digest": (
                    paid_attempt_authority.get("authorization_digest")
                    if paid_attempt_authority
                    else None
                ),
                "bundle_sha256": (
                    prepared_bundle.get("bundle_sha256") if prepared_bundle else None
                ),
                "content_agents_source_commit": ADP_CONTENT_AGENTS_SOURCE_COMMIT,
                "content_agents_source_tree": ADP_CONTENT_AGENTS_SOURCE_TREE,
                "container_image": ADP_CONTENT_AGENTS_IMAGE,
                "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                "hard_cap_usd": args.adp_max_spend_usd,
                "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                "allowed_active_vast_instance_ids": sorted(
                    set(args.adp_allowed_active_vast_instance_id)
                ),
                "retry_cap": 0,
            }
            allocation_binding_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(allocation_binding, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()
            )
            paid_admission = build_paid_lane_admission(
                resource_class="vast_provider_adapter", blockers=sorted(set(blockers))
            )
            public_sage_collision_uploaded = bool(
                prepared_bundle
                and (prepared_bundle.get("native_probe") or {}).get("sage_collision_sha256")
            )
            content_agents_authority_state = (
                "explicit_content_agents_paid_attempt_authority_bound"
                if paid_attempt_authority is not None
                else "missing_content_agents_paid_attempt_authority"
                if args.execute
                else "dry_run_no_paid_authority_required"
            )
            paid_admission.update(
                {
                    "program_id": "arm-decision-proof-v1",
                    "probe_kind": ADP_CONTENT_AGENTS_PROBE_KIND,
                    "control_plane_identity": control_identity,
                    "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                    "hard_cap_usd": args.adp_max_spend_usd,
                    "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                    "retry_cap": 0,
                    "authority": content_agents_authority_state,
                    "paid_attempt_authority_required_for_execute": True,
                    "private_or_licensed_dataset_bytes_uploaded": (public_sage_collision_uploaded),
                    "private_or_gated_dataset_bytes_uploaded": False,
                    "public_licensed_sage_collision_bytes_uploaded": (
                        public_sage_collision_uploaded
                    ),
                    "public_licensed_dataset_identity": (
                        {
                            "repository": "spatialverse/SAGE-3D_Collision_Mesh",
                            "license": "CC-BY-NC-4.0",
                            "use_ceiling": "internal_noncommercial_validation",
                            "raw_bytes_redistributed": False,
                        }
                        if public_sage_collision_uploaded
                        else None
                    ),
                    "input_is_blueprint_owned_parametric_control": not (
                        public_sage_collision_uploaded
                    ),
                    "input_contains_blueprint_owned_parametric_control": True,
                    "allocation_binding": allocation_binding,
                    "allocation_binding_digest": allocation_binding_digest,
                }
            )
            write_json(Path(args.admission_out), paid_admission)
            grant = None
            if args.execute:
                try:
                    grant = require_paid_resource_admission(
                        paid_admission,
                        resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                    )
                except PaidResourceAdmissionBlocked as exc:
                    result = {
                        "status": "blocked",
                        "blockers": exc.blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.adapter_output), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
                if paid_attempt_authority is not None:
                    consumption = consume_content_agents_paid_attempt_authority_once(
                        paid_attempt_authority,
                        blueprint_commit=str(
                            control_identity.get("orchestrator_source_commit") or ""
                        ),
                    )
                    if consumption.get("status") != "consumed":
                        result = {
                            "status": "blocked",
                            "blockers": consumption.get("blockers")
                            or ["adp_content_agents_paid_attempt_authority_not_consumed"],
                            "provider_mutations_performed": 0,
                            "authorization_consumption": consumption,
                        }
                        write_json(Path(args.adapter_output), result)
                        print(json.dumps({"success": False}, sort_keys=True))
                        return 2
            if blockers or prepared_bundle is None or config_preflight is None:
                result = {
                    "status": "blocked",
                    "blockers": sorted(set(blockers)),
                    "provider_mutations_performed": 0,
                }
            else:
                result = run_content_agents_vast(
                    job_dir=args.adp_job_dir,
                    paid_resource_admission_grant=grant,
                    execute=args.execute,
                    prepared_bundle=prepared_bundle,
                    max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                    hard_cap_usd=args.adp_max_spend_usd,
                    hard_ttl_seconds=args.adp_hard_ttl_seconds,
                    public_image=ADP_CONTENT_AGENTS_IMAGE,
                    allowed_active_instance_ids=args.adp_allowed_active_vast_instance_id,
                )
            write_json(Path(args.adapter_output), result)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == ADP_SIMREADY_ISAAC_PROBE_KIND:
            missing = [
                name
                for name in ("adp_simready_isaac_bundle_receipt", "adp_job_dir")
                if not getattr(args, name, None)
            ]
            control_blockers, control_identity = _control_plane_checkout_blockers()
            source_blockers, expected_source_commit = _adp_expected_source_commit_blockers(
                args.expected_source_commit or "", control_identity
            )
            blockers = [*missing, *control_blockers, *source_blockers]
            if args.provider != "vast":
                blockers.append("simready_isaac_provider_must_be_vast")
            if not 0 < args.adp_max_hourly_rate_usd <= args.adp_max_spend_usd:
                blockers.append("simready_isaac_budget_invalid")
            if not 1800 <= args.adp_hard_ttl_seconds <= 14_400:
                blockers.append("simready_isaac_hard_ttl_invalid")
            prepared_bundle: dict[str, Any] | None = None
            receipt_path: Path | None = None
            if args.adp_simready_isaac_bundle_receipt:
                receipt_path = Path(args.adp_simready_isaac_bundle_receipt).expanduser().resolve()
                try:
                    prepared_bundle = _load(receipt_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    blockers.append("simready_isaac_bundle_receipt_invalid")
            bundle_path: Path | None = None
            if prepared_bundle is not None:
                bundle_path = (
                    Path(str(prepared_bundle.get("bundle_path") or "")).expanduser().resolve()
                )
                observed_bundle_sha256 = (
                    "sha256:" + hashlib.sha256(bundle_path.read_bytes()).hexdigest()
                    if bundle_path.is_file()
                    else ""
                )
                if (
                    prepared_bundle.get("status") != "ready"
                    or prepared_bundle.get("source_commit_sha") != expected_source_commit
                    or prepared_bundle.get("container_image") != ADP_SIMREADY_ISAAC_IMAGE
                    or prepared_bundle.get("retry_cap") != 0
                    or prepared_bundle.get("blockers") not in ([], None)
                    or not prepared_bundle.get("probe_spec_sha256")
                    or not bundle_path.is_file()
                    or observed_bundle_sha256 != prepared_bundle.get("bundle_sha256")
                ):
                    blockers.append("simready_isaac_bundle_binding_invalid")
            receipt_sha256 = (
                "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                if receipt_path and receipt_path.is_file()
                else None
            )
            paid_attempt_authority: dict[str, Any] | None = None
            if args.adp_simready_isaac_attempt_authority:
                try:
                    paid_attempt_authority = _load(
                        Path(args.adp_simready_isaac_attempt_authority).expanduser().resolve()
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    blockers.append("simready_isaac_paid_attempt_authority_invalid")
            elif args.execute:
                blockers.append("simready_isaac_paid_attempt_authority_missing")
            if paid_attempt_authority is not None and prepared_bundle is not None:
                try:
                    validate_simready_isaac_paid_attempt_authority(
                        paid_attempt_authority,
                        prepared_bundle=prepared_bundle,
                        bundle_receipt_sha256=receipt_sha256,
                        max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                        hard_cap_usd=args.adp_max_spend_usd,
                        hard_ttl_seconds=args.adp_hard_ttl_seconds,
                        allowed_active_instance_ids=args.adp_allowed_active_vast_instance_id,
                    )
                except ValueError:
                    blockers.append("simready_isaac_paid_attempt_authority_invalid")
            avoidlist_sha256 = None
            if args.adp_machine_avoidlist:
                avoidlist_path = Path(args.adp_machine_avoidlist).expanduser().resolve()
                if not avoidlist_path.is_file():
                    blockers.append("simready_isaac_machine_avoidlist_missing")
                else:
                    avoidlist_sha256 = (
                        "sha256:" + hashlib.sha256(avoidlist_path.read_bytes()).hexdigest()
                    )
            allocation_binding = {
                "program_id": "arm-decision-proof-v1",
                "probe_kind": ADP_SIMREADY_ISAAC_PROBE_KIND,
                "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
                "expected_source_commit": expected_source_commit or None,
                "bundle_receipt_sha256": receipt_sha256,
                "bundle_sha256": (
                    prepared_bundle.get("bundle_sha256") if prepared_bundle else None
                ),
                "probe_spec_sha256": (
                    prepared_bundle.get("probe_spec_sha256") if prepared_bundle else None
                ),
                "container_image": ADP_SIMREADY_ISAAC_IMAGE,
                "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                "hard_cap_usd": args.adp_max_spend_usd,
                "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                "allowed_active_vast_instance_ids": sorted(
                    set(args.adp_allowed_active_vast_instance_id)
                ),
                "machine_avoidlist_sha256": avoidlist_sha256,
                "retry_cap": 0,
                "paid_attempt_authority_digest": (
                    paid_attempt_authority.get("authorization_digest")
                    if paid_attempt_authority
                    else None
                ),
            }
            allocation_binding_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(allocation_binding, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()
            )
            paid_admission = build_paid_lane_admission(
                resource_class="vast_provider_adapter", blockers=sorted(set(blockers))
            )
            paid_admission.update(
                {
                    "program_id": "arm-decision-proof-v1",
                    "probe_kind": ADP_SIMREADY_ISAAC_PROBE_KIND,
                    "control_plane_identity": control_identity,
                    "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                    "hard_cap_usd": args.adp_max_spend_usd,
                    "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                    "retry_cap": 0,
                    "authority": "user_authorized_all_in_scope_goal_resources_including_gpu_usage",
                    "paid_attempt_authority_required_for_execute": True,
                    "paid_attempt_authority_digest": (
                        paid_attempt_authority.get("authorization_digest")
                        if paid_attempt_authority
                        else None
                    ),
                    "private_data_uploaded": False,
                    "physical_success_established": False,
                    "allocation_binding": allocation_binding,
                    "allocation_binding_digest": allocation_binding_digest,
                }
            )
            write_json(Path(args.admission_out), paid_admission)
            grant = None
            if args.execute:
                try:
                    grant = require_paid_resource_admission(
                        paid_admission,
                        resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                    )
                except PaidResourceAdmissionBlocked as exc:
                    result = {
                        "status": "blocked",
                        "blockers": sorted(set([*exc.blockers, *blockers])),
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.adapter_output), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
            if prepared_bundle is None or blockers:
                result = {
                    "status": "blocked",
                    "blockers": sorted(set(blockers)),
                    "provider_mutations_performed": 0,
                }
            else:
                result = run_simready_isaac_vast(
                    job_dir=args.adp_job_dir,
                    prepared_bundle=prepared_bundle,
                    paid_resource_admission_grant=grant,
                    paid_attempt_authority=paid_attempt_authority,
                    bundle_receipt_sha256=receipt_sha256,
                    execute=args.execute,
                    machine_avoidlist_path=args.adp_machine_avoidlist,
                    max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                    hard_cap_usd=args.adp_max_spend_usd,
                    hard_ttl_seconds=args.adp_hard_ttl_seconds,
                    allowed_active_instance_ids=args.adp_allowed_active_vast_instance_id,
                )
            write_json(Path(args.adapter_output), result)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == ADP009D_OVRTX_LIVE_CAMERA_PROBE_KIND:
            missing = [
                name
                for name in ("adp009d_ovrtx_probe_manifest", "adp_job_dir")
                if not getattr(args, name, None)
            ]
            control_blockers, control_identity = _control_plane_checkout_blockers()
            blockers = [*missing, *control_blockers]
            if args.provider != "vast":
                blockers.append("adp009d_ovrtx_provider_must_be_vast")
            if not 0 < args.adp_max_hourly_rate_usd <= args.adp_max_spend_usd:
                blockers.append("adp009d_ovrtx_budget_invalid")
            if not 1800 <= args.adp_hard_ttl_seconds <= 3600:
                blockers.append("adp009d_ovrtx_hard_ttl_invalid")
            avoidlist_digest = None
            if args.adp_machine_avoidlist:
                avoidlist_path = Path(args.adp_machine_avoidlist).expanduser().resolve()
                if not avoidlist_path.is_file():
                    blockers.append("adp009d_ovrtx_machine_avoidlist_missing")
                else:
                    avoidlist_digest = (
                        "sha256:" + hashlib.sha256(avoidlist_path.read_bytes()).hexdigest()
                    )
            prepared_bundle = None
            if not blockers:
                try:
                    prepared_bundle = build_ovrtx_live_camera_bundle(
                        job_dir=Path(args.adp_job_dir) / "bundle",
                        probe_manifest_path=args.adp009d_ovrtx_probe_manifest,
                        implementation_commit=control_identity["orchestrator_source_commit"],
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    blockers.append(f"adp009d_ovrtx_bundle_preparation_failed:{type(exc).__name__}")
            allocation_binding = {
                "program_id": "arm-decision-proof-v1",
                "probe_kind": ADP009D_OVRTX_LIVE_CAMERA_PROBE_KIND,
                "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
                "bundle_sha256": (
                    prepared_bundle.get("bundle_sha256") if prepared_bundle else None
                ),
                "input_digest": (prepared_bundle.get("input_digest") if prepared_bundle else None),
                "candidate_policy_queried": False,
                "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                "hard_cap_usd": args.adp_max_spend_usd,
                "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                "retry_cap": 0,
                "machine_avoidlist_digest": avoidlist_digest,
            }
            allocation_binding_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(allocation_binding, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()
            )
            paid_admission = build_paid_lane_admission(
                resource_class="vast_provider_adapter", blockers=blockers
            )
            paid_admission.update(
                {
                    "program_id": "arm-decision-proof-v1",
                    "probe_kind": ADP009D_OVRTX_LIVE_CAMERA_PROBE_KIND,
                    "control_plane_identity": control_identity,
                    "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                    "hard_cap_usd": args.adp_max_spend_usd,
                    "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                    "retry_cap": 0,
                    "authority": "user_authorized_bounded_gpu_compute_in_goal_scope",
                    "private_data_uploaded": False,
                    "candidate_policy_queried": False,
                    "physical_outcome_values_uploaded": False,
                    "allocation_binding": allocation_binding,
                    "allocation_binding_digest": allocation_binding_digest,
                }
            )
            write_json(Path(args.admission_out), paid_admission)
            grant = None
            if args.execute:
                try:
                    grant = require_paid_resource_admission(
                        paid_admission,
                        resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                    )
                except PaidResourceAdmissionBlocked as exc:
                    result = {
                        "status": "blocked",
                        "blockers": exc.blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.adapter_output), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
            if prepared_bundle is None:
                result = {
                    "status": "blocked",
                    "blockers": sorted(set(blockers)),
                    "provider_mutations_performed": 0,
                }
            else:
                result = run_ovrtx_live_camera_vast(
                    job_dir=args.adp_job_dir,
                    prepared_bundle=prepared_bundle,
                    paid_resource_admission_grant=grant,
                    execute=args.execute,
                    machine_avoidlist_path=args.adp_machine_avoidlist,
                    max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                    hard_cap_usd=args.adp_max_spend_usd,
                    hard_ttl_seconds=args.adp_hard_ttl_seconds,
                )
            write_json(Path(args.adapter_output), result)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == ADP009D_AURA_NATIVE_LIVE_CAMERA_PROBE_KIND:
            missing = [
                name
                for name in (
                    "adp009d_aura_native_probe_manifest",
                    "adp009d_aura_source_root",
                    "adp_job_dir",
                )
                if not getattr(args, name, None)
            ]
            control_blockers, control_identity = _control_plane_checkout_blockers()
            blockers = [*missing, *control_blockers]
            if args.provider != "vast":
                blockers.append("adp009d_aura_native_provider_must_be_vast")
            if not 0 < args.adp_max_hourly_rate_usd <= args.adp_max_spend_usd:
                blockers.append("adp009d_aura_native_budget_invalid")
            if not 1800 <= args.adp_hard_ttl_seconds <= 3600:
                blockers.append("adp009d_aura_native_hard_ttl_invalid")
            avoidlist_digest = None
            if args.adp_machine_avoidlist:
                avoidlist_path = Path(args.adp_machine_avoidlist).expanduser().resolve()
                if not avoidlist_path.is_file():
                    blockers.append("adp009d_aura_native_machine_avoidlist_missing")
                else:
                    avoidlist_digest = (
                        "sha256:" + hashlib.sha256(avoidlist_path.read_bytes()).hexdigest()
                    )
            prepared_bundle = None
            if not blockers:
                try:
                    prepared_bundle = build_aura_native_live_camera_bundle(
                        job_dir=Path(args.adp_job_dir) / "bundle",
                        probe_manifest_path=args.adp009d_aura_native_probe_manifest,
                        aura_root=args.adp009d_aura_source_root,
                        implementation_commit=control_identity["orchestrator_source_commit"],
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    blockers.append(
                        f"adp009d_aura_native_bundle_preparation_failed:{type(exc).__name__}"
                    )
            allocation_binding = {
                "program_id": "arm-decision-proof-v1",
                "probe_kind": ADP009D_AURA_NATIVE_LIVE_CAMERA_PROBE_KIND,
                "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
                "bundle_sha256": (
                    prepared_bundle.get("bundle_sha256") if prepared_bundle else None
                ),
                "input_digest": (prepared_bundle.get("input_digest") if prepared_bundle else None),
                "aura_source_commit": (
                    prepared_bundle.get("source_commit") if prepared_bundle else None
                ),
                "aura_source_tree": (
                    prepared_bundle.get("source_tree") if prepared_bundle else None
                ),
                "aura_ply_sha256": (
                    prepared_bundle.get("aura_ply_sha256") if prepared_bundle else None
                ),
                "candidate_policy_queried": False,
                "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                "hard_cap_usd": args.adp_max_spend_usd,
                "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                "retry_cap": 0,
                "machine_avoidlist_digest": avoidlist_digest,
            }
            allocation_binding_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(allocation_binding, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()
            )
            paid_admission = build_paid_lane_admission(
                resource_class="vast_provider_adapter", blockers=blockers
            )
            paid_admission.update(
                {
                    "program_id": "arm-decision-proof-v1",
                    "probe_kind": ADP009D_AURA_NATIVE_LIVE_CAMERA_PROBE_KIND,
                    "control_plane_identity": control_identity,
                    "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                    "hard_cap_usd": args.adp_max_spend_usd,
                    "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                    "retry_cap": 0,
                    "authority": "user_authorized_bounded_gpu_compute_in_goal_scope",
                    "private_data_uploaded": False,
                    "candidate_policy_queried": False,
                    "physical_outcome_values_uploaded": False,
                    "allocation_binding": allocation_binding,
                    "allocation_binding_digest": allocation_binding_digest,
                }
            )
            write_json(Path(args.admission_out), paid_admission)
            grant = None
            if args.execute:
                try:
                    grant = require_paid_resource_admission(
                        paid_admission,
                        resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                    )
                except PaidResourceAdmissionBlocked as exc:
                    result = {
                        "status": "blocked",
                        "blockers": exc.blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.adapter_output), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
            if prepared_bundle is None:
                result = {
                    "status": "blocked",
                    "blockers": sorted(set(blockers)),
                    "provider_mutations_performed": 0,
                }
            else:
                result = run_aura_native_live_camera_vast(
                    job_dir=args.adp_job_dir,
                    prepared_bundle=prepared_bundle,
                    paid_resource_admission_grant=grant,
                    execute=args.execute,
                    machine_avoidlist_path=args.adp_machine_avoidlist,
                    max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                    hard_cap_usd=args.adp_max_spend_usd,
                    hard_ttl_seconds=args.adp_hard_ttl_seconds,
                )
            write_json(Path(args.adapter_output), result)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind in {
            NATIVE_TASK_ARENA_CONSTRUCTION_PROBE_KIND,
            NATIVE_TASK_ARENA_CONTROLS_PROBE_KIND,
            NATIVE_TASK_ARENA_POLICY_PROBE_KIND,
        }:
            controls_requested = args.probe_kind == NATIVE_TASK_ARENA_CONTROLS_PROBE_KIND
            policy_requested = args.probe_kind == NATIVE_TASK_ARENA_POLICY_PROBE_KIND
            missing = [
                name
                for name in (
                    "native_task_arena_packet",
                    "native_task_arena_runtime_source_packet",
                    "adp_job_dir",
                )
                if not getattr(args, name, None)
            ]
            if (controls_requested or policy_requested) and not (
                args.native_task_arena_construction_result
            ):
                missing.append("native_task_arena_construction_result")
            if policy_requested and not args.native_task_arena_control_result:
                missing.append("native_task_arena_control_result")
            if policy_requested and not args.native_task_arena_policy_execution_spec:
                missing.append("native_task_arena_policy_execution_spec")
            control_blockers, control_identity = _control_plane_checkout_blockers()
            blockers = [*missing, *control_blockers]
            if args.execute and not args.native_task_arena_bundle_receipt:
                blockers.append("native_task_arena_execute_requires_dry_run_bundle_receipt")
            if args.provider != "vast":
                blockers.append("native_task_arena_provider_must_be_vast")
            if not 0 < args.adp_max_hourly_rate_usd <= args.adp_max_spend_usd:
                blockers.append("native_task_arena_budget_invalid")
            if not 1800 <= args.adp_hard_ttl_seconds <= 14_400:
                blockers.append("native_task_arena_hard_ttl_invalid")
            if any(value <= 0 for value in args.adp_allowed_active_vast_instance_id):
                blockers.append("native_task_arena_allowed_active_instance_id_invalid")
            avoidlist_digest = None
            if args.adp_machine_avoidlist:
                avoidlist_path = Path(args.adp_machine_avoidlist).expanduser().resolve()
                if not avoidlist_path.is_file():
                    blockers.append("native_task_arena_machine_avoidlist_missing")
                else:
                    avoidlist_digest = (
                        "sha256:" + hashlib.sha256(avoidlist_path.read_bytes()).hexdigest()
                    )
            prepared_bundle = None
            if not blockers:
                try:
                    source_packet = verify_native_task_runtime_source_packet(
                        args.native_task_arena_runtime_source_packet
                    )
                    if args.native_task_arena_bundle_receipt:
                        packet_receipt_path = (
                            Path(args.native_task_arena_packet).expanduser().resolve()
                            / "native_task_arena_packet_receipt.v1.json"
                        )
                        packet_receipt = json.loads(packet_receipt_path.read_text(encoding="utf-8"))
                        bundle_loader = (
                            load_verified_native_task_arena_policy_bundle
                            if policy_requested
                            else load_verified_native_task_arena_controls_bundle
                            if controls_requested
                            else load_verified_native_task_arena_construction_bundle
                        )
                        prepared_bundle = bundle_loader(
                            args.native_task_arena_bundle_receipt,
                            expected_implementation_commit=control_identity[
                                "orchestrator_source_commit"
                            ],
                            expected_packet_receipt_digest=packet_receipt.get("receipt_digest"),
                            expected_runtime_source_packet_digest=source_packet.get(
                                "receipt_digest"
                            ),
                        )
                    else:
                        bundle_kwargs = {
                            "job_dir": Path(args.adp_job_dir) / "bundle",
                            "packet_dir": args.native_task_arena_packet,
                            "runtime_source_packet_receipt": (
                                args.native_task_arena_runtime_source_packet
                            ),
                            "implementation_commit": control_identity["orchestrator_source_commit"],
                        }
                        prepared_bundle = (
                            build_native_task_arena_policy_bundle(
                                **bundle_kwargs,
                                construction_result_path=(
                                    args.native_task_arena_construction_result
                                ),
                                control_result_path=(args.native_task_arena_control_result),
                                policy_execution_spec=json.loads(
                                    Path(args.native_task_arena_policy_execution_spec).read_text(
                                        encoding="utf-8"
                                    )
                                ),
                            )
                            if policy_requested
                            else build_native_task_arena_controls_bundle(
                                **bundle_kwargs,
                                construction_result_path=(
                                    args.native_task_arena_construction_result
                                ),
                            )
                            if controls_requested
                            else build_native_task_arena_construction_bundle(**bundle_kwargs)
                        )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    blockers.append(
                        f"native_task_arena_bundle_preparation_failed:{type(exc).__name__}"
                    )
            allocation_binding = {
                "program_id": "arm-decision-proof-v1",
                "probe_kind": args.probe_kind,
                "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
                "bundle_sha256": (
                    prepared_bundle.get("bundle_sha256") if prepared_bundle else None
                ),
                "input_digest": (prepared_bundle.get("input_digest") if prepared_bundle else None),
                "packet_receipt_digest": (
                    prepared_bundle.get("packet_receipt_digest") if prepared_bundle else None
                ),
                "runtime_source_packet_receipt_digest": (
                    (prepared_bundle.get("runtime_source_packet") or {}).get("receipt_digest")
                    if prepared_bundle
                    else None
                ),
                "runtime_source_packet_sha256": (
                    (prepared_bundle.get("runtime_source_packet") or {}).get("packet_sha256")
                    if prepared_bundle
                    else None
                ),
                "arena_scene_plan_digest": (
                    prepared_bundle.get("arena_scene_plan_digest") if prepared_bundle else None
                ),
                "runtime_contract_digest": (
                    prepared_bundle.get("runtime_contract_digest") if prepared_bundle else None
                ),
                "scenario_instance_digest": (
                    prepared_bundle.get("scenario_instance_digest") if prepared_bundle else None
                ),
                "execution_mode": (
                    "policy"
                    if policy_requested
                    else "controls"
                    if controls_requested
                    else "construction_canary"
                ),
                "candidate_policy_queried": policy_requested,
                "policy_candidate_id": (
                    prepared_bundle.get("policy_candidate_id") if prepared_bundle else None
                ),
                "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                "hard_cap_usd": args.adp_max_spend_usd,
                "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                "retry_cap": 0,
                "machine_avoidlist_digest": avoidlist_digest,
                "allowed_active_vast_instance_ids": sorted(
                    set(args.adp_allowed_active_vast_instance_id)
                ),
            }
            allocation_binding_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(
                        allocation_binding,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
            )
            paid_admission = build_paid_lane_admission(
                resource_class="vast_provider_adapter", blockers=blockers
            )
            paid_admission.update(
                {
                    "program_id": "arm-decision-proof-v1",
                    "probe_kind": args.probe_kind,
                    "control_plane_identity": control_identity,
                    "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                    "hard_cap_usd": args.adp_max_spend_usd,
                    "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                    "retry_cap": 0,
                    "authority": (
                        "user_authorized_private_scene_derived_upload_and_bounded_gpu_compute"
                    ),
                    "private_data_uploaded": True,
                    "raw_dataset_bytes_uploaded": False,
                    "candidate_policy_queried": policy_requested,
                    "physical_outcome_values_uploaded": False,
                    "explicit_concurrent_gpu_authority_bound": bool(
                        args.adp_allowed_active_vast_instance_id
                    ),
                    "allocation_binding": allocation_binding,
                    "allocation_binding_digest": allocation_binding_digest,
                }
            )
            write_json(Path(args.admission_out), paid_admission)
            grant = None
            if args.execute:
                try:
                    grant = require_paid_resource_admission(
                        paid_admission,
                        resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                    )
                except PaidResourceAdmissionBlocked as exc:
                    result = {
                        "status": "blocked",
                        "blockers": exc.blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.adapter_output), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
            if prepared_bundle is None:
                result = {
                    "status": "blocked",
                    "blockers": sorted(set(blockers)),
                    "provider_mutations_performed": 0,
                }
            else:
                run_native = (
                    run_native_task_arena_policy_vast
                    if policy_requested
                    else run_native_task_arena_controls_vast
                    if controls_requested
                    else run_native_task_arena_vast
                )
                result = run_native(
                    job_dir=args.adp_job_dir,
                    prepared_bundle=prepared_bundle,
                    paid_resource_admission_grant=grant,
                    execute=args.execute,
                    machine_avoidlist_path=args.adp_machine_avoidlist,
                    max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                    hard_cap_usd=args.adp_max_spend_usd,
                    hard_ttl_seconds=args.adp_hard_ttl_seconds,
                    allowed_active_instance_ids=(args.adp_allowed_active_vast_instance_id),
                )
            write_json(Path(args.adapter_output), result)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == ADP009D_NATIVE_MICROCHECK_PROBE_KIND:
            articulated_native_requested = bool(
                args.adp009d_articulated_diagnostic_asset
                or args.adp009d_articulated_diagnostic_request
            )
            missing = [
                name
                for name in (
                    "adp009d_harness_manifest",
                    "adp_job_dir",
                )
                if not getattr(args, name, None)
            ]
            if articulated_native_requested:
                missing.extend(
                    name
                    for name in (
                        "adp009d_articulated_diagnostic_asset",
                        "adp009d_articulated_diagnostic_request",
                    )
                    if not getattr(args, name, None)
                )
            else:
                missing.extend(
                    name
                    for name in (
                        "adp009d_approved_can",
                        "adp009d_sage_collision",
                    )
                    if not getattr(args, name, None)
                )
            control_blockers, control_identity = _control_plane_checkout_blockers()
            blockers = [*missing, *control_blockers]
            if args.adp009d_controls and not args.adp009d_scenario_instance:
                blockers.append("adp009d_control_scenario_instance_missing")
            selected_candidates = {
                item.strip()
                for item in str(args.adp009d_policy_candidate or "").split(",")
                if item.strip()
            }
            execution_modes = sum(
                (
                    bool(selected_candidates),
                    bool(args.adp009d_controls),
                    bool(args.adp009d_diagnostic_only),
                )
            )
            if execution_modes == 0:
                blockers.append("adp009d_execution_mode_missing")
            elif args.adp009d_diagnostic_only and execution_modes != 1:
                blockers.append("adp009d_execution_modes_conflict")
            if articulated_native_requested and not args.adp009d_diagnostic_only:
                blockers.append("adp009d_articulated_native_requires_diagnostic_only")
            if articulated_native_requested and (selected_candidates or args.adp009d_controls):
                blockers.append("adp009d_articulated_native_policy_or_controls_forbidden")
            gated_backbone_selected = "groot_n17_droid" in selected_candidates
            gated_backbone_access: dict[str, Any] | None = None
            if gated_backbone_selected and not args.adp009d_authorize_gated_backbone:
                blockers.append("adp009d_gated_backbone_authority_missing")
            if args.adp009d_authorize_gated_backbone:
                if not gated_backbone_selected:
                    blockers.append("adp009d_gated_backbone_authority_without_candidate")
                else:
                    normalize_model_access_env()
                    secret_status = model_access_secret_status()
                    if secret_status["huggingface"]["auth_ready"] is not True:
                        blockers.append("adp009d_gated_backbone_token_missing")
                    else:
                        gated_backbone_access = probe_gated_backbone_access()
                        blockers.extend(gated_backbone_access.get("blockers") or [])
            if args.provider != "vast":
                blockers.append("adp009d_provider_must_be_vast")
            if not 0 < args.adp_max_hourly_rate_usd <= args.adp_max_spend_usd:
                blockers.append("adp009d_budget_invalid")
            if not 1800 <= args.adp_hard_ttl_seconds <= 14_400:
                blockers.append("adp009d_hard_ttl_invalid")
            if any(value <= 0 for value in args.adp_allowed_active_vast_instance_id):
                blockers.append("adp009d_allowed_active_vast_instance_id_invalid")
            avoidlist_digest = None
            if args.adp_machine_avoidlist:
                avoidlist_path = Path(args.adp_machine_avoidlist).expanduser().resolve()
                if not avoidlist_path.is_file():
                    blockers.append("adp009d_machine_avoidlist_missing")
                else:
                    avoidlist_digest = (
                        "sha256:" + hashlib.sha256(avoidlist_path.read_bytes()).hexdigest()
                    )
            prepared_bundle = None
            if not blockers:
                try:
                    if articulated_native_requested:
                        prepared_bundle = build_articulated_native_diagnostic_bundle(
                            job_dir=Path(args.adp_job_dir) / "bundle",
                            asset_path=args.adp009d_articulated_diagnostic_asset,
                            request_path=args.adp009d_articulated_diagnostic_request,
                            harness_manifest_path=args.adp009d_harness_manifest,
                            implementation_commit=control_identity["orchestrator_source_commit"],
                        )
                    else:
                        prepared_bundle = build_native_microcheck_bundle(
                            job_dir=Path(args.adp_job_dir) / "bundle",
                            approved_can_path=args.adp009d_approved_can,
                            sage_collision_path=args.adp009d_sage_collision,
                            harness_manifest_path=args.adp009d_harness_manifest,
                            implementation_commit=control_identity["orchestrator_source_commit"],
                            policy_candidate_id=args.adp009d_policy_candidate,
                            run_controls=args.adp009d_controls,
                            scenario_instance_path=args.adp009d_scenario_instance,
                            aura_particlefield_path=args.adp009d_aura_particlefield,
                        )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    blockers.append(f"adp009d_bundle_preparation_failed:{type(exc).__name__}")
            allocation_binding = {
                "program_id": "arm-decision-proof-v1",
                "probe_kind": ADP009D_NATIVE_MICROCHECK_PROBE_KIND,
                "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
                "bundle_sha256": (
                    prepared_bundle.get("bundle_sha256") if prepared_bundle else None
                ),
                "input_digest": (prepared_bundle.get("input_digest") if prepared_bundle else None),
                "candidate_policy_queried": False,
                "controls_requested": bool(args.adp009d_controls),
                "diagnostic_only_requested": bool(args.adp009d_diagnostic_only),
                "articulated_native_diagnostic_requested": articulated_native_requested,
                "diagnostic_kind": (
                    prepared_bundle.get("diagnostic_kind") if prepared_bundle else None
                ),
                "articulated_native_request_digest": (
                    prepared_bundle.get("request_digest")
                    if prepared_bundle and articulated_native_requested
                    else None
                ),
                "scenario_instance_digest": (
                    prepared_bundle.get("scenario_instance_digest") if prepared_bundle else None
                ),
                "control_plan_digest": (
                    prepared_bundle.get("control_plan_digest") if prepared_bundle else None
                ),
                "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                "hard_cap_usd": args.adp_max_spend_usd,
                "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                "retry_cap": 0,
                "machine_avoidlist_digest": avoidlist_digest,
                "gated_backbone_authorized": bool(args.adp009d_authorize_gated_backbone),
                "gated_backbone_access_receipt_digest": (
                    gated_backbone_access.get("receipt_digest") if gated_backbone_access else None
                ),
                "allowed_active_vast_instance_ids": sorted(
                    set(args.adp_allowed_active_vast_instance_id)
                ),
            }
            allocation_binding_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(allocation_binding, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()
            )
            paid_admission = build_paid_lane_admission(
                resource_class="vast_provider_adapter", blockers=blockers
            )
            paid_admission.update(
                {
                    "program_id": "arm-decision-proof-v1",
                    "probe_kind": ADP009D_NATIVE_MICROCHECK_PROBE_KIND,
                    "control_plane_identity": control_identity,
                    "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                    "hard_cap_usd": args.adp_max_spend_usd,
                    "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                    "retry_cap": 0,
                    "authority": "user_authorized_bounded_gpu_compute_in_goal_scope",
                    "private_data_uploaded": False,
                    "candidate_policy_queried": False,
                    "physical_outcome_values_uploaded": False,
                    "explicit_concurrent_gpu_authority_bound": bool(
                        args.adp_allowed_active_vast_instance_id
                    ),
                    "allocation_binding": allocation_binding,
                    "allocation_binding_digest": allocation_binding_digest,
                    "gated_backbone_access": gated_backbone_access,
                }
            )
            write_json(Path(args.admission_out), paid_admission)
            grant = None
            if args.execute:
                try:
                    grant = require_paid_resource_admission(
                        paid_admission,
                        resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                    )
                except PaidResourceAdmissionBlocked as exc:
                    result = {
                        "status": "blocked",
                        "blockers": exc.blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.adapter_output), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
            if prepared_bundle is None:
                result = {
                    "status": "blocked",
                    "blockers": sorted(set(blockers)),
                    "provider_mutations_performed": 0,
                }
            else:
                result = run_adp009d_native_microcheck_vast(
                    job_dir=args.adp_job_dir,
                    prepared_bundle=prepared_bundle,
                    paid_resource_admission_grant=grant,
                    execute=args.execute,
                    machine_avoidlist_path=args.adp_machine_avoidlist,
                    max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                    hard_cap_usd=args.adp_max_spend_usd,
                    hard_ttl_seconds=args.adp_hard_ttl_seconds,
                    authorize_gated_backbone=args.adp009d_authorize_gated_backbone,
                    allowed_active_instance_ids=(args.adp_allowed_active_vast_instance_id),
                )
            write_json(Path(args.adapter_output), result)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == ADP_ISAAC_LAB_ARENA_PROBE_KIND:
            missing = [
                name
                for name in ("adp_arena_approval", "adp_job_dir")
                if not getattr(args, name, None)
            ]
            control_blockers, control_identity = _control_plane_checkout_blockers()
            blockers = [*missing, *control_blockers]
            if args.provider != "vast":
                blockers.append("adp_arena_provider_must_be_vast")
            if not 0 < args.adp_max_hourly_rate_usd <= args.adp_max_spend_usd:
                blockers.append("adp_arena_budget_invalid")
            if not 1800 <= args.adp_hard_ttl_seconds <= 14_400:
                blockers.append("adp_arena_hard_ttl_invalid")
            avoidlist_digest = None
            if args.adp_machine_avoidlist:
                avoidlist_path = Path(args.adp_machine_avoidlist).expanduser().resolve()
                if not avoidlist_path.is_file():
                    blockers.append("adp_arena_machine_avoidlist_missing")
                else:
                    avoidlist_digest = (
                        "sha256:" + hashlib.sha256(avoidlist_path.read_bytes()).hexdigest()
                    )
            prepared_bundle = None
            if not blockers:
                try:
                    prepared_bundle = build_arena_native_control_bundle(
                        approval_path=args.adp_arena_approval,
                        job_dir=Path(args.adp_job_dir) / "bundle",
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    blockers.append(f"adp_arena_bundle_preparation_failed:{type(exc).__name__}")
            allocation_binding = {
                "program_id": "arm-decision-proof-v1",
                "probe_kind": ADP_ISAAC_LAB_ARENA_PROBE_KIND,
                "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
                "approval_path": args.adp_arena_approval,
                "protocol_digest": (
                    prepared_bundle.get("protocol_digest") if prepared_bundle else None
                ),
                "bundle_sha256": (
                    prepared_bundle.get("bundle_sha256") if prepared_bundle else None
                ),
                "control_id": "arena_zero_action_negative",
                "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                "hard_cap_usd": args.adp_max_spend_usd,
                "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                "retry_cap": 0,
                "machine_avoidlist_digest": avoidlist_digest,
            }
            allocation_binding_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(allocation_binding, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()
            )
            paid_admission = build_paid_lane_admission(
                resource_class="vast_provider_adapter", blockers=blockers
            )
            paid_admission.update(
                {
                    "program_id": "arm-decision-proof-v1",
                    "probe_kind": ADP_ISAAC_LAB_ARENA_PROBE_KIND,
                    "control_plane_identity": control_identity,
                    "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                    "hard_cap_usd": args.adp_max_spend_usd,
                    "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                    "retry_cap": 0,
                    "authority": (
                        "founder_exact_protocol_approval_plus_user_authorized_vast_spend"
                    ),
                    "private_data_uploaded": False,
                    "candidate_policy_queried": False,
                    "physical_outcome_values_uploaded": False,
                    "allocation_binding": allocation_binding,
                    "allocation_binding_digest": allocation_binding_digest,
                }
            )
            write_json(Path(args.admission_out), paid_admission)
            grant = None
            if args.execute:
                try:
                    grant = require_paid_resource_admission(
                        paid_admission,
                        resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                    )
                except PaidResourceAdmissionBlocked as exc:
                    result = {
                        "status": "blocked",
                        "blockers": exc.blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.adapter_output), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
            if prepared_bundle is None:
                result = {
                    "status": "blocked",
                    "blockers": blockers,
                    "provider_mutations_performed": 0,
                }
            else:
                result = run_arena_native_control_vast(
                    approval_path=args.adp_arena_approval,
                    job_dir=args.adp_job_dir,
                    paid_resource_admission_grant=grant,
                    execute=args.execute,
                    prepared_bundle=prepared_bundle,
                    machine_avoidlist_path=args.adp_machine_avoidlist,
                    max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                    hard_cap_usd=args.adp_max_spend_usd,
                    hard_ttl_seconds=args.adp_hard_ttl_seconds,
                )
            write_json(Path(args.adapter_output), result)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == ADP_SIMPLER_PUBLIC_REFERENCE_PROBE_KIND:
            missing = [
                name
                for name in ("adp_public_reference_manifest", "adp_job_dir")
                if not getattr(args, name, None)
            ]
            control_blockers, control_identity = _control_plane_checkout_blockers()
            blockers = [*missing, *control_blockers]
            if args.provider != "vast":
                blockers.append("adp_simpler_provider_must_be_vast")
            if not 0 < args.adp_max_hourly_rate_usd <= args.adp_max_spend_usd:
                blockers.append("adp_simpler_budget_invalid")
            if not 1800 <= args.adp_hard_ttl_seconds <= 14_400:
                blockers.append("adp_simpler_hard_ttl_invalid")
            avoidlist_digest = None
            if args.adp_machine_avoidlist:
                avoidlist_path = Path(args.adp_machine_avoidlist).expanduser().resolve()
                if not avoidlist_path.is_file():
                    blockers.append("adp_simpler_machine_avoidlist_missing")
                else:
                    avoidlist_digest = (
                        "sha256:" + hashlib.sha256(avoidlist_path.read_bytes()).hexdigest()
                    )
            prepared_bundle = None
            if not blockers:
                try:
                    prepared_bundle = build_simpler_public_vast_bundle(
                        manifest_path=args.adp_public_reference_manifest,
                        job_dir=Path(args.adp_job_dir) / "bundle",
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    blockers.append(f"adp_simpler_bundle_preparation_failed:{type(exc).__name__}")
            if prepared_bundle is not None and prepared_bundle.get("status") != "ready":
                blockers.extend(prepared_bundle.get("blockers") or ["adp_simpler_bundle_blocked"])
            allocation_binding = {
                "program_id": "arm-decision-proof-v1",
                "probe_kind": ADP_SIMPLER_PUBLIC_REFERENCE_PROBE_KIND,
                "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
                "public_reference_manifest": args.adp_public_reference_manifest,
                "source_identity_digest": (
                    prepared_bundle.get("source_identity_digest") if prepared_bundle else None
                ),
                "bundle_sha256": prepared_bundle.get("bundle_sha256") if prepared_bundle else None,
                "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                "hard_cap_usd": args.adp_max_spend_usd,
                "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                "retry_cap": 0,
                "machine_avoidlist_digest": avoidlist_digest,
            }
            allocation_binding_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(allocation_binding, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()
            )
            paid_admission = build_paid_lane_admission(
                resource_class="vast_provider_adapter", blockers=blockers
            )
            paid_admission.update(
                {
                    "program_id": "arm-decision-proof-v1",
                    "probe_kind": ADP_SIMPLER_PUBLIC_REFERENCE_PROBE_KIND,
                    "control_plane_identity": control_identity,
                    "public_reference_manifest": args.adp_public_reference_manifest,
                    "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
                    "hard_cap_usd": args.adp_max_spend_usd,
                    "hard_ttl_seconds": args.adp_hard_ttl_seconds,
                    "retry_cap": 0,
                    "authority": "user_authorized_vast_spend_for_arm_decision_proof_v1",
                    "private_data_uploaded": False,
                    "physical_outcome_values_uploaded": False,
                    "allocation_binding": allocation_binding,
                    "allocation_binding_digest": allocation_binding_digest,
                }
            )
            write_json(Path(args.admission_out), paid_admission)
            grant = None
            if args.execute:
                try:
                    grant = require_paid_resource_admission(
                        paid_admission,
                        resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                    )
                except PaidResourceAdmissionBlocked as exc:
                    result = {
                        "status": "blocked",
                        "blockers": exc.blockers,
                        "provider_mutations_performed": 0,
                    }
                    write_json(Path(args.adapter_output), result)
                    print(json.dumps({"success": False}, sort_keys=True))
                    return 2
            result = run_simpler_public_vast(
                manifest_path=args.adp_public_reference_manifest,
                job_dir=args.adp_job_dir,
                paid_resource_admission_grant=grant,
                execute=args.execute,
                prepared_bundle=prepared_bundle,
                machine_avoidlist_path=args.adp_machine_avoidlist,
                max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                hard_cap_usd=args.adp_max_spend_usd,
                hard_ttl_seconds=args.adp_hard_ttl_seconds,
            )
            write_json(Path(args.adapter_output), result)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == RECONSTRUCTION_WORKER_SMOKE_PROBE_KIND:
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
                result = _run_reconstruction_gpu_canary(args, checkout_commit=checkout_commit)
            success = result.get("status") in {"dry_run_ready", "completed"}
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
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
        if args.probe_kind == POLICY_RANKING_SUCCESSOR_COSMOS_PROBE_KIND:
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
        generic_missing = [
            name
            for name in (
                "provider_launch_request",
                "release_evidence",
                "model_cache_evidence",
                "preflight_bundle",
                "bound_request_out",
                "pod_name",
            )
            if not getattr(args, name, None)
        ]
        if generic_missing:
            result = {
                "status": "blocked",
                "blockers": [
                    "generic_gpu_canary_required_arguments_missing:"
                    + ",".join(sorted(generic_missing))
                ],
                "provider_mutations_performed": 0,
            }
            write_json(Path(args.admission_out), result)
            write_json(Path(args.adapter_output), result)
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
