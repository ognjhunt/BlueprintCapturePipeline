"""Capped Vast transport for sealed, task-neutral native Arena packets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp_isaac_lab_arena_vast import run_arena_native_control_vast
from .native_task_arena_construction_bundle import (
    PROVIDER_BUNDLE_KIND,
    RESULT_SCHEMA_VERSION as EXECUTION_RESULT_SCHEMA_VERSION,
)
from .native_task_arena_controls_bundle import (
    RESULT_FILENAME as CONTROLS_RESULT_FILENAME,
)
from .native_task_arena_policy_bundle import RESULT_FILENAME as POLICY_RESULT_FILENAME
from .native_task_arena_policy_diagnostic_bundle import (
    RESULT_FILENAME as POLICY_DIAGNOSTIC_RESULT_FILENAME,
)
from .native_task_arena_runtime_preflight_bundle import (
    RESULT_FILENAME as RUNTIME_PREFLIGHT_RESULT_FILENAME,
    RESULT_SCHEMA_VERSION as RUNTIME_PREFLIGHT_RESULT_SCHEMA_VERSION,
)
from .native_task_arena_paid_authority import (
    consume_native_task_arena_authority_once,
    validate_native_task_arena_paid_attempt_authority,
)
from .paid_resource_admission import PaidResourceAdmissionGrant


PROBE_KIND = "native-task-arena-construction"
RESULT_SCHEMA_VERSION = "native_task_arena_vast_run.v1"
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/native-task-arena"
MINIMUM_DRIVER_VERSION = "580.65.06"
NO_POLICY_MIN_GPU_RAM_MB = 24_000
NO_POLICY_PREFERRED_GPU_KEYWORDS = ("L40S", "RTX 6000 Ada", "RTX 4090")


def run_native_task_arena_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    machine_avoidlist_path: str | Path | None = None,
    max_hourly_rate_usd: float = 0.80,
    hard_cap_usd: float = 1.00,
    hard_ttl_seconds: int = 5_400,
    allowed_active_instance_ids: Sequence[int] = (),
    paid_attempt_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one zero-retry construction gate through the shared Vast transport."""

    if (
        prepared_bundle.get("schema_version")
        != "native_task_arena_provider_bundle.v1"
        or prepared_bundle.get("execution_mode") != "construction_canary"
        or prepared_bundle.get("policy_candidate_id") is not None
        or prepared_bundle.get("candidate_policy_queried") is not False
        or prepared_bundle.get("expected_output_filename")
        != "native_task_arena_construction_result.v1.json"
    ):
        raise ValueError("native_task_arena_prepared_bundle_contract_invalid")
    job = Path(job_dir).expanduser().resolve()
    allowed_ids = tuple(sorted({int(value) for value in allowed_active_instance_ids}))
    authority = (
        validate_native_task_arena_paid_attempt_authority(
            paid_attempt_authority,
            prepared_bundle=prepared_bundle,
            max_hourly_rate_usd=max_hourly_rate_usd,
            hard_cap_usd=hard_cap_usd,
            hard_ttl_seconds=hard_ttl_seconds,
            allowed_active_instance_ids=allowed_ids,
        )
        if paid_attempt_authority is not None
        else None
    )
    if execute and authority is None:
        raise ValueError("native_task_arena_paid_execution_authority_missing")
    consumption = consume_native_task_arena_authority_once(authority) if execute else None
    if consumption is not None and consumption.get("status") != "consumed":
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "authorization_consumption": consumption,
            "blockers": list(consumption.get("blockers") or []),
        }
    return run_arena_native_control_vast(
        approval_path=".",
        job_dir=job_dir,
        paid_resource_admission_grant=paid_resource_admission_grant,
        execute=execute,
        prepared_bundle=prepared_bundle,
        machine_avoidlist_path=machine_avoidlist_path,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        expected_output_filename="native_task_arena_construction_result.v1.json",
        container_image=str(prepared_bundle["container_image"]),
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        result_schema_version=RESULT_SCHEMA_VERSION,
        object_store_key_prefix=DEFAULT_KEY_PREFIX,
        instance_label_prefix="blueprint-native-task-arena-",
        blocker_prefix="native_task_arena",
        # Construction loads Isaac, NuRec, and one robot/scene but no policy
        # checkpoint. NVIDIA lists 8 GB minimum and 16 GB "good" for Isaac;
        # exact Scene 840920 runs peaked near 5.2 GB on A6000/6000 Ada. Keep a
        # 24 GB floor, matching this module's already-qualified runtime
        # preflight, so live RTX 4090 capacity is not falsely rejected.
        min_gpu_ram_mb=NO_POLICY_MIN_GPU_RAM_MB,
        allowed_active_instance_ids=allowed_ids,
        vast_launch_lock_file=(
            job / "native_task_arena_paid_launch.lock" if allowed_ids else None
        ),
        candidate_policy_query_expected=False,
        preferred_gpu_keywords=NO_POLICY_PREFERRED_GPU_KEYWORDS,
        minimum_driver_version=MINIMUM_DRIVER_VERSION,
        require_independent_watchdog=True,
        authorization_consumption=consumption,
    )


def run_native_task_arena_runtime_preflight_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    machine_avoidlist_path: str | Path | None = None,
    max_hourly_rate_usd: float = 0.80,
    hard_cap_usd: float = 1.00,
    hard_ttl_seconds: int = 5_400,
    allowed_active_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Run one no-motion, zero-retry Arena compatibility preflight."""

    if (
        prepared_bundle.get("schema_version")
        != "native_task_arena_provider_bundle.v1"
        or prepared_bundle.get("execution_mode") != "runtime_preflight"
        or prepared_bundle.get("policy_candidate_id") is not None
        or prepared_bundle.get("candidate_policy_queried") is not False
        or prepared_bundle.get("expected_output_filename")
        != RUNTIME_PREFLIGHT_RESULT_FILENAME
    ):
        raise ValueError("native_task_arena_runtime_preflight_bundle_invalid")
    job = Path(job_dir).expanduser().resolve()
    allowed_ids = tuple(sorted({int(value) for value in allowed_active_instance_ids}))
    return run_arena_native_control_vast(
        approval_path=".",
        job_dir=job,
        paid_resource_admission_grant=paid_resource_admission_grant,
        execute=execute,
        prepared_bundle=prepared_bundle,
        machine_avoidlist_path=machine_avoidlist_path,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        expected_output_filename=RUNTIME_PREFLIGHT_RESULT_FILENAME,
        container_image=str(prepared_bundle["container_image"]),
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        result_schema_version=RUNTIME_PREFLIGHT_RESULT_SCHEMA_VERSION,
        object_store_key_prefix=f"{DEFAULT_KEY_PREFIX}/runtime-preflight",
        instance_label_prefix="blueprint-native-task-arena-preflight-",
        blocker_prefix="native_task_arena_runtime_preflight",
        # This no-motion preflight captures three 320x180 cameras and runs no
        # policy or training workload.  NVIDIA lists 16 GB as the Isaac Sim
        # minimum; 24 GB keeps RTX 4090 offers usable, matching this lane's
        # explicit Ada preference.  Construction/controls/policy retain their
        # 46 GB floor.
        min_gpu_ram_mb=24_000,
        allowed_active_instance_ids=allowed_ids,
        vast_launch_lock_file=(
            job / "native_task_arena_runtime_preflight_paid_launch.lock"
            if allowed_ids
            else None
        ),
        candidate_policy_query_expected=False,
        preferred_gpu_keywords=("L40S", "RTX 6000 Ada", "RTX 4090"),
        gpu_selection_policy={
            "policy_id": "native_task_arena_runtime_preflight_ada_only",
            "allowed_gpu_keywords": ("L40S", "6000ADA", "RTX 4090"),
            "reason": "NuRec runtime preflight is qualified only on Ada GPUs",
            "minimum_cuda_max_good": 12.8,
        },
        minimum_driver_version=MINIMUM_DRIVER_VERSION,
        require_independent_watchdog=True,
    )


def run_native_task_arena_controls_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    machine_avoidlist_path: str | Path | None = None,
    max_hourly_rate_usd: float = 0.80,
    hard_cap_usd: float = 1.00,
    hard_ttl_seconds: int = 5_400,
    allowed_active_instance_ids: Sequence[int] = (),
    paid_attempt_authority: Mapping[str, Any] | None = None,
    retain_warm_instance: bool = False,
) -> dict[str, Any]:
    """Run one zero-retry control pair through the same capped transport."""

    if (
        prepared_bundle.get("schema_version")
        != "native_task_arena_provider_bundle.v1"
        or prepared_bundle.get("execution_mode") != "controls"
        or prepared_bundle.get("policy_candidate_id") is not None
        or prepared_bundle.get("candidate_policy_queried") is not False
        or prepared_bundle.get("expected_output_filename")
        != CONTROLS_RESULT_FILENAME
    ):
        raise ValueError("native_task_arena_controls_prepared_bundle_contract_invalid")
    allowed_ids = tuple(sorted({int(value) for value in allowed_active_instance_ids}))
    authority = (
        validate_native_task_arena_paid_attempt_authority(
            paid_attempt_authority,
            prepared_bundle=prepared_bundle,
            max_hourly_rate_usd=max_hourly_rate_usd,
            hard_cap_usd=hard_cap_usd,
            hard_ttl_seconds=hard_ttl_seconds,
            allowed_active_instance_ids=allowed_ids,
            retain_warm_session=retain_warm_instance,
        )
        if paid_attempt_authority is not None
        else None
    )
    if execute and authority is None:
        raise ValueError("native_task_arena_paid_execution_authority_missing")
    consumption = consume_native_task_arena_authority_once(authority) if execute else None
    if consumption is not None and consumption.get("status") != "consumed":
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "authorization_consumption": consumption,
            "blockers": list(consumption.get("blockers") or []),
        }
    return run_arena_native_control_vast(
        approval_path=".",
        job_dir=job_dir,
        paid_resource_admission_grant=paid_resource_admission_grant,
        execute=execute,
        prepared_bundle=prepared_bundle,
        machine_avoidlist_path=machine_avoidlist_path,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        expected_output_filename=CONTROLS_RESULT_FILENAME,
        container_image=str(prepared_bundle["container_image"]),
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        result_schema_version=RESULT_SCHEMA_VERSION,
        object_store_key_prefix=DEFAULT_KEY_PREFIX,
        instance_label_prefix="blueprint-native-task-controls-",
        blocker_prefix="native_task_arena_controls",
        # Deterministic controls load the same Isaac scene and no model.
        # Policies retain their independent 46 GB floor below.
        min_gpu_ram_mb=NO_POLICY_MIN_GPU_RAM_MB,
        allowed_active_instance_ids=allowed_ids,
        # Active-instance allowlisting permits an explicitly coordinated
        # concurrent run; it must not also replace the provider-wide semaphore
        # with a lane-local lock. ``None`` makes the transport resolve the
        # canonical VAST_LAUNCH_LOCK_FILE and its bounded concurrency slots.
        vast_launch_lock_file=None,
        candidate_policy_query_expected=False,
        preferred_gpu_keywords=NO_POLICY_PREFERRED_GPU_KEYWORDS,
        minimum_driver_version=MINIMUM_DRIVER_VERSION,
        require_independent_watchdog=True,
        authorization_consumption=consumption,
        retain_warm_instance=retain_warm_instance,
    )


def run_native_task_arena_policy_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    machine_avoidlist_path: str | Path | None = None,
    max_hourly_rate_usd: float = 0.80,
    hard_cap_usd: float = 1.00,
    hard_ttl_seconds: int = 5_400,
    allowed_active_instance_ids: Sequence[int] = (),
    paid_attempt_authority: Mapping[str, Any] | None = None,
    authorize_gated_backbone: bool = False,
) -> dict[str, Any]:
    """Run one admitted candidate through the same zero-retry Vast transport."""

    return _run_native_task_arena_policy_vast(
        job_dir=job_dir,
        prepared_bundle=prepared_bundle,
        paid_resource_admission_grant=paid_resource_admission_grant,
        execute=execute,
        machine_avoidlist_path=machine_avoidlist_path,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=allowed_active_instance_ids,
        paid_attempt_authority=paid_attempt_authority,
        authorize_gated_backbone=authorize_gated_backbone,
        expected_execution_mode="policy",
        expected_output_filename=POLICY_RESULT_FILENAME,
        label_prefix="blueprint-native-task-policy-",
        blocker_prefix="native_task_arena_policy",
    )


def run_native_task_arena_policy_diagnostic_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    machine_avoidlist_path: str | Path | None = None,
    max_hourly_rate_usd: float = 0.80,
    hard_cap_usd: float = 1.00,
    hard_ttl_seconds: int = 5_400,
    allowed_active_instance_ids: Sequence[int] = (),
    paid_attempt_authority: Mapping[str, Any] | None = None,
    authorize_gated_backbone: bool = False,
) -> dict[str, Any]:
    """Run a canonical policy diagnostic that is ineligible for scoring."""

    return _run_native_task_arena_policy_vast(
        job_dir=job_dir,
        prepared_bundle=prepared_bundle,
        paid_resource_admission_grant=paid_resource_admission_grant,
        execute=execute,
        machine_avoidlist_path=machine_avoidlist_path,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=allowed_active_instance_ids,
        paid_attempt_authority=paid_attempt_authority,
        authorize_gated_backbone=authorize_gated_backbone,
        expected_execution_mode="policy_diagnostic",
        expected_output_filename=POLICY_DIAGNOSTIC_RESULT_FILENAME,
        label_prefix="blueprint-native-task-policy-diagnostic-",
        blocker_prefix="native_task_arena_policy_diagnostic",
    )


def _run_native_task_arena_policy_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    machine_avoidlist_path: str | Path | None,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    allowed_active_instance_ids: Sequence[int],
    paid_attempt_authority: Mapping[str, Any] | None,
    authorize_gated_backbone: bool,
    expected_execution_mode: str,
    expected_output_filename: str,
    label_prefix: str,
    blocker_prefix: str,
) -> dict[str, Any]:

    candidate = str(prepared_bundle.get("policy_candidate_id") or "")
    if (
        prepared_bundle.get("schema_version")
        != "native_task_arena_provider_bundle.v1"
        or prepared_bundle.get("execution_mode") != expected_execution_mode
        or candidate not in {"pi05_droid", "groot_n17_droid"}
        or prepared_bundle.get("candidate_policy_queried") is not False
        or prepared_bundle.get("expected_output_filename")
        != expected_output_filename
    ):
        raise ValueError("native_task_arena_policy_prepared_bundle_contract_invalid")
    allowed_ids = tuple(sorted({int(value) for value in allowed_active_instance_ids}))
    authority = (
        validate_native_task_arena_paid_attempt_authority(
            paid_attempt_authority,
            prepared_bundle=prepared_bundle,
            max_hourly_rate_usd=max_hourly_rate_usd,
            hard_cap_usd=hard_cap_usd,
            hard_ttl_seconds=hard_ttl_seconds,
            allowed_active_instance_ids=allowed_ids,
        )
        if paid_attempt_authority is not None
        else None
    )
    if execute and authority is None:
        raise ValueError("native_task_arena_paid_execution_authority_missing")
    if candidate == "groot_n17_droid" and not authorize_gated_backbone:
        raise ValueError("native_task_arena_groot_gated_backbone_authority_missing")
    if candidate != "groot_n17_droid" and authorize_gated_backbone:
        raise ValueError("native_task_arena_gated_backbone_authority_without_groot")
    consumption = consume_native_task_arena_authority_once(authority) if execute else None
    if consumption is not None and consumption.get("status") != "consumed":
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "authorization_consumption": consumption,
            "blockers": list(consumption.get("blockers") or []),
        }
    return run_arena_native_control_vast(
        approval_path=".",
        job_dir=job_dir,
        paid_resource_admission_grant=paid_resource_admission_grant,
        execute=execute,
        prepared_bundle=prepared_bundle,
        machine_avoidlist_path=machine_avoidlist_path,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        expected_output_filename=expected_output_filename,
        container_image=str(prepared_bundle["container_image"]),
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        result_schema_version=RESULT_SCHEMA_VERSION,
        object_store_key_prefix=f"{DEFAULT_KEY_PREFIX}/policy/{candidate}",
        instance_label_prefix=label_prefix,
        blocker_prefix=blocker_prefix,
        min_gpu_ram_mb=46_000,
        allowed_active_instance_ids=allowed_ids,
        # Policy and controls share the provider-wide semaphore even when a
        # predecessor GPU is explicitly allowlisted. The allowlist governs
        # inventory admission; it is not permission to bypass launch slots.
        vast_launch_lock_file=None,
        candidate_policy_query_expected=True,
        forward_hf_token=(candidate == "groot_n17_droid"),
        preferred_gpu_keywords=("L40S", "RTX 6000 Ada", "RTX A6000"),
        minimum_driver_version=MINIMUM_DRIVER_VERSION,
        require_independent_watchdog=True,
        authorization_consumption=consumption,
    )


__all__ = [
    "EXECUTION_RESULT_SCHEMA_VERSION",
    "MINIMUM_DRIVER_VERSION",
    "NO_POLICY_MIN_GPU_RAM_MB",
    "NO_POLICY_PREFERRED_GPU_KEYWORDS",
    "PROBE_KIND",
    "RESULT_SCHEMA_VERSION",
    "run_native_task_arena_vast",
    "run_native_task_arena_runtime_preflight_vast",
    "run_native_task_arena_controls_vast",
    "run_native_task_arena_policy_vast",
    "run_native_task_arena_policy_diagnostic_vast",
]
