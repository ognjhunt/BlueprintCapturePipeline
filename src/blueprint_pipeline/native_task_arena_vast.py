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
from .native_task_arena_paid_authority import (
    consume_native_task_arena_authority_once,
    validate_native_task_arena_paid_attempt_authority,
)
from .paid_resource_admission import PaidResourceAdmissionGrant


PROBE_KIND = "native-task-arena-construction"
RESULT_SCHEMA_VERSION = "native_task_arena_vast_run.v1"
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/native-task-arena"
MINIMUM_DRIVER_VERSION = "580.65.06"


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
        min_gpu_ram_mb=46_000,
        allowed_active_instance_ids=allowed_ids,
        vast_launch_lock_file=(
            job / "native_task_arena_paid_launch.lock" if allowed_ids else None
        ),
        candidate_policy_query_expected=False,
        preferred_gpu_keywords=("L40S", "RTX 6000 Ada", "RTX A6000"),
        minimum_driver_version=MINIMUM_DRIVER_VERSION,
        require_independent_watchdog=True,
        authorization_consumption=consumption,
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
        expected_output_filename=CONTROLS_RESULT_FILENAME,
        container_image=str(prepared_bundle["container_image"]),
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        result_schema_version=RESULT_SCHEMA_VERSION,
        object_store_key_prefix=DEFAULT_KEY_PREFIX,
        instance_label_prefix="blueprint-native-task-controls-",
        blocker_prefix="native_task_arena_controls",
        min_gpu_ram_mb=46_000,
        allowed_active_instance_ids=allowed_ids,
        vast_launch_lock_file=(
            job / "native_task_arena_controls_paid_launch.lock"
            if allowed_ids
            else None
        ),
        candidate_policy_query_expected=False,
        preferred_gpu_keywords=("L40S", "RTX 6000 Ada", "RTX A6000"),
        minimum_driver_version=MINIMUM_DRIVER_VERSION,
        require_independent_watchdog=True,
        authorization_consumption=consumption,
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
) -> dict[str, Any]:
    """Run one admitted candidate through the same zero-retry Vast transport."""

    candidate = str(prepared_bundle.get("policy_candidate_id") or "")
    if (
        prepared_bundle.get("schema_version")
        != "native_task_arena_provider_bundle.v1"
        or prepared_bundle.get("execution_mode") != "policy"
        or candidate not in {"pi05_droid", "groot_n17_droid"}
        or prepared_bundle.get("candidate_policy_queried") is not False
        or prepared_bundle.get("expected_output_filename") != POLICY_RESULT_FILENAME
    ):
        raise ValueError("native_task_arena_policy_prepared_bundle_contract_invalid")
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
        expected_output_filename=POLICY_RESULT_FILENAME,
        container_image=str(prepared_bundle["container_image"]),
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        result_schema_version=RESULT_SCHEMA_VERSION,
        object_store_key_prefix=f"{DEFAULT_KEY_PREFIX}/policy/{candidate}",
        instance_label_prefix="blueprint-native-task-policy-",
        blocker_prefix="native_task_arena_policy",
        min_gpu_ram_mb=46_000,
        allowed_active_instance_ids=allowed_ids,
        vast_launch_lock_file=(
            job / "native_task_arena_policy_paid_launch.lock" if allowed_ids else None
        ),
        candidate_policy_query_expected=True,
        preferred_gpu_keywords=("L40S", "RTX 6000 Ada", "RTX A6000"),
        minimum_driver_version=MINIMUM_DRIVER_VERSION,
        require_independent_watchdog=True,
        authorization_consumption=consumption,
    )


__all__ = [
    "EXECUTION_RESULT_SCHEMA_VERSION",
    "MINIMUM_DRIVER_VERSION",
    "PROBE_KIND",
    "RESULT_SCHEMA_VERSION",
    "run_native_task_arena_vast",
    "run_native_task_arena_controls_vast",
    "run_native_task_arena_policy_vast",
]
