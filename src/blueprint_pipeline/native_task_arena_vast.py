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
from .native_task_arena_destination_qualification_bundle import (
    RESULT_FILENAME as DESTINATION_QUALIFICATION_RESULT_FILENAME,
)
from .native_task_arena_paid_authority import (
    consume_native_task_arena_authority_once,
    validate_native_task_arena_paid_attempt_authority,
)
from .paid_resource_admission import PaidResourceAdmissionGrant
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_MINIMUM_DRIVER_VERSION as MINIMUM_DRIVER_VERSION


PROBE_KIND = "native-task-arena-construction"
RESULT_SCHEMA_VERSION = "native_task_arena_vast_run.v1"
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/native-task-arena"
NO_POLICY_MIN_GPU_RAM_MB = 24_000
NO_POLICY_PREFERRED_GPU_KEYWORDS = ("L40S", "RTX 6000 Ada", "RTX 4090")
# Runtime layers and evidence are served from Blueprint's US object store.
# A soft geolocation preference selected Chile and North Macedonia twice,
# consuming most of a 30-minute render probe before the 4.09 GB layer arrived.
# Keep every native-task allocation in the same country as those immutable
# inputs; the state-level regex remains a ranking preference inside this hard
# boundary.
NATIVE_TASK_ARENA_ALLOWED_GEOLOCATION_COUNTRY_CODES = ("US",)
def _run_native_task_arena_bounded_vast(
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
    allowed_machine_ids: Sequence[int] = (),
    paid_attempt_authority: Mapping[str, Any] | None = None,
    retain_warm_instance: bool = False,
) -> dict[str, Any]:
    """Run one zero-retry construction gate through the shared Vast transport."""

    expected_mode = "construction_canary"
    expected_output = "native_task_arena_construction_result.v1.json"
    if (
        prepared_bundle.get("schema_version")
        != "native_task_arena_provider_bundle.v1"
        or prepared_bundle.get("execution_mode") != expected_mode
        or prepared_bundle.get("policy_candidate_id") is not None
        or prepared_bundle.get("candidate_policy_queried") is not False
        or prepared_bundle.get("expected_output_filename")
        != expected_output
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
        expected_output_filename=expected_output,
        container_image=str(prepared_bundle["container_image"]),
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        result_schema_version=RESULT_SCHEMA_VERSION,
        object_store_key_prefix=DEFAULT_KEY_PREFIX,
        instance_label_prefix="blueprint-task-evaluation-native-arena-construction-",
        blocker_prefix="native_task_arena",
        # Construction loads Isaac, NuRec, and one robot/scene but no policy
        # checkpoint. NVIDIA lists 8 GB minimum and 16 GB "good" for Isaac;
        # exact Scene 840920 runs peaked near 5.2 GB on A6000/6000 Ada. Keep a
        # 24 GB floor, matching this module's already-qualified runtime
        # preflight, so live RTX 4090 capacity is not falsely rejected.
        min_gpu_ram_mb=NO_POLICY_MIN_GPU_RAM_MB,
        allowed_active_instance_ids=allowed_ids,
        allowed_machine_ids=allowed_machine_ids,
        vast_launch_lock_file=(
            job / "native_task_arena_paid_launch.lock" if allowed_ids else None
        ),
        candidate_policy_query_expected=False,
        preferred_gpu_keywords=NO_POLICY_PREFERRED_GPU_KEYWORDS,
        minimum_driver_version=MINIMUM_DRIVER_VERSION,
        require_independent_watchdog=True,
        allowed_geolocation_country_codes=(
            NATIVE_TASK_ARENA_ALLOWED_GEOLOCATION_COUNTRY_CODES
        ),
        authorization_consumption=consumption,
        retain_warm_instance=retain_warm_instance,
    )


def run_native_task_arena_vast(**kwargs: Any) -> dict[str, Any]:
    return _run_native_task_arena_bounded_vast(**kwargs)


def run_native_task_arena_destination_qualification_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    machine_avoidlist_path: str | Path | None = None,
    max_hourly_rate_usd: float = 0.80,
    hard_cap_usd: float = 1.00,
    hard_ttl_seconds: int = 1_800,
    allowed_active_instance_ids: Sequence[int] = (),
    paid_attempt_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one zero-retry, policy-free destination qualification probe."""

    if (
        prepared_bundle.get("schema_version")
        != "native_task_arena_provider_bundle.v1"
        or prepared_bundle.get("execution_mode") != "destination_qualification"
        or prepared_bundle.get("policy_candidate_id") is not None
        or prepared_bundle.get("candidate_policy_queried") is not False
        or prepared_bundle.get("expected_output_filename")
        != DESTINATION_QUALIFICATION_RESULT_FILENAME
    ):
        raise ValueError("native_task_arena_destination_qualification_bundle_invalid")
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
    job = Path(job_dir).expanduser().resolve()
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
        expected_output_filename=DESTINATION_QUALIFICATION_RESULT_FILENAME,
        container_image=str(prepared_bundle["container_image"]),
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        result_schema_version=RESULT_SCHEMA_VERSION,
        object_store_key_prefix=f"{DEFAULT_KEY_PREFIX}/destination-qualification",
        instance_label_prefix="blueprint-native-task-destination-qualification-",
        blocker_prefix="native_task_arena_destination_qualification",
        min_gpu_ram_mb=NO_POLICY_MIN_GPU_RAM_MB,
        allowed_active_instance_ids=allowed_ids,
        vast_launch_lock_file=(
            job / "native_task_arena_destination_qualification_paid_launch.lock"
            if allowed_ids
            else None
        ),
        candidate_policy_query_expected=False,
        preferred_gpu_keywords=NO_POLICY_PREFERRED_GPU_KEYWORDS,
        minimum_driver_version=MINIMUM_DRIVER_VERSION,
        require_independent_watchdog=True,
        allowed_geolocation_country_codes=(
            NATIVE_TASK_ARENA_ALLOWED_GEOLOCATION_COUNTRY_CODES
        ),
        authorization_consumption=consumption,
        stale_offer_create_retry_limit=0,
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
        allowed_geolocation_country_codes=(
            NATIVE_TASK_ARENA_ALLOWED_GEOLOCATION_COUNTRY_CODES
        ),
        authorization_consumption=consumption,
        retain_warm_instance=retain_warm_instance,
    )


__all__ = [
    "EXECUTION_RESULT_SCHEMA_VERSION",
    "MINIMUM_DRIVER_VERSION",
    "NO_POLICY_MIN_GPU_RAM_MB",
    "NO_POLICY_PREFERRED_GPU_KEYWORDS",
    "NATIVE_TASK_ARENA_ALLOWED_GEOLOCATION_COUNTRY_CODES",
    "PROBE_KIND",
    "RESULT_SCHEMA_VERSION",
    "run_native_task_arena_vast",
    "run_native_task_arena_destination_qualification_vast",
    "run_native_task_arena_controls_vast",
]
