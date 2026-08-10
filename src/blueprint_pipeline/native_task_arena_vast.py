"""Capped Vast transport for sealed, task-neutral native Arena packets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp_isaac_lab_arena_vast import DEFAULT_IMAGE, run_arena_native_control_vast
from .native_task_arena_construction_bundle import (
    PROVIDER_BUNDLE_KIND,
    RESULT_SCHEMA_VERSION as EXECUTION_RESULT_SCHEMA_VERSION,
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
        container_image=DEFAULT_IMAGE,
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        result_schema_version=RESULT_SCHEMA_VERSION,
        object_store_key_prefix=DEFAULT_KEY_PREFIX,
        instance_label_prefix="blueprint-native-task-arena-",
        blocker_prefix="native_task_arena",
        min_gpu_ram_mb=46_000,
        allowed_active_instance_ids=allowed_active_instance_ids,
        candidate_policy_query_expected=False,
        preferred_gpu_keywords=("L40S", "RTX 6000 Ada", "RTX A6000"),
        minimum_driver_version=MINIMUM_DRIVER_VERSION,
    )


__all__ = [
    "EXECUTION_RESULT_SCHEMA_VERSION",
    "MINIMUM_DRIVER_VERSION",
    "PROBE_KIND",
    "RESULT_SCHEMA_VERSION",
    "run_native_task_arena_vast",
]
