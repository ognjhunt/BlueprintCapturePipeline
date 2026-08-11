"""Capped Vast transport for one immutable deformable-asset native canary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp_isaac_lab_arena_vast import run_arena_native_control_vast
from .native_deformable_asset_provider_bundle import (
    EXPECTED_OUTPUT_FILENAME,
    PROVIDER_BUNDLE_KIND,
    RESULT_SCHEMA_VERSION,
    SCHEMA_VERSION,
)
from .paid_resource_admission import PaidResourceAdmissionGrant


PROBE_KIND = "native-deformable-asset-preparation"
VAST_RESULT_SCHEMA_VERSION = "native_deformable_asset_vast_run.v1"
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/native-deformable-asset"
MINIMUM_DRIVER_VERSION = "580.65.06"


def run_native_deformable_asset_vast(
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
    """Run one zero-retry cook/readback canary through the shared Vast transport."""

    if (
        prepared_bundle.get("schema_version") != SCHEMA_VERSION
        or prepared_bundle.get("execution_mode") != "asset_preparation_canary"
        or prepared_bundle.get("provider_bundle_kind") != PROVIDER_BUNDLE_KIND
        or prepared_bundle.get("expected_output_filename") != EXPECTED_OUTPUT_FILENAME
        or prepared_bundle.get("candidate_policy_queried") is not False
        or prepared_bundle.get("native_cook_qualified") is not False
    ):
        raise ValueError("native_deformable_asset_prepared_bundle_contract_invalid")
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
        expected_output_filename=EXPECTED_OUTPUT_FILENAME,
        container_image=str(prepared_bundle["container_image"]),
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        result_schema_version=VAST_RESULT_SCHEMA_VERSION,
        object_store_key_prefix=DEFAULT_KEY_PREFIX,
        instance_label_prefix="blueprint-native-deformable-asset-",
        blocker_prefix="native_deformable_asset",
        min_gpu_ram_mb=24_000,
        allowed_active_instance_ids=allowed_ids,
        vast_launch_lock_file=(
            job / "native_deformable_asset_paid_launch.lock" if allowed_ids else None
        ),
        candidate_policy_query_expected=False,
        preferred_gpu_keywords=("L40S", "RTX 6000 Ada", "RTX A6000", "L40"),
        minimum_driver_version=MINIMUM_DRIVER_VERSION,
    )


__all__ = [
    "MINIMUM_DRIVER_VERSION",
    "PROBE_KIND",
    "RESULT_SCHEMA_VERSION",
    "VAST_RESULT_SCHEMA_VERSION",
    "run_native_deformable_asset_vast",
]
