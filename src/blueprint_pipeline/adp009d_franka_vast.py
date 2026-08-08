"""Canonical capped Vast transport for the ADP-009D native Franka micro-check."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .adp009d_native_microcheck_bundle import DEFAULT_IMAGE, PROBE_KIND
from .adp_isaac_lab_arena_vast import run_arena_native_control_vast
from .paid_resource_admission import PaidResourceAdmissionGrant


RESULT_SCHEMA_VERSION = "adp009d_franka_vast_run.v1"
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/adp009d-native-microcheck"


def run_adp009d_native_microcheck_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    machine_avoidlist_path: str | Path | None = None,
    max_hourly_rate_usd: float = 1.00,
    hard_cap_usd: float = 4.00,
    hard_ttl_seconds: int = 14_400,
    authorize_gated_backbone: bool = False,
    allowed_active_instance_ids: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Run one zero-retry native infrastructure check through the shared transport."""

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
        expected_output_filename="adp009d_native_microcheck.json",
        container_image=DEFAULT_IMAGE,
        provider_bundle_kind="adp009d_isaac",
        result_schema_version=RESULT_SCHEMA_VERSION,
        object_store_key_prefix=DEFAULT_KEY_PREFIX,
        instance_label_prefix="blueprint-adp009d-",
        blocker_prefix="adp009d",
        min_gpu_ram_mb=46_000,
        forward_hf_token=authorize_gated_backbone,
        allowed_active_instance_ids=tuple(allowed_active_instance_ids),
        preferred_gpu_keywords=("L40S", "RTX 6000 Ada", "RTX A6000"),
    )


__all__ = ["PROBE_KIND", "run_adp009d_native_microcheck_vast"]
