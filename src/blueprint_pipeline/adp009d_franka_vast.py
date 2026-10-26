"""Canonical capped Vast transport for the ADP-009D native Franka micro-check."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Mapping

from .adp009d_native_microcheck_bundle import DEFAULT_IMAGE, PROBE_KIND
from .adp_isaac_lab_arena_vast import run_arena_native_control_vast
from .paid_resource_admission import PaidResourceAdmissionGrant


RESULT_SCHEMA_VERSION = "adp009d_franka_vast_run.v1"
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/adp009d-native-microcheck"
MINIMUM_DRIVER_VERSION = "580.65.06"


@contextmanager
def _native_continuation_driver_scope(enabled):
    """The pinned 6.0.1 image uses NVIDIA's 595 branch; older images keep 580."""
    from .isaac_driver_support import MAX_ISAAC_DRIVER_ENV
    from .task_evaluation_scene_configuration_native_continuation import MAXIMUM_DRIVER_EXCLUSIVE
    old = os.environ.get(MAX_ISAAC_DRIVER_ENV)
    if enabled:
        os.environ[MAX_ISAAC_DRIVER_ENV] = MAXIMUM_DRIVER_EXCLUSIVE
    try:
        yield
    finally:
        if enabled:
            if old is None:
                os.environ.pop(MAX_ISAAC_DRIVER_ENV, None)
            else:
                os.environ[MAX_ISAAC_DRIVER_ENV] = old


def controls_only_max_compute_cap(prepared_bundle: Mapping[str, Any]) -> int | None:
    """Lift the TensorRT ceiling only for a sealed Newton controls-only bundle.

    Blackwell is incompatible with the pinned TensorRT policy engine, so the
    shared Vast adapter defaults to ``sm_90``.  Newton controls-only runs never
    build or query that engine.  Keep the opt-out fail-closed: any policy
    identity, missing controls binding, or ambiguous query status retains the
    shared ceiling.
    """

    if (
        prepared_bundle.get("physics_backend") == "newton"
        and prepared_bundle.get("controls_requested") is True
        and prepared_bundle.get("candidate_policy_queried") is False
        and not str(prepared_bundle.get("policy_candidate_id") or "").strip()
    ):
        return 0
    return None


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
    allowed_active_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Run one zero-retry native infrastructure check through the shared transport."""

    max_compute_cap = controls_only_max_compute_cap(prepared_bundle)
    policy_runtime_smoke = (
        prepared_bundle.get("execution_mode")
        == "outcome_blind_policy_runtime_smoke"
    )
    native_continuation = prepared_bundle.get("diagnostic_kind") == "scene_configuration_native_continuation"
    if native_continuation:
        from .public_scene_simready_isaac_bundle import DEFAULT_IMAGE as native_image
        from .task_evaluation_scene_configuration_native_continuation import MINIMUM_DRIVER_VERSION as native_driver
        from .task_evaluation_scene_configuration_native_continuation import MAXIMUM_DRIVER_EXCLUSIVE
        if (prepared_bundle.get("container_image") != native_image
                or prepared_bundle.get("minimum_driver_version") != native_driver
                or prepared_bundle.get("maximum_driver_exclusive") != MAXIMUM_DRIVER_EXCLUSIVE
                or prepared_bundle.get("candidate_policy_queried") is not False
                or prepared_bundle.get("policy_candidate_id") is not None
                or prepared_bundle.get("model_calls_performed") != 0
                or prepared_bundle.get("generated_tool_runtime_used") is not False
                or prepared_bundle.get("provider_disclosure_authorized") is not True
                or prepared_bundle.get("claim_ceiling") != "development_only"
                or authorize_gated_backbone):
            raise ValueError("native_continuation_transport_contract_invalid")

    with _native_continuation_driver_scope(native_continuation):
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
            container_image=native_image if native_continuation else DEFAULT_IMAGE,
            provider_bundle_kind=(
                "scene_configuration_native_continuation"
                if native_continuation
                else "adp009d_policy_runtime_smoke"
                if policy_runtime_smoke
                else "adp009d_articulated_native"
                if prepared_bundle.get("diagnostic_kind")
                == "blank_stage_articulated_asset"
                else "adp009d_isaac"
            ),
            result_schema_version=RESULT_SCHEMA_VERSION,
            object_store_key_prefix=DEFAULT_KEY_PREFIX,
            instance_label_prefix=(
                "blueprint-adp009d-policy-smoke-"
                if policy_runtime_smoke
                else "blueprint-adp009d-"
            ),
            blocker_prefix="adp009d",
            min_gpu_ram_mb=16_000 if native_continuation else 46_000,
            min_compute_cap=890 if native_continuation else 0,
            max_compute_cap=max_compute_cap,
            require_independent_watchdog=True,
            enable_isaac_smoke=True,
            forward_hf_token=authorize_gated_backbone,
            allowed_active_instance_ids=allowed_active_instance_ids,
            candidate_policy_query_expected=bool(
                str(prepared_bundle.get("policy_candidate_id") or "").strip()
            ),
            preferred_gpu_keywords=(("RTX 4000 Ada", "RTX4000Ada", "L4", "L40S")
                if native_continuation else ("L40S", "RTX 6000 Ada", "RTX A6000")),
            minimum_driver_version=native_driver if native_continuation else MINIMUM_DRIVER_VERSION,
        )


__all__ = [
    "MINIMUM_DRIVER_VERSION",
    "PROBE_KIND",
    "controls_only_max_compute_cap",
    "run_adp009d_native_microcheck_vast",
]
