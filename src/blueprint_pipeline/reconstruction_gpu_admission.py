"""Vast-first, provider-neutral admission for reconstruction GPU canaries."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .reconstruction_isaac_image_release import (
    ReconstructionIsaacImageReleaseError,
    validate_reconstruction_isaac_image_release,
)
from .measurement_isaac_runtime_release import (
    MeasurementIsaacRuntimeReleaseError,
    validate_measurement_isaac_runtime_release,
)
from .measurement_dlo_lab_runtime_release import (
    MeasurementDloLabRuntimeReleaseError,
    validate_measurement_dlo_lab_runtime_release,
)


REQUEST_SCHEMA_VERSION = "reconstruction_gpu_canary_request.v1"
PREFLIGHT_SCHEMA_VERSION = "reconstruction_gpu_provider_preflight.v1"
ADMISSION_SCHEMA_VERSION = "reconstruction_gpu_canary_admission.v1"
PROBE_KIND = "reconstruction-worker-smoke"
MIN_GPU_MEMORY_BYTES = 24 * 1024**3
MIN_CONTAINER_DISK_BYTES = 100 * 1024**3
MAX_PREFLIGHT_AGE_SECONDS = 300
MAX_TTL_SECONDS = 14_400
MAX_RETRY_CAP = 2
CAPTURE_PROFILES = {
    "iphone_arkit_lidar",
    "camera_360_native",
    "camera_360_equirectangular",
    "trainer_smoke_fixture",
    "public_provider_sample",
    "user_managed_provider_export",
    "synthetic_measurement",
}
OPERATIONS = {
    "worker_smoke",
    "pose_canary",
    "trainer_canary",
    "isaac_canary",
    "provider_nurec_isaac_canary",
    "external_scene_isaac_canary",
    "measurement_isaac_canary",
    "measurement_dlo_lab_canary",
}
EXECUTABLE_OPERATIONS = {
    "worker_smoke",
    "pose_canary",
    "trainer_canary",
    "isaac_canary",
    "provider_nurec_isaac_canary",
    "external_scene_isaac_canary",
    "measurement_isaac_canary",
    "measurement_dlo_lab_canary",
}
EXPECTED_RUNTIME_RESULT_SCHEMAS = {
    "worker_smoke": "reconstruction_vast_worker_smoke_result.v1",
    "pose_canary": "pose_estimation_result.v1",
    "trainer_canary": "reconstruction_training_result.v1",
    "isaac_canary": "isaac_splat_nurec_render_result.v3",
    "provider_nurec_isaac_canary": "provider_nurec_isaac_runtime_result.v1",
    "external_scene_isaac_canary": "isaac_splat_nurec_render_result.v3",
    "measurement_isaac_canary": "measurement_isaac_physx_vast_runtime_result.v1",
    "measurement_dlo_lab_canary": "measurement_dlo_lab_cuda_vast_runtime_result.v1",
}
PROVIDER_NUREC_ISAAC_OPERATION = "provider_nurec_isaac_canary"
EXTERNAL_SCENE_ISAAC_OPERATION = "external_scene_isaac_canary"
EXTERNAL_DERIVED_ISAAC_OPERATIONS = {
    PROVIDER_NUREC_ISAAC_OPERATION,
    EXTERNAL_SCENE_ISAAC_OPERATION,
}
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


def build_reconstruction_gpu_canary_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an immutable provider-neutral request for one registered operation."""

    source = json.loads(json.dumps(dict(value)))
    source.pop("request_digest", None)
    operation = str(source.get("operation") or "")
    errors: list[str] = []
    if source.get("schema_version") != REQUEST_SCHEMA_VERSION:
        errors.append("reconstruction_gpu_request_schema_invalid")
    if operation not in OPERATIONS:
        errors.append("reconstruction_gpu_operation_unsupported")
    if source.get("capture_profile") not in CAPTURE_PROFILES:
        errors.append("reconstruction_gpu_capture_profile_unsupported")
    if _COMMIT.fullmatch(str(source.get("source_commit_sha") or "")) is None:
        errors.append("reconstruction_gpu_source_commit_invalid")
    if _IMAGE.fullmatch(str(source.get("worker_image_digest") or "")) is None:
        errors.append("reconstruction_gpu_worker_image_digest_invalid")
    for key in (
        "worker_stack_manifest_digest",
        "deterministic_configuration_digest",
        "operation_request_digest",
        "operation_input_bundle_digest",
    ):
        if _DIGEST.fullmatch(str(source.get(key) or "")) is None:
            errors.append(f"reconstruction_gpu_{key}_invalid")
    if operation in EXTERNAL_DERIVED_ISAAC_OPERATIONS:
        required_external_digests = (
            ("external_import_receipt_digest", "provider_qualification_report_digest")
            if operation == PROVIDER_NUREC_ISAAC_OPERATION
            else (
                "remote_processing_authorization_digest",
                "package_result_digest",
                "collision_candidate_digest",
                "scene_frame_binding_digest",
                "target_analysis_digest",
                "target_binding_digest",
                "placement_proposal_digest",
            )
        )
        for key in required_external_digests:
            if _DIGEST.fullmatch(str(source.get(key) or "")) is None:
                errors.append(f"reconstruction_gpu_{key}_invalid")
        for key in (
            "reconstruction_dataset_digest",
            "frozen_split_digest",
            "calibration_digest",
        ):
            if key in source:
                boundary = "provider" if operation == PROVIDER_NUREC_ISAAC_OPERATION else "external"
                errors.append(f"reconstruction_gpu_{boundary}_capture_binding_forbidden:{key}")
        expected_boundaries = {
            "source_relationship_to_blueprint_raw_capture": "none",
            "external_derived_support_asset": True,
            "blueprint_raw_capture_truth": False,
        }
        for key, expected in expected_boundaries.items():
            if source.get(key) != expected:
                boundary = "provider" if operation == PROVIDER_NUREC_ISAAC_OPERATION else "external"
                errors.append(f"reconstruction_gpu_{boundary}_source_boundary_invalid:{key}")
        if operation == EXTERNAL_SCENE_ISAAC_OPERATION:
            for key, expected in {
                "source_video_available": False,
                "source_video_required_for_candidate_execution": False,
                "independent_metric_scale_proven": False,
                "remote_upload_authorized": True,
                "paid_compute_authorized": True,
            }.items():
                if source.get(key) != expected:
                    errors.append(f"reconstruction_gpu_external_boundary_invalid:{key}")
    else:
        for key in (
            "reconstruction_dataset_digest",
            "frozen_split_digest",
            "calibration_digest",
        ):
            if _DIGEST.fullmatch(str(source.get(key) or "")) is None:
                errors.append(f"reconstruction_gpu_{key}_invalid")
    if source.get("expected_runtime_result_schema") != (
        EXPECTED_RUNTIME_RESULT_SCHEMAS.get(operation)
    ):
        errors.append("reconstruction_gpu_expected_runtime_result_schema_invalid")
    if source.get("candidate_may_read_hidden_heldout") is not False:
        errors.append("reconstruction_gpu_hidden_heldout_access_forbidden")
    if source.get("trainer_may_grade_heldout") is not False:
        errors.append("reconstruction_gpu_trainer_self_grading_forbidden")
    if not _finite(source.get("max_spend_usd"), minimum=0.000001):
        errors.append("reconstruction_gpu_explicit_budget_missing")
    ttl = source.get("hard_ttl_seconds")
    if not isinstance(ttl, int) or isinstance(ttl, bool) or not 1 <= ttl <= MAX_TTL_SECONDS:
        errors.append("reconstruction_gpu_explicit_ttl_invalid")
    retries = source.get("retry_cap")
    if (
        not isinstance(retries, int)
        or isinstance(retries, bool)
        or not 0 <= retries <= MAX_RETRY_CAP
    ):
        errors.append("reconstruction_gpu_explicit_retry_cap_invalid")
    if (
        not isinstance(source.get("authority_id"), str)
        or not str(source.get("authority_id")).strip()
    ):
        errors.append("reconstruction_gpu_paid_authority_missing")
    if source.get("proof_effect") != "none":
        errors.append("reconstruction_gpu_request_proof_effect_invalid")
    if errors:
        raise ValueError(";".join(sorted(set(errors))))
    source["request_digest"] = canonical_digest(source, digest_field="request_digest")
    return source


def collect_reconstruction_vast_preflight(
    *,
    name_prefix: str,
    container_disk_bytes: int,
    watchdog: Mapping[str, Any],
    conflicting_owner_present: bool,
    capacity_probe: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    inventory_probe: Callable[[str], Mapping[str, Any]],
    max_hourly_rate_usd: float,
    minimum_gpu_ram_mb: int = 24_000,
    minimum_reliability: float = 0.98,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Collect mutation-free Vast capacity and provider-zero evidence."""

    capacity_request = {
        "max_hourly_rate_usd": float(max_hourly_rate_usd),
        "min_gpu_ram_mb": int(minimum_gpu_ram_mb),
        "min_reliability": float(minimum_reliability),
        "require_avx": True,
        "require_known_supported_isaac_driver": False,
        "require_direct_port": False,
        "preferred_gpu_keywords": [
            "A40",
            "RTX A6000",
            "L40",
            "L40S",
            "RTX 6000Ada",
        ],
    }
    capacity = dict(capacity_probe(capacity_request))
    scoped_inventory = dict(inventory_probe(name_prefix))
    global_inventory = dict(inventory_probe(""))
    selected_value = capacity.get("selected_offer")
    selected = dict(selected_value) if isinstance(selected_value, Mapping) else {}
    gpu_memory = int(selected.get("gpu_ram_mb") or 0) * 1_000_000
    price = float(
        selected.get("on_demand_price_usd_per_hour") or selected.get("hourly_rate_usd") or 0
    )
    provider_api_verified = bool(
        capacity.get("status") == "available"
        and scoped_inventory.get("api_confirmed") is True
        and global_inventory.get("api_confirmed") is True
    )
    provider_zero = bool(
        scoped_inventory.get("live_resource_count") == 0
        and global_inventory.get("live_resource_count") == 0
    )
    single_gpu_available = bool(
        selected and gpu_memory >= MIN_GPU_MEMORY_BYTES and 0 < price <= float(max_hourly_rate_usd)
    )
    watchdog_value = dict(watchdog)
    blockers: list[str] = []
    if not provider_api_verified:
        blockers.append("reconstruction_gpu_provider_api_not_verified")
    if not provider_zero:
        blockers.append("reconstruction_gpu_provider_inventory_not_zero")
    if conflicting_owner_present:
        blockers.append("reconstruction_gpu_conflicting_owner_present")
    if (
        watchdog_value.get("status") != "armed"
        or watchdog_value.get("independent_process") is not True
    ):
        blockers.append("reconstruction_gpu_independent_watchdog_not_armed")
    if not single_gpu_available:
        blockers.append("reconstruction_gpu_single_gpu_unavailable")
    if container_disk_bytes < MIN_CONTAINER_DISK_BYTES:
        blockers.append("reconstruction_gpu_container_disk_below_floor")
    result = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "verified" if not blockers else "blocked",
        "provider": "vast",
        "observed_at_epoch": float(clock()),
        "provider_api_verified": provider_api_verified,
        "provider_inventory_verified_zero": provider_zero,
        "conflicting_owner_present": bool(conflicting_owner_present),
        "watchdog": watchdog_value,
        "single_gpu_available": single_gpu_available,
        "gpu_type_id": selected.get("gpu_name") or selected.get("gpu_type_id"),
        "gpu_memory_bytes": gpu_memory,
        "container_disk_bytes": int(container_disk_bytes),
        "on_demand_price_usd_per_hour": price or None,
        "selected_offer": selected or None,
        "capacity_request": capacity_request,
        "capacity_snapshot": capacity,
        "scoped_billable_inventory": scoped_inventory,
        "global_billable_inventory": global_inventory,
        "blockers": sorted(set(blockers)),
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
        "capacity_reserved": False,
        "proof_effect": "none",
        "claim_ceiling": "provider_capacity_and_zero_inventory_snapshot_only",
    }
    result["preflight_digest"] = canonical_digest(result, digest_field="preflight_digest")
    return result


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected_json_object:{path}")
    return dict(value)


def _finite(value: Any, *, minimum: float = 0.0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= minimum
    )


def build_reconstruction_gpu_canary_admission(
    *,
    request: Mapping[str, Any],
    preflight: Mapping[str, Any],
    provider: str,
    expected_source_commit: str,
    checkout_source_commit: str,
    checkout_clean: bool,
    max_spend_usd: float | None,
    hard_ttl_seconds: int | None,
    retry_cap: int | None,
    authority_id: str | None,
    execute: bool,
    execution_adapter_qualified: bool = False,
    image_release: Mapping[str, Any] | None = None,
    measurement_isaac_runtime_release: Mapping[str, Any] | None = None,
    measurement_dlo_lab_runtime_release: Mapping[str, Any] | None = None,
    observed_now_epoch: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind a canary request to immutable inputs without allocating a machine."""

    source = json.loads(json.dumps(dict(request)))
    provider_snapshot = json.loads(json.dumps(dict(preflight)))
    blockers: list[str] = []

    supplied_digest = source.pop("request_digest", None)
    expected_request_digest = canonical_digest(source, digest_field="request_digest")
    source["request_digest"] = supplied_digest
    if supplied_digest != expected_request_digest:
        blockers.append("reconstruction_gpu_request_digest_mismatch")
    if source.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("reconstruction_gpu_request_schema_invalid")
    if source.get("operation") not in OPERATIONS:
        blockers.append("reconstruction_gpu_operation_unsupported")
    expected_result_schema = EXPECTED_RUNTIME_RESULT_SCHEMAS.get(str(source.get("operation") or ""))
    if source.get("expected_runtime_result_schema") != expected_result_schema:
        blockers.append("reconstruction_gpu_expected_runtime_result_schema_invalid")
    if source.get("capture_profile") not in CAPTURE_PROFILES:
        blockers.append("reconstruction_gpu_capture_profile_unsupported")
    if source.get("source_commit_sha") != expected_source_commit:
        blockers.append("reconstruction_gpu_request_source_commit_mismatch")
    if _COMMIT.fullmatch(expected_source_commit) is None:
        blockers.append("reconstruction_gpu_expected_source_commit_invalid")
    if checkout_source_commit != expected_source_commit:
        blockers.append("reconstruction_gpu_checkout_source_commit_mismatch")
    if not checkout_clean:
        blockers.append("reconstruction_gpu_checkout_not_clean")
    if _IMAGE.fullmatch(str(source.get("worker_image_digest") or "")) is None:
        blockers.append("reconstruction_gpu_worker_image_digest_invalid")
    for key in (
        "worker_stack_manifest_digest",
        "deterministic_configuration_digest",
        "operation_request_digest",
        "operation_input_bundle_digest",
    ):
        if _DIGEST.fullmatch(str(source.get(key) or "")) is None:
            blockers.append(f"reconstruction_gpu_{key}_invalid")
    operation = str(source.get("operation") or "")
    if operation in EXTERNAL_DERIVED_ISAAC_OPERATIONS:
        required_external_digests = (
            ("external_import_receipt_digest", "provider_qualification_report_digest")
            if operation == PROVIDER_NUREC_ISAAC_OPERATION
            else (
                "remote_processing_authorization_digest",
                "package_result_digest",
                "collision_candidate_digest",
                "scene_frame_binding_digest",
                "target_analysis_digest",
                "target_binding_digest",
                "placement_proposal_digest",
            )
        )
        for key in required_external_digests:
            if _DIGEST.fullmatch(str(source.get(key) or "")) is None:
                blockers.append(f"reconstruction_gpu_{key}_invalid")
        for key in (
            "reconstruction_dataset_digest",
            "frozen_split_digest",
            "calibration_digest",
        ):
            if key in source:
                boundary = "provider" if operation == PROVIDER_NUREC_ISAAC_OPERATION else "external"
                blockers.append(f"reconstruction_gpu_{boundary}_capture_binding_forbidden:{key}")
        for key, expected in {
            "source_relationship_to_blueprint_raw_capture": "none",
            "external_derived_support_asset": True,
            "blueprint_raw_capture_truth": False,
        }.items():
            if source.get(key) != expected:
                boundary = "provider" if operation == PROVIDER_NUREC_ISAAC_OPERATION else "external"
                blockers.append(f"reconstruction_gpu_{boundary}_source_boundary_invalid:{key}")
        if operation == EXTERNAL_SCENE_ISAAC_OPERATION:
            for key, expected in {
                "source_video_available": False,
                "source_video_required_for_candidate_execution": False,
                "independent_metric_scale_proven": False,
                "remote_upload_authorized": True,
                "paid_compute_authorized": True,
            }.items():
                if source.get(key) != expected:
                    blockers.append(f"reconstruction_gpu_external_boundary_invalid:{key}")
    else:
        for key in (
            "reconstruction_dataset_digest",
            "frozen_split_digest",
            "calibration_digest",
        ):
            if _DIGEST.fullmatch(str(source.get(key) or "")) is None:
                blockers.append(f"reconstruction_gpu_{key}_invalid")
    if source.get("candidate_may_read_hidden_heldout") is not False:
        blockers.append("reconstruction_gpu_hidden_heldout_access_forbidden")
    if source.get("trainer_may_grade_heldout") is not False:
        blockers.append("reconstruction_gpu_trainer_self_grading_forbidden")
    if source.get("proof_effect") != "none":
        blockers.append("reconstruction_gpu_request_proof_effect_invalid")

    image_release_digest: str | None = None
    if source.get("operation") in {
        "isaac_canary",
        "provider_nurec_isaac_canary",
        "external_scene_isaac_canary",
    }:
        if image_release is None:
            if execute:
                blockers.append("reconstruction_isaac_image_release_missing")
        else:
            try:
                release = validate_reconstruction_isaac_image_release(image_release)
            except ReconstructionIsaacImageReleaseError:
                blockers.append("reconstruction_isaac_image_release_invalid")
            else:
                image_release_digest = str(release["image_release_digest"])
                if release.get("resolved_image_digest") != source.get("worker_image_digest"):
                    blockers.append("reconstruction_isaac_image_release_digest_mismatch")
                if release.get("source_commit_sha") != source.get("source_commit_sha"):
                    blockers.append("reconstruction_isaac_image_release_source_mismatch")
    measurement_runtime_release_digest: str | None = None
    if source.get("operation") == "measurement_isaac_canary":
        if measurement_isaac_runtime_release is None:
            if execute:
                blockers.append("measurement_isaac_runtime_release_missing")
        else:
            try:
                measurement_release = validate_measurement_isaac_runtime_release(
                    measurement_isaac_runtime_release
                )
            except MeasurementIsaacRuntimeReleaseError:
                blockers.append("measurement_isaac_runtime_release_invalid")
            else:
                measurement_runtime_release_digest = str(
                    measurement_release["runtime_release_digest"]
                )
                if measurement_release.get("resolved_image_digest") != source.get(
                    "worker_image_digest"
                ):
                    blockers.append("measurement_isaac_runtime_release_image_mismatch")
    measurement_dlo_runtime_release_digest: str | None = None
    if source.get("operation") == "measurement_dlo_lab_canary":
        if measurement_dlo_lab_runtime_release is None:
            if execute:
                blockers.append("measurement_dlo_lab_runtime_release_missing")
        else:
            try:
                dlo_release = validate_measurement_dlo_lab_runtime_release(
                    measurement_dlo_lab_runtime_release
                )
            except MeasurementDloLabRuntimeReleaseError:
                blockers.append("measurement_dlo_lab_runtime_release_invalid")
            else:
                measurement_dlo_runtime_release_digest = str(
                    dlo_release["runtime_release_digest"]
                )
                if dlo_release.get("runtime_image_digest") != source.get(
                    "worker_image_digest"
                ):
                    blockers.append("measurement_dlo_lab_runtime_release_image_mismatch")

    if provider != "vast" or provider_snapshot.get("provider") != "vast":
        blockers.append("reconstruction_gpu_vast_first_required")
    if provider_snapshot.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        blockers.append("reconstruction_gpu_preflight_schema_invalid")
    if provider_snapshot.get("status") != "verified":
        blockers.append("reconstruction_gpu_preflight_not_verified")
    if provider_snapshot.get("provider_api_verified") is not True:
        blockers.append("reconstruction_gpu_provider_api_not_verified")
    if provider_snapshot.get("provider_inventory_verified_zero") is not True:
        blockers.append("reconstruction_gpu_provider_inventory_not_zero")
    if provider_snapshot.get("conflicting_owner_present") is not False:
        blockers.append("reconstruction_gpu_conflicting_owner_present")
    watchdog = provider_snapshot.get("watchdog")
    watchdog = watchdog if isinstance(watchdog, Mapping) else {}
    if watchdog.get("status") != "armed" or watchdog.get("independent_process") is not True:
        blockers.append("reconstruction_gpu_independent_watchdog_not_armed")
    if provider_snapshot.get("single_gpu_available") is not True:
        blockers.append("reconstruction_gpu_single_gpu_unavailable")
    gpu_memory = provider_snapshot.get("gpu_memory_bytes")
    if (
        not isinstance(gpu_memory, int)
        or isinstance(gpu_memory, bool)
        or gpu_memory < MIN_GPU_MEMORY_BYTES
    ):
        blockers.append("reconstruction_gpu_memory_below_floor")
    disk = provider_snapshot.get("container_disk_bytes")
    if not isinstance(disk, int) or isinstance(disk, bool) or disk < MIN_CONTAINER_DISK_BYTES:
        blockers.append("reconstruction_gpu_container_disk_below_floor")
    hourly = provider_snapshot.get("on_demand_price_usd_per_hour")
    if not _finite(hourly, minimum=0.000001):
        blockers.append("reconstruction_gpu_hourly_price_invalid")
    observed = provider_snapshot.get("observed_at_epoch")
    now = time.time() if observed_now_epoch is None else float(observed_now_epoch)
    if not _finite(observed):
        blockers.append("reconstruction_gpu_preflight_observed_at_invalid")
    elif not 0 <= now - float(observed) <= MAX_PREFLIGHT_AGE_SECONDS:
        blockers.append("reconstruction_gpu_preflight_stale_or_future")

    if not _finite(max_spend_usd, minimum=0.000001):
        blockers.append("reconstruction_gpu_explicit_budget_missing")
    if (
        not isinstance(hard_ttl_seconds, int)
        or isinstance(hard_ttl_seconds, bool)
        or not 1 <= hard_ttl_seconds <= MAX_TTL_SECONDS
    ):
        blockers.append("reconstruction_gpu_explicit_ttl_invalid")
    if (
        not isinstance(retry_cap, int)
        or isinstance(retry_cap, bool)
        or not 0 <= retry_cap <= MAX_RETRY_CAP
    ):
        blockers.append("reconstruction_gpu_explicit_retry_cap_invalid")
    if not isinstance(authority_id, str) or not authority_id.strip():
        blockers.append("reconstruction_gpu_paid_authority_missing")
    if max_spend_usd != source.get("max_spend_usd"):
        blockers.append("reconstruction_gpu_budget_binding_mismatch")
    if hard_ttl_seconds != source.get("hard_ttl_seconds"):
        blockers.append("reconstruction_gpu_ttl_binding_mismatch")
    if retry_cap != source.get("retry_cap"):
        blockers.append("reconstruction_gpu_retry_binding_mismatch")
    if authority_id != source.get("authority_id"):
        blockers.append("reconstruction_gpu_authority_binding_mismatch")
    worst_case_cost = (
        float(hourly) * float(hard_ttl_seconds) / 3600.0
        if _finite(hourly, minimum=0.000001)
        and isinstance(hard_ttl_seconds, int)
        and not isinstance(hard_ttl_seconds, bool)
        else math.inf
    )
    if _finite(max_spend_usd, minimum=0.000001) and worst_case_cost > float(max_spend_usd):
        blockers.append("reconstruction_gpu_budget_below_worst_case_cost")

    operation = source.get("operation")
    operation_adapter_qualified = bool(
        execution_adapter_qualified and operation in EXECUTABLE_OPERATIONS
    )
    if execute and operation not in EXECUTABLE_OPERATIONS:
        blockers.append("reconstruction_gpu_operation_execution_adapter_unavailable")
    elif execute and not operation_adapter_qualified and not blockers:
        blockers.append("reconstruction_vast_execution_adapter_not_qualified")
    bound_request = {
        **source,
        "request_digest": expected_request_digest,
        "bound_provider": provider,
        "bound_preflight_digest": canonical_digest(provider_snapshot),
        "bound_checkout_source_commit": checkout_source_commit,
        "bound_checkout_clean": checkout_clean,
        "isaac_image_release_digest": image_release_digest,
        "measurement_isaac_runtime_release_digest": measurement_runtime_release_digest,
        "measurement_dlo_lab_runtime_release_digest": measurement_dlo_runtime_release_digest,
        "provider_mutation_authorized": bool(
            execute and operation_adapter_qualified and not blockers
        ),
    }
    bound_request["bound_request_digest"] = canonical_digest(
        bound_request, digest_field="bound_request_digest"
    )
    status = (
        "execute_ready"
        if execute and operation_adapter_qualified and not blockers
        else ("dry_run_ready" if not blockers else "blocked")
    )
    admission = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "status": status,
        "probe_kind": PROBE_KIND,
        "provider": provider,
        "blockers": sorted(set(blockers)),
        "request_digest": expected_request_digest,
        "bound_request_digest": bound_request["bound_request_digest"],
        "source_commit_sha": checkout_source_commit,
        "operation": operation,
        "operation_request_digest": source.get("operation_request_digest"),
        "operation_input_bundle_digest": source.get("operation_input_bundle_digest"),
        "expected_runtime_result_schema": source.get("expected_runtime_result_schema"),
        "worker_image_digest": source.get("worker_image_digest"),
        "isaac_image_release_digest": image_release_digest,
        "measurement_isaac_runtime_release_digest": measurement_runtime_release_digest,
        "measurement_dlo_lab_runtime_release_digest": measurement_dlo_runtime_release_digest,
        "reconstruction_dataset_digest": source.get("reconstruction_dataset_digest"),
        "frozen_split_digest": source.get("frozen_split_digest"),
        "external_import_receipt_digest": source.get("external_import_receipt_digest"),
        "provider_qualification_report_digest": source.get("provider_qualification_report_digest"),
        "max_spend_usd": max_spend_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": retry_cap,
        "authority_id": authority_id,
        "watchdog_armed": watchdog.get("status") == "armed",
        "provider_zero_verified": provider_snapshot.get("provider_inventory_verified_zero") is True,
        "provider_mutations_performed": 0,
        "paid_execution_started": False,
        "execution_adapter_qualified": operation_adapter_qualified,
        "allocation_success_is_scientific_success": False,
        "proof_effect": "none",
        "claim_ceiling": "paid_gpu_admission_only",
        "legal_next_actions": (
            ["qualify_reconstruction_operation_execution_adapter"]
            if operation not in EXECUTABLE_OPERATIONS
            else (
                ["qualify_vast_execution_adapter"]
                if blockers == ["reconstruction_vast_execution_adapter_not_qualified"]
                else (
                    ["invoke_canonical_gpu_canary_with_explicit_execute_authority"]
                    if not blockers
                    else ["resolve_admission_blockers"]
                )
            )
        ),
    }
    admission["admission_digest"] = canonical_digest(admission, digest_field="admission_digest")
    return admission, bound_request


def prepare_reconstruction_gpu_canary(
    *,
    request_path: str | Path,
    preflight_path: str | Path,
    admission_out: str | Path,
    bound_request_out: str | Path,
    adapter_output: str | Path,
    provider: str,
    expected_source_commit: str,
    checkout_source_commit: str,
    checkout_clean: bool,
    max_spend_usd: float | None,
    hard_ttl_seconds: int | None,
    retry_cap: int | None,
    authority_id: str | None,
    execute: bool,
    execution_adapter_qualified: bool = False,
    image_release_path: str | Path | None = None,
    measurement_isaac_runtime_release_path: str | Path | None = None,
    measurement_dlo_lab_runtime_release_path: str | Path | None = None,
) -> dict[str, Any]:
    image_release = _read(image_release_path) if image_release_path else None
    measurement_runtime_release = (
        _read(measurement_isaac_runtime_release_path)
        if measurement_isaac_runtime_release_path
        else None
    )
    measurement_dlo_runtime_release = (
        _read(measurement_dlo_lab_runtime_release_path)
        if measurement_dlo_lab_runtime_release_path
        else None
    )
    admission, bound = build_reconstruction_gpu_canary_admission(
        request=_read(request_path),
        preflight=_read(preflight_path),
        provider=provider,
        expected_source_commit=expected_source_commit,
        checkout_source_commit=checkout_source_commit,
        checkout_clean=checkout_clean,
        max_spend_usd=max_spend_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        retry_cap=retry_cap,
        authority_id=authority_id,
        execute=execute,
        execution_adapter_qualified=execution_adapter_qualified,
        image_release=image_release,
        measurement_isaac_runtime_release=measurement_runtime_release,
        measurement_dlo_lab_runtime_release=measurement_dlo_runtime_release,
    )
    write_json(Path(admission_out), admission)
    write_json(Path(bound_request_out), bound)
    write_json(
        Path(adapter_output),
        {
            "schema_version": "reconstruction_gpu_canary_adapter_result.v1",
            "status": "not_started",
            "admission_digest": admission["admission_digest"],
            "provider_mutations_performed": 0,
            "cost_usd": 0.0,
            "proof_effect": "none",
            "claim_ceiling": "no_execution_evidence",
        },
    )
    return admission


__all__ = [
    "ADMISSION_SCHEMA_VERSION",
    "EXECUTABLE_OPERATIONS",
    "EXPECTED_RUNTIME_RESULT_SCHEMAS",
    "PREFLIGHT_SCHEMA_VERSION",
    "PROBE_KIND",
    "REQUEST_SCHEMA_VERSION",
    "build_reconstruction_gpu_canary_request",
    "build_reconstruction_gpu_canary_admission",
    "collect_reconstruction_vast_preflight",
    "prepare_reconstruction_gpu_canary",
]
