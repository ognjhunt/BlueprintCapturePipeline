"""Fail-closed contracts for the ADP-009D PhysX/Newton controls comparison.

This module never launches a provider.  It binds one physics backend at run
construction, validates provider-free profile bytes, validates native capability
probe receipts, and compiles a backend-neutral controls comparison receipt.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

try:  # pragma: no cover - flat layout inside the sealed runtime bundle
    from decision_evidence_contracts import canonical_digest
except ImportError:  # pragma: no cover - installed package layout
    from .decision_evidence_contracts import canonical_digest


ALLOWED_PHYSICS_BACKENDS = ("physx", "newton")
DEFAULT_PHYSICS_BACKEND = "physx"
BACKEND_PROFILE_SCHEMA_VERSION = "adp009d_physics_backend_profile.v1"
PROBE_SCHEMA_VERSION = "adp009d_physics_backend_probe.v1"
CANARY_ADMISSION_SCHEMA_VERSION = "adp009d_newton_canary_admission.v1"
CANARY_TERMINAL_SCHEMA_VERSION = "adp009d_newton_canary_terminal.v1"
CONTROL_RUN_SCHEMA_VERSION = "adp009d_physics_backend_control_run.v1"
COMPARISON_SCHEMA_VERSION = "adp009d_physics_backend_comparison.v1"

CONTAINER_IMAGE = (
    "nvcr.io/nvidia/isaac-sim:6.0.0-dev2@"
    "sha256:c3e7bef5b2bfdb9972807c34195206078372bf8c6cff79716be130a3fe3e9ce9"
)
ISAAC_SIM_VERSION = "6.0.0-dev2"
ARENA_REVISION = "8b4a3a47fc53de23e8205089d71109a2e2348acd"
ARENA_TREE = "03f31f3dd56c56d00f24dbfb09711ec0ab345de8"
ISAAC_LAB_REVISION = "e57379c634b42db5a0fe9f754341be6e2a7c7c43"
ISAAC_LAB_TREE = "454115265327a80acabd07cbd36e10071fc0c065"
NEWTON_REVISION = "2684d75bfa4bb8b058a93b81c458a74b7701c997"
NEWTON_TREE = "46bd29c9b0027cc31116f60250b61f2ac20b7aaa"
SOURCE_BUNDLE_DIGEST = (
    "sha256:4cbf6781cd43cdf02353e0417aefd9ee4df1a65a99e7dbb2ef69a0a0170f22ba"
)
SCENARIO_INSTANCE_DIGEST = (
    "sha256:243c0e62697da0298081a53c6530cee16cf94cde5a73df08f3773629b52c3001"
)
APPROVED_CAN_DIGEST = (
    "sha256:61c2a03bef425803d82cc5ef24ced5b2ccb4160923c53bb10c6ad0e3f52532ec"
)
SAGE_COLLISION_DIGEST = (
    "sha256:b265706c24f6a8ace3ee6743fd138583c4e21d83f61b99a06fd435e6ac2d6b41"
)

PHYSX_ONLY_FIELD_NAMES = frozenset(
    {
        "sdf_margin_m",
        "sdf_narrow_band_thickness_m",
        "sdf_resolution",
        "sdf_subgrid_resolution",
        "gpu_max_rigid_contact_count",
        "gpu_max_rigid_patch_count",
        "solver_position_iteration_count",
        "solver_velocity_iteration_count",
        "enable_enhanced_determinism",
    }
)

COMPARABILITY_BINDINGS = (
    "source_bundle_digest",
    "sealed_scene_digest",
    "task_object_digest",
    "robot_profile_digest",
    "scenario_cell_id",
    "scenario_instance_digest",
    "seed",
    "intended_grasp_geometry_digest",
    "controls_plan_semantic_digest",
)

MEASUREMENT_FIELDS = (
    "initialization_reset",
    "target_robot_pose",
    "contacts_and_force_vectors",
    "torque_utilization_and_clipping",
    "closest_geometric_clearance",
    "action_delivery",
    "phase_completion",
    "lossless_frames",
    "review_media",
    "teardown",
    "spend",
    "provider_zero",
)


class PhysicsBackendContractError(ValueError):
    """Raised when a comparison contract would be ambiguous or overclaiming."""


def normalize_physics_backend(value: object) -> str:
    """Return one exact backend id; aliases and missing values are rejected."""

    if not isinstance(value, str) or value not in ALLOWED_PHYSICS_BACKENDS:
        raise PhysicsBackendContractError("adp009d_physics_backend_invalid")
    return value


def _runtime_identity() -> dict[str, Any]:
    return {
        "container_image": CONTAINER_IMAGE,
        "isaac_sim_version": ISAAC_SIM_VERSION,
        "isaac_lab_arena": {
            "revision": ARENA_REVISION,
            "tree": ARENA_TREE,
        },
        "isaac_lab": {
            "revision": ISAAC_LAB_REVISION,
            "tree": ISAAC_LAB_TREE,
            "core_version": "4.5.24",
        },
    }


def build_backend_profile(physics_backend: str) -> dict[str, Any]:
    """Build the immutable provider-free profile for one backend."""

    backend = normalize_physics_backend(physics_backend)
    common: dict[str, Any] = {
        "schema_version": BACKEND_PROFILE_SCHEMA_VERSION,
        "profile_id": f"adp009d-840313-{backend}-controls-v1",
        "program_id": "arm-decision-proof-v1",
        "physics_backend": backend,
        "backend_selected_at_simulation_construction": True,
        "mid_run_backend_switch_allowed": False,
        "controls_only": True,
        "policy_query_allowed": False,
        "candidate_outcome_access_allowed": False,
        "runtime_identity": _runtime_identity(),
        "source_bindings": {
            "source_bundle_digest": SOURCE_BUNDLE_DIGEST,
            "scenario_instance_digest": SCENARIO_INSTANCE_DIGEST,
            "approved_can_digest": APPROVED_CAN_DIGEST,
            "sage_collision_digest": SAGE_COLLISION_DIGEST,
        },
        "required_capabilities": {
            "franka_import": True,
            "robotiq_import": True,
            "eight_dimensional_actuation_roundtrip": True,
            "lossless_camera_frame_retention": True,
            "contact_force_vector_readback": True,
            "contact_partner_readback": True,
            "joint_torque_and_limit_readback": True,
            "closest_geometric_clearance_measurement": True,
        },
        "claim_ceiling": "controls_comparison_evidence_only",
    }
    if backend == "physx":
        common.update(
            {
                "maturity": "production_baseline",
                "backend_runtime": {
                    "package": "isaaclab_physx",
                    "version": "0.5.13",
                },
                "solver_configuration": {
                    "configuration_schema": "adp009d_physx_tgs_solver.v1",
                    "solver": "TGS",
                    "dt_seconds": 1.0 / 120.0,
                    "decimation": 8,
                    "enable_enhanced_determinism": True,
                    "gpu_max_rigid_contact_count": 2**23,
                    "gpu_max_rigid_patch_count": 2**15,
                },
                "contact_model": {
                    "configuration_schema": "adp009d_physx_sdf_contact_model.v1",
                    "source": "approved_can_physx_sdf_adapter.usda",
                    "approximation": "sdf",
                    "sdf_margin_m": 0.0025,
                    "sdf_narrow_band_thickness_m": 0.0025,
                    "sdf_resolution": 256,
                    "sdf_subgrid_resolution": 6,
                    "finger_contact_offset_m": 0.005,
                    "effective_contact_envelope_m": 0.01,
                    "semantics": "physx_specific_not_portable",
                },
                "asset_conversion": {
                    "mode": "physx_sdf_overlay",
                    "source_usd_mutated": False,
                    "conversion_receipt_required": True,
                },
            }
        )
    else:
        common.update(
            {
                "maturity": "experimental_comparison_candidate",
                "backend_runtime": {
                    "package": "isaaclab_newton",
                    "version": "0.5.9",
                    "newton": {
                        "revision": NEWTON_REVISION,
                        "tree": NEWTON_TREE,
                        "package_version": "1.1.0.dev0",
                    },
                    "warp_version": "1.12.0",
                    "mujoco_version": "3.5.0",
                    "mujoco_warp_version": "3.5.0.2",
                },
                "solver_configuration": {
                    "configuration_schema": "adp009d_newton_mjwarp_solver.v1",
                    "solver": "mujoco_warp",
                    "integrator": "implicitfast",
                    "solver_algorithm": "newton",
                    "cone": "pyramidal",
                    "dt_seconds": 1.0 / 120.0,
                    "decimation": 8,
                    "num_substeps": 1,
                    "iterations": 100,
                    "line_search_iterations": 20,
                    "njmax": 2048,
                    "nconmax": 1024,
                    "use_mujoco_contacts": True,
                    "use_mujoco_cpu": False,
                    "use_cuda_graph": True,
                },
                "contact_model": {
                    "configuration_schema": "adp009d_newton_mjwarp_contact_model.v1",
                    "contact_generation": "mujoco_warp_from_generic_usd_import",
                    "friction_cone": "pyramidal",
                    "contact_buffer_limit": {"field": "nconmax", "value": 1024},
                    "physx_sdf_overlay_allowed": False,
                    "physx_sdf_semantics_inherited": False,
                    "unsupported_or_ignored_setting_policy": "block",
                },
                "asset_conversion": {
                    "mode": "generic_usd_overlay_then_newton_model_builder",
                    "source_usd_mutated": False,
                    "source_approximation_token": "sdf",
                    "source_approximation_token_blocked_by_overlay": True,
                    "adapter": "approved_can_newton_generic_adapter.usda",
                    "source_approximation_semantics_assumed": False,
                    "runtime_conversion_receipt_required": True,
                    "franka_robotiq_conversion_probe_required": True,
                },
            }
        )
    common["profile_digest"] = canonical_digest(common, digest_field="profile_digest")
    return common


def build_backend_contact_configuration(physics_backend: str) -> dict[str, Any]:
    """Build the separately explicit contact and planner configuration."""

    profile = build_backend_profile(physics_backend)
    backend = str(profile["physics_backend"])
    contact_model = dict(profile["contact_model"])
    if backend == "physx":
        planner_allowance = float(contact_model["effective_contact_envelope_m"])
        allowance_source = "physx_sdf_margin_plus_narrow_band_plus_finger_contact_offset"
    else:
        planner_allowance = 0.0
        allowance_source = "generic_surface_geometry_only_no_physx_standoff_assumed"
    configuration: dict[str, Any] = {
        "schema_version": "adp009d_backend_contact_configuration.v1",
        "physics_backend": backend,
        "backend_profile_digest": profile["profile_digest"],
        "contact_model": contact_model,
        "planner_contact_allowance_m": planner_allowance,
        "planner_contact_allowance_source": allowance_source,
        "actual_contact_onset_and_clearance_measurement_required": True,
        "configuration_digest": "",
    }
    configuration["configuration_digest"] = canonical_digest(
        configuration, digest_field="configuration_digest"
    )
    return configuration


def _nested_field_names(value: object) -> set[str]:
    names: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            names.add(str(key))
            names.update(_nested_field_names(child))
    elif isinstance(value, list):
        for child in value:
            names.update(_nested_field_names(child))
    return names


def validate_backend_contact_configuration(value: Mapping[str, Any]) -> list[str]:
    """Reject contact configuration drift and cross-backend field leakage."""

    try:
        backend = normalize_physics_backend(value.get("physics_backend"))
    except PhysicsBackendContractError as exc:
        return [str(exc)]
    blockers: list[str] = []
    if dict(value) != build_backend_contact_configuration(backend):
        blockers.append("adp009d_backend_contact_configuration_drifted")
    if value.get("configuration_digest") != canonical_digest(
        value, digest_field="configuration_digest"
    ):
        blockers.append("adp009d_backend_contact_configuration_digest_invalid")
    if backend == "newton" and PHYSX_ONLY_FIELD_NAMES.intersection(
        _nested_field_names(value)
    ):
        blockers.append("adp009d_newton_contact_configuration_contains_physx_only_fields")
    return sorted(set(blockers))


def validate_backend_profile(value: Mapping[str, Any]) -> list[str]:
    """Validate a profile byte-for-byte against the pinned contract."""

    blockers: list[str] = []
    try:
        backend = normalize_physics_backend(value.get("physics_backend"))
    except PhysicsBackendContractError as exc:
        return [str(exc)]
    expected = build_backend_profile(backend)
    if dict(value) != expected:
        blockers.append("adp009d_backend_profile_drifted")
    if value.get("profile_digest") != canonical_digest(value, digest_field="profile_digest"):
        blockers.append("adp009d_backend_profile_digest_invalid")
    if value.get("mid_run_backend_switch_allowed") is not False:
        blockers.append("adp009d_backend_switch_not_fail_closed")
    if backend == "newton":
        forbidden = PHYSX_ONLY_FIELD_NAMES.intersection(_nested_field_names(value))
        if forbidden:
            blockers.append("adp009d_newton_profile_contains_physx_only_fields")
        contact = value.get("contact_model")
        contact_row = dict(contact) if isinstance(contact, Mapping) else {}
        if (
            contact_row.get("physx_sdf_overlay_allowed") is not False
            or contact_row.get("physx_sdf_semantics_inherited") is not False
        ):
            blockers.append("adp009d_newton_contact_model_not_distinct")
    return sorted(set(blockers))


def _finite_vector(value: object, *, minimum_length: int = 3) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= minimum_length
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in value
        )
    )


def validate_backend_probe(
    value: Mapping[str, Any], *, profile: Mapping[str, Any]
) -> list[str]:
    """Validate native evidence that a pinned backend supports this exact task."""

    blockers = validate_backend_profile(profile)
    backend = str(profile.get("physics_backend") or "")
    if (
        value.get("schema_version") != PROBE_SCHEMA_VERSION
        or value.get("status") != "passed"
        or value.get("physics_backend") != backend
        or value.get("backend_profile_digest") != profile.get("profile_digest")
        or value.get("probe_digest") != canonical_digest(value, digest_field="probe_digest")
    ):
        blockers.append("adp009d_backend_probe_identity_invalid")
    if (
        value.get("backend_active_at_simulation_construction") is not True
        or value.get("backend_switch_attempted") is not False
        or value.get("backend_switch_observed") is not False
    ):
        blockers.append("adp009d_backend_probe_immutability_invalid")
    if value.get("runtime_identity") != profile.get("runtime_identity"):
        blockers.append("adp009d_backend_probe_runtime_identity_invalid")
    if value.get("source_bindings") != profile.get("source_bindings"):
        blockers.append("adp009d_backend_probe_source_binding_invalid")
    capabilities = value.get("capabilities")
    capability_row = dict(capabilities) if isinstance(capabilities, Mapping) else {}
    required = dict(profile.get("required_capabilities") or {})
    if set(capability_row) != set(required) or any(
        capability_row.get(name) is not True for name in required
    ):
        blockers.append("adp009d_backend_probe_capability_missing")
    if value.get("solver_configuration") != profile.get("solver_configuration"):
        blockers.append("adp009d_backend_probe_solver_configuration_invalid")
    contact = value.get("contact_readback")
    contact_row = dict(contact) if isinstance(contact, Mapping) else {}
    force_vectors = contact_row.get("force_vectors_world_n")
    if (
        not isinstance(force_vectors, list)
        or not force_vectors
        or any(not _finite_vector(vector) for vector in force_vectors)
        or not isinstance(contact_row.get("partner_prim_paths"), list)
        or not contact_row.get("partner_prim_paths")
    ):
        blockers.append("adp009d_backend_probe_contact_readback_invalid")
    conversion = value.get("asset_conversion")
    conversion_row = dict(conversion) if isinstance(conversion, Mapping) else {}
    if (
        conversion_row.get("source_asset_digest") != APPROVED_CAN_DIGEST
        or not isinstance(conversion_row.get("converted_model_digest"), str)
        or not str(conversion_row.get("converted_model_digest")).startswith("sha256:")
        or conversion_row.get("silently_ignored_settings") != []
    ):
        blockers.append("adp009d_backend_probe_asset_conversion_invalid")
    if backend == "newton":
        if (
            conversion_row.get("physx_sdf_overlay_loaded") is not False
            or conversion_row.get("physx_only_fields_observed") != []
            or value.get("contact_buffer", {}).get("nconmax") != 1024
            or value.get("contact_buffer", {}).get("overflow_observed") is not False
        ):
            blockers.append("adp009d_newton_probe_contact_model_invalid")
    if (
        value.get("policy_query_count") != 0
        or value.get("candidate_outcomes_accessed") is not False
        or value.get("task_success_claimed") is not False
        or value.get("physical_claimed") is not False
    ):
        blockers.append("adp009d_backend_probe_claim_boundary_invalid")
    return sorted(set(blockers))


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def validate_newton_canary_admission(
    value: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    now: datetime | None = None,
) -> list[str]:
    """Validate all non-mutation gates before an allocator may receive Newton."""

    blockers = validate_backend_profile(profile)
    if profile.get("physics_backend") != "newton":
        blockers.append("adp009d_newton_admission_profile_invalid")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expires = _parse_time(value.get("expires_at"))
    issued = _parse_time(value.get("issued_at"))
    allowed_active_ids = value.get("allowed_active_vast_instance_ids")
    allowed_ids_valid = bool(
        isinstance(allowed_active_ids, list)
        and all(
            not isinstance(item, bool) and isinstance(item, int) and item > 0
            for item in allowed_active_ids
        )
        and allowed_active_ids == sorted(set(allowed_active_ids))
    )
    expected_mode = (
        "exact_allowed_concurrency" if allowed_active_ids else "provider_zero"
    )
    if (
        value.get("schema_version") != CANARY_ADMISSION_SCHEMA_VERSION
        or value.get("status") != "passed"
        or value.get("backend_profile_digest") != profile.get("profile_digest")
        or value.get("controls_only") is not True
        or value.get("policy_query_allowed") is not False
        or value.get("candidate_outcome_access_allowed") is not False
        or value.get("canonical_allocator")
        != "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"
        or value.get("explicit_paid_run_authorization") is not True
        or value.get("canonical_spend_admission") is not True
        or value.get("watchdog_required") is not True
        or value.get("artifact_storage_required") is not True
        or value.get("teardown_required") is not True
        or not allowed_ids_valid
        or value.get("provider_inventory_precheck_mode") != expected_mode
        or value.get("provider_zero_precheck_passed") is not (not allowed_active_ids)
        or value.get("exact_concurrency_precheck_passed") is not bool(
            allowed_active_ids
        )
        or value.get("unapproved_live_instance_count") != 0
        or value.get("retry_cap") != 0
        or isinstance(value.get("max_spend_usd"), bool)
        or not isinstance(value.get("max_spend_usd"), (int, float))
        or not math.isfinite(float(value.get("max_spend_usd", math.nan)))
        or float(value.get("max_spend_usd", 0.0)) <= 0.0
        or isinstance(value.get("hard_ttl_seconds"), bool)
        or not isinstance(value.get("hard_ttl_seconds"), int)
        or not 1800 <= int(value.get("hard_ttl_seconds", 0)) <= 14_400
        or expires is None
        or expires <= current
        or (expires - current).total_seconds() > 3600
        or issued is None
        or issued > current + timedelta(seconds=5)
        or (current - issued).total_seconds() > 1800
        or expires <= issued
        or (expires - issued).total_seconds() > 1800
        or value.get("provider_mutation_performed") is not False
        or not isinstance(value.get("authorization_evidence_ref"), str)
        or not str(value.get("authorization_evidence_ref") or "").strip()
        or not isinstance(value.get("canonical_spend_admission_digest"), str)
        or not str(value.get("canonical_spend_admission_digest")).startswith("sha256:")
        or not isinstance(value.get("provider_zero_precheck_digest"), str)
        or not str(value.get("provider_zero_precheck_digest")).startswith("sha256:")
        or value.get("admission_digest")
        != canonical_digest(value, digest_field="admission_digest")
    ):
        blockers.append("adp009d_newton_canary_admission_invalid")
    return sorted(set(blockers))


def build_newton_canary_admission(
    *,
    authorization_evidence_ref: str,
    spend_admission_lock: Mapping[str, Any],
    provider_zero_precheck: Mapping[str, Any],
    max_spend_usd: float,
    hard_ttl_seconds: int,
    allowed_active_vast_instance_ids: Sequence[int] = (),
    issued_at: datetime | None = None,
) -> dict[str, Any]:
    """Compile the current explicit authority and read-only gates into one receipt."""

    authority = str(authorization_evidence_ref or "").strip()
    if not authority:
        raise PhysicsBackendContractError("adp009d_newton_authorization_evidence_missing")
    current = (issued_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        from .spend_admission_lock import validate_spend_admission_lock
    except ImportError:  # pragma: no cover - control-plane builder is package-only
        from spend_admission_lock import validate_spend_admission_lock

    if validate_spend_admission_lock(spend_admission_lock, now=current):
        raise PhysicsBackendContractError("adp009d_newton_spend_admission_invalid")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in allowed_active_vast_instance_ids
    ):
        raise PhysicsBackendContractError(
            "adp009d_newton_allowed_active_vast_instance_invalid"
        )
    allowed_active_ids = sorted(set(allowed_active_vast_instance_ids))
    inventory_rows = provider_zero_precheck.get("inventory_results")
    required_rows = {
        str(row.get("provider")): dict(row)
        for row in inventory_rows or []
        if isinstance(row, Mapping) and row.get("required") is True
    }
    expected_row_counts = {
        "runpod": 0,
        "vast": len(allowed_active_ids),
        "digitalocean": 0,
    }
    observed_instances = [
        dict(row)
        for row in provider_zero_precheck.get("instances") or []
        if isinstance(row, Mapping) and row.get("live") is True
    ]
    observed_allowed_ids: list[int] = []
    invalid_live_instance = False
    for row in observed_instances:
        try:
            instance_id = int(str(row.get("id") or ""))
        except ValueError:
            invalid_live_instance = True
            continue
        if row.get("provider") != "vast":
            invalid_live_instance = True
        observed_allowed_ids.append(instance_id)
    generated_at = _parse_time(provider_zero_precheck.get("generated_at"))
    if (
        provider_zero_precheck.get("schema_version") != "gpu_spend_guard.v1"
        or provider_zero_precheck.get("live_instance_count")
        != len(allowed_active_ids)
        or provider_zero_precheck.get("blockers") != []
        or set(required_rows) != set(expected_row_counts)
        or any(
            required_rows[provider].get("status") != "succeeded"
            or required_rows[provider].get("row_count") != expected_count
            or required_rows[provider].get("blockers") != []
            for provider, expected_count in expected_row_counts.items()
        )
        or invalid_live_instance
        or sorted(observed_allowed_ids) != allowed_active_ids
        or generated_at is None
        or abs((current - generated_at).total_seconds()) > 300
    ):
        raise PhysicsBackendContractError(
            "adp009d_newton_provider_inventory_precheck_invalid"
        )
    profile = build_backend_profile("newton")
    receipt: dict[str, Any] = {
        "schema_version": CANARY_ADMISSION_SCHEMA_VERSION,
        "status": "passed",
        "backend_profile_digest": profile["profile_digest"],
        "controls_only": True,
        "policy_query_allowed": False,
        "candidate_outcome_access_allowed": False,
        "canonical_allocator": (
            "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"
        ),
        "authorization_evidence_ref": authority,
        "issued_at": current.isoformat(),
        "expires_at": (current.replace(microsecond=0) + timedelta(minutes=30)).isoformat(),
        "explicit_paid_run_authorization": True,
        "canonical_spend_admission": True,
        "canonical_spend_admission_digest": canonical_digest(
            spend_admission_lock, digest_field="receipt_digest"
        ),
        "watchdog_required": True,
        "artifact_storage_required": True,
        "teardown_required": True,
        "provider_inventory_precheck_mode": (
            "exact_allowed_concurrency"
            if allowed_active_ids
            else "provider_zero"
        ),
        "allowed_active_vast_instance_ids": allowed_active_ids,
        "provider_zero_precheck_passed": not allowed_active_ids,
        "exact_concurrency_precheck_passed": bool(allowed_active_ids),
        "unapproved_live_instance_count": 0,
        "provider_zero_precheck_digest": canonical_digest(
            provider_zero_precheck, digest_field="receipt_digest"
        ),
        "retry_cap": 0,
        "max_spend_usd": max_spend_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "provider_mutation_performed": False,
        "admission_digest": "",
    }
    receipt["admission_digest"] = canonical_digest(
        receipt, digest_field="admission_digest"
    )
    blockers = validate_newton_canary_admission(
        receipt, profile=profile, now=current
    )
    if blockers:
        raise PhysicsBackendContractError("adp009d_newton_canary_admission_invalid")
    return receipt


def build_newton_canary_terminal_receipt(
    *,
    admission: Mapping[str, Any],
    bundle_receipt: Mapping[str, Any],
    allocator_result: Mapping[str, Any],
    native_result: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
    teardown_manifest: Mapping[str, Any],
    provider_inventory: Mapping[str, Any],
    vast_charge: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a paid Newton canary's terminal evidence without a verdict."""

    profile = build_backend_profile("newton")
    if (
        admission.get("schema_version") != CANARY_ADMISSION_SCHEMA_VERSION
        or admission.get("status") != "passed"
        or admission.get("backend_profile_digest") != profile["profile_digest"]
        or admission.get("controls_only") is not True
        or admission.get("policy_query_allowed") is not False
        or admission.get("retry_cap") != 0
        or admission.get("admission_digest")
        != canonical_digest(admission, digest_field="admission_digest")
    ):
        raise PhysicsBackendContractError("adp009d_newton_terminal_admission_invalid")
    if (
        bundle_receipt.get("status") != "ready"
        or bundle_receipt.get("physics_backend") != "newton"
        or bundle_receipt.get("physics_backend_profile_digest")
        != profile["profile_digest"]
        or bundle_receipt.get("controls_requested") is not True
        or bundle_receipt.get("policy_candidate_id") not in {None, ""}
        or bundle_receipt.get("candidate_policy_queried") is not False
        or bundle_receipt.get("candidate_outcomes_accessed") is not False
        or bundle_receipt.get("retry_cap") != 0
    ):
        raise PhysicsBackendContractError("adp009d_newton_terminal_bundle_invalid")
    native_status = native_result.get("status")
    native_blockers = sorted(
        {str(item) for item in native_result.get("blockers") or [] if str(item)}
    )
    if (
        native_result.get("schema_version") != "adp009d_native_microcheck.v1"
        or native_status not in {"completed", "blocked"}
        or (native_status == "blocked") is not bool(native_blockers)
        or native_result.get("candidate_policy_queried") is not False
        or native_result.get("candidate_outcomes_accessed") is not False
        or allocator_result.get("status") != native_status
        or sorted(allocator_result.get("blockers") or []) != native_blockers
        or allocator_result.get("retry_cap") != 0
        or allocator_result.get("continuing_spend_from_this_run") is not False
        or allocator_result.get("all_staged_objects_absent") is not True
    ):
        raise PhysicsBackendContractError("adp009d_newton_terminal_runtime_invalid")
    control_pair = native_result.get("control_episode")
    if native_status == "completed":
        probe = native_result.get("physics_backend_probe")
        if (
            native_result.get("controls_requested") is not True
            or not isinstance(probe, Mapping)
            or validate_backend_probe(probe, profile=profile)
            or not isinstance(control_pair, Mapping)
            or control_pair.get("schema_version") != "adp009d_control_pair.v1"
            or control_pair.get("physics_backend") != "newton"
            or control_pair.get("instance_digest")
            != bundle_receipt.get("scenario_instance_digest")
            or control_pair.get("control_plan_digest")
            != bundle_receipt.get("control_plan_digest")
            or control_pair.get("control_plan_semantic_digest")
            != bundle_receipt.get("control_plan_semantic_digest")
            or control_pair.get("cell_admitted_for_policy_execution") is not True
            or control_pair.get("policy_execution_blockers") != []
            or control_pair.get("candidate_policy_queried") is not False
            or control_pair.get("pair_digest")
            != canonical_digest(control_pair, digest_field="pair_digest")
        ):
            raise PhysicsBackendContractError(
                "adp009d_newton_terminal_completed_controls_invalid"
            )
    instance_ids = teardown_manifest.get("vast_instance_ids")
    actions = teardown_manifest.get("teardown_actions_performed")
    if (
        teardown_manifest.get("status") != "completed"
        or not isinstance(instance_ids, list)
        or len(instance_ids) != 1
        or isinstance(instance_ids[0], bool)
        or not isinstance(instance_ids[0], int)
        or teardown_manifest.get("runner_gpu_teardown_completed") is not True
        or teardown_manifest.get("continuing_spend_from_this_run") is not False
        or not isinstance(actions, list)
        or len(actions) != 1
        or actions[0].get("instance_id") != instance_ids[0]
        or actions[0].get("action") != "destroy_instance"
        or actions[0].get("status") != "completed"
    ):
        raise PhysicsBackendContractError("adp009d_newton_terminal_teardown_invalid")
    required_inventory = {
        str(row.get("provider")): dict(row)
        for row in provider_inventory.get("inventory_results") or []
        if isinstance(row, Mapping) and row.get("required") is True
    }
    if (
        provider_inventory.get("schema_version") != "gpu_spend_guard.v1"
        or provider_inventory.get("live_instance_count") != 0
        or provider_inventory.get("instances") != []
        or provider_inventory.get("blockers") != []
        or set(required_inventory) != {"runpod", "vast", "digitalocean"}
        or any(
            row.get("status") != "succeeded"
            or row.get("row_count") != 0
            or row.get("blockers") != []
            for row in required_inventory.values()
        )
    ):
        raise PhysicsBackendContractError(
            "adp009d_newton_terminal_provider_zero_invalid"
        )
    instance_id = instance_ids[0]
    charge_amount = vast_charge.get("amount")
    charge_items = vast_charge.get("items")
    if (
        vast_charge.get("type") != "instance"
        or vast_charge.get("source") != f"instance-{instance_id}"
        or isinstance(charge_amount, bool)
        or not isinstance(charge_amount, (int, float))
        or not math.isfinite(float(charge_amount))
        or float(charge_amount) < 0.0
        or not isinstance(charge_items, list)
        or not charge_items
        or any(
            not isinstance(item, Mapping)
            or isinstance(item.get("amount"), bool)
            or not isinstance(item.get("amount"), (int, float))
            or not math.isfinite(float(item["amount"]))
            or float(item["amount"]) < 0.0
            for item in charge_items
        )
        or not math.isclose(
            sum(float(item["amount"]) for item in charge_items),
            float(charge_amount),
            rel_tol=0.0,
            abs_tol=0.001,
        )
    ):
        raise PhysicsBackendContractError("adp009d_newton_terminal_charge_invalid")
    if (
        artifact_manifest.get("status") != "completed"
        or artifact_manifest.get("blockers") != []
        or not isinstance(artifact_manifest.get("file_count"), int)
        or int(artifact_manifest.get("file_count", 0)) <= 0
        or not isinstance(artifact_manifest.get("manifest_digest"), str)
        or not str(artifact_manifest.get("manifest_digest")).startswith("sha256:")
        or set(artifact_manifest.get("required_roles") or [])
        - set(artifact_manifest.get("observed_roles") or [])
    ):
        raise PhysicsBackendContractError("adp009d_newton_terminal_artifacts_invalid")

    receipt: dict[str, Any] = {
        "schema_version": CANARY_TERMINAL_SCHEMA_VERSION,
        "status": native_status,
        "evidence_type": "physics_backend_comparison_evidence_only",
        "physics_backend": "newton",
        "backend_profile_digest": profile["profile_digest"],
        "implementation_commit": bundle_receipt.get("implementation_commit"),
        "bundle_sha256": bundle_receipt.get("bundle_sha256"),
        "input_digest": bundle_receipt.get("input_digest"),
        "scenario_instance_digest": bundle_receipt.get("scenario_instance_digest"),
        "control_plan_digest": bundle_receipt.get("control_plan_digest"),
        "semantic_control_plan_digest": bundle_receipt.get(
            "control_plan_semantic_digest"
        ),
        "admission_digest": admission.get("admission_digest"),
        "evidence_input_digests": {
            "admission": canonical_digest(admission),
            "bundle_receipt": canonical_digest(bundle_receipt),
            "allocator_result": canonical_digest(allocator_result),
            "native_result": canonical_digest(native_result),
            "artifact_manifest": canonical_digest(artifact_manifest),
            "teardown_manifest": canonical_digest(teardown_manifest),
            "provider_inventory": canonical_digest(provider_inventory),
            "vast_charge": canonical_digest(vast_charge),
        },
        "provider_instance_id": instance_id,
        "scientific_phase": (
            "controls_completed"
            if native_status == "completed"
            else "pre_controls_blocked"
        ),
        "scientific_blockers": native_blockers,
        "controls_evidence_observed": native_status == "completed",
        "control_pair_digest": (
            control_pair.get("pair_digest")
            if isinstance(control_pair, Mapping)
            else None
        ),
        "media_gap": None
        if native_status == "completed"
        else {
            "status": "typed_gap",
            "reason": native_blockers[0],
            "lossless_policy_input_equivalent_frames_observed": False,
            "review_media_observed": False,
        },
        "policy_query_count": 0,
        "candidate_outcomes_accessed": False,
        "policy_verdict": None,
        "task_success_verdict": None,
        "physical_verdict": None,
        "artifact_manifest": {
            "digest": artifact_manifest["manifest_digest"],
            "file_count": artifact_manifest["file_count"],
            "total_size_bytes": artifact_manifest.get("total_size_bytes"),
        },
        "spend": {
            "currency": "USD",
            "actual_provider_charge_usd": float(charge_amount),
            "provider_charge_source": vast_charge.get("source"),
            "provider_charge_items": [
                {
                    "type": item.get("type"),
                    "description": item.get("description"),
                    "amount_usd": float(item["amount"]),
                }
                for item in charge_items
            ],
            "hard_cap_usd": float(admission["max_spend_usd"]),
            "under_hard_cap": float(charge_amount)
            <= float(admission["max_spend_usd"]),
        },
        "teardown": {
            "completed": True,
            "continuing_spend": False,
            "destroyed_instance_ids": instance_ids,
        },
        "provider_zero": {
            "api_confirmed": True,
            "generated_at": provider_inventory.get("generated_at"),
            "live_instance_count": 0,
            "required_providers": sorted(required_inventory),
        },
        "retry_count": 0,
        "engine_promotion_performed": False,
        "claim_ceiling": "controls_comparison_evidence_only",
    }
    receipt["terminal_receipt_digest"] = canonical_digest(
        receipt, digest_field="terminal_receipt_digest"
    )
    return receipt


def _validate_control_run(
    value: Mapping[str, Any], *, profile: Mapping[str, Any]
) -> list[str]:
    blockers: list[str] = []
    if (
        value.get("schema_version") != CONTROL_RUN_SCHEMA_VERSION
        or value.get("status") not in {"completed", "blocked"}
        or value.get("physics_backend") != profile.get("physics_backend")
        or value.get("backend_profile_digest") != profile.get("profile_digest")
        or value.get("run_digest") != canonical_digest(value, digest_field="run_digest")
    ):
        blockers.append("adp009d_backend_control_run_identity_invalid")
    if (
        value.get("backend_selected_at_simulation_construction") is not True
        or value.get("backend_switch_attempted") is not False
        or value.get("policy_query_count") != 0
        or value.get("candidate_outcomes_accessed") is not False
        or value.get("task_success_claimed") is not False
        or value.get("physical_claimed") is not False
    ):
        blockers.append("adp009d_backend_control_run_boundary_invalid")
    contact_configuration = value.get("backend_contact_configuration")
    if not isinstance(contact_configuration, Mapping):
        blockers.append("adp009d_backend_control_run_contact_configuration_missing")
    else:
        blockers.extend(validate_backend_contact_configuration(contact_configuration))
        if (
            contact_configuration.get("physics_backend")
            != profile.get("physics_backend")
            or contact_configuration.get("backend_profile_digest")
            != profile.get("profile_digest")
        ):
            blockers.append("adp009d_backend_control_run_contact_configuration_invalid")
    if not isinstance(value.get("backend_control_plan_digest"), str) or not str(
        value.get("backend_control_plan_digest")
    ).startswith("sha256:"):
        blockers.append("adp009d_backend_control_run_plan_digest_invalid")
    bindings = value.get("comparability_bindings")
    binding_row = dict(bindings) if isinstance(bindings, Mapping) else {}
    if set(binding_row) != set(COMPARABILITY_BINDINGS) or any(
        binding_row.get(name) in {None, ""} for name in COMPARABILITY_BINDINGS
    ):
        blockers.append("adp009d_backend_control_run_binding_invalid")
    measurements = value.get("measurements")
    measurement_row = dict(measurements) if isinstance(measurements, Mapping) else {}
    if set(measurement_row) != set(MEASUREMENT_FIELDS) or any(
        not isinstance(measurement_row.get(name), Mapping) for name in MEASUREMENT_FIELDS
    ):
        blockers.append("adp009d_backend_control_run_measurements_incomplete")
    teardown = dict(measurement_row.get("teardown") or {})
    provider_zero = dict(measurement_row.get("provider_zero") or {})
    spend = dict(measurement_row.get("spend") or {})
    if (
        teardown.get("completed") is not True
        or teardown.get("continuing_spend") is not False
        or provider_zero.get("api_confirmed") is not True
        or provider_zero.get("live_instance_count") != 0
        or not isinstance(spend.get("total_usd"), (int, float))
        or float(spend.get("total_usd", -1.0)) < 0.0
    ):
        blockers.append("adp009d_backend_control_run_terminal_evidence_invalid")
    initialization = dict(measurement_row.get("initialization_reset") or {})
    poses = dict(measurement_row.get("target_robot_pose") or {})
    contacts = dict(measurement_row.get("contacts_and_force_vectors") or {})
    torque = dict(measurement_row.get("torque_utilization_and_clipping") or {})
    clearance = dict(measurement_row.get("closest_geometric_clearance") or {})
    action = dict(measurement_row.get("action_delivery") or {})
    phases = dict(measurement_row.get("phase_completion") or {})
    frames = dict(measurement_row.get("lossless_frames") or {})
    review = dict(measurement_row.get("review_media") or {})
    if initialization.get("initialization_completed") is not True or initialization.get(
        "reset_completed"
    ) is not True:
        blockers.append("adp009d_backend_control_run_initialization_invalid")
    if not _finite_vector(poses.get("target_pose_world"), minimum_length=7) or not _finite_vector(
        poses.get("robot_pose_world"), minimum_length=7
    ):
        blockers.append("adp009d_backend_control_run_pose_invalid")
    if (
        not isinstance(contacts.get("force_vectors_world_n"), list)
        or not contacts.get("force_vectors_world_n")
        or not isinstance(contacts.get("partner_prim_paths"), list)
        or not contacts.get("partner_prim_paths")
        or any(
            not _finite_vector(vector)
            for vector in contacts.get("force_vectors_world_n", [])
        )
    ):
        blockers.append("adp009d_backend_control_run_contacts_invalid")
    if (
        not isinstance(torque.get("maximum_utilization"), (int, float))
        or not math.isfinite(float(torque.get("maximum_utilization", math.nan)))
        or torque.get("clipping_observed") not in {True, False}
    ):
        blockers.append("adp009d_backend_control_run_torque_invalid")
    if not isinstance(clearance.get("minimum_m"), (int, float)) or not math.isfinite(
        float(clearance.get("minimum_m", math.nan))
    ):
        blockers.append("adp009d_backend_control_run_clearance_invalid")
    if (
        not isinstance(action.get("requested_count"), int)
        or not isinstance(action.get("delivered_count"), int)
        or action.get("requested_count") != action.get("delivered_count")
        or action.get("nontrivial_action_delivered") is not True
    ):
        blockers.append("adp009d_backend_control_run_action_delivery_invalid")
    if not isinstance(phases.get("rows"), list) or not phases.get("rows"):
        blockers.append("adp009d_backend_control_run_phase_completion_invalid")
    if (
        not isinstance(frames.get("frame_manifest_digest"), str)
        or not isinstance(frames.get("frame_count"), int)
        or frames.get("frame_count", 0) <= 0
        or frames.get("lossless") is not True
    ):
        blockers.append("adp009d_backend_control_run_lossless_frames_invalid")
    if (
        not isinstance(review.get("media_digest"), str)
        or review.get("derived_from_lossless_frames") is not True
    ):
        blockers.append("adp009d_backend_control_run_review_media_invalid")
    return sorted(set(blockers))


def build_backend_control_run_receipt(
    *,
    physics_backend: str,
    comparability_bindings: Mapping[str, Any],
    backend_control_plan_digest: str,
    measurements: Mapping[str, Mapping[str, Any]],
    blockers: Sequence[str] = (),
) -> dict[str, Any]:
    """Seal one controls-only terminal run without caller-authored verdicts."""

    backend = normalize_physics_backend(physics_backend)
    profile = build_backend_profile(backend)
    typed_blockers = sorted({str(item) for item in blockers if str(item)})
    receipt: dict[str, Any] = {
        "schema_version": CONTROL_RUN_SCHEMA_VERSION,
        "status": "blocked" if typed_blockers else "completed",
        "physics_backend": backend,
        "backend_profile_digest": profile["profile_digest"],
        "backend_selected_at_simulation_construction": True,
        "backend_switch_attempted": False,
        "backend_contact_configuration": build_backend_contact_configuration(
            backend
        ),
        "backend_control_plan_digest": backend_control_plan_digest,
        "comparability_bindings": dict(comparability_bindings),
        "measurements": {
            str(name): dict(row) for name, row in measurements.items()
        },
        "policy_query_count": 0,
        "candidate_outcomes_accessed": False,
        "task_success_claimed": False,
        "physical_claimed": False,
        "blockers": typed_blockers,
        "run_digest": "",
    }
    receipt["run_digest"] = canonical_digest(receipt, digest_field="run_digest")
    return receipt


def build_comparison_receipt(
    *,
    physx_run: Mapping[str, Any],
    newton_run: Mapping[str, Any],
    fidelity_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile a deterministic two-backend receipt without promoting an engine."""

    profiles = {
        "physx": build_backend_profile("physx"),
        "newton": build_backend_profile("newton"),
    }
    runs = {"physx": dict(physx_run), "newton": dict(newton_run)}
    blockers: list[str] = []
    for backend, run in runs.items():
        blockers.extend(_validate_control_run(run, profile=profiles[backend]))
    physx_bindings = dict(runs["physx"].get("comparability_bindings") or {})
    newton_bindings = dict(runs["newton"].get("comparability_bindings") or {})
    if physx_bindings != newton_bindings:
        blockers.append("adp009d_backend_comparison_bindings_differ")
    fidelity = dict(fidelity_result)
    required_fidelity = {
        "metric_id",
        "metric_authority",
        "direction",
        "physx_value",
        "newton_value",
        "delta",
        "meaningful_threshold",
        "meaningful_improvement_observed",
    }
    numeric_fidelity = (
        fidelity.get("physx_value"),
        fidelity.get("newton_value"),
        fidelity.get("delta"),
        fidelity.get("meaningful_threshold"),
    )
    numeric_valid = all(
        not isinstance(item, bool)
        and isinstance(item, (int, float))
        and math.isfinite(float(item))
        for item in numeric_fidelity
    )
    delta_consistent = numeric_valid and math.isclose(
        float(fidelity["delta"]),
        float(fidelity["newton_value"]) - float(fidelity["physx_value"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    direction = fidelity.get("direction")
    threshold = float(fidelity.get("meaningful_threshold", 0.0))
    delta = float(fidelity.get("delta", 0.0))
    expected_meaningful = (
        delta >= threshold
        if direction == "higher_is_better" and threshold > 0.0
        else delta <= threshold
        if direction == "lower_is_better" and threshold < 0.0
        else None
    )
    if (
        set(fidelity) != required_fidelity
        or direction not in {"higher_is_better", "lower_is_better"}
        or fidelity.get("metric_authority")
        not in {"deterministic_geometry", "deterministic_simulator_state"}
        or not numeric_valid
        or not delta_consistent
        or expected_meaningful is None
        or fidelity.get("meaningful_improvement_observed")
        is not expected_meaningful
    ):
        blockers.append("adp009d_backend_comparison_fidelity_result_invalid")
    blockers = sorted(set(blockers))
    evidence_parity = not blockers and all(run.get("status") == "completed" for run in runs.values())
    receipt: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "status": "completed" if evidence_parity else "blocked",
        "program_id": "arm-decision-proof-v1",
        "comparison_id": "sealed-840313-franka-robotiq-controls-physx-vs-newton",
        "physics_backends": list(ALLOWED_PHYSICS_BACKENDS),
        "comparability_bindings": physx_bindings,
        "backend_profiles": {
            backend: profile["profile_digest"] for backend, profile in profiles.items()
        },
        "backend_runs": {
            backend: {
                "run_digest": run.get("run_digest"),
                "status": run.get("status"),
                "backend_control_plan_digest": run.get(
                    "backend_control_plan_digest"
                ),
                "contact_configuration": run.get(
                    "backend_contact_configuration"
                ),
                "measurements": run.get("measurements"),
                "typed_blockers": run.get("blockers") or [],
            }
            for backend, run in runs.items()
        },
        "evidence_parity_observed": evidence_parity,
        "fidelity_result": fidelity,
        "meaningful_improvement_observed": (
            fidelity.get("meaningful_improvement_observed") if evidence_parity else False
        ),
        "promotion_review_eligible": bool(
            evidence_parity and fidelity.get("meaningful_improvement_observed") is True
        ),
        "engine_promotion_performed": False,
        "default_backend_after_comparison": DEFAULT_PHYSICS_BACKEND,
        "policy_verdict": None,
        "task_success_verdict": None,
        "physical_verdict": None,
        "claim_ceiling": "controls_comparison_evidence_only",
        "blockers": blockers,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def validate_comparison_receipt(value: Mapping[str, Any]) -> list[str]:
    """Validate a comparison receipt by deterministic reconstruction."""

    try:
        bindings = dict(value["comparability_bindings"])
        rows = dict(value["backend_runs"])
        if set(rows) != set(ALLOWED_PHYSICS_BACKENDS):
            raise ValueError
        reconstructed: dict[str, dict[str, Any]] = {}
        for backend in ALLOWED_PHYSICS_BACKENDS:
            row = dict(rows[backend])
            run = build_backend_control_run_receipt(
                physics_backend=backend,
                comparability_bindings=bindings,
                backend_control_plan_digest=str(
                    row["backend_control_plan_digest"]
                ),
                measurements=dict(row["measurements"]),
                blockers=list(row.get("typed_blockers") or []),
            )
            if (
                row.get("run_digest") != run["run_digest"]
                or row.get("status") != run["status"]
                or row.get("contact_configuration")
                != run["backend_contact_configuration"]
            ):
                raise ValueError
            reconstructed[backend] = run
        expected = build_comparison_receipt(
            physx_run=reconstructed["physx"],
            newton_run=reconstructed["newton"],
            fidelity_result=dict(value["fidelity_result"]),
        )
    except (KeyError, TypeError, ValueError):
        return ["adp009d_backend_comparison_receipt_invalid"]
    return (
        []
        if dict(value) == expected
        else ["adp009d_backend_comparison_receipt_invalid"]
    )


def build_comparison_design_contract() -> dict[str, Any]:
    """Return the immutable provider-free design for the two controls runs."""

    profiles = {
        backend: build_backend_profile(backend) for backend in ALLOWED_PHYSICS_BACKENDS
    }
    contract: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "contract_kind": "provider_free_design",
        "status": "validated_without_provider_launch",
        "program_id": "arm-decision-proof-v1",
        "comparison_id": "sealed-840313-franka-robotiq-controls-physx-vs-newton",
        "default_physics_backend": DEFAULT_PHYSICS_BACKEND,
        "one_backend_per_run": True,
        "mid_run_backend_switch_allowed": False,
        "required_backends": list(ALLOWED_PHYSICS_BACKENDS),
        "backend_profiles": profiles,
        "backend_contact_configurations": {
            backend: build_backend_contact_configuration(backend)
            for backend in ALLOWED_PHYSICS_BACKENDS
        },
        "common_comparability_bindings": list(COMPARABILITY_BINDINGS),
        "backend_specific_contact_configuration_required": True,
        "required_measurements_per_backend": list(MEASUREMENT_FIELDS),
        "required_controls": [
            "zero_action_negative",
            "deterministic_scripted_positive",
        ],
        "newton_canary_admission": {
            "schema_version": CANARY_ADMISSION_SCHEMA_VERSION,
            "explicit_paid_run_authorization_required": True,
            "canonical_spend_admission_required": True,
            "watchdog_required": True,
            "artifact_storage_required": True,
            "teardown_required": True,
            "provider_zero_or_exact_allowed_concurrency_precheck_required": True,
            "unapproved_live_instance_count_required": 0,
            "concurrent_instance_ids_must_be_explicitly_authorized": True,
            "retry_cap": 0,
        },
        "newton_canary_terminal_receipt": {
            "schema_version": CANARY_TERMINAL_SCHEMA_VERSION,
            "actual_provider_charge_required": True,
            "artifact_manifest_required": True,
            "all_evidence_inputs_digest_bound_required": True,
            "teardown_required": True,
            "api_confirmed_provider_zero_required": True,
            "pre_controls_failure_requires_typed_media_gap": True,
            "claim_ceiling": "controls_comparison_evidence_only",
            "policy_verdict_allowed": False,
            "engine_promotion_allowed": False,
        },
        "comparison_acceptance": {
            "evidence_parity_required": True,
            "independently_meaningful_fidelity_result_required": True,
            "automatic_engine_promotion_allowed": False,
            "policy_verdict_allowed": False,
            "task_success_verdict_allowed": False,
            "physical_verdict_allowed": False,
        },
        "provider_mutation_performed": False,
        "claim_ceiling": "controls_comparison_design_only",
    }
    contract["design_digest"] = canonical_digest(contract, digest_field="design_digest")
    return contract


def validate_comparison_design_contract(value: Mapping[str, Any]) -> list[str]:
    """Validate the committed design byte-for-byte against code authority."""

    blockers: list[str] = []
    if dict(value) != build_comparison_design_contract():
        blockers.append("adp009d_backend_comparison_design_drifted")
    if value.get("design_digest") != canonical_digest(value, digest_field="design_digest"):
        blockers.append("adp009d_backend_comparison_design_digest_invalid")
    return sorted(set(blockers))


__all__ = [
    "ALLOWED_PHYSICS_BACKENDS",
    "BACKEND_PROFILE_SCHEMA_VERSION",
    "CANARY_ADMISSION_SCHEMA_VERSION",
    "CANARY_TERMINAL_SCHEMA_VERSION",
    "COMPARISON_SCHEMA_VERSION",
    "CONTROL_RUN_SCHEMA_VERSION",
    "DEFAULT_PHYSICS_BACKEND",
    "MEASUREMENT_FIELDS",
    "PHYSX_ONLY_FIELD_NAMES",
    "PROBE_SCHEMA_VERSION",
    "PhysicsBackendContractError",
    "build_backend_contact_configuration",
    "build_backend_control_run_receipt",
    "build_backend_profile",
    "build_comparison_design_contract",
    "build_comparison_receipt",
    "normalize_physics_backend",
    "build_newton_canary_admission",
    "build_newton_canary_terminal_receipt",
    "validate_backend_contact_configuration",
    "validate_backend_probe",
    "validate_backend_profile",
    "validate_comparison_receipt",
    "validate_comparison_design_contract",
    "validate_newton_canary_admission",
]
