"""Deterministic metric-geometry, collider, packaging, and Isaac contracts.

The existing geometry and ParticleField adapters remain the executors.  This
module qualifies their recorded outputs without promoting appearance into
metric or collision truth.
"""

from __future__ import annotations

import json
import math
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


METRIC_GEOMETRY_SCHEMA = "metric_geometry_manifest.v1"
COLLIDER_CANDIDATE_SCHEMA = "mesh_collider_candidate_manifest.v1"
COLLIDER_QUALIFICATION_SCHEMA = "collider_qualification_report.v1"
PACKAGING_REQUEST_SCHEMA = "nurec_openusd_packaging_request.v1"
PACKAGING_RESULT_SCHEMA = "nurec_openusd_packaging_result.v1"
ISAAC_VERIFICATION_SCHEMA = "isaac_asset_verification_result.v1"

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class ReconstructionGeometryContractError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _clone(value: Any) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ReconstructionGeometryContractError(["artifact_not_json_serializable"]) from exc
    if not isinstance(result, dict):
        raise ReconstructionGeometryContractError(["artifact_not_object"])
    return result


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _number(value: Any, *, minimum: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        return None
    return result


def _finalize(value: Mapping[str, Any], schema: str, digest_field: str) -> dict[str, Any]:
    artifact = _clone(value)
    supplied = artifact.pop(digest_field, None)
    artifact["schema_version"] = schema
    expected = canonical_digest(artifact, digest_field=digest_field)
    if supplied is not None and supplied != expected:
        raise ReconstructionGeometryContractError([f"{digest_field}_mismatch"])
    artifact[digest_field] = expected
    return artifact


def _lineage(value: Mapping[str, Any], errors: list[str]) -> None:
    for key in (
        "stable_run_identity",
        "source_capture_identity",
        "producing_method",
        "implementation_version",
        "timestamp",
    ):
        if not isinstance(value.get(key), str) or not value[key]:
            errors.append(f"{key}_missing")
    for key in (
        "source_capture_digest",
        "deterministic_configuration_digest",
        "train_heldout_split_digest",
    ):
        if not _is_digest(value.get(key)):
            errors.append(f"{key}_invalid")
    if _COMMIT.fullmatch(str(value.get("source_commit_sha") or "")) is None:
        errors.append("source_commit_sha_invalid")
    if value.get("units") != "meters":
        errors.append("units_must_be_meters")
    for key in (
        "original_file_references",
        "input_digests",
        "output_digests",
        "warnings",
        "blockers",
    ):
        if not isinstance(value.get(key), list):
            errors.append(f"{key}_invalid")
    for key in (
        "camera_calibration_binding",
        "coordinate_frame_declaration",
        "provider_runtime_identity",
        "authority_used",
        "parent_artifact_or_event",
    ):
        if not isinstance(value.get(key), Mapping):
            errors.append(f"{key}_invalid")
    if _number(value.get("cost_usd"), minimum=0) is None or _number(
        value.get("duration_seconds"), minimum=0
    ) is None:
        errors.append("cost_or_duration_invalid")


def build_metric_geometry_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(value)
    errors: list[str] = []
    _lineage(artifact, errors)
    if artifact.get("metric_scale_status") not in {
        "validated",
        "sensor_metric_unvalidated",
        "anchor_required",
    }:
        errors.append("metric_scale_status_invalid")
    if artifact.get("generated_fill_used") is not False:
        errors.append("generated_or_unseen_fill_forbidden")
    if artifact.get("appearance_asset_used_as_geometry_truth") is not False:
        errors.append("appearance_cannot_be_geometry_truth")
    if not isinstance(artifact.get("observed_region_ids"), list) or not isinstance(
        artifact.get("unsupported_region_ids"), list
    ):
        errors.append("region_ledger_invalid")
    if not isinstance(artifact.get("confidence_filter"), Mapping):
        errors.append("confidence_filter_missing")
    if not _is_digest(artifact.get("geometry_asset_digest")):
        errors.append("geometry_asset_digest_invalid")
    if artifact.get("proof_effect") != "metric_reference_candidate_only" or artifact.get(
        "claim_ceiling"
    ) != "metric_reference_geometry":
        errors.append("metric_geometry_claim_boundary_invalid")
    if errors:
        raise ReconstructionGeometryContractError(errors)
    return _finalize(artifact, METRIC_GEOMETRY_SCHEMA, "metric_geometry_manifest_digest")


def build_collider_candidate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(value)
    errors: list[str] = []
    _lineage(artifact, errors)
    if not _is_digest(artifact.get("metric_geometry_manifest_digest")) or not _is_digest(
        artifact.get("collider_asset_digest")
    ):
        errors.append("collider_input_or_output_digest_invalid")
    if artifact.get("unobserved_regions_filled") is not False:
        errors.append("unobserved_region_fill_forbidden")
    if artifact.get("collision_validated") is not False:
        errors.append("candidate_cannot_declare_collision_valid")
    if not isinstance(artifact.get("component_statistics"), Mapping) or not isinstance(
        artifact.get("hole_statistics"), Mapping
    ):
        errors.append("collider_topology_statistics_missing")
    if artifact.get("proof_effect") != "collision_candidate_only" or artifact.get(
        "claim_ceiling"
    ) != "collision_geometry_candidate":
        errors.append("collider_candidate_claim_boundary_invalid")
    if errors:
        raise ReconstructionGeometryContractError(errors)
    return _finalize(artifact, COLLIDER_CANDIDATE_SCHEMA, "collider_candidate_manifest_digest")


_UPPER_LIMITS = {
    "scale_error_fraction",
    "gravity_alignment_error_deg",
    "floor_height_residual_m",
    "wall_offset_residual_m",
    "visual_to_collider_disagreement_m",
    "clearance_error_m",
}
_LOWER_LIMITS = {"mesh_coverage_fraction", "minimum_obstacle_thickness_m"}


def build_collider_qualification_report(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(value)
    errors: list[str] = []
    _lineage(artifact, errors)
    if not _is_digest(artifact.get("collider_candidate_manifest_digest")) or not _is_digest(
        artifact.get("qa_thresholds_digest")
    ):
        errors.append("collider_qualification_binding_invalid")
    measurements = artifact.get("measurements")
    thresholds = artifact.get("thresholds")
    if not isinstance(measurements, Mapping) or not isinstance(thresholds, Mapping):
        errors.append("collider_measurements_or_thresholds_missing")
        measurements, thresholds = {}, {}
    required = _UPPER_LIMITS | _LOWER_LIMITS
    if set(measurements) < required or set(thresholds) < required:
        errors.append("collider_required_measurements_missing")
    passed = not errors and not artifact.get("blockers")
    for key in required:
        measured = _number(measurements.get(key), minimum=0)
        threshold = _number(thresholds.get(key), minimum=0)
        if measured is None or threshold is None:
            passed = False
        elif key in _UPPER_LIMITS and measured > threshold:
            passed = False
        elif key in _LOWER_LIMITS and measured < threshold:
            passed = False
    if artifact.get("metric_scale_status") != "validated":
        passed = False
    if artifact.get("robot_footprint_navigability_checked") is not True:
        passed = False
    expected = "accepted_bounded_navigation" if passed else "rejected"
    if artifact.get("decision") != expected:
        errors.append("collider_decision_not_deterministic")
    forbidden = set(artifact.get("unsupported_claims") or [])
    if not {"grasping", "articulation", "contact_force", "deployment"} <= forbidden:
        errors.append("collider_unsupported_claims_incomplete")
    if artifact.get("proof_effect") != "bounded_navigation_collision_qualification" or artifact.get(
        "claim_ceiling"
    ) != "bounded_navigation_simulation":
        errors.append("collider_qualification_claim_boundary_invalid")
    if errors:
        raise ReconstructionGeometryContractError(errors)
    return _finalize(artifact, COLLIDER_QUALIFICATION_SCHEMA, "collider_qualification_digest")


def build_nurec_openusd_packaging_request(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(value)
    errors: list[str] = []
    _lineage(artifact, errors)
    for key in (
        "metric_geometry_manifest_digest",
        "collider_candidate_manifest_digest",
        "collider_qualification_digest",
    ):
        if not _is_digest(artifact.get(key)):
            errors.append(f"{key}_invalid")
    if artifact.get("collider_qualification_decision") != "accepted_bounded_navigation":
        errors.append("qualified_collider_required_for_packaging")
    original_digests = {
        str(item.get("digest") or "")
        for item in artifact.get("original_file_references") or []
        if isinstance(item, Mapping)
    }
    input_digests = {
        str(item.get("digest") or "")
        for item in artifact.get("input_digests") or []
        if isinstance(item, Mapping)
    }
    for name in ("appearance_asset", "collider_asset"):
        binding = artifact.get(name)
        if not isinstance(binding, Mapping):
            errors.append(f"{name}_binding_missing")
            continue
        if not _is_digest(binding.get("digest")):
            errors.append(f"{name}_digest_invalid")
        elif binding.get("digest") not in original_digests or binding.get("digest") not in input_digests:
            errors.append(f"{name}_provenance_binding_missing")
        relative_path = str(binding.get("relative_path") or "").replace("\\", "/")
        relative = PurePosixPath(relative_path)
        if (
            not relative_path
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or ":" in relative.parts[0]
        ):
            errors.append(f"{name}_relative_path_unsafe")
        prim_path = str(binding.get("source_prim_path") or "")
        if not prim_path.startswith("/") or ".." in prim_path.split("/"):
            errors.append(f"{name}_source_prim_path_invalid")
    if artifact.get("stage_meters_per_unit") != 1.0 or artifact.get("up_axis") != "Z":
        errors.append("stage_units_or_up_axis_invalid")
    if artifact.get("shared_visual_physics_frame") is not True:
        errors.append("visual_physics_frame_mismatch")
    if artifact.get("output_format") != "usdz":
        errors.append("self_contained_usdz_output_required")
    if artifact.get("output_digests") != []:
        errors.append("packaging_request_cannot_predeclare_outputs")
    output_name = str(artifact.get("output_name") or "")
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.usdz", output_name)
        or "/" in output_name
        or "\\" in output_name
    ):
        errors.append("packaging_output_name_unsafe")
    targets = artifact.get("target_prim_paths")
    if not isinstance(targets, Mapping) or targets.get("appearance") != (
        "/World/BlueprintReconstruction/Appearance"
    ) or targets.get("collision") != "/World/BlueprintReconstruction/Collision":
        errors.append("packaging_target_prim_paths_invalid")
    if artifact.get("proof_effect") != "packaging_request_only" or artifact.get(
        "claim_ceiling"
    ) != "none":
        errors.append("packaging_request_claim_boundary_invalid")
    if errors:
        raise ReconstructionGeometryContractError(errors)
    return _finalize(artifact, PACKAGING_REQUEST_SCHEMA, "packaging_request_digest")


def build_nurec_openusd_packaging_result(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(value)
    errors: list[str] = []
    _lineage(artifact, errors)
    for key in (
        "appearance_asset_digest",
        "metric_geometry_manifest_digest",
        "collider_candidate_manifest_digest",
        "collider_qualification_digest",
        "packaging_request_digest",
        "package_digest",
    ):
        if not _is_digest(artifact.get(key)):
            errors.append(f"{key}_invalid")
    if artifact.get("collider_qualification_decision") != "accepted_bounded_navigation":
        errors.append("packaged_collider_not_independently_qualified")
    if artifact.get("package_format") != "usdz" or artifact.get("self_contained") is not True:
        errors.append("self_contained_usdz_package_required")
    if artifact.get("deterministic_archive") is not True:
        errors.append("deterministic_package_archive_required")
    if artifact.get("missing_asset_count") != 0:
        errors.append("packaged_dependency_missing")
    for key in ("package_member_count", "particlefield_prim_count", "collision_api_prim_count"):
        value_number = artifact.get(key)
        if isinstance(value_number, bool) or not isinstance(value_number, int) or value_number < 1:
            errors.append(f"{key}_invalid")
    package_reference = str(artifact.get("package_artifact_reference") or "").replace("\\", "/")
    package_path = PurePosixPath(package_reference)
    if (
        not package_reference
        or package_path.is_absolute()
        or any(part in {"", ".", ".."} for part in package_path.parts)
        or package_path.suffix.lower() != ".usdz"
    ):
        errors.append("package_artifact_reference_unsafe")
    if artifact.get("stage_meters_per_unit") != 1.0 or artifact.get("up_axis") != "Z":
        errors.append("stage_units_or_up_axis_invalid")
    if artifact.get("shared_visual_physics_frame") is not True:
        errors.append("visual_physics_frame_mismatch")
    if artifact.get("appearance_prim_present") is not True or artifact.get(
        "collision_prim_present"
    ) is not True:
        errors.append("required_package_prims_missing")
    if artifact.get("collision_api_configured") is not True:
        errors.append("collision_api_missing")
    if artifact.get("proof_effect") != "packaging_compatibility_candidate_only" or artifact.get(
        "claim_ceiling"
    ) != "openusd_package":
        errors.append("packaging_claim_boundary_invalid")
    if errors:
        raise ReconstructionGeometryContractError(errors)
    return _finalize(artifact, PACKAGING_RESULT_SCHEMA, "packaging_result_digest")


def build_isaac_asset_verification_result(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(value)
    errors: list[str] = []
    _lineage(artifact, errors)
    if not _is_digest(artifact.get("packaging_result_digest")):
        errors.append("packaging_result_digest_invalid")
    checks = {
        "exact_package_opened": True,
        "expected_prims_present": True,
        "stage_units_valid": True,
        "transforms_valid": True,
        "missing_assets_detected": False,
        "particlefield_loaded": True,
        "collision_geometry_active": True,
        "ground_contact_surface_present": True,
        "test_body_fell_through_floor": False,
        "fixed_camera_renders_nonblank": True,
        "nan_or_corrupt_render_detected": False,
        "obvious_scale_mismatch_detected": False,
    }
    observed = artifact.get("checks")
    if not isinstance(observed, Mapping) or any(observed.get(key) is not expected for key, expected in checks.items()):
        errors.append("isaac_required_checks_failed")
    if not isinstance(artifact.get("fixed_camera_render_references"), list) or not artifact.get(
        "fixed_camera_render_references"
    ):
        errors.append("isaac_fixed_camera_renders_missing")
    if artifact.get("status") != "verified_compatibility_only":
        errors.append("isaac_status_invalid")
    if artifact.get("simulator_task_success_proven") is not False or artifact.get(
        "physical_success_proven"
    ) is not False or artifact.get("deployment_readiness_proven") is not False:
        errors.append("isaac_forbidden_claim_promotion")
    if artifact.get("proof_effect") != "isaac_load_render_physics_presence_only" or artifact.get(
        "claim_ceiling"
    ) != "isaac_load_render_compatibility":
        errors.append("isaac_claim_boundary_invalid")
    if errors:
        raise ReconstructionGeometryContractError(errors)
    return _finalize(artifact, ISAAC_VERIFICATION_SCHEMA, "isaac_verification_result_digest")


__all__ = [name for name in globals() if name.startswith("build_") or name.endswith("_SCHEMA")]
