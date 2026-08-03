"""Bridge qualified geometry and registration into routing site evidence.

The reconstruction lane already owns observed-surface compilation and
independent collider qualification. This module does not repeat either job.
It adds a checked robot-base-to-site-frame registration record and converts
the exact, lineage-consistent artifacts into ``site_evidence_profile.v1``.

No method qualification, R5 evidence, R6 decision, R7 admission, physical
success, deployment readiness, or safety authority is created here.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .reconstruction_geometry_contracts import (
    ReconstructionGeometryContractError,
    build_collider_candidate_manifest,
    build_collider_qualification_report,
    build_metric_geometry_manifest,
)
from .task_site_measurement_routing import (
    MeasurementRoutingError,
    validate_site_evidence_profile,
)


REGISTRATION_SCHEMA_VERSION = "measurement_robot_site_registration.v1"
REGISTRATION_DECISIONS = frozenset({"accepted", "rejected", "development_only"})


class MeasurementSiteEvidenceBridgeError(ValueError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise MeasurementSiteEvidenceBridgeError("site_evidence_bridge_not_json") from exc
    return result


def _digest(value: Mapping[str, Any], field: str) -> str:
    return canonical_digest(dict(value), digest_field=field)


def _valid_digest(value: Any) -> bool:
    raw = str(value).strip() if value is not None else ""
    return (
        len(raw) == 71
        and raw.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in raw[7:])
    )


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: Any, *, minimum: float | None = None) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        return None
    return result


def _validate_se3(value: Any) -> list[list[float]] | None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        return None
    matrix: list[list[float]] = []
    for row in value:
        normalized = [_number(item) for item in row]
        if any(item is None for item in normalized):
            return None
        matrix.append([float(item) for item in normalized if item is not None])
    if any(abs(matrix[3][index] - expected) > 1e-9 for index, expected in enumerate([0, 0, 0, 1])):
        return None
    rotation = [row[:3] for row in matrix[:3]]
    for left in range(3):
        for right in range(3):
            dot = sum(rotation[row][left] * rotation[row][right] for row in range(3))
            expected = 1.0 if left == right else 0.0
            if abs(dot - expected) > 1e-8:
                return None
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    return matrix if abs(determinant - 1.0) <= 1e-8 else None


def build_robot_site_registration_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an independently measured robot-base-to-site transform."""

    record = _clone(value)
    supplied_digest = record.pop("registration_digest", None)
    errors: list[str] = []
    if record.get("schema_version") != REGISTRATION_SCHEMA_VERSION:
        errors.append("robot_site_registration_schema_invalid")
    for key in (
        "registration_id",
        "site_frame_id",
        "robot_base_frame_id",
        "measurement_method",
    ):
        if not _string(record.get(key)):
            errors.append(f"robot_site_registration_{key}_missing")
    for key in (
        "source_capture_digest",
        "metric_geometry_manifest_digest",
        "collider_qualification_digest",
        "robot_model_digest",
        "tool_geometry_digest",
        "independent_measurement_digest",
    ):
        if not _valid_digest(record.get(key)):
            errors.append(f"robot_site_registration_{key}_invalid")
    transform = _validate_se3(record.get("site_from_robot_base_transform"))
    if transform is None:
        errors.append("robot_site_registration_transform_invalid")
    else:
        record["site_from_robot_base_transform"] = transform
    evaluator = record.get("independent_evaluator")
    if (
        not isinstance(evaluator, Mapping)
        or not _string(evaluator.get("evaluator_id"))
        or evaluator.get("candidate_method_independent") is not True
        or evaluator.get("agent_is_evaluator") is not False
    ):
        errors.append("robot_site_registration_evaluator_invalid")
    if record.get("measurement_signature_status") not in {"verified", "unverified"}:
        errors.append("robot_site_registration_signature_status_invalid")
    signature_id = _string(record.get("evaluator_approval_signature_id"))
    for key in (
        "candidate_self_graded",
        "agent_generated_measurements",
        "thresholds_modified_after_measurement",
    ):
        if record.get(key) is not False:
            errors.append(f"robot_site_registration_{key}_must_be_false")
    sample_count = record.get("measurement_sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 3:
        errors.append("robot_site_registration_sample_count_invalid")
    translation_error = _number(record.get("translation_rms_error_m"), minimum=0.0)
    rotation_error = _number(record.get("rotation_rms_error_deg"), minimum=0.0)
    translation_limit = _number(record.get("maximum_translation_rms_error_m"), minimum=0.0)
    rotation_limit = _number(record.get("maximum_rotation_rms_error_deg"), minimum=0.0)
    if None in {translation_error, rotation_error, translation_limit, rotation_limit}:
        errors.append("robot_site_registration_error_or_threshold_invalid")
    regions = record.get("task_region_ids")
    if (
        not isinstance(regions, list)
        or not regions
        or any(not _string(item) for item in regions)
        or len({_string(item) for item in regions}) != len(regions)
    ):
        errors.append("robot_site_registration_task_regions_invalid")
    blockers = record.get("blockers")
    if not isinstance(blockers, list) or any(not _string(item) for item in blockers):
        errors.append("robot_site_registration_blockers_invalid")
        blockers = []
    for key in (
        "metric_scale_verified",
        "physical_measurements_included",
        "development_only",
    ):
        if record.get(key) not in {True, False}:
            errors.append(f"robot_site_registration_{key}_invalid")
    if record.get("collider_qualification_decision") not in {
        "accepted_bounded_navigation",
        "rejected",
    }:
        errors.append("robot_site_registration_collider_decision_invalid")
    if record.get("agent_may_approve") is not False:
        errors.append("robot_site_registration_agent_approval_forbidden")
    for key in (
        "r5_evidence",
        "r6_decision",
        "r7_admission",
        "physical_success_established",
        "deployment_readiness_established",
        "safety_established",
    ):
        if record.get(key) is not False:
            errors.append(f"robot_site_registration_{key}_must_be_false")
    development = record.get("development_only") is True
    accepted = (
        not errors
        and not development
        and record.get("physical_measurements_included") is True
        and record.get("metric_scale_verified") is True
        and record.get("collider_qualification_decision") == "accepted_bounded_navigation"
        and record.get("measurement_signature_status") == "verified"
        and bool(signature_id)
        and not blockers
        and translation_error is not None
        and translation_limit is not None
        and translation_error <= translation_limit
        and rotation_error is not None
        and rotation_limit is not None
        and rotation_error <= rotation_limit
    )
    expected_decision = (
        "development_only" if development else "accepted" if accepted else "rejected"
    )
    if record.get("decision") != expected_decision:
        errors.append("robot_site_registration_decision_not_deterministic")
    if record.get("proof_effect") != "robot_site_registration_evidence":
        errors.append("robot_site_registration_proof_effect_invalid")
    if record.get("claim_ceiling") != "C2":
        errors.append("robot_site_registration_claim_ceiling_invalid")
    expected_digest = _digest(record, "registration_digest")
    if supplied_digest is not None and supplied_digest != expected_digest:
        errors.append("robot_site_registration_digest_mismatch")
    if errors:
        raise MeasurementSiteEvidenceBridgeError(*errors)
    record["registration_digest"] = expected_digest
    return record


def _evidence_record(
    *, available: bool, validated: bool, record_id: str | None, scope: str
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "available": available,
        "validated": validated,
        "scope": scope,
    }
    if available and record_id:
        value["record_id"] = record_id
    return value


def build_site_evidence_profile_from_geometry(
    *,
    metric_geometry_manifest: Mapping[str, Any],
    collider_candidate_manifest: Mapping[str, Any],
    collider_qualification_report: Mapping[str, Any],
    robot_site_registration: Mapping[str, Any],
    profile_id: str,
    bundle_id: str,
    bundle_hash: str,
    provenance_record_id: str,
    rights: Mapping[str, Any],
    privacy: Mapping[str, Any],
    additional_evidence: Mapping[str, Any] | None = None,
    additional_limitations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create router evidence only when all geometry lineage joins exactly."""

    try:
        metric = build_metric_geometry_manifest(metric_geometry_manifest)
        candidate = build_collider_candidate_manifest(collider_candidate_manifest)
        qualification = build_collider_qualification_report(collider_qualification_report)
    except ReconstructionGeometryContractError as exc:
        raise MeasurementSiteEvidenceBridgeError(
            *(f"geometry_artifact_invalid:{code}" for code in exc.codes)
        ) from exc
    registration = build_robot_site_registration_record(robot_site_registration)
    errors: list[str] = []
    if candidate.get("metric_geometry_manifest_digest") != metric.get(
        "metric_geometry_manifest_digest"
    ):
        errors.append("site_evidence_bridge_metric_candidate_mismatch")
    if qualification.get("collider_candidate_manifest_digest") != candidate.get(
        "collider_candidate_manifest_digest"
    ):
        errors.append("site_evidence_bridge_candidate_qualification_mismatch")
    if qualification.get("collider_asset_digest") != candidate.get("collider_asset_digest"):
        errors.append("site_evidence_bridge_collider_asset_mismatch")
    qualification_evaluator = qualification.get("independent_evaluator")
    if (
        not isinstance(qualification_evaluator, Mapping)
        or qualification_evaluator.get("candidate_method_independent") is not True
        or qualification.get("candidate_self_graded") is not False
        or not _valid_digest(qualification.get("measurement_artifact_digest"))
    ):
        errors.append("site_evidence_bridge_independent_collider_evidence_incomplete")
    qualification_regions = sorted(
        _string(item) for item in qualification.get("task_region_ids") or []
    )
    registration_regions = sorted(
        _string(item) for item in registration.get("task_region_ids") or []
    )
    if not qualification_regions or qualification_regions != registration_regions:
        errors.append("site_evidence_bridge_task_region_mismatch")
    if registration.get("metric_geometry_manifest_digest") != metric.get(
        "metric_geometry_manifest_digest"
    ):
        errors.append("site_evidence_bridge_metric_registration_mismatch")
    if registration.get("collider_qualification_digest") != qualification.get(
        "collider_qualification_digest"
    ):
        errors.append("site_evidence_bridge_qualification_registration_mismatch")
    capture_digests = {
        metric.get("source_capture_digest"),
        candidate.get("source_capture_digest"),
        qualification.get("source_capture_digest"),
        registration.get("source_capture_digest"),
    }
    if len(capture_digests) != 1:
        errors.append("site_evidence_bridge_source_capture_mismatch")
    if not _string(profile_id) or not _string(bundle_id) or not _valid_digest(bundle_hash):
        errors.append("site_evidence_bridge_profile_identity_invalid")
    if not _string(provenance_record_id):
        errors.append("site_evidence_bridge_provenance_missing")
    if not isinstance(rights, Mapping) or not isinstance(privacy, Mapping):
        errors.append("site_evidence_bridge_governance_invalid")
    protected = {
        "metric_scale",
        "validated_mesh",
        "validated_collider",
        "robot_site_registration",
    }
    extra = dict(additional_evidence or {})
    if protected & set(extra):
        errors.append("site_evidence_bridge_protected_evidence_override")
    if errors:
        raise MeasurementSiteEvidenceBridgeError(*errors)
    scale_valid = metric.get("metric_scale_status") == "validated"
    collider_valid = (
        scale_valid
        and qualification.get("decision") == "accepted_bounded_navigation"
        and not qualification.get("blockers")
    )
    registration_valid = collider_valid and registration.get("decision") == "accepted"
    evidence = {
        **extra,
        "metric_scale": _evidence_record(
            available=True,
            validated=scale_valid,
            record_id=metric["metric_geometry_manifest_digest"],
            scope="observed_metric_surface",
        ),
        "validated_mesh": _evidence_record(
            available=True,
            validated=collider_valid,
            record_id=candidate["collider_candidate_manifest_digest"],
            scope="observed_surface_no_fill",
        ),
        "validated_collider": _evidence_record(
            available=True,
            validated=collider_valid,
            record_id=qualification["collider_qualification_digest"],
            scope="bounded_navigation_task_regions",
        ),
        "robot_site_registration": _evidence_record(
            available=True,
            validated=registration_valid,
            record_id=registration["registration_digest"],
            scope="registered_robot_base_to_site_frame",
        ),
    }
    limitations = dict(additional_limitations or {})
    limitations["known_missing_regions"] = sorted(
        {
            *[str(item) for item in metric.get("unsupported_region_ids") or []],
            *[str(item) for item in limitations.get("known_missing_regions") or []],
        }
    )
    limitations["forbidden_claims"] = sorted(
        {
            *[str(item) for item in limitations.get("forbidden_claims") or []],
            "physical_task_success",
            "deployment_readiness",
            "safety_certification",
        }
    )
    profile = {
        "schema_version": "site_evidence_profile.v1",
        "profile_id": profile_id,
        "bundle_id": bundle_id,
        "bundle_hash": bundle_hash,
        "provenance_record_id": provenance_record_id,
        "rights": dict(rights),
        "privacy": dict(privacy),
        "coordinate_system": {
            "units": "meters",
            "up_axis": "Z",
            "site_frame_id": registration["site_frame_id"],
            "robot_base_frame_id": registration["robot_base_frame_id"],
            "metric_scale_verified": scale_valid,
            "site_from_robot_base_transform": registration["site_from_robot_base_transform"],
            "registration_digest": registration["registration_digest"],
        },
        "evidence": evidence,
        "limitations": limitations,
        "source_capture_digest": metric["source_capture_digest"],
        "geometry_bridge": {
            "metric_geometry_manifest_digest": metric["metric_geometry_manifest_digest"],
            "collider_candidate_manifest_digest": candidate["collider_candidate_manifest_digest"],
            "collider_qualification_digest": qualification["collider_qualification_digest"],
            "robot_site_registration_digest": registration["registration_digest"],
            "development_only": registration["development_only"],
            "method_qualification_created": False,
            "r5_evidence_created": False,
            "r6_decision_created": False,
            "r7_admission_created": False,
            "physical_success_established": False,
            "deployment_readiness_established": False,
            "safety_established": False,
            "agent_may_promote": False,
        },
    }
    try:
        return validate_site_evidence_profile(profile)
    except MeasurementRoutingError as exc:
        raise MeasurementSiteEvidenceBridgeError(f"site_evidence_profile_invalid:{exc}") from exc


__all__ = [
    "MeasurementSiteEvidenceBridgeError",
    "REGISTRATION_DECISIONS",
    "REGISTRATION_SCHEMA_VERSION",
    "build_robot_site_registration_record",
    "build_site_evidence_profile_from_geometry",
]
