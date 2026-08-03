"""Deterministic qualification for one-walk Capture reconstruction evidence.

Capture requests checks and preserves observations.  It does not set thresholds
or authorize a reconstruction.  This module applies an exact task/site profile
to independently produced measurements bound to the same capture, candidate
manifest, geometry, and appearance bytes.  Missing or failed evidence always
abstains with the first profile-defined measurement or recapture action.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .post_capture_evidence_spine import (
    GEOMETRY_QUALIFICATION_SCHEMA,
    GEOMETRY_SCHEMA,
    NATIVE_3DGS_SCHEMA,
    REGISTRATION_QUALIFICATION_SCHEMA,
    SOURCE_PROFILE_SCHEMA,
    build_qualified_site_geometry,
    build_registered_site_reconstruction,
)


REQUEST_SCHEMA = "reconstruction_qualification_request.v1"
CANDIDATE_MANIFEST_SCHEMA = "downstream_candidate_manifest.v1"
PROFILE_SCHEMA = "capture_reconstruction_evidence_profile.v1"
MEASUREMENT_SCHEMA = "capture_reconstruction_measurement.v1"
DECISION_SCHEMA = "capture_reconstruction_qualification.v1"

REQUIRED_CHECKS = (
    "loop_closure",
    "tracking_quality",
    "depth_reprojection_error",
    "mesh_coverage",
    "floor_support_continuity",
    "physical_collision_probes",
    "postshot_registered_reconstruction",
)
SCALE_CHECKS = REQUIRED_CHECKS[:3]
COLLISION_CHECKS = REQUIRED_CHECKS[:6]
_GEOMETRY_CHECKS = frozenset(REQUIRED_CHECKS[3:6])
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class CaptureReconstructionQualificationError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CaptureReconstructionQualificationError(
            ["capture_reconstruction_value_not_json"]
        ) from exc


def _digest(value: Any) -> bool:
    return _DIGEST.fullmatch(str(value or "")) is not None


def _finalize(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = _clone(dict(value))
    result[field] = canonical_digest(result, digest_field=field)
    return result


def _validate_digest_artifact(
    value: Mapping[str, Any], *, schema: str, digest_field: str, code: str
) -> dict[str, Any]:
    result = _clone(dict(value))
    if result.get("schema_version") != schema:
        raise CaptureReconstructionQualificationError([f"{code}_schema_invalid"])
    if result.get(digest_field) != canonical_digest(result, digest_field=digest_field):
        raise CaptureReconstructionQualificationError([f"{code}_digest_invalid"])
    return result


def _parse_time(value: Any, *, code: str) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CaptureReconstructionQualificationError([code]) from exc
    if parsed.tzinfo is None:
        raise CaptureReconstructionQualificationError([code])
    return parsed.astimezone(timezone.utc)


def _profile_checks(profile: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = profile.get("checks")
    if not isinstance(raw, Mapping) or set(raw) != set(REQUIRED_CHECKS):
        raise CaptureReconstructionQualificationError(
            ["capture_reconstruction_profile_checks_incomplete"]
        )
    checks: dict[str, dict[str, Any]] = {}
    for check in REQUIRED_CHECKS:
        row = raw.get(check)
        if not isinstance(row, Mapping):
            raise CaptureReconstructionQualificationError(
                [f"capture_reconstruction_profile_check_invalid:{check}"]
            )
        normalized = _clone(dict(row))
        operator = normalized.get("operator")
        if operator not in {"eq", "gte", "lte", "gt", "lt", "between_inclusive"}:
            raise CaptureReconstructionQualificationError(
                [f"capture_reconstruction_profile_operator_invalid:{check}"]
            )
        if "threshold" not in normalized:
            raise CaptureReconstructionQualificationError(
                [f"capture_reconstruction_profile_threshold_missing:{check}"]
            )
        if not str(normalized.get("failure_action") or "").strip():
            raise CaptureReconstructionQualificationError(
                [f"capture_reconstruction_profile_failure_action_missing:{check}"]
            )
        checks[check] = normalized
    return checks


def validate_evidence_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    profile = _validate_digest_artifact(
        value,
        schema=PROFILE_SCHEMA,
        digest_field="evidence_profile_digest",
        code="capture_reconstruction_profile",
    )
    for key in (
        "profile_id",
        "task_id",
        "site_id",
        "source_capture_digest",
        "coordinate_frame_session_id",
    ):
        if not str(profile.get(key) or "").strip():
            raise CaptureReconstructionQualificationError(
                [f"capture_reconstruction_profile_{key}_missing"]
            )
    if not _digest(profile.get("source_capture_digest")):
        raise CaptureReconstructionQualificationError(
            ["capture_reconstruction_profile_source_capture_digest_invalid"]
        )
    _profile_checks(profile)
    calibration = profile.get("device_calibration")
    if not isinstance(calibration, Mapping) or calibration.get("required") not in {
        True,
        False,
    }:
        raise CaptureReconstructionQualificationError(
            ["capture_reconstruction_profile_device_calibration_invalid"]
        )
    for key in ("maximum_relative_error", "maximum_median_absolute_deviation_m"):
        value = calibration.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise CaptureReconstructionQualificationError(
                [f"capture_reconstruction_profile_calibration_{key}_invalid"]
            )
    minimum_samples = calibration.get("minimum_accepted_sample_count")
    if (
        not isinstance(minimum_samples, int)
        or isinstance(minimum_samples, bool)
        or minimum_samples < 1
    ):
        raise CaptureReconstructionQualificationError(
            ["capture_reconstruction_profile_calibration_minimum_accepted_sample_count_invalid"]
        )
    return profile


def _validate_request(value: Mapping[str, Any], coordinate_frame: str) -> dict[str, Any]:
    request = _clone(dict(value))
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("capture_reconstruction_request_schema_invalid")
    requested = request.get("requested_checks")
    check_names = [
        str(row.get("check"))
        for row in (requested or [])
        if isinstance(row, Mapping)
    ]
    if check_names != list(REQUIRED_CHECKS):
        errors.append("capture_reconstruction_request_checks_invalid")
    if request.get("coordinate_frame_session_id") != coordinate_frame:
        errors.append("capture_reconstruction_request_coordinate_frame_mismatch")
    if request.get("threshold_source") != "task_site_evidence_profile_digest_bound":
        errors.append("capture_reconstruction_request_threshold_authority_invalid")
    if request.get("qualification_authority") != "blueprint_pipeline":
        errors.append("capture_reconstruction_request_qualification_authority_invalid")
    if request.get("capture_decision") != "abstain_pending_downstream_measurements":
        errors.append("capture_reconstruction_request_capture_authority_invalid")
    if request.get("collision_artifact_status") != "candidate_only":
        errors.append("capture_reconstruction_request_collision_status_invalid")
    if errors:
        raise CaptureReconstructionQualificationError(errors)
    return request


def _validate_candidate_manifest(
    value: Mapping[str, Any], coordinate_frame: str, source: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = _validate_digest_artifact(
        value,
        schema=CANDIDATE_MANIFEST_SCHEMA,
        digest_field="manifest_digest",
        code="capture_reconstruction_candidate_manifest",
    )
    boundary = candidate.get("claim_boundary")
    neutrality = candidate.get("provider_neutrality")
    selection = candidate.get("selection_contract")
    allowed_use = candidate.get("allowed_use_scope")
    neutrality_map = dict(neutrality) if isinstance(neutrality, Mapping) else {}
    errors: list[str] = []
    if candidate.get("coordinate_frame_session_id") != coordinate_frame:
        errors.append("capture_reconstruction_candidate_coordinate_frame_mismatch")
    rows = candidate.get("candidates")
    if not isinstance(rows, list) or not rows:
        errors.append("capture_reconstruction_candidate_rows_missing")
        rows = []
    if candidate.get("candidate_count") != len(rows):
        errors.append("capture_reconstruction_candidate_count_mismatch")
    raw_video_digest = "sha256:" + str(candidate.get("source_video_sha256") or "")
    source_files = source.get("verified_source_files")
    verified_digests = {
        row.get("digest")
        for row in (source_files or [])
        if isinstance(row, Mapping)
    }
    if not _digest(raw_video_digest) or raw_video_digest not in verified_digests:
        errors.append("capture_reconstruction_candidate_source_video_unbound")
    if (
        not isinstance(selection, Mapping)
        or selection.get("selection_authority")
        != "blueprint_pipeline_task_site_profile"
        or selection.get("capture_default_selection") is not None
        or selection.get("selection_parameters_required") is not True
    ):
        errors.append("capture_reconstruction_candidate_selection_contract_invalid")
    if not isinstance(boundary, Mapping) or any(
        boundary.get(key) is not False
        for key in (
            "candidate_manifest_qualifies_reconstruction",
            "candidate_manifest_qualifies_metric_scale",
            "candidate_manifest_qualifies_collision_or_physics",
            "candidate_manifest_proves_task_success",
        )
    ):
        errors.append("capture_reconstruction_candidate_claim_boundary_invalid")
    if not isinstance(boundary, Mapping) or boundary.get(
        "raw_capture_remains_authoritative"
    ) is not True:
        errors.append("capture_reconstruction_candidate_raw_authority_invalid")
    if (
        not isinstance(neutrality, Mapping)
        or neutrality_map.get("third_party_provider_upload_authorized") is not False
        or neutrality_map.get("provider_selection_authority") != "blueprint_pipeline"
    ):
        errors.append("capture_reconstruction_candidate_provider_authority_invalid")
    if (
        neutrality_map.get("mobile_app_direct_provider_upload_allowed") is not False
        or neutrality_map.get("provider_authorization_status")
        != "not_granted_by_capture_manifest"
    ):
        errors.append("capture_reconstruction_candidate_provider_authorization_invalid")
    if (
        not isinstance(allowed_use, Mapping)
        or allowed_use.get("latest_revocation_check_required") is not True
        or allowed_use.get("provider_upload_requires_separate_downstream_authorization")
        is not True
    ):
        errors.append("capture_reconstruction_candidate_allowed_use_invalid")
    if any(
        not isinstance(row, Mapping)
        or row.get("coordinate_frame_session_id") != coordinate_frame
        or row.get("raw_observation_authority") is not True
        or row.get("downstream_artifact_authority") is not False
        for row in rows
    ):
        errors.append("capture_reconstruction_candidate_row_authority_invalid")
    if errors:
        raise CaptureReconstructionQualificationError(errors)
    return candidate


def _compare(observed: Any, operator: str, threshold: Any) -> bool:
    if operator == "eq":
        return observed == threshold
    if operator == "between_inclusive":
        if (
            not isinstance(threshold, list)
            or len(threshold) != 2
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in threshold)
            or isinstance(observed, bool)
            or not isinstance(observed, (int, float))
        ):
            return False
        return math.isfinite(float(observed)) and threshold[0] <= observed <= threshold[1]
    if (
        isinstance(observed, bool)
        or isinstance(threshold, bool)
        or not isinstance(observed, (int, float))
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(observed))
        or not math.isfinite(float(threshold))
    ):
        return False
    return {
        "gte": observed >= threshold,
        "lte": observed <= threshold,
        "gt": observed > threshold,
        "lt": observed < threshold,
    }.get(operator, False)


def _calibration_result(
    calibration_value: Mapping[str, Any] | None,
    *,
    profile: Mapping[str, Any],
    hardware_model_identifier: str,
    evaluated_at: datetime,
) -> dict[str, Any]:
    contract = dict(profile["device_calibration"])
    if calibration_value is None:
        return {
            "required": contract["required"],
            "status": "missing" if contract["required"] else "not_required",
            "qualified_for_device_sensor_scale": not contract["required"],
            "calibration_digest": None,
            "blocker": (
                "current_device_known_rig_calibration_missing"
                if contract["required"]
                else None
            ),
        }
    calibration = _clone(dict(calibration_value))
    hardware = calibration.get("hardwareModelIdentifier") or calibration.get(
        "hardware_model_identifier"
    )
    expires = calibration.get("expiresAt") or calibration.get("expires_at")
    relative_error = calibration.get("relativeError")
    if relative_error is None:
        relative_error = calibration.get("relative_error")
    mad = calibration.get("medianAbsoluteDeviationM")
    if mad is None:
        mad = calibration.get("median_absolute_deviation_m")
    count = calibration.get("acceptedSampleCount")
    if count is None:
        count = calibration.get("accepted_sample_count")
    qualified = bool(
        calibration.get("schemaVersion", calibration.get("schema_version"))
        == "device_calibration.v1"
        and calibration.get("status") == "qualified"
        and hardware == hardware_model_identifier
        and _parse_time(expires, code="device_calibration_expiry_invalid")
        > evaluated_at
        and isinstance(relative_error, (int, float))
        and not isinstance(relative_error, bool)
        and relative_error <= contract["maximum_relative_error"]
        and isinstance(mad, (int, float))
        and not isinstance(mad, bool)
        and mad <= contract["maximum_median_absolute_deviation_m"]
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= contract["minimum_accepted_sample_count"]
    )
    return {
        "required": contract["required"],
        "status": "qualified" if qualified else "rejected",
        "qualified_for_device_sensor_scale": qualified or not contract["required"],
        "calibration_digest": canonical_digest(calibration),
        "hardware_model_identifier": hardware,
        "expires_at": expires,
        "blocker": None if qualified or not contract["required"] else "current_device_known_rig_calibration_invalid",
        "claim_boundary": {
            "device_sensor_scale_supported": qualified,
            "site_geometry_qualified": False,
            "collision_geometry_qualified": False,
        },
    }


def _measurement_outcome(
    check: str,
    measurement_value: Mapping[str, Any] | None,
    *,
    profile: Mapping[str, Any],
    request_digest: str,
    source_capture_digest: str,
    coordinate_frame: str,
    candidate_manifest_digest: str,
    geometry: Mapping[str, Any],
    appearance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    rule = dict(_profile_checks(profile)[check])
    failure = {
        "check": check,
        "status": "missing",
        "passed": False,
        "observed_value": None,
        "operator": rule["operator"],
        "threshold": rule["threshold"],
        "measurement_digest": None,
        "failure_action": rule["failure_action"],
    }
    if measurement_value is None:
        return failure
    measurement = _clone(dict(measurement_value))
    digest_field = "measurement_digest"
    bindings_valid = bool(
        measurement.get("schema_version") == MEASUREMENT_SCHEMA
        and measurement.get("check") == check
        and measurement.get("request_digest") == request_digest
        and measurement.get("evidence_profile_digest")
        == profile["evidence_profile_digest"]
        and measurement.get("source_capture_digest") == source_capture_digest
        and measurement.get("coordinate_frame_session_id") == coordinate_frame
        and measurement.get("candidate_manifest_digest") == candidate_manifest_digest
        and measurement.get("candidate_may_self_qualify") is False
        and str(measurement.get("qualifier_identity") or "").strip()
        and str(measurement.get("producer_identity") or "").strip()
        and measurement.get("qualifier_identity") != measurement.get("producer_identity")
        and measurement.get(digest_field)
        == canonical_digest(measurement, digest_field=digest_field)
    )
    if check in _GEOMETRY_CHECKS:
        bindings_valid = bool(
            bindings_valid
            and measurement.get("derived_site_geometry_digest")
            == geometry["derived_site_geometry_digest"]
            and measurement.get("geometry_asset_digest")
            == geometry["geometry_asset_digest"]
            and measurement.get("collider_candidate_digest")
            == geometry["collider_candidate_digest"]
        )
    if check == "postshot_registered_reconstruction":
        bindings_valid = bool(
            bindings_valid
            and appearance is not None
            and measurement.get("native_3dgs_candidate_digest")
            == appearance.get("native_3dgs_candidate_digest")
            and measurement.get("appearance_asset_digest")
            == appearance.get("appearance_asset_digest")
            and measurement.get("derived_site_geometry_digest")
            == geometry["derived_site_geometry_digest"]
            and measurement.get("qualifier_identity")
            != appearance.get("provider_identity")
            and all(
                _digest(measurement.get(field))
                for field in (
                    "scene_registration_digest",
                    "registration_transform_digest",
                    "residual_measurement_digest",
                )
            )
        )
    passed = bindings_valid and _compare(
        measurement.get("observed_value"), rule["operator"], rule["threshold"]
    )
    return {
        **failure,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "observed_value": measurement.get("observed_value"),
        "measurement_digest": measurement.get(digest_field),
        "bindings_valid": bindings_valid,
        "qualifier_identity": measurement.get("qualifier_identity"),
    }


def compile_capture_reconstruction_qualification(
    *,
    request_value: Mapping[str, Any],
    evidence_profile_value: Mapping[str, Any],
    candidate_manifest_value: Mapping[str, Any],
    source_profile_value: Mapping[str, Any],
    geometry_candidate_value: Mapping[str, Any],
    appearance_candidate_value: Mapping[str, Any] | None,
    measurement_values: Sequence[Mapping[str, Any]],
    hardware_model_identifier: str,
    evaluated_at: str,
    device_calibration_value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile exact qualifications or an evidence-bounded abstention."""

    profile = validate_evidence_profile(evidence_profile_value)
    source = _validate_digest_artifact(
        source_profile_value,
        schema=SOURCE_PROFILE_SCHEMA,
        digest_field="source_profile_digest",
        code="capture_reconstruction_source_profile",
    )
    geometry = _validate_digest_artifact(
        geometry_candidate_value,
        schema=GEOMETRY_SCHEMA,
        digest_field="derived_site_geometry_digest",
        code="capture_reconstruction_geometry",
    )
    appearance = (
        _validate_digest_artifact(
            appearance_candidate_value,
            schema=NATIVE_3DGS_SCHEMA,
            digest_field="native_3dgs_candidate_digest",
            code="capture_reconstruction_appearance",
        )
        if appearance_candidate_value is not None
        else None
    )
    source_capture_digest = source.get("source_capture_digest")
    coordinate_frame = str(profile["coordinate_frame_session_id"])
    if (
        profile.get("source_capture_digest") != source_capture_digest
        or geometry.get("source_profile_digest") != source.get("source_profile_digest")
        or geometry.get("source_capture_digest") != source_capture_digest
        or (appearance is not None and appearance.get("source_profile_digest") != source.get("source_profile_digest"))
    ):
        raise CaptureReconstructionQualificationError(
            ["capture_reconstruction_exact_source_join_mismatch"]
        )
    geometry_state = geometry.get("qualification_state")
    if (
        geometry.get("status") != "derived_candidate_unqualified"
        or not isinstance(geometry_state, Mapping)
        or geometry_state.get("candidate_may_self_qualify") is not False
    ):
        raise CaptureReconstructionQualificationError(
            ["capture_reconstruction_geometry_candidate_authority_invalid"]
        )
    if appearance is not None and (
        appearance.get("provider_self_qualified") is not False
        or appearance.get("appearance_is_geometry_authority") is not False
    ):
        raise CaptureReconstructionQualificationError(
            ["capture_reconstruction_appearance_candidate_authority_invalid"]
        )
    request = _validate_request(request_value, coordinate_frame)
    request_digest = canonical_digest(request)
    candidate = _validate_candidate_manifest(
        candidate_manifest_value, coordinate_frame, source
    )
    candidate_digest = candidate["manifest_digest"]
    evaluated = _parse_time(evaluated_at, code="capture_reconstruction_evaluated_at_invalid")
    calibration = _calibration_result(
        device_calibration_value,
        profile=profile,
        hardware_model_identifier=str(hardware_model_identifier).strip(),
        evaluated_at=evaluated,
    )

    by_check: dict[str, Mapping[str, Any]] = {}
    duplicate_checks: list[str] = []
    for measurement in measurement_values:
        check = str(measurement.get("check") or "")
        if check in by_check:
            duplicate_checks.append(check)
        elif check in REQUIRED_CHECKS:
            by_check[check] = measurement
    if duplicate_checks:
        raise CaptureReconstructionQualificationError(
            [f"capture_reconstruction_duplicate_measurement:{check}" for check in duplicate_checks]
        )
    outcomes = [
        _measurement_outcome(
            check,
            by_check.get(check),
            profile=profile,
            request_digest=request_digest,
            source_capture_digest=source_capture_digest,
            coordinate_frame=coordinate_frame,
            candidate_manifest_digest=candidate_digest,
            geometry=geometry,
            appearance=appearance,
        )
        for check in REQUIRED_CHECKS
    ]
    outcome_map = {row["check"]: row for row in outcomes}
    calibration_ok = bool(calibration["qualified_for_device_sensor_scale"])
    scale_qualified = calibration_ok and all(outcome_map[key]["passed"] for key in SCALE_CHECKS)
    collision_qualified = scale_qualified and all(
        outcome_map[key]["passed"] for key in COLLISION_CHECKS
    )
    registration_qualified = bool(
        collision_qualified
        and appearance is not None
        and appearance.get("full_resolution_appearance_preserved") is True
        and outcome_map["postshot_registered_reconstruction"]["passed"]
    )
    first_failed = next((row for row in outcomes if not row["passed"]), None)
    if not calibration_ok:
        smallest = {
            "code": str(calibration.get("blocker") or "device_calibration_missing"),
            "instruction": "Run the current iPhone against the configured known rig, then re-evaluate the same capture.",
            "stage": "device_metric_calibration",
        }
    elif first_failed is not None:
        smallest = {
            "code": f"{first_failed['check']}_measurement_missing_or_failed",
            "instruction": first_failed["failure_action"],
            "stage": "capture_reconstruction_qualification",
        }
    else:
        smallest = None

    blockers = [] if collision_qualified else [smallest["code"] if smallest else "collision_evidence_incomplete"]
    geometry_qualification = _finalize(
        {
            "schema_version": GEOMETRY_QUALIFICATION_SCHEMA,
            "status": "qualified" if collision_qualified else "abstained",
            "derived_site_geometry_digest": geometry["derived_site_geometry_digest"],
            "geometry_asset_digest": geometry["geometry_asset_digest"],
            "collider_candidate_digest": geometry["collider_candidate_digest"],
            "request_digest": request_digest,
            "evidence_profile_digest": profile["evidence_profile_digest"],
            "candidate_manifest_digest": candidate_digest,
            "metric_scale_qualified": scale_qualified,
            "collision_geometry_qualified": collision_qualified,
            "isaac_contact_qualified": outcome_map["physical_collision_probes"]["passed"],
            "candidate_may_self_qualify": False,
            "qualifier_identity": "blueprint.capture_reconstruction_qualification_gate",
            "blockers": blockers,
            "smallest_missing_measurement": None if collision_qualified else smallest,
        },
        "geometry_qualification_digest",
    )
    qualified_geometry = build_qualified_site_geometry(
        geometry_candidate=geometry,
        independent_qualification=geometry_qualification,
    )

    registration_measurement = by_check.get("postshot_registered_reconstruction")
    registration_qualification = None
    if appearance is not None and registration_qualified:
        registration_qualification = _finalize(
            {
                "schema_version": REGISTRATION_QUALIFICATION_SCHEMA,
                "status": "qualified",
                "source_profile_digest": source["source_profile_digest"],
                "native_3dgs_candidate_digest": appearance["native_3dgs_candidate_digest"],
                "appearance_asset_digest": appearance["appearance_asset_digest"],
                "derived_site_geometry_digest": qualified_geometry[
                    "derived_site_geometry_digest"
                ],
                "geometry_asset_digest": qualified_geometry["geometry_asset_digest"],
                "scene_registration_digest": registration_measurement.get(
                    "scene_registration_digest"
                ),
                "registration_transform_digest": registration_measurement.get(
                    "registration_transform_digest"
                ),
                "residual_measurement_digest": registration_measurement.get(
                    "residual_measurement_digest"
                ),
                "request_digest": request_digest,
                "evidence_profile_digest": profile["evidence_profile_digest"],
                "candidate_manifest_digest": candidate_digest,
                "candidate_may_self_qualify": False,
                "qualifier_identity": "blueprint.capture_reconstruction_qualification_gate",
                "blockers": [],
                "smallest_missing_measurement": None,
            },
            "registration_qualification_digest",
        )

    decision = _finalize(
        {
            "schema_version": DECISION_SCHEMA,
            "status": "qualified" if registration_qualified else "abstained",
            "evaluated_at": evaluated.isoformat().replace("+00:00", "Z"),
            "task_id": profile["task_id"],
            "site_id": profile["site_id"],
            "source_capture_digest": source_capture_digest,
            "source_profile_digest": source["source_profile_digest"],
            "coordinate_frame_session_id": coordinate_frame,
            "request_digest": request_digest,
            "evidence_profile_digest": profile["evidence_profile_digest"],
            "candidate_manifest_digest": candidate_digest,
            "derived_site_geometry_digest": geometry["derived_site_geometry_digest"],
            "qualified_site_geometry_digest": qualified_geometry[
                "derived_site_geometry_digest"
            ],
            "native_3dgs_candidate_digest": (
                appearance.get("native_3dgs_candidate_digest") if appearance else None
            ),
            "checks": outcomes,
            "device_calibration": calibration,
            "claims": {
                "metric_scale": "qualified" if scale_qualified else "abstained",
                "collision_geometry": "qualified" if collision_qualified else "abstained",
                "registered_reconstruction": "qualified" if registration_qualified else "abstained",
            },
            "smallest_missing_measurement": smallest,
            "geometry_qualification_digest": geometry_qualification[
                "geometry_qualification_digest"
            ],
            "registration_qualification_digest": (
                registration_qualification["registration_qualification_digest"]
                if registration_qualification
                else None
            ),
            "claim_boundary": {
                "capture_request_is_qualification": False,
                "candidate_manifest_is_qualification": False,
                "device_calibration_qualifies_site_geometry": False,
                "postshot_appearance_is_geometry_authority": False,
                "collision_geometry_qualified": collision_qualified,
                "physical_site_surface_proven": False,
                "task_success_proven": False,
                "deployment_readiness_proven": False,
            },
        },
        "qualification_decision_digest",
    )
    registered_reconstruction = build_registered_site_reconstruction(
        source_profile=source,
        appearance_candidate=appearance,
        site_geometry=qualified_geometry,
        registration_qualification=registration_qualification,
    )
    return {
        "decision": decision,
        "geometry_qualification": geometry_qualification,
        "qualified_geometry": qualified_geometry,
        "registration_qualification": registration_qualification,
        "registered_reconstruction": registered_reconstruction,
    }


__all__ = [
    "CaptureReconstructionQualificationError",
    "DECISION_SCHEMA",
    "MEASUREMENT_SCHEMA",
    "PROFILE_SCHEMA",
    "REQUIRED_CHECKS",
    "compile_capture_reconstruction_qualification",
    "validate_evidence_profile",
]


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise CaptureReconstructionQualificationError(
                ["capture_reconstruction_output_conflict"]
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Qualify exact one-walk Capture reconstruction evidence."
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--evidence-profile", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--geometry-candidate", type=Path, required=True)
    parser.add_argument("--appearance-candidate", type=Path)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--device-calibration", type=Path)
    parser.add_argument("--hardware-model-identifier", required=True)
    parser.add_argument("--evaluated-at", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    measurements = _load(arguments.measurements)
    if not isinstance(measurements, list):
        raise CaptureReconstructionQualificationError(
            ["capture_reconstruction_measurements_not_list"]
        )
    result = compile_capture_reconstruction_qualification(
        request_value=_load(arguments.request),
        evidence_profile_value=_load(arguments.evidence_profile),
        candidate_manifest_value=_load(arguments.candidate_manifest),
        source_profile_value=_load(arguments.source_profile),
        geometry_candidate_value=_load(arguments.geometry_candidate),
        appearance_candidate_value=(
            _load(arguments.appearance_candidate)
            if arguments.appearance_candidate
            else None
        ),
        measurement_values=measurements,
        hardware_model_identifier=arguments.hardware_model_identifier,
        evaluated_at=arguments.evaluated_at,
        device_calibration_value=(
            _load(arguments.device_calibration)
            if arguments.device_calibration
            else None
        ),
    )
    filenames = {
        "decision": "qualification_decision.json",
        "geometry_qualification": "geometry_qualification.json",
        "qualified_geometry": "qualified_geometry.json",
        "registration_qualification": "registration_qualification.json",
        "registered_reconstruction": "registered_reconstruction.json",
    }
    for key, filename in filenames.items():
        value = result.get(key)
        if isinstance(value, Mapping):
            _write_immutable(arguments.output_dir / filename, value)
    print(
        canonical_json(
            {
                "status": result["decision"]["status"],
                "qualification_decision_digest": result["decision"][
                    "qualification_decision_digest"
                ],
                "smallest_missing_measurement": result["decision"][
                    "smallest_missing_measurement"
                ],
                "output_dir": str(arguments.output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
