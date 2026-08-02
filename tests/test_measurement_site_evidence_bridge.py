from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_site_evidence_bridge import (
    MeasurementSiteEvidenceBridgeError,
    build_robot_site_registration_record,
    build_site_evidence_profile_from_geometry,
)
from blueprint_pipeline.reconstruction_geometry_contracts import (
    build_collider_candidate_manifest,
    build_collider_qualification_report,
    build_metric_geometry_manifest,
)
from blueprint_pipeline.task_site_measurement_routing import (
    audit_site_evidence_profile,
    validate_site_evidence_profile,
)


ROOT = Path(__file__).parents[1]
SOURCE_DIGEST = "sha256:" + "1" * 64
SPLIT_DIGEST = "sha256:" + "2" * 64
GEOMETRY_DIGEST = "sha256:" + "3" * 64
COLLIDER_DIGEST = "sha256:" + "4" * 64
THRESHOLD_DIGEST = "sha256:" + "5" * 64
MEASUREMENT_DIGEST = "sha256:" + "6" * 64
ROBOT_DIGEST = "sha256:" + "7" * 64
TOOL_DIGEST = "sha256:" + "8" * 64


def _lineage(*, producing_method: str) -> dict:
    return {
        "stable_run_identity": "geometry-bridge-run-001",
        "source_capture_identity": "capture-001",
        "source_capture_digest": SOURCE_DIGEST,
        "original_file_references": [],
        "producing_method": producing_method,
        "implementation_version": "1.0.0",
        "source_commit_sha": "a" * 40,
        "deterministic_configuration_digest": "sha256:" + "9" * 64,
        "input_digests": [],
        "output_digests": [],
        "train_heldout_split_digest": SPLIT_DIGEST,
        "camera_calibration_binding": {"calibration_id": "camera-cal-001"},
        "coordinate_frame_declaration": {"units": "meters", "up_axis": "Z"},
        "units": "meters",
        "provider_runtime_identity": {"provider": "local"},
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "authority_used": {"mode": "execute_non_spend"},
        "warnings": [],
        "blockers": [],
        "parent_artifact_or_event": {"digest": SOURCE_DIGEST},
        "timestamp": "2026-08-02T12:00:00+00:00",
    }


def _geometry_artifacts() -> tuple[dict, dict, dict]:
    metric = build_metric_geometry_manifest(
        {
            **_lineage(producing_method="blueprint.observed_surface_confidence_filter"),
            "metric_scale_status": "validated",
            "generated_fill_used": False,
            "appearance_asset_used_as_geometry_truth": False,
            "observed_region_ids": ["task-zone"],
            "unsupported_region_ids": [],
            "confidence_filter": {"minimum_confidence": 0.9},
            "geometry_asset_digest": GEOMETRY_DIGEST,
            "proof_effect": "metric_reference_candidate_only",
            "claim_ceiling": "metric_reference_geometry",
        }
    )
    candidate = build_collider_candidate_manifest(
        {
            **_lineage(producing_method="blueprint.observed_surface_collider_baseline"),
            "metric_geometry_manifest_digest": metric["metric_geometry_manifest_digest"],
            "collider_asset_digest": COLLIDER_DIGEST,
            "unobserved_regions_filled": False,
            "collision_validated": False,
            "component_statistics": {"count": 1},
            "hole_statistics": {"count": 0},
            "proof_effect": "collision_candidate_only",
            "claim_ceiling": "collision_geometry_candidate",
        }
    )
    thresholds = {
        "scale_error_fraction": 0.01,
        "gravity_alignment_error_deg": 1.0,
        "floor_height_residual_m": 0.01,
        "wall_offset_residual_m": 0.01,
        "visual_to_collider_disagreement_m": 0.01,
        "clearance_error_m": 0.01,
        "mesh_coverage_fraction": 0.95,
        "minimum_obstacle_thickness_m": 0.02,
    }
    measurements = {
        "scale_error_fraction": 0.001,
        "gravity_alignment_error_deg": 0.1,
        "floor_height_residual_m": 0.001,
        "wall_offset_residual_m": 0.001,
        "visual_to_collider_disagreement_m": 0.002,
        "clearance_error_m": 0.002,
        "mesh_coverage_fraction": 0.99,
        "minimum_obstacle_thickness_m": 0.04,
    }
    qualification = build_collider_qualification_report(
        {
            **_lineage(producing_method="blueprint.independent_collider_measurement_evaluator"),
            "collider_candidate_manifest_digest": candidate["collider_candidate_manifest_digest"],
            "collider_asset_digest": COLLIDER_DIGEST,
            "qa_thresholds_digest": THRESHOLD_DIGEST,
            "measurements": measurements,
            "thresholds": thresholds,
            "metric_scale_status": "validated",
            "robot_footprint_navigability_checked": True,
            "measurement_artifact_digest": MEASUREMENT_DIGEST,
            "task_region_ids": ["task-zone"],
            "independent_evaluator": {
                "evaluator_id": "independent-collider-evaluator-001",
                "candidate_method_independent": True,
            },
            "candidate_self_graded": False,
            "decision": "accepted_bounded_navigation",
            "unsupported_claims": [
                "grasping",
                "articulation",
                "contact_force",
                "deployment",
                "physical_success",
            ],
            "proof_effect": "bounded_navigation_collision_qualification",
            "claim_ceiling": "bounded_navigation_simulation",
        }
    )
    return metric, candidate, qualification


def _registration(
    metric: dict,
    qualification: dict,
    *,
    development_only: bool = False,
    translation_error: float = 0.002,
    decision: str | None = None,
) -> dict:
    expected = (
        "development_only"
        if development_only
        else "accepted"
        if translation_error <= 0.01
        else "rejected"
    )
    return build_robot_site_registration_record(
        {
            "schema_version": "measurement_robot_site_registration.v1",
            "registration_id": "robot-site-registration-001",
            "source_capture_digest": SOURCE_DIGEST,
            "metric_geometry_manifest_digest": metric["metric_geometry_manifest_digest"],
            "collider_qualification_digest": qualification["collider_qualification_digest"],
            "robot_model_digest": ROBOT_DIGEST,
            "tool_geometry_digest": TOOL_DIGEST,
            "site_frame_id": "site-frame-001",
            "robot_base_frame_id": "robot-base-001",
            "site_from_robot_base_transform": [
                [1.0, 0.0, 0.0, 0.2],
                [0.0, 1.0, 0.0, -0.1],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "measurement_method": "surveyed_fiducial_registration",
            "independent_measurement_digest": MEASUREMENT_DIGEST,
            "independent_evaluator": {
                "evaluator_id": "independent-metrology-001",
                "candidate_method_independent": True,
                "agent_is_evaluator": False,
            },
            "measurement_signature_status": ("unverified" if development_only else "verified"),
            "evaluator_approval_signature_id": (
                "" if development_only else "signature-registration-001"
            ),
            "candidate_self_graded": False,
            "agent_generated_measurements": False,
            "thresholds_modified_after_measurement": False,
            "measurement_sample_count": 12,
            "translation_rms_error_m": translation_error,
            "rotation_rms_error_deg": 0.2,
            "maximum_translation_rms_error_m": 0.01,
            "maximum_rotation_rms_error_deg": 1.0,
            "task_region_ids": ["task-zone"],
            "blockers": [],
            "metric_scale_verified": True,
            "collider_qualification_decision": "accepted_bounded_navigation",
            "physical_measurements_included": not development_only,
            "development_only": development_only,
            "decision": decision or expected,
            "agent_may_approve": False,
            "r5_evidence": False,
            "r6_decision": False,
            "r7_admission": False,
            "physical_success_established": False,
            "deployment_readiness_established": False,
            "safety_established": False,
            "proof_effect": "robot_site_registration_evidence",
            "claim_ceiling": "C2",
        }
    )


def _profile(registration: dict) -> dict:
    metric, candidate, qualification = _geometry_artifacts()
    return build_site_evidence_profile_from_geometry(
        metric_geometry_manifest=metric,
        collider_candidate_manifest=candidate,
        collider_qualification_report=qualification,
        robot_site_registration=registration,
        profile_id="site-evidence-geometry-bridge-001",
        bundle_id="capture-001",
        bundle_hash=SOURCE_DIGEST,
        provenance_record_id="provenance-001",
        rights={"commercial_use_allowed": True},
        privacy={"customer_site_data": True},
        additional_evidence={
            "sensor_calibration": {
                "available": True,
                "validated": False,
                "record_id": "sensor-calibration-candidate-001",
            }
        },
    )


def test_registration_and_bridge_create_validated_scoped_site_evidence() -> None:
    metric, candidate, qualification = _geometry_artifacts()
    registration = _registration(metric, qualification)
    profile = build_site_evidence_profile_from_geometry(
        metric_geometry_manifest=metric,
        collider_candidate_manifest=candidate,
        collider_qualification_report=qualification,
        robot_site_registration=registration,
        profile_id="site-evidence-geometry-bridge-001",
        bundle_id="capture-001",
        bundle_hash=SOURCE_DIGEST,
        provenance_record_id="provenance-001",
        rights={"commercial_use_allowed": True},
        privacy={"customer_site_data": True},
    )
    assert registration["decision"] == "accepted"
    assert profile["coordinate_system"]["metric_scale_verified"] is True
    assert profile["evidence"]["validated_mesh"]["validated"] is True
    assert profile["evidence"]["validated_collider"]["validated"] is True
    assert profile["evidence"]["robot_site_registration"]["validated"] is True
    assert profile["geometry_bridge"]["method_qualification_created"] is False
    assert profile["geometry_bridge"]["r5_evidence_created"] is False
    assert profile["geometry_bridge"]["physical_success_established"] is False
    gaps = {gap["evidence_id"] for gap in audit_site_evidence_profile(profile)["gaps"]}
    assert gaps.isdisjoint(
        {
            "metric_scale",
            "validated_mesh",
            "validated_collider",
            "robot_site_registration",
        }
    )


def test_development_registration_never_becomes_validated_site_registration() -> None:
    metric, candidate, qualification = _geometry_artifacts()
    registration = _registration(metric, qualification, development_only=True)
    profile = build_site_evidence_profile_from_geometry(
        metric_geometry_manifest=metric,
        collider_candidate_manifest=candidate,
        collider_qualification_report=qualification,
        robot_site_registration=registration,
        profile_id="site-evidence-development-001",
        bundle_id="capture-001",
        bundle_hash=SOURCE_DIGEST,
        provenance_record_id="provenance-001",
        rights={},
        privacy={},
    )
    assert registration["decision"] == "development_only"
    assert profile["evidence"]["validated_collider"]["validated"] is True
    assert profile["evidence"]["robot_site_registration"]["validated"] is False
    assert profile["geometry_bridge"]["development_only"] is True


def test_registration_threshold_failure_is_valid_rejection() -> None:
    metric, _, qualification = _geometry_artifacts()
    registration = _registration(metric, qualification, translation_error=0.02)
    assert registration["decision"] == "rejected"
    assert registration["translation_rms_error_m"] == 0.02

    unsigned = copy.deepcopy(_registration(metric, qualification))
    unsigned.pop("registration_digest")
    unsigned["measurement_signature_status"] = "unverified"
    unsigned["evaluator_approval_signature_id"] = ""
    unsigned["decision"] = "rejected"
    unsigned = build_robot_site_registration_record(unsigned)
    assert unsigned["decision"] == "rejected"


def test_registration_rejects_invalid_se3_and_decision_tampering() -> None:
    metric, _, qualification = _geometry_artifacts()
    valid = _registration(metric, qualification)
    transform = copy.deepcopy(valid)
    transform.pop("registration_digest")
    transform["site_from_robot_base_transform"][0][0] = 2.0
    with pytest.raises(MeasurementSiteEvidenceBridgeError, match="transform_invalid"):
        build_robot_site_registration_record(transform)

    with pytest.raises(MeasurementSiteEvidenceBridgeError, match="decision_not_deterministic"):
        _registration(metric, qualification, decision="rejected")


def test_bridge_rejects_lineage_mismatch_and_protected_override() -> None:
    metric, candidate, qualification = _geometry_artifacts()
    registration = _registration(metric, qualification)
    mismatch = copy.deepcopy(registration)
    mismatch.pop("registration_digest")
    mismatch["source_capture_digest"] = "sha256:" + "0" * 64
    mismatch = build_robot_site_registration_record(mismatch)
    with pytest.raises(MeasurementSiteEvidenceBridgeError, match="source_capture_mismatch"):
        build_site_evidence_profile_from_geometry(
            metric_geometry_manifest=metric,
            collider_candidate_manifest=candidate,
            collider_qualification_report=qualification,
            robot_site_registration=mismatch,
            profile_id="site-evidence-001",
            bundle_id="capture-001",
            bundle_hash=SOURCE_DIGEST,
            provenance_record_id="provenance-001",
            rights={},
            privacy={},
        )
    with pytest.raises(MeasurementSiteEvidenceBridgeError, match="protected_evidence_override"):
        build_site_evidence_profile_from_geometry(
            metric_geometry_manifest=metric,
            collider_candidate_manifest=candidate,
            collider_qualification_report=qualification,
            robot_site_registration=registration,
            profile_id="site-evidence-001",
            bundle_id="capture-001",
            bundle_hash=SOURCE_DIGEST,
            provenance_record_id="provenance-001",
            rights={},
            privacy={},
            additional_evidence={"validated_collider": {"available": True, "validated": True}},
        )

    incomplete_qualification = copy.deepcopy(qualification)
    for key in (
        "collider_qualification_digest",
        "measurement_artifact_digest",
        "independent_evaluator",
        "candidate_self_graded",
    ):
        incomplete_qualification.pop(key, None)
    incomplete_qualification = build_collider_qualification_report(incomplete_qualification)
    incomplete_registration = _registration(metric, incomplete_qualification)
    with pytest.raises(
        MeasurementSiteEvidenceBridgeError,
        match="independent_collider_evidence_incomplete",
    ):
        build_site_evidence_profile_from_geometry(
            metric_geometry_manifest=metric,
            collider_candidate_manifest=candidate,
            collider_qualification_report=incomplete_qualification,
            robot_site_registration=incomplete_registration,
            profile_id="site-evidence-001",
            bundle_id="capture-001",
            bundle_hash=SOURCE_DIGEST,
            provenance_record_id="provenance-001",
            rights={},
            privacy={},
        )


def test_registration_and_profile_match_checked_json_schemas() -> None:
    metric, candidate, qualification = _geometry_artifacts()
    registration = _registration(metric, qualification)
    profile = build_site_evidence_profile_from_geometry(
        metric_geometry_manifest=metric,
        collider_candidate_manifest=candidate,
        collider_qualification_report=qualification,
        robot_site_registration=registration,
        profile_id="site-evidence-geometry-bridge-001",
        bundle_id="capture-001",
        bundle_hash=SOURCE_DIGEST,
        provenance_record_id="provenance-001",
        rights={},
        privacy={},
    )
    jsonschema.validate(
        registration,
        json.loads(
            (ROOT / "docs/schemas/measurement_robot_site_registration.v1.schema.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    jsonschema.validate(
        profile,
        json.loads(
            (ROOT / "docs/schemas/task_site_measurement_routing.v1.schema.json").read_text(
                encoding="utf-8"
            )
        ),
    )

    authority = copy.deepcopy(profile)
    authority.pop("site_evidence_digest")
    authority["geometry_bridge"]["r7_admission_created"] = True
    with pytest.raises(ValueError, match="r7_admission_created_must_be_false"):
        validate_site_evidence_profile(authority)

    binding = copy.deepcopy(profile)
    binding.pop("site_evidence_digest")
    binding["evidence"]["validated_collider"]["record_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="validated_collider_record_mismatch"):
        validate_site_evidence_profile(binding)
