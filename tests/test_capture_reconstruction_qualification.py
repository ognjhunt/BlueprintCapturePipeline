from __future__ import annotations

from copy import deepcopy
import json

import pytest

from blueprint_pipeline.capture_reconstruction_qualification import (
    CaptureReconstructionQualificationError,
    MEASUREMENT_SCHEMA,
    PROFILE_SCHEMA,
    REQUIRED_CHECKS,
    compile_capture_reconstruction_qualification,
    main,
    validate_evidence_profile,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _digest(label: str) -> str:
    return "sha256:" + label.encode().hex().ljust(64, "0")[:64]


def _finalize(value: dict, field: str) -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _profile(*, calibration_required: bool = False) -> dict:
    rules = {
        "loop_closure": ("eq", True, "Recapture only the return-to-start segment."),
        "tracking_quality": ("gte", 0.95, "Recapture the tracking-degraded segment."),
        "depth_reprojection_error": ("lte", 0.02, "Recapture the depth-inconsistent segment."),
        "mesh_coverage": ("gte", 0.85, "Recapture only the uncovered task zone."),
        "floor_support_continuity": ("gte", 0.90, "Recapture the discontinuous support boundary."),
        "physical_collision_probes": ("gte", 0.99, "Re-run probes for the exact collider bytes."),
        "postshot_registered_reconstruction": ("lte", 0.015, "Re-register the exact Postshot appearance bytes."),
    }
    profile = {
        "schema_version": PROFILE_SCHEMA,
        "profile_id": "site-1-task-1-v1",
        "task_id": "task-1",
        "site_id": "site-1",
        "source_capture_digest": _digest("capture"),
        "coordinate_frame_session_id": "arkit-session-1",
        "checks": {
            check: {
                "operator": rule[0],
                "threshold": rule[1],
                "failure_action": rule[2],
            }
            for check, rule in rules.items()
        },
        "device_calibration": {
            "required": calibration_required,
            "maximum_relative_error": 0.015,
            "maximum_median_absolute_deviation_m": 0.008,
            "minimum_accepted_sample_count": 45,
        },
    }
    return _finalize(profile, "evidence_profile_digest")


def _request() -> dict:
    return {
        "schema_version": "reconstruction_qualification_request.v1",
        "coordinate_frame_session_id": "arkit-session-1",
        "capture_workflow": "single_guided_closed_loop_walk",
        "requested_checks": [
            {"check": check, "capture_observation_available": check in REQUIRED_CHECKS[:2]}
            for check in REQUIRED_CHECKS
        ],
        "threshold_source": "task_site_evidence_profile_digest_bound",
        "qualification_authority": "blueprint_pipeline",
        "capture_decision": "abstain_pending_downstream_measurements",
        "smallest_missing_evidence": [],
        "collision_artifact_status": "candidate_only",
        "automatic_promotion_rule": "qualify_only_when_every_requested_check_passes_under_the_bound_task_site_profile",
    }


def _candidate_manifest() -> dict:
    manifest = {
        "schema_version": "downstream_candidate_manifest.v1",
        "coordinate_frame_session_id": "arkit-session-1",
        "source_video_sha256": "a" * 64,
        "candidate_count": 1,
        "selection_contract": {
            "selection_authority": "blueprint_pipeline_task_site_profile",
            "capture_default_selection": None,
            "selection_parameters_required": True,
        },
        "provider_neutrality": {
            "mobile_app_direct_provider_upload_allowed": False,
            "third_party_provider_upload_authorized": False,
            "provider_selection_authority": "blueprint_pipeline",
            "provider_authorization_status": "not_granted_by_capture_manifest",
        },
        "allowed_use_scope": {
            "latest_revocation_check_required": True,
            "provider_upload_requires_separate_downstream_authorization": True,
        },
        "claim_boundary": {
            "raw_capture_remains_authoritative": True,
            "candidate_manifest_qualifies_reconstruction": False,
            "candidate_manifest_qualifies_metric_scale": False,
            "candidate_manifest_qualifies_collision_or_physics": False,
            "candidate_manifest_proves_task_success": False,
        },
        "candidates": [
            {
                "candidate_id": "rgb_000000",
                "coordinate_frame_session_id": "arkit-session-1",
                "raw_observation_authority": True,
                "downstream_artifact_authority": False,
            }
        ],
    }
    return _finalize(manifest, "manifest_digest")


def _source_profile() -> dict:
    return _finalize(
        {
            "schema_version": "post_capture_source_profile.v1",
            "source_capture_digest": _digest("capture"),
            "metric_scale_status": "sensor_declared_not_independently_validated",
            "verified_source_files": [
                {"relative_path": "walkthrough.mov", "digest": "sha256:" + "a" * 64}
            ],
        },
        "source_profile_digest",
    )


def _geometry(source: dict) -> dict:
    return _finalize(
        {
            "schema_version": "derived_site_geometry.v1",
            "status": "derived_candidate_unqualified",
            "source_profile_digest": source["source_profile_digest"],
            "source_capture_digest": source["source_capture_digest"],
            "geometry_asset_digest": _digest("geometry"),
            "collider_candidate_digest": _digest("collider"),
            "qualification_state": {
                "metric_scale": "unqualified",
                "collision_geometry": "unqualified",
                "isaac_contact": "unqualified",
                "candidate_may_self_qualify": False,
            },
            "claim_boundary": {
                "metric_geometry_proven": False,
                "collision_geometry_proven": False,
                "physical_surface_proven": False,
            },
        },
        "derived_site_geometry_digest",
    )


def _appearance(source: dict) -> dict:
    return _finalize(
        {
            "schema_version": "native_3dgs_candidate.v1",
            "source_profile_digest": source["source_profile_digest"],
            "source_capture_digest": source["source_capture_digest"],
            "provider_identity": "postshot",
            "appearance_asset_digest": _digest("appearance"),
            "full_resolution_appearance_preserved": True,
            "provider_self_qualified": False,
            "appearance_is_geometry_authority": False,
        },
        "native_3dgs_candidate_digest",
    )


def _measurements(
    *, profile: dict, request: dict, candidate: dict, geometry: dict, appearance: dict
) -> list[dict]:
    observed = {
        "loop_closure": True,
        "tracking_quality": 0.98,
        "depth_reprojection_error": 0.01,
        "mesh_coverage": 0.92,
        "floor_support_continuity": 0.96,
        "physical_collision_probes": 1.0,
        "postshot_registered_reconstruction": 0.01,
    }
    rows = []
    for check in REQUIRED_CHECKS:
        row = {
            "schema_version": MEASUREMENT_SCHEMA,
            "check": check,
            "request_digest": canonical_digest(request),
            "evidence_profile_digest": profile["evidence_profile_digest"],
            "source_capture_digest": _digest("capture"),
            "coordinate_frame_session_id": "arkit-session-1",
            "candidate_manifest_digest": candidate["manifest_digest"],
            "observed_value": observed[check],
            "qualifier_identity": "blueprint.independent_measurement_runner",
            "producer_identity": (
                "postshot" if check == "postshot_registered_reconstruction" else "blueprint.capture_observer"
            ),
            "candidate_may_self_qualify": False,
        }
        if check in REQUIRED_CHECKS[3:6] or check == "postshot_registered_reconstruction":
            row.update(
                {
                    "derived_site_geometry_digest": geometry[
                        "derived_site_geometry_digest"
                    ],
                    "geometry_asset_digest": geometry["geometry_asset_digest"],
                    "collider_candidate_digest": geometry["collider_candidate_digest"],
                }
            )
        if check == "postshot_registered_reconstruction":
            row.update(
                {
                    "native_3dgs_candidate_digest": appearance[
                        "native_3dgs_candidate_digest"
                    ],
                    "appearance_asset_digest": appearance["appearance_asset_digest"],
                    "scene_registration_digest": _digest("registration"),
                    "registration_transform_digest": _digest("transform"),
                    "residual_measurement_digest": _digest("residual"),
                }
            )
        rows.append(_finalize(row, "measurement_digest"))
    return rows


def _inputs(*, calibration_required: bool = False) -> dict:
    profile = _profile(calibration_required=calibration_required)
    request = _request()
    candidate = _candidate_manifest()
    source = _source_profile()
    geometry = _geometry(source)
    appearance = _appearance(source)
    return {
        "request_value": request,
        "evidence_profile_value": profile,
        "candidate_manifest_value": candidate,
        "source_profile_value": source,
        "geometry_candidate_value": geometry,
        "appearance_candidate_value": appearance,
        "measurement_values": _measurements(
            profile=profile,
            request=request,
            candidate=candidate,
            geometry=geometry,
            appearance=appearance,
        ),
        "hardware_model_identifier": "iPhone18,1",
        "evaluated_at": "2026-08-03T15:00:00Z",
    }


def test_all_exact_checks_qualify_scale_collision_and_registered_reconstruction() -> None:
    result = compile_capture_reconstruction_qualification(**_inputs())

    assert result["decision"]["status"] == "qualified"
    assert result["decision"]["claims"] == {
        "metric_scale": "qualified",
        "collision_geometry": "qualified",
        "registered_reconstruction": "qualified",
    }
    assert result["decision"]["smallest_missing_measurement"] is None
    assert result["qualified_geometry"]["status"] == "qualified"
    assert result["registered_reconstruction"]["status"] == "qualified"
    assert result["decision"]["claim_boundary"]["physical_site_surface_proven"] is False
    assert result["decision"]["claim_boundary"]["task_success_proven"] is False


def test_missing_collision_probe_abstains_with_profile_defined_smallest_recapture() -> None:
    inputs = _inputs()
    inputs["measurement_values"] = [
        row
        for row in inputs["measurement_values"]
        if row["check"] != "physical_collision_probes"
    ]

    result = compile_capture_reconstruction_qualification(**inputs)

    assert result["decision"]["status"] == "abstained"
    assert result["decision"]["claims"]["metric_scale"] == "qualified"
    assert result["decision"]["claims"]["collision_geometry"] == "abstained"
    assert result["decision"]["smallest_missing_measurement"] == {
        "code": "physical_collision_probes_measurement_missing_or_failed",
        "instruction": "Re-run probes for the exact collider bytes.",
        "stage": "capture_reconstruction_qualification",
    }
    assert result["registered_reconstruction"]["status"] == "abstained"


def test_required_calibration_is_device_bound_and_cannot_qualify_site_geometry() -> None:
    inputs = _inputs(calibration_required=True)
    inputs["device_calibration_value"] = {
        "schemaVersion": "device_calibration.v1",
        "hardwareModelIdentifier": "iPhone17,1",
        "expiresAt": "2026-11-01T00:00:00Z",
        "relativeError": 0.001,
        "medianAbsoluteDeviationM": 0.001,
        "acceptedSampleCount": 60,
        "status": "qualified",
    }

    result = compile_capture_reconstruction_qualification(**inputs)

    assert result["decision"]["claims"]["metric_scale"] == "abstained"
    assert result["decision"]["smallest_missing_measurement"]["code"] == (
        "current_device_known_rig_calibration_invalid"
    )
    assert result["decision"]["device_calibration"]["claim_boundary"] == {
        "device_sensor_scale_supported": False,
        "site_geometry_qualified": False,
        "collision_geometry_qualified": False,
    }


def test_current_known_rig_calibration_can_support_device_scale_gate() -> None:
    inputs = _inputs(calibration_required=True)
    inputs["device_calibration_value"] = {
        "schemaVersion": "device_calibration.v1",
        "hardwareModelIdentifier": "iPhone18,1",
        "expiresAt": "2026-11-01T00:00:00Z",
        "relativeError": 0.001,
        "medianAbsoluteDeviationM": 0.001,
        "acceptedSampleCount": 60,
        "status": "qualified",
    }

    result = compile_capture_reconstruction_qualification(**inputs)

    assert result["decision"]["status"] == "qualified"
    calibration = result["decision"]["device_calibration"]
    assert calibration["qualified_for_device_sensor_scale"] is True
    assert calibration["claim_boundary"]["site_geometry_qualified"] is False


def test_provider_or_candidate_cannot_self_qualify_registration() -> None:
    inputs = _inputs()
    rows = deepcopy(inputs["measurement_values"])
    registration = next(
        row for row in rows if row["check"] == "postshot_registered_reconstruction"
    )
    registration["qualifier_identity"] = "postshot"
    _finalize(registration, "measurement_digest")
    inputs["measurement_values"] = rows

    result = compile_capture_reconstruction_qualification(**inputs)

    check = result["decision"]["checks"][-1]
    assert check["check"] == "postshot_registered_reconstruction"
    assert check["bindings_valid"] is False
    assert result["decision"]["status"] == "abstained"


def test_profile_must_define_every_site_task_threshold() -> None:
    profile = _profile()
    del profile["checks"]["mesh_coverage"]
    _finalize(profile, "evidence_profile_digest")

    with pytest.raises(CaptureReconstructionQualificationError) as exc_info:
        validate_evidence_profile(profile)

    assert "capture_reconstruction_profile_checks_incomplete" in exc_info.value.codes


def test_cli_writes_immutable_spine_compatible_artifacts(tmp_path) -> None:
    inputs = _inputs()
    paths = {}
    for key, value in (
        ("request", inputs["request_value"]),
        ("profile", inputs["evidence_profile_value"]),
        ("candidate", inputs["candidate_manifest_value"]),
        ("source", inputs["source_profile_value"]),
        ("geometry", inputs["geometry_candidate_value"]),
        ("appearance", inputs["appearance_candidate_value"]),
        ("measurements", inputs["measurement_values"]),
    ):
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[key] = path
    output = tmp_path / "output"
    argv = [
        "--request",
        str(paths["request"]),
        "--evidence-profile",
        str(paths["profile"]),
        "--candidate-manifest",
        str(paths["candidate"]),
        "--source-profile",
        str(paths["source"]),
        "--geometry-candidate",
        str(paths["geometry"]),
        "--appearance-candidate",
        str(paths["appearance"]),
        "--measurements",
        str(paths["measurements"]),
        "--hardware-model-identifier",
        "iPhone18,1",
        "--evaluated-at",
        "2026-08-03T15:00:00Z",
        "--output-dir",
        str(output),
    ]

    assert main(argv) == 0
    assert main(argv) == 0
    assert json.loads((output / "qualification_decision.json").read_text())["status"] == (
        "qualified"
    )
    assert json.loads((output / "registered_reconstruction.json").read_text())["status"] == (
        "qualified"
    )
