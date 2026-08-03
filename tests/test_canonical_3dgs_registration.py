from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import jsonschema
import pytest

from blueprint_pipeline.canonical_3dgs_registration import (
    Canonical3DGSRegistrationError,
    build_canonical_3dgs_registration_measurement,
    build_canonical_registered_appearance,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _write_standard_splat(path: Path) -> str:
    properties = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        + "".join(f"property float {name}\n" for name in properties)
        + "end_header\n"
    )
    path.write_bytes(
        header.encode("ascii")
        + struct.pack("<14f", 0, 0, 1, 0, 0, 0, 1, -3, -3, -3, 1, 0, 0, 0)
    )
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source(geometry_digest: str) -> dict:
    value = {
        "schema_version": "canonical_3dgs_source_admission.v1",
        "status": "admitted_candidate_training_source",
        "source_profile": "public_dataset_arkitscenes_proxy",
        "source_capture_digest": _sha("a"),
        "world_frame": "arkitscenes_official_loader_world",
        "coordinate_frame_declaration": {"frame": "arkitscenes_official_loader_world"},
        "metric_scale_status": "sensor_metric_unvalidated",
        "metric_scale_independently_validated": False,
        "input_artifacts": [{"artifact_id": "observed_surface", "digest": geometry_digest}],
    }
    value["canonical_3dgs_source_admission_digest"] = canonical_digest(
        value, digest_field="canonical_3dgs_source_admission_digest"
    )
    return value


def _campaign(source: dict, asset_digest: str) -> dict:
    value = {
        "schema_version": "canonical_3dgs_campaign_result.v1",
        "status": "candidates_ready_for_independent_evaluation",
        "source_capture_digest": source["source_capture_digest"],
        "source_commit_sha": "c" * 40,
        "canonical_3dgs_source_admission_digest": source[
            "canonical_3dgs_source_admission_digest"
        ],
        "world_frame": source["world_frame"],
        "metric_scale_status": source["metric_scale_status"],
        "appearance_fidelity_candidate_bindings": [
            {
                "candidate_arm_id": "postshot-primary",
                "candidate_method_id": "jawset_postshot_splat3_v1",
                "candidate_role": "primary",
                "asset_digest": asset_digest,
                "coordinate_basis_digest": _sha("b"),
                "representation": "standard_3dgs_ply",
                "splat_count": 1,
                "sh_degree": 0,
                "global_decimation_applied": False,
            }
        ],
        "timestamp": "2026-08-03T12:00:00Z",
    }
    value["canonical_3dgs_campaign_result_digest"] = canonical_digest(
        value, digest_field="canonical_3dgs_campaign_result_digest"
    )
    return value


def _quality(campaign: dict) -> dict:
    value = {
        "schema_version": "canonical_3dgs_quality_comparison.v1",
        "status": "quality_winner_selected",
        "canonical_3dgs_campaign_result_digest": campaign[
            "canonical_3dgs_campaign_result_digest"
        ],
        "candidate_hidden_pixel_access": False,
        "quality_winner": "postshot-primary",
        "candidate_reports": [
            {
                "arm_id": "postshot-primary",
                "appearance_fidelity_status": "qualified",
                "appearance_fidelity_qualification_digest": _sha("d"),
            }
        ],
    }
    value["canonical_3dgs_quality_comparison_digest"] = canonical_digest(
        value, digest_field="canonical_3dgs_quality_comparison_digest"
    )
    return value


def test_registered_appearance_remains_candidate_until_both_gates_pass(
    tmp_path: Path,
) -> None:
    splat = tmp_path / "candidate.ply"
    asset_digest = _write_standard_splat(splat)
    geometry_digest = _sha("e")
    source = _source(geometry_digest)
    campaign = _campaign(source, asset_digest)

    candidate = build_canonical_registered_appearance(
        source_admission=source,
        campaign_result=campaign,
        appearance_asset_path=splat,
        appearance_asset_reference="results/postshot-primary/candidate.ply",
        geometry_asset_digest=geometry_digest,
    )
    assert candidate["status"] == "candidate_only"
    assert candidate["registration_residual_summary"] is None
    assert candidate["metric_scale_status"] == "sensor_metric_unvalidated"
    assert candidate["metric_scale_proven"] is False
    assert candidate["collision_geometry_validated"] is False

    measurement = build_canonical_3dgs_registration_measurement(
        source_capture_digest=source["source_capture_digest"],
        appearance_asset_digest=asset_digest,
        world_frame=source["world_frame"],
        metric_scale_status=source["metric_scale_status"],
        transform_appearance_to_site=[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        correspondences=[
            {"correspondence_id": "a", "appearance_point": [0, 0, 0], "site_point": [0, 0, 0]},
            {"correspondence_id": "b", "appearance_point": [1, 0, 0], "site_point": [1, 0, 0]},
            {"correspondence_id": "c", "appearance_point": [0, 1, 0], "site_point": [0, 1, 0]},
        ],
        thresholds_m={
            "maximum_rmse_m": 0.01,
            "maximum_p95_m": 0.01,
            "maximum_residual_m": 0.01,
        },
        method_id="identity_frame_correspondence.v1",
        threshold_frozen_before_measurement=True,
        timestamp="2026-08-03T12:00:00Z",
    )
    registered = build_canonical_registered_appearance(
        source_admission=source,
        campaign_result=campaign,
        appearance_asset_path=splat,
        appearance_asset_reference="results/postshot-primary/candidate.ply",
        geometry_asset_digest=geometry_digest,
        quality_comparison=_quality(campaign),
        registration_measurement=measurement,
    )
    assert registered["status"] == "qualified"
    assert registered["registration_status"] == "qualified"
    assert registered["registration_residual_summary"]["rmse_m"] == 0.0
    assert registered["metric_scale_proven"] is False
    assert registered["collision_geometry_validated"] is False
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/canonical_registered_appearance.v1.schema.json"
        ).read_text()
    )
    jsonschema.validate(registered, schema)


def test_registration_computes_residuals_and_fails_closed_on_tampering(tmp_path: Path) -> None:
    splat = tmp_path / "candidate.ply"
    asset_digest = _write_standard_splat(splat)
    measurement = build_canonical_3dgs_registration_measurement(
        source_capture_digest=_sha("a"),
        appearance_asset_digest=asset_digest,
        world_frame="arkitscenes_official_loader_world",
        metric_scale_status="sensor_metric_unvalidated",
        transform_appearance_to_site=[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        correspondences=[
            {"correspondence_id": "a", "appearance_point": [0, 0, 0], "site_point": [0.2, 0, 0]},
            {"correspondence_id": "b", "appearance_point": [1, 0, 0], "site_point": [1, 0, 0]},
            {"correspondence_id": "c", "appearance_point": [0, 1, 0], "site_point": [0, 1, 0]},
        ],
        thresholds_m={
            "maximum_rmse_m": 0.01,
            "maximum_p95_m": 0.01,
            "maximum_residual_m": 0.01,
        },
        method_id="fixture_correspondence.v1",
        threshold_frozen_before_measurement=True,
        timestamp="2026-08-03T12:00:00Z",
    )
    assert measurement["status"] == "failed_residual_gate"
    assert measurement["residual_summary"]["maximum_residual_m"] == pytest.approx(0.2)

    source = _source(_sha("e"))
    campaign = _campaign(source, asset_digest)
    tampered = dict(measurement)
    tampered["registration_gate_passed"] = True
    with pytest.raises(
        Canonical3DGSRegistrationError,
        match="registered_reconstruction_registration_binding_invalid",
    ):
        build_canonical_registered_appearance(
            source_admission=source,
            campaign_result=campaign,
            appearance_asset_path=splat,
            appearance_asset_reference="candidate.ply",
            geometry_asset_digest=_sha("e"),
            quality_comparison=_quality(campaign),
            registration_measurement=tampered,
        )


def test_registration_rejects_non_surface_source_artifact(tmp_path: Path) -> None:
    splat = tmp_path / "candidate.ply"
    asset_digest = _write_standard_splat(splat)
    source = _source(_sha("e"))
    source["input_artifacts"].append(
        {"artifact_id": "hidden_evaluator_input", "digest": _sha("f")}
    )
    source["canonical_3dgs_source_admission_digest"] = canonical_digest(
        source, digest_field="canonical_3dgs_source_admission_digest"
    )
    campaign = _campaign(source, asset_digest)
    with pytest.raises(
        Canonical3DGSRegistrationError,
        match="registered_reconstruction_geometry_source_binding_invalid",
    ):
        build_canonical_registered_appearance(
            source_admission=source,
            campaign_result=campaign,
            appearance_asset_path=splat,
            appearance_asset_reference="candidate.ply",
            geometry_asset_digest=_sha("f"),
        )
