from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.site_reference_database import (
    EVALUATION_SITE_ADMISSION_SCHEMA_VERSION,
    SITE_REFERENCE_DATABASE_SCHEMA_VERSION,
    WEBAPP_PROJECTION_SCHEMA_VERSION,
    SiteReferenceContractError,
    _path_to_gs_uri,
    _read_optional_json,
    assert_summary_projection_safe,
    build_reference_record_lineage,
    build_site_reference_manifest_payload,
    build_site_reference_summary_projection,
    validate_site_reference_manifest,
    validate_site_reference_record,
    validate_evaluation_site_admission,
)


def _pose(tx: float = 0.0) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _record() -> dict[str, object]:
    return {
        "reference_id": "ref-1",
        "site_id": "site-1",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "authority_level": "derived_reference_record",
        "storage_class": "jsonl_reference_record",
        "capture_session_id": "session-1",
        "coordinate_frame_session_id": "coord-1",
        "pass_id": "pass-1",
        "pass_index": 1,
        "chunk_id": "chunk-001",
        "chunk_order": 1,
        "frame_id": "000001",
        "frame_index": 1,
        "t_capture_sec": 1.0,
        "T_world_camera": _pose(),
        "T_site_camera": None,
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0},
        "depth_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/arkit/depth/000001.png",
        "confidence_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/arkit/confidence/000001.png",
        "embedding_uri": "gs://bucket/scenes/scene-1/captures/capture-1/world_model_export/embeddings/000001.bin",
        "frame_uri": "gs://bucket/scenes/scene-1/captures/capture-1/world_model_export/frames/000001.jpg",
        "thumbnail_uri": "gs://bucket/sites/site-1/reference_memory/thumbnails/ref-1.jpg",
        "privacy_source": "privacy/final_walkthrough.mov",
        "geometry_source": "arkit",
        "provenance_lineage": {
            "raw_capture_prefix_uri": "gs://bucket/scenes/scene-1/captures/capture-1",
            "capture_descriptor_uri": "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
            "derived_from": ["raw_capture", "capture_descriptor"],
            "geometry_source": "arkit",
        },
        "privacy_lineage": {
            "privacy_source": "privacy/final_walkthrough.mov",
            "privacy_safe_required": True,
            "privacy_status": "privacy_safe_source",
        },
        "rights_lineage": {
            "rights_source_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/rights_consent.json",
            "rights_status": "documented",
            "derived_scene_generation_allowed": True,
            "claim_policy": "do_not_infer_rights_clearance",
        },
        "quality": {"tracking_state": "normal", "sharpness_score": 91.0},
        "retrieval_signals": {"capture_confidence": 0.95, "staticness_score": 0.9},
        "visibility_cells": ["0,0", "0,1"],
        "zone_id": "zone-a",
        "anchor_observations": ["anchor-entry"],
        "captured_at": "2026-05-15T00:00:00+00:00",
        "indexed_at": "2026-05-15T00:00:01+00:00",
    }


def _evaluation_admission() -> dict[str, object]:
    digest = "a" * 64
    return {
        "schema_version": EVALUATION_SITE_ADMISSION_SCHEMA_VERSION,
        "importer_kind": "scaniverse_assisted_import",
        "immutable_source_identity": {
            "site_id": "held-out-site",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "source_bundle_id": "bundle-1",
            "capture_sha256": digest,
            "source_bundle_sha256": digest,
            "manifest_sha256": digest,
        },
        "independent_evidence_verification": {
            "status": "verified",
            "independent_of_importer_and_model_backend": True,
            "verifier_id": "site-admission-verifier",
            "verifier_version": "2.0.0",
            "verification_report_sha256": digest,
            "source_artifact_index_sha256": digest,
            "verified_source_manifest_sha256": digest,
        },
        "rights_privacy_provenance": {
            "consent_active": True,
            "rights_verified": True,
            "privacy_review_passed": True,
            "provenance_verified": True,
            "commercial_sim_evaluation_allowed": True,
            "rights_manifest_sha256": digest,
            "consent_scope_id": "consent-sim-eval-v1",
            "privacy_policy_id": "privacy-v1",
            "provenance_chain_id": "capture-chain-1",
            "commercial_use_scope": ["sim_evaluation", "buyer_delivery"],
        },
        "metric_coordinate_contract": {
            "scale_status": "verified_metric",
            "length_unit": "m",
            "up_axis": "+Z",
            "gravity_m_s2": [0.0, 0.0, -9.81],
            "coordinate_frame_manifest_sha256": digest,
            "world_frame_id": "world-z-up",
            "site_frame_id": "site-held-out-site",
            "capture_frame_id": "capture-1-origin",
            "scale_evidence_sha256": digest,
            "gravity_alignment_sha256": digest,
            "uncertainty": {
                "scale_sigma": 0.001,
                "translation_sigma_m": 0.002,
                "rotation_sigma_deg": 0.1,
            },
        },
        "camera_time_calibration": {
            "intrinsics_calibrated": True,
            "extrinsics_calibrated": True,
            "timestamps_synchronized": True,
            "reprojection_check_passed": True,
            "reprojection_rmse_px": 0.4,
            "maximum_reprojection_rmse_px": 1.0,
            "calibration_manifest_sha256": digest,
            "intrinsics_sha256": digest,
            "extrinsics_sha256": digest,
            "timestamps_sha256": digest,
        },
        "static_robot_evaluation_viewpoints": [
            {
                "viewpoint_id": "vp-1",
                "camera_profile_id": "camera-1",
                "robot_profile_id": "robot-1",
                "source_capture_id": "capture-1",
                "source_frame_id": "frame-1",
                "derived_from_moving_scan": True,
                "status": "calibrated_static_viewpoint",
                "pose_sha256": digest,
                "source_trajectory_sha256": digest,
            }
        ],
        "robot_camera_embodiment": {
            "robot_profile_id": "robot-1",
            "camera_profile_id": "camera-1",
            "embodiment_id": "embodiment-1",
            "robot_profile_sha256": digest,
            "camera_profile_sha256": digest,
            "embodiment_manifest_sha256": digest,
        },
        "task_scene_grounding": {
            "scene_identity": "scene-1",
            "task_objects": [{"object_id": "door"}],
            "articulated_parts": [{"part_id": "door-hinge"}],
            "target_zones": [{"zone_id": "open-angle"}],
            "grounding_manifest_sha256": digest,
        },
        "task_contracts": [
            {
                "task_id": "open-door",
                "criterion_id": "door-angle",
                "evidence_type": "articulation_state",
                "tolerance": 0.2,
                "tolerance_unit": "radian",
                "evaluator_mapping": "isaac.articulation_transition.v1",
            }
        ],
        "task_contract_manifest_sha256": digest,
        "truth_layers": {
            "visual_geometry": {"status": "verified", "evidence_sha256": digest},
            "collision": {"status": "verified", "evidence_sha256": digest},
            "contact": {"status": "verified", "evidence_sha256": digest},
            "dynamics": {"status": "verified", "evidence_sha256": digest},
        },
        "deduplication": {
            "status": "passed",
            "site_dedup_id": "site-dedup-1",
            "task_dedup_id": "task-dedup-1",
            "trajectory_dedup_id": "trajectory-dedup-1",
            "dedup_report_sha256": digest,
        },
        "frozen_splits": {
            "locked_before_evaluation": True,
            "split_manifest_sha256": digest,
            "train_sites": ["train-site"],
            "dev_sites": ["dev-site"],
            "held_out_sites": ["held-out-site"],
        },
        "ood_abstention": {
            "abstention_enabled": True,
            "out_of_distribution_behavior": "abstain",
            "calibration_manifest_sha256": digest,
            "axes": [{"axis": "site"}, {"axis": "task"}, {"axis": "embodiment"}],
        },
    }


def test_evaluation_site_admission_derives_readiness_across_all_truth_layers() -> None:
    result = validate_evaluation_site_admission(_evaluation_admission())

    assert result["status"] == "evaluation_ready"
    assert result["scaniverse_assisted_import"] is True
    assert result["claim_boundary"]["assisted_import_is_not_evaluation_readiness"] is True


def test_legacy_site_admission_schema_cannot_enter_v2_contract() -> None:
    candidate = _evaluation_admission()
    candidate["schema_version"] = "evaluation_site_admission.v1"

    result = validate_evaluation_site_admission(candidate)

    assert result["status"] == "blocked"
    assert "site_admission_schema_missing_or_unsupported" in result["blockers"]


@pytest.mark.parametrize(
    ("mutation", "expected_blocker"),
    [
        (
            lambda value: value["metric_coordinate_contract"].update(
                {"scale_status": "review_only"}
            ),
            "metric_scale_not_verified_in_meters",
        ),
        (
            lambda value: value["camera_time_calibration"].update({"reprojection_rmse_px": 4.0}),
            "camera_reprojection_error_missing_or_above_limit",
        ),
        (
            lambda value: value["camera_time_calibration"].update({"reprojection_rmse_px": -0.1}),
            "camera_reprojection_error_missing_or_above_limit",
        ),
        (
            lambda value: value["truth_layers"]["collision"].update({"status": "review_only"}),
            "collision_truth_not_verified",
        ),
        (
            lambda value: value["frozen_splits"].update({"train_sites": ["held-out-site"]}),
            "site_split_overlap_detected",
        ),
    ],
)
def test_assisted_import_cannot_self_declare_evaluation_readiness(
    mutation,
    expected_blocker: str,
) -> None:
    candidate = _evaluation_admission()
    mutation(candidate)

    result = validate_evaluation_site_admission(candidate)

    assert result["status"] == "blocked"
    assert expected_blocker in result["blockers"]


@pytest.mark.parametrize(
    ("mutation", "expected_blocker"),
    [
        (
            lambda value: value["static_robot_evaluation_viewpoints"].append("corrupt-row"),
            "static_robot_evaluation_viewpoints_payload_invalid",
        ),
        (
            lambda value: value["task_scene_grounding"].update(
                {"scene_identity": "different-scene"}
            ),
            "task_scene_grounding_scene_identity_mismatch",
        ),
        (
            lambda value: value["metric_coordinate_contract"].update(
                {"gravity_m_s2": [0.0, 0.0, 9.81]}
            ),
            "gravity_vector_inconsistent_with_up_axis",
        ),
        (
            lambda value: value["ood_abstention"].update(
                {"out_of_distribution_behavior": "force_decision"}
            ),
            "ood_behavior_must_abstain",
        ),
        (
            lambda value: value["independent_evidence_verification"].update(
                {"verified_source_manifest_sha256": "b" * 64}
            ),
            "independent_verification_source_manifest_digest_mismatch",
        ),
        (
            lambda value: value["independent_evidence_verification"].update(
                {"independent_of_importer_and_model_backend": False}
            ),
            "site_evidence_verifier_independence_not_proven",
        ),
        (
            lambda value: value["static_robot_evaluation_viewpoints"].append(
                dict(value["static_robot_evaluation_viewpoints"][0])
            ),
            "static_viewpoint_duplicate_identity:1",
        ),
        (
            lambda value: value["task_scene_grounding"]["task_objects"].append(
                {"object_id": "door"}
            ),
            "task_scene_grounding_duplicate_identity:task_objects",
        ),
    ],
)
def test_site_admission_v2_rejects_malformed_or_contradictory_evidence(
    mutation,
    expected_blocker: str,
) -> None:
    candidate = _evaluation_admission()
    mutation(candidate)

    result = validate_evaluation_site_admission(candidate)

    assert result["status"] == "blocked"
    assert expected_blocker in result["blockers"]


def test_site_reference_record_contract_requires_lineage_and_camera_fields() -> None:
    validate_site_reference_record(_record())

    invalid = dict(_record())
    invalid.pop("rights_lineage")

    with pytest.raises(SiteReferenceContractError, match="rights_lineage"):
        validate_site_reference_record(invalid)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("authority_level", "raw_capture", "authority_level_invalid"),
        ("storage_class", "blob", "storage_class_invalid"),
        ("intrinsics", {}, "intrinsics_missing"),
        ("privacy_lineage", "not-a-mapping", "privacy_lineage_invalid"),
        ("T_world_camera", [], "T_world_camera_invalid"),
        (
            "T_world_camera",
            [[1.0, 0.0, 0.0, 0.0], [0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "T_world_camera_invalid",
        ),
    ],
)
def test_site_reference_record_rejects_invalid_required_shapes(
    field: str,
    value: object,
    message: str,
) -> None:
    invalid = dict(_record())
    invalid[field] = value

    with pytest.raises(SiteReferenceContractError, match=message):
        validate_site_reference_record(invalid)


def test_site_reference_manifest_contract_is_canonical_v1() -> None:
    payload = build_site_reference_manifest_payload(
        site_id="site-1",
        total_reference_frames=3,
        capture_count=1,
        chunk_count=2,
        captures=[
            {
                "capture_id": "capture-1",
                "scene_id": "scene-1",
                "captured_at": "2026-05-15T00:00:00+00:00",
                "frame_count": 3,
                "chunk_count": 2,
                "coordinate_frame_session_id": "coord-1",
                "site_frame_aligned": False,
                "path_length_m": 1.2,
            }
        ],
        coverage_summary={"coverage_fraction": 0.5},
        artifact_uris={
            "site_reference_index_uri": "gs://bucket/sites/site-1/reference_memory/site_reference_index.jsonl"
        },
        readiness={
            "state": "degraded",
            "blockers": ["site_frame_not_established"],
            "operational_launch_ready": False,
        },
        site_frame_established=False,
        last_updated="2026-05-15T00:00:02+00:00",
    )

    assert payload["schema_version"] == SITE_REFERENCE_DATABASE_SCHEMA_VERSION
    validate_site_reference_manifest(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "wrong", "schema_version_invalid"),
        ("authority_level", "raw", "authority_level_invalid"),
        ("storage_class", "firestore", "storage_class_invalid"),
        ("artifact_uris", [], "artifact_uris_invalid"),
        ("readiness", [], "readiness_invalid"),
    ],
)
def test_site_reference_manifest_rejects_invalid_required_shapes(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = build_site_reference_manifest_payload(
        site_id="site-1",
        total_reference_frames=1,
        capture_count=1,
        chunk_count=1,
        captures=[],
        coverage_summary={},
        artifact_uris={"site_reference_index_uri": "gs://bucket/index.jsonl"},
        readiness={"state": "ready", "blockers": []},
        site_frame_established=True,
    )
    payload[field] = value

    with pytest.raises(SiteReferenceContractError, match=message):
        validate_site_reference_manifest(payload)

    missing = dict(payload)
    missing.pop("site_id")
    with pytest.raises(SiteReferenceContractError, match="missing_fields:site_id"):
        validate_site_reference_manifest(missing)


def test_webapp_summary_projection_allows_family_uris_but_rejects_dense_record_fields(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path
    site_root = storage_root / "bucket" / "sites" / "site-1" / "reference_memory"
    site_root.mkdir(parents=True)
    site_index_path = site_root / "site_reference_index.jsonl"
    site_index_path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    (site_root / "retrieval_validation.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "site_id": "site-1",
                "reference_frame_count": 1,
                "chunk_count": 1,
                "geometry_fingerprint_coverage": 1.0,
                "mean_staticness_score": 0.9,
                "aligned_fraction": 0.0,
            }
        ),
        encoding="utf-8",
    )
    (site_root / "site_reference_manifest.json").write_text(
        json.dumps(
            build_site_reference_manifest_payload(
                site_id="site-1",
                total_reference_frames=1,
                capture_count=1,
                chunk_count=1,
                captures=[],
                coverage_summary={"coverage_fraction": 0.25},
                artifact_uris={
                    "site_reference_index_uri": "gs://bucket/sites/site-1/reference_memory/site_reference_index.jsonl"
                },
                readiness={"state": "degraded", "blockers": ["site_frame_not_established"]},
                site_frame_established=False,
            )
        ),
        encoding="utf-8",
    )

    projection = build_site_reference_summary_projection(
        site_id="site-1",
        site_root=site_root,
        site_index_path=site_index_path,
        storage_root=storage_root,
    )

    assert projection["storage_class"] == "firestore_summary_safe"
    assert projection["artifact_uris"]["site_reference_index_uri"].endswith(
        "site_reference_index.jsonl"
    )
    assert "depth_uri" not in json.dumps(projection)
    assert "embedding_uri" not in json.dumps(projection)

    projection["depth_uri"] = "gs://bucket/dense/depth.png"
    with pytest.raises(SiteReferenceContractError, match="dense_fields"):
        assert_summary_projection_safe(projection)


def test_webapp_summary_projection_rejects_wrong_schema_and_storage_class() -> None:
    with pytest.raises(SiteReferenceContractError, match="schema_version_invalid"):
        assert_summary_projection_safe({"schema_version": "wrong"})

    with pytest.raises(SiteReferenceContractError, match="storage_class_invalid"):
        assert_summary_projection_safe(
            {
                "schema_version": WEBAPP_PROJECTION_SCHEMA_VERSION,
                "storage_class": "dense_record",
            }
        )


def test_summary_projection_readiness_states_and_path_fallbacks(tmp_path: Path) -> None:
    site_root = tmp_path / "bucket" / "sites" / "site-1" / "reference_memory"
    site_root.mkdir(parents=True)
    site_index_path = site_root / "site_reference_index.jsonl"
    site_index_path.write_text("", encoding="utf-8")

    ready = build_site_reference_summary_projection(
        site_id="site-1",
        site_root=site_root,
        site_index_path=site_index_path,
        storage_root=tmp_path,
        manifest_payload={
            "total_reference_frames": 5,
            "capture_count": 1,
            "chunk_count": 1,
            "coverage_summary": {},
            "site_frame_established": True,
        },
        validation_payload={"geometry_fingerprint_coverage": 1.0},
    )
    assert ready["readiness"]["state"] == "ready"
    assert ready["artifact_uris"]["site_reference_manifest_uri"].startswith("gs://bucket/")

    blocked = build_site_reference_summary_projection(
        site_id="site-1",
        site_root=site_root,
        site_index_path=site_index_path,
        storage_root=tmp_path / "bucket",
        manifest_payload={
            "total_reference_frames": 0,
            "capture_count": 0,
            "chunk_count": 0,
            "coverage_summary": {},
            "site_frame_established": False,
        },
        validation_payload={"geometry_fingerprint_coverage": "not-a-number"},
    )
    assert blocked["readiness"]["state"] == "blocked"
    assert (
        blocked["artifact_uris"]["site_reference_manifest_uri"]
        == "gs://sites/site-1/reference_memory/site_reference_manifest.json"
    )
    assert "no_reference_frames" in blocked["blockers"]
    assert "no_captures_indexed" in blocked["blockers"]

    degraded = build_site_reference_summary_projection(
        site_id="site-1",
        site_root=site_root,
        site_index_path=site_index_path,
        storage_root=tmp_path,
        manifest_payload={
            "total_reference_frames": 1,
            "capture_count": 1,
            "chunk_count": 1,
            "coverage_summary": {},
            "site_frame_established": True,
        },
        validation_payload={"geometry_fingerprint_coverage": 0.25},
    )
    assert degraded["readiness"]["state"] == "degraded"
    assert degraded["blockers"] == ["low_geometry_fingerprint_coverage"]

    assert _path_to_gs_uri(tmp_path / "outside.json", storage_root=site_root) == str(
        tmp_path / "outside.json"
    )
    assert _path_to_gs_uri(tmp_path / "bucket", storage_root=tmp_path) is None
    (site_root / "site_reference_manifest.json").write_text("{bad-json", encoding="utf-8")
    assert _read_optional_json(site_root / "site_reference_manifest.json") == {}


def test_lineage_preserves_unknown_rights_without_inventing_clearance() -> None:
    lineage = build_reference_record_lineage(
        capture_prefix_uri="gs://bucket/scenes/scene-1/captures/capture-1",
        descriptor_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        geometry_source="video_to_world",
        privacy_source="privacy/final_walkthrough.mov",
        descriptor={"metadata": {}},
    )

    assert lineage["rights_lineage"]["rights_status"] == "unknown"
    assert lineage["rights_lineage"]["derived_scene_generation_allowed"] is None
    assert lineage["rights_lineage"]["claim_policy"] == "do_not_infer_rights_clearance"


@pytest.mark.parametrize(
    ("rights_value", "expected"),
    [("allowed", True), ("blocked", False)],
)
def test_lineage_normalizes_string_rights_flags(rights_value: str, expected: bool) -> None:
    lineage = build_reference_record_lineage(
        capture_prefix_uri=None,
        descriptor_uri=None,
        geometry_source="arkit",
        privacy_source="raw/walkthrough.mov",
        descriptor={
            "capture_rights": {
                "rights_status": "documented",
                "derived_generation_allowed": rights_value,
            },
        },
    )

    assert lineage["rights_lineage"]["derived_scene_generation_allowed"] is expected
    assert lineage["privacy_lineage"]["privacy_status"] == "raw_or_unknown_source"
