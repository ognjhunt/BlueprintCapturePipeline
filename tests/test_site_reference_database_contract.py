from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.site_reference_database import (
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
        artifact_uris={"site_reference_index_uri": "gs://bucket/sites/site-1/reference_memory/site_reference_index.jsonl"},
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


def test_webapp_summary_projection_allows_family_uris_but_rejects_dense_record_fields(tmp_path: Path) -> None:
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
                artifact_uris={"site_reference_index_uri": "gs://bucket/sites/site-1/reference_memory/site_reference_index.jsonl"},
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
    assert projection["artifact_uris"]["site_reference_index_uri"].endswith("site_reference_index.jsonl")
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
