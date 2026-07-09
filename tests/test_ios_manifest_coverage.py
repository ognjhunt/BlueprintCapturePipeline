from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from blueprint_pipeline import ios_manifest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _bundle_hash(artifacts: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}:{artifacts[path]}" for path in sorted(artifacts))
    return sha256(canonical.encode("utf-8")).hexdigest()


def test_ios_manifest_parses_defaults_and_resolves_uris(tmp_path: Path) -> None:
    raw_manifest = {
        "scene_id": "site-1",
        "video_uri": "gs://bucket/scenes/site-1/raw/walkthrough.mov",
        "fps_source": "",
        "width": "",
        "height": "",
        "capture_start_epoch_ms": "",
        "has_lidar": True,
        "scale_hint_m_per_unit": "",
        "intended_space_type": "",
        "exposure_samples": [{"iso": 100}, "bad"],
        "object_index_uri": " objects/index.json ",
        "object_point_cloud_index": "fallback/index.json",
        "object_point_cloud_count": "",
        "capture_schema_version": " v2 ",
        "capture_source": " IOS ",
        "capture_tier_hint": " beta ",
        "site_extent": {
            "approx_floor_area_m2": "2500.5",
            "ceiling_height_m": "8.2",
            "floor_count": "2",
            "dominant_aisle_width_m": "3.1",
            "site_scale_class": "multi_zone",
            "site_levels": [
                {"level_id": "floor_1", "label": "main floor"},
                {"level_id": "mezzanine", "label": "mezzanine"},
            ],
            "coverage_by_level": [
                {"level_id": "floor_1", "coverage_status": "captured_primary_route"},
                {"level_id": "mezzanine", "coverage_status": "not_captured_restricted"},
            ],
            "vertical_structure_notes": ["mezzanine above packout station"],
            "status": "capturer_declared_review_required",
            "source": "capturer_declared",
        },
        "site_operating_conditions": {
            "lighting_class": "dim_mixed_led",
            "floor_surface": "sealed_concrete",
            "thermal_zone": "cold_storage",
        },
    }
    manifest = ios_manifest.IOSManifest.from_json(json.dumps(raw_manifest))

    assert manifest.device_model == "iPhone"
    assert manifest.os_version == "unknown"
    assert manifest.fps_source == 30.0
    assert manifest.width == 1920
    assert manifest.height == 1080
    assert manifest.capture_start_epoch_ms == 0
    assert manifest.scale_hint_m_per_unit == 1.0
    assert manifest.intended_space_type == "unknown"
    assert manifest.exposure_samples == [{"iso": 100}]
    assert manifest.object_index_uri == "objects/index.json"
    assert manifest.capture_source == "ios"
    assert manifest.approx_floor_area_m2 == 2500.5
    assert manifest.ceiling_height_m == 8.2
    assert manifest.floor_count == 2
    assert manifest.dominant_aisle_width_m == 3.1
    assert manifest.site_scale_class == "multi_zone"
    assert manifest.site_extent_status == "capturer_declared_review_required"
    assert manifest.site_extent_source == "capturer_declared"
    assert manifest.site_levels == [
        {"level_id": "floor_1", "label": "main floor"},
        {"level_id": "mezzanine", "label": "mezzanine"},
    ]
    assert manifest.coverage_by_level == [
        {"level_id": "floor_1", "coverage_status": "captured_primary_route"},
        {"level_id": "mezzanine", "coverage_status": "not_captured_restricted"},
    ]
    assert manifest.vertical_structure_notes == ["mezzanine above packout station"]
    assert manifest.site_operating_conditions == {
        "lighting_class": "dim_mixed_led",
        "floor_surface": "sealed_concrete",
        "thermal_zone": "cold_storage",
    }

    gcs_root = tmp_path / "gcs"
    manifest_path = gcs_root / "bucket" / "raw" / "manifest.json"
    _write_json(manifest_path, {"scene_id": "loaded", "video_uri": "gs://bucket/raw.mov"})
    loaded = ios_manifest.load_ios_manifest_from_uri("gs://bucket/raw/manifest.json", gcs_root=gcs_root)
    assert loaded.scene_id == "loaded"

    assert ios_manifest.resolve_object_index_uri("gs://bucket/raw", manifest) == "gs://bucket/raw/objects/index.json"
    assert ios_manifest.resolve_object_index_uri(
        "gs://bucket/raw",
        {"object_index_uri": "", "object_point_cloud_index": "gs://other/index.json"},
    ) == "gs://other/index.json"
    assert ios_manifest.resolve_object_index_uri("gs://bucket/raw", {}) is None
    assert ios_manifest.object_index_path(tmp_path / "raw", manifest) == tmp_path / "raw" / "fallback/index.json"
    assert ios_manifest.object_index_path(tmp_path / "raw", ios_manifest.IOSManifest.from_dict({})) is None


def test_load_raw_manifest_and_object_index_payload_shapes(tmp_path: Path) -> None:
    gcs_root = tmp_path / "gcs"
    _write_json(gcs_root / "bucket" / "raw" / "manifest.json", {"scene_id": "site", "video_uri": "gs://bucket/raw.mov"})
    assert ios_manifest.load_raw_manifest("gs://bucket/raw", gcs_root=gcs_root).scene_id == "site"

    index_dir = gcs_root / "bucket" / "raw" / "objects"
    _write_json(
        index_dir / "index.json",
        [
            {
                "object_id": "drawer",
                "reference_crop": "crops/drawer.jpg",
                "all_crops": ["", "gs://bucket/absolute.jpg", "crops/a.jpg", str(tmp_path / "already_abs.jpg")],
            },
            "bad",
        ],
    )
    entries = ios_manifest.load_object_index("gs://bucket/raw/objects/index.json", gcs_root=gcs_root)
    assert entries[0]["reference_crop"] == str((index_dir / "crops/drawer.jpg").resolve())
    assert entries[0]["all_crops"] == [
        "gs://bucket/absolute.jpg",
        str((index_dir / "crops/a.jpg").resolve()),
        str(tmp_path / "already_abs.jpg"),
    ]

    for field in ["objects", "items", "summaries"]:
        _write_json(index_dir / f"{field}.json", {field: [{"object_id": field}]})
        assert ios_manifest.load_object_index(f"gs://bucket/raw/objects/{field}.json", gcs_root=gcs_root) == [
            {"object_id": field}
        ]

    _write_json(index_dir / "unsupported_mapping.json", {"not_objects": []})
    with pytest.raises(ValueError, match="Unsupported object index payload"):
        ios_manifest.load_object_index("gs://bucket/raw/objects/unsupported_mapping.json", gcs_root=gcs_root)

    _write_json(index_dir / "unsupported_scalar.json", "bad")
    with pytest.raises(ValueError, match="Unsupported object index payload"):
        ios_manifest.load_object_index("gs://bucket/raw/objects/unsupported_scalar.json", gcs_root=gcs_root)


def test_v3_raw_manifest_verifies_hashes_and_fails_mismatch(tmp_path: Path) -> None:
    gcs_root = tmp_path / "gcs"
    raw = gcs_root / "bucket" / "raw"
    _write_json(raw / "manifest.json", {"schema_version": "v3", "scene_id": "site"})
    (raw / "walkthrough.mov").write_bytes(b"video")
    _write_json(raw / "capture_upload_complete.json", {"status": "complete"})
    artifacts = {
        "capture_upload_complete.json": _sha_file(raw / "capture_upload_complete.json"),
        "manifest.json": _sha_file(raw / "manifest.json"),
        "walkthrough.mov": _sha_file(raw / "walkthrough.mov"),
    }
    _write_json(
        raw / "hashes.json",
        {
            "schema_version": "v1",
            "bundle_sha256": _bundle_hash(artifacts),
            "artifacts": artifacts,
        },
    )

    report = ios_manifest.verify_raw_bundle_hashes("gs://bucket/raw", gcs_root=gcs_root)
    assert report["valid"] is True
    assert report["bundle_sha256_matches"] is True
    assert ios_manifest.load_raw_manifest("gs://bucket/raw", gcs_root=gcs_root).capture_schema_version == "v3"

    (raw / "walkthrough.mov").write_bytes(b"corrupted")
    mismatch = ios_manifest.verify_raw_bundle_hashes_path(raw)
    assert mismatch["valid"] is False
    assert "hash_mismatch:walkthrough.mov" in mismatch["errors"]
    assert "bundle_sha256_mismatch" in mismatch["errors"]
    with pytest.raises(ValueError, match="hash_mismatch:walkthrough.mov"):
        ios_manifest.load_raw_manifest("gs://bucket/raw", gcs_root=gcs_root)

    _write_json(raw / "hashes.json", {"schema_version": "v1", "bundle_sha256": "abc", "artifacts": {}})
    missing_coverage = ios_manifest.verify_raw_bundle_hashes_path(raw)
    assert "hash_coverage_missing:manifest.json" in missing_coverage["errors"]
