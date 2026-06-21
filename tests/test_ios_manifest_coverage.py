from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import ios_manifest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
