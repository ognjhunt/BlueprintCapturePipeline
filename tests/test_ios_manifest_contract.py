"""Contract tests for iOS manifest helpers."""

from pathlib import Path

from blueprint_pipeline.ios_manifest import IOSManifest, load_object_index, resolve_object_index_uri


def test_ios_manifest_from_dict_preserves_bridge_fields() -> None:
    manifest = IOSManifest.from_dict(
        {
            "scene_id": "scene_001",
            "video_uri": "walkthrough.mov",
            "capture_schema_version": "2.0.0",
            "capture_source": "iphone",
            "capture_tier_hint": "tier1_iphone",
            "object_point_cloud_index": "arkit/objects/index.json",
        }
    )

    assert manifest.capture_schema_version == "2.0.0"
    assert manifest.capture_source == "iphone"
    assert manifest.capture_tier_hint == "tier1_iphone"
    assert manifest.object_point_cloud_index == "arkit/objects/index.json"


def test_resolve_object_index_uri_joins_relative_path() -> None:
    manifest = IOSManifest.from_dict({"object_point_cloud_index": "arkit/objects/index.json"})
    uri = resolve_object_index_uri(
        "gs://bucket/scenes/scene_001/iphone/capture_001/raw",
        manifest,
    )
    assert uri == "gs://bucket/scenes/scene_001/iphone/capture_001/raw/arkit/objects/index.json"


def test_load_object_index_supports_list_and_objects_payload(tmp_path: Path) -> None:
    root = tmp_path

    list_path = root / "bucket/scenes/scene_1/raw/arkit/objects/index.json"
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text('[{"id":"a"}]', encoding="utf-8")

    dict_path = root / "bucket/scenes/scene_2/raw/arkit/objects/index.json"
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    dict_path.write_text('{"objects":[{"id":"b"}]}', encoding="utf-8")

    list_result = load_object_index(
        "gs://bucket/scenes/scene_1/raw/arkit/objects/index.json",
        gcs_root=root,
    )
    dict_result = load_object_index(
        "gs://bucket/scenes/scene_2/raw/arkit/objects/index.json",
        gcs_root=root,
    )

    assert list_result[0]["id"] == "a"
    assert dict_result[0]["id"] == "b"
