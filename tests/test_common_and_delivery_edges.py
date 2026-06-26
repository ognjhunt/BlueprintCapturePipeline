from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from blueprint_pipeline import arena_package_delivery_local as delivery
from blueprint_pipeline import common


def test_common_uri_path_json_and_parse_helpers(tmp_path: Path) -> None:
    parsed = common.parse_gs_uri("gs://bucket/scenes/scene-1/file.json")
    assert parsed.bucket == "bucket"
    assert parsed.key == "scenes/scene-1/file.json"
    assert parsed.uri == "gs://bucket/scenes/scene-1/file.json"
    with pytest.raises(ValueError, match="Expected gs://"):
        common.parse_gs_uri("https://bucket/key")
    with pytest.raises(ValueError, match="Invalid gs:// URI"):
        common.parse_gs_uri("gs://bucket")
    assert common.is_gs_uri("gs://bucket/key") is True
    assert common.is_gs_uri("/tmp/key") is False

    flat = tmp_path / "scenes" / "scene-1" / "file.json"
    common.write_json(flat, {"ok": True})
    assert common.resolve_gs_uri_to_path("gs://bucket/scenes/scene-1/file.json", tmp_path) == flat

    bucket_root = tmp_path / "mount"
    bucket_file = bucket_root / "bucket" / "scenes" / "scene-1" / "file.json"
    common.write_json(bucket_file, {"ok": True})
    assert common.resolve_gs_uri_to_path("gs://bucket/scenes/scene-1/file.json", bucket_root) == bucket_file

    named_bucket_root = tmp_path / "bucket"
    nested_stale = named_bucket_root / "bucket" / "scenes" / "scene-1" / "file.json"
    common.write_json(nested_stale, {"stale": True})
    assert common.resolve_gs_uri_to_path(
        "gs://bucket/scenes/scene-1/file.json",
        named_bucket_root,
    ) == named_bucket_root / "scenes" / "scene-1" / "file.json"
    assert common.resolve_gs_uri_to_path(
        "gs://bucket/scenes/missing/file.json",
        named_bucket_root,
    ) == named_bucket_root / "scenes" / "missing" / "file.json"
    assert common.resolve_gs_uri_to_path(
        "gs://other/scenes/scene-2/file.json",
        bucket_root,
    ) == bucket_root / "scenes" / "scene-2" / "file.json"
    (bucket_root / "other").mkdir()
    assert common.resolve_gs_uri_to_path(
        "gs://other/scenes/scene-2/file.json",
        bucket_root,
    ) == bucket_root / "other" / "scenes" / "scene-2" / "file.json"

    local_path = tmp_path / "local.txt"
    local_path.write_text("hello", encoding="utf-8")
    assert common.ensure_local_uri_path(str(local_path), gcs_root=tmp_path) == local_path
    assert common.ensure_local_uri_path(
        "gs://bucket/scenes/scene-1/file.json",
        gcs_root=bucket_root,
    ) == bucket_file

    storage_module = types.ModuleType("google.cloud.storage")

    class FakeBlob:
        def download_to_filename(self, filename: str) -> None:
            Path(filename).write_text("downloaded", encoding="utf-8")

    class FakeBucket:
        def blob(self, _key: str) -> FakeBlob:
            return FakeBlob()

    class FakeClient:
        def bucket(self, _bucket: str) -> FakeBucket:
            return FakeBucket()

    storage_module.Client = FakeClient  # type: ignore[attr-defined]
    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    cloud_module.storage = storage_module  # type: ignore[attr-defined]
    google_module.cloud = cloud_module  # type: ignore[attr-defined]
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_module)
    try:
        downloaded = common.ensure_local_uri_path(
            "gs://download-bucket/path/file.bin",
            gcs_root=tmp_path / "download-root",
            scratch_dir=tmp_path / "scratch",
        )
    finally:
        monkeypatch.undo()
    assert downloaded.read_text(encoding="utf-8") == "downloaded"

    assert common.infer_storage_root_from_scene_path(Path("scenes/scene-1")) == Path("/")
    assert common.infer_storage_root_from_scene_path(tmp_path / "root" / "scenes" / "scene-1") == tmp_path / "root"
    with pytest.raises(ValueError, match="Cannot infer storage root"):
        common.infer_storage_root_from_scene_path(tmp_path / "not-scenes" / "scene-1")

    text_path = tmp_path / "nested" / "note.txt"
    common.write_text(text_path, "note")
    assert text_path.read_text(encoding="utf-8") == "note"
    assert common.has_nonempty_file(text_path) is True
    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("", encoding="utf-8")
    assert common.has_nonempty_file(empty_path) is False

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected JSON object"):
        common.read_json(list_json)
    assert common.read_json(flat) == {"ok": True}
    assert common.read_json_any(list_json) == []
    assert common.optional_read_json(tmp_path / "missing.json") is None
    assert common.optional_read_json(flat) == {"ok": True}

    assert common.to_scene_prefix("scene-1") == "scenes/scene-1"
    assert common.to_capture_prefix("scene-1", "capture-1") == "scenes/scene-1/captures/capture-1"
    assert common.to_pipeline_prefix("scene-1", "capture-1").endswith("/pipeline")
    assert common.join_gs_uri("gs://bucket/prefix/", "/child.json") == "gs://bucket/prefix/child.json"
    assert common.try_parse_float("bad", default=1.5) == 1.5
    assert common.try_parse_int("bad", default=7) == 7
    assert common.relative_scene_path(bucket_file, bucket_root) == "bucket/scenes/scene-1/file.json"
    assert common.maybe_as_dict({"a": 1}) == {"a": 1}
    assert common.maybe_as_dict([("a", 1)]) == {}
    assert common.maybe_as_list([1, 2]) == [1, 2]
    assert common.maybe_as_list("not-list") == []
    assert common.parse_bool(True) is True
    assert common.parse_bool(0) is False
    assert common.parse_bool("yes") is True
    assert common.parse_bool("off") is False
    assert common.parse_bool("unknown", default=True) is True
    assert common.flatten_scene_paths(tmp_path, "a", "b") == (tmp_path / "a", tmp_path / "b")

    error = common.StageError("stage", "failed")
    assert str(error) == "stage: failed"
    assert error.stage == "stage"


def test_arena_package_delivery_local_success_blockers_and_cli(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    output_dir = tmp_path / "arena-package"
    bundle = output_dir / "delivery_bundle"
    nested = bundle / "clips" / "clip.mp4"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"clip")
    delivery_root = tmp_path / "deliveries"

    monkeypatch.setenv(delivery.GATE_ENV, "true")
    manifest = delivery.build_local_delivery_command_manifest(
        output_dir=output_dir,
        delivery_root=delivery_root,
    )
    assert manifest["status"] == "local_delivery_ready_review_required"
    assert manifest["local_access_paths"][0]["relative_path"] == "clips/clip.mp4"
    assert (delivery_root / output_dir.name / "clips" / "clip.mp4").read_bytes() == b"clip"
    assert delivery._relative_or_absolute(output_dir, tmp_path / "outside.txt") == str(
        tmp_path / "outside.txt"
    )

    assert delivery.main(["--output-dir", str(output_dir), "--delivery-root", str(delivery_root)]) == 0
    assert "[arena-local-delivery] status=local_delivery_ready_review_required" in capsys.readouterr().out

    empty_output = tmp_path / "empty-package"
    (empty_output / "delivery_bundle").mkdir(parents=True)
    empty = delivery.build_local_delivery_command_manifest(
        output_dir=empty_output,
        delivery_root=delivery_root,
    )
    assert empty["status"] == "blocked"
    assert "delivery_bundle_empty" in empty["blockers"]

    monkeypatch.delenv(delivery.GATE_ENV, raising=False)
    monkeypatch.delenv(delivery.DELIVERY_ROOT_ENV, raising=False)
    blocked_exit = delivery.main(["--output-dir", str(tmp_path / "missing-package")])
    blocked_output = capsys.readouterr().out
    assert blocked_exit == 1
    assert "[arena-local-delivery] blockers=3" in blocked_output
