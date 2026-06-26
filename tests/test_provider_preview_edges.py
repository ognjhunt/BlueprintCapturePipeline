from __future__ import annotations

import builtins
import types
import urllib.error
from pathlib import Path

import pytest

from blueprint_pipeline import provider_preview as pp


class _Response:
    def __init__(self, body: bytes = b"{}", status: int = 200) -> None:
        self._body = body
        self.status = status

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def test_worldlabs_http_upload_and_uri_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLDLABS_API_KEY", "secret-worldlabs-key")

    error = urllib.error.HTTPError(
        "https://api.worldlabs.ai/fail",
        429,
        "rate limited",
        {},
        _Response(b"too many"),
    )
    monkeypatch.setattr(pp._urllib_request, "urlopen", lambda request, timeout: (_ for _ in ()).throw(error))
    with pytest.raises(RuntimeError, match="worldlabs_api_429:too many"):
        pp._worldlabs_api_request("/fail")

    monkeypatch.setattr(pp._urllib_request, "urlopen", lambda request, timeout: _Response(b"[1, 2]"))
    assert pp._worldlabs_api_request("/list") == {}

    with pytest.raises(RuntimeError, match="worldlabs_upload_url_invalid"):
        pp._presigned_upload("ftp://bad", content_type="video/mp4", data=b"data")

    upload_error = urllib.error.HTTPError(
        "https://upload.example",
        403,
        "forbidden",
        {},
        _Response(b"denied"),
    )
    monkeypatch.setattr(
        pp._urllib_request,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(upload_error),
    )
    with pytest.raises(RuntimeError, match="worldlabs_upload_failed:403:denied"):
        pp._presigned_upload("https://upload.example", content_type="video/mp4", data=b"data")

    url_error = urllib.error.URLError("offline")
    monkeypatch.setattr(
        pp._urllib_request,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(url_error),
    )
    with pytest.raises(RuntimeError, match="worldlabs_upload_failed:url_error:offline"):
        pp._presigned_upload("https://upload.example", content_type="video/mp4", data=b"data")

    captured_upload: dict[str, object] = {}

    def upload_success(request: object, timeout: int) -> _Response:
        captured_upload["method"] = request.get_method()  # type: ignore[attr-defined]
        captured_upload["headers"] = dict(request.header_items())  # type: ignore[attr-defined]
        captured_upload["data"] = request.data  # type: ignore[attr-defined]
        return _Response(b"uploaded")

    monkeypatch.setattr(pp._urllib_request, "urlopen", upload_success)
    pp._presigned_upload(
        "https://upload.example",
        method="post",
        content_type="video/mp4",
        data=b"data",
        required_headers={"x-required": "yes"},
    )
    assert captured_upload["method"] == "POST"
    captured_headers = {key.lower(): value for key, value in captured_upload["headers"].items()}  # type: ignore[union-attr]
    assert captured_headers["x-required"] == "yes"
    assert captured_upload["data"] == b"data"

    local_file = tmp_path / "video.mp4"
    local_file.write_bytes(b"local-video")
    assert pp._read_uri_bytes(str(local_file)) == b"local-video"
    monkeypatch.setattr(pp._urllib_request, "urlopen", lambda request, timeout: _Response(b"remote-video"))
    assert pp._read_uri_bytes("https://example.test/video.mp4") == b"remote-video"

    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "google.cloud":
            raise ImportError("no gcs")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="google-cloud-storage is required"):
        pp._read_uri_bytes("gs://bucket/object.mp4", capture_root=Path("/outside-scenes"))

    class FakeBlob:
        def download_as_bytes(self) -> bytes:
            return b"gcs-video"

    class FakeBucket:
        def blob(self, object_path: str) -> FakeBlob:
            assert object_path == "folder/video.mp4"
            return FakeBlob()

    class FakeClient:
        def bucket(self, bucket_name: str) -> FakeBucket:
            assert bucket_name == "bucket"
            return FakeBucket()

    fake_cloud = types.SimpleNamespace(
        storage=types.SimpleNamespace(Client=lambda: FakeClient())
    )

    def fake_import_gcs(name: str, *args: object, **kwargs: object) -> object:
        if name == "google.cloud":
            return fake_cloud
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import_gcs)
    assert pp._read_uri_bytes("gs://bucket/folder/video.mp4") == b"gcs-video"

    class BadUri:
        def rstrip(self, _chars: str = "/") -> "BadUri":
            raise RuntimeError("bad uri")

    assert pp._extension_from_uri(BadUri(), fallback="mp4") == "mp4"  # type: ignore[arg-type]
    assert pp._filename_from_uri(BadUri(), fallback="fallback.mp4") == "fallback.mp4"  # type: ignore[arg-type]
    assert pp._normalize_permission({"public": False}) == {"public": False}
    assert pp._normalize_permission("public")["public"] is True


def test_worldlabs_poll_and_manifest_builder_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = pp.WorldLabsPreviewProvider()
    calls = iter([
        {"status": "processing"},
        {"status": "ready", "launch_url": "https://world.example"},
    ])
    monkeypatch.setattr(provider, "poll", lambda run_id: next(calls))
    monkeypatch.setattr(pp.time, "sleep", lambda seconds: None)
    assert pp._poll_worldlabs_until_terminal(provider=provider, operation_id="op-1")[
        "launch_url"
    ] == "https://world.example"

    monkeypatch.setattr(pp, "_WORLDLABS_POLL_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(provider, "poll", lambda run_id: {"status": "processing"})
    assert pp._poll_worldlabs_until_terminal(provider=provider, operation_id="op-timeout")[
        "failure_reason"
    ] == "polling_timeout_after_1_attempts"

    assert pp.StubPreviewProvider().poll(run_id="stub-1") == {
        "provider_run_id": "stub-1",
        "status": "succeeded",
    }
    assert provider._privacy_processing({"metadata": {"privacy_processing": {"status": "ok"}}}) == {
        "status": "ok"
    }
    candidates = provider._world_prompt_candidates(
        {
            "world_model_video_uri": "https://example.test/world.mp4",
            "privacy_processed_video_uri": "https://example.test/privacy.mp4",
        }
    )
    assert [item["source_id"] for item in candidates["candidates"]] == [
        "world_model_video_uri",
        "privacy_processed_video_uri",
    ]

    monkeypatch.setattr(pp, "production_launch_mode", lambda: True)
    with pytest.raises(RuntimeError, match="production_worldlabs_input_audit_incomplete"):
        provider._build_request_manifest(
            descriptor={
                "metadata": {
                    "worldlabs_input_video_uri": "https://example.test/input.mp4",
                    "worldlabs_input_audit_uri": "audit.json",
                    "worldlabs_input_audit": {
                        "privacy_safe_input": True,
                        "output_video_uri": "https://example.test/input.mp4",
                    },
                }
            },
            capture_root=Path("/tmp/capture"),
        )


def test_worldlabs_upload_and_submit_edge_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = pp.WorldLabsPreviewProvider()
    video = tmp_path / "video.mov"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        pp,
        "_worldlabs_api_request",
        lambda *args, **kwargs: {"media_asset": {"media_asset_id": "asset-1"}, "upload_info": "bad"},
    )
    with pytest.raises(RuntimeError, match="worldlabs_upload_url_missing"):
        provider._upload_video_as_media_asset(str(video), descriptor={}, capture_root=tmp_path)

    monkeypatch.setattr(
        pp,
        "_worldlabs_api_request",
        lambda *args, **kwargs: {"media_asset": {}, "upload_info": {"upload_url": "https://upload"}},
    )
    monkeypatch.setattr(pp, "_read_uri_bytes", lambda uri, capture_root=None: b"video")
    monkeypatch.setattr(pp, "_presigned_upload", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="worldlabs_media_asset_id_missing"):
        provider._upload_video_as_media_asset(str(video), descriptor={}, capture_root=tmp_path)

    no_video = provider.submit(descriptor={"capture_id": "cap-1"}, capture_root=tmp_path)
    assert no_video["failure_reason"] == "no_eligible_video"
    blocked = provider.submit(
        descriptor={"capture_id": "cap-1"},
        capture_root=tmp_path,
        provider_adapter_input={"status": "blocked", "blockers": ["privacy_failed"]},
    )
    assert blocked["failure_reason"] == "provider_adapter_input_blocked:privacy_failed"

    def fake_request_manifest(**kwargs: object) -> dict[str, object]:
        return {
            "provider_model": "marble-test",
            "selected_video_uri": "https://example.test/video.mp4",
            "generation_source_type": "video_uri",
            "generation_request": {"world_prompt": {"video_prompt": {"media_asset_id": "old"}}},
        }

    monkeypatch.setattr(provider, "_build_request_manifest", fake_request_manifest)
    monkeypatch.setattr(
        pp,
        "_worldlabs_api_request",
        lambda *args, **kwargs: {"operation_id": "op-video-uri"},
    )
    submitted = provider.submit(descriptor={"capture_id": "cap-1"}, capture_root=tmp_path)
    assert submitted["status"] == "processing"
    assert submitted["generation_source_type"] == "video_uri"


def test_worldlabs_poll_variants_and_run_preview_provider_edge_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = pp.WorldLabsPreviewProvider()
    responses = {
        "/marble/v1/operations/op-fetch": {
            "done": True,
            "metadata": {"world_id": "world-1"},
        },
        "/marble/v1/worlds/world-1": {
            "world_id": "world-1",
            "world_marble_url": "https://world.example/1",
        },
        "/marble/v1/operations/op-queued": {"done": False, "status": "queued"},
    }
    monkeypatch.setattr(pp, "_worldlabs_api_request", lambda path, **kwargs: responses[path])
    ready = provider.poll(run_id="op-fetch")
    assert ready["status"] == "ready"
    queued = provider.poll(run_id="op-queued")
    assert queued["status"] == "queued"
    with pytest.raises(ValueError, match="unsupported_preview_provider:unknown"):
        pp.resolve_preview_provider("unknown")

    class FakeProvider:
        provider_name = "fake_world"
        provider_model = "fake-v1"

        def submit(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["provider_adapter_input"] == {"input": True}
            return {
                "provider_name": self.provider_name,
                "provider_model": self.provider_model,
                "provider_run_id": "op-1",
                "status": "processing",
                "artifact_uris": {},
                "worldlabs_operation": {"id": "op-1"},
            }

        def normalize(self, payload: dict[str, object]) -> dict[str, object]:
            return dict(payload)

        def emit_preview_manifest(self, *, normalized: dict[str, object], output_path: Path) -> dict[str, object]:
            pp.write_json(output_path, {"status": normalized["status"]})
            return {"status": normalized["status"]}

        def emit_provenance(self, *, descriptor: dict[str, object], normalized: dict[str, object]) -> dict[str, object]:
            return {"canonical": False, "derived": True}

        def poll(self, *, run_id: str) -> dict[str, object]:
            return {}

    monkeypatch.setattr(pp, "resolve_preview_provider", lambda name: FakeProvider())
    result = pp.run_preview_provider(
        provider_name="fake",
        descriptor={"capture_id": "cap-1"},
        capture_root=tmp_path,
        pipeline_dir=tmp_path,
        provider_adapter_input={"input": True},
    )

    assert result["status"] == "processing"
    assert (tmp_path / "worldlabs_operation_manifest.json").is_file()
