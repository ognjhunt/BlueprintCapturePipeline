from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from typing import Iterator

import pytest

import blueprint_pipeline.privacy_runner_service as service
from blueprint_pipeline.privacy_runner_service import _Handler
from blueprint_pipeline.privacy_service_runtime import execute_privacy_service_request
from http.server import ThreadingHTTPServer


pytestmark = pytest.mark.slow


def _write_file(path: Path, payload: bytes = b"data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_sam3_service_materializes_gcs_input_and_uploads_masks(monkeypatch, tmp_path: Path) -> None:
    gcs_root = tmp_path
    bucket_root = gcs_root / "bucket"
    input_video = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "raw" / "walkthrough.mov"
    _write_file(input_video, b"video")

    monkeypatch.setenv("GCS_ROOT", str(gcs_root))

    def _fake_run_sam3_backend(**kwargs):
        mask_path = kwargs["masks_dir"] / "frame_000000.png"
        _write_file(mask_path, b"mask")
        return {
            "status": "succeeded",
            "people_detected": True,
            "people_count": 1,
            "mask_paths": [str(mask_path)],
        }

    monkeypatch.setattr(
        "blueprint_pipeline.privacy_service_runtime._run_sam3_backend",
        _fake_run_sam3_backend,
    )

    result = execute_privacy_service_request(
        "sam3",
        {
            "input_video_uri": "gs://bucket/scenes/scene-1/captures/cap-1/raw/walkthrough.mov",
            "output_json_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_sam3_detection.json",
            "masks_prefix_uri": "gs://bucket/scenes/scene-1/captures/cap-1/privacy/masks/sam3_initial",
            "prompt": "person",
            "stage_name": "initial_detection",
        },
    )

    assert result["status"] == "succeeded"
    assert result["people_detected"] is True
    uploaded_mask = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "privacy" / "masks" / "sam3_initial" / "frame_000000.png"
    assert uploaded_mask.is_file()
    uploaded_json = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "pipeline" / "privacy_sam3_detection.json"
    assert uploaded_json.is_file()
    payload = json.loads(uploaded_json.read_text(encoding="utf-8"))
    assert payload["status"] == "succeeded"
    assert payload["people_count"] == 1


def test_vip_service_prefers_arkit_depth_and_uploads_video(monkeypatch, tmp_path: Path) -> None:
    gcs_root = tmp_path
    bucket_root = gcs_root / "bucket"
    input_video = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "raw" / "walkthrough.mov"
    mask_dir = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "privacy" / "masks" / "sam3_initial"
    depth_dir = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "raw" / "arkit" / "depth"
    confidence_dir = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "raw" / "arkit" / "confidence"
    _write_file(input_video, b"video")
    _write_file(mask_dir / "frame_000000.png", b"mask")
    _write_file(depth_dir / "depth_000000.png", b"depth")
    _write_file(confidence_dir / "confidence_000000.png", b"confidence")

    monkeypatch.setenv("GCS_ROOT", str(gcs_root))

    def _fake_run_vip_backend(**kwargs):
        assert kwargs["arkit_depth_dir"] == depth_dir
        assert kwargs["arkit_confidence_dir"] == confidence_dir
        _write_file(kwargs["output_video"], b"vip-video")
        return {
            "status": "succeeded",
            "depth_source": "arkit",
            "output_video": str(kwargs["output_video"]),
        }

    monkeypatch.setattr(
        "blueprint_pipeline.privacy_service_runtime._run_vip_backend",
        _fake_run_vip_backend,
    )

    result = execute_privacy_service_request(
        "vip",
        {
            "input_video_uri": "gs://bucket/scenes/scene-1/captures/cap-1/raw/walkthrough.mov",
            "masks_prefix_uri": "gs://bucket/scenes/scene-1/captures/cap-1/privacy/masks/sam3_initial",
            "arkit_depth_prefix_uri": "gs://bucket/scenes/scene-1/captures/cap-1/raw/arkit/depth",
            "arkit_confidence_prefix_uri": "gs://bucket/scenes/scene-1/captures/cap-1/raw/arkit/confidence",
            "preferred_depth_source": "arkit",
            "output_video_uri": "gs://bucket/scenes/scene-1/captures/cap-1/privacy/intermediate_vip_walkthrough.mov",
            "output_json_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_vip_result.json",
        },
    )

    assert result["status"] == "succeeded"
    assert result["depth_source"] == "arkit"
    uploaded_video = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "privacy" / "intermediate_vip_walkthrough.mov"
    assert uploaded_video.is_file()
    uploaded_json = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "pipeline" / "privacy_vip_result.json"
    assert uploaded_json.is_file()


def test_vip_service_generates_depth_anything_artifacts_without_masks(monkeypatch, tmp_path: Path) -> None:
    gcs_root = tmp_path
    bucket_root = gcs_root / "bucket"
    input_video = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "raw" / "walkthrough.mov"
    _write_file(input_video, b"video")

    monkeypatch.setenv("GCS_ROOT", str(gcs_root))

    def _fake_run_depth_anything_backend(**kwargs):
        depth_path = kwargs["depth_dir"] / "depth_000000.npy"
        confidence_path = kwargs["confidence_dir"] / "confidence_000000.npy"
        _write_file(depth_path, b"depth")
        _write_file(confidence_path, b"confidence")
        return {
            "status": "succeeded",
            "runner_kind": "depth_anything",
            "provider": "depth_anything_3",
            "model_name": "da3metric-large",
            "frame_count": 1,
            "depth_artifacts": [
                {
                    "frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "path": str(depth_path),
                    "relative_path": depth_path.name,
                    "format": "npy",
                    "width": 16,
                    "height": 16,
                    "min_depth_m": 0.5,
                    "max_depth_m": 2.0,
                }
            ],
            "confidence_artifacts": [
                {
                    "frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "path": str(confidence_path),
                    "relative_path": confidence_path.name,
                    "format": "npy",
                    "width": 16,
                    "height": 16,
                    "value_range": [0.0, 1.0],
                }
            ],
            "warnings": [],
        }

    monkeypatch.setattr(
        "blueprint_pipeline.privacy_service_runtime._run_depth_anything_backend",
        _fake_run_depth_anything_backend,
    )

    result = execute_privacy_service_request(
        "vip",
        {
            "input_video_uri": "gs://bucket/scenes/scene-1/captures/cap-1/raw/walkthrough.mov",
            "depth_generation_only": True,
            "depth_output_prefix_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/depth",
            "confidence_output_prefix_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/confidence",
            "output_depth_manifest_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/depth_manifest.json",
            "output_confidence_manifest_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/confidence_manifest.json",
            "output_json_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth_generation.json",
        },
    )

    assert result["status"] == "succeeded"
    assert result["depth_source"] == "depth_anything"
    assert (
        bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "pipeline" / "privacy_depth" / "depth_manifest.json"
    ).is_file()
    assert (
        bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "pipeline" / "privacy_depth" / "confidence_manifest.json"
    ).is_file()


def test_vip_service_reuses_precomputed_depth_manifests(monkeypatch, tmp_path: Path) -> None:
    gcs_root = tmp_path
    bucket_root = gcs_root / "bucket"
    input_video = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "raw" / "walkthrough.mov"
    mask_dir = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "privacy" / "masks" / "sam3_initial"
    depth_dir = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "pipeline" / "privacy_depth" / "depth"
    confidence_dir = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "pipeline" / "privacy_depth" / "confidence"
    _write_file(input_video, b"video")
    _write_file(mask_dir / "frame_000000.png", b"mask")
    _write_file(depth_dir / "depth_000000.npy", b"depth")
    _write_file(confidence_dir / "confidence_000000.npy", b"confidence")
    (bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "pipeline" / "privacy_depth").mkdir(
        parents=True, exist_ok=True
    )
    (bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "pipeline" / "privacy_depth" / "depth_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "artifacts": [
                    {
                        "frame_index": 0,
                        "relative_path": "depth_000000.npy",
                        "uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/depth/depth_000000.npy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "pipeline" / "privacy_depth" / "confidence_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "artifacts": [
                    {
                        "frame_index": 0,
                        "relative_path": "confidence_000000.npy",
                        "uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/confidence/confidence_000000.npy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GCS_ROOT", str(gcs_root))

    def _fake_run_vip_backend(**kwargs):
        assert kwargs["arkit_depth_dir"] is None
        assert kwargs["precomputed_depth_frames"]
        assert kwargs["precomputed_confidence_frames"]
        _write_file(kwargs["output_video"], b"vip-video")
        return {
            "status": "succeeded",
            "depth_source": "depth_anything",
            "used_precomputed_depth": True,
            "output_video": str(kwargs["output_video"]),
        }

    monkeypatch.setattr(
        "blueprint_pipeline.privacy_service_runtime._run_vip_backend",
        _fake_run_vip_backend,
    )

    result = execute_privacy_service_request(
        "vip",
        {
            "input_video_uri": "gs://bucket/scenes/scene-1/captures/cap-1/raw/walkthrough.mov",
            "masks_prefix_uri": "gs://bucket/scenes/scene-1/captures/cap-1/privacy/masks/sam3_initial",
            "depth_manifest_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/depth_manifest.json",
            "confidence_manifest_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/confidence_manifest.json",
            "preferred_depth_source": "depth_anything",
            "output_video_uri": "gs://bucket/scenes/scene-1/captures/cap-1/privacy/intermediate_vip_walkthrough.mov",
            "output_json_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_vip_result.json",
        },
    )

    assert result["status"] == "succeeded"
    assert result["depth_source"] == "depth_anything"
    uploaded_video = (
        bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "privacy" / "intermediate_vip_walkthrough.mov"
    )
    assert uploaded_video.is_file()


def test_deepprivacy2_service_uploads_result_manifest(monkeypatch, tmp_path: Path) -> None:
    gcs_root = tmp_path
    bucket_root = gcs_root / "bucket"
    input_video = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "privacy" / "intermediate_vip_walkthrough.mov"
    _write_file(input_video, b"video")

    monkeypatch.setenv("GCS_ROOT", str(gcs_root))

    def _fake_run_deepprivacy2_backend(**kwargs):
        _write_file(kwargs["output_video"], b"deepprivacy-video")
        return {
            "status": "succeeded",
            "output_video": str(kwargs["output_video"]),
            "face_anonymized_segments": ["0.0-end"],
        }

    monkeypatch.setattr(
        "blueprint_pipeline.privacy_service_runtime._run_deepprivacy2_backend",
        _fake_run_deepprivacy2_backend,
    )

    result = execute_privacy_service_request(
        "deepprivacy2",
        {
            "input_video_uri": "gs://bucket/scenes/scene-1/captures/cap-1/privacy/intermediate_vip_walkthrough.mov",
            "output_video_uri": "gs://bucket/scenes/scene-1/captures/cap-1/privacy/intermediate_deepprivacy2_walkthrough.mov",
            "output_json_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_deepprivacy2_result.json",
        },
    )

    assert result["status"] == "succeeded"
    assert result["face_anonymized_segments"] == ["0.0-end"]
    uploaded_video = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "privacy" / "intermediate_deepprivacy2_walkthrough.mov"
    assert uploaded_video.is_file()
    uploaded_json = bucket_root / "scenes" / "scene-1" / "captures" / "cap-1" / "pipeline" / "privacy_deepprivacy2_result.json"
    assert uploaded_json.is_file()


# ---------------------------------------------------------------------------
# HTTP service surface (ThreadingHTTPServer + _Handler) coverage.
#
# These tests boot the real handler bound to an ephemeral port in a daemon
# thread and drive it with http.client. ``execute_privacy_service_request`` is
# always monkeypatched so NO real privacy backend, GPU, or network call runs.
# ---------------------------------------------------------------------------


@contextmanager
def _running_service() -> Iterator[int]:
    """Start the privacy runner handler on an ephemeral port in a thread."""

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _request(
    port: int,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        raw = response.read()
        status = response.status
    finally:
        conn.close()
    payload = json.loads(raw.decode("utf-8")) if raw else {}
    return status, payload


@pytest.fixture
def _no_token(monkeypatch) -> None:
    monkeypatch.delenv("PRIVACY_RUNNER_TOKEN", raising=False)
    monkeypatch.delenv("PRIVACY_RUNNER_KIND", raising=False)


def _stub_dispatch(monkeypatch, calls: list[tuple[str, dict]], result: dict[str, object]) -> None:
    def _fake(kind: str, body: dict) -> dict[str, object]:
        calls.append((kind, dict(body)))
        return dict(result)

    monkeypatch.setattr(service, "execute_privacy_service_request", _fake)


def test_http_healthz_returns_ok(_no_token, monkeypatch) -> None:
    monkeypatch.setenv("PRIVACY_RUNNER_KIND", "sam3")
    # The dispatcher must never be touched by a health probe.
    monkeypatch.setattr(
        service,
        "execute_privacy_service_request",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("dispatch must not run for GET")),
    )
    with _running_service() as port:
        status, payload = _request(port, "GET", "/healthz")
    assert status == int(HTTPStatus.OK)
    assert payload["status"] == "ok"
    assert payload["runner_kind"] == "sam3"


def test_http_unknown_get_path_returns_404(_no_token) -> None:
    with _running_service() as port:
        status, payload = _request(port, "GET", "/does-not-exist")
    assert status == int(HTTPStatus.NOT_FOUND)
    assert payload["status"] == "failed"
    assert payload["reason"] == "not_found"


def test_http_post_without_token_dispatches(_no_token, monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []
    _stub_dispatch(monkeypatch, calls, {"status": "succeeded", "people_detected": True})
    body = json.dumps({"input_video_uri": "gs://bucket/in.mov"}).encode("utf-8")
    with _running_service() as port:
        status, payload = _request(
            port,
            "POST",
            "/run",
            body=body,
            headers={"Content-Type": "application/json"},
        )
    assert status == int(HTTPStatus.OK)
    assert payload["status"] == "succeeded"
    assert calls and calls[0][1] == {"input_video_uri": "gs://bucket/in.mov"}


def test_http_post_unknown_path_returns_404(_no_token, monkeypatch) -> None:
    monkeypatch.setattr(
        service,
        "execute_privacy_service_request",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("dispatch must not run for unknown path")),
    )
    with _running_service() as port:
        status, payload = _request(port, "POST", "/unknown", body=b"{}")
    assert status == int(HTTPStatus.NOT_FOUND)
    assert payload["reason"] == "not_found"


def test_http_post_missing_authorization_returns_401(monkeypatch) -> None:
    monkeypatch.setenv("PRIVACY_RUNNER_TOKEN", "secret-token")
    monkeypatch.setattr(
        service,
        "execute_privacy_service_request",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("dispatch must not run when unauthorized")),
    )
    with _running_service() as port:
        status, payload = _request(port, "POST", "/run", body=b"{}")
    assert status == int(HTTPStatus.UNAUTHORIZED)
    assert payload["reason"] == "unauthorized"


def test_http_post_wrong_authorization_returns_401(monkeypatch) -> None:
    monkeypatch.setenv("PRIVACY_RUNNER_TOKEN", "secret-token")
    monkeypatch.setattr(
        service,
        "execute_privacy_service_request",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("dispatch must not run when unauthorized")),
    )
    with _running_service() as port:
        status, payload = _request(
            port,
            "POST",
            "/run",
            body=b"{}",
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert status == int(HTTPStatus.UNAUTHORIZED)
    assert payload["reason"] == "unauthorized"


def test_http_post_valid_bearer_token_dispatches(monkeypatch) -> None:
    monkeypatch.setenv("PRIVACY_RUNNER_TOKEN", "secret-token")
    calls: list[tuple[str, dict]] = []
    _stub_dispatch(monkeypatch, calls, {"status": "succeeded"})
    body = json.dumps({"input_video_uri": "gs://bucket/in.mov"}).encode("utf-8")
    with _running_service() as port:
        status, payload = _request(
            port,
            "POST",
            "/run",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer secret-token",
            },
        )
    assert status == int(HTTPStatus.OK)
    assert payload["status"] == "succeeded"
    assert calls


def test_http_post_invalid_json_returns_400(_no_token, monkeypatch) -> None:
    monkeypatch.setattr(
        service,
        "execute_privacy_service_request",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("dispatch must not run for invalid JSON")),
    )
    with _running_service() as port:
        status, payload = _request(port, "POST", "/run", body=b"{not-json")
    assert status == int(HTTPStatus.BAD_REQUEST)
    assert payload["reason"] == "invalid_json"


def test_http_post_non_dict_body_returns_400(_no_token, monkeypatch) -> None:
    monkeypatch.setattr(
        service,
        "execute_privacy_service_request",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("dispatch must not run for non-dict body")),
    )
    body = json.dumps([1, 2, 3]).encode("utf-8")
    with _running_service() as port:
        status, payload = _request(port, "POST", "/run", body=body)
    assert status == int(HTTPStatus.BAD_REQUEST)
    assert payload["reason"] == "invalid_payload"


def test_http_post_backend_non_succeeded_returns_502(_no_token, monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []
    _stub_dispatch(
        monkeypatch,
        calls,
        {"status": "failed", "reason": "sam3_runner_not_configured"},
    )
    body = json.dumps({"input_video_uri": "gs://bucket/in.mov"}).encode("utf-8")
    with _running_service() as port:
        status, payload = _request(port, "POST", "/run", body=body)
    assert status == int(HTTPStatus.BAD_GATEWAY)
    assert payload["status"] == "failed"
    assert payload["reason"] == "sam3_runner_not_configured"
    assert calls
