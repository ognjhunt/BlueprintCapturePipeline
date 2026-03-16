from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.privacy_service_runtime import execute_privacy_service_request


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
