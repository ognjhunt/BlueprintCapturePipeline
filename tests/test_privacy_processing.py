from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.privacy_processing import run_privacy_postprocess


def _write_video(path: Path, payload: bytes = b"fake-video") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _depth_anything_result() -> dict[str, object]:
    return {
        "status": "succeeded",
        "source": "depth_anything",
        "provider": "depth_anything_3",
        "model_name": "da3metric-large",
        "depth_prefix_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/depth",
        "confidence_prefix_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/confidence",
        "depth_manifest_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/depth_manifest.json",
        "confidence_manifest_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/confidence_manifest.json",
        "depth_manifest_path": "/tmp/depth_manifest.json",
        "confidence_manifest_path": "/tmp/confidence_manifest.json",
        "frame_count": 12,
    }


def test_privacy_postprocess_non_arkit_passthrough_still_generates_depth(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "cap-1"
    pipeline_dir = capture_root / "pipeline"
    raw_video = capture_root / "raw" / "walkthrough.mov"
    _write_video(raw_video)

    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "true")
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_sam3",
        lambda **_kwargs: {
            "status": "succeeded",
            "people_detected": False,
            "people_count": 0,
            "mask_paths": [],
        },
    )
    depth_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_depth_anything",
        lambda **kwargs: depth_calls.append(kwargs) or _depth_anything_result(),
    )

    result = run_privacy_postprocess(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="cap-1",
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        raw_video_path=raw_video,
    )

    assert result["status"] == "no_people_detected"
    assert result["depth_source"] == "depth_anything"
    assert result["depth_conditioning"]["depth_manifest_uri"].endswith("/pipeline/privacy_depth/depth_manifest.json")
    assert result["world_model_video_uri"] == result["privacy_processed_video_uri"]
    assert depth_calls
    assert (capture_root / "privacy" / "final_walkthrough.mov").is_file()
    manifest = json.loads((pipeline_dir / "privacy_processing_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "no_people_detected"
    assert manifest["depth_source"] == "depth_anything"


def test_privacy_postprocess_uses_anonymized_fallback(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "cap-1"
    pipeline_dir = capture_root / "pipeline"
    raw_video = capture_root / "raw" / "walkthrough.mov"
    _write_video(raw_video)

    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "true")
    sam3_results = iter(
        [
            {
                "status": "succeeded",
                "people_detected": True,
                "people_count": 2,
                "mask_paths": ["mask-1.png"],
            },
            {
                "status": "succeeded",
                "people_detected": True,
                "people_count": 1,
                "mask_paths": ["mask-2.png"],
            },
            {
                "status": "succeeded",
                "people_detected": True,
                "people_count": 1,
                "mask_paths": [],
            },
        ]
    )
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_sam3",
        lambda **_kwargs: next(sam3_results),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_depth_anything",
        lambda **_kwargs: _depth_anything_result(),
    )

    def _vip(**kwargs):
        output = kwargs["output_video"]
        _write_video(output, b"vip-video")
        return {
            "status": "succeeded",
            "output_video": str(output),
            "depth_source": "depth_anything",
        }

    def _deepprivacy(**kwargs):
        output = kwargs["output_video"]
        _write_video(output, b"deepprivacy-video")
        return {
            "status": "succeeded",
            "output_video": str(output),
            "face_anonymized_segments": ["segment-1"],
        }

    monkeypatch.setattr("blueprint_pipeline.privacy_processing._run_vip", _vip)
    monkeypatch.setattr("blueprint_pipeline.privacy_processing._run_deepprivacy2", _deepprivacy)

    result = run_privacy_postprocess(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="cap-1",
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        raw_video_path=raw_video,
    )

    assert result["status"] == "face_anonymized_fallback"
    assert result["fallback_used"] is True
    assert result["face_anonymized_segments"] == ["segment-1"]
    assert result["depth_source"] == "depth_anything"
    assert (capture_root / "privacy" / "final_walkthrough.mov").is_file()


def test_privacy_postprocess_prefers_arkit_depth_for_vip(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "cap-1"
    pipeline_dir = capture_root / "pipeline"
    raw_video = capture_root / "raw" / "walkthrough.mov"
    depth_dir = capture_root / "raw" / "arkit" / "depth"
    confidence_dir = capture_root / "raw" / "arkit" / "confidence"
    _write_video(raw_video)
    depth_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "true")
    sam3_results = iter(
        [
            {"status": "succeeded", "people_detected": True, "people_count": 1, "mask_paths": ["mask-1.png"]},
            {"status": "succeeded", "people_detected": False, "people_count": 0, "mask_paths": []},
        ]
    )
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_sam3",
        lambda **_kwargs: next(sam3_results),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_depth_anything",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("Depth Anything should not run for ARKit captures")),
    )

    def _vip(**kwargs):
        assert kwargs["arkit_depth_prefix_uri"] == "gs://bucket/scenes/scene-1/captures/cap-1/raw/arkit/depth"
        assert kwargs["arkit_confidence_prefix_uri"] == "gs://bucket/scenes/scene-1/captures/cap-1/raw/arkit/confidence"
        assert kwargs["depth_manifest_uri"] is None
        assert kwargs["confidence_manifest_uri"] is None
        output = kwargs["output_video"]
        _write_video(output, b"vip-video")
        return {"status": "succeeded", "output_video": str(output), "depth_source": "arkit"}

    monkeypatch.setattr("blueprint_pipeline.privacy_processing._run_vip", _vip)

    result = run_privacy_postprocess(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="cap-1",
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        raw_video_path=raw_video,
    )

    assert result["status"] == "person_removed"
    assert result["depth_source"] == "arkit"


def test_privacy_postprocess_uses_depth_anything_for_non_arkit_capture(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "cap-1"
    pipeline_dir = capture_root / "pipeline"
    raw_video = capture_root / "raw" / "walkthrough.mov"
    _write_video(raw_video)

    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "true")
    sam3_results = iter(
        [
            {"status": "succeeded", "people_detected": True, "people_count": 1, "mask_paths": ["mask-1.png"]},
            {"status": "succeeded", "people_detected": False, "people_count": 0, "mask_paths": []},
        ]
    )
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_sam3",
        lambda **_kwargs: next(sam3_results),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_depth_anything",
        lambda **_kwargs: _depth_anything_result(),
    )

    def _vip(**kwargs):
        assert kwargs["arkit_depth_prefix_uri"] is None
        assert kwargs["arkit_confidence_prefix_uri"] is None
        assert kwargs["depth_manifest_uri"] == "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/depth_manifest.json"
        assert kwargs["confidence_manifest_uri"] == "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/confidence_manifest.json"
        output = kwargs["output_video"]
        _write_video(output, b"vip-video")
        return {
            "status": "succeeded",
            "output_video": str(output),
            "depth_source": "depth_anything",
        }

    monkeypatch.setattr("blueprint_pipeline.privacy_processing._run_vip", _vip)

    result = run_privacy_postprocess(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="cap-1",
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        raw_video_path=raw_video,
    )

    assert result["status"] == "person_removed"
    assert result["depth_source"] == "depth_anything"
