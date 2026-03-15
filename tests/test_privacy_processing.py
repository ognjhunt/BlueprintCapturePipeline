from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.privacy_processing import run_privacy_postprocess


def _write_video(path: Path, payload: bytes = b"fake-video") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_privacy_postprocess_passthrough_when_no_people(monkeypatch, tmp_path: Path) -> None:
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

    result = run_privacy_postprocess(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="cap-1",
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        raw_video_path=raw_video,
    )

    assert result["status"] == "no_people_detected"
    assert result["world_model_video_uri"] == result["privacy_processed_video_uri"]
    assert (capture_root / "privacy" / "final_walkthrough.mov").is_file()
    manifest = json.loads((pipeline_dir / "privacy_processing_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "no_people_detected"


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

    def _vip(**kwargs):
        output = kwargs["output_video"]
        _write_video(output, b"vip-video")
        return {"status": "succeeded", "output_video": str(output)}

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
    assert (capture_root / "privacy" / "final_walkthrough.mov").is_file()
