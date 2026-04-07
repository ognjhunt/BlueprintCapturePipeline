from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.object_index_stage import (
    _copy_crop,
    _existing_index_is_reusable,
    _extract_keyframe_images,
    _ffprobe_duration_seconds,
)
from blueprint_pipeline.object_index_stage import _Keyframe


def test_existing_index_is_not_reused_when_empty_and_runtime_was_missing() -> None:
    reusable = _existing_index_is_reusable(
        loaded=[],
        report={
            "status": "built",
            "object_count": 0,
            "empty_index_cause": "runtime_missing",
            "runtime_preflight": {
                "backends": {
                    "yolo_world": {
                        "support_level": "required",
                        "status": "runtime_missing",
                    }
                }
            },
        },
    )

    assert reusable is False


def test_existing_index_is_reused_when_zero_objects_were_a_real_result() -> None:
    reusable = _existing_index_is_reusable(
        loaded=[],
        report={
            "status": "built",
            "object_count": 0,
            "empty_index_cause": "zero_detections",
            "runtime_preflight": {
                "backends": {
                    "yolo_world": {
                        "support_level": "required",
                        "status": "configured",
                    }
                }
            },
        },
    )

    assert reusable is True


def test_ffprobe_duration_returns_zero_when_binary_is_missing(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "walkthrough.mov"
    video_path.write_bytes(b"not-a-real-video")

    def _missing_ffprobe(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr("blueprint_pipeline.object_index_stage.subprocess.run", _missing_ffprobe)

    assert _ffprobe_duration_seconds(video_path) == 0.0


def test_extract_keyframe_images_writes_placeholder_when_ffmpeg_is_missing(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "walkthrough.mov"
    video_path.write_bytes(b"not-a-real-video")
    frame_path = tmp_path / "frames" / "frame_000000.png"
    keyframe = _Keyframe(
        frame_index=0,
        timestamp=0.0,
        image_width=1920,
        image_height=1080,
        image_path=frame_path,
        intrinsics=[],
        camera_translation=[0.0, 0.0, 0.0],
        motion_score=0.0,
    )

    def _missing_ffmpeg(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr("blueprint_pipeline.object_index_stage.subprocess.run", _missing_ffmpeg)

    _extract_keyframe_images(video_path, [keyframe])

    assert frame_path.is_file()


def test_copy_crop_falls_back_to_source_frame_when_ffmpeg_is_missing(monkeypatch, tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    crop_path = tmp_path / "crops" / "crop.png"
    frame_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\xf8\x0f\x00\x01\x04\x01\x00"
        b"\x18\xdd\x8d\xb1\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    def _missing_ffmpeg(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr("blueprint_pipeline.object_index_stage.subprocess.run", _missing_ffmpeg)

    _copy_crop(frame_path, crop_path, [0, 0, 1, 1])

    assert crop_path.is_file()
    assert crop_path.read_bytes() == frame_path.read_bytes()
