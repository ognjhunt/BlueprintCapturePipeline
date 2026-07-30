from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import jsonschema
import numpy as np
import pytest

import blueprint_pipeline.capture_quality_analyzer as analyzer
from blueprint_pipeline.capture_intake import CaptureIntakeError


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _textured_frames(count: int = 6) -> list[np.ndarray]:
    rows, columns = np.indices((360, 640))
    checker = ((rows // 24 + columns // 24) % 2).astype(np.uint8)
    gray = np.where(checker == 0, 40, 210).astype(np.uint8)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return [np.roll(base, shift=index * 3, axis=1) for index in range(count)]


def test_measure_sampled_frames_emits_bounded_deterministic_quality() -> None:
    first = analyzer.measure_sampled_frames(_textured_frames())
    second = analyzer.measure_sampled_frames(_textured_frames())

    assert first == second
    assert first["sample_count"] == 6
    measurements = first["measurements"]
    for key in (
        "sharp_frame_fraction",
        "well_exposed_frame_fraction",
        "visual_overlap_fraction",
        "rolling_shutter_symptom_fraction",
    ):
        assert 0.0 <= measurements[key] <= 1.0
    assert measurements["sharp_frame_fraction"] == 1.0
    assert measurements["well_exposed_frame_fraction"] == 1.0
    assert measurements["visual_overlap_fraction"] >= 0.8
    assert measurements["rolling_shutter_symptom_fraction"] == 0.0
    assert measurements["median_interframe_motion_pixels"] > 0.0


def test_dark_textureless_frames_fail_quality_without_inventing_overlap() -> None:
    frames = [np.full((360, 640, 3), 5, dtype=np.uint8) for _ in range(4)]

    result = analyzer.measure_sampled_frames(frames)

    assert result["measurements"]["sharp_frame_fraction"] == 0.0
    assert result["measurements"]["well_exposed_frame_fraction"] == 0.0
    assert "visual_overlap_fraction" not in result["measurements"]
    assert "rolling_shutter_symptom_fraction" not in result["measurements"]


def test_analyzer_binds_packet_to_exact_source_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"video-bytes"
    video = tmp_path / "capture.mp4"
    video.write_bytes(payload)
    monkeypatch.setattr(
        analyzer,
        "_sample_video",
        lambda _path, *, sample_count: _textured_frames(min(sample_count, 6)),
    )
    monkeypatch.setattr(analyzer, "_compression_sufficiency", lambda _path: (1.0, 0.1))

    with pytest.raises(CaptureIntakeError, match="source_digest_mismatch"):
        analyzer.analyze_capture_video_quality(
            video,
            intake_id="intake-1",
            source_file_sha256=_digest(b"other"),
        )

    packet = analyzer.analyze_capture_video_quality(
        video,
        intake_id="intake-1",
        source_file_sha256=_digest(payload),
        sample_count=6,
    )

    assert packet["source"] == "local_analyzer"
    assert packet["source_file_sha256"] == _digest(payload)
    assert packet["analyzer"]["analyzer_id"] == "blueprint_local_frame_quality.v1"
    assert packet["measurements"]["compression_quality_fraction"] == 1.0
    assert packet["observations_digest"].startswith("sha256:")
    assert "absence_of_privacy_sensitive_content_is_not_certified" in packet["limitations"]
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs"
            / "schemas"
            / "capture_quality_observations.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(packet)


@pytest.mark.parametrize(
    ("bitrate_kbps", "expected"),
    [(2100.0, 1.0), (150.0, 0.0)],
)
def test_compression_proxy_uses_bits_per_pixel_per_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bitrate_kbps: float,
    expected: float,
) -> None:
    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            return {
                cv2.CAP_PROP_BITRATE: bitrate_kbps,
                cv2.CAP_PROP_FPS: 15.0,
                cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
            }[property_id]

        def release(self) -> None:
            return None

    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: FakeCapture())

    fraction, bpp = analyzer._compression_sufficiency(tmp_path / "video.mp4")

    assert fraction == expected
    assert bpp == round(bitrate_kbps * 1000.0 / (15.0 * 1280.0 * 720.0), 6)
