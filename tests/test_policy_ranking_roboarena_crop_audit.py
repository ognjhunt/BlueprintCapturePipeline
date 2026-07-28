from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from blueprint_pipeline.policy_ranking_roboarena_calibration import canonical_sha256, file_sha256
from blueprint_pipeline.policy_ranking_roboarena_crop_audit import (
    EXPECTED_ROLLOUT_README_SHA256,
    _audit_one,
)


def _paired_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 480))
    assert writer.isOpened()
    for index in range(40):
        frame = np.zeros((480, 1280, 3), dtype=np.uint8)
        frame[:, :640] = (index + 10, 0, 220)
        frame[:, 640:] = (0, 255, index)
        writer.write(frame)
    writer.release()


def test_audit_materializes_only_generated_left_pixels(tmp_path: Path) -> None:
    source = tmp_path / "session" / "policy" / "left" / "compare_overlay_vs_gt.mp4"
    source.parent.mkdir(parents=True)
    _paired_video(source)
    request_identity = {
        "session_id": "session",
        "policy_id_internal_only": "policy",
        "task_instruction": "test",
        "relative_path": source.relative_to(tmp_path).as_posix(),
        "video_sha256": file_sha256(source),
        "evaluator_digest": "e" * 64,
    }
    request = {
        **request_identity,
        "request_id": canonical_sha256(request_identity),
    }
    output = tmp_path / "output"
    row = _audit_one(request, rollout_root=tmp_path, output_root=output)
    assert row["generated_crop_xyxy"] == [0, 0, 640, 480]
    assert row["physical_right_half_pixels_encoded"] is False
    assert row["sampled_frame_count"] == 32
    assert row["unique_sampled_frame_count"] == 32
    assert row["repeated_sample_count"] == 0
    for frame in row["sampled_frames"]:
        payload = (output / frame["relative_output_path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == frame["encoded_jpeg_sha256"]
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (480, 640, 3)
        assert float(image[:, :, 1].mean()) < 15.0


def test_rollout_readme_identity_is_frozen() -> None:
    assert EXPECTED_ROLLOUT_README_SHA256 == (
        "f94076393ecbfa0b9373241a701b068e76a4fc5d8cab542cda13de31f313b34e"
    )
