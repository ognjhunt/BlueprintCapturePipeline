from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from blueprint_pipeline.oscar_review_views import build_oscar_review_views
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256, file_sha256


def _video(path: Path, frames: list[np.ndarray], fps: float = 6.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def _read(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def test_review_views_are_attributable_and_mask_only_conditioned_region(tmp_path: Path) -> None:
    generated = tmp_path / "generated.mp4"
    skeleton = tmp_path / "skeleton.mp4"
    generated_frames = [np.full((32, 48, 3), 180, dtype=np.uint8) for _ in range(5)]
    skeleton_frames = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(7)]
    for frame in skeleton_frames:
        frame[14:18, 10:38] = (0, 255, 255)
    _video(generated, generated_frames)
    _video(skeleton, skeleton_frames)

    report = build_oscar_review_views(
        generated_video=generated,
        skeleton_video=skeleton,
        expected_generated_sha256=file_sha256(generated),
        expected_skeleton_sha256=file_sha256(skeleton),
        output_dir=tmp_path / "out",
        start_frame=2,
        frame_count=5,
        width=48,
        height=32,
        fps=6.0,
        dilation_radius_pixels=2,
    )

    assert report["status"] == "completed"
    assert report["attribution"]["all_media_in_this_report_are_review_derivatives"] is True
    assert report["attribution"]["visible_skeleton_review_is_not_native_oscar_output"] is True
    assert report["evidence_boundaries"] == {
        "physical_future_rgb_pixels_used": False,
        "physical_outcome_labels_accessed": False,
        "provider_called": False,
        "paid_resource_used": False,
        "ranking_credit": False,
        "wam_qualification_credit": False,
        "physical_success_credit": False,
        "thesis_support_credit": False,
    }
    payload = {key: value for key, value in report.items() if key != "report_sha256"}
    assert report["report_sha256"] == canonical_sha256(payload)

    masked = _read(tmp_path / "out" / "skeleton_region_occluded_scene.mp4")
    assert len(masked) == 5
    assert masked[0][16, 24].max() < 20
    assert masked[0][0, 0].min() > 150


def test_review_views_fail_closed_on_frame_or_fps_mismatch(tmp_path: Path) -> None:
    generated = tmp_path / "generated.mp4"
    skeleton = tmp_path / "skeleton.mp4"
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)]
    _video(generated, frames, fps=5.0)
    _video(skeleton, frames, fps=5.0)

    try:
        build_oscar_review_views(
            generated_video=generated,
            skeleton_video=skeleton,
            expected_generated_sha256=file_sha256(generated),
            expected_skeleton_sha256=file_sha256(skeleton),
            output_dir=tmp_path / "out",
            start_frame=0,
            frame_count=4,
            width=16,
            height=16,
            fps=5.0,
        )
    except ValueError as error:
        assert str(error).startswith("unexpected_frame_count:")
    else:
        raise AssertionError("expected frame-count mismatch")

    try:
        build_oscar_review_views(
            generated_video=generated,
            skeleton_video=skeleton,
            expected_generated_sha256=file_sha256(generated),
            expected_skeleton_sha256=file_sha256(skeleton),
            output_dir=tmp_path / "out2",
            start_frame=0,
            frame_count=3,
            width=16,
            height=16,
            fps=6.0,
        )
    except ValueError as error:
        assert str(error).startswith("fps_contract_mismatch:")
    else:
        raise AssertionError("expected fps mismatch")
