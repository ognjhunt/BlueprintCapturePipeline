"""Hermetic tests for the tier-1 rollout-reliability gate (synthetic videos only)."""

from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.wam_rollout_reliability import (
    ACTION_DIM,
    FLAG_BLANK_FRAMES,
    FLAG_MOTION_WITHOUT_COMMAND,
    FLAG_STATIC_UNDER_COMMAND,
    ROT6D_IDENTITY,
    action_energy_series,
    assess_frame_sequence_reliability,
    assess_rollout_reliability,
)

cv2 = pytest.importorskip("cv2")

W, H, N_FRAMES = 128, 96, 17
RNG = np.random.default_rng(20260728)
BACKGROUND = (RNG.integers(40, 216, size=(H, W), dtype=np.uint8)).astype(np.uint8)


def _chunk(translation_x: np.ndarray) -> np.ndarray:
    arr = np.zeros((len(translation_x), ACTION_DIM))
    arr[:, 0] = translation_x
    arr[:, 3:9] = ROT6D_IDENTITY[None, :]
    return arr


ACTIVE_CHUNK = _chunk(np.array([0.02] * 8 + [0.0] * 8))
NULL_CHUNK = _chunk(np.zeros(16))


def _write_video(path, frames) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    writer.release()


def _square_frames(step_px: list[int]) -> list[np.ndarray]:
    frames, x = [], 8
    for dx in [0, *step_px]:
        x += dx
        f = BACKGROUND.copy()
        f[36:60, x : x + 20] = 255
        frames.append(f)
    return frames


def test_moving_square_tracking_commands_is_reliable(tmp_path):
    video = tmp_path / "tracking.mp4"
    _write_video(video, _square_frames([6] * 8 + [0] * 8))
    report = assess_rollout_reliability(video, ACTIVE_CHUNK)
    assert report.reliable, report.flags
    assert report.timing_correlation is not None and report.timing_correlation > 0.5


def test_exact_frame_sequence_tracking_commands_is_reliable(tmp_path):
    paths = []
    for index, frame in enumerate(_square_frames([6] * 8 + [0] * 8)):
        path = tmp_path / f"frame_{index:02d}.png"
        assert cv2.imwrite(str(path), frame)
        paths.append(path)

    report = assess_frame_sequence_reliability(paths, ACTIVE_CHUNK)

    assert report.reliable, report.flags
    assert report.video_path.startswith("frame_sequence:")
    assert report.n_frames == N_FRAMES
    assert report.timing_correlation is not None and report.timing_correlation > 0.5


def test_static_video_under_active_commands_is_flagged(tmp_path):
    video = tmp_path / "static.mp4"
    _write_video(video, [BACKGROUND.copy() for _ in range(N_FRAMES)])
    report = assess_rollout_reliability(video, ACTIVE_CHUNK)
    assert not report.reliable
    assert FLAG_STATIC_UNDER_COMMAND in report.flags


def test_motion_under_null_commands_is_flagged(tmp_path):
    video = tmp_path / "phantom.mp4"
    _write_video(video, _square_frames([6] * 16))
    report = assess_rollout_reliability(video, NULL_CHUNK)
    assert not report.reliable
    assert FLAG_MOTION_WITHOUT_COMMAND in report.flags


def test_blank_video_is_flagged(tmp_path):
    video = tmp_path / "blank.mp4"
    _write_video(video, [np.full((H, W), 128, dtype=np.uint8) for _ in range(N_FRAMES)])
    report = assess_rollout_reliability(video, ACTIVE_CHUNK)
    assert not report.reliable
    assert FLAG_BLANK_FRAMES in report.flags


def test_action_energy_series_rejects_bad_shape():
    with pytest.raises(ValueError):
        action_energy_series(np.zeros((16, 7)))


def test_degenerate_rot6d_zero_chunk_is_flagged(tmp_path):
    video = tmp_path / "any.mp4"
    _write_video(video, _square_frames([6] * 16))
    all_zero = np.zeros((16, ACTION_DIM))
    report = assess_rollout_reliability(video, all_zero)
    assert not report.reliable
    assert "invalid_action_rot6d" in report.flags
