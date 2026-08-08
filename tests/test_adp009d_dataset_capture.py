"""Contract tests for the 15 Hz per-camera dataset capture recorder.

The policy-input PNGs at query cadence stay the authoritative record of what a
policy consumed.  This recorder is the lab-facing stream: one H.264 video per
DROID camera at the true control rate, one frame per environment step, with
per-frame raw digests so the lossy review stream stays audit-linked to the
exact rendered pixels.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from blueprint_pipeline.adp009d_dataset_capture import (
    DATASET_CAPTURE_SCHEMA_VERSION,
    DROID_CONTROL_FPS,
    DatasetCaptureError,
    DatasetCaptureRecorder,
    droid_stream_id_for_view,
)


def _frame(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.integers(0, 255, size=(32, 64, 3), dtype=np.uint8)


VIEWS = (
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
)


def _record_episode(tmp_path, *, steps: int = 6) -> tuple[DatasetCaptureRecorder, dict]:
    recorder = DatasetCaptureRecorder(
        output_dir=tmp_path,
        episode_id="episode-000",
        view_keys=VIEWS,
    )
    for step in range(steps):
        recorder.record_step(
            step_index=step,
            views={view: _frame(step * 10 + offset) for offset, view in enumerate(VIEWS)},
        )
    record = recorder.finalize(
        terminal_views={view: _frame(999 + offset) for offset, view in enumerate(VIEWS)}
    )
    return recorder, record


def test_droid_stream_ids_drop_observation_prefix() -> None:
    assert droid_stream_id_for_view("observation/exterior_image_1_left") == (
        "exterior_image_1_left"
    )
    assert droid_stream_id_for_view("observation/wrist_image_left") == (
        "wrist_image_left"
    )
    with pytest.raises(DatasetCaptureError):
        droid_stream_id_for_view("not_a_droid_view")


def test_recorder_writes_one_control_rate_video_per_camera(tmp_path) -> None:
    _, record = _record_episode(tmp_path, steps=6)

    assert record["schema_version"] == DATASET_CAPTURE_SCHEMA_VERSION
    assert record["frames_per_second"] == DROID_CONTROL_FPS == 15.0
    assert record["frame_count"] == 6
    assert record["terminal_frame_included"] is True
    assert sorted(record["streams"]) == ["exterior_image_1_left", "wrist_image_left"]
    for stream in record["streams"].values():
        video_path = tmp_path / stream["video"]["relative_path"]
        assert video_path.read_bytes()[4:8] == b"ftyp"
        assert stream["video"]["fourcc"] == "avc1"
        # 6 control-step frames plus the terminal observation.
        assert stream["video"]["decoded_frame_count"] == 7
        assert stream["video"]["decode_round_trip_passed"] is True
        assert stream["width"] == 64
        assert stream["height"] == 32
        assert len(stream["frame_raw_rgb_sha256"]) == 7

    manifest_path = tmp_path / record["manifest_relative_path"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["capture_digest"] == record["capture_digest"]
    assert manifest["frame_alignment"] == (
        "frame_index_i_is_the_observation_before_control_step_i"
    )


def test_recorder_rejects_gaps_shape_changes_and_double_finalize(tmp_path) -> None:
    recorder = DatasetCaptureRecorder(
        output_dir=tmp_path,
        episode_id="episode-001",
        view_keys=VIEWS,
    )
    recorder.record_step(
        step_index=0,
        views={view: _frame(index) for index, view in enumerate(VIEWS)},
    )
    with pytest.raises(DatasetCaptureError):
        recorder.record_step(
            step_index=2,
            views={view: _frame(index) for index, view in enumerate(VIEWS)},
        )
    with pytest.raises(DatasetCaptureError):
        recorder.record_step(
            step_index=1,
            views={VIEWS[0]: _frame(3)},
        )
    with pytest.raises(DatasetCaptureError):
        recorder.record_step(
            step_index=1,
            views={
                VIEWS[0]: _frame(4),
                VIEWS[1]: np.zeros((16, 16, 3), dtype=np.uint8),
            },
        )
    record = recorder.finalize(terminal_views=None)
    assert record["terminal_frame_included"] is False
    assert record["frame_count"] == 1
    with pytest.raises(DatasetCaptureError):
        recorder.finalize(terminal_views=None)
    with pytest.raises(DatasetCaptureError):
        recorder.record_step(
            step_index=1,
            views={view: _frame(index) for index, view in enumerate(VIEWS)},
        )


def test_recorder_rejects_non_uint8_frames(tmp_path) -> None:
    recorder = DatasetCaptureRecorder(
        output_dir=tmp_path,
        episode_id="episode-002",
        view_keys=VIEWS,
    )
    with pytest.raises(DatasetCaptureError):
        recorder.record_step(
            step_index=0,
            views={
                VIEWS[0]: np.zeros((32, 64, 3), dtype=np.float32),
                VIEWS[1]: _frame(1),
            },
        )
