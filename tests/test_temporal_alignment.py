from __future__ import annotations

import pytest

from blueprint_pipeline.temporal_alignment import (
    align_frame_pose_streams,
    canonical_stream_id,
)


def test_canonical_alignment_is_one_to_one_and_reports_distribution() -> None:
    frames = [
        {"frame_index": 1, "t_device_sec": 0.0},
        {"frame_index": 2, "t_device_sec": 0.1},
        {"frame_index": 3, "t_device_sec": 0.2},
    ]
    poses = [
        {"frame_id": "1", "t_device_sec": 0.01},
        {"frame_id": "2", "t_device_sec": 0.12},
        {"frame_id": "3", "t_device_sec": 0.23},
    ]

    result = align_frame_pose_streams(frames, poses)

    assert result["status"] == "verified"
    assert result["metrics"]["match_rate"] == 1.0
    assert result["metrics"]["delta_p50_sec"] == pytest.approx(0.02)
    assert result["metrics"]["delta_p95_sec"] == pytest.approx(0.029)
    assert result["metrics"]["delta_max_sec"] == pytest.approx(0.03)
    assert {join["pose_row_index"] for join in result["joins"]} == {0, 1, 2}
    assert result["drop_ledger"] == []


@pytest.mark.parametrize(
    "rows, blocker",
    [
        (
            [
                {"frame_id": 1, "t_device_sec": 0.0},
                {"frame_id": 2, "timestamp_ms": 100.0},
            ],
            "frames:mixed_timestamp_units",
        ),
        (
            [
                {"frame_id": 1, "t_device_sec": 0.1},
                {"frame_id": 2, "t_device_sec": 0.0},
            ],
            "frames:timestamps_not_strictly_monotonic",
        ),
        (
            [
                {"frame_id": 1, "t_device_sec": 0.0},
                {"frame_id": 1, "t_device_sec": 0.1},
            ],
            "frames:duplicate_canonical_id:frame-000000000001",
        ),
        (
            [
                {"frame_id": 1, "t_device_sec": 0.0},
                {"frame_id": 2, "t_device_sec": 0.0},
            ],
            "frames:duplicate_timestamp:0.0",
        ),
    ],
)
def test_mixed_duplicate_and_nonmonotonic_streams_block(rows, blocker) -> None:
    result = align_frame_pose_streams(
        rows,
        [{"frame_id": index + 1, "t_device_sec": index * 0.1} for index in range(2)],
    )

    assert result["status"] == "blocked"
    assert blocker in result["blockers"]


def test_id_match_without_valid_delta_does_not_count() -> None:
    result = align_frame_pose_streams(
        [{"frame_id": 1, "t_device_sec": 0.0}],
        [{"frame_id": 1, "t_device_sec": 5.0}],
        max_delta_sec=0.2,
    )

    assert result["metrics"]["matched_count"] == 0
    assert result["metrics"]["match_rate"] == 0.0
    assert result["drop_ledger"][0]["reason"] == "id_match_delta_exceeded"


def test_ambiguous_generic_timestamp_blocks() -> None:
    result = align_frame_pose_streams(
        [{"frame_id": 1, "timestamp": 0.0}],
        [{"frame_id": 1, "timestamp": 0.0}],
    )

    assert result["status"] == "blocked"
    assert "frames:row_0:timestamp_missing_or_ambiguous" in result["blockers"]
    assert "poses:row_0:timestamp_missing_or_ambiguous" in result["blockers"]


def test_canonical_stream_id_normalizes_numeric_ids() -> None:
    assert canonical_stream_id(7) == "frame-000000000007"
    assert canonical_stream_id("000007") == "frame-000000000007"
    assert canonical_stream_id("cameraA:7") == "cameraA:7"
    assert canonical_stream_id("../escape") is None
