from __future__ import annotations

from blueprint_pipeline.synthesis.sparse_view_interpolation import build_sparse_view_interpolation_manifest


def _pose(tx: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_sparse_view_interpolation_interpolates_when_sparse_context_is_truthful() -> None:
    records = [
        {"reference_id": "target", "frame_id": "frame_0001", "frame_index": 1, "frame_uri": "/tmp/1.jpg", "t_capture_sec": 0.0, "T_world_camera": _pose(0.0)},
        {"reference_id": "ref-a", "frame_id": "frame_0002", "frame_index": 8, "frame_uri": "/tmp/2.jpg", "t_capture_sec": 0.8, "T_world_camera": _pose(0.45)},
    ]
    selection_entries = [
        {
            "target_frame_id": "frame_0001",
            "selected_reference_ids": ["ref-a"],
            "selected_reference_frame_ids": ["frame_0002"],
            "decoupling": {"mode": "temporal_gap_with_pose_and_anchor_reranking"},
        }
    ]
    trajectory_entries = [
        {
            "target_frame_id": "frame_0001",
            "status": "skipped",
            "reason": "insufficient_context_density",
        }
    ]

    manifest = build_sparse_view_interpolation_manifest(
        records=records,
        selection_entries=selection_entries,
        trajectory_entries=trajectory_entries,
        interpolation_name="unit_test",
    )

    assert manifest["interpolated_target_count"] == 1
    assert manifest["interpolated_view_count"] >= 1
    entry = manifest["entries"][0]
    assert entry["status"] == "interpolated"
    assert entry["interpolated_view_count"] >= 1


def test_sparse_view_interpolation_skips_when_local_density_is_already_good() -> None:
    records = [
        {"reference_id": "target", "frame_id": "frame_0001", "frame_index": 1, "frame_uri": "/tmp/1.jpg", "t_capture_sec": 0.0, "T_world_camera": _pose(0.0)},
        {"reference_id": "ref-a", "frame_id": "frame_0002", "frame_index": 8, "frame_uri": "/tmp/2.jpg", "t_capture_sec": 0.8, "T_world_camera": _pose(0.45)},
    ]
    selection_entries = [
        {
            "target_frame_id": "frame_0001",
            "selected_reference_ids": ["ref-a"],
            "selected_reference_frame_ids": ["frame_0002"],
            "decoupling": {"mode": "temporal_gap_with_pose_and_anchor_reranking"},
        }
    ]
    trajectory_entries = [
        {
            "target_frame_id": "frame_0001",
            "status": "augmented",
            "reason": None,
        }
    ]

    manifest = build_sparse_view_interpolation_manifest(
        records=records,
        selection_entries=selection_entries,
        trajectory_entries=trajectory_entries,
        interpolation_name="unit_test",
    )

    assert manifest["interpolated_target_count"] == 0
    entry = manifest["entries"][0]
    assert entry["status"] == "skipped"
    assert entry["reason"] == "local_density_already_sufficient"
