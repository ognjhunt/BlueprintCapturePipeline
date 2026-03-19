from __future__ import annotations

from blueprint_pipeline.synthesis.trajectory_augmentation import build_synthetic_trajectory_manifest


def _pose(tx: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_synthetic_trajectory_manifest_augmentes_dense_local_context() -> None:
    records = [
        {"frame_id": "frame_0001", "frame_index": 1, "frame_uri": "/tmp/1.jpg", "t_capture_sec": 0.0, "T_world_camera": _pose(0.0)},
        {"frame_id": "frame_0002", "frame_index": 2, "frame_uri": "/tmp/2.jpg", "t_capture_sec": 0.4, "T_world_camera": _pose(0.2)},
        {"frame_id": "frame_0003", "frame_index": 3, "frame_uri": "/tmp/3.jpg", "t_capture_sec": 0.8, "T_world_camera": _pose(0.4)},
        {"frame_id": "frame_0004", "frame_index": 4, "frame_uri": "/tmp/4.jpg", "t_capture_sec": 1.2, "T_world_camera": _pose(0.6)},
    ]
    selection_entries = [
        {
            "target_index": 1,
            "target_frame_id": "frame_0002",
            "selected_reference_ids": ["ref-a"],
            "selected_reference_frame_ids": ["frame_0004"],
            "decoupling": {"mode": "temporal_gap_with_pose_and_anchor_reranking"},
        }
    ]

    manifest = build_synthetic_trajectory_manifest(
        records=records,
        selection_entries=selection_entries,
        augmentation_name="unit_test",
    )

    assert manifest["augmented_target_count"] == 1
    assert manifest["synthetic_waypoint_count"] >= 1
    entry = manifest["entries"][0]
    assert entry["status"] == "augmented"
    assert entry["synthetic_waypoint_count"] >= 1
    assert entry["trajectory_context_id"] == "trajectory-frame_0002"


def test_synthetic_trajectory_manifest_skips_sparse_context_truthfully() -> None:
    records = [
        {"frame_id": "frame_0001", "frame_index": 1, "frame_uri": "/tmp/1.jpg", "t_capture_sec": 0.0, "T_world_camera": _pose(0.0)},
        {"frame_id": "frame_0002", "frame_index": 2, "frame_uri": "/tmp/2.jpg", "t_capture_sec": 3.0, "T_world_camera": _pose(2.0)},
        {"frame_id": "frame_0003", "frame_index": 3, "frame_uri": "/tmp/3.jpg", "t_capture_sec": 6.5, "T_world_camera": _pose(4.5)},
    ]
    selection_entries = [
        {
            "target_index": 1,
            "target_frame_id": "frame_0002",
            "selected_reference_ids": ["ref-a"],
            "selected_reference_frame_ids": ["frame_0003"],
            "decoupling": {"mode": "temporal_gap_with_pose_and_anchor_reranking"},
        }
    ]

    manifest = build_synthetic_trajectory_manifest(
        records=records,
        selection_entries=selection_entries,
        augmentation_name="unit_test",
    )

    assert manifest["augmented_target_count"] == 0
    assert manifest["skipped_sparse_context_count"] == 1
    entry = manifest["entries"][0]
    assert entry["status"] == "skipped"
    assert entry["reason"] == "insufficient_context_density"
    assert entry["synthetic_waypoint_count"] == 0
