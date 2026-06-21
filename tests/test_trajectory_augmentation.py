from __future__ import annotations

import numpy as np

from blueprint_pipeline.synthesis import trajectory_augmentation as trajectory
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


def test_synthetic_trajectory_manifest_covers_policy_and_helper_edges(
    monkeypatch,
) -> None:
    dense_records = [
        {"frame_id": "frame_0001", "frame_index": 1, "t_capture_sec": 0.0, "T_world_camera": _pose(0.0)},
        {"frame_id": "frame_0002", "frame_index": 2, "t_capture_sec": 0.5, "T_world_camera": _pose(0.25)},
        {"frame_id": "frame_0003", "frame_index": 3, "t_capture_sec": 1.0, "T_world_camera": _pose(0.5)},
        {"frame_id": "frame_0004", "frame_index": 4, "t_capture_sec": 1.5, "T_world_camera": _pose(0.75)},
    ]
    selection = {
        "target_index": 1,
        "target_frame_id": "frame_0002",
        "selected_reference_ids": ["ref-a"],
        "selected_reference_frame_ids": ["frame_0004"],
    }

    assert trajectory.resolve_trajectory_augmentation_policy(
        {"max_synthetic_waypoints_per_target": 1}
    )["max_synthetic_waypoints_per_target"] == 1
    assert build_synthetic_trajectory_manifest(
        records=dense_records,
        selection_entries=[{"target_frame_id": "missing", "target_index": 99}],
        augmentation_name="missing-target",
    )["entries"] == []

    limited = build_synthetic_trajectory_manifest(
        records=dense_records,
        selection_entries=[selection],
        augmentation_name="limited",
        policy={"max_synthetic_waypoints_per_target": 1},
    )
    assert limited["entries"][0]["synthetic_waypoint_count"] == 1

    assert trajectory._find_sorted_index(dense_records, "missing", 2) == 2
    assert trajectory._find_sorted_index(dense_records, "missing", 99) is None
    assert trajectory._has_sparse_gaps(
        [{"frame_id": "a", "t_capture_sec": 0.0}, dense_records[1]],
        max_gap_sec=1.0,
        max_gap_m=1.0,
    ) is True
    assert trajectory._interpolate_pose(np.zeros((3, 3)), np.zeros((4, 4))) is None

    flat_pose = np.array(_pose(0.25), dtype=np.float32).reshape(16).tolist()
    assert trajectory._pose_matrix({"T_world_camera": None}) is None
    assert trajectory._pose_matrix({"T_world_camera": flat_pose}).shape == (4, 4)
    assert trajectory._pose_matrix({"T_world_camera": [1.0, 2.0, 3.0]}) is None
    assert trajectory._pose_distance(None, trajectory._pose_matrix({"T_world_camera": _pose(1.0)})) is None
    assert trajectory._frame_index({"frame_index": "not-int", "frame_id": "frame_0009"}) is None
    assert trajectory._frame_index({"frame_id": "frame_without_digits"}) is None
    assert trajectory._optional_float(None) is None
    assert trajectory._optional_float("not-a-number") is None

    monkeypatch.setattr(trajectory, "_has_sparse_gaps", lambda *_args, **_kwargs: False)
    missing_pose = trajectory._augment_target(
        sorted_records=[dense_records[0], {"frame_id": "missing-pose", "t_capture_sec": 0.5}, dense_records[2]],
        target_sorted_index=1,
        selection={"target_frame_id": "missing-pose"},
        policy=trajectory.resolve_trajectory_augmentation_policy({}),
    )
    assert missing_pose["status"] == "skipped"
    assert missing_pose["reason"] == "no_augmented_midpoints"

    monkeypatch.setattr(trajectory, "_interpolate_pose", lambda *_args, **_kwargs: None)
    no_midpoint = trajectory._augment_target(
        sorted_records=dense_records[:3],
        target_sorted_index=1,
        selection=selection,
        policy=trajectory.resolve_trajectory_augmentation_policy({}),
    )
    assert no_midpoint["status"] == "skipped"
    assert no_midpoint["reason"] == "no_augmented_midpoints"
