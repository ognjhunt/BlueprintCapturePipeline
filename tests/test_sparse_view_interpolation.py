from __future__ import annotations

import numpy as np

from blueprint_pipeline.synthesis import sparse_view_interpolation as sparse
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


def test_sparse_view_interpolation_covers_skip_reasons_and_policy_overrides() -> None:
    records = [
        {"reference_id": "target", "frame_id": "target", "t_capture_sec": 0.0, "T_world_camera": _pose(0.0)},
        {"reference_id": "near", "frame_id": "near", "t_capture_sec": 0.1, "T_world_camera": _pose(0.1)},
        {"reference_id": "far", "frame_id": "far", "t_capture_sec": 12.0, "T_world_camera": _pose(3.0)},
        {"reference_id": "good", "frame_id": "good", "t_capture_sec": 1.0, "T_world_camera": _pose(0.6)},
        {"reference_id": "missing-pose", "frame_id": "missing-pose", "t_capture_sec": 1.0},
    ]

    assert sparse.resolve_sparse_view_interpolation_policy({"max_interpolated_views_per_target": 1})[
        "max_interpolated_views_per_target"
    ] == 1
    assert build_sparse_view_interpolation_manifest(
        records=records,
        selection_entries=[{"target_frame_id": "not-present"}],
        trajectory_entries=[],
        interpolation_name="missing-target",
    )["entries"] == []

    cases = [
        ("trajectory_context_unavailable", "good", {"status": "pending"}, None),
        ("missing_primary_reference", "not-present", {"status": "skipped"}, None),
        ("missing_pose_or_time_support", "missing-pose", {"status": "skipped"}, None),
        ("support_gap_too_small", "near", {"status": "skipped"}, None),
        ("support_gap_too_large", "far", {"status": "skipped"}, None),
        (
            "no_interpolation_fractions",
            "good",
            {"status": "skipped"},
            {"interpolation_fractions": ["bad", None, 0, 1], "max_interpolated_views_per_target": 4},
        ),
    ]
    for expected_reason, reference_frame_id, trajectory_entry, policy in cases:
        manifest = build_sparse_view_interpolation_manifest(
            records=records,
            selection_entries=[
                {
                    "target_frame_id": "target",
                    "selected_reference_frame_ids": [reference_frame_id],
                    "selected_reference_ids": [reference_frame_id],
                }
            ],
            trajectory_entries=[{"target_frame_id": "target", **trajectory_entry}],
            interpolation_name=expected_reason,
            policy=policy,
        )
        assert manifest["entries"][0]["status"] == "skipped"
        assert manifest["entries"][0]["reason"] == expected_reason

    flat_pose = np.array(_pose(0.25), dtype=np.float32).reshape(16).tolist()
    assert sparse._pose_matrix({"T_world_camera": None}) is None
    assert sparse._pose_matrix({"T_world_camera": [1.0, 2.0, 3.0]}) is None
    assert sparse._pose_matrix({"T_world_camera": flat_pose}).shape == (4, 4)
    assert sparse._pose_distance(None, sparse._pose_matrix({"T_world_camera": _pose(1.0)})) is None
    assert sparse._optional_float(None) is None
    assert sparse._optional_float("") is None
    assert sparse._optional_float("not-a-number") is None
