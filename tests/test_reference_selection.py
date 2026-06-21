from __future__ import annotations

import pytest

from blueprint_pipeline.synthesis import reference_selection as ref
from blueprint_pipeline.synthesis.reference_selection import (
    build_legacy_reference_selection_manifest,
    build_reference_selection_comparison,
    build_reference_selection_manifest,
    select_references_for_target,
)


def _pose(tx: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_reference_selection_prefers_anchor_rich_decoupled_reference() -> None:
    records = [
        {
            "reference_id": "target",
            "frame_id": "frame_0001",
            "frame_index": 1,
            "frame_uri": "/tmp/target.jpg",
            "t_capture_sec": 10.0,
            "T_world_camera": _pose(0.0),
            "anchor_observations": [],
            "retrieval_signals": {"anchor_observation_count": 0, "route_anchor_density": 0.0, "capture_confidence": 0.8},
        },
        {
            "reference_id": "near-duplicate",
            "frame_id": "frame_0002",
            "frame_index": 2,
            "frame_uri": "/tmp/near.jpg",
            "t_capture_sec": 10.1,
            "T_world_camera": _pose(0.02),
            "anchor_observations": [],
            "retrieval_signals": {"anchor_observation_count": 0, "route_anchor_density": 0.0, "capture_confidence": 0.9},
        },
        {
            "reference_id": "anchor-rich",
            "frame_id": "frame_0003",
            "frame_index": 12,
            "frame_uri": "/tmp/rich.jpg",
            "t_capture_sec": 11.6,
            "T_world_camera": _pose(0.45),
            "anchor_observations": ["entry", "checkpoint_a", "checkpoint_b"],
            "retrieval_signals": {
                "anchor_observation_count": 3,
                "route_anchor_density": 1.0,
                "checkpoint_proximity_sec": 0.1,
                "capture_confidence": 0.95,
                "geometry_grounding_quality": 1.0,
            },
        },
        {
            "reference_id": "too-far",
            "frame_id": "frame_0004",
            "frame_index": 25,
            "frame_uri": "/tmp/far.jpg",
            "t_capture_sec": 12.0,
            "T_world_camera": _pose(5.0),
            "anchor_observations": ["entry"],
            "retrieval_signals": {"anchor_observation_count": 1, "route_anchor_density": 0.5, "capture_confidence": 0.9},
        },
    ]

    selection = select_references_for_target(records=records, target_index=0, k=1)

    assert selection["selected_reference_ids"] == ["anchor-rich"]
    assert selection["rejected_counts"]["near_duplicate"] == 1
    assert selection["rejected_counts"]["outside_pose_window"] == 1


def test_reference_selection_manifest_records_decoupling_policy() -> None:
    records = [
        {
            "reference_id": "a",
            "frame_id": "frame_0001",
            "frame_index": 1,
            "frame_uri": "/tmp/a.jpg",
            "t_capture_sec": 0.0,
            "T_world_camera": _pose(0.0),
        },
        {
            "reference_id": "b",
            "frame_id": "frame_0002",
            "frame_index": 2,
            "frame_uri": "/tmp/b.jpg",
            "t_capture_sec": 0.05,
            "T_world_camera": _pose(0.01),
        },
        {
            "reference_id": "c",
            "frame_id": "frame_0003",
            "frame_index": 20,
            "frame_uri": "/tmp/c.jpg",
            "t_capture_sec": 1.2,
            "T_world_camera": _pose(0.7),
            "anchor_observations": ["anchor_entry"],
            "retrieval_signals": {"anchor_observation_count": 1, "route_anchor_density": 0.5, "capture_confidence": 0.9},
        },
    ]

    manifest = build_reference_selection_manifest(
        records=records,
        k=1,
        selection_name="unit_test",
        max_targets=2,
    )

    assert manifest["policy"]["target_reference_decoupling_mode"] == "temporal_gap_with_pose_and_anchor_reranking"
    assert manifest["selected_target_count"] >= 1
    assert manifest["rejected_near_duplicate_count"] >= 1
    first_entry = manifest["entries"][0]
    assert first_entry["selected_reference_frame_ids"] == ["frame_0003"]


def test_reference_selection_comparison_measures_delta_against_legacy() -> None:
    records = [
        {
            "reference_id": "target",
            "frame_id": "frame_0001",
            "frame_index": 1,
            "frame_uri": "/tmp/target.jpg",
            "t_capture_sec": 0.0,
            "T_world_camera": _pose(0.0),
        },
        {
            "reference_id": "legacy-near",
            "frame_id": "frame_0002",
            "frame_index": 2,
            "frame_uri": "/tmp/near.jpg",
            "t_capture_sec": 0.05,
            "T_world_camera": _pose(0.02),
            "retrieval_signals": {"anchor_observation_count": 0, "route_anchor_density": 0.0, "capture_confidence": 0.8},
        },
        {
            "reference_id": "decoupled-rich",
            "frame_id": "frame_0003",
            "frame_index": 20,
            "frame_uri": "/tmp/rich.jpg",
            "t_capture_sec": 1.8,
            "T_world_camera": _pose(0.5),
            "anchor_observations": ["entry", "checkpoint"],
            "retrieval_signals": {
                "anchor_observation_count": 2,
                "route_anchor_density": 1.0,
                "checkpoint_proximity_sec": 0.1,
                "capture_confidence": 0.97,
                "geometry_grounding_quality": 1.0,
            },
        },
    ]

    current = build_reference_selection_manifest(records=records, k=1, selection_name="current")
    legacy = build_legacy_reference_selection_manifest(records=records, k=1, selection_name="legacy")
    comparison = build_reference_selection_comparison(
        current_manifest=current,
        legacy_manifest=legacy,
        selection_name="comparison",
    )

    assert comparison["changed_primary_reference_count"] >= 1
    assert comparison["rejected_near_duplicate_delta"] >= 1
    assert comparison["quality_metrics"]["primary_temporal_gap_sec"]["delta"] > 0
    assert comparison["quality_metrics"]["primary_anchor_observation_count"]["delta"] > 0


def test_reference_selection_rejects_invalid_targets_and_empty_manifests() -> None:
    single_record = [
        {
            "reference_id": "only",
            "frame_id": "frame_0001",
            "frame_index": 1,
            "frame_uri": "/tmp/only.jpg",
            "t_capture_sec": 0.0,
            "T_world_camera": _pose(0.0),
        }
    ]

    with pytest.raises(IndexError, match="target_index out of bounds"):
        select_references_for_target(records=single_record, target_index=10, k=1)

    current_manifest = build_reference_selection_manifest(
        records=single_record,
        k=1,
        selection_name="empty-current",
    )
    legacy_manifest = build_legacy_reference_selection_manifest(
        records=single_record,
        k=1,
        selection_name="empty-legacy",
    )

    assert current_manifest["selected_target_count"] == 0
    assert legacy_manifest["selected_target_count"] == 0
    assert current_manifest["skipped_target_count"] == 1
    assert legacy_manifest["skipped_target_count"] == 1


def test_reference_selection_rejection_reason_edges() -> None:
    records = [
        {
            "reference_id": "target",
            "frame_id": "frame_0010",
            "frame_index": 10,
            "frame_uri": "/tmp/target.jpg",
            "t_capture_sec": 0.0,
            "T_world_camera": _pose(0.0),
        },
        {
            "reference_id": "missing-uri",
            "frame_id": "frame_0011",
            "frame_index": 11,
            "t_capture_sec": 2.0,
            "T_world_camera": _pose(1.0),
        },
        {
            "reference_id": "duplicate-frame-id",
            "frame_id": "frame_0010",
            "frame_index": 30,
            "frame_uri": "/tmp/duplicate-id.jpg",
            "t_capture_sec": 2.0,
            "T_world_camera": _pose(1.0),
        },
        {
            "reference_id": "frame-close",
            "frame_id": "frame_0012",
            "frame_index": 11,
            "frame_uri": "/tmp/frame-close.jpg",
            "t_capture_sec": 0.75,
            "T_world_camera": _pose(1.0),
        },
        {
            "reference_id": "pose-close",
            "frame_id": "frame_0013",
            "frame_index": 20,
            "frame_uri": "/tmp/pose-close.jpg",
            "t_capture_sec": 0.75,
            "T_world_camera": _pose(0.01),
        },
        {
            "reference_id": "time-far",
            "frame_id": "frame_0014",
            "frame_index": 40,
            "frame_uri": "/tmp/time-far.jpg",
            "t_capture_sec": 20.0,
            "T_world_camera": _pose(1.0),
        },
        {
            "reference_id": "valid",
            "frame_id": "frame_0015",
            "frame_index": 50,
            "frame_uri": "/tmp/valid.jpg",
            "t_capture_sec": 2.0,
            "T_world_camera": _pose(1.0),
        },
    ]

    selection = select_references_for_target(records=records, target_index=0, k=1)

    assert selection["selected_reference_ids"] == ["valid"]
    assert selection["rejected_counts"]["missing_frame_uri"] == 1
    assert selection["rejected_counts"]["duplicate_identity"] == 1
    assert selection["rejected_counts"]["near_duplicate"] == 2
    assert selection["rejected_counts"]["outside_temporal_window"] == 1

    legacy = build_legacy_reference_selection_manifest(
        records=records[:2],
        k=1,
        selection_name="legacy-missing-uri",
    )
    assert legacy["aggregate_rejected_counts"]["missing_frame_uri"] == 1

    limited_legacy = build_legacy_reference_selection_manifest(
        records=[records[0], records[6], {**records[6], "reference_id": "valid-2", "frame_id": "frame_0016"}],
        k=1,
        selection_name="legacy-limited",
        max_targets=1,
    )
    assert limited_legacy["selected_target_count"] == 1


def test_reference_selection_comparison_handles_empty_primary_references() -> None:
    current_manifest = {
        "policy": {"policy_version": "current"},
        "selected_target_count": 1,
        "rejected_near_duplicate_count": 0,
        "entries": [
            {
                "target_frame_id": "frame_0001",
                "selected_reference_ids": [],
                "selected_references": [],
            }
        ],
    }
    legacy_manifest = {
        "policy": {"policy_version": "legacy"},
        "selected_target_count": 1,
        "rejected_near_duplicate_count": 0,
        "entries": [
            {
                "target_frame_id": "frame_0001",
                "selected_reference_ids": [],
                "selected_references": [],
            }
        ],
    }

    comparison = build_reference_selection_comparison(
        current_manifest=current_manifest,
        legacy_manifest=legacy_manifest,
        selection_name="empty-primary",
    )

    assert comparison["changed_primary_reference_count"] == 0
    assert comparison["quality_metrics"]["primary_temporal_gap_sec"]["delta"] is None
    assert comparison["quality_metrics"]["primary_pose_distance_m"]["current_avg"] is None


def test_reference_selection_helper_fallbacks() -> None:
    assert ref._temporal_gap_sec(None, {"t_capture_sec": 1.0}) is None
    assert ref._frame_gap(None, {"frame_index": 1}) is None
    assert ref._frame_index({"frame_id": "frame_without_numeric_suffix"}) is None
    assert ref._frame_index({"frame_id": "camera_0042"}) == 42
    assert ref._pose_distance_m(None, {"T_world_camera": _pose(1.0)}) is None
    assert ref._pose_matrix({}) is None
    assert ref._pose_matrix({"T_world_camera": list(range(16))}).shape == (4, 4)
    assert ref._pose_matrix({"T_world_camera": [[1.0, 0.0], [0.0, 1.0]]}) is None
    assert ref._anchor_ids([{"anchorId": "entry"}, {"anchor_id": "exit"}, "entry"]) == ["entry", "exit"]
    assert ref._world_mapping_confidence("mapped") == 1.0
    assert ref._world_mapping_confidence("limited_tracking") == 0.65
    assert ref._world_mapping_confidence("initializing") == 0.5
    assert ref._optional_float(object()) is None
    assert ref._optional_int("not-an-int") is None
