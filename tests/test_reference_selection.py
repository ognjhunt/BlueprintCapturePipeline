from __future__ import annotations

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
