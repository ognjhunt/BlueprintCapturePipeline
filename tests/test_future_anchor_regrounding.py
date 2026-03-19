from __future__ import annotations

from blueprint_pipeline.synthesis.future_anchor_regrounding import build_future_anchor_regrounding_manifest


def test_future_anchor_regrounding_uses_real_future_reference_support() -> None:
    records = [
        {
            "reference_id": "target",
            "frame_id": "frame_0001",
            "t_capture_sec": 0.0,
            "anchor_observations": [],
            "retrieval_signals": {"checkpoint_proximity_sec": 0.4},
        },
        {
            "reference_id": "future",
            "frame_id": "frame_0002",
            "t_capture_sec": 1.2,
            "anchor_observations": ["anchor_entry", "checkpoint_pick"],
            "retrieval_signals": {"checkpoint_proximity_sec": 0.1},
        },
    ]
    selection_entries = [
        {
            "target_frame_id": "frame_0001",
            "selected_reference_ids": ["future"],
            "selected_reference_frame_ids": ["frame_0002"],
            "decoupling": {"mode": "temporal_gap_with_pose_and_anchor_reranking"},
        }
    ]
    task_anchor_manifest = {"tasks": [{"task_id": "task-1", "target_object_ids": ["obj-1"]}]}
    protected_regions_manifest = {"grounding_status": "grounded"}

    manifest = build_future_anchor_regrounding_manifest(
        records=records,
        selection_entries=selection_entries,
        task_anchor_manifest=task_anchor_manifest,
        protected_regions_manifest=protected_regions_manifest,
        regrounding_name="unit_test",
    )

    assert manifest["re_grounded_target_count"] == 1
    entry = manifest["entries"][0]
    assert entry["status"] == "re_grounded"
    assert entry["future_anchor_frame_ids"] == ["frame_0002"]


def test_future_anchor_regrounding_skips_when_protected_regions_are_ungrounded() -> None:
    records = [
        {
            "reference_id": "target",
            "frame_id": "frame_0001",
            "t_capture_sec": 0.0,
            "anchor_observations": [],
            "retrieval_signals": {"checkpoint_proximity_sec": 0.4},
        },
        {
            "reference_id": "future",
            "frame_id": "frame_0002",
            "t_capture_sec": 1.2,
            "anchor_observations": ["anchor_entry", "checkpoint_pick"],
            "retrieval_signals": {"checkpoint_proximity_sec": 0.1},
        },
    ]
    selection_entries = [
        {
            "target_frame_id": "frame_0001",
            "selected_reference_ids": ["future"],
            "selected_reference_frame_ids": ["frame_0002"],
            "decoupling": {"mode": "temporal_gap_with_pose_and_anchor_reranking"},
        }
    ]

    manifest = build_future_anchor_regrounding_manifest(
        records=records,
        selection_entries=selection_entries,
        task_anchor_manifest={"tasks": [{"task_id": "task-1", "target_object_ids": ["obj-1"]}]},
        protected_regions_manifest={"grounding_status": "ungrounded"},
        regrounding_name="unit_test",
    )

    assert manifest["re_grounded_target_count"] == 0
    entry = manifest["entries"][0]
    assert entry["status"] == "skipped"
    assert entry["reason"] == "protected_regions_ungrounded"
