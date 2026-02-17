"""Regression tests for task-target inference and task-aware candidate selection."""

from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.capture_bridge import CaptureDescriptor
from blueprint_pipeline.ios_manifest import IOSManifest
from blueprint_pipeline.task_targets import (
    build_task_aware_swap_candidates_payload,
    infer_task_targets,
)


def _descriptor(*, manip: list[dict] | None = None, artic: list[dict] | None = None) -> CaptureDescriptor:
    return CaptureDescriptor.from_dict(
        {
            "schema_version": "v1",
            "scene_id": "scene_task",
            "capture_id": "cap_task",
            "capture_source": "iphone",
            "capture_tier": "tier1_iphone",
            "raw_prefix_uri": "gs://bucket/scenes/scene_task/iphone/cap_task/raw",
            "frames_index_uri": "gs://bucket/scenes/scene_task/captures/cap_task/frames/index.jsonl",
            "nurec_mode": "mono_pose_assisted",
            "environment_type_hint": "warehouse",
            "manipulation_candidates": manip or [],
            "articulation_hints": artic or [],
        }
    )


def _manifest() -> IOSManifest:
    return IOSManifest.from_dict(
        {
            "scene_id": "scene_task",
            "video_uri": "gs://bucket/scenes/scene_task/iphone/cap_task/raw/video.mp4",
            "device_model": "iPhone",
            "os_version": "17",
            "fps_source": 30.0,
            "width": 1920,
            "height": 1080,
            "capture_start_epoch_ms": 0,
            "has_lidar": False,
            "scale_hint_m_per_unit": 1.0,
            "intended_space_type": "warehouse",
            "object_point_cloud_index": "gs://bucket/scenes/scene_task/captures/cap_task/objects/object_point_cloud_index.json",
            "object_point_cloud_count": 3,
        }
    )


def _ensure_index_file(gcs_root: Path, entries: list[dict]) -> str:
    rel = Path("scenes/scene_task/captures/cap_task/objects/object_point_cloud_index.json")
    index_path = gcs_root / rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({"objects": entries}, indent=2), encoding="utf-8")
    return "gs://bucket/scenes/scene_task/captures/cap_task/objects/object_point_cloud_index.json"


def test_infer_task_targets_heuristic_from_object_index(tmp_path: Path) -> None:
    descriptor = _descriptor()
    manifest = _manifest()
    entries = [
        {
            "id": "door_1",
            "label": "door",
            "mean_confidence": 0.92,
            "n_total_detections": 25,
            "n_frame_detections": 10,
            "mean_box_px": {"width": 220, "height": 500},
            "all_crops": ["door.png"],
        },
        {
            "id": "box_1",
            "label": "package box",
            "mean_confidence": 0.88,
            "n_total_detections": 20,
            "n_frame_detections": 8,
            "mean_box_px": {"width": 180, "height": 180},
            "all_crops": ["box.png"],
        },
        {
            "id": "wall_1",
            "label": "wall segment",
            "mean_confidence": 0.95,
            "n_total_detections": 30,
        },
    ]
    index_uri = _ensure_index_file(tmp_path / "bucket", entries)

    payload = infer_task_targets(
        descriptor=descriptor,
        manifest=manifest,
        object_index_entries=entries,
        object_index_uri=index_uri,
        storage_root=tmp_path,
        max_targets=10,
    )

    assert "door_1" in payload["articulation_required_ids"]
    assert "box_1" in payload["target_object_ids"]
    assert "wall_1" not in payload["target_object_ids"]
    assert payload["inference_mode"] in {"heuristic", "external+heuristic", "descriptor_only"}


def test_explicit_only_mode_filters_policy_candidates(tmp_path: Path) -> None:
    descriptor = _descriptor(manip=[{"instance_id": "obj_focus", "label": "focus_item"}])
    object_entries = [
        {
            "id": "obj_focus",
            "label": "custom focus object",
            "boundingBox": {"extents": [0.3, 0.3, 0.3]},
            "mean_confidence": 0.5,
            "n_total_detections": 5,
        },
        {
            "id": "obj_box",
            "label": "box",
            "boundingBox": {"extents": [0.4, 0.3, 0.3]},
            "mean_confidence": 0.9,
            "n_total_detections": 30,
        },
    ]

    payload = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        selection_mode="explicit_only",
        max_candidates=10,
    )
    ids = [item["object_id"] for item in payload["candidates"]]
    assert ids == ["obj_focus"]
    assert payload["selection_mode"] == "explicit_only"


def test_hybrid_mode_cap_reserves_required_articulated(tmp_path: Path) -> None:
    descriptor = _descriptor()
    object_entries = [
        {
            "id": "drawer_1",
            "label": "drawer",
            "boundingBox": {"extents": [0.8, 0.3, 0.5]},
            "mean_confidence": 0.95,
            "n_total_detections": 30,
        },
        {
            "id": "door_1",
            "label": "door",
            "boundingBox": {"extents": [0.9, 1.8, 0.1]},
            "mean_confidence": 0.9,
            "n_total_detections": 22,
        },
        {
            "id": "box_1",
            "label": "box",
            "boundingBox": {"extents": [0.4, 0.3, 0.3]},
            "mean_confidence": 0.85,
            "n_total_detections": 18,
        },
    ]
    task_targets = {
        "articulation_hints": [
            {"instance_id": "drawer_1", "label": "drawer"},
            {"instance_id": "door_1", "label": "door"},
        ],
        "manipulation_candidates": [{"instance_id": "box_1", "label": "box"}],
    }

    payload = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        task_targets=task_targets,
        selection_mode="hybrid",
        max_candidates=1,
    )

    ids = {item["object_id"] for item in payload["candidates"]}
    assert "drawer_1" in ids
    assert "door_1" in ids
    summary = payload["selection_summary"]
    assert summary["cap_overridden_by_required"] is True
    assert summary["reserved_articulated_count"] >= 2


def test_hybrid_mode_uses_policy_only_as_backfill() -> None:
    descriptor = _descriptor(manip=[{"instance_id": "obj_focus", "label": "focus"}])
    object_entries = [
        {
            "id": "obj_focus",
            "label": "custom focus",
            "boundingBox": {"extents": [0.3, 0.3, 0.3]},
            "mean_confidence": 0.7,
            "n_total_detections": 10,
        },
        {
            "id": "obj_box",
            "label": "box",
            "boundingBox": {"extents": [0.4, 0.4, 0.4]},
            "mean_confidence": 0.95,
            "n_total_detections": 20,
        },
    ]

    payload_one = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        selection_mode="hybrid",
        max_candidates=1,
    )
    ids_one = [item["object_id"] for item in payload_one["candidates"]]
    assert ids_one == ["obj_focus"]
    assert payload_one["selection_summary"]["policy_backfill_count"] == 0

    payload_two = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        selection_mode="hybrid",
        max_candidates=2,
    )
    ids_two = {item["object_id"] for item in payload_two["candidates"]}
    assert "obj_focus" in ids_two
    assert "obj_box" in ids_two


def test_policy_only_mode_ignores_explicit_targets() -> None:
    descriptor = _descriptor(manip=[{"instance_id": "obj_focus", "label": "focus"}])
    object_entries = [
        {
            "id": "obj_focus",
            "label": "custom focus",
            "boundingBox": {"extents": [0.3, 0.3, 0.3]},
        },
        {
            "id": "obj_box",
            "label": "box",
            "boundingBox": {"extents": [0.4, 0.4, 0.4]},
        },
    ]

    payload = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        selection_mode="policy_only",
        max_candidates=10,
    )
    ids = {item["object_id"] for item in payload["candidates"]}
    assert "obj_box" in ids
    assert "obj_focus" not in ids
