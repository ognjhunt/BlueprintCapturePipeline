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


def test_infer_task_targets_stays_empty_with_object_index_only(tmp_path: Path) -> None:
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

    assert payload["articulation_required_ids"] == []
    assert payload["target_object_ids"] == []
    assert payload["tasks"] == []
    assert payload["inference_mode"] == "empty"


def test_infer_task_targets_uses_descriptor_hints_and_enriches_with_grounding(tmp_path: Path) -> None:
    descriptor = _descriptor(manip=[{"instance_id": "box_1", "label": "object"}])
    manifest = _manifest()
    entries = [
        {
            "id": "box_1",
            "label": "package box",
            "boundingBox": {"center": [1.0, 2.0, 0.5], "extents": [0.4, 0.3, 0.2]},
            "mean_confidence": 0.88,
            "n_total_detections": 20,
        }
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

    assert payload["target_object_ids"] == ["box_1"]
    assert payload["inference_mode"] == "descriptor_only"
    assert payload["manipulation_candidates"][0]["label"] == "package box"
    assert payload["manipulation_candidates"][0]["boundingBox"]["center"] == [1.0, 2.0, 0.5]


def test_infer_task_targets_uses_grounding_backend_candidates(tmp_path: Path) -> None:
    descriptor = _descriptor()
    manifest = _manifest()
    entries = [
        {
            "id": "drawer_1",
            "label": "drawer",
            "boundingBox": {"center": [0.5, 1.0, 0.5], "extents": [0.7, 0.4, 0.4]},
            "mean_confidence": 0.9,
            "n_total_detections": 10,
        }
    ]
    index_uri = _ensure_index_file(tmp_path / "bucket", entries)
    grounding_payload = {
        "backend": "holi_adapter",
        "grounded_objects": [
            {
                "object_id": "drawer_1",
                "label": "drawer",
                "confidence": 0.92,
                "boundingBox": {"center": [0.5, 1.0, 0.5], "extents": [0.7, 0.4, 0.4]},
            }
        ],
        "articulation_hints": [{"instance_id": "drawer_1", "label": "drawer"}],
        "tasks": [{"task_id": "open_drawer", "target_object_ids": ["drawer_1"]}],
        "backend_report": {"status": "ok"},
    }

    payload = infer_task_targets(
        descriptor=descriptor,
        manifest=manifest,
        object_index_entries=entries,
        object_index_uri=index_uri,
        storage_root=tmp_path,
        grounding_payload=grounding_payload,
        max_targets=10,
    )

    assert payload["articulation_required_ids"] == ["drawer_1"]
    assert payload["inference_mode"] == "external"
    assert payload["tasks"][0]["task_id"] == "open_drawer"


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


def test_object_dedupe_merges_overlapping_same_label_fragments() -> None:
    descriptor = _descriptor()
    object_entries = [
        {
            "id": "door_a",
            "label": "door",
            "boundingBox": {"center": [0.0, 1.0, 0.0], "extents": [1.0, 2.0, 0.2]},
            "mean_confidence": 0.92,
            "n_total_detections": 20,
            "all_crops": ["door_a.png"],
        },
        {
            "id": "door_b",
            "label": "door",
            "boundingBox": {"center": [0.08, 1.02, 0.02], "extents": [1.02, 1.95, 0.22]},
            "mean_confidence": 0.89,
            "n_total_detections": 18,
            "all_crops": ["door_b.png"],
        },
        {
            "id": "box_1",
            "label": "box",
            "boundingBox": {"center": [2.0, 0.4, 1.0], "extents": [0.4, 0.3, 0.3]},
            "mean_confidence": 0.8,
            "n_total_detections": 12,
        },
    ]

    payload = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        selection_mode="policy_only",
        max_candidates=10,
        per_class_caps={"door": 10},
    )
    ids = [item["object_id"] for item in payload["candidates"]]
    door_count = sum(1 for item in payload["candidates"] if str(item.get("label")) == "door")
    assert door_count == 1
    assert any(obj_id.startswith("door_") for obj_id in ids)

    preprocess = payload["index_preprocessing"]["dedupe"]
    assert preprocess["original_count"] == 3
    assert preprocess["deduped_count"] == 2


def test_per_class_caps_applied_before_candidate_selection() -> None:
    descriptor = _descriptor()
    object_entries = [
        {
            "id": "door_1",
            "label": "door",
            "boundingBox": {"center": [0.0, 1.0, 0.0], "extents": [1.0, 2.0, 0.2]},
            "mean_confidence": 0.95,
            "n_total_detections": 20,
        },
        {
            "id": "door_2",
            "label": "door",
            "boundingBox": {"center": [3.0, 1.0, 0.0], "extents": [1.0, 2.0, 0.2]},
            "mean_confidence": 0.9,
            "n_total_detections": 18,
        },
        {
            "id": "door_3",
            "label": "door",
            "boundingBox": {"center": [6.0, 1.0, 0.0], "extents": [1.0, 2.0, 0.2]},
            "mean_confidence": 0.85,
            "n_total_detections": 15,
        },
        {
            "id": "box_1",
            "label": "box",
            "boundingBox": {"center": [2.0, 0.4, 1.0], "extents": [0.4, 0.3, 0.3]},
            "mean_confidence": 0.8,
            "n_total_detections": 12,
        },
    ]

    payload = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        selection_mode="policy_only",
        max_candidates=20,
        per_class_caps={"door": 2},
    )

    doors = [item for item in payload["candidates"] if str(item.get("label")) == "door"]
    assert len(doors) == 2

    class_caps = payload["index_preprocessing"]["class_caps"]
    assert class_caps["dropped_count"] >= 1
    assert class_caps["dropped_by_label"].get("door", 0) >= 1


def test_grounding_backend_targets_become_explicit() -> None:
    descriptor = _descriptor()
    object_entries = [
        {
            "id": "door_1",
            "label": "door",
            "boundingBox": {"center": [0.0, 1.0, 0.0], "extents": [1.0, 2.0, 0.2]},
            "mean_confidence": 0.95,
            "n_frame_detections": 3,
            "n_total_detections": 3,
        },
        {
            "id": "box_1",
            "label": "box",
            "boundingBox": {"center": [2.0, 0.4, 1.0], "extents": [0.4, 0.3, 0.3]},
            "mean_confidence": 0.8,
            "n_frame_detections": 3,
            "n_total_detections": 3,
        },
    ]
    task_targets = {
        "inference_mode": "external",
        "manipulation_candidates": [{"instance_id": "box_1", "label": "box", "source": "grounding_backend"}],
        "articulation_hints": [{"instance_id": "door_1", "label": "door", "source": "grounding_backend"}],
        "target_object_ids": ["box_1"],
        "articulation_required_ids": ["door_1"],
    }

    payload = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        task_targets=task_targets,
        selection_mode="hybrid",
        max_candidates=2,
    )

    assert payload["selection_summary"]["explicit_count"] == 2


def test_detection_support_filter_drops_low_support_fragments() -> None:
    descriptor = _descriptor()
    object_entries = [
        {
            "id": "box_1",
            "label": "box",
            "boundingBox": {"center": [0.0, 0.5, 0.0], "extents": [0.4, 0.4, 0.4]},
            "mean_confidence": 0.8,
            "n_frame_detections": 1,
            "n_total_detections": 1,
        },
        {
            "id": "box_2",
            "label": "box",
            "boundingBox": {"center": [1.0, 0.5, 0.0], "extents": [0.4, 0.4, 0.4]},
            "mean_confidence": 0.8,
            "n_frame_detections": 1,
            "n_total_detections": 1,
        },
        {
            "id": "box_3",
            "label": "box",
            "boundingBox": {"center": [2.0, 0.5, 0.0], "extents": [0.4, 0.4, 0.4]},
            "mean_confidence": 0.8,
            "n_frame_detections": 2,
            "n_total_detections": 2,
        },
    ]

    payload = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        selection_mode="policy_only",
        max_candidates=10,
        per_class_caps={"box": 10},
    )

    ids = [item["object_id"] for item in payload["candidates"]]
    assert ids == ["box_3"]
    support = payload["index_preprocessing"]["detection_support"]
    assert support["dropped_low_support_count"] == 2


def test_semantic_box_cap_groups_package_and_container_labels() -> None:
    descriptor = _descriptor()
    object_entries = [
        {
            "id": "obj_package",
            "label": "package",
            "boundingBox": {"center": [0.0, 0.5, 0.0], "extents": [0.4, 0.4, 0.4]},
            "mean_confidence": 0.9,
            "n_frame_detections": 2,
            "n_total_detections": 2,
        },
        {
            "id": "obj_box",
            "label": "box",
            "boundingBox": {"center": [1.0, 0.5, 0.0], "extents": [0.4, 0.4, 0.4]},
            "mean_confidence": 0.85,
            "n_frame_detections": 2,
            "n_total_detections": 2,
        },
        {
            "id": "obj_container",
            "label": "container",
            "boundingBox": {"center": [2.0, 0.5, 0.0], "extents": [0.4, 0.4, 0.4]},
            "mean_confidence": 0.8,
            "n_frame_detections": 2,
            "n_total_detections": 2,
        },
    ]

    payload = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        selection_mode="policy_only",
        max_candidates=10,
        per_class_caps={"box": 2},
    )

    assert len(payload["candidates"]) == 2
    class_caps = payload["index_preprocessing"]["class_caps"]
    assert class_caps["dropped_by_label"].get("box", 0) == 1


def test_per_class_caps_enforced_when_explicitly_set() -> None:
    """Per-class caps are enforced when explicitly provided."""
    descriptor = _descriptor()
    object_entries = [
        {
            "id": f"door_{i}",
            "label": "door",
            "boundingBox": {"center": [float(i * 3), 1.0, 0.0], "extents": [1.0, 2.0, 0.2]},
            "mean_confidence": 0.9 - i * 0.03,
            "n_total_detections": 20 - i,
        }
        for i in range(10)
    ]
    task_targets = {
        "inference_mode": "external",
        "manipulation_candidates": [],
        "articulation_hints": [
            {"instance_id": f"door_{i}", "label": "door", "source": "grounding_backend"}
            for i in range(10)
        ],
        "target_object_ids": [],
        "articulation_required_ids": [f"door_{i}" for i in range(10)],
    }

    payload = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        task_targets=task_targets,
        selection_mode="hybrid",
        max_candidates=24,
        per_class_caps={"door": 4},
    )

    doors = [c for c in payload["candidates"] if c["label"] == "door"]
    assert len(doors) == 10, f"expected all explicit doors to bypass caps, got {len(doors)}"
    assert payload["selection_summary"]["explicit_count"] == 10
    class_caps = payload["index_preprocessing"]["class_caps"]
    assert class_caps["dropped_by_label"].get("door", 0) == 0


def test_no_default_per_class_caps() -> None:
    """Without explicit caps, all qualifying objects pass through."""
    descriptor = _descriptor()
    object_entries = [
        {
            "id": f"door_{i}",
            "label": "door",
            "boundingBox": {"center": [float(i * 3), 1.0, 0.0], "extents": [1.0, 2.0, 0.2]},
            "mean_confidence": 0.9,
            "n_total_detections": 10,
        }
        for i in range(8)
    ]

    payload = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        selection_mode="hybrid",
        max_candidates=24,
        # no per_class_caps — defaults should be empty
    )

    doors = [c for c in payload["candidates"] if c["label"] == "door"]
    assert len(doors) == 8, f"expected all 8 doors without caps, got {len(doors)}"


def test_single_detection_objects_filtered_by_support() -> None:
    """Objects with only 1 detection are dropped by the detection support filter."""
    descriptor = _descriptor()
    # 5 multi-detection doors + 10 single-detection doors
    object_entries = [
        {
            "id": f"door_multi_{i}",
            "label": "door",
            "boundingBox": {"center": [float(i * 3), 1.0, 0.0], "extents": [1.0, 2.0, 0.2]},
            "mean_confidence": 0.9,
            "n_total_detections": 5,
        }
        for i in range(5)
    ] + [
        {
            "id": f"door_single_{i}",
            "label": "door",
            "boundingBox": {"center": [float(i * 3 + 20), 1.0, 0.0], "extents": [1.0, 2.0, 0.2]},
            "mean_confidence": 0.7,
            "n_total_detections": 1,
        }
        for i in range(10)
    ]

    payload = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        selection_mode="hybrid",
        max_candidates=24,
    )

    door_ids = [c["object_id"] for c in payload["candidates"] if c["label"] == "door"]
    assert all(did.startswith("door_multi_") for did in door_ids), (
        f"single-detection doors should be filtered, got: {door_ids}"
    )
    assert len(door_ids) == 5
    support = payload["index_preprocessing"]["detection_support"]
    assert support["dropped_low_support_count"] == 10


def test_descriptor_explicit_targets_still_bypass_caps() -> None:
    """Targets explicitly requested in the descriptor bypass per-class caps."""
    descriptor = _descriptor(
        artic=[
            {"instance_id": "door_0", "label": "door"},
            {"instance_id": "door_1", "label": "door"},
            {"instance_id": "door_2", "label": "door"},
        ],
    )
    object_entries = [
        {
            "id": f"door_{i}",
            "label": "door",
            "boundingBox": {"center": [float(i * 3), 1.0, 0.0], "extents": [1.0, 2.0, 0.2]},
            "mean_confidence": 0.9,
            "n_total_detections": 10,
        }
        for i in range(5)
    ]

    payload = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        selection_mode="hybrid",
        max_candidates=24,
        per_class_caps={"door": 1},
    )

    doors = [c for c in payload["candidates"] if c["label"] == "door"]
    # 3 explicit + at most 1 policy-selected = 4, but only 5 total
    assert len(doors) >= 3, f"explicit doors must survive cap, got {len(doors)}"
    explicit_ids = {
        c["object_id"]
        for c in payload["candidates"]
        if c.get("selection", {}).get("explicit")
    }
    assert {"door_0", "door_1", "door_2"}.issubset(explicit_ids)


def test_bedroom_environment_applies_default_residential_caps() -> None:
    descriptor = CaptureDescriptor.from_dict(
        {
            "schema_version": "v1",
            "scene_id": "scene_bedroom",
            "capture_id": "cap_bedroom",
            "capture_source": "iphone",
            "capture_tier": "tier1_iphone",
            "raw_prefix_uri": "gs://bucket/scenes/scene_bedroom/iphone/cap_bedroom/raw",
            "frames_index_uri": "gs://bucket/scenes/scene_bedroom/captures/cap_bedroom/frames/index.jsonl",
            "nurec_mode": "mono_pose_assisted",
            "environment_type_hint": "bedroom",
        }
    )
    object_entries = [
        {
            "id": f"door_{i}",
            "label": "door",
            "boundingBox": {"center": [float(i * 2), 1.0, 0.0], "extents": [1.0, 2.0, 0.2]},
            "mean_confidence": 0.9 - i * 0.03,
            "n_total_detections": 10,
        }
        for i in range(8)
    ]

    payload = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        selection_mode="policy_only",
        max_candidates=20,
    )

    doors = [item for item in payload["candidates"] if item["label"] == "door"]
    assert len(doors) == 4
    class_caps = payload["index_preprocessing"]["class_caps"]
    assert class_caps["diagnostics"]["source"] == "environment_default:bedroom"
    assert class_caps["dropped_by_label"].get("door", 0) == 4


def test_residential_default_caps_allow_explicit_id_bypass() -> None:
    descriptor = CaptureDescriptor.from_dict(
        {
            "schema_version": "v1",
            "scene_id": "scene_bedroom",
            "capture_id": "cap_bedroom",
            "capture_source": "iphone",
            "capture_tier": "tier1_iphone",
            "raw_prefix_uri": "gs://bucket/scenes/scene_bedroom/iphone/cap_bedroom/raw",
            "frames_index_uri": "gs://bucket/scenes/scene_bedroom/captures/cap_bedroom/frames/index.jsonl",
            "nurec_mode": "mono_pose_assisted",
            "environment_type_hint": "bedroom",
            "articulation_hints": [{"instance_id": f"door_{i}", "label": "door"} for i in range(6)],
        }
    )
    object_entries = [
        {
            "id": f"door_{i}",
            "label": "door",
            "boundingBox": {"center": [float(i * 2), 1.0, 0.0], "extents": [1.0, 2.0, 0.2]},
            "mean_confidence": 0.9 - i * 0.03,
            "n_total_detections": 10,
        }
        for i in range(8)
    ]

    payload = build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=object_entries,
        selection_mode="hybrid",
        max_candidates=20,
    )

    explicit_ids = {
        item["object_id"]
        for item in payload["candidates"]
        if item.get("selection", {}).get("explicit")
    }
    assert {"door_0", "door_1", "door_2", "door_3", "door_4", "door_5"}.issubset(explicit_ids)
    class_caps = payload["index_preprocessing"]["class_caps"]
    assert class_caps["diagnostics"]["explicit_bypass_mode"] == "descriptor_and_external_object_ids_only"
