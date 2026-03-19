from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.capture_orchestrator import PipelineConfig
from blueprint_pipeline.synthesis.cosmos_benchmark import run_cosmos_zero_shot_validation_lane


def _pose(tx: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_cosmos_benchmark_records_decoupled_reference_selection(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    pipeline_root = capture_root / "pipeline"
    (capture_root / "world_model_export").mkdir(parents=True)
    (pipeline_root / "scene_memory").mkdir(parents=True)
    (pipeline_root / "evaluation_prep").mkdir(parents=True)

    dense_rows = [
        {
            "reference_id": "target",
            "frame_id": "frame_0001",
            "frame_index": 1,
            "frame_uri": "gs://bucket/frames/frame_0001.jpg",
            "included_in_index": True,
            "t_capture_sec": 0.0,
            "T_world_camera": _pose(0.0),
            "anchor_observations": [],
            "retrieval_signals": {"anchor_observation_count": 0, "route_anchor_density": 0.0, "capture_confidence": 0.8},
        },
        {
            "reference_id": "near-duplicate",
            "frame_id": "frame_0002",
            "frame_index": 2,
            "frame_uri": "gs://bucket/frames/frame_0002.jpg",
            "included_in_index": True,
            "t_capture_sec": 0.05,
            "T_world_camera": _pose(0.01),
            "anchor_observations": [],
            "retrieval_signals": {"anchor_observation_count": 0, "route_anchor_density": 0.0, "capture_confidence": 0.85},
        },
        {
            "reference_id": "decoupled",
            "frame_id": "frame_0003",
            "frame_index": 8,
            "frame_uri": "gs://bucket/frames/frame_0003.jpg",
            "included_in_index": True,
            "t_capture_sec": 0.7,
            "T_world_camera": _pose(0.35),
            "anchor_observations": [],
            "retrieval_signals": {"anchor_observation_count": 0, "route_anchor_density": 0.3, "capture_confidence": 0.92},
        },
        {
            "reference_id": "decoupled-rich",
            "frame_id": "frame_0004",
            "frame_index": 12,
            "frame_uri": "gs://bucket/frames/frame_0004.jpg",
            "included_in_index": True,
            "t_capture_sec": 1.2,
            "T_world_camera": _pose(0.55),
            "anchor_observations": ["anchor_entry"],
            "retrieval_signals": {
                "anchor_observation_count": 1,
                "route_anchor_density": 0.75,
                "checkpoint_proximity_sec": 0.1,
                "capture_confidence": 0.95,
                "geometry_grounding_quality": 1.0,
            },
        },
    ]
    (capture_root / "world_model_export" / "dense_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dense_rows),
        encoding="utf-8",
    )
    (pipeline_root / "scene_memory" / "conditioning_bundle.json").write_text("{}", encoding="utf-8")
    (pipeline_root / "evaluation_prep" / "task_anchor_manifest.json").write_text(
        json.dumps({"tasks": [{"task_id": "task-1", "target_object_ids": ["obj-1"]}]}),
        encoding="utf-8",
    )
    (pipeline_root / "evaluation_prep" / "protected_regions_manifest.json").write_text(
        json.dumps({"regions": []}),
        encoding="utf-8",
    )

    descriptor_path = capture_root / "capture_descriptor.json"
    descriptor_path.write_text(
        json.dumps({"capture_id": "capture-1", "scene_id": "scene-1", "quality": {}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.cosmos_benchmark._probe_cosmos_runtime",
        lambda: {"status": "blocked", "blockers": ["missing_cosmos_runtime_package"], "packages": {}, "model_id": "test"},
    )

    manifest = run_cosmos_zero_shot_validation_lane(
        capture_root=capture_root,
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        cfg=PipelineConfig(gcs_root=tmp_path),
        max_examples=3,
    )

    reference_selection_manifest = json.loads(
        Path(manifest["reference_selection_manifest_path"]).read_text(encoding="utf-8")
    )
    reference_selection_comparison = json.loads(
        Path(manifest["reference_selection_comparison_path"]).read_text(encoding="utf-8")
    )
    synthetic_trajectory_manifest = json.loads(
        Path(manifest["synthetic_trajectory_manifest_path"]).read_text(encoding="utf-8")
    )
    future_anchor_manifest = json.loads(
        Path(manifest["future_anchor_regrounding_manifest_path"]).read_text(encoding="utf-8")
    )
    target_entry = next(
        entry for entry in reference_selection_manifest["entries"] if entry["target_frame_id"] == "frame_0001"
    )
    validation_entry = next(
        entry for entry in manifest["validation_set"] if entry["frame_id"] == "frame_0001"
    )

    assert manifest["status"] == "blocked"
    assert manifest["target_reference_decoupling_mode"] == "temporal_gap_with_pose_and_anchor_reranking"
    assert manifest["rejected_near_duplicate_count"] >= 1
    assert target_entry["selected_reference_frame_ids"][0] == "frame_0004"
    assert target_entry["rejected_near_duplicate_count"] >= 1
    assert validation_entry["selected_reference_frame_ids"][0] == "frame_0004"
    assert reference_selection_comparison["changed_primary_reference_count"] >= 1
    assert reference_selection_comparison["quality_metrics"]["primary_temporal_gap_sec"]["delta"] > 0
    assert synthetic_trajectory_manifest["augmented_target_count"] >= 1
    assert validation_entry["synthetic_waypoint_count"] >= 1
    assert validation_entry["synthetic_trajectory_status"] == "augmented"
    assert manifest["sparse_view_interpolation"]["interpolated_target_count"] == 0
    assert future_anchor_manifest["re_grounded_target_count"] >= 1
    assert validation_entry["future_anchor_status"] == "re_grounded"
    assert validation_entry["future_anchor_count"] >= 1


def test_cosmos_benchmark_surfaces_sparse_view_interpolation_when_context_is_sparse(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    pipeline_root = capture_root / "pipeline"
    (capture_root / "world_model_export").mkdir(parents=True)
    (pipeline_root / "scene_memory").mkdir(parents=True)
    (pipeline_root / "evaluation_prep").mkdir(parents=True)

    dense_rows = [
        {
            "reference_id": "target",
            "frame_id": "frame_0001",
            "frame_index": 1,
            "frame_uri": "gs://bucket/frames/frame_0001.jpg",
            "included_in_index": True,
            "t_capture_sec": 0.0,
            "T_world_camera": _pose(0.0),
            "anchor_observations": [],
            "retrieval_signals": {"anchor_observation_count": 0, "route_anchor_density": 0.0, "capture_confidence": 0.8},
        },
        {
            "reference_id": "decoupled-ref",
            "frame_id": "frame_0002",
            "frame_index": 10,
            "frame_uri": "gs://bucket/frames/frame_0002.jpg",
            "included_in_index": True,
            "t_capture_sec": 1.1,
            "T_world_camera": _pose(0.5),
            "anchor_observations": ["anchor_entry"],
            "retrieval_signals": {
                "anchor_observation_count": 1,
                "route_anchor_density": 0.8,
                "checkpoint_proximity_sec": 0.2,
                "capture_confidence": 0.94,
                "geometry_grounding_quality": 1.0,
            },
        },
    ]
    (capture_root / "world_model_export" / "dense_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dense_rows),
        encoding="utf-8",
    )
    (pipeline_root / "scene_memory" / "conditioning_bundle.json").write_text("{}", encoding="utf-8")
    (pipeline_root / "evaluation_prep" / "task_anchor_manifest.json").write_text(
        json.dumps({"tasks": [{"task_id": "task-1", "target_object_ids": ["obj-1"]}]}),
        encoding="utf-8",
    )
    (pipeline_root / "evaluation_prep" / "protected_regions_manifest.json").write_text(
        json.dumps({"regions": []}),
        encoding="utf-8",
    )
    descriptor_path = capture_root / "capture_descriptor.json"
    descriptor_path.write_text(
        json.dumps({"capture_id": "capture-1", "scene_id": "scene-1", "quality": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.cosmos_benchmark._probe_cosmos_runtime",
        lambda: {"status": "blocked", "blockers": ["missing_cosmos_runtime_package"], "packages": {}, "model_id": "test"},
    )

    manifest = run_cosmos_zero_shot_validation_lane(
        capture_root=capture_root,
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        cfg=PipelineConfig(gcs_root=tmp_path),
        max_examples=2,
    )
    interpolation_manifest = json.loads(
        Path(manifest["sparse_view_interpolation_manifest_path"]).read_text(encoding="utf-8")
    )
    future_anchor_manifest = json.loads(
        Path(manifest["future_anchor_regrounding_manifest_path"]).read_text(encoding="utf-8")
    )
    validation_entry = next(entry for entry in manifest["validation_set"] if entry["frame_id"] == "frame_0001")

    assert interpolation_manifest["interpolated_target_count"] >= 1
    assert validation_entry["synthetic_trajectory_status"] == "skipped"
    assert validation_entry["sparse_view_interpolation_status"] == "interpolated"
    assert validation_entry["interpolated_view_count"] >= 1
    assert future_anchor_manifest["re_grounded_target_count"] >= 1
    assert validation_entry["future_anchor_status"] == "re_grounded"
    assert validation_entry["future_anchor_count"] >= 1
