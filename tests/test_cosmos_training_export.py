from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.synthesis import cosmos_training_export as export_module
from blueprint_pipeline.synthesis.cosmos_training_export import export_cosmos_training_substrate


pytestmark = pytest.mark.slow


def test_export_cosmos_training_substrate_writes_real_artifacts(tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    pipeline_root = capture_root / "pipeline"
    (capture_root / "world_model_export").mkdir(parents=True)
    (pipeline_root / "scene_memory").mkdir(parents=True)
    (pipeline_root / "evaluation_prep").mkdir(parents=True)

    dense_rows = [
        {
            "frame_id": "frame_0001",
            "frame_uri": "gs://bucket/frames/frame_0001.jpg",
            "embedding_uri": "gs://bucket/embeddings/frame_0001.bin",
            "included_in_index": True,
            "t_capture_sec": 0.0,
            "T_world_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240, "width": 640, "height": 480},
            "anchor_observations": [{"anchor_id": "anchor_entry"}],
        },
        {
            "frame_id": "frame_0002",
            "frame_uri": "gs://bucket/frames/frame_0002.jpg",
            "embedding_uri": "gs://bucket/embeddings/frame_0002.bin",
            "included_in_index": True,
            "t_capture_sec": 1.0,
            "T_world_camera": [[1, 0, 0, 0.2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240, "width": 640, "height": 480},
            "anchor_observations": [{"anchor_id": "anchor_pick"}],
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
        json.dumps({"regions": [{"region_id": "rg-1"}]}),
        encoding="utf-8",
    )

    manifest = export_cosmos_training_substrate(capture_root=capture_root)

    assert manifest["status"] == "ready"
    assert manifest["paired_example_count"] == 2
    assert Path(manifest["paired_reference_target_path"]).is_file()
    assert Path(manifest["k_reference_conditioning_path"]).is_file()
    assert Path(manifest["trainer_config_path"]).is_file()
    assert Path(manifest["checkpoint_layout_path"]).is_file()
    assert Path(manifest["inference_backend_shape_path"]).is_file()
    assert Path(manifest["reference_selection_manifest_path"]).is_file()
    assert Path(manifest["reference_selection_comparison_path"]).is_file()
    assert Path(manifest["synthetic_trajectory_manifest_path"]).is_file()
    assert Path(manifest["sparse_view_interpolation_manifest_path"]).is_file()
    assert Path(manifest["future_anchor_regrounding_manifest_path"]).is_file()
    assert manifest["target_reference_decoupling_mode"] == "temporal_gap_with_pose_and_anchor_reranking"


def test_export_cosmos_training_substrate_bootstraps_from_capture_video(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")

    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    raw_root = capture_root / "raw"
    arkit_root = raw_root / "arkit"
    pipeline_root = capture_root / "pipeline"
    (pipeline_root / "scene_memory").mkdir(parents=True)
    (pipeline_root / "evaluation_prep").mkdir(parents=True)
    arkit_root.mkdir(parents=True)

    video_path = raw_root / "walkthrough.mov"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (64, 48))
    for index in range(6):
        frame = np.full((48, 64, 3), index * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    pose_rows = [
        {
            "frame_id": f"{index:06d}",
            "t_device_sec": float(index) * 0.25,
            "T_world_camera": [[1, 0, 0, index * 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        }
        for index in range(6)
    ]
    (arkit_root / "poses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in pose_rows),
        encoding="utf-8",
    )
    (arkit_root / "intrinsics.json").write_text(
        json.dumps({"fx": 500, "fy": 500, "cx": 32, "cy": 24, "width": 64, "height": 48}),
        encoding="utf-8",
    )
    (pipeline_root / "scene_memory" / "conditioning_bundle.json").write_text(
        json.dumps(
            {
                "raw_video_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/walkthrough.mov",
                "arkit": {
                    "poses_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/arkit/poses.jsonl",
                    "intrinsics_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/arkit/intrinsics.json",
                },
            }
        ),
        encoding="utf-8",
    )
    (pipeline_root / "evaluation_prep" / "task_anchor_manifest.json").write_text(
        json.dumps({"tasks": [{"task_id": "task-1", "target_object_ids": ["obj-1"]}]}),
        encoding="utf-8",
    )
    (pipeline_root / "evaluation_prep" / "protected_regions_manifest.json").write_text(
        json.dumps({"regions": []}),
        encoding="utf-8",
    )

    manifest = export_cosmos_training_substrate(capture_root=capture_root)
    paired_rows = Path(manifest["paired_reference_target_path"]).read_text(encoding="utf-8").splitlines()

    assert manifest["status"] == "ready"
    assert manifest["source_mode"] == "video_bootstrap"
    assert manifest["paired_example_count"] >= 2
    assert '"source_mode":"video_bootstrap"' in paired_rows[0]


def test_export_cosmos_training_substrate_rejects_near_duplicate_reference_targets(tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    pipeline_root = capture_root / "pipeline"
    (capture_root / "world_model_export").mkdir(parents=True)
    (pipeline_root / "scene_memory").mkdir(parents=True)
    (pipeline_root / "evaluation_prep").mkdir(parents=True)

    dense_rows = [
        {
            "reference_id": "target-ref",
            "frame_id": "frame_0001",
            "frame_index": 1,
            "frame_uri": "gs://bucket/frames/frame_0001.jpg",
            "embedding_uri": "gs://bucket/embeddings/frame_0001.bin",
            "included_in_index": True,
            "t_capture_sec": 0.0,
            "T_world_camera": [[1, 0, 0, 0.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240, "width": 640, "height": 480},
            "anchor_observations": [],
            "retrieval_signals": {"anchor_observation_count": 0, "route_anchor_density": 0.0, "capture_confidence": 0.8},
        },
        {
            "reference_id": "near-duplicate",
            "frame_id": "frame_0002",
            "frame_index": 2,
            "frame_uri": "gs://bucket/frames/frame_0002.jpg",
            "embedding_uri": "gs://bucket/embeddings/frame_0002.bin",
            "included_in_index": True,
            "t_capture_sec": 0.05,
            "T_world_camera": [[1, 0, 0, 0.01], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240, "width": 640, "height": 480},
            "anchor_observations": [],
            "retrieval_signals": {"anchor_observation_count": 0, "route_anchor_density": 0.0, "capture_confidence": 0.9},
        },
        {
            "reference_id": "decoupled",
            "frame_id": "frame_0003",
            "frame_index": 8,
            "frame_uri": "gs://bucket/frames/frame_0003.jpg",
            "embedding_uri": "gs://bucket/embeddings/frame_0003.bin",
            "included_in_index": True,
            "t_capture_sec": 0.7,
            "T_world_camera": [[1, 0, 0, 0.35], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240, "width": 640, "height": 480},
            "anchor_observations": [],
            "retrieval_signals": {"anchor_observation_count": 0, "route_anchor_density": 0.3, "capture_confidence": 0.92},
        },
        {
            "reference_id": "decoupled-rich",
            "frame_id": "frame_0004",
            "frame_index": 12,
            "frame_uri": "gs://bucket/frames/frame_0004.jpg",
            "embedding_uri": "gs://bucket/embeddings/frame_0004.bin",
            "included_in_index": True,
            "t_capture_sec": 1.2,
            "T_world_camera": [[1, 0, 0, 0.6], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240, "width": 640, "height": 480},
            "anchor_observations": ["anchor_entry", "checkpoint_pick"],
            "retrieval_signals": {
                "anchor_observation_count": 2,
                "route_anchor_density": 1.0,
                "checkpoint_proximity_sec": 0.1,
                "capture_confidence": 0.97,
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

    manifest = export_cosmos_training_substrate(capture_root=capture_root, k_references=1)
    reference_selection_manifest = json.loads(
        Path(manifest["reference_selection_manifest_path"]).read_text(encoding="utf-8")
    )
    reference_selection_comparison = json.loads(
        Path(manifest["reference_selection_comparison_path"]).read_text(encoding="utf-8")
    )
    synthetic_trajectory_manifest = json.loads(
        Path(manifest["synthetic_trajectory_manifest_path"]).read_text(encoding="utf-8")
    )
    sparse_view_interpolation_manifest = json.loads(
        Path(manifest["sparse_view_interpolation_manifest_path"]).read_text(encoding="utf-8")
    )
    future_anchor_manifest = json.loads(
        Path(manifest["future_anchor_regrounding_manifest_path"]).read_text(encoding="utf-8")
    )
    target_entry = next(
        entry for entry in reference_selection_manifest["entries"] if entry["target_frame_id"] == "frame_0001"
    )
    paired_rows = [
        json.loads(line)
        for line in Path(manifest["paired_reference_target_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    paired_target = next(row for row in paired_rows if row["frame_id"] == "frame_0001")

    assert manifest["rejected_near_duplicate_count"] >= 1
    assert target_entry["selected_reference_frame_ids"] == ["frame_0004"]
    assert target_entry["rejected_near_duplicate_count"] >= 1
    assert paired_target["selected_reference_frame_ids"] == ["frame_0004"]
    assert paired_target["target_reference_decoupling_mode"] == "temporal_gap_with_pose_and_anchor_reranking"
    assert reference_selection_comparison["changed_primary_reference_count"] >= 1
    assert reference_selection_comparison["quality_metrics"]["primary_temporal_gap_sec"]["delta"] > 0
    assert synthetic_trajectory_manifest["augmented_target_count"] >= 1
    assert paired_target["synthetic_waypoint_count"] >= 1
    assert paired_target["synthetic_trajectory_status"] == "augmented"
    assert sparse_view_interpolation_manifest["interpolated_target_count"] == 0
    assert paired_target["interpolated_view_count"] == 0
    assert paired_target["sparse_view_interpolation_status"] == "skipped"
    assert future_anchor_manifest["re_grounded_target_count"] >= 1
    assert paired_target["future_anchor_status"] == "re_grounded"
    assert paired_target["future_anchor_count"] >= 1


def test_export_cosmos_training_substrate_surfaces_sparse_view_interpolation_when_context_is_sparse(tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    pipeline_root = capture_root / "pipeline"
    (capture_root / "world_model_export").mkdir(parents=True)
    (pipeline_root / "scene_memory").mkdir(parents=True)
    (pipeline_root / "evaluation_prep").mkdir(parents=True)

    dense_rows = [
        {
            "reference_id": "target-ref",
            "frame_id": "frame_0001",
            "frame_index": 1,
            "frame_uri": "gs://bucket/frames/frame_0001.jpg",
            "embedding_uri": "gs://bucket/embeddings/frame_0001.bin",
            "included_in_index": True,
            "t_capture_sec": 0.0,
            "T_world_camera": [[1, 0, 0, 0.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240, "width": 640, "height": 480},
            "anchor_observations": [],
            "retrieval_signals": {"anchor_observation_count": 0, "route_anchor_density": 0.0, "capture_confidence": 0.8},
        },
        {
            "reference_id": "decoupled-ref",
            "frame_id": "frame_0002",
            "frame_index": 10,
            "frame_uri": "gs://bucket/frames/frame_0002.jpg",
            "embedding_uri": "gs://bucket/embeddings/frame_0002.bin",
            "included_in_index": True,
            "t_capture_sec": 1.2,
            "T_world_camera": [[1, 0, 0, 0.55], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240, "width": 640, "height": 480},
            "anchor_observations": ["anchor_entry"],
            "retrieval_signals": {
                "anchor_observation_count": 1,
                "route_anchor_density": 0.8,
                "checkpoint_proximity_sec": 0.2,
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

    manifest = export_cosmos_training_substrate(capture_root=capture_root, k_references=1)
    interpolation_manifest = json.loads(
        Path(manifest["sparse_view_interpolation_manifest_path"]).read_text(encoding="utf-8")
    )
    future_anchor_manifest = json.loads(
        Path(manifest["future_anchor_regrounding_manifest_path"]).read_text(encoding="utf-8")
    )
    paired_rows = [
        json.loads(line)
        for line in Path(manifest["paired_reference_target_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    target_row = next(row for row in paired_rows if row["frame_id"] == "frame_0001")

    assert interpolation_manifest["interpolated_target_count"] >= 1
    assert target_row["synthetic_trajectory_status"] == "skipped"
    assert target_row["sparse_view_interpolation_status"] == "interpolated"
    assert target_row["interpolated_view_count"] >= 1
    assert future_anchor_manifest["re_grounded_target_count"] >= 1
    assert target_row["future_anchor_status"] == "re_grounded"
    assert target_row["future_anchor_count"] >= 1


def test_cosmos_training_export_missing_and_selection_edge_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text('\n{"a":1}\n', encoding="utf-8")
    assert export_module._read_jsonl(jsonl_path) == [{"a": 1}]

    missing_root = tmp_path / "bucket" / "scenes" / "scene-missing" / "captures" / "capture-missing"
    missing_manifest = export_cosmos_training_substrate(capture_root=missing_root)
    assert missing_manifest["status"] == "missing"
    assert missing_manifest["reason"] == "insufficient_dense_index_records"

    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    (capture_root / "world_model_export").mkdir(parents=True)
    dense_rows = [
        {
            "frame_id": "flat",
            "frame_uri": "gs://bucket/flat.jpg",
            "embedding_uri": "gs://bucket/flat.bin",
            "included_in_index": True,
            "T_world_camera": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "intrinsics": {"width": 16, "height": 16},
        },
        {
            "frame_id": "bad-shape",
            "frame_uri": "gs://bucket/bad.jpg",
            "embedding_uri": "gs://bucket/bad.bin",
            "included_in_index": True,
            "T_world_camera": [[1, 2], [3, 4]],
            "intrinsics": {"width": 16, "height": 16},
        },
    ]
    (capture_root / "world_model_export" / "dense_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dense_rows),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        export_module,
        "build_reference_selection_manifest",
        lambda **_kwargs: {
            "entries": [
                {"target_index": 99, "selected_references": []},
                {"target_index": 0, "selected_references": []},
                {
                    "target_index": 0,
                    "target_frame_id": "flat",
                    "selected_references": [{"candidate_index": 1, "score": 0.9}],
                    "selected_reference_ids": ["bad-shape"],
                    "selected_reference_frame_ids": ["bad-shape"],
                },
                {
                    "target_index": 1,
                    "target_frame_id": "bad-shape",
                    "selected_references": [{"candidate_index": 0, "score": 0.8}],
                    "selected_reference_ids": ["flat"],
                    "selected_reference_frame_ids": ["flat"],
                },
            ],
            "target_reference_decoupling_mode": "unit_test",
        },
    )
    monkeypatch.setattr(
        export_module,
        "build_legacy_reference_selection_manifest",
        lambda **_kwargs: {"entries": []},
    )
    monkeypatch.setattr(
        export_module,
        "build_reference_selection_comparison",
        lambda **_kwargs: {"entries": []},
    )
    monkeypatch.setattr(
        export_module,
        "build_synthetic_trajectory_manifest",
        lambda **_kwargs: {"entries": []},
    )
    monkeypatch.setattr(
        export_module,
        "build_sparse_view_interpolation_manifest",
        lambda **_kwargs: {"entries": []},
    )
    monkeypatch.setattr(
        export_module,
        "build_future_anchor_regrounding_manifest",
        lambda **_kwargs: {"entries": []},
    )

    manifest = export_cosmos_training_substrate(capture_root=capture_root)
    paired_rows = [
        json.loads(line)
        for line in Path(manifest["paired_reference_target_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Neither fixture record has real fx/fy calibration, so both are
    # rejected instead of exported with guessed intrinsics; "bad-shape" is
    # separately unusable on pose grounds too. No identity-filled poses or
    # guessed intrinsics reach the exported training targets.
    assert manifest["status"] == "missing"
    assert paired_rows == []
    assert manifest["rejected_record_count"] == 2
    rejections = json.loads(
        Path(manifest["export_rejection_manifest_path"]).read_text(encoding="utf-8")
    )["rejections"]
    reasons_by_frame = {row["frame_id"]: row["reasons"] for row in rejections}
    assert "intrinsics_missing_or_implausible" in reasons_by_frame["flat"]
    assert "intrinsics_missing_or_implausible" in reasons_by_frame["bad-shape"]
    assert "pose_missing_or_misshaped" in reasons_by_frame["bad-shape"]
    assert "pose_missing_or_misshaped" not in reasons_by_frame["flat"]

    # The pose-specific skip ledger only captures "bad-shape" (its pose is
    # genuinely malformed); "flat"'s pose is fine, only its intrinsics are missing.
    assert manifest["skipped_missing_pose_count"] == 1
    assert manifest["skipped_missing_pose_rows"] == [
        {
            "target_index": 1,
            "target_frame_id": "bad-shape",
            "reason": "target_T_world_camera_missing_or_invalid",
        }
    ]


def test_cosmos_training_export_blocks_synthetic_geometry_in_production(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-syn" / "captures" / "capture-syn"
    geometry_dir = capture_root / "pipeline" / "geometry"
    geometry_dir.mkdir(parents=True)
    (geometry_dir / "geometry_summary.json").write_text(
        json.dumps({"fallback_used": True, "fallback_kind": "internal_synthetic_geometry"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")

    manifest = export_cosmos_training_substrate(capture_root=capture_root)

    assert manifest["status"] == "blocked"
    assert "synthetic_or_fallback_geometry_disallowed" in manifest["reason"]


def test_cosmos_training_export_stamps_dev_synthetic_geometry_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-dev" / "captures" / "capture-dev"
    geometry_dir = capture_root / "pipeline" / "geometry"
    geometry_dir.mkdir(parents=True)
    (geometry_dir / "geometry_summary.json").write_text(
        json.dumps({"fallback_used": True, "fallback_kind": "local_sfm_synthetic_dev"}),
        encoding="utf-8",
    )
    monkeypatch.delenv("BLUEPRINT_LAUNCH_PROOF_MODE", raising=False)

    manifest = export_cosmos_training_substrate(capture_root=capture_root)

    # Dev allowance: export proceeds (no dense records -> missing) but the
    # manifest must carry the synthetic-geometry provenance stamp.
    assert manifest["status"] == "missing"
    assert manifest["geometry_provenance"]["synthetic_geometry"] is True
    assert manifest["geometry_provenance"]["export_allowed_by"] == "synthetic_geometry_dev_allowance"
