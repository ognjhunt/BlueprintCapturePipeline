from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.synthesis.cosmos_training_export import export_cosmos_training_substrate


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
