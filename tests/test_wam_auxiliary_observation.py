from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from blueprint_pipeline.wam_auxiliary_observation import (
    build_wam_auxiliary_observation_manifest,
    summarize_wam_auxiliary_observation_manifest,
)


def _png(path: Path, *, size: tuple[int, int] = (640, 360)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(160, 180, 200)).save(path)
    return path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_build_wam_auxiliary_observation_manifest_records_modalities_and_boundaries(
    tmp_path: Path,
) -> None:
    source_image = _png(tmp_path / "initial_policy_frame.png")
    depth = tmp_path / "depth.npy"
    depth.write_bytes(b"depth")
    target_mask = _png(tmp_path / "target_mask.png")
    robot_mask = _png(tmp_path / "robot_mask.png")
    trace = tmp_path / "g1_projected_skeleton_trace.jsonl"
    trace.write_text('{"frame_index":0}\n', encoding="utf-8")

    manifest = build_wam_auxiliary_observation_manifest(
        output_dir=tmp_path / "aux",
        source_image_path=source_image,
        generated_at="2026-06-25T00:00:00+00:00",
        source_kind="synthetic_gpt_image_2_seed",
        camera_id="synthetic_head_pov",
        robot_profile_id="unitree_g1_sonic",
        task_id="turn_on_sink_handle",
        target_object_id="faucet_handle",
        depth_map_path=depth,
        target_segmentation_mask_path=target_mask,
        robot_mask_path=robot_mask,
        camera_intrinsics={
            "fx_pixels": 540.0,
            "fy_pixels": 540.0,
            "cx_pixels": 320.0,
            "cy_pixels": 180.0,
        },
        head_pose={"frame": "synthetic_head_pov", "pose_truth": False},
        target_bbox={"x_min": 0.44, "y_min": 0.30, "x_max": 0.56, "y_max": 0.48},
        target_keypoints={"handle_tip": {"x": 0.52, "y": 0.40}},
        affordance_points={"turn_handle_axis": {"center": {"x": 0.50, "y": 0.39}}},
        timestamp_ns=123456789,
        source_policy_action={
            "action_type": "unitree_g1_sonic_latent_action_chunk",
            "action_chunk": [0.1, -0.2, 0.3],
        },
        projected_skeleton_trace_path=trace,
    )

    assert manifest["schema_version"] == "wam_auxiliary_observation_manifest.v1"
    assert Path(manifest["manifest_path"]).is_file()
    assert manifest["modalities_available"]["rgb"] is True
    assert manifest["modalities_available"]["depth"] is True
    assert manifest["modalities_available"]["target_segmentation_mask"] is True
    assert manifest["modalities_available"]["robot_mask"] is True
    assert manifest["modalities_available"]["camera_intrinsics"] is True
    assert manifest["modalities_available"]["target_bbox"] is True
    assert manifest["modalities_available"]["target_keypoints"] is True
    assert manifest["modalities_available"]["affordance_points"] is True
    assert manifest["modalities_available"]["action_conditioning"] is True
    assert manifest["depth"]["depth_map_path_exists"] is True
    assert manifest["segmentation"]["target_segmentation_mask_path_exists"] is True
    assert manifest["action_conditioning"]["action_chunk_value_count"] == 3
    assert manifest["action_conditioning"]["projected_trace_path_exists"] is True
    assert manifest["claim_boundary"]["capture_truth"] is False
    assert manifest["claim_boundary"]["geometry_truth"] is False
    assert manifest["claim_boundary"]["collision_truth"] is False
    assert manifest["claim_boundary"]["synthetic_2d_sidecars_are_estimated_support_only"] is True
    assert (
        manifest["oscar_conditioning_support"][
            "raw_aux_modalities_consumed_by_public_oscar_entrypoint"
        ]
        is False
    )

    persisted = _read(tmp_path / "aux" / "wam_auxiliary_observation_manifest.json")
    assert persisted["task_id"] == "turn_on_sink_handle"
    summary = summarize_wam_auxiliary_observation_manifest(persisted)
    assert summary["available"] is True
    assert summary["modalities_available"]["depth"] is True
