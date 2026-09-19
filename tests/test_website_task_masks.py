from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.website_task_masks import decode_track_mask, estimate_target_bounds, select_task_track
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def _track(identifier="cup-1", *, start=0, label="task-cup"):
    return {"track_id": identifier, "label": label, "observations": [
        {"source_frame_id": "frame-0", "height": 4, "width": 4,
         "runs": [{"start": start, "length": 2, "probability": 0.9},
                  {"start": start + 4, "length": 2, "probability": 0.9}]}]}


def _target():
    return {"target_id": "task-cup", "spatial_evidence": [
        {"timestamp_seconds": 0, "box_xywh_normalized": [0, 0, 0.5, 0.5]}]}


def test_same_class_neighbor_is_not_removed():
    selected = select_task_track(target=_target(), tracks=[_track("wrong", start=2), _track("right")],
                                 frames=[{"frame_id": "frame-0", "timestamp_seconds": 0}])
    assert selected["track_id"] == "right"


@pytest.mark.parametrize("tracks", [[], [_track(start=2)], [_track("a"), _track("b")], [_track(label="other-target")]])
def test_missing_or_ambiguous_instance_holds_editing(tracks):
    with pytest.raises(ValueError, match="track_ambiguous"):
        select_task_track(target=_target(), tracks=tracks, frames=[{"frame_id": "frame-0", "timestamp_seconds": 0}])


def test_masks_cannot_escape_the_image():
    observation = _track()["observations"][0]
    observation["runs"][0]["length"] = 17
    with pytest.raises(ValueError, match="mask_run_invalid"):
        decode_track_mask(observation)


def test_masked_geometry_preserves_world_position_and_estimated_scale(tmp_path):
    geometry_path = tmp_path / "geometry.npz"
    np.savez_compressed(geometry_path, depth_m=np.full((4, 4), 2.0), valid_mask=np.ones((4, 4), dtype=bool))
    pose = np.eye(4)
    pose[:3, 3] = [10, 20, 30]
    frame = {"frame_id": "frame-0", "geometry_path": str(geometry_path),
             "geometry_digest": _sha256_file(geometry_path), "intrinsics": np.eye(3).tolist(),
             "world_from_camera": pose.tolist()}
    bounds = estimate_target_bounds(_track(), [frame])
    assert bounds["center"] == [11, 21, 32]
    assert bounds["unit"] == "estimated_meters"
    assert bounds["metric_measurement_proven"] is False
    assert bounds["complete_object_dimensions"] is False
    geometry_path.write_bytes(b"changed geometry")
    with pytest.raises(ValueError, match="geometry_changed"):
        estimate_target_bounds(_track(), [frame])


def test_hosted_masks_use_original_upright_pixels_and_bind_to_depth(tmp_path, monkeypatch):
    from PIL import Image
    from blueprint_pipeline.website_task_masks import run_website_task_masks

    original = tmp_path / "original.png"
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    pixels[:2, 4:] = [0, 0, 255]
    Image.fromarray(pixels).save(original)
    geometry_path = tmp_path / "geometry.npz"
    np.savez_compressed(geometry_path, depth_m=np.ones((4, 2)), valid_mask=np.ones((4, 2), dtype=bool))
    frame = {"frame_id": "frame-0", "timestamp_seconds": 0,
             "source_image_path": str(original), "source_image_digest": _sha256_file(original),
             "display_rotation_degrees": 90, "width": 2, "height": 4,
             "geometry_path": str(geometry_path), "geometry_digest": _sha256_file(geometry_path),
             "intrinsics": np.eye(3).tolist(), "world_from_camera": np.eye(4).tolist()}
    target = {**_target(), "semantic_label": "small container beside picture", "segmentation_prompt": "blue object",
              "task_effect": "manipulated", "disposition": "remove"}
    calls = []

    def hosted(**kwargs):
        calls.append(kwargs)
        assert kwargs["prompts"][0]["text"] == "blue object"
        assert (kwargs["frame_registry"][0]["width"], kwargs["frame_registry"][0]["height"]) == (4, 8)
        with Image.open(kwargs["frame_artifacts"][0]["path"]) as submitted:
            assert submitted.size == (4, 8)
            assert submitted.getpixel((0, 0))[2] > 240
        return {"tracks": [{"track_id": "selected", "label": "task-cup", "observations": [
            {"source_frame_id": "frame-0", "width": 4, "height": 8,
             "runs": [{"start": y * 4, "length": 2} for y in range(4)]}]}]}

    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.run_meta_sam31", hosted)
    kwargs = dict(plan={"targets": [target], "task_context_sha256": "task"},
                  source_geometry={"digest": "geometry", "frames": [frame],
                                   "binding": {"source_video_digest": "video"}}, output_root=tmp_path / "masks")
    result = run_website_task_masks(**kwargs)
    observation = result["targets"][0]["track"]["observations"][0]
    np.testing.assert_array_equal(decode_track_mask(observation), [[True, False], [True, False], [False, False], [False, False]])
    assert observation["source_mask_width"] == 4
    assert result["targets"][0]["estimated_visible_bounds"]["metric_measurement_proven"] is False
    original.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source_frame_changed"):
        run_website_task_masks(**kwargs)
    assert len(calls) == 1
