from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import binary_dilation

from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_object_removal import prepare_object_removal_frames, select_reconstruction_frames


def _source(tmp_path):
    path = tmp_path / "original.png"
    pixels = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    Image.fromarray(pixels).save(path)
    # No depth or camera data is required for preparing image edits.
    return {"frame_id": "f0", "source_image_path": str(path), "source_image_digest": _sha256_file(path),
            "display_rotation_degrees": 90}


def _target(x, y, effect="manipulated", disposition="remove"):
    return {"task_effect": effect, "disposition": disposition,
            "source_track": {"observations": [{"source_frame_id": "f0", "height": 16, "width": 12,
                "runs": [{"start": y * 12 + x, "length": 2}, {"start": (y + 1) * 12 + x, "length": 2}]}]}}


def test_multiple_task_objects_share_one_mask_without_removing_unrelated_objects(tmp_path):
    frame = _source(tmp_path)
    targets = [_target(2, 2), _target(8, 12), _target(5, 7, "unrelated", "keep"),
               _target(5, 7, "static_contact", "keep")]
    result = prepare_object_removal_frames(frames=[frame], task_masks={"targets": targets}, output_root=tmp_path / "edit")[0]
    expected = np.zeros((16, 12), dtype=bool)
    expected[2:4, 2:4] = True
    expected[12:14, 8:10] = True
    expected = binary_dilation(expected, iterations=3)
    np.testing.assert_array_equal(np.asarray(Image.open(result["remaining_mask_path"])) == 255, expected)
    assert not expected[7, 5]
    original = Image.open(frame["source_image_path"]).rotate(90, expand=True)
    np.testing.assert_array_equal(np.asarray(Image.open(result["image_path"])), np.asarray(original))
    assert result["remaining_pixel_count"] == int(expected.sum())
    assert result["generated_pixels_present"] is False
    assert result["edge_feather_pixels"] == 1
    assert _sha256_file(Path(frame["source_image_path"])) == frame["source_image_digest"]


def test_unchanged_view_needs_no_generation(tmp_path):
    frame = _source(tmp_path)
    result = prepare_object_removal_frames(frames=[frame], task_masks={"targets": []}, output_root=tmp_path / "edit")[0]
    assert result["remaining_pixel_count"] == 0
    assert result["original_image_digest"] == result["image_digest"]


@pytest.mark.parametrize("fault", ["source", "mask"])
def test_invalid_source_or_pixel_mapping_cannot_prepare_an_edit(tmp_path, fault):
    frame, target = _source(tmp_path), _target(2, 2)
    if fault == "source":
        Path(frame["source_image_path"]).write_bytes(b"changed")
    else:
        target["source_track"]["observations"][0]["width"] = 13
    with pytest.raises(ValueError, match="source_changed|source_mask_mismatch"):
        prepare_object_removal_frames(frames=[frame], task_masks={"targets": [target]}, output_root=tmp_path / "edit")


@pytest.mark.parametrize("target_frames", [[], ["f5"], ["f5", "f8"]])
@pytest.mark.parametrize("limit,total", [(8, 13), (128, 140), (128, 13)])
def test_reconstruction_includes_context_without_target_and_avoids_duplicates(tmp_path, target_frames, limit, total):
    frames = []
    for index in range(total):
        path = tmp_path / f"view-{index}.png"
        pixels = np.random.default_rng(index).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        frames.append({"frame_id": f"f{index}", "image_path": str(path), "image_digest": _sha256_file(path),
                       "remaining_pixel_count": 20 if index in (5, 8) else 0})
    frames[12] = {**frames[0], "frame_id": "f12"}
    masks = {"targets": [{"task_effect": "manipulated", "track": {"observations": [
        {"source_frame_id": frame} for frame in target_frames]}}]}
    selected = select_reconstruction_frames(frames=frames, task_masks=masks, limit=limit)
    assert len(selected) == min(limit, total - 1)
    assert set(target_frames) <= {f["frame_id"] for f in selected}
    assert len({f["image_digest"] for f in selected}) == len(selected)
    if len(target_frames) == 2:
        assert sum(f["remaining_pixel_count"] > 0 for f in selected) == 2
    assert select_reconstruction_frames(frames=frames, task_masks=masks, limit=limit) == selected
    Path(frames[3]["image_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source_changed"):
        select_reconstruction_frames(frames=frames, task_masks=masks, limit=limit)


def test_large_provider_budget_decodes_context_independently_of_geometry_batch(tmp_path, monkeypatch):
    from blueprint_pipeline import website_object_removal as module
    from blueprint_pipeline.website_reconstruction_profile import reconstruction_profile
    video = tmp_path / "video.mov"
    video.write_bytes(b"bound video")
    registry = [{"source_frame_id": f"decoded-{i:09d}", "decoded_pts_seconds": i / 30} for i in range(140)]
    existing = [{"frame_id": registry[i]["source_frame_id"], "timestamp_seconds": i / 30,
                 "display_rotation_degrees": -90} for i in (0, 139)]
    geometry = {"frames": existing, "binding": {"source_video_digest": _sha256_file(video)}}
    masks = {"source_video_digest": _sha256_file(video), "source_frame_registry": registry}
    calls = []
    def extract(**kwargs):
        calls.append(kwargs)
        return [{"frame_id": registry[i]["source_frame_id"], "t_video_sec": i / 30, "digest": f"digest-{i}"}
                for i in kwargs["indexes"]]
    monkeypatch.setattr(module, "_extract_frames", extract)
    monkeypatch.setattr(module.shutil, "which", lambda _: "/ffmpeg")
    profile = reconstruction_profile({"provider": "future_provider", "model": "future_model", "max_input_images": 128})
    frames = module.reconstruction_source_frames(source_geometry=geometry, task_masks=masks,
        source_video=video, limit=profile["max_input_images"], output_root=tmp_path / "frames")
    assert len(frames) == 128
    assert len(geometry["frames"]) == 2
    assert len(calls[0]["indexes"]) == 126
    assert all(f["display_rotation_degrees"] == -90 for f in frames)
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="video_changed"):
        module.reconstruction_source_frames(source_geometry=geometry, task_masks=masks,
            source_video=video, limit=128, output_root=tmp_path / "frames")


def test_provider_profile_selects_capacity_without_silently_reusing_marble_limits(monkeypatch):
    import json
    from blueprint_pipeline.website_reconstruction_profile import reconstruction_profile
    monkeypatch.setenv("WORLDLABS_DEFAULT_MODEL", "future-model")
    monkeypatch.delenv("BLUEPRINT_WEBSITE_RECONSTRUCTION_PROFILE_JSON", raising=False)
    with pytest.raises(ValueError, match="profile_required"):
        reconstruction_profile()
    profile = {"provider": "world_labs", "model": "future-model", "max_input_images": 128}
    monkeypatch.setenv("BLUEPRINT_WEBSITE_RECONSTRUCTION_PROFILE_JSON", json.dumps(profile))
    assert reconstruction_profile() == profile
    monkeypatch.setenv("WORLDLABS_DEFAULT_MODEL", "different-model")
    with pytest.raises(ValueError, match="model_mismatch"):
        reconstruction_profile()
