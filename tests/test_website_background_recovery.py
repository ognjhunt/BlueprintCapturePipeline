import numpy as np
from PIL import Image

from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_background_recovery import recover_observed_background, removal_masks


def _frames(tmp_path):
    frames = []
    for index, color in enumerate(("red", "green")):
        image, geometry = tmp_path / f"{index}.png", tmp_path / f"{index}.npz"
        Image.new("RGB", (4, 4), color).save(image)
        np.savez_compressed(geometry, depth_m=np.full((4, 4), index + 1.0), valid_mask=np.ones((4, 4), dtype=bool))
        frames.append({"frame_id": f"f{index}", "image_path": str(image), "image_digest": _sha256_file(image),
                       "geometry_path": str(geometry), "geometry_digest": _sha256_file(geometry),
                       "width": 4, "height": 4, "intrinsics": np.eye(3).tolist(),
                       "world_from_camera": np.eye(4).tolist(), "camera_from_world": np.eye(4).tolist()})
    return frames


def _masks():
    return {"targets": [{"task_effect": "manipulated", "disposition": "remove", "track": {"observations": [
        {"source_frame_id": "f0", "height": 4, "width": 4, "runs": [{"start": 0, "length": 1}]}]}}]}


def test_recovery_only_copies_observed_background_and_preserves_every_other_pixel(tmp_path):
    frames, masks = _frames(tmp_path), _masks()
    results = recover_observed_background(frames=frames, task_masks=masks, output_root=tmp_path / "prepared")
    original = np.asarray(Image.open(frames[0]["image_path"]))
    edited = np.asarray(Image.open(results[0]["image_path"]))
    removed = removal_masks(masks, frames)["f0"]
    assert np.array_equal(edited[~removed], original[~removed])
    assert np.all(edited[removed] == [0, 128, 0])
    assert results[0]["remaining_pixel_count"] == 0
    assert results[0]["recovered_pixel_count"] == int(removed.sum())
    assert results[0]["generated_pixels_present"] is False
    assert results[0]["metric_measurement_proven"] is False


def test_unseen_background_stays_an_explicit_hole(tmp_path):
    frames, masks = _frames(tmp_path), _masks()
    masks["targets"][0]["track"]["observations"].append(
        {**masks["targets"][0]["track"]["observations"][0], "source_frame_id": "f1"})
    results = recover_observed_background(frames=frames, task_masks=masks, output_root=tmp_path / "prepared")
    assert results[0]["recovered_pixel_count"] == 0
    assert results[0]["remaining_pixel_count"] > 0


def test_static_supports_and_unrelated_movables_are_preserved(tmp_path):
    frames, masks = _frames(tmp_path), _masks()
    masks["targets"][0].update(task_effect="static_contact", disposition="keep")
    results = recover_observed_background(frames=frames, task_masks=masks, output_root=tmp_path / "prepared")
    for frame, result in zip(frames, results):
        assert np.array_equal(np.asarray(Image.open(frame["image_path"])), np.asarray(Image.open(result["image_path"])))
        assert result["remaining_pixel_count"] == result["recovered_pixel_count"] == 0
