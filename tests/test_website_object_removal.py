from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import binary_dilation

from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_object_removal import prepare_object_removal_frames


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
    expected = binary_dilation(expected, iterations=2)
    np.testing.assert_array_equal(np.asarray(Image.open(result["remaining_mask_path"])) == 255, expected)
    assert not expected[7, 5]
    original = Image.open(frame["source_image_path"]).rotate(90, expand=True)
    np.testing.assert_array_equal(np.asarray(Image.open(result["image_path"])), np.asarray(original))
    assert result["remaining_pixel_count"] == int(expected.sum())
    assert result["generated_pixels_present"] is False
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
