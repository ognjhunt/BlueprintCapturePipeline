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
