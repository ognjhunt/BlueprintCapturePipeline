"""Where and how big the object was placed, against what the site's own video shows.

A synthetic rig with known truth: four cameras around a 31 x 52 x 4.5 cm tray on a table,
looking down at 35 degrees from 1.5 m. The masks are the true tray's silhouettes; the check is
given the placed box. Source frame = runtime = simulator here, so the registration is exact.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from blueprint_pipeline.website_task_placement_check import (
    DEFAULT_PLACEMENT_UNCERTAINTY_M, FLAG_PLACEMENT, FLAG_SIZE, MINIMUM_THRESHOLD_M, _fill, _hull, check_video_placement,
)

TRUE_MIN = np.array([-0.155, -0.26, 0.80])
TRUE_MAX = np.array([0.155, 0.26, 0.845])
K = np.array([[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]])


def _camera(azimuth_deg: float, elevation_deg: float = 35.0, distance: float = 1.5) -> np.ndarray:
    """world_from_camera for an OpenCV camera (+z forward, +y down) looking at the tray."""
    target = (TRUE_MIN + TRUE_MAX) / 2
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    position = target + distance * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, forward], axis=1)
    pose[:3, 3] = position
    return pose


def _rig():
    frames, masks = [], {}
    for index, azimuth in enumerate((-60.0, -20.0, 20.0, 60.0)):
        pose = _camera(azimuth)
        frames.append({"frame_id": f"f{index}", "intrinsics": K.tolist(), "world_from_camera": pose.tolist()})
        corners = np.array([[x, y, z, 1] for x in (TRUE_MIN[0], TRUE_MAX[0]) for y in (TRUE_MIN[1], TRUE_MAX[1])
                            for z in (TRUE_MIN[2], TRUE_MAX[2])])
        local = (np.linalg.inv(pose) @ corners.T).T[:, :3] @ K.T
        masks[f"f{index}"] = _fill(_hull(local[:, :2] / local[:, 2:3]), (720, 1280))
    return frames, masks


def _check(low, high, **kwargs):
    frames, masks = _rig()
    return check_video_placement(subject_min_sim=low, subject_max_sim=high, frames=frames, masks=masks,
                                 source_to_runtime=np.eye(4), runtime_to_simulator=np.eye(4), **kwargs)


def test_a_correct_placement_passes_with_matching_numbers() -> None:
    result = _check(TRUE_MIN, TRUE_MAX)
    summary = result["summary"]
    assert result["status"] == "assessed" and result["flags"] == [] and result["stops_preparation"] is False
    assert summary["median_overlap"] > 0.97
    assert summary["median_plane_offset_m"] < 0.003
    assert summary["triangulation_conditioning"] > 0.1    # cameras from four sides pin the centre down
    assert all(abs(value - 1.0) < 0.03 for value in summary["median_size_ratio"].values())
    assert set(summary["median_size_ratio"]) == {"x", "y"}


def test_a_placement_shifted_by_10_cm_is_flagged_in_one_direction() -> None:
    shift = np.array([0.10, 0.0, 0.0])
    result = _check(TRUE_MIN + shift, TRUE_MAX + shift)
    summary = result["summary"]
    assert FLAG_PLACEMENT in result["flags"] and FLAG_SIZE not in result["flags"]
    assert summary["median_plane_offset_m"] == pytest.approx(0.10, abs=0.01)
    assert summary["triangulated_offset_m"] == pytest.approx([-0.10, 0.0, 0.0], abs=0.01)
    assert summary["share_of_frames_offset_in_the_mean_direction"] == 1.0


def test_an_object_placed_20_percent_too_big_is_flagged_for_size() -> None:
    centre, half = (TRUE_MIN + TRUE_MAX) / 2, (TRUE_MAX - TRUE_MIN) / 2 * 1.2
    centre[2] = TRUE_MIN[2] + half[2]                     # still resting on the table
    result = _check(centre - half, centre + half)
    assert FLAG_SIZE in result["flags"]
    assert all(value == pytest.approx(1 / 1.2, abs=0.04) for value in result["summary"]["median_size_ratio"].values())


def test_the_threshold_is_the_stated_uncertainty_with_a_floor() -> None:
    shift = np.array([0.04, 0.0, 0.0])                    # 4 cm: under the 5 cm floor
    assert _check(TRUE_MIN + shift, TRUE_MAX + shift)["flags"] == []
    # A tiny stated uncertainty cannot flag every job: 3 cm reads as the 5 cm floor.
    tight = _check(TRUE_MIN + shift, TRUE_MAX + shift, placement_uncertainty_m=0.03)
    assert tight["flags"] == [] and tight["thresholds"]["placement_offset_m"] == MINIMUM_THRESHOLD_M
    wide = _check(TRUE_MIN + 2 * shift, TRUE_MAX + 2 * shift, placement_uncertainty_m=0.072)
    assert wide["flags"] == [FLAG_PLACEMENT] and wide["thresholds"]["placement_offset_m"] == 0.072
    assert wide["measures_room_placement"] is False and FLAG_PLACEMENT == "task_object_box_fit_offset"
    assert _check(TRUE_MIN, TRUE_MAX)["thresholds"]["placement_offset_m"] == DEFAULT_PLACEMENT_UNCERTAINTY_M


def test_a_mask_cut_by_the_frame_gives_no_centre_or_size_and_too_few_frames_is_not_assessed() -> None:
    frames, masks = _rig()
    masks = {key: value.copy() for key, value in masks.items()}
    masks["f0"][:, :5] = True                              # touches the left border
    result = check_video_placement(subject_min_sim=TRUE_MIN, subject_max_sim=TRUE_MAX, frames=frames, masks=masks,
                                   source_to_runtime=np.eye(4), runtime_to_simulator=np.eye(4))
    first = next(row for row in result["frames"] if row["frame_id"] == "f0")
    assert first["mask_cut_by_frame"] is True and "centre_offset_m" not in first and "size_ratio" not in first
    few = check_video_placement(subject_min_sim=TRUE_MIN, subject_max_sim=TRUE_MAX, frames=frames[:2],
                                masks=masks, source_to_runtime=np.eye(4), runtime_to_simulator=np.eye(4))
    assert few["status"] == "not_assessed_too_few_masked_frames" and few["flags"] == []


def test_preparation_records_the_check_and_never_stops_on_it() -> None:
    from blueprint_pipeline.website_task_placement_check import box_fit_record as video_placement_check

    frames, masks = _rig()
    from blueprint_pipeline.website_track_mask_codec import decode_track_mask  # noqa: F401  (codec used inside)
    observations = []
    for frame_id, mask in masks.items():
        edges = np.diff(np.pad(mask.reshape(-1).astype(np.int8), (1, 1)))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        observations.append({"source_frame_id": frame_id, "width": 1280, "height": 720,
                             "runs": [{"start": int(a), "length": int(b - a)} for a, b in zip(starts, ends)]})
    shift = np.array([0.10, 0.0, 0.0])
    preparation = {"subject": {"aabb_min_xyz": (TRUE_MIN + shift).tolist(), "aabb_max_xyz": (TRUE_MAX + shift).tolist()},
                   "registration": {"source_to_runtime": np.eye(4).tolist()},
                   "coordinate_frame": {"runtime_to_simulator": np.eye(4).tolist(), "placement_uncertainty_m": 0.072}}
    record = video_placement_check(preparation, track={"observations": observations}, frames=frames)
    assert record["flags"] == [FLAG_PLACEMENT] and record["thresholds"]["placement_offset_m"] == 0.072
    broken = video_placement_check({"subject": {}}, track={"observations": observations}, frames=frames)
    # Any failure at all, not only the four kinds it used to catch, is recorded and never stops the job.
    import blueprint_pipeline.website_task_placement_check as check
    def explode(**_kwargs):
        raise RuntimeError("anything")
    original, check.check_video_placement = check.check_video_placement, explode
    try:
        assert video_placement_check(preparation, track={"observations": observations},
                                     frames=frames)["status"] == "not_assessed"
    finally:
        check.check_video_placement = original
    assert broken["status"] == "not_assessed" and broken["flags"] == [] and broken["stops_preparation"] is False


def test_the_stated_placement_uncertainty_and_the_box_fit_reach_the_published_result(tmp_path) -> None:
    import json

    from blueprint_pipeline.atlas_scene_estimates import (
        PLACEMENT_UNCERTAINTY, SUPPORT_HEIGHT, TASK_OBJECT_BOX_FIT, preparation_flags,
    )
    from blueprint_pipeline.beta_policy_screening_publication_flags import scene_feasibility

    path = tmp_path / "preparation.json"
    path.write_text(json.dumps({"coordinate_frame": {"placement_uncertainty_m": 0.072},
                                "video_box_fit_check": {"flags": ["task_object_box_fit_offset"]},
                                "support_height_agreement": {"flags": [SUPPORT_HEIGHT]}}))
    codes = preparation_flags(path)
    assert codes == {PLACEMENT_UNCERTAINTY, TASK_OBJECT_BOX_FIT, SUPPORT_HEIGHT}
    # The plan carries them as scene_estimate_flags; the publication lists them in scene_feasibility.
    assert scene_feasibility([], estimates=sorted(codes)) == {"status": "flagged", "flags": sorted(codes)}
    path.write_text(json.dumps({"coordinate_frame": {"placement_uncertainty_m": 0.03}}))
    assert preparation_flags(path) == set()   # under the 5 cm floor
