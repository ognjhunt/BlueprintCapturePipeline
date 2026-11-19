"""An object's size from its own points when its masks spill onto what lies under or behind it.

Hermetic: the ray-cast room of ``test_website_unscanned_surface`` (a cup on a table by a wall, six frames, compile
on its exact registration). Spilled masks are the cup's grown over the table top and the wall behind it, as test
job 4's lamp masks took in the round table under it in 3 of 33 frames. No provider is reached; nothing is paid for.
"""
# Covers: src/blueprint_pipeline/website_object_extent.py
# Covers: src/blueprint_pipeline/website_task_preparation.py
# Covers: src/blueprint_pipeline/website_packet_facts.py
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from blueprint_pipeline import website_object_extent as extent
from blueprint_pipeline import website_task_preparation as preparation
from blueprint_pipeline.atlas_scene_estimates import (OBJECT_SIZE_DISAGREES_WITH_PUBLISHED, OBJECT_SIZE_FROM_ITS_POINTS,
                                                      preparation_flags)
from blueprint_pipeline.website_task_masks import estimate_target_bounds  # noqa: F401
from tests.test_website_unscanned_surface import (CUP, FLIP, RUNTIME_SCALE, RUNTIME_TRANSLATION, SOURCE, _arguments,
                                                  _track, _views, exact_registration, no_model)

__all__ = ["exact_registration", "no_model"]  # the room's own registration, and no model call, for every test here
CUP_SIZE = np.subtract(CUP[1], CUP[0])  # 0.12 m on each side
#: Six views of the cup on its table from in front and both sides, about a metre away: it is 70 to 110 pixels
#: across, as test job 4's lamp was 90 to 140 in its frames, enough for the box-fit check to measure its size.
CENTRE = tuple(np.mean(CUP, axis=0))
CAMERAS = (((-1.4, 1.35, 2.0), CENTRE), ((-1.1, 1.45, 1.7), CENTRE), ((-0.7, 1.5, 1.55), CENTRE),
           ((-0.3, 1.45, 1.6), CENTRE), ((0.05, 1.4, 1.85), CENTRE), ((0.2, 1.3, 2.3), CENTRE))
FOCAL = 600.0


def _matrix(flipped: bool = False) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = RUNTIME_SCALE * (FLIP if flipped else np.eye(3)) @ SOURCE, RUNTIME_TRANSLATION
    return matrix


RUNTIME_TO_SIM = np.diag([1 / RUNTIME_SCALE] * 3 + [1.0]) @ np.array(
    [[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])
#: Compile's simulator frame for a collider whose up is -Y.
RUNTIME_TO_SIM_FLIPPED = np.diag([1 / RUNTIME_SCALE] * 3 + [1.0]) @ np.array(
    [[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, -1.0, 0, 0], [0, 0, 0, 1.0]])


def _masks(spilled: tuple[int, ...] = ()) -> list:
    """The cup's masks, grown 40 pixels over the table and what is behind it in ``spilled`` frames."""
    return [ndimage.binary_dilation(view[4], iterations=40) if index in spilled else view[4]
            for index, view in enumerate(_views(False, CAMERAS, FOCAL))]


def _room(root, spilled: tuple[int, ...] = (), flipped: bool = False) -> dict:
    return _arguments(root, cup_masks=_masks(spilled), cameras=CAMERAS, focal=FOCAL, onto_table=True, flipped=flipped)


def _cup(spilled: tuple[int, ...] = ()) -> dict:
    return {"target_id": "cup-1", "track": _track("cup", _masks(spilled))}


def _raw(target, frames, flipped: bool = False):
    bounds = estimate_target_bounds(target["track"], frames, source_to_target=_matrix(flipped))
    return bounds["minimum"], bounds["maximum"]


def _dimensions(low, high) -> np.ndarray:
    return np.abs(np.asarray(high) - np.asarray(low)) / RUNTIME_SCALE  # runtime units to metres, axis by axis


def test_a_mask_spilling_onto_the_table_and_the_wall_behind_is_cleaned_to_the_object_s_own_points(tmp_path):
    frames = _room(tmp_path)["source_geometry"]["frames"]
    target = _cup(spilled=(0, 3))
    low, high = _raw(target, frames)
    cleaned_low, cleaned_high, record = extent.own_points_extent(low, high, target, frames, _matrix(),
                                                                 RUNTIME_TO_SIM, 0.01)
    row = record["subject_extent"]
    assert row["status"] == "from_its_own_points" and row["flag"] == OBJECT_SIZE_FROM_ITS_POINTS
    assert "task_object_box_fit_size" in row["before"]["box_fit"]["flags"]
    assert "task_object_box_fit_size" not in row["after"]["box_fit"]["flags"]
    before, after = _dimensions(low, high), _dimensions(cleaned_low, cleaned_high)
    assert (before - CUP_SIZE).max() > 0.1  # the spill made it far too big
    # The cup's own points: about its 12 cm, never shrunk below what they show.
    assert np.all(np.abs(after - CUP_SIZE) <= 0.03), after
    assert sorted(row["after"]["dimensions_m"]) == pytest.approx(sorted(after), abs=1e-9)
    frames_rows = {item["frame_id"]: item for item in row["frames"]}
    spilled = [frames_rows[f"frame-{index}"] for index in (0, 3)]
    # Every step acted where the mask spilled: edges eroded, the wall behind dropped, the table voted out.
    assert all(item["off_edges"] < item["with_depth"] for item in spilled)
    assert any(item["nearest_surface"] < item["off_edges"] for item in spilled)
    assert all(item["agreed"] < item["tested"] for item in spilled)


def test_an_object_masked_correctly_keeps_its_box_and_records_nothing(tmp_path):
    frames = _room(tmp_path)["source_geometry"]["frames"]
    target = _cup()
    low, high = _raw(target, frames)
    kept_low, kept_high, record = extent.own_points_extent(low, high, target, frames, _matrix(), RUNTIME_TO_SIM, 0.01)
    assert (kept_low, kept_high, record) == (list(low), list(high), {})


def _published(match: str, depth: float, width: float, height: float):
    def size(object_spec, *, max_relative_half_range):
        assert object_spec == {"stand-in": True} and max_relative_half_range == extent.DIMENSION_RELATIVE_ERROR
        return {"depth_m": depth, "width_m": width, "height_m": height, "dimension_match": match,
                "dimension_authority": "published_product_specification", "object_spec_digest": "sha256:" + "4" * 64}
    return size


@pytest.mark.parametrize("match, dims, status", [
    ("exact_model", (0.5, 0.45, 0.9), "disagrees"), ("exact_model", (0.12, 0.13, 0.12), "agrees"),
    ("model_family", (0.5, 0.45, 0.9), None)])
def test_an_exact_model_s_published_size_is_a_cross_check_flagged_never_used(tmp_path, monkeypatch, match, dims,
                                                                            status):
    from blueprint_pipeline import website_object_spec_research as research
    monkeypatch.setattr(research, "published_body_size", _published(match, *dims))
    frames = _room(tmp_path)["source_geometry"]["frames"]
    target = {**_cup(), "object_spec": {"stand-in": True}}
    low, high = _raw(target, frames)
    kept_low, kept_high, record = extent.own_points_extent(low, high, target, frames, _matrix(), RUNTIME_TO_SIM, 0.01)
    assert (kept_low, kept_high) == (list(low), list(high))  # the video's box, whatever is published
    if status is None:
        assert record == {}
        return
    check = record["subject_extent"]["published_cross_check"]
    assert check["status"] == status and check["used_as_size"] is False
    assert ("flag" in check) is (status == "disagrees")
    path = tmp_path / "preparation.json"
    path.write_text(preparation.json.dumps(record))
    assert (OBJECT_SIZE_DISAGREES_WITH_PUBLISHED in preparation_flags(path)) is (status == "disagrees")
    assert OBJECT_SIZE_FROM_ITS_POINTS not in preparation_flags(path)


@pytest.mark.parametrize("spilled", [(), (0, 3)])
def test_compile_keeps_identical_boxes_without_spill_and_says_where_a_spilled_size_came_from(tmp_path, monkeypatch,
                                                                                             spilled):
    from blueprint_pipeline.website_packet_facts import measured_and_estimated
    args = _room(tmp_path / "with", spilled)
    value = preparation.compile_website_scene_preparation(**args)
    monkeypatch.setattr(extent, "own_points_extent", lambda low, high, *_: (list(low), list(high), {}))
    without = preparation.compile_website_scene_preparation(**{**args, "output_root": tmp_path / "without"})
    line = next(row["text"] for row in measured_and_estimated(value, None, None)[1] if row["text"].startswith("Object size"))
    if not spilled:
        assert "subject_extent" not in value and value["subject"] == without["subject"]
        assert "own points" not in line
        return
    assert value["subject_extent"]["status"] == "from_its_own_points"
    assert OBJECT_SIZE_FROM_ITS_POINTS in preparation_flags(tmp_path / "with" / "out" / "preparation.json")
    size = np.subtract(value["subject"]["aabb_max_xyz"], value["subject"]["aabb_min_xyz"])
    spill = np.subtract(without["subject"]["aabb_max_xyz"], without["subject"]["aabb_min_xyz"])
    assert np.all(np.abs(size - CUP_SIZE[[0, 2, 1]]) <= 0.03) and (spill - size).max() > 0.1
    assert "from its own points that the frames agree on" in line and line.startswith("Object size: ")


def test_the_spill_is_cleaned_the_same_in_a_collider_whose_up_is_minus_y(tmp_path):
    frames = _room(tmp_path, flipped=True)["source_geometry"]["frames"]
    target = _cup(spilled=(0, 3))
    low, high = _raw(target, frames, flipped=True)
    cleaned_low, cleaned_high, record = extent.own_points_extent(low, high, target, frames, _matrix(True),
                                                                 RUNTIME_TO_SIM_FLIPPED, 0.01)
    assert record["subject_extent"]["status"] == "from_its_own_points"
    assert np.all(np.abs(_dimensions(cleaned_low, cleaned_high) - CUP_SIZE) <= 0.03)


def test_a_size_the_frames_cannot_measure_or_too_few_views_keep_the_old_box(tmp_path, monkeypatch):
    frames = _room(tmp_path)["source_geometry"]["frames"]
    target = _cup(spilled=(0, 3))
    low, high = _raw(target, frames)
    real_fit, fits = extent._fit, []
    monkeypatch.setattr(extent, "outline_hull", lambda *args, **kwargs: None)  # the points read alone, as tested here

    def unmeasured(*args, **kwargs):  # the new box too small in every frame for the check to measure its size
        fits.append(1)
        found = real_fit(*args, **kwargs)
        return found if len(fits) == 1 else {**found, "median_size_ratio": {}, "size_error": None}
    monkeypatch.setattr(extent, "_fit", unmeasured)
    kept_low, kept_high, record = extent.own_points_extent(low, high, target, frames, _matrix(), RUNTIME_TO_SIM, 0.01)
    assert (kept_low, kept_high) == (list(low), list(high))
    assert record["subject_extent"]["status"] == "kept"
    assert record["subject_extent"]["reason"] == "own_points_size_not_measured_in_the_frames"
    assert "flag" not in record["subject_extent"]
    monkeypatch.setattr(extent, "_fit", real_fit)
    real_points = extent.own_points

    def two_views(*args):  # fewer views than the box-fit check measures a size from
        kept, rows = real_points(*args)
        return [points if index < 2 else points[:0] for index, points in enumerate(kept)], rows
    monkeypatch.setattr(extent, "own_points", two_views)
    kept_low, kept_high, record = extent.own_points_extent(low, high, target, frames, _matrix(), RUNTIME_TO_SIM, 0.01)
    assert (kept_low, kept_high) == (list(low), list(high))
    assert record["subject_extent"]["reason"] == "too_few_views_or_points_of_the_object"


@pytest.mark.parametrize("spilled", [(), (0, 3), (0, 1, 3)])
def test_the_outline_carve_bounds_the_object_from_above_and_cuts_the_spill(tmp_path, spilled):
    """Every point of the cup lies inside every view's outline, so the carve never cuts it; a band of table or wall
    a few masks took in lies outside the others' outlines, so it is carved away. Masks only: depth plays no part."""
    frames = _room(tmp_path)["source_geometry"]["frames"]
    target = _cup(spilled=spilled)
    low, high = _raw(target, frames)
    hull = extent.outline_hull(low, high, target["track"], frames, _matrix())
    carved = _dimensions(hull["minimum"], hull["maximum"])
    assert np.all(carved >= CUP_SIZE - 0.005), carved  # never below the object, within a cell
    # The spill is gone; along the six views' lines of sight (about 45 degrees apart) a carve stays loose.
    assert np.all(carved <= CUP_SIZE + 0.15), carved
    if spilled:
        assert np.all(carved < _dimensions(low, high))


def test_too_few_frames_carve_nothing(tmp_path):
    frames = _room(tmp_path)["source_geometry"]["frames"]
    target = _cup()
    track = {**target["track"], "observations": target["track"]["observations"][:1]}
    low, high = _raw(target, frames)
    assert extent.outline_hull(low, high, track, frames, _matrix()) is None


def test_a_view_that_cannot_see_part_of_the_object_does_not_carve_it(tmp_path, monkeypatch):
    """One view's mask misses part of the cup (an occluder in front, nearer in its depth there): that view cannot
    see those cells and does not vote on them, so the carve keeps the cup whole."""
    frames = _room(tmp_path)["source_geometry"]["frames"]
    target = _cup()
    low, high = _raw(target, frames)
    observations = [dict(row) for row in target["track"]["observations"]]
    mask = extent.decode_track_mask(observations[2])
    columns = np.nonzero(mask.any(axis=0))[0]
    hidden = np.zeros_like(mask)
    hidden[:, columns[0]:(columns[0] + columns[-1]) // 2] = True  # its left half hidden in this view
    observations[2]["runs"] = extent.mask_runs(mask & ~hidden)
    real = extent._frame_depth

    def nearer_there(frame):
        depth = real(frame)
        if frame["frame_id"] == observations[2]["source_frame_id"] and depth is not None:
            depth = np.where(hidden, 0.1, depth)  # the occluder, 10 cm from the camera
        return depth

    monkeypatch.setattr(extent, "_frame_depth", nearer_there)
    track = {**target["track"], "observations": observations}
    hull = extent.outline_hull(low, high, track, frames, _matrix())
    assert np.all(_dimensions(hull["minimum"], hull["maximum"]) >= CUP_SIZE - 0.005)
