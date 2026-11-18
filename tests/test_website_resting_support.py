"""Internal test job 4 (2026-11-18): the surface an object visibly rests on (``website_resting_support``).

Job 4 ("move the lamp", hotel-room footage) stopped on ``support_surface_not_found_under_subject``. Its
lamp's masks spilled onto the round table it stood on and a few of its lifted points fell below it, so
its box reached 22 cm under the table top; the search looked 7.5 cm up (the residual), and three of the
box's four corners missed the round top. The fixtures below are built from the job's recorded numbers
(the replay on its own collider, frames and masks is in the PR): the lamp's box bottom at 0.539 m, the
table top 0.226 m above it, the box-fit check's flags and overlap (0.187), the named table's visible top
0.259 m above the bottom, and the 7.5 cm residual.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_resting_support.py
#   src/blueprint_pipeline/website_support_geometry.py
#   src/blueprint_pipeline/website_task_masks.py
#   src/blueprint_pipeline/website_task_preparation.py
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from blueprint_pipeline import website_resting_support as resting
from blueprint_pipeline import website_task_preparation as preparation
from blueprint_pipeline.atlas_scene_estimates import SUPPORT_GUESSED, preparation_flags
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_support_geometry import (OUTLINE_BASIS, nearest_support_under,
                                                         resting_surface_under, same_support_surface, support_under)

RESIDUAL_M = 0.075  # job 4's registration residual near the task
TABLE_TOP_M, TABLE_RADIUS_M, TABLE_CENTRE = 0.765, 0.23, np.array([0.25, -0.15])
FLOOR_TOP_M = 0.10  # flat, within the nearest-surface guess's 0.5 m under the lamp's estimated bottom
RECORDED_BOX_FIT = {"status": "assessed", "flags": ["task_object_box_fit_offset", "task_object_box_fit_size"],
                    "summary": {"median_overlap": 0.1866721136943227,
                                "median_size_ratio": {"x": 0.6434749303986047, "y": 0.46733140755294217}}}


def _room(*, floor: bool = True, table: bool = True) -> trimesh.Trimesh:
    parts = []
    if table:
        top = trimesh.creation.cylinder(radius=TABLE_RADIUS_M, height=0.03, sections=64)
        top.apply_translation([*TABLE_CENTRE, TABLE_TOP_M - 0.015])
        parts.append(top)
    if floor:
        slab = trimesh.creation.box(extents=[3.0, 3.0, 0.02],
                                    transform=trimesh.transformations.translation_matrix([0, 0, FLOOR_TOP_M - 0.01]))
        for _ in range(5):  # a reconstruction's floor is many small triangles
            slab = slab.subdivide()
        parts.append(slab)
    return trimesh.util.concatenate(parts)


def _lamp_views(count: int = 6) -> list[np.ndarray]:
    """A lamp on the table, as the frames lift it: its stem and shade, its mask's spill onto the table top
    around it, and a thin tail of points that fell below the table (2%), which sets the box's bottom."""
    rng = np.random.default_rng(4)
    centre = TABLE_CENTRE + [0.03, 0.02]
    views = []
    for _ in range(count):
        angle, radius = rng.uniform(0, 2 * np.pi, 900), 0.205 * np.sqrt(rng.uniform(0, 1, 900))
        spill = np.c_[centre + np.c_[np.cos(angle), np.sin(angle)] * radius[:, None],
                      TABLE_TOP_M + rng.normal(0, 0.004, 900)]
        height = rng.uniform(0.80, 1.21, 2700)
        shade = np.where(height > 0.95, 0.08 + 0.04 * (height - 0.95) / 0.26, 0.02)
        angle = rng.uniform(0, 2 * np.pi, 2700)
        body = np.c_[centre + np.c_[np.cos(angle), np.sin(angle)] * shade[:, None], height]
        tail = np.c_[centre + rng.normal(0, 0.02, (73, 2)), rng.uniform(0.318, 0.76, 73)]
        views.append(np.concatenate([spill, body, tail]))
    return views


def _box(views: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The object's box as ``estimate_target_bounds`` makes it: its points' 1st-99th percentile."""
    low, high = np.quantile(np.concatenate(views), [0.01, 0.99], axis=0)
    return low, high


def test_the_lamp_rests_on_the_round_table_its_frames_show_it_on():
    mesh, views = _room(), _lamp_views()
    low, high = _box(views)
    assert TABLE_TOP_M - low[2] == pytest.approx(0.226, abs=0.03)  # the box reaches about 22 cm under the top
    corners = [[x, y] for x in (low[0], high[0]) for y in (low[1], high[1])]
    assert sum(np.linalg.norm(np.subtract(corner, TABLE_CENTRE)) > TABLE_RADIUS_M for corner in corners) >= 3
    # Before: nothing within the residual of the bottom; the box's corners miss the round top even at its
    # height; and the nearest-surface guess, which needs no contact, takes the floor under the table.
    assert support_under(mesh, low, high, up=2, meters_per_unit=1.0, placement_residual_m=RESIDUAL_M) is None
    raised = [low + [0, 0, TABLE_TOP_M - low[2]], high + [0, 0, TABLE_TOP_M - low[2]]]
    assert support_under(mesh, *raised, up=2, meters_per_unit=1.0) is None
    guess = nearest_support_under(mesh, low, high, up=2, meters_per_unit=1.0, placement_residual_m=RESIDUAL_M)
    assert guess["top_runtime_units"] == pytest.approx(FLOOR_TOP_M, abs=1e-6)
    # Now: the object's own outline, contact in its frames, nearest its lowest observed points.
    support, record = resting_surface_under(mesh, low, high, views=views, up=2, meters_per_unit=1.0,
                                            placement_residual_m=RESIDUAL_M, rise_m=0.334)
    assert support["top_runtime_units"] == pytest.approx(TABLE_TOP_M, abs=0.005)
    assert support["basis"] == OUTLINE_BASIS and support["physical_measurement"] is False
    assert support["outline_share"] >= 0.5 and support["contact_frames"] == support["frames"] == 6
    assert record["outline"]["basis"] == "convex_hull_of_observed_points"
    assert record["outline"]["area_m2"] < record["outline"]["box_area_m2"]
    assert record["candidates"][-1]["chosen"] is True
    json.dumps(record, allow_nan=False)
    # A surface fitted under the outline is the same surface a box lookup fits there.
    on_table = [np.r_[TABLE_CENTRE - 0.05, TABLE_TOP_M], np.r_[TABLE_CENTRE + 0.05, TABLE_TOP_M + 0.2]]
    assert same_support_surface(support, support_under(mesh, *on_table, up=2, meters_per_unit=1.0))


def test_the_floor_under_the_table_is_not_chosen_without_contact():
    mesh, views = _room(table=False), _lamp_views()
    low, high = _box(views)
    support, record = resting_surface_under(mesh, low, high, views=views, up=2, meters_per_unit=1.0,
                                            placement_residual_m=RESIDUAL_M, rise_m=0.334)
    assert support is None and record["candidates"]
    assert {row["refused"] for row in record["candidates"]} == {"no_contact_in_the_frames"}
    floor = record["candidates"][0]
    assert floor["height_m"] == pytest.approx(FLOOR_TOP_M - low[2], abs=0.01) and floor["contact_frames"] == 0


def test_a_y_down_collider_finds_the_same_table():
    # The same room in a frame whose up is -Y: (x, y, z) -> (x, -z, y).
    swap = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    mesh = _room()
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices) @ swap.T, faces=mesh.faces, process=False)
    views = [view @ swap.T for view in _lamp_views()]
    low, high = _box(views)
    support, _ = resting_surface_under(mesh, low, high, views=views, up=1, up_sign=-1, meters_per_unit=1.0,
                                       placement_residual_m=RESIDUAL_M, rise_m=0.334)
    assert support["top_runtime_units"] == pytest.approx(-TABLE_TOP_M, abs=0.005)
    assert support["aabb_min"][1] == pytest.approx(-TABLE_TOP_M, abs=1e-6)  # the top's own faces, back in -Y
    assert support["aabb_max"][0] - support["aabb_min"][0] == pytest.approx(2 * TABLE_RADIUS_M, abs=0.01)


def _search(monkeypatch, *, fit=RECORDED_BOX_FIT, mesh=None, views=None):
    """``visible_resting_support`` on the fixture: the frames' points and the box-fit check as job 4 recorded them."""
    views = _lamp_views() if views is None else views
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.masked_view_points", lambda track, frames: views)
    monkeypatch.setattr("blueprint_pipeline.website_task_placement_check.check_video_placement", lambda **_: fit)
    low, high = _box(views)
    table = {"target_id": "bedside_table", "target_role": "support", "semantic_label": "round bedside table",
             "estimated_visible_bounds": {"minimum": [0.02, -0.38, 0.36], "maximum": [0.48, 0.08, low[2] + 0.259]}}
    shelf = {"target_id": "glass_shelf", "target_role": "destination", "placement_relation": "on",
             "estimated_visible_bounds": {"minimum": [0.3, -1.5, 1.2], "maximum": [0.6, -0.9, 1.3]}}
    found = resting.visible_resting_support(
        collider=_room() if mesh is None else mesh, subject_min=low.tolist(), subject_max=high.tolist(),
        track={"observations": []}, frames=[], matrix=np.eye(4), runtime_to_sim=np.eye(4), targets=[table, shelf],
        up=2, up_sign=1, meters_per_unit=1.0, placement_residual_m=RESIDUAL_M)
    return (low, high), found


def test_a_box_sitting_too_low_is_raised_onto_the_table_and_the_correction_recorded(monkeypatch):
    (low, high), (support, record, new_low, new_high) = _search(monkeypatch)
    uncertainty = record["estimate_uncertainty"]
    # The box-fit check flags the box: its masks cover 0.187 of it, so its bottom may be most of its height low ...
    assert uncertainty["basis"] == "box_fit_masks_do_not_cover_the_box"
    assert uncertainty["rise_m"] == pytest.approx((1 - 0.1866721136943227) * (high[2] - low[2]))
    # ... but never above where the lamp visibly rests: the named table's top, plus the search window.
    assert record["visible_rest"]["target_id"] == "bedside_table"
    assert record["range_m"]["up"] == pytest.approx(0.259 + RESIDUAL_M)
    assert support["top_runtime_units"] == pytest.approx(TABLE_TOP_M, abs=0.005)
    # Points below a surface it rests on are not the lamp: its bottom rises to the table, its observed top stays.
    assert new_low[2] == support["top_runtime_units"] and new_high[2] == high[2]
    assert new_low[:2] == low[:2].tolist() and new_high[:2] == high[:2].tolist()
    correction = record["correction"]
    assert correction["basis"] == "bottom_raised_to_the_surface_observed_top_kept"
    assert correction["bottom_moved_m"] == pytest.approx(TABLE_TOP_M - low[2], abs=0.005)
    assert correction["height_after_m"] == pytest.approx(high[2] - support["top_runtime_units"])
    assert support["guess"]["flag"] == SUPPORT_GUESSED and support["guess"]["basis"] == resting.GUESS_BASIS
    assert record["status"] == "found"
    json.dumps(record, allow_nan=False)


def test_a_box_that_fits_its_masks_is_searched_only_within_the_window(monkeypatch):
    fit = {"status": "assessed", "flags": [], "summary": {"median_overlap": 0.8, "median_size_ratio": {}}}
    (low, high), (support, record, new_low, new_high) = _search(monkeypatch, fit=fit, mesh=_room(floor=False))
    assert record["estimate_uncertainty"]["basis"] == "search_window"
    assert record["range_m"]["up"] == pytest.approx(RESIDUAL_M)
    assert support is None and record["status"] == "not_found"
    assert (new_low, new_high) == (low.tolist(), high.tolist())


def test_no_surface_under_the_lamp_finds_nothing(monkeypatch):
    elsewhere = trimesh.creation.box(extents=[0.4, 0.4, 0.02],
                                     transform=trimesh.transformations.translation_matrix([3.0, 3.0, TABLE_TOP_M]))
    (low, high), (support, record, new_low, new_high) = _search(monkeypatch, mesh=elsewhere)
    assert support is None and record["status"] == "not_found" and record["search"]["candidates"] == []
    assert (new_low, new_high) == (low.tolist(), high.tolist())


# Compile: the code with no surface is today's, and a job that found a support before finds the same one.
def _preparation_fixture():
    import tests.test_website_task_preparation as fixture
    return fixture


def _rebind(arguments, frame, depth, valid):
    np.savez(frame["geometry_path"], depth_m=depth, valid_mask=valid)
    frame["geometry_digest"] = _sha256_file(Path(frame["geometry_path"]))
    arguments["source_geometry"]["digest"] = canonical_digest(arguments["source_geometry"], digest_field="digest")
    arguments["task_masks"]["source_geometry_digest"] = arguments["source_geometry"]["digest"]
    arguments["task_masks"]["digest"] = canonical_digest(arguments["task_masks"], digest_field="digest")
    return arguments


def _far_above(arguments):
    """The cup's depth scaled toward the camera: it floats far above, over no surface at all."""
    frame = arguments["source_geometry"]["frames"][0]
    with np.load(frame["geometry_path"]) as data:
        depth, valid = data["depth_m"], data["valid_mask"]
    depth[20:25, 21:25] *= 0.2
    return _rebind(arguments, frame, depth, valid)


def _held_above(arguments):
    """The cup seen five rows higher at the same depth: held about 0.3 m above the table, over it."""
    frame = arguments["source_geometry"]["frames"][0]
    with np.load(frame["geometry_path"]) as data:
        depth, valid = data["depth_m"], data["valid_mask"]
    yy, xx = np.indices((5, 4))
    depth[15:20, 21:25] = 1.57 + 0.005 * (xx + yy)
    width = depth.shape[1]
    cup = arguments["task_masks"]["targets"][0]
    cup["track"] = {**cup["track"], "observations": [
        {**cup["track"]["observations"][0], "runs": [{"start": row * width + 21, "length": 4, "probability": 0.9}
                                                     for row in range(15, 20)]}]}
    return _rebind(arguments, frame, depth, valid)


def test_a_support_found_at_the_estimated_bottom_is_unchanged(tmp_path, monkeypatch):
    fixture = _preparation_fixture()

    def unexpected(**_):
        pytest.fail("the visible-rest search runs only when no surface lies at the estimated bottom")

    monkeypatch.setattr(resting, "visible_resting_support", unexpected)
    value = fixture._compile(tmp_path)
    assert value["status"] == "intake_ready" and "support_search" not in value
    assert value["support"]["top_runtime_units"] == pytest.approx(1.75, abs=1e-6)
    assert "guess" not in value["support"]


def test_an_object_held_above_its_table_keeps_the_nearest_surface_guess(tmp_path, monkeypatch):
    # The cup is held about 0.3 m above the table, no frame shows it touching: the visible-rest search finds
    # nothing, and the nearest-surface guess places it exactly as before the search existed.
    fixture = _preparation_fixture()
    (tmp_path / "new").mkdir()
    (tmp_path / "old").mkdir()
    value = preparation.compile_website_scene_preparation(**_held_above(fixture._arguments(tmp_path / "new")))
    real = resting.visible_resting_support

    def without_search(**kwargs):
        _, record, low, high = real(**kwargs)
        return None, record, low, high

    monkeypatch.setattr(resting, "visible_resting_support", without_search)
    before = preparation.compile_website_scene_preparation(**_held_above(fixture._arguments(tmp_path / "old")))
    assert value["support"]["guess"]["basis"] == "nearest_surface_under_footprint"
    assert value["support"]["guess"]["drop_m"] > 0.1
    assert value["support_search"]["status"] == "not_found"
    for key in ("support", "subject", "destination", "compose_back"):
        assert json.dumps(value[key], sort_keys=True, default=str).replace(str(tmp_path / "new"), "") == \
            json.dumps(before[key], sort_keys=True, default=str).replace(str(tmp_path / "old"), "")


def test_no_plausible_surface_raises_the_same_code_and_keeps_the_record(tmp_path):
    fixture = _preparation_fixture()
    value = preparation.compile_website_scene_preparation(**_far_above(fixture._arguments(tmp_path)))
    assert value["status"] == "needs_input"
    assert "support_surface_not_found_under_subject" in value["blockers"]
    assert value["support"] is None and value["compose_back"]["pose_world"]["support_snap_runtime_units"] == 0
    assert value["support_search"]["status"] == "not_found"
    assert value["support_search"]["estimate_uncertainty"]["box_fit"]["status"] != "assessed"  # one masked frame
    assert value["digest"] == canonical_digest(value, digest_field="digest")
    assert SUPPORT_GUESSED not in preparation_flags(tmp_path / "out" / "preparation.json")


def test_the_object_outline_is_its_points_hull_not_its_box():
    from blueprint_pipeline.website_support_geometry import object_outline

    angle = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    disc = np.c_[0.1 * np.cos(angle), 0.1 * np.sin(angle), np.zeros(400)]
    outline = object_outline(disc, [-0.1, -0.1, -0.01], [0.1, 0.1, 0.01], up=2)
    x, y = outline[:, 0], outline[:, 1]
    area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))  # positive: counter-clockwise
    assert area == pytest.approx(math.pi * 0.01, rel=0.01)
    assert len(object_outline(disc[:2], [-0.1, -0.1, -0.01], [0.1, 0.1, 0.01], up=2)) == 4  # the box's rectangle
