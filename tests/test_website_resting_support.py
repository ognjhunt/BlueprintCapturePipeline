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
#   src/blueprint_pipeline/website_collider_conditioning.py
#   src/blueprint_pipeline/website_packet_facts.py
#   src/blueprint_pipeline/website_surface_target_admission.py
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


def _room(*, floor: bool = True, table: bool = True, thickness: float = 0.03) -> trimesh.Trimesh:
    """A closed round table top (its underside faces down) over a flat floor."""
    parts = []
    if table:
        top = trimesh.creation.cylinder(radius=TABLE_RADIUS_M, height=thickness, sections=64)
        top.apply_translation([*TABLE_CENTRE, TABLE_TOP_M - thickness / 2])
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


@pytest.mark.parametrize("thickness", [0.018, 0.025, 0.03])
def test_the_lamp_rests_on_the_round_table_its_frames_show_it_on(thickness):
    # Review of the first build: a closed top 1.8 or 2.5 cm thick put the lamp on its underside.
    mesh, views = _room(thickness=thickness), _lamp_views()
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
    assert support["top_runtime_units"] == pytest.approx(TABLE_TOP_M, abs=1e-6)  # the top, never the underside
    assert support["basis"] == OUTLINE_BASIS and support["physical_measurement"] is False
    assert support["resting_correction"]["basis"] == "bottom_raised_to_the_surface_observed_top_kept"
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


@pytest.mark.parametrize("thickness", [0.018, 0.03])
def test_a_y_down_collider_finds_the_same_table(thickness):
    # The same room in a frame whose up is -Y: (x, y, z) -> (x, -z, y).
    swap = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    mesh = _room(thickness=thickness)
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices) @ swap.T, faces=mesh.faces, process=False)
    views = [view @ swap.T for view in _lamp_views()]
    low, high = _box(views)
    support, _ = resting_surface_under(mesh, low, high, views=views, up=1, up_sign=-1, meters_per_unit=1.0,
                                       placement_residual_m=RESIDUAL_M, rise_m=0.334)
    assert support["top_runtime_units"] == pytest.approx(-TABLE_TOP_M, abs=1e-6)
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


def _small_object_views(low_by_m: float, count: int = 6) -> list[np.ndarray]:
    """A 2 cm, 0.4 m wide object on the round top whose points all sit ``low_by_m`` (to 2 cm less) too low:
    its box corners lie off the round edge, and its own outline is the box's square."""
    rng = np.random.default_rng(9)
    return [np.c_[TABLE_CENTRE + rng.uniform(-0.2, 0.2, (600, 2)),
                  TABLE_TOP_M - low_by_m + rng.uniform(0.0, 0.02, 600)] for _ in range(count)]


def test_a_box_low_within_the_window_moves_up_whole_and_keeps_its_height(monkeypatch):
    # Review probe: points 3-5 cm low. The bottom was raised past the top and the height went negative.
    views = _small_object_views(0.05)
    low, high = _box(views)
    assert support_under(_room(), low, high, up=2, meters_per_unit=1.0, placement_residual_m=RESIDUAL_M) is None
    _, (support, record, new_low, new_high) = _search(monkeypatch, views=views)
    correction = record["correction"]
    assert correction["basis"] == "moved_up_onto_the_surface"
    assert correction["height_before_m"] > 0
    assert correction["height_after_m"] == pytest.approx(correction["height_before_m"])
    assert new_low[2] == pytest.approx(TABLE_TOP_M, abs=1e-6)  # on the top, not inside the table
    assert new_high[2] - new_low[2] == pytest.approx(high[2] - low[2])


def test_a_surface_above_the_observed_top_is_never_taken(monkeypatch):
    # The same object 10-12 cm low, past the window: raising its bottom would put it above its own top.
    views = _small_object_views(0.12)
    low, high = _box(views)
    support, record = resting_surface_under(_room(floor=False), low, high, views=views, up=2, meters_per_unit=1.0,
                                            placement_residual_m=RESIDUAL_M, rise_m=0.3)
    assert support is None
    assert {row["refused"] for row in record["candidates"]} == {"surface_above_the_observed_top"}


def test_a_plane_through_the_object_is_no_contact():
    # A shelf at the height of the middle of a tall object's points: points meet its plane in every
    # frame, but half the object lies below it. Its lowest points do not sit there.
    rng = np.random.default_rng(5)
    shelf = trimesh.creation.box(extents=[0.6, 0.6, 0.02],
                                 transform=trimesh.transformations.translation_matrix([0, 0, 0.99])).subdivide().subdivide()
    views = [np.c_[rng.uniform(-0.05, 0.05, (500, 2)), rng.uniform(0.80, 1.20, 500)] for _ in range(6)]
    low, high = _box(views)
    support, record = resting_surface_under(shelf, low, high, views=views, up=2, meters_per_unit=1.0,
                                            placement_residual_m=RESIDUAL_M, rise_m=0.3)
    assert support is None
    row = record["candidates"][0]
    assert row["refused"] == "no_contact_in_the_frames" and row["contact_frames"] == 6
    assert row["lowest_points_from_plane_m"] < -0.15


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


def _bled_below(arguments):
    """The cup wider, one of its 40 pixels' depth bled behind the table (below its top, 2.5%), and the
    room's collider wound outward (the fixture's grid faces down): its box reaches under the table."""
    frame = arguments["source_geometry"]["frames"][0]
    with np.load(frame["geometry_path"]) as data:
        depth, valid = data["depth_m"], data["valid_mask"]
    yy, xx = np.indices((5, 8))
    depth[20:25, 21:29] = 1.57 + 0.005 * (xx % 4 + yy)
    depth[24, 28] = 2.2
    cup = arguments["task_masks"]["targets"][0]
    cup["track"] = {**cup["track"], "observations": [
        {**cup["track"]["observations"][0], "runs": [{"start": row * depth.shape[1] + 21, "length": 8,
                                                      "probability": 0.9} for row in range(20, 25)]}]}
    base = arguments["base_scene"]
    mesh = trimesh.load(base["collision_mesh_path"], force="mesh", process=False)
    trimesh.Trimesh(vertices=mesh.vertices, faces=np.asarray(mesh.faces)[:, ::-1], process=False).export(
        base["collision_mesh_path"])
    base["collision_mesh_digest"] = _sha256_file(Path(base["collision_mesh_path"]))
    return _rebind(arguments, frame, depth, valid)


def test_compile_rests_a_box_reaching_under_its_table_on_it_and_flattens_that_surface(tmp_path, monkeypatch):
    monkeypatch.setattr("blueprint_pipeline.website_task_placement_check.check_video_placement",
                        lambda **_: RECORDED_BOX_FIT)
    value = preparation.compile_website_scene_preparation(**_bled_below(_preparation_fixture()._arguments(tmp_path)))
    assert "support_surface_not_found_under_subject" not in value["blockers"]
    support, search = value["support"], value["support_search"]
    assert support["basis"] == OUTLINE_BASIS and support["guess"]["flag"] == SUPPORT_GUESSED
    assert search["status"] == "found" and search["correction"]["basis"] == "bottom_raised_to_the_surface_observed_top_kept"
    # The table top is runtime y = 1.75, simulator z = 3.5: the bottom rests there, the observed top stays.
    assert value["subject"]["aabb_min_xyz"][2] == pytest.approx(3.5, abs=1e-6)
    assert support["subject_snap_m"] == pytest.approx(0.0, abs=1e-9)
    # The surface found is flattened under the corrected box, as every surface an object rests on is.
    surfaces = value["collider_conditioning"]["flattening"]["surfaces"]
    assert [row["roles"] for row in surfaces][0] == ["start_support"] and surfaces[0]["status"] == "flattened"
    assert SUPPORT_GUESSED in preparation_flags(tmp_path / "out" / "preparation.json")


def test_a_bumpy_top_found_under_the_outline_is_flattened_level_at_its_top():
    from blueprint_pipeline.website_collider_conditioning import flatten_placement_surfaces

    mesh = _room(floor=False).subdivide().subdivide()
    vertices = np.asarray(mesh.vertices).copy()
    rim = np.hypot(*(vertices[:, :2] - TABLE_CENTRE).T) > TABLE_RADIUS_M - 0.02
    top = (np.abs(vertices[:, 2] - TABLE_TOP_M) < 1e-9) & ~rim
    vertices[top, 2] += np.random.default_rng(2).uniform(-0.004, 0.004, int(top.sum()))
    bumpy = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
    views = _lamp_views()
    support, _ = resting_surface_under(bumpy, *_box(views), views=views, up=2, meters_per_unit=1.0,
                                       placement_residual_m=RESIDUAL_M, rise_m=0.334)
    assert support["basis"] == OUTLINE_BASIS
    flattened, record = flatten_placement_surfaces(vertices, np.asarray(mesh.faces),
        [{"role": "start_support", "support": support}], up=2, up_sign=1, meters_per_unit=1.0,
        placement_uncertainty_m=RESIDUAL_M)
    surface = record["surfaces"][0]
    assert surface["status"] == "flattened" and surface["max_displacement_m"] > 0
    moved = np.unique(np.asarray(mesh.faces)[support["face_indices"]])
    interior = moved[~rim[moved]]
    assert np.allclose(flattened[interior, 2], support["top_runtime_units"])


def test_the_package_says_where_the_object_was_set_down_and_why():
    from blueprint_pipeline.website_packet_facts import ESTIMATE_SENTENCES, measured_and_estimated

    def _measured_and_estimated(*args):
        measured, estimated = measured_and_estimated(*args)
        return [row["text"] for row in measured], [row["text"] for row in estimated]

    subject = {"aabb_min_xyz": [0.0, 0.0, 0.765], "aabb_max_xyz": [0.4, 0.375, 1.208]}
    raised = {"subject": subject, "support": {"guess": {"basis": resting.GUESS_BASIS, "flag": SUPPORT_GUESSED,
                                                        "bottom_moved_m": 0.226,
                                                        "correction": "bottom_raised_to_the_surface_observed_top_kept"}},
              "support_search": {"correction": {"height_before_m": 0.669, "height_after_m": 0.443}}}
    _, estimated = _measured_and_estimated(raised, {"scene_estimates": []}, None)
    size = next(line for line in estimated if line.startswith("Object size"))
    assert size.endswith("from the video; its box reached 23 cm below the surface its frames show it resting on, "
                         "so Blueprint raised its bottom onto that surface and kept its observed top (height 67 cm "
                         "before, 44 cm after).")
    assert ESTIMATE_SENTENCES[SUPPORT_GUESSED] in estimated
    nearest = {"subject": subject, "support": {"guess": {"basis": "nearest_surface_under_footprint",
                                                         "drop_m": 0.2, "flag": SUPPORT_GUESSED}}}
    assert "moved it down 20 cm onto the nearest surface under it, its size unchanged." in \
        next(line for line in _measured_and_estimated(nearest, None, None)[1] if line.startswith("Object size"))
    plain = _measured_and_estimated({"subject": subject, "support": {"basis": "fitted_surface_under_footprint"}},
                                    {"scene_estimates": []}, None)[1]
    assert next(line for line in plain if line.startswith("Object size")).endswith(", from the video.")
    assert ESTIMATE_SENTENCES[SUPPORT_GUESSED] not in plain


def test_the_object_outline_is_its_points_hull_not_its_box():
    from blueprint_pipeline.website_support_geometry import object_outline

    angle = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    disc = np.c_[0.1 * np.cos(angle), 0.1 * np.sin(angle), np.zeros(400)]
    outline = object_outline(disc, [-0.1, -0.1, -0.01], [0.1, 0.1, 0.01], up=2)
    x, y = outline[:, 0], outline[:, 1]
    area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))  # positive: counter-clockwise
    assert area == pytest.approx(math.pi * 0.01, rel=0.01)
    assert len(object_outline(disc[:2], [-0.1, -0.1, -0.01], [0.1, 0.1, 0.01], up=2)) == 4  # the box's rectangle


def _slab(top: float = 0.6, half: float = 0.3) -> trimesh.Trimesh:
    """A closed 3 cm table top, ``half`` each way about the origin, its top at ``top``."""
    return trimesh.creation.box(extents=[2 * half, 2 * half, 0.03], transform=trimesh.transformations.translation_matrix(
        [0, 0, top - 0.015])).subdivide().subdivide().subdivide()


def test_noisy_depth_reads_the_table_the_object_stands_on_from_the_support_s_own_spread():
    # The synthetic set's seed 35: 1.5% depth noise put the object's lowest points 3.2 cm under the table it
    # stands on, past the 3 cm window, and the table was refused. The table's own points show that noise.
    rng, sigma = np.random.default_rng(35), 0.02
    views = []
    for _ in range(6):
        spill = np.c_[rng.uniform(-0.06, 0.06, (800, 2)), 0.6 + rng.normal(0, sigma, 800)]  # its mask on the table
        body = np.c_[rng.uniform(-0.04, 0.04, (200, 2)), 0.6 + rng.uniform(0, 0.1, 200) + rng.normal(0, sigma, 200)]
        views.append(np.concatenate([spill, body]))
    table = np.c_[rng.uniform(-0.28, 0.28, (4000, 2)), 0.6 + rng.normal(0, sigma, 4000)]
    table = table[np.abs(table[:, :2]).max(axis=1) > 0.04]  # the object hides the table under it
    low, high = _box(views)
    options = dict(views=views, up=2, meters_per_unit=1.0, rise_m=0.05)
    before, record = resting_surface_under(_slab(), low, high, **options)
    assert before is None and record["candidates"][0]["refused"] == "no_contact_in_the_frames"
    assert record["candidates"][0]["lowest_points_from_plane_m"] < -0.03  # past the window, from noise alone
    support, record = resting_surface_under(_slab(), low, high, surface_points=table, **options)
    assert support is not None and support["top_runtime_units"] == pytest.approx(0.6, abs=0.005)
    contact = support["contact_tolerance"]
    assert contact["basis"] == "window_and_support_points_spread_about_the_plane" and contact["window_m"] == 0.03
    # The spread read within the window of the plane, robustly: a little under the noise's own, never above it.
    assert 0.7 * sigma <= contact["noise_standard_deviation_m"] <= sigma
    assert contact["tolerance_m"] == pytest.approx(0.03 + 1.6449 * contact["noise_standard_deviation_m"], abs=1e-4)
    assert abs(support["lowest_points_from_plane_m"]) <= contact["tolerance_m"]
    json.dumps(record, allow_nan=False)


def test_a_plane_through_the_object_stays_no_contact_with_the_noise_measured():
    rng = np.random.default_rng(5)
    shelf = trimesh.creation.box(extents=[0.6, 0.6, 0.02],
                                 transform=trimesh.transformations.translation_matrix([0, 0, 0.99])).subdivide().subdivide()
    views = [np.c_[rng.uniform(-0.05, 0.05, (500, 2)), rng.uniform(0.80, 1.20, 500)] for _ in range(6)]
    shelf_points = np.c_[rng.uniform(-0.28, 0.28, (3000, 2)), 1.0 + rng.normal(0, 0.005, 3000)]
    low, high = _box(views)
    support, record = resting_surface_under(shelf, low, high, views=views, up=2, meters_per_unit=1.0,
                                            placement_residual_m=RESIDUAL_M, rise_m=0.3, surface_points=shelf_points)
    assert support is None
    row = record["candidates"][0]
    assert row["refused"] == "no_contact_in_the_frames" and row["lowest_points_from_plane_m"] < -0.15
    assert RESIDUAL_M < row["contact_tolerance"]["tolerance_m"] < RESIDUAL_M + 0.01  # 5 mm noise grows it a little


def test_a_mask_spilled_down_the_table_s_side_is_not_the_object_sunk_into_it():
    # The synthetic set's seed 10: no depth noise, but the object's mask ran over the table's front edge and down
    # its side, 2 to 7 cm under the top just past the edge. Those points are beside the table, not in it.
    rng = np.random.default_rng(10)
    views = []
    for _ in range(6):
        body = np.c_[rng.uniform(-0.04, 0.04, 500), 0.22 + rng.uniform(-0.04, 0.04, 500), rng.uniform(0.6, 0.72, 500)]
        spill = np.c_[rng.uniform(-0.06, 0.06, 300), rng.uniform(0.16, 0.3, 300), np.full(300, 0.6)]
        side = np.c_[rng.uniform(-0.06, 0.06, 150), np.full(150, 0.302), rng.uniform(0.53, 0.58, 150)]
        views.append(np.concatenate([body, spill, side]))
    low, high = _box(views)
    assert high[1] > 0.3 and low[2] < 0.55  # the box reaches past the edge and below the top
    options = dict(views=views, up=2, meters_per_unit=1.0, rise_m=0.1)
    support, record = resting_surface_under(_slab(), low, high, **options)
    assert support is not None and support["top_runtime_units"] == pytest.approx(0.6, abs=0.005)
    contact = support["contact_tolerance"]
    assert contact["basis"] == "window" and contact["tolerance_m"] == 0.03  # no support points: the window
    assert contact["object_points_over_the_surface_share"] <= 1 - 150 / 1100  # the side's points are not read
    assert support["lowest_points_from_plane_m"] == pytest.approx(0.0, abs=0.005)


def test_compile_s_search_reads_the_noise_from_the_named_support_s_own_points(monkeypatch):
    rng = np.random.default_rng(4)
    angle, radius = rng.uniform(0, 2 * np.pi, 3000), TABLE_RADIUS_M * np.sqrt(rng.uniform(0, 1, 3000))
    table_points = np.c_[TABLE_CENTRE + np.c_[np.cos(angle), np.sin(angle)] * radius[:, None],
                         TABLE_TOP_M + rng.normal(0, 0.004, 3000)]
    lamp, table_track = {"observations": []}, {"observations": [{"source_frame_id": "t"}]}
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.masked_view_points",
                        lambda track, frames: _lamp_views() if track is lamp else [table_points[:1500], table_points[1500:]])
    monkeypatch.setattr("blueprint_pipeline.website_task_placement_check.check_video_placement",
                        lambda **_: RECORDED_BOX_FIT)
    low, high = _box(_lamp_views())
    table = {"target_id": "bedside_table", "target_role": "support", "track": table_track,
             "estimated_visible_bounds": {"minimum": [0.02, -0.38, 0.36], "maximum": [0.48, 0.08, low[2] + 0.259]}}
    support, record, _, _ = resting.visible_resting_support(
        collider=_room(), subject_min=low.tolist(), subject_max=high.tolist(), track=lamp, frames=[],
        matrix=np.eye(4), runtime_to_sim=np.eye(4), targets=[table], up=2, up_sign=1, meters_per_unit=1.0,
        placement_residual_m=RESIDUAL_M)
    assert record["visible_rest"]["support_points"] == 3000
    assert support is not None, json.dumps({k: v for k, v in record.items() if k != "search"}, default=float)
    contact = support["contact_tolerance"]
    assert support["top_runtime_units"] == pytest.approx(TABLE_TOP_M, abs=0.005)
    assert contact["basis"] == "window_and_support_points_spread_about_the_plane"
    assert contact["noise_standard_deviation_m"] == pytest.approx(0.004, abs=0.001)
    assert contact["tolerance_m"] == pytest.approx(RESIDUAL_M + 1.6449 * contact["noise_standard_deviation_m"], abs=1e-4)
