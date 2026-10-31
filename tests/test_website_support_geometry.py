import numpy as np
import pytest
import trimesh

from blueprint_pipeline.website_support_geometry import ground_on_observed_floor, support_under


def plate(x0=-1, x1=1, y0=-1, y1=1, height=0.7):
    return trimesh.Trimesh(vertices=[[x0, y0, height], [x1, y0, height],
                                    [x1, y1, height], [x0, y1, height]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)


def test_contacts_use_surface_extent_not_subject_extent():
    value = support_under(plate(), [-.03, -.03, .71], [.03, .03, .8], up=2, meters_per_unit=1)
    assert value["aabb_min"] == [-1, -1, .7]
    assert value["aabb_max"] == [1, 1, .7]
    assert value["face_indices"] == [0, 1]
    assert value["physical_measurement"] is False


@pytest.mark.parametrize("lower,upper", [([.98, -.03, .7], [1.04, .03, .8]),
                                         ([-.03, -.03, 1.7], [.03, .03, 1.8])])
def test_table_edge_and_remote_floor_do_not_support_object(lower, upper):
    assert support_under(plate(), lower, upper, up=2, meters_per_unit=1) is None


def test_gap_in_enclosing_bounds_is_not_a_support():
    mesh = trimesh.util.concatenate([plate(x1=-.01), plate(x0=.01)])
    assert support_under(mesh, [-.03, -.03, .7], [.03, .03, .8], up=2, meters_per_unit=1) is None


def test_disconnected_contacts_do_not_create_a_table():
    # All nine probe locations have contact, but they are separate islands.
    mesh = trimesh.util.concatenate([plate(x-.005, x+.005, y-.005, y+.005)
                                     for x in [-.03, 0, .03] for y in [-.03, 0, .03]])
    assert support_under(mesh, [-.03, -.03, .7], [.03, .03, .8], up=2, meters_per_unit=1) is None


def test_y_up_and_estimated_scale_use_same_physical_contact_tolerance():
    mesh = plate()
    mesh.vertices = np.asarray(mesh.vertices)[:, [0, 2, 1]] * 10
    value = support_under(mesh, [-.3, 7.1, -.3], [.3, 8, .3], up=1, meters_per_unit=.1)
    assert value["top_runtime_units"] == 7


def test_sloped_surface_is_not_silently_treated_as_flat():
    mesh = plate()
    mesh.vertices[:, 2] += mesh.vertices[:, 0] * .5
    assert support_under(mesh, [-.03, -.03, .7], [.03, .03, .8], up=2, meters_per_unit=1) is None


def _floor_with_bay(height=0.0):
    """A floor around a 0.6 m bay that the generated world left without a face."""
    return trimesh.util.concatenate([plate(-2, -.3, -2, 2, height), plate(.3, 2, -2, 2, height),
                                     plate(-.3, .3, .3, 2, height), plate(-.3, .3, -2, -.3, height)])


BAY = ([-.3, -.3, .2], [.3, .3, .82])


def test_built_in_body_reaches_the_collider_floor_beside_its_empty_bay():
    value = ground_on_observed_floor(_floor_with_bay(), *BAY, up=2, meters_per_unit=1, floor_height=.01)
    assert value["basis"] == "collider_floor_beside_footprint"
    assert value["top_runtime_units"] == pytest.approx(0)
    assert value["extended_runtime_units"] == pytest.approx(.2)
    assert value["physical_measurement"] is False


def test_without_a_collider_floor_the_footage_floor_is_used_and_named():
    value = ground_on_observed_floor(plate(-3, -2.5, -3, -2.5, height=1.5), *BAY, up=2, meters_per_unit=1,
                                     floor_height=0.0)
    assert value["basis"] == "registered_observed_floor_plane"
    assert value["face_indices"] == []
    assert value["aabb_min"] == [-.3, -.3, 0.0] and value["aabb_max"] == [.3, .3, 0.0]


@pytest.mark.parametrize("floor_height", [-.3, .2, .5])
def test_a_floating_or_sunken_box_is_not_stretched_to_the_floor(floor_height):
    # 0.5 m below is not a hidden kick band; at or above the bottom is no gap.
    assert ground_on_observed_floor(_floor_with_bay(floor_height), *BAY, up=2, meters_per_unit=1,
                                    floor_height=floor_height) is None


def test_a_real_shelf_under_the_footprint_is_not_bridged_to_the_floor():
    mesh = trimesh.util.concatenate([_floor_with_bay(), plate(-.3, .3, -.3, .3, height=.1)])
    assert ground_on_observed_floor(mesh, *BAY, up=2, meters_per_unit=1, floor_height=0.0) is None


def test_slivers_in_a_generated_bay_do_not_count_as_a_shelf():
    mesh = trimesh.util.concatenate([_floor_with_bay(), plate(-.05, .05, -.05, .05, height=.1)])
    value = ground_on_observed_floor(mesh, *BAY, up=2, meters_per_unit=1, floor_height=0.0)
    assert value is not None and value["intermediate_surface_area_m2"] == pytest.approx(.01)


def test_minus_y_up_worlds_ground_in_the_source_frame():
    mesh = _floor_with_bay()
    mesh.vertices = np.asarray(mesh.vertices)[:, [0, 2, 1]] * [1, -1, 1]
    value = ground_on_observed_floor(mesh, [-.3, -.82, -.3], [.3, -.2, .3], up=1, meters_per_unit=1,
                                     floor_height=0.0, up_sign=-1)
    assert value["top_runtime_units"] == pytest.approx(0)
    assert value["extended_runtime_units"] == pytest.approx(.2)


def bumpy_plate(x0=-.3, x1=.3, y0=-.3, y1=.3, height=.7, *, noise=.006, slope=0.0, seed=0, step=.01):
    """A reconstructed counter: a dense 1 cm grid whose vertices wander millimetres, as Atlas meshes do."""
    rng = np.random.default_rng(seed)
    xs, ys = np.arange(x0, x1 + 1e-9, step), np.arange(y0, y1 + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    gz = height + slope * gx + rng.normal(0.0, noise, gx.shape)
    k = (np.arange(len(xs) - 1)[:, None] * len(ys) + np.arange(len(ys) - 1)[None, :]).reshape(-1)
    faces = np.concatenate([np.c_[k, k + len(ys), k + 1], np.c_[k + 1, k + len(ys), k + len(ys) + 1]])
    return trimesh.Trimesh(vertices=np.c_[gx.ravel(), gy.ravel(), gz.ravel()], faces=faces, process=False)


BOX = ([-.05, -.05, .71], [.05, .05, .85])


def test_a_bumpy_reconstructed_counter_is_fitted_when_no_face_is_level(monkeypatch):
    from blueprint_pipeline import website_support_geometry as geometry

    mesh = bumpy_plate()
    value = support_under(mesh, *BOX, up=2, meters_per_unit=1)
    assert value["basis"] == "fitted_surface_under_footprint" and value["physical_measurement"] is False
    assert value["top_runtime_units"] == pytest.approx(.7, abs=.003)
    # Face centres average three vertices: 6 mm of vertex noise fits as about 6 / sqrt(3) mm.
    assert value["fit_tilt_degrees"] < 1 and value["fit_rms_m"] == pytest.approx(.006 / 3 ** .5, abs=.001)
    assert len(value["face_indices"]) > .75 * len(mesh.faces)  # minus the few faces noise tips past 60 degrees
    assert value["aabb_min"][:2] == pytest.approx([-.3, -.3]) and value["aabb_max"][:2] == pytest.approx([.3, .3])
    # The exact level-face search alone finds nothing on this surface.
    monkeypatch.setattr(geometry, "_fitted_support", lambda *args, **kwargs: None)
    assert support_under(mesh, *BOX, up=2, meters_per_unit=1) is None


@pytest.mark.parametrize("case", ["edge", "gap", "slope", "rough", "islands", "remote"])
def test_a_fitted_surface_keeps_every_refusal(case):
    mesh, lower, upper = bumpy_plate(), *BOX
    if case == "edge":
        lower, upper = [.25, -.05, .71], [.35, .05, .85]
    elif case == "gap":
        mesh = trimesh.util.concatenate([bumpy_plate(x1=-.03), bumpy_plate(x0=.03, seed=1)])
    elif case == "slope":
        mesh = bumpy_plate(slope=np.tan(np.radians(8)))
    elif case == "rough":
        mesh = bumpy_plate(noise=.06)
    elif case == "islands":
        # Two counters side by side, touching only across a hairline that shares no edge.
        mesh = trimesh.util.concatenate([bumpy_plate(x1=-.005), bumpy_plate(x0=.005, seed=1)])
    else:
        lower, upper = [-.05, -.05, 1.0], [.05, .05, 1.1]
    assert support_under(mesh, lower, upper, up=2, meters_per_unit=1) is None


def test_two_fits_on_one_bumpy_counter_name_the_same_surface():
    from blueprint_pipeline.website_support_geometry import same_support_surface

    counter = bumpy_plate(-.6, .6)
    start = support_under(counter, [-.45, -.05, .71], [-.35, .05, .85], up=2, meters_per_unit=1)
    place = support_under(counter, [.35, -.05, .71], [.45, .05, .85], up=2, meters_per_unit=1)
    assert same_support_surface(start, place)
    mesh = trimesh.util.concatenate([counter, bumpy_plate(1.0, 1.6, seed=2)])
    start = support_under(mesh, [-.45, -.05, .71], [-.35, .05, .85], up=2, meters_per_unit=1)
    elsewhere = support_under(mesh, [1.25, -.05, .71], [1.35, .05, .85], up=2, meters_per_unit=1)
    assert elsewhere is not None and not same_support_surface(start, elsewhere)
    exact = support_under(plate(), [-.03, -.03, .71], [.03, .03, .8], up=2, meters_per_unit=1)
    assert same_support_surface(exact, exact) and not same_support_surface(exact, start)


def test_a_level_patch_within_a_fitted_surface_is_that_surface():
    """A conditioned collider (website_collider_conditioning) reads level under a footprint
    wholly on its flattened patch and fitted under one over a gap in it: one surface."""
    from blueprint_pipeline.website_support_geometry import same_support_surface

    fitted = {"basis": "fitted_surface_under_footprint", "face_indices": list(range(100))}
    level = {"basis": "reconstructed_triangle_footprint_contacts", "face_indices": list(range(5, 95))}
    assert same_support_surface(level, fitted) and same_support_surface(fitted, level)
    assert not same_support_surface({**level, "face_indices": list(range(50, 140))}, fitted)
    assert not same_support_surface({**level, "face_indices": []}, fitted)
    assert not same_support_surface({**fitted, "basis": "collider_floor_beside_footprint"}, fitted)


def test_a_y_up_bumpy_counter_in_estimated_units_is_fitted():
    mesh = bumpy_plate()
    mesh.vertices = np.asarray(mesh.vertices)[:, [0, 2, 1]] * 10
    value = support_under(mesh, [-.5, 7.1, -.5], [.5, 8.5, .5], up=1, meters_per_unit=.1)
    assert value["basis"] == "fitted_surface_under_footprint"
    assert value["top_runtime_units"] == pytest.approx(7, abs=.03)


def test_a_corrugated_surface_whose_ridges_would_cut_the_object_is_refused():
    # Codex review on #2805: a 10 cm grid alternating 0.66/0.74 m fits a smooth plane at
    # 0.70 m with ~1.3 cm centroid RMS, yet the contacts sit 4 cm off the object's bottom.
    xs = np.arange(-.5, .5 + 1e-9, .1)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    gz = .7 + .04 * (-1.0) ** (np.rint(gx / .1) + np.rint(gy / .1))
    n = len(xs)
    k = (np.arange(n - 1)[:, None] * n + np.arange(n - 1)[None, :]).reshape(-1)
    faces = np.concatenate([np.c_[k, k + n, k + 1], np.c_[k + 1, k + n, k + n + 1]])
    mesh = trimesh.Trimesh(vertices=np.c_[gx.ravel(), gy.ravel(), gz.ravel()], faces=faces, process=False)
    assert support_under(mesh, [-.11, -.11, .70], [.11, .11, .85], up=2, meters_per_unit=1) is None


def test_measured_placement_residual_widens_the_search_window():
    # capture-447ae4f4: the estimated tray bottom sat 4 cm into the counter and the
    # registration's residual near the task was 7.2 cm; a 3 cm window found nothing.
    lower, upper = [-.03, -.03, .66], [.03, .03, .74]
    value = support_under(plate(), lower, upper, up=2, meters_per_unit=1, placement_residual_m=.07)
    assert value["top_runtime_units"] == pytest.approx(.7)
    assert value["top_runtime_units"] - lower[2] == pytest.approx(.04)  # the snap compile applies
    assert value["search_window_m"] == .07
    assert value["search_window_basis"] == "registration_task_region_residual"
    # A tight residual keeps the 3 cm window, and so does an unmeasured one.
    for residual in (.01, None, float("nan")):
        assert support_under(plate(), lower, upper, up=2, meters_per_unit=1, placement_residual_m=residual) is None
    within = support_under(plate(), [-.03, -.03, .72], [.03, .03, .8], up=2, meters_per_unit=1,
                           placement_residual_m=.01)
    assert within["search_window_m"] == .03 and within["search_window_basis"] == "default"
    # A remote surface stays remote: the window is the residual, not unbounded.
    assert support_under(plate(), [-.03, -.03, .8], [.03, .03, .9], up=2, meters_per_unit=1,
                         placement_residual_m=.07) is None


def test_measured_placement_residual_widens_the_fitted_window_and_y_down_frames():
    bumpy = trimesh.creation.box(extents=[2, 2, .1])
    bumpy.apply_translation([0, 0, .65])
    found = support_under(bumpy, [-.03, -.03, .66], [.03, .03, .74], up=2, meters_per_unit=1,
                          placement_residual_m=.07)
    assert found is not None and found["top_runtime_units"] == pytest.approx(.7)
    flipped = plate(height=-.7)
    down = support_under(flipped, [-.03, -.03, -.74], [.03, .03, -.66], up=2, meters_per_unit=1,
                         up_sign=-1, placement_residual_m=.07)
    assert down is not None and down["top_runtime_units"] == pytest.approx(-.7)
    assert support_under(flipped, [-.03, -.03, -.74], [.03, .03, -.66], up=2, meters_per_unit=1,
                         up_sign=-1) is None


def test_a_wider_window_seeds_on_the_highest_surface_it_reaches():
    # capture-447ae4f4: the 7 cm window around the tray's estimated bottom also reached the
    # counter's edge 3-10 cm below its bumpy top, and the mixed seed failed the level fit.
    top = plate(height=.7).subdivide().subdivide().subdivide().subdivide().subdivide()
    vertices = top.vertices.copy()
    vertices[:, 2] += np.random.default_rng(7).uniform(-.006, .006, len(vertices))
    top = trimesh.Trimesh(vertices=vertices, faces=top.faces, process=False)
    lip = plate(-.2, -.15, -.2, .2, height=.64).subdivide().subdivide().subdivide()
    mesh = trimesh.util.concatenate([top, lip])
    lower, upper = [-.2, -.2, .66], [.2, .2, .74]
    found = support_under(mesh, lower, upper, up=2, meters_per_unit=1, placement_residual_m=.07)
    assert found is not None and found["top_runtime_units"] == pytest.approx(.7, abs=.01)
    assert found["search_window_m"] == .07
    # The default window does not reach the top from this estimate, as before.
    assert support_under(mesh, lower, upper, up=2, meters_per_unit=1) is None


def _counter(x1=.4, hole=None, step=.01):
    xs, ys = np.arange(-.4, x1 + 1e-9, step), np.arange(-.4, .4 + 1e-9, step)
    grid = np.array([[x, y, .7] for y in ys for x in xs])
    n = len(xs)
    faces = []
    for j in range(len(ys) - 1):
        for i in range(n - 1):
            k = j * n + i
            for face in ([k, k + 1, k + n + 1], [k, k + n + 1, k + n]):
                centre = grid[face].mean(axis=0)
                if hole is None or np.hypot(*(centre[:2] - hole)) > .02:
                    faces.append(face)
    return trimesh.Trimesh(vertices=grid, faces=faces, process=False)


def test_a_widened_window_accepts_a_probe_over_a_hole_and_records_its_gap():
    # capture-447ae4f4: a 2-3 cm hole in the counter mesh under the tray's corner, where
    # the tray itself sat, refused a 31x52 cm tray; the registration's residual was 7.2 cm.
    mesh = _counter(hole=np.array([-.2, -.2]))
    lower, upper = [-.2, -.2, .66], [.2, .2, .74]
    found = support_under(mesh, lower, upper, up=2, meters_per_unit=1, placement_residual_m=.07)
    assert found is not None and found["top_runtime_units"] == pytest.approx(.7)
    assert found["probe_gap_basis"] == "registration_task_region_residual"
    [gap] = found["probe_gaps_m"]
    assert gap["probe_runtime_units"] == pytest.approx([-.2, -.2])
    assert 0 < gap["gap_m"] <= .03
    # The default window keeps refusing the hole, as before.
    assert support_under(mesh, [-.2, -.2, .7], [.2, .2, .78], up=2, meters_per_unit=1) is None


def test_a_widened_window_still_refuses_a_probe_past_the_surface_edge():
    # The counter ends 1 cm short of the footprint: within the residual, but surface on one side only.
    mesh = _counter(x1=.19)
    assert support_under(mesh, [-.2, -.2, .66], [.2, .2, .74], up=2, meters_per_unit=1,
                         placement_residual_m=.07) is None
    assert "probe_gaps_m" not in support_under(_counter(), [-.2, -.2, .66], [.2, .2, .74], up=2,
                                               meters_per_unit=1, placement_residual_m=.07)


def test_a_destination_grounded_as_the_start_surface_moves_to_its_farthest_supported_spot():
    """capture-447ae4f4: "the other end of the same table" grounded as the whole table, centre under the tray."""
    from blueprint_pipeline.website_support_geometry import (destination_clear_of_start,
                                                             farthest_supported_destination)

    table = bumpy_plate(-.6, .6, -.4, .4)
    start_min, start_max = [-.15, -.1, .7], [.05, .1, .75]
    support = support_under(table, start_min, start_max, up=2, meters_per_unit=1)
    clearance = np.hypot(.1, .1) + .01
    assert not destination_clear_of_start(start_min, start_max, [0, 0, .7], up=2, clearance=clearance)
    centre, found, on_surface = farthest_supported_destination(table, start_min, start_max, [-.6, -.4, .6], [.6, .4, .8],
                                                   support, up=2, meters_per_unit=1, clearance=clearance)
    assert centre[0] > .3  # the far end from a start left of centre
    assert destination_clear_of_start(start_min, start_max, centre, up=2, clearance=clearance)
    # The target disc stays on the surface's bounds, and the footprint rests on the same surface.
    assert all(support["aabb_min"][a] + clearance - 1e-9 <= centre[a] <= support["aabb_max"][a] - clearance + 1e-9
               for a in (0, 1))
    assert centre[2] == pytest.approx(.7, abs=.01) and found["basis"] == support["basis"]


def test_a_relocated_disc_tangent_to_its_surface_stays_inside_it():
    """capture-447ae4f4 at the 0.25 size tolerance: the farthest spot left the disc on the table's
    bounds, and the conversion to the simulator frame put it 5.6e-17 m past them (refused)."""
    from blueprint_pipeline.website_support_geometry import DESTINATION_EDGE_GUARD_M, farthest_supported_destination

    # A clean rim at the far corners: at 6 mm of noise a dent of dropped faces opens there,
    # which the region now keeps clear of instead of resting tangent to the box.
    table = bumpy_plate(-.6, .6, -.4, .4, noise=.002)
    start_min, start_max = [-.15, -.1, .7], [.05, .1, .75]
    support = support_under(table, start_min, start_max, up=2, meters_per_unit=1)
    clearance = 1.25 * np.hypot(.1, .1) + .01
    # The destination's region reaches past the surface: the surface's bounds limit the move.
    centre, _, _ = farthest_supported_destination(table, start_min, start_max, [-1, -1, .6], [1, 1, .8], support,
                                               up=2, meters_per_unit=1, clearance=clearance)
    margins = [min(centre[a] - clearance - support["aabb_min"][a], support["aabb_max"][a] - centre[a] - clearance)
               for a in (0, 1)]
    assert min(margins) > 0 and min(margins) == pytest.approx(DESTINATION_EDGE_GUARD_M, abs=1e-12)


def test_no_destination_is_placed_when_the_surface_leaves_no_room_off_the_start():
    from blueprint_pipeline.website_support_geometry import farthest_supported_destination

    table = bumpy_plate(-.2, .2, -.2, .2)
    start_min, start_max = [-.1, -.1, .7], [.1, .1, .75]
    support = support_under(table, start_min, start_max, up=2, meters_per_unit=1)
    assert farthest_supported_destination(table, start_min, start_max, [-.2, -.2, .6], [.2, .2, .8], support,
                                          up=2, meters_per_unit=1, clearance=np.hypot(.1, .1) + .01) is None


def test_a_relocated_destination_never_lands_on_another_surface():
    from blueprint_pipeline.website_support_geometry import farthest_supported_destination

    # The grounded region spans a second, separate counter: only the start's own surface is eligible.
    counter = bumpy_plate(-.6, .1, -.4, .4)
    mesh = trimesh.util.concatenate([counter, bumpy_plate(.3, 1.0, -.4, .4, seed=3)])
    start_min, start_max = [-.25, -.1, .7], [-.05, .1, .75]
    support = support_under(mesh, start_min, start_max, up=2, meters_per_unit=1)
    clearance = np.hypot(.1, .1) + .01
    moved = farthest_supported_destination(mesh, start_min, start_max, [-.6, -.4, .6], [1.0, .4, .8], support,
                                           up=2, meters_per_unit=1, clearance=clearance)
    assert moved is None or moved[0][0] <= .1


def test_a_round_table_is_searched_past_its_unsupported_bounding_box_corners():
    """Codex review of #3136: on a round top the farthest grid points are its bounding box's empty corners."""
    from blueprint_pipeline.website_support_geometry import (destination_clear_of_start,
                                                             farthest_supported_destination)

    plate = bumpy_plate(-.8, .8, -.8, .8, noise=.002)
    keep = np.linalg.norm(plate.triangles_center[:, :2], axis=1) <= .8
    table = trimesh.Trimesh(vertices=plate.vertices, faces=plate.faces[keep], process=False)
    start_min, start_max = [-.1, -.1, .7], [.1, .1, .75]
    support = support_under(table, start_min, start_max, up=2, meters_per_unit=1)
    clearance = np.hypot(.1, .1) + .01
    moved = farthest_supported_destination(table, start_min, start_max, [-.8, -.8, .6], [.8, .8, .8], support,
                                           up=2, meters_per_unit=1, clearance=clearance)
    assert moved is not None
    centre, found, on_surface = moved
    assert destination_clear_of_start(start_min, start_max, centre, up=2, clearance=clearance)
    # Every corner of the moved footprint is on the round top.
    assert all(np.hypot(centre[0] + a * .1, centre[1] + b * .1) <= .8 for a in (-1, 1) for b in (-1, 1))


def test_a_coarsely_triangulated_surface_keeps_its_supported_destinations():
    """Codex review of #3139: a 2 m plate of two triangles has no face centre near most footprints."""
    from blueprint_pipeline.website_support_geometry import farthest_supported_destination

    start_min, start_max = [-.1, -.1, .7], [.1, .1, .75]
    support = support_under(plate(), start_min, start_max, up=2, meters_per_unit=1)
    moved = farthest_supported_destination(plate(), start_min, start_max, [-1, -1, .6], [1, 1, .8], support,
                                           up=2, meters_per_unit=1, clearance=np.hypot(.1, .1) + .01)
    assert moved is not None and max(abs(moved[0][0]), abs(moved[0][1])) > .5


def _cut(plate, keep):
    """The faces of ``plate`` whose centres pass ``keep(x, y)``: a top of any outline, holes and all."""
    centres = plate.triangles_center
    return trimesh.Trimesh(vertices=plate.vertices, faces=plate.faces[keep(centres[:, 0], centres[:, 1])],
                           process=False)


@pytest.mark.parametrize("hole, centre, radius, holds", [
    (None, (0, 0), .5, True),
    # A 4 cm gap (a face the fit dropped, where the removed object sat): within twice the 3 cm window.
    (.02, (0, 0), .5, True),
    # A 30 cm sink: an edge, wherever the region reaches it, or when the region sits inside it.
    (.15, (0, 0), .5, False),
    (.15, (0, 0), .1, False),
    (.15, (.4, .4), .3, True),
    # Past the top's own edge, though inside its bounding box.
    (None, (.75, .75), .3, False),
])
def test_a_support_outline_is_the_surface_its_faces_cover(hole, centre, radius, holds):
    from blueprint_pipeline.website_support_geometry import support_outline

    plate = bumpy_plate(-1, 1, -1, 1, noise=.002, step=.02)
    top = _cut(plate, lambda x, y: (np.hypot(x, y) <= 1) & ((np.abs(x) >= (hole or 0)) | (np.abs(y) >= (hole or 0))))
    outline = support_outline(top, {"face_indices": list(range(len(top.faces)))}, up=2, meters_per_unit=1)
    assert outline.window == .03
    assert outline.holds(centre, radius, tolerance=1e-9) is holds
    # A footprint's own support with no faces (a registered floor plane) has no outline: its box is all there is.
    assert support_outline(top, {"face_indices": []}, up=2, meters_per_unit=1) is None



def test_a_tapered_gap_is_an_edge_where_it_is_wider_than_its_search_bridges():
    """Codex review of #3170: a coarse top with a triangular hole, 10 cm wide at one end, closing to a point
    a metre away. Surface resumes across each long edge's middle within twice the 3 cm window, but not
    across the wide part, where a point 3.5 cm from any face lies under the region."""
    from blueprint_pipeline.website_support_geometry import support_outline

    corners = [(-.5, -.5), (1.5, -.5), (1.5, .5), (-.5, .5), (0, -.05), (1, 0), (0, .05)]
    faces = [[0, 1, 5], [0, 5, 4], [1, 2, 5], [2, 3, 5], [3, 6, 5], [3, 0, 4], [3, 4, 6]]
    top = trimesh.Trimesh(vertices=[[x, y, .7] for x, y in corners], faces=faces, process=False)
    outline = support_outline(top, {"face_indices": list(range(len(faces)))}, up=2, meters_per_unit=1)
    assert outline.distance((.3, 0), .1) == pytest.approx(.035 * np.cos(np.arctan(.05)), abs=1e-3)
    assert not outline.holds((.3, .08), .2, tolerance=1e-9)
    # Where the hole narrows below twice the window it is bridged: a region over its point is held.
    assert outline.holds((.9, .2), .18, tolerance=1e-9)

#: capture-447ae4f4 in its simulator frame (Z up, metres): the dining table's oval top, 1.29 x 1.57 m
#: around its bounding box's centre at 0.802 m, the tray's estimated box, the registration's residual
#: near the task, and the runtime frame (Y up, 1.1033 m per unit) the relocation searches in.
OVAL_CENTRE, OVAL_SEMI_AXES, OVAL_TOP = (-0.356, -0.271), (0.646, 0.786), 0.802
TRAY_447 = ([-0.5893, -0.5637, 0.8019], [-0.2759, -0.0425, 0.8468])
RESIDUAL_447, MPU_447 = 0.072, 1.103344777840341
RUNTIME_TO_SIMULATOR_447 = [[MPU_447, 0, 0, 0], [0, 0, -MPU_447, 0], [0, MPU_447, 0, 0], [0, 0, 0, 1]]


def _to_runtime(points):
    return np.asarray(points, dtype=float) @ np.linalg.inv(np.asarray(RUNTIME_TO_SIMULATOR_447)[:3, :3]).T


def _to_simulator(point):
    return (np.asarray(RUNTIME_TO_SIMULATOR_447)[:3, :3] @ np.asarray(point, dtype=float)).tolist()


def oval_table_447():
    """A capture-447ae4f4-like reconstructed oval top, in its runtime frame: a 2 cm grid wandering 3 mm,
    with a 12 x 20 cm gap beside the tray where the removed tray left one in the real mesh."""
    (cx, cy), (a, b) = OVAL_CENTRE, OVAL_SEMI_AXES
    plate = bumpy_plate(cx - a, cx + a, cy - b, cy + b, OVAL_TOP, noise=.003, step=.02)
    top = _cut(plate, lambda x, y: (((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1)
               & ~((np.abs(x + .59) < .06) & (np.abs(y + .57) < .1)))
    return trimesh.Trimesh(vertices=_to_runtime(top.vertices), faces=top.faces, process=False)


def _on_oval(centre, radius, slack):
    """Every point of the disc lies within the oval grown by ``slack`` (the grid's jagged rim)."""
    (cx, cy), (a, b) = OVAL_CENTRE, OVAL_SEMI_AXES
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    x, y = centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)
    return bool(np.all(((x - cx) / (a + slack)) ** 2 + ((y - cy) / (b + slack)) ** 2 <= 1))


def test_a_relocated_region_on_an_oval_table_lies_wholly_on_it(monkeypatch):
    """capture-447ae4f4: the region was admitted against the oval top's bounding box; 8.9 cm of it hung past the edge."""
    from blueprint_pipeline.website_support_geometry import (SupportOutline, destination_clear_of_start,
                                                             farthest_supported_destination, support_outline)
    from blueprint_pipeline.website_surface_target_admission import admit_surface_target, website_target_radius

    table = oval_table_447()
    lower, upper = (_to_runtime(corner).tolist() for corner in TRAY_447)
    lower, upper = np.minimum(lower, upper).tolist(), np.maximum(lower, upper).tolist()
    support = support_under(table, lower, upper, up=1, meters_per_unit=MPU_447, placement_residual_m=RESIDUAL_447)
    assert support["basis"] == "fitted_surface_under_footprint"
    clearance = website_target_radius(lower, upper, relative_error=.25, up=1, meters_per_unit=MPU_447)
    radius = website_target_radius(*TRAY_447, relative_error=.25)
    assert radius == pytest.approx(.39, abs=.001)
    search = {"up": 1, "meters_per_unit": MPU_447, "placement_residual_m": RESIDUAL_447, "clearance": clearance}
    # The destination was grounded as the whole table, its centre under the tray.
    moved = farthest_supported_destination(table, lower, upper, support["aabb_min"], support["aabb_max"], support,
                                           **search)
    centre = _to_simulator(moved.centre)
    assert moved.region_on_surface and _on_oval(centre, radius, slack=.02)
    assert destination_clear_of_start(lower, upper, moved.centre, up=1, clearance=clearance)
    # Still the other end of the table: its far half along the long axis.
    assert abs(centre[1] - OVAL_CENTRE[1]) > .3
    # The rule before: the farthest supported footprint, its region bounded by the top's box alone.
    with monkeypatch.context() as before:
        before.setattr(SupportOutline, "holds", lambda *args, **kwargs: False)
        old = farthest_supported_destination(table, lower, upper, support["aabb_min"], support["aabb_max"], support,
                                             **search)
    assert not old.region_on_surface and not _on_oval(_to_simulator(old.centre), radius, slack=.02)
    # Admission: both inside the top's box; against the top itself only the new one.
    outline = support_outline(table, support, up=1, meters_per_unit=MPU_447,
                              placement_residual_m=RESIDUAL_447).in_frame(RUNTIME_TO_SIMULATOR_447)
    bounds = dict(zip(("aabb_min_xyz", "aabb_max_xyz"), (np.sort([_to_simulator(support["aabb_min"]),
                                                                  _to_simulator(support["aabb_max"])], axis=0)
                                                          .tolist())))

    def admit(position, **kwargs):
        destination = {"relation": "on", "visible_label": "the other end of the same table",
                       "mode": "existing_support_surface", "position_world_m": position,
                       "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                       "basis": "registered_estimated_visible_bounds_and_collider_contact"}
        return admit_surface_target(destination=destination, support=bounds, subject_min=TRAY_447[0],
                                    subject_max=TRAY_447[1], placement_uncertainty_m=RESIDUAL_447,
                                    development_support=False, relative_error=.25, **kwargs)

    assert admit(_to_simulator(old.centre))["radius_m"] == radius
    with pytest.raises(ValueError, match="surface_target_outside_support"):
        admit(_to_simulator(old.centre), support_outline=outline)
    assert admit(centre, support_outline=outline) == admit(centre)


def test_a_region_that_fits_nowhere_on_its_surface_is_placed_by_the_rule_before_and_flagged():
    """An L-shaped counter 44 cm deep: the region holding a 30 cm tray at any yaw is wider than either arm."""
    from blueprint_pipeline.website_support_geometry import farthest_supported_destination, support_outline

    counter = _cut(bumpy_plate(0, 2, 0, 2, noise=.002, step=.02), lambda x, y: (x < .45) | (y < .45))
    start_min, start_max = [.85, .075, .7], [1.15, .375, .75]
    support = support_under(counter, start_min, start_max, up=2, meters_per_unit=1)
    clearance = 1.25 * np.hypot(.15, .15) + .01
    moved = farthest_supported_destination(counter, start_min, start_max, [0, 0, .6], [2, 2, .8], support,
                                           up=2, meters_per_unit=1, clearance=clearance)
    # The footprint rests on the far arm, as before; the admission is told its region does not.
    assert moved is not None and not moved.region_on_surface
    assert moved.centre[1] > 1.5 and moved.centre[0] - .15 >= 0 and moved.centre[0] + .15 <= .45
    outline = support_outline(counter, support, up=2, meters_per_unit=1)
    assert not outline.holds(moved.centre[:2], clearance, tolerance=1e-9)


#: origin/main before the region had to lie on its surface: the farthest corner of a rectangle and its seal.
RECTANGLE_BEFORE = {
    "exact": ([-0.2616297273083672, -0.4116297273083672, 0.8],
              "sha256:fa171e5da127acf89c791d0740f67cb40b155f4e2352f0a1c22c41e2be4a8bdd"),
    "reconstructed": ([0.2616297273083683, 0.41162972730836855, 0.8012306227985251],
                      "sha256:1fc6a3aee9675228321fff697518b869aba5caa622aa252e20b5ffa4c586602e"),
}


@pytest.mark.parametrize("case", sorted(RECTANGLE_BEFORE))
def test_a_rectangular_top_places_and_seals_its_region_bit_for_bit_as_before(case):
    """A rectangle's bounding box is its surface: the same corner, the same sealed region and digest."""
    from blueprint_pipeline.website_support_geometry import farthest_supported_destination, support_outline
    from blueprint_pipeline.website_surface_target_admission import admit_surface_target, website_target_radius

    table = (plate(-.65, .65, -.8, .8, height=.8) if case == "exact"
             else bumpy_plate(-.65, .65, -.8, .8, height=.8, noise=.002))
    tray_min, tray_max = [-0.155, -0.26, 0.80], [0.155, 0.26, 0.845]
    support = support_under(table, tray_min, tray_max, up=2, meters_per_unit=1, placement_residual_m=.072)
    moved = farthest_supported_destination(table, tray_min, tray_max, [-.65, -.8, .7], [.65, .8, .9], support,
        up=2, meters_per_unit=1, placement_residual_m=.072,
        clearance=website_target_radius(tray_min, tray_max, relative_error=.25))
    centre, digest = RECTANGLE_BEFORE[case]
    assert moved.centre == centre and moved.region_on_surface
    destination = {"relation": "on", "visible_label": "the other end of the table", "mode": "existing_support_surface",
                   "position_world_m": moved.centre, "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                   "basis": "registered_estimated_visible_bounds_and_collider_contact"}
    target = admit_surface_target(destination=destination,
        support={"aabb_min_xyz": support["aabb_min"], "aabb_max_xyz": support["aabb_max"]},
        subject_min=tray_min, subject_max=tray_max, placement_uncertainty_m=.072, development_support=False,
        relative_error=.25, support_outline=support_outline(table, support, up=2, meters_per_unit=1,
                                                            placement_residual_m=.072))
    assert target["target_digest"] == digest
