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


def bumpy_plate(x0=-.3, x1=.3, y0=-.3, y1=.3, height=.7, *, noise=.006, slope=0.0, seed=0):
    """A reconstructed counter: a dense 1 cm grid whose vertices wander millimetres, as Atlas meshes do."""
    rng = np.random.default_rng(seed)
    xs, ys = np.arange(x0, x1 + 1e-9, .01), np.arange(y0, y1 + 1e-9, .01)
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
    centre, found = farthest_supported_destination(table, start_min, start_max, [-.6, -.4, .6], [.6, .4, .8],
                                                   support, up=2, meters_per_unit=1, clearance=clearance)
    assert centre[0] > .3  # the far end from a start left of centre
    assert destination_clear_of_start(start_min, start_max, centre, up=2, clearance=clearance)
    # The target disc stays on the surface's bounds, and the footprint rests on the same surface.
    assert all(support["aabb_min"][a] + clearance - 1e-9 <= centre[a] <= support["aabb_max"][a] - clearance + 1e-9
               for a in (0, 1))
    assert centre[2] == pytest.approx(.7, abs=.01) and found["basis"] == support["basis"]


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
    centre, found = moved
    assert destination_clear_of_start(start_min, start_max, centre, up=2, clearance=clearance)
    # Every corner of the moved footprint is on the round top.
    assert all(np.hypot(centre[0] + a * .1, centre[1] + b * .1) <= .8 for a in (-1, 1) for b in (-1, 1))
