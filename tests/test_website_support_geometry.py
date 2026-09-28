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
