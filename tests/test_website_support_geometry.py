import numpy as np
import pytest
import trimesh

from blueprint_pipeline.website_support_geometry import support_under


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
