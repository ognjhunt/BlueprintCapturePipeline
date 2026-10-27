"""Beta containers need observed cavity geometry, never an enclosing mask box."""
import numpy as np
import pytest
import trimesh

from blueprint_pipeline.website_container_geometry import container_interior


def cavity(*, missing=None):
    parts = [
        ("floor", [.62, .62, .02], [0, 0, .69]),
        ("left", [.02, .62, .4], [-.31, 0, .9]),
        ("right", [.02, .62, .4], [.31, 0, .9]),
        ("front", [.62, .02, .4], [0, -.31, .9]),
        ("back", [.62, .02, .4], [0, .31, .9]),
    ]
    meshes = []
    for name, size, center in parts:
        if name != missing:
            mesh = trimesh.creation.box(size)
            mesh.apply_translation(center)
            meshes.append(mesh)
    return trimesh.util.concatenate(meshes)


def query(mesh, **kwargs):
    return container_interior(mesh, [-.33, -.33, .68], [.33, .33, 1.1],
                              subject_dimensions=[.08, .06, .14], up=2, meters_per_unit=1, **kwargs)


def test_open_bin_retains_floor_wall_evidence_and_conservative_bounds():
    value = query(cavity())
    assert value["interior_bounds"]["minimum"] == pytest.approx([-.298, -.298, .7])
    assert value["interior_bounds"]["maximum"] == pytest.approx([.298, .298, 1.1])
    assert value["floor"]["face_indices"] and value["wall_face_indices"]
    assert value["physical_measurement"] is False


def test_two_rails_cannot_substitute_for_a_continuous_container_wall():
    rails = []
    for height in (.8, 1.0):
        rail = trimesh.creation.box([.62, .02, .01])
        rail.apply_translation([0, -.31, height])
        rails.append(rail)
    assert query(trimesh.util.concatenate([cavity(missing="front"), *rails])) is None


@pytest.mark.parametrize("defect", ["floor", "left", "lid", "obstruction", "too_tall"])
def test_unseen_open_or_obstructed_interiors_are_not_invented(defect):
    mesh = cavity(missing=defect)
    if defect in {"lid", "obstruction"}:
        block = trimesh.creation.box([.5, .5, .03] if defect == "lid" else [.02, .02, .1])
        block.apply_translation([0, 0, 1.1 if defect == "lid" else .9])
        mesh = trimesh.util.concatenate([mesh, block])
    if defect == "too_tall":
        assert container_interior(mesh, [-.33, -.33, .68], [.33, .33, 1.1],
                                  subject_dimensions=[.08, .06, .5], up=2, meters_per_unit=1) is None
    else:
        assert query(mesh) is None


@pytest.mark.parametrize("sign", [-1, 1])
def test_scaled_y_up_cavity_keeps_source_frame_geometry(sign):
    mesh = cavity()
    mesh.vertices = np.asarray(mesh.vertices)[:, [0, 2, 1]] * [10, 10 * sign, 10]
    lower, upper = ([-3.3, 6.8, -3.3], [3.3, 11, 3.3]) if sign == 1 else (
        [-3.3, -11, -3.3], [3.3, -6.8, 3.3])
    value = container_interior(mesh, lower, upper, subject_dimensions=[.8, 1.4, .6],
                               up=1, meters_per_unit=.1, up_sign=sign)
    assert value is not None
    assert value["floor"]["top_runtime_units"] == pytest.approx(7 * sign)
    assert value["interior_bounds"]["minimum"][1] == pytest.approx(7 if sign == 1 else -11)
