"""Saved-view free space protects observed/occluded surfaces without provider calls."""
import numpy as np
import pytest

from blueprint_pipeline.atlas_depth_splat import posed_frame
from blueprint_pipeline.atlas_mesh_registration import mesh_depth_free_space_filter
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def view(frame_id, depth=2.0, confidence=1.0, x=0.0):
    return posed_frame(frame_id=frame_id, rgb=np.zeros((64, 64, 3), np.uint8),
        depth=np.full((64, 64), depth, np.float32),
        confidence=np.full((64, 64), confidence, np.float32),
        camera={"intrinsics": {"fx": 32., "fy": 32., "cx": 32., "cy": 32., "width": 64, "height": 64},
                "extrinsics": {"position": [x, 0., 0.], "quaternion": [0., 0., 0., 1.], "coordinateSystem": "rub"}})


def face(z):
    return np.array([[-.1, -.1, z], [.1, -.1, z], [0., .1, z]])


def test_only_multi_view_free_space_faces_are_removed():
    triangles = np.array([face(-1.), face(-2.), face(-3.), face(1.), face(-2.01)])
    before = triangles.copy()
    keep, report = mesh_depth_free_space_filter(triangles,
        frames=[view("left", x=-.2), view("right", x=.2)], absolute_tolerance=.03)
    assert keep.tolist() == [False, True, True, True, True]
    assert np.array_equal(before, triangles)
    assert report["faces_removed"] == 1
    assert report["digest"] == canonical_digest(report, digest_field="digest")


@pytest.mark.parametrize("frames", [[view("one")], [view("a"), view("b", confidence=0)],
    [view("a"), view("b", depth=np.nan)], [view("a"), view("b"), view("agrees", depth=1.)]])
def test_missing_second_witness_or_any_agreement_preserves_face(frames):
    keep, _ = mesh_depth_free_space_filter(np.array([face(-1.)]), frames=frames, absolute_tolerance=.03)
    assert keep.tolist() == [True]


def test_a_vertex_touching_observed_surface_protects_a_crossing_face():
    triangle = face(-1.)
    triangle[0, 2] = -2.
    keep, _ = mesh_depth_free_space_filter(np.array([triangle]),
        frames=[view("a", x=-.2), view("b", x=.2)], absolute_tolerance=.03)
    assert keep.tolist() == [True]


def test_depth_discontinuity_and_duplicate_reference_are_not_empty_space():
    frame = view("a")
    frame["depth"][::2, ::2] = 20.
    keep, _ = mesh_depth_free_space_filter(np.array([face(-1.)]),
        frames=[frame, view("b", x=.2)], absolute_tolerance=.03)
    assert keep.tolist() == [True]
    with pytest.raises(ValueError, match="reference_identity"):
        mesh_depth_free_space_filter(np.array([face(-1.)]), frames=[view("a"), view("a")], absolute_tolerance=.03)


def test_aliases_of_the_same_depth_and_camera_count_as_one_witness():
    keep, report = mesh_depth_free_space_filter(np.array([face(-1.)]),
        frames=[view("a"), view("alias")], absolute_tolerance=.03)
    assert keep.tolist() == [True]
    assert report["distinct_reference_count"] == 1


def test_nonfinite_geometry_and_invalid_tolerance_refuse():
    for triangles, tolerance in [(np.array([face(np.nan)]), .03), (np.array([face(-1.)]), 0)]:
        with pytest.raises(ValueError, match="geometry_or_tolerance"):
            mesh_depth_free_space_filter(triangles, frames=[view("a")], absolute_tolerance=tolerance)


def test_projected_camera_must_match_the_retained_pose():
    frame = view("a")
    frame["position"] = np.array([10., 0., 0.])
    with pytest.raises(ValueError, match="camera_binding_invalid"):
        mesh_depth_free_space_filter(np.array([face(-1.)]), frames=[frame], absolute_tolerance=.03)
