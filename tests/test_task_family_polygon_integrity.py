"""AUT-14: convex scoring must never accept non-simple native rings."""
from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline import task_evaluation_task_family_native_adapter as adapter
from blueprint_pipeline.task_family_native_contracts import _area, _polygon


def _star():
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points[[0, 2, 4, 1, 3]]


@pytest.mark.parametrize("ring", [
    _star(),
    [[0, 0], [1, 1], [0, 1], [1, 0]],  # bowtie
    [[0, 0], [2, 0], [1, 0], [2, 1], [0, 1]],  # backtracking
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 1]],  # repeated nonadjacent vertex
    [[0, 0], [2, 0], [2, 2], [1, 0], [0, 2]],  # edge touches another edge
])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_non_simple_ring_is_refused_at_every_scale(ring, scale):
    with pytest.raises(ValueError, match="task_family_native_polygon_"):
        _polygon(np.asarray(ring) * scale)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_convex_boundary_with_collinear_vertices_normalizes_both_orientations(reverse, scale):
    ring = np.array([[0, 0], [.5, 0], [1, 0], [1, 1], [0, 1]]) * scale
    polygon = _polygon(ring[::-1] if reverse else ring)
    assert _area(polygon) == pytest.approx(scale * scale)
    assert adapter._cloth_overlap_fraction(polygon, polygon) == pytest.approx(1)


def test_cloth_self_intersection_is_zero_progress_with_retained_diagnostic():
    star = _star().tolist()
    row = adapter._stage_measurements("cloth_folding", {
        "keypoint_positions_m": [[0, 0, 0]], "target_keypoint_positions_m": [[0, 0, 0]],
        "first_fold_polygon_xy_m": star, "second_fold_polygon_xy_m": star,
        "vertex_velocities_m_s": [[0, 0, 0]], "robot_contact_active": False,
    })
    assert row["fold_overlap_fraction"] == 0
    assert row["measurement_diagnostics"] == ["cloth_projection_not_simple:first"]


@pytest.mark.parametrize("center,expected", [([.5, .5, 0], .5), ([0, .5, 0], 0), ([-.1, .5, 0], -.1)])
def test_simple_support_margins_match_inside_boundary_and_outside(center, expected):
    row = adapter._stage_measurements("stacking", {
        "support_polygon_xy_m": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "center_of_mass_m": center, "support_normal_forces_n": [1],
        "robot_contact_active": False,
        "linear_velocity_m_s": [0, 0, 0], "angular_velocity_rad_s": [0, 0, 0],
    })
    assert row["com_support_margin_m"] == pytest.approx(expected)


def test_self_intersecting_support_cannot_produce_a_positive_margin():
    with pytest.raises(ValueError, match="polygon_not_simple"):
        adapter._stage_measurements("stacking", {
            "support_polygon_xy_m": _star(), "center_of_mass_m": [0, 0, 0],
            "support_normal_forces_n": [1], "robot_contact_active": False,
            "linear_velocity_m_s": [0, 0, 0], "angular_velocity_rad_s": [0, 0, 0],
        })


def test_authored_star_support_is_refused_by_prelaunch_materialization():
    from tests.test_task_evaluation_task_family_native_adapter import _runtime_contract, _task

    task = _task("stacking", criteria={"minimum_support_margin_m": .05,
        "maximum_linear_speed_m_s": .1, "maximum_angular_speed_rad_s": .1},
        reset={"support_contact": False, "com_support_margin_m": -.1,
               "linear_speed_m_s": 0, "angular_speed_rad_s": 0, "robot_contact_active": False})
    with pytest.raises(ValueError, match="polygon_not_simple"):
        _runtime_contract(task, roles={"object": "object1", "support": "support1"},
                          geometry={"support_polygon_local_xy_m": _star().tolist()})
