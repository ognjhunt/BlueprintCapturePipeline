"""The table's height by physics against the height the cameras see (capture-447ae4f4's shapes)."""
from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline import website_support_height_agreement as agreement

START = ([-0.59, -0.56], [-0.28, -0.04])            # 447's tray footprint
GOAL = ([-0.23, 0.10], 0.39)                         # 447's destination disc


def _plane_triangles(z_of_xy, step=0.05):
    xs = np.arange(-1.0, 0.6, step)
    ys = np.arange(-0.8, 0.6, step)
    tris = []
    for x in xs:
        for y in ys:
            corners = [(x, y), (x + step, y), (x + step, y + step), (x, y + step)]
            points = [[cx, cy, z_of_xy(cx, cy)] for cx, cy in corners]
            tris += [[points[0], points[1], points[2]], [points[0], points[2], points[3]]]
    return np.array(tris)


def _particle_sheet(z_of_xy, step=0.01):
    """A sheet of flat, opaque particles (3 mm thick): the table as the cameras see it."""
    xs, ys = np.arange(-1.0, 0.6, step), np.arange(-0.8, 0.6, step)
    positions = np.array([[x, y, z_of_xy(x, y)] for x in xs for y in ys])
    inverse = np.repeat(np.diag([1 / 0.01 ** 2, 1 / 0.01 ** 2, 1 / 0.003 ** 2])[None], len(positions), axis=0)
    return positions, inverse, np.full(len(positions), 0.95)


def _measure(collider, picture, *, plane=0.8039):
    positions, inverse, opacities = _particle_sheet(picture)
    return agreement.measure(triangles=_plane_triangles(collider), positions=positions,
                             inverse_covariances=inverse, opacities=opacities, start_minimum_xy=START[0],
                             start_maximum_xy=START[1], goal_centre_xy=GOAL[0], goal_radius_m=0.2,
                             task_plane_z_m=plane)


def test_447s_shape_is_recorded_and_not_flagged() -> None:
    # Collider 4 mm over the picture's table, the tray resting 1.2 cm over it: under the 2 cm flag.
    record = _measure(lambda x, y: 0.796, lambda x, y: 0.792)
    assert record["flags"] == [] and record["one_flat_offset"] is True and record["correction_applied"] is False
    assert record["start"]["gap_collider_minus_picture"]["median_m"] == pytest.approx(0.004, abs=0.002)
    assert record["start_task_plane_minus_picture_m"] == pytest.approx(0.012, abs=0.002)


def test_more_than_2_cm_under_the_object_or_at_the_goal_is_flagged() -> None:
    record = _measure(lambda x, y: 0.80, lambda x, y: 0.774)          # 3 cm under the resting height
    assert record["flags"] == [agreement.FLAG] and record["stops_preparation"] is False
    goal_only = _measure(lambda x, y: 0.80, lambda x, y: 0.79 if y < 0.0 else 0.775)
    assert goal_only["goal_task_plane_minus_picture_m"] > agreement.FLAG_GAP_M and goal_only["flags"] == [agreement.FLAG]


def test_a_tilt_is_recorded_beside_the_flag_and_never_corrected() -> None:
    record = _measure(lambda x, y: 0.80, lambda x, y: 0.79 + 0.13 * (x + 0.59))
    assert record["one_flat_offset"] is False and record["correction_applied"] is False


def test_a_measurement_that_cannot_run_never_stops_preparation() -> None:
    from blueprint_pipeline.website_support_height_agreement import preparation_record as support_height_agreement

    record = support_height_agreement({}, base_scene={}, masks={}, frames=[])
    assert record["status"] == "not_assessed" and record["flags"] == [] and record["stops_preparation"] is False
