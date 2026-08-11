"""A static overview camera that a human can actually watch.

Every task run's review stream should show the WHOLE episode - robot, task
object, and the space between - from one fixed, upright viewpoint. The
policy cameras cannot do this: they are robot-mounted, tightly framed, and
exist for the policy, not the reviewer. This module turns a handful of
scene points into a camera pose with three guarantees: every point is
inside the view cone with margin, the horizon is upright (a camera nobody
can watch sideways is not review evidence), and the answer is
deterministic.
"""

from __future__ import annotations

import math

import pytest

from blueprint_pipeline.overview_camera_placement import (
    OverviewCameraPlacementError,
    plan_overview_camera,
)


ROBOT = (1.75, 1.99, 0.0)
FRIDGE = (1.9742142, 1.4792181, 0.0)
HANDLE = (2.1117, 1.8369, 1.023)
FOV = math.radians(62.0)


def _rotation_from_wxyz(q):
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def test_every_scene_point_lands_inside_the_view_cone():
    plan = plan_overview_camera(
        scene_points_world_m=[ROBOT, FRIDGE, HANDLE],
        fov_horizontal_rad=FOV,
        image_aspect=16 / 9,
    )

    rotation = _rotation_from_wxyz(plan["rotation_wxyz_opengl"])
    forward = [-rotation[i][2] for i in range(3)]
    position = plan["position_world_m"]
    half = min(FOV, FOV * (9 / 16)) / 2.0
    for point in (ROBOT, FRIDGE, HANDLE):
        ray = [point[i] - position[i] for i in range(3)]
        norm = math.sqrt(sum(v * v for v in ray))
        cosine = sum(forward[i] * ray[i] for i in range(3)) / norm
        assert math.acos(min(1.0, cosine)) < half, point


def test_the_horizon_is_upright():
    """§24: a rolled review camera is the defect this exists to prevent."""

    plan = plan_overview_camera(
        scene_points_world_m=[ROBOT, FRIDGE, HANDLE],
        fov_horizontal_rad=FOV,
        image_aspect=16 / 9,
    )

    rotation = _rotation_from_wxyz(plan["rotation_wxyz_opengl"])
    camera_up = [rotation[i][1] for i in range(3)]
    assert camera_up[2] > 0.9
    assert plan["receipt"]["horizon_upright"] is True


def test_the_camera_stands_back_from_the_scene():
    plan = plan_overview_camera(
        scene_points_world_m=[ROBOT, FRIDGE, HANDLE],
        fov_horizontal_rad=FOV,
        image_aspect=16 / 9,
    )

    position = plan["position_world_m"]
    for point in (ROBOT, FRIDGE, HANDLE):
        assert math.dist(position, point) > 1.0
    assert position[2] > max(p[2] for p in (ROBOT, FRIDGE, HANDLE))


def test_placement_is_deterministic():
    kwargs = dict(
        scene_points_world_m=[ROBOT, FRIDGE, HANDLE],
        fov_horizontal_rad=FOV,
        image_aspect=16 / 9,
    )

    assert plan_overview_camera(**kwargs) == plan_overview_camera(**kwargs)


def test_fewer_than_two_points_is_refused():
    with pytest.raises(OverviewCameraPlacementError):
        plan_overview_camera(
            scene_points_world_m=[ROBOT],
            fov_horizontal_rad=FOV,
            image_aspect=16 / 9,
        )
