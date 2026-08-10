"""A hollow shell's convex hull is a solid block."""

from __future__ import annotations

import pytest

from blueprint_pipeline.hollow_collider_approximation import (
    HollowColliderApproximationError,
    classify_collider_approximation,
)


def test_a_hollow_shell_needs_a_concave_approximation():
    """component_008: 606 litres of AABB from 95 points, holding a cavity.

    Its convexHull collider is a solid block filling the fridge, so the door
    bins that belong inside the cavity are permanently interpenetrating it.
    PhysX pushes them out and the door swings 35 degrees before anything
    touches it.
    """

    verdict = classify_collider_approximation(
        aabb_extents_m=[0.713, 0.528, 1.61],
        point_count=95,
        current_approximation="convexHull",
    )

    assert verdict["hull_is_unsafe"] is True
    assert verdict["recommended_approximation"] == "convexDecomposition"


def test_a_small_solid_part_is_fine_as_a_hull():
    """A knob is convex enough that its hull is the part."""

    verdict = classify_collider_approximation(
        aabb_extents_m=[0.04, 0.033, 0.02],
        point_count=25,
        current_approximation="convexHull",
    )

    assert verdict["hull_is_unsafe"] is False
    assert verdict["recommended_approximation"] == "convexHull"


def test_a_thin_slab_is_fine_however_large():
    """A shelf spans the cavity but encloses nothing."""

    verdict = classify_collider_approximation(
        aabb_extents_m=[0.572, 0.381, 0.010],
        point_count=8,
        current_approximation="convexHull",
    )

    assert verdict["hull_is_unsafe"] is False


def test_an_already_concave_collider_is_left_alone():
    verdict = classify_collider_approximation(
        aabb_extents_m=[0.713, 0.528, 1.61],
        point_count=95,
        current_approximation="convexDecomposition",
    )

    assert verdict["hull_is_unsafe"] is False
    assert verdict["recommended_approximation"] == "convexDecomposition"


def test_a_degenerate_part_refuses_rather_than_guessing():
    with pytest.raises(HollowColliderApproximationError):
        classify_collider_approximation(
            aabb_extents_m=[0.0, 0.5, 0.5],
            point_count=8,
            current_approximation="convexHull",
        )
