"""Clear sub-millimetre interference that a solver turns into motion."""

from __future__ import annotations

import pytest

from blueprint_pipeline.rest_pose_interference import (
    RestPoseInterferenceError,
    plan_axis_clearance,
)


def test_a_shallow_overlap_is_cleared_along_its_shallowest_axis():
    """0.8 mm in y opened a fridge door 35 degrees; shift along y, not x.

    The same pair overlapped 13 mm in x and 6 mm in z. Pushing along either of
    those would move the part visibly to fix a fault that is under a
    millimetre deep.
    """

    plan = plan_axis_clearance(
        moving_min=[0.10, 0.1768, 1.00],
        moving_max=[0.30, 0.1948, 1.10],
        blocking_min=[0.05, -0.3504, 0.90],
        blocking_max=[0.40, 0.1776, 1.20],
        clearance_m=0.001,
    )

    assert plan["axis"] == 1
    assert plan["overlap_m"] == pytest.approx(0.0008, abs=1e-6)
    # Moving part sits above the blocker, so it shifts further positive.
    assert plan["shift_m"] == pytest.approx(0.0018, abs=1e-6)


def test_the_shift_direction_follows_which_side_the_part_is_on():
    plan = plan_axis_clearance(
        moving_min=[0.0, -0.20, 0.0],
        moving_max=[0.1, -0.0992, 0.1],
        blocking_min=[0.0, -0.10, 0.0],
        blocking_max=[0.1, 0.30, 0.1],
        clearance_m=0.001,
    )

    assert plan["axis"] == 1
    assert plan["shift_m"] < 0.0


def test_parts_that_do_not_overlap_need_no_shift():
    plan = plan_axis_clearance(
        moving_min=[0.0, 0.20, 0.0],
        moving_max=[0.1, 0.30, 0.1],
        blocking_min=[0.0, -0.10, 0.0],
        blocking_max=[0.1, 0.10, 0.1],
        clearance_m=0.001,
    )

    assert plan["shift_m"] == 0.0
    assert plan["already_clear"] is True


def test_a_deep_overlap_refuses_rather_than_nudging():
    """Centimetres of overlap is a modelling error, not a rest-pose gap.

    Shifting a part 40 mm to resolve it would move it somewhere it does not
    belong and hide the real fault.
    """

    with pytest.raises(RestPoseInterferenceError) as excinfo:
        plan_axis_clearance(
            moving_min=[0.0, 0.10, 0.0],
            moving_max=[0.1, 0.30, 0.1],
            blocking_min=[0.0, -0.10, 0.0],
            blocking_max=[0.1, 0.14, 0.1],
            clearance_m=0.001,
            maximum_shallow_overlap_m=0.005,
        )

    assert any("overlap_too_deep" in e for e in excinfo.value.errors)
