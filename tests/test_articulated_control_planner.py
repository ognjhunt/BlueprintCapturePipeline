from __future__ import annotations

import math

import pytest

from blueprint_pipeline.articulated_control_planner import (
    ARTICULATED_CONTROL_PLAN_SCHEMA_VERSION,
    ArticulatedControlPlannerError,
    plan_articulated_handle_trajectory,
)


def _hinge() -> dict:
    """The 840796 upper door, in the units the sealed asset actually uses."""

    return {
        "hinge_point_world_m": [-0.357, 0.350, 1.276],
        "hinge_axis_world": [0.0, 0.0, 1.0],
        "handle_grasp_point_closed_world_m": [0.300, 0.349, 1.300],
        "open_angle_degrees": 50.0,
        "authored_limit_degrees": 90.0,
    }


def _plan(**overrides):
    arguments = {**_hinge(), "waypoint_count": 6, "approach_standoff_m": 0.12}
    arguments.update(overrides)
    return plan_articulated_handle_trajectory(**arguments)


def test_the_grasp_point_rides_the_arc_the_door_actually_sweeps() -> None:
    """A door handle does not travel in a straight line.

    Every waypoint has to keep the handle's distance from the hinge axis
    constant, or the arm is being commanded to pull the handle off the door.
    """

    plan = _plan()

    radii = {round(row["radius_m"], 9) for row in plan["waypoints"]}
    assert len(radii) == 1
    assert plan["waypoints"][0]["door_angle_degrees"] == 0.0
    assert plan["waypoints"][-1]["door_angle_degrees"] == pytest.approx(50.0)
    assert plan["schema_version"] == ARTICULATED_CONTROL_PLAN_SCHEMA_VERSION


def test_the_arc_is_perpendicular_to_the_hinge_axis() -> None:
    """A revolute joint moves the handle in one plane and no other."""

    plan = _plan()

    heights = {round(row["position_world_m"][2], 9) for row in plan["waypoints"]}
    assert len(heights) == 1


def test_the_approach_stands_off_along_the_outward_normal() -> None:
    """Approaching through the door would drive the gripper into it."""

    plan = _plan()

    approach = plan["approach_pose"]["position_world_m"]
    grasp = plan["waypoints"][0]["position_world_m"]
    assert approach[1] > grasp[1]
    assert math.dist(approach, grasp) == pytest.approx(0.12)


def test_opening_beyond_the_authored_limit_fails_closed() -> None:
    """Commanding past the limit would score the solver, not the door."""

    with pytest.raises(ArticulatedControlPlannerError) as excinfo:
        _plan(open_angle_degrees=120.0)

    assert any("beyond_authored_limit" in e for e in excinfo.value.errors)


def test_a_handle_on_the_hinge_axis_fails_closed() -> None:
    """Zero lever arm cannot be opened by pulling, and no arc exists."""

    with pytest.raises(ArticulatedControlPlannerError) as excinfo:
        _plan(handle_grasp_point_closed_world_m=[-0.357, 0.350, 1.400])

    assert any("handle_on_hinge_axis" in e for e in excinfo.value.errors)


def test_the_planner_reports_the_torque_the_arm_must_supply() -> None:
    """The lever arm is what converts a hinge drive into a handle force.

    Without it a passing scripted positive says nothing about whether a real
    arm could do the same thing.
    """

    plan = _plan(joint_damping_n_m_s_per_rad=14.0, sweep_duration_s=2.0)

    load = plan["required_load"]
    assert load["lever_arm_m"] == pytest.approx(0.657, abs=0.01)
    assert load["mean_angular_velocity_rad_s"] == pytest.approx(0.436, abs=0.01)
    assert load["hinge_torque_n_m"] == pytest.approx(6.11, abs=0.05)
    assert load["handle_force_n"] == pytest.approx(9.3, abs=0.2)


def test_the_planner_is_deterministic() -> None:
    assert _plan() == _plan()


def test_a_non_axis_aligned_hinge_still_produces_a_planar_arc() -> None:
    """Nothing here may assume the door hangs on a world-vertical hinge."""

    axis = [1.0, 1.0, 1.0]
    plan = _plan(
        hinge_axis_world=axis,
        hinge_point_world_m=[0.0, 0.0, 0.0],
        handle_grasp_point_closed_world_m=[0.5, -0.5, 0.0],
    )

    norm = math.sqrt(3.0)
    unit = [value / norm for value in axis]
    projections = {
        round(sum(p * u for p, u in zip(row["position_world_m"], unit)), 9)
        for row in plan["waypoints"]
    }
    assert len(projections) == 1


def test_the_arm_retreats_along_the_final_outward_normal() -> None:
    """The task scores the door holding *after release*.

    Backing off along the door's closed-state normal would swing the gripper
    through the opened panel, so the retreat has to follow the normal as it is
    once the door has moved.
    """

    plan = _plan()

    retreat = plan["retreat_pose"]["position_world_m"]
    final = plan["waypoints"][-1]["position_world_m"]
    assert math.dist(retreat, final) == pytest.approx(0.12)
    hinge = _hinge()["hinge_point_world_m"]
    assert math.dist(retreat[:2], hinge[:2]) > math.dist(final[:2], hinge[:2])


def test_every_phase_is_named_so_a_failure_says_where_it_stopped() -> None:
    plan = _plan()

    phases = [row["phase_id"] for row in plan["phases"]]
    assert phases[0] == "approach"
    assert "grasp" in phases
    assert "release" in phases
    assert phases[-1] == "retreat"
    assert len({row["phase_id"] for row in plan["phases"]}) == len(phases)
