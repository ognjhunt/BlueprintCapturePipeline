from __future__ import annotations

import math

import pytest

from blueprint_pipeline.franka_kinematics import (
    FRANKA_JOINT_LIMITS_RAD,
    FrankaKinematicsError,
    forward_kinematics,
    manipulability,
    radial_force_capability_n,
    solve_position_ik,
)


def test_the_home_pose_matches_the_published_flange_height() -> None:
    """Franka's own documentation puts the flange 0.926 m up at all-zeros."""

    position, _ = forward_kinematics([0.0] * 7)

    assert position[0] == pytest.approx(0.088, abs=1e-3)
    assert position[1] == pytest.approx(0.0, abs=1e-6)
    assert position[2] == pytest.approx(0.926, abs=1e-3)


def test_maximum_reach_is_the_published_855_millimetres() -> None:
    """A reach model that quietly over-reaches would admit impossible placements."""

    stretched = [0.0, 0.0, 0.0, -0.5, 0.0, 2.7, 0.0]
    position, _ = forward_kinematics(stretched)

    assert math.dist(position, [0.0, 0.0, 0.333]) == pytest.approx(0.855, abs=0.01)


def test_ik_returns_joints_that_reach_the_asked_for_point() -> None:
    target = [0.45, 0.10, 0.60]

    result = solve_position_ik(target_position_world_m=target, seed_joint_positions=None)

    assert result["solved"] is True
    reached, _ = forward_kinematics(result["joint_positions"])
    assert math.dist(reached, target) < 1e-3


def test_ik_never_returns_joints_outside_their_limits() -> None:
    result = solve_position_ik(
        target_position_world_m=[0.55, -0.30, 0.25], seed_joint_positions=None
    )

    for value, (low, high) in zip(result["joint_positions"], FRANKA_JOINT_LIMITS_RAD):
        assert low - 1e-9 <= value <= high + 1e-9


def test_a_point_beyond_reach_is_reported_not_approximated() -> None:
    """Returning the closest reachable pose would look like a solved grasp."""

    result = solve_position_ik(
        target_position_world_m=[1.60, 0.0, 0.40], seed_joint_positions=None
    )

    assert result["solved"] is False
    assert result["position_error_m"] > 0.1


def test_manipulability_peaks_mid_workspace_and_falls_off_both_ends() -> None:
    """Dexterity is not monotonic in reach, and assuming it is misleads twice.

    Folded against the base is as badly conditioned as stretched to the limit;
    only the middle of the workspace is comfortable. A placement search that
    treated "closer is safer" as a rule would happily park the arm in the
    cramped inner region and call it conservative.
    """

    def _at(radius: float) -> float:
        result = solve_position_ik(
            target_position_world_m=[radius, 0.0, 0.45], seed_joint_positions=None
        )
        assert result["solved"], radius
        return manipulability(result["joint_positions"])

    folded, middle, extended = _at(0.35), _at(0.65), _at(0.82)
    assert folded < middle
    assert extended < middle


def test_a_straightening_arm_braces_along_itself_and_weakens_across() -> None:
    """Which direction the load pulls in decides whether extension helps or hurts.

    A nearly straight arm holds enormous force along its own axis - it is a
    strut - while losing capability across it. A door handle sweeps an arc, so
    the load is the across direction, which is the one that gets worse. Quoting
    a single "max force" number for a pose would report the strut and hide the
    limit that actually binds.
    """

    def _at(radius: float):
        result = solve_position_ik(
            target_position_world_m=[radius, 0.0, 0.45], seed_joint_positions=None
        )
        assert result["solved"], radius
        return result["joint_positions"]

    middle, extended = _at(0.65), _at(0.82)
    along = [1.0, 0.0, 0.0]
    across = [0.0, 1.0, 0.0]

    assert radial_force_capability_n(extended, direction_world=along) > (
        radial_force_capability_n(middle, direction_world=along)
    )
    assert radial_force_capability_n(extended, direction_world=across) < (
        radial_force_capability_n(middle, direction_world=across)
    )
    assert radial_force_capability_n(extended, direction_world=across) > 0.0


def test_a_wrong_length_joint_vector_fails_closed() -> None:
    with pytest.raises(FrankaKinematicsError) as excinfo:
        forward_kinematics([0.0] * 6)

    assert any("joint_vector_invalid" in e for e in excinfo.value.errors)


def test_kinematics_are_deterministic() -> None:
    a = solve_position_ik(
        target_position_world_m=[0.45, 0.10, 0.60], seed_joint_positions=None
    )
    b = solve_position_ik(
        target_position_world_m=[0.45, 0.10, 0.60], seed_joint_positions=None
    )
    assert a == b
