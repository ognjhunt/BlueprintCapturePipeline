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


def test_axis_aligned_ik_points_the_tool_at_the_target():
    """Position-only IK leaves the wrist wherever the solver falls.

    rt42: the flange reached its waypoint 89 mm from the handle - correct -
    with the tool axis pointing +x while the handle lay -y. The fingers closed
    on air 166 mm away, 0 contact in 91 samples, and the door never moved. The
    grasp needs 5 constraints: where the flange is, and which way the tool
    points. Roll stays free.
    """

    from blueprint_pipeline.franka_kinematics import (
        forward_kinematics,
        solve_axis_aligned_ik,
    )

    target = [0.45, -0.20, 0.55]
    approach = [0.0, -1.0, 0.0]  # tool must point along -y

    result = solve_axis_aligned_ik(
        target_position_world_m=target,
        tool_axis_world=approach,
    )

    assert result["converged"] is True
    q = result["joint_positions_rad"]
    position, rotation = forward_kinematics(q)
    for axis in range(3):
        assert abs(position[axis] - target[axis]) < 2e-3
    tool_z = [rotation[0][2], rotation[1][2], rotation[2][2]]
    dot = sum(tool_z[i] * approach[i] for i in range(3))
    assert dot > 0.996  # within ~5 degrees


def test_axis_aligned_ik_reports_both_errors_when_unreachable():
    from blueprint_pipeline.franka_kinematics import solve_axis_aligned_ik

    result = solve_axis_aligned_ik(
        target_position_world_m=[2.5, 0.0, 0.3],  # far outside reach
        tool_axis_world=[0.0, -1.0, 0.0],
    )

    assert result["converged"] is False
    assert result["position_error_m"] > 0.01


def test_axis_aligned_ik_respects_joint_limits():
    from blueprint_pipeline.franka_kinematics import (
        FRANKA_JOINT_LIMITS_RAD,
        solve_axis_aligned_ik,
    )

    result = solve_axis_aligned_ik(
        target_position_world_m=[0.4, 0.1, 0.5],
        tool_axis_world=[0.0, 0.0, -1.0],
    )

    for value, (lo, hi) in zip(result["joint_positions_rad"], FRANKA_JOINT_LIMITS_RAD):
        assert lo - 1e-9 <= value <= hi + 1e-9


class TestOrientedIk:
    """Position plus tool axis plus finger axis: the full grasp frame.

    rt54's pads arrived at the door with the roll wherever the solver fell,
    and a horizontal bar cannot be pinched by fingers straddling it
    sideways: the lower finger jammed against the door 132 mm below the
    bar. A bar grasp is eight constraints; leaving roll free is a coin
    flip on every waypoint.
    """

    HANDLE_BASE_FRAME = (0.3617, -0.1531, 1.023)  # world handle minus robot base
    DOOR_NORMAL = (0.0, 1.0, 0.0)

    def test_solves_position_and_both_axes(self):
        from blueprint_pipeline.franka_kinematics import (
            forward_kinematics,
            solve_oriented_ik,
        )

        result = solve_oriented_ik(
            target_position_world_m=self.HANDLE_BASE_FRAME,
            tool_axis_world=[-v for v in self.DOOR_NORMAL],
            finger_axis_world=(0.0, 0.0, 1.0),
        )

        assert result["converged"] is True
        assert result["position_error_m"] < 0.001
        _, rotation = forward_kinematics(result["joint_positions_rad"])
        tool_z = [rotation[row][2] for row in range(3)]
        tool_y = [rotation[row][1] for row in range(3)]
        assert sum(a * b for a, b in zip(tool_z, [0, -1, 0])) >= 0.996
        assert abs(sum(a * b for a, b in zip(tool_y, [0, 0, 1]))) >= 0.996

    def test_both_roll_signs_are_reachable(self):
        from blueprint_pipeline.franka_kinematics import solve_oriented_ik

        for sign in (1.0, -1.0):
            result = solve_oriented_ik(
                target_position_world_m=self.HANDLE_BASE_FRAME,
                tool_axis_world=[-v for v in self.DOOR_NORMAL],
                finger_axis_world=(0.0, 0.0, sign),
            )
            assert result["converged"] is True, sign

    def test_a_finger_axis_parallel_to_the_tool_axis_is_refused(self):
        from blueprint_pipeline.franka_kinematics import (
            FrankaKinematicsError,
            solve_oriented_ik,
        )

        with pytest.raises(FrankaKinematicsError):
            solve_oriented_ik(
                target_position_world_m=self.HANDLE_BASE_FRAME,
                tool_axis_world=(0.0, -1.0, 0.0),
                finger_axis_world=(0.0, -1.0, 0.0),
            )

    def test_an_unreachable_target_reports_rather_than_pretending(self):
        from blueprint_pipeline.franka_kinematics import solve_oriented_ik

        result = solve_oriented_ik(
            target_position_world_m=(2.5, 0.0, 0.2),
            tool_axis_world=(0.0, -1.0, 0.0),
            finger_axis_world=(0.0, 0.0, 1.0),
        )

        assert result["converged"] is False
