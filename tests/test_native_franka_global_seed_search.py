from __future__ import annotations

import math


from blueprint_pipeline.native_franka_global_seed_search import (
    DEFAULT_DIVERSE_SEED_COUNT,
    GLOBAL_SEED_SEARCH_SCHEMA_VERSION,
    diverse_joint_seeds,
    high_margin_joint_seeds,
)


def test_diverse_seed_design_preserves_inputs_and_covers_over_one_hundred() -> None:
    seeds = diverse_joint_seeds(
        seeds=[[0.0] * 7, [0.1] * 7],
        lower_joint_position_limits_rad=[-2.0] * 7,
        upper_joint_position_limits_rad=[2.0] * 7,
    )

    assert len(seeds) == DEFAULT_DIVERSE_SEED_COUNT == 128
    assert seeds[:2] == [[0.0] * 7, [0.1] * 7]
    assert len({tuple(row) for row in seeds}) == len(seeds)
    assert all(-1.8 <= value <= 1.8 for row in seeds[2:] for value in row)


def _planar_arm(link_lengths):
    """A planar chain whose pose and Jacobian are exact and checkable."""

    def frame_pose(joints):
        x = y = 0.0
        angle = 0.0
        for length, joint in zip(link_lengths, joints, strict=True):
            angle += float(joint)
            x += length * math.cos(angle)
            y += length * math.sin(angle)
        half = angle / 2.0
        return [x, y, 0.0], [0.0, 0.0, math.sin(half), math.cos(half)]

    def frame_jacobian(joints):
        angles = []
        angle = 0.0
        for joint in joints:
            angle += float(joint)
            angles.append(angle)
        columns = []
        for index in range(len(link_lengths)):
            dx = -sum(
                link_lengths[k] * math.sin(angles[k]) for k in range(index, len(link_lengths))
            )
            dy = sum(
                link_lengths[k] * math.cos(angles[k]) for k in range(index, len(link_lengths))
            )
            columns.append([dx, dy, 0.0, 0.0, 0.0, 1.0])
        return [[columns[c][r] for c in range(len(link_lengths))] for r in range(6)]

    return frame_pose, frame_jacobian


def test_the_search_finds_a_configuration_far_from_any_joint_stop() -> None:
    """The live solver refines locally; this is what it cannot reach.

    C45 measured every branch the live solver found sitting 0.014 to 0.024 rad
    from a joint stop, while an unregularised search over the same seeds and
    the same limits reached 0.45 rad at the same pose -- a configuration
    2.66 rad away in joint space, which a tracker with a posture cost is
    precisely designed not to travel to.
    """

    lengths = [1.0, 1.0, 0.5]
    frame_pose, frame_jacobian = _planar_arm(lengths)
    lower, upper = [-2.9] * 3, [2.9] * 3
    target, quaternion = frame_pose([0.6, -0.4, 0.3])

    report = high_margin_joint_seeds(
        frame_pose=frame_pose,
        frame_jacobian=frame_jacobian,
        # One seed hard against a stop, one comfortably interior.
        seeds=[[-2.88, -2.88, 2.88], [0.2, 0.2, 0.2], [1.4, -1.0, 0.5]],
        target_position_m=target,
        target_quaternion_xyzw=quaternion,
        lower_joint_position_limits_rad=lower,
        upper_joint_position_limits_rad=upper,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.05,
    )

    assert report["schema_version"] == GLOBAL_SEED_SEARCH_SCHEMA_VERSION
    assert report["status"] == "searched"
    assert report["configurations_found"] >= 1
    # Every returned configuration actually reaches the pose...
    for seed in report["seeds"]:
        reached, _ = frame_pose(seed)
        assert math.dist(reached, target) <= 0.005
        # ...and lies inside the limits, by the margin claimed.
        assert all(lower[i] <= seed[i] <= upper[i] for i in range(3))
    # Seeds are ordered by margin, best first, and the best is genuinely roomy.
    assert report["margins_rad"] == sorted(report["margins_rad"], reverse=True)
    assert report["best_margin_rad"] > 0.1
    # It reports what it is, not a verdict.
    assert "the_live_solver_refines" in report["claim_boundary"]
    assert "task_succeeded" not in report


def test_an_unreachable_pose_is_reported_not_invented() -> None:
    lengths = [1.0, 1.0]
    frame_pose, frame_jacobian = _planar_arm(lengths)

    report = high_margin_joint_seeds(
        frame_pose=frame_pose,
        frame_jacobian=frame_jacobian,
        seeds=[[0.1, 0.1], [1.0, -1.0]],
        target_position_m=[9.0, 9.0, 0.0],  # far outside a 2 m reach
        target_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        lower_joint_position_limits_rad=[-2.9, -2.9],
        upper_joint_position_limits_rad=[2.9, 2.9],
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.05,
    )

    assert report["status"] == "no_configuration_converged"
    assert report["seeds"] == []
    assert report["best_margin_rad"] is None


def test_a_seed_that_cannot_be_evaluated_does_not_lose_the_others() -> None:
    lengths = [1.0, 1.0, 0.5]
    frame_pose, frame_jacobian = _planar_arm(lengths)
    target, quaternion = frame_pose([0.5, -0.3, 0.2])

    def _explodes_on_one_seed(joints):
        if abs(joints[0] - 2.5) < 1e-9:
            raise RuntimeError("this configuration cannot be evaluated")
        return frame_pose(joints)

    report = high_margin_joint_seeds(
        frame_pose=_explodes_on_one_seed,
        frame_jacobian=frame_jacobian,
        seeds=[[2.5, 0.0, 0.0], [0.3, 0.3, 0.3]],
        target_position_m=target,
        target_quaternion_xyzw=quaternion,
        lower_joint_position_limits_rad=[-2.9] * 3,
        upper_joint_position_limits_rad=[2.9] * 3,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.05,
    )

    assert report["seeds_evaluated"] == 2
    assert report["status"] == "searched"


def test_invalid_limits_are_refused() -> None:
    frame_pose, frame_jacobian = _planar_arm([1.0, 1.0])
    report = high_margin_joint_seeds(
        frame_pose=frame_pose,
        frame_jacobian=frame_jacobian,
        seeds=[[0.0, 0.0]],
        target_position_m=[1.0, 0.0, 0.0],
        target_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        lower_joint_position_limits_rad=[1.0, 1.0],
        upper_joint_position_limits_rad=[-1.0, -1.0],
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.05,
    )
    assert report["status"] == "unavailable"
    assert report["reason"] == "joint_limits_invalid"


def test_returned_seeds_are_distinct_basins_not_near_duplicates() -> None:
    """Highest-margin is not the same as diverse.

    Descents from several seeds can land in one basin, and three
    near-duplicates of the same configuration cover no more of the solution
    space than one does -- they just spend the slots a genuinely different
    branch could have used.
    """

    import math

    lengths = [1.0, 1.0, 0.5]
    frame_pose, frame_jacobian = _planar_arm(lengths)
    target, quaternion = frame_pose([0.5, -0.35, 0.25])

    report = high_margin_joint_seeds(
        frame_pose=frame_pose,
        frame_jacobian=frame_jacobian,
        # Six seeds, several clustered so they converge together.
        seeds=[
            [0.50, -0.35, 0.25], [0.51, -0.34, 0.26], [0.49, -0.36, 0.24],
            [-1.2, 1.4, -0.6], [1.6, -1.5, 0.9], [0.0, 0.0, 0.0],
        ],
        target_position_m=target,
        target_quaternion_xyzw=quaternion,
        lower_joint_position_limits_rad=[-2.9] * 3,
        upper_joint_position_limits_rad=[2.9] * 3,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.05,
    )

    seeds = report["seeds"]
    assert len(seeds) >= 1
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            distance = math.dist(seeds[i], seeds[j])
            assert distance >= 0.35, f"seeds {i} and {j} are the same basin"
