"""Pin the handling of an unauthored (identity) grasp orientation.

r22's sealed affordance carried
``gripper_orientation_contact_xyzw = [0.0, 0.0, 0.0, 1.0]`` -- identity, which is
what a quaternion field holds when nobody authored it. The plan composed that
with the door pose, so every phase commanded the hand to world identity. The
Franka hand's natural pose is (0.5, 0.5, 0.5, 0.5), exactly 120 degrees away, so
every phase carried a 2.09 rad orientation error against an 0.08 rad arrival
tolerance -- unsatisfiable by construction. Near that error the differential IK
alternated between solution branches every step, the arm thrashed into its joint
limits at saturated torque (exactly +/-87.0 and +/-12.0 N*m), and the
end-effector position error never fell below its first-step value.

Two responses, applied where each belongs:
  * the contact replay closes the gripper on the handle, so it refuses;
  * construction runs open-gripper clearance probes, so it binds the measured
    reset orientation, which is what the design already documented.
"""

from __future__ import annotations

import math

import pytest

from blueprint_pipeline import native_task_arena_construction_worker as worker
from blueprint_pipeline.native_franka_action_math import (
    is_unauthored_identity_quaternion_xyzw,
)

IDENTITY = [0.0, 0.0, 0.0, 1.0]
# The pose the Franka hand actually rests in, and the value r22 measured.
FRANKA_HAND_RESET = [0.5, 0.5, 0.5, 0.5]


def _angle(a: list[float], b: list[float]) -> float:
    return 2.0 * math.acos(min(1.0, abs(sum(x * y for x, y in zip(a, b)))))


def test_identity_is_120_degrees_from_the_arm_natural_hand_pose() -> None:
    """Why identity is never a harmless default on this robot."""

    assert _angle(IDENTITY, FRANKA_HAND_RESET) == pytest.approx(
        2.0943951, abs=1e-6
    )
    assert math.degrees(_angle(IDENTITY, FRANKA_HAND_RESET)) == pytest.approx(
        120.0, abs=1e-4
    )


@pytest.mark.parametrize(
    "value",
    [
        IDENTITY,
        [0.0, 0.0, 0.0, -1.0],
        [0.0, -1.4086566911091288e-24, 0.0, 1.0],
        None,
    ],
)
def test_identity_and_its_float_noise_read_as_unauthored(value) -> None:
    """r22's plan carried identity with 1e-24 noise; that is still identity."""

    assert is_unauthored_identity_quaternion_xyzw(value) is True


@pytest.mark.parametrize(
    "value",
    [
        FRANKA_HAND_RESET,
        [0.0, 0.0, 0.10294231213501348, 0.9946873279439612],
        [0.7071067811865476, 0.0, 0.0, 0.7071067811865476],
    ],
)
def test_a_real_orientation_is_not_read_as_unauthored(value) -> None:
    assert is_unauthored_identity_quaternion_xyzw(value) is False


def test_clearance_phase_binds_the_measured_reset_orientation() -> None:
    """The documented fallback, restored.

    ``phase.get(key, default)`` only falls back when the key is absent, so an
    identity value -- which means the same thing -- was executed as a real
    120 degree rotation target instead.
    """

    bound = worker._phase_target_orientation(
        {"orientation_world_xyzw": IDENTITY},
        reset_body_orientation_xyzw=FRANKA_HAND_RESET,
    )

    assert bound == pytest.approx(FRANKA_HAND_RESET)


def test_absent_orientation_still_binds_the_measured_reset() -> None:
    assert worker._phase_target_orientation(
        {}, reset_body_orientation_xyzw=FRANKA_HAND_RESET
    ) == pytest.approx(FRANKA_HAND_RESET)


def test_an_authored_phase_orientation_is_never_overridden() -> None:
    """The sweep phases carry a real door yaw; that must survive untouched."""

    authored = [0.0, 0.0, -0.10294231213501348, 0.9946873279439612]

    assert worker._phase_target_orientation(
        {"orientation_world_xyzw": authored},
        reset_body_orientation_xyzw=FRANKA_HAND_RESET,
    ) == pytest.approx(authored)


def test_bound_orientation_never_leaves_a_120_degree_error_at_reset() -> None:
    """The regression itself: at reset the commanded orientation error must be
    zero, not the 2.09 rad that made every arrival check unsatisfiable."""

    bound = worker._phase_target_orientation(
        {"orientation_world_xyzw": IDENTITY},
        reset_body_orientation_xyzw=FRANKA_HAND_RESET,
    )

    assert _angle(bound, FRANKA_HAND_RESET) == pytest.approx(0.0, abs=1e-9)


def test_reference_alignment_rotation_is_the_120_degrees_we_are_missing() -> None:
    """Isaac Lab's Franka handle-grasp reference authors this slot explicitly.

    ``cabinet_env_cfg.py`` gives the drawer handle frame
    ``rot=(0.5, -0.5, -0.5, 0.5)`` with the comment "align with end-effector
    frame", and the state machine then grasps with an identity offset on top of
    that already-aligned frame.  The reference rotation is 120 degrees -- the
    exact error r22 measured from authoring identity instead.
    """

    reference_alignment = [0.5, -0.5, -0.5, 0.5]

    assert is_unauthored_identity_quaternion_xyzw(reference_alignment) is False
    assert math.degrees(_angle(reference_alignment, IDENTITY)) == pytest.approx(
        120.0, abs=1e-4
    )
