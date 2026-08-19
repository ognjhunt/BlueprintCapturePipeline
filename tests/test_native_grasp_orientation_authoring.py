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
    NativeFrankaActionMathError,
    grasp_orientation_contact_xyzw,
    is_unauthored_identity_quaternion_xyzw,
)
from blueprint_pipeline.paired_target_native_arena_request import (
    PairedTargetNativeArenaRequestError,
    _grasp_orientation_contact_xyzw,
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


# --------------------------------------------------------------------------
# Deriving the value, rather than detecting that nobody authored one.
# --------------------------------------------------------------------------

# The rotation Isaac Lab authors for its Franka handle grasp, and the axis pair
# it means.  cabinet_env_cfg.py writes the quaternion; franka_cabinet_env.py
# names the same alignment as gripper_forward_axis(+Z) <-> drawer_inward_axis
# (-X) and gripper_up_axis(+Y) <-> drawer_up_axis(+Z).
UPSTREAM_REFERENCE_ALIGNMENT = [0.5, -0.5, -0.5, 0.5]
UPSTREAM_APPROACH_AXIS = (-1.0, 0.0, 0.0)
UPSTREAM_JAW_AXIS = (0.0, 0.0, 1.0)

# The r23 packet's sealed axes, verbatim.
SEALED_APPROACH_UNIT = [-0.0, -1.0, -0.0]
SEALED_PINCH_AXIS = [-0.0, -1.0, -0.0]


def test_convention_regenerates_the_upstream_authored_quaternion_exactly() -> None:
    """The convention, pinned to the only value upstream actually authored.

    If our reading of "align with end-effector frame" were wrong by any axis
    swap or sign, feeding the reference's own axes back in could not reproduce
    the reference's own literal.  It does, to the bit.
    """

    derived = grasp_orientation_contact_xyzw(
        approach_axis=UPSTREAM_APPROACH_AXIS, jaw_axis=UPSTREAM_JAW_AXIS
    )

    assert derived == pytest.approx(UPSTREAM_REFERENCE_ALIGNMENT, abs=1e-12)
    assert is_unauthored_identity_quaternion_xyzw(derived) is False


def test_the_derived_reference_is_a_frame_not_a_rotation_amount() -> None:
    """Its columns are the gripper axes in contact coordinates."""

    derived = grasp_orientation_contact_xyzw(
        approach_axis=UPSTREAM_APPROACH_AXIS, jaw_axis=UPSTREAM_JAW_AXIS
    )
    x, y, z, w = derived
    columns = [
        [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
        [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
        [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)],
    ]
    # ee_z is the approach axis, ee_y the jaw axis -- exactly what went in.
    assert columns[2] == pytest.approx(list(UPSTREAM_APPROACH_AXIS), abs=1e-12)
    assert columns[1] == pytest.approx(list(UPSTREAM_JAW_AXIS), abs=1e-12)


def test_jaw_axis_only_fixes_roll_and_need_not_be_exactly_perpendicular() -> None:
    """Measured geometry is never exactly orthogonal; the approach wins."""

    derived = grasp_orientation_contact_xyzw(
        approach_axis=UPSTREAM_APPROACH_AXIS, jaw_axis=(0.02, 0.0, 1.0)
    )

    assert derived == pytest.approx(UPSTREAM_REFERENCE_ALIGNMENT, abs=2e-2)
    assert is_unauthored_identity_quaternion_xyzw(derived) is False


def test_sealed_washer_door_axes_cannot_define_a_grasp_frame() -> None:
    """The reason this task's orientation is still unauthored.

    ``target_driven_link_far_edge_pinch`` assigns the panel normal to BOTH
    ``approach_unit_registered_stage`` and ``pinch_axis_registered_stage``, so
    the r23 affordance carries one axis twice.  A parallel-jaw frame needs two
    independent axes: with these, ee_x = ee_y x ee_z is the zero vector and the
    roll about the approach is unconstrained.  No quaternion is derivable, so
    nothing may be guessed here.
    """

    assert SEALED_APPROACH_UNIT == SEALED_PINCH_AXIS

    with pytest.raises(NativeFrankaActionMathError) as excinfo:
        grasp_orientation_contact_xyzw(
            approach_axis=SEALED_APPROACH_UNIT, jaw_axis=SEALED_PINCH_AXIS
        )

    assert "native_franka_grasp_orientation_axes_degenerate" in excinfo.value.errors


@pytest.mark.parametrize(
    "approach,jaw",
    [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((float("nan"), 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((-1.0, 0.0), (0.0, 0.0, 1.0)),
    ],
)
def test_unusable_axes_fail_closed(approach, jaw) -> None:
    with pytest.raises(NativeFrankaActionMathError):
        grasp_orientation_contact_xyzw(approach_axis=approach, jaw_axis=jaw)


def test_author_refuses_the_sealed_affordance_instead_of_emitting_identity() -> None:
    """The authoring site, which used to hard-code the identity placeholder."""

    candidate = {
        "approach_unit_registered_stage": SEALED_APPROACH_UNIT,
        "pinch_axis_registered_stage": SEALED_PINCH_AXIS,
    }
    path_receipt = {
        "joint_contact_path": [
            {"contact_pose_asset_root": {"orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}}
        ]
    }

    with pytest.raises(PairedTargetNativeArenaRequestError) as excinfo:
        _grasp_orientation_contact_xyzw(candidate, path_receipt)

    assert "grasp_orientation_unauthorable" in str(excinfo.value)
    assert "degenerate" in str(excinfo.value)


def test_author_derives_in_the_contact_frame_not_the_stage_frame() -> None:
    """The offset is composed as ``contact_world * offset``, so it is relative
    to the contact frame.  A contact frame yawed 90 degrees about +Z must shift
    the authored quaternion by exactly that much, or the hand would be commanded
    into the stage's axes instead of the door's."""

    candidate = {
        "approach_unit_registered_stage": [-1.0, 0.0, 0.0],
        "pinch_axis_registered_stage": [0.0, 0.0, 1.0],
    }
    aligned = _grasp_orientation_contact_xyzw(
        candidate,
        {
            "joint_contact_path": [
                {"contact_pose_asset_root": {"orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}}
            ]
        },
    )
    assert aligned == pytest.approx(UPSTREAM_REFERENCE_ALIGNMENT, abs=1e-12)

    half = math.sqrt(0.5)
    yawed = _grasp_orientation_contact_xyzw(
        candidate,
        {
            "joint_contact_path": [
                {"contact_pose_asset_root": {"orientation_xyzw": [0.0, 0.0, half, half]}}
            ]
        },
    )

    assert yawed != pytest.approx(aligned, abs=1e-9)
    assert math.degrees(_angle(yawed, aligned)) == pytest.approx(90.0, abs=1e-6)
