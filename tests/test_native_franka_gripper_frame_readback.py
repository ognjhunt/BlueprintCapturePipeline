"""Settle, by measurement, which frame the controlled gripper body is in.

Three sealed artifacts pairwise contradict each other about this, and any two
of them can be true while all three cannot:

  1. The DROID asset has no ``panda_hand`` -- ``native_task_robot_contact_topology``
     lists ``panda_link0..7`` plus Robotiq bodies -- so the controlled body is
     ``base_link``.  But the ``+Z_ee`` approach / ``+Y_ee`` jaw convention that
     ``grasp_orientation_contact_xyzw`` derives, and that PR #802 authored
     against, is ``panda_hand``'s.
  2. At the sealed reset joints this repository's own forward kinematics puts
     the flange ``+Z`` straight down.  The measured reset body quaternion
     ``(0.5, 0.5, 0.5, 0.5)`` has no axis pointing down except ``-Y``.
  3. The sealed wrist-camera extrinsic, expressed in ``base_link``, has its
     optical axis at ``(0.9498, -0.3130, 0.0022)`` ~ ``+X_body``.

None of the three can settle it, because they are the things that disagree.
The reset pose can, and needs no convention: the direction between the two
finger bodies is the jaw axis, and the direction from the controlled body's
origin to their midpoint is the direction the tool extends.  Both buffers are
already read every control tick and the direction discarded.

These tests pin that the readback measures rather than asserts, that its
derived axes are recomputable from the retained raw numbers, that each of the
three hypotheses is separable from the other two, and that absence refuses by
name instead of defaulting to an axis nobody measured.
"""

from __future__ import annotations

import math

import pytest

from blueprint_pipeline.native_franka_pose_servo import (
    GRIPPER_FRAME_APPROACH_HYPOTHESES,
    GRIPPER_FRAME_HYPOTHESIS_TOLERANCE_RAD,
    GRIPPER_FRAME_READBACK_SCHEMA_VERSION,
    JAW_AXIS_ORDERING,
    NativeFrankaPoseServoError,
    gripper_frame_axis_readback,
)


IDENTITY_XYZW = [0.0, 0.0, 0.0, 1.0]
# The controlled body's measured reset orientation, as retained by the r22
# readback and restated by tests/test_native_grasp_approach_axis_authoring.py.
MEASURED_RESET_BODY_QUAT_XYZW = [0.5, 0.5, 0.5, 0.5]


def _rotate(quaternion_xyzw, vector):
    x, y, z, w = quaternion_xyzw
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _readback(
    *,
    body_quaternion_xyzw=IDENTITY_XYZW,
    approach_axis_body=(0.0, 0.0, 1.0),
    jaw_axis_body=(0.0, 1.0, 0.0),
    body_position=(0.0, 0.0, 0.0),
    tool_offset_m=0.16,
    finger_separation_m=0.08,
    **overrides,
):
    """Place two fingers so the body frame has a chosen approach and jaw axis.

    The fingers are authored in *body* coordinates and then rotated into world
    by the body quaternion, which is the inverse of what the readback does, so
    a correct readback returns the body axes it was given.
    """

    midpoint_world = [
        body_position[index]
        + _rotate(
            body_quaternion_xyzw,
            [value * tool_offset_m for value in approach_axis_body],
        )[index]
        for index in range(3)
    ]
    half = [
        value * finger_separation_m / 2.0
        for value in _rotate(body_quaternion_xyzw, list(jaw_axis_body))
    ]
    origin_name, tip_name = JAW_AXIS_ORDERING
    fingers = {
        origin_name: [midpoint_world[i] - half[i] for i in range(3)],
        tip_name: [midpoint_world[i] + half[i] for i in range(3)],
    }
    fingers.update(overrides.pop("finger_positions_world_m", {}))
    return gripper_frame_axis_readback(
        controlled_body_name=overrides.pop("controlled_body_name", "base_link"),
        body_position_world_m=body_position,
        body_quaternion_world_xyzw=body_quaternion_xyzw,
        finger_positions_world_m=fingers,
        **overrides,
    )


def test_readback_recovers_body_axes_through_a_nonidentity_body_rotation() -> None:
    """The measurement must survive the body being rotated in the world.

    A readback that accidentally reported world axes would pass under an
    identity body quaternion and be wrong in every real run, so the body pose
    used here is the one actually measured at reset.
    """

    result = _readback(
        body_quaternion_xyzw=MEASURED_RESET_BODY_QUAT_XYZW,
        approach_axis_body=(0.0, -1.0, 0.0),
        jaw_axis_body=(0.0, 0.0, 1.0),
    )

    assert result["derived"]["approach_unit_body"] == pytest.approx(
        [0.0, -1.0, 0.0], abs=1e-9
    )
    assert result["derived"]["jaw_unit_body"] == pytest.approx(
        [0.0, 0.0, 1.0], abs=1e-9
    )
    # The world-frame axes differ from the body-frame ones, so the rotation was
    # genuinely applied rather than the two frames coinciding.
    assert result["derived"]["approach_unit_world"] != pytest.approx(
        result["derived"]["approach_unit_body"], abs=1e-3
    )


def test_every_derived_axis_is_recomputable_from_the_retained_raw_numbers() -> None:
    """The receipt must carry the measurement, not only a conclusion from it.

    Today's lesson, three times over, was 'a sealed artifact asserted something
    nobody measured'.  So the raw body pose and both finger world positions are
    retained, and this recomputes the derived axes from them alone.
    """

    result = _readback(
        body_quaternion_xyzw=MEASURED_RESET_BODY_QUAT_XYZW,
        approach_axis_body=(1.0, 0.0, 0.0),
        jaw_axis_body=(0.0, 1.0, 0.0),
        body_position=(0.3, -0.2, 0.9),
    )
    measured = result["measured"]
    derived = result["derived"]

    origin_name, tip_name = derived["jaw_axis_ordering"]
    fingers = measured["finger_body_positions_world_m"]
    jaw_world = [
        fingers[tip_name][index] - fingers[origin_name][index] for index in range(3)
    ]
    jaw_norm = math.sqrt(sum(value * value for value in jaw_world))
    assert jaw_norm == pytest.approx(measured["finger_separation_m"], abs=1e-12)
    assert derived["jaw_unit_world"] == pytest.approx(
        [value / jaw_norm for value in jaw_world], abs=1e-12
    )

    midpoint = [
        (fingers[origin_name][index] + fingers[tip_name][index]) / 2.0
        for index in range(3)
    ]
    assert measured["finger_midpoint_world_m"] == pytest.approx(midpoint, abs=1e-12)
    approach_world = [
        midpoint[index] - measured["controlled_body_position_world_m"][index]
        for index in range(3)
    ]
    approach_norm = math.sqrt(sum(value * value for value in approach_world))
    assert approach_norm == pytest.approx(
        measured["body_origin_to_finger_midpoint_m"], abs=1e-12
    )

    # World -> body is the inverse of the retained body quaternion.
    quaternion = measured["controlled_body_quaternion_world_xyzw"]
    inverse = [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]]
    assert derived["approach_unit_body"] == pytest.approx(
        _rotate(inverse, [value / approach_norm for value in approach_world]),
        abs=1e-12,
    )
    assert derived["jaw_unit_body"] == pytest.approx(
        _rotate(inverse, derived["jaw_unit_world"]), abs=1e-12
    )


@pytest.mark.parametrize(
    ("approach_axis_body", "hypothesis_id", "nearest_body_axis"),
    [
        ((0.0, 0.0, 1.0), "tool_frame_convention_holds_for_controlled_body", "+z"),
        ((0.0, -1.0, 0.0), "reset_body_quaternion_implies_a_coupler_rotation", "-y"),
        ((1.0, 0.0, 0.0), "wrist_camera_optical_axis_is_the_approach_axis", "+x"),
    ],
)
def test_each_hypothesis_is_separable_from_the_other_two(
    approach_axis_body, hypothesis_id, nearest_body_axis
) -> None:
    """The three predicted approach axes are mutually orthogonal, so exactly
    one can be within tolerance of any measured axis.  A run that lands on one
    of them therefore names it, and cannot name a second."""

    jaw = (0.0, 1.0, 0.0) if approach_axis_body != (0.0, 1.0, 0.0) else (0.0, 0.0, 1.0)
    if approach_axis_body == (0.0, -1.0, 0.0):
        jaw = (0.0, 0.0, 1.0)
    result = _readback(
        body_quaternion_xyzw=MEASURED_RESET_BODY_QUAT_XYZW,
        approach_axis_body=approach_axis_body,
        jaw_axis_body=jaw,
    )
    assessment = result["assessment"]

    assert assessment["resolution"] == "supported"
    assert assessment["supported_hypothesis_id"] == hypothesis_id
    assert assessment["supported_hypothesis_ids"] == [hypothesis_id]
    assert assessment["approach"]["nearest_body_axis"] == nearest_body_axis
    assert assessment["approach"]["nearest_body_axis_angle_rad"] == pytest.approx(
        0.0, abs=1e-6
    )
    matched = [
        row
        for row in assessment["approach_hypotheses"]
        if row["hypothesis_id"] == hypothesis_id
    ]
    assert len(matched) == 1
    assert matched[0]["approach_angle_rad"] == pytest.approx(0.0, abs=1e-6)
    others = [
        row
        for row in assessment["approach_hypotheses"]
        if row["hypothesis_id"] != hypothesis_id
    ]
    assert len(others) == 2
    assert all(row["within_tolerance"] is False for row in others)
    assert all(
        row["approach_angle_rad"] > GRIPPER_FRAME_HYPOTHESIS_TOLERANCE_RAD
        for row in others
    )


def test_an_axis_matching_no_hypothesis_refuses_to_name_one() -> None:
    """A measured axis between the candidates is a real outcome and must not be
    rounded to the nearest hypothesis.  It names none of them and still retains
    every number needed to see why."""

    diagonal = 1.0 / math.sqrt(3.0)
    result = _readback(
        approach_axis_body=(diagonal, diagonal, diagonal),
        jaw_axis_body=(1.0, -1.0, 0.0),
    )
    assessment = result["assessment"]

    assert assessment["resolution"] == "none_within_tolerance"
    assert assessment["supported_hypothesis_id"] is None
    assert assessment["supported_hypothesis_ids"] == []
    assert result["derived"]["approach_unit_body"] == pytest.approx(
        [diagonal] * 3, abs=1e-9
    )


def test_the_hypotheses_are_recorded_with_the_artifact_that_asserts_each() -> None:
    """The verdict is only readable if the receipt says what each candidate was
    and who claimed it, so a completed run needs no outside context."""

    result = _readback()
    recorded = {
        row["hypothesis_id"]: row for row in result["assessment"]["approach_hypotheses"]
    }

    assert set(recorded) == {
        candidate["hypothesis_id"] for candidate in GRIPPER_FRAME_APPROACH_HYPOTHESES
    }
    assert len(recorded) == 3
    for row in recorded.values():
        assert row["asserted_by"]
        assert len(row["predicted_approach_axis_body"]) == 3
    # The three predictions must stay mutually orthogonal; if a future edit made
    # two of them agree, a run could no longer tell those two apart.
    axes = [row["predicted_approach_axis_body"] for row in recorded.values()]
    for first in range(3):
        for second in range(first + 1, 3):
            assert sum(
                a * b for a, b in zip(axes[first], axes[second], strict=True)
            ) == pytest.approx(0.0, abs=1e-12)


def test_the_jaw_sign_is_recorded_as_a_label_rather_than_claimed_as_measured() -> None:
    """The two fingers are interchangeable, so the jaw is a line and its arrow
    is this module's naming choice.  Saying so in the receipt is what keeps the
    sign from being read later as something the asset stated."""

    result = _readback()

    assert result["derived"]["jaw_axis_ordering"] == list(JAW_AXIS_ORDERING)
    assert result["derived"]["jaw_axis_sign_is_a_label_not_a_measurement"] is True
    assert result["derived"]["approach_axis_source"] == (
        "controlled_body_origin_to_measured_finger_midpoint"
    )


def test_the_readback_carries_its_own_schema_version_and_controlled_body() -> None:
    result = _readback(controlled_body_name="base_link")

    assert result["schema_version"] == GRIPPER_FRAME_READBACK_SCHEMA_VERSION
    assert result["schema_version"] == "native_franka_gripper_frame_readback.v1"
    # Which body was measured is the whole question, so it is never implicit.
    assert result["controlled_body_name"] == "base_link"


def test_jaw_and_approach_orthogonality_is_measured_not_assumed() -> None:
    """A gripper whose jaw is not perpendicular to its tool axis would make the
    +Z/+Y frame unbuildable, so the residual is recorded rather than asserted
    away."""

    square = _readback(approach_axis_body=(0.0, 0.0, 1.0), jaw_axis_body=(0.0, 1.0, 0.0))
    assert square["derived"]["jaw_approach_orthogonality_dot"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert square["derived"]["jaw_approach_angle_rad"] == pytest.approx(
        math.pi / 2.0, abs=1e-9
    )

    skewed = _readback(
        approach_axis_body=(0.0, 0.0, 1.0),
        jaw_axis_body=(0.0, 1.0, 1.0),
    )
    assert skewed["derived"]["jaw_approach_orthogonality_dot"] == pytest.approx(
        1.0 / math.sqrt(2.0), abs=1e-9
    )


@pytest.mark.parametrize("missing", ["left_inner_finger", "right_inner_finger"])
def test_an_unresolvable_finger_body_refuses_by_name(missing) -> None:
    """Absence is a refusal, not a default axis: a readback that silently
    skipped would leave the next run with the same three-way contradiction and
    no way to tell that it had."""

    fingers = {
        "left_inner_finger": [0.0, 0.04, 0.16],
        "right_inner_finger": [0.0, -0.04, 0.16],
    }
    del fingers[missing]

    with pytest.raises(NativeFrankaPoseServoError) as excinfo:
        gripper_frame_axis_readback(
            controlled_body_name="base_link",
            body_position_world_m=[0.0, 0.0, 0.0],
            body_quaternion_world_xyzw=IDENTITY_XYZW,
            finger_positions_world_m=fingers,
        )

    assert excinfo.value.errors == (
        f"native_franka_pose_servo_finger_body_missing:{missing}",
    )


def test_coincident_fingers_refuse_instead_of_publishing_an_invented_axis() -> None:
    """Zero separation has no direction.  Normalising it anyway would put a
    unit vector nobody measured into the sealed evidence."""

    with pytest.raises(
        NativeFrankaPoseServoError,
        match="native_franka_pose_servo_gripper_frame_jaw_degenerate",
    ):
        gripper_frame_axis_readback(
            controlled_body_name="base_link",
            body_position_world_m=[0.0, 0.0, 0.0],
            body_quaternion_world_xyzw=IDENTITY_XYZW,
            finger_positions_world_m={
                "left_inner_finger": [0.0, 0.0, 0.16],
                "right_inner_finger": [0.0, 0.0, 0.16],
            },
        )


def test_a_midpoint_on_the_body_origin_refuses_the_approach_axis() -> None:
    """If the finger midpoint coincides with the controlled body's origin there
    is no tool direction to report."""

    with pytest.raises(
        NativeFrankaPoseServoError,
        match="native_franka_pose_servo_gripper_frame_approach_degenerate",
    ):
        gripper_frame_axis_readback(
            controlled_body_name="base_link",
            body_position_world_m=[0.0, 0.0, 0.0],
            body_quaternion_world_xyzw=IDENTITY_XYZW,
            finger_positions_world_m={
                "left_inner_finger": [0.0, 0.04, 0.0],
                "right_inner_finger": [0.0, -0.04, 0.0],
            },
        )


@pytest.mark.parametrize(
    "positions",
    [
        {"left_inner_finger": [0.0, 0.04], "right_inner_finger": [0.0, -0.04, 0.16]},
        {
            "left_inner_finger": [0.0, float("nan"), 0.16],
            "right_inner_finger": [0.0, -0.04, 0.16],
        },
    ],
)
def test_malformed_finger_positions_refuse_by_name(positions) -> None:
    with pytest.raises(
        NativeFrankaPoseServoError,
        match=(
            "native_franka_pose_servo_gripper_frame_position_invalid:"
            "left_inner_finger"
        ),
    ):
        gripper_frame_axis_readback(
            controlled_body_name="base_link",
            body_position_world_m=[0.0, 0.0, 0.0],
            body_quaternion_world_xyzw=IDENTITY_XYZW,
            finger_positions_world_m=positions,
        )


def test_an_unusable_body_quaternion_refuses_at_the_existing_boundary() -> None:
    with pytest.raises(
        NativeFrankaPoseServoError,
        match="native_franka_pose_servo_quaternion_invalid",
    ):
        gripper_frame_axis_readback(
            controlled_body_name="base_link",
            body_position_world_m=[0.0, 0.0, 0.0],
            body_quaternion_world_xyzw=[0.0, 0.0, 0.0, 0.0],
            finger_positions_world_m={
                "left_inner_finger": [0.0, 0.04, 0.16],
                "right_inner_finger": [0.0, -0.04, 0.16],
            },
        )


@pytest.mark.parametrize(
    ("finger_direction_world", "hypothesis_id"),
    [
        # The flange +Z points straight down at the sealed reset joints, which
        # is the observation hypothesis 2 rests on.
        ((0.0, 0.0, -1.0), "reset_body_quaternion_implies_a_coupler_rotation"),
        ((1.0, 0.0, 0.0), "tool_frame_convention_holds_for_controlled_body"),
        ((0.0, 1.0, 0.0), "wrist_camera_optical_axis_is_the_approach_axis"),
    ],
)
def test_the_sealed_reset_quaternion_makes_the_three_answers_visibly_different(
    finger_direction_world, hypothesis_id
) -> None:
    """State the run's answer in terms anyone can check by eye.

    At the measured reset quaternion ``(0.5, 0.5, 0.5, 0.5)`` the body axes are
    a cyclic permutation of the world axes -- ``+x_body`` is world ``+Y``,
    ``+y_body`` is world ``+Z``, ``+z_body`` is world ``+X`` -- so each
    hypothesis predicts the fingers sitting in a *different world direction*
    from the controlled body's origin:

      * fingers hanging straight down (world ``-Z``)  -> hypothesis 2
      * fingers along world ``+X``                    -> hypothesis 1
      * fingers along world ``+Y``                    -> hypothesis 3

    Three orthogonal outcomes, so the completed run names one and cannot name
    two.  This pins that mapping, so the reading of the run is arithmetic
    rather than a judgement call.
    """

    midpoint = [0.16 * value for value in finger_direction_world]
    jaw = (
        (1.0, 0.0, 0.0)
        if abs(finger_direction_world[1]) > 0.5
        else (0.0, 1.0, 0.0)
    )
    result = gripper_frame_axis_readback(
        controlled_body_name="base_link",
        body_position_world_m=[0.0, 0.0, 0.0],
        body_quaternion_world_xyzw=MEASURED_RESET_BODY_QUAT_XYZW,
        finger_positions_world_m={
            "right_inner_finger": [
                midpoint[index] - 0.04 * jaw[index] for index in range(3)
            ],
            "left_inner_finger": [
                midpoint[index] + 0.04 * jaw[index] for index in range(3)
            ],
        },
    )

    assert result["assessment"]["resolution"] == "supported"
    assert result["assessment"]["supported_hypothesis_id"] == hypothesis_id


def test_the_readback_survives_the_json_receipt_it_has_to_land_in() -> None:
    """The whole point is that a completed run carries this on disk.  A value
    that cannot be serialised would take the receipt down with it, which is the
    one failure mode that would lose the measurement silently."""

    import json

    result = _readback(body_quaternion_xyzw=MEASURED_RESET_BODY_QUAT_XYZW)

    assert json.loads(json.dumps(result)) == result
