"""Contract tests for the frozen ADP-009D pick-lift-translate-place grader.

Two kinds of test live here, deliberately kept apart:

* predicate tests, which feed a hand-built measurement mapping so a threshold
  can be probed exactly at its boundary without float representation error
  standing in the way; and
* episode tests, which run whole trajectories through
  :func:`score_task_episode` and therefore keep a safe margin either side of a
  threshold, because ``support_z + 0.08 - support_z`` is not ``0.08``.

Mixing the two would leave the boundary semantics untested and the episode
tests flaky, so each threshold gets both treatments.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.adp009d_task_scoring import (
    APPROVED_CAN_TOP_ABOVE_SUPPORT_M,
    CAN_START_POSITION_M,
    DESTINATION_MIN_DISTANCE_FROM_START_M,
    FAILURE_DROPPED,
    FAILURE_KNOCKED_OVER,
    FAILURE_NEVER_MOVED,
    FAILURE_PUSHED_OUT_OF_TASK_ENVELOPE,
    FRANKA_REACH_M,
    GRASP_CAPTURE_RADIUS_XY_M,
    GRASP_EVIDENCE_CONTACT,
    GRASP_EVIDENCE_HELD_CLEAR,
    GRASP_EVIDENCE_NOT_GRASPED,
    GRASP_EVIDENCE_UNAVAILABLE,
    GRIPPER_CLOSED_WIDTH_MAX_M,
    GRIPPER_FULL_OPENING_M,
    HOLD_TILT_TOLERANCE_DEG,
    HOLD_XY_TOLERANCE_M,
    HOLD_Z_TOLERANCE_M,
    JUDGEMENT_SOURCE,
    LIFT_CLEARANCE_M,
    OUTCOME_GRASPED,
    OUTCOME_LADDER,
    OUTCOME_LIFTED,
    OUTCOME_MOVED,
    OUTCOME_NEVER_MOVED,
    OUTCOME_PLACED,
    OUTCOME_TRANSLATED,
    PLACE_MAX_TILT_DEG,
    PLACE_RADIUS_M,
    ROBOT_BASE_POSITION_M,
    SETTLE_WINDOW_SAMPLES,
    STATUS_SCORED,
    STATUS_UNDETERMINED,
    SUPPORT_CLEARANCE_EPSILON_M,
    SUPPORT_PLANE_Z_M,
    TASK_SCORING_SCHEMA_VERSION,
    TaskScoringError,
    canonical_digest,
    dropped,
    grasp_evidence_for_sample,
    grasped,
    inside_task_envelope,
    knocked_over,
    lifted,
    measure_episode,
    moved,
    never_moved,
    normalize_object_samples,
    placed,
    pushed_out_of_task_envelope,
    resolve_outcome_ladder,
    score_task_episode,
    translated,
    validate_destination,
)

UPRIGHT_XYZW = (0.0, 0.0, 0.0, 1.0)
START = CAN_START_POSITION_M
# A destination 0.20 m along -x on the same support: clear of the 0.15 m floor,
# 0.5985 m from the robot base, and inside the SAGE collision ROI.
DESTINATION = (START[0] - 0.20, START[1], START[2])
CLOSED_WIDTH = GRIPPER_CLOSED_WIDTH_MAX_M
OPEN_WIDTH = GRIPPER_FULL_OPENING_M
# Episode-level margin.  Big enough to swamp float representation error at the
# support height, far smaller than any threshold under test.
EPS = 1.0e-6
LIFTED_Z = START[2] + LIFT_CLEARANCE_M + EPS


def _tilt_quaternion_xyzw(degrees: float) -> tuple[float, float, float, float]:
    """Unit quaternion tilting the body z axis by ``degrees`` about world x.

    Emitted in xyzw, matching what Isaac actually retains for this scene: a live
    upright can's root_pose_w quaternion is (-1.2e-05, -8.2e-05, -0.0, 1.0),
    with w last.  Fixtures in the other order would let a convention mistake
    pass the suite while breaking on real data.
    """

    half = math.radians(degrees) / 2.0
    return (math.sin(half), 0.0, 0.0, math.cos(half))


def _sample(
    step_index: int,
    position,
    *,
    tilt_deg: float = 0.0,
    gripper_width_m: float | None = OPEN_WIDTH,
    grasp_frame_position_world_m=None,
    finger_contact_forces_n=None,
) -> dict:
    quaternion = UPRIGHT_XYZW if tilt_deg == 0.0 else _tilt_quaternion_xyzw(tilt_deg)
    sample: dict = {"step_index": step_index, "can_pose_world": [*position, *quaternion]}
    if gripper_width_m is not None:
        sample["gripper_width_m"] = gripper_width_m
    if grasp_frame_position_world_m is not None:
        sample["grasp_frame_position_world_m"] = list(grasp_frame_position_world_m)
    if finger_contact_forces_n is not None:
        sample["finger_contact_forces_n"] = list(finger_contact_forces_n)
    return sample


def _hold(position, count: int, *, start_step: int = 0, **kwargs) -> list[dict]:
    """``count`` identical samples: the shape of a settled object."""

    return [_sample(start_step + index, position, **kwargs) for index in range(count)]


def _measure(samples, *, destination=DESTINATION, settle_window_samples=SETTLE_WINDOW_SAMPLES):
    return measure_episode(
        normalize_object_samples(samples),
        destination_position_world_m=destination,
        settle_window_samples=settle_window_samples,
    )


def _measurements(**overrides) -> dict:
    """A settled, untouched can.  Override only the fields under test.

    Hand-building the mapping is what lets a predicate be probed exactly at its
    threshold: these numbers are exact, whereas a pose differenced against the
    support height is not.
    """

    base: dict = {
        "max_horizontal_displacement_from_start_m": 0.0,
        "max_abs_z_displacement_from_start_m": 0.0,
        "max_tilt_deg": 0.0,
        "max_lift_above_start_m": 0.0,
        "min_horizontal_distance_to_destination_m": 0.20,
        "final_horizontal_distance_to_destination_m": 0.20,
        "min_clearance_above_support_m": 0.0,
        "final_clearance_above_support_m": 0.0,
        "final_inside_task_envelope": True,
        "grasped_sample_indices": [],
        "grasp_evidence_unavailable_sample_indices": [],
        "first_lifted_sample_index": None,
        "post_lift_landing_sample_index": None,
        "post_lift_landing_inside_destination": None,
        "settle_window_available": True,
        "settle_xy_span_m": 0.0,
        "settle_z_span_m": 0.0,
        "settle_tilt_span_deg": 0.0,
        "settle_max_tilt_deg": 0.0,
        "settle_min_clearance_above_support_m": 0.0,
        "settle_grasped": False,
    }
    base.update(overrides)
    return base


def _settled_at_destination(**overrides) -> dict:
    """A can that was grasped, carried, released and settled on the patch."""

    settled: dict = {
        "max_horizontal_displacement_from_start_m": 0.20,
        "max_lift_above_start_m": LIFT_CLEARANCE_M,
        "min_horizontal_distance_to_destination_m": 0.0,
        "final_horizontal_distance_to_destination_m": 0.0,
        "grasped_sample_indices": [1, 2, 3],
        "first_lifted_sample_index": 2,
        "post_lift_landing_sample_index": 4,
        "post_lift_landing_inside_destination": True,
    }
    settled.update(overrides)
    return _measurements(**settled)


def _successful_episode(
    *,
    final_position=None,
    final_tilt_deg: float = 0.0,
    settle_count: int = SETTLE_WINDOW_SAMPLES,
) -> list[dict]:
    """A canonical grasp, lift, carry, release and settle at the destination."""

    resting = final_position if final_position is not None else DESTINATION
    gripped = {"gripper_width_m": CLOSED_WIDTH, "finger_contact_forces_n": [3.0, 3.1]}
    return [
        _sample(0, START),
        # Close on the can at the sealed start pose, both fingers in contact.
        _sample(1, START, **gripped),
        # Lift clear of the support.
        _sample(2, (START[0], START[1], LIFTED_Z), **gripped),
        # Carry to above the destination.
        _sample(3, (resting[0], resting[1], LIFTED_Z), **gripped),
        # Lower onto the destination, still gripped.
        _sample(4, resting, **gripped),
        # Release, then hold still for the settle window.
        *_hold(
            resting,
            settle_count,
            start_step=5,
            tilt_deg=final_tilt_deg,
            gripper_width_m=OPEN_WIDTH,
            finger_contact_forces_n=[0.0, 0.0],
        ),
    ]


# ---------------------------------------------------------------------------
# Geometry and quaternion convention.
# ---------------------------------------------------------------------------


def test_tilt_is_read_in_the_declared_convention() -> None:
    """The order is the whole risk, so it is declared rather than assumed."""

    from blueprint_pipeline.adp009d_task_scoring import (
        QUATERNION_ORDER_WXYZ,
        QUATERNION_ORDER_XYZW,
        tilt_degrees_from_quaternion,
    )

    for degrees in (0.0, 15.0, 90.0):
        assert tilt_degrees_from_quaternion(
            _tilt_quaternion_xyzw(degrees), order=QUATERNION_ORDER_XYZW
        ) == pytest.approx(degrees, abs=1e-9)

    # Yaw does not tilt: a can spun about world z is still upright.
    half = math.sqrt(0.5)
    assert tilt_degrees_from_quaternion(
        (0.0, 0.0, half, half), order=QUATERNION_ORDER_XYZW
    ) == pytest.approx(0.0, abs=1e-9)

    # The failure this guards is silent, not loud.  Near identity the two
    # orders agree, so a settled can looks right either way -- exactly why a
    # wrong default survives a test suite.
    upright = _tilt_quaternion_xyzw(0.0)
    assert tilt_degrees_from_quaternion(
        upright, order=QUATERNION_ORDER_WXYZ
    ) == pytest.approx(
        tilt_degrees_from_quaternion(upright, order=QUATERNION_ORDER_XYZW), abs=1e-9
    )

    # But a can knocked fully onto its side reads as perfectly upright in the
    # wrong order, so knocked_over would never fire on the case it exists for.
    knocked = _tilt_quaternion_xyzw(90.0)
    assert tilt_degrees_from_quaternion(
        knocked, order=QUATERNION_ORDER_XYZW
    ) == pytest.approx(90.0, abs=1e-9)
    assert tilt_degrees_from_quaternion(
        knocked, order=QUATERNION_ORDER_WXYZ
    ) == pytest.approx(0.0, abs=1e-9)

    # An unknown order is refused rather than defaulted.
    with pytest.raises(TaskScoringError):
        tilt_degrees_from_quaternion(upright, order="wxzy")


def test_default_convention_matches_retained_live_isaac_data() -> None:
    """Retained ADP-009D data puts w last; the default must follow the data."""

    from blueprint_pipeline.adp009d_task_scoring import (
        QUATERNION_ORDER_XYZW,
        normalize_object_samples,
    )

    # Verbatim from a live run's canonical_hold_object_stability.final_pose_world.
    live_upright = [-1.2e-05, -8.2e-05, -0.0, 1.0]
    normalized = normalize_object_samples(
        [{"step_index": 0, "can_pose_world": [*START, *live_upright]}]
    )
    assert normalized[0]["quaternion_order"] == QUATERNION_ORDER_XYZW
    assert normalized[0]["tilt_deg"] == pytest.approx(0.0095, abs=1e-3)


def test_task_envelope_is_the_sage_roi_intersected_with_reach() -> None:
    assert inside_task_envelope(START) is True
    assert inside_task_envelope(DESTINATION) is True
    # Inside the SAGE ROI but past the arm's reach.
    beyond_reach = (
        ROBOT_BASE_POSITION_M[0],
        ROBOT_BASE_POSITION_M[1] - (FRANKA_REACH_M + 0.01),
        ROBOT_BASE_POSITION_M[2],
    )
    assert inside_task_envelope(beyond_reach) is False
    # Within reach of the base but below the ROI floor.
    assert inside_task_envelope((START[0], START[1], -0.2)) is False


# ---------------------------------------------------------------------------
# Malformed input.  Every one of these must raise, never score.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        ([], "task_scoring_samples_empty"),
        ("not-a-sequence", "task_scoring_samples_not_a_sequence"),
        ([{"step_index": 0}], "task_scoring_sample_0_can_pose_world_invalid"),
        (
            [{"step_index": 0, "can_pose_world": [*START, 1.0, 0.0, 0.0]}],
            "task_scoring_sample_0_can_pose_world_invalid",
        ),
        (
            [{"step_index": 0, "can_pose_world": [*START, float("nan"), 0.0, 0.0, 0.0]}],
            "task_scoring_sample_0_can_pose_world_invalid",
        ),
        (
            [{"step_index": 0, "can_pose_world": [*START, float("inf"), 0.0, 0.0, 0.0]}],
            "task_scoring_sample_0_can_pose_world_invalid",
        ),
        (
            [{"step_index": 0, "can_pose_world": [*START, 0.5, 0.0, 0.0, 0.0]}],
            "task_scoring_sample_0_quaternion_not_unit_norm",
        ),
        (
            [{"can_pose_world": [*START, *UPRIGHT_XYZW]}],
            "task_scoring_sample_0_step_index_invalid",
        ),
        (["not-a-mapping"], "task_scoring_sample_0_not_a_mapping"),
    ],
)
def test_malformed_samples_fail_closed(samples, expected) -> None:
    with pytest.raises(TaskScoringError) as exc_info:
        normalize_object_samples(samples)
    assert expected in exc_info.value.errors


def test_step_indices_must_strictly_increase() -> None:
    with pytest.raises(TaskScoringError) as exc_info:
        normalize_object_samples([_sample(0, START), _sample(0, START)])
    assert "task_scoring_sample_1_step_index_not_increasing" in exc_info.value.errors


def test_gripper_width_wider_than_the_2f85_stroke_is_malformed() -> None:
    with pytest.raises(TaskScoringError) as exc_info:
        normalize_object_samples([_sample(0, START, gripper_width_m=0.11)])
    assert "task_scoring_sample_0_gripper_width_exceeds_stroke" in exc_info.value.errors


def test_negative_gripper_width_and_contact_force_are_malformed() -> None:
    with pytest.raises(TaskScoringError) as exc_info:
        normalize_object_samples([_sample(0, START, gripper_width_m=-0.001)])
    assert "task_scoring_sample_0_gripper_width_invalid" in exc_info.value.errors

    with pytest.raises(TaskScoringError) as exc_info:
        normalize_object_samples([_sample(0, START, finger_contact_forces_n=[-1.0, 2.0])])
    assert "task_scoring_sample_0_contact_forces_invalid" in exc_info.value.errors


def test_a_shifted_quaternion_payload_is_rejected_as_a_convention_error() -> None:
    """Identity landing in the ``x`` slot reads as a can standing on its head."""

    samples = [{"step_index": 0, "can_pose_world": [*START, 0.0, 1.0, 0.0, 0.0]}]
    with pytest.raises(TaskScoringError) as exc_info:
        normalize_object_samples(samples)
    assert exc_info.value.errors == ("task_scoring_start_pose_not_upright_check_quaternion_order",)


def test_the_upright_guard_cannot_detect_an_xyzw_payload_of_an_upright_can() -> None:
    """The documented limit of the guard, pinned so nobody over-trusts it.

    ``(0, 0, 0, 1)`` is identity under xyzw and a 180 degree yaw under wxyz.
    Both describe an upright can, so no pose check can tell them apart.  The
    convention stays a contract with the caller.
    """

    samples = [{"step_index": 0, "can_pose_world": [*START, 0.0, 0.0, 0.0, 1.0]}]
    assert normalize_object_samples(samples)[0]["tilt_deg"] == pytest.approx(0.0, abs=1e-9)


def test_start_pose_must_be_the_sealed_can_pose_unless_explicitly_waived() -> None:
    displaced = (START[0] + 0.05, START[1], START[2])
    with pytest.raises(TaskScoringError) as exc_info:
        normalize_object_samples([_sample(0, displaced)])
    assert "task_scoring_start_pose_not_at_sealed_can_position" in exc_info.value.errors

    with pytest.raises(TaskScoringError) as exc_info:
        normalize_object_samples([_sample(0, (START[0], START[1], START[2] + 0.05))])
    assert "task_scoring_start_pose_not_at_sealed_support_height" in exc_info.value.errors

    # A scenario family that relocates the object must say so explicitly.
    assert normalize_object_samples([_sample(0, displaced)], require_sealed_start_pose=False)


def test_sealed_start_check_sits_on_the_canonical_hold_tolerance() -> None:
    at_tolerance = (START[0] + HOLD_XY_TOLERANCE_M, START[1], START[2])
    assert normalize_object_samples([_sample(0, at_tolerance)])
    just_past = (START[0] + HOLD_XY_TOLERANCE_M + EPS, START[1], START[2])
    with pytest.raises(TaskScoringError):
        normalize_object_samples([_sample(0, just_past)])


def test_an_invalid_settle_window_is_malformed() -> None:
    with pytest.raises(TaskScoringError) as exc_info:
        _measure(_hold(START, 2), settle_window_samples=0)
    assert "task_scoring_settle_window_invalid" in exc_info.value.errors


# ---------------------------------------------------------------------------
# Destination validation.
# ---------------------------------------------------------------------------


def test_destination_must_clear_the_minimum_translation() -> None:
    at_floor = (START[0] - (DESTINATION_MIN_DISTANCE_FROM_START_M + EPS), START[1], START[2])
    validated = validate_destination(at_floor, start_position_world_m=START)
    assert validated["distance_from_start_m"] == pytest.approx(
        DESTINATION_MIN_DISTANCE_FROM_START_M, abs=1e-5
    )

    just_short = (START[0] - (DESTINATION_MIN_DISTANCE_FROM_START_M - EPS), START[1], START[2])
    with pytest.raises(TaskScoringError) as exc_info:
        validate_destination(just_short, start_position_world_m=START)
    assert "task_scoring_destination_below_minimum_translation" in exc_info.value.errors


def test_destination_must_be_on_the_same_support_and_reachable() -> None:
    off_support = (DESTINATION[0], DESTINATION[1], DESTINATION[2] + 0.05)
    with pytest.raises(TaskScoringError) as exc_info:
        validate_destination(off_support, start_position_world_m=START)
    assert "task_scoring_destination_not_on_same_support" in exc_info.value.errors

    unreachable = (START[0], START[1] - 0.60, START[2])
    with pytest.raises(TaskScoringError) as exc_info:
        validate_destination(unreachable, start_position_world_m=START)
    assert "task_scoring_destination_outside_reachable_workspace" in exc_info.value.errors


def test_destination_outside_the_sage_roi_is_rejected() -> None:
    with pytest.raises(TaskScoringError) as exc_info:
        validate_destination((0.0, 0.0, START[2]), start_position_world_m=START)
    assert "task_scoring_destination_outside_sage_collision_roi" in exc_info.value.errors


def test_malformed_destination_fails_closed() -> None:
    with pytest.raises(TaskScoringError) as exc_info:
        validate_destination((1.0, 2.0), start_position_world_m=START)
    assert "task_scoring_destination_invalid" in exc_info.value.errors


def test_score_rejects_a_destination_the_frozen_task_could_not_have_used() -> None:
    too_close = (START[0] - 0.10, START[1], START[2])
    with pytest.raises(TaskScoringError):
        score_task_episode(
            samples=_successful_episode(), destination_position_world_m=too_close
        )


# ---------------------------------------------------------------------------
# Grasp evidence.
# ---------------------------------------------------------------------------


def _evidence(sample: dict) -> str:
    normalized = normalize_object_samples(
        [sample], require_sealed_start_pose=False
    )[0]
    return grasp_evidence_for_sample(normalized, support_plane_z_m=SUPPORT_PLANE_Z_M)


def test_grasp_evidence_prefers_contact_and_needs_two_bodies() -> None:
    assert (
        _evidence(_sample(0, START, gripper_width_m=OPEN_WIDTH, finger_contact_forces_n=[0.4, 0.6]))
        == GRASP_EVIDENCE_CONTACT
    )
    # One finger touching is a nudge, not a grasp.
    assert (
        _evidence(_sample(0, START, gripper_width_m=OPEN_WIDTH, finger_contact_forces_n=[0.4, 0.0]))
        == GRASP_EVIDENCE_NOT_GRASPED
    )


def test_held_clear_evidence_needs_a_closed_gripper_and_real_clearance() -> None:
    clear = (START[0], START[1], LIFTED_Z)
    assert (
        _evidence(_sample(0, clear, gripper_width_m=CLOSED_WIDTH)) == GRASP_EVIDENCE_HELD_CLEAR
    )
    # Exactly at the closed-width boundary counts; a micron wider does not.
    assert (
        _evidence(_sample(0, clear, gripper_width_m=GRIPPER_CLOSED_WIDTH_MAX_M))
        == GRASP_EVIDENCE_HELD_CLEAR
    )
    assert (
        _evidence(_sample(0, clear, gripper_width_m=GRIPPER_CLOSED_WIDTH_MAX_M + EPS))
        == GRASP_EVIDENCE_NOT_GRASPED
    )
    # A closed gripper with the can still on its support is not yet a grasp.
    assert _evidence(_sample(0, START, gripper_width_m=CLOSED_WIDTH)) == GRASP_EVIDENCE_NOT_GRASPED
    # Nor is a lift that has not cleared the settle-noise floor.
    barely = (START[0], START[1], START[2] + SUPPORT_CLEARANCE_EPSILON_M - EPS)
    assert _evidence(_sample(0, barely, gripper_width_m=CLOSED_WIDTH)) == GRASP_EVIDENCE_NOT_GRASPED


def test_held_clear_evidence_requires_the_can_between_the_fingers() -> None:
    clear = (START[0], START[1], LIFTED_Z)

    def _with_grasp_frame(dx: float, dz: float) -> str:
        return _evidence(
            _sample(
                0,
                clear,
                gripper_width_m=CLOSED_WIDTH,
                grasp_frame_position_world_m=(clear[0] + dx, clear[1], clear[2] + dz),
            )
        )

    # Inside the finger span and somewhere along the can's body.
    assert _with_grasp_frame(GRASP_CAPTURE_RADIUS_XY_M - EPS, 0.0) == GRASP_EVIDENCE_HELD_CLEAR
    assert (
        _with_grasp_frame(0.0, APPROVED_CAN_TOP_ABOVE_SUPPORT_M - EPS)
        == GRASP_EVIDENCE_HELD_CLEAR
    )
    # Beside the can rather than around it.
    assert (
        _with_grasp_frame(GRASP_CAPTURE_RADIUS_XY_M + 1e-3, 0.0) == GRASP_EVIDENCE_NOT_GRASPED
    )
    # Above the can's top, so the fingers cannot be on it.
    assert (
        _with_grasp_frame(0.0, APPROVED_CAN_TOP_ABOVE_SUPPORT_M + 1e-3)
        == GRASP_EVIDENCE_NOT_GRASPED
    )
    # Below the can's base.
    assert _with_grasp_frame(0.0, -1e-3) == GRASP_EVIDENCE_NOT_GRASPED


def test_no_gripper_channel_is_unavailable_not_not_grasped() -> None:
    assert _evidence(_sample(0, START, gripper_width_m=None)) == GRASP_EVIDENCE_UNAVAILABLE


# ---------------------------------------------------------------------------
# Predicates at their exact thresholds.
# ---------------------------------------------------------------------------


def test_never_moved_sits_exactly_on_the_canonical_hold_tolerances() -> None:
    assert never_moved(_measurements()) is True
    assert moved(_measurements()) is False

    for field, tolerance in (
        ("max_horizontal_displacement_from_start_m", HOLD_XY_TOLERANCE_M),
        ("max_abs_z_displacement_from_start_m", HOLD_Z_TOLERANCE_M),
        ("max_tilt_deg", HOLD_TILT_TOLERANCE_DEG),
    ):
        assert never_moved(_measurements(**{field: tolerance})) is True, field
        assert never_moved(_measurements(**{field: tolerance * 1.001})) is False, field
        assert moved(_measurements(**{field: tolerance * 1.001})) is True, field


def test_lifted_sits_exactly_on_the_frozen_lift_clearance() -> None:
    assert lifted(_measurements(max_lift_above_start_m=LIFT_CLEARANCE_M)) is True
    assert lifted(_measurements(max_lift_above_start_m=LIFT_CLEARANCE_M - 1e-9)) is False


def test_translated_sits_exactly_on_the_place_radius() -> None:
    assert (
        translated(_measurements(min_horizontal_distance_to_destination_m=PLACE_RADIUS_M)) is True
    )
    assert (
        translated(_measurements(min_horizontal_distance_to_destination_m=PLACE_RADIUS_M + 1e-9))
        is False
    )


def test_placed_sits_exactly_on_the_tilt_and_radius_tolerances() -> None:
    assert placed(_settled_at_destination()) is True
    assert (
        placed(_settled_at_destination(final_horizontal_distance_to_destination_m=PLACE_RADIUS_M))
        is True
    )
    assert (
        placed(
            _settled_at_destination(
                final_horizontal_distance_to_destination_m=PLACE_RADIUS_M + 1e-9
            )
        )
        is False
    )
    assert placed(_settled_at_destination(settle_max_tilt_deg=PLACE_MAX_TILT_DEG)) is True
    assert placed(_settled_at_destination(settle_max_tilt_deg=PLACE_MAX_TILT_DEG + 1e-9)) is False


@pytest.mark.parametrize(
    "field", ["settle_xy_span_m", "settle_z_span_m", "settle_tilt_span_deg"]
)
def test_placed_requires_the_can_to_have_come_to_rest(field: str) -> None:
    tolerance = {
        "settle_xy_span_m": HOLD_XY_TOLERANCE_M,
        "settle_z_span_m": HOLD_Z_TOLERANCE_M,
        "settle_tilt_span_deg": HOLD_TILT_TOLERANCE_DEG,
    }[field]
    assert placed(_settled_at_destination(**{field: tolerance})) is True
    assert placed(_settled_at_destination(**{field: tolerance * 1.001})) is False


def test_placed_requires_release_and_contact_with_the_support() -> None:
    assert placed(_settled_at_destination(settle_grasped=True)) is False
    assert (
        placed(
            _settled_at_destination(final_clearance_above_support_m=HOLD_Z_TOLERANCE_M + 1e-9)
        )
        is False
    )
    assert (
        placed(_settled_at_destination(final_clearance_above_support_m=HOLD_Z_TOLERANCE_M)) is True
    )


def test_knocked_over_sits_exactly_on_the_place_tilt_tolerance() -> None:
    assert knocked_over(_measurements(settle_max_tilt_deg=PLACE_MAX_TILT_DEG)) is False
    assert knocked_over(_measurements(settle_max_tilt_deg=PLACE_MAX_TILT_DEG + 1e-9)) is True
    # Tilt in flight is not a topple: the limit is a property of the settled can.
    airborne = _measurements(
        settle_max_tilt_deg=90.0, settle_min_clearance_above_support_m=0.30
    )
    assert knocked_over(airborne) is False


def test_dropped_separates_a_release_from_a_fall_by_where_it_lands() -> None:
    assert dropped(_measurements()) is False
    assert (
        dropped(
            _measurements(
                post_lift_landing_sample_index=7, post_lift_landing_inside_destination=True
            )
        )
        is False
    )
    assert (
        dropped(
            _measurements(
                post_lift_landing_sample_index=7, post_lift_landing_inside_destination=False
            )
        )
        is True
    )
    # Falling through the support plane is unconditional.
    assert (
        dropped(_measurements(min_clearance_above_support_m=-HOLD_Z_TOLERANCE_M - 1e-9)) is True
    )
    assert dropped(_measurements(min_clearance_above_support_m=-HOLD_Z_TOLERANCE_M)) is False


def test_pushed_requires_displacement_without_a_grasp_and_an_exit() -> None:
    swiped = _measurements(
        max_horizontal_displacement_from_start_m=0.6, final_inside_task_envelope=False
    )
    assert pushed_out_of_task_envelope(swiped) is True
    # Displaced but still inside the envelope.
    assert (
        pushed_out_of_task_envelope(_measurements(max_horizontal_displacement_from_start_m=0.6))
        is False
    )
    # Outside the envelope but never displaced is not physically reachable; the
    # predicate still refuses to claim a push on it.
    assert pushed_out_of_task_envelope(_measurements(final_inside_task_envelope=False)) is False
    # Grasped at some point, so whatever happened is not a push.
    assert (
        pushed_out_of_task_envelope(
            _measurements(
                grasped_sample_indices=[3],
                max_horizontal_displacement_from_start_m=0.6,
                final_inside_task_envelope=False,
            )
        )
        is False
    )


def test_grasped_distinguishes_no_evidence_from_no_grasp() -> None:
    assert grasped(_measurements()) is False
    assert grasped(_measurements(grasped_sample_indices=[2, 3])) is True
    assert grasped(_measurements(grasp_evidence_unavailable_sample_indices=[0, 1])) is None
    # Evidence somewhere beats a gap elsewhere: a grasp was actually observed.
    assert (
        grasped(
            _measurements(
                grasped_sample_indices=[2], grasp_evidence_unavailable_sample_indices=[0]
            )
        )
        is True
    )


# ---------------------------------------------------------------------------
# Whole episodes.
# ---------------------------------------------------------------------------


def test_a_successful_episode_scores_placed_with_a_complete_receipt() -> None:
    report = score_task_episode(
        samples=_successful_episode(), destination_position_world_m=DESTINATION
    )

    assert report["schema_version"] == TASK_SCORING_SCHEMA_VERSION
    assert report["status"] == STATUS_SCORED
    assert report["outcome"] == OUTCOME_PLACED
    assert report["outcome_rank"] == len(OUTCOME_LADDER) - 1
    assert report["task_succeeded"] is True
    assert report["ladder_truncated_at"] is None
    assert report["undetermined_reasons"] == []
    assert report["failure_modes"] == {
        FAILURE_DROPPED: False,
        FAILURE_KNOCKED_OVER: False,
        FAILURE_PUSHED_OUT_OF_TASK_ENVELOPE: False,
        FAILURE_NEVER_MOVED: False,
    }
    assert report["failure_modes_fully_determined"] is True
    assert report["judgement_source"] == JUDGEMENT_SOURCE
    assert report["quaternion_convention"] == "wxyz"
    for field in (
        "rendered_image_consulted",
        "learned_judge_consulted",
        "candidate_policy_queried",
        "caller_asserted_success_accepted",
    ):
        assert report[field] is False, field


def test_lift_is_measured_against_the_start_height_which_is_the_support_plane() -> None:
    measurements = _measure(_successful_episode())
    assert measurements["start_position_m"][2] == pytest.approx(SUPPORT_PLANE_Z_M, abs=1e-12)
    # The can's root origin rests on the support, so the two readings coincide.
    assert measurements["max_lift_above_start_m"] == pytest.approx(
        LIFT_CLEARANCE_M, abs=1e-5
    )


def test_translated_ignores_z_so_a_can_with_height_can_reach_the_patch() -> None:
    """The patch is on the support; folding z in would make 0.05 m unreachable."""

    measurements = _measure(_successful_episode())
    assert measurements["min_horizontal_distance_to_destination_m"] == pytest.approx(0.0, abs=1e-9)
    assert translated(measurements) is True


def test_a_can_settled_past_the_patch_is_translated_but_not_placed() -> None:
    outside = (DESTINATION[0] + PLACE_RADIUS_M + 1e-3, DESTINATION[1], DESTINATION[2])
    report = score_task_episode(
        samples=_successful_episode(final_position=outside),
        destination_position_world_m=DESTINATION,
    )
    assert report["outcome"] == OUTCOME_LIFTED
    assert report["predicates"][OUTCOME_TRANSLATED] is False
    assert report["ladder_truncated_at"] == OUTCOME_TRANSLATED


def test_placed_rejects_a_can_still_held_by_the_gripper() -> None:
    samples = _successful_episode()[:5]
    samples.extend(
        _hold(
            DESTINATION,
            SETTLE_WINDOW_SAMPLES,
            start_step=5,
            gripper_width_m=CLOSED_WIDTH,
            finger_contact_forces_n=[3.0, 3.0],
        )
    )
    measurements = _measure(samples)
    assert measurements["settle_grasped"] is True
    assert placed(measurements) is False


def test_placed_rejects_a_can_that_never_came_to_rest() -> None:
    samples = _successful_episode()[:5]
    for index in range(SETTLE_WINDOW_SAMPLES):
        drift = (DESTINATION[0] + index * 0.001, DESTINATION[1], DESTINATION[2])
        samples.append(_sample(5 + index, drift, gripper_width_m=OPEN_WIDTH))
    measurements = _measure(samples)
    assert measurements["settle_xy_span_m"] > HOLD_XY_TOLERANCE_M
    assert placed(measurements) is False


def test_dropped_when_the_can_lands_away_from_the_destination() -> None:
    midpoint = ((START[0] + DESTINATION[0]) / 2.0, START[1], START[2])
    samples = _successful_episode()[:4]
    samples.append(_sample(4, midpoint, gripper_width_m=OPEN_WIDTH))
    samples.extend(_hold(midpoint, SETTLE_WINDOW_SAMPLES, start_step=5, gripper_width_m=OPEN_WIDTH))
    measurements = _measure(samples)
    assert measurements["first_lifted_sample_index"] == 2
    assert measurements["post_lift_landing_inside_destination"] is False
    assert dropped(measurements) is True
    assert placed(measurements) is False


def test_a_clean_release_at_the_destination_is_not_a_drop() -> None:
    assert dropped(_measure(_successful_episode())) is False


def test_falling_below_the_support_plane_is_always_a_drop() -> None:
    floor = (DESTINATION[0], DESTINATION[1], SUPPORT_PLANE_Z_M - 0.4)
    samples = _successful_episode()[:4]
    samples.extend(_hold(floor, SETTLE_WINDOW_SAMPLES, start_step=4, gripper_width_m=OPEN_WIDTH))
    assert dropped(_measure(samples)) is True


def test_knocked_over_beyond_the_tilt_tolerance_at_rest() -> None:
    toppled = _measure(_successful_episode(final_tilt_deg=PLACE_MAX_TILT_DEG + 1.0))
    assert knocked_over(toppled) is True
    assert placed(toppled) is False

    upright = _measure(_successful_episode(final_tilt_deg=PLACE_MAX_TILT_DEG - 1.0))
    assert knocked_over(upright) is False
    assert placed(upright) is True


def test_a_can_swiped_over_without_ever_being_grasped_is_knocked_over() -> None:
    nudged = (START[0] + 0.02, START[1], START[2])
    samples = [_sample(0, START)]
    samples.extend(
        _hold(
            nudged, SETTLE_WINDOW_SAMPLES, start_step=1, tilt_deg=80.0, gripper_width_m=OPEN_WIDTH
        )
    )
    measurements = _measure(samples)
    assert knocked_over(measurements) is True
    assert grasped(measurements) is False


def test_pushed_out_of_the_task_envelope_requires_never_having_grasped() -> None:
    outside = (START[0], START[1] - 0.62, START[2])
    assert inside_task_envelope(outside) is False
    samples = [_sample(0, START)]
    samples.extend(_hold(outside, SETTLE_WINDOW_SAMPLES, start_step=1, gripper_width_m=OPEN_WIDTH))
    measurements = _measure(samples)
    assert pushed_out_of_task_envelope(measurements) is True
    assert measurements["final_inside_task_envelope"] is False


def test_a_carried_can_that_leaves_the_envelope_is_not_reported_as_pushed() -> None:
    outside = (START[0], START[1] - 0.62, START[2])
    samples = _successful_episode()[:3]
    samples.extend(
        _hold(
            outside,
            SETTLE_WINDOW_SAMPLES,
            start_step=3,
            gripper_width_m=CLOSED_WIDTH,
            finger_contact_forces_n=[3.0, 3.0],
        )
    )
    measurements = _measure(samples)
    assert grasped(measurements) is True
    assert pushed_out_of_task_envelope(measurements) is False
    # The fact itself is still on the receipt, just not under this name.
    assert measurements["final_inside_task_envelope"] is False


def test_never_moved_is_reported_as_a_failure_mode() -> None:
    report = score_task_episode(
        samples=_hold(START, SETTLE_WINDOW_SAMPLES),
        destination_position_world_m=DESTINATION,
    )
    assert report["failure_modes"][FAILURE_NEVER_MOVED] is True
    assert report["outcome"] == OUTCOME_NEVER_MOVED
    assert report["outcome_rank"] == 0
    assert report["ladder_truncated_at"] == OUTCOME_MOVED
    assert report["status"] == STATUS_SCORED


# ---------------------------------------------------------------------------
# Undetermined paths.
# ---------------------------------------------------------------------------


def _episode_without_gripper_evidence() -> list[dict]:
    carried = (DESTINATION[0], DESTINATION[1], LIFTED_Z)
    return [
        _sample(0, START, gripper_width_m=None),
        _sample(1, (START[0], START[1], LIFTED_Z), gripper_width_m=None),
        _sample(2, carried, gripper_width_m=None),
        *_hold(DESTINATION, SETTLE_WINDOW_SAMPLES, start_step=3, gripper_width_m=None),
    ]


def test_missing_gripper_evidence_leaves_grasp_undetermined_not_false() -> None:
    measurements = _measure(_episode_without_gripper_evidence())
    assert grasped(measurements) is None
    assert placed(measurements) is None
    assert measurements["settle_grasped"] is None


def test_an_episode_shorter_than_the_settle_window_cannot_claim_placed() -> None:
    short = _successful_episode(settle_count=SETTLE_WINDOW_SAMPLES - 6)
    measurements = _measure(short)
    assert measurements["sample_count"] < SETTLE_WINDOW_SAMPLES
    assert measurements["settle_window_available"] is False
    assert placed(measurements) is None
    assert knocked_over(measurements) is None


def test_undetermined_grasp_stops_the_ladder_at_moved() -> None:
    report = score_task_episode(
        samples=_episode_without_gripper_evidence(),
        destination_position_world_m=DESTINATION,
    )

    assert report["status"] == STATUS_UNDETERMINED
    assert report["outcome"] == OUTCOME_MOVED
    assert report["ladder_truncated_at"] == OUTCOME_GRASPED
    assert "grasped_undetermined" in report["undetermined_reasons"]
    assert report["task_succeeded"] is False
    # The higher rungs were measurable, and are recorded, but are not claimed.
    assert report["predicates"][OUTCOME_LIFTED] is True
    assert report["predicates"][OUTCOME_TRANSLATED] is True


def test_undetermined_failure_mode_is_surfaced_without_faking_a_verdict() -> None:
    report = score_task_episode(
        samples=_successful_episode(settle_count=SETTLE_WINDOW_SAMPLES - 6),
        destination_position_world_m=DESTINATION,
    )
    assert report["failure_modes_fully_determined"] is False
    assert report["failure_modes"][FAILURE_KNOCKED_OVER] is None
    assert "failure_mode_knocked_over_undetermined" in report["undetermined_reasons"]
    assert report["status"] == STATUS_UNDETERMINED
    assert report["task_succeeded"] is False


# ---------------------------------------------------------------------------
# The ladder.
# ---------------------------------------------------------------------------


def test_ladder_order_is_the_frozen_order() -> None:
    assert OUTCOME_LADDER == (
        OUTCOME_NEVER_MOVED,
        OUTCOME_MOVED,
        OUTCOME_GRASPED,
        OUTCOME_LIFTED,
        OUTCOME_TRANSLATED,
        OUTCOME_PLACED,
    )


def test_a_rung_cannot_be_skipped_even_when_a_higher_predicate_is_true() -> None:
    """The whole point of the ladder: evidence for a high rung is not a claim."""

    resolved = resolve_outcome_ladder(
        {
            OUTCOME_NEVER_MOVED: False,
            OUTCOME_MOVED: True,
            OUTCOME_GRASPED: False,
            OUTCOME_LIFTED: True,
            OUTCOME_TRANSLATED: True,
            OUTCOME_PLACED: True,
        }
    )
    assert resolved["outcome"] == OUTCOME_MOVED
    assert resolved["outcome_rank"] == 1
    assert resolved["ladder_truncated_at"] == OUTCOME_GRASPED
    assert resolved["status"] == STATUS_SCORED


@pytest.mark.parametrize(
    ("truncate_at", "expected_outcome", "expected_rank"),
    [
        (OUTCOME_MOVED, OUTCOME_NEVER_MOVED, 0),
        (OUTCOME_GRASPED, OUTCOME_MOVED, 1),
        (OUTCOME_LIFTED, OUTCOME_GRASPED, 2),
        (OUTCOME_TRANSLATED, OUTCOME_LIFTED, 3),
        (OUTCOME_PLACED, OUTCOME_TRANSLATED, 4),
        (None, OUTCOME_PLACED, 5),
    ],
)
def test_every_rung_is_reachable_and_the_walk_stops_where_evidence_stops(
    truncate_at, expected_outcome, expected_rank
) -> None:
    predicates: dict[str, bool | None] = dict.fromkeys(OUTCOME_LADDER, True)
    predicates[OUTCOME_NEVER_MOVED] = truncate_at != OUTCOME_MOVED
    if truncate_at is not None:
        predicates[truncate_at] = False
    resolved = resolve_outcome_ladder(predicates)
    assert resolved["outcome"] == expected_outcome
    assert resolved["outcome_rank"] == expected_rank
    assert resolved["ladder_truncated_at"] == truncate_at
    assert resolved["status"] == STATUS_SCORED


@pytest.mark.parametrize(
    ("undetermined_rung", "expected_outcome"),
    [
        (OUTCOME_MOVED, OUTCOME_NEVER_MOVED),
        (OUTCOME_GRASPED, OUTCOME_MOVED),
        (OUTCOME_LIFTED, OUTCOME_GRASPED),
        (OUTCOME_TRANSLATED, OUTCOME_LIFTED),
        (OUTCOME_PLACED, OUTCOME_TRANSLATED),
    ],
)
def test_an_undetermined_rung_never_lets_a_higher_rung_through(
    undetermined_rung, expected_outcome
) -> None:
    predicates: dict[str, bool | None] = dict.fromkeys(OUTCOME_LADDER, True)
    predicates[OUTCOME_NEVER_MOVED] = False
    predicates[undetermined_rung] = None
    resolved = resolve_outcome_ladder(predicates)
    assert resolved["outcome"] == expected_outcome
    assert resolved["status"] == STATUS_UNDETERMINED
    assert resolved["undetermined_reasons"] == [f"{undetermined_rung}_undetermined"]


# ---------------------------------------------------------------------------
# The receipt.
# ---------------------------------------------------------------------------


def test_receipt_carries_every_threshold_and_measurement_a_reviewer_needs() -> None:
    report = score_task_episode(
        samples=_successful_episode(), destination_position_world_m=DESTINATION
    )
    thresholds = report["thresholds"]

    assert thresholds["lift_clearance_m"] == LIFT_CLEARANCE_M
    assert thresholds["place_radius_m"] == PLACE_RADIUS_M
    assert thresholds["place_max_tilt_deg"] == PLACE_MAX_TILT_DEG
    assert thresholds["destination_min_distance_from_start_m"] == (
        DESTINATION_MIN_DISTANCE_FROM_START_M
    )
    assert thresholds["settle_window_samples"] == SETTLE_WINDOW_SAMPLES

    measurements = report["measurements"]
    for field in (
        "max_lift_above_start_m",
        "min_horizontal_distance_to_destination_m",
        "final_horizontal_distance_to_destination_m",
        "final_horizontal_displacement_from_start_m",
        "settle_xy_span_m",
        "settle_max_tilt_deg",
        "grasp_evidence_by_sample",
    ):
        assert field in measurements, field

    # Every predicate re-derives from the emitted measurements alone: this is
    # what lets a reviewer check the verdict without the simulator.
    assert lifted(measurements) is report["predicates"][OUTCOME_LIFTED]
    assert translated(measurements) is report["predicates"][OUTCOME_TRANSLATED]
    assert placed(measurements) is report["predicates"][OUTCOME_PLACED]
    assert grasped(measurements) is report["predicates"][OUTCOME_GRASPED]
    assert dropped(measurements) is report["failure_modes"][FAILURE_DROPPED]


def test_receipt_records_achieved_displacement_separately_from_the_patch_test() -> None:
    """The 0.15 m floor binds the destination, not the achieved displacement.

    A can settled on the near edge of the patch has travelled less than the
    destination's own distance from the start.  That is admitted by the frozen
    tolerances, so the achieved number is recorded rather than re-gated.
    """

    near_side = (DESTINATION[0] + 0.04, DESTINATION[1], DESTINATION[2])
    report = score_task_episode(
        samples=_successful_episode(final_position=near_side),
        destination_position_world_m=DESTINATION,
    )
    assert report["outcome"] == OUTCOME_PLACED
    achieved = report["measurements"]["final_horizontal_displacement_from_start_m"]
    assert achieved == pytest.approx(0.16, abs=1e-6)
    assert achieved < report["destination"]["distance_from_start_m"]
    assert report["destination"]["distance_from_start_m"] == pytest.approx(0.20, abs=1e-6)


def test_report_digest_binds_the_receipt() -> None:
    report = score_task_episode(
        samples=_successful_episode(), destination_position_world_m=DESTINATION
    )
    assert report["report_digest"] == canonical_digest(report, digest_field="report_digest")

    tampered = dict(report)
    tampered["outcome"] = OUTCOME_MOVED
    assert canonical_digest(tampered, digest_field="report_digest") != report["report_digest"]


def test_bundled_canonical_digest_matches_the_repository_contract() -> None:
    from blueprint_pipeline.decision_evidence_contracts import (
        canonical_digest as repository_digest,
    )

    payload = {"b": [1, 2, 3], "a": "x", "digest": "ignored"}
    assert canonical_digest(payload) == repository_digest(payload)
    assert canonical_digest(payload, digest_field="digest") == repository_digest(
        payload, digest_field="digest"
    )


def test_receipt_is_json_serializable_end_to_end() -> None:
    report = score_task_episode(
        samples=_successful_episode(), destination_position_world_m=DESTINATION
    )
    round_tripped = json.loads(json.dumps(report))
    assert round_tripped["report_digest"] == report["report_digest"]
    assert round_tripped["measurements"] == report["measurements"]


def test_numpy_inputs_are_accepted_verbatim() -> None:
    """Poses arrive as arrays from the runtime; they must not need pre-conversion."""

    samples = []
    for sample in _successful_episode():
        converted = dict(sample)
        converted["can_pose_world"] = np.asarray(sample["can_pose_world"], dtype=float)
        samples.append(converted)
    report = score_task_episode(
        samples=samples, destination_position_world_m=np.asarray(DESTINATION, dtype=float)
    )
    assert report["outcome"] == OUTCOME_PLACED


# ---------------------------------------------------------------------------
# Cross-module contracts.
# ---------------------------------------------------------------------------


def test_thresholds_match_the_harness_manifest_contract() -> None:
    """Changing one of these here without the harness breaks the frozen task."""

    from blueprint_pipeline import adp009d_franka_evaluation_harness as harness

    source = Path(harness.__file__).read_text(encoding="utf-8")
    assert f'"minimum_lift_m": {LIFT_CLEARANCE_M}' in source
    assert f'"minimum_translation_m": {DESTINATION_MIN_DISTANCE_FROM_START_M}' in source
    assert f'"maximum_center_error_m": {PLACE_RADIUS_M}' in source
    assert f'"maximum_tilt_degrees": {PLACE_MAX_TILT_DEG}' in source
    assert '"deterministic_simulator_state"' in source


def test_scene_constants_match_the_isaac_runtime_and_approach_capture() -> None:
    from blueprint_pipeline import adp009d_approach_capture as approach

    assert SUPPORT_PLANE_Z_M == approach.SUPPORT_HEIGHT_M
    assert APPROVED_CAN_TOP_ABOVE_SUPPORT_M == approach.APPROVED_CAN_TOP_ABOVE_SUPPORT_M
    assert (START[0], START[1]) == approach.CAN_AXIS_XY_M


def test_scoring_never_imports_a_simulator_or_a_learned_judge() -> None:
    from blueprint_pipeline import adp009d_task_scoring

    text = Path(adp009d_task_scoring.__file__).read_text(encoding="utf-8")
    for banned in ("import torch", "import omni", "from pxr", "import isaaclab", "import carb"):
        assert banned not in text, banned
