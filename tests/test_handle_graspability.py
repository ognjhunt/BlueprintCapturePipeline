from __future__ import annotations

import pytest

from blueprint_pipeline.handle_graspability import (
    HANDLE_GRASPABILITY_SCHEMA_VERSION,
    HandleGraspabilityError,
    evaluate_handle_graspability,
)


def _evaluate(**overrides):
    arguments = {
        "handle_aabb_min_m": [-0.003, 0.3063, 1.004],
        "handle_aabb_max_m": [0.242, 0.3490, 1.041],
        "panel_face_offset_m": 0.0,
        "outward_normal_world": [0.0, 1.0, 0.0],
        "hinge_axis_world": [0.0, 0.0, 1.0],
        "required_pull_force_n": 27.0,
        "gripper_clamp_force_n": 70.0,
        "gripper_stroke_m": 0.085,
        "gripper_finger_clearance_m": 0.018,
        "friction_coefficient": 0.4,
    }
    arguments.update(overrides)
    return evaluate_handle_graspability(**arguments)


def test_a_handle_flush_to_the_panel_cannot_be_hooked() -> None:
    """Form closure needs somewhere to put the fingers.

    The 840796 handle protrudes 43mm but sits flat against the door, so there
    is no gap to reach behind. That leaves friction only, which is the grasp
    mode most likely to slip against a gasket.
    """

    report = _evaluate()

    assert report["form_closure_available"] is False
    assert report["grasp_mode"] == "friction_pinch_only"
    assert report["schema_version"] == HANDLE_GRASPABILITY_SCHEMA_VERSION


def test_a_bar_standing_off_the_panel_can_be_hooked() -> None:
    report = _evaluate(panel_face_offset_m=0.036)

    assert report["form_closure_available"] is True
    assert report["grasp_mode"] == "form_closure_available"


def test_pull_out_capacity_is_two_friction_faces_not_one() -> None:
    """Both jaws bear on the handle, so the friction budget is doubled."""

    report = _evaluate()

    assert report["pull_out_capacity_n"] == pytest.approx(2 * 0.4 * 70.0)


def test_a_friction_grasp_that_barely_beats_the_load_is_flagged() -> None:
    """A 1.1x margin on a friction pinch is a coin toss, not a plan.

    Clamp force, friction coefficient and the seal peak are all uncertain to
    tens of percent, so anything close to unity will fail some of the time and
    the failures will look like policy failures.
    """

    report = _evaluate(gripper_clamp_force_n=35.0, friction_coefficient=0.4)

    assert report["pull_out_margin"] == pytest.approx(28.0 / 27.0, abs=0.01)
    assert report["margin_sufficient"] is False
    assert any("margin_insufficient" in f for f in report["findings"])


def test_a_comfortable_margin_passes() -> None:
    report = _evaluate(gripper_clamp_force_n=70.0, friction_coefficient=0.5)

    assert report["margin_sufficient"] is True
    assert report["findings"] == []


def test_a_handle_too_big_across_every_useful_axis_fails_closed() -> None:
    """A gripper that cannot close across the pull has no grasp at any friction.

    Closing along the pull instead does not count: the jaws would simply let
    the handle slide out from between them, whatever the clamp force.
    """

    with pytest.raises(HandleGraspabilityError) as excinfo:
        _evaluate(
            handle_aabb_min_m=[0.0, 0.3063, 1.000],
            handle_aabb_max_m=[0.300, 0.3490, 1.140],
            gripper_stroke_m=0.085,
        )

    assert any("exceeds_gripper_stroke" in e for e in excinfo.value.errors)


def test_the_report_says_which_span_the_jaws_close_on() -> None:
    """Pinching across the hinge axis and across the normal are different grasps."""

    report = _evaluate()

    assert report["pinch_span_m"] == pytest.approx(0.037, abs=1e-6)
    assert report["pinch_axis"] == "hinge_axis"


def test_evaluation_is_deterministic() -> None:
    assert _evaluate() == _evaluate()
