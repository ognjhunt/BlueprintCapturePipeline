from __future__ import annotations

import pytest

from blueprint_pipeline.adp009d_control_episode import (
    TASK_CONTROL_RECOVERY_LADDER,
    recovery_ladder_for_plan,
)
from blueprint_pipeline.native_task_arena_recovery_advisor import (
    ADVISORY_SCHEMA_VERSION,
    RecoveryAdvisoryError,
    plan_with_advised_ladder,
    summarize_attempts_for_advice,
    validate_recovery_advisory,
)


def _advisory(ladder, **overrides) -> dict:
    advisory = {
        "schema_version": ADVISORY_SCHEMA_VERSION,
        "recovery_strategy_ladder": list(ladder),
        "rationale": "re-entry rungs landed on an identical endpoint",
    }
    advisory.update(overrides)
    return advisory


def _episode() -> dict:
    """C32's sealed shape: five attempts and a saturated wrist."""

    return {
        "phase_arrivals": [
            {
                "phase_id": "contact_open",
                "attempt": 1,
                "recovery_strategy": None,
                "terminal_position_error_m": 0.01163,
                "terminal_orientation_error_rad": 0.098,
                "commanded_position_bias_m": [0.0, 0.0, 0.0],
            },
            {
                "phase_id": "contact_open",
                "attempt": 5,
                "recovery_strategy": "extended_standoff_reentry",
                "terminal_position_error_m": 0.01539,
                "terminal_orientation_error_rad": 0.098,
                "commanded_position_bias_m": [0.0, 0.0, 0.0],
            },
        ],
        "action_trace": [
            {
                "phase_id": "contact_open",
                "arm_dynamics_after": {
                    "joint_effort_utilization": [0.2, 0.0, 0.1, 0.5, 0.4, 1.0, 0.4]
                },
            },
            {
                "phase_id": "contact_open",
                "arm_dynamics_after": {
                    "joint_effort_utilization": [0.1, 0.0, 0.1, 0.2, 0.1, 0.3, 0.1]
                },
            },
        ],
        # Vocabulary the summary must not carry through to an adviser.
        "score": {"outcome": "joint_limit_or_containment_violation"},
    }


def test_the_summary_shows_measured_behaviour_and_no_verdict() -> None:
    """An adviser reasons about what the arm did, not about the grade."""

    summary = summarize_attempts_for_advice(_episode())

    attempts = summary["phases"]["contact_open"]
    assert [row["attempt"] for row in attempts] == [1, 5]
    assert attempts[0]["position_error_m"] == pytest.approx(0.01163)
    # The saturation that took thirty-two runs to notice is surfaced directly.
    saturation = summary["actuator_saturation"]["contact_open"]
    assert saturation["steps"] == 2
    assert saturation["saturated_steps"] == 1
    assert saturation["maximum_utilization"] == pytest.approx(1.0)
    assert summary["available_strategies"] == list(TASK_CONTROL_RECOVERY_LADDER)
    # No outcome, no tolerance, nothing an adviser could mistake for a
    # verdict -- checked as structure rather than as text, so the boundary
    # statement describing the rule does not itself trip it.
    from blueprint_pipeline.native_task_arena_recovery_advisor import (
        _forbidden_paths,
    )

    assert _forbidden_paths(summary) == []
    # The source episode did carry a verdict; the summary dropped it.
    assert _forbidden_paths(_episode()) != []


def test_an_advised_ladder_reorders_the_search_and_is_sealed() -> None:
    third, first = TASK_CONTROL_RECOVERY_LADDER[2], TASK_CONTROL_RECOVERY_LADDER[0]
    plan = {"schema_version": "adp_task_control_plan.v1"}

    bound, receipt = plan_with_advised_ladder(
        control_plan=plan, advisory=_advisory([third, first])
    )

    assert recovery_ladder_for_plan(bound) == (third, first)
    assert receipt["source"] == "agent_advisory"
    assert receipt["ladder"] == [third, first]
    assert receipt["advisory_digest"].startswith("sha256:")
    assert receipt["advisory_rationale"]
    assert receipt["blockers"] == []
    # The advice moves the search order and nothing else.
    assert "gate" not in bound
    assert bound["schema_version"] == "adp_task_control_plan.v1"


def test_an_advisory_may_not_invent_a_rung() -> None:
    """A model can rank the physics we implement, never propose new physics."""

    with pytest.raises(RecoveryAdvisoryError) as excinfo:
        validate_recovery_advisory(
            _advisory([TASK_CONTROL_RECOVERY_LADDER[0], "teleport_the_gripper"])
        )

    assert any("unknown_strategy" in error for error in excinfo.value.errors)


def test_an_advisory_may_not_carry_outcome_vocabulary() -> None:
    """A ranking is the whole of what an adviser is allowed to express."""

    for forbidden in (
        {"task_succeeded": True},
        {"controls_qualified": True},
        {"arrival_tolerance_m": 0.05},
        {"nested": {"outcome": "opened_and_settled"}},
    ):
        with pytest.raises(RecoveryAdvisoryError) as excinfo:
            validate_recovery_advisory(
                _advisory(list(TASK_CONTROL_RECOVERY_LADDER), **forbidden)
            )
        assert any(
            "outcome_vocabulary_forbidden" in error for error in excinfo.value.errors
        )


def test_a_refused_or_absent_advisory_degrades_order_not_recovery() -> None:
    """A provider outage must not disable the search it was ranking."""

    plan = {"schema_version": "adp_task_control_plan.v1"}

    absent_plan, absent = plan_with_advised_ladder(control_plan=plan, advisory=None)
    assert absent["source"] == "default_ladder"
    assert recovery_ladder_for_plan(absent_plan) == TASK_CONTROL_RECOVERY_LADDER

    refused_plan, refused = plan_with_advised_ladder(
        control_plan=plan,
        advisory=_advisory(["teleport_the_gripper"]),
    )
    assert refused["source"] == "default_ladder_after_refused_advisory"
    assert refused["blockers"]
    # Still the full default ladder: recovery is never silently switched off.
    assert recovery_ladder_for_plan(refused_plan) == TASK_CONTROL_RECOVERY_LADDER


def test_a_rationale_is_required_so_a_ranking_is_auditable() -> None:
    with pytest.raises(RecoveryAdvisoryError):
        validate_recovery_advisory(
            _advisory(list(TASK_CONTROL_RECOVERY_LADDER), rationale="   ")
        )
