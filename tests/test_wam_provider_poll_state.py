from __future__ import annotations

from blueprint_pipeline.wam_provider_poll_state import (
    decide_runpod_poll_teardown,
    decide_vast_poll_teardown,
    normalize_runpod_teardown_action,
)


def test_runpod_teardown_action_normalization_preserves_operator_contract() -> None:
    assert normalize_runpod_teardown_action("preserve") == "stop"
    assert normalize_runpod_teardown_action("HOT_REUSE") == "keep_on_success"
    assert normalize_runpod_teardown_action("unexpected") == "delete"


def test_runpod_explicit_keep_waits_without_terminal_evidence() -> None:
    decision = decide_runpod_poll_teardown(
        explicit_requested=True,
        requested_action="keep_on_success",
        output_present=False,
        provider_terminal=False,
        runtime_stalled=False,
        runtime_result_failed=False,
        output_validation_failed=False,
        keep_running=True,
        allocation_actionable=True,
        blockers_present=False,
    )

    assert decision.should_teardown is False
    assert decision.action == "keep_on_success"
    assert decision.teardown_pending is False


def test_runpod_failure_forces_delete_and_records_all_reasons() -> None:
    decision = decide_runpod_poll_teardown(
        explicit_requested=False,
        requested_action="stop",
        output_present=True,
        provider_terminal=False,
        runtime_stalled=True,
        runtime_result_failed=True,
        output_validation_failed=True,
        keep_running=False,
        allocation_actionable=True,
        blockers_present=False,
    )

    assert decision.should_teardown is True
    assert decision.action == "delete"
    assert decision.teardown_pending is True
    assert decision.automatic_reasons == (
        "runtime_stall",
        "runtime_result_failed",
        "provider_output_validation_failed",
    )


def test_runpod_blocker_prevents_mutation_without_erasing_decision() -> None:
    decision = decide_runpod_poll_teardown(
        explicit_requested=True,
        requested_action="stop",
        output_present=True,
        provider_terminal=False,
        runtime_stalled=False,
        runtime_result_failed=False,
        output_validation_failed=False,
        keep_running=False,
        allocation_actionable=True,
        blockers_present=True,
    )

    assert decision.should_teardown is True
    assert decision.teardown_pending is False


def test_vast_deadline_forces_destroy_without_explicit_request() -> None:
    decision = decide_vast_poll_teardown(
        explicit_requested=False,
        max_live_deadline_expired=True,
        provider_completed_or_blocked=False,
        allocation_actionable=True,
    )

    assert decision.effective_requested is True
    assert decision.action == "destroy"
    assert decision.teardown_pending is True
    assert decision.automatic_reasons == ("max_live_deadline_expired",)
