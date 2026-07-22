"""Provider-specific WAM poll state mapped onto the shared teardown contract."""

from __future__ import annotations

from .wam_async_runner_common import AsyncTeardownDecision, decide_async_teardown


def normalize_runpod_teardown_action(value: object) -> str:
    """Normalize the operator teardown setting to the three supported actions."""

    action = str(value or "").strip().lower()
    if action in {"stop", "stopped", "preserve", "warm"}:
        return "stop"
    if action in {"keep", "keep_running", "keep_on_success", "hot", "hot_reuse"}:
        return "keep_on_success"
    return "delete"


def decide_runpod_poll_teardown(
    *,
    explicit_requested: bool,
    requested_action: str,
    output_present: bool,
    provider_terminal: bool,
    runtime_stalled: bool,
    runtime_result_failed: bool,
    output_validation_failed: bool,
    keep_running: bool,
    allocation_actionable: bool,
    blockers_present: bool,
) -> AsyncTeardownDecision:
    """Translate RunPod WAM poll evidence into a fail-closed teardown decision."""

    return decide_async_teardown(
        explicit_requested=explicit_requested,
        requested_ready=bool(
            (output_present or provider_terminal or runtime_stalled) and not keep_running
        ),
        requested_action=requested_action,
        automatic_reasons=tuple(
            reason
            for reason, active in (
                ("runtime_stall", runtime_stalled),
                ("runtime_result_failed", runtime_result_failed),
                ("provider_output_validation_failed", output_validation_failed),
            )
            if active
        ),
        automatic_action="delete",
        allocation_actionable=allocation_actionable,
        blockers_present=blockers_present,
    )


def decide_vast_poll_teardown(
    *,
    explicit_requested: bool,
    max_live_deadline_expired: bool,
    provider_completed_or_blocked: bool,
    allocation_actionable: bool,
) -> AsyncTeardownDecision:
    """Translate Vast WAM poll evidence into a fail-closed destroy decision."""

    return decide_async_teardown(
        explicit_requested=explicit_requested,
        requested_ready=True,
        requested_action="destroy",
        automatic_reasons=tuple(
            reason
            for reason, active in (
                ("max_live_deadline_expired", max_live_deadline_expired),
                ("provider_completed_or_blocked", provider_completed_or_blocked),
            )
            if active
        ),
        automatic_action="destroy",
        allocation_actionable=allocation_actionable,
    )
