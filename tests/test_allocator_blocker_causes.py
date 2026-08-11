"""Admission blockers must carry the cause, not just the exception type.

rt53's first two launches blocked with the literal string
``adp009d_bundle_preparation_failed:ValueError`` - the exception's own
message (``adp009d_bound_asset_digest_mismatch:approved_can.usda``, then
``adp009d_task_collision_target_not_in_roi``) was discarded at the except
site. Each opaque blocker cost a separate local reproduction to recover a
cause that had already been raised in full. Same swallow pattern as rt26,
one layer up.
"""

from __future__ import annotations

from blueprint_pipeline.paid_resource_allocator import blocker_with_cause


def test_the_exception_message_travels_in_the_blocker():
    exc = ValueError("adp009d_bound_asset_digest_mismatch:approved_can.usda")

    blocker = blocker_with_cause("adp009d_bundle_preparation_failed", exc)

    assert "adp009d_bundle_preparation_failed" in blocker
    assert "ValueError" in blocker
    assert "adp009d_bound_asset_digest_mismatch:approved_can.usda" in blocker


def test_structured_errors_are_preferred_over_the_message():
    class Structured(ValueError):
        def __init__(self):
            super().__init__("joined")
            self.errors = ("cause_b", "cause_a")

    blocker = blocker_with_cause("prefix", Structured())

    assert "cause_a" in blocker and "cause_b" in blocker


def test_a_messageless_exception_still_names_its_type():
    blocker = blocker_with_cause("prefix", ValueError())

    assert blocker.startswith("prefix:ValueError")


def test_the_cause_is_bounded_so_blockers_stay_loggable():
    blocker = blocker_with_cause("prefix", ValueError("x" * 5000))

    assert len(blocker) <= 400


def test_terminal_interrupt_hardening_ignores_sigint_and_sighup():
    """rt58b and rt58c both died to stray terminal signals mid-probe -
    $0.05 of aborted instances for signals no human sent. A paid probe in
    flight must not be interruptible by the session plumbing around it;
    deliberate stops use SIGTERM, which stays honored."""

    import signal

    from blueprint_pipeline.paid_resource_allocator import (
        harden_against_terminal_interrupts,
    )

    before_int = signal.getsignal(signal.SIGINT)
    before_hup = signal.getsignal(signal.SIGHUP)
    before_term = signal.getsignal(signal.SIGTERM)
    try:
        ignored = harden_against_terminal_interrupts()

        assert set(ignored) == {"SIGINT", "SIGHUP"}
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) is before_term
    finally:
        signal.signal(signal.SIGINT, before_int)
        signal.signal(signal.SIGHUP, before_hup)
