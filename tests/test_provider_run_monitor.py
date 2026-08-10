"""Telling a finished run from a quiet one."""

from __future__ import annotations

import pytest

from blueprint_pipeline.provider_run_monitor import (
    ProviderRunMonitorError,
    classify_run_progress,
    estimate_idle_cost_usd,
)


def _classify(first, second, gap=75.0):
    return classify_run_progress(
        first_sample=first, second_sample=second, seconds_between_samples=gap
    )


def test_a_log_that_advanced_means_work_is_happening():
    verdict = _classify("step 1\n", "step 1\nstep 2\n")

    assert verdict["verdict"] == "working"
    assert verdict["billing_is_buying_nothing"] is False


def test_a_static_log_ending_in_a_terminal_marker_is_a_finished_run():
    """This is rt22: payload done, output uploaded, instance still billing.

    The instance stays up after its entrypoint exits, so "up for 43 minutes"
    was read as progress when nothing had happened for most of it.
    """

    log = "work\nBLUEPRINT_VAST_ONSTART_DONE\n"

    verdict = _classify(log, log)

    assert verdict["verdict"] == "finished_and_idle"
    assert verdict["billing_is_buying_nothing"] is True
    assert "BLUEPRINT_VAST_ONSTART_DONE" in verdict["terminal_markers_seen"]


def test_a_static_log_with_no_marker_is_quiet_not_finished():
    """Arena's pip install goes minutes without printing; that is not death."""

    log = "Collecting isaaclab\n"

    verdict = _classify(log, log)

    assert verdict["verdict"] == "quiet_no_terminal_marker"
    assert verdict["billing_is_buying_nothing"] is False


def test_a_finished_payload_without_entrypoint_signoff_is_not_yet_idle():
    """It may still be uploading; stopping there would lose the output."""

    log = "BLUEPRINT_ADP009D_WORKER_BLOCKED\n"

    verdict = _classify(log, log)

    assert verdict["verdict"] == "payload_done_entrypoint_pending"
    assert verdict["billing_is_buying_nothing"] is False


def test_samples_taken_too_close_together_are_refused():
    """Five seconds of silence proves nothing about a minute-long step."""

    with pytest.raises(ProviderRunMonitorError) as excinfo:
        _classify("same\n", "same\n", gap=5.0)

    assert any("sample_gap_too_short" in error for error in excinfo.value.errors)


def test_the_idle_cost_is_a_number_not_a_feeling():
    assert estimate_idle_cost_usd(hourly_rate_usd=0.57, idle_seconds=1800) == pytest.approx(0.285)
    assert estimate_idle_cost_usd(hourly_rate_usd=0.57, idle_seconds=0) == 0.0
