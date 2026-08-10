"""Decide whether a running provider instance is still doing anything.

A Vast instance does not stop when its entrypoint exits. It keeps running, and
keeps billing, until something tears it down. So "the instance is up" and "work
is happening" are different facts, and for twenty-odd launches I treated the
first as evidence of the second - reporting a run as "in progress" when the
payload had finished, uploaded its output, and gone quiet.

Two signals separate them, and both are already in the container log:

- a **terminal marker** the entrypoint prints on its way out
- whether the log has **changed at all** between two samples

Either alone can mislead. A marker can be absent from a run that died hard, and
a log can be briefly static during a long silent step - Arena's pip install
goes minutes without printing. Together they are decisive: a log that has not
moved AND ends in a terminal marker is a finished run, and every second after
that is paid for nothing.

The verdict is advisory. This module reads and reports; it never tears anything
down. Deciding to spend or stop spending is the allocator's job, and a monitor
that could kill a run would be a monitor that could kill a good one.
"""

from __future__ import annotations

from typing import Any, Sequence


PROVIDER_RUN_MONITOR_SCHEMA_VERSION = "provider_run_monitor.v1"
# Printed by the provider entrypoint on the way out, whatever the outcome.
DEFAULT_TERMINAL_MARKERS = (
    "BLUEPRINT_VAST_ONSTART_DONE",
    "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED",
)
# Printed by a payload that reached its own end, successfully or not.
DEFAULT_PAYLOAD_MARKERS = (
    "BLUEPRINT_ADP009D_WORKER_BLOCKED",
    "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK",
)


class ProviderRunMonitorError(ValueError):
    """Stable, sorted monitor failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def classify_run_progress(
    *,
    first_sample: str,
    second_sample: str,
    seconds_between_samples: float,
    terminal_markers: Sequence[str] = DEFAULT_TERMINAL_MARKERS,
    payload_markers: Sequence[str] = DEFAULT_PAYLOAD_MARKERS,
    minimum_sample_gap_seconds: float = 30.0,
) -> dict[str, Any]:
    """Say whether the run is working, finished, or merely quiet."""

    if seconds_between_samples < float(minimum_sample_gap_seconds):
        # Two samples taken close together prove nothing: any step that prints
        # once a minute looks identical to a dead one over five seconds.
        raise ProviderRunMonitorError(
            [
                "provider_run_monitor_sample_gap_too_short:"
                f"{seconds_between_samples}s<{minimum_sample_gap_seconds}s"
            ]
        )

    changed = first_sample != second_sample
    terminal = [m for m in terminal_markers if m in second_sample]
    payload_done = [m for m in payload_markers if m in second_sample]

    if changed:
        verdict = "working"
        reason = "log_advanced_between_samples"
    elif terminal:
        verdict = "finished_and_idle"
        reason = "log_static_and_terminal_marker_present"
    elif payload_done:
        # The payload ended but the entrypoint has not signed off; it may still
        # be uploading. Quiet, not yet provably over.
        verdict = "payload_done_entrypoint_pending"
        reason = "log_static_and_payload_marker_present"
    else:
        verdict = "quiet_no_terminal_marker"
        reason = "log_static_but_nothing_says_it_finished"

    return {
        "schema_version": PROVIDER_RUN_MONITOR_SCHEMA_VERSION,
        "verdict": verdict,
        "reason": reason,
        "log_advanced": changed,
        "terminal_markers_seen": sorted(terminal),
        "payload_markers_seen": sorted(payload_done),
        "seconds_between_samples": float(seconds_between_samples),
        # Only one verdict justifies stopping the meter, and even then the
        # monitor recommends rather than acts.
        "billing_is_buying_nothing": verdict == "finished_and_idle",
        "claim_boundary": {
            "monitor_reads_and_reports_it_never_tears_down": True,
            "a_static_log_alone_is_not_a_finished_run": True,
        },
    }


def estimate_idle_cost_usd(
    *, hourly_rate_usd: float, idle_seconds: float
) -> float:
    """What the idle stretch cost, so the waste is a number and not a feeling."""

    return max(0.0, float(hourly_rate_usd)) * max(0.0, float(idle_seconds)) / 3600.0


__all__ = [
    "DEFAULT_PAYLOAD_MARKERS",
    "DEFAULT_TERMINAL_MARKERS",
    "PROVIDER_RUN_MONITOR_SCHEMA_VERSION",
    "ProviderRunMonitorError",
    "classify_run_progress",
    "estimate_idle_cost_usd",
]
