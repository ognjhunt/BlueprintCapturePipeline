"""Interpret retained Vast log observations without provider side effects."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _log_result_saw_container_missing(log_result: Mapping[str, Any]) -> bool:
    """Did any poll see "No such container"?

    This answers a transport question -- *should we try another channel* -- and
    any sighting is the right trigger for that, including one during startup.
    It deliberately does not answer whether the container died; see
    `_log_result_container_vanished_after_output`, which is the one a blocker
    may rely on.
    """

    attempts = log_result.get("log_poll_attempts")
    if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
        return False
    return any(
        isinstance(item, Mapping) and bool(item.get("container_missing_marker_observed"))
        for item in attempts
    )


def _log_result_container_vanished_after_output(log_result: Mapping[str, Any]) -> bool:
    """Did the container go missing *after* we watched it working?

    "No such container" before the first byte of output is a startup race, not
    a dead container: the poll simply arrived before Docker created it. Treating
    that as terminal kills runs that are about to work.

    That is not hypothetical. `adp-gaussian-excision-live-20260813T160321Z` was
    torn down four minutes in on `vast_heartbeat_container_missing`, having
    compiled and installed three CUDA rasterizer extensions -- the whole log
    ends on `Successfully installed`, with no error and no terminal marker,
    because the workload was still going. The final fetched log contains zero
    occurrences of the marker the blocker is named for.

    So the marker only counts once output has been seen. After that, a missing
    container is a real one: it was there, and now it is not. The same shape as
    the transport-failure and instance-exited blockers beside it, both of which
    were misattributions until they were made to corroborate.
    """

    attempts = log_result.get("log_poll_attempts")
    if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
        return False
    seen_output = False
    for item in attempts:
        if not isinstance(item, Mapping):
            continue
        if seen_output and bool(item.get("container_missing_marker_observed")):
            return True
        if int(item.get("output_size_bytes") or 0) > 0:
            seen_output = True
    return False
