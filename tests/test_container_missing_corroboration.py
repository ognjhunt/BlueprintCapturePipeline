"""A container that was never seen cannot have gone missing.

`adp-gaussian-excision-live-20260813T160321Z` was torn down four minutes in on
`vast_heartbeat_container_missing`, having compiled and installed three CUDA
rasterizer extensions. The retained log ends on `Successfully installed` with no
error, no terminal marker, and **zero** occurrences of the string the blocker is
named for. The workload was still running when the run was killed.

"No such container" before the first byte of output is a startup race: the poll
arrived before Docker created it. After output has been seen, the same marker
means something real -- it was there and now it is not.

This is the third blocker in that group to need corroboration. The transport
failure and instance-exited blockers were both misattributions first.
"""

from __future__ import annotations

import pytest

from blueprint_pipeline.vast_provider_adapter import (
    _log_result_container_vanished_after_output,
    _log_result_saw_container_missing,
)


def _attempt(*, missing: bool = False, bytes_out: int = 0) -> dict:
    return {
        "container_missing_marker_observed": missing,
        "output_size_bytes": bytes_out,
    }


def test_missing_before_any_output_is_a_startup_race() -> None:
    """The poll arrived before Docker created the container."""

    result = {
        "log_poll_attempts": [
            _attempt(missing=True),
            _attempt(missing=True),
            _attempt(bytes_out=4096),
            _attempt(bytes_out=33155),
        ]
    }

    assert _log_result_container_vanished_after_output(result) is False


def test_missing_after_output_is_a_real_disappearance() -> None:
    """It was there, we watched it, and now it is gone."""

    result = {
        "log_poll_attempts": [
            _attempt(bytes_out=4096),
            _attempt(bytes_out=33155),
            _attempt(missing=True, bytes_out=0),
        ]
    }

    assert _log_result_container_vanished_after_output(result) is True


def test_the_gaussian_excision_run_would_no_longer_be_killed() -> None:
    """The shape of the run that paid for this lesson.

    Startup polls found nothing, then the container appeared and streamed a
    CUDA build. Nothing ever observed it disappear.
    """

    result = {
        "log_poll_attempts": [
            _attempt(missing=True),
            _attempt(bytes_out=512),
            _attempt(bytes_out=18000),
            _attempt(bytes_out=33155),
        ]
    }

    assert _log_result_container_vanished_after_output(result) is False


def test_a_run_with_no_attempts_claims_nothing() -> None:
    assert _log_result_container_vanished_after_output({"log_poll_attempts": []}) is False
    assert _log_result_container_vanished_after_output({}) is False


@pytest.mark.parametrize("value", ["not-a-list", 17, None], ids=["str", "int", "none"])
def test_a_malformed_attempt_list_claims_nothing(value) -> None:
    """Absence of a well-formed record is not evidence of a missing container."""

    assert _log_result_container_vanished_after_output({"log_poll_attempts": value}) is False


def test_missing_observed_only_in_the_very_first_poll_is_not_terminal() -> None:
    """The single most common shape: one racy poll at instance creation."""

    result = {
        "log_poll_attempts": [_attempt(missing=True), _attempt(bytes_out=900)]
    }

    assert _log_result_container_vanished_after_output(result) is False


def test_the_transport_question_still_sees_any_sighting() -> None:
    """Splitting the predicate must not break the log-channel fallback.

    "Should we try another channel" and "did the container die" are different
    questions. One sighting answers the first; only a sighting after output
    answers the second.
    """

    startup_race = {"log_poll_attempts": [_attempt(missing=True), _attempt(bytes_out=900)]}

    assert _log_result_saw_container_missing(startup_race) is True
    assert _log_result_container_vanished_after_output(startup_race) is False
