"""A single-attempt paid lane may re-select an offer the provider refused before any mutation."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from blueprint_pipeline import adp_retained_scene_render_vast as render_lane
from blueprint_pipeline import task_evaluation_scene_configuration_vast as configuration_lane
from blueprint_pipeline import vast_pre_mutation_reselection as reselection

PACKAGE = Path(reselection.__file__).resolve().parent
ZERO_PIN = re.compile(r'os\.environ\[(?:_VAST_STALE_OFFER_RETRY_ENV|_VAST_SINGLE_ATTEMPT_ENV|_RETRY_ENV)\]\s*=\s*"0"')


def test_the_reselection_bound_is_small_and_explicit() -> None:
    assert 1 <= reselection.PRE_MUTATION_OFFER_RESELECTION_ATTEMPTS <= 3
    assert reselection.pre_mutation_offer_reselection_attempts() == str(reselection.PRE_MUTATION_OFFER_RESELECTION_ATTEMPTS)
    assert reselection.RESELECTION_ENV == "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"


@pytest.mark.parametrize("lane", [render_lane, configuration_lane], ids=["calibration_render", "scene_configuration"])
def test_the_841757_path_lanes_allow_bounded_reselection_inside_their_sealed_environment(lane, monkeypatch) -> None:
    """Submission #9 (2026-09-05): the calibration render selected RTX 4090 offer
    49588631 and Vast refused the create with 400 ``no_such_ask`` because the
    offer had just been rented.  Nothing was created or spent, yet the lane
    pinned re-selection to zero, so one stale offer cost a whole submission
    cycle.  The single-attempt doctrine governs instances that were created; a
    refused create is not an attempt."""

    monkeypatch.setenv(reselection.RESELECTION_ENV, "caller")

    with lane._authority_environment():
        assert os.environ[reselection.RESELECTION_ENV] == reselection.pre_mutation_offer_reselection_attempts()
        assert os.environ["BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH"] == "1"

    assert os.environ[reselection.RESELECTION_ENV] == "caller"


def test_no_paid_lane_pins_reselection_to_zero_any_more() -> None:
    pinned = sorted(path.name for path in PACKAGE.glob("*.py") if ZERO_PIN.search(path.read_text(encoding="utf-8")))
    assert pinned == []
