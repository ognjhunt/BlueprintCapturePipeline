from __future__ import annotations

import pytest

from blueprint_pipeline.job_retry_policy import (
    MONEY,
    PRIVACY_RIGHTS,
    REFUSAL_RETRY_DELAYS_SECONDS,
    hard_stop_kind,
    outage_retry_delay,
    refusal_retry_delay,
)


def test_refusal_schedule_is_exact() -> None:
    assert REFUSAL_RETRY_DELAYS_SECONDS == (900, 3600, 10800, 21600, 43200, 86400)
    assert sum(REFUSAL_RETRY_DELAYS_SECONDS) == 166_500
    assert [refusal_retry_delay(n) for n in range(6)] == list(REFUSAL_RETRY_DELAYS_SECONDS)


@pytest.mark.parametrize("retries_done", [6, 7, 1_000_000])
def test_refusal_schedule_is_spent_after_sixth(retries_done: int) -> None:
    assert refusal_retry_delay(retries_done) is None


@pytest.mark.parametrize("retries_done", [-1, True, False, 1.0, "1", None])
def test_corrupt_refusal_counter_is_spent(retries_done: object) -> None:
    assert refusal_retry_delay(retries_done) is None  # type: ignore[arg-type]


def test_outage_backoff_doubles_to_cap() -> None:
    delays = [outage_retry_delay(n) for n in range(1, 9)]
    assert delays == [120, 240, 480, 960, 1920, 3600, 3600, 3600]


def test_outage_backoff_stays_capped_for_huge_failure_counts() -> None:
    assert outage_retry_delay(10_000) == 3600
    assert outage_retry_delay(10**18) == 3600


@pytest.mark.parametrize("failures", [0, -5, True, 2.5, "3", None])
def test_outage_backoff_treats_invalid_failures_as_first(failures: object) -> None:
    assert outage_retry_delay(failures) == 120  # type: ignore[arg-type]


def test_outage_backoff_custom_bounds() -> None:
    assert outage_retry_delay(1, first=5, cap=5) == 5
    assert outage_retry_delay(3, first=10, cap=1000) == 40


@pytest.mark.parametrize(
    ("first", "cap"),
    [(0, 3600), (120, 0), (-1, 3600), (4000, 3600), (True, 3600), (120, 3600.0), ("120", 3600)],
)
def test_outage_backoff_invalid_bounds_raise(first: object, cap: object) -> None:
    with pytest.raises(ValueError, match="job_retry_policy_bounds_invalid"):
        outage_retry_delay(1, first=first, cap=cap)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("consent_expired", PRIVACY_RIGHTS),
        ("website_capture_withdrawn", PRIVACY_RIGHTS),
        ("source_revoked", PRIVACY_RIGHTS),
        ("website_control_scene-sponsorship_http_409:consent_expired", PRIVACY_RIGHTS),
        ("website_atlas_authorization_invalid", PRIVACY_RIGHTS),
        ("scene_intake_authority_expired", PRIVACY_RIGHTS),
        ("dataset_license_missing", PRIVACY_RIGHTS),
        ("website_scene_preparation_budget_exhausted", MONEY),
        ("account_monthly_spend_limit_reached", MONEY),
        ("spend_admission:spend_admission_lock_stale", MONEY),
        ("beta_screening_job_ceiling_exceeded", MONEY),
        ("provider_credit_insufficient", MONEY),
        ("beta_screening_cost_quote_expired", MONEY),
        ("paid_resource_admission_not_admitted", MONEY),
        # Contains the word "budget": erring toward a hard stop is intended.
        ("control_plane_disk_budget_exceeded", MONEY),
        ("atlas_scene_planner_abstained", None),
        ("beta_screening_gpu_capacity_unavailable", None),
        ("vast_offer_selection_blocked", None),
        ("website_atlas_api_503", None),
        ("capacity", None),
        ("costume_missing", None),
        ("loan_repaid", None),
        ("CONSENT_EXPIRED", PRIVACY_RIGHTS),
        ("", None),
        (None, None),
        (42, None),
        ("paid_resource_consent_revoked", PRIVACY_RIGHTS),
        ("budget_exhausted:license_missing", PRIVACY_RIGHTS),
    ],
)
def test_hard_stop_kind(code: object, expected: str | None) -> None:
    assert hard_stop_kind(code) == expected
