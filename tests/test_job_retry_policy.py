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
        # Stress round 2 (2026-11-24): codes whose words read as a wall but guard none.
        ("atlas_coverage_exceeds_frame_budget", None),
        ("website_atlas_consent_unconfigured", None),
        ("website_control_http_409:website_atlas_consent_unconfigured", None),
        ("website_scene_grant_policy_unconfigured", None),
        # A planner answer in the wrong form, retained on jobs failed before proposal review r2.
        ("atlas_scene_planner_coverage_or_privacy_invalid", None),
        # A request's token count, a review answer's frame list in the wrong form, a probe count (2026-11-24).
        ("atlas_scene_planner_context_budget_exceeded", None),
        ("atlas_observed_room_privacy_selection_invalid", None),
        ("website_concept_probe_budget_invalid", None),
        ("clean_plate: website_concept_probe_budget_invalid", None),
        # The review's dollars against its maximum stay a money wall.
        ("atlas_scene_planner_token_budget_exceeded", MONEY),
        # Only the whole token is taken out: the rest of a message, and a longer code, still count.
        ("website_atlas_consent_unconfigured:consent_revoked", PRIVACY_RIGHTS),
        ("atlas_coverage_exceeds_frame_budget_usd", MONEY),
        ("website_atlas_consent_unconfigured_x", PRIVACY_RIGHTS),
    ],
)
def test_hard_stop_kind(code: object, expected: str | None) -> None:
    assert hard_stop_kind(code) == expected


def test_provider_answer_names_an_outside_services_http_status_only():
    from blueprint_pipeline.job_retry_policy import provider_answer

    assert provider_answer("meta_sam_http_413") == 413
    assert provider_answer("meta_sam_upload_http_503:detail") == 503
    assert provider_answer("website_atlas_api_429") is None  # no http_ marker: decided elsewhere
    assert provider_answer("meta_sam_http_5030") is None
    assert provider_answer("meta_sam_video_upload_requires_reconciliation") is None
    # Blueprint's own WebApp answers carry their typed reason after the colon.
    assert provider_answer("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted") is None
    assert provider_answer("owned_capture_atlas_control_x_http_503") is None
    assert provider_answer(None) is None


def test_request_cannot_succeed_names_refusals_of_the_request_itself():
    from blueprint_pipeline.job_retry_policy import request_cannot_succeed

    assert request_cannot_succeed("meta_sam_http_413") and request_cannot_succeed("meta_sam_http_400")
    assert request_cannot_succeed("meta_sam_inline_clip_exceeds_request_ceiling")
    assert not request_cannot_succeed("meta_sam_http_429") and not request_cannot_succeed("meta_sam_http_503")
    assert not request_cannot_succeed("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted")
    assert not request_cannot_succeed("atlas_background_visual_review_rejected") and not request_cannot_succeed(None)


def test_try_again_answers_are_outages_and_never_end_a_step_at_once():
    from blueprint_pipeline.job_retry_policy import provider_says_retry_later, request_cannot_succeed

    for status in (408, 409, 425, 429, 500, 502, 503, 504):
        assert provider_says_retry_later(f"meta_sam_http_{status}")
        assert not request_cannot_succeed(f"meta_sam_http_{status}")
    for status in (400, 404, 413, 422):
        assert not provider_says_retry_later(f"meta_sam_http_{status}")
        assert request_cannot_succeed(f"meta_sam_http_{status}")


def test_a_refusal_of_blueprints_own_key_or_account_is_an_outage_never_an_end():
    from blueprint_pipeline.job_retry_policy import provider_account_refused, request_cannot_succeed

    for status in (401, 402, 403):
        assert provider_account_refused(f"world_labs_http_{status}:invalid api key")
        assert not request_cannot_succeed(f"world_labs_http_{status}")
    assert not provider_account_refused("world_labs_http_400") and not provider_account_refused(None)
    # Blueprint's own WebApp answers are not an outside service's.
    assert not provider_account_refused("website_control_preparation-spend_http_403:website_scene_rights_required")
    assert request_cannot_succeed("meta_sam_submission_requires_reconciliation")


def _sdk_error(status, code, message="refused"):
    import httpx
    import openai
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    kind = {401: openai.AuthenticationError, 402: openai.APIStatusError, 403: openai.PermissionDeniedError,
            429: openai.RateLimitError, 500: openai.InternalServerError, 503: openai.InternalServerError}[status]
    return kind(f"Error code: {status}", response=response, body={"message": message, "type": code, "code": code})


@pytest.mark.parametrize(("error", "unbilled"), [
    (lambda: _sdk_error(429, "insufficient_quota", "You have no credits remaining."), True),
    (lambda: _sdk_error(402, "payment_required"), True),
    (lambda: _sdk_error(401, "invalid_api_key"), True),
    (lambda: _sdk_error(403, "unsupported_country_region_territory"), True),
    (lambda: _sdk_error(429, "enforced_spend_limit_reached"), True),
    (lambda: _sdk_error(429, "rate_limit_exceeded", "Rate limit reached for requests"), False),
    (lambda: _sdk_error(500, "server_error"), False),
    (lambda: _sdk_error(503, "server_error"), False),
    (lambda: __import__("openai").APITimeoutError(
        request=__import__("httpx").Request("POST", "https://api.openai.com/v1/responses")), False),
    (lambda: RuntimeError("worker stopped mid-call"), False),
])
def test_only_a_refusal_known_to_bill_nothing_is_unbilled(error, unbilled):
    from blueprint_pipeline.job_retry_policy import provider_error_waits, provider_refusal_unbilled
    value = error()
    assert provider_refusal_unbilled(value) is unbilled
    # Every refusal known to bill nothing is also an outage the step waits on.
    assert not unbilled or provider_error_waits(value)
