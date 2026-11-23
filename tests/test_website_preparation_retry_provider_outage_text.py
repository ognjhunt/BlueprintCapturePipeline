"""A provider's own 5xx or 429 text is an outage, retried in minutes, not a refusal retried in fifteen.

The strings are what the provider SDKs print for the error the job's step re-raises unchanged
(``website_gemini_receipts.retained_gemini_call`` re-raises the SDK's exception; ``_texts`` reads only
``str()`` of each link of the chain): google-genai 3.6.1 ``ServerError(500, body)`` and
``ServerError(504, body)``, and the OpenAI SDK's ``Error code: 500 - ...``.
"""
from __future__ import annotations

import pytest

from blueprint_pipeline.website_preparation_retry import OUTAGE, classify, retry_delay

PROVIDER_TRANSIENT_TEXTS = [
    "500 INTERNAL. {'error': {'code': 500, 'message': 'An internal error has occurred. Please retry or report in "
    "https://developers.generativeai.google/guide/troubleshooting', 'status': 'INTERNAL'}}",
    "504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'message': 'Deadline expired before operation could complete.', "
    "'status': 'DEADLINE_EXCEEDED'}}",
    "Error code: 500 - {'error': {'message': 'The server had an error while processing your request. Sorry about "
    "that!', 'type': 'server_error'}}",
    "Error code: 429 - {'error': {'message': 'Rate limit reached for requests', 'type': 'requests', "
    "'code': 'rate_limit_exceeded'}}",
    "502 Bad Gateway",
]


@pytest.mark.parametrize("text", PROVIDER_TRANSIENT_TEXTS)
def test_provider_transient_error_text_is_an_outage(text: str) -> None:
    assert classify(text) == OUTAGE
    assert retry_delay(classify(text), 1) == 120


# --- the error's own status, read before its text ------------------------------------------------

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import httpx  # noqa: E402
import openai  # noqa: E402

from blueprint_pipeline import website_preparation_retry as retry  # noqa: E402
from blueprint_pipeline.deterministic_refusal import DeterministicRefusal  # noqa: E402
from blueprint_pipeline.website_preparation_terminal import GRANT_FILE  # noqa: E402

FAILED_AT = "2026-11-23T12:00:00Z"


class GenaiError(Exception):
    """``google.genai.errors.APIError``'s shape: an int ``code``, a string ``status``, its own text."""

    def __init__(self, code: int, status: str, message: str, response=None) -> None:
        self.code, self.status, self.message, self.response = code, status, message, response
        super().__init__(message)


def _openai(cls, status: int, headers: dict | None = None):
    response = httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "https://provider.test"))
    return cls("provider said no", response=response, body=None)


def _retry(tmp_path: Path, exc: BaseException, *, grant: bool = True) -> dict | None:
    root = tmp_path / "capture"
    root.mkdir(parents=True)
    (root / retry.PAYLOAD_FILE).write_text(json.dumps({"source_payload_sha256": "d", "producer_delivery_key": "k"}))
    if grant:
        (root / GRANT_FILE).parent.mkdir(parents=True)
        (root / GRANT_FILE).write_text("{}")
    record = retry.failure_retry(exc=exc, capture_root=root, payload_digest="d", producer_delivery_key="k",
                                 history=[{"status": "failed_retryable"}], failed_at=FAILED_AT, owned_elsewhere=False)
    return record


def _gap(record: dict) -> float:
    from blueprint_pipeline.website_preparation_terminal import _parse_utc
    return (_parse_utc(record["next_attempt_at"]) - _parse_utc(FAILED_AT)).total_seconds()


@pytest.mark.parametrize("exc", [
    GenaiError(500, "INTERNAL", "the model failed"),
    GenaiError(504, "DEADLINE_EXCEEDED", "too slow"),
    RuntimeError("clean_plate step stopped"),  # a wrapper whose cause is the provider's 502
])
def test_an_error_carrying_a_provider_5xx_status_is_an_outage(tmp_path: Path, exc: BaseException) -> None:
    if isinstance(exc, RuntimeError):
        exc.__cause__ = _openai(openai.InternalServerError, 502)
    record = _retry(tmp_path, exc)
    assert record["kind"] == OUTAGE
    assert _gap(record) == 120


@pytest.mark.parametrize("exc", [
    GenaiError(400, "INVALID_ARGUMENT", "Request contains an invalid argument."),
    _openai(openai.BadRequestError, 400),
    _openai(openai.UnprocessableEntityError, 422),
    _openai(openai.NotFoundError, 404),
])
def test_a_provider_refusing_the_request_itself_stays_a_refusal(tmp_path: Path, exc: BaseException) -> None:
    record = _retry(tmp_path, exc)
    assert record["kind"] == retry.REFUSAL
    assert _gap(record) == 900
    assert classify("400 INVALID_ARGUMENT. {'error': {'code': 400, 'status': 'INVALID_ARGUMENT'}}") == retry.REFUSAL
    assert classify("Error code: 400 - {'error': {'type': 'invalid_request_error'}}") == retry.REFUSAL


def test_a_hard_stop_or_a_deterministic_refusal_still_wins_over_a_5xx_status(tmp_path: Path) -> None:
    rights = ValueError("capture_rights_not_granted")
    rights.__cause__ = GenaiError(500, "INTERNAL", "x")
    assert _retry(tmp_path / "a", rights) is None  # a privacy or rights stop is never retried on a timer
    money = ValueError("website_scene_preparation_budget_exhausted")
    money.__cause__ = GenaiError(503, "UNAVAILABLE", "x")
    assert _retry(tmp_path / "b", money)["kind"] == "money"
    fixed = DeterministicRefusal("atlas_x", step="atlas_proposal_review", inputs={})
    fixed.__cause__ = GenaiError(500, "INTERNAL", "x")
    assert _retry(tmp_path / "c", fixed)["kind"] == retry.DETERMINISTIC


def test_a_429_is_not_retried_sooner_than_its_retry_after(tmp_path: Path) -> None:
    record = _retry(tmp_path / "a", _openai(openai.RateLimitError, 429, {"retry-after": "600"}))
    assert record["kind"] == OUTAGE and _gap(record) == 600
    # A shorter Retry-After never shortens the schedule.
    assert _gap(_retry(tmp_path / "b", _openai(openai.RateLimitError, 429, {"retry-after": "5"}))) == 120
    dated = _openai(openai.RateLimitError, 429, {"retry-after": "Mon, 23 Nov 2026 12:30:00 GMT"})
    assert _gap(_retry(tmp_path / "c", dated)) == 1800
