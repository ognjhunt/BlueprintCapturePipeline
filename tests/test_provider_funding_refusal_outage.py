"""A provider's "no credit" answer, and any 429 or 5xx its SDK raises, waits as an outage in both website classifiers.

The error is the one the room build met at stage ``source_pose`` on 2026-11-23 (host journal 17:47:27Z): the OpenAI
SDK's ``RateLimitError`` for a 429 whose body says ``insufficient_quota``. The room build's classifier
(``website_atlas_preview._outage``) read it as an unclassified refusal: 15 min, 1 h, 3 h ... and then the job ended.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import openai
import pytest

from blueprint_pipeline import preparation_fault_alert as alert
from blueprint_pipeline import website_atlas_preview as preview
from blueprint_pipeline import website_preparation_retry as retry
from blueprint_pipeline.job_retry_policy import provider_error_waits
from blueprint_pipeline.website_preparation_retry_record import FIELD, SCHEMA
from blueprint_pipeline.website_preparation_terminal import GRANT_FILE

NO_CREDIT_BODY = {"error": {"message": "You have no credits remaining. Add credits to continue using the API at "
                                       "https://platform.openai.com/settings/organization/billing/.",
                            "type": "insufficient_quota", "param": None, "code": "insufficient_quota"}}
NOW = 1_800_000_000


def _openai(cls, status: int, body: dict | None = None, headers: dict | None = None):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status, headers=headers or {}, request=request, json=body)
    message = f"Error code: {status} - {body}" if body else "provider said no"
    return cls(message, response=response, body=body["error"] if body else None)


def no_credit() -> openai.RateLimitError:
    return _openai(openai.RateLimitError, 429, NO_CREDIT_BODY)


def _job(**fields) -> dict:
    return {"capture_id": "walkthrough-x", "state": "processing", "stage": "source_pose",
            "enqueued_at_epoch": NOW - 3600, "transient_failures": 0, **fields}


# --- one rule, both classifiers ------------------------------------------------------------------

@pytest.mark.parametrize("exc", [
    no_credit(),
    _openai(openai.RateLimitError, 429),
    _openai(openai.InternalServerError, 500),
    _openai(openai.InternalServerError, 503),
    _openai(openai.AuthenticationError, 401),
    _openai(openai.PermissionDeniedError, 403),
])
def test_the_room_build_waits_on_a_provider_sdk_429_5xx_or_account_refusal_as_an_outage(exc) -> None:
    assert provider_error_waits(exc)
    after = preview._failed_step(_job(), exc, NOW)
    assert after["state"] == "processing"
    assert after["transient_failures"] == 1
    assert after["next_attempt_at_epoch"] == NOW + 120
    assert "refusal_retries" not in after and "refusal_retries_total" not in after


def test_the_no_credit_error_waits_in_the_preparation_retry_too(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    (root / retry.PAYLOAD_FILE).write_text(json.dumps({"source_payload_sha256": "d", "producer_delivery_key": "k"}))
    (root / GRANT_FILE).parent.mkdir(parents=True)
    (root / GRANT_FILE).write_text("{}")
    record = retry.failure_retry(exc=no_credit(), capture_root=root, payload_digest="d", producer_delivery_key="k",
                                 history=[{"status": "failed_retryable"}], failed_at="2026-11-23T17:47:27Z",
                                 owned_elsewhere=False)
    assert record["kind"] in (retry.OUTAGE, "money")  # both wait 2 min doubling to 1 h until the grant expires
    assert record["next_attempt_at"] == "2026-11-23T17:49:27Z"


@pytest.mark.parametrize("exc", [
    _openai(openai.BadRequestError, 400),
    _openai(openai.NotFoundError, 404),
    _openai(openai.UnprocessableEntityError, 422),
    PermissionError(13, "denied"),  # an OSError: an outage by its own branch, never by the provider rule
    RuntimeError("a defect"),
])
def test_the_provider_rule_leaves_other_answers_alone(exc) -> None:
    assert not provider_error_waits(exc)


def test_a_provider_refusing_the_request_itself_still_ends_on_its_own_schedule() -> None:
    after = preview._failed_step(_job(), _openai(openai.BadRequestError, 400), NOW)
    assert after.get("transient_failures") == 0 and after["refusal_retries"] == 1


# --- a job already holding refusal retries from this cause ---------------------------------------

def test_a_job_whose_refusal_retries_this_cause_consumed_gets_them_back() -> None:
    stuck = _job(refusal_retries=1, refusal_retries_total=2, last_refusal_reason="atlas_preview_ratelimiterror",
                 next_attempt_at_epoch=NOW)
    after = preview._failed_step(stuck, no_credit(), NOW)
    assert after["state"] == "processing" and after["transient_failures"] == 1
    assert after["refusal_retries"] == 0 and after["refusal_retries_total"] == 1
    assert "last_refusal_reason" not in after
    # A schedule one step from spent does not end the job either: the outage waits until the deadline.
    nearly = _job(refusal_retries=6, refusal_retries_total=11, last_refusal_reason="atlas_preview_ratelimiterror")
    for attempt in range(20):
        nearly = preview._failed_step(nearly, no_credit(), NOW + attempt)
        assert nearly["state"] == "processing"
    assert nearly["refusal_retries"] == 0 and nearly["refusal_retries_total"] == 5


def test_another_codes_refusal_streak_is_kept() -> None:
    other = _job(refusal_retries=2, refusal_retries_total=3, last_refusal_reason="atlas_preview_valueerror")
    after = preview._failed_step(other, no_credit(), NOW)
    assert after["refusal_retries"] == 2 and after["refusal_retries_total"] == 3


# --- the owner is paged ----------------------------------------------------------------------------

def _waiting_job(root: Path, name: str) -> None:
    path = root / "u" / "scenes" / "site-a" / "captures" / name / "pipeline" / "atlas" / "preview_job.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(preview._failed_step(_job(), no_credit(), NOW)))


def test_a_provider_with_no_credit_pages_the_owner_with_the_jobs_waiting(tmp_path: Path) -> None:
    _waiting_job(tmp_path, "walkthrough-a")
    _waiting_job(tmp_path, "walkthrough-b")
    memory: dict = {}
    _, alerts = alert.check_preparation_faults(tmp_path, now=NOW, remembered={}, memory_out=memory)
    page = [row for row in alerts if row["code"] == alert.KEY_CODE]
    assert len(page) == 1 and page[0]["target"] == f"provider:openai:{alert.KEY_CODE}"
    assert page[0]["label"] == "openai (2 jobs waiting)" and page[0]["blockers"] == ["insufficient_quota"]
    assert "no credit" in alert.describe(page[0])
    # First seen is remembered; the page is announced once, then left to the stall alarm's repeat.
    _, later = alert.check_preparation_faults(tmp_path, now=NOW + alert.ANNOUNCE_SECONDS, remembered=memory)
    assert not [row for row in later if row["code"] == alert.KEY_CODE]


# --- a provider's Retry-After holds in the job's last two hours ----------------------------------

class _Listener:
    def __init__(self, ledger: dict) -> None:
        self.ledger = ledger

    def _read_job_ledger(self, _root):
        return self.ledger


def test_the_last_call_waits_for_the_providers_retry_after(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    (root / retry.PAYLOAD_FILE).write_text(json.dumps({"source_payload_sha256": "d", "producer_delivery_key": "k"}))
    (root / GRANT_FILE).parent.mkdir(parents=True)
    (root / GRANT_FILE).write_text("{}")
    limited = _openai(openai.RateLimitError, 429, headers={"retry-after": "10800"})
    record = retry.failure_retry(exc=limited, capture_root=root, payload_digest="d", producer_delivery_key="k",
                                 history=[{"status": "failed_retryable"}], failed_at="2026-11-23T12:00:00Z",
                                 owned_elsewhere=False)
    assert record["provider_not_before"] == "2026-11-23T15:00:00Z"
    ledger = {"status": "failed_retryable", "last_error": record["error"], "source_payload_sha256": "d",
              "producer_delivery_key": "k", FIELD: {**record, "schema_version": SCHEMA}}
    monkeypatch.setattr(retry, "last_call_due", lambda *_: True)  # the window's last two hours opened at 13:00

    def ran(*_a, **_k):
        raise RuntimeError("redriven")
    monkeypatch.setattr(retry, "retained_payload", ran)
    early = datetime(2026, 11, 23, 14, 0, tzinfo=timezone.utc)
    assert retry._redrive(_Listener(ledger), root, storage_root=tmp_path, process_args={}, now=early) is None
    with pytest.raises(RuntimeError, match="redriven"):
        retry._redrive(_Listener(ledger), root, storage_root=tmp_path, process_args={},
                       now=early + timedelta(hours=1))
