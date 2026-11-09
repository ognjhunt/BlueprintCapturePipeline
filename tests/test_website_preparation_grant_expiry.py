# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_preparation_terminal.py
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/pubsub_handoff_listener.py
"""Beta exit E1: a silent preparation whose grant expired ends by itself.

The real birth, lease, ledger, terminal, status and sweep code run here; only the
owner/context transport and the WebApp callback are local fakes. Nothing is
retried, bought or renewed.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_preparation_status as status_reader
from blueprint_pipeline import website_preparation_terminal as terminal
from blueprint_pipeline import website_task_context as context_reader
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_website_preparation_terminal_status import (
    LONG_AGO, WINDOW, _acceptance, _claim, _finish, _listening_since, _read,
)

pytest_plugins = ["tests.test_website_preparation_terminal_status"]  # the website_capture fixture

PENDING = "website_control_scene-sponsorship_http_409:website_assessment_preparation_pending"


def _grant(source, *, expires_at):
    grant = {"schema_version": "website_scene_sponsorship.v1", "sponsor": "blueprint",
             **{key: source.owner[key] for key in ("request_id", "scene_id", "capture_id")},
             "expires_at_epoch": expires_at.timestamp()}
    grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    write_json(source.root / "pipeline/website_scene_sponsorship.json", grant)  # website_task_evidence


def _silent(source, monkeypatch, *, grant_expires, **update):
    """A retryable attempt stopped a full window ago while the listener kept pulling."""
    now = datetime.now(timezone.utc)
    ledger = _finish(source, _claim(source), updated_at=(now - WINDOW - timedelta(minutes=5)).isoformat(),
                     **(update or {"status": "failed_retryable", "last_error": "website_reconstruction_pending"}))
    _listening_since(source, now - WINDOW - timedelta(minutes=10), last_tick=now - timedelta(minutes=1))
    _grant(source, expires_at=now + grant_expires)
    monkeypatch.setattr(context_reader, "website_webapp_request",
                        lambda **_: _acceptance(source, status_reader.STATUS_V2))
    return ledger


def _ledger(source):
    return listener._read_job_ledger(source.root)


def test_expired_grant_and_silence_end_the_job_once_and_the_site_is_told(website_capture, monkeypatch):
    stopped = _silent(website_capture, monkeypatch, grant_expires=-timedelta(minutes=1))
    status_reader.reconcile_preparation_wakeups(website_capture.storage_root)
    ended = _ledger(website_capture)
    assert ended["status"] == "terminal_authority_ended" and ended["terminal_code"] == "consent_expired"
    assert ended["terminal_operation"] == terminal.EXPIRY_OPERATION
    assert ended["terminal_payload_sha256"] == stopped["source_payload_sha256"]
    assert ended["producer_delivery_key"] == stopped["producer_delivery_key"]
    assert ended["lease_token"] is None and ended["revision"] == stopped["revision"] + 2  # claim + ending
    assert terminal.OVERDUE_FIELD not in ended
    assert [item["status"] for item in ended["attempt_history"]][-1] == "terminal_authority_ended"
    receipt = json.loads((website_capture.root / listener.JOB_TERMINAL_RECEIPT_FILENAME).read_text())
    assert receipt["status"] == "authority_ended" and listener._terminal_receipt_current(receipt, ended)
    assert _read(website_capture)["state"] == "authority_ended"
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "authority_ended"
    delivered = json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())
    assert delivered["state"] == "delivered" and delivered["acceptance"]["state"] == "authority_ended"
    # The next sweep writes nothing: the ending is recorded once.
    status_reader.reconcile_preparation_wakeups(website_capture.storage_root)
    assert _ledger(website_capture) == ended
    assert terminal.end_expired_preparation(website_capture.root, alive_since=LONG_AGO) is None
    # A redelivery of the same message is answered from the ending, never run again.
    state, _ = listener._claim_job_lease(website_capture.root, scene_id=ended["scene_id"],
        capture_id=ended["capture_id"], owner="redelivery", lease_seconds=60,
        payload_sha256=ended["source_payload_sha256"], producer_delivery_key=ended["producer_delivery_key"])
    assert state == "terminal"


def test_valid_grant_and_silence_record_the_overdue_observation_only(website_capture, monkeypatch):
    stopped = _silent(website_capture, monkeypatch, grant_expires=timedelta(days=1))
    status_reader.reconcile_preparation_wakeups(website_capture.storage_root)
    ledger = _ledger(website_capture)
    assert ledger["status"] == "failed_retryable" and ledger["revision"] == stopped["revision"] + 1
    assert ledger[terminal.OVERDUE_FIELD]["reason"] == "recovery_needed"
    assert "terminal_code" not in ledger
    assert not (website_capture.root / listener.JOB_TERMINAL_RECEIPT_FILENAME).exists()
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "failed_retryable"


@pytest.mark.parametrize("grant_expires", [timedelta(days=1), -timedelta(minutes=1)])
def test_a_running_preparation_is_untouched(website_capture, monkeypatch, grant_expires):
    """A live lease is the running job's own: its next WebApp read decides, not the sweep."""
    now = datetime.now(timezone.utc)
    running = _claim(website_capture, lease_seconds=900)
    _listening_since(website_capture, now - 3 * WINDOW, last_tick=now - timedelta(minutes=1))
    _grant(website_capture, expires_at=now + grant_expires)
    monkeypatch.setattr(context_reader, "website_webapp_request",
                        lambda **_: _acceptance(website_capture, status_reader.STATUS_V2))
    status_reader.reconcile_preparation_wakeups(website_capture.storage_root)
    assert _ledger(website_capture) == running
    assert terminal.end_expired_preparation(website_capture.root, alive_since=LONG_AGO,
                                            now=now + 2 * WINDOW) is None
    assert _ledger(website_capture) == running


def test_the_window_and_a_sealed_grant_for_this_capture_are_both_required(website_capture, monkeypatch):
    stopped = _silent(website_capture, monkeypatch, grant_expires=-timedelta(minutes=1))
    now = datetime.now(timezone.utc)
    # Inside the silence window a redelivery may still come: nothing ends.
    assert terminal.end_expired_preparation(website_capture.root, alive_since=now) is None
    grant = json.loads((website_capture.root / terminal.GRANT_FILE).read_text())
    for forged in ({**grant, "capture_id": "walkthrough-other"}, {**grant, "expires_at_epoch": grant["expires_at_epoch"] - 1}):
        write_json(website_capture.root / terminal.GRANT_FILE, forged)
        assert terminal.end_expired_preparation(website_capture.root, alive_since=LONG_AGO) is None
    (website_capture.root / terminal.GRANT_FILE).unlink()
    assert terminal.end_expired_preparation(website_capture.root, alive_since=LONG_AGO) is None
    assert _ledger(website_capture) == stopped


def test_a_claim_of_a_changed_revision_is_refused(website_capture):
    ledger = _claim(website_capture)
    state, current = listener._claim_job_lease(website_capture.root, scene_id=ledger["scene_id"],
        capture_id=ledger["capture_id"], owner="other", lease_seconds=60, expected_revision=ledger["revision"] - 1)
    assert state == "revision_changed" and current == ledger == _ledger(website_capture)


# Every non-terminal ledger state a silent preparation can sit in, and whether the
# sweep ends it at grant expiry. "processing" with a live lease is the running job
# (it reads the WebApp itself); "completed" is a final ledger state.
SILENT_STATES = {
    "failed_retryable": {"status": "failed_retryable", "last_error": "website_scene_assembly_coverage_key_missing"},
    "retryable_blocked": {"status": "retryable_blocked", "retry_blockers": ["website_reconstruction_pending"]},
    "capacity_delayed": {"status": "retryable_blocked", "retry_blockers": ["pubsub_handoff_staging_capacity_insufficient"],
                         "last_error": "pubsub_handoff_staging_capacity_insufficient"},
}


def test_every_non_terminal_ledger_state_is_listed():
    assert set(status_reader._LEDGER_STATES) - {"terminal_authority_ended"} == {
        "processing", "failed_retryable", "retryable_blocked"}


@pytest.mark.parametrize("name", sorted(SILENT_STATES))
def test_each_silent_state_ends_at_grant_expiry(website_capture, monkeypatch, name):
    _silent(website_capture, monkeypatch, grant_expires=-timedelta(minutes=1), **SILENT_STATES[name])
    assert terminal.end_expired_preparation(website_capture.root, alive_since=LONG_AGO)["status"] == (
        "terminal_authority_ended")


def test_a_dead_workers_lease_ends_at_grant_expiry(website_capture):
    started = datetime.now(timezone.utc) - 2 * WINDOW
    _claim(website_capture, lease_seconds=60, now=started)
    _grant(website_capture, expires_at=started)
    ended = terminal.end_expired_preparation(website_capture.root, alive_since=LONG_AGO)
    assert ended["status"] == "terminal_authority_ended" and ended["attempt_count"] == 2


def test_the_assessment_wait_has_no_grant_and_is_not_ended_here(website_capture, monkeypatch):
    """Resumed by reconcile_waiting_assessments, not by Pub/Sub (reported, not changed here)."""
    stopped = _silent(website_capture, monkeypatch, grant_expires=-timedelta(minutes=1), status="failed_retryable",
                      last_error=PENDING, assessment_resume={"schema_version": "website_assessment_resume.v1"})
    assert terminal.end_expired_preparation(website_capture.root, alive_since=LONG_AGO) is None
    assert _ledger(website_capture) == stopped
