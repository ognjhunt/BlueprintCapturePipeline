# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_preparation_terminal.py
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/website_task_context.py
#   src/blueprint_pipeline/pubsub_handoff_listener.py
"""Beta exit E1: a silent preparation whose grant the WebApp refuses as expired ends by itself.

The real birth, lease, ledger, terminal, status and sweep code run here; only the
owner/context transport and the WebApp are local fakes. The WebApp alone decides
that a grant ended; the retained grant file only triggers the question. Nothing
is retried, bought or renewed.
"""
import json
from datetime import datetime, timedelta, timezone
from urllib.error import URLError

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

EXPIRED = "website_control_scene-sponsorship_http_409:consent_expired"
PENDING = "website_control_scene-sponsorship_http_409:website_assessment_preparation_pending"


def _grant(source, *, expires_at, **change):
    grant = {"schema_version": "website_scene_sponsorship.v1", "sponsor": "blueprint",
             **{key: source.owner[key] for key in ("request_id", "scene_id", "capture_id")},
             "expires_at_epoch": expires_at.timestamp(), **change}
    grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    write_json(source.root / "pipeline/website_scene_sponsorship.json", grant)  # website_task_evidence
    return grant


class WebApp:
    """The signed transport: scene-sponsorship answers per `answer`; status reads are accepted."""

    def __init__(self, source, answer):
        self.source, self.answer, self.reads = source, answer, []

    def __call__(self, *, capture_id, operation, payload, **_):
        if operation != "scene-sponsorship":
            return _acceptance(self.source, status_reader.STATUS_V2)
        self.reads.append(dict(payload))
        if isinstance(self.answer, BaseException):
            raise self.answer
        return dict(self.answer)


def _silent(source, monkeypatch, *, grant_expires, answer, **update):
    """A retryable attempt stopped a full window ago while the listener kept pulling."""
    now = datetime.now(timezone.utc)
    ledger = _finish(source, _claim(source), updated_at=(now - WINDOW - timedelta(minutes=5)).isoformat(),
                     **(update or {"status": "failed_retryable", "last_error": "website_reconstruction_pending"}))
    _listening_since(source, now - WINDOW - timedelta(minutes=10), last_tick=now - timedelta(minutes=1))
    _grant(source, expires_at=now + grant_expires)
    webapp = WebApp(source, answer)
    monkeypatch.setattr(context_reader, "website_webapp_request", webapp)
    return ledger, webapp


def _ledger(source):
    return listener._read_job_ledger(source.root)


def _end(source):
    return terminal.end_expired_preparation(source.root, alive_since=LONG_AGO, listener=listener)


def test_webapp_refusal_ends_the_job_once_and_the_site_is_told(website_capture, monkeypatch):
    stopped, webapp = _silent(website_capture, monkeypatch, grant_expires=-timedelta(minutes=1),
                              answer=ValueError(EXPIRED))
    status_reader.reconcile_preparation_wakeups(website_capture.storage_root, listener=listener)
    # The read is the non-creating, window-honouring one, bound to the retained context.
    assert webapp.reads == [{"request_id": website_capture.owner["request_id"],
                             "scene_id": website_capture.owner["scene_id"], "purpose": "scene_preparation",
                             "expected_task_context_digest": website_capture.context["context_digest"],
                             "create": False}]
    ended = _ledger(website_capture)
    assert ended["status"] == "terminal_authority_ended" and ended["terminal_code"] == "consent_expired"
    assert ended["terminal_operation"] == "scene-sponsorship" and ended["last_error"] == EXPIRED
    assert ended["terminal_payload_sha256"] == stopped["source_payload_sha256"]
    assert ended["producer_delivery_key"] == stopped["producer_delivery_key"]
    assert ended["lease_token"] is None and ended["revision"] == stopped["revision"] + 2  # claim + ending
    assert terminal.OVERDUE_FIELD not in ended
    assert ended["attempt_history"][-1]["status"] == "terminal_authority_ended"
    receipt = json.loads((website_capture.root / listener.JOB_TERMINAL_RECEIPT_FILENAME).read_text())
    assert receipt["status"] == "authority_ended" and listener._terminal_receipt_current(receipt, ended)
    assert _read(website_capture)["state"] == "authority_ended"
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "authority_ended"
    delivered = json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())
    assert delivered["state"] == "delivered" and delivered["acceptance"]["state"] == "authority_ended"
    # The next sweep neither asks again nor writes: the ending is recorded once.
    status_reader.reconcile_preparation_wakeups(website_capture.storage_root, listener=listener)
    assert _ledger(website_capture) == ended and len(webapp.reads) == 1
    assert _end(website_capture) is None and len(webapp.reads) == 1
    # A redelivery of the same message is answered from the ending, never run again.
    state, _ = listener._claim_job_lease(website_capture.root, scene_id=ended["scene_id"],
        capture_id=ended["capture_id"], owner="redelivery", lease_seconds=60,
        payload_sha256=ended["source_payload_sha256"], producer_delivery_key=ended["producer_delivery_key"])
    assert state == "terminal"


@pytest.mark.parametrize("answer", [
    # Extended: the raw expiry passed, but the owner's window keeps the grant valid.
    {"schema_version": "website_scene_sponsorship.v1", "expires_at_epoch": 1.0, "window": "extended"},
    # Renewed or replaced on the WebApp while the local file is stale.
    {"schema_version": "website_scene_sponsorship.v1", "renewal": {"parent_authority_digest": "sha256:" + "a" * 64}},
    URLError("unreachable"), TimeoutError("timed out"), OSError("reset"),
    ValueError("website_control_scene-sponsorship_http_409:website_task_context_changed"),
    ValueError("website_control_scene-sponsorship_http_429:pipeline_sync_rate_limited"),
    ValueError("website_control_scene-sponsorship_http_409:source_revoked"),
    ValueError(EXPIRED + "_later"),
], ids=["extended", "renewed", "unreachable", "timeout", "reset", "context_changed", "rate_limited",
        "other_refusal", "not_the_whole_token"])
def test_only_the_webapp_refusal_ends_a_job(website_capture, monkeypatch, answer):
    stopped, webapp = _silent(website_capture, monkeypatch, grant_expires=-timedelta(minutes=1), answer=answer)
    assert _end(website_capture) is None and len(webapp.reads) == 1
    assert _ledger(website_capture) == stopped
    # The sweep still records the overdue observation, as today.
    status_reader.reconcile_preparation_wakeups(website_capture.storage_root, listener=listener)
    ledger = _ledger(website_capture)
    assert ledger["status"] == "failed_retryable" and ledger[terminal.OVERDUE_FIELD]["reason"] == "recovery_needed"
    assert "terminal_code" not in ledger
    assert not (website_capture.root / listener.JOB_TERMINAL_RECEIPT_FILENAME).exists()
    assert len(webapp.reads) == 2  # At most one question per sweep.


def test_valid_grant_and_silence_record_the_overdue_observation_only(website_capture, monkeypatch):
    stopped, webapp = _silent(website_capture, monkeypatch, grant_expires=timedelta(days=1),
                              answer=ValueError(EXPIRED))
    status_reader.reconcile_preparation_wakeups(website_capture.storage_root, listener=listener)
    ledger = _ledger(website_capture)
    assert webapp.reads == []  # A grant whose own expiry is ahead is valid: nothing is asked.
    assert ledger["status"] == "failed_retryable" and ledger["revision"] == stopped["revision"] + 1
    assert ledger[terminal.OVERDUE_FIELD]["reason"] == "recovery_needed"
    assert "terminal_code" not in ledger
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "failed_retryable"


@pytest.mark.parametrize("grant_expires", [timedelta(days=1), -timedelta(minutes=1)])
def test_a_running_preparation_is_untouched(website_capture, monkeypatch, grant_expires):
    """A live lease is the running job's own: its next WebApp read decides, not the sweep."""
    now = datetime.now(timezone.utc)
    running = _claim(website_capture, lease_seconds=900)
    _listening_since(website_capture, now - 3 * WINDOW, last_tick=now - timedelta(minutes=1))
    _grant(website_capture, expires_at=now + grant_expires)
    webapp = WebApp(website_capture, ValueError(EXPIRED))
    monkeypatch.setattr(context_reader, "website_webapp_request", webapp)
    status_reader.reconcile_preparation_wakeups(website_capture.storage_root, listener=listener)
    assert _end(website_capture) is None
    assert _ledger(website_capture) == running and webapp.reads == []


@pytest.mark.parametrize("damage", ["other_capture", "other_scene", "unsealed", "malformed", "not_object",
                                    "string_expiry", "missing", "symlink"])
def test_a_grant_file_not_for_this_capture_never_triggers(website_capture, monkeypatch, tmp_path, damage):
    stopped, webapp = _silent(website_capture, monkeypatch, grant_expires=-timedelta(minutes=1),
                              answer=ValueError(EXPIRED))
    path = website_capture.root / "pipeline/website_scene_sponsorship.json"
    grant = json.loads(path.read_text())
    if damage == "other_capture":
        _grant(website_capture, expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
               capture_id="walkthrough-other")
    elif damage == "other_scene":
        _grant(website_capture, expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
               scene_id="site-other")
    elif damage == "unsealed":
        write_json(path, {**grant, "expires_at_epoch": grant["expires_at_epoch"] - 1})
    elif damage == "malformed":
        path.write_text("{not json")
    elif damage == "not_object":
        path.write_text("[]")
    elif damage == "string_expiry":
        _grant(website_capture, expires_at=datetime.now(timezone.utc), expires_at_epoch="0")
    elif damage == "missing":
        path.unlink()
    else:
        outside = tmp_path / "grant.json"
        outside.write_text(json.dumps(grant))
        path.unlink()
        path.symlink_to(outside)
    assert _end(website_capture) is None
    assert webapp.reads == [] and _ledger(website_capture) == stopped


def test_inside_the_silence_window_nothing_is_asked(website_capture, monkeypatch):
    stopped, webapp = _silent(website_capture, monkeypatch, grant_expires=-timedelta(minutes=1),
                              answer=ValueError(EXPIRED))
    assert terminal.end_expired_preparation(website_capture.root, alive_since=datetime.now(timezone.utc),
                                            listener=listener) is None
    assert webapp.reads == [] and _ledger(website_capture) == stopped


def test_a_ledger_that_moved_while_the_webapp_answered_is_not_ended(website_capture, monkeypatch):
    stopped, webapp = _silent(website_capture, monkeypatch, grant_expires=-timedelta(minutes=1),
                              answer=ValueError(EXPIRED))

    def redelivered(*, capture_id, operation, payload, **_):
        if operation == "scene-sponsorship":
            _finish(website_capture, _claim(website_capture), status="failed_retryable", last_error="again")
        return webapp(capture_id=capture_id, operation=operation, payload=payload)

    monkeypatch.setattr(context_reader, "website_webapp_request", redelivered)
    assert _end(website_capture) is None
    assert _ledger(website_capture)["status"] == "failed_retryable"
    assert _ledger(website_capture)["revision"] == stopped["revision"] + 2


def test_a_claim_of_a_changed_revision_is_refused(website_capture):
    ledger = _claim(website_capture)
    state, current = listener._claim_job_lease(website_capture.root, scene_id=ledger["scene_id"],
        capture_id=ledger["capture_id"], owner="other", lease_seconds=60, expected_revision=ledger["revision"] - 1)
    assert state == "revision_changed" and current == ledger == _ledger(website_capture)


# Every non-terminal ledger state a silent preparation can sit in, and whether the
# sweep ends it on the WebApp's refusal. "processing" with a live lease is the
# running job (it reads the WebApp itself); "completed" is a final ledger state.
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
def test_each_silent_state_ends_on_the_webapp_refusal(website_capture, monkeypatch, name):
    _silent(website_capture, monkeypatch, grant_expires=-timedelta(minutes=1), answer=ValueError(EXPIRED),
            **SILENT_STATES[name])
    assert _end(website_capture)["status"] == "terminal_authority_ended"


def test_a_dead_workers_lease_ends_on_the_webapp_refusal(website_capture, monkeypatch):
    started = datetime.now(timezone.utc) - 2 * WINDOW
    _claim(website_capture, lease_seconds=60, now=started)
    _grant(website_capture, expires_at=started)
    monkeypatch.setattr(context_reader, "website_webapp_request", WebApp(website_capture, ValueError(EXPIRED)))
    ended = _end(website_capture)
    assert ended["status"] == "terminal_authority_ended" and ended["attempt_count"] == 2


def test_the_assessment_wait_is_never_asked_about(website_capture, monkeypatch):
    """Resumed by reconcile_waiting_assessments, not by Pub/Sub (reported, not changed here)."""
    stopped, webapp = _silent(website_capture, monkeypatch, grant_expires=-timedelta(minutes=1),
                              answer=ValueError(EXPIRED), status="failed_retryable", last_error=PENDING,
                              assessment_resume={"schema_version": "website_assessment_resume.v1"})
    assert _end(website_capture) is None
    assert webapp.reads == [] and _ledger(website_capture) == stopped


def test_without_the_running_listener_the_sweep_only_observes(website_capture, monkeypatch):
    """The listener tick passes its own module in; any other caller keeps the overdue-only sweep."""
    stopped, webapp = _silent(website_capture, monkeypatch, grant_expires=-timedelta(minutes=1),
                              answer=ValueError(EXPIRED))
    status_reader.reconcile_preparation_wakeups(website_capture.storage_root)
    assert webapp.reads == [] and _ledger(website_capture)["status"] == "failed_retryable"
    assert terminal.OVERDUE_FIELD in _ledger(website_capture)
