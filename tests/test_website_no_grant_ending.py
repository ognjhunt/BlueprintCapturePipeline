# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_task_context.py
#   src/blueprint_pipeline/website_assessment_resume.py
#   src/blueprint_pipeline/website_preparation_terminal.py
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/website_task_evidence.py
#   src/blueprint_pipeline/selected_handoff_recovery.py
#   src/blueprint_pipeline/pubsub_handoff_listener.py
#   src/blueprint_pipeline/website_preparation_retry.py
"""The WebApp ended a website job that never got a grant: the host ends its preparation too.

Before a capture's first grant the host asks the WebApp's non-creating read; only the exact
typed ``website_scene_job_no_grant_ended`` ends the job, through the listener's one terminal
path, and no creating request is sent. Anything else (the plain missing grant, an outage, a
malformed answer) leaves the flow exactly as before. The silence sweep asks the same question of
a silent ledger with no retained grant, at most once per silence window. The real listener,
claim, ledger, retry tick and sweep run here; run_e2e and the WebApp are fakes.
"""
import json
from datetime import datetime, timedelta, timezone
from urllib.error import URLError

import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_atlas_preview_store as preview_store
from blueprint_pipeline import website_preparation_retry as retry
from blueprint_pipeline import website_preparation_status as status_reader
from blueprint_pipeline import website_preparation_terminal as terminal
from blueprint_pipeline import website_task_context as context_reader
from blueprint_pipeline.common import PipelineError, write_json
from tests.test_website_preparation_retry import _due, _tick, failed  # noqa: F401
from tests.test_website_preparation_status import prepared_source  # noqa: F401
from tests.test_website_preparation_terminal_status import (
    LONG_AGO, WINDOW, _acceptance, _claim, _finish, _listening_since, _read,
)
from tests.test_website_reconstruction_resume import waiting  # noqa: F401

pytest_plugins = ["tests.test_website_preparation_terminal_status"]  # the website_capture fixture

NO_GRANT = "website_control_scene-sponsorship_http_409:website_scene_job_no_grant_ended"
MISSING = "website_control_scene-sponsorship_http_409:website_scene_sponsorship_missing"
EXPIRED = "website_control_scene-sponsorship_http_409:consent_expired"


@pytest.fixture
def grantless(failed):  # noqa: F811 - imported pytest fixture
    """The retry fixture's failed delivery, for a capture that never received a grant (none retained)."""
    (failed[0] / terminal.GRANT_FILE).unlink()
    return failed


class WebApp:
    """The signed transport: the non-creating read answers ``answer``; the creating request is counted."""

    def __init__(self, answer, grant):
        self.answer, self.grant, self.reads, self.creates, self.other = answer, grant, [], [], []

    def request(self, *, capture_id, operation, payload, **_):
        if operation != "scene-sponsorship":
            self.other.append(operation)
            raise ValueError("website_control_" + operation + "_http_503:unavailable")
        assert payload.get("create") is False  # The creating request goes through load_website_scene_sponsorship.
        self.reads.append(dict(payload))
        if isinstance(self.answer, BaseException):
            raise self.answer
        return dict(self.answer)

    def create(self, **kwargs):
        assert kwargs.get("create", True) is True
        self.creates.append(kwargs)
        if isinstance(self.grant, BaseException):
            raise self.grant
        return self.grant


def _install(monkeypatch, answer, grant=None):
    if grant is None:  # The grant the fixture's WebApp issues on the creating request.
        grant = context_reader.load_website_scene_sponsorship(task_context={}, now=0)
    webapp = WebApp(answer, grant)
    monkeypatch.setattr(context_reader, "website_webapp_request", webapp.request)
    monkeypatch.setattr(context_reader, "load_website_scene_sponsorship", webapp.create)
    return webapp


def _ran(calls):
    def run_e2e(**kwargs):
        calls.append(kwargs)
        return {"status": "completed"}
    return run_e2e


def _process(root, raw, run_e2e, **kwargs):
    return listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai", run_e2e=run_e2e,
                                            **kwargs)


def _ended_as_no_grant(root, stage="website_assessment_preparation"):
    ledger = listener._read_job_ledger(root)
    assert ledger["status"] == listener.TERMINAL_AUTHORITY_STATUS
    assert ledger["terminal_code"] == context_reader.NO_GRANT_ENDED_CODE
    assert ledger["terminal_operation"] == "scene-sponsorship" and ledger["last_error"] == NO_GRANT
    assert ledger["attempt_history"][-1]["stage"] == stage and ledger["lease_token"] is None
    receipt = json.loads((root / listener.JOB_TERMINAL_RECEIPT_FILENAME).read_text())
    assert receipt["status"] == "authority_ended" and listener._terminal_receipt_current(receipt, ledger)
    return ledger


def test_the_typed_answer_ends_the_job_and_no_first_grant_is_asked_for(grantless, monkeypatch):  # noqa: F811
    root, raw, context = grantless
    webapp = _install(monkeypatch, ValueError(NO_GRANT))
    result = _process(root, raw, lambda **_: pytest.fail("an ended job ran"))
    assert result["status"] == listener.TERMINAL_AUTHORITY_STATUS
    ledger = _ended_as_no_grant(root)
    # One non-creating read, bound to the current context; no creating request.
    assert webapp.reads == [{"request_id": context["request_id"], "scene_id": context["scene_id"],
                             "purpose": "scene_preparation", "expected_task_context_digest": context["context_digest"],
                             "create": False}]
    assert webapp.creates == []
    # Never retried by the host's own retry tick, however late.
    assert retry.armed(ledger) is None
    later = datetime.now(timezone.utc) + timedelta(days=30)
    assert _tick(root, lambda **_: pytest.fail("retried an ended job"), later)["redriven"] == 0
    assert listener._read_job_ledger(root) == ledger and webapp.creates == []


@pytest.mark.parametrize("answer", [
    ValueError(MISSING),
    ValueError("website_control_scene-sponsorship_http_503:website_scene_sponsorship_unavailable"),
    ValueError("website_control_scene-sponsorship_http_409:website_task_context_changed"),
    ValueError("website_control_scene-sponsorship_http_503:website_scene_job_no_grant_ended"),
    ValueError(NO_GRANT + "_later"),
    ValueError("Expecting value: line 1 column 1 (char 0)"),
    ValueError("website_task_context_response_invalid"),
    TimeoutError("timed out"), URLError("unreachable"), OSError("reset"),
    {"schema_version": "website_scene_sponsorship.v1"},  # The job has a grant on the WebApp.
], ids=["missing", "http_503", "other_refusal", "wrong_status", "not_the_whole_token", "malformed", "not_object",
        "timeout", "unreachable", "reset", "grant_returned"])
def test_any_other_answer_sends_the_creating_request_as_today(grantless, monkeypatch, answer):  # noqa: F811
    root, raw, _ = grantless
    webapp = _install(monkeypatch, answer)
    calls = []
    result = _process(root, raw, _ran(calls))
    assert result["status"] == "processed" and len(calls) == 1
    assert len(webapp.reads) == 1 and len(webapp.creates) == 1
    assert listener._read_job_ledger(root)["status"] == "completed"


def test_consent_expired_keeps_its_own_ending(grantless, monkeypatch):  # noqa: F811
    """The pre-grant read acts on the no-grant code only; the creating request meets consent_expired as before."""
    root, raw, _ = grantless
    webapp = _install(monkeypatch, ValueError(EXPIRED), grant=ValueError(EXPIRED))
    result = _process(root, raw, lambda **_: pytest.fail("ran"))
    assert result["status"] == listener.TERMINAL_AUTHORITY_STATUS
    assert len(webapp.reads) == 1 and len(webapp.creates) == 1
    assert listener._read_job_ledger(root)["terminal_code"] == "consent_expired"


def test_a_capture_that_retains_a_grant_is_never_asked(failed, monkeypatch):  # noqa: F811
    root, raw, _ = failed
    assert (root / terminal.GRANT_FILE).is_file()  # The fixture's delivery retained its grant.
    write_json(root / "pipeline/website_scene_sponsorship.json", {"schema_version": "website_scene_sponsorship.v1"})
    webapp = _install(monkeypatch, ValueError(NO_GRANT))
    calls = []
    assert _process(root, raw, _ran(calls))["status"] == "processed"
    assert webapp.reads == [] and len(webapp.creates) == 1 and len(calls) == 1


def test_a_withdrawn_capture_is_never_asked(grantless, monkeypatch):  # noqa: F811
    root, raw, context = grantless
    webapp = _install(monkeypatch, ValueError(NO_GRANT))
    monkeypatch.setattr(preview_store, "capture_withdrawn", lambda _root: True)
    assert context_reader.no_grant_ending(capture_root=root, task_context=context) is None
    assert webapp.reads == []


def test_the_host_retry_goes_through_the_same_check(grantless, monkeypatch):  # noqa: F811
    root, _, _ = grantless
    webapp = _install(monkeypatch, ValueError(NO_GRANT))
    assert _tick(root, lambda **_: pytest.fail("an ended job ran"), _due(root))["redriven"] == 1
    _ended_as_no_grant(root)
    assert len(webapp.reads) == 1 and webapp.creates == []


def test_the_unclaimed_redrive_goes_through_the_same_check(grantless, monkeypatch):  # noqa: F811
    root, raw, _ = grantless
    storage_root = root.parents[4]
    (root / listener.JOB_LEDGER_FILENAME).unlink()  # A delivery refused before any claim: no ledger.
    retry.spool_unclaimed(listener, storage_root=storage_root, payload=raw, payload_digest=listener.payload_sha256(raw),
                          result={"status": "capture_owner_observation_unavailable_retryable",
                                  "queue_disposition": "retryable"})
    assert len(list((storage_root / retry.UNCLAIMED_DIR).glob("*.json"))) == 1
    webapp = _install(monkeypatch, ValueError(NO_GRANT))
    later = datetime.now(timezone.utc) + timedelta(days=1)
    assert _tick(root, lambda **_: pytest.fail("an ended job ran"), later)["redriven"] == 1
    _ended_as_no_grant(root)
    assert len(webapp.reads) == 1 and webapp.creates == []
    assert list((storage_root / retry.UNCLAIMED_DIR).glob("*.json")) == []


def test_a_later_payload_for_the_same_capture_follows_the_existing_reopen_rule(grantless, monkeypatch):  # noqa: F811
    root, raw, _ = grantless
    _install(monkeypatch, ValueError(NO_GRANT))
    _process(root, raw, lambda **_: pytest.fail("ran"))
    ended = listener._read_job_ledger(root)
    # A different message of the same delivery: a producer delivery's ending stands, as for any ending.
    later = json.loads(raw)
    later["generated_at"] = "2026-11-11T12:00:00Z"
    later_raw = json.dumps(later, indent=2).encode()
    assert listener.payload_sha256(later_raw) != ended["source_payload_sha256"]
    webapp = _install(monkeypatch, MISSING)
    result = _process(root, later_raw, lambda **_: pytest.fail("an ended delivery ran again"))
    assert result["status"] == "skipped_terminal_authority_ended"
    assert listener._read_job_ledger(root) == ended and webapp.reads == [] and webapp.creates == []


def test_website_task_evidence_asks_before_a_first_grant_it_would_create(tmp_path, monkeypatch):
    """A capture admit_browser_preparation did not admit (a link bundle) reaches its first grant here."""
    from blueprint_pipeline import website_task_evidence as evidence
    root = tmp_path / "scenes/site-job1/captures/walkthrough-job1"
    context = {"request_id": "job1", "scene_id": "site-job1", "capture_id": "walkthrough-job1",
               "context_digest": "sha256:" + "a" * 64}
    monkeypatch.setattr(evidence, "read_consent_state", lambda _root: {"state": "active"})
    monkeypatch.setattr(evidence, "ingest_website_task_evidence", lambda **_: {})
    monkeypatch.setattr(evidence, "publish_website_item_evidence", lambda **_: None)
    webapp = WebApp(ValueError(NO_GRANT), {"granted": True})
    monkeypatch.setattr(context_reader, "website_webapp_request", webapp.request)

    class Descriptor:
        site_submission_id, scene_id, capture_id, metadata = "job1", "site-job1", "walkthrough-job1", {}

    with pytest.raises(ValueError, match=NO_GRANT):
        evidence.prepare_website_task_descriptor(descriptor=Descriptor(), capture_root=root,
            pipeline_dir=root / "pipeline", bucket="bucket", load_task_context=lambda **_: dict(context),
            load_sponsorship=webapp.create)
    assert len(webapp.reads) == 1 and webapp.creates == []
    assert listener.authority_ending(PipelineError(NO_GRANT)) == ("scene-sponsorship",
                                                                    context_reader.NO_GRANT_ENDED_CODE)


def test_operator_recovery_of_an_empty_capture_asks_before_its_first_grant(monkeypatch):
    from blueprint_pipeline import selected_handoff_recovery as recovery
    context = {"request_id": "job1", "scene_id": "site-job1", "capture_id": "walkthrough-job1",
               "context_digest": "sha256:" + "a" * 64}
    monkeypatch.setattr(context_reader, "load_current_website_task_context", lambda **_: dict(context))
    webapp = WebApp(ValueError(NO_GRANT), {"granted": True})
    monkeypatch.setattr(context_reader, "website_webapp_request", webapp.request)
    monkeypatch.setattr(context_reader, "load_website_scene_sponsorship", webapp.create)
    with pytest.raises(ValueError, match=NO_GRANT):
        recovery.current_admission({"scene_id": "site-job1", "capture_id": "walkthrough-job1"},
                                   {"request_id": "job1"})
    assert len(webapp.reads) == 1 and webapp.creates == []


@pytest.mark.parametrize("text", [NO_GRANT, f"run_e2e failed: {NO_GRANT}",
                                  f"website_control_preparation-spend_http_409:{context_reader.NO_GRANT_ENDED_CODE}"])
def test_the_listener_ends_on_the_whole_token_from_any_control_operation(text):
    ending = listener.authority_ending(PipelineError(text))
    assert ending is not None and ending[1] == context_reader.NO_GRANT_ENDED_CODE


@pytest.mark.parametrize("text", [NO_GRANT + "_later", "x" + NO_GRANT,
                                  "website_control_scene-sponsorship_http_503:website_scene_job_no_grant_ended", MISSING])
def test_the_listener_ignores_anything_but_the_whole_typed_token(text):
    assert listener.authority_ending(ValueError(text)) is None


# --- the silence sweep -------------------------------------------------------------------------

class SweepWebApp:
    def __init__(self, source, answer):
        self.source, self.answer, self.reads = source, answer, []

    def __call__(self, *, capture_id, operation, payload, **_):
        if operation != "scene-sponsorship":
            return _acceptance(self.source, status_reader.STATUS_V2)
        assert payload.get("create") is False
        self.reads.append(dict(payload))
        if isinstance(self.answer, BaseException):
            raise self.answer
        return dict(self.answer)


def _silent(source, monkeypatch, answer, **update):
    """A retryable attempt with no retained grant, stopped a full window ago while the listener kept pulling."""
    now = datetime.now(timezone.utc)
    ledger = _finish(source, _claim(source), updated_at=(now - WINDOW - timedelta(minutes=5)).isoformat(),
                     **(update or {"status": "failed_retryable", "last_error": "website_reconstruction_pending"}))
    _listening_since(source, now - WINDOW - timedelta(minutes=10), last_tick=now - timedelta(minutes=1))
    assert not (source.root / terminal.GRANT_FILE).exists()
    webapp = SweepWebApp(source, answer)
    monkeypatch.setattr(context_reader, "website_webapp_request", webapp)
    return ledger, webapp


def _sweep_end(source, now=None):
    return terminal.end_no_grant_preparation(source.root, alive_since=LONG_AGO, listener=listener, now=now)


def test_the_sweep_ends_a_silent_no_grant_ledger_on_the_typed_answer(website_capture, monkeypatch):
    stopped, webapp = _silent(website_capture, monkeypatch, ValueError(NO_GRANT))
    status_reader.reconcile_preparation_wakeups(website_capture.storage_root, listener=listener)
    assert webapp.reads == [{"request_id": website_capture.owner["request_id"],
                             "scene_id": website_capture.owner["scene_id"], "purpose": "scene_preparation",
                             "expected_task_context_digest": website_capture.context["context_digest"],
                             "create": False}]
    ended = _ended_as_no_grant(website_capture.root, stage=terminal.NO_GRANT_STAGE)
    assert ended["revision"] == stopped["revision"] + 2 and terminal.OVERDUE_FIELD not in ended
    assert ended["terminal_payload_sha256"] == stopped["source_payload_sha256"]
    # The status the site is told: authority_ended (the WebApp's acceptance is faked here).
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "authority_ended"
    status_reader.reconcile_preparation_wakeups(website_capture.storage_root, listener=listener)
    assert listener._read_job_ledger(website_capture.root) == ended and len(webapp.reads) == 1


@pytest.mark.parametrize("answer", [
    ValueError(MISSING), ValueError(EXPIRED), ValueError(NO_GRANT + "_later"),
    ValueError("website_control_scene-sponsorship_http_503:website_scene_sponsorship_unavailable"),
    URLError("unreachable"), TimeoutError("timed out"), {"schema_version": "website_scene_sponsorship.v1"},
], ids=["missing", "consent_expired", "not_the_whole_token", "http_503", "unreachable", "timeout", "grant"])
def test_the_sweep_leaves_a_no_grant_ledger_on_any_other_answer(website_capture, monkeypatch, answer):
    stopped, webapp = _silent(website_capture, monkeypatch, answer)
    assert _sweep_end(website_capture) is None and len(webapp.reads) == 1
    assert listener._read_job_ledger(website_capture.root) == stopped
    status_reader.reconcile_preparation_wakeups(website_capture.storage_root, listener=listener)
    ledger = listener._read_job_ledger(website_capture.root)
    assert ledger["status"] == "failed_retryable" and ledger[terminal.OVERDUE_FIELD]["reason"] == "recovery_needed"
    assert len(webapp.reads) == 1  # Asked once per silence window, not again on the next tick.


def test_the_sweep_asks_at_most_once_per_silence_window(website_capture, monkeypatch):
    _, webapp = _silent(website_capture, monkeypatch, ValueError(MISSING))
    now = datetime.now(timezone.utc)
    for minutes in range(0, 120, 7):
        _sweep_end(website_capture, now + timedelta(minutes=minutes))
    assert len(webapp.reads) == 1
    _sweep_end(website_capture, now + WINDOW)
    assert len(webapp.reads) == 2


def test_the_sweep_never_asks_inside_the_window_about_a_live_lease_or_a_retained_grant(website_capture, monkeypatch):
    stopped, webapp = _silent(website_capture, monkeypatch, ValueError(NO_GRANT))
    assert terminal.end_no_grant_preparation(website_capture.root, alive_since=datetime.now(timezone.utc),
                                             listener=listener) is None
    write_json(website_capture.root / terminal.GRANT_FILE, {"schema_version": "website_scene_sponsorship.v1"})
    assert _sweep_end(website_capture) is None
    (website_capture.root / terminal.GRANT_FILE).unlink()
    monkeypatch.setattr(preview_store, "capture_withdrawn", lambda _root: True)
    assert _sweep_end(website_capture) is None
    assert webapp.reads == [] and listener._read_job_ledger(website_capture.root) == stopped


def test_the_sweep_leaves_a_running_preparation_alone(website_capture, monkeypatch):
    now = datetime.now(timezone.utc)
    running = _claim(website_capture, lease_seconds=900)
    _listening_since(website_capture, now - 3 * WINDOW, last_tick=now - timedelta(minutes=1))
    webapp = SweepWebApp(website_capture, ValueError(NO_GRANT))
    monkeypatch.setattr(context_reader, "website_webapp_request", webapp)
    assert _sweep_end(website_capture) is None
    assert listener._read_job_ledger(website_capture.root) == running and webapp.reads == []


def test_the_sweep_ends_a_dead_workers_lease_on_the_typed_answer(website_capture, monkeypatch):
    started = datetime.now(timezone.utc) - 2 * WINDOW
    _claim(website_capture, lease_seconds=60, now=started)
    monkeypatch.setattr(context_reader, "website_webapp_request", SweepWebApp(website_capture, ValueError(NO_GRANT)))
    assert _sweep_end(website_capture)["terminal_code"] == context_reader.NO_GRANT_ENDED_CODE


def test_a_ledger_that_moved_while_the_webapp_answered_is_not_ended(website_capture, monkeypatch):
    stopped, webapp = _silent(website_capture, monkeypatch, ValueError(NO_GRANT))

    def redelivered(*, capture_id, operation, payload, **_):
        if operation == "scene-sponsorship":
            _finish(website_capture, _claim(website_capture), status="failed_retryable", last_error="again")
        return webapp(capture_id=capture_id, operation=operation, payload=payload)

    monkeypatch.setattr(context_reader, "website_webapp_request", redelivered)
    assert _sweep_end(website_capture) is None
    assert listener._read_job_ledger(website_capture.root)["status"] == "failed_retryable"
    assert listener._read_job_ledger(website_capture.root)["revision"] == stopped["revision"] + 2


# --- what is owed to the WebApp after the ending ----------------------------------------------

class CountingWebApp(SweepWebApp):
    """Counts every preparation-status report sent; the sweep's read answers ``answer``."""

    def __init__(self, source, answer):
        super().__init__(source, answer)
        self.reports = []

    def __call__(self, *, capture_id, operation, payload, **kwargs):
        if operation == "preparation-status":
            self.reports.append(dict(payload))
            raise ValueError("website_control_preparation-status_http_503:website_preparation_status_unavailable")
        return super().__call__(capture_id=capture_id, operation=operation, payload=payload, **kwargs)


def _index_hints(source):
    from blueprint_pipeline.website_preparation_delivery_index import DIRECTORY as DIRECTORY_NAME
    directory = source.storage_root / DIRECTORY_NAME
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def _passes(source, count=4):
    for step in range(count):
        status_reader.reconcile_preparation_wakeups(source.storage_root, listener=listener,
                                                    now_epoch=datetime.now(timezone.utc).timestamp() + step * 7200)


def test_after_the_no_grant_ending_nothing_is_owed_sent_or_due(website_capture, monkeypatch):
    stopped, _ = _silent(website_capture, monkeypatch, ValueError(NO_GRANT))
    # An earlier, undelivered report of the same job is pending (the WebApp refused it).
    status_reader.retain_preparation_wakeup(website_capture.root)
    earlier = json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())
    assert earlier["state"] == "pending" and earlier["revision"] == stopped["revision"]
    webapp = CountingWebApp(website_capture, ValueError(NO_GRANT))
    monkeypatch.setattr(context_reader, "website_webapp_request", webapp)
    _passes(website_capture)
    ended = _ended_as_no_grant(website_capture.root, stage=terminal.NO_GRANT_STAGE)
    assert webapp.reports == []  # Neither the earlier report nor the ending is sent, across several passes.
    record = json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())
    assert record["state"] == status_reader.RETIRED_STATE and record["revision"] == ended["revision"]
    assert record["retired_reason"] == status_reader.NOT_OWED_REASON
    assert record["report_not_owed"] == status_reader.NOT_OWED_EXPLANATION
    assert record["superseded_delivery"] == {"state": "pending", "revision": stopped["revision"]}
    assert _index_hints(website_capture) == []  # Nothing due, nothing parked.
    receipt = json.loads((website_capture.root / listener.JOB_TERMINAL_RECEIPT_FILENAME).read_text())
    assert receipt["preparation_status_report"] == "not_owed_webapp_recorded_this_ending"
    _passes(website_capture)
    assert webapp.reports == [] and _index_hints(website_capture) == []
    assert json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text()) == record


@pytest.mark.parametrize("code", ["consent_expired", "source_revoked"])
def test_other_endings_still_retain_and_send_their_report(website_capture, monkeypatch, code):
    ledger = _claim(website_capture)
    error = ValueError(f"website_control_scene-sponsorship_http_409:{code}")
    handoff = listener.HandoffMessage(bucket=website_capture.root.parents[3].name, scene_id=ledger["scene_id"],
        capture_id=ledger["capture_id"], raw_prefix_uri="gs://b/raw/", pipeline_handoff_uri=None)
    listener._finish_terminal_authority_ending(website_capture.root, handoff=handoff, owner=ledger["lease_owner"],
        token=ledger["lease_token"], operation="scene-sponsorship", code=code, error=error, stage="run_e2e",
        attempt_count=ledger["attempt_count"], attempt_started_at=ledger["last_attempt_started_at"],
        previous_history=[], payload_digest=ledger["source_payload_sha256"],
        producer_delivery_key=ledger["producer_delivery_key"])
    receipt = json.loads((website_capture.root / listener.JOB_TERMINAL_RECEIPT_FILENAME).read_text())
    assert "preparation_status_report" not in receipt
    webapp = CountingWebApp(website_capture, ValueError(NO_GRANT))
    monkeypatch.setattr(context_reader, "website_webapp_request", webapp)
    status_reader.reconcile_preparation_wakeups(website_capture.storage_root, listener=listener)
    record = json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())
    assert record["state"] == "pending" and "retired_reason" not in record
    assert len(webapp.reports) == 1 and _index_hints(website_capture) != []
