# Covers (for impacted-test selection):
#   src/blueprint_pipeline/job_retry_policy.py
#   src/blueprint_pipeline/pubsub_handoff_listener.py
#   src/blueprint_pipeline/website_preparation_retry.py
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/website_preparation_terminal.py
#   src/blueprint_pipeline/website_atlas_preview.py
#   src/blueprint_pipeline/website_scene_workspace_retention.py
"""A run the site's owner closed is closed: the job ends, nothing is retried, nothing more is owed.

The WebApp answers the reads that start or pay for work on a closed run with the typed 409
``website_run_closed_by_owner`` (``websiteRunClosure.ts``: no new execution, an explicit new
instruction required). Before this,
the host recorded that answer as an ordinary retryable refusal: one job re-ran about every two
hours for ever (50 attempts, revision 168 on the host), its status callback paged and never
parked, and the clean-up held its 4.85 GiB workspace as "not terminal". The ended workspace stays
local under the authority-ended rule: the Pipeline cannot tell a closed run from a closed and
revoked one (the WebApp checks the closure first). Here
the real listener, claim, ledger, retry tick, sweep, status delivery, Atlas preview step and
workspace retention run; only the WebApp and run_e2e are fakes.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from blueprint_pipeline import job_retry_policy as policy
from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_preparation_retry as retry
from blueprint_pipeline import website_preparation_status as status_reader
from blueprint_pipeline import website_preparation_terminal as terminal
from blueprint_pipeline import website_scene_workspace_retention as retention
from blueprint_pipeline import website_task_context as context_reader
from blueprint_pipeline.common import write_json
from tests.test_website_preparation_retry import _due, _tick, failed  # noqa: F401
from tests.test_website_preparation_status import prepared_source  # noqa: F401
from tests.test_website_preparation_terminal_status import LONG_AGO, _claim, _finish, _listening_since
from tests.test_website_reconstruction_resume import waiting  # noqa: F401
from tests.test_website_scene_workspace_retention import (  # noqa: F401 - the autouse disk fixture
    BUCKET, CAPTURE, HOUR, SCENE, FakeCloud, _plan, isolated_disk_ledger,
)

pytest_plugins = ["tests.test_website_preparation_terminal_status"]  # the website_capture fixture

CLOSED = "website_control_task-context_http_409:website_run_closed_by_owner"
SUPERSEDED = "website_control_task-context_http_409:website_run_capture_superseded"


# --- the code itself ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [CLOSED, f"run_e2e failed: {CLOSED}",
                                  "website_control_preparation-spend_http_409:website_run_closed_by_owner"])
def test_the_listener_ends_a_job_on_the_typed_closure_from_any_control_operation(text):
    ending = listener.authority_ending(ValueError(text))
    assert ending is not None and ending[1] == policy.OWNER_CLOSED_CODE
    assert policy.owner_closed(text) and retry.classify(text) == retry.OWNER_CLOSED
    assert retry.retry_delay(retry.OWNER_CLOSED, 1) is None


@pytest.mark.parametrize("text", [
    SUPERSEDED, CLOSED + "_later", "x" + CLOSED,
    "website_control_task-context_http_503:website_run_closed_by_owner",
    "website_control_task-context_http_409:website_task_context_changed",
])
def test_any_other_answer_is_not_a_closure(text):
    assert listener.authority_ending(ValueError(text)) is None
    assert not policy.owner_closed(text)
    assert retry.classify(text) != retry.OWNER_CLOSED


def test_the_bare_code_is_a_closure_only_as_the_whole_value():
    assert policy.owner_closed("website_run_closed_by_owner")
    assert not policy.owner_closed("website_run_closed_by_owner_x")
    assert not policy.owner_closed(None)


# --- a run that meets the closure now ----------------------------------------------------------

def _closed_run(**_kwargs):
    raise ValueError(CLOSED)


def test_a_due_retry_that_meets_the_closure_ends_the_job_and_is_never_run_again(failed):  # noqa: F811
    root, _, _ = failed
    assert _tick(root, _closed_run, _due(root))["redriven"] == 1
    ledger = listener._read_job_ledger(root)
    assert ledger["status"] == listener.TERMINAL_AUTHORITY_STATUS
    assert ledger["terminal_code"] == policy.OWNER_CLOSED_CODE and ledger["terminal_operation"] == "task-context"
    assert retry.armed(ledger) is None
    receipt = json.loads((root / listener.JOB_TERMINAL_RECEIPT_FILENAME).read_text())
    assert receipt["preparation_status_report"] == "not_owed_owner_closed_this_run"
    assert listener._terminal_receipt_current(receipt, ledger)
    later = datetime.now(timezone.utc) + timedelta(days=30)
    assert _tick(root, lambda **_: pytest.fail("a closed run ran again"), later)["redriven"] == 0
    assert listener._read_job_ledger(root) == ledger


def test_another_409_keeps_the_retry_schedule(failed):  # noqa: F811
    root, _, _ = failed

    def superseded(**_kwargs):
        raise ValueError(SUPERSEDED)
    assert _tick(root, superseded, _due(root))["redriven"] == 1
    ledger = listener._read_job_ledger(root)
    assert ledger["status"] == "failed_retryable" and retry.armed(ledger) is not None


# --- the two jobs on the host: ledgers the old code left in its loop ----------------------------

def _left_in_the_loop(source, *, attempts, retry_count, next_attempt_at):
    """A ledger as the old code wrote it: the closure recorded as a retryable refusal, a retry armed."""
    ledger = _finish(source, _claim(source), status="failed_retryable", last_error=CLOSED,
                     last_error_type="ValueError",
                     updated_at=(datetime.now(timezone.utc) - timedelta(hours=3)).isoformat())
    ledger = {**ledger, "attempt_count": attempts,
              retry.FIELD: {"schema_version": retry.SCHEMA, "source_payload_sha256": ledger["source_payload_sha256"],
                            "producer_delivery_key": ledger["producer_delivery_key"], "error": CLOSED,
                            "kind": "refusal", "retry_count": retry_count, "next_attempt_at": next_attempt_at}}
    write_json(source.root / listener.JOB_LEDGER_FILENAME, ledger)
    assert retry.armed(ledger) is not None
    return ledger


class WebApp:
    """Counts every call and refuses each with the closure (the test asserts no call is made at all)."""

    def __init__(self):
        self.calls = []

    def __call__(self, *, capture_id, operation, payload, **_):
        self.calls.append(operation)
        raise ValueError(f"website_control_{operation}_http_409:website_run_closed_by_owner")


def _index_hints(source):
    from blueprint_pipeline.website_preparation_delivery_index import DIRECTORY
    directory = source.storage_root / DIRECTORY
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


@pytest.mark.parametrize("attempts, retry_count, next_attempt_at", [
    (50, 1, "2026-11-11T22:26:19.758062Z"),   # site-capture-05d7f712: re-run every ~2 h 16 min
    (6, 6, "2026-11-12T13:55:36.897883Z"),    # site-capture-00a25d9f: its sixth and last timed retry ahead
], ids=["05d7f712", "00a25d9f"])
def test_a_job_left_in_the_old_loop_ends_on_the_next_sweep(website_capture, monkeypatch, attempts, retry_count,
                                                           next_attempt_at):
    stopped = _left_in_the_loop(website_capture, attempts=attempts, retry_count=retry_count,
                                next_attempt_at=next_attempt_at)
    # Its undelivered status report is pending, as on the host (the status read meets the closure).
    status_reader.retain_preparation_wakeup(website_capture.root)
    pending = json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())
    assert pending["state"] == "pending" and _index_hints(website_capture) != []
    now = datetime.now(timezone.utc)
    _listening_since(website_capture, LONG_AGO, last_tick=now - timedelta(minutes=1))
    webapp = WebApp()
    monkeypatch.setattr(context_reader, "website_webapp_request", webapp)

    status_reader.reconcile_preparation_wakeups(website_capture.storage_root, listener=listener)

    ended = listener._read_job_ledger(website_capture.root)
    assert ended["status"] == listener.TERMINAL_AUTHORITY_STATUS
    assert ended["terminal_code"] == policy.OWNER_CLOSED_CODE
    assert ended["attempt_history"][-1]["stage"] == terminal.OWNER_CLOSED_STAGE
    assert ended["terminal_payload_sha256"] == stopped["source_payload_sha256"]
    assert webapp.calls == []  # Ended from the refusal the ledger records: nothing asked, nothing reported.
    # No retry is scheduled, and the tick never runs it again, however late.
    assert retry.armed(ended) is None
    later = datetime.now(timezone.utc) + timedelta(days=30)
    assert _tick(website_capture.root, lambda **_: pytest.fail("a closed run ran again"), later)["redriven"] == 0
    # The overdue observer does not re-arm it.
    assert terminal.record_unretried_preparation(website_capture.root, alive_since=LONG_AGO,
                                                 now=later) is None
    # Its status delivery ended: retired as not owed, nothing due, nothing parked, nothing sent.
    record = json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())
    assert record["state"] == status_reader.RETIRED_STATE and record["revision"] == ended["revision"]
    assert record["retired_reason"] == status_reader.NOT_OWED_REASON
    assert record["report_not_owed"] == status_reader.NOT_OWED_CLOSED_EXPLANATION
    assert record["superseded_delivery"] == {"state": "pending", "revision": pending["revision"]}
    assert _index_hints(website_capture) == []
    for step in range(4):
        status_reader.reconcile_preparation_wakeups(website_capture.storage_root, listener=listener,
                                                    now_epoch=now.timestamp() + step * 7200)
    assert webapp.calls == [] and listener._read_job_ledger(website_capture.root) == ended
    assert json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text()) == record
    # The clean-up's own verdict: the capture is terminal now (its acknowledgement follows the usual rule).
    _, reasons = retention._capture(website_capture.root, scene_id=ended["scene_id"],
                                    capture_id=ended["capture_id"], now=later.timestamp(),
                                    ack_retention_seconds=7 * 24 * HOUR)
    assert reasons == []


def test_a_closure_inside_a_longer_message_with_nothing_armed_is_ended_not_left(failed):  # noqa: F811
    """The review's probe: the classifier refuses to retry it, so the sweep must end it (one match)."""
    root, _, _ = failed
    path = root / listener.JOB_LEDGER_FILENAME
    ledger = json.loads(path.read_text())
    ledger.pop(retry.FIELD, None)
    ledger["last_error"] = f"run_e2e failed: {CLOSED}"
    write_json(path, ledger)
    assert retry.classify(ledger["last_error"]) == retry.OWNER_CLOSED
    ended = terminal.end_owner_closed_preparation(root, listener=listener)
    assert ended is not None and ended["terminal_code"] == policy.OWNER_CLOSED_CODE
    later = datetime.now(timezone.utc) + timedelta(days=3)
    assert _tick(root, lambda **_: pytest.fail("a closed run ran again"), later)["redriven"] == 0


def test_a_report_of_a_result_finished_before_the_closure_stays_owed(website_capture, monkeypatch):
    stopped = _left_in_the_loop(website_capture, attempts=2, retry_count=1, next_attempt_at="2026-11-11T00:00:00Z")
    # The latest attempt before the pending report is a finished one: the report is about its result.
    finished_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    write_json(website_capture.root / listener.JOB_LEDGER_FILENAME, {**stopped, "attempt_history": [
        {"attempt_number": 1, "status": "completed", "stage": "run_e2e", "started_at": finished_at,
         "completed_at": finished_at}]})
    status_reader.retain_preparation_wakeup(website_capture.root)
    pending = json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())
    assert pending["state"] == "pending"
    _listening_since(website_capture, LONG_AGO, last_tick=datetime.now(timezone.utc) - timedelta(minutes=1))
    monkeypatch.setattr(context_reader, "website_webapp_request", WebApp())
    ended = terminal.end_owner_closed_preparation(website_capture.root, listener=listener)
    assert ended is not None
    status_reader.retain_preparation_wakeup(website_capture.root)
    record = json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())
    assert record["state"] == "pending" and "retired_reason" not in record
    assert record["revision"] == ended["revision"] and _index_hints(website_capture) != []


@pytest.mark.parametrize("error", [SUPERSEDED, CLOSED + "_later", "website_reconstruction_pending",
                                   "website_run_closed_by_owner",
                                   "website_control_task-context_http_503:website_run_closed_by_owner"],
                         ids=["other_409", "longer_token", "other_error", "bare_code", "other_status"])
def test_the_sweep_leaves_any_other_recorded_error_as_it_is(website_capture, monkeypatch, error):
    ledger = _finish(website_capture, _claim(website_capture), status="failed_retryable", last_error=error)
    _listening_since(website_capture, LONG_AGO, last_tick=datetime.now(timezone.utc) - timedelta(minutes=1))
    monkeypatch.setattr(context_reader, "website_webapp_request", WebApp())
    assert terminal.end_owner_closed_preparation(website_capture.root, listener=listener) is None
    assert listener._read_job_ledger(website_capture.root) == ledger


def test_the_sweep_never_ends_a_live_lease(website_capture):
    running = _claim(website_capture, lease_seconds=900)
    assert terminal.end_owner_closed_preparation(website_capture.root, listener=listener) is None
    assert listener._read_job_ledger(website_capture.root) == running


# --- the Atlas preview's own step ---------------------------------------------------------------

def test_the_atlas_preview_ends_on_the_closure_instead_of_counting_refusal_retries():
    from blueprint_pipeline import website_atlas_preview as preview
    job = {"schema_version": preview.SCHEMA, "state": "processing", "stage": "queued", "reason": None,
           "revision": 3, "enqueued_at_epoch": 1_000, "updated_at_epoch": 1_000, "refusal_retries": 4,
           "refusal_retries_total": 4, "last_refusal_reason": policy.OWNER_CLOSED_CODE,
           "next_attempt_at_epoch": 1_000, "retry_at_epoch": None}
    ended = preview._failed_step(job, ValueError(CLOSED), 2_000)
    assert ended["state"] == "failed" and ended["reason"] == policy.OWNER_CLOSED_CODE
    assert ended["refusal_retries"] == 4  # Not counted again: the schedule is not used.
    other = preview._failed_step(job, ValueError(SUPERSEDED), 2_000)
    assert other["state"] == "processing" and other["refusal_retries"] == 5


def test_the_atlas_preview_owes_no_delivery_to_a_closed_run(monkeypatch):
    from blueprint_pipeline import website_atlas_preview as preview
    job = {"request_id": "job1", "scene_id": "site-job1", "capture_id": "walkthrough-job1", "delivered_revision": 1,
           "revision": 2, "delivery_next_at_epoch": 0, "delivery_attempts": 3}
    monkeypatch.setattr(preview, "preview", lambda _job: {"preview_digest": "sha256:" + "0" * 64})
    monkeypatch.setattr(context_reader, "website_webapp_request", WebApp())
    delivered = preview._deliver(job, now=lambda: 5_000)
    assert delivered["delivered_revision"] == 2 and delivered["delivery_attempts"] == 0
    assert delivered["delivery_not_owed"] == policy.OWNER_CLOSED_CODE


# --- the clean-up after the ending --------------------------------------------------------------

@pytest.mark.parametrize("revoked_earlier", [True, False], ids=["closed_after_source_revoked", "closed_only"])
def test_a_closed_run_keeps_the_authority_ended_guard(tmp_path, revoked_earlier):
    """A closed run may also be revoked (the WebApp answers the closure first), so it is never archived."""
    from tests.test_website_scene_workspace_retention import _bundle, _payload
    from tests.test_pubsub_handoff_listener import FakeStorageClient, SUBSCRIPTION
    cloud = FakeCloud()
    storage = tmp_path / "pubsub-handoffs"
    blobs = _bundle(CAPTURE)
    for blob in blobs:
        cloud.put(blob.name, blob._data)

    def closed(**_kwargs):
        try:
            raise ValueError(CLOSED)
        except ValueError as exc:
            raise listener.PipelineError("website scene failed") from exc
    listener.process_handoff_payload(_payload(CAPTURE), storage_root=storage, provider="openai",
                                     storage_client=FakeStorageClient(blobs), run_e2e=closed,
                                     stage_control_plane=True, run_e2e_enabled=False)
    capture_root = storage / BUCKET / "scenes" / SCENE / "captures" / CAPTURE
    ledger = listener._read_job_ledger(capture_root)
    assert ledger["terminal_code"] == policy.OWNER_CLOSED_CODE
    if revoked_earlier:
        write_json(capture_root / listener.JOB_LEDGER_FILENAME, {**ledger, "attempt_history": [
            {"attempt_number": 0, "status": listener.TERMINAL_AUTHORITY_STATUS, "code": "source_revoked",
             "payload_sha256": "f" * 64}, *ledger["attempt_history"]]})
    assert listener._write_ack_receipt(
        capture_root=capture_root, subscription=SUBSCRIPTION, message_id=f"msg-{CAPTURE}",
        payload_digest=listener.payload_sha256(_payload(CAPTURE)), delivery_attempt=1,
        disposition="terminal_authority_ended") is None
    plan = _plan(tmp_path, cloud)
    assert plan["status"] == "retained" and plan["reasons"] == ["authority_ended_capture_kept_local"]
