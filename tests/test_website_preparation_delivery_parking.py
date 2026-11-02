"""D-05: a status callback whose read keeps failing is parked, not dropped; a live one pages.

On 2026-11-02 seven old test captures had held the preparation status channel
for 4.6 days: every status read failed before the WebApp, each retried every 30
minutes, and the backlog read ``overloaded``. A delivery whose status read has
failed on every attempt for a day is now parked: retried every 6 hours, outside
the due backlog, counted on its own, and live again after one successful read
or a newer revision from the producer. A live delivery older than an hour pages
through the capacity tick; parked ones only warn. Nothing is deleted.
"""
from __future__ import annotations

import json
import time
from collections import namedtuple

import pytest

from blueprint_pipeline import control_plane_capacity_controller as capacity
from blueprint_pipeline import website_preparation_delivery_index as index
from blueprint_pipeline import website_preparation_status as status
from blueprint_pipeline import website_task_context as context
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_website_preparation_due_delivery import _synthetic_ack, _synthetic_pending, _synthetic_status
from tests.test_website_preparation_terminal_status import website_capture, _claim, _finish  # noqa: F401, F811

DAY = 24 * 3600.0
HOUR = 3600.0
START = 1_000_000.0
PARKING_FIELDS = ("parked_reason", "parked_at_epoch", "status_unavailable_since_epoch", "last_unavailable_reason")
DISK = namedtuple("Usage", "total used free")(100 * 1024**3, 10 * 1024**3, 90 * 1024**3)


def _record(root):
    return json.loads((root / status.DELIVERY_FILE).read_text())


def _set(root, **fields):
    record = {**_record(root), **fields}
    write_json(root / status.DELIVERY_FILE, record)
    index.update(root, record)


def _hint(storage, root):
    for path in (storage / index.DIRECTORY).glob("*.json"):
        hint = json.loads(path.read_text())
        if hint["capture_relative_path"].endswith("/" + root.name):
            return hint
    return None


def _tick(storage, now):
    return status.reconcile_preparation_wakeups(storage, limit=1, now_epoch=now)


def _refuse_reads(monkeypatch, calls=None, reason="task_context_changed"):
    def refuse(*, capture_root, selectors, **_):
        if calls is not None:
            calls.append(capture_root)
        raise status.PreparationStatusUnavailable(reason=reason)

    monkeypatch.setattr(status, "retain_preparation_wakeup", lambda root: None)
    monkeypatch.setattr(status, "read_preparation_status", refuse)


def _fail_until_parked(storage, root, start):
    """Attempt at every retry time the delivery asks for; return when it parks."""
    now = start
    for _ in range(200):
        _tick(storage, now)
        record = _record(root)
        if "parked_reason" in record:
            return now, record
        assert now - start < DAY, "a full day of failed reads parks the delivery"
        now = record["next_attempt_at_epoch"]
    raise AssertionError("the delivery never parked")


# ------------------------------------------------------------------ parking

def test_a_day_of_failed_status_reads_parks_the_delivery_and_keeps_its_evidence(tmp_path, monkeypatch):
    root = _synthetic_pending(tmp_path, 0, now=START)
    before = _record(root)
    ledger = (root / "pipeline_job_ledger.json").read_bytes()
    _refuse_reads(monkeypatch)
    parked_at, record = _fail_until_parked(tmp_path, root, START)
    # It parks on the first failed attempt a full day after the first failure.
    assert DAY <= parked_at - START < DAY + 1800
    assert record["parked_reason"] == "preparation_status_persistently_unavailable"
    assert record["last_error_type"] == "PreparationStatusUnavailable"
    assert record["last_unavailable_reason"] == "task_context_changed"
    assert record["status_unavailable_since_epoch"] == START
    assert record["parked_at_epoch"] == parked_at
    assert record["next_attempt_at_epoch"] == parked_at + 6 * HOUR
    # Parking keeps the record: same selectors, revision, state and age; counters only grow.
    kept = ("schema_version", "state", "selectors", "revision", "pending_since_epoch", "delivery_attempt_count")
    assert {key: record[key] for key in kept} == {key: before[key] for key in kept}
    assert record["delivery_failure_count"] > 40
    assert (root / "pipeline_job_ledger.json").read_bytes() == ledger
    assert _hint(tmp_path, root)["parked_at_epoch"] == parked_at


def test_a_successful_read_inside_the_day_restarts_the_clock(tmp_path, monkeypatch):
    root = _synthetic_pending(tmp_path, 0, now=START)
    readable = {"now": False}

    def read(*, capture_root, selectors, schema_version=status.STATUS_V1):
        if not readable["now"]:
            raise status.PreparationStatusUnavailable(reason="capture_owner_changed")
        return _synthetic_status(capture_root=capture_root, selectors=selectors, schema_version=schema_version)

    def outage(**_):
        raise TimeoutError("private WebApp outage")

    monkeypatch.setattr(status, "retain_preparation_wakeup", lambda root: None)
    monkeypatch.setattr(status, "read_preparation_status", read)
    monkeypatch.setattr(context, "website_webapp_request", outage)
    now = START
    while now - START < 20 * HOUR:
        _tick(tmp_path, now)
        now = _record(root)["next_attempt_at_epoch"]
    readable["now"] = True
    _tick(tmp_path, now)
    record = _record(root)
    # The read succeeded (the callback then failed), so the failed-read clock is gone.
    assert record["state"] == "pending" and record["delivery_attempt_count"] == 1
    assert not any(key in record for key in PARKING_FIELDS)
    readable["now"] = False
    restart = record["next_attempt_at_epoch"]
    parked_at, record = _fail_until_parked(tmp_path, root, restart)
    assert parked_at - START >= 20 * HOUR + DAY
    assert record["status_unavailable_since_epoch"] == restart
    assert record["last_unavailable_reason"] == "capture_owner_changed"


def test_a_parked_delivery_is_retried_at_most_every_six_hours(tmp_path, monkeypatch):
    root = _synthetic_pending(tmp_path, 0, now=START)
    calls = []
    _refuse_reads(monkeypatch, calls)
    parked_at, _ = _fail_until_parked(tmp_path, root, START)
    calls.clear()
    for hour in range(1, 6):
        _tick(tmp_path, parked_at + hour * HOUR)
    assert calls == []
    _tick(tmp_path, parked_at + 6 * HOUR)
    assert calls == [root]
    record = _record(root)
    assert record["parked_at_epoch"] == parked_at
    assert record["next_attempt_at_epoch"] == parked_at + 12 * HOUR


@pytest.mark.parametrize("webapp_accepts", [True, False])
def test_a_later_successful_read_unparks_the_delivery(tmp_path, monkeypatch, webapp_accepts):
    root = _synthetic_pending(tmp_path, 0, now=START)
    _refuse_reads(monkeypatch)
    parked_at, _ = _fail_until_parked(tmp_path, root, START)

    def callback(**_):
        if not webapp_accepts:
            raise TimeoutError("private WebApp outage")
        return _synthetic_ack(root)

    monkeypatch.setattr(status, "read_preparation_status", _synthetic_status)
    monkeypatch.setattr(context, "website_webapp_request", callback)
    now = parked_at + 6 * HOUR
    _tick(tmp_path, now)
    record = _record(root)
    assert not any(key in record for key in PARKING_FIELDS)
    if webapp_accepts:
        assert record["state"] == "delivered" and _hint(tmp_path, root) is None
    else:
        assert record["state"] == "pending" and record["next_attempt_at_epoch"] <= now + 1800
        assert "parked_at_epoch" not in _hint(tmp_path, root)
        _, metrics = index.due(tmp_path, now=now, limit=1)
        assert (metrics["parked_count"], metrics["live_count"]) == (0, 1)


def test_a_newer_revision_from_the_producer_unparks_the_delivery(website_capture):  # noqa: F811
    source = website_capture
    _finish(source, _claim(source), status="failed_retryable", last_error="retained failure")
    status.retain_preparation_wakeup(source.root)
    # The site edited its job after this attempt retained its context, so the
    # real status read refuses every attempt with a changed context digest.
    changed = {**source.context, "description": "Edited job"}
    changed["context_digest"] = canonical_digest(changed, digest_field="context_digest")
    source.current["context"] = changed
    now = time.time()
    _tick(source.storage_root, now)
    first = _record(source.root)
    assert first["status_unavailable_since_epoch"] == now and "parked_reason" not in first
    assert first["last_unavailable_reason"] == "task_context_changed"
    _tick(source.storage_root, now + DAY)
    parked = _record(source.root)
    assert parked["parked_reason"] == "preparation_status_persistently_unavailable"
    # A new attempt retains the edited context and commits a newer revision.
    write_json(source.root / "pipeline/website_task_context.json", changed)
    _finish(source, _claim(source), status="failed_retryable", last_error="next attempt")
    status.retain_preparation_wakeup(source.root)
    record = _record(source.root)
    assert record["revision"] > parked["revision"] and record["state"] == "pending"
    assert record["selectors"]["task_context_digest"] == changed["context_digest"]
    assert not any(key in record for key in PARKING_FIELDS)
    assert "parked_at_epoch" not in _hint(source.storage_root, source.root)
    rows, metrics = index.due(source.storage_root, now=now + DAY + 1, limit=1)
    assert [row.capture_root for row in rows] == [source.root]
    assert (metrics["parked_count"], metrics["live_count"]) == (0, 1)


def test_parked_rows_leave_the_due_backlog_and_count_on_their_own(tmp_path):
    now = START + 10 * DAY
    old = _synthetic_pending(tmp_path, 0, now=now)
    _set(old, pending_since_epoch=now - 5 * DAY, parked_reason="preparation_status_persistently_unavailable",
         parked_at_epoch=now - 7 * HOUR, status_unavailable_since_epoch=now - 4 * DAY,
         next_attempt_at_epoch=now - HOUR)  # due for its six-hour retry
    waiting = _synthetic_pending(tmp_path, 1, now=now)
    _set(waiting, pending_since_epoch=now - 2 * HOUR, next_attempt_at_epoch=now + 600)  # live, in backoff
    fresh = _synthetic_pending(tmp_path, 2, now=now)  # live and due, pending for 120 s
    rows, metrics = index.due(tmp_path, now=now, limit=1)
    assert [row.capture_root for row in rows] == [fresh]
    assert (metrics["due_backlog"], metrics["oldest_due_age_seconds"], metrics["overloaded"]) == (1, 120, False)
    assert (metrics["live_count"], metrics["oldest_live_age_seconds"]) == (2, 2 * HOUR)
    assert (metrics["parked_count"], metrics["oldest_parked_age_seconds"]) == (1, 5 * DAY)
    rows, _ = index.due(tmp_path, now=now, limit=5)
    # A parked retry runs only after the live work that is due.
    assert [row.capture_root for row in rows] == [fresh, old]


def test_a_record_from_before_parking_that_never_reached_the_webapp_parks_on_its_next_failure(
    tmp_path, monkeypatch,
):
    now = START + 10 * DAY
    # The 2026-11-02 shape: about 197 failures since the index began tracking
    # the record, and not one callback (no ``last_attempt_at_epoch``).
    legacy = _synthetic_pending(tmp_path, 0, now=now)
    _set(legacy, pending_since_epoch=now - 4.7 * DAY, delivery_failure_count=197,
         last_error_type="PreparationStatusUnavailable", next_attempt_at_epoch=now - 60)
    # One that did reach the WebApp starts a fresh clock at its next failure.
    reached = _synthetic_pending(tmp_path, 1, now=now)
    _set(reached, pending_since_epoch=now - 3 * DAY, delivery_failure_count=40, delivery_attempt_count=30,
         last_attempt_at_epoch=now - 2 * DAY, last_error_type="PreparationStatusUnavailable",
         next_attempt_at_epoch=now - 60)
    _refuse_reads(monkeypatch)
    _tick(tmp_path, now)
    _tick(tmp_path, now)
    assert _record(legacy)["parked_at_epoch"] == now
    assert _record(legacy)["status_unavailable_since_epoch"] == now - 4.7 * DAY
    assert "parked_reason" not in _record(reached)
    assert _record(reached)["status_unavailable_since_epoch"] == now


def test_only_a_fixed_code_is_kept_as_the_unavailable_reason():
    def raised(cause):
        try:
            try:
                raise cause
            except Exception as exc:
                raise status.PreparationStatusUnavailable() from exc
        except status.PreparationStatusUnavailable as exc:
            return exc

    missing = ValueError("website_control_scene-sponsorship_http_409:website_scene_sponsorship_missing")
    assert status.unavailable_reason(raised(missing)) == str(missing)
    assert status.unavailable_reason(raised(TimeoutError("private response body"))) is None
    assert status.unavailable_reason(raised(KeyError("raw_video"))) is None
    assert status.unavailable_reason(status.PreparationStatusUnavailable(reason="consent_revoked")) == "consent_revoked"
    assert str(status.PreparationStatusUnavailable(reason="consent_revoked")) == "website_preparation_status_unavailable"


# ------------------------------------------------------------------- paging

def _metrics(storage, observed, **fields):
    write_json(storage / index.METRICS_FILE, {
        "schema_version": "website_preparation_delivery_metrics.v1", "observed_at_epoch": observed,
        "due_backlog": 0, "oldest_due_age_seconds": 0.0, "live_count": 0, "oldest_live_age_seconds": 0.0,
        "parked_count": 0, "oldest_parked_age_seconds": 0.0, "hint_count_scanned": 0,
        "hint_scan_limit_reached": False, "invalid_hint_count": 0, "overloaded": False, "attempted": 0,
        "delivered": 0, "unavailable": 0, "repair_scanned": 0, "successful_delivery_lag_seconds": [], **fields})


def _capacity_tick(tmp_path, storage, now, posted):
    return capacity.run_controller(
        mounts=["/test"], report_root=tmp_path / "report", reservation_root=tmp_path / "ledger",
        webhook_url="https://alerts.example/hook", volume=None, ack="", token="", disk_usage=lambda _: DISK,
        now=now, survey=None, poster=lambda _url, report: posted.append(report),
        delivery_check=lambda observed: index.observe_delivery_backlog(storage, now=observed))


def _codes(report):
    return {alert["code"]: alert for alert in report["alerts"] if alert["code"].startswith("preparation_delivery")}


def test_a_live_delivery_undelivered_for_over_an_hour_pages(tmp_path):
    storage = tmp_path / "handoffs"
    _metrics(storage, START, live_count=2, oldest_live_age_seconds=3601.0, parked_count=7,
             oldest_parked_age_seconds=4.7 * DAY)
    posted = []
    report = _capacity_tick(tmp_path, storage, START, posted)
    page = _codes(report)["preparation_delivery_stalled"]
    assert page["severity"] == "page" and report["level"] == "critical" and len(posted) == 1
    assert (page["count"], page["age_seconds"], page["threshold_seconds"], page["parked_count"]) == (2, 3601, 3600, 7)
    assert _codes(report)["preparation_delivery_parked"]["severity"] == "warn"
    summary = _codes(capacity.capacity_summary(report))["preparation_delivery_stalled"]
    assert (summary["severity"], summary["count"], summary["parked_count"]) == ("page", 2, 7)


def test_parked_deliveries_and_a_live_one_under_an_hour_only_warn(tmp_path):
    storage = tmp_path / "handoffs"
    _metrics(storage, START, live_count=1, oldest_live_age_seconds=3599.0, parked_count=7,
             oldest_parked_age_seconds=4.7 * DAY)
    report = _capacity_tick(tmp_path, storage, START, [])
    assert "preparation_delivery_stalled" not in _codes(report)
    parked = _codes(report)["preparation_delivery_parked"]
    assert (parked["severity"], parked["count"], parked["age_seconds"]) == ("warn", 7, round(4.7 * DAY))
    assert report["level"] == "warning"
    assert not [alert for alert in report["alerts"] if alert["severity"] == "page"]


def test_the_listener_metrics_page_for_a_live_row_and_never_for_parked_rows(tmp_path, monkeypatch):
    storage = tmp_path / "handoffs"
    now = START + 10 * DAY
    parked = _synthetic_pending(storage, 0, now=now)
    _set(parked, pending_since_epoch=now - 4.7 * DAY, delivery_failure_count=197,
         last_error_type="PreparationStatusUnavailable", next_attempt_at_epoch=now - 60)
    _refuse_reads(monkeypatch)
    _tick(storage, now)
    _tick(storage, now + 60)
    quiet = _capacity_tick(tmp_path, storage, now + 60, [])
    assert "preparation_delivery_stalled" not in _codes(quiet)
    assert _codes(quiet)["preparation_delivery_parked"]["count"] == 1
    live = _synthetic_pending(storage, 1, now=now)
    _set(live, pending_since_epoch=now - 2 * HOUR)
    _tick(storage, now + 120)
    _tick(storage, now + 180)
    page = _codes(_capacity_tick(tmp_path, storage, now + 180, []))["preparation_delivery_stalled"]
    assert page["severity"] == "page" and page["count"] == 1 and page["parked_count"] == 1
    assert page["age_seconds"] >= 2 * HOUR


def test_silent_listener_metrics_age_the_backlog_and_unreadable_metrics_warn(tmp_path):
    storage = tmp_path / "handoffs"
    _metrics(storage, START, live_count=1, oldest_live_age_seconds=1800.0)
    report = _capacity_tick(tmp_path, storage, START + 2 * HOUR, [])
    assert _codes(report)["preparation_delivery_stalled"]["age_seconds"] == 1800 + 2 * HOUR
    assert _codes(report)["preparation_delivery_metrics_unreadable"]["reason"] == "metrics_stale"
    (storage / index.METRICS_FILE).write_text("{broken")
    report = _capacity_tick(tmp_path, storage, START + 2 * HOUR, [])
    assert set(_codes(report)) == {"preparation_delivery_metrics_unreadable"}
    assert _codes(report)["preparation_delivery_metrics_unreadable"]["severity"] == "warn"
    assert index.observe_delivery_backlog(tmp_path / "absent", now=START) == {"status": "absent", "alerts": []}


def test_the_page_says_how_long_the_oldest_live_callback_has_waited(monkeypatch):
    payloads = []

    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(capacity.urllib.request, "urlopen",
                        lambda request, *, timeout: payloads.append(json.loads(request.data)) or Response())
    capacity.post_alert("https://alerts.example/hook", {"level": "critical", "mounts": [], "alerts": [
        {"code": "preparation_delivery_stalled", "severity": "page", "count": 2, "age_seconds": 5400,
         "threshold_seconds": 3600, "parked_count": 7}]})
    assert payloads[0]["page"] is True
    assert payloads[0]["summary"] == ("website preparation status: a callback undelivered for 1.5h "
                                      "(2 live, 7 parked)")
