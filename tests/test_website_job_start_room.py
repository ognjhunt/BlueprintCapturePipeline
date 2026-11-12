# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_job_start_room.py
#   src/blueprint_pipeline/pubsub_handoff_scene_operations.py
#   src/blueprint_pipeline/website_assessment_resume.py
#   src/blueprint_pipeline/website_preparation_terminal.py
"""A website job starts its paid work only when the month has room for a whole typical job (owner 2026-11-11)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from blueprint_pipeline import account_monthly_spend as account
from blueprint_pipeline import website_job_start_room as start
from blueprint_pipeline.job_retry_policy import MONEY
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionBlocked
from blueprint_pipeline.website_preparation_retry import classify, retry_delay
from blueprint_pipeline.website_preparation_status import _v2_state
from blueprint_pipeline.website_preparation_terminal import capacity_only

NOW = datetime(2026, 11, 20, 16, 0, tzinfo=timezone.utc).timestamp()
GRANT = {"schema_version": "website_scene_sponsorship.v1", "expires_at_epoch": NOW + 86_400, "authority_digest": "x"}


def switch_to(tmp_path, monkeypatch, mode):
    """Run with the website money switch (``website_job_limits``) at ``mode``."""
    from blueprint_pipeline import website_job_limits

    path = tmp_path / f"website_job_limits.{mode}.json"
    path.write_text(json.dumps({"schema_version": website_job_limits.SCHEMA, "mode": mode}))
    monkeypatch.setenv(website_job_limits.CONFIG_ENV, str(path))


@pytest.fixture(autouse=True)
def _fuse(tmp_path, monkeypatch):
    """These tests are the 2026-11-11 model: the switch at ``fuse``."""
    switch_to(tmp_path, monkeypatch, "fuse")


def _host(tmp_path: Path, *, counted: str, limit: str = "300", jobs=()) -> dict[str, str]:
    """A limit file and a fresh report of the month, as the spend guard writes them."""
    report_path = tmp_path / "account" / "report.json"
    policy_path = tmp_path / "account_monthly_spend_limit.v1.json"
    policy_path.write_text(json.dumps({"schema_version": account.POLICY_SCHEMA, "monthly_limit_usd": limit,
                                       "monthly_limit_overrides": {}, "report_path": str(report_path)}))
    policy = account.load_policy(policy_path)
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps({
        "schema_version": account.REPORT_SCHEMA, "cap_status": "set", "policy_sha256": policy["policy_sha256"],
        "limit_usd": limit, "counted_usd": float(counted), "period": "2026-11",
        "generated_at": datetime.fromtimestamp(NOW - 60, timezone.utc).isoformat(),
        "meter": {"jobs": [{"job_id": job_id, "metered_usd": spent, "admissions": admissions}
                           for job_id, spent, admissions in jobs]}}))
    os.chmod(report_path, 0o644)
    return {account.CONFIG_ENV: str(policy_path), "BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT": str(tmp_path / "queue")}


def _capture(tmp_path: Path, request_id: str, capture_id: str = "walkthrough-1") -> Path:
    root = tmp_path / "storage" / "bucket" / "scenes" / f"site-{request_id}" / "captures" / capture_id
    (root / "pipeline").mkdir(parents=True)
    return root


def _started(tmp_path: Path, request_id: str, *, grant_expires: float = NOW + 86_400) -> Path:
    root = _capture(tmp_path, request_id)
    (root.parent.parent / start.START_FILE).write_text(json.dumps({"schema_version": start.START_SCHEMA}))
    (root / start.GRANT_FILE).write_text(json.dumps({**GRANT, "expires_at_epoch": grant_expires}))
    return root


def test_the_typical_job_is_an_estimate_of_70_dollars():
    assert start.TYPICAL_JOB_USD == Decimal("70")


def test_the_screening_names_it_reads_are_the_coordinators_own():
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from blueprint_pipeline.beta_policy_screening_result import ACTIVE_STATUSES, QUEUED_STATUSES

    assert start.SCREENING_LEDGER_FILENAME == coordinator.LEDGER_FILENAME
    assert start.SCREENING_OPEN_STATUSES == QUEUED_STATUSES | ACTIVE_STATUSES


def test_a_job_starts_when_the_month_has_room_for_a_whole_typical_job(tmp_path):
    env = _host(tmp_path, counted="160.00")
    root = _capture(tmp_path, "new")
    # $300 - $160 = $140 with nothing else started: room for a $70 job.
    record = start.require_job_start_room(capture_root=root, authority=GRANT, now=NOW, environ=env)
    assert record["basis"] == "room_checked" and Decimal(record["room_usd"]) == Decimal("140.00")
    assert json.loads((root.parent.parent / start.START_FILE).read_text()) == record


def test_a_job_waits_when_started_jobs_will_take_the_room_and_keeps_its_grant_for_the_deadline(tmp_path):
    # $300 - $160 counted - ($70 - $20 already metered for the started job) = $90 >= $70: it may start ...
    env = _host(tmp_path, counted="160.00", jobs=[("busy", 20.0, 3)])
    _started(tmp_path, "busy")
    assert start.require_job_start_room(capture_root=_capture(tmp_path, "first"), now=NOW, environ=env)
    # ... and now two started jobs reserve $50 + $70: $300 - $160 - $120 = $20 < $70, so the next one waits.
    root = _capture(tmp_path, "second")
    with pytest.raises(PaidResourceAdmissionBlocked) as refused:
        start.require_job_start_room(capture_root=root, authority=GRANT, now=NOW, environ=env)
    assert refused.value.blockers == [start.START_WAIT]
    assert not (root.parent.parent / start.START_FILE).exists()
    assert json.loads((root / start.GRANT_FILE).read_text()) == GRANT


def test_a_started_job_is_never_stopped_by_the_check(tmp_path):
    env = _host(tmp_path, counted="299.00")
    root = _started(tmp_path, "running")
    assert start.require_job_start_room(capture_root=root, now=NOW, environ=env) == {"schema_version": start.START_SCHEMA}
    # A job already spending this month when the check shipped counts as started too.
    (tmp_path / "other").mkdir()
    env = _host(tmp_path / "other", counted="299.00", jobs=[("inflight", 0.0, 2)])
    record = start.require_job_start_room(capture_root=_capture(tmp_path / "other", "inflight"), now=NOW, environ=env)
    assert record["basis"] == "metered_before_check"


def test_finished_jobs_reserve_nothing_and_are_never_read_again(tmp_path):
    env = _host(tmp_path, counted="160.00")
    _started(tmp_path, "expired", grant_expires=NOW - 1)
    tested = _started(tmp_path, "tested")
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "screening_ledger.v1.json").write_text(json.dumps({"campaigns": {"c1": {
        "site_id": tested.parent.parent.name, "candidates": {"a": {"status": "completed"}, "b": {"status": "deferred"}}}}}))
    record = start.require_job_start_room(capture_root=_capture(tmp_path, "new"), now=NOW, environ=env)
    assert record["started_jobs"] == [] and Decimal(record["room_usd"]) == Decimal("140.00")
    finished = json.loads((tested.parent.parent / start.START_FILE).read_text())
    assert finished["finished_reason"] == "screening_ended"
    assert json.loads((tmp_path / "storage/bucket/scenes/site-expired" / start.START_FILE).read_text())[
        "finished_reason"] == "grant_expired"
    # A finished record is not read again, even if a row of its site reopens; the job that just started
    # (no campaign yet, its grant open) keeps its whole remainder reserved.
    (queue / "screening_ledger.v1.json").write_text(json.dumps({"campaigns": {"c1": {
        "site_id": tested.parent.parent.name, "candidates": {"a": {"status": "waitlisted"}}}}}))
    (tmp_path / "storage/bucket/scenes/site-new/captures/walkthrough-1" / start.GRANT_FILE).write_text(json.dumps(GRANT))
    reserve, rows = start.started_job_reserve(tested.parent.parent.parent, exclude="site-x", meter={}, environ=env,
                                              now=NOW)
    assert reserve == Decimal("70") and [row["scene_id"] for row in rows] == ["site-new"]


def test_the_reviewers_three_dead_jobs_reserve_nothing(tmp_path):
    """Review 2026-11-11: two jobs with open rows and long-expired grants, one with no grant at all."""
    scenes, queue = tmp_path / "scenes", tmp_path / "q"
    queue.mkdir()
    for index in range(2):
        capture = scenes / f"site-job{index}" / "captures/cap/pipeline"
        capture.mkdir(parents=True)
        (capture / "website_scene_sponsorship.json").write_text(json.dumps({"expires_at_epoch": 1.0}))
        (scenes / f"site-job{index}" / start.START_FILE).write_text(json.dumps({"schema_version": start.START_SCHEMA}))
    (scenes / "site-job2" / "captures/cap").mkdir(parents=True)
    (scenes / "site-job2" / start.START_FILE).write_text("{}")
    (queue / "screening_ledger.v1.json").write_text(json.dumps({"campaigns": {
        "a": {"site_id": "site-job0", "candidates": {"x": {"status": "reconciliation_required"}}},
        "b": {"site_id": "site-job1", "candidates": {"x": {"status": "waitlisted"}}}}}))
    env = {"BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT": str(queue)}
    assert start.started_job_reserve(scenes, exclude="site-new", meter={}, environ=env, now=4e9) == (Decimal(0), [])


def test_a_job_running_across_a_month_boundary_keeps_what_it_spent_before(tmp_path):
    scenes = tmp_path / "scenes"
    (scenes / "site-busy" / "captures/cap/pipeline").mkdir(parents=True)
    env = {"BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT": str(tmp_path / "q")}
    november = datetime(2026, 11, 29, tzinfo=timezone.utc).timestamp()
    december = datetime(2026, 12, 2, tzinfo=timezone.utc).timestamp()
    (scenes / "site-busy" / start.START_FILE).write_text(json.dumps(
        {"schema_version": start.START_SCHEMA, "started_at_epoch": november,
         "grant": {"expires_at_epoch": december + 86_400}}))
    assert start.started_job_reserve(scenes, exclude="x", meter={"busy": (Decimal(50), 4)}, environ=env,
                                     now=november)[0] == Decimal(20)
    # A new month's meter shows nothing for it yet: it still reserves 70 - 50, never the whole 70 again.
    assert start.started_job_reserve(scenes, exclude="x", meter={}, environ=env, now=december)[0] == Decimal(20)
    assert start.started_job_reserve(scenes, exclude="x", meter={"busy": (Decimal(5), 1)}, environ=env,
                                     now=december)[0] == Decimal(15)


def test_a_reserve_ends_at_the_grant_whatever_its_rows_say_or_at_the_bound_without_a_grant(tmp_path):
    scenes, queue = tmp_path / "scenes", tmp_path / "q"
    queue.mkdir()
    (queue / "screening_ledger.v1.json").write_text(json.dumps({"campaigns": {
        "a": {"site_id": "site-open", "candidates": {"x": {"status": "running"}}}}}))
    env = {"BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT": str(queue)}
    for name, record in (("open", {"grant": {"expires_at_epoch": NOW + 10, "valid_until": NOW + 100}}),
                         ("nogrant", {})):
        (scenes / f"site-{name}" / "captures").mkdir(parents=True)
        (scenes / f"site-{name}" / start.START_FILE).write_text(json.dumps(
            {"schema_version": start.START_SCHEMA, "started_at_epoch": NOW, **record}))
    assert start.started_job_reserve(scenes, exclude="x", meter={}, environ=env, now=NOW + 50)[0] == Decimal(140)
    # The owner's extension (valid_until) ends it, open rows or not; the bound ends the one with no grant.
    later = NOW + start.NO_GRANT_BOUND_SECONDS
    assert start.started_job_reserve(scenes, exclude="x", meter={}, environ=env, now=later)[0] == Decimal(0)
    assert {json.loads((scenes / f"site-{name}" / start.START_FILE).read_text())["finished_reason"]
            for name in ("open", "nogrant")} == {"grant_expired", "no_grant_bound_reached"}
    assert start.NO_GRANT_BOUND_SECONDS == 14 * 86_400


def test_a_job_the_meter_shows_spending_gets_a_record_with_the_same_end(tmp_path):
    scenes = tmp_path / "scenes"
    (scenes / "site-old" / "captures").mkdir(parents=True)
    env = {"BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT": str(tmp_path / "q")}
    assert start.started_job_reserve(scenes, exclude="x", meter={"old": (Decimal(30), 2)}, environ=env,
                                     now=NOW)[0] == Decimal(40)
    assert json.loads((scenes / "site-old" / start.START_FILE).read_text())["basis"] == "metered_before_check"
    assert start.started_job_reserve(scenes, exclude="x", meter={}, environ=env,
                                     now=NOW + start.NO_GRANT_BOUND_SECONDS)[0] == Decimal(0)


def test_waiting_jobs_feed_one_warn_with_how_many_and_the_oldest(tmp_path):
    from blueprint_pipeline.account_monthly_spend import JOBS_WAITING_CODE, jobs_waiting_alert

    env = _host(tmp_path, counted="290.00")
    for index, at in enumerate((NOW - 7_200, NOW - 60)):
        with pytest.raises(PaidResourceAdmissionBlocked):
            start.require_job_start_room(capture_root=_capture(tmp_path, f"w{index}"), now=at, environ=env)
    # The first one is refused again later: its wait keeps its first moment.
    with pytest.raises(PaidResourceAdmissionBlocked):
        start.require_job_start_room(capture_root=tmp_path / "storage/bucket/scenes/site-w0/captures/walkthrough-1",
                                     now=NOW - 30, environ=env)
    waits = start.waiting_jobs(tmp_path / "storage", now=NOW)
    assert waits == {"count": 2, "oldest_since_epoch": NOW - 7_200}
    [alert] = jobs_waiting_alert(waits, now=NOW, period="2026-11")
    assert alert["code"] == JOBS_WAITING_CODE and alert["count"] == 2 and alert["age_seconds"] == 7_200
    assert jobs_waiting_alert({"count": 0, "oldest_since_epoch": None}, now=NOW, period="2026-11") == []
    # A wait nobody re-asked for two hours (its job ended) no longer counts.
    assert start.waiting_jobs(tmp_path / "storage", now=NOW + 3 * 3_600)["count"] == 0


def test_a_grant_that_runs_out_while_the_job_waits_ends_with_its_own_code():
    waited = [{"stage": start.STAGE, "status": "failed_retryable"}]
    assert start.room_wait_ending("consent_expired", waited) == start.ROOM_WAIT_EXPIRED_CODE
    assert start.room_wait_ending("consent_expired", [{"stage": "run_e2e"}]) == "consent_expired"
    assert start.room_wait_ending("source_revoked", waited) == "source_revoked"
    ledger = {"status": "terminal_authority_ended", "terminal_code": start.ROOM_WAIT_EXPIRED_CODE}
    assert _v2_state(ledger) == ("failed_terminal", "capacity_blocked")
    assert _v2_state({**ledger, "terminal_code": "consent_expired"}) == ("authority_ended", None)


def test_with_the_limit_reached_the_job_waits_on_the_limit_by_its_own_name(tmp_path):
    env = _host(tmp_path, counted="300.00")
    with pytest.raises(PaidResourceAdmissionBlocked) as refused:
        start.require_job_start_room(capture_root=_capture(tmp_path, "new"), now=NOW, environ=env)
    assert refused.value.blockers == [account.LIMIT_REACHED]


def test_with_no_limit_configured_off_the_host_nothing_is_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(account, "limit_required", lambda environ=None: False)
    env = {account.CONFIG_ENV: str(tmp_path / "absent.json")}
    root = _capture(tmp_path, "new")
    assert start.require_job_start_room(capture_root=root, now=NOW, environ=env) is None
    assert not (root.parent.parent / start.START_FILE).exists()


def test_the_wait_is_a_money_wait_retried_until_the_deadline_and_reported_as_capacity():
    error = str(PaidResourceAdmissionBlocked([start.START_WAIT]))
    assert classify(error) == MONEY
    # Money waits back off to hourly and are ended by the grant's expiry, never by a count.
    assert retry_delay(MONEY, 50, deadline=True) == 3600
    ledger = {"status": "failed_retryable", "last_error": error, "retry_blockers": []}
    assert capacity_only(ledger)
    assert _v2_state(ledger) == ("capacity_delayed", None)


# ---- the real listener, ledger and retry tick (run_e2e and the WebApp are the retry tests' fakes) ----

from tests.test_website_preparation_retry import _due, _tick, failed  # noqa: E402,F401
from tests.test_website_preparation_status import prepared_source  # noqa: E402,F401
from tests.test_website_reconstruction_resume import waiting  # noqa: E402,F401


def test_the_listener_holds_a_new_job_before_run_e2e_and_starts_it_by_itself_when_room_appears(
        failed, tmp_path, monkeypatch):  # noqa: F811 - imported pytest fixture
    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline import website_preparation_retry as retry

    root, raw, _ = failed
    scene_dir = root.parent.parent
    (tmp_path / "host").mkdir()
    for name, value in _host(tmp_path / "host", counted="290.00").items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(start, "_now", lambda now: NOW)
    calls = []

    def run_e2e(**kwargs):
        calls.append(kwargs)
        return {"status": "completed"}
    with pytest.raises(PaidResourceAdmissionBlocked):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai", run_e2e=run_e2e)
    ledger = listener._read_job_ledger(root)
    # Nothing paid ran: the job waits before run_e2e, as a money wait the host's tick retries until its deadline.
    assert calls == [] and not (scene_dir / start.START_FILE).exists()
    assert ledger["status"] == "failed_retryable" and ledger["attempt_history"][-1]["stage"] == start.STAGE
    assert start.START_WAIT in ledger["last_error"] and retry.armed(ledger)["kind"] == MONEY
    assert _v2_state(ledger) == ("capacity_delayed", None)
    # A higher limit (or a new month) gives room: the next due tick starts it by itself.
    (tmp_path / "raised").mkdir()
    for name, value in _host(tmp_path / "raised", counted="290.00", limit="400").items():
        monkeypatch.setenv(name, value)
    assert _tick(root, run_e2e, _due(root))["redriven"] == 1
    assert len(calls) == 1 and json.loads((scene_dir / start.START_FILE).read_text())["basis"] == "room_checked"
    assert listener._read_job_ledger(root)["status"] == "completed"


def test_through_the_listener_a_grant_that_runs_out_while_waiting_ends_held_not_consent_expired(
        failed, tmp_path, monkeypatch):  # noqa: F811 - imported pytest fixture
    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline import website_task_context as context_reader

    root, raw, _ = failed
    (tmp_path / "host").mkdir()
    for name, value in _host(tmp_path / "host", counted="290.00").items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(start, "_now", lambda now: NOW)
    with pytest.raises(PaidResourceAdmissionBlocked):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
                                         run_e2e=lambda **_: pytest.fail("ran while waiting"))

    def expired(**_):
        raise ValueError("website_control_scene-sponsorship_http_409:consent_expired")
    monkeypatch.setattr(context_reader, "load_website_scene_sponsorship", expired)
    result = listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
                                              run_e2e=lambda **_: pytest.fail("ran after its grant ended"))
    assert result["status"] == listener.TERMINAL_AUTHORITY_STATUS
    ledger = listener._read_job_ledger(root)
    assert ledger["terminal_code"] == start.ROOM_WAIT_EXPIRED_CODE
    assert _v2_state(ledger) == ("failed_terminal", "capacity_blocked")
