"""A website preparation the broker stopped delivering is run again by the drain tick, on a timer.

The actual listener, claim, lease, ledger, status reader, terminal sweep and drain run; ``run_e2e``
stands in for the orchestrator and fails, blocks or completes as each case needs. Paid calls are
stood in for by stage errors; no provider is called.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_preparation_timed_retry.py
#   src/blueprint_pipeline/website_preparation_timed_retry_drain.py
#   src/blueprint_pipeline/pubsub_handoff_scene_operations.py
#   src/blueprint_pipeline/pubsub_handoff_listener.py
#   src/blueprint_pipeline/website_preparation_terminal.py

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from blueprint_pipeline import capture_original_owner_observer as observer
from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_preparation_terminal as terminal
from blueprint_pipeline import website_preparation_timed_retry as timed
from blueprint_pipeline import website_preparation_timed_retry_drain as drain
from blueprint_pipeline.common import PipelineError, write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.job_retry_policy import REFUSAL_RETRY_DELAYS_SECONDS
from blueprint_pipeline.website_preparation_status import (
    DELIVERY_FILE, STATUS_V6, read_preparation_status, retain_preparation_wakeup,
)
from tests.test_website_preparation_status import prepared_source  # noqa: F401
from tests.test_website_robot_fit_hold import _delivery

DEFECT = "clean_plate: unclassified_stage_defect"
BUDGET = "website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted"
WITHDRAWN = "website_control_scene-sponsorship_http_409:capture_rights_withdrawn"


def _raises(code):
    def run_e2e(**_):
        raise PipelineError(code)
    return run_e2e


@pytest.fixture
def job(prepared_source, monkeypatch):  # noqa: F811 - imported pytest fixture
    """A browser delivery whose first run failed with an unclassified error."""
    root, owner, _, _, _, _ = prepared_source
    monkeypatch.delenv("BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY", raising=False)
    raw = _delivery(root, owner, monkeypatch)
    storage = root.parents[4]
    runs = []

    def deliver(run_e2e=_raises(DEFECT)):
        try:
            return listener.process_handoff_payload(raw, storage_root=storage, provider="openai",
                                                    run_evaluation_prep=False, run_e2e=run_e2e)
        except PipelineError:
            return None

    def tick(now, run_e2e=_raises(DEFECT)):
        def counted(**kwargs):
            runs.append(kwargs)
            return run_e2e(**kwargs)
        return drain.reconcile_timed_retries(listener, storage_root=storage, now=now, process_args={
            "provider": "openai", "run_evaluation_prep": False, "run_e2e": counted})

    deliver()
    retain_preparation_wakeup(root)
    return SimpleNamespace(root=root, owner=owner, raw=raw, storage=storage, runs=runs, deliver=deliver, tick=tick)


def _ledger(job):
    return listener._read_job_ledger(job.root)


def _stopped(ledger):
    return terminal._parse_utc(ledger["updated_at"]).timestamp()


def _started(ledger, kind=None):
    return [row for row in ledger["attempt_history"] if row.get("status") == timed.STARTED_STATUS
            and (kind is None or row[timed.MARK]["kind"] == kind)]


def _rewrite(job, **fields):
    ledger = {**_ledger(job), **fields}
    write_json(job.root / listener.JOB_LEDGER_FILENAME, ledger)
    return ledger


def _grant(job, expires_at_epoch):
    grant = {"schema_version": "website_scene_sponsorship.v1", "scene_id": job.owner["scene_id"],
             "capture_id": job.owner["capture_id"], "expires_at_epoch": expires_at_epoch}
    grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    write_json(job.root / terminal.GRANT_FILE, grant)


def test_the_exact_bytes_are_kept_from_the_claim_and_survive_every_commit(job, monkeypatch):
    ledger = _ledger(job)
    record = ledger[timed.FIELD]
    assert ledger["status"] == "failed_retryable" and ledger["last_error"] == DEFECT
    assert base64.b64decode(record["payload_base64"], validate=True) == job.raw
    assert record["source_payload_sha256"] == ledger["source_payload_sha256"] == listener.payload_sha256(job.raw)
    assert record["producer_delivery_key"] == ledger["producer_delivery_key"]
    seen = {}

    def heartbeating(**_):
        # Kept from the claim on (a killed run never reaches the failure path) and through heartbeats.
        live = _ledger(job)
        seen["claimed"] = live[timed.FIELD]
        assert listener._heartbeat_job_lease(job.root, owner=live["lease_owner"], token=live["lease_token"],
                                             lease_seconds=900)
        seen["heartbeat"] = _ledger(job)[timed.FIELD]
        return {"status": "completed"}
    assert job.deliver(heartbeating)["status"] == "processed"
    assert seen == {"claimed": record, "heartbeat": record} and _ledger(job)[timed.FIELD] == record
    # The terminal sweep's overdue commit keeps it too.
    _rewrite(job, status="failed_retryable", last_error=DEFECT)
    far = datetime.now(timezone.utc) + timedelta(days=30)
    overdue = terminal.record_unretried_preparation(job.root, alive_since=far - timedelta(days=29), now=far)
    assert overdue is not None and overdue[timed.FIELD] == record


def test_the_bytes_never_reach_a_status_projection_or_a_log(job, caplog):
    import logging

    encoded = _ledger(job)[timed.FIELD]["payload_base64"]
    selectors = json.loads((job.root / DELIVERY_FILE).read_text())["selectors"]
    for version in ("website_preparation_status.v1", STATUS_V6):
        status = read_preparation_status(capture_root=job.root, selectors=selectors, schema_version=version)
        assert "payload_base64" not in json.dumps(status) and encoded not in json.dumps(status)
    assert encoded not in (job.root / DELIVERY_FILE).read_text()
    with caplog.at_level(logging.DEBUG):
        job.tick(_stopped(_ledger(job)) + 900)
    assert len(job.runs) == 1 and encoded not in caplog.text


def test_the_six_timed_retries_follow_the_schedule_and_then_stop(job):
    for index, delay in enumerate(REFUSAL_RETRY_DELAYS_SECONDS):
        ledger = _ledger(job)
        stopped = _stopped(ledger)
        assert job.tick(stopped + delay - 1) == {"retried": 0, "waiting": 1} and len(job.runs) == index
        assert job.tick(stopped + delay)["retried"] == 1 and len(job.runs) == index + 1
        after = _ledger(job)
        assert after["attempt_count"] == ledger["attempt_count"] + 1 and after["status"] == "failed_retryable"
        row = _started(after)[-1]
        assert row[timed.MARK]["kind"] == timed.REFUSAL and row[timed.MARK]["retry"] == index + 1
        assert row["attempt_number"] == after["attempt_count"]
        # Append only: the earlier rows stand as they were.
        assert after["attempt_history"][:len(ledger["attempt_history"])] == ledger["attempt_history"]
    spent = _ledger(job)
    _grant(job, _stopped(spent) + 10**7)  # a window long enough that only the spent schedule stops it
    assert timed.timed_plan(job.root, spent, _stopped(spent) + 10**6) is None
    assert job.tick(_stopped(spent) + 10**6)["retried"] == 0 and len(job.runs) == 6


def _history_with(job, *runs):
    """The ledger's history, extended by synthetic timed attempts of earlier runs of the job."""
    ledger = _ledger(job)
    rows = []
    for count, closing in runs:
        rows += [{"status": timed.STARTED_STATUS, timed.MARK: {"kind": timed.REFUSAL}}] * count
        if closing:
            rows.append(closing)
    return _rewrite(job, attempt_history=[*ledger["attempt_history"][:1], *rows, ledger["attempt_history"][-1]])


def test_the_lifetime_cap_holds_across_runs_and_only_progress_resets_the_schedule(job):
    from blueprint_pipeline.website_task_target_choice import CORRECTION_MARK

    completed = {"status": "completed"}
    plan = timed.timed_plan(job.root, _history_with(job, (6, completed), (5, None)), 0)
    assert plan is not None and plan["retry"] == 6
    assert timed.timed_plan(job.root, _history_with(job, (6, completed), (6, None)), 0) is None
    # A correction by the site starts a new run of the job.
    corrected = {"status": "failed_retryable", CORRECTION_MARK: "sha256:" + "1" * 64}
    assert timed.timed_plan(job.root, _history_with(job, (6, corrected)), 0)["retry"] == 1
    # Twelve in the ledger's life, whatever progress came between.
    assert timed.timed_plan(job.root, _history_with(job, (6, completed), (6, completed)), 0) is None
    # Broker attempts and failures in between reset nothing.
    failure = {"status": "failed_retryable", "error": DEFECT}
    assert timed.timed_plan(job.root, _history_with(job, (3, failure), (3, failure)), 0) is None


def test_a_money_refusal_asks_the_same_gate_again_every_hour_and_is_never_counted(job):
    assert job.deliver(_raises(BUDGET)) is None
    for _ in range(len(REFUSAL_RETRY_DELAYS_SECONDS) + 2):
        stopped = _stopped(_ledger(job))
        assert job.tick(stopped + timed.MONEY_RETRY_SECONDS - 1)["retried"] == 0
        assert job.tick(stopped + timed.MONEY_RETRY_SECONDS, _raises(BUDGET))["retried"] == 1
    ledger = _ledger(job)
    assert len(_started(ledger, "money")) == 8 and not _started(ledger, timed.REFUSAL)
    assert timed.timed_plan(job.root, ledger, _stopped(ledger))["kind"] == "money"


def test_a_privacy_or_rights_refusal_is_never_retried_by_the_clock(job):
    assert job.deliver(_raises(WITHDRAWN)) is None
    ledger = _ledger(job)
    assert ledger["status"] == "failed_retryable" and timed.classify(ledger) == "privacy_rights"
    assert timed.timed_plan(job.root, ledger, _stopped(ledger) + 86400) is None
    assert job.tick(_stopped(ledger) + 86400)["retried"] == 0 and job.runs == []


def test_a_capacity_shortage_backs_off_as_an_outage_without_spending_retries(job):
    def full(**_):
        raise listener.HandoffStagingCapacityError("pubsub_handoff_staging_capacity_insufficient")
    job.deliver(full)
    ledger = _ledger(job)
    assert ledger["status"] == "retryable_blocked" and timed.classify(ledger) == timed.CAPACITY
    for delay in (120, 240, 480):
        stopped = _stopped(_ledger(job))
        assert job.tick(stopped + delay - 1)["retried"] == 0
        assert job.tick(stopped + delay, full)["retried"] == 1
    ledger = _ledger(job)
    assert len(_started(ledger, timed.CAPACITY)) == 3 and not _started(ledger, timed.REFUSAL)


def test_an_attempt_that_stops_before_its_claim_backs_off_and_spends_nothing(job, monkeypatch):
    before = _ledger(job)
    first = _stopped(before) + 900
    monkeypatch.setattr(observer, "load_original_owner_observation",
                        lambda **_: (_ for _ in ()).throw(ValueError("capture_owner_response_unavailable")))
    stamps = []
    for at in (first, first + 120, first + 120 + 240):
        assert job.tick(at - 1 if stamps else at)["retried"] == (0 if stamps else 1)
        if stamps:
            assert job.tick(at)["retried"] == 1
        stamps.append(json.loads((job.root / "pipeline/website_scene_preparation" / timed.STATE_FILE).read_text()))
    assert [stamp["unchanged_attempts"] for stamp in stamps] == [1, 2, 3]
    assert "payload_base64" not in json.dumps(stamps)
    assert _ledger(job) == before and job.runs == []
    # Within the backoff, a tick a minute later drives nothing.
    assert job.tick(first + 120 + 240 + 60)["retried"] == 0


def test_the_drive_after_a_pre_claim_outage_takes_the_first_retry(job, monkeypatch):
    first = _stopped(_ledger(job)) + 900
    real = observer.load_original_owner_observation
    monkeypatch.setattr(observer, "load_original_owner_observation",
                        lambda **_: (_ for _ in ()).throw(ValueError("capture_owner_response_unavailable")))
    job.tick(first)
    monkeypatch.setattr(observer, "load_original_owner_observation", real)
    assert job.tick(first + 120)["retried"] == 1 and len(job.runs) == 1
    assert _started(_ledger(job))[-1][timed.MARK]["retry"] == 1


def _killed(job, *, started_ago):
    ledger = _ledger(job)
    status, dead = listener._claim_job_lease(job.root, scene_id=ledger["scene_id"], capture_id=ledger["capture_id"],
        owner="control-plane:4242:killed", lease_seconds=listener.DEFAULT_JOB_LEASE_SECONDS,
        now=datetime.now(timezone.utc) - started_ago, payload_sha256=ledger["source_payload_sha256"],
        producer_delivery_key=ledger["producer_delivery_key"])
    assert status == "claimed" and dead[timed.FIELD] == ledger[timed.FIELD]
    return dead


def test_a_dead_workers_lease_is_taken_over_only_after_it_expires(job):
    import time

    live = _killed(job, started_ago=timedelta(minutes=1))
    assert timed.timed_plan(job.root, live, time.time()) is None
    assert job.tick(time.time())["retried"] == 0
    # A clock that runs ahead of the lease still meets the claim's own check of it.
    job.tick(10**10, lambda **_: pytest.fail("a live lease was taken over"))
    assert _ledger(job)["lease_token"] == live["lease_token"]
    dead = _killed(job, started_ago=timedelta(hours=1))
    expires = terminal._parse_utc(dead["lease_expires_at"]).timestamp()
    assert job.tick(expires + 899)["retried"] == 0
    assert job.tick(expires + 900, lambda **_: {"status": "completed"})["retried"] == 1
    after = _ledger(job)
    assert after["status"] == "completed" and after["recovered_expired_lease"] is True
    assert after["previous_lease_owner"] == "control-plane:4242:killed"
    assert _started(after)[-1][timed.MARK]["retry"] == 1


def test_a_killed_timed_attempt_still_counts(job):
    stopped = _stopped(_ledger(job))

    def killed(**_):
        # The process dies: nothing after the claim is recorded.
        raise SystemExit
    with pytest.raises(SystemExit):
        job.tick(stopped + 900, killed)
    dead = _ledger(job)
    assert dead["status"] == "processing" and len(_started(dead)) == 1
    _rewrite(job, lease_expires_at=listener._iso_at(datetime.now(timezone.utc) - timedelta(seconds=1)))
    assert timed.timed_plan(job.root, _ledger(job), _stopped(_ledger(job)) + 86400)["retry"] == 2


def test_a_broker_redelivery_during_a_timed_run_finds_the_lease(job):
    seen = {}

    def running(**_):
        seen["broker"] = job.deliver(lambda **__: pytest.fail("two runs at once"))
        return {"status": "completed"}
    assert job.tick(_stopped(_ledger(job)) + 900, running)["retried"] == 1
    assert seen["broker"]["status"] == "lease_active_retryable" and _ledger(job)["status"] == "completed"


def test_a_broker_claim_between_the_read_and_the_timed_claim_wins_alone(job, monkeypatch):
    before = _ledger(job)
    claim = listener._claim_job_lease

    def raced(*args, **kwargs):
        if kwargs.get("expected_timed_retry") is not None:
            assert job.deliver() is None  # the broker's redelivery runs and fails first
        return claim(*args, **kwargs)
    monkeypatch.setattr(listener, "_claim_job_lease", raced)
    job.tick(_stopped(before) + 900, lambda **_: pytest.fail("the timed retry ran on a moved ledger"))
    after = _ledger(job)
    assert after["attempt_count"] == before["attempt_count"] + 1 and not _started(after)


def test_a_grant_whose_own_expiry_passed_is_not_retried_and_the_sweep_still_ends_it(job, monkeypatch):
    ledger = _ledger(job)
    stopped = _stopped(ledger)
    _grant(job, stopped + 10**6)
    owed = timed.timed_plan(job.root, ledger, stopped)
    assert owed is not None
    # While a retry is owed the job may still move on its own: silence starts at its due time.
    current = datetime.fromtimestamp(stopped, tz=timezone.utc)
    assert terminal._silent_since(ledger, current, job.root).timestamp() == owed["due_at_epoch"]
    assert terminal._silent_since(ledger, current) == terminal._parse_utc(ledger["updated_at"])
    _grant(job, stopped - 1)
    assert timed.timed_plan(job.root, ledger, stopped) is None
    assert job.tick(stopped + 900)["retried"] == 0 and job.runs == []
    # No retry is owed, so the sweep's silence is the attempt's own, as before.
    assert terminal._silent_since(ledger, current, job.root) == terminal._parse_utc(ledger["updated_at"])
    monkeypatch.setattr(terminal, "_webapp_ended", lambda *_: ValueError(
        "website_control_scene-sponsorship_http_409:consent_expired"))
    now = current + timedelta(seconds=terminal.silence_seconds() + 60)
    ended = terminal.end_expired_preparation(job.root, alive_since=current - timedelta(hours=1),
                                             listener=listener, now=now)
    assert ended["status"] == listener.TERMINAL_AUTHORITY_STATUS and ended["terminal_code"] == "consent_expired"


def test_the_overdue_record_names_the_owed_retry(job):
    ledger = _ledger(job)
    due = timed.timed_plan(job.root, ledger, _stopped(ledger))["due_at_epoch"]
    stopped = datetime.fromtimestamp(_stopped(ledger), tz=timezone.utc)
    window = timedelta(seconds=terminal.silence_seconds())
    assert terminal.record_unretried_preparation(job.root, alive_since=stopped, now=stopped + window + timedelta(
        seconds=1)) is None  # owed in 15 minutes: not overdue at the plain silence window
    late = datetime.fromtimestamp(due, tz=timezone.utc) + window + timedelta(seconds=1)
    overdue = terminal.record_unretried_preparation(job.root, alive_since=stopped, now=late)
    record = terminal.current_overdue(overdue)
    assert record["next_retry_at"] == terminal._iso(datetime.fromtimestamp(due, tz=timezone.utc))


@pytest.mark.parametrize("change", ["retired", "withdrawn"])
def test_a_retirement_or_withdrawal_after_the_read_never_runs(job, monkeypatch, change):
    before = _ledger(job)
    if change == "retired":
        from blueprint_pipeline import website_scene_workspace_retention as retention
        monkeypatch.setattr(retention, "retired_capture_status", lambda **_: {
            "producer_delivery_keys": [before["producer_delivery_key"]], "payload_sha256s": [],
            "queue_disposition": "terminal_success", "receipt": {"retired": True},
            "status": listener.TERMINAL_AUTHORITY_STATUS})
    else:
        from blueprint_pipeline import consent_takedown
        monkeypatch.setattr(consent_takedown, "read_consent_state", lambda *_: {"state": "revoked"})
    job.tick(_stopped(before) + 900)
    assert job.runs == [] and _ledger(job) == before


def test_an_ended_delivery_key_is_never_retried(job):
    ledger = _ledger(job)
    ended = _rewrite(job, attempt_history=[*ledger["attempt_history"], {
        "status": listener.TERMINAL_AUTHORITY_STATUS, "producer_delivery_key": ledger["producer_delivery_key"]}])
    assert timed.timed_plan(job.root, ended, _stopped(ended) + 86400) is None


@pytest.mark.parametrize("change", ["absent", "schema", "digest", "fields"])
def test_a_ledger_without_a_valid_record_is_left_as_it_is(job, change):
    ledger = _ledger(job)
    record = dict(ledger.pop(timed.FIELD))
    if change == "schema":
        ledger[timed.FIELD] = {**record, "schema_version": "other.v1"}
    elif change == "digest":
        ledger[timed.FIELD] = {**record, "source_payload_sha256": "f" * 64}
    elif change == "fields":
        ledger[timed.FIELD] = {**record, "extra": True}
    write_json(job.root / listener.JOB_LEDGER_FILENAME, ledger)
    assert timed.timed_plan(job.root, ledger, _stopped(ledger) + 86400) is None
    assert job.tick(_stopped(ledger) + 86400)["retried"] == 0 and _ledger(job) == ledger


@pytest.mark.parametrize("status", ["completed", listener.TERMINAL_AUTHORITY_STATUS, "corrupt"])
def test_settled_or_corrupt_ledgers_are_never_driven(job, status):
    ledger = _rewrite(job, status=status)
    assert timed.timed_plan(job.root, ledger, _stopped(ledger) + 86400) is None


def test_ledgers_another_reconciler_owns_are_never_driven(job, monkeypatch):
    from blueprint_pipeline import website_reconstruction_resume as reconstruction
    from blueprint_pipeline import website_robot_fit_hold as fit
    from blueprint_pipeline import website_task_target_choice as choice

    base = _ledger(job)
    at = _stopped(base) + 86400
    record = base[timed.FIELD]
    # The assessment wait.
    assert timed.timed_plan(job.root, {**base, "assessment_resume": {"schema_version": "x"}}, at) is None
    # An armed Atlas wait.
    armed = {**base, "last_error": reconstruction.WAIT_ERROR,
             reconstruction.FIELD: {"schema_version": reconstruction.SCHEMA, "error": reconstruction.WAIT_ERROR}}
    assert reconstruction.armed(armed) and timed.timed_plan(job.root, armed, at) is None
    # A correction of the target the drain still owes a replay.
    (job.root / "pipeline" / choice.QUESTION_DIR).mkdir(parents=True, exist_ok=True)
    corrected = {**base, choice.REPLAY_FIELD: {**record, "schema_version": choice.REPLAY_SCHEMA}}
    monkeypatch.setattr(choice, "correction_pending", lambda *_: "sha256:" + "2" * 64)
    assert timed.timed_plan(job.root, corrected, at) is None
    monkeypatch.setattr(choice, "_replay_due", lambda *_: False)  # its own replays are spent
    assert timed.timed_plan(job.root, corrected, at) is not None
    # A robot fit recheck that stopped short and still has replays left.
    rows = [*base["attempt_history"][:-1], {**base["attempt_history"][-1], fit.RECHECK_MARK: {}}]
    rechecked = {**base, "attempt_history": rows,
                 fit.FIELD: {**record, "schema_version": fit.SCHEMA, "recheck_inputs": {}}}
    assert fit._rerun_stopped_short(rechecked) and timed.robot_fit_stopped_short(rechecked)
    spent = drain.robot_fit_spent
    assert timed.timed_plan(job.root, rechecked, at, robot_fit_spent=spent) is None
    monkeypatch.setattr(fit, "_due", lambda *_: False)  # its own replays for these inputs are spent
    assert timed.timed_plan(job.root, rechecked, at, robot_fit_spent=spent) is not None
    # Without the catalog (the terminal sweep), a stopped-short recheck stays the robot fit drain's.
    assert timed.timed_plan(job.root, rechecked, at) is None
    # And the claim refuses a ledger an owned wait took over after the read.
    status, _ = listener._claim_job_lease(job.root, scene_id=base["scene_id"], capture_id=base["capture_id"],
        owner="x", lease_seconds=900, payload_sha256=base["source_payload_sha256"],
        producer_delivery_key=base["producer_delivery_key"],
        expected_timed_retry={"revision": base["revision"] - 1, "record": record, "kind": timed.REFUSAL})
    assert status == "timed_retry_changed" and _ledger(job) == base


def test_an_unreadable_capture_does_not_starve_another(job):
    ledger = _ledger(job)
    other = job.storage / job.root.parents[3].name / "scenes/site-0000/captures/walkthrough-0000"
    write_json(other / listener.JOB_LEDGER_FILENAME, {**ledger, timed.FIELD: {
        **ledger[timed.FIELD], "payload_base64": "!!not base64!!"}})
    assert sorted([other, job.root])[0] == other
    assert job.tick(_stopped(ledger) + 900)["retried"] == 1 and len(job.runs) == 1


def test_the_existing_drain_tick_runs_the_timed_retries(monkeypatch, tmp_path):
    import google.cloud.pubsub_v1
    calls = []
    monkeypatch.setattr(drain, "reconcile_timed_retries", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(google.cloud.pubsub_v1, "SubscriberClient",
                        lambda: SimpleNamespace(pull=lambda **_: SimpleNamespace(received_messages=[])))
    listener.pull_and_process(subscription="projects/synthetic/subscriptions/handoff", storage_root=tmp_path,
        provider="openai", max_messages=1, run_evaluation_prep=False, run_e2e_enabled=False)
    assert len(calls) == 1 and calls[0]["limit"] == 1 and calls[0]["process_args"]["provider"] == "openai"


def test_the_robot_fit_names_match_without_importing_its_catalog():
    from blueprint_pipeline import website_robot_fit_hold as fit

    assert (timed.ROBOT_FIT_FIELD, timed.ROBOT_FIT_MARK) == (fit.FIELD, fit.RECHECK_MARK)



def _window():
    from blueprint_pipeline.website_job_window import WEBSITE_JOB_WINDOW_MAX_SECONDS
    return WEBSITE_JOB_WINDOW_MAX_SECONDS


def _deadline(ledger):
    return terminal._parse_utc(ledger["started_at"]).timestamp() + _window()


@pytest.mark.parametrize("kind", ["money", "capacity", "outage"])
def test_a_job_without_a_grant_stops_being_retried_a_window_after_its_first_claim(job, monkeypatch, kind):
    """The WebApp ends a no-grant job the longest window after its video arrived; the clock stops then too."""
    if kind == "money":
        job.deliver(_raises(BUDGET))
    elif kind == "capacity":
        def full(**_):
            raise listener.HandoffStagingCapacityError("pubsub_handoff_staging_capacity_insufficient")
        job.deliver(full)
    ledger = _ledger(job)
    assert not (job.root / terminal.GRANT_FILE).exists()
    deadline = _deadline(ledger)
    # Late in the window the job is still owed (an outage's stamp, a money hour, a capacity backoff) ...
    if kind == "outage":
        monkeypatch.setattr(observer, "load_original_owner_observation",
                            lambda **_: (_ for _ in ()).throw(ValueError("capture_owner_response_unavailable")))
        assert job.tick(deadline - 4000)["retried"] == 1 and _ledger(job) == ledger
    before = timed.timed_plan(job.root, _ledger(job), deadline - 1)
    assert before is not None and before["due_at_epoch"] <= deadline - 1
    assert job.tick(deadline - 1, _raises(BUDGET) if kind == "money" else _raises(DEFECT))["retried"] == 1
    # ... and after it, nothing is retried by the clock, whatever the kind.
    after = _ledger(job)
    assert timed.timed_plan(job.root, after, deadline) is None
    assert job.tick(deadline + 10**5)["retried"] == 0
    assert _ledger(job) == after
    # The terminal sweep then sees no retry owed, so its overdue observation is not put off.
    current = datetime.fromtimestamp(deadline + 1, tz=timezone.utc)
    assert terminal._silent_since(after, current, job.root) == terminal._silent_since(after, current)


def test_a_valid_grant_keeps_its_own_window_whatever_the_ledger_age(job):
    ledger = _rewrite(job, started_at="2020-01-01T00:00:00Z")
    stopped = _stopped(ledger)
    assert timed.timed_plan(job.root, ledger, stopped + 900) is None  # no grant: long past the local deadline
    _grant(job, stopped + 10**6)
    assert timed.timed_plan(job.root, ledger, stopped + 900) is not None
    assert job.tick(stopped + 900)["retried"] == 1


@pytest.mark.parametrize("started_at", [None, "", "not a time"])
def test_a_job_without_a_grant_or_a_readable_start_is_never_retried(job, started_at):
    ledger = _rewrite(job, started_at=started_at)
    assert timed.timed_plan(job.root, ledger, _stopped(ledger) + 900) is None
    assert job.tick(_stopped(ledger) + 900)["retried"] == 0 and job.runs == []


def test_an_invalid_grant_counts_as_no_grant(job):
    ledger = _rewrite(job, started_at="2020-01-01T00:00:00Z")
    _grant(job, _stopped(ledger) + 10**6)
    grant = json.loads((job.root / terminal.GRANT_FILE).read_text())
    write_json(job.root / terminal.GRANT_FILE, {**grant, "capture_id": "walkthrough-other"})
    assert timed.timed_plan(job.root, ledger, _stopped(ledger) + 900) is None


def test_no_claim_rewrites_the_first_claim_time(job):
    first = _ledger(job)["started_at"]
    job.tick(_stopped(_ledger(job)) + 900)
    job.deliver(lambda **_: {"status": "completed"})
    assert _ledger(job)["started_at"] == first and _ledger(job)["attempt_count"] == 3
