# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_reconstruction_resume.py
#   src/blueprint_pipeline/pubsub_handoff_scene_operations.py
#   src/blueprint_pipeline/pubsub_handoff_listener.py
#   src/blueprint_pipeline/website_preparation_terminal.py
"""ADP-040/day 28: a capture waiting on its Atlas job is replayed when the job ends, never a second job.

The actual listener, lease, ledger and scene handoff run; the capture's Atlas
job is a retained job file, and run_e2e stands in only for the orchestrator's
stages before the handoff. No provider call.
"""
import base64
import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from blueprint_pipeline import capture_original_owner_observer as observer
from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_preparation_terminal as terminal
from blueprint_pipeline import website_reconstruction_resume as resume
from blueprint_pipeline import website_task_context as context_reader
from blueprint_pipeline.common import PipelineError, StageError, write_json
from blueprint_pipeline.task_evaluation_scene_retirement_generations import capture_birth_source_projection
from blueprint_pipeline.website_atlas_preview import MAXIMUM_JOB_SECONDS
from blueprint_pipeline.website_atlas_preview_store import JOB_FILE, SCHEMA as JOB_SCHEMA
from blueprint_pipeline.website_scene_handoff import prepare_website_scene_handoff
from tests.test_website_assessment_resume import WIRE_FIXTURE
from tests.test_website_preparation_status import prepared_source  # noqa: F401


def _job(root, state, **fields):
    write_json(root / "pipeline/atlas" / JOB_FILE, {"schema_version": JOB_SCHEMA, "state": state, **fields})


def _orchestrated(root, owner, context):
    """The real scene handoff, raised exactly as the capture orchestrator raises its blockers."""
    def run_e2e(**_):
        handoff = prepare_website_scene_handoff(
            descriptor={"scene_id": owner["scene_id"], "capture_id": owner["capture_id"],
                        "metadata": {"site_task_context": context}},
            clean_plate={"privacy_verified": True, "status": "noop"}, capture_root=root, now=0)
        error = StageError("website_scene_preparation", ",".join(handoff["blockers"]))
        raise PipelineError(str(error)) from error
    return run_e2e


@pytest.fixture
def waiting(prepared_source, monkeypatch):  # noqa: F811 - imported pytest fixture
    root, owner, context, _, _, _ = prepared_source
    birth = capture_birth_source_projection(root)
    payload = json.loads(WIRE_FIXTURE.read_text().replace("walkthrough-req-1", owner["capture_id"]))
    payload["source_finalize"] = {"bucket": owner["bucket"], "object_name": owner["completion_marker"]["object_name"],
        "generation": owner["completion_marker"]["generation"], "event_id": "original-event",
        "event_source": "original-source"}
    payload["source_membership_selector"] = birth["source_membership_selector"]
    payload["pipeline_status_event"]["source_finalize"] = payload["source_finalize"]
    payload["pipeline_handoff_uri"] = (f"gs://{owner['bucket']}/"
        f"{birth['source_membership_selector']['object_name'].rsplit('/', 1)[0]}/pipeline_handoff.json")
    payload["pipeline_status_event"]["pipeline_handoff_uri"] = payload["pipeline_handoff_uri"]
    raw = json.dumps(payload, indent=2).encode()
    # The reusable fixture's setup-only lease has run nothing; the processor makes its own.
    (root / listener.JOB_LEDGER_FILENAME).unlink()
    write_json(root / "raw/manifest.json", {"capture_source": "browser_self_capture"})
    monkeypatch.setattr(listener, "stage_handoff_capture", lambda *_, **__: root)
    proposal = {"schema_version": "site_assessment_preparation_proposal.v1", "request_id": owner["request_id"],
        "capture_id": owner["capture_id"], "job_id": "advisory-" + "a" * 64, "run_id": "site-assessment-test",
        "source_key": "sha256:" + "b" * 64, "context_digest": "c" * 64, "packet_sha256": "d" * 64,
        "questions_pending": True, "scope": "scene_preparation_only", "robot_suitability_verified": False,
        "physical_trial_authorized": False}
    monkeypatch.setattr(context_reader, "load_website_scene_sponsorship",
                        lambda **_: {"assessment_preparation_proposal": proposal})
    _job(root, "processing", stage="pose")
    with pytest.raises(PipelineError, match="website_atlas_job_processing"):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
                                         run_e2e=_orchestrated(root, owner, context))
    ledger = listener._read_job_ledger(root)
    record = ledger[resume.FIELD]
    assert ledger["status"] == "failed_retryable" and ledger["last_error"] == resume.WAIT_ERROR
    assert ledger["attempt_history"][0][resume.FIELD] == record and ledger["attempt_count"] == 1
    assert base64.b64decode(record["payload_base64"], validate=True) == raw
    return root, owner, context, raw, record


def test_drain_waits_while_the_atlas_job_processes_and_keeps_the_exact_bytes(waiting, monkeypatch):
    root = waiting[0]
    before = listener._read_job_ledger(root)
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *_, **__: pytest.fail("replayed before the scene"))
    assert resume.reconcile_waiting_reconstructions(listener, storage_root=root.parents[4],
                                                     process_args={"provider": "openai"}) == {"resumed": 0, "pending": 1}
    assert listener._read_job_ledger(root) == before


def test_a_redelivery_while_still_waiting_arms_again(waiting):
    root, owner, context, raw, _ = waiting
    with pytest.raises(PipelineError, match="website_atlas_job_processing"):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
                                         run_e2e=_orchestrated(root, owner, context))
    ledger = listener._read_job_ledger(root)
    assert ledger["attempt_count"] == 2 and resume.armed(ledger)


def test_reviewed_native_resume_goes_through_the_existing_processor(waiting):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    root, _, _, raw, _ = waiting
    before = listener._read_job_ledger(root)
    before.pop(resume.FIELD)
    before["last_error"] = "clean_plate: task_target_requires_clarification:cardboard_box_candidate"
    before["attempt_history"][-1].pop(resume.FIELD)
    before["attempt_history"][-1]["error"] = before["last_error"]
    write_json(root / listener.JOB_LEDGER_FILENAME, before)
    _job(root, "ready")
    calls = []
    result = listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
        run_evaluation_prep=False, expected_reconstruction_resume={"review_schema": resume.REVIEW_SCHEMA,
            "ledger_digest": canonical_digest(before), "selected_payload_sha256": listener.payload_sha256(raw)},
        run_e2e=lambda **kw: calls.append(kw) or {"status": "completed"})
    assert result["status"] == "processed" and len(calls) == 1
    assert calls[0]["resume_completed_stages"] is True
    after = listener._read_job_ledger(root)
    assert after["attempt_count"] == before["attempt_count"] + 1
    assert after["attempt_history"][0] == before["attempt_history"][0]


def test_a_stale_native_review_never_enters_the_pipeline(waiting):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    root, _, _, raw, _ = waiting
    before = listener._read_job_ledger(root)
    reviewed = canonical_digest(before)
    before["revision"] += 1
    write_json(root / listener.JOB_LEDGER_FILENAME, before)
    result = listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
        expected_reconstruction_resume={"review_schema": resume.REVIEW_SCHEMA,
            "ledger_digest": reviewed, "selected_payload_sha256": listener.payload_sha256(raw)},
        run_e2e=lambda **_: pytest.fail("stale operator review entered pipeline"))
    assert result["status"] == "reconstruction_resume_changed"
    assert listener._read_job_ledger(root) == before


@pytest.mark.parametrize("prepared_source", [None, "continuation"], indirect=True)
@pytest.mark.parametrize("state", ["ready", "failed", "needs_input"])
def test_a_final_atlas_job_replays_the_same_original_payload_once(waiting, state):
    root, _, _, raw, record = waiting
    _job(root, state)
    calls = []

    def run_e2e(**kwargs):
        calls.append(kwargs)
        return {"status": "completed"}
    process_args = {"provider": "openai", "run_evaluation_prep": False, "run_e2e": run_e2e}
    assert resume.reconcile_waiting_reconstructions(listener, storage_root=root.parents[4],
                                                     process_args=process_args) == {"resumed": 1, "pending": 0}
    ledger = listener._read_job_ledger(root)
    assert len(calls) == 1 and calls[0]["resume_completed_stages"] is True
    # The claim consumed the record; the failure row keeps the original bytes.
    assert ledger["status"] == "completed" and ledger["attempt_count"] == 2 and resume.FIELD not in ledger
    assert ledger["attempt_history"][0][resume.FIELD] == record
    assert resume.reconcile_waiting_reconstructions(listener, storage_root=root.parents[4],
                                                     process_args=process_args) == {"resumed": 0, "pending": 0}
    assert len(calls) == 1


def test_a_job_waiting_to_recheck_authority_is_not_final(waiting, monkeypatch):
    root = waiting[0]
    _job(root, "needs_input", retry_at_epoch=1)
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *_, **__: pytest.fail("replayed a live job"))
    assert resume.reconcile_waiting_reconstructions(listener, storage_root=root.parents[4],
                                                     process_args={"provider": "openai"}) == {"resumed": 0, "pending": 1}


REOPENABLE = "atlas_scene_planner_abstained"


def _replays(root):
    calls = []
    args = {"provider": "openai", "run_evaluation_prep": False,
            "run_e2e": lambda **kwargs: calls.append(kwargs) or {"status": "completed"}}
    return calls, lambda: resume.reconcile_waiting_reconstructions(listener, storage_root=root.parents[4],
                                                                   process_args=args)


def test_a_reopened_atlas_job_replays_the_preparation_once_after_ready_and_never_while_it_needs_input(waiting):
    """Armed while Atlas processed; Atlas stops for input; Blueprint reopens it; it turns ready."""
    from blueprint_pipeline.website_preparation_status import _v2_state
    root, owner, context, _, _ = waiting
    runs = []

    def orchestrated(**kwargs):
        runs.append("needs_input")
        return _orchestrated(root, owner, context)(**kwargs)
    drain = lambda run_e2e: resume.reconcile_waiting_reconstructions(  # noqa: E731
        listener, storage_root=root.parents[4],
        process_args={"provider": "openai", "run_evaluation_prep": False, "run_e2e": run_e2e})
    now = time.time()
    _job(root, "needs_input", reason=REOPENABLE, enqueued_at_epoch=now)
    # One replay records the needs-input failure: the status asks for Blueprint, never "preparing".
    drain(orchestrated)
    ledger = listener._read_job_ledger(root)
    assert runs == ["needs_input"] and ledger["last_error"] == resume.NEEDS_INPUT_ERROR and resume.armed(ledger)
    assert _v2_state(ledger) == ("failed_retryable", None)
    # Then it waits: no replay while Atlas still needs input, nor while the reopened job processes.
    assert drain(orchestrated) == {"resumed": 0, "pending": 1} and len(runs) == 1
    _job(root, "processing", reason=None, enqueued_at_epoch=now)  # reopened (``retry_failed_job``)
    assert drain(orchestrated) == {"resumed": 0, "pending": 1} and len(runs) == 1
    _job(root, "ready", enqueued_at_epoch=now)
    completed = lambda **_: runs.append("ready") or {"status": "completed"}  # noqa: E731
    assert drain(completed) == {"resumed": 1, "pending": 0} and runs == ["needs_input", "ready"]
    assert drain(completed) == {"resumed": 0, "pending": 0} and len(runs) == 2
    assert listener._read_job_ledger(root)["status"] == "completed"


@pytest.mark.parametrize("reason, armed", [(REOPENABLE, True), ("atlas_insufficient_overlapping_distinct_views", False)])
def test_a_run_that_meets_an_atlas_job_needing_input_waits_only_while_it_may_be_reopened(waiting, reason, armed):
    root, owner, context, raw, _ = waiting
    _job(root, "needs_input", reason=reason, enqueued_at_epoch=time.time())
    with pytest.raises(PipelineError, match="website_atlas_job_needs_input"):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
                                         run_e2e=_orchestrated(root, owner, context))
    ledger = listener._read_job_ledger(root)
    assert ledger["last_error"] == resume.NEEDS_INPUT_ERROR and resume.armed(ledger) is armed
    assert ledger["lease_token"] is None


def test_an_expired_or_withdrawn_capture_ends_the_wait(waiting, monkeypatch):
    from blueprint_pipeline import website_atlas_preview_store as store
    root = waiting[0]
    calls, drain = _replays(root)
    _job(root, "needs_input", reason=REOPENABLE, enqueued_at_epoch=time.time())
    monkeypatch.setattr(store, "capture_withdrawn", lambda _root: True)
    # Withdrawn: never replayed and never counted as waiting; the withdrawal's deletion removes it.
    assert drain() == {"resumed": 0, "pending": 0} and calls == []
    monkeypatch.setattr(store, "capture_withdrawn", lambda _root: False)
    # A retired status delivery ends the wait the same way.
    from blueprint_pipeline import website_preparation_status as status
    live = status.delivery_retired
    monkeypatch.setattr(status, "delivery_retired", lambda _root, _ledger: True)
    _job(root, "ready")
    assert drain() == {"resumed": 0, "pending": 0} and calls == []
    monkeypatch.setattr(status, "delivery_retired", live)
    # Past the job's seven days it can no longer be reopened: final, replayed once as before.
    _job(root, "needs_input", reason=REOPENABLE, enqueued_at_epoch=time.time() - MAXIMUM_JOB_SECONDS - 1)
    assert drain() == {"resumed": 1, "pending": 0} and len(calls) == 1
    assert resume.FIELD not in listener._read_job_ledger(root)


@pytest.mark.parametrize("prepared_source", [None, "continuation"], indirect=True)
@pytest.mark.parametrize("change", ["payload", "source", "active", "wrong_error", "owner", "context"])
def test_a_changed_source_ledger_or_authority_is_never_replayed(waiting, monkeypatch, change):
    root = waiting[0]
    _job(root, "ready")
    ledger = listener._read_job_ledger(root)
    if change == "payload":
        ledger[resume.FIELD]["payload_base64"] = base64.b64encode(b"{}").decode()
    elif change == "source":
        ledger["producer_delivery_key"] = "sha256:" + "e" * 64
    elif change == "active":
        ledger["status"] = "processing"
    elif change == "wrong_error":
        ledger["last_error"] = "unknown_provider_charge"
    elif change == "owner":
        monkeypatch.setattr(observer, "load_original_owner_observation",
                            lambda **_: (_ for _ in ()).throw(ValueError("source_revoked")))
    else:
        monkeypatch.setattr(context_reader, "load_current_website_task_context",
                            lambda **_: {**waiting[2], "context_digest": "sha256:" + "e" * 64})
    write_json(root / listener.JOB_LEDGER_FILENAME, ledger)
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *_, **__: pytest.fail("unadmitted replay"))
    assert resume.reconcile_waiting_reconstructions(listener, storage_root=root.parents[4],
                                                     process_args={"provider": "openai"})["resumed"] == 0


def test_a_ledger_that_moves_on_before_the_claim_refuses_the_replay(waiting, monkeypatch):
    root = waiting[0]
    _job(root, "ready")
    claim = listener._claim_job_lease

    def raced(*args, **kwargs):
        assert kwargs["expected_reconstruction_resume"]["record"] == waiting[4]
        prior = listener._read_job_ledger(root)
        prior.pop(resume.FIELD)
        prior.update(revision=prior["revision"] + 2, last_error="unknown_provider_charge")
        write_json(root / listener.JOB_LEDGER_FILENAME, prior)
        return claim(*args, **kwargs)
    monkeypatch.setattr(listener, "_claim_job_lease", raced)
    result = resume.reconcile_waiting_reconstructions(listener, storage_root=root.parents[4], process_args={
        "provider": "openai", "run_e2e": lambda **_: pytest.fail("stale replay entered provider stages")})
    assert result["resumed"] == 0
    ledger = listener._read_job_ledger(root)
    assert ledger["last_error"] == "unknown_provider_charge" and ledger["attempt_count"] == 1


def test_atlas_wait_becomes_overdue_retryable_only_after_the_job_window(waiting):
    root = waiting[0]
    stopped = terminal._parse_utc(listener._read_job_ledger(root)["updated_at"])
    window = timedelta(seconds=terminal.silence_seconds())
    early = stopped + window + timedelta(hours=1)
    assert terminal.record_unretried_preparation(root, alive_since=stopped, now=early) is None
    late = stopped + timedelta(seconds=MAXIMUM_JOB_SECONDS) + window + timedelta(minutes=1)
    overdue = terminal.record_unretried_preparation(root, alive_since=stopped, now=late)
    assert overdue["status"] == "failed_retryable" and overdue["last_error"] == resume.WAIT_ERROR
    record = terminal.current_overdue(overdue)
    assert record["reason"] == "recovery_needed" and record["retry_exhaustion_proven"] is False
    assert record["silent_since"] == terminal._iso(stopped + timedelta(seconds=MAXIMUM_JOB_SECONDS))
    assert terminal.TERMINAL_FIELD not in overdue and terminal.current_terminal(overdue) is None
    assert resume.armed(overdue) and overdue[resume.FIELD] == waiting[4]


def test_only_the_whole_atlas_wait_counts_anywhere_in_the_chain():
    wait = StageError("website_scene_preparation", "website_atlas_job_processing")
    try:
        raise PipelineError(str(wait)) from wait
    except PipelineError as wrapped:
        assert resume.atlas_wait(wrapped)
    assert resume.atlas_wait(PipelineError("outer")) is False
    assert not resume.atlas_wait(StageError("website_scene_preparation", "website_atlas_job_processing,other"))
    assert not resume.atlas_wait(StageError("website_scene_preparation", "website_atlas_job_failed"))
    assert resume.atlas_wait(StageError("website_scene_preparation", "website_atlas_job_needs_input"))


def test_a_wait_after_a_real_failure_or_a_dead_attempt_still_arms(waiting):
    """The resume is armed on the wait itself: an earlier failure no longer disarms it (finding 3)."""
    root, owner, context, raw, record = waiting
    history = listener._read_job_ledger(root)["attempt_history"]
    unknown = {**history[0], "error": "clean_plate: meta_sam_http_500", resume.FIELD: None}
    dead = {"attempt_number": 2, "status": "failed_retryable", "stage": resume.LEASE_EXPIRED_STAGE,
            "error": resume.LEASE_EXPIRED_ERROR, "outcome": "unknown", "recovered_expired_lease": True}
    ledger = listener._read_job_ledger(root)
    ledger.update(attempt_history=[unknown, dead], attempt_count=2, last_error=resume.LEASE_EXPIRED_ERROR)
    ledger.pop(resume.FIELD)
    write_json(root / listener.JOB_LEDGER_FILENAME, ledger)
    with pytest.raises(PipelineError, match="website_atlas_job_processing"):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
                                         run_e2e=_orchestrated(root, owner, context))
    armed = listener._read_job_ledger(root)
    assert resume.armed(armed) and armed[resume.FIELD] == record and armed["attempt_count"] == 3
    # And it resumes when the Atlas job ends.
    _job(root, "ready")
    calls = []
    process_args = {"provider": "openai", "run_evaluation_prep": False,
                    "run_e2e": lambda **kw: calls.append(kw) or {"status": "completed"}}
    assert resume.reconcile_waiting_reconstructions(listener, storage_root=root.parents[4],
                                                     process_args=process_args) == {"resumed": 1, "pending": 0}
    assert len(calls) == 1 and listener._read_job_ledger(root)["status"] == "completed"


def test_a_job_that_is_no_longer_processing_is_not_armed(waiting):
    root, _, _, raw, record = waiting
    _job(root, "ready")
    wait = StageError("website_scene_preparation", "website_atlas_job_processing")
    assert resume.resume_record(listener, exc=wait, payload=raw, payload_digest=record["source_payload_sha256"],
        producer_delivery_key=record["producer_delivery_key"], capture_root=root) is None


def test_the_existing_empty_queue_tick_also_replays_finished_atlas_waits(monkeypatch, tmp_path):
    import google.cloud.pubsub_v1
    calls = []
    monkeypatch.setattr(resume, "reconcile_waiting_reconstructions", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(google.cloud.pubsub_v1, "SubscriberClient",
                        lambda: SimpleNamespace(pull=lambda **_: SimpleNamespace(received_messages=[])))
    listener.pull_and_process(subscription="projects/synthetic/subscriptions/handoff", storage_root=tmp_path,
        provider="openai", max_messages=1, run_evaluation_prep=False, run_e2e_enabled=False)
    assert len(calls) == 1 and calls[0]["process_args"]["provider"] == "openai"
    assert calls[0]["process_args"]["run_e2e_enabled"] is False


def _killed_attempt(root, raw, *, started_ago):
    """The real claim of an attempt whose process the kernel then killed: its lease is never finished."""
    failed = listener._read_job_ledger(root)
    status, dead = listener._claim_job_lease(root, scene_id=failed["scene_id"], capture_id=failed["capture_id"],
        owner="control-plane:4242:killed", lease_seconds=listener.DEFAULT_JOB_LEASE_SECONDS,
        now=datetime.now(timezone.utc) - started_ago, payload_sha256=listener.payload_sha256(raw),
        producer_delivery_key=failed["producer_delivery_key"])
    assert status == "claimed" and dead["status"] == "processing"
    return failed, dead


def _review(ledger, raw):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    return {"review_schema": resume.REVIEW_SCHEMA, "ledger_digest": canonical_digest(ledger),
            "selected_payload_sha256": listener.payload_sha256(raw)}


def test_an_attempt_killed_holding_its_lease_resumes_once_reviewed(waiting):
    root, _, _, raw, _ = waiting
    failed, dead = _killed_attempt(root, raw, started_ago=timedelta(hours=1))
    _job(root, "ready")
    calls = []
    result = listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
        run_evaluation_prep=False, expected_reconstruction_resume=_review(dead, raw),
        run_e2e=lambda **kw: calls.append(kw) or {"status": "completed"})
    assert result["status"] == "processed" and len(calls) == 1 and calls[0]["resume_completed_stages"] is True
    after = listener._read_job_ledger(root)
    assert after["status"] == "completed" and after["attempt_count"] == dead["attempt_count"] + 1 == 3
    first, killed, resumed = after["attempt_history"]
    assert first == failed["attempt_history"][0]
    assert (killed["attempt_number"], killed["status"], killed["stage"], killed["error"]) == (
        2, "failed_retryable", resume.LEASE_EXPIRED_STAGE, resume.LEASE_EXPIRED_ERROR)
    assert killed["recovered_expired_lease"] is True and killed["previous_lease_owner"] == dead["lease_owner"]
    assert all(killed[key] == dead[key] for key in ("lease_acquired_at", "lease_heartbeat_at", "lease_expires_at"))
    assert killed["started_at"] == dead["last_attempt_started_at"]
    assert (resumed["attempt_number"], resumed["status"]) == (3, "completed")


def test_a_reviewed_resume_never_takes_a_live_lease(waiting):
    root, _, _, raw, _ = waiting
    _, live = _killed_attempt(root, raw, started_ago=timedelta(minutes=1))
    _job(root, "ready")
    result = listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
        expected_reconstruction_resume=_review(live, raw),
        run_e2e=lambda **_: pytest.fail("a live attempt's lease was taken over"))
    assert result["status"] == "lease_active_retryable" and result["blockers"] == ["handoff_job_active_lease"]
    assert listener._read_job_ledger(root) == live


def test_a_resumed_dead_attempt_that_fails_again_stays_reviewable(waiting):
    root, _, _, raw, _ = waiting
    _, dead = _killed_attempt(root, raw, started_ago=timedelta(hours=1))
    _job(root, "ready")

    def unreconciled(**_):
        # The dead attempt's SAM submission has no recorded outcome; the stage refuses to buy it again.
        raise PipelineError("clean_plate: meta_sam_submission_requires_reconciliation")
    with pytest.raises(PipelineError, match="meta_sam_submission_requires_reconciliation"):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
            run_evaluation_prep=False, expected_reconstruction_resume=_review(dead, raw), run_e2e=unreconciled)
    again = listener._read_job_ledger(root)
    assert again["status"] == "failed_retryable" and again["attempt_count"] == 3
    assert again["recovered_expired_lease"] is False and not resume.armed(again)
    assert [row["stage"] for row in again["attempt_history"][1:]] == [resume.LEASE_EXPIRED_STAGE, "run_e2e"]
    # Once that submission is reconciled, the ordinary review resumes the job again.
    result = listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
        run_evaluation_prep=False, expected_reconstruction_resume=_review(again, raw),
        run_e2e=lambda **_: {"status": "completed"})
    assert result["status"] == "processed" and listener._read_job_ledger(root)["attempt_count"] == 4


def _status(root, owner, record, schema_version):
    from blueprint_pipeline.website_preparation_status import read_preparation_status

    context = json.loads((root / "pipeline" / "website_task_context.json").read_text())
    payload = json.loads(base64.b64decode(record["payload_base64"]))
    return read_preparation_status(capture_root=root, schema_version=schema_version, selectors={
        "request_id": owner["request_id"], "scene_id": owner["scene_id"], "capture_id": owner["capture_id"],
        "completion_marker_generation": payload["source_finalize"]["generation"],
        "producer_delivery_key": record["producer_delivery_key"],
        "source_payload_sha256": record["source_payload_sha256"], "task_context_digest": context["context_digest"]})


def test_an_armed_reconstruction_wait_reports_preparing_to_the_website(waiting):
    """2026-11-11 (capture-447 nine times on 10-30): the normal Atlas wait was told to the site as a failure."""
    from blueprint_pipeline.website_preparation_status import STATUS_V1, STATUS_V7

    root, owner, _, _, record = waiting
    status = _status(root, owner, record, STATUS_V7)
    assert status["state"] == "preparing"
    assert status.get("terminal_reason") is None
    # The v1 projection the resume itself reads is unchanged, so the replay still happens.
    assert _status(root, owner, record, STATUS_V1)["state"] == "failed_retryable"


def test_a_real_retryable_failure_or_an_overdue_wait_reports_as_before(waiting):
    from blueprint_pipeline.website_preparation_status import _v2_state

    ledger = listener._read_job_ledger(waiting[0])
    assert _v2_state(ledger) == ("preparing", None)
    failure = {key: value for key, value in ledger.items() if key != resume.FIELD}
    failure["last_error"] = "website_scene_preparation: some_real_failure"
    assert not resume.armed(failure) and _v2_state(failure) == ("failed_retryable", None)
    stopped = terminal._parse_utc(ledger["updated_at"])
    late = stopped + timedelta(seconds=MAXIMUM_JOB_SECONDS + terminal.silence_seconds() + 60)
    overdue = terminal.record_unretried_preparation(waiting[0], alive_since=stopped, now=late)
    assert resume.armed(overdue) and _v2_state(overdue)[0] != "preparing"


@pytest.mark.parametrize("error, state", [(resume.WAIT_ERROR, "preparing"), (resume.NEEDS_INPUT_ERROR, "failed_retryable")])
def test_a_wait_on_an_atlas_job_that_needs_input_reports_as_needing_attention(waiting, error, state):
    """Only a reopen by Blueprint moves it: the site is not told it is being prepared, and Blueprint is told."""
    from blueprint_pipeline.website_preparation_status import _v2_state
    ledger = listener._read_job_ledger(waiting[0])
    ledger = {**ledger, "last_error": error, resume.FIELD: {**ledger[resume.FIELD], "error": error}}
    assert resume.armed(ledger) and _v2_state(ledger) == (state, None)


def test_a_withdrawn_capture_job_is_never_reopenable(tmp_path, monkeypatch):
    from blueprint_pipeline import website_atlas_preview_store as store
    job = {"state": "needs_input", "reason": REOPENABLE, "enqueued_at_epoch": time.time()}
    monkeypatch.setattr(store, "capture_withdrawn", lambda _root: False)
    assert resume.reopenable(job, tmp_path) and not resume._final(job, tmp_path)
    monkeypatch.setattr(store, "capture_withdrawn", lambda _root: True)
    assert not resume.reopenable(job, tmp_path) and resume._final(job, tmp_path)


def test_a_wait_on_an_atlas_job_whose_reopens_are_spent_ends_at_its_next_check(waiting):
    """2026-11-24 (move-the-lamp-2): the preparation waited seven days on an Atlas job that only deploys reopened,
    refusing alike on the same retained bytes. Once the job's automatic reopens are spent its stop is its end:
    the next drain replays the preparation once, which records the stop and arms no wait again."""
    root, owner, context, _, _ = waiting
    runs = []

    def orchestrated(**kwargs):
        runs.append(1)
        return _orchestrated(root, owner, context)(**kwargs)
    drain = lambda: resume.reconcile_waiting_reconstructions(  # noqa: E731
        listener, storage_root=root.parents[4],
        process_args={"provider": "openai", "run_evaluation_prep": False, "run_e2e": orchestrated})
    now = time.time()
    reopened = [{"reason": "release_reopened", "at_epoch": now - 60}] * 3
    parked = {"reason": "atlas_context_field_of_view_lost", "stage": "preparation", "enqueued_at_epoch": now - 86400}
    _job(root, "needs_input", **parked, retries=reopened)
    drain()  # the wait armed while Atlas processed replays once and records the needs-input stop
    assert len(runs) == 1 and resume.armed(listener._read_job_ledger(root))
    assert listener._read_job_ledger(root)["last_error"] == resume.NEEDS_INPUT_ERROR
    assert drain() == {"resumed": 0, "pending": 1} and len(runs) == 1  # three of six: it may yet be reopened
    _job(root, "needs_input", **parked, retries=[*reopened, *[{"reason": "parked_retry", "at_epoch": now}] * 3])
    assert drain()["pending"] == 0 and len(runs) == 2  # spent: replayed at its next check
    assert not resume.armed(listener._read_job_ledger(root))
    assert drain() == {"resumed": 0, "pending": 0} and len(runs) == 2


#: A stop on no wall that the clock and a release reopen (``website_atlas_preview._reopen_parked``).
PARKED_STOP = "atlas_context_field_of_view_lost"


def _spent(now, stop=PARKED_STOP):
    return [{"reason": "release_reopened", "at_epoch": now - 60, "stopped_on": stop}] * 6


@pytest.mark.parametrize("fields, ended", [
    ({"state": "failed", "reason": PARKED_STOP, "spent": True}, True),
    ({"state": "needs_input", "reason": PARKED_STOP, "spent": True}, True),
    ({"state": "failed", "reason": PARKED_STOP, "age_days": 6.5}, True),        # no room left for a reopen
    ({"state": "failed", "reason": PARKED_STOP}, False),                         # the next release reopens it
    ({"state": "failed", "reason": PARKED_STOP, "spent": True, "retry_at_epoch": 1}, False),  # rechecked
    ({"state": "failed", "reason": "atlas_source_rights_unverified", "spent": True}, False),  # a wall ends it
    ({"state": "failed", "reason": "spending_cap_reached", "spent": True}, False),            # ships what it has
    ({"state": "needs_input", "reason": "atlas_insufficient_overlapping_distinct_views", "spent": True}, False),
    ({"state": "needs_input", "reason": "atlas_scene_requires_preserved_capture", "spent": True}, False),
    ({"state": "processing", "reason": None}, False),
    ({"state": "ready", "reason": None}, False),
])
def test_an_atlas_job_ended_for_good_is_one_nothing_reopens_by_itself_any_more(tmp_path, fields, ended):
    """The Austin test job (2026-11-24): its Atlas job ended and its preparation stayed failed_retryable, retried
    for its whole window. Only a job parked on no wall whose reopens are spent, or whose window leaves no room for
    one, has ended for good: one a release still reopens, one waiting on a wall, the site, a new video or a
    recheck, and one that shipped what it had are not."""
    from blueprint_pipeline.atlas_preparation_job_control import ended_for_good
    now = time.time()
    fields = dict(fields)
    spent, age = fields.pop("spent", False), fields.pop("age_days", 1)
    job = {"schema_version": JOB_SCHEMA, "enqueued_at_epoch": now - 86400 * age, "retry_at_epoch": None, **fields}
    if spent:
        job["retries"] = _spent(now, job["reason"])
    assert ended_for_good(job, now=now) is ended
    (tmp_path / "pipeline/atlas").mkdir(parents=True)
    write_json(tmp_path / "pipeline/atlas" / JOB_FILE, job)
    evidence = resume.atlas_job_ended(tmp_path, now=now)
    assert (evidence is not None) is ended
    if ended:
        assert evidence == {"state": job["state"], "reason": job["reason"], "revision": None,
                            "automatic_reopens": 6 if "retries" in job else 0}


def test_a_capture_with_no_readable_atlas_job_has_none_that_ended(tmp_path):
    assert resume.atlas_job_ended(tmp_path) is None
    (tmp_path / "pipeline/atlas").mkdir(parents=True)
    (tmp_path / "pipeline/atlas" / JOB_FILE).write_text("{not json")
    assert resume.atlas_job_ended(tmp_path) is None


def test_a_failed_preparation_whose_atlas_job_ended_for_good_records_an_ending_bound_to_its_attempt(waiting):
    """It can never finish, so its ending is recorded (``failed_terminal``, ``retry_exhausted``: "We tried several
    times and could not get past a problem on our side"), not left reading as work still going on. The record binds
    the attempt: a later one leaves it behind. A silence-based record of the old producer is still never one."""
    from blueprint_pipeline.website_preparation_status import _v2_state
    root = waiting[0]
    ledger = listener._read_job_ledger(root)
    failure = {key: value for key, value in ledger.items() if key != resume.FIELD}
    failure["last_error"] = "website_scene_preparation: website_atlas_job_failed"
    now = time.time()
    _job(root, "failed", reason=PARKED_STOP, enqueued_at_epoch=now - 86400, retries=_spent(now))
    ended = terminal.atlas_job_ending(failure, resume.atlas_job_ended(root, now=now),
                                      datetime.fromtimestamp(now, timezone.utc))
    record = terminal.current_terminal(ended)
    assert record["reason"] == "retry_exhausted" and record["basis"] == terminal.ATLAS_JOB_ENDED
    assert record["atlas_job"] == {"state": "failed", "reason": PARKED_STOP, "revision": None, "automatic_reopens": 6}
    assert ended["attempt_history"][-1]["status"] == "preparation_terminal_recorded"
    assert _v2_state(ended) == ("failed_terminal", "retry_exhausted")
    assert _v2_state(failure) == ("failed_retryable", None)
    # A new attempt (an operator reopened the Atlas job, a redelivery) leaves the ending behind.
    assert terminal.current_terminal({**ended, "attempt_count": ended["attempt_count"] + 1}) is None
    assert terminal.current_terminal({**ended, "status": "processing", "lease_token": "t"}) is None
    # The old producer's silence-based annotation is history, never a disposition.
    legacy = {**ended, terminal.TERMINAL_FIELD: {**record, "schema_version": terminal.SCHEMA_VERSION}}
    assert terminal.current_terminal(legacy) is None and _v2_state(legacy) == ("failed_retryable", None)
