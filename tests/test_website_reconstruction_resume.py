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
from datetime import timedelta
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


ASSESSMENT_ROW = {"stage": "website_assessment_preparation",
                  "error": "website_control_scene-sponsorship_http_409:website_assessment_preparation_pending",
                  "assessment_resume": {"schema_version": "website_assessment_resume.v1"}}
WAIT_ROW = {"stage": "run_e2e", "error": resume.WAIT_ERROR,
            resume.FIELD: {"schema_version": resume.SCHEMA, "error": resume.WAIT_ERROR}}


@pytest.mark.parametrize("history,recovered,allowed", [
    ([], False, True), ([WAIT_ROW], False, True), ([ASSESSMENT_ROW, WAIT_ROW], False, True),
    ([{"stage": "run_e2e", "error": "unknown_provider_charge"}], False, False), ([], True, False),
    ([{**WAIT_ROW, resume.FIELD: {"schema_version": resume.SCHEMA, "error": "other"}}], False, False)])
def test_only_explicit_waits_before_it_arm_the_replay(history, recovered, allowed):
    assert resume.safe_to_arm(len(history) + 1, history, recovered) is allowed


def test_a_job_that_is_no_longer_processing_is_not_armed(waiting):
    root, _, _, raw, record = waiting
    _job(root, "ready")
    wait = StageError("website_scene_preparation", "website_atlas_job_processing")
    assert resume.resume_record(listener, exc=wait, payload=raw, payload_digest=record["source_payload_sha256"],
        producer_delivery_key=record["producer_delivery_key"], capture_root=root, attempt_count=1,
        previous_history=[], recovered_expired_lease=False) is None


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
