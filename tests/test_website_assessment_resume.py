# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_assessment_resume.py
#   src/blueprint_pipeline/pubsub_handoff_scene_operations.py
"""ADP-010/day14: resume the existing browser producer, never a second job."""
import base64
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_task_context as context_reader
from blueprint_pipeline import capture_original_owner_observer as observer
from blueprint_pipeline import website_assessment_resume as resume
from blueprint_pipeline.common import write_json
from blueprint_pipeline.task_evaluation_scene_retirement_generations import capture_birth_source_projection
from tests.test_website_preparation_status import prepared_source  # noqa: F401

WIRE_FIXTURE = Path(__file__).parent / "fixtures/website_assessment_resume/browser_handoff.json"


@pytest.fixture
def waiting(request, monkeypatch):
    root, owner, context, selected, _, _ = request.getfixturevalue("prepared_source")
    birth = capture_birth_source_projection(root)
    payload = json.loads(WIRE_FIXTURE.read_bytes())
    payload["source_finalize"] = {"bucket": owner["bucket"], "object_name": owner["completion_marker"]["object_name"],
        "generation": owner["completion_marker"]["generation"], "event_id": "original-event", "event_source": "original-source"}
    payload["source_membership_selector"] = birth["source_membership_selector"]
    payload["pipeline_status_event"]["source_finalize"] = payload["source_finalize"]
    # Fixture's synthetic membership generation/hash is replaced by this
    # existing test's actual validated birth membership, before bytes are fixed.
    payload["pipeline_handoff_uri"] = f"gs://{owner['bucket']}/{birth['source_membership_selector']['object_name'].rsplit('/', 1)[0]}/pipeline_handoff.json"
    payload["pipeline_status_event"]["pipeline_handoff_uri"] = payload["pipeline_handoff_uri"]
    raw = json.dumps(payload, indent=2).encode()
    # The reusable fixture's synthetic lease has never run a provider. Remove
    # that setup-only ledger so the actual processor creates its first lease.
    (root / listener.JOB_LEDGER_FILENAME).unlink()
    write_json(root / "raw/manifest.json", {"capture_source": "browser_self_capture"})
    monkeypatch.setattr(listener, "stage_handoff_capture", lambda *_, **__: root)
    monkeypatch.setattr(context_reader, "load_website_scene_sponsorship", lambda **_: (_ for _ in ()).throw(ValueError(resume.PENDING_CODE)))
    with pytest.raises(resume.AssessmentPreparationPending) as error:
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
            run_e2e=lambda **_: pytest.fail("pending preflight entered provider stages"))
    record = error.value.resume_record
    assert record is not None
    actual = listener._read_job_ledger(root)
    assert actual["assessment_resume"] == actual["attempt_history"][0]["assessment_resume"] == record
    assert actual["attempt_count"] == 1
    return root, owner, context, raw, record


def _ready(waiting, monkeypatch):
    root, owner, context, raw, record = waiting
    proposal = {"schema_version": "site_assessment_preparation_proposal.v1", "request_id": owner["request_id"],
        "capture_id": owner["capture_id"], "job_id": "advisory-" + "a" * 64, "run_id": "site-assessment-test",
        "source_key": "sha256:" + "b" * 64, "context_digest": "c" * 64, "packet_sha256": "d" * 64,
        "questions_pending": True, "scope": "scene_preparation_only", "robot_suitability_verified": False,
        "physical_trial_authorized": False}
    monkeypatch.setattr(context_reader, "load_website_scene_sponsorship", lambda **_: {"assessment_preparation_proposal": proposal})


def test_pending_preflight_retains_exact_bytes_and_never_claims_another_attempt(waiting, monkeypatch):
    root, _, _, raw, record = waiting
    before = listener._read_job_ledger(root)
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *_, **__: pytest.fail("pending assessment ran the producer"))
    result = resume.reconcile_waiting_assessments(listener, storage_root=root.parents[4], process_args={"provider": "openai"})
    assert result == {"resumed": 0, "pending": 1}
    assert base64.b64decode(record["payload_base64"], validate=True) == raw
    assert listener._read_job_ledger(root) == before


def test_completed_assessment_resumes_same_original_payload_after_empty_pubsub(waiting, monkeypatch):
    root, _, _, raw, _ = waiting
    _ready(waiting, monkeypatch)
    calls = []
    def process(payload, **kwargs):
        calls.append((payload, kwargs))
        state, lease = listener._claim_job_lease(root, scene_id=waiting[1]["scene_id"], capture_id=waiting[1]["capture_id"],
            owner="resumed-worker", lease_seconds=900, producer_delivery_key=waiting[4]["producer_delivery_key"], payload_sha256=listener.payload_sha256(payload),
            expected_assessment_resume=kwargs["expected_assessment_resume"])
        assert state == "claimed" and "assessment_resume" not in lease
        assert lease["attempt_history"][0]["assessment_resume"]["payload_base64"] == waiting[4]["payload_base64"]
        listener._finish_job_lease(root, owner="resumed-worker", token=lease["lease_token"], update={"status": "failed_retryable", "last_error": "unknown_provider_charge"})
        return {"status": "retryable_blocked", "queue_disposition": "retryable"}
    monkeypatch.setattr(listener, "process_handoff_payload", process)
    result = resume.reconcile_waiting_assessments(listener, storage_root=root.parents[4], process_args={"provider": "openai", "run_evaluation_prep": False})
    assert result == {"resumed": 1, "pending": 0}
    assert calls[0][0] == raw and calls[0][1]["payload_digest"] == listener.payload_sha256(raw)
    assert calls[0][1]["run_evaluation_prep"] is False
    assert resume.reconcile_waiting_assessments(listener, storage_root=root.parents[4], process_args={"provider": "openai"}) == {"resumed": 0, "pending": 0}
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["payload", "source", "active", "terminal", "wrong_error", "physical", "owner", "context"])
def test_refuses_changed_source_terminal_active_unknown_failure_or_false_authority(waiting, monkeypatch, change):
    root, owner, _, _, _ = waiting
    _ready(waiting, monkeypatch)
    ledger = listener._read_job_ledger(root)
    if change == "payload":
        ledger["assessment_resume"]["payload_base64"] = base64.b64encode(b"{}").decode()
    elif change == "source":
        ledger["producer_delivery_key"] = "sha256:" + "e" * 64
    elif change == "active":
        ledger["status"] = "processing"
    elif change == "terminal":
        ledger["status"] = "terminal_authority_ended"
    elif change == "wrong_error":
        ledger["last_error"] = "unknown_provider_charge"
    elif change == "owner":
        monkeypatch.setattr(observer, "load_original_owner_observation", lambda **_: (_ for _ in ()).throw(ValueError("source_revoked")))
    elif change == "context":
        monkeypatch.setattr(context_reader, "load_current_website_task_context", lambda **_: {**waiting[2], "context_digest": "sha256:" + "e" * 64})
    elif change == "physical":
        monkeypatch.setattr(context_reader, "load_website_scene_sponsorship", lambda **_: {"assessment_preparation_proposal": {"physical_trial_authorized": True}})
    write_json(root / listener.JOB_LEDGER_FILENAME, ledger)
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *_, **__: pytest.fail("unadmitted resume"))
    assert resume.reconcile_waiting_assessments(listener, storage_root=root.parents[4], process_args={"provider": "openai"})["resumed"] == 0


def test_existing_empty_queue_tick_reconciles_waiting_job_with_same_execution_options(monkeypatch, tmp_path):
    import google.cloud.pubsub_v1
    calls = []
    monkeypatch.setattr(resume, "reconcile_waiting_assessments", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(google.cloud.pubsub_v1, "SubscriberClient", lambda: SimpleNamespace(pull=lambda **_: SimpleNamespace(received_messages=[])))
    listener.pull_and_process(subscription="projects/synthetic/subscriptions/handoff", storage_root=tmp_path,
        provider="openai", max_messages=1, run_evaluation_prep=False, run_e2e_enabled=False)
    assert len(calls) == 1
    assert calls[0]["process_args"]["provider"] == "openai"
    assert calls[0]["process_args"]["run_e2e_enabled"] is False


def test_intervening_unknown_provider_failure_is_rejected_inside_actual_resume_claim(waiting, monkeypatch):
    root = waiting[0]
    _ready(waiting, monkeypatch)
    claim = listener._claim_job_lease
    def raced(*args, **kwargs):
        assert kwargs["expected_assessment_resume"]["record"] == waiting[4]
        prior = listener._read_job_ledger(root)
        prior.pop("assessment_resume")
        prior.update(revision=prior["revision"] + 2, last_error="unknown_provider_charge", status="failed_retryable")
        write_json(root / listener.JOB_LEDGER_FILENAME, prior)
        return claim(*args, **kwargs)
    monkeypatch.setattr(listener, "_claim_job_lease", raced)
    result = resume.reconcile_waiting_assessments(listener, storage_root=root.parents[4], process_args={
        "provider": "openai", "run_e2e": lambda **_: pytest.fail("stale resume entered provider stages")})
    assert result["resumed"] == 0
    assert listener._read_job_ledger(root)["last_error"] == "unknown_provider_charge"
    assert listener._read_job_ledger(root)["attempt_count"] == 1


@pytest.mark.parametrize("history,recovered,allowed", [([], False, True),
    ([{"stage": "website_assessment_preparation", "error": resume.PENDING_CODE, "assessment_resume": {"schema_version": resume.SCHEMA}}], False, True),
    ([{"stage": "run_e2e", "error": "unknown_provider_charge"}], False, False),
    ([], True, False), ([{"stage": "website_assessment_preparation", "error": resume.PENDING_CODE}], False, False)])
def test_only_pre_provider_assessment_denials_arm_durable_replay(history, recovered, allowed):
    assert resume.safe_to_arm(len(history) + 1, history, recovered) is allowed


def test_reuses_rotating_status_cursor_so_blocked_first_capture_does_not_starve_ready_one(waiting, monkeypatch):
    root = waiting[0]
    _ready(waiting, monkeypatch)
    storage = root.parents[4]
    earlier = storage / "aaa" / "scenes" / "site-a" / "captures" / "walkthrough-a"
    write_json(earlier / listener.JOB_LEDGER_FILENAME, {"status": "failed_retryable"})
    write_json(storage / ".website_preparation_delivery_cursor.json", {"schema_version": "website_preparation_delivery_cursor.v1",
        "last_capture_digest": hashlib.sha256(str(earlier).encode()).hexdigest()})
    calls = []
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *args, **kwargs: calls.append(args[0]) or {"status": "processed"})
    assert resume.reconcile_waiting_assessments(listener, storage_root=storage, process_args={"provider": "openai"}, limit=1)["resumed"] == 1
    assert calls == [waiting[3]]


@pytest.mark.parametrize("raw", [b'{"bucket":"a","bucket":"b"}', b'{"bucket":"a","private_metadata":"synthetic"}'])
def test_exact_raw_retention_refuses_duplicate_keys_and_unknown_private_fields(raw):
    with pytest.raises(ValueError, match="website_assessment_resume_binding_invalid"):
        resume._payload_bytes(listener, raw, listener.payload_sha256(raw))


def test_exact_actual_browser_builder_envelope_is_retained_without_reencoding():
    raw = WIRE_FIXTURE.read_bytes()
    assert resume._payload_bytes(listener, raw, listener.payload_sha256(raw)) == raw
    assert json.loads(raw)["handoff_source"] == "BlueprintCapture.extractFrames"


@pytest.mark.parametrize("change", ["secret", "lineage", "candidate", "signed_url", "nested_selector", "nonfinite"])
def test_browser_retention_refuses_unsupported_nested_records_or_credentials(change):
    payload = json.loads(WIRE_FIXTURE.read_bytes())
    if change == "secret":
        payload["media_metadata"]["access_token"] = "synthetic-private-value"
    elif change == "lineage":
        payload["privacy_lineage"] = {"private_metadata": "synthetic"}
    elif change == "candidate":
        payload["task_site_context"]["robot_eval_task_anchor_candidates"] = [{"private_metadata": "synthetic"}]
    elif change == "signed_url":
        payload["capture_rights"]["permission_document_uri"] = "https://example.invalid/private?X-Goog-Signature=synthetic"
    elif change == "nested_selector":
        payload["pipeline_status_event"]["source_finalize"] = {"private_metadata": "synthetic"}
    elif change == "nonfinite":
        payload["media_metadata"]["width"] = float("nan")
    raw = json.dumps(payload).encode()
    with pytest.raises(ValueError, match="website_assessment_resume_binding_invalid"):
        resume._payload_bytes(listener, raw, listener.payload_sha256(raw))
