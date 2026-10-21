"""Development fixtures prove the job bridge, not site or physical performance."""

from __future__ import annotations

import copy
import hmac
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blueprint_pipeline.decision_evidence_contracts import (
    MaintainedSiteTaskTestbed,
    cross_runtime_canonical_digest,
)
from blueprint_pipeline.job_evidence import JobEvidenceError, JobEvidenceStore, canonical_artifact_body
from blueprint_pipeline.job_evidence_routes import register_job_evidence_routes
from blueprint_pipeline.job_evidence_webapp_sync import sync_job_evidence_to_webapp
from blueprint_pipeline.task_evaluation_method_catalog import (
    TASK_EVALUATION_METHOD_CATALOG_PATH_ENV,
    validate_task_evaluation_method_catalog,
)
from tests.test_task_evaluation_run_control_plane import (
    SHA_A, _profile, _qualification, _request, _testbed,
)


def _job(*, scientific=False):
    value = {
        "schema_version": "job_evidence_request.v1", "job_id": "job-1",
        "request_id": "job-question-1", "brief_digest": SHA_A,
        "question": "What evidence is needed to decide?",
        "testbed": None, "decision_evidence_request": None,
        "approved_task_digest": None, "capture_session_id": None, "intake_id": None,
        "testbed_canonical_json": None, "decision_request_canonical_json": None,
        "compiled_card_canonical_json": None,
        "requested_by": {"uid": "operator-1"}, "idempotency_key": "question-1",
    }
    if scientific:
        testbed = _testbed()
        testbed.pop("testbed_digest")
        testbed["approved_task_definition"] = {"digest": SHA_A, "status": "owner_approved"}
        testbed = MaintainedSiteTaskTestbed.from_mapping(testbed).to_mapping()
        native = _request(testbed)
        value.update({
            "question": native["decision_question"], "testbed": testbed,
            "decision_evidence_request": native, "capture_session_id": "walkthrough-job-1",
            "intake_id": "capture-1", "approved_task_digest": SHA_A,
            "testbed_canonical_json": canonical_artifact_body(testbed, "testbed_digest"),
            "decision_request_canonical_json": canonical_artifact_body(native, "request_digest"),
        })
    return _seal(value)


def _seal(value):
    value["request_digest"] = cross_runtime_canonical_digest(value, digest_field="request_digest")
    return value


def _catalog(tmp_path, monkeypatch):
    profile = _profile()
    catalog = validate_task_evaluation_method_catalog({
        "schema_version": "task_evaluation_method_catalog.v1",
        "catalog_id": "development-fixture", "version": "1",
        "method_profiles": [profile], "qualifications": [_qualification(profile)],
    })
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    monkeypatch.setenv(TASK_EVALUATION_METHOD_CATALOG_PATH_ENV, str(path))


def test_description_only_question_has_no_fake_capture_or_run(tmp_path):
    store = JobEvidenceStore(tmp_path / "jobs", tmp_path / "runs")
    job = _job()
    projection = store.submit(job)
    assert projection["state"] == "information_required"
    assert projection["run_id"] is None
    assert projection["testbed_digest"] is None
    assert projection["decision_envelope"] is None
    assert not (tmp_path / "runs").exists()
    assert store.submit(job) == projection
    conflict = copy.deepcopy(job)
    conflict["request_id"] = "second-request"
    with pytest.raises(JobEvidenceError, match="immutable_conflict"):
        store.submit(_seal(conflict))


def test_real_native_plan_executes_local_evidence_once_and_preserves_verdict(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)
    # No test may publish to a real service via environment inherited from the host.
    monkeypatch.delenv("PIPELINE_TASK_EVALUATION_RUN_WEBAPP_URL", raising=False)
    store = JobEvidenceStore(tmp_path / "jobs", tmp_path / "runs")
    job = _job(scientific=True)
    planned = store.submit(job)
    assert planned["state"] == "planned"
    assert planned["approved_task_digest"] == SHA_A
    result = store.execute(job["request_id"], job["request_digest"], {"role": "human", "identity": "operator-1"})
    envelope = result["decision_envelope"]
    assert result["state"] == "decision"
    assert envelope["per_claim_verdicts"][0]["verdict"] == "supported"
    assert envelope["claim_ceiling"]["physical_success"] is False
    assert result["decision_request_digest"] == job["decision_evidence_request"]["request_digest"]
    assert result["plan_digest"] == envelope["plan_digest"]
    assert result["sequence"] == 2
    assert store.execute(job["request_id"], job["request_digest"], {"role": "human", "identity": "operator-1"}) == result
    assert store.submit(job) == result
    assert len(list((tmp_path / "runs" / "runs").iterdir())) == 1


def test_unsupported_claim_abstains_without_creating_a_pass(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)
    job = _job(scientific=True)
    native = job["decision_evidence_request"]
    native.pop("request_digest")
    native["claims"][0]["claim_type"] = "physical_task_success"
    from blueprint_pipeline.decision_evidence_contracts import DecisionEvidenceRequest
    job["decision_evidence_request"] = DecisionEvidenceRequest.from_mapping(native).to_mapping()
    job["decision_request_canonical_json"] = canonical_artifact_body(job["decision_evidence_request"], "request_digest")
    _seal(job)
    store = JobEvidenceStore(tmp_path / "jobs", tmp_path / "runs")
    store.submit(job)
    result = store.execute(job["request_id"], job["request_digest"], {"role": "human", "identity": "operator-1"})
    assert result["state"] == "abstention"
    assert result["decision_envelope"]["evidence_accepted"] == []
    assert result["decision_envelope"]["claim_ceiling"]["physical_success"] is False


def test_negative_reach_result_and_node_number_roundtrip_preserve_science(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)
    job = _job(scientific=True)
    testbed = job["testbed"]
    testbed.pop("testbed_digest")
    testbed["target_regions"][0]["position_site_m"] = [2.0, 0.0, 0.0]
    job["testbed"] = MaintainedSiteTaskTestbed.from_mapping(testbed).to_mapping()
    job["testbed_canonical_json"] = canonical_artifact_body(job["testbed"], "testbed_digest")
    job["decision_evidence_request"] = _request(job["testbed"])
    job["decision_request_canonical_json"] = canonical_artifact_body(job["decision_evidence_request"], "request_digest")
    # JSON.parse/stringify in Node loses .0. Producer canonical bytes must
    # restore the exact native artifact rather than reinterpret its digest.
    def node_numbers(value):
        if isinstance(value, dict):
            return {key: node_numbers(child) for key, child in value.items()}
        if isinstance(value, list):
            return [node_numbers(child) for child in value]
        return int(value) if isinstance(value, float) and value.is_integer() else value
    job = _seal(node_numbers(job))
    store = JobEvidenceStore(tmp_path / "jobs", tmp_path / "runs")
    store.submit(job)
    result = store.execute(job["request_id"], job["request_digest"], {"role": "human", "identity": "operator-1"})
    assert result["state"] == "decision"
    assert result["decision_envelope"]["per_claim_verdicts"][0]["verdict"] == "not_supported"
    assert result["decision_envelope"]["claim_ceiling"]["physical_success"] is False


def test_binding_mismatch_and_cancellation_prevent_execution(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)
    job = _job(scientific=True)
    store = JobEvidenceStore(tmp_path / "jobs", tmp_path / "runs")
    store.submit(job)
    with pytest.raises(JobEvidenceError, match="binding_mismatch"):
        store.execute(job["request_id"], "sha256:" + "b" * 64, {"role": "human", "identity": "operator-1"})
    cancelled = store.cancel(job["request_id"], job["request_digest"], "operator-1")
    assert store.submit(job) == cancelled
    with pytest.raises(JobEvidenceError, match="cancelled"):
        store.execute(job["request_id"], job["request_digest"], {"role": "human", "identity": "operator-1"})
    from blueprint_pipeline.task_evaluation_run_control_plane import execute_and_aggregate_task_evaluation_run
    with pytest.raises(ValueError, match="cancelled"):
        execute_and_aggregate_task_evaluation_run(state_root=tmp_path / "runs", run_id=cancelled["run_id"])
    changed = copy.deepcopy(job)
    changed["approved_task_digest"] = "sha256:" + "b" * 64
    with pytest.raises(JobEvidenceError, match="approved_task_binding_mismatch"):
        store.submit(_seal(changed))


def test_catalog_unavailable_preserves_real_task_binding_and_can_retry(tmp_path, monkeypatch):
    monkeypatch.delenv(TASK_EVALUATION_METHOD_CATALOG_PATH_ENV, raising=False)
    job = _job(scientific=True)
    store = JobEvidenceStore(tmp_path / "jobs", tmp_path / "runs")
    unavailable = store.submit(job)
    assert unavailable["state"] == "unavailable"
    assert unavailable["testbed_digest"] == job["testbed"]["testbed_digest"]
    assert unavailable["run_id"] is None
    _catalog(tmp_path, monkeypatch)
    planned = store.submit(job)
    assert planned["state"] == "planned"
    assert planned["sequence"] == 2


def test_admitted_routes_publish_and_retry_receipt_without_external_effects(tmp_path, monkeypatch):
    app = FastAPI()
    admitted = []
    def admission():
        admitted.append(True)
    register_job_evidence_routes(app, require_admission=admission,
                                state_root=lambda: tmp_path / "jobs", run_state_root=lambda: tmp_path / "runs")
    from blueprint_pipeline import job_evidence_routes
    monkeypatch.setattr(job_evidence_routes, "sync_job_evidence_to_webapp", lambda payload: {"status": "skipped"})
    client = TestClient(app)
    first = client.post("/api/live-pipeline/job-evidence-requests", json=_job())
    assert first.status_code == 200
    assert client.get("/api/live-pipeline/job-evidence-requests/job-question-1").json() == first.json()
    assert len(admitted) == 2
    assert client.post("/api/live-pipeline/job-evidence-requests", content="{").status_code == 400


def test_signed_sync_requires_exact_receipt(tmp_path, monkeypatch):
    publication = JobEvidenceStore(tmp_path / "jobs", tmp_path / "runs").submit(_job())
    from blueprint_pipeline import job_evidence_webapp_sync
    receipt = {"schema_version": "job_evidence_publication_receipt.v1", "status": "published",
               **{field: publication[field] for field in ("request_id", "job_id", "request_digest", "publication_digest")}}
    seen = []
    def transport(request, timeout):
        seen.append(request)
        return BytesIO(json.dumps(receipt).encode())
    monkeypatch.setattr(job_evidence_webapp_sync.urllib_request, "urlopen", transport)
    result = sync_job_evidence_to_webapp(publication, endpoint_url="https://webapp.example/api/internal/job-evidence-publications", token="fixture-only")
    assert result["status"] == "succeeded"
    assert seen[0].has_header("X-blueprint-signature") or any("signature" in key.lower() for key in seen[0].headers)
    receipt["job_id"] = "other-job"
    assert sync_job_evidence_to_webapp(publication, endpoint_url="https://webapp.example/api/internal/job-evidence-publications", token="fixture-only")["reason"] == "publication_receipt_mismatch"


def test_real_intake_admission_requires_fresh_service_signature(tmp_path, monkeypatch):
    from blueprint_pipeline import live_pipeline_intake_service as service
    from blueprint_pipeline import job_evidence_routes
    secret = "development-only-secret"
    monkeypatch.setenv(service.INTAKE_CLIENT_SECRETS_ENV, json.dumps({"blueprint-webapp": secret}))
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "work"))
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.delenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, raising=False)
    monkeypatch.setattr(job_evidence_routes, "sync_job_evidence_to_webapp", lambda value: {"status": "skipped"})
    client = TestClient(service.create_app())
    body = json.dumps(_job(), separators=(",", ":"))
    url = "/api/live-pipeline/job-evidence-requests"
    assert client.post(url, content=body).status_code == 401
    timestamp = datetime.now(timezone.utc).isoformat()
    nonce = "job-evidence-service-fixture"
    signature = hmac.new(secret.encode(), f"{timestamp}.blueprint-webapp.{nonce}.{body}".encode(), "sha256").hexdigest()
    headers = {
        "Content-Type": "application/json", "x-blueprint-pipeline-timestamp": timestamp,
        "x-blueprint-pipeline-client-id": "blueprint-webapp", "x-blueprint-pipeline-nonce": nonce,
        "x-blueprint-pipeline-signature": "sha256=" + signature,
    }
    accepted = client.post(url, content=body, headers=headers)
    assert accepted.status_code == 200
    assert accepted.json()["publication"]["state"] == "information_required"
    assert client.post(url, content=body, headers=headers).status_code == 401


def test_python_producer_wire_examples_still_match_actual_execution(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)
    from blueprint_pipeline import task_evaluation_run_control_plane as control
    monkeypatch.setattr(control, "sync_task_evaluation_run_to_webapp", lambda **kwargs: {"status": "skipped"})
    examples = json.loads((Path(__file__).parent / "fixtures/job_evidence_v1/producer-examples.json").read_text())
    assert examples["development_only"] is True
    store = JobEvidenceStore(tmp_path / "jobs", tmp_path / "runs")
    assert store.submit(examples["description_request"]) == examples["description_publication"]
    native = examples["native_request"]
    assert store.submit(native) == examples["native_planned"]
    assert store.execute(native["request_id"], native["request_digest"], {
        "role": "human", "identity": "operator-1",
    }) == examples["native_result"]
