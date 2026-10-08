# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/website_preparation_status_http.py
#   src/blueprint_pipeline/live_pipeline_intake_service.py
#   src/blueprint_pipeline/pubsub_handoff_listener.py
#   src/blueprint_pipeline/pubsub_handoff_scene_operations.py
#   src/blueprint_pipeline/website_task_context.py
"""ADP-010/day14: source-bound preparation recovery never grants completion."""
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from blueprint_pipeline import live_pipeline_intake_service as service
from tests.test_live_pipeline_intake_service import _signed_intake_headers

import fcntl
import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import capture_original_owner_observer as observer
from blueprint_pipeline import website_task_context as context_reader
from blueprint_pipeline import website_preparation_status as status_reader
from blueprint_pipeline.common import PipelineError, StageError, write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member
from tests.test_capture_generation_birth import _fixture


def test_signed_status_refuses_unconfigured_server_root(tmp_path, monkeypatch):
    monkeypatch.setenv(service.INTAKE_CLIENT_SECRETS_ENV, json.dumps({"blueprint-webapp": "synthetic-secret"}))
    monkeypatch.setenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, "false")
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.setenv("BLUEPRINT_LIVE_PIPELINE_INTAKE_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_OUTPUT_PATH", str(tmp_path / "absent.json"))
    monkeypatch.delenv(service.INTAKE_CLIENT_ROOTS_ENV, raising=False)
    body = json.dumps({"request_id": "synthetic-status", "scene_id": "site-synthetic-status",
                       "capture_id": "walkthrough-synthetic-status", "completion_marker_generation": "19",
                       "producer_delivery_key": "sha256:" + "a" * 64,
                       "source_payload_sha256": "b" * 64, "task_context_digest": "sha256:" + "c" * 64})
    response = TestClient(service.create_app()).post(
        "/api/live-pipeline/website-preparation-status", content=body,
        headers=_signed_intake_headers("synthetic-secret", body, nonce="status-root", client_id="blueprint-webapp"))
    assert response.status_code == 503
    assert response.json()["detail"] == "website_preparation_status_unavailable"


@pytest.fixture
def prepared_source(tmp_path, monkeypatch):
    _, _, root, owner, member, raw = _fixture(tmp_path, monkeypatch)
    birth_capture_member(root, observation=owner, membership_selector=member, membership_raw=raw)
    context = {"schema_version": "website_site_task_context.v1", **{key: owner[key]
                for key in ("request_id", "scene_id", "capture_id")}, "confirmed": True,
               "confirmed_at": "2026-10-08T00:00:00Z", "description": "Synthetic owned preparation control",
               "success_criteria": {"successDefinition": None, "successRate": None,
                                    "cycleTimeSeconds": None, "unknown": True}}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    write_json(root / "pipeline/website_task_context.json", context)
    def fresh_owner(**kwargs):
        return observer.validate_observation(owner, **kwargs)
    monkeypatch.setattr(observer, "load_original_owner_observation", fresh_owner)
    monkeypatch.setattr(context_reader, "load_current_website_task_context", lambda **kwargs: dict(context))
    selected = {**{key: owner[key] for key in ("request_id", "scene_id", "capture_id")},
                "completion_marker_generation": owner["completion_marker"]["generation"],
                "producer_delivery_key": owner["producer_delivery"]["delivery_key"],
                "source_payload_sha256": listener.payload_sha256({"source_finalize": json.loads(raw)["source_finalize"]}),
                "task_context_digest": context["context_digest"]}
    def claim():
        state, ledger = listener._claim_job_lease(root, scene_id=owner["scene_id"],
            capture_id=owner["capture_id"], owner="synthetic-worker", lease_seconds=900,
            producer_delivery_key=selected["producer_delivery_key"], payload_sha256=selected["source_payload_sha256"])
        assert state == "claimed"
        return ledger
    ledger = claim()
    monkeypatch.setenv(service.INTAKE_CLIENT_SECRETS_ENV, json.dumps({"blueprint-webapp": "synthetic-secret", "other": "other-secret"}))
    monkeypatch.setenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, "false")
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.setenv("BLUEPRINT_LIVE_PIPELINE_INTAKE_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_OUTPUT_PATH", str(tmp_path / "absent.json"))
    monkeypatch.setenv(service.INTAKE_CLIENT_ROOTS_ENV, json.dumps({"blueprint-webapp": {owner["request_id"]: str(root)}}))
    return root, owner, context, selected, ledger, claim


def _finish(source, state="failed_retryable"):
    root, _, _, _, ledger, _ = source
    return listener._finish_job_lease(root, owner="synthetic-worker", token=ledger["lease_token"],
            update={"status": state, "last_error": "PRIVATE provider failure and path /sensitive/control"})


def _post(source, *, client="blueprint-webapp", body=None, nonce="status-read"):
    selected = source[3] if body is None else body
    encoded = json.dumps(selected)
    return TestClient(service.create_app()).post("/api/live-pipeline/website-preparation-status", content=encoded,
        headers=_signed_intake_headers("synthetic-secret" if client == "blueprint-webapp" else "other-secret",
                        encoded, nonce=nonce, client_id=client))


@pytest.mark.parametrize("state, expected", [("processing", "preparing"),
        ("failed_retryable", "failed_retryable"), ("retryable_blocked", "awaiting_inputs"),
        ("terminal_authority_ended", "authority_ended")])
def test_actual_birth_lease_signed_readback_is_bounded(prepared_source, state, expected):
    if state != "processing":
        _finish(prepared_source, state)
    response = _post(prepared_source)
    assert response.status_code == 200, response.text
    result = response.json()
    assert set(result) == status_reader.SELECTORS | {"schema_version", "attempt_count", "revision", "state", "code", "correlation_id", "status_digest"}
    assert result["state"] == expected and result["attempt_count"] == 1
    assert result["status_digest"] == cross_runtime_canonical_digest({key: val for key, val in result.items() if key != "status_digest"})
    assert response.headers["cache-control"] == "no-store"
    assert "PRIVATE" not in response.text and "owner-1" not in response.text and "/sensitive" not in response.text
    assert "completed" not in result["state"]


def test_signed_route_does_not_mutate_capture(prepared_source):
    root = prepared_source[0]
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert _post(prepared_source).status_code == 200
    after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.parametrize("field,value", [("completion_marker_generation", "999"),
    ("producer_delivery_key", "sha256:" + "9" * 64), ("source_payload_sha256", "9" * 64),
    ("task_context_digest", "sha256:" + "9" * 64)])
def test_signed_status_refuses_another_source(prepared_source, field, value):
    assert _post(prepared_source, body={**prepared_source[3], field: value}).status_code == 503


def test_route_rejects_caller_paths_and_foreign_client(prepared_source):
    assert _post(prepared_source, body={**prepared_source[3], "capture_root": "/tmp"}).status_code == 422
    assert _post(prepared_source, client="other", nonce="other-client").status_code == 403
    assert TestClient(service.create_app()).post("/api/live-pipeline/website-preparation-status", json=prepared_source[3]).status_code == 401


def test_retained_context_is_not_current_authority(prepared_source, monkeypatch):
    context = {**prepared_source[2], "description": "Changed operator statement"}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    monkeypatch.setattr(context_reader, "load_current_website_task_context", lambda **kwargs: context)
    assert _post(prepared_source).status_code == 503


def test_revision_change_during_current_context_refuses_snapshot(prepared_source, monkeypatch):
    root, _, context, _, ledger, _ = prepared_source
    def heartbeat(**kwargs):
        assert listener._heartbeat_job_lease(root, owner="synthetic-worker", token=ledger["lease_token"], lease_seconds=900)
        return context
    monkeypatch.setattr(context_reader, "load_current_website_task_context", heartbeat)
    assert _post(prepared_source).status_code == 503


def test_new_attempt_is_read_even_without_its_callback(prepared_source):
    _finish(prepared_source)
    failed = _post(prepared_source).json()
    new = prepared_source[5]()
    now = _post(prepared_source, nonce="new-attempt").json()
    assert failed["state"] == "failed_retryable" and now["state"] == "preparing"
    assert now["attempt_count"] == 2 and now["revision"] == new["revision"] > failed["revision"]


def test_completed_ledger_without_required_handoff_is_unavailable(prepared_source):
    _finish(prepared_source, "completed")
    # Actual lease commit deliberately lacks required source-stage handoff. This
    # negative control does not fabricate native/assessment output or publication.
    assert _post(prepared_source).status_code == 503


def test_withdrawal_precedes_retained_failure(prepared_source):
    from blueprint_pipeline.website_capture_withdrawal import acknowledge_withdrawal
    root, owner, _, _, _, _ = prepared_source
    _finish(prepared_source)
    acknowledge_withdrawal(capture_root=root, command={"schema_version": "website_capture_withdrawal.v1",
        **{key: owner[key] for key in ("request_id", "scene_id", "capture_id")},
        "withdrawal_id": "synthetic-withdrawal", "requested_at_iso": "2026-10-08T00:00:00Z"})
    assert _post(prepared_source).status_code == 503


def test_durable_delivery_retry_never_reopens_provider_work(prepared_source, monkeypatch):
    root, _, _, selected, _, _ = prepared_source
    _finish(prepared_source)
    before = listener._read_job_ledger(root)
    calls = []
    def sink(**kwargs):
        lock = (root / ".pipeline_job_ledger.json.lock").open("rb")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            lock.close()
        calls.append(kwargs)
        if len(calls) == 1:
            raise TimeoutError("PRIVATE response")
        return _acceptance(prepared_source)
    monkeypatch.setattr(context_reader, "website_webapp_request", sink)
    storage_root = root.parents[4]
    assert status_reader.reconcile_preparation_wakeups(storage_root)["attempted"] == 1
    pending = json.loads((root / status_reader.DELIVERY_FILE).read_text())
    assert pending["state"] == "pending" and pending["delivery_attempt_count"] == 1
    assert status_reader.reconcile_preparation_wakeups(storage_root)["delivered"] == 1
    assert status_reader.reconcile_preparation_wakeups(storage_root)["attempted"] == 0
    assert listener._read_job_ledger(root) == before
    assert len(calls) == 2 and all(call["payload"] == selected for call in calls)
    assert "PRIVATE" not in (root / status_reader.DELIVERY_FILE).read_text()


def test_new_revision_pending_is_not_consumed_by_delayed_delivery(prepared_source, monkeypatch):
    root = prepared_source[0]
    _finish(prepared_source)
    def delayed(**kwargs):
        prepared_source[5]()
        status_reader.retain_preparation_wakeup(root)
        return _acceptance(prepared_source)
    monkeypatch.setattr(context_reader, "website_webapp_request", delayed)
    assert status_reader.reconcile_preparation_wakeups(root.parents[4])["delivered"] == 0
    pending = json.loads((root / status_reader.DELIVERY_FILE).read_text())
    assert pending["state"] == "pending" and pending["revision"] == listener._read_job_ledger(root)["revision"]
    assert _post(prepared_source).json()["state"] == "preparing"


def test_actual_worker_refusal_retains_wakeup_and_signed_current_failure(prepared_source, monkeypatch):
    from blueprint_pipeline.website_scene_handoff import prepare_website_scene_handoff
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import capture_birth_source_projection
    root, owner, context, selected, _, _ = prepared_source
    _finish(prepared_source)
    birth = capture_birth_source_projection(root)
    delivery = json.loads(open(birth["birth_delivery_raw_ref"]["path"]).read())
    payload = {"bucket": owner["bucket"], "scene_id": owner["scene_id"], "capture_id": owner["capture_id"],
               "raw_prefix_uri": owner["raw_prefix_uri"],
               "source_finalize": {**delivery["source_finalize"], "event_id": "synthetic-finalize", "event_source": "synthetic-storage"},
               "source_membership_selector": birth["source_membership_selector"]}
    selected["source_payload_sha256"] = listener.payload_sha256(payload)
    write_json(root / "raw/manifest.json", {"capture_source": "browser_self_capture",
        "site_submission_id": owner["request_id"], "scene_id": owner["scene_id"], "capture_id": owner["capture_id"]})
    # Existing captured-member fixture, explicitly fake download/staging seam;
    # actual owner validation, lease, handoff refusal and final ledger execute.
    monkeypatch.setattr(listener, "stage_handoff_capture", lambda *args, **kwargs: root)
    runner_calls = []
    def held(**kwargs):
        assert kwargs["pipeline_lane"] == "qualification" and kwargs["run_evaluation_prep"] is False
        assert kwargs["resume_completed_stages"] is True
        runner_calls.append(kwargs)
        result = prepare_website_scene_handoff(descriptor={"scene_id": owner["scene_id"], "capture_id": owner["capture_id"],
            "metadata": {"site_task_context": context}}, clean_plate={"privacy_verified": True, "status": "noop"},
            provider_run={"status": "not_requested"}, capture_root=root, now=0)
        assert result["status"] == "awaiting_inputs" and result["blockers"] == ["website_reconstruction_pending"]
        raise PipelineError("website_reconstruction_pending") from StageError("website_scene_preparation", "website_reconstruction_pending")
    with pytest.raises(PipelineError, match="website_reconstruction_pending"):
        listener.process_handoff_payload(payload, storage_root=root.parents[4], provider="openai", run_e2e=held,
                                         run_e2e_enabled=False, stage_control_plane=True)
    assert len(runner_calls) == 1
    pending = json.loads((root / status_reader.DELIVERY_FILE).read_text())
    assert pending["state"] == "pending" and pending["selectors"] == selected
    response = _post(prepared_source)
    assert response.status_code == 200 and response.json()["state"] == "failed_retryable"
    assert response.json()["attempt_count"] == 2
    assert not (root / listener.JOB_OUTPUT_COMMIT_FILENAME).exists()
    calls = []
    def sink(**kwargs):
        calls.append(kwargs)
        # Wake-up consumption must reopen the real signed latest-read route.
        actual = _post(prepared_source, body=kwargs["payload"], nonce="wakeup-latest")
        assert actual.status_code == 200 and actual.json()["state"] == "failed_retryable"
        return _acceptance(prepared_source)
    monkeypatch.setattr(context_reader, "website_webapp_request", sink)
    assert status_reader.reconcile_preparation_wakeups(root.parents[4])["delivered"] == 1
    assert len(calls) == 1
    if trace_path := os.getenv("BLUEPRINT_RELIABILITY_STATUS_TRACE"):
        trace = {"layer": "actual source-selected worker/lease/handoff refusal/signed current-status API/durable delivery",
                 "simulation": ["owned synthetic member fixture", "fake source download/staging", "fake owner/context HTTP", "local wakeup sink"],
                 "provider_calls": 0, "new_unique_case_credit": 0, "payload": payload,
                 "worker_flags": {"run_e2e_enabled": False, "stage_control_plane": True},
                 "runner_lane": runner_calls[0]["pipeline_lane"], "website_manifest_source": "browser_self_capture",
                 "selectors": selected, "status": response.json(), "pending_delivery": pending,
                 "final_delivery": json.loads((root / status_reader.DELIVERY_FILE).read_text()),
                 "ledger": listener._read_job_ledger(root), "native_or_assessment_completion": False}
        write_json(Path(trace_path), trace)
        Path(trace_path).chmod(0o600)


@pytest.mark.parametrize("missing", ["ledger", "context", "birth_member"])
def test_missing_required_read_inputs_never_imply_success(prepared_source, missing):
    root = prepared_source[0]
    if missing == "ledger":
        (root / "pipeline_job_ledger.json").unlink()
    elif missing == "context":
        (root / "pipeline/website_task_context.json").unlink()
    else:
        from blueprint_pipeline.task_evaluation_scene_retirement_generations import capture_birth_source_projection
        Path(capture_birth_source_projection(root)["source_membership_raw_ref"]["path"]).unlink()
    assert _post(prepared_source).status_code == 503


def test_signed_route_rejects_nonce_replay_and_legacy_bearer(prepared_source, monkeypatch):
    assert _post(prepared_source).status_code == 200
    assert _post(prepared_source).status_code in {401, 409}
    monkeypatch.setenv(service.INTAKE_TOKEN_ENV, "synthetic-bearer")
    monkeypatch.setenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, "true")
    response = TestClient(service.create_app()).post("/api/live-pipeline/website-preparation-status",
        json=prepared_source[3], headers={"authorization": "Bearer synthetic-bearer"})
    assert response.status_code == 403


def test_owner_read_failure_is_fixed_unavailable(prepared_source, monkeypatch):
    def unavailable(**kwargs):
        raise TimeoutError("PRIVATE owner transport details")
    monkeypatch.setattr(observer, "load_original_owner_observation", unavailable)
    response = _post(prepared_source)
    assert response.status_code == 503 and "PRIVATE" not in response.text


def test_bounded_reconcile_scan_is_fair_across_restarts(tmp_path, monkeypatch):
    # Directory-count control only; no source authority/completion is fabricated.
    roots = []
    for n in range(3):
        root = tmp_path / f"bucket/scenes/site-{n}/captures/walkthrough-{n}"
        write_json(root / "pipeline_job_ledger.json", {"test_scan_control": True})
        roots.append(root)
    visited = []
    monkeypatch.setattr(status_reader, "retain_preparation_wakeup", lambda root: visited.append(root))
    for _ in range(3):
        status_reader.reconcile_preparation_wakeups(tmp_path, limit=1)
    assert visited == roots
    assert json.loads((tmp_path / ".website_preparation_delivery_cursor.json").read_text())["schema_version"] == "website_preparation_delivery_cursor.v1"


def _acceptance(source):
    status = status_reader.read_preparation_status(capture_root=source[0], selectors=source[3])
    return {"schema_version": "website_preparation_status_acceptance.v1", "accepted": True,
            **source[3], "status_digest": status["status_digest"], "state": status["state"],
            "native_execution_complete": False,
            "correlation_id": "bp-prep-" + status["status_digest"].removeprefix("sha256:")[:16]}


@pytest.mark.parametrize("fault", ["arbitrary_200", "not_accepted", "wrong_source", "native_complete",
                                  "bad_digest", "wrong_correlation", "extra_sensitive_field"])
def test_arbitrary_or_unbound_200_cannot_consume_durable_delivery(prepared_source, monkeypatch, fault):
    root = prepared_source[0]
    _finish(prepared_source)
    ack = _acceptance(prepared_source)
    if fault == "arbitrary_200":
        ack = {}
    elif fault == "not_accepted":
        ack["accepted"] = False
    elif fault == "wrong_source":
        ack["producer_delivery_key"] = "sha256:" + "9" * 64
    elif fault == "native_complete":
        ack["native_execution_complete"] = True
    elif fault == "bad_digest":
        ack["status_digest"] = "PRIVATE unverified provider result"
    elif fault == "wrong_correlation":
        ack["correlation_id"] = "bp-prep-" + "9" * 16
    else:
        ack["private_error"] = "PRIVATE body"
    monkeypatch.setattr(context_reader, "website_webapp_request", lambda **kwargs: ack)
    assert status_reader.reconcile_preparation_wakeups(root.parents[4])["delivered"] == 0
    assert json.loads((root / status_reader.DELIVERY_FILE).read_text())["state"] == "pending"


@pytest.mark.parametrize("transition", ["preparing_to_failed", "failed_to_new_attempt"])
def test_old_receipt_cannot_acknowledge_already_new_pending_revision(prepared_source, monkeypatch, transition):
    root = prepared_source[0]
    if transition == "preparing_to_failed":
        old = _acceptance(prepared_source)
        _finish(prepared_source)
    else:
        _finish(prepared_source)
        old = _acceptance(prepared_source)
        prepared_source[5]()
    current = status_reader.read_preparation_status(capture_root=root, selectors=prepared_source[3])
    assert old["status_digest"] != current["status_digest"]
    monkeypatch.setattr(context_reader, "website_webapp_request", lambda **kwargs: old)
    assert status_reader.reconcile_preparation_wakeups(root.parents[4])["delivered"] == 0
    delivery = json.loads((root / status_reader.DELIVERY_FILE).read_text())
    assert delivery["state"] == "pending" and delivery["revision"] == current["revision"]
    assert delivery["last_error_type"] == "PreparationStatusUnavailable"


def test_receipt_newer_than_preflight_matches_fresh_status(prepared_source, monkeypatch):
    root = prepared_source[0]
    _finish(prepared_source)
    def advance_without_retention(**kwargs):
        prepared_source[5]()
        return _acceptance(prepared_source)
    monkeypatch.setattr(context_reader, "website_webapp_request", advance_without_retention)
    assert status_reader.reconcile_preparation_wakeups(root.parents[4])["delivered"] == 1
    delivery = json.loads((root / status_reader.DELIVERY_FILE).read_text())
    assert delivery["state"] == "delivered" and delivery["acceptance"]["state"] == "preparing"


def test_ledger_change_after_postflight_cannot_commit_old_receipt(prepared_source, monkeypatch):
    root = prepared_source[0]
    _finish(prepared_source)
    monkeypatch.setattr(context_reader, "website_webapp_request", lambda **kwargs: _acceptance(prepared_source))
    read = status_reader.read_preparation_status
    calls = 0
    def advance_after_snapshot(**kwargs):
        nonlocal calls
        result = read(**kwargs)
        calls += 1
        # Preflight, callback's own read, then postflight authority read.
        if calls == 3:
            prepared_source[5]()
        return result
    monkeypatch.setattr(status_reader, "read_preparation_status", advance_after_snapshot)
    assert status_reader.reconcile_preparation_wakeups(root.parents[4])["delivered"] == 0
    assert json.loads((root / status_reader.DELIVERY_FILE).read_text())["state"] == "pending"
