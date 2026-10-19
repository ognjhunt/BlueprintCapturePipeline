# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/website_preparation_status_http.py
#   src/blueprint_pipeline/website_task_context.py
#   src/blueprint_pipeline/website_assessment_resume.py
#   src/blueprint_pipeline/pubsub_handoff_listener.py
#   src/blueprint_pipeline/pubsub_handoff_scene_operations.py
"""Beta journey L3/L6: a Blueprint-funded preparation reaches its site after its grant.

On 2026-10-18 no website preparation status had ever been delivered. A
preparation-only task context is unconfirmed, and its status read required a
current grant. Grants expire a day after issue and an assessment wait has none
yet, so every read refused before any callback was made.

Real birth, listener admission, lease, ledger, signed status route and sweep run
here. Only the WebApp's HTTP answers (status and typed code as production returns
them), the owner transport and the WebApp's callback are local fakes. No provider,
network or production state is touched, and a status read never creates, renews or
spends a grant. A preparation with no grant yet stays unread: the WebApp cannot
verify it, and the paid assessment resume reads status before it asks for one.
"""
import copy
import json
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline import capture_original_owner_observer as observer
from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_assessment_resume as resume
from blueprint_pipeline import website_preparation_status as status_reader
from blueprint_pipeline import website_task_context as context_reader
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_scene_retirement_generations import (
    birth_capture_member, capture_birth_source_projection,
)
from tests.test_live_pipeline_intake_service import _signed_intake_headers
from tests.test_website_preparation_contract_v2 import VECTOR as CONTRACT
from tests.test_website_preparation_owner_data import prep_birth_fixture

EXPIRED = "consent_expired"
PENDING = "website_assessment_preparation_pending"
MISSING = "website_scene_sponsorship_missing"
ENDED = f"website_control_scene-sponsorship_http_409:{EXPIRED}"


def _grant(task, *, expires_in=300.0):
    grant = copy.deepcopy(CONTRACT["sponsorship_response"])
    grant.update({key: task[key] for key in ("request_id", "scene_id", "capture_id")})
    grant["task_context_digest"] = task["context_digest"]
    grant["consent"]["rights_reference"] = cross_runtime_canonical_digest(task["capture_rights"])
    grant["consent"]["accepted_at_epoch"] = time.time() - 1
    grant["expires_at_epoch"] = time.time() + expires_in
    grant["assessment_preparation_proposal"].update({key: task[key] for key in ("request_id", "capture_id")})
    grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    return grant


class WebApp:
    """The WebApp's signed control routes, answering as production does.

    `sponsorship` is a typed 409 code, "grant" (a current grant), "expired_body"
    (a grant past expiry by this clock), "unavailable" (503), "expired_503" (the
    expiry code under another status) or "timeout"; or a callable choosing one of
    those from the request payload.
    """

    def __init__(self, task, post):
        self.task, self.post = task, post
        self.sponsorship = EXPIRED
        self.calls = []

    def __call__(self, url, **kwargs):
        operation = url.rsplit("/", 1)[1]
        payload = json.loads(kwargs["data"])
        self.calls.append((operation, payload))
        if operation == "task-context":
            return SimpleNamespace(body=json.dumps(self.task).encode())
        if operation == "scene-sponsorship":
            answer = self.sponsorship(payload) if callable(self.sponsorship) else self.sponsorship
            if answer == "timeout":
                raise TimeoutError("synthetic WebApp timeout")
            if answer == "unavailable":
                self._refuse(url, 503, "website_scene_sponsorship_unavailable")
            if answer == "expired_503":
                self._refuse(url, 503, EXPIRED)
            if answer in {"grant", "expired_body"}:
                grant = _grant(self.task, expires_in=300.0 if answer == "grant" else -1.0)
                return SimpleNamespace(body=json.dumps(grant).encode())
            self._refuse(url, 409, answer)
        if operation == "preparation-status":
            # syncWebsitePreparationStatus: two signed reads of the ledger that
            # agree, then the acceptance naming the receipt version it read.
            first, latest = self.post(payload, version="2"), self.post(payload, version="2")
            if (first.status_code, latest.status_code) != (200, 200) or first.json() != latest.json():
                self._refuse(url, 503, "website_preparation_status_unavailable")
            status = latest.json()
            acceptance = {"schema_version": status["schema_version"].replace("status.", "status_acceptance."),
                          "accepted": True, **payload, "status_digest": status["status_digest"],
                          "state": status["state"], "native_execution_complete": False,
                          "correlation_id": "bp-prep-" + status["status_digest"].removeprefix("sha256:")[:16]}
            return SimpleNamespace(body=json.dumps(acceptance).encode())
        raise AssertionError(f"website preparation status never calls {operation}")

    @staticmethod
    def _refuse(url, status, code):
        raise HTTPError(url, status, "refused", {}, BytesIO(json.dumps({"code": code}).encode()))


@pytest.fixture
def prep_capture(tmp_path, monkeypatch):
    """One born, Blueprint-funded browser capture whose task the site has not confirmed."""
    _, _, root, owner, member, raw = prep_birth_fixture(tmp_path, monkeypatch)
    birth_capture_member(root, observation=owner, membership_selector=member, membership_raw=raw,
                         expected_purpose="scene_preparation")
    task = copy.deepcopy(CONTRACT["task_context_response_preparation"])
    task.update({key: owner[key] for key in ("request_id", "scene_id", "capture_id")})
    task["task_items"] = []  # This capture fixture has no supplementary item photos.
    task["capture_rights"] = copy.deepcopy(owner["capture_rights"])
    task["context_digest"] = canonical_digest(task, digest_field="context_digest")
    assert task["confirmed"] is False and task["purpose"] == "scene_preparation"
    write_json(root / "pipeline/website_task_context.json", task)
    write_json(root / "raw/manifest.json", {"capture_source": "browser_self_capture"})
    monkeypatch.setattr(observer, "load_original_owner_observation",
                        lambda **kwargs: observer.validate_observation(owner, **kwargs))
    birth = capture_birth_source_projection(root, expected_purpose="scene_preparation")
    delivery = json.loads(Path(birth["birth_delivery_raw_ref"]["path"]).read_text())
    payload = {"bucket": owner["bucket"], "scene_id": owner["scene_id"], "capture_id": owner["capture_id"],
               "raw_prefix_uri": owner["raw_prefix_uri"],
               "source_finalize": {**delivery["source_finalize"], "event_id": "synthetic-finalize",
                                   "event_source": "synthetic-storage"},
               "source_membership_selector": birth["source_membership_selector"]}
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    selected = {**{key: owner[key] for key in ("request_id", "scene_id", "capture_id")},
                "completion_marker_generation": owner["completion_marker"]["generation"],
                "producer_delivery_key": owner["producer_delivery"]["delivery_key"],
                "source_payload_sha256": listener.payload_sha256(data),
                "task_context_digest": task["context_digest"]}
    storage_root = root.parents[4]
    monkeypatch.setenv(service.INTAKE_CLIENT_SECRETS_ENV, json.dumps({"blueprint-webapp": "synthetic-secret"}))
    monkeypatch.setenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, "false")
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.setenv("BLUEPRINT_LIVE_PIPELINE_INTAKE_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_OUTPUT_PATH", str(tmp_path / "absent.json"))
    # As in production: no per-request capture-root map, so the route finds the
    # capture in the listener's tree.
    monkeypatch.delenv(service.INTAKE_CLIENT_ROOTS_ENV, raising=False)
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT", str(storage_root))
    nonces = iter(range(1_000_000))

    def post(body, *, version=None):
        encoded = json.dumps(body)
        headers = _signed_intake_headers("synthetic-secret", encoded, nonce=f"status-{next(nonces)}",
                                         client_id="blueprint-webapp")
        if version is not None:
            headers[status_reader.VERSION_HEADER] = version
        return TestClient(service.create_app()).post("/api/live-pipeline/website-preparation-status",
                                                     content=encoded, headers=headers)

    webapp = WebApp(task, post)
    monkeypatch.setattr(context_reader, "safe_request", webapp)
    monkeypatch.setattr(context_reader, "load_pipeline_sync_token", lambda: "synthetic-sync-secret")
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.invalid/api/internal/pipeline/sync")
    # Selected staging is the birth's own workspace; nothing is downloaded.
    monkeypatch.setattr(listener, "stage_handoff_capture", lambda *args, **kwargs: root)
    return SimpleNamespace(root=root, owner=owner, task=task, data=data, selected=selected,
                           storage_root=storage_root, webapp=webapp, post=post)


def _admit(source):
    """The listener's own processing of this delivery, as on the production host."""
    return listener.process_handoff_payload(source.data, storage_root=source.storage_root, provider="openai",
        run_e2e=lambda **_: pytest.fail("an unadmitted preparation reached provider stages"))


def _claim(source):
    state, ledger = listener._claim_job_lease(source.root, scene_id=source.owner["scene_id"],
        capture_id=source.owner["capture_id"], owner="synthetic-worker", lease_seconds=900,
        producer_delivery_key=source.selected["producer_delivery_key"],
        payload_sha256=source.selected["source_payload_sha256"])
    assert state == "claimed"
    return ledger


def _finish(source, ledger, **update):
    return listener._finish_job_lease(source.root, owner="synthetic-worker", token=ledger["lease_token"],
                                      update=update)


def _read(source, version=status_reader.STATUS_V1):
    return status_reader.read_preparation_status(capture_root=source.root, selectors=source.selected,
                                                 schema_version=version)


def _files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _delivery(source):
    return json.loads((source.root / status_reader.DELIVERY_FILE).read_text())


def _wait_for_assessment(source):
    source.webapp.sponsorship = PENDING
    with pytest.raises(resume.AssessmentPreparationPending):
        _admit(source)
    ledger = listener._read_job_ledger(source.root)
    assert ledger["status"] == "failed_retryable" and ledger["last_error"] == resume.PENDING_CODE
    assert ledger["assessment_resume"]["task_context_digest"] == source.selected["task_context_digest"]
    return ledger


# ------------------------------------------------------------ expired grants

def test_an_expired_grant_is_reported_as_ended_and_delivered_without_new_authority(prep_capture):
    """The production shape: the listener ended preparation on consent_expired; now the site is told."""
    source = prep_capture
    result = _admit(source)
    ledger = listener._read_job_ledger(source.root)
    assert result["queue_disposition"] == "terminal_authority_ended"
    assert ledger["status"] == "terminal_authority_ended" and ledger["last_error"] == ENDED
    assert _delivery(source)["state"] == "pending" and _delivery(source)["delivery_attempt_count"] == 0
    before = _files(source.root)
    source.webapp.calls.clear()

    counts = status_reader.reconcile_preparation_wakeups(source.storage_root)

    assert counts == {"attempted": 1, "delivered": 1, "unavailable": 0}
    delivery = _delivery(source)
    assert delivery["state"] == "delivered" and delivery["delivery_attempt_count"] == 1
    assert delivery["acceptance"]["schema_version"] == "website_preparation_status_acceptance.v2"
    assert delivery["acceptance"]["state"] == "authority_ended"
    # Nothing was granted: reads only, and every grant read refused creation.
    assert {operation for operation, _ in source.webapp.calls} == {"task-context", "scene-sponsorship",
                                                                    "preparation-status"}
    grant_reads = [payload for operation, payload in source.webapp.calls if operation == "scene-sponsorship"]
    assert grant_reads and all(payload == {"request_id": source.selected["request_id"],
                                           "scene_id": source.selected["scene_id"], "purpose": "scene_preparation",
                                           "expected_task_context_digest": source.selected["task_context_digest"],
                                           "create": False} for payload in grant_reads)
    after = _files(source.root)
    assert {name for name in after if after[name] != before.get(name)} == {status_reader.DELIVERY_FILE}
    assert listener._read_job_ledger(source.root) == ledger


def test_the_signed_route_answers_ended_for_an_expired_grant(prep_capture):
    _admit(prep_capture)
    response = prep_capture.post(prep_capture.selected, version="2")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema_version"] == status_reader.STATUS_V2
    assert (body["state"], body["code"], body["terminal_reason"]) == (
        "authority_ended", "preparation_authority_ended", None)


@pytest.mark.parametrize("update, state", [
    (None, "preparing"),
    ({"status": "failed_retryable", "last_error": "synthetic retryable failure"}, "failed_retryable"),
    ({"status": "retryable_blocked", "retry_blockers": ["website_reconstruction_pending"]}, "awaiting_inputs"),
    ({"status": "terminal_authority_ended", "last_error": ENDED}, "authority_ended"),
])
def test_an_expired_grant_reports_the_ledger_never_a_state_of_its_own(prep_capture, update, state):
    ledger = _claim(prep_capture)
    if update is not None:
        _finish(prep_capture, ledger, **update)
    prep_capture.webapp.sponsorship = "grant"
    current = {version: _read(prep_capture, version) for version in (status_reader.STATUS_V1, status_reader.STATUS_V2)}
    prep_capture.webapp.sponsorship = EXPIRED
    for version, receipt in current.items():
        # The same ledger revision reads the same receipt whether or not the
        # grant has expired since, so a WebApp never sees one revision change.
        assert receipt["state"] == state and _read(prep_capture, version) == receipt


@pytest.mark.parametrize("answer", ["website_scene_sponsorship_changed", "website_task_context_changed",
                                    "source_revoked", MISSING, PENDING, "unavailable", "expired_503", "timeout",
                                    "expired_body"])
def test_any_other_grant_answer_still_refuses_the_read(prep_capture, answer):
    ledger = _claim(prep_capture)
    _finish(prep_capture, ledger, status="terminal_authority_ended", last_error=ENDED)
    status_reader.retain_preparation_wakeup(prep_capture.root)
    prep_capture.webapp.sponsorship = answer
    with pytest.raises(status_reader.PreparationStatusUnavailable):
        _read(prep_capture)
    assert prep_capture.post(prep_capture.selected).status_code == 503
    counts = status_reader.reconcile_preparation_wakeups(prep_capture.storage_root)
    assert counts["attempted"] == 0 and _delivery(prep_capture)["state"] == "pending"
    assert "preparation-status" not in {operation for operation, _ in prep_capture.webapp.calls}


@pytest.mark.parametrize("missing", ["retained_context", "handoff"])
def test_a_hand_off_is_never_reported_on_an_expired_grant_without_its_evidence(prep_capture, missing):
    if missing == "retained_context":
        (prep_capture.root / "pipeline/website_task_context.json").unlink()
    ledger = _claim(prep_capture)
    _finish(prep_capture, ledger, status="completed")
    for answer in (EXPIRED, "grant"):
        prep_capture.webapp.sponsorship = answer
        with pytest.raises(status_reader.PreparationStatusUnavailable):
            _read(prep_capture)


# ----------------------------------------------------------- assessment wait

def test_a_preparation_with_no_grant_yet_is_still_not_read_and_nothing_resumes(prep_capture, monkeypatch):
    """The WebApp cannot verify a preparation it has issued no grant for, and the paid
    assessment resume reads status before it asks for one. Both stay as they were."""
    _wait_for_assessment(prep_capture)
    monkeypatch.setattr(listener, "process_handoff_payload",
                        lambda *args, **kwargs: pytest.fail("a status read resumed a paid preparation"))
    for answer in (PENDING, MISSING):
        prep_capture.webapp.sponsorship = answer
        prep_capture.webapp.calls.clear()
        with pytest.raises(status_reader.PreparationStatusUnavailable):
            _read(prep_capture)
        assert status_reader.reconcile_preparation_wakeups(prep_capture.storage_root)["attempted"] == 0
        assert resume.reconcile_waiting_assessments(listener, storage_root=prep_capture.storage_root,
                                                    process_args={"provider": "openai"}) == {"resumed": 0, "pending": 1}
        grant_reads = [payload for operation, payload in prep_capture.webapp.calls if operation == "scene-sponsorship"]
        assert grant_reads and all(payload.get("create") is False for payload in grant_reads)
        assert "preparation-status" not in {operation for operation, _ in prep_capture.webapp.calls}
