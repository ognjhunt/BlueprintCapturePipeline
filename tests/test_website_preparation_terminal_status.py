# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_preparation_terminal.py
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/website_preparation_status_http.py
#   src/blueprint_pipeline/pubsub_handoff_listener.py
#   src/blueprint_pipeline/pubsub_handoff_scene_operations.py
"""Beta journey L3/L6: every website capture's preparation reaches a state the site is told.

Real birth, lease, ledger, signed status route, wake-up and Pub/Sub pull code
run here; only the owner/context transport, the WebApp callback and the Pub/Sub
client are local fakes. No provider, network or production state is touched.
"""
import json
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import google.cloud
import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline import capture_original_owner_observer as observer
from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_preparation_status as status_reader
from blueprint_pipeline import website_preparation_terminal as terminal
from blueprint_pipeline import website_task_context as context_reader
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_scene_retirement_generations import (
    birth_capture_member, capture_birth_source_projection,
)
from tests.test_capture_generation_birth import _fixture
from tests.test_live_pipeline_intake_service import _signed_intake_headers

BUDGET = "website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted"
CAPACITY = "pubsub_handoff_staging_capacity_insufficient"
V1_FIELDS = status_reader.SELECTORS | {"schema_version", "attempt_count", "revision", "state", "code",
                                       "correlation_id", "status_digest"}


@pytest.fixture
def website_capture(tmp_path, monkeypatch):
    """One born browser capture with its real handoff payload and a claimed ledger."""
    _, _, root, owner, member, raw = _fixture(tmp_path, monkeypatch)
    birth_capture_member(root, observation=owner, membership_selector=member, membership_raw=raw)
    context = {"schema_version": "website_site_task_context.v1", **{key: owner[key]
                for key in ("request_id", "scene_id", "capture_id")}, "confirmed": True,
               "confirmed_at": "2026-10-08T00:00:00Z", "description": "Synthetic owned preparation control",
               "success_criteria": {"successDefinition": None, "successRate": None,
                                    "cycleTimeSeconds": None, "unknown": True}}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    write_json(root / "pipeline/website_task_context.json", context)
    monkeypatch.setattr(observer, "load_original_owner_observation",
                        lambda **kwargs: observer.validate_observation(owner, **kwargs))
    current = {"context": dict(context)}
    monkeypatch.setattr(context_reader, "load_current_website_task_context", lambda **kwargs: dict(current["context"]))
    birth = capture_birth_source_projection(root)
    delivery = json.loads(Path(birth["birth_delivery_raw_ref"]["path"]).read_text())
    payload = {"bucket": owner["bucket"], "scene_id": owner["scene_id"], "capture_id": owner["capture_id"],
               "raw_prefix_uri": owner["raw_prefix_uri"],
               "source_finalize": {**delivery["source_finalize"], "event_id": "synthetic-finalize",
                                   "event_source": "synthetic-storage"},
               "source_membership_selector": birth["source_membership_selector"]}
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = listener.payload_sha256(data)
    selected = {**{key: owner[key] for key in ("request_id", "scene_id", "capture_id")},
                "completion_marker_generation": owner["completion_marker"]["generation"],
                "producer_delivery_key": owner["producer_delivery"]["delivery_key"],
                "source_payload_sha256": digest, "task_context_digest": context["context_digest"]}
    storage_root = root.parents[4]
    monkeypatch.setenv(service.INTAKE_CLIENT_SECRETS_ENV, json.dumps({"blueprint-webapp": "synthetic-secret"}))
    monkeypatch.setenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, "false")
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.setenv("BLUEPRINT_LIVE_PIPELINE_INTAKE_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_OUTPUT_PATH", str(tmp_path / "absent.json"))
    monkeypatch.setenv(service.INTAKE_CLIENT_ROOTS_ENV, json.dumps({"blueprint-webapp": {owner["request_id"]: str(root)}}))
    # Hermetic: the derived-root fallback must never see a host directory.
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT", str(tmp_path / "no-handoff-root"))
    return types.SimpleNamespace(root=root, owner=owner, context=context, current=current, payload=payload,
                                 data=data, digest=digest, selected=selected, storage_root=storage_root)


def _claim(source, *, final=False, lease_seconds=900, now=None):
    state, ledger = listener._claim_job_lease(source.root, scene_id=source.owner["scene_id"],
        capture_id=source.owner["capture_id"], owner="synthetic-worker", lease_seconds=lease_seconds,
        producer_delivery_key=source.selected["producer_delivery_key"], payload_sha256=source.digest,
        final_delivery_attempt=final, now=now)
    assert state == "claimed"
    return ledger


def _finish(source, ledger, **update):
    return listener._finish_job_lease(source.root, owner="synthetic-worker", token=ledger["lease_token"],
                                      update=update)


def _read(source, version=status_reader.STATUS_V1):
    return status_reader.read_preparation_status(capture_root=source.root, selectors=source.selected,
                                                 schema_version=version)


def _post(source, *, version=None, nonce="status-read", body=None):
    encoded = json.dumps(source.selected if body is None else body)
    headers = _signed_intake_headers("synthetic-secret", encoded, nonce=nonce, client_id="blueprint-webapp")
    if version is not None:
        headers[status_reader.VERSION_HEADER] = version
    return TestClient(service.create_app()).post("/api/live-pipeline/website-preparation-status",
                                                 content=encoded, headers=headers)


def _acceptance(source, version):
    status = _read(source, version)
    return {"schema_version": status["schema_version"].replace("status.", "status_acceptance."), "accepted": True,
            **source.selected, "status_digest": status["status_digest"], "state": status["state"],
            "native_execution_complete": False,
            "correlation_id": "bp-prep-" + status["status_digest"].removeprefix("sha256:")[:16]}


# --------------------------------------------------------- versioned receipts

@pytest.mark.parametrize("update, v1_state, v2_state", [
    ({}, "preparing", "preparing"),
    ({"status": "failed_retryable", "last_error": "PRIVATE stage failure"}, "failed_retryable", "failed_retryable"),
    ({"status": "retryable_blocked", "retry_blockers": ["website_reconstruction_pending"]},
     "awaiting_inputs", "awaiting_inputs"),
    ({"status": "retryable_blocked", "retry_blockers": [CAPACITY], "last_error": CAPACITY},
     "awaiting_inputs", "capacity_delayed"),
    ({"status": "failed_retryable", "last_error": "StageError: control_plane_disk_budget_exceeded:handoff_staging:"
      "need_bytes=9:available_bytes=1:free_bytes=1:floor_bytes=0:reserved_bytes=0"},
     "failed_retryable", "capacity_delayed"),
    # A budget refusal is never shown as waiting for capacity.
    ({"status": "failed_retryable", "last_error": f"{BUDGET}; control_plane_disk_budget_exceeded:x"},
     "failed_retryable", "failed_retryable"),
    ({"status": "terminal_authority_ended", "terminal_code": "consent_expired"}, "authority_ended", "authority_ended"),
])
def test_v1_receipt_is_unchanged_and_v2_separates_capacity(website_capture, update, v1_state, v2_state):
    ledger = _claim(website_capture)
    if update:
        _finish(website_capture, ledger, **update)
    old = _post(website_capture)
    assert old.status_code == 200, old.text
    assert set(old.json()) == V1_FIELDS
    assert old.json()["schema_version"] == "website_preparation_status.v1" and old.json()["state"] == v1_state
    new = _post(website_capture, version="2", nonce="status-read-v2")
    assert new.status_code == 200, new.text
    value = new.json()
    assert set(value) == V1_FIELDS | {"terminal_reason"}
    assert value["schema_version"] == "website_preparation_status.v2" and value["state"] == v2_state
    assert value["terminal_reason"] is None and value["code"] == status_reader._CODES[v2_state]
    assert value["status_digest"] == cross_runtime_canonical_digest(
        {key: item for key, item in value.items() if key != "status_digest"})
    assert value["correlation_id"] == old.json()["correlation_id"] and value["status_digest"] != old.json()["status_digest"]
    assert "PRIVATE" not in new.text and "need_bytes" not in new.text


@pytest.mark.parametrize("header, expected", [(None, "website_preparation_status.v1"),
    ("1", "website_preparation_status.v1"), ("2", "website_preparation_status.v2"),
    ("3", "website_preparation_status.v2"), ("two", "website_preparation_status.v1"),
    ("02", "website_preparation_status.v1"), ("9" * 40, "website_preparation_status.v1")])
def test_status_version_is_opt_in_and_capped_at_newest_known(header, expected):
    assert status_reader.requested_status_version(header) == expected


# ------------------------------------------------------------ dead-lettering

class _Subscriber:
    def __init__(self, received):
        self.received, self.acknowledged, self.deadlines = list(received), [], []

    def pull(self, *, request, timeout):
        rows, self.received = self.received, []
        return types.SimpleNamespace(received_messages=rows)

    def acknowledge(self, *, request):
        self.acknowledged.extend(request["ack_ids"])

    def modify_ack_deadline(self, *, request):
        self.deadlines.append(request)


def _pull(monkeypatch, source, *, attempt, process, data=None):
    pubsub_v1 = types.SimpleNamespace()
    monkeypatch.setattr(google.cloud, "pubsub_v1", pubsub_v1, raising=False)
    subscriber = _Subscriber([types.SimpleNamespace(ack_id="ack-1", delivery_attempt=attempt,
        message=types.SimpleNamespace(message_id="msg-1", data=data or source.data, attributes={}))])
    monkeypatch.setattr(pubsub_v1, "SubscriberClient", lambda: subscriber, raising=False)
    monkeypatch.setattr(listener, "process_handoff_payload", process)

    def webapp_unavailable(**_):
        # The tick's own delivery retry may run first; it keeps the wake-up pending.
        raise RuntimeError("synthetic WebApp unavailable")
    monkeypatch.setattr(context_reader, "website_webapp_request", webapp_unavailable)
    assert listener.pull_and_process(subscription="projects/p/subscriptions/s", storage_root=source.storage_root,
                                     provider="openai", max_messages=1) == 0
    assert subscriber.acknowledged == []  # Pub/Sub still owns the dead-letter hand-off.
    return subscriber


def _failing_worker(source, received, **update):
    def process(payload, **kwargs):
        received.append(kwargs)
        ledger = _claim(source, final=kwargs.get("final_delivery_attempt", False))
        _finish(source, ledger, **update)
        if update["status"] == "retryable_blocked":
            return {"schema_version": "v1", "status": "retryable_blocked", "queue_disposition": "retryable",
                    "capture_root": str(source.root), "blockers": update.get("retry_blockers", [])}
        raise RuntimeError("synthetic retryable failure")
    return process


@pytest.mark.parametrize("update, reason", [
    ({"status": "failed_retryable", "last_error": f"{BUDGET}"}, "budget_exhausted"),
    ({"status": "retryable_blocked", "retry_blockers": [CAPACITY], "last_error": CAPACITY}, "capacity_blocked"),
    ({"status": "failed_retryable", "last_error": "website_scene_assembly_coverage_key_missing"}, "retry_exhausted"),
    ({"status": "retryable_blocked", "retry_blockers": ["website_reconstruction_pending"]}, "retry_exhausted"),
])
def test_last_delivery_records_terminal_reason_and_queues_one_wakeup(website_capture, monkeypatch, update, reason):
    received = []
    _pull(monkeypatch, website_capture, attempt=5, process=_failing_worker(website_capture, received, **update))
    assert received[0]["final_delivery_attempt"] is True
    ledger = listener._read_job_ledger(website_capture.root)
    record = ledger[terminal.TERMINAL_FIELD]
    assert record["reason"] == reason and record["basis"] == "delivery_attempts_exhausted"
    assert ledger["status"] == update["status"]  # The job status itself is unchanged.
    v2 = _read(website_capture, status_reader.STATUS_V2)
    assert v2["state"] == "failed_terminal" and v2["terminal_reason"] == reason
    assert v2["code"] == "preparation_failed_terminal"
    # A WebApp that predates v2 still reads the exact v1 meaning of the ledger.
    assert _read(website_capture)["state"] == status_reader._LEDGER_STATES[update["status"]]
    pending = json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())
    assert pending["state"] == "pending" and pending["revision"] == ledger["revision"]
    assert pending["selectors"] == website_capture.selected


def test_earlier_delivery_or_other_source_records_nothing(website_capture, monkeypatch):
    received = []
    update = {"status": "failed_retryable", "last_error": BUDGET}
    _pull(monkeypatch, website_capture, attempt=4, process=_failing_worker(website_capture, received, **update))
    assert "final_delivery_attempt" not in received[0]
    assert terminal.TERMINAL_FIELD not in listener._read_job_ledger(website_capture.root)
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "failed_retryable"
    # The storage trigger's duplicate handoff (no membership selector) dead-letters
    # too, but it never claimed this ledger, so it cannot end the site's preparation.
    dud = json.dumps({key: value for key, value in website_capture.payload.items()
                      if key != "source_membership_selector"}, sort_keys=True).encode()
    _pull(monkeypatch, website_capture, attempt=5, data=dud,
          process=lambda payload, **_: {"status": "capture_original_birth_unavailable_retryable",
                                        "queue_disposition": "retryable", "capture_root": str(website_capture.root),
                                        "blockers": ["capture_original_birth_unavailable"]})
    assert terminal.TERMINAL_FIELD not in listener._read_job_ledger(website_capture.root)


def test_terminal_record_is_dropped_by_the_next_claim(website_capture):
    ledger = _claim(website_capture)
    _finish(website_capture, ledger, status="failed_retryable", last_error=BUDGET)
    assert terminal.record_delivery_exhaustion(website_capture.root, payload_sha256=website_capture.digest)
    assert terminal.record_delivery_exhaustion(website_capture.root, payload_sha256=website_capture.digest) is None
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "failed_terminal"
    redrive = _claim(website_capture)  # An operator re-drive starts a fresh attempt.
    assert terminal.TERMINAL_FIELD not in redrive
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "preparing"


def test_record_refuses_other_payloads_and_live_leases(website_capture):
    ledger = _claim(website_capture)
    assert terminal.record_delivery_exhaustion(website_capture.root, payload_sha256=website_capture.digest) is None
    _finish(website_capture, ledger, status="failed_retryable", last_error="x")
    assert terminal.record_delivery_exhaustion(website_capture.root, payload_sha256="0" * 64) is None
    assert terminal.record_delivery_exhaustion(website_capture.root, payload_sha256="not-a-digest") is None
    assert terminal.record_delivery_exhaustion(website_capture.root / "absent", payload_sha256=website_capture.digest) is None


@pytest.mark.parametrize("mutation", ["observed_status", "attempt_count", "source_payload_sha256", "lease_token",
                                      "reason", "schema_version"])
def test_stale_or_forged_terminal_record_is_ignored(website_capture, mutation):
    ledger = _claim(website_capture)
    ledger = _finish(website_capture, ledger, status="failed_retryable", last_error="x")
    committed = terminal.record_delivery_exhaustion(website_capture.root, payload_sha256=website_capture.digest)
    record = dict(committed[terminal.TERMINAL_FIELD])
    record[mutation] = {"observed_status": "processing", "attempt_count": 99, "source_payload_sha256": "f" * 64,
                        "lease_token": "other", "reason": "provider_failed", "schema_version": "v0"}[mutation]
    write_json(website_capture.root / listener.JOB_LEDGER_FILENAME,
               {**committed, terminal.TERMINAL_FIELD: record, "revision": committed["revision"] + 1})
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "failed_retryable"


# -------------------------------------------------- crashed final attempts

def test_abandoned_final_lease_becomes_terminal_through_the_sweep(website_capture, monkeypatch):
    started = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    ledger = _claim(website_capture, final=True, lease_seconds=900, now=started)
    assert ledger["final_delivery_attempt"] is True
    expires = started + timedelta(seconds=900)
    grace = timedelta(seconds=terminal.ABANDONED_LEASE_GRACE_SECONDS)
    assert terminal.record_abandoned_final_delivery(website_capture.root, now=expires + grace - timedelta(seconds=1)) is None
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "preparing"
    calls = []

    def webapp(**kwargs):
        calls.append(kwargs)
        return _acceptance(website_capture, status_reader.STATUS_V2)
    monkeypatch.setattr(context_reader, "website_webapp_request", webapp)
    real = terminal.record_abandoned_final_delivery
    monkeypatch.setattr(terminal, "record_abandoned_final_delivery",
                        lambda root, **_: real(root, now=expires + grace + timedelta(seconds=1)))
    assert status_reader.reconcile_preparation_wakeups(website_capture.storage_root)["delivered"] == 1
    current = listener._read_job_ledger(website_capture.root)
    assert current["status"] == "processing"
    assert current[terminal.TERMINAL_FIELD]["basis"] == "final_delivery_lease_abandoned"
    assert current[terminal.TERMINAL_FIELD]["lease_token"] == ledger["lease_token"]
    delivered = json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())
    assert delivered["state"] == "delivered" and delivered["acceptance"]["state"] == "failed_terminal"
    assert len(calls) == 1


def test_expired_lease_on_an_earlier_attempt_waits_for_redelivery(website_capture):
    started = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    _claim(website_capture, final=False, lease_seconds=60, now=started)
    assert terminal.record_abandoned_final_delivery(website_capture.root, now=started + timedelta(days=3)) is None
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "preparing"


def test_slow_final_worker_that_finishes_supersedes_the_abandoned_record(website_capture):
    started = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    ledger = _claim(website_capture, final=True, lease_seconds=60, now=started)
    assert terminal.record_abandoned_final_delivery(website_capture.root, now=started + timedelta(hours=1))
    _finish(website_capture, ledger, status="failed_retryable", last_error="late failure")
    # The record described the processing attempt, not this finished one.
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "failed_retryable"


# -------------------------------------------- failures before a context exists

def test_failure_before_context_is_reported_against_current_context(website_capture):
    (website_capture.root / "pipeline/website_task_context.json").unlink()
    ledger = _claim(website_capture)
    _finish(website_capture, ledger, status="failed_retryable", last_error="staging failed")
    status_reader.retain_preparation_wakeup(website_capture.root)
    pending = json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())
    assert pending["selectors"]["task_context_digest"] == website_capture.context["context_digest"]
    response = _post(website_capture, version="2")
    assert response.status_code == 200 and response.json()["state"] == "failed_retryable"
    # The site's later brief edit binds the next report to the new context only.
    changed = {**website_capture.context, "description": "Edited job"}
    changed["context_digest"] = canonical_digest(changed, digest_field="context_digest")
    website_capture.current["context"] = changed
    assert _post(website_capture, nonce="status-read-stale").status_code == 503


def test_handoff_never_reports_without_its_retained_context(website_capture):
    (website_capture.root / "pipeline/website_task_context.json").unlink()
    ledger = _claim(website_capture)
    _finish(website_capture, ledger, status="completed")
    assert _post(website_capture, version="2").status_code == 503


def test_symlinked_context_is_refused_not_treated_as_absent(website_capture, tmp_path):
    path = website_capture.root / "pipeline/website_task_context.json"
    target = tmp_path / "elsewhere.json"
    target.write_text(path.read_text())
    path.unlink()
    path.symlink_to(target)
    _claim(website_capture)
    assert _post(website_capture).status_code == 503


# ------------------------------------------------- acceptance compatibility

@pytest.mark.parametrize("version", [status_reader.STATUS_V1, status_reader.STATUS_V2])
def test_old_and_new_webapp_acceptances_both_consume_delivery(website_capture, monkeypatch, version):
    ledger = _claim(website_capture)
    _finish(website_capture, ledger, status="failed_retryable", last_error=BUDGET)
    terminal.record_delivery_exhaustion(website_capture.root, payload_sha256=website_capture.digest)
    status_reader.retain_preparation_wakeup(website_capture.root)
    monkeypatch.setattr(context_reader, "website_webapp_request", lambda **_: _acceptance(website_capture, version))
    assert status_reader.reconcile_preparation_wakeups(website_capture.storage_root)["delivered"] == 1
    accepted = json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())["acceptance"]
    assert accepted["state"] == ("failed_terminal" if version == status_reader.STATUS_V2 else "failed_retryable")


@pytest.mark.parametrize("fault", ["v2_state_in_v1", "v1_digest_as_v2", "unknown_version"])
def test_mismatched_acceptance_version_cannot_consume_delivery(website_capture, monkeypatch, fault):
    ledger = _claim(website_capture)
    _finish(website_capture, ledger, status="failed_retryable", last_error=BUDGET)
    terminal.record_delivery_exhaustion(website_capture.root, payload_sha256=website_capture.digest)
    status_reader.retain_preparation_wakeup(website_capture.root)
    ack = _acceptance(website_capture, status_reader.STATUS_V2)
    if fault == "v2_state_in_v1":
        ack["schema_version"] = "website_preparation_status_acceptance.v1"
    elif fault == "v1_digest_as_v2":
        old = _acceptance(website_capture, status_reader.STATUS_V1)
        ack.update(status_digest=old["status_digest"], correlation_id=old["correlation_id"])
    else:
        ack["schema_version"] = "website_preparation_status_acceptance.v3"
    monkeypatch.setattr(context_reader, "website_webapp_request", lambda **_: ack)
    assert status_reader.reconcile_preparation_wakeups(website_capture.storage_root)["delivered"] == 0
    assert json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())["state"] == "pending"


# ------------------------------------------------------ supplements and roots

@pytest.mark.parametrize("capture, valid", [("supplement-" + "a" * 40, True), ("supplement-" + "A" * 40, False),
    ("supplement-" + "a" * 39, False), ("supplement-other", False), ("walkthrough-other", False)])
def test_supplement_captures_of_the_request_are_valid_selectors(capture, valid):
    selected = {"request_id": "req-1", "scene_id": "site-req-1", "capture_id": capture,
                "completion_marker_generation": "19", "producer_delivery_key": "sha256:" + "a" * 64,
                "source_payload_sha256": "b" * 64, "task_context_digest": "sha256:" + "c" * 64}
    if valid:
        assert status_reader.validate_selectors(selected) == selected
    else:
        with pytest.raises(status_reader.PreparationStatusUnavailable):
            status_reader.validate_selectors(selected)


def test_sweep_visits_walkthrough_and_supplement_ledgers_only(tmp_path, monkeypatch):
    names = ["walkthrough-1", "supplement-" + "b" * 40, "creator-capture-1"]
    for name in names:
        write_json(tmp_path / f"bucket/scenes/site-1/captures/{name}/pipeline_job_ledger.json", {"test_scan_control": True})
    visited = []
    monkeypatch.setattr(status_reader, "retain_preparation_wakeup", lambda root: visited.append(root.name))
    for _ in range(3):
        status_reader.reconcile_preparation_wakeups(tmp_path, limit=1)
    assert sorted(set(visited)) == sorted(names[:2])


def test_unmapped_capture_is_read_from_the_listener_tree(website_capture, monkeypatch):
    monkeypatch.delenv(service.INTAKE_CLIENT_ROOTS_ENV)
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT", str(website_capture.storage_root))
    _claim(website_capture)
    response = _post(website_capture, version="2")
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "preparing"


@pytest.mark.parametrize("configured", [None, "", "relative/handoffs"])
def test_fallback_needs_an_explicit_absolute_listener_root(website_capture, monkeypatch, configured):
    monkeypatch.delenv(service.INTAKE_CLIENT_ROOTS_ENV)
    if configured is None:
        monkeypatch.delenv("BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT")
    else:
        monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT", configured)
    _claim(website_capture)
    assert _post(website_capture).status_code == 503


def test_capture_in_two_buckets_is_ambiguous_and_unavailable(website_capture, monkeypatch):
    monkeypatch.delenv(service.INTAKE_CLIENT_ROOTS_ENV)
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT", str(website_capture.storage_root))
    twin = website_capture.storage_root / "other-bucket" / "scenes" / website_capture.owner["scene_id"] / "captures"
    (twin / website_capture.owner["capture_id"]).mkdir(parents=True)
    _claim(website_capture)
    assert _post(website_capture).status_code == 503
