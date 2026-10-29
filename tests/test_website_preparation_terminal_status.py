# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_preparation_terminal.py
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/website_preparation_status_http.py
#   src/blueprint_pipeline/pubsub_handoff_listener.py
#   src/blueprint_pipeline/pubsub_handoff_scene_operations.py
"""Beta journey L3/L6: every website capture's preparation reaches a state the site is told.

Real birth, lease, ledger, signed status route, wake-up and sweep code run here;
only the owner/context transport and the WebApp callback are local fakes, and
time is passed explicitly. No provider, network or production state is touched.
"""
import json
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def _claim(source, *, lease_seconds=900, now=None):
    state, ledger = listener._claim_job_lease(source.root, scene_id=source.owner["scene_id"],
        capture_id=source.owner["capture_id"], owner="synthetic-worker", lease_seconds=lease_seconds,
        producer_delivery_key=source.selected["producer_delivery_key"], payload_sha256=source.digest, now=now)
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
    ("3", "website_preparation_status.v3"), ("4", "website_preparation_status.v4"),
    ("5", "website_preparation_status.v4"),
    ("two", "website_preparation_status.v1"),
    ("02", "website_preparation_status.v1"), ("9" * 40, "website_preparation_status.v1")])
def test_status_version_is_opt_in_and_capped_at_newest_known(header, expected):
    assert status_reader.requested_status_version(header) == expected


# ------------------------------------------- silence proves delay, never exhaustion

LONG_AGO = datetime(2026, 1, 1, tzinfo=timezone.utc)  # The listener has been pulling since.
WINDOW = timedelta(seconds=terminal.DEFAULT_SILENCE_SECONDS)


def _stop(source, *, at, **update):
    """One claimed attempt that finished retryably at a fixed time."""
    return _finish(source, _claim(source), updated_at=at.isoformat(), **update)


def _end_silently(source, *, stopped):
    """What the sweep records once a full window of continuous listening has passed."""
    return terminal.record_unretried_preparation(source.root, alive_since=LONG_AGO,
                                                 now=stopped + WINDOW + timedelta(seconds=1))


def _listening_since(source, since, *, last_tick):
    write_json(source.storage_root / terminal.LIVENESS_FILE, {"schema_version": terminal.LIVENESS_SCHEMA_VERSION,
               "continuous_since": since.isoformat(), "last_tick_at": last_tick.isoformat()})


@pytest.mark.parametrize("update, cause", [
    ({"status": "failed_retryable", "last_error": f"{BUDGET}"}, "capture_budget"),
    ({"status": "retryable_blocked", "retry_blockers": [CAPACITY], "last_error": CAPACITY}, "control_plane_capacity"),
    ({"status": "failed_retryable", "last_error": "website_scene_assembly_coverage_key_missing"}, "queue_or_dependency_delay"),
    ({"status": "retryable_blocked", "retry_blockers": ["website_reconstruction_pending"]}, "queue_or_dependency_delay"),
])
def test_live_listener_and_backlog_emit_overdue_without_terminal_claim(website_capture, monkeypatch, update, cause):
    now = datetime.now(timezone.utc)
    _stop(website_capture, at=now - WINDOW - timedelta(minutes=5), **update)
    _listening_since(website_capture, now - WINDOW - timedelta(minutes=10), last_tick=now - timedelta(minutes=1))
    monkeypatch.setattr(context_reader, "website_webapp_request",
                        lambda **_: _acceptance(website_capture, status_reader.STATUS_V2))
    assert status_reader.reconcile_preparation_wakeups(website_capture.storage_root)["delivered"] == 1
    ledger = listener._read_job_ledger(website_capture.root)
    record = ledger[terminal.OVERDUE_FIELD]
    assert record["reason"] == "recovery_needed" and record["basis"] == "no_claim_within_silence_window"
    assert record["known_cause"] == cause and record["retry_exhaustion_proven"] is False
    assert record["age_seconds"] > terminal.DEFAULT_SILENCE_SECONDS
    assert terminal.TERMINAL_FIELD not in ledger
    assert ledger["status"] == update["status"]  # The job status itself is unchanged.
    v2 = _read(website_capture, status_reader.STATUS_V2)
    expected = "capacity_delayed" if cause == "control_plane_capacity" else status_reader._LEDGER_STATES[update["status"]]
    assert v2["state"] == expected and v2["terminal_reason"] is None
    # A WebApp that predates v2 still reads the exact v1 meaning of the ledger.
    assert _read(website_capture)["state"] == status_reader._LEDGER_STATES[update["status"]]
    delivered = json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())
    assert delivered["state"] == "delivered" and delivered["acceptance"]["state"] == expected


def test_a_redelivery_inside_the_window_is_not_an_ending(website_capture):
    stopped = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    _stop(website_capture, at=stopped, status="failed_retryable", last_error=BUDGET)
    assert terminal.record_unretried_preparation(website_capture.root, alive_since=LONG_AGO,
                                                 now=stopped + WINDOW - timedelta(seconds=1)) is None
    _claim(website_capture)  # Pub/Sub delivered again after all (its count is best-effort).
    assert _end_silently(website_capture, stopped=stopped) is None  # A live lease is not silence.
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "preparing"


def test_a_listener_gap_restarts_the_silence_clock(website_capture):
    now = datetime.now(timezone.utc)
    _stop(website_capture, at=now - 3 * WINDOW, status="failed_retryable", last_error="x")
    # Last tick 30 minutes ago: an outage or one busy run, so nothing could be received.
    _listening_since(website_capture, now - 4 * WINDOW, last_tick=now - timedelta(minutes=30))
    alive = terminal.note_sweep_liveness(website_capture.storage_root, now=now)
    assert abs((alive - now).total_seconds()) < 1
    assert terminal.record_unretried_preparation(website_capture.root, alive_since=alive, now=now) is None
    later = now + WINDOW + timedelta(seconds=1)
    assert terminal.record_unretried_preparation(website_capture.root, alive_since=alive, now=later)
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "failed_retryable"


def test_liveness_is_continuous_only_across_short_gaps(tmp_path):
    start = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    assert terminal.note_sweep_liveness(tmp_path, now=start) == start
    assert terminal.note_sweep_liveness(tmp_path, now=start + timedelta(minutes=14)) == start
    assert terminal.note_sweep_liveness(tmp_path, now=start + timedelta(minutes=40)) == start + timedelta(minutes=40)
    (tmp_path / terminal.LIVENESS_FILE).write_text("{not json")
    assert terminal.note_sweep_liveness(tmp_path, now=start + timedelta(hours=1)) == start + timedelta(hours=1)


def test_the_assessment_wait_is_never_ended_by_silence(website_capture):
    stopped = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    pending = "website_control_scene-sponsorship_http_409:website_assessment_preparation_pending"
    _stop(website_capture, at=stopped, status="failed_retryable", last_error=pending,
          assessment_resume={"schema_version": "website_assessment_resume.v1"})
    assert _end_silently(website_capture, stopped=stopped) is None


def test_a_dead_worker_lease_ends_only_after_it_expires_and_the_window_passes(website_capture):
    started = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    ledger = _claim(website_capture, lease_seconds=60, now=started)
    expires = started + timedelta(seconds=60)
    assert terminal.record_unretried_preparation(website_capture.root, alive_since=LONG_AGO,
                                                 now=started + timedelta(seconds=30)) is None
    assert terminal.record_unretried_preparation(website_capture.root, alive_since=LONG_AGO,
                                                 now=expires + WINDOW - timedelta(seconds=1)) is None
    committed = terminal.record_unretried_preparation(website_capture.root, alive_since=LONG_AGO,
                                                      now=expires + WINDOW + timedelta(seconds=1))
    assert committed[terminal.OVERDUE_FIELD]["basis"] == "lease_expired_without_reclaim"
    assert committed[terminal.OVERDUE_FIELD]["lease_token"] == ledger["lease_token"]
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "failed_retryable"
    # An expired worker cannot replace the later observation of its attempt.
    before = listener._read_job_ledger(website_capture.root)
    with pytest.raises(listener.HandoffLeaseError) as lost:
        _finish(website_capture, ledger, status="failed_retryable", last_error="late failure")
    assert lost.value.state == "expired"
    assert listener._read_job_ledger(website_capture.root) == before


@pytest.mark.parametrize("update", [{"status": "completed"}, {"status": "terminal_authority_ended"}])
def test_final_states_are_never_ended_by_silence(website_capture, update):
    stopped = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    _stop(website_capture, at=stopped, **update)
    assert _end_silently(website_capture, stopped=stopped) is None


def test_overdue_record_is_superseded_by_next_claim_and_preserved_in_history(website_capture):
    stopped = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    _stop(website_capture, at=stopped, status="failed_retryable", last_error=BUDGET)
    assert _end_silently(website_capture, stopped=stopped)
    assert _end_silently(website_capture, stopped=stopped) is None
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "failed_retryable"
    redrive = _claim(website_capture)  # An operator re-drive starts a fresh attempt.
    assert terminal.TERMINAL_FIELD not in redrive
    assert terminal.OVERDUE_FIELD not in redrive
    assert any(item.get("status") == "preparation_overdue_observed" for item in redrive["attempt_history"])
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "preparing"


def test_unbound_or_absent_ledgers_record_nothing(website_capture):
    stopped = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    ledger = _stop(website_capture, at=stopped, status="failed_retryable", last_error="x")
    legacy = {key: value for key, value in ledger.items() if key != "source_payload_sha256"}
    write_json(website_capture.root / listener.JOB_LEDGER_FILENAME, legacy)
    assert _end_silently(website_capture, stopped=stopped) is None
    assert terminal.record_unretried_preparation(website_capture.root / "absent", alive_since=LONG_AGO,
                                                 now=stopped + 2 * WINDOW) is None


@pytest.mark.parametrize("mutation", ["observed_status", "attempt_count", "source_payload_sha256", "lease_token",
                                      "reason", "schema_version"])
def test_stale_or_forged_terminal_record_is_ignored(website_capture, mutation):
    stopped = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    _stop(website_capture, at=stopped, status="failed_retryable", last_error="x")
    committed = _end_silently(website_capture, stopped=stopped)
    record = dict(committed[terminal.OVERDUE_FIELD])
    record[mutation] = {"observed_status": "processing", "attempt_count": 99, "source_payload_sha256": "f" * 64,
                        "lease_token": "other", "reason": "provider_failed", "schema_version": "v0"}[mutation]
    write_json(website_capture.root / listener.JOB_LEDGER_FILENAME,
               {**committed, terminal.OVERDUE_FIELD: record, "revision": committed["revision"] + 1})
    assert terminal.current_terminal(committed) is None
    assert _read(website_capture, status_reader.STATUS_V2)["state"] == "failed_retryable"


@pytest.mark.parametrize("value, seconds", [(None, terminal.DEFAULT_SILENCE_SECONDS), ("abc", terminal.DEFAULT_SILENCE_SECONDS),
                                            ("60", 30 * 60), ("10800", 10800)])
def test_silence_window_is_configurable_but_never_shorter_than_redelivery(monkeypatch, value, seconds):
    if value is None:
        monkeypatch.delenv(terminal.SILENCE_SECONDS_ENV, raising=False)
    else:
        monkeypatch.setenv(terminal.SILENCE_SECONDS_ENV, value)
    assert terminal.silence_seconds() == seconds


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
    stopped = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    _stop(website_capture, at=stopped, status="failed_retryable", last_error=BUDGET)
    _end_silently(website_capture, stopped=stopped)
    status_reader.retain_preparation_wakeup(website_capture.root)
    monkeypatch.setattr(context_reader, "website_webapp_request", lambda **_: _acceptance(website_capture, version))
    assert status_reader.reconcile_preparation_wakeups(website_capture.storage_root)["delivered"] == 1
    accepted = json.loads((website_capture.root / status_reader.DELIVERY_FILE).read_text())["acceptance"]
    assert accepted["state"] == "failed_retryable"


@pytest.mark.parametrize("fault", ["v2_state_in_v1", "v1_digest_as_v2", "unknown_version"])
def test_mismatched_acceptance_version_cannot_consume_delivery(website_capture, monkeypatch, fault):
    stopped = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    _stop(website_capture, at=stopped, status="failed_retryable", last_error=BUDGET)
    _end_silently(website_capture, stopped=stopped)
    status_reader.retain_preparation_wakeup(website_capture.root)
    ack = _acceptance(website_capture, status_reader.STATUS_V2)
    if fault == "v2_state_in_v1":
        ack["schema_version"] = "website_preparation_status_acceptance.v1"
        ack["state"] = "failed_terminal"
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
