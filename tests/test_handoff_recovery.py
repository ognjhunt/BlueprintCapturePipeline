"""Retained DLQ crash/replay tests with a fake broker and no provider calls."""

import base64
from datetime import datetime, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline import handoff_recovery as recovery
from blueprint_pipeline import pubsub_handoff_listener as listener

SOURCE = "projects/p/subscriptions/source"
DLQ = "projects/p/subscriptions/retained"
SAFE = {"status": "capture_owner_observation_unavailable_retryable", "queue_disposition": "retryable",
        "owner_observation_reason": "capture_owner_response_unavailable"}


def payload(suffix="one"):
    bucket, scene, capture = "capture-bucket", "site-" + suffix, "walkthrough-" + suffix
    marker = f"scenes/{scene}/captures/{capture}/raw/capture_upload_complete.json"
    delivery = hashlib.sha256(json.dumps([bucket, marker, "1"], separators=(",", ":")).encode()).hexdigest()
    return json.dumps({"bucket": bucket, "scene_id": scene, "capture_id": capture,
        "raw_prefix_uri": f"gs://{bucket}/scenes/{scene}/captures/{capture}/raw",
        "source_finalize": {"bucket": bucket, "object_name": marker, "generation": "1",
                            "event_id": "finalize-one", "event_source": "storage"},
        "source_membership_selector": {"object_name": f"scenes/{scene}/captures/{capture}/deliveries/{delivery}/capture_delivery_membership.json",
            "generation": "2", "size_bytes": 100, "sha256": "sha256:" + "a" * 64}}).encode()


def received(raw=None, *, source=SOURCE, message_id="dlq-one", attempt=1):
    return SimpleNamespace(ack_id="private-ack", delivery_attempt=attempt, message=SimpleNamespace(
        data=raw if raw is not None else payload(), message_id=message_id,
        publish_time=datetime.fromtimestamp(1000, timezone.utc),
        attributes={recovery.SOURCE_ATTRIBUTE: source, "CloudPubSubDeadLetterSourceDeliveryCount": "5"}))


class Broker:
    def __init__(self, messages, *, ack_error=False):
        self.messages, self.acks, self.ack_error = list(messages), [], ack_error

    def pull(self, request, **kwargs):
        return SimpleNamespace(received_messages=[self.messages.pop(0)] if self.messages else [])

    def acknowledge(self, request):
        if self.ack_error:
            raise OSError("ack outcome uncertain")
        self.acks.append(request)

    def modify_ack_deadline(self, request):
        pass


def source_failed(storage, raw=None):
    raw = raw if raw is not None else payload()
    for number in range(1, 6):
        receipt = received(raw, message_id="source-one", attempt=number)
        recovery.record_source_result(listener, storage_root=storage, subscription=SOURCE,
            message=receipt.message, received=receipt, result=SAFE)
    return raw


def complete_handler(calls):
    def process(raw, *, storage_root, payload_digest, require_unattempted_delivery, **kwargs):
        assert require_unattempted_delivery is True
        handoff = listener.parse_handoff_payload(raw)
        calls.append({"payload_sha256": payload_digest, "current_membership_rights_checked": True})
        capture = listener._handoff_capture_root(handoff, storage_root=storage_root)
        claim, ledger = listener._claim_job_lease(capture, scene_id=handoff.scene_id, capture_id=handoff.capture_id,
            owner="recovery-test", lease_seconds=60, payload_sha256=payload_digest, require_unattempted_delivery=True)
        assert claim == "claimed"
        listener._write_output_commit(capture, scene_id=handoff.scene_id, capture_id=handoff.capture_id,
            attempt_count=ledger["attempt_count"], result={"status": "completed"}, owner="recovery-test",
            token=ledger["lease_token"], ledger_update={"status": "completed", "source_payload_sha256": payload_digest})
        return {"status": "completed", "queue_disposition": "terminal_success"}
    return process


def tick(storage, broker, *, enabled=False, now=1001, generation="a"):
    return recovery.pull_retained(listener, subscriber=broker, storage_root=storage, subscription=DLQ,
        source_subscription=SOURCE, process_args={"provider": "local"}, enable_safe=enabled,
        max_messages=1, now=now, dependency_generation=generation)


def entry(storage, raw=None):
    return storage / recovery.DIRECTORY / hashlib.sha256(raw if raw is not None else payload()).hexdigest()


def test_default_observe_retains_private_exact_bytes_but_neither_acks_nor_executes(tmp_path, monkeypatch):
    raw = source_failed(tmp_path)
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *a, **k: pytest.fail("observe mode executed"))
    broker = Broker([received(raw)])
    result = tick(tmp_path, broker)
    retained = recovery._read(entry(tmp_path) / "event.json")
    assert base64.b64decode(retained["payload_base64"]) == raw
    assert broker.acks == [] and result["safe_replay_enabled"] is False
    assert recovery._read(entry(tmp_path) / "state.json")["status"] == "safe_pre_effect"
    assert (entry(tmp_path).stat().st_mode & 0o077) == 0
    assert not listener._handoff_capture_root(listener.parse_handoff_payload(raw), storage_root=tmp_path).exists()
    assert "payload_base64" not in json.dumps(result) and "private-ack" not in json.dumps(result)


def test_five_source_failures_then_healthy_retained_job_executes_once_with_original_lease(tmp_path, monkeypatch):
    raw = source_failed(tmp_path)
    calls = []
    monkeypatch.setattr(listener, "process_handoff_payload", complete_handler(calls))
    broker = Broker([received(raw)])
    result = tick(tmp_path, broker, enabled=True)
    assert result["acknowledged"] == 1 and len(broker.acks) == len(calls) == 1
    assert recovery._read(entry(tmp_path) / "state.json")["status"] == "completed"
    assert calls[0]["payload_sha256"] == hashlib.sha256(raw).hexdigest()
    # Redelivery after ack/commit and a duplicate timer can only adopt completion.
    assert tick(tmp_path, Broker([received(raw)]), enabled=True, now=10000)["acknowledged"] == 1
    recovery.process_retained(listener, storage_root=tmp_path, source_subscription=SOURCE,
        process_args={}, enable_safe=True, now=20000)
    assert len(calls) == 1
    capture = listener._handoff_capture_root(listener.parse_handoff_payload(raw), storage_root=tmp_path)
    ledger = listener._read_job_ledger(capture)
    proof = recovery.read_disposition(listener, storage_root=tmp_path,
        source_payload_sha256=hashlib.sha256(raw).hexdigest(), expected_ledger=ledger)
    assert proof["status"] == "completed" and proof["retry_exhaustion_proven"] is False


def test_real_source_drain_records_the_named_failure_across_all_five_deliveries(tmp_path, monkeypatch):
    import google.cloud
    broker = Broker([received(message_id="source-one", attempt=number) for number in range(1, 6)])
    monkeypatch.setattr(google.cloud, "pubsub_v1", SimpleNamespace(SubscriberClient=lambda: broker), raising=False)
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *a, **k: SAFE)
    assert listener.pull_and_process(subscription=SOURCE, storage_root=tmp_path, provider="local", max_messages=5) == 0
    root = recovery._root(tmp_path)
    retained = recovery._read(root / "source_observations" / (hashlib.sha256(payload()).hexdigest() + ".json"))
    assert retained["source_delivery_attempt"] == 5 and retained["effect_state"] == "pre_lease"
    assert broker.acks == [] and retained["source_subscription"] == SOURCE
    assert not listener._handoff_capture_root(listener.parse_handoff_payload(payload()), storage_root=tmp_path).exists()


def test_ack_response_loss_retains_successor_and_does_not_claim_ack(tmp_path, monkeypatch):
    raw = source_failed(tmp_path)
    calls = []
    monkeypatch.setattr(listener, "process_handoff_payload", complete_handler(calls))
    with pytest.raises(OSError, match="uncertain"):
        tick(tmp_path, Broker([received(raw)], ack_error=True), enabled=True)
    assert (entry(tmp_path) / "event.json").is_file()
    assert not list((entry(tmp_path) / "acknowledged").glob("*.json")) and calls == []
    result = tick(tmp_path, Broker([received(raw)]), enabled=True)
    assert result["acknowledged"] == 1 and len(calls) == 1


def test_persistence_failure_before_successor_prevents_ack(tmp_path, monkeypatch):
    real = recovery._once
    def crash(path, value):
        if path.name == "event.json":
            raise OSError("fsync unavailable")
        return real(path, value)
    monkeypatch.setattr(recovery, "_once", crash)
    broker = Broker([received()])
    with pytest.raises(OSError, match="fsync unavailable"):
        tick(tmp_path, broker, enabled=True)
    assert broker.acks == []


def test_unknown_provider_effect_never_reexecutes_or_resets_attempt(tmp_path, monkeypatch):
    raw = source_failed(tmp_path)
    calls = []
    def ambiguous(raw, *, storage_root, payload_digest, require_unattempted_delivery, **kwargs):
        handoff = listener.parse_handoff_payload(raw)
        capture = listener._handoff_capture_root(handoff, storage_root=storage_root)
        listener._claim_job_lease(capture, scene_id=handoff.scene_id, capture_id=handoff.capture_id,
            owner="original", lease_seconds=60, payload_sha256=payload_digest, require_unattempted_delivery=True)
        calls.append("original provider request has unknown outcome")
        raise OSError("provider response lost")
    monkeypatch.setattr(listener, "process_handoff_payload", ambiguous)
    tick(tmp_path, Broker([received(raw)]), enabled=True)
    before = recovery._read(entry(tmp_path) / "state.json")
    assert before["status"] == "reconciliation_required"
    recovery.process_retained(listener, storage_root=tmp_path, source_subscription=SOURCE,
        process_args={}, enable_safe=True, now=100000)
    assert len(calls) == 1
    assert recovery._read(entry(tmp_path) / "state.json")["attempts"] == 1


def test_unknown_generic_failure_without_ledger_is_absorbing(tmp_path, monkeypatch):
    source_failed(tmp_path)
    calls = []
    def unproven(*args, **kwargs):
        calls.append("one")
        return {"status": "unclassified", "queue_disposition": "retryable"}
    monkeypatch.setattr(listener, "process_handoff_payload", unproven)
    tick(tmp_path, Broker([received()]), enabled=True)
    recovery.process_retained(listener, storage_root=tmp_path, source_subscription=SOURCE,
        process_args={}, enable_safe=True, now=100000)
    assert calls == ["one"]


def test_canonical_ended_authority_and_explicit_wait_are_not_replayed(tmp_path, monkeypatch):
    raw = source_failed(tmp_path)
    handoff = listener.parse_handoff_payload(raw)
    capture = listener._handoff_capture_root(handoff, storage_root=tmp_path)
    _, ledger = listener._claim_job_lease(capture, scene_id=handoff.scene_id, capture_id=handoff.capture_id,
        owner="test", lease_seconds=60, payload_sha256=hashlib.sha256(raw).hexdigest())
    listener._finish_job_lease(capture, owner="test", token=ledger["lease_token"], update={
        "status": listener.TERMINAL_AUTHORITY_STATUS, "terminal_payload_sha256": hashlib.sha256(raw).hexdigest(),
        "terminal_code": "consent_expired"})
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *a, **k: pytest.fail("ended grant replayed"))
    tick(tmp_path, Broker([received()]), enabled=True)
    assert recovery._read(entry(tmp_path) / "state.json")["status"] == "authority_ended"
    assert listener._read_job_ledger(capture)["attempt_count"] == 1


def test_safe_pre_effect_outage_backoff_and_one_health_generation_wake(tmp_path, monkeypatch):
    source_failed(tmp_path)
    calls = []
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *a, **k: calls.append("read") or SAFE)
    tick(tmp_path, Broker([received()]), enabled=True)
    after = recovery._read(entry(tmp_path) / "state.json")
    assert 1061 <= after["next_attempt_at_epoch"] <= 1090 and len(calls) == 1
    recovery.process_retained(listener, storage_root=tmp_path, source_subscription=SOURCE,
        process_args={}, enable_safe=True, now=1002, dependency_generation="a")
    assert len(calls) == 1
    recovery.process_retained(listener, storage_root=tmp_path, source_subscription=SOURCE,
        process_args={}, enable_safe=True, now=1002, dependency_generation="b")
    assert len(calls) == 2
    recovery.process_retained(listener, storage_root=tmp_path, source_subscription=SOURCE,
        process_args={}, enable_safe=True, now=1003, dependency_generation="b")
    assert len(calls) == 2


def test_unproven_broker_source_stays_unacked_and_orphan_alerts_without_payload(tmp_path, monkeypatch):
    source_failed(tmp_path)
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *a, **k: pytest.fail("orphan replayed"))
    broker = Broker([received(source="projects/other/subscriptions/wrong")])
    observed = tick(tmp_path, broker, enabled=True)
    assert broker.acks == [] and "handoff_recovery_orphaned_dlq" in observed["alerts"]
    assert observed["raw_payload_exposed"] is False


def test_retention_alert_and_actual_disposition_bytes_cannot_be_forged(tmp_path, monkeypatch):
    source_failed(tmp_path)
    tick(tmp_path, Broker([received()]))
    observed = recovery.status(storage_root=tmp_path, now=1000 + recovery.RETENTION_SECONDS - 60)
    assert "handoff_recovery_retention_deadline_due" in observed["alerts"]
    proof = recovery.read_disposition(listener, storage_root=tmp_path, source_payload_sha256=entry(tmp_path).name)
    path = entry(tmp_path) / "dispositions" / (proof["disposition_digest"][7:] + ".json")
    path.write_text(json.dumps({**proof, "retry_exhaustion_proven": True}))
    with pytest.raises(ValueError, match="disposition_binding_invalid"):
        recovery.read_disposition(listener, storage_root=tmp_path, source_payload_sha256=entry(tmp_path).name)


def test_invalid_payload_quarantine_commits_before_ack_and_never_creates_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *a, **k: pytest.fail("invalid raw executed"))
    broker = Broker([received(b"invalid wire payload")])
    result = tick(tmp_path, broker, enabled=True)
    assert result["acknowledged"] == 1
    assert recovery._read(entry(tmp_path, b"invalid wire payload") / "state.json")["status"] == "permanent_invalid"
    assert not list(tmp_path.glob("*/scenes"))


def test_explicit_owner_access_denial_waits_for_authorized_action_without_reread(tmp_path, monkeypatch):
    raw = payload()
    receipt = received(raw)
    recovery.record_source_result(listener, storage_root=tmp_path, subscription=SOURCE,
        message=receipt.message, received=receipt, result={**SAFE, "owner_observation_reason": "capture_owner_response_denied"})
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *a, **k: pytest.fail("denied owner read retried"))
    tick(tmp_path, Broker([receipt]), enabled=True)
    state = recovery._read(entry(tmp_path) / "state.json")
    assert state["status"] == "waiting_authorized_action"
    assert state["next_action"] == "restore_authorized_capture_owner_read"
    recovery.process_retained(listener, storage_root=tmp_path, source_subscription=SOURCE,
        process_args={}, enable_safe=True, now=100000, dependency_generation="newrelease")
    assert recovery._read(entry(tmp_path) / "state.json")["attempts"] == 0


def test_recovery_status_uses_existing_listener_cli_without_capture_selectors(tmp_path, capsys):
    assert listener.main(["--storage-root", str(tmp_path), "--recovery-status"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["schema_version"] == recovery.STATUS_SCHEMA and value["retained"] == 0


@pytest.mark.parametrize("failure,reason,expected,reads", [
    ("denied", "capture_staging_access_denied", "waiting_authorized_action", 1),
    ("metadata", "capture_staging_source_invalid", "permanent_invalid", 1),
    ("transient", "capture_staging_transport_unavailable", "safe_pre_effect", 3),
])
def test_canonical_wrapped_membership_failures_preserve_denial_and_invalidity(
        tmp_path, monkeypatch, failure, reason, expected, reads):
    from google.api_core.exceptions import Forbidden, ServiceUnavailable
    from blueprint_pipeline import capture_original_owner_observer as observer
    from tests.test_capture_generation_birth import _fixture
    _, _, target, owner, selector, membership = _fixture(tmp_path, monkeypatch, prepare_parent=False)
    raw = json.dumps({"bucket": owner["bucket"], "scene_id": owner["scene_id"],
        "capture_id": owner["capture_id"], "raw_prefix_uri": owner["raw_prefix_uri"],
        "source_finalize": {**json.loads(membership)["source_finalize"], "event_id": "evt-one", "event_source": "storage"},
        "source_membership_selector": selector}).encode()
    monkeypatch.setattr(observer, "load_original_owner_observation", lambda **_: owner)
    observed = []
    class Blob:
        generation = selector["generation"]
        size = selector["size_bytes"] + 1
        def reload(self, **_kwargs):
            observed.append("read")
            if failure == "denied":
                raise Forbidden("private provider body must stay hidden")
            if failure == "transient":
                raise ServiceUnavailable("private provider body must stay hidden")
    class Bucket:
        def blob(self, *_args, **_kwargs):
            return Blob()
    class Client:
        def bucket(self, *_args):
            return Bucket()
    storage = target.parents[4]
    args = {"provider": "local", "storage_client": Client(), "run_e2e_enabled": False, "run_evaluation_prep": False}
    result = listener.process_handoff_payload(raw, storage_root=storage, **args)
    assert result["staging_reason"] == reason
    receipt = received(raw)
    recovery.record_source_result(listener, storage_root=storage, subscription=SOURCE,
        message=receipt.message, received=receipt, result=result)
    recovery.pull_retained(listener, subscriber=Broker([receipt]), storage_root=storage, subscription=DLQ,
        source_subscription=SOURCE, process_args=args, enable_safe=True, max_messages=1, now=1001)
    recovery.process_retained(listener, storage_root=storage, source_subscription=SOURCE,
        process_args=args, enable_safe=True, now=100000, dependency_generation="new-release")
    retained = recovery._read(entry(storage, raw) / "state.json")
    assert retained["status"] == expected and len(observed) == reads
    assert "private" not in json.dumps(retained) and not target.exists()
    if failure != "transient":
        # A later unavailable duplicate and ordinary source redelivery do not
        # constitute a correction to the denied/invalid exact source authority.
        recovery.record_source_result(listener, storage_root=storage, subscription=SOURCE,
            message=receipt.message, received=receipt, result=SAFE)
        replay = listener.process_handoff_payload(raw, storage_root=storage, **args)
        assert replay["staging_reason"] == reason and len(observed) == 1
        changed_delivery = json.loads(raw)
        changed_delivery["source_finalize"]["event_id"] = "duplicate-event-with-same-authority"
        reformatted = json.dumps(changed_delivery, indent=2).encode()
        replay = listener.process_handoff_payload(reformatted, storage_root=storage, **args)
        assert replay["staging_reason"] == reason and len(observed) == 1


def test_denial_is_absorbing_across_later_outage_observations(tmp_path, monkeypatch):
    receipt = received()
    recovery.record_source_result(listener, storage_root=tmp_path, subscription=SOURCE,
        message=receipt.message, received=receipt, result={**SAFE, "owner_observation_reason": "capture_owner_response_denied"})
    path = recovery._root(tmp_path) / "source_observations" / (hashlib.sha256(payload()).hexdigest() + ".json")
    denied_bytes = path.read_bytes()
    for _ in range(3):
        recovery.record_source_result(listener, storage_root=tmp_path, subscription=SOURCE,
            message=receipt.message, received=receipt, result=SAFE)
    assert path.read_bytes() == denied_bytes
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *a, **k: pytest.fail("denied source retried"))
    tick(tmp_path, Broker([receipt]), enabled=True)
    recovery.process_retained(listener, storage_root=tmp_path, source_subscription=SOURCE,
        process_args={}, enable_safe=True, now=100000, dependency_generation="another-release")
    assert recovery._read(entry(tmp_path) / "state.json")["attempts"] == 0


@pytest.mark.parametrize("fault", ["initial_state_crash", "corrupt_state"])
def test_partial_journal_does_not_starve_healthy_successor_or_infer_ack(tmp_path, monkeypatch, fault):
    from blueprint_pipeline import beta_policy_screening_launch as launch
    good, bad = source_failed(tmp_path, payload("good")), source_failed(tmp_path, payload("bad"))
    recovery.retain_dlq_message(listener, storage_root=tmp_path, subscription=DLQ,
        source_subscription=SOURCE, received=received(good), now=1001)
    write = launch._write
    def crash(path, value):
        if path == entry(tmp_path, bad) / "state.json":
            raise OSError("power loss before initial successor state")
        return write(path, value)
    monkeypatch.setattr(launch, "_write", crash)
    with pytest.raises(OSError):
        recovery.retain_dlq_message(listener, storage_root=tmp_path, subscription=DLQ,
            source_subscription=SOURCE, received=received(bad), now=1001)
    monkeypatch.setattr(launch, "_write", write)
    if fault == "corrupt_state":
        (entry(tmp_path, bad) / "state.json").write_text("{interrupted")
    evidence = (entry(tmp_path, bad) / "event.json").read_bytes()
    calls = []
    monkeypatch.setattr(listener, "process_handoff_payload", complete_handler(calls))
    observed = recovery.process_retained(listener, storage_root=tmp_path, source_subscription=SOURCE,
        process_args={}, enable_safe=True, limit=1, now=1002)
    assert len(calls) == 1 and recovery._read(entry(tmp_path, good) / "state.json")["status"] == "completed"
    assert observed["incomplete_entries"] == 1 and "handoff_recovery_partial_entry" in observed["alerts"]
    assert (entry(tmp_path, bad) / "event.json").read_bytes() == evidence
    assert not list((entry(tmp_path, bad) / "acknowledged").glob("*.json"))
    if fault == "initial_state_crash":
        assert not (entry(tmp_path, bad) / "state.json").exists()
        tick(tmp_path, Broker([received(bad)]), enabled=True, now=1003)
        assert len(calls) == 2 and recovery._read(entry(tmp_path, bad) / "state.json")["status"] == "completed"


def test_failure_retaining_replay_outcome_cannot_reuse_old_safe_observation(tmp_path, monkeypatch):
    source_failed(tmp_path)
    calls = []
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *a, **k: calls.append("one") or SAFE)
    def crash(*_args, **_kwargs):
        raise OSError("lost outcome persistence")
    monkeypatch.setattr(recovery, "_record_result", crash)
    tick(tmp_path, Broker([received()]), enabled=True)
    recovery.process_retained(listener, storage_root=tmp_path, source_subscription=SOURCE,
        process_args={}, enable_safe=True, now=100000)
    assert calls == ["one"]
    assert recovery._read(entry(tmp_path) / "state.json")["status"] == "reconciliation_required"
