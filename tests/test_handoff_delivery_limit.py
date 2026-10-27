"""Beta capture preparation: broker redeliveries cannot exceed the execution cap."""

import json

import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from tests.test_pubsub_handoff_listener import (
    FakeSubscriber, PAYLOAD_BYTES, _install_fake_pubsub, _received,
)


def test_delivery_beyond_the_limit_retains_evidence_without_processing_or_ack(tmp_path, monkeypatch):
    subscriber = FakeSubscriber([_received(ack_id="over-limit", data=PAYLOAD_BYTES, delivery_attempt=6)])
    _install_fake_pubsub(monkeypatch, subscriber)
    monkeypatch.setattr(listener, "process_handoff_payload",
                        lambda *_a, **_k: pytest.fail("an exhausted delivery must not reach providers"))
    assert listener.pull_and_process(subscription="projects/p/subscriptions/s", storage_root=tmp_path,
                                     provider="openai", max_messages=1, max_delivery_attempts=5) == 0
    assert subscriber.acknowledged == []
    assert subscriber.ack_deadline_requests[-1]["ack_deadline_seconds"] == listener.RETRY_DEFER_SECONDS
    path, = (tmp_path / ".pubsub_delivery_evidence" / "retry_exhausted_pending_pubsub_dlq").glob("*.json")
    evidence = json.loads(path.read_text())
    assert evidence["delivery_attempt"] == 6
    assert evidence["payload_sha256"] == listener.payload_sha256(PAYLOAD_BYTES)
    assert evidence["blockers"] == ["handoff_delivery_attempt_limit_reached"]


def test_the_last_permitted_delivery_can_still_complete(tmp_path, monkeypatch):
    subscriber = FakeSubscriber([_received(ack_id="last", data=PAYLOAD_BYTES, delivery_attempt=5)])
    _install_fake_pubsub(monkeypatch, subscriber)
    calls = []
    monkeypatch.setattr(listener, "process_handoff_payload",
                        lambda *_a, **_k: calls.append(True) or {"status": "completed"})
    assert listener.pull_and_process(subscription="projects/p/subscriptions/s", storage_root=tmp_path,
                                     provider="openai", max_messages=1, max_delivery_attempts=5) == 1
    assert calls == [True]
    assert subscriber.acknowledged == ["last"]


@pytest.mark.parametrize("limit", [0, -1, True, "5"])
def test_invalid_limits_fail_before_any_work(tmp_path, limit):
    with pytest.raises(ValueError, match="pubsub_handoff_max_delivery_attempts_invalid"):
        listener.pull_and_process(subscription="projects/p/subscriptions/s", storage_root=tmp_path,
                                  provider="openai", max_messages=1, max_delivery_attempts=limit)
