"""A deploy wakes Pub/Sub handoffs that an older release deferred.

A retryable handoff is deferred by holding its Pub/Sub lease for
``RETRY_DEFER_SECONDS``; only Pub/Sub knows it is waiting. After a deploy that
fixed the cause, the first listener tick under the new release releases those
leases so Pub/Sub redelivers them now (still through its retry policy and its
dead-letter cap) instead of after the remaining lease.
"""

from __future__ import annotations

import json
import stat
import types
from pathlib import Path

import google.cloud
import pytest

import blueprint_pipeline.pubsub_handoff_listener as listener_module
import blueprint_pipeline.pubsub_retry_deferrals as deferrals
from blueprint_pipeline.pubsub_handoff_listener import pull_and_process

OLD_RELEASE = "1" * 40
NEW_RELEASE = "2" * 40
SUBSCRIPTION = "projects/p/subscriptions/s"
PAYLOAD = json.dumps(
    {
        "bucket": "capture-bucket",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
    }
).encode()


class FakeSubscriber:
    def __init__(self, received_messages: list[object]) -> None:
        self.received_messages = list(received_messages)
        self.acknowledged: list[str] = []
        self.ack_deadline_requests: list[dict] = []
        self.events: list[str] = []

    def pull(self, *, request: dict, timeout: int) -> object:
        self.events.append("pull")
        batch = self.received_messages[: request["max_messages"]]
        del self.received_messages[: len(batch)]
        return types.SimpleNamespace(received_messages=batch)

    def acknowledge(self, *, request: dict) -> None:
        self.acknowledged.extend(request["ack_ids"])

    def modify_ack_deadline(self, *, request: dict) -> None:
        self.events.append(f"modify:{request['ack_deadline_seconds']}")
        self.ack_deadline_requests.append(request)


def _received(ack_id: str, *, attempt: int = 1, message_id: str = "msg-1") -> object:
    return types.SimpleNamespace(
        ack_id=ack_id,
        delivery_attempt=attempt,
        message=types.SimpleNamespace(message_id=message_id, data=PAYLOAD, attributes={}),
    )


def _tick(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    release: str,
    received: list[object],
    outcome: dict,
) -> FakeSubscriber:
    subscriber = FakeSubscriber(received)
    pubsub_v1 = types.SimpleNamespace(SubscriberClient=lambda: subscriber)
    monkeypatch.setattr(google.cloud, "pubsub_v1", pubsub_v1, raising=False)
    monkeypatch.setattr(deferrals, "running_release_commit", lambda: release)
    monkeypatch.setattr(
        listener_module, "process_handoff_payload", lambda *_args, **_kwargs: dict(outcome)
    )
    pull_and_process(
        subscription=SUBSCRIPTION, storage_root=tmp_path, provider="openai", max_messages=1
    )
    return subscriber


RETRYABLE = {"status": "retryable_blocked", "queue_disposition": "retryable", "blockers": ["fix_me"]}


def _records(tmp_path: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((tmp_path / deferrals.DEFERRALS_DIRNAME).glob("*.json"))
    ]


def test_a_retryable_deferral_is_recorded_privately_with_its_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _tick(monkeypatch, tmp_path, release=OLD_RELEASE, received=[_received("ack-1")], outcome=RETRYABLE)

    [record] = _records(tmp_path)
    assert record["schema_version"] == deferrals.SCHEMA_VERSION
    assert record["subscription"] == SUBSCRIPTION
    assert record["ack_id"] == "ack-1"
    assert record["message_id"] == "msg-1"
    assert record["delivery_attempt"] == 1
    assert record["release_commit"] == OLD_RELEASE
    assert record["lease_expires_at_epoch"] - record["deferred_at_epoch"] == (
        listener_module.RETRY_DEFER_SECONDS
    )
    [path] = (tmp_path / deferrals.DEFERRALS_DIRNAME).glob("*.json")
    # An ack ID is a lease credential for this subscription.
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_first_tick_under_a_new_release_releases_older_deferrals_before_pulling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _tick(monkeypatch, tmp_path, release=OLD_RELEASE, received=[_received("ack-1")], outcome=RETRYABLE)

    subscriber = _tick(monkeypatch, tmp_path, release=NEW_RELEASE, received=[], outcome=RETRYABLE)

    assert subscriber.ack_deadline_requests == [
        {"subscription": SUBSCRIPTION, "ack_ids": ["ack-1"], "ack_deadline_seconds": 0}
    ]
    assert subscriber.events[0] == "modify:0"
    assert _records(tmp_path) == []

    # Each deferral is released once; the next tick has nothing left to wake.
    subscriber = _tick(monkeypatch, tmp_path, release=NEW_RELEASE, received=[], outcome=RETRYABLE)
    assert subscriber.ack_deadline_requests == []


def test_same_release_deferrals_keep_their_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _tick(monkeypatch, tmp_path, release=OLD_RELEASE, received=[_received("ack-1")], outcome=RETRYABLE)

    subscriber = _tick(monkeypatch, tmp_path, release=OLD_RELEASE, received=[], outcome=RETRYABLE)

    assert subscriber.ack_deadline_requests == []
    assert len(_records(tmp_path)) == 1


def test_a_last_attempt_or_expired_lease_is_left_to_pubsub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    last = listener_module.DEFAULT_MAX_DELIVERY_ATTEMPTS
    _tick(
        monkeypatch,
        tmp_path,
        release=OLD_RELEASE,
        received=[_received("ack-last", attempt=last, message_id="msg-last")],
        outcome=RETRYABLE,
    )
    _tick(
        monkeypatch,
        tmp_path,
        release=OLD_RELEASE,
        received=[_received("ack-expired", message_id="msg-expired")],
        outcome=RETRYABLE,
    )
    for path in (tmp_path / deferrals.DEFERRALS_DIRNAME).glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["ack_id"] == "ack-expired":
            record["lease_expires_at_epoch"] = 1.0
            path.write_text(json.dumps(record), encoding="utf-8")

    subscriber = _tick(monkeypatch, tmp_path, release=NEW_RELEASE, received=[], outcome=RETRYABLE)

    # Releasing a last attempt would only dead-letter it sooner; an expired
    # lease is already back with Pub/Sub. Neither is touched.
    assert subscriber.ack_deadline_requests == []
    assert [record["ack_id"] for record in _records(tmp_path)] == ["ack-last"]


def test_acknowledged_handoff_clears_its_deferral(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _tick(monkeypatch, tmp_path, release=OLD_RELEASE, received=[_received("ack-1")], outcome=RETRYABLE)
    monkeypatch.setattr(listener_module, "_record_acknowledgement", lambda *_args, **_kwargs: None)

    subscriber = _tick(
        monkeypatch,
        tmp_path,
        release=OLD_RELEASE,
        received=[_received("ack-2")],
        outcome={"status": "processed", "queue_disposition": "terminal_success"},
    )

    assert subscriber.acknowledged == ["ack-2"]
    assert _records(tmp_path) == []


def test_a_failed_release_leaves_the_message_with_pubsub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _tick(monkeypatch, tmp_path, release=OLD_RELEASE, received=[_received("ack-1")], outcome=RETRYABLE)

    class RefusingSubscriber(FakeSubscriber):
        def modify_ack_deadline(self, *, request: dict) -> None:
            super().modify_ack_deadline(request=request)
            raise RuntimeError("ack id expired")

    subscriber = RefusingSubscriber([])
    monkeypatch.setattr(
        google.cloud, "pubsub_v1", types.SimpleNamespace(SubscriberClient=lambda: subscriber), raising=False
    )
    monkeypatch.setattr(deferrals, "running_release_commit", lambda: NEW_RELEASE)

    pull_and_process(subscription=SUBSCRIPTION, storage_root=tmp_path, provider="openai", max_messages=1)

    # The unacked message still retries on its own lease; the record is spent.
    assert len(subscriber.ack_deadline_requests) == 1
    assert _records(tmp_path) == []


def test_unknown_running_release_wakes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _tick(monkeypatch, tmp_path, release=OLD_RELEASE, received=[_received("ack-1")], outcome=RETRYABLE)

    subscriber = _tick(monkeypatch, tmp_path, release="", received=[], outcome=RETRYABLE)

    assert subscriber.ack_deadline_requests == []
    assert len(_records(tmp_path)) == 1


def test_a_malformed_deferral_entry_never_stops_the_tick(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _tick(monkeypatch, tmp_path, release=OLD_RELEASE, received=[_received("ack-1")], outcome=RETRYABLE)
    (tmp_path / deferrals.DEFERRALS_DIRNAME / "not-a-record.json").mkdir()

    subscriber = _tick(monkeypatch, tmp_path, release=NEW_RELEASE, received=[], outcome=RETRYABLE)

    # The valid deferral is still released and the tick still pulls.
    assert subscriber.ack_deadline_requests == [
        {"subscription": SUBSCRIPTION, "ack_ids": ["ack-1"], "ack_deadline_seconds": 0}
    ]
    assert "pull" in subscriber.events
